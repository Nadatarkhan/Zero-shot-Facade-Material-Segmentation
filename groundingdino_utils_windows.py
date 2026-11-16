"""Grounding DINO + SAM detection of facade objects (windows and doors).

This module provides ``GROUNDING_DINO_OUTPUT``, which ``run_zero_shot_inference.py``
imports. It is the "GroundingDino + SAM, prompts: ['window', 'door']" branch of the
workflow in Fig. 1 of the paper:

  1. Grounding DINO detects bounding boxes for each text prompt.
  2. SAM turns every box into a pixel-accurate mask (box-prompted segmentation).
  3. The masks are painted into an RGB "panoptic" mask, using the colours that
     ``label_df_DINO`` in ``run_zero_shot_inference.py`` expects:
         window -> (0, 0, 255)      -> class 10 (glazing)
         door   -> (210, 105, 30)   -> class 32 (door)
     Pixels with no detection are (0, 0, 0).

The returned array is (H, W, 3) uint8, the same size as the input image.

Configuration (environment variables, all optional):
  GDINO_CONFIG      path to the Grounding DINO config
                    (default: GroundingDINO_SwinT_OGC.py shipped with the package)
  GDINO_CHECKPOINT  path to the Grounding DINO weights
                    (default: ckpt/groundingdino_swint_ogc.pth)
  GDINO_DEVICE      device for Grounding DINO (default: "cpu", as in the paper)
  SAM_CHECKPOINT    path to the SAM ViT-H weights (default: sam_vit_h_4b8939.pth)
  SAM_DEVICE        device for SAM (default: "cuda" if available, else "cpu")
  GDINO_BOX_THRESHOLD, GDINO_TEXT_THRESHOLD
                    detection thresholds (default: 0.35 / 0.25, the Grounding DINO
                    reference defaults)

Models are loaded lazily on the first call and then cached. Call
``set_sam_model(sam)`` to share a SAM model that is already in memory.
"""

import os

import numpy as np
import torch
from PIL import Image

# Prompt -> RGB colour. Keep these colours in sync with ``label_df_DINO`` in
# run_zero_shot_inference.py, which maps them to class numbers.
DINO_CLASSES = {
    "window": (0, 0, 255),
    "door": (210, 105, 30),
}

BOX_THRESHOLD = float(os.environ.get("GDINO_BOX_THRESHOLD", 0.35))
TEXT_THRESHOLD = float(os.environ.get("GDINO_TEXT_THRESHOLD", 0.25))

_GDINO_MODEL = None
_SAM_PREDICTOR = None


def _default_gdino_config():
    import groundingdino

    return os.path.join(
        os.path.dirname(groundingdino.__file__), "config", "GroundingDINO_SwinT_OGC.py"
    )


def _gdino_device():
    return os.environ.get("GDINO_DEVICE", "cpu")


def _sam_device():
    return os.environ.get("SAM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")


def _load_gdino():
    global _GDINO_MODEL
    if _GDINO_MODEL is None:
        from groundingdino.util.inference import load_model

        config = os.environ.get("GDINO_CONFIG", _default_gdino_config())
        checkpoint = os.environ.get("GDINO_CHECKPOINT", "ckpt/groundingdino_swint_ogc.pth")
        _GDINO_MODEL = load_model(config, checkpoint, device=_gdino_device())
    return _GDINO_MODEL


def set_sam_model(sam):
    """Reuse an already-loaded SAM model instead of loading a second copy."""
    global _SAM_PREDICTOR
    from segment_anything import SamPredictor

    _SAM_PREDICTOR = SamPredictor(sam)


def _load_sam():
    global _SAM_PREDICTOR
    if _SAM_PREDICTOR is None:
        from segment_anything import SamPredictor, sam_model_registry

        checkpoint = os.environ.get("SAM_CHECKPOINT", "sam_vit_h_4b8939.pth")
        sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
        sam.to(device=_sam_device())
        _SAM_PREDICTOR = SamPredictor(sam)
    return _SAM_PREDICTOR


def _gdino_transform(image_pil):
    """Same preprocessing as groundingdino.util.inference.load_image, for a PIL image."""
    import groundingdino.datasets.transforms as T

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_tensor, _ = transform(image_pil, None)
    return image_tensor


def detect_boxes(image_pil, prompt):
    """Run Grounding DINO for one text prompt.

    Returns an (N, 4) float array of boxes in absolute xyxy pixel coordinates.
    """
    from groundingdino.util.inference import predict
    from torchvision.ops import box_convert

    model = _load_gdino()
    boxes, _logits, _phrases = predict(
        model=model,
        image=_gdino_transform(image_pil),
        caption=prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=_gdino_device(),
    )
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    w, h = image_pil.size
    boxes = boxes * torch.tensor([w, h, w, h])
    return box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()


def paint_masks(shape_hw, masks_by_class):
    """Paint boolean masks into an RGB array.

    ``masks_by_class`` is a list of (rgb_colour, bool_mask) pairs. Larger masks are
    painted first, so that smaller objects (e.g. a door inside a glazed storefront
    box) stay visible on top.
    """
    out = np.zeros((shape_hw[0], shape_hw[1], 3), dtype=np.uint8)
    for colour, mask in sorted(masks_by_class, key=lambda cm: -int(cm[1].sum())):
        out[mask] = colour
    return out


def GROUNDING_DINO_OUTPUT(image, classes=None):
    """Detect windows and doors and return a colour-coded (H, W, 3) uint8 mask.

    ``image`` is a PIL image (or an RGB numpy array). ``classes`` optionally
    overrides ``DINO_CLASSES`` (a dict of prompt -> RGB colour).
    """
    classes = DINO_CLASSES if classes is None else classes
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image).astype(np.uint8))
    image = image.convert("RGB")
    image_np = np.asarray(image)

    predictor = _load_sam()
    predictor.set_image(image_np)

    masks_by_class = []
    for prompt, colour in classes.items():
        for box in detect_boxes(image, prompt):
            masks, _scores, _ = predictor.predict(box=box, multimask_output=False)
            masks_by_class.append((colour, masks[0].astype(bool)))

    return paint_masks(image_np.shape[:2], masks_by_class)
