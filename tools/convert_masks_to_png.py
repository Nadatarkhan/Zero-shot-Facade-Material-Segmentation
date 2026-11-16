"""Convert the ground-truth masks from JPEG to lossless PNG.

The masks were originally saved as JPEG, which blurs class IDs along class
boundaries (e.g. a pixel between brick=12 and stucco=16 can become 14).
This script reassigns every pixel whose value is not a valid class ID to the
label of the nearest valid pixel, then writes the mask as PNG with the same
base name. About 1% of pixels change (at most ~4% in any one image), all of
them along class edges.

Usage:
    python tools/convert_masks_to_png.py DATASETS/*/MASKS [--delete-jpg]
"""

import argparse
import glob
import os

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

VALID_CLASSES = [0, 10, 11, 12, 13, 16, 17, 32]


def clean_mask(mask):
    """Replace invalid class values with the value of the nearest valid pixel."""
    invalid = ~np.isin(mask, VALID_CLASSES)
    if not invalid.any():
        return mask, 0
    # For every pixel, indices of the nearest valid (False) pixel.
    _, (rows, cols) = distance_transform_edt(invalid, return_indices=True)
    return mask[rows, cols], int(invalid.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="+", help="Folders containing *.jpg masks")
    parser.add_argument("--delete-jpg", action="store_true", help="Delete the JPEG after writing the PNG")
    args = parser.parse_args()

    total_changed = total_pixels = 0
    for folder in args.folders:
        for path in sorted(glob.glob(os.path.join(folder, "*.jpg"))):
            mask = np.array(Image.open(path).convert("L"))
            cleaned, changed = clean_mask(mask)
            Image.fromarray(cleaned.astype(np.uint8), mode="L").save(path[:-4] + ".png")
            total_changed += changed
            total_pixels += mask.size
            if args.delete_jpg:
                os.remove(path)
    if total_pixels:
        print(f"Reassigned {total_changed} of {total_pixels} pixels ({100 * total_changed / total_pixels:.2f}%).")


if __name__ == "__main__":
    main()
