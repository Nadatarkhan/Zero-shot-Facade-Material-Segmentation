"""Evaluate predicted material masks against the ground-truth masks.

Predictions are the class-ID arrays written by run_zero_shot_inference.py
(``<output-dir>/CLIPSEG/*.npy``); ground truth are the masks in
``DATASETS/<CITY>/MASKS/*.png``. Files are matched by base name.

Metrics (per class, averaged over images, as in validation_workflow.ipynb):
  - IoU, precision, recall. A class is skipped for an image when it covers less
    than ``--presence-threshold`` (default 5%) of the pixels in BOTH the
    prediction and the ground truth.
  - Total (mean of classes): unweighted mean of the per-class values. This is
    how the "Total (weighted)" rows in the paper were computed.
  - Total (pixel-weighted): per-class values weighted by each class's share of
    ground-truth pixels (Eq. 1 in the paper), reported for reference.
Facade-level metrics (paper, Eqs. 2-3):
  - Predominant material accuracy: the material covering the largest area is
    the same in prediction and ground truth.
  - Top-3 material presence: share of the (up to) three largest ground-truth
    materials that also appear in the prediction (>= --detect-threshold of pixels).

Usage:
    python evaluate.py --pred-dir OUTPUTS/CLIPSEG --gt-dirs DATASETS/*/MASKS
    python evaluate.py --pred-dir OUTPUTS/CLIPSEG --gt-dirs DATASETS/*/MASKS --by-city --csv results.csv
"""

import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

CLASSES = {10: "glass", 11: "concrete", 12: "brick", 13: "siding", 16: "plaster", 17: "metal", 32: "door"}
MATERIALS = [10, 11, 12, 13, 16, 17]  # classes considered for the facade-level metrics


def load_gt(path):
    return np.array(Image.open(path).convert("L"))


def load_pred(path, shape):
    pred = np.load(path).astype(np.uint8)
    if pred.shape != shape:
        # Nearest-neighbour keeps class IDs intact. (The original notebook used
        # bilinear resizing, which can create intermediate IDs along edges, so
        # results may differ very slightly from the paper.)
        pred = np.array(Image.fromarray(pred).resize((shape[1], shape[0]), Image.NEAREST))
    return pred


def class_stats(pred, gt, c, presence_threshold):
    p, g = pred == c, gt == c
    if p.mean() < presence_threshold and g.mean() < presence_threshold:
        return np.nan, np.nan, np.nan
    tp = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    iou = tp / union if union else 0.0
    precision = tp / p.sum() if p.sum() else 0.0
    recall = tp / g.sum() if g.sum() else 0.0
    return iou, precision, recall


def facade_stats(pred, gt, detect_threshold):
    gt_area = np.array([(gt == c).sum() for c in MATERIALS])
    if gt_area.sum() == 0:
        return np.nan, np.nan
    pred_area = np.array([(pred == c).sum() for c in MATERIALS])
    predominant_ok = float(np.argmax(gt_area) == np.argmax(pred_area)) if pred_area.sum() else 0.0
    present = [i for i in np.argsort(-gt_area)[:3] if gt_area[i] / gt.size >= detect_threshold]
    if not present:
        present = [int(np.argmax(gt_area))]
    detected = sum(pred_area[i] / pred.size >= detect_threshold for i in present)
    return predominant_ok, detected / len(present)


def find_pairs(pred_dir, gt_dirs):
    gt_files = {}
    for d in gt_dirs:
        city = os.path.basename(os.path.dirname(os.path.normpath(d))) or d
        for f in os.listdir(d):
            stem, ext = os.path.splitext(f)
            if ext.lower() in (".png", ".jpg", ".jpeg"):
                gt_files.setdefault(stem, (os.path.join(d, f), city))
    pairs, missing = [], []
    for f in sorted(os.listdir(pred_dir)):
        if not f.endswith(".npy"):
            continue
        stem = f[:-4]
        if stem in gt_files:
            pairs.append((os.path.join(pred_dir, f), *gt_files[stem]))
        else:
            missing.append(f)
    return pairs, missing


def summarize(rows):
    df = pd.DataFrame(rows)
    out = []
    for c, name in CLASSES.items():
        out.append({
            "class": name,
            "IoU": np.nanmean(df[f"iou_{c}"]) if df[f"iou_{c}"].notna().any() else np.nan,
            "precision": np.nanmean(df[f"prec_{c}"]) if df[f"prec_{c}"].notna().any() else np.nan,
            "recall": np.nanmean(df[f"rec_{c}"]) if df[f"rec_{c}"].notna().any() else np.nan,
            "gt_pixels": df[f"gtpx_{c}"].sum(),
        })
    table = pd.DataFrame(out).set_index("class")
    metrics = ["IoU", "precision", "recall"]
    mean_row = table[metrics].mean(skipna=True)
    valid = table.dropna(subset=["IoU"])
    weights = valid["gt_pixels"] / valid["gt_pixels"].sum() if valid["gt_pixels"].sum() else valid["gt_pixels"]
    weighted_row = (valid[metrics].fillna(0).mul(weights, axis=0)).sum()
    table.loc["Total (mean of classes)", metrics] = mean_row
    table.loc["Total (pixel-weighted)", metrics] = weighted_row
    facade = {
        "images": len(df),
        "predominant_material_accuracy_%": 100 * np.nanmean(df["predominant_ok"]),
        "top3_material_presence_%": 100 * np.nanmean(df["top3_presence"]),
    }
    return table.drop(columns="gt_pixels"), facade


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-dir", required=True, help="Folder with predicted *.npy class-ID arrays")
    parser.add_argument("--gt-dirs", nargs="+", required=True, help="Ground-truth mask folders (e.g. DATASETS/*/MASKS)")
    parser.add_argument("--presence-threshold", type=float, default=0.05,
                        help="Skip a class in an image if it covers less than this share of pixels in both prediction and ground truth (default 0.05)")
    parser.add_argument("--detect-threshold", type=float, default=0.01,
                        help="Minimum pixel share for a material to count as present, for the facade-level metrics (default 0.01)")
    parser.add_argument("--by-city", action="store_true", help="Also report results per ground-truth folder (city)")
    parser.add_argument("--csv", help="Write the per-class table(s) to this CSV file")
    args = parser.parse_args()

    pairs, missing = find_pairs(args.pred_dir, args.gt_dirs)
    if missing:
        print(f"Warning: {len(missing)} prediction(s) have no matching ground truth, e.g. {missing[0]}")
    if not pairs:
        raise SystemExit("No prediction/ground-truth pairs found.")

    rows = []
    for pred_path, gt_path, city in pairs:
        gt = load_gt(gt_path)
        pred = load_pred(pred_path, gt.shape)
        row = {"image": os.path.basename(pred_path), "city": city}
        for c in CLASSES:
            row[f"iou_{c}"], row[f"prec_{c}"], row[f"rec_{c}"] = class_stats(pred, gt, c, args.presence_threshold)
            row[f"gtpx_{c}"] = int((gt == c).sum()) if not np.isnan(row[f"iou_{c}"]) else 0
        row["predominant_ok"], row["top3_presence"] = facade_stats(pred, gt, args.detect_threshold)
        rows.append(row)

    groups = [("ALL", rows)]
    if args.by_city:
        for city in sorted({r["city"] for r in rows}):
            groups.append((city, [r for r in rows if r["city"] == city]))

    tables = []
    for name, group in groups:
        table, facade = summarize(group)
        print(f"\n=== {name} ({facade['images']} images) ===")
        print(table.round(3).to_string())
        print(f"Predominant material accuracy: {facade['predominant_material_accuracy_%']:.1f}%")
        print(f"Top-3 material presence:       {facade['top3_material_presence_%']:.1f}%")
        tables.append(table.assign(group=name))

    if args.csv:
        pd.concat(tables).round(4).to_csv(args.csv)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
