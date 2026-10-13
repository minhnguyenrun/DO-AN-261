"""Tinh Precision/Recall/F1 cua du doan mot method so voi Ground Truth.

Chi danh gia tren nhung anh da co file du doan trong
outputs/predictions/<method>/<dataset>/ (vd: 20 anh dau da chay docling).

Usage:
    python scripts/evaluate_predictions.py --method docling --dataset doclaynet
    python scripts/evaluate_predictions.py --method docling --dataset doclaynet --loose
"""

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR / ".."
sys.path.insert(0, str(ROOT_DIR / "src"))

from common.metrics import evaluate  # noqa: E402


def load_gt(gt_path: Path) -> tuple:
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    category_names = {c["id"]: c["name"] for c in gt["categories"]}
    gt_by_image = {image["id"]: [] for image in gt["images"]}
    for ann in gt["annotations"]:
        gt_by_image.setdefault(ann["image_id"], []).append(
            {"category_id": ann["category_id"], "bbox": ann["bbox"]})
    return gt_by_image, category_names


def load_predictions(predictions_dir: Path) -> dict:
    pred_by_image = {}
    for path in predictions_dir.glob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            pred = json.load(f)
        pred_by_image[pred["image_id"]] = [
            {"category_id": d["category_id"], "bbox": d["bbox"]}
            for d in pred["detections"]
        ]
    return pred_by_image


def print_report(result: dict, title: str):
    print(f"\n=== {title} ===")
    o = result["overall"]
    print(f"Overall: precision={o['precision']:.3f}  recall={o['recall']:.3f}  "
          f"f1={o['f1']:.3f}  (TP={o['tp']} FP={o['fp']} FN={o['fn']}, "
          f"mean IoU cac box khop={o['mean_iou_matched']:.3f})")
    print(f"{'Category':<16}{'Support':>8}{'TP':>6}{'FP':>6}{'FN':>6}"
          f"{'Prec':>8}{'Rec':>8}{'F1':>8}")
    for cat_id, c in sorted(result["per_category"].items(),
                             key=lambda kv: kv[1]["support"], reverse=True):
        print(f"{c['name']:<16}{c['support']:>8}{c['tp']:>6}{c['fp']:>6}{c['fn']:>6}"
              f"{c['precision']:>8.3f}{c['recall']:>8.3f}{c['f1']:>8.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="docling")
    parser.add_argument("--dataset", default="doclaynet")
    parser.add_argument("--split", default="train")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--loose", action="store_true",
                         help="Them bao cao so khop chi theo vi tri (bo qua nhan), "
                              "de chan doan loi phan loai sai vs khong phat hien duoc")
    args = parser.parse_args()

    gt_path = ROOT_DIR / "datasets" / args.dataset / "processed" / "annotations" / f"{args.split}.json"
    predictions_dir = ROOT_DIR / "outputs" / "predictions" / args.method / args.dataset

    gt_by_image, category_names = load_gt(gt_path)
    pred_by_image = load_predictions(predictions_dir)

    # Chi danh gia tren tap anh da co du doan.
    gt_subset = {k: v for k, v in gt_by_image.items() if k in pred_by_image}
    print(f"Danh gia {args.method} tren {len(gt_subset)} anh ({args.dataset}/{args.split}), "
          f"IoU >= {args.iou_threshold}")

    strict = evaluate(gt_subset, pred_by_image, category_names, args.iou_threshold,
                       require_same_category=True)
    print_report(strict, "So khop theo CA vi tri lan nhan (strict)")

    if args.loose:
        loose = evaluate(gt_subset, pred_by_image, category_names, args.iou_threshold,
                          require_same_category=False)
        print_report(loose, "So khop chi theo vi tri, bo qua nhan (loose / localization-only)")


if __name__ == "__main__":
    main()
