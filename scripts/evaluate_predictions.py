"""Tinh Precision/Recall/F1 cua du doan mot method so voi Ground Truth.

Chi danh gia tren nhung anh da co file du doan trong
outputs/predictions/<method>/<dataset>/ (vd: 20 anh dau da chay docling).

Ngoai Precision/Recall/F1 tai 1 nguong IoU, script con tinh mAP kieu COCO
(mAP@0.5:0.95, mAP@0.5, mAP@0.75, AR) neu file du doan co truong "score".

Usage:
    python scripts/evaluate_predictions.py --method docling --dataset doclaynet
    python scripts/evaluate_predictions.py --method docling --dataset doclaynet --loose
    python scripts/evaluate_predictions.py --method docling --dataset doclaynet --no-map
"""

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR / ".."
sys.path.insert(0, str(ROOT_DIR / "src"))

from common.metrics import coco_map, evaluate  # noqa: E402


def load_gt(gt_path: Path) -> tuple:
    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    category_names = {c["id"]: c["name"] for c in gt["categories"]}
    gt_by_image = {image["id"]: [] for image in gt["images"]}
    for ann in gt["annotations"]:
        gt_by_image.setdefault(ann["image_id"], []).append(
            {"category_id": ann["category_id"], "bbox": ann["bbox"]})
    return gt, gt_by_image, category_names


def load_predictions(predictions_dir: Path) -> dict:
    pred_by_image = {}
    for path in predictions_dir.glob("*.json"):
        with open(path, "r", encoding="utf-8") as f:
            pred = json.load(f)
        pred_by_image[pred["image_id"]] = [
            {"category_id": d["category_id"], "bbox": d["bbox"], "score": d.get("score")}
            for d in pred["detections"]
        ]
    return pred_by_image


def run_coco_map(gt: dict, pred_by_image: dict, image_ids: set):
    """In bang mAP kieu COCO tren tap anh image_ids."""
    has_score = all(d["score"] is not None
                    for img_id in image_ids for d in pred_by_image[img_id])
    if not has_score:
        print("\n[mAP] Bo qua: file du doan khong co truong 'score' "
              "(chay lai inference de luu confidence).")
        return

    cat_ids = {c["id"] for c in gt["categories"]}
    gt_subset = {
        "images": [img for img in gt["images"] if img["id"] in image_ids],
        "annotations": [ann for ann in gt["annotations"] if ann["image_id"] in image_ids],
        "categories": gt["categories"],
    }
    # Box co nhan khong thuoc bo nhan cua dataset (category_id None) bi loai.
    predictions = [
        {"image_id": img_id, "category_id": d["category_id"], "bbox": d["bbox"], "score": d["score"]}
        for img_id in image_ids for d in pred_by_image[img_id]
        if d["category_id"] in cat_ids
    ]
    result = coco_map(gt_subset, predictions)

    def fmt(x):
        return f"{x:.3f}" if x >= 0 else "  -  "

    o = result["overall"]
    print("\n=== mAP kieu COCO (maxDets=100) ===")
    print(f"mAP@[.5:.95]={fmt(o['map'])}  mAP@.5={fmt(o['map50'])}  mAP@.75={fmt(o['map75'])}")
    print(f"mAP small/medium/large={fmt(o['map_small'])}/{fmt(o['map_medium'])}/{fmt(o['map_large'])}  "
          f"AR@1/10/100={fmt(o['ar1'])}/{fmt(o['ar10'])}/{fmt(o['ar100'])}")
    print(f"{'Category':<16}{'AP':>8}{'AP50':>8}")
    for c in sorted(result["per_category"].values(), key=lambda c: c["ap"], reverse=True):
        print(f"{c['name']:<16}{fmt(c['ap']):>8}{fmt(c['ap50']):>8}")


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
    parser.add_argument("--no-map", action="store_true", help="Khong tinh mAP kieu COCO")
    args = parser.parse_args()

    gt_path = ROOT_DIR / "datasets" / args.dataset / "processed" / "annotations" / f"{args.split}.json"
    predictions_dir = ROOT_DIR / "outputs" / "predictions" / args.method / args.dataset

    gt, gt_by_image, category_names = load_gt(gt_path)
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

    if not args.no_map:
        run_coco_map(gt, pred_by_image, set(gt_subset))


if __name__ == "__main__":
    main()
