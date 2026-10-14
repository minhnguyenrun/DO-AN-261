"""Evaluate layout predictions against COCO-style ground-truth annotations."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.common.metrics import evaluate  # noqa: E402


def load_ground_truth(annotation_path: Path) -> tuple[dict, dict]:
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    category_names = {category["id"]: category["name"] for category in data["categories"]}
    ground_truth = {image["id"]: [] for image in data["images"]}

    for annotation in data["annotations"]:
        ground_truth[annotation["image_id"]].append(
            {
                "category_id": annotation["category_id"],
                "bbox": annotation["bbox"],
            }
        )

    return ground_truth, category_names


def load_predictions(predictions_dir: Path) -> tuple[dict, list[dict]]:
    predictions = {}
    documents = []
    for path in predictions_dir.glob("*.json"):
        prediction = json.loads(path.read_text(encoding="utf-8"))
        image_id = prediction["image_id"]
        predictions[image_id] = prediction["detections"]
        documents.extend(prediction["detections"])
    return predictions, documents


def macro_scores(result: dict, category_names: dict) -> dict:
    class_count = sum(
        result["per_category"].get(category_id, {}).get("support", 0) > 0
        for category_id in category_names
    )
    return {
        metric: sum(
            result["per_category"].get(category_id, {}).get(metric, 0.0)
            for category_id in category_names
            if result["per_category"].get(category_id, {}).get("support", 0) > 0
        )
        / class_count
        if class_count
        else 0.0
        for metric in ("precision", "recall", "f1")
    }


def print_report(result: dict, category_names: dict, title: str) -> None:
    overall = result["overall"]
    macro = macro_scores(result, category_names)
    macro_class_count = sum(
        result["per_category"].get(category_id, {}).get("support", 0) > 0
        for category_id in category_names
    )
    print(f"\n=== {title} ===")
    print(
        "Micro: "
        f"Precision={overall['precision']:.3f}  Recall={overall['recall']:.3f}  "
        f"F1={overall['f1']:.3f}  "
        f"TP={overall['tp']} FP={overall['fp']} FN={overall['fn']}  "
        f"mean IoU={overall['mean_iou_matched']:.3f}"
    )
    print(
        f"Macro (mean across {macro_class_count} ground-truth classes): "
        f"Precision={macro['precision']:.3f}  Recall={macro['recall']:.3f}  "
        f"F1={macro['f1']:.3f}"
    )
    print(
        f"{'Class':<18}{'GT':>8}{'Pred':>8}{'TP':>7}{'FP':>7}{'FN':>7}"
        f"{'Prec':>9}{'Recall':>9}{'F1':>9}"
    )

    for category_id, name in sorted(
        category_names.items(),
        key=lambda item: (
            -result["per_category"].get(item[0], {}).get("support", 0),
            item[1],
        ),
    ):
        metrics = result["per_category"].get(
            category_id,
            {
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "support": 0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            },
        )
        pred_count = result["pred_counts"].get(category_id, 0)
        print(
            f"{name:<18}{metrics['support']:>8}{pred_count:>8}{metrics['tp']:>7}"
            f"{metrics['fp']:>7}{metrics['fn']:>7}{metrics['precision']:>9.3f}"
            f"{metrics['recall']:>9.3f}{metrics['f1']:>9.3f}"
        )


def plot_evaluation_results(
    strict: dict,
    category_names: dict,
    loose: dict | None = None,
    title: str = "Evaluation Metrics",
    output_path: Path | str | None = None,
) -> plt.Figure:
    """Vẽ biểu đồ so sánh Precision, Recall, F1 theo từng class và so sánh Strict vs Loose."""
    sorted_items = sorted(
        category_names.items(),
        key=lambda item: -strict["per_category"].get(item[0], {}).get("support", 0)
    )
    labels = [name for _, name in sorted_items]
    y = np.arange(len(labels))
    height = 0.25

    precisions = [strict["per_category"].get(cid, {}).get("precision", 0.0) for cid, _ in sorted_items]
    recalls = [strict["per_category"].get(cid, {}).get("recall", 0.0) for cid, _ in sorted_items]
    f1s = [strict["per_category"].get(cid, {}).get("f1", 0.0) for cid, _ in sorted_items]

    if loose is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    else:
        fig, ax1 = plt.subplots(figsize=(9, 6))
        ax2 = None

    # Subplot 1: Strict Precision, Recall, F1
    ax1.barh(y - height, precisions, height, label="Precision", color="#3b82f6")
    ax1.barh(y, recalls, height, label="Recall", color="#f97316")
    ax1.barh(y + height, f1s, height, label="F1-Score", color="#10b981")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1.05)
    ax1.set_xlabel("Score", fontsize=11)
    ax1.set_title("Strict Metrics (Location & Class)", fontsize=12, fontweight="bold")
    ax1.grid(axis="x", linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    ov = strict["overall"]
    ax1.text(
        0.02, 0.02,
        f"Micro: P={ov['precision']:.3f} | R={ov['recall']:.3f} | F1={ov['f1']:.3f}",
        transform=ax1.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#ccc")
    )

    # Subplot 2: Strict F1 vs Loose F1 comparison
    if ax2 is not None:
        loose_f1s = [loose["per_category"].get(cid, {}).get("f1", 0.0) for cid, _ in sorted_items]
        h2 = 0.35
        ax2.barh(y - h2 / 2, f1s, h2, label="Strict F1 (Class & Location)", color="#10b981")
        ax2.barh(y + h2 / 2, loose_f1s, h2, label="Loose F1 (Location only)", color="#8b5cf6")
        ax2.set_xlim(0, 1.05)
        ax2.set_xlabel("F1-Score", fontsize=11)
        ax2.set_title("F1 Comparison: Strict vs Loose (Localization)", fontsize=12, fontweight="bold")
        ax2.grid(axis="x", linestyle="--", alpha=0.5)
        ax2.legend(loc="lower right")

        lov = loose["overall"]
        ax2.text(
            0.02, 0.02,
            f"Loose Micro: P={lov['precision']:.3f} | R={lov['recall']:.3f} | F1={lov['f1']:.3f}",
            transform=ax2.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#ccc")
        )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=200, bbox_inches="tight")
        print(f"[eval] Evaluation chart saved to: {out_p}")

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="docling")
    parser.add_argument("--dataset", default="doclaynet")
    parser.add_argument("--split", default="train")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--loose",
        action="store_true",
        help="Also report localization-only scores, ignoring class labels.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Cho phép đánh giá trên tập ảnh hiện có dự đoán (hữu ích khi test với --limit).",
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        const="default",
        default=None,
        help="Lưu biểu đồ đánh giá ra file ảnh (mặc định: outputs/reports/<method>_<dataset>/metrics_chart.png).",
    )
    args = parser.parse_args()

    annotation_path = (
        ROOT_DIR
        / "datasets"
        / args.dataset
        / "processed"
        / "annotations"
        / f"{args.split}.json"
    )
    predictions_dir = ROOT_DIR / "outputs" / "predictions" / args.method / args.dataset

    ground_truth, category_names = load_ground_truth(annotation_path)
    predictions, detections = load_predictions(predictions_dir)
    missing_ids = set(ground_truth) - set(predictions)
    if missing_ids:
        if args.allow_incomplete:
            ground_truth = {k: v for k, v in ground_truth.items() if k in predictions}
        else:
            examples = ", ".join(map(str, sorted(missing_ids, key=str)[:10]))
            raise SystemExit(
                f"Incomplete predictions: {len(missing_ids)} of {len(ground_truth)} "
                f"ground-truth images are missing (e.g. {examples}). Run inference first (hoặc dùng --allow-incomplete)."
            )

    pred_counts = Counter(detection["category_id"] for detection in detections)
    strict = evaluate(
        ground_truth,
        predictions,
        category_names,
        args.iou_threshold,
        require_same_category=True,
    )
    strict["pred_counts"] = pred_counts

    ground_truth_count = sum(map(len, ground_truth.values()))
    print(
        f"Evaluation: {args.method} on {args.dataset}/{args.split} | "
        f"{len(ground_truth)} images | GT={ground_truth_count:,} boxes | "
        f"Predictions={len(detections):,} boxes | IoU >= {args.iou_threshold}"
    )
    print_report(strict, category_names, "Strict (location and class)")

    unsupported_labels = Counter(
        detection["category_name"]
        for detection in detections
        if detection.get("category_id") not in category_names
    )
    if unsupported_labels:
        print(f"\n{args.method.capitalize()} labels outside the {args.dataset.capitalize()} taxonomy (counted as strict FP):")
        for label, count in sorted(unsupported_labels.items(), key=lambda item: (-item[1], item[0])):
            print(f"  {label}: {count}")

    loose = None
    if args.loose:
        loose = evaluate(
            ground_truth,
            predictions,
            category_names,
            args.iou_threshold,
            require_same_category=False,
        )
        loose["pred_counts"] = pred_counts
        print_report(loose, category_names, "Localization-only (class ignored)")

    if args.plot:
        if args.plot == "default":
            plot_path = ROOT_DIR / "outputs" / "reports" / f"{args.method}_{args.dataset}" / "metrics_chart.png"
        else:
            plot_path = Path(args.plot)
        plot_evaluation_results(
            strict,
            category_names,
            loose=loose,
            title=f"{args.method.capitalize()} Layout Evaluation ({args.dataset}/{args.split}, IoU >= {args.iou_threshold})",
            output_path=plot_path,
        )


if __name__ == "__main__":
    main()
