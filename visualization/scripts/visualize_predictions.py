"""Ve bounding box Ground Truth vs. du doan cua mot method, canh nhau, de
dua vao bao cao.

Doc annotation goc tu datasets/<dataset>/processed/annotations/<split>.json
va ket qua du doan tu outputs/predictions/<method>/<dataset>/<image_id>.json,
roi ve 2 panel (Ground Truth | <Method>) len cung mot anh, luu vao
visualization/figures/<method>_<dataset>/<image_id>.png

Usage:
    python visualization/scripts/visualize_predictions.py --method docling --dataset doclaynet --limit 10
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR / "../.."

CATEGORY_NAMES = {
    1: "Caption", 2: "Footnote", 3: "Formula", 4: "List-item",
    5: "Page-footer", 6: "Page-header", 7: "Picture", 8: "Section-header",
    9: "Table", 10: "Text", 11: "Title",
}

# Bang mau co dinh cho 11 nhan DocLayNet, de GT va Prediction dung cung mau
# cho cung mot nhan.
CATEGORY_COLORS = {
    1: (230, 25, 75), 2: (60, 180, 75), 3: (255, 225, 25), 4: (0, 130, 200),
    5: (245, 130, 48), 6: (145, 30, 180), 7: (70, 240, 240), 8: (240, 50, 230),
    9: (210, 245, 60), 10: (0, 128, 128), 11: (170, 110, 40),
}
UNKNOWN_COLOR = (128, 128, 128)
PAD = 40


def load_font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_boxes(image: Image.Image, boxes: list, font: ImageFont.ImageFont) -> Image.Image:
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    for box in boxes:
        category_id = box["category_id"]
        label = box.get("category_name") or CATEGORY_NAMES.get(category_id, "unknown")
        color = CATEGORY_COLORS.get(category_id, UNKNOWN_COLOR)
        x, y, w, h = box["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
        draw.text((x + 2, max(y - 14, 0)), label, fill=color, font=font)
    return out


def make_side_by_side(left: Image.Image, right: Image.Image, left_title: str, right_title: str) -> Image.Image:
    font = load_font(18)
    w, h = left.size
    canvas = Image.new("RGB", (w * 2 + PAD, h + PAD), (255, 255, 255))
    canvas.paste(left, (0, PAD))
    canvas.paste(right, (w + PAD, PAD))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), left_title, fill=(0, 0, 0), font=font)
    draw.text((w + PAD + 4, 4), right_title, fill=(0, 0, 0), font=font)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="docling", help="Ten method (vd: docling)")
    parser.add_argument("--dataset", default="doclaynet", help="Ten dataset (vd: doclaynet)")
    parser.add_argument("--split", default="train", help="Split cua ground truth (vd: train)")
    parser.add_argument("--limit", type=int, default=10, help="So anh toi da se ve (mac dinh 10)")
    args = parser.parse_args()

    images_dir = ROOT_DIR / "datasets" / args.dataset / "processed" / "images"
    predictions_dir = ROOT_DIR / "outputs" / "predictions" / args.method / args.dataset
    gt_path = ROOT_DIR / "datasets" / args.dataset / "processed" / "annotations" / f"{args.split}.json"
    output_dir = ROOT_DIR / "visualization" / "figures" / f"{args.method}_{args.dataset}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(gt_path, "r", encoding="utf-8") as f:
        gt = json.load(f)
    gt_by_image = {}
    for ann in gt["annotations"]:
        gt_by_image.setdefault(ann["image_id"], []).append(ann)

    prediction_paths = sorted(predictions_dir.glob("*.json"),
                               key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    if args.limit > 0:
        prediction_paths = prediction_paths[: args.limit]

    font = load_font(14)
    print(f"[viz] Ve {len(prediction_paths)} anh ({args.method} vs ground truth) tu {predictions_dir}")
    for pred_path in prediction_paths:
        with open(pred_path, "r", encoding="utf-8") as f:
            pred = json.load(f)
        image_id = pred["image_id"]
        image_path = images_dir / pred["file_name"]
        if not image_path.exists():
            print(f"  [skip] khong tim thay anh {image_path}")
            continue

        base = Image.open(image_path)
        gt_boxes = gt_by_image.get(image_id, [])
        left = draw_boxes(base, gt_boxes, font)
        right = draw_boxes(base, pred["detections"], font)
        combined = make_side_by_side(
            left, right,
            f"Ground Truth ({len(gt_boxes)} box)",
            f"{args.method} ({len(pred['detections'])} box)",
        )

        out_path = output_dir / f"{image_id}.png"
        combined.save(out_path)
        print(f"  [ok] {out_path.name}")

    print(f"[viz] Xong. Anh luu tai {output_dir}")


if __name__ == "__main__":
    main()
