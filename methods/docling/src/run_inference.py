"""Chạy layout detection bằng Docling trên ảnh đã xử lý của một dataset.

Chỉ chạy stage layout (tắt OCR, table structure, các enrichment khác) để
nhanh và đủ cho bài toán Document Layout Analysis. Model mặc định của
Docling (`docling-layout-heron`) được huấn luyện trên chính DocLayNet, nên
nhãn đầu ra (DocItemLabel) khớp trực tiếp với 11 nhãn của DocLayNet.

Kết quả dự đoán được lưu mỗi ảnh một file JSON tại
outputs/predictions/docling/<dataset>/<image_id>.json

Usage:
    python methods/docling/src/run_inference.py --dataset doclaynet --limit 20
"""

import argparse
import json
import time
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, ImageFormatOption

THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR / "../../.."

# Map nhãn Docling (DocItemLabel) -> category_id DocLayNet (xem
# scripts/download_datasets/download_doclaynet.py). Nhãn Docling không có
# trong DocLayNet (chart, code, ...) được giữ lại với category_id = None.
LABEL_TO_DOCLAYNET_ID = {
    "caption": 1,
    "footnote": 2,
    "formula": 3,
    "list_item": 4,
    "page_footer": 5,
    "page_header": 6,
    "picture": 7,
    "section_header": 8,
    "table": 9,
    "text": 10,
    "title": 11,
}


def build_converter() -> DocumentConverter:
    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = False
    opts.do_picture_classification = False
    opts.do_picture_description = False
    opts.do_chart_extraction = False
    opts.do_code_enrichment = False
    opts.do_formula_enrichment = False
    opts.generate_page_images = False
    # Khong OCR nen cluster text se "rong" (khong co noi dung) -> phai giu lai
    # de khong bi loai khoi ket qua layout.
    opts.layout_options.keep_empty_clusters = True

    return DocumentConverter(
        allowed_formats=[InputFormat.IMAGE],
        format_options={InputFormat.IMAGE: ImageFormatOption(pipeline_options=opts)},
    )


def run_on_image(converter: DocumentConverter, image_path: Path) -> dict:
    result = converter.convert(str(image_path))
    doc = result.document
    page = doc.pages[1]
    width, height = page.size.width, page.size.height

    detections = []
    for item, _level in doc.iterate_items():
        if not item.prov:
            continue
        label = item.label.value
        bbox = item.prov[0].bbox.to_top_left_origin(height)
        detections.append({
            "category_id": LABEL_TO_DOCLAYNET_ID.get(label),
            "category_name": label,
            "bbox": [bbox.l, bbox.t, bbox.r - bbox.l, bbox.b - bbox.t],
        })

    return {
        "image_id": int(image_path.stem) if image_path.stem.isdigit() else image_path.stem,
        "file_name": image_path.name,
        "width": width,
        "height": height,
        "detections": detections,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="doclaynet", help="Ten dataset (vd: doclaynet)")
    parser.add_argument("--limit", type=int, default=20, help="So anh toi da se chay (mac dinh 20)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Bo qua anh da co file prediction de tiep tuc lan chay truoc")
    args = parser.parse_args()

    images_dir = ROOT_DIR / "datasets" / args.dataset / "processed" / "images"
    output_dir = ROOT_DIR / "outputs" / "predictions" / "docling" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(images_dir.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    if args.limit > 0:
        image_paths = image_paths[: args.limit]
    if args.skip_existing:
        image_paths = [
            image_path for image_path in image_paths
            if not (output_dir / f"{image_path.stem}.json").exists()
        ]

    print(f"[docling] Chay layout detection tren {len(image_paths)} anh tu {images_dir}")
    converter = build_converter()

    total_time = 0.0
    for i, image_path in enumerate(image_paths, 1):
        t0 = time.time()
        prediction = run_on_image(converter, image_path)
        dt = time.time() - t0
        total_time += dt

        out_path = output_dir / f"{image_path.stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(prediction, f)

        print(f"  [{i}/{len(image_paths)}] {image_path.name}: "
              f"{len(prediction['detections'])} box ({dt:.2f}s)")

    print(f"[docling] Xong. Trung binh {total_time / max(len(image_paths), 1):.2f}s/anh. "
          f"Ket qua tai {output_dir}")


if __name__ == "__main__":
    main()
