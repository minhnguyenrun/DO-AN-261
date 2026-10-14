"""Inference script for M2Doc (Multi-Modal Document Layout Analysis).

Paper: "M2Doc: A Multi-Modal Fusion Approach for Document Layout Analysis" (AAAI 2024)
Scans images from datasets/<dataset>/processed/images and writes prediction
JSON files to outputs/predictions/m2doc/<dataset>/<image_id>.json.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import torch

# Ensure local src directory is on sys.path
SRC_DIR = Path(__file__).resolve().parent
METHOD_DIR = SRC_DIR.parent
ROOT_DIR = METHOD_DIR.parent.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mmcv_shim import setup_m2doc_environment
setup_m2doc_environment()

from ocr_provider import OCRProvider

MMDET_DIR = METHOD_DIR / "mmdetection"
if not (MMDET_DIR / "mmdet").is_dir():
    print(f"[M2Doc] Chưa tìm thấy mã nguồn tại: {MMDET_DIR}")
    print("[M2Doc] Đang tự động clone mã nguồn M2Doc từ GitHub (https://github.com/SCUT-HCIILab/M2Doc.git)...")
    import tempfile
    import shutil
    import subprocess
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/SCUT-HCIILab/M2Doc.git", tmpdir])
        src_mmdet = Path(tmpdir) / "mmdetection"
        if src_mmdet.is_dir():
            shutil.copytree(src_mmdet, MMDET_DIR, dirs_exist_ok=True)
            print("[M2Doc] Thiết lập mã nguồn MMDetection thành công!")
        else:
            raise RuntimeError("Không tìm thấy thư mục mmdetection trong repository M2Doc.")

if str(MMDET_DIR) not in sys.path:
    sys.path.insert(0, str(MMDET_DIR))

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample


M2DOC_CLASSES = (
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
)

# 1-to-1 mapping for DocLayNet
M2DOC_TO_DOCLAYNET = {
    "Caption": 1,
    "Footnote": 2,
    "Formula": 3,
    "List-item": 4,
    "Page-footer": 5,
    "Page-header": 6,
    "Picture": 7,
    "Section-header": 8,
    "Table": 9,
    "Text": 10,
    "Title": 11,
}

# Mapping to PubLayNet 5 categories
M2DOC_TO_PUBLAYNET = {
    "Text": 1,
    "Title": 2,
    "List-item": 3,
    "Table": 4,
    "Picture": 5,
}


GDRIVE_WEIGHTS_ID = "14cf3cmVMWTG8g6w0qMd_lkKYfyR0EDX-"


def download_weights_if_missing(weights_path: Path):
    """Tự động tải pre-trained weights từ Google Drive nếu file chưa tồn tại."""
    if not weights_path.exists() or weights_path.stat().st_size < 1000000:
        print(f"[M2Doc] Chưa tìm thấy trọng số tại: {weights_path}")
        print(f"[M2Doc] Đang tự động tải pre-trained weights từ Google Drive (ID: {GDRIVE_WEIGHTS_ID})...")
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import gdown
        except ImportError:
            import subprocess
            print("[M2Doc] Cài đặt thư viện gdown để tải trọng số...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown

        url = f"https://drive.google.com/uc?id={GDRIVE_WEIGHTS_ID}"
        gdown.download(url, str(weights_path), quiet=False)
        if not weights_path.exists() or weights_path.stat().st_size < 1000000:
            raise FileNotFoundError(
                f"Tải trọng số thất bại. Vui lòng tải thủ công file cascade_mask_rcnn_m2doc_r50.pth "
                f"vào {weights_path.parent} theo hướng dẫn trong README.md."
            )
        print(f"[M2Doc] Tải trọng số thành công ({weights_path.stat().st_size / (1024*1024):.2f} MB)!")


def load_m2doc_model(config_path: Path, weights_path: Path, device: str = "cpu"):
    """Loads M2Doc model from config and weights checkpoint."""
    init_default_scope("mmdet")
    cfg = Config.fromfile(str(config_path))
    model = MODELS.build(cfg.model)

    if not weights_path.exists():
        download_weights_if_missing(weights_path)

    ckpt = torch.load(str(weights_path), map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def run_inference_single(
    model,
    ocr_provider: Optional[OCRProvider],
    image_path: Path,
    mapping: Dict[str, int],
    threshold: float = 0.3,
    device: str = "cpu",
    use_ocr: bool = True,
) -> Dict:
    """Runs inference on a single image and produces standard prediction schema."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    orig_h, orig_w = img.shape[:2]
    target_size = (1025, 1025)
    resized_img = cv2.resize(img, target_size)
    scale_w = target_size[0] / orig_w
    scale_h = target_size[1] / orig_h

    img_tensor = torch.from_numpy(resized_img).permute(2, 0, 1).float().to(device)

    image_id = int(image_path.stem) if image_path.stem.isdigit() else image_path.stem

    data_sample = DetDataSample()
    data_sample.set_metainfo(
        {
            "img_id": image_id,
            "img_path": str(image_path),
            "ori_shape": (orig_h, orig_w),
            "img_shape": target_size,
            "scale_factor": (scale_w, scale_h),
        }
    )

    text_instances = InstanceData()
    if use_ocr and ocr_provider is not None:
        ocr_boxes, ocr_texts = ocr_provider.get_ocr_for_image(
            str(image_path), image_id=image_id if isinstance(image_id, int) else None, target_size=target_size
        )
        if ocr_boxes and ocr_texts:
            text_instances["text_bboxes"] = torch.tensor(ocr_boxes, dtype=torch.float32, device=device)
            text_instances["texts"] = ocr_texts
        else:
            text_instances["text_bboxes"] = torch.zeros((0, 4), dtype=torch.float32, device=device)
            text_instances["texts"] = []
    else:
        text_instances["text_bboxes"] = torch.zeros((0, 4), dtype=torch.float32, device=device)
        text_instances["texts"] = []

    data_sample.text_instances = text_instances

    batch_data = {
        "inputs": [img_tensor],
        "data_samples": [data_sample],
    }

    with torch.no_grad():
        results = model.test_step(batch_data)

    pred = results[0].pred_instances
    bboxes = pred.bboxes.cpu().numpy()
    scores = pred.scores.cpu().numpy()
    labels = pred.labels.cpu().numpy()

    detections = []
    for bbox, score, label_idx in zip(bboxes, scores, labels):
        if score < threshold:
            continue

        class_name = M2DOC_CLASSES[label_idx] if 0 <= label_idx < len(M2DOC_CLASSES) else "Unknown"
        category_id = mapping.get(class_name)

        # Rescale bbox back to original image space: [x1, y1, x2, y2] -> [x, y, w, h]
        x1, y1, x2, y2 = bbox
        x = max(0.0, float(x1))
        y = max(0.0, float(y1))
        w = max(0.0, float(x2 - x1))
        h = max(0.0, float(y2 - y1))

        # Clamp to image dimensions
        x = min(x, float(orig_w))
        y = min(y, float(orig_h))
        w = min(w, float(orig_w - x))
        h = min(h, float(orig_h - y))

        detections.append(
            {
                "category_id": category_id,
                "category_name": class_name,
                "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                "score": round(float(score), 4),
            }
        )

    return {
        "image_id": image_id,
        "file_name": image_path.name,
        "width": orig_w,
        "height": orig_h,
        "detections": detections,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="doclaynet", help="Dataset name (doclaynet or publaynet)")
    parser.add_argument("--threshold", type=float, default=0.3, help="Confidence score threshold (default: 0.3)")
    parser.add_argument("--limit", type=int, default=0, help="Max images to process; 0 for all (default)")
    parser.add_argument("--image-ids", type=str, default="", help="Optional comma-separated image IDs")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(SRC_DIR.parent / "weights" / "cascade_mask_rcnn_m2doc_r50.pth"),
        help="Path to M2Doc checkpoint file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(SRC_DIR.parent / "mmdetection" / "m2doc_config" / "cascade_mask_rcnn_w_m2doc_r50_doclaynet_12epoch.py"),
        help="Path to M2Doc config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Custom output predictions directory",
    )
    parser.add_argument("--device", default="cpu", help="Device: 'cpu' or 'cuda' (default: cpu)")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR text features (visual only)")
    
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip images that already have a prediction JSON (default).",
    )
    run_mode.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing prediction JSON files.",
    )

    args = parser.parse_args()
    skip_existing = not args.force

    repo_root = SRC_DIR.parents[2]
    images_dir = repo_root / "datasets" / args.dataset / "processed" / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "outputs" / "predictions" / "m2doc" / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping = M2DOC_TO_DOCLAYNET if args.dataset == "doclaynet" else M2DOC_TO_PUBLAYNET

    # Collect images
    image_paths = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
    if args.image_ids:
        selected_ids = {s.strip() for s in args.image_ids.split(",") if s.strip()}
        image_paths = [p for p in image_paths if p.stem in selected_ids]

    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    print(f"Dataset: {args.dataset}")
    print(f"Total target images: {len(image_paths)}")
    print(f"Output directory: {output_dir}")
    print(f"Threshold: {args.threshold}")
    print(f"Device: {args.device}")
    print(f"Use OCR: {not args.no_ocr}")

    # Load Model & OCR
    print(f"Loading M2Doc model from: {args.weights}")
    model = load_m2doc_model(Path(args.config), Path(args.weights), device=args.device)
    ocr_provider = OCRProvider() if not args.no_ocr else None
    print("M2Doc model loaded successfully!")

    start_time = time.time()
    processed_count = 0
    skipped_count = 0

    for idx, img_path in enumerate(image_paths, 1):
        image_id = img_path.stem
        out_json_path = output_dir / f"{image_id}.json"

        if skip_existing and out_json_path.exists():
            skipped_count += 1
            continue

        try:
            t0 = time.time()
            pred_record = run_inference_single(
                model=model,
                ocr_provider=ocr_provider,
                image_path=img_path,
                mapping=mapping,
                threshold=args.threshold,
                device=args.device,
                use_ocr=not args.no_ocr,
            )
            with open(out_json_path, "w", encoding="utf-8") as f:
                json.dump(pred_record, f, indent=2, ensure_ascii=False)
            dt = time.time() - t0
            processed_count += 1
            print(
                f"[{idx}/{len(image_paths)}] {img_path.name} -> "
                f"{len(pred_record['detections'])} detections ({dt:.2f}s)"
            )
        except Exception as e:
            print(f"[{idx}/{len(image_paths)}] ERROR processing {img_path.name}: {e}")

    total_time = time.time() - start_time
    print("\n--- Summary ---")
    print(f"Processed: {processed_count}")
    print(f"Skipped:   {skipped_count}")
    print(f"Elapsed:   {total_time:.1f}s")
    if processed_count > 0:
        print(f"Avg/img:   {total_time / processed_count:.2f}s")


if __name__ == "__main__":
    main()
