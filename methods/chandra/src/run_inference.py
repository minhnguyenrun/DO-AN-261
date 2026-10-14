"""Inference CLI for Chandra OCR (Datalab) layout analysis.

Supports DocLayNet and PubLayNet datasets with automatic weight downloading
from Hugging Face Hub (datalab-to/chandra or datalab-to/chandra-ocr-2),
parsing layout blocks and bounding boxes into evaluation-ready outputs.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
METHOD_DIR = SRC_DIR.parent
METHODS_ROOT = METHOD_DIR.parent
PROJECT_ROOT = METHODS_ROOT.parent

# Fix Windows console UTF-8 encoding
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from parser import parse_chandra_layout_response

DEFAULT_MODEL_ID = "datalab-to/chandra-ocr-2"

PROMPT_CHANDRA_LAYOUT = (
    "Extract all document layout structures, blocks, and categories with their bounding boxes. "
    "Include bounding boxes in format [x1, y1, x2, y2] and classify into categories: "
    "['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', "
    "'Table', 'Text', 'Title']. Return the layout blocks in JSON format."
)


def ensure_model_weights(weights_path: Path, model_id: str = DEFAULT_MODEL_ID) -> str:
    """Checks whether local weights exist, or downloads from Hugging Face Hub."""
    has_safetensors = any(weights_path.glob("*.safetensors")) if weights_path.is_dir() else False
    if weights_path.is_dir() and (weights_path / "config.json").exists() and has_safetensors:
        print(f"[Chandra] Sử dụng trọng số có sẵn tại: {weights_path}")
        return str(weights_path)

    print(f"[Chandra] Chưa tìm thấy trọng số đầy đủ tại: {weights_path}")
    print(f"[Chandra] Đang chuẩn bị tải từ Hugging Face Hub: {model_id}...")
    try:
        from huggingface_hub import snapshot_download

        weights_path.mkdir(parents=True, exist_ok=True)
        local_dir = snapshot_download(
            repo_id=model_id,
            local_dir=str(weights_path),
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
        )
        print(f"[Chandra] Tải model thành công về: {local_dir}")
        return str(local_dir)
    except Exception as e:
        print(f"[Chandra] Cảnh báo lỗi tải offline ({e}). Sẽ thử gọi trực tiếp Hugging Face Hub: {model_id}")
        return model_id


def load_chandra_model(model_path_or_id: str, device: str = "cpu"):
    """Loads Chandra model and processor."""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

    print(f"[Chandra] Khởi tạo mô hình từ: {model_path_or_id} (device={device})")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() or hasattr(torch, "bfloat16") else torch.float32

    model = None
    # Try AutoModelForImageTextToText first, fallback to AutoModelForCausalLM
    for model_cls in [AutoModelForImageTextToText, AutoModelForCausalLM]:
        try:
            model = model_cls.from_pretrained(
                model_path_or_id,
                torch_dtype=torch_dtype,
                device_map=device if device != "cpu" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            break
        except Exception:
            continue

    if model is None:
        raise RuntimeError(f"Không thể khởi tạo mô hình Chandra từ {model_path_or_id}")

    if device == "cpu":
        model.to("cpu")
    model.eval()

    processor = AutoProcessor.from_pretrained(model_path_or_id, trust_remote_code=True)
    return model, processor


def run_inference_single(
    model,
    processor,
    image_path: Path,
    device: str = "cpu",
    max_new_tokens: int = 1536,
) -> Dict:
    """Runs layout extraction on a single image and returns standard detection format."""
    pil_img = Image.open(image_path).convert("RGB")
    width, height = pil_img.size

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": PROMPT_CHANDRA_LAYOUT},
            ],
        }
    ]

    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[pil_img],
            padding=True,
            return_tensors="pt",
        )
    except Exception:
        inputs = processor(
            text=PROMPT_CHANDRA_LAYOUT,
            images=pil_img,
            return_tensors="pt",
        )

    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    in_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
    out_ids = generated_ids[0][in_len:]
    output_text = processor.decode(out_ids, skip_special_tokens=True).strip()

    # Parse layout response
    detections = parse_chandra_layout_response(output_text, img_width=width, img_height=height)

    image_id = int(image_path.stem) if image_path.stem.isdigit() else image_path.stem
    return {
        "image_id": image_id,
        "file_name": image_path.name,
        "width": width,
        "height": height,
        "detections": detections,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="doclaynet", help="Dataset name (doclaynet or publaynet)")
    parser.add_argument("--limit", type=int, default=20, help="Max images to process; 0 for all")
    parser.add_argument("--image-ids", default="", help="Optional comma-separated image IDs")
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model ID (datalab-to/chandra or datalab-to/chandra-ocr-2)",
    )
    parser.add_argument(
        "--weights",
        default=str(METHOD_DIR / "weights"),
        help="Path to local weights directory or Hugging Face model ID",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Custom output predictions directory",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device (cuda or cpu)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1536,
        help="Max generated tokens for layout JSON",
    )
    args = parser.parse_args()

    # Determine image folder
    dataset_name = args.dataset.lower()
    if dataset_name == "doclaynet":
        img_dir = PROJECT_ROOT / "data" / "doclaynet" / "images"
    elif dataset_name == "publaynet":
        img_dir = PROJECT_ROOT / "data" / "publaynet" / "val"
    else:
        img_dir = Path(args.dataset)

    if not img_dir.exists():
        print(f"Error: Dataset image directory does not exist: {img_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect images
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    all_images = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in image_extensions])

    if args.image_ids:
        selected_ids = {s.strip() for s in args.image_ids.split(",") if s.strip()}
        all_images = [p for p in all_images if p.stem in selected_ids or p.name in selected_ids]

    if args.limit > 0:
        all_images = all_images[: args.limit]

    print(f"Dataset: {args.dataset}")
    print(f"Total target images: {len(all_images)}")

    # Output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = PROJECT_ROOT / "outputs" / "predictions" / "chandra" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")
    print(f"Device: {args.device}")

    # Prepare weights
    weights_arg = Path(args.weights)
    if weights_arg.exists() or not args.weights.startswith("datalab-to"):
        model_target = ensure_model_weights(weights_arg, args.model_id)
    else:
        model_target = args.weights

    # Load Model
    start_load = time.time()
    model, processor = load_chandra_model(model_target, device=args.device)
    print(f"Model loaded successfully in {time.time() - start_load:.2f}s")

    # Run inference loop
    success_count = 0
    total_time = 0.0

    for i, img_path in enumerate(all_images, start=1):
        print(f"[{i}/{len(all_images)}] Đang xử lý: {img_path.name} ... ", end="", flush=True)
        t0 = time.time()
        try:
            res = run_inference_single(
                model=model,
                processor=processor,
                image_path=img_path,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
            )
            elapsed = time.time() - t0
            total_time += elapsed

            out_file = out_dir / f"{img_path.stem}.json"
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)

            det_count = len(res["detections"])
            print(f"Xong ({det_count} boxes, {elapsed:.2f}s)")
            success_count += 1
        except Exception as e:
            print(f"LỖI: {e}")

    avg_speed = (total_time / success_count) if success_count > 0 else 0.0
    print("\n--- KẾT QUẢ SUY LUẬN CHANDRA ---")
    print(f"Thành công: {success_count}/{len(all_images)} ảnh")
    print(f"Tổng thời gian: {total_time:.2f}s (Trung bình: {avg_speed:.2f}s/ảnh)")
    print(f"Kết quả lưu tại: {out_dir}")


if __name__ == "__main__":
    main()
