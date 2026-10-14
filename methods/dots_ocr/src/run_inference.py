"""Inference CLI for dots.ocr (1.7B Multilingual Vision-Language Model).

Paper & Model by RedNote HI Lab / Dots Studio:
"dots.ocr: A Multimodal Vision-Language Model for Document Layout Analysis and OCR"
Hugging Face: dots-studio/dots.ocr (or rednote-hilab/dots.ocr)

Scans document images from datasets/<dataset>/processed/images and produces
standard prediction JSON files in outputs/predictions/dots_ocr/<dataset>/<image_id>.json.

Usage:
    python methods/dots_ocr/src/run_inference.py --dataset doclaynet --limit 20
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
from PIL import Image

# Ensure local src directory is on sys.path
SRC_DIR = Path(__file__).resolve().parent
METHOD_DIR = SRC_DIR.parent
ROOT_DIR = METHOD_DIR.parents[1]

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

from parser import parse_dots_layout_response

DEFAULT_MODEL_ID = "dots-studio/dots.ocr"

PROMPT_LAYOUT_ONLY = (
    "Please output the layout information from this PDF image, including each layout's bbox and its category. "
    "The bbox should be in the format [x1, y1, x2, y2]. The layout categories for the PDF document include "
    "['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', "
    "'Table', 'Text', 'Title']. Do not output the corresponding text. The layout result should be in JSON format."
)


def ensure_model_weights(weights_path: Path, model_id: str = DEFAULT_MODEL_ID) -> str:
    has_safetensors = any(weights_path.glob("*.safetensors")) if weights_path.is_dir() else False
    if weights_path.is_dir() and (weights_path / "config.json").exists() and has_safetensors:
        print(f"[dots.ocr] Sử dụng trọng số có sẵn tại: {weights_path}")
        return str(weights_path)

    print(f"[dots.ocr] Chưa tìm thấy trọng số đầy đủ tại: {weights_path}")
    print(f"[dots.ocr] Đang chuẩn bị tải từ Hugging Face Hub: {model_id}...")
    try:
        from huggingface_hub import snapshot_download
        weights_path.mkdir(parents=True, exist_ok=True)
        local_dir = snapshot_download(
            repo_id=model_id,
            local_dir=str(weights_path),
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
        )
        print(f"[dots.ocr] Tải model thành công về: {local_dir}")
        return str(local_dir)
    except Exception as e:
        print(f"[dots.ocr] Cảnh báo khi tải local snapshot: {e}. Thử nạp trực tiếp qua model ID.")
        return model_id


def load_dots_ocr_model(model_path_or_id: str, device: str = "cpu"):
    """Loads dots.ocr AutoModel and AutoProcessor."""
    from transformers import AutoModelForCausalLM, AutoProcessor

    print(f"[dots.ocr] Khởi tạo mô hình từ: {model_path_or_id} (device={device})")
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() or hasattr(torch, "bfloat16") else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path_or_id,
        torch_dtype=torch_dtype,
        device_map=device if device != "cpu" else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if device != "cpu":
        try:
            model.to(device)
        except Exception:
            pass
    else:
        model.to("cpu")
    model.eval()

    processor = AutoProcessor.from_pretrained(model_path_or_id, trust_remote_code=True)
    return model, processor


def run_inference_single(
    model,
    processor,
    image_path: Path,
    device: str = "cpu",
    max_new_tokens: int = 768,
) -> Dict:
    """Runs layout extraction on a single image and returns standard detection format."""
    pil_img = Image.open(image_path).convert("RGB")
    width, height = pil_img.size

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": PROMPT_LAYOUT_ONLY},
            ],
        }
    ]

    try:
        from qwen_vl_utils import process_vision_info
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
    except Exception:
        # Direct processor call fallback
        inputs = processor(
            text=PROMPT_LAYOUT_ONLY,
            images=pil_img,
            return_tensors="pt",
        )

    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    inputs.pop("mm_token_type_ids", None)

    eos_token_id = [151643, 151645, 151673]
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            pad_token_id=151643,
            do_sample=False,
        )

    # Decode response tokens
    in_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
    out_ids = generated_ids[0][in_len:]
    output_text = processor.decode(out_ids, skip_special_tokens=True).strip()

    # Parse JSON detections into standard format
    detections = parse_dots_layout_response(output_text, img_width=width, img_height=height)

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
        help="Inference device: 'cuda' or 'cpu'",
    )

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

    repo_root = ROOT_DIR
    images_dir = repo_root / "datasets" / args.dataset / "processed" / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else repo_root / "outputs" / "predictions" / "dots_ocr" / args.dataset
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect target images
    image_paths = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpg"))
    if args.image_ids:
        selected_ids = {s.strip() for s in args.image_ids.split(",") if s.strip()}
        image_paths = [p for p in image_paths if p.stem in selected_ids]

    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    print(f"Dataset: {args.dataset}")
    print(f"Total target images: {len(image_paths)}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")

    # Resolve weights and load model
    resolved_weights = ensure_model_weights(Path(args.weights))
    model, processor = load_dots_ocr_model(resolved_weights, device=args.device)
    print("Model dots.ocr ready for inference!")

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
                processor=processor,
                image_path=img_path,
                device=args.device,
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
            import traceback
            traceback.print_exc()
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
