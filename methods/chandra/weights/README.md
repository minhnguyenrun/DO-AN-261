# Trọng số mô hình Chandra

Mô hình [Chandra OCR](https://huggingface.co/datalab-to/chandra) phát triển bởi **Datalab** (`datalab-to`).

Hỗ trợ các model ID sau từ Hugging Face Hub:
- `datalab-to/chandra` (Mô hình gốc 9B)
- `datalab-to/chandra-ocr-2` (Phiên bản nâng cấp 5B nhẹ hơn, tối ưu tốc độ và đa ngữ)

## Tải tự động (Khuyến nghị)
Khi chạy script `run_inference.py` hoặc notebook `chandra_report.ipynb`, trọng số sẽ được tự động tải từ Hugging Face Hub nếu thư mục này chưa có trọng số:
```bash
python methods/chandra/src/run_inference.py --dataset doclaynet --limit 10
```

## Tải thủ công từ Hugging Face Hub
Nếu muốn tải thủ công trước vào thư mục này:
```bash
pip install huggingface_hub
huggingface-cli download datalab-to/chandra-ocr-2 --local-dir methods/chandra/weights
```

## Lưu ý Git
Thư mục `weights/` đã được cấu hình trong `.gitignore` để không commit các file trọng số nhị phân lớn (`*.safetensors`, `*.bin`, `.cache/`) lên kho mã nguồn Git.
