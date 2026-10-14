# Trọng số mô hình dots.mocr

Mô hình [dots.mocr](https://huggingface.co/dots-studio/dots.mocr) (~3B tham số) phát triển bởi Dots Studio / RedNote HI Lab.

## Tải tự động (Khuyến nghị)
Khi chạy script `run_inference.py` hoặc notebook `dots_mocr_report.ipynb`, trọng số sẽ được tự động tải từ Hugging Face Hub nếu thư mục này chưa có trọng số:
```bash
python methods/dots_mocr/src/run_inference.py --dataset doclaynet --limit 10
```

## Tải thủ công từ Hugging Face Hub
Nếu muốn tải thủ công trước vào thư mục này:
```bash
pip install huggingface_hub
huggingface-cli download dots-studio/dots.mocr --local-dir methods/dots_mocr/weights
```

## Lưu ý Git
Thư mục `weights/` đã được cấu hình trong `.gitignore` để không commit các file trọng số nhị phân lớn (`*.safetensors`, `*.bin`, `.cache/`) lên kho mã nguồn Git.
