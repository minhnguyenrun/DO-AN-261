# Docling

Phương pháp phân tích layout tài liệu dùng [Docling](https://github.com/docling-project/docling).

## Cấu trúc

- `src/` — code chạy inference, xuất dự đoán ra `outputs/predictions/docling/<dataset>/*.json`
- `notebooks/` — notebook thử nghiệm, phân tích kết quả
- `report/` — báo cáo PDF cho phương pháp này

## Cách chạy

```bash
pip install -r methods/docling/requirements.txt
python methods/docling/src/run_inference.py --dataset doclaynet
```
