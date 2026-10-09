# PaddleOCR-VL

Phương pháp phân tích layout tài liệu dùng [PaddleOCR-VL](https://github.com/PaddlePaddle/PaddleOCR).

## Cấu trúc

- `src/` — code chạy inference, xuất dự đoán ra `outputs/predictions/paddleocr_vl/<dataset>/*.json`
- `notebooks/` — notebook thử nghiệm, phân tích kết quả
- `report/` — báo cáo PDF cho phương pháp này

## Cách chạy

```bash
pip install -r methods/paddleocr_vl/requirements.txt
python methods/paddleocr_vl/src/run_inference.py --dataset doclaynet
```
