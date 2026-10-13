# PaddleOCR-VL

Phương pháp phân tích layout tài liệu dùng [PaddleOCR-VL](https://github.com/PaddlePaddle/PaddleOCR).

## Cấu trúc

- `src/` — vị trí dự kiến cho code inference, xuất dự đoán ra `outputs/predictions/paddleocr_vl/<dataset>/*.json`
- `notebooks/paddleocr_vl_report.ipynb` — notebook kết hợp inference, metric và trực quan hóa khi method có source/prediction
- `report/` — báo cáo PDF cho phương pháp này

## Cách chạy

```bash
pip install -r methods/paddleocr_vl/requirements.txt
python methods/paddleocr_vl/src/run_inference.py --dataset doclaynet
```

Notebook report hiện đã có, nhưng source `src/run_inference.py` chưa được triển
khai. Vì vậy các cell inference, metric và visualization sẽ báo rõ khi chưa có
prediction; notebook không sinh dữ liệu giả. Khi source được bổ sung, notebook
sẽ gọi entrypoint đó và dùng chung evaluator/visualizer của project.
