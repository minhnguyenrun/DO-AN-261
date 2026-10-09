# DO-AN-261 — So sánh các phương pháp Document Layout Analysis

Đồ án môn học: đánh giá và so sánh các phương pháp phân tích cấu trúc layout tài liệu
(Document Layout Analysis) trên các bộ dữ liệu chuẩn (DocLayNet, PubLayNet).

## Cấu trúc project

```
DOAN261/
├── datasets/               # dữ liệu (chạy local, KHÔNG push, trừ README.md/.gitkeep)
├── scripts/download_datasets/  # script tải dataset tự động
├── src/common/              # code dùng chung: io, label_mapping, metrics, vis_utils
├── methods/                 # mỗi phương pháp (docling, paddleocr_vl, ...) một folder con
├── outputs/                 # kết quả thô chạy local (KHÔNG push)
└── visualization/           # ghép PDF/ảnh + JSON thành hình trực quan để đưa vào báo cáo
```

Xem chi tiết nguồn và cách tải dữ liệu tại [datasets/README.md](datasets/README.md).

## Cách chạy

1. Tạo virtualenv và cài dependency của phương pháp muốn chạy (mỗi folder trong
   `methods/<method>/` có `requirements.txt` riêng):

   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   pip install -r scripts/download_datasets/requirements.txt
   pip install -r methods/<method>/requirements.txt
   ```

2. Tải dataset:

   ```bash
   python scripts/download_datasets/download_doclaynet.py --max-size-gb 1
   python scripts/download_datasets/download_publaynet.py --max-size-gb 1
   ```

3. Chạy phương pháp tương ứng trong `methods/<method>/src/` hoặc notebook trong
   `methods/<method>/notebooks/`. Kết quả JSON dự đoán được lưu vào
   `outputs/predictions/<method>/<dataset>/`.

4. Dùng `visualization/scripts/` để ghép ảnh/PDF gốc với JSON dự đoán thành hình
   trực quan, lưu vào `visualization/figures/` để đưa vào báo cáo.

## Phân công

| Thành viên | Phương pháp phụ trách |
|---|---|
| TBD | `methods/docling` |
| TBD | `methods/paddleocr_vl` |
