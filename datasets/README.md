# Datasets

Thư mục này chứa dữ liệu chạy **local only** — không push lên git (xem `.gitignore`
ở thư mục gốc), ngoại trừ các file `README.md` và `.gitkeep` để giữ cấu trúc.

## Cấu trúc

```
datasets/
├── doclaynet/
│   ├── raw/          # file .parquet tải thẳng từ HuggingFace (chưa xử lý)
│   └── processed/    # ảnh (.png) + annotation (COCO-style .json) đã chuyển định dạng
└── publaynet/
    ├── raw/
    └── processed/
```

## Nguồn dữ liệu

- **DocLayNet** — `docling-project/DocLayNet-v1.2` trên HuggingFace Hub
  (https://huggingface.co/datasets/docling-project/DocLayNet-v1.2). 11 nhãn layout
  (Caption, Footnote, Formula, List-item, Page-footer, Page-header, Picture,
  Section-header, Table, Text, Title), chia theo shard parquet `train/validation/test`.
- **PubLayNet** — `creative-graphic-design/PubLayNet` trên HuggingFace Hub
  (https://huggingface.co/datasets/creative-graphic-design/PubLayNet). Annotation
  theo chuẩn COCO, 5 nhãn layout (Text, Title, List, Table, Figure).

Do dữ liệu gốc rất lớn (DocLayNet ~40GB, PubLayNet ~uhi), script tải mặc định chỉ
lấy một phần giới hạn theo kích thước (mặc định 1GB) để đủ dùng cho việc phát triển
và test local.

## Cách tải

```bash
pip install -r scripts/download_datasets/requirements.txt

# tải ~1GB đầu tiên (mặc định), chia theo train/validation/test
python scripts/download_datasets/download_doclaynet.py --max-size-gb 1

# chỉ tải raw, không convert qua processed/
python scripts/download_datasets/download_doclaynet.py --max-size-gb 1 --no-extract

python scripts/download_datasets/download_publaynet.py --max-size-gb 1
```

Script sẽ:
1. Tải các shard `.parquet` từ HuggingFace Hub vào `raw/`, dừng lại khi đã đạt
   khoảng dung lượng chỉ định (`--max-size-gb`).
2. Trích ảnh + annotation từ các shard đã tải, lưu vào `processed/images/` và
   `processed/annotations/<split>.json` (COCO-style: `images`, `annotations`,
   `categories`).

## Thống kê dữ liệu đã tải

- **DocLayNet**: đã tải 2 shard `train` (~1.04GB raw parquet) từ
  `docling-project/DocLayNet-v1.2`, trích xuất được **1928 ảnh** (`.png`) và
  **24229 bounding box** trên 11 nhãn layout, lưu tại:
  - `datasets/doclaynet/raw/data/train-0000{0,1}-of-00072.parquet`
  - `datasets/doclaynet/processed/images/*.png`
  - `datasets/doclaynet/processed/annotations/train.json` (COCO-style)
- **PubLayNet**: chưa tải (dùng `download_publaynet.py` khi cần).
