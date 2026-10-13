# Docling

Phương pháp phân tích layout tài liệu dùng [Docling](https://github.com/docling-project/docling).

## Cấu trúc

- `src/` — code chạy inference, xuất dự đoán ra `outputs/predictions/docling/<dataset>/*.json`
- `notebooks/` — notebook thử nghiệm, phân tích kết quả
- `report/` — báo cáo PDF cho phương pháp này

## Cách chạy

```bash
pip install -r methods/docling/requirements.txt

# chay layout detection tren N anh dau tien cua datasets/doclaynet/processed/images
python methods/docling/src/run_inference.py --dataset doclaynet --limit 20

# ve hinh so sanh Ground Truth vs. du doan
python visualization/scripts/visualize_predictions.py --method docling --dataset doclaynet --limit 20
```

Kết quả JSON lưu tại `outputs/predictions/docling/doclaynet/<image_id>.json`
(mỗi ảnh một file, gồm `detections`: list `{category_id, category_name, bbox}`).
Hình so sánh lưu tại `visualization/figures/docling_doclaynet/<image_id>.png`.

## Ghi chú kỹ thuật

- Docling dùng model layout mặc định `docling-project/docling-layout-heron`
  (tải tự động từ HuggingFace Hub lần chạy đầu). Model này được huấn luyện
  trên chính DocLayNet nên nhãn đầu ra (`DocItemLabel`) khớp trực tiếp với
  11 nhãn của DocLayNet (xem mapping trong `src/run_inference.py`).
- Script chỉ bật stage layout, tắt OCR/table-structure/các enrichment khác
  để chạy nhanh (~1-1.5s/ảnh trên CPU sau khi model đã load).
- Vì tắt OCR nên các cluster text "rỗng" (không có nội dung) sẽ bị Docling
  loại bỏ theo mặc định — phải set `layout_options.keep_empty_clusters =
  True` để giữ lại đầy đủ box layout.

## Kết quả đánh giá (20 ảnh đầu, doclaynet/train, IoU >= 0.5)

Chạy bằng `python scripts/evaluate_predictions.py --method docling --dataset doclaynet --loose`.

| | Precision | Recall | F1 |
|---|---|---|---|
| Overall (so khớp cả vị trí lẫn nhãn) | 0.765 | 0.780 | 0.772 |
| Overall (chỉ so vị trí, bỏ qua nhãn) | 0.808 | 0.824 | 0.816 |

Per-category đáng chú ý: `Table` và `Picture` đạt recall 1.00 (20/20, 6/6).
`Text` khá tốt (recall 0.83). `Section-header` recall rất cao (0.98) nhưng
precision thấp (0.57) — Docling tách tên người/chức danh thành nhiều
`section_header` nhỏ trong khi Ground Truth gộp vào 1 khối `Text`.

**`Page-header` và `Page-footer` recall = 0.00 (0/7 và 0/17) — kể cả khi bỏ
qua nhãn (loose).** Nghĩa là Docling không vẽ bất kỳ box nào đè lên vùng này,
không phải lỗi gán nhãn sai. Đã xác nhận nguyên nhân bằng cách đọc source code
Docling (`layout_object_detection_model.py`) và thực nghiệm trực tiếp:

1. Stage `layout` chạy **trước** stage `ocr` trong pipeline (xem
   `standard_pdf_pipeline.py`), nên việc phát hiện box hoàn toàn dựa vào thị
   giác, KHÔNG dùng text OCR để quyết định có cluster ở đâu — việc tắt
   `do_ocr` không ảnh hưởng tới bước này.
2. Model layout mặc định (`docling-layout-heron`) là kiến trúc **RT-DETR**,
   luôn resize ảnh trang về đúng **640x640** trước khi detect (kiểm tra bằng
   `AutoImageProcessor.from_pretrained(...)`), bất kể ảnh gốc là 1025x1025
   hay bao nhiêu. Các box Page-header/Page-footer trong DocLayNet vốn đã rất
   mỏng (~5-20px trên ảnh gốc 1025px), sau khi scale xuống 640px chỉ còn
   ~3-13px, nằm sát mép ảnh.
3. Đã thử hạ `score_threshold` từ mặc định 0.3 xuống 0.05 (gần như chấp nhận
   mọi độ tin cậy) trên 3 ảnh có GT header/footer — vẫn ra 0 box trong vùng
   đó. Vậy đây **không phải do ngưỡng confidence lọc mất**, mà do RT-DETR
   (dùng số lượng object query cố định, học cách bắt các object "kích thước
   điển hình") không hề sinh ra đề xuất nào cho các dải mỏng dính sát biên
   ảnh — một điểm yếu kinh điển của họ model DETR với vật thể nhỏ ở rìa ảnh.

Hướng cải thiện có thể thử: dùng ảnh trang ở độ phân giải cao hơn kết hợp
crop riêng dải rìa trên/dưới trước khi đưa vào model, hoặc thử model layout
khác của Docling (`DOCLING_LAYOUT_EGRET_*`, độ chính xác cao hơn heron).
