# Báo cáo benchmark cải tiến ingest

Ngày chạy: 2026-05-12

## 1. Mục tiêu

Báo cáo này so sánh chất lượng ingest trước và sau khi cải tiến pipeline xử lý PDF. Trọng tâm là các thay đổi liên quan đến:

- `region_routed` backend làm main ingest path.
- Nhận diện và giữ block type tốt hơn: `heading`, `list_item`, `table`, `caption`, `metadata`, `figure`.
- Trích xuất bảng theo hướng có cấu trúc: hàng, cột, header, cell.
- Metric đánh giá theo tinh thần text extraction benchmark: text fidelity, reading order, noise removal, chunk preservation và table structure.

## 2. Thiết lập so sánh

Hai phiên bản được chạy trên cùng benchmark synthetic `ingest_layout_quality`.

| Phiên bản | Code | Backend chính | Output |
|---|---|---|---|
| Baseline trước cải tiến | commit `0607655` | `text` | `C:\Users\admin\Documents\GitHub\BOXTALK_ingest_baseline_0607655\results\ingest_layout_quality\baseline_0607655` |
| Sau cải tiến | worktree hiện tại nhánh `feature/ingest-improvements` | `region_routed` | `results/ingest_layout_quality/current_region_routed` |

Lệnh chạy bản hiện tại:

```powershell
.\.venv-gpu\Scripts\python.exe scripts/create_ingest_layout_benchmark.py
.\.venv-gpu\Scripts\python.exe scripts/benchmark_ingest_layout_quality.py `
  --manifest data/ingest_layout_benchmark/manifest.json `
  --output-dir results/ingest_layout_quality/current_region_routed
```

Baseline được chạy trong worktree riêng tại commit `0607655`, dùng cùng benchmark script và cùng loại dữ liệu synthetic.

## 3. Bộ benchmark

Benchmark synthetic gồm 3 PDF:

| Tài liệu | Nội dung kiểm tra |
|---|---|
| `mixed_policy` | Text thường, heading, metadata, list, bảng, figure, caption |
| `legal_policy` | Heading pháp quy, metadata, list item, paragraph |
| `grid_table` | Bảng dạng grid line, caption, heading |

Các metric chính:

| Metric | Ý nghĩa |
|---|---|
| `substring_coverage` | Nội dung quan trọng có được trích xuất không |
| `edit_similarity` | Độ giống giữa text output và expected full text |
| `reading_order_score` | Thứ tự đọc của các đoạn có đúng không |
| `noise_score` | Header/footer/noise không mong muốn có bị lọt vào không |
| `chunk_preservation_score` | Chunking có giữ đủ chunk và table chunk không |
| `table_cell_coverage` | Các cell kỳ vọng trong bảng có còn tìm được không |
| `table_shape_score` | Cấu trúc bảng có đúng số hàng/cột/header không |
| `macro_expected_type_recall` | Recall của các block type kỳ vọng |
| `probe_match_rate` | Probe mode có đúng kỳ vọng không |
| `latency_mean_sec` | Thời gian ingest trung bình |

## 4. Kết quả tổng hợp

| Metric | Baseline | Sau cải tiến | Delta |
|---|---:|---:|---:|
| `quality_score_mean` | 0.736 | 1.000 | +0.264 |
| `substring_coverage_mean` | 1.000 | 1.000 | +0.000 |
| `edit_similarity_mean` | 0.997 | 1.000 | +0.003 |
| `reading_order_score_mean` | 1.000 | 1.000 | +0.000 |
| `noise_score_mean` | 1.000 | 1.000 | +0.000 |
| `chunk_preservation_score_mean` | 0.667 | 1.000 | +0.333 |
| `table_cell_coverage_mean` | 0.333 | 1.000 | +0.667 |
| `table_shape_score_mean` | 0.333 | 1.000 | +0.667 |
| `macro_expected_type_recall_mean` | 0.556 | 1.000 | +0.444 |
| `probe_match_rate` | 1.000 | 1.000 | +0.000 |
| `latency_mean_sec` | 0.0068s | 0.0093s | +0.0025s |

## 5. Kết quả theo tài liệu

### Baseline trước cải tiến

| Document | Quality | Text | Edit | Order | Noise | Chunks | Table cells | Table shape | Type recall | Backend | Missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `mixed_policy` | 0.603 | 1.000 | 0.992 | 1.000 | 1.000 | 0.500 | 0.000 | 0.000 | 0.333 | `text` | 9 |
| `legal_policy` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | `text` | 0 |
| `grid_table` | 0.604 | 1.000 | 1.000 | 1.000 | 1.000 | 0.500 | 0.000 | 0.000 | 0.333 | `text` | 9 |

### Sau cải tiến

| Document | Quality | Text | Edit | Order | Noise | Chunks | Table cells | Table shape | Type recall | Backend | Missing |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `mixed_policy` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | `region_routed` | 0 |
| `legal_policy` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | `region_routed` | 0 |
| `grid_table` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | `region_routed` | 0 |

## 6. Nhận xét

Bản cải tiến tốt hơn rõ nhất ở các nhóm metric liên quan đến bảng và cấu trúc:

- `table_cell_coverage_mean` tăng từ `0.333` lên `1.000`.
- `table_shape_score_mean` tăng từ `0.333` lên `1.000`.
- `macro_expected_type_recall_mean` tăng từ `0.556` lên `1.000`.
- `chunk_preservation_score_mean` tăng từ `0.667` lên `1.000`.

Điều này cho thấy baseline cũ vẫn đọc được text tổng quát, nhưng không giữ tốt cấu trúc bảng và block type. Sau cải tiến, bảng được giữ thành `table` block/table chunk và có metadata hàng/cột/cell/header.

Latency tăng nhẹ từ khoảng `0.0068s` lên `0.0093s` trên bộ synthetic. Mức tăng này nhỏ trong benchmark hiện tại, đổi lại chất lượng cấu trúc tăng rõ rệt.

## 7. Giới hạn

Benchmark này là synthetic benchmark nhỏ, phù hợp để kiểm tra nhanh regression cho các thay đổi ingest rule-based. Nó chưa thay thế các benchmark lớn hơn như:

- `benchmark_ingest_standard.py` trên `data/test_probe`.
- `benchmark_ingest_scientific.py` trên DocLayNet/PubTables.

Trong lần chạy này, `data/test_probe/labels.json` và DocLayNet không có sẵn trong workspace. PubTables có thư mục dữ liệu, nhưng chưa chạy lại vì benchmark đó nặng hơn và chủ yếu đo layout/table detection dataset-level, không trực tiếp đo các metric text/order/chunk mới.

## 8. Kết luận

Trên benchmark synthetic hiện tại, bản cải tiến ingest đạt chất lượng tốt hơn baseline cũ. Cải thiện chính nằm ở khả năng giữ cấu trúc tài liệu, đặc biệt là bảng, table chunk và block type. Đây là bước tiến phù hợp với mục tiêu biến PDF từ text theo tọa độ thành evidence có cấu trúc cho retrieval và grounded QA.
