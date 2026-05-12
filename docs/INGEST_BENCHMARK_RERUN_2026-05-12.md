# Báo cáo chạy lại benchmark ingest

Ngày chạy: 2026-05-12

Nhánh: `feature/ingest-improvements`

Mục tiêu: chạy lại các benchmark ingest khả dụng trong workspace sau các cải tiến `region_routed`, table structure extraction và benchmark metric mới.

## 1. Tình trạng dữ liệu benchmark

| Benchmark | Dataset cần | Trạng thái |
|---|---|---|
| `benchmark_ingest_layout_quality.py` | `data/ingest_layout_benchmark` | Có, đã tạo lại và chạy |
| `benchmark_ingest_standard.py` | `data/test_probe/labels.json` | Thiếu, không chạy được |
| `benchmark_ingest_scientific.py` / DocLayNet | `data/benchmarks/doclaynet` | Thiếu, đã skip |
| `benchmark_ingest_scientific.py` / PubTables | `data/benchmarks/pubtables_detection` | Có, đã chạy sample `25` |

## 2. Ingest layout quality benchmark

Lệnh chạy:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_layout_quality.py `
  --create-dataset `
  --output-dir results/ingest_layout_quality/rerun_all_current
```

Output:

```text
results/ingest_layout_quality/rerun_all_current
```

Kết quả:

| Metric | Giá trị |
|---|---:|
| Documents | 3 |
| Success rate | 1.000 |
| Quality score mean | 1.000 |
| Substring coverage mean | 1.000 |
| Edit similarity mean | 1.000 |
| Reading order score mean | 1.000 |
| Noise score mean | 1.000 |
| Chunk preservation mean | 1.000 |
| Table cell coverage mean | 1.000 |
| Table shape score mean | 1.000 |
| Block type recall mean | 1.000 |
| Probe match rate | 1.000 |
| Mean latency | 0.008s |
| Backend counts | `{"region_routed": 3}` |

Kết luận: benchmark synthetic trực tiếp cho các cải tiến ingest hiện đạt toàn bộ metric ở mức `1.000`. Backend chính được dùng là `region_routed`.

## 3. Standard ingest benchmark

Lệnh đã thử:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_standard.py `
  --profiles baseline model_routed_doclaynet `
  --repeats 1 `
  --warmup-per-label 0 `
  --max-per-label 0 `
  --output-dir results/ingest_benchmark/rerun_all_standard
```

Kết quả: không chạy được vì thiếu file nhãn:

```text
FileNotFoundError: data\test_probe\labels.json
```

Kết luận: cần khôi phục hoặc tạo lại `data/test_probe/labels.json` và các PDF tương ứng trước khi chạy benchmark này.

## 4. Scientific ingest benchmark: PubTables sample 25

Lệnh chạy:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_scientific.py `
  --skip-doclaynet `
  --pubtables-root data/benchmarks/pubtables_detection `
  --pubtables-split test `
  --pubtables-limit 25 `
  --profiles baseline model_routed_doclaynet `
  --output-dir results/ingest_benchmark_scientific/rerun_all_pubtables25
```

Output:

```text
results/ingest_benchmark_scientific/rerun_all_pubtables25
```

### 4.1. Profile `baseline`

| Metric | Giá trị |
|---|---:|
| Images total | 25 |
| Images success | 0 |
| Images failed | 25 |
| Success rate | 0.000 |
| PubTables F1@0.50 | 0.000 |
| PubTables F1@0.75 | 0.000 |
| Non-empty predicted table rate | 0.000 |

Lỗi chính:

```text
region_routed returned weak result
ocr failed: PaddleOCR is not installed: partially initialized module 'paddle' has no attribute 'tensor'
text returned weak result
```

Diễn giải: PubTables là ảnh bảng, nên baseline cần OCR hoặc layout model. Trong môi trường hiện tại, OCR/PaddleOCR baseline lỗi, vì vậy baseline fail toàn bộ sample.

### 4.2. Profile `model_routed_doclaynet`

| Metric | Giá trị |
|---|---:|
| Images total | 25 |
| Images success | 25 |
| Images failed | 0 |
| Success rate | 1.000 |
| Backend counts | `{"model_layout": 25}` |
| Route counts | `{"layout": 126}` |
| PubTables micro F1@0.50 | 1.000 |
| PubTables micro F1@0.75 | 0.818 |
| Mean latency | 10.127s |
| Median latency | 7.624s |
| P95 latency | 29.106s |
| Non-empty predicted table rate | 0.000 |

Kết luận: profile model layout chạy được trên sample PubTables 25 và đạt detection tốt: F1@0.50 đạt `1.000`, F1@0.75 đạt `0.818`. Tuy nhiên `pred_table_nonempty_rate` vẫn `0.000`, nghĩa là benchmark này đang xác nhận tốt phần phát hiện vùng bảng, chưa xác nhận tốt phần nội dung bảng trên ảnh.

## 5. Tổng kết

| Nhóm benchmark | Trạng thái | Kết luận |
|---|---|---|
| Layout quality synthetic | Đã chạy | Cải tiến hiện tại đạt `1.000` trên toàn bộ metric text/order/chunk/table/type |
| Standard ingest | Blocked | Thiếu `data/test_probe/labels.json` |
| PubTables sample 25 | Đã chạy | `model_routed_doclaynet` đạt F1@0.50 `1.000`, F1@0.75 `0.818`; baseline fail do OCR/Paddle |
| DocLayNet | Blocked | Thiếu `data/benchmarks/doclaynet` |

## 6. Việc nên làm tiếp

1. Khôi phục `data/test_probe` để chạy lại `benchmark_ingest_standard.py`.
2. Cài/sửa PaddleOCR trong `.venv-gpu` nếu muốn baseline OCR chạy được trên PubTables/image-only PDFs.
3. Nếu cần benchmark đầy đủ PubTables, chạy lại với `--pubtables-limit 100`, `500` hoặc `0`; lưu ý full test có hơn 100k file và sẽ mất nhiều thời gian.
4. Nếu muốn đo nội dung bảng trên ảnh, cần bổ sung OCR/table-structure path cho `model_layout` hoặc dùng `model_routed` với OCR/table extraction nội dung ổn định hơn.
