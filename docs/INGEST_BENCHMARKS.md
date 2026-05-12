# Ingest Benchmarks

## Recent additions

- OCR GPU on Windows should run in a separate `.venv-ocr-gpu` process. This avoids CUDA/cuDNN DLL conflicts between `paddlepaddle-gpu` and PyTorch CUDA in `.venv-gpu`.
- External dataset helpers now include `scripts/prepare_funsd_ocr_subset.py`, `scripts/prepare_ocrd_pagexml_subset.py`, and `scripts/prepare_nougat_arxiv_subset.py`.
- Text-only benchmarks can call `app.ingest.extract.text.extract_with_text_backend()` directly. Image layout/table benchmarks can call the model layout detector directly.
- Long-document edit-distance metrics are bounded by `BOXBIIBOO_BENCHMARK_MAX_CHAR_EDIT_LENGTH` and `BOXBIIBOO_BENCHMARK_MAX_TOKEN_EDIT_LENGTH`; token F1 is the most stable text metric for long academic PDFs.

Tài liệu này mô tả framework benchmark ingest PDF thống nhất cho BOXTALK. Mục tiêu là đánh giá các thành phần từ PDF input đến `pages / blocks / chunks` trước khi đi vào retrieval và grounded QA.

## Thành phần và benchmark tham chiếu

| Thành phần ingest | Benchmark | Đánh giá |
|---|---|---|
| Text extraction | Bast & Korzen PDF Extraction Benchmark | text accuracy, token F1, normalized similarity, reading order |
| Layout detection | DocLayNet | heading, paragraph, table, figure, caption, header/footer |
| Scientific PDF layout | PubLayNet | title, text, list, table, figure |
| Table extraction | PubTables-1M | table detection, row/column/cell structure |
| OCR / scan PDF | Nougat/OCR-D/FUNSD | OCR text, scan PDF, academic markup/form |

## Unified CLI

Unified runner nằm trong:

```text
scripts/benchmark_ingest_suite.py
```

Khi truyền `--dataset`, script chạy benchmark adapter mới. Nếu không truyền `--dataset`, script vẫn chạy legacy production/scientific suite cũ.

Ví dụ:

```powershell
python scripts/benchmark_ingest_suite.py --dataset mock --limit 10 --out results/ingest/mock

python scripts/benchmark_ingest_suite.py --dataset pubtables --data-dir data/benchmarks/pubtables_detection --limit 100 --out results/ingest/pubtables_100 --mode table

python scripts/benchmark_ingest_suite.py --dataset doclaynet --data-dir data/benchmarks/doclaynet --limit 100 --out results/ingest/doclaynet_100 --mode layout

python scripts/benchmark_ingest_suite.py --dataset publaynet --data-dir data/benchmarks/publaynet --limit 100 --out results/ingest/publaynet_100 --mode layout

python scripts/benchmark_ingest_suite.py --dataset bastkorzen --data-dir data/benchmarks/text_extraction --limit 100 --out results/ingest/text_100 --mode text

python scripts/benchmark_ingest_suite.py --dataset ocr --data-dir data/benchmarks/ocr --limit 100 --out results/ingest/ocr_100 --mode ocr

python scripts/benchmark_ingest_suite.py --dataset nougat --data-dir data/benchmarks/nougat_arxiv_small/text --limit 25 --out results/ingest/nougat_arxiv_text_direct_25 --mode text
```

CLI chính:

| Argument | Ý nghĩa |
|---|---|
| `--dataset` | `mock`, `bastkorzen`, `doclaynet`, `publaynet`, `pubtables`, `ocr`, `nougat` |
| `--data-dir` | thư mục dataset local |
| `--limit` | giới hạn số sample, `0` là toàn bộ |
| `--out` | thư mục output |
| `--device` | hint device cho model layout, ví dụ `cpu` hoặc `cuda` |
| `--mode` | `text`, `layout`, `table`, `ocr`, `all` |
| `--save-predictions` | lưu prediction JSON từng sample |
| `--seed` | seed chọn sample |

## Output

Mỗi lần chạy sinh:

```text
summary.json
per_sample.jsonl
README.md
predictions/*.json   # nếu dùng --save-predictions
```

`summary.json` gồm:

- `dataset_name`
- `mode`
- `num_samples`
- `success_rate`
- `metric_summary`
- `latency.mean/p50/p95`
- `error_count`
- `config`
- `issues`

## Metric được hỗ trợ

### Text extraction

- `char_accuracy`
- `token_precision / token_recall / token_f1`
- `normalized_text_similarity`
- `reading_order_score`

### Layout detection

- IoU@0.50 và IoU@0.75
- micro precision/recall/F1
- macro F1
- per-label precision/recall/F1
- confusion summary theo label

### Scientific PDF layout

PubLayNet dùng cùng metric layout detection, nhưng label phổ biến là:

```text
title, text, list, table, figure
```

### Table extraction

- table detection precision/recall/F1 theo IoU
- `table_structure` cell-level precision/recall/F1 nếu có ground-truth cell
- `table_exact_csv` hoặc `table_exact_html` nếu có ground truth dạng CSV/HTML
- detection-only mode nếu chỉ có bbox bảng

### OCR / scan PDF

- `cer`
- `wer`
- token F1
- optional `form_field_f1` nếu ground truth có form fields

## Dataset format

### Mock

Không cần dataset. Runner tự tạo PDF synthetic trong output.

### PubTables

Đặt dữ liệu theo một trong các cấu trúc:

```text
data/benchmarks/pubtables_detection/extracted/images/test/*.jpg
data/benchmarks/pubtables_detection/extracted/annotations/test/*.xml
```

hoặc:

```text
data/benchmarks/pubtables_detection/images/test
data/benchmarks/pubtables_detection/annotations/test
```

Adapter đọc Pascal VOC XML và chạy table detection mode.

### DocLayNet

Đặt COCO-style annotations và images, ví dụ:

```text
data/benchmarks/doclaynet/annotations/test.json
data/benchmarks/doclaynet/images/test
```

hoặc cấu trúc tương đương có `COCO/test.json`.

### PubLayNet

Tương tự DocLayNet, dùng COCO-style annotations:

```text
data/benchmarks/publaynet/annotations/test.json
data/benchmarks/publaynet/images/test
```

### BastKorzen text extraction

Do benchmark gốc có nhiều biến thể đóng gói, adapter local dùng JSONL:

```text
data/benchmarks/text_extraction/bastkorzen_samples.jsonl
```

Mỗi dòng:

```json
{
  "doc_id": "sample_001",
  "pdf_path": "sample_001.pdf",
  "ground_truth": {
    "text": "full expected text",
    "ordered_text": ["block 1", "block 2"]
  }
}
```

### OCR / OCR-D / FUNSD / Nougat-style local subset

Adapter OCR cũng dùng JSONL:

```text
data/benchmarks/ocr/ocr_samples.jsonl
```

Mỗi dòng:

```json
{
  "doc_id": "scan_001",
  "image_path": "scan_001.png",
  "ground_truth": {
    "text": "expected OCR text",
    "form_fields": {
      "question": "answer"
    }
  }
}
```

Nếu chỉ có `text`, benchmark báo OCR text metrics. Nếu có `form_fields`, benchmark có thêm form field F1 placeholder.

## Pipeline được gọi

Unified benchmark gọi pipeline ingest thật:

```text
app.ingest.pipeline.ingest_pdf()
```

Nếu sample là image, runner tạm chuyển image thành một PDF một trang rồi gọi `ingest_pdf()`. Prediction được quy đổi về schema chung:

```json
{
  "text": "...",
  "ordered_text": ["..."],
  "layout_regions": [{"label": "...", "bbox": [...]}],
  "table_regions": [{"label": "table", "bbox": [...]}],
  "table_cells": [{"row": 0, "col": 0, "text": "..."}],
  "backend": "region_routed"
}
```

## Lưu ý

- Framework này không tải dataset lớn tự động.
- Có thể chạy sample nhỏ bằng `--limit`.
- Nếu dataset thiếu, runner ghi rõ `issues` trong `summary.json` và `README.md`.
- Các benchmark retrieval/QA hiện có không bị thay đổi.

## PubTables Structure / OTSL Add-on

PubTables detection XML chi danh gia bbox bang. De danh gia row/column/cell structure, dung subset PubTables-1M OTSL co cell bbox va HTML ground truth:

```powershell
python scripts/prepare_pubtables_structure_subset.py --limit 25 --out data/benchmarks/pubtables_structure
python scripts/benchmark_ingest_suite.py --dataset pubtables_structure --data-dir data/benchmarks/pubtables_structure --limit 25 --out results/ingest/pubtables_structure_25 --mode table --save-predictions
```

Metric bo sung:

- `table_cell_iou50` va `table_cell_iou75`: do cell bbox detection.
- `table_structure`: do khop row/col/text cua cell.
- `text_assignment_f1`, `row_count_error`, `col_count_error`, `empty_cell_rate`: debug loi structure.
- `table_exact_csv` va `table_exact_html`: do khop output co cau truc neu ground truth co san.

Khi dung `--save-predictions`, runner ghi them debug JSON vao `table_debug/<doc_id>.json`, gom predicted cells, ground-truth cells, matched cells va unmatched cells.

OCR-D hoac tai lieu lich su co them cac metric `ocr_historical_*`. Cac metric nay chuan hoa long-s, ligature va mot so ky tu co de do noi dung doc duoc; raw `ocr_*` metrics van duoc giu nguyen.
