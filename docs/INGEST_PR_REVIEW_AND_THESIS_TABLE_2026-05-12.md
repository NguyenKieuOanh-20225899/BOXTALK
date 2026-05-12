# Ingest PR Review And Thesis Result Table - 2026-05-12

## PR Review Summary

Branch: `feature/ingest-improvements`

Commit scope reviewed:

| Group | Files | Review result |
|---|---|---|
| Ingest/eval code | `app/ingest`, `app/eval`, `scripts/benchmark_ingest_suite.py` | OK after CI fix. Benchmark runner supports text/layout/table/OCR/Nougat adapters, direct component evaluation, summary JSON, per-sample JSONL, and README output. |
| Dataset preparation scripts | `scripts/prepare_funsd_ocr_subset.py`, `scripts/prepare_ocrd_pagexml_subset.py`, `scripts/prepare_nougat_arxiv_subset.py`, `scripts/prepare_doclaynet_small_subset.py`, `scripts/prepare_publaynet_wds_subset.py` | OK. Scripts prepare local subsets/manifests and avoid hardcoding one PDF. Large datasets are not downloaded automatically except requested subset files. |
| Documentation/report | `docs/INGEST_BENCHMARKS.md`, `docs/INGEST_REAL_BENCHMARK_RUN_2026-05-12.md` | OK. Documents explain dataset format, CLI, OCR GPU isolation, commands, metrics, and limitations. |

Blocking issue fixed:

- `scripts/test_pipeline.py` previously executed `ingest_pdf("data/sample_layout.pdf")` at import time, causing `pytest -q` to fail when that local PDF was missing.
- It is now a proper pytest smoke test that creates a temporary PDF under `tmp_path`.

Validation:

```powershell
.\.venv-gpu\Scripts\python.exe -m compileall app scripts
.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset mock --limit 5 --out results\ingest\mock_validation_after_ocr_gpu --mode all --save-predictions
.\.venv-gpu\Scripts\python.exe -m pytest -q
```

Result:

```text
30 passed
```

## Thesis Result Table

| Ingest component | Benchmark / dataset | Samples | Main metric | Result | Notes |
|---|---|---:|---|---:|---|
| Table extraction | PubTables-1M detection subset | 25 | Table F1@IoU 0.50 / 0.75 | 0.987 / 0.887 | Detection-only because available ground truth is table bbox. |
| General layout detection | DocLayNet subset | 25 | Layout micro F1@IoU 0.50 / 0.75 | 0.815 / 0.772 | Evaluates heading, paragraph, table, figure, caption, metadata-like labels. |
| Scientific PDF layout | PubLayNet subset | 25 | Layout micro F1@IoU 0.50 / 0.75 | 0.771 / 0.743 | Evaluates title, text, list, table, figure classes. |
| OCR / scan PDF | FUNSD OCR subset | 25 | OCR token F1 | 0.749 | FUNSD ground truth is word-level annotation, not exact OCR line transcription. |
| OCR / scan PDF | OCR-D PAGE-XML subset | 19 | OCR token F1 | 0.657 | Historical German/Latin/Fraktur-like pages; harder than modern scans. |
| Academic PDF text extraction proxy | Nougat/arXiv subset | 25 | Token F1 | 0.628 | Text extraction against markdown-like content; not full Nougat markup generation. |

## Suggested Thesis Wording

Hệ thống được đánh giá theo từng thành phần của pipeline ingest PDF thay vì chỉ đánh giá kết quả hỏi đáp cuối cùng. Với bảng, hệ thống đạt F1@IoU 0.50 là 0.987 trên subset PubTables-1M, cho thấy khả năng phát hiện vùng bảng ổn định. Với phân tích layout, hệ thống đạt micro F1@IoU 0.50 lần lượt là 0.815 trên DocLayNet và 0.771 trên PubLayNet, phản ánh hiệu quả của hướng xử lý theo vùng tài liệu. Với OCR, kết quả trên FUNSD và OCR-D thấp hơn do ground truth khác biệt về cách ghi nhận văn bản và độ khó của tài liệu scan lịch sử, nhưng benchmark đã chứng minh được pipeline có thể chạy OCR GPU ổn định trên dữ liệu ngoài. Với Nougat/arXiv, kết quả được xem như benchmark proxy cho trích xuất văn bản học thuật, chưa phải đánh giá đầy đủ bài toán chuyển PDF sang markup.
