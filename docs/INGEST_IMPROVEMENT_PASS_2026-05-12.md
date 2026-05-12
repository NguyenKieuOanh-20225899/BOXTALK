# Ingest Improvement Pass - 2026-05-12

Muc tieu cua lan cai tien nay la xu ly diem yeu tiep theo cua ingest PDF sau khi da co benchmark framework: cau truc bang, thu tu doc PDF nhieu cot va OCR scan.

## Thay doi da implement

| Nhom | Thay doi | File chinh |
|---|---|---|
| Table structure extraction | Trich xuat `table_rows`, `table_records`, `table_cells`, `table_csv`, `table_html`, bbox cua hang, cot va tung cell khi co du word geometry tu PDF. | `app/ingest/extract/table.py` |
| Table region trong model layout | Vung `table` do layout model detect duoc se goi table extractor thay vi chi tao placeholder. | `app/ingest/extract/model_layout.py` |
| Reading order | Them sorter dung chung, co heuristic nhan dien PDF hai cot va sort theo thu tu doc tot hon `(y, x)`. | `app/ingest/reading_order.py` |
| Text/layout/region/OCR | Dung chung reading-order sorter trong text extraction, region routing, model layout va OCR. | `app/ingest/extract/text.py`, `app/ingest/extract/region_routed.py`, `app/ingest/extract/ocr.py` |
| Table QA metadata | QA fallback uu tien `table_records` khi normalize bang, giup lookup theo header on dinh hon. | `app/qa/llm_fallback.py` |
| OCR bbox | Chuyen bbox OCR tu toa do anh render ve toa do trang PDF truoc khi sort. | `app/ingest/extract/ocr.py` |
| OCR preprocessing | Them preprocessing anh opt-in qua `BOXBIIBOO_OCR_PREPROCESS=auto|contrast|binarize`; default la `none` vi khong cai thien deu tren FUNSD/OCR-D. | `app/ingest/extract/ocr.py` |

## Validation

Commands da chay:

```powershell
.\.venv-gpu\Scripts\python.exe -m compileall app scripts
.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset mock --limit 5 --out results\ingest\mock_after_table_records --mode all --save-predictions
.\.venv-gpu\Scripts\python.exe scripts\create_ingest_layout_benchmark.py --out data\ingest_layout_benchmark
.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_layout_quality.py --manifest data\ingest_layout_benchmark\manifest.json --output-dir results\ingest_layout_quality\table_structure_records_final
.\.venv-gpu\Scripts\python.exe -m pytest -q
```

Ket qua:

| Run | Samples | Metric | Result |
|---|---:|---|---:|
| `mock_after_table_records` | 5 | success rate | 1.000 |
| `mock_after_table_records` | 5 | table structure F1 | 1.000 |
| `mock_after_table_records` | 5 | table exact CSV | 1.000 |
| `table_structure_records_final` | 3 | quality score mean | 1.000 |
| `table_structure_records_final` | 3 | reading order score mean | 1.000 |
| `table_structure_records_final` | 3 | table cell coverage mean | 1.000 |
| `pytest -q` | 33 tests | pass rate | 1.000 |

## Expanded Real Benchmarks

| Dataset | Run | Samples | Success | Key metric | Latency mean / p50 |
|---|---|---:|---:|---|---:|
| PubTables-1M | `pubtables_real_cuda_100_after_structure` | 100 | 1.00 | table F1@0.50 = 0.960, F1@0.75 = 0.870 | 0.598s / 0.409s |
| DocLayNet-small | `doclaynet_real_cuda_49_after_structure` | 49 | 1.00 | layout micro F1@0.50 = 0.849, F1@0.75 = 0.807 | 0.627s / 0.345s |
| PubLayNet subset | `publaynet_real_cuda_100_after_structure` | 100 | 1.00 | layout micro F1@0.50 = 0.739, F1@0.75 = 0.708 | 0.521s / 0.388s |
| Synthetic OCR scan | `ocr_scan_gpu_25_after_ocr_sort` | 25 | 1.00 | OCR token F1 = 1.000, CER = 0.000, WER = 0.000 | 0.680s / 0.343s |
| FUNSD OCR | `funsd_ocr_gpu_25_after_ocr_sort` | 25 | 1.00 | OCR token F1 = 0.734, CER = 0.714, WER = 0.961 | 0.938s / 0.719s |
| OCR-D PAGE-XML | `ocrd_pagexml_gpu_19_after_ocr_sort` | 19 | 1.00 | OCR token F1 = 0.657, CER = 0.612, WER = 0.885 | 4.321s / 4.457s |

## Interpretation

PubTables trong local subset hien tai van la detection-only vi ground truth dang co la bbox XML, chua co cell/row/column ground truth. Phan table structure moi duoc validate bang synthetic benchmark va test PDF tao tai runtime, dong thoi da duoc noi vao pipeline ingest that de phuc vu QA ve sau.

DocLayNet-small chi co 49 anh hop le sau khi prepare, nen run 100 mau thuc te dung 49 mau. PubLayNet da duoc mo rong len 100 mau.

OCR preprocessing khong duoc bat mac dinh. Khi bat `auto`, no giu synthetic OCR o 1.000 nhung lam giam token F1 tren FUNSD/OCR-D. Vi vay pipeline chinh giu `BOXBIIBOO_OCR_PREPROCESS=none`, con preprocessing chi dung de thu nghiem voi anh scan kem chat luong.

## Suggested Thesis Update

Bang trong bao cao nen ghi ro hai lop ket qua:

| Thanh phan | Ket qua nen bao cao | Luu y |
|---|---|---|
| Table detection | PubTables-1M 100 mau: F1@0.50 = 0.960, F1@0.75 = 0.870 | Detection-only theo bbox. |
| Table structure | Synthetic structure benchmark: table cell coverage = 1.000 | Framework da xuat CSV/HTML/cell bbox, nhung can PubTables structure GT de danh gia chuan hon. |
| General layout | DocLayNet-small 49 mau: micro F1@0.50 = 0.849 | Local subset chi co 49 mau hop le. |
| Scientific layout | PubLayNet 100 mau: micro F1@0.50 = 0.739 | Danh gia title/text/list/table/figure. |
| OCR scan | FUNSD token F1 = 0.734; OCR-D token F1 = 0.657 | OCR-D CER/WER cai thien sau khi sua bbox/sort; FUNSD token F1 giam nhe so voi run cu. |
