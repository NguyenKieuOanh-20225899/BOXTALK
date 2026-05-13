# Table Structure Row/Column Fix - 2026-05-12

Muc tieu cua pass nay la cai thien deterministic post-processing cho PubTables-1M OTSL table structure, tap trung vao row grouping, column grouping, cell matching va debug visibility. Khong train model moi, khong dung LLM, khong hardcode theo sample.

## Before / After

| Metric | Before structure fix 25 | Previous structure pass 25 | After row/column fix 25 |
|---|---:|---:|---:|
| table detection F1@0.50 | 0.927 | 0.967 | 0.967 |
| table cell IoU@0.50 F1 | 0.521 | 0.668 | 0.659 |
| table cell IoU@0.75 F1 | 0.163 | 0.185 | 0.183 |
| table text cell structure F1 | 0.158 | 0.169 | 0.202 |
| text assignment F1 | n/a | 0.963 | 0.963 |
| row count MAE | n/a | 2.240 | 2.040 |
| col count MAE | n/a | 0.840 | 0.840 |
| exact CSV | 0.000 | 0.000 | 0.000 |
| exact HTML | 0.000 | 0.000 | 0.000 |

Detection khong giam. Table structure F1 tang tu 0.169 len 0.202 va row count MAE giam tu 2.24 len 2.04. Cell IoU@0.50 giam nhe so voi previous structure pass, nhung van cao hon baseline 25 mau truoc khi sua post-processing.

## Error Summary After Fix

Run debug:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\analyze_pubtables_structure_debug.py --run-dir results\ingest\pubtables_structure_25_after_rowcol_fix --data-dir data\benchmarks\pubtables_structure --out results\ingest\pubtables_structure_debug --limit-visualizations 10
```

Summary:

| Error type | Count / 25 |
|---|---:|
| Row over-segmentation | 14 |
| Row under-segmentation | 5 |
| Row exact | 6 |
| Column over-segmentation | 4 |
| Column under-segmentation | 9 |
| Column exact | 12 |

Worst samples after fix:

| doc_id | cell F1@0.50 | structure F1 | gt rows | pred rows | gt cols | pred cols |
|---|---:|---:|---:|---:|---:|---:|
| PMC5146761_table_2 | 0.017 | 0.033 | 15 | 18 | 5 | 3 |
| PMC5675408_table_2 | 0.074 | 0.000 | 2 | 10 | 3 | 2 |
| PMC3464982_table_2 | 0.085 | 0.122 | 28 | 25 | 3 | 4 |
| PMC5995222_table_3 | 0.146 | 0.000 | 7 | 6 | 10 | 5 |
| PMC4424527_table_4 | 0.167 | 0.083 | 8 | 8 | 4 | 2 |

Debug images are saved in:

```text
results/ingest/pubtables_structure_debug/visualizations
```

Legend: yellow = table bbox, green = ground-truth cells, red = predicted cells, orange = predicted row bands, cyan = predicted column bands.

## Implemented Changes

- Row grouping now compares each OCR/text box with all current row candidates, not only the latest row.
- Same-row decision uses adaptive signals:
  - vertical overlap ratio above 0.35, or
  - y-center distance below 0.5 * median text box height.
- A single leading caption-like row such as `Table 4...` is trimmed when it is wide edge noise before the actual grid.
- Metrics now expose:
  - `row_count_mae`
  - `col_count_mae`
  - `row_oversegmentation_count`
  - `row_undersegmentation_count`
  - `col_oversegmentation_count`
  - `col_undersegmentation_count`
- Added `scripts/analyze_pubtables_structure_debug.py` for per-sample error reports and optional visualization.

## Remaining Weaknesses

- Column under-segmentation is still common on wide tables with sparse text, empty columns or compact numeric columns.
- Merged cells are only handled heuristically through spans. PubTables OTSL exact HTML requires precise row/colspan and markup, so exact HTML/CSV remains 0.
- OCR text is good enough for matched cells, but character differences and missing boxes still break exact structure matching.
- Some samples have correct table detection but weak structure because the OCR cluster includes captions/notes or misses internal cells.

## Thesis Interpretation

Nen dua ket qua nay vao bao cao do an theo huong minh bach:

- He thong dat ket qua tot o table detection.
- Post-processing da cai thien cell bbox va row grouping, the hien qua cell IoU va structure F1.
- Table structure reconstruction van la diem kho, dac biet voi merged cell, sparse column va exact HTML/CSV.

Khong nen claim da giai quyet hoan toan PubTables structure. Nen viet la "da co benchmark PubTables-1M OTSL that va cai tien deterministic post-processing, nhung exact structure recognition van la han che can nghien cuu tiep".
