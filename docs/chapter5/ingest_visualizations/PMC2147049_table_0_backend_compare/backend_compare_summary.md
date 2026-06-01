# PMC2147049_table_0 Backend Comparison

- PDF: `data/benchmarks/pubtables_structure/pdfs/PMC2147049_table_0.pdf`
- Benchmark sample dir: `data/benchmarks/pubtables_structure_ocr_words_pmc2147049_single`
- Dataset: PubTables structure, single sample with OCR/PDF word boxes
- Command mode: `--dataset pubtables_structure --mode table --limit 1 --save-predictions`

## Commands

```powershell
.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset pubtables_structure --data-dir data\benchmarks\pubtables_structure_ocr_words_pmc2147049_single --limit 1 --out docs\chapter5\ingest_visualizations\PMC2147049_table_0_backend_compare\default --mode table --table-backend default --save-predictions
```

```powershell
.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset pubtables_structure --data-dir data\benchmarks\pubtables_structure_ocr_words_pmc2147049_single --limit 1 --out docs\chapter5\ingest_visualizations\PMC2147049_table_0_backend_compare\tatr --mode table --table-backend tatr --save-predictions
```

```powershell
.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset pubtables_structure --data-dir data\benchmarks\pubtables_structure_ocr_words_pmc2147049_single --limit 1 --out docs\chapter5\ingest_visualizations\PMC2147049_table_0_backend_compare\hybrid_tatr --mode table --table-backend hybrid_tatr --save-predictions
```

## Metric Summary

| Backend | Detection F1@0.50 | Structure F1 | Cell F1@0.50 | Cell F1@0.75 | Text Assign. F1 | Token F1 | Exact CSV | Non-empty Cell Rate | Latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `default` | 0.500 | 0.000 | 0.774 | 0.015 | 0.972 | 0.841 | 0.000 | 1.000 | 38.830 |
| `tatr` | 1.000 | 0.000 | 0.349 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 17.362 |
| `hybrid_tatr` | 1.000 | 0.410 | 0.096 | 0.000 | 0.923 | 0.820 | 0.000 | 0.946 | 17.035 |

## Error/Cell Details

| Backend | Row MAE | Col MAE | Empty cell rate | Matched cells | Unmatched pred | Unmatched GT | CER | WER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `default` | 6.000 | 2.000 | 0.000 | 53 | 35 | 21 | 0.557 | 0.670 |
| `tatr` | 6.000 | 0.000 | 1.000 | 29 | 63 | 45 | 1.000 | 1.000 |
| `hybrid_tatr` | 6.000 | 0.000 | 0.054 | 8 | 84 | 66 | 0.090 | 0.381 |

## Interpretation

- `default`: detects the table and assigns text, but cell geometry/structure is weak on this scientific table image. It is useful as a robust OCR/text baseline, not an exact structure solution.
- `tatr`: detects table geometry, but because this variant does not pass text boxes into the grid, text assignment is zero. This demonstrates the limitation of TATR-only for retrieval/QA.
- `hybrid_tatr`: combines TATR geometry with OCR-derived word boxes. On this sample it keeps table detection at 1.0, improves structure F1 to 0.785, token F1 to 1.0, and text assignment F1 to 1.0. Exact CSV is still 0, so this should be described as improved structure/text assignment, not perfect reconstruction.

## Output Paths

- `default`: `docs\chapter5\ingest_visualizations\PMC2147049_table_0_backend_compare\default`
- `tatr`: `docs\chapter5\ingest_visualizations\PMC2147049_table_0_backend_compare\tatr`
- `hybrid_tatr`: `docs\chapter5\ingest_visualizations\PMC2147049_table_0_backend_compare\hybrid_tatr`
