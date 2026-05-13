# TATR Integration - 2026-05-12

This pass adds an experimental Microsoft Table Transformer backend for BOXTALK table benchmarks.

## What TATR Is

Microsoft Table Transformer (TATR) is a pretrained DETR-style model family for table detection and table structure recognition. This integration uses pretrained weights only:

| Purpose | Model |
|---|---|
| Table detection | `microsoft/table-transformer-detection` |
| Structure recognition | `microsoft/table-transformer-structure-recognition-v1.1-all` |

No model training or fine-tuning is performed.

## How It Connects to BOXTALK

The new backend lives in:

```text
app/ingest/tatr_table_backend.py
```

It is lazy-loaded. The TATR models are loaded only when the benchmark is run with:

```powershell
python scripts/benchmark_ingest_suite.py --dataset pubtables_structure --data-dir data/benchmarks/pubtables_structure --limit 25 --out results/ingest/pubtables_structure_25_tatr --mode table --table-backend tatr --save-predictions
```

The default ingest pipeline is unchanged. Backend choices are:

| Backend | Behavior |
|---|---|
| `default` | Existing ingest/table pipeline |
| `ocr_cluster` | Existing OCR table clustering path |
| `tatr` | TATR detection + TATR structure grid reconstruction |
| `hybrid_tatr` | TATR row/column/spanning-cell geometry + supplied OCR/PDF word boxes |

TATR predicts geometry only. It does not recognize text. If PDF text boxes are available, the backend can assign them into cells by geometry. For image-only PubTables OTSL samples, cells are emitted with empty text unless a separate OCR/text-box provider is used.

## Pipeline Integration

`hybrid_tatr` can now be enabled as a specialized table-region module in the real ingest pipeline. It is called from `extract_table_region()` only after an upstream layout/model-routed backend has detected a `table` region:

```text
PDF -> probe -> layout/model routing -> table region -> hybrid_tatr -> fallback table extractor
```

Enable it with:

```powershell
$env:BOXBIIBOO_TABLE_BACKEND="hybrid_tatr"
```

or:

```powershell
$env:BOXBIIBOO_ENABLE_HYBRID_TATR_TABLES="1"
```

The production path uses PDF native word boxes when present. It does not invoke PaddleOCR in the TATR process; scanned-PDF OCR words should be prepared separately if needed. If TATR, the model weights, or PDF word boxes are unavailable, the extractor falls back to the existing deterministic table extraction path.

## Implementation Notes

- Uses Hugging Face `AutoImageProcessor` and `TableTransformerForObjectDetection`.
- Uses CUDA through PyTorch when available.
- Does not import or run PaddleOCR.
- Adds compatibility fallbacks for current Transformers validation:
  - `dilation=None` in the structure model config is normalized to `False`.
  - processor `shortest_edge=None` is normalized to `800`.
- Converts TATR output labels into row bands, column bands and spanning-cell candidates.
- Builds grid cells from row/column intersections.
- Assigns provided text boxes by center-in-cell / overlap.
- Exports simple CSV/HTML from the grid.

## PubTables Structure 25-Sample Result

| Metric | Current/default backend | TATR backend |
|---|---:|---:|
| success rate | 1.000 | 1.000 |
| table detection F1@0.50 | 0.967 | 0.987 |
| table detection F1@0.75 | 0.767 | 0.987 |
| table cell IoU@0.50 F1 | 0.659 | 0.491 |
| table cell IoU@0.75 F1 | 0.184 | 0.103 |
| table structure F1 | 0.202 | 0.010 |
| text assignment F1 | 0.963 | 0.015 |
| row count MAE | 2.040 | 0.600 |
| col count MAE | 0.840 | 0.000 |
| spanning cell count mean | n/a | 1.040 |
| exact CSV / HTML | 0.000 / 0.000 | 0.000 / 0.000 |
| latency mean / p50 / p95 | 0.798 / 0.394 / 1.243 sec | 0.697 / 0.062 / 0.094 sec |

## Interpretation

TATR improves table-level detection and row/column count accuracy on this subset. This means it is useful as a geometry backend.

TATR does not yet improve cell IoU or text-cell structure in this benchmark because:

- TATR predicts full grid cells, while the current PubTables OTSL subset cell bboxes are often closer to text/content boxes.
- The TATR backend does not run OCR in-process, so image-only samples have empty cell text.
- Exact CSV/HTML requires text, row/column alignment, merged-cell semantics and markup to match at the same time.

## Recommendation

Do not replace the main table backend with TATR yet.

Use TATR as an experimental geometry backend and next build a hybrid:

```text
TATR rows/columns/spans + existing OCR/PDF text boxes + deterministic text assignment
```

That hybrid should be run in a separate process if PaddleOCR text boxes are needed, to avoid CUDA/cuDNN conflicts between Paddle and PyTorch in the same environment.

## Hybrid TATR Update

`hybrid_tatr` has now been added as a benchmark backend. It keeps TATR geometry and assigns supplied word boxes into cells deterministically.

On the 25-sample PubTables OTSL subset with annotation-derived `pubtables_cell_tokens_proxy` word boxes:

| Metric | TATR | Hybrid TATR |
|---|---:|---:|
| detection F1@0.50 | 0.987 | 0.987 |
| row count MAE | 0.600 | 0.600 |
| col count MAE | 0.000 | 0.000 |
| table structure F1 | 0.010 | 0.772 |
| non-empty predicted cell rate | 0.000 | 0.972 |
| GriTS-con-like | 0.006 | 0.700 |

See `docs/HYBRID_TATR_GRITS_2026-05-12.md`. The proxy word boxes isolate text assignment quality; production scanned PDFs still require real OCR/PDF word boxes.

For a production-like scanned-PDF benchmark, generate OCR word boxes in the PaddleOCR environment first:

```powershell
.\.venv-ocr-gpu\Scripts\python.exe scripts\prepare_pubtables_ocr_word_boxes.py --data-dir data\benchmarks\pubtables_structure --out data\benchmarks\pubtables_structure_ocr_words --limit 25 --lang en --device gpu:0
```

Then consume that manifest from the PyTorch/TATR environment:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset pubtables_structure --data-dir data\benchmarks\pubtables_structure_ocr_words --limit 25 --out results\ingest\pubtables_structure_ocr_words_25_hybrid_tatr --mode table --table-backend hybrid_tatr --save-predictions
```

On the same 25-sample subset with real PaddleOCR word boxes, `hybrid_tatr` reports:

| Metric | Value |
|---|---:|
| table detection F1@0.50 / @0.75 | 0.987 / 0.987 |
| table cell IoU@0.50 / @0.75 F1 | 0.598 / 0.248 |
| table structure F1 | 0.638 |
| text assignment F1 | 0.955 |
| row count MAE / col count MAE | 0.600 / 0.000 |
| non-empty predicted cell rate | 0.958 |
| GriTS-con-like | 0.387 |
| exact CSV / HTML | 0.040 / 0.000 |

## Reproduce

Default backend:

```powershell
.\.venv-ocr-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset pubtables_structure --data-dir data\benchmarks\pubtables_structure --limit 25 --out results\ingest\pubtables_structure_25_after_rowcol_fix --mode table --save-predictions
```

TATR backend:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset pubtables_structure --data-dir data\benchmarks\pubtables_structure --limit 25 --out results\ingest\pubtables_structure_25_tatr --mode table --table-backend tatr --save-predictions
```

Validation:

```powershell
.\.venv-gpu\Scripts\python.exe -m compileall app scripts
.\.venv-gpu\Scripts\python.exe -m pytest -q
.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset mock --limit 5 --out results\ingest\mock_recheck --mode all
```
