# Hybrid TATR + GriTS-like Evaluation - 2026-05-12

This pass adds an experimental `hybrid_tatr` table backend:

```text
TATR table geometry + word boxes + deterministic text assignment
```

It does not train a model, does not use an LLM, and does not replace the default ingest backend.

## Research Mapping

| Research idea | How it is used |
|---|---|
| PubTables-1M / Table Transformer | Pretrained TATR detection and structure-recognition models provide table, row, column and spanning-cell geometry. |
| DeepDeSRT | Architectural pattern: table detection first, then table structure recognition. The DeepDeSRT model itself is not used. |
| GriTS | Adds GriTS-like topology/location/content metrics that are softer than exact CSV/HTML. |
| PubTabNet | Used as background for HTML reconstruction and exact/HTML-style table output. |
| SciTSR | Used as background for complex scientific tables, merged cells and multi-row/multi-column headers. |

## Word Boxes

`hybrid_tatr` needs text boxes because TATR is image-only geometry.

For the current PubTables-1M OTSL local subset, `scripts/prepare_pubtables_structure_subset.py` now writes:

```json
"word_boxes": [
  {
    "text": "...",
    "bbox": [x0, y0, x1, y1],
    "confidence": 1.0,
    "source": "pubtables_cell_tokens_proxy"
  }
]
```

Important: `pubtables_cell_tokens_proxy` is produced from the PubTables annotation cell tokens and bboxes. It is useful for isolating the geometry-to-cell text assignment problem, but it is not a production OCR source. Production use still needs OCR/PDF text boxes.

## Metrics

Existing metrics are preserved:

- table detection F1@0.50 / F1@0.75
- table cell IoU@0.50 / IoU@0.75
- table text cell structure F1
- row/column count MAE
- exact CSV/HTML

New metrics:

- `non_empty_pred_cell_rate`
- `text_source_missing_count`
- `grits_top_like`
- `grits_loc_like`
- `grits_con_like`

The GriTS-like metrics are local approximations, not the official Microsoft GriTS implementation:

- `grits_top_like`: row/col/span topology match.
- `grits_loc_like`: topology match weighted by bbox IoU.
- `grits_con_like`: topology/location match weighted by text F1.

## PubTables Structure 25-Sample Results

| Metric | Default | TATR | Hybrid TATR |
|---|---:|---:|---:|
| success rate | 1.000 | 1.000 | 1.000 |
| detection F1@0.50 | 0.967 | 0.987 | 0.987 |
| detection F1@0.75 | 0.767 | 0.987 | 0.987 |
| cell IoU@0.50 F1 | 0.659 | 0.491 | 0.958 |
| cell IoU@0.75 F1 | 0.184 | 0.103 | 0.944 |
| table structure F1 | 0.202 | 0.010 | 0.772 |
| text assignment F1 | 0.963 | 0.015 | 0.999 |
| row count MAE | 2.040 | 0.600 | 0.600 |
| col count MAE | 0.840 | 0.000 | 0.000 |
| non-empty predicted cell rate | 1.000 | 0.000 | 0.972 |
| text source missing count | 0.000 | 0.000 | 0.000 |
| GriTS-top-like | 0.780 | 0.933 | 0.933 |
| GriTS-loc-like | 0.161 | 0.368 | 0.701 |
| GriTS-con-like | 0.147 | 0.006 | 0.700 |
| exact CSV | 0.000 | 0.000 | 0.480 |
| exact HTML | 0.000 | 0.000 | 0.000 |
| latency mean / p50 / p95 | 1.020 / 0.429 / 1.335 sec | 0.890 / 0.092 / 0.137 sec | 0.794 / 0.087 / 0.308 sec |

## Interpretation

Hybrid TATR confirms that TATR geometry is strong for row/column layout. When credible word boxes are available, deterministic assignment can recover much stronger cell text structure.

The result should be reported carefully:

- Strong evidence for the geometry + text-assignment design.
- Not yet proof of production OCR quality, because the current word source is annotation-derived proxy text.
- Exact HTML remains 0 because row/colspan semantics and markup still differ from ground truth.

## Recommendation

Do not replace the default backend globally yet.

Use `hybrid_tatr` as a specialized table-region module. The production-safe integration is:

```text
layout/model detects table region
-> extract_table_region()
-> hybrid_tatr tries TATR structure + PDF word boxes
-> fallback to table_words_grid / table_clip_text / OCR table text if hybrid_tatr is unavailable
```

For scanned PDFs, collect OCR words in a separate PaddleOCR process or from an OCR manifest to avoid Paddle/PyTorch CUDA conflicts.

Enable the module in the ingest pipeline:

```powershell
$env:BOXBIIBOO_TABLE_BACKEND="hybrid_tatr"
```

or:

```powershell
$env:BOXBIIBOO_ENABLE_HYBRID_TATR_TABLES="1"
```

The module is called only when an upstream layout/model-routed backend has already classified a region as `table`. It is not run for normal paragraphs, captions, headings or figures.

## Real OCR Word-Box Path

`scripts/prepare_pubtables_ocr_word_boxes.py` adds the production-like path for the same hybrid backend:

```text
PubTables image -> PaddleOCR line boxes -> split line text into word boxes -> hybrid_tatr
```

This script should be run in `.venv-ocr-gpu`, because it imports PaddleOCR. The TATR benchmark should still be run in `.venv-gpu`, because it imports PyTorch/Transformers. The two-step flow avoids the Windows CUDA/cuDNN DLL conflict between Paddle and PyTorch.

Generate OCR word-box manifest:

```powershell
.\.venv-ocr-gpu\Scripts\python.exe scripts\prepare_pubtables_ocr_word_boxes.py --data-dir data\benchmarks\pubtables_structure --out data\benchmarks\pubtables_structure_ocr_words --limit 25 --lang en --device gpu:0 --min-confidence 0.5
```

Run hybrid TATR with real OCR-derived words:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset pubtables_structure --data-dir data\benchmarks\pubtables_structure_ocr_words --limit 25 --out results\ingest\pubtables_structure_ocr_words_25_hybrid_tatr --mode table --table-backend hybrid_tatr --save-predictions
```

The annotation-proxy result above is an upper-bound style diagnostic for geometry-to-cell assignment. The OCR-word result is the result to use when discussing production scanned PDFs.

### PubTables Structure 25-Sample OCR-Word Result

| Metric | Hybrid TATR + annotation proxy | Hybrid TATR + PaddleOCR word boxes |
|---|---:|---:|
| success rate | 1.000 | 1.000 |
| detection F1@0.50 | 0.987 | 0.987 |
| detection F1@0.75 | 0.987 | 0.987 |
| cell IoU@0.50 F1 | 0.958 | 0.598 |
| cell IoU@0.75 F1 | 0.944 | 0.248 |
| table structure F1 | 0.772 | 0.638 |
| text assignment F1 | 0.999 | 0.955 |
| row count MAE | 0.600 | 0.600 |
| col count MAE | 0.000 | 0.000 |
| non-empty predicted cell rate | 0.972 | 0.958 |
| text source missing count | 0.000 | 0.000 |
| GriTS-top-like | 0.933 | 0.933 |
| GriTS-loc-like | 0.701 | 0.409 |
| GriTS-con-like | 0.700 | 0.387 |
| exact CSV | 0.480 | 0.040 |
| exact HTML | 0.000 | 0.000 |

The OCR-word path keeps the main structural gains without using ground-truth cell tokens. Exact CSV drops because real OCR introduces recognition errors, line splitting differences, and occasional caption/footnote words near the table.

Implementation detail: hybrid cells keep TATR grid geometry in `grid_bbox`, but when text boxes are available the public `bbox` is the assigned content bbox. This matches PubTables OTSL cell annotations better while preserving grid geometry for debugging.

## Reproduce

Prepare subset with word-box proxy:

```powershell
.\.venv-ocr-gpu\Scripts\python.exe scripts\prepare_pubtables_structure_subset.py --limit 25 --out data\benchmarks\pubtables_structure
```

Default:

```powershell
.\.venv-ocr-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset pubtables_structure --data-dir data\benchmarks\pubtables_structure --limit 25 --out results\ingest\pubtables_structure_25_default_recheck --mode table --save-predictions
```

TATR geometry-only:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset pubtables_structure --data-dir data\benchmarks\pubtables_structure --limit 25 --out results\ingest\pubtables_structure_25_tatr_recheck --mode table --table-backend tatr --save-predictions
```

Hybrid TATR:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset pubtables_structure --data-dir data\benchmarks\pubtables_structure --limit 25 --out results\ingest\pubtables_structure_25_hybrid_tatr --mode table --table-backend hybrid_tatr --save-predictions
```
