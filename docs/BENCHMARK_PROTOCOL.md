# Benchmark Protocol

This repo now separates benchmark work into two layers:

## 1. Production benchmark

Purpose:
- measure end-to-end ingest behavior on the repo's internal PDF set
- compare baseline ingest against model-routed ingest

Dataset:
- `data/test_probe`

Runner:
- `scripts/benchmark_ingest_standard.py`

Useful local control:
- `--max-per-label N` to run a lighter sampled production benchmark on Mac while keeping the same metric protocol

Outputs:
- per-profile JSON
- per-file CSV / JSON
- production summary JSON / Markdown

Primary metrics:
- success rate
- probe accuracy
- latency mean / median / p95
- pages per second
- bbox coverage ratio
- backend distribution
- route distribution

## 2. Scientific benchmark

Purpose:
- evaluate region detection and table specialization against paper-backed datasets
- keep metrics grounded in annotation-level overlap rather than only pipeline logs

Datasets:
- `DocLayNet` for layout detection
- `PubTables-1M` detection split for table detection

Runner:
- `scripts/benchmark_ingest_scientific.py`

Expected local dataset roots:
- `data/benchmarks/doclaynet`
- `data/benchmarks/pubtables_detection`

Standardized local layout:

```text
data/benchmarks/
  manifest.json
  doclaynet/
    manifest.json
    raw/
      DocLayNet_core.zip
    extracted/
      DocLayNet_core/
        COCO/
        PNG/
  pubtables_detection/
    manifest.json
    raw/
      PubTables-1M-Detection_*.tar.gz
    extracted/
      images/
        train/
        val/
        test/
      annotations/
        train/
        val/
        test/
```

Dataset setup runner:
- `scripts/setup_benchmark_datasets.py`

Examples:

```bash
.venv/bin/python scripts/setup_benchmark_datasets.py --dataset doclaynet
.venv/bin/python scripts/setup_benchmark_datasets.py --dataset pubtables --pubtables-splits test
.venv/bin/python scripts/setup_benchmark_datasets.py --dataset all --pubtables-splits test
.venv/bin/python scripts/setup_benchmark_datasets.py --dataset all --pubtables-splits test --validate-only
make benchmark-setup-pubtables-test
make benchmark-validate-pubtables-test
make benchmark-reclaim-pubtables-raw
make benchmark-mac-quick
make benchmark-suite-all BENCHMARKS_ROOT=/Volumes/External/boxbiiboo-benchmarks
```

Notes:
- `DocLayNet` downloads the official `DocLayNet_core.zip`.
- `PubTables-1M` setup is detection-focused by default and only needs image + XML annotation archives for the selected split(s).
- default `PubTables-1M` setup downloads `test` only, because `train` is much larger.
- each dataset root writes its own `manifest.json`, and `data/benchmarks/manifest.json` records the combined local benchmark state.
- `--validate-only` checks whether the standardized local layout is complete enough for the scientific benchmark without downloading anything.
- `Makefile` provides one-command flows for Mac-safe `PubTables test` runs and for full setup on larger external storage.
- once extraction is complete, raw archives are optional for `ready=true`; `make benchmark-reclaim-pubtables-raw` can recover disk while keeping the extracted dataset usable for benchmarks.

Metrics:
- IoU@0.50 and IoU@0.75
- micro precision / recall / F1
- macro F1
- per-class TP / FP / FN
- table non-empty extraction rate for PubTables predictions

Protocol note:
- benchmark pages are rendered into temporary single-page PDFs before running the ingest pipeline
- this preserves the repo's actual `ingest_pdf()` path while still allowing evaluation against image-based benchmark annotations

## 3. Combined suite

Runner:
- `scripts/benchmark_ingest_suite.py`

Purpose:
- execute production and scientific benchmarks separately
- write a merged suite summary while keeping both result groups independent

Useful local control:
- `--production-max-per-label N` to keep suite runs manageable on a laptop
- `--skip-doclaynet` and `--skip-pubtables` for scientific / suite runs when only one academic dataset is present locally

## Model choice

Default benchmark model:

```bash
export BOXBIIBOO_LAYOUT_MODEL_NAME="Aryn/deformable-detr-DocLayNet"
```

This is the concrete model choice used for the `model_routed_doclaynet` profile.
