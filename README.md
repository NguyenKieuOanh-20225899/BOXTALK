# BOXBIIBOO

## Model-based ingest

The ingest pipeline can optionally use a model-based layout detector before the
rule-based layout fallback.

Required environment variable:

```bash
export BOXBIIBOO_LAYOUT_MODEL_NAME="Aryn/deformable-detr-DocLayNet"
```

Model-based layout routing also requires the `transformers` vision stack and
`timm`. Install from [requirements_routed_rag.txt](/Users/kieuoanh/git/BOXTALK/requirements_routed_rag.txt).

Optional knobs:

```bash
export BOXBIIBOO_LAYOUT_DEVICE="cpu"
export BOXBIIBOO_LAYOUT_SCORE_THRESHOLD="0.35"
export BOXBIIBOO_LAYOUT_RENDER_SCALE="2.0"
export BOXBIIBOO_ENABLE_MODEL_ROUTING="1"
export BOXBIIBOO_OCR_REGION_SCALE="3.0"
```

When `BOXBIIBOO_LAYOUT_MODEL_NAME` is set, `ingest_pdf()` tries:

1. `model_routed`: detect regions with the model, then route each region to
   specialized text / OCR / table extraction.
2. `model_layout`: use the model detections directly as layout blocks.
3. `layout`: fall back to the existing document-level layout backend.

`model_routed` routes table regions to a table-specific extractor, routes text
regions to native PDF text extraction when available, and falls back to OCR for
regions without usable text.

Recommended concrete model for this repo:

```bash
export BOXBIIBOO_LAYOUT_MODEL_NAME="Aryn/deformable-detr-DocLayNet"
```

You can also use:

```bash
export BOXBIIBOO_LAYOUT_MODEL_NAME="default"
```

## Benchmark protocol

Benchmark workflow is split into:

- production benchmark on `data/test_probe`
- scientific benchmark on `DocLayNet` and `PubTables-1M`
- a suite runner that keeps both reports separated and writes a merged summary

Protocol details are documented in [docs/BENCHMARK_PROTOCOL.md](/Users/kieuoanh/git/BOXTALK/docs/BENCHMARK_PROTOCOL.md).

Scientific benchmark datasets can be standardized locally with:

```bash
.venv/bin/python scripts/setup_benchmark_datasets.py --dataset all --pubtables-splits test
.venv/bin/python scripts/setup_benchmark_datasets.py --dataset all --pubtables-splits test --validate-only
make benchmark-setup-pubtables-test
make benchmark-validate-pubtables-test
make benchmark-reclaim-pubtables-raw
make benchmark-mac-quick
```

This prepares the standardized local layout under `data/benchmarks/` and writes
per-dataset manifests so the scientific benchmark can detect missing archives or
incomplete extraction early.

`Makefile` targets are intended as the one-command entrypoint:

```bash
make benchmark-mac-quick
make benchmark-mac-full
make benchmark-suite-pubtables-test PRODUCTION_MAX_PER_LABEL=1
make benchmark-setup-all BENCHMARKS_ROOT=/Volumes/External/boxbiiboo-benchmarks
```

If local disk is tight, use `PubTables test` only on the internal drive and put
full `DocLayNet` on an external volume via `BENCHMARKS_ROOT=/path/to/drive`.
After extraction, `make benchmark-reclaim-pubtables-raw` removes the large raw
tarballs while keeping the extracted dataset benchmark-ready.
