# BOXBIIBOO

## Model-based ingest

The ingest pipeline can optionally use a model-based layout detector before the
rule-based layout fallback.

OCR-backed ingest and the scientific PubTables benchmark also require
`paddleocr` plus a CPU-compatible `paddlepaddle` build. On Windows, use
`paddlepaddle==3.2.2`; `3.3.x` currently breaks OCR inference with
`ConvertPirAttribute2RuntimeAttribute` errors.

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

## Retrieval and Routed RAG

The repo now includes a minimal, benchmarkable retrieval stack for PDF QA:

- BM25 sparse baseline with `rank-bm25`
- dense retrieval baselines:
  - MiniLM via `sentence-transformers/all-MiniLM-L6-v2`
  - multilingual MiniLM via `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  - Contriever via `facebook/contriever`
  - DPR via separate question/context encoders
- hybrid retrieval with weighted-sum and reciprocal-rank-fusion modes
- ColBERT-style late interaction with `colbert-ir/colbertv2.0`
- optional rerank stage with no-op, heuristic, cross-encoder, or ColBERT implementations
- query-aware retrieval planning for routed RAG query types
- BEIR benchmark runner for real IR datasets such as SciFact/NFCorpus/FiQA

Retrieval modules live under `app/retrieval/`:

- `schemas.py`: `DocumentChunkRef`, `RetrievedHit`, `RetrievalResult`, `RetrievalConfig`
- `embedding_backends.py`: MiniLM, Contriever, and DPR encoder backends
- `bm25_retriever.py`: BM25 sparse retriever
- `dense_retriever.py`: sentence-transformer dense retriever
- `colbert_retriever.py`: exact ColBERT-style late interaction retriever
- `hybrid_retriever.py`: BM25+dense fusion
- `reranker.py`: rerank interfaces and implementations
- `index_store.py`: filesystem index/corpus store
- `route_planner.py`: query type to retrieval strategy/config mapping
- `service.py`: retrieval service boundary used by routed RAG

Build a retrieval index from a PDF:

```powershell
.\.venv\Scripts\python.exe scripts\build_retrieval_index.py `
  --pdf data\retrieval_smoke\employee_handbook_smoke.pdf `
  --output-dir results\retrieval_index\smoke_test
```

Select a real dense baseline:

```powershell
# DPR baseline
.\.venv\Scripts\python.exe scripts\build_retrieval_index.py `
  --pdf data\retrieval_smoke\employee_handbook_smoke.pdf `
  --output-dir results\retrieval_index\dpr_test `
  --dense-preset dpr-single-nq

# Contriever baseline
.\.venv\Scripts\python.exe scripts\build_retrieval_index.py `
  --pdf data\retrieval_smoke\employee_handbook_smoke.pdf `
  --output-dir results\retrieval_index\contriever_test `
  --dense-preset contriever

# Multilingual MiniLM baseline, useful for Vietnamese PDFs
.\.venv\Scripts\python.exe scripts\build_retrieval_index.py `
  --pdf data\real_pdfs\QCDT_2025_5445_QD-DHBK.pdf `
  --output-dir results\retrieval_index\real_qcdt_multilingual_minilm `
  --dense-preset multilingual-minilm

# ColBERT late-interaction index
.\.venv\Scripts\python.exe scripts\build_retrieval_index.py `
  --pdf data\retrieval_smoke\employee_handbook_smoke.pdf `
  --output-dir results\retrieval_index\colbert_test `
  --dense-preset minilm `
  --build-colbert
```

Build from existing chunk output:

```powershell
.\.venv\Scripts\python.exe scripts\build_retrieval_index.py `
  --chunks-json path\to\chunks.json `
  --output-dir results\retrieval_index\my_index

.\.venv\Scripts\python.exe scripts\build_retrieval_index.py `
  --chunks-jsonl path\to\chunks.jsonl `
  --output-dir results\retrieval_index\my_index
```

Run the smoke retrieval benchmark:

```powershell
.\.venv\Scripts\python.exe scripts\create_retrieval_smoke_dataset.py

.\.venv\Scripts\python.exe scripts\build_retrieval_index.py `
  --pdf data\retrieval_smoke\employee_handbook_smoke.pdf `
  --output-dir results\retrieval_index\smoke_test

.\.venv\Scripts\python.exe scripts\benchmark_retrieval.py `
  --index-dir results\retrieval_index\smoke_test `
  --queries data\retrieval_smoke\queries.jsonl `
  --output-dir results\retrieval_benchmark\smoke_test `
  --top-k 5 `
  --strategy all
```

The benchmark writes:

- `benchmark_summary.json`
- `per_question.json`
- `per_question.csv`
- `README.md`

Supported benchmark strategies:

- `bm25`
- `dense`
- `colbert` when the index was built with `--build-colbert`
- `hybrid`
- `hybrid_rerank`
- `all`

Run a real BEIR benchmark sample:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_beir_retrieval.py `
  --dataset scifact `
  --query-limit 50 `
  --corpus-limit 2000 `
  --strategy bm25 `
  --strategy dense `
  --strategy hybrid `
  --dense-preset minilm
```

Use `--dense-preset multilingual-minilm`, `--dense-preset contriever`, or
`--dense-preset dpr-single-nq` to run the corresponding baseline. Add
`--build-colbert --strategy colbert` for ColBERT late-interaction evaluation on
small samples.

Make targets:

```bash
make retrieval-smoke
make retrieval-build RETRIEVAL_DENSE_PRESET=contriever
make retrieval-benchmark
make retrieval-beir-scifact BEIR_QUERY_LIMIT=50 BEIR_CORPUS_LIMIT=2000
```

Run routed RAG from a prebuilt retrieval index:

```powershell
$env:BOXTALK_RETRIEVAL_INDEX_DIR="results\retrieval_index\smoke_test"
.\.venv\Scripts\uvicorn.exe app.routed_rag_starter:app --reload
```

Or let the starter build retrieval from a PDF at startup:

```powershell
$env:PDF_PATH="data\retrieval_smoke\employee_handbook_smoke.pdf"
$env:BOXTALK_ROUTED_RAG_BUILD_DENSE="1"
.\.venv\Scripts\uvicorn.exe app.routed_rag_starter:app --reload
```

## UI MVP and Grounded Fallback

The repo now includes a static MVP UI served by the FastAPI app. It focuses on:

- PDF upload and indexing
- document library management
- grounded QA with citations
- a source viewer for page-level inspection
- a developer-details toggle for route and fallback trace

Run the backend plus UI:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.routed_rag_starter:app --reload
```

Or via Make:

```bash
make ui-dev PYTHON=.venv/Scripts/python.exe
```

Open `http://127.0.0.1:8000/`.

Useful UI/runtime env flags:

```powershell
# default main path
$env:BOXTALK_UI_QA_PIPELINE="routed_grounded"

# enable grounded LLM fallback on the standard routed pipeline
$env:BOXTALK_ENABLE_LLM_FALLBACK="1"

# experimental: adaptive retry, but only allow fallback on the final selected route
$env:BOXTALK_UI_QA_PIPELINE="adaptive_route_retry"
$env:BOXTALK_ENABLE_LLM_FALLBACK_ON_FINAL_ROUTE_ONLY="1"
```

Fallback benchmarking stays separate from the main regression gates.

Dummy smoke check:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_llm_fallback.py `
  --manifest data\llm_fallback_benchmark\manifest.json `
  --output-dir results\llm_fallback_benchmark\dummy_smoke `
  --llm-fallback-provider dummy `
  --no-warmup
```

OpenAI-compatible provider benchmark:

```powershell
$env:BOXTALK_LLM_BASE_URL="https://your-endpoint/v1"
$env:BOXTALK_LLM_API_KEY="your-key"
$env:BOXTALK_LLM_MODEL="your-model"

.\.venv\Scripts\python.exe scripts\benchmark_llm_fallback.py `
  --manifest data\llm_fallback_benchmark\manifest.json `
  --output-dir results\llm_fallback_benchmark\openai_run `
  --llm-fallback-provider openai-compatible `
  --skip-build `
  --no-warmup
```

Useful debug endpoint:

```text
GET /debug/retrieval-plan?question=How%20long%20do%20staff%20have%20to%20submit%20receipts%3F
```

## Grounded QA Benchmark

The QA layer connects routed retrieval to:

- evidence checking and sufficiency decisions
- extractive grounded answer generation
- citations tied to retrieved chunks
- QA metrics over `gold_answer`, `match_text`, evidence match, and groundedness
- adaptive route-retry QA that retries alternate query routes when evidence quality is weak

Create the controlled 40-question QA benchmark:

```powershell
.\.venv\Scripts\python.exe scripts\create_qa_benchmark_dataset.py
```

It writes:

- `data\qa_benchmark\operations_handbook.pdf`
- `data\qa_benchmark\queries.jsonl`
- `data\qa_benchmark\manifest.json`

The query set covers `factoid`, `policy`, `procedural`, `comparison`, and
`ambiguous` / insufficient-evidence questions. Each row includes `gold_answer`,
`query_type`, `should_answer`, and evidence targets such as `expected_section`
or `match_text`.

Build the QA retrieval index:

```powershell
.\.venv\Scripts\python.exe scripts\build_retrieval_index.py `
  --pdf data\qa_benchmark\operations_handbook.pdf `
  --output-dir results\retrieval_index\qa_operations_minilm `
  --dense-preset minilm
```

Run the full QA baseline + ablation benchmark:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_qa.py `
  --index-dir results\retrieval_index\qa_operations_minilm `
  --queries data\qa_benchmark\queries.jsonl `
  --output-dir results\qa_benchmark\qa_operations_minilm_all `
  --config all
```

`--config all` runs:

- `bm25_only`
- `dense_only`
- `hybrid_no_routing`
- `routed_grounded`
- `adaptive_route_retry`
- `no_evidence_checker`
- `no_router`
- `no_citation_grounding`
- `no_metadata_filter`

Run a single QA config on any built retrieval index:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_qa.py `
  --index-dir results\retrieval_index\smoke_minilm `
  --queries data\retrieval_smoke\queries.jsonl `
  --output-dir results\qa_benchmark\smoke_minilm `
  --config routed_grounded
```

Outputs:

- `qa_summary.json`
- `per_question.json`
- `per_question.csv`
- `README.md`

The QA summary reports overall metrics, per-query-type metrics, abstention
precision/recall, hallucination rate, average latency, route retry rate,
average route attempts, route match rates, and ablation deltas. The per-question
files include route trace fields: `initial_route_type`, `selected_route_type`,
`route_attempt_count`, `route_attempts`, `retrieval_strategy`,
`retrieval_top_k`, `selected_hit_ids`, `evidence_reason`, `answer_status`, and
citation chunk IDs.

Run only the routed baseline versus adaptive route-retry:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_qa.py `
  --index-dir results\retrieval_index\qa_operations_minilm `
  --queries data\qa_benchmark\queries.jsonl `
  --output-dir results\qa_benchmark\qa_operations_adaptive `
  --config routed_grounded,adaptive_route_retry
```

For Vietnamese regulatory PDFs, inspect `bm25_only` alongside hybrid configs.
The current QA planner increases BM25 weight for Vietnamese queries and expands
one neighboring chunk so split table/list evidence can be grounded.

Make target:

```bash
make qa-benchmark-all

# Windows PowerShell with the local venv:
make qa-benchmark-all PYTHON=.venv/Scripts/python.exe

make qa-benchmark \
  QA_INDEX_DIR=results/retrieval_index/qa_operations_minilm \
  QA_QUERIES=data/qa_benchmark/queries.jsonl \
  QA_BENCHMARK_DIR=results/qa_benchmark/qa_operations_minilm_routed \
  QA_CONFIGS=routed_grounded
```

## User PDF Benchmark Suite

Use this suite to check whether the QA pipeline generalizes across real user PDF
types, not just one smoke dataset. It currently covers a Vietnamese
policy/regulation PDF, an English scientific paper, and a handbook/manual QA
set.

Run the full suite:

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_user_pdf_suite.py `
  --manifest data\user_pdf_benchmark_suite.json `
  --output-dir results\user_pdf_benchmark_suite\current
```

Outputs:

- `results\user_pdf_benchmark_suite\current\suite_summary.json`
- `results\user_pdf_benchmark_suite\current\per_question.json`
- `results\user_pdf_benchmark_suite\current\per_question.csv`
- `results\user_pdf_benchmark_suite\current\README.md`

Run through Make:

```bash
make user-pdf-suite

make user-pdf-suite \
  USER_PDF_SUITE_ARGS="--config routed_grounded,adaptive_route_retry"
```

The protocol and manifest format are documented in
`docs/USER_PDF_BENCHMARK_SUITE.md`.

Full retrieval protocol details are documented in
`docs/RETRIEVAL_BENCHMARK_PROTOCOL.md`.

## Baseline regression gate

The current locked interpretation is:

- `bm25_only`: strongest overall QA baseline
- `routed_grounded`: safe grounded architecture baseline
- `adaptive_route_retry`: experimental branch, not default

Run the gate after benchmark-producing changes:

```powershell
.\.venv\Scripts\python.exe scripts\check_regression_gates.py
```

or:

```bash
make baseline-gate PYTHON=.venv/Scripts/python.exe
```

The gate and current baseline metrics are documented in
`docs/BASELINE_REGRESSION_GATES.md`.

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
