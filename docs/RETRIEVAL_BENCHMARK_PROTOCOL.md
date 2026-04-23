# Retrieval Benchmark Protocol

This repo has a minimal, benchmarkable retrieval stack on top of `ingest_pdf()` chunks.

## Architecture

- `app/ingest/chunker.py` emits `ChunkNode`.
- `scripts/build_retrieval_index.py` adapts `ChunkNode` or saved chunk JSON/JSONL into `DocumentChunkRef`.
- `app/retrieval/` provides BM25, dense, hybrid fusion, optional rerank, index storage, and a small service boundary.

Core modules:

- `app/retrieval/schemas.py`
- `app/retrieval/embedding_backends.py`
- `app/retrieval/bm25_retriever.py`
- `app/retrieval/dense_retriever.py`
- `app/retrieval/colbert_retriever.py`
- `app/retrieval/hybrid_retriever.py`
- `app/retrieval/reranker.py`
- `app/retrieval/index_store.py`
- `app/retrieval/service.py`

## Build Index

From PDF:

```powershell
.\.venv\Scripts\python.exe scripts\build_retrieval_index.py `
  --pdf data\retrieval_smoke\employee_handbook_smoke.pdf `
  --output-dir results\retrieval_index\smoke_test
```

From existing chunk output:

```powershell
.\.venv\Scripts\python.exe scripts\build_retrieval_index.py `
  --chunks-json path\to\chunks.json `
  --output-dir results\retrieval_index\my_index

.\.venv\Scripts\python.exe scripts\build_retrieval_index.py `
  --chunks-jsonl path\to\chunks.jsonl `
  --output-dir results\retrieval_index\my_index
```

Outputs include:

- `corpus.jsonl`
- `bm25_config.json`
- `bm25_tokens.json`
- `dense_embeddings.npy`
- `dense_config.json`
- `dense.faiss` when FAISS is available
- `colbert_embeddings.npz` when `--build-colbert` is enabled
- `colbert_config.json` when `--build-colbert` is enabled
- `index_manifest.json`
- `index_summary.json`

## Paper Baselines

Implemented baselines:

- BEIR: real benchmark runner in `scripts/benchmark_beir_retrieval.py`
- DPR: `--dense-preset dpr-single-nq` or `--dense-preset dpr-multiset`
- Contriever: `--dense-preset contriever`
- ColBERT: `--build-colbert` plus strategy `colbert`, or reranker `--reranker colbert`
- RAG: `app/routed_rag_starter.py` consumes retrieved chunks as grounded evidence

Dense model presets:

```powershell
.\.venv\Scripts\python.exe scripts\build_retrieval_index.py `
  --pdf data\retrieval_smoke\employee_handbook_smoke.pdf `
  --output-dir results\retrieval_index\contriever_test `
  --dense-preset contriever

.\.venv\Scripts\python.exe scripts\build_retrieval_index.py `
  --pdf data\retrieval_smoke\employee_handbook_smoke.pdf `
  --output-dir results\retrieval_index\dpr_test `
  --dense-preset dpr-single-nq
```

ColBERT index:

```powershell
.\.venv\Scripts\python.exe scripts\build_retrieval_index.py `
  --pdf data\retrieval_smoke\employee_handbook_smoke.pdf `
  --output-dir results\retrieval_index\colbert_test `
  --build-colbert

.\.venv\Scripts\python.exe scripts\benchmark_retrieval.py `
  --index-dir results\retrieval_index\colbert_test `
  --queries data\retrieval_smoke\queries.jsonl `
  --strategy colbert
```

BEIR sample:

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

## Query Index

```powershell
.\.venv\Scripts\python.exe scripts\query_retrieval.py `
  --index-dir results\retrieval_index\smoke_test `
  --query "How long do staff have to submit receipts?" `
  --top-k 3 `
  --strategy hybrid `
  --combination rrf
```

Strategies:

- `bm25`
- `dense`
- `colbert`
- `hybrid`
- `hybrid_rerank`

Hybrid fusion supports:

- `weighted_sum`
- `rrf`

## Routed RAG Integration

`app/routed_rag_starter.py` now uses `RetrievalService` and `QueryAwareRetrievalPlanner`.

The planner maps query type to retrieval strategy/config:

- `factoid`: compact hybrid retrieval
- `definition`: balanced hybrid retrieval
- `policy`: BM25-leaning hybrid with rerank
- `procedural`: hybrid with rerank and optional list-like block filtering
- `comparison`: RRF hybrid with rerank
- `multi_hop`: broader RRF hybrid retrieval
- `ambiguous`: balanced RRF hybrid retrieval

Run the API from a prebuilt index:

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

Debug selected retrieval config:

```text
GET /debug/retrieval-plan?question=How%20long%20do%20staff%20have%20to%20submit%20receipts%3F
```

## Benchmark Input

Use JSONL or JSON. Each case can use `question` or `query`.

```json
{
  "query_id": "q1",
  "question": "When must employees badge in?",
  "expected_chunk_ids": ["employee_handbook_smoke.pdf:chunk_00001"],
  "expected_page": 1,
  "expected_section": "Attendance Policy",
  "match_text": "before 09:00",
  "gold_answer": "Employees must badge in before 09:00 on workdays."
}
```

At least one of `expected_chunk_id`, `expected_chunk_ids`, `expected_page`, `expected_section`, or `match_text` should be present.

## Run Benchmark

```powershell
.\.venv\Scripts\python.exe scripts\benchmark_retrieval.py `
  --index-dir results\retrieval_index\smoke_test `
  --queries data\retrieval_smoke\queries.jsonl `
  --output-dir results\retrieval_benchmark\smoke_test `
  --top-k 5 `
  --strategy all
```

Metrics:

- Hit@k
- Recall@k
- MRR@k
- nDCG@k
- average latency
- average retrieval count

Outputs:

- `benchmark_summary.json`
- `per_question.json`
- `per_question.csv`
- `README.md`

## Smoke Dataset

```powershell
.\.venv\Scripts\python.exe scripts\create_retrieval_smoke_dataset.py
.\.venv\Scripts\python.exe scripts\build_retrieval_index.py --pdf data\retrieval_smoke\employee_handbook_smoke.pdf --output-dir results\retrieval_index\smoke_test
.\.venv\Scripts\python.exe scripts\benchmark_retrieval.py --index-dir results\retrieval_index\smoke_test --queries data\retrieval_smoke\queries.jsonl --output-dir results\retrieval_benchmark\smoke_test --top-k 5 --strategy all
```

Smoke gate for retrieval prototyping:

- Hit@5 >= `1.0`
- Recall@5 >= `1.0`
- MRR@5 >= `0.85`

This is only a systems check. Production retrieval quality still requires real PDFs and labeled retrieval queries.
