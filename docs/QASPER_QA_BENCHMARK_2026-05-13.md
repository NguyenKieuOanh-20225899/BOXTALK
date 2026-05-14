# QASPER QA Benchmark 2026-05-13

## Purpose

QASPER is added as a natural scientific QA benchmark for BOXTALK. It complements SciFact:

- SciFact is mainly a claim-evidence retrieval benchmark.
- QASPER asks natural questions over scientific papers and includes answer annotations plus evidence text.

This benchmark is used to evaluate answer correctness, evidence/citation correctness, groundedness and hallucination behavior on scientific paper text.

## Files Added

- `scripts/prepare_qasper_qa_benchmark.py`
- `tests/test_prepare_qasper_qa_benchmark.py`

Generated local data:

- `data/benchmarks/qasper_qa/qasper.jsonl`
- `data/benchmarks/qasper_qa/queries.jsonl`
- `data/benchmarks/qasper_qa/manifest.json`
- `data/benchmarks/qasper_qa/README.md`

## Dataset Handling

The prepare script supports two sources:

- `--source hf`: downloads the official QASPER v0.3 tarball from the URL used by the Hugging Face dataset script. This path does not require the `datasets` package.
- `--source json --input-file ...`: reads a local original-format QASPER JSON file.

QASPER is not a PDF page benchmark. The script converts paper title, abstract and full-text section paragraphs into retrieval chunks. Citations therefore point to pseudo pages/sections/chunks rather than real PDF pages.

## Commands

```powershell
.\.venv-gpu\Scripts\python.exe scripts\prepare_qasper_qa_benchmark.py `
  --output-dir data\benchmarks\qasper_qa `
  --split validation `
  --limit 100 `
  --seed 42

.\.venv-gpu\Scripts\python.exe scripts\build_retrieval_index.py `
  --chunks-jsonl data\benchmarks\qasper_qa\qasper.jsonl `
  --output-dir results\retrieval_index\qasper_qa_minilm_20260513 `
  --dense-preset minilm `
  --dense-device cuda

.\.venv-gpu\Scripts\python.exe scripts\benchmark_qa.py `
  --index-dir results\retrieval_index\qasper_qa_minilm_20260513 `
  --queries data\benchmarks\qasper_qa\queries.jsonl `
  --output-dir results\qa_benchmark\qasper_qa_minilm_20260513 `
  --config routed_grounded `
  --no-warmup
```

## Metrics

The QA benchmark now supports QASPER-specific fields without breaking SciFact:

- `gold_answers`: multiple gold answer annotations.
- `gold_evidence_texts`: evidence strings used for approximate evidence matching if chunk id mapping is unavailable.
- `should_answer=false`: unanswerable QASPER questions.

Reported metrics include:

- `answer_match_rate`
- `evidence_match_rate`
- `grounded_rate`
- `hallucination_rate`
- `end_to_end_success_rate`
- `answerable_success_rate`
- `unanswerable_success_rate`
- `abstain_accuracy`

## Results

Run: validation split, `limit=100`, seed `42`, `routed_grounded`, MiniLM dense index on CUDA.

### Dataset Subset

| Field | Value |
|---|---:|
| Paper count | 82 |
| Chunk count | 3,630 |
| Query count | 100 |
| Answerable questions | 95 |
| Unanswerable questions | 5 |
| Evidence mapped to chunk | 90 |

### Retrieval-Only

| Strategy | hit@5 | recall@5 | MRR@5 | NDCG@5 |
|---|---:|---:|---:|---:|
| BM25 | 0.390 | 0.331 | 0.273 | 0.404 |
| Dense MiniLM | 0.280 | 0.235 | 0.155 | 0.276 |
| Hybrid | 0.390 | 0.339 | 0.230 | 0.352 |
| Hybrid rerank | 0.400 | 0.336 | 0.252 | 0.363 |

Additional retrieval-only probes with a larger cutoff:

| Strategy | hit@10 | recall@10 | hit@20 | recall@20 |
|---|---:|---:|---:|---:|
| BM25 | 0.490 | 0.423 | 0.570 | 0.502 |
| Dense MiniLM | 0.350 | 0.307 | 0.420 | 0.369 |
| Hybrid | 0.510 | 0.439 | 0.550 | 0.503 |
| Hybrid rerank | 0.520 | 0.451 | 0.580 | 0.530 |

### QA End-to-End

| Metric | Value |
|---|---:|
| query_count | 100 |
| answerable_count | 95 |
| unanswerable_count | 5 |
| answer_match_rate | 0.100 |
| evidence_match_rate | 0.360 |
| grounded_rate | 1.000 |
| hallucination_rate | 0.050 |
| end_to_end_success_rate | 0.020 |
| answerable_success_rate | 0.021 |
| unanswerable_success_rate | 0.000 |
| abstain_accuracy | 0.000 |

Fixed hybrid QA probes with larger `top_k`:

| Config | top_k | answer_match | evidence_match | grounded | hallucination | E2E |
|---|---:|---:|---:|---:|---:|---:|
| hybrid_no_routing | 10 | 0.090 | 0.470 | 1.000 | 0.050 | 0.040 |
| hybrid_no_routing | 20 | 0.090 | 0.510 | 1.000 | 0.050 | 0.040 |

## Interpretation

QASPER is substantially harder for the current extractive/rule-based QA path than SciFact:

- Retrieval is harder because each paper is chunked into many section paragraphs. The best QASPER hit@5 is only 0.400 on this subset.
- Answer matching is harder because QASPER has free-form and extractive answers, while the current answer generator often returns a grounded evidence sentence rather than the exact annotated answer phrase.
- `grounded_rate = 1.000` still shows the system did not generate unsupported answers for answered cases.
- `hallucination_rate = 0.050` comes from unanswerable QASPER questions where the current pipeline still answered instead of abstaining.
- Increasing retrieval cutoff helps citation/evidence recall (`hybrid_rerank hit@20 = 0.580`), but it does not materially solve answer correctness. The current `routed_grounded` QA path does not expose a direct top-k override; top-k/rerank tuning should be treated as a controlled future improvement rather than merged into the default path from this small subset.

For the thesis, report QASPER as a natural scientific QA stress test, not as a solved benchmark. It is useful because it exposes the next work items: better QASPER-style retrieval over long papers, better answer synthesis, and stronger abstention handling.

## Limitations

- QASPER citation is section/chunk-based, not page-based.
- Free-form answer annotations make exact answer matching stricter and noisier than retrieval metrics.
- Evidence mapping is approximate because evidence text may not exactly match chunk boundaries.
- The subset is small when `--limit 100` is used; final reporting should state the subset size clearly.
