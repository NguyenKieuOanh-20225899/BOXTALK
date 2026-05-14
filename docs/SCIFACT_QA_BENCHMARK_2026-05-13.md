# SciFact QA/Citation Benchmark 2026-05-13

## Purpose

This benchmark adds a public scientific benchmark for checking whether BOXTALK returns grounded scientific answers and cites the correct evidence documents.

The source dataset is BEIR SciFact. Unlike the local QCDT/Attention QA files, SciFact provides public qrels for relevant scientific abstracts. Therefore:

- Citation/evidence correctness is evaluated against official SciFact qrels.
- Answer correctness is evaluated against an evidence sentence derived from the relevant abstract.

This conversion is strongest for citation/evidence evaluation. It is not a replacement for a natural-answer benchmark such as Qasper, because BEIR SciFact is primarily a retrieval/claim-evidence benchmark.

## Files Added

- `scripts/prepare_scifact_qa_benchmark.py`
- `tests/test_prepare_scifact_qa_benchmark.py`

Generated local data:

- `data/benchmarks/scifact_qa/scifact.jsonl`
- `data/benchmarks/scifact_qa/queries_test.jsonl`
- `data/benchmarks/scifact_qa/manifest.json`
- `data/benchmarks/scifact_qa/README.md`

## Dataset

| Field | Value |
|---|---:|
| Source | BEIR SciFact |
| Corpus size | 5,183 scientific abstracts |
| Test query count | 300 |
| Citation gold | Official qrels/test.tsv |
| Answer gold | Evidence sentence from SciFact metadata when available, otherwise best-overlap abstract sentence |

## Commands

```powershell
.\.venv-gpu\Scripts\python.exe scripts\prepare_scifact_qa_benchmark.py `
  --beir-dir data\beir\scifact `
  --output-dir data\benchmarks\scifact_qa `
  --split test

.\.venv-gpu\Scripts\python.exe scripts\build_retrieval_index.py `
  --chunks-jsonl data\benchmarks\scifact_qa\scifact.jsonl `
  --output-dir results\retrieval_index\scifact_qa_minilm_20260513 `
  --dense-preset minilm `
  --dense-device cuda

.\.venv-gpu\Scripts\python.exe scripts\benchmark_qa.py `
  --index-dir results\retrieval_index\scifact_qa_minilm_20260513 `
  --queries data\benchmarks\scifact_qa\queries_test.jsonl `
  --output-dir results\qa_benchmark\scifact_qa_minilm_heuristic_20260513 `
  --config routed_grounded `
  --no-warmup

.\.venv-gpu\Scripts\python.exe scripts\benchmark_retrieval.py `
  --index-dir results\retrieval_index\scifact_qa_minilm_20260513 `
  --queries data\benchmarks\scifact_qa\queries_test.jsonl `
  --output-dir results\retrieval_benchmark\scifact_qa_minilm_20260513 `
  --top-k 5 `
  --strategy all
```

## Results

### Retrieval/Citation

| Strategy | hit@5 | recall@5 | MRR@5 | NDCG@5 |
|---|---:|---:|---:|---:|
| BM25 | 0.713 | 0.693 | 0.569 | 0.594 |
| Dense MiniLM | 0.713 | 0.697 | 0.580 | 0.604 |
| Hybrid | 0.793 | 0.771 | 0.654 | 0.675 |
| Hybrid rerank | 0.790 | 0.768 | 0.658 | 0.678 |

### QA End-to-End

| Metric | Value |
|---|---:|
| query_count | 300 |
| answer_rate | 0.810 |
| answer_match_rate | 0.220 |
| evidence_match_rate | 0.727 |
| grounded_rate | 1.000 |
| hallucination_rate | 0.000 |
| end_to_end_success_rate | 0.203 |
| avg_answer_token_f1 | 0.271 |

## Interpretation

The system retrieves/cites an official relevant SciFact document for about 72.7% of test claims in the QA run. Hybrid retrieval reaches 79.3% hit@5 in the retrieval-only benchmark.

Answer-match is much lower because the generated answer often uses a different sentence from the same relevant abstract than the single evidence sentence selected as `gold_answer`. This is a strict conversion artifact, not necessarily hallucination: `grounded_rate = 1.000` and `hallucination_rate = 0.000`.

For the thesis, this benchmark should be described as:

> A public scientific citation/evidence benchmark derived from BEIR SciFact, used to evaluate whether the system retrieves and cites correct scientific abstracts. Answer matching is reported as an auxiliary metric because SciFact is not originally a natural-answer QA benchmark.

## Limitation

QASPER has now been added as the complementary public benchmark for natural scientific QA. QASPER is more appropriate for free-form answer correctness, while SciFact remains stronger for citation/evidence correctness against official qrels.
