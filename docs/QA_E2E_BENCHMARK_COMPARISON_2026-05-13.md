# QA E2E Benchmark Comparison 2026-05-13

## Scope

This pass re-ran validation, ingest mock, retrieval smoke, QA smoke, and QA end-to-end benchmarks after improving table answer generation.

LLM fallback was not enabled for the main routed QA runs. The reported main results therefore reflect retrieval + grounded extractive/rule-based QA.

## Commands Run

```powershell
.\.venv-gpu\Scripts\python.exe -m compileall app scripts
.\.venv-gpu\Scripts\python.exe -m pytest -q

$env:BOXBIIBOO_TABLE_BACKEND='hybrid_tatr'
.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset mock --limit 5 --out results\ingest\mock_after_table_answer_20260513 --mode all

.\.venv-gpu\Scripts\python.exe scripts\benchmark_retrieval.py --index-dir results\retrieval_index\smoke_hybrid_tatr_bm25 --queries data\retrieval_smoke\queries.jsonl --output-dir results\retrieval_benchmark\smoke_hybrid_tatr_after_table_answer_20260513 --top-k 5 --strategy all

.\.venv-gpu\Scripts\python.exe scripts\benchmark_qa.py --index-dir results\retrieval_index\smoke_hybrid_tatr_bm25 --queries data\retrieval_smoke\queries.jsonl --output-dir results\qa_benchmark\smoke_hybrid_tatr_after_table_answer_20260513 --config all --reranker none --no-warmup

.\.venv-gpu\Scripts\python.exe scripts\benchmark_qa.py --index-dir results\retrieval_index\real_qcdt_e2e_hybrid_tatr_20260513 --queries data\real_pdfs\queries.jsonl --output-dir results\qa_benchmark\real_qcdt_all_after_table_answer_20260513 --config all --reranker none --no-warmup

.\.venv-gpu\Scripts\python.exe scripts\build_retrieval_index.py --pdf data\real_pdfs\1706.03762v7.pdf --output-dir results\retrieval_index\real_attention_after_table_answer_20260513 --dense-preset minilm --dense-device cuda
.\.venv-gpu\Scripts\python.exe scripts\benchmark_qa.py --index-dir results\retrieval_index\real_attention_after_table_answer_20260513 --queries data\real_pdfs\attention_queries.jsonl --output-dir results\qa_benchmark\real_attention_all_after_table_answer_20260513 --config all --reranker none --no-warmup

.\.venv-gpu\Scripts\python.exe scripts\benchmark_qa.py --index-dir results\retrieval_index\real_attention_1706_03762 --queries data\real_pdfs\attention_queries.jsonl --output-dir results\qa_benchmark\real_attention_old_index_heuristic_after_table_answer_20260513 --config routed_grounded --no-warmup

.\.venv-gpu\Scripts\python.exe scripts\benchmark_qa.py --index-dir results\retrieval_index\qa_operations_minilm --queries data\qa_benchmark\queries.jsonl --output-dir results\qa_benchmark\operations_all_heuristic_after_table_answer_20260513 --config all --no-warmup
```

## Validation

| Check | Result |
|---|---:|
| compileall app/scripts | pass |
| pytest | 49 passed |
| ingest mock success_rate | 1.000 |
| ingest mock text token F1 | 1.000 |
| ingest mock table structure F1 | 1.000 |

## Ingest Benchmark Recheck

| Benchmark | Key metric | Before | After | After success |
|---|---|---:|---:|---:|
| Bast & Korzen proxy | token F1 | - | 0.998 | 1.000 |
| DocLayNet 25 | layout F1@IoU 0.50 | 0.815 | 0.879 | 1.000 |
| PubLayNet 25 | layout F1@IoU 0.50 | 0.771 | 0.778 | 1.000 |
| PubTables detection 25 | table detection F1@IoU 0.50 | - | 0.987 | 1.000 |
| PubTables structure OCR-words 25 | table structure F1 | 0.638 | 0.638 | 1.000 |
| OCR scan 25 | OCR token F1 | - | 1.000 | 1.000 |
| Nougat/arXiv proxy 25 | token F1 | - | 0.628 | 1.000 |

DocLayNet/PubLayNet/PubTables detection require `BOXBIIBOO_LAYOUT_MODEL_NAME=default`.
Runs without that environment variable fail by design with `Model layout backend is disabled`.

## Main QA Comparison

| Benchmark | Before answer match | After answer match | Delta | Before E2E | After E2E | Before hallucination | After hallucination | After table success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| QA smoke routed | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 |
| QCDT routed, same 2026-05-13 index | 0.725 | 0.725 | +0.000 | 0.725 | 0.725 | 0.000 | 0.000 | 1.000 |
| QCDT routed, older baseline index | 0.675 | 0.725 | +0.050 | 0.675 | 0.725 | 0.000 | 0.000 | 1.000 |
| Attention routed, comparable old text index + heuristic reranker | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | 0.000 | 0.000 | 1.000 |
| Attention rebuilt with region_routed + no reranker | 1.000 | 0.870 | -0.130 | 1.000 | 0.870 | 0.000 | 0.000 | 0.000 |
| Operations routed | 0.925 | 0.925 | +0.000 | 0.925 | 0.925 | 0.025 | 0.000 | 1.000 |
| SciFact public scientific QA/citation | - | 0.220 | - | - | 0.203 | - | 0.000 | - |
| QASPER public scientific natural QA | - | 0.100 | - | - | 0.020 | - | 0.050 | - |

SciFact is not directly comparable with the local PDF QA rows because it is a public claim-evidence benchmark converted into QA format. Its strongest metric here is citation/evidence correctness: `evidence_match_rate = 0.727` against official BEIR SciFact qrels. See `docs/SCIFACT_QA_BENCHMARK_2026-05-13.md`.

QASPER is also not directly comparable with the local PDF rows because it is distributed as paper text, not PDF pages. It is more natural than SciFact for scientific QA, but harder for the current extractive/rule-based QA path: `answer_match_rate = 0.100`, `evidence_match_rate = 0.360`, `grounded_rate = 1.000`, and `hallucination_rate = 0.050` on the 100-question validation subset. See `docs/QASPER_QA_BENCHMARK_2026-05-13.md`.

## QCDT All Configs After

| Config | Answer match | E2E success | Grounded | Hallucination | Table success |
|---|---:|---:|---:|---:|---:|
| bm25_only | 0.800 | 0.800 | 1.000 | 0.000 | 1.000 |
| dense_only | 0.350 | 0.350 | 1.000 | 0.000 | 1.000 |
| hybrid_no_routing | 0.675 | 0.650 | 1.000 | 0.000 | 1.000 |
| routed_grounded | 0.725 | 0.725 | 1.000 | 0.000 | 1.000 |
| adaptive_route_retry | 0.750 | 0.750 | 1.000 | 0.000 | 1.000 |
| routed_grounded_with_llm_fallback | 0.725 | 0.725 | 1.000 | 0.000 | 1.000 |
| routed_grounded_with_table_llm | 0.725 | 0.725 | 1.000 | 0.000 | 1.000 |
| no_citation_grounding | 0.725 | 0.000 | 0.000 | 1.000 | 0.000 |

## Table Answer Output

The smoke table answer became cleaner:

| Run | Answer |
|---|---|
| Before first table-answer tightening | `VPN access is owned by IT Support Benefits.` |
| After tightening | `VPN access is owned by IT Support.` |

QCDT real table questions remain correct:

| Query | Answer |
|---|---|
| q05 | `4 năm và 132 tín chỉ.` |
| q06 | `1,5 năm và 48 tín chỉ.` |

## Notes

- The table-answer change improves output formatting/precision without enabling LLM fallback.
- On comparable indexes/configs, QA E2E did not regress.
- The Attention rebuilt-index result dropped because the new rebuild used `region_routed` and was run with `--reranker none`; the comparable old text index with heuristic reranker remains 1.000.
- `no_citation_grounding` intentionally shows poor E2E/grounded behavior, confirming citation grounding is still a necessary QA component.
