# Thesis Readiness Report

Generated at UTC: `20260505T071906Z`

Research topic: `Nghiên cứu các kĩ thuật truy xuất và hỏi đáp thông tin trên tài liệu PDF`.

This report maps current benchmark artifacts to thesis-level claims. It is intentionally conservative: main QA claims stay separate from experimental grounded LLM fallback and table reasoning.

## Verdict

- Research prototype readiness: **PASS**
- Production-readiness claim: **NOT READY**
- Recommended positioning: Ready to position as a research prototype for retrieval and grounded QA over PDFs, with LLM fallback/table reasoning documented as experimental.

## Evidence Map

| Area | Status | Key evidence | Source |
| --- | --- | --- | --- |
| Main grounded QA | PASS | routed success 86.4%, answer match 86.4%, grounded 100.0%, hallucination 0.0%; bm25 success 83.5% | `results\user_pdf_benchmark_suite\llm_fallback_gate_recheck\suite_summary.json` |
| Retrieval comparison | PASS | bm25 recall@5 100.0%; dense recall@5 100.0%; hybrid recall@5 100.0%; hybrid_rerank recall@5 100.0% | `results\retrieval_benchmark\smoke_real_minilm_after\benchmark_summary.json` |
| Scientific/table ingest | PASS | runs 3, success min 100.0%, IoU50 min 97.7%, IoU75 min 81.8%, p95 max 0.844s | `results\retrieval_readiness\20260420T150853Z\readiness_report.json` |
| Ingest PDF component benchmarks | PASS | PubTables F1@0.50 0.987; DocLayNet micro F1@0.50 0.815; PubLayNet micro F1@0.50 0.771; FUNSD OCR token F1 0.749; OCR-D token F1 0.657; Nougat/arXiv token F1 0.628 | `docs\INGEST_REAL_BENCHMARK_RUN_2026-05-12.md` |
| External-style retrieval | PASS | 5 BEIR/SciFact sample runs; best bm25 nDCG@k 0.844, recall@k 95.0% | `results\beir_retrieval_benchmark` |
| Experimental LLM fallback | PASS | repeat 3, success gain mean 0.133, groundedness min 1.000, hallucination delta max 0.000, table LLM resolved min 1 | `results\llm_fallback_benchmark\table_patch_ollama_repeats_gpu\repeat_summary.json` |
| Extended table reasoning | PASS | queries 46, table success 58.7%, rule resolved 6, LLM attempts 15, LLM resolved 0 | `results\llm_fallback_benchmark\table_reasoning_ollama_after_shape_gate\comparison_summary.json` |

## What This Supports

- Main routed_grounded QA has strong grounded user-PDF benchmark results.
- Retrieval benchmark compares lexical, dense, hybrid, and rerank-style strategies.
- Scientific/PubTables ingest readiness has stable sampled evidence.
- External-style retrieval evidence exists through BEIR/SciFact samples.
- Experimental grounded LLM fallback has stable gain without groundedness regression.
- Extended table benchmark covers lookup, interval, numerical, and verification cases.

## Limitations To State Clearly

- Extended table benchmark still shows no resolved LLM-table wins; present this as a limitation.
- Probe-classification evaluation artifact is not present.
- Do not claim production readiness without labeled production-PDF evidence.

## Next Required Work Before Final Submission

- Freeze benchmark result folders used in the final report.
- Add limitations/future-work text for table LLM reasoning and production-readiness scope.
- Prepare a reproducible command appendix for all reported metrics.

## Reproducible Commands

```powershell
.\.venv-gpu\Scripts\python.exe scripts\generate_thesis_readiness_report.py
.\.venv-gpu\Scripts\python.exe scripts\check_regression_gates.py
.\.venv-gpu\Scripts\python.exe scripts\create_extended_table_benchmark.py --output-dir data/table_reasoning_benchmark
.\.venv-gpu\Scripts\python.exe scripts\benchmark_llm_fallback.py --manifest data/table_reasoning_benchmark/manifest.json --output-dir results/llm_fallback_benchmark/table_reasoning_ollama_after_shape_gate --llm-fallback-provider ollama --skip-build --no-warmup
.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset pubtables --data-dir data\benchmarks\pubtables_detection --limit 25 --out results\ingest\pubtables_real_cuda_25 --mode table --device cuda --save-predictions
.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset doclaynet --data-dir data\benchmarks\doclaynet --limit 25 --out results\ingest\doclaynet_real_cuda_25 --mode layout --device cuda --save-predictions
.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset publaynet --data-dir data\benchmarks\publaynet --limit 25 --out results\ingest\publaynet_real_cuda_25 --mode layout --device cuda --save-predictions
.\.venv-ocr-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset ocr --data-dir data\benchmarks\funsd\ocr --limit 25 --out results\ingest\funsd_ocr_gpu_25 --mode ocr --save-predictions
.\.venv-ocr-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset ocr --data-dir data\benchmarks\ocrd_pagexml\ocr --limit 19 --out results\ingest\ocrd_pagexml_gpu_19 --mode ocr --save-predictions
.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset nougat --data-dir data\benchmarks\nougat_arxiv_small\text --limit 25 --out results\ingest\nougat_arxiv_text_direct_25 --mode text --save-predictions
```

## Claim Boundary

- Safe claim: the project implements and evaluates a grounded PDF QA prototype with lexical/dense/hybrid retrieval, routed QA, scientific/table-aware ingest evidence, an experimental grounded LLM fallback, and an optional LLM explanation layer that does not change the final answer.
- Unsafe claim right now: production-ready QA, fully solved table reasoning, LLM fallback as the default main path, or LLM explanation as a source of new facts.
