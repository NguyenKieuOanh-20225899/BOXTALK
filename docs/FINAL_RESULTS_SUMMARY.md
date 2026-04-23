# Final Results Summary

This document summarizes the current end-to-end state of the project based on
the latest code and benchmark artifacts in the repo.

## 1. Current System Overview

### Main architecture

- Main QA path: `routed_grounded`
- Strong lexical baseline: `bm25_only`
- Experimental branches:
  - `adaptive_route_retry`
  - `grounded_llm_fallback`
  - `adaptive_route_retry_with_final_route_llm_fallback`

### Current product surface

The repo now has:

- PDF ingest with text/layout/OCR/mixed paths
- retrieval stack: BM25, dense, hybrid, rerank
- grounded QA with evidence checker and citations
- adaptive retry as a research branch
- opt-in grounded LLM fallback by modality:
  - `text`
  - `table`
  - `formula`
  - `figure/caption` via textual fallback
  - `multi_span`
- benchmark and regression gate infrastructure
- MVP UI served by FastAPI:
  - upload PDF
  - document library
  - grounded QA
  - source viewer
  - developer trace toggle

### Current interpretation

- `routed_grounded` is the production-like grounded answer path.
- `bm25_only` remains the strongest cheap lexical baseline.
- `adaptive_route_retry` is still experimental because of latency and mixed aggregate benefit.
- `grounded_llm_fallback` is experimental and currently useful mainly on focused table-style cases.

## 2. Source Artifacts Used For This Summary

- `results/user_pdf_benchmark_suite/llm_fallback_gate_recheck/suite_summary.json`
- `results/user_pdf_benchmark_suite/current/suite_summary.json`
- `results/user_pdf_benchmark_suite/attention_smoke/documents/attention_scientific_en/qa_summary.json`
- `results/qa_benchmark/attention_scientific_focus/qa_summary.json`
- `results/llm_fallback_benchmark/dummy_smoke/comparison_summary.json`
- `results/qa_benchmark/llm_fallback_smoke/qa_summary.json`
- `results/retrieval_readiness/20260420T150853Z/readiness_report.json`
- `results/retrieval_benchmark/smoke_bm25_only/benchmark_summary.json`
- `results/retrieval_benchmark/smoke_real_minilm_after/benchmark_summary.json`
- `results/beir_retrieval_benchmark/scifact_*/beir_summary.json`
- `docs/BASELINE_REGRESSION_GATES.md`

## 3. Main Benchmark Results

### A. Ingest

Scientific ingest readiness is currently based on the GPU `model_routed_doclaynet`
path and is stable on PubTables samples.

| PubTables limit | Success | IoU@0.50 micro F1 | IoU@0.75 micro F1 | P95 latency |
|---:|---:|---:|---:|---:|
| 25 | 1.000 | 1.000 | 0.818 | 0.844s |
| 100 | 1.000 | 0.988 | 0.870 | 0.757s |
| 500 | 1.000 | 0.977 | 0.910 | 0.744s |

Key takeaways:

- Scientific ingest passed all readiness gates in the current GPU path.
- Dominant backend is stable: `model_layout`.
- Scientific readiness verdict is `true`.
- Production ingest is still not claim-ready because labeled production PDFs are not yet available.

Note:

- A separate baseline-only PubTables run in the current environment failed because of OCR/runtime issues, so the retained ingest baseline for reporting is the stable `model_routed_doclaynet` readiness path rather than that failing local baseline reference.

### B. Retrieval

#### Controlled smoke retrieval

`results/retrieval_benchmark/smoke_bm25_only/benchmark_summary.json`

| Strategy | Hit@5 | Recall@5 | MRR@5 | NDCG@5 | Avg latency |
|---|---:|---:|---:|---:|---:|
| `bm25` | 1.000 | 1.000 | 0.900 | 0.926 | 0.055 ms |

`results/retrieval_benchmark/smoke_real_minilm_after/benchmark_summary.json`

| Strategy | Hit@5 | Recall@5 | MRR@5 | NDCG@5 | Avg latency |
|---|---:|---:|---:|---:|---:|
| `bm25` | 1.000 | 1.000 | 0.900 | 0.926 | 0.152 ms |
| `dense` | 1.000 | 1.000 | 1.000 | 1.000 | 7.892 ms |
| `hybrid` | 1.000 | 1.000 | 0.900 | 0.926 | 6.708 ms |
| `hybrid_rerank` | 1.000 | 1.000 | 0.900 | 0.926 | 7.894 ms |

Interpretation:

- BM25 is still the fastest lexical baseline.
- Dense MiniLM gives the best ranking quality on the small controlled smoke set.

#### BEIR / SciFact sample

| Backend / strategy | Hit@10 | Recall@10 | MRR@10 | NDCG@10 | Avg latency |
|---|---:|---:|---:|---:|---:|
| `bm25` | 0.950 | 0.950 | 0.826 | 0.844 | 3.61 ms |
| `contriever dense` | 0.750 | 0.750 | 0.667 | 0.691 | 38.69 ms |
| `contriever hybrid` | 0.950 | 0.950 | 0.846 | 0.872 | 55.41 ms |
| `dpr dense` | 0.400 | 0.375 | 0.335 | 0.343 | 207.64 ms |
| `dpr hybrid` | 0.950 | 0.925 | 0.882 | 0.870 | 60.43 ms |
| `colbert` (10-query sample) | 0.800 | 0.800 | 0.720 | 0.731 | 1892.17 ms |

Interpretation:

- BM25 is already strong on SciFact as a cheap baseline.
- Best ranking on the current SciFact sample comes from hybrid retrieval, especially DPR hybrid by MRR and Contriever hybrid by NDCG.
- ColBERT is currently too slow and not yet competitive enough on the sampled setup.

#### Retrieval readiness verdict

From `results/retrieval_readiness/20260420T150853Z/readiness_report.json`:

- `retrieval_ready_for_prototyping = true`
- `retrieval_ready_for_production = false`

Reason:

- Retrieval is strong enough for prototyping and demos.
- Production-ready claims are blocked by the lack of labeled production PDFs.

### C. QA

#### Latest official gate recheck

`results/user_pdf_benchmark_suite/llm_fallback_gate_recheck/suite_summary.json`

| Config | Role | Success | Answer match | Evidence | Grounded | Hallucination | Avg latency |
|---|---|---:|---:|---:|---:|---:|---:|
| `bm25_only` | strong lexical baseline | 0.835 | 0.835 | 0.942 | 1.000 | 0.010 | 4.18 ms |
| `routed_grounded` | main grounded path | 0.864 | 0.864 | 1.000 | 1.000 | 0.000 | 18.51 ms |

Aggregate interpretation:

- `routed_grounded` is now the strongest overall QA path on the latest official recheck.
- `bm25_only` remains strong and especially useful for Vietnamese policy/regulation.
- Groundedness remains fully controlled on `routed_grounded`.

#### Earlier locked baseline snapshot

From `docs/BASELINE_REGRESSION_GATES.md` and `results/user_pdf_benchmark_suite/current/suite_summary.json`:

| Config | Success | Grounded | Hallucination | Avg latency |
|---|---:|---:|---:|---:|
| `bm25_only` | 0.825 | 1.000 | 0.010 | 2.4 ms |
| `routed_grounded` | 0.845 | 1.000 | 0.000 | 12.6 ms |
| `adaptive_route_retry` | 0.816 | 1.000 | 0.000 | 18.6 ms |

Interpretation:

- `adaptive_route_retry` improved some cases but did not win clearly enough to become default.
- Later rechecks improved the two main baselines further while keeping gates green.

#### QA by document type on latest recheck

| Document type | `bm25_only` success | `routed_grounded` success | Winner |
|---|---:|---:|---|
| `policy_regulation` | 0.800 | 0.725 | `bm25_only` |
| `handbook_manual` | 0.925 | 0.925 | tie |
| `scientific_paper` | 0.739 | 1.000 | `routed_grounded` |

This is the cleanest current summary of where each path is strongest.

### D. Scientific Paper QA

This was the main previous weakness and is now the clearest QA improvement area.

#### Before improvement

From `results/user_pdf_benchmark_suite/attention_smoke/documents/attention_scientific_en/qa_summary.json`:

| Config | Success | Answer match | Evidence | Grounded | Hallucination |
|---|---:|---:|---:|---:|---:|
| `routed_grounded` | 0.478 | 0.478 | 0.870 | 1.000 | 0.000 |
| `adaptive_route_retry` | 0.478 | 0.478 | 0.870 | 1.000 | 0.000 |

#### After improvement

From `results/qa_benchmark/attention_scientific_focus/qa_summary.json` and the latest suite snapshots:

| Config | Success | Answer match | Evidence | Grounded | Hallucination |
|---|---:|---:|---:|---:|---:|
| `bm25_only` | 0.739 | 0.739 | 0.783 | 1.000 | 0.000 |
| `routed_grounded` | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

Net result:

- `routed_grounded` on the scientific paper benchmark improved from `0.478` to `1.000` end-to-end success.
- Evidence quality also improved from `0.870` to `1.000`.
- This is the strongest concrete QA improvement in the project.

### E. LLM Fallback

#### Focused fallback benchmark, dummy provider

From `results/llm_fallback_benchmark/dummy_smoke/comparison_summary.json`:

| Metric | Standard (`routed_grounded`) | Fallback config |
|---|---:|---:|
| Success rate | 0.640 | 0.720 |
| Answer match rate | 0.640 | 0.720 |
| Fallback call rate | - | 0.240 |
| Fallback used rate | - | 0.080 |
| Helped count | - | 2 |
| Override count | - | 2 |
| Hallucination delta | - | 0.000 |

Important modality breakdown:

- Table questions:
  - standard success: `0.444`
  - fallback success: `0.667`
  - gain: `+0.222`
  - `table_rule_resolved_count = 2`
  - `table_llm_resolved_count = 0`

- Queries marked `should_require_fallback = true`:
  - standard success: `0.556`
  - fallback success: `0.667`
  - gain: `+0.111`

- Weak-standard-answer cases:
  - standard success: `0.400`
  - fallback success: `0.600`
  - gain: `+0.200`

Interpretation:

- Current measured gain exists, but in the dummy benchmark it comes from the rule-based table path rather than a real provider.
- The LLM plumbing, policy, trace, and benchmark infrastructure are ready.
- Real-provider gain is still pending stable benchmarking.

#### Controlled handbook smoke with fallback enabled

From `results/qa_benchmark/llm_fallback_smoke/qa_summary.json`:

- `routed_grounded`: success `0.925`
- `routed_grounded_with_llm_fallback`: success `0.925`
- fallback call rate `0.325`
- fallback used rate `0.000`
- hallucination stayed `0.000`

Interpretation:

- On the normal controlled handbook benchmark, fallback currently stays safe and non-disruptive.
- Real measured gain is therefore not yet general-purpose; it is still focused on targeted fallback cases.

## 4. Baseline And Gate Status

### Baseline lock history

- Initial baseline lock commit: `5413c70` - `Lock retrieval QA baseline and gates`
- Gate raise commit: `edad8f9` - `Raise QA baseline regression gates`
- Current code/UI/fallback snapshot: `b0a0f15` - `Add grounded fallback benchmarks and MVP PDF QA UI`

### Current hard gates

From `docs/BASELINE_REGRESSION_GATES.md`:

- user PDF suite has at least `100` unique questions and `3` documents
- `bm25_only.end_to_end_success_rate >= 0.82`
- `routed_grounded.end_to_end_success_rate >= 0.83`
- `routed_grounded.grounded_rate >= 1.0`
- `routed_grounded.hallucination_rate <= 0.0`
- `scientific_paper / routed_grounded.end_to_end_success_rate >= 0.95`
- `scientific_paper / routed_grounded.evidence_match_rate >= 0.95`
- `scientific_paper / routed_grounded.hallucination_rate <= 0.0`
- scientific readiness verdict must pass

### Latest gate status

Using `results/user_pdf_benchmark_suite/llm_fallback_gate_recheck/suite_summary.json`:

| Gate | Threshold | Latest result | Status |
|---|---:|---:|---|
| `bm25_only` success | 0.82 | 0.835 | pass |
| `routed_grounded` success | 0.83 | 0.864 | pass |
| `routed_grounded` grounded | 1.00 | 1.000 | pass |
| `routed_grounded` hallucination | 0.00 max | 0.000 | pass |
| scientific paper `routed_grounded` success | 0.95 | 1.000 | pass |
| scientific paper `routed_grounded` evidence | 0.95 | 1.000 | pass |
| scientific paper `routed_grounded` hallucination | 0.00 max | 0.000 | pass |

Overall status:

- Main gates are currently passing.

## 5. Main Conclusions

1. `bm25_only` is still the strongest baseline for Vietnamese policy/regulation documents.
2. `routed_grounded` is the strongest overall QA architecture on the multi-document user suite.
3. Scientific paper QA improved sharply and is now a strength rather than the main weakness.
4. Groundedness and hallucination control are currently strong on the main path:
   - `grounded_rate = 1.0`
   - `hallucination_rate = 0.0` for `routed_grounded`
5. Retrieval is strong enough for prototyping across BM25, dense, and hybrid modes, but not yet ready for production claims.
6. The LLM fallback stack is now benchmarkable, traceable, and UI-ready, but still experimental.
7. Current fallback gain is real on the focused benchmark, but mostly from the table rule-first path rather than a real LLM provider.
8. The repo now has a usable MVP UI on top of the grounded QA stack.

## 6. Current Limitations

- Real-provider LLM fallback has plumbing and benchmarking support, but not yet a stable benchmark result that is strong enough to lock into a gate.
- `figure` handling is still textual fallback only; there is no vision-grounded answer path yet.
- Table rule-based support currently covers simpler lookup/range mapping cases better than harder table reasoning.
- `adaptive_route_retry` is still experimental and is not the default main path.
- UI is now usable, but still MVP quality rather than fully polished product UI.
- Production ingest/retrieval claims are blocked by the absence of labeled production PDFs.
- Some older baseline-only scientific ingest runs failed due local OCR/runtime environment issues, so the scientific readiness baseline is effectively the stable GPU `model_routed_doclaynet` path.

## 7. Recommended Next Steps

1. Run a stable real-provider benchmark for `grounded_llm_fallback`.
2. Strengthen table QA beyond simple rule-based lookup and interval mapping.
3. Surface fallback usage more clearly in the UI developer view and lightly in user mode when useful.
4. Keep adaptive integration limited to final-route-only until real-provider fallback quality is measured.
5. Improve figure textual packaging:
   - caption
   - nearby paragraph
   - figure references
   - evidence packet quality
6. Add labeled production PDFs to support stronger production-readiness claims.
7. If real-provider fallback proves stable, add a separate experimental gate for the focused fallback benchmark rather than folding it into the main gate immediately.

## 8. Short Report Version

If only one slide or one paragraph is needed:

> The project currently uses `routed_grounded` as the main grounded QA path, with `bm25_only` kept as the strongest lexical baseline and `adaptive_route_retry` plus `grounded_llm_fallback` kept experimental. Scientific ingest is stable on PubTables 25/100/500 with 100% success and strong IoU/F1 under the retained GPU path. On the latest user PDF suite recheck, `bm25_only` achieved `0.835` success and `routed_grounded` achieved `0.864` with `grounded_rate = 1.0` and `hallucination_rate = 0.0`. The largest QA improvement was scientific paper QA, where `routed_grounded` improved from `0.478` to `1.000` success. The LLM fallback layer is now benchmarkable and safe, with current dummy-provider gains concentrated in focused table-style cases, but it is still experimental and not yet part of the main release gate.
