# Baseline And Regression Gates

This document locks the current benchmark interpretation so later retrieval or
QA changes can be checked against a stable baseline.

## Current Baselines

Source artifacts:

- `results/user_pdf_benchmark_suite/current/suite_summary.json`
- `results/retrieval_readiness/20260420T150853Z/readiness_report.json`
- `results/qa_benchmark/qa_operations_minilm_all/qa_summary.json`

User PDF suite snapshot:

| Config | Role | Success | Evidence | Grounded | Hallucination | Latency ms |
|---|---|---:|---:|---:|---:|---:|
| `bm25_only` | strongest fast baseline | 0.825 | 0.942 | 1.000 | 0.010 | 2.4 |
| `routed_grounded` | production-like grounded architecture baseline | 0.845 | 1.000 | 1.000 | 0.000 | 12.6 |
| `adaptive_route_retry` | experimental branch | 0.816 | 0.990 | 1.000 | 0.000 | 18.6 |

Scientific paper QA snapshot:

| Config | Success | Evidence | Grounded | Hallucination | Latency ms |
|---|---:|---:|---:|---:|---:|
| `bm25_only` | 0.739 | 0.783 | 1.000 | 0.000 | 3.6 |
| `routed_grounded` | 1.000 | 1.000 | 1.000 | 0.000 | 17.4 |
| `adaptive_route_retry` | 0.913 | 0.957 | 1.000 | 0.000 | 29.6 |

Controlled QA snapshot on `operations_handbook_en`:

| Config | Success | Evidence | Grounded | Hallucination |
|---|---:|---:|---:|---:|
| `dense_only` | 0.950 | 1.000 | 1.000 | 0.000 |
| `bm25_only` | 0.925 | 0.975 | 1.000 | 0.025 |
| `routed_grounded` | 0.925 | 0.975 | 1.000 | 0.025 |

Scientific ingest readiness:

| PubTables limit | Success | IoU50 F1 | IoU75 F1 | P95 latency |
|---:|---:|---:|---:|---:|
| 25 | 1.000 | 1.000 | 0.818 | 0.844s |
| 100 | 1.000 | 0.988 | 0.870 | 0.757s |
| 500 | 1.000 | 0.977 | 0.910 | 0.744s |

The current interpretation is:

- `bm25_only` remains the strongest fast baseline.
- `routed_grounded` is now the strongest grounded architecture baseline.
- `adaptive_route_retry` remains experimental because it increases latency
  without a clear aggregate quality win over `routed_grounded`.
- Scientific paper QA is protected as a separate gate because it was the main
  previously weak document type.
- Retrieval is ready for prototyping, but not for production claims until
  labeled production PDFs are added.

## Gates

The default regression gate checks:

- user PDF suite has at least `100` unique questions and `3` documents
- `bm25_only.end_to_end_success_rate >= 0.82`
- `routed_grounded.end_to_end_success_rate >= 0.83`
- `routed_grounded.grounded_rate >= 1.0`
- `routed_grounded.hallucination_rate <= 0.0`
- `scientific_paper / routed_grounded.end_to_end_success_rate >= 0.95`
- `scientific_paper / routed_grounded.evidence_match_rate >= 0.95`
- `scientific_paper / routed_grounded.hallucination_rate <= 0.0`
- scientific readiness verdict is true for prototyping
- all scientific ingest gates in the latest readiness report pass

Run:

```powershell
.\.venv\Scripts\python.exe scripts\check_regression_gates.py
```

or:

```bash
make baseline-gate PYTHON=.venv/Scripts/python.exe
```

Use `--require-production-ready` only after the production labeled PDF benchmark
exists and should become a hard release gate.
