# Retrieval Readiness Benchmark

Purpose:

- prove the ingest stack is strong enough on scientific table detection before retrieval work starts
- separate "good enough for retrieval prototyping" from "ready for production claims"

Current benchmark standard:

1. Run `PubTables test` on GPU with `model_routed_doclaynet` at three sizes: `25`, `100`, `500`.
2. Require stable scientific quality:
   - success rate `>= 0.99`
   - IoU@0.50 micro F1 `>= 0.97`
   - IoU@0.75 micro F1 `>= 0.80`
   - p95 latency `<= 2.0s`
   - consistent backend path across runs
3. Record a production benchmark separately on `data/test_probe`.
4. Mark retrieval as:
   - `ready for prototyping` when scientific gates pass
   - `ready for production claims` only when both scientific gates pass and production data is benchmarked successfully

Why this standard exists:

- retrieval quality depends on ingest quality first
- PubTables gives annotation-backed evidence for table localization stability
- production benchmark checks real repo PDFs, routing, and latency behavior
- the two layers answer different questions and should not be conflated

Report builder:

```powershell
.\\.venv-gpu\\Scripts\\python.exe scripts/build_retrieval_readiness_report.py `
  --scientific-summary results/ingest_benchmark_scientific/<run25>/scientific_summary.json `
  --scientific-summary results/ingest_benchmark_scientific/<run100>/scientific_summary.json `
  --scientific-summary results/ingest_benchmark_scientific/<run500>/scientific_summary.json `
  --production-summary results/ingest_benchmark/<production_run>/benchmark_summary.json `
  --baseline-reference results/ingest_benchmark_scientific/<baseline_run>/scientific_summary.json
```

The builder writes:

- `results/retrieval_readiness/<timestamp>/readiness_report.json`
- `results/retrieval_readiness/<timestamp>/README.md`

Interpretation:

- if scientific passes but production data is missing, retrieval prototyping can start, but end-to-end production claims are blocked
- if `pred_table_nonempty_rate` stays at `0.0`, table extraction text quality still needs follow-up even if localization is strong
