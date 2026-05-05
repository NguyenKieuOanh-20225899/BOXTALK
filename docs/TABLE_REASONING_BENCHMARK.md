# Extended Table Reasoning Benchmark

This benchmark is a small internal dataset for PDF grounded table QA. It is
inspired by WikiTableQuestions, TAT-QA, and TabFact, but it is not a copy or
full adapter of those datasets.

## Scope

The benchmark covers:

- `simple_lookup`
- `reverse_lookup`
- `interval_mapping`
- `multi_column_lookup`
- `boundary_case`
- `table_text_reasoning`
- `numerical_reasoning`
- `fact_verification`

Each query carries:

- `benchmark_family`: `wikitable_like`, `tatqa_like`, or `tabfact_like`
- `table_reasoning_type`
- `expected_modality`: `table` or `table_text`
- `expected_fallback_mode`: `table` or `multi_span`
- grounded evidence expectations through `expected_chunk_ids` and `expected_pages`

## Commands

Create the benchmark data:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\create_extended_table_benchmark.py --output-dir data\table_reasoning_benchmark
```

Build the retrieval index:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\build_retrieval_index.py --chunks-jsonl data\table_reasoning_benchmark\table_reasoning_reference_chunks.jsonl --output-dir results\retrieval_index\table_reasoning_reference --dense-preset minilm
```

Run the benchmark with the dummy provider:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\benchmark_llm_fallback.py --manifest data\table_reasoning_benchmark\manifest.json --output-dir results\llm_fallback_benchmark\table_reasoning_dummy --llm-fallback-provider dummy --skip-build --no-warmup
```

Run with Ollama after the local provider is available:

```powershell
$env:BOXTALK_LLM_PROVIDER="ollama"
.\.venv-gpu\Scripts\python.exe scripts\benchmark_llm_fallback.py --manifest data\table_reasoning_benchmark\manifest.json --output-dir results\llm_fallback_benchmark\table_reasoning_ollama --llm-fallback-provider ollama --skip-build --no-warmup
```

## Metrics

The fallback comparison summary reports the existing grounded fallback metrics
plus table-specific metrics:

- `table_rule_resolved_count`
- `table_llm_resolved_count`
- `table_total_success`
- `reverse_lookup_success`
- `interval_mapping_success`
- `boundary_case_success`
- `multi_column_lookup_success`
- `table_text_reasoning_success`
- `numerical_reasoning_success`
- `fact_verification_success`

The report also includes:

- `by_table_reasoning_type`
- `by_benchmark_family`
- `table_resolution_breakdown`

`table_resolution_breakdown` separates `solved_by_rule_based`,
`solved_by_llm_fallback`, `solved_by_standard_or_other`, and heuristic failure
buckets for interval/boundary, row/column mapping, and weak packaging/reasoning.
