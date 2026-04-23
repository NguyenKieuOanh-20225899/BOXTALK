.PHONY: help \
	benchmark-setup-all \
	benchmark-setup-pubtables-test \
	benchmark-validate-all \
	benchmark-validate-pubtables-test \
	benchmark-reclaim-pubtables-raw \
	benchmark-production \
	benchmark-scientific-all \
	benchmark-scientific-pubtables-test \
	benchmark-suite-all \
	benchmark-suite-pubtables-test \
	benchmark-mac-quick \
	benchmark-mac-full \
	retrieval-smoke \
	retrieval-build \
	retrieval-benchmark \
	retrieval-beir-scifact \
	qa-dataset \
	qa-index \
	qa-benchmark \
	qa-benchmark-all \
	llm-fallback-dataset \
	llm-fallback-index \
	llm-fallback-smoke \
	llm-fallback-benchmark \
	ui-dev \
	user-pdf-suite \
	baseline-gate

PYTHON ?= .venv/bin/python
BENCHMARKS_ROOT ?= data/benchmarks
DOCLAYNET_ROOT ?= $(BENCHMARKS_ROOT)/doclaynet
PUBTABLES_ROOT ?= $(BENCHMARKS_ROOT)/pubtables_detection
PUBTABLES_SPLITS ?= test
PRODUCTION_PROFILES ?= baseline model_routed_doclaynet
SCIENTIFIC_PROFILES ?= baseline model_routed_doclaynet
PRODUCTION_REPEATS ?= 1
PRODUCTION_WARMUP_PER_LABEL ?= 1
PRODUCTION_MAX_PER_LABEL ?= 0
DOCLAYNET_SPLIT ?= test
DOCLAYNET_LIMIT ?= 0
PUBTABLES_SPLIT ?= test
PUBTABLES_LIMIT ?= 0
RETRIEVAL_PDF ?= data/retrieval_smoke/employee_handbook_smoke.pdf
RETRIEVAL_INDEX_DIR ?= results/retrieval_index/smoke_test
RETRIEVAL_BENCHMARK_DIR ?= results/retrieval_benchmark/smoke_test
RETRIEVAL_QUERIES ?= data/retrieval_smoke/queries.jsonl
RETRIEVAL_DENSE_PRESET ?= minilm
QA_BENCHMARK_DIR ?= results/qa_benchmark/smoke_test
QA_PDF ?= data/qa_benchmark/operations_handbook.pdf
QA_QUERIES ?= data/qa_benchmark/queries.jsonl
QA_INDEX_DIR ?= results/retrieval_index/qa_operations_minilm
QA_CONFIGS ?= routed_grounded
USER_PDF_SUITE_MANIFEST ?= data/user_pdf_benchmark_suite.json
USER_PDF_SUITE_DIR ?= results/user_pdf_benchmark_suite/current
USER_PDF_SUITE_ARGS ?=
LLM_FALLBACK_DATASET_DIR ?= data/llm_fallback_benchmark
LLM_FALLBACK_MANIFEST ?= $(LLM_FALLBACK_DATASET_DIR)/manifest.json
LLM_FALLBACK_INDEX_DIR ?= results/retrieval_index/llm_fallback_reference
LLM_FALLBACK_BENCHMARK_DIR ?= results/llm_fallback_benchmark/current
LLM_FALLBACK_PROVIDER ?= dummy
LLM_FALLBACK_ARGS ?=
BASELINE_GATE_ARGS ?=
BEIR_DATASET ?= scifact
BEIR_QUERY_LIMIT ?= 50
BEIR_CORPUS_LIMIT ?= 2000
UI_HOST ?= 127.0.0.1
UI_PORT ?= 8000

help:
	@printf '%s\n' \
		'benchmark-setup-all             download/extract DocLayNet + PubTables using BENCHMARKS_ROOT' \
		'benchmark-setup-pubtables-test  download/extract PubTables detection test split only' \
		'benchmark-validate-all          inspect local academic benchmark layout and write manifests' \
		'benchmark-validate-pubtables-test inspect PubTables test layout and write manifests' \
		'benchmark-reclaim-pubtables-raw remove PubTables raw tar.gz after extraction to save disk' \
		'benchmark-production            run production ingest benchmark on data/test_probe' \
		'benchmark-scientific-all        run scientific benchmark on DocLayNet + PubTables' \
		'benchmark-scientific-pubtables-test run scientific benchmark on PubTables test only' \
		'benchmark-suite-all             run production + scientific suite' \
		'benchmark-suite-pubtables-test  run production + PubTables-only scientific suite' \
		'benchmark-mac-quick            one-command Mac-safe flow: setup + validate + baseline scientific sample' \
		'benchmark-mac-full             one-command Mac flow: setup + validate + full PubTables-only scientific run' \
		'retrieval-smoke                create smoke PDF/queries, build index, benchmark retrieval' \
		'retrieval-build                build retrieval index from RETRIEVAL_PDF' \
		'retrieval-benchmark            benchmark RETRIEVAL_INDEX_DIR against RETRIEVAL_QUERIES' \
		'retrieval-beir-scifact         run a BEIR SciFact retrieval benchmark sample' \
		'qa-dataset                     create controlled QA benchmark PDF + queries' \
		'qa-index                       build retrieval index for QA_PDF' \
		'qa-benchmark                   benchmark grounded QA over QA_INDEX_DIR' \
		'qa-benchmark-all               create QA dataset, build index, run baseline + ablation' \
		'llm-fallback-dataset           create the focused fallback benchmark chunks + queries' \
		'llm-fallback-index             build retrieval index for the focused fallback benchmark' \
		'llm-fallback-smoke             run the fallback benchmark with the dummy provider' \
		'llm-fallback-benchmark         run the fallback benchmark with the configured provider' \
		'ui-dev                         run the FastAPI backend plus static MVP UI' \
		'user-pdf-suite                 run aggregate QA benchmark over user PDF suite manifest' \
		'baseline-gate                  fail if locked benchmark baselines regress'

benchmark-setup-all:
	$(PYTHON) scripts/setup_benchmark_datasets.py --dataset all --benchmarks-root $(BENCHMARKS_ROOT) --pubtables-splits $(PUBTABLES_SPLITS)

benchmark-setup-pubtables-test:
	$(PYTHON) scripts/setup_benchmark_datasets.py --dataset pubtables --benchmarks-root $(BENCHMARKS_ROOT) --pubtables-splits test

benchmark-validate-all:
	$(PYTHON) scripts/setup_benchmark_datasets.py --dataset all --benchmarks-root $(BENCHMARKS_ROOT) --pubtables-splits $(PUBTABLES_SPLITS) --validate-only

benchmark-validate-pubtables-test:
	$(PYTHON) scripts/setup_benchmark_datasets.py --dataset pubtables --benchmarks-root $(BENCHMARKS_ROOT) --pubtables-splits test --validate-only

benchmark-reclaim-pubtables-raw:
	rm -f $(PUBTABLES_ROOT)/raw/*.tar.gz
	$(PYTHON) scripts/setup_benchmark_datasets.py --dataset pubtables --benchmarks-root $(BENCHMARKS_ROOT) --pubtables-splits test --validate-only

benchmark-production:
	$(PYTHON) scripts/benchmark_ingest_standard.py --profiles $(PRODUCTION_PROFILES) --repeats $(PRODUCTION_REPEATS) --warmup-per-label $(PRODUCTION_WARMUP_PER_LABEL) --max-per-label $(PRODUCTION_MAX_PER_LABEL)

benchmark-scientific-all:
	$(PYTHON) scripts/benchmark_ingest_scientific.py --doclaynet-root $(DOCLAYNET_ROOT) --doclaynet-split $(DOCLAYNET_SPLIT) --doclaynet-limit $(DOCLAYNET_LIMIT) --pubtables-root $(PUBTABLES_ROOT) --pubtables-split $(PUBTABLES_SPLIT) --pubtables-limit $(PUBTABLES_LIMIT) --profiles $(SCIENTIFIC_PROFILES)

benchmark-scientific-pubtables-test:
	$(PYTHON) scripts/benchmark_ingest_scientific.py --skip-doclaynet --pubtables-root $(PUBTABLES_ROOT) --pubtables-split test --pubtables-limit $(PUBTABLES_LIMIT) --profiles $(SCIENTIFIC_PROFILES)

benchmark-suite-all:
	$(PYTHON) scripts/benchmark_ingest_suite.py --production-repeats $(PRODUCTION_REPEATS) --production-warmup-per-label $(PRODUCTION_WARMUP_PER_LABEL) --production-max-per-label $(PRODUCTION_MAX_PER_LABEL) --production-profiles $(PRODUCTION_PROFILES) --doclaynet-root $(DOCLAYNET_ROOT) --doclaynet-split $(DOCLAYNET_SPLIT) --doclaynet-limit $(DOCLAYNET_LIMIT) --pubtables-root $(PUBTABLES_ROOT) --pubtables-split $(PUBTABLES_SPLIT) --pubtables-limit $(PUBTABLES_LIMIT) --scientific-profiles $(SCIENTIFIC_PROFILES)

benchmark-suite-pubtables-test:
	$(PYTHON) scripts/benchmark_ingest_suite.py --production-repeats $(PRODUCTION_REPEATS) --production-warmup-per-label $(PRODUCTION_WARMUP_PER_LABEL) --production-max-per-label $(PRODUCTION_MAX_PER_LABEL) --production-profiles $(PRODUCTION_PROFILES) --skip-doclaynet --doclaynet-root $(DOCLAYNET_ROOT) --doclaynet-split $(DOCLAYNET_SPLIT) --doclaynet-limit $(DOCLAYNET_LIMIT) --pubtables-root $(PUBTABLES_ROOT) --pubtables-split test --pubtables-limit $(PUBTABLES_LIMIT) --scientific-profiles $(SCIENTIFIC_PROFILES)

benchmark-mac-quick:
	$(MAKE) benchmark-setup-pubtables-test
	$(MAKE) benchmark-validate-pubtables-test
	$(MAKE) benchmark-scientific-pubtables-test SCIENTIFIC_PROFILES=baseline PUBTABLES_LIMIT=25

benchmark-mac-full:
	$(MAKE) benchmark-setup-pubtables-test
	$(MAKE) benchmark-validate-pubtables-test
	$(MAKE) benchmark-scientific-pubtables-test

retrieval-smoke:
	$(PYTHON) scripts/create_retrieval_smoke_dataset.py
	$(MAKE) retrieval-build
	$(MAKE) retrieval-benchmark

retrieval-build:
	$(PYTHON) scripts/build_retrieval_index.py --pdf $(RETRIEVAL_PDF) --output-dir $(RETRIEVAL_INDEX_DIR) --dense-preset $(RETRIEVAL_DENSE_PRESET)

retrieval-benchmark:
	$(PYTHON) scripts/benchmark_retrieval.py --index-dir $(RETRIEVAL_INDEX_DIR) --queries $(RETRIEVAL_QUERIES) --output-dir $(RETRIEVAL_BENCHMARK_DIR) --top-k 5 --strategy all

retrieval-beir-scifact:
	$(PYTHON) scripts/benchmark_beir_retrieval.py --dataset $(BEIR_DATASET) --query-limit $(BEIR_QUERY_LIMIT) --corpus-limit $(BEIR_CORPUS_LIMIT) --strategy bm25 --strategy dense --strategy hybrid --dense-preset $(RETRIEVAL_DENSE_PRESET)

qa-dataset:
	$(PYTHON) scripts/create_qa_benchmark_dataset.py

qa-index:
	$(PYTHON) scripts/build_retrieval_index.py --pdf $(QA_PDF) --output-dir $(QA_INDEX_DIR) --dense-preset $(RETRIEVAL_DENSE_PRESET)

qa-benchmark:
	$(PYTHON) scripts/benchmark_qa.py --index-dir $(QA_INDEX_DIR) --queries $(QA_QUERIES) --output-dir $(QA_BENCHMARK_DIR) --config $(QA_CONFIGS)

qa-benchmark-all:
	$(MAKE) qa-dataset
	$(MAKE) qa-index
	$(MAKE) qa-benchmark QA_BENCHMARK_DIR=results/qa_benchmark/qa_operations_minilm_all QA_CONFIGS=all

llm-fallback-dataset:
	$(PYTHON) scripts/create_llm_fallback_benchmark.py --output-dir $(LLM_FALLBACK_DATASET_DIR)

llm-fallback-index:
	$(MAKE) llm-fallback-dataset
	$(PYTHON) scripts/build_retrieval_index.py --chunks-jsonl $(LLM_FALLBACK_DATASET_DIR)/llm_fallback_reference_chunks.jsonl --output-dir $(LLM_FALLBACK_INDEX_DIR) --dense-preset $(RETRIEVAL_DENSE_PRESET)

llm-fallback-smoke:
	$(MAKE) llm-fallback-benchmark LLM_FALLBACK_PROVIDER=dummy

llm-fallback-benchmark:
	$(MAKE) llm-fallback-dataset
	$(PYTHON) scripts/benchmark_llm_fallback.py --manifest $(LLM_FALLBACK_MANIFEST) --output-dir $(LLM_FALLBACK_BENCHMARK_DIR) --llm-fallback-provider $(LLM_FALLBACK_PROVIDER) $(LLM_FALLBACK_ARGS)

ui-dev:
	$(PYTHON) -m uvicorn app.routed_rag_starter:app --host $(UI_HOST) --port $(UI_PORT) --reload

user-pdf-suite:
	$(PYTHON) scripts/benchmark_user_pdf_suite.py --manifest $(USER_PDF_SUITE_MANIFEST) --output-dir $(USER_PDF_SUITE_DIR) $(USER_PDF_SUITE_ARGS)

baseline-gate:
	$(PYTHON) scripts/check_regression_gates.py $(BASELINE_GATE_ARGS)
