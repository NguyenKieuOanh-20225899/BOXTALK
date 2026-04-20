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
	benchmark-mac-full

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
		'benchmark-mac-full             one-command Mac flow: setup + validate + full PubTables-only scientific run'

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
