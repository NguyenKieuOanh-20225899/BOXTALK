# README Chapter 5 Results

## Muc tieu Chuong 5

Danh gia pipeline truy xuat va hoi dap thong tin tren PDF theo nhieu tang: ingest, table extraction, retrieval, QA grounded/citation, ablation va error analysis. Toan bo ket qua phuc vu viet bao cao, khong fine-tune model va khong claim SOTA.

## Bang docs

| File | Noi dung | Trang thai | Ket qua chinh |
| --- | --- | --- | --- |
| `00_chapter5_experiment_plan.md` | Ke hoach, command validation, benchmark chay lai | Done | compileall pass, pytest 82 passed |
| `01_benchmark_inventory.md` | Inventory scripts/data/results/commands | Done | Liet ke ingest, retrieval, QA, table QA |
| `02_ingest_results.md` | Ket qua ingest/layout/OCR | Done | mock success 1.0; layout large tong hop tu results cu |
| `03_table_extraction_results.md` | Table extraction va constraint reconstruction | Done | hybrid_tatr structure F1 0.772 tren PubTables 25 rerun |
| `04_retrieval_results.md` | BM25 vs dense vs hybrid | Done | SciFact hybrid Hit@5 0.793; QCDT hybrid Hit@10 0.600 |
| `05_qa_results.md` | Routed grounded QA | Done | QCDT answer 0.725, Operations answer 0.925, hallucination 0.0 |
| `06_ablation_study.md` | Ablation retrieval/table/routing/QA | Done | Hybrid retrieval va cell citation la diem noi bat |
| `07_error_analysis.md` | Phan tich loi | Done | Bottleneck: retrieval cau hoi rong, synthesis, bang/OCR |
| `08_final_chapter5_tables.md` | Bang LaTeX copy vao bao cao | Done | Co bang dataset, ingest, table, retrieval, QA, error |
| `09_chapter5_flags_and_additions.md` | Flags, rerun bo sung, phan them vao Chuong 5 | Done | Liet ke env flags va output rerun large |
| `10_region_routing_ablation.md` | Ablation bat/tat region routing | Done | Mock khong regression; PubTables/FUNSD task-specific khong chung minh truc tiep region |
| `14_ocr_to_hybrid_tatr_extension.md` | Huong mo rong OCR word boxes sang Hybrid TATR | Done | De xuat flag bat/tat de so sanh scan table OCR-only voi OCR+Hybrid TATR |

## Commands da chay

```powershell
python -m compileall app scripts
python -m pytest -q
python scripts/benchmark_ingest_suite.py --dataset mock --limit 5 --out results/ingest/mock_chapter5_final --mode all
$env:BOXBIIBOO_LAYOUT_MODEL_NAME='Aryn/deformable-detr-DocLayNet'; python scripts/benchmark_ingest_suite.py --dataset doclaynet --data-dir data/benchmarks/doclaynet --limit 0 --out results/ingest/chapter5_doclaynet_full_rerun --mode layout --device cuda
$env:BOXBIIBOO_LAYOUT_MODEL_NAME='Aryn/deformable-detr-DocLayNet'; python scripts/benchmark_ingest_suite.py --dataset publaynet --data-dir data/benchmarks/publaynet --limit 0 --out results/ingest/chapter5_publaynet_full_rerun --mode layout --device cuda
$env:BOXBIIBOO_LAYOUT_MODEL_NAME='Aryn/deformable-detr-DocLayNet'; python scripts/benchmark_ingest_suite.py --dataset pubtables --data-dir data/benchmarks/pubtables_detection --limit 500 --out results/ingest/chapter5_pubtables_detection_500_rerun_model --mode table --device cuda
python scripts/benchmark_ingest_suite.py --dataset ocr --data-dir data/benchmarks/ocr_scan_25/ocr --limit 25 --out results/ingest/chapter5_ocr_scan_25_rerun_fixed_seq --mode ocr --device cuda
python scripts/benchmark_ingest_suite.py --dataset ocr --data-dir data/benchmarks/funsd/ocr --limit 25 --out results/ingest/chapter5_funsd_ocr_25_rerun_fixed_seq --mode ocr --device cuda
python scripts/benchmark_ingest_suite.py --dataset pubtables_structure --data-dir data/benchmarks/pubtables_structure_ocr_words_5 --limit 5 --out results/ingest/chapter5_pubtables_structure_default_5 --mode table --table-backend default
python scripts/benchmark_ingest_suite.py --dataset pubtables_structure --data-dir data/benchmarks/pubtables_structure_ocr_words_5 --limit 5 --out results/ingest/chapter5_pubtables_structure_tatr_5 --mode table --table-backend tatr
python scripts/benchmark_ingest_suite.py --dataset pubtables_structure --data-dir data/benchmarks/pubtables_structure_ocr_words_5 --limit 5 --out results/ingest/chapter5_pubtables_structure_hybrid_tatr_5 --mode table --table-backend hybrid_tatr
python scripts/benchmark_table_qa.py --queries data/benchmarks/table_qa_vi/queries.jsonl --out results/table_qa_vi/chapter5_final --variant all
python scripts/benchmark_retrieval.py --index-dir results/retrieval_index/scifact_qa_minilm_20260513 --queries data/benchmarks/scifact_qa/queries_test.jsonl --output-dir results/retrieval_benchmark/scifact_qa_chapter5_top5 --top-k 5 --strategy all --no-warmup
python scripts/benchmark_retrieval.py --index-dir results/retrieval_index/qasper_qa_500_minilm_20260517 --queries data/benchmarks/qasper_qa_500_20260517/queries.jsonl --output-dir results/retrieval_benchmark/qasper_qa_chapter5_top20 --top-k 20 --strategy all --no-warmup
python scripts/benchmark_retrieval.py --index-dir results/retrieval_index/real_qcdt_e2e_hybrid_tatr_20260513 --queries results/retrieval_benchmark/real_qcdt_domain_queries_expected_pages_20260526.jsonl --output-dir results/retrieval_benchmark/qcdt_chapter5_top10_labeled --top-k 10 --strategy all --no-warmup
python scripts/benchmark_qa.py --index-dir results/retrieval_index/real_qcdt_e2e_hybrid_tatr_20260513 --queries data/real_pdfs/queries.jsonl --output-dir results/qa_benchmark/qcdt_chapter5_routed_grounded --config routed_grounded --no-warmup
python scripts/benchmark_qa.py --index-dir results/retrieval_index/qa_operations_minilm --queries data/qa_benchmark/queries.jsonl --output-dir results/qa_benchmark/operations_chapter5_routed_grounded --config routed_grounded --no-warmup
python scripts/benchmark_qa.py --index-dir results/retrieval_index/scifact_qa_minilm_20260513 --queries data/benchmarks/scifact_qa/queries_test.jsonl --output-dir results/qa_benchmark/scifact_chapter5_routed_grounded --config routed_grounded --no-warmup
python scripts/benchmark_qa.py --index-dir results/retrieval_index/qasper_qa_500_minilm_20260517 --queries data/benchmarks/qasper_qa_500_20260517/queries.jsonl --output-dir results/qa_benchmark/qasper_chapter5_routed_grounded --config routed_grounded --no-warmup
```

## Link results chinh

- `results/ingest/mock_chapter5_final/summary.json`
- `results/ingest/chapter5_doclaynet_full_rerun/summary.json`
- `results/ingest/chapter5_publaynet_full_rerun/summary.json`
- `results/ingest/chapter5_pubtables_detection_500_rerun_model/summary.json`
- `results/ingest/chapter5_ocr_scan_25_rerun_fixed_seq/summary.json`
- `results/ingest/chapter5_funsd_ocr_25_rerun_fixed_seq/summary.json`
- `results/ingest/chapter5_pubtables_structure_default_5/summary.json`
- `results/ingest/chapter5_pubtables_structure_tatr_5/summary.json`
- `results/ingest/chapter5_pubtables_structure_hybrid_tatr_5/summary.json`
- `results/table_qa_vi/chapter5_final/summary.json`
- `results/retrieval_benchmark/scifact_qa_chapter5_top5/benchmark_summary.json`
- `results/retrieval_benchmark/qasper_qa_chapter5_top20/benchmark_summary.json`
- `results/retrieval_benchmark/qcdt_chapter5_top10_labeled/benchmark_summary.json`
- `results/qa_benchmark/qcdt_chapter5_routed_grounded/qa_summary.json`
- `results/qa_benchmark/operations_chapter5_routed_grounded/qa_summary.json`
- `results/qa_benchmark/scifact_chapter5_routed_grounded/qa_summary.json`
- `results/qa_benchmark/qasper_chapter5_routed_grounded/qa_summary.json`

## Ket qua chinh

- Validation pass: compileall pass, pytest `82 passed`.
- DocLayNet full rerun: F1@0.50 0.849, F1@0.75 0.807.
- PubLayNet full rerun: F1@0.50 0.739, F1@0.75 0.708.
- PubTables detection 500 rerun: F1@0.50 0.975, F1@0.75 0.914.
- OCR scan 25 rerun: OCR token F1 1.000; FUNSD OCR 25 rerun: OCR token F1 0.826.
- Hybrid retrieval dat SciFact Hit@5 0.793, cao hon BM25/Dense 0.713.
- QCDT retrieval top-10: hybrid Hit@10 0.600, BM25 0.550, dense 0.400.
- QCDT QA routed_grounded: answer_match 0.725, evidence_match 1.000, grounded_rate 1.000, hallucination_rate 0.000.
- Operations QA routed_grounded: answer_match 0.925, evidence_match 1.000, hallucination_rate 0.000.
- SciFact QA: evidence_match 0.727, grounded_rate 1.000, nhung answer_match 0.220.
- QASPER QA: answer_match 0.084, evidence_match 0.240, hallucination_rate 0.054, cho thay gioi han natural scientific QA.
- Table QA Vietnamese: table-aware retrieval + cell citation dat answer/evidence/cell citation 1.000 tren 8 cau mock-safe.

## Safe claims

- He thong co danh gia theo nhieu tang: ingest, retrieval, QA, table, citation.
- Hybrid retrieval phu hop vi can bang keyword va semantic retrieval.
- Grounded QA co citation giup kiem soat hallucination trong benchmark chinh.
- Hybrid TATR cai thien table structure tren subset da danh gia.
- QASPER cho thay han che cua answer synthesis/free-form scientific QA.

## Limitations

- Khong claim SOTA.
- Khong claim PDF nao cung trich xuat hoan hao.
- Khong claim khong bao gio hallucinate.
- Table reconstruction van chua hoan chinh voi moi merged cell/multi-header.
- PDF scan/image table hien chua tu dong noi OCR word boxes sang Hybrid TATR trong production pipeline; day la huong mo rong co flag bat/tat de ablation.
- Fine-tune TATR structure la viec de sau, chua lam trong giai doan nay.
