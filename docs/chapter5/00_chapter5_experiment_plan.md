# Chapter 5 Experiment Plan

## Muc tieu

Chuong 5 danh gia pipeline "Nghien cuu cac ki thuat truy xuat va hoi dap thong tin tren tai lieu PDF" theo nhieu tang:

- ingest PDF: text, layout, table, OCR;
- retrieval: BM25, dense, hybrid, hybrid + rerank;
- QA: routed grounded answer, evidence, citation, hallucination;
- table QA: table-aware retrieval va cell-level citation;
- error analysis: gioi han cua PDF phap quy, scientific QA, bang phuc tap va OCR.

Pipeline chinh duoc danh gia:

```text
PDF ingest -> region-level routing -> chunk/index -> retrieval -> routed_grounded -> grounded answer + citation
```

Khong fine-tune model trong giai doan nay. Cac ket qua chi duoc dien giai nhu benchmark noi bo / subset da chay, khong claim SOTA.

## Validation da chay

| Command | Ket qua | Ghi chu |
| --- | --- | --- |
| `python -m compileall app scripts` | Pass | Khong loi compile. |
| `python -m pytest -q` | `82 passed in 40.20s` | Validation unit/integration hien co pass. |
| `python scripts/benchmark_ingest_suite.py --dataset mock --limit 5 --out results/ingest/mock_chapter5_final --mode all` | Pass | success_rate 1.0, error_count 0. |
| `python scripts/benchmark_table_qa.py --queries data/benchmarks/table_qa_vi/queries.jsonl --out results/table_qa_vi/chapter5_final --variant all` | Pass | 8 cau hoi table QA mock-safe. |

## Benchmark da chay lai trong dot nay

| Nhom | Output |
| --- | --- |
| Mock ingest | `results/ingest/mock_chapter5_final/summary.json` |
| PubTables structure default, 5 mau | `results/ingest/chapter5_pubtables_structure_default_5/summary.json` |
| PubTables structure TATR, 5 mau | `results/ingest/chapter5_pubtables_structure_tatr_5/summary.json` |
| PubTables structure hybrid_tatr, 5 mau | `results/ingest/chapter5_pubtables_structure_hybrid_tatr_5/summary.json` |
| Table QA Vietnamese | `results/table_qa_vi/chapter5_final/summary.json` |
| SciFact retrieval top-5 | `results/retrieval_benchmark/scifact_qa_chapter5_top5/benchmark_summary.json` |
| QASPER retrieval top-20 | `results/retrieval_benchmark/qasper_qa_chapter5_top20/benchmark_summary.json` |
| QCDT retrieval top-10 labeled | `results/retrieval_benchmark/qcdt_chapter5_top10_labeled/benchmark_summary.json` |
| QCDT QA routed_grounded | `results/qa_benchmark/qcdt_chapter5_routed_grounded/qa_summary.json` |
| Operations QA routed_grounded | `results/qa_benchmark/operations_chapter5_routed_grounded/qa_summary.json` |
| SciFact QA routed_grounded | `results/qa_benchmark/scifact_chapter5_routed_grounded/qa_summary.json` |
| QASPER QA routed_grounded | `results/qa_benchmark/qasper_chapter5_routed_grounded/qa_summary.json` |

## Benchmark khong chay lai duoc hoac chi tong hop

| Benchmark | Trang thai | Ly do |
| --- | --- | --- |
| DocLayNet full rerun | Da chay lai | Bat `BOXBIIBOO_LAYOUT_MODEL_NAME=Aryn/deformable-detr-DocLayNet`, output `results/ingest/chapter5_doclaynet_full_rerun`. |
| PubLayNet full rerun | Da chay lai | Bat `BOXBIIBOO_LAYOUT_MODEL_NAME=Aryn/deformable-detr-DocLayNet`, output `results/ingest/chapter5_publaynet_full_rerun`. |
| PubTables detection 500 | Da chay lai | Output `results/ingest/chapter5_pubtables_detection_500_rerun_model`. |
| OCR scan 25 va FUNSD 25 | Da chay lai | OCR adapter can `data-dir` tro vao thu muc `ocr`; outputs `chapter5_ocr_scan_25_rerun_fixed_seq`, `chapter5_funsd_ocr_25_rerun_fixed_seq`. |
| PubTables structure default 25 | Timeout | Default OCR-backed table structure qua 20 phut chua ghi summary; dung rerun 25 cu va rerun mau 5 trong dot nay. |
| OCR-D / Nougat full | Tong hop tu results cu | Benchmark OCR/scientific rieng ton thoi gian; da co results cu de trich dan. |

## Nguyen tac dien giai

- Chi noi "cai thien tren subset da danh gia", khong noi hoan hao.
- Hybrid retrieval duoc chon vi can bang keyword matching va semantic matching.
- Grounded QA co citation giam rui ro hallucination trong benchmark chinh, khong bao dam tuyet doi.
- Hybrid TATR va constraint-aware reconstruction ho tro table structure/table QA, nhung exact reconstruction cho moi PDF van la gioi han.
