# QA Results

## Commands chay lai

```powershell
python scripts/benchmark_qa.py --index-dir results/retrieval_index/real_qcdt_e2e_hybrid_tatr_20260513 --queries data/real_pdfs/queries.jsonl --output-dir results/qa_benchmark/qcdt_chapter5_routed_grounded --config routed_grounded --no-warmup
python scripts/benchmark_qa.py --index-dir results/retrieval_index/qa_operations_minilm --queries data/qa_benchmark/queries.jsonl --output-dir results/qa_benchmark/operations_chapter5_routed_grounded --config routed_grounded --no-warmup
python scripts/benchmark_qa.py --index-dir results/retrieval_index/scifact_qa_minilm_20260513 --queries data/benchmarks/scifact_qa/queries_test.jsonl --output-dir results/qa_benchmark/scifact_chapter5_routed_grounded --config routed_grounded --no-warmup
python scripts/benchmark_qa.py --index-dir results/retrieval_index/qasper_qa_500_minilm_20260517 --queries data/benchmarks/qasper_qa_500_20260517/queries.jsonl --output-dir results/qa_benchmark/qasper_chapter5_routed_grounded --config routed_grounded --no-warmup
```

## Summary

| Benchmark | Queries | Answer match | Evidence match | Grounded rate | Hallucination rate | End-to-end success | Avg latency ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| QCDT routed_grounded | 40 | 0.725 | 1.000 | 1.000 | 0.000 | 0.725 | 467.40 |
| Operations routed_grounded | 40 | 0.925 | 1.000 | 1.000 | 0.000 | 0.925 | 411.27 |
| SciFact routed_grounded | 300 | 0.220 | 0.727 | 1.000 | 0.000 | 0.203 | 109.43 |
| QASPER routed_grounded | 500 | 0.084 | 0.240 | 1.000 | 0.054 | 0.050 | 89.41 |

## Nhan xet theo benchmark

QCDT va Operations la hai benchmark gan pipeline chinh nhat. Evidence match va grounded rate deu dat 1.0, hallucination rate 0.0 trong dot chay nay. Answer match chua dat tuyet doi vi mot so cau hoi can tong hop nhieu y, cau hoi so sanh hoac cau hoi co section ambiguity.

SciFact co evidence_match 0.727 va grounded_rate 1.0. Day la bang chung tot cho evidence/citation, nhung answer_match 0.220 cho thay answer synthesis dang con han che voi claim scientific.

QASPER la benchmark kho nhat: answer_match 0.084, evidence_match 0.240 va hallucination_rate 0.054. Day nen duoc dua vao bao cao nhu phan phan tich gioi han cua natural scientific QA, khong phai benchmark ma pipeline dang toi uu.

## Ablation QA cu can trich dan

Tu `results/qa_benchmark/rerun_real_qcdt_all_20260526/qa_summary.json` va `results/qa_benchmark/rerun_operations_all_20260526/qa_summary.json`:

| Benchmark | Config | Answer match | Evidence match | Grounded | Hallucination |
| --- | --- | ---: | ---: | ---: | ---: |
| QCDT | routed_grounded | 0.725 | 1.000 | 1.000 | 0.000 |
| QCDT | adaptive_route_retry | 0.750 | 1.000 | 1.000 | 0.000 |
| QCDT | no_citation_grounding | 0.725 | 1.000 | 0.000 | 1.000 |
| Operations | routed_grounded | 0.925 | 1.000 | 1.000 | 0.000 |
| Operations | no_evidence_checker | 0.775 | 0.850 | 1.000 | 0.150 |
| Operations | no_citation_grounding | 0.925 | 1.000 | 0.175 | 0.825 |

## Ket luan an toan

- Citation va evidence checker giup kiem soat hallucination trong benchmark chinh.
- Retrieval dung khong dong nghia answer synthesis dung; QASPER va SciFact cho thay bottleneck o tong hop cau tra loi.
- Nen trinh bay routed_grounded nhu mot co che giam rui ro, khong phai bao dam khong bao gio hallucinate.
