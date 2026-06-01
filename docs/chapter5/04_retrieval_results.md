# Retrieval Results

## Commands chay lai

```powershell
python scripts/benchmark_retrieval.py --index-dir results/retrieval_index/scifact_qa_minilm_20260513 --queries data/benchmarks/scifact_qa/queries_test.jsonl --output-dir results/retrieval_benchmark/scifact_qa_chapter5_top5 --top-k 5 --strategy all --no-warmup
python scripts/benchmark_retrieval.py --index-dir results/retrieval_index/qasper_qa_500_minilm_20260517 --queries data/benchmarks/qasper_qa_500_20260517/queries.jsonl --output-dir results/retrieval_benchmark/qasper_qa_chapter5_top20 --top-k 20 --strategy all --no-warmup
python scripts/benchmark_retrieval.py --index-dir results/retrieval_index/real_qcdt_e2e_hybrid_tatr_20260513 --queries results/retrieval_benchmark/real_qcdt_domain_queries_expected_pages_20260526.jsonl --output-dir results/retrieval_benchmark/qcdt_chapter5_top10_labeled --top-k 10 --strategy all --no-warmup
```

## SciFact top-5

| Strategy | Hit@5 | Recall@5 | MRR@5 | NDCG@5 | Latency ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.713 | 0.693 | 0.569 | 0.594 | 26.55 |
| Dense | 0.713 | 0.697 | 0.580 | 0.604 | 67.80 |
| Hybrid | 0.793 | 0.771 | 0.654 | 0.675 | 38.68 |
| Hybrid + rerank | 0.780 | 0.762 | 0.654 | 0.674 | 47.05 |

Hybrid dat Hit@5 cao nhat trong dot chay nay, cho thay ket hop lexical va semantic co loi tren evidence retrieval.

## QASPER top-20

| Strategy | Hit@20 | Recall@20 | MRR@20 | NDCG@20 | Latency ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.378 | 0.322 | 0.187 | 0.610 | 26.45 |
| Dense | 0.316 | 0.273 | 0.156 | 0.565 | 44.06 |
| Hybrid | 0.392 | 0.344 | 0.202 | 0.617 | 41.70 |
| Hybrid + rerank | 0.392 | 0.344 | 0.202 | 0.617 | 48.92 |

QASPER kho hon SciFact: evidence dai, multi-evidence va cau hoi free-form lam Recall@20 van thap.

## QCDT top-10 labeled

| Strategy | Hit@10 | Recall@10 | MRR@10 | NDCG@10 | Latency ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.550 | 0.550 | 0.387 | 0.469 | 0.67 |
| Dense | 0.400 | 0.400 | 0.258 | 0.331 | 413.81 |
| Hybrid | 0.600 | 0.600 | 0.407 | 0.492 | 6.35 |
| Hybrid + rerank | 0.575 | 0.575 | 0.413 | 0.490 | 11.35 |

QCDT la tai lieu phap quy tieng Viet, BM25 manh do cum tu/khoi dieu khoan rat dac trung. Hybrid van tang Hit@10 va Recall@10 so voi BM25.

## Ghi chu ve mot lan chay sai nhan

Command dung `data/real_pdfs/queries.jsonl` cho retrieval QCDT cho ket qua 0 vi file nay la QA gold answer, khong phai retrieval expected-page label dung dinh dang. Ket qua dung cho retrieval la `qcdt_chapter5_top10_labeled`.

## Ket luan

- BM25 phu hop voi cau hoi keyword phap quy.
- Dense co ich voi semantic paraphrase nhung khong on dinh tren tieng Viet phap quy neu dung MiniLM.
- Hybrid la lua chon chinh vi tang recall/hit trong SciFact va QCDT, trong khi latency van chap nhan duoc.
- Rerank heuristic khong luon tang Hit@k, nhung co the tang MRR nhe tren QCDT.
