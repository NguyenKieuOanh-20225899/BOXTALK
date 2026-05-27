# Benchmark rerun summary 2026-05-26

Muc tieu: ghi lai cac benchmark da chay lai de lam nguon moi nhat cho bao cao va slide bao ve.

## Da chay

| Nhom | Lenh/nguon | Output |
|---|---|---|
| Test suite | `python -m pytest -q` | `65 passed` |
| SciFact retrieval | `scripts/benchmark_retrieval.py`, top-k=5, strategy `all` | `results/retrieval_benchmark/rerun_scifact_qa_minilm_top5_20260526_labeled` |
| QASPER retrieval | top-k=5/10/20, strategy `all` | `results/retrieval_benchmark/rerun_qasper_qa_minilm_top{5,10,20}_20260526` |
| QCDT retrieval | page/section-level top-k=5/10/20, strategy `all` | `results/retrieval_benchmark/rerun_real_qcdt_page_top{5,10,20}_20260526` |
| QCDT QA | config `all` | `results/qa_benchmark/rerun_real_qcdt_all_20260526` |
| Operations QA | config `all` | `results/qa_benchmark/rerun_operations_all_20260526` |
| PubTables table ingest | default/TATR/hybrid_tatr, 25 samples | `results/ingest/rerun_pubtables_structure_*_25_20260526` |

## Ket qua chinh

### Retrieval

| Dataset | Best/important result | Dien giai |
|---|---|---|
| SciFact top-5 | Hybrid Hit@5 0.793, Recall@5 0.771 | Hybrid vuot BM25 va dense don le. |
| QASPER top-20 | Hybrid/hybrid_rerank Recall@20 0.518 | Tang top-k giup recall; QASPER van kho vi free-form scientific QA. |
| QCDT top-5 | Hybrid rerank Hit@5 0.525, MRR@5 0.406 | Rerank huu ich cho xep hang evidence som. |
| QCDT top-10 | Hybrid Hit@10 0.600 | Hybrid tot nhat o top-k=10 tren domain chinh. |
| QCDT top-20 | BM25 Hit@20 0.675 | BM25 van la baseline manh cho van ban phap quy tieng Viet. |

### QA

| Dataset | Config | Answer match | Evidence match | Grounded | Hallucination |
|---|---|---:|---:|---:|---:|
| QCDT | routed_grounded | 0.725 | 1.000 | 1.000 | 0.000 |
| QCDT | adaptive_route_retry | 0.750 | 1.000 | 1.000 | 0.000 |
| QCDT | no_citation_grounding | 0.725 | 1.000 | 0.000 | 1.000 |
| Operations | routed_grounded | 0.925 | 1.000 | 1.000 | 0.000 |
| Operations | no_evidence_checker | 0.775 | 0.850 | 1.000 | 0.150 |
| Operations | no_citation_grounding | 0.925 | 1.000 | 0.175 | 0.825 |

### Table ingest

| Backend | Det F1@0.50 | Cell F1@0.50 | Cell F1@0.75 | Structure F1 | Text assign F1 | Exact CSV |
|---|---:|---:|---:|---:|---:|---:|
| Default | 0.940 | 0.650 | 0.149 | 0.199 | 0.909 | 0.000 |
| TATR | 0.987 | 0.491 | 0.103 | 0.010 | 0.015 | 0.000 |
| hybrid_tatr | 0.987 | 0.958 | 0.944 | 0.772 | 0.999 | 0.480 |

## Ket luan cap nhat

1. Do an co bang chung ablation du manh de bao ve: retrieval, QA, citation grounding va table backend deu co so sanh rieng.
2. Hybrid retrieval co ich ro tren SciFact va QCDT top-k=10, nhung BM25 van rat manh tren van ban phap quy tieng Viet. Khong nen claim dense/hybrid luon thang moi truong hop.
3. Citation grounding la thanh phan bat buoc: khi tat citation grounding, QCDT hallucination metric len 1.000 va Operations len 0.825.
4. Hybrid TATR la diem noi bat moi cua phan table: Structure F1 0.772 va Exact CSV 0.480 tren 25 mau. Tuy nhien Exact CSV chua hoan chinh, nen claim dung la "ho tro table QA va table reconstruction co dieu kien", khong phai "exact reconstruction cho moi bang".
5. Ket qua rerun da du de chot bao cao neu trinh bay ro pham vi: PDF text-layer ban cau truc, tai lieu quy che/quy dinh/thong tu/huong dan/policy.

## Luu y ve run bi loai

Co mot lan goi SciFact nham `data/beir/scifact/queries.jsonl`, file nay khong co nhan expected evidence cho benchmark retrieval nen metric ra 0. Run do khong duoc dung trong tai lieu. Nguon SciFact hop le la:

```text
data/benchmarks/scifact_qa/queries_test.jsonl
```
