# Ablation study va demo case dung cho bao ve

Ngay cap nhat: 2026-05-26  
Nguon moi nhat: cac benchmark `rerun_*_20260526` trong `results/`.

Muc tieu: tong hop bang chung thuc nghiem cho de tai **"Nghien cuu cac ki thuat truy xuat va hoi dap thong tin tren tai lieu PDF"**. Cac bang duoi day chi dung so lieu da chay lai trong workspace, ngoai tru demo case QA QCDT lay tu run truoc vi noi dung cau tra loi/citation khong doi.

## 1. Retrieval ablation: BM25 vs dense vs hybrid

### 1.1 SciFact, top-k = 5

Nguon: `results/retrieval_benchmark/rerun_scifact_qa_minilm_top5_20260526_labeled/README.md`

| Strategy | Queries | Hit@5 | Recall@5 | MRR@5 | nDCG@5 | Avg latency ms |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 300 | 0.713 | 0.693 | 0.569 | 0.594 | 27.90 |
| Dense | 300 | 0.713 | 0.697 | 0.580 | 0.604 | 8.28 |
| Hybrid | 300 | 0.793 | 0.771 | 0.653 | 0.675 | 41.44 |
| Hybrid rerank | 300 | 0.780 | 0.762 | 0.654 | 0.674 | 50.24 |

Ket luan dung khi bao ve:

```text
Tren SciFact, hybrid retrieval van vuot BM25 va dense don le ve Hit@5 va Recall@5. Rerank giu MRR gan tuong duong hybrid, nhung khong tang Hit@5 tren bo nay; do do nen claim rang rerank ho tro sap xep evidence, khong phai luc nao cung tang moi metric.
```

### 1.2 QASPER, top-k = 5/10/20

Nguon:

- `results/retrieval_benchmark/rerun_qasper_qa_minilm_top5_20260526/README.md`
- `results/retrieval_benchmark/rerun_qasper_qa_minilm_top10_20260526/README.md`
- `results/retrieval_benchmark/rerun_qasper_qa_minilm_top20_20260526/README.md`

| Top-k | Strategy | Queries | Hit@k | Recall@k | MRR@k | nDCG@k |
|---:|---|---:|---:|---:|---:|---:|
| 5 | BM25 | 100 | 0.390 | 0.331 | 0.273 | 0.404 |
| 5 | Dense | 100 | 0.280 | 0.235 | 0.154 | 0.276 |
| 5 | Hybrid | 100 | 0.390 | 0.339 | 0.230 | 0.352 |
| 5 | Hybrid rerank | 100 | 0.360 | 0.296 | 0.255 | 0.357 |
| 10 | BM25 | 100 | 0.490 | 0.423 | 0.285 | 0.532 |
| 10 | Dense | 100 | 0.350 | 0.307 | 0.164 | 0.404 |
| 10 | Hybrid | 100 | 0.500 | 0.437 | 0.245 | 0.490 |
| 10 | Hybrid rerank | 100 | 0.510 | 0.465 | 0.276 | 0.521 |
| 20 | BM25 | 100 | 0.570 | 0.502 | 0.290 | 0.717 |
| 20 | Dense | 100 | 0.420 | 0.369 | 0.170 | 0.571 |
| 20 | Hybrid | 100 | 0.570 | 0.518 | 0.250 | 0.670 |
| 20 | Hybrid rerank | 100 | 0.570 | 0.518 | 0.281 | 0.691 |

Ket luan dung khi bao ve:

```text
Voi QASPER, tang top-k lam evidence recall tang ro. Hybrid rerank co ich hon o top-k=10/20, dac biet MRR va Recall@10, nhung ket qua top-k=5 cho thay rerank co the lam mat mot so hit som. Day la ly do can ablation thay vi chi khang dinh mot cau hinh la tot nhat moi luc.
```

### 1.3 QCDT domain chinh, page/section-level top-k = 5/10/20

Nguon:

- `results/retrieval_benchmark/rerun_real_qcdt_page_top5_20260526/README.md`
- `results/retrieval_benchmark/rerun_real_qcdt_page_top10_20260526/README.md`
- `results/retrieval_benchmark/rerun_real_qcdt_page_top20_20260526/README.md`

Luu y: day la metric page/section-level, chua phai exact chunk-level qrels.

| Top-k | Strategy | Queries | Hit@k | Recall@k | MRR@k | nDCG@k |
|---:|---|---:|---:|---:|---:|---:|
| 5 | BM25 | 40 | 0.500 | 0.500 | 0.379 | 0.452 |
| 5 | Dense | 40 | 0.300 | 0.300 | 0.242 | 0.280 |
| 5 | Hybrid | 40 | 0.500 | 0.500 | 0.394 | 0.443 |
| 5 | Hybrid rerank | 40 | 0.525 | 0.525 | 0.406 | 0.458 |
| 10 | BM25 | 40 | 0.550 | 0.550 | 0.387 | 0.469 |
| 10 | Dense | 40 | 0.400 | 0.400 | 0.258 | 0.331 |
| 10 | Hybrid | 40 | 0.600 | 0.600 | 0.407 | 0.492 |
| 10 | Hybrid rerank | 40 | 0.575 | 0.575 | 0.413 | 0.490 |
| 20 | BM25 | 40 | 0.675 | 0.675 | 0.395 | 0.526 |
| 20 | Dense | 40 | 0.525 | 0.525 | 0.265 | 0.368 |
| 20 | Hybrid | 40 | 0.650 | 0.650 | 0.411 | 0.517 |
| 20 | Hybrid rerank | 40 | 0.650 | 0.650 | 0.419 | 0.523 |

Ket luan dung khi bao ve:

```text
Tren QCDT, BM25 la baseline manh vi tai lieu phap quy tieng Viet co nhieu keyword trung truc tiep. Dense don le yeu hon, nen pipeline khong nen chi dua vao dense retrieval. Hybrid/rerank van co ich o xep hang evidence, dac biet MRR, nhung can giu BM25 la thanh phan cot loi.
```

## 2. QA ablation: routing, evidence checking va citation grounding

### 2.1 QCDT real PDF

Nguon: `results/qa_benchmark/rerun_real_qcdt_all_20260526/README.md`

| Config | Queries | Answer match | Evidence match | Grounded | Hallucination | Avg latency ms |
|---|---:|---:|---:|---:|---:|---:|
| BM25 only | 40 | 0.800 | 1.000 | 1.000 | 0.000 | 5.5 |
| Dense only | 40 | 0.350 | 0.700 | 1.000 | 0.000 | 8.1 |
| Hybrid no routing | 40 | 0.675 | 0.950 | 1.000 | 0.000 | 12.5 |
| Routed grounded | 40 | 0.725 | 1.000 | 1.000 | 0.000 | 14.3 |
| Adaptive route retry | 40 | 0.750 | 1.000 | 1.000 | 0.000 | 15.1 |
| No evidence checker | 40 | 0.750 | 1.000 | 1.000 | 0.000 | 16.2 |
| No router | 40 | 0.675 | 0.950 | 1.000 | 0.000 | 12.8 |
| No citation grounding | 40 | 0.725 | 1.000 | 0.000 | 1.000 | 15.9 |

Ket luan dung khi bao ve:

```text
Tren QCDT, routed_grounded cai thien so voi no_router/hybrid_no_routing ve answer match va evidence match. Adaptive retry dat answer_match cao nhat. No_citation_grounding giu answer_match 0.725 nhung grounded = 0 va hallucination = 1.000 theo metric, chung minh citation grounding la thanh phan bat buoc cua grounded QA.
```

### 2.2 Operations QA

Nguon: `results/qa_benchmark/rerun_operations_all_20260526/README.md`

| Config | Queries | Answer match | Evidence match | Grounded | Hallucination | Avg latency ms |
|---|---:|---:|---:|---:|---:|---:|
| BM25 only | 40 | 0.925 | 0.975 | 1.000 | 0.025 | 2.7 |
| Dense only | 40 | 0.950 | 1.000 | 1.000 | 0.000 | 14.7 |
| Hybrid no routing | 40 | 0.925 | 0.975 | 1.000 | 0.025 | 17.4 |
| Routed grounded | 40 | 0.925 | 1.000 | 1.000 | 0.000 | 15.3 |
| Adaptive route retry | 40 | 0.900 | 1.000 | 1.000 | 0.000 | 27.6 |
| No evidence checker | 40 | 0.775 | 0.850 | 1.000 | 0.150 | 16.1 |
| No router | 40 | 0.825 | 0.975 | 1.000 | 0.025 | 10.1 |
| No citation grounding | 40 | 0.925 | 1.000 | 0.175 | 0.825 | 12.1 |

Ket luan dung khi bao ve:

```text
Tren Operations QA, evidence checker va citation grounding co vai tro ro. Bo evidence checker lam answer_match giam tu 0.925 xuong 0.775 va hallucination tang len 0.150. Bo citation grounding lam grounded giam con 0.175 va hallucination tang len 0.825.
```

## 3. Table ablation: default table vs TATR vs hybrid_tatr

Nguon:

- `results/ingest/rerun_pubtables_structure_default_25_20260526/summary.json`
- `results/ingest/rerun_pubtables_structure_tatr_25_20260526/summary.json`
- `results/ingest/rerun_pubtables_structure_hybrid_tatr_25_20260526/summary.json`

| Backend | Det F1@0.50 | Cell F1@0.50 | Cell F1@0.75 | Structure F1 | Text assign F1 | Row MAE | Col MAE | GriTS-con-like | Exact CSV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Default | 0.940 | 0.650 | 0.149 | 0.199 | 0.909 | 1.960 | 0.800 | 0.146 | 0.000 |
| TATR | 0.987 | 0.491 | 0.103 | 0.010 | 0.015 | 0.600 | 0.000 | 0.006 | 0.000 |
| hybrid_tatr | 0.987 | 0.958 | 0.944 | 0.772 | 0.999 | 0.600 | 0.000 | 0.700 | 0.480 |

Ket luan dung khi bao ve:

```text
TATR thuan bat duoc hinh hoc bang tot nhung gan nhu khong gan duoc text vao cell. Default giu text tot hon nhung cau truc bang con yeu. Hybrid_tatr ket hop hinh hoc bang voi word boxes nen tang manh Cell F1, Structure F1, Text assignment va Exact CSV. Tuy vay Exact CSV moi dat 0.480 tren 25 mau, nen khong claim exact reconstruction hoan chinh cho moi bang.
```

## 4. Ba demo case nen dua vao slide/bao ve

Nguon case: `results/qa_benchmark/real_qcdt_all_after_table_answer_20260513/per_question.json`, config `routed_grounded`.

| Case | Query id | Cau hoi | Cau tra loi ngan | Citation | Ket qua |
|---|---|---|---|---|---|
| Text phap quy | `q03` | Chuong trinh tich hop cu nhan - ky su hoac cu nhan - thac si duoc thiet ke trong bao lau va bao nhieu tin chi? | 5,5 nam va 180 tin chi | Trang 5-6, Dieu 2 | answer_match=true, evidence_match=true, grounded=true |
| Bang | `q05` | Chuong trinh cu nhan chinh quy danh cho nguoi tot nghiep THPT co thoi gian va khoi luong toi thieu la bao nhieu? | 4 nam va 132 tin chi | Trang 6, bang CTDT | answer_match=true, evidence_match=true, grounded=true |
| Phu dinh | `q38` | Nguoi hoc co duoc thi lai cuoi ky neu truot hoc phan khong? | Khong co lan thi lai cuoi ky | Trang 10, Dieu 6 | answer_match=true, evidence_match=true, grounded=true |

Y nghia:

```text
Ba case nay chung minh he thong xu ly duoc paragraph phap quy, bang, va cau hoi phu dinh co citation. Khi demo, nen hien cau tra loi kem trang/Dieu/bang de hoi dong thay ro day khong chi la goi LLM hoi PDF.
```

## 5. Failure case nen chuan bi de tra loi phan bien

Nguon: `results/qa_benchmark/real_qcdt_all_after_table_answer_20260513/per_question.json`, config `routed_grounded`.

| Truong | Noi dung |
|---|---|
| Query id | `q37` |
| Cau hoi | Quy che nay co quy dinh muc hoc phi cu the theo so tien cho tung chuong trinh khong? |
| Gold answer | Khong. Tai lieu chi quy dinh nghia vu nop hoc phi va mot so nguyen tac xu ly hoc phi, khong neu muc hoc phi cu the bang so tien cho tung chuong trinh. |
| Ket qua benchmark | answer_match=false, evidence_match=true, grounded=true, hallucinated=false |
| Loi thuc te | He thong grounded theo citation nhung answer synthesis chua xu ly tot cau hoi xac nhan su vang mat cua thong tin. |
| Huong cai thien | Abstention/negative-evidence reasoning va kiem tra intent "co quy dinh cu the khong". |

Cach noi truoc hoi dong:

```text
Day la failure case co gia tri phan tich: he thong co the grounded theo citation nhung van sai intent khi cau hoi yeu cau xac nhan thong tin khong ton tai. Vi vay huong phat trien quan trong la abstention va negative-evidence reasoning, khong chi tang top-k retrieval.
```

## 6. Checklist da hoan thanh tu yeu cau phan bien

- [x] Chay lai test suite: `65 passed`.
- [x] Chay lai retrieval ablation BM25/dense/hybrid/rerank.
- [x] Chay lai retrieval top-k=5/10/20 cho QASPER va QCDT.
- [x] Chay lai QA ablation routing/evidence checker/citation grounding cho QCDT va Operations.
- [x] Chay lai table ablation default/TATR/hybrid_tatr tren PubTables structure 25 samples.
- [x] Tong hop 3 demo case co citation.
- [x] Giu 1 failure case trung thuc de tra loi phan bien.
