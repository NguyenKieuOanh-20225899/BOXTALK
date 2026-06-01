# Ablation Study

## 1. Retrieval: BM25 vs Dense vs Hybrid vs Hybrid + rerank

| Dataset | Top-k | Best Hit@k | Ket qua chinh |
| --- | ---: | --- | --- |
| SciFact | 5 | Hybrid 0.793 | Hybrid tang so voi BM25/Dense 0.713. |
| QCDT | 10 | Hybrid 0.600 | BM25 manh 0.550, hybrid tang nhe recall/hit. |
| QASPER | 20 | Hybrid/Hybrid rerank 0.392 | Tat ca chien luoc con thap, do evidence dai va multi-hop. |

Kien truc chon hybrid vi phu hop ca keyword-heavy legal PDF va semantic scientific evidence. Rerank khong luon tang Hit@k nen nen coi la tuy chon, khong phai mac dinh bat buoc.

## 2. Table extraction: default vs TATR vs hybrid_tatr

| Backend | Detection F1 | Cell F1@0.75 | Structure F1 | Text assignment F1 | Exact CSV |
| --- | ---: | ---: | ---: | ---: | ---: |
| default | 0.940 | 0.149 | 0.199 | 0.909 | 0.000 |
| tatr | 0.987 | 0.103 | 0.010 | 0.015 | 0.000 |
| hybrid_tatr | 0.987 | 0.944 | 0.772 | 0.999 | 0.480 |

Source: PubTables structure 25 rerun. Hybrid TATR noi bat vi ket hop detection/structure boxes voi OCR/PDF word assignment, thay vi chi dung model box hoac rule text.

## 3. Region-level routing

| Config | Metric | Ket qua |
| --- | --- | --- |
| Mock after region routing | success_rate | 1.000 |
| Mock after region routing | token_f1 | 1.000 |
| Mock after region routing | reading_order_score | 1.000 |
| Mock after region routing | table_structure F1 | 1.000 |

Vi mock dat tran, day la architectural ablation: muc tieu chinh la dam bao router khong pha regression va cho phep cac vung text/layout/table/OCR di qua backend phu hop.

## 4. QASPER top-k

Tu rerun cu va ket qua top-20 dot nay:

| top-k | BM25 Hit | Dense Hit | Hybrid Hit | Hybrid rerank Hit | Nhan xet |
| ---: | ---: | ---: | ---: | ---: | --- |
| 5 | 0.390 | 0.280 | 0.390 | 0.360 | Evidence recall thap ngay ca top-5. |
| 10 | 0.490 | 0.350 | 0.500 | 0.510 | Tang top-k giup retrieval. |
| 20 | 0.378-0.570 | 0.316-0.420 | 0.392-0.570 | 0.392-0.570 | Khac biet theo query set/label; van chua du cho QA free-form. |

Ket luan: tang top-k giup retrieval nhung answer synthesis van la bottleneck lon.

## 5. Grounded QA

| Benchmark | Config | Grounded | Hallucination | Ket luan |
| --- | --- | ---: | ---: | --- |
| QCDT | routed_grounded | 1.000 | 0.000 | On voi benchmark phap quy noi bo. |
| Operations | routed_grounded | 1.000 | 0.000 | Xu ly absence/ambiguous tot hon baseline. |
| SciFact | routed_grounded | 1.000 | 0.000 | Citation tot, answer synthesis con yeu. |
| QASPER | routed_grounded | 1.000 | 0.054 | Van co hallucination trong free-form scientific QA. |

## Ket luan ablation

- Retrieval hybrid la diem nen dua vao pipeline chinh.
- Table-aware retrieval + cell citation can cho cau hoi bang; normal retrieval khong du.
- Grounded QA nen giu citation/evidence checker; bo citation lam hallucination metric xau ro trong ablation cu.
- QASPER khong nen dung de claim chat luong cao, ma nen dung de chi ra gioi han cua pipeline.
