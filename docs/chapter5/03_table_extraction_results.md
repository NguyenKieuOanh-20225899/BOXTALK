# Table Extraction Results

## Commands chay lai

```powershell
python scripts/benchmark_ingest_suite.py --dataset pubtables_structure --data-dir data/benchmarks/pubtables_structure_ocr_words_5 --limit 5 --out results/ingest/chapter5_pubtables_structure_default_5 --mode table --table-backend default
python scripts/benchmark_ingest_suite.py --dataset pubtables_structure --data-dir data/benchmarks/pubtables_structure_ocr_words_5 --limit 5 --out results/ingest/chapter5_pubtables_structure_tatr_5 --mode table --table-backend tatr
python scripts/benchmark_ingest_suite.py --dataset pubtables_structure --data-dir data/benchmarks/pubtables_structure_ocr_words_5 --limit 5 --out results/ingest/chapter5_pubtables_structure_hybrid_tatr_5 --mode table --table-backend hybrid_tatr
```

## Ket qua mau 5 trong dot nay

| Backend | Detection F1@0.50 | Cell F1@0.50 | Cell F1@0.75 | Structure F1 | Text assignment F1 | Exact CSV | GriTS-con-like |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| default | 0.900 | 0.749 | 0.133 | 0.310 | 0.959 | 0.000 | 0.259 |
| tatr | 1.000 | 0.404 | 0.001 | 0.001 | 0.000 | 0.000 | 0.001 |
| hybrid_tatr | 1.000 | 0.470 | 0.062 | 0.407 | 0.933 | 0.000 | 0.275 |

Mau 5 rat nho nen khong dung de ket luan chinh. No cho thay pipeline chay duoc va hybrid_tatr giu duoc text assignment, nhung exact CSV van bang 0.

## Ket qua ablation 25 mau rerun cu

| Backend | Detection F1@0.50 | Cell F1@0.50 | Cell F1@0.75 | Structure F1 | Text assignment F1 | Row MAE | Col MAE | GriTS-con-like | Exact CSV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| default | 0.940 | 0.650 | 0.149 | 0.199 | 0.909 | 1.960 | 0.800 | 0.146 | 0.000 |
| tatr | 0.987 | 0.491 | 0.103 | 0.010 | 0.015 | 0.600 | 0.000 | 0.006 | 0.000 |
| hybrid_tatr | 0.987 | 0.958 | 0.944 | 0.772 | 0.999 | 0.600 | 0.000 | 0.700 | 0.480 |

Source: `results/ingest/rerun_pubtables_structure_default_25_20260526`, `results/ingest/rerun_pubtables_structure_tatr_25_20260526`, `results/ingest/rerun_pubtables_structure_hybrid_tatr_25_20260526`.

## Constraint-aware reconstruction tren PDF QCDT

Bang "Thoi gian va khoi luong hoc tap chuan" trang 6 trong `QCDT_2025_5445_QD-DHBK.pdf` duoc chuan hoa thanh 4 cot:

| Chuong trinh | Nguoi hoc | Thoi gian | Khoi luong toi thieu |
| --- | --- | --- | --- |
| Cu nhan | Tot nghiep THPT | 4 nam | 132 tin chi |
| Ky su | Tot nghiep cu nhan theo chuong trinh tich hop | 1,5 nam | 48 tin chi |
| Ky su | Tot nghiep cu nhan | 2 nam | 60 tin chi |
| Thac si | Tot nghiep cu nhan theo chuong trinh tich hop | 1,5 nam | 48 tin chi |
| Tien si | Tot nghiep thac si | 3 nam | 106 tin chi |
| Tien si | Tot nghiep dai hoc | 4 nam | 151 tin chi |

Trace noi bat: suy ra 4 cot, tach row Tien si bi merge, day gia tri cot Chuong trinh theo vertical merged cell, dua token thoi gian ra dung cot, deduplicate row lap.

## Ket luan an toan

- Hybrid TATR cai thien table structure tren PubTables subset da chay.
- Cell-level citation co ich khi cau hoi can tra loi o cap o bang.
- Exact CSV/HTML van la gioi han; khong nen claim reconstruct hoan chinh moi bang PDF.
