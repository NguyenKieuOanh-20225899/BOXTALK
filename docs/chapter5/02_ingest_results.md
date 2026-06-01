# Ingest Results

## Commands

```powershell
python -m compileall app scripts
python -m pytest -q
python scripts/benchmark_ingest_suite.py --dataset mock --limit 5 --out results/ingest/mock_chapter5_final --mode all
$env:BOXBIIBOO_LAYOUT_MODEL_NAME='Aryn/deformable-detr-DocLayNet'; python scripts/benchmark_ingest_suite.py --dataset doclaynet --data-dir data/benchmarks/doclaynet --limit 0 --out results/ingest/chapter5_doclaynet_full_rerun --mode layout --device cuda
$env:BOXBIIBOO_LAYOUT_MODEL_NAME='Aryn/deformable-detr-DocLayNet'; python scripts/benchmark_ingest_suite.py --dataset publaynet --data-dir data/benchmarks/publaynet --limit 0 --out results/ingest/chapter5_publaynet_full_rerun --mode layout --device cuda
$env:BOXBIIBOO_LAYOUT_MODEL_NAME='Aryn/deformable-detr-DocLayNet'; python scripts/benchmark_ingest_suite.py --dataset pubtables --data-dir data/benchmarks/pubtables_detection --limit 500 --out results/ingest/chapter5_pubtables_detection_500_rerun_model --mode table --device cuda
python scripts/benchmark_ingest_suite.py --dataset ocr --data-dir data/benchmarks/ocr_scan_25/ocr --limit 25 --out results/ingest/chapter5_ocr_scan_25_rerun_fixed_seq --mode ocr --device cuda
python scripts/benchmark_ingest_suite.py --dataset ocr --data-dir data/benchmarks/funsd/ocr --limit 25 --out results/ingest/chapter5_funsd_ocr_25_rerun_fixed_seq --mode ocr --device cuda
```

## Ket qua chay lai

| Benchmark | Samples | Success | Metric chinh | Ghi chu |
| --- | ---: | ---: | --- | --- |
| Mock ingest | 5 | 1.000 | token_f1 1.000, reading_order 1.000, table_structure F1 1.000 | Regression pass. |
| DocLayNet layout | 49 | 1.000 | F1@0.50 0.849, F1@0.75 0.807 | Full rerun voi layout model bat. |
| PubLayNet layout | 100 | 1.000 | F1@0.50 0.739, F1@0.75 0.708 | Full rerun voi layout model bat. |
| PubTables detection | 500 | 1.000 | F1@0.50 0.975, F1@0.75 0.914 | Full rerun 500 mau. |
| OCR scan synthetic | 25 | 1.000 | OCR token F1 1.000, CER 0.000 | Synthetic image-only PDF, khong dai dien OCR thuc te. |
| FUNSD OCR | 25 | 1.000 | OCR token F1 0.826, CER 0.466, WER 0.515 | OCR tai lieu form thuc te. |

Lan chay DocLayNet loi cau hinh ban dau duoc thay bang rerun full sau khi bat `BOXBIIBOO_LAYOUT_MODEL_NAME`.

## Ket qua tong hop tu results cu

| Benchmark | Samples | Metric chinh | Source |
| --- | ---: | --- | --- |
| OCR-D PAGE-XML | 19 | OCR token F1 0.725, CER 0.602, WER 0.636 | `results/ingest/ocrd_pagexml_19_ocr_improve_after_aux_filter_20260517/summary.json` |
| Bast-Korzen proxy | all | token F1 0.998 | `results/ingest/bastkorzen_all_large_20260517/summary.json` |
| Nougat proxy | all | token F1 0.628 | `results/ingest/nougat_all_large_20260517/summary.json` |

## Nhan xet

- Region-level routing giup tach text/layout/table/OCR thanh cac duong xu ly rieng, phu hop voi PDF hon hop.
- Mock benchmark dat tran, nen chi co gia tri regression va architectural sanity check.
- Layout/OCR that phu thuoc model backend va chat luong scan; can ghi ro cau hinh khi trinh bay.
- Ket qua OCR scan synthetic khong nen dien giai thanh OCR tieng Viet thuc te hoan hao.
