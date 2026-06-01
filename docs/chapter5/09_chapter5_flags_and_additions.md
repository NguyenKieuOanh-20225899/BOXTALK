# Chapter 5 Flags And Additions

## Muc tieu bo sung

File nay ghi lai cac flag/env can bat va cac benchmark large da chay lai them sau khi phan tich ban dau. Cac muc nay nen duoc them vao Chuong 5 de chung minh ket qua khong chi la tong hop tu results cu.

## Flags / environment

| Flag / env | Gia tri | Dung cho | Ly do |
| --- | --- | --- | --- |
| `BOXBIIBOO_LAYOUT_MODEL_NAME` | `Aryn/deformable-detr-DocLayNet` | DocLayNet, PubLayNet, PubTables detection | Bat model layout backend; neu khong set thi adapter bao "Model layout backend is disabled". |
| `--device` | `cuda` | Layout/table/OCR benchmark | Dung GPU neu co de chay model-backed ingest. |
| `--dataset doclaynet` | N/A | Layout benchmark | Adapter COCO-style DocLayNet. |
| `--dataset publaynet` | N/A | Layout benchmark | Adapter COCO-style PubLayNet. |
| `--dataset pubtables` | N/A | Table detection | PubTables detection XML/images. |
| `--dataset pubtables_structure` | N/A | Table structure | PubTables structure + OCR/PDF word boxes. |
| `--table-backend default` | N/A | Table baseline | OCR/grid baseline; co the cham voi 25 mau. |
| `--table-backend tatr` | N/A | TATR baseline | Pretrained TATR boxes. |
| `--table-backend hybrid_tatr` | N/A | Table ablation chinh | TATR boxes + word assignment. |
| `--dataset ocr --data-dir <folder>/ocr` | N/A | OCR benchmarks | OCR adapter can `ocr_samples.jsonl` nam truc tiep trong data-dir. |

## Commands bo sung da chay

```powershell
$env:BOXBIIBOO_LAYOUT_MODEL_NAME='Aryn/deformable-detr-DocLayNet'
python scripts/benchmark_ingest_suite.py --dataset doclaynet --data-dir data/benchmarks/doclaynet --limit 0 --out results/ingest/chapter5_doclaynet_full_rerun --mode layout --device cuda
python scripts/benchmark_ingest_suite.py --dataset publaynet --data-dir data/benchmarks/publaynet --limit 0 --out results/ingest/chapter5_publaynet_full_rerun --mode layout --device cuda
python scripts/benchmark_ingest_suite.py --dataset pubtables --data-dir data/benchmarks/pubtables_detection --limit 500 --out results/ingest/chapter5_pubtables_detection_500_rerun_model --mode table --device cuda
python scripts/benchmark_ingest_suite.py --dataset ocr --data-dir data/benchmarks/ocr_scan_25/ocr --limit 25 --out results/ingest/chapter5_ocr_scan_25_rerun_fixed_seq --mode ocr --device cuda
python scripts/benchmark_ingest_suite.py --dataset ocr --data-dir data/benchmarks/funsd/ocr --limit 25 --out results/ingest/chapter5_funsd_ocr_25_rerun_fixed_seq --mode ocr --device cuda
```

## Ket qua bo sung

| Benchmark | Samples | Output | Metric chinh |
| --- | ---: | --- | --- |
| DocLayNet full rerun | 49 | `results/ingest/chapter5_doclaynet_full_rerun/summary.json` | F1@0.50 0.849, F1@0.75 0.807 |
| PubLayNet full rerun | 100 | `results/ingest/chapter5_publaynet_full_rerun/summary.json` | F1@0.50 0.739, F1@0.75 0.708 |
| PubTables detection rerun | 500 | `results/ingest/chapter5_pubtables_detection_500_rerun_model/summary.json` | F1@0.50 0.975, F1@0.75 0.914 |
| OCR scan synthetic rerun | 25 | `results/ingest/chapter5_ocr_scan_25_rerun_fixed_seq/summary.json` | OCR token F1 1.000, CER 0.000 |
| FUNSD OCR rerun | 25 | `results/ingest/chapter5_funsd_ocr_25_rerun_fixed_seq/summary.json` | OCR token F1 0.826, CER 0.466, WER 0.515 |

## Runs can ghi chu trong bao cao

| Run | Trang thai | Ghi chu |
| --- | --- | --- |
| `results/ingest/chapter5_doclaynet_5` | Failed config | Chay truoc khi bat `BOXBIIBOO_LAYOUT_MODEL_NAME`; khong dung lam ket qua model. |
| `results/ingest/chapter5_pubtables_detection_500_rerun` | Failed config | Chay PubTables detection khi layout backend chua bat; da thay bang `_rerun_model`. |
| `results/ingest/chapter5_pubtables_structure_default_25_rerun` | Timeout | Default OCR-backed structure 25 mau qua 20 phut chua tao summary; dung rerun 25 cu va mau 5 moi chay. |
| OCR parallel runs | Failed/no summary | Chay PaddleOCR song song bi dung truoc khi ghi summary; da rerun OCR scan/FUNSD tuan tu thanh cong. |

## Phan nen them vao Chuong 5

- Them bang "Cau hinh benchmark" gom `BOXBIIBOO_LAYOUT_MODEL_NAME`, `--device cuda`, `--table-backend`.
- Cap nhat bang ingest results de dung output rerun moi thay vi noi DocLayNet/PubLayNet chi tong hop tu results cu.
- Ghi ro PubTables detection 500 da rerun that voi model layout bat.
- Ghi ro OCR scan synthetic khong dai dien OCR thuc te; FUNSD OCR la benchmark thuc te hon va co CER/WER cao hon.
- Giu han che: PubTables structure default 25 rerun moi bi timeout, nhung da co rerun 25 cu va mau 5 moi de kiem tra pipeline.
