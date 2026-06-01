# Region-level Routing Ablation

## Muc tieu

Muc tieu cua ablation nay la kiem tra anh huong cua bien cau hinh
`BOXBIIBOO_ENABLE_REGION_ROUTING` khi bat/tat region-level routing trong ingest
pipeline.

Region-level routing duoc thiet ke de xu ly PDF hon hop theo tung vung:

```text
PDF page
-> detect_regions(page)
-> text/table/image regions
-> route tung region sang text/table/OCR backend
-> normalize/clean/structure/chunk
```

Ket qua trong file nay khong duoc dung de claim "region routing luon tot hon".
Ket luan an toan hon la: region routing tang tinh linh hoat cua kien truc ingest,
nhung chat luong cuoi cung can duoc kiem chung theo tung dataset va tung backend.

## Commands da chay

### Mock ingest

```powershell
$env:BOXBIIBOO_ENABLE_REGION_ROUTING='0'
.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset mock --limit 5 --out results/ingest/ablation_region_off_mock --mode all

$env:BOXBIIBOO_ENABLE_REGION_ROUTING='1'
.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset mock --limit 5 --out results/ingest/ablation_region_on_mock --mode all
```

### PubTables detection

```powershell
$env:BOXBIIBOO_LAYOUT_MODEL_NAME='Aryn/deformable-detr-DocLayNet'
$env:BOXBIIBOO_ENABLE_REGION_ROUTING='0'
.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset pubtables --data-dir data/benchmarks/pubtables_detection --limit 100 --out results/ingest/ablation_region_off_pubtables_100 --mode table --device cuda

$env:BOXBIIBOO_LAYOUT_MODEL_NAME='Aryn/deformable-detr-DocLayNet'
$env:BOXBIIBOO_ENABLE_REGION_ROUTING='1'
.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset pubtables --data-dir data/benchmarks/pubtables_detection --limit 100 --out results/ingest/ablation_region_on_pubtables_100 --mode table --device cuda
```

### FUNSD OCR

```powershell
$env:BOXBIIBOO_ENABLE_REGION_ROUTING='0'
.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset ocr --data-dir data/benchmarks/funsd/ocr --limit 25 --out results/ingest/ablation_region_off_funsd_25 --mode ocr --device cuda

$env:BOXBIIBOO_ENABLE_REGION_ROUTING='1'
.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset ocr --data-dir data/benchmarks/funsd/ocr --limit 25 --out results/ingest/ablation_region_on_funsd_25 --mode ocr --device cuda
```

Lenh FUNSD region ON bi loi sau khi khoi tao PaddleOCR va khong tao duoc
`summary.json`. De doi chieu chat luong OCR khi region mac dinh bat, su dung lai
ket qua Chapter 5 da chay truoc:

```text
results/ingest/chapter5_funsd_ocr_25_rerun_fixed_seq/summary.json
```

## Ket qua

| Dataset | Region | Samples | Backend count | Success | Main metric | Latency mean |
| --- | --- | ---: | --- | ---: | --- | ---: |
| Mock | OFF | 5 | `text: 5` | 1.000 | token F1 1.000, reading order 1.000, table structure F1 1.000 | 0.006s |
| Mock | ON | 5 | `region_routed: 5` | 1.000 | token F1 1.000, reading order 1.000, table structure F1 1.000 | 0.010s |
| PubTables detection | OFF | 100 | `model_layout_direct: 100` | 1.000 | table detection F1@0.50 0.960, F1@0.75 0.870 | 0.593s |
| PubTables detection | ON | 100 | `model_layout_direct: 100` | 1.000 | table detection F1@0.50 0.960, F1@0.75 0.870 | 0.578s |
| FUNSD OCR | OFF | 25 | `ocr: 25` | 1.000 | OCR token F1 0.826, CER 0.466, WER 0.515 | 10.442s |
| FUNSD OCR | ON* | 25 | `ocr: 25` | 1.000 | OCR token F1 0.826, CER 0.466, WER 0.515 | 33.890s |

`ON*`: su dung ket qua `chapter5_funsd_ocr_25_rerun_fixed_seq` do lenh rerun
region ON rieng bi loi PaddleOCR va khong tao summary.

## Dien giai

### Mock

Mock la benchmark duy nhat trong nhom nay the hien ro tac dong cua flag region
trong full ingest path:

- Region OFF dung backend `text`.
- Region ON dung backend `region_routed`.
- Cac metric noi dung deu giu nguyen o muc 1.000.
- Latency tang nhe tu 0.006s len 0.010s do co buoc detect va route regions.

Ket luan: tren mock, region routing khong gay regression ve noi dung, reading
order hay table structure. Tuy nhien mock la du lieu synthetic, khong phai bang
chung chat luong tren PDF thuc te.

### PubTables detection

Ca Region OFF va ON deu co backend count la `model_layout_direct: 100`.
Dieu nay cho thay benchmark PubTables detection trong `benchmark_ingest_suite.py`
dang chay duong danh gia table detection truc tiep bang layout model, khong di
qua full `region_routed` ingest pipeline.

Vi vay, ket qua PubTables detection khong phai bang chung truc tiep rang region
ON tot hon region OFF. No chi cho thay voi duong benchmark detection hien tai,
flag region routing khong lam thay doi ket qua: F1@0.50 deu bang 0.960.

### FUNSD OCR

Region OFF di qua backend `ocr: 25`. Ket qua OCR trung voi ket qua Chapter 5 da
chay truoc: OCR token F1 0.826, CER 0.466, WER 0.515.

Lenh region ON rieng bi loi sau khi khoi tao PaddleOCR va khong tao summary.
Ket qua ON trong bang lay tu rerun Chapter 5 truoc do, cung co backend `ocr: 25`.
Do do khong nen so sanh latency OFF/ON tren FUNSD nhu mot ket qua ablation chat
che; OCR latency phu thuoc khoi tao model, cache va trang thai runtime.

## Co nen noi ve thoi gian khong?

Co, nhung chi nen noi nhu metric phu.

Nen trinh bay:

- latency mean de cho thay chi phi tinh toan;
- region routing co overhead nhe tren mock do them buoc detect/route;
- latency OCR khong on dinh vi phu thuoc khoi tao PaddleOCR va cache model;
- voi benchmark task-specific nhu PubTables detection, latency khong phan anh
  full region pipeline neu backend count khong phai `region_routed`.

Khong nen ket luan:

- region ON nhanh hon hay cham hon mot cach tong quat;
- latency FUNSD ON/OFF la so sanh cong bang, vi ket qua ON lay tu lan chay khac.

## Ket luan an toan dua vao do an

Region-level routing la cai tien kien truc cho ingest PDF hon hop. Tren mock
benchmark, region routing khong gay regression va tao duoc duong backend
`region_routed`. Tuy nhien, cac benchmark PubTables detection va FUNSD OCR hien
tai la task-specific path nen chua chung minh truc tiep chat luong ON/OFF cua
full region pipeline. Vi vay, trong Chuong 5 nen trinh bay region-level routing
nhu mot architectural ablation/sanity check, con cac claim chat luong chinh nen
dua tren DocLayNet, PubLayNet, PubTables structure, OCR va QA/retrieval.

De chung minh manh hon trong tuong lai, can them benchmark full ingest paired:

```text
same real PDFs
-> region OFF full ingest
-> region ON full ingest
-> compare block type accuracy, table recall, reading order, downstream retrieval/QA
```

