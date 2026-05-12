# Table Structure Improvement - 2026-05-12

Muc tieu cua pass nay la cai thien post-processing table structure tren PubTables-1M OTSL, khong train model moi va khong hardcode theo sample.

## Before / After

Dataset: `pubtables_structure`, 25 mau, OCR GPU, mode `table`.

| Metric | Before | After | Ghi chu |
|---|---:|---:|---|
| table detection F1@IoU 0.50 | 0.927 | 0.967 | Khong giam detection; table bbox tot hon sau khi trim caption/footnote. |
| table cell IoU@0.50 F1 | 0.521 | 0.668 | Tang ro, dat tieu chi pass. |
| table cell IoU@0.75 F1 | 0.163 | 0.185 | Tang nhe; bbox cell van chua that khit. |
| table text cell structure F1 | 0.158 | 0.169 | Tang nhe; con bi anh huong boi row/col exact va OCR text. |
| text assignment F1 | n/a | 0.963 | Khi cell bbox match duoc, text gan vao cell kha dung. |
| row count error mean | n/a | 2.24 | Loi chinh con lai la tach/gop row. |
| col count error mean | n/a | 0.84 | Column grouping da kha hon row grouping. |
| exact CSV | 0.000 | 0.000 | Exact match qua nghiem ngat voi OCR + merged cell. |
| exact HTML | n/a | 0.000 | HTML khac ground truth do rowspan/colspan va OCR text chua hoan hao. |

## Loi Chinh Phat Hien

- Caption va footnote o tren/duoi bang bi OCR table cluster keo vao lam row gia.
- Header/group header rong nhieu cot lam thuat toan cu gom nham nhieu cot thanh mot cot lon.
- Cac cell co text nhieu dong lam row bi tach thanh nhieu physical line.
- Text OCR co loi nho, dau cau va ky hieu khoa hoc khac ground truth, lam exact CSV/HTML kho dat.
- Merged cell/group header trong PubTables OTSL co colspan, trong khi OCR chi tra ve text box rieng le.

## Cai Tien Da Lam

- Them schema noi bo `Table`, `TableRow`, `TableCell` voi `row_span`, `col_span`, `bbox`, `source_boxes`, `confidence`.
- Infer column band bang interval-overlap thay vi chi dua vao `x0`.
- Loai wide caption/footnote o canh tren/duoi neu chung la chuoi edge noise.
- Dung narrow text boxes de tao cot, sau do cho header/value rong vua phai mo rong column band neu chi overlap mot cot.
- Dung row band + column band de reconstruct cell bbox thay vi lay nguyen OCR box.
- Phat hien colspan don gian khi mot text box overlap nhieu column bands.
- Them debug JSON moi sample trong `results/ingest/pubtables_structure_25_after_structure_fix/table_debug/`.
- Them metric breakdown: `cell_precision_iou50`, `cell_recall_iou50`, `cell_f1_iou50`, `text_assignment_f1`, `row_count_error`, `col_count_error`, `empty_cell_rate`.

## Lenh Da Chay

```powershell
.\.venv-ocr-gpu\Scripts\python.exe scripts\prepare_pubtables_structure_subset.py --limit 25 --out data\benchmarks\pubtables_structure

.\.venv-ocr-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset pubtables_structure --data-dir data\benchmarks\pubtables_structure --limit 25 --out results\ingest\pubtables_structure_25_before_structure_fix --mode table --save-predictions

.\.venv-ocr-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset pubtables_structure --data-dir data\benchmarks\pubtables_structure --limit 25 --out results\ingest\pubtables_structure_25_after_structure_fix --mode table --save-predictions

.\.venv-gpu\Scripts\python.exe -m compileall app scripts
.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset mock --limit 5 --out results\ingest\mock_recheck --mode all --save-predictions
.\.venv-gpu\Scripts\python.exe -m pytest -q
```

Validation:

```text
compileall: OK
mock_recheck: success_rate = 1.000
pytest -q: 34 passed
```

## Han Che Con Lai

Exact CSV/HTML van kho tang manh vi no yeu cau dung dong, cot, text, merged cell va markup gan nhu tuyet doi. Hien tai cell bbox va text assignment da tot hon, nhung row grouping van sai o nhieu mau: 21/25 mau con row_count_error > 0. Vi vay nen dua ket qua nay vao bao cao theo huong trung thuc: "da co benchmark structure that va cell bbox F1 tang ro", khong claim full table reconstruction da hoan chinh.
