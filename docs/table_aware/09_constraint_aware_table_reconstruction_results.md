# Constraint-aware table reconstruction results

## Trạng thái triển khai

Đã thêm module:

```text
app/ingest/table_reconstruct.py
```

Flag tích hợp:

```text
BOXBIIBOO_ENABLE_CONSTRAINT_TABLE_RECONSTRUCTION=true/false
```

Khi flag bật, `table_structure_from_rows()` thử chạy constraint-aware reconstruction. Nếu reconstruction lỗi, pipeline fallback về normalized table cũ và không crash.

## Before markdown

Ví dụ lỗi từ bảng thời gian/khối lượng học tập:

```markdown
| Chương trình | Người học | Thời gian Khối tối thiểu lượng |  |
| --- | --- | --- | --- |
| Cử nhân | Tốt nghiệp THPT | 4 năm | 132 tín chỉ |
| Kỹ sư | Tốt nghiệp cử nhân theo chương trình tích hợp | 1,5 năm | 48 tín chỉ |
|  | Tốt nghiệp cử nhân | 2 năm | 60 tín chỉ |
| Thạc sĩ | Tốt nghiệp cử nhân | 2 năm | 60 tín chỉ |
|  | Tốt nghiệp cử nhân theo chương trình tích hợp | 1,5 năm | 48 tín chỉ |
| Tiến sĩ | Tốt Tốt nghiệp nghiệp thạc đại học sĩ | 3 4 năm năm | 106 151 tín tín chỉ chỉ |
```

## After markdown

Sau reconstruction trên bảng mẫu:

```markdown
| Chương trình | Người học | Thời gian | Khối lượng tối thiểu |
| --- | --- | --- | --- |
| Cử nhân | Tốt nghiệp THPT | 4 năm | 132 tín chỉ |
| Kỹ sư | Tốt nghiệp cử nhân theo chương trình tích hợp | 1,5 năm | 48 tín chỉ |
| Kỹ sư | Tốt nghiệp cử nhân | 2 năm | 60 tín chỉ |
| Thạc sĩ | Tốt nghiệp cử nhân | 2 năm | 60 tín chỉ |
| Thạc sĩ | Tốt nghiệp cử nhân theo chương trình tích hợp | 1,5 năm | 48 tín chỉ |
| Tiến sĩ | Tốt nghiệp thạc sĩ | 3 năm | 106 tín chỉ |
| Tiến sĩ | Tốt nghiệp đại học | 4 năm | 151 tín chỉ |
```

## Table records

```json
[
  {"Chương trình": "Cử nhân", "Người học": "Tốt nghiệp THPT", "Thời gian": "4 năm", "Khối lượng tối thiểu": "132 tín chỉ"},
  {"Chương trình": "Kỹ sư", "Người học": "Tốt nghiệp cử nhân theo chương trình tích hợp", "Thời gian": "1,5 năm", "Khối lượng tối thiểu": "48 tín chỉ"},
  {"Chương trình": "Kỹ sư", "Người học": "Tốt nghiệp cử nhân", "Thời gian": "2 năm", "Khối lượng tối thiểu": "60 tín chỉ"},
  {"Chương trình": "Thạc sĩ", "Người học": "Tốt nghiệp cử nhân", "Thời gian": "2 năm", "Khối lượng tối thiểu": "60 tín chỉ"},
  {"Chương trình": "Thạc sĩ", "Người học": "Tốt nghiệp cử nhân theo chương trình tích hợp", "Thời gian": "1,5 năm", "Khối lượng tối thiểu": "48 tín chỉ"},
  {"Chương trình": "Tiến sĩ", "Người học": "Tốt nghiệp thạc sĩ", "Thời gian": "3 năm", "Khối lượng tối thiểu": "106 tín chỉ"},
  {"Chương trình": "Tiến sĩ", "Người học": "Tốt nghiệp đại học", "Thời gian": "4 năm", "Khối lượng tối thiểu": "151 tín chỉ"}
]
```

## Trace

Trace chính trong test:

```text
inferred 4 columns
fill-down vertical merged cells
split merged rows by duration/credit constraints
row 'Tiến sĩ' split into 2 rows
selected best score=...
```

Constraint đạt:

- `duration_pattern = 1.0`
- `credit_pattern = 1.0`
- `no_same_type_merge = 1.0`

## Real PDF check

Đã chạy lại:

```powershell
$env:BOXBIIBOO_ENABLE_TABLE_AWARE_CHUNKING='true'
$env:BOXBIIBOO_ENABLE_CONSTRAINT_TABLE_RECONSTRUCTION='true'
.\.venv-gpu\Scripts\python.exe scripts\build_retrieval_index.py `
  --pdf data\real_pdfs\QCDT_2025_5445_QD-DHBK.pdf `
  --output-dir results\retrieval_index\qcdt_2025_5445_constraint_table_reconstruction `
  --skip-dense
```

Kết quả bảng trang 6 `page_6_p0005_b0005`:

- Constraint reconstruction `status=applied`, score `7.7`.
- Header sau reconstruction: `Chương trình`, `Người học`, `Thời gian`, `Khối lượng tối thiểu`.
- Cell tốt hơn:
  - `Cử nhân` / `Thời gian` = `4 năm`.
  - `Cử nhân` / `Khối lượng tối thiểu` = `132 tín chỉ`.
  - `Tiến sĩ` / `Người học` = `Tốt nghiệp thạc sĩ`, `Thời gian` = `3 năm`, `Khối lượng tối thiểu` = `106 tín chỉ`.
  - `Tiến sĩ` / `Người học` = `Tốt nghiệp đại học`, `Thời gian` = `4 năm`, `Khối lượng tối thiểu` = `151 tín chỉ`.

Output:

```text
results/retrieval_index/qcdt_2025_5445_constraint_table_reconstruction/
```

## Tests/benchmark đã chạy

Tests mới:

```text
tests/test_constraint_table_reconstruction.py
```

Coverage:

- build cell graph từ word boxes;
- nhận ra 4 cột;
- fill-down `Kỹ sư`, `Thạc sĩ`, `Tiến sĩ`;
- split row có 2 duration và 2 credit values;
- markdown đúng;
- csv đúng;
- JSON/table records đúng;
- trace giải thích chọn hypothesis;
- baseline cũ giữ nguyên khi flag tắt.

Benchmark final đã chạy:

```powershell
python -m compileall app scripts
python -m pytest -q
python scripts/benchmark_ingest_suite.py --dataset mock --limit 5 --out results/ingest/mock_after_constraint_table_reconstruction --mode all
```

Kết quả:

- `python -m compileall app scripts`: pass.
- `.\.venv-gpu\Scripts\python.exe -m pytest -q`: `81 passed`.
- Mock ingest benchmark: `success_rate=1.0`, `error_count=0`, `table_structure.f1=1.0`.

## Hạn chế

- Reconstruction hiện mới xử lý nhóm constraint phổ biến cho bảng đào tạo: chương trình, người học, thời gian, tín chỉ.
- Learner split cho case `Tốt Tốt nghiệp nghiệp thạc đại học sĩ` là heuristic có kiểm soát, chưa phải model tổng quát.
- Chưa dùng trực tiếp OCR confidence vào scoring ngoài placeholder `ocr_confidence`.
- Chưa xử lý đầy đủ multi-row header phức tạp, spanning cell nhiều cấp hoặc bảng scan OCR lỗi nặng.
- Real PDF vẫn còn lỗi word assignment ở một số cell, ví dụ `Tốt chương nghiệp trình cử tích nhân hợp theo 1,5 năm`.
- Một số hàng lặp từ extractor gốc có thể được fill-down thành record trùng, cần bước dedupe/row alignment sâu hơn.
