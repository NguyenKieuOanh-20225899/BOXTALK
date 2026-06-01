# Constraint-aware table reconstruction plan

## Lỗi markdown hiện tại

Khi chạy `QCDT_2025_5445_QD-DHBK.pdf`, bảng trang 6 đã được phát hiện thành table-aware chunks nhưng markdown vẫn có lỗi cấu trúc:

```text
| Chương trình | Người học | Thời gian Khối tối thiểu lượng |  |
| Cử nhân | Tốt nghiệp THPT | 4 năm | 132 tín chỉ |
| Kỹ sư | Tốt chương nghiệp trình cử tích nhân hợp theo 1,5 năm |  | 48 tín chỉ |
|  | Tốt nghiệp cử nhân | 2 năm | 60 tín chỉ |
| Tiến sĩ | Tốt Tốt nghiệp nghiệp thạc đại học sĩ | 3 4 năm năm | 106 151 tín tín chỉ chỉ |
```

Các lỗi chính:

- Header `Thời gian` và `Khối lượng tối thiểu` bị dính.
- Một số hàng thiếu giá trị ở cột `Chương trình` do merged cell theo chiều dọc.
- Một số cell bị gom nhiều giá trị cùng kiểu, ví dụ `3 4 năm năm` và `106 151 tín tín chỉ chỉ`.
- Word assignment có thể đọc xen kẽ hai dòng trong cùng một ô.

## Vì sao rule post-process chưa đủ

Sửa markdown bằng rule sau khi bảng đã hỏng chỉ nhìn thấy chuỗi text phẳng. Cách đó khó phân biệt:

- cell thật với cell bị merge;
- header bị dính với header nhiều dòng;
- hai giá trị cùng kiểu nằm trong hai hàng khác nhau;
- lỗi OCR/PDF word order với lỗi cấu trúc bảng.

Rule post-process cũng dễ hardcode theo một bảng cụ thể, làm tăng rủi ro regression cho các bảng khác.

## Hướng graph + constraint

Hướng mới giữ baseline cũ và thêm một bước tùy chọn:

```text
table region
-> TATR structure boxes
-> OCR/PDF word boxes
-> cell graph
-> schema inference
-> multi-hypothesis reconstruction
-> constraint scoring
-> normalized TableCells/table records
-> CSV/Markdown/JSON
-> table-aware retrieval/citation
```

Các bước:

1. Gán word boxes vào row/column boxes để tạo cell graph.
2. Suy luận schema: header, role cột, kiểu dữ liệu.
3. Sinh nhiều hypothesis:
   - baseline matrix;
   - fill-down vertical merged cells;
   - split row khi cell có nhiều duration/credit cùng kiểu.
4. Chấm điểm hypothesis bằng constraint.
5. Chọn bảng logic tốt nhất và export lại records/cells/markdown/csv.

## Constraint scoring

Score dựa trên:

- số cột ổn định;
- header hợp lý;
- alignment hàng/cột;
- datatype consistency;
- duration pattern: `\d+(,\d+)? năm`;
- credit pattern: `\d+ tín chỉ`;
- fill-down vertical merged cells hợp lý;
- không merge nhiều giá trị cùng kiểu vào một cell;
- OCR confidence nếu có.

## Output mong muốn

Với bảng mẫu, output records mong muốn:

| Chương trình | Người học | Thời gian | Khối lượng tối thiểu |
|---|---|---|---|
| Cử nhân | Tốt nghiệp THPT | 4 năm | 132 tín chỉ |
| Kỹ sư | Tốt nghiệp cử nhân theo chương trình tích hợp | 1,5 năm | 48 tín chỉ |
| Kỹ sư | Tốt nghiệp cử nhân | 2 năm | 60 tín chỉ |
| Thạc sĩ | Tốt nghiệp cử nhân | 2 năm | 60 tín chỉ |
| Thạc sĩ | Tốt nghiệp cử nhân theo chương trình tích hợp | 1,5 năm | 48 tín chỉ |
| Tiến sĩ | Tốt nghiệp thạc sĩ | 3 năm | 106 tín chỉ |
| Tiến sĩ | Tốt nghiệp đại học | 4 năm | 151 tín chỉ |

Output vẫn giữ baseline cũ khi flag tắt:

```text
BOXBIIBOO_ENABLE_CONSTRAINT_TABLE_RECONSTRUCTION=false
```

