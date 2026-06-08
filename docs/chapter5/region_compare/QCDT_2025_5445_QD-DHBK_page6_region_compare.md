# Region Routing ON/OFF Comparison

- PDF: `data\real_pdfs\QCDT_2025_5445_QD-DHBK.pdf`
- Page: `6`

## Summary

| Config | Used backend | Page blocks | Page chunks | Block types | Route backends |
| --- | --- | ---: | ---: | --- | --- |
| Region OFF | `text` | 12 | 8 | `{'paragraph': 4, 'list_item': 2, 'heading': 6}` | `{'<none>': 12}` |
| Region ON | `region_routed` | 6 | 5 | `{'paragraph': 1, 'list_item': 2, 'heading': 2, 'table': 1}` | `{'text': 5, 'table': 1}` |

## Interpretation Checklist

- `Used backend` cho biet pipeline chon backend nao sau validation/fallback.
- `Route backends` chi co y nghia ro nhat khi `region_routed` duoc dung.
- Neu Region ON co `route_backend`/`region_id` trong block metadata, co the trace tung block ve region goc.
- So sanh block/chunk count de xem region co lam tach nho noi dung hay giu cau truc bang/hinh tot hon khong.

## Region OFF Blocks

### `p0005_b0001`

- Type: `paragraph`
- Source mode: `text`
- Reading order: `0`
- BBox: `(70.94400024414062, 68.15264892578125, 541.8831176757812, 181.1393280029297)`
- Trace meta: `{'backend': 'pymupdf'}`

```text
ưu hóa thời gian đào tạo cho người học. CTĐT tích hợp cử nhân-kỹ sư, cử nhân-thạc sĩ có thời gian thiết kế là 5,5 năm và khối lượng học tập 180 TC. CTĐT tích hợp bao gồm hai bậc trình độ: Cử nhân (thời gian đào tạo 4 năm
```

### `p0005_b0002`

- Type: `list_item`
- Source mode: `text`
- Reading order: `1`
- BBox: `(70.94400024414062, 189.23260498046875, 541.7549438476562, 335.12933349609375)`
- Trace meta: `{'backend': 'pymupdf'}`

```text
4. Chương trình ELITECH (từ viết tắt của cụm từ Elite Technology Program) là CTĐT chất lượng cao, thể hiện ở một số yếu tố sau: chất lượng đầu vào; chuẩn đầu ra; giá trị văn bằng tốt nghiệp; phương thức tổ chức đào tạo; 
```

### `p0005_b0003`

- Type: `list_item`
- Source mode: `text`
- Reading order: `2`
- BBox: `(70.94400024414062, 343.2226257324219, 541.8865356445312, 505.54931640625)`
- Trace meta: `{'backend': 'pymupdf'}`

```text
5. Chương trình đào tạo Tài năng thuộc nhóm chương trình ELITECH, được thiết kế nhằm phát hiện, bồi dưỡng những sinh viên có năng lực xuất sắc, tư duy sáng tạo và khả năng nghiên cứu chuyên sâu. Sinh viên tham gia chương
```

### `p0005_b0004`

- Type: `heading`
- Source mode: `text`
- Reading order: `3`
- BBox: `(70.94400024414062, 513.642578125, 541.6923828125, 544.4293823242188)`
- Trace meta: `{'backend': 'pymupdf'}`

```text
6. Thời gian và khối lượng học tập chuẩn đối với các CTĐT theo hình thức chính quy không kể các học phần bổ sung kiến thức được quy định như sau:
```

### `p0005_b0005`

- Type: `heading`
- Source mode: `text`
- Reading order: `4`
- BBox: `(119.30000305175781, 547.1226196289062, 491.8599853515625, 569.6593627929688)`
- Trace meta: `{'backend': 'pymupdf'}`

```text
Chương trình Người học Thời gian Khối lượng
```

### `p0005_b0006`

- Type: `paragraph`
- Source mode: `text`
- Reading order: `5`
- BBox: `(119.30000305175781, 563.5926513671875, 483.3399963378906, 628.2193603515625)`
- Trace meta: `{'backend': 'pymupdf'}`

```text
tối thiểu Cử nhân Tốt nghiệp THPT 4 năm 132 tín chỉ Tốt nghiệp cử nhân theo chương trình tích hợp 1,5 năm 48 tín chỉ
```

### `p0005_b0008`

- Type: `heading`
- Source mode: `text`
- Reading order: `6`
- BBox: `(119.30000305175781, 614.1126098632812, 153.74000549316406, 628.4593505859375)`
- Trace meta: `{'backend': 'pymupdf'}`

```text
Kỹ sư
```

### `p0005_b0009`

- Type: `heading`
- Source mode: `text`
- Reading order: `7`
- BBox: `(211.49000549316406, 630.7926025390625, 476.2599792480469, 645.1393432617188)`
- Trace meta: `{'backend': 'pymupdf'}`

```text
Tốt nghiệp cử nhân 2 năm 60 tín chỉ
```

### `p0005_b0010`

- Type: `paragraph`
- Source mode: `text`
- Reading order: `8`
- BBox: `(211.49000549316406, 647.712646484375, 476.2599792480469, 695.5393676757812)`
- Trace meta: `{'backend': 'pymupdf'}`

```text
Tốt nghiệp cử nhân 2 năm 60 tín chỉ Tốt nghiệp cử nhân theo chương trình tích hợp 1,5 năm 48 tín chỉ
```

### `p0005_b0011`

- Type: `heading`
- Source mode: `text`
- Reading order: `9`
- BBox: `(119.30000305175781, 664.5126342773438, 160.4600067138672, 678.859375)`
- Trace meta: `{'backend': 'pymupdf'}`

```text
Thạc sĩ
```

### `p0005_b0012`

- Type: `paragraph`
- Source mode: `text`
- Reading order: `10`
- BBox: `(119.30000305175781, 698.1126098632812, 482.739990234375, 729.3793334960938)`
- Trace meta: `{'backend': 'pymupdf'}`

```text
Tiến sĩ Tốt nghiệp thạc sĩ 3 năm 106 tín chỉ Tốt nghiệp đại học 4 năm 151 tín chỉ
```

### `p0005_b0013`

- Type: `heading`
- Source mode: `text`
- Reading order: `11`
- BBox: `(70.94400024414062, 737.9725952148438, 541.4331665039062, 768.7593383789062)`
- Trace meta: `{'backend': 'pymupdf'}`

```text
7. Thời gian theo kế hoạch học tập chuẩn toàn khoá đối với hình thức đào tạo vừa làm vừa học dài hơn tối thiểu 20% so với hình thức đào tạo chính quy của cùng CTĐT.
```


## Region ON Blocks

### `p0005_b0001`

- Type: `paragraph`
- Source mode: `text`
- Reading order: `0`
- BBox: `(70.94400024414062, 68.15264892578125, 541.8831176757812, 181.1393280029297)`
- Trace meta: `{'backend': 'region_routed', 'region_id': 'p0005_text_0002', 'region_type': 'paragraph', 'region_kind': 'paragraph', 'region_bbox': (70.94400024414062, 68.15264892578125, 541.8831176757812, 181.1393280029297), 'page_number': 6, 'route_backend': 'text', 'route_reason': 'detected_text_region', 'confidence': 1.0, 'source': 'pdf_text_block', 'detection_source': 'pdf_text_block', 'fallback_used': False}`

```text
ưu hóa thời gian đào tạo cho người học. CTĐT tích hợp cử nhân-kỹ sư, cử nhân-thạc sĩ có thời gian thiết kế là 5,5 năm và khối lượng học tập 180 TC. CTĐT tích hợp bao gồm hai bậc trình độ: Cử nhân (thời gian đào tạo 4 năm
```

### `p0005_b0002`

- Type: `list_item`
- Source mode: `text`
- Reading order: `1`
- BBox: `(70.94400024414062, 189.23260498046875, 541.7549438476562, 335.12933349609375)`
- Trace meta: `{'backend': 'region_routed', 'region_id': 'p0005_text_0003', 'region_type': 'list_item', 'region_kind': 'list_item', 'region_bbox': (70.94400024414062, 189.23260498046875, 541.7549438476562, 335.12933349609375), 'page_number': 6, 'route_backend': 'text', 'route_reason': 'detected_text_region', 'confidence': 1.0, 'source': 'pdf_text_block', 'detection_source': 'pdf_text_block', 'fallback_used': False}`

```text
4. Chương trình ELITECH (từ viết tắt của cụm từ Elite Technology Program) là CTĐT chất lượng cao, thể hiện ở một số yếu tố sau: chất lượng đầu vào; chuẩn đầu ra; giá trị văn bằng tốt nghiệp; phương thức tổ chức đào tạo; 
```

### `p0005_b0003`

- Type: `list_item`
- Source mode: `text`
- Reading order: `2`
- BBox: `(70.94400024414062, 343.2226257324219, 541.8865356445312, 505.54931640625)`
- Trace meta: `{'backend': 'region_routed', 'region_id': 'p0005_text_0004', 'region_type': 'list_item', 'region_kind': 'list_item', 'region_bbox': (70.94400024414062, 343.2226257324219, 541.8865356445312, 505.54931640625), 'page_number': 6, 'route_backend': 'text', 'route_reason': 'detected_text_region', 'confidence': 1.0, 'source': 'pdf_text_block', 'detection_source': 'pdf_text_block', 'fallback_used': False}`

```text
5. Chương trình đào tạo Tài năng thuộc nhóm chương trình ELITECH, được thiết kế nhằm phát hiện, bồi dưỡng những sinh viên có năng lực xuất sắc, tư duy sáng tạo và khả năng nghiên cứu chuyên sâu. Sinh viên tham gia chương
```

### `p0005_b0004`

- Type: `heading`
- Source mode: `text`
- Reading order: `3`
- BBox: `(70.94400024414062, 513.642578125, 541.6923828125, 544.4293823242188)`
- Trace meta: `{'backend': 'region_routed', 'region_id': 'p0005_text_0005', 'region_type': 'list_item', 'region_kind': 'list_item', 'region_bbox': (70.94400024414062, 513.642578125, 541.6923828125, 544.4293823242188), 'page_number': 6, 'route_backend': 'text', 'route_reason': 'detected_text_region', 'confidence': 1.0, 'source': 'pdf_text_block', 'detection_source': 'pdf_text_block', 'fallback_used': False}`

```text
6. Thời gian và khối lượng học tập chuẩn đối với các CTĐT theo hình thức chính quy không kể các học phần bổ sung kiến thức được quy định như sau:
```

### `p0005_b0005`

- Type: `table`
- Source mode: `layout`
- Reading order: `4`
- BBox: `(113.87333848741319, 546.0220092773437, 495.84664238823785, 731.0879516601562)`
- Trace meta: `{'backend': 'table_words_grid', 'region_id': 'p0005_table_0000', 'region_type': 'table', 'region_kind': 'table', 'region_bbox': (113.87333848741319, 546.0220092773437, 495.84664238823785, 731.0879516601562), 'page_number': 6, 'route_backend': 'table', 'route_reason': 'detected_table_region', 'confidence': 1.0, 'source': 'native_or_text_cluster', 'detection_source': 'native_or_text_cluster', 'fallback_used': False, 'table_backend': 'table_words_grid', 'table_row_count': 7, 'table_col_count': 4, 'table_cell_count': 23}`

```text
Chương trình | Người học | Thời gian Khối tối thiểu lượng | Cử nhân | Tốt nghiệp THPT | 4 năm | 132 tín chỉ Kỹ sư | Tốt chương nghiệp trình cử tích nhân hợp theo 1,5 năm | | 48 tín chỉ | Tốt nghiệp cử nhân | 2 năm | 60 t
```

### `p0005_b0006`

- Type: `heading`
- Source mode: `text`
- Reading order: `5`
- BBox: `(70.94400024414062, 737.9725952148438, 541.4331665039062, 768.7593383789062)`
- Trace meta: `{'backend': 'region_routed', 'region_id': 'p0005_text_0014', 'region_type': 'list_item', 'region_kind': 'list_item', 'region_bbox': (70.94400024414062, 737.9725952148438, 541.4331665039062, 768.7593383789062), 'page_number': 6, 'route_backend': 'text', 'route_reason': 'detected_text_region', 'confidence': 1.0, 'source': 'pdf_text_block', 'detection_source': 'pdf_text_block', 'fallback_used': False}`

```text
7. Thời gian theo kế hoạch học tập chuẩn toàn khoá đối với hình thức đào tạo vừa làm vừa học dài hơn tối thiểu 20% so với hình thức đào tạo chính quy của cùng CTĐT.
```
