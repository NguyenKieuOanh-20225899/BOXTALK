# Phase 1 - Hybrid TATR Output

## 1. File đã sửa
- `app/ingest/extract/table.py`
- `app/ingest/schemas.py`

## 2. Schema mới
`TableCell` hỗ trợ `table_id`, `page`, `row_index`, `col_index`, `row_span`, `col_span`, `text`, `bbox`, `grid_bbox`, `confidence`, `row_header`, `col_header`, `is_header`, `source_words`, `metadata`.

`TableBlock` là adapter quanh `BlockNode` table, giữ `block_id`, `table_id`, `page`, `bbox`, `caption`, `cells`, `csv`, `markdown`, `html`, `source`, `extraction_trace`, `citation_metadata`.

## 3. Ví dụ TableCell JSON
```json
{
  "table_id": "page_9_table_1",
  "page": 9,
  "row": 1,
  "col": 1,
  "row_span": 1,
  "col_span": 1,
  "text": "2.5",
  "row_header": "C+",
  "col_header": "Điểm thang 4",
  "is_header": false
}
```

## 4. Ví dụ CSV/Markdown sinh từ cells
CSV:
```text
Điểm chữ,Điểm thang 4
C+,2.5
```

Markdown:
```markdown
| Điểm chữ | Điểm thang 4 |
| --- | --- |
| C+ | 2.5 |
```

## 5. Test đã thêm
- `tests/test_table_aware_ingest.py`

## 6. Hạn chế
Header inference hiện an toàn và đơn giản: hàng đầu là column header, cột đầu là row header. Multi-row header và merged header phức tạp vẫn cần cải tiến.
