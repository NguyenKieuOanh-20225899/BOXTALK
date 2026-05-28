# Phase 4 - Cell-Level Citation

## 1. Format mới
- Có caption, row, col: `Trang {page}, bảng '{caption}', hàng '{row_header}', cột '{col_header}'.`
- Không caption: `Trang {page}, bảng {table_id}, hàng '{row_header}', cột '{col_header}'.`
- Thiếu row/col: `Trang {page}, bảng {table_id}.`

## 2. Ví dụ citation text
```text
Trang 3, bảng 'Quy đổi', hàng 'C+', cột 'Điểm thang 4'.
```

## 3. Metadata cần thiết
`page`, `block_type`, `table_id`, `caption`, `row_index`, `col_index`, `row_header`, `col_header`, `cell_text`, `bbox/source_bbox`.

## 4. Test đã thêm
- `tests/test_table_aware_retrieval_and_citations.py`

## 5. Hạn chế
Nếu extractor không tạo được row/column header, citation fallback về table-level để không tạo thông tin giả.
