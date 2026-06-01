# Phase 2 - Table-Aware Chunking

## 1. Vì sao cần
Bảng cần chunk riêng để retrieval giữ quan hệ hàng/cột. Nếu chỉ nhét markdown vào paragraph chunk, câu hỏi kiểu lookup dễ lấy sai ô.

## 2. Ba loại chunk
- Summary chunk: mô tả bảng, trang, caption, headers.
- Structure chunk: markdown/csv giữ cấu trúc bảng.
- Row/cell chunks: text hóa từng hàng/ô với metadata citation.

## 3. Ví dụ chunk đầu ra
```text
Bảng Bảng quy đổi điểm, trang 9. Hàng C+, cột Điểm thang 4: 2.5.
```

## 4. Metadata retrieval/citation
`chunk_id`, `doc_id`, `page`, `block_type=table`, `table_id`, `caption`, `row_index`, `col_index`, `row_header`, `col_header`, `cell_text`, `source_bbox`, `citation_target`.

## 5. Test đã thêm
- `tests/test_table_aware_ingest.py`

## 6. Hạn chế
Flag `BOXBIIBOO_ENABLE_TABLE_AWARE_CHUNKING` mặc định tắt để tránh regression. Row/cell chunk phụ thuộc chất lượng cell metadata.
