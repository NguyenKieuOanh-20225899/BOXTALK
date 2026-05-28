# Phase 3 - Table-Aware Retrieval

## 1. Query types
- `table_lookup`
- `table_cell_lookup`
- `table_reverse_lookup`
- `general`

## 2. Scoring/boosting
Khi `BOXBIIBOO_ENABLE_TABLE_AWARE_RETRIEVAL=true`, heuristic reranker cộng điểm cho table evidence theo:
- `block_type=table` hoặc `is_table_chunk`;
- match `row_header`;
- match `col_header`;
- match caption/table title;
- match `cell_text` cho reverse lookup.

Paragraph chunk bị giảm nhẹ trong table lookup nếu không có table metadata.

## 3. Ví dụ trace
```json
{
  "query_type": "table_lookup",
  "table_boost_applied": true,
  "row_matched": "C+",
  "column_matched": "Điểm thang 4",
  "top_table_candidates": ["page_9_table_1"]
}
```

## 4. Test đã thêm
- `tests/test_table_aware_retrieval_and_citations.py`

## 5. Hạn chế
Classifier hiện là rule-based, không document-specific. Nó ưu tiên an toàn hơn recall cao; câu hỏi bảng diễn đạt quá mơ hồ có thể không được boost.
