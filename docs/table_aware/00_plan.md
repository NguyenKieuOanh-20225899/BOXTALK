# Phase 0 - Plan

## 1. Mục tiêu
Xây dựng lớp an toàn cho table-aware PDF QA: bảng PDF được biểu diễn bằng `TableCell`, được chunk/index như evidence có cấu trúc, retrieval ưu tiên đúng hàng/cột khi câu hỏi là table lookup, và câu trả lời có citation tới trang/bảng/hàng/cột.

## 2. Vì sao bảng khác text thường
Bảng mang nghĩa theo giao điểm hàng/cột. Nếu index bảng như paragraph, hệ thống dễ mất header, ghép sai ô, hoặc citation chỉ trỏ về trang mà không chỉ ra giá trị nằm ở hàng/cột nào.

## 3. Pipeline trước cải tiến
PDF ingest -> region-level routing -> conditional hybrid_tatr table enhancement -> chunk/index -> retrieval -> routed_grounded -> grounded answer + citation.

## 4. Pipeline sau cải tiến
PDF table region -> Hybrid TATR/default extractor -> `TableCell` objects -> table-aware chunks -> table-aware retrieval -> grounded answer -> cell-level citation.

## 5. Các phase
0. Plan và README tổng hợp.
1. Chuẩn hóa output Hybrid TATR thành cell schema.
2. Table-aware chunking.
3. Table-aware retrieval.
4. Cell-level citation.
5. Benchmark nhỏ cho bảng tiếng Việt.
6. So sánh default/TATR/hybrid/table-aware.
7. Error analysis.
8. Final validation.
9. README tổng hợp cuối.

## 6. Rủi ro regression
- Làm đổi metadata bảng cũ.
- Chunk bảng mới làm nhiễu câu hỏi text thường.
- Retrieval boost đẩy table chunk không liên quan lên đầu.
- Citation thiếu fallback khi bảng không có caption hoặc header.
- Hybrid TATR bị hiểu nhầm là default ổn định dù chưa benchmark đủ.

## 7. Tiêu chí thành công
- Backward compatible với table chunk cũ khi flag tắt.
- Có `TableCell` row/col/text/header metadata.
- Table-aware chunking/retrieval có flag riêng.
- Citation bảng có page/table/row/column khi metadata đủ.
- `compileall`, `pytest`, mock benchmark chạy được.
- Không merge vào `main`; nhánh làm việc là `TART-UP`.
