# Table-aware PDF QA với Hybrid TATR

## Mục tiêu

Triển khai mức an toàn để biến Hybrid TATR thành điểm nổi bật của đồ án: PDF QA nhận biết bảng, index bảng như evidence có cấu trúc và trả citation tới trang/bảng/hàng/cột khi câu trả lời đến từ một cell.

Không train model trong giai đoạn này, không rewrite pipeline chính, không claim SOTA và không bật `hybrid_tatr` làm default khi chưa có benchmark thật chứng minh ổn định.

## Pipeline tổng quan

Trước cải tiến:

```text
PDF ingest -> region-level routing -> conditional hybrid_tatr table enhancement
-> chunk/index -> retrieval -> routed_grounded -> grounded answer + citation
```

Sau cải tiến:

```text
PDF table region -> Hybrid TATR/default extractor -> TableCell objects
-> table-aware chunks -> table-aware retrieval -> grounded answer
-> cell-level citation
```

Các flag an toàn:

- `BOXBIIBOO_ENABLE_TABLE_AWARE_CHUNKING=true|false`
- `BOXBIIBOO_ENABLE_TABLE_AWARE_RETRIEVAL=true|false`

## Phase docs

| Phase | File docs | Mục tiêu | Trạng thái |
|---|---|---|---|
| 0 | [00_plan.md](00_plan.md) | Plan tổng quan, rủi ro và tiêu chí thành công | Done |
| 1 | [01_hybrid_tatr_output.md](01_hybrid_tatr_output.md) | Chuẩn hóa `TableCell`/`TableBlock` và adapter backward-compatible | Done |
| 2 | [02_table_aware_chunking.md](02_table_aware_chunking.md) | Sinh summary, structure, row/cell chunks cho bảng | Done |
| 3 | [03_table_aware_retrieval.md](03_table_aware_retrieval.md) | Query type và boosting cho table lookup | Done |
| 4 | [04_cell_level_citation.md](04_cell_level_citation.md) | Citation tới page/table/row/cell | Done |
| 5 | [05_table_qa_benchmark.md](05_table_qa_benchmark.md) | Dataset nhỏ và benchmark table QA tiếng Việt | Done |
| 6 | [06_comparison_results.md](06_comparison_results.md) | So sánh mock-safe giữa các biến thể pipeline | Done |
| 7 | [07_error_analysis.md](07_error_analysis.md) | Phân tích lỗi thường gặp của table extraction/QA/citation | Done |
| 8 | [08_final_validation.md](08_final_validation.md) | Commands validation và kết quả regression | Done |
| 9a | [09_constraint_aware_table_reconstruction_plan.md](09_constraint_aware_table_reconstruction_plan.md) | Plan Hybrid TATR + constraint-aware reconstruction | Done |
| 9b | [09_constraint_aware_table_reconstruction_results.md](09_constraint_aware_table_reconstruction_results.md) | Before/after, records, trace và validation cho reconstruction | Done |

## Dataset và results

- Dataset benchmark: [data/benchmarks/table_qa_vi/queries.jsonl](../../data/benchmarks/table_qa_vi/queries.jsonl)
- Table QA results: [results/table_qa_vi/](../../results/table_qa_vi/)
- Comparison results: [results/table_aware_comparison/](../../results/table_aware_comparison/)
- Final table QA summary: [results/table_qa_vi/final_safe/summary.json](../../results/table_qa_vi/final_safe/summary.json)
- Mock ingest benchmark: [results/ingest/mock_after_table_aware_safe/summary.json](../../results/ingest/mock_after_table_aware_safe/summary.json)

## Files changed quan trọng

| File | Vai trò |
|---|---|
| `app/ingest/extract/table.py` | Mở rộng `TableCell`, sinh CSV/Markdown từ cells, thêm trace/citation metadata |
| `app/ingest/schemas.py` | Thêm schema-facing `TableCell` và `TableBlock` |
| `app/ingest/table_chunking.py` | Table-aware chunking mới |
| `app/ingest/table_reconstruct.py` | Constraint-aware table reconstruction sau flag |
| `app/ingest/chunker.py` | Tích hợp table-aware chunking sau flag |
| `app/retrieval/table_aware.py` | Query classifier và table-aware scoring |
| `app/retrieval/reranker.py` | Tích hợp boosting + retrieval trace sau flag |
| `app/qa/citations.py` | Formatter citation page/table/row/cell |
| `app/qa/answer_generator.py` | Dùng formatter citation mới, giữ paragraph citation cũ |
| `scripts/benchmark_table_qa.py` | Benchmark table QA tiếng Việt mock-safe |
| `tests/test_table_aware_ingest.py` | Tests schema/chunking |
| `tests/test_table_aware_retrieval_and_citations.py` | Tests retrieval/citation |
| `tests/test_benchmark_table_qa.py` | Tests benchmark metrics/output |

## Kết luận

Đã hoàn thành mức an toàn cho table-aware QA:

- Hybrid TATR/default table output có thể được chuẩn hóa thành `TableCell` với row/column/text metadata.
- Table-aware chunking giúp bảng được index như evidence có cấu trúc.
- Table-aware retrieval ưu tiên table/cell evidence khi query là table lookup.
- Cell-level citation giúp câu trả lời bảng kiểm chứng được tới trang, bảng, hàng và cột.

Metric chính trên benchmark mock-safe 8 câu:

| Variant | Answer accuracy | Evidence match | Cell citation accuracy | Hit@k |
|---|---:|---:|---:|---:|
| default extractor + normal retrieval | 0.000 | 0.000 | 0.000 | 0.000 |
| hybrid_tatr + normal retrieval | 0.000 | 0.000 | 0.000 | 0.000 |
| hybrid_tatr + table-aware retrieval | 1.000 | 1.000 | 0.000 | 1.000 |
| hybrid_tatr + table-aware retrieval + cell-level citation | 1.000 | 1.000 | 1.000 | 1.000 |

Hạn chế:

- Benchmark hiện là mock-safe/sample nhỏ, chưa đủ để claim độ ổn định trên PDF thật.
- PubTables subset không có trong workspace nên chưa chạy extraction comparison thật.
- Multi-row header, merged cell, bảng scan/OCR lỗi và caption/footnote lẫn bảng vẫn là nhóm rủi ro chính.
- QA smoke/QCDT/Operations route chưa có command riêng rõ ràng để chạy lại ngoài bộ pytest hiện có.

Khuyến nghị báo cáo:

- Có thể đưa vào báo cáo như một hướng cải tiến thực nghiệm có kiểm soát.
- Chưa nên bật `hybrid_tatr` hoặc table-aware retrieval làm default trên production path.
- Nên giữ flag off mặc định, bật theo benchmark hoặc theo route tài liệu có nhiều bảng.
