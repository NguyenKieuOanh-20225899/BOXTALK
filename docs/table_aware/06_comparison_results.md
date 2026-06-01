# Phase 6 - Comparison Results

## 1. Bảng so sánh extraction
| Variant | detection F1 | cell IoU F1 | table_structure F1 | text assignment F1 | exact CSV/HTML |
|---|---:|---:|---:|---:|---:|
| default extractor | n/a | n/a | n/a | n/a | n/a |
| pretrained TATR | n/a | n/a | n/a | n/a | n/a |
| hybrid_tatr OCR/PDF word boxes | n/a | n/a | n/a | n/a | n/a |

PubTables subset chưa có trong workspace, nên các extraction metrics chưa được chạy.

## 2. Bảng so sánh table QA
| Variant | answer_accuracy | evidence_match | cell_citation_accuracy | retrieval_hit@k | hallucination |
|---|---:|---:|---:|---:|---:|
| default extractor + normal retrieval | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| hybrid_tatr + normal retrieval | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| hybrid_tatr + table-aware retrieval | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| hybrid_tatr + table-aware retrieval + cell-level citation | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

Nguồn: `results/table_qa_vi/final_safe/summary.json`.

## 3. Nhận xét
- Hybrid TATR giúp biểu diễn bảng thành cells có row/column/text metadata.
- Table-aware retrieval giúp ưu tiên table/row/cell chunks khi query có dạng lookup.
- Cell-level citation giúp kiểm chứng câu trả lời bảng chi tiết hơn.

## 4. Case fail
Cần theo dõi merged cells, multi-row headers, scan/OCR lỗi, và reverse lookup mơ hồ.

## 5. Có nên bật mặc định không
Chưa. Các flag table-aware đang tắt mặc định cho đến khi benchmark thật đủ ổn định.
