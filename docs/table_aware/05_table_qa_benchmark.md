# Phase 5 - Vietnamese Table QA Benchmark

## 1. Dataset format
Dataset ở `data/benchmarks/table_qa_vi/queries.jsonl`, mỗi dòng gồm `id`, `question`, `gold_answer`, `gold_page`, `gold_table_id`, `gold_row_header`, `gold_col_header`, `query_type`.

## 2. Số lượng câu hỏi
Hiện có 8 sample an toàn để smoke benchmark. Cần bổ sung lên 20-50 câu thật trước khi dùng như benchmark chính.

## 3. Nhóm câu hỏi
- Direct cell lookup.
- Reverse lookup.
- Range lookup.
- Row lookup.
- Column lookup.

## 4. Metrics
- `table_answer_accuracy`
- `table_evidence_match`
- `cell_citation_accuracy`
- `table_retrieval_hit@k`
- `hallucination_rate`
- `latency`

## 5. Command
```powershell
python scripts/benchmark_table_qa.py --queries data/benchmarks/table_qa_vi/queries.jsonl --out results/table_qa_vi/final_safe
```

## 6. Kết quả
Đã chạy mock-safe benchmark tại `results/table_qa_vi/final_safe/summary.json`.

| Variant | Answer accuracy | Evidence match | Cell citation | Retrieval hit@k | Hallucination |
|---|---:|---:|---:|---:|---:|
| default extractor + normal retrieval | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| hybrid_tatr + normal retrieval | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| hybrid_tatr + table-aware retrieval | 1.000 | 1.000 | 0.000 | 1.000 | 0.000 |
| hybrid_tatr + table-aware retrieval + cell-level citation | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

## 7. Hạn chế
Script hiện chạy mock-safe variants để kiểm tra metric/output contract. Chưa thay thế benchmark end-to-end trên index thật.
