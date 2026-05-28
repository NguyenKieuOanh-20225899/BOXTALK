from __future__ import annotations

import json

from scripts.benchmark_table_qa import evaluate_prediction, load_queries, run_mock_variant, summarize, write_outputs


def test_table_qa_metrics_match_cell_citation() -> None:
    query = {
        "id": "q1",
        "question": "C+ tương ứng bao nhiêu điểm thang 4?",
        "gold_answer": "2.5",
        "gold_page": 9,
        "gold_table_id": "page_9_table_1",
        "gold_row_header": "C+",
        "gold_col_header": "Điểm thang 4",
        "query_type": "table_lookup",
    }
    prediction = {
        "answer": "2.5",
        "evidence": {"page": 9, "table_id": "page_9_table_1", "row_header": "C+", "col_header": "Điểm thang 4"},
        "citation": {"page": 9, "table_id": "page_9_table_1", "row_header": "C+", "col_header": "Điểm thang 4"},
        "retrieval_hit": True,
    }
    row = evaluate_prediction(query, prediction)
    assert row["table_answer_accuracy"] == 1.0
    assert row["table_evidence_match"] == 1.0
    assert row["cell_citation_accuracy"] == 1.0


def test_mock_benchmark_runs_with_two_samples(tmp_path) -> None:
    queries_path = tmp_path / "queries.jsonl"
    queries = [
        {"id": "q1", "question": "C+ tương ứng bao nhiêu điểm thang 4?", "gold_answer": "2.5", "gold_page": 9, "gold_table_id": "t1", "gold_row_header": "C+", "gold_col_header": "Điểm thang 4", "query_type": "table_lookup"},
        {"id": "q2", "question": "Khoảng điểm nào quy đổi ra B?", "gold_answer": "8.0-8.4", "gold_page": 9, "gold_table_id": "t1", "gold_row_header": "8.0-8.4", "gold_col_header": "Điểm chữ", "query_type": "table_reverse_lookup"},
    ]
    queries_path.write_text("\n".join(json.dumps(query, ensure_ascii=False) for query in queries), encoding="utf-8")
    loaded = load_queries(queries_path)
    rows = run_mock_variant(loaded, "hybrid_tatr_table_aware_retrieval_cell_citation")
    summary = summarize(rows)
    assert summary["query_count"] == 2
    assert summary["variants"]["hybrid_tatr_table_aware_retrieval_cell_citation"]["cell_citation_accuracy"] == 1.0
    out_dir = tmp_path / "out"
    write_outputs(rows, out_dir)
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "per_question.csv").exists()
