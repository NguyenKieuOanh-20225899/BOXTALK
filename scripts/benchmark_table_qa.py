from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


VARIANTS = (
    "default_extractor_normal_retrieval",
    "hybrid_tatr_normal_retrieval",
    "hybrid_tatr_table_aware_retrieval",
    "hybrid_tatr_table_aware_retrieval_cell_citation",
)


def load_queries(path: Path) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                queries.append(json.loads(line))
    return queries


def evaluate_prediction(query: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    answer = str(prediction.get("answer") or "").strip().casefold()
    gold_answer = str(query.get("gold_answer") or "").strip().casefold()
    evidence = dict(prediction.get("evidence") or {})
    citation = dict(prediction.get("citation") or {})
    table_evidence_match = (
        evidence.get("page") == query.get("gold_page")
        and evidence.get("table_id") == query.get("gold_table_id")
        and _match_optional(evidence.get("row_header"), query.get("gold_row_header"))
        and _match_optional(evidence.get("col_header"), query.get("gold_col_header"))
    )
    cell_citation_accuracy = table_evidence_match and (
        citation.get("row_header") == query.get("gold_row_header")
        and citation.get("col_header") == query.get("gold_col_header")
    )
    answer_correct = bool(gold_answer and gold_answer in answer)
    retrieval_hit = bool(prediction.get("retrieval_hit"))
    hallucinated = bool(prediction.get("hallucinated")) or (not answer_correct and bool(answer))
    return {
        "id": query.get("id"),
        "question": query.get("question"),
        "query_type": query.get("query_type"),
        "answer": prediction.get("answer"),
        "table_answer_accuracy": float(answer_correct),
        "table_evidence_match": float(table_evidence_match),
        "cell_citation_accuracy": float(cell_citation_accuracy),
        "table_retrieval_hit@k": float(retrieval_hit),
        "hallucination": float(hallucinated),
        "latency_ms": float(prediction.get("latency_ms") or 0.0),
    }


def run_mock_variant(queries: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query in queries:
        start = time.perf_counter()
        has_cell_citation = variant.endswith("cell_citation")
        has_table_retrieval = "table_aware_retrieval" in variant
        answer = query["gold_answer"] if has_table_retrieval else ""
        evidence = {
            "page": query.get("gold_page"),
            "table_id": query.get("gold_table_id"),
            "row_header": query.get("gold_row_header") if has_table_retrieval else None,
            "col_header": query.get("gold_col_header") if has_table_retrieval else None,
        }
        citation = dict(evidence) if has_cell_citation else {"page": query.get("gold_page"), "table_id": query.get("gold_table_id")}
        prediction = {
            "answer": answer,
            "evidence": evidence,
            "citation": citation,
            "retrieval_hit": has_table_retrieval,
            "hallucinated": False,
            "latency_ms": (time.perf_counter() - start) * 1000.0,
        }
        row = evaluate_prediction(query, prediction)
        row["variant"] = variant
        rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_variant.setdefault(str(row.get("variant")), []).append(row)
    variants: dict[str, dict[str, float]] = {}
    metric_names = (
        "table_answer_accuracy",
        "table_evidence_match",
        "cell_citation_accuracy",
        "table_retrieval_hit@k",
        "hallucination",
        "latency_ms",
    )
    for variant, variant_rows in by_variant.items():
        variants[variant] = {
            metric: _mean(float(row.get(metric) or 0.0) for row in variant_rows)
            for metric in metric_names
        }
    return {"query_count": len({row["id"] for row in rows}), "variants": variants}


def write_outputs(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summarize(rows), ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "per_question.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "per_question.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "variant",
            "id",
            "question",
            "query_type",
            "answer",
            "table_answer_accuracy",
            "table_evidence_match",
            "cell_citation_accuracy",
            "table_retrieval_hit@k",
            "hallucination",
            "latency_ms",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _match_optional(actual: Any, expected: Any) -> bool:
    if expected is None:
        return True
    return str(actual or "").strip().casefold() == str(expected).strip().casefold()


def _mean(values: Any) -> float:
    numbers = list(values)
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Vietnamese table QA with mock-safe table-aware variants.")
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", choices=[*VARIANTS, "all"], default="all")
    args = parser.parse_args()

    queries = load_queries(args.queries)
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    rows: list[dict[str, Any]] = []
    for variant in variants:
        rows.extend(run_mock_variant(queries, variant))
    write_outputs(rows, args.out)
    print(json.dumps(summarize(rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
