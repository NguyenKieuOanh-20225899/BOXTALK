from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import unicodedata
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.schemas import RetrievalConfig


CONFIGS = [
    ("Normal chunking", "normal"),
    ("Table-aware chunking", "table_aware_no_cell"),
    ("Table-aware chunking + cell-level evidence", "table_aware_cell"),
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = text.replace("−", "-").replace("–", "-").replace("≥", ">=").replace("≤", "<=")
    text = _strip_vietnamese_accents(text)
    text = re.sub(r"[:;,.]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _strip_vietnamese_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def same(expected: Any, actual: Any) -> bool:
    return normalize_text(expected) == normalize_text(actual)


def contains(expected: Any, actual: Any) -> bool:
    expected_norm = normalize_text(expected)
    actual_norm = normalize_text(actual)
    return bool(expected_norm) and expected_norm in actual_norm


def is_table_chunk(row: dict[str, Any]) -> bool:
    meta = row.get("metadata") or {}
    return bool(meta.get("is_table_chunk") or meta.get("table_id") or row.get("block_type") == "table")


def chunking_strategy(row: dict[str, Any]) -> str:
    return str((row.get("metadata") or {}).get("chunking_strategy") or "")


def make_corpus_variant(rows: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        strategy = chunking_strategy(row)
        table_chunk = is_table_chunk(row)

        if variant == "normal":
            if table_chunk:
                if strategy != "table_structure":
                    continue
                item = deepcopy(row)
                meta = dict(item.get("metadata") or {})
                for key in ["row_index", "col_index", "row_header", "col_header", "cell_text"]:
                    meta.pop(key, None)
                meta["chunking_strategy"] = "normal_table_flat"
                meta["citation_target"] = "table"
                item["metadata"] = meta
                output.append(item)
            else:
                output.append(deepcopy(row))
            continue

        if variant == "table_aware_no_cell":
            if strategy == "table_cell":
                continue
            output.append(deepcopy(row))
            continue

        if variant == "table_aware_cell":
            output.append(deepcopy(row))
            continue

        raise ValueError(f"Unknown corpus variant: {variant}")
    return output


def evaluate_hit_set(query: dict[str, Any], hit_dicts: list[dict[str, Any]]) -> dict[str, bool]:
    gold_table_id = query.get("gold_table_id")
    gold_row = query.get("gold_row_header")
    gold_col = query.get("gold_col_header")
    gold_answer = query.get("gold_answer")

    table_hit = False
    row_match = False
    column_match = False
    cell_match = False

    for hit in hit_dicts:
        meta = hit.get("metadata") or {}
        text = hit.get("text") or ""
        same_table = str(meta.get("table_id") or "") == str(gold_table_id)
        if not same_table:
            continue

        table_hit = True

        if same(gold_row, meta.get("row_header")):
            row_match = True

        if same(gold_col, meta.get("col_header")) or contains(gold_col, text):
            column_match = True

        if (
            str(meta.get("citation_target") or "") == "cell"
            and same(gold_row, meta.get("row_header"))
            and same(gold_col, meta.get("col_header"))
            and same(gold_answer, meta.get("cell_text"))
        ):
            cell_match = True

    return {
        "table_hit": table_hit,
        "row_match": row_match,
        "column_match": column_match,
        "cell_match": cell_match,
    }


def run_config(
    label: str,
    variant: str,
    source_rows: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    top_k: int,
    out_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    corpus_rows = make_corpus_variant(source_rows, variant)
    corpus_path = out_dir / "corpora" / f"{variant}.jsonl"
    write_jsonl(corpus_path, corpus_rows)

    retriever = HybridRetriever(corpus_rows, build_dense=False, build_colbert=False)
    config = RetrievalConfig(top_k=top_k, candidate_k=max(50, top_k))

    per_query: list[dict[str, Any]] = []
    latencies: list[float] = []
    for query in queries:
        result = retriever.search_result(str(query["question"]), strategy="bm25", config=config)
        hit_dicts = [hit.to_dict() for hit in result.hits]
        metrics = evaluate_hit_set(query, hit_dicts)
        latencies.append(result.latency_ms)
        per_query.append(
            {
                "config": label,
                "variant": variant,
                "query_id": query.get("id"),
                "question": query.get("question"),
                "gold_answer": query.get("gold_answer"),
                "gold_table_id": query.get("gold_table_id"),
                "gold_row_header": query.get("gold_row_header"),
                "gold_col_header": query.get("gold_col_header"),
                **metrics,
                "latency_ms": result.latency_ms,
                "top_hits": [
                    {
                        "rank": hit.get("rank"),
                        "chunk_id": hit.get("chunk_id"),
                        "score": hit.get("score"),
                        "text": hit.get("text"),
                        "metadata": {
                            "table_id": (hit.get("metadata") or {}).get("table_id"),
                            "chunking_strategy": (hit.get("metadata") or {}).get("chunking_strategy"),
                            "citation_target": (hit.get("metadata") or {}).get("citation_target"),
                            "row_header": (hit.get("metadata") or {}).get("row_header"),
                            "col_header": (hit.get("metadata") or {}).get("col_header"),
                            "cell_text": (hit.get("metadata") or {}).get("cell_text"),
                        },
                    }
                    for hit in hit_dicts
                ],
            }
        )

    table_hit = mean(float(row["table_hit"]) for row in per_query)
    row_match = mean(float(row["row_match"]) for row in per_query)
    column_match = mean(float(row["column_match"]) for row in per_query)
    cell_match = mean(float(row["cell_match"]) for row in per_query)
    summary = {
        "config": label,
        "variant": variant,
        "query_count": len(queries),
        "corpus_chunk_count": len(corpus_rows),
        "top_k": top_k,
        "strategy": "bm25",
        "table_hit": table_hit,
        "row_match": row_match,
        "column_match": column_match,
        "cell_match": cell_match,
        f"table_hit_at_{top_k}": table_hit,
        f"row_match_at_{top_k}": row_match,
        f"column_match_at_{top_k}": column_match,
        f"cell_match_at_{top_k}": cell_match,
        "latency_ms_mean": mean(latencies) if latencies else 0.0,
        "derived_corpus": str(corpus_path),
    }
    return summary, per_query


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "config",
        "variant",
        "query_id",
        "question",
        "gold_answer",
        "gold_table_id",
        "gold_row_header",
        "gold_col_header",
        "table_hit",
        "row_match",
        "column_match",
        "cell_match",
        "latency_ms",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_latex(path: Path, summaries: list[dict[str, Any]]) -> None:
    top_k = summaries[0].get("top_k", 5) if summaries else 5
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Kết quả truy xuất bảng trên hệ thống thật với tập Table QA tiếng Việt}",
        r"\label{tab:real-table-qa-retrieval-result}",
        r"\begin{tabular}{|p{4.2cm}|c|c|c|c|}",
        r"\hline",
        rf"\textbf{{Cấu hình}} & \textbf{{Table Hit@{top_k}}} & \textbf{{Row Match@{top_k}}} & \textbf{{Column Match@{top_k}}} & \textbf{{Cell Match@{top_k}}} \\",
        r"\hline",
    ]
    for summary in summaries:
        lines.append(
            f"{summary['config']} & "
            f"{summary['table_hit']:.3f} & "
            f"{summary['row_match']:.3f} & "
            f"{summary['column_match']:.3f} & "
            f"{summary['cell_match']:.3f} \\\\"
        )
        lines.append(r"\hline")
    lines.extend([r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_readme(path: Path, args: argparse.Namespace, summaries: list[dict[str, Any]]) -> None:
    rows = "\n".join(
        f"| {s['config']} | {s['table_hit']:.3f} | {s['row_match']:.3f} | "
        f"{s['column_match']:.3f} | {s['cell_match']:.3f} |"
        for s in summaries
    )
    content = f"""# Real Table QA Retrieval Benchmark

Benchmark này chạy retrieval thật trên corpus QCDT, không dùng LLM và không dùng mock answer.
Mục tiêu là so sánh tác động của biểu diễn bảng trong retrieval index.

## Lệnh chạy lại

```powershell
.\\.venv-gpu\\Scripts\\python.exe scripts\\benchmark_real_table_qa_retrieval.py `
  --source-corpus {args.source_corpus} `
  --queries {args.queries} `
  --out {args.out} `
  --top-k {args.top_k}
```

## Kết quả

| Cấu hình | Table Hit@{args.top_k} | Row Match@{args.top_k} | Column Match@{args.top_k} | Cell Match@{args.top_k} |
|---|---:|---:|---:|---:|
{rows}

## Ghi chú

- `Normal chunking`: giữ bảng ở dạng phẳng/Markdown table, không có row/cell evidence.
- `Table-aware chunking`: giữ table summary, table structure và table row, bỏ table cell để tách riêng tác động của cell-level evidence.
- `Table-aware chunking + cell-level evidence`: giữ đầy đủ table summary, structure, row và cell chunks.
- Các corpus biến thể được sinh từ cùng corpus nguồn để cô lập ảnh hưởng của chunking/evidence representation.
"""
    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark real Table QA retrieval over table-aware corpora.")
    parser.add_argument("--source-corpus", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    source_rows = load_jsonl(args.source_corpus)
    queries = load_jsonl(args.queries)

    summaries: list[dict[str, Any]] = []
    all_per_query: list[dict[str, Any]] = []
    for label, variant in CONFIGS:
        summary, per_query = run_config(label, variant, source_rows, queries, args.top_k, out_dir)
        summaries.append(summary)
        all_per_query.extend(per_query)

    (out_dir / "summary.json").write_text(
        json.dumps({"summaries": summaries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "per_query.json").write_text(
        json.dumps(all_per_query, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(out_dir / "per_query.csv", all_per_query)
    write_latex(out_dir / "latex_table.tex", summaries)
    write_readme(out_dir / "README.md", args, summaries)

    print(json.dumps({"summaries": summaries}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
