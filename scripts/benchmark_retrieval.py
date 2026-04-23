from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import make_reranker
from app.retrieval.schemas import RetrievedHit, RetrievalConfig


RESULTS_ROOT = Path("results/retrieval_benchmark")
DEFAULT_STRATEGIES = ["bm25", "dense", "hybrid"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark retrieval index against labeled QA queries.")
    parser.add_argument("--index-dir", type=Path, required=True, help="Retrieval index directory.")
    parser.add_argument("--queries", type=Path, required=True, help="JSON/JSONL benchmark queries.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory.")
    parser.add_argument("--top-k", type=int, default=5, help="Evaluate up to this cutoff.")
    parser.add_argument(
        "--strategy",
        action="append",
        default=[],
        choices=["bm25", "dense", "colbert", "hybrid", "hybrid_rerank", "all"],
        help="Strategy to benchmark. Pass multiple times; default: bm25+dense+hybrid.",
    )
    parser.add_argument("--combination", choices=["weighted_sum", "rrf"], default="weighted_sum")
    parser.add_argument("--bm25-weight", type=float, default=0.5)
    parser.add_argument("--dense-weight", type=float, default=0.5)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--rerank-top-n", type=int, default=20)
    parser.add_argument("--reranker", choices=["none", "heuristic", "cross-encoder", "colbert"], default="heuristic")
    parser.add_argument("--reranker-model", default=None, help="Optional cross-encoder model name.")
    parser.add_argument("--no-warmup", action="store_true", help="Disable one unmeasured warmup query per strategy.")
    parser.add_argument("--block-type", action="append", default=[], help="Optional block_type filter.")
    parser.add_argument("--section", action="append", default=[], help="Optional section substring filter.")
    parser.add_argument("--doc-id", action="append", default=[], help="Optional doc_id filter.")
    parser.add_argument(
        "--metadata-filter",
        action="append",
        default=[],
        help="Optional metadata filter in key=value form. Pass multiple times.",
    )
    return parser.parse_args()


def load_benchmark_cases(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("queries"), list):
        return payload["queries"]
    raise ValueError(f"{path} must be a JSONL file, a JSON list, or an object with a queries list")


def expand_strategies(values: list[str]) -> list[str]:
    if not values:
        return DEFAULT_STRATEGIES
    expanded: list[str] = []
    for value in values:
        if value == "all":
            expanded.extend(["bm25", "dense", "colbert", "hybrid", "hybrid_rerank"])
        else:
            expanded.append(value)
    return list(dict.fromkeys(expanded))


def parse_metadata_filters(values: list[str]) -> dict[str, str]:
    filters: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --metadata-filter {value!r}; expected key=value")
        key, raw = value.split("=", 1)
        filters[key.strip()] = raw.strip()
    return filters


def build_config(args: argparse.Namespace, *, strategy: str) -> RetrievalConfig:
    return RetrievalConfig(
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        bm25_weight=args.bm25_weight,
        dense_weight=args.dense_weight,
        rerank_top_n=args.rerank_top_n if strategy == "hybrid_rerank" else 0,
        combination=args.combination,
        use_rerank=strategy == "hybrid_rerank",
        block_type_filter=list(args.block_type),
        section_filter=list(args.section),
        doc_id_filter=list(args.doc_id),
        metadata_filters=parse_metadata_filters(args.metadata_filter),
    )


def question_text(case: dict[str, Any]) -> str:
    return str(case.get("question") or case.get("query") or "")


def expected_chunk_ids(case: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    single = case.get("expected_chunk_id")
    if single:
        values.add(str(single))
    many = case.get("expected_chunk_ids") or []
    if isinstance(many, str):
        values.add(many)
    else:
        values.update(str(item) for item in many)
    return values


def expected_pages(case: dict[str, Any]) -> set[int]:
    values: set[int] = set()
    single = case.get("expected_page")
    if single is not None:
        values.add(int(single))
    many = case.get("expected_pages") or []
    values.update(int(item) for item in many)
    return values


def is_match(case: dict[str, Any], hit: RetrievedHit) -> bool:
    gold_chunk_ids = expected_chunk_ids(case)
    if gold_chunk_ids:
        return hit.chunk_id in gold_chunk_ids

    checks: list[bool] = []
    expected_source = case.get("source_name")
    if expected_source:
        checks.append(str(hit.chunk.source_name or "") == str(expected_source))

    expected_doc_id = case.get("doc_id")
    if expected_doc_id:
        checks.append(str(hit.chunk.doc_id or "") == str(expected_doc_id))

    pages = expected_pages(case)
    if pages:
        checks.append(hit.page in pages)

    expected_section = str(case.get("expected_section") or case.get("section") or "").strip().lower()
    if expected_section:
        checks.append(expected_section in str(hit.section or "").strip().lower())

    match_text = str(case.get("match_text") or "").strip().lower()
    if match_text:
        checks.append(match_text in str(hit.text or "").lower())

    return all(checks) if checks else False


def matched_gold_count(case: dict[str, Any], hits: list[RetrievedHit]) -> int:
    gold_chunk_ids = expected_chunk_ids(case)
    if gold_chunk_ids:
        return len({hit.chunk_id for hit in hits if hit.chunk_id in gold_chunk_ids})
    return int(any(is_match(case, hit) for hit in hits))


def relevant_count(case: dict[str, Any]) -> int:
    return max(1, len(expected_chunk_ids(case)))


def reciprocal_rank(matches: list[bool]) -> float:
    for rank, matched in enumerate(matches, start=1):
        if matched:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(matches: list[bool], gold_count: int, k: int) -> float:
    dcg = 0.0
    for rank, matched in enumerate(matches[:k], start=1):
        if matched:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(gold_count, k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0.0 else 0.0


def evaluate_case(
    *,
    strategy: str,
    case: dict[str, Any],
    hits: list[RetrievedHit],
    latency_ms: float,
    top_k: int,
) -> dict[str, Any]:
    scoped_hits = hits[:top_k]
    matches = [is_match(case, hit) for hit in scoped_hits]
    first_match_rank = next((idx for idx, matched in enumerate(matches, start=1) if matched), None)
    gold_count = relevant_count(case)
    recall = matched_gold_count(case, scoped_hits) / gold_count
    top_hit = scoped_hits[0] if scoped_hits else None
    return {
        "strategy": strategy,
        "query_id": case.get("query_id") or case.get("id"),
        "question": question_text(case),
        "hit_at_k": bool(any(matches)),
        "recall_at_k": recall,
        "mrr_at_k": reciprocal_rank(matches),
        "ndcg_at_k": ndcg_at_k(matches, gold_count, top_k),
        "first_match_rank": first_match_rank,
        "latency_ms": latency_ms,
        "retrieval_count": len(scoped_hits),
        "top_hit_chunk_id": top_hit.chunk_id if top_hit else None,
        "top_hit_score": round(float(top_hit.final_score or top_hit.score), 6) if top_hit else None,
        "top_hit_page": top_hit.page if top_hit else None,
        "top_hit_section": top_hit.section if top_hit else None,
        "top_hit_text": top_hit.snippet if top_hit else None,
    }


def summarize_rows(rows: list[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    if not rows:
        return {
            "query_count": 0,
            f"hit_at_{top_k}": 0.0,
            f"recall_at_{top_k}": 0.0,
            f"mrr_at_{top_k}": 0.0,
            f"ndcg_at_{top_k}": 0.0,
            "avg_latency_ms": 0.0,
            "avg_retrieval_count": 0.0,
        }

    return {
        "query_count": len(rows),
        f"hit_at_{top_k}": statistics.mean(float(row["hit_at_k"]) for row in rows),
        f"recall_at_{top_k}": statistics.mean(float(row["recall_at_k"]) for row in rows),
        f"mrr_at_{top_k}": statistics.mean(float(row["mrr_at_k"]) for row in rows),
        f"ndcg_at_{top_k}": statistics.mean(float(row["ndcg_at_k"]) for row in rows),
        "avg_latency_ms": statistics.mean(float(row["latency_ms"]) for row in rows),
        "avg_retrieval_count": statistics.mean(float(row["retrieval_count"]) for row in rows),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary: dict[str, Any], *, top_k: int) -> None:
    lines = [
        "# Retrieval Benchmark",
        "",
        f"- Index: `{summary['index_dir']}`",
        f"- Queries: `{summary['queries_file']}`",
        f"- Top-k: {top_k}",
        "",
        "| Strategy | Queries | Hit@k | Recall@k | MRR@k | nDCG@k | Avg latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for strategy, metrics in summary["strategies"].items():
        lines.append(
            "| {strategy} | {query_count} | {hit:.3f} | {recall:.3f} | {mrr:.3f} | {ndcg:.3f} | {latency:.2f} |".format(
                strategy=strategy,
                query_count=metrics["query_count"],
                hit=metrics[f"hit_at_{top_k}"],
                recall=metrics[f"recall_at_{top_k}"],
                mrr=metrics[f"mrr_at_{top_k}"],
                ndcg=metrics[f"ndcg_at_{top_k}"],
                latency=metrics["avg_latency_ms"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def warmup_retriever(
    retriever: HybridRetriever,
    *,
    strategies: list[str],
    cases: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    if args.no_warmup or not cases:
        return
    warmup_query = question_text(cases[0])
    for strategy in strategies:
        config = build_config(args, strategy=strategy)
        retriever.search_result(warmup_query, strategy=strategy, config=config)


def main() -> None:
    args = parse_args()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (RESULTS_ROOT / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    strategies = expand_strategies(args.strategy)
    reranker = make_reranker(args.reranker, model_name=args.reranker_model)
    retriever = HybridRetriever.load(args.index_dir, reranker=reranker)
    if "colbert" in strategies and not (retriever.colbert and retriever.colbert.doc_embeddings):
        if "colbert" in args.strategy:
            raise RuntimeError("Strategy 'colbert' requires an index built with --build-colbert.")
        strategies = [strategy for strategy in strategies if strategy != "colbert"]
    cases = load_benchmark_cases(args.queries)
    warmup_retriever(retriever, strategies=strategies, cases=cases, args=args)

    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}

    for strategy in strategies:
        strategy_rows: list[dict[str, Any]] = []
        for case in cases:
            query = question_text(case)
            config = build_config(args, strategy=strategy)
            result = retriever.search_result(query, strategy=strategy, config=config)
            row = evaluate_case(
                strategy=strategy,
                case=case,
                hits=result.hits,
                latency_ms=result.latency_ms,
                top_k=args.top_k,
            )
            strategy_rows.append(row)
            all_rows.append(row)
        summaries[strategy] = summarize_rows(strategy_rows, top_k=args.top_k)

    summary = {
        "timestamp_utc": timestamp,
        "index_dir": str(args.index_dir),
        "queries_file": str(args.queries),
        "top_k": args.top_k,
        "strategies": summaries,
    }
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "per_question.json").write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(output_dir / "per_question.csv", all_rows)
    write_markdown(output_dir / "README.md", summary, top_k=args.top_k)

    print(output_dir)


if __name__ == "__main__":
    main()
