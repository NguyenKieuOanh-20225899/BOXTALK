from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import make_reranker
from app.retrieval.schemas import DocumentChunkRef, RetrievedHit, RetrievalConfig


RESULTS_ROOT = Path("results/beir_retrieval_benchmark")
BEIR_DATASET_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real BEIR-style retrieval benchmarks.")
    parser.add_argument("--dataset", default="scifact", help="BEIR dataset name, e.g. scifact, nfcorpus, fiqa.")
    parser.add_argument("--split", default="test", help="BEIR split.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/beir"), help="BEIR download/cache directory.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional benchmark output directory.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--query-limit", type=int, default=0, help="Limit queries for local smoke runs.")
    parser.add_argument("--corpus-limit", type=int, default=0, help="Limit corpus docs while preserving gold docs.")
    parser.add_argument(
        "--strategy",
        action="append",
        default=[],
        choices=["bm25", "dense", "colbert", "hybrid", "hybrid_rerank", "all"],
    )
    parser.add_argument(
        "--dense-preset",
        choices=["minilm", "multilingual-minilm", "contriever", "dpr-single-nq", "dpr-multiset"],
        default="minilm",
    )
    parser.add_argument("--dense-model", default=None)
    parser.add_argument("--dense-backend", choices=["sentence-transformers", "transformers", "dpr"], default=None)
    parser.add_argument("--dense-query-model", default=None)
    parser.add_argument("--dense-passage-model", default=None)
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--build-colbert", action="store_true")
    parser.add_argument("--colbert-model", default="colbert-ir/colbertv2.0")
    parser.add_argument("--reranker", choices=["none", "heuristic", "cross-encoder", "colbert"], default="heuristic")
    parser.add_argument("--reranker-model", default=None)
    parser.add_argument("--combination", choices=["weighted_sum", "rrf"], default="weighted_sum")
    parser.add_argument("--bm25-weight", type=float, default=0.5)
    parser.add_argument("--dense-weight", type=float, default=0.5)
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--rerank-top-n", type=int, default=50)
    return parser.parse_args()


def load_beir_dataset(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, str], dict[str, dict[str, int]]]:
    try:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader
    except Exception as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("Install beir>=2.0.0 to run scripts/benchmark_beir_retrieval.py") from exc

    args.data_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.data_dir / args.dataset
    if not dataset_path.exists():
        url = BEIR_DATASET_URL.format(dataset=args.dataset)
        dataset_path = Path(util.download_and_unzip(url, str(args.data_dir)))

    corpus, queries, qrels = GenericDataLoader(data_folder=str(dataset_path)).load(split=args.split)
    return corpus, queries, qrels


def limit_dataset(
    corpus: dict[str, Any],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    *,
    query_limit: int,
    corpus_limit: int,
) -> tuple[dict[str, Any], dict[str, str], dict[str, dict[str, int]]]:
    if query_limit > 0:
        query_ids = list(queries)[:query_limit]
        queries = {qid: queries[qid] for qid in query_ids}
        qrels = {qid: qrels.get(qid, {}) for qid in query_ids}

    if corpus_limit > 0:
        gold_doc_ids = {
            doc_id
            for per_query in qrels.values()
            for doc_id, relevance in per_query.items()
            if relevance > 0
        }
        selected_ids = list(dict.fromkeys([*gold_doc_ids, *list(corpus)[:corpus_limit]]))
        corpus = {doc_id: corpus[doc_id] for doc_id in selected_ids if doc_id in corpus}

    return corpus, queries, qrels


def corpus_to_chunks(dataset: str, corpus: dict[str, Any]) -> list[DocumentChunkRef]:
    chunks: list[DocumentChunkRef] = []
    for doc_id, doc in corpus.items():
        title = str(doc.get("title") or "").strip()
        text = str(doc.get("text") or "").strip()
        chunks.append(
            DocumentChunkRef(
                chunk_id=str(doc_id),
                doc_id=str(doc_id),
                source_name=dataset,
                title=title or None,
                section=title or None,
                text="\n".join(part for part in [title, text] if part),
                block_type="document",
                metadata={"beir_dataset": dataset},
            )
        )
    return chunks


def expand_strategies(values: list[str]) -> list[str]:
    if not values:
        return ["bm25", "dense", "hybrid"]
    expanded: list[str] = []
    for value in values:
        if value == "all":
            expanded.extend(["bm25", "dense", "colbert", "hybrid", "hybrid_rerank"])
        else:
            expanded.append(value)
    return list(dict.fromkeys(expanded))


def build_config(args: argparse.Namespace, *, strategy: str) -> RetrievalConfig:
    return RetrievalConfig(
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        bm25_weight=args.bm25_weight,
        dense_weight=args.dense_weight,
        rerank_top_n=args.rerank_top_n if strategy == "hybrid_rerank" else 0,
        combination=args.combination,
        use_rerank=strategy == "hybrid_rerank",
    )


def evaluate_hits(qid: str, hits: list[RetrievedHit], qrels: dict[str, dict[str, int]], *, k: int) -> dict[str, Any]:
    relevant = {doc_id for doc_id, score in qrels.get(qid, {}).items() if score > 0}
    retrieved = hits[:k]
    matches = [hit.chunk_id in relevant for hit in retrieved]
    hit_count = len({hit.chunk_id for hit in retrieved if hit.chunk_id in relevant})
    relevant_count = max(1, len(relevant))
    return {
        "hit_at_k": bool(any(matches)),
        "recall_at_k": hit_count / relevant_count,
        "mrr_at_k": reciprocal_rank(matches),
        "ndcg_at_k": ndcg_at_k(matches, min(relevant_count, k)),
        "top_hit_doc_id": retrieved[0].chunk_id if retrieved else None,
        "first_match_rank": next((idx for idx, matched in enumerate(matches, start=1) if matched), None),
    }


def reciprocal_rank(matches: list[bool]) -> float:
    for rank, matched in enumerate(matches, start=1):
        if matched:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(matches: list[bool], ideal_hits: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, matched in enumerate(matches, start=1) if matched)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0.0 else 0.0


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return {
        "query_count": len(rows),
        "hit_at_k": statistics.mean(float(row["hit_at_k"]) for row in rows),
        "recall_at_k": statistics.mean(float(row["recall_at_k"]) for row in rows),
        "mrr_at_k": statistics.mean(float(row["mrr_at_k"]) for row in rows),
        "ndcg_at_k": statistics.mean(float(row["ndcg_at_k"]) for row in rows),
        "avg_latency_ms": statistics.mean(float(row["latency_ms"]) for row in rows),
        "avg_retrieval_count": statistics.mean(float(row["retrieval_count"]) for row in rows),
    }


def write_outputs(output_dir: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "beir_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "per_query.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if rows:
        with (output_dir / "per_query.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    lines = [
        "# BEIR Retrieval Benchmark",
        "",
        f"- Dataset: `{summary['dataset']}`",
        f"- Split: `{summary['split']}`",
        f"- Corpus docs: {summary['corpus_count']}",
        f"- Queries: {summary['query_count']}",
        f"- Top-k: {summary['top_k']}",
        "",
        "| Strategy | Hit@k | Recall@k | MRR@k | nDCG@k | Avg latency ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for strategy, metrics in summary["strategies"].items():
        lines.append(
            f"| {strategy} | {metrics['hit_at_k']:.3f} | {metrics['recall_at_k']:.3f} | "
            f"{metrics['mrr_at_k']:.3f} | {metrics['ndcg_at_k']:.3f} | {metrics['avg_latency_ms']:.2f} |"
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (RESULTS_ROOT / f"{args.dataset}_{timestamp}")

    corpus, queries, qrels = load_beir_dataset(args)
    corpus, queries, qrels = limit_dataset(
        corpus,
        queries,
        qrels,
        query_limit=args.query_limit,
        corpus_limit=args.corpus_limit,
    )
    chunks = corpus_to_chunks(args.dataset, corpus)
    reranker = make_reranker(args.reranker, model_name=args.reranker_model)
    strategies = expand_strategies(args.strategy)
    dense_needed = any(strategy in {"dense", "hybrid", "hybrid_rerank"} for strategy in strategies)
    retriever = HybridRetriever(
        chunks,
        model_name=args.dense_model,
        dense_backend=args.dense_backend,
        dense_preset=args.dense_preset,
        dense_query_model_name=args.dense_query_model,
        dense_passage_model_name=args.dense_passage_model,
        build_dense=(not args.skip_dense and dense_needed),
        build_colbert=args.build_colbert,
        colbert_model_name=args.colbert_model,
        reranker=reranker,
    )

    if "colbert" in strategies and not args.build_colbert:
        if "colbert" in args.strategy:
            raise RuntimeError("Strategy 'colbert' requires --build-colbert.")
        strategies = [strategy for strategy in strategies if strategy != "colbert"]

    rows: list[dict[str, Any]] = []
    summary_by_strategy: dict[str, Any] = {}
    for strategy in strategies:
        strategy_rows: list[dict[str, Any]] = []
        for qid, query in queries.items():
            config = build_config(args, strategy=strategy)
            start = time.perf_counter()
            result = retriever.search_result(query, strategy=strategy, config=config)
            latency_ms = (time.perf_counter() - start) * 1000.0
            metrics = evaluate_hits(qid, result.hits, qrels, k=args.top_k)
            row = {
                "strategy": strategy,
                "query_id": qid,
                "query": query,
                "latency_ms": latency_ms,
                "retrieval_count": len(result.hits),
                **metrics,
            }
            rows.append(row)
            strategy_rows.append(row)
        summary_by_strategy[strategy] = summarize(strategy_rows)

    summary = {
        "timestamp_utc": timestamp,
        "dataset": args.dataset,
        "split": args.split,
        "top_k": args.top_k,
        "corpus_count": len(corpus),
        "query_count": len(queries),
        "dense_preset": args.dense_preset,
        "dense_model": retriever.dense.model_name,
        "dense_backend": retriever.dense.backend_name,
        "colbert_built": args.build_colbert,
        "colbert_model": args.colbert_model,
        "strategies": summary_by_strategy,
    }
    write_outputs(output_dir, summary, rows)
    print(output_dir)


if __name__ == "__main__":
    main()
