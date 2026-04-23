from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import make_reranker
from app.retrieval.schemas import RetrievalConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query a saved retrieval index.")
    parser.add_argument("--index-dir", type=Path, required=True, help="Retrieval index directory.")
    parser.add_argument("--query", required=True, help="Query string.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of hits.")
    parser.add_argument(
        "--strategy",
        choices=["bm25", "dense", "colbert", "hybrid", "hybrid_rerank"],
        default="hybrid",
        help="Retrieval strategy.",
    )
    parser.add_argument("--combination", choices=["weighted_sum", "rrf"], default="weighted_sum")
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--bm25-weight", type=float, default=0.5)
    parser.add_argument("--dense-weight", type=float, default=0.5)
    parser.add_argument("--rerank-top-n", type=int, default=20)
    parser.add_argument("--reranker", choices=["none", "heuristic", "cross-encoder", "colbert"], default="heuristic")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reranker = make_reranker(args.reranker)
    retriever = HybridRetriever.load(args.index_dir, reranker=reranker)
    config = RetrievalConfig(
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        bm25_weight=args.bm25_weight,
        dense_weight=args.dense_weight,
        combination=args.combination,
        rerank_top_n=args.rerank_top_n if args.strategy == "hybrid_rerank" else 0,
        use_rerank=args.strategy == "hybrid_rerank",
    )
    result = retriever.search_result(args.query, strategy=args.strategy, config=config)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
