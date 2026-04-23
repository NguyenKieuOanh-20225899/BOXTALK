from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.schemas import RetrievalConfig, RetrievalResult


class RetrievalService:
    """Thin service boundary for routers to choose retrieval strategy/config."""

    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever

    @classmethod
    def from_chunks(
        cls,
        chunks: Iterable[Any],
        *,
        model_name: str | None = None,
        dense_backend: str | None = None,
        dense_preset: str | None = None,
        dense_query_model_name: str | None = None,
        dense_passage_model_name: str | None = None,
        build_dense: bool = True,
        reranker: Reranker | None = None,
    ) -> "RetrievalService":
        return cls(
            HybridRetriever(
                chunks,
                model_name=model_name,
                dense_backend=dense_backend,
                dense_preset=dense_preset,
                dense_query_model_name=dense_query_model_name,
                dense_passage_model_name=dense_passage_model_name,
                build_dense=build_dense,
                reranker=reranker,
            )
        )

    @classmethod
    def from_index(cls, index_dir: str | Path, *, reranker: Reranker | None = None) -> "RetrievalService":
        return cls(HybridRetriever.load(index_dir, reranker=reranker))

    def retrieve(
        self,
        query: str,
        *,
        strategy: str = "hybrid",
        config: RetrievalConfig | None = None,
    ) -> RetrievalResult:
        return self.retriever.search_result(query, strategy=strategy, config=config)
