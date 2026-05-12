from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.colbert_retriever import ColBERTRetriever, DEFAULT_COLBERT_MODEL_NAME
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.index_store import RetrievalIndexStore
from app.retrieval.reranker import NoOpReranker, Reranker
from app.retrieval.schemas import (
    DocumentChunkRef,
    RankedScore,
    RetrievedHit,
    RetrievalConfig,
    RetrievalResult,
    coerce_chunk_refs,
)


class HybridRetriever:
    """Hybrid sparse+dense retriever with weighted sum and RRF fusion."""

    def __init__(
        self,
        chunks: Iterable[Any],
        *,
        model_name: str | None = None,
        dense_backend: str | None = None,
        dense_preset: str | None = None,
        dense_query_model_name: str | None = None,
        dense_passage_model_name: str | None = None,
        build_dense: bool = True,
        build_colbert: bool = False,
        colbert_model_name: str = DEFAULT_COLBERT_MODEL_NAME,
        reranker: Reranker | None = None,
    ) -> None:
        self.chunks: list[DocumentChunkRef] = coerce_chunk_refs(chunks)
        self.bm25 = BM25Retriever(self.chunks)
        self.dense = DenseRetriever(
            self.chunks,
            model_name=model_name,
            backend=dense_backend,
            preset=dense_preset,
            query_model_name=dense_query_model_name,
            passage_model_name=dense_passage_model_name,
        )
        self.colbert: ColBERTRetriever | None = ColBERTRetriever(
            self.chunks,
            model_name=colbert_model_name,
        )
        self.reranker: Reranker = reranker or NoOpReranker()
        if build_dense:
            self.dense.build()
        if build_colbert and self.colbert is not None:
            self.colbert.build()

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        candidate_k: int | None = None,
        config: RetrievalConfig | None = None,
    ) -> list[RetrievedHit]:
        runtime_config = config or RetrievalConfig()
        if top_k is not None:
            runtime_config.top_k = top_k
        if candidate_k is not None:
            runtime_config.candidate_k = candidate_k

        bm25_scores = self.bm25.search_scores(
            query,
            top_k=runtime_config.candidate_limit(),
            config=runtime_config,
        )
        dense_scores = self.dense.search_scores(
            query,
            top_k=runtime_config.candidate_limit(),
            config=runtime_config,
        )

        hits = self._fuse_scores(query, bm25_scores, dense_scores, runtime_config)
        return hits[: runtime_config.top_k]

    def search_result(
        self,
        query: str,
        *,
        strategy: str = "hybrid",
        config: RetrievalConfig | None = None,
    ) -> RetrievalResult:
        start = time.perf_counter()
        runtime_config = config or RetrievalConfig()

        if strategy == "bm25":
            hits = self.bm25.search(query, top_k=runtime_config.top_k, config=runtime_config)
        elif strategy == "dense":
            hits = self.dense.search(query, top_k=runtime_config.top_k, config=runtime_config)
        elif strategy == "colbert":
            if self.colbert is None or not self.colbert.doc_embeddings:
                raise RuntimeError("ColBERT index is not built or loaded")
            hits = self.colbert.search(query, top_k=runtime_config.top_k, config=runtime_config)
        elif strategy in {"hybrid", "hybrid_rerank"}:
            if strategy == "hybrid_rerank":
                runtime_config.use_rerank = True
                runtime_config.rerank_top_n = runtime_config.rerank_top_n or runtime_config.candidate_limit()
            hits = self.search(query, config=runtime_config)
        else:
            raise ValueError(f"Unknown retrieval strategy: {strategy}")

        if runtime_config.context_window > 0:
            hits = self._expand_context_hits(hits, runtime_config)

        latency_ms = (time.perf_counter() - start) * 1000.0
        return RetrievalResult(
            query=query,
            strategy=strategy,
            hits=hits,
            config=runtime_config,
            latency_ms=latency_ms,
            retrieval_count=len(hits),
        )

    def _expand_context_hits(self, hits: list[RetrievedHit], config: RetrievalConfig) -> list[RetrievedHit]:
        """Attach adjacent chunks so QA can read split table/list evidence."""

        if not hits or config.context_window <= 0:
            return hits

        index_by_chunk_id = {chunk.chunk_id: idx for idx, chunk in enumerate(self.chunks)}
        expanded: list[RetrievedHit] = []
        seen: set[str] = set()
        for hit in hits:
            base_idx = index_by_chunk_id.get(hit.chunk_id)
            if base_idx is None:
                if hit.chunk_id not in seen:
                    expanded.append(hit)
                    seen.add(hit.chunk_id)
                continue

            neighbor_indices: list[int] = []
            for distance in range(1, config.context_window + 1):
                left_idx = base_idx - distance
                right_idx = base_idx + distance
                if left_idx >= 0:
                    neighbor_indices.append(left_idx)
                if right_idx < len(self.chunks):
                    neighbor_indices.append(right_idx)

            for idx in [base_idx, *neighbor_indices]:
                chunk = self.chunks[idx]
                if chunk.chunk_id in seen:
                    continue
                if idx == base_idx:
                    expanded.append(hit)
                else:
                    base_score = float(hit.final_score or hit.score)
                    expanded.append(
                        RetrievedHit(
                            chunk=chunk,
                            score=max(0.0, base_score - 0.01 * abs(idx - base_idx)),
                            source=hit.source,
                            bm25_score=hit.bm25_score,
                            dense_score=hit.dense_score,
                            rerank_score=hit.rerank_score,
                            final_score=max(0.0, base_score - 0.01 * abs(idx - base_idx)),
                            source_scores=dict(hit.source_scores),
                            raw_scores=dict(hit.raw_scores),
                            metadata={**hit.metadata, "context_neighbor_of": hit.chunk_id},
                        )
                    )
                seen.add(chunk.chunk_id)

        return [as_ranked_hit(hit, rank) for rank, hit in enumerate(expanded, start=1)]

    def save(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        store = RetrievalIndexStore(output_path)
        chunk_refs = store.write_corpus(self.chunks)
        self.bm25.save_metadata(output_path)
        if self.dense.embeddings is not None:
            self.dense.save(output_path)
        if self.colbert is not None and self.colbert.doc_embeddings:
            self.colbert.save(output_path)

        dense_built = self.dense.embeddings is not None
        colbert_built = self.colbert is not None and bool(self.colbert.doc_embeddings)
        files = {
            "corpus": RetrievalIndexStore.CORPUS_FILE,
            "bm25_config": "bm25_config.json",
            "bm25_tokens": "bm25_tokens.json",
        }
        if dense_built:
            files.update(
                {
                    "dense_config": "dense_config.json",
                    "dense_embeddings": "dense_embeddings.npy",
                }
            )
        if colbert_built:
            files.update(
                {
                    "colbert_config": "colbert_config.json",
                    "colbert_embeddings": "colbert_embeddings.npz",
                }
            )

        manifest = {
            "retrievers": [
                "bm25",
                *([] if not dense_built else ["dense"]),
                *([] if not colbert_built else ["colbert"]),
            ],
            "chunk_count": len(chunk_refs),
            "dense_model_name": self.dense.model_name,
            "dense_backend": self.dense.backend_name,
            "colbert_model_name": self.colbert.model_name if self.colbert else None,
            "files": files,
        }
        store.write_manifest(manifest)
        (output_path / "hybrid_config.json").write_text(
            json.dumps(
                {
                    "chunk_count": len(chunk_refs),
                    "dense_model_name": self.dense.model_name,
                    "dense_backend": self.dense.backend_name,
                    "colbert_model_name": self.colbert.model_name if self.colbert else None,
                    "default_combination": "weighted_sum",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        input_dir: str | Path,
        *,
        reranker: Reranker | None = None,
        load_dense: bool = True,
        load_colbert: bool = True,
    ) -> "HybridRetriever":
        input_path = Path(input_dir)
        chunks = RetrievalIndexStore(input_path).read_corpus()
        retriever = cls(chunks, build_dense=False, reranker=reranker)
        dense_config = input_path / "dense_config.json"
        dense_embeddings = input_path / "dense_embeddings.npy"
        if load_dense and dense_config.exists() and dense_embeddings.exists():
            retriever.dense = DenseRetriever.load(input_path, chunks)
        colbert_config = input_path / "colbert_config.json"
        colbert_embeddings = input_path / "colbert_embeddings.npz"
        if load_colbert and colbert_config.exists() and colbert_embeddings.exists():
            retriever.colbert = ColBERTRetriever.load(input_path, chunks)
        return retriever

    def _fuse_scores(
        self,
        query: str,
        bm25_scores: list[RankedScore],
        dense_scores: list[RankedScore],
        config: RetrievalConfig,
    ) -> list[RetrievedHit]:
        bm25_by_idx = {score.index: score for score in bm25_scores}
        dense_by_idx = {score.index: score for score in dense_scores}
        candidate_indices = sorted(set(bm25_by_idx) | set(dense_by_idx))
        if not candidate_indices:
            return []

        bm25_norm = _normalize_raw_scores(bm25_scores)
        dense_norm = _normalize_raw_scores(dense_scores)

        if config.combination == "rrf":
            final_scores = self._rrf_scores(candidate_indices, bm25_by_idx, dense_by_idx, config)
        else:
            final_scores = self._weighted_scores(candidate_indices, bm25_norm, dense_norm, config)

        hits: list[RetrievedHit] = []
        for idx in candidate_indices:
            chunk = self.chunks[idx]
            bm25_score = float(bm25_norm.get(idx, 0.0))
            dense_score = float(dense_norm.get(idx, 0.0))
            raw_scores = {
                "bm25": float(bm25_by_idx[idx].raw_score) if idx in bm25_by_idx else 0.0,
                "dense": float(dense_by_idx[idx].raw_score) if idx in dense_by_idx else 0.0,
            }
            final_score = float(final_scores.get(idx, 0.0))
            hits.append(
                RetrievedHit(
                    chunk=chunk,
                    score=final_score,
                    source="hybrid",
                    bm25_score=bm25_score,
                    dense_score=dense_score,
                    final_score=final_score,
                    source_scores={"bm25": bm25_score, "dense": dense_score},
                    raw_scores=raw_scores,
                    metadata={"combination": config.combination},
                )
            )

        hits.sort(key=lambda item: float(item.final_score or item.score), reverse=True)
        hits = [as_ranked_hit(hit, rank) for rank, hit in enumerate(hits, start=1)]

        if config.use_rerank and config.rerank_top_n > 0:
            hits = self.reranker.rerank(query, hits, top_n=config.rerank_top_n)

        return hits

    def _weighted_scores(
        self,
        candidate_indices: list[int],
        bm25_norm: dict[int, float],
        dense_norm: dict[int, float],
        config: RetrievalConfig,
    ) -> dict[int, float]:
        bm25_weight, dense_weight = config.normalized_weights()
        return {
            idx: bm25_weight * bm25_norm.get(idx, 0.0) + dense_weight * dense_norm.get(idx, 0.0)
            for idx in candidate_indices
        }

    def _rrf_scores(
        self,
        candidate_indices: list[int],
        bm25_by_idx: dict[int, RankedScore],
        dense_by_idx: dict[int, RankedScore],
        config: RetrievalConfig,
    ) -> dict[int, float]:
        bm25_weight, dense_weight = config.normalized_weights()
        raw_scores: dict[int, float] = {}
        for idx in candidate_indices:
            score = 0.0
            if idx in bm25_by_idx:
                score += bm25_weight / (config.rrf_k + bm25_by_idx[idx].rank)
            if idx in dense_by_idx:
                score += dense_weight / (config.rrf_k + dense_by_idx[idx].rank)
            raw_scores[idx] = score

        max_score = max(raw_scores.values()) if raw_scores else 0.0
        if max_score <= 0.0:
            return raw_scores
        return {idx: score / max_score for idx, score in raw_scores.items()}


def as_ranked_hit(hit: RetrievedHit, rank: int) -> RetrievedHit:
    payload = asdict(hit)
    payload["rank"] = rank
    payload["chunk"] = hit.chunk
    return RetrievedHit(**payload)


def _normalize_raw_scores(scores: list[RankedScore]) -> dict[int, float]:
    if not scores:
        return {}

    raw_values = [score.raw_score for score in scores]
    min_score = min(raw_values)
    max_score = max(raw_values)
    if max_score == min_score:
        return {score.index: 1.0 if max_score > 0.0 else 0.0 for score in scores}

    return {
        score.index: max(0.0, min(1.0, (score.raw_score - min_score) / (max_score - min_score)))
        for score in scores
    }


RetrievalHit = RetrievedHit
