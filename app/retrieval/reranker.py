from __future__ import annotations

import math
import re
from dataclasses import replace
from typing import Any, Protocol

from app.retrieval.colbert_retriever import DEFAULT_COLBERT_MODEL_NAME, ColBERTRetriever
from app.retrieval.schemas import DocumentChunkRef, RetrievedHit


class Reranker(Protocol):
    """Interface for optional second-stage reranking."""

    def score(self, query: str, chunk: DocumentChunkRef) -> float:
        ...

    def rerank(self, query: str, hits: list[RetrievedHit], top_n: int | None = None) -> list[RetrievedHit]:
        ...


class NoOpReranker:
    """Default reranker that preserves first-stage order."""

    def score(self, query: str, chunk: DocumentChunkRef) -> float:
        _ = query, chunk
        return 0.0

    def rerank(self, query: str, hits: list[RetrievedHit], top_n: int | None = None) -> list[RetrievedHit]:
        _ = query, top_n
        return [replace(hit, rank=rank) for rank, hit in enumerate(hits, start=1)]


class HeuristicReranker:
    """Lightweight lexical/structure reranker useful before adding cross-encoders."""

    WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

    TABLE_TERMS = {
        "table",
        "price",
        "cost",
        "amount",
        "total",
        "schedule",
        "fee",
        "invoice",
        "budget",
        "bảng",
        "giá",
        "chi phí",
    }

    def __init__(self, *, blend_weight: float = 0.2) -> None:
        self.blend_weight = max(0.0, min(1.0, blend_weight))

    def score(self, query: str, chunk: DocumentChunkRef) -> float:
        query_lower = query.lower().strip()
        chunk_text = (chunk.text or "").lower()
        heading = chunk.heading_path_text.lower()
        section = (chunk.section or "").lower()
        tokens = {token.lower() for token in self.WORD_RE.findall(query_lower)}
        chunk_tokens = {
            token.lower()
            for token in self.WORD_RE.findall(" ".join([heading, section, chunk_text]))
        }
        chunk_word_count = len((chunk.text or "").split())

        score = 0.0
        overlap = len(tokens & chunk_tokens)
        if tokens:
            score += 0.45 * (overlap / len(tokens))

        if heading and any(token in heading for token in tokens):
            score += 0.20
        if section and any(token in section for token in tokens):
            score += 0.15
        if query_lower and query_lower in chunk_text:
            score += 0.20

        if chunk.block_type == "table" or chunk.metadata.get("is_table_chunk"):
            if tokens & self.TABLE_TERMS:
                score += 0.20

        if chunk.block_type == "heading":
            score += 0.05

        if query_lower.startswith(("who ", "when ", "which ", "how ", "what ")):
            if chunk.block_type == "heading" and chunk_word_count <= 3:
                score -= 0.20
            if "@" in chunk_text and ("email" in tokens or "contact" in tokens):
                score += 0.25
            if tokens & {"when", "time", "long", "days"} and any(ch.isdigit() for ch in chunk_text):
                score += 0.15
            if "who" in tokens and chunk_word_count >= 4:
                score += 0.10

        return max(0.0, min(1.0, score))

    def rerank(self, query: str, hits: list[RetrievedHit], top_n: int | None = None) -> list[RetrievedHit]:
        limit = top_n or len(hits)
        selected = hits[:limit]
        rest = hits[limit:]

        reranked: list[RetrievedHit] = []
        for hit in selected:
            rerank_score = self.score(query, hit.chunk)
            base_score = float(hit.final_score if hit.final_score is not None else hit.score)
            final_score = (1.0 - self.blend_weight) * base_score + self.blend_weight * rerank_score
            source_scores = {**hit.source_scores, "rerank": rerank_score}
            raw_scores = {**hit.raw_scores, "rerank": rerank_score}
            reranked.append(
                replace(
                    hit,
                    score=final_score,
                    source="rerank",
                    rerank_score=rerank_score,
                    final_score=final_score,
                    source_scores=source_scores,
                    raw_scores=raw_scores,
                )
            )

        reranked.sort(key=lambda item: float(item.final_score or item.score), reverse=True)
        merged = reranked + rest
        return [replace(hit, rank=rank) for rank, hit in enumerate(merged, start=1)]


class CrossEncoderReranker:
    """Cross-encoder reranker with a sentence-transformers backend."""

    def __init__(
        self,
        *,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any | None = None

    def score(self, query: str, chunk: DocumentChunkRef) -> float:
        model = self._get_model()
        raw = float(model.predict([(query, chunk.searchable_text())])[0])
        return 1.0 / (1.0 + math.exp(-raw))

    def rerank(self, query: str, hits: list[RetrievedHit], top_n: int | None = None) -> list[RetrievedHit]:
        limit = top_n or len(hits)
        selected = hits[:limit]
        rest = hits[limit:]
        if not selected:
            return hits

        model = self._get_model()
        pairs = [(query, hit.chunk.searchable_text()) for hit in selected]
        raw_scores = [float(score) for score in model.predict(pairs)]
        normalized_scores = [1.0 / (1.0 + math.exp(-score)) for score in raw_scores]

        reranked = []
        for hit, raw_score, score in zip(selected, raw_scores, normalized_scores):
            reranked.append(
                replace(
                    hit,
                    score=score,
                    source="rerank",
                    rerank_score=score,
                    final_score=score,
                    source_scores={**hit.source_scores, "rerank": score},
                    raw_scores={**hit.raw_scores, "rerank": raw_score},
                )
            )
        reranked.sort(key=lambda item: float(item.final_score or item.score), reverse=True)
        merged = reranked + rest
        return [replace(hit, rank=rank) for rank, hit in enumerate(merged, start=1)]

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("CrossEncoderReranker requires sentence-transformers.") from exc
        kwargs: dict[str, Any] = {}
        if self.device:
            kwargs["device"] = self.device
        self._model = CrossEncoder(self.model_name, **kwargs)
        return self._model


class ColBERTReranker:
    """ColBERT late-interaction reranker over a first-stage candidate set."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_COLBERT_MODEL_NAME,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device

    def score(self, query: str, chunk: DocumentChunkRef) -> float:
        retriever = ColBERTRetriever([chunk], model_name=self.model_name, device=self.device)
        retriever.build()
        scores = retriever.search_scores(query, top_k=1)
        return scores[0].score if scores else 0.0

    def rerank(self, query: str, hits: list[RetrievedHit], top_n: int | None = None) -> list[RetrievedHit]:
        limit = top_n or len(hits)
        selected = hits[:limit]
        rest = hits[limit:]
        if not selected:
            return hits

        retriever = ColBERTRetriever(
            [hit.chunk for hit in selected],
            model_name=self.model_name,
            device=self.device,
        )
        retriever.build()
        scores = retriever.search_scores(query, top_k=len(selected))
        score_by_chunk_id = {score.chunk.chunk_id: score for score in scores}

        reranked: list[RetrievedHit] = []
        for hit in selected:
            scored = score_by_chunk_id.get(hit.chunk.chunk_id)
            rerank_score = float(scored.score) if scored else 0.0
            raw_score = float(scored.raw_score) if scored else 0.0
            reranked.append(
                replace(
                    hit,
                    score=rerank_score,
                    source="rerank",
                    rerank_score=rerank_score,
                    final_score=rerank_score,
                    source_scores={**hit.source_scores, "colbert": rerank_score, "rerank": rerank_score},
                    raw_scores={**hit.raw_scores, "colbert": raw_score, "rerank": raw_score},
                )
            )

        reranked.sort(key=lambda item: float(item.final_score or item.score), reverse=True)
        merged = reranked + rest
        return [replace(hit, rank=rank) for rank, hit in enumerate(merged, start=1)]


def make_reranker(name: str | None, *, model_name: str | None = None, device: str | None = None) -> Reranker:
    normalized = (name or "none").strip().lower()
    if normalized in {"none", "noop", "no-op"}:
        return NoOpReranker()
    if normalized in {"heuristic", "light"}:
        return HeuristicReranker()
    if normalized in {"cross-encoder", "cross_encoder", "crossencoder"}:
        kwargs: dict[str, Any] = {}
        if model_name:
            kwargs["model_name"] = model_name
        if device:
            kwargs["device"] = device
        return CrossEncoderReranker(**kwargs)
    if normalized == "colbert":
        kwargs = {}
        if model_name:
            kwargs["model_name"] = model_name
        if device:
            kwargs["device"] = device
        return ColBERTReranker(**kwargs)
    raise ValueError(f"Unknown reranker: {name}")
