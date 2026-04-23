from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

from app.retrieval.schemas import (
    DocumentChunkRef,
    RankedScore,
    RetrievedHit,
    RetrievalConfig,
    chunk_matches_config,
    coerce_chunk_refs,
)


class BM25Retriever:
    """Sparse lexical retriever over chunk text and structural headings."""

    WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

    def __init__(self, chunks: Iterable[Any] = ()) -> None:
        self.chunks: list[DocumentChunkRef] = coerce_chunk_refs(chunks)
        self._tokenized_chunks: list[list[str]] = []
        self._bm25: Any | None = None
        self.build()

    @property
    def tokenized_chunks(self) -> list[list[str]]:
        return self._tokenized_chunks

    def build(self, chunks: Iterable[Any] | None = None) -> None:
        """Build the BM25 index from the provided chunks."""

        if chunks is not None:
            self.chunks = coerce_chunk_refs(chunks)

        self._tokenized_chunks = [self.tokenize(chunk.searchable_text()) for chunk in self.chunks]
        if not self._tokenized_chunks:
            self._bm25 = None
            return

        try:
            from rank_bm25 import BM25Okapi
        except Exception as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("BM25Retriever requires rank-bm25. Install rank-bm25>=0.2.2.") from exc

        self._bm25 = BM25Okapi(self._tokenized_chunks)

    def search(
        self,
        query: str,
        top_k: int = 10,
        config: RetrievalConfig | None = None,
    ) -> list[RetrievedHit]:
        scores = self.search_scores(query, top_k=top_k, config=config)
        hits: list[RetrievedHit] = []
        for scored in scores:
            hits.append(
                RetrievedHit(
                    chunk=scored.chunk,
                    score=scored.score,
                    source="bm25",
                    rank=scored.rank,
                    bm25_score=scored.score,
                    source_scores={"bm25": scored.score},
                    raw_scores={"bm25": scored.raw_score},
                )
            )
        return hits

    def search_scores(
        self,
        query: str,
        top_k: int = 10,
        config: RetrievalConfig | None = None,
    ) -> list[RankedScore]:
        """Return normalized BM25 scores with corpus indices."""

        if self._bm25 is None or not self.chunks:
            return []

        q_tokens = self.tokenize(query)
        if not q_tokens:
            return []

        raw_scores = [float(score) for score in self._bm25.get_scores(q_tokens)]
        candidates: list[tuple[int, float]] = []
        for idx, raw_score in enumerate(raw_scores):
            if raw_score <= 0.0:
                continue
            chunk = self.chunks[idx]
            if not chunk_matches_config(chunk, config):
                continue
            candidates.append((idx, raw_score))

        if not candidates:
            return []

        max_score = max(score for _, score in candidates)
        if max_score <= 0.0:
            return []

        ranked = sorted(candidates, key=lambda item: item[1], reverse=True)[:top_k]
        return [
            RankedScore(
                index=idx,
                chunk=self.chunks[idx],
                score=float(raw_score / max_score),
                raw_score=float(raw_score),
                rank=rank,
                source="bm25",
            )
            for rank, (idx, raw_score) in enumerate(ranked, start=1)
        ]

    def save_metadata(self, output_dir: str | Path) -> None:
        """Persist lightweight BM25 metadata; the index is rebuilt from corpus on load."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "retriever": "bm25",
            "chunk_count": len(self.chunks),
            "tokenizer": "unicode_word_lower_nfkc",
            "tokens_path": "bm25_tokens.json",
        }
        (output_path / "bm25_config.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_path / "bm25_tokens.json").write_text(
            json.dumps(self._tokenized_chunks, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, input_dir: str | Path, chunks: Iterable[Any]) -> "BM25Retriever":
        _ = input_dir
        return cls(chunks)

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        normalized = unicodedata.normalize("NFKC", text or "").lower()
        return [token for token in cls.WORD_RE.findall(normalized) if token]
