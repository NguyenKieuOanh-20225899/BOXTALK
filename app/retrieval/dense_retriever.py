from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from app.retrieval.embedding_backends import (
    DENSE_MODEL_PRESETS,
    DenseModelSpec,
    EmbeddingBackend,
    make_embedding_backend,
    resolve_dense_model_spec,
)
from app.retrieval.schemas import (
    DocumentChunkRef,
    RankedScore,
    RetrievedHit,
    RetrievalConfig,
    chunk_matches_config,
    coerce_chunk_refs,
)


DEFAULT_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DENSE_PRESET = "minilm"


class DenseRetriever:
    """Sentence-transformer dual-encoder dense retrieval baseline."""

    def __init__(
        self,
        chunks: Iterable[Any] = (),
        *,
        model_name: str | None = None,
        backend: str | None = None,
        preset: str | None = None,
        query_model_name: str | None = None,
        passage_model_name: str | None = None,
        batch_size: int = 32,
        device: str | None = None,
        max_length: int = 256,
    ) -> None:
        self.chunks: list[DocumentChunkRef] = coerce_chunk_refs(chunks)
        self.model_spec: DenseModelSpec = resolve_dense_model_spec(
            model_name=model_name or os.getenv("BOXTALK_EMBED_MODEL_NAME"),
            backend=backend or os.getenv("BOXTALK_DENSE_BACKEND"),
            preset=preset or os.getenv("BOXTALK_DENSE_PRESET") or DEFAULT_DENSE_PRESET,
            query_model_name=query_model_name or os.getenv("BOXTALK_DENSE_QUERY_MODEL_NAME"),
            passage_model_name=passage_model_name or os.getenv("BOXTALK_DENSE_PASSAGE_MODEL_NAME"),
        )
        self.model_name = self.model_spec.model_name
        self.backend_name = self.model_spec.backend
        self.query_model_name = self.model_spec.query_model_name
        self.passage_model_name = self.model_spec.passage_model_name
        self.batch_size = batch_size
        self.device = device or os.getenv("BOXTALK_EMBED_DEVICE") or self._default_device()
        self.max_length = max_length
        self._backend: EmbeddingBackend | None = None
        self._embeddings: np.ndarray | None = None

    @property
    def embeddings(self) -> np.ndarray | None:
        return self._embeddings

    def build(self, chunks: Iterable[Any] | None = None) -> None:
        """Encode all chunks and keep L2-normalized embeddings in memory."""

        if chunks is not None:
            self.chunks = coerce_chunk_refs(chunks)

        if not self.chunks:
            self._embeddings = np.zeros((0, 0), dtype=np.float32)
            return

        texts = [chunk.searchable_text() for chunk in self.chunks]
        embeddings = self._encode_passages(texts)
        self._embeddings = self._normalize_rows(embeddings)

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
                    source="dense",
                    rank=scored.rank,
                    dense_score=scored.score,
                    source_scores={"dense": scored.score},
                    raw_scores={"dense": scored.raw_score},
                )
            )
        return hits

    def search_scores(
        self,
        query: str,
        top_k: int = 10,
        config: RetrievalConfig | None = None,
    ) -> list[RankedScore]:
        """Return cosine-similarity scores with corpus indices."""

        if self._embeddings is None or self._embeddings.size == 0 or not self.chunks:
            return []

        query_embedding = self._normalize_rows(self._encode_queries([query]))
        if query_embedding.size == 0:
            return []

        similarities = np.dot(self._embeddings, query_embedding[0])
        candidates: list[tuple[int, float]] = []
        for idx, raw_score in enumerate(similarities.tolist()):
            chunk = self.chunks[idx]
            if not chunk_matches_config(chunk, config):
                continue
            candidates.append((idx, float(raw_score)))

        if not candidates:
            return []

        ranked = sorted(candidates, key=lambda item: item[1], reverse=True)[:top_k]
        return [
            RankedScore(
                index=idx,
                chunk=self.chunks[idx],
                score=self._cosine_to_unit(raw_score),
                raw_score=float(raw_score),
                rank=rank,
                source="dense",
            )
            for rank, (idx, raw_score) in enumerate(ranked, start=1)
        ]

    def save(self, output_dir: str | Path) -> None:
        if self._embeddings is None:
            raise RuntimeError("DenseRetriever index is not built")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        np.save(output_path / "dense_embeddings.npy", self._embeddings)
        self._write_faiss_if_available(output_path)
        (output_path / "dense_config.json").write_text(
            json.dumps(
                {
                    "model_name": self.model_name,
                    "backend": self.backend_name,
                    "query_model_name": self.query_model_name,
                    "passage_model_name": self.passage_model_name,
                    "batch_size": self.batch_size,
                    "device": self.device,
                    "max_length": self.max_length,
                    "embedding_shape": list(self._embeddings.shape),
                    "similarity": "cosine",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, input_dir: str | Path, chunks: Iterable[Any]) -> "DenseRetriever":
        input_path = Path(input_dir)
        config = json.loads((input_path / "dense_config.json").read_text(encoding="utf-8"))
        retriever = cls(
            chunks,
            model_name=config.get("model_name"),
            backend=config.get("backend"),
            query_model_name=config.get("query_model_name"),
            passage_model_name=config.get("passage_model_name"),
            batch_size=int(config.get("batch_size", 32)),
            device=config.get("device"),
            max_length=int(config.get("max_length", 256)),
        )
        retriever._embeddings = np.load(input_path / "dense_embeddings.npy", allow_pickle=False)
        return retriever

    @staticmethod
    def available_presets() -> dict[str, DenseModelSpec]:
        return dict(DENSE_MODEL_PRESETS)

    def _encode_queries(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        embeddings = self._get_backend().encode_queries(texts, batch_size=self.batch_size)
        return np.asarray(embeddings, dtype=np.float32)

    def _encode_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        embeddings = self._get_backend().encode_passages(texts, batch_size=self.batch_size)
        return np.asarray(embeddings, dtype=np.float32)

    def _get_backend(self) -> EmbeddingBackend:
        if self._backend is not None:
            return self._backend
        self._backend = make_embedding_backend(
            self.model_spec,
            device=self.device,
            max_length=self.max_length,
        )
        return self._backend

    @staticmethod
    def _normalize_rows(embeddings: np.ndarray) -> np.ndarray:
        if embeddings.size == 0:
            return embeddings.astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (embeddings / norms).astype(np.float32)

    @staticmethod
    def _cosine_to_unit(score: float) -> float:
        return max(0.0, min(1.0, (float(score) + 1.0) / 2.0))

    def _write_faiss_if_available(self, output_path: Path) -> None:
        if self._embeddings is None or self._embeddings.size == 0:
            return
        try:
            import faiss  # type: ignore
        except Exception:
            return
        index = faiss.IndexFlatIP(self._embeddings.shape[1])
        index.add(self._embeddings)
        faiss.write_index(index, str(output_path / "dense.faiss"))

    @staticmethod
    def _default_device() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"
