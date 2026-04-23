from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from app.retrieval.schemas import (
    DocumentChunkRef,
    RankedScore,
    RetrievedHit,
    RetrievalConfig,
    chunk_matches_config,
    coerce_chunk_refs,
)


DEFAULT_COLBERT_MODEL_NAME = "colbert-ir/colbertv2.0"


class ColBERTRetriever:
    """Exact ColBERT-style late-interaction retriever for small/medium local corpora."""

    def __init__(
        self,
        chunks: Iterable[Any] = (),
        *,
        model_name: str = DEFAULT_COLBERT_MODEL_NAME,
        device: str | None = None,
        batch_size: int = 8,
        query_max_length: int = 48,
        doc_max_length: int = 256,
    ) -> None:
        self.chunks: list[DocumentChunkRef] = coerce_chunk_refs(chunks)
        self.model_name = model_name
        self.device = device or self._default_device()
        self.batch_size = batch_size
        self.query_max_length = query_max_length
        self.doc_max_length = doc_max_length
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._doc_embeddings: list[np.ndarray] = []

    @property
    def doc_embeddings(self) -> list[np.ndarray]:
        return self._doc_embeddings

    def build(self, chunks: Iterable[Any] | None = None) -> None:
        """Encode document chunks as token embeddings."""

        if chunks is not None:
            self.chunks = coerce_chunk_refs(chunks)
        texts = [chunk.searchable_text() for chunk in self.chunks]
        self._doc_embeddings = self._encode_token_embeddings(
            texts,
            max_length=self.doc_max_length,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
        config: RetrievalConfig | None = None,
    ) -> list[RetrievedHit]:
        scores = self.search_scores(query, top_k=top_k, config=config)
        return [
            RetrievedHit(
                chunk=scored.chunk,
                score=scored.score,
                source="colbert",
                rank=scored.rank,
                final_score=scored.score,
                source_scores={"colbert": scored.score},
                raw_scores={"colbert": scored.raw_score},
            )
            for scored in scores
        ]

    def search_scores(
        self,
        query: str,
        top_k: int = 10,
        config: RetrievalConfig | None = None,
    ) -> list[RankedScore]:
        """Return ColBERT late-interaction scores with corpus indices."""

        if not self._doc_embeddings or not self.chunks:
            return []

        query_embeddings = self._encode_token_embeddings([query], max_length=self.query_max_length)
        if not query_embeddings or query_embeddings[0].size == 0:
            return []

        query_matrix = query_embeddings[0]
        candidates: list[tuple[int, float]] = []
        for idx, doc_matrix in enumerate(self._doc_embeddings):
            if doc_matrix.size == 0:
                continue
            chunk = self.chunks[idx]
            if not chunk_matches_config(chunk, config):
                continue
            raw_score = self._late_interaction_score(query_matrix, doc_matrix)
            candidates.append((idx, raw_score))

        if not candidates:
            return []

        min_score = min(score for _, score in candidates)
        max_score = max(score for _, score in candidates)
        ranked = sorted(candidates, key=lambda item: item[1], reverse=True)[:top_k]
        return [
            RankedScore(
                index=idx,
                chunk=self.chunks[idx],
                score=_minmax(raw_score, min_score=min_score, max_score=max_score),
                raw_score=float(raw_score),
                rank=rank,
                source="colbert",
            )
            for rank, (idx, raw_score) in enumerate(ranked, start=1)
        ]

    def save(self, output_dir: str | Path) -> None:
        if not self._doc_embeddings:
            raise RuntimeError("ColBERTRetriever index is not built")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        arrays = {f"doc_{idx}": embedding for idx, embedding in enumerate(self._doc_embeddings)}
        np.savez_compressed(output_path / "colbert_embeddings.npz", **arrays)
        (output_path / "colbert_config.json").write_text(
            json.dumps(
                {
                    "model_name": self.model_name,
                    "device": self.device,
                    "batch_size": self.batch_size,
                    "query_max_length": self.query_max_length,
                    "doc_max_length": self.doc_max_length,
                    "chunk_count": len(self.chunks),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, input_dir: str | Path, chunks: Iterable[Any]) -> "ColBERTRetriever":
        input_path = Path(input_dir)
        config = json.loads((input_path / "colbert_config.json").read_text(encoding="utf-8"))
        retriever = cls(
            chunks,
            model_name=config.get("model_name", DEFAULT_COLBERT_MODEL_NAME),
            device=config.get("device"),
            batch_size=int(config.get("batch_size", 8)),
            query_max_length=int(config.get("query_max_length", 48)),
            doc_max_length=int(config.get("doc_max_length", 256)),
        )
        data = np.load(input_path / "colbert_embeddings.npz")
        retriever._doc_embeddings = [data[f"doc_{idx}"] for idx in range(len(retriever.chunks))]
        return retriever

    def _encode_token_embeddings(self, texts: list[str], *, max_length: int) -> list[np.ndarray]:
        import torch

        tokenizer, model = self._get_tokenizer_model()
        outputs: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                model_output = model(**encoded)

            hidden = model_output.last_hidden_state
            mask = encoded["attention_mask"].bool()
            mask = self._remove_special_tokens(encoded["input_ids"], mask)
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=-1)
            for row_idx in range(hidden.shape[0]):
                token_embeddings = hidden[row_idx][mask[row_idx]]
                outputs.append(token_embeddings.detach().cpu().numpy().astype(np.float32))
        return outputs

    def _get_tokenizer_model(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        try:
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("Install transformers>=4.45.0 for ColBERT retrieval.") from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self._model.eval()
        return self._tokenizer, self._model

    def _remove_special_tokens(self, input_ids: Any, mask: Any) -> Any:
        tokenizer, _ = self._get_tokenizer_model()
        special_ids = set(tokenizer.all_special_ids or [])
        if not special_ids:
            return mask
        filtered = mask.clone()
        for special_id in special_ids:
            filtered = filtered & (input_ids != special_id)
        return filtered

    @staticmethod
    def _late_interaction_score(query_matrix: np.ndarray, doc_matrix: np.ndarray) -> float:
        token_scores = np.matmul(query_matrix, doc_matrix.T)
        return float(token_scores.max(axis=1).sum())

    @staticmethod
    def _default_device() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"


def _minmax(score: float, *, min_score: float, max_score: float) -> float:
    if max_score == min_score:
        return 1.0 if max_score > 0.0 else 0.0
    return max(0.0, min(1.0, (float(score) - min_score) / (max_score - min_score)))
