from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class DenseModelSpec:
    """Named dense retrieval model configuration."""

    backend: str
    model_name: str
    query_model_name: str | None = None
    passage_model_name: str | None = None


DENSE_MODEL_PRESETS: dict[str, DenseModelSpec] = {
    "minilm": DenseModelSpec(
        backend="sentence-transformers",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    ),
    "multilingual-minilm": DenseModelSpec(
        backend="sentence-transformers",
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ),
    "contriever": DenseModelSpec(
        backend="transformers",
        model_name="facebook/contriever",
    ),
    "dpr-single-nq": DenseModelSpec(
        backend="dpr",
        model_name="facebook/dpr-question_encoder-single-nq-base",
        query_model_name="facebook/dpr-question_encoder-single-nq-base",
        passage_model_name="facebook/dpr-ctx_encoder-single-nq-base",
    ),
    "dpr-multiset": DenseModelSpec(
        backend="dpr",
        model_name="facebook/dpr-question_encoder-multiset-base",
        query_model_name="facebook/dpr-question_encoder-multiset-base",
        passage_model_name="facebook/dpr-ctx_encoder-multiset-base",
    ),
}


class EmbeddingBackend(Protocol):
    """Encoder interface for dual-encoder dense retrieval."""

    model_name: str
    backend_name: str

    def encode_queries(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        ...

    def encode_passages(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        ...


def resolve_dense_model_spec(
    *,
    model_name: str | None,
    backend: str | None,
    preset: str | None,
    query_model_name: str | None = None,
    passage_model_name: str | None = None,
) -> DenseModelSpec:
    """Resolve explicit dense model options into a backend spec."""

    if preset:
        if preset not in DENSE_MODEL_PRESETS:
            choices = ", ".join(sorted(DENSE_MODEL_PRESETS))
            raise ValueError(f"Unknown dense preset {preset!r}. Available: {choices}")
        spec = DENSE_MODEL_PRESETS[preset]
        return DenseModelSpec(
            backend=backend or spec.backend,
            model_name=model_name or spec.model_name,
            query_model_name=query_model_name or spec.query_model_name,
            passage_model_name=passage_model_name or spec.passage_model_name,
        )

    resolved_backend = backend or "sentence-transformers"
    resolved_model = model_name or DENSE_MODEL_PRESETS["minilm"].model_name
    return DenseModelSpec(
        backend=resolved_backend,
        model_name=resolved_model,
        query_model_name=query_model_name,
        passage_model_name=passage_model_name,
    )


def make_embedding_backend(
    spec: DenseModelSpec,
    *,
    device: str,
    max_length: int = 256,
) -> EmbeddingBackend:
    backend = spec.backend.strip().lower()
    if backend in {"sentence-transformers", "sentence_transformers", "sbert"}:
        return SentenceTransformerBackend(model_name=spec.model_name, device=device)
    if backend in {"transformers", "contriever", "hf"}:
        return MeanPoolingTransformersBackend(
            model_name=spec.model_name,
            device=device,
            max_length=max_length,
        )
    if backend == "dpr":
        return DPREncoderBackend(
            query_model_name=spec.query_model_name or spec.model_name,
            passage_model_name=spec.passage_model_name or spec.model_name,
            device=device,
            max_length=max_length,
        )
    raise ValueError(f"Unknown dense backend: {spec.backend}")


class SentenceTransformerBackend:
    """SentenceTransformers backend for symmetric dual encoders."""

    backend_name = "sentence-transformers"

    def __init__(self, *, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any | None = None

    def encode_queries(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        return self._encode(texts, batch_size=batch_size)

    def encode_passages(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        return self._encode(texts, batch_size=batch_size)

    def _encode(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("Install sentence-transformers>=3.0.0 for dense retrieval.") from exc
        self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model


class MeanPoolingTransformersBackend:
    """Hugging Face encoder with masked mean pooling, used for Contriever."""

    backend_name = "transformers"

    def __init__(self, *, model_name: str, device: str, max_length: int) -> None:
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    def encode_queries(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        return self._encode(texts, batch_size=batch_size)

    def encode_passages(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        return self._encode(texts, batch_size=batch_size)

    def _encode(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        import torch

        tokenizer, model = self._get_tokenizer_model()
        outputs: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                model_output = model(**encoded)
            embeddings = mean_pool(
                model_output.last_hidden_state,
                encoded["attention_mask"],
            )
            outputs.append(embeddings.detach().cpu().numpy().astype(np.float32))
        return np.vstack(outputs) if outputs else np.zeros((0, 0), dtype=np.float32)

    def _get_tokenizer_model(self) -> tuple[Any, Any]:
        if self._tokenizer is not None and self._model is not None:
            return self._tokenizer, self._model
        try:
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("Install transformers>=4.45.0 for HF dense retrieval.") from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self._model.eval()
        return self._tokenizer, self._model


class DPREncoderBackend:
    """DPR backend with separate question and context encoders."""

    backend_name = "dpr"

    def __init__(
        self,
        *,
        query_model_name: str,
        passage_model_name: str,
        device: str,
        max_length: int,
    ) -> None:
        self.model_name = query_model_name
        self.query_model_name = query_model_name
        self.passage_model_name = passage_model_name
        self.device = device
        self.max_length = max_length
        self._query_tokenizer: Any | None = None
        self._query_model: Any | None = None
        self._passage_tokenizer: Any | None = None
        self._passage_model: Any | None = None

    def encode_queries(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        tokenizer, model = self._get_query_tokenizer_model()
        return self._encode(texts, tokenizer=tokenizer, model=model, batch_size=batch_size)

    def encode_passages(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        tokenizer, model = self._get_passage_tokenizer_model()
        return self._encode(texts, tokenizer=tokenizer, model=model, batch_size=batch_size)

    def _encode(self, texts: list[str], *, tokenizer: Any, model: Any, batch_size: int) -> np.ndarray:
        import torch

        outputs: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                model_output = model(**encoded)
            if hasattr(model_output, "pooler_output") and model_output.pooler_output is not None:
                embeddings = model_output.pooler_output
            else:
                embeddings = mean_pool(model_output.last_hidden_state, encoded["attention_mask"])
            outputs.append(embeddings.detach().cpu().numpy().astype(np.float32))
        return np.vstack(outputs) if outputs else np.zeros((0, 0), dtype=np.float32)

    def _get_query_tokenizer_model(self) -> tuple[Any, Any]:
        if self._query_tokenizer is not None and self._query_model is not None:
            return self._query_tokenizer, self._query_model
        self._query_tokenizer, self._query_model = self._load_auto_model(self.query_model_name)
        return self._query_tokenizer, self._query_model

    def _get_passage_tokenizer_model(self) -> tuple[Any, Any]:
        if self._passage_tokenizer is not None and self._passage_model is not None:
            return self._passage_tokenizer, self._passage_model
        self._passage_tokenizer, self._passage_model = self._load_auto_model(self.passage_model_name)
        return self._passage_tokenizer, self._passage_model

    def _load_auto_model(self, model_name: str) -> tuple[Any, Any]:
        try:
            from transformers import (
                DPRContextEncoder,
                DPRContextEncoderTokenizer,
                DPRQuestionEncoder,
                DPRQuestionEncoderTokenizer,
            )
        except Exception as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("Install transformers>=4.45.0 for DPR retrieval.") from exc
        if "ctx_encoder" in model_name:
            tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_name)
            model = DPRContextEncoder.from_pretrained(model_name).to(self.device)
        else:
            tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(model_name)
            model = DPRQuestionEncoder.from_pretrained(model_name).to(self.device)
        model.eval()
        return tokenizer, model


def mean_pool(token_embeddings: Any, attention_mask: Any) -> Any:
    """Masked mean pooling for transformer token embeddings."""

    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = (token_embeddings * mask).sum(1)
    counts = mask.sum(1).clamp(min=1e-9)
    return summed / counts
