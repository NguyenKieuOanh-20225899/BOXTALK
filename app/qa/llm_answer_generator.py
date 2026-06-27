from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from app.qa.context_builder import GroundedContext
from app.qa.grounded_llm_client import (
    BaseGroundedLLMClient,
    EvidencePacket,
    GroundedLLMRequest,
    make_grounded_llm_client,
    normalize_llm_provider_name,
)


@dataclass(frozen=True)
class GeneratedGroundedAnswer:
    answer: str
    used_evidence_ids: list[str]
    abstain: bool
    reason: str | None
    raw_response: dict[str, Any] | None = None
    latency_ms: float = 0.0
    provider: str = "unknown"
    model: str | None = None


class LLMGroundedAnswerGenerator:
    """Primary grounded answer generator backed by a configured LLM client."""

    generator_type = "llm_grounded"

    def __init__(
        self,
        *,
        client: BaseGroundedLLMClient,
        model: str | None,
        provider: str,
    ) -> None:
        self.client = client
        self.model = model
        self.provider = provider

    @classmethod
    def from_env(cls, *, required: bool = True) -> "LLMGroundedAnswerGenerator":
        provider = os.getenv("BOXTALK_LLM_PROVIDER")
        model = os.getenv("BOXTALK_LLM_MODEL")
        if required and not provider:
            raise RuntimeError(
                "BOXTALK_LLM_PROVIDER is required for the grounded LLM answer generator. "
                "Use --llm-provider dummy for local dry-runs or configure ollama/openai-compatible."
            )

        normalized_provider = normalize_llm_provider_name(provider)
        if required and normalized_provider != "dummy" and not model:
            raise RuntimeError("BOXTALK_LLM_MODEL is required for the grounded LLM answer generator.")
        if normalized_provider == "openai-compatible":
            missing = [
                name
                for name in ("BOXTALK_LLM_BASE_URL", "BOXTALK_LLM_API_KEY", "BOXTALK_LLM_MODEL")
                if not os.getenv(name)
            ]
            if required and missing:
                raise RuntimeError(
                    "Missing OpenAI-compatible LLM environment variable(s): "
                    + ", ".join(missing)
                )

        timeout_s = _float_env("BOXTALK_LLM_TIMEOUT_SECONDS", 30.0)
        client = make_grounded_llm_client(provider, timeout_s=timeout_s)
        return cls(
            client=client,
            model=model or getattr(client, "model", None),
            provider=getattr(client, "provider_name", normalized_provider),
        )

    def generate(self, *, question: str, context: GroundedContext) -> GeneratedGroundedAnswer:
        if not context.evidence:
            return GeneratedGroundedAnswer(
                answer="",
                used_evidence_ids=[],
                abstain=True,
                reason="empty_context",
                provider=self.provider,
                model=self.model,
            )

        packets = self._packets_from_context(context)
        request = GroundedLLMRequest(
            question=question,
            query_type="grounded",
            expected_answer_shape="concise_grounded_answer",
            reasoning_mode="text",
            evidence_packets=packets,
        )
        start = time.perf_counter()
        try:
            response = self.client.generate(request)
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000.0
            return GeneratedGroundedAnswer(
                answer="",
                used_evidence_ids=[],
                abstain=True,
                reason=f"llm_error: {exc}",
                raw_response={"error": str(exc)},
                latency_ms=latency_ms,
                provider=self.provider,
                model=self.model,
            )

        latency_ms = (time.perf_counter() - start) * 1000.0
        if response.decision == "insufficient_evidence":
            return GeneratedGroundedAnswer(
                answer="",
                used_evidence_ids=list(response.used_evidence_ids),
                abstain=True,
                reason=str(response.raw_response.get("reason") or "llm_reported_insufficient_evidence"),
                raw_response=dict(response.raw_response),
                latency_ms=latency_ms,
                provider=self.provider,
                model=self.model,
            )

        return GeneratedGroundedAnswer(
            answer=response.answer,
            used_evidence_ids=list(response.used_evidence_ids),
            abstain=False,
            reason=str(response.raw_response.get("reason") or "") or None,
            raw_response=dict(response.raw_response),
            latency_ms=latency_ms,
            provider=self.provider,
            model=self.model,
        )

    def _packets_from_context(self, context: GroundedContext) -> list[EvidencePacket]:
        packets: list[EvidencePacket] = []
        for item in context.evidence:
            metadata = dict(item.metadata or {})
            modality = _modality_from_context_item(item.citation_target, metadata)
            packets.append(
                EvidencePacket(
                    evidence_id=item.evidence_id,
                    chunk_id=item.chunk_id,
                    modality=modality,
                    text=item.text,
                    source_name=item.source_name,
                    doc_id=item.doc_id,
                    page=item.page,
                    section=item.section,
                    score=float(metadata.get("score") or 0.0),
                    table_text=item.text if modality == "table" else None,
                    table_rows=metadata.get("table_rows") or metadata.get("table_records") or [],
                    table_json=metadata.get("table_json") or metadata.get("table"),
                    metadata=metadata,
                )
            )
        return packets


def _modality_from_context_item(citation_target: str | None, metadata: dict[str, Any]) -> str:
    target = str(citation_target or "").lower()
    block_type = str(metadata.get("block_type") or "").lower()
    if target in {"table", "row", "cell"} or "table" in block_type or metadata.get("is_table_chunk"):
        return "table"
    if target in {"formula", "equation"} or metadata.get("formula_text") or metadata.get("equation"):
        return "formula"
    if target in {"figure", "image"} or metadata.get("caption"):
        return "figure"
    return "text"


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default
