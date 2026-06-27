from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from app.qa.grounded_llm_client import (
    BaseGroundedLLMClient,
    DummyGroundedLLMClient,
    EvidencePacket,
    GroundedLLMRequest,
    GroundedLLMResponse,
    OLLAMA_DEFAULT_API_KEY,
    OLLAMA_DEFAULT_BASE_URL,
    OLLAMA_DEFAULT_MODEL,
    OpenAICompatibleGroundedLLMClient,
    ReasoningMode,
    build_grounded_messages,
    make_grounded_llm_client,
    normalize_llm_provider_name,
    parse_llm_json,
    provider_runtime_info,
    response_from_payload,
)
from app.qa.schemas import EvidenceAssessment
from app.qa.text_utils import normalize_text
from app.retrieval.schemas import RetrievedHit


FallbackDecision = Literal["answer", "insufficient_evidence", "not_called", "error"]


@dataclass(slots=True)
class LLMFallbackConfig:
    """Compatibility config for the legacy optional fallback path."""

    enable_llm_fallback: bool = False
    enable_table_llm_reasoning: bool = False
    enable_formula_llm_reasoning: bool = False
    enable_figure_llm_reasoning: bool = False
    fallback_only_if_grounded_evidence_present: bool = True
    min_evidence_relevance: float = 0.30
    sufficiency_threshold: float = 0.72
    min_llm_confidence: float = 0.30
    min_non_answer_override_confidence: float = 0.65
    max_evidence_packets: int = 6
    max_packet_chars: int = 1800
    request_timeout_s: float = 30.0

    @property
    def enabled(self) -> bool:
        return self.enable_llm_fallback


@dataclass(slots=True)
class LLMFallbackResult:
    called: bool
    used: bool
    decision: FallbackDecision
    reason: str
    answer: str | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)
    support_sentences: list[str] = field(default_factory=list)
    reasoning_mode: ReasoningMode = "text"
    confidence: float = 0.0
    used_evidence_ids: list[str] = field(default_factory=list)
    provider: str = "none"
    latency_ms: float = 0.0
    llm_called: bool = False
    final_answer_source: str = "standard"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def not_called(cls, reason: str, *, diagnostics: dict[str, Any] | None = None) -> "LLMFallbackResult":
        return cls(called=False, used=False, decision="not_called", reason=reason, diagnostics=diagnostics or {})

    def to_trace(self) -> dict[str, Any]:
        return {
            "called": self.called,
            "fallback_called": self.called,
            "used": self.used,
            "fallback_used": self.used,
            "decision": self.decision,
            "reason": self.reason,
            "fallback_reason": self.reason,
            "answer": self.answer,
            "reasoning_mode": self.reasoning_mode,
            "confidence": self.confidence,
            "override_confidence": self.confidence,
            "used_evidence_ids": list(self.used_evidence_ids),
            "provider": self.provider,
            "provider_name": self.provider,
            "latency_ms": self.latency_ms,
            "llm_called": self.llm_called,
            "final_answer_source": self.final_answer_source,
            "diagnostics": dict(self.diagnostics),
        }


class GroundedLLMFallback:
    """Legacy wrapper kept for old benchmark/adaptive paths.

    The primary QA path uses LLMGroundedAnswerGenerator directly. This wrapper now
    only performs a simple grounded LLM call over retrieved evidence.
    """

    def __init__(self, *, config: LLMFallbackConfig, client: BaseGroundedLLMClient | None = None) -> None:
        self.config = config
        self.client = client or DummyGroundedLLMClient()

    def maybe_generate(
        self,
        *,
        question: str,
        query_type: str,
        hits: list[RetrievedHit],
        evidence: EvidenceAssessment,
        standard_answer: Any,
    ) -> LLMFallbackResult:
        _ = standard_answer
        if not self.config.enabled:
            return LLMFallbackResult.not_called("llm_fallback is off")

        packets = build_evidence_packets(
            hits,
            max_packets=self.config.max_evidence_packets,
            max_chars=self.config.max_packet_chars,
        )
        if not packets:
            return LLMFallbackResult.not_called("no evidence packets")
        if self.config.fallback_only_if_grounded_evidence_present and evidence.relevance < self.config.min_evidence_relevance:
            return LLMFallbackResult.not_called("retrieved evidence is below fallback relevance threshold")

        request = GroundedLLMRequest(
            question=question,
            query_type=query_type,
            expected_answer_shape="concise_grounded_answer",
            reasoning_mode=_reasoning_mode(question, packets),
            evidence_packets=packets,
        )
        start = time.perf_counter()
        try:
            response = self.client.generate(request)
        except Exception as exc:  # pragma: no cover - provider failures depend on runtime.
            return LLMFallbackResult(
                called=True,
                used=False,
                decision="error",
                reason="llm_provider_error",
                provider=getattr(self.client, "provider_name", self.client.__class__.__name__),
                latency_ms=(time.perf_counter() - start) * 1000.0,
                llm_called=True,
                diagnostics={"error": str(exc)},
            )

        latency_ms = (time.perf_counter() - start) * 1000.0
        if response.decision == "insufficient_evidence" or not response.answer.strip():
            return LLMFallbackResult(
                called=True,
                used=False,
                decision="insufficient_evidence",
                reason=str(response.raw_response.get("reason") or "llm_reported_insufficient_evidence"),
                reasoning_mode=response.reasoning_mode,
                confidence=response.confidence,
                used_evidence_ids=response.used_evidence_ids,
                provider=getattr(self.client, "provider_name", self.client.__class__.__name__),
                latency_ms=latency_ms,
                llm_called=True,
                diagnostics={"raw_response": response.raw_response},
            )
        if response.confidence < self.config.min_llm_confidence:
            return LLMFallbackResult(
                called=True,
                used=False,
                decision="insufficient_evidence",
                reason="llm_confidence_too_low",
                reasoning_mode=response.reasoning_mode,
                confidence=response.confidence,
                used_evidence_ids=response.used_evidence_ids,
                provider=getattr(self.client, "provider_name", self.client.__class__.__name__),
                latency_ms=latency_ms,
                llm_called=True,
                diagnostics={"raw_response": response.raw_response},
            )

        packet_by_id = {packet.evidence_id: packet for packet in packets}
        used_packets = [packet_by_id[eid] for eid in response.used_evidence_ids if eid in packet_by_id]
        return LLMFallbackResult(
            called=True,
            used=True,
            decision="answer",
            reason="llm_fallback_generated",
            answer=normalize_text(response.answer),
            citations=[citation_from_packet(packet) for packet in used_packets],
            support_sentences=[packet.text[:320] for packet in used_packets],
            reasoning_mode=response.reasoning_mode,
            confidence=response.confidence,
            used_evidence_ids=response.used_evidence_ids,
            provider=getattr(self.client, "provider_name", self.client.__class__.__name__),
            latency_ms=latency_ms,
            llm_called=True,
            final_answer_source="llm_fallback",
            diagnostics={"raw_response": response.raw_response},
        )


def build_evidence_packets(
    hits: list[RetrievedHit],
    *,
    max_packets: int = 6,
    max_chars: int = 1800,
) -> list[EvidencePacket]:
    packets: list[EvidencePacket] = []
    for hit in hits[:max_packets]:
        metadata = {**dict(hit.chunk.metadata or {}), **dict(hit.metadata or {})}
        raw_text = str(hit.text or hit.chunk.text or "")
        modality = _modality_from_hit(hit, metadata, raw_text)
        packets.append(
            EvidencePacket(
                evidence_id=f"E{len(packets) + 1}",
                chunk_id=hit.chunk_id,
                modality=modality,
                text=_compact_text(raw_text, max_chars, preserve_newlines=modality == "table"),
                source_name=hit.chunk.source_name,
                doc_id=hit.chunk.doc_id,
                page=hit.page,
                section=hit.section,
                heading_path=_coerce_str_list(metadata.get("heading_path")),
                score=float(hit.final_score or hit.score or 0.0),
                table_text=_compact_text(raw_text, max_chars, preserve_newlines=True) if modality == "table" else None,
                table_rows=_coerce_list(metadata.get("table_rows") or metadata.get("table_records")),
                table_json=metadata.get("table_json") or metadata.get("table"),
                formula_text=str(metadata.get("formula_text") or metadata.get("equation") or "") or None,
                caption=str(metadata.get("caption") or "") or None,
                metadata=metadata,
            )
        )
    return packets


def citation_from_packet(packet: EvidencePacket) -> dict[str, Any]:
    return {
        "chunk_id": packet.chunk_id,
        "doc_id": packet.doc_id,
        "source_name": packet.source_name,
        "page": packet.page,
        "section": packet.section,
        "heading_path": list(packet.heading_path),
        "score": packet.score,
        "evidence_id": packet.evidence_id,
        "modality": packet.modality,
    }


def make_llm_fallback_from_env(*, enabled: bool | None = None, provider: str | None = None) -> GroundedLLMFallback | None:
    is_enabled = _bool_env("BOXTALK_ENABLE_LLM_FALLBACK", False) if enabled is None else enabled
    if not is_enabled:
        return None
    config = LLMFallbackConfig(
        enable_llm_fallback=True,
        fallback_only_if_grounded_evidence_present=_bool_env("BOXTALK_LLM_FALLBACK_REQUIRES_EVIDENCE", True),
        min_evidence_relevance=float(os.getenv("BOXTALK_LLM_FALLBACK_MIN_RELEVANCE", "0.30")),
        min_llm_confidence=float(os.getenv("BOXTALK_LLM_FALLBACK_MIN_CONFIDENCE", "0.30")),
        max_evidence_packets=int(os.getenv("BOXTALK_LLM_FALLBACK_MAX_PACKETS", "6")),
        max_packet_chars=int(os.getenv("BOXTALK_LLM_FALLBACK_MAX_PACKET_CHARS", "1800")),
        request_timeout_s=float(os.getenv("BOXTALK_LLM_TIMEOUT_SECONDS", "30")),
    )
    return GroundedLLMFallback(
        config=config,
        client=make_grounded_llm_client(provider, timeout_s=config.request_timeout_s),
    )


def _reasoning_mode(question: str, packets: list[EvidencePacket]) -> ReasoningMode:
    folded = _fold(question)
    if any(packet.modality == "table" for packet in packets) and any(
        term in folded for term in ("bang", "table", "cot", "hang", "o", "tuong ung", "quy doi")
    ):
        return "table"
    return "text"


def _modality_from_hit(hit: RetrievedHit, metadata: dict[str, Any], text: str) -> ReasoningMode:
    target = str(metadata.get("citation_target") or "").lower()
    block_type = str(hit.chunk.block_type or metadata.get("block_type") or "").lower()
    if target in {"table", "row", "cell"} or "table" in block_type or metadata.get("table_json") or _looks_like_table(text):
        return "table"
    if target in {"formula", "equation"} or metadata.get("formula_text") or metadata.get("equation"):
        return "formula"
    if target in {"figure", "image"} or metadata.get("caption"):
        return "figure"
    return "text"


def _compact_text(text: str, max_chars: int, *, preserve_newlines: bool = False) -> str:
    if preserve_newlines:
        compact = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines() if line.strip())
    else:
        compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "..."


def _looks_like_table(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    pipe_lines = [line for line in lines if line.startswith("|") and line.endswith("|") and line.count("|") >= 2]
    return len(pipe_lines) >= 2


def _coerce_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _coerce_str_list(value: Any) -> list[str]:
    return [str(item) for item in _coerce_list(value) if str(item).strip()]


def _fold(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_text(text)).strip().lower()


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}
