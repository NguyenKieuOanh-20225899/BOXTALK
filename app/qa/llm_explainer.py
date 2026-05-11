from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.qa.llm_fallback import (
    OLLAMA_DEFAULT_API_KEY,
    OLLAMA_DEFAULT_BASE_URL,
    OLLAMA_DEFAULT_MODEL,
    build_evidence_packets,
    normalize_llm_provider_name,
    parse_llm_json,
)
from app.qa.schemas import EvidenceAssessment
from app.retrieval.schemas import RetrievedHit


@dataclass(slots=True)
class LLMExplanationConfig:
    """Runtime controls for the optional user-facing explanation layer."""

    enable_llm_explanation: bool = False
    min_confidence: float = 0.20
    max_evidence_packets: int = 4
    max_packet_chars: int = 1200
    request_timeout_s: float = 20.0

    @property
    def enabled(self) -> bool:
        return self.enable_llm_explanation


@dataclass(slots=True)
class LLMExplanationRequest:
    question: str
    answer: str
    query_type: str
    evidence_packets: list[dict[str, Any]]
    language_hint: str = "match_user_question"

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "final_answer": self.answer,
            "query_type": self.query_type,
            "language_hint": self.language_hint,
            "evidence_packets": self.evidence_packets,
        }


@dataclass(slots=True)
class LLMExplanationResponse:
    explanation: str
    used_evidence_ids: list[str]
    confidence: float = 0.0
    raw_response: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LLMExplanationResult:
    called: bool
    used: bool
    reason: str
    explanation: str | None = None
    provider: str = "none"
    latency_ms: float = 0.0
    confidence: float = 0.0
    used_evidence_ids: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def not_called(cls, reason: str, *, diagnostics: dict[str, Any] | None = None) -> "LLMExplanationResult":
        return cls(called=False, used=False, reason=reason, diagnostics=diagnostics or {})

    def to_trace(self) -> dict[str, Any]:
        return {
            "called": self.called,
            "used": self.used,
            "reason": self.reason,
            "explanation": self.explanation,
            "provider": self.provider,
            "latency_ms": self.latency_ms,
            "confidence": self.confidence,
            "used_evidence_ids": list(self.used_evidence_ids),
            "diagnostics": dict(self.diagnostics),
        }


class ExplanationLLMClient(Protocol):
    provider_name: str

    def explain(self, request: LLMExplanationRequest) -> LLMExplanationResponse:
        ...


class DummyExplanationLLMClient:
    """Deterministic explainer used by tests and offline smoke checks."""

    provider_name = "dummy"

    def explain(self, request: LLMExplanationRequest) -> LLMExplanationResponse:
        if not request.answer.strip() or not request.evidence_packets:
            return LLMExplanationResponse(explanation="", used_evidence_ids=[], confidence=0.0)
        evidence_id = str(request.evidence_packets[0].get("evidence_id") or "E1")
        explanation = (
            "In short: the answer is drawn directly from the cited document evidence, "
            f"because evidence {evidence_id} contains support for the answer."
        )
        return LLMExplanationResponse(
            explanation=explanation,
            used_evidence_ids=[evidence_id],
            confidence=0.50,
            raw_response={"provider": "dummy"},
        )


class OpenAICompatibleExplanationLLMClient:
    """OpenAI-compatible chat-completions client for explanation-only output."""

    provider_name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        provider_name: str | None = None,
        timeout_s: float = 20.0,
    ) -> None:
        if provider_name is not None:
            self.provider_name = provider_name
        self.base_url = (base_url or os.getenv("BOXTALK_LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("BOXTALK_LLM_API_KEY")
        self.model = model or os.getenv("BOXTALK_LLM_MODEL") or "gpt-4o-mini"
        self.timeout_s = timeout_s

    def explain(self, request: LLMExplanationRequest) -> LLMExplanationResponse:
        messages = build_explanation_messages(request)
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": 0,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        http_request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            method="POST",
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_s) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM explanation provider HTTP {exc.code}: {body[:500]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM explanation provider request failed: {exc}") from exc

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("LLM explanation provider returned no choices")
        message = choices[0].get("message") or {}
        content = str(message.get("content") or "")
        parsed = parse_llm_json(content)
        return explanation_response_from_payload(parsed)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


class GroundedLLMExplainer:
    """Optional final-step explainer. It never changes the selected answer."""

    def __init__(self, *, config: LLMExplanationConfig, client: ExplanationLLMClient | None = None) -> None:
        self.config = config
        self.client = client or DummyExplanationLLMClient()

    def maybe_explain(
        self,
        *,
        question: str,
        query_type: str,
        answer: str,
        hits: list[RetrievedHit],
        evidence: EvidenceAssessment,
        citations: list[dict[str, Any]],
        grounded: bool,
    ) -> LLMExplanationResult:
        if not self.config.enabled:
            return LLMExplanationResult.not_called("llm_explanation is off")
        if not answer.strip():
            return LLMExplanationResult.not_called("empty_answer")
        if evidence.decision != "answer":
            return LLMExplanationResult.not_called("evidence_decision_is_not_answer")
        if not grounded or not citations:
            return LLMExplanationResult.not_called("answer_is_not_grounded_or_has_no_citations")

        packets = build_evidence_packets(
            hits,
            max_packets=self.config.max_evidence_packets,
            max_chars=self.config.max_packet_chars,
        )
        cited_chunk_ids = {str(citation.get("chunk_id")) for citation in citations if citation.get("chunk_id")}
        if cited_chunk_ids:
            packets = [packet for packet in packets if packet.chunk_id in cited_chunk_ids]
        if not packets:
            return LLMExplanationResult.not_called("no_cited_evidence_packets")

        request = LLMExplanationRequest(
            question=question,
            answer=answer,
            query_type=query_type,
            evidence_packets=[packet.prompt_dict() for packet in packets],
        )
        start = time.perf_counter()
        try:
            response = self.client.explain(request)
        except Exception as exc:  # pragma: no cover - provider failures are environment specific
            latency_ms = (time.perf_counter() - start) * 1000.0
            return LLMExplanationResult(
                called=True,
                used=False,
                reason="llm_explanation_provider_error",
                provider=getattr(self.client, "provider_name", self.client.__class__.__name__),
                latency_ms=latency_ms,
                diagnostics={"error": str(exc)},
            )

        latency_ms = (time.perf_counter() - start) * 1000.0
        known_ids = {str(packet.evidence_id) for packet in packets}
        used_ids = [evidence_id for evidence_id in response.used_evidence_ids if evidence_id in known_ids]
        if not response.explanation.strip():
            return LLMExplanationResult(
                called=True,
                used=False,
                reason="empty_llm_explanation",
                provider=getattr(self.client, "provider_name", self.client.__class__.__name__),
                latency_ms=latency_ms,
                confidence=response.confidence,
                diagnostics={"raw_response": response.raw_response},
            )
        if response.confidence < self.config.min_confidence:
            return LLMExplanationResult(
                called=True,
                used=False,
                reason="llm_explanation_confidence_too_low",
                provider=getattr(self.client, "provider_name", self.client.__class__.__name__),
                latency_ms=latency_ms,
                confidence=response.confidence,
                used_evidence_ids=used_ids,
                diagnostics={"raw_response": response.raw_response},
            )

        return LLMExplanationResult(
            called=True,
            used=True,
            reason="llm_explanation_generated",
            explanation=response.explanation.strip(),
            provider=getattr(self.client, "provider_name", self.client.__class__.__name__),
            latency_ms=latency_ms,
            confidence=response.confidence,
            used_evidence_ids=used_ids or response.used_evidence_ids,
            diagnostics={"raw_response": response.raw_response},
        )


def build_explanation_messages(request: LLMExplanationRequest) -> list[dict[str, str]]:
    system = (
        "You explain a grounded PDF QA answer to an end user. Do not change the answer. "
        "Use only the final answer and cited evidence packets. Do not add outside knowledge. "
        "Match the user's language. Keep it clear, short, and easy to understand. Return JSON only."
    )
    output_schema = {
        "explanation": "2-4 short sentences or bullets explaining why the answer follows from the cited evidence",
        "used_evidence_ids": ["E1"],
        "confidence": "0.0-1.0",
    }
    user = {
        "task": "Explain a grounded answer without changing it",
        "instructions": [
            "Do not introduce facts that are not in the final_answer or evidence_packets.",
            "If the evidence does not support a useful explanation, return an empty explanation and confidence 0.",
            "Use plain language for a non-technical user.",
            "When the user asks in Vietnamese, explain in Vietnamese.",
        ],
        "expected_output_json": output_schema,
        "input": request.prompt_payload(),
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def explanation_response_from_payload(payload: dict[str, Any]) -> LLMExplanationResponse:
    used_raw = payload.get("used_evidence_ids") or []
    if isinstance(used_raw, str):
        used_ids = [used_raw]
    else:
        used_ids = [str(item) for item in used_raw if str(item).strip()]
    explanation_raw = payload.get("explanation") or ""
    if isinstance(explanation_raw, list):
        explanation = "\n".join(str(item).strip() for item in explanation_raw if str(item).strip())
    else:
        explanation = str(explanation_raw).strip()
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return LLMExplanationResponse(
        explanation=explanation,
        used_evidence_ids=used_ids,
        confidence=confidence,
        raw_response=dict(payload),
    )


def make_llm_explanation_client(
    provider: str | None = None,
    *,
    timeout_s: float = 20.0,
) -> ExplanationLLMClient:
    selected_provider = provider or os.getenv("BOXTALK_LLM_EXPLANATION_PROVIDER") or os.getenv("BOXTALK_LLM_PROVIDER") or "ollama"
    normalized_provider = normalize_llm_provider_name(selected_provider)
    if normalized_provider == "openai-compatible":
        return OpenAICompatibleExplanationLLMClient(timeout_s=timeout_s)
    if normalized_provider == "ollama":
        return OpenAICompatibleExplanationLLMClient(
            base_url=os.getenv("BOXTALK_LLM_BASE_URL") or OLLAMA_DEFAULT_BASE_URL,
            api_key=os.getenv("BOXTALK_LLM_API_KEY") or OLLAMA_DEFAULT_API_KEY,
            model=os.getenv("BOXTALK_LLM_MODEL") or OLLAMA_DEFAULT_MODEL,
            provider_name="ollama",
            timeout_s=timeout_s,
        )
    return DummyExplanationLLMClient()


def make_llm_explainer_from_env(*, enabled: bool | None = None, provider: str | None = None) -> GroundedLLMExplainer | None:
    is_enabled = _bool_env("BOXTALK_ENABLE_LLM_EXPLANATION", False) if enabled is None else enabled
    if not is_enabled:
        return None
    config = LLMExplanationConfig(
        enable_llm_explanation=True,
        min_confidence=float(os.getenv("BOXTALK_LLM_EXPLANATION_MIN_CONFIDENCE", "0.20")),
        max_evidence_packets=int(os.getenv("BOXTALK_LLM_EXPLANATION_MAX_PACKETS", "4")),
        max_packet_chars=int(os.getenv("BOXTALK_LLM_EXPLANATION_MAX_PACKET_CHARS", "1200")),
        request_timeout_s=float(os.getenv("BOXTALK_LLM_EXPLANATION_TIMEOUT_SECONDS", "20")),
    )
    client = make_llm_explanation_client(provider, timeout_s=config.request_timeout_s)
    return GroundedLLMExplainer(config=config, client=client)


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
