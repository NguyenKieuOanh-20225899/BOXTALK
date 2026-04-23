from __future__ import annotations

import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

from app.qa.schemas import EvidenceAssessment
from app.qa.text_utils import normalize_text, split_sentences, token_set
from app.retrieval.schemas import RetrievedHit


FallbackDecision = Literal["answer", "insufficient_evidence", "not_called", "error"]
ReasoningMode = Literal["text", "table", "formula", "figure", "multi_span"]

NUMBER_RE = re.compile(r"\b\d+(?:[,.]\d+)?\b")
NUMBER_UNIT_RE = re.compile(
    r"\b\d+(?:[,.]\d+)?\s*(?:%|years?|months?|weeks?|days?|hours?|credits?|points?|scores?|"
    r"layers?|heads?|dimensions?|parameters?|tokens?|samples?|steps?|epochs?|gpa|bleu|f1|"
    r"nam|thang|tuan|ngay|gio|tin chi|diem|phan tram)\b",
    re.I,
)
GRADE_RE = re.compile(r"(?<!\w)([A-F][+-]?)(?!\w)", re.I)
MATH_RE = re.compile(
    r"(?:\b[A-Za-z]\w*\([^)]*\)\s*=|[A-Za-z_]\w*\s*(?:=|≈|<=|>=)\s*-?\d+(?:[,.]\d+)?|"
    r"\b[A-Za-z_]\w*\s*=\s*[A-Za-z_]\w*(?:\s*[+\-*/^]\s*[A-Za-z_]\w*)+)"
)
RANGE_RE = re.compile(
    r"(?P<low>\d+(?:[,.]\d+)?)\s*(?:-|--|to|den|toi|through|<=|<)\s*(?P<high>\d+(?:[,.]\d+)?)",
    re.I,
)
LOWER_BOUND_RE = re.compile(r"(?:>=|from|tu|at least|min(?:imum)?)\s*(?P<low>\d+(?:[,.]\d+)?)", re.I)
UPPER_BOUND_RE = re.compile(r"(?:<=|under|below|duoi|max(?:imum)?|khong qua)\s*(?P<high>\d+(?:[,.]\d+)?)", re.I)


@dataclass(slots=True)
class LLMFallbackConfig:
    """Runtime controls for the grounded LLM fallback layer."""

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
class EvidencePacket:
    evidence_id: str
    chunk_id: str
    modality: ReasoningMode
    text: str
    source_name: str | None = None
    doc_id: str | None = None
    page: int | None = None
    section: str | None = None
    heading_path: list[str] = field(default_factory=list)
    score: float = 0.0
    table_text: str | None = None
    table_rows: list[Any] = field(default_factory=list)
    table_json: Any | None = None
    formula_text: str | None = None
    caption: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def prompt_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["metadata"] = _prompt_safe_metadata(self.metadata)
        return payload


@dataclass(slots=True)
class GroundedLLMRequest:
    question: str
    query_type: str
    expected_answer_shape: str
    reasoning_mode: ReasoningMode
    evidence_packets: list[EvidencePacket]

    def prompt_payload(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "query_type": self.query_type,
            "expected_answer_shape": self.expected_answer_shape,
            "reasoning_mode": self.reasoning_mode,
            "evidence_packets": [packet.prompt_dict() for packet in self.evidence_packets],
        }


@dataclass(slots=True)
class GroundedLLMResponse:
    decision: Literal["answer", "insufficient_evidence"]
    answer: str
    used_evidence_ids: list[str]
    reasoning_mode: ReasoningMode = "text"
    confidence: float = 0.0
    raw_response: dict[str, Any] = field(default_factory=dict)


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
        return cls(
            called=False,
            used=False,
            decision="not_called",
            reason=reason,
            diagnostics=diagnostics or {},
        )

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


class BaseGroundedLLMClient(Protocol):
    provider_name: str

    def generate(self, request: GroundedLLMRequest) -> GroundedLLMResponse:
        ...


class DummyGroundedLLMClient:
    """Deterministic local client used for tests and offline benchmarks.

    It never uses outside knowledge. It picks concise evidence-backed spans and
    returns insufficient evidence when no packet contains an answer-shaped span.
    """

    provider_name = "dummy"

    def generate(self, request: GroundedLLMRequest) -> GroundedLLMResponse:
        if not request.evidence_packets:
            return GroundedLLMResponse(
                decision="insufficient_evidence",
                answer="",
                used_evidence_ids=[],
                reasoning_mode=request.reasoning_mode,
                confidence=0.0,
            )

        question_folded = _fold_text(request.question)
        if request.expected_answer_shape in {"numeric", "table", "formula"}:
            numeric = _best_numeric_span(question_folded, request.evidence_packets)
            if numeric is not None:
                packet, span = numeric
                return GroundedLLMResponse(
                    decision="answer",
                    answer=span,
                    used_evidence_ids=[packet.evidence_id],
                    reasoning_mode=request.reasoning_mode,
                    confidence=0.25,
                )

        if request.reasoning_mode in {"multi_span", "figure", "text"}:
            spans: list[str] = []
            used: list[str] = []
            for packet in request.evidence_packets[:3]:
                sentence = _first_relevant_sentence(question_folded, packet.text)
                if sentence:
                    spans.append(sentence)
                    used.append(packet.evidence_id)
            if spans:
                if request.reasoning_mode == "multi_span":
                    answer = "\n".join(f"- {span}" for span in spans[:3])
                else:
                    answer = spans[0]
                return GroundedLLMResponse(
                    decision="answer",
                    answer=answer,
                    used_evidence_ids=used,
                    reasoning_mode=request.reasoning_mode,
                    confidence=0.25,
                )

        return GroundedLLMResponse(
            decision="insufficient_evidence",
            answer="",
            used_evidence_ids=[],
            reasoning_mode=request.reasoning_mode,
            confidence=0.0,
        )


class OpenAICompatibleGroundedLLMClient:
    """Small OpenAI-compatible chat-completions client using only stdlib."""

    provider_name = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("BOXTALK_LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key if api_key is not None else os.getenv("BOXTALK_LLM_API_KEY")
        self.model = model or os.getenv("BOXTALK_LLM_MODEL") or "gpt-4o-mini"
        self.timeout_s = timeout_s

    def generate(self, request: GroundedLLMRequest) -> GroundedLLMResponse:
        messages = build_grounded_messages(request)
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
            raise RuntimeError(f"LLM provider HTTP {exc.code}: {body[:500]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM provider request failed: {exc}") from exc

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("LLM provider returned no choices")
        message = choices[0].get("message") or {}
        content = str(message.get("content") or "")
        parsed = parse_llm_json(content)
        return response_from_payload(parsed, request.reasoning_mode)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


class GroundedLLMFallback:
    """Controlled fallback layer that can replace weak standard synthesis."""

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
        packets = build_evidence_packets(hits, max_packets=self.config.max_evidence_packets, max_chars=self.config.max_packet_chars)
        should_call, reasons, diagnostics = should_call_llm_fallback(
            question=question,
            query_type=query_type,
            evidence=evidence,
            packets=packets,
            standard_answer_text=str(getattr(standard_answer, "answer", "") or ""),
            standard_grounded=bool(getattr(standard_answer, "grounded", False)),
            standard_citations=list(getattr(standard_answer, "citations", []) or []),
            config=self.config,
        )
        if not should_call:
            return LLMFallbackResult.not_called("; ".join(reasons) or "fallback policy did not trigger", diagnostics=diagnostics)

        expected_shape = expected_answer_shape(question, query_type, packets)
        reasoning_mode = choose_reasoning_mode(question, query_type, packets)
        if reasoning_mode == "table" and self.config.enable_table_llm_reasoning:
            table_answer = try_rule_based_table_lookup(question, packets)
            if table_answer is not None:
                packet, answer = table_answer
                citation = citation_from_packet(packet)
                return LLMFallbackResult(
                    called=True,
                    used=True,
                    decision="answer",
                    reason="rule_based_table_lookup",
                    answer=answer,
                    citations=[citation],
                    support_sentences=[answer],
                    reasoning_mode="table",
                    confidence=0.70,
                    used_evidence_ids=[packet.evidence_id],
                    provider="rule_based_table_lookup",
                    latency_ms=0.0,
                    llm_called=False,
                    final_answer_source="table_rule_fallback",
                    diagnostics=diagnostics,
                )

        request = GroundedLLMRequest(
            question=question,
            query_type=query_type,
            expected_answer_shape=expected_shape,
            reasoning_mode=reasoning_mode,
            evidence_packets=packets,
        )
        start = time.perf_counter()
        try:
            response = self.client.generate(request)
        except Exception as exc:  # pragma: no cover - provider failures are environment specific
            latency_ms = (time.perf_counter() - start) * 1000.0
            return LLMFallbackResult(
                called=True,
                used=False,
                decision="error",
                reason="llm_provider_error",
                reasoning_mode=reasoning_mode,
                provider=getattr(self.client, "provider_name", self.client.__class__.__name__),
                latency_ms=latency_ms,
                llm_called=True,
                diagnostics={**diagnostics, "error": str(exc)},
            )

        latency_ms = (time.perf_counter() - start) * 1000.0
        validation_error = validate_llm_response(response, request, self.config)
        if validation_error:
            return LLMFallbackResult(
                called=True,
                used=False,
                decision="insufficient_evidence",
                reason=validation_error,
                reasoning_mode=reasoning_mode,
                confidence=response.confidence,
                used_evidence_ids=response.used_evidence_ids,
                provider=getattr(self.client, "provider_name", self.client.__class__.__name__),
                latency_ms=latency_ms,
                llm_called=True,
                diagnostics={**diagnostics, "raw_response": response.raw_response},
            )

        if response.decision == "insufficient_evidence":
            return LLMFallbackResult(
                called=True,
                used=False,
                decision="insufficient_evidence",
                reason="llm_reported_insufficient_evidence",
                reasoning_mode=response.reasoning_mode,
                confidence=response.confidence,
                provider=getattr(self.client, "provider_name", self.client.__class__.__name__),
                latency_ms=latency_ms,
                llm_called=True,
                diagnostics={**diagnostics, "raw_response": response.raw_response},
            )

        if evidence.decision != "answer" and response.confidence < self.config.min_non_answer_override_confidence:
            return LLMFallbackResult(
                called=True,
                used=False,
                decision="insufficient_evidence",
                reason="llm_confidence_too_low_to_override_non_answer_evidence",
                reasoning_mode=response.reasoning_mode,
                confidence=response.confidence,
                used_evidence_ids=response.used_evidence_ids,
                provider=getattr(self.client, "provider_name", self.client.__class__.__name__),
                latency_ms=latency_ms,
                llm_called=True,
                diagnostics={**diagnostics, "raw_response": response.raw_response},
            )

        packet_by_id = {packet.evidence_id: packet for packet in packets}
        used_packets = [packet_by_id[eid] for eid in response.used_evidence_ids if eid in packet_by_id]
        citations = [citation_from_packet(packet) for packet in used_packets]
        return LLMFallbackResult(
            called=True,
            used=True,
            decision="answer",
            reason="; ".join(reasons) or "llm_fallback_policy_triggered",
            answer=normalize_text(response.answer),
            citations=citations,
            support_sentences=[_support_snippet(packet, response.answer) for packet in used_packets],
            reasoning_mode=response.reasoning_mode,
            confidence=response.confidence,
            used_evidence_ids=response.used_evidence_ids,
            provider=getattr(self.client, "provider_name", self.client.__class__.__name__),
            latency_ms=latency_ms,
            llm_called=True,
            final_answer_source="llm_fallback",
            diagnostics={**diagnostics, "raw_response": response.raw_response},
        )


def build_grounded_messages(request: GroundedLLMRequest) -> list[dict[str, str]]:
    system = (
        "You are a grounded answer synthesizer. Answer only from the provided evidence packets. "
        "Do not use outside knowledge. If the evidence is insufficient, return insufficient_evidence. "
        "Cite the evidence IDs that directly support the answer. For table questions, identify the row, "
        "column, cell, or interval used. For formula questions, rely only on the formula text and surrounding "
        "explanation. For figure questions, rely only on captions and nearby text. Return JSON only."
    )
    output_schema = {
        "decision": "answer | insufficient_evidence",
        "answer": "concise grounded answer, empty when insufficient",
        "used_evidence_ids": ["E1"],
        "reasoning_mode": "text | table | formula | figure | multi_span",
        "confidence": "0.0-1.0",
    }
    user = {
        "task": "Grounded QA fallback synthesis",
        "instructions": [
            "Use only the supplied evidence_packets.",
            "If a claim cannot be tied to at least one evidence_id, do not include it.",
            "If the evidence is partial but sufficient for a concise answer, answer and cite every supporting packet.",
            "If figure/image evidence lacks caption or nearby text, return insufficient_evidence.",
            "Keep the answer concise and avoid unsupported background explanation.",
        ],
        "expected_output_json": output_schema,
        "input": request.prompt_payload(),
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def parse_llm_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object")
    return payload


def response_from_payload(payload: dict[str, Any], default_mode: ReasoningMode) -> GroundedLLMResponse:
    decision = str(payload.get("decision") or "").strip().lower()
    if decision not in {"answer", "insufficient_evidence"}:
        decision = "insufficient_evidence"
    used_raw = payload.get("used_evidence_ids") or []
    if isinstance(used_raw, str):
        used_ids = [used_raw]
    else:
        used_ids = [str(item) for item in used_raw if str(item).strip()]
    mode = str(payload.get("reasoning_mode") or default_mode).strip().lower()
    if mode not in {"text", "table", "formula", "figure", "multi_span"}:
        mode = default_mode
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return GroundedLLMResponse(
        decision=decision,  # type: ignore[arg-type]
        answer=str(payload.get("answer") or "").strip(),
        used_evidence_ids=used_ids,
        reasoning_mode=mode,  # type: ignore[arg-type]
        confidence=max(0.0, min(1.0, confidence)),
        raw_response=payload,
    )


def should_call_llm_fallback(
    *,
    question: str,
    query_type: str,
    evidence: EvidenceAssessment,
    packets: list[EvidencePacket],
    standard_answer_text: str,
    standard_grounded: bool,
    standard_citations: list[Any],
    config: LLMFallbackConfig,
) -> tuple[bool, list[str], dict[str, Any]]:
    if not config.enabled:
        return False, ["enable_llm_fallback is off"], {"enabled": False}

    has_grounded_evidence = bool(packets) and (
        evidence.relevance >= config.min_evidence_relevance
        or bool(evidence.selected_hit_ids)
        or bool(evidence.support_sentences)
    )
    diagnostics: dict[str, Any] = {
        "has_grounded_evidence": has_grounded_evidence,
        "packet_count": len(packets),
        "modalities": sorted({packet.modality for packet in packets}),
        "standard_grounded": standard_grounded,
        "standard_citation_count": len(standard_citations),
    }
    if config.fallback_only_if_grounded_evidence_present and not has_grounded_evidence:
        return False, ["no grounded evidence packets"], diagnostics

    reasons: list[str] = []
    q_folded = _fold_text(question)
    answer_folded = _fold_text(standard_answer_text)
    shape = expected_answer_shape(question, query_type, packets)
    mode = choose_reasoning_mode(question, query_type, packets)
    shape_mismatch = _standard_answer_shape_mismatch(
        question_folded=q_folded,
        answer_folded=answer_folded,
        expected_shape=shape,
    )
    standard_is_strong = (
        evidence.decision == "answer"
        and evidence.sufficiency >= config.sufficiency_threshold
        and standard_grounded
        and bool(standard_citations)
        and not shape_mismatch
    )

    diagnostics["standard_answer_shape_mismatch"] = shape_mismatch
    diagnostics["standard_answer_strong"] = standard_is_strong
    if standard_is_strong:
        return False, ["standard_answer_strong"], diagnostics

    if evidence.decision in {"expand_retrieval", "abstain"} and evidence.sufficiency >= 0.45:
        reasons.append(f"answer_status={evidence.decision}")
    if evidence.sufficiency < config.sufficiency_threshold and evidence.relevance >= config.min_evidence_relevance:
        reasons.append(f"sufficiency_below_{config.sufficiency_threshold:.2f}")
    if not standard_grounded or not standard_citations:
        reasons.append("standard_answer_not_fully_grounded")
    if shape_mismatch:
        reasons.append("standard_answer_shape_mismatch")
    if query_type in {"comparison", "procedural", "multi_hop"} or _needs_multi_span(q_folded):
        reasons.append("complex_synthesis_query")
    if mode == "table" and config.enable_table_llm_reasoning:
        reasons.append("table_evidence_or_query")
    if mode == "formula" and config.enable_formula_llm_reasoning:
        reasons.append("formula_evidence_or_query")
    if mode == "figure" and config.enable_figure_llm_reasoning:
        reasons.append("figure_textual_evidence_or_query")
    if len(evidence.selected_hit_ids) >= 2 and query_type in {"comparison", "procedural", "multi_hop"}:
        reasons.append("multi_span_evidence")

    diagnostics["expected_answer_shape"] = shape
    diagnostics["reasoning_mode"] = mode
    diagnostics["question_signals"] = {
        "numeric": _is_numeric_question(q_folded),
        "table": _is_table_question(q_folded),
        "formula": _is_formula_question(q_folded),
        "figure": _is_figure_question(q_folded),
        "multi_span": _needs_multi_span(q_folded),
    }
    return bool(reasons), reasons, diagnostics


def build_evidence_packets(
    hits: list[RetrievedHit],
    *,
    max_packets: int = 6,
    max_chars: int = 1800,
) -> list[EvidencePacket]:
    packets: list[EvidencePacket] = []
    for idx, hit in enumerate(hits[:max_packets], start=1):
        metadata = {**dict(hit.chunk.metadata or {}), **dict(hit.metadata or {})}
        modality = detect_modality(hit)
        raw_text = str(hit.chunk.text or "")
        text = _truncate(normalize_text(raw_text), max_chars)
        table_text = _truncate(raw_text.strip(), max_chars) if modality == "table" else None
        packets.append(
            EvidencePacket(
                evidence_id=f"E{idx}",
                chunk_id=hit.chunk_id,
                modality=modality,
                text=text,
                source_name=hit.chunk.source_name,
                doc_id=hit.chunk.doc_id,
                page=hit.page,
                section=hit.section,
                heading_path=list(hit.heading_path),
                score=round(float(hit.final_score or hit.score), 4),
                table_text=table_text,
                table_rows=_coerce_list(metadata.get("table_rows") or metadata.get("rows")),
                table_json=metadata.get("table_json") or metadata.get("table"),
                formula_text=_extract_formula_text(text) if modality == "formula" else None,
                caption=_extract_caption(text, metadata) if modality == "figure" else None,
                metadata=metadata,
            )
        )
    return packets


def detect_modality(hit: RetrievedHit) -> ReasoningMode:
    metadata = {**dict(hit.chunk.metadata or {}), **dict(hit.metadata or {})}
    block_type = str(hit.chunk.block_type or metadata.get("block_type") or "").lower()
    text = normalize_text(hit.chunk.text)
    folded = _fold_text(" ".join([block_type, hit.chunk.section or "", text[:500]]))

    if (
        "table" in block_type
        or metadata.get("is_table_chunk")
        or metadata.get("table_json") is not None
        or metadata.get("table_rows") is not None
        or _looks_like_table(text)
    ):
        return "table"
    if (
        block_type in {"formula", "equation"}
        or metadata.get("formula_text")
        or metadata.get("equation")
        or _looks_like_formula(text)
    ):
        return "formula"
    if (
        block_type in {"figure", "image", "caption", "chart"}
        or metadata.get("caption")
        or any(term in folded for term in ("figure", "fig.", "chart", "caption", "image"))
    ):
        return "figure"
    return "text"


def expected_answer_shape(question: str, query_type: str, packets: list[EvidencePacket]) -> str:
    q_folded = _fold_text(question)
    mode = choose_reasoning_mode(question, query_type, packets)
    if mode == "table":
        return "table"
    if mode == "formula":
        return "formula"
    if _is_numeric_question(q_folded):
        return "numeric"
    if query_type == "definition" or any(term in q_folded for term in ("what is", "definition", "la gi", "means")):
        return "definition"
    if query_type == "procedural" or any(term in q_folded for term in ("steps", "procedure", "process", "quy trinh")):
        return "procedure"
    if query_type == "comparison" or any(term in q_folded for term in ("compare", "difference", "versus", "khac nhau")):
        return "comparison"
    return "text"


def choose_reasoning_mode(question: str, query_type: str, packets: list[EvidencePacket]) -> ReasoningMode:
    q_folded = _fold_text(question)
    modalities = {packet.modality for packet in packets}
    if ("table" in modalities or _is_table_question(q_folded)):
        return "table"
    if ("formula" in modalities or _is_formula_question(q_folded)):
        return "formula"
    if ("figure" in modalities or _is_figure_question(q_folded)):
        return "figure"
    if query_type in {"comparison", "procedural", "multi_hop"} or _needs_multi_span(q_folded):
        return "multi_span"
    return "text"


def try_rule_based_table_lookup(question: str, packets: list[EvidencePacket]) -> tuple[EvidencePacket, str] | None:
    q_folded = _fold_text(question)
    query_number = _first_float(q_folded)
    query_grade = _first_grade(question)

    for packet in packets:
        if packet.modality != "table":
            continue
        lines = _table_lines(packet)
        if query_number is not None:
            for line in lines:
                grade = _first_grade(line)
                interval = _line_interval_text(line)
                payload = _line_primary_payload(line)
                if _line_range_contains(line, query_number):
                    if grade and interval:
                        return packet, f"{_format_number(query_number)} falls in {interval}, which maps to {grade}."
                    if payload and interval:
                        return packet, f"{_format_number(query_number)} falls in {interval}, which maps to {payload}."
        if query_grade:
            for line in lines:
                grade = _first_grade(line)
                if grade and grade.upper() == query_grade.upper():
                    interval = _line_interval_text(line)
                    payload = _line_secondary_payload(line, query_grade.upper())
                    if interval:
                        suffix = f" with {payload}" if payload else ""
                        return packet, f"{query_grade.upper()} corresponds to {interval}{suffix}."
    return None


def validate_llm_response(
    response: GroundedLLMResponse,
    request: GroundedLLMRequest,
    config: LLMFallbackConfig,
) -> str | None:
    if response.decision == "insufficient_evidence":
        return None
    if not response.answer.strip():
        return "llm_answer_empty"
    if response.confidence < config.min_llm_confidence:
        return "llm_confidence_below_threshold"
    allowed_ids = {packet.evidence_id for packet in request.evidence_packets}
    used_ids = set(response.used_evidence_ids)
    if not used_ids:
        return "llm_answer_missing_used_evidence_ids"
    if not used_ids <= allowed_ids:
        return "llm_used_unknown_evidence_id"
    if response.reasoning_mode == "figure":
        used_packets = [packet for packet in request.evidence_packets if packet.evidence_id in used_ids]
        if not any(packet.caption or packet.text for packet in used_packets):
            return "figure_evidence_lacks_textual_grounding"
    if not _answer_grounded_in_packets(response.answer, [packet for packet in request.evidence_packets if packet.evidence_id in used_ids]):
        return "llm_answer_not_grounded_in_used_packets"
    return None


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
    selected_provider = (provider or os.getenv("BOXTALK_LLM_PROVIDER") or "dummy").strip().lower()
    config = LLMFallbackConfig(
        enable_llm_fallback=True,
        enable_table_llm_reasoning=_bool_env("BOXTALK_ENABLE_TABLE_LLM_REASONING", False),
        enable_formula_llm_reasoning=_bool_env("BOXTALK_ENABLE_FORMULA_LLM_REASONING", False),
        enable_figure_llm_reasoning=_bool_env("BOXTALK_ENABLE_FIGURE_LLM_REASONING", False),
        fallback_only_if_grounded_evidence_present=_bool_env("BOXTALK_LLM_FALLBACK_REQUIRES_EVIDENCE", True),
        min_evidence_relevance=float(os.getenv("BOXTALK_LLM_FALLBACK_MIN_RELEVANCE", "0.30")),
        sufficiency_threshold=float(os.getenv("BOXTALK_LLM_FALLBACK_SUFFICIENCY_THRESHOLD", "0.72")),
        min_llm_confidence=float(os.getenv("BOXTALK_LLM_FALLBACK_MIN_CONFIDENCE", "0.30")),
        min_non_answer_override_confidence=float(os.getenv("BOXTALK_LLM_FALLBACK_MIN_OVERRIDE_CONFIDENCE", "0.65")),
        max_evidence_packets=int(os.getenv("BOXTALK_LLM_FALLBACK_MAX_PACKETS", "6")),
        max_packet_chars=int(os.getenv("BOXTALK_LLM_FALLBACK_MAX_PACKET_CHARS", "1800")),
        request_timeout_s=float(os.getenv("BOXTALK_LLM_TIMEOUT_SECONDS", "30")),
    )
    if selected_provider in {"openai", "openai-compatible", "openai_compatible"}:
        client: BaseGroundedLLMClient = OpenAICompatibleGroundedLLMClient(timeout_s=config.request_timeout_s)
    else:
        client = DummyGroundedLLMClient()
    return GroundedLLMFallback(config=config, client=client)


def provider_runtime_info(provider: str | None = None) -> dict[str, Any]:
    selected_provider = (provider or os.getenv("BOXTALK_LLM_PROVIDER") or "dummy").strip().lower()
    normalized_provider = "openai-compatible" if selected_provider in {"openai", "openai-compatible", "openai_compatible"} else "dummy"
    missing_envs: list[str] = []
    if normalized_provider == "openai-compatible":
        for name in ("BOXTALK_LLM_BASE_URL", "BOXTALK_LLM_API_KEY", "BOXTALK_LLM_MODEL"):
            if not os.getenv(name):
                missing_envs.append(name)
    return {
        "provider": normalized_provider,
        "ready": not missing_envs,
        "missing_envs": missing_envs,
        "env": {
            "BOXTALK_ENABLE_LLM_FALLBACK": os.getenv("BOXTALK_ENABLE_LLM_FALLBACK"),
            "BOXTALK_ENABLE_TABLE_LLM_REASONING": os.getenv("BOXTALK_ENABLE_TABLE_LLM_REASONING"),
            "BOXTALK_ENABLE_FORMULA_LLM_REASONING": os.getenv("BOXTALK_ENABLE_FORMULA_LLM_REASONING"),
            "BOXTALK_ENABLE_FIGURE_LLM_REASONING": os.getenv("BOXTALK_ENABLE_FIGURE_LLM_REASONING"),
            "BOXTALK_LLM_PROVIDER": os.getenv("BOXTALK_LLM_PROVIDER"),
            "BOXTALK_LLM_BASE_URL": os.getenv("BOXTALK_LLM_BASE_URL"),
            "BOXTALK_LLM_MODEL": os.getenv("BOXTALK_LLM_MODEL"),
            "BOXTALK_LLM_FALLBACK_SUFFICIENCY_THRESHOLD": os.getenv("BOXTALK_LLM_FALLBACK_SUFFICIENCY_THRESHOLD"),
            "BOXTALK_LLM_FALLBACK_MIN_CONFIDENCE": os.getenv("BOXTALK_LLM_FALLBACK_MIN_CONFIDENCE"),
            "BOXTALK_LLM_FALLBACK_MIN_OVERRIDE_CONFIDENCE": os.getenv("BOXTALK_LLM_FALLBACK_MIN_OVERRIDE_CONFIDENCE"),
        },
    }


def _fold_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", normalize_text(text))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_text).strip().lower()


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def _prompt_safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "block_type",
        "block_types",
        "caption",
        "figure_label",
        "table_backend",
        "table_shape",
        "formula_text",
        "equation",
        "section",
        "heading_path",
        "item_number",
    }
    return {key: value for key, value in metadata.items() if key in allowed}


def _coerce_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _looks_like_table(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if sum(1 for line in lines if "|" in line) >= 2:
        return True
    if sum(1 for line in lines if len(re.split(r"\s{2,}|\t", line)) >= 3) >= 2:
        return True
    return False


def _looks_like_formula(text: str) -> bool:
    if MATH_RE.search(text):
        return True
    folded = _fold_text(text[:500])
    return any(term in folded for term in ("equation", "formula", "where ", "denote", "let ")) and "=" in text


def _extract_formula_text(text: str) -> str | None:
    match = MATH_RE.search(text)
    if match:
        return match.group(0)
    return text[:500] if "=" in text else None


def _extract_caption(text: str, metadata: dict[str, Any]) -> str | None:
    caption = metadata.get("caption")
    if caption:
        return normalize_text(str(caption))
    for sentence in split_sentences(text):
        if any(term in _fold_text(sentence) for term in ("figure", "fig.", "chart", "caption")):
            return sentence
    return None


def _is_numeric_question(q_folded: str) -> bool:
    return any(
        term in q_folded
        for term in (
            "how many",
            "how much",
            "how long",
            "what score",
            "what value",
            "what rate",
            "bao nhieu",
            "bao lau",
            "may ",
            "thoi gian",
            "toi da",
            "toi thieu",
            "tin chi",
            "phan tram",
            "diem",
            "percent",
            "score",
            "gpa",
            "bleu",
            "f1",
            "layers",
            "heads",
            "dimensions",
        )
    )


def _is_table_question(q_folded: str) -> bool:
    return any(term in q_folded for term in ("table", "bang", "row", "column", "cell", "muc nao", "tuong ung"))


def _is_formula_question(q_folded: str) -> bool:
    return any(term in q_folded for term in ("formula", "equation", "cong thuc", "symbol", "ffn(", "=", "metric"))


def _is_figure_question(q_folded: str) -> bool:
    return any(term in q_folded for term in ("figure", "fig.", "chart", "image", "caption", "bieu do", "hinh"))


def _needs_multi_span(q_folded: str) -> bool:
    return any(
        term in q_folded
        for term in (
            "compare",
            "difference",
            "versus",
            "steps",
            "procedure",
            "process",
            "summarize",
            "explain",
            "why",
            "khac nhau",
            "so sanh",
            "cac buoc",
            "quy trinh",
            "giai thich",
            "tai sao",
        )
    )


def _has_number_or_formula(text_folded: str) -> bool:
    return bool(NUMBER_RE.search(text_folded) or NUMBER_UNIT_RE.search(text_folded) or "=" in text_folded)


def _standard_answer_shape_mismatch(*, question_folded: str, answer_folded: str, expected_shape: str) -> bool:
    if expected_shape in {"numeric", "formula"}:
        return not _has_number_or_formula(answer_folded)
    if expected_shape == "table":
        if _is_numeric_question(question_folded) and not _has_number_or_formula(answer_folded):
            return True
        if GRADE_RE.search(question_folded) and not GRADE_RE.search(answer_folded):
            return True
    if expected_shape == "procedure":
        return any(term in question_folded for term in ("steps", "cac buoc")) and not re.search(r"(?:^|\n)\s*(?:[-*]|\d+[.)])", answer_folded)
    if expected_shape == "comparison":
        return any(term in question_folded for term in ("difference", "compare", "khac nhau")) and len(split_sentences(answer_folded)) < 2
    return False


def _best_numeric_span(question_folded: str, packets: list[EvidencePacket]) -> tuple[EvidencePacket, str] | None:
    scored: list[tuple[float, EvidencePacket, str]] = []
    q_terms = set(question_folded.split())
    for packet_idx, packet in enumerate(packets):
        for sentence in split_sentences(packet.text):
            folded = _fold_text(sentence)
            if not _has_number_or_formula(folded):
                continue
            terms = set(folded.split())
            score = len(q_terms & terms) + (2.0 if NUMBER_UNIT_RE.search(folded) else 0.5)
            score += 1.0 if packet.modality in {"table", "formula"} else 0.0
            score += max(0.0, 0.6 - packet_idx * 0.1)
            scored.append((score, packet, normalize_text(sentence)))
    if not scored:
        return None
    scored.sort(key=lambda item: (item[0], -len(item[2])), reverse=True)
    return scored[0][1], scored[0][2]


def _first_relevant_sentence(question_folded: str, text: str) -> str | None:
    q_terms = set(question_folded.split())
    candidates: list[tuple[float, str]] = []
    for sentence in split_sentences(text):
        folded = _fold_text(sentence)
        terms = set(folded.split())
        overlap = len(q_terms & terms)
        if overlap == 0 and q_terms:
            continue
        score = overlap + min(1.0, len(terms) / 80.0)
        candidates.append((score, normalize_text(sentence)))
    if not candidates:
        sentences = split_sentences(text)
        return normalize_text(sentences[0]) if sentences else None
    candidates.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)
    return candidates[0][1]


def _table_lines(packet: EvidencePacket) -> list[str]:
    source = packet.table_text or packet.text
    raw_lines = [line.strip(" |-") for line in source.splitlines() if line.strip(" |-")]
    if len(raw_lines) >= 2:
        return raw_lines
    if packet.table_rows:
        return [normalize_text(json.dumps(row, ensure_ascii=False)) for row in packet.table_rows]
    if len(raw_lines) <= 1:
        raw_lines = [part.strip(" |-") for part in re.split(r";|\n", source) if part.strip(" |-")]
    return raw_lines


def _first_float(text: str) -> float | None:
    match = NUMBER_RE.search(text.replace(",", "."))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", "."))
    except ValueError:
        return None


def _first_grade(text: str) -> str | None:
    match = GRADE_RE.search(text)
    return match.group(1).upper() if match else None


def _line_range_contains(line: str, value: float) -> bool:
    folded = _fold_text(line).replace(",", ".")
    for match in RANGE_RE.finditer(folded):
        low = _to_float(match.group("low"))
        high = _to_float(match.group("high"))
        if low is not None and high is not None and min(low, high) <= value <= max(low, high):
            return True
    lower = LOWER_BOUND_RE.search(folded)
    upper = UPPER_BOUND_RE.search(folded)
    if lower and upper:
        low = _to_float(lower.group("low"))
        high = _to_float(upper.group("high"))
        return low is not None and high is not None and low <= value <= high
    if lower:
        low = _to_float(lower.group("low"))
        return low is not None and value >= low
    if upper:
        high = _to_float(upper.group("high"))
        return high is not None and value <= high
    return False


def _line_interval_text(line: str) -> str | None:
    folded = _fold_text(line)
    range_match = RANGE_RE.search(folded)
    if range_match:
        return f"{range_match.group('low')} to {range_match.group('high')}"
    lower = LOWER_BOUND_RE.search(folded)
    upper = UPPER_BOUND_RE.search(folded)
    if lower and upper:
        return f"{lower.group('low')} to {upper.group('high')}"
    if lower:
        return f"at least {lower.group('low')}"
    if upper:
        return f"at most {upper.group('high')}"
    return None


def _pipe_cells(line: str) -> list[str]:
    return [normalize_text(cell) for cell in line.split("|") if normalize_text(cell)]


def _line_primary_payload(line: str) -> str | None:
    cells = _pipe_cells(line)
    if len(cells) >= 2:
        for cell in cells[1:]:
            if cell and not RANGE_RE.search(_fold_text(cell)):
                return cell
    return None


def _line_secondary_payload(line: str, excluded_grade: str) -> str | None:
    cells = _pipe_cells(line)
    for cell in cells[1:]:
        if not cell:
            continue
        if _first_grade(cell) == excluded_grade:
            continue
        if RANGE_RE.search(_fold_text(cell)):
            continue
        return cell
    return None


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value.replace(",", "."))
    except ValueError:
        return None


def _format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return str(value).rstrip("0").rstrip(".")


def _answer_grounded_in_packets(answer: str, packets: list[EvidencePacket]) -> bool:
    if not packets:
        return False
    evidence_text = "\n".join(packet.text for packet in packets)
    answer_norm = normalize_text(answer)
    if answer_norm and answer_norm.casefold() in normalize_text(evidence_text).casefold():
        return True
    answer_terms = token_set(answer_norm)
    evidence_terms = token_set(evidence_text)
    if not answer_terms:
        return False
    overlap = len(answer_terms & evidence_terms) / len(answer_terms)
    answer_numbers = set(NUMBER_RE.findall(answer_norm))
    if answer_numbers and all(number in evidence_text for number in answer_numbers):
        return overlap >= 0.25
    return overlap >= 0.45


def _support_snippet(packet: EvidencePacket, answer: str) -> str:
    answer_folded = _fold_text(answer)
    for sentence in split_sentences(packet.text):
        sentence_folded = _fold_text(sentence)
        if answer_folded and answer_folded in sentence_folded:
            return sentence
    return normalize_text(packet.text[:320])


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}
