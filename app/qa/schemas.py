from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from app.retrieval.schemas import RetrievedHit, RetrievalConfig


QADecision = Literal["answer", "expand_retrieval", "switch_strategy", "abstain"]


@dataclass(slots=True)
class EvidenceAssessment:
    """Evidence sufficiency report for grounded QA."""

    relevance: float
    coverage: float
    consistency: float
    citation_support: float
    grounding: float
    sufficiency: float
    decision: QADecision
    reason: str
    selected_hit_ids: list[str] = field(default_factory=list)
    support_sentences: list[str] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GroundedAnswer:
    """Answer text and citations produced only from retrieved evidence."""

    answer: str
    citations: list[dict[str, Any]]
    support_sentences: list[str]
    grounded: bool
    answer_type: str = "extractive"
    source: str = "standard"
    fallback_trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class QAResult:
    """End-to-end routed retrieval plus grounded QA result."""

    question: str
    query_type: str
    answer: str
    decision: QADecision
    evidence: EvidenceAssessment
    citations: list[dict[str, Any]]
    retrieved_hits: list[RetrievedHit]
    retrieval_strategy: str
    retrieval_config: RetrievalConfig
    retrieval_latency_ms: float
    answer_latency_ms: float
    route_attempts: list[dict[str, Any]] = field(default_factory=list)
    selected_route_attempt: int = 0
    grounded: bool = False
    standard_answer: str | None = None
    final_answer_source: str = "standard"
    fallback_trace: dict[str, Any] = field(default_factory=dict)

    @property
    def total_latency_ms(self) -> float:
        return self.retrieval_latency_ms + self.answer_latency_ms

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "query_type": self.query_type,
            "answer": self.answer,
            "standard_answer": self.standard_answer,
            "final_answer_source": self.final_answer_source,
            "decision": self.decision,
            "evidence": self.evidence.to_dict(),
            "citations": self.citations,
            "retrieved_hits": [hit.to_dict() for hit in self.retrieved_hits],
            "retrieval_strategy": self.retrieval_strategy,
            "retrieval_config": self.retrieval_config.to_dict(),
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "answer_latency_ms": self.answer_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "route_attempts": list(self.route_attempts),
            "selected_route_attempt": self.selected_route_attempt,
            "grounded": self.grounded,
            "fallback_trace": dict(self.fallback_trace),
        }
