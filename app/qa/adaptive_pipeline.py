from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

from app.qa.answer_generator import GroundedAnswerGenerator
from app.qa.evidence_checker import EvidenceChecker
from app.qa.llm_fallback import GroundedLLMFallback
from app.qa.schemas import EvidenceAssessment, GroundedAnswer, QAResult
from app.retrieval.route_planner import QueryAwareRetrievalPlanner, QueryRetrievalPlan
from app.retrieval.schemas import RetrievalResult
from app.retrieval.service import RetrievalService


class QueryRouterProtocol(Protocol):
    def route(self, question: str):
        ...


@dataclass(slots=True)
class RouteAttemptResult:
    """One routed retrieval and grounded answer attempt."""

    attempt_index: int
    query_type: str
    retry_reason: str
    plan: QueryRetrievalPlan
    retrieval_result: RetrievalResult
    evidence: EvidenceAssessment
    answer: GroundedAnswer
    answer_latency_ms: float
    quality_score: float

    @property
    def total_latency_ms(self) -> float:
        return self.retrieval_result.latency_ms + self.answer_latency_ms


class AdaptiveRouteRetryQAPipeline:
    """Lightweight route-retry controller for grounded QA.

    The controller follows the Adaptive-RAG/CRAG/Self-RAG idea pragmatically:
    route once, evaluate evidence, retry alternative route types only when the
    evidence/answer quality is weak, then return the best grounded attempt.
    """

    FALLBACK_ROUTES: dict[str, list[str]] = {
        "factoid": ["definition", "policy", "ambiguous"],
        "definition": ["factoid", "policy", "ambiguous"],
        "policy": ["factoid", "procedural", "ambiguous"],
        "procedural": ["policy", "factoid", "comparison"],
        "comparison": ["multi_hop", "factoid", "policy"],
        "multi_hop": ["comparison", "policy", "factoid"],
        "ambiguous": ["factoid", "policy", "comparison", "procedural"],
        "ambiguous_or_insufficient": ["factoid", "policy", "comparison", "procedural"],
    }

    def __init__(
        self,
        *,
        retrieval_service: RetrievalService,
        router: QueryRouterProtocol,
        retrieval_planner: QueryAwareRetrievalPlanner | None = None,
        evidence_checker: EvidenceChecker | None = None,
        answer_generator: GroundedAnswerGenerator | None = None,
        llm_fallback: GroundedLLMFallback | None = None,
        max_attempts: int = 3,
        retry_quality_threshold: float = 0.82,
        min_answer_quality: float = 0.40,
        enable_final_route_llm_fallback: bool = False,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.router = router
        self.retrieval_planner = retrieval_planner or QueryAwareRetrievalPlanner()
        self.evidence_checker = evidence_checker or EvidenceChecker()
        self.answer_generator = answer_generator or GroundedAnswerGenerator()
        self.llm_fallback = llm_fallback
        self.max_attempts = max(1, max_attempts)
        self.retry_quality_threshold = retry_quality_threshold
        self.min_answer_quality = min_answer_quality
        self.enable_final_route_llm_fallback = enable_final_route_llm_fallback

    def answer(self, question: str) -> QAResult:
        initial_query_type = self._normalize_query_type(self.router.route(question))
        query_types = self._candidate_query_types(initial_query_type, question)
        attempts: list[RouteAttemptResult] = []

        for idx, query_type in enumerate(query_types[: self.max_attempts]):
            retry_reason = "initial_route" if idx == 0 else "previous_quality_below_threshold"
            attempt = self._run_attempt(question, query_type, idx, retry_reason)
            attempts.append(attempt)
            if self._can_stop(attempt):
                break

        best_idx = self._select_best_attempt(attempts)
        best = attempts[best_idx]
        evidence = best.evidence
        answer = best.answer

        if evidence.decision == "answer" and best.quality_score < self.min_answer_quality:
            evidence = self._force_abstain(evidence, best.quality_score)
            answer = self.answer_generator.generate(
                question=question,
                query_type=best.query_type,
                hits=best.retrieval_result.hits,
                evidence=evidence,
            )

        final_answer = answer.answer
        final_citations = answer.citations
        final_decision = evidence.decision
        final_answer_source = answer.source
        final_grounded = answer.grounded
        fallback_trace: dict[str, object] = {}
        if self.enable_final_route_llm_fallback and self.llm_fallback is not None:
            fallback_result = self.llm_fallback.maybe_generate(
                question=question,
                query_type=best.query_type,
                hits=best.retrieval_result.hits,
                evidence=evidence,
                standard_answer=answer,
            )
            fallback_trace = fallback_result.to_trace()
            if fallback_result.used and fallback_result.answer:
                final_answer = fallback_result.answer
                final_citations = fallback_result.citations or answer.citations
                final_decision = "answer"
                final_answer_source = fallback_result.final_answer_source
                final_grounded = bool(final_citations)

        route_attempts = [
            self._attempt_trace(attempt, selected=(idx == best_idx))
            for idx, attempt in enumerate(attempts)
        ]
        if route_attempts:
            route_attempts[best_idx].update(
                {
                    "fallback_called": bool(fallback_trace.get("called", False)),
                    "fallback_used": bool(fallback_trace.get("used", False)),
                    "fallback_reason": fallback_trace.get("reason"),
                    "fallback_reasoning_mode": fallback_trace.get("reasoning_mode"),
                    "final_answer_source": final_answer_source,
                }
            )

        return QAResult(
            question=question,
            query_type=best.query_type,
            answer=final_answer,
            decision=final_decision,
            evidence=evidence,
            citations=final_citations,
            retrieved_hits=best.retrieval_result.hits,
            retrieval_strategy=best.retrieval_result.strategy,
            retrieval_config=best.retrieval_result.config,
            retrieval_latency_ms=sum(attempt.retrieval_result.latency_ms for attempt in attempts),
            answer_latency_ms=sum(attempt.answer_latency_ms for attempt in attempts),
            route_attempts=route_attempts,
            selected_route_attempt=best_idx,
            grounded=final_grounded,
            standard_answer=answer.answer,
            final_answer_source=final_answer_source,
            fallback_trace=fallback_trace,
        )

    def _run_attempt(
        self,
        question: str,
        query_type: str,
        attempt_index: int,
        retry_reason: str,
    ) -> RouteAttemptResult:
        plan = self.retrieval_planner.plan(query_type, question)
        retrieval_result = self.retrieval_service.retrieve(
            question,
            strategy=plan.strategy,
            config=plan.config,
        )
        start = time.perf_counter()
        evidence = self.evidence_checker.assess(question, query_type, retrieval_result.hits)
        answer = self.answer_generator.generate(
            question=question,
            query_type=query_type,
            hits=retrieval_result.hits,
            evidence=evidence,
        )
        answer_latency_ms = (time.perf_counter() - start) * 1000.0
        quality_score = self._quality_score(evidence, answer, retrieval_result)
        return RouteAttemptResult(
            attempt_index=attempt_index,
            query_type=query_type,
            retry_reason=retry_reason,
            plan=plan,
            retrieval_result=retrieval_result,
            evidence=evidence,
            answer=answer,
            answer_latency_ms=answer_latency_ms,
            quality_score=quality_score,
        )

    def _candidate_query_types(self, initial_query_type: str, question: str) -> list[str]:
        candidates = [initial_query_type]
        candidates.extend(self.FALLBACK_ROUTES.get(initial_query_type, self.FALLBACK_ROUTES["ambiguous"]))

        question_lower = question.lower()
        if any(term in question_lower for term in ("khác", "so sánh", "compare", "difference")):
            candidates.insert(1, "comparison")
        if any(term in question_lower for term in ("quy trình", "bước", "thủ tục", "how", "steps")):
            candidates.insert(1, "procedural")
        if any(term in question_lower for term in ("quy định", "điều kiện", "bắt buộc", "policy", "must")):
            candidates.insert(1, "policy")

        deduped: list[str] = []
        for candidate in candidates:
            normalized = candidate.strip().lower()
            if normalized and normalized not in deduped:
                deduped.append(normalized)
        return deduped

    def _quality_score(
        self,
        evidence: EvidenceAssessment,
        answer: GroundedAnswer,
        retrieval_result: RetrievalResult,
    ) -> float:
        top_hit = retrieval_result.hits[0] if retrieval_result.hits else None
        top_score = float(top_hit.final_score or top_hit.score) if top_hit else 0.0
        citation_bonus = 1.0 if answer.citations else 0.0
        answer_bonus = 1.0 if evidence.decision == "answer" else 0.0
        grounded_bonus = 1.0 if answer.grounded else 0.0
        score = (
            0.42 * evidence.sufficiency
            + 0.18 * evidence.relevance
            + 0.14 * evidence.coverage
            + 0.12 * evidence.grounding
            + 0.06 * min(1.0, top_score)
            + 0.04 * citation_bonus
            + 0.04 * answer_bonus
            + 0.04 * grounded_bonus
        )
        if evidence.decision != "answer":
            score -= 0.08
        return round(max(0.0, min(1.0, score)), 4)

    def _can_stop(self, attempt: RouteAttemptResult) -> bool:
        return (
            attempt.evidence.decision == "answer"
            and attempt.answer.grounded
            and bool(attempt.answer.citations)
            and attempt.quality_score >= self.retry_quality_threshold
        )

    def _select_best_attempt(self, attempts: list[RouteAttemptResult]) -> int:
        if not attempts:
            return 0
        ranked = sorted(
            enumerate(attempts),
            key=lambda item: (
                item[1].quality_score,
                item[1].evidence.sufficiency,
                item[1].evidence.relevance,
                -item[0],
            ),
            reverse=True,
        )
        return ranked[0][0]

    def _force_abstain(self, evidence: EvidenceAssessment, quality_score: float) -> EvidenceAssessment:
        return EvidenceAssessment(
            relevance=evidence.relevance,
            coverage=evidence.coverage,
            consistency=evidence.consistency,
            citation_support=evidence.citation_support,
            grounding=evidence.grounding,
            sufficiency=evidence.sufficiency,
            decision="abstain",
            reason=f"Adaptive route retry found no answer-quality route above {self.min_answer_quality:.2f}; best quality={quality_score:.3f}.",
            selected_hit_ids=evidence.selected_hit_ids,
            support_sentences=evidence.support_sentences,
            diagnostics={**evidence.diagnostics, "adaptive_forced_abstain": True, "quality_score": quality_score},
        )

    def _attempt_trace(self, attempt: RouteAttemptResult, *, selected: bool) -> dict[str, object]:
        top_hit = attempt.retrieval_result.hits[0] if attempt.retrieval_result.hits else None
        return {
            "attempt_index": attempt.attempt_index,
            "query_type": attempt.query_type,
            "retrieval_strategy": attempt.retrieval_result.strategy,
            "retrieval_config": attempt.retrieval_result.config.to_dict(),
            "evidence_decision": attempt.evidence.decision,
            "sufficiency": attempt.evidence.sufficiency,
            "relevance": attempt.evidence.relevance,
            "coverage": attempt.evidence.coverage,
            "grounding": attempt.evidence.grounding,
            "quality_score": attempt.quality_score,
            "selected": selected,
            "retry_reason": attempt.retry_reason,
            "top_hit_chunk_id": top_hit.chunk_id if top_hit else None,
            "top_hit_score": float(top_hit.final_score or top_hit.score) if top_hit else 0.0,
            "retrieval_latency_ms": attempt.retrieval_result.latency_ms,
            "answer_latency_ms": attempt.answer_latency_ms,
        }

    def _normalize_query_type(self, value: object) -> str:
        return getattr(value, "value", str(value)).strip().lower() or "ambiguous"
