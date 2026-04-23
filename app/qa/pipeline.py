from __future__ import annotations

import time
from typing import Protocol

from app.qa.answer_generator import GroundedAnswerGenerator
from app.qa.evidence_checker import EvidenceChecker
from app.qa.schemas import QAResult
from app.retrieval.route_planner import QueryAwareRetrievalPlanner
from app.retrieval.service import RetrievalService


class QueryRouterProtocol(Protocol):
    def route(self, question: str):
        ...


class GroundedQAPipeline:
    """End-to-end routed retrieval, evidence checking, and grounded answering."""

    def __init__(
        self,
        *,
        retrieval_service: RetrievalService,
        router: QueryRouterProtocol,
        retrieval_planner: QueryAwareRetrievalPlanner | None = None,
        evidence_checker: EvidenceChecker | None = None,
        answer_generator: GroundedAnswerGenerator | None = None,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.router = router
        self.retrieval_planner = retrieval_planner or QueryAwareRetrievalPlanner()
        self.evidence_checker = evidence_checker or EvidenceChecker()
        self.answer_generator = answer_generator or GroundedAnswerGenerator()

    def answer(self, question: str) -> QAResult:
        query_type_value = self.router.route(question)
        query_type = getattr(query_type_value, "value", str(query_type_value))
        retrieval_plan = self.retrieval_planner.plan(query_type, question)
        retrieval_result = self.retrieval_service.retrieve(
            question,
            strategy=retrieval_plan.strategy,
            config=retrieval_plan.config,
        )

        start = time.perf_counter()
        evidence = self.evidence_checker.assess(question, query_type, retrieval_result.hits)
        grounded_answer = self.answer_generator.generate(
            question=question,
            query_type=query_type,
            hits=retrieval_result.hits,
            evidence=evidence,
        )
        answer_latency_ms = (time.perf_counter() - start) * 1000.0
        top_hit = retrieval_result.hits[0] if retrieval_result.hits else None

        return QAResult(
            question=question,
            query_type=query_type,
            answer=grounded_answer.answer,
            decision=evidence.decision,
            evidence=evidence,
            citations=grounded_answer.citations,
            retrieved_hits=retrieval_result.hits,
            retrieval_strategy=retrieval_result.strategy,
            retrieval_config=retrieval_result.config,
            retrieval_latency_ms=retrieval_result.latency_ms,
            answer_latency_ms=answer_latency_ms,
            route_attempts=[
                {
                    "attempt_index": 0,
                    "query_type": query_type,
                    "retrieval_strategy": retrieval_result.strategy,
                    "retrieval_config": retrieval_result.config.to_dict(),
                    "evidence_decision": evidence.decision,
                    "sufficiency": evidence.sufficiency,
                    "relevance": evidence.relevance,
                    "coverage": evidence.coverage,
                    "grounding": evidence.grounding,
                    "quality_score": evidence.sufficiency,
                    "selected": True,
                    "retry_reason": "initial_route",
                    "top_hit_chunk_id": top_hit.chunk_id if top_hit else None,
                    "top_hit_score": float(top_hit.final_score or top_hit.score) if top_hit else 0.0,
                    "retrieval_latency_ms": retrieval_result.latency_ms,
                    "answer_latency_ms": answer_latency_ms,
                }
            ],
            selected_route_attempt=0,
        )
