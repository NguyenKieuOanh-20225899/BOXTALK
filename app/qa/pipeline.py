from __future__ import annotations

import time
from typing import Any, Protocol

from app.qa.answer_generator import GroundedAnswerGenerator
from app.qa.answer_validator import AnswerValidationResult, AnswerValidator
from app.qa.citation_builder import CitationBuildResult, CitationBuilder
from app.qa.context_builder import ContextBuilder, GroundedContext
from app.qa.evidence_checker import EvidenceChecker
from app.qa.llm_answer_generator import GeneratedGroundedAnswer, LLMGroundedAnswerGenerator
from app.qa.schemas import EvidenceAssessment, QAResult
from app.qa.table_query_utils import augment_table_lookup_query
from app.retrieval.route_planner import QueryAwareRetrievalPlanner
from app.retrieval.schemas import RetrievedHit
from app.retrieval.service import RetrievalService


ABSTAIN_TEXT = "Không tìm thấy đủ bằng chứng trong tài liệu để trả lời."
LLM_ERROR_TEXT = "Không thể gọi bộ sinh câu trả lời LLM. Bằng chứng đã được truy xuất nhưng quá trình sinh câu trả lời thất bại."
LLM_ABSTAIN_TEXT = "Bằng chứng đã được truy xuất nhưng bộ sinh câu trả lời LLM kết luận chưa đủ để trả lời chắc chắn."
VALIDATION_FAILED_TEXT = "Không thể xác minh câu trả lời từ các bằng chứng đã truy xuất."


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
        context_builder: ContextBuilder | None = None,
        answer_generator: Any | None = None,
        citation_builder: CitationBuilder | None = None,
        answer_validator: AnswerValidator | None = None,
        llm_fallback: Any | None = None,
        llm_explainer: Any | None = None,
    ) -> None:
        _ = llm_fallback, llm_explainer  # Deprecated runtime parameters kept for constructor compatibility.
        self.retrieval_service = retrieval_service
        self.router = router
        self.retrieval_planner = retrieval_planner or QueryAwareRetrievalPlanner()
        self.evidence_checker = evidence_checker or EvidenceChecker()
        self.context_builder = context_builder or ContextBuilder()
        self.answer_generator = answer_generator or LLMGroundedAnswerGenerator.from_env(required=True)
        self.citation_builder = citation_builder or CitationBuilder()
        self.answer_validator = answer_validator or AnswerValidator()

    def answer(self, question: str) -> QAResult:
        query_type_value = self.router.route(question)
        query_type = getattr(query_type_value, "value", str(query_type_value))
        retrieval_plan = self.retrieval_planner.plan(query_type, question)
        retrieval_query = augment_retrieval_query(question)
        retrieval_result = self.retrieval_service.retrieve(
            retrieval_query,
            strategy=retrieval_plan.strategy,
            config=retrieval_plan.config,
        )

        start = time.perf_counter()
        evidence = self._assess_evidence(question, query_type, retrieval_result.hits)
        selected_hits = self._selected_hits_from_evidence(evidence, retrieval_result.hits)

        if self._is_extractive_generator():
            grounded_answer = self.answer_generator.generate(
                question=question,
                query_type=query_type,
                hits=retrieval_result.hits,
                evidence=evidence,
            )
            answer_latency_ms = (time.perf_counter() - start) * 1000.0
            return self._result(
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
                grounded=grounded_answer.grounded,
                standard_answer=grounded_answer.answer,
                final_answer_source=grounded_answer.source,
                generator_type="extractive",
                selected_evidence_ids=list(evidence.selected_hit_ids),
                selected_evidence_count=len(selected_hits),
                validation_passed=True,
            )

        if not _evidence_is_sufficient(evidence):
            answer_latency_ms = (time.perf_counter() - start) * 1000.0
            return self._abstain_result(
                question=question,
                query_type=query_type,
                evidence=evidence,
                retrieved_hits=retrieval_result.hits,
                retrieval_strategy=retrieval_result.strategy,
                retrieval_config=retrieval_result.config,
                retrieval_latency_ms=retrieval_result.latency_ms,
                answer_latency_ms=answer_latency_ms,
                final_answer_source="evidence_insufficient",
                validation_reason=evidence.reason,
                selected_evidence_ids=[],
                selected_evidence_count=len(selected_hits),
            )

        context = self.context_builder.build(question=question, selected_hits=selected_hits)
        if not context.evidence:
            answer_latency_ms = (time.perf_counter() - start) * 1000.0
            return self._abstain_result(
                question=question,
                query_type=query_type,
                evidence=evidence,
                retrieved_hits=retrieval_result.hits,
                retrieval_strategy=retrieval_result.strategy,
                retrieval_config=retrieval_result.config,
                retrieval_latency_ms=retrieval_result.latency_ms,
                answer_latency_ms=answer_latency_ms,
                final_answer_source="evidence_insufficient",
                validation_reason="context_empty",
                selected_evidence_ids=[],
                selected_evidence_count=0,
                context_token_count=0,
                context_evidence=[],
            )

        generated = self.answer_generator.generate(question=question, context=context)
        if generated.abstain:
            answer_latency_ms = (time.perf_counter() - start) * 1000.0
            final_source = "llm_error" if str(generated.reason or "").startswith("llm_error") else "llm_abstain"
            return self._abstain_result(
                question=question,
                query_type=query_type,
                evidence=evidence,
                retrieved_hits=retrieval_result.hits,
                retrieval_strategy=retrieval_result.strategy,
                retrieval_config=retrieval_result.config,
                retrieval_latency_ms=retrieval_result.latency_ms,
                answer_latency_ms=answer_latency_ms,
                final_answer_source=final_source,
                validation_reason=generated.reason,
                selected_evidence_ids=context.evidence_ids,
                selected_evidence_count=len(context.evidence),
                context_token_count=context.token_count,
                context_evidence=_context_evidence_trace(context),
                generated=generated,
            )

        final_answer_source = "llm_grounded"

        citation_result = self.citation_builder.build(
            context=context,
            used_evidence_ids=generated.used_evidence_ids,
        )
        validation = self.answer_validator.validate(
            question=question,
            answer=generated.answer,
            used_evidence_ids=generated.used_evidence_ids,
            citations=citation_result.citations,
            context=context,
        )
        if citation_result.invalid_evidence_ids and validation.valid:
            validation = AnswerValidationResult(
                valid=False,
                reason="unknown_evidence_id",
                details={"invalid_evidence_ids": citation_result.invalid_evidence_ids},
            )

        if not validation.valid:
            answer_latency_ms = (time.perf_counter() - start) * 1000.0
            return self._abstain_result(
                question=question,
                query_type=query_type,
                evidence=evidence,
                retrieved_hits=retrieval_result.hits,
                retrieval_strategy=retrieval_result.strategy,
                retrieval_config=retrieval_result.config,
                retrieval_latency_ms=retrieval_result.latency_ms,
                answer_latency_ms=answer_latency_ms,
                final_answer_source="validation_failed",
                validation_reason=validation.reason,
                selected_evidence_ids=context.evidence_ids,
                selected_evidence_count=len(context.evidence),
                context_token_count=context.token_count,
                context_evidence=_context_evidence_trace(context),
                generated=generated,
                citation_result=citation_result,
                validation=validation,
            )

        answer_latency_ms = (time.perf_counter() - start) * 1000.0
        return self._result(
            question=question,
            query_type=query_type,
            answer=generated.answer,
            decision="answer",
            evidence=evidence,
            citations=citation_result.citations,
            retrieved_hits=retrieval_result.hits,
            retrieval_strategy=retrieval_result.strategy,
            retrieval_config=retrieval_result.config,
            retrieval_latency_ms=retrieval_result.latency_ms,
            answer_latency_ms=answer_latency_ms,
            grounded=True,
            standard_answer=None,
            final_answer_source=final_answer_source,
            generator_type="llm_grounded",
            generator_provider=generated.provider,
            generator_model=generated.model,
            llm_latency_ms=generated.latency_ms,
            selected_evidence_ids=context.evidence_ids,
            selected_evidence_count=len(context.evidence),
            context_token_count=context.token_count,
            context_evidence=_context_evidence_trace(context),
            validation_passed=True,
            validation_reason=None,
            used_evidence_ids=list(generated.used_evidence_ids),
        )

    def _assess_evidence(self, question: str, query_type: str, hits: list[RetrievedHit]) -> EvidenceAssessment:
        checker = self.evidence_checker
        if hasattr(checker, "check"):
            return checker.check(question=question, query_type=query_type, hits=hits)  # type: ignore[call-arg]
        return checker.assess(question, query_type, hits)

    def _selected_hits_from_evidence(
        self,
        evidence: EvidenceAssessment,
        hits: list[RetrievedHit],
    ) -> list[RetrievedHit]:
        if evidence.selected_hit_ids:
            hit_by_id = {hit.chunk_id: hit for hit in hits}
            selected = [hit_by_id[chunk_id] for chunk_id in evidence.selected_hit_ids if chunk_id in hit_by_id]
            if selected:
                return selected
        return hits[: min(2, len(hits))]

    def _is_extractive_generator(self) -> bool:
        return isinstance(self.answer_generator, GroundedAnswerGenerator)

    def _abstain_result(
        self,
        *,
        question: str,
        query_type: str,
        evidence: EvidenceAssessment,
        retrieved_hits: list[RetrievedHit],
        retrieval_strategy: str,
        retrieval_config: Any,
        retrieval_latency_ms: float,
        answer_latency_ms: float,
        final_answer_source: str,
        validation_reason: str | None,
        selected_evidence_ids: list[str],
        selected_evidence_count: int,
        context_token_count: int | None = None,
        context_evidence: list[dict[str, Any]] | None = None,
        generated: GeneratedGroundedAnswer | None = None,
        citation_result: CitationBuildResult | None = None,
        validation: AnswerValidationResult | None = None,
    ) -> QAResult:
        if final_answer_source == "validation_failed":
            answer = VALIDATION_FAILED_TEXT
        elif final_answer_source == "llm_error":
            answer = LLM_ERROR_TEXT
        elif final_answer_source == "llm_abstain":
            answer = LLM_ABSTAIN_TEXT
        else:
            answer = ABSTAIN_TEXT
        return self._result(
            question=question,
            query_type=query_type,
            answer=answer,
            decision="abstain",
            evidence=evidence,
            citations=(citation_result.citations if citation_result else []),
            retrieved_hits=retrieved_hits,
            retrieval_strategy=retrieval_strategy,
            retrieval_config=retrieval_config,
            retrieval_latency_ms=retrieval_latency_ms,
            answer_latency_ms=answer_latency_ms,
            grounded=False,
            standard_answer=None,
            final_answer_source=final_answer_source,
            generator_type="llm_grounded",
            generator_model=getattr(generated, "model", None),
            generator_provider=getattr(generated, "provider", None),
            llm_latency_ms=float(getattr(generated, "latency_ms", 0.0) or 0.0),
            selected_evidence_ids=selected_evidence_ids,
            selected_evidence_count=selected_evidence_count,
            context_token_count=context_token_count,
            context_evidence=context_evidence or [],
            validation_passed=bool(validation.valid) if validation else False,
            validation_reason=validation_reason,
            used_evidence_ids=list(getattr(generated, "used_evidence_ids", []) or []),
        )

    def _result(
        self,
        *,
        question: str,
        query_type: str,
        answer: str,
        decision: str,
        evidence: EvidenceAssessment,
        citations: list[dict[str, Any]],
        retrieved_hits: list[RetrievedHit],
        retrieval_strategy: str,
        retrieval_config: Any,
        retrieval_latency_ms: float,
        answer_latency_ms: float,
        grounded: bool,
        standard_answer: str | None,
        final_answer_source: str,
        generator_type: str,
        selected_evidence_ids: list[str],
        selected_evidence_count: int,
        validation_passed: bool,
        generator_provider: str | None = None,
        generator_model: str | None = None,
        llm_latency_ms: float = 0.0,
        context_token_count: int | None = None,
        context_evidence: list[dict[str, Any]] | None = None,
        validation_reason: str | None = None,
        used_evidence_ids: list[str] | None = None,
    ) -> QAResult:
        top_hit = retrieved_hits[0] if retrieved_hits else None
        route_attempts = [
            {
                "attempt_index": 0,
                "query_type": query_type,
                "retrieval_strategy": retrieval_strategy,
                "retrieval_config": retrieval_config.to_dict(),
                "evidence_decision": evidence.decision,
                "evidence_sufficient": _evidence_is_sufficient(evidence),
                "sufficiency": evidence.sufficiency,
                "relevance": evidence.relevance,
                "coverage": evidence.coverage,
                "grounding": evidence.grounding,
                "missing_constraints": list(evidence.missing_constraints),
                "quality_score": evidence.sufficiency,
                "selected": True,
                "retry_reason": "initial_route",
                "top_hit_chunk_id": top_hit.chunk_id if top_hit else None,
                "top_hit_score": float(top_hit.final_score or top_hit.score) if top_hit else 0.0,
                "retrieval_latency_ms": retrieval_latency_ms,
                "answer_latency_ms": answer_latency_ms,
                "final_answer_source": final_answer_source,
                "generator_type": generator_type,
                "validation_passed": validation_passed,
                "validation_reason": validation_reason,
            }
        ]
        return QAResult(
            question=question,
            query_type=query_type,
            answer=answer,
            decision=decision,  # type: ignore[arg-type]
            evidence=evidence,
            citations=citations,
            retrieved_hits=retrieved_hits,
            retrieval_strategy=retrieval_strategy,
            retrieval_config=retrieval_config,
            retrieval_latency_ms=retrieval_latency_ms,
            answer_latency_ms=answer_latency_ms,
            route_attempts=route_attempts,
            selected_route_attempt=0,
            grounded=grounded,
            standard_answer=standard_answer,
            final_answer_source=final_answer_source,
            fallback_trace={},
            explanation=None,
            explanation_trace={},
            selected_evidence_count=selected_evidence_count,
            selected_evidence_ids=selected_evidence_ids,
            evidence_sufficient=_evidence_is_sufficient(evidence),
            evidence_reason=evidence.reason,
            missing_constraints=list(evidence.missing_constraints),
            context_token_count=context_token_count,
            context_evidence=context_evidence or [],
            generator_type=generator_type,
            generator_provider=generator_provider,
            generator_model=generator_model,
            llm_latency_ms=llm_latency_ms,
            validation_passed=validation_passed,
            validation_reason=validation_reason,
            used_evidence_ids=list(used_evidence_ids or []),
        )


def _evidence_is_sufficient(evidence: EvidenceAssessment) -> bool:
    explicit = bool(getattr(evidence, "sufficient", False))
    return explicit or (evidence.decision == "answer" and not evidence.missing_constraints)


def _context_evidence_trace(context: GroundedContext) -> list[dict[str, Any]]:
    return [item.to_dict() for item in context.evidence]


def augment_retrieval_query(question: str) -> str:
    return augment_table_lookup_query(question)
