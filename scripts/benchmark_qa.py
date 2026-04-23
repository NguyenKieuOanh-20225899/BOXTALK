from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import unicodedata
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.qa.adaptive_pipeline import AdaptiveRouteRetryQAPipeline
from app.qa.answer_generator import GroundedAnswerGenerator
from app.qa.evidence_checker import EvidenceChecker
from app.qa.llm_fallback import (
    DummyGroundedLLMClient,
    GroundedLLMFallback,
    LLMFallbackConfig,
    OpenAICompatibleGroundedLLMClient,
)
from app.qa.pipeline import GroundedQAPipeline
from app.qa.router import QueryRouter
from app.qa.schemas import EvidenceAssessment
from app.qa.text_utils import contains_text, split_sentences, token_f1, token_set
from app.retrieval.reranker import make_reranker
from app.retrieval.route_planner import QueryAwareRetrievalPlanner, QueryRetrievalPlan
from app.retrieval.schemas import RetrievedHit, RetrievalConfig
from app.retrieval.service import RetrievalService


RESULTS_ROOT = Path("results/qa_benchmark")
DEFAULT_CONFIGS = [
    "bm25_only",
    "dense_only",
    "hybrid_no_routing",
    "routed_grounded",
    "adaptive_route_retry",
    "routed_grounded_with_llm_fallback",
    "routed_grounded_with_table_llm",
    "routed_grounded_with_formula_llm",
    "routed_grounded_with_multimodal_textual_fallback",
    "adaptive_route_retry_with_final_route_llm_fallback",
    "no_evidence_checker",
    "no_router",
    "no_citation_grounding",
    "no_metadata_filter",
]


class FixedRouter:
    """Router ablation that assigns every query to one query type."""

    def __init__(self, query_type: str = "ambiguous") -> None:
        self.query_type = query_type

    def route(self, question: str) -> str:
        return self.query_type


class FixedRetrievalPlanner:
    """Planner for sparse/dense/hybrid baselines with fixed retrieval settings."""

    def __init__(self, *, strategy: str, config: RetrievalConfig, reason: str) -> None:
        self.strategy = strategy
        self.config = config
        self.reason = reason

    def plan(self, query_type: str, question: str) -> QueryRetrievalPlan:
        return QueryRetrievalPlan(strategy=self.strategy, config=replace(self.config), reason=self.reason)


class NoMetadataFilterPlanner:
    """Planner ablation that keeps routed strategy but clears metadata filters."""

    def __init__(self, base: QueryAwareRetrievalPlanner | None = None) -> None:
        self.base = base or QueryAwareRetrievalPlanner()

    def plan(self, query_type: str, question: str) -> QueryRetrievalPlan:
        plan = self.base.plan(query_type, question)
        config = replace(
            plan.config,
            block_type_filter=[],
            section_filter=[],
            doc_id_filter=[],
            source_name_filter=[],
            version_filter=[],
            date_filter=[],
            metadata_filters={},
        )
        return QueryRetrievalPlan(
            strategy=plan.strategy,
            config=config,
            reason=f"{plan.reason}; metadata filters disabled for ablation",
        )


class PermissiveEvidenceChecker:
    """Evidence-checker ablation that answers from whatever retrieval returned."""

    def assess(self, question: str, query_type: str, hits: list[RetrievedHit]) -> EvidenceAssessment:
        selected = hits[:3]
        support_sentences: list[str] = []
        for hit in selected:
            support_sentences.extend(split_sentences(hit.chunk.text)[:1])
            if len(support_sentences) >= 3:
                break

        return EvidenceAssessment(
            relevance=1.0 if selected else 0.0,
            coverage=1.0 if selected else 0.0,
            consistency=1.0 if selected else 0.0,
            citation_support=1.0 if selected else 0.0,
            grounding=1.0 if selected else 0.0,
            sufficiency=1.0 if selected else 0.0,
            decision="answer",
            reason="Evidence checker disabled; answer generation is forced from retrieved hits.",
            selected_hit_ids=[hit.chunk_id for hit in selected],
            support_sentences=support_sentences,
            diagnostics={"checker_disabled": True, "support_hit_count": len(selected)},
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark grounded QA over a retrieval index.")
    parser.add_argument("--index-dir", type=Path, required=True, help="Retrieval index directory.")
    parser.add_argument("--queries", type=Path, required=True, help="JSON/JSONL QA benchmark file.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory.")
    parser.add_argument("--reranker", choices=["none", "heuristic", "cross-encoder", "colbert"], default="heuristic")
    parser.add_argument("--reranker-model", default=None)
    parser.add_argument("--min-token-f1", type=float, default=0.45, help="Token-F1 threshold for gold_answer success.")
    parser.add_argument("--no-warmup", action="store_true", help="Disable one unmeasured warmup question.")
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help="Benchmark config. Repeat or comma-separate. Use 'all' for baseline + ablation.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Fixed baseline top-k.")
    parser.add_argument("--candidate-k", type=int, default=60, help="Fixed baseline candidate-k.")
    parser.add_argument("--bm25-weight", type=float, default=0.5, help="Fixed hybrid BM25 weight.")
    parser.add_argument("--dense-weight", type=float, default=0.5, help="Fixed hybrid dense weight.")
    parser.add_argument("--rerank-top-n", type=int, default=20, help="Fixed hybrid rerank pool.")
    parser.add_argument("--hybrid-combination", choices=["weighted_sum", "rrf"], default="weighted_sum")
    parser.add_argument("--adaptive-max-attempts", type=int, default=3, help="Max route attempts for adaptive QA.")
    parser.add_argument(
        "--adaptive-retry-threshold",
        type=float,
        default=0.82,
        help="Retry when adaptive route quality is below this threshold.",
    )
    parser.add_argument(
        "--llm-fallback-provider",
        choices=["dummy", "openai-compatible"],
        default="dummy",
        help="Provider for routed_grounded_with_* fallback configs.",
    )
    parser.add_argument(
        "--llm-fallback-sufficiency-threshold",
        type=float,
        default=0.72,
        help="Call fallback below this evidence sufficiency threshold when evidence is still relevant.",
    )
    parser.add_argument(
        "--llm-fallback-min-confidence",
        type=float,
        default=0.30,
        help="Minimum accepted grounded fallback confidence.",
    )
    parser.add_argument(
        "--llm-fallback-min-override-confidence",
        type=float,
        default=0.65,
        help="Minimum confidence required when fallback overrides a non-answer standard decision.",
    )
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("queries"), list):
        return payload["queries"]
    raise ValueError(f"{path} must be JSONL, JSON list, or {{queries: [...]}}")


def expand_config_names(raw_names: Iterable[str] | None) -> list[str]:
    names: list[str] = []
    for raw in raw_names or ["routed_grounded"]:
        names.extend(part.strip() for part in raw.split(",") if part.strip())
    if any(name == "all" for name in names):
        return list(DEFAULT_CONFIGS)

    valid = set(DEFAULT_CONFIGS)
    unknown = [name for name in names if name not in valid]
    if unknown:
        raise ValueError(f"Unknown QA benchmark config(s): {', '.join(unknown)}")

    deduped: list[str] = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    return deduped


def fixed_retrieval_config(args: argparse.Namespace, *, use_rerank: bool = False) -> RetrievalConfig:
    return RetrievalConfig(
        top_k=args.top_k,
        candidate_k=max(args.candidate_k, args.top_k),
        bm25_weight=args.bm25_weight,
        dense_weight=args.dense_weight,
        rerank_top_n=args.rerank_top_n if use_rerank else 0,
        combination=args.hybrid_combination,
        use_rerank=use_rerank,
    )


def make_benchmark_llm_fallback(config_name: str, args: argparse.Namespace) -> GroundedLLMFallback:
    enable_table = config_name in {
        "routed_grounded_with_llm_fallback",
        "routed_grounded_with_table_llm",
        "routed_grounded_with_multimodal_textual_fallback",
        "adaptive_route_retry_with_final_route_llm_fallback",
    }
    enable_formula = config_name in {
        "routed_grounded_with_llm_fallback",
        "routed_grounded_with_formula_llm",
        "routed_grounded_with_multimodal_textual_fallback",
        "adaptive_route_retry_with_final_route_llm_fallback",
    }
    enable_figure = config_name in {
        "routed_grounded_with_llm_fallback",
        "routed_grounded_with_multimodal_textual_fallback",
        "adaptive_route_retry_with_final_route_llm_fallback",
    }
    config = LLMFallbackConfig(
        enable_llm_fallback=True,
        enable_table_llm_reasoning=enable_table,
        enable_formula_llm_reasoning=enable_formula,
        enable_figure_llm_reasoning=enable_figure,
        fallback_only_if_grounded_evidence_present=True,
        sufficiency_threshold=args.llm_fallback_sufficiency_threshold,
        min_llm_confidence=args.llm_fallback_min_confidence,
        min_non_answer_override_confidence=args.llm_fallback_min_override_confidence,
    )
    if args.llm_fallback_provider == "openai-compatible":
        client = OpenAICompatibleGroundedLLMClient(timeout_s=config.request_timeout_s)
    else:
        client = DummyGroundedLLMClient()
    return GroundedLLMFallback(config=config, client=client)


def make_pipeline(
    *,
    config_name: str,
    args: argparse.Namespace,
    retrieval_service: RetrievalService,
):
    query_router = QueryRouter()
    fixed_hybrid = fixed_retrieval_config(args)

    if config_name == "bm25_only":
        return GroundedQAPipeline(
            retrieval_service=retrieval_service,
            router=query_router,
            retrieval_planner=FixedRetrievalPlanner(
                strategy="bm25",
                config=fixed_retrieval_config(args),
                reason="BM25-only QA baseline",
            ),
        )

    if config_name == "dense_only":
        return GroundedQAPipeline(
            retrieval_service=retrieval_service,
            router=query_router,
            retrieval_planner=FixedRetrievalPlanner(
                strategy="dense",
                config=fixed_retrieval_config(args),
                reason="Dense-only QA baseline",
            ),
        )

    if config_name == "hybrid_no_routing":
        return GroundedQAPipeline(
            retrieval_service=retrieval_service,
            router=query_router,
            retrieval_planner=FixedRetrievalPlanner(
                strategy="hybrid",
                config=fixed_hybrid,
                reason="Fixed hybrid retrieval without query-aware retrieval routing",
            ),
        )

    if config_name in {
        "routed_grounded_with_llm_fallback",
        "routed_grounded_with_table_llm",
        "routed_grounded_with_formula_llm",
        "routed_grounded_with_multimodal_textual_fallback",
    }:
        return GroundedQAPipeline(
            retrieval_service=retrieval_service,
            router=query_router,
            retrieval_planner=QueryAwareRetrievalPlanner(),
            llm_fallback=make_benchmark_llm_fallback(config_name, args),
        )

    if config_name == "no_evidence_checker":
        return GroundedQAPipeline(
            retrieval_service=retrieval_service,
            router=query_router,
            retrieval_planner=QueryAwareRetrievalPlanner(),
            evidence_checker=PermissiveEvidenceChecker(),  # type: ignore[arg-type]
        )

    if config_name == "adaptive_route_retry":
        return AdaptiveRouteRetryQAPipeline(
            retrieval_service=retrieval_service,
            router=query_router,
            retrieval_planner=QueryAwareRetrievalPlanner(),
            max_attempts=args.adaptive_max_attempts,
            retry_quality_threshold=args.adaptive_retry_threshold,
        )

    if config_name == "adaptive_route_retry_with_final_route_llm_fallback":
        return AdaptiveRouteRetryQAPipeline(
            retrieval_service=retrieval_service,
            router=query_router,
            retrieval_planner=QueryAwareRetrievalPlanner(),
            llm_fallback=make_benchmark_llm_fallback(config_name, args),
            max_attempts=args.adaptive_max_attempts,
            retry_quality_threshold=args.adaptive_retry_threshold,
            enable_final_route_llm_fallback=True,
        )

    if config_name == "no_router":
        return GroundedQAPipeline(
            retrieval_service=retrieval_service,
            router=FixedRouter("ambiguous"),
            retrieval_planner=FixedRetrievalPlanner(
                strategy="hybrid",
                config=fixed_hybrid,
                reason="Router disabled; all queries use one ambiguous-query hybrid plan",
            ),
        )

    if config_name == "no_citation_grounding":
        return GroundedQAPipeline(
            retrieval_service=retrieval_service,
            router=query_router,
            retrieval_planner=QueryAwareRetrievalPlanner(),
            answer_generator=GroundedAnswerGenerator(emit_citations=False),
        )

    if config_name == "no_metadata_filter":
        return GroundedQAPipeline(
            retrieval_service=retrieval_service,
            router=query_router,
            retrieval_planner=NoMetadataFilterPlanner(),
        )

    return GroundedQAPipeline(
        retrieval_service=retrieval_service,
        router=query_router,
        retrieval_planner=QueryAwareRetrievalPlanner(),
    )


def question_text(case: dict[str, Any]) -> str:
    return str(case.get("question") or case.get("query") or "")


def expected_chunk_ids(case: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    if case.get("expected_chunk_id"):
        values.add(str(case["expected_chunk_id"]))
    many = case.get("expected_chunk_ids") or []
    if isinstance(many, str):
        values.add(many)
    else:
        values.update(str(item) for item in many)
    return values


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value if str(item).strip()]
    return [str(value)]


def int_set(value: Any) -> set[int]:
    values: set[int] = set()
    for item in string_list(value):
        try:
            values.add(int(item))
        except ValueError:
            continue
    return values


def should_answer(case: dict[str, Any]) -> bool:
    if "should_answer" in case:
        return bool(case["should_answer"])
    if "answerable" in case:
        return bool(case["answerable"])
    expected_decision = str(case.get("expected_decision") or "").strip().lower()
    if expected_decision in {"abstain", "switch_strategy", "expand_retrieval", "insufficient_evidence"}:
        return False
    return True


def evidence_match(case: dict[str, Any], result_dict: dict[str, Any]) -> bool:
    hits = result_dict["retrieved_hits"]
    citations = result_dict["citations"]
    candidates = [*hits, *citations]
    gold_ids = expected_chunk_ids(case)
    if gold_ids:
        cited_or_hit_ids = {item.get("chunk_id") for item in candidates}
        return bool(gold_ids & cited_or_hit_ids)

    expected_section = str(case.get("expected_section") or case.get("section") or "").lower().strip()
    expected_sections = [
        item.lower().strip()
        for item in [
            *string_list(case.get("expected_sections")),
            *string_list(case.get("gold_section")),
            *string_list(case.get("gold_sections")),
        ]
        if item.strip()
    ]
    if expected_section:
        expected_sections.append(expected_section)
    expected_pages = int_set(case.get("expected_pages")) | int_set(case.get("gold_page")) | int_set(case.get("gold_pages"))
    match_text = str(case.get("match_text") or "").lower().strip()
    expected_source = case.get("source_name")

    for item in candidates:
        if expected_source and item.get("source_name") != expected_source:
            continue
        section_blob = " ".join(
            [
                str(item.get("section") or ""),
                " ".join(str(part) for part in item.get("heading_path") or []),
                str(item.get("snippet") or ""),
                str(item.get("text") or ""),
            ]
        ).lower()
        try:
            item_page = int(item.get("page"))
        except (TypeError, ValueError):
            item_page = None
        page_matches = bool(expected_pages and item_page in expected_pages)
        section_matches = bool(expected_sections and any(section in section_blob for section in expected_sections))
        text_matches = bool(match_text and match_text in section_blob)

        if page_matches or section_matches or text_matches:
            return True
    return False


def answer_match(case: dict[str, Any], answer: str, *, min_token_f1: float) -> tuple[bool, float, bool]:
    gold_answer = str(case.get("gold_answer") or "").strip()
    match_text = str(case.get("match_text") or "").strip()
    contains_expected = contains_text(answer, match_text) if match_text else False
    f1 = token_f1(answer, gold_answer) if gold_answer else 0.0
    if gold_answer:
        return bool(contains_expected or f1 >= min_token_f1), f1, contains_expected
    if match_text:
        return contains_expected, f1, contains_expected
    return False, f1, contains_expected


def answer_supported_by_citations(answer: str, result_dict: dict[str, Any]) -> bool:
    if result_dict["decision"] != "answer":
        return True
    if not result_dict["citations"]:
        return False
    support_sentences = result_dict["evidence"].get("support_sentences") or []
    if support_sentences:
        if any(contains_text(answer, sentence) or contains_text(sentence, answer) for sentence in support_sentences):
            return True
    cited_ids = {citation.get("chunk_id") for citation in result_dict["citations"]}
    cited_text = "\n".join(
        hit.get("text") or ""
        for hit in result_dict["retrieved_hits"]
        if hit.get("chunk_id") in cited_ids
    )
    fallback_trace = result_dict.get("fallback_trace") or {}
    reasoning_mode = str(fallback_trace.get("reasoning_mode") or "")
    if contains_text(cited_text, answer):
        return True
    if reasoning_mode in {"table", "formula", "figure"} and structured_answer_supported(
        answer=answer,
        cited_text=cited_text,
        reasoning_mode=reasoning_mode,
    ):
        return True
    answer_terms = token_set(answer)
    cited_terms = token_set(cited_text)
    return bool(answer_terms) and len(answer_terms & cited_terms) / len(answer_terms) >= 0.45


def structured_answer_supported(*, answer: str, cited_text: str, reasoning_mode: str) -> bool:
    answer_folded = folded_text(answer)
    cited_folded = folded_text(cited_text)
    if not answer_folded or not cited_folded:
        return False

    answer_numbers = re.findall(r"\d+(?:[.,]\d+)?", answer_folded)
    answer_grades = re.findall(r"(?<!\w)[a-f][+-]?(?!\w)", answer_folded)
    if answer_numbers and not all(number in cited_folded for number in answer_numbers):
        return False
    if reasoning_mode == "table" and answer_grades and not all(grade in cited_folded for grade in answer_grades):
        return False

    answer_terms = token_set(answer)
    cited_terms = token_set(cited_text)
    if not answer_terms:
        return False
    overlap = len(answer_terms & cited_terms) / len(answer_terms)
    if reasoning_mode == "table":
        return overlap >= 0.20 or bool(answer_numbers or answer_grades)
    if reasoning_mode == "formula":
        return overlap >= 0.30 or bool(answer_numbers)
    if reasoning_mode == "figure":
        return overlap >= 0.30
    return overlap >= 0.45


def answer_status(*, expected_answerable: bool, decision: str) -> str:
    if expected_answerable and decision == "answer":
        return "answered"
    if expected_answerable:
        return "missed_answer"
    if decision == "answer":
        return "wrongly_answered"
    return "correct_abstention"


def compact_hits(result_dict: dict[str, Any]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for hit in result_dict["retrieved_hits"]:
        compact.append(
            {
                "chunk_id": hit.get("chunk_id"),
                "score": hit.get("final_score", hit.get("score")),
                "bm25_score": hit.get("bm25_score"),
                "dense_score": hit.get("dense_score"),
                "page": hit.get("page"),
                "section": hit.get("section"),
                "snippet": hit.get("snippet"),
            }
        )
    return compact


def evaluate_case(
    case: dict[str, Any],
    result_dict: dict[str, Any],
    *,
    config_name: str,
    min_token_f1: float,
) -> dict[str, Any]:
    expected_answerable = should_answer(case)
    decision = result_dict["decision"]
    if expected_answerable:
        answer_ok, f1, contains_expected = answer_match(
            case,
            result_dict["answer"],
            min_token_f1=min_token_f1,
        )
        evidence_ok = evidence_match(case, result_dict)
    else:
        answer_ok = decision != "answer"
        f1 = 0.0
        contains_expected = False
        evidence_ok = decision != "answer"

    grounded_ok = answer_supported_by_citations(result_dict["answer"], result_dict)
    status = answer_status(expected_answerable=expected_answerable, decision=decision)
    hallucinated = (not expected_answerable and decision == "answer") or (
        expected_answerable and decision == "answer" and not grounded_ok
    )
    retrieval_config = result_dict["retrieval_config"]
    route_attempts = result_dict.get("route_attempts") or []
    selected_route_attempt = int(result_dict.get("selected_route_attempt") or 0)
    initial_route_type = route_attempts[0].get("query_type") if route_attempts else result_dict["query_type"]
    selected_route_type = result_dict["query_type"]
    gold_query_type = str(case.get("query_type") or "").strip().lower()
    initial_route_matches_gold = bool(gold_query_type and str(initial_route_type).lower() == gold_query_type)
    selected_route_matches_gold = bool(gold_query_type and str(selected_route_type).lower() == gold_query_type)
    selected_hit_ids = result_dict["evidence"].get("selected_hit_ids") or []
    citation_chunk_ids = [citation.get("chunk_id") for citation in result_dict["citations"]]
    fallback_trace = result_dict.get("fallback_trace") or {}

    return {
        "config_name": config_name,
        "query_id": case.get("query_id") or case.get("id"),
        "question": question_text(case),
        "query_type": case.get("query_type") or result_dict["query_type"],
        "fallback_category": case.get("fallback_category"),
        "weak_standard_answer_case": bool(case.get("weak_standard_answer_case", False)),
        "expected_modality": case.get("expected_modality"),
        "expected_fallback_mode": case.get("expected_fallback_mode"),
        "should_require_fallback": bool(case.get("should_require_fallback", False)),
        "routed_query_type": result_dict["query_type"],
        "initial_route_type": initial_route_type,
        "selected_route_type": selected_route_type,
        "selected_route_attempt": selected_route_attempt,
        "route_attempt_count": len(route_attempts) or 1,
        "route_retry_used": (len(route_attempts) or 1) > 1,
        "route_changed": str(initial_route_type) != str(selected_route_type),
        "route_initial_matches_gold": initial_route_matches_gold,
        "route_selected_matches_gold": selected_route_matches_gold,
        "should_answer": expected_answerable,
        "expected_decision": case.get("expected_decision") or ("answer" if expected_answerable else "abstain"),
        "decision": decision,
        "answer_status": status,
        "retrieval_strategy": result_dict["retrieval_strategy"],
        "retrieval_top_k": retrieval_config.get("top_k"),
        "retrieval_candidate_k": retrieval_config.get("candidate_k"),
        "retrieval_combination": retrieval_config.get("combination"),
        "metadata_filters": {
            "block_type_filter": retrieval_config.get("block_type_filter") or [],
            "section_filter": retrieval_config.get("section_filter") or [],
            "doc_id_filter": retrieval_config.get("doc_id_filter") or [],
            "source_name_filter": retrieval_config.get("source_name_filter") or [],
            "metadata_filters": retrieval_config.get("metadata_filters") or {},
        },
        "answer": result_dict["answer"],
        "standard_answer": result_dict.get("standard_answer"),
        "final_answer_source": result_dict.get("final_answer_source") or "standard",
        "fallback_called": bool(fallback_trace.get("called", False)),
        "fallback_used": bool(fallback_trace.get("used", False)),
        "fallback_reason": fallback_trace.get("reason"),
        "fallback_reasoning_mode": fallback_trace.get("reasoning_mode"),
        "fallback_decision": fallback_trace.get("decision"),
        "fallback_confidence": fallback_trace.get("confidence", 0.0),
        "override_confidence": fallback_trace.get("confidence", 0.0),
        "fallback_latency_ms": fallback_trace.get("latency_ms", 0.0),
        "fallback_provider": fallback_trace.get("provider"),
        "provider_name": fallback_trace.get("provider") or "standard",
        "fallback_llm_called": bool(fallback_trace.get("llm_called", False)),
        "fallback_answer": fallback_trace.get("answer"),
        "fallback_used_evidence_ids": fallback_trace.get("used_evidence_ids") or [],
        "gold_answer": case.get("gold_answer"),
        "match_text": case.get("match_text"),
        "answer_match": answer_ok,
        "answer_token_f1": f1,
        "answer_contains_expected": contains_expected,
        "evidence_match": evidence_ok,
        "grounded_answer": grounded_ok,
        "hallucinated": hallucinated,
        "end_to_end_success": bool(
            (expected_answerable and answer_ok and evidence_ok and grounded_ok and decision == "answer")
            or (not expected_answerable and decision != "answer")
        ),
        "citation_count": len(result_dict["citations"]),
        "citation_chunk_ids": citation_chunk_ids,
        "selected_hit_ids": selected_hit_ids,
        "selected_hits": compact_hits(result_dict),
        "top_hit_chunk_id": result_dict["retrieved_hits"][0]["chunk_id"] if result_dict["retrieved_hits"] else None,
        "top_hit_section": result_dict["retrieved_hits"][0].get("section") if result_dict["retrieved_hits"] else None,
        "sufficiency": result_dict["evidence"]["sufficiency"],
        "evidence_relevance": result_dict["evidence"]["relevance"],
        "evidence_coverage": result_dict["evidence"]["coverage"],
        "evidence_grounding": result_dict["evidence"]["grounding"],
        "evidence_reason": result_dict["evidence"]["reason"],
        "support_sentences": result_dict["evidence"].get("support_sentences") or [],
        "route_attempts": route_attempts,
        "retrieval_latency_ms": result_dict["retrieval_latency_ms"],
        "answer_latency_ms": result_dict["answer_latency_ms"],
        "total_latency_ms": result_dict["total_latency_ms"],
        "route_trace": {
            "initial_query_type": initial_route_type,
            "selected_query_type": selected_route_type,
            "selected_route_attempt": selected_route_attempt,
            "route_attempt_count": len(route_attempts) or 1,
            "retrieval_strategy": result_dict["retrieval_strategy"],
            "top_k": retrieval_config.get("top_k"),
            "selected_hit_ids": selected_hit_ids,
            "evidence_decision": decision,
            "answer_status": status,
            "fallback_called": bool(fallback_trace.get("called", False)),
            "fallback_reasoning_mode": fallback_trace.get("reasoning_mode"),
            "final_answer_source": result_dict.get("final_answer_source") or "standard",
        },
    }


def mean_float(values: Iterable[float]) -> float:
    values_list = list(values)
    return statistics.mean(values_list) if values_list else 0.0


def ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def folded_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_text.lower().split())


def row_has_table_signal(row: dict[str, Any]) -> bool:
    text = folded_text(
        " ".join(
            [
                str(row.get("question") or ""),
                str(row.get("fallback_reasoning_mode") or ""),
                str(row.get("expected_modality") or ""),
            ]
        )
    )
    return any(term in text for term in ("table", "bang", "row", "column", "cell", "tuong ung"))


def row_has_formula_signal(row: dict[str, Any]) -> bool:
    text = folded_text(
        " ".join(
            [
                str(row.get("question") or ""),
                str(row.get("fallback_reasoning_mode") or ""),
                str(row.get("expected_modality") or ""),
            ]
        )
    )
    return any(term in text for term in ("formula", "equation", "cong thuc", "symbol", "metric", "ffn("))


def row_has_figure_signal(row: dict[str, Any]) -> bool:
    text = folded_text(
        " ".join(
            [
                str(row.get("question") or ""),
                str(row.get("fallback_reasoning_mode") or ""),
                str(row.get("expected_modality") or ""),
            ]
        )
    )
    return any(term in text for term in ("figure", "fig.", "chart", "image", "caption", "bieu do", "hinh"))


def summarize_flat(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}

    answerable = [row for row in rows if row["should_answer"]]
    unanswerable = [row for row in rows if not row["should_answer"]]
    abstentions = [row for row in rows if row["decision"] != "answer"]
    correct_abstentions = [row for row in unanswerable if row["decision"] != "answer"]
    answerable_abstentions = [row for row in answerable if row["decision"] != "answer"]
    false_answers = [row for row in unanswerable if row["decision"] == "answer"]
    rows_with_gold_route = [row for row in rows if row.get("query_type")]
    fallback_called = [row for row in rows if row.get("fallback_called")]
    fallback_used = [row for row in rows if row.get("fallback_used")]
    fallback_answered = [row for row in rows if row.get("final_answer_source") != "standard"]
    table_rows = [row for row in rows if row_has_table_signal(row)]
    formula_rows = [row for row in rows if row_has_formula_signal(row)]
    figure_rows = [row for row in rows if row_has_figure_signal(row)]
    table_rule_resolved = [
        row
        for row in table_rows
        if row.get("final_answer_source") == "table_rule_fallback" and row.get("end_to_end_success")
    ]
    table_llm_resolved = [
        row
        for row in table_rows
        if row.get("fallback_llm_called")
        and row.get("fallback_used")
        and row.get("fallback_reasoning_mode") == "table"
        and row.get("end_to_end_success")
    ]
    fallback_helped = [row for row in rows if row.get("fallback_used") and row.get("end_to_end_success")]

    return {
        "query_count": len(rows),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "answer_rate": mean_float(1.0 if row["decision"] == "answer" else 0.0 for row in rows),
        "answer_match_rate": mean_float(float(row["answer_match"]) for row in rows),
        "answerable_answer_match_rate": mean_float(float(row["answer_match"]) for row in answerable),
        "evidence_match_rate": mean_float(float(row["evidence_match"]) for row in rows),
        "answerable_evidence_match_rate": mean_float(float(row["evidence_match"]) for row in answerable),
        "grounded_rate": mean_float(float(row["grounded_answer"]) for row in rows),
        "answerable_grounded_rate": mean_float(float(row["grounded_answer"]) for row in answerable),
        "end_to_end_success_rate": mean_float(float(row["end_to_end_success"]) for row in rows),
        "avg_answer_token_f1": mean_float(float(row["answer_token_f1"]) for row in answerable),
        "avg_sufficiency": mean_float(float(row["sufficiency"]) for row in rows),
        "avg_total_latency_ms": mean_float(float(row["total_latency_ms"]) for row in rows),
        "avg_retrieval_latency_ms": mean_float(float(row["retrieval_latency_ms"]) for row in rows),
        "abstention_count": len(abstentions),
        "correct_abstention_count": len(correct_abstentions),
        "answerable_abstention_count": len(answerable_abstentions),
        "false_answer_count": len(false_answers),
        "abstention_precision": ratio(len(correct_abstentions), len(abstentions)),
        "abstention_recall": ratio(len(correct_abstentions), len(unanswerable)),
        "hallucination_rate": mean_float(float(row["hallucinated"]) for row in rows),
        "avg_route_attempt_count": mean_float(float(row.get("route_attempt_count") or 1) for row in rows),
        "route_retry_rate": mean_float(float(row.get("route_retry_used", False)) for row in rows),
        "route_changed_rate": mean_float(float(row.get("route_changed", False)) for row in rows),
        "route_initial_match_rate": mean_float(float(row.get("route_initial_matches_gold", False)) for row in rows_with_gold_route),
        "route_selected_match_rate": mean_float(float(row.get("route_selected_matches_gold", False)) for row in rows_with_gold_route),
        "llm_fallback_call_rate": mean_float(float(row.get("fallback_called", False)) for row in rows),
        "llm_fallback_llm_call_rate": mean_float(float(row.get("fallback_llm_called", False)) for row in rows),
        "llm_fallback_used_rate": mean_float(float(row.get("fallback_used", False)) for row in rows),
        "llm_fallback_success_gain": mean_float(
            float(row["end_to_end_success"] and row.get("fallback_used", False)) for row in rows
        ),
        "llm_fallback_latency_overhead_ms": mean_float(float(row.get("fallback_latency_ms") or 0.0) for row in fallback_called),
        "fallback_helped_count": len(
            [row for row in fallback_answered if row["end_to_end_success"] and row.get("fallback_used")]
        ),
        "fallback_helped_rate": mean_float(float(row.get("fallback_used", False) and row.get("end_to_end_success", False)) for row in rows),
        "fallback_override_count": len(fallback_answered),
        "fallback_hallucination_rate": mean_float(float(row["hallucinated"]) for row in fallback_called),
        "table_question_success": mean_float(float(row["end_to_end_success"]) for row in table_rows),
        "table_rule_resolved_count": len(table_rule_resolved),
        "table_llm_resolved_count": len(table_llm_resolved),
        "table_total_success": mean_float(float(row["end_to_end_success"]) for row in table_rows),
        "formula_question_success": mean_float(float(row["end_to_end_success"]) for row in formula_rows),
        "figure_caption_question_success": mean_float(float(row["end_to_end_success"]) for row in figure_rows),
        "fallback_used_count": len(fallback_used),
        "fallback_success_count": len(fallback_helped),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_flat(rows)
    by_query_type: dict[str, dict[str, Any]] = {}
    for query_type in sorted({str(row["query_type"]) for row in rows}):
        by_query_type[query_type] = summarize_flat([row for row in rows if row["query_type"] == query_type])
    summary["by_query_type"] = by_query_type
    return summary


def ablation_deltas(config_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    routed = config_summaries.get("routed_grounded")
    if not routed:
        return deltas

    for baseline_name in ("bm25_only", "dense_only", "hybrid_no_routing", "no_evidence_checker", "no_router"):
        baseline = config_summaries.get(baseline_name)
        if not baseline:
            continue
        deltas[f"routed_grounded_vs_{baseline_name}"] = {
            "end_to_end_success_delta": routed.get("end_to_end_success_rate", 0.0)
            - baseline.get("end_to_end_success_rate", 0.0),
            "grounded_rate_delta": routed.get("grounded_rate", 0.0) - baseline.get("grounded_rate", 0.0),
            "hallucination_rate_delta": routed.get("hallucination_rate", 0.0) - baseline.get("hallucination_rate", 0.0),
            "avg_total_latency_ms_delta": routed.get("avg_total_latency_ms", 0.0)
            - baseline.get("avg_total_latency_ms", 0.0),
        }
    adaptive = config_summaries.get("adaptive_route_retry")
    if adaptive:
        deltas["adaptive_route_retry_vs_routed_grounded"] = {
            "end_to_end_success_delta": adaptive.get("end_to_end_success_rate", 0.0)
            - routed.get("end_to_end_success_rate", 0.0),
            "answer_match_delta": adaptive.get("answer_match_rate", 0.0) - routed.get("answer_match_rate", 0.0),
            "evidence_match_delta": adaptive.get("evidence_match_rate", 0.0) - routed.get("evidence_match_rate", 0.0),
            "grounded_rate_delta": adaptive.get("grounded_rate", 0.0) - routed.get("grounded_rate", 0.0),
            "hallucination_rate_delta": adaptive.get("hallucination_rate", 0.0)
            - routed.get("hallucination_rate", 0.0),
            "avg_total_latency_ms_delta": adaptive.get("avg_total_latency_ms", 0.0)
            - routed.get("avg_total_latency_ms", 0.0),
            "route_retry_rate_delta": adaptive.get("route_retry_rate", 0.0) - routed.get("route_retry_rate", 0.0),
            "route_selected_match_delta": adaptive.get("route_selected_match_rate", 0.0)
            - routed.get("route_selected_match_rate", 0.0),
        }
    for fallback_name in (
        "routed_grounded_with_llm_fallback",
        "routed_grounded_with_table_llm",
        "routed_grounded_with_formula_llm",
        "routed_grounded_with_multimodal_textual_fallback",
        "adaptive_route_retry_with_final_route_llm_fallback",
    ):
        fallback = config_summaries.get(fallback_name)
        if not fallback:
            continue
        deltas[f"{fallback_name}_vs_routed_grounded"] = {
            "llm_fallback_success_gain": fallback.get("end_to_end_success_rate", 0.0)
            - routed.get("end_to_end_success_rate", 0.0),
            "answer_match_delta": fallback.get("answer_match_rate", 0.0) - routed.get("answer_match_rate", 0.0),
            "grounded_rate_delta": fallback.get("grounded_rate", 0.0) - routed.get("grounded_rate", 0.0),
            "hallucination_rate_delta": fallback.get("hallucination_rate", 0.0)
            - routed.get("hallucination_rate", 0.0),
            "avg_total_latency_ms_delta": fallback.get("avg_total_latency_ms", 0.0)
            - routed.get("avg_total_latency_ms", 0.0),
            "fallback_call_rate": fallback.get("llm_fallback_call_rate", 0.0),
            "fallback_helped_count": fallback.get("fallback_helped_count", 0),
        }
    return deltas


def json_for_csv(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_outputs(output_dir: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "qa_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "per_question.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if rows:
        with (output_dir / "per_question.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json_for_csv(value) for key, value in row.items()})

    lines = [
        "# QA Benchmark",
        "",
        f"- Index: `{summary.get('index_dir')}`",
        f"- Queries: `{summary.get('queries_file')}`",
        "",
        "| Config | Queries | Success | Answer match | Evidence match | Grounded | Hallucination | Retry rate | Avg attempts | Avg latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config_name, config_summary in summary.get("configs", {}).items():
        lines.append(
            "| {name} | {count} | {success:.3f} | {answer:.3f} | {evidence:.3f} | {grounded:.3f} | {hallucination:.3f} | {retry:.3f} | {attempts:.2f} | {latency:.1f} |".format(
                name=config_name,
                count=config_summary.get("query_count", 0),
                success=config_summary.get("end_to_end_success_rate", 0.0),
                answer=config_summary.get("answer_match_rate", 0.0),
                evidence=config_summary.get("evidence_match_rate", 0.0),
                grounded=config_summary.get("grounded_rate", 0.0),
                hallucination=config_summary.get("hallucination_rate", 0.0),
                retry=config_summary.get("route_retry_rate", 0.0),
                attempts=config_summary.get("avg_route_attempt_count", 0.0),
                latency=config_summary.get("avg_total_latency_ms", 0.0),
            )
        )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (RESULTS_ROOT / timestamp)
    config_names = expand_config_names(args.config)

    retrieval_service = RetrievalService.from_index(
        args.index_dir,
        reranker=make_reranker(args.reranker, model_name=args.reranker_model),
    )
    cases = load_cases(args.queries)

    rows: list[dict[str, Any]] = []
    for config_name in config_names:
        pipeline = make_pipeline(
            config_name=config_name,
            args=args,
            retrieval_service=retrieval_service,
        )
        if cases and not args.no_warmup:
            pipeline.answer(question_text(cases[0]))

        for case in cases:
            result = pipeline.answer(question_text(case))
            result_dict = result.to_dict()
            rows.append(
                evaluate_case(
                    case,
                    result_dict,
                    config_name=config_name,
                    min_token_f1=args.min_token_f1,
                )
            )

    config_summaries = {
        config_name: summarize([row for row in rows if row["config_name"] == config_name])
        for config_name in config_names
    }
    summary = {
        "timestamp_utc": timestamp,
        "index_dir": str(args.index_dir),
        "queries_file": str(args.queries),
        "config_names": config_names,
        "configs": config_summaries,
        "ablation_deltas": ablation_deltas(config_summaries),
    }
    if len(config_names) == 1:
        summary.update(config_summaries[config_names[0]])

    write_outputs(output_dir, summary, rows)
    print(output_dir)


if __name__ == "__main__":
    main()
