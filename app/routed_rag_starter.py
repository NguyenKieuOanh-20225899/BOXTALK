from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.loaders.pdf_loader import PDFLoader
from app.qa.pipeline import GroundedQAPipeline
from app.retrieval.reranker import make_reranker
from app.retrieval.route_planner import QueryAwareRetrievalPlanner
from app.retrieval.schemas import RetrievedHit
from app.retrieval.service import RetrievalService



# Data models

class QueryType(str, Enum):
    FACTOID = "factoid"
    DEFINITION = "definition"
    POLICY = "policy"
    COMPARISON = "comparison"
    PROCEDURAL = "procedural"
    MULTI_HOP = "multi_hop"
    AMBIGUOUS = "ambiguous"


class RouteAction(str, Enum):
    ANSWER = "answer"
    EXPAND_RETRIEVAL = "expand_retrieval"
    SWITCH_STRATEGY = "switch_strategy"
    ABSTAIN = "abstain"


@dataclass
class EvidenceReport:
    relevance: float
    coverage: float
    consistency: float
    citation_support: float
    sufficiency: float
    decision: RouteAction
    reason: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)


class AskResponse(BaseModel):
    question: str
    query_type: QueryType
    route_action: RouteAction
    answer: str
    evidence_report: Dict[str, Any]
    citations: List[Dict[str, Any]]
    retrieved_chunks: List[Dict[str, Any]]


# Corpus

class InMemoryCorpus:
    def __init__(self) -> None:
        self.chunks: List[Any] = []

    def add_chunks(self, chunks: List[Any]) -> None:
        self.chunks.extend(chunks)

    def all_chunks(self) -> List[Any]:
        return self.chunks



# Query-aware router


class QueryRouter:
    POLICY_TERMS = {
        "quy định", "quy chế", "điều kiện", "bắt buộc", "được phép", "không được",
        "có được", "được mang", "cấm", "nội quy",
        "policy", "regulation", "rule", "requirement", "must", "may", "eligible"
    }
    PROCEDURAL_TERMS = {
        "cách", "làm sao", "quy trình", "thủ tục", "hướng dẫn", "bước",
        "how", "procedure", "process", "steps", "apply", "submit"
    }
    COMPARISON_TERMS = {
        "so sánh", "khác nhau", "giữa", "compare", "difference", "versus", "vs"
    }
    DEFINITION_TERMS = {
        "là gì", "định nghĩa", "what is", "meaning", "khái niệm", "mô tả"
    }

    def route(self, question: str) -> QueryType:
        q = question.lower().strip()

        if any(term in q for term in self.COMPARISON_TERMS):
            return QueryType.COMPARISON
        if any(term in q for term in self.PROCEDURAL_TERMS):
            return QueryType.PROCEDURAL
        if any(term in q for term in self.POLICY_TERMS):
            return QueryType.POLICY
        if any(term in q for term in self.DEFINITION_TERMS):
            return QueryType.DEFINITION

        multi_hop_signals = [
            "đồng thời",
            "điều kiện tiên quyết",
            "prerequisite",
            "bao gồm những bước nào và",
            "vừa",
        ]
        if any(term in q for term in multi_hop_signals) and len(q.split()) > 10:
            return QueryType.MULTI_HOP

        if len(q.split()) <= 6:
            return QueryType.FACTOID

        return QueryType.AMBIGUOUS



# Evidence-aware routing

class EvidenceChecker:
    WORD_RE = re.compile(r"\w+", re.UNICODE)

    def assess(self, question: str, hits: List[RetrievedHit], query_type: QueryType) -> EvidenceReport:
        if not hits:
            return EvidenceReport(
                relevance=0.0,
                coverage=0.0,
                consistency=0.0,
                citation_support=0.0,
                sufficiency=0.0,
                decision=RouteAction.SWITCH_STRATEGY,
                reason="Không có đoạn nào đủ liên quan. Cần đổi chiến lược retrieval hoặc mở rộng chỉ mục.",
            )

        top_scores = [h.hybrid_score for h in hits[:3]]
        top1 = hits[0].hybrid_score
        top2 = hits[1].hybrid_score if len(hits) > 1 else 0.0
        top_gap = max(0.0, top1 - top2)

        relevance = min(1.0, (0.55 * top1 + 0.30 * (sum(top_scores) / len(top_scores)) + 0.15 * min(1.0, top_gap)))

        q_terms = {t.lower() for t in self.WORD_RE.findall(question)}
        covered_terms = set()
        for h in hits[:3]:
            covered_terms |= q_terms & {t.lower() for t in self.WORD_RE.findall(h.chunk.text)}
        coverage = len(covered_terms) / max(1, len(q_terms))

        sections = {(h.chunk.section or "") for h in hits[:3]}
        consistency = 1.0 if len(sections) <= 2 else 0.7

        citation_support = 1.0 if hits[0].chunk.page is not None and hits[0].chunk.section is not None else 0.6

        if query_type == QueryType.POLICY and top_gap >= 0.35:
            sufficiency = 0.80 * relevance + 0.10 * coverage + 0.05 * consistency + 0.05 * citation_support
        else:
            sufficiency = 0.35 * relevance + 0.30 * coverage + 0.20 * consistency + 0.15 * citation_support

        if query_type == QueryType.POLICY and top_gap >= 0.35 and top1 >= 0.65:
            decision = RouteAction.ANSWER
            reason = "Top-1 evidence áp đảo và đủ mạnh cho câu hỏi policy."
        elif sufficiency >= 0.72:
            decision = RouteAction.ANSWER
            reason = "Bằng chứng đủ mạnh để trả lời có trích dẫn."
        elif relevance >= 0.45 and coverage < 0.50:
            decision = RouteAction.EXPAND_RETRIEVAL
            reason = "Đã có tín hiệu đúng chủ đề nhưng chưa phủ đủ ý. Nên mở rộng truy hồi."
        elif relevance < 0.30:
            decision = RouteAction.SWITCH_STRATEGY
            reason = "Kết quả retrieval hiện tại yếu. Nên đổi strategy hoặc filter metadata khác."
        else:
            decision = RouteAction.ABSTAIN
            reason = "Chưa đủ căn cứ đáng tin cậy để sinh câu trả lời chắc chắn."

        return EvidenceReport(
            relevance=round(relevance, 3),
            coverage=round(coverage, 3),
            consistency=round(consistency, 3),
            citation_support=round(citation_support, 3),
            sufficiency=round(sufficiency, 3),
            decision=decision,
            reason=reason,
        )



# Answer synthesis

class AnswerGenerator:
    def _select_top_hits(self, query_type: QueryType, hits: List[RetrievedHit]) -> List[RetrievedHit]:
        if not hits:
            return []

        if query_type == QueryType.POLICY:
            if len(hits) == 1:
                return hits[:1]
            top1 = hits[0].hybrid_score
            top2 = hits[1].hybrid_score
            if top1 >= 1.5 * max(top2, 1e-6):
                return hits[:1]
            return hits[:2]

        if query_type in {QueryType.PROCEDURAL, QueryType.COMPARISON, QueryType.MULTI_HOP}:
            return hits[:3]

        return hits[:2]

    def _is_yes_no_question(self, question: str) -> bool:
        q = question.lower().strip()
        patterns = [
            "có được",
            "được không",
            "có phải",
            "có cần",
            "có bắt buộc",
            "có thể",
            "không được",
        ]
        return any(p in q for p in patterns)

    def _policy_yes_no_answer(self, question: str, text: str, item_number: Optional[int], section: Optional[str]) -> str:
        t = text.lower()
        prefix = ""
        if item_number is not None:
            prefix = f"Theo mục {item_number}"
            if section:
                prefix += f" của phần '{section}'"
            prefix += ": "
        elif section:
            prefix = f"Theo phần '{section}': "

        negative_markers = [
            "không mang",
            "không được",
            "cấm",
            "nghiêm cấm",
            "không cho phép",
        ]
        positive_markers = [
            "được phép",
            "có thể",
        ]

        if self._is_yes_no_question(question):
            if any(m in t for m in negative_markers):
                return f"Không. {prefix}{text}"
            if any(m in t for m in positive_markers):
                return f"Có. {prefix}{text}"

        return f"{prefix}{text}"

    def generate(
        self,
        question: str,
        query_type: QueryType,
        hits: List[RetrievedHit],
        evidence: EvidenceReport,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        selected_hits = self._select_top_hits(query_type, hits)

        citations = [
            {
                "chunk_id": h.chunk.chunk_id,
                "source_name": h.chunk.source_name,
                "page": h.chunk.page,
                "section": h.chunk.section,
                "item_number": h.chunk.metadata.get("item_number"),
            }
            for h in selected_hits
        ]

        if evidence.decision == RouteAction.ABSTAIN:
            return (
                "Mình chưa thể trả lời chắc chắn vì bằng chứng hiện tại chưa đủ. Bạn nên tải thêm tài liệu liên quan hoặc mở rộng truy hồi.",
                citations,
            )

        if evidence.decision == RouteAction.EXPAND_RETRIEVAL:
            return (
                "Mình đã tìm thấy một phần thông tin liên quan nhưng chưa đủ để kết luận đầy đủ. Ở bước tiếp theo, hệ thống nên tăng top-k, mở rộng section, hoặc đổi filter metadata.",
                citations,
            )

        if evidence.decision == RouteAction.SWITCH_STRATEGY:
            return (
                "Truy hồi hiện tại chưa đúng trọng tâm câu hỏi. Nên chuyển chiến lược, ví dụ ưu tiên section quy định, bảng, hoặc truy hồi dense mạnh hơn.",
                citations,
            )

        if not selected_hits:
            return ("Mình chưa tìm thấy đoạn phù hợp để trả lời.", citations)

        top = selected_hits[0]
        top_text = top.chunk.text.strip()
        item_number = top.chunk.metadata.get("item_number")
        section = top.chunk.section

        if query_type == QueryType.POLICY:
            return self._policy_yes_no_answer(question, top_text, item_number, section), citations[:1]

        if query_type == QueryType.PROCEDURAL:
            steps = []
            for i, h in enumerate(selected_hits, start=1):
                step_text = h.chunk.text.strip()
                item_no = h.chunk.metadata.get("item_number")
                label = f"Bước {i}"
                if item_no is not None:
                    label = f"Mục {item_no}"
                steps.append(f"{label}: {step_text}")
            return "Theo tài liệu, các bước/thông tin liên quan gồm:\n" + "\n".join(steps), citations

        if query_type == QueryType.DEFINITION:
            return f"Theo tài liệu: {top_text}", citations[:1]

        if query_type == QueryType.COMPARISON:
            lines = [f"- {h.chunk.text.strip()}" for h in selected_hits]
            return "Các thông tin liên quan để so sánh gồm:\n" + "\n".join(lines), citations

        return f"Dựa trên tài liệu: {top_text}", citations[:1]



# Orchestrator

class RoutedRAGService:
    def __init__(self, pdf_path: str | None = None, index_dir: str | None = None) -> None:
        self.corpus = InMemoryCorpus()
        self.router = QueryRouter()
        self.retrieval_planner = QueryAwareRetrievalPlanner()
        if index_dir:
            self.retrieval_service = self._load_retrieval_index(index_dir)
        else:
            if not pdf_path:
                raise ValueError("RoutedRAGService requires either pdf_path or index_dir")
            self._load_pdf_data(pdf_path)
            self.retrieval_service = self._build_retrieval_service()
        self.qa_pipeline = GroundedQAPipeline(
            retrieval_service=self.retrieval_service,
            router=self.router,
            retrieval_planner=self.retrieval_planner,
        )
        self.evidence_checker = EvidenceChecker()
        self.answer_generator = AnswerGenerator()

    def ask(self, question: str) -> AskResponse:
        qa_result = self.qa_pipeline.answer(question)
        query_type = QueryType(qa_result.query_type)
        route_action = RouteAction(qa_result.decision)
        evidence_report = qa_result.evidence.to_dict()
        evidence_report.update(
            {
                "retrieval_strategy": qa_result.retrieval_strategy,
                "retrieval_latency_ms": round(qa_result.retrieval_latency_ms, 3),
                "answer_latency_ms": round(qa_result.answer_latency_ms, 3),
                "total_latency_ms": round(qa_result.total_latency_ms, 3),
                "retrieval_config": qa_result.retrieval_config.to_dict(),
            }
        )

        return AskResponse(
            question=question,
            query_type=query_type,
            route_action=route_action,
            answer=qa_result.answer,
            evidence_report=evidence_report,
            citations=qa_result.citations,
            retrieved_chunks=[
                {
                    "chunk_id": h.chunk.chunk_id,
                    "source_name": h.chunk.source_name,
                    "page": h.chunk.page,
                    "section": h.chunk.section,
                    "heading_path": h.chunk.heading_path,
                    "block_type": h.chunk.block_type,
                    "order": h.chunk.order,
                    "metadata": h.chunk.metadata,
                    "bm25_score": round(h.bm25_score, 3),
                    "dense_score": round(h.dense_score, 3),
                    "rerank_score": round(h.rerank_score, 3),
                    "hybrid_score": round(h.hybrid_score, 3),
                    "source": h.source,
                    "rank": h.rank,
                    "source_scores": h.source_scores,
                    "text": h.chunk.text,
                }
                for h in qa_result.retrieved_hits
            ],
        )

    def _load_pdf_data(self, pdf_path: str) -> None:
        loader = PDFLoader()
        chunks = loader.load_pdf(pdf_path)
        self.corpus.add_chunks(chunks)

    def _load_retrieval_index(self, index_dir: str) -> RetrievalService:
        reranker_name = os.getenv("BOXTALK_ROUTED_RAG_RERANKER", "heuristic")
        retrieval_service = RetrievalService.from_index(index_dir, reranker=make_reranker(reranker_name))
        self.corpus.add_chunks(retrieval_service.retriever.chunks)
        return retrieval_service

    def _build_retrieval_service(self) -> RetrievalService:
        build_dense = os.getenv("BOXTALK_ROUTED_RAG_BUILD_DENSE", "1").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        reranker_name = os.getenv("BOXTALK_ROUTED_RAG_RERANKER", "heuristic")
        return RetrievalService.from_chunks(
            self.corpus.all_chunks(),
            model_name=os.getenv("BOXTALK_EMBED_MODEL_NAME"),
            build_dense=build_dense,
            reranker=make_reranker(reranker_name),
        )


# FastAPI app

app = FastAPI(title="Routed RAG Starter", version="0.2.0")
PDF_PATH = os.getenv("PDF_PATH", "data/sample.pdf")
INDEX_DIR = os.getenv("BOXTALK_RETRIEVAL_INDEX_DIR")
service = RoutedRAGService(pdf_path=PDF_PATH, index_dir=INDEX_DIR)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    return service.ask(req.question)


@app.get("/debug/chunks")
def debug_chunks(limit: int = 20):
    chunks = service.corpus.all_chunks()
    return {
        "total_chunks": len(chunks),
        "samples": [
            {
                "chunk_id": c.chunk_id,
                "source_name": c.source_name,
                "page": c.page,
                "section": c.section,
                "heading_path": c.heading_path,
                "block_type": c.block_type,
                "order": c.order,
                "metadata": c.metadata,
                "text": c.text,
            }
            for c in chunks[:limit]
        ],
    }


@app.get("/debug/retrieval-plan")
def debug_retrieval_plan(question: str) -> Dict[str, Any]:
    query_type = service.router.route(question)
    plan = service.retrieval_planner.plan(query_type.value, question)
    return {
        "question": question,
        "query_type": query_type,
        "strategy": plan.strategy,
        "reason": plan.reason,
        "config": plan.config.to_dict(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.routed_rag_starter:app", host="0.0.0.0", port=8000, reload=True)
