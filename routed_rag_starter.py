from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI
from pydantic import BaseModel, Field



# Data models
class QueryType(str, Enum):
    FACTOID = "factoid" # cau hoi ve su that ngan gon don gian
    DEFINITION = "definition"
    POLICY = "policy" # cau hoi ve quy dinh, chinh sach
    COMPARISON = "comparison" #so sanh
    PROCEDURAL = "procedural" # quy trinh
    MULTI_HOP = "multi_hop"# cau hoi phuc tap, can nhieu buoc suy luan
    AMBIGUOUS = "ambiguous"  # cau hoi mo ho

class RouteAction(str, Enum):
    ANSWER = "answer" # tra loi cau hoi
    EXPAND_RETRIEVAL = "expand_retrieval" # mo rong truy hoi hien tai
    SWITCH_STRATEGY = "switch_strategy" # doi chien luoc truy hoi, vd: uu tien section quy dinh, bang bieu, hoac tang trong so truy hoi dense
    ABSTAIN = "abstain" # chua the quyet dinh, can them bang chung hoac thong tin tu nguoi dung de co the tra loi chac chan hon

@dataclass
class DocumentChunk:
    chunk_id: str
    text: str
    source_name: str
    page: Optional[int] = None
    section: Optional[str] = None
    block_type: str = "paragraph"  # paragraph | table | heading | list
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RetrievalHit:
    chunk: DocumentChunk #moi chunk di kem 1 diem so
    bm25_score: float = 0.0
    dense_score: float = 0.0
    rerank_score: float = 0.0

    @property
    def hybrid_score(self) -> float:
        return 0.45 * self.bm25_score + 0.45 * self.dense_score + 0.10 * self.rerank_score

@dataclass
class EvidenceReport:
    relevance: float #mức liên quan của evidence với câu hỏi
    coverage: float  # mức bao phủ: evidence đã đủ khía cạnh cần thiết chưa
    consistency: float #độ nhất quán giữa các evidence, có mâu thuẫn không
    citation_support: float #mức hỗ trợ trích dẫn: evidence có cung cấp bằng chứng cho câu trả lời không
    sufficiency: float # mức đủ bằng chứng để trả lời
    decision: RouteAction # hành động router chọn, ví dụ: answer / expand / rerank / abstain
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

# In-memory document store
class InMemoryCorpus:
    def __init__(self) -> None:
        self.chunks: List[DocumentChunk] = []

    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        self.chunks.extend(chunks)

    def all_chunks(self) -> List[DocumentChunk]:
        return self.chunks

# Router: query-aware
class QueryRouter:
    """Rule-based starter router. Replace later with a classifier/LLM router."""

    POLICY_TERMS = {
        "quy định", "quy chế", "điều kiện", "bắt buộc", "được phép", "không được",
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
        "là gì", "định nghĩa", "what is", "meaning", "khái niệm"
    }
    MULTI_HOP_TERMS = {
        "đồng thời", "và", "nếu", "when", "and", "prerequisite", "điều kiện tiên quyết"
    }

    def route(self, question: str) -> QueryType:
        q = question.lower()
        if any(term in q for term in self.COMPARISON_TERMS):
            return QueryType.COMPARISON
        if any(term in q for term in self.PROCEDURAL_TERMS):
            return QueryType.PROCEDURAL
        if any(term in q for term in self.POLICY_TERMS):
            return QueryType.POLICY
        if any(term in q for term in self.DEFINITION_TERMS):
            return QueryType.DEFINITION
        if any(term in q for term in self.MULTI_HOP_TERMS) and len(q.split()) > 8:
            return QueryType.MULTI_HOP
        if len(q.split()) <= 4:
            return QueryType.FACTOID
        return QueryType.AMBIGUOUS

# Retrieval stack
class HybridRetriever:
    """
    Bộ truy hồi lai (hybrid retriever) bản tối giản để demo.

    Ý tưởng:
    - BM25 score: ở đây chưa dùng BM25 thật, mà chỉ xấp xỉ bằng độ trùng từ khóa.
    - Dense score: chưa dùng embedding thật, mà xấp xỉ bằng Jaccard similarity giữa token câu hỏi và token chunk.
    - rerank/structure bonus: cộng điểm ưu tiên theo loại câu hỏi và cấu trúc chunk.
    """
    #Ví dụ: "Điều 12, hợp đồng" -> ["Điều", "12", "hợp", "đồng"]
    WORD_RE = re.compile(r"\w+", re.UNICODE)
     # corpus là nơi lưu toàn bộ DocumentChunk trong bộ nhớ
    def __init__(self, corpus: InMemoryCorpus) -> None:
        self.corpus = corpus

    def retrieve(self, question: str, query_type: QueryType, top_k: int = 5) -> List[RetrievalHit]:
        hits: List[RetrievalHit] = []
        q_tokens = self._tokenize(question)
        for chunk in self.corpus.all_chunks():
            c_tokens = self._tokenize(chunk.text)
            bm25 = self._keyword_overlap_score(q_tokens, c_tokens)
            dense = self._jaccard_score(q_tokens, c_tokens)
            rerank = self._structure_bonus(query_type, chunk)
            if bm25 + dense + rerank <= 0:
                continue
            hits.append(RetrievalHit(chunk=chunk, bm25_score=bm25, dense_score=dense, rerank_score=rerank))
        hits.sort(key=lambda x: x.hybrid_score, reverse=True)
        return hits[:top_k]

    def _tokenize(self, text: str) -> List[str]:
        return [t.lower() for t in self.WORD_RE.findall(text)]

    def _keyword_overlap_score(self, q_tokens: List[str], c_tokens: List[str]) -> float:
        if not q_tokens or not c_tokens:
            return 0.0
        c_set = set(c_tokens)
        overlap = sum(1 for t in q_tokens if t in c_set)
        return overlap / math.sqrt(len(q_tokens) + 1)

    def _jaccard_score(self, q_tokens: List[str], c_tokens: List[str]) -> float:
        q_set, c_set = set(q_tokens), set(c_tokens)
        if not q_set or not c_set:
            return 0.0
        return len(q_set & c_set) / len(q_set | c_set)

    def _structure_bonus(self, query_type: QueryType, chunk: DocumentChunk) -> float:
        bonus = 0.0
        if query_type == QueryType.DEFINITION and chunk.block_type in {"paragraph", "heading"}:
            bonus += 0.20
        if query_type == QueryType.POLICY and (chunk.section or "").lower().find("quy") >= 0:
            bonus += 0.30
        if query_type == QueryType.COMPARISON and chunk.block_type in {"table", "list"}:
            bonus += 0.25
        if query_type == QueryType.PROCEDURAL and chunk.block_type in {"list", "paragraph"}:
            bonus += 0.20
        return bonus


# Evidence-aware routing
class EvidenceChecker:
    def assess(self, question: str, hits: List[RetrievalHit]) -> EvidenceReport:
        if not hits:
            return EvidenceReport(
                relevance=0.0, #mức độ liên quan tổng thể của các đoạn truy hồi so với câu hỏi
                coverage=0.0, #mức độ bao phủ: các đoạn truy hồi đã bao phủ đủ khía cạnh cần thiết của câu hỏi chưa
                consistency=0.0,#độ nhất quán giữa các đoạn truy hồi, có mâu thuẫn hay thông tin trái ngược không
                citation_support=0.0, #mức độ hỗ trợ trích dẫn: các đoạn truy hồi có cung cấp thông tin để trích dẫn không
                sufficiency=0.0,#mức độ đủ bằng chứng để trả lời câu hỏi dựa trên các đoạn truy hồi hiện có
                decision=RouteAction.SWITCH_STRATEGY,
                reason="Không có đoạn nào đủ liên quan. Cần đổi chiến lược retrieval hoặc mở rộng chỉ mục.",
            )

        top_scores = [h.hybrid_score for h in hits[:3]]
        relevance = min(1.0, sum(top_scores) / max(1.0, len(top_scores)) / 1.5)

        q_terms = {t.lower() for t in re.findall(r"\w+", question)}
        covered_terms = set()
        for h in hits[:3]:
            covered_terms |= q_terms & {t.lower() for t in re.findall(r"\w+", h.chunk.text)}
        coverage = len(covered_terms) / max(1, len(q_terms))

        sections = {(h.chunk.section or "") for h in hits[:3]}
        consistency = 1.0 if len(sections) <= 2 else 0.7

        citation_support = min(1.0, len(hits[:2]) / 2)
        sufficiency = 0.35 * relevance + 0.30 * coverage + 0.20 * consistency + 0.15 * citation_support

        if sufficiency >= 0.72:
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
    def generate(
        self,
        question: str,
        query_type: QueryType,
        hits: List[RetrievalHit],
        evidence: EvidenceReport,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        citations = [
            {
                "chunk_id": h.chunk.chunk_id,
                "source_name": h.chunk.source_name,
                "page": h.chunk.page,
                "section": h.chunk.section,
            }
            for h in hits[:2]
        ]

        if evidence.decision == RouteAction.ABSTAIN:
            return (
                "Mình chưa thể trả lời chắc chắn vì bằng chứng hiện tại chưa đủ. "
                "Bạn nên tải thêm tài liệu liên quan hoặc mở rộng truy hồi.",
                citations,
            )

        if evidence.decision == RouteAction.EXPAND_RETRIEVAL:
            return (
                "Mình đã tìm thấy một phần thông tin liên quan nhưng chưa đủ để kết luận đầy đủ. "
                "Ở bước tiếp theo, hệ thống nên tăng top-k, mở rộng section, hoặc đổi filter metadata.",
                citations,
            )

        if evidence.decision == RouteAction.SWITCH_STRATEGY:
            return (
                "Truy hồi hiện tại chưa đúng trọng tâm câu hỏi. "
                "Nên chuyển chiến lược, ví dụ ưu tiên section quy định, bảng, hoặc truy hồi dense mạnh hơn.",
                citations,
            )

        snippets = [f"- {h.chunk.text.strip()}" for h in hits[:2]]
        answer = (
            f"Loại câu hỏi được nhận diện là '{query_type.value}'. "
            f"Dựa trên các đoạn truy hồi tốt nhất, câu trả lời tóm tắt là:\n" + "\n".join(snippets)
        )
        return answer, citations


# =========================
# Orchestrator
# =========================

class RoutedRAGService:
    def __init__(self) -> None:
        self.corpus = InMemoryCorpus()
        self.router = QueryRouter()
        self.retriever = HybridRetriever(self.corpus)
        self.evidence_checker = EvidenceChecker()
        self.answer_generator = AnswerGenerator()
        self._load_demo_data()

    def ask(self, question: str) -> AskResponse:
        query_type = self.router.route(question)

        # Example route-specific retrieval depth
        top_k_map = {
            QueryType.FACTOID: 3,
            QueryType.DEFINITION: 4,
            QueryType.POLICY: 6,
            QueryType.COMPARISON: 6,
            QueryType.PROCEDURAL: 5,
            QueryType.MULTI_HOP: 7,
            QueryType.AMBIGUOUS: 4,
        }
        hits = self.retriever.retrieve(question, query_type, top_k=top_k_map[query_type])
        evidence = self.evidence_checker.assess(question, hits)
        answer, citations = self.answer_generator.generate(question, query_type, hits, evidence)

        return AskResponse(
            question=question,
            query_type=query_type,
            route_action=evidence.decision,
            answer=answer,
            evidence_report={
                "relevance": evidence.relevance,
                "coverage": evidence.coverage,
                "consistency": evidence.consistency,
                "citation_support": evidence.citation_support,
                "sufficiency": evidence.sufficiency,
                "reason": evidence.reason,
            },
            citations=citations,
            retrieved_chunks=[
                {
                    "chunk_id": h.chunk.chunk_id,
                    "source_name": h.chunk.source_name,
                    "page": h.chunk.page,
                    "section": h.chunk.section,
                    "block_type": h.chunk.block_type,
                    "hybrid_score": round(h.hybrid_score, 3),
                    "text": h.chunk.text,
                }
                for h in hits
            ],
        )

    def _load_demo_data(self) -> None:
        demo_chunks = [
            DocumentChunk(
                chunk_id="c1",
                source_name="Student Handbook",
                page=10,
                section="Điều kiện tốt nghiệp",
                block_type="paragraph",
                text="Sinh viên được xét tốt nghiệp khi tích lũy đủ số tín chỉ theo chương trình đào tạo và đáp ứng chuẩn đầu ra ngoại ngữ.",
            ),
            DocumentChunk(
                chunk_id="c2",
                source_name="Academic Regulations",
                page=14,
                section="Quy chế học vụ",
                block_type="paragraph",
                text="Sinh viên bị cảnh báo học vụ nếu điểm trung bình tích lũy dưới ngưỡng quy định trong hai học kỳ liên tiếp.",
            ),
            DocumentChunk(
                chunk_id="c3",
                source_name="Academic Regulations",
                page=18,
                section="Quy trình xin hoãn thi",
                block_type="list",
                text="Thủ tục xin hoãn thi gồm các bước: nộp đơn, đính kèm minh chứng, chờ khoa xác nhận và nhận phản hồi từ phòng đào tạo.",
            ),
            DocumentChunk(
                chunk_id="c4",
                source_name="Course Catalog",
                page=7,
                section="Môn tiên quyết",
                block_type="table",
                text="Môn Cơ sở dữ liệu yêu cầu hoàn thành môn Nhập môn lập trình trước khi đăng ký.",
            ),
            DocumentChunk(
                chunk_id="c5",
                source_name="Student FAQ",
                page=2,
                section="Định nghĩa tín chỉ",
                block_type="paragraph",
                text="Tín chỉ là đơn vị dùng để đo khối lượng học tập của sinh viên, bao gồm thời lượng lên lớp, tự học và đánh giá.",
            ),
        ]
        self.corpus.add_chunks(demo_chunks)


# =========================
# FastAPI app
# =========================

app = FastAPI(title="Routed RAG Starter", version="0.1.0")
service = RoutedRAGService()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    return service.ask(req.question)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("routed_rag_starter:app", host="0.0.0.0", port=8000, reload=True)
