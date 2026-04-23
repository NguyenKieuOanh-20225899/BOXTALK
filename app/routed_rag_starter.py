from __future__ import annotations

import os
import re
import json
import shutil
import time
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.ingest.pipeline import ingest_pdf
from app.loaders.pdf_loader import PDFLoader
from app.qa.pipeline import GroundedQAPipeline
from app.retrieval.reranker import make_reranker
from app.retrieval.route_planner import QueryAwareRetrievalPlanner
from app.retrieval.schemas import DocumentChunkRef, RetrievedHit, coerce_chunk_ref
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


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    status: str
    document_type: str = "unknown"
    chunk_count: int = 0
    page_count: int = 0
    block_count: int = 0
    used_backend: Optional[str] = None
    probe_mode: Optional[str] = None
    index_mode: str = "bm25"
    build_ms: Optional[float] = None
    created_at: str
    indexed_at: Optional[str] = None
    last_error: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    pdf_url: Optional[str] = None


class DocumentAskRequest(BaseModel):
    question: str = Field(..., min_length=3)


class DocumentAskResponse(BaseModel):
    document: DocumentInfo
    result: AskResponse


class ReindexRequest(BaseModel):
    build_dense: Optional[bool] = None


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


# Document registry and UI helpers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UI_DIR = PROJECT_ROOT / "app" / "ui"
DOCUMENT_STORE_ROOT = Path(os.getenv("BOXTALK_DOCUMENT_STORE_DIR", "data/ui_documents"))
DOCUMENT_INDEX_ROOT = Path(os.getenv("BOXTALK_DOCUMENT_INDEX_DIR", "results/ui_document_indexes"))
DOCUMENTS_METADATA_PATH = DOCUMENT_STORE_ROOT / "documents.json"


class DocumentRegistry:
    def __init__(self, metadata_path: Path) -> None:
        self.metadata_path = metadata_path
        self._documents: Dict[str, Dict[str, Any]] = {}
        self._load()

    def list(self) -> List[Dict[str, Any]]:
        return sorted(
            self._documents.values(),
            key=lambda item: item.get("created_at", ""),
            reverse=True,
        )

    def get(self, doc_id: str) -> Optional[Dict[str, Any]]:
        document = self._documents.get(doc_id)
        return dict(document) if document else None

    def upsert(self, document: Dict[str, Any]) -> Dict[str, Any]:
        self._documents[document["doc_id"]] = dict(document)
        self._save()
        return dict(document)

    def delete(self, doc_id: str) -> Optional[Dict[str, Any]]:
        document = self._documents.pop(doc_id, None)
        self._save()
        return document

    def _load(self) -> None:
        if not self.metadata_path.exists():
            self._documents = {}
            return
        payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        rows = payload.get("documents", [])
        self._documents = {
            str(item["doc_id"]): dict(item)
            for item in rows
            if isinstance(item, dict) and item.get("doc_id")
        }

    def _save(self) -> None:
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"documents": self.list()}
        temp_path = self.metadata_path.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(self.metadata_path)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _new_doc_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    return f"{timestamp}_{uuid.uuid4().hex[:8]}"


def _safe_filename(filename: str | None) -> str:
    raw = Path(filename or "document.pdf").name.strip()
    if not raw.lower().endswith(".pdf"):
        raw = f"{raw}.pdf"
    return re.sub(r"[^A-Za-z0-9._ -]+", "_", raw).strip(" .") or "document.pdf"


def _public_document_info(document: Dict[str, Any]) -> DocumentInfo:
    public = {
        key: value
        for key, value in document.items()
        if key not in {"pdf_path", "index_dir"}
    }
    public["pdf_url"] = f"/documents/{document['doc_id']}/file"
    return DocumentInfo(**public)


def _adapt_ingest_chunk(chunk: Any, *, source_name: str, doc_id: str) -> DocumentChunkRef:
    ref = coerce_chunk_ref(chunk, source_name=source_name, doc_id=doc_id)
    chunk_id = ref.chunk_id
    if not chunk_id.startswith(f"{doc_id}:"):
        chunk_id = f"{doc_id}:{chunk_id}"
    metadata = {
        **ref.metadata,
        "doc_id": doc_id,
        "source_name": source_name,
        "filename": source_name,
    }
    return replace(
        ref,
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_name=source_name,
        metadata=metadata,
    )


def _infer_document_type(filename: str, chunks: List[DocumentChunkRef], report: Dict[str, Any]) -> str:
    sample = " ".join(chunk.text for chunk in chunks[:30]).lower()
    name = filename.lower()
    scientific_terms = {
        "abstract",
        "introduction",
        "method",
        "experiment",
        "results",
        "references",
        "transformer",
        "attention",
        "dataset",
        "model",
    }
    policy_terms = {
        "quy định",
        "điều",
        "nghị định",
        "thông tư",
        "chính sách",
        "regulation",
        "policy",
    }
    handbook_terms = {"handbook", "manual", "guide", "procedure", "hướng dẫn", "quy trình"}

    if any(term in sample or term in name for term in scientific_terms):
        return "scientific_paper"
    if any(term in sample or term in name for term in policy_terms):
        return "policy_regulation"
    if any(term in sample or term in name for term in handbook_terms):
        return "handbook_manual"
    return report.get("probe", {}).get("probe_detected_mode") or "general_pdf"


def _build_document_index(document: Dict[str, Any], *, build_dense: bool) -> Dict[str, Any]:
    pdf_path = Path(document["pdf_path"])
    index_dir = Path(document["index_dir"])
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if index_dir.exists():
        _remove_tree_inside(index_dir, DOCUMENT_INDEX_ROOT)
    index_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    report = ingest_pdf(pdf_path)
    chunks = [
        _adapt_ingest_chunk(chunk, source_name=document["filename"], doc_id=document["doc_id"])
        for chunk in report.get("chunks", [])
    ]
    if not chunks:
        raise RuntimeError("PDF ingest produced no chunks")

    warnings: List[str] = []
    try:
        retrieval_service = RetrievalService.from_chunks(
            chunks,
            model_name=os.getenv("BOXTALK_EMBED_MODEL_NAME"),
            dense_backend=os.getenv("BOXTALK_DENSE_BACKEND"),
            dense_preset=os.getenv("BOXTALK_DENSE_PRESET"),
            dense_query_model_name=os.getenv("BOXTALK_DENSE_QUERY_MODEL_NAME"),
            dense_passage_model_name=os.getenv("BOXTALK_DENSE_PASSAGE_MODEL_NAME"),
            build_dense=build_dense,
            reranker=make_reranker(os.getenv("BOXTALK_ROUTED_RAG_RERANKER", "heuristic")),
        )
    except Exception as exc:
        if not build_dense:
            raise
        warnings.append(f"Dense index failed, fell back to BM25: {exc}")
        retrieval_service = RetrievalService.from_chunks(
            chunks,
            build_dense=False,
            reranker=make_reranker(os.getenv("BOXTALK_ROUTED_RAG_RERANKER", "heuristic")),
        )
        build_dense = False

    retrieval_service.retriever.save(index_dir)
    build_ms = round((time.perf_counter() - started) * 1000.0, 3)
    return {
        **document,
        "status": "ready",
        "document_type": _infer_document_type(document["filename"], chunks, report),
        "chunk_count": len(chunks),
        "page_count": len(report.get("pages", [])),
        "block_count": len(report.get("blocks", [])),
        "used_backend": report.get("used_backend"),
        "probe_mode": report.get("probe", {}).get("probe_detected_mode"),
        "index_mode": "hybrid" if build_dense else "bm25",
        "build_ms": build_ms,
        "indexed_at": _utc_now(),
        "last_error": None,
        "warnings": warnings,
    }


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _remove_tree_inside(path: Path, root: Path) -> None:
    if not _is_inside(path, root):
        raise RuntimeError(f"Refusing to remove path outside {root}: {path}")
    shutil.rmtree(path, ignore_errors=True)


# FastAPI app

app = FastAPI(title="BOXTALK Document QA", version="0.3.0")
if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

PDF_PATH = os.getenv("PDF_PATH", "data/sample.pdf")
INDEX_DIR = os.getenv("BOXTALK_RETRIEVAL_INDEX_DIR")
registry = DocumentRegistry(DOCUMENTS_METADATA_PATH)
_default_service: Optional[RoutedRAGService] = None
_document_services: Dict[str, Tuple[Optional[str], RoutedRAGService]] = {}


def _get_default_service() -> RoutedRAGService:
    global _default_service
    if _default_service is not None:
        return _default_service

    if INDEX_DIR:
        if not Path(INDEX_DIR).exists():
            raise HTTPException(status_code=503, detail=f"Index not found: {INDEX_DIR}")
        _default_service = RoutedRAGService(index_dir=INDEX_DIR)
        return _default_service

    if PDF_PATH and Path(PDF_PATH).exists():
        _default_service = RoutedRAGService(pdf_path=PDF_PATH)
        return _default_service

    raise HTTPException(
        status_code=503,
        detail="No default PDF/index is configured. Upload a document or set PDF_PATH/BOXTALK_RETRIEVAL_INDEX_DIR.",
    )


def _get_document_or_404(doc_id: str) -> Dict[str, Any]:
    document = registry.get(doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


def _get_document_service(document: Dict[str, Any]) -> RoutedRAGService:
    if document.get("status") != "ready":
        raise HTTPException(status_code=409, detail="Document is not ready")
    index_dir = Path(document["index_dir"])
    if not index_dir.exists():
        raise HTTPException(status_code=503, detail="Document index is missing")

    doc_id = document["doc_id"]
    indexed_at = document.get("indexed_at")
    cached = _document_services.get(doc_id)
    if cached and cached[0] == indexed_at:
        return cached[1]

    service = RoutedRAGService(index_dir=str(index_dir))
    _document_services[doc_id] = (indexed_at, service)
    return service


@app.get("/", response_class=HTMLResponse)
def ui() -> HTMLResponse:
    index_path = UI_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI assets are not available")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/documents", response_model=List[DocumentInfo])
def list_documents() -> List[DocumentInfo]:
    return [_public_document_info(document) for document in registry.list()]


@app.post("/documents", response_model=DocumentInfo)
async def upload_document(
    file: UploadFile = File(...),
    build_dense: bool = Form(False),
) -> DocumentInfo:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported")

    doc_id = _new_doc_id()
    filename = _safe_filename(file.filename)
    doc_dir = DOCUMENT_STORE_ROOT / doc_id
    pdf_path = doc_dir / filename
    index_dir = DOCUMENT_INDEX_ROOT / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    with pdf_path.open("wb") as handle:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)

    document = {
        "doc_id": doc_id,
        "filename": filename,
        "status": "processing",
        "document_type": "unknown",
        "chunk_count": 0,
        "page_count": 0,
        "block_count": 0,
        "used_backend": None,
        "probe_mode": None,
        "index_mode": "hybrid" if build_dense else "bm25",
        "build_ms": None,
        "created_at": _utc_now(),
        "indexed_at": None,
        "last_error": None,
        "warnings": [],
        "pdf_path": str(pdf_path),
        "index_dir": str(index_dir),
    }
    registry.upsert(document)

    try:
        document = _build_document_index(document, build_dense=build_dense)
    except Exception as exc:
        document = {
            **document,
            "status": "error",
            "last_error": str(exc),
            "indexed_at": None,
        }
    registry.upsert(document)
    return _public_document_info(document)


@app.get("/documents/{doc_id}", response_model=DocumentInfo)
def get_document(doc_id: str) -> DocumentInfo:
    return _public_document_info(_get_document_or_404(doc_id))


@app.get("/documents/{doc_id}/file")
def get_document_file(doc_id: str) -> FileResponse:
    document = _get_document_or_404(doc_id)
    pdf_path = Path(document["pdf_path"])
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found")
    return FileResponse(pdf_path, media_type="application/pdf", filename=document["filename"])


@app.post("/documents/{doc_id}/ask", response_model=DocumentAskResponse)
def ask_document(doc_id: str, req: DocumentAskRequest) -> DocumentAskResponse:
    document = _get_document_or_404(doc_id)
    service = _get_document_service(document)
    return DocumentAskResponse(
        document=_public_document_info(document),
        result=service.ask(req.question),
    )


@app.post("/documents/{doc_id}/reindex", response_model=DocumentInfo)
def reindex_document(doc_id: str, req: ReindexRequest) -> DocumentInfo:
    document = _get_document_or_404(doc_id)
    build_dense = req.build_dense
    if build_dense is None:
        build_dense = document.get("index_mode") == "hybrid"
    document = {
        **document,
        "status": "processing",
        "last_error": None,
        "warnings": [],
    }
    registry.upsert(document)

    try:
        document = _build_document_index(document, build_dense=build_dense)
    except Exception as exc:
        document = {
            **document,
            "status": "error",
            "last_error": str(exc),
            "indexed_at": None,
        }
    registry.upsert(document)
    _document_services.pop(doc_id, None)
    return _public_document_info(document)


@app.delete("/documents/{doc_id}", response_model=Dict[str, str])
def delete_document(doc_id: str) -> Dict[str, str]:
    document = registry.delete(doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")
    _document_services.pop(doc_id, None)

    pdf_path = Path(document["pdf_path"])
    doc_dir = pdf_path.parent
    index_dir = Path(document["index_dir"])
    if doc_dir.exists():
        _remove_tree_inside(doc_dir, DOCUMENT_STORE_ROOT)
    if index_dir.exists():
        _remove_tree_inside(index_dir, DOCUMENT_INDEX_ROOT)
    return {"status": "deleted", "doc_id": doc_id}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    return _get_default_service().ask(req.question)


@app.get("/debug/chunks")
def debug_chunks(limit: int = 20):
    service = _get_default_service()
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
    service = _get_default_service()
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
