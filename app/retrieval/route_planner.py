from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from app.retrieval.schemas import RetrievalConfig


@dataclass(slots=True)
class QueryRetrievalPlan:
    """Retrieval strategy selected by the query router."""

    strategy: str
    config: RetrievalConfig
    reason: str


class QueryAwareRetrievalPlanner:
    """Map routed query types to retrieval strategies and runtime configs."""

    def plan(self, query_type: str, question: str) -> QueryRetrievalPlan:
        normalized_type = query_type.strip().lower()
        question_lower = question.lower()

        if normalized_type == "factoid":
            if _is_table_lookup_query(question):
                return QueryRetrievalPlan(
                    strategy="hybrid",
                    config=RetrievalConfig(
                        top_k=8,
                        candidate_k=80,
                        bm25_weight=0.85,
                        dense_weight=0.15,
                        combination="weighted_sum",
                        context_window=2,
                    ),
                    reason="table lookup queries need wider lexical recall plus adjacent rows/columns",
                )
            bm25_weight, dense_weight = self._weights(question, bm25_default=0.45, dense_default=0.55)
            scientific_like = self._is_scientific_like(question)
            return QueryRetrievalPlan(
                strategy="hybrid_rerank" if scientific_like else "hybrid",
                config=RetrievalConfig(
                    top_k=5 if scientific_like else 3,
                    candidate_k=50 if scientific_like else 30,
                    bm25_weight=bm25_weight,
                    dense_weight=dense_weight,
                    rerank_top_n=20 if scientific_like else 0,
                    combination="weighted_sum",
                    use_rerank=scientific_like,
                    context_window=2 if scientific_like else 1,
                ),
                reason="factoid queries need compact evidence with lexical anchoring and dense recall",
            )

        if normalized_type == "definition":
            bm25_weight, dense_weight = self._weights(question, bm25_default=0.50, dense_default=0.50)
            scientific_like = self._is_scientific_like(question)
            return QueryRetrievalPlan(
                strategy="hybrid_rerank" if scientific_like else "hybrid",
                config=RetrievalConfig(
                    top_k=5 if scientific_like else 4,
                    candidate_k=70 if scientific_like else 50,
                    bm25_weight=bm25_weight,
                    dense_weight=dense_weight,
                    rerank_top_n=25 if scientific_like else 0,
                    combination="weighted_sum",
                    use_rerank=scientific_like,
                    context_window=2 if scientific_like else 1,
                ),
                reason="definition queries benefit from balanced sparse and dense evidence",
            )

        if normalized_type == "policy":
            bm25_weight, dense_weight = self._weights(question, bm25_default=0.60, dense_default=0.40)
            return QueryRetrievalPlan(
                strategy="hybrid_rerank",
                config=RetrievalConfig(
                    top_k=6,
                    candidate_k=70,
                    bm25_weight=bm25_weight,
                    dense_weight=dense_weight,
                    rerank_top_n=20,
                    combination="weighted_sum",
                    use_rerank=True,
                    context_window=1,
                ),
                reason="policy queries favor exact terms, section headings, and reranking",
            )

        if normalized_type == "procedural":
            bm25_weight, dense_weight = self._weights(question, bm25_default=0.45, dense_default=0.55)
            block_filter = []
            if any(term in question_lower for term in ("step", "steps", "buoc", "bước", "procedure", "process")):
                block_filter = ["list_item", "list", "paragraph"]
            return QueryRetrievalPlan(
                strategy="hybrid_rerank",
                config=RetrievalConfig(
                    top_k=5,
                    candidate_k=70,
                    bm25_weight=bm25_weight,
                    dense_weight=dense_weight,
                    rerank_top_n=20,
                    combination="weighted_sum",
                    use_rerank=True,
                    block_type_filter=block_filter,
                    context_window=1,
                ),
                reason="procedural queries need ordered/list-like evidence with adjacent context",
            )

        if normalized_type == "comparison":
            bm25_weight, dense_weight = self._weights(question, bm25_default=0.45, dense_default=0.55)
            return QueryRetrievalPlan(
                strategy="hybrid_rerank",
                config=RetrievalConfig(
                    top_k=6,
                    candidate_k=80,
                    bm25_weight=bm25_weight,
                    dense_weight=dense_weight,
                    rerank_top_n=25,
                    combination="weighted_sum" if self._is_vietnamese(question) else "rrf",
                    use_rerank=True,
                    context_window=1,
                ),
                reason="comparison queries need broader recall and route-aware fusion across signals",
            )

        if normalized_type == "multi_hop":
            bm25_weight, dense_weight = self._weights(question, bm25_default=0.40, dense_default=0.60)
            return QueryRetrievalPlan(
                strategy="hybrid",
                config=RetrievalConfig(
                    top_k=8,
                    candidate_k=100,
                    bm25_weight=bm25_weight,
                    dense_weight=dense_weight,
                    combination="rrf",
                    context_window=1,
                ),
                reason="multi-hop queries need broad retrieval before evidence synthesis",
            )

        bm25_weight, dense_weight = self._weights(question, bm25_default=0.50, dense_default=0.50)
        return QueryRetrievalPlan(
            strategy="hybrid",
            config=RetrievalConfig(
                top_k=4,
                candidate_k=60,
                bm25_weight=bm25_weight,
                dense_weight=dense_weight,
                combination="rrf",
                context_window=1,
            ),
            reason="ambiguous queries use balanced retrieval with adjacent context",
        )

    def _weights(self, question: str, *, bm25_default: float, dense_default: float) -> tuple[float, float]:
        """Use stronger lexical anchoring for Vietnamese PDFs unless overridden later by router policy."""

        if self._is_vietnamese(question):
            return max(bm25_default, 0.72), min(dense_default, 0.28)
        return bm25_default, dense_default

    def _is_vietnamese(self, question: str) -> bool:
        return any("à" <= char.lower() <= "ỹ" or char.lower() == "đ" for char in question)

    def _is_scientific_like(self, question: str) -> bool:
        question_lower = question.lower()
        return any(
            term in question_lower
            for term in (
                "paper",
                "model",
                "architecture",
                "attention",
                "transformer",
                "encoder",
                "decoder",
                "bleu",
                "f1",
                "wmt",
                "feed-forward",
                "positional encoding",
                "dot-product",
                "dot products",
                "square root",
                "beam size",
                "label smoothing",
            )
        )


def _is_table_lookup_query(question: str) -> bool:
    folded = _fold(question)
    if not folded:
        return False
    lookup_cues = (
        "correspond",
        "mapped",
        "belongs to",
        "which range",
        "what range",
        "which level",
        "what level",
        "what score",
        "what value",
        "ung voi",
        "tuong ung",
        "quy doi",
        "thuoc khoang",
        "khoang nao",
        "muc nao",
        "bao nhieu diem",
        "diem chu",
        "diem so",
        "thang diem",
        "thang 4",
    )
    if not any(cue in folded for cue in lookup_cues):
        return False
    return bool(re.search(r"(?<![\w.])[-+]?\d+(?:[,.]\d+)?%?(?![\w.])", question) or re.search(r"(?<!\w)[A-Za-z][A-Za-z0-9]{0,5}[+-]?(?!\w)", question))


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_text).strip().lower()
