from __future__ import annotations

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
            bm25_weight, dense_weight = self._weights(question, bm25_default=0.45, dense_default=0.55)
            return QueryRetrievalPlan(
                strategy="hybrid",
                config=RetrievalConfig(
                    top_k=3,
                    candidate_k=30,
                    bm25_weight=bm25_weight,
                    dense_weight=dense_weight,
                    combination="weighted_sum",
                    context_window=1,
                ),
                reason="factoid queries need compact evidence with lexical anchoring and dense recall",
            )

        if normalized_type == "definition":
            bm25_weight, dense_weight = self._weights(question, bm25_default=0.50, dense_default=0.50)
            return QueryRetrievalPlan(
                strategy="hybrid",
                config=RetrievalConfig(
                    top_k=4,
                    candidate_k=50,
                    bm25_weight=bm25_weight,
                    dense_weight=dense_weight,
                    combination="weighted_sum",
                    context_window=1,
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
