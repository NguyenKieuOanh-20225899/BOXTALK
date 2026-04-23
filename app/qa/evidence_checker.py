from __future__ import annotations

from app.qa.schemas import EvidenceAssessment
from app.qa.text_utils import split_sentences, token_set
from app.retrieval.schemas import RetrievedHit


class EvidenceChecker:
    """Assess whether retrieved chunks are sufficient for grounded answering."""

    def __init__(
        self,
        *,
        answer_threshold: float = 0.52,
        weak_threshold: float = 0.35,
        max_support_hits: int = 3,
    ) -> None:
        self.answer_threshold = answer_threshold
        self.weak_threshold = weak_threshold
        self.max_support_hits = max_support_hits

    def assess(self, question: str, query_type: str, hits: list[RetrievedHit]) -> EvidenceAssessment:
        if not hits:
            return EvidenceAssessment(
                relevance=0.0,
                coverage=0.0,
                consistency=0.0,
                citation_support=0.0,
                grounding=0.0,
                sufficiency=0.0,
                decision="switch_strategy",
                reason="No retrieved evidence was returned.",
            )

        selected = self._select_support_hits(query_type, hits)
        question_terms = token_set(question)
        top_scores = [float(hit.final_score or hit.score) for hit in selected]
        top1 = float(hits[0].final_score or hits[0].score)
        top2 = float(hits[1].final_score or hits[1].score) if len(hits) > 1 else 0.0
        top_gap = max(0.0, top1 - top2)

        relevance = min(1.0, 0.65 * top1 + 0.25 * _mean(top_scores) + 0.10 * min(1.0, top_gap))
        coverage = self._coverage(question_terms, selected)
        consistency = self._consistency(selected)
        citation_support = self._citation_support(selected)
        grounding = self._grounding(question_terms, selected)
        sufficiency = self._sufficiency(
            query_type=query_type,
            relevance=relevance,
            coverage=coverage,
            consistency=consistency,
            citation_support=citation_support,
            grounding=grounding,
            top_gap=top_gap,
        )
        decision, reason = self._decision(
            sufficiency=sufficiency,
            relevance=relevance,
            coverage=coverage,
            grounding=grounding,
        )

        return EvidenceAssessment(
            relevance=round(relevance, 3),
            coverage=round(coverage, 3),
            consistency=round(consistency, 3),
            citation_support=round(citation_support, 3),
            grounding=round(grounding, 3),
            sufficiency=round(sufficiency, 3),
            decision=decision,
            reason=reason,
            selected_hit_ids=[hit.chunk_id for hit in selected],
            support_sentences=self._support_sentences(question_terms, selected),
            diagnostics={
                "top_score": round(top1, 4),
                "top_gap": round(top_gap, 4),
                "support_hit_count": len(selected),
            },
        )

    def _select_support_hits(self, query_type: str, hits: list[RetrievedHit]) -> list[RetrievedHit]:
        if query_type in {"comparison", "multi_hop", "procedural"}:
            return hits[: min(self.max_support_hits, len(hits))]
        return hits[: min(2, len(hits))]

    def _coverage(self, question_terms: set[str], hits: list[RetrievedHit]) -> float:
        if not question_terms:
            return 0.0
        covered: set[str] = set()
        for hit in hits:
            covered |= question_terms & token_set(" ".join([hit.chunk.section or "", hit.chunk.text or ""]))
        return len(covered) / len(question_terms)

    def _consistency(self, hits: list[RetrievedHit]) -> float:
        if len(hits) <= 1:
            return 1.0
        doc_ids = {hit.chunk.doc_id or hit.chunk.source_name or "" for hit in hits}
        sections = {hit.chunk.section or "" for hit in hits if hit.chunk.section}
        if len(doc_ids) <= 1 and len(sections) <= max(2, len(hits) - 1):
            return 1.0
        if len(doc_ids) <= 2:
            return 0.75
        return 0.55

    def _citation_support(self, hits: list[RetrievedHit]) -> float:
        if not hits:
            return 0.0
        supported = 0
        for hit in hits:
            if hit.chunk_id and (hit.chunk.source_name or hit.chunk.doc_id):
                supported += 1
        return supported / len(hits)

    def _grounding(self, question_terms: set[str], hits: list[RetrievedHit]) -> float:
        sentences = self._support_sentences(question_terms, hits)
        if not sentences:
            return 0.0
        if not question_terms:
            return 0.5
        sentence_terms = token_set(" ".join(sentences))
        return len(question_terms & sentence_terms) / len(question_terms)

    def _support_sentences(self, question_terms: set[str], hits: list[RetrievedHit]) -> list[str]:
        candidates: list[tuple[float, str]] = []
        for hit in hits:
            for sentence in split_sentences(hit.chunk.text):
                sentence_terms = token_set(sentence)
                overlap = len(question_terms & sentence_terms)
                if overlap == 0 and question_terms:
                    continue
                score = overlap + 0.01 * len(sentence_terms)
                candidates.append((score, sentence))
        candidates.sort(key=lambda item: item[0], reverse=True)
        seen: set[str] = set()
        selected: list[str] = []
        for _, sentence in candidates:
            normalized = sentence.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            selected.append(sentence)
            if len(selected) >= 3:
                break
        return selected

    def _sufficiency(
        self,
        *,
        query_type: str,
        relevance: float,
        coverage: float,
        consistency: float,
        citation_support: float,
        grounding: float,
        top_gap: float,
    ) -> float:
        if query_type == "policy" and top_gap >= 0.30:
            return 0.45 * relevance + 0.15 * coverage + 0.10 * consistency + 0.10 * citation_support + 0.20 * grounding
        if query_type in {"comparison", "multi_hop"}:
            return 0.30 * relevance + 0.25 * coverage + 0.20 * consistency + 0.10 * citation_support + 0.15 * grounding
        return 0.35 * relevance + 0.25 * coverage + 0.15 * consistency + 0.10 * citation_support + 0.15 * grounding

    def _decision(
        self,
        *,
        sufficiency: float,
        relevance: float,
        coverage: float,
        grounding: float,
    ) -> tuple[str, str]:
        if sufficiency >= self.answer_threshold and grounding >= 0.25:
            return "answer", "Retrieved evidence is sufficient and grounded enough to answer."
        if relevance >= self.weak_threshold and coverage < 0.45:
            return "expand_retrieval", "Evidence is on-topic but does not cover enough of the question."
        if relevance < self.weak_threshold:
            return "switch_strategy", "Retrieved evidence is weak for this question."
        return "abstain", "Evidence is relevant but not sufficient for a grounded answer."


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
