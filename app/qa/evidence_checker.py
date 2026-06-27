from __future__ import annotations

import re
import unicodedata

from app.qa.schemas import EvidenceAssessment
from app.qa.text_utils import split_sentences, token_set
from app.retrieval.schemas import RetrievedHit


YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
MONEY_RE = re.compile(
    r"(?:\b\d+(?:[,.]\d+)?\s*(?:dong|vnd|usd|trieu|nghin|ngan|million|k|m)\b|[$])",
    re.I,
)
NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:[,.]\d+)?")


class EvidenceChecker:
    """Assess whether retrieved chunks are sufficient for grounded answering."""

    def __init__(
        self,
        *,
        answer_threshold: float = 0.52,
        weak_threshold: float = 0.35,
        max_support_hits: int = 5,
        max_table_support_hits: int = 5,
    ) -> None:
        self.answer_threshold = answer_threshold
        self.weak_threshold = weak_threshold
        self.max_support_hits = max_support_hits
        self.max_table_support_hits = max_table_support_hits

    def check(self, *, question: str, hits: list[RetrievedHit], query_type: str = "factoid") -> EvidenceAssessment:
        return self.assess(question, query_type, hits)

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
                sufficient=False,
                missing_constraints=["retrieved_evidence"],
                coverage_details={"retrieved_evidence": False},
            )

        selected = self._select_support_hits(query_type, hits, question=question)
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
        coverage_details, missing_constraints = self._constraint_coverage(question, selected)
        if decision == "answer" and missing_constraints:
            decision = "abstain"
            reason = "Missing required evidence constraints: " + ", ".join(missing_constraints)
            sufficiency = min(sufficiency, self.answer_threshold - 0.01)
        sufficient = decision == "answer" and not missing_constraints

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
                "constraint_coverage": dict(coverage_details),
            },
            sufficient=sufficient,
            missing_constraints=missing_constraints,
            coverage_details=coverage_details,
        )

    def _select_support_hits(self, query_type: str, hits: list[RetrievedHit], *, question: str = "") -> list[RetrievedHit]:
        question_folded = _fold(question)
        if _is_table_constraint_question(question_folded):
            return _select_table_support_hits(
                hits,
                limit=self.max_table_support_hits,
                question_folded=question_folded,
            )
        return hits[: min(self.max_support_hits, len(hits))]

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

    def _constraint_coverage(self, question: str, hits: list[RetrievedHit]) -> tuple[dict[str, bool], list[str]]:
        question_folded = _fold(question)
        evidence_folded = _fold(_evidence_text(hits))
        details: dict[str, bool] = {}

        if _is_money_topic(question_folded):
            details["topic"] = "hoc phi" in evidence_folded or "tuition" in evidence_folded or "fee" in evidence_folded

        programme = _extract_programme(question_folded)
        if programme:
            details["programme"] = _tokens_covered(programme, evidence_folded)

        years = set(YEAR_RE.findall(question_folded))
        if years:
            details["year"] = years <= set(YEAR_RE.findall(evidence_folded))

        if _needs_numeric_value(question_folded):
            details["numeric_value_type"] = bool(re.search(r"\d", evidence_folded))

        if _is_money_question(question_folded):
            details["money_amount"] = bool(MONEY_RE.search(evidence_folded))

        if _is_percentage_question(question_folded):
            details["percentage"] = any(term in evidence_folded for term in ("%", "phan tram", "percent", "percentage"))

        if _is_table_constraint_question(question_folded):
            details["table_row_column"] = _has_table_grounding(hits)

        missing = [name for name, covered in details.items() if not covered]
        return details, missing


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = normalized.replace("đ", "d").replace("Đ", "D")
    folded = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    folded = folded.replace("≥", ">=").replace("≤", "<=").replace("–", "-").replace("−", "-")
    return re.sub(r"\s+", " ", folded).strip().lower()


def _evidence_text(hits: list[RetrievedHit]) -> str:
    parts: list[str] = []
    for hit in hits:
        metadata = {**dict(hit.chunk.metadata or {}), **dict(hit.metadata or {})}
        parts.extend(
            str(value)
            for value in (
                hit.chunk.section,
                hit.chunk.title,
                hit.chunk.text,
                metadata.get("row_header"),
                metadata.get("col_header"),
                metadata.get("cell_text"),
                metadata.get("table_id"),
            )
            if value not in (None, "", [])
        )
    return "\n".join(parts)


def _is_money_topic(question_folded: str) -> bool:
    return any(term in question_folded for term in ("hoc phi", "le phi", "tuition", "fee", "chi phi"))


def _is_money_question(question_folded: str) -> bool:
    return _is_money_topic(question_folded) and any(
        term in question_folded for term in ("bao nhieu", "how much", "muc", "so tien")
    )


def _extract_programme(question_folded: str) -> str | None:
    for marker in ("nganh ", "chuong trinh ", "programme ", "program "):
        if marker not in question_folded:
            continue
        start = question_folded.find(marker) + len(marker)
        tail = question_folded[start:]
        tail = re.split(r"\b(?:nam|la|bao|co|duoc|cua|theo|how|what)\b|[?.,;]", tail, maxsplit=1)[0]
        tokens = [token for token in re.findall(r"[a-z0-9]+", tail) if len(token) > 1]
        if len(tokens) >= 2:
            return " ".join(tokens[:6])
    if "cong nghe thong tin" in question_folded:
        return "cong nghe thong tin"
    return None


def _tokens_covered(phrase: str, evidence_folded: str) -> bool:
    tokens = [token for token in re.findall(r"[a-z0-9]+", phrase) if len(token) > 1]
    if not tokens:
        return True
    evidence_tokens = set(re.findall(r"[a-z0-9]+", evidence_folded))
    required = max(1, int(len(tokens) * 0.75 + 0.999))
    return sum(1 for token in tokens if token in evidence_tokens) >= required


def _needs_numeric_value(question_folded: str) -> bool:
    return any(
        term in question_folded
        for term in (
            "bao nhieu",
            "how many",
            "how much",
            "muc nao",
            "diem",
            "tin chi",
            "thoi gian",
            "ti le",
            "ty le",
            "phan tram",
        )
    )


def _is_percentage_question(question_folded: str) -> bool:
    return any(term in question_folded for term in ("%", "ti le", "ty le", "phan tram", "percent", "percentage"))


def _is_table_constraint_question(question_folded: str) -> bool:
    return any(
        term in question_folded
        for term in (
            "bang",
            "table",
            "tuong ung",
            "ung voi",
            "quy doi",
            "diem chu",
            "thang 4",
            "cot",
            "hang",
        )
    )


def _has_table_grounding(hits: list[RetrievedHit]) -> bool:
    for hit in hits:
        metadata = {**dict(hit.chunk.metadata or {}), **dict(hit.metadata or {})}
        target = str(metadata.get("citation_target") or "").lower()
        block_type = str(hit.chunk.block_type or "").lower()
        if target in {"cell", "row", "table"} or "table" in block_type:
            return True
        if metadata.get("cell_text") not in (None, "") or metadata.get("table_id") not in (None, ""):
            return True
        lines = [line for line in str(hit.chunk.text or "").splitlines() if "|" in line]
        if len(lines) >= 2:
            return True
    return False


def _select_table_support_hits(
    hits: list[RetrievedHit],
    *,
    limit: int,
    question_folded: str = "",
) -> list[RetrievedHit]:
    if limit <= 0:
        return []

    selected: list[RetrievedHit] = []
    selected_ids: set[str] = set()

    markdown_hit = next((hit for hit in hits if _is_markdown_table_hit(hit)), None)
    table_id = ""
    if markdown_hit is not None:
        _append_hit(selected, selected_ids, markdown_hit, limit)
        table_id = _table_id(markdown_hit)

    useful_cells = [hit for hit in hits if _is_useful_cell(hit, question_folded=question_folded)]
    if table_id:
        useful_cells = [hit for hit in useful_cells if _table_id(hit) == table_id]
    for hit in useful_cells[:2]:
        _append_hit(selected, selected_ids, hit, limit)

    if len(selected) < limit:
        row_or_text_hits = [hit for hit in hits if _is_row_hit(hit)]
        if table_id:
            row_or_text_hits = [hit for hit in row_or_text_hits if _table_id(hit) == table_id]
        if not row_or_text_hits:
            row_or_text_hits = [hit for hit in hits if _is_text_hit(hit)]
        for hit in row_or_text_hits[:1]:
            _append_hit(selected, selected_ids, hit, limit)

    if selected:
        return selected
    return hits[: min(limit, len(hits))]


def _append_hit(selected: list[RetrievedHit], selected_ids: set[str], hit: RetrievedHit, limit: int) -> None:
    if len(selected) >= limit or hit.chunk_id in selected_ids:
        return
    selected.append(hit)
    selected_ids.add(hit.chunk_id)


def _is_markdown_table_hit(hit: RetrievedHit) -> bool:
    text = _hit_text(hit)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    pipe_lines = [line for line in lines if line.startswith("|") and line.endswith("|") and line.count("|") >= 2]
    has_separator = any(
        re.fullmatch(r"\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)+\|?", line)
        for line in lines
    )
    return len(pipe_lines) >= 2 and has_separator


def _is_useful_cell(hit: RetrievedHit, *, question_folded: str = "") -> bool:
    metadata = _hit_metadata(hit)
    target = str(metadata.get("citation_target") or "").lower()
    row_header = str(metadata.get("row_header") or "").strip()
    col_header = str(metadata.get("col_header") or "").strip()
    cell_text = str(metadata.get("cell_text") or "").strip()
    if target != "cell" and not (row_header and col_header and cell_text):
        return False
    if not row_header or not col_header or not cell_text:
        return False
    folded_cell = _fold(cell_text)
    if folded_cell in {_fold(row_header), _fold(col_header)}:
        return False
    return _column_range_matches_question(question_folded, col_header)


def _is_row_hit(hit: RetrievedHit) -> bool:
    metadata = _hit_metadata(hit)
    target = str(metadata.get("citation_target") or "").lower()
    return target == "row"


def _is_text_hit(hit: RetrievedHit) -> bool:
    metadata = _hit_metadata(hit)
    target = str(metadata.get("citation_target") or "").lower()
    block_type = str(hit.chunk.block_type or "").lower()
    return target not in {"table", "row", "cell"} and "table" not in block_type


def _table_id(hit: RetrievedHit) -> str:
    return str(_hit_metadata(hit).get("table_id") or "")


def _hit_metadata(hit: RetrievedHit) -> dict[str, object]:
    return {**dict(hit.chunk.metadata or {}), **dict(hit.metadata or {})}


def _hit_text(hit: RetrievedHit) -> str:
    return str(getattr(hit, "text", "") or hit.chunk.text or "")


def _column_range_matches_question(question_folded: str, col_header: str) -> bool:
    lookup_numbers = _lookup_numbers(question_folded)
    if not lookup_numbers:
        return True
    range_numbers = _numbers_as_float(_fold(col_header))
    if len(range_numbers) < 2:
        return True
    low, high = min(range_numbers[:2]), max(range_numbers[:2])
    return any(low <= value <= high for value in lookup_numbers)


def _lookup_numbers(question_folded: str) -> list[float]:
    cleaned = re.sub(r"\b(?:he|thang)\s*4\b", " ", question_folded)
    return _numbers_as_float(cleaned)


def _numbers_as_float(text: str) -> list[float]:
    values: list[float] = []
    for match in NUMBER_RE.finditer(text or ""):
        try:
            values.append(float(match.group(0).replace(",", ".")))
        except ValueError:
            continue
    return values
