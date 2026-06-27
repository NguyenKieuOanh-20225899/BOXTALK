from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from app.qa.context_builder import GroundedContext


NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:[,.]\d+)?")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
MONEY_RE = re.compile(
    r"(?:\b\d+(?:[,.]\d+)?\s*(?:dong|vnd|usd|trieu|nghin|ngan|million|k|m)\b|[$])",
    re.I,
)


@dataclass(frozen=True)
class AnswerValidationResult:
    valid: bool
    reason: str | None
    details: dict[str, Any] = field(default_factory=dict)


class AnswerValidator:
    """Deterministic post-checks for grounded LLM answers."""

    def validate(
        self,
        *,
        question: str,
        answer: str,
        used_evidence_ids: list[str],
        citations: list[dict[str, Any]],
        context: GroundedContext,
    ) -> AnswerValidationResult:
        answer_text = (answer or "").strip()
        if not answer_text:
            return self._invalid("empty_answer")
        if not used_evidence_ids:
            return self._invalid("missing_used_evidence_ids")

        context_by_id = context.item_by_id()
        invalid_ids = [eid for eid in used_evidence_ids if eid not in context_by_id]
        if invalid_ids:
            return self._invalid("unknown_evidence_id", invalid_evidence_ids=invalid_ids)
        if not citations:
            return self._invalid("missing_citations")

        cited_items = [context_by_id[eid] for eid in used_evidence_ids if eid in context_by_id]
        cited_text = "\n".join(item.text for item in cited_items)
        if not cited_text.strip():
            return self._invalid("empty_cited_evidence")

        question_folded = _fold(question)
        answer_folded = _fold(answer_text)
        evidence_folded = _fold(cited_text)

        if self._looks_like_section_number_answer(question_folded, answer_folded):
            return self._invalid("answer_looks_like_section_number")

        question_numbers = _numbers(question_folded)
        answer_numbers = _numbers(answer_folded) - question_numbers
        evidence_numbers = _numbers(evidence_folded)
        if answer_numbers and not answer_numbers <= evidence_numbers:
            return self._invalid(
                "answer_number_not_in_cited_evidence",
                answer_numbers=sorted(answer_numbers),
                evidence_numbers=sorted(evidence_numbers),
                question_numbers=sorted(question_numbers),
            )

        required_years = set(YEAR_RE.findall(question_folded))
        if required_years and not required_years <= set(YEAR_RE.findall(evidence_folded)):
            return self._invalid(
                "question_year_not_covered",
                required_years=sorted(required_years),
            )

        if _is_money_question(question_folded) and not MONEY_RE.search(evidence_folded):
            return self._invalid("money_question_without_money_amount")

        if _is_table_lookup_question(question_folded):
            if not self._has_table_grounding(cited_items):
                return self._invalid("table_question_without_table_grounding")

        return AnswerValidationResult(
            valid=True,
            reason=None,
            details={
                "used_evidence_ids": list(used_evidence_ids),
                "citation_count": len(citations),
            },
        )

    def _looks_like_section_number_answer(self, question_folded: str, answer_folded: str) -> bool:
        if not _expects_value(question_folded):
            return False
        cleaned = answer_folded.strip(" .")
        if re.fullmatch(r"(?:dieu|khoan|muc)?\s*\d+", cleaned):
            return True
        return bool(re.fullmatch(r"(?:hoc phi|tuition|fee)\s+\d+", cleaned))

    def _has_table_grounding(self, cited_items: list[Any]) -> bool:
        for item in cited_items:
            metadata = item.metadata or {}
            target = str(item.citation_target or metadata.get("citation_target") or "").lower()
            if target in {"cell", "row", "table"}:
                return True
            if metadata.get("cell_text") not in (None, ""):
                return True
            if metadata.get("table_id") not in (None, "") or metadata.get("is_table_chunk"):
                return True
        return False

    def _invalid(self, reason: str, **details: Any) -> AnswerValidationResult:
        return AnswerValidationResult(valid=False, reason=reason, details=details)


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = normalized.replace("đ", "d").replace("Đ", "D")
    folded = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    folded = folded.replace("≥", ">=").replace("≤", "<=").replace("–", "-").replace("−", "-")
    return re.sub(r"\s+", " ", folded).strip().lower()


def _numbers(text: str) -> set[str]:
    return {match.group(0).replace(",", ".") for match in NUMBER_RE.finditer(text or "")}


def _is_money_question(question_folded: str) -> bool:
    return any(
        term in question_folded
        for term in (
            "hoc phi",
            "le phi",
            "tuition",
            "fee",
            "money",
            "cost",
            "chi phi",
        )
    ) and any(term in question_folded for term in ("bao nhieu", "how much", "muc", "so tien"))


def _expects_value(question_folded: str) -> bool:
    return any(
        term in question_folded
        for term in (
            "bao nhieu",
            "how many",
            "how much",
            "diem",
            "tin chi",
            "hoc phi",
            "ti le",
            "ty le",
            "phan tram",
            "thoi gian",
            "nam nao",
            "muc nao",
            "tuong ung",
            "ung voi",
        )
    )


def _is_table_lookup_question(question_folded: str) -> bool:
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
            "muc nao",
            "khoang nao",
            "row",
            "column",
            "cell",
        )
    )
