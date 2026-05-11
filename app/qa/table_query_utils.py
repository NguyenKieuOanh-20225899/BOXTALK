from __future__ import annotations

import re
import unicodedata

from app.qa.text_utils import normalize_text


LOOKUP_CUES: tuple[str, ...] = (
    "correspond",
    "corresponds",
    "mapped",
    "maps to",
    "equivalent",
    "belongs to",
    "which range",
    "what range",
    "which band",
    "what band",
    "which level",
    "what level",
    "which category",
    "what category",
    "which label",
    "what label",
    "what score",
    "what value",
    "what point",
    "range for",
    "score for",
    "value for",
    "threshold",
    "interval",
    "band",
    "lookup",
    "mapping",
    "ung voi",
    "tuong ung",
    "quy doi",
    "thuoc khoang",
    "thuoc muc",
    "thuoc loai",
    "khoang nao",
    "muc nao",
    "loai nao",
    "nhom nao",
    "nhan nao",
    "bao nhieu diem",
    "bao nhieu gia tri",
    "diem chu",
    "diem so",
    "thang diem",
    "thang 4",
    "gpa",
    "kpi",
)

QUESTION_CUES: tuple[str, ...] = (
    "what",
    "which",
    "how many",
    "how much",
    "bao nhieu",
    "nao",
    "gi",
    "may",
)

COMPARISON_CUES: tuple[str, ...] = (
    "compare",
    "difference",
    "different",
    "versus",
    " vs ",
    "so sanh",
    "khac nhau",
    "khac biet",
)

GENERIC_TABLE_QUERY_EXPANSION: tuple[str, ...] = (
    "table",
    "row",
    "column",
    "cell",
    "lookup",
    "mapping",
    "range",
    "interval",
    "threshold",
    "category",
    "label",
    "value",
    "score",
    "point",
    "bang",
    "cot",
    "hang",
    "quy doi",
    "khoang",
    "muc",
    "nhan",
    "gia tri",
)

SCORE_TABLE_EXPANSION: tuple[str, ...] = (
    "score",
    "point",
    "points",
    "grade",
    "grade point",
    "gpa",
    "diem so",
    "diem chu",
    "diem he",
    "thang diem",
)

PERCENT_TABLE_EXPANSION: tuple[str, ...] = (
    "percent",
    "percentage",
    "rate",
    "completion",
    "phan tram",
    "ty le",
)

SYMBOLIC_LABEL_RE = re.compile(r"(?<!\w)[A-Za-z][A-Za-z0-9]{0,5}[+-]?(?!\w)")
NUMERIC_VALUE_RE = re.compile(r"(?<![\w.])[-+]?\d+(?:[,.]\d+)?%?(?![\w.])")


def fold_query_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", normalize_text(text or ""))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_text).strip().lower()


def is_table_lookup_query(question: str) -> bool:
    """Detect table factoid/reverse lookup by query shape, not by document identity."""

    folded = fold_query_text(question)
    if not folded:
        return False

    has_lookup_cue = any(cue in folded for cue in LOOKUP_CUES)
    if not has_lookup_cue:
        return False

    has_question_cue = any(cue in folded for cue in QUESTION_CUES)
    has_value = bool(NUMERIC_VALUE_RE.search(question) or _has_symbolic_label(question))
    if not (has_question_cue or has_value):
        return False

    has_comparison_cue = any(cue in folded for cue in COMPARISON_CUES) or bool(re.search(r"\bthan\b", folded))
    if has_comparison_cue and not _has_direct_lookup_connector(folded):
        return False
    return True


def augment_table_lookup_query(question: str) -> str:
    if not is_table_lookup_query(question):
        return question

    folded = fold_query_text(question)
    terms = list(GENERIC_TABLE_QUERY_EXPANSION)
    if any(term in folded for term in ("score", "point", "grade", "gpa", "diem", "thang diem", "thang 4")):
        terms.extend(SCORE_TABLE_EXPANSION)
    if any(term in folded for term in ("percent", "percentage", "%", "phan tram", "ty le")):
        terms.extend(PERCENT_TABLE_EXPANSION)
    return normalize_text(f"{question} {' '.join(_dedupe(terms))}")


def _has_symbolic_label(text: str) -> bool:
    for match in SYMBOLIC_LABEL_RE.finditer(text):
        token = match.group(0)
        if len(token) <= 1 and not token.isupper():
            continue
        if token.lower() in {"what", "which", "how", "bao", "nhieu", "diem", "thang"}:
            continue
        return True
    return False


def _has_direct_lookup_connector(folded: str) -> bool:
    return any(cue in folded for cue in ("correspond", "mapped", "maps to", "ung voi", "tuong ung", "quy doi"))


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
