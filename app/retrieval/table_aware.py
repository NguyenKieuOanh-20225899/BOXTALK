from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from app.retrieval.schemas import DocumentChunkRef


TABLE_LOOKUP_CUES = (
    "tuong ung bao nhieu",
    "tương ứng bao nhiêu",
    "quy doi ra",
    "quy đổi ra",
    "khoang diem nao",
    "khoảng điểm nào",
    "muc",
    "mức",
    "la bao nhieu",
    "là bao nhiêu",
    "trong bang",
    "trong bảng",
    "cot",
    "cột",
    "hang",
    "hàng",
    "thang 4",
    "diem thang",
    "điểm thang",
)


@dataclass
class TableRetrievalTrace:
    query_type: str
    table_boost_applied: bool = False
    row_matched: str | None = None
    column_matched: str | None = None
    score: float = 0.0
    top_table_candidates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_type": self.query_type,
            "table_boost_applied": self.table_boost_applied,
            "row_matched": self.row_matched,
            "column_matched": self.column_matched,
            "score": self.score,
            "top_table_candidates": list(self.top_table_candidates),
        }


def table_aware_retrieval_enabled() -> bool:
    return os.getenv("BOXBIIBOO_ENABLE_TABLE_AWARE_RETRIEVAL", "").strip().lower() in {"1", "true", "yes", "on"}


def classify_table_query(question: str) -> str:
    folded = _fold(question)
    if not folded:
        return "general"
    if any(cue in folded for cue in ("quy doi ra", "quy đổi ra", "mapped to", "maps to")):
        return "table_reverse_lookup"
    if any(cue in folded for cue in TABLE_LOOKUP_CUES):
        return "table_lookup"
    if _has_symbolic_label(question) and any(term in folded for term in ("diem", "điểm", "score", "grade", "thang")):
        return "table_cell_lookup"
    return "general"


def table_aware_score(question: str, chunk: DocumentChunkRef) -> tuple[float, TableRetrievalTrace]:
    query_type = classify_table_query(question)
    trace = TableRetrievalTrace(query_type=query_type)
    if query_type == "general":
        return 0.0, trace

    metadata = dict(chunk.metadata or {})
    block_type = str(chunk.block_type or metadata.get("block_type") or "").lower()
    is_table = block_type == "table" or bool(metadata.get("is_table_chunk")) or metadata.get("table_id") is not None
    if not is_table:
        return -0.05, trace

    query = _fold(question)
    score = 0.30
    trace.table_boost_applied = True

    row_header = str(metadata.get("row_header") or "")
    col_header = str(metadata.get("col_header") or "")
    caption = str(metadata.get("caption") or metadata.get("table_caption") or "")
    cell_text = str(metadata.get("cell_text") or "")

    if row_header and _contains_phrase(query, row_header):
        score += 0.32
        trace.row_matched = row_header
    if col_header and _contains_phrase(query, col_header):
        score += 0.28
        trace.column_matched = col_header
    if caption and _overlap_ratio(query, caption) >= 0.35:
        score += 0.14
    if cell_text and query_type == "table_reverse_lookup" and _contains_phrase(query, cell_text):
        score += 0.28
    if metadata.get("citation_target") == "cell":
        score += 0.08
    elif metadata.get("citation_target") == "row":
        score += 0.05

    trace.score = max(-0.2, min(1.0, score))
    if metadata.get("table_id"):
        trace.top_table_candidates.append(str(metadata["table_id"]))
    return trace.score, trace


def _contains_phrase(haystack_folded: str, needle: str) -> bool:
    folded = _fold(needle)
    if not folded:
        return False
    if folded in haystack_folded:
        return True
    return bool(set(_tokens(folded)) & set(_tokens(haystack_folded)))


def _overlap_ratio(left: str, right: str) -> float:
    left_tokens = set(_tokens(left))
    right_tokens = set(_tokens(_fold(right)))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(right_tokens)


def _has_symbolic_label(text: str) -> bool:
    return bool(re.search(r"(?<!\w)[A-F][+-]?(?!\w)", text or "", flags=re.I))


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "").replace("đ", "d").replace("Đ", "D")
    folded = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", folded).strip().lower()
