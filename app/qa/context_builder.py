from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from app.retrieval.schemas import RetrievedHit


DEFAULT_MAX_CONTEXT_CHARS = 12000
DEFAULT_MAX_EVIDENCE_CHARS = 2400
DEFAULT_MAX_EVIDENCE_ITEMS = 5


@dataclass(frozen=True)
class EvidenceContextItem:
    evidence_id: str
    text: str
    source_name: str | None
    doc_id: str | None
    page: int | None
    section: str | None
    chunk_id: str
    citation_target: str | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_prompt_text(self) -> str:
        lines = [
            f"EVIDENCE {self.evidence_id}:",
            f"Source: {self.source_name or ''}",
            f"Document ID: {self.doc_id or ''}",
            f"Page: {self.page if self.page is not None else ''}",
            f"Section: {self.section or ''}",
            f"Chunk ID: {self.chunk_id}",
        ]
        if self.citation_target:
            lines.append(f"Citation target: {self.citation_target}")
        for key, label in (
            ("table_id", "Table ID"),
            ("row_header", "Row header"),
            ("col_header", "Column header"),
            ("cell_text", "Cell text"),
        ):
            value = self.metadata.get(key)
            if value not in (None, "", []):
                lines.append(f"{label}: {value}")
        if "\n" in self.text:
            lines.append(f"Content:\n{self.text}")
        else:
            lines.append(f"Content: {self.text}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "text": self.text,
            "source_name": self.source_name,
            "doc_id": self.doc_id,
            "page": self.page,
            "section": self.section,
            "chunk_id": self.chunk_id,
            "citation_target": self.citation_target,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class GroundedContext:
    question: str
    evidence: list[EvidenceContextItem]
    token_count: int | None = None

    @property
    def evidence_ids(self) -> list[str]:
        return [item.evidence_id for item in self.evidence]

    def item_by_id(self) -> dict[str, EvidenceContextItem]:
        return {item.evidence_id: item for item in self.evidence}

    def to_prompt_text(self) -> str:
        parts = ["QUESTION:", self.question, ""]
        for item in self.evidence:
            parts.append(item.to_prompt_text())
            parts.append("")
        return "\n".join(parts).strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "evidence": [item.to_dict() for item in self.evidence],
            "token_count": self.token_count,
        }


class ContextBuilder:
    """Build a compact, metadata-preserving context from selected evidence."""

    def __init__(
        self,
        *,
        max_context_chars: int | None = None,
        max_evidence_chars: int | None = None,
        max_evidence_items: int | None = None,
    ) -> None:
        self.max_context_chars = max_context_chars or _int_env(
            "BOXTALK_CONTEXT_MAX_CHARS",
            DEFAULT_MAX_CONTEXT_CHARS,
        )
        self.max_evidence_chars = max_evidence_chars or _int_env(
            "BOXTALK_CONTEXT_MAX_EVIDENCE_CHARS",
            DEFAULT_MAX_EVIDENCE_CHARS,
        )
        self.max_evidence_items = max_evidence_items or _int_env(
            "BOXTALK_CONTEXT_MAX_EVIDENCE_ITEMS",
            DEFAULT_MAX_EVIDENCE_ITEMS,
        )

    def build(self, *, question: str, selected_hits: list[RetrievedHit]) -> GroundedContext:
        evidence: list[EvidenceContextItem] = []
        seen_chunk_ids: set[str] = set()
        current_chars = len(question)

        for hit in selected_hits:
            if hit.chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(hit.chunk_id)
            if len(evidence) >= self.max_evidence_items:
                break

            metadata = {**dict(hit.chunk.metadata or {}), **dict(hit.metadata or {})}
            raw_text = str(hit.text or "")
            item = EvidenceContextItem(
                evidence_id=f"E{len(evidence) + 1}",
                text=_compact_text(
                    raw_text,
                    self.max_evidence_chars,
                    preserve_newlines=_looks_like_markdown_table(raw_text),
                ),
                source_name=hit.chunk.source_name,
                doc_id=hit.chunk.doc_id,
                page=hit.page,
                section=hit.section,
                chunk_id=hit.chunk_id,
                citation_target=_citation_target(hit, metadata),
                metadata=metadata,
            )
            item_chars = len(item.to_prompt_text())
            if evidence and current_chars + item_chars > self.max_context_chars:
                break
            evidence.append(item)
            current_chars += item_chars

        prompt_text = GroundedContext(question=question, evidence=evidence).to_prompt_text()
        return GroundedContext(
            question=question,
            evidence=evidence,
            token_count=_approx_token_count(prompt_text),
        )


def _citation_target(hit: RetrievedHit, metadata: dict[str, Any]) -> str | None:
    value = metadata.get("citation_target")
    if value not in (None, ""):
        return str(value)
    if metadata.get("cell_text") not in (None, ""):
        return "cell"
    if metadata.get("row_header") not in (None, ""):
        return "row"
    block_type = str(hit.chunk.block_type or "").strip()
    return block_type or None


def _compact_text(text: str, max_chars: int, *, preserve_newlines: bool = False) -> str:
    if preserve_newlines:
        compact = "\n".join(
            re.sub(r"[ \t]+", " ", line).strip()
            for line in (text or "").splitlines()
            if line.strip()
        )
    else:
        compact = re.sub(r"\s+", " ", text or "").strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "..."


def _looks_like_markdown_table(text: str) -> bool:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    pipe_lines = [line for line in lines if line.startswith("|") and line.endswith("|") and line.count("|") >= 2]
    has_separator = any(
        re.fullmatch(r"\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)+\|?", line)
        for line in lines
    )
    return len(pipe_lines) >= 2 and has_separator


def _approx_token_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default
