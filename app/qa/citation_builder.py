from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.qa.context_builder import GroundedContext


@dataclass(frozen=True)
class CitationBuildResult:
    citations: list[dict[str, Any]]
    invalid_evidence_ids: list[str] = field(default_factory=list)


class CitationBuilder:
    """Map LLM-returned evidence IDs to real retrieval metadata."""

    def build(
        self,
        *,
        context: GroundedContext,
        used_evidence_ids: list[str],
    ) -> CitationBuildResult:
        by_id = context.item_by_id()
        citations: list[dict[str, Any]] = []
        invalid_ids: list[str] = []
        seen: set[tuple[str, str]] = set()

        for evidence_id in used_evidence_ids:
            item = by_id.get(evidence_id)
            if item is None:
                invalid_ids.append(evidence_id)
                continue
            dedupe_key = (item.evidence_id, item.chunk_id)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            citations.append(
                {
                    "evidence_id": item.evidence_id,
                    "source_name": item.source_name,
                    "doc_id": item.doc_id,
                    "page": item.page,
                    "section": item.section,
                    "chunk_id": item.chunk_id,
                    "citation_target": item.citation_target,
                    "metadata": dict(item.metadata),
                }
            )

        return CitationBuildResult(citations=citations, invalid_evidence_ids=invalid_ids)
