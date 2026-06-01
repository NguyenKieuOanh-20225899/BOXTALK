from __future__ import annotations

from typing import Any

from app.retrieval.schemas import RetrievedHit


def format_evidence_citation(hit: RetrievedHit) -> dict[str, Any]:
    metadata = {**dict(hit.chunk.metadata or {}), **dict(hit.metadata or {})}
    citation = {
        "chunk_id": hit.chunk_id,
        "doc_id": hit.chunk.doc_id,
        "source_name": hit.chunk.source_name,
        "page": hit.page,
        "section": hit.section,
        "heading_path": hit.heading_path,
        "score": round(float(hit.final_score or hit.score), 4),
    }
    if not _is_table_evidence(hit, metadata):
        return citation

    citation.update(
        {
            "block_type": "table",
            "table_id": metadata.get("table_id"),
            "caption": metadata.get("caption") or metadata.get("table_caption"),
            "row_index": metadata.get("row_index"),
            "col_index": metadata.get("col_index"),
            "row_header": metadata.get("row_header"),
            "col_header": metadata.get("col_header"),
            "cell_text": metadata.get("cell_text"),
            "bbox": metadata.get("source_bbox") or metadata.get("bbox"),
            "citation_target": metadata.get("citation_target") or "table",
        }
    )
    citation["citation_text"] = table_citation_text(citation)
    return citation


def table_citation_text(evidence: dict[str, Any]) -> str:
    page = evidence.get("page")
    table_id = evidence.get("table_id") or "không rõ"
    caption = evidence.get("caption")
    row_header = evidence.get("row_header")
    col_header = evidence.get("col_header")
    table_label = f"bảng '{caption}'" if caption else f"bảng {table_id}"
    if row_header and col_header:
        return f"Trang {page}, {table_label}, hàng '{row_header}', cột '{col_header}'."
    if row_header:
        return f"Trang {page}, {table_label}, hàng '{row_header}'."
    if col_header:
        return f"Trang {page}, {table_label}, cột '{col_header}'."
    return f"Trang {page}, bảng {table_id}."


def _is_table_evidence(hit: RetrievedHit, metadata: dict[str, Any]) -> bool:
    return (
        str(hit.chunk.block_type or "").lower() == "table"
        or bool(metadata.get("is_table_chunk"))
        or metadata.get("table_id") is not None
    )
