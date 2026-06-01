from __future__ import annotations

import os
from typing import Any

from app.ingest.extract.table import cells_to_csv, cells_to_markdown
from app.ingest.schemas import BlockNode, ChunkNode


def table_aware_chunking_enabled() -> bool:
    value = os.getenv("BOXBIIBOO_ENABLE_TABLE_AWARE_CHUNKING", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def build_table_chunks(
    block: BlockNode,
    *,
    doc_id: str | None = None,
    start_index: int = 0,
) -> list[ChunkNode]:
    """Create table-specific chunks without changing paragraph chunking."""

    meta = dict(block.meta or {})
    table_id = str(meta.get("table_id") or f"page_{block.page_index + 1}_{block.block_id}")
    caption = _first_text(meta.get("caption"), meta.get("table_caption"), _caption_from_heading(block))
    cells = [cell for cell in meta.get("table_cells", []) if isinstance(cell, dict)]
    headers = [str(item) for item in meta.get("table_headers", []) if str(item).strip()]
    page = block.page_index + 1
    base_meta = {
        "doc_id": doc_id,
        "page": page,
        "block_type": "table",
        "table_id": table_id,
        "caption": caption,
        "source_bbox": block.bbox or meta.get("table_bbox"),
        "table_backend": meta.get("table_backend") or meta.get("backend"),
        "extraction_trace": meta.get("extraction_trace") or {},
        "is_table_chunk": True,
    }

    chunks: list[ChunkNode] = []
    index = start_index

    summary_text = _summary_text(table_id=table_id, page=page, caption=caption, headers=headers, meta=meta)
    chunks.append(_chunk(block, index, summary_text, "table_summary", base_meta, citation_target="table"))
    index += 1

    structure_text = (
        str(meta.get("table_markdown") or "").strip()
        or cells_to_markdown(cells)
        or str(meta.get("table_csv") or "").strip()
        or cells_to_csv(cells)
        or block.markdown
        or block.text
    )
    chunks.append(_chunk(block, index, structure_text, "table_structure", base_meta, citation_target="table"))
    index += 1

    for row_index, row_cells in _cells_by_row(cells).items():
        non_header_cells = [cell for cell in row_cells if not cell.get("is_header")]
        if not non_header_cells:
            continue
        row_header = _first_text(*(cell.get("row_header") for cell in non_header_cells))
        row_text = "; ".join(
            _cell_sentence(caption, page, row_header, cell.get("col_header"), cell.get("text"))
            for cell in non_header_cells
            if str(cell.get("text") or "").strip()
        )
        if row_text:
            row_meta = {
                **base_meta,
                "row_index": row_index,
                "row_header": row_header,
                "citation_target": "row",
            }
            chunks.append(_chunk(block, index, row_text, "table_row", row_meta, citation_target="row"))
            index += 1

        for cell in non_header_cells:
            text = str(cell.get("text") or "").strip()
            if not text:
                continue
            cell_meta = {
                **base_meta,
                "row_index": cell.get("row"),
                "col_index": cell.get("col"),
                "row_header": cell.get("row_header") or row_header,
                "col_header": cell.get("col_header"),
                "cell_text": text,
                "source_bbox": cell.get("bbox") or cell.get("grid_bbox") or base_meta["source_bbox"],
                "citation_target": "cell",
            }
            cell_text = _cell_sentence(caption, page, cell_meta["row_header"], cell_meta["col_header"], text)
            chunks.append(_chunk(block, index, cell_text, "table_cell", cell_meta, citation_target="cell"))
            index += 1

    return chunks


def _chunk(
    block: BlockNode,
    index: int,
    text: str,
    strategy: str,
    meta: dict[str, Any],
    *,
    citation_target: str,
) -> ChunkNode:
    chunk_meta = {**meta, "chunking_strategy": strategy, "citation_target": citation_target}
    return ChunkNode(
        chunk_id=f"chunk_{index:05d}",
        chunk_index=index,
        text=(text or "").strip(),
        markdown=(text or "").strip(),
        heading_path=list(block.heading_path or []),
        page_start=block.page_index,
        page_end=block.page_index,
        page_indices=[block.page_index],
        block_ids=[block.block_id],
        block_types=["table"],
        source_mode=block.source_mode,
        meta=chunk_meta,
    )


def _summary_text(*, table_id: str, page: int, caption: str | None, headers: list[str], meta: dict[str, Any]) -> str:
    label = f"Bảng {caption}" if caption else f"Bảng {table_id}"
    parts = [f"{label}, trang {page}."]
    if headers:
        parts.append("Các cột: " + ", ".join(headers) + ".")
    row_count = meta.get("table_row_count")
    col_count = meta.get("table_col_count")
    if row_count and col_count:
        parts.append(f"Kích thước: {row_count} hàng, {col_count} cột.")
    return " ".join(parts)


def _cell_sentence(
    caption: str | None,
    page: int,
    row_header: Any,
    col_header: Any,
    cell_text: Any,
) -> str:
    label = f"Bảng {caption}" if caption else "Bảng"
    row = str(row_header or "").strip() or "không rõ hàng"
    col = str(col_header or "").strip() or "không rõ cột"
    text = str(cell_text or "").strip()
    return f"{label}, trang {page}. Hàng {row}, cột {col}: {text}."


def _cells_by_row(cells: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    by_row: dict[int, list[dict[str, Any]]] = {}
    for cell in cells:
        try:
            row = int(cell.get("row"))
        except (TypeError, ValueError):
            continue
        by_row.setdefault(row, []).append(cell)
    return {row: sorted(items, key=lambda cell: int(cell.get("col") or 0)) for row, items in sorted(by_row.items())}


def _caption_from_heading(block: BlockNode) -> str | None:
    for heading in reversed(block.heading_path or []):
        if heading:
            return heading
    return None


def _first_text(*values: Any) -> str | None:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return None
