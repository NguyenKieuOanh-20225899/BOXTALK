from __future__ import annotations

from app.ingest.schemas import BlockNode, PageNode


def normalize_pages_blocks(
    pages: list[PageNode],
    blocks: list[BlockNode],
) -> tuple[list[PageNode], list[BlockNode]]:
    normalized_blocks: list[BlockNode] = []

    for i, block in enumerate(blocks):
        if not block.block_type:
            block.block_type = "paragraph"

        if block.markdown is None:
            block.markdown = block.text

        if block.reading_order is None:
            block.reading_order = i

        if block.meta is None:
            block.meta = {}

        block.block_type = normalize_block_type(block.block_type)
        normalized_blocks.append(block)

    normalized_pages: list[PageNode] = []
    for page in pages:
        if page.meta is None:
            page.meta = {}
        normalized_pages.append(page)

    return normalized_pages, normalized_blocks


def normalize_block_type(block_type: str) -> str:
    bt = (block_type or "").lower()

    if bt in {"heading", "paragraph", "list_item", "table", "caption", "figure", "metadata"}:
        return bt

    if "title" in bt or "header" in bt:
        return "heading"
    if "list" in bt:
        return "list_item"
    if "table" in bt:
        return "table"
    if "caption" in bt:
        return "caption"
    if "figure" in bt or "image" in bt:
        return "figure"
    if "meta" in bt:
        return "metadata"

    return "paragraph"
