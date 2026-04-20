from __future__ import annotations

import re

from app.ingest.schemas import BlockNode


def enrich_structure(blocks: list[BlockNode]) -> list[BlockNode]:
    current_headings: list[str] = []
    heading_stack: list[BlockNode] = []

    for block in blocks:
        if block.meta is None:
            block.meta = {}

        if block.block_type == "heading":
            level = block.level or _detect_heading_level(block.text)
            block.level = level

            while len(current_headings) >= level:
                current_headings.pop()

            while len(heading_stack) >= level:
                heading_stack.pop()

            parent_block_id = heading_stack[-1].block_id if heading_stack else None
            block.parent_block_id = parent_block_id

            current_headings.append(block.text.strip())
            heading_stack.append(block)

        block.heading_path = current_headings.copy()
        block.meta["heading_path"] = current_headings.copy()
        block.item_number = _extract_item_number(block.text)

    return blocks


def _detect_heading_level(text: str) -> int:
    s = text.strip()

    if re.match(r"^\d+\.\d+\.\d+", s):
        return 3
    if re.match(r"^\d+\.\d+", s):
        return 2
    if re.match(r"^\d+\.", s):
        return 1
    return 1


def _extract_item_number(text: str) -> str | None:
    s = text.strip()
    m = re.match(r"^(\d+(?:\.\d+)*)(?:[.)]|\s)", s)
    if m:
        return m.group(1)
    return None
