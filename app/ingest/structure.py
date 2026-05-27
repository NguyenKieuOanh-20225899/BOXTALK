from __future__ import annotations

import re

from app.ingest.schemas import BlockNode


def enrich_structure(blocks: list[BlockNode]) -> list[BlockNode]:
    heading_stack: list[BlockNode] = []

    for block in blocks:
        if block.meta is None:
            block.meta = {}

        if block.block_type == "heading":
            level = block.level or _detect_heading_level(block.text)
            block.level = level

            while heading_stack and (heading_stack[-1].level or 1) >= level:
                heading_stack.pop()

            parent_block_id = heading_stack[-1].block_id if heading_stack else None
            block.parent_block_id = parent_block_id

            heading_stack.append(block)

        current_headings = [heading.text.strip() for heading in heading_stack if heading.text.strip()]
        block.heading_path = current_headings
        block.meta["heading_path"] = current_headings.copy()
        block.item_number = _extract_item_number(block.text)

    return blocks


def _detect_heading_level(text: str) -> int:
    s = text.strip()
    lowered = s.lower()

    if re.match(r"^(?:chương|chuong|phần|phan)\b", lowered):
        return 1
    if re.match(r"^(?:mục|muc)\b", lowered):
        return 2
    if re.match(r"^(?:điều|dieu)\b", lowered):
        return 3

    if re.match(r"^\d+\.\d+\.\d+", s):
        return 6
    if re.match(r"^\d+\.\d+", s):
        return 5
    if re.match(r"^\d+[\.)]?\s+\S+", s):
        return 4
    return 1


def _extract_item_number(text: str) -> str | None:
    s = text.strip()
    m = re.match(r"^(\d+(?:\.\d+)*)(?:[.)]|\s)", s)
    if m:
        return m.group(1)
    return None
