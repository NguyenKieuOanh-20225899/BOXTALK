from __future__ import annotations

from pathlib import Path

import fitz

from app.ingest.schemas import BlockNode, PageNode


def extract_with_text_backend(pdf_path: str | Path) -> tuple[list[PageNode], list[BlockNode]]:
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))

    pages: list[PageNode] = []
    blocks: list[BlockNode] = []

    for page in doc:
        page_index = page.number
        page_label = str(page_index + 1)

        raw_blocks = page.get_text("blocks") or []
        page_blocks: list[BlockNode] = []

        # sort reading order top-to-bottom, left-to-right
        raw_blocks = sorted(raw_blocks, key=lambda b: (float(b[1]), float(b[0])))

        for i, raw in enumerate(raw_blocks):
            x0, y0, x1, y1, text, *_ = raw
            text = (text or "").strip()
            if not text:
                continue

            block_type = _guess_text_block_type(text)

            block = BlockNode(
                block_id=f"p{page_index:04d}_b{i:04d}",
                page_index=page_index,
                block_type=block_type,
                text=text,
                markdown=_to_markdown(text, block_type),
                reading_order=i,
                bbox=(float(x0), float(y0), float(x1), float(y1)),
                source_mode="text",
                meta={"backend": "pymupdf"},
            )
            page_blocks.append(block)

        page_text = "\n".join(b.text for b in page_blocks).strip()
        page_md = "\n\n".join(b.markdown for b in page_blocks if b.markdown).strip()

        pages.append(
            PageNode(
                page_index=page_index,
                page_label=page_label,
                text=page_text,
                markdown=page_md,
                source_mode="text",
                has_ocr=False,
                has_table=any(b.block_type == "table" for b in page_blocks),
                meta={"backend": "pymupdf"},
            )
        )
        blocks.extend(page_blocks)

    doc.close()
    return pages, blocks


def extract_text_region(
    page_or_region: fitz.Page | dict,
    bbox_or_page_index: tuple[float, float, float, float] | int | None = None,
    block_index: int = 0,
    *,
    reading_order: int | None = None,
    block_type_hint: str | None = None,
    region_meta: dict | None = None,
) -> BlockNode | None:
    """
    Supports two call styles:
    - extract_text_region(page, bbox, ...)
    - extract_text_region(region_dict, page_index, ...)
    """
    bbox: tuple[float, float, float, float] | None = None
    page_index = 0
    text = ""

    if isinstance(page_or_region, dict):
        region = page_or_region
        bbox = region.get("bbox")
        page_index = int(
            bbox_or_page_index
            if isinstance(bbox_or_page_index, int)
            else region.get("page_index", 0)
        )
        text = str(region.get("text") or "").strip()
        if block_type_hint is None:
            block_type_hint = str(region.get("block_type") or "").strip() or None
        region_meta = {**dict(region.get("meta") or {}), **dict(region_meta or {})}
    else:
        page = page_or_region
        if not isinstance(bbox_or_page_index, tuple):
            raise TypeError("bbox is required when extracting text from a fitz.Page")
        bbox = bbox_or_page_index
        page_index = page.number
        text = extract_text_in_bbox(page, bbox)

    if not text:
        return None

    block_type = _resolve_text_block_type(text, block_type_hint)

    meta = dict(region_meta or {})
    meta.setdefault("backend", "pymupdf_region")

    return BlockNode(
        block_id=f"p{page_index:04d}_b{block_index:04d}",
        page_index=page_index,
        block_type=block_type,
        text=text,
        markdown=_to_markdown(text, block_type),
        reading_order=block_index if reading_order is None else reading_order,
        bbox=bbox,
        source_mode="text",
        meta=meta,
    )


def extract_text_in_bbox(
    page: fitz.Page,
    bbox: tuple[float, float, float, float],
) -> str:
    rect = fitz.Rect(bbox)
    if rect.is_empty or rect.width < 2 or rect.height < 2:
        return ""

    text = page.get_textbox(rect).strip()
    if text:
        return text

    return page.get_text("text", clip=rect, sort=True).strip()


def _guess_text_block_type(text: str) -> str:
    s = text.strip()

    if not s:
        return "paragraph"

    if s.startswith(("- ", "* ", "• ")):
        return "list_item"

    if len(s) < 120 and (s.isupper() or s.startswith(("1.", "2.", "3.", "4.", "5."))):
        return "heading"

    if "|" in s and "\n" in s:
        return "table"

    return "paragraph"


def _resolve_text_block_type(text: str, block_type_hint: str | None) -> str:
    hinted = (block_type_hint or "").strip().lower()
    if hinted in {"heading", "list_item", "table", "caption", "figure", "metadata"}:
        return hinted
    return _guess_text_block_type(text)


def _to_markdown(text: str, block_type: str) -> str:
    if block_type == "heading":
        return f"## {text}"
    if block_type == "list_item":
        return text if text.startswith(("- ", "* ", "• ")) else f"- {text}"
    return text
