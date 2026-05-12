from __future__ import annotations

import re
from pathlib import Path

import fitz

from app.ingest.reading_order import sort_in_reading_order
from app.ingest.schemas import BlockNode, PageNode


LIST_ITEM_RE = re.compile(r"^(?:[-*•]\s+|\d+[\.)]\s+|[A-Za-z][\.)]\s+|[IVXLCDMivxlcdm]+[\.)]\s+)\S+")
NUMBERED_HEADING_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+\S+")
LEGAL_HEADING_RE = re.compile(
    r"^(?:chương|chuong|phần|phan|mục|muc|điều|dieu|khoản|khoan)\s+[0-9A-Za-zIVXLCDMivxlcdm]+[\.:]?\s*\S*",
    re.I,
)
CAPTION_RE = re.compile(r"^(?:figure|fig\.|hình|bảng|table)\s+\d+(?:[\.:]\s*|\s+-\s+).+", re.I)
METADATA_RE = re.compile(r"^[^:\n]{1,80}:\s+\S+")


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

        raw_blocks = sort_in_reading_order(
            raw_blocks,
            bbox_getter=lambda b: (float(b[0]), float(b[1]), float(b[2]), float(b[3])),
            page_width=float(page.rect.width),
            page_height=float(page.rect.height),
        )

        for reading_order, raw in enumerate(raw_blocks):
            x0, y0, x1, y1, text, *_ = raw
            text = (text or "").strip()
            if not text:
                continue

            bbox = (float(x0), float(y0), float(x1), float(y1))
            block_type = _guess_text_block_type(text, bbox=bbox, page_rect=page.rect)
            meta = {"backend": "pymupdf"}
            if block_type == "table":
                from app.ingest.extract.table import table_structure_from_text

                meta.update(table_structure_from_text(text, backend="text_table"))

            block = BlockNode(
                block_id=f"p{page_index:04d}_b{reading_order:04d}",
                page_index=page_index,
                block_type=block_type,
                text=text,
                markdown=_to_markdown(text, block_type),
                reading_order=reading_order,
                bbox=bbox,
                source_mode="text",
                meta=meta,
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

    block_type = _resolve_text_block_type(text, block_type_hint, bbox=bbox)

    meta = dict(region_meta or {})
    meta.setdefault("backend", "pymupdf_region")
    if block_type == "table":
        from app.ingest.extract.table import table_structure_from_text

        meta.update(table_structure_from_text(text, backend="text_region_table"))

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


def _guess_text_block_type(
    text: str,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    page_rect: fitz.Rect | None = None,
) -> str:
    s = text.strip()

    if not s:
        return "paragraph"

    if bbox is not None and page_rect is not None and _looks_like_header_footer(s, bbox, page_rect):
        return "metadata"

    if _looks_like_table_text(s):
        return "table"

    if CAPTION_RE.match(s):
        return "caption"

    if METADATA_RE.match(s):
        return "metadata"

    if LIST_ITEM_RE.match(s):
        return "list_item"

    if len(s) < 140 and (
        s.isupper()
        or NUMBERED_HEADING_RE.match(s)
        or LEGAL_HEADING_RE.match(s)
    ):
        return "heading"

    return "paragraph"


def _looks_like_table_text(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    if sum(1 for line in lines if "|" in line) >= 2:
        return True
    return sum(1 for line in lines if len(re.split(r"\s{2,}|\t", line)) >= 3) >= 2


def _resolve_text_block_type(
    text: str,
    block_type_hint: str | None,
    *,
    bbox: tuple[float, float, float, float] | None = None,
) -> str:
    hinted = (block_type_hint or "").strip().lower()
    if hinted in {"heading", "list_item", "table", "caption", "figure", "metadata"}:
        return hinted
    return _guess_text_block_type(text, bbox=bbox)


def _looks_like_header_footer(
    text: str,
    bbox: tuple[float, float, float, float],
    page_rect: fitz.Rect,
) -> bool:
    page_height = max(float(page_rect.height), 1.0)
    y0, y1 = bbox[1], bbox[3]
    near_top = y1 <= page_height * 0.075
    near_bottom = y0 >= page_height * 0.925
    if not (near_top or near_bottom):
        return False
    if len(text) > 160:
        return False
    return bool(re.search(r"\d|page|trang|copyright|confidential|draft", text, re.I)) or len(text.split()) <= 8


def _to_markdown(text: str, block_type: str) -> str:
    if block_type == "heading":
        return f"## {text}"
    if block_type == "list_item":
        return text if text.startswith(("- ", "* ", "• ")) else f"- {text}"
    if block_type == "table":
        from app.ingest.extract.table import table_text_to_markdown

        return table_text_to_markdown(text)
    return text
