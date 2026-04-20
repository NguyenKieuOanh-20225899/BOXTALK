from __future__ import annotations

import os
import re
import signal
from pathlib import Path

from app.ingest.schemas import BlockNode, PageNode


class LayoutTimeoutError(RuntimeError):
    pass


_CONVERTER = None


def _timeout_handler(signum, frame):
    raise LayoutTimeoutError("Docling timeout")


def _run_with_timeout(func, timeout_seconds: int = 12):
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        return func()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def _get_converter():
    global _CONVERTER
    if _CONVERTER is None:
        try:
            from docling.document_converter import DocumentConverter
        except Exception as e:
            raise RuntimeError(f"Docling is not installed: {e}") from e
        _CONVERTER = DocumentConverter()
    return _CONVERTER


def extract_with_layout_backend(pdf_path: str | Path) -> tuple[list[PageNode], list[BlockNode]]:
    pdf_path = Path(pdf_path)
    converter = _get_converter()
    timeout_seconds = int(os.getenv("BOXBIIBOO_DOCLING_TIMEOUT_SECONDS", "12"))

    try:
        result = _run_with_timeout(
            lambda: converter.convert(str(pdf_path)),
            timeout_seconds=timeout_seconds,
        )
    except LayoutTimeoutError:
        raise RuntimeError("Docling timeout")
    except Exception as e:
        raise RuntimeError(f"Docling failed: {e}") from e

    doc = result.document
    markdown = _extract_document_markdown(doc).strip()

    if not markdown:
        raise RuntimeError("Docling returned no markdown/text for layout extraction")

    raw_blocks = _markdown_to_blocks(markdown)

    blocks: list[BlockNode] = []
    page_index = 0

    for i, item in enumerate(raw_blocks):
        text = str(item.get("text") or "").strip()
        if not text:
            continue

        block_type = normalize_layout_block_type(str(item.get("block_type", "paragraph")))
        level = int(item.get("level", 0) or 0) if block_type == "heading" else None

        block = BlockNode(
            block_id=f"p{page_index:04d}_b{i:04d}",
            page_index=page_index,
            block_type=block_type,
            text=text,
            markdown=to_markdown_layout(text, block_type, level=level),
            reading_order=i,
            bbox=None,
            level=level,
            source_mode="layout",
            meta={"backend": "docling", "mode": "markdown_parse"},
        )
        blocks.append(block)

    if len(blocks) <= 1:
        raise RuntimeError("Layout backend produced too few blocks")

    page = PageNode(
        page_index=0,
        page_label="1",
        text="\n".join(b.text for b in blocks).strip(),
        markdown="\n\n".join(b.markdown for b in blocks if b.markdown).strip(),
        source_mode="layout",
        has_ocr=False,
        has_table=any(b.block_type == "table" for b in blocks),
        meta={"backend": "docling", "mode": "markdown_parse"},
    )

    return [page], blocks


def _extract_document_markdown(doc) -> str:
    for attr in ("export_to_markdown", "to_markdown"):
        if hasattr(doc, attr):
            try:
                md = getattr(doc, attr)()
                if isinstance(md, str) and md.strip():
                    return md.strip()
            except Exception:
                pass

    for attr in ("export_to_text", "text", "markdown", "md"):
        value = getattr(doc, attr, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def _markdown_to_blocks(markdown: str) -> list[dict]:
    lines = markdown.splitlines()
    blocks: list[dict] = []

    current_lines: list[str] = []
    current_type = "paragraph"

    def flush_current() -> None:
        nonlocal current_lines, current_type
        text = "\n".join(current_lines).strip()
        if text:
            blocks.append({"text": text, "block_type": current_type})
        current_lines = []
        current_type = "paragraph"

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            flush_current()
            i += 1
            continue

        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            flush_current()
            level = len(m.group(1))
            heading_text = m.group(2).strip()
            if heading_text:
                blocks.append({"text": heading_text, "block_type": "heading", "level": level})
            i += 1
            continue

        if re.match(r"^[-*+]\s+", stripped):
            flush_current()
            item_text = re.sub(r"^[-*+]\s+", "", stripped).strip()
            if item_text:
                blocks.append({"text": item_text, "block_type": "list_item"})
            i += 1
            continue

        if _looks_like_markdown_table_start(lines, i):
            flush_current()
            table_lines = [lines[i].rstrip(), lines[i + 1].rstrip()]
            i += 2
            while i < len(lines):
                row = lines[i].rstrip()
                if not row.strip():
                    break
                if "|" not in row:
                    break
                table_lines.append(row)
                i += 1
            table_text = "\n".join(table_lines).strip()
            if table_text:
                blocks.append({"text": table_text, "block_type": "table"})
            continue

        current_lines.append(line.rstrip())
        i += 1

    flush_current()
    return blocks


def _looks_like_markdown_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False

    header = lines[index].strip()
    separator = lines[index + 1].strip()

    if "|" not in header or "|" not in separator:
        return False

    normalized = separator.replace("|", "").replace(":", "").replace("-", "").replace(" ", "")
    return normalized == "" and "-" in separator


def normalize_layout_block_type(block_type: str) -> str:
    bt = (block_type or "").lower()

    if "header" in bt or "title" in bt or "heading" in bt:
        return "heading"
    if "list" in bt:
        return "list_item"
    if "table" in bt:
        return "table"
    if "caption" in bt:
        return "caption"
    if "figure" in bt or "image" in bt:
        return "figure"
    if "meta" in bt or "key-value" in bt:
        return "metadata"
    return "paragraph"


def to_markdown_layout(text: str, block_type: str, level: int | None = None) -> str:
    if not text:
        return ""

    if block_type == "heading":
        level = max(1, min(level or 2, 6))
        return f"{'#' * level} {text}"
    if block_type == "list_item":
        return f"- {text}"
    return text
