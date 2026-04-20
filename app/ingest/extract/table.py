from __future__ import annotations

from dataclasses import replace
from statistics import median

import fitz

from app.ingest.schemas import BlockNode


def extract_table_region(
    page: fitz.Page,
    bbox: tuple[float, float, float, float],
    *,
    block_index: int,
    reading_order: int | None = None,
    region_meta: dict | None = None,
) -> BlockNode | None:
    rect = fitz.Rect(bbox)
    if rect.is_empty or rect.width < 2 or rect.height < 2:
        return None

    rows = _extract_rows_from_words(page, rect)
    if rows:
        normalized_rows = _normalize_rows(rows)
        text = "\n".join(" | ".join(row) for row in normalized_rows).strip()
        markdown = _rows_to_markdown(normalized_rows)
        return BlockNode(
            block_id=f"p{page.number:04d}_b{block_index:04d}",
            page_index=page.number,
            block_type="table",
            text=text,
            markdown=markdown,
            reading_order=block_index if reading_order is None else reading_order,
            bbox=bbox,
            source_mode="layout",
            meta={
                **dict(region_meta or {}),
                "backend": "table_words",
                "table_row_count": len(normalized_rows),
                "table_col_count": max((len(row) for row in normalized_rows), default=0),
            },
        )

    fallback_text = page.get_text("text", clip=rect, sort=True).strip()
    if fallback_text:
        return BlockNode(
            block_id=f"p{page.number:04d}_b{block_index:04d}",
            page_index=page.number,
            block_type="table",
            text=fallback_text,
            markdown=_table_text_to_markdown(fallback_text),
            reading_order=block_index if reading_order is None else reading_order,
            bbox=bbox,
            source_mode="layout",
            meta={**dict(region_meta or {}), "backend": "table_clip_text"},
        )

    # OCR fallback still returns a table block, but notes that the text came
    # from OCR because the PDF region had no native words/text.
    from app.ingest.extract.ocr import extract_ocr_region

    ocr_block = extract_ocr_region(
        page,
        bbox,
        block_index=block_index,
        reading_order=reading_order,
        block_type_hint="table",
        region_meta={**dict(region_meta or {}), "table_backend": "ocr_fallback"},
    )
    if ocr_block is None:
        return None

    return replace(
        ocr_block,
        block_type="table",
        markdown=_table_text_to_markdown(ocr_block.text),
    )


def _extract_rows_from_words(page: fitz.Page, rect: fitz.Rect) -> list[list[str]]:
    raw_words = page.get_text("words", clip=rect, sort=True) or []
    if len(raw_words) < 4:
        return []

    words = [
        {
            "x0": float(word[0]),
            "y0": float(word[1]),
            "x1": float(word[2]),
            "y1": float(word[3]),
            "text": str(word[4]).strip(),
        }
        for word in raw_words
        if str(word[4]).strip()
    ]
    if len(words) < 4:
        return []

    heights = [w["y1"] - w["y0"] for w in words]
    y_tolerance = max(4.0, median(heights) * 0.65) if heights else 5.0

    row_groups: list[dict] = []
    for word in sorted(words, key=lambda item: ((item["y0"] + item["y1"]) / 2.0, item["x0"])):
        y_mid = (word["y0"] + word["y1"]) / 2.0
        if row_groups and abs(y_mid - row_groups[-1]["y_mid"]) <= y_tolerance:
            row_groups[-1]["words"].append(word)
            row_groups[-1]["y_mid"] = (
                row_groups[-1]["y_mid"] * (len(row_groups[-1]["words"]) - 1) + y_mid
            ) / len(row_groups[-1]["words"])
            continue

        row_groups.append({"y_mid": y_mid, "words": [word]})

    rows = [_split_row_into_cells(row["words"]) for row in row_groups]
    rows = [row for row in rows if any(cell.strip() for cell in row)]
    if len(rows) < 2:
        return []
    return rows


def _split_row_into_cells(words: list[dict]) -> list[str]:
    ordered = sorted(words, key=lambda item: item["x0"])
    widths = [max(1.0, item["x1"] - item["x0"]) for item in ordered]
    positive_gaps = [
        max(0.0, ordered[i]["x0"] - ordered[i - 1]["x1"])
        for i in range(1, len(ordered))
        if ordered[i]["x0"] > ordered[i - 1]["x1"]
    ]

    gap_threshold = max(12.0, median(widths) * 1.25) if widths else 12.0
    if positive_gaps:
        gap_threshold = max(gap_threshold, median(positive_gaps) * 1.4)

    cells: list[list[str]] = []
    current: list[str] = []

    for i, word in enumerate(ordered):
        if i > 0:
            gap = word["x0"] - ordered[i - 1]["x1"]
            if gap > gap_threshold and current:
                cells.append(current)
                current = []

        current.append(word["text"])

    if current:
        cells.append(current)

    return [" ".join(cell).strip() for cell in cells if " ".join(cell).strip()]


def _normalize_rows(rows: list[list[str]]) -> list[list[str]]:
    max_cols = max((len(row) for row in rows), default=0)
    if max_cols <= 1:
        return rows
    return [row + [""] * (max_cols - len(row)) for row in rows]


def _rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""

    max_cols = max((len(row) for row in rows), default=0)
    if len(rows) < 2 or max_cols <= 1:
        return _table_text_to_markdown("\n".join(" | ".join(row) for row in rows))

    header = _escape_cells(rows[0])
    body = [_escape_cells(row) for row in rows[1:]]
    separator = ["---"] * max_cols

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _escape_cells(row: list[str]) -> list[str]:
    return [cell.replace("|", "\\|").strip() for cell in row]


def _table_text_to_markdown(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    if len(lines) == 1:
        return lines[0]
    return "\n".join(f"- {line}" for line in lines)
