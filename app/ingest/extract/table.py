from __future__ import annotations

import os
import re
from dataclasses import replace
from html import escape
from statistics import median
from typing import Any, Iterable

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

    grid = _extract_table_grid_from_words(page, rect)
    if grid:
        normalized_rows = grid["rows"]
        text = "\n".join(" | ".join(row) for row in normalized_rows).strip()
        markdown = _rows_to_markdown(normalized_rows)
        structure = table_structure_from_rows(
            normalized_rows,
            backend="table_words_grid",
            cell_bboxes=grid.get("cell_bboxes"),
            column_bounds=grid.get("column_bounds"),
            row_bboxes=grid.get("row_bboxes"),
        )
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
                "backend": "table_words_grid",
                **structure,
            },
        )

    fallback_text = page.get_text("text", clip=rect, sort=True).strip()
    if fallback_text:
        structure = table_structure_from_text(fallback_text, backend="table_clip_text")
        return BlockNode(
            block_id=f"p{page.number:04d}_b{block_index:04d}",
            page_index=page.number,
            block_type="table",
            text=fallback_text,
            markdown=table_text_to_markdown(fallback_text),
            reading_order=block_index if reading_order is None else reading_order,
            bbox=bbox,
            source_mode="layout",
            meta={**dict(region_meta or {}), **structure},
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

    structure = table_structure_from_text(ocr_block.text, backend="ocr_table_text")
    return replace(
        ocr_block,
        block_type="table",
        markdown=table_text_to_markdown(ocr_block.text),
        meta={**dict(ocr_block.meta or {}), **structure},
    )


def _extract_table_grid_from_words(page: fitz.Page, rect: fitz.Rect) -> dict[str, Any] | None:
    raw_words = page.get_text("words", clip=rect, sort=True) or []
    if len(raw_words) < 4:
        return None

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
        return None

    row_groups = _group_words_into_rows(words)
    cell_rows = [_split_row_into_cell_infos(row["words"]) for row in row_groups]
    cell_rows = [row for row in cell_rows if row]
    if len(cell_rows) < 2:
        return None
    if sum(1 for row in cell_rows if len(row) >= 2) < 2:
        return None

    column_anchors = _infer_column_anchors(cell_rows)
    if len(column_anchors) < 2:
        return None
    if len(column_anchors) > int(os.getenv("BOXBIIBOO_TABLE_MAX_INFERRED_COLS", "12")):
        return None

    rows: list[list[str]] = []
    cell_bboxes: list[list[tuple[float, float, float, float] | None]] = []
    row_bboxes: list[tuple[float, float, float, float]] = []
    for cells in cell_rows:
        row = [""] * len(column_anchors)
        bboxes: list[tuple[float, float, float, float] | None] = [None] * len(column_anchors)
        for cell in cells:
            col = _nearest_anchor_index(cell["x0"], column_anchors)
            if row[col]:
                row[col] = f"{row[col]} {cell['text']}".strip()
                bboxes[col] = _merge_bbox(bboxes[col], cell["bbox"])
            else:
                row[col] = cell["text"]
                bboxes[col] = cell["bbox"]
        if any(value.strip() for value in row):
            rows.append(row)
            cell_bboxes.append(bboxes)
            row_bboxes.append(_union_bbox(bbox for bbox in bboxes if bbox is not None))

    trim_left, trim_right = _empty_edge_bounds(rows)
    rows = [row[trim_left:trim_right] for row in rows]
    cell_bboxes = [row[trim_left:trim_right] for row in cell_bboxes[: len(rows)]]
    if len(rows) < 2 or max((len(row) for row in rows), default=0) < 2:
        return None

    return {
        "rows": rows,
        "cell_bboxes": cell_bboxes,
        "row_bboxes": row_bboxes[: len(rows)],
        "column_bounds": _column_bounds_from_cells(cell_bboxes, len(rows[0])),
    }


def _extract_rows_from_words(page: fitz.Page, rect: fitz.Rect) -> list[list[str]]:
    grid = _extract_table_grid_from_words(page, rect)
    return list(grid["rows"]) if grid else []


def _group_words_into_rows(words: list[dict]) -> list[dict]:
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
    return row_groups


def _split_row_into_cell_infos(words: list[dict]) -> list[dict]:
    grouped = _split_row_words(words)
    cells: list[dict] = []
    for group in grouped:
        text = " ".join(word["text"] for word in group).strip()
        if not text:
            continue
        bbox = _union_bbox((word["x0"], word["y0"], word["x1"], word["y1"]) for word in group)
        cells.append({"text": text, "bbox": bbox, "x0": bbox[0], "x1": bbox[2]})
    return cells


def _split_row_into_cells(words: list[dict]) -> list[str]:
    return [" ".join(word["text"] for word in group).strip() for group in _split_row_words(words)]


def _split_row_words(words: list[dict]) -> list[list[dict]]:
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

    cells: list[list[dict]] = []
    current: list[dict] = []

    for i, word in enumerate(ordered):
        if i > 0:
            gap = word["x0"] - ordered[i - 1]["x1"]
            if gap > gap_threshold and current:
                cells.append(current)
                current = []

        current.append(word)

    if current:
        cells.append(current)

    return [cell for cell in cells if cell]


def _infer_column_anchors(cell_rows: list[list[dict]]) -> list[float]:
    tolerance = float(os.getenv("BOXBIIBOO_TABLE_COLUMN_TOLERANCE", "18"))
    anchors: list[dict[str, float]] = []
    for cell in sorted((cell for row in cell_rows for cell in row), key=lambda item: item["x0"]):
        matched = None
        for anchor in anchors:
            if abs(cell["x0"] - anchor["x0"]) <= tolerance:
                matched = anchor
                break
        if matched is None:
            anchors.append({"x0": cell["x0"], "count": 1.0})
            continue
        matched["x0"] = (matched["x0"] * matched["count"] + cell["x0"]) / (matched["count"] + 1.0)
        matched["count"] += 1.0

    min_count = 2 if len(cell_rows) >= 3 else 1
    frequent = [anchor["x0"] for anchor in anchors if anchor["count"] >= min_count]
    if len(frequent) >= 2:
        return sorted(frequent)

    widest_row = max(cell_rows, key=len)
    return [cell["x0"] for cell in widest_row]


def _nearest_anchor_index(x0: float, anchors: list[float]) -> int:
    return min(range(len(anchors)), key=lambda idx: abs(x0 - anchors[idx]))


def _empty_edge_bounds(rows: list[list[str]]) -> tuple[int, int]:
    if not rows:
        return (0, 0)
    left = 0
    right = max(len(row) for row in rows)
    while left < right and all(left >= len(row) or not row[left].strip() for row in rows):
        left += 1
    while right > left and all(right - 1 >= len(row) or not row[right - 1].strip() for row in rows):
        right -= 1
    return (left, right)


def _merge_bbox(
    left: tuple[float, float, float, float] | None,
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    if left is None:
        return right
    return (
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    )


def _union_bbox(boxes: Iterable[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    box_list = list(boxes)
    return (
        min(box[0] for box in box_list),
        min(box[1] for box in box_list),
        max(box[2] for box in box_list),
        max(box[3] for box in box_list),
    )


def _column_bounds_from_cells(
    cell_bboxes: list[list[tuple[float, float, float, float] | None]],
    column_count: int,
) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = []
    for col_index in range(column_count):
        boxes = [
            row[col_index]
            for row in cell_bboxes
            if col_index < len(row) and row[col_index] is not None
        ]
        if not boxes:
            bounds.append((0.0, 0.0))
            continue
        bounds.append((min(box[0] for box in boxes), max(box[2] for box in boxes)))
    return bounds


def _normalize_rows(rows: list[list[str]]) -> list[list[str]]:
    return normalize_table_rows(rows)


def normalize_table_rows(rows: list[list[str]]) -> list[list[str]]:
    max_cols = max((len(row) for row in rows), default=0)
    if max_cols <= 1:
        return rows
    return [row + [""] * (max_cols - len(row)) for row in rows]


def table_structure_from_text(text: str, *, backend: str = "table_text") -> dict[str, Any]:
    rows = _rows_from_text_lines([line.strip() for line in text.splitlines() if line.strip()])
    normalized_rows = normalize_table_rows(rows) if rows else []
    return table_structure_from_rows(normalized_rows, backend=backend)


def table_structure_from_rows(
    rows: list[list[str]],
    *,
    backend: str,
    cell_bboxes: list[list[tuple[float, float, float, float] | None]] | None = None,
    column_bounds: list[tuple[float, float]] | None = None,
    row_bboxes: list[tuple[float, float, float, float]] | None = None,
) -> dict[str, Any]:
    max_cols = max((len(row) for row in rows), default=0)
    headers = list(rows[0]) if rows and max_cols > 1 else []
    body_rows = rows[1:] if headers else rows
    cells: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        for col_index, cell in enumerate(row):
            if not cell:
                continue
            payload = {
            "row": row_index,
            "col": col_index,
            "text": cell,
            "is_header": bool(headers and row_index == 0),
        }
            if cell_bboxes and row_index < len(cell_bboxes) and col_index < len(cell_bboxes[row_index]):
                bbox = cell_bboxes[row_index][col_index]
                if bbox is not None:
                    payload["bbox"] = bbox
            cells.append(payload)

    csv_text = rows_to_csv(rows)
    html_text = rows_to_html(rows)
    return {
        "backend": backend,
        "table_backend": backend,
        "table_row_count": len(rows),
        "table_col_count": max_cols,
        "table_headers": headers,
        "table_body_row_count": len(body_rows),
        "table_rows": rows,
        "table_records": _rows_to_records(headers, body_rows),
        "table_csv": csv_text,
        "table_html": html_text,
        "table_column_bounds": column_bounds or [],
        "table_row_bboxes": row_bboxes or [],
        "table_cells": cells,
        "table_cell_count": len(cells),
    }


def _rows_to_records(headers: list[str], body_rows: list[list[str]]) -> list[dict[str, str]]:
    if not headers:
        return []
    records: list[dict[str, str]] = []
    for row in body_rows:
        record = {
            header: row[index] if index < len(row) else ""
            for index, header in enumerate(headers)
            if header
        }
        if record:
            records.append(record)
    return records


def rows_to_csv(rows: list[list[str]]) -> str:
    import csv
    import io

    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(rows)
    return output.getvalue().strip()


def rows_to_html(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    lines = ["<table>"]
    for row_index, row in enumerate(rows):
        tag = "th" if row_index == 0 and len(rows) > 1 else "td"
        lines.append("  <tr>" + "".join(f"<{tag}>{escape(cell)}</{tag}>" for cell in row) + "</tr>")
    lines.append("</table>")
    return "\n".join(lines)


def _rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""

    max_cols = max((len(row) for row in rows), default=0)
    if len(rows) < 2 or max_cols <= 1:
        return table_text_to_markdown("\n".join(" | ".join(row) for row in rows))

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


def table_text_to_markdown(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    if len(lines) == 1:
        return lines[0]

    rows = _rows_from_text_lines(lines)
    if rows:
        return _rows_to_markdown(normalize_table_rows(rows))

    return "\n".join(f"- {line}" for line in lines)


def _rows_from_text_lines(lines: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    token_rows: list[list[str]] = []
    for line in lines:
        if "|" in line:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
        elif "\t" in line:
            cells = [cell.strip() for cell in line.split("\t")]
        else:
            cells = [cell.strip() for cell in re.split(r"\s{2,}", line)]
            if len(cells) < 2:
                token_rows.append(line.split())
        cells = [cell for cell in cells if cell]
        if len(cells) < 2:
            continue
        rows.append(cells)

    if len(rows) < 2:
        return _infer_rows_from_tokens(token_rows)
    return rows


def _infer_rows_from_tokens(token_rows: list[list[str]]) -> list[list[str]]:
    token_rows = [tokens for tokens in token_rows if len(tokens) >= 3]
    if len(token_rows) < 2:
        return []

    expected_cols = _infer_column_count(token_rows)
    if expected_cols < 2:
        return []

    inferred = [_split_tokens_into_columns(tokens, expected_cols) for tokens in token_rows]
    if any(len(row) != expected_cols for row in inferred):
        return []
    return inferred


def _infer_column_count(token_rows: list[list[str]]) -> int:
    first_row_cols = _compound_header_column_count(token_rows[0])
    if first_row_cols:
        return first_row_cols

    short_counts = [len(tokens) for tokens in token_rows if 2 <= len(tokens) <= 5]
    if not short_counts:
        return 0
    candidate = min(short_counts)
    return candidate if 2 <= candidate <= 5 else 0


def _compound_header_column_count(tokens: list[str]) -> int:
    if len(tokens) < 3:
        return 0
    merged = _merge_compound_header_tokens(tokens)
    return len(merged) if len(merged) != len(tokens) and len(merged) >= 2 else 0


def _split_tokens_into_columns(tokens: list[str], expected_cols: int) -> list[str]:
    if len(tokens) == expected_cols + 1:
        merged_headers = _merge_compound_header_tokens(tokens)
        if len(merged_headers) == expected_cols:
            return merged_headers

    if len(tokens) == expected_cols:
        return tokens
    if expected_cols == 3:
        return _split_tokens_into_three_columns(tokens)
    if len(tokens) < expected_cols:
        return tokens + [""] * (expected_cols - len(tokens))
    head = tokens[: expected_cols - 1]
    tail = " ".join(tokens[expected_cols - 1 :])
    return [*head, tail]


def _split_tokens_into_three_columns(tokens: list[str]) -> list[str]:
    if len(tokens) <= 3:
        return tokens + [""] * (3 - len(tokens))
    if len(tokens) == 4:
        if _looks_like_duration(tokens[1], tokens[2]):
            return [tokens[0], f"{tokens[1]} {tokens[2]}", tokens[3]]
        if _looks_like_duration(tokens[2], tokens[3]):
            return [f"{tokens[0]} {tokens[1]}", f"{tokens[2]} {tokens[3]}", ""]
        return [f"{tokens[0]} {tokens[1]}", tokens[2], tokens[3]]
    if len(tokens) == 5:
        if _looks_like_duration(tokens[2], tokens[3]):
            return [f"{tokens[0]} {tokens[1]}", f"{tokens[2]} {tokens[3]}", tokens[4]]
        return [f"{tokens[0]} {tokens[1]}", tokens[2], " ".join(tokens[3:])]
    if _looks_like_duration(tokens[2], tokens[3]):
        return [f"{tokens[0]} {tokens[1]}", f"{tokens[2]} {tokens[3]}", " ".join(tokens[4:])]
    return [f"{tokens[0]} {tokens[1]}", f"{tokens[2]} {tokens[3]}", " ".join(tokens[4:])]


def _looks_like_duration(left: str, right: str) -> bool:
    return bool(re.match(r"^\d+(?:[.,]\d+)?$", left)) and right.lower() in {
        "day",
        "days",
        "week",
        "weeks",
        "month",
        "months",
        "year",
        "years",
        "ngay",
        "tuan",
        "thang",
        "nam",
    }


def _merge_compound_header_tokens(tokens: list[str]) -> list[str]:
    compound_headers = {
        ("waiting", "period"),
        ("due", "date"),
        ("start", "date"),
        ("end", "date"),
        ("effective", "date"),
        ("risk", "owner"),
        ("table", "name"),
    }
    merged: list[str] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens) and (tokens[i].lower(), tokens[i + 1].lower()) in compound_headers:
            merged.append(f"{tokens[i]} {tokens[i + 1]}")
            i += 2
            continue
        merged.append(tokens[i])
        i += 1
    return merged
