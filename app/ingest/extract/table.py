from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field, replace
from html import escape
from statistics import median
from typing import Any, Iterable

import fitz

from app.ingest.schemas import BlockNode

BBox = tuple[float, float, float, float]


@dataclass(slots=True)
class TableCell:
    row_index: int
    col_index: int
    row_span: int = 1
    col_span: int = 1
    bbox: BBox = (0.0, 0.0, 0.0, 0.0)
    text: str = ""
    confidence: float | None = None
    source_boxes: list[BBox] = field(default_factory=list)
    source_words: list[dict[str, Any]] = field(default_factory=list)
    grid_bbox: BBox | None = None
    page: int | None = None
    table_id: str | None = None

    def to_meta(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["row"] = payload.pop("row_index")
        payload["col"] = payload.pop("col_index")
        payload["is_header"] = self.row_index == 0
        return payload


@dataclass(slots=True)
class TableRow:
    row_index: int
    bbox: BBox
    cells: list[TableCell] = field(default_factory=list)


@dataclass(slots=True)
class Table:
    table_id: str
    page: int | None
    bbox: BBox
    rows: list[TableRow] = field(default_factory=list)

    @property
    def cells(self) -> list[TableCell]:
        return [cell for row in self.rows for cell in row.cells]

    @property
    def row_values(self) -> list[list[str]]:
        col_count = self.col_count
        values: list[list[str]] = []
        for row in self.rows:
            row_values = [""] * col_count
            for cell in row.cells:
                if cell.col_index < col_count:
                    row_values[cell.col_index] = cell.text
            values.append(row_values)
        return values

    @property
    def col_count(self) -> int:
        return max((cell.col_index + cell.col_span for cell in self.cells), default=0)


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

    effective_region_meta = dict(region_meta or {})
    force_hybrid_tatr = _hybrid_tatr_table_backend_forced()
    if not force_hybrid_tatr:
        grid = _extract_table_grid_from_words(page, rect)
        if grid:
            return _block_from_table_grid(
                page=page,
                bbox=bbox,
                block_index=block_index,
                reading_order=reading_order,
                region_meta=effective_region_meta,
                grid=grid,
            )
        clip_block = _block_from_table_clip_text(
            page=page,
            bbox=bbox,
            rect=rect,
            block_index=block_index,
            reading_order=reading_order,
            region_meta=effective_region_meta,
            require_structure=True,
        )
        if clip_block is not None:
            return clip_block

    if _hybrid_tatr_table_backend_enabled():
        try:
            from app.ingest.extract.hybrid_tatr_table import extract_hybrid_tatr_table_region

            hybrid_block = extract_hybrid_tatr_table_region(
                page,
                bbox,
                block_index=block_index,
                reading_order=reading_order,
                region_meta=effective_region_meta,
            )
            if hybrid_block is not None:
                return hybrid_block
            effective_region_meta["hybrid_tatr_skipped_reason"] = (
                "missing_pdf_word_boxes_or_structure"
            )
        except Exception as exc:
            effective_region_meta["hybrid_tatr_error"] = str(exc)

    grid = _extract_table_grid_from_words(page, rect)
    if grid:
        return _block_from_table_grid(
            page=page,
            bbox=bbox,
            block_index=block_index,
            reading_order=reading_order,
            region_meta=effective_region_meta,
            grid=grid,
        )

    clip_block = _block_from_table_clip_text(
        page=page,
        bbox=bbox,
        rect=rect,
        block_index=block_index,
        reading_order=reading_order,
        region_meta=effective_region_meta,
        require_structure=False,
    )
    if clip_block is not None:
        return clip_block

    # OCR fallback still returns a table block, but notes that the text came
    # from OCR because the PDF region had no native words/text.
    from app.ingest.extract.ocr import extract_ocr_region

    ocr_block = extract_ocr_region(
        page,
        bbox,
        block_index=block_index,
        reading_order=reading_order,
        block_type_hint="table",
        region_meta={**effective_region_meta, "table_backend": "ocr_fallback"},
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


def _hybrid_tatr_table_backend_enabled() -> bool:
    from app.ingest.extract.hybrid_tatr_table import is_hybrid_tatr_table_enabled

    return is_hybrid_tatr_table_enabled()


def _hybrid_tatr_table_backend_forced() -> bool:
    backend = os.getenv("BOXBIIBOO_TABLE_BACKEND", "").strip().lower()
    if backend == "hybrid_tatr":
        return True
    explicit = os.getenv("BOXBIIBOO_ENABLE_HYBRID_TATR_TABLES")
    return explicit is not None and explicit.strip().lower() not in {"0", "false", "no", "off"}


def _block_from_table_grid(
    *,
    page: fitz.Page,
    bbox: tuple[float, float, float, float],
    block_index: int,
    reading_order: int | None,
    region_meta: dict,
    grid: dict[str, Any],
) -> BlockNode:
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
            **region_meta,
            "backend": "table_words_grid",
            **structure,
        },
    )


def _block_from_table_clip_text(
    *,
    page: fitz.Page,
    bbox: tuple[float, float, float, float],
    rect: fitz.Rect,
    block_index: int,
    reading_order: int | None,
    region_meta: dict,
    require_structure: bool,
) -> BlockNode | None:
    fallback_text = page.get_text("text", clip=rect, sort=True).strip()
    if not fallback_text:
        return None
    structure = table_structure_from_text(fallback_text, backend="table_clip_text")
    if require_structure and (
        int(structure.get("table_row_count") or 0) < 2
        or int(structure.get("table_col_count") or 0) < 2
    ):
        return None
    return BlockNode(
        block_id=f"p{page.number:04d}_b{block_index:04d}",
        page_index=page.number,
        block_type="table",
        text=fallback_text,
        markdown=table_text_to_markdown(fallback_text),
        reading_order=block_index if reading_order is None else reading_order,
        bbox=bbox,
        source_mode="layout",
        meta={**region_meta, **structure},
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
    return _table_grid_from_cell_rows(cell_rows)


def table_structure_from_positioned_cells(
    cells: list[dict[str, Any]],
    *,
    backend: str,
    table_bbox: tuple[float, float, float, float] | None = None,
    page: int | None = None,
    table_id: str | None = None,
) -> dict[str, Any]:
    normalized_cells: list[dict[str, Any]] = []
    for cell in cells:
        text = str(cell.get("text") or "").strip()
        bbox = cell.get("bbox")
        if not text or not bbox or len(bbox) < 4:
            continue
        x0, y0, x1, y1 = [float(value) for value in bbox[:4]]
        if x1 <= x0 or y1 <= y0:
            continue
        normalized_cells.append(
            {
                "text": text,
                "bbox": (x0, y0, x1, y1),
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
            }
        )

    if len(normalized_cells) < 4:
        return {}

    table = _table_from_positioned_cells(
        normalized_cells,
        table_bbox=table_bbox,
        page=page,
        table_id=table_id or backend,
    )
    if table is None:
        return {}

    rows = table.row_values
    return table_structure_from_rows(
        rows,
        backend=backend,
        cells=table.cells,
        column_bounds=_column_bounds_from_table(table),
        row_bboxes=[row.bbox for row in table.rows],
        table_bbox=table.bbox,
    )


def _table_from_positioned_cells(
    cells: list[dict[str, Any]],
    *,
    table_bbox: BBox | None,
    page: int | None,
    table_id: str,
) -> Table | None:
    row_groups = _group_words_into_rows(cells)
    physical_rows = [
        [
            {
                "text": item["text"],
                "bbox": (item["x0"], item["y0"], item["x1"], item["y1"]),
                "x0": item["x0"],
                "y0": item["y0"],
                "x1": item["x1"],
                "y1": item["y1"],
            }
            for item in sorted(row["words"], key=lambda value: value["x0"])
        ]
        for row in row_groups
    ]
    physical_rows = [row for row in physical_rows if row]
    if len(physical_rows) < 2:
        return None

    all_boxes = [cell["bbox"] for row in physical_rows for cell in row]
    overall_bbox = table_bbox or _union_bbox(all_boxes)
    column_bands = _infer_column_bands(physical_rows, overall_bbox)
    if len(column_bands) < 2:
        grid = _table_grid_from_cell_rows(physical_rows)
        if not grid:
            return None
        return _table_from_legacy_grid(grid, table_id=table_id, page=page)

    logical_rows = _merge_continuation_rows(
        _trim_edge_noise_rows(physical_rows, column_bands),
        column_bands,
    )
    if len(logical_rows) < 2:
        return None

    table_rows: list[TableRow] = []
    for row_index, row in enumerate(logical_rows):
        row_bbox = _union_bbox(cell["bbox"] for cell in row)
        assigned: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for cell in row:
            span_start, span_end = _column_span_for_box(cell["bbox"], column_bands)
            assigned.setdefault((span_start, span_end), []).append(cell)

        row_cells: list[TableCell] = []
        for (span_start, span_end), source_cells in sorted(assigned.items()):
            source_cells.sort(key=lambda item: (item["y0"], item["x0"]))
            text = " ".join(item["text"] for item in source_cells if item["text"]).strip()
            if not text:
                continue
            bbox = (
                min(column_bands[span_start][0], min(item["bbox"][0] for item in source_cells)),
                min(row_bbox[1], min(item["bbox"][1] for item in source_cells)),
                max(column_bands[span_end][1], max(item["bbox"][2] for item in source_cells)),
                max(row_bbox[3], max(item["bbox"][3] for item in source_cells)),
            )
            row_cells.append(
                TableCell(
                    row_index=row_index,
                    col_index=span_start,
                    row_span=1,
                    col_span=span_end - span_start + 1,
                    bbox=bbox,
                    text=text,
                    confidence=_mean_optional(item.get("confidence") for item in source_cells),
                    source_boxes=[item["bbox"] for item in source_cells],
                    page=page,
                    table_id=table_id,
                )
            )

        if row_cells:
            table_rows.append(TableRow(row_index=row_index, bbox=_union_bbox(cell.bbox for cell in row_cells), cells=row_cells))

    if len(table_rows) < 2:
        return None
    return Table(
        table_id=table_id,
        page=page,
        bbox=_union_bbox(row.bbox for row in table_rows),
        rows=table_rows,
    )


def _table_from_legacy_grid(grid: dict[str, Any], *, table_id: str, page: int | None) -> Table | None:
    rows = grid.get("rows") or []
    cell_bboxes = grid.get("cell_bboxes") or []
    table_rows: list[TableRow] = []
    for row_index, row in enumerate(rows):
        cells: list[TableCell] = []
        for col_index, text in enumerate(row):
            if not text:
                continue
            bbox = None
            if row_index < len(cell_bboxes) and col_index < len(cell_bboxes[row_index]):
                bbox = cell_bboxes[row_index][col_index]
            if bbox is None:
                continue
            cells.append(
                TableCell(
                    row_index=row_index,
                    col_index=col_index,
                    bbox=bbox,
                    text=text,
                    source_boxes=[bbox],
                    page=page,
                    table_id=table_id,
                )
            )
        if cells:
            table_rows.append(TableRow(row_index=row_index, bbox=_union_bbox(cell.bbox for cell in cells), cells=cells))
    if not table_rows:
        return None
    return Table(table_id=table_id, page=page, bbox=_union_bbox(row.bbox for row in table_rows), rows=table_rows)


def _infer_column_bands(cell_rows: list[list[dict]], table_bbox: BBox) -> list[tuple[float, float]]:
    table_width = max(1.0, table_bbox[2] - table_bbox[0])
    candidates = [
        cell["bbox"]
        for row in cell_rows
        for cell in row
        if (cell["bbox"][2] - cell["bbox"][0]) <= table_width * 0.72
    ]
    if len(candidates) < 2:
        candidates = [cell["bbox"] for row in cell_rows for cell in row]
    all_candidates = list(candidates)

    widths = [max(1.0, box[2] - box[0]) for box in candidates]
    if widths:
        median_width = median(widths)
        narrow_limit = max(median_width * 2.4, table_width * 0.10)
        narrow_candidates = [box for box in candidates if (box[2] - box[0]) <= narrow_limit]
        if len(narrow_candidates) >= 2:
            candidates = narrow_candidates
            widths = [max(1.0, box[2] - box[0]) for box in candidates]
    tolerance = max(4.0, median(widths) * 0.30) if widths else 6.0
    bands: list[dict[str, float]] = []
    for box in sorted(candidates, key=lambda item: ((item[0] + item[2]) / 2.0, item[0])):
        matched = None
        for band in bands:
            if _intervals_same_column((band["x0"], band["x1"]), (box[0], box[2]), tolerance):
                matched = band
                break
        if matched is None:
            bands.append({"x0": box[0], "x1": box[2], "count": 1.0})
            continue
        matched["x0"] = min(matched["x0"], box[0])
        matched["x1"] = max(matched["x1"], box[2])
        matched["count"] += 1.0

    min_count = 2 if len(cell_rows) >= 4 else 1
    filtered = [band for band in bands if band["count"] >= min_count]
    if len(filtered) >= 2:
        bands = filtered

    bands = sorted(bands, key=lambda item: item["x0"])
    merged: list[tuple[float, float]] = []
    for band in bands:
        current = (float(band["x0"]), float(band["x1"]))
        if merged and current[0] <= merged[-1][1] + tolerance:
            merged[-1] = (min(merged[-1][0], current[0]), max(merged[-1][1], current[1]))
        else:
            merged.append(current)
    for box in all_candidates:
        if (box[2] - box[0]) > table_width * 0.45:
            continue
        overlaps = [
            idx
            for idx, band in enumerate(merged)
            if _interval_overlap((box[0], box[2]), band) >= min(box[2] - box[0], band[1] - band[0]) * 0.20
        ]
        if len(overlaps) == 1:
            idx = overlaps[0]
            merged[idx] = (min(merged[idx][0], box[0]), max(merged[idx][1], box[2]))
    return merged


def _intervals_same_column(left: tuple[float, float], right: tuple[float, float], tolerance: float) -> bool:
    overlap = min(left[1], right[1]) - max(left[0], right[0])
    if overlap > 0:
        min_width = max(1.0, min(left[1] - left[0], right[1] - right[0]))
        if overlap / min_width >= 0.25:
            return True
    left_center = (left[0] + left[1]) / 2.0
    right_center = (right[0] + right[1]) / 2.0
    if left[0] - tolerance <= right_center <= left[1] + tolerance:
        return True
    if right[0] - tolerance <= left_center <= right[1] + tolerance:
        return True
    return min(abs(right[0] - left[1]), abs(left[0] - right[1])) <= tolerance


def _trim_edge_noise_rows(cell_rows: list[list[dict]], column_bands: list[tuple[float, float]]) -> list[list[dict]]:
    if len(cell_rows) < 4 or len(column_bands) < 2:
        return cell_rows
    occupancy = [_row_occupied_columns(row, column_bands) for row in cell_rows]
    core_count = sum(1 for cols in occupancy if len(cols) >= 2)
    if core_count < 2:
        return cell_rows

    start = 0
    end = len(cell_rows)
    leading_noise = 0
    while leading_noise < end and _looks_edge_noise_row(cell_rows[leading_noise], occupancy[leading_noise], column_bands):
        leading_noise += 1
    if leading_noise >= 2 or (leading_noise == 1 and _looks_caption_like_row(cell_rows[0])):
        start = leading_noise

    trailing_noise = 0
    while end - trailing_noise - 1 >= start and _looks_edge_noise_row(
        cell_rows[end - trailing_noise - 1],
        occupancy[end - trailing_noise - 1],
        column_bands,
    ):
        trailing_noise += 1
    if trailing_noise >= 2:
        end -= trailing_noise
    return cell_rows[start:end] if start < end else cell_rows


def _looks_caption_like_row(row: list[dict]) -> bool:
    text = " ".join(str(cell.get("text") or "") for cell in row).strip().lower()
    return bool(re.match(r"^(table|figure|fig\.?)\s+\d+[\s.:;-]", text))


def _looks_edge_noise_row(
    row: list[dict],
    occupied_columns: set[int],
    column_bands: list[tuple[float, float]],
) -> bool:
    row_bbox = _union_bbox(cell["bbox"] for cell in row)
    table_width = max(1.0, column_bands[-1][1] - column_bands[0][0])
    row_width = row_bbox[2] - row_bbox[0]
    if row_width >= table_width * 0.80 and len(row) <= 2:
        return True
    if len(occupied_columns) >= 2:
        return False
    return len(row) <= 2


def _merge_continuation_rows(
    cell_rows: list[list[dict]],
    column_bands: list[tuple[float, float]],
) -> list[list[dict]]:
    if len(cell_rows) < 3:
        return cell_rows
    row_gaps = [
        max(0.0, _union_bbox(cell["bbox"] for cell in cell_rows[i + 1])[1] - _union_bbox(cell["bbox"] for cell in cell_rows[i])[3])
        for i in range(len(cell_rows) - 1)
    ]
    gap_limit = max(2.0, median(row_gaps) * 1.25) if row_gaps else 3.0
    merged: list[list[dict]] = []
    for row in cell_rows:
        occupied = _row_occupied_columns(row, column_bands)
        row_bbox = _union_bbox(cell["bbox"] for cell in row)
        if merged:
            previous_bbox = _union_bbox(cell["bbox"] for cell in merged[-1])
            previous_occupied = _row_occupied_columns(merged[-1], column_bands)
            vertical_gap = max(0.0, row_bbox[1] - previous_bbox[3])
            if (
                len(occupied) <= 1
                and occupied
                and occupied.issubset(previous_occupied or occupied)
                and vertical_gap <= gap_limit
            ):
                merged[-1].extend(row)
                continue
        merged.append(list(row))
    return merged


def _row_occupied_columns(row: list[dict], column_bands: list[tuple[float, float]]) -> set[int]:
    occupied: set[int] = set()
    for cell in row:
        start, end = _column_span_for_box(cell["bbox"], column_bands)
        occupied.update(range(start, end + 1))
    return occupied


def _column_span_for_box(box: BBox, column_bands: list[tuple[float, float]]) -> tuple[int, int]:
    center = (box[0] + box[2]) / 2.0
    overlapping = [
        idx
        for idx, band in enumerate(column_bands)
        if _interval_overlap((box[0], box[2]), band) >= min(box[2] - box[0], band[1] - band[0]) * 0.20
    ]
    if overlapping:
        return min(overlapping), max(overlapping)
    nearest = min(range(len(column_bands)), key=lambda idx: abs(center - (column_bands[idx][0] + column_bands[idx][1]) / 2.0))
    return nearest, nearest


def _interval_overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def _column_bounds_from_table(table: Table) -> list[tuple[float, float]]:
    bounds: list[tuple[float, float]] = []
    for col_index in range(table.col_count):
        boxes = [
            cell.bbox
            for cell in table.cells
            if cell.col_index <= col_index < cell.col_index + cell.col_span
        ]
        if not boxes:
            bounds.append((0.0, 0.0))
            continue
        bounds.append((min(box[0] for box in boxes), max(box[2] for box in boxes)))
    return bounds


def _mean_optional(values: Iterable[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    return (sum(numeric) / len(numeric)) if numeric else None


def _table_grid_from_cell_rows(cell_rows: list[list[dict]]) -> dict[str, Any] | None:
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
    heights = [max(1.0, w["y1"] - w["y0"]) for w in words]
    median_height = median(heights) if heights else 5.0
    y_tolerance = max(3.0, median_height * 0.50)
    row_groups: list[dict] = []
    for word in sorted(words, key=lambda item: ((item["y0"] + item["y1"]) / 2.0, item["x0"])):
        y_mid = (word["y0"] + word["y1"]) / 2.0
        best_row: dict | None = None
        best_score = -1.0
        for row in row_groups:
            overlap_ratio = _vertical_overlap_ratio((word["y0"], word["y1"]), (row["y0"], row["y1"]))
            center_diff = abs(y_mid - row["y_mid"])
            if overlap_ratio <= 0.35 and center_diff > y_tolerance:
                continue
            score = overlap_ratio + max(0.0, 1.0 - center_diff / max(median_height * 2.0, 1.0))
            if score > best_score:
                best_score = score
                best_row = row

        if best_row is not None:
            best_row["words"].append(word)
            best_row["y0"] = min(best_row["y0"], word["y0"])
            best_row["y1"] = max(best_row["y1"], word["y1"])
            best_row["y_mid"] = (
                best_row["y_mid"] * (len(best_row["words"]) - 1) + y_mid
            ) / len(best_row["words"])
            continue

        row_groups.append({"y0": word["y0"], "y1": word["y1"], "y_mid": y_mid, "words": [word]})
    return sorted(row_groups, key=lambda row: (row["y0"], row["y_mid"]))


def _vertical_overlap_ratio(left: tuple[float, float], right: tuple[float, float]) -> float:
    overlap = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    min_height = max(1.0, min(left[1] - left[0], right[1] - right[0]))
    return overlap / min_height


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
    cells: list[TableCell] | None = None,
    cell_bboxes: list[list[tuple[float, float, float, float] | None]] | None = None,
    column_bounds: list[tuple[float, float]] | None = None,
    row_bboxes: list[tuple[float, float, float, float]] | None = None,
    table_bbox: tuple[float, float, float, float] | None = None,
) -> dict[str, Any]:
    max_cols = max((len(row) for row in rows), default=0)
    headers = list(rows[0]) if rows and max_cols > 1 else []
    body_rows = rows[1:] if headers else rows
    meta_cells: list[dict[str, Any]] = []
    if cells is not None:
        meta_cells = [cell.to_meta() for cell in cells if cell.text]
    else:
        for row_index, row in enumerate(rows):
            for col_index, cell in enumerate(row):
                if not cell:
                    continue
                payload = {
                    "row": row_index,
                    "col": col_index,
                    "row_span": 1,
                    "col_span": 1,
                    "text": cell,
                    "is_header": bool(headers and row_index == 0),
                }
                if cell_bboxes and row_index < len(cell_bboxes) and col_index < len(cell_bboxes[row_index]):
                    bbox = cell_bboxes[row_index][col_index]
                    if bbox is not None:
                        payload["bbox"] = bbox
                        payload["source_boxes"] = [bbox]
                meta_cells.append(payload)

    csv_text = rows_to_csv(rows)
    html_text = cells_to_html(cells) if cells is not None else rows_to_html(rows)
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
        "table_bbox": table_bbox,
        "table_column_bounds": column_bounds or [],
        "table_row_bboxes": row_bboxes or [],
        "table_cells": meta_cells,
        "table_cell_count": len(meta_cells),
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


def cells_to_html(cells: list[TableCell] | None) -> str:
    if not cells:
        return ""
    by_row: dict[int, list[TableCell]] = {}
    for cell in cells:
        by_row.setdefault(cell.row_index, []).append(cell)
    lines = ["<table>"]
    for row_index in sorted(by_row):
        tag = "th" if row_index == 0 and len(by_row) > 1 else "td"
        parts = []
        for cell in sorted(by_row[row_index], key=lambda item: item.col_index):
            attrs = ""
            if cell.row_span > 1:
                attrs += f' rowspan="{cell.row_span}"'
            if cell.col_span > 1:
                attrs += f' colspan="{cell.col_span}"'
            parts.append(f"<{tag}{attrs}>{escape(cell.text)}</{tag}>")
        lines.append("  <tr>" + "".join(parts) + "</tr>")
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
