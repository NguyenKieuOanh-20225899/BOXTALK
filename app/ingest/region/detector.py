from __future__ import annotations

import os
import re
from statistics import median
from typing import Any, Iterable

import fitz

from app.ingest.reading_order import sort_in_reading_order

BBox = tuple[float, float, float, float]

CAPTION_RE = re.compile(r"^(?:figure|fig\.?|table|hinh|bang)\s+\d+(?:[\.:]\s*|\s+-\s+).+", re.I)
HEADING_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+\S+")
LIST_RE = re.compile(r"^(?:[-*+]\s+|\d+[\.)]\s+|[A-Za-z][\.)]\s+)\S+")


def detect_regions(page: fitz.Page) -> list[dict[str, Any]]:
    """Detect page-level regions and assign an initial routing kind.

    The detector is intentionally deterministic. It uses native PDF geometry
    first (text blocks, image rects, PyMuPDF table finder when available), then
    conservative heuristics to group table-like text rows. The output is a list
    of region dicts consumed by the region-routed extractor.
    """

    text_regions = _detect_text_regions(page)
    table_bboxes = _detect_table_bboxes(page, text_regions)
    page_area = max(1.0, float(page.rect.width * page.rect.height))

    regions: list[dict[str, Any]] = []

    for table_index, bbox in enumerate(table_bboxes):
        text = page.get_text("text", clip=fitz.Rect(bbox), sort=True).strip()
        regions.append(
            {
                "region_id": f"p{page.number:04d}_table_{table_index:04d}",
                "type": "table",
                "kind": "table",
                "block_type": "table",
                "bbox": bbox,
                "text": text,
                "page_index": page.number,
                "route_backend": "table",
                "detection_source": "native_or_text_cluster",
                "confidence": 1.0,
            }
        )

    for region in text_regions:
        bbox = tuple(region["bbox"])
        if _overlaps_any(bbox, table_bboxes, threshold=0.45):
            continue

        kind = _classify_text_region(
            str(region.get("text") or ""),
            bbox,
            page_rect=page.rect,
            page_area=page_area,
        )
        block_type = "metadata" if kind in {"header", "footer"} else kind
        route_backend = "text"
        regions.append(
            {
                **region,
                "type": kind,
                "kind": kind,
                "block_type": block_type,
                "route_backend": route_backend,
                "detection_source": "pdf_text_block",
                "confidence": 1.0,
            }
        )

    text_bboxes = [tuple(region["bbox"]) for region in regions if region.get("kind") not in {"image"}]
    regions.extend(_detect_image_regions(page, text_bboxes=text_bboxes, table_bboxes=table_bboxes))
    regions.extend(_detect_vector_figure_regions(page, text_bboxes=text_bboxes, table_bboxes=table_bboxes))

    return sort_in_reading_order(
        regions,
        bbox_getter=lambda item: tuple(item["bbox"]),
        page_width=float(page.rect.width),
        page_height=float(page.rect.height),
    )


def _detect_text_regions(page: fitz.Page) -> list[dict[str, Any]]:
    raw_blocks = page.get_text("blocks") or []
    regions: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_blocks):
        if len(raw) < 5:
            continue
        x0, y0, x1, y1, text, *_ = raw
        text = str(text or "").strip()
        if not text:
            continue
        bbox = (float(x0), float(y0), float(x1), float(y1))
        if _area(bbox) <= 0:
            continue
        regions.append(
                {
                    "region_id": f"p{page.number:04d}_text_{idx:04d}",
                    "type": "text",
                    "kind": "text",
                    "block_type": "paragraph",
                    "bbox": bbox,
                    "text": text,
                    "page_index": page.number,
                    "confidence": 1.0,
                }
            )
    return regions


def _detect_table_bboxes(page: fitz.Page, text_regions: list[dict[str, Any]]) -> list[BBox]:
    bboxes: list[BBox] = []
    bboxes.extend(_detect_native_table_bboxes(page))
    bboxes.extend(_detect_text_table_bboxes(text_regions, page_width=float(page.rect.width)))
    return _merge_nearby_bboxes(
        bboxes,
        x_tolerance=max(4.0, float(page.rect.width) * 0.01),
        y_tolerance=max(4.0, float(page.rect.height) * 0.006),
    )


def _detect_native_table_bboxes(page: fitz.Page) -> list[BBox]:
    try:
        finder = page.find_tables()
    except Exception:
        return []

    tables = getattr(finder, "tables", None) or []
    bboxes: list[BBox] = []
    for table in tables:
        bbox = getattr(table, "bbox", None)
        if not bbox or len(bbox) < 4:
            continue
        normalized = _normalize_bbox(bbox)
        if _area(normalized) > 0:
            bboxes.append(normalized)
    return bboxes


def _detect_text_table_bboxes(text_regions: list[dict[str, Any]], *, page_width: float) -> list[BBox]:
    if not text_regions:
        return []

    ordered = sorted(text_regions, key=lambda item: (item["bbox"][1], item["bbox"][0]))
    bboxes: list[BBox] = []

    for region in ordered:
        text = str(region.get("text") or "")
        bbox = tuple(region["bbox"])
        if _looks_like_table_text(text):
            bboxes.append(bbox)

    row_candidates = [
        region
        for region in ordered
        if _looks_like_table_row(str(region.get("text") or ""), page_width=page_width, bbox=tuple(region["bbox"]))
    ]
    if len(row_candidates) < 2:
        return bboxes

    heights = [max(1.0, float(region["bbox"][3] - region["bbox"][1])) for region in row_candidates]
    median_height = median(heights) if heights else 10.0
    max_gap = max(4.0, median_height * 2.25)

    clusters: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for region in row_candidates:
        if not current:
            current = [region]
            continue
        previous_bbox = tuple(current[-1]["bbox"])
        bbox = tuple(region["bbox"])
        vertical_gap = bbox[1] - previous_bbox[3]
        horizontal_overlap = _horizontal_overlap_ratio(previous_bbox, bbox)
        if vertical_gap <= max_gap and horizontal_overlap >= 0.18:
            current.append(region)
            continue
        clusters.append(current)
        current = [region]
    if current:
        clusters.append(current)

    for cluster in clusters:
        if len(cluster) < 2:
            continue
        if sum(1 for item in cluster if _table_row_column_count(str(item.get("text") or "")) >= 2) < 2:
            continue
        bbox = _union_bbox(tuple(item["bbox"]) for item in cluster)
        width_ratio = max(0.0, bbox[2] - bbox[0]) / max(page_width, 1.0)
        if width_ratio < 0.18:
            continue
        bboxes.append(bbox)

    return bboxes


def _detect_image_regions(
    page: fitz.Page,
    *,
    text_bboxes: list[BBox],
    table_bboxes: list[BBox],
) -> list[dict[str, Any]]:
    page_area = max(1.0, float(page.rect.width * page.rect.height))
    image_min_area = float(os.getenv("BOXBIIBOO_REGION_IMAGE_MIN_AREA", "1600"))
    regions: list[dict[str, Any]] = []
    for image_index, image in enumerate(page.get_images(full=True) or []):
        xref = int(image[0])
        try:
            rects = page.get_image_rects(xref) or []
        except Exception:
            rects = []

        for rect_index, rect in enumerate(rects):
            bbox = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
            if _area(bbox) < image_min_area:
                continue
            if _overlaps_any(bbox, table_bboxes, threshold=0.20):
                continue
            if _overlaps_any(bbox, text_bboxes, threshold=0.35):
                continue
            regions.append(
                {
                    "region_id": f"p{page.number:04d}_image_{image_index:04d}_{rect_index:04d}",
                    "type": "image",
                    "kind": "image",
                    "block_type": "figure",
                    "bbox": bbox,
                    "text": "",
                    "page_index": page.number,
                    "route_backend": "ocr",
                    "image_area_ratio": _area(bbox) / page_area,
                    "detection_source": "pdf_image_rect",
                    "confidence": 1.0,
                }
            )
    return regions


def _detect_vector_figure_regions(
    page: fitz.Page,
    *,
    text_bboxes: list[BBox],
    table_bboxes: list[BBox],
) -> list[dict[str, Any]]:
    enabled = os.getenv("BOXBIIBOO_ENABLE_REGION_VECTOR_FIGURES", "0").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return []

    page_area = max(1.0, float(page.rect.width * page.rect.height))
    min_area = float(os.getenv("BOXBIIBOO_REGION_VECTOR_MIN_AREA", "1200"))
    regions: list[dict[str, Any]] = []
    try:
        drawings = page.get_drawings() or []
    except Exception:
        return []

    for drawing_index, drawing in enumerate(drawings):
        rect = drawing.get("rect") if isinstance(drawing, dict) else None
        if rect is None:
            continue
        bbox = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
        if _area(bbox) < min_area:
            continue
        if _overlaps_any(bbox, table_bboxes, threshold=0.20):
            continue
        if _overlaps_any(bbox, text_bboxes, threshold=0.20):
            continue
        regions.append(
            {
                "region_id": f"p{page.number:04d}_vector_{drawing_index:04d}",
                "type": "image",
                "kind": "image",
                "block_type": "figure",
                "bbox": bbox,
                "text": "",
                "page_index": page.number,
                "route_backend": "placeholder",
                "has_text_regions": True,
                "image_area_ratio": _area(bbox) / page_area,
                "detection_source": "pdf_vector_drawing",
                "confidence": 1.0,
            }
        )
    return regions


def _classify_text_region(text: str, bbox: BBox, *, page_rect: fitz.Rect, page_area: float) -> str:
    stripped = text.strip()
    if not stripped:
        return "empty"

    page_height = max(float(page_rect.height), 1.0)
    if _looks_like_header_footer(stripped, bbox, page_height=page_height):
        return "header" if bbox[3] <= page_height * 0.10 else "footer"
    if CAPTION_RE.match(stripped):
        return "caption"
    if _looks_like_table_text(stripped):
        return "table"
    if LIST_RE.match(stripped):
        return "list_item"
    if len(stripped) < 140 and (stripped.isupper() or HEADING_RE.match(stripped)):
        return "heading"
    if _area(bbox) / max(page_area, 1.0) <= 0.002 and len(stripped.split()) <= 4:
        return "metadata"
    return "paragraph"


def _looks_like_header_footer(text: str, bbox: BBox, *, page_height: float) -> bool:
    near_top = bbox[3] <= page_height * 0.075
    near_bottom = bbox[1] >= page_height * 0.925
    if not (near_top or near_bottom):
        return False
    if len(text) > 160:
        return False
    return bool(re.search(r"\d|page|trang|copyright|confidential|draft", text, re.I)) or len(text.split()) <= 8


def _looks_like_table_text(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    return sum(1 for line in lines if _table_row_column_count(line) >= 2) >= 2


def _looks_like_table_row(text: str, *, page_width: float, bbox: BBox) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if len(stripped.split()) > 16:
        return False
    if _table_row_column_count(stripped) >= 2:
        return True
    width_ratio = max(0.0, bbox[2] - bbox[0]) / max(page_width, 1.0)
    tokens = stripped.split()
    if 3 <= len(tokens) <= 10 and width_ratio >= 0.28:
        numeric_tokens = sum(1 for token in tokens if re.search(r"\d", token))
        short_tokens = sum(1 for token in tokens if len(token) <= 12)
        return numeric_tokens >= 1 or short_tokens == len(tokens)
    return False


def _table_row_column_count(text: str) -> int:
    if "|" in text:
        return len([cell for cell in text.strip("|").split("|") if cell.strip()])
    if "\t" in text:
        return len([cell for cell in text.split("\t") if cell.strip()])
    return len([cell for cell in re.split(r"\s{2,}", text.strip()) if cell.strip()])


def _merge_nearby_bboxes(
    bboxes: list[BBox],
    *,
    x_tolerance: float,
    y_tolerance: float,
) -> list[BBox]:
    merged: list[BBox] = []
    for bbox in sorted((_normalize_bbox(bbox) for bbox in bboxes), key=lambda item: (item[1], item[0])):
        if _area(bbox) <= 0:
            continue
        match_index = None
        for idx, existing in enumerate(merged):
            if _should_merge_bbox(existing, bbox, x_tolerance=x_tolerance, y_tolerance=y_tolerance):
                match_index = idx
                break
        if match_index is None:
            merged.append(bbox)
        else:
            merged[match_index] = _union_bbox([merged[match_index], bbox])
    return merged


def _should_merge_bbox(left: BBox, right: BBox, *, x_tolerance: float, y_tolerance: float) -> bool:
    if _intersection_area(left, right) > 0:
        smaller = max(1.0, min(_area(left), _area(right)))
        if _intersection_area(left, right) / smaller >= 0.10:
            return True
    vertical_gap = max(0.0, max(left[1], right[1]) - min(left[3], right[3]))
    return vertical_gap <= y_tolerance and _horizontal_overlap_ratio(left, right) >= 0.25


def _overlaps_any(bbox: BBox, others: list[BBox], *, threshold: float) -> bool:
    bbox_area = max(_area(bbox), 1.0)
    return any(_intersection_area(bbox, other) / bbox_area >= threshold for other in others)


def _horizontal_overlap_ratio(left: BBox, right: BBox) -> float:
    overlap = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    min_width = max(1.0, min(left[2] - left[0], right[2] - right[0]))
    return overlap / min_width


def _area(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _intersection_area(left: BBox, right: BBox) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _union_bbox(boxes: Iterable[BBox]) -> BBox:
    box_list = list(boxes)
    return (
        min(box[0] for box in box_list),
        min(box[1] for box in box_list),
        max(box[2] for box in box_list),
        max(box[3] for box in box_list),
    )


def _normalize_bbox(bbox: Any) -> BBox:
    x0, y0, x1, y1 = [float(value) for value in bbox[:4]]
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
