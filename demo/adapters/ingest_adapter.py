from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import asdict
import io
from pathlib import Path
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Iterator

from demo.bootstrap import ensure_repo_on_path

ensure_repo_on_path()

import fitz

from app.ingest.extract.text import extract_with_text_backend
from app.ingest.extract.region_routed import _extract_region
from app.ingest.probe import probe_pdf
from app.ingest.region.detector import detect_regions
from app.ingest.region.router import route_region
from app.ingest.schemas import BlockNode


class DemoError(RuntimeError):
    """Base class for expected demo errors."""


class DemoInputError(DemoError):
    """Raised for invalid PDF/page/output inputs."""


class DemoUnsupportedOption(DemoError):
    """Raised when the existing pipeline has no matching option."""


SUPPORTED_OCR_MODES = {"auto", "always", "never"}
SUPPORTED_TABLE_EXTRACTORS = {"configured", "default", "tatr", "hybrid_tatr"}


def run_ingest_page(
    *,
    pdf_path: Path,
    page_number: int,
    ocr_mode: str,
    table_extractor: str,
    show_library_logs: bool = False,
) -> dict[str, Any]:
    """Run the existing PDF ingest components on one page.

    This adapter intentionally delegates the actual work to existing modules:
    ``probe_pdf``, ``detect_regions`` and the region-routed extractor.
    """

    pdf_path = Path(pdf_path)
    _validate_options(ocr_mode=ocr_mode, table_extractor=table_extractor)
    if not pdf_path.exists():
        raise DemoInputError(f"PDF khong ton tai: {pdf_path}")
    if not pdf_path.is_file():
        raise DemoInputError(f"Duong dan PDF khong phai tep: {pdf_path}")
    if page_number < 1:
        raise DemoInputError("--page phai la so nguyen bat dau tu 1")

    env_updates = {
        "BOXBIIBOO_ENABLE_REGION_IMAGE_OCR": _ocr_env_value(ocr_mode),
        **_table_env_values(table_extractor),
    }

    started = time.perf_counter()
    warnings: list[str] = []

    with _patched_env(env_updates):
        probe_started = time.perf_counter()
        try:
            with _maybe_suppress_library_output(show_library_logs):
                probe = probe_pdf(pdf_path)
        except Exception as exc:
            raise DemoInputError(f"Khong tham do duoc PDF: {exc}") from exc
        probe_ms = _elapsed_ms(probe_started)

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:
            raise DemoInputError(f"Khong mo duoc PDF: {exc}") from exc

        try:
            if getattr(doc, "needs_pass", False):
                raise DemoInputError("PDF co mat khau; demo khong tu mo khoa tep.")
            if page_number > len(doc):
                raise DemoInputError(
                    f"Trang {page_number} nam ngoai pham vi; tep co {len(doc)} trang."
                )

            page = doc[page_number - 1]
            page_text = page.get_text("text", sort=True).strip()
            page_info = {
                "number": page_number,
                "page_index": page_number - 1,
                "width": round(float(page.rect.width), 3),
                "height": round(float(page.rect.height), 3),
                "has_text_layer": bool(page_text),
                "embedded_image_count": len(page.get_images(full=True) or []),
            }

            layout_started = time.perf_counter()
            try:
                with _maybe_suppress_library_output(show_library_logs):
                    regions = detect_regions(page)
            except Exception as exc:
                raise DemoError(f"Bo phat hien vung bi loi: {exc}") from exc
            layout_ms = _elapsed_ms(layout_started)

            extraction_started = time.perf_counter()
            blocks: list[BlockNode] = []
            route_plan: list[dict[str, Any]] = []
            for reading_order, region in enumerate(regions):
                kind = str(region.get("kind") or region.get("type") or "unknown")
                planned_route = str(
                    "tatr"
                    if table_extractor == "tatr" and kind == "table"
                    else region.get("route_backend")
                    or route_region(kind, probe.probe_detected_mode)
                )
                block: BlockNode | None = None
                error: str | None = None
                try:
                    with _maybe_suppress_library_output(show_library_logs):
                        if table_extractor == "tatr" and kind == "table":
                            block = _extract_tatr_only_table_region(
                                page=page,
                                region=region,
                                block_index=len(blocks),
                                reading_order=reading_order,
                            )
                        else:
                            block = _extract_region(
                                page=page,
                                region=region,
                                block_index=len(blocks),
                                reading_order=reading_order,
                            )
                except Exception as exc:
                    error = str(exc)
                    warnings.append(
                        f"Vung {region.get('region_id') or reading_order}: khong trich xuat duoc ({exc})"
                    )

                if block is not None:
                    blocks.append(block)

                route_plan.append(
                    _route_record(
                        region=region,
                        block=block,
                        reading_order=reading_order,
                        planned_route=planned_route,
                        error=error,
                    )
                )
            extraction_ms = _elapsed_ms(extraction_started)
        finally:
            doc.close()

    total_ms = _elapsed_ms(started)
    serialized_blocks = [_serialize_block(block) for block in blocks]
    serialized_regions = _serialize_regions(regions, route_plan)
    summary = _build_summary(serialized_regions, serialized_blocks)

    return {
        "document": {
            "name": pdf_path.name,
            "path": str(pdf_path),
            "page_count": probe.page_count,
            "probe_mode": probe.probe_detected_mode,
            "probe": probe.to_dict(),
        },
        "page": page_info,
        "summary": summary,
        "regions": serialized_regions,
        "route_plan": route_plan,
        "blocks": serialized_blocks,
        "timing": {
            "probe_time_ms": probe_ms,
            "layout_time_ms": layout_ms,
            "extraction_time_ms": extraction_ms,
            "total_time_ms": total_ms,
        },
        "config": {
            "ocr_mode": ocr_mode,
            "table_extractor": table_extractor,
        },
        "warnings": warnings,
    }


def run_ingest_page_region_off(
    *,
    pdf_path: Path,
    page_number: int,
    show_library_logs: bool = False,
) -> dict[str, Any]:
    """Run one-page ingest without region detection/routing.

    This is the controlled baseline for the demo: it uses the existing full-page
    text backend and keeps the output schema compatible with ``write_outputs``.
    """

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise DemoInputError(f"PDF khong ton tai: {pdf_path}")
    if not pdf_path.is_file():
        raise DemoInputError(f"Duong dan PDF khong phai tep: {pdf_path}")
    if page_number < 1:
        raise DemoInputError("--page phai la so nguyen bat dau tu 1")

    started = time.perf_counter()
    with _maybe_suppress_library_output(show_library_logs):
        probe = probe_pdf(pdf_path)
        pages, blocks = extract_with_text_backend(pdf_path)

    page_index = page_number - 1
    if page_number > len(pages):
        raise DemoInputError(f"Trang {page_number} nam ngoai pham vi; tep co {len(pages)} trang.")

    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_index]
        page_text = page.get_text("text", sort=True).strip()
        page_info = {
            "number": page_number,
            "page_index": page_index,
            "width": round(float(page.rect.width), 3),
            "height": round(float(page.rect.height), 3),
            "has_text_layer": bool(page_text),
            "embedded_image_count": len(page.get_images(full=True) or []),
        }
    finally:
        doc.close()

    page_blocks = [block for block in blocks if block.page_index == page_index]
    regions = [_plain_text_region(block) for block in page_blocks]
    route_plan = [_plain_text_route(block) for block in page_blocks]
    serialized_blocks = [_serialize_block(block) for block in page_blocks]
    serialized_regions = _serialize_regions(regions, route_plan)

    return {
        "document": {
            "name": pdf_path.name,
            "path": str(pdf_path),
            "page_count": probe.page_count,
            "probe_mode": probe.probe_detected_mode,
            "probe": probe.to_dict(),
        },
        "page": page_info,
        "summary": _build_summary(serialized_regions, serialized_blocks),
        "regions": serialized_regions,
        "route_plan": route_plan,
        "blocks": serialized_blocks,
        "timing": {
            "probe_time_ms": 0.0,
            "layout_time_ms": 0.0,
            "extraction_time_ms": _elapsed_ms(started),
            "total_time_ms": _elapsed_ms(started),
        },
        "config": {
            "region_routing": "off",
            "ocr_mode": "never",
            "table_extractor": "disabled",
        },
        "warnings": [
            "Region routing da tat: khong chay detect_regions, OCR theo vung hoac trich xuat bang theo vung."
        ],
    }


def _validate_options(*, ocr_mode: str, table_extractor: str) -> None:
    if ocr_mode not in SUPPORTED_OCR_MODES:
        raise DemoUnsupportedOption(
            f"ocr-mode={ocr_mode!r} khong hop le; chon auto, always hoac never."
        )
    if table_extractor not in SUPPORTED_TABLE_EXTRACTORS:
        raise DemoUnsupportedOption(
            "table-extractor khong hop le; chon configured, default, tatr hoac hybrid_tatr."
        )


def _ocr_env_value(ocr_mode: str) -> str:
    if ocr_mode == "never":
        return "off"
    if ocr_mode == "always":
        return "always"
    return "auto"


def _table_env_values(table_extractor: str) -> dict[str, str]:
    if table_extractor == "configured":
        return {}
    if table_extractor == "tatr":
        # TATR-only is handled directly inside the demo adapter for table
        # regions. Keep the normal table extractor in default mode for any
        # fallback path so it does not accidentally enable Hybrid TATR.
        return {"BOXBIIBOO_TABLE_BACKEND": "default"}
    return {"BOXBIIBOO_TABLE_BACKEND": table_extractor}


def _extract_tatr_only_table_region(
    *,
    page: fitz.Page,
    region: dict[str, Any],
    block_index: int,
    reading_order: int,
) -> BlockNode | None:
    bbox = tuple(float(value) for value in region["bbox"][:4])
    rect = fitz.Rect(bbox)
    if rect.is_empty or rect.width < 2 or rect.height < 2:
        return None

    from app.ingest.extract.hybrid_tatr_table import _crop_to_page_bbox, _render_region_image
    from app.ingest.tatr_table_backend import (
        DEFAULT_TATR_DETECTION_MODEL,
        DEFAULT_TATR_STRUCTURE_MODEL,
        TatrObject,
        build_table_from_tatr_objects,
        recognize_table_structure,
        structure_debug_payload,
    )

    scale = float(os.getenv("BOXBIIBOO_HYBRID_TATR_REGION_SCALE", "2.0"))
    image = _render_region_image(page, rect, scale=scale)
    if image is None:
        return None

    crop_objects = recognize_table_structure(
        image,
        table_offset=(0.0, 0.0),
        device=os.getenv("BOXBIIBOO_TATR_DEVICE"),
    )
    page_objects = [
        TatrObject(
            obj.label,
            _crop_to_page_bbox(obj.bbox, table_bbox=bbox, scale=scale),
            obj.score,
        )
        for obj in crop_objects
    ]

    table = build_table_from_tatr_objects(
        page_objects,
        table_bbox=bbox,
        text_boxes=[],
        table_id=f"p{page.number:04d}_tatr_only_{block_index:04d}",
        page=page.number,
    )
    if table is None:
        return None

    cells = [cell.to_meta() for cell in table.cells]
    rows = table.row_values
    row_count = len(rows)
    col_count = table.col_count
    geometry_debug = structure_debug_payload(page_objects, table_bbox=bbox)
    markdown = _tatr_geometry_markdown(row_count=row_count, col_count=col_count, cell_count=len(cells))
    text = (
        f"TATR-only geometry: {row_count} rows, {col_count} columns, "
        f"{len(cells)} cells. No text assignment."
    )

    region_meta = {
        "backend": "tatr_region_geometry",
        "route_backend": "tatr",
        "route_reason": "detected_table_region_tatr_only",
        "region_id": region.get("region_id"),
        "region_type": region.get("kind") or region.get("type"),
        "region_kind": region.get("kind") or region.get("type"),
        "region_bbox": bbox,
        "page_number": page.number + 1,
        "confidence": region.get("confidence"),
        "source": region.get("detection_source"),
        "detection_source": region.get("detection_source"),
        "fallback_used": False,
        "table_backend": "tatr",
        "table_row_count": row_count,
        "table_col_count": col_count,
        "table_body_row_count": max(0, row_count - 1),
        "table_rows": rows,
        "table_records": [],
        "table_cells": cells,
        "table_cell_count": len(cells),
        "table_markdown": markdown,
        "table_csv": "",
        "table_html": "",
        "table_bbox": bbox,
        "table_column_bounds": [],
        "table_row_bboxes": [row.bbox for row in table.rows],
        "text_source": "none",
        "text_source_missing_count": 1,
        "source_model_name": {
            "detection": os.getenv("BOXBIIBOO_TATR_DETECTION_MODEL") or DEFAULT_TATR_DETECTION_MODEL,
            "structure": os.getenv("BOXBIIBOO_TATR_STRUCTURE_MODEL") or DEFAULT_TATR_STRUCTURE_MODEL,
        },
        "tatr_mode": "region_structure_recognition_geometry_only",
        "tatr_rows": geometry_debug["rows"],
        "tatr_columns": geometry_debug["columns"],
        "tatr_spanning_cells": geometry_debug["spanning_cells"],
        "coordinate_space": "pdf_page",
        "citation_metadata": {
            "block_type": "table",
            "citation_target": "table",
            "table_bbox": bbox,
        },
        "extraction_trace": {
            "backend": "tatr",
            "row_count": row_count,
            "col_count": col_count,
            "cell_count": len(cells),
            "has_cell_geometry": True,
            "has_text_assignment": False,
        },
    }

    return BlockNode(
        block_id=f"p{page.number:04d}_b{block_index:04d}",
        page_index=page.number,
        block_type="table",
        text=text,
        markdown=markdown,
        reading_order=reading_order,
        bbox=bbox,
        source_mode="layout",
        meta=region_meta,
    )


def _tatr_geometry_markdown(*, row_count: int, col_count: int, cell_count: int) -> str:
    return "\n".join(
        [
            "# TATR-only geometry",
            "",
            "| Thuoc tinh | Gia tri |",
            "| --- | ---: |",
            f"| Rows | {row_count} |",
            f"| Columns | {col_count} |",
            f"| Cells | {cell_count} |",
            "",
            "TATR-only chi du doan cau truc hinh hoc cua bang; khong gan van ban vao o.",
        ]
    )


@contextmanager
def _patched_env(values: dict[str, str]) -> Iterator[None]:
    old_values: dict[str, str | None] = {}
    for key, value in values.items():
        old_values[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


@contextmanager
def _maybe_suppress_library_output(show_library_logs: bool) -> Iterator[None]:
    if show_library_logs:
        yield
        return
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        yield


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def _route_record(
    *,
    region: dict[str, Any],
    block: BlockNode | None,
    reading_order: int,
    planned_route: str,
    error: str | None,
) -> dict[str, Any]:
    block_meta = dict(block.meta or {}) if block is not None else {}
    actual_route = str(block_meta.get("route_backend") or (block.source_mode if block else planned_route))
    ocr_applied = bool(block is not None and block.source_mode == "ocr")
    reason = str(block_meta.get("route_reason") or block_meta.get("ocr_skipped_reason") or "")
    if not reason and str(region.get("kind")) == "image" and not ocr_applied:
        reason = "khong tao duoc noi dung OCR; giu metadata vung anh"
    return {
        "region_id": region.get("region_id"),
        "block_id": block.block_id if block is not None else None,
        "type": _region_bucket(str(region.get("kind") or region.get("type") or "unknown")),
        "original_type": str(region.get("kind") or region.get("type") or "unknown"),
        "bbox": _bbox(region.get("bbox")),
        "confidence": region.get("confidence"),
        "reading_order": reading_order + 1,
        "planned_route": planned_route,
        "actual_route": actual_route,
        "ocr_applied": ocr_applied,
        "reason": reason,
        "error": error,
    }


def _serialize_regions(regions: list[dict[str, Any]], route_plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    route_by_id = {item.get("region_id"): item for item in route_plan if item.get("region_id")}
    serialized: list[dict[str, Any]] = []
    for index, region in enumerate(regions):
        route = route_by_id.get(region.get("region_id")) or route_plan[index]
        serialized.append(
            {
                "region_id": region.get("region_id"),
                "block_id": route.get("block_id"),
                "type": route.get("type"),
                "original_type": route.get("original_type"),
                "bbox": route.get("bbox"),
                "confidence": route.get("confidence"),
                "reading_order": route.get("reading_order"),
                "planned_route": route.get("planned_route"),
                "actual_route": route.get("actual_route"),
                "detection_source": region.get("detection_source"),
                "text_preview": _preview(str(region.get("text") or ""), 160),
            }
        )
    return serialized


def _serialize_block(block: BlockNode) -> dict[str, Any]:
    meta = dict(block.meta or {})
    route = str(meta.get("route_backend") or block.source_mode)
    table_cells = meta.get("table_cells") if isinstance(meta.get("table_cells"), list) else []
    row_count, col_count = _table_shape(table_cells)
    payload: dict[str, Any] = {
        "block_id": block.block_id,
        "type": _block_type(block.block_type),
        "original_type": meta.get("region_type") or block.block_type,
        "content": block.text,
        "markdown": block.markdown,
        "bbox": _bbox(block.bbox),
        "confidence": meta.get("confidence"),
        "reading_order": int(block.reading_order) + 1,
        "source_mode": block.source_mode,
        "heading_path": list(block.heading_path),
        "extraction": {
            "route": route,
            "source": meta.get("detection_source") or meta.get("source") or block.source_mode,
            "ocr_applied": block.source_mode == "ocr",
            "ocr_reason": meta.get("ocr_reason") or meta.get("ocr_skipped_reason") or meta.get("ocr_error"),
        },
        "metadata": meta,
    }
    if _block_type(block.block_type) == "table":
        payload["table"] = {
            "backend": meta.get("table_backend") or meta.get("backend") or route,
            "row_count": row_count,
            "column_count": col_count,
            "cell_count": len(table_cells),
            "markdown": meta.get("table_markdown") or block.markdown,
            "csv": meta.get("table_csv"),
        }
    return payload


def _plain_text_region(block: BlockNode) -> dict[str, Any]:
    return {
        "region_id": block.block_id,
        "kind": block.block_type,
        "type": block.block_type,
        "bbox": block.bbox,
        "confidence": None,
        "detection_source": "text_backend_no_region_routing",
        "text": block.text,
    }


def _plain_text_route(block: BlockNode) -> dict[str, Any]:
    kind = _region_bucket(block.block_type)
    return {
        "region_id": block.block_id,
        "block_id": block.block_id,
        "type": kind,
        "original_type": block.block_type,
        "bbox": _bbox(block.bbox),
        "confidence": None,
        "reading_order": int(block.reading_order) + 1,
        "planned_route": "region_off_text_layer",
        "actual_route": "region_off_text_layer",
        "ocr_applied": False,
        "reason": "region routing off; full-page text backend",
        "error": None,
    }


def _block_type(value: str) -> str:
    normalized = str(value or "unknown").strip().lower()
    if normalized in {"paragraph", "heading", "caption", "metadata", "header", "footer", "list_item"}:
        return "text"
    if normalized in {"figure", "image"}:
        return "image"
    return normalized or "unknown"


def _region_bucket(kind: str) -> str:
    normalized = str(kind or "unknown").strip().lower()
    if normalized == "heading":
        return "title"
    if normalized == "list_item":
        return "list"
    if normalized in {"paragraph", "text", "caption", "metadata", "header", "footer"}:
        return "text"
    if normalized in {"image", "table"}:
        return normalized
    return "unknown"


def _bbox(value: Any) -> list[float] | None:
    if not value or len(value) < 4:
        return None
    return [round(float(v), 3) for v in value[:4]]


def _preview(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)] + "..."


def _table_shape(cells: list[Any]) -> tuple[int, int]:
    rows = set()
    cols = set()
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        row = cell.get("row", cell.get("row_index"))
        col = cell.get("col", cell.get("col_index"))
        if row is not None:
            rows.add(int(row))
        if col is not None:
            cols.add(int(col))
    return len(rows), len(cols)


def _build_summary(regions: list[dict[str, Any]], blocks: list[dict[str, Any]]) -> dict[str, Any]:
    region_counts = {key: 0 for key in ("text", "title", "list", "image", "table", "unknown")}
    for region in regions:
        key = str(region.get("type") or "unknown")
        region_counts[key if key in region_counts else "unknown"] += 1

    block_counts = {"text_blocks": 0, "image_blocks": 0, "table_blocks": 0}
    for block in blocks:
        kind = str(block.get("type") or "unknown")
        if kind == "table":
            block_counts["table_blocks"] += 1
        elif kind == "image":
            block_counts["image_blocks"] += 1
        else:
            block_counts["text_blocks"] += 1

    return {
        **block_counts,
        "total_blocks": len(blocks),
        "region_counts": region_counts,
        "total_regions": len(regions),
    }


def block_to_dict(block: BlockNode) -> dict[str, Any]:
    """Expose the existing dataclass as a plain dict for tests/debugging."""

    return asdict(block)
