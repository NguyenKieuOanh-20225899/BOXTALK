from __future__ import annotations

import os
from typing import Any

import fitz

from app.ingest.schemas import BlockNode


def is_hybrid_tatr_table_enabled() -> bool:
    backend = os.getenv("BOXBIIBOO_TABLE_BACKEND", "auto").strip().lower()
    if backend == "hybrid_tatr":
        return True
    if backend in {"default", "ocr_cluster", "none", "off", "0", "false", "no"}:
        return False
    explicit = os.getenv("BOXBIIBOO_ENABLE_HYBRID_TATR_TABLES")
    if explicit is not None:
        return _env_bool("BOXBIIBOO_ENABLE_HYBRID_TATR_TABLES", default=False)
    return _env_bool("BOXBIIBOO_ENABLE_PIPELINE_HYBRID_TATR_TABLES", default=True)


def extract_hybrid_tatr_table_region(
    page: fitz.Page,
    bbox: tuple[float, float, float, float],
    *,
    block_index: int,
    reading_order: int | None = None,
    region_meta: dict | None = None,
) -> BlockNode | None:
    """Extract one detected table region using TATR geometry + PDF word boxes.

    This module intentionally does not invoke PaddleOCR. On Windows we keep OCR
    and TATR in separate processes/envs to avoid Paddle/PyTorch CUDA DLL
    conflicts. If the PDF has no native word boxes, this extractor returns None
    by default so the stable table extractor can fall back.
    """

    rect = fitz.Rect(bbox)
    if rect.is_empty or rect.width < 2 or rect.height < 2:
        return None

    text_boxes = extract_pdf_word_boxes_for_region(page, bbox)
    text_source = "pdf_text_words" if text_boxes else "none"
    allow_geometry_only = _env_bool("BOXBIIBOO_HYBRID_TATR_ALLOW_GEOMETRY_ONLY", default=False)
    if not text_boxes and not allow_geometry_only:
        return None
    if not _can_attempt_tatr_load():
        return None

    scale = float(os.getenv("BOXBIIBOO_HYBRID_TATR_REGION_SCALE", "2.0"))
    image = _render_region_image(page, rect, scale=scale)
    if image is None:
        return None

    from app.ingest.extract.table import _rows_to_markdown, table_structure_from_rows
    from app.ingest.tatr_table_backend import (
        DEFAULT_TATR_DETECTION_MODEL,
        DEFAULT_TATR_STRUCTURE_MODEL,
        TatrObject,
        build_table_from_tatr_objects,
        recognize_table_structure,
        structure_debug_payload,
    )

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
        text_boxes=text_boxes,
        table_id=f"p{page.number:04d}_tatr_{block_index:04d}",
        page=page.number,
    )
    if table is None:
        return None

    rows = table.row_values
    if not rows:
        return None

    text = "\n".join(" | ".join(cell for cell in row if cell) for row in rows).strip()
    if not text and not allow_geometry_only:
        return None

    geometry_debug = structure_debug_payload(page_objects, table_bbox=bbox)
    structure = table_structure_from_rows(
        rows,
        backend="hybrid_tatr_region",
        cells=table.cells,
        row_bboxes=[row.bbox for row in table.rows],
        table_bbox=table.bbox,
    )
    meta = {
        **dict(region_meta or {}),
        **structure,
        "backend": "hybrid_tatr_region",
        "route_backend": "hybrid_tatr",
        "table_backend": "hybrid_tatr",
        "text_source": text_source,
        "text_source_missing_count": 0 if text_boxes else 1,
        "word_box_count": len(text_boxes),
        "source_model_name": {
            "detection": os.getenv("BOXBIIBOO_TATR_DETECTION_MODEL") or DEFAULT_TATR_DETECTION_MODEL,
            "structure": os.getenv("BOXBIIBOO_TATR_STRUCTURE_MODEL") or DEFAULT_TATR_STRUCTURE_MODEL,
        },
        "tatr_mode": "region_structure_recognition",
        "tatr_rows": geometry_debug["rows"],
        "tatr_columns": geometry_debug["columns"],
        "tatr_spanning_cells": geometry_debug["spanning_cells"],
        "coordinate_space": "pdf_page",
        "cell_bbox_mode": "content_bbox_with_grid_bbox",
    }

    return BlockNode(
        block_id=f"p{page.number:04d}_b{block_index:04d}",
        page_index=page.number,
        block_type="table",
        text=text,
        markdown=_rows_to_markdown(rows),
        reading_order=block_index if reading_order is None else reading_order,
        bbox=bbox,
        source_mode="layout",
        meta=meta,
    )


def extract_pdf_word_boxes_for_region(
    page: fitz.Page,
    bbox: tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    rect = fitz.Rect(bbox)
    if rect.is_empty:
        return []
    raw_words = page.get_text("words", clip=rect, sort=True) or []
    words: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_words):
        if len(raw) < 5:
            continue
        text = str(raw[4] or "").strip()
        if not text:
            continue
        x0, y0, x1, y1 = [float(value) for value in raw[:4]]
        if x1 <= x0 or y1 <= y0:
            continue
        words.append(
            {
                "text": text,
                "bbox": [x0, y0, x1, y1],
                "confidence": 1.0,
                "source": "pdf_text_words",
                "word_index": index,
            }
        )
    return words


def _render_region_image(page: fitz.Page, rect: fitz.Rect, *, scale: float) -> Any | None:
    if scale <= 0:
        scale = 1.0
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=rect, alpha=False)
    if pix.width < 2 or pix.height < 2:
        return None
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("hybrid_tatr table backend requires pillow") from exc
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _crop_to_page_bbox(
    bbox: tuple[float, float, float, float] | list[float],
    *,
    table_bbox: tuple[float, float, float, float],
    scale: float,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = [float(value) for value in bbox[:4]]
    if scale <= 0:
        scale = 1.0
    return (
        table_bbox[0] + x0 / scale,
        table_bbox[1] + y0 / scale,
        table_bbox[0] + x1 / scale,
        table_bbox[1] + y1 / scale,
    )


def _env_bool(name: str, *, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _can_attempt_tatr_load() -> bool:
    """Avoid accidental model downloads in automatic pipeline mode.

    Explicit requests (`BOXBIIBOO_TABLE_BACKEND=hybrid_tatr` or
    `BOXBIIBOO_ENABLE_HYBRID_TATR_TABLES=1`) may download/load models. The
    default pipeline auto mode only runs when the TATR model configs are already
    present in the Hugging Face cache, then falls back silently otherwise.
    """

    if os.getenv("BOXBIIBOO_TABLE_BACKEND", "").strip().lower() == "hybrid_tatr":
        return True
    explicit = os.getenv("BOXBIIBOO_ENABLE_HYBRID_TATR_TABLES")
    if explicit is not None:
        return _env_bool("BOXBIIBOO_ENABLE_HYBRID_TATR_TABLES", default=False)
    if _env_bool("BOXBIIBOO_HYBRID_TATR_AUTO_DOWNLOAD", default=False):
        return True
    return _tatr_models_look_cached()


def _tatr_models_look_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return False

    from app.ingest.tatr_table_backend import DEFAULT_TATR_DETECTION_MODEL, DEFAULT_TATR_STRUCTURE_MODEL

    detection_model = os.getenv("BOXBIIBOO_TATR_DETECTION_MODEL") or DEFAULT_TATR_DETECTION_MODEL
    structure_model = os.getenv("BOXBIIBOO_TATR_STRUCTURE_MODEL") or DEFAULT_TATR_STRUCTURE_MODEL
    return all(_tatr_model_files_look_cached(model_name, try_to_load_from_cache) for model_name in (detection_model, structure_model))


def _tatr_model_files_look_cached(model_name: str, try_to_load_from_cache: Any) -> bool:
    def cached(filename: str) -> bool:
        value = try_to_load_from_cache(model_name, filename)
        return isinstance(value, str) and bool(value)

    return (
        cached("config.json")
        and cached("preprocessor_config.json")
        and (cached("model.safetensors") or cached("pytorch_model.bin"))
    )
