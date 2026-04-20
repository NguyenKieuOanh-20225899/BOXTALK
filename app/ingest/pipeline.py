from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

from app.ingest.chunker import build_chunks
from app.ingest.cleaners import clean_blocks
from app.ingest.normalize import normalize_pages_blocks
from app.ingest.probe import probe_pdf
from app.ingest.schemas import BlockNode, PageNode
from app.ingest.structure import enrich_structure
from app.ingest.extract.model_layout import (
    extract_with_model_layout_backend,
    is_model_layout_enabled,
)
from app.ingest.extract.routed_model import (
    extract_with_model_routed_backend,
    is_model_routing_enabled,
)
from app.ingest.extract.text import extract_with_text_backend
from app.ingest.extract.layout import extract_with_layout_backend
from app.ingest.extract.ocr import extract_with_ocr_backend


ExtractorFn = Callable[[str | Path], tuple[list[PageNode], list[BlockNode]]]


def ingest_pdf(pdf_path: str | Path) -> dict:
    pdf_path = Path(pdf_path)
    probe = probe_pdf(pdf_path)

    mode = probe.probe_detected_mode
    extractor_plan = _build_extractor_plan(probe)

    pages: list[PageNode] = []
    blocks: list[BlockNode] = []
    used_backend = None
    errors: list[str] = []

    for backend_name, extractor in extractor_plan:
        try:
            pages, blocks = extractor(pdf_path)

            if not _looks_valid_result(pages, blocks, backend_name=backend_name):
                raise RuntimeError(f"{backend_name} returned weak result")

            used_backend = backend_name
            break

        except Exception as e:
            errors.append(f"{backend_name} failed: {e}")

    if not pages or not blocks:
        raise RuntimeError(
            f"All ingest backends failed for {pdf_path}. "
            + " | ".join(errors)
        )

    # Normalize
    pages, blocks = normalize_pages_blocks(pages, blocks)

    # Clean
    blocks = clean_blocks(blocks)

    # Attach cleaned block_ids back to pages
    page_to_block_ids: dict[int, list[str]] = {}
    for b in blocks:
        page_to_block_ids.setdefault(b.page_index, []).append(b.block_id)

    for page in pages:
        page.block_ids = page_to_block_ids.get(page.page_index, [])
        page.meta["used_backend"] = used_backend
        page.meta["probe_mode"] = mode
        if errors:
            page.meta["fallback_errors"] = errors

    # Structure
    blocks = enrich_structure(blocks)

    # Chunk
    chunks = build_chunks(blocks)

    return {
        "probe": probe.to_dict(),
        "pages": pages,
        "blocks": blocks,
        "chunks": chunks,
        "used_backend": used_backend,
        "errors": errors,
    }


def _build_extractor_plan(probe) -> list[tuple[str, ExtractorFn]]:
    mode = probe.probe_detected_mode
    layout_backends = _layout_extractors(probe)
    prefer_layout_for_scan = _should_prefer_layout_for_scan(probe, layout_backends)

    if mode == "text":
        return [
            ("text", extract_with_text_backend),
            *layout_backends,
            ("ocr", extract_with_ocr_backend),
        ]

    if mode == "layout":
        return [
            *layout_backends,
            ("text", extract_with_text_backend),
            ("ocr", extract_with_ocr_backend),
        ]

    if mode == "ocr":
        if prefer_layout_for_scan:
            return [
                *layout_backends,
                ("ocr", extract_with_ocr_backend),
                ("text", extract_with_text_backend),
            ]
        return [
            ("ocr", extract_with_ocr_backend),
            ("text", extract_with_text_backend),
            *layout_backends,
        ]

    if mode == "mixed":
        # mixed production-safe:
        # text trước vì nhiều file mixed vẫn có text layer usable
        if prefer_layout_for_scan:
            return [
                *layout_backends,
                ("text", extract_with_text_backend),
                ("ocr", extract_with_ocr_backend),
            ]
        return [
            ("text", extract_with_text_backend),
            *layout_backends,
            ("ocr", extract_with_ocr_backend),
        ]

    return [
        ("text", extract_with_text_backend),
        *layout_backends,
        ("ocr", extract_with_ocr_backend),
    ]


def _layout_extractors(probe) -> list[tuple[str, ExtractorFn]]:
    backends: list[tuple[str, ExtractorFn]] = []
    if is_model_layout_enabled():
        if is_model_routing_enabled():
            backends.append(("model_routed", extract_with_model_routed_backend))
        backends.append(("model_layout", extract_with_model_layout_backend))
    if not _should_skip_docling_layout(probe):
        backends.append(("layout", extract_with_layout_backend))
    return backends


def _should_skip_docling_layout(probe) -> bool:
    override = os.getenv("BOXBIIBOO_ENABLE_DOCLING_ON_SCANS", "0").strip().lower()
    if override in {"1", "true", "yes"}:
        return False

    if probe.probe_detected_mode != "ocr":
        return False

    # Scan-heavy / image-only PDFs are where Docling has been the slowest and
    # least reliable in local benchmarks. Keep model-based layout enabled if
    # configured, but skip the Docling markdown backend unless explicitly forced.
    return (
        probe.likely_scanned_ratio >= 0.8
        and probe.text_layer_ratio < 0.25
        and probe.avg_images_per_page >= 1.0
    )


def _should_prefer_layout_for_scan(probe, layout_backends: list[tuple[str, ExtractorFn]]) -> bool:
    if not layout_backends:
        return False

    has_model_routed = any(name == "model_routed" for name, _ in layout_backends)
    if not has_model_routed:
        return False

    return (
        probe.avg_images_per_page >= 1.0
        and probe.text_layer_ratio < 0.25
        and probe.image_heavy_ratio >= 0.5
    )


def _looks_valid_result(
    pages: list[PageNode],
    blocks: list[BlockNode],
    *,
    backend_name: str,
) -> bool:
    if not pages or not blocks:
        return False

    if backend_name in {"layout", "model_layout", "model_routed"}:
        # layout mà chỉ có 1 block/page thì thường là fallback kiểu "full doc"
        if len(blocks) <= len(pages):
            return False

        # nếu toàn paragraph, không có heading/list/table thì cũng nghi ngờ
        unique_types = {b.block_type for b in blocks}
        if unique_types == {"paragraph"} and len(blocks) < 5:
            return False

        with_bbox = sum(1 for b in blocks if b.bbox is not None)
        if backend_name in {"model_layout", "model_routed"} and with_bbox == 0:
            return False

        if backend_name == "model_routed":
            route_backends = {b.meta.get("route_backend") for b in blocks if b.meta}
            if not route_backends:
                return False

    if backend_name == "text":
        total_chars = sum(len(b.text or "") for b in blocks)
        if total_chars < 50:
            return False

    if backend_name == "ocr":
        non_empty = sum(1 for b in blocks if (b.text or "").strip())
        if non_empty == 0:
            return False

    return True
