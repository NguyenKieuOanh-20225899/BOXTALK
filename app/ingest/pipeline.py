from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Callable

import fitz

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
from app.ingest.extract.region_routed import (
    extract_with_region_routed_backend,
    is_region_routing_enabled,
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

    # Production table enhancement: hybrid TATR is part of the main ingest path
    # for table blocks, but it is conditional and falls back to the stable table
    # extraction result when unavailable.
    blocks = _enhance_table_blocks_with_hybrid_tatr(pdf_path, blocks)

    # Clean
    blocks = clean_blocks(blocks)

    # Attach cleaned block_ids back to pages
    page_to_block_ids: dict[int, list[str]] = {}
    for b in blocks:
        page_to_block_ids.setdefault(b.page_index, []).append(b.block_id)

    for page in pages:
        page.block_ids = page_to_block_ids.get(page.page_index, [])
        page_blocks = [b for b in blocks if b.page_index == page.page_index]
        page.text = "\n".join(b.text for b in page_blocks if b.text).strip()
        page.markdown = "\n\n".join(b.markdown for b in page_blocks if b.markdown).strip()
        page.has_table = any(b.block_type == "table" for b in page_blocks)
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
    region_backends = _region_extractors()
    prefer_layout_for_scan = _should_prefer_layout_for_scan(probe, layout_backends)

    if mode == "text":
        return [
            *region_backends,
            ("text", extract_with_text_backend),
            *layout_backends,
            ("ocr", extract_with_ocr_backend),
        ]

    if mode == "layout":
        return [
            *region_backends,
            *layout_backends,
            ("text", extract_with_text_backend),
            ("ocr", extract_with_ocr_backend),
        ]

    if mode == "ocr":
        if prefer_layout_for_scan:
            return [
                *layout_backends,
                *region_backends,
                ("ocr", extract_with_ocr_backend),
                ("text", extract_with_text_backend),
            ]
        return [
            *region_backends,
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
                *region_backends,
                ("text", extract_with_text_backend),
                ("ocr", extract_with_ocr_backend),
            ]
        return [
            *region_backends,
            ("text", extract_with_text_backend),
            *layout_backends,
            ("ocr", extract_with_ocr_backend),
        ]

    return [
        *region_backends,
        ("text", extract_with_text_backend),
        *layout_backends,
        ("ocr", extract_with_ocr_backend),
    ]


def _enhance_table_blocks_with_hybrid_tatr(
    pdf_path: Path,
    blocks: list[BlockNode],
) -> list[BlockNode]:
    table_candidates = [
        block
        for block in blocks
        if block.block_type == "table"
        and block.bbox is not None
        and not _should_keep_stable_table_without_hybrid(block)
        and (block.meta or {}).get("table_backend") != "hybrid_tatr"
        and not (block.meta or {}).get("hybrid_tatr_error")
        and not (block.meta or {}).get("hybrid_tatr_skipped_reason")
    ]
    if not table_candidates:
        return blocks

    try:
        from app.ingest.extract.hybrid_tatr_table import (
            extract_hybrid_tatr_table_region,
            is_hybrid_tatr_table_enabled,
        )
    except Exception:
        return blocks
    if not is_hybrid_tatr_table_enabled():
        return blocks

    enhanced_by_id: dict[str, BlockNode] = {}
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return blocks

    try:
        for block in table_candidates:
            if block.page_index < 0 or block.page_index >= len(doc):
                continue
            region_meta = {
                **dict(block.meta or {}),
                "pipeline_table_backend": "hybrid_tatr_auto",
                "fallback_table_backend": (block.meta or {}).get("table_backend") or (block.meta or {}).get("backend"),
                "fallback_source_mode": block.source_mode,
            }
            try:
                hybrid_block = extract_hybrid_tatr_table_region(
                    doc[block.page_index],
                    block.bbox,  # type: ignore[arg-type]
                    block_index=block.reading_order,
                    reading_order=block.reading_order,
                    region_meta=region_meta,
                )
            except Exception as exc:
                enhanced_by_id[block.block_id] = replace(
                    block,
                    meta={**dict(block.meta or {}), "hybrid_tatr_pipeline_error": str(exc)},
                )
                continue
            if hybrid_block is None:
                enhanced_by_id[block.block_id] = replace(
                    block,
                    meta={
                        **dict(block.meta or {}),
                        "hybrid_tatr_pipeline_skipped": "unavailable_or_no_result",
                    },
                )
                continue

            enhanced_by_id[block.block_id] = replace(
                hybrid_block,
                block_id=block.block_id,
                level=block.level,
                item_number=block.item_number,
                parent_block_id=block.parent_block_id,
                heading_path=list(block.heading_path),
                source_mode=block.source_mode,
                meta={
                    **dict(hybrid_block.meta or {}),
                    "pipeline_table_backend": "hybrid_tatr_auto",
                    "pipeline_enhanced_from": (block.meta or {}).get("table_backend")
                    or (block.meta or {}).get("backend")
                    or block.source_mode,
                },
            )
    finally:
        doc.close()

    if not enhanced_by_id:
        return blocks
    return [enhanced_by_id.get(block.block_id, block) for block in blocks]


def _should_keep_stable_table_without_hybrid(block: BlockNode) -> bool:
    if _hybrid_tatr_explicitly_forced():
        return False
    meta = block.meta or {}
    row_count = int(meta.get("table_row_count") or 0)
    col_count = int(meta.get("table_col_count") or 0)
    cell_count = int(meta.get("table_cell_count") or 0)
    table_backend = str(meta.get("table_backend") or meta.get("backend") or "")
    stable_backends = {
        "table_words_grid",
        "text_table",
        "text_region_table",
        "text_row_cluster",
        "table_clip_text",
    }
    return (
        table_backend in stable_backends
        and row_count >= 2
        and col_count >= 2
        and cell_count >= row_count * min(col_count, 2)
    )


def _hybrid_tatr_explicitly_forced() -> bool:
    backend = os.getenv("BOXBIIBOO_TABLE_BACKEND", "").strip().lower()
    if backend == "hybrid_tatr":
        return True
    explicit = os.getenv("BOXBIIBOO_ENABLE_HYBRID_TATR_TABLES")
    return explicit is not None and explicit.strip().lower() not in {"0", "false", "no", "off"}


def _region_extractors() -> list[tuple[str, ExtractorFn]]:
    if not is_region_routing_enabled():
        return []
    return [("region_routed", extract_with_region_routed_backend)]


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

    if backend_name in {"layout", "model_layout", "model_routed", "region_routed"}:
        # layout mà chỉ có 1 block/page thì thường là fallback kiểu "full doc"
        if len(blocks) <= len(pages):
            return False

        # nếu toàn paragraph, không có heading/list/table thì cũng nghi ngờ
        unique_types = {b.block_type for b in blocks}
        if unique_types == {"paragraph"} and len(blocks) < 5:
            return False

        with_bbox = sum(1 for b in blocks if b.bbox is not None)
        if backend_name in {"model_layout", "model_routed", "region_routed"} and with_bbox == 0:
            return False

        if backend_name in {"model_routed", "region_routed"}:
            route_backends = {b.meta.get("route_backend") for b in blocks if b.meta}
            if not route_backends:
                return False

    if backend_name == "region_routed":
        substantive_chars = sum(
            len((b.text or "").strip())
            for b in blocks
            if (b.meta or {}).get("route_backend") != "placeholder"
        )
        if substantive_chars < 50:
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
