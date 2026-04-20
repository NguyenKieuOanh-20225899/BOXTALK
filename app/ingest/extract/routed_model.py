from __future__ import annotations

import os
from pathlib import Path

import fitz

from app.ingest.extract.model_layout import detect_model_layout_regions_for_page
from app.ingest.extract.ocr import extract_ocr_region
from app.ingest.extract.table import extract_table_region
from app.ingest.extract.text import extract_text_region
from app.ingest.schemas import BlockNode, PageNode


def is_model_routing_enabled() -> bool:
    return os.getenv("BOXBIIBOO_ENABLE_MODEL_ROUTING", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def extract_with_model_routed_backend(
    pdf_path: str | Path,
) -> tuple[list[PageNode], list[BlockNode]]:
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))

    pages: list[PageNode] = []
    all_blocks: list[BlockNode] = []

    for page in doc:
        regions = detect_model_layout_regions_for_page(page)
        page_blocks: list[BlockNode] = []
        route_counts: dict[str, int] = {}

        for reading_order, region in enumerate(regions):
            route_backend = _route_region(region)

            region_meta = {
                "route_backend": route_backend,
                "model_label": region.get("label_name"),
                "model_score": region.get("score"),
                "region_id": region.get("region_id"),
            }

            block = _extract_region_block(
                page=page,
                region=region,
                route_backend=route_backend,
                block_index=len(page_blocks),
                reading_order=reading_order,
                region_meta=region_meta,
            )
            if block is None:
                continue

            page_blocks.append(block)
            effective_route = str((block.meta or {}).get("route_backend") or route_backend)
            route_counts[effective_route] = route_counts.get(effective_route, 0) + 1

        pages.append(
            PageNode(
                page_index=page.number,
                page_label=str(page.number + 1),
                text="\n".join(block.text for block in page_blocks if block.text).strip(),
                markdown="\n\n".join(
                    block.markdown for block in page_blocks if block.markdown
                ).strip(),
                source_mode="mixed",
                has_ocr=any(block.source_mode == "ocr" for block in page_blocks),
                has_table=any(block.block_type == "table" for block in page_blocks),
                meta={
                    "backend": "model_routed",
                    "region_count": len(regions),
                    "block_count": len(page_blocks),
                    "route_counts": route_counts,
                },
            )
        )
        all_blocks.extend(page_blocks)

    doc.close()

    if not all_blocks:
        raise RuntimeError("Model-routed backend produced no blocks")

    return pages, all_blocks


def _route_region(region: dict) -> str:
    block_type = str(region.get("block_type") or "").strip().lower()
    direct_text = str(region.get("direct_text") or "").strip()

    if block_type == "table":
        return "table"

    if block_type == "figure":
        return "text" if direct_text else "placeholder"

    if block_type in {"caption", "metadata"}:
        return "text" if direct_text else "ocr"

    if block_type in {"heading", "list_item", "paragraph"}:
        return "text" if direct_text else "ocr"

    return "text" if direct_text else "ocr"


def _extract_region_block(
    *,
    page: fitz.Page,
    region: dict,
    route_backend: str,
    block_index: int,
    reading_order: int,
    region_meta: dict,
) -> BlockNode | None:
    bbox = tuple(region["bbox"])
    block_type = str(region.get("block_type") or "paragraph")

    if route_backend == "table":
        return extract_table_region(
            page,
            bbox,
            block_index=block_index,
            reading_order=reading_order,
            region_meta=region_meta,
        )

    if route_backend == "ocr":
        block = extract_ocr_region(
            page,
            bbox,
            block_index=block_index,
            reading_order=reading_order,
            block_type_hint=block_type,
            region_meta=region_meta,
        )
        if block is not None:
            return block

    if route_backend == "placeholder":
        return BlockNode(
            block_id=f"p{page.number:04d}_b{block_index:04d}",
            page_index=page.number,
            block_type=block_type,
            text="Figure",
            markdown="Figure",
            reading_order=reading_order,
            bbox=bbox,
            source_mode="layout",
            meta={**region_meta, "backend": "model_placeholder"},
        )

    region_dict = {
        "bbox": bbox,
        "page_index": page.number,
        "text": str(region.get("direct_text") or "").strip(),
        "block_type": block_type,
        "meta": region_meta,
    }
    block = extract_text_region(
        region_dict,
        page.number,
        block_index,
        reading_order=reading_order,
        block_type_hint=block_type,
        region_meta=region_meta,
    )
    if block is not None:
        return block

    if route_backend != "ocr":
        return extract_ocr_region(
            page,
            bbox,
            block_index=block_index,
            reading_order=reading_order,
            block_type_hint=block_type,
            region_meta={**region_meta, "route_backend": "ocr_fallback"},
        )

    return None
