from __future__ import annotations

import os
from pathlib import Path

import fitz

from app.ingest.extract.ocr import extract_ocr_region
from app.ingest.extract.table import extract_table_region
from app.ingest.extract.text import extract_text_region
from app.ingest.region.detector import detect_regions
from app.ingest.schemas import BlockNode, PageNode


def is_region_routing_enabled() -> bool:
    return os.getenv("BOXBIIBOO_ENABLE_REGION_ROUTING", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def extract_with_region_routed_backend(
    pdf_path: str | Path,
) -> tuple[list[PageNode], list[BlockNode]]:
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))

    pages: list[PageNode] = []
    all_blocks: list[BlockNode] = []

    for page in doc:
        regions = _detect_page_regions(page)
        page_blocks: list[BlockNode] = []
        route_counts: dict[str, int] = {}

        for reading_order, region in enumerate(regions):
            block = _extract_region(
                page=page,
                region=region,
                block_index=len(page_blocks),
                reading_order=reading_order,
            )
            if block is None:
                continue

            page_blocks.append(block)
            route_backend = str((block.meta or {}).get("route_backend") or block.source_mode)
            route_counts[route_backend] = route_counts.get(route_backend, 0) + 1

        pages.append(
            PageNode(
                page_index=page.number,
                page_label=str(page.number + 1),
                text="\n".join(block.text for block in page_blocks if block.text).strip(),
                markdown="\n\n".join(block.markdown for block in page_blocks if block.markdown).strip(),
                source_mode="mixed",
                has_ocr=any(block.source_mode == "ocr" for block in page_blocks),
                has_table=any(block.block_type == "table" for block in page_blocks),
                meta={
                    "backend": "region_routed",
                    "region_count": len(regions),
                    "block_count": len(page_blocks),
                    "route_counts": route_counts,
                },
            )
        )
        all_blocks.extend(page_blocks)

    doc.close()

    if not all_blocks:
        raise RuntimeError("Region-routed backend produced no blocks")

    return pages, all_blocks


def _detect_page_regions(page: fitz.Page) -> list[dict]:
    return detect_regions(page)


def _extract_region(
    *,
    page: fitz.Page,
    region: dict,
    block_index: int,
    reading_order: int,
) -> BlockNode | None:
    kind = str(region.get("kind") or "")
    bbox = tuple(region["bbox"])
    region_meta = {
        "backend": "region_routed",
        "region_id": region.get("region_id"),
        "region_kind": kind,
        "detection_source": region.get("detection_source"),
    }

    if kind == "table":
        block = extract_table_region(
            page,
            bbox,
            block_index=block_index,
            reading_order=reading_order,
            region_meta={
                **region_meta,
                "route_backend": "table",
                "route_reason": "detected_table_region",
            },
        )
        if block is not None:
            return block

        region_meta["table_route_fallback"] = "text_region"

    if kind in {"text", "paragraph", "heading", "list_item", "caption", "metadata", "header", "footer", "table"}:
        block_type_hint = str(region.get("block_type") or kind).strip().lower()
        if block_type_hint in {"header", "footer"}:
            block_type_hint = "metadata"
        region_dict = {
            "bbox": bbox,
            "page_index": page.number,
            "text": str(region.get("text") or "").strip(),
            "block_type": block_type_hint,
            "meta": {
                **region_meta,
                "route_backend": "text",
                "route_reason": "detected_text_region" if kind != "table" else "table_fallback_to_text",
            },
        }
        return extract_text_region(
            region_dict,
            page.number,
            block_index,
            reading_order=reading_order,
            block_type_hint=block_type_hint,
            region_meta=region_dict["meta"],
        )

    if kind == "image":
        ocr_block = _try_extract_image_ocr(
            page=page,
            bbox=bbox,
            block_index=block_index,
            reading_order=reading_order,
            region_meta={
                **region_meta,
                "route_backend": "ocr",
                "has_text_regions": bool(region.get("has_text_regions")),
                "image_area_ratio": float(region.get("image_area_ratio") or 0.0),
            },
        )
        if ocr_block is not None:
            return ocr_block

        return BlockNode(
            block_id=f"p{page.number:04d}_b{block_index:04d}",
            page_index=page.number,
            block_type="figure",
            text="Figure",
            markdown="[Figure]",
            reading_order=reading_order,
            bbox=bbox,
            source_mode="layout",
            meta={**region_meta, "route_backend": "placeholder"},
        )

    return None


def _try_extract_image_ocr(
    *,
    page: fitz.Page,
    bbox: tuple[float, float, float, float],
    block_index: int,
    reading_order: int,
    region_meta: dict,
) -> BlockNode | None:
    ocr_mode = os.getenv("BOXBIIBOO_ENABLE_REGION_IMAGE_OCR", "auto").strip().lower()
    if ocr_mode in {"0", "false", "no", "off"}:
        return None
    if ocr_mode == "auto":
        has_text_regions = bool(region_meta.get("has_text_regions"))
        image_area_ratio = float(region_meta.get("image_area_ratio") or 0.0)
        if has_text_regions and image_area_ratio < 0.50:
            region_meta["ocr_skipped_reason"] = "text_page_small_image"
            return None

    try:
        return extract_ocr_region(
            page,
            bbox,
            block_index=block_index,
            reading_order=reading_order,
            block_type_hint="figure",
            region_meta=region_meta,
        )
    except Exception as exc:
        region_meta["ocr_error"] = str(exc)
        return None

