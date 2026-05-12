from __future__ import annotations

import os
from pathlib import Path

import fitz

from app.ingest.extract.ocr import extract_ocr_region
from app.ingest.extract.text import extract_text_region
from app.ingest.reading_order import sort_in_reading_order
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
    regions: list[dict] = []

    raw_blocks = page.get_text("blocks") or []
    for idx, raw in enumerate(raw_blocks):
        x0, y0, x1, y1, text, *_ = raw
        text = str(text or "").strip()
        bbox = (float(x0), float(y0), float(x1), float(y1))
        if text:
            regions.append(
                {
                    "region_id": f"p{page.number:04d}_text_{idx:04d}",
                    "kind": "text",
                    "bbox": bbox,
                    "text": text,
                    "page_index": page.number,
                }
            )

    text_bboxes = [tuple(region["bbox"]) for region in regions if region["kind"] == "text"]
    has_text_regions = bool(text_bboxes)
    page_area = max(1.0, float(page.rect.width * page.rect.height))
    image_min_area = float(os.getenv("BOXBIIBOO_REGION_IMAGE_MIN_AREA", "1600"))
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
            if _mostly_text_overlap(bbox, text_bboxes):
                continue
            regions.append(
                {
                    "region_id": f"p{page.number:04d}_image_{image_index:04d}_{rect_index:04d}",
                    "kind": "image",
                    "bbox": bbox,
                    "text": "",
                    "page_index": page.number,
                    "has_text_regions": has_text_regions,
                    "image_area_ratio": _area(bbox) / page_area,
                }
            )

    return sort_in_reading_order(
        regions,
        bbox_getter=lambda item: tuple(item["bbox"]),
        page_width=float(page.rect.width),
        page_height=float(page.rect.height),
    )


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
    }

    if kind == "text":
        region_dict = {
            "bbox": bbox,
            "page_index": page.number,
            "text": str(region.get("text") or "").strip(),
            "meta": {**region_meta, "route_backend": "text"},
        }
        return extract_text_region(
            region_dict,
            page.number,
            block_index,
            reading_order=reading_order,
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


def _area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _mostly_text_overlap(
    bbox: tuple[float, float, float, float],
    text_bboxes: list[tuple[float, float, float, float]],
) -> bool:
    bbox_area = max(_area(bbox), 1.0)
    overlap = sum(_intersection_area(bbox, text_bbox) for text_bbox in text_bboxes)
    return overlap / bbox_area >= 0.35


def _intersection_area(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)
