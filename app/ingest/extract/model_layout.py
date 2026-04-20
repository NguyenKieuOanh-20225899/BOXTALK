from __future__ import annotations

import os
from pathlib import Path

import fitz

from app.ingest.schemas import BlockNode, PageNode

_MODEL_BUNDLE = None
DEFAULT_LAYOUT_MODEL_NAME = "Aryn/deformable-detr-DocLayNet"

_LABEL_TO_BLOCK_TYPE = {
    "caption": "caption",
    "figure": "figure",
    "footnote": "metadata",
    "formula": "paragraph",
    "list": "list_item",
    "page-footer": "metadata",
    "page-header": "metadata",
    "picture": "figure",
    "section-header": "heading",
    "table": "table",
    "text": "paragraph",
    "title": "heading",
}


def is_model_layout_enabled() -> bool:
    return _model_name() is not None


def extract_with_model_layout_backend(
    pdf_path: str | Path,
) -> tuple[list[PageNode], list[BlockNode]]:
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    pages: list[PageNode] = []
    blocks: list[BlockNode] = []

    for page in doc:
        page_regions = detect_model_layout_regions_for_page(page)
        page_blocks = _regions_to_blocks(page_regions)

        pages.append(
            PageNode(
                page_index=page.number,
                page_label=str(page.number + 1),
                text="\n".join(b.text for b in page_blocks).strip(),
                markdown="\n\n".join(b.markdown for b in page_blocks if b.markdown).strip(),
                source_mode="layout",
                has_ocr=False,
                has_table=any(b.block_type == "table" for b in page_blocks),
                meta={
                    "backend": "transformers_object_detection",
                    "model_name": _model_name(),
                    "region_count": len(page_blocks),
                },
            )
        )
        blocks.extend(page_blocks)

    doc.close()

    if not blocks:
        raise RuntimeError("Model layout backend produced no blocks")

    return pages, blocks


def detect_model_layout_regions_for_page(page: fitz.Page) -> list[dict]:
    processor, model, torch, image_cls = _get_model_bundle()

    threshold = float(os.getenv("BOXBIIBOO_LAYOUT_SCORE_THRESHOLD", "0.35"))
    render_scale = float(os.getenv("BOXBIIBOO_LAYOUT_RENDER_SCALE", "2.0"))

    matrix = fitz.Matrix(render_scale, render_scale)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    image = image_cls.frombytes("RGB", (pix.width, pix.height), pix.samples)

    inputs = processor(images=image, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_object_detection(
        outputs,
        threshold=threshold,
        target_sizes=[(image.height, image.width)],
    )[0]

    scale_x = page.rect.width / float(image.width)
    scale_y = page.rect.height / float(image.height)

    regions: list[dict] = []
    for idx, (score, label_id, bbox_tensor) in enumerate(
        zip(results["scores"], results["labels"], results["boxes"])
    ):
        label_name = _label_name(model.config.id2label, int(label_id))
        canonical_type = _normalize_model_label(label_name)
        if canonical_type is None:
            continue

        x0, y0, x1, y1 = [float(v) for v in bbox_tensor.tolist()]
        pdf_bbox = (
            max(0.0, x0 * scale_x),
            max(0.0, y0 * scale_y),
            min(page.rect.width, x1 * scale_x),
            min(page.rect.height, y1 * scale_y),
        )
        rect = fitz.Rect(pdf_bbox)
        if rect.is_empty or rect.width < 4 or rect.height < 4:
            continue

        regions.append(
            {
                "region_id": f"p{page.number:04d}_r{idx:04d}",
                "idx": idx,
                "page_index": page.number,
                "page_label": str(page.number + 1),
                "score": float(score),
                "label_name": label_name,
                "block_type": canonical_type,
                "direct_text": _extract_text_for_region(page, rect),
                "bbox": pdf_bbox,
            }
        )

    regions = _dedupe_regions(regions)
    regions.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], -item["score"]))
    return regions


def _regions_to_blocks(regions: list[dict]) -> list[BlockNode]:
    if not regions:
        return []

    page_index = int(regions[0]["page_index"])
    page_blocks: list[BlockNode] = []
    for reading_order, region in enumerate(regions):
        text = str(region.get("direct_text") or "").strip()
        block_type = str(region.get("block_type") or "paragraph")
        if not text and block_type not in {"table", "figure", "caption", "metadata"}:
            continue

        placeholder = _placeholder_text(block_type)
        resolved_text = text or placeholder

        page_blocks.append(
            BlockNode(
                block_id=f"p{page_index:04d}_b{reading_order:04d}",
                page_index=page_index,
                block_type=block_type,
                text=resolved_text,
                markdown=_to_markdown(resolved_text, block_type),
                reading_order=reading_order,
                bbox=region["bbox"],
                source_mode="layout",
                meta={
                    "backend": "transformers_object_detection",
                    "model_score": region["score"],
                    "model_label": region.get("label_name"),
                    "region_id": region.get("region_id"),
                },
            )
        )

    return page_blocks


def _get_model_bundle():
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is not None:
        return _MODEL_BUNDLE

    model_name = _model_name()
    if not model_name:
        raise RuntimeError(
            "Model layout backend is disabled. Set BOXBIIBOO_LAYOUT_MODEL_NAME to enable it."
        )

    try:
        import torch
        from PIL import Image
        from transformers import AutoImageProcessor, AutoModelForObjectDetection
    except Exception as e:
        raise RuntimeError(
            "Model layout backend requires torch, pillow, and transformers"
        ) from e

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForObjectDetection.from_pretrained(model_name)
    model.eval()

    device_name = os.getenv("BOXBIIBOO_LAYOUT_DEVICE", "cpu")
    device = torch.device(device_name)
    model.to(device)

    _MODEL_BUNDLE = (processor, model, torch, Image)
    return _MODEL_BUNDLE


def _model_name() -> str | None:
    model_name = os.getenv("BOXBIIBOO_LAYOUT_MODEL_NAME")
    if not model_name:
        return None

    normalized = model_name.strip()
    if not normalized or normalized.lower() in {"0", "false", "off", "none"}:
        return None
    if normalized.lower() == "default":
        return DEFAULT_LAYOUT_MODEL_NAME
    return normalized


def _extract_text_for_region(page: fitz.Page, rect: fitz.Rect) -> str:
    text = page.get_textbox(rect).strip()
    if text:
        return text

    clipped = page.get_text("text", clip=rect, sort=True).strip()
    if clipped:
        return clipped

    return ""


def _label_name(id2label: dict[int, str] | dict[str, str], label_id: int) -> str:
    return str(id2label.get(label_id) or id2label.get(str(label_id)) or "").strip().lower()


def _normalize_model_label(label_name: str) -> str | None:
    name = label_name.lower().strip()
    if not name:
        return None

    if name in _LABEL_TO_BLOCK_TYPE:
        return _LABEL_TO_BLOCK_TYPE[name]

    if "table" in name:
        return "table"
    if "header" in name or "title" in name:
        return "heading"
    if "list" in name:
        return "list_item"
    if "caption" in name:
        return "caption"
    if "figure" in name or "image" in name or "picture" in name:
        return "figure"
    if "footer" in name or "meta" in name:
        return "metadata"
    if "text" in name or "paragraph" in name:
        return "paragraph"

    return None


def _dedupe_regions(regions: list[dict]) -> list[dict]:
    kept: list[dict] = []
    for region in sorted(regions, key=lambda item: (-item["score"], item["bbox"][1], item["bbox"][0])):
        if any(
            region["block_type"] == other["block_type"]
            and _iou(region["bbox"], other["bbox"]) >= 0.75
            for other in kept
        ):
            continue
        kept.append(region)
    return kept


def _iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0

    inter = (x1 - x0) * (y1 - y0)
    left_area = max(1.0, (left[2] - left[0]) * (left[3] - left[1]))
    right_area = max(1.0, (right[2] - right[0]) * (right[3] - right[1]))
    return inter / (left_area + right_area - inter)


def _placeholder_text(block_type: str) -> str:
    if block_type == "table":
        return "Table"
    if block_type == "figure":
        return "Figure"
    if block_type == "caption":
        return "Caption"
    if block_type == "metadata":
        return "Metadata"
    return ""


def _to_markdown(text: str, block_type: str) -> str:
    if not text:
        return ""

    if block_type == "heading":
        return f"## {text}"
    if block_type == "list_item":
        return text if text.startswith(("- ", "* ", "• ")) else f"- {text}"
    if block_type == "figure":
        return f"[{text}]"
    return text
