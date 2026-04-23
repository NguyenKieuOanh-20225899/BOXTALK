from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from statistics import median

import fitz

from app.ingest.schemas import BlockNode, PageNode

_OCR = None


def _get_ocr():
    global _OCR
    if _OCR is None:
        try:
            from paddleocr import PaddleOCR
            import paddle
        except Exception as e:
            raise RuntimeError(f"PaddleOCR is not installed: {e}") from e

        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        lang = os.getenv("BOXBIIBOO_OCR_LANG", "vi")
        use_textline_orientation = os.getenv("BOXBIIBOO_OCR_USE_TEXTLINE_ORIENTATION", "1")
        orientation_enabled = use_textline_orientation.strip().lower() not in {"0", "false", "no"}

        try:
            _OCR = PaddleOCR(
                use_textline_orientation=orientation_enabled,
                lang=lang,
            )
        except TypeError:
            _OCR = PaddleOCR(use_angle_cls=orientation_enabled, lang=lang)
        except Exception as e:
            message = str(e)
            if "ConvertPirAttribute2RuntimeAttribute" in message:
                raise RuntimeError(
                    "PaddleOCR failed during CPU inference because the installed "
                    f"paddlepaddle build is incompatible ({paddle.__version__}). "
                    "Use paddlepaddle==3.2.2 for this repo's OCR and benchmark flow."
                ) from e
            raise
    return _OCR


def extract_with_ocr_backend(pdf_path: str | Path) -> tuple[list[PageNode], list[BlockNode]]:
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))
    ocr = _get_ocr()
    page_scale = float(os.getenv("BOXBIIBOO_OCR_PAGE_SCALE", "2.0"))

    pages: list[PageNode] = []
    blocks: list[BlockNode] = []

    for page in doc:
        page_index = page.number
        page_label = str(page_index + 1)

        pix = page.get_pixmap(matrix=fitz.Matrix(page_scale, page_scale), alpha=False)
        image_path = pdf_path.parent / f"__tmp_ocr_page_{page_index}.png"
        pix.save(str(image_path))

        raw_lines = _run_ocr(ocr, image_path)

        raw_lines = sorted(raw_lines, key=_ocr_sort_key)

        page_blocks: list[BlockNode] = []
        page_text_parts: list[str] = []
        accepted_lines: list[dict] = []

        for i, line in enumerate(raw_lines):
            quad = line["quad"]
            text = line["text"]
            score = line["score"]

            if not text:
                continue
            if score is not None and score < 0.5:
                continue

            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            bbox = (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))

            block_type = _guess_ocr_block_type(text)
            accepted_lines.append(
                {
                    "bbox": bbox,
                    "quad": quad,
                    "text": text,
                    "score": score,
                    "reading_order": i,
                    "block_type": block_type,
                }
            )

            block = BlockNode(
                block_id=f"p{page_index:04d}_b{i:04d}",
                page_index=page_index,
                block_type=block_type,
                text=text,
                markdown=_to_markdown(text, block_type),
                reading_order=i,
                bbox=bbox,
                source_mode="ocr",
                meta={"backend": "paddleocr", "ocr_confidence": score},
            )
            page_blocks.append(block)
            page_text_parts.append(text)

        synthetic_table = _build_synthetic_table_block(
            page=page,
            line_infos=accepted_lines,
            block_index=len(page_blocks),
        )
        if synthetic_table is not None:
            page_blocks.append(synthetic_table)

        visible_blocks = [
            block for block in page_blocks
            if not (block.meta or {}).get("synthetic_table_cluster")
        ]
        pages.append(
            PageNode(
                page_index=page_index,
                page_label=page_label,
                text="\n".join(page_text_parts).strip(),
                markdown="\n\n".join(b.markdown for b in visible_blocks if b.markdown).strip(),
                source_mode="ocr",
                has_ocr=True,
                has_table=any(b.block_type == "table" for b in page_blocks),
                meta={"backend": "paddleocr"},
            )
        )
        blocks.extend(page_blocks)

        try:
            image_path.unlink()
        except Exception:
            pass

    doc.close()
    return pages, blocks


def extract_ocr_region(
    page: fitz.Page,
    bbox: tuple[float, float, float, float],
    *,
    block_index: int,
    reading_order: int | None = None,
    block_type_hint: str | None = None,
    region_meta: dict | None = None,
) -> BlockNode | None:
    rect = fitz.Rect(bbox)
    if rect.is_empty or rect.width < 2 or rect.height < 2:
        return None

    ocr = _get_ocr()
    scale = float(os.getenv("BOXBIIBOO_OCR_REGION_SCALE", "3.0"))
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=rect, alpha=False)

    if pix.width < 2 or pix.height < 2:
        return None

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image_path = Path(tmp.name)

    try:
        pix.save(str(image_path))
        raw_lines = _run_ocr(ocr, image_path)
    finally:
        try:
            image_path.unlink()
        except Exception:
            pass

    raw_lines = sorted(raw_lines, key=_ocr_sort_key)

    texts: list[str] = []
    scores: list[float] = []

    for line in raw_lines:
        text = line["text"]
        score = line["score"]
        if not text:
            continue
        if score is not None and score < 0.5:
            continue
        texts.append(text)
        if score is not None:
            scores.append(score)

    text = "\n".join(texts).strip()
    if not text:
        return None

    block_type = _resolve_ocr_block_type(text, block_type_hint)
    meta = dict(region_meta or {})
    meta.update(
        {
            "backend": "paddleocr_region",
            "ocr_confidence": (sum(scores) / len(scores)) if scores else None,
            "ocr_line_count": len(texts),
        }
    )

    return BlockNode(
        block_id=f"p{page.number:04d}_b{block_index:04d}",
        page_index=page.number,
        block_type=block_type,
        text=text,
        markdown=_to_markdown(text, block_type),
        reading_order=block_index if reading_order is None else reading_order,
        bbox=bbox,
        source_mode="ocr",
        meta=meta,
    )


def _guess_ocr_block_type(text: str) -> str:
    s = text.strip()
    if not s:
        return "paragraph"
    if s.startswith(("- ", "* ", "• ")):
        return "list_item"
    if len(s) < 100 and (s.isupper() or s.startswith(("1.", "2.", "3.", "4.", "5."))):
        return "heading"
    return "paragraph"


def _resolve_ocr_block_type(text: str, block_type_hint: str | None) -> str:
    hinted = (block_type_hint or "").strip().lower()
    if hinted in {"heading", "list_item", "table", "caption", "figure", "metadata"}:
        return hinted
    return _guess_ocr_block_type(text)


def _build_synthetic_table_block(
    *,
    page: fitz.Page,
    line_infos: list[dict],
    block_index: int,
) -> BlockNode | None:
    enabled = os.getenv("BOXBIIBOO_ENABLE_OCR_TABLE_CLUSTER", "1").strip().lower()
    if enabled in {"0", "false", "no"}:
        return None
    if len(line_infos) < 6:
        return None

    ordered = sorted(line_infos, key=lambda item: (item["bbox"][1], item["bbox"][0]))
    heights = [max(1.0, info["bbox"][3] - info["bbox"][1]) for info in ordered]
    median_height = median(heights) if heights else 10.0
    gap_tolerance = max(6.0, median_height * 1.8)

    clusters: list[list[dict]] = []
    current: list[dict] = []
    for info in ordered:
        if not current:
            current = [info]
            continue

        previous = current[-1]
        gap = info["bbox"][1] - previous["bbox"][3]
        if gap <= gap_tolerance:
            current.append(info)
            continue

        clusters.append(current)
        current = [info]

    if current:
        clusters.append(current)

    best_candidate: dict | None = None
    page_width = max(1.0, float(page.rect.width))
    page_height = max(1.0, float(page.rect.height))
    total_lines = max(1, len(ordered))
    structured_page_ratio = sum(
        1
        for info in ordered
        if info["block_type"] in {"heading", "list_item"} or _looks_structured_row(info["text"])
    ) / total_lines

    for cluster in clusters:
        if len(cluster) < 4:
            continue

        bbox = _union_bbox(info["bbox"] for info in cluster)
        width_ratio = (bbox[2] - bbox[0]) / page_width
        height_ratio = (bbox[3] - bbox[1]) / page_height
        if width_ratio < 0.45 or height_ratio < 0.18:
            continue

        digit_ratio = sum(_digit_ratio(info["text"]) for info in cluster) / len(cluster)
        numeric_heavy_ratio = sum(
            1 for info in cluster if _looks_numeric_row(info["text"])
        ) / len(cluster)
        structured_ratio = sum(
            1
            for info in cluster
            if info["block_type"] in {"heading", "list_item"} or _looks_structured_row(info["text"])
        ) / len(cluster)
        coverage_ratio = len(cluster) / total_lines

        if digit_ratio < 0.10 and numeric_heavy_ratio < 0.35 and structured_ratio < 0.65:
            continue

        score = (
            len(cluster) * 0.5
            + digit_ratio * 12.0
            + numeric_heavy_ratio * 10.0
            + structured_ratio * 8.0
            + width_ratio * 2.0
            + coverage_ratio * 4.0
        )

        candidate = {
            "score": score,
            "bbox": bbox,
            "cluster": cluster,
        }
        if best_candidate is None or candidate["score"] > best_candidate["score"]:
            best_candidate = candidate

    if best_candidate is None:
        if len(ordered) < 20 or structured_page_ratio < 0.7:
            return None

        fallback_cluster = ordered[2:-2] if len(ordered) > 8 else ordered
        fallback_bbox = _union_bbox(info["bbox"] for info in fallback_cluster)
        width_ratio = (fallback_bbox[2] - fallback_bbox[0]) / page_width
        height_ratio = (fallback_bbox[3] - fallback_bbox[1]) / page_height
        if width_ratio < 0.45 or height_ratio < 0.2:
            return None

        best_candidate = {
            "score": 0.0,
            "bbox": fallback_bbox,
            "cluster": fallback_cluster,
        }

    cluster = best_candidate["cluster"]
    table_text = "\n".join(info["text"] for info in cluster).strip()
    if not table_text:
        return None

    scores = [info["score"] for info in cluster if info.get("score") is not None]
    return BlockNode(
        block_id=f"p{page.number:04d}_b{block_index:04d}",
        page_index=page.number,
        block_type="table",
        text=table_text,
        markdown=table_text,
        reading_order=block_index,
        bbox=best_candidate["bbox"],
        source_mode="ocr",
        meta={
            "backend": "paddleocr_table_cluster",
            "synthetic_table_cluster": True,
            "ocr_confidence": (sum(scores) / len(scores)) if scores else None,
            "ocr_line_count": len(cluster),
        },
    )


def _union_bbox(boxes) -> tuple[float, float, float, float]:
    box_list = list(boxes)
    return (
        min(box[0] for box in box_list),
        min(box[1] for box in box_list),
        max(box[2] for box in box_list),
        max(box[3] for box in box_list),
    )


def _digit_ratio(text: str) -> float:
    stripped = text.strip()
    if not stripped:
        return 0.0
    total = sum(1 for ch in stripped if ch.isalnum())
    if total <= 0:
        return 0.0
    digits = sum(1 for ch in stripped if ch.isdigit())
    return digits / total


def _looks_numeric_row(text: str) -> bool:
    tokens = re.findall(r"\d+(?:[\.,]\d+)?", text)
    return len(tokens) >= 2


def _looks_structured_row(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if len(stripped.split()) <= 3:
        return True
    return bool(re.search(r"\b\d+(?:[\.,]\d+)?\b", stripped)) and len(stripped.split()) <= 8


def _run_ocr(ocr, image_path: str | Path) -> list[dict]:
    image_path = str(image_path)

    predict = getattr(ocr, "predict", None)
    try:
        if callable(predict):
            result = predict(image_path)
        else:
            try:
                result = ocr.ocr(image_path, cls=True)
            except TypeError:
                result = ocr.ocr(image_path)
    except Exception as e:
        message = str(e)
        if "ConvertPirAttribute2RuntimeAttribute" in message:
            raise RuntimeError(
                "PaddleOCR inference hit a known paddlepaddle 3.3.x CPU regression. "
                "Use paddlepaddle==3.2.2 for OCR in this repo."
            ) from e
        raise

    return _normalize_ocr_result(result)


def _normalize_ocr_result(result) -> list[dict]:
    if not result:
        return []

    if _looks_like_ocr_result_mapping(result):
        return _normalize_ocr_result_mapping(result)

    if isinstance(result, list):
        if not result:
            return []

        first = result[0]
        if _looks_like_ocr_result_mapping(first):
            return _normalize_ocr_result_mapping(first)

        if _looks_like_classic_ocr_line(first):
            lines: list[dict] = []
            for line in result:
                normalized = _normalize_classic_ocr_line(line)
                if normalized is not None:
                    lines.append(normalized)
            return lines

        if isinstance(first, list) and first:
            if _looks_like_classic_ocr_line(first[0]):
                lines = []
                for line in first:
                    normalized = _normalize_classic_ocr_line(line)
                    if normalized is not None:
                        lines.append(normalized)
                return lines

            if _looks_like_ocr_result_mapping(first[0]):
                return _normalize_ocr_result_mapping(first[0])

    return []


def _looks_like_ocr_result_mapping(item) -> bool:
    if not hasattr(item, "keys"):
        return False
    keys = set(item.keys())
    return bool(keys & {"rec_texts", "rec_scores", "dt_polys", "rec_polys", "rec_boxes"})


def _normalize_ocr_result_mapping(item) -> list[dict]:
    texts = list(item.get("rec_texts") or [])
    scores = list(item.get("rec_scores") or [])
    polys = item.get("rec_polys") or item.get("dt_polys") or item.get("rec_boxes") or []

    line_count = len(texts)
    if polys:
        line_count = min(line_count, len(polys))

    lines: list[dict] = []
    for i in range(line_count):
        text = str(texts[i] or "").strip()
        if not text:
            continue

        quad = _to_quad(polys[i]) if polys else None
        if quad is None:
            continue

        score = None
        if i < len(scores) and scores[i] is not None:
            try:
                score = float(scores[i])
            except Exception:
                score = None

        lines.append({"quad": quad, "text": text, "score": score})

    return lines


def _looks_like_classic_ocr_line(line) -> bool:
    if not isinstance(line, (list, tuple)) or len(line) < 2:
        return False
    quad, payload = line[0], line[1]
    return isinstance(payload, (list, tuple)) and len(payload) >= 1 and quad is not None


def _normalize_classic_ocr_line(line) -> dict | None:
    if not _looks_like_classic_ocr_line(line):
        return None

    quad = _to_quad(line[0])
    if quad is None:
        return None

    payload = line[1]
    text = str(payload[0] or "").strip()
    if not text:
        return None

    score = None
    if len(payload) > 1 and payload[1] is not None:
        try:
            score = float(payload[1])
        except Exception:
            score = None

    return {"quad": quad, "text": text, "score": score}


def _to_quad(poly) -> list[tuple[float, float]] | None:
    if poly is None:
        return None

    if hasattr(poly, "tolist"):
        poly = poly.tolist()

    if not isinstance(poly, (list, tuple)) or not poly:
        return None

    if len(poly) == 4 and all(isinstance(v, (int, float)) for v in poly):
        x0, y0, x1, y1 = [float(v) for v in poly]
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    quad: list[tuple[float, float]] = []
    for point in poly:
        if hasattr(point, "tolist"):
            point = point.tolist()
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        quad.append((float(point[0]), float(point[1])))

    if len(quad) < 4:
        return None
    return quad[:4]


def _ocr_sort_key(line):
    quad = line["quad"]
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return (min(ys), min(xs))


def _to_markdown(text: str, block_type: str) -> str:
    if block_type == "heading":
        return f"## {text}"
    if block_type == "list_item":
        return text if text.startswith(("- ", "* ", "• ")) else f"- {text}"
    return text
