from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import fitz


@dataclass
class PdfProbeResult:
    file_path: str
    page_count: int

    total_chars: int
    total_blocks: int
    total_images: int

    avg_chars_per_page: float
    avg_blocks_per_page: float
    avg_images_per_page: float

    pages_with_text: int
    pages_without_text: int

    text_layer_ratio: float
    empty_text_ratio: float
    likely_scanned_ratio: float
    image_heavy_ratio: float

    avg_text_quality: float

    probe_detected_mode: str
    notes: list[str]
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def probe_pdf(pdf_path: str | Path) -> PdfProbeResult:
    pdf_path = str(pdf_path)
    notes: list[str] = []
    errors: list[str] = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Cannot open PDF: {e}") from e

    page_count = len(doc)

    total_chars = 0
    total_blocks = 0
    total_images = 0

    pages_with_text = 0
    pages_without_text = 0
    likely_scanned_pages = 0
    image_heavy_pages = 0

    text_quality_scores: list[float] = []

    for page in doc:
        try:
            text = page.get_text("text") or ""
        except Exception as e:
            errors.append(f"page {page.number}: get_text failed: {e}")
            text = ""

        try:
            blocks = page.get_text("blocks") or []
        except Exception as e:
            errors.append(f"page {page.number}: get_text('blocks') failed: {e}")
            blocks = []

        try:
            images = page.get_images(full=True) or []
        except Exception as e:
            errors.append(f"page {page.number}: get_images failed: {e}")
            images = []

        char_count = len(text)
        block_count = len(blocks)
        image_count = len(images)

        total_chars += char_count
        total_blocks += block_count
        total_images += image_count

        if char_count > 50:
            pages_with_text += 1
        else:
            pages_without_text += 1

        quality = _estimate_text_quality(text)
        text_quality_scores.append(quality)

        # heuristic: page giống scan nếu có ảnh khá nhiều / text layer kém chất lượng / rất ít text
        page_likely_scanned = (
            (image_count >= 1 and char_count < 80)
            or (image_count >= 1 and quality < 0.35)
            or (char_count < 20 and block_count <= 3)
        )
        if page_likely_scanned:
            likely_scanned_pages += 1

        # heuristic: page image-heavy nếu có ít nhất 1 image và block text không quá nhiều
        page_image_heavy = image_count >= 1 and block_count <= 10
        if page_image_heavy:
            image_heavy_pages += 1

    doc.close()

    safe_page_count = max(page_count, 1)

    avg_chars_per_page = total_chars / safe_page_count
    avg_blocks_per_page = total_blocks / safe_page_count
    avg_images_per_page = total_images / safe_page_count

    text_layer_ratio = pages_with_text / safe_page_count
    empty_text_ratio = pages_without_text / safe_page_count
    likely_scanned_ratio = likely_scanned_pages / safe_page_count
    image_heavy_ratio = image_heavy_pages / safe_page_count
    avg_text_quality = (
        sum(text_quality_scores) / len(text_quality_scores) if text_quality_scores else 0.0
    )

    mode_hint, extra_notes = _decide_mode(
        avg_chars_per_page=avg_chars_per_page,
        avg_blocks_per_page=avg_blocks_per_page,
        avg_images_per_page=avg_images_per_page,
        text_layer_ratio=text_layer_ratio,
        likely_scanned_ratio=likely_scanned_ratio,
        image_heavy_ratio=image_heavy_ratio,
        avg_text_quality=avg_text_quality,
    )
    notes.extend(extra_notes)

    return PdfProbeResult(
        file_path=pdf_path,
        page_count=page_count,
        total_chars=total_chars,
        total_blocks=total_blocks,
        total_images=total_images,
        avg_chars_per_page=avg_chars_per_page,
        avg_blocks_per_page=avg_blocks_per_page,
        avg_images_per_page=avg_images_per_page,
        pages_with_text=pages_with_text,
        pages_without_text=pages_without_text,
        text_layer_ratio=text_layer_ratio,
        empty_text_ratio=empty_text_ratio,
        likely_scanned_ratio=likely_scanned_ratio,
        image_heavy_ratio=image_heavy_ratio,
        avg_text_quality=avg_text_quality,
        probe_detected_mode=mode_hint,
        notes=notes,
        errors=errors,
    )


def _decide_mode(
    *,
    avg_chars_per_page: float,
    avg_blocks_per_page: float,
    avg_images_per_page: float,
    text_layer_ratio: float,
    likely_scanned_ratio: float,
    image_heavy_ratio: float,
    avg_text_quality: float,
) -> tuple[str, list[str]]:
    notes: list[str] = []

    # 1) OCR: gần như scan rõ ràng
    if likely_scanned_ratio >= 0.8 and text_layer_ratio < 0.3:
        notes.append("Most pages look scanned and have weak / missing text layers.")
        return "ocr", notes

    if avg_chars_per_page < 40 and avg_images_per_page >= 1:
        notes.append("Very low extracted text with image-heavy pages suggests OCR-first.")
        return "ocr", notes

    # 2) Mixed: có text layer nhưng không đáng tin hoàn toàn, thường là scan đã OCR sẵn / lai
    if likely_scanned_ratio >= 0.4 and text_layer_ratio >= 0.3:
        notes.append("Document appears mixed: some text layer exists, but many pages look scanned.")
        return "mixed", notes

    if image_heavy_ratio >= 0.4 and avg_text_quality < 0.45:
        notes.append("Image-heavy pages with weak text quality suggest mixed processing.")
        return "mixed", notes

    # 3) Layout: có text layer tốt và block tương đối nhiều
    if text_layer_ratio >= 0.7 and avg_blocks_per_page >= 12:
        notes.append("Rich text layer with many blocks suggests layout-aware extraction.")
        return "layout", notes

    # 4) Text: tài liệu text khá sạch, layout không quá phức tạp
    notes.append("Readable text layer and moderate structure suggest text-first extraction.")
    return "text", notes


def _estimate_text_quality(text: str) -> float:
    """
    Trả về điểm 0..1.
    Đánh giá chất lượng text layer, đặc biệt phát hiện OCR kém.
    """
    if not text or not text.strip():
        return 0.0

    stripped = text.strip()
    total = max(len(stripped), 1)

    # Basic character ratios
    printable_ratio = sum(1 for c in stripped if c.isprintable()) / total
    alnum_ratio = sum(1 for c in stripped if c.isalnum()) / total
    whitespace_ratio = sum(1 for c in stripped if c.isspace()) / total

    # Line analysis
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    if not lines:
        return 0.0

    avg_line_len = sum(len(ln) for ln in lines) / len(lines)

    # OCR quality indicators
    weird_chars = set("@#$%^&*_+=~`|<>[]{}")
    weird_ratio = sum(1 for c in stripped if c in weird_chars) / total

    # Very short lines suggest poor OCR word segmentation
    very_short_lines = sum(1 for ln in lines if len(ln) <= 5) / len(lines)

    # Check for common OCR errors (question marks, random chars)
    question_marks = stripped.count('?') / total
    single_chars_lines = sum(1 for ln in lines if len(ln) == 1) / len(lines)

    # Vietnamese OCR often has these patterns when poor
    viet_ocr_errors = sum(1 for c in stripped.lower() if c in 'qxwv') / total  # common OCR misreads

    score = 0.0
    score += 0.25 * printable_ratio          # Basic readability
    score += 0.25 * alnum_ratio              # Content richness
    score += 0.15 * min(avg_line_len / 40.0, 1.0)  # Reasonable line length
    score += 0.10 * min(whitespace_ratio / 0.15, 1.0)  # Proper spacing
    score -= 0.30 * weird_ratio              # Strange characters
    score -= 0.40 * very_short_lines         # Too many fragmented lines
    score -= 0.50 * min(question_marks * 10, 0.5)  # Question marks indicate OCR errors
    score -= 0.30 * single_chars_lines       # Isolated characters
    score -= 0.20 * min(viet_ocr_errors * 5, 0.2)  # Vietnamese OCR artifacts

    return max(0.0, min(score, 1.0))


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m app.ingest.probe <pdf_path>")

    result = probe_pdf(sys.argv[1])
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
