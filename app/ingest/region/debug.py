from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import fitz


REGION_COLORS: dict[str, str] = {
    "heading": "#1f77b4",
    "paragraph": "#7f7f7f",
    "text": "#7f7f7f",
    "list_item": "#2ca02c",
    "table": "#d62728",
    "image": "#ff7f0e",
    "caption": "#9467bd",
    "metadata": "#8c564b",
    "header": "#17becf",
    "footer": "#17becf",
}


def draw_regions_debug(
    page: fitz.Page,
    regions: list[dict[str, Any]],
    output_path: str | Path,
    *,
    scale: float = 2.0,
) -> Path:
    """Render a PDF page with colored region bounding boxes.

    This helper is intentionally isolated from the ingest path. It is used for
    thesis/debug artifacts that explain which regions were detected and which
    backend each region is routed to.
    """

    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:  # pragma: no cover - depends on optional Pillow
        raise RuntimeError("Pillow is required to draw region overlays") from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for region in regions:
        kind = str(region.get("kind") or region.get("type") or "unknown")
        bbox = region.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x0, y0, x1, y1 = [float(value) * scale for value in bbox[:4]]
        color = REGION_COLORS.get(kind, "#000000")
        label = _region_label(region, kind)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=max(2, int(scale * 1.5)))
        text_y = max(0.0, y0 - 12.0 * scale)
        draw.rectangle([x0, text_y, x0 + max(40.0, len(label) * 6.0), y0], fill="white")
        draw.text((x0 + 2.0, text_y), label, fill=color, font=font)

    image.save(output_path)
    return output_path


def _region_label(region: dict[str, Any], kind: str) -> str:
    backend = region.get("route_backend")
    if backend:
        return f"{kind}->{backend}"
    return kind

