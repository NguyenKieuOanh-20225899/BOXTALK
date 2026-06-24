from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from demo.bootstrap import ensure_repo_on_path

ensure_repo_on_path()


def ensure_output_root(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def write_outputs(
    result: dict[str, Any],
    *,
    output_dir: Path,
    save_overlay: bool,
) -> dict[str, Path]:
    output_dir = ensure_output_root(output_dir)
    page_number = int(result["page"]["number"])
    slug = page_slug(page_number)

    blocks_path = output_dir / f"{slug}_blocks.json"
    summary_path = output_dir / f"{slug}_summary.json"
    text_path = output_dir / f"{slug}_text.md"

    blocks_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "document": result["document"],
                "page": result["page"],
                "summary": result["summary"],
                "timing": result["timing"],
                "warnings": result.get("warnings", []),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    text_path.write_text(_page_markdown(result), encoding="utf-8")

    files: dict[str, Path] = {
        "blocks_json": blocks_path,
        "summary_json": summary_path,
        "text_markdown": text_path,
    }

    table_index = 0
    for block in result.get("blocks", []):
        if block.get("type") != "table":
            continue
        table_index += 1
        table_path = output_dir / f"{slug}_table_{table_index:02d}.md"
        table_md = (
            (block.get("table") or {}).get("markdown")
            or block.get("markdown")
            or block.get("content")
            or ""
        )
        table_path.write_text(
            _table_markdown(block=block, page_number=page_number, table_md=str(table_md)),
            encoding="utf-8",
        )
        files[f"table_{table_index:02d}_markdown"] = table_path

    if save_overlay:
        overlay_path = output_dir / f"{slug}_overlay.png"
        try:
            save_overlay_image(
                pdf_path=Path(result["document"]["path"]),
                page_number=page_number,
                regions=result.get("regions", []),
                output_path=overlay_path,
            )
            files["overlay_png"] = overlay_path
        except Exception as exc:
            result.setdefault("warnings", []).append(f"Khong tao duoc overlay: {exc}")
            summary_path.write_text(
                json.dumps(
                    {
                        "document": result["document"],
                        "page": result["page"],
                        "summary": result["summary"],
                        "timing": result["timing"],
                        "warnings": result.get("warnings", []),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

    result["output_files"] = {key: str(path) for key, path in files.items()}
    blocks_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return files


def page_slug(page_number: int) -> str:
    return f"page_{page_number:02d}"


def _page_markdown(result: dict[str, Any]) -> str:
    lines = [
        f"# {result['document']['name']} - page {result['page']['number']}",
        "",
        "## Blocks",
        "",
    ]
    for block in result.get("blocks", []):
        lines.append(f"### {block.get('block_id')} ({block.get('type')})")
        content = block.get("markdown") or block.get("content") or ""
        lines.append(str(content).strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _table_markdown(*, block: dict[str, Any], page_number: int, table_md: str) -> str:
    metadata = block.get("metadata") if isinstance(block.get("metadata"), dict) else {}
    table = block.get("table") if isinstance(block.get("table"), dict) else {}
    block_id = str(block.get("block_id") or metadata.get("block_id") or "unknown")
    route = (
        metadata.get("route_backend")
        or (block.get("extraction") or {}).get("route")
        or block.get("source_mode")
        or "unknown"
    )
    table_backend = (
        table.get("backend")
        or metadata.get("table_backend")
        or metadata.get("backend")
        or route
    )
    bbox = block.get("bbox") or metadata.get("table_bbox") or metadata.get("region_bbox")

    lines = [
        f"# Table chunk `{block_id}`",
        "",
        "## Metadata",
        "",
        f"- `chunk_id`: `{block_id}`",
        f"- `block_id`: `{block_id}`",
        f"- `page`: `{page_number}`",
        f"- `type`: `{block.get('type') or 'table'}`",
        f"- `route`: `{route}`",
        f"- `table_backend`: `{table_backend}`",
    ]
    if bbox:
        lines.append(f"- `bbox`: `{bbox}`")
    lines.extend(["", "## Table", "", table_md.strip(), ""])
    return "\n".join(lines)


def save_overlay_image(
    *,
    pdf_path: Path,
    page_number: int,
    regions: list[dict[str, Any]],
    output_path: Path,
) -> Path:
    import fitz
    from PIL import Image, ImageDraw, ImageFont

    from app.ingest.region.debug import REGION_COLORS, draw_regions_debug

    draw_regions = [_region_for_overlay(region) for region in regions]
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_number - 1]
        draw_regions_debug(page, draw_regions, output_path, scale=2.0)
    finally:
        doc.close()

    image = Image.open(output_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    scale = 2.0

    for region in regions:
        bbox = region.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        label = _demo_overlay_label(region)
        x0, y0, _, _ = [float(value) * scale for value in bbox[:4]]
        y = max(0, y0 + 2)
        color = REGION_COLORS.get(str(region.get("original_type") or region.get("type")), "#000000")
        text_width = max(60, len(label) * 6)
        draw.rectangle([x0, y, x0 + text_width, y + 12], fill="white")
        draw.text((x0 + 2, y), label, fill=color, font=font)

    _draw_legend(draw, font, REGION_COLORS)
    image.save(output_path)
    return output_path


def _region_for_overlay(region: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": region.get("original_type") or region.get("type") or "unknown",
        "type": region.get("original_type") or region.get("type") or "unknown",
        "bbox": region.get("bbox"),
        "route_backend": region.get("actual_route") or region.get("planned_route"),
    }


def _demo_overlay_label(region: dict[str, Any]) -> str:
    block_id = region.get("block_id") or region.get("region_id") or "no_block"
    kind = region.get("type") or "unknown"
    route = region.get("actual_route") or region.get("planned_route") or "unknown"
    return f"{block_id} | {kind} | {route}"


def _draw_legend(draw: Any, font: Any, colors: dict[str, str]) -> None:
    x = 12
    y = 12
    items = [
        ("text", colors.get("text", "#7f7f7f")),
        ("heading", colors.get("heading", "#1f77b4")),
        ("table", colors.get("table", "#d62728")),
        ("image", colors.get("image", "#ff7f0e")),
    ]
    width = 190
    height = 16 * len(items) + 10
    draw.rectangle([x, y, x + width, y + height], fill="white", outline="#444444")
    for index, (label, color) in enumerate(items):
        yy = y + 6 + index * 16
        draw.rectangle([x + 8, yy + 2, x + 20, yy + 12], fill=color)
        draw.text((x + 26, yy), label, fill="#111111", font=font)
