from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import fitz

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ingest.pipeline import ingest_pdf
from app.ingest.region.debug import draw_regions_debug
from app.ingest.region.detector import detect_regions


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _short(text: str, limit: int = 240) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def _write_preview(
    path: Path,
    *,
    pdf_path: Path,
    result: dict[str, Any],
    overlay_path: Path | None,
    max_blocks: int,
    max_chunks: int,
) -> None:
    pages = result.get("pages") or []
    blocks = result.get("blocks") or []
    chunks = result.get("chunks") or []
    probe = result.get("probe") or {}

    block_type_counts: dict[str, int] = {}
    route_counts: dict[str, int] = {}
    for block in blocks:
        block_type_counts[block.block_type] = block_type_counts.get(block.block_type, 0) + 1
        meta = block.meta or {}
        route = str(meta.get("route_backend") or block.source_mode or "unknown")
        route_counts[route] = route_counts.get(route, 0) + 1

    lines: list[str] = []
    lines.append(f"# Ingest preview: {pdf_path.name}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- PDF: `{pdf_path}`")
    lines.append(f"- Used backend: `{result.get('used_backend')}`")
    lines.append(f"- Probe mode: `{probe.get('probe_detected_mode')}`")
    lines.append(f"- Pages: {len(pages)}")
    lines.append(f"- Blocks: {len(blocks)}")
    lines.append(f"- Chunks: {len(chunks)}")
    lines.append(f"- Block type counts: `{block_type_counts}`")
    lines.append(f"- Route backend counts: `{route_counts}`")
    if result.get("errors"):
        lines.append(f"- Fallback errors: `{result.get('errors')}`")
    if overlay_path is not None:
        lines.append(f"- Region overlay: `{overlay_path}`")
    lines.append("")

    lines.append("## Blocks")
    lines.append("")
    lines.append("| # | page | type | route | bbox | text |")
    lines.append("|---:|---:|---|---|---|---|")
    for i, block in enumerate(blocks[:max_blocks], start=1):
        meta = block.meta or {}
        route = meta.get("route_backend") or block.source_mode
        bbox = block.bbox
        bbox_text = "" if bbox is None else "[" + ", ".join(f"{float(v):.1f}" for v in bbox) + "]"
        text = _short(block.text).replace("|", "\\|")
        lines.append(
            f"| {i} | {block.page_index + 1} | `{block.block_type}` | `{route}` | `{bbox_text}` | {text} |"
        )
    if len(blocks) > max_blocks:
        lines.append(f"| ... | ... | ... | ... | ... | truncated; see `blocks.jsonl` |")
    lines.append("")

    table_blocks = [block for block in blocks if block.block_type == "table"]
    if table_blocks:
        lines.append("## Table Blocks")
        lines.append("")
        for table_i, block in enumerate(table_blocks, start=1):
            meta = block.meta or {}
            lines.append(f"### Table {table_i}: `{block.block_id}`")
            lines.append("")
            lines.append(f"- Page: {block.page_index + 1}")
            lines.append(f"- Backend: `{meta.get('table_backend') or meta.get('backend') or block.source_mode}`")
            lines.append(f"- Route backend: `{meta.get('route_backend')}`")
            trace = meta.get("extraction_trace") or {}
            if trace:
                lines.append(f"- Extraction trace: `{trace}`")
            lines.append("")
            markdown = block.markdown or meta.get("table_markdown") or ""
            if markdown:
                lines.append("```markdown")
                lines.append(markdown[:4000])
                lines.append("```")
            else:
                lines.append("_No table markdown available._")
            lines.append("")

    lines.append("## Chunks")
    lines.append("")
    lines.append("| # | pages | block_types | text |")
    lines.append("|---:|---|---|---|")
    for chunk in chunks[:max_chunks]:
        pages_text = ",".join(str(i + 1) for i in (chunk.page_indices or []))
        types_text = ",".join(chunk.block_types or [])
        text = _short(chunk.text).replace("|", "\\|")
        lines.append(f"| {chunk.chunk_index} | `{pages_text}` | `{types_text}` | {text} |")
    if len(chunks) > max_chunks:
        lines.append(f"| ... | ... | ... | truncated; see `chunks.jsonl` |")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a visual and textual preview of ingest output for one PDF.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--overlay-page", type=int, default=1, help="1-based physical PDF page for region overlay.")
    parser.add_argument("--max-blocks", type=int, default=80)
    parser.add_argument("--max-chunks", type=int, default=40)
    args = parser.parse_args()

    pdf_path = args.pdf
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    result = ingest_pdf(pdf_path)

    _write_json(out_dir / "summary.json", {
        "pdf": str(pdf_path),
        "used_backend": result.get("used_backend"),
        "probe": result.get("probe"),
        "page_count": len(result.get("pages") or []),
        "block_count": len(result.get("blocks") or []),
        "chunk_count": len(result.get("chunks") or []),
        "errors": result.get("errors") or [],
    })
    _write_jsonl(out_dir / "blocks.jsonl", result.get("blocks") or [])
    _write_jsonl(out_dir / "chunks.jsonl", result.get("chunks") or [])
    _write_json(out_dir / "pages.json", result.get("pages") or [])

    overlay_path: Path | None = None
    with fitz.open(str(pdf_path)) as doc:
        if 1 <= args.overlay_page <= len(doc):
            page = doc[args.overlay_page - 1]
            regions = detect_regions(page)
            overlay_path = out_dir / f"region_overlay_page{args.overlay_page}.png"
            draw_regions_debug(page, regions, overlay_path)

    _write_preview(
        out_dir / "preview.md",
        pdf_path=pdf_path,
        result=result,
        overlay_path=overlay_path,
        max_blocks=args.max_blocks,
        max_chunks=args.max_chunks,
    )

    print(f"Wrote ingest visualization to: {out_dir}")
    print(f"Preview: {out_dir / 'preview.md'}")
    if overlay_path is not None:
        print(f"Overlay: {overlay_path}")


if __name__ == "__main__":
    main()
