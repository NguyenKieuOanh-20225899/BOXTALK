from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ingest.pipeline import ingest_pdf


META_KEYS = [
    "backend",
    "region_id",
    "region_type",
    "region_kind",
    "region_bbox",
    "page_number",
    "route_backend",
    "route_reason",
    "confidence",
    "source",
    "detection_source",
    "fallback_used",
    "table_route_fallback",
    "table_backend",
    "table_id",
    "table_row_count",
    "table_col_count",
    "table_cell_count",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare ingest output with region routing ON and OFF for the same PDF."
    )
    parser.add_argument("pdf", type=Path, help="Input PDF path.")
    parser.add_argument("--page", type=int, default=1, help="1-based page number to inspect.")
    parser.add_argument("--out-dir", type=Path, default=Path("docs/chapter5/region_compare"))
    parser.add_argument("--text-preview", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    off = _run_ingest(args.pdf, region_enabled=False, page=args.page, text_preview=args.text_preview)
    on = _run_ingest(args.pdf, region_enabled=True, page=args.page, text_preview=args.text_preview)
    payload = {
        "pdf": str(args.pdf),
        "page": args.page,
        "region_off": off,
        "region_on": on,
        "comparison": _compare(off, on),
    }

    stem = _safe_stem(args.pdf)
    json_path = args.out_dir / f"{stem}_page{args.page}_region_compare.json"
    md_path = args.out_dir / f"{stem}_page{args.page}_region_compare.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(payload), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def _run_ingest(pdf: Path, *, region_enabled: bool, page: int, text_preview: int) -> dict[str, Any]:
    previous = os.environ.get("BOXBIIBOO_ENABLE_REGION_ROUTING")
    os.environ["BOXBIIBOO_ENABLE_REGION_ROUTING"] = "1" if region_enabled else "0"
    try:
        result = ingest_pdf(pdf)
    finally:
        if previous is None:
            os.environ.pop("BOXBIIBOO_ENABLE_REGION_ROUTING", None)
        else:
            os.environ["BOXBIIBOO_ENABLE_REGION_ROUTING"] = previous

    page_index = page - 1
    pages = result["pages"]
    blocks = [block for block in result["blocks"] if block.page_index == page_index]
    chunks = [chunk for chunk in result["chunks"] if page_index in chunk.page_indices]
    page_node = pages[page_index] if 0 <= page_index < len(pages) else None

    return {
        "region_enabled": region_enabled,
        "used_backend": result.get("used_backend"),
        "errors": result.get("errors") or [],
        "page_meta": dict(page_node.meta or {}) if page_node is not None else {},
        "block_count_page": len(blocks),
        "chunk_count_page": len(chunks),
        "block_type_counts": _count(block.block_type for block in blocks),
        "route_backend_counts": _count((block.meta or {}).get("route_backend", "<none>") for block in blocks),
        "blocks": [_block_summary(block, text_preview=text_preview) for block in blocks],
        "chunks": [_chunk_summary(chunk, text_preview=text_preview) for chunk in chunks],
    }


def _block_summary(block: Any, *, text_preview: int) -> dict[str, Any]:
    meta = dict(block.meta or {})
    return {
        "block_id": block.block_id,
        "block_type": block.block_type,
        "source_mode": block.source_mode,
        "reading_order": block.reading_order,
        "bbox": block.bbox,
        "text_preview": _preview(block.text, text_preview),
        "trace_meta": {key: meta[key] for key in META_KEYS if key in meta},
    }


def _chunk_summary(chunk: Any, *, text_preview: int) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "chunk_index": chunk.chunk_index,
        "page_indices": chunk.page_indices,
        "block_ids": chunk.block_ids,
        "block_types": chunk.block_types,
        "source_mode": chunk.source_mode,
        "text_preview": _preview(chunk.text, text_preview),
        "meta": dict(chunk.meta or {}),
    }


def _compare(off: dict[str, Any], on: dict[str, Any]) -> dict[str, Any]:
    return {
        "used_backend_changed": off["used_backend"] != on["used_backend"],
        "block_count_delta": on["block_count_page"] - off["block_count_page"],
        "chunk_count_delta": on["chunk_count_page"] - off["chunk_count_page"],
        "region_on_has_route_trace": any(
            block.get("trace_meta", {}).get("route_backend") for block in on.get("blocks", [])
        ),
        "region_off_has_route_trace": any(
            block.get("trace_meta", {}).get("route_backend") for block in off.get("blocks", [])
        ),
    }


def _to_markdown(payload: dict[str, Any]) -> str:
    off = payload["region_off"]
    on = payload["region_on"]
    lines = [
        "# Region Routing ON/OFF Comparison",
        "",
        f"- PDF: `{payload['pdf']}`",
        f"- Page: `{payload['page']}`",
        "",
        "## Summary",
        "",
        "| Config | Used backend | Page blocks | Page chunks | Block types | Route backends |",
        "| --- | --- | ---: | ---: | --- | --- |",
        _summary_row("Region OFF", off),
        _summary_row("Region ON", on),
        "",
        "## Interpretation Checklist",
        "",
        "- `Used backend` cho biet pipeline chon backend nao sau validation/fallback.",
        "- `Route backends` chi co y nghia ro nhat khi `region_routed` duoc dung.",
        "- Neu Region ON co `route_backend`/`region_id` trong block metadata, co the trace tung block ve region goc.",
        "- So sanh block/chunk count de xem region co lam tach nho noi dung hay giu cau truc bang/hinh tot hon khong.",
        "",
        "## Region OFF Blocks",
        "",
        *_blocks_markdown(off["blocks"]),
        "",
        "## Region ON Blocks",
        "",
        *_blocks_markdown(on["blocks"]),
    ]
    return "\n".join(lines).strip() + "\n"


def _summary_row(label: str, result: dict[str, Any]) -> str:
    return (
        f"| {label} | `{result['used_backend']}` | {result['block_count_page']} | "
        f"{result['chunk_count_page']} | `{result['block_type_counts']}` | "
        f"`{result['route_backend_counts']}` |"
    )


def _blocks_markdown(blocks: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for block in blocks:
        lines.extend(
            [
                f"### `{block['block_id']}`",
                "",
                f"- Type: `{block['block_type']}`",
                f"- Source mode: `{block['source_mode']}`",
                f"- Reading order: `{block['reading_order']}`",
                f"- BBox: `{block['bbox']}`",
                f"- Trace meta: `{block['trace_meta']}`",
                "",
                "```text",
                block["text_preview"],
                "```",
                "",
            ]
        )
    if not lines:
        return ["No blocks on this page."]
    return lines


def _count(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _preview(text: str, max_chars: int) -> str:
    return (text or "").replace("\n", " ").strip()[:max_chars]


def _safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem)


if __name__ == "__main__":
    main()

