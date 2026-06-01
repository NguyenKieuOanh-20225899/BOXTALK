from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import fitz

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ingest.region.detector import detect_regions
from app.ingest.tatr_table_backend import predict_tables_from_image


def _json_write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _render_region(page: fitz.Page, bbox: list[float], output_path: Path, *, scale: float) -> None:
    rect = fitz.Rect(bbox)
    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=rect, alpha=False)
    pix.save(str(output_path))


def _rows_from_tatr_result(result: dict[str, Any]) -> list[list[str]]:
    cells = result.get("table_cells") or []
    if not cells:
        return []
    row_count = max(int(cell.get("row", 0)) + int(cell.get("row_span", 1) or 1) for cell in cells)
    col_count = max(int(cell.get("col", 0)) + int(cell.get("col_span", 1) or 1) for cell in cells)
    rows = [["" for _ in range(col_count)] for _ in range(row_count)]
    for cell in cells:
        row = int(cell.get("row", 0))
        col = int(cell.get("col", 0))
        rows[row][col] = str(cell.get("text") or "")
    return rows


def _markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return "_No rows predicted._"
    col_count = max((len(row) for row in rows), default=0)
    padded = [row + [""] * (col_count - len(row)) for row in rows]
    lines = [
        "| " + " | ".join(cell or " " for cell in padded[0]) + " |",
        "| " + " | ".join("---" for _ in range(col_count)) + " |",
    ]
    for row in padded[1:]:
        lines.append("| " + " | ".join(cell or " " for cell in row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TATR-only structure recognition on table regions of one PDF page.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--page", type=int, required=True, help="1-based physical PDF page.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = args.out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    with fitz.open(str(args.pdf)) as doc:
        if args.page < 1 or args.page > len(doc):
            raise ValueError(f"Page {args.page} out of range 1..{len(doc)}")
        page = doc[args.page - 1]
        regions = [region for region in detect_regions(page) if str(region.get("kind")) == "table"]

        table_results: list[dict[str, Any]] = []
        for index, region in enumerate(regions, start=1):
            bbox = [float(value) for value in region["bbox"]]
            crop_path = crops_dir / f"page{args.page}_table{index}.png"
            _render_region(page, bbox, crop_path, scale=args.scale)
            result = predict_tables_from_image(
                crop_path,
                text_boxes=None,
                device=args.device,
                backend_name="tatr",
                text_source="none",
            )
            rows = _rows_from_tatr_result(result)
            table_results.append(
                {
                    "table_index": index,
                    "page": args.page,
                    "bbox": bbox,
                    "crop_path": str(crop_path),
                    "backend": result.get("table_backend"),
                    "text_source": result.get("text_source"),
                    "warnings": result.get("warnings") or [],
                    "detected_table_count": len(result.get("table_regions") or []),
                    "cell_count": len(result.get("table_cells") or []),
                    "row_count": len(rows),
                    "col_count": max((len(row) for row in rows), default=0),
                    "non_empty_cell_count": sum(1 for row in rows for cell in row if cell.strip()),
                    "tatr_rows": result.get("tatr_rows") or [],
                    "tatr_columns": result.get("tatr_columns") or [],
                    "tatr_spanning_cells": result.get("tatr_spanning_cells") or [],
                    "table_cells": result.get("table_cells") or [],
                    "rows": rows,
                }
            )

    summary = {
        "pdf": str(args.pdf),
        "page": args.page,
        "mode": "tatr_only_geometry_no_text_boxes",
        "table_region_count": len(table_results),
        "tables": table_results,
    }
    _json_write(args.out_dir / "summary.json", summary)

    lines = [
        f"# TATR-only table preview: {args.pdf.name}, page {args.page}",
        "",
        "This run passes no OCR/PDF word boxes to TATR. It visualizes geometry-only table structure.",
        "",
        "| Table | Rows | Columns | Cells | Non-empty cells | Text source | Warnings |",
        "|---:|---:|---:|---:|---:|---|---|",
    ]
    for table in table_results:
        warnings = "; ".join(table["warnings"])
        lines.append(
            f"| {table['table_index']} | {table['row_count']} | {table['col_count']} | "
            f"{table['cell_count']} | {table['non_empty_cell_count']} | "
            f"`{table['text_source']}` | {warnings} |"
        )
    lines.append("")
    for table in table_results:
        lines.extend(
            [
                f"## Table {table['table_index']}",
                "",
                f"- BBox: `{table['bbox']}`",
                f"- Crop: `{table['crop_path']}`",
                f"- TATR rows predicted: `{len(table['tatr_rows'])}`",
                f"- TATR columns predicted: `{len(table['tatr_columns'])}`",
                f"- TATR spanning cells predicted: `{len(table['tatr_spanning_cells'])}`",
                "",
                _markdown_table(table["rows"]),
                "",
            ]
        )
    (args.out_dir / "preview.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote TATR-only preview to: {args.out_dir / 'preview.md'}")
    print(f"Wrote JSON summary to: {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
