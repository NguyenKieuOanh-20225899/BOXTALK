from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from html import escape
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DATASET_ID = "docling-project/PubTables-1M_OTSL-v1.1"
ROWS_API = "https://datasets-server.huggingface.co/rows"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a small PubTables-1M OTSL structure subset")
    parser.add_argument("--out", type=Path, default=Path("data/benchmarks/pubtables_structure"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    image_dir = args.out / "images"
    pdf_dir = args.out / "pdfs"
    image_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    rows = _fetch_rows(args.dataset_id, args.split, args.offset, args.limit)
    manifest_path = args.out / "pubtables_structure_samples.jsonl"
    records: list[dict[str, Any]] = []
    for item in rows:
        row = item["row"]
        filename = Path(str(row["filename"])).name
        stem = Path(filename).stem
        image_path = image_dir / filename
        pdf_path = pdf_dir / f"{stem}.pdf"
        _download_image(row["image"]["src"], image_path)
        _image_to_pdf(image_path, pdf_path)

        gt_rows, gt_cells, word_boxes = _parse_cells(row.get("cells") or [])
        table_region = {
            "label": "table",
            "bbox": [float(value) for value in row.get("table_bbox", [0, 0, row["image"]["width"], row["image"]["height"]])[:4]],
            "text": "\n".join(" | ".join(c for c in values if c) for values in gt_rows),
        }
        records.append(
            {
                "doc_id": stem,
                "pdf_path": _relative(args.out, pdf_path),
                "image_path": _relative(args.out, image_path),
                "ground_truth": {
                    "text": "\n".join(" ".join(c for c in values if c) for values in gt_rows),
                    "ordered_text": [" ".join(c for c in values if c) for values in gt_rows],
                    "layout_regions": [table_region],
                    "table_regions": [table_region],
                    "table_cells": gt_cells,
                    "table_csv": _rows_to_csv(gt_rows),
                    "table_html": "".join(row.get("html_with_text") or []) or _rows_to_html(gt_rows),
                },
                "metadata": {
                    "benchmark": "pubtables_structure",
                    "source": args.dataset_id,
                    "split": args.split,
                    "offset": args.offset,
                    "rows": row.get("rows"),
                    "cols": row.get("cols"),
                    "filename": filename,
                    "word_box_source": "pubtables_cell_tokens_proxy",
                    "word_box_count": len(word_boxes),
                },
                "word_boxes": word_boxes,
            }
        )

    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "dataset": "pubtables_structure",
        "source": args.dataset_id,
        "split": args.split,
        "offset": args.offset,
        "limit": args.limit,
        "sample_count": len(records),
        "manifest": str(manifest_path),
    }
    (args.out / "README.md").write_text(render_readme(summary), encoding="utf-8")
    (args.out / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(manifest_path)


def _fetch_rows(dataset_id: str, split: str, offset: int, limit: int) -> list[dict[str, Any]]:
    import requests

    response = requests.get(
        ROWS_API,
        params={
            "dataset": dataset_id,
            "config": "default",
            "split": split,
            "offset": offset,
            "length": limit,
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    return list(payload.get("rows", []) or [])


def _download_image(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    import requests

    response = requests.get(url, timeout=120)
    response.raise_for_status()
    path.write_bytes(response.content)


def _image_to_pdf(image_path: Path, pdf_path: Path) -> None:
    if pdf_path.exists() and pdf_path.stat().st_size > 0:
        return
    import fitz
    from PIL import Image

    with Image.open(image_path) as image:
        width, height = image.size
    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    page.insert_image(fitz.Rect(0, 0, width, height), filename=str(image_path))
    doc.save(pdf_path)
    doc.close()


def _parse_cells(raw_rows: list[Any]) -> tuple[list[list[str]], list[dict[str, Any]], list[dict[str, Any]]]:
    flat_cells: list[dict[str, Any]] = []
    for item in raw_rows:
        if isinstance(item, dict):
            flat_cells.append(item)
        elif isinstance(item, list):
            flat_cells.extend(cell for cell in item if isinstance(cell, dict))

    positioned: list[dict[str, Any]] = []
    word_boxes: list[dict[str, Any]] = []
    for raw_cell in flat_cells:
        raw_tokens = raw_cell.get("tokens") or []
        text = " ".join("".join(raw_tokens).split())
        bbox = raw_cell.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x0, y0, x1, y1 = [float(value) for value in bbox[:4]]
        if x1 <= x0 or y1 <= y0:
            continue
        cell_bbox = [x0, y0, x1, y1]
        positioned.append({"text": text, "bbox": cell_bbox, "x0": x0, "y0": y0, "x1": x1, "y1": y1})
        word_boxes.extend(_word_boxes_from_cell_tokens(raw_tokens, cell_bbox))

    if not positioned:
        return [], [], []

    heights = [max(1.0, cell["y1"] - cell["y0"]) for cell in positioned]
    row_tolerance = max(4.0, sorted(heights)[len(heights) // 2] * 0.75)
    row_groups: list[list[dict[str, Any]]] = []
    for cell in sorted(positioned, key=lambda item: ((item["y0"] + item["y1"]) / 2.0, item["x0"])):
        y_mid = (cell["y0"] + cell["y1"]) / 2.0
        for group in row_groups:
            group_mid = sum((item["y0"] + item["y1"]) / 2.0 for item in group) / len(group)
            if abs(y_mid - group_mid) <= row_tolerance:
                group.append(cell)
                break
        else:
            row_groups.append([cell])

    anchors = _infer_column_anchors_from_cells(positioned)
    rows: list[list[str]] = []
    cells: list[dict[str, Any]] = []
    for row_index, group in enumerate(row_groups):
        row_values = [""] * max(len(anchors), 1)
        for cell in sorted(group, key=lambda item: item["x0"]):
            col_index = _nearest_anchor(cell["x0"], anchors) if anchors else 0
            if col_index >= len(row_values):
                row_values.extend("" for _ in range(col_index - len(row_values) + 1))
            row_values[col_index] = f"{row_values[col_index]} {cell['text']}".strip() if row_values[col_index] else cell["text"]
            cells.append(
                {
                    "row": row_index,
                    "col": col_index,
                    "text": cell["text"],
                    "is_header": row_index == 0,
                    "bbox": cell["bbox"],
                }
            )
        rows.append(row_values)
    return rows, cells, word_boxes


def _word_boxes_from_cell_tokens(raw_tokens: list[Any], bbox: list[float]) -> list[dict[str, Any]]:
    text = " ".join("".join(str(token) for token in raw_tokens).split())
    if not text:
        return []
    tokens = text.split()
    if not tokens:
        return []
    x0, y0, x1, y1 = bbox
    usable_width = max(1.0, x1 - x0)
    total_chars = sum(max(len(token), 1) for token in tokens)
    cursor = x0
    words: list[dict[str, Any]] = []
    for index, token in enumerate(tokens):
        if index == len(tokens) - 1:
            next_cursor = x1
        else:
            next_cursor = cursor + usable_width * (max(len(token), 1) / max(total_chars, 1))
        words.append(
            {
                "text": token,
                "bbox": [cursor, y0, next_cursor, y1],
                "confidence": 1.0,
                "source": "pubtables_cell_tokens_proxy",
            }
        )
        cursor = next_cursor
    return words


def _infer_column_anchors_from_cells(cells: list[dict[str, Any]]) -> list[float]:
    anchors: list[dict[str, float]] = []
    tolerance = 8.0
    for cell in sorted(cells, key=lambda item: item["x0"]):
        matched = None
        for anchor in anchors:
            if abs(cell["x0"] - anchor["x0"]) <= tolerance:
                matched = anchor
                break
        if matched is None:
            anchors.append({"x0": cell["x0"], "count": 1.0})
        else:
            matched["x0"] = (matched["x0"] * matched["count"] + cell["x0"]) / (matched["count"] + 1.0)
            matched["count"] += 1.0
    frequent = [anchor["x0"] for anchor in anchors if anchor["count"] >= 2]
    return sorted(frequent or [anchor["x0"] for anchor in anchors])


def _nearest_anchor(x0: float, anchors: list[float]) -> int:
    return min(range(len(anchors)), key=lambda index: abs(x0 - anchors[index]))


def _rows_to_csv(rows: list[list[str]]) -> str:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows(rows)
    return output.getvalue().strip()


def _rows_to_html(rows: list[list[str]]) -> str:
    lines = ["<table>"]
    for row_index, row in enumerate(rows):
        tag = "th" if row_index == 0 and len(rows) > 1 else "td"
        lines.append("  <tr>" + "".join(f"<{tag}>{escape(cell)}</{tag}>" for cell in row) + "</tr>")
    lines.append("</table>")
    return "\n".join(lines)


def _relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def render_readme(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# PubTables Structure Subset",
            "",
            f"- Source: `{summary['source']}`",
            f"- Split: `{summary['split']}`",
            f"- Offset: `{summary['offset']}`",
            f"- Samples: `{summary['sample_count']}`",
            f"- Manifest: `{summary['manifest']}`",
            "",
            "Run:",
            "",
            "```powershell",
            "python scripts/benchmark_ingest_suite.py --dataset pubtables_structure --data-dir data/benchmarks/pubtables_structure --limit 25 --out results/ingest/pubtables_structure_25 --mode table --save-predictions",
            "```",
        ]
    )


if __name__ == "__main__":
    main()
