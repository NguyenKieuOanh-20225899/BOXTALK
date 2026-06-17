from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from PIL import Image


METRICS = [
    "table_exact_csv",
    "cell_f1_iou50",
    "text_assignment_f1",
    "table_structure_f1",
    "latency_sec",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Group PubTables structure samples by table complexity and summarize backend metrics."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--result", action="append", required=True, help="backend=path/to/per_sample.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_samples(args.data_dir / "pubtables_structure_samples.jsonl", args.data_dir)
    backend_rows = {}
    for item in args.result:
        if "=" not in item:
            raise SystemExit(f"Invalid --result {item!r}; expected backend=path")
        backend, raw_path = item.split("=", 1)
        backend_rows[backend] = load_results(Path(raw_path))

    grouped = defaultdict(list)
    sample_groups = {}
    for doc_id, sample in samples.items():
        groups = classify_sample(sample)
        sample_groups[doc_id] = groups
        for group in groups:
            grouped[group].append(doc_id)

    summary = {
        "sample_count": len(samples),
        "groups": {},
        "backends": {},
    }
    for group, doc_ids in sorted(grouped.items()):
        summary["groups"][group] = {
            "sample_count": len(doc_ids),
            "doc_ids": doc_ids,
        }

    for backend, rows_by_id in backend_rows.items():
        backend_summary = {}
        for group, doc_ids in sorted(grouped.items()):
            rows = [rows_by_id[doc_id] for doc_id in doc_ids if doc_id in rows_by_id]
            backend_summary[group] = summarize_rows(rows)
        backend_summary["all"] = summarize_rows(list(rows_by_id.values()))
        summary["backends"][backend] = backend_summary

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "README.md").write_text(render_markdown(summary), encoding="utf-8")
    print(args.out)


def load_samples(path: Path, data_dir: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row["_image_abs_path"] = str(data_dir / row.get("image_path", ""))
            rows[str(row["doc_id"])] = row
    return rows


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            metrics = dict(row)
            structure = row.get("table_structure") or {}
            metrics["table_structure_f1"] = structure.get("f1", row.get("table_structure_f1"))
            rows[str(row["doc_id"])] = metrics
    return rows


def classify_sample(sample: dict[str, Any]) -> list[str]:
    gt = sample.get("ground_truth") or {}
    cells = gt.get("table_cells") or []
    html = str(gt.get("table_html") or "").lower()
    header_rows = {
        int(cell.get("row", -1))
        for cell in cells
        if cell.get("is_header") and cell.get("row") is not None
    }
    thead_match = re.search(r"<thead>(.*?)</thead>", html, flags=re.IGNORECASE | re.DOTALL)
    thead_row_count = len(re.findall(r"<tr", thead_match.group(1), flags=re.IGNORECASE)) if thead_match else 0
    has_merged = "colspan" in html or "rowspan" in html
    has_multi_header = len(header_rows) > 1 or thead_row_count > 1
    no_grid = not has_grid_lines(Path(sample["_image_abs_path"]))

    groups = []
    if not has_merged and not has_multi_header and not no_grid:
        groups.append("simple")
    if has_merged:
        groups.append("merged_cell")
    if has_multi_header:
        groups.append("multi_row_header")
    if no_grid:
        groups.append("no_grid_lines")
    if not groups:
        groups.append("other")
    return groups


def has_grid_lines(image_path: Path) -> bool:
    if not image_path.exists():
        return False
    image = Image.open(image_path).convert("L")
    arr = np.asarray(image)
    if arr.size == 0:
        return False
    dark = arr < 80
    height, width = dark.shape
    horizontal = int(np.sum(np.mean(dark, axis=1) > 0.45))
    vertical = int(np.sum(np.mean(dark, axis=0) > 0.45))
    return horizontal >= 2 or vertical >= 2


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"sample_count": 0}
    output: dict[str, Any] = {"sample_count": len(rows)}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        output[metric] = mean(values) if values else None
    return output


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# PubTables Structure Group Analysis",
        "",
        f"- Samples: {summary['sample_count']}",
        "",
        "## Group Counts",
        "",
        "| Group | Samples |",
        "|---|---:|",
    ]
    for group, info in summary["groups"].items():
        lines.append(f"| {group} | {info['sample_count']} |")

    lines.extend(["", "## Backend Metrics", ""])
    for backend, groups in summary["backends"].items():
        lines.extend(
            [
                f"### {backend}",
                "",
                "| Group | N | Exact CSV | Cell F1@0.50 | Structure F1 | Text Assignment F1 | Latency s |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        ordered = ["all", "simple", "merged_cell", "multi_row_header", "no_grid_lines", "other"]
        for group in ordered:
            if group not in groups:
                continue
            row = groups[group]
            lines.append(
                f"| {group} | {row.get('sample_count', 0)} | "
                f"{fmt(row.get('table_exact_csv'))} | "
                f"{fmt(row.get('cell_f1_iou50'))} | "
                f"{fmt(row.get('table_structure_f1'))} | "
                f"{fmt(row.get('text_assignment_f1'))} | "
                f"{fmt(row.get('latency_sec'))} |"
            )
        lines.append("")
    return "\n".join(lines)


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
