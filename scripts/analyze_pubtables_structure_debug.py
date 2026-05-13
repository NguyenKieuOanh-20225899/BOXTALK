from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize and visualize PubTables structure benchmark errors")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("results/ingest/pubtables_structure_25_after_structure_fix"),
        help="Benchmark output directory containing per_sample.jsonl and optional table_debug/*.json",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/benchmarks/pubtables_structure"),
        help="PubTables structure subset root containing pubtables_structure_samples.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/ingest/pubtables_structure_debug"),
        help="Output directory for analysis artifacts",
    )
    parser.add_argument("--limit-visualizations", type=int, default=10, help="Number of worst samples to visualize")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(args.data_dir)
    records = _load_jsonl(args.run_dir / "per_sample.jsonl")

    samples = []
    for record in records:
        if not record.get("success"):
            continue
        doc_id = str(record.get("doc_id") or "")
        debug = _load_optional_json(args.run_dir / "table_debug" / f"{doc_id}.json")
        prediction = _load_optional_json(args.run_dir / "predictions" / f"{doc_id}.json")
        sample_info = manifest.get(doc_id, {})
        gt = (sample_info.get("ground_truth") or {}) if isinstance(sample_info, dict) else {}

        pred_cells = _cells_from_debug_or_prediction(debug, prediction)
        gt_cells = _gt_cells_from_debug_or_manifest(debug, gt)
        pred_row_count = _row_count(pred_cells)
        gt_row_count = _row_count(gt_cells)
        pred_col_count = _col_count(pred_cells)
        gt_col_count = _col_count(gt_cells)
        row_delta = pred_row_count - gt_row_count
        col_delta = pred_col_count - gt_col_count
        samples.append(
            {
                "doc_id": doc_id,
                "gt_row_count": gt_row_count,
                "pred_row_count": pred_row_count,
                "row_count_error": abs(row_delta),
                "row_error_direction": _direction(row_delta),
                "gt_col_count": gt_col_count,
                "pred_col_count": pred_col_count,
                "col_count_error": abs(col_delta),
                "col_error_direction": _direction(col_delta),
                "cell_f1_iou50": _nested_metric(record, "table_cell_iou50", "f1", "cell_f1_iou50"),
                "table_structure_f1": _nested_metric(record, "table_structure", "f1"),
                "table_detection_f1_iou50": _nested_metric(record, "table_detection_iou50", "micro_f1"),
                "pred_cell_count": len(pred_cells),
                "gt_cell_count": len(gt_cells),
                "matched_cell_count": int(record.get("matched_cell_count") or _debug_count(debug, "matched_cells")),
                "unmatched_pred_count": int(record.get("unmatched_pred_count") or _debug_count(debug, "unmatched_predicted")),
                "unmatched_gt_count": int(record.get("unmatched_gt_count") or _debug_count(debug, "unmatched_ground_truth")),
                "image_path": str(_resolve_data_path(args.data_dir, sample_info.get("image_path"))),
                "prediction_path": str(args.run_dir / "predictions" / f"{doc_id}.json"),
                "debug_path": str(args.run_dir / "table_debug" / f"{doc_id}.json"),
            }
        )

    worst = sorted(samples, key=lambda item: (item["cell_f1_iou50"], item["table_structure_f1"]))[:5]
    summary = {
        "run_dir": str(args.run_dir),
        "data_dir": str(args.data_dir),
        "sample_count": len(samples),
        "row_count_mae": _mean([item["row_count_error"] for item in samples]),
        "col_count_mae": _mean([item["col_count_error"] for item in samples]),
        "row_oversegmentation_count": sum(1 for item in samples if item["row_error_direction"] == "over"),
        "row_undersegmentation_count": sum(1 for item in samples if item["row_error_direction"] == "under"),
        "row_exact_count": sum(1 for item in samples if item["row_error_direction"] == "exact"),
        "col_oversegmentation_count": sum(1 for item in samples if item["col_error_direction"] == "over"),
        "col_undersegmentation_count": sum(1 for item in samples if item["col_error_direction"] == "under"),
        "col_exact_count": sum(1 for item in samples if item["col_error_direction"] == "exact"),
        "mean_cell_f1_iou50": _mean([item["cell_f1_iou50"] for item in samples]),
        "mean_table_structure_f1": _mean([item["table_structure_f1"] for item in samples]),
        "mean_table_detection_f1_iou50": _mean([item["table_detection_f1_iou50"] for item in samples]),
        "worst_samples": worst,
    }

    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "per_sample.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in samples),
        encoding="utf-8",
    )

    visualization_errors = _write_visualizations(
        samples,
        manifest,
        args.data_dir,
        args.run_dir,
        args.out / "visualizations",
        limit=args.limit_visualizations,
    )
    if visualization_errors:
        summary["visualization_warnings"] = visualization_errors
        (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    (args.out / "README.md").write_text(_render_markdown(summary), encoding="utf-8")
    print(json.dumps({"summary": str(args.out / "summary.json"), "readme": str(args.out / "README.md")}, ensure_ascii=False))


def _load_manifest(data_dir: Path) -> dict[str, dict[str, Any]]:
    path = data_dir / "pubtables_structure_samples.jsonl"
    if not path.exists():
        return {}
    return {str(item.get("doc_id")): item for item in _load_jsonl(path)}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _cells_from_debug_or_prediction(debug: dict[str, Any], prediction: dict[str, Any]) -> list[dict[str, Any]]:
    predicted = debug.get("predicted") if isinstance(debug, dict) else None
    if isinstance(predicted, dict) and isinstance(predicted.get("cells"), list):
        return list(predicted["cells"])
    return list(prediction.get("table_cells") or [])


def _gt_cells_from_debug_or_manifest(debug: dict[str, Any], gt: dict[str, Any]) -> list[dict[str, Any]]:
    expected = debug.get("ground_truth") if isinstance(debug, dict) else None
    if isinstance(expected, dict) and isinstance(expected.get("cells"), list):
        return list(expected["cells"])
    return list(gt.get("table_cells") or [])


def _row_count(cells: list[dict[str, Any]]) -> int:
    return max((int(cell.get("row", 0) or 0) + int(cell.get("row_span", 1) or 1) for cell in cells), default=0)


def _col_count(cells: list[dict[str, Any]]) -> int:
    return max((int(cell.get("col", 0) or 0) + int(cell.get("col_span", 1) or 1) for cell in cells), default=0)


def _direction(delta: int) -> str:
    if delta > 0:
        return "over"
    if delta < 0:
        return "under"
    return "exact"


def _nested_metric(record: dict[str, Any], parent: str, child: str, fallback: str | None = None) -> float:
    value = record.get(parent)
    if isinstance(value, dict) and value.get(child) is not None:
        return float(value[child])
    if fallback and record.get(fallback) is not None:
        return float(record[fallback])
    return 0.0


def _debug_count(debug: dict[str, Any], key: str) -> int:
    value = debug.get(key) if isinstance(debug, dict) else None
    return len(value) if isinstance(value, list) else 0


def _resolve_data_path(data_dir: Path, value: Any) -> Path:
    if not value:
        return Path()
    path = Path(str(value))
    return path if path.is_absolute() else data_dir / path


def _mean(values: list[float | int]) -> float:
    return float(mean(values)) if values else 0.0


def _write_visualizations(
    samples: list[dict[str, Any]],
    manifest: dict[str, dict[str, Any]],
    data_dir: Path,
    run_dir: Path,
    out_dir: Path,
    *,
    limit: int,
) -> list[str]:
    if limit <= 0:
        return []
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:  # pragma: no cover - depends on optional local dependency
        return [f"PIL not available: {exc}"]

    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    worst = sorted(samples, key=lambda item: (item["cell_f1_iou50"], item["table_structure_f1"]))[:limit]
    for sample in worst:
        doc_id = sample["doc_id"]
        manifest_item = manifest.get(doc_id, {})
        image_path = _resolve_data_path(data_dir, manifest_item.get("image_path"))
        if not image_path.exists():
            warnings.append(f"Missing image for {doc_id}: {image_path}")
            continue

        debug = _load_optional_json(run_dir / "table_debug" / f"{doc_id}.json")
        gt = (manifest_item.get("ground_truth") or {}) if isinstance(manifest_item, dict) else {}
        pred_cells = _cells_from_debug_or_prediction(debug, _load_optional_json(run_dir / "predictions" / f"{doc_id}.json"))
        gt_cells = _gt_cells_from_debug_or_manifest(debug, gt)

        with Image.open(image_path) as image:
            canvas = image.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        for region in gt.get("table_regions") or gt.get("layout_regions") or []:
            if region.get("label") == "table":
                _draw_rect(draw, region.get("bbox"), "yellow", width=3)
        for cell in gt_cells:
            _draw_rect(draw, cell.get("bbox"), "lime", width=1)
        for cell in pred_cells:
            _draw_rect(draw, cell.get("bbox"), "red", width=1)
        for band in _row_bands(pred_cells):
            draw.line([(band[0], band[1]), (band[2], band[1])], fill="orange", width=1)
            draw.line([(band[0], band[3]), (band[2], band[3])], fill="orange", width=1)
        for band in _col_bands(pred_cells):
            draw.line([(band[0], band[1]), (band[0], band[3])], fill="cyan", width=1)
            draw.line([(band[2], band[1]), (band[2], band[3])], fill="cyan", width=1)

        canvas.save(out_dir / f"{_safe_filename(doc_id)}.png")
    return warnings


def _draw_rect(draw: Any, bbox: Any, color: str, *, width: int) -> None:
    box = _bbox(bbox)
    if box is None:
        return
    draw.rectangle(box, outline=color, width=width)


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    x0, y0, x1, y1 = [float(item) for item in value[:4]]
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _row_bands(cells: list[dict[str, Any]]) -> list[tuple[float, float, float, float]]:
    by_row: dict[int, list[tuple[float, float, float, float]]] = {}
    for cell in cells:
        box = _bbox(cell.get("bbox"))
        if box is not None:
            by_row.setdefault(int(cell.get("row", 0) or 0), []).append(box)
    return [_union(boxes) for _, boxes in sorted(by_row.items()) if boxes]


def _col_bands(cells: list[dict[str, Any]]) -> list[tuple[float, float, float, float]]:
    by_col: dict[int, list[tuple[float, float, float, float]]] = {}
    for cell in cells:
        box = _bbox(cell.get("bbox"))
        if box is not None:
            by_col.setdefault(int(cell.get("col", 0) or 0), []).append(box)
    return [_union(boxes) for _, boxes in sorted(by_col.items()) if boxes]


def _union(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _safe_filename(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)


def _render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# PubTables Structure Debug",
        "",
        f"- Run: `{summary['run_dir']}`",
        f"- Samples: {summary['sample_count']}",
        f"- Mean table detection F1@0.50: {summary['mean_table_detection_f1_iou50']:.3f}",
        f"- Mean cell IoU@0.50 F1: {summary['mean_cell_f1_iou50']:.3f}",
        f"- Mean table structure F1: {summary['mean_table_structure_f1']:.3f}",
        f"- Row count MAE: {summary['row_count_mae']:.3f}",
        f"- Column count MAE: {summary['col_count_mae']:.3f}",
        "",
        "## Segmentation Error Summary",
        "",
        f"- Row over-segmentation: {summary['row_oversegmentation_count']}",
        f"- Row under-segmentation: {summary['row_undersegmentation_count']}",
        f"- Row exact count: {summary['row_exact_count']}",
        f"- Column over-segmentation: {summary['col_oversegmentation_count']}",
        f"- Column under-segmentation: {summary['col_undersegmentation_count']}",
        f"- Column exact count: {summary['col_exact_count']}",
        "",
        "## Worst Samples",
        "",
        "| doc_id | cell_f1_iou50 | structure_f1 | gt rows | pred rows | gt cols | pred cols | unmatched pred | unmatched gt |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["worst_samples"]:
        lines.append(
            "| {doc_id} | {cell_f1_iou50:.3f} | {table_structure_f1:.3f} | {gt_row_count} | {pred_row_count} | "
            "{gt_col_count} | {pred_col_count} | {unmatched_pred_count} | {unmatched_gt_count} |".format(**item)
        )
    lines.extend(
        [
            "",
            "Visualization legend: yellow = table bbox, green = ground-truth cells, red = predicted cells, orange = predicted row bands, cyan = predicted column bands.",
        ]
    )
    warnings = summary.get("visualization_warnings") or []
    if warnings:
        lines.extend(["", "## Visualization Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
