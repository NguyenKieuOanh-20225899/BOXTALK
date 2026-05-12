from __future__ import annotations

import argparse
import csv
import json
import difflib
import statistics
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ingest.pipeline import ingest_pdf


DATA_ROOT = Path("data/ingest_layout_benchmark")
MANIFEST_PATH = DATA_ROOT / "manifest.json"
RESULTS_ROOT = Path("results/ingest_layout_quality")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark layout-aware ingest quality on a synthetic PDF suite.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--create-dataset", action="store_true")
    return parser.parse_args()


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}. Run scripts/create_ingest_layout_benchmark.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _norm(text: str) -> str:
    return " ".join((text or "").casefold().split())


def _contains(haystack: str, needle: str) -> bool:
    return _norm(needle) in _norm(haystack)


def _text_payload(report: dict[str, Any]) -> str:
    blocks = report.get("blocks", [])
    return "\n".join(str(getattr(block, "text", "") or "") for block in blocks)


def _table_payload(report: dict[str, Any]) -> str:
    blocks = report.get("blocks", [])
    chunks = report.get("chunks", [])
    parts: list[str] = []
    for block in blocks:
        if getattr(block, "block_type", "") == "table":
            parts.append(str(getattr(block, "text", "") or ""))
            parts.append(str(getattr(block, "markdown", "") or ""))
    for chunk in chunks:
        if "table" in list(getattr(chunk, "block_types", []) or []):
            parts.append(str(getattr(chunk, "text", "") or ""))
            parts.append(str(getattr(chunk, "markdown", "") or ""))
    return "\n".join(parts)


def _table_blocks(report: dict[str, Any]) -> list[Any]:
    return [
        block for block in report.get("blocks", [])
        if getattr(block, "block_type", "") == "table"
    ]


def _coverage(expected: list[str], payload: str) -> tuple[float, list[str]]:
    if not expected:
        return 1.0, []
    missing = [item for item in expected if not _contains(payload, item)]
    return (len(expected) - len(missing)) / len(expected), missing


def _edit_similarity(expected: str | None, actual: str) -> float:
    if not expected:
        return 1.0
    return difflib.SequenceMatcher(None, _norm(expected), _norm(actual)).ratio()


def _reading_order_score(expected_order: list[str], payload: str) -> tuple[float, list[str]]:
    if not expected_order:
        return 1.0, []
    folded_payload = _norm(payload)
    positions: list[int] = []
    missing: list[str] = []
    for item in expected_order:
        pos = folded_payload.find(_norm(item))
        if pos < 0:
            missing.append(item)
        positions.append(pos)

    present_positions = [pos for pos in positions if pos >= 0]
    if not present_positions:
        return 0.0, missing

    ordered_pairs = 0
    total_pairs = 0
    for i in range(len(present_positions)):
        for j in range(i + 1, len(present_positions)):
            total_pairs += 1
            if present_positions[i] <= present_positions[j]:
                ordered_pairs += 1

    order_score = ordered_pairs / total_pairs if total_pairs else 1.0
    coverage_score = (len(expected_order) - len(missing)) / len(expected_order)
    return statistics.mean([order_score, coverage_score]), missing


def _noise_score(forbidden_substrings: list[str], payload: str) -> tuple[float, list[str]]:
    if not forbidden_substrings:
        return 1.0, []
    found = [item for item in forbidden_substrings if _contains(payload, item)]
    return (len(forbidden_substrings) - len(found)) / len(forbidden_substrings), found


def _chunk_preservation_score(
    *,
    actual_chunk_count: int,
    actual_table_chunk_count: int,
    expected_min_chunk_count: int | None,
    expected_table_chunk_count: int | None,
) -> float:
    chunk_score = 1.0
    if expected_min_chunk_count is not None:
        chunk_score = 1.0 if actual_chunk_count >= expected_min_chunk_count else actual_chunk_count / max(expected_min_chunk_count, 1)

    table_chunk_score = 1.0
    if expected_table_chunk_count is not None:
        table_chunk_score = 1.0 if actual_table_chunk_count == expected_table_chunk_count else (
            min(actual_table_chunk_count, expected_table_chunk_count) / max(actual_table_chunk_count, expected_table_chunk_count, 1)
        )

    return statistics.mean([chunk_score, table_chunk_score])


def _block_type_metrics(report: dict[str, Any], expected_counts: dict[str, int]) -> dict[str, Any]:
    blocks = report.get("blocks", [])
    counts = Counter(str(getattr(block, "block_type", "")) for block in blocks)
    per_type: dict[str, Any] = {}
    recalls: list[float] = []
    for block_type, expected_count in expected_counts.items():
        actual = counts.get(block_type, 0)
        recall = min(actual / expected_count, 1.0) if expected_count else 1.0
        recalls.append(recall)
        per_type[block_type] = {
            "expected": expected_count,
            "actual": actual,
            "recall": recall,
        }
    return {
        "counts": dict(counts),
        "expected_recalls": per_type,
        "macro_expected_type_recall": statistics.mean(recalls) if recalls else 1.0,
    }


def _table_shape_metrics(report: dict[str, Any], expected_shape: dict[str, Any] | None) -> dict[str, Any]:
    if not expected_shape:
        return {
            "table_shape_score": 1.0,
            "best_table_shape": None,
        }

    expected_rows = int(expected_shape.get("rows") or 0)
    expected_cols = int(expected_shape.get("cols") or 0)
    expected_headers = [str(header) for header in expected_shape.get("headers", [])]
    best_score = 0.0
    best_shape: dict[str, Any] | None = None

    for block in _table_blocks(report):
        meta = getattr(block, "meta", {}) or {}
        actual_rows = int(meta.get("table_row_count") or 0)
        actual_cols = int(meta.get("table_col_count") or 0)
        actual_headers = [str(header) for header in meta.get("table_headers", [])]

        row_score = _exact_or_ratio(actual_rows, expected_rows)
        col_score = _exact_or_ratio(actual_cols, expected_cols)
        header_score, missing_headers = _coverage(expected_headers, " ".join(actual_headers))
        score = statistics.mean([row_score, col_score, header_score])
        if score > best_score:
            best_score = score
            best_shape = {
                "rows": actual_rows,
                "cols": actual_cols,
                "headers": actual_headers,
                "missing_headers": missing_headers,
                "table_backend": meta.get("table_backend") or meta.get("backend"),
            }

    return {
        "table_shape_score": best_score,
        "best_table_shape": best_shape,
    }


def _exact_or_ratio(actual: int, expected: int) -> float:
    if expected <= 0:
        return 1.0
    if actual == expected:
        return 1.0
    if actual <= 0:
        return 0.0
    return min(actual, expected) / max(actual, expected)


def evaluate_document(row: dict[str, Any], manifest_dir: Path) -> dict[str, Any]:
    pdf_path = manifest_dir / row["file"]
    started = time.perf_counter()
    report = ingest_pdf(pdf_path)
    elapsed = time.perf_counter() - started

    payload = _text_payload(report)
    table_payload = _table_payload(report)
    substring_coverage, missing_substrings = _coverage(row.get("expected_substrings", []), payload)
    edit_similarity = _edit_similarity(row.get("expected_full_text"), payload)
    reading_order_score, missing_order_items = _reading_order_score(row.get("expected_order", []), payload)
    noise_score, found_forbidden_substrings = _noise_score(row.get("forbidden_substrings", []), payload)
    table_cell_coverage, missing_table_cells = _coverage(row.get("expected_table_cells", []), table_payload)
    block_metrics = _block_type_metrics(report, row.get("expected_block_types", {}))
    table_shape_metrics = _table_shape_metrics(report, row.get("expected_table_shape"))

    probe = report.get("probe", {})
    chunks = report.get("chunks", [])
    table_chunks = [
        chunk for chunk in chunks
        if "table" in list(getattr(chunk, "block_types", []) or [])
        or bool((getattr(chunk, "meta", {}) or {}).get("is_table_chunk"))
    ]
    expected_probe_mode = row.get("expected_probe_mode")
    probe_mode = probe.get("probe_detected_mode")
    chunk_preservation_score = _chunk_preservation_score(
        actual_chunk_count=len(chunks),
        actual_table_chunk_count=len(table_chunks),
        expected_min_chunk_count=row.get("expected_min_chunk_count"),
        expected_table_chunk_count=row.get("expected_table_chunk_count"),
    )

    return {
        "id": row["id"],
        "file": row["file"],
        "success": 1,
        "elapsed_sec": elapsed,
        "used_backend": report.get("used_backend"),
        "probe_mode": probe_mode,
        "probe_expected": expected_probe_mode,
        "probe_match": int(probe_mode == expected_probe_mode) if expected_probe_mode else 1,
        "page_count": len(report.get("pages", [])),
        "block_count": len(report.get("blocks", [])),
        "chunk_count": len(chunks),
        "table_chunk_count": len(table_chunks),
        "substring_coverage": substring_coverage,
        "missing_substrings": missing_substrings,
        "edit_similarity": edit_similarity,
        "reading_order_score": reading_order_score,
        "missing_order_items": missing_order_items,
        "noise_score": noise_score,
        "found_forbidden_substrings": found_forbidden_substrings,
        "chunk_preservation_score": chunk_preservation_score,
        "table_cell_coverage": table_cell_coverage,
        "missing_table_cells": missing_table_cells,
        "table_shape_score": table_shape_metrics["table_shape_score"],
        "best_table_shape": table_shape_metrics["best_table_shape"],
        "macro_expected_type_recall": block_metrics["macro_expected_type_recall"],
        "block_type_counts": block_metrics["counts"],
        "expected_type_recalls": block_metrics["expected_recalls"],
        "quality_score": statistics.mean(
            [
                substring_coverage,
                edit_similarity,
                reading_order_score,
                noise_score,
                chunk_preservation_score,
                table_cell_coverage,
                table_shape_metrics["table_shape_score"],
                block_metrics["macro_expected_type_recall"],
            ]
        ),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    success_records = [record for record in records if int(record.get("success", 0)) == 1]
    if not success_records:
        return {
            "documents": len(records),
            "success_rate": 0.0,
        }
    latencies = sorted(float(r["elapsed_sec"]) for r in success_records)
    return {
        "documents": len(records),
        "success_rate": len(success_records) / len(records),
        "quality_score_mean": statistics.mean(float(r["quality_score"]) for r in success_records),
        "substring_coverage_mean": statistics.mean(float(r["substring_coverage"]) for r in success_records),
        "edit_similarity_mean": statistics.mean(float(r["edit_similarity"]) for r in success_records),
        "reading_order_score_mean": statistics.mean(float(r["reading_order_score"]) for r in success_records),
        "noise_score_mean": statistics.mean(float(r["noise_score"]) for r in success_records),
        "chunk_preservation_score_mean": statistics.mean(float(r["chunk_preservation_score"]) for r in success_records),
        "table_cell_coverage_mean": statistics.mean(float(r["table_cell_coverage"]) for r in success_records),
        "table_shape_score_mean": statistics.mean(float(r["table_shape_score"]) for r in success_records),
        "macro_expected_type_recall_mean": statistics.mean(float(r["macro_expected_type_recall"]) for r in success_records),
        "probe_match_rate": statistics.mean(float(r["probe_match"]) for r in success_records),
        "latency_mean_sec": statistics.mean(latencies),
        "latency_p95_sec": _percentile(latencies, 0.95),
        "backend_counts": dict(Counter(str(r.get("used_backend")) for r in success_records)),
    }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * q
    low = int(rank)
    high = min(low + 1, len(values) - 1)
    weight = rank - low
    return values[low] * (1.0 - weight) + values[high] * weight


def _save_csv(path: Path, records: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        for key in (
            "missing_substrings",
            "missing_order_items",
            "found_forbidden_substrings",
            "missing_table_cells",
            "best_table_shape",
            "block_type_counts",
            "expected_type_recalls",
        ):
            row[key] = json.dumps(row.get(key), ensure_ascii=False, sort_keys=True)
        rows.append(row)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(ROOT),
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _render_markdown(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    lines = [
        "# Ingest Layout Quality Benchmark",
        "",
        f"- Documents: {summary.get('documents', 0)}",
        f"- Success rate: {summary.get('success_rate', 0.0):.3f}",
        f"- Quality score mean: {summary.get('quality_score_mean', 0.0):.3f}",
        f"- Substring coverage mean: {summary.get('substring_coverage_mean', 0.0):.3f}",
        f"- Edit similarity mean: {summary.get('edit_similarity_mean', 0.0):.3f}",
        f"- Reading order score mean: {summary.get('reading_order_score_mean', 0.0):.3f}",
        f"- Noise score mean: {summary.get('noise_score_mean', 0.0):.3f}",
        f"- Chunk preservation mean: {summary.get('chunk_preservation_score_mean', 0.0):.3f}",
        f"- Table cell coverage mean: {summary.get('table_cell_coverage_mean', 0.0):.3f}",
        f"- Table shape score mean: {summary.get('table_shape_score_mean', 0.0):.3f}",
        f"- Block type recall mean: {summary.get('macro_expected_type_recall_mean', 0.0):.3f}",
        f"- Probe match rate: {summary.get('probe_match_rate', 0.0):.3f}",
        f"- Mean latency: {summary.get('latency_mean_sec', 0.0):.3f}s",
        f"- Backend counts: `{json.dumps(summary.get('backend_counts', {}), ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Per Document",
        "",
        "| Document | Quality | Text | Edit | Order | Noise | Chunks | Table cells | Table shape | Type recall | Backend | Missing |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for record in records:
        missing = (
            len(record.get("missing_substrings", []))
            + len(record.get("missing_order_items", []))
            + len(record.get("missing_table_cells", []))
            + len(record.get("found_forbidden_substrings", []))
        )
        lines.append(
            "| {id} | {quality_score:.3f} | {substring_coverage:.3f} | {edit_similarity:.3f} | "
            "{reading_order_score:.3f} | {noise_score:.3f} | {chunk_preservation_score:.3f} | "
            "{table_cell_coverage:.3f} | {table_shape_score:.3f} | {macro_expected_type_recall:.3f} | "
            "{used_backend} | {missing} |".format(
                missing=missing,
                **record,
            )
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.create_dataset:
        from scripts.create_ingest_layout_benchmark import main as create_dataset

        create_dataset()

    manifest = _load_manifest(args.manifest)
    output_dir = args.output_dir or RESULTS_ROOT / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for row in manifest.get("documents", []):
        try:
            records.append(evaluate_document(row, args.manifest.parent))
        except Exception as exc:
            records.append(
                {
                    "id": row.get("id"),
                    "file": row.get("file"),
                    "success": 0,
                    "error": str(exc),
                }
            )

    summary = summarize(records)
    payload = {
        "metadata": {
            "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": _git_commit(),
            "manifest": str(args.manifest),
        },
        "summary": summary,
        "records": records,
    }
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(_render_markdown(summary, records), encoding="utf-8")
    _save_csv(output_dir / "per_document.csv", records)
    print(output_dir)


if __name__ == "__main__":
    main()
