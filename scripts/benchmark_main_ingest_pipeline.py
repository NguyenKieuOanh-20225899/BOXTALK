from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ingest.probe import probe_pdf


DEFAULT_REGION_RESULT_DIR = Path("results/region_routing_ablation/qcdt_pubtables_ocr31")
DEFAULT_OCR_SCAN_SUMMARY = Path("results/ingest/ocr_scan_25_ocr_improve_after_aux_filter_20260517/summary.json")
DEFAULT_FUNSD_SUMMARY = Path("results/ingest/funsd_ocr_25_ocr_improve_after_aux_filter_20260517/summary.json")
DEFAULT_TABLE_SUMMARY = Path("results/ingest/pubtables_structure_100_hybrid_tatr_20260616/summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize evidence for the main ingest pipeline: "
            "probe -> region routing -> OCR/text/table -> block/chunk/metadata."
        )
    )
    parser.add_argument("--region-result-dir", type=Path, default=DEFAULT_REGION_RESULT_DIR)
    parser.add_argument("--ocr-scan-summary", type=Path, default=DEFAULT_OCR_SCAN_SUMMARY)
    parser.add_argument("--funsd-summary", type=Path, default=DEFAULT_FUNSD_SUMMARY)
    parser.add_argument("--table-summary", type=Path, default=DEFAULT_TABLE_SUMMARY)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/ingest/main_pipeline_ingest_evidence"),
        help="Output directory for summary.json, per_doc.jsonl, and README.md.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def case_expectation(pdf_path: str) -> dict[str, Any]:
    normalized = pdf_path.replace("\\", "/").lower()
    if "qcdt_2025_5445" in normalized:
        return {
            "case_type": "text_layer_with_tables",
            "expected_probe_modes": ["layout", "text", "mixed"],
            "expected_region_backend": "region_routed",
            "expected_routes": ["text", "table"],
            "expected_has_table": True,
            "expected_has_ocr": False,
        }
    if "ocr_scan_25" in normalized:
        return {
            "case_type": "scan_pdf",
            "expected_probe_modes": ["ocr"],
            "expected_region_backend": "ocr",
            "expected_routes": [],
            "expected_has_table": False,
            "expected_has_ocr": True,
        }
    if "pubtables_structure" in normalized:
        return {
            "case_type": "table_image_pdf",
            "expected_probe_modes": ["ocr"],
            "expected_region_backend": "ocr",
            "expected_routes": [],
            "expected_has_table": True,
            "expected_has_ocr": True,
        }
    return {
        "case_type": "unknown",
        "expected_probe_modes": [],
        "expected_region_backend": None,
        "expected_routes": [],
        "expected_has_table": None,
        "expected_has_ocr": None,
    }


def route_matches(expected_routes: list[str], route_counts: dict[str, int]) -> bool | None:
    if not expected_routes:
        return None
    observed = set(route_counts)
    if "table" in expected_routes:
        has_table_route = "table" in observed or "hybrid_tatr" in observed
    else:
        has_table_route = True
    has_other_routes = all(route in observed for route in expected_routes if route != "table")
    return has_table_route and has_other_routes


def sample_metadata_completeness(blocks: list[dict[str, Any]], *, require_trace: bool) -> dict[str, float]:
    total = len(blocks)
    if total == 0:
        return {
            "sample_block_count": 0,
            "page_id_rate": 0.0,
            "bbox_rate": 0.0,
            "block_type_rate": 0.0,
            "text_rate": 0.0,
            "route_trace_rate": 0.0 if require_trace else 1.0,
        }
    page_id = sum(1 for block in blocks if block.get("page_index") is not None)
    bbox = sum(1 for block in blocks if block.get("bbox"))
    block_type = sum(1 for block in blocks if block.get("block_type"))
    text = sum(1 for block in blocks if str(block.get("text_preview") or "").strip())
    trace = sum(1 for block in blocks if (block.get("trace") or {}).get("route_backend"))
    return {
        "sample_block_count": total,
        "page_id_rate": page_id / total,
        "bbox_rate": bbox / total,
        "block_type_rate": block_type / total,
        "text_rate": text / total,
        "route_trace_rate": (trace / total) if require_trace else 1.0,
    }


def nested_mean(summary: dict[str, Any], path: list[str], default: float | None = None) -> float | None:
    value: Any = summary
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    if isinstance(value, (int, float)):
        return float(value)
    return default


def summarize(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    region_summary = load_json(args.region_result_dir / "summary.json")
    rows = load_jsonl(args.region_result_dir / "per_doc.jsonl")

    per_doc: list[dict[str, Any]] = []
    probe_correct = 0
    probe_total = 0
    backend_correct = 0
    backend_total = 0
    route_doc_correct = 0
    route_doc_total = 0
    block_success_count = 0
    metadata_samples: list[dict[str, float]] = []
    fallback_triggered = 0
    fallback_triggered_success = 0
    case_counts = Counter()

    for row in rows:
        pdf = str(row["pdf"])
        expectation = case_expectation(pdf)
        case_counts[expectation["case_type"]] += 1
        region_on = row.get("region_on") or {}
        probe = probe_pdf(ROOT / pdf)

        expected_probe_modes = expectation["expected_probe_modes"]
        probe_ok = probe.probe_detected_mode in expected_probe_modes if expected_probe_modes else None
        if probe_ok is not None:
            probe_total += 1
            probe_correct += int(probe_ok)

        expected_backend = expectation["expected_region_backend"]
        backend_ok = region_on.get("used_backend") == expected_backend if expected_backend else None
        if backend_ok is not None:
            backend_total += 1
            backend_correct += int(backend_ok)

        route_ok = route_matches(expectation["expected_routes"], region_on.get("route_backend_counts") or {})
        if route_ok is not None:
            route_doc_total += 1
            route_doc_correct += int(route_ok)

        block_success = bool(region_on.get("success")) and int(region_on.get("block_count") or 0) > 0 and int(region_on.get("chunk_count") or 0) > 0
        block_success_count += int(block_success)

        require_trace = region_on.get("used_backend") == "region_routed"
        metadata = sample_metadata_completeness(list(region_on.get("sample_blocks") or []), require_trace=require_trace)
        metadata_samples.append(metadata)

        fallback_count = int(region_on.get("fallback_error_count") or 0)
        if fallback_count > 0:
            fallback_triggered += 1
            fallback_triggered_success += int(bool(region_on.get("success")))

        per_doc.append(
            {
                "pdf": pdf,
                "case_type": expectation["case_type"],
                "probe_detected_mode": probe.probe_detected_mode,
                "expected_probe_modes": expected_probe_modes,
                "probe_correct": probe_ok,
                "region_on_backend": region_on.get("used_backend"),
                "expected_region_backend": expected_backend,
                "backend_correct": backend_ok,
                "route_backend_counts": region_on.get("route_backend_counts") or {},
                "route_expectation_correct": route_ok,
                "success": bool(region_on.get("success")),
                "block_count": int(region_on.get("block_count") or 0),
                "chunk_count": int(region_on.get("chunk_count") or 0),
                "table_block_count": int(region_on.get("table_block_count") or 0),
                "ocr_block_count": int(region_on.get("ocr_block_count") or 0),
                "route_traced_block_count": int(region_on.get("route_traced_block_count") or 0),
                "fallback_error_count": fallback_count,
                "latency_sec": float(region_on.get("latency_sec") or 0.0),
                "sample_metadata_completeness": metadata,
            }
        )

    ocr_scan = load_json(args.ocr_scan_summary)
    funsd = load_json(args.funsd_summary)
    table = load_json(args.table_summary)

    metadata_mean = {
        key: mean([float(item[key]) for item in metadata_samples])
        for key in ["page_id_rate", "bbox_rate", "block_type_rate", "text_rate", "route_trace_rate"]
    }

    summary = {
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_results": {
            "region_result_dir": str(args.region_result_dir),
            "ocr_scan_summary": str(args.ocr_scan_summary),
            "funsd_summary": str(args.funsd_summary),
            "table_summary": str(args.table_summary),
        },
        "case_counts": dict(case_counts),
        "probe": {
            "accuracy": probe_correct / probe_total if probe_total else None,
            "correct": probe_correct,
            "total": probe_total,
        },
        "document_backend_routing": {
            "accuracy": backend_correct / backend_total if backend_total else None,
            "correct": backend_correct,
            "total": backend_total,
            "note": "Document-level check: expected top-level backend after probe/fallback.",
        },
        "region_routing": {
            "document_route_accuracy": route_doc_correct / route_doc_total if route_doc_total else None,
            "correct": route_doc_correct,
            "total": route_doc_total,
            "qcdt_region_type_counts": next(
                (item["region_on"].get("region_type_counts") for item in rows if "QCDT_2025_5445" in item["pdf"]),
                {},
            ),
            "qcdt_route_backend_counts": next(
                (item["region_on"].get("route_backend_counts") for item in rows if "QCDT_2025_5445" in item["pdf"]),
                {},
            ),
            "limitation": "No manually labeled region-level gold set; this is not per-region Precision/Recall/F1.",
        },
        "block_output": {
            "success_rate": block_success_count / len(rows) if rows else None,
            "block_count_mean": nested_mean(region_summary, ["region_on", "block_count_mean"]),
            "chunk_count_mean": nested_mean(region_summary, ["region_on", "chunk_count_mean"]),
            "table_block_count_mean": nested_mean(region_summary, ["region_on", "table_block_count_mean"]),
            "route_traced_block_count_mean": nested_mean(region_summary, ["region_on", "route_traced_block_count_mean"]),
            "sample_metadata_completeness": metadata_mean,
        },
        "ocr_backend": {
            "scan_token_f1": nested_mean(ocr_scan, ["metric_summary", "token_f1", "mean"]),
            "scan_cer": nested_mean(ocr_scan, ["metric_summary", "cer", "mean"]),
            "funsd_token_f1": nested_mean(funsd, ["metric_summary", "token_f1", "mean"]),
            "funsd_cer": nested_mean(funsd, ["metric_summary", "cer", "mean"]),
        },
        "table_backend": {
            "sample_count": int(table.get("num_samples") or 0),
            "cell_f1_iou50": nested_mean(table, ["metric_summary", "cell_f1_iou50", "mean"]),
            "structure_f1": nested_mean(table, ["metric_summary", "table_structure", "f1", "mean"]),
            "text_assignment_f1": nested_mean(table, ["metric_summary", "text_assignment_f1", "mean"]),
            "exact_csv": nested_mean(table, ["metric_summary", "table_exact_csv", "mean"]),
            "note": "Backend-level table reconstruction benchmark for the table backend called by ingest.",
        },
        "fallback": {
            "triggered_count": fallback_triggered,
            "triggered_success_count": fallback_triggered_success,
            "fallback_success_rate": fallback_triggered_success / fallback_triggered if fallback_triggered else None,
            "failed_document_count": int(region_summary.get("region_on", {}).get("error_count") or 0),
            "latency_sec_mean_region_off": nested_mean(region_summary, ["region_off", "latency_sec_mean"]),
            "latency_sec_mean_region_on": nested_mean(region_summary, ["region_on", "latency_sec_mean"]),
            "latency_sec_delta_mean": nested_mean(region_summary, ["delta", "latency_sec_delta_mean"]),
        },
    }
    return summary, per_doc


def render_readme(summary: dict[str, Any]) -> str:
    return f"""# Main Ingest Pipeline Benchmark

This report summarizes evidence for the main ingest pipeline:

`PDF -> probe -> region routing -> OCR/text/table -> block/chunk/metadata -> index`

## A. Probe

- Probe accuracy: `{summary['probe']['accuracy']:.3f}` ({summary['probe']['correct']} / {summary['probe']['total']})

## B. Region routing

- Document-level backend routing accuracy: `{summary['document_backend_routing']['accuracy']:.3f}`
- Region-route expectation accuracy: `{summary['region_routing']['document_route_accuracy']:.3f}`
- QCDT route backend counts: `{summary['region_routing']['qcdt_route_backend_counts']}`

## C. Block/chunk/metadata

- Block/chunk success rate: `{summary['block_output']['success_rate']:.3f}`
- Mean block count: `{summary['block_output']['block_count_mean']:.3f}`
- Mean chunk count: `{summary['block_output']['chunk_count_mean']:.3f}`
- Mean route-traced blocks: `{summary['block_output']['route_traced_block_count_mean']:.3f}`
- Sample metadata completeness: `{summary['block_output']['sample_metadata_completeness']}`

## OCR Backend

- Scan Token F1 / CER: `{summary['ocr_backend']['scan_token_f1']:.3f}` / `{summary['ocr_backend']['scan_cer']:.3f}`
- FUNSD Token F1 / CER: `{summary['ocr_backend']['funsd_token_f1']:.3f}` / `{summary['ocr_backend']['funsd_cer']:.3f}`

## Table Backend

- Samples: `{summary['table_backend']['sample_count']}`
- Cell F1@0.50: `{summary['table_backend']['cell_f1_iou50']:.3f}`
- Structure F1: `{summary['table_backend']['structure_f1']:.3f}`
- Text Assignment F1: `{summary['table_backend']['text_assignment_f1']:.3f}`
- Exact CSV: `{summary['table_backend']['exact_csv']:.3f}`

## D. Fallback

- Fallback triggered count: `{summary['fallback']['triggered_count']}`
- Fallback success rate: `{summary['fallback']['fallback_success_rate']:.3f}`
- Failed document count: `{summary['fallback']['failed_document_count']}`
- Latency mean OFF/ON/delta: `{summary['fallback']['latency_sec_mean_region_off']:.3f}s` / `{summary['fallback']['latency_sec_mean_region_on']:.3f}s` / `{summary['fallback']['latency_sec_delta_mean']:.3f}s`

## Limitation

This benchmark does not provide per-region Precision/Recall/F1 because the
current repository does not include a manually labeled region-level gold set for
the main region routing detector.
"""


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary, per_doc = summarize(args)
    (args.out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.out / "per_doc.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_doc:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out / "README.md").write_text(render_readme(summary), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
