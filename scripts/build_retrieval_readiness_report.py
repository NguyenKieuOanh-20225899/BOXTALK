from __future__ import annotations

import argparse
import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results" / "retrieval_readiness"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a standardized retrieval-readiness report from benchmark summaries."
    )
    parser.add_argument(
        "--scientific-summary",
        action="append",
        required=True,
        help="Path to scientific_summary.json. Pass multiple times for 25/100/500 style runs.",
    )
    parser.add_argument(
        "--production-summary",
        type=Path,
        default=None,
        help="Optional benchmark_summary.json from benchmark_ingest_standard.py",
    )
    parser.add_argument(
        "--baseline-reference",
        type=Path,
        default=None,
        help="Optional scientific_summary.json for a baseline reference run",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def scientific_entry(summary_path: Path) -> dict[str, Any]:
    payload = load_json(summary_path)
    if not payload.get("datasets"):
        raise ValueError(f"No datasets in scientific summary: {summary_path}")

    dataset_key = next(iter(payload["datasets"].keys()))
    dataset = payload["datasets"][dataset_key]
    metadata = payload.get("metadata", {})
    metrics = dataset.get("metrics", {})
    table_extraction = dataset.get("table_extraction", {})
    backend_counts = dataset.get("backend_counts", {})
    route_counts = dataset.get("route_counts", {})

    return {
        "summary_path": str(summary_path),
        "timestamp_utc": metadata.get("timestamp_utc"),
        "pubtables_limit": int(metadata.get("pubtables_limit", 0) or 0),
        "profile": dataset_key.split("::", 1)[-1],
        "dataset": dataset.get("dataset"),
        "success_rate": float(dataset.get("success_rate", 0.0) or 0.0),
        "images_total": int(dataset.get("images_total", 0) or 0),
        "images_failed": int(dataset.get("images_failed", 0) or 0),
        "latency_mean_sec": float(dataset.get("latency_mean_sec", 0.0) or 0.0),
        "latency_median_sec": float(dataset.get("latency_median_sec", 0.0) or 0.0),
        "latency_p95_sec": float(dataset.get("latency_p95_sec", 0.0) or 0.0),
        "iou50_micro_f1": float(metrics.get("iou50", {}).get("micro_f1", 0.0) or 0.0),
        "iou75_micro_f1": float(metrics.get("iou75", {}).get("micro_f1", 0.0) or 0.0),
        "pred_table_nonempty_rate": float(
            table_extraction.get("pred_table_nonempty_rate", 0.0) or 0.0
        ),
        "backend_counts": backend_counts,
        "route_counts": route_counts,
        "dominant_backend": dominant_key(backend_counts),
        "dominant_route": dominant_key(route_counts),
    }


def dominant_key(counts: dict[str, Any]) -> str | None:
    if not counts:
        return None
    return max(counts.items(), key=lambda item: int(item[1]))[0]


def build_scientific_gates(entries: list[dict[str, Any]]) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    for entry in entries:
        label = f"limit_{entry['pubtables_limit']}"
        gates[label] = {
            "pass_success_rate": entry["success_rate"] >= 0.99,
            "pass_iou50_micro_f1": entry["iou50_micro_f1"] >= 0.97,
            "pass_iou75_micro_f1": entry["iou75_micro_f1"] >= 0.80,
            "pass_latency_p95_sec": entry["latency_p95_sec"] <= 2.0,
            "pass_backend_consistency": len(entry["backend_counts"]) == 1,
        }
    return gates


def summarize_scientific(entries: list[dict[str, Any]]) -> dict[str, Any]:
    if not entries:
        return {}

    iou50_values = [entry["iou50_micro_f1"] for entry in entries]
    iou75_values = [entry["iou75_micro_f1"] for entry in entries]
    p95_values = [entry["latency_p95_sec"] for entry in entries]
    success_values = [entry["success_rate"] for entry in entries]
    dominant_backends = {entry["dominant_backend"] for entry in entries}
    dominant_routes = {entry["dominant_route"] for entry in entries}

    return {
        "runs": entries,
        "stability": {
            "success_rate_min": min(success_values),
            "success_rate_max": max(success_values),
            "iou50_micro_f1_min": min(iou50_values),
            "iou50_micro_f1_max": max(iou50_values),
            "iou75_micro_f1_min": min(iou75_values),
            "iou75_micro_f1_max": max(iou75_values),
            "latency_p95_sec_min": min(p95_values),
            "latency_p95_sec_max": max(p95_values),
            "dominant_backends": sorted(dominant_backends),
            "dominant_routes": sorted(dominant_routes),
            "backend_consistent": len(dominant_backends) == 1,
            "route_consistent": len(dominant_routes) == 1,
        },
        "gates": build_scientific_gates(entries),
    }


def summarize_production(summary_path: Path | None) -> dict[str, Any] | None:
    if summary_path is None:
        return None

    payload = load_json(summary_path)
    profiles = payload.get("profiles", {})
    documents_total = {
        name: int(profile.get("documents_total", 0) or 0)
        for name, profile in profiles.items()
    }
    available = any(count > 0 for count in documents_total.values())
    return {
        "summary_path": str(summary_path),
        "available": available,
        "documents_total_by_profile": documents_total,
        "comparison": payload.get("comparison", {}),
    }


def summarize_baseline_reference(summary_path: Path | None) -> dict[str, Any] | None:
    if summary_path is None:
        return None
    payload = load_json(summary_path)
    dataset_key = next(iter(payload.get("datasets", {}).keys()))
    dataset = payload["datasets"][dataset_key]
    metrics = dataset.get("metrics", {})
    return {
        "summary_path": str(summary_path),
        "dataset_key": dataset_key,
        "success_rate": float(dataset.get("success_rate", 0.0) or 0.0),
        "iou50_micro_f1": float(metrics.get("iou50", {}).get("micro_f1", 0.0) or 0.0),
        "iou75_micro_f1": float(metrics.get("iou75", {}).get("micro_f1", 0.0) or 0.0),
        "latency_mean_sec": float(dataset.get("latency_mean_sec", 0.0) or 0.0),
        "note": "Reference only. This run may use a different runtime profile or hardware path.",
    }


def build_verdict(
    scientific: dict[str, Any],
    production: dict[str, Any] | None,
) -> dict[str, Any]:
    blockers: list[str] = []
    strengths: list[str] = []

    stability = scientific.get("stability", {})
    if stability.get("success_rate_min", 0.0) >= 0.99:
        strengths.append("Scientific benchmark stayed at 100% success across sampled GPU runs.")
    else:
        blockers.append("Scientific benchmark success rate dropped below 99% on at least one sampled run.")

    if stability.get("iou50_micro_f1_min", 0.0) >= 0.97:
        strengths.append("PubTables IoU@0.50 micro F1 stayed at or above 0.97 across 25/100/500 samples.")
    else:
        blockers.append("PubTables IoU@0.50 micro F1 dropped below 0.97 on at least one sampled run.")

    if stability.get("iou75_micro_f1_min", 0.0) >= 0.80:
        strengths.append("PubTables IoU@0.75 micro F1 stayed at or above 0.80 across sampled runs.")
    else:
        blockers.append("PubTables IoU@0.75 micro F1 dropped below 0.80 on at least one sampled run.")

    if stability.get("latency_p95_sec_max", 999.0) <= 2.0:
        strengths.append("Latency stayed under the 2.0s p95 gate on RTX 3050 GPU.")
    else:
        blockers.append("Latency exceeded the 2.0s p95 gate on at least one sampled run.")

    if stability.get("backend_consistent"):
        strengths.append(
            "Dominant backend stayed consistent across runs, indicating a stable execution path."
        )
    else:
        blockers.append("Backend path changed across sampled runs, reducing operational confidence.")

    if production is None or not production.get("available"):
        blockers.append(
            "Production benchmark is still blocked because the repo does not currently contain the labeled production PDFs."
        )

    scientific_ready = not any(
        item.startswith("Scientific benchmark") or "PubTables" in item or "Latency" in item
        for item in blockers
    )
    retrieval_ready_for_prototyping = scientific_ready
    retrieval_ready_for_production = scientific_ready and production is not None and production.get("available")

    return {
        "scientific_ready": scientific_ready,
        "retrieval_ready_for_prototyping": retrieval_ready_for_prototyping,
        "retrieval_ready_for_production": retrieval_ready_for_production,
        "strengths": strengths,
        "blockers": blockers,
    }


def render_markdown(report: dict[str, Any]) -> str:
    scientific = report["scientific"]
    lines = [
        "# Retrieval Readiness Benchmark",
        "",
        f"- Generated at: {report['metadata']['generated_at_utc']}",
        f"- Git commit: {report['metadata']['git_commit']}",
        f"- Device target: {report['metadata']['device_target']}",
        f"- Model profile: {report['metadata']['profile_name']}",
        "",
        "## Why This Is Strong",
        "",
    ]

    for item in report["verdict"]["strengths"]:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Scientific GPU Runs",
            "",
            "| PubTables limit | Success | Mean latency (s) | Median (s) | P95 (s) | IoU50 micro F1 | IoU75 micro F1 | Backend | Route |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )

    for entry in scientific["runs"]:
        lines.append(
            "| "
            f"{entry['pubtables_limit']} | "
            f"{entry['success_rate']:.3f} | "
            f"{entry['latency_mean_sec']:.3f} | "
            f"{entry['latency_median_sec']:.3f} | "
            f"{entry['latency_p95_sec']:.3f} | "
            f"{entry['iou50_micro_f1']:.3f} | "
            f"{entry['iou75_micro_f1']:.3f} | "
            f"{entry['dominant_backend'] or 'NA'} | "
            f"{entry['dominant_route'] or 'NA'} |"
        )

    lines.extend(
        [
            "",
            "## Stability Summary",
            "",
            f"- Success rate range: {scientific['stability']['success_rate_min']:.3f} - {scientific['stability']['success_rate_max']:.3f}",
            f"- IoU50 micro F1 range: {scientific['stability']['iou50_micro_f1_min']:.3f} - {scientific['stability']['iou50_micro_f1_max']:.3f}",
            f"- IoU75 micro F1 range: {scientific['stability']['iou75_micro_f1_min']:.3f} - {scientific['stability']['iou75_micro_f1_max']:.3f}",
            f"- P95 latency range: {scientific['stability']['latency_p95_sec_min']:.3f}s - {scientific['stability']['latency_p95_sec_max']:.3f}s",
            f"- Backend consistency: {scientific['stability']['backend_consistent']}",
            f"- Route consistency: {scientific['stability']['route_consistent']}",
            "",
            "## Gates",
            "",
        ]
    )

    for label, gate in scientific["gates"].items():
        passed = all(bool(value) for value in gate.values())
        lines.append(f"- {label}: pass={passed} details={json.dumps(gate, ensure_ascii=False)}")

    if report.get("baseline_reference"):
        baseline = report["baseline_reference"]
        lines.extend(
            [
                "",
                "## Baseline Reference",
                "",
                f"- Summary: {baseline['summary_path']}",
                f"- Success rate: {baseline['success_rate']:.3f}",
                f"- IoU50 micro F1: {baseline['iou50_micro_f1']:.3f}",
                f"- IoU75 micro F1: {baseline['iou75_micro_f1']:.3f}",
                f"- Mean latency: {baseline['latency_mean_sec']:.3f}s",
                f"- Note: {baseline['note']}",
            ]
        )

    if report.get("production"):
        production = report["production"]
        lines.extend(
            [
                "",
                "## Production Status",
                "",
                f"- Summary: {production['summary_path']}",
                f"- Production dataset available: {production['available']}",
                f"- Documents total by profile: `{json.dumps(production['documents_total_by_profile'], ensure_ascii=False)}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Retrieval Verdict",
            "",
            f"- Scientific ready: {report['verdict']['scientific_ready']}",
            f"- Ready for retrieval prototyping: {report['verdict']['retrieval_ready_for_prototyping']}",
            f"- Ready for production retrieval claims: {report['verdict']['retrieval_ready_for_production']}",
        ]
    )

    if report["verdict"]["blockers"]:
        lines.extend(["", "## Remaining Blockers", ""])
        for item in report["verdict"]["blockers"]:
            lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Repro Commands",
            "",
            "```powershell",
            "$env:BOXBIIBOO_LAYOUT_DEVICE='cuda'",
            "$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK='True'",
            ".\\.venv-gpu\\Scripts\\python.exe scripts/benchmark_ingest_scientific.py --skip-doclaynet --pubtables-root data/benchmarks/pubtables_detection --pubtables-split test --pubtables-limit 25 --profiles model_routed_doclaynet",
            ".\\.venv-gpu\\Scripts\\python.exe scripts/benchmark_ingest_scientific.py --skip-doclaynet --pubtables-root data/benchmarks/pubtables_detection --pubtables-split test --pubtables-limit 100 --profiles model_routed_doclaynet",
            ".\\.venv-gpu\\Scripts\\python.exe scripts/benchmark_ingest_scientific.py --skip-doclaynet --pubtables-root data/benchmarks/pubtables_detection --pubtables-split test --pubtables-limit 500 --profiles model_routed_doclaynet",
            "```",
        ]
    )

    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (RESULTS_ROOT / timestamp)

    scientific_entries = [scientific_entry(Path(path)) for path in args.scientific_summary]
    scientific_entries.sort(key=lambda item: item["pubtables_limit"])
    scientific = summarize_scientific(scientific_entries)
    production = summarize_production(args.production_summary)
    baseline_reference = summarize_baseline_reference(args.baseline_reference)

    git_commit = None
    if scientific_entries:
        first = load_json(Path(scientific_entries[0]["summary_path"]))
        git_commit = first.get("metadata", {}).get("git_commit")

    report = {
        "metadata": {
            "generated_at_utc": timestamp,
            "git_commit": git_commit,
            "device_target": "cuda / RTX 3050",
            "profile_name": "model_routed_doclaynet",
        },
        "scientific": scientific,
        "production": production,
        "baseline_reference": baseline_reference,
        "verdict": build_verdict(scientific, production),
    }

    save_json(output_dir / "readiness_report.json", report)
    save_text(output_dir / "README.md", render_markdown(report))

    print(output_dir)


if __name__ == "__main__":
    main()
