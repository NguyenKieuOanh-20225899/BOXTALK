from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ingest.extract.model_layout import DEFAULT_LAYOUT_MODEL_NAME


DATASET_ROOT = Path("data/test_probe")
LABELS_FILE = DATASET_ROOT / "labels.json"
RESULTS_ROOT = Path("results/ingest_benchmark")
VALID_LABELS = ["text", "layout", "ocr", "mixed"]

PROFILE_ENVS: dict[str, dict[str, str]] = {
    "baseline": {
        "BOXBIIBOO_LAYOUT_MODEL_NAME": "0",
        "BOXBIIBOO_ENABLE_MODEL_ROUTING": "0",
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
        "TOKENIZERS_PARALLELISM": "false",
    },
    "model_routed_doclaynet": {
        "BOXBIIBOO_LAYOUT_MODEL_NAME": DEFAULT_LAYOUT_MODEL_NAME,
        "BOXBIIBOO_ENABLE_MODEL_ROUTING": "1",
        "BOXBIIBOO_LAYOUT_SCORE_THRESHOLD": "0.35",
        "BOXBIIBOO_LAYOUT_RENDER_SCALE": "2.0",
        "BOXBIIBOO_OCR_REGION_SCALE": "3.0",
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
        "TOKENIZERS_PARALLELISM": "false",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standard ingest benchmark")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="Dataset root containing labels.json and PDFs",
    )
    parser.add_argument(
        "--labels-file",
        type=Path,
        default=LABELS_FILE,
        help="Path to labels.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional benchmark output directory",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Measured repeats for each document",
    )
    parser.add_argument(
        "--warmup-per-label",
        type=int,
        default=1,
        help="Warmup documents per gold label before measured runs",
    )
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=0,
        help="Optional maximum measured documents per gold label (0 means all available)",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["baseline", "model_routed_doclaynet"],
        help="Profiles to benchmark",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--profile", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def load_labels(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"labels.json must be a dict: {path}")
    return data


def normalize_label_entry(entry: Any) -> dict[str, Any]:
    if isinstance(entry, str):
        label = entry
        meta: dict[str, Any] = {}
    elif isinstance(entry, dict):
        label = entry.get("label")
        meta = {k: v for k, v in entry.items() if k != "label"}
    else:
        raise ValueError(f"Invalid label entry: {entry!r}")

    if label not in VALID_LABELS:
        raise ValueError(f"Invalid label: {label!r}")

    return {"label": label, **meta}


def resolve_dataset_entries(
    dataset_root: Path,
    labels_file: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = load_labels(labels_file)
    entries: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for rel_path in sorted(labels.keys()):
        info = normalize_label_entry(labels[rel_path])
        pdf_path = dataset_root / rel_path
        if not pdf_path.exists():
            issues.append(
                {
                    "type": "missing_pdf",
                    "file": rel_path,
                    "gold_label": info["label"],
                }
            )
            continue
        entries.append(
            {
                "rel_path": rel_path,
                "pdf_path": pdf_path,
                "gold_label": info["label"],
                "label_meta": {k: v for k, v in info.items() if k != "label"},
            }
        )
    return entries, issues


def select_warmup_entries(entries: list[dict[str, Any]], warmup_per_label: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[entry["gold_label"]].append(entry)

    warmup: list[dict[str, Any]] = []
    for label in VALID_LABELS:
        warmup.extend(grouped.get(label, [])[: max(0, warmup_per_label)])
    return warmup


def limit_entries_per_label(entries: list[dict[str, Any]], max_per_label: int) -> list[dict[str, Any]]:
    if max_per_label <= 0:
        return entries

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[entry["gold_label"]].append(entry)

    limited: list[dict[str, Any]] = []
    for label in VALID_LABELS:
        limited.extend(grouped.get(label, [])[:max_per_label])
    return limited


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * q
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return values[low]
    weight = rank - low
    return values[low] * (1.0 - weight) + values[high] * weight


def backend_family(backend_name: str | None) -> str:
    backend = (backend_name or "").strip().lower()
    if backend in {"model_routed", "model_layout", "layout"}:
        return "layout"
    if backend == "ocr":
        return "ocr"
    if backend == "text":
        return "text"
    return backend or "unknown"


def run_single_ingest(entry: dict[str, Any], *, repeat_index: int, warmup: bool) -> dict[str, Any]:
    from app.ingest.pipeline import ingest_pdf

    pdf_path = entry["pdf_path"]
    started = time.perf_counter()
    out = ingest_pdf(pdf_path)
    elapsed = time.perf_counter() - started

    probe = out.get("probe", {})
    pages = out.get("pages", [])
    blocks = out.get("blocks", [])
    chunks = out.get("chunks", [])
    used_backend = out.get("used_backend")

    bbox_blocks = sum(1 for b in blocks if b.bbox is not None)
    route_counts = Counter(
        (b.meta or {}).get("route_backend", b.source_mode)
        for b in blocks
    )
    block_type_counts = Counter(b.block_type for b in blocks)
    extracted_chars = sum(len((b.text or "").strip()) for b in blocks)
    page_count = len(pages) or int(probe.get("page_count") or 0)

    record = {
        "file": entry["rel_path"],
        "gold_label": entry["gold_label"],
        "repeat_index": repeat_index,
        "warmup": int(warmup),
        "success": 1,
        "elapsed_sec": elapsed,
        "used_backend": used_backend or "NA",
        "backend_family": backend_family(used_backend),
        "probe_mode": probe.get("probe_detected_mode") or "NA",
        "probe_correct": int((probe.get("probe_detected_mode") or "NA") == entry["gold_label"]),
        "page_count": page_count,
        "block_count": len(blocks),
        "chunk_count": len(chunks),
        "bbox_ratio": (bbox_blocks / len(blocks)) if blocks else 0.0,
        "extracted_chars": extracted_chars,
        "pages_per_sec": (page_count / elapsed) if elapsed > 0 and page_count else 0.0,
        "blocks_per_page": (len(blocks) / page_count) if page_count else 0.0,
        "chunks_per_page": (len(chunks) / page_count) if page_count else 0.0,
        "route_counts": dict(route_counts),
        "block_type_counts": dict(block_type_counts),
        "errors": out.get("errors", []),
    }
    record.update({f"gold_meta_{k}": v for k, v in entry["label_meta"].items()})
    return record


def benchmark_profile(
    *,
    profile_name: str,
    dataset_root: Path,
    labels_file: Path,
    repeats: int,
    warmup_per_label: int,
    max_per_label: int,
) -> dict[str, Any]:
    entries, dataset_issues = resolve_dataset_entries(dataset_root, labels_file)
    entries = limit_entries_per_label(entries, max_per_label)
    warmup_entries = select_warmup_entries(entries, warmup_per_label)

    warmup_results: list[dict[str, Any]] = []
    warmup_started = time.perf_counter()
    for entry in warmup_entries:
        try:
            warmup_results.append(run_single_ingest(entry, repeat_index=0, warmup=True))
        except Exception as e:
            warmup_results.append(
                {
                    "file": entry["rel_path"],
                    "gold_label": entry["gold_label"],
                    "repeat_index": 0,
                    "warmup": 1,
                    "success": 0,
                    "error": str(e),
                }
            )
    warmup_elapsed = time.perf_counter() - warmup_started

    measured_records: list[dict[str, Any]] = []
    for repeat_index in range(repeats):
        for entry in entries:
            try:
                measured_records.append(
                    run_single_ingest(entry, repeat_index=repeat_index, warmup=False)
                )
            except Exception as e:
                measured_records.append(
                    {
                        "file": entry["rel_path"],
                        "gold_label": entry["gold_label"],
                        "repeat_index": repeat_index,
                        "warmup": 0,
                        "success": 0,
                        "error": str(e),
                    }
                )

    summary = summarize_profile_records(profile_name, measured_records)
    return {
        "profile": profile_name,
        "env": PROFILE_ENVS[profile_name],
        "dataset_issues": dataset_issues,
        "warmup": {
            "documents": [entry["rel_path"] for entry in warmup_entries],
            "elapsed_sec": warmup_elapsed,
            "success_count": sum(int(r.get("success", 0)) for r in warmup_results),
            "results": warmup_results,
        },
        "measured": measured_records,
        "summary": summary,
    }


def summarize_profile_records(profile_name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    success_records = [r for r in records if int(r.get("success", 0)) == 1]
    error_records = [r for r in records if int(r.get("success", 0)) == 0]

    latencies = sorted(float(r["elapsed_sec"]) for r in success_records)
    total_pages = sum(int(r.get("page_count", 0) or 0) for r in success_records)
    total_time = sum(float(r.get("elapsed_sec", 0.0) or 0.0) for r in success_records)
    backend_counts = Counter(str(r.get("used_backend", "NA")) for r in success_records)
    route_counts: Counter[str] = Counter()
    for r in success_records:
        route_counts.update(r.get("route_counts", {}))

    summary: dict[str, Any] = {
        "profile": profile_name,
        "documents_total": len(records),
        "documents_success": len(success_records),
        "documents_failed": len(error_records),
        "success_rate": (len(success_records) / len(records)) if records else 0.0,
        "probe_accuracy": (
            sum(int(r.get("probe_correct", 0)) for r in success_records) / len(success_records)
            if success_records
            else 0.0
        ),
        "latency_mean_sec": statistics.mean(latencies) if latencies else 0.0,
        "latency_median_sec": statistics.median(latencies) if latencies else 0.0,
        "latency_p95_sec": percentile(latencies, 0.95) if latencies else 0.0,
        "total_elapsed_sec": total_time,
        "total_pages": total_pages,
        "pages_per_sec_micro": (total_pages / total_time) if total_time > 0 and total_pages else 0.0,
        "docs_per_sec_micro": (len(success_records) / total_time) if total_time > 0 else 0.0,
        "bbox_ratio_mean": (
            statistics.mean(float(r.get("bbox_ratio", 0.0)) for r in success_records)
            if success_records
            else 0.0
        ),
        "blocks_per_page_mean": (
            statistics.mean(float(r.get("blocks_per_page", 0.0)) for r in success_records)
            if success_records
            else 0.0
        ),
        "chunks_per_page_mean": (
            statistics.mean(float(r.get("chunks_per_page", 0.0)) for r in success_records)
            if success_records
            else 0.0
        ),
        "backend_counts": dict(backend_counts),
        "route_counts": dict(route_counts),
        "errors": [
            {
                "file": r.get("file"),
                "repeat_index": r.get("repeat_index"),
                "error": r.get("error"),
            }
            for r in error_records
        ],
    }

    by_label: dict[str, Any] = {}
    for label in VALID_LABELS:
        subset = [r for r in records if r.get("gold_label") == label]
        subset_success = [r for r in subset if int(r.get("success", 0)) == 1]
        subset_latencies = sorted(float(r["elapsed_sec"]) for r in subset_success)
        subset_pages = sum(int(r.get("page_count", 0) or 0) for r in subset_success)
        subset_time = sum(float(r.get("elapsed_sec", 0.0) or 0.0) for r in subset_success)
        by_label[label] = {
            "documents_total": len(subset),
            "documents_success": len(subset_success),
            "success_rate": (len(subset_success) / len(subset)) if subset else 0.0,
            "probe_accuracy": (
                sum(int(r.get("probe_correct", 0)) for r in subset_success) / len(subset_success)
                if subset_success
                else 0.0
            ),
            "latency_mean_sec": statistics.mean(subset_latencies) if subset_latencies else 0.0,
            "latency_median_sec": statistics.median(subset_latencies) if subset_latencies else 0.0,
            "latency_p95_sec": percentile(subset_latencies, 0.95) if subset_latencies else 0.0,
            "pages_per_sec_micro": (subset_pages / subset_time) if subset_time > 0 and subset_pages else 0.0,
            "bbox_ratio_mean": (
                statistics.mean(float(r.get("bbox_ratio", 0.0)) for r in subset_success)
                if subset_success
                else 0.0
            ),
            "backend_counts": dict(Counter(str(r.get("used_backend", "NA")) for r in subset_success)),
        }

    summary["by_label"] = by_label
    return summary


def flatten_records_for_csv(profile_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in profile_payloads:
        profile = payload["profile"]
        for record in payload["measured"]:
            row = dict(record)
            row["profile"] = profile
            row["route_counts"] = json.dumps(record.get("route_counts", {}), ensure_ascii=False, sort_keys=True)
            row["block_type_counts"] = json.dumps(record.get("block_type_counts", {}), ensure_ascii=False, sort_keys=True)
            row["errors"] = json.dumps(record.get("errors", []), ensure_ascii=False)
            rows.append(row)
    return rows


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_profile_comparison(profile_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    payload_by_name = {payload["profile"]: payload for payload in profile_payloads}
    baseline = payload_by_name.get("baseline")
    candidate = payload_by_name.get("model_routed_doclaynet")

    if not baseline or not candidate:
        return {}

    base_summary = baseline["summary"]
    cand_summary = candidate["summary"]

    metrics = [
        "success_rate",
        "probe_accuracy",
        "latency_mean_sec",
        "latency_median_sec",
        "latency_p95_sec",
        "pages_per_sec_micro",
        "bbox_ratio_mean",
        "blocks_per_page_mean",
        "chunks_per_page_mean",
    ]

    comparison = {}
    for metric in metrics:
        base_value = float(base_summary.get(metric, 0.0) or 0.0)
        cand_value = float(cand_summary.get(metric, 0.0) or 0.0)
        comparison[metric] = {
            "baseline": base_value,
            "model_routed_doclaynet": cand_value,
            "delta_absolute": cand_value - base_value,
            "delta_relative": ((cand_value - base_value) / base_value) if base_value else None,
        }

    return comparison


def render_markdown_summary(
    *,
    metadata: dict[str, Any],
    profile_payloads: list[dict[str, Any]],
    comparison: dict[str, Any],
) -> str:
    lines = [
        "# Ingest Benchmark",
        "",
        f"- Timestamp: {metadata['timestamp_utc']}",
        f"- Git commit: {metadata['git_commit']}",
        f"- Python: {metadata['python_version']}",
        f"- Platform: {metadata['platform']}",
        f"- Dataset: {metadata['dataset_root']}",
        f"- Labels: {metadata['labels_file']}",
        f"- Chosen model: {metadata['chosen_model']}",
        "",
        "## Profiles",
        "",
    ]

    for payload in profile_payloads:
        summary = payload["summary"]
        lines.extend(
            [
                f"### {payload['profile']}",
                "",
                f"- Success rate: {summary['success_rate']:.3f}",
                f"- Probe accuracy: {summary['probe_accuracy']:.3f}",
                f"- Mean latency: {summary['latency_mean_sec']:.3f}s",
                f"- Median latency: {summary['latency_median_sec']:.3f}s",
                f"- P95 latency: {summary['latency_p95_sec']:.3f}s",
                f"- Pages/sec: {summary['pages_per_sec_micro']:.3f}",
                f"- Mean bbox ratio: {summary['bbox_ratio_mean']:.3f}",
                f"- Backend counts: `{json.dumps(summary['backend_counts'], ensure_ascii=False, sort_keys=True)}`",
                f"- Route counts: `{json.dumps(summary['route_counts'], ensure_ascii=False, sort_keys=True)}`",
                "",
            ]
        )

    if comparison:
        lines.extend(["## Comparison", ""])
        for metric, values in comparison.items():
            delta = values["delta_absolute"]
            rel = values["delta_relative"]
            rel_str = f"{rel:.3f}" if rel is not None else "NA"
            lines.append(
                f"- {metric}: baseline={values['baseline']:.3f}, model={values['model_routed_doclaynet']:.3f}, delta={delta:.3f}, rel={rel_str}"
            )
        lines.append("")

    return "\n".join(lines)


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def run_profile_subprocess(profile: str, args: argparse.Namespace) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(PROFILE_ENVS[profile])

    cmd = [
        sys.executable,
        __file__,
        "--worker",
        "--profile",
        profile,
        "--dataset-root",
        str(args.dataset_root),
        "--labels-file",
        str(args.labels_file),
        "--repeats",
        str(args.repeats),
        "--warmup-per-label",
        str(args.warmup_per_label),
        "--max-per-label",
        str(args.max_per_label),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Benchmark worker failed for profile={profile}\n"
            f"returncode={result.returncode}\n"
            f"stderr:\n{result.stderr}\n"
            f"stdout:\n{result.stdout}"
        )

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Worker output is not valid JSON for profile={profile}\n{result.stdout}"
        ) from e


def worker_entry(args: argparse.Namespace) -> None:
    if args.profile not in PROFILE_ENVS:
        raise ValueError(f"Unknown profile: {args.profile}")

    payload = benchmark_profile(
        profile_name=args.profile,
        dataset_root=args.dataset_root,
        labels_file=args.labels_file,
        repeats=args.repeats,
        warmup_per_label=args.warmup_per_label,
        max_per_label=args.max_per_label,
    )
    print(json.dumps(payload, ensure_ascii=False))


def main() -> None:
    args = parse_args()

    if args.worker:
        worker_entry(args)
        return

    for profile in args.profiles:
        if profile not in PROFILE_ENVS:
            raise ValueError(f"Unknown profile: {profile}")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (RESULTS_ROOT / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_payloads: list[dict[str, Any]] = []
    for profile in args.profiles:
        print(f"[benchmark] running profile={profile}", file=sys.stderr)
        payload = run_profile_subprocess(profile, args)
        profile_payloads.append(payload)
        save_json(output_dir / f"profile_{profile}.json", payload)

    comparison = build_profile_comparison(profile_payloads)
    per_file_rows = flatten_records_for_csv(profile_payloads)

    metadata = {
        "timestamp_utc": timestamp,
        "git_commit": git_commit(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "dataset_root": str(args.dataset_root),
        "labels_file": str(args.labels_file),
        "chosen_model": DEFAULT_LAYOUT_MODEL_NAME,
        "profiles": args.profiles,
        "repeats": args.repeats,
        "warmup_per_label": args.warmup_per_label,
        "max_per_label": args.max_per_label,
    }

    summary = {
        "metadata": metadata,
        "profiles": {payload["profile"]: payload["summary"] for payload in profile_payloads},
        "comparison": comparison,
    }

    save_json(output_dir / "benchmark_summary.json", summary)
    save_json(output_dir / "per_file_metrics.json", per_file_rows)
    save_csv(output_dir / "per_file_metrics.csv", per_file_rows)
    save_json(output_dir / "comparison.json", comparison)
    (output_dir / "summary.md").write_text(
        render_markdown_summary(
            metadata=metadata,
            profile_payloads=profile_payloads,
            comparison=comparison,
        ),
        encoding="utf-8",
    )

    print(str(output_dir))


if __name__ == "__main__":
    main()
