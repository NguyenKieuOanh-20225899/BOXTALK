from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.qa.llm_fallback import provider_runtime_info


DEFAULT_MANIFEST = Path("data/llm_fallback_benchmark/manifest.json")
RESULTS_ROOT = Path("results/llm_fallback_benchmark")

PRIMARY_METRICS = [
    "success_gain_vs_standard",
    "answer_match_gain_vs_standard",
    "groundedness",
    "groundedness_delta",
    "hallucination_delta",
    "fallback_call_rate",
    "fallback_used_rate",
    "table_rule_resolved_count",
    "table_llm_resolved_count",
    "table_total_success",
    "latency_overhead_ms",
    "fallback_override_success_rate",
    "fallback_override_grounded_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the grounded LLM fallback benchmark repeatedly and aggregate stability metrics."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Fallback benchmark manifest JSON.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory containing per-run outputs and aggregate summary.")
    parser.add_argument("--repeat", type=int, default=3, help="Number of benchmark repeats to run.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for child scripts.")
    parser.add_argument("--dense-preset", default="minilm", help="Dense preset for retrieval index build.")
    parser.add_argument("--standard-config", default="routed_grounded", help="Standard QA config.")
    parser.add_argument("--fallback-config", default="routed_grounded_with_llm_fallback", help="Fallback QA config.")
    parser.add_argument(
        "--llm-fallback-provider",
        choices=["dummy", "openai-compatible"],
        default=os.getenv("BOXTALK_LLM_PROVIDER", "dummy"),
        help="Provider used by the fallback QA config.",
    )
    parser.add_argument(
        "--llm-fallback-sufficiency-threshold",
        type=float,
        default=float(os.getenv("BOXTALK_LLM_FALLBACK_SUFFICIENCY_THRESHOLD", "0.72")),
    )
    parser.add_argument(
        "--llm-fallback-min-confidence",
        type=float,
        default=float(os.getenv("BOXTALK_LLM_FALLBACK_MIN_CONFIDENCE", "0.30")),
    )
    parser.add_argument(
        "--llm-fallback-min-override-confidence",
        type=float,
        default=float(os.getenv("BOXTALK_LLM_FALLBACK_MIN_OVERRIDE_CONFIDENCE", "0.65")),
    )
    parser.add_argument("--skip-build", action="store_true", help="Do not build missing index.")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild the retrieval index before the first run.")
    parser.add_argument("--recreate-dataset", action="store_true", help="Recreate the controlled fallback dataset before the first run.")
    parser.add_argument("--no-warmup", action="store_true", help="Pass --no-warmup to benchmark_qa.py.")
    parser.add_argument("--dry-run", action="store_true", help="Print child commands without executing them.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def validate_provider_or_exit(provider: str) -> dict[str, Any]:
    info = provider_runtime_info(provider)
    if info["provider"] == "openai-compatible" and not info["ready"]:
        missing = ", ".join(info["missing_envs"])
        raise SystemExit(
            "OpenAI-compatible fallback provider is not ready. Missing required env: "
            f"{missing}. Set BOXTALK_LLM_BASE_URL, BOXTALK_LLM_API_KEY, and BOXTALK_LLM_MODEL."
        )
    return info


def print_provider_runtime(info: dict[str, Any]) -> None:
    print(
        "LLM fallback provider: "
        f"{info.get('provider')} | ready={info.get('ready')} | "
        f"mode={'real_provider' if info.get('provider') == 'openai-compatible' else 'plumbing_check'} | "
        f"base_url={info.get('base_url') or 'n/a'} | "
        f"model={info.get('model') or 'n/a'} | "
        f"api_key_present={info.get('api_key_present')}",
        flush=True,
    )


def run_command(command: list[str], *, dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=ROOT, check=True)


def resolve_output_dir(output_dir: Path | None) -> Path:
    if output_dir is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return ROOT / RESULTS_ROOT / f"openai_repeats_{timestamp}"
    return output_dir if output_dir.is_absolute() else ROOT / output_dir


def build_run_command(args: argparse.Namespace, *, run_dir: Path, run_index: int) -> list[str]:
    command = [
        args.python,
        str(ROOT / "scripts" / "benchmark_llm_fallback.py"),
        "--manifest",
        str(args.manifest),
        "--output-dir",
        str(run_dir),
        "--python",
        args.python,
        "--dense-preset",
        args.dense_preset,
        "--standard-config",
        args.standard_config,
        "--fallback-config",
        args.fallback_config,
        "--llm-fallback-provider",
        args.llm_fallback_provider,
        "--llm-fallback-sufficiency-threshold",
        str(args.llm_fallback_sufficiency_threshold),
        "--llm-fallback-min-confidence",
        str(args.llm_fallback_min_confidence),
        "--llm-fallback-min-override-confidence",
        str(args.llm_fallback_min_override_confidence),
    ]
    if args.skip_build:
        command.append("--skip-build")
    if args.rebuild_index and run_index == 1:
        command.append("--rebuild-index")
    if args.recreate_dataset and run_index == 1:
        command.append("--recreate-dataset")
    if args.no_warmup:
        command.append("--no-warmup")
    if args.dry_run:
        command.append("--dry-run")
    return command


def float_values(values: Iterable[Any]) -> list[float]:
    floats: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            floats.append(float(value))
        except (TypeError, ValueError):
            continue
    return floats


def metric_stats(values: Iterable[Any]) -> dict[str, Any]:
    numeric = float_values(values)
    if not numeric:
        return {"count": 0, "mean": None, "min": None, "max": None, "std": None}
    return {
        "count": len(numeric),
        "mean": statistics.mean(numeric),
        "min": min(numeric),
        "max": max(numeric),
        "std": statistics.stdev(numeric) if len(numeric) > 1 else 0.0,
    }


def stats_for_metrics(payloads: list[dict[str, Any]], metrics: list[str] = PRIMARY_METRICS) -> dict[str, Any]:
    return {metric: metric_stats(payload.get(metric) for payload in payloads) for metric in metrics}


def grouped_stats(summaries: list[dict[str, Any]], group_name: str) -> dict[str, Any]:
    group_keys = sorted(
        {
            key
            for summary in summaries
            for key in (summary.get(group_name) or {}).keys()
        }
    )
    result: dict[str, Any] = {}
    for key in group_keys:
        payloads = [
            group_payload
            for summary in summaries
            if isinstance((group_payload := (summary.get(group_name) or {}).get(key)), dict)
        ]
        result[key] = stats_for_metrics(payloads)
    return result


def stat_value(stats: dict[str, Any], metric: str, field: str, default: float = 0.0) -> float:
    value = (stats.get(metric) or {}).get(field)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_stability_readout(metric_summary: dict[str, Any], *, benchmark_mode: str) -> dict[str, Any]:
    success_gain_min = stat_value(metric_summary, "success_gain_vs_standard", "min")
    success_gain_mean = stat_value(metric_summary, "success_gain_vs_standard", "mean")
    answer_gain_min = stat_value(metric_summary, "answer_match_gain_vs_standard", "min")
    answer_gain_mean = stat_value(metric_summary, "answer_match_gain_vs_standard", "mean")
    hallucination_delta_max = stat_value(metric_summary, "hallucination_delta", "max")
    groundedness_delta_min = stat_value(metric_summary, "groundedness_delta", "min")
    table_llm_resolved_min = stat_value(metric_summary, "table_llm_resolved_count", "min")
    table_llm_resolved_max = stat_value(metric_summary, "table_llm_resolved_count", "max")
    is_real_provider = benchmark_mode == "real_provider"
    gate_candidate = (
        is_real_provider
        and success_gain_min > 0.0
        and answer_gain_min >= 0.0
        and hallucination_delta_max <= 0.0
        and groundedness_delta_min >= 0.0
    )
    return {
        "real_gain": {
            "status": "stable_positive" if success_gain_min > 0.0 or answer_gain_min > 0.0 else "mixed_or_none",
            "success_gain_mean": success_gain_mean,
            "success_gain_min": success_gain_min,
            "answer_match_gain_mean": answer_gain_mean,
            "answer_match_gain_min": answer_gain_min,
        },
        "grounded_safety": {
            "status": "kept" if hallucination_delta_max <= 0.0 and groundedness_delta_min >= 0.0 else "needs_review",
            "hallucination_delta_max": hallucination_delta_max,
            "groundedness_delta_min": groundedness_delta_min,
        },
        "targeting": {
            "status": "provider_helped_table" if table_llm_resolved_max > 0.0 else "no_table_llm_gain_observed",
            "table_llm_resolved_min": table_llm_resolved_min,
            "table_llm_resolved_max": table_llm_resolved_max,
        },
        "experimental_gate_suggestion": {
            "candidate": gate_candidate,
            "reason": (
                "Repeated real-provider runs are stable enough to review a separate experimental fallback gate."
                if gate_candidate
                else (
                    "Dummy provider repeats are a plumbing check only; do not add a fallback gate from this run."
                    if not is_real_provider
                    else "Do not add a fallback gate yet; inspect unstable gain, safety, or table targeting first."
                )
            ),
        },
    }


def format_metric(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    provider = summary["provider"]
    stability = summary["stability_readout"]
    metric_summary = summary["metric_summary"]
    lines = [
        "# LLM Fallback Repeated Benchmark",
        "",
        f"- Provider: `{provider['provider']}`",
        f"- Provider ready: `{provider['ready']}`",
        f"- Provider base URL: `{provider.get('base_url') or 'n/a'}`",
        f"- Provider model: `{provider.get('model') or 'n/a'}`",
        f"- API key present: `{provider.get('api_key_present')}`",
        f"- Repeat count: `{summary['repeat']}`",
        f"- Benchmark mode: `{summary['benchmark_mode']}`",
        "",
        "## Stability Readout",
        "",
        "| Question | Status | Key values |",
        "|---|---|---|",
        "| Provider gain? | `{status}` | success mean `{success_mean}`, success min `{success_min}`, answer min `{answer_min}` |".format(
            status=stability["real_gain"]["status"],
            success_mean=format_metric(stability["real_gain"]["success_gain_mean"]),
            success_min=format_metric(stability["real_gain"]["success_gain_min"]),
            answer_min=format_metric(stability["real_gain"]["answer_match_gain_min"]),
        ),
        "| Grounded safety? | `{status}` | hallucination delta max `{hallucination}`, groundedness delta min `{grounded}` |".format(
            status=stability["grounded_safety"]["status"],
            hallucination=format_metric(stability["grounded_safety"]["hallucination_delta_max"]),
            grounded=format_metric(stability["grounded_safety"]["groundedness_delta_min"]),
        ),
        "| Helps in the right place? | `{status}` | table LLM resolved min `{table_min}`, max `{table_max}` |".format(
            status=stability["targeting"]["status"],
            table_min=format_metric(stability["targeting"]["table_llm_resolved_min"], digits=0),
            table_max=format_metric(stability["targeting"]["table_llm_resolved_max"], digits=0),
        ),
        "| Experimental gate suggestion | `{candidate}` | {reason} |".format(
            candidate=stability["experimental_gate_suggestion"]["candidate"],
            reason=stability["experimental_gate_suggestion"]["reason"],
        ),
        "",
        "## Metric Stability",
        "",
        "| Metric | Mean | Min | Max | Std |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in PRIMARY_METRICS:
        stats = metric_summary.get(metric, {})
        lines.append(
            "| {metric} | {mean} | {min} | {max} | {std} |".format(
                metric=metric,
                mean=format_metric(stats.get("mean")),
                min=format_metric(stats.get("min")),
                max=format_metric(stats.get("max")),
                std=format_metric(stats.get("std")),
            )
        )
    lines.extend(
        [
            "",
            "## Run Outputs",
            "",
            "| Run | Output | Success Gain | Answer Gain | Hallucination Delta | Table LLM Resolved |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for run in summary["runs"]:
        aggregate = run["aggregate"]
        lines.append(
            "| {run_number} | `{output}` | {success} | {answer} | {hallucination} | {table_llm} |".format(
                run_number=run["run_number"],
                output=run["output_dir"],
                success=format_metric(aggregate.get("success_gain_vs_standard")),
                answer=format_metric(aggregate.get("answer_match_gain_vs_standard")),
                hallucination=format_metric(aggregate.get("hallucination_delta")),
                table_llm=aggregate.get("table_llm_resolved_count", 0),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")

    provider_info = validate_provider_or_exit(args.llm_fallback_provider)
    print_provider_runtime(provider_info)

    output_dir = resolve_output_dir(args.output_dir)
    if args.dry_run:
        for run_index in range(1, args.repeat + 1):
            run_dir = output_dir / f"run_{run_index:02d}"
            run_command(build_run_command(args, run_dir=run_dir, run_index=run_index), dry_run=True)
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    run_summaries: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []

    for run_index in range(1, args.repeat + 1):
        run_dir = output_dir / f"run_{run_index:02d}"
        command = build_run_command(args, run_dir=run_dir, run_index=run_index)
        run_command(command, dry_run=False)
        summary_path = run_dir / "comparison_summary.json"
        run_summary = load_json(summary_path)
        run_summaries.append(run_summary)
        runs.append(
            {
                "run_number": run_index,
                "output_dir": str(run_dir),
                "summary_path": str(summary_path),
                "benchmark_mode": run_summary.get("benchmark_mode"),
                "provider": run_summary.get("provider"),
                "aggregate": run_summary.get("aggregate", {}),
                "decision_readout": run_summary.get("decision_readout", {}),
            }
        )

    aggregate_payloads = [summary.get("aggregate", {}) for summary in run_summaries]
    metric_summary = stats_for_metrics(aggregate_payloads)
    benchmark_mode = "plumbing_check" if provider_info["provider"] == "dummy" else "real_provider"
    repeat_summary = {
        "timestamp_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "manifest": str(args.manifest),
        "output_dir": str(output_dir),
        "repeat": args.repeat,
        "benchmark_mode": benchmark_mode,
        "provider": provider_info,
        "standard_config": args.standard_config,
        "fallback_config": args.fallback_config,
        "metric_summary": metric_summary,
        "stability_readout": build_stability_readout(metric_summary, benchmark_mode=benchmark_mode),
        "by_reasoning_mode": grouped_stats(run_summaries, "by_reasoning_mode"),
        "by_expected_modality": grouped_stats(run_summaries, "by_expected_modality"),
        "runs": runs,
    }
    (output_dir / "repeat_summary.json").write_text(
        json.dumps(repeat_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(output_dir / "README.md", repeat_summary)
    print(output_dir)


if __name__ == "__main__":
    main()
