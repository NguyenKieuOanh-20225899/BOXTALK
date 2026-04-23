from __future__ import annotations

import argparse
import csv
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark grounded LLM fallback gain against the standard routed QA path.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Fallback benchmark manifest JSON.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional benchmark output directory.")
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
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild the retrieval index before benchmarking.")
    parser.add_argument("--recreate-dataset", action="store_true", help="Recreate the controlled fallback benchmark dataset first.")
    parser.add_argument("--no-warmup", action="store_true", help="Pass --no-warmup to benchmark_qa.py.")
    parser.add_argument("--dry-run", action="store_true", help="Print child commands without executing them.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def resolve_path(path_value: str | Path, *, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def child_env(*, provider: str) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env["BOXTALK_ENABLE_LLM_FALLBACK"] = "1"
    env.setdefault("BOXTALK_ENABLE_TABLE_LLM_REASONING", "1")
    env.setdefault("BOXTALK_ENABLE_FORMULA_LLM_REASONING", "1")
    env.setdefault("BOXTALK_ENABLE_FIGURE_LLM_REASONING", "1")
    env["BOXTALK_LLM_PROVIDER"] = provider
    return env


def run_command(command: list[str], *, env: dict[str, str], dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def ensure_dataset(manifest_path: Path, args: argparse.Namespace) -> None:
    if manifest_path.exists() and not args.recreate_dataset:
        return
    command = [args.python, str(ROOT / "scripts" / "create_llm_fallback_benchmark.py")]
    if manifest_path.parent != (ROOT / "data" / "llm_fallback_benchmark"):
        command.extend(["--output-dir", str(manifest_path.parent)])
    run_command(command, env=child_env(provider=args.llm_fallback_provider), dry_run=args.dry_run)


def ensure_index(*, doc: dict[str, Any], manifest_dir: Path, args: argparse.Namespace) -> Path:
    index_dir = resolve_path(doc["index_dir"], base_dir=manifest_dir)
    if (index_dir / "corpus.jsonl").exists() and not args.rebuild_index:
        return index_dir
    if args.skip_build:
        raise FileNotFoundError(f"Missing fallback benchmark index: {index_dir}")
    chunks_jsonl = resolve_path(doc["chunks_jsonl"], base_dir=manifest_dir)
    command = [
        args.python,
        str(ROOT / "scripts" / "build_retrieval_index.py"),
        "--chunks-jsonl",
        str(chunks_jsonl),
        "--output-dir",
        str(index_dir),
        "--dense-preset",
        args.dense_preset,
    ]
    run_command(command, env=child_env(provider=args.llm_fallback_provider), dry_run=args.dry_run)
    return index_dir


def validate_provider_or_exit(provider: str) -> dict[str, Any]:
    info = provider_runtime_info(provider)
    if info["provider"] == "openai-compatible" and not info["ready"]:
        missing = ", ".join(info["missing_envs"])
        raise SystemExit(
            "OpenAI-compatible fallback provider is not ready. Missing required env: "
            f"{missing}. Set BOXTALK_LLM_BASE_URL, BOXTALK_LLM_API_KEY, and BOXTALK_LLM_MODEL."
        )
    return info


def run_raw_benchmark(
    *,
    manifest_dir: Path,
    doc: dict[str, Any],
    index_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    queries = resolve_path(doc["queries"], base_dir=manifest_dir)
    command = [
        args.python,
        str(ROOT / "scripts" / "benchmark_qa.py"),
        "--index-dir",
        str(index_dir),
        "--queries",
        str(queries),
        "--output-dir",
        str(output_dir),
        "--config",
        args.standard_config,
        "--config",
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
    if args.no_warmup:
        command.append("--no-warmup")
    run_command(command, env=child_env(provider=args.llm_fallback_provider), dry_run=args.dry_run)


def mean_float(values: Iterable[float]) -> float:
    values_list = list(values)
    return statistics.mean(values_list) if values_list else 0.0


def ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def mode_key(row: dict[str, Any]) -> str:
    if row.get("fallback_reasoning_mode"):
        return str(row["fallback_reasoning_mode"])
    if row.get("expected_fallback_mode"):
        return f"expected:{row['expected_fallback_mode']}"
    return "not_called"


def build_comparison_rows(
    *,
    rows: list[dict[str, Any]],
    standard_config: str,
    fallback_config: str,
) -> list[dict[str, Any]]:
    standard_rows = {str(row["query_id"]): row for row in rows if row["config_name"] == standard_config}
    fallback_rows = {str(row["query_id"]): row for row in rows if row["config_name"] == fallback_config}
    ordered_query_ids = [query_id for query_id in standard_rows if query_id in fallback_rows]
    comparisons: list[dict[str, Any]] = []

    for query_id in ordered_query_ids:
        standard = standard_rows[query_id]
        fallback = fallback_rows[query_id]
        standard_success = bool(standard.get("end_to_end_success"))
        fallback_success = bool(fallback.get("end_to_end_success"))
        standard_answer_match = bool(standard.get("answer_match"))
        fallback_answer_match = bool(fallback.get("answer_match"))
        fallback_used = bool(fallback.get("fallback_used"))
        final_answer_source = str(fallback.get("final_answer_source") or "standard")
        comparisons.append(
            {
                "query_id": query_id,
                "question": standard.get("question"),
                "query_type": standard.get("query_type"),
                "fallback_category": standard.get("fallback_category"),
                "weak_standard_answer_case": bool(standard.get("weak_standard_answer_case", False)),
                "expected_modality": standard.get("expected_modality"),
                "expected_fallback_mode": standard.get("expected_fallback_mode"),
                "should_require_fallback": bool(standard.get("should_require_fallback", False)),
                "standard_decision": standard.get("decision"),
                "standard_answer": standard.get("answer"),
                "standard_answer_match": standard_answer_match,
                "standard_end_to_end_success": standard_success,
                "standard_hallucinated": bool(standard.get("hallucinated")),
                "standard_total_latency_ms": float(standard.get("total_latency_ms") or 0.0),
                "fallback_decision": fallback.get("decision"),
                "fallback_answer": fallback.get("answer"),
                "fallback_model_answer": fallback.get("fallback_answer"),
                "fallback_answer_match": fallback_answer_match,
                "fallback_end_to_end_success": fallback_success,
                "fallback_hallucinated": bool(fallback.get("hallucinated")),
                "fallback_total_latency_ms": float(fallback.get("total_latency_ms") or 0.0),
                "fallback_called": bool(fallback.get("fallback_called")),
                "fallback_used": fallback_used,
                "fallback_reason": fallback.get("fallback_reason"),
                "reasoning_mode": fallback.get("fallback_reasoning_mode"),
                "provider_name": fallback.get("provider_name") or fallback.get("fallback_provider") or "standard",
                "override_confidence": float(fallback.get("override_confidence") or 0.0),
                "final_answer_source": final_answer_source,
                "used_evidence_ids": list(fallback.get("fallback_used_evidence_ids") or []),
                "fallback_llm_called": bool(fallback.get("fallback_llm_called")),
                "success_delta": int(fallback_success) - int(standard_success),
                "answer_match_delta": int(fallback_answer_match) - int(standard_answer_match),
                "latency_delta_ms": float(fallback.get("total_latency_ms") or 0.0)
                - float(standard.get("total_latency_ms") or 0.0),
                "fallback_helped": fallback_success and not standard_success,
                "fallback_overrode_standard": final_answer_source != "standard",
                "table_rule_resolved": final_answer_source == "table_rule_fallback" and fallback_success,
                "table_llm_resolved": (
                    str(standard.get("expected_modality") or "") == "table"
                    and fallback_used
                    and bool(fallback.get("fallback_llm_called"))
                    and str(fallback.get("fallback_reasoning_mode") or "") == "table"
                    and fallback_success
                ),
            }
        )
    return comparisons


def summarize_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"query_count": 0}
    standard_success = mean_float(float(bool(row["standard_end_to_end_success"])) for row in rows)
    fallback_success = mean_float(float(bool(row["fallback_end_to_end_success"])) for row in rows)
    standard_answer_match = mean_float(float(bool(row["standard_answer_match"])) for row in rows)
    fallback_answer_match = mean_float(float(bool(row["fallback_answer_match"])) for row in rows)
    standard_hallucination = mean_float(float(bool(row["standard_hallucinated"])) for row in rows)
    fallback_hallucination = mean_float(float(bool(row["fallback_hallucinated"])) for row in rows)
    fallback_helped_count = sum(1 for row in rows if row["fallback_helped"])
    table_rows = [row for row in rows if str(row.get("expected_modality") or "") == "table"]
    return {
        "query_count": len(rows),
        "standard_success_rate": standard_success,
        "fallback_success_rate": fallback_success,
        "success_gain_vs_standard": fallback_success - standard_success,
        "standard_answer_match_rate": standard_answer_match,
        "fallback_answer_match_rate": fallback_answer_match,
        "answer_match_gain_vs_standard": fallback_answer_match - standard_answer_match,
        "fallback_call_rate": mean_float(float(bool(row["fallback_called"])) for row in rows),
        "fallback_used_rate": mean_float(float(bool(row["fallback_used"])) for row in rows),
        "fallback_helped_count": fallback_helped_count,
        "fallback_helped_rate": ratio(fallback_helped_count, len(rows)),
        "fallback_override_count": sum(1 for row in rows if row["fallback_overrode_standard"]),
        "latency_overhead_ms": mean_float(float(row["latency_delta_ms"]) for row in rows),
        "standard_hallucination_rate": standard_hallucination,
        "fallback_hallucination_rate": fallback_hallucination,
        "hallucination_delta": fallback_hallucination - standard_hallucination,
        "table_rule_resolved_count": sum(1 for row in rows if row["table_rule_resolved"]),
        "table_llm_resolved_count": sum(1 for row in rows if row["table_llm_resolved"]),
        "table_total_success": mean_float(float(bool(row["fallback_end_to_end_success"])) for row in table_rows),
        "table_question_count": len(table_rows),
        "multi_span_helped_count": sum(
            1
            for row in rows
            if str(row.get("expected_fallback_mode") or "") == "multi_span" and row["fallback_helped"]
        ),
        "comparison_helped_count": sum(
            1 for row in rows if str(row.get("query_type") or "") == "comparison" and row["fallback_helped"]
        ),
        "procedural_helped_count": sum(
            1 for row in rows if str(row.get("query_type") or "") == "procedural" and row["fallback_helped"]
        ),
        "requires_fallback_query_count": sum(1 for row in rows if row["should_require_fallback"]),
        "requires_fallback_helped_count": sum(
            1 for row in rows if row["should_require_fallback"] and row["fallback_helped"]
        ),
        "weak_standard_answer_query_count": sum(1 for row in rows if row["weak_standard_answer_case"]),
        "weak_standard_answer_helped_count": sum(
            1 for row in rows if row["weak_standard_answer_case"] and row["fallback_helped"]
        ),
    }


def grouped_summary(rows: list[dict[str, Any]], key_name: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get(key_name) or "unknown")
        groups.setdefault(key, []).append(row)
    return {key: summarize_comparison(group_rows) for key, group_rows in sorted(groups.items())}


def json_for_csv(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_for_csv(value) for key, value in row.items()})


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    aggregate = summary["aggregate"]
    provider = summary["provider"]
    lines = [
        "# LLM Fallback Benchmark",
        "",
        f"- Provider: `{provider['provider']}`",
        f"- Provider ready: `{provider['ready']}`",
        f"- Benchmark mode: `{summary['benchmark_mode']}`",
        f"- Standard config: `{summary['standard_config']}`",
        f"- Fallback config: `{summary['fallback_config']}`",
        f"- Query count: {aggregate.get('query_count', 0)}",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Success gain vs standard | {aggregate.get('success_gain_vs_standard', 0.0):.3f} |",
        f"| Answer match gain vs standard | {aggregate.get('answer_match_gain_vs_standard', 0.0):.3f} |",
        f"| Fallback call rate | {aggregate.get('fallback_call_rate', 0.0):.3f} |",
        f"| Fallback used rate | {aggregate.get('fallback_used_rate', 0.0):.3f} |",
        f"| Fallback helped count | {aggregate.get('fallback_helped_count', 0)} |",
        f"| Fallback override count | {aggregate.get('fallback_override_count', 0)} |",
        f"| Latency overhead ms | {aggregate.get('latency_overhead_ms', 0.0):.1f} |",
        f"| Hallucination delta | {aggregate.get('hallucination_delta', 0.0):.3f} |",
        f"| Table rule resolved count | {aggregate.get('table_rule_resolved_count', 0)} |",
        f"| Table LLM resolved count | {aggregate.get('table_llm_resolved_count', 0)} |",
        "",
        "## By Expected Modality",
        "",
        "| Modality | Queries | Success Gain | Call Rate | Used Rate | Helped |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in summary.get("by_expected_modality", {}).items():
        lines.append(
            "| {name} | {queries} | {gain:.3f} | {call:.3f} | {used:.3f} | {helped} |".format(
                name=name,
                queries=metrics.get("query_count", 0),
                gain=metrics.get("success_gain_vs_standard", 0.0),
                call=metrics.get("fallback_call_rate", 0.0),
                used=metrics.get("fallback_used_rate", 0.0),
                helped=metrics.get("fallback_helped_count", 0),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else (ROOT / args.manifest)
    ensure_dataset(manifest_path, args)
    if args.dry_run:
        return

    if not manifest_path.exists():
        raise SystemExit(f"Fallback benchmark manifest not found: {manifest_path}")

    provider_info = validate_provider_or_exit(args.llm_fallback_provider)
    manifest = load_json(manifest_path)
    doc = manifest.get("document")
    if not isinstance(doc, dict):
        raise SystemExit(f"{manifest_path} must define a document object")
    manifest_dir = manifest_path.parent
    index_dir = ensure_index(doc=doc, manifest_dir=manifest_dir, args=args)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (ROOT / RESULTS_ROOT / timestamp)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_output_dir = output_dir / "raw_benchmark"

    run_raw_benchmark(
        manifest_dir=manifest_dir,
        doc=doc,
        index_dir=index_dir,
        output_dir=raw_output_dir,
        args=args,
    )

    qa_summary = load_json(raw_output_dir / "qa_summary.json")
    per_question = json.loads((raw_output_dir / "per_question.json").read_text(encoding="utf-8"))
    if not isinstance(per_question, list):
        raise SystemExit(f"{raw_output_dir / 'per_question.json'} must contain a JSON list")

    comparison_rows = build_comparison_rows(
        rows=per_question,
        standard_config=args.standard_config,
        fallback_config=args.fallback_config,
    )
    benchmark_mode = "plumbing_check" if provider_info["provider"] == "dummy" else "real_provider"
    summary = {
        "timestamp_utc": timestamp,
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "benchmark_mode": benchmark_mode,
        "document": doc,
        "provider": provider_info,
        "feature_flags": {
            "enable_llm_fallback": True,
            "enable_table_llm_reasoning": True,
            "enable_formula_llm_reasoning": True,
            "enable_figure_llm_reasoning": True,
            "llm_fallback_sufficiency_threshold": args.llm_fallback_sufficiency_threshold,
            "llm_fallback_min_confidence": args.llm_fallback_min_confidence,
            "llm_fallback_min_override_confidence": args.llm_fallback_min_override_confidence,
        },
        "standard_config": args.standard_config,
        "fallback_config": args.fallback_config,
        "raw_benchmark_output": str(raw_output_dir),
        "raw_benchmark_summary": qa_summary.get("configs", {}),
        "aggregate": summarize_comparison(comparison_rows),
        "by_reasoning_mode": grouped_summary(
            [{**row, "reasoning_mode_group": mode_key(row)} for row in comparison_rows],
            "reasoning_mode_group",
        ),
        "by_expected_fallback_mode": grouped_summary(comparison_rows, "expected_fallback_mode"),
        "by_query_type": grouped_summary(comparison_rows, "query_type"),
        "by_expected_modality": grouped_summary(comparison_rows, "expected_modality"),
        "by_category": grouped_summary(comparison_rows, "fallback_category"),
        "requires_fallback": {
            "true": summarize_comparison([row for row in comparison_rows if row["should_require_fallback"]]),
            "false": summarize_comparison([row for row in comparison_rows if not row["should_require_fallback"]]),
        },
        "weak_standard_answer": summarize_comparison(
            [row for row in comparison_rows if row["weak_standard_answer_case"]]
        ),
    }

    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "comparison_per_question.json").write_text(
        json.dumps(comparison_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(output_dir / "comparison_per_question.csv", comparison_rows)
    write_markdown(output_dir / "README.md", summary)
    print(output_dir)


if __name__ == "__main__":
    main()
