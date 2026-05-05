from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_USER_SUITE_SUMMARY = Path("results/user_pdf_benchmark_suite/current/suite_summary.json")
READINESS_ROOT = Path("results/retrieval_readiness")


@dataclass(slots=True)
class GateResult:
    name: str
    actual: object
    expected: str
    passed: bool
    skipped: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail when locked baseline metrics regress.")
    parser.add_argument("--user-suite-summary", type=Path, default=DEFAULT_USER_SUITE_SUMMARY)
    parser.add_argument("--readiness-report", type=Path, default=None)
    parser.add_argument("--skip-user-suite", action="store_true")
    parser.add_argument("--skip-readiness", action="store_true")
    parser.add_argument("--require-production-ready", action="store_true")
    parser.add_argument(
        "--fallback-summary",
        type=Path,
        default=None,
        help=(
            "Optional experimental grounded_llm_fallback benchmark summary. "
            "Accepts repeat_summary.json or comparison_summary.json."
        ),
    )
    parser.add_argument("--fallback-success-gain-min", type=float, default=0.0)
    parser.add_argument("--fallback-answer-gain-min", type=float, default=0.0)
    parser.add_argument("--fallback-groundedness-min", type=float, default=1.0)
    parser.add_argument("--fallback-hallucination-delta-max", type=float, default=0.0)
    parser.add_argument("--fallback-table-llm-resolved-min", type=float, default=0.0)
    parser.add_argument(
        "--min-suite-unique-questions",
        "--min-suite-queries",
        dest="min_suite_unique_questions",
        type=int,
        default=100,
    )
    parser.add_argument("--min-suite-documents", type=int, default=3)
    parser.add_argument("--bm25-success-min", type=float, default=0.82)
    parser.add_argument("--routed-success-min", type=float, default=0.83)
    parser.add_argument("--routed-grounded-min", type=float, default=1.0)
    parser.add_argument("--routed-hallucination-max", type=float, default=0.0)
    parser.add_argument("--scientific-routed-success-min", type=float, default=0.95)
    parser.add_argument("--scientific-routed-evidence-min", type=float, default=0.95)
    parser.add_argument("--scientific-routed-hallucination-max", type=float, default=0.0)
    parser.add_argument("--tolerance", type=float, default=1e-9)
    parser.add_argument("--write-report", type=Path, default=None)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Missing required benchmark artifact: {display_path(resolved)}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{display_path(resolved)} must contain a JSON object")
    return payload


def latest_readiness_report() -> Path:
    root = resolve_path(READINESS_ROOT)
    candidates = sorted(root.glob("*/readiness_report.json"), key=lambda item: item.parent.name)
    if not candidates:
        raise FileNotFoundError(f"No readiness reports found under {display_path(root)}")
    return candidates[-1]


def display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def metric(summary: dict[str, Any], config_name: str, metric_name: str) -> float:
    by_config = summary.get("by_config")
    if not isinstance(by_config, dict):
        raise KeyError("suite_summary.json is missing by_config")
    config = by_config.get(config_name)
    if not isinstance(config, dict):
        raise KeyError(f"suite_summary.json is missing by_config.{config_name}")
    value = config.get(metric_name)
    if value is None:
        raise KeyError(f"suite_summary.json is missing by_config.{config_name}.{metric_name}")
    return float(value)


def grouped_metric(
    summary: dict[str, Any],
    group_key: str,
    group_name: str,
    config_name: str,
    metric_name: str,
) -> float:
    groups = summary.get(group_key)
    if not isinstance(groups, dict):
        raise KeyError(f"suite_summary.json is missing {group_key}")
    payload = groups.get(f"{group_name} / {config_name}")
    if not isinstance(payload, dict):
        raise KeyError(f"suite_summary.json is missing {group_key}.{group_name} / {config_name}")
    value = payload.get(metric_name)
    if value is None:
        raise KeyError(f"suite_summary.json is missing {group_key}.{group_name} / {config_name}.{metric_name}")
    return float(value)


def count_metric(summary: dict[str, Any], metric_name: str) -> int:
    overall = summary.get("overall")
    if isinstance(overall, dict) and metric_name in overall:
        return int(overall[metric_name])
    by_config = summary.get("by_config")
    if isinstance(by_config, dict) and by_config:
        first_config = next(iter(by_config.values()))
        if isinstance(first_config, dict) and metric_name in first_config:
            return int(first_config[metric_name])
    raise KeyError(f"suite_summary.json is missing {metric_name}")


def min_gate(name: str, actual: float, expected: float, tolerance: float) -> GateResult:
    return GateResult(name, actual, f">= {expected}", actual + tolerance >= expected)


def greater_gate(name: str, actual: float, expected: float, tolerance: float) -> GateResult:
    return GateResult(name, actual, f"> {expected}", actual > expected + tolerance)


def max_gate(name: str, actual: float, expected: float, tolerance: float) -> GateResult:
    return GateResult(name, actual, f"<= {expected}", actual <= expected + tolerance)


def bool_gate(name: str, actual: object, expected: bool = True) -> GateResult:
    return GateResult(name, actual, f"== {expected}", bool(actual) is expected)


def skipped_gate(name: str, reason: str) -> GateResult:
    return GateResult(name, reason, "provided to run experimental fallback gate", True, skipped=True)


def check_user_suite(summary: dict[str, Any], args: argparse.Namespace) -> list[GateResult]:
    tolerance = float(args.tolerance)
    return [
        min_gate(
            "user_suite.unique_question_count",
            count_metric(summary, "unique_question_count"),
            args.min_suite_unique_questions,
            tolerance,
        ),
        min_gate("user_suite.document_count", count_metric(summary, "document_count"), args.min_suite_documents, tolerance),
        min_gate(
            "user_suite.bm25_only.end_to_end_success_rate",
            metric(summary, "bm25_only", "end_to_end_success_rate"),
            args.bm25_success_min,
            tolerance,
        ),
        min_gate(
            "user_suite.routed_grounded.end_to_end_success_rate",
            metric(summary, "routed_grounded", "end_to_end_success_rate"),
            args.routed_success_min,
            tolerance,
        ),
        min_gate(
            "user_suite.routed_grounded.grounded_rate",
            metric(summary, "routed_grounded", "grounded_rate"),
            args.routed_grounded_min,
            tolerance,
        ),
        max_gate(
            "user_suite.routed_grounded.hallucination_rate",
            metric(summary, "routed_grounded", "hallucination_rate"),
            args.routed_hallucination_max,
            tolerance,
        ),
        min_gate(
            "user_suite.scientific_paper.routed_grounded.end_to_end_success_rate",
            grouped_metric(
                summary,
                "by_document_type_and_config",
                "scientific_paper",
                "routed_grounded",
                "end_to_end_success_rate",
            ),
            args.scientific_routed_success_min,
            tolerance,
        ),
        min_gate(
            "user_suite.scientific_paper.routed_grounded.evidence_match_rate",
            grouped_metric(
                summary,
                "by_document_type_and_config",
                "scientific_paper",
                "routed_grounded",
                "evidence_match_rate",
            ),
            args.scientific_routed_evidence_min,
            tolerance,
        ),
        max_gate(
            "user_suite.scientific_paper.routed_grounded.hallucination_rate",
            grouped_metric(
                summary,
                "by_document_type_and_config",
                "scientific_paper",
                "routed_grounded",
                "hallucination_rate",
            ),
            args.scientific_routed_hallucination_max,
            tolerance,
        ),
    ]


def check_readiness(report: dict[str, Any], args: argparse.Namespace) -> list[GateResult]:
    verdict = report.get("verdict") or {}
    scientific = report.get("scientific") or {}
    gates = scientific.get("gates") or {}

    results = [
        bool_gate("readiness.verdict.scientific_ready", verdict.get("scientific_ready")),
        bool_gate(
            "readiness.verdict.retrieval_ready_for_prototyping",
            verdict.get("retrieval_ready_for_prototyping"),
        ),
    ]
    if args.require_production_ready:
        results.append(
            bool_gate(
                "readiness.verdict.retrieval_ready_for_production",
                verdict.get("retrieval_ready_for_production"),
            )
        )

    if not isinstance(gates, dict) or not gates:
        results.append(GateResult("readiness.scientific.gates_present", False, "== True", False))
        return results

    for run_name, gate_payload in sorted(gates.items()):
        if not isinstance(gate_payload, dict):
            results.append(GateResult(f"readiness.scientific.{run_name}", gate_payload, "all pass", False))
            continue
        for gate_name, value in sorted(gate_payload.items()):
            results.append(bool_gate(f"readiness.scientific.{run_name}.{gate_name}", value))
    return results


def fallback_metric(summary: dict[str, Any], metric_name: str, *, repeat_field: str) -> float:
    metric_summary = summary.get("metric_summary")
    if isinstance(metric_summary, dict):
        metric_payload = metric_summary.get(metric_name)
        if not isinstance(metric_payload, dict):
            raise KeyError(f"fallback repeat_summary.json is missing metric_summary.{metric_name}")
        value = metric_payload.get(repeat_field)
        if value is None:
            raise KeyError(f"fallback repeat_summary.json is missing metric_summary.{metric_name}.{repeat_field}")
        return float(value)

    aggregate = summary.get("aggregate")
    if isinstance(aggregate, dict):
        value = aggregate.get(metric_name)
        if value is None and metric_name == "groundedness":
            value = aggregate.get("fallback_grounded_rate")
        if value is None:
            raw_summary = summary.get("raw_benchmark_summary")
            fallback_config = summary.get("fallback_config")
            if isinstance(raw_summary, dict) and isinstance(fallback_config, str):
                fallback_payload = raw_summary.get(fallback_config)
                if isinstance(fallback_payload, dict) and metric_name == "groundedness":
                    value = fallback_payload.get("grounded_rate")
        if value is not None:
            return float(value)
        raise KeyError(f"fallback comparison_summary.json is missing aggregate.{metric_name}")

    raise KeyError("fallback summary must contain metric_summary or aggregate")


def check_fallback_experimental(summary: dict[str, Any], args: argparse.Namespace) -> list[GateResult]:
    tolerance = float(args.tolerance)
    return [
        greater_gate(
            "fallback_experimental.success_gain_vs_standard",
            fallback_metric(summary, "success_gain_vs_standard", repeat_field="min"),
            args.fallback_success_gain_min,
            tolerance,
        ),
        min_gate(
            "fallback_experimental.answer_match_gain_vs_standard",
            fallback_metric(summary, "answer_match_gain_vs_standard", repeat_field="min"),
            args.fallback_answer_gain_min,
            tolerance,
        ),
        min_gate(
            "fallback_experimental.groundedness",
            fallback_metric(summary, "groundedness", repeat_field="min"),
            args.fallback_groundedness_min,
            tolerance,
        ),
        max_gate(
            "fallback_experimental.hallucination_delta",
            fallback_metric(summary, "hallucination_delta", repeat_field="max"),
            args.fallback_hallucination_delta_max,
            tolerance,
        ),
        greater_gate(
            "fallback_experimental.table_llm_resolved_count",
            fallback_metric(summary, "table_llm_resolved_count", repeat_field="min"),
            args.fallback_table_llm_resolved_min,
            tolerance,
        ),
    ]


def print_results(results: list[GateResult]) -> None:
    for result in results:
        status = "SKIP" if result.skipped else ("PASS" if result.passed else "FAIL")
        print(f"{status} {result.name}: actual={result.actual!r} expected {result.expected}")


def write_report(path: Path, results: list[GateResult]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "passed": all(result.passed for result in results),
        "gates": [
            {
                "name": result.name,
                "actual": result.actual,
                "expected": result.expected,
                "passed": result.passed,
                "skipped": result.skipped,
            }
            for result in results
        ],
    }
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    results: list[GateResult] = []

    if not args.skip_user_suite:
        user_summary = load_json(args.user_suite_summary)
        results.extend(check_user_suite(user_summary, args))

    if not args.skip_readiness:
        readiness_path = args.readiness_report or latest_readiness_report()
        readiness_report = load_json(readiness_path)
        results.extend(check_readiness(readiness_report, args))

    if args.fallback_summary is None:
        results.append(skipped_gate("fallback_experimental.summary", "not provided"))
    else:
        fallback_summary_path = resolve_path(args.fallback_summary)
        print(f"INFO fallback_experimental.summary_path={display_path(fallback_summary_path)}")
        fallback_summary = load_json(args.fallback_summary)
        results.extend(check_fallback_experimental(fallback_summary, args))

    print_results(results)
    if args.write_report:
        write_report(args.write_report, results)

    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
