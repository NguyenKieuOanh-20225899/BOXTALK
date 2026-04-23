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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail when locked baseline metrics regress.")
    parser.add_argument("--user-suite-summary", type=Path, default=DEFAULT_USER_SUITE_SUMMARY)
    parser.add_argument("--readiness-report", type=Path, default=None)
    parser.add_argument("--skip-user-suite", action="store_true")
    parser.add_argument("--skip-readiness", action="store_true")
    parser.add_argument("--require-production-ready", action="store_true")
    parser.add_argument(
        "--min-suite-unique-questions",
        "--min-suite-queries",
        dest="min_suite_unique_questions",
        type=int,
        default=100,
    )
    parser.add_argument("--min-suite-documents", type=int, default=3)
    parser.add_argument("--bm25-success-min", type=float, default=0.77)
    parser.add_argument("--routed-success-min", type=float, default=0.72)
    parser.add_argument("--routed-grounded-min", type=float, default=1.0)
    parser.add_argument("--routed-hallucination-max", type=float, default=0.0)
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


def max_gate(name: str, actual: float, expected: float, tolerance: float) -> GateResult:
    return GateResult(name, actual, f"<= {expected}", actual <= expected + tolerance)


def bool_gate(name: str, actual: object, expected: bool = True) -> GateResult:
    return GateResult(name, actual, f"== {expected}", bool(actual) is expected)


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


def print_results(results: list[GateResult]) -> None:
    for result in results:
        status = "PASS" if result.passed else "FAIL"
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
