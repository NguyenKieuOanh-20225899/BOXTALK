from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_USER_SUITE_SUMMARY = (
    ROOT / "results" / "user_pdf_benchmark_suite" / "llm_fallback_gate_recheck" / "suite_summary.json"
)
DEFAULT_RETRIEVAL_SMOKE_SUMMARY = (
    ROOT / "results" / "retrieval_benchmark" / "smoke_real_minilm_after" / "benchmark_summary.json"
)
DEFAULT_READINESS_REPORT = (
    ROOT / "results" / "retrieval_readiness" / "20260420T150853Z" / "readiness_report.json"
)
DEFAULT_TABLE_REASONING_SUMMARY = (
    ROOT
    / "results"
    / "llm_fallback_benchmark"
    / "table_reasoning_ollama_after_shape_gate"
    / "comparison_summary.json"
)
DEFAULT_FALLBACK_REPEAT_SUMMARY = (
    ROOT
    / "results"
    / "llm_fallback_benchmark"
    / "table_patch_ollama_repeats_gpu"
    / "repeat_summary.json"
)
DEFAULT_BEIR_ROOT = ROOT / "results" / "beir_retrieval_benchmark"
DEFAULT_OUTPUT_MD = ROOT / "docs" / "THESIS_READINESS_REPORT.md"
DEFAULT_OUTPUT_JSON = ROOT / "results" / "thesis_readiness_report" / "summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a thesis-readiness report for the PDF retrieval and grounded QA project "
            "from existing benchmark summaries."
        )
    )
    parser.add_argument("--user-suite-summary", type=Path, default=DEFAULT_USER_SUITE_SUMMARY)
    parser.add_argument("--retrieval-smoke-summary", type=Path, default=DEFAULT_RETRIEVAL_SMOKE_SUMMARY)
    parser.add_argument("--readiness-report", type=Path, default=DEFAULT_READINESS_REPORT)
    parser.add_argument("--table-reasoning-summary", type=Path, default=DEFAULT_TABLE_REASONING_SUMMARY)
    parser.add_argument("--fallback-repeat-summary", type=Path, default=DEFAULT_FALLBACK_REPEAT_SUMMARY)
    parser.add_argument("--beir-root", type=Path, default=DEFAULT_BEIR_ROOT)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT / path


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def load_optional_json(path: Path) -> tuple[dict[str, Any] | None, str]:
    resolved = resolve_path(path)
    if not resolved.exists():
        return None, rel(resolved)
    with resolved.open("r", encoding="utf-8") as f:
        return json.load(f), rel(resolved)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_text(path: Path, content: str) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")


def as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def metric_mean(payload: dict[str, Any], metric: str, default: float = 0.0) -> float:
    summary = payload.get("metric_summary", {})
    value = summary.get(metric, {})
    if isinstance(value, dict):
        return as_float(value.get("mean"), default)
    return as_float(value, default)


def metric_min(payload: dict[str, Any], metric: str, default: float = 0.0) -> float:
    summary = payload.get("metric_summary", {})
    value = summary.get(metric, {})
    if isinstance(value, dict):
        return as_float(value.get("min"), default)
    return as_float(value, default)


def metric_max(payload: dict[str, Any], metric: str, default: float = 0.0) -> float:
    summary = payload.get("metric_summary", {})
    value = summary.get(metric, {})
    if isinstance(value, dict):
        return as_float(value.get("max"), default)
    return as_float(value, default)


def summarize_user_suite(payload: dict[str, Any] | None, path: str) -> dict[str, Any]:
    if payload is None:
        return {"available": False, "path": path}

    by_config = payload.get("by_config", {})
    bm25 = by_config.get("bm25_only", {})
    routed = by_config.get("routed_grounded", {})
    return {
        "available": True,
        "path": path,
        "suite_name": payload.get("suite_name"),
        "document_count": len(payload.get("documents", [])),
        "bm25_only": extract_qa_metrics(bm25),
        "routed_grounded": extract_qa_metrics(routed),
        "routed_gain_vs_bm25": {
            "answer_match_rate": as_float(routed.get("answer_match_rate"))
            - as_float(bm25.get("answer_match_rate")),
            "end_to_end_success_rate": as_float(routed.get("end_to_end_success_rate"))
            - as_float(bm25.get("end_to_end_success_rate")),
        },
    }


def extract_qa_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "query_count": as_int(metrics.get("query_count")),
        "answer_match_rate": as_float(metrics.get("answer_match_rate")),
        "evidence_match_rate": as_float(metrics.get("evidence_match_rate")),
        "grounded_rate": as_float(metrics.get("grounded_rate")),
        "end_to_end_success_rate": as_float(metrics.get("end_to_end_success_rate")),
        "hallucination_rate": as_float(metrics.get("hallucination_rate")),
        "avg_total_latency_ms": as_float(metrics.get("avg_total_latency_ms")),
    }


def summarize_retrieval_smoke(payload: dict[str, Any] | None, path: str) -> dict[str, Any]:
    if payload is None:
        return {"available": False, "path": path}

    strategies: dict[str, dict[str, Any]] = {}
    for name, metrics in payload.get("strategies", {}).items():
        top_k = as_int(payload.get("top_k"), 5)
        strategies[name] = {
            f"hit_at_{top_k}": as_float(metrics.get(f"hit_at_{top_k}", metrics.get("hit_at_k"))),
            f"recall_at_{top_k}": as_float(
                metrics.get(f"recall_at_{top_k}", metrics.get("recall_at_k"))
            ),
            f"mrr_at_{top_k}": as_float(metrics.get(f"mrr_at_{top_k}", metrics.get("mrr_at_k"))),
            f"ndcg_at_{top_k}": as_float(metrics.get(f"ndcg_at_{top_k}", metrics.get("ndcg_at_k"))),
            "avg_latency_ms": as_float(metrics.get("avg_latency_ms")),
        }

    return {
        "available": True,
        "path": path,
        "top_k": as_int(payload.get("top_k"), 5),
        "query_count": max((as_int(m.get("query_count")) for m in payload.get("strategies", {}).values()), default=0),
        "strategies": strategies,
        "strategy_count": len(strategies),
    }


def summarize_scientific_readiness(payload: dict[str, Any] | None, path: str) -> dict[str, Any]:
    if payload is None:
        return {"available": False, "path": path}

    scientific = payload.get("scientific", {})
    stability = scientific.get("stability", {})
    verdict = payload.get("verdict", {})
    return {
        "available": True,
        "path": path,
        "generated_at_utc": payload.get("metadata", {}).get("generated_at_utc"),
        "run_count": len(scientific.get("runs", [])),
        "success_rate_min": as_float(stability.get("success_rate_min")),
        "iou50_micro_f1_min": as_float(stability.get("iou50_micro_f1_min")),
        "iou75_micro_f1_min": as_float(stability.get("iou75_micro_f1_min")),
        "latency_p95_sec_max": as_float(stability.get("latency_p95_sec_max")),
        "backend_consistent": bool(stability.get("backend_consistent")),
        "scientific_ready": bool(verdict.get("scientific_ready")),
        "production_ready": bool(verdict.get("production_ready")),
        "blockers": verdict.get("blockers", []),
        "strengths": verdict.get("strengths", []),
    }


def summarize_fallback_repeat(payload: dict[str, Any] | None, path: str) -> dict[str, Any]:
    if payload is None:
        return {"available": False, "path": path}

    return {
        "available": True,
        "path": path,
        "benchmark_mode": payload.get("benchmark_mode"),
        "provider": payload.get("provider", {}).get("provider"),
        "model": payload.get("provider", {}).get("model"),
        "repeat": as_int(payload.get("repeat")),
        "success_gain_vs_standard_mean": metric_mean(payload, "success_gain_vs_standard"),
        "success_gain_vs_standard_min": metric_min(payload, "success_gain_vs_standard"),
        "answer_match_gain_vs_standard_mean": metric_mean(payload, "answer_match_gain_vs_standard"),
        "groundedness_min": metric_min(payload, "groundedness"),
        "hallucination_delta_max": metric_max(payload, "hallucination_delta"),
        "fallback_call_rate_mean": metric_mean(payload, "fallback_call_rate"),
        "fallback_used_rate_mean": metric_mean(payload, "fallback_used_rate"),
        "table_rule_resolved_count_mean": metric_mean(payload, "table_rule_resolved_count"),
        "table_llm_resolved_count_min": metric_min(payload, "table_llm_resolved_count"),
        "table_total_success_mean": metric_mean(payload, "table_total_success"),
        "latency_overhead_ms_mean": metric_mean(payload, "latency_overhead_ms"),
    }


def summarize_table_reasoning(payload: dict[str, Any] | None, path: str) -> dict[str, Any]:
    if payload is None:
        return {"available": False, "path": path}

    metrics = payload.get("metrics") or payload.get("aggregate", {})
    return {
        "available": True,
        "path": path,
        "benchmark_mode": payload.get("benchmark_mode"),
        "provider": payload.get("provider", {}).get("provider"),
        "model": payload.get("provider", {}).get("model"),
        "query_count": as_int(metrics.get("query_count")),
        "success_gain_vs_standard": as_float(metrics.get("success_gain_vs_standard")),
        "answer_match_gain_vs_standard": as_float(metrics.get("answer_match_gain_vs_standard")),
        "groundedness": as_float(metrics.get("groundedness")),
        "hallucination_delta": as_float(metrics.get("hallucination_delta")),
        "fallback_call_rate": as_float(metrics.get("fallback_call_rate")),
        "fallback_used_rate": as_float(metrics.get("fallback_used_rate")),
        "table_rule_resolved_count": as_int(metrics.get("table_rule_resolved_count")),
        "table_llm_attempt_count": as_int(metrics.get("table_llm_attempt_count")),
        "table_llm_resolved_count": as_int(metrics.get("table_llm_resolved_count")),
        "table_total_success": as_float(metrics.get("table_total_success")),
        "table_text_reasoning_success": as_float(metrics.get("table_text_reasoning_success")),
        "numerical_reasoning_success": as_float(metrics.get("numerical_reasoning_success")),
        "fact_verification_success": as_float(metrics.get("fact_verification_success")),
        "table_resolution_breakdown": metrics.get("table_resolution_breakdown", {}),
    }


def summarize_beir(beir_root: Path) -> dict[str, Any]:
    root = resolve_path(beir_root)
    if not root.exists():
        return {"available": False, "path": rel(root), "runs": []}

    runs: list[dict[str, Any]] = []
    for summary_path in sorted(root.glob("*/beir_summary.json")):
        payload, _ = load_optional_json(summary_path)
        if payload is None:
            continue
        strategy_name = next(iter(payload.get("strategies", {}).keys()), None)
        metrics = payload.get("strategies", {}).get(strategy_name or "", {})
        runs.append(
            {
                "path": rel(summary_path),
                "dataset": payload.get("dataset"),
                "strategy": strategy_name,
                "query_count": as_int(payload.get("query_count")),
                "corpus_count": as_int(payload.get("corpus_count")),
                "hit_at_k": as_float(metrics.get("hit_at_k")),
                "recall_at_k": as_float(metrics.get("recall_at_k")),
                "mrr_at_k": as_float(metrics.get("mrr_at_k")),
                "ndcg_at_k": as_float(metrics.get("ndcg_at_k")),
                "avg_latency_ms": as_float(metrics.get("avg_latency_ms")),
            }
        )

    return {
        "available": bool(runs),
        "path": rel(root),
        "runs": runs,
        "run_count": len(runs),
    }


def build_verdict(summary: dict[str, Any]) -> dict[str, Any]:
    user_suite = summary["user_pdf_suite"]
    retrieval = summary["retrieval_smoke"]
    scientific = summary["scientific_readiness"]
    fallback = summary["fallback_repeat"]
    table = summary["table_reasoning"]
    beir = summary["beir"]

    strengths: list[str] = []
    limitations: list[str] = []
    required_before_final: list[str] = []

    main_qa_ready = False
    if user_suite.get("available"):
        routed = user_suite.get("routed_grounded", {})
        main_qa_ready = (
            routed.get("end_to_end_success_rate", 0.0) >= 0.83
            and routed.get("grounded_rate", 0.0) >= 1.0
            and routed.get("hallucination_rate", 1.0) <= 0.0
        )
        if main_qa_ready:
            strengths.append("Main routed_grounded QA has strong grounded user-PDF benchmark results.")
        else:
            limitations.append("Main routed_grounded QA does not meet the thesis-ready evidence threshold.")
    else:
        required_before_final.append("Run the user PDF QA suite and keep its summary artifact.")

    retrieval_comparison_ready = retrieval.get("available") and retrieval.get("strategy_count", 0) >= 3
    if retrieval_comparison_ready:
        strengths.append("Retrieval benchmark compares lexical, dense, hybrid, and rerank-style strategies.")
    else:
        required_before_final.append("Run retrieval benchmark with at least BM25, dense, and hybrid strategies.")

    scientific_ready = bool(scientific.get("available") and scientific.get("scientific_ready"))
    if scientific_ready:
        strengths.append("Scientific/PubTables ingest readiness has stable sampled evidence.")
    else:
        limitations.append("Scientific ingest readiness evidence is missing or below target.")

    beir_ready = bool(beir.get("available") and beir.get("run_count", 0) >= 1)
    if beir_ready:
        strengths.append("External-style retrieval evidence exists through BEIR/SciFact samples.")
    else:
        limitations.append("BEIR-style retrieval comparison is missing.")

    fallback_experimental_ready = (
        fallback.get("available")
        and fallback.get("success_gain_vs_standard_min", 0.0) > 0.0
        and fallback.get("groundedness_min", 0.0) >= 1.0
        and fallback.get("hallucination_delta_max", 1.0) <= 0.0
        and fallback.get("table_llm_resolved_count_min", 0.0) > 0.0
    )
    if fallback_experimental_ready:
        strengths.append("Experimental grounded LLM fallback has stable gain without groundedness regression.")
    else:
        limitations.append("LLM fallback should remain experimental unless its repeat gate passes.")

    table_benchmark_ready = bool(table.get("available") and table.get("query_count", 0) >= 30)
    table_llm_gap = bool(table_benchmark_ready and table.get("table_llm_resolved_count", 0) <= 0)
    if table_benchmark_ready:
        strengths.append("Extended table benchmark covers lookup, interval, numerical, and verification cases.")
    else:
        required_before_final.append("Generate/run the extended table reasoning benchmark.")
    if table_llm_gap:
        limitations.append(
            "Extended table benchmark still shows no resolved LLM-table wins; present this as a limitation."
        )

    probe_eval_ready = (ROOT / "results" / "eval_summary.json").exists()
    if not probe_eval_ready:
        limitations.append("Probe-classification evaluation artifact is not present.")

    production_claim_ready = bool(scientific.get("production_ready"))
    if not production_claim_ready:
        limitations.append("Do not claim production readiness without labeled production-PDF evidence.")

    research_prototype_ready = all(
        [main_qa_ready, retrieval_comparison_ready, scientific_ready, table_benchmark_ready]
    )

    if research_prototype_ready:
        recommended_positioning = (
            "Ready to position as a research prototype for retrieval and grounded QA over PDFs, "
            "with LLM fallback/table reasoning documented as experimental."
        )
    else:
        recommended_positioning = (
            "Not ready to freeze the thesis artifact yet; complete required benchmark artifacts first."
        )

    if not required_before_final:
        required_before_final.extend(
            [
                "Freeze benchmark result folders used in the final report.",
                "Add limitations/future-work text for table LLM reasoning and production-readiness scope.",
                "Prepare a reproducible command appendix for all reported metrics.",
            ]
        )

    return {
        "research_prototype_ready": research_prototype_ready,
        "production_claim_ready": production_claim_ready,
        "main_qa_ready": main_qa_ready,
        "retrieval_comparison_ready": retrieval_comparison_ready,
        "scientific_ready": scientific_ready,
        "beir_ready": beir_ready,
        "fallback_experimental_ready": fallback_experimental_ready,
        "table_benchmark_ready": table_benchmark_ready,
        "probe_eval_ready": probe_eval_ready,
        "strengths": strengths,
        "limitations": limitations,
        "required_before_final": required_before_final,
        "recommended_positioning": recommended_positioning,
    }


def fmt_pct(value: Any) -> str:
    return f"{as_float(value) * 100:.1f}%"


def fmt_num(value: Any, digits: int = 3) -> str:
    return f"{as_float(value):.{digits}f}"


def status_label(value: bool) -> str:
    return "PASS" if value else "NOT READY"


def build_markdown(summary: dict[str, Any]) -> str:
    verdict = summary["verdict"]
    user_suite = summary["user_pdf_suite"]
    retrieval = summary["retrieval_smoke"]
    scientific = summary["scientific_readiness"]
    fallback = summary["fallback_repeat"]
    table = summary["table_reasoning"]
    beir = summary["beir"]
    generated_at = summary["generated_at_utc"]

    lines: list[str] = [
        "# Thesis Readiness Report",
        "",
        f"Generated at UTC: `{generated_at}`",
        "",
        "Research topic: `Nghiên cứu các kĩ thuật truy xuất và hỏi đáp thông tin trên tài liệu PDF`.",
        "",
        "This report maps current benchmark artifacts to thesis-level claims. It is intentionally conservative: main QA claims stay separate from experimental grounded LLM fallback and table reasoning.",
        "",
        "## Verdict",
        "",
        f"- Research prototype readiness: **{status_label(verdict['research_prototype_ready'])}**",
        f"- Production-readiness claim: **{status_label(verdict['production_claim_ready'])}**",
        f"- Recommended positioning: {verdict['recommended_positioning']}",
        "",
        "## Evidence Map",
        "",
        "| Area | Status | Key evidence | Source |",
        "| --- | --- | --- | --- |",
        evidence_row_main_qa(user_suite, verdict["main_qa_ready"]),
        evidence_row_retrieval(retrieval, verdict["retrieval_comparison_ready"]),
        evidence_row_scientific(scientific, verdict["scientific_ready"]),
        evidence_row_beir(beir, verdict["beir_ready"]),
        evidence_row_fallback(fallback, verdict["fallback_experimental_ready"]),
        evidence_row_table(table, verdict["table_benchmark_ready"]),
        "",
        "## What This Supports",
        "",
    ]

    for strength in verdict["strengths"]:
        lines.append(f"- {strength}")

    lines.extend(["", "## Limitations To State Clearly", ""])
    for limitation in verdict["limitations"]:
        lines.append(f"- {limitation}")

    lines.extend(["", "## Next Required Work Before Final Submission", ""])
    for item in verdict["required_before_final"]:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Reproducible Commands",
            "",
            "```powershell",
            ".\\.venv-gpu\\Scripts\\python.exe scripts\\generate_thesis_readiness_report.py",
            ".\\.venv-gpu\\Scripts\\python.exe scripts\\check_regression_gates.py",
            ".\\.venv-gpu\\Scripts\\python.exe scripts\\create_extended_table_benchmark.py --output-dir data/table_reasoning_benchmark",
            ".\\.venv-gpu\\Scripts\\python.exe scripts\\benchmark_llm_fallback.py --manifest data/table_reasoning_benchmark/manifest.json --output-dir results/llm_fallback_benchmark/table_reasoning_ollama_after_shape_gate --llm-fallback-provider ollama --skip-build --no-warmup",
            "```",
            "",
            "## Claim Boundary",
            "",
            "- Safe claim: the project implements and evaluates a grounded PDF QA prototype with lexical/dense/hybrid retrieval, routed QA, scientific/table-aware ingest evidence, and an experimental grounded LLM fallback.",
            "- Unsafe claim right now: production-ready QA, fully solved table reasoning, or LLM fallback as the default main path.",
        ]
    )

    return "\n".join(lines) + "\n"


def evidence_row_main_qa(user_suite: dict[str, Any], passed: bool) -> str:
    if not user_suite.get("available"):
        return f"| Main grounded QA | {status_label(False)} | Missing user PDF suite summary | `{user_suite['path']}` |"
    routed = user_suite.get("routed_grounded", {})
    bm25 = user_suite.get("bm25_only", {})
    evidence = (
        f"routed success {fmt_pct(routed.get('end_to_end_success_rate'))}, "
        f"answer match {fmt_pct(routed.get('answer_match_rate'))}, "
        f"grounded {fmt_pct(routed.get('grounded_rate'))}, "
        f"hallucination {fmt_pct(routed.get('hallucination_rate'))}; "
        f"bm25 success {fmt_pct(bm25.get('end_to_end_success_rate'))}"
    )
    return f"| Main grounded QA | {status_label(passed)} | {evidence} | `{user_suite['path']}` |"


def evidence_row_retrieval(retrieval: dict[str, Any], passed: bool) -> str:
    if not retrieval.get("available"):
        return f"| Retrieval comparison | {status_label(False)} | Missing retrieval smoke summary | `{retrieval['path']}` |"
    top_k = retrieval.get("top_k", 5)
    strategy_bits = []
    for name, metrics in retrieval.get("strategies", {}).items():
        strategy_bits.append(f"{name} recall@{top_k} {fmt_pct(metrics.get(f'recall_at_{top_k}'))}")
    evidence = "; ".join(strategy_bits)
    return f"| Retrieval comparison | {status_label(passed)} | {evidence} | `{retrieval['path']}` |"


def evidence_row_scientific(scientific: dict[str, Any], passed: bool) -> str:
    if not scientific.get("available"):
        return f"| Scientific/table ingest | {status_label(False)} | Missing readiness report | `{scientific['path']}` |"
    evidence = (
        f"runs {scientific.get('run_count')}, success min {fmt_pct(scientific.get('success_rate_min'))}, "
        f"IoU50 min {fmt_pct(scientific.get('iou50_micro_f1_min'))}, "
        f"IoU75 min {fmt_pct(scientific.get('iou75_micro_f1_min'))}, "
        f"p95 max {fmt_num(scientific.get('latency_p95_sec_max'))}s"
    )
    return f"| Scientific/table ingest | {status_label(passed)} | {evidence} | `{scientific['path']}` |"


def evidence_row_beir(beir: dict[str, Any], passed: bool) -> str:
    if not beir.get("available"):
        return f"| External-style retrieval | {status_label(False)} | Missing BEIR/SciFact sample summaries | `{beir['path']}` |"
    best = max(beir.get("runs", []), key=lambda item: item.get("ndcg_at_k", 0.0), default={})
    evidence = (
        f"{beir.get('run_count')} BEIR/SciFact sample runs; "
        f"best {best.get('strategy')} nDCG@k {fmt_num(best.get('ndcg_at_k'))}, "
        f"recall@k {fmt_pct(best.get('recall_at_k'))}"
    )
    return f"| External-style retrieval | {status_label(passed)} | {evidence} | `{beir['path']}` |"


def evidence_row_fallback(fallback: dict[str, Any], passed: bool) -> str:
    if not fallback.get("available"):
        return f"| Experimental LLM fallback | {status_label(False)} | Missing fallback repeat summary | `{fallback['path']}` |"
    evidence = (
        f"repeat {fallback.get('repeat')}, success gain mean {fmt_num(fallback.get('success_gain_vs_standard_mean'))}, "
        f"groundedness min {fmt_num(fallback.get('groundedness_min'))}, "
        f"hallucination delta max {fmt_num(fallback.get('hallucination_delta_max'))}, "
        f"table LLM resolved min {fmt_num(fallback.get('table_llm_resolved_count_min'), 0)}"
    )
    return f"| Experimental LLM fallback | {status_label(passed)} | {evidence} | `{fallback['path']}` |"


def evidence_row_table(table: dict[str, Any], passed: bool) -> str:
    if not table.get("available"):
        return f"| Extended table reasoning | {status_label(False)} | Missing table reasoning comparison summary | `{table['path']}` |"
    evidence = (
        f"queries {table.get('query_count')}, table success {fmt_pct(table.get('table_total_success'))}, "
        f"rule resolved {table.get('table_rule_resolved_count')}, "
        f"LLM attempts {table.get('table_llm_attempt_count')}, "
        f"LLM resolved {table.get('table_llm_resolved_count')}"
    )
    return f"| Extended table reasoning | {status_label(passed)} | {evidence} | `{table['path']}` |"


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    user_payload, user_path = load_optional_json(args.user_suite_summary)
    retrieval_payload, retrieval_path = load_optional_json(args.retrieval_smoke_summary)
    readiness_payload, readiness_path = load_optional_json(args.readiness_report)
    table_payload, table_path = load_optional_json(args.table_reasoning_summary)
    fallback_payload, fallback_path = load_optional_json(args.fallback_repeat_summary)

    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "topic": "Nghiên cứu các kĩ thuật truy xuất và hỏi đáp thông tin trên tài liệu PDF",
        "user_pdf_suite": summarize_user_suite(user_payload, user_path),
        "retrieval_smoke": summarize_retrieval_smoke(retrieval_payload, retrieval_path),
        "scientific_readiness": summarize_scientific_readiness(readiness_payload, readiness_path),
        "fallback_repeat": summarize_fallback_repeat(fallback_payload, fallback_path),
        "table_reasoning": summarize_table_reasoning(table_payload, table_path),
        "beir": summarize_beir(args.beir_root),
    }
    summary["verdict"] = build_verdict(summary)
    return summary


def main() -> None:
    args = parse_args()
    summary = build_summary(args)
    write_json(args.output_json, summary)
    write_text(args.output_md, build_markdown(summary))
    print(f"Wrote thesis readiness JSON: {rel(resolve_path(args.output_json))}")
    print(f"Wrote thesis readiness report: {rel(resolve_path(args.output_md))}")
    print(f"Research prototype ready: {summary['verdict']['research_prototype_ready']}")
    print(f"Production claim ready: {summary['verdict']['production_claim_ready']}")


if __name__ == "__main__":
    main()
