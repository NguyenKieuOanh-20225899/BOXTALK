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
DEFAULT_MANIFEST = Path("data/user_pdf_benchmark_suite.json")
RESULTS_ROOT = Path("results/user_pdf_benchmark_suite")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run QA benchmarks across multiple user-PDF document types and aggregate results."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Suite manifest JSON.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Suite output directory.")
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        help="Override QA configs for every document. Repeat or comma-separate; use 'all' for ablations.",
    )
    parser.add_argument("--dense-preset", default=None, help="Override dense preset for newly built indexes.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for child scripts.")
    parser.add_argument("--skip-build", action="store_true", help="Do not build missing indexes.")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild every index before benchmarking.")
    parser.add_argument("--no-warmup", action="store_true", help="Pass --no-warmup to benchmark_qa.py.")
    parser.add_argument("--dry-run", action="store_true", help="Print child commands without running them.")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep running remaining documents on failure.")
    parser.add_argument("--doc-id", action="append", default=[], help="Optional document id filter.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def resolve_path(path_value: str | Path | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def expand_csv_values(values: Iterable[str] | None) -> list[str]:
    expanded: list[str] = []
    for value in values or []:
        expanded.extend(part.strip() for part in value.split(",") if part.strip())
    return expanded


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def child_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def run_command(command: list[str], *, dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if dry_run:
        return
    subprocess.run(command, cwd=ROOT, env=child_env(), check=True)


def input_args_for_build(doc: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for key, flag in (
        ("pdf", "--pdf"),
        ("chunks_json", "--chunks-json"),
        ("chunks_jsonl", "--chunks-jsonl"),
    ):
        for value in as_list(doc.get(key)):
            path = resolve_path(value)
            if path is None:
                continue
            args.extend([flag, str(path)])
    return args


def ensure_index(
    *,
    doc: dict[str, Any],
    index_dir: Path,
    args: argparse.Namespace,
    manifest: dict[str, Any],
) -> None:
    index_exists = (index_dir / "corpus.jsonl").exists()
    if index_exists and not args.rebuild_index:
        return
    if args.skip_build:
        raise FileNotFoundError(f"Missing retrieval index for {doc['id']}: {index_dir}")

    build_inputs = input_args_for_build(doc)
    if not build_inputs:
        raise ValueError(f"Document {doc['id']} needs pdf, chunks_json, or chunks_jsonl in the manifest")

    dense_preset = args.dense_preset or doc.get("dense_preset") or manifest.get("default_dense_preset") or "minilm"
    command = [
        args.python,
        str(ROOT / "scripts" / "build_retrieval_index.py"),
        *build_inputs,
        "--output-dir",
        str(index_dir),
        "--dense-preset",
        str(dense_preset),
    ]
    if doc.get("skip_dense"):
        command.append("--skip-dense")
    if doc.get("build_colbert"):
        command.append("--build-colbert")
    run_command(command, dry_run=args.dry_run)


def config_arg(doc: dict[str, Any], manifest: dict[str, Any], override: list[str]) -> str:
    if override:
        return ",".join(override)
    value = doc.get("configs", manifest.get("default_configs", ["routed_grounded"]))
    if isinstance(value, str):
        return value
    return ",".join(str(item) for item in value)


def run_qa_benchmark(
    *,
    doc: dict[str, Any],
    index_dir: Path,
    queries_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    override_configs: list[str],
) -> None:
    command = [
        args.python,
        str(ROOT / "scripts" / "benchmark_qa.py"),
        "--index-dir",
        str(index_dir),
        "--queries",
        str(queries_path),
        "--output-dir",
        str(output_dir),
        "--config",
        config_arg(doc, manifest, override_configs),
    ]
    for extra in as_list(doc.get("benchmark_args")):
        command.append(str(extra))
    for extra in as_list(manifest.get("benchmark_args")):
        command.append(str(extra))
    if args.no_warmup:
        command.append("--no-warmup")
    run_command(command, dry_run=args.dry_run)


def mean_float(values: Iterable[float]) -> float:
    values_list = list(values)
    return statistics.mean(values_list) if values_list else 0.0


def ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"query_count": 0}
    unique_questions = {
        (str(row.get("document_id") or ""), str(row.get("query_id") or row.get("question") or ""))
        for row in rows
    }
    answerable = [row for row in rows if row.get("should_answer")]
    unanswerable = [row for row in rows if not row.get("should_answer")]
    abstentions = [row for row in rows if row.get("decision") != "answer"]
    correct_abstentions = [row for row in unanswerable if row.get("decision") != "answer"]
    false_answers = [row for row in unanswerable if row.get("decision") == "answer"]
    return {
        "query_count": len(rows),
        "unique_question_count": len(unique_questions),
        "document_count": len({row.get("document_id") for row in rows}),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        "answer_match_rate": mean_float(float(bool(row.get("answer_match"))) for row in rows),
        "evidence_match_rate": mean_float(float(bool(row.get("evidence_match"))) for row in rows),
        "grounded_rate": mean_float(float(bool(row.get("grounded_answer"))) for row in rows),
        "end_to_end_success_rate": mean_float(float(bool(row.get("end_to_end_success"))) for row in rows),
        "hallucination_rate": mean_float(float(bool(row.get("hallucinated"))) for row in rows),
        "abstention_precision": ratio(len(correct_abstentions), len(abstentions)),
        "abstention_recall": ratio(len(correct_abstentions), len(unanswerable)),
        "false_answer_count": len(false_answers),
        "avg_answer_token_f1": mean_float(float(row.get("answer_token_f1") or 0.0) for row in answerable),
        "avg_total_latency_ms": mean_float(float(row.get("total_latency_ms") or 0.0) for row in rows),
        "avg_route_attempt_count": mean_float(float(row.get("route_attempt_count") or 1.0) for row in rows),
        "route_retry_rate": mean_float(float(bool(row.get("route_retry_used"))) for row in rows),
    }


def grouped_summary(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group_key = " / ".join(str(row.get(key) or "unknown") for key in keys)
        groups.setdefault(group_key, []).append(row)
    return {key: summarize_rows(group_rows) for key, group_rows in sorted(groups.items())}


def add_document_metadata(
    rows: list[dict[str, Any]],
    *,
    doc: dict[str, Any],
    index_dir: Path,
    queries_path: Path,
) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        enriched.append(
            {
                "document_id": doc["id"],
                "document_type": doc.get("document_type", "unknown"),
                "language": doc.get("language", "unknown"),
                "pdf": doc.get("pdf"),
                "queries": display_path(queries_path),
                "index_dir": display_path(index_dir),
                **row,
            }
        )
    return enriched


def json_for_csv(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    preferred = [
        "document_id",
        "document_type",
        "language",
        "config_name",
        "query_id",
        "query_type",
        "question",
        "decision",
        "answer_match",
        "evidence_match",
        "grounded_answer",
        "end_to_end_success",
        "hallucinated",
        "total_latency_ms",
    ]
    fieldnames = [field for field in preferred if any(field in row for row in rows)]
    fieldnames.extend(sorted({key for row in rows for key in row.keys()} - set(fieldnames)))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json_for_csv(row.get(key)) for key in fieldnames})


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# User PDF Benchmark Suite",
        "",
        f"- Manifest: `{summary['manifest']}`",
        f"- Documents: {len(summary.get('documents', []))}",
        f"- Unique questions: {summary.get('overall', {}).get('unique_question_count', 0)}",
        f"- Question-config rows: {summary.get('overall', {}).get('query_count', 0)}",
        "",
        "## Aggregate By Config",
        "",
        "| Config | Docs | Queries | Success | Answer | Evidence | Grounded | Hallucination | Retry | Latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config_name, metrics in summary.get("by_config", {}).items():
        lines.append(
            "| {name} | {docs} | {queries} | {success:.3f} | {answer:.3f} | {evidence:.3f} | {grounded:.3f} | {hallucination:.3f} | {retry:.3f} | {latency:.1f} |".format(
                name=config_name,
                docs=metrics.get("document_count", 0),
                queries=metrics.get("query_count", 0),
                success=metrics.get("end_to_end_success_rate", 0.0),
                answer=metrics.get("answer_match_rate", 0.0),
                evidence=metrics.get("evidence_match_rate", 0.0),
                grounded=metrics.get("grounded_rate", 0.0),
                hallucination=metrics.get("hallucination_rate", 0.0),
                retry=metrics.get("route_retry_rate", 0.0),
                latency=metrics.get("avg_total_latency_ms", 0.0),
            )
        )

    lines.extend(
        [
            "",
            "## Per Document",
            "",
            "| Document | Type | Language | Config | Queries | Success | Answer | Evidence | Grounded | Latency ms |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run in summary.get("documents", []):
        for config_name, metrics in run.get("configs", {}).items():
            lines.append(
                "| {doc} | {typ} | {lang} | {config} | {queries} | {success:.3f} | {answer:.3f} | {evidence:.3f} | {grounded:.3f} | {latency:.1f} |".format(
                    doc=run["document_id"],
                    typ=run.get("document_type", "unknown"),
                    lang=run.get("language", "unknown"),
                    config=config_name,
                    queries=metrics.get("query_count", 0),
                    success=metrics.get("end_to_end_success_rate", 0.0),
                    answer=metrics.get("answer_match_rate", 0.0),
                    evidence=metrics.get("evidence_match_rate", 0.0),
                    grounded=metrics.get("grounded_rate", 0.0),
                    latency=metrics.get("avg_total_latency_ms", 0.0),
                )
            )
    if summary.get("failed_runs"):
        lines.extend(["", "## Failed Runs", ""])
        for failure in summary["failed_runs"]:
            lines.append(f"- `{failure['document_id']}`: {failure['error']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_document(
    *,
    doc: dict[str, Any],
    manifest: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
    override_configs: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if "id" not in doc:
        raise ValueError("Every document entry must include an id")

    queries_path = resolve_path(doc.get("queries"))
    if queries_path is None or not queries_path.exists():
        raise FileNotFoundError(f"Missing queries file for {doc['id']}: {queries_path}")

    index_dir = resolve_path(doc.get("index_dir"))
    if index_dir is None:
        index_dir = ROOT / "results" / "retrieval_index" / str(doc["id"])
    ensure_index(doc=doc, index_dir=index_dir, args=args, manifest=manifest)

    doc_output_dir = output_dir / "documents" / str(doc["id"])
    run_qa_benchmark(
        doc=doc,
        index_dir=index_dir,
        queries_path=queries_path,
        output_dir=doc_output_dir,
        args=args,
        manifest=manifest,
        override_configs=override_configs,
    )

    if args.dry_run:
        return (
            {
                "document_id": doc["id"],
                "document_type": doc.get("document_type", "unknown"),
                "language": doc.get("language", "unknown"),
                "index_dir": display_path(index_dir),
                "queries": display_path(queries_path),
                "output_dir": display_path(doc_output_dir),
                "configs": {},
            },
            [],
        )

    run_summary = load_json(doc_output_dir / "qa_summary.json")
    rows_payload = json.loads((doc_output_dir / "per_question.json").read_text(encoding="utf-8"))
    if not isinstance(rows_payload, list):
        raise ValueError(f"{doc_output_dir / 'per_question.json'} must contain a JSON list")
    rows = add_document_metadata(rows_payload, doc=doc, index_dir=index_dir, queries_path=queries_path)
    return (
        {
            "document_id": doc["id"],
            "document_type": doc.get("document_type", "unknown"),
            "language": doc.get("language", "unknown"),
            "index_dir": display_path(index_dir),
            "queries": display_path(queries_path),
            "output_dir": display_path(doc_output_dir),
            "configs": run_summary.get("configs", {}),
            "ablation_deltas": run_summary.get("ablation_deltas", {}),
        },
        rows,
    )


def main() -> None:
    args = parse_args()
    manifest_path = resolve_path(args.manifest)
    if manifest_path is None or not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    manifest = load_json(manifest_path)
    documents = manifest.get("documents")
    if not isinstance(documents, list) or not documents:
        raise SystemExit(f"{manifest_path} must define a non-empty documents list")

    wanted_doc_ids = set(args.doc_id)
    if wanted_doc_ids:
        documents = [doc for doc in documents if str(doc.get("id")) in wanted_doc_ids]
    if not documents:
        raise SystemExit("No documents selected.")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (ROOT / RESULTS_ROOT / timestamp)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    override_configs = expand_csv_values(args.config)
    all_rows: list[dict[str, Any]] = []
    document_runs: list[dict[str, Any]] = []
    failed_runs: list[dict[str, Any]] = []

    for doc in documents:
        try:
            run_summary, rows = process_document(
                doc=doc,
                manifest=manifest,
                output_dir=output_dir,
                args=args,
                override_configs=override_configs,
            )
            document_runs.append(run_summary)
            all_rows.extend(rows)
        except Exception as exc:
            failure = {"document_id": doc.get("id", "unknown"), "error": str(exc)}
            failed_runs.append(failure)
            if not args.continue_on_error:
                raise
            print(f"FAILED {failure['document_id']}: {failure['error']}", file=sys.stderr)

    summary = {
        "timestamp_utc": timestamp,
        "suite_name": manifest.get("suite_name", "user_pdf_benchmark_suite"),
        "manifest": display_path(manifest_path),
        "output_dir": display_path(output_dir),
        "documents": document_runs,
        "failed_runs": failed_runs,
        "overall": summarize_rows(all_rows),
        "by_config": grouped_summary(all_rows, ("config_name",)),
        "by_document_type_and_config": grouped_summary(all_rows, ("document_type", "config_name")),
        "by_language_and_config": grouped_summary(all_rows, ("language", "config_name")),
    }

    (output_dir / "suite_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "per_question.json").write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(output_dir / "per_question.csv", all_rows)
    write_markdown(output_dir / "README.md", summary)
    print(output_dir)


if __name__ == "__main__":
    main()
