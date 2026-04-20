from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ingest.extract.model_layout import DEFAULT_LAYOUT_MODEL_NAME
from scripts.benchmark_ingest_standard import git_commit


RESULTS_ROOT = Path("results/benchmark_suite")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run production + scientific ingest benchmarks")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional suite output directory")
    parser.add_argument("--skip-production", action="store_true", help="Skip production benchmark")
    parser.add_argument("--skip-scientific", action="store_true", help="Skip scientific benchmark")

    parser.add_argument("--production-repeats", type=int, default=1)
    parser.add_argument("--production-warmup-per-label", type=int, default=1)
    parser.add_argument("--production-max-per-label", type=int, default=0)
    parser.add_argument(
        "--production-profiles",
        nargs="+",
        default=["baseline", "model_routed_doclaynet"],
    )

    parser.add_argument("--doclaynet-root", type=Path, default=Path("data/benchmarks/doclaynet"))
    parser.add_argument("--doclaynet-split", default="test")
    parser.add_argument("--doclaynet-limit", type=int, default=0)
    parser.add_argument("--skip-doclaynet", action="store_true")
    parser.add_argument("--pubtables-root", type=Path, default=Path("data/benchmarks/pubtables_detection"))
    parser.add_argument("--pubtables-split", default="test")
    parser.add_argument("--pubtables-limit", type=int, default=0)
    parser.add_argument("--skip-pubtables", action="store_true")
    parser.add_argument(
        "--scientific-profiles",
        nargs="+",
        default=["baseline", "model_routed_doclaynet"],
    )
    return parser.parse_args()


def _run_and_capture(cmd: list[str]) -> str:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed:\n{' '.join(cmd)}\n"
            f"returncode={result.returncode}\n"
            f"stderr:\n{result.stderr}\n"
            f"stdout:\n{result.stdout}"
        )
    return result.stdout.strip().splitlines()[-1].strip()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def render_markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Ingest Benchmark Suite",
        "",
        f"- Timestamp: {summary['metadata']['timestamp_utc']}",
        f"- Git commit: {summary['metadata']['git_commit']}",
        f"- Chosen model: {summary['metadata']['chosen_model']}",
        "",
    ]

    production = summary.get("production")
    if production:
        lines.extend(
            [
                "## Production",
                "",
                f"- Output dir: `{production['output_dir']}`",
                f"- Summary file: `{production['summary_file']}`",
                "",
            ]
        )

    scientific = summary.get("scientific")
    if scientific:
        lines.extend(
            [
                "## Scientific",
                "",
                f"- Output dir: `{scientific['output_dir']}`",
                f"- Summary file: `{scientific['summary_file']}`",
                "",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (RESULTS_ROOT / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    suite_summary: dict[str, Any] = {
        "metadata": {
            "timestamp_utc": timestamp,
            "git_commit": git_commit(),
            "chosen_model": DEFAULT_LAYOUT_MODEL_NAME,
        }
    }

    if not args.skip_production:
        cmd = [
            sys.executable,
            str(ROOT / "scripts/benchmark_ingest_standard.py"),
            "--repeats",
            str(args.production_repeats),
            "--warmup-per-label",
            str(args.production_warmup_per_label),
            "--max-per-label",
            str(args.production_max_per_label),
            "--profiles",
            *args.production_profiles,
        ]
        production_output = _run_and_capture(cmd)
        production_output_dir = Path(production_output)
        suite_summary["production"] = {
            "output_dir": str(production_output_dir),
            "summary_file": str(production_output_dir / "benchmark_summary.json"),
            "summary": _load_json(production_output_dir / "benchmark_summary.json"),
        }

    if not args.skip_scientific:
        cmd = [
            sys.executable,
            str(ROOT / "scripts/benchmark_ingest_scientific.py"),
            "--doclaynet-root",
            str(args.doclaynet_root),
            "--doclaynet-split",
            args.doclaynet_split,
            "--doclaynet-limit",
            str(args.doclaynet_limit),
            "--skip-doclaynet" if args.skip_doclaynet else "",
            "--pubtables-root",
            str(args.pubtables_root),
            "--pubtables-split",
            args.pubtables_split,
            "--pubtables-limit",
            str(args.pubtables_limit),
            "--skip-pubtables" if args.skip_pubtables else "",
            "--profiles",
            *args.scientific_profiles,
        ]
        cmd = [part for part in cmd if part]
        scientific_output = _run_and_capture(cmd)
        scientific_output_dir = Path(scientific_output)
        suite_summary["scientific"] = {
            "output_dir": str(scientific_output_dir),
            "summary_file": str(scientific_output_dir / "scientific_summary.json"),
            "summary": _load_json(scientific_output_dir / "scientific_summary.json"),
        }

    with (output_dir / "suite_summary.json").open("w", encoding="utf-8") as f:
        json.dump(suite_summary, f, ensure_ascii=False, indent=2)
    (output_dir / "suite_summary.md").write_text(
        render_markdown_summary(suite_summary),
        encoding="utf-8",
    )

    print(str(output_dir))


if __name__ == "__main__":
    main()
