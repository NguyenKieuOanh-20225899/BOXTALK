from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from app.eval.ingest_metrics import detection_metrics, normalized_text_similarity, token_f1
from app.eval.ingest_schemas import LayoutRegion
from scripts.benchmark_ingest_suite import MockAdapter


def test_text_metrics() -> None:
    scores = token_f1("The quick brown fox", "quick brown fox")
    assert scores is not None
    assert scores["recall"] == 1.0
    assert scores["precision"] == 0.75
    assert normalized_text_similarity("A  B", "a b") == 1.0


def test_detection_metrics() -> None:
    pred = [LayoutRegion("table", (0, 0, 100, 100))]
    gt = [LayoutRegion("table", (0, 0, 100, 100))]
    metrics = detection_metrics(pred, gt, labels=["table"], iou_threshold=0.5)
    assert metrics["micro_f1"] == 1.0
    assert metrics["per_label"]["table"]["tp"] == 1.0


def test_mock_adapter(tmp_path: Path) -> None:
    adapter = MockAdapter(None, limit=2, out_dir=tmp_path)
    samples = adapter.load_samples()
    assert len(samples) == 2
    assert samples[0].pdf_path is not None
    assert samples[0].pdf_path.exists()
    assert samples[0].ground_truth.text


def test_cli_mock_run_creates_summary(tmp_path: Path) -> None:
    out_dir = tmp_path / "mock_run"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_ingest_suite.py",
            "--dataset",
            "mock",
            "--limit",
            "1",
            "--out",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    summary_path = out_dir / "summary.json"
    readme_path = out_dir / "README.md"
    assert summary_path.exists()
    assert readme_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["dataset_name"] == "mock"
    assert summary["num_samples"] == 1
    assert "metric_summary" in summary
