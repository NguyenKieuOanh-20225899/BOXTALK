from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from app.eval.ingest_metrics import detection_metrics, normalized_text_similarity, table_structure_breakdown, token_f1
from app.eval.ingest_schemas import LayoutRegion
from scripts.benchmark_ingest_suite import MockAdapter, _is_auxiliary_text_block


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


def test_table_structure_breakdown_reports_row_col_segmentation() -> None:
    gt_cells = [
        {"row": 0, "col": 0, "text": "A", "bbox": (0, 0, 50, 20)},
        {"row": 0, "col": 1, "text": "B", "bbox": (50, 0, 100, 20)},
        {"row": 1, "col": 0, "text": "C", "bbox": (0, 20, 50, 40)},
        {"row": 1, "col": 1, "text": "D", "bbox": (50, 20, 100, 40)},
    ]
    pred_cells = [
        {"row": 0, "col": 0, "text": "A", "bbox": (0, 0, 50, 20)},
        {"row": 0, "col": 1, "text": "B", "bbox": (50, 0, 100, 20)},
        {"row": 1, "col": 0, "text": "C", "bbox": (0, 20, 50, 30)},
        {"row": 2, "col": 0, "text": "D", "bbox": (50, 30, 100, 40)},
    ]

    breakdown = table_structure_breakdown(pred_cells, gt_cells)

    assert breakdown is not None
    assert breakdown["row_count_mae"] == 1
    assert breakdown["row_oversegmentation_count"] == 1
    assert breakdown["row_undersegmentation_count"] == 0
    assert breakdown["col_count_mae"] == 0


def test_auxiliary_ocr_table_block_is_not_primary_text() -> None:
    class DummyBlock:
        def __init__(self, meta: dict | None) -> None:
            self.meta = meta

    assert _is_auxiliary_text_block(DummyBlock({"synthetic_table_cluster": True}))
    assert not _is_auxiliary_text_block(DummyBlock({"synthetic_table_cluster": False}))
    assert not _is_auxiliary_text_block(DummyBlock({}))


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
