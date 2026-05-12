from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ingest.extract.model_layout import DEFAULT_LAYOUT_MODEL_NAME
from app.eval.ingest_metrics import (
    cer,
    char_accuracy,
    confusion_summary,
    detection_metrics,
    normalized_text_similarity,
    reading_order_score,
    summarize_numeric,
    table_exact_match,
    table_structure_score,
    token_f1,
    wer,
)
from app.eval.ingest_schemas import IngestBenchmarkSample, IngestGroundTruth, IngestPrediction, LayoutRegion
from scripts.benchmark_ingest_standard import git_commit


RESULTS_ROOT = Path("results/benchmark_suite")
UNIFIED_RESULTS_ROOT = Path("results/ingest")
UNIFIED_DATASETS = {"mock", "bastkorzen", "doclaynet", "publaynet", "pubtables", "ocr", "nougat"}
LAYOUT_LABELS = ["heading", "paragraph", "list_item", "table", "figure", "caption", "metadata"]
SCIENTIFIC_LABELS = ["title", "text", "list", "table", "figure"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run production + scientific ingest benchmarks")
    parser.add_argument("--dataset", choices=sorted(UNIFIED_DATASETS), default=None, help="Run unified ingest benchmark dataset adapter")
    parser.add_argument("--data-dir", type=Path, default=None, help="Dataset root for unified benchmark")
    parser.add_argument("--limit", type=int, default=0, help="Maximum unified benchmark samples; 0 means all available")
    parser.add_argument("--out", type=Path, default=None, help="Output directory for unified benchmark")
    parser.add_argument("--device", default=None, help="Optional device hint for model-backed ingest")
    parser.add_argument("--mode", choices=["text", "layout", "table", "ocr", "all"], default="all", help="Unified benchmark mode")
    parser.add_argument("--save-predictions", action="store_true", help="Save prediction payloads in per_sample.jsonl")
    parser.add_argument("--seed", type=int, default=13, help="Sampling seed for unified benchmark")

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


# ---------------------------------------------------------------------------
# Unified ingest benchmark adapter path. The legacy production/scientific suite
# remains below and is used when --dataset is not provided.


class DatasetAdapter:
    dataset_name = "base"

    def __init__(self, data_dir: Path | None, *, limit: int = 0, seed: int = 13, out_dir: Path | None = None) -> None:
        self.data_dir = data_dir
        self.limit = limit
        self.seed = seed
        self.out_dir = out_dir
        self.issues: list[dict[str, Any]] = []

    def load_samples(self) -> list[IngestBenchmarkSample]:
        raise NotImplementedError

    def _limit_samples(self, samples: list[IngestBenchmarkSample]) -> list[IngestBenchmarkSample]:
        if self.limit <= 0 or len(samples) <= self.limit:
            return samples
        rng = random.Random(self.seed)
        selected = list(samples)
        rng.shuffle(selected)
        return selected[: self.limit]


class MockAdapter(DatasetAdapter):
    dataset_name = "mock"

    def load_samples(self) -> list[IngestBenchmarkSample]:
        root = self.data_dir or ((self.out_dir or UNIFIED_RESULTS_ROOT / "mock") / "mock_data")
        root.mkdir(parents=True, exist_ok=True)
        pdf_path = root / "mock_ingest.pdf"
        _create_mock_pdf(pdf_path)
        sample = IngestBenchmarkSample(
            doc_id="mock_ingest",
            pdf_path=pdf_path,
            ground_truth=IngestGroundTruth(
                text=(
                    "Mock Ingest Benchmark\n"
                    "1. Overview\n"
                    "The pipeline should preserve text order.\n"
                    "Metric Value Owner\n"
                    "Latency Low Platform\n"
                    "Accuracy High QA\n"
                    "Figure 1: Mock figure caption"
                ),
                ordered_text=[
                    "Mock Ingest Benchmark",
                    "1. Overview",
                    "The pipeline should preserve text order.",
                    "Metric Value Owner",
                    "Latency Low Platform",
                    "Accuracy High QA",
                    "Figure",
                    "Figure 1: Mock figure caption",
                ],
                layout_regions=[
                    LayoutRegion("heading", (72, 60, 260, 90), "Mock Ingest Benchmark"),
                    LayoutRegion("heading", (72, 115, 180, 140), "1. Overview"),
                    LayoutRegion("paragraph", (72, 145, 390, 170), "The pipeline should preserve text order."),
                    LayoutRegion("table", (72, 190, 360, 252), "Metric Value Owner\nLatency Low Platform\nAccuracy High QA"),
                    LayoutRegion("figure", (390, 190, 470, 238), "Figure"),
                    LayoutRegion("caption", (72, 270, 280, 290), "Figure 1: Mock figure caption"),
                ],
                table_regions=[LayoutRegion("table", (72, 190, 360, 252), "Metric Value Owner\nLatency Low Platform\nAccuracy High QA")],
                table_cells=[
                    {"row": 0, "col": 0, "text": "Metric"},
                    {"row": 0, "col": 1, "text": "Value"},
                    {"row": 0, "col": 2, "text": "Owner"},
                    {"row": 1, "col": 0, "text": "Latency"},
                    {"row": 1, "col": 1, "text": "Low"},
                    {"row": 1, "col": 2, "text": "Platform"},
                    {"row": 2, "col": 0, "text": "Accuracy"},
                    {"row": 2, "col": 1, "text": "High"},
                    {"row": 2, "col": 2, "text": "QA"},
                ],
                table_csv="Metric,Value,Owner\nLatency,Low,Platform\nAccuracy,High,QA",
            ),
            metadata={"benchmark": "mock", "components": ["text", "layout", "table"]},
        )
        return self._limit_samples([sample] * max(self.limit, 1))


class PubTablesAdapter(DatasetAdapter):
    dataset_name = "pubtables"

    def load_samples(self) -> list[IngestBenchmarkSample]:
        root = self.data_dir or Path("data/benchmarks/pubtables_detection")
        image_root = _first_existing(
            [
                root / "extracted" / "images" / "test",
                root / "PubTables-1M-Detection_Images_Test",
                root / "images" / "test",
                root / "images",
            ]
        )
        annotation_root = _first_existing(
            [
                root / "extracted" / "annotations" / "test",
                root / "PubTables-1M-Detection_Annotations_Test",
                root / "annotations" / "test",
                root / "annotations",
            ]
        )
        if image_root is None or annotation_root is None:
            self.issues.append(
                {
                    "type": "missing_dataset",
                    "message": "PubTables adapter needs images and Pascal VOC XML annotations.",
                    "expected_paths": [
                        str(root / "extracted/images/test"),
                        str(root / "extracted/annotations/test"),
                    ],
                }
            )
            return []

        samples: list[IngestBenchmarkSample] = []
        for xml_path in sorted(annotation_root.glob("*.xml")):
            image_path = _find_image_for_stem(image_root, xml_path.stem)
            if image_path is None:
                self.issues.append({"type": "missing_image", "annotation": str(xml_path)})
                continue
            regions = _parse_pascal_voc_regions(xml_path, label="table")
            samples.append(
                IngestBenchmarkSample(
                    doc_id=xml_path.stem,
                    image_path=image_path,
                    ground_truth=IngestGroundTruth(table_regions=regions, layout_regions=regions),
                    metadata={"benchmark": "pubtables", "annotation_path": str(xml_path)},
                )
            )
        return self._limit_samples(samples)


class CocoLayoutAdapter(DatasetAdapter):
    dataset_name = "coco_layout"
    label_map: dict[str, str] = {}
    default_root = Path("data/benchmarks/layout")

    def load_samples(self) -> list[IngestBenchmarkSample]:
        root = self.data_dir or self.default_root
        annotation_path = _first_existing(
            [
                root / "COCO" / "test.json",
                root / "annotations" / "test.json",
                root / "test.json",
                root / "val.json",
            ]
        )
        image_root = _first_existing([root / "PNG" / "test", root / "images" / "test", root / "images", root])
        if annotation_path is None or image_root is None:
            self.issues.append(
                {
                    "type": "missing_dataset",
                    "message": f"{self.dataset_name} adapter expects COCO-style annotations and images.",
                    "expected_paths": [
                        str(root / "annotations/test.json"),
                        str(root / "images/test"),
                    ],
                }
            )
            return []
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        categories = {int(cat["id"]): str(cat["name"]) for cat in payload.get("categories", [])}
        images = {int(img["id"]): img for img in payload.get("images", [])}
        anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for ann in payload.get("annotations", []):
            anns_by_image[int(ann["image_id"])].append(ann)

        samples: list[IngestBenchmarkSample] = []
        for image_id, image in images.items():
            image_path = _resolve_image_path(image_root, image.get("file_name", ""))
            if image_path is None:
                continue
            regions = []
            for ann in anns_by_image.get(image_id, []):
                raw_label = categories.get(int(ann.get("category_id", -1)), "unknown")
                label = self.label_map.get(raw_label, raw_label.lower())
                x, y, w, h = [float(v) for v in ann.get("bbox", [0, 0, 0, 0])]
                regions.append(LayoutRegion(label, (x, y, x + w, y + h)))
            samples.append(
                IngestBenchmarkSample(
                    doc_id=str(image.get("id", image_path.stem)),
                    image_path=image_path,
                    ground_truth=IngestGroundTruth(layout_regions=regions, table_regions=[r for r in regions if r.label == "table"]),
                    metadata={"benchmark": self.dataset_name, "annotation_path": str(annotation_path)},
                )
            )
        return self._limit_samples(samples)


class DocLayNetAdapter(CocoLayoutAdapter):
    dataset_name = "doclaynet"
    default_root = Path("data/benchmarks/doclaynet")
    label_map = {
        "Title": "heading",
        "Section-header": "heading",
        "Text": "paragraph",
        "List-item": "list_item",
        "Table": "table",
        "Picture": "figure",
        "Caption": "caption",
        "Page-header": "metadata",
        "Page-footer": "metadata",
        "Footnote": "metadata",
        "Formula": "paragraph",
    }


class PubLayNetAdapter(CocoLayoutAdapter):
    dataset_name = "publaynet"
    default_root = Path("data/benchmarks/publaynet")
    label_map = {
        "title": "title",
        "text": "text",
        "list": "list",
        "table": "table",
        "figure": "figure",
    }


class TextJsonlAdapter(DatasetAdapter):
    dataset_name = "text"
    default_filename = "samples.jsonl"

    def load_samples(self) -> list[IngestBenchmarkSample]:
        root = self.data_dir or Path("data/benchmarks/text_extraction")
        manifest = root / self.default_filename
        if not manifest.exists():
            self.issues.append(
                {
                    "type": "missing_dataset",
                    "message": f"{self.dataset_name} adapter expects JSONL samples with pdf_path/image_path and ground_truth.text.",
                    "expected_path": str(manifest),
                }
            )
            return []
        samples: list[IngestBenchmarkSample] = []
        with manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                pdf_path = _optional_path(root, row.get("pdf_path"))
                image_path = _optional_path(root, row.get("image_path"))
                gt = row.get("ground_truth", {})
                samples.append(
                    IngestBenchmarkSample(
                        doc_id=str(row.get("doc_id") or (pdf_path or image_path).stem),
                        pdf_path=pdf_path,
                        image_path=image_path,
                        ground_truth=IngestGroundTruth(
                            text=gt.get("text"),
                            ordered_text=list(gt.get("ordered_text", []) or []),
                            form_fields=dict(gt.get("form_fields", {}) or {}),
                        ),
                        metadata=dict(row.get("metadata", {}) or {}),
                    )
                )
        return self._limit_samples(samples)


class BastKorzenTextAdapter(TextJsonlAdapter):
    dataset_name = "bastkorzen"
    default_filename = "bastkorzen_samples.jsonl"


class NougatTextAdapter(TextJsonlAdapter):
    dataset_name = "nougat"
    default_filename = "nougat_samples.jsonl"


class OCRDatasetAdapter(TextJsonlAdapter):
    dataset_name = "ocr"
    default_filename = "ocr_samples.jsonl"


def run_unified_ingest_benchmark(args: argparse.Namespace) -> Path:
    out_dir = args.out or (UNIFIED_RESULTS_ROOT / f"{args.dataset}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.device:
        os.environ["BOXBIIBOO_LAYOUT_DEVICE"] = args.device

    adapter = make_adapter(args.dataset, args.data_dir, limit=args.limit, seed=args.seed, out_dir=out_dir)
    samples = adapter.load_samples()
    records: list[dict[str, Any]] = []
    prediction_dir = out_dir / "predictions"
    if args.save_predictions:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    for sample in samples:
        prediction = predict_ingest(sample, work_dir=out_dir / "_work", mode=args.mode)
        record = score_sample(sample, prediction, mode=args.mode)
        records.append(record)
        if args.save_predictions:
            (prediction_dir / f"{sample.doc_id}.json").write_text(
                json.dumps(prediction.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

    summary = summarize_unified_records(
        records,
        dataset_name=args.dataset,
        mode=args.mode,
        config={
            "dataset": args.dataset,
            "data_dir": str(args.data_dir) if args.data_dir else None,
            "limit": args.limit,
            "out": str(out_dir),
            "device": args.device,
            "mode": args.mode,
            "save_predictions": args.save_predictions,
            "seed": args.seed,
        },
        issues=adapter.issues,
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "per_sample.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (out_dir / "README.md").write_text(render_unified_readme(summary, records), encoding="utf-8")
    if (out_dir / "_work").exists():
        shutil.rmtree(out_dir / "_work", ignore_errors=True)
    print(str(out_dir))
    return out_dir


def make_adapter(dataset: str, data_dir: Path | None, *, limit: int, seed: int, out_dir: Path) -> DatasetAdapter:
    adapter_cls: type[DatasetAdapter]
    adapter_cls = {
        "mock": MockAdapter,
        "bastkorzen": BastKorzenTextAdapter,
        "doclaynet": DocLayNetAdapter,
        "publaynet": PubLayNetAdapter,
        "pubtables": PubTablesAdapter,
        "ocr": OCRDatasetAdapter,
        "nougat": NougatTextAdapter,
    }[dataset]
    return adapter_cls(data_dir, limit=limit, seed=seed, out_dir=out_dir)


def predict_ingest(sample: IngestBenchmarkSample, *, work_dir: Path, mode: str = "all") -> IngestPrediction:
    from app.ingest.pipeline import ingest_pdf

    work_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    pdf_path = sample.pdf_path
    generated_pdf: Path | None = None
    try:
        if pdf_path is None and sample.image_path is not None:
            generated_pdf = image_to_pdf(sample.image_path, work_dir / f"{sample.doc_id}.pdf")
            pdf_path = generated_pdf
            if mode in {"layout", "table", "all"} and (sample.ground_truth.layout_regions or sample.ground_truth.table_regions):
                return predict_model_layout_direct(
                    pdf_path,
                    generated_pdf=generated_pdf,
                    started=started,
                    dataset_name=str(sample.metadata.get("benchmark") or ""),
                )
        if pdf_path is None:
            raise ValueError("Sample has neither pdf_path nor image_path")
        if mode == "text" and not sample.ground_truth.layout_regions and not sample.ground_truth.table_regions:
            return predict_text_extraction_direct(pdf_path, started=started)
        report = ingest_pdf(pdf_path)
        latency = time.perf_counter() - started
        blocks = report.get("blocks", [])
        text = "\n".join(str(getattr(block, "text", "") or "") for block in blocks)
        layout_regions = [
            LayoutRegion(_canonical_label(str(getattr(block, "block_type", "paragraph"))), tuple(getattr(block, "bbox", None) or (0, 0, 0, 0)), str(getattr(block, "text", "") or ""))
            for block in blocks
            if getattr(block, "bbox", None) is not None
        ]
        table_regions = [region for region in layout_regions if region.label == "table"]
        table_cells: list[dict[str, Any]] = []
        for block in blocks:
            if getattr(block, "block_type", "") == "table":
                table_cells.extend(list((getattr(block, "meta", {}) or {}).get("table_cells", []) or []))
        return IngestPrediction(
            text=text,
            ordered_text=[str(getattr(block, "text", "") or "") for block in blocks],
            layout_regions=layout_regions,
            table_regions=table_regions,
            table_cells=table_cells,
            backend=str(report.get("used_backend") or "unknown"),
            latency_sec=latency,
            success=True,
            metadata={
                "page_count": len(report.get("pages", [])),
                "block_count": len(blocks),
                "chunk_count": len(report.get("chunks", [])),
                "generated_pdf": str(generated_pdf) if generated_pdf else None,
            },
        )
    except Exception as exc:
        return IngestPrediction(
            latency_sec=time.perf_counter() - started,
            success=False,
            error=str(exc),
        )


def predict_text_extraction_direct(pdf_path: Path, *, started: float) -> IngestPrediction:
    from app.ingest.extract.text import extract_with_text_backend

    pages, blocks = extract_with_text_backend(pdf_path)
    layout_regions = [
        LayoutRegion(
            _canonical_label(str(getattr(block, "block_type", "paragraph"))),
            tuple(getattr(block, "bbox", None) or (0, 0, 0, 0)),
            str(getattr(block, "text", "") or ""),
        )
        for block in blocks
        if getattr(block, "bbox", None) is not None
    ]
    ordered_text = [str(getattr(block, "text", "") or "") for block in blocks]
    return IngestPrediction(
        text="\n".join(ordered_text),
        ordered_text=ordered_text,
        layout_regions=layout_regions,
        table_regions=[region for region in layout_regions if region.label == "table"],
        backend="text_direct",
        latency_sec=time.perf_counter() - started,
        success=True,
        metadata={
            "page_count": len(pages),
            "block_count": len(blocks),
            "direct_component_benchmark": True,
        },
    )


def predict_model_layout_direct(
    pdf_path: Path,
    *,
    generated_pdf: Path | None,
    started: float,
    dataset_name: str,
) -> IngestPrediction:
    import fitz

    from app.ingest.extract.model_layout import detect_model_layout_regions_for_page

    doc = fitz.open(str(pdf_path))
    layout_regions: list[LayoutRegion] = []
    ordered_text: list[str] = []
    try:
        for page in doc:
            for region in detect_model_layout_regions_for_page(page):
                label = _benchmark_label(
                    _canonical_label(str(region.get("block_type") or "paragraph")),
                    dataset_name=dataset_name,
                )
                text = str(region.get("direct_text") or region.get("label_name") or label)
                layout_regions.append(LayoutRegion(label, tuple(region["bbox"]), text))
                ordered_text.append(text)
    finally:
        doc.close()

    return IngestPrediction(
        text="\n".join(ordered_text),
        ordered_text=ordered_text,
        layout_regions=layout_regions,
        table_regions=[region for region in layout_regions if region.label == "table"],
        backend="model_layout_direct",
        latency_sec=time.perf_counter() - started,
        success=True,
        metadata={
            "generated_pdf": str(generated_pdf) if generated_pdf else None,
            "block_count": len(layout_regions),
            "direct_component_benchmark": True,
        },
    )


def score_sample(sample: IngestBenchmarkSample, prediction: IngestPrediction, *, mode: str) -> dict[str, Any]:
    gt = sample.ground_truth
    record: dict[str, Any] = {
        "doc_id": sample.doc_id,
        "success": prediction.success,
        "error": prediction.error,
        "backend": prediction.backend,
        "latency_sec": prediction.latency_sec,
        "mode": mode,
    }
    if not prediction.success:
        return record

    text_f1 = token_f1(prediction.text, gt.text)
    record.update(
        {
            "char_accuracy": char_accuracy(prediction.text, gt.text),
            "token_precision": text_f1["precision"] if text_f1 else None,
            "token_recall": text_f1["recall"] if text_f1 else None,
            "token_f1": text_f1["f1"] if text_f1 else None,
            "normalized_text_similarity": normalized_text_similarity(prediction.text, gt.text),
            "cer": cer(prediction.text, gt.text),
            "wer": wer(prediction.text, gt.text),
            "reading_order_score": reading_order_score(prediction.ordered_text, gt.ordered_text),
        }
    )
    predicted_layout_regions = _aligned_prediction_regions(sample, prediction.layout_regions)
    predicted_table_regions = [region for region in predicted_layout_regions if region.label == "table"]
    if mode in {"layout", "all"} and gt.layout_regions:
        labels = sorted({r.label for r in gt.layout_regions} | {r.label for r in predicted_layout_regions})
        record["layout_iou50"] = detection_metrics(predicted_layout_regions, gt.layout_regions, labels=labels, iou_threshold=0.50)
        record["layout_iou75"] = detection_metrics(predicted_layout_regions, gt.layout_regions, labels=labels, iou_threshold=0.75)
        record["layout_confusion_iou50"] = confusion_summary(predicted_layout_regions, gt.layout_regions, iou_threshold=0.50)
    if mode in {"table", "all"}:
        if gt.table_regions:
            record["table_detection_iou50"] = detection_metrics(predicted_table_regions, gt.table_regions, labels=["table"], iou_threshold=0.50)
            record["table_detection_iou75"] = detection_metrics(predicted_table_regions, gt.table_regions, labels=["table"], iou_threshold=0.75)
        record["table_structure"] = table_structure_score(prediction.table_cells, gt.table_cells)
        record["table_exact_csv"] = table_exact_match(_cells_to_csv(prediction.table_cells), gt.table_csv)
        record["table_exact_html"] = table_exact_match(None, gt.table_html)
    if mode in {"ocr", "all"}:
        record["ocr_cer"] = record["cer"]
        record["ocr_wer"] = record["wer"]
        record["ocr_token_f1"] = record["token_f1"]
        if gt.form_fields:
            record["form_field_f1"] = _form_field_f1({}, gt.form_fields)
    return record


def summarize_unified_records(
    records: list[dict[str, Any]],
    *,
    dataset_name: str,
    mode: str,
    config: dict[str, Any],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    success_records = [record for record in records if record.get("success")]
    metric_summary: dict[str, Any] = {}
    for key in (
        "char_accuracy",
        "token_f1",
        "normalized_text_similarity",
        "cer",
        "wer",
        "reading_order_score",
        "ocr_cer",
        "ocr_wer",
        "ocr_token_f1",
        "form_field_f1",
        "table_exact_csv",
    ):
        values = [float(record[key]) for record in success_records if record.get(key) is not None]
        if values:
            metric_summary[key] = summarize_numeric(values)
    for key in ("layout_iou50", "layout_iou75", "table_detection_iou50", "table_detection_iou75", "table_structure"):
        metric_summary[key] = _summarize_nested_metric(success_records, key)

    latencies = [float(record.get("latency_sec", 0.0) or 0.0) for record in records if record.get("success")]
    return {
        "dataset_name": dataset_name,
        "mode": mode,
        "num_samples": len(records),
        "success_rate": (len(success_records) / len(records)) if records else 0.0,
        "metric_summary": metric_summary,
        "latency": summarize_numeric(latencies),
        "error_count": len(records) - len(success_records),
        "errors": [
            {"doc_id": record.get("doc_id"), "error": record.get("error")}
            for record in records
            if not record.get("success")
        ],
        "issues": issues,
        "config": config,
        "backend_counts": dict(Counter(str(record.get("backend")) for record in success_records)),
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": git_commit(),
    }


def _summarize_nested_metric(records: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    values = [record.get(key) for record in records if isinstance(record.get(key), dict)]
    if not values:
        return None
    fields = ("micro_precision", "micro_recall", "micro_f1", "macro_f1", "precision", "recall", "f1")
    summary: dict[str, Any] = {}
    for field in fields:
        nums = [float(value[field]) for value in values if value.get(field) is not None]
        if nums:
            summary[field] = summarize_numeric(nums)
    return summary or None


def render_unified_readme(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    metrics = summary.get("metric_summary", {})
    lines = [
        f"# Ingest Benchmark: {summary['dataset_name']}",
        "",
        f"- Mode: `{summary['mode']}`",
        f"- Samples: {summary['num_samples']}",
        f"- Success rate: {summary['success_rate']:.3f}",
        f"- Latency mean/p50/p95: {summary['latency']['mean']:.3f}s / {summary['latency']['p50']:.3f}s / {summary['latency']['p95']:.3f}s",
        f"- Error count: {summary['error_count']}",
        f"- Backend counts: `{json.dumps(summary.get('backend_counts', {}), ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Key Metrics",
        "",
    ]
    if not metrics:
        lines.append("- No ground-truth metrics were available for this run.")
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, dict) and "mean" in value:
            lines.append(f"- {key}: mean={value['mean']:.3f}, p50={value['p50']:.3f}, p95={value['p95']:.3f}")
        else:
            lines.append(f"- {key}: `{json.dumps(value, ensure_ascii=False, sort_keys=True)}`")
    if summary.get("issues"):
        lines.extend(["", "## Dataset Issues", ""])
        for issue in summary["issues"]:
            lines.append(f"- `{json.dumps(issue, ensure_ascii=False)}`")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- This runner evaluates local subsets and does not download large datasets automatically.",
            "- Some adapters run in detection-only mode when cell/text ground truth is unavailable.",
            "- OCR/form metrics are reported only when ground-truth text or form fields are provided.",
        ]
    )
    return "\n".join(lines)


def _create_mock_pdf(path: Path) -> None:
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    lines = [
        ("Mock Ingest Benchmark", 18, (72, 72)),
        ("1. Overview", 15, (72, 128)),
        ("The pipeline should preserve text order.", 11, (72, 158)),
        ("Metric        Value        Owner", 11, (72, 210)),
        ("Latency       Low          Platform", 11, (72, 232)),
        ("Accuracy      High         QA", 11, (72, 254)),
        ("Figure 1: Mock figure caption", 10, (72, 300)),
    ]
    for text, size, point in lines:
        page.insert_text(point, text, fontsize=size)
    page.draw_rect((390, 190, 470, 238), color=(0.2, 0.2, 0.2), width=1)
    doc.save(path)
    doc.close()


def image_to_pdf(image_path: Path, output_path: Path) -> Path:
    import fitz

    doc = fitz.open()
    pix = fitz.Pixmap(str(image_path))
    page = doc.new_page(width=pix.width, height=pix.height)
    page.insert_image(page.rect, filename=str(image_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    doc.close()
    return output_path


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _find_image_for_stem(root: Path, stem: str) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
        candidate = root / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    matches = list(root.rglob(f"{stem}.*"))
    return matches[0] if matches else None


def _parse_pascal_voc_regions(xml_path: Path, *, label: str) -> list[LayoutRegion]:
    root = ET.parse(xml_path).getroot()
    regions: list[LayoutRegion] = []
    for obj in root.findall(".//object"):
        name = (obj.findtext("name") or label).strip().lower()
        bbox = obj.find("bndbox")
        if bbox is None:
            continue
        x0 = float(bbox.findtext("xmin") or 0)
        y0 = float(bbox.findtext("ymin") or 0)
        x1 = float(bbox.findtext("xmax") or 0)
        y1 = float(bbox.findtext("ymax") or 0)
        regions.append(LayoutRegion(label if "table" in name else name, (x0, y0, x1, y1)))
    return regions


def _resolve_image_path(root: Path, file_name: str) -> Path | None:
    candidate = root / file_name
    if candidate.exists():
        return candidate
    matches = list(root.rglob(Path(file_name).name))
    return matches[0] if matches else None


def _optional_path(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _aligned_prediction_regions(sample: IngestBenchmarkSample, regions: list[LayoutRegion]) -> list[LayoutRegion]:
    dataset_name = str(sample.metadata.get("benchmark") or "").strip().lower()
    if not dataset_name:
        return regions
    return [
        LayoutRegion(
            _benchmark_label(region.label, dataset_name=dataset_name),
            region.bbox,
            region.text,
            dict(region.metadata),
        )
        for region in regions
    ]


def _benchmark_label(label: str, *, dataset_name: str) -> str:
    normalized = _canonical_label(label)
    if dataset_name == "publaynet":
        return {
            "heading": "title",
            "paragraph": "text",
            "list_item": "list",
            "table": "table",
            "figure": "figure",
            "caption": "text",
            "metadata": "text",
        }.get(normalized, normalized)
    return normalized


def _canonical_label(label: str) -> str:
    if label == "metadata":
        return "metadata"
    return label


def _cells_to_csv(cells: list[dict[str, Any]]) -> str | None:
    if not cells:
        return None
    max_row = max(int(cell.get("row", 0) or 0) for cell in cells)
    max_col = max(int(cell.get("col", 0) or 0) for cell in cells)
    table = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]
    for cell in cells:
        table[int(cell.get("row", 0) or 0)][int(cell.get("col", 0) or 0)] = str(cell.get("text", ""))
    return "\n".join(",".join(row) for row in table)


def _form_field_f1(predicted: dict[str, str], expected: dict[str, str]) -> float:
    if not expected:
        return 1.0
    pred_items = {(k, v) for k, v in predicted.items()}
    gt_items = {(k, v) for k, v in expected.items()}
    overlap = len(pred_items & gt_items)
    precision = overlap / len(pred_items) if pred_items else 0.0
    recall = overlap / len(gt_items) if gt_items else 0.0
    return (2 * precision * recall / (precision + recall)) if precision + recall else 0.0


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

    if args.dataset:
        run_unified_ingest_benchmark(args)
        return

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
