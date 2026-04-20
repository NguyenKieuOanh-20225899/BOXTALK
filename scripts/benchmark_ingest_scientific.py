from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
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

import fitz

from app.ingest.extract.model_layout import DEFAULT_LAYOUT_MODEL_NAME
from scripts.benchmark_ingest_standard import PROFILE_ENVS, git_commit, percentile


RESULTS_ROOT = Path("results/ingest_benchmark_scientific")
SCIENTIFIC_THRESHOLDS = (0.5, 0.75)
DOCLAYNET_CLASSES = [
    "heading",
    "paragraph",
    "list_item",
    "table",
    "figure",
    "caption",
    "metadata",
]
PUBTABLES_CLASSES = ["table"]
DOCLAYNET_LABELS = [
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
]
DOCLAYNET_TO_CANONICAL = {
    "Caption": "caption",
    "Footnote": "metadata",
    "Formula": "paragraph",
    "List-item": "list_item",
    "Page-footer": "metadata",
    "Page-header": "metadata",
    "Picture": "figure",
    "Section-header": "heading",
    "Table": "table",
    "Text": "paragraph",
    "Title": "heading",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scientific ingest benchmark")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["baseline", "model_routed_doclaynet"],
        help="Profiles to benchmark",
    )
    parser.add_argument(
        "--doclaynet-root",
        type=Path,
        default=Path("data/benchmarks/doclaynet"),
        help="Root of extracted official DocLayNet core dataset",
    )
    parser.add_argument(
        "--doclaynet-split",
        default="test",
        choices=["train", "val", "validation", "test"],
        help="DocLayNet split",
    )
    parser.add_argument(
        "--doclaynet-limit",
        type=int,
        default=0,
        help="Optional entry limit for DocLayNet (0 means all available)",
    )
    parser.add_argument(
        "--skip-doclaynet",
        action="store_true",
        help="Skip DocLayNet evaluation",
    )
    parser.add_argument(
        "--pubtables-root",
        type=Path,
        default=Path("data/benchmarks/pubtables_detection"),
        help="Root of extracted official PubTables-1M detection dataset",
    )
    parser.add_argument(
        "--pubtables-split",
        default="test",
        choices=["train", "val", "validation", "test"],
        help="PubTables detection split",
    )
    parser.add_argument(
        "--pubtables-limit",
        type=int,
        default=0,
        help="Optional entry limit for PubTables-1M detection (0 means all available)",
    )
    parser.add_argument(
        "--skip-pubtables",
        action="store_true",
        help="Skip PubTables-1M detection evaluation",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--profile", default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def _normalize_split_name(split: str) -> str:
    normalized = split.strip().lower()
    if normalized == "validation":
        return "val"
    return normalized


def load_doclaynet_entries(root: Path, split: str, limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    split_name = _normalize_split_name(split)
    core_root = None
    for candidate in (
        root / "extracted" / "DocLayNet_core",
        root / "DocLayNet_core",
        root,
    ):
        if (candidate / "COCO").exists() and (candidate / "PNG").exists():
            core_root = candidate
            break

    if core_root is None:
        return [], [{"type": "missing_dataset_root", "dataset": "doclaynet", "root": str(root)}]

    json_name = "val.json" if split_name == "val" else f"{split_name}.json"
    json_path = core_root / "COCO" / json_name
    if not json_path.exists():
        return [], [{"type": "missing_annotation_file", "dataset": "doclaynet", "path": str(json_path)}]

    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    ann_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in payload.get("annotations", []):
        ann_by_image[int(ann["image_id"])].append(ann)

    entries: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for image in payload.get("images", []):
        image_path = core_root / "PNG" / image["file_name"]
        if not image_path.exists():
            issues.append(
                {
                    "type": "missing_image",
                    "dataset": "doclaynet",
                    "file": image["file_name"],
                }
            )
            continue

        gt_regions = []
        for ann in ann_by_image.get(int(image["id"]), []):
            category_id = int(ann["category_id"]) - 1
            if category_id < 0 or category_id >= len(DOCLAYNET_LABELS):
                continue
            canonical = DOCLAYNET_TO_CANONICAL.get(DOCLAYNET_LABELS[category_id])
            if canonical is None:
                continue
            x, y, w, h = [float(v) for v in ann["bbox"]]
            gt_regions.append(
                {
                    "block_type": canonical,
                    "bbox": (x, y, x + w, y + h),
                }
            )

        entries.append(
            {
                "dataset": "doclaynet",
                "entry_id": str(image["id"]),
                "rel_path": image["file_name"],
                "image_path": image_path,
                "width": int(image["width"]),
                "height": int(image["height"]),
                "gt_regions": gt_regions,
                "meta": {
                    "doc_category": image.get("doc_category"),
                    "collection": image.get("collection"),
                    "doc_name": image.get("doc_name"),
                    "page_no": image.get("page_no"),
                },
            }
        )

        if limit > 0 and len(entries) >= limit:
            break

    return entries, issues


def load_pubtables_entries(root: Path, split: str, limit: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    split_name = _normalize_split_name(split)
    split_title = {"train": "Train", "val": "Val", "test": "Test"}[split_name]

    image_dir = _find_first_existing(
        [
            root / "extracted" / "images" / split_name,
            root / f"PubTables-1M-Detection_Images_{split_title}",
            root / "images" / split_name,
            root / "images",
        ]
    )
    annotation_dir = _find_first_existing(
        [
            root / "extracted" / "annotations" / split_name,
            root / f"PubTables-1M-Detection_Annotations_{split_title}",
            root / "annotations" / split_name,
            root / "annotations",
        ]
    )

    if image_dir is None or annotation_dir is None:
        return [], [
            {
                "type": "missing_dataset_root",
                "dataset": "pubtables_detection",
                "root": str(root),
            }
        ]

    xml_paths = sorted(annotation_dir.glob("*.xml"))
    if limit > 0:
        xml_paths = xml_paths[:limit]

    entries: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for xml_path in xml_paths:
        tree = ET.parse(xml_path)
        root_node = tree.getroot()
        filename = root_node.findtext("filename") or f"{xml_path.stem}.jpg"
        image_path = image_dir / filename
        if not image_path.exists():
            alt_candidates = list(image_dir.glob(f"{xml_path.stem}.*"))
            if alt_candidates:
                image_path = alt_candidates[0]
            else:
                issues.append(
                    {
                        "type": "missing_image",
                        "dataset": "pubtables_detection",
                        "file": filename,
                    }
                )
                continue

        size_node = root_node.find("size")
        width = int(float(size_node.findtext("width", "0"))) if size_node is not None else 0
        height = int(float(size_node.findtext("height", "0"))) if size_node is not None else 0

        gt_regions = []
        for obj in root_node.findall("object"):
            name = (obj.findtext("name") or "").strip().lower()
            if name != "table":
                continue
            bbox_node = obj.find("bndbox")
            if bbox_node is None:
                continue
            x0 = float(bbox_node.findtext("xmin", "0"))
            y0 = float(bbox_node.findtext("ymin", "0"))
            x1 = float(bbox_node.findtext("xmax", "0"))
            y1 = float(bbox_node.findtext("ymax", "0"))
            gt_regions.append({"block_type": "table", "bbox": (x0, y0, x1, y1)})

        entries.append(
            {
                "dataset": "pubtables_detection",
                "entry_id": xml_path.stem,
                "rel_path": image_path.name,
                "image_path": image_path,
                "width": width,
                "height": height,
                "gt_regions": gt_regions,
                "meta": {},
            }
        )

    return entries, issues


def _find_first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _synthesize_single_page_pdf(image_path: Path, pdf_path: Path) -> tuple[float, float]:
    image_doc = fitz.open(str(image_path))
    pdf_bytes = image_doc.convert_to_pdf()
    pdf_doc = fitz.open("pdf", pdf_bytes)
    page = pdf_doc[0]
    page_width = float(page.rect.width)
    page_height = float(page.rect.height)
    pdf_doc.save(str(pdf_path))
    pdf_doc.close()
    image_doc.close()
    return page_width, page_height


def _bbox_iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0

    inter = (x1 - x0) * (y1 - y0)
    left_area = max(1.0, (left[2] - left[0]) * (left[3] - left[1]))
    right_area = max(1.0, (right[2] - right[0]) * (right[3] - right[1]))
    return inter / (left_area + right_area - inter)


def _greedy_match(
    gt_regions: list[dict[str, Any]],
    pred_regions: list[dict[str, Any]],
    *,
    iou_threshold: float,
) -> list[tuple[int, int, float]]:
    pairs: list[tuple[float, int, int]] = []
    for gt_idx, gt in enumerate(gt_regions):
        for pred_idx, pred in enumerate(pred_regions):
            if gt["block_type"] != pred["block_type"]:
                continue
            iou = _bbox_iou(tuple(gt["bbox"]), tuple(pred["bbox"]))
            if iou >= iou_threshold:
                pairs.append((iou, gt_idx, pred_idx))

    matches: list[tuple[int, int, float]] = []
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    for iou, gt_idx, pred_idx in sorted(pairs, key=lambda item: item[0], reverse=True):
        if gt_idx in matched_gt or pred_idx in matched_pred:
            continue
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
        matches.append((gt_idx, pred_idx, iou))

    return matches


def _canonical_pred_regions(
    *,
    blocks: list[Any],
    image_width: int,
    image_height: int,
    page_width: float,
    page_height: float,
) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    scale_x = (image_width / page_width) if page_width else 1.0
    scale_y = (image_height / page_height) if page_height else 1.0

    for block in blocks:
        if block.bbox is None:
            continue
        block_type = str(block.block_type or "").strip().lower()
        if block_type not in set(DOCLAYNET_CLASSES) | {"table"}:
            continue
        x0, y0, x1, y1 = block.bbox
        regions.append(
            {
                "block_type": block_type,
                "bbox": (
                    float(x0) * scale_x,
                    float(y0) * scale_y,
                    float(x1) * scale_x,
                    float(y1) * scale_y,
                ),
                "text": (block.text or "").strip(),
                "route_backend": (block.meta or {}).get("route_backend", block.source_mode),
            }
        )
    return regions


def _record_metric_counts(
    record: dict[str, Any],
    *,
    prefix: str,
    classes: list[str],
    gt_regions: list[dict[str, Any]],
    pred_regions: list[dict[str, Any]],
) -> None:
    for threshold in SCIENTIFIC_THRESHOLDS:
        matches = _greedy_match(gt_regions, pred_regions, iou_threshold=threshold)
        matched_gt = {item[0] for item in matches}
        matched_pred = {item[1] for item in matches}
        suffix = int(threshold * 100)

        for cls in classes:
            gt_idx = [i for i, region in enumerate(gt_regions) if region["block_type"] == cls]
            pred_idx = [i for i, region in enumerate(pred_regions) if region["block_type"] == cls]
            tp = sum(1 for gt_i, pred_i, _ in matches if gt_i in gt_idx and pred_i in pred_idx)
            fp = len(pred_idx) - tp
            fn = len(gt_idx) - tp
            record[f"{prefix}_iou{suffix}_{cls}_tp"] = tp
            record[f"{prefix}_iou{suffix}_{cls}_fp"] = fp
            record[f"{prefix}_iou{suffix}_{cls}_fn"] = fn

        record[f"{prefix}_iou{suffix}_match_count"] = len(matches)
        record[f"{prefix}_iou{suffix}_matched_pred_indices"] = sorted(matched_pred)
        record[f"{prefix}_iou{suffix}_matched_gt_indices"] = sorted(matched_gt)


def evaluate_scientific_entry(entry: dict[str, Any]) -> dict[str, Any]:
    from app.ingest.pipeline import ingest_pdf

    with tempfile.TemporaryDirectory(prefix=f"bench_{entry['dataset']}_") as tmp_dir:
        pdf_path = Path(tmp_dir) / f"{entry['entry_id']}.pdf"
        page_width, page_height = _synthesize_single_page_pdf(entry["image_path"], pdf_path)

        started = time.perf_counter()
        out = ingest_pdf(pdf_path)
        elapsed = time.perf_counter() - started

    blocks = out.get("blocks", [])
    probe = out.get("probe", {})
    pred_regions = _canonical_pred_regions(
        blocks=blocks,
        image_width=int(entry["width"]),
        image_height=int(entry["height"]),
        page_width=page_width,
        page_height=page_height,
    )

    record: dict[str, Any] = {
        "dataset": entry["dataset"],
        "entry_id": entry["entry_id"],
        "file": entry["rel_path"],
        "success": 1,
        "elapsed_sec": elapsed,
        "used_backend": out.get("used_backend") or "NA",
        "probe_mode": probe.get("probe_detected_mode") or "NA",
        "gt_region_count": len(entry["gt_regions"]),
        "pred_region_count": len(pred_regions),
        "route_counts": dict(Counter(region["route_backend"] for region in pred_regions)),
        "pred_block_type_counts": dict(Counter(region["block_type"] for region in pred_regions)),
        "page_count": len(out.get("pages", [])),
        "block_count": len(blocks),
        "chunk_count": len(out.get("chunks", [])),
    }

    if entry["dataset"] == "doclaynet":
        _record_metric_counts(
            record,
            prefix="doclaynet",
            classes=DOCLAYNET_CLASSES,
            gt_regions=entry["gt_regions"],
            pred_regions=pred_regions,
        )
    elif entry["dataset"] == "pubtables_detection":
        _record_metric_counts(
            record,
            prefix="pubtables",
            classes=PUBTABLES_CLASSES,
            gt_regions=entry["gt_regions"],
            pred_regions=[region for region in pred_regions if region["block_type"] == "table"],
        )

        table_preds = [region for region in pred_regions if region["block_type"] == "table"]
        nonempty = [
            region
            for region in table_preds
            if region["text"] and region["text"].strip().lower() not in {"table"}
        ]
        record["pubtables_pred_table_count"] = len(table_preds)
        record["pubtables_pred_table_nonempty_count"] = len(nonempty)

    record.update({f"meta_{k}": v for k, v in entry["meta"].items()})
    return record


def benchmark_profile_on_dataset(
    *,
    profile_name: str,
    dataset_name: str,
    entries: list[dict[str, Any]],
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for entry in entries:
        try:
            records.append(evaluate_scientific_entry(entry))
        except Exception as e:
            records.append(
                {
                    "dataset": dataset_name,
                    "entry_id": entry["entry_id"],
                    "file": entry["rel_path"],
                    "success": 0,
                    "error": str(e),
                }
            )

    return {
        "profile": profile_name,
        "dataset": dataset_name,
        "issues": issues,
        "records": records,
        "summary": summarize_dataset_records(dataset_name, records),
    }


def summarize_dataset_records(dataset_name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    success_records = [r for r in records if int(r.get("success", 0)) == 1]
    error_records = [r for r in records if int(r.get("success", 0)) == 0]
    latencies = sorted(float(r.get("elapsed_sec", 0.0) or 0.0) for r in success_records)

    summary: dict[str, Any] = {
        "dataset": dataset_name,
        "images_total": len(records),
        "images_success": len(success_records),
        "images_failed": len(error_records),
        "success_rate": (len(success_records) / len(records)) if records else 0.0,
        "latency_mean_sec": statistics.mean(latencies) if latencies else 0.0,
        "latency_median_sec": statistics.median(latencies) if latencies else 0.0,
        "latency_p95_sec": percentile(latencies, 0.95) if latencies else 0.0,
        "backend_counts": dict(Counter(str(r.get("used_backend", "NA")) for r in success_records)),
        "route_counts": dict(
            Counter(
                route
                for r in success_records
                for route, count in (r.get("route_counts") or {}).items()
                for _ in range(int(count))
            )
        ),
        "errors": [
            {"file": r.get("file"), "error": r.get("error")}
            for r in error_records
        ],
    }

    if dataset_name == "doclaynet":
        summary["metrics"] = _summarize_detection_counts(
            success_records,
            prefix="doclaynet",
            classes=DOCLAYNET_CLASSES,
        )
    elif dataset_name == "pubtables_detection":
        summary["metrics"] = _summarize_detection_counts(
            success_records,
            prefix="pubtables",
            classes=PUBTABLES_CLASSES,
        )
        total_pred_tables = sum(int(r.get("pubtables_pred_table_count", 0)) for r in success_records)
        nonempty_pred_tables = sum(
            int(r.get("pubtables_pred_table_nonempty_count", 0)) for r in success_records
        )
        summary["table_extraction"] = {
            "pred_table_count": total_pred_tables,
            "pred_table_nonempty_count": nonempty_pred_tables,
            "pred_table_nonempty_rate": (
                nonempty_pred_tables / total_pred_tables if total_pred_tables else 0.0
            ),
        }

    return summary


def _summarize_detection_counts(
    records: list[dict[str, Any]],
    *,
    prefix: str,
    classes: list[str],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for threshold in SCIENTIFIC_THRESHOLDS:
        suffix = int(threshold * 100)
        micro_tp = 0
        micro_fp = 0
        micro_fn = 0
        per_class = {}

        for cls in classes:
            tp = sum(int(r.get(f"{prefix}_iou{suffix}_{cls}_tp", 0)) for r in records)
            fp = sum(int(r.get(f"{prefix}_iou{suffix}_{cls}_fp", 0)) for r in records)
            fn = sum(int(r.get(f"{prefix}_iou{suffix}_{cls}_fn", 0)) for r in records)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            per_class[cls] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            micro_tp += tp
            micro_fp += fp
            micro_fn += fn

        micro_precision = micro_tp / (micro_tp + micro_fp) if (micro_tp + micro_fp) else 0.0
        micro_recall = micro_tp / (micro_tp + micro_fn) if (micro_tp + micro_fn) else 0.0
        micro_f1 = (
            2 * micro_precision * micro_recall / (micro_precision + micro_recall)
            if (micro_precision + micro_recall)
            else 0.0
        )

        active_classes = [metrics for metrics in per_class.values() if (metrics["tp"] + metrics["fp"] + metrics["fn"]) > 0]
        macro_f1 = statistics.mean(item["f1"] for item in active_classes) if active_classes else 0.0

        metrics[f"iou{suffix}"] = {
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
            "per_class": per_class,
        }

    return metrics


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_profile_comparison(profile_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = defaultdict(dict)
    for payload in profile_payloads:
        grouped[payload["dataset"]][payload["profile"]] = payload["summary"]

    comparison: dict[str, Any] = {}
    for dataset_name, summaries in grouped.items():
        baseline = summaries.get("baseline")
        candidate = summaries.get("model_routed_doclaynet")
        if not baseline or not candidate:
            continue

        dataset_comparison: dict[str, Any] = {}
        scalar_metrics = [
            "success_rate",
            "latency_mean_sec",
            "latency_median_sec",
            "latency_p95_sec",
        ]
        for metric in scalar_metrics:
            base_value = float(baseline.get(metric, 0.0) or 0.0)
            cand_value = float(candidate.get(metric, 0.0) or 0.0)
            dataset_comparison[metric] = {
                "baseline": base_value,
                "model_routed_doclaynet": cand_value,
                "delta_absolute": cand_value - base_value,
                "delta_relative": ((cand_value - base_value) / base_value) if base_value else None,
            }

        if dataset_name == "doclaynet":
            for key in ("micro_f1", "macro_f1"):
                base_value = float(baseline["metrics"]["iou50"][key])
                cand_value = float(candidate["metrics"]["iou50"][key])
                dataset_comparison[f"iou50_{key}"] = {
                    "baseline": base_value,
                    "model_routed_doclaynet": cand_value,
                    "delta_absolute": cand_value - base_value,
                    "delta_relative": ((cand_value - base_value) / base_value) if base_value else None,
                }

        if dataset_name == "pubtables_detection":
            base_value = float(baseline["metrics"]["iou50"]["micro_f1"])
            cand_value = float(candidate["metrics"]["iou50"]["micro_f1"])
            dataset_comparison["iou50_table_micro_f1"] = {
                "baseline": base_value,
                "model_routed_doclaynet": cand_value,
                "delta_absolute": cand_value - base_value,
                "delta_relative": ((cand_value - base_value) / base_value) if base_value else None,
            }

        comparison[dataset_name] = dataset_comparison

    return comparison


def render_markdown_summary(
    *,
    metadata: dict[str, Any],
    profile_payloads: list[dict[str, Any]],
    comparison: dict[str, Any],
) -> str:
    lines = [
        "# Scientific Ingest Benchmark",
        "",
        f"- Timestamp: {metadata['timestamp_utc']}",
        f"- Git commit: {metadata['git_commit']}",
        f"- Chosen model: {metadata['chosen_model']}",
        "",
    ]

    for payload in profile_payloads:
        summary = payload["summary"]
        lines.extend(
            [
                f"## {payload['dataset']} / {payload['profile']}",
                "",
                f"- Success rate: {summary['success_rate']:.3f}",
                f"- Mean latency: {summary['latency_mean_sec']:.3f}s",
                f"- P95 latency: {summary['latency_p95_sec']:.3f}s",
                f"- Backend counts: `{json.dumps(summary['backend_counts'], ensure_ascii=False, sort_keys=True)}`",
                f"- Route counts: `{json.dumps(summary['route_counts'], ensure_ascii=False, sort_keys=True)}`",
            ]
        )

        if payload["dataset"] == "doclaynet":
            lines.append(f"- DocLayNet micro F1@0.50: {summary['metrics']['iou50']['micro_f1']:.3f}")
            lines.append(f"- DocLayNet macro F1@0.50: {summary['metrics']['iou50']['macro_f1']:.3f}")
        elif payload["dataset"] == "pubtables_detection":
            lines.append(f"- PubTables table micro F1@0.50: {summary['metrics']['iou50']['micro_f1']:.3f}")
            extraction = summary.get("table_extraction", {})
            lines.append(f"- PubTables non-empty predicted table rate: {extraction.get('pred_table_nonempty_rate', 0.0):.3f}")

        if payload.get("issues"):
            lines.append(f"- Dataset issues: `{json.dumps(payload['issues'], ensure_ascii=False)}`")

        lines.append("")

    if comparison:
        lines.extend(["## Comparison", ""])
        for dataset_name, dataset_metrics in comparison.items():
            lines.append(f"### {dataset_name}")
            for metric, values in dataset_metrics.items():
                rel = values["delta_relative"]
                rel_str = f"{rel:.3f}" if rel is not None else "NA"
                lines.append(
                    f"- {metric}: baseline={values['baseline']:.3f}, model={values['model_routed_doclaynet']:.3f}, delta={values['delta_absolute']:.3f}, rel={rel_str}"
                )
            lines.append("")

    return "\n".join(lines)


def run_profile_subprocess(profile: str, args: argparse.Namespace) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(PROFILE_ENVS[profile])

    cmd = [
        sys.executable,
        __file__,
        "--worker",
        "--profile",
        profile,
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
    ]
    cmd = [part for part in cmd if part]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Scientific benchmark worker failed for profile={profile}\n"
            f"returncode={result.returncode}\n"
            f"stderr:\n{result.stderr}\n"
            f"stdout:\n{result.stdout}"
        )

    return json.loads(result.stdout)


def worker_entry(args: argparse.Namespace) -> None:
    if args.profile not in PROFILE_ENVS:
        raise ValueError(f"Unknown profile: {args.profile}")

    datasets: list[dict[str, Any]] = []
    if not args.skip_doclaynet:
        doclaynet_entries, doclaynet_issues = load_doclaynet_entries(
            args.doclaynet_root, args.doclaynet_split, args.doclaynet_limit
        )
        datasets.append(
            benchmark_profile_on_dataset(
                profile_name=args.profile,
                dataset_name="doclaynet",
                entries=doclaynet_entries,
                issues=doclaynet_issues,
            )
        )
    if not args.skip_pubtables:
        pubtables_entries, pubtables_issues = load_pubtables_entries(
            args.pubtables_root, args.pubtables_split, args.pubtables_limit
        )
        datasets.append(
            benchmark_profile_on_dataset(
                profile_name=args.profile,
                dataset_name="pubtables_detection",
                entries=pubtables_entries,
                issues=pubtables_issues,
            )
        )

    payload = {
        "profile": args.profile,
        "datasets": datasets,
    }
    print(json.dumps(payload, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    if args.worker:
        worker_entry(args)
        return

    for profile in args.profiles:
        if profile not in PROFILE_ENVS:
            raise ValueError(f"Unknown profile: {profile}")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (RESULTS_ROOT / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    profile_payloads: list[dict[str, Any]] = []
    for profile in args.profiles:
        print(f"[scientific] running profile={profile}", file=sys.stderr)
        worker_payload = run_profile_subprocess(profile, args)
        for dataset_payload in worker_payload["datasets"]:
            profile_payloads.append(dataset_payload)
            save_json(
                output_dir / f"{dataset_payload['dataset']}_{dataset_payload['profile']}.json",
                dataset_payload,
            )
            save_csv(
                output_dir / f"{dataset_payload['dataset']}_{dataset_payload['profile']}.csv",
                dataset_payload["records"],
            )

    comparison = build_profile_comparison(profile_payloads)
    metadata = {
        "timestamp_utc": timestamp,
        "git_commit": git_commit(),
        "chosen_model": DEFAULT_LAYOUT_MODEL_NAME,
        "profiles": args.profiles,
        "doclaynet_root": str(args.doclaynet_root),
        "pubtables_root": str(args.pubtables_root),
        "doclaynet_split": args.doclaynet_split,
        "pubtables_split": args.pubtables_split,
        "doclaynet_limit": args.doclaynet_limit,
        "pubtables_limit": args.pubtables_limit,
        "skip_doclaynet": args.skip_doclaynet,
        "skip_pubtables": args.skip_pubtables,
    }

    summary = {
        "metadata": metadata,
        "datasets": {
            f"{payload['dataset']}::{payload['profile']}": payload["summary"]
            for payload in profile_payloads
        },
        "comparison": comparison,
    }
    save_json(output_dir / "scientific_summary.json", summary)
    save_json(output_dir / "scientific_comparison.json", comparison)
    (output_dir / "scientific_summary.md").write_text(
        render_markdown_summary(
            metadata=metadata,
            profile_payloads=profile_payloads,
            comparison=comparison,
        ),
        encoding="utf-8",
    )

    print(str(output_dir))


if __name__ == "__main__":
    main()
