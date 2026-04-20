from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


DATASET_ROOT = Path("data/test_probe")
LABELS_FILE = DATASET_ROOT / "labels.json"
RESULTS_DIR = Path("results")

RESULTS_JSON = RESULTS_DIR / "probe_results.json"
RESULTS_CSV = RESULTS_DIR / "probe_results.csv"
CONFUSION_COUNT_CSV = RESULTS_DIR / "confusion_matrix_count.csv"
CONFUSION_PERCENT_CSV = RESULTS_DIR / "confusion_matrix_percent.csv"
FEATURE_STATS_CSV = RESULTS_DIR / "feature_stats.csv"
SUMMARY_JSON = RESULTS_DIR / "eval_summary.json"
ERRORS_JSON = RESULTS_DIR / "probe_errors.json"

VALID_LABELS = ["text", "layout", "ocr", "mixed"]
FEATURES = [
    "text_layer_ratio",
    "likely_scanned_ratio",
    "avg_text_quality",
    "image_heavy_ratio",
]


def load_labels(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy labels.json: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("labels.json phải là object/dict")

    return data


def normalize_label_entry(entry: Any) -> dict[str, Any]:
    """
    Hỗ trợ 2 dạng:
    1) "mixed"
    2) {"label": "mixed", "note": "..."}
    """
    if isinstance(entry, str):
        label = entry
        meta: dict[str, Any] = {}
    elif isinstance(entry, dict):
        label = entry.get("label")
        meta = {k: v for k, v in entry.items() if k != "label"}
    else:
        raise ValueError(f"Label entry không hợp lệ: {entry!r}")

    if label not in VALID_LABELS:
        raise ValueError(f"Label không hợp lệ: {label!r}")

    return {"label": label, **meta}


def run_probe(pdf_path: Path) -> dict[str, Any]:
    cmd = [sys.executable, "-m", "app.ingest.probe", str(pdf_path)]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Probe failed: {pdf_path}\n"
            f"returncode={result.returncode}\n"
            f"stderr:\n{result.stderr}\n"
            f"stdout:\n{result.stdout}"
        )

    stdout = result.stdout.strip()
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Không parse được JSON output cho file: {pdf_path}\n"
            f"stdout:\n{stdout}"
        ) from e

    if not isinstance(payload, dict):
        raise RuntimeError(f"Probe output không phải object JSON: {pdf_path}")

    return payload


def safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_record(rel_path: str, label_entry: Any) -> dict[str, Any]:
    label_info = normalize_label_entry(label_entry)
    gt = label_info["label"]

    pdf_path = DATASET_ROOT / rel_path
    if not pdf_path.exists():
        raise FileNotFoundError(f"File không tồn tại: {pdf_path}")

    probe = run_probe(pdf_path)
    pred = probe.get("probe_detected_mode", "unknown")

    record: dict[str, Any] = {
        "file": rel_path,
        "gt": gt,
        "pred": pred,
        "correct": int(gt == pred),
        "file_path": probe.get("file_path"),
        "page_count": probe.get("page_count"),
        "total_chars": probe.get("total_chars"),
        "total_blocks": probe.get("total_blocks"),
        "total_images": probe.get("total_images"),
        "avg_chars_per_page": probe.get("avg_chars_per_page"),
        "avg_blocks_per_page": probe.get("avg_blocks_per_page"),
        "avg_images_per_page": probe.get("avg_images_per_page"),
        "pages_with_text": probe.get("pages_with_text"),
        "pages_without_text": probe.get("pages_without_text"),
        "text_layer_ratio": probe.get("text_layer_ratio"),
        "empty_text_ratio": probe.get("empty_text_ratio"),
        "likely_scanned_ratio": probe.get("likely_scanned_ratio"),
        "image_heavy_ratio": probe.get("image_heavy_ratio"),
        "avg_text_quality": probe.get("avg_text_quality"),
        "probe_detected_mode": pred,
        "notes": " | ".join(probe.get("notes", [])),
        "errors": " | ".join(probe.get("errors", [])),
    }

    for k, v in label_info.items():
        if k != "label":
            record[f"gt_{k}"] = v

    return record


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


def print_table(headers: list[str], rows: list[list[Any]]) -> None:
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    fmt = " | ".join("{:<" + str(w) + "}" for w in widths)
    sep = "-+-".join("-" * w for w in widths)

    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))
    print()


def overall_accuracy(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    correct = sum(int(r["correct"]) for r in records)
    acc = correct / total if total else 0.0
    return {
        "correct": correct,
        "total": total,
        "accuracy": acc,
    }


def per_class_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []

    for label in VALID_LABELS:
        tp = sum(1 for r in records if r["gt"] == label and r["pred"] == label)
        fp = sum(1 for r in records if r["gt"] != label and r["pred"] == label)
        fn = sum(1 for r in records if r["gt"] == label and r["pred"] != label)
        total_gt = sum(1 for r in records if r["gt"] == label)

        accuracy = tp / total_gt if total_gt else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) else 0.0
        )

        metrics.append({
            "class": label,
            "correct": tp,
            "total": total_gt,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    return metrics


def build_confusion_matrix(records: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    matrix = {
        gt: {pred: 0 for pred in VALID_LABELS}
        for gt in VALID_LABELS
    }

    for r in records:
        gt = r["gt"]
        pred = r["pred"]
        if gt not in matrix:
            matrix[gt] = {p: 0 for p in VALID_LABELS}
        if pred not in matrix[gt]:
            matrix[gt][pred] = 0
        matrix[gt][pred] += 1

    return matrix


def save_confusion_csv(
    count_path: Path,
    percent_path: Path,
    matrix: dict[str, dict[str, int]],
) -> None:
    count_rows: list[dict[str, Any]] = []
    percent_rows: list[dict[str, Any]] = []

    for gt in VALID_LABELS:
        total = sum(matrix[gt].values())

        count_row: dict[str, Any] = {"GT\\Pred": gt}
        percent_row: dict[str, Any] = {"GT\\Pred": gt}

        for pred in VALID_LABELS:
            count_row[pred] = matrix[gt][pred]
            percent_row[pred] = (matrix[gt][pred] / total) if total else 0.0

        count_row["total"] = total
        count_rows.append(count_row)
        percent_rows.append(percent_row)

    save_csv(count_path, count_rows)
    save_csv(percent_path, percent_rows)


def feature_stats(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        grouped[r["gt"]].append(r)

    rows: list[dict[str, Any]] = []
    for feature in FEATURES:
        row: dict[str, Any] = {"feature": feature}
        for label in VALID_LABELS:
            values = [safe_float(rec.get(feature)) for rec in grouped[label]]
            values = [v for v in values if v is not None]

            if values:
                row[f"{label}_mean"] = mean(values)
                row[f"{label}_std"] = stdev(values) if len(values) > 1 else 0.0
                row[f"{label}_min"] = min(values)
                row[f"{label}_max"] = max(values)
            else:
                row[f"{label}_mean"] = None
                row[f"{label}_std"] = None
                row[f"{label}_min"] = None
                row[f"{label}_max"] = None

        rows.append(row)

    return rows


def top_error_cases(records: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    errors = [r for r in records if r["gt"] != r["pred"]]

    def sort_key(r: dict[str, Any]) -> tuple:
        return (
            r["gt"],
            r["pred"],
            r["file"],
        )

    errors = sorted(errors, key=sort_key)
    return errors[:limit]


def print_summary(
    records: list[dict[str, Any]],
    class_metrics: list[dict[str, Any]],
    matrix: dict[str, dict[str, int]],
    feat_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
) -> None:
    overall = overall_accuracy(records)

    print("\n" + "=" * 72)
    print("MASTER SUMMARY")
    print("=" * 72)
    print(f"Dataset size      : {overall['total']}")
    print(f"Overall accuracy  : {overall['accuracy']:.4f} ({overall['accuracy'] * 100:.2f}%)")

    best_class = max(class_metrics, key=lambda x: x["accuracy"])
    worst_class = min(class_metrics, key=lambda x: x["accuracy"])
    print(f"Best class        : {best_class['class']} ({best_class['accuracy'] * 100:.2f}%)")
    print(f"Worst class       : {worst_class['class']} ({worst_class['accuracy'] * 100:.2f}%)")
    print()

    rows1 = []
    for m in class_metrics:
        rows1.append([
            m["class"],
            m["correct"],
            m["total"],
            f"{m['accuracy'] * 100:.2f}%",
            f"{m['precision']:.3f}",
            f"{m['recall']:.3f}",
            f"{m['f1']:.3f}",
        ])

    print("ACCURACY / PRECISION / RECALL / F1")
    print_table(
        ["Class", "Correct", "Total", "Accuracy", "Precision", "Recall", "F1"],
        rows1,
    )

    rows2: list[list[Any]] = []
    for gt in VALID_LABELS:
        total = sum(matrix[gt].values())
        rows2.append([gt] + [matrix[gt][pred] for pred in VALID_LABELS] + [total])

    print("CONFUSION MATRIX (COUNT)")
    print_table(["GT\\Pred"] + VALID_LABELS + ["Total"], rows2)

    rows3: list[list[Any]] = []
    for gt in VALID_LABELS:
        total = sum(matrix[gt].values())
        row = [gt]
        for pred in VALID_LABELS:
            value = (matrix[gt][pred] / total) if total else 0.0
            row.append(f"{value * 100:.1f}%")
        rows3.append(row)

    print("CONFUSION MATRIX (ROW %)")
    print_table(["GT\\Pred"] + VALID_LABELS, rows3)

    rows4: list[list[Any]] = []
    for row in feat_rows:
        out = [row["feature"]]
        for label in VALID_LABELS:
            m = row[f"{label}_mean"]
            s = row[f"{label}_std"]
            if m is None:
                out.append("N/A")
            else:
                out.append(f"{m:.4f} ± {s:.4f}")
        rows4.append(out)

    print("FEATURE ANALYSIS (MEAN ± STD BY GT CLASS)")
    print_table(["Feature"] + VALID_LABELS, rows4)

    print("TOP ERROR CASES")
    if not error_rows:
        print("No errors.\n")
    else:
        rows5: list[list[Any]] = []
        for r in error_rows:
            rows5.append([
                r["file"],
                r["gt"],
                r["pred"],
                r.get("text_layer_ratio"),
                r.get("likely_scanned_ratio"),
                r.get("avg_text_quality"),
                r.get("image_heavy_ratio"),
            ])
        print_table(
            [
                "File",
                "GT",
                "Pred",
                "text_layer_ratio",
                "scan_ratio",
                "text_quality",
                "image_heavy_ratio",
            ],
            rows5,
        )


def main() -> None:
    labels = load_labels(LABELS_FILE)
    rel_paths = sorted(labels.keys())

    records: list[dict[str, Any]] = []
    probe_errors: list[dict[str, Any]] = []

    for idx, rel_path in enumerate(rel_paths, start=1):
        print(f"[{idx}/{len(rel_paths)}] Running probe: {rel_path}")
        try:
            record = build_record(rel_path, labels[rel_path])
            records.append(record)
        except Exception as e:
            probe_errors.append({
                "file": rel_path,
                "error": str(e),
            })
            print(f"ERROR: {e}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    save_json(RESULTS_JSON, records)
    save_csv(RESULTS_CSV, records)
    save_json(ERRORS_JSON, probe_errors)

    class_metrics = per_class_metrics(records)
    matrix = build_confusion_matrix(records)
    feat_rows = feature_stats(records)
    error_rows = top_error_cases(records, limit=20)

    save_csv(FEATURE_STATS_CSV, feat_rows)
    save_confusion_csv(CONFUSION_COUNT_CSV, CONFUSION_PERCENT_CSV, matrix)

    summary = {
        "overall": overall_accuracy(records),
        "per_class": class_metrics,
        "confusion_matrix": matrix,
        "feature_stats": feat_rows,
        "top_errors": error_rows,
        "probe_errors": probe_errors,
    }
    save_json(SUMMARY_JSON, summary)

    print_summary(records, class_metrics, matrix, feat_rows, error_rows)

    print("Saved files:")
    print(f"- {RESULTS_JSON}")
    print(f"- {RESULTS_CSV}")
    print(f"- {CONFUSION_COUNT_CSV}")
    print(f"- {CONFUSION_PERCENT_CSV}")
    print(f"- {FEATURE_STATS_CSV}")
    print(f"- {SUMMARY_JSON}")
    print(f"- {ERRORS_JSON}")


if __name__ == "__main__":
    main()
