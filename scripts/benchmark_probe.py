import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

DATASET_ROOT = Path("data/test_probe")
LABELS_FILE = DATASET_ROOT / "labels.json"

OUTPUT_JSON = Path("results/probe_results.json")
OUTPUT_CSV = Path("results/probe_results.csv")

VALID_LABELS = {"text", "layout", "ocr", "mixed"}


def load_labels(labels_file: Path) -> Dict[str, Any]:
    if not labels_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file labels: {labels_file}")

    with labels_file.open("r", encoding="utf-8") as f:
        labels = json.load(f)

    if not isinstance(labels, dict):
        raise ValueError("labels.json phải là object/dict")

    return labels


def normalize_label_entry(entry: Any) -> Dict[str, Any]:
    """
    Hỗ trợ 2 kiểu:
    1) "mixed"
    2) {"label": "mixed", "note": "..."}
    """
    if isinstance(entry, str):
        label = entry
        meta = {}
    elif isinstance(entry, dict):
        label = entry.get("label")
        meta = {k: v for k, v in entry.items() if k != "label"}
    else:
        raise ValueError(f"Label entry không hợp lệ: {entry}")

    if label not in VALID_LABELS:
        raise ValueError(f"Label không hợp lệ: {label}")

    return {"label": label, **meta}


def run_probe(pdf_path: Path) -> Dict[str, Any]:
    cmd = [sys.executable, "-m", "app.ingest.probe", str(pdf_path)]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Probe failed for {pdf_path}\n"
            f"Return code: {result.returncode}\n"
            f"STDERR:\n{result.stderr}\n"
            f"STDOUT:\n{result.stdout}"
        )

    stdout = result.stdout.strip()

    try:
        return json.loads(stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Không parse được JSON output của probe cho file {pdf_path}\n"
            f"STDOUT:\n{stdout}"
        ) from e


def get_pred_label(probe_output: Dict[str, Any]) -> str:
    pred = (
        probe_output.get("probe_mode")
        or probe_output.get("pred")
        or probe_output.get("mode")
        or "unknown"
    )
    return pred


def build_record(rel_path: str, labels: Dict[str, Any]) -> Dict[str, Any]:
    pdf_path = DATASET_ROOT / rel_path
    if not pdf_path.exists():
        raise FileNotFoundError(f"File trong labels.json không tồn tại: {pdf_path}")

    label_info = normalize_label_entry(labels[rel_path])
    gt = label_info["label"]

    probe_output = run_probe(pdf_path)
    pred = get_pred_label(probe_output)

    record = {
        "file": rel_path,
        "gt": gt,
        "pred": pred,
        "correct": int(gt == pred),

        "file_path": probe_output.get("file_path"),
        "page_count": probe_output.get("page_count"),
        "total_chars": probe_output.get("total_chars"),
        "total_blocks": probe_output.get("total_blocks"),
        "total_images": probe_output.get("total_images"),
        "avg_chars_per_page": probe_output.get("avg_chars_per_page"),
        "avg_blocks_per_page": probe_output.get("avg_blocks_per_page"),
        "avg_images_per_page": probe_output.get("avg_images_per_page"),
        "pages_with_text": probe_output.get("pages_with_text"),
        "pages_without_text": probe_output.get("pages_without_text"),
        "text_layer_ratio": probe_output.get("text_layer_ratio"),
        "empty_text_ratio": probe_output.get("empty_text_ratio"),
        "likely_scanned_ratio": probe_output.get("likely_scanned_ratio"),
        "image_heavy_ratio": probe_output.get("image_heavy_ratio"),
        "avg_text_quality": probe_output.get("avg_text_quality"),
    }

    for k, v in label_info.items():
        if k != "label":
            record[f"gt_{k}"] = v

    return record


def save_json(records: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def save_csv(records: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        return

    all_keys = []
    seen = set()
    for record in records:
        for key in record.keys():
            if key not in seen:
                seen.add(key)
                all_keys.append(key)

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    labels = load_labels(LABELS_FILE)
    rel_paths = sorted(labels.keys())

    records: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for idx, rel_path in enumerate(rel_paths, start=1):
        print(f"[{idx}/{len(rel_paths)}] Running probe: {rel_path}")
        try:
            record = build_record(rel_path, labels)
            records.append(record)
        except Exception as e:
            print(f"ERROR: {e}")
            errors.append({
                "file": rel_path,
                "error": str(e),
            })

    save_json(records, OUTPUT_JSON)
    save_csv(records, OUTPUT_CSV)

    if errors:
        error_json = OUTPUT_JSON.parent / "probe_errors.json"
        with error_json.open("w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        print(f"\nCó lỗi. Đã lưu danh sách lỗi tại: {error_json}")

    print(f"\nĐã lưu JSON: {OUTPUT_JSON}")
    print(f"Đã lưu CSV : {OUTPUT_CSV}")
    print(f"Tổng số file xử lý thành công: {len(records)}")
    print(f"Tổng số file lỗi: {len(errors)}")


if __name__ == "__main__":
    main()
