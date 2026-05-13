from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ingest.extract.ocr import _get_ocr, _ocr_line_bbox, _run_ocr, _sort_ocr_lines


DEFAULT_SOURCE = "paddleocr_line_words"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a PubTables structure manifest with real OCR-derived word boxes. "
            "Run this in the PaddleOCR environment, then run hybrid_tatr in the PyTorch/TATR environment."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/benchmarks/pubtables_structure"))
    parser.add_argument("--manifest", default="pubtables_structure_samples.jsonl")
    parser.add_argument("--out", type=Path, default=Path("data/benchmarks/pubtables_structure_ocr_words"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--lang", default=None, help="Optional PaddleOCR language override, e.g. en")
    parser.add_argument("--device", default=None, help="Optional PaddleOCR device override, e.g. gpu:0 or cpu")
    parser.add_argument("--preprocess", default=None, help="Optional BOXBIIBOO_OCR_PREPROCESS override")
    parser.add_argument("--min-confidence", type=float, default=0.50)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.lang:
        os.environ["BOXBIIBOO_OCR_LANG"] = args.lang
    if args.device:
        os.environ["BOXBIIBOO_OCR_DEVICE"] = args.device
    if args.preprocess:
        os.environ["BOXBIIBOO_OCR_PREPROCESS"] = args.preprocess

    args.out.mkdir(parents=True, exist_ok=True)
    records = _read_manifest(args.data_dir / args.manifest)
    if args.limit > 0:
        records = records[: args.limit]

    try:
        ocr = _get_ocr()
    except Exception as exc:
        raise RuntimeError(
            "Could not initialize PaddleOCR. Run this script in the OCR environment "
            "(.venv-ocr-gpu) and keep the TATR benchmark in the PyTorch environment (.venv-gpu)."
        ) from exc

    output_records: list[dict[str, Any]] = []
    for record in records:
        image_path = _resolve_path(args.data_dir, record.get("image_path"))
        if image_path is None:
            output_records.append(
                _augment_record(
                    record,
                    [],
                    source_root=args.data_dir,
                    out_root=args.out,
                    metadata_updates={"ocr_word_error": "missing_image_path"},
                )
            )
            continue

        try:
            word_boxes, line_count = _ocr_word_boxes_for_image(
                ocr,
                image_path,
                source=args.source,
                min_confidence=args.min_confidence,
            )
            output_records.append(
                _augment_record(
                    record,
                    word_boxes,
                    source_root=args.data_dir,
                    out_root=args.out,
                    metadata_updates={"ocr_line_count": line_count},
                )
            )
        except Exception as exc:
            output_records.append(
                _augment_record(
                    record,
                    [],
                    source_root=args.data_dir,
                    out_root=args.out,
                    metadata_updates={"ocr_word_error": str(exc)},
                )
            )

    manifest_path = args.out / args.manifest
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in output_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "dataset": "pubtables_structure",
        "source_manifest": str(args.data_dir / args.manifest),
        "sample_count": len(output_records),
        "word_box_source": args.source,
        "min_confidence": args.min_confidence,
        "manifest": str(manifest_path),
    }
    (args.out / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out / "README.md").write_text(_render_readme(summary), encoding="utf-8")
    print(manifest_path)


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing PubTables structure manifest: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _ocr_word_boxes_for_image(
    ocr: Any,
    image_path: Path,
    *,
    source: str,
    min_confidence: float,
) -> tuple[list[dict[str, Any]], int]:
    from PIL import Image

    raw_lines = _run_ocr(ocr, image_path)
    with Image.open(image_path) as image:
        width, height = image.size
    lines = _sort_ocr_lines(raw_lines, page_width=float(width), page_height=float(height))
    word_boxes: list[dict[str, Any]] = []
    for line_index, line in enumerate(lines):
        word_boxes.extend(
            _line_to_word_boxes(
                line,
                line_index=line_index,
                source=source,
                min_confidence=min_confidence,
            )
        )
    return word_boxes, len(lines)


def _line_to_word_boxes(
    line: dict[str, Any],
    *,
    line_index: int = 0,
    source: str = DEFAULT_SOURCE,
    min_confidence: float = 0.50,
) -> list[dict[str, Any]]:
    text = _normalize_text(str(line.get("text") or ""))
    if not text:
        return []

    score = line.get("score")
    confidence = None
    if score is not None:
        try:
            confidence = float(score)
        except Exception:
            confidence = None
    if confidence is not None and confidence < min_confidence:
        return []

    x0, y0, x1, y1 = _ocr_line_bbox(line)
    if x1 <= x0 or y1 <= y0:
        return []

    matches = list(re.finditer(r"\S+", text))
    if not matches:
        return []

    width = x1 - x0
    text_len = max(len(text), 1)
    words: list[dict[str, Any]] = []
    for word_index, match in enumerate(matches):
        word = match.group(0)
        word_x0 = x0 + width * (match.start() / text_len)
        word_x1 = x0 + width * (match.end() / text_len)
        if word_x1 <= word_x0:
            continue
        payload = {
            "text": word,
            "bbox": [float(word_x0), float(y0), float(word_x1), float(y1)],
            "source": source,
            "line_index": line_index,
            "word_index": word_index,
            "line_text": text,
        }
        if confidence is not None:
            payload["confidence"] = confidence
        words.append(payload)
    return words


def _augment_record(
    record: dict[str, Any],
    word_boxes: list[dict[str, Any]],
    *,
    source_root: Path,
    out_root: Path,
    metadata_updates: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output = dict(record)
    output["metadata"] = dict(record.get("metadata", {}) or {})
    output["word_boxes"] = word_boxes

    source = _word_box_source(word_boxes)
    output["metadata"]["word_box_source"] = source or DEFAULT_SOURCE
    output["metadata"]["word_box_count"] = len(word_boxes)
    output["metadata"]["ocr_word_box_manifest"] = True
    if metadata_updates:
        output["metadata"].update(metadata_updates)

    for key in ("image_path", "pdf_path"):
        resolved = _resolve_path(source_root, record.get(key))
        if resolved is not None:
            output[key] = _relative(out_root, resolved)
    return output


def _word_box_source(word_boxes: list[dict[str, Any]]) -> str | None:
    for word in word_boxes:
        if word.get("source"):
            return str(word["source"])
    return None


def _resolve_path(root: Path, value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _relative(root: Path, path: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), root.resolve())
    except Exception:
        return str(path)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _render_readme(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# PubTables Structure OCR Word Boxes",
            "",
            "This manifest keeps the PubTables structure ground truth but replaces annotation-derived word boxes with OCR-derived word boxes.",
            "",
            f"- Source manifest: `{summary['source_manifest']}`",
            f"- Samples: `{summary['sample_count']}`",
            f"- Word box source: `{summary['word_box_source']}`",
            f"- Min confidence: `{summary['min_confidence']}`",
            "",
            "Run hybrid TATR:",
            "",
            "```powershell",
            ".\\.venv-gpu\\Scripts\\python.exe scripts\\benchmark_ingest_suite.py --dataset pubtables_structure --data-dir data\\benchmarks\\pubtables_structure_ocr_words --limit 25 --out results\\ingest\\pubtables_structure_ocr_words_25_hybrid_tatr --mode table --table-backend hybrid_tatr --save-predictions",
            "```",
        ]
    )


if __name__ == "__main__":
    main()
