from __future__ import annotations

import argparse
import json
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create 25 local text-layer PDF samples for text/OCR manifest benchmarks")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "benchmarks" / "text_ocr_manifest_25")
    parser.add_argument("--count", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_dir = args.out / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for index in range(1, args.count + 1):
        doc_id = f"local_text_ocr_{index:03d}"
        title = f"Local PDF Sample {index:03d}"
        lines = [
            title,
            "1. Overview",
            f"This sample validates text extraction and reading order for document {index}.",
            "Metric Value Owner",
            f"Latency {index}ms Platform",
            f"Accuracy {90 + (index % 10)} QA",
            "Figure 1: Local benchmark figure caption",
            "Conclusion",
            "The benchmark should preserve all visible text in top-to-bottom order.",
        ]
        pdf_path = pdf_dir / f"{doc_id}.pdf"
        _write_pdf(pdf_path, lines)
        rows.append(
            {
                "doc_id": doc_id,
                "pdf_path": str(pdf_path.resolve()),
                "ground_truth": {
                    "text": "\n".join(lines),
                    "ordered_text": lines,
                },
                "metadata": {
                    "source": "generated_local_text_layer_pdf",
                    "note": "Synthetic local PDF with real text layer; useful for adapter and metric regression, not OCR scan quality.",
                },
            }
        )

    text_dir = args.out / "text_extraction"
    ocr_dir = args.out / "ocr"
    text_dir.mkdir(parents=True, exist_ok=True)
    ocr_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(text_dir / "bastkorzen_samples.jsonl", rows)
    _write_jsonl(ocr_dir / "ocr_samples.jsonl", rows)
    print(
        json.dumps(
            {
                "count": len(rows),
                "pdf_dir": str(pdf_dir),
                "text_manifest": str(text_dir / "bastkorzen_samples.jsonl"),
                "ocr_manifest": str(ocr_dir / "ocr_samples.jsonl"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _write_pdf(path: Path, lines: list[str]) -> None:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    y = 72
    for line_index, line in enumerate(lines):
        size = 16 if line_index == 0 else 11
        page.insert_text((72, y), line, fontsize=size)
        y += 28 if line_index == 0 else 20
    page.draw_rect((72, 145, 330, 210), color=(0, 0, 0), width=0.5)
    doc.save(path)
    doc.close()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
