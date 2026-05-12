from __future__ import annotations

import argparse
import json
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create 25 image-only scanned PDFs for OCR benchmark smoke runs")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "benchmarks" / "ocr_scan_25")
    parser.add_argument("--count", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = args.out / "images"
    pdf_dir = args.out / "pdfs"
    image_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for index in range(1, args.count + 1):
        doc_id = f"scan_ocr_{index:03d}"
        lines = [
            f"OCR SCAN SAMPLE {index:03d}",
            f"Invoice total {100 + index} USD",
            f"Due date 2026-05-{(index % 28) + 1:02d}",
            "Approved by QA team",
        ]
        image_path = image_dir / f"{doc_id}.png"
        pdf_path = pdf_dir / f"{doc_id}.pdf"
        _write_scan_image(image_path, lines)
        _write_image_pdf(image_path, pdf_path)
        rows.append(
            {
                "doc_id": doc_id,
                "pdf_path": str(pdf_path.resolve()),
                "ground_truth": {
                    "text": "\n".join(lines),
                    "ordered_text": lines,
                },
                "metadata": {
                    "source": "generated_image_only_scan_pdf",
                    "note": "Synthetic image-only PDF used to verify OCR backend and OCR metrics.",
                },
            }
        )

    manifest_dir = args.out / "ocr"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "ocr_samples.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "count": len(rows),
                "pdf_dir": str(pdf_dir),
                "image_dir": str(image_dir),
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _write_scan_image(path: Path, lines: list[str]) -> None:
    image = Image.new("RGB", (1200, 420), "white")
    draw = ImageDraw.Draw(image)
    try:
        title_font = ImageFont.truetype("arial.ttf", 48)
        body_font = ImageFont.truetype("arial.ttf", 40)
    except Exception:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    y = 45
    for index, line in enumerate(lines):
        font = title_font if index == 0 else body_font
        draw.text((70, y), line, fill="black", font=font)
        y += 82
    image.save(path)


def _write_image_pdf(image_path: Path, pdf_path: Path) -> None:
    doc = fitz.open()
    pix = fitz.Pixmap(str(image_path))
    page = doc.new_page(width=pix.width, height=pix.height)
    page.insert_image(page.rect, filename=str(image_path))
    doc.save(pdf_path)
    doc.close()


if __name__ == "__main__":
    main()
