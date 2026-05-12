from __future__ import annotations

import argparse
import json
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create local JSONL manifests for text/OCR ingest benchmarks from real PDFs")
    parser.add_argument("--pdf-dir", type=Path, default=ROOT / "data" / "real_pdfs")
    parser.add_argument("--text-out", type=Path, default=ROOT / "data" / "benchmarks" / "text_extraction")
    parser.add_argument("--ocr-out", type=Path, default=ROOT / "data" / "benchmarks" / "ocr")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    if args.limit > 0:
        pdfs = pdfs[: args.limit]

    text_rows = []
    ocr_rows = []
    for pdf_path in pdfs:
        gt_text, ordered_text = _extract_reference_text(pdf_path)
        if not gt_text.strip():
            continue
        row = {
            "doc_id": pdf_path.stem,
            "pdf_path": str(pdf_path.resolve()),
            "ground_truth": {
                "text": gt_text,
                "ordered_text": ordered_text,
            },
            "metadata": {
                "source": "local_real_pdf_manifest",
                "note": "Reference text was generated from the PDF text layer for local regression benchmarking.",
            },
        }
        text_rows.append(row)
        ocr_rows.append(row)

    args.text_out.mkdir(parents=True, exist_ok=True)
    args.ocr_out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.text_out / "bastkorzen_samples.jsonl", text_rows)
    _write_jsonl(args.ocr_out / "ocr_samples.jsonl", ocr_rows)
    print(
        json.dumps(
            {
                "pdf_count": len(pdfs),
                "text_samples": len(text_rows),
                "ocr_samples": len(ocr_rows),
                "text_manifest": str(args.text_out / "bastkorzen_samples.jsonl"),
                "ocr_manifest": str(args.ocr_out / "ocr_samples.jsonl"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _extract_reference_text(pdf_path: Path) -> tuple[str, list[str]]:
    doc = fitz.open(str(pdf_path))
    page_texts: list[str] = []
    ordered: list[str] = []
    for page in doc:
        page_texts.append(page.get_text("text", sort=True).strip())
        for block in page.get_text("blocks", sort=True) or []:
            text = str(block[4]).strip()
            if text:
                ordered.append(text)
    doc.close()
    return "\n".join(text for text in page_texts if text).strip(), ordered


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
