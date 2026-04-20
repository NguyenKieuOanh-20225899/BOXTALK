from __future__ import annotations

import json
import sys
from pathlib import Path

from app.ingest.chunker import build_chunks
from app.ingest.extract.text import extract_text_region
from app.ingest.normalize import export_normalized_artifacts
from app.ingest.probe import probe_pdf
from app.ingest.region.detector import detect_regions
from app.ingest.region.classifier import classify_region
from app.ingest.region.router import route_region
from app.ingest.schemas import BlockNode, PageNode

import fitz


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python -m app.main_ingest_demo <pdf_path> <out_dir>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])

    probe_result = probe_pdf(pdf_path)
    mode = probe_result.probe_detected_mode

    print("=== PROBE ===")
    print(f"Detected mode: {mode}")
    print(f"Text quality: {probe_result.avg_text_quality:.2f}")
    print(f"Likely scanned ratio: {probe_result.likely_scanned_ratio:.2f}")
    print(f"Image heavy ratio: {probe_result.image_heavy_ratio:.2f}")
    if probe_result.notes:
        print("Notes:", probe_result.notes)

    if mode != "text":
        print(f"Detected mode={mode}. Tạm thời demo này mới implement text mode.")
        print("Bạn vẫn có thể ép test với PDF có text layer tốt.")
        sys.exit(0)

    # Simple text extraction using new pipeline
    doc = fitz.open(str(pdf_path))
    pages: list[PageNode] = []
    all_blocks: list[BlockNode] = []

    for page in doc:
        page_index = page.number
        page_label = str(page_index + 1)
        page_regions = detect_regions(page)
        page_blocks: list[BlockNode] = []

        for i, region in enumerate(page_regions):
            region_type = classify_region(region)
            backend = route_region(region_type)

            if backend == "text":
                block = extract_text_region(region, page_index, len(page_blocks))
                if block.text:
                    page_blocks.append(block)

        if page_blocks:
            all_blocks.extend(page_blocks)

        page_text = page.get_text("text", sort=True).strip()
        page_markdown = "\n\n".join(block.markdown for block in page_blocks if block.markdown)

        pages.append(
            PageNode(
                page_index=page_index,
                page_label=page_label,
                text=page_text,
                markdown=page_markdown,
                source_mode="text",
                has_ocr=False,
                has_table=False,
                meta={
                    "region_count": len(page_regions),
                    "block_count": len(page_blocks),
                },
            )
        )

    doc.close()

    chunks = build_chunks(all_blocks)
    export_normalized_artifacts(out_dir, pages, all_blocks, chunks)

    print(f"\nDone. Artifacts written to: {out_dir}")
    print(f"Pages:  {len(pages)}")
    print(f"Blocks: {len(all_blocks)}")
    print(f"Chunks: {len(chunks)}")


if __name__ == "__main__":
    main()
