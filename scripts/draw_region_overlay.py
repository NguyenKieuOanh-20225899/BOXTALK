from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fitz

from app.ingest.region.debug import draw_regions_debug
from app.ingest.region.detector import detect_regions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw region-level routing overlay for PDF pages.")
    parser.add_argument("pdf", type=Path, help="Input PDF path.")
    parser.add_argument("--page", type=int, default=1, help="1-based page number to render.")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--scale", type=float, default=2.0, help="Rendering scale.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    doc = fitz.open(str(args.pdf))
    try:
        page_count = len(doc)
        page_index = max(0, args.page - 1)
        if page_index >= page_count:
            raise SystemExit(f"Page {args.page} is out of range for {args.pdf} ({page_count} pages)")
        page = doc[page_index]
        regions = detect_regions(page)
        output = draw_regions_debug(page, regions, args.out, scale=args.scale)
    finally:
        doc.close()
    print(f"Rendered physical page {args.page} (0-based page_index={page_index}) of {page_count}")
    print(f"Detected {len(regions)} regions")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
