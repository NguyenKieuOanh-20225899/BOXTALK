from __future__ import annotations

import fitz


def detect_regions(page: fitz.Page) -> list[dict]:
    raw_blocks = page.get_text("blocks") or []
    raw_blocks = sorted(raw_blocks, key=lambda b: (float(b[1]), float(b[0])))

    regions: list[dict] = []
    for i, raw in enumerate(raw_blocks):
        x0, y0, x1, y1, text, *_ = raw
        regions.append(
            {
                "region_id": f"r{i:04d}",
                "bbox": (float(x0), float(y0), float(x1), float(y1)),
                "text": (text or "").strip(),
                "page_index": page.number,
            }
        )
    return regions
