from __future__ import annotations

import re


CAPTION_RE = re.compile(r"^(?:figure|fig\.?|table|hinh|bang)\s+\d+(?:[\.:]\s*|\s+-\s+).+", re.I)


def classify_region(region: dict) -> str:
    text = (region.get("text") or "").strip()
    bbox = region.get("bbox")
    page_height = float(region.get("page_height") or 0.0)

    if not text:
        return "empty"

    if bbox and len(bbox) >= 4 and page_height > 0:
        y0 = float(bbox[1])
        y1 = float(bbox[3])
        near_top = y1 <= page_height * 0.075
        near_bottom = y0 >= page_height * 0.925
        if len(text) <= 160 and (near_top or near_bottom):
            return "header" if near_top else "footer"

    if re.match(r"^#{1,6}\s+", text):
        return "heading"

    if CAPTION_RE.match(text):
        return "caption"

    if "|" in text and "\n" in text:
        return "table"

    if len(text) < 120 and (text.isupper() or re.match(r"^\d+(\.\d+)*\s+", text)):
        return "heading"

    if re.match(r"^[-*+]\s+", text):
        return "list_item"

    return "paragraph"
