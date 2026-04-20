from __future__ import annotations

import re


def classify_region(region: dict) -> str:
    text = (region.get("text") or "").strip()

    if not text:
        return "empty"

    if re.match(r"^#{1,6}\s+", text):
        return "heading"

    if len(text) < 120 and (text.isupper() or re.match(r"^\d+(\.\d+)*\s+", text)):
        return "heading"

    if re.match(r"^[-*+•]\s+", text):
        return "list_item"

    if "|" in text and "\n" in text:
        return "table"

    return "paragraph"
