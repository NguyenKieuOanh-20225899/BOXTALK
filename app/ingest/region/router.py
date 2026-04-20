from __future__ import annotations


def route_region(region_type: str, probe_mode: str | None = None) -> str:
    if region_type == "empty":
        return "ocr"

    if region_type == "table":
        return "layout"

    if region_type in {"heading", "list_item", "paragraph"}:
        if probe_mode == "ocr":
            return "ocr"
        if probe_mode == "layout":
            return "layout"
        return "text"

    return "text"
