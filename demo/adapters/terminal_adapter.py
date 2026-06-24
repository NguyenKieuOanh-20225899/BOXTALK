from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any


def print_result(
    result: dict[str, Any],
    *,
    files: dict[str, Path],
    color: bool = True,
    verbose: bool = False,
) -> None:
    _ = verbose
    document = result["document"]
    page = result["page"]
    summary = result["summary"]
    timing = result["timing"]
    region_routing = str((result.get("config") or {}).get("region_routing") or "on")

    print()
    _panel(
        "TIEP NHAN PDF",
        [
            f"Tep: {document['name']}",
            f"Trang: {page['number']}/{document['page_count']}",
            f"Probe mode: {document.get('probe_mode')}",
        ],
        color=color,
    )
    print()
    _step("1/4", "Tham do trang PDF", color=color)
    print(f"      Kich thuoc: {page['width']} x {page['height']}")
    print(f"      Lop van ban: {'Co' if page['has_text_layer'] else 'Khong'}")
    print(f"      Anh nhung: {page['embedded_image_count']}")

    print()
    _step(
        "2/4",
        "Phat hien vung noi dung" if region_routing != "off" else "Doc block tu lop van ban",
        color=color,
    )
    counts = summary.get("region_counts", {})
    for key in ("text", "title", "list", "image", "table", "unknown"):
        unit = "vung" if region_routing != "off" else "block"
        print(f"      {_label(key)}: {int(counts.get(key, 0))} {unit}")

    print()
    _step("3/4", "Dinh tuyen xu ly" if region_routing != "off" else "Bo qua dinh tuyen vung", color=color)
    for route in result.get("route_plan", []):
        label = route.get("block_id") or route.get("region_id") or f"region_{route.get('reading_order')}"
        actual = route.get("actual_route") or route.get("planned_route")
        print(f"      {label:<22} -> {actual}")
        reason = route.get("reason")
        if reason:
            print(f"      Ly do: {reason}")
        if route.get("error"):
            print(f"      Canh bao: {route['error']}")

    print()
    _step("4/4", "Hoan tat", color=color)
    print(f"      Tong so khoi: {summary['total_blocks']}")
    for key, path in files.items():
        print(f"      {_file_label(key)}: {path}")
    print(f"      Thoi gian: {timing['total_time_ms'] / 1000.0:.2f} giay")

    if result.get("warnings"):
        print()
        print("[CANH BAO]")
        for warning in result["warnings"]:
            print(f"- {warning}")


def print_error(message: str, *, verbose: bool = False, exc: BaseException | None = None) -> None:
    print(f"[LOI] {message}")
    if verbose and exc is not None:
        traceback.print_exception(type(exc), exc, exc.__traceback__)


def _panel(title: str, lines: list[str], *, color: bool) -> None:
    width = max([len(title), *(len(line) for line in lines)], default=40) + 4
    title_text = f" {title} "
    print("+" + title_text + "-" * max(0, width - len(title_text)) + "+")
    for line in lines:
        print("| " + line.ljust(width - 2) + " |")
    print("+" + "-" * width + "+")


def _step(index: str, title: str, *, color: bool) -> None:
    if color:
        print(f"\033[36m[{index}]\033[0m {title}")
    else:
        print(f"[{index}] {title}")


def _label(key: str) -> str:
    labels = {
        "text": "Van ban",
        "title": "Tieu de",
        "list": "Danh sach",
        "image": "Anh",
        "table": "Bang",
        "unknown": "Khong ro",
    }
    return labels.get(key, key)


def _file_label(key: str) -> str:
    labels = {
        "blocks_json": "JSON",
        "summary_json": "Tom tat",
        "text_markdown": "Van ban",
        "overlay_png": "Lop phu",
    }
    if key.startswith("table_"):
        return "Bang Markdown"
    return labels.get(key, key)
