from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import fitz


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "ingest_layout_benchmark"
UNICODE_FONT = Path("C:/Windows/Fonts/arial.ttf")


def _sample_png_bytes() -> bytes:
    from PIL import Image

    image = Image.new("RGB", (32, 18), color=(210, 52, 52))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _write_manifest(rows: list[dict[str, Any]]) -> None:
    manifest = {
        "version": 1,
        "description": "Synthetic layout-aware ingest benchmark for text, headings, captions, metadata, and tables.",
        "documents": rows,
    }
    (DATA_ROOT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _attach_common_expectations(row: dict[str, Any], ordered_text: list[str]) -> dict[str, Any]:
    row["expected_order"] = ordered_text
    row["expected_full_text"] = "\n".join(ordered_text)
    row["forbidden_substrings"] = ["Page 1 of", "CONFIDENTIAL FOOTER", "HEADER REPEAT"]
    return row


def _insert_lines(page: fitz.Page, lines: list[tuple[str, float, tuple[float, float]]]) -> None:
    for text, size, point in lines:
        kwargs: dict[str, Any] = {"fontsize": size}
        if UNICODE_FONT.exists():
            kwargs["fontfile"] = str(UNICODE_FONT)
        page.insert_text(point, text, **kwargs)


def _create_mixed_policy_pdf(path: Path) -> dict[str, Any]:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    _insert_lines(
        page,
        [
            ("BOXTALK Ingest Quality Fixture", 18, (72, 72)),
            ("Document type: Synthetic policy", 10, (72, 100)),
            ("1. Overview", 15, (72, 136)),
            ("This fixture checks whether PDF text blocks keep heading and paragraph structure.", 11, (72, 162)),
            ("- Preserve headings as heading blocks", 11, (90, 190)),
            ("- Preserve bullet items as list item blocks", 11, (90, 208)),
            ("2. Benefits Table", 15, (72, 248)),
            ("Benefit        Waiting period        Owner", 11, (72, 278)),
            ("Health plan    30 days               HR Ops", 11, (72, 296)),
            ("VPN access     Same day              IT Support", 11, (72, 314)),
            ("Figure 1: Ingest routing overview", 10, (72, 386)),
            ("3. Closing Notes", 15, (72, 430)),
            ("The routing path should keep table cells available for grounded QA.", 11, (72, 456)),
        ],
    )
    page.draw_rect(fitz.Rect(72, 340, 220, 374), color=(0.2, 0.2, 0.2), width=1.0)
    page.insert_image(fitz.Rect(250, 340, 340, 390), stream=_sample_png_bytes())
    doc.save(path)
    doc.close()
    ordered = [
        "BOXTALK Ingest Quality Fixture",
        "Document type: Synthetic policy",
        "1. Overview",
        "This fixture checks whether PDF text blocks keep heading and paragraph structure.",
        "- Preserve headings as heading blocks",
        "- Preserve bullet items as list item blocks",
        "2. Benefits Table",
        "Benefit Waiting period Owner",
        "Health plan 30 days HR Ops",
        "VPN access Same day IT Support",
        "Figure",
        "Figure 1: Ingest routing overview",
        "3. Closing Notes",
        "The routing path should keep table cells available for grounded QA.",
    ]
    return _attach_common_expectations({
        "id": "mixed_policy",
        "file": path.name,
        "expected_probe_mode": "layout",
        "expected_substrings": [
            "BOXTALK Ingest Quality Fixture",
            "Preserve headings as heading blocks",
            "VPN access",
            "IT Support",
            "Figure 1: Ingest routing overview",
        ],
        "expected_block_types": {
            "heading": 3,
            "list_item": 2,
            "table": 1,
            "caption": 1,
            "metadata": 1,
            "figure": 1,
        },
        "expected_table_cells": [
            "Benefit",
            "Waiting period",
            "Owner",
            "Health plan",
            "30 days",
            "HR Ops",
            "VPN access",
            "Same day",
            "IT Support",
        ],
        "expected_table_shape": {
            "rows": 3,
            "cols": 3,
            "headers": ["Benefit", "Waiting period", "Owner"],
        },
        "expected_table_chunk_count": 1,
        "expected_min_chunk_count": 5,
    }, ordered)


def _create_legal_pdf(path: Path) -> dict[str, Any]:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    _insert_lines(
        page,
        [
            ("LEGAL INGEST TEST POLICY", 17, (72, 72)),
            ("Version: 1.0", 10, (72, 102)),
            ("Chuong 1. Quy dinh chung", 15, (72, 140)),
            ("Dieu 1. Pham vi ap dung", 13, (72, 172)),
            ("Khoan 1. Tai lieu nay dung de danh gia nhan dien heading phap quy.", 11, (90, 200)),
            ("a) He thong can nhan dien muc chu cai la list item.", 11, (90, 220)),
            ("b) He thong can giu noi dung tieng Viet trong text layer.", 11, (90, 238)),
            ("Dieu 2. Trach nhiem", 13, (72, 278)),
            ("Bo phan QA chiu trach nhiem kiem tra grounded evidence.", 11, (90, 306)),
        ],
    )
    doc.save(path)
    doc.close()
    ordered = [
        "LEGAL INGEST TEST POLICY",
        "Version: 1.0",
        "Chuong 1. Quy dinh chung",
        "Dieu 1. Pham vi ap dung",
        "Khoan 1. Tai lieu nay dung de danh gia nhan dien heading phap quy.",
        "a) He thong can nhan dien muc chu cai la list item.",
        "b) He thong can giu noi dung tieng Viet trong text layer.",
        "Dieu 2. Trach nhiem",
        "Bo phan QA chiu trach nhiem kiem tra grounded evidence.",
    ]
    return _attach_common_expectations({
        "id": "legal_policy",
        "file": path.name,
        "expected_probe_mode": "text",
        "expected_substrings": [
            "LEGAL INGEST TEST POLICY",
            "Chuong 1. Quy dinh chung",
            "Dieu 1. Pham vi ap dung",
            "He thong can giu noi dung tieng Viet trong text layer",
        ],
        "expected_block_types": {
            "heading": 4,
            "list_item": 2,
            "metadata": 1,
        },
        "expected_table_cells": [],
        "expected_table_shape": None,
        "expected_table_chunk_count": 0,
        "expected_min_chunk_count": 3,
    }, ordered)


def _create_grid_table_pdf(path: Path) -> dict[str, Any]:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    _insert_lines(
        page,
        [
            ("4. Risk Matrix", 16, (72, 72)),
            ("Risk        Severity        Owner", 11, (72, 112)),
            ("Latency     Medium          Platform", 11, (72, 134)),
            ("OCR Error   High            Ingest", 11, (72, 156)),
            ("Table 1: Synthetic risk matrix", 10, (72, 194)),
        ],
    )
    for y in (94, 120, 142, 164, 186):
        page.draw_line((68, y), (420, y), color=(0, 0, 0), width=0.5)
    for x in (68, 170, 280, 420):
        page.draw_line((x, 94), (x, 186), color=(0, 0, 0), width=0.5)
    doc.save(path)
    doc.close()
    ordered = [
        "4. Risk Matrix",
        "Risk Severity Owner",
        "Latency Medium Platform",
        "OCR Error High Ingest",
        "Table 1: Synthetic risk matrix",
    ]
    return _attach_common_expectations({
        "id": "grid_table",
        "file": path.name,
        "expected_probe_mode": "text",
        "expected_substrings": [
            "Risk Matrix",
            "Latency",
            "OCR Error",
            "Synthetic risk matrix",
        ],
        "expected_block_types": {
            "heading": 1,
            "table": 1,
            "caption": 1,
        },
        "expected_table_cells": [
            "Risk",
            "Severity",
            "Owner",
            "Latency",
            "Medium",
            "Platform",
            "OCR Error",
            "High",
            "Ingest",
        ],
        "expected_table_shape": {
            "rows": 3,
            "cols": 3,
            "headers": ["Risk", "Severity", "Owner"],
        },
        "expected_table_chunk_count": 1,
        "expected_min_chunk_count": 3,
    }, ordered)


def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    rows = [
        _create_mixed_policy_pdf(DATA_ROOT / "mixed_policy.pdf"),
        _create_legal_pdf(DATA_ROOT / "legal_policy.pdf"),
        _create_grid_table_pdf(DATA_ROOT / "grid_table.pdf"),
    ]
    _write_manifest(rows)
    print(DATA_ROOT)


if __name__ == "__main__":
    main()
