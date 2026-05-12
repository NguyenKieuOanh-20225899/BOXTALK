from __future__ import annotations

from pathlib import Path

import fitz

from app.ingest.extract.table import (
    extract_table_region,
    rows_to_csv,
    rows_to_html,
    table_structure_from_positioned_cells,
    table_structure_from_text,
)
from app.ingest.reading_order import sort_in_reading_order


def test_table_region_extracts_structured_cells(tmp_path: Path) -> None:
    pdf_path = tmp_path / "table.pdf"
    doc = fitz.open()
    page = doc.new_page(width=420, height=260)
    for y, cells in [
        (72, ("Metric", "Value", "Owner")),
        (100, ("Latency", "Low", "Platform")),
        (128, ("Accuracy", "High", "QA")),
    ]:
        for x, text in zip((72, 180, 285), cells):
            page.insert_text((x, y), text, fontsize=11)
    doc.save(pdf_path)
    doc.close()

    doc = fitz.open(pdf_path)
    try:
        block = extract_table_region(doc[0], (60, 50, 360, 150), block_index=0)
    finally:
        doc.close()

    assert block is not None
    assert block.block_type == "table"
    assert block.meta["table_row_count"] == 3
    assert block.meta["table_col_count"] == 3
    assert block.meta["table_headers"] == ["Metric", "Value", "Owner"]
    assert {"row": 2, "col": 2, "text": "QA", "is_header": False} in [
        {k: v for k, v in cell.items() if k in {"row", "col", "text", "is_header"}}
        for cell in block.meta["table_cells"]
    ]
    assert "Latency,Low,Platform" in block.meta["table_csv"]
    assert "<table>" in block.meta["table_html"]
    assert block.meta["table_records"][0]["Metric"] == "Latency"


def test_table_structure_from_text_exports_csv_and_html() -> None:
    structure = table_structure_from_text(
        "Metric  Value  Owner\nLatency  Low  Platform\nAccuracy  High  QA",
        backend="test",
    )
    assert structure["table_col_count"] == 3
    assert rows_to_csv(structure["table_rows"]).splitlines()[0] == "Metric,Value,Owner"
    assert "<th>Metric</th>" in rows_to_html(structure["table_rows"])


def test_positioned_cells_trim_edge_notes_and_merge_column_intervals() -> None:
    positioned = [
        {"text": "caption outside table", "bbox": (25, 7, 335, 19)},
        {"text": "continued caption", "bbox": (52, 20, 307, 32)},
        {"text": "Parameter", "bbox": (44, 39, 100, 52)},
        {"text": "Value", "bbox": (142, 39, 174, 52)},
        {"text": "Reported value", "bbox": (208, 40, 322, 52)},
        {"text": "Cmax", "bbox": (35, 58, 107, 79)},
        {"text": "785.8", "bbox": (141, 60, 169, 73)},
        {"text": "1010-1050", "bbox": (236, 60, 294, 72)},
        {"text": "tmax", "bbox": (34, 78, 79, 101)},
        {"text": "1.5", "bbox": (150, 79, 170, 93)},
        {"text": "1.5-2", "bbox": (242, 79, 277, 92)},
        {"text": "footnote one", "bbox": (18, 160, 341, 174)},
        {"text": "footnote two", "bbox": (18, 175, 340, 186)},
    ]
    structure = table_structure_from_positioned_cells(
        positioned,
        backend="test_positioned",
        table_bbox=(18, 7, 341, 186),
    )

    assert structure["table_row_count"] == 3
    assert structure["table_col_count"] == 3
    assert structure["table_rows"][0] == ["Parameter", "Value", "Reported value"]
    assert structure["table_rows"][1] == ["Cmax", "785.8", "1010-1050"]
    assert "caption" not in structure["table_csv"]
    right_col_cells = [cell for cell in structure["table_cells"] if cell["col"] == 2]
    assert right_col_cells[1]["bbox"][0] <= 208


def test_two_column_reading_order_keeps_columns_together() -> None:
    items = [
        {"id": "right1", "bbox": (330, 100, 520, 130)},
        {"id": "left2", "bbox": (72, 135, 250, 165)},
        {"id": "title", "bbox": (70, 40, 520, 70)},
        {"id": "right2", "bbox": (330, 135, 520, 165)},
        {"id": "left1", "bbox": (72, 100, 250, 130)},
        {"id": "left3", "bbox": (72, 170, 250, 200)},
        {"id": "right3", "bbox": (330, 170, 520, 200)},
    ]
    ordered = sort_in_reading_order(
        items,
        bbox_getter=lambda item: item["bbox"],
        page_width=595,
        page_height=842,
    )
    assert [item["id"] for item in ordered] == [
        "title",
        "left1",
        "left2",
        "left3",
        "right1",
        "right2",
        "right3",
    ]
