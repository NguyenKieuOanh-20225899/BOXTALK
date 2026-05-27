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
from app.ingest.chunker import build_chunks
from app.ingest.cleaners import detect_repeated_header_footer_candidates, normalize_text_for_matching
from app.ingest.schemas import BlockNode
from app.ingest.structure import enrich_structure
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


def test_single_column_policy_reading_order_does_not_chain_long_blocks() -> None:
    items = [
        {"id": "previous_d", "bbox": (70.9, 477.4, 542.2, 513.9)},
        {"id": "previous_e", "bbox": (70.9, 525.3, 542.2, 561.8)},
        {"id": "previous_5", "bbox": (70.9, 573.2, 542.2, 609.6)},
        {"id": "article", "bbox": (106.9, 621.0, 231.4, 636.6)},
        {"id": "composition", "bbox": (106.9, 648.0, 198.3, 663.5)},
        {"id": "item_a", "bbox": (106.9, 675.0, 411.3, 690.5)},
        {"id": "item_b", "bbox": (70.9, 701.8, 542.2, 738.3)},
        {"id": "item_c", "bbox": (106.9, 749.7, 542.2, 765.2)},
    ]

    ordered = sort_in_reading_order(
        items,
        bbox_getter=lambda item: item["bbox"],
        page_width=595.0,
        page_height=842.0,
    )

    assert [item["id"] for item in ordered] == [
        "previous_d",
        "previous_e",
        "previous_5",
        "article",
        "composition",
        "item_a",
        "item_b",
        "item_c",
    ]


def test_policy_chunk_keeps_heading_and_clause_list_together() -> None:
    blocks = [
        BlockNode(
            block_id="h1",
            page_index=0,
            block_type="heading",
            text="Điều 13. Ban Coi thi",
            markdown="Điều 13. Ban Coi thi",
            reading_order=0,
            heading_path=["Điều 13. Ban Coi thi"],
        ),
        BlockNode(
            block_id="s1",
            page_index=0,
            block_type="list_item",
            text="1. Thành phần:",
            markdown="1. Thành phần:",
            reading_order=1,
            heading_path=["Điều 13. Ban Coi thi"],
        ),
        BlockNode(
            block_id="a",
            page_index=0,
            block_type="list_item",
            text="a) Trưởng ban do lãnh đạo Hội đồng thi kiêm nhiệm;",
            markdown="a) Trưởng ban do lãnh đạo Hội đồng thi kiêm nhiệm;",
            reading_order=2,
            heading_path=["Điều 13. Ban Coi thi"],
        ),
        BlockNode(
            block_id="b",
            page_index=0,
            block_type="metadata",
            text="b) Phó Trưởng ban là lãnh đạo sở GDĐT;",
            markdown="b) Phó Trưởng ban là lãnh đạo sở GDĐT;",
            reading_order=3,
            heading_path=["Điều 13. Ban Coi thi"],
        ),
        BlockNode(
            block_id="c",
            page_index=1,
            block_type="metadata",
            text="c) Ủy viên, thư ký là lãnh đạo, chuyên viên các phòng của sở GDĐT.",
            markdown="c) Ủy viên, thư ký là lãnh đạo, chuyên viên các phòng của sở GDĐT.",
            reading_order=4,
            heading_path=["Điều 13. Ban Coi thi"],
        ),
    ]

    chunks = build_chunks(blocks, max_chars=160)

    combined = "\n".join(chunk.text for chunk in chunks)
    assert "Điều 13. Ban Coi thi" in chunks[0].text
    assert "a) Trưởng ban" in combined
    assert "b) Phó Trưởng ban" in combined
    assert "c) Ủy viên" in combined
    assert chunks[0].page_indices == [0, 1]


def test_repeated_body_structure_labels_are_not_removed_as_headers() -> None:
    blocks = [
        BlockNode(
            block_id=f"p{i}_section",
            page_index=i,
            block_type="list_item",
            text="1. Thành phần:",
            markdown="1. Thành phần:",
            reading_order=0,
            bbox=(106.0, 640.0, 200.0, 664.0),
        )
        for i in range(5)
    ]
    blocks.extend(
        BlockNode(
            block_id=f"p{i}_number",
            page_index=i,
            block_type="metadata",
            text="10",
            markdown="10",
            reading_order=1,
            bbox=(298.0, 36.0, 315.0, 52.0),
        )
        for i in range(5)
    )

    repeated = detect_repeated_header_footer_candidates(
        blocks,
        min_pages=3,
        repeated_ratio_threshold=0.6,
    )

    assert normalize_text_for_matching("1. Thành phần:") not in repeated
    assert normalize_text_for_matching("10") in repeated


def test_legal_heading_hierarchy_keeps_article_parent_for_numbered_subsection() -> None:
    blocks = [
        BlockNode(
            block_id="chapter",
            page_index=0,
            block_type="heading",
            text="Chương II",
            markdown="Chương II",
            reading_order=0,
        ),
        BlockNode(
            block_id="article",
            page_index=0,
            block_type="heading",
            text="Điều 13. Ban Coi thi",
            markdown="Điều 13. Ban Coi thi",
            reading_order=1,
        ),
        BlockNode(
            block_id="composition",
            page_index=0,
            block_type="heading",
            text="1. Thành phần:",
            markdown="1. Thành phần:",
            reading_order=2,
        ),
        BlockNode(
            block_id="item",
            page_index=0,
            block_type="list_item",
            text="a) Trưởng ban do lãnh đạo Hội đồng thi kiêm nhiệm;",
            markdown="a) Trưởng ban do lãnh đạo Hội đồng thi kiêm nhiệm;",
            reading_order=3,
        ),
        BlockNode(
            block_id="next_article",
            page_index=0,
            block_type="heading",
            text="Điều 14. Điểm thi",
            markdown="Điều 14. Điểm thi",
            reading_order=4,
        ),
    ]

    enriched = enrich_structure(blocks)

    assert enriched[2].heading_path == ["Chương II", "Điều 13. Ban Coi thi", "1. Thành phần:"]
    assert enriched[3].heading_path == ["Chương II", "Điều 13. Ban Coi thi", "1. Thành phần:"]
    assert enriched[4].heading_path == ["Chương II", "Điều 14. Điểm thi"]
