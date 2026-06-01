from __future__ import annotations

from app.ingest.chunker import build_chunks
from app.ingest.extract.table import TableCell, cells_to_csv, cells_to_markdown, table_structure_from_rows
from app.ingest.schemas import BlockNode, TableBlock
from app.ingest.table_chunking import build_table_chunks


def _table_block() -> BlockNode:
    rows = [["Điểm chữ", "Điểm thang 4"], ["C+", "2.5"], ["B", "3.0"]]
    meta = table_structure_from_rows(rows, backend="hybrid_tatr")
    meta["table_id"] = "page_9_table_1"
    meta["caption"] = "Bảng quy đổi điểm"
    return BlockNode(
        block_id="b1",
        page_index=8,
        block_type="table",
        text="Điểm chữ | Điểm thang 4\nC+ | 2.5",
        markdown=meta["table_markdown"],
        reading_order=0,
        bbox=(0, 0, 100, 50),
        heading_path=["Bảng quy đổi điểm"],
        source_mode="layout",
        meta=meta,
    )


def test_table_cell_regular_and_span_serialization() -> None:
    cell = TableCell(table_id="t1", page=1, row_index=1, col_index=2, text="2.5")
    assert cell.to_meta()["row"] == 1
    assert cell.to_meta()["col"] == 2

    span = TableCell.from_dict({"table_id": "t1", "page": 1, "row": 0, "col": 0, "row_span": 2, "col_span": 3, "text": "Header"})
    assert span.row_span == 2
    assert span.col_span == 3
    assert span.to_dict()["row"] == 0


def test_cells_to_csv_and_markdown() -> None:
    cells = [
        TableCell(row_index=0, col_index=0, text="Điểm chữ"),
        TableCell(row_index=0, col_index=1, text="Điểm thang 4"),
        TableCell(row_index=1, col_index=0, text="C+"),
        TableCell(row_index=1, col_index=1, text="2.5"),
    ]
    assert cells_to_csv(cells) == "Điểm chữ,Điểm thang 4\nC+,2.5"
    markdown = cells_to_markdown(cells)
    assert "| Điểm chữ | Điểm thang 4 |" in markdown
    assert "| C+ | 2.5 |" in markdown


def test_table_block_from_block_node_has_cells() -> None:
    table_block = TableBlock.from_block_node(_table_block())
    assert table_block.table_id == "page_9_table_1"
    assert table_block.cells
    target = next(cell for cell in table_block.cells if cell.text == "2.5")
    assert target.row_header == "C+"
    assert target.col_header == "Điểm thang 4"


def test_table_aware_chunks_include_summary_structure_row_and_cell(monkeypatch) -> None:
    block = _table_block()
    chunks = build_table_chunks(block, start_index=3)
    strategies = {chunk.meta["chunking_strategy"] for chunk in chunks}
    assert {"table_summary", "table_structure", "table_row", "table_cell"} <= strategies
    cell_chunks = [chunk for chunk in chunks if chunk.meta["citation_target"] == "cell"]
    assert any(chunk.meta["row_header"] == "C+" and chunk.meta["col_header"] == "Điểm thang 4" for chunk in cell_chunks)
    assert any("Bảng quy đổi điểm" in chunk.text for chunk in chunks)

    text_block = BlockNode("p1", 0, "paragraph", "Một đoạn văn.", "Một đoạn văn.", 0)
    monkeypatch.setenv("BOXBIIBOO_ENABLE_TABLE_AWARE_CHUNKING", "true")
    mixed = build_chunks([text_block, block], max_chars=100)
    assert any(chunk.block_types == ["paragraph"] for chunk in mixed)
    assert any(chunk.meta.get("citation_target") == "cell" for chunk in mixed)


def test_default_table_chunking_remains_single_chunk(monkeypatch) -> None:
    monkeypatch.delenv("BOXBIIBOO_ENABLE_TABLE_AWARE_CHUNKING", raising=False)
    chunks = build_chunks([_table_block()], max_chars=100)
    assert len(chunks) == 1
    assert chunks[0].meta == {"is_table_chunk": True}
