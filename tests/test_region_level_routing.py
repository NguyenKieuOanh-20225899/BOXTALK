from __future__ import annotations

from pathlib import Path

import fitz

from app.ingest.extract.region_routed import extract_with_region_routed_backend
from app.ingest.region.detector import detect_regions


def _make_region_pdf(path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 30), "Page 1", fontsize=8)
    page.insert_text((72, 72), "Region Routing Demo", fontsize=18)
    page.insert_text((72, 125), "The page has normal text before a table.", fontsize=11)
    page.insert_text((72, 205), "Metric        Value        Owner", fontsize=11)
    page.insert_text((72, 228), "Latency       Low          Platform", fontsize=11)
    page.insert_text((72, 251), "Accuracy      High         QA", fontsize=11)
    page.draw_rect((390, 190, 470, 238), color=(0.2, 0.2, 0.2), width=1)
    page.insert_text((72, 300), "Figure 1: Region routing caption", fontsize=10)
    page.insert_text((72, 805), "Confidential 2026", fontsize=8)
    doc.save(path)
    doc.close()


def test_detect_regions_routes_text_table_caption_and_metadata(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "regions.pdf"
    _make_region_pdf(pdf_path)
    monkeypatch.setenv("BOXBIIBOO_ENABLE_REGION_VECTOR_FIGURES", "1")

    doc = fitz.open(pdf_path)
    try:
        regions = detect_regions(doc[0])
    finally:
        doc.close()

    kinds = [str(region.get("kind")) for region in regions]
    assert "table" in kinds
    assert "caption" in kinds
    assert "header" in kinds
    assert "footer" in kinds
    assert "image" in kinds

    table_regions = [region for region in regions if region.get("kind") == "table"]
    assert len(table_regions) == 1
    assert "Latency" in table_regions[0]["text"]


def test_region_routed_backend_extracts_table_region_without_duplicate_text(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "regions.pdf"
    _make_region_pdf(pdf_path)
    monkeypatch.setenv("BOXBIIBOO_TABLE_BACKEND", "default")
    monkeypatch.setenv("BOXBIIBOO_ENABLE_REGION_IMAGE_OCR", "0")
    monkeypatch.setenv("BOXBIIBOO_ENABLE_REGION_VECTOR_FIGURES", "1")

    pages, blocks = extract_with_region_routed_backend(pdf_path)

    table_blocks = [block for block in blocks if block.block_type == "table"]
    assert pages[0].has_table is True
    assert len(table_blocks) == 1
    assert table_blocks[0].meta["route_backend"] == "table"
    assert table_blocks[0].meta["table_row_count"] == 3
    assert table_blocks[0].meta["table_col_count"] == 3

    paragraph_text = "\n".join(block.text for block in blocks if block.block_type == "paragraph")
    assert "Latency" not in paragraph_text
    assert any(block.block_type == "caption" for block in blocks)
    assert any(block.block_type == "figure" for block in blocks)
    assert sum(1 for block in blocks if block.block_type == "metadata") >= 2
