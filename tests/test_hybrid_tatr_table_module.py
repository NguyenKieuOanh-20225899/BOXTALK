from __future__ import annotations

from pathlib import Path

import fitz

from app.ingest.extract.hybrid_tatr_table import (
    _crop_to_page_bbox,
    extract_pdf_word_boxes_for_region,
    is_hybrid_tatr_table_enabled,
)
from app.ingest.extract.table import extract_table_region
from app.ingest.pipeline import _enhance_table_blocks_with_hybrid_tatr
from app.ingest.schemas import BlockNode


def test_hybrid_tatr_enable_flags(monkeypatch) -> None:
    monkeypatch.delenv("BOXBIIBOO_TABLE_BACKEND", raising=False)
    monkeypatch.delenv("BOXBIIBOO_ENABLE_HYBRID_TATR_TABLES", raising=False)
    monkeypatch.delenv("BOXBIIBOO_ENABLE_PIPELINE_HYBRID_TATR_TABLES", raising=False)
    assert is_hybrid_tatr_table_enabled() is True

    monkeypatch.setenv("BOXBIIBOO_TABLE_BACKEND", "default")
    assert is_hybrid_tatr_table_enabled() is False

    monkeypatch.setenv("BOXBIIBOO_TABLE_BACKEND", "hybrid_tatr")
    assert is_hybrid_tatr_table_enabled() is True

    monkeypatch.setenv("BOXBIIBOO_TABLE_BACKEND", "")
    monkeypatch.setenv("BOXBIIBOO_ENABLE_HYBRID_TATR_TABLES", "1")
    assert is_hybrid_tatr_table_enabled() is True


def test_crop_to_page_bbox_scales_region_coordinates() -> None:
    assert _crop_to_page_bbox(
        (20, 40, 120, 80),
        table_bbox=(100, 50, 300, 150),
        scale=2.0,
    ) == (110.0, 70.0, 160.0, 90.0)


def test_extract_pdf_word_boxes_for_region(tmp_path: Path) -> None:
    pdf_path = tmp_path / "words.pdf"
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((50, 80), "Metric Value", fontsize=12)
    doc.save(pdf_path)
    doc.close()

    doc = fitz.open(pdf_path)
    try:
        words = extract_pdf_word_boxes_for_region(doc[0], (40, 60, 180, 100))
    finally:
        doc.close()

    assert [word["text"] for word in words] == ["Metric", "Value"]
    assert all(word["source"] == "pdf_text_words" for word in words)


def test_extract_table_region_routes_to_hybrid_tatr_when_enabled(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "table.pdf"
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((50, 80), "Metric Value", fontsize=12)
    doc.save(pdf_path)
    doc.close()

    from app.ingest.extract import hybrid_tatr_table

    def fake_hybrid_extract(page, bbox, *, block_index, reading_order=None, region_meta=None):
        return BlockNode(
            block_id="p0000_b0000",
            page_index=page.number,
            block_type="table",
            text="Metric | Value",
            markdown="| Metric | Value |",
            reading_order=reading_order or block_index,
            bbox=bbox,
            source_mode="layout",
            meta={**dict(region_meta or {}), "table_backend": "hybrid_tatr"},
        )

    monkeypatch.setenv("BOXBIIBOO_TABLE_BACKEND", "hybrid_tatr")
    monkeypatch.setattr(hybrid_tatr_table, "extract_hybrid_tatr_table_region", fake_hybrid_extract)

    doc = fitz.open(pdf_path)
    try:
        block = extract_table_region(doc[0], (40, 60, 180, 100), block_index=0)
    finally:
        doc.close()

    assert block is not None
    assert block.meta["table_backend"] == "hybrid_tatr"
    assert block.text == "Metric | Value"


def test_pipeline_auto_enhances_table_blocks_with_hybrid_tatr(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "pipeline_table.pdf"
    doc = fitz.open()
    page = doc.new_page(width=300, height=200)
    page.insert_text((50, 80), "Metric Value", fontsize=12)
    doc.save(pdf_path)
    doc.close()

    from app.ingest.extract import hybrid_tatr_table

    def fake_hybrid_extract(page, bbox, *, block_index, reading_order=None, region_meta=None):
        return BlockNode(
            block_id="temporary",
            page_index=page.number,
            block_type="table",
            text="Metric | Value",
            markdown="| Metric | Value |",
            reading_order=reading_order or block_index,
            bbox=bbox,
            source_mode="layout",
            meta={**dict(region_meta or {}), "table_backend": "hybrid_tatr"},
        )

    monkeypatch.delenv("BOXBIIBOO_TABLE_BACKEND", raising=False)
    monkeypatch.delenv("BOXBIIBOO_ENABLE_HYBRID_TATR_TABLES", raising=False)
    monkeypatch.delenv("BOXBIIBOO_ENABLE_PIPELINE_HYBRID_TATR_TABLES", raising=False)
    monkeypatch.setattr(hybrid_tatr_table, "extract_hybrid_tatr_table_region", fake_hybrid_extract)

    block = BlockNode(
        block_id="table_block",
        page_index=0,
        block_type="table",
        text="Metric   Value",
        markdown="Metric   Value",
        reading_order=0,
        bbox=(40, 60, 180, 100),
        source_mode="text",
        meta={"backend": "text_table"},
    )

    enhanced = _enhance_table_blocks_with_hybrid_tatr(pdf_path, [block])

    assert enhanced[0].block_id == "table_block"
    assert enhanced[0].text == "Metric | Value"
    assert enhanced[0].meta["table_backend"] == "hybrid_tatr"
    assert enhanced[0].meta["pipeline_table_backend"] == "hybrid_tatr_auto"
