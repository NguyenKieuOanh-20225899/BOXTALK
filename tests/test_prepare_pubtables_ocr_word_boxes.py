from __future__ import annotations

from pathlib import Path

from scripts.prepare_pubtables_ocr_word_boxes import _augment_record, _line_to_word_boxes


def test_line_to_word_boxes_splits_line_geometry_by_text_offsets() -> None:
    line = {
        "text": "Alpha Beta",
        "quad": [(10, 5), (110, 5), (110, 25), (10, 25)],
        "score": 0.91,
    }

    words = _line_to_word_boxes(line, line_index=2, source="test_ocr", min_confidence=0.5)

    assert [word["text"] for word in words] == ["Alpha", "Beta"]
    assert words[0]["bbox"] == [10.0, 5.0, 60.0, 25.0]
    assert words[1]["bbox"] == [70.0, 5.0, 110.0, 25.0]
    assert words[0]["source"] == "test_ocr"
    assert words[0]["line_index"] == 2
    assert words[0]["confidence"] == 0.91


def test_line_to_word_boxes_filters_low_confidence_lines() -> None:
    line = {
        "text": "Ignored",
        "quad": [(0, 0), (100, 0), (100, 20), (0, 20)],
        "score": 0.2,
    }

    assert _line_to_word_boxes(line, min_confidence=0.5) == []


def test_augment_record_retargets_paths_and_sets_metadata(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    out_root = tmp_path / "out"
    image_path = source_root / "images" / "sample.png"
    pdf_path = source_root / "pdfs" / "sample.pdf"
    image_path.parent.mkdir(parents=True)
    pdf_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image")
    pdf_path.write_bytes(b"pdf")
    out_root.mkdir()
    record = {
        "doc_id": "sample",
        "image_path": "images/sample.png",
        "pdf_path": "pdfs/sample.pdf",
        "metadata": {"benchmark": "pubtables_structure"},
    }
    words = [{"text": "A", "bbox": [1, 2, 3, 4], "source": "paddleocr_line_words"}]

    output = _augment_record(record, words, source_root=source_root, out_root=out_root)

    assert output["word_boxes"] == words
    assert output["metadata"]["word_box_source"] == "paddleocr_line_words"
    assert output["metadata"]["word_box_count"] == 1
    assert output["metadata"]["ocr_word_box_manifest"] is True
    assert Path(output["image_path"]).name == "sample.png"
    assert Path(output["pdf_path"]).name == "sample.pdf"
