from __future__ import annotations

from app.ingest.extract.table import table_structure_from_rows
from app.ingest.table_reconstruct import (
    CellGraphNode,
    build_cell_graph,
    export_csv,
    export_json,
    export_markdown,
    export_table_records,
    generate_reconstruction_hypotheses,
    infer_table_schema,
    reconstruct_from_rows,
    score_table_hypothesis,
    select_best_hypothesis,
)


def _noisy_training_rows() -> list[list[str]]:
    return [
        ["Chương trình", "Người học", "Thời gian", "Khối lượng tối thiểu"],
        ["Cử nhân", "Tốt nghiệp THPT", "4 năm", "132 tín chỉ"],
        ["Kỹ sư", "Tốt nghiệp cử nhân theo chương trình tích hợp", "1,5 năm", "48 tín chỉ"],
        ["", "Tốt nghiệp cử nhân", "2 năm", "60 tín chỉ"],
        ["Thạc sĩ", "Tốt nghiệp cử nhân", "2 năm", "60 tín chỉ"],
        ["", "Tốt nghiệp cử nhân theo chương trình tích hợp", "1,5 năm", "48 tín chỉ"],
        ["Tiến sĩ", "Tốt Tốt nghiệp nghiệp thạc đại học sĩ", "3 4 năm năm", "106 151 tín tín chỉ chỉ"],
    ]


def test_build_cell_graph_assigns_words_to_row_and_column_boxes() -> None:
    words = [
        {"text": "Cử", "bbox": (5, 12, 15, 20), "confidence": 0.9},
        {"text": "nhân", "bbox": (16, 12, 30, 20), "confidence": 0.8},
        {"text": "4", "bbox": (105, 12, 110, 20), "confidence": 0.9},
        {"text": "năm", "bbox": (112, 12, 130, 20), "confidence": 0.9},
    ]
    graph = build_cell_graph(
        words,
        row_boxes=[(0, 0, 200, 10), (0, 10, 200, 30)],
        col_boxes=[(0, 0, 80, 30), (80, 0, 160, 30)],
        table_bbox=(0, 0, 200, 30),
    )
    assert [(node.row_index, node.col_index, node.text) for node in graph] == [
        (1, 0, "Cử nhân"),
        (1, 1, "4 năm"),
    ]


def test_schema_inference_recognizes_four_columns() -> None:
    cells = [
        CellGraphNode(row_index=row_index, col_index=col_index, text=value)
        for row_index, row in enumerate(_noisy_training_rows())
        for col_index, value in enumerate(row)
    ]
    schema = infer_table_schema(cells)
    assert schema.headers == ["Chương trình", "Người học", "Thời gian", "Khối lượng tối thiểu"]
    assert schema.column_roles == {0: "program", 1: "learner", 2: "duration", 3: "credits"}


def test_reconstruction_fill_down_split_and_exports() -> None:
    best = reconstruct_from_rows(_noisy_training_rows())
    records = export_table_records(best)

    assert best.score > 6.0
    assert len(records) == 7
    assert records[2]["Chương trình"] == "Kỹ sư"
    assert records[4]["Chương trình"] == "Thạc sĩ"
    assert records[5] == {
        "Chương trình": "Tiến sĩ",
        "Người học": "Tốt nghiệp thạc sĩ",
        "Thời gian": "3 năm",
        "Khối lượng tối thiểu": "106 tín chỉ",
    }
    assert records[6] == {
        "Chương trình": "Tiến sĩ",
        "Người học": "Tốt nghiệp đại học",
        "Thời gian": "4 năm",
        "Khối lượng tối thiểu": "151 tín chỉ",
    }

    markdown = export_markdown(best)
    assert "| Chương trình | Người học | Thời gian | Khối lượng tối thiểu |" in markdown
    assert "| Tiến sĩ | Tốt nghiệp đại học | 4 năm | 151 tín chỉ |" in markdown

    csv_text = export_csv(best)
    assert "Tiến sĩ,Tốt nghiệp thạc sĩ,3 năm,106 tín chỉ" in csv_text
    assert '"1,5 năm"' in csv_text

    payload = export_json(best)
    assert '"selected best score=' in payload


def test_hypothesis_scoring_prefers_split_candidate() -> None:
    cells = [
        CellGraphNode(row_index=row_index, col_index=col_index, text=value)
        for row_index, row in enumerate(_noisy_training_rows())
        for col_index, value in enumerate(row)
    ]
    schema = infer_table_schema(cells)
    hypotheses = generate_reconstruction_hypotheses(cells, schema)
    scored = [score_table_hypothesis(item) for item in hypotheses]
    best = select_best_hypothesis(scored)
    assert "split merged rows" in " ".join(best.trace)
    assert best.constraints["duration_pattern"] == 1.0
    assert best.constraints["credit_pattern"] == 1.0
    assert best.constraints["no_same_type_merge"] == 1.0


def test_table_structure_uses_constraint_reconstruction_only_when_flag_enabled(monkeypatch) -> None:
    monkeypatch.delenv("BOXBIIBOO_ENABLE_CONSTRAINT_TABLE_RECONSTRUCTION", raising=False)
    baseline = table_structure_from_rows(_noisy_training_rows(), backend="table_words_grid")
    assert baseline["table_row_count"] == 7
    assert baseline["table_records"][-1]["Thời gian"] == "3 4 năm năm"

    monkeypatch.setenv("BOXBIIBOO_ENABLE_CONSTRAINT_TABLE_RECONSTRUCTION", "true")
    improved = table_structure_from_rows(_noisy_training_rows(), backend="table_words_grid")
    assert improved["table_row_count"] == 8
    assert improved["table_records"][-1]["Thời gian"] == "4 năm"
    assert improved["table_records"][-1]["Khối lượng tối thiểu"] == "151 tín chỉ"
    assert improved["extraction_trace"]["constraint_reconstruction"]["status"] == "applied"


def test_reconstruction_repairs_real_qcdt_table_noise() -> None:
    rows = [
        ["Chương trình", "Người học", "Thời gian Khối tối thiểu lượng", ""],
        ["Cử nhân", "Tốt nghiệp THPT", "4 năm", "132 tín chỉ"],
        ["Kỹ sư", "Tốt chương nghiệp trình cử tích nhân hợp theo 1,5 năm", "", "48 tín chỉ"],
        ["", "Tốt nghiệp cử nhân", "2 năm", "60 tín chỉ"],
        ["", "Tốt nghiệp cử nhân", "2 năm", "60 tín chỉ"],
        ["Thạc sĩ", "Tốt chương nghiệp trình cử tích nhân hợp theo 1,5 năm", "", "48 tín chỉ"],
        ["Tiến sĩ", "Tốt Tốt nghiệp nghiệp thạc đại học sĩ", "3 4 năm năm", "106 151 tín tín chỉ chỉ"],
    ]

    best = reconstruct_from_rows(rows)
    records = export_table_records(best)

    assert records == [
        {
            "Chương trình": "Cử nhân",
            "Người học": "Tốt nghiệp THPT",
            "Thời gian": "4 năm",
            "Khối lượng tối thiểu": "132 tín chỉ",
        },
        {
            "Chương trình": "Kỹ sư",
            "Người học": "Tốt nghiệp cử nhân theo chương trình tích hợp",
            "Thời gian": "1,5 năm",
            "Khối lượng tối thiểu": "48 tín chỉ",
        },
        {
            "Chương trình": "Kỹ sư",
            "Người học": "Tốt nghiệp cử nhân",
            "Thời gian": "2 năm",
            "Khối lượng tối thiểu": "60 tín chỉ",
        },
        {
            "Chương trình": "Thạc sĩ",
            "Người học": "Tốt nghiệp cử nhân theo chương trình tích hợp",
            "Thời gian": "1,5 năm",
            "Khối lượng tối thiểu": "48 tín chỉ",
        },
        {
            "Chương trình": "Tiến sĩ",
            "Người học": "Tốt nghiệp thạc sĩ",
            "Thời gian": "3 năm",
            "Khối lượng tối thiểu": "106 tín chỉ",
        },
        {
            "Chương trình": "Tiến sĩ",
            "Người học": "Tốt nghiệp đại học",
            "Thời gian": "4 năm",
            "Khối lượng tối thiểu": "151 tín chỉ",
        },
    ]
    assert "deduplicated 1 duplicate rows" in best.trace
    assert "normalized learner word order" in best.trace
