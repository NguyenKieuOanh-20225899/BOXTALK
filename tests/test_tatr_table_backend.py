from __future__ import annotations

from app.ingest.tatr_table_backend import TatrObject, build_table_from_tatr_objects
from app.ingest.tatr_table_backend import (
    bbox_center,
    bbox_intersection_area,
    bbox_iou,
    bbox_overlap_ratio,
    crop_to_page_bbox,
    page_to_crop_bbox,
)
from app.ingest.extract.table import cells_to_html, rows_to_csv
from app.eval.ingest_metrics import grits_like_metrics


def test_tatr_rows_columns_build_grid_and_assign_text() -> None:
    objects = [
        TatrObject("table row", (0, 0, 100, 20), 0.99),
        TatrObject("table row", (0, 20, 100, 40), 0.98),
        TatrObject("table column", (0, 0, 50, 40), 0.97),
        TatrObject("table column", (50, 0, 100, 40), 0.96),
    ]
    text_boxes = [
        {"text": "Metric", "bbox": (5, 5, 35, 15)},
        {"text": "Value", "bbox": (55, 5, 90, 15)},
        {"text": "Latency", "bbox": (5, 25, 40, 35)},
        {"text": "Low", "bbox": (55, 25, 75, 35)},
    ]

    table = build_table_from_tatr_objects(objects, table_bbox=(0, 0, 100, 40), text_boxes=text_boxes)

    assert table is not None
    assert table.row_values == [["Metric", "Value"], ["Latency", "Low"]]
    assert table.cells[0].bbox == (5.0, 5.0, 35.0, 15.0)
    assert table.cells[0].grid_bbox == (0, 0, 50, 20)
    assert rows_to_csv(table.row_values) == "Metric,Value\nLatency,Low"
    assert "<th>Metric</th>" in cells_to_html(table.cells)


def test_tatr_spanning_cell_suppresses_covered_grid_cells() -> None:
    objects = [
        TatrObject("table row", (0, 0, 120, 20), 0.99),
        TatrObject("table row", (0, 20, 120, 40), 0.98),
        TatrObject("table column", (0, 0, 40, 40), 0.97),
        TatrObject("table column", (40, 0, 80, 40), 0.96),
        TatrObject("table column", (80, 0, 120, 40), 0.95),
        TatrObject("table spanning cell", (40, 0, 120, 20), 0.93),
    ]

    table = build_table_from_tatr_objects(objects, table_bbox=(0, 0, 120, 40), text_boxes=[])

    assert table is not None
    span_cells = [cell for cell in table.cells if cell.col_span > 1 or cell.row_span > 1]
    assert len(span_cells) == 1
    assert span_cells[0].row_index == 0
    assert span_cells[0].col_index == 1
    assert span_cells[0].col_span == 2
    assert table.col_count == 3


def test_tatr_coordinate_helpers() -> None:
    table_bbox = (100, 50, 300, 250)
    page_bbox = (120, 70, 180, 110)

    assert page_to_crop_bbox(page_bbox, table_bbox) == (20, 20, 80, 60)
    assert crop_to_page_bbox((20, 20, 80, 60), table_bbox) == page_bbox
    assert bbox_center(page_bbox) == (150, 90)
    assert bbox_intersection_area((0, 0, 10, 10), (5, 5, 15, 15)) == 25
    assert 0 < bbox_iou((0, 0, 10, 10), (5, 5, 15, 15)) < 1
    assert bbox_overlap_ratio((0, 0, 10, 10), (5, 5, 15, 15)) == 0.25


def test_tatr_text_assignment_uses_overlap_when_center_is_outside() -> None:
    objects = [
        TatrObject("table row", (0, 0, 100, 20), 0.99),
        TatrObject("table column", (0, 0, 100, 20), 0.98),
    ]
    text_boxes = [{"text": "Wide", "bbox": (-90, 2, 70, 18), "source": "mock"}]

    table = build_table_from_tatr_objects(objects, table_bbox=(0, 0, 100, 20), text_boxes=text_boxes)

    assert table is not None
    assert table.cells[0].text == "Wide"
    assert table.cells[0].source_words[0]["source"] == "mock"


def test_grits_like_metrics_perfect_and_partial() -> None:
    gt_cells = [
        {"row": 0, "col": 0, "row_span": 1, "col_span": 1, "text": "A", "bbox": (0, 0, 10, 10)},
        {"row": 0, "col": 1, "row_span": 1, "col_span": 1, "text": "B", "bbox": (10, 0, 20, 10)},
    ]
    perfect = grits_like_metrics(list(gt_cells), gt_cells)
    partial = grits_like_metrics([gt_cells[0]], gt_cells)

    assert perfect == {"grits_top_like": 1.0, "grits_loc_like": 1.0, "grits_con_like": 1.0}
    assert partial is not None
    assert 0.0 < partial["grits_top_like"] < 1.0
    assert 0.0 < partial["grits_loc_like"] < 1.0
