from __future__ import annotations

import os
import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from app.ingest.extract.table import BBox, Table, TableCell, TableRow, cells_to_html, rows_to_csv

DEFAULT_TATR_DETECTION_MODEL = "microsoft/table-transformer-detection"
DEFAULT_TATR_STRUCTURE_MODEL = "microsoft/table-transformer-structure-recognition-v1.1-all"


@dataclass(slots=True)
class TatrObject:
    label: str
    bbox: BBox
    score: float


def predict_tables_from_image(
    image_path: str | Path,
    *,
    text_boxes: list[dict[str, Any]] | None = None,
    device: str | None = None,
    detection_model_name: str | None = None,
    structure_model_name: str | None = None,
    detection_threshold: float | None = None,
    structure_threshold: float | None = None,
    backend_name: str = "tatr",
    text_source: str | None = None,
) -> dict[str, Any]:
    """Run pretrained TATR detection + structure models on an image.

    Text is not recognized by TATR. If `text_boxes` are supplied from a PDF
    text layer or another OCR process, they are assigned into the predicted
    grid by geometry. Otherwise the cells are kept with empty text.
    """

    Image = _pil_image_class()
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")

    width, height = rgb_image.size
    detection_objects = detect_tables(
        rgb_image,
        device=device,
        model_name=detection_model_name,
        threshold=detection_threshold,
    )
    if not detection_objects:
        detection_objects = [
            TatrObject(
                label="table",
                bbox=(0.0, 0.0, float(width), float(height)),
                score=0.0,
            )
        ]

    table_regions: list[dict[str, Any]] = []
    all_cells: list[dict[str, Any]] = []
    csv_parts: list[str] = []
    html_parts: list[str] = []
    ordered_rows: list[str] = []
    metadata_tables: list[dict[str, Any]] = []
    debug_rows: list[dict[str, Any]] = []
    debug_columns: list[dict[str, Any]] = []
    debug_spanning: list[dict[str, Any]] = []
    assigned_words_per_cell: list[dict[str, Any]] = []
    warnings: list[str] = []
    if not text_boxes:
        warnings.append("No text boxes supplied; TATR output is geometry-only.")

    for table_index, table_obj in enumerate(detection_objects):
        table_bbox = _clip_bbox(table_obj.bbox, (0.0, 0.0, float(width), float(height)))
        crop = rgb_image.crop(tuple(int(round(value)) for value in table_bbox))
        structure_objects = recognize_table_structure(
            crop,
            table_offset=(table_bbox[0], table_bbox[1]),
            device=device,
            model_name=structure_model_name,
            threshold=structure_threshold,
        )
        table = build_table_from_tatr_objects(
            structure_objects,
            table_bbox=table_bbox,
            text_boxes=text_boxes or [],
            table_id=f"tatr_table_{table_index}",
        )
        geometry_debug = structure_debug_payload(structure_objects, table_bbox=table_bbox)
        debug_rows.extend(geometry_debug["rows"])
        debug_columns.extend(geometry_debug["columns"])
        debug_spanning.extend(geometry_debug["spanning_cells"])
        table_regions.append(
            {
                "label": "table",
                "bbox": table_bbox,
                "score": table_obj.score,
                "model_label": table_obj.label,
            }
        )
        if table is None:
            metadata_tables.append(
                {
                    "bbox": table_bbox,
                    "score": table_obj.score,
                    "row_count": 0,
                    "col_count": 0,
                    "cell_count": 0,
                }
            )
            continue

        cells = [cell.to_meta() for cell in table.cells]
        assigned_words_per_cell.extend(
            {
                "table_id": cell.table_id,
                "row": cell.row_index,
                "col": cell.col_index,
                "row_span": cell.row_span,
                "col_span": cell.col_span,
                "bbox": cell.bbox,
                "text": cell.text,
                "source_words": cell.source_words,
            }
            for cell in table.cells
        )
        rows = table.row_values
        csv_text = rows_to_csv(rows)
        html_text = cells_to_html(table.cells)
        all_cells.extend(cells)
        if csv_text:
            csv_parts.append(csv_text)
        if html_text:
            html_parts.append(html_text)
        ordered_rows.extend(" ".join(value for value in row if value).strip() for row in rows)
        metadata_tables.append(
            {
                "bbox": table_bbox,
                "score": table_obj.score,
                "row_count": len(rows),
                "col_count": table.col_count,
                "cell_count": len(cells),
                "structure_object_count": len(structure_objects),
            }
        )

    return {
        "table_backend": backend_name,
        "source_model_name": {
            "detection": detection_model_name or os.getenv("BOXBIIBOO_TATR_DETECTION_MODEL") or DEFAULT_TATR_DETECTION_MODEL,
            "structure": structure_model_name or os.getenv("BOXBIIBOO_TATR_STRUCTURE_MODEL") or DEFAULT_TATR_STRUCTURE_MODEL,
        },
        "table_regions": table_regions,
        "table_cells": all_cells,
        "table_csv": "\n\n".join(csv_parts) if csv_parts else None,
        "table_html": "\n".join(html_parts) if html_parts else None,
        "ordered_text": [row for row in ordered_rows if row],
        "text": "\n".join(row for row in ordered_rows if row),
        "coordinate_space": "page_image",
        "text_source": text_source or ("none" if not text_boxes else "provided_words"),
        "warnings": warnings,
        "tatr_rows": debug_rows,
        "tatr_columns": debug_columns,
        "tatr_spanning_cells": debug_spanning,
        "assigned_words_per_cell": assigned_words_per_cell,
        "spanning_cell_count": sum(
            1
            for cell in all_cells
            if int(cell.get("row_span", 1) or 1) > 1 or int(cell.get("col_span", 1) or 1) > 1
        ),
        "tables": metadata_tables,
    }


def detect_tables(
    image: Any,
    *,
    device: str | None = None,
    model_name: str | None = None,
    threshold: float | None = None,
) -> list[TatrObject]:
    model_name = model_name or os.getenv("BOXBIIBOO_TATR_DETECTION_MODEL") or DEFAULT_TATR_DETECTION_MODEL
    threshold = threshold if threshold is not None else float(os.getenv("BOXBIIBOO_TATR_DETECTION_THRESHOLD", "0.50"))
    objects = _run_tatr_object_detection(image, model_name=model_name, device=device, threshold=threshold)
    tables = [obj for obj in objects if _is_table_detection_label(obj.label)]
    return _nms(tables, iou_threshold=0.80)


def recognize_table_structure(
    table_crop: Any,
    *,
    table_offset: tuple[float, float] = (0.0, 0.0),
    device: str | None = None,
    model_name: str | None = None,
    threshold: float | None = None,
) -> list[TatrObject]:
    model_name = model_name or os.getenv("BOXBIIBOO_TATR_STRUCTURE_MODEL") or DEFAULT_TATR_STRUCTURE_MODEL
    threshold = threshold if threshold is not None else float(os.getenv("BOXBIIBOO_TATR_STRUCTURE_THRESHOLD", "0.50"))
    objects = _run_tatr_object_detection(table_crop, model_name=model_name, device=device, threshold=threshold)
    dx, dy = table_offset
    shifted = [
        TatrObject(
            label=obj.label,
            bbox=(obj.bbox[0] + dx, obj.bbox[1] + dy, obj.bbox[2] + dx, obj.bbox[3] + dy),
            score=obj.score,
        )
        for obj in objects
    ]
    return shifted


def build_table_from_tatr_objects(
    objects: list[TatrObject | dict[str, Any]],
    *,
    table_bbox: BBox,
    text_boxes: list[dict[str, Any]] | None = None,
    table_id: str = "tatr_table",
    page: int | None = None,
) -> Table | None:
    normalized = [_coerce_object(obj) for obj in objects]
    rows = _nms([obj for obj in normalized if _is_row_label(obj.label)], iou_threshold=0.70)
    columns = _nms([obj for obj in normalized if _is_column_label(obj.label)], iou_threshold=0.70)
    header_regions = [obj for obj in normalized if _is_header_label(obj.label)]
    spanning_regions = _nms([obj for obj in normalized if _is_spanning_label(obj.label)], iou_threshold=0.70)

    rows = sorted(rows, key=lambda obj: (obj.bbox[1], obj.bbox[0]))
    columns = sorted(columns, key=lambda obj: (obj.bbox[0], obj.bbox[1]))
    if not rows or not columns:
        return None

    row_bands = [_clip_row_band(obj.bbox, table_bbox) for obj in rows]
    col_bands = [_clip_col_band(obj.bbox, table_bbox) for obj in columns]
    text_boxes = [box for box in (text_boxes or []) if _bbox(box.get("bbox")) is not None]
    span_map = _spanning_cell_map(spanning_regions, row_bands, col_bands, table_bbox)

    table_rows: list[TableRow] = []
    for row_index, row_bbox in enumerate(row_bands):
        cells: list[TableCell] = []
        for col_index, col_bbox in enumerate(col_bands):
            span = _covering_span(span_map, row_index, col_index)
            if span is not None and (row_index, col_index) != (span["row_start"], span["col_start"]):
                continue

            row_span = int(span["row_span"]) if span is not None else 1
            col_span = int(span["col_span"]) if span is not None else 1
            cell_bbox = span["bbox"] if span is not None else _intersect_bbox(row_bbox, col_bbox)
            if cell_bbox is None:
                continue
            assigned_boxes = _assign_text_boxes(cell_bbox, text_boxes)
            output_bbox = _content_bbox_from_assigned_boxes(cell_bbox, assigned_boxes)
            text = _merge_text(assigned_boxes)
            confidence_values = [rows[row_index].score, columns[col_index].score]
            if span is not None:
                confidence_values.append(float(span["score"]))
            cells.append(
                TableCell(
                    row_index=row_index,
                    col_index=col_index,
                    row_span=row_span,
                    col_span=col_span,
                    bbox=output_bbox,
                    text=text,
                    confidence=mean(confidence_values),
                    source_boxes=[_bbox(box["bbox"]) for box in assigned_boxes if _bbox(box.get("bbox")) is not None],
                    source_words=[_source_word_payload(box) for box in assigned_boxes],
                    grid_bbox=cell_bbox,
                    page=page,
                    table_id=table_id,
                )
            )
        if cells:
            table_rows.append(TableRow(row_index=row_index, bbox=_union_bbox(cell.bbox for cell in cells), cells=cells))

    if not table_rows:
        return None
    return Table(table_id=table_id, page=page, bbox=table_bbox, rows=table_rows)


def structure_debug_payload(objects: list[TatrObject | dict[str, Any]], *, table_bbox: BBox) -> dict[str, list[dict[str, Any]]]:
    normalized = [_coerce_object(obj) for obj in objects]
    rows = sorted(_nms([obj for obj in normalized if _is_row_label(obj.label)], iou_threshold=0.70), key=lambda item: item.bbox[1])
    columns = sorted(_nms([obj for obj in normalized if _is_column_label(obj.label)], iou_threshold=0.70), key=lambda item: item.bbox[0])
    spans = sorted(_nms([obj for obj in normalized if _is_spanning_label(obj.label)], iou_threshold=0.70), key=lambda item: (item.bbox[1], item.bbox[0]))
    return {
        "rows": [_object_payload(obj, bbox=_clip_row_band(obj.bbox, table_bbox)) for obj in rows],
        "columns": [_object_payload(obj, bbox=_clip_col_band(obj.bbox, table_bbox)) for obj in columns],
        "spanning_cells": [_object_payload(obj, bbox=_clip_bbox(obj.bbox, table_bbox)) for obj in spans],
    }


def _run_tatr_object_detection(
    image: Any,
    *,
    model_name: str,
    device: str | None,
    threshold: float,
) -> list[TatrObject]:
    processor, model, torch = _get_tatr_bundle(model_name, device or os.getenv("BOXBIIBOO_TATR_DEVICE", "cuda"))
    inputs = processor(images=image, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]], device=model.device)
    results = processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]
    id2label = dict(getattr(model.config, "id2label", {}) or {})

    objects: list[TatrObject] = []
    labels = results["labels"].detach().cpu().tolist()
    scores = results["scores"].detach().cpu().tolist()
    boxes = results["boxes"].detach().cpu().tolist()
    for label_id, score, box in zip(labels, scores, boxes):
        x0, y0, x1, y1 = [float(value) for value in box[:4]]
        if x1 <= x0 or y1 <= y0:
            continue
        objects.append(
            TatrObject(
                label=_label_name(id2label, int(label_id)),
                bbox=(x0, y0, x1, y1),
                score=float(score),
            )
        )
    return objects


@lru_cache(maxsize=4)
def _get_tatr_bundle(model_name: str, device_name: str):
    try:
        import torch
        from transformers import AutoImageProcessor, TableTransformerConfig

        try:
            from transformers import TableTransformerForObjectDetection
        except Exception:
            from transformers import AutoModelForObjectDetection as TableTransformerForObjectDetection
    except Exception as exc:
        raise RuntimeError("TATR backend requires torch, pillow, and transformers") from exc

    processor = AutoImageProcessor.from_pretrained(model_name)
    _normalize_processor_size(processor)
    try:
        model = TableTransformerForObjectDetection.from_pretrained(model_name)
    except Exception as exc:
        if "dilation" not in str(exc):
            raise
        config = _load_table_transformer_config(model_name, TableTransformerConfig)
        model = TableTransformerForObjectDetection.from_pretrained(model_name, config=config)
    model.eval()
    requested = device_name or "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        requested = "cpu"
    model.to(torch.device(requested))
    return processor, model, torch


def _load_table_transformer_config(model_name: str, config_cls: Any) -> Any:
    from huggingface_hub import hf_hub_download

    config_path = hf_hub_download(model_name, "config.json")
    with open(config_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("dilation") is None:
        payload["dilation"] = False
    return config_cls(**payload)


def _normalize_processor_size(processor: Any) -> None:
    size = getattr(processor, "size", None)
    if not size:
        return
    shortest = getattr(size, "shortest_edge", None)
    longest = getattr(size, "longest_edge", None)
    if isinstance(size, dict):
        shortest = size.get("shortest_edge")
        longest = size.get("longest_edge")
    if shortest is not None:
        return
    if longest is None:
        return
    processor.size = {"shortest_edge": int(longest), "longest_edge": int(longest)}


def _pil_image_class():
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("TATR backend requires pillow") from exc
    return Image


def _coerce_object(obj: TatrObject | dict[str, Any]) -> TatrObject:
    if isinstance(obj, TatrObject):
        return obj
    bbox = _bbox(obj.get("bbox"))
    if bbox is None:
        bbox = (0.0, 0.0, 0.0, 0.0)
    return TatrObject(label=str(obj.get("label") or obj.get("label_name") or ""), bbox=bbox, score=float(obj.get("score", 1.0) or 0.0))


def _is_table_detection_label(label: str) -> bool:
    name = label.lower().strip()
    return "table" in name and all(token not in name for token in ("row", "column", "cell", "header"))


def _is_row_label(label: str) -> bool:
    name = label.lower().strip()
    return name == "table row"


def _is_column_label(label: str) -> bool:
    name = label.lower().strip()
    return name == "table column"


def _is_header_label(label: str) -> bool:
    name = label.lower().strip()
    return "header" in name


def _is_spanning_label(label: str) -> bool:
    name = label.lower().strip()
    return "spanning" in name and "cell" in name


def _label_name(id2label: dict[int, str] | dict[str, str], label_id: int) -> str:
    return str(id2label.get(label_id) or id2label.get(str(label_id)) or label_id).strip().lower()


def _clip_row_band(row_bbox: BBox, table_bbox: BBox) -> BBox:
    return (table_bbox[0], max(table_bbox[1], row_bbox[1]), table_bbox[2], min(table_bbox[3], row_bbox[3]))


def _clip_col_band(col_bbox: BBox, table_bbox: BBox) -> BBox:
    return (max(table_bbox[0], col_bbox[0]), table_bbox[1], min(table_bbox[2], col_bbox[2]), table_bbox[3])


def _spanning_cell_map(
    spanning_regions: list[TatrObject],
    row_bands: list[BBox],
    col_bands: list[BBox],
    table_bbox: BBox,
) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for region in spanning_regions:
        bbox = _clip_bbox(region.bbox, table_bbox)
        row_indexes = [idx for idx, row in enumerate(row_bands) if _vertical_overlap_ratio(bbox, row) >= 0.25]
        col_indexes = [idx for idx, col in enumerate(col_bands) if _horizontal_overlap_ratio(bbox, col) >= 0.25]
        if not row_indexes or not col_indexes:
            continue
        row_start, row_end = min(row_indexes), max(row_indexes)
        col_start, col_end = min(col_indexes), max(col_indexes)
        if row_end == row_start and col_end == col_start:
            continue
        spans.append(
            {
                "row_start": row_start,
                "col_start": col_start,
                "row_span": row_end - row_start + 1,
                "col_span": col_end - col_start + 1,
                "bbox": bbox,
                "score": region.score,
            }
        )
    return spans


def _covering_span(spans: list[dict[str, Any]], row_index: int, col_index: int) -> dict[str, Any] | None:
    for span in spans:
        if (
            span["row_start"] <= row_index < span["row_start"] + span["row_span"]
            and span["col_start"] <= col_index < span["col_start"] + span["col_span"]
        ):
            return span
    return None


def _assign_text_boxes(cell_bbox: BBox, text_boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assigned = []
    for box in text_boxes:
        bbox = _bbox(box.get("bbox"))
        if bbox is None:
            continue
        center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        if cell_bbox[0] <= center[0] <= cell_bbox[2] and cell_bbox[1] <= center[1] <= cell_bbox[3]:
            assigned.append(box)
            continue
        min_overlap = float(os.getenv("BOXBIIBOO_TATR_WORD_MIN_OVERLAP", "0.15"))
        if _overlap_area(cell_bbox, bbox) / max(_area(bbox), 1.0) >= min_overlap:
            assigned.append(box)
    return sorted(assigned, key=lambda item: (_bbox(item.get("bbox"))[1], _bbox(item.get("bbox"))[0]))


def _content_bbox_from_assigned_boxes(grid_bbox: BBox, assigned_boxes: list[dict[str, Any]]) -> BBox:
    if os.getenv("BOXBIIBOO_TATR_USE_CONTENT_CELL_BBOX", "1").strip().lower() in {"0", "false", "no", "off"}:
        return grid_bbox
    boxes = [_bbox(box.get("bbox")) for box in assigned_boxes]
    valid_boxes = [box for box in boxes if box is not None]
    if not valid_boxes:
        return grid_bbox
    content_bbox = _union_bbox(valid_boxes)
    clipped = _intersect_bbox(content_bbox, grid_bbox)
    return clipped if clipped is not None else content_bbox


def _merge_text(text_boxes: list[dict[str, Any]]) -> str:
    return _normalize_text(" ".join(str(box.get("text") or "").strip() for box in text_boxes if str(box.get("text") or "").strip()))


def _normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", value).strip()


def _source_word_payload(word: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "text": _normalize_text(str(word.get("text") or "")),
        "bbox": list(_bbox(word.get("bbox")) or (0.0, 0.0, 0.0, 0.0)),
    }
    if word.get("confidence") is not None:
        payload["confidence"] = float(word.get("confidence") or 0.0)
    if word.get("source") is not None:
        payload["source"] = str(word.get("source"))
    return payload


def _object_payload(obj: TatrObject, *, bbox: BBox) -> dict[str, Any]:
    return {"label": obj.label, "bbox": bbox, "score": obj.score}


def _nms(objects: list[TatrObject], *, iou_threshold: float) -> list[TatrObject]:
    kept: list[TatrObject] = []
    for obj in sorted(objects, key=lambda item: item.score, reverse=True):
        if any(_iou(obj.bbox, other.bbox) >= iou_threshold for other in kept):
            continue
        kept.append(obj)
    return kept


def _clip_bbox(bbox: BBox, bounds: BBox) -> BBox:
    return (
        max(bounds[0], bbox[0]),
        max(bounds[1], bbox[1]),
        min(bounds[2], bbox[2]),
        min(bounds[3], bbox[3]),
    )


def page_to_crop_bbox(bbox: BBox | list[float] | tuple[float, ...], table_bbox: BBox | list[float] | tuple[float, ...]) -> BBox:
    box = normalize_bbox(bbox)
    table = normalize_bbox(table_bbox)
    return (box[0] - table[0], box[1] - table[1], box[2] - table[0], box[3] - table[1])


def crop_to_page_bbox(bbox: BBox | list[float] | tuple[float, ...], table_bbox: BBox | list[float] | tuple[float, ...]) -> BBox:
    box = normalize_bbox(bbox)
    table = normalize_bbox(table_bbox)
    return (box[0] + table[0], box[1] + table[1], box[2] + table[0], box[3] + table[1])


def normalize_bbox(bbox: BBox | list[float] | tuple[float, ...]) -> BBox:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        raise ValueError(f"Invalid bbox: {bbox!r}")
    x0, y0, x1, y1 = [float(value) for value in bbox[:4]]
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def bbox_center(bbox: BBox | list[float] | tuple[float, ...]) -> tuple[float, float]:
    box = normalize_bbox(bbox)
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def bbox_intersection_area(left: BBox | list[float] | tuple[float, ...], right: BBox | list[float] | tuple[float, ...]) -> float:
    return _overlap_area(normalize_bbox(left), normalize_bbox(right))


def bbox_iou(left: BBox | list[float] | tuple[float, ...], right: BBox | list[float] | tuple[float, ...]) -> float:
    return _iou(normalize_bbox(left), normalize_bbox(right))


def bbox_overlap_ratio(left: BBox | list[float] | tuple[float, ...], right: BBox | list[float] | tuple[float, ...]) -> float:
    left_box = normalize_bbox(left)
    right_box = normalize_bbox(right)
    return bbox_intersection_area(left_box, right_box) / max(min(_area(left_box), _area(right_box)), 1.0)


def _intersect_bbox(left: BBox, right: BBox) -> BBox | None:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _bbox(value: Any) -> BBox | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    x0, y0, x1, y1 = [float(item) for item in value[:4]]
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _union_bbox(boxes: Iterable[BBox]) -> BBox:
    values = list(boxes)
    return (
        min(box[0] for box in values),
        min(box[1] for box in values),
        max(box[2] for box in values),
        max(box[3] for box in values),
    )


def _area(bbox: BBox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _overlap_area(left: BBox, right: BBox) -> float:
    inter = _intersect_bbox(left, right)
    return _area(inter) if inter is not None else 0.0


def _iou(left: BBox, right: BBox) -> float:
    inter = _overlap_area(left, right)
    if inter <= 0:
        return 0.0
    return inter / max(_area(left) + _area(right) - inter, 1.0)


def _vertical_overlap_ratio(left: BBox, right: BBox) -> float:
    overlap = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    return overlap / max(min(left[3] - left[1], right[3] - right[1]), 1.0)


def _horizontal_overlap_ratio(left: BBox, right: BBox) -> float:
    overlap = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    return overlap / max(min(left[2] - left[0], right[2] - right[0]), 1.0)
