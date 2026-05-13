from __future__ import annotations

import difflib
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from statistics import mean
from typing import Any

from app.eval.ingest_schemas import LayoutRegion


TOKEN_RE = re.compile(r"\w+", re.UNICODE)
MAX_CHAR_EDIT_LENGTH = int(os.getenv("BOXBIIBOO_BENCHMARK_MAX_CHAR_EDIT_LENGTH", "1000"))
MAX_TOKEN_EDIT_LENGTH = int(os.getenv("BOXBIIBOO_BENCHMARK_MAX_TOKEN_EDIT_LENGTH", "1000"))


def normalize_text(text: str | None) -> str:
    return " ".join((text or "").casefold().split())


def normalize_historical_ocr_text(text: str | None) -> str:
    value = (text or "").casefold()
    replacements = {
        "ſ": "s",
        "å¿": "s",
        "ꝛ": "r",
        "ê›": "r",
        "ꝑ": "p",
        "ê‘": "p",
        "æ": "ae",
        "ã¦": "ae",
        "Ã¦": "ae",
        "ß": "ss",
        "ÃŸ": "ss",
        "î¢¿": "",
        "ê°": "",
        "âŠ": " et ",
        "â¸—": "-",
        "ȣ": "u",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    return " ".join(value.split())


def historical_ocr_token_f1(prediction: str, ground_truth: str | None) -> dict[str, float] | None:
    if ground_truth is None:
        return None
    pred_tokens = TOKEN_RE.findall(normalize_historical_ocr_text(prediction))
    gt_tokens = TOKEN_RE.findall(normalize_historical_ocr_text(ground_truth))
    if not pred_tokens and not gt_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not gt_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    pred_counts = Counter(pred_tokens)
    gt_counts = Counter(gt_tokens)
    overlap = sum((pred_counts & gt_counts).values())
    precision = overlap / len(pred_tokens) if pred_tokens else 0.0
    recall = overlap / len(gt_tokens) if gt_tokens else 0.0
    return {"precision": precision, "recall": recall, "f1": f1_from_pr(precision, recall)}


def historical_ocr_cer(prediction: str, ground_truth: str | None) -> float | None:
    if ground_truth is None:
        return None
    expected = normalize_historical_ocr_text(ground_truth)
    actual = normalize_historical_ocr_text(prediction)
    if not expected:
        return 0.0 if not actual else 1.0
    actual, expected = _bounded_pair(actual, expected, MAX_CHAR_EDIT_LENGTH)
    return levenshtein_distance(actual, expected) / max(len(expected), 1)


def historical_ocr_wer(prediction: str, ground_truth: str | None) -> float | None:
    if ground_truth is None:
        return None
    expected = TOKEN_RE.findall(normalize_historical_ocr_text(ground_truth))
    actual = TOKEN_RE.findall(normalize_historical_ocr_text(prediction))
    if not expected:
        return 0.0 if not actual else 1.0
    actual, expected = _bounded_pair(actual, expected, MAX_TOKEN_EDIT_LENGTH)
    return levenshtein_distance(actual, expected) / max(len(expected), 1)


def char_accuracy(prediction: str, ground_truth: str | None) -> float | None:
    if ground_truth is None:
        return None
    expected = normalize_text(ground_truth)
    actual = normalize_text(prediction)
    if not expected:
        return 1.0 if not actual else 0.0
    actual, expected = _bounded_pair(actual, expected, MAX_CHAR_EDIT_LENGTH)
    distance = levenshtein_distance(actual, expected)
    return max(0.0, 1.0 - distance / max(len(expected), 1))


def normalized_text_similarity(prediction: str, ground_truth: str | None) -> float | None:
    if ground_truth is None:
        return None
    actual, expected = _bounded_pair(normalize_text(prediction), normalize_text(ground_truth), MAX_CHAR_EDIT_LENGTH)
    return difflib.SequenceMatcher(None, actual, expected).ratio()


def token_f1(prediction: str, ground_truth: str | None) -> dict[str, float] | None:
    if ground_truth is None:
        return None
    pred_tokens = TOKEN_RE.findall(normalize_text(prediction))
    gt_tokens = TOKEN_RE.findall(normalize_text(ground_truth))
    if not pred_tokens and not gt_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not gt_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    pred_counts = Counter(pred_tokens)
    gt_counts = Counter(gt_tokens)
    overlap = sum((pred_counts & gt_counts).values())
    precision = overlap / len(pred_tokens) if pred_tokens else 0.0
    recall = overlap / len(gt_tokens) if gt_tokens else 0.0
    return {"precision": precision, "recall": recall, "f1": f1_from_pr(precision, recall)}


def cer(prediction: str, ground_truth: str | None) -> float | None:
    if ground_truth is None:
        return None
    expected = normalize_text(ground_truth)
    actual = normalize_text(prediction)
    if not expected:
        return 0.0 if not actual else 1.0
    actual, expected = _bounded_pair(actual, expected, MAX_CHAR_EDIT_LENGTH)
    return levenshtein_distance(actual, expected) / max(len(expected), 1)


def wer(prediction: str, ground_truth: str | None) -> float | None:
    if ground_truth is None:
        return None
    expected = TOKEN_RE.findall(normalize_text(ground_truth))
    actual = TOKEN_RE.findall(normalize_text(prediction))
    if not expected:
        return 0.0 if not actual else 1.0
    actual, expected = _bounded_pair(actual, expected, MAX_TOKEN_EDIT_LENGTH)
    return levenshtein_distance(actual, expected) / max(len(expected), 1)


def reading_order_score(predicted_order: list[str], expected_order: list[str]) -> float | None:
    if not expected_order:
        return None
    predicted_payload = "\n".join(predicted_order)
    folded_payload = normalize_text(predicted_payload)
    positions: list[int] = []
    for item in expected_order:
        positions.append(folded_payload.find(normalize_text(item)))
    present = [pos for pos in positions if pos >= 0]
    if not present:
        return 0.0
    ordered_pairs = 0
    total_pairs = 0
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            total_pairs += 1
            if present[i] <= present[j]:
                ordered_pairs += 1
    order = ordered_pairs / total_pairs if total_pairs else 1.0
    coverage = len(present) / len(expected_order)
    return mean([order, coverage])


def iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    left_area = max((left[2] - left[0]) * (left[3] - left[1]), 1.0)
    right_area = max((right[2] - right[0]) * (right[3] - right[1]), 1.0)
    return inter / (left_area + right_area - inter)


def detection_metrics(
    predicted: list[LayoutRegion],
    ground_truth: list[LayoutRegion],
    *,
    labels: list[str] | None = None,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    labels = labels or sorted({r.label for r in predicted} | {r.label for r in ground_truth})
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    candidates: list[tuple[float, int, int]] = []
    for pred_idx, pred in enumerate(predicted):
        for gt_idx, gt in enumerate(ground_truth):
            if pred.label != gt.label:
                continue
            score = iou(pred.bbox, gt.bbox)
            if score >= iou_threshold:
                candidates.append((score, pred_idx, gt_idx))

    for score, pred_idx, gt_idx in sorted(candidates, reverse=True):
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)
        matches.append((pred_idx, gt_idx, score))

    per_label: dict[str, dict[str, float]] = {}
    totals = Counter()
    for label in labels:
        gt_count = sum(1 for region in ground_truth if region.label == label)
        pred_count = sum(1 for region in predicted if region.label == label)
        tp = sum(1 for pred_idx, _, _ in matches if predicted[pred_idx].label == label)
        fp = pred_count - tp
        fn = gt_count - tp
        precision = tp / pred_count if pred_count else 0.0
        recall = tp / gt_count if gt_count else 0.0
        f1 = f1_from_pr(precision, recall)
        per_label[label] = {
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        totals.update({"tp": tp, "fp": fp, "fn": fn})

    micro_precision = totals["tp"] / (totals["tp"] + totals["fp"]) if (totals["tp"] + totals["fp"]) else 0.0
    micro_recall = totals["tp"] / (totals["tp"] + totals["fn"]) if (totals["tp"] + totals["fn"]) else 0.0
    macro_f1 = mean([per_label[label]["f1"] for label in labels]) if labels else 0.0
    return {
        "iou_threshold": iou_threshold,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": f1_from_pr(micro_precision, micro_recall),
        "macro_f1": macro_f1,
        "per_label": per_label,
        "matches": len(matches),
    }


def confusion_summary(
    predicted: list[LayoutRegion],
    ground_truth: list[LayoutRegion],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, dict[str, int]]:
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    used_pred: set[int] = set()
    for gt in ground_truth:
        best_idx = -1
        best_iou = 0.0
        for pred_idx, pred in enumerate(predicted):
            if pred_idx in used_pred:
                continue
            score = iou(pred.bbox, gt.bbox)
            if score > best_iou:
                best_iou = score
                best_idx = pred_idx
        if best_idx >= 0 and best_iou >= iou_threshold:
            used_pred.add(best_idx)
            confusion[gt.label][predicted[best_idx].label] += 1
        else:
            confusion[gt.label]["<missing>"] += 1
    return {label: dict(counts) for label, counts in confusion.items()}


def table_structure_score(predicted_cells: list[dict[str, Any]], gt_cells: list[dict[str, Any]]) -> dict[str, float] | None:
    if not gt_cells:
        return None
    pred_keys = {_cell_key(cell) for cell in predicted_cells}
    gt_keys = {_cell_key(cell) for cell in gt_cells}
    if not pred_keys and not gt_keys:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    overlap = len(pred_keys & gt_keys)
    precision = overlap / len(pred_keys) if pred_keys else 0.0
    recall = overlap / len(gt_keys) if gt_keys else 0.0
    return {"precision": precision, "recall": recall, "f1": f1_from_pr(precision, recall)}


def table_cell_bbox_metrics(
    predicted_cells: list[dict[str, Any]],
    gt_cells: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, float] | None:
    pred_regions = [
        LayoutRegion("cell", tuple(float(value) for value in cell["bbox"][:4]))
        for cell in predicted_cells
        if isinstance(cell.get("bbox"), (list, tuple)) and len(cell["bbox"]) >= 4
    ]
    gt_regions = [
        LayoutRegion("cell", tuple(float(value) for value in cell["bbox"][:4]))
        for cell in gt_cells
        if isinstance(cell.get("bbox"), (list, tuple)) and len(cell["bbox"]) >= 4
    ]
    if not gt_regions:
        return None
    metrics = detection_metrics(pred_regions, gt_regions, labels=["cell"], iou_threshold=iou_threshold)
    return {
        "precision": metrics["micro_precision"],
        "recall": metrics["micro_recall"],
        "f1": metrics["micro_f1"],
    }


def table_structure_breakdown(
    predicted_cells: list[dict[str, Any]],
    gt_cells: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any] | None:
    if not gt_cells:
        return None
    matches, unmatched_pred, unmatched_gt = table_cell_matches(predicted_cells, gt_cells, iou_threshold=iou_threshold)
    text_scores = [
        token_f1(str(predicted_cells[pred_idx].get("text", "")), str(gt_cells[gt_idx].get("text", "")))["f1"]
        for pred_idx, gt_idx, _ in matches
    ]
    has_bbox_gt = any(_cell_bbox(cell) is not None for cell in gt_cells)
    pred_row_count = _max_cell_index(predicted_cells, "row") + 1
    gt_row_count = _max_cell_index(gt_cells, "row") + 1
    pred_col_count = _max_cell_index(predicted_cells, "col") + 1
    gt_col_count = _max_cell_index(gt_cells, "col") + 1
    row_delta = pred_row_count - gt_row_count
    col_delta = pred_col_count - gt_col_count
    empty_pred = sum(1 for cell in predicted_cells if not normalize_text(str(cell.get("text", ""))))
    return {
        "text_assignment_f1": (sum(text_scores) / len(text_scores)) if text_scores else (0.0 if has_bbox_gt else None),
        "row_count_error": abs(row_delta),
        "col_count_error": abs(col_delta),
        "row_count_mae": abs(row_delta),
        "col_count_mae": abs(col_delta),
        "row_oversegmentation_count": 1 if row_delta > 0 else 0,
        "row_undersegmentation_count": 1 if row_delta < 0 else 0,
        "col_oversegmentation_count": 1 if col_delta > 0 else 0,
        "col_undersegmentation_count": 1 if col_delta < 0 else 0,
        "empty_cell_rate": empty_pred / len(predicted_cells) if predicted_cells else 0.0,
        "matched_cell_count": len(matches),
        "unmatched_pred_count": len(unmatched_pred),
        "unmatched_gt_count": len(unmatched_gt),
    }


def grits_like_metrics(predicted_cells: list[dict[str, Any]], gt_cells: list[dict[str, Any]]) -> dict[str, float] | None:
    """Lightweight GriTS-style approximation.

    This is not the official Microsoft GriTS implementation. It provides a
    stable local signal for topology, location and content when exact CSV/HTML
    is too strict for benchmark iteration.
    """

    if not gt_cells:
        return None
    pred_by_topology = {_cell_topology_key(cell): cell for cell in predicted_cells}
    gt_by_topology = {_cell_topology_key(cell): cell for cell in gt_cells}
    pred_keys = set(pred_by_topology)
    gt_keys = set(gt_by_topology)
    if not pred_keys and not gt_keys:
        return {"grits_top_like": 1.0, "grits_loc_like": 1.0, "grits_con_like": 1.0}
    overlap = pred_keys & gt_keys
    precision = len(overlap) / len(pred_keys) if pred_keys else 0.0
    recall = len(overlap) / len(gt_keys) if gt_keys else 0.0
    top = f1_from_pr(precision, recall)
    if not overlap:
        return {"grits_top_like": top, "grits_loc_like": 0.0, "grits_con_like": 0.0}

    loc_scores: list[float] = []
    con_scores: list[float] = []
    for key in overlap:
        pred = pred_by_topology[key]
        gt = gt_by_topology[key]
        pred_bbox = _cell_bbox(pred)
        gt_bbox = _cell_bbox(gt)
        loc = iou(pred_bbox, gt_bbox) if pred_bbox is not None and gt_bbox is not None else 0.0
        text_score = token_f1(str(pred.get("text", "")), str(gt.get("text", "")))["f1"]
        loc_scores.append(loc)
        con_scores.append(loc * text_score)
    return {
        "grits_top_like": top,
        "grits_loc_like": top * (sum(loc_scores) / len(loc_scores)),
        "grits_con_like": top * (sum(con_scores) / len(con_scores)),
    }


def table_cell_debug_payload(
    predicted_cells: list[dict[str, Any]],
    gt_cells: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any] | None:
    if not gt_cells:
        return None
    matches, unmatched_pred, unmatched_gt = table_cell_matches(predicted_cells, gt_cells, iou_threshold=iou_threshold)
    return {
        "predicted": {
            "row_count": _max_cell_index(predicted_cells, "row") + 1,
            "col_count": _max_cell_index(predicted_cells, "col") + 1,
            "cell_count": len(predicted_cells),
            "cells": predicted_cells,
        },
        "ground_truth": {
            "row_count": _max_cell_index(gt_cells, "row") + 1,
            "col_count": _max_cell_index(gt_cells, "col") + 1,
            "cell_count": len(gt_cells),
            "cells": gt_cells,
        },
        "matched_cells": [
            {
                "pred_index": pred_idx,
                "gt_index": gt_idx,
                "iou": score,
                "pred": predicted_cells[pred_idx],
                "gt": gt_cells[gt_idx],
                "text_f1": token_f1(
                    str(predicted_cells[pred_idx].get("text", "")),
                    str(gt_cells[gt_idx].get("text", "")),
                )["f1"],
            }
            for pred_idx, gt_idx, score in matches
        ],
        "unmatched_predicted": [predicted_cells[idx] for idx in unmatched_pred],
        "unmatched_ground_truth": [gt_cells[idx] for idx in unmatched_gt],
    }


def table_cell_matches(
    predicted_cells: list[dict[str, Any]],
    gt_cells: list[dict[str, Any]],
    *,
    iou_threshold: float,
) -> tuple[list[tuple[int, int, float]], list[int], list[int]]:
    candidates: list[tuple[float, int, int]] = []
    for pred_idx, pred in enumerate(predicted_cells):
        pred_bbox = _cell_bbox(pred)
        if pred_bbox is None:
            continue
        for gt_idx, gt in enumerate(gt_cells):
            gt_bbox = _cell_bbox(gt)
            if gt_bbox is None:
                continue
            score = iou(pred_bbox, gt_bbox)
            if score >= iou_threshold:
                candidates.append((score, pred_idx, gt_idx))

    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for score, pred_idx, gt_idx in sorted(candidates, reverse=True):
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        matched_pred.add(pred_idx)
        matched_gt.add(gt_idx)
        matches.append((pred_idx, gt_idx, score))

    unmatched_pred = [idx for idx in range(len(predicted_cells)) if idx not in matched_pred]
    unmatched_gt = [idx for idx in range(len(gt_cells)) if idx not in matched_gt]
    return matches, unmatched_pred, unmatched_gt


def table_exact_match(prediction: str | None, ground_truth: str | None) -> float | None:
    if ground_truth is None:
        return None
    return 1.0 if normalize_text(prediction or "") == normalize_text(ground_truth) else 0.0


def summarize_numeric(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    ordered = sorted(values)
    return {
        "mean": sum(ordered) / len(ordered),
        "p50": percentile(ordered, 0.50),
        "p95": percentile(ordered, 0.95),
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * q
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return values[low]
    weight = rank - low
    return values[low] * (1 - weight) + values[high] * weight


def f1_from_pr(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _bounded_pair(left: Any, right: Any, max_length: int) -> tuple[Any, Any]:
    if max_length <= 0:
        return left, right
    if len(left) <= max_length and len(right) <= max_length:
        return left, right
    return _head_tail(left, max_length), _head_tail(right, max_length)


def _head_tail(value: Any, max_length: int) -> Any:
    if len(value) <= max_length:
        return value
    head = max_length // 2
    tail = max_length - head
    return value[:head] + value[-tail:]


def levenshtein_distance(left: Any, right: Any) -> int:
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_value in enumerate(left, start=1):
        current = [i]
        for j, right_value in enumerate(right, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (0 if left_value == right_value else 1)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def _cell_key(cell: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(cell.get("row", 0) or 0),
        int(cell.get("col", 0) or 0),
        normalize_text(str(cell.get("text", ""))),
    )


def _cell_topology_key(cell: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(cell.get("row", 0) or 0),
        int(cell.get("col", 0) or 0),
        int(cell.get("row_span", 1) or 1),
        int(cell.get("col_span", 1) or 1),
    )


def _cell_bbox(cell: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = cell.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    return tuple(float(value) for value in bbox[:4])


def _max_cell_index(cells: list[dict[str, Any]], field: str) -> int:
    if not cells:
        return -1
    return max(int(cell.get(field, 0) or 0) for cell in cells)
