from __future__ import annotations

import csv
import io
import json
import os
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

BBox = tuple[float, float, float, float]


@dataclass(slots=True)
class CellGraphNode:
    row_index: int
    col_index: int
    text: str
    bbox: BBox | None = None
    confidence: float | None = None
    words: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TableSchema:
    headers: list[str]
    column_roles: dict[int, str]
    confidence: float
    trace: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TableHypothesis:
    headers: list[str]
    rows: list[list[str]]
    cells: list[dict[str, Any]]
    score: float
    constraints: dict[str, float]
    trace: list[str] = field(default_factory=list)


def constraint_table_reconstruction_enabled() -> bool:
    return os.getenv("BOXBIIBOO_ENABLE_CONSTRAINT_TABLE_RECONSTRUCTION", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def build_cell_graph(
    words: list[dict[str, Any]],
    row_boxes: list[BBox],
    col_boxes: list[BBox],
    table_bbox: BBox | None,
) -> list[CellGraphNode]:
    """Assign OCR/PDF words into a row/column graph using geometry."""

    buckets: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for word in words:
        bbox = _coerce_bbox(word.get("bbox") or word.get("box"))
        if bbox is None:
            continue
        if table_bbox and not _intersects(bbox, table_bbox):
            continue
        center_x = (bbox[0] + bbox[2]) / 2.0
        center_y = (bbox[1] + bbox[3]) / 2.0
        row_index = _band_index(center_y, row_boxes, axis="y")
        col_index = _band_index(center_x, col_boxes, axis="x")
        if row_index is None or col_index is None:
            continue
        buckets.setdefault((row_index, col_index), []).append({**word, "bbox": bbox})

    nodes: list[CellGraphNode] = []
    for (row_index, col_index), assigned in sorted(buckets.items()):
        ordered = sorted(assigned, key=lambda item: (item["bbox"][1], item["bbox"][0]))
        text = " ".join(str(item.get("text") or "").strip() for item in ordered if str(item.get("text") or "").strip())
        confidences = [
            float(item["confidence"])
            for item in ordered
            if item.get("confidence") is not None
        ]
        nodes.append(
            CellGraphNode(
                row_index=row_index,
                col_index=col_index,
                text=_normalize_space(text),
                bbox=_union_bbox(item["bbox"] for item in ordered),
                confidence=sum(confidences) / len(confidences) if confidences else None,
                words=ordered,
            )
        )
    return nodes


def infer_table_schema(cells: list[CellGraphNode] | list[dict[str, Any]]) -> TableSchema:
    nodes = [_coerce_node(cell) for cell in cells]
    matrix = _nodes_to_matrix(nodes)
    headers = _normalize_headers(matrix[0] if matrix else [])
    roles: dict[int, str] = {}
    for idx, header in enumerate(headers):
        normalized = _norm(header)
        if "chuong trinh" in normalized:
            roles[idx] = "program"
        elif "nguoi hoc" in normalized:
            roles[idx] = "learner"
        elif "thoi gian" in normalized:
            roles[idx] = "duration"
        elif "khoi luong" in normalized or "tin chi" in normalized:
            roles[idx] = "credits"
    if len(headers) >= 4:
        roles.setdefault(0, "program")
        roles.setdefault(1, "learner")
        roles.setdefault(2, "duration")
        roles.setdefault(3, "credits")

    canonical = list(headers)
    if {0, 1, 2, 3}.issubset(set(range(len(headers)))) and _looks_like_training_duration_table(headers):
        canonical = ["Chương trình", "Người học", "Thời gian", "Khối lượng tối thiểu"]
        roles = {0: "program", 1: "learner", 2: "duration", 3: "credits"}

    trace = [f"inferred {len(canonical)} columns"]
    if canonical != headers:
        trace.append("normalized noisy duration/credit headers")
    return TableSchema(headers=canonical, column_roles=roles, confidence=1.0 if roles else 0.5, trace=trace)


def generate_reconstruction_hypotheses(
    cells: list[CellGraphNode] | list[dict[str, Any]],
    schema: TableSchema,
) -> list[TableHypothesis]:
    nodes = [_coerce_node(cell) for cell in cells]
    base_matrix = _nodes_to_matrix(nodes)
    body = base_matrix[1:] if base_matrix else []
    repaired_body, repair_trace = _repair_rows(_fill_down_rows(body, schema), schema)
    hypotheses = [
        _build_hypothesis(schema.headers, body, schema, trace=["baseline matrix"]),
        _build_hypothesis(
            schema.headers,
            _fill_down_rows(body, schema),
            schema,
            trace=["fill-down vertical merged cells"],
        ),
        _build_hypothesis(schema.headers, repaired_body, schema, trace=repair_trace),
    ]
    split_rows, split_trace = _split_merged_rows(repaired_body, schema)
    hypotheses.append(_build_hypothesis(schema.headers, split_rows, schema, trace=[*repair_trace, *split_trace]))
    return hypotheses


def score_table_hypothesis(hypothesis: TableHypothesis) -> TableHypothesis:
    rows = hypothesis.rows
    col_count = len(hypothesis.headers)
    constraints = {
        "stable_columns": 1.0 if all(len(row) == col_count for row in rows) else 0.0,
        "header_quality": _header_quality(hypothesis.headers),
        "datatype_consistency": _datatype_consistency(rows, hypothesis.headers),
        "duration_pattern": _column_pattern_score(rows, hypothesis.headers, "duration", _DURATION_RE),
        "credit_pattern": _column_pattern_score(rows, hypothesis.headers, "credits", _CREDIT_RE),
        "fill_down": _fill_down_score(rows),
        "no_same_type_merge": _no_same_type_merge_score(rows, hypothesis.headers),
        "ocr_confidence": 1.0,
    }
    score = (
        constraints["stable_columns"] * 1.5
        + constraints["header_quality"] * 1.2
        + constraints["datatype_consistency"] * 1.3
        + constraints["duration_pattern"] * 1.0
        + constraints["credit_pattern"] * 1.0
        + constraints["fill_down"] * 0.5
        + constraints["no_same_type_merge"] * 1.0
        + constraints["ocr_confidence"] * 0.2
    )
    hypothesis.constraints.clear()
    hypothesis.constraints.update(constraints)
    hypothesis.score = round(score, 6)
    return hypothesis


def select_best_hypothesis(hypotheses: list[TableHypothesis]) -> TableHypothesis:
    if not hypotheses:
        return TableHypothesis(headers=[], rows=[], cells=[], score=0.0, constraints={}, trace=["no hypotheses"])
    scored = [score_table_hypothesis(hypothesis) for hypothesis in hypotheses]
    best = max(scored, key=lambda item: item.score)
    best.trace.append(f"selected best score={best.score}")
    return best


def export_table_records(best_hypothesis: TableHypothesis) -> list[dict[str, str]]:
    return [dict(zip(best_hypothesis.headers, row, strict=False)) for row in best_hypothesis.rows]


def export_markdown(best_hypothesis: TableHypothesis) -> str:
    rows = [best_hypothesis.headers, *best_hypothesis.rows] if best_hypothesis.headers else best_hypothesis.rows
    if not rows:
        return ""
    col_count = max((len(row) for row in rows), default=0)
    padded = [row + [""] * (col_count - len(row)) for row in rows]
    lines = ["| " + " | ".join(row) + " |" for row in padded]
    if len(lines) == 1:
        return lines[0]
    separator = "| " + " | ".join(["---"] * col_count) + " |"
    return "\n".join([lines[0], separator, *lines[1:]])


def export_csv(best_hypothesis: TableHypothesis) -> str:
    output = io.StringIO()
    writer = csv.writer(output)
    if best_hypothesis.headers:
        writer.writerow(best_hypothesis.headers)
    writer.writerows(best_hypothesis.rows)
    return output.getvalue().strip("\r\n")


def export_json(best_hypothesis: TableHypothesis) -> str:
    payload = {
        "headers": best_hypothesis.headers,
        "records": export_table_records(best_hypothesis),
        "score": best_hypothesis.score,
        "constraints": best_hypothesis.constraints,
        "trace": best_hypothesis.trace,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def reconstruct_from_rows(rows: list[list[str]]) -> TableHypothesis:
    cells = [
        CellGraphNode(row_index=row_index, col_index=col_index, text=cell)
        for row_index, row in enumerate(rows)
        for col_index, cell in enumerate(row)
    ]
    schema = infer_table_schema(cells)
    return select_best_hypothesis(generate_reconstruction_hypotheses(cells, schema))


def cells_from_hypothesis(best_hypothesis: TableHypothesis, *, table_id: str | None = None, page: int | None = None) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for col_index, header in enumerate(best_hypothesis.headers):
        cells.append(
            {
                "table_id": table_id,
                "page": page,
                "row": 0,
                "col": col_index,
                "row_span": 1,
                "col_span": 1,
                "text": header,
                "row_header": None,
                "col_header": header,
                "is_header": True,
                "metadata": {"reconstructed": True},
            }
        )
    for body_index, row in enumerate(best_hypothesis.rows, start=1):
        row_header = row[0] if row else None
        for col_index, text in enumerate(row):
            if not text:
                continue
            col_header = best_hypothesis.headers[col_index] if col_index < len(best_hypothesis.headers) else None
            cells.append(
                {
                    "table_id": table_id,
                    "page": page,
                    "row": body_index,
                    "col": col_index,
                    "row_span": 1,
                    "col_span": 1,
                    "text": text,
                    "row_header": row_header,
                    "col_header": col_header,
                    "is_header": False,
                    "metadata": {"reconstructed": True},
                }
            )
    return cells


_DURATION_RE = re.compile(r"^\d+(?:,\d+)?\s*năm$", re.IGNORECASE)
_CREDIT_RE = re.compile(r"^\d+\s+tín\s+chỉ$", re.IGNORECASE)


def _build_hypothesis(headers: list[str], rows: list[list[str]], schema: TableSchema, *, trace: list[str]) -> TableHypothesis:
    width = len(headers)
    normalized_rows = [_pad_row(row, width) for row in rows if any(cell.strip() for cell in row)]
    cells = []
    for row_index, row in enumerate([headers, *normalized_rows]):
        for col_index, text in enumerate(row):
            if text:
                cells.append({"row": row_index, "col": col_index, "text": text})
    return TableHypothesis(
        headers=headers,
        rows=normalized_rows,
        cells=cells,
        score=0.0,
        constraints={},
        trace=[*schema.trace, *trace],
    )


def _split_merged_rows(rows: list[list[str]], schema: TableSchema) -> tuple[list[list[str]], list[str]]:
    roles = schema.column_roles
    duration_col = _role_col(roles, "duration")
    credit_col = _role_col(roles, "credits")
    learner_col = _role_col(roles, "learner")
    program_col = _role_col(roles, "program")
    if duration_col is None or credit_col is None:
        return rows, ["split skipped: missing duration/credit columns"]

    result: list[list[str]] = []
    trace = ["split merged rows by duration/credit constraints"]
    for row in rows:
        durations = _split_durations(row[duration_col] if duration_col < len(row) else "")
        credits = _split_credits(row[credit_col] if credit_col < len(row) else "")
        split_count = max(len(durations), len(credits))
        if split_count <= 1:
            result.append(row)
            continue
        learners = _split_learners(row[learner_col] if learner_col is not None and learner_col < len(row) else "", split_count)
        trace.append(f"row '{row[program_col] if program_col is not None and program_col < len(row) else ''}' split into {split_count} rows")
        for index in range(split_count):
            new_row = list(row)
            if duration_col < len(new_row):
                new_row[duration_col] = durations[index] if index < len(durations) else ""
            if credit_col < len(new_row):
                new_row[credit_col] = credits[index] if index < len(credits) else ""
            if learner_col is not None and learner_col < len(new_row) and learners:
                new_row[learner_col] = learners[index] if index < len(learners) else new_row[learner_col]
            result.append(new_row)
    deduped, dedupe_trace = _dedupe_rows(result)
    return deduped, [*trace, *dedupe_trace]


def _repair_rows(rows: list[list[str]], schema: TableSchema) -> tuple[list[list[str]], list[str]]:
    roles = schema.column_roles
    learner_col = _role_col(roles, "learner")
    duration_col = _role_col(roles, "duration")
    credit_col = _role_col(roles, "credits")
    if learner_col is None:
        return rows, ["row semantic repair skipped: missing learner column"]

    repaired: list[list[str]] = []
    trace = ["repair row semantics"]
    changed = False
    for row in rows:
        new_row = list(row)
        learner = new_row[learner_col] if learner_col < len(new_row) else ""
        cleaned_learner, moved_duration = _repair_learner_text(learner)
        if cleaned_learner != learner:
            new_row[learner_col] = cleaned_learner
            changed = True
        if (
            moved_duration
            and duration_col is not None
            and duration_col < len(new_row)
            and not new_row[duration_col].strip()
        ):
            new_row[duration_col] = moved_duration
            changed = True
        if credit_col is not None and credit_col < len(new_row):
            new_row[credit_col] = _normalize_credit_text(new_row[credit_col])
        if duration_col is not None and duration_col < len(new_row):
            new_row[duration_col] = _normalize_duration_text(new_row[duration_col])
        repaired.append(new_row)

    deduped, dedupe_trace = _dedupe_rows(repaired)
    if changed:
        trace.append("moved duration tokens from learner cells")
        trace.append("normalized learner word order")
    else:
        trace.append("no semantic row repair needed")
    return deduped, [*trace, *dedupe_trace]


def _fill_down_rows(rows: list[list[str]], schema: TableSchema) -> list[list[str]]:
    program_col = _role_col(schema.column_roles, "program")
    if program_col is None:
        return rows
    filled: list[list[str]] = []
    last = ""
    for row in rows:
        new_row = list(row)
        if program_col < len(new_row):
            if new_row[program_col].strip():
                last = new_row[program_col].strip()
            elif last:
                new_row[program_col] = last
        filled.append(new_row)
    return filled


def _split_durations(text: str) -> list[str]:
    text = _normalize_space(text)
    direct = re.findall(r"\d+(?:,\d+)?\s*năm", text, flags=re.IGNORECASE)
    if len(direct) >= 2:
        return [_normalize_space(item) for item in direct]
    numbers = re.findall(r"\d+(?:,\d+)?", text)
    if len(numbers) >= 2 and "năm" in text.lower():
        return [f"{number} năm" for number in numbers]
    return direct or ([text] if text else [])


def _split_credits(text: str) -> list[str]:
    text = _normalize_space(text)
    direct = re.findall(r"\d+\s+tín\s+chỉ", text, flags=re.IGNORECASE)
    if len(direct) >= 2:
        return [_normalize_space(item) for item in direct]
    numbers = re.findall(r"\d+", text)
    if len(numbers) >= 2 and "tín" in text.lower() and "chỉ" in text.lower():
        return [f"{number} tín chỉ" for number in numbers]
    return direct or ([text] if text else [])


def _split_learners(text: str, count: int) -> list[str]:
    normalized = _normalize_space(text)
    if count == 2 and _norm(normalized) == _norm("Tốt Tốt nghiệp nghiệp thạc đại học sĩ"):
        return ["Tốt nghiệp thạc sĩ", "Tốt nghiệp đại học"]
    if count == 2 and ";" in normalized:
        parts = [_normalize_space(part) for part in normalized.split(";") if part.strip()]
        if len(parts) == 2:
            return parts
    return [normalized] * count if normalized else []


def _repair_learner_text(text: str) -> tuple[str, str]:
    normalized = _normalize_space(text)
    durations = _split_durations(normalized)
    moved_duration = durations[0] if len(durations) == 1 else ""
    without_duration = normalized
    if moved_duration:
        without_duration = _normalize_space(_DURATION_RE.sub("", without_duration))

    norm = _norm(without_duration)
    if _has_words(norm, "tot", "nghiep", "thac", "dai", "hoc", "si"):
        return without_duration, moved_duration
    if _has_words(norm, "tot", "nghiep", "cu", "nhan", "chuong", "trinh", "tich", "hop"):
        return "Tốt nghiệp cử nhân theo chương trình tích hợp", moved_duration
    if _has_words(norm, "tot", "nghiep", "cu", "nhan"):
        return "Tốt nghiệp cử nhân", moved_duration
    if _has_words(norm, "tot", "nghiep", "thpt"):
        return "Tốt nghiệp THPT", moved_duration
    if _has_words(norm, "tot", "nghiep", "thac", "si"):
        return "Tốt nghiệp thạc sĩ", moved_duration
    if _has_words(norm, "tot", "nghiep", "dai", "hoc"):
        return "Tốt nghiệp đại học", moved_duration
    return without_duration, moved_duration


def _normalize_duration_text(text: str) -> str:
    durations = _split_durations(text)
    if len(durations) == 1:
        return durations[0]
    return _normalize_space(text)


def _normalize_credit_text(text: str) -> str:
    credits = _split_credits(text)
    if len(credits) == 1:
        return credits[0]
    return _normalize_space(text)


def _dedupe_rows(rows: list[list[str]]) -> tuple[list[list[str]], list[str]]:
    seen: set[tuple[str, ...]] = set()
    deduped: list[list[str]] = []
    removed = 0
    for row in rows:
        key = tuple(_norm(cell) for cell in row)
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        deduped.append(row)
    return deduped, [f"deduplicated {removed} duplicate rows"] if removed else []


def _has_words(text: str, *words: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", text))
    return all(word in tokens for word in words)


def _nodes_to_matrix(nodes: list[CellGraphNode]) -> list[list[str]]:
    row_count = max((node.row_index + 1 for node in nodes), default=0)
    col_count = max((node.col_index + 1 for node in nodes), default=0)
    matrix = [[""] * col_count for _ in range(row_count)]
    for node in nodes:
        matrix[node.row_index][node.col_index] = _normalize_space(node.text)
    return matrix


def _normalize_headers(headers: list[str]) -> list[str]:
    return [_normalize_space(header) for header in headers]


def _looks_like_training_duration_table(headers: list[str]) -> bool:
    joined = _norm(" ".join(headers))
    return (
        "chuong trinh" in joined
        and "nguoi hoc" in joined
        and ("thoi gian" in joined or "khoi" in joined or "tin chi" in joined)
    )


def _role_col(roles: dict[int, str], role: str) -> int | None:
    return next((idx for idx, value in roles.items() if value == role), None)


def _pad_row(row: list[str], width: int) -> list[str]:
    return [_normalize_space(cell) for cell in row[:width]] + [""] * max(0, width - len(row))


def _header_quality(headers: list[str]) -> float:
    non_empty = sum(1 for header in headers if header.strip())
    return non_empty / len(headers) if headers else 0.0


def _datatype_consistency(rows: list[list[str]], headers: list[str]) -> float:
    if not rows or len(headers) < 4:
        return 0.0
    duration_col = _find_header_col(headers, "thời gian", "thoi gian")
    credit_col = _find_header_col(headers, "khối lượng", "khoi luong", "tín chỉ", "tin chi")
    scores = []
    if duration_col is not None:
        scores.append(_column_pattern_score_by_index(rows, duration_col, _DURATION_RE))
    if credit_col is not None:
        scores.append(_column_pattern_score_by_index(rows, credit_col, _CREDIT_RE))
    return sum(scores) / len(scores) if scores else 0.5


def _column_pattern_score(rows: list[list[str]], headers: list[str], role: str, pattern: re.Pattern[str]) -> float:
    if role == "duration":
        col = _find_header_col(headers, "thời gian", "thoi gian")
    else:
        col = _find_header_col(headers, "khối lượng", "khoi luong", "tín chỉ", "tin chi")
    return _column_pattern_score_by_index(rows, col, pattern) if col is not None else 0.0


def _column_pattern_score_by_index(rows: list[list[str]], col: int, pattern: re.Pattern[str]) -> float:
    values = [row[col] for row in rows if col < len(row) and row[col].strip()]
    if not values:
        return 0.0
    return sum(1 for value in values if pattern.match(value.strip())) / len(values)


def _fill_down_score(rows: list[list[str]]) -> float:
    if not rows:
        return 0.0
    blanks = sum(1 for row in rows if row and not row[0].strip())
    return 1.0 if blanks == 0 else max(0.0, 1.0 - blanks / len(rows))


def _no_same_type_merge_score(rows: list[list[str]], headers: list[str]) -> float:
    duration_col = _find_header_col(headers, "thời gian", "thoi gian")
    credit_col = _find_header_col(headers, "khối lượng", "khoi luong", "tín chỉ", "tin chi")
    bad = 0
    total = 0
    for row in rows:
        for col, splitter in ((duration_col, _split_durations), (credit_col, _split_credits)):
            if col is None or col >= len(row) or not row[col].strip():
                continue
            total += 1
            if len(splitter(row[col])) > 1:
                bad += 1
    return 1.0 if total == 0 else max(0.0, 1.0 - bad / total)


def _find_header_col(headers: list[str], *needles: str) -> int | None:
    normalized_needles = [_norm(needle) for needle in needles]
    for index, header in enumerate(headers):
        normalized = _norm(header)
        if any(needle in normalized for needle in normalized_needles):
            return index
    return None


def _coerce_node(cell: CellGraphNode | dict[str, Any]) -> CellGraphNode:
    if isinstance(cell, CellGraphNode):
        return cell
    row = int(cell.get("row_index", cell.get("row", 0)) or 0)
    col = int(cell.get("col_index", cell.get("col", 0)) or 0)
    return CellGraphNode(
        row_index=row,
        col_index=col,
        text=str(cell.get("text") or ""),
        bbox=_coerce_bbox(cell.get("bbox")),
        confidence=cell.get("confidence"),
        words=list(cell.get("source_words") or cell.get("words") or []),
        metadata=dict(cell.get("metadata") or {}),
    )


def _coerce_bbox(value: Any) -> BBox | None:
    if value is None:
        return None
    if len(value) != 4:
        return None
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def _band_index(center: float, boxes: list[BBox], *, axis: str) -> int | None:
    offset = 1 if axis == "y" else 0
    for index, box in enumerate(boxes):
        if float(box[offset]) <= center <= float(box[offset + 2]):
            return index
    return None


def _intersects(a: BBox, b: BBox) -> bool:
    return not (a[2] <= b[0] or a[0] >= b[2] or a[3] <= b[1] or a[1] >= b[3])


def _union_bbox(boxes: Iterable[BBox]) -> BBox | None:
    collected = list(boxes)
    if not collected:
        return None
    return (
        min(box[0] for box in collected),
        min(box[1] for box in collected),
        max(box[2] for box in collected),
        max(box[3] for box in collected),
    )


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", _repair_mojibake(str(text or ""))).strip()


def _repair_mojibake(text: str) -> str:
    if not text or not any(marker in text for marker in ("Ä", "Æ", "Ã", "á", "º", "»")):
        return text
    for encoding in ("latin1", "cp1252"):
        try:
            repaired = text.encode(encoding).decode("utf-8")
        except Exception:
            continue
        if repaired and repaired != text:
            return repaired
    return text


def _norm(text: str) -> str:
    replacements = {
        "đ": "d",
        "Đ": "d",
        "á": "a",
        "à": "a",
        "ả": "a",
        "ã": "a",
        "ạ": "a",
        "ă": "a",
        "ắ": "a",
        "ằ": "a",
        "ẳ": "a",
        "ẵ": "a",
        "ặ": "a",
        "â": "a",
        "ấ": "a",
        "ầ": "a",
        "ẩ": "a",
        "ẫ": "a",
        "ậ": "a",
        "é": "e",
        "è": "e",
        "ẻ": "e",
        "ẽ": "e",
        "ẹ": "e",
        "ê": "e",
        "ế": "e",
        "ề": "e",
        "ể": "e",
        "ễ": "e",
        "ệ": "e",
        "í": "i",
        "ì": "i",
        "ỉ": "i",
        "ĩ": "i",
        "ị": "i",
        "ó": "o",
        "ò": "o",
        "ỏ": "o",
        "õ": "o",
        "ọ": "o",
        "ô": "o",
        "ố": "o",
        "ồ": "o",
        "ổ": "o",
        "ỗ": "o",
        "ộ": "o",
        "ơ": "o",
        "ớ": "o",
        "ờ": "o",
        "ở": "o",
        "ỡ": "o",
        "ợ": "o",
        "ú": "u",
        "ù": "u",
        "ủ": "u",
        "ũ": "u",
        "ụ": "u",
        "ư": "u",
        "ứ": "u",
        "ừ": "u",
        "ử": "u",
        "ữ": "u",
        "ự": "u",
        "ý": "y",
        "ỳ": "y",
        "ỷ": "y",
        "ỹ": "y",
        "ỵ": "y",
    }
    normalized = "".join(replacements.get(ch, ch) for ch in text)
    return _normalize_space(normalized).lower()
