from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from app.qa.text_utils import normalize_text


NUMBER_RE = re.compile(r"(?<![\w.])[-+]?\d+(?:[,.]\d+)?%?(?![\w.])")
GRADE_RE = re.compile(r"(?<!\w)([A-F][+-]?)(?!\w)", re.I)
PIPE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)+\|?\s*$")
RANGE_RE = re.compile(
    r"(?P<low>[-+]?\d+(?:[,.]\d+)?%?)\s*(?:-|to|through|den|toi)\s*(?P<high>[-+]?\d+(?:[,.]\d+)?%?)",
    re.I,
)
LOWER_BOUND_RE = re.compile(
    r"(?:>=|>|from|tu|at\s+least|min(?:imum)?|above|over)\s*(?P<low>[-+]?\d+(?:[,.]\d+)?%?)",
    re.I,
)
UPPER_BOUND_RE = re.compile(
    r"(?:<=|<|under|below|duoi|at\s+most|up\s+to|max(?:imum)?|khong\s+qua)\s*(?P<high>[-+]?\d+(?:[,.]\d+)?%?)",
    re.I,
)
TRAILING_LOWER_BOUND_RE = re.compile(
    r"(?P<low>[-+]?\d+(?:[,.]\d+)?%?)\s*(?:or\s+more|and\s+above|tro\s+len|upwards)",
    re.I,
)
TRAILING_UPPER_BOUND_RE = re.compile(
    r"(?P<high>[-+]?\d+(?:[,.]\d+)?%?)\s*(?:or\s+less|and\s+below|tro\s+xuong|downwards)",
    re.I,
)

HEADER_KEYWORDS: dict[str, tuple[str, ...]] = {
    "grade": ("grade", "letter_grade", "letter grade", "letter", "diem_chu", "diem chu"),
    "grade_point": (
        "grade_point",
        "grade point",
        "gpa",
        "point",
        "points",
        "numeric value",
        "diem_so",
        "diem so",
        "diem_he",
        "diem he",
        "thang diem",
    ),
    "range": (
        "range",
        "score range",
        "score band",
        "band",
        "interval",
        "threshold",
        "rate",
        "completion",
        "percent",
        "percentage",
        "khoang",
        "nguong",
        "diem",
    ),
    "classification": (
        "classification",
        "class",
        "category",
        "label",
        "type",
        "xep_loai",
        "xep loai",
        "level",
        "muc",
        "loai",
        "nhom",
        "action",
        "status",
    ),
    "model": ("model", "configuration", "config", "variant"),
    "heads": ("head", "heads"),
    "layers": ("layer", "layers"),
    "d_model": ("d_model", "model dimension", "dimension"),
    "bleu": ("bleu",),
}


@dataclass(slots=True, frozen=True)
class NumericInterval:
    low: float | None = None
    high: float | None = None
    low_inclusive: bool = True
    high_inclusive: bool = True
    text: str = ""

    def contains(self, value: float) -> bool:
        if self.low is not None:
            if value < self.low or (value == self.low and not self.low_inclusive):
                return False
        if self.high is not None:
            if value > self.high or (value == self.high and not self.high_inclusive):
                return False
        return self.low is not None or self.high is not None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class TableColumn:
    index: int
    label: str
    key: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class TableCell:
    column: TableColumn
    raw: str
    text: str
    folded: str
    number: float | None = None
    grade: str | None = None
    interval: NumericInterval | None = None

    def to_prompt_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "column": self.column.label,
            "value": self.text,
        }
        if self.number is not None:
            payload["number"] = self.number
        if self.grade is not None:
            payload["grade"] = self.grade
        if self.interval is not None:
            payload["interval"] = self.interval.to_dict()
        return payload


@dataclass(slots=True, frozen=True)
class TableRow:
    index: int
    cells: tuple[TableCell, ...]

    def get(self, column: TableColumn) -> TableCell | None:
        for cell in self.cells:
            if cell.column.index == column.index:
                return cell
        return None

    def non_empty_cells(self) -> list[TableCell]:
        return [cell for cell in self.cells if cell.text]

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.index,
            "cells": [cell.to_prompt_dict() for cell in self.non_empty_cells()],
        }


@dataclass(slots=True, frozen=True)
class NormalizedTable:
    columns: tuple[TableColumn, ...]
    rows: tuple[TableRow, ...]
    header_rows: tuple[tuple[str, ...], ...] = field(default_factory=tuple)
    original_text: str = ""

    @property
    def rendered_text(self) -> str:
        if not self.columns:
            return normalize_text(self.original_text)
        lines = [" | ".join(column.label for column in self.columns)]
        for row in self.rows:
            values = []
            for column in self.columns:
                cell = row.get(column)
                values.append(cell.text if cell is not None else "")
            lines.append(" | ".join(values))
        return "\n".join(lines)

    def prompt_metadata(self) -> dict[str, Any]:
        return {
            "table_header_rows": [list(row) for row in self.header_rows],
            "logical_columns": [column.to_dict() for column in self.columns],
            "normalized_intervals": self.interval_traces(),
            "lookup_index": self.lookup_index(),
        }

    def interval_traces(self) -> list[dict[str, Any]]:
        traces: list[dict[str, Any]] = []
        for row in self.rows:
            for cell in row.cells:
                if cell.interval is None:
                    continue
                traces.append(
                    {
                        "row_index": row.index,
                        "column": cell.column.label,
                        "cell": cell.text,
                        "interval": cell.interval.to_dict(),
                    }
                )
        return traces

    def lookup_index(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in self.rows:
            values = {
                cell.column.label: cell.text
                for cell in row.non_empty_cells()
            }
            rows.append({"row_index": row.index, "values": values})
        return rows


@dataclass(slots=True, frozen=True)
class TableLookupResult:
    answer: str
    row_index: int
    confidence: float
    trace: dict[str, Any]


def normalize_table_from_sources(
    *,
    table_text: str | None = None,
    table_rows: Any = None,
    table_json: Any = None,
) -> NormalizedTable | None:
    rows_from_metadata = _coerce_mapping_rows(table_rows)
    if not rows_from_metadata and table_json is not None:
        rows_from_metadata = _coerce_mapping_rows(_extract_rows_from_json(table_json))
    if rows_from_metadata:
        return _table_from_mapping_rows(rows_from_metadata, original_text=table_text or "")
    if table_text:
        parsed = _table_from_text(table_text)
        if parsed is not None:
            return parsed
        parallel = _parallel_sequence_table_from_text(table_text)
        if parallel is not None:
            return parallel
    return None


def lookup_table_answer(question: str, table: NormalizedTable) -> TableLookupResult | None:
    query = _parse_query(question)
    if _requires_multi_row_reasoning(query):
        return None
    candidates: list[tuple[float, TableRow, list[TableCell], str]] = []
    for row in table.rows:
        score, matched_cells, reason = _score_row_match(query, row)
        if score > 0:
            candidates.append((score, row, matched_cells, reason))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    score, row, matched_cells, reason = candidates[0]
    if score < 2.0:
        return None

    target_cells = _select_target_cells(query, table, row, matched_cells)
    if not target_cells:
        return None

    answer = _compose_lookup_answer(question, row, matched_cells, target_cells, reason)
    if answer is None:
        return None
    return TableLookupResult(
        answer=answer,
        row_index=row.index,
        confidence=min(0.95, max(0.65, score / 8.0)),
        trace={
            "row_index": row.index,
            "match_reason": reason,
            "matched_cells": [cell.to_prompt_dict() for cell in matched_cells],
            "target_cells": [cell.to_prompt_dict() for cell in target_cells],
        },
    )


def lookup_table_answer_from_text(question: str, text: str) -> TableLookupResult | None:
    table = normalize_table_from_sources(table_text=text)
    if table is not None:
        result = lookup_table_answer(question, table)
        if result is not None:
            return result
    for candidate in _parallel_sequence_table_candidates_from_text(text):
        result = lookup_table_answer(question, candidate)
        if result is not None:
            return result
    return None


def table_rows_for_prompt(table: NormalizedTable | None) -> list[dict[str, Any]]:
    if table is None:
        return []
    return [row.to_prompt_dict() for row in table.rows]


def table_metadata_for_prompt(table: NormalizedTable | None) -> dict[str, Any]:
    if table is None:
        return {}
    return table.prompt_metadata()


def _table_from_mapping_rows(rows: list[Mapping[str, Any]], *, original_text: str = "") -> NormalizedTable | None:
    if not rows:
        return None
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            key_text = str(key)
            if key_text not in keys:
                keys.append(key_text)
    columns = tuple(TableColumn(index=idx, label=_header_label(key), key=_header_key(key)) for idx, key in enumerate(keys))
    table_rows: list[TableRow] = []
    for row_idx, row in enumerate(rows, start=1):
        cells = tuple(_make_cell(column, row.get(keys[column.index], "")) for column in columns)
        table_rows.append(TableRow(index=row_idx, cells=cells))
    return NormalizedTable(
        columns=columns,
        rows=tuple(table_rows),
        header_rows=(tuple(column.label for column in columns),),
        original_text=original_text,
    )


def _table_from_text(text: str) -> NormalizedTable | None:
    lines = [_normalize_table_text(line) for line in text.splitlines()]
    lines = [line for line in lines if line and not PIPE_SEPARATOR_RE.match(line)]
    if not lines:
        return None
    parsed_lines = [_split_table_line(line) for line in lines]
    parsed_lines = [cells for cells in parsed_lines if cells]
    if not parsed_lines:
        return None
    width = max(len(cells) for cells in parsed_lines)
    if width <= 1:
        return None
    first = parsed_lines[0]
    has_header = any(not NUMBER_RE.fullmatch(cell) for cell in first) and len(parsed_lines) > 1
    header_cells = first if has_header else [f"Column {idx + 1}" for idx in range(width)]
    header_cells = header_cells + [f"Column {idx + 1}" for idx in range(len(header_cells), width)]
    columns = tuple(
        TableColumn(index=idx, label=_header_label(header_cells[idx]), key=_header_key(header_cells[idx]))
        for idx in range(width)
    )
    body_lines = parsed_lines[1:] if has_header else parsed_lines
    table_rows: list[TableRow] = []
    for row_idx, raw_cells in enumerate(body_lines, start=1):
        padded = raw_cells + [""] * (width - len(raw_cells))
        cells = tuple(_make_cell(column, padded[column.index]) for column in columns)
        table_rows.append(TableRow(index=row_idx, cells=cells))
    return NormalizedTable(
        columns=columns,
        rows=tuple(table_rows),
        header_rows=(tuple(column.label for column in columns),),
        original_text=text,
    )


def _parallel_sequence_table_from_text(text: str) -> NormalizedTable | None:
    candidates = _parallel_sequence_table_candidates_from_text(text)
    if candidates:
        return candidates[0]
    return None


def _parallel_sequence_table_candidates_from_text(text: str) -> list[NormalizedTable]:
    candidates: list[NormalizedTable] = []
    current: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        parsed = _split_labeled_sequence_line(line)
        if parsed is None:
            if current:
                candidate = _parallel_sequence_table_from_sequences(current, original_text=text)
                if candidate is not None:
                    candidates.append(candidate)
                current = []
            continue
        current.append(parsed)
    if current:
        candidate = _parallel_sequence_table_from_sequences(current, original_text=text)
        if candidate is not None:
            candidates.append(candidate)

    sequences: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        parsed = _split_labeled_sequence_line(line)
        if parsed is not None:
            sequences.append(parsed)
    global_candidate = _parallel_sequence_table_from_sequences(sequences, original_text=text)
    if global_candidate is not None:
        candidates.append(global_candidate)
    return _dedupe_tables(candidates)


def _parallel_sequence_table_from_sequences(
    sequences: list[tuple[str, list[str]]],
    *,
    original_text: str,
) -> NormalizedTable | None:
    if len(sequences) < 2:
        return None

    width_counts = Counter(len(values) for _, values in sequences if len(values) >= 2)
    if not width_counts:
        return None
    width, _ = max(width_counts.items(), key=lambda item: (item[1], item[0]))
    selected = [(label, values) for label, values in sequences if len(values) >= width]
    if len(selected) < 2:
        return None
    width = min(len(values) for _, values in selected)
    if width < 2:
        return None

    rows: list[dict[str, str]] = []
    for value_idx in range(width):
        row: dict[str, str] = {}
        for label, values in selected:
            row[label] = values[value_idx]
        rows.append(row)
    return _table_from_mapping_rows(rows, original_text=original_text)


def _split_labeled_sequence_line(line: str) -> tuple[str, list[str]] | None:
    normalized = _normalize_table_text(line).strip().strip("|")
    if not normalized or "|" in normalized or "\t" in normalized or len(normalized) > 500:
        return None
    tokens = _merge_range_tokens(normalized.split())
    if len(tokens) < 3:
        return None

    best: tuple[float, int, list[str]] | None = None
    max_label_tokens = min(5, len(tokens) - 2)
    for split_idx in range(1, max_label_tokens + 1):
        label_tokens = tokens[:split_idx]
        value_tokens = tokens[split_idx:]
        if not any(re.search(r"[^\W\d_]", token, flags=re.UNICODE) for token in label_tokens):
            continue
        first_scores = [_sequence_value_score(token) for token in value_tokens[:2]]
        if not first_scores or max(first_scores) < 0.60:
            continue
        scores = [_sequence_value_score(token) for token in value_tokens]
        strong_count = sum(1 for score in scores if score >= 0.60)
        if strong_count < 2:
            continue
        ratio = sum(scores) / max(1, len(scores))
        if ratio < 0.55:
            continue
        score = ratio + min(split_idx, 4) * 0.03
        if best is None or score > best[0]:
            best = (score, split_idx, value_tokens)

    if best is None:
        return None
    _, split_idx, value_tokens = best
    label = _header_label(" ".join(tokens[:split_idx]))
    values = [_normalize_cell_text(token) for token in value_tokens]
    return label, values


def _merge_range_tokens(tokens: list[str]) -> list[str]:
    merged: list[str] = []
    idx = 0
    while idx < len(tokens):
        if (
            idx + 2 < len(tokens)
            and _is_number_token(tokens[idx])
            and tokens[idx + 1].lower() in {"-", "to", "through", "den", "toi"}
            and _is_number_token(tokens[idx + 2])
        ):
            merged.append(f"{tokens[idx]}-{tokens[idx + 2]}")
            idx += 3
            continue
        merged.append(tokens[idx])
        idx += 1
    return merged


def _sequence_value_score(token: str) -> float:
    cleaned = token.strip(" ,;:()[]")
    if not cleaned:
        return 0.0
    normalized = _normalize_cell_text(cleaned)
    if _parse_interval(normalized) is not None:
        return 1.0
    if _is_number_token(normalized):
        return 1.0
    if GRADE_RE.fullmatch(normalized):
        return 1.0
    if re.fullmatch(r"[A-Z][A-Z0-9_+.-]{0,9}", cleaned):
        return 0.85
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_+.-]{1,24}", cleaned) and cleaned[:1].isupper():
        return 0.65
    folded = _fold(cleaned)
    if folded and folded not in _SEQUENCE_STOPWORDS and len(folded) <= 24:
        return 0.45
    return 0.0


def _is_number_token(token: str) -> bool:
    return _to_float(token.strip("%")) is not None


_SEQUENCE_STOPWORDS = {
    "and",
    "or",
    "the",
    "of",
    "for",
    "with",
    "va",
    "hoac",
    "cua",
    "cho",
    "theo",
    "quy",
    "doi",
}


def _split_table_line(line: str) -> list[str]:
    stripped = line.strip().strip("|")
    if "|" in stripped:
        return [_normalize_cell_text(part) for part in stripped.split("|")]
    if "\t" in stripped:
        return [_normalize_cell_text(part) for part in stripped.split("\t")]
    parts = re.split(r"\s{2,}", stripped)
    return [_normalize_cell_text(part) for part in parts]


def _make_cell(column: TableColumn, value: Any) -> TableCell:
    raw = "" if value is None else str(value)
    text = _normalize_cell_text(raw)
    folded = _fold(text)
    interval = _parse_interval(text)
    grade = _first_grade(text)
    number = _first_number(text) if interval is None else None
    return TableCell(column=column, raw=raw, text=text, folded=folded, number=number, grade=grade, interval=interval)


def _normalize_table_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-")
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    return normalize_text(text)


def _normalize_cell_text(text: str) -> str:
    normalized = _normalize_table_text(text).strip()
    normalized = re.sub(r"(?<=\d),(?=\d)", ".", normalized)
    normalized = re.sub(r"\s*-\s*", " - ", normalized)
    normalized = re.sub(r"\b([A-Fa-f])\s*([+-])\b", lambda m: f"{m.group(1).upper()}{m.group(2)}", normalized)
    normalized = re.sub(r"\b([A-Fa-f])\b", lambda m: m.group(1).upper(), normalized)
    return normalize_text(normalized)


def _header_label(value: str) -> str:
    text = _normalize_cell_text(str(value)).strip()
    return text.replace("_", " ").strip().title() if text.islower() or "_" in text else text


def _header_key(value: str) -> str:
    folded = _fold(str(value))
    folded = re.sub(r"[^a-z0-9]+", "_", folded).strip("_")
    return folded or "column"


def _parse_interval(text: str) -> NumericInterval | None:
    normalized = _normalize_cell_text(text)
    folded = _fold(normalized)
    range_match = RANGE_RE.search(folded)
    if range_match:
        low = _to_float(range_match.group("low"))
        high = _to_float(range_match.group("high"))
        if low is not None and high is not None:
            low_value, high_value = min(low, high), max(low, high)
            return NumericInterval(low=low_value, high=high_value, text=normalized)
    lower = LOWER_BOUND_RE.search(folded) or TRAILING_LOWER_BOUND_RE.search(folded)
    upper = UPPER_BOUND_RE.search(folded) or TRAILING_UPPER_BOUND_RE.search(folded)
    if lower and upper:
        low = _to_float(lower.group("low"))
        high = _to_float(upper.group("high"))
        if low is not None and high is not None:
            return NumericInterval(low=min(low, high), high=max(low, high), text=normalized)
    if lower:
        low = _to_float(lower.group("low"))
        if low is not None:
            inclusive = not folded.strip().startswith(">")
            return NumericInterval(low=low, low_inclusive=inclusive, text=normalized)
    if upper:
        high = _to_float(upper.group("high"))
        if high is not None:
            inclusive = not any(term in folded for term in ("below", "under", "duoi", "<"))
            return NumericInterval(high=high, high_inclusive=inclusive, text=normalized)
    return None


def _parse_query(question: str) -> dict[str, Any]:
    folded = _fold(question)
    return {
        "folded": folded,
        "numbers": _numbers(question),
        "grades": _grades(question),
        "target_groups": _target_groups(folded),
        "tokens": set(re.findall(r"[a-z0-9_]+", folded)),
    }


def _requires_multi_row_reasoning(query: dict[str, Any]) -> bool:
    folded = str(query["folded"])
    if len(query["grades"]) < 2 or query["numbers"]:
        return False
    return any(
        term in folded
        for term in (
            "compare",
            "difference",
            "versus",
            " vs ",
            "higher",
            "lower",
            "hon",
            "cao hon",
            "thap hon",
            "khac",
            "khac nhau",
        )
    )


def _score_row_match(query: dict[str, Any], row: TableRow) -> tuple[float, list[TableCell], str]:
    matched: list[TableCell] = []
    score = 0.0
    reason = ""

    for number in query["numbers"]:
        interval_matches = [cell for cell in row.cells if cell.interval is not None and cell.interval.contains(number)]
        if interval_matches:
            matched.extend(interval_matches)
            score += 5.0
            reason = "interval_contains_number"
            continue
        exact_matches = [
            cell
            for cell in row.cells
            if cell.number is not None and abs(cell.number - number) < 1e-9
        ]
        if exact_matches:
            matched.extend(exact_matches)
            score += 4.0
            reason = "exact_number_match"

    for grade in query["grades"]:
        grade_matches = [cell for cell in row.cells if cell.grade == grade]
        if grade_matches:
            matched.extend(grade_matches)
            score += 4.5
            reason = "grade_match"

    query_tokens = query["tokens"]
    for cell in row.cells:
        if not cell.folded or cell.interval is not None:
            continue
        cell_tokens = set(re.findall(r"[a-z0-9_]+", cell.folded))
        overlap = query_tokens & cell_tokens
        if overlap and len(cell.folded) > 1:
            overlap_score = min(3.0, len(overlap) * 1.5)
            if cell.folded in query["folded"]:
                overlap_score += 1.0
            matched.append(cell)
            score += overlap_score
            reason = reason or "text_cell_match"

    deduped = _dedupe_cells(matched)
    return score, deduped, reason or "row_match"


def _select_target_cells(
    query: dict[str, Any],
    table: NormalizedTable,
    row: TableRow,
    matched_cells: list[TableCell],
) -> list[TableCell]:
    has_interval_match = any(cell.interval is not None for cell in matched_cells)
    matched_column_ids = {
        cell.column.index
        for cell in matched_cells
        if not has_interval_match or cell.interval is not None
    }
    target_columns = _columns_for_target_groups(table, query["target_groups"])
    target_cells: list[TableCell] = []
    for column in target_columns:
        cell = row.get(column)
        if cell is not None and cell.text and cell.column.index not in matched_column_ids:
            target_cells.append(cell)

    if target_cells:
        return _dedupe_cells(target_cells)

    if any(cell.interval is not None for cell in matched_cells):
        preferred = _first_cell_by_groups(row, ("grade", "classification", "model"), exclude=matched_column_ids)
        if preferred is not None:
            return [preferred]

    if query["grades"]:
        preferred = _first_cell_by_groups(row, ("range", "grade_point", "classification"), exclude=matched_column_ids)
        if preferred is not None:
            return [preferred]

    return [
        cell
        for cell in row.non_empty_cells()
        if cell.column.index not in matched_column_ids and cell.interval is None
    ][:2]


def _compose_lookup_answer(
    question: str,
    row: TableRow,
    matched_cells: list[TableCell],
    target_cells: list[TableCell],
    reason: str,
) -> str | None:
    if not matched_cells or not target_cells:
        return None
    first_match = matched_cells[0]
    targets = _dedupe_cells(target_cells)
    target_text = _format_target_cells(targets)
    if _looks_vietnamese(question):
        if first_match.interval is not None:
            number = _first_number(question)
            if number is not None:
                return f"{_format_number(number)} thuộc {first_match.text}, tương ứng {target_text}."
            return f"{first_match.text} tương ứng {target_text}."
        if first_match.grade is not None and reason == "grade_match":
            return f"{first_match.grade} tương ứng {target_text}."
        return f"{first_match.text} tương ứng {target_text}."
    if first_match.interval is not None:
        number = _first_number(question)
        if number is not None:
            return f"{_format_number(number)} falls in {first_match.text}, which maps to {target_text}."
        return f"{first_match.text} maps to {target_text}."
    if first_match.grade is not None and reason == "grade_match":
        return f"{first_match.grade} corresponds to {target_text}."
    return f"For {first_match.column.label} = {first_match.text}, {target_text}."


def _format_target_cells(cells: list[TableCell]) -> str:
    if len(cells) == 1:
        cell = cells[0]
        if _column_group(cell.column) in {"grade", "classification", "model"}:
            return cell.text
        return f"{cell.column.label} {cell.text}"
    return " and ".join(f"{cell.column.label} {cell.text}" for cell in cells)


def _target_groups(folded_question: str) -> list[str]:
    groups: list[str] = []
    for group, keywords in HEADER_KEYWORDS.items():
        if any(_fold(keyword) in folded_question for keyword in keywords):
            groups.append(group)
    if "which" in folded_question and "model" in folded_question:
        groups.insert(0, "model")
    if any(term in folded_question for term in ("bao nhieu diem", "score band", "score range", "what range", "which range", "khoang diem", "khoang nao")):
        groups.insert(0, "range")
    if any(term in folded_question for term in ("what point", "what value", "bao nhieu gia tri", "diem so", "gpa", "thang diem", "thang 4")):
        groups.insert(0, "grade_point")
    if any(term in folded_question for term in ("which level", "what level", "which category", "what category", "muc nao", "loai nao", "nhom nao")):
        groups.insert(0, "classification")
    if any(term in folded_question for term in ("letter grade", "diem chu", "grade")) and "grade_point" not in groups:
        groups.insert(0, "grade")
    return _dedupe_strings(groups)


def _columns_for_target_groups(table: NormalizedTable, groups: list[str]) -> list[TableColumn]:
    columns: list[TableColumn] = []
    for group in groups:
        for column in table.columns:
            if _column_group(column) == group:
                columns.append(column)
    return _dedupe_columns(columns)


def _first_cell_by_groups(row: TableRow, groups: Iterable[str], *, exclude: set[int]) -> TableCell | None:
    group_list = list(groups)
    for group in group_list:
        for cell in row.cells:
            if cell.column.index in exclude or not cell.text:
                continue
            if _column_group(cell.column) == group:
                return cell
    return None


def _column_group(column: TableColumn) -> str:
    label = f"{column.key} {_fold(column.label)}"
    for group, keywords in HEADER_KEYWORDS.items():
        if any(_fold(keyword) in label for keyword in keywords):
            return group
    return "value"


def _coerce_mapping_rows(value: Any) -> list[Mapping[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return _coerce_mapping_rows(parsed)
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        rows: list[Mapping[str, Any]] = []
        for item in value:
            if isinstance(item, Mapping):
                rows.append(item)
        return rows
    return []


def _extract_rows_from_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        for key in ("rows", "table_rows", "data"):
            if key in value:
                return value[key]
    return value


def _numbers(text: str) -> list[float]:
    values: list[float] = []
    for match in NUMBER_RE.finditer(_normalize_cell_text(text)):
        value = _to_float(match.group(0))
        if value is not None:
            values.append(value)
    return values


def _grades(text: str) -> list[str]:
    return _dedupe_strings(match.group(1).upper() for match in GRADE_RE.finditer(text))


def _first_number(text: str) -> float | None:
    values = _numbers(text)
    return values[0] if values else None


def _first_grade(text: str) -> str | None:
    grades = _grades(text)
    return grades[0] if grades else None


def _to_float(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = value.strip().replace("%", "")
    cleaned = re.sub(r"(?<=\d),(?=\d)", ".", cleaned)
    try:
        return float(cleaned)
    except ValueError:
        return None


def _format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _looks_vietnamese(text: str) -> bool:
    folded = _fold(text)
    return any(term in folded for term in ("bao nhieu", "ung voi", "tuong ung", "diem", "muc", "khoang", "thuoc", "thang"))


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", normalize_text(text or ""))
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_text).strip().lower()


def _dedupe_cells(cells: Iterable[TableCell]) -> list[TableCell]:
    seen: set[tuple[int, str]] = set()
    result: list[TableCell] = []
    for cell in cells:
        key = (cell.column.index, cell.text.casefold())
        if key in seen:
            continue
        seen.add(key)
        result.append(cell)
    return result


def _dedupe_columns(columns: Iterable[TableColumn]) -> list[TableColumn]:
    seen: set[int] = set()
    result: list[TableColumn] = []
    for column in columns:
        if column.index in seen:
            continue
        seen.add(column.index)
        result.append(column)
    return result


def _dedupe_tables(tables: Iterable[NormalizedTable]) -> list[NormalizedTable]:
    seen: set[str] = set()
    result: list[NormalizedTable] = []
    for table in tables:
        key = table.rendered_text
        if key in seen:
            continue
        seen.add(key)
        result.append(table)
    return result


def _dedupe_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result
