from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class PdfProbeResult:
    file_path: str
    page_count: int

    total_chars: int
    total_blocks: int
    total_images: int

    avg_chars_per_page: float
    avg_blocks_per_page: float
    avg_images_per_page: float

    pages_with_text: int
    pages_without_text: int

    text_layer_ratio: float
    empty_text_ratio: float
    likely_scanned_ratio: float
    image_heavy_ratio: float

    avg_text_quality: float

    probe_detected_mode: str
    notes: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PageNode:
    page_index: int
    page_label: str
    text: str
    markdown: str
    source_mode: str
    has_ocr: bool = False
    has_table: bool = False
    block_ids: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class BlockNode:
    block_id: str
    page_index: int
    block_type: str
    text: str
    markdown: str
    reading_order: int
    bbox: tuple[float, float, float, float] | None = None
    level: int | None = None
    item_number: str | None = None
    parent_block_id: str | None = None
    heading_path: list[str] = field(default_factory=list)
    source_mode: str = "text"
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class TableCell:
    table_id: str | None
    page: int | None
    row_index: int
    col_index: int
    row_span: int = 1
    col_span: int = 1
    text: str = ""
    bbox: tuple[float, float, float, float] | None = None
    grid_bbox: tuple[float, float, float, float] | None = None
    confidence: float | None = None
    row_header: str | None = None
    col_header: str | None = None
    is_header: bool | None = None
    source_words: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TableCell":
        payload = dict(value)
        if "row" in payload and "row_index" not in payload:
            payload["row_index"] = payload.pop("row")
        if "col" in payload and "col_index" not in payload:
            payload["col_index"] = payload.pop("col")
        payload.setdefault("table_id", None)
        payload.setdefault("page", None)
        allowed = set(cls.__dataclass_fields__)
        return cls(**{key: payload[key] for key in payload if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["row"] = payload.pop("row_index")
        payload["col"] = payload.pop("col_index")
        return payload


@dataclass
class TableBlock:
    block_id: str
    table_id: str
    page: int | None
    bbox: tuple[float, float, float, float] | None = None
    caption: str | None = None
    cells: list[TableCell] = field(default_factory=list)
    csv: str | None = None
    markdown: str | None = None
    html: str | None = None
    source: str = "default"
    extraction_trace: dict[str, Any] = field(default_factory=dict)
    citation_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_block_node(cls, block: BlockNode) -> "TableBlock":
        meta = dict(block.meta or {})
        cells = [TableCell.from_dict(cell) for cell in meta.get("table_cells", []) if isinstance(cell, dict)]
        table_id = str(meta.get("table_id") or f"page_{block.page_index + 1}_{block.block_id}")
        return cls(
            block_id=block.block_id,
            table_id=table_id,
            page=block.page_index + 1,
            bbox=block.bbox,
            caption=meta.get("caption") or meta.get("table_caption"),
            cells=cells,
            csv=meta.get("table_csv"),
            markdown=meta.get("table_markdown") or block.markdown,
            html=meta.get("table_html"),
            source=str(meta.get("table_backend") or meta.get("backend") or block.source_mode or "default"),
            extraction_trace=dict(meta.get("extraction_trace") or {}),
            citation_metadata=dict(meta.get("citation_metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChunkNode:
    chunk_id: str
    chunk_index: int
    text: str
    markdown: str
    heading_path: list[str] = field(default_factory=list)
    page_start: int = 0
    page_end: int = 0
    page_indices: list[int] = field(default_factory=list)
    block_ids: list[str] = field(default_factory=list)
    block_types: list[str] = field(default_factory=list)
    source_mode: str = "text"
    meta: dict[str, Any] = field(default_factory=dict)


def to_dict(obj: Any) -> dict[str, Any]:
    return asdict(obj)
