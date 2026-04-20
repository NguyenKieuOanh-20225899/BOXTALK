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
