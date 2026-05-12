from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


BenchmarkMode = Literal["text", "layout", "table", "ocr", "all"]


@dataclass(slots=True)
class LayoutRegion:
    label: str
    bbox: tuple[float, float, float, float]
    text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class IngestGroundTruth:
    text: str | None = None
    ordered_text: list[str] = field(default_factory=list)
    layout_regions: list[LayoutRegion] = field(default_factory=list)
    table_regions: list[LayoutRegion] = field(default_factory=list)
    table_cells: list[dict[str, Any]] = field(default_factory=list)
    table_html: str | None = None
    table_csv: str | None = None
    form_fields: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["layout_regions"] = [region.to_dict() for region in self.layout_regions]
        payload["table_regions"] = [region.to_dict() for region in self.table_regions]
        return payload


@dataclass(slots=True)
class IngestBenchmarkSample:
    doc_id: str
    pdf_path: Path | None = None
    image_path: Path | None = None
    ground_truth: IngestGroundTruth = field(default_factory=IngestGroundTruth)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pdf_path"] = str(self.pdf_path) if self.pdf_path else None
        payload["image_path"] = str(self.image_path) if self.image_path else None
        payload["ground_truth"] = self.ground_truth.to_dict()
        return payload


@dataclass(slots=True)
class IngestPrediction:
    text: str = ""
    ordered_text: list[str] = field(default_factory=list)
    layout_regions: list[LayoutRegion] = field(default_factory=list)
    table_regions: list[LayoutRegion] = field(default_factory=list)
    table_cells: list[dict[str, Any]] = field(default_factory=list)
    backend: str = "unknown"
    latency_sec: float = 0.0
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["layout_regions"] = [region.to_dict() for region in self.layout_regions]
        payload["table_regions"] = [region.to_dict() for region in self.table_regions]
        return payload

