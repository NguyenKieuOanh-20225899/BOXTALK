from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Iterable, Literal, Mapping


RetrievalSource = Literal["bm25", "dense", "hybrid", "rerank", "colbert"]
CombinationStrategy = Literal["weighted_sum", "rrf"]


@dataclass(slots=True)
class DocumentChunkRef:
    """Canonical chunk reference used by retrieval components."""

    chunk_id: str
    text: str
    doc_id: str | None = None
    source_name: str | None = None
    page: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    page_indices: list[int] = field(default_factory=list)
    section: str | None = None
    title: str | None = None
    heading_path: list[str] = field(default_factory=list)
    block_type: str = "paragraph"
    order: int | None = None
    version: str | None = None
    date: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def heading_path_text(self) -> str:
        return " > ".join(part for part in self.heading_path if part)

    @property
    def primary_page(self) -> int | None:
        if self.page is not None:
            return self.page
        if self.page_start is not None:
            return self.page_start + 1
        if self.page_indices:
            return self.page_indices[0] + 1
        return None

    def searchable_text(self) -> str:
        """Text indexed by both sparse and dense retrievers."""

        parts = [
            self.title or "",
            self.section or "",
            self.heading_path_text,
            self.block_type or "",
            self.text or "",
        ]
        return "\n".join(part for part in parts if part).strip()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievalConfig:
    """Runtime retrieval knobs designed for future query-aware routing."""

    top_k: int = 5
    candidate_k: int = 50
    bm25_weight: float = 0.5
    dense_weight: float = 0.5
    rerank_top_n: int = 0
    combination: CombinationStrategy = "weighted_sum"
    rrf_k: int = 60
    use_rerank: bool = False
    block_type_filter: list[str] = field(default_factory=list)
    section_filter: list[str] = field(default_factory=list)
    doc_id_filter: list[str] = field(default_factory=list)
    source_name_filter: list[str] = field(default_factory=list)
    version_filter: list[str] = field(default_factory=list)
    date_filter: list[str] = field(default_factory=list)
    metadata_filters: dict[str, Any] = field(default_factory=dict)
    context_window: int = 0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "RetrievalConfig":
        if value is None:
            return cls()
        allowed = {field_name for field_name in cls.__dataclass_fields__}
        payload = {key: val for key, val in dict(value).items() if key in allowed}
        return cls(**payload)

    def candidate_limit(self) -> int:
        return max(self.top_k, self.candidate_k, self.rerank_top_n)

    def normalized_weights(self) -> tuple[float, float]:
        total = max(0.0, self.bm25_weight) + max(0.0, self.dense_weight)
        if total <= 0.0:
            return 0.5, 0.5
        return max(0.0, self.bm25_weight) / total, max(0.0, self.dense_weight) / total

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievedHit:
    """A retrieved chunk with normalized score and provenance."""

    chunk: DocumentChunkRef
    score: float
    source: RetrievalSource
    rank: int | None = None
    bm25_score: float = 0.0
    dense_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float | None = None
    source_scores: dict[str, float] = field(default_factory=dict)
    raw_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.final_score is None:
            self.final_score = self.score

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def page(self) -> int | None:
        return self.chunk.primary_page

    @property
    def section(self) -> str | None:
        return self.chunk.section

    @property
    def heading_path(self) -> list[str]:
        return self.chunk.heading_path

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def hybrid_score(self) -> float:
        """Compatibility score used by the starter RAG evidence layer."""

        return float(self.final_score if self.final_score is not None else self.score)

    @property
    def snippet(self) -> str:
        return make_snippet(self.chunk.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "doc_id": self.chunk.doc_id,
            "source_name": self.chunk.source_name,
            "score": self.score,
            "final_score": self.final_score,
            "source": self.source,
            "rank": self.rank,
            "bm25_score": self.bm25_score,
            "dense_score": self.dense_score,
            "rerank_score": self.rerank_score,
            "source_scores": dict(self.source_scores),
            "raw_scores": dict(self.raw_scores),
            "page": self.chunk.primary_page,
            "page_start": self.chunk.page_start,
            "page_end": self.chunk.page_end,
            "section": self.chunk.section,
            "title": self.chunk.title,
            "heading_path": list(self.chunk.heading_path),
            "block_type": self.chunk.block_type,
            "snippet": self.snippet,
            "text": self.chunk.text,
            "metadata": {**self.chunk.metadata, **self.metadata},
        }


@dataclass(slots=True)
class RetrievalResult:
    """Result envelope for benchmarkable retrieval calls."""

    query: str
    strategy: str
    hits: list[RetrievedHit]
    config: RetrievalConfig
    latency_ms: float
    retrieval_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "strategy": self.strategy,
            "latency_ms": self.latency_ms,
            "retrieval_count": self.retrieval_count,
            "config": self.config.to_dict(),
            "hits": [hit.to_dict() for hit in self.hits],
        }


@dataclass(slots=True)
class RankedScore:
    """Internal score carrier returned by first-stage retrievers."""

    index: int
    chunk: DocumentChunkRef
    score: float
    raw_score: float
    rank: int
    source: RetrievalSource


def coerce_chunk_ref(
    chunk: Any,
    *,
    doc_id: str | None = None,
    source_name: str | None = None,
) -> DocumentChunkRef:
    """Convert existing repo chunk shapes into the retrieval schema."""

    if isinstance(chunk, DocumentChunkRef):
        return chunk

    data = _object_to_mapping(chunk)
    metadata = dict(data.get("metadata") or data.get("meta") or {})
    input_source = data.get("source_name") or source_name or metadata.get("source_name")
    input_doc_id = data.get("doc_id") or doc_id or metadata.get("doc_id") or input_source
    heading_path = _normalize_heading_path(data.get("heading_path") or metadata.get("heading_path"))
    page_start = _optional_int(data.get("page_start", metadata.get("page_start")))
    page_end = _optional_int(data.get("page_end", metadata.get("page_end")))
    page_indices = _int_list(data.get("page_indices", metadata.get("page_indices", [])))
    page = _optional_int(data.get("page", metadata.get("page")))
    if page is None and page_start is not None:
        page = page_start + 1
    block_type = _primary_block_type(data, metadata)
    section = data.get("section") or metadata.get("section")
    if not section and heading_path:
        section = heading_path[-1]

    chunk_id = data.get("chunk_id") or metadata.get("chunk_id")
    if not chunk_id:
        raise ValueError("Retrieval chunks must include chunk_id")

    return DocumentChunkRef(
        chunk_id=str(chunk_id),
        text=str(data.get("text") or data.get("markdown") or ""),
        doc_id=str(input_doc_id) if input_doc_id else None,
        source_name=str(input_source) if input_source else None,
        page=page,
        page_start=page_start,
        page_end=page_end,
        page_indices=page_indices,
        section=str(section) if section else None,
        title=_optional_str(data.get("title") or metadata.get("title")),
        heading_path=heading_path,
        block_type=block_type,
        order=_optional_int(data.get("order", data.get("chunk_index"))),
        version=_optional_str(data.get("version") or metadata.get("version")),
        date=_optional_str(data.get("date") or metadata.get("date")),
        metadata=metadata,
    )


def coerce_chunk_refs(chunks: Iterable[Any]) -> list[DocumentChunkRef]:
    return [coerce_chunk_ref(chunk) for chunk in chunks]


def chunk_matches_config(chunk: DocumentChunkRef, config: RetrievalConfig | None) -> bool:
    if config is None:
        return True

    if config.block_type_filter and not _matches_any(chunk.block_type, config.block_type_filter):
        return False
    if config.section_filter and not _contains_any(chunk.section, config.section_filter):
        return False
    if config.doc_id_filter and not _matches_any(chunk.doc_id, config.doc_id_filter):
        return False
    if config.source_name_filter and not _matches_any(chunk.source_name, config.source_name_filter):
        return False
    if config.version_filter and not _matches_any(chunk.version, config.version_filter):
        return False
    if config.date_filter and not _matches_any(chunk.date, config.date_filter):
        return False

    for key, expected in config.metadata_filters.items():
        if not _metadata_matches(chunk.metadata.get(key), expected):
            return False

    return True


def make_snippet(text: str, *, max_chars: int = 320) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 1].rstrip() + "..."


def _object_to_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    attrs = [
        "chunk_id",
        "text",
        "markdown",
        "doc_id",
        "source_name",
        "page",
        "page_start",
        "page_end",
        "page_indices",
        "section",
        "title",
        "heading_path",
        "block_type",
        "block_types",
        "order",
        "chunk_index",
        "version",
        "date",
        "metadata",
        "meta",
    ]
    return {attr: getattr(value, attr) for attr in attrs if hasattr(value, attr)}


def _normalize_heading_path(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(" > ") if part.strip()]
    if isinstance(value, Iterable):
        return [str(part).strip() for part in value if str(part).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _primary_block_type(data: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    if metadata.get("is_table_chunk"):
        return "table"
    block_types = data.get("block_types") or metadata.get("block_types") or []
    if block_types:
        normalized = [str(item) for item in block_types]
        if any(item == "table" for item in normalized):
            return "table"
        for preferred in ("paragraph", "list_item", "list", "metadata", "caption", "figure"):
            if preferred in normalized:
                return preferred
        return normalized[0]
    block_type = data.get("block_type") or metadata.get("block_type")
    if block_type:
        return str(block_type)
    return "paragraph"


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return []
    result: list[int] = []
    for item in value:
        parsed = _optional_int(item)
        if parsed is not None:
            result.append(parsed)
    return result


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _matches_any(value: str | None, expected_values: list[str]) -> bool:
    if value is None:
        return False
    value_norm = value.strip().lower()
    return any(value_norm == expected.strip().lower() for expected in expected_values)


def _contains_any(value: str | None, expected_values: list[str]) -> bool:
    if value is None:
        return False
    value_norm = value.strip().lower()
    return any(expected.strip().lower() in value_norm for expected in expected_values)


def _metadata_matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return any(_metadata_matches(actual, item) for item in expected)
    if isinstance(actual, str) or isinstance(expected, str):
        return str(actual).strip().lower() == str(expected).strip().lower()
    return actual == expected
