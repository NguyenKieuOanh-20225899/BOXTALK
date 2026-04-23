from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from app.retrieval.schemas import DocumentChunkRef, coerce_chunk_refs


INDEX_FORMAT_VERSION = 1


class RetrievalIndexStore:
    """Small filesystem store for retrieval corpus and index metadata."""

    CORPUS_FILE = "corpus.jsonl"
    MANIFEST_FILE = "index_manifest.json"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @property
    def corpus_path(self) -> Path:
        return self.root / self.CORPUS_FILE

    @property
    def manifest_path(self) -> Path:
        return self.root / self.MANIFEST_FILE

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def write_corpus(self, chunks: Iterable[Any]) -> list[DocumentChunkRef]:
        self.ensure()
        chunk_refs = coerce_chunk_refs(chunks)
        with self.corpus_path.open("w", encoding="utf-8") as handle:
            for chunk in chunk_refs:
                handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        return chunk_refs

    def read_corpus(self) -> list[DocumentChunkRef]:
        chunks: list[DocumentChunkRef] = []
        with self.corpus_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                chunks.append(DocumentChunkRef(**json.loads(line)))
        return chunks

    def write_manifest(self, payload: dict[str, Any]) -> None:
        self.ensure()
        manifest = {
            "index_format_version": INDEX_FORMAT_VERSION,
            **payload,
        }
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def read_manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))
