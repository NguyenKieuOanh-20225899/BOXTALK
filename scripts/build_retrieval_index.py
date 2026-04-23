from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ingest.pipeline import ingest_pdf
from app.retrieval.colbert_retriever import DEFAULT_COLBERT_MODEL_NAME
from app.retrieval.dense_retriever import DenseRetriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.schemas import DocumentChunkRef, coerce_chunk_ref


RESULTS_ROOT = Path("results/retrieval_index")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BM25+dense retrieval index from ingest chunks.")
    parser.add_argument("--pdf", action="append", default=[], help="PDF path. Pass multiple times.")
    parser.add_argument("--chunks-json", action="append", default=[], help="JSON file containing chunks or {chunks: [...]}.")
    parser.add_argument("--chunks-jsonl", action="append", default=[], help="JSONL file containing one chunk per line.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional output directory.")
    parser.add_argument("--dense-model", default=None, help="Optional sentence-transformers model name.")
    parser.add_argument(
        "--dense-preset",
        choices=["minilm", "multilingual-minilm", "contriever", "dpr-single-nq", "dpr-multiset"],
        default="minilm",
        help="Named dense baseline preset.",
    )
    parser.add_argument(
        "--dense-backend",
        choices=["sentence-transformers", "transformers", "dpr"],
        default=None,
        help="Override dense encoder backend.",
    )
    parser.add_argument("--dense-query-model", default=None, help="Optional DPR query encoder model.")
    parser.add_argument("--dense-passage-model", default=None, help="Optional DPR passage encoder model.")
    parser.add_argument("--dense-batch-size", type=int, default=32, help="Dense encoding batch size.")
    parser.add_argument("--dense-device", default=None, help="Dense model device, e.g. cpu, cuda.")
    parser.add_argument("--skip-dense", action="store_true", help="Build BM25-only index for fast debugging.")
    parser.add_argument("--build-colbert", action="store_true", help="Build a ColBERT late-interaction index.")
    parser.add_argument("--colbert-model", default=DEFAULT_COLBERT_MODEL_NAME, help="ColBERT HF model name.")
    return parser.parse_args()


def adapt_chunk(chunk: Any, *, source_name: str, doc_id: str | None = None) -> DocumentChunkRef:
    ref = coerce_chunk_ref(chunk, source_name=source_name, doc_id=doc_id or source_name)
    chunk_id = ref.chunk_id
    if source_name and not chunk_id.startswith(f"{source_name}:"):
        chunk_id = f"{source_name}:{chunk_id}"
    metadata = {
        **ref.metadata,
        "source_name": ref.source_name or source_name,
        "doc_id": ref.doc_id or doc_id or source_name,
    }
    return replace(
        ref,
        chunk_id=chunk_id,
        source_name=ref.source_name or source_name,
        doc_id=ref.doc_id or doc_id or source_name,
        metadata=metadata,
    )


def load_chunks_from_pdf(pdf_path: Path) -> tuple[list[DocumentChunkRef], dict[str, Any]]:
    report = ingest_pdf(pdf_path)
    chunks = [
        adapt_chunk(chunk, source_name=pdf_path.name, doc_id=pdf_path.name)
        for chunk in report.get("chunks", [])
    ]
    summary = {
        "input_type": "pdf",
        "path": str(pdf_path),
        "source_name": pdf_path.name,
        "used_backend": report.get("used_backend"),
        "page_count": len(report.get("pages", [])),
        "block_count": len(report.get("blocks", [])),
        "chunk_count": len(chunks),
        "probe_mode": report.get("probe", {}).get("probe_detected_mode"),
    }
    return chunks, summary


def load_chunks_from_json(path: Path) -> tuple[list[DocumentChunkRef], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("chunks", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a list of chunks or an object with a chunks list")
    source_name = path.stem
    chunks = [adapt_chunk(row, source_name=source_name, doc_id=source_name) for row in rows]
    return chunks, {
        "input_type": "chunks_json",
        "path": str(path),
        "source_name": source_name,
        "chunk_count": len(chunks),
    }


def load_chunks_from_jsonl(path: Path) -> tuple[list[DocumentChunkRef], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    source_name = path.stem
    chunks = [adapt_chunk(row, source_name=source_name, doc_id=source_name) for row in rows]
    return chunks, {
        "input_type": "chunks_jsonl",
        "path": str(path),
        "source_name": source_name,
        "chunk_count": len(chunks),
    }


def main() -> None:
    args = parse_args()
    if not args.pdf and not args.chunks_json and not args.chunks_jsonl:
        raise SystemExit("Pass at least one --pdf, --chunks-json, or --chunks-jsonl input.")

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (RESULTS_ROOT / timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_chunks: list[DocumentChunkRef] = []
    source_summaries: list[dict[str, Any]] = []

    for pdf_arg in args.pdf:
        chunks, summary = load_chunks_from_pdf(Path(pdf_arg))
        all_chunks.extend(chunks)
        source_summaries.append(summary)

    for json_arg in args.chunks_json:
        chunks, summary = load_chunks_from_json(Path(json_arg))
        all_chunks.extend(chunks)
        source_summaries.append(summary)

    for jsonl_arg in args.chunks_jsonl:
        chunks, summary = load_chunks_from_jsonl(Path(jsonl_arg))
        all_chunks.extend(chunks)
        source_summaries.append(summary)

    retriever = HybridRetriever(
        all_chunks,
        model_name=args.dense_model,
        dense_backend=args.dense_backend,
        dense_preset=args.dense_preset,
        dense_query_model_name=args.dense_query_model,
        dense_passage_model_name=args.dense_passage_model,
        build_dense=False,
        build_colbert=False,
        colbert_model_name=args.colbert_model,
    )
    if not args.skip_dense:
        retriever.dense = DenseRetriever(
            all_chunks,
            model_name=args.dense_model,
            backend=args.dense_backend,
            preset=args.dense_preset,
            query_model_name=args.dense_query_model,
            passage_model_name=args.dense_passage_model,
            batch_size=args.dense_batch_size,
            device=args.dense_device,
        )
        retriever.dense.build()
    if args.build_colbert and retriever.colbert is not None:
        retriever.colbert.build()
    retriever.save(output_dir)

    summary = {
        "timestamp_utc": timestamp,
        "source_count": len(source_summaries),
        "chunk_count": len(all_chunks),
        "bm25_index": "bm25_config.json",
        "dense_built": not args.skip_dense,
        "dense_model_name": retriever.dense.model_name,
        "dense_backend": retriever.dense.backend_name,
        "dense_preset": args.dense_preset,
        "colbert_built": args.build_colbert,
        "colbert_model_name": retriever.colbert.model_name if retriever.colbert else None,
        "sources": source_summaries,
        "sample_chunks": [chunk.to_dict() for chunk in all_chunks[:5]],
    }
    (output_dir / "index_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(output_dir)


if __name__ == "__main__":
    main()
