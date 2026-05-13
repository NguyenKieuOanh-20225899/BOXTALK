from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_BEIR_DIR = Path("data/beir/scifact")
DEFAULT_OUTPUT_DIR = Path("data/benchmarks/scifact_qa")
SCIFACT_SOURCE_NAME = "scifact"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "with",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert BEIR SciFact into a grounded QA benchmark with expected citation ids. "
            "The official qrels provide citation/evidence labels; answer text is derived "
            "from the relevant abstract sentence."
        )
    )
    parser.add_argument("--beir-dir", type=Path, default=DEFAULT_BEIR_DIR, help="BEIR SciFact directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output benchmark directory.")
    parser.add_argument("--split", choices=["train", "test"], default="test", help="Qrels split to convert.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of queries to emit.")
    parser.add_argument("--seed", type=int, default=13, help="Reserved for reproducible future sampling.")
    return parser.parse_args()


def prepare_scifact_qa(
    *,
    beir_dir: Path,
    output_dir: Path,
    split: str = "test",
    limit: int | None = None,
) -> dict[str, Any]:
    corpus = load_jsonl_by_id(beir_dir / "corpus.jsonl")
    queries = load_jsonl_by_id(beir_dir / "queries.jsonl")
    qrels = load_qrels(beir_dir / "qrels" / f"{split}.tsv")

    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = output_dir / "scifact.jsonl"
    queries_path = output_dir / f"queries_{split}.jsonl"
    manifest_path = output_dir / "manifest.json"
    readme_path = output_dir / "README.md"

    write_chunks(corpus, chunks_path)

    cases: list[dict[str, Any]] = []
    for query_id in sorted(qrels, key=lambda value: int(value) if value.isdigit() else value):
        if limit is not None and len(cases) >= limit:
            break
        query = queries.get(query_id)
        if not query:
            continue
        relevant_ids = [doc_id for doc_id, score in qrels[query_id] if score > 0 and doc_id in corpus]
        if not relevant_ids:
            continue
        claim = normalize_space(str(query.get("text") or ""))
        evidence_doc_id, evidence_sentence = choose_gold_evidence_sentence(
            claim=claim,
            query_metadata=query.get("metadata", {}),
            relevant_ids=relevant_ids,
            corpus=corpus,
        )
        cases.append(
            {
                "id": f"scifact_{query_id}",
                "question": f"What scientific evidence addresses this claim: {claim}",
                "query_type": "factoid",
                "gold_answer": evidence_sentence,
                "match_text": evidence_sentence,
                "expected_chunk_ids": [chunk_id_for(doc_id) for doc_id in relevant_ids],
                "gold_sections": [str(corpus[doc_id].get("title") or "") for doc_id in relevant_ids],
                "source_name": SCIFACT_SOURCE_NAME,
                "benchmark_family": "beir_scifact",
                "document_type": "scientific_abstract",
                "evidence_type": "scientific_abstract",
                "should_answer": True,
                "metadata": {
                    "beir_query_id": query_id,
                    "claim": claim,
                    "primary_evidence_doc_id": evidence_doc_id,
                    "relevant_doc_ids": relevant_ids,
                    "qrels_split": split,
                },
            }
        )

    write_jsonl(cases, queries_path)
    manifest = {
        "benchmark": "beir_scifact_qa",
        "source": "BEIR SciFact",
        "split": split,
        "corpus_size": len(corpus),
        "query_count": len(cases),
        "chunks_jsonl": str(chunks_path),
        "queries_jsonl": str(queries_path),
        "citation_gold_source": str(beir_dir / "qrels" / f"{split}.tsv"),
        "answer_gold_method": (
            "Evidence sentence from SciFact query metadata when available; otherwise the relevant "
            "abstract sentence with highest token overlap with the claim."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    readme_path.write_text(render_readme(manifest), encoding="utf-8")
    return manifest


def load_jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row.get("_id") or row.get("id") or "").strip()
            if row_id:
                rows[row_id] = row
    return rows


def load_qrels(path: Path) -> dict[str, list[tuple[str, int]]]:
    qrels: dict[str, list[tuple[str, int]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if line_no == 0 and line.lower().startswith("query-id"):
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            query_id, corpus_id, score_raw = parts[:3]
            try:
                score = int(float(score_raw))
            except ValueError:
                score = 0
            qrels[str(query_id)].append((str(corpus_id), score))
    return dict(qrels)


def write_chunks(corpus: dict[str, dict[str, Any]], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for doc_id in sorted(corpus, key=lambda value: int(value) if value.isdigit() else value):
        doc = corpus[doc_id]
        title = normalize_space(str(doc.get("title") or ""))
        abstract = normalize_space(str(doc.get("text") or ""))
        text = "\n".join(part for part in (title, abstract) if part).strip()
        rows.append(
            {
                "chunk_id": chunk_id_for(doc_id),
                "text": text,
                "doc_id": doc_id,
                "source_name": SCIFACT_SOURCE_NAME,
                "title": title,
                "section": title,
                "block_type": "paragraph",
                "metadata": {
                    "benchmark": "beir_scifact",
                    "corpus_id": doc_id,
                    "title": title,
                },
            }
        )
    write_jsonl(rows, path)


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def choose_gold_evidence_sentence(
    *,
    claim: str,
    query_metadata: Any,
    relevant_ids: list[str],
    corpus: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    if isinstance(query_metadata, dict):
        for doc_id in relevant_ids:
            annotations = query_metadata.get(doc_id)
            if not annotations:
                continue
            sentences = split_sentences(str(corpus[doc_id].get("text") or ""))
            for annotation in annotations:
                if not isinstance(annotation, dict):
                    continue
                for sentence_idx in annotation.get("sentences", []) or []:
                    try:
                        idx = int(sentence_idx)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= idx < len(sentences):
                        return doc_id, sentences[idx]

    scored: list[tuple[float, str, str]] = []
    for doc_id in relevant_ids:
        title = normalize_space(str(corpus[doc_id].get("title") or ""))
        for sentence in split_sentences(str(corpus[doc_id].get("text") or "")):
            scored.append((overlap_score(claim, sentence), doc_id, sentence))
        if title:
            scored.append((overlap_score(claim, title) * 0.9, doc_id, title))
    if not scored:
        doc_id = relevant_ids[0]
        fallback = normalize_space(str(corpus[doc_id].get("title") or corpus[doc_id].get("text") or ""))
        return doc_id, fallback
    scored.sort(key=lambda item: (item[0], len(item[2])), reverse=True)
    _, doc_id, sentence = scored[0]
    return doc_id, sentence


def split_sentences(text: str) -> list[str]:
    normalized = normalize_space(text)
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", normalized)
    return [part.strip() for part in parts if part.strip()]


def overlap_score(left: str, right: str) -> float:
    left_tokens = content_tokens(left)
    right_tokens = content_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = left_tokens & right_tokens
    return (2.0 * len(overlap)) / (len(left_tokens) + len(right_tokens))


def content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def chunk_id_for(doc_id: str) -> str:
    return f"{SCIFACT_SOURCE_NAME}:{doc_id}"


def render_readme(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# SciFact QA Benchmark",
            "",
            "This directory contains a grounded QA conversion of BEIR SciFact.",
            "",
            "- `scifact.jsonl`: retrieval chunks built from SciFact abstracts.",
            f"- `{Path(manifest['queries_jsonl']).name}`: QA cases for the `{manifest['split']}` split.",
            "- `expected_chunk_ids` comes from the official BEIR SciFact qrels and is used to evaluate citation/evidence correctness.",
            "- `gold_answer` is a relevant abstract sentence. It is taken from SciFact query metadata when sentence-level evidence is available; otherwise it is selected by token overlap with the claim.",
            "",
            "Example:",
            "",
            "```powershell",
            ".\\.venv-gpu\\Scripts\\python.exe scripts\\build_retrieval_index.py --chunks-jsonl data\\benchmarks\\scifact_qa\\scifact.jsonl --output-dir results\\retrieval_index\\scifact_qa_minilm --dense-preset minilm --dense-device cuda",
            ".\\.venv-gpu\\Scripts\\python.exe scripts\\benchmark_qa.py --index-dir results\\retrieval_index\\scifact_qa_minilm --queries data\\benchmarks\\scifact_qa\\queries_test.jsonl --output-dir results\\qa_benchmark\\scifact_qa_minilm --config routed_grounded --no-warmup",
            "```",
            "",
            f"Corpus size: {manifest['corpus_size']}",
            f"Query count: {manifest['query_count']}",
        ]
    )


def main() -> None:
    args = parse_args()
    manifest = prepare_scifact_qa(
        beir_dir=args.beir_dir,
        output_dir=args.output_dir,
        split=args.split,
        limit=args.limit,
    )
    print(args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
