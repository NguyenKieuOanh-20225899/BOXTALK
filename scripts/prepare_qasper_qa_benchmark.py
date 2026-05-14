from __future__ import annotations

import argparse
import json
import random
import re
import sys
import tarfile
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


QASPER_SOURCE_NAME = "qasper"
DEFAULT_OUTPUT_DIR = Path("data/benchmarks/qasper_qa")
TRAIN_DEV_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
TEST_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz"
SPLIT_TO_FILE = {
    "train": "qasper-train-v0.3.json",
    "validation": "qasper-dev-v0.3.json",
    "dev": "qasper-dev-v0.3.json",
    "test": "qasper-test-v0.3.json",
}

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
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a small QASPER scientific QA benchmark for BOXTALK retrieval + grounded QA. "
            "The HF source downloads the official QASPER JSON tarball directly, so it does not "
            "require the datasets package."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", choices=["train", "validation", "dev", "test"], default="validation")
    parser.add_argument("--limit", type=int, default=100, help="Number of QA cases to emit.")
    parser.add_argument("--source", choices=["hf", "json"], default="hf")
    parser.add_argument("--input-file", type=Path, default=None, help="Local QASPER JSON file for --source json.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def prepare_qasper_qa(
    *,
    output_dir: Path,
    split: str = "validation",
    limit: int | None = 100,
    source: str = "hf",
    input_file: Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    payload, source_detail = load_qasper_payload(
        output_dir=output_dir,
        split=split,
        source=source,
        input_file=input_file,
    )
    papers = list(iter_papers(payload))
    if not papers:
        raise ValueError("No QASPER papers found in the input payload.")

    chunks_by_paper: dict[str, list[dict[str, Any]]] = {}
    for paper in papers:
        paper_id = paper_id_for(paper)
        chunks_by_paper[paper_id] = build_chunks_for_paper(paper, paper_id=paper_id)

    candidates: list[dict[str, Any]] = []
    for paper in papers:
        paper_id = paper_id_for(paper)
        title = normalize_space(str(paper.get("title") or ""))
        for qa_index, qa in enumerate(iter_qas(paper)):
            query = build_query_case(
                qa,
                paper_id=paper_id,
                title=title,
                split=split,
                qa_index=qa_index,
                chunks=chunks_by_paper.get(paper_id, []),
            )
            if query is not None:
                candidates.append(query)

    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected_queries = candidates[:limit] if limit is not None else candidates
    selected_paper_ids = {str(query["metadata"]["paper_id"]) for query in selected_queries}
    selected_chunks = [
        chunk
        for paper_id in sorted(selected_paper_ids)
        for chunk in chunks_by_paper.get(paper_id, [])
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = output_dir / "qasper.jsonl"
    queries_path = output_dir / "queries.jsonl"
    manifest_path = output_dir / "manifest.json"
    readme_path = output_dir / "README.md"

    write_jsonl(selected_chunks, chunks_path)
    write_jsonl(selected_queries, queries_path)

    answerable_count = sum(1 for query in selected_queries if query.get("should_answer"))
    unanswerable_count = len(selected_queries) - answerable_count
    evidence_mapped_count = sum(1 for query in selected_queries if query.get("expected_chunk_ids"))
    manifest = {
        "benchmark": "qasper_qa",
        "source": "QASPER",
        "source_detail": source_detail,
        "split": "validation" if split == "dev" else split,
        "seed": seed,
        "paper_count": len(selected_paper_ids),
        "chunk_count": len(selected_chunks),
        "query_count": len(selected_queries),
        "answerable_count": answerable_count,
        "unanswerable_count": unanswerable_count,
        "evidence_mapped_count": evidence_mapped_count,
        "chunks_jsonl": str(chunks_path),
        "queries_jsonl": str(queries_path),
        "notes": [
            "QASPER is a paper-text benchmark, not a PDF-page benchmark; citations use pseudo pages/sections.",
            "Multiple answer annotations are preserved in gold_answers.",
            "Evidence text is mapped to the best-overlap chunk when possible.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    readme_path.write_text(render_readme(manifest), encoding="utf-8")
    return manifest


def load_qasper_payload(
    *,
    output_dir: Path,
    split: str,
    source: str,
    input_file: Path | None,
) -> tuple[Any, str]:
    if source == "json":
        if input_file is None:
            raise ValueError("Pass --input-file when using --source json.")
        if not input_file.exists():
            raise FileNotFoundError(f"QASPER JSON file not found: {input_file}")
        return json.loads(input_file.read_text(encoding="utf-8")), str(input_file)

    filename = SPLIT_TO_FILE.get(split, SPLIT_TO_FILE["validation"])
    archive_url = TEST_URL if filename == SPLIT_TO_FILE["test"] else TRAIN_DEV_URL
    archive_path = output_dir / "raw" / Path(archive_url).name
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        try:
            urllib.request.urlretrieve(archive_url, archive_path)  # noqa: S310 - public dataset URL.
        except Exception as exc:  # pragma: no cover - network failure depends on environment.
            raise RuntimeError(
                "Could not download QASPER from the official public URL. "
                "Download the QASPER JSON manually and rerun with "
                "--source json --input-file <path-to-qasper-*-v0.3.json>."
            ) from exc

    with tarfile.open(archive_path, "r:gz") as tar:
        member = next((item for item in tar.getmembers() if Path(item.name).name == filename), None)
        if member is None:
            available = ", ".join(Path(item.name).name for item in tar.getmembers()[:20])
            raise FileNotFoundError(f"{filename} not found in {archive_path}. Available: {available}")
        extracted = tar.extractfile(member)
        if extracted is None:
            raise FileNotFoundError(f"Could not read {filename} from {archive_path}")
        return json.loads(extracted.read().decode("utf-8")), f"{archive_url}::{filename}"


def iter_papers(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    if not isinstance(payload, dict):
        return
    if isinstance(payload.get("papers"), list):
        for item in payload["papers"]:
            if isinstance(item, dict):
                yield item
        return
    if looks_like_paper(payload):
        yield payload
        return
    for key, value in payload.items():
        if isinstance(value, dict):
            paper = dict(value)
            paper.setdefault("paper_id", key)
            yield paper


def looks_like_paper(value: dict[str, Any]) -> bool:
    return any(key in value for key in ("qas", "questions")) and any(
        key in value for key in ("full_text", "abstract", "title")
    )


def iter_qas(paper: dict[str, Any]) -> Iterable[dict[str, Any]]:
    qas = paper.get("qas") or paper.get("questions") or []
    if isinstance(qas, list):
        for qa in qas:
            if isinstance(qa, dict):
                yield qa


def build_chunks_for_paper(paper: dict[str, Any], *, paper_id: str) -> list[dict[str, Any]]:
    title = normalize_space(str(paper.get("title") or ""))
    rows: list[dict[str, Any]] = []
    order = 0

    abstract = normalize_text_value(paper.get("abstract"))
    if abstract:
        rows.append(make_chunk(paper_id, title, "Abstract", abstract, order=order, pseudo_page=1))
        order += 1

    for section_index, section in enumerate(iter_sections(paper), start=1):
        section_name = normalize_space(str(section.get("section_name") or section.get("heading") or "Section"))
        paragraphs = list(iter_paragraphs(section))
        for paragraph_index, paragraph in enumerate(paragraphs):
            text = normalize_space(paragraph)
            if not text:
                continue
            rows.append(
                make_chunk(
                    paper_id,
                    title,
                    section_name,
                    text,
                    order=order,
                    pseudo_page=section_index + 1,
                    paragraph_index=paragraph_index,
                )
            )
            order += 1

    if not rows and title:
        rows.append(make_chunk(paper_id, title, "Title", title, order=0, pseudo_page=1))
    return rows


def iter_sections(paper: dict[str, Any]) -> Iterable[dict[str, Any]]:
    full_text = paper.get("full_text") or []
    if isinstance(full_text, dict):
        for key, value in full_text.items():
            yield {"section_name": key, "paragraphs": value}
    elif isinstance(full_text, list):
        for item in full_text:
            if isinstance(item, dict):
                yield item
            elif isinstance(item, str):
                yield {"section_name": "Full text", "paragraphs": [item]}


def iter_paragraphs(section: dict[str, Any]) -> Iterable[str]:
    paragraphs = section.get("paragraphs") or section.get("text") or []
    if isinstance(paragraphs, str):
        yield paragraphs
        return
    if isinstance(paragraphs, list):
        for paragraph in paragraphs:
            if isinstance(paragraph, str):
                yield paragraph
            elif isinstance(paragraph, dict):
                yield normalize_text_value(paragraph.get("text") or paragraph.get("paragraph") or paragraph)


def make_chunk(
    paper_id: str,
    title: str,
    section: str,
    text: str,
    *,
    order: int,
    pseudo_page: int,
    paragraph_index: int | None = None,
) -> dict[str, Any]:
    section_clean = normalize_space(section or "Section")
    chunk_id = f"{QASPER_SOURCE_NAME}:{paper_id}:{order:04d}"
    heading_path = [part for part in (title, section_clean) if part]
    chunk_text = "\n".join(part for part in (title, section_clean, text) if part).strip()
    return {
        "chunk_id": chunk_id,
        "id": chunk_id,
        "doc_id": paper_id,
        "source_name": QASPER_SOURCE_NAME,
        "title": title,
        "section": section_clean,
        "heading_path": heading_path,
        "page": pseudo_page,
        "text": chunk_text,
        "block_type": "paragraph",
        "order": order,
        "metadata": {
            "dataset": "qasper",
            "paper_id": paper_id,
            "section_name": section_clean,
            "paragraph_index": paragraph_index,
            "evidence_source": "qasper_full_text",
            "pseudo_page": pseudo_page,
        },
    }


def build_query_case(
    qa: dict[str, Any],
    *,
    paper_id: str,
    title: str,
    split: str,
    qa_index: int,
    chunks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    question = normalize_space(str(qa.get("question") or qa.get("query") or ""))
    if not question:
        return None

    gold_answers: list[str] = []
    evidence_texts: list[str] = []
    answer_types: list[str] = []
    annotation_count = 0
    unanswerable_votes = 0
    answer_records = qa.get("answers") or qa.get("answer") or []
    if isinstance(answer_records, dict):
        answer_records = [answer_records]
    if not isinstance(answer_records, list):
        answer_records = []

    for record in answer_records:
        answer = normalize_answer_record(record)
        if not answer:
            continue
        annotation_count += 1
        if bool(answer.get("unanswerable")):
            unanswerable_votes += 1
            answer_types.append("unanswerable")
        texts, answer_type = answer_texts(answer)
        answer_types.append(answer_type)
        gold_answers.extend(texts)
        evidence_texts.extend(string_list(answer.get("evidence")))

    gold_answers = dedupe_normalized(gold_answers)
    evidence_texts = dedupe_normalized(evidence_texts)
    answerable = bool(gold_answers)
    if annotation_count and unanswerable_votes == annotation_count:
        answerable = False

    evidence_chunk_ids = map_evidence_to_chunks(evidence_texts, chunks)
    match_text = evidence_texts[0] if evidence_texts else (gold_answers[0] if gold_answers else "")
    question_id = str(qa.get("question_id") or qa.get("id") or f"{paper_id}_{qa_index}")
    answer_type = choose_answer_type(answer_types)
    return {
        "id": f"qasper_{question_id}",
        "query_id": f"qasper_{question_id}",
        "question": question,
        "query_type": "factoid",
        "gold_answer": gold_answers[0] if gold_answers else "",
        "gold_answers": gold_answers,
        "match_text": match_text,
        "expected_chunk_ids": evidence_chunk_ids,
        "gold_evidence_texts": evidence_texts,
        "source_name": QASPER_SOURCE_NAME,
        "benchmark_family": "qasper",
        "document_type": "scientific_paper_text",
        "evidence_type": "scientific_paper_section",
        "should_answer": answerable,
        "answerable": answerable,
        "expected_decision": "answer" if answerable else "abstain",
        "metadata": {
            "dataset": "qasper",
            "paper_id": paper_id,
            "paper_title": title,
            "question_id": question_id,
            "answer_type": answer_type,
            "annotation_count": annotation_count,
            "unanswerable_votes": unanswerable_votes,
            "split": "validation" if split == "dev" else split,
        },
    }


def normalize_answer_record(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    answer = record.get("answer") if isinstance(record.get("answer"), dict) else record
    return dict(answer) if isinstance(answer, dict) else {}


def answer_texts(answer: dict[str, Any]) -> tuple[list[str], str]:
    if bool(answer.get("unanswerable")):
        return [], "unanswerable"

    texts: list[str] = []
    answer_type = "free_form"
    yes_no = answer.get("yes_no")
    if yes_no is not None and str(yes_no).strip().lower() not in {"", "none", "null"}:
        if isinstance(yes_no, bool):
            texts.append("yes" if yes_no else "no")
        else:
            value = str(yes_no).strip().lower()
            if value in {"yes", "true", "1"}:
                texts.append("yes")
            elif value in {"no", "false", "0"}:
                texts.append("no")
        answer_type = "yes_no"

    spans = string_list(answer.get("extractive_spans"))
    if spans:
        texts.extend(spans)
        if len(spans) > 1:
            texts.append("; ".join(spans))
        answer_type = "extractive"

    free_form = normalize_space(str(answer.get("free_form_answer") or answer.get("freeform_answer") or ""))
    if free_form:
        texts.append(free_form)
        if answer_type == "free_form":
            answer_type = "free_form"

    return dedupe_normalized(texts), answer_type


def map_evidence_to_chunks(evidence_texts: list[str], chunks: list[dict[str, Any]]) -> list[str]:
    mapped: list[str] = []
    for evidence in evidence_texts:
        scored = [
            (evidence_overlap_score(evidence, str(chunk.get("text") or "")), str(chunk.get("chunk_id") or ""))
            for chunk in chunks
        ]
        scored = [(score, chunk_id) for score, chunk_id in scored if chunk_id]
        if not scored:
            continue
        scored.sort(reverse=True)
        best_score, best_chunk_id = scored[0]
        if best_score >= 0.55 or contains_normalized(str(chunks_by_id(chunks).get(best_chunk_id, {}).get("text") or ""), evidence):
            mapped.append(best_chunk_id)
    return dedupe_normalized(mapped)


def chunks_by_id(chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(chunk.get("chunk_id") or ""): chunk for chunk in chunks}


def evidence_overlap_score(evidence: str, chunk_text: str) -> float:
    evidence_clean = normalize_space(evidence)
    chunk_clean = normalize_space(chunk_text)
    if not evidence_clean or not chunk_clean:
        return 0.0
    if contains_normalized(chunk_clean, evidence_clean):
        return 1.0
    evidence_tokens = content_tokens(evidence_clean)
    chunk_tokens = content_tokens(chunk_clean)
    if not evidence_tokens or not chunk_tokens:
        return 0.0
    return len(evidence_tokens & chunk_tokens) / len(evidence_tokens)


def paper_id_for(paper: dict[str, Any]) -> str:
    value = paper.get("paper_id") or paper.get("id") or paper.get("paperId") or paper.get("arxiv_id")
    if value:
        return sanitize_id(str(value))
    title = normalize_space(str(paper.get("title") or "untitled"))
    return sanitize_id(title[:80] or "untitled")


def choose_answer_type(answer_types: list[str]) -> str:
    for preferred in ("yes_no", "extractive", "free_form", "unanswerable"):
        if preferred in answer_types:
            return preferred
    return "unknown"


def normalize_text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return normalize_space(value)
    if isinstance(value, list):
        return normalize_space(" ".join(normalize_text_value(item) for item in value))
    if isinstance(value, dict):
        if "text" in value:
            return normalize_text_value(value["text"])
        return normalize_space(" ".join(normalize_text_value(item) for item in value.values()))
    return normalize_space(str(value))


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [normalize_space(value)] if normalize_space(value) else []
    if isinstance(value, dict):
        return [normalize_text_value(value)]
    if isinstance(value, Iterable):
        result: list[str] = []
        for item in value:
            text = normalize_text_value(item)
            if text:
                result.append(text)
        return result
    text = normalize_space(str(value))
    return [text] if text else []


def dedupe_normalized(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = normalize_space(value)
        if not text:
            continue
        key = normalize_for_key(text)
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", normalize_for_key(text))
        if len(token) > 1 and token not in STOPWORDS
    }


def contains_normalized(haystack: str, needle: str) -> bool:
    needle_clean = normalize_for_key(needle)
    return bool(needle_clean and needle_clean in normalize_for_key(haystack))


def normalize_for_key(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().casefold()


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def sanitize_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "unknown"


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_readme(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# QASPER QA Benchmark",
            "",
            "This directory contains a BOXTALK-ready subset of QASPER for natural scientific QA.",
            "",
            "- `qasper.jsonl`: retrieval chunks built from paper title, abstract and section paragraphs.",
            "- `queries.jsonl`: QA cases with multiple gold answers and evidence text.",
            "- Citations use section/chunk ids because QASPER is not distributed as PDF pages.",
            "",
            "Example:",
            "",
            "```powershell",
            ".\\.venv-gpu\\Scripts\\python.exe scripts\\build_retrieval_index.py --chunks-jsonl data\\benchmarks\\qasper_qa\\qasper.jsonl --output-dir results\\retrieval_index\\qasper_qa_minilm --dense-preset minilm --dense-device cuda",
            ".\\.venv-gpu\\Scripts\\python.exe scripts\\benchmark_qa.py --index-dir results\\retrieval_index\\qasper_qa_minilm --queries data\\benchmarks\\qasper_qa\\queries.jsonl --output-dir results\\qa_benchmark\\qasper_qa_minilm --config routed_grounded --no-warmup",
            "```",
            "",
            f"Source detail: {manifest['source_detail']}",
            f"Split: {manifest['split']}",
            f"Paper count: {manifest['paper_count']}",
            f"Chunk count: {manifest['chunk_count']}",
            f"Query count: {manifest['query_count']}",
            f"Answerable: {manifest['answerable_count']}",
            f"Unanswerable: {manifest['unanswerable_count']}",
            f"Evidence mapped to chunk: {manifest['evidence_mapped_count']}",
        ]
    )


def main() -> None:
    args = parse_args()
    manifest = prepare_qasper_qa(
        output_dir=args.output_dir,
        split=args.split,
        limit=args.limit,
        source=args.source,
        input_file=args.input_file,
        seed=args.seed,
    )
    print(args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
