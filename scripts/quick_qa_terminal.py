from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

from app.qa.pipeline import GroundedQAPipeline
from app.qa.llm_explainer import make_llm_explainer_from_env
from app.qa.router import QueryRouter
from app.retrieval.route_planner import QueryRetrievalPlan
from app.retrieval.reranker import make_reranker
from app.retrieval.schemas import RetrievalConfig
from app.retrieval.service import RetrievalService


DEFAULT_INDEX_DIR = Path("results/retrieval_index/qcdt_2025_5445_constraint_table_reconstruction")


class FixedRetrievalPlanner:
    def __init__(self, *, strategy: str, config: RetrievalConfig) -> None:
        self.strategy = strategy
        self.config = config

    def plan(self, query_type: str, question: str) -> QueryRetrievalPlan:
        _ = query_type, question
        return QueryRetrievalPlan(
            strategy=self.strategy,
            config=self.config,
            reason="manual CLI override",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick terminal QA over a saved retrieval index.")
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR, help="Saved retrieval index directory.")
    parser.add_argument("--question", help="Ask one question and exit. If omitted, start interactive mode.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k evidence for manual strategy override.")
    parser.add_argument("--candidate-k", type=int, default=80, help="Candidate pool for manual strategy override.")
    parser.add_argument("--strategy", choices=["auto", "bm25", "hybrid", "hybrid_rerank"], default="auto")
    parser.add_argument("--reranker", choices=["none", "heuristic", "cross-encoder", "colbert"], default="heuristic")
    parser.add_argument("--combination", choices=["weighted_sum", "rrf"], default="weighted_sum")
    parser.add_argument("--bm25-weight", type=float, default=0.85)
    parser.add_argument("--dense-weight", type=float, default=0.15)
    parser.add_argument("--rerank-top-n", type=int, default=20)
    parser.add_argument("--context-window", type=int, default=1)
    parser.add_argument("--load-dense", action="store_true", help="Load dense embeddings if present.")
    parser.add_argument("--load-colbert", action="store_true", help="Load ColBERT embeddings if present.")
    parser.add_argument("--table-aware-retrieval", action="store_true", help="Enable table-aware retrieval boost.")
    parser.add_argument("--show-evidence", type=int, default=5, help="Number of retrieved hits to print.")
    parser.add_argument("--no-cell-answer", action="store_true", help="Do not prefer direct table-cell answers.")
    parser.add_argument(
        "--llm-explain",
        action="store_true",
        help="Use an LLM to explain the grounded answer from cited evidence. This does not replace the answer.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["dummy", "ollama", "openai", "openai-compatible", "openai_compatible"],
        help="LLM provider for --llm-explain. Defaults to BOXTALK_LLM_EXPLANATION_PROVIDER/BOXTALK_LLM_PROVIDER.",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON for single-question mode.")
    return parser.parse_args()


def build_pipeline(args: argparse.Namespace) -> GroundedQAPipeline:
    if args.table_aware_retrieval:
        os.environ["BOXBIIBOO_ENABLE_TABLE_AWARE_RETRIEVAL"] = "true"

    reranker = make_reranker(args.reranker)
    retrieval_service = RetrievalService.from_index(
        args.index_dir,
        reranker=reranker,
        load_dense=args.load_dense,
        load_colbert=args.load_colbert,
    )
    planner = None
    if args.strategy != "auto":
        planner = FixedRetrievalPlanner(
            strategy=args.strategy,
            config=RetrievalConfig(
                top_k=args.top_k,
                candidate_k=args.candidate_k,
                bm25_weight=args.bm25_weight,
                dense_weight=args.dense_weight,
                combination=args.combination,
                use_rerank=args.strategy == "hybrid_rerank",
                rerank_top_n=args.rerank_top_n if args.strategy == "hybrid_rerank" else 0,
                context_window=args.context_window,
            ),
        )
    return GroundedQAPipeline(
        retrieval_service=retrieval_service,
        router=QueryRouter(),
        retrieval_planner=planner,
        llm_explainer=make_llm_explainer_from_env(enabled=args.llm_explain, provider=args.llm_provider),
    )


def print_result(result: Any, *, show_evidence: int = 5, prefer_cell_answer: bool = True) -> None:
    cell_answer = extract_cell_answer(result.question, result.retrieved_hits) if prefer_cell_answer else None
    display_answer = cell_answer["answer"] if cell_answer else result.answer
    hit_text_by_chunk_id = {hit.chunk_id: hit.text for hit in result.retrieved_hits}
    print("\n" + "=" * 96)
    print(f"Question: {result.question}")
    print(f"Route: {result.query_type} | strategy: {result.retrieval_strategy} | decision: {result.decision}")
    print(
        "Latency: "
        f"retrieval={result.retrieval_latency_ms:.1f} ms, "
        f"answer={result.answer_latency_ms:.1f} ms, "
        f"total={result.total_latency_ms:.1f} ms"
    )
    print(f"Grounded: {result.grounded} | final_source: {result.final_answer_source}")
    print("-" * 96)
    print("Answer:")
    print(display_answer)
    if cell_answer:
        print(
            "Cell evidence: "
            f"row={cell_answer['row_header']} | "
            f"col={cell_answer['col_header']} | "
            f"cell={cell_answer['cell_text']} | "
            f"chunk_id={cell_answer['chunk_id']}"
        )
        if result.answer != display_answer:
            print(f"Pipeline answer: {result.answer}")

    if result.citations:
        print("-" * 96)
        print("Citations:")
        for idx, citation in enumerate(result.citations, start=1):
            print(f"[{idx}] {format_citation(citation)}")
            chunk_text = hit_text_by_chunk_id.get(str(citation.get("chunk_id") or ""))
            if chunk_text:
                print("    chunk:")
                print(f"    {format_chunk_text(chunk_text)}")

    if result.explanation:
        print("-" * 96)
        print("LLM explanation:")
        print(result.explanation)
    elif result.explanation_trace:
        print("-" * 96)
        print(
            "LLM explanation trace: "
            f"called={bool(result.explanation_trace.get('called', False))} | "
            f"used={bool(result.explanation_trace.get('used', False))} | "
            f"reason={result.explanation_trace.get('reason')} | "
            f"provider={result.explanation_trace.get('provider')}"
        )

    hits = result.retrieved_hits[: max(0, show_evidence)]
    if hits:
        print("-" * 96)
        print(f"Top evidence ({len(hits)}):")
        for hit in hits:
            meta = hit.chunk.metadata or {}
            citation_target = meta.get("citation_target")
            cell = ""
            if citation_target == "cell":
                cell = f" | row={meta.get('row_header')} | col={meta.get('col_header')} | cell={meta.get('cell_text')}"
            elif citation_target == "row":
                cell = f" | row={meta.get('row_header')}"
            print(
                f"#{hit.rank} score={float(hit.final_score or hit.score):.3f} "
                f"page={hit.page} type={hit.chunk.block_type} "
                f"strategy={meta.get('chunking_strategy')}{cell}"
            )
            print(f"   chunk_id={hit.chunk_id}")
            print(f"   {format_chunk_text(hit.text)}")
    print("=" * 96 + "\n")


def format_citation(citation: dict[str, Any]) -> str:
    parts = []
    for key in ("source_name", "doc_id", "page", "section", "chunk_id", "citation_target"):
        value = citation.get(key)
        if value not in (None, "", []):
            parts.append(f"{key}={value}")
    metadata = citation.get("metadata") if isinstance(citation.get("metadata"), dict) else {}
    for key in ("table_id", "row_header", "col_header", "cell_text"):
        value = metadata.get(key) or citation.get(key)
        if value not in (None, "", []):
            parts.append(f"{key}={value}")
    return " | ".join(parts) if parts else json.dumps(citation, ensure_ascii=False)


def snippet(text: str, limit: int = 280) -> str:
    clean = " ".join((text or "").split())
    if len(clean) <= limit:
        return clean
    return clean[: limit - 3] + "..."


def format_chunk_text(text: str) -> str:
    return " ".join((text or "").split())


def extract_cell_answer(question: str, hits: list[Any]) -> dict[str, Any] | None:
    folded_question = fold_text(question)
    candidates: list[tuple[float, dict[str, Any]]] = []
    for hit in hits:
        meta = hit.chunk.metadata or {}
        if meta.get("citation_target") != "cell":
            continue
        row_header = str(meta.get("row_header") or "")
        col_header = str(meta.get("col_header") or "")
        cell_text = str(meta.get("cell_text") or "")
        if not cell_text or is_header_like_cell(row_header, col_header, cell_text):
            continue

        folded_row = fold_text(row_header)
        folded_col = fold_text(col_header)
        folded_cell = fold_text(cell_text)
        score = 0.0
        if folded_col and phrase_or_token_match(folded_question, folded_col):
            score += 3.0
        if folded_row and token_overlap(folded_question, folded_row) >= 0.25:
            score += 0.8
        if folded_cell and folded_cell in folded_question:
            score += 1.0
        if any(char.isdigit() for char in col_header) and any(char.isdigit() for char in question):
            score += 0.4
        if str(meta.get("table_id") or ""):
            score += 0.1

        if score > 0.0:
            candidates.append(
                (
                    score,
                    {
                        "answer": cell_text,
                        "row_header": row_header,
                        "col_header": col_header,
                        "cell_text": cell_text,
                        "chunk_id": hit.chunk_id,
                        "score": score,
                    },
                )
            )

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def is_header_like_cell(row_header: str, col_header: str, cell_text: str) -> bool:
    folded_cell = fold_text(cell_text)
    if folded_cell == fold_text(row_header) or folded_cell == fold_text(col_header):
        return True
    return folded_cell in {"", "none", "nan"}


def phrase_or_token_match(haystack: str, needle: str) -> bool:
    if needle in haystack:
        return True
    return bool(set(tokens(haystack)) & set(tokens(needle)))


def token_overlap(left: str, right: str) -> float:
    left_tokens = set(tokens(left))
    right_tokens = set(tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(right_tokens)


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9+-]+", fold_text(text))


def fold_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = normalized.replace("đ", "d").replace("Đ", "D")
    folded = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    folded = folded.replace("≥", ">=").replace("≤", "<=").replace("−", "-").replace("–", "-")
    return re.sub(r"\s+", " ", folded).strip().lower()


def interactive_loop(pipeline: GroundedQAPipeline, args: argparse.Namespace) -> None:
    print("Quick QA terminal. Type 'exit', 'quit', or Ctrl+C to stop.")
    print(f"Index: {args.index_dir}")
    print(f"Strategy: {args.strategy}; table-aware retrieval: {args.table_aware_retrieval}")
    print(f"LLM explanation: {args.llm_explain}; provider: {args.llm_provider or 'env/default'}")
    while True:
        try:
            question = input("\nQ> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return
        if not question:
            continue
        if question.lower() in {"exit", "quit", ":q"}:
            print("Bye.")
            return
        result = pipeline.answer(question)
        print_result(result, show_evidence=args.show_evidence, prefer_cell_answer=not args.no_cell_answer)


def main() -> int:
    args = parse_args()
    pipeline = build_pipeline(args)
    if args.question:
        result = pipeline.answer(args.question)
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        else:
            print_result(result, show_evidence=args.show_evidence, prefer_cell_answer=not args.no_cell_answer)
        return 0
    interactive_loop(pipeline, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
