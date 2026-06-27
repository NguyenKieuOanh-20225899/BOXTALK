from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8")

from app.qa.answer_generator import GroundedAnswerGenerator
from app.qa.pipeline import GroundedQAPipeline
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
    parser.add_argument("--show-context", action="store_true", help="Print the exact evidence context sent to the LLM.")
    parser.add_argument(
        "--answer-generator",
        choices=["llm", "extractive"],
        default="llm",
        help="Answer generator used after retrieval. 'extractive' preserves the old baseline.",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["dummy", "ollama", "openai", "openai-compatible", "openai_compatible"],
        help="LLM provider for the main grounded answer generator.",
    )
    parser.add_argument("--llm-model", help="LLM model for the main grounded answer generator.")
    parser.add_argument("--llm-base-url", help="OpenAI-compatible/Ollama base URL.")
    parser.add_argument(
        "--no-start-ollama",
        action="store_true",
        help="Do not auto-start 'ollama serve' when --llm-provider ollama is used.",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON for single-question mode.")
    return parser.parse_args()


def build_pipeline(args: argparse.Namespace) -> GroundedQAPipeline:
    if args.table_aware_retrieval:
        os.environ["BOXBIIBOO_ENABLE_TABLE_AWARE_RETRIEVAL"] = "true"
    if args.llm_provider:
        os.environ["BOXTALK_LLM_PROVIDER"] = args.llm_provider
    if args.llm_model:
        os.environ["BOXTALK_LLM_MODEL"] = args.llm_model
    if args.llm_base_url:
        os.environ["BOXTALK_LLM_BASE_URL"] = args.llm_base_url
    ensure_ollama_server(args)

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
    answer_generator = GroundedAnswerGenerator() if args.answer_generator == "extractive" else None
    return GroundedQAPipeline(
        retrieval_service=retrieval_service,
        router=QueryRouter(),
        retrieval_planner=planner,
        answer_generator=answer_generator,
    )


def ensure_ollama_server(args: argparse.Namespace) -> None:
    provider = (args.llm_provider or os.getenv("BOXTALK_LLM_PROVIDER") or "").strip().lower()
    if args.answer_generator == "extractive" or args.no_start_ollama or provider != "ollama":
        return
    if ollama_ready(args):
        return

    print("Ollama server is not responding; starting 'ollama serve' in the background...")
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW") else 0
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    except FileNotFoundError:
        print("Warning: 'ollama' executable was not found. Start Ollama manually before asking questions.")
        return
    except Exception as exc:
        print(f"Warning: could not start Ollama automatically: {exc}")
        return

    for _ in range(12):
        time.sleep(1)
        if ollama_ready(args):
            print("Ollama server is ready.")
            return
    print("Warning: Ollama did not become ready within 12 seconds. The LLM call may still fail.")


def ollama_ready(args: argparse.Namespace) -> bool:
    request = urllib.request.Request(ollama_tags_url(args), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            return 200 <= int(response.status) < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def ollama_tags_url(args: argparse.Namespace) -> str:
    base_url = (args.llm_base_url or os.getenv("BOXTALK_LLM_BASE_URL") or "http://localhost:11434/v1").rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]
    return f"{base_url}/api/tags"


def print_result(result: Any, *, show_evidence: int = 5, show_context: bool = False) -> None:
    hit_text_by_chunk_id = {hit.chunk_id: hit.text for hit in result.retrieved_hits}
    print("\n" + "=" * 96)
    print(f"Question: {result.question}")
    print(f"Route: {result.query_type}")
    print(f"Strategy: {result.retrieval_strategy}")
    print(f"Decision: {result.decision}")
    print(
        "Latency: "
        f"retrieval={result.retrieval_latency_ms:.1f} ms, "
        f"answer={result.answer_latency_ms:.1f} ms, "
        f"total={result.total_latency_ms:.1f} ms"
    )
    print(f"Grounded: {result.grounded} | final_source: {result.final_answer_source}")
    print("-" * 96)
    print("Evidence:")
    print(f"  Retrieved: {len(result.retrieved_hits)}")
    print(f"  Selected: {result.selected_evidence_count}")
    print(f"  Sufficient: {result.evidence_sufficient}")
    print(f"  Reason: {result.evidence_reason}")
    if result.missing_constraints:
        print(f"  Missing constraints: {', '.join(result.missing_constraints)}")

    if result.context_token_count is not None:
        print("-" * 96)
        print("Context:")
        print(f"  Evidence IDs: {', '.join(result.selected_evidence_ids) if result.selected_evidence_ids else '-'}")
        print(f"  Context tokens: {result.context_token_count if result.context_token_count is not None else '-'}")
        if show_context and result.context_evidence:
            print_context_evidence(result.context_evidence)

    if result.evidence_sufficient or result.generator_type != "llm_grounded":
        print("-" * 96)
        print("Generator:")
        print(f"  Type: {result.generator_type}")
        if result.generator_provider:
            print(f"  Provider: {result.generator_provider}")
        if result.generator_model:
            print(f"  Model: {result.generator_model}")
        print(f"  LLM latency: {result.llm_latency_ms:.1f} ms")

    print("-" * 96)
    print("Validation:")
    print(f"  Passed: {result.validation_passed}")
    if result.validation_reason:
        print(f"  Reason: {result.validation_reason}")

    print("-" * 96)
    print("Answer:")
    print(result.answer)

    if result.citations:
        print("-" * 96)
        print("Citations:")
        for idx, citation in enumerate(result.citations, start=1):
            print(f"[{idx}] {format_citation(citation)}")
            chunk_text = hit_text_by_chunk_id.get(str(citation.get("chunk_id") or ""))
            if chunk_text:
                print("    chunk:")
                print(f"    {format_chunk_text(chunk_text)}")

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


def print_context_evidence(context_evidence: list[dict[str, Any]]) -> None:
    print("  LLM context evidence:")
    for item in context_evidence:
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        parts = [
            f"evidence_id={item.get('evidence_id')}",
            f"page={item.get('page')}",
            f"chunk_id={item.get('chunk_id')}",
            f"citation_target={item.get('citation_target')}",
        ]
        for key in ("table_id", "row_header", "col_header", "cell_text"):
            value = metadata.get(key)
            if value not in (None, "", []):
                parts.append(f"{key}={value}")
        print(f"  - {' | '.join(parts)}")
        print(f"    content: {format_chunk_text(str(item.get('text') or ''))}")


def format_chunk_text(text: str) -> str:
    return " ".join((text or "").split())


def interactive_loop(pipeline: GroundedQAPipeline, args: argparse.Namespace) -> None:
    print("Quick QA terminal. Type 'exit', 'quit', or Ctrl+C to stop.")
    print(f"Index: {args.index_dir}")
    print(f"Strategy: {args.strategy}; table-aware retrieval: {args.table_aware_retrieval}")
    print(f"Answer generator: {args.answer_generator}; LLM provider: {args.llm_provider or os.getenv('BOXTALK_LLM_PROVIDER') or 'not configured'}")
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
        print_result(result, show_evidence=args.show_evidence, show_context=args.show_context)


def main() -> int:
    args = parse_args()
    pipeline = build_pipeline(args)
    if args.question:
        result = pipeline.answer(args.question)
        if args.json:
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        else:
            print_result(result, show_evidence=args.show_evidence, show_context=args.show_context)
        return 0
    interactive_loop(pipeline, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
