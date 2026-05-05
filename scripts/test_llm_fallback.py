from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.qa.llm_fallback import (
    DummyGroundedLLMClient,
    GroundedLLMFallback,
    LLMFallbackConfig,
    make_grounded_llm_client,
    provider_runtime_info,
    response_from_payload,
)
from app.qa.schemas import EvidenceAssessment, GroundedAnswer
from app.qa.table_lookup_utils import lookup_table_answer, normalize_table_from_sources
from app.retrieval.schemas import DocumentChunkRef, RetrievedHit


def make_hit(chunk_id: str, text: str, *, block_type: str = "paragraph", score: float = 0.8) -> RetrievedHit:
    return RetrievedHit(
        chunk=DocumentChunkRef(
            chunk_id=chunk_id,
            text=text,
            doc_id="doc",
            source_name="doc.pdf",
            page=1,
            block_type=block_type,
        ),
        score=score,
        final_score=score,
        source="bm25",
        rank=1,
        bm25_score=score,
    )


def make_evidence(decision: str, *, relevance: float = 0.7, sufficiency: float = 0.45) -> EvidenceAssessment:
    return EvidenceAssessment(
        relevance=relevance,
        coverage=0.5,
        consistency=1.0,
        citation_support=1.0,
        grounding=0.5,
        sufficiency=sufficiency,
        decision=decision,  # type: ignore[arg-type]
        reason="test",
        selected_hit_ids=["h1"],
        support_sentences=[],
    )


def standard_answer(text: str) -> GroundedAnswer:
    return GroundedAnswer(answer=text, citations=[], support_sentences=[], grounded=False)


class LLMFallbackTest(unittest.TestCase):
    def test_numeric_fallback_uses_grounded_evidence_span(self) -> None:
        fallback = GroundedLLMFallback(
            config=LLMFallbackConfig(enable_llm_fallback=True, min_llm_confidence=0.10),
            client=DummyGroundedLLMClient(),
        )
        result = fallback.maybe_generate(
            question="How many attention heads does the model use?",
            query_type="factoid",
            hits=[make_hit("h1", "The model uses h = 8 parallel attention heads in the attention layer.")],
            evidence=make_evidence("answer"),
            standard_answer=standard_answer("The paper describes the model architecture."),
        )

        self.assertTrue(result.called)
        self.assertTrue(result.used)
        self.assertIn("8", result.answer or "")
        self.assertEqual(result.used_evidence_ids, ["E1"])

    def test_fallback_does_not_call_without_grounded_evidence(self) -> None:
        fallback = GroundedLLMFallback(
            config=LLMFallbackConfig(enable_llm_fallback=True),
            client=DummyGroundedLLMClient(),
        )
        result = fallback.maybe_generate(
            question="What is the answer?",
            query_type="factoid",
            hits=[],
            evidence=make_evidence("switch_strategy", relevance=0.0, sufficiency=0.0),
            standard_answer=standard_answer("I do not have enough grounded evidence to answer."),
        )

        self.assertFalse(result.called)
        self.assertFalse(result.used)

    def test_table_rule_based_lookup_runs_before_llm(self) -> None:
        fallback = GroundedLLMFallback(
            config=LLMFallbackConfig(enable_llm_fallback=True, enable_table_llm_reasoning=True),
            client=DummyGroundedLLMClient(),
        )
        table_text = "Score range | Grade\n8.5 - 10 | A\n7.0 - 8.4 | B\n6.0 - 6.9 | C"
        result = fallback.maybe_generate(
            question="6.5 corresponds to which grade?",
            query_type="factoid",
            hits=[make_hit("h1", table_text, block_type="table")],
            evidence=make_evidence("answer", sufficiency=0.85),
            standard_answer=standard_answer("The table lists score ranges and grades."),
        )

        self.assertTrue(result.called)
        self.assertTrue(result.used)
        self.assertFalse(result.llm_called)
        self.assertEqual(result.final_answer_source, "table_rule_fallback")
        self.assertIn("C", result.answer or "")

    def test_table_rule_based_lookup_keeps_plus_grade(self) -> None:
        fallback = GroundedLLMFallback(
            config=LLMFallbackConfig(enable_llm_fallback=True, enable_table_llm_reasoning=True),
            client=DummyGroundedLLMClient(),
        )
        table_text = "Score range | Grade\n8.0 - 8.9 | B+\n6.5 - 6.9 | C+\n5.5 - 6.4 | C"
        result = fallback.maybe_generate(
            question="6.5 la C hay C+?",
            query_type="factoid",
            hits=[make_hit("h1", table_text, block_type="table")],
            evidence=make_evidence("answer", sufficiency=0.85),
            standard_answer=standard_answer("The table maps score ranges to grades."),
        )

        self.assertTrue(result.used)
        self.assertIn("C+", result.answer or "")
        self.assertNotIn("maps to C.", result.answer or "")

    def test_table_rule_based_reverse_lookup_returns_interval_and_grade_point(self) -> None:
        fallback = GroundedLLMFallback(
            config=LLMFallbackConfig(enable_llm_fallback=True, enable_table_llm_reasoning=True),
            client=DummyGroundedLLMClient(),
        )
        table_text = "Score Range | Letter Grade | Grade Point\n8.0 - 8.9 | B+ | 3.5\n6.5 - 6.9 | C+ | 2.5"
        result = fallback.maybe_generate(
            question="What score band corresponds to B+ and what grade point does it carry?",
            query_type="factoid",
            hits=[make_hit("h1", table_text, block_type="table")],
            evidence=make_evidence("answer", sufficiency=0.85),
            standard_answer=standard_answer("The table maps score ranges to grades."),
        )

        self.assertTrue(result.used)
        self.assertFalse(result.llm_called)
        self.assertIn("8.0 - 8.9", result.answer or "")
        self.assertIn("3.5", result.answer or "")

    def test_table_rule_based_boundary_and_below_are_not_confused(self) -> None:
        table = normalize_table_from_sources(
            table_text=(
                "Score Range | Grade\n"
                "4.0 - 5.4 | D\n"
                "Below 4.0 | F"
            )
        )
        self.assertIsNotNone(table)
        boundary = lookup_table_answer("4.0 belongs to which range?", table)  # type: ignore[arg-type]
        below = lookup_table_answer("3.9 belongs to which grade?", table)  # type: ignore[arg-type]

        self.assertIsNotNone(boundary)
        self.assertIn("D", boundary.answer if boundary else "")
        self.assertIsNotNone(below)
        self.assertIn("F", below.answer if below else "")

    def test_table_normalizes_mixed_decimal_separators(self) -> None:
        table = normalize_table_from_sources(
            table_text=(
                "Khoang diem | Diem chu\n"
                "6,5 - 6,9 | C+\n"
                "5.5 - 6.4 | C"
            )
        )
        self.assertIsNotNone(table)
        result = lookup_table_answer("6.5 la C hay C+?", table)  # type: ignore[arg-type]

        self.assertIsNotNone(result)
        self.assertIn("C+", result.answer if result else "")

    def test_table_rule_based_multiple_column_lookup(self) -> None:
        fallback = GroundedLLMFallback(
            config=LLMFallbackConfig(enable_llm_fallback=True, enable_table_llm_reasoning=True),
            client=DummyGroundedLLMClient(),
        )
        table_text = "Model | Heads | Layers | BLEU\nBase | 8 | 6 | 27.3\nLarge | 16 | 12 | 29.8"
        result = fallback.maybe_generate(
            question="Which configuration uses 12 layers and what BLEU does it reach?",
            query_type="factoid",
            hits=[make_hit("h1", table_text, block_type="table")],
            evidence=make_evidence("answer", sufficiency=0.85),
            standard_answer=standard_answer("The table compares model configurations."),
        )

        self.assertTrue(result.used)
        self.assertIn("Large", result.answer or "")
        self.assertIn("29.8", result.answer or "")

    def test_openai_compatible_provider_reports_missing_envs(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            info = provider_runtime_info("openai-compatible")

        self.assertFalse(info["ready"])
        self.assertEqual(
            set(info["missing_envs"]),
            {"BOXTALK_LLM_BASE_URL", "BOXTALK_LLM_API_KEY", "BOXTALK_LLM_MODEL"},
        )

    def test_openai_compatible_provider_does_not_expose_api_key(self) -> None:
        with patch.dict(
            os.environ,
            {
                "BOXTALK_LLM_BASE_URL": "https://example.test/v1",
                "BOXTALK_LLM_API_KEY": "secret-test-key",
                "BOXTALK_LLM_MODEL": "test-model",
            },
            clear=True,
        ):
            info = provider_runtime_info("openai-compatible")

        self.assertTrue(info["ready"])
        self.assertTrue(info["api_key_present"])
        self.assertEqual(info["base_url"], "https://example.test/v1")
        self.assertEqual(info["model"], "test-model")
        self.assertNotIn("BOXTALK_LLM_API_KEY", info["env"])
        self.assertNotIn("secret-test-key", json.dumps(info))

    def test_ollama_provider_uses_local_openai_compatible_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            info = provider_runtime_info("ollama")
            client = make_grounded_llm_client("ollama")

        self.assertTrue(info["ready"])
        self.assertEqual(info["provider"], "ollama")
        self.assertEqual(info["base_url"], "http://localhost:11434/v1")
        self.assertEqual(info["model"], "qwen2.5:7b-instruct")
        self.assertTrue(info["api_key_present"])
        self.assertEqual(getattr(client, "provider_name"), "ollama")
        self.assertEqual(getattr(client, "base_url"), "http://localhost:11434/v1")
        self.assertEqual(getattr(client, "model"), "qwen2.5:7b-instruct")

    def test_llm_response_with_answer_and_evidence_ids_implies_answer_decision(self) -> None:
        response = response_from_payload(
            {
                "answer": "B+ has a higher grade point than C+.",
                "used_evidence_ids": ["E1"],
                "reasoning_mode": "table",
                "confidence": 0.95,
            },
            "table",
        )

        self.assertEqual(response.decision, "answer")
        self.assertEqual(response.used_evidence_ids, ["E1"])
        self.assertEqual(response.reasoning_mode, "table")


if __name__ == "__main__":
    unittest.main()
