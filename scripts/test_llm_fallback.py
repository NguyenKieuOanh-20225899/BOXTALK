from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.qa.llm_fallback import DummyGroundedLLMClient, GroundedLLMFallback, LLMFallbackConfig
from app.qa.schemas import EvidenceAssessment, GroundedAnswer
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


if __name__ == "__main__":
    unittest.main()
