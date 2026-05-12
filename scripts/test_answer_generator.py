from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.qa.answer_generator import GroundedAnswerGenerator
from app.qa.schemas import EvidenceAssessment
from app.retrieval.schemas import DocumentChunkRef, RetrievedHit


def make_hit(chunk_id: str, text: str, *, rank: int = 1, score: float = 1.0) -> RetrievedHit:
    return RetrievedHit(
        chunk=DocumentChunkRef(
            chunk_id=chunk_id,
            text=text,
            doc_id="doc",
            source_name="doc.pdf",
            page=1,
        ),
        score=score,
        final_score=score,
        source="bm25",
        rank=rank,
        bm25_score=score,
    )


def answer_evidence(*chunk_ids: str) -> EvidenceAssessment:
    return EvidenceAssessment(
        relevance=1.0,
        coverage=1.0,
        consistency=1.0,
        citation_support=1.0,
        grounding=1.0,
        sufficiency=1.0,
        decision="answer",
        reason="test",
        selected_hit_ids=list(chunk_ids),
        support_sentences=[],
    )


class AnswerGeneratorTest(unittest.TestCase):
    def test_duration_question_prefers_payload_span_over_preamble(self) -> None:
        generator = GroundedAnswerGenerator()
        hits = [
            make_hit(
                "h1",
                (
                    "CTDT tich hop bao gom hai bac trinh do: Cu nhan (thoi gian dao tao 4 nam, "
                    "cap bang cu nhan) va ky su (thoi gian dao tao 1,5 nam, cap bang ky su) "
                    "hoac thac si (thoi gian dao tao 1,5 nam, cap bang thac si)."
                ),
                rank=1,
            ),
            make_hit(
                "h2",
                "2 Quy che tuyen sinh va dao tao trinh do thac si, ban hanh theo thong tu nam 2021.",
                rank=2,
                score=0.99,
            ),
        ]

        answer = generator.generate(
            question="thac si can bao nhieu nam dao tao",
            query_type="factoid",
            hits=hits,
            evidence=answer_evidence("h1", "h2"),
        )

        self.assertIn("1,5 nam", answer.answer)
        self.assertIn("thac si", answer.answer.lower())
        self.assertNotIn("Quy che tuyen sinh", answer.answer)

    def test_english_duration_question_prefers_answer_shape(self) -> None:
        generator = GroundedAnswerGenerator()
        hits = [
            make_hit(
                "h1",
                (
                    "The access policy is issued under the security handbook. "
                    "Privileged access is valid for 24 hours after approval. "
                    "Standard access lasts 7 days."
                ),
            )
        ]

        answer = generator.generate(
            question="How long is privileged access valid?",
            query_type="factoid",
            hits=hits,
            evidence=answer_evidence("h1"),
        )

        self.assertIn("24 hours", answer.answer)
        self.assertIn("Privileged access", answer.answer)


if __name__ == "__main__":
    unittest.main()
