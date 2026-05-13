from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.qa.answer_generator import GroundedAnswerGenerator
from app.qa.schemas import EvidenceAssessment
from app.qa.table_lookup_utils import lookup_table_answer_from_text
from app.qa.table_query_utils import is_table_lookup_query
from app.retrieval.schemas import DocumentChunkRef, RetrievedHit


def make_hit(
    chunk_id: str,
    text: str,
    *,
    rank: int = 1,
    score: float = 1.0,
    block_type: str = "paragraph",
    metadata: dict | None = None,
) -> RetrievedHit:
    return RetrievedHit(
        chunk=DocumentChunkRef(
            chunk_id=chunk_id,
            text=text,
            doc_id="doc",
            source_name="doc.pdf",
            page=1,
            block_type=block_type,
            metadata=metadata or {},
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

    def test_table_owner_lookup_returns_concise_answer_from_flattened_chunk(self) -> None:
        question = "Who owns VPN access in the benefits table?"
        flattened_table = (
            "Benefits Table\n"
            "Benefit Waiting period Owner Health plan 30 days HR Ops VPN access Same day IT Support\n"
            "Benefits Table"
        )

        self.assertTrue(is_table_lookup_query(question))
        lookup = lookup_table_answer_from_text(question, flattened_table)
        self.assertIsNotNone(lookup)
        self.assertEqual(lookup.answer, "VPN access is owned by IT Support.")

        answer = GroundedAnswerGenerator().generate(
            question=question,
            query_type="factoid",
            hits=[make_hit("h1", flattened_table, block_type="table")],
            evidence=answer_evidence("h1"),
        )

        self.assertEqual(answer.answer, "VPN access is owned by IT Support.")

    def test_table_lookup_reads_positioned_table_cells_metadata(self) -> None:
        metadata = {
            "table_cells": [
                {"row": 0, "col": 0, "text": "Benefit"},
                {"row": 0, "col": 1, "text": "Waiting period"},
                {"row": 0, "col": 2, "text": "Owner"},
                {"row": 1, "col": 0, "text": "Health plan"},
                {"row": 1, "col": 1, "text": "30 days"},
                {"row": 1, "col": 2, "text": "HR Ops"},
                {"row": 2, "col": 0, "text": "VPN access"},
                {"row": 2, "col": 1, "text": "Same day"},
                {"row": 2, "col": 2, "text": "IT Support"},
            ]
        }

        answer = GroundedAnswerGenerator().generate(
            question="Who owns VPN access in the benefits table?",
            query_type="factoid",
            hits=[make_hit("h1", "Benefits Table", block_type="table", metadata=metadata)],
            evidence=answer_evidence("h1"),
        )

        self.assertEqual(answer.answer, "VPN access is owned by IT Support.")


if __name__ == "__main__":
    unittest.main()
