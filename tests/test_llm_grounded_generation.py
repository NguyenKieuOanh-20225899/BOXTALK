from __future__ import annotations

import unittest
from typing import Any

from app.qa.answer_generator import GroundedAnswerGenerator
from app.qa.answer_validator import AnswerValidator
from app.qa.citation_builder import CitationBuilder
from app.qa.context_builder import ContextBuilder
from app.qa.evidence_checker import EvidenceChecker
from app.qa.llm_answer_generator import GeneratedGroundedAnswer
from app.qa.llm_fallback import response_from_payload
from app.qa.pipeline import GroundedQAPipeline
from app.qa.schemas import EvidenceAssessment
from app.retrieval.route_planner import QueryRetrievalPlan
from app.retrieval.schemas import DocumentChunkRef, RetrievedHit, RetrievalConfig, RetrievalResult


def make_hit(
    chunk_id: str,
    text: str,
    *,
    rank: int = 1,
    score: float = 0.9,
    block_type: str = "paragraph",
    metadata: dict[str, Any] | None = None,
) -> RetrievedHit:
    return RetrievedHit(
        chunk=DocumentChunkRef(
            chunk_id=chunk_id,
            text=text,
            doc_id="doc",
            source_name="doc.pdf",
            page=1,
            section="Điều 1",
            block_type=block_type,
            metadata=metadata or {},
        ),
        score=score,
        final_score=score,
        source="hybrid",
        rank=rank,
    )


def sufficient_evidence(*chunk_ids: str) -> EvidenceAssessment:
    return EvidenceAssessment(
        relevance=1.0,
        coverage=1.0,
        consistency=1.0,
        citation_support=1.0,
        grounding=1.0,
        sufficiency=1.0,
        decision="answer",
        reason="test sufficient",
        selected_hit_ids=list(chunk_ids),
        sufficient=True,
    )


class FakeRouter:
    def __init__(self, query_type: str = "factoid") -> None:
        self.query_type = query_type

    def route(self, question: str) -> str:
        return self.query_type


class FakePlanner:
    def plan(self, query_type: str, question: str) -> QueryRetrievalPlan:
        _ = query_type, question
        return QueryRetrievalPlan(strategy="hybrid", config=RetrievalConfig(top_k=3), reason="test")


class FakeRetrievalService:
    def __init__(self, hits: list[RetrievedHit]) -> None:
        self.hits = hits

    def retrieve(self, query: str, *, strategy: str = "hybrid", config: RetrievalConfig | None = None) -> RetrievalResult:
        return RetrievalResult(
            query=query,
            strategy=strategy,
            hits=self.hits,
            config=config or RetrievalConfig(),
            latency_ms=1.0,
            retrieval_count=len(self.hits),
        )


class FakeEvidenceChecker:
    def __init__(self, evidence: EvidenceAssessment) -> None:
        self.evidence = evidence

    def check(self, *, question: str, hits: list[RetrievedHit], query_type: str = "factoid") -> EvidenceAssessment:
        _ = question, hits, query_type
        return self.evidence


class FakeLLMGenerator:
    generator_type = "llm_grounded"

    def __init__(self, result: GeneratedGroundedAnswer) -> None:
        self.result = result
        self.calls = 0

    def generate(self, *, question: str, context: Any) -> GeneratedGroundedAnswer:
        _ = question, context
        self.calls += 1
        return self.result


class ContextBuilderTest(unittest.TestCase):
    def test_maps_evidence_ids_and_preserves_table_metadata(self) -> None:
        hit = make_hit(
            "c1",
            "Điểm 8,2 tương ứng B+ và 3,5.",
            block_type="table",
            metadata={
                "citation_target": "cell",
                "table_id": "t1",
                "row_header": "8,0-8,4",
                "col_header": "Điểm chữ",
                "cell_text": "B+",
            },
        )

        context = ContextBuilder().build(question="Điểm 8,2 là gì?", selected_hits=[hit, hit])

        self.assertEqual(context.evidence_ids, ["E1"])
        self.assertEqual(context.evidence[0].chunk_id, "c1")
        self.assertEqual(context.evidence[0].metadata["cell_text"], "B+")
        self.assertIn("EVIDENCE E1", context.to_prompt_text())

    def test_compacts_plain_text_but_preserves_markdown_table_lines(self) -> None:
        plain = make_hit("plain", "Dòng 1\n\n   Dòng    2")
        table = make_hit(
            "table",
            "| Điểm học phần theo thang 10 | 7,0÷7,9 |\n"
            "| --- | --- |\n"
            "| Điểm chữ quy đổi | B |\n"
            "| Điểm số quy đổi | 3,0 |",
            block_type="table",
            metadata={"citation_target": "table", "table_id": "t1"},
        )

        plain_context = ContextBuilder().build(question="Q", selected_hits=[plain])
        table_context = ContextBuilder().build(question="Q", selected_hits=[table])
        table_prompt = table_context.to_prompt_text()

        self.assertEqual(plain_context.evidence[0].text, "Dòng 1 Dòng 2")
        self.assertIn("Content:\n| Điểm học phần theo thang 10 | 7,0÷7,9 |", table_prompt)
        self.assertIn("| --- | --- |", table_prompt)
        self.assertIn("| Điểm số quy đổi | 3,0 |", table_prompt)

    def test_context_builder_keeps_cell_metadata_and_limits_items(self) -> None:
        hits = [
            make_hit(
                f"c{idx}",
                f"Nội dung {idx}",
                block_type="table",
                metadata={
                    "citation_target": "cell",
                    "table_id": "t1",
                    "row_header": "Điểm chữ quy đổi",
                    "col_header": "7,0÷7,9",
                    "cell_text": "B",
                },
            )
            for idx in range(6)
        ]

        context = ContextBuilder().build(question="Điểm 7.0 là gì?", selected_hits=hits)
        prompt = context.to_prompt_text()

        self.assertEqual(len(context.evidence), 5)
        self.assertIn("Row header: Điểm chữ quy đổi", prompt)
        self.assertIn("Column header: 7,0÷7,9", prompt)
        self.assertIn("Cell text: B", prompt)
        self.assertIn("Điểm 7.0", prompt)


class CitationBuilderTest(unittest.TestCase):
    def test_maps_valid_ids_and_records_invalid_ids(self) -> None:
        context = ContextBuilder().build(question="Q", selected_hits=[make_hit("c1", "Answer text")])

        result = CitationBuilder().build(context=context, used_evidence_ids=["E1", "E99"])

        self.assertEqual(len(result.citations), 1)
        self.assertEqual(result.citations[0]["evidence_id"], "E1")
        self.assertEqual(result.citations[0]["chunk_id"], "c1")
        self.assertEqual(result.invalid_evidence_ids, ["E99"])


class AnswerValidatorTest(unittest.TestCase):
    def test_rejects_money_question_without_money_amount(self) -> None:
        context = ContextBuilder().build(
            question="Học phí ngành Công nghệ thông tin năm 2026 là bao nhiêu?",
            selected_hits=[make_hit("c1", "Điều 9. Học phí. 6. Học phí đối với NCS được tính theo năm học.")],
        )
        citations = CitationBuilder().build(context=context, used_evidence_ids=["E1"]).citations

        result = AnswerValidator().validate(
            question=context.question,
            answer="Học phí 6.",
            used_evidence_ids=["E1"],
            citations=citations,
            context=context,
        )

        self.assertFalse(result.valid)
        self.assertIn(result.reason, {"answer_looks_like_section_number", "money_question_without_money_amount"})

    def test_rejects_unknown_evidence_id(self) -> None:
        context = ContextBuilder().build(question="Q", selected_hits=[make_hit("c1", "Answer text")])

        result = AnswerValidator().validate(
            question="Q",
            answer="Answer text",
            used_evidence_ids=["E99"],
            citations=[],
            context=context,
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "unknown_evidence_id")

    def test_allows_lookup_number_from_question_but_requires_answer_number_in_evidence(self) -> None:
        context = ContextBuilder().build(
            question="Điểm 8,2 tương ứng với điểm chữ và điểm hệ 4 nào?",
            selected_hits=[
                make_hit(
                    "c1",
                    "| thang 10 | 7,9 8,4 9,4 |\n| Điểm chữ quy đổi | B B+ A |\n| Điểm số quy đổi | 3,0 3,5 4,0 |",
                    block_type="table",
                    metadata={"citation_target": "table", "table_id": "grade_table"},
                )
            ],
        )
        citations = CitationBuilder().build(context=context, used_evidence_ids=["E1"]).citations

        result = AnswerValidator().validate(
            question=context.question,
            answer="Điểm 8,2 tương ứng với điểm chữ B+ và điểm hệ 4 là 3,5.",
            used_evidence_ids=["E1"],
            citations=citations,
            context=context,
        )

        self.assertTrue(result.valid)


class EvidenceCheckerTest(unittest.TestCase):
    def test_money_question_requires_programme_year_and_money_amount(self) -> None:
        hit = make_hit(
            "h1",
            "Điều 9. Học phí. 6. Học phí đối với NCS được tính theo năm học.",
            score=0.99,
        )

        evidence = EvidenceChecker().assess(
            "Học phí ngành Công nghệ thông tin năm 2026 là bao nhiêu?",
            "factoid",
            [hit],
        )

        self.assertFalse(evidence.sufficient)
        self.assertIn("programme", evidence.missing_constraints)
        self.assertIn("year", evidence.missing_constraints)
        self.assertIn("money_amount", evidence.missing_constraints)

    def test_text_question_selects_at_most_five_hits(self) -> None:
        hits = [make_hit(f"h{idx}", f"Relevant text {idx}", score=0.9 - idx * 0.01) for idx in range(5)]

        evidence = EvidenceChecker().assess("Sinh viên bị cảnh báo khi nào?", "policy", hits)

        self.assertEqual(evidence.selected_hit_ids, ["h0", "h1", "h2", "h3", "h4"])

    def test_table_question_prioritizes_markdown_cells_and_same_table_id(self) -> None:
        other_table_cell = make_hit(
            "other_cell",
            "Bảng khác. Hàng Điểm chữ quy đổi, cột 7,0÷7,9: X.",
            block_type="table",
            metadata={
                "citation_target": "cell",
                "table_id": "other",
                "row_header": "Điểm chữ quy đổi",
                "col_header": "7,0÷7,9",
                "cell_text": "X",
            },
        )
        corner_cell = make_hit(
            "corner",
            "Hàng Điểm chữ quy đổi, cột Điểm học phần theo thang 10: Điểm chữ quy đổi.",
            block_type="table",
            metadata={
                "citation_target": "cell",
                "table_id": "grade",
                "row_header": "Điểm chữ quy đổi",
                "col_header": "Điểm học phần theo thang 10",
                "cell_text": "Điểm chữ quy đổi",
            },
        )
        markdown_table = make_hit(
            "table",
            "| Điểm học phần theo thang 10 | 7,0÷7,9 |\n"
            "| --- | --- |\n"
            "| Điểm chữ quy đổi | B |\n"
            "| Điểm số quy đổi | 3,0 |",
            block_type="table",
            metadata={"citation_target": "table", "table_id": "grade"},
        )
        adjacent_cell = make_hit(
            "adjacent_cell",
            "Hàng Điểm chữ quy đổi, cột 6,5÷6,9: C+.",
            block_type="table",
            metadata={
                "citation_target": "cell",
                "table_id": "grade",
                "row_header": "Điểm chữ quy đổi",
                "col_header": "6,5÷6,9",
                "cell_text": "C+",
            },
        )
        grade_cell = make_hit(
            "grade_cell",
            "Hàng Điểm chữ quy đổi, cột 7,0÷7,9: B.",
            block_type="table",
            metadata={
                "citation_target": "cell",
                "table_id": "grade",
                "row_header": "Điểm chữ quy đổi",
                "col_header": "7,0÷7,9",
                "cell_text": "B",
            },
        )
        point_cell = make_hit(
            "point_cell",
            "Hàng Điểm số quy đổi, cột 7,0÷7,9: 3,0.",
            block_type="table",
            metadata={
                "citation_target": "cell",
                "table_id": "grade",
                "row_header": "Điểm số quy đổi",
                "col_header": "7,0÷7,9",
                "cell_text": "3,0",
            },
        )
        row_hit = make_hit(
            "row",
            "Hàng Điểm chữ quy đổi: F D D+ C C+ B B+ A A+.",
            block_type="table",
            metadata={"citation_target": "row", "table_id": "grade", "row_header": "Điểm chữ quy đổi"},
        )
        hits = [other_table_cell, corner_cell, markdown_table, adjacent_cell, grade_cell, point_cell, row_hit]

        evidence = EvidenceChecker().assess(
            "Điểm 7.0 tương ứng với điểm chữ và điểm hệ 4 nào?",
            "factoid",
            hits,
        )

        self.assertEqual(evidence.selected_hit_ids, ["table", "grade_cell", "point_cell", "row"])
        self.assertEqual(len(evidence.selected_hit_ids), 4)
        self.assertNotIn("corner", evidence.selected_hit_ids)
        self.assertNotIn("other_cell", evidence.selected_hit_ids)
        self.assertNotIn("adjacent_cell", evidence.selected_hit_ids)

    def test_table_question_without_markdown_still_keeps_useful_cells_in_order(self) -> None:
        hits = [
            make_hit(
                "bad_col",
                "Hàng thang 10, cột 7,0÷7,9: 7,0÷7,9.",
                block_type="table",
                metadata={
                    "citation_target": "cell",
                    "table_id": "grade",
                    "row_header": "Điểm học phần theo thang 10",
                    "col_header": "7,0÷7,9",
                    "cell_text": "7,0÷7,9",
                },
            ),
            make_hit(
                "grade_cell",
                "Hàng Điểm chữ quy đổi, cột 7,0÷7,9: B.",
                block_type="table",
                metadata={
                    "citation_target": "cell",
                    "table_id": "grade",
                    "row_header": "Điểm chữ quy đổi",
                    "col_header": "7,0÷7,9",
                    "cell_text": "B",
                },
            ),
            make_hit(
                "adjacent_cell",
                "Hàng Điểm chữ quy đổi, cột 6,5÷6,9: C+.",
                block_type="table",
                metadata={
                    "citation_target": "cell",
                    "table_id": "grade",
                    "row_header": "Điểm chữ quy đổi",
                    "col_header": "6,5÷6,9",
                    "cell_text": "C+",
                },
            ),
            make_hit(
                "point_cell",
                "Hàng Điểm số quy đổi, cột 7,0÷7,9: 3,0.",
                block_type="table",
                metadata={
                    "citation_target": "cell",
                    "table_id": "grade",
                    "row_header": "Điểm số quy đổi",
                    "col_header": "7,0÷7,9",
                    "cell_text": "3,0",
                },
            ),
        ]

        evidence = EvidenceChecker().assess(
            "Điểm 7.0 tương ứng với điểm chữ và điểm hệ 4 nào?",
            "factoid",
            hits,
        )

        self.assertEqual(evidence.selected_hit_ids, ["grade_cell", "point_cell"])


class LLMResponseParsingTest(unittest.TestCase):
    def test_abstain_schema_maps_to_insufficient_evidence(self) -> None:
        response = response_from_payload(
            {"answer": "", "used_evidence_ids": [], "abstain": True, "reason": "missing year"},
            "text",
        )

        self.assertEqual(response.decision, "insufficient_evidence")


class PipelineTest(unittest.TestCase):
    def test_insufficient_evidence_does_not_call_llm(self) -> None:
        hits = [make_hit("h1", "Điều 9. Học phí. 6. Học phí đối với NCS được tính theo năm học.", score=0.99)]
        generator = FakeLLMGenerator(
            GeneratedGroundedAnswer(
                answer="bad",
                used_evidence_ids=["E1"],
                abstain=False,
                reason=None,
            )
        )
        pipeline = GroundedQAPipeline(
            retrieval_service=FakeRetrievalService(hits),  # type: ignore[arg-type]
            router=FakeRouter(),
            retrieval_planner=FakePlanner(),  # type: ignore[arg-type]
            evidence_checker=EvidenceChecker(),
            answer_generator=generator,
        )

        result = pipeline.answer("Học phí ngành Công nghệ thông tin năm 2026 là bao nhiêu?")

        self.assertEqual(generator.calls, 0)
        self.assertEqual(result.decision, "abstain")
        self.assertEqual(result.final_answer_source, "evidence_insufficient")
        self.assertFalse(result.grounded)

    def test_sufficient_evidence_calls_llm_and_builds_citation(self) -> None:
        hits = [make_hit("h1", "Sinh viên bị cảnh báo học tập khi không đạt đủ tín chỉ.")]
        generator = FakeLLMGenerator(
            GeneratedGroundedAnswer(
                answer="Sinh viên bị cảnh báo học tập khi không đạt đủ tín chỉ.",
                used_evidence_ids=["E1"],
                abstain=False,
                reason=None,
                provider="dummy",
                model="dummy",
            )
        )
        pipeline = GroundedQAPipeline(
            retrieval_service=FakeRetrievalService(hits),  # type: ignore[arg-type]
            router=FakeRouter(),
            retrieval_planner=FakePlanner(),  # type: ignore[arg-type]
            evidence_checker=FakeEvidenceChecker(sufficient_evidence("h1")),  # type: ignore[arg-type]
            answer_generator=generator,
        )

        result = pipeline.answer("Sinh viên bị cảnh báo học tập trong những trường hợp nào?")

        self.assertEqual(generator.calls, 1)
        self.assertEqual(result.decision, "answer")
        self.assertEqual(result.final_answer_source, "llm_grounded")
        self.assertEqual(result.citations[0]["evidence_id"], "E1")
        self.assertTrue(result.validation_passed)

    def test_pipeline_preserves_evidence_checker_selection_order(self) -> None:
        hits = [
            make_hit("row", "Hàng Điểm chữ quy đổi, cột 7,0÷7,9: B.", block_type="table"),
            make_hit(
                "table",
                "| Điểm học phần theo thang 10 | 7,0÷7,9 |\n| --- | --- |\n| Điểm chữ quy đổi | B |",
                block_type="table",
                metadata={"citation_target": "table", "table_id": "grade"},
            ),
        ]
        generator = FakeLLMGenerator(
            GeneratedGroundedAnswer(answer="B", used_evidence_ids=["E1"], abstain=False, reason=None)
        )
        pipeline = GroundedQAPipeline(
            retrieval_service=FakeRetrievalService(hits),  # type: ignore[arg-type]
            router=FakeRouter(),
            retrieval_planner=FakePlanner(),  # type: ignore[arg-type]
            evidence_checker=FakeEvidenceChecker(sufficient_evidence("table", "row")),  # type: ignore[arg-type]
            answer_generator=generator,
        )

        result = pipeline.answer("Điểm 7.0 tương ứng với điểm chữ nào?")

        self.assertEqual(result.context_evidence[0]["chunk_id"], "table")
        self.assertEqual(result.context_evidence[1]["chunk_id"], "row")
        self.assertEqual(result.citations[0]["chunk_id"], "table")

    def test_llm_abstain_returns_abstain(self) -> None:
        hits = [make_hit("h1", "Some relevant text.")]
        generator = FakeLLMGenerator(
            GeneratedGroundedAnswer(answer="", used_evidence_ids=[], abstain=True, reason="not enough")
        )
        pipeline = GroundedQAPipeline(
            retrieval_service=FakeRetrievalService(hits),  # type: ignore[arg-type]
            router=FakeRouter(),
            retrieval_planner=FakePlanner(),  # type: ignore[arg-type]
            evidence_checker=FakeEvidenceChecker(sufficient_evidence("h1")),  # type: ignore[arg-type]
            answer_generator=generator,
        )

        result = pipeline.answer("Question?")

        self.assertEqual(result.decision, "abstain")
        self.assertEqual(result.final_answer_source, "llm_abstain")

    def test_invalid_llm_evidence_id_fails_validation(self) -> None:
        hits = [make_hit("h1", "The answer is 12 credits.")]
        generator = FakeLLMGenerator(
            GeneratedGroundedAnswer(answer="12 credits", used_evidence_ids=["E99"], abstain=False, reason=None)
        )
        pipeline = GroundedQAPipeline(
            retrieval_service=FakeRetrievalService(hits),  # type: ignore[arg-type]
            router=FakeRouter(),
            retrieval_planner=FakePlanner(),  # type: ignore[arg-type]
            evidence_checker=FakeEvidenceChecker(sufficient_evidence("h1")),  # type: ignore[arg-type]
            answer_generator=generator,
        )

        result = pipeline.answer("How many credits?")

        self.assertEqual(result.decision, "abstain")
        self.assertEqual(result.final_answer_source, "validation_failed")
        self.assertEqual(result.validation_reason, "unknown_evidence_id")

    def test_llm_error_does_not_fallback_to_extractive(self) -> None:
        hits = [make_hit("h1", "The answer is 12 credits.")]
        generator = FakeLLMGenerator(
            GeneratedGroundedAnswer(answer="", used_evidence_ids=[], abstain=True, reason="llm_error: boom")
        )
        pipeline = GroundedQAPipeline(
            retrieval_service=FakeRetrievalService(hits),  # type: ignore[arg-type]
            router=FakeRouter(),
            retrieval_planner=FakePlanner(),  # type: ignore[arg-type]
            evidence_checker=FakeEvidenceChecker(sufficient_evidence("h1")),  # type: ignore[arg-type]
            answer_generator=generator,
        )

        result = pipeline.answer("How many credits?")

        self.assertEqual(result.decision, "abstain")
        self.assertEqual(result.final_answer_source, "llm_error")
        self.assertNotIn("12 credits", result.answer)

    def test_table_answer_keeps_cell_citation(self) -> None:
        hits = [
            make_hit(
                "h1",
                "Điểm 8,2 tương ứng điểm chữ B+ và điểm hệ 4 là 3,5.",
                block_type="table",
                metadata={
                    "citation_target": "cell",
                    "table_id": "grade_table",
                    "row_header": "8,0-8,4",
                    "col_header": "Điểm chữ; Điểm hệ 4",
                    "cell_text": "B+; 3,5",
                },
            )
        ]
        generator = FakeLLMGenerator(
            GeneratedGroundedAnswer(answer="B+ và 3,5.", used_evidence_ids=["E1"], abstain=False, reason=None)
        )
        pipeline = GroundedQAPipeline(
            retrieval_service=FakeRetrievalService(hits),  # type: ignore[arg-type]
            router=FakeRouter(),
            retrieval_planner=FakePlanner(),  # type: ignore[arg-type]
            evidence_checker=FakeEvidenceChecker(sufficient_evidence("h1")),  # type: ignore[arg-type]
            answer_generator=generator,
        )

        result = pipeline.answer("Điểm 8,2 tương ứng với điểm chữ và điểm hệ 4 nào?")

        self.assertEqual(result.decision, "answer")
        self.assertEqual(result.citations[0]["citation_target"], "cell")
        self.assertEqual(result.citations[0]["metadata"]["cell_text"], "B+; 3,5")

    def test_extracting_baseline_still_runs(self) -> None:
        hits = [make_hit("h1", "The model uses 8 heads.")]
        pipeline = GroundedQAPipeline(
            retrieval_service=FakeRetrievalService(hits),  # type: ignore[arg-type]
            router=FakeRouter(),
            retrieval_planner=FakePlanner(),  # type: ignore[arg-type]
            evidence_checker=FakeEvidenceChecker(sufficient_evidence("h1")),  # type: ignore[arg-type]
            answer_generator=GroundedAnswerGenerator(),
        )

        result = pipeline.answer("How many heads does the model use?")

        self.assertEqual(result.generator_type, "extractive")
        self.assertIn("8", result.answer)


if __name__ == "__main__":
    unittest.main()
