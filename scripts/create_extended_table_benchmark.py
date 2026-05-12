from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "table_reasoning_benchmark"
SOURCE_NAME = "table_reasoning_reference.pdf"
CHUNK_PREFIX = "table_reasoning_reference_chunks"


def chunk_id(name: str) -> str:
    return f"{CHUNK_PREFIX}:{name}"


def table_text(headers: list[str], rows: list[dict[str, str]]) -> str:
    lines = [" | ".join(headers)]
    for row in rows:
        lines.append(" | ".join(row.get(header, "") for header in headers))
    return "\n".join(lines)


def paragraph_chunk(name: str, *, page: int, section: str, text: str) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id(name),
        "doc_id": SOURCE_NAME,
        "source_name": SOURCE_NAME,
        "page": page,
        "section": section,
        "heading_path": ["Extended Table Reasoning Benchmark", section],
        "block_type": "paragraph",
        "text": text,
    }


def table_chunk(
    name: str,
    *,
    page: int,
    section: str,
    headers: list[str],
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id(name),
        "doc_id": SOURCE_NAME,
        "source_name": SOURCE_NAME,
        "page": page,
        "section": section,
        "heading_path": ["Extended Table Reasoning Benchmark", section],
        "block_type": "table",
        "text": table_text(headers, rows),
        "metadata": {
            "is_table_chunk": True,
            "table_rows": rows,
        },
    }


GRADE_ROWS = [
    {"Score Range": "9.0 - 10.0", "Letter Grade": "A+", "Grade Point": "4.0", "Classification": "Excellent distinction"},
    {"Score Range": "8.5 - 8.9", "Letter Grade": "A", "Grade Point": "3.8", "Classification": "Excellent"},
    {"Score Range": "8.0 - 8.4", "Letter Grade": "B+", "Grade Point": "3.5", "Classification": "Very good"},
    {"Score Range": "7.0 - 7.9", "Letter Grade": "B", "Grade Point": "3.0", "Classification": "Good"},
    {"Score Range": "6.5 - 6.9", "Letter Grade": "C+", "Grade Point": "2.5", "Classification": "Satisfactory"},
    {"Score Range": "5.5 - 6.4", "Letter Grade": "C", "Grade Point": "2.0", "Classification": "Adequate"},
    {"Score Range": "4.0 - 5.4", "Letter Grade": "D", "Grade Point": "1.0", "Classification": "Conditional pass"},
    {"Score Range": "Below 4.0", "Letter Grade": "F", "Grade Point": "0.0", "Classification": "Fail"},
]

DECIMAL_COMMA_ROWS = [
    {"Khoang diem": "8,5 - 10,0", "Diem chu": "A", "Diem he": "4,0", "Xep loai": "Excellent"},
    {"Khoang diem": "7,0 - 8,4", "Diem chu": "B+", "Diem he": "3,5", "Xep loai": "Very good"},
    {"Khoang diem": "6,5 - 6,9", "Diem chu": "C+", "Diem he": "2,5", "Xep loai": "Satisfactory"},
    {"Khoang diem": "5,5 - 6,4", "Diem chu": "C", "Diem he": "2,0", "Xep loai": "Adequate"},
]

COMPLETION_ROWS = [
    {"Completion Rate": "90% - 100%", "Action Level": "Full compliance", "Owner": "Compliance team", "Cadence": "Quarterly"},
    {"Completion Rate": "75% - 89%", "Action Level": "Watch list", "Owner": "Department lead", "Cadence": "Monthly"},
    {"Completion Rate": "50% - 74%", "Action Level": "Corrective action", "Owner": "Program owner", "Cadence": "Biweekly"},
    {"Completion Rate": "Below 50%", "Action Level": "Escalation required", "Owner": "Director", "Cadence": "Weekly"},
]

MODEL_ROWS = [
    {"Model": "Base", "Heads": "8", "Layers": "6", "d_model": "512", "BLEU": "27.3"},
    {"Model": "Large", "Heads": "16", "Layers": "12", "d_model": "1024", "BLEU": "29.8"},
    {"Model": "Distilled", "Heads": "8", "Layers": "4", "d_model": "384", "BLEU": "26.1"},
]

FINANCE_ROWS = [
    {"Program": "Alpha", "Budget": "120", "Actual": "132", "Variance": "12", "Margin": "18%"},
    {"Program": "Beta", "Budget": "80", "Actual": "72", "Variance": "-8", "Margin": "12%"},
    {"Program": "Gamma", "Budget": "200", "Actual": "230", "Variance": "30", "Margin": "22%"},
    {"Program": "Delta", "Budget": "150", "Actual": "150", "Variance": "0", "Margin": "15%"},
]

AWARD_ROWS = [
    {"Award": "Merit Award", "Minimum GPA": "3.5", "Minimum Credits": "30", "Amount": "1200"},
    {"Award": "Research Grant", "Minimum GPA": "3.2", "Minimum Credits": "24", "Amount": "1800"},
    {"Award": "Lab Grant", "Minimum GPA": "3.0", "Minimum Credits": "18", "Amount": "900"},
]

SLA_ROWS = [
    {"Tier": "Standard", "Severity": "High", "Response Target": "4 hours", "Credit Percent": "5%"},
    {"Tier": "Standard", "Severity": "Critical", "Response Target": "1 hour", "Credit Percent": "10%"},
    {"Tier": "Premium", "Severity": "High", "Response Target": "2 hours", "Credit Percent": "8%"},
    {"Tier": "Premium", "Severity": "Critical", "Response Target": "30 minutes", "Credit Percent": "15%"},
]

CHUNKS: list[dict[str, Any]] = [
    paragraph_chunk(
        "grade_intro",
        page=1,
        section="Grade Conversion Matrix",
        text="The grade conversion matrix maps exact score ranges to letter grades, grade points, and classifications.",
    ),
    table_chunk("grade_table", page=1, section="Grade Conversion Matrix", headers=list(GRADE_ROWS[0]), rows=GRADE_ROWS),
    table_chunk(
        "grade_decimal_comma_table",
        page=1,
        section="Grade Conversion Matrix Decimal Comma Variant",
        headers=list(DECIMAL_COMMA_ROWS[0]),
        rows=DECIMAL_COMMA_ROWS,
    ),
    paragraph_chunk(
        "completion_intro",
        page=2,
        section="Completion Escalation Matrix",
        text="The completion escalation matrix maps completion bands to action level, owner, and review cadence.",
    ),
    table_chunk(
        "completion_table",
        page=2,
        section="Completion Escalation Matrix",
        headers=list(COMPLETION_ROWS[0]),
        rows=COMPLETION_ROWS,
    ),
    table_chunk("model_table", page=3, section="Model Comparison Table", headers=list(MODEL_ROWS[0]), rows=MODEL_ROWS),
    paragraph_chunk(
        "finance_policy",
        page=4,
        section="Program Finance Review Policy",
        text=(
            "For program finance reviews, an absolute variance greater than 10 requires CFO review. "
            "A variance of 10 or less is reviewed by the department owner."
        ),
    ),
    table_chunk("finance_table", page=4, section="Program Finance Review Policy", headers=list(FINANCE_ROWS[0]), rows=FINANCE_ROWS),
    paragraph_chunk(
        "award_policy",
        page=5,
        section="Student Award Eligibility",
        text=(
            "Students on probation cannot receive the Merit Award even if table thresholds are met. "
            "The Lab Grant also requires department endorsement."
        ),
    ),
    table_chunk("award_table", page=5, section="Student Award Eligibility", headers=list(AWARD_ROWS[0]), rows=AWARD_ROWS),
    paragraph_chunk(
        "sla_policy",
        page=6,
        section="Support SLA Matrix",
        text="For regulated customers, Premium incidents add one business day to the response target; credit percentages are unchanged.",
    ),
    table_chunk("sla_table", page=6, section="Support SLA Matrix", headers=list(SLA_ROWS[0]), rows=SLA_ROWS),
]


def query(
    query_id: str,
    question: str,
    *,
    table_reasoning_type: str,
    benchmark_family: str,
    gold_answer: str,
    match_text: str,
    expected_chunk_ids: list[str],
    expected_pages: list[int],
    query_type: str = "factoid",
    expected_modality: str = "table",
    expected_fallback_mode: str = "table",
    should_require_fallback: bool = True,
    weak_standard_answer_case: bool = True,
) -> dict[str, Any]:
    return {
        "id": query_id,
        "query_id": query_id,
        "query_type": query_type,
        "question": question,
        "fallback_category": f"table_{table_reasoning_type}",
        "benchmark_family": benchmark_family,
        "table_reasoning_type": table_reasoning_type,
        "weak_standard_answer_case": weak_standard_answer_case,
        "expected_modality": expected_modality,
        "expected_fallback_mode": expected_fallback_mode,
        "should_require_fallback": should_require_fallback,
        "expected_chunk_ids": expected_chunk_ids,
        "expected_pages": expected_pages,
        "gold_answer": gold_answer,
        "match_text": match_text,
        "should_answer": True,
        "source_name": SOURCE_NAME,
    }


QUERIES: list[dict[str, Any]] = [
    query("simple_lookup_001", "6.5 is C or C+?", table_reasoning_type="simple_lookup", benchmark_family="wikitable_like", gold_answer="6.5 corresponds to C+.", match_text="C+", expected_chunk_ids=[chunk_id("grade_table")], expected_pages=[1], should_require_fallback=False, weak_standard_answer_case=False),
    query("simple_lookup_002", "What grade point corresponds to A+?", table_reasoning_type="simple_lookup", benchmark_family="wikitable_like", gold_answer="A+ corresponds to grade point 4.0.", match_text="4.0", expected_chunk_ids=[chunk_id("grade_table")], expected_pages=[1], should_require_fallback=False, weak_standard_answer_case=False),
    query("simple_lookup_003", "Which action level maps to 75% completion?", table_reasoning_type="simple_lookup", benchmark_family="wikitable_like", gold_answer="75% completion maps to Watch list.", match_text="Watch list", expected_chunk_ids=[chunk_id("completion_table")], expected_pages=[2], should_require_fallback=False, weak_standard_answer_case=False),
    query("simple_lookup_004", "Which model has BLEU 26.1?", table_reasoning_type="simple_lookup", benchmark_family="wikitable_like", gold_answer="The Distilled model has BLEU 26.1.", match_text="Distilled", expected_chunk_ids=[chunk_id("model_table")], expected_pages=[3], should_require_fallback=False, weak_standard_answer_case=False),
    query("simple_lookup_005", "Premium High severity has which credit percent?", table_reasoning_type="simple_lookup", benchmark_family="wikitable_like", gold_answer="Premium High severity has credit percent 8%.", match_text="8%", expected_chunk_ids=[chunk_id("sla_table")], expected_pages=[6]),
    query("reverse_lookup_001", "What score range gives B+?", table_reasoning_type="reverse_lookup", benchmark_family="wikitable_like", gold_answer="B+ corresponds to the 8.0 - 8.4 score range.", match_text="8.0 - 8.4", expected_chunk_ids=[chunk_id("grade_table")], expected_pages=[1]),
    query("reverse_lookup_002", "Diem chu C+ tuong ung khoang diem nao trong bang dau phay?", table_reasoning_type="reverse_lookup", benchmark_family="wikitable_like", gold_answer="C+ corresponds to the 6.5 - 6.9 score range.", match_text="6.5 - 6.9", expected_chunk_ids=[chunk_id("grade_decimal_comma_table")], expected_pages=[1]),
    query("reverse_lookup_003", "Which completion band has Director as owner?", table_reasoning_type="reverse_lookup", benchmark_family="wikitable_like", gold_answer="The Below 50% band has Director as owner.", match_text="Below 50%", expected_chunk_ids=[chunk_id("completion_table")], expected_pages=[2]),
    query("reverse_lookup_004", "Which award has amount 1800?", table_reasoning_type="reverse_lookup", benchmark_family="wikitable_like", gold_answer="The Research Grant has amount 1800.", match_text="Research Grant", expected_chunk_ids=[chunk_id("award_table")], expected_pages=[5]),
    query("reverse_lookup_005", "Which severity and tier have a 15% credit?", table_reasoning_type="reverse_lookup", benchmark_family="wikitable_like", gold_answer="Premium Critical has a 15% credit.", match_text="Premium Critical", expected_chunk_ids=[chunk_id("sla_table")], expected_pages=[6], query_type="comparison"),
    query("interval_mapping_001", "7.0 belongs to which grade interval?", table_reasoning_type="interval_mapping", benchmark_family="wikitable_like", gold_answer="7.0 falls in 7.0 - 7.9 and maps to B.", match_text="B", expected_chunk_ids=[chunk_id("grade_table")], expected_pages=[1]),
    query("interval_mapping_002", "50% completion corresponds to which action level?", table_reasoning_type="interval_mapping", benchmark_family="wikitable_like", gold_answer="50% falls in 50% - 74%, which maps to Corrective action.", match_text="Corrective action", expected_chunk_ids=[chunk_id("completion_table")], expected_pages=[2]),
    query("interval_mapping_003", "88% completion falls under which cadence?", table_reasoning_type="interval_mapping", benchmark_family="wikitable_like", gold_answer="88% falls in 75% - 89%, which has Monthly cadence.", match_text="Monthly", expected_chunk_ids=[chunk_id("completion_table")], expected_pages=[2]),
    query("interval_mapping_004", "8,2 tuong ung diem chu nao trong bang dau phay thap phan?", table_reasoning_type="interval_mapping", benchmark_family="wikitable_like", gold_answer="8.2 falls in 7.0 - 8.4 and maps to B+.", match_text="B+", expected_chunk_ids=[chunk_id("grade_decimal_comma_table")], expected_pages=[1]),
    query("interval_mapping_005", "3.8 belongs to which grade?", table_reasoning_type="interval_mapping", benchmark_family="wikitable_like", gold_answer="3.8 is below 4.0 and maps to F.", match_text="F", expected_chunk_ids=[chunk_id("grade_table")], expected_pages=[1]),
    query("multi_column_lookup_001", "For B+, return both score range and grade point.", table_reasoning_type="multi_column_lookup", benchmark_family="wikitable_like", gold_answer="B+ corresponds to score range 8.0 - 8.4 and grade point 3.5.", match_text="3.5", expected_chunk_ids=[chunk_id("grade_table")], expected_pages=[1]),
    query("multi_column_lookup_002", "Which model uses 12 layers and what d_model does it use?", table_reasoning_type="multi_column_lookup", benchmark_family="wikitable_like", gold_answer="The Large model uses 12 layers and d_model 1024.", match_text="1024", expected_chunk_ids=[chunk_id("model_table")], expected_pages=[3]),
    query("multi_column_lookup_003", "For the 50% - 74% completion band, who owns it and how often is it reviewed?", table_reasoning_type="multi_column_lookup", benchmark_family="wikitable_like", gold_answer="The 50% - 74% band is owned by the Program owner and reviewed biweekly.", match_text="Biweekly", expected_chunk_ids=[chunk_id("completion_table")], expected_pages=[2]),
    query("multi_column_lookup_004", "Which award requires 24 minimum credits and what amount does it provide?", table_reasoning_type="multi_column_lookup", benchmark_family="wikitable_like", gold_answer="Research Grant requires 24 minimum credits and provides amount 1800.", match_text="1800", expected_chunk_ids=[chunk_id("award_table")], expected_pages=[5]),
    query("multi_column_lookup_005", "For Premium Critical, return response target and credit percent.", table_reasoning_type="multi_column_lookup", benchmark_family="wikitable_like", gold_answer="Premium Critical has response target 30 minutes and credit percent 15%.", match_text="30 minutes", expected_chunk_ids=[chunk_id("sla_table")], expected_pages=[6]),
    query("multi_column_lookup_006", "For Gamma, return actual value and margin.", table_reasoning_type="multi_column_lookup", benchmark_family="wikitable_like", gold_answer="Gamma has actual value 230 and margin 22%.", match_text="22%", expected_chunk_ids=[chunk_id("finance_table")], expected_pages=[4]),
    query("boundary_case_001", "At exactly 8.5, which letter grade applies?", table_reasoning_type="boundary_case", benchmark_family="wikitable_like", gold_answer="At exactly 8.5, the letter grade is A.", match_text="A", expected_chunk_ids=[chunk_id("grade_table")], expected_pages=[1]),
    query("boundary_case_002", "At exactly 8.4, is the grade B+ or A?", table_reasoning_type="boundary_case", benchmark_family="wikitable_like", gold_answer="At exactly 8.4, the grade is B+.", match_text="B+", expected_chunk_ids=[chunk_id("grade_table")], expected_pages=[1]),
    query("boundary_case_003", "At exactly 4.0, is the grade D or F?", table_reasoning_type="boundary_case", benchmark_family="wikitable_like", gold_answer="At exactly 4.0, the grade is D.", match_text="D", expected_chunk_ids=[chunk_id("grade_table")], expected_pages=[1]),
    query("boundary_case_004", "At exactly 89% completion, what action level applies?", table_reasoning_type="boundary_case", benchmark_family="wikitable_like", gold_answer="At exactly 89%, the action level is Watch list.", match_text="Watch list", expected_chunk_ids=[chunk_id("completion_table")], expected_pages=[2]),
    query("boundary_case_005", "At exactly 90% completion, what action level applies?", table_reasoning_type="boundary_case", benchmark_family="wikitable_like", gold_answer="At exactly 90%, the action level is Full compliance.", match_text="Full compliance", expected_chunk_ids=[chunk_id("completion_table")], expected_pages=[2]),
    query("boundary_case_006", "At 6,5 in the decimal-comma table, which letter grade applies?", table_reasoning_type="boundary_case", benchmark_family="wikitable_like", gold_answer="At 6.5, the decimal-comma table maps to C+.", match_text="C+", expected_chunk_ids=[chunk_id("grade_decimal_comma_table")], expected_pages=[1]),
    query("table_text_reasoning_001", "Alpha has variance 12. According to the policy text, who reviews it?", table_reasoning_type="table_text_reasoning", benchmark_family="tatqa_like", gold_answer="Alpha's variance is 12, which is greater than 10, so CFO review is required.", match_text="CFO review", expected_chunk_ids=[chunk_id("finance_table"), chunk_id("finance_policy")], expected_pages=[4], expected_modality="table_text", expected_fallback_mode="multi_span", query_type="comparison"),
    query("table_text_reasoning_002", "Beta has variance -8. Does it require CFO review or department owner review?", table_reasoning_type="table_text_reasoning", benchmark_family="tatqa_like", gold_answer="Beta's absolute variance is 8, so it is reviewed by the department owner.", match_text="department owner", expected_chunk_ids=[chunk_id("finance_table"), chunk_id("finance_policy")], expected_pages=[4], expected_modality="table_text", expected_fallback_mode="multi_span", query_type="comparison"),
    query("table_text_reasoning_003", "A student has GPA 3.6 and 32 credits but is on probation. Can they receive the Merit Award?", table_reasoning_type="table_text_reasoning", benchmark_family="tatqa_like", gold_answer="No. The table thresholds are met, but the policy says students on probation cannot receive the Merit Award.", match_text="No", expected_chunk_ids=[chunk_id("award_table"), chunk_id("award_policy")], expected_pages=[5], expected_modality="table_text", expected_fallback_mode="multi_span", query_type="verification"),
    query("table_text_reasoning_004", "What extra requirement applies to Lab Grant besides the table thresholds?", table_reasoning_type="table_text_reasoning", benchmark_family="tatqa_like", gold_answer="The Lab Grant requires department endorsement in addition to the table thresholds.", match_text="department endorsement", expected_chunk_ids=[chunk_id("award_table"), chunk_id("award_policy")], expected_pages=[5], expected_modality="table_text", expected_fallback_mode="multi_span"),
    query("table_text_reasoning_005", "For a regulated customer with Premium Critical incident, what response-target adjustment applies?", table_reasoning_type="table_text_reasoning", benchmark_family="tatqa_like", gold_answer="Premium Critical has a 30 minute response target, and regulated customers add one business day to the response target.", match_text="one business day", expected_chunk_ids=[chunk_id("sla_table"), chunk_id("sla_policy")], expected_pages=[6], expected_modality="table_text", expected_fallback_mode="multi_span"),
    query("numerical_reasoning_001", "What is the difference between Gamma actual and Alpha actual?", table_reasoning_type="numerical_reasoning", benchmark_family="tatqa_like", gold_answer="Gamma actual is 230 and Alpha actual is 132, so the difference is 98.", match_text="98", expected_chunk_ids=[chunk_id("finance_table")], expected_pages=[4], query_type="comparison"),
    query("numerical_reasoning_002", "Which program has the largest variance and what is that variance?", table_reasoning_type="numerical_reasoning", benchmark_family="tatqa_like", gold_answer="Gamma has the largest variance, 30.", match_text="Gamma", expected_chunk_ids=[chunk_id("finance_table")], expected_pages=[4], query_type="comparison"),
    query("numerical_reasoning_003", "What is the total actual value for Alpha and Beta?", table_reasoning_type="numerical_reasoning", benchmark_family="tatqa_like", gold_answer="Alpha actual 132 plus Beta actual 72 equals 204.", match_text="204", expected_chunk_ids=[chunk_id("finance_table")], expected_pages=[4], query_type="comparison"),
    query("numerical_reasoning_004", "How many more heads does Large use than Base?", table_reasoning_type="numerical_reasoning", benchmark_family="tatqa_like", gold_answer="Large uses 16 heads and Base uses 8, so Large uses 8 more heads.", match_text="8 more", expected_chunk_ids=[chunk_id("model_table")], expected_pages=[3], query_type="comparison"),
    query("numerical_reasoning_005", "Which program has the highest margin?", table_reasoning_type="numerical_reasoning", benchmark_family="tatqa_like", gold_answer="Gamma has the highest margin at 22%.", match_text="Gamma", expected_chunk_ids=[chunk_id("finance_table")], expected_pages=[4], query_type="comparison"),
    query("numerical_reasoning_006", "What is the budget difference between Gamma and Beta?", table_reasoning_type="numerical_reasoning", benchmark_family="tatqa_like", gold_answer="Gamma budget 200 minus Beta budget 80 equals 120.", match_text="120", expected_chunk_ids=[chunk_id("finance_table")], expected_pages=[4], query_type="comparison"),
    query("fact_verification_001", "True or false: C+ corresponds to grade point 2.5.", table_reasoning_type="fact_verification", benchmark_family="tabfact_like", gold_answer="True. C+ corresponds to grade point 2.5.", match_text="True", expected_chunk_ids=[chunk_id("grade_table")], expected_pages=[1], query_type="verification"),
    query("fact_verification_002", "True or false: 6.5 belongs to C, not C+.", table_reasoning_type="fact_verification", benchmark_family="tabfact_like", gold_answer="False. 6.5 belongs to C+.", match_text="False", expected_chunk_ids=[chunk_id("grade_table")], expected_pages=[1], query_type="verification"),
    query("fact_verification_003", "True or false: Completion of 50% requires escalation.", table_reasoning_type="fact_verification", benchmark_family="tabfact_like", gold_answer="False. 50% maps to Corrective action.", match_text="False", expected_chunk_ids=[chunk_id("completion_table")], expected_pages=[2], query_type="verification"),
    query("fact_verification_004", "True or false: Premium Critical has a 15% credit.", table_reasoning_type="fact_verification", benchmark_family="tabfact_like", gold_answer="True. Premium Critical has a 15% credit.", match_text="True", expected_chunk_ids=[chunk_id("sla_table")], expected_pages=[6], query_type="verification"),
    query("fact_verification_005", "True or false: Beta has the largest actual value.", table_reasoning_type="fact_verification", benchmark_family="tabfact_like", gold_answer="False. Gamma has the largest actual value.", match_text="False", expected_chunk_ids=[chunk_id("finance_table")], expected_pages=[4], query_type="verification"),
    query("fact_verification_006", "True or false: Merit Award requires minimum GPA 3.5.", table_reasoning_type="fact_verification", benchmark_family="tabfact_like", gold_answer="True. Merit Award requires minimum GPA 3.5.", match_text="True", expected_chunk_ids=[chunk_id("award_table")], expected_pages=[5], query_type="verification"),
    query("fact_verification_007", "True or false: Large has d_model 512.", table_reasoning_type="fact_verification", benchmark_family="tabfact_like", gold_answer="False. Large has d_model 1024.", match_text="False", expected_chunk_ids=[chunk_id("model_table")], expected_pages=[3], query_type="verification"),
    query("fact_verification_008", "True or false: Gamma's variance requires CFO review under the policy.", table_reasoning_type="fact_verification", benchmark_family="tabfact_like", gold_answer="True. Gamma's variance is 30, which is greater than 10 and requires CFO review.", match_text="True", expected_chunk_ids=[chunk_id("finance_table"), chunk_id("finance_policy")], expected_pages=[4], expected_modality="table_text", expected_fallback_mode="multi_span", query_type="verification"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an extended internal benchmark for table reasoning over grounded PDF QA.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunks-name", default=f"{CHUNK_PREFIX}.jsonl")
    parser.add_argument("--queries-name", default="queries.jsonl")
    parser.add_argument("--manifest-name", default="manifest.json")
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_manifest(path: Path, *, chunks_path: Path, queries_path: Path) -> None:
    category_counts = Counter(str(row["fallback_category"]) for row in QUERIES)
    family_counts = Counter(str(row["benchmark_family"]) for row in QUERIES)
    reasoning_counts = Counter(str(row["table_reasoning_type"]) for row in QUERIES)
    modality_counts = Counter(str(row["expected_modality"]) for row in QUERIES)
    manifest = {
        "name": "extended_table_reasoning_benchmark",
        "description": (
            "Small internal table QA benchmark inspired by WikiTableQuestions, TAT-QA, and TabFact, "
            "adapted for evidence-first PDF grounded QA."
        ),
        "document": {
            "id": "table_reasoning_reference",
            "chunks_jsonl": str(chunks_path),
            "queries": str(queries_path),
            "index_dir": str(ROOT / "results" / "retrieval_index" / "table_reasoning_reference"),
            "source_name": SOURCE_NAME,
        },
        "query_count": len(QUERIES),
        "should_require_fallback_count": sum(1 for row in QUERIES if row["should_require_fallback"]),
        "weak_standard_answer_count": sum(1 for row in QUERIES if row["weak_standard_answer_case"]),
        "category_counts": dict(sorted(category_counts.items())),
        "benchmark_family_counts": dict(sorted(family_counts.items())),
        "table_reasoning_type_counts": dict(sorted(reasoning_counts.items())),
        "expected_modality_counts": dict(sorted(modality_counts.items())),
        "recommended_configs": {
            "standard": "routed_grounded",
            "fallback": "routed_grounded_with_llm_fallback",
        },
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = output_dir / args.chunks_name
    queries_path = output_dir / args.queries_name
    manifest_path = output_dir / args.manifest_name
    write_jsonl(chunks_path, CHUNKS)
    write_jsonl(queries_path, QUERIES)
    write_manifest(manifest_path, chunks_path=chunks_path, queries_path=queries_path)
    print(output_dir)


if __name__ == "__main__":
    main()
