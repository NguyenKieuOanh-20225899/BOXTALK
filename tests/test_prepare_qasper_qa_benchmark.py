from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_qa import answer_match, evidence_match
from scripts.prepare_qasper_qa_benchmark import prepare_qasper_qa


def write_mock_qasper(path: Path) -> None:
    payload = {
        "paper-a": {
            "title": "A Scientific QA Paper",
            "abstract": "This paper introduces a grounded QA model.",
            "full_text": [
                {
                    "section_name": "Method",
                    "paragraphs": [
                        "The model retrieves evidence sentences before generating an answer.",
                        "The ablation shows that reranking improves citation quality.",
                    ],
                }
            ],
            "qas": [
                {
                    "question_id": "q1",
                    "question": "What does the model retrieve before answering?",
                    "answers": [
                        {
                            "answer": {
                                "unanswerable": False,
                                "extractive_spans": ["evidence sentences"],
                                "free_form_answer": "It retrieves evidence sentences.",
                                "yes_no": None,
                                "evidence": [
                                    "The model retrieves evidence sentences before generating an answer."
                                ],
                            }
                        }
                    ],
                },
                {
                    "question_id": "q2",
                    "question": "What dataset was used for human evaluation?",
                    "answers": [
                        {
                            "answer": {
                                "unanswerable": True,
                                "extractive_spans": [],
                                "free_form_answer": "",
                                "yes_no": None,
                                "evidence": [],
                            }
                        }
                    ],
                },
            ],
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_prepare_qasper_outputs_chunks_queries_and_evidence_mapping(tmp_path: Path) -> None:
    input_file = tmp_path / "qasper-dev-v0.3.json"
    write_mock_qasper(input_file)

    manifest = prepare_qasper_qa(
        output_dir=tmp_path / "out",
        split="validation",
        limit=None,
        source="json",
        input_file=input_file,
        seed=1,
    )

    chunks = load_jsonl(tmp_path / "out" / "qasper.jsonl")
    queries = sorted(load_jsonl(tmp_path / "out" / "queries.jsonl"), key=lambda row: row["id"])

    assert manifest["query_count"] == 2
    assert manifest["answerable_count"] == 1
    assert manifest["unanswerable_count"] == 1
    assert chunks
    assert queries[0]["gold_answers"] == ["evidence sentences", "It retrieves evidence sentences."]
    assert queries[0]["expected_chunk_ids"]
    assert queries[0]["gold_evidence_texts"] == [
        "The model retrieves evidence sentences before generating an answer."
    ]
    assert queries[1]["should_answer"] is False
    assert queries[1]["expected_decision"] == "abstain"


def test_qasper_multiple_gold_answers_work_with_benchmark_answer_match() -> None:
    ok, f1, contains = answer_match(
        {"gold_answers": ["wrong option", "It retrieves evidence sentences."]},
        "The answer is that it retrieves evidence sentences.",
        min_token_f1=0.45,
    )

    assert ok is True
    assert f1 > 0.45
    assert contains is True


def test_qasper_yes_no_and_evidence_text_matching() -> None:
    ok, _, _ = answer_match({"gold_answers": ["yes"]}, "Yes, the model uses reranking.", min_token_f1=0.45)
    assert ok is True

    result = {
        "retrieved_hits": [
            {
                "chunk_id": "qasper:paper-a:0001",
                "source_name": "qasper",
                "section": "Method",
                "text": "The model retrieves evidence sentences before generating an answer.",
                "snippet": "",
            }
        ],
        "citations": [],
    }
    assert evidence_match(
        {
            "source_name": "qasper",
            "gold_evidence_texts": ["retrieves evidence sentences before generating an answer"],
        },
        result,
    )
