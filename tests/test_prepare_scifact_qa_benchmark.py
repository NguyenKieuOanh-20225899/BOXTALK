from __future__ import annotations

import json
from pathlib import Path

from scripts.prepare_scifact_qa_benchmark import (
    choose_gold_evidence_sentence,
    load_qrels,
    prepare_scifact_qa,
)


def test_load_qrels_skips_header(tmp_path: Path) -> None:
    qrels = tmp_path / "test.tsv"
    qrels.write_text("query-id\tcorpus-id\tscore\n1\t10\t1\n1\t11\t0\n", encoding="utf-8")

    assert load_qrels(qrels) == {"1": [("10", 1), ("11", 0)]}


def test_choose_gold_evidence_sentence_uses_scifact_metadata() -> None:
    corpus = {
        "10": {
            "title": "Example",
            "text": "First sentence. Target evidence sentence. Third sentence.",
        }
    }

    doc_id, sentence = choose_gold_evidence_sentence(
        claim="target evidence",
        query_metadata={"10": [{"sentences": [1], "label": "SUPPORT"}]},
        relevant_ids=["10"],
        corpus=corpus,
    )

    assert doc_id == "10"
    assert sentence == "Target evidence sentence."


def test_prepare_scifact_qa_outputs_chunks_and_queries(tmp_path: Path) -> None:
    beir_dir = tmp_path / "beir"
    qrels_dir = beir_dir / "qrels"
    qrels_dir.mkdir(parents=True)
    (beir_dir / "corpus.jsonl").write_text(
        json.dumps(
            {
                "_id": "10",
                "title": "Vitamin B12 and homocysteine",
                "text": "Vitamin B12 deficiency increases homocysteine. Other sentence.",
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (beir_dir / "queries.jsonl").write_text(
        json.dumps(
            {
                "_id": "1",
                "text": "A deficiency of vitamin B12 increases blood levels of homocysteine.",
                "metadata": {"10": [{"sentences": [0], "label": "SUPPORT"}]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (qrels_dir / "test.tsv").write_text("query-id\tcorpus-id\tscore\n1\t10\t1\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    manifest = prepare_scifact_qa(beir_dir=beir_dir, output_dir=out_dir, split="test")

    assert manifest["query_count"] == 1
    assert (out_dir / "scifact.jsonl").exists()
    query = json.loads((out_dir / "queries_test.jsonl").read_text(encoding="utf-8").strip())
    assert query["expected_chunk_ids"] == ["scifact:10"]
    assert query["gold_answer"] == "Vitamin B12 deficiency increases homocysteine."
