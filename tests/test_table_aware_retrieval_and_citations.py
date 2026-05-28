from __future__ import annotations

from app.qa.citations import format_evidence_citation, table_citation_text
from app.retrieval.reranker import HeuristicReranker
from app.retrieval.schemas import DocumentChunkRef, RetrievedHit
from app.retrieval.table_aware import classify_table_query, table_aware_score


def _hit(chunk: DocumentChunkRef, score: float = 0.5) -> RetrievedHit:
    return RetrievedHit(chunk=chunk, score=score, source="bm25", final_score=score)


def test_table_lookup_boost_prefers_matching_cell(monkeypatch) -> None:
    monkeypatch.setenv("BOXBIIBOO_ENABLE_TABLE_AWARE_RETRIEVAL", "true")
    query = "C+ tương ứng bao nhiêu điểm thang 4?"
    table = DocumentChunkRef(
        chunk_id="cell-c-plus",
        text="Bảng quy đổi điểm, trang 9. Hàng C+, cột Điểm thang 4: 2.5.",
        block_type="table",
        metadata={
            "is_table_chunk": True,
            "table_id": "page_9_table_1",
            "row_header": "C+",
            "col_header": "Điểm thang 4",
            "cell_text": "2.5",
            "citation_target": "cell",
        },
    )
    paragraph = DocumentChunkRef(chunk_id="p1", text="C+ là một ký hiệu trong phụ lục.", block_type="paragraph")
    reranked = HeuristicReranker(blend_weight=0.9).rerank(query, [_hit(paragraph, 0.8), _hit(table, 0.2)], top_n=2)
    assert reranked[0].chunk_id == "cell-c-plus"
    assert table.metadata["table_retrieval_trace"]["row_matched"] == "C+"


def test_reverse_lookup_and_non_table_query() -> None:
    assert classify_table_query("Khoảng điểm nào quy đổi ra C+?") == "table_reverse_lookup"
    assert classify_table_query("Quy chế có hiệu lực từ ngày nào?") == "general"
    table = DocumentChunkRef(
        chunk_id="range",
        text="Hàng 8.0-8.4, cột Điểm chữ: C+.",
        block_type="table",
        metadata={"table_id": "t1", "cell_text": "C+", "citation_target": "cell"},
    )
    boost, trace = table_aware_score("Khoảng điểm nào quy đổi ra C+?", table)
    assert boost > 0.5
    assert trace.table_boost_applied


def test_table_retrieval_flag_off_preserves_order(monkeypatch) -> None:
    monkeypatch.delenv("BOXBIIBOO_ENABLE_TABLE_AWARE_RETRIEVAL", raising=False)
    query = "C+ tương ứng bao nhiêu điểm thang 4?"
    table = DocumentChunkRef(chunk_id="table", text="Hàng C+, cột Điểm thang 4: 2.5.", block_type="table")
    paragraph = DocumentChunkRef(chunk_id="paragraph", text="C+ là ký hiệu.", block_type="paragraph")
    reranked = HeuristicReranker().rerank(query, [_hit(paragraph, 0.8), _hit(table, 0.2)], top_n=2)
    assert reranked[0].chunk_id == "paragraph"


def test_paragraph_and_table_citations() -> None:
    paragraph_hit = _hit(DocumentChunkRef(chunk_id="p1", text="Text", page=2, block_type="paragraph"))
    assert "citation_text" not in format_evidence_citation(paragraph_hit)

    page_hit = _hit(DocumentChunkRef(chunk_id="t1", text="Table", page=3, block_type="table", metadata={"table_id": "page_3_table_1"}))
    assert format_evidence_citation(page_hit)["citation_text"] == "Trang 3, bảng page_3_table_1."

    row_hit = _hit(
        DocumentChunkRef(
            chunk_id="r1",
            text="Row",
            page=3,
            block_type="table",
            metadata={"table_id": "page_3_table_1", "caption": "Quy đổi", "row_header": "C+"},
        )
    )
    assert "hàng 'C+'" in format_evidence_citation(row_hit)["citation_text"]

    cell_hit = _hit(
        DocumentChunkRef(
            chunk_id="c1",
            text="Cell",
            page=3,
            block_type="table",
            metadata={
                "table_id": "page_3_table_1",
                "caption": "Quy đổi",
                "row_header": "C+",
                "col_header": "Điểm thang 4",
                "cell_text": "2.5",
            },
        )
    )
    citation = format_evidence_citation(cell_hit)
    assert citation["citation_text"] == "Trang 3, bảng 'Quy đổi', hàng 'C+', cột 'Điểm thang 4'."
    assert citation["cell_text"] == "2.5"
    assert table_citation_text({"page": 3, "table_id": "t2"}) == "Trang 3, bảng t2."
