from __future__ import annotations

from app.retrieval.reranker import HeuristicReranker
from app.retrieval.schemas import DocumentChunkRef, RetrievedHit


def _hit(chunk: DocumentChunkRef, score: float) -> RetrievedHit:
    return RetrievedHit(chunk=chunk, score=score, source="bm25", final_score=score)


def test_vietnamese_policy_reranker_prefers_subject_registration_over_graduation_noise() -> None:
    reranker = HeuristicReranker()
    query = "Kỳ thi tốt nghiệp THPT tổ chức những môn thi nào?"
    correct = DocumentChunkRef(
        chunk_id="doc:correct",
        text=(
            "Thí sinh phải đăng ký dự thi môn Ngữ văn, môn Toán và 01 bài thi tự chọn "
            "gồm 02 môn thi trong số các môn: Vật lí, Hóa học, Sinh học, Lịch sử, "
            "Địa lí, Giáo dục kinh tế và pháp luật, Tin học, Công nghệ, Ngoại ngữ."
        ),
        section="2. Đăng ký môn thi:",
        heading_path=["Điều 20. Đăng ký dự thi", "2. Đăng ký môn thi:"],
    )
    noisy = DocumentChunkRef(
        chunk_id="doc:noisy",
        text="Những thí sinh đủ điều kiện dự thi được công nhận tốt nghiệp THPT.",
        section="Điều 45. Công nhận tốt nghiệp THPT",
        heading_path=["Điều 45. Công nhận tốt nghiệp THPT"],
    )

    assert reranker.score(query, correct) > reranker.score(query, noisy)

    reranked = reranker.rerank(query, [_hit(noisy, 0.50), _hit(correct, 0.25)], top_n=2)
    assert reranked[0].chunk_id == "doc:correct"


def test_vietnamese_policy_reranker_keeps_article_and_heading_matches() -> None:
    reranker = HeuristicReranker()
    query = "Ban coi thi gồm những thành phần nào?"
    correct = DocumentChunkRef(
        chunk_id="doc:ban-coi-thi",
        text="a) Trưởng ban; b) Phó Trưởng ban; c) Ủy viên, thư ký.",
        section="1. Thành phần:",
        heading_path=["Điều 13. Ban Coi thi", "1. Thành phần:"],
    )
    noisy = DocumentChunkRef(
        chunk_id="doc:ban-cham-thi",
        text="a) Trưởng ban Chấm thi; b) Phó Trưởng ban Chấm thi.",
        section="1. Thành phần:",
        heading_path=["Điều 17. Ban Chấm thi", "1. Thành phần:"],
    )

    assert reranker.score(query, correct) > reranker.score(query, noisy)
