# Final Results Summary 2026-05-13

## 1. Mục tiêu đồ án

Đề tài “Nghiên cứu các kĩ thuật truy xuất và hỏi đáp thông tin trên tài liệu PDF” xây dựng và đánh giá một hệ thống hỏi đáp trên tài liệu PDF theo hướng có căn cứ. Hệ thống tập trung vào ba vấn đề chính:

- Trích xuất nội dung PDF thành các đơn vị có thể lập chỉ mục.
- Truy xuất đoạn bằng kết hợp sparse/dense retrieval và routing theo loại câu hỏi.
- Sinh câu trả lời ngắn có dẫn chứng, hạn chế trả lời khi thiếu bằng chứng.

Mục tiêu không phải đạt SOTA trên mọi benchmark, mà là xây dựng một pipeline hoàn chỉnh, có đánh giá nhiều tầng và phân tích rõ giới hạn.

## 2. Pipeline chính

Luồng chính được dùng trong báo cáo:

```text
PDF ingest -> conditional hybrid_tatr table enhancement -> chunk/index -> retrieval -> routed_grounded -> grounded answer + citation
```

Các lựa chọn chính:

- Ingest backend chính: `default ingest backend` có thêm bước `hybrid_tatr` có điều kiện cho block/vùng bảng.
- QA path chính: `routed_grounded`.
- Câu trả lời chính: grounded QA có citation.
- LLM thật: không bật làm pipeline chính.

## 3. Nhánh thực nghiệm

Các nhánh sau được giữ ở mức thực nghiệm:

- `hybrid_tatr`: TATR geometry + OCR/PDF word boxes để cải thiện table structure ở tầng ingest. Module này đã được nối vào pipeline chính theo kiểu chỉ chạy cho bảng và luôn fallback.
- LLM fallback/explanation: chỉ dùng như hướng mở rộng, không phải lõi chính.
- QASPER top-k/rerank probe: dùng để phân tích hướng cải thiện retrieval, chưa thay pipeline mặc định.

## 4. Kết quả ingest

| Thành phần | Benchmark/subset | Metric chính | Kết quả |
|---|---|---:|---:|
| Text extraction | Bast-Korzen proxy | token F1 | 0.998 |
| Layout detection | DocLayNet 25 | layout F1@0.50 | 0.879 |
| Scientific layout | PubLayNet 25 | layout F1@0.50 | 0.778 |
| Table detection | PubTables detection 25 | table detection F1@0.50 | 0.987 |
| Table structure | PubTables structure OCR words 25 | structure F1 | 0.638 |
| OCR scan | OCR scan 25 | OCR token F1 | 1.000 |
| Academic PDF proxy | Nougat proxy 25 | token F1 | 0.628 |

Diễn giải: ingest đã đủ để phục vụ demo và báo cáo, nhưng một số benchmark chỉ chạy trên subset nhỏ. Không nên khẳng định hệ thống xử lý hoàn hảo mọi PDF.

## 5. Kết quả table structure

| Backend | Det F1@0.50 | Cell F1@0.50 | Cell F1@0.75 | Structure F1 | Text assign F1 | Row MAE | Col MAE | GriTS-con-like | Exact CSV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Default | 0.967 | 0.659 | - | 0.202 | 0.963 | 2.040 | 0.840 | 0.147 | 0.000 |
| TATR | 0.987 | 0.491 | 0.103 | 0.010 | 0.015 | 0.600 | 0.000 | 0.006 | 0.000 |
| hybrid_tatr OCR words | 0.987 | 0.598 | 0.248 | 0.638 | 0.955 | 0.600 | 0.000 | 0.387 | 0.040 |

Kết luận: TATR mạnh về hình học bảng, còn default mạnh hơn ở text assignment. `hybrid_tatr OCR words` kết hợp hai nguồn nên tăng structure F1 và GriTS-con-like rõ rệt. Tuy nhiên exact CSV/HTML vẫn thấp vì cần trùng tuyệt đối về hàng/cột/text/merged cell/markup.

## 6. Kết quả QA E2E

| Benchmark | Answer match | Grounded | Hallucination | Ghi chú |
|---|---:|---:|---:|---|
| QA smoke routed | 1.000 | 1.000 | 0.000 | Smoke test nhỏ |
| QCDT routed same index | 0.725 | 1.000 | 0.000 | Real PDF/local gold |
| Operations routed | 0.925 | 1.000 | 0.000 | QA operations benchmark |
| QCDT real PDF table QA | table_question_success 1.000 | 1.000 | 0.000 | Câu hỏi bảng thực tế |

Diễn giải: các benchmark chính cho thấy pipeline `routed_grounded` trả lời có căn cứ tốt và không ghi nhận hallucination trên các tập đã chạy. QCDT còn `answer_match_rate = 0.725`, nên chưa thể nói hệ thống trả lời đúng mọi câu.

## 7. SciFact

SciFact là benchmark công khai thiên về claim-evidence/citation khoa học, không phải natural QA đầy đủ.

| Metric | Kết quả |
|---|---:|
| query_count | 300 |
| answer_match_rate | 0.220 |
| evidence_match_rate | 0.727 |
| grounded_rate | 1.000 |
| hallucination_rate | 0.000 |
| end_to_end_success_rate | 0.203 |
| hybrid hit@5 | 0.793 |
| hybrid recall@5 | 0.771 |
| hybrid MRR@5 | 0.654 |
| hybrid NDCG@5 | 0.675 |

Diễn giải: SciFact chứng minh khả năng truy xuất/citation trên benchmark công khai. Answer match thấp vì SciFact không có thiết kế gốc cho natural QA answer generation.

## 8. QASPER

QASPER là benchmark natural scientific QA khó hơn vì câu hỏi tự nhiên, bài báo dài, nhiều answer annotation và có câu unanswerable.

Subset:

| Field | Value |
|---|---:|
| papers | 82 |
| chunks | 3,630 |
| questions | 100 |
| answerable | 95 |
| unanswerable | 5 |
| evidence mapped to chunks | 90 |

QA:

| Metric | Kết quả |
|---|---:|
| answer_match_rate | 0.100 |
| evidence_match_rate | 0.360 |
| grounded_rate | 1.000 |
| hallucination_rate | 0.050 |
| end_to_end_success_rate | 0.020 |
| abstain_accuracy | 0.000 |

Retrieval-only:

| Run | Hybrid rerank hit | Hybrid rerank recall |
|---|---:|---:|
| top_k=5 | 0.400 | 0.336 |
| top_k=10 | 0.520 | 0.451 |
| top_k=20 | 0.580 | 0.530 |

QA probe:

| Config | Answer match | Evidence match | E2E |
|---|---:|---:|---:|
| routed_grounded default | 0.100 | 0.360 | 0.020 |
| hybrid_no_routing top_k=10 | 0.090 | 0.470 | 0.040 |
| hybrid_no_routing top_k=20 | 0.090 | 0.510 | 0.040 |

Kết luận QASPER: tăng top-k/rerank cải thiện evidence recall, nhưng chưa cải thiện answer correctness. Nút thắt chính là answer synthesis/free-form QA và abstention, không chỉ retrieval.

## 9. Hạn chế

- QCDT answer match còn 0.725, chưa hoàn hảo.
- SciFact là claim-evidence benchmark, không đại diện đầy đủ cho natural QA.
- QASPER thấp vì paper dài, free-form answer và unanswerable handling còn yếu.
- Table exact CSV/HTML còn thấp do merged cell, text OCR và yêu cầu exact markup.
- Một số benchmark ingest dùng subset nhỏ 25 hoặc 100 mẫu.
- `hybrid_tatr` phụ thuộc OCR/PDF word boxes và chỉ là module tăng cường bảng có điều kiện, không thay toàn bộ ingest backend.
- LLM fallback chưa được đưa vào pipeline chính.

## 10. Safe Claims

- Hệ thống có pipeline PDF QA hoàn chỉnh với retrieval, grounded answer và citation.
- Hệ thống đạt grounded_rate cao trong các benchmark chính đã chạy.
- Không ghi nhận hallucination trong QA smoke, QCDT, Operations và SciFact với cấu hình chính.
- `hybrid_tatr OCR words` cải thiện table_structure F1 trên PubTables structure subset và đã được đưa vào pipeline chính theo cơ chế có điều kiện/fallback.
- SciFact cho thấy khả năng evidence/citation trên benchmark khoa học công khai.
- QASPER chỉ ra hạn chế thật của natural scientific QA trên paper dài.

## 11. Do Not Claim

- Không claim SOTA.
- Không claim xử lý hoàn hảo mọi PDF.
- Không claim table extraction hoàn chỉnh.
- Không claim `hybrid_tatr` thay thế toàn bộ production ingest backend; chỉ claim nó là module tăng cường bảng có điều kiện.
- Không claim LLM là lõi chính của hệ thống.
- Không claim exact CSV/HTML đã giải quyết xong.

## 12. Kết luận đưa vào báo cáo

Kết quả thực nghiệm cho thấy BOXTALK đã đáp ứng mục tiêu đồ án ở mức hệ thống: trích xuất PDF, tăng cường bảng có điều kiện bằng `hybrid_tatr`, lập chỉ mục, truy xuất, trả lời có dẫn chứng và đánh giá nhiều tầng. Pipeline chính `routed_grounded` phù hợp để demo và báo cáo vì có grounded_rate cao và kiểm soát hallucination tốt trên các benchmark chính.

Phần table structure và QASPER nên được trình bày như phân tích mở rộng. `hybrid_tatr` là đóng góp thực nghiệm có kết quả tốt ở tầng ingest, còn QASPER là bằng chứng trung thực về giới hạn của hệ thống khi chuyển sang natural scientific QA khó hơn.
