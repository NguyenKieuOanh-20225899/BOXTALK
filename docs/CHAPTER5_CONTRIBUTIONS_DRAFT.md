# Chương 5. Đóng góp của hệ thống

## 5.1. Tổng quan chương

Chương này tổng hợp các đóng góp chính của đồ án từ góc nhìn hệ thống và thực nghiệm. Đồ án không chỉ xây dựng một ứng dụng hỏi đáp PDF, mà còn thiết kế pipeline xử lý tài liệu, truy xuất, trả lời có dẫn chứng và bộ đánh giá nhiều tầng để kiểm chứng chất lượng.

Bốn đóng góp chính gồm:

- Xây dựng pipeline PDF QA có citation.
- So sánh và triển khai nhiều kỹ thuật retrieval.
- Xây dựng benchmark nhiều tầng: ingest, retrieval, QA và citation.
- Thử nghiệm `hybrid_tatr` nhằm cải thiện table structure extraction.

## 5.2. Quy trình Routed RAG cho hỏi đáp PDF có dẫn chứng

Hệ thống sử dụng quy trình Routed RAG gồm các bước:

```text
PDF ingest -> chunk/index -> query routing -> retrieval -> evidence checking -> grounded answer + citation
```

Điểm khác biệt so với RAG đơn giản là hệ thống không chỉ truy xuất top-k đoạn rồi trả lời, mà còn:

- Phân loại/routing câu hỏi để chọn chiến lược truy xuất phù hợp.
- Kết hợp sparse và dense retrieval.
- Kiểm tra độ phù hợp của evidence trước khi trả lời.
- Sinh câu trả lời gắn với citation.
- Hạn chế trả lời khi evidence chưa đủ.

Trong đồ án, `routed_grounded` được chọn làm luồng QA chính vì cân bằng giữa chất lượng câu trả lời, groundedness và khả năng giải thích thông qua citation.

## 5.3. Cơ chế đánh giá nhiều tầng

Một đóng góp quan trọng là hệ thống đánh giá không chỉ đo câu trả lời cuối cùng. Các tầng đánh giá gồm:

- Ingest: text extraction, OCR, layout detection, table detection và table structure.
- Retrieval: hit@k, recall@k, MRR, NDCG.
- QA end-to-end: answer match, evidence match, grounded rate, hallucination rate.
- Citation/evidence: kiểm tra chunk/citation có khớp gold evidence hay không.

Cách đánh giá nhiều tầng giúp phân tích lỗi chính xác hơn. Ví dụ, với QASPER, tăng top-k/rerank cải thiện evidence recall nhưng không cải thiện answer correctness đáng kể. Điều này cho thấy lỗi không chỉ nằm ở retrieval, mà còn ở answer synthesis và abstention.

## 5.4. Cải thiện xử lý bảng bằng hybrid TATR và OCR word boxes

Bảng là thành phần khó trong PDF vì cấu trúc logic không được lưu trực tiếp trong file PDF. Đồ án thử nghiệm ba hướng xử lý:

- Backend mặc định dựa trên OCR/PDF text boxes và post-processing.
- TATR pretrained model để nhận diện geometry của bảng, hàng, cột.
- `hybrid_tatr OCR words`, kết hợp TATR geometry với OCR/PDF word boxes.

Kết quả PubTables structure subset:

| Backend | Det F1@0.50 | Structure F1 | Text assign F1 | GriTS-con-like | Exact CSV |
|---|---:|---:|---:|---:|---:|
| Default | 0.967 | 0.202 | 0.963 | 0.147 | 0.000 |
| TATR | 0.987 | 0.010 | 0.015 | 0.006 | 0.000 |
| hybrid_tatr OCR words | 0.987 | 0.638 | 0.955 | 0.387 | 0.040 |

Kết quả cho thấy TATR mạnh ở geometry nhưng thiếu text assignment, còn backend mặc định mạnh ở text nhưng yếu row/column grouping. `hybrid_tatr` kết hợp hai nguồn nên cải thiện rõ structure F1. Đây là nhánh thực nghiệm có giá trị nghiên cứu, nhưng chưa thay backend chính vì còn phụ thuộc OCR, merged cell và xử lý caption/footnote.

## 5.5. Kiểm soát hallucination bằng evidence và citation

Hệ thống được thiết kế để ưu tiên câu trả lời có căn cứ. Các benchmark chính ghi nhận:

| Benchmark | Grounded rate | Hallucination rate |
|---|---:|---:|
| QA smoke routed | 1.000 | 0.000 |
| QCDT routed | 1.000 | 0.000 |
| Operations routed | 1.000 | 0.000 |
| SciFact | 1.000 | 0.000 |
| QASPER | 1.000 | 0.050 |

Kết quả cho thấy cơ chế evidence/citation giúp kiểm soát hallucination tốt trên các benchmark chính. QASPER xuất hiện hallucination 0.050 chủ yếu do các câu unanswerable chưa được abstain đúng. Đây là hướng cần cải thiện tiếp.

## 5.6. Kết chương

Đồ án đóng góp một pipeline PDF QA có dẫn chứng, bộ đánh giá nhiều tầng và một nhánh cải thiện table structure có kết quả thực nghiệm rõ ràng. Các đóng góp này phù hợp với mục tiêu nghiên cứu kỹ thuật truy xuất và hỏi đáp thông tin trên tài liệu PDF. Hệ thống vẫn còn hạn chế ở natural scientific QA, exact table reconstruction và xử lý câu hỏi không có thông tin, nhưng các hạn chế này đã được đo lường và phân tích minh bạch.
