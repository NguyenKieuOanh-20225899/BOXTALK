# Defense Slide Outline

## Slide 1. Tên đề tài và mục tiêu

- Đề tài: Nghiên cứu các kĩ thuật truy xuất và hỏi đáp thông tin trên tài liệu PDF.
- Hệ thống minh họa: BOXTALK.
- Mục tiêu: trả lời câu hỏi từ PDF có dẫn chứng.
- Trọng tâm: ingest, retrieval, grounded QA, citation.

Hình/bảng nên dùng: sơ đồ pipeline một dòng.

Speaker note: Giới thiệu bài toán và nhấn mạnh hệ thống không chỉ trả lời, mà còn phải chỉ ra bằng chứng trong tài liệu.

## Slide 2. Vấn đề thực tế

- PDF là định dạng phổ biến trong học thuật, quy chế, báo cáo.
- Người dùng khó tìm nhanh thông tin trong tài liệu dài.
- Câu trả lời cần đáng tin và có nguồn trích dẫn.
- Hỏi đáp PDF cần xử lý cả văn bản, bảng, ảnh scan và layout.

Hình/bảng nên dùng: ảnh PDF thật hoặc screenshot UI.

Speaker note: Nêu ví dụ người dùng hỏi quy định, điểm, điều kiện học, hoặc nội dung trong paper khoa học.

## Slide 3. Thách thức PDF QA

- PDF lưu nội dung theo tọa độ hiển thị, không có cấu trúc logic rõ ràng.
- Văn bản nhiều cột dễ sai thứ tự đọc.
- Bảng cần khôi phục hàng/cột/cell.
- Scan PDF cần OCR.
- QA dễ hallucinate nếu không kiểm soát evidence.

Hình/bảng nên dùng: minh họa block text/table/caption trên một trang PDF.

Speaker note: Giải thích vì sao bài toán không đơn giản là đọc text rồi hỏi đáp.

## Slide 4. Kiến trúc tổng thể

- PDF ingest.
- Chunk/index.
- Retrieval.
- `routed_grounded`.
- Grounded answer + citation.

Hình/bảng nên dùng: kiến trúc tổng thể từ PDF đến answer.

Speaker note: Đây là luồng chính dùng trong báo cáo; nhánh LLM fallback và hybrid TATR là thực nghiệm.

## Slide 5. Routed RAG / Grounded QA

- Router phân loại câu hỏi.
- Retrieval chọn chiến lược phù hợp.
- Evidence checker đánh giá độ đủ bằng chứng.
- Answer generator trả lời ngắn và gắn citation.
- Không bật LLM thật làm pipeline chính.

Hình/bảng nên dùng: flow `question -> route -> retrieve -> evidence -> answer`.

Speaker note: Nhấn mạnh citation và groundedness là cơ chế giảm hallucination.

## Slide 6. Ingest Benchmark

- Text extraction: Bast-Korzen proxy token F1 0.998.
- Layout: DocLayNet F1@0.50 0.879.
- Scientific layout: PubLayNet F1@0.50 0.778.
- Table detection: PubTables F1@0.50 0.987.
- OCR scan subset: token F1 1.000.

Hình/bảng nên dùng: bảng ingest results.

Speaker note: Nói rõ một số benchmark là subset nhỏ, không claim hoàn hảo mọi PDF.

## Slide 7. Table Structure / Hybrid TATR

- Default tốt ở text assignment nhưng row/column còn yếu.
- TATR tốt ở geometry nhưng thiếu text cell.
- Hybrid TATR OCR words kết hợp geometry và word boxes.
- Structure F1 tăng lên 0.638.
- Exact CSV vẫn thấp vì merged cell/markup/OCR.

Hình/bảng nên dùng: bảng so sánh Default/TATR/Hybrid.

Speaker note: Đây là nhánh thực nghiệm ở tầng ingest, chưa thay backend chính.

## Slide 8. QA E2E Results

- QA smoke routed answer match 1.000.
- QCDT routed answer match 0.725.
- Operations routed answer match 0.925.
- QCDT table QA success 1.000.
- Hallucination 0.000 trên các benchmark chính.

Hình/bảng nên dùng: bảng QA E2E.

Speaker note: QCDT chưa hoàn hảo nhưng groundedness tốt, hệ thống ưu tiên không bịa.

## Slide 9. SciFact + QASPER

- SciFact: claim-evidence/citation benchmark.
- SciFact evidence_match 0.727, hallucination 0.000.
- QASPER: natural scientific QA khó hơn.
- QASPER answer_match 0.100, evidence_match 0.360.
- Tăng top-k cải thiện evidence recall nhưng không giải quyết answer synthesis.

Hình/bảng nên dùng: bảng SciFact/QASPER ngắn.

Speaker note: Dùng SciFact để chứng minh citation công khai; dùng QASPER để phân tích giới hạn thực tế.

## Slide 10. Demo UI

- Upload/chọn PDF.
- Hỏi câu hỏi văn bản.
- Hỏi câu hỏi bảng.
- Hiển thị answer + citation.
- Hỏi câu không có trong tài liệu để xem hệ thống xử lý thiếu evidence.

Hình/bảng nên dùng: screenshot UI và câu trả lời có citation.

Speaker note: Chạy demo ngắn, tránh câu hỏi quá mở hoặc cần suy luận ngoài tài liệu.

## Slide 11. Hạn chế

- QCDT answer match còn 0.725.
- QASPER thấp do free-form QA và paper dài.
- Exact CSV/HTML còn thấp.
- Benchmark ingest một số phần dùng subset nhỏ.
- Abstention cho unanswerable chưa tốt.

Hình/bảng nên dùng: bảng limitation -> hướng cải thiện.

Speaker note: Chủ động nói giới hạn để tránh overclaim.

## Slide 12. Kết luận và hướng phát triển

- Đã xây dựng pipeline PDF QA có citation.
- Có benchmark nhiều tầng.
- Kiểm soát hallucination tốt trong benchmark chính.
- Hybrid TATR là hướng cải thiện bảng có tiềm năng.
- Hướng tiếp: section-aware retrieval, answer synthesis, abstention, table-aware retrieval.

Hình/bảng nên dùng: sơ đồ “đã làm” và “tiếp theo”.

Speaker note: Kết lại bằng phạm vi đúng: hệ thống hoàn chỉnh, đo lường minh bạch, còn hướng phát triển rõ ràng.

## Slide bổ sung A. Vì sao không chỉ là gọi LLM hỏi PDF?

- PDF cần ingest: text layer, layout, OCR, table, reading order.
- Cần chunking có metadata để citation đúng trang/mục.
- Cần retrieval để tìm evidence trước khi trả lời.
- Cần evidence checker để hạn chế trả lời thiếu căn cứ.
- Cần benchmark riêng cho ingest, retrieval và QA.
- LLM thật không phải lõi bắt buộc của pipeline chính.

Hình/bảng nên dùng: bảng `kĩ thuật -> vai trò -> metric đánh giá`.

Speaker note: Đây là slide phòng thủ trước câu hỏi "đồ án có phải chỉ ghép chatbot với PDF không?". Nhấn mạnh đóng góp nằm ở pipeline, đánh giá nhiều tầng và kiểm soát evidence.

## Slide bổ sung B. Ablation / vai trò từng thành phần

- BM25 vs dense vs hybrid retrieval.
- Chunk thường vs structure-aware chunking.
- Không region routing vs có region routing.
- Default table extractor vs TATR vs hybrid_tatr.
- QA không kiểm evidence vs grounded QA.
- top_k=5/10/20 cho retrieval evidence.

Hình/bảng nên dùng: bảng ablation ngắn, mỗi dòng một thành phần.

Speaker note: Nếu chưa có đủ số liệu, trình bày như kế hoạch thực nghiệm bổ sung và chỉ đưa các dòng đã chạy chắc chắn. Không đưa số liệu chưa kiểm chứng.

## Slide bổ sung C. Safe claims và giới hạn

- Claim chính: pipeline PDF QA có citation, đánh giá nhiều tầng, grounded_rate cao trên benchmark chính.
- Không claim: SOTA, xử lý hoàn hảo mọi PDF, table extraction hoàn chỉnh, production-ready.
- Phạm vi mạnh: PDF text-layer/bán cấu trúc như quy chế, quy định, hướng dẫn nghiệp vụ.
- Phạm vi mở rộng: scan mờ, bảng phức tạp, paper khoa học dài, free-form synthesis.
- QASPER và exact CSV được dùng để phân tích giới hạn, không phải claim chính.

Hình/bảng nên dùng: bảng `Có thể claim / Không nên claim`.

Speaker note: Chủ động nêu giới hạn giúp phần bảo vệ đáng tin hơn và giảm rủi ro bị phản biện bắt lỗi overclaim.

## Slide bổ sung D. Case study phản biện

- Case 1: văn bản pháp quy, giữ đúng Điều/Khoản/danh sách.
- Case 2: câu hỏi bảng, chứng minh table-aware ingest/retrieval.
- Case 3: câu hỏi ngoài tài liệu, kiểm tra abstention hoặc limitation.
- Với mỗi case: câu hỏi, evidence, answer, citation, lỗi cũ nếu có, cải tiến đã làm.

Hình/bảng nên dùng: screenshot answer + citation và một bảng evidence ngắn.

Speaker note: Case study phải ngắn, chắc, đã chạy thử trước. Không demo PDF hoặc câu hỏi mới ngay trong buổi bảo vệ nếu chưa kiểm tra.
