# Chương 6. Kết luận và hướng phát triển

## 6.1. Kết luận

Đồ án đã xây dựng hệ thống BOXTALK cho bài toán hỏi đáp thông tin trên tài liệu PDF. Hệ thống bao gồm các thành phần chính: xử lý PDF đầu vào, chia chunk và lập chỉ mục, truy xuất thông tin, định tuyến câu hỏi, kiểm tra bằng chứng và sinh câu trả lời có dẫn chứng.

Pipeline chính được sử dụng trong báo cáo là:

```text
PDF ingest -> chunk/index -> retrieval -> routed_grounded -> grounded answer + citation
```

Kết quả thực nghiệm cho thấy hệ thống hoạt động ổn định trên các benchmark chính. Các benchmark QA smoke, QCDT, Operations và SciFact không ghi nhận hallucination trong cấu hình chính. Hệ thống cũng đạt kết quả tốt ở một số benchmark ingest như text extraction, layout detection và table detection.

Đồ án cũng thử nghiệm hướng cải thiện table structure bằng `hybrid_tatr OCR words`. Kết quả cho thấy việc kết hợp TATR geometry với OCR/PDF word boxes giúp tăng structure F1 trên PubTables structure subset. Đây là một hướng mở rộng có ý nghĩa cho các câu hỏi liên quan đến bảng trong tài liệu PDF.

## 6.2. Hạn chế

Hệ thống vẫn còn các hạn chế sau:

- QCDT answer_match_rate còn 0.725, nghĩa là vẫn có câu trả lời chưa khớp gold answer.
- SciFact answer_match thấp vì SciFact là claim-evidence benchmark, không phải natural QA benchmark.
- QASPER thấp do natural scientific QA khó hơn: paper dài, free-form answer, nhiều evidence và câu hỏi unanswerable.
- Table exact CSV/HTML còn thấp vì reconstruction đòi hỏi trùng tuyệt đối hàng/cột/cell/text/merged cell/markup.
- Một số benchmark ingest chỉ chạy trên subset nhỏ, chưa đại diện cho toàn bộ phân phối PDF thực tế.
- `hybrid_tatr` phụ thuộc OCR/PDF word boxes và còn yếu ở merged cell, caption, footnote và cell rỗng.
- Abstention handling còn yếu, thể hiện ở QASPER unanswerable.
- LLM fallback chưa được đưa vào pipeline chính, nên khả năng diễn giải free-form answer còn hạn chế.

Các hạn chế này không phủ nhận kết quả hệ thống, mà giúp xác định rõ phạm vi đúng của đồ án và hướng nghiên cứu tiếp theo.

## 6.3. Hướng phát triển

Các hướng phát triển khả thi gồm:

- Section-aware retrieval cho paper dài: tận dụng cấu trúc section, heading và discourse để cải thiện truy xuất trên QASPER-like documents.
- Better answer synthesis/free-form QA: cải thiện khả năng tổng hợp câu trả lời ngắn từ nhiều evidence thay vì chỉ trích xuất câu gần nhất.
- Abstention handling: phát hiện tốt hơn khi tài liệu không chứa đủ thông tin để trả lời.
- Official GriTS metric: tích hợp GriTS chính thức để đánh giá table structure mềm và chuẩn hơn exact CSV/HTML.
- Table-aware retrieval: lập chỉ mục bảng theo cell/row/column để hỗ trợ table QA tốt hơn.
- OCR tiếng Việt và scan PDF thực tế: mở rộng đánh giá trên tài liệu scan tiếng Việt có nhiễu, xoay nghiêng hoặc chất lượng thấp.
- LLM fallback có kiểm soát: dùng LLM cho explanation hoặc answer synthesis khi evidence đã đủ, nhưng vẫn ràng buộc bằng citation để hạn chế hallucination.

Tóm lại, BOXTALK đã đạt mục tiêu xây dựng một hệ thống hỏi đáp PDF có dẫn chứng và bộ đánh giá nhiều tầng. Hệ thống đủ ổn định để trình bày trong đồ án, đồng thời còn nhiều hướng phát triển rõ ràng cho nghiên cứu tiếp theo.
