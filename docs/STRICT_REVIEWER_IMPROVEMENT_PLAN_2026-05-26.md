# Kế hoạch cải thiện đồ án theo góc nhìn hội đồng phản biện nghiêm khắc

Ngày cập nhật: 2026-05-26  
Đề tài: **Nghiên cứu các kĩ thuật truy xuất và hỏi đáp thông tin trên tài liệu PDF**

## 1. Định vị lại đồ án cho thật thuyết phục

Đồ án nên được trình bày như một **research prototype về grounded PDF QA**, không phải một ứng dụng hỏi đáp PDF tổng quát. Điểm nổi bật cần nhấn mạnh là hệ thống không chỉ trả lời câu hỏi, mà còn:

- trích xuất nội dung PDF có cấu trúc;
- chia chunk và giữ metadata phục vụ citation;
- truy xuất evidence bằng các kĩ thuật retrieval;
- sinh câu trả lời dựa trên evidence;
- đánh giá nhiều tầng từ ingest, retrieval đến QA;
- chủ động phân tích giới hạn thay vì claim quá rộng.

Câu định vị nên dùng:

```text
Đồ án tập trung nghiên cứu và kết hợp các kĩ thuật xử lý PDF, truy xuất thông tin và hỏi đáp có căn cứ để xây dựng một pipeline trả lời câu hỏi trên PDF có citation. Trọng tâm không phải là sinh câu trả lời bằng mọi giá, mà là truy xuất đúng bằng chứng và kiểm soát hallucination trong phạm vi tài liệu PDF text/bán cấu trúc.
```

## 2. Những điểm cần cải thiện để đồ án nổi bật

### 2.1. Bổ sung ablation study

Đây là phần có sức thuyết phục cao nhất vì chứng minh mỗi thành phần kỹ thuật trong hệ thống có lý do tồn tại.

Các thí nghiệm nên có:

| Câu hỏi phản biện | Thí nghiệm cần có | Metric nên báo cáo |
|---|---|---|
| Hybrid retrieval có cần thiết không? | BM25 vs dense vs hybrid | hit@k, recall@k, MRR, NDCG |
| Chunking có thật sự ảnh hưởng QA không? | chunk thường vs structure-aware chunking | retrieval recall, answer_match |
| Region routing có đóng góp gì? | không region routing vs có region routing | reading order, table detection, QA case study |
| Hybrid TATR có tốt hơn không? | default table vs TATR vs hybrid_tatr | structure F1, text assign F1, exact CSV |
| Grounded QA có giảm bịa không? | QA không kiểm evidence vs grounded QA | grounded_rate, hallucination_rate |
| Tăng top-k có giải quyết QASPER không? | top_k=5/10/20 | evidence recall, answer_match |

Thông điệp cần rút ra:

```text
Ablation cho thấy chất lượng QA không chỉ phụ thuộc answer generator, mà phụ thuộc chuỗi ingest -> chunking -> retrieval -> evidence checking. Các cải tiến như structure-aware chunking, hybrid retrieval và grounded QA giúp tăng khả năng tìm đúng bằng chứng và giảm trả lời thiếu căn cứ.
```

### 2.2. Chuẩn bị 3 case study thực tế

Hội đồng thường bị thuyết phục bởi case study rõ ràng hơn là chỉ nhìn bảng số.

Nên chuẩn bị 3 case:

| Case | Mục tiêu chứng minh | Câu hỏi mẫu |
|---|---|---|
| Văn bản pháp quy | hệ thống giữ được cấu trúc Điều/Khoản/danh sách | "Ban coi thi gồm những thành phần nào?" |
| Câu hỏi bảng | hệ thống không chỉ đọc text phẳng mà có xử lý table | "Chương trình có thời gian đào tạo bao lâu và bao nhiêu tín chỉ?" |
| Câu không có bằng chứng | hệ thống biết thận trọng khi thiếu evidence | "Tài liệu này có nói về học phí năm 2030 không?" |

Mỗi case nên có:

- câu hỏi;
- đoạn evidence được retrieve;
- câu trả lời;
- citation trang/mục;
- giải thích ngắn vì sao case này khó;
- trước/sau nếu có cải tiến.

### 2.3. Tách rõ hai bài toán: truy xuất và hỏi đáp

Tên đề tài có cả **truy xuất** và **hỏi đáp**, nên báo cáo cần tách rõ:

- truy xuất: tìm đúng chunk/evidence;
- hỏi đáp: sinh câu trả lời dựa trên evidence;
- citation/grounding: kiểm tra câu trả lời có dựa trên nguồn hay không.

Không nên chỉ trình bày kết quả answer cuối cùng. Nếu answer sai, cần biết lỗi nằm ở retrieval, chunking, ingest hay synthesis.

Cấu trúc phân tích lỗi nên dùng:

| Loại lỗi | Dấu hiệu | Nguyên nhân có thể | Hướng xử lý |
|---|---|---|---|
| Ingest error | thiếu đoạn, sai thứ tự đọc | PDF layout/cột/header/footer | cải thiện reading order, region routing |
| Chunking error | evidence bị tách mất ngữ cảnh | chunk theo token thô | structure-aware chunking |
| Retrieval error | chunk đúng không vào top-k | query mismatch, sparse/dense lệch | hybrid retrieval, rerank |
| Evidence error | có chunk liên quan nhưng chưa đủ | top-k thấp, citation thiếu | evidence checker, tăng context |
| QA synthesis error | evidence đúng nhưng trả lời sai | answer generator yếu | template/rule/LLM có kiểm chứng |
| Abstention error | vẫn trả lời khi thiếu evidence | thiếu ngưỡng confidence | abstention policy |

### 2.4. Làm rõ đóng góp nghiên cứu

Nên trình bày đóng góp theo 4 nhóm:

1. **Pipeline PDF QA hoàn chỉnh**  
   Từ ingest PDF đến answer có citation, có thể đánh giá end-to-end.

2. **Ingest nhiều tầng cho PDF phức tạp**  
   Có probe, region-level routing, text extraction, OCR, table extraction và fallback.

3. **Retrieval + grounded QA**  
   Kết hợp truy xuất evidence và trả lời có căn cứ, đo bằng grounded_rate/hallucination_rate.

4. **Đánh giá nhiều tầng và phân tích giới hạn**  
   Có benchmark ingest, layout, table, retrieval, QA, cùng các benchmark khó như QASPER để chỉ ra giới hạn.

Không nên mô tả đóng góp là "xây dựng chatbot hỏi PDF", vì cách nói đó làm giảm giá trị nghiên cứu.

### 2.5. Bổ sung một bảng "claim an toàn"

Trong slide hoặc báo cáo nên có bảng này để tránh bị phản biện bắt lỗi claim quá rộng.

| Có thể claim | Không nên claim |
|---|---|
| Có pipeline PDF QA có citation | Xử lý hoàn hảo mọi PDF |
| Có đánh giá nhiều tầng | Đạt SOTA |
| Grounded_rate cao trên benchmark chính | Không bao giờ hallucinate |
| Hybrid TATR cải thiện structure F1 trên subset đã chạy | Table extraction đã hoàn chỉnh |
| QASPER được dùng để phân tích giới hạn | Hệ thống QA tốt trên mọi paper khoa học dài |
| OCR là nhánh mở rộng | OCR tiếng Việt thực tế đã ổn định mọi trường hợp |

## 3. Góc nhìn của hội đồng phản biện nghiêm khắc

Nếu đóng vai phản biện khó tính, các câu hỏi chính sẽ là:

### 3.1. "Đây là nghiên cứu hay chỉ là ghép thư viện?"

Cần trả lời bằng:

- sơ đồ pipeline tự thiết kế;
- bảng so sánh kỹ thuật;
- ablation study;
- phân tích lỗi;
- quyết định kỹ thuật có lý do rõ ràng.

Điều cần làm thêm:

- thêm bảng so sánh BM25/dense/hybrid;
- thêm ablation chunking/retrieval/grounding;
- ghi rõ module nào là thư viện, module nào là logic tích hợp/đánh giá của đồ án.

### 3.2. "Vì sao bài toán PDF khó?"

Cần chứng minh bằng các vấn đề đặc thù:

- PDF lưu theo tọa độ, không theo cấu trúc logic;
- thứ tự đọc có thể sai;
- bảng không có row/column thật;
- header/footer gây nhiễu;
- scan cần OCR;
- mixed layout cần route theo vùng.

Điều cần làm thêm:

- thêm 1 slide hoặc 1 mục trong báo cáo về thách thức PDF;
- đưa ví dụ lỗi thật: câu "Ban coi thi..." sai do chunking/reading order, sau đó sửa được.

### 3.3. "Kết quả có đủ tin cậy không?"

Điểm yếu hiện tại:

- một số benchmark dùng subset nhỏ;
- QASPER thấp;
- exact CSV thấp;
- QCDT answer_match còn 0.725.

Cách phòng thủ:

- nói rõ benchmark chính bám sát phạm vi PDF quy chế/quy định;
- dùng QASPER và exact CSV như benchmark phân tích giới hạn;
- không claim vượt quá kết quả;
- bổ sung case study thực tế có citation.

Điều cần làm thêm:

- ghi rõ số lượng mẫu của từng benchmark;
- phân nhóm benchmark chính/hỗ trợ/mở rộng;
- thêm confidence statement: kết quả chứng minh trong phạm vi nào, chưa chứng minh trong phạm vi nào.

### 3.4. "Hệ thống có thật sự giảm hallucination không?"

Cần trả lời bằng metric và cơ chế:

- grounded_rate;
- hallucination_rate;
- evidence checker;
- abstention khi thiếu bằng chứng;
- demo câu hỏi không có trong tài liệu.

Điều cần làm thêm:

- chuẩn bị một câu hỏi ngoài tài liệu để demo;
- nếu hệ thống chưa abstain tốt, nói thẳng đó là hạn chế và hướng cải thiện;
- không nói "không hallucinate tuyệt đối".

### 3.5. "Đóng góp mới nằm ở đâu?"

Câu trả lời nên tránh claim quá lớn. Nên nói:

```text
Đóng góp của đồ án nằm ở việc thiết kế, tích hợp và đánh giá một pipeline PDF QA có căn cứ, trong đó các kĩ thuật ingest, chunking, retrieval và grounded QA được kết hợp có kiểm chứng. Đồ án không claim phát minh một mô hình nền tảng mới, mà tập trung vào bài toán hệ thống: làm thế nào để PDF được xử lý thành evidence đáng tin và câu trả lời có citation.
```

Điều cần làm thêm:

- có bảng "kĩ thuật nghiên cứu -> vai trò -> bằng chứng đánh giá";
- nhấn mạnh điểm mới ở cấp hệ thống và đánh giá nhiều tầng.

## 4. Checklist việc cần làm thêm trước bảo vệ

### P0 - Rất nên làm

- [x] Tạo bảng ablation BM25 vs dense vs hybrid. Xem `docs/ABLATION_AND_DEMO_CASES_2026-05-26.md`.
- [x] Tạo bảng ablation default table vs TATR vs hybrid_tatr. Xem `docs/ABLATION_AND_DEMO_CASES_2026-05-26.md`.
- [x] Chuẩn bị 3 case study: text pháp quy, bảng, câu phủ định có citation. Xem `docs/ABLATION_AND_DEMO_CASES_2026-05-26.md`.
- [ ] Thêm slide "Vì sao không chỉ là gọi LLM hỏi PDF?"
- [ ] Thêm slide "Safe claims / limitations".
- [ ] Chụp screenshot demo có answer + citation.

### P1 - Nên làm nếu còn thời gian

- [x] Chạy thêm retrieval top_k=5/10/20 trên tập domain chính. Xem `docs/P1_ADDITIONAL_ANALYSIS_2026-05-26.md`.
- [x] Bổ sung error analysis cho QCDT answer_match 0.725. Xem `docs/P1_ADDITIONAL_ANALYSIS_2026-05-26.md`.
- [x] Thêm bảng phân loại lỗi: ingest/chunking/retrieval/QA/abstention. Xem `docs/P1_ADDITIONAL_ANALYSIS_2026-05-26.md`.
- [x] Kiểm tra demo với một PDF chưa dùng trong benchmark chính nhưng cùng miền tài liệu. Xem `docs/P1_ADDITIONAL_ANALYSIS_2026-05-26.md`.
- [x] Viết phụ lục mô tả metric: hit@k, recall@k, MRR, NDCG, grounded_rate. Xem `docs/P1_ADDITIONAL_ANALYSIS_2026-05-26.md`.

### P2 - Không nên ưu tiên sát ngày bảo vệ

- [ ] Làm UI mới.
- [ ] Thêm nhiều loại PDF mới chưa test.
- [ ] Đưa LLM thật vào pipeline chính nếu chưa có benchmark.
- [ ] Cố sửa QASPER mạnh trong thời gian ngắn.
- [ ] Claim production-ready.

## 5. Cách trình bày để nổi bật trước hội đồng

Thứ tự trình bày nên là:

1. PDF QA khó vì PDF thiếu cấu trúc logic.
2. Đồ án xử lý bài toán theo pipeline nhiều tầng.
3. Mỗi tầng có kỹ thuật và metric riêng.
4. Hệ thống trả lời có evidence/citation, không chỉ sinh text.
5. Kết quả tốt trong phạm vi PDF text/bán cấu trúc.
6. Các benchmark khó được dùng để chỉ ra giới hạn thật.
7. Hướng phát triển bám vào giới hạn đã đo được.

Thông điệp kết thúc nên dùng:

```text
Điểm chính của đồ án là xây dựng và đánh giá một pipeline hỏi đáp PDF có căn cứ. Hệ thống không cố trả lời mọi câu hỏi trên mọi PDF, mà tập trung vào việc biến PDF thành evidence có thể truy xuất, sau đó sinh câu trả lời có citation và đo lường rõ giới hạn của từng tầng kỹ thuật.
```

## 6. Nếu chỉ còn rất ít thời gian

Nếu thời gian chuẩn bị hạn chế, chỉ cần làm 4 việc:

1. Chốt 3 demo case có citation.
2. Làm 1 bảng ablation retrieval: BM25/dense/hybrid.
3. Làm 1 bảng limitations + safe claims.
4. Chuẩn bị câu trả lời cho câu hỏi: "Đây là nghiên cứu hay chỉ là ghép thư viện?"

Đây là 4 phần có tác động lớn nhất đến cảm nhận của hội đồng phản biện.
