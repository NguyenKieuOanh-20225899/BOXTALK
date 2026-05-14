# Chương 4. Thực nghiệm và đánh giá

## 4.1. Tổng quan chương

Chương này trình bày thiết lập thực nghiệm và kết quả đánh giá hệ thống BOXTALK. Mục tiêu đánh giá không chỉ là độ chính xác của câu trả lời cuối cùng, mà còn kiểm tra từng tầng trong pipeline: xử lý PDF đầu vào, nhận dạng bố cục, trích xuất bảng, OCR, truy xuất thông tin, hỏi đáp có dẫn chứng và kiểm soát hallucination.

Pipeline chính được đánh giá là:

```text
PDF ingest -> chunk/index -> retrieval -> routed_grounded -> grounded answer + citation
```

Trong các thí nghiệm chính, hệ thống không bật LLM thật làm lõi trả lời. Các nhánh như `hybrid_tatr` và LLM fallback được xem là nhánh thực nghiệm để phân tích khả năng mở rộng.

## 4.2. Thiết lập thực nghiệm

Các thí nghiệm được chia thành ba nhóm:

- Đánh giá ingest PDF: text extraction, layout detection, OCR, table detection và table structure.
- Đánh giá retrieval: đo khả năng truy xuất đúng đoạn/chunk chứa bằng chứng.
- Đánh giá QA end-to-end: đo answer match, evidence match, grounded rate, hallucination rate và success rate.

Các benchmark công khai được sử dụng gồm DocLayNet, PubLayNet, PubTables, SciFact và QASPER. Một số benchmark được chạy trên subset nhỏ do giới hạn tài nguyên và thời gian thực nghiệm. Do đó, kết quả cần được hiểu là bằng chứng thực nghiệm có kiểm soát, không phải khẳng định SOTA.

## 4.3. Đánh giá ingest PDF

| Thành phần | Benchmark/subset | Metric | Kết quả |
|---|---|---:|---:|
| Text extraction | Bast-Korzen proxy | token F1 | 0.998 |
| Layout detection | DocLayNet 25 | layout F1@0.50 | 0.879 |
| Scientific layout | PubLayNet 25 | layout F1@0.50 | 0.778 |
| Table detection | PubTables detection 25 | table detection F1@0.50 | 0.987 |
| Table structure | PubTables structure OCR words 25 | structure F1 | 0.638 |
| Academic PDF proxy | Nougat proxy 25 | token F1 | 0.628 |

Kết quả cho thấy pipeline ingest hoạt động tốt với văn bản và phát hiện bảng. Layout detection đạt kết quả khá trên DocLayNet và PubLayNet. Tuy nhiên, kết quả PubLayNet thấp hơn DocLayNet, phản ánh đặc thù của bài báo khoa học: nhiều cột, hình, bảng, caption và thứ tự đọc phức tạp hơn.

Gợi ý bảng LaTeX:

```latex
\begin{table}[h]
\centering
\caption{Kết quả đánh giá ingest PDF}
\begin{tabular}{l l l r}
\hline
Thành phần & Benchmark & Metric & Kết quả \\
\hline
Text extraction & Bast-Korzen proxy & Token F1 & 0.998 \\
Layout detection & DocLayNet 25 & F1@0.50 & 0.879 \\
Scientific layout & PubLayNet 25 & F1@0.50 & 0.778 \\
Table detection & PubTables 25 & F1@0.50 & 0.987 \\
Table structure & PubTables structure 25 & Structure F1 & 0.638 \\
\hline
\end{tabular}
\end{table}
```

## 4.4. Đánh giá OCR

| Benchmark/subset | Metric | Kết quả |
|---|---:|---:|
| OCR scan 25 | OCR token F1 | 1.000 |

Kết quả OCR scan đạt cao trên subset đã chạy. Tuy nhiên, cần lưu ý rằng subset này chưa đại diện cho toàn bộ các trường hợp scan thực tế như ảnh mờ, xoay nghiêng, nhiễu, font cũ hoặc tài liệu tiếng Việt phức tạp. Vì vậy, trong báo cáo nên trình bày kết quả này như kiểm chứng pipeline OCR có thể chạy ổn định, không nên kết luận OCR đã hoàn hảo.

## 4.5. Đánh giá phát hiện và nhận dạng cấu trúc bảng

Bảng là thành phần quan trọng trong PDF QA vì nhiều câu hỏi cần truy xuất theo hàng/cột/cell. Hệ thống đánh giá ba backend:

- `default`: backend chính, dùng OCR/PDF text boxes và post-processing.
- `TATR`: Microsoft Table Transformer, mạnh về geometry nhưng không tự gán text vào cell.
- `hybrid_tatr OCR words`: kết hợp TATR geometry với OCR/PDF word boxes.

| Backend | Det F1@0.50 | Cell F1@0.50 | Structure F1 | Text assign F1 | Row MAE | Col MAE | GriTS-con-like | Exact CSV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Default | 0.967 | 0.659 | 0.202 | 0.963 | 2.040 | 0.840 | 0.147 | 0.000 |
| TATR | 0.987 | 0.491 | 0.010 | 0.015 | 0.600 | 0.000 | 0.006 | 0.000 |
| hybrid_tatr OCR words | 0.987 | 0.598 | 0.638 | 0.955 | 0.600 | 0.000 | 0.387 | 0.040 |

Kết quả cho thấy TATR cải thiện phát hiện bảng và đếm hàng/cột, nhưng nếu chỉ dùng TATR thì cấu trúc chứa text rất thấp vì model dự đoán hình học từ ảnh, không sinh nội dung cell. Backend `hybrid_tatr OCR words` khắc phục điểm này bằng cách dùng row/column geometry từ TATR và gán word boxes vào cell theo tọa độ. Nhờ đó, structure F1 tăng lên 0.638 và GriTS-con-like tăng lên 0.387.

Exact CSV/HTML vẫn thấp vì metric exact yêu cầu trùng tuyệt đối cả số hàng/cột, nội dung cell, merged cell, thứ tự text và markup. Trong thực tế, chỉ cần sai một ký tự OCR hoặc một merged cell là exact match có thể bằng 0. Vì vậy, các metric mềm hơn như cell IoU, structure F1 và GriTS-like phản ánh tiến bộ tốt hơn.

## 4.6. Đánh giá truy xuất thông tin

Trong benchmark SciFact, truy xuất hybrid đạt kết quả tốt:

| Strategy | hit@5 | recall@5 | MRR@5 | NDCG@5 |
|---|---:|---:|---:|---:|
| Hybrid | 0.793 | 0.771 | 0.654 | 0.675 |

Trên QASPER, retrieval khó hơn vì mỗi paper được chia thành nhiều chunk dài theo section. Kết quả hybrid rerank:

| top_k | Hit | Recall |
|---:|---:|---:|
| 5 | 0.400 | 0.336 |
| 10 | 0.520 | 0.451 |
| 20 | 0.580 | 0.530 |

Việc tăng top-k cải thiện evidence recall, nhưng không tự động cải thiện answer correctness. Điều này cho thấy retrieval chỉ là một phần của bài toán; answer synthesis và abstention cũng cần được cải thiện.

## 4.7. Đánh giá hỏi đáp end-to-end

| Benchmark | Answer match | Grounded | Hallucination |
|---|---:|---:|---:|
| QA smoke routed | 1.000 | 1.000 | 0.000 |
| QCDT routed same index | 0.725 | 1.000 | 0.000 |
| Operations routed | 0.925 | 1.000 | 0.000 |
| QCDT real PDF table QA | table success 1.000 | 1.000 | 0.000 |

Kết quả cho thấy pipeline `routed_grounded` có khả năng trả lời có căn cứ tốt trên các benchmark chính. Đặc biệt, không ghi nhận hallucination trên các tập smoke, QCDT, Operations và SciFact. Tuy vậy, QCDT answer match 0.725 cho thấy hệ thống vẫn còn sai hoặc trả lời chưa khớp gold answer ở một phần câu hỏi.

## 4.8. Đánh giá trên SciFact

SciFact là benchmark công khai thiên về claim-evidence/citation. Kết quả:

| Metric | Kết quả |
|---|---:|
| query_count | 300 |
| answer_match_rate | 0.220 |
| evidence_match_rate | 0.727 |
| grounded_rate | 1.000 |
| hallucination_rate | 0.000 |
| end_to_end_success_rate | 0.203 |

Evidence match đạt 0.727, cho thấy hệ thống truy xuất/citation được nhiều tài liệu liên quan theo qrels. Answer match thấp hơn vì SciFact không được thiết kế gốc như natural QA benchmark; gold answer được chuyển đổi từ evidence sentence nên có thể không trùng với câu trả lời hệ thống sinh ra từ cùng abstract.

## 4.9. Đánh giá trên QASPER

QASPER là benchmark natural scientific QA khó hơn SciFact. Subset chạy gồm 82 paper, 3.630 chunks và 100 câu hỏi.

| Metric | Kết quả |
|---|---:|
| answer_match_rate | 0.100 |
| evidence_match_rate | 0.360 |
| grounded_rate | 1.000 |
| hallucination_rate | 0.050 |
| end_to_end_success_rate | 0.020 |
| abstain_accuracy | 0.000 |

Kết quả QASPER thấp vì ba nguyên nhân chính. Thứ nhất, paper dài và nhiều chunk làm retrieval khó hơn. Thứ hai, QASPER có free-form answer, trong khi pipeline hiện tại thiên về trích xuất câu trả lời có căn cứ từ evidence. Thứ ba, hệ thống chưa xử lý tốt câu hỏi unanswerable, dẫn đến hallucination_rate 0.050.

Thử nghiệm tăng top-k cho thấy evidence match tăng từ 0.360 lên 0.510 ở fixed hybrid top_k=20, nhưng answer_match vẫn khoảng 0.090. Điều này xác nhận nút thắt chính không chỉ là retrieval mà còn là answer synthesis/free-form QA và abstention.

## 4.10. Phân tích lỗi và thảo luận

Các lỗi chính quan sát được:

- PDF nhiều cột có thể gây sai thứ tự đọc hoặc tách block chưa tối ưu.
- Table structure vẫn khó ở merged cell, header nhiều dòng, footnote/caption và OCR text noise.
- Exact CSV/HTML nhạy với lỗi nhỏ nên thường thấp hơn các metric mềm.
- SciFact answer match thấp do bản chất claim-evidence, không phải natural QA.
- QASPER thấp do paper dài, câu hỏi tự nhiên, free-form answer và unanswerable handling.
- Pipeline chính chưa dùng LLM thật, nên khả năng diễn giải/synthesis còn hạn chế.

Các kết quả này phù hợp với mục tiêu đồ án: xây dựng pipeline grounded QA và đánh giá minh bạch các tầng, đồng thời chỉ ra giới hạn còn lại.

## 4.11. Kết chương

Thực nghiệm cho thấy BOXTALK đã xây dựng được pipeline hỏi đáp PDF có dẫn chứng, hoạt động ổn định trên các benchmark chính và có grounded_rate cao. Hệ thống kiểm soát hallucination tốt trong các cấu hình chính đã chạy. Nhánh `hybrid_tatr` cho thấy hướng cải thiện rõ rệt đối với table structure ở tầng ingest. Tuy nhiên, natural scientific QA trên QASPER và exact table reconstruction vẫn là các bài toán khó, cần được tiếp tục nghiên cứu ở các hướng phát triển sau.
