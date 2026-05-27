# Benchmark cần giữ theo phạm vi PDF quy chế/quy định

Phạm vi chính của đồ án được chốt là: **hỏi đáp có dẫn chứng trên PDF text-layer có cấu trúc bán hình thức**, ví dụ quy chế, quy định, thông tư, hướng dẫn nghiệp vụ, policy nội bộ, tài liệu vận hành và tài liệu đào tạo.

Vì vậy benchmark nên được trình bày theo 3 mức: benchmark chính, benchmark hỗ trợ và benchmark phân tích mở rộng.

## 1. Benchmark chính cần đưa vào báo cáo

Các benchmark này bám sát pipeline chính và nên xuất hiện trong chương thực nghiệm/slide bảo vệ.

| Tầng đánh giá | Benchmark | Vì sao cần giữ |
|---|---|---|
| Ingest text/layout | Bast-Korzen proxy | Đánh giá text extraction và reading order, liên quan trực tiếp đến PDF text-layer. |
| Layout/region | DocLayNet | Đánh giá khả năng nhận diện vùng heading, paragraph, table, figure, caption. |
| Layout khoa học/phức tạp | PubLayNet | Dùng làm kiểm tra bổ sung cho layout nhiều vùng; không phải claim chính. |
| Table detection | PubTables detection | Chứng minh hệ thống phát hiện vùng bảng tốt. |
| Table structure | PubTables structure OCR/proxy words | Chứng minh bảng không chỉ được đọc như text phẳng mà có hàng/cột/cell. |
| QA end-to-end domain gần | QCDT / Operations / real PDF quy chế | Quan trọng nhất để chứng minh hệ thống chạy tốt trên phạm vi tài liệu mục tiêu. |
| Retrieval | Hit@k, Recall@k, MRR@k, NDCG@k | Chứng minh chất lượng lấy evidence trước khi trả lời. |
| Grounded QA | answer_match, evidence_match, grounded_rate, hallucination_rate | Chứng minh câu trả lời dựa trên evidence và có citation. |

## 2. Benchmark hỗ trợ nên nhắc ngắn

Các benchmark này vẫn có giá trị, nhưng không nên biến thành trọng tâm.

| Tầng đánh giá | Benchmark | Cách diễn giải |
|---|---|---|
| OCR scan | OCR-D / FUNSD / OCR scan synthetic | Cho thấy hệ thống có nhánh OCR, nhưng scan mờ/nhiễu không phải phạm vi chính. |
| Citation công khai | SciFact | Dùng để chứng minh khả năng evidence/citation trên benchmark công khai, dù SciFact không phải natural PDF QA. |
| Scientific QA khó | QASPER | Dùng làm phân tích hạn chế với paper dài/free-form QA, không phải benchmark chính của phạm vi mới. |
| Academic extraction proxy | Nougat/arXiv proxy | Chỉ nên nêu như hướng mở rộng cho paper khoa học, không phải trọng tâm. |

## 3. Benchmark không nên nhấn mạnh

Không nên lấy các kết quả sau làm claim chính:

- QASPER answer correctness thấp: dùng để phân tích hạn chế answer synthesis, không dùng để phủ định phạm vi chính.
- Exact CSV/HTML của bảng: metric này quá nghiêm ngặt, chỉ cần sai một ký tự/merged cell là fail; nên báo cáo kèm F1 cell/structure.
- OCR-D CER cao: tài liệu lịch sử có ký tự cổ/Fraktur-like, nên token F1 phù hợp hơn để diễn giải.
- Nougat/arXiv proxy thấp: không thuộc nhóm PDF quy chế/quy định.

## 4. Bộ benchmark tối thiểu để bảo vệ

Nếu thời gian trình bày ngắn, chỉ cần 5 nhóm:

1. **Ingest/read order**: Bast-Korzen proxy.
2. **Layout/region**: DocLayNet hoặc PubLayNet.
3. **Table**: PubTables detection + PubTables structure.
4. **Retrieval**: Hit@k, Recall@k, MRR@k, NDCG@k.
5. **QA end-to-end**: QCDT/Operations/real PDF quy chế với grounded_rate và hallucination_rate.

## 5. Cách kết luận

Nên trình bày:

> Hệ thống được đánh giá nhiều tầng: ingest, layout/table, retrieval và QA end-to-end. Với phạm vi PDF text-layer dạng quy chế/quy định/hướng dẫn, benchmark chính là các bộ đo text extraction, region/table handling, retrieval evidence và grounded QA. Các benchmark OCR, QASPER, Nougat được giữ như phần phân tích mở rộng để chỉ ra giới hạn và hướng phát triển, không phải claim chính của đồ án.

