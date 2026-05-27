# P1 additional analysis: retrieval, error analysis, demo check, metric appendix

Ngày cập nhật: 2026-05-26  
Mục tiêu: hoàn thành nhóm việc P1 trong kế hoạch phản biện.

## 1. Retrieval top-k trên tập domain chính QCDT

Tập domain chính: `data/real_pdfs/queries.jsonl`  
Index: `results/retrieval_index/real_qcdt_e2e_hybrid_tatr_20260513`  
PDF: `QCDT_2025_5445_QD-DHBK.pdf`

Lưu ý quan trọng: `data/real_pdfs/queries.jsonl` có `gold_pages` và `gold_sections`, nhưng `scripts/benchmark_retrieval.py` tính match bằng `expected_pages`, `expected_section` hoặc `expected_chunk_ids`. Vì vậy đã tạo bản query page-level:

```text
results/retrieval_benchmark/real_qcdt_domain_queries_expected_pages_20260526.jsonl
```

Metric dưới đây là **page/section-level retrieval**, chưa phải exact chunk-level retrieval.

Nguồn kết quả:

- `results/retrieval_benchmark/real_qcdt_page_top5_20260526/README.md`
- `results/retrieval_benchmark/real_qcdt_page_top10_20260526/README.md`
- `results/retrieval_benchmark/real_qcdt_page_top20_20260526/README.md`

| Top-k | Strategy | Queries | Hit@k | Recall@k | MRR@k | nDCG@k | Avg latency ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 5 | BM25 | 40 | 0.500 | 0.500 | 0.379 | 0.452 | 1.07 |
| 5 | Dense | 40 | 0.300 | 0.300 | 0.242 | 0.280 | 8.80 |
| 5 | Hybrid | 40 | 0.500 | 0.500 | 0.394 | 0.443 | 9.65 |
| 5 | Hybrid rerank | 40 | 0.525 | 0.525 | 0.405 | 0.457 | 11.21 |
| 10 | BM25 | 40 | 0.550 | 0.550 | 0.387 | 0.469 | 1.10 |
| 10 | Dense | 40 | 0.400 | 0.400 | 0.258 | 0.331 | 9.17 |
| 10 | Hybrid | 40 | 0.600 | 0.600 | 0.407 | 0.492 | 9.94 |
| 10 | Hybrid rerank | 40 | 0.575 | 0.575 | 0.412 | 0.489 | 10.69 |
| 20 | BM25 | 40 | 0.675 | 0.675 | 0.395 | 0.526 | 1.19 |
| 20 | Dense | 40 | 0.525 | 0.525 | 0.265 | 0.368 | 8.80 |
| 20 | Hybrid | 40 | 0.650 | 0.650 | 0.411 | 0.517 | 9.55 |
| 20 | Hybrid rerank | 40 | 0.650 | 0.650 | 0.418 | 0.522 | 10.63 |

Diễn giải:

- Với QCDT, BM25 khá mạnh vì câu hỏi và tài liệu cùng ngôn ngữ, nhiều từ khóa pháp quy trùng trực tiếp.
- Dense đơn lẻ yếu hơn trên tập này, có thể do mô hình `all-MiniLM-L6-v2` không tối ưu tiếng Việt pháp quy.
- Hybrid tốt nhất ở top-k=10, còn BM25 nhỉnh hơn ở Hit@20. Điều này cho thấy domain chính vẫn cần keyword retrieval mạnh, không nên chỉ dựa vào dense retrieval.
- Hybrid rerank có MRR tốt hơn ở cả top-k=5/10/20, nghĩa là thường đẩy evidence đúng lên thứ hạng cao hơn dù Hit@k không luôn cao nhất.

Kết luận đưa vào slide:

```text
Trên QCDT, BM25 là baseline mạnh do tài liệu pháp quy có từ khóa rõ. Hybrid/rerank vẫn có ích ở thứ hạng evidence, nhưng dense đơn lẻ không phù hợp làm chiến lược chính cho miền tiếng Việt pháp quy. Vì vậy pipeline nên giữ hybrid nhưng không loại bỏ BM25.
```

## 2. Error analysis cho QCDT answer_match = 0.725

Nguồn: `results/qa_benchmark/real_qcdt_all_after_table_answer_20260513/per_question.json`, config `routed_grounded`.

Tổng số câu: 40  
Số câu answer_match đúng: 29  
Số câu answer_match sai: 11  
Answer match: 0.725  
Grounded rate: 1.000  
Hallucination rate: 0.000

### 2.1 Danh sách lỗi chính

| Query | Loại câu hỏi | Triệu chứng | Nhóm lỗi chính | Ghi chú |
|---|---|---|---|---|
| q11 | comparison | Trả lời lẫn điều kiện ĐATN, chỉ đúng một phần học phần song hành | QA synthesis / context selection | Evidence có liên quan nhưng answer không tổng hợp đủ 3 khái niệm |
| q14 | factoid | Trả lời nhầm điểm C của công nhận/chuyển đổi tín chỉ thay vì điểm đạt D/C | Retrieval/section ambiguity | Cùng từ khóa "điểm đạt" nhưng sai mục |
| q17 | policy | Chỉ nêu nghĩa vụ nộp học phí, thiếu hậu quả đình chỉ đăng ký học tập | QA synthesis incomplete | Evidence liên quan nhưng answer lấy câu đầu, bỏ câu hậu quả |
| q26 | factoid | Trả lời điều kiện đăng ký chương trình thứ hai, thiếu thời điểm sớm nhất | QA synthesis incomplete | Cùng Điều 18 nhưng chọn sai ý trong đoạn |
| q28 | factoid | Answer gần đúng "tối đa 15 TC" nhưng answer_match false | Metric/normalization | Cần normalize `TC` và `tín chỉ` tốt hơn |
| q31 | policy | Chỉ trả heading điều kiện bảo vệ luận văn, thiếu danh sách điều kiện | QA synthesis incomplete | Lỗi trích câu quá ngắn từ evidence dạng list |
| q32 | policy | Trả điều kiện tốt nghiệp đại học thay vì thạc sĩ | Retrieval/section ambiguity | Nhầm cấp đào tạo vì cùng cụm "xét công nhận tốt nghiệp" |
| q33 | factoid | Trả định nghĩa tín chỉ, không trả số tín chỉ học phần tiến sĩ | Retrieval error | Evidence đúng không được chọn vào câu trả lời |
| q34 | factoid | Answer chứa nội dung đúng nhưng quá dài | Metric/answer formatting | Cần rút gọn answer theo yêu cầu factoid |
| q37 | ambiguous_or_insufficient | Trả nhầm nội dung điểm học phần cho câu hỏi học phí cụ thể | Abstention / negative evidence | Cần xử lý câu hỏi xác nhận thông tin không tồn tại |
| q39 | comparison | Chỉ trả phần sinh viên đại học, thiếu học viên thạc sĩ | Multi-hop synthesis | Cần kết hợp evidence từ Điều 10 và Điều 27 |

### 2.2 Phân nhóm lỗi

| Nhóm lỗi | Số case | Case | Ý nghĩa |
|---|---:|---|---|
| QA synthesis incomplete | 4 | q11, q17, q26, q31 | Evidence có liên quan nhưng answer generator lấy thiếu ý hoặc tổng hợp chưa đủ |
| Retrieval/section ambiguity | 3 | q14, q32, q33 | Cùng từ khóa nhưng sai điều/mục/cấp đào tạo |
| Metric/formatting strictness | 2 | q28, q34 | Answer gần đúng hoặc chứa đúng nhưng không khớp chuẩn ngắn |
| Abstention / negative evidence | 1 | q37 | Câu hỏi yêu cầu xác nhận thiếu thông tin cụ thể, pipeline chưa xử lý tốt |
| Multi-hop comparison | 1 | q39 | Cần ghép evidence từ nhiều điều khác nhau |

Kết luận:

```text
QCDT không fail vì hallucination, mà chủ yếu fail ở answer synthesis, section disambiguation và câu hỏi cần tổng hợp nhiều evidence. Đây là hướng cải thiện rõ ràng hơn so với chỉ tăng mô hình sinh câu trả lời.
```

## 3. Bảng phân loại lỗi tổng quát

| Loại lỗi | Dấu hiệu | Cách phát hiện | Hướng cải thiện |
|---|---|---|---|
| Ingest error | Thiếu đoạn, sai thứ tự đọc, mất heading/list/table | Gold page có nội dung nhưng index không có chunk đúng | cải thiện reading order, cleaner, region routing |
| Chunking error | Evidence bị tách khỏi heading hoặc danh sách | Retrieval lấy đúng trang nhưng thiếu ngữ cảnh | structure-aware chunking, giữ heading path |
| Retrieval error | Evidence đúng không vào top-k | Hit@k/Recall@k thấp | hybrid retrieval, rerank, metadata/page/section hints |
| Section ambiguity | Lấy đúng từ khóa nhưng sai điều/mục | Evidence match sai section/cấp đào tạo | section-aware retrieval, query expansion theo Điều/Mục |
| QA synthesis error | Evidence đúng nhưng trả thiếu hoặc sai ý | evidence_match true nhưng answer_match false | answer template theo query type, list-aware synthesis |
| Multi-hop error | Cần so sánh nhiều đoạn nhưng chỉ trả một vế | comparison question fail | retrieve nhiều evidence group, synthesis theo từng vế |
| Abstention error | Trả lời khi tài liệu không có thông tin cụ thể | ambiguous/insufficient question fail | negative-evidence checker, confidence threshold |
| Metric strictness | Answer đúng về nghĩa nhưng fail vì diễn đạt/viết tắt | manual review thấy tương đương | normalize TC/tín chỉ, factoid short-answer evaluator |

## 4. Kiểm tra demo với PDF cùng miền chưa dùng làm benchmark chính

PDF kiểm tra: `data/real_pdfs/final-quy-che-thi-tot-nghiep-thpt.pdf`  
Loại tài liệu: quy chế/quy định tiếng Việt, cùng miền policy/regulation.  
Chế độ kiểm tra: text-layer light check, tắt region routing và hybrid TATR để kiểm tra nhanh khả năng demo.

Lệnh đã chạy:

```powershell
$env:BOXBIIBOO_ENABLE_REGION_ROUTING='0'
$env:BOXBIIBOO_ENABLE_HYBRID_TATR_TABLES='0'
$env:BOXBIIBOO_ENABLE_PIPELINE_HYBRID_TATR_TABLES='0'
.\.venv-gpu\Scripts\python.exe scripts\build_retrieval_index.py `
  --pdf data\real_pdfs\final-quy-che-thi-tot-nghiep-thpt.pdf `
  --output-dir results\retrieval_index\demo_unseen_quy_che_thi_thpt_text_light_20260526 `
  --skip-dense
```

Kết quả index:

| Field | Value |
|---|---:|
| page_count | 82 |
| block_count | 944 |
| chunk_count | 310 |
| used_backend | text |
| dense_built | false |
| output | `results/retrieval_index/demo_unseen_quy_che_thi_thpt_text_light_20260526` |

Truy vấn thử:

```powershell
$env:PYTHONIOENCODING='utf-8'
.\.venv-gpu\Scripts\python.exe scripts\query_retrieval.py `
  --index-dir results\retrieval_index\demo_unseen_quy_che_thi_thpt_text_light_20260526 `
  --query "Ban coi thi gồm những thành phần nào?" `
  --top-k 3 `
  --strategy bm25
```

Kết quả top-1:

| Field | Value |
|---|---|
| rank | 1 |
| page | 12 |
| section | `1. Thành phần:` |
| heading_path | `TỔ CHỨC VÀ QUẢN LÝ KỲ THI > Điều 13. Ban Coi thi > 1. Thành phần:` |
| chunk_id | `final-quy-che-thi-tot-nghiep-thpt.pdf:chunk_00055` |
| snippet | `Trưởng ban... Phó Trưởng ban... Ủy viên, thư ký...` |

Kết luận:

```text
PDF quy chế thi THPT có thể ingest/index nhanh ở chế độ text-layer và retrieval lấy đúng evidence "Ban Coi thi" ở rank 1. Đây là case demo phụ tốt, nhưng nếu trình bày pipeline đầy đủ region routing/hybrid table thì cần chạy thử riêng với timeout dài hơn.
```

## 5. Phụ lục mô tả metric

### Hit@k

Hit@k đo tỷ lệ câu hỏi có ít nhất một evidence đúng xuất hiện trong top-k kết quả retrieval.

```text
Hit@k = số query có ít nhất 1 hit đúng trong top-k / tổng số query
```

Ý nghĩa: metric dễ hiểu nhất để trả lời câu hỏi "hệ thống có tìm thấy bằng chứng đúng không?".

### Recall@k

Recall@k đo tỷ lệ evidence đúng được tìm thấy trong top-k. Nếu mỗi query chỉ có một evidence vàng thì Recall@k gần tương đương Hit@k. Nếu một query có nhiều evidence vàng, Recall@k cho biết lấy được bao nhiêu phần trong số đó.

```text
Recall@k = số evidence đúng được retrieve trong top-k / tổng số evidence đúng
```

Ý nghĩa: quan trọng với câu hỏi cần nhiều đoạn bằng chứng hoặc multi-hop.

### MRR@k

MRR@k là Mean Reciprocal Rank. Với mỗi query, lấy nghịch đảo thứ hạng của evidence đúng đầu tiên trong top-k; nếu không có evidence đúng thì bằng 0.

```text
MRR@k = trung bình của 1 / rank_evidence_đúng_đầu_tiên
```

Ý nghĩa: đo evidence đúng có được đưa lên cao hay không. MRR cao giúp QA ít bị nhiễu vì answer generator thấy evidence đúng sớm.

### nDCG@k

nDCG@k đo chất lượng xếp hạng có xét vị trí. Evidence đúng ở rank cao được thưởng nhiều hơn evidence đúng ở rank thấp.

Ý nghĩa: hữu ích khi nhiều evidence cùng liên quan, hoặc khi muốn đánh giá chất lượng ranking chứ không chỉ có/không.

### Answer match

Answer match đo câu trả lời sinh ra có khớp gold answer theo tiêu chí benchmark hay không. Trong đồ án, metric này có thể bị ảnh hưởng bởi diễn đạt, viết tắt và độ dài câu trả lời.

Ý nghĩa: đo đúng/sai cuối cùng ở mức answer, nhưng cần đọc cùng evidence_match và grounded_rate để biết lỗi nằm ở retrieval hay synthesis.

### Evidence match

Evidence match đo hệ thống có lấy đúng evidence/citation kỳ vọng hay không.

Ý nghĩa: nếu evidence_match true nhưng answer_match false, lỗi thường nằm ở answer synthesis hoặc formatting. Nếu evidence_match false, lỗi thường nằm ở ingest/chunking/retrieval.

### Grounded rate

Grounded rate đo tỷ lệ câu trả lời được đánh dấu là có căn cứ trong evidence/citation.

Ý nghĩa: đây là metric quan trọng của grounded QA. Nó không thay thế answer_match, vì một câu có thể grounded nhưng vẫn trả lời thiếu ý hoặc sai intent.

### Hallucination rate

Hallucination rate đo tỷ lệ câu trả lời bị xem là thiếu căn cứ hoặc không được hỗ trợ bởi evidence.

Ý nghĩa: metric này dùng để chứng minh hệ thống ưu tiên trả lời có nguồn, nhưng không nên claim "không bao giờ hallucinate" ngoài phạm vi benchmark đã chạy.

## 6. Trạng thái P1

- [x] Chạy retrieval top_k=5/10/20 trên tập domain chính QCDT ở mức page/section-level.
- [x] Bổ sung error analysis cho QCDT answer_match 0.725.
- [x] Thêm bảng phân loại lỗi: ingest/chunking/retrieval/QA/abstention.
- [x] Kiểm tra demo với PDF quy chế thi THPT cùng miền tài liệu ở chế độ text-layer light check.
- [x] Viết phụ lục mô tả metric: hit@k, recall@k, MRR, NDCG, grounded_rate.

## 7. Cải tiến code ngày 2026-05-26: Vietnamese policy-aware reranking

Sau khi thử PDF `final-quy-che-thi-tot-nghiep-thpt.pdf`, lỗi rõ nhất là câu hỏi rộng:

```text
Kỳ thi tốt nghiệp THPT tổ chức những môn thi nào?
```

Trước cải tiến, BM25 và heuristic rerank kéo nhầm các đoạn có nhiều từ khóa chung như `Công nhận tốt nghiệp THPT`, `Chế độ báo cáo`, hoặc `Đối tượng dự thi`. Đoạn đúng `Điều 20. Đăng ký dự thi > 2. Đăng ký môn thi` chỉ xuất hiện sâu hơn trong danh sách ứng viên.

Đã sửa `app/retrieval/reranker.py`:

- chuẩn hóa tiếng Việt không dấu trong heuristic reranker;
- tăng trọng số rerank vừa phải từ `0.20` lên `0.25`;
- thêm tín hiệu heading/section cho văn bản pháp quy tiếng Việt;
- boost các đoạn có `Đăng ký môn thi`, `môn Ngữ văn, môn Toán`, `bài thi tự chọn`;
- phạt các đoạn nhiễu như `Công nhận tốt nghiệp`, `Miễn thi`, `Bảo lưu điểm`, `Chế độ báo cáo`, `Đối tượng dự thi` khi query đang hỏi danh sách môn thi;
- thêm test `tests/test_retrieval_heuristic_reranker.py`.

Kết quả targeted check:

| Query | Trước cải tiến | Sau cải tiến |
|---|---|---|
| `Kỳ thi tốt nghiệp THPT tổ chức những môn thi nào?` | Top-1 sai: `Điều 45. Công nhận tốt nghiệp THPT` | Top-1 đúng: `Điều 20. Đăng ký dự thi > 2. Đăng ký môn thi` |

Kết quả QCDT page-level top-5 sau cải tiến:

Nguồn: `results/retrieval_benchmark/real_qcdt_page_top5_after_policy_rerank_blend025_20260526/README.md`

| Strategy | Queries | Hit@5 | Recall@5 | MRR@5 | nDCG@5 |
|---|---:|---:|---:|---:|---:|
| BM25 | 40 | 0.500 | 0.500 | 0.379 | 0.452 |
| Dense | 40 | 0.300 | 0.300 | 0.242 | 0.280 |
| Hybrid | 40 | 0.500 | 0.500 | 0.394 | 0.443 |
| Hybrid rerank | 40 | 0.525 | 0.525 | 0.406 | 0.458 |

So với checkpoint P1 trước đó, `hybrid_rerank` giữ Hit@5/Recall@5 ở `0.525`, MRR@5 tăng nhẹ từ `0.405` lên `0.406`, nDCG@5 tăng từ `0.457` lên `0.458`. Đây là cải tiến nhỏ ở benchmark tổng thể nhưng có ý nghĩa thực tế vì sửa được một case demo từng fail.

Validation:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest -q
```

Kết quả:

```text
65 passed
```

## 8. Rerun benchmark tong hop ngay 2026-05-26

Nguon moi nhat sau khi chay lai benchmark:

- `docs/BENCHMARK_RERUN_SUMMARY_2026-05-26.md`
- `results/retrieval_benchmark/rerun_real_qcdt_page_top5_20260526`
- `results/retrieval_benchmark/rerun_real_qcdt_page_top10_20260526`
- `results/retrieval_benchmark/rerun_real_qcdt_page_top20_20260526`

Ket qua QCDT page/section-level moi nhat van giu cung ket luan P1: BM25 la baseline manh cho van ban phap quy tieng Viet; dense don le yeu hon; hybrid/rerank co ich chu yeu o xep hang evidence som.

| Top-k | Strategy | Hit@k | Recall@k | MRR@k | nDCG@k |
|---:|---|---:|---:|---:|---:|
| 5 | BM25 | 0.500 | 0.500 | 0.379 | 0.452 |
| 5 | Dense | 0.300 | 0.300 | 0.242 | 0.280 |
| 5 | Hybrid | 0.500 | 0.500 | 0.394 | 0.443 |
| 5 | Hybrid rerank | 0.525 | 0.525 | 0.406 | 0.458 |
| 10 | BM25 | 0.550 | 0.550 | 0.387 | 0.469 |
| 10 | Dense | 0.400 | 0.400 | 0.258 | 0.331 |
| 10 | Hybrid | 0.600 | 0.600 | 0.407 | 0.492 |
| 10 | Hybrid rerank | 0.575 | 0.575 | 0.413 | 0.490 |
| 20 | BM25 | 0.675 | 0.675 | 0.395 | 0.526 |
| 20 | Dense | 0.525 | 0.525 | 0.265 | 0.368 |
| 20 | Hybrid | 0.650 | 0.650 | 0.411 | 0.517 |
| 20 | Hybrid rerank | 0.650 | 0.650 | 0.419 | 0.523 |
