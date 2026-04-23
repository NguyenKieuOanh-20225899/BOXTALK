# Tổng hợp kết quả cuối cùng

Tài liệu này tóm tắt trạng thái end-to-end hiện tại của dự án dựa trên mã nguồn mới nhất và các tệp benchmark trong repo.

## 1. Tổng quan hệ thống hiện tại

### Kiến trúc chính

- Luồng QA chính: `routed_grounded`
- Baseline từ vựng mạnh: `bm25_only`
- Các nhánh thử nghiệm:
  - `adaptive_route_retry`
  - `grounded_llm_fallback`
  - `adaptive_route_retry_with_final_route_llm_fallback`

### Bề mặt sản phẩm hiện tại

Repo hiện có:

- Pipeline ingest PDF với các đường xử lý `text/layout/OCR/mixed`
- Stack truy xuất: BM25, dense, hybrid, rerank
- Grounded QA với evidence checker và citations
- Adaptive retry như một nhánh nghiên cứu
- Grounded LLM fallback bật theo yêu cầu, theo từng modality:
  - `text`
  - `table`
  - `formula`
  - `figure/caption` qua textual fallback
  - `multi_span`
- Hạ tầng benchmark và regression gate
- MVP UI chạy bằng FastAPI:
  - tải PDF lên
  - thư viện tài liệu
  - grounded QA
  - source viewer
  - nút bật/tắt developer trace

### Diễn giải hiện tại

- `routed_grounded` là luồng trả lời grounded mang tính production-like.
- `bm25_only` vẫn là baseline lexical rẻ và mạnh nhất.
- `adaptive_route_retry` vẫn đang ở mức thử nghiệm vì latency cao và lợi ích tổng hợp chưa đủ rõ.
- `grounded_llm_fallback` là experimental và hiện chủ yếu hữu ích ở các case tập trung vào bảng biểu.

## 2. Các artifact nguồn dùng cho bản tổng hợp này

- `results/user_pdf_benchmark_suite/llm_fallback_gate_recheck/suite_summary.json`
- `results/user_pdf_benchmark_suite/current/suite_summary.json`
- `results/user_pdf_benchmark_suite/attention_smoke/documents/attention_scientific_en/qa_summary.json`
- `results/qa_benchmark/attention_scientific_focus/qa_summary.json`
- `results/llm_fallback_benchmark/dummy_smoke/comparison_summary.json`
- `results/qa_benchmark/llm_fallback_smoke/qa_summary.json`
- `results/retrieval_readiness/20260420T150853Z/readiness_report.json`
- `results/retrieval_benchmark/smoke_bm25_only/benchmark_summary.json`
- `results/retrieval_benchmark/smoke_real_minilm_after/benchmark_summary.json`
- `results/beir_retrieval_benchmark/scifact_*/beir_summary.json`
- `docs/BASELINE_REGRESSION_GATES.md`

## 3. Các kết quả benchmark chính

### A. Ingest

Mức độ sẵn sàng của scientific ingest hiện dựa trên đường GPU `model_routed_doclaynet` và ổn định trên các mẫu PubTables.

| Giới hạn PubTables | Success | IoU@0.50 micro F1 | IoU@0.75 micro F1 | P95 latency |
|---:|---:|---:|---:|---:|
| 25 | 1.000 | 1.000 | 0.818 | 0.844s |
| 100 | 1.000 | 0.988 | 0.870 | 0.757s |
| 500 | 1.000 | 0.977 | 0.910 | 0.744s |

Các điểm rút ra:

- Scientific ingest đã vượt qua toàn bộ readiness gates trên đường GPU hiện tại.
- Backend chiếm ưu thế đang ổn định: `model_layout`.
- Kết luận về readiness phía scientific là `true`.
- Production ingest vẫn chưa thể khẳng định là sẵn sàng vì chưa có bộ PDF production có nhãn.

Lưu ý:

- Một lần chạy PubTables chỉ dùng baseline riêng trong môi trường hiện tại đã thất bại do lỗi OCR/runtime, vì vậy baseline ingest được giữ lại để báo cáo là đường readiness ổn định `model_routed_doclaynet`, thay vì tham chiếu baseline local bị lỗi đó.

### B. Retrieval

#### Smoke retrieval có kiểm soát

`results/retrieval_benchmark/smoke_bm25_only/benchmark_summary.json`

| Chiến lược | Hit@5 | Recall@5 | MRR@5 | NDCG@5 | Avg latency |
|---|---:|---:|---:|---:|---:|
| `bm25` | 1.000 | 1.000 | 0.900 | 0.926 | 0.055 ms |

`results/retrieval_benchmark/smoke_real_minilm_after/benchmark_summary.json`

| Chiến lược | Hit@5 | Recall@5 | MRR@5 | NDCG@5 | Avg latency |
|---|---:|---:|---:|---:|---:|
| `bm25` | 1.000 | 1.000 | 0.900 | 0.926 | 0.152 ms |
| `dense` | 1.000 | 1.000 | 1.000 | 1.000 | 7.892 ms |
| `hybrid` | 1.000 | 1.000 | 0.900 | 0.926 | 6.708 ms |
| `hybrid_rerank` | 1.000 | 1.000 | 0.900 | 0.926 | 7.894 ms |

Diễn giải:

- BM25 vẫn là baseline lexical nhanh nhất.
- Dense MiniLM cho chất lượng xếp hạng tốt nhất trên bộ smoke nhỏ có kiểm soát.

#### Mẫu BEIR / SciFact

| Backend / chiến lược | Hit@10 | Recall@10 | MRR@10 | NDCG@10 | Avg latency |
|---|---:|---:|---:|---:|---:|
| `bm25` | 0.950 | 0.950 | 0.826 | 0.844 | 3.61 ms |
| `contriever dense` | 0.750 | 0.750 | 0.667 | 0.691 | 38.69 ms |
| `contriever hybrid` | 0.950 | 0.950 | 0.846 | 0.872 | 55.41 ms |
| `dpr dense` | 0.400 | 0.375 | 0.335 | 0.343 | 207.64 ms |
| `dpr hybrid` | 0.950 | 0.925 | 0.882 | 0.870 | 60.43 ms |
| `colbert` (mẫu 10 query) | 0.800 | 0.800 | 0.720 | 0.731 | 1892.17 ms |

Diễn giải:

- BM25 đã khá mạnh trên SciFact với vai trò baseline rẻ.
- Xếp hạng tốt nhất trên mẫu SciFact hiện tại đến từ hybrid retrieval, đặc biệt là DPR hybrid theo MRR và Contriever hybrid theo NDCG.
- ColBERT hiện còn quá chậm và chưa đủ cạnh tranh trên thiết lập mẫu đang dùng.

#### Kết luận về readiness của retrieval

Từ `results/retrieval_readiness/20260420T150853Z/readiness_report.json`:

- `retrieval_ready_for_prototyping = true`
- `retrieval_ready_for_production = false`

Lý do:

- Retrieval đủ mạnh cho prototyping và demo.
- Khả năng tuyên bố production-ready còn bị chặn bởi việc thiếu PDF production có nhãn.

### C. QA

#### Lần recheck gate chính thức mới nhất

`results/user_pdf_benchmark_suite/llm_fallback_gate_recheck/suite_summary.json`

| Config | Vai trò | Success | Answer match | Evidence | Grounded | Hallucination | Avg latency |
|---|---|---:|---:|---:|---:|---:|---:|
| `bm25_only` | baseline lexical mạnh | 0.835 | 0.835 | 0.942 | 1.000 | 0.010 | 4.18 ms |
| `routed_grounded` | luồng grounded chính | 0.864 | 0.864 | 1.000 | 1.000 | 0.000 | 18.51 ms |

Diễn giải tổng hợp:

- `routed_grounded` hiện là luồng QA tốt nhất tổng thể trong lần recheck chính thức mới nhất.
- `bm25_only` vẫn rất mạnh và đặc biệt hữu ích cho các tài liệu quy chế/chính sách tiếng Việt.
- Mức groundedness vẫn được kiểm soát hoàn toàn trên `routed_grounded`.

#### Ảnh chụp baseline đã khóa trước đó

Từ `docs/BASELINE_REGRESSION_GATES.md` và `results/user_pdf_benchmark_suite/current/suite_summary.json`:

| Config | Success | Grounded | Hallucination | Avg latency |
|---|---:|---:|---:|---:|
| `bm25_only` | 0.825 | 1.000 | 0.010 | 2.4 ms |
| `routed_grounded` | 0.845 | 1.000 | 0.000 | 12.6 ms |
| `adaptive_route_retry` | 0.816 | 1.000 | 0.000 | 18.6 ms |

Diễn giải:

- `adaptive_route_retry` có cải thiện một số trường hợp nhưng chưa thắng đủ rõ để trở thành mặc định.
- Các lần recheck sau đó đã cải thiện thêm hai baseline chính trong khi vẫn giữ gates ở trạng thái xanh.

#### QA theo loại tài liệu ở lần recheck mới nhất

| Loại tài liệu | `bm25_only` success | `routed_grounded` success | Bên thắng |
|---|---:|---:|---|
| `policy_regulation` | 0.800 | 0.725 | `bm25_only` |
| `handbook_manual` | 0.925 | 0.925 | hòa |
| `scientific_paper` | 0.739 | 1.000 | `routed_grounded` |

Đây là bản tóm tắt sạch nhất hiện tại về việc mỗi luồng mạnh ở đâu.

### D. QA trên bài báo khoa học

Đây từng là điểm yếu lớn nhất trước đây và hiện nay là khu vực cải thiện rõ nhất.

#### Trước khi cải thiện

Từ `results/user_pdf_benchmark_suite/attention_smoke/documents/attention_scientific_en/qa_summary.json`:

| Config | Success | Answer match | Evidence | Grounded | Hallucination |
|---|---:|---:|---:|---:|---:|
| `routed_grounded` | 0.478 | 0.478 | 0.870 | 1.000 | 0.000 |
| `adaptive_route_retry` | 0.478 | 0.478 | 0.870 | 1.000 | 0.000 |

#### Sau khi cải thiện

Từ `results/qa_benchmark/attention_scientific_focus/qa_summary.json` và các suite snapshot mới nhất:

| Config | Success | Answer match | Evidence | Grounded | Hallucination |
|---|---:|---:|---:|---:|---:|
| `bm25_only` | 0.739 | 0.739 | 0.783 | 1.000 | 0.000 |
| `routed_grounded` | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 |

Kết quả ròng:

- `routed_grounded` trên benchmark bài báo khoa học đã tăng từ `0.478` lên `1.000` về success end-to-end.
- Chất lượng evidence cũng tăng từ `0.870` lên `1.000`.
- Đây là cải thiện QA cụ thể mạnh nhất trong toàn dự án.

### E. LLM Fallback

#### Benchmark fallback tập trung, dùng dummy provider

Từ `results/llm_fallback_benchmark/dummy_smoke/comparison_summary.json`:

| Chỉ số | Standard (`routed_grounded`) | Config fallback |
|---|---:|---:|
| Success rate | 0.640 | 0.720 |
| Answer match rate | 0.640 | 0.720 |
| Fallback call rate | - | 0.240 |
| Fallback used rate | - | 0.080 |
| Helped count | - | 2 |
| Override count | - | 2 |
| Hallucination delta | - | 0.000 |

Phân rã quan trọng theo modality:

- Câu hỏi dạng bảng:
  - standard success: `0.444`
  - fallback success: `0.667`
  - tăng: `+0.222`
  - `table_rule_resolved_count = 2`
  - `table_llm_resolved_count = 0`

- Các query được đánh dấu `should_require_fallback = true`:
  - standard success: `0.556`
  - fallback success: `0.667`
  - tăng: `+0.111`

- Các case `weak_standard_answer`:
  - standard success: `0.400`
  - fallback success: `0.600`
  - tăng: `+0.200`

Diễn giải:

- Gain đo được hiện có thật, nhưng trong benchmark dummy thì nó đến từ rule-based table path chứ chưa phải từ provider LLM thật.
- Hạ tầng plumbing, policy, trace và benchmark cho fallback đã sẵn sàng.
- Gain của provider thật vẫn còn chờ benchmark ổn định.

#### Controlled handbook smoke với fallback bật

Từ `results/qa_benchmark/llm_fallback_smoke/qa_summary.json`:

- `routed_grounded`: success `0.925`
- `routed_grounded_with_llm_fallback`: success `0.925`
- fallback call rate `0.325`
- fallback used rate `0.000`
- hallucination vẫn là `0.000`

Diễn giải:

- Trên benchmark handbook có kiểm soát bình thường, fallback hiện tại vẫn an toàn và không gây nhiễu.
- Vì vậy, gain đo được hiện chưa mang tính tổng quát, mà vẫn tập trung ở những case fallback chuyên biệt.

## 4. Trạng thái baseline và gates

### Lịch sử khóa baseline

- Commit khóa baseline ban đầu: `5413c70` - `Lock retrieval QA baseline and gates`
- Commit nâng gate: `edad8f9` - `Raise QA baseline regression gates`
- Snapshot code/UI/fallback hiện tại: `b0a0f15` - `Add grounded fallback benchmarks and MVP PDF QA UI`

### Hard gates hiện tại

Từ `docs/BASELINE_REGRESSION_GATES.md`:

- user PDF suite có ít nhất `100` câu hỏi duy nhất và `3` tài liệu
- `bm25_only.end_to_end_success_rate >= 0.82`
- `routed_grounded.end_to_end_success_rate >= 0.83`
- `routed_grounded.grounded_rate >= 1.0`
- `routed_grounded.hallucination_rate <= 0.0`
- `scientific_paper / routed_grounded.end_to_end_success_rate >= 0.95`
- `scientific_paper / routed_grounded.evidence_match_rate >= 0.95`
- `scientific_paper / routed_grounded.hallucination_rate <= 0.0`
- verdict về scientific readiness phải pass

### Trạng thái gate mới nhất

Dùng `results/user_pdf_benchmark_suite/llm_fallback_gate_recheck/suite_summary.json`:

| Gate | Ngưỡng | Kết quả mới nhất | Trạng thái |
|---|---:|---:|---|
| `bm25_only` success | 0.82 | 0.835 | pass |
| `routed_grounded` success | 0.83 | 0.864 | pass |
| `routed_grounded` grounded | 1.00 | 1.000 | pass |
| `routed_grounded` hallucination | tối đa 0.00 | 0.000 | pass |
| scientific paper `routed_grounded` success | 0.95 | 1.000 | pass |
| scientific paper `routed_grounded` evidence | 0.95 | 1.000 | pass |
| scientific paper `routed_grounded` hallucination | tối đa 0.00 | 0.000 | pass |

Trạng thái chung:

- Các gate chính hiện đang pass.

## 5. Các kết luận chính

1. `bm25_only` vẫn là baseline mạnh nhất cho tài liệu quy chế/chính sách tiếng Việt.
2. `routed_grounded` là kiến trúc QA tốt nhất tổng thể trên user suite đa tài liệu.
3. QA trên bài báo khoa học đã cải thiện mạnh và hiện là điểm mạnh thay vì điểm yếu lớn nhất.
4. Mức groundedness và kiểm soát hallucination hiện rất tốt trên luồng chính:
   - `grounded_rate = 1.0`
   - `hallucination_rate = 0.0` với `routed_grounded`
5. Retrieval đủ mạnh cho prototyping trên các chế độ BM25, dense và hybrid, nhưng chưa thể tuyên bố production-ready.
6. Stack LLM fallback hiện đã benchmark được, trace được và sẵn sàng cho UI, nhưng vẫn là experimental.
7. Gain fallback hiện tại có thật trên benchmark tập trung, nhưng chủ yếu đến từ table rule-first path chứ chưa phải provider LLM thật.
8. Repo hiện đã có MVP UI dùng được trên nền grounded QA stack.

## 6. Các hạn chế hiện tại

- LLM fallback với real provider đã có plumbing và benchmark support, nhưng chưa có kết quả benchmark ổn định đủ mạnh để khóa thành gate.
- Phần `figure` hiện vẫn chỉ là textual fallback; chưa có đường trả lời grounded theo vision.
- Table rule-based hiện bao phủ tốt hơn các case lookup/range mapping đơn giản so với table reasoning khó hơn.
- `adaptive_route_retry` vẫn là experimental và chưa phải main path mặc định.
- UI hiện đã dùng được nhưng vẫn là mức MVP, chưa phải giao diện sản phẩm hoàn thiện.
- Khả năng tuyên bố production ingest/retrieval còn bị chặn bởi việc thiếu PDF production có nhãn.
- Một số lần chạy scientific ingest chỉ-baseline cũ đã lỗi do môi trường OCR/runtime local, nên baseline scientific readiness hiện thực chất là đường GPU `model_routed_doclaynet` ổn định.

## 7. Các bước tiếp theo được khuyến nghị

1. Chạy benchmark ổn định với real provider cho `grounded_llm_fallback`.
2. Nâng chất lượng table QA vượt ra ngoài rule-based lookup và interval mapping đơn giản.
3. Hiển thị rõ hơn việc fallback có được dùng trong UI developer view, và hiển thị nhẹ trong user mode khi hữu ích.
4. Giữ adaptive integration ở mức final-route-only cho đến khi chất lượng fallback với real provider được đo rõ.
5. Cải thiện việc đóng gói figure textual:
   - caption
   - đoạn văn lân cận
   - figure references
   - chất lượng evidence packet
6. Bổ sung PDF production có nhãn để hỗ trợ tuyên bố mạnh hơn về production readiness.
7. Nếu fallback với real provider chứng minh được độ ổn định, hãy thêm một gate thử nghiệm riêng cho benchmark fallback tập trung thay vì gộp ngay vào gate chính.

## 8. Phiên bản báo cáo ngắn

Nếu chỉ cần một slide hoặc một đoạn ngắn:

> Dự án hiện sử dụng `routed_grounded` làm luồng QA grounded chính, trong khi `bm25_only` được giữ như lexical baseline mạnh nhất và `adaptive_route_retry` cùng `grounded_llm_fallback` được giữ ở mức thử nghiệm. Scientific ingest ổn định trên PubTables 25/100/500 với success 100% và IoU/F1 cao trên đường GPU được giữ lại. Ở lần recheck mới nhất trên user PDF suite, `bm25_only` đạt success `0.835`, còn `routed_grounded` đạt `0.864` với `grounded_rate = 1.0` và `hallucination_rate = 0.0`. Cải thiện QA lớn nhất nằm ở bài báo khoa học, nơi `routed_grounded` tăng từ `0.478` lên `1.000` success. Lớp LLM fallback hiện đã benchmark được và an toàn, với gain hiện tại từ các case bảng tập trung trong benchmark dummy provider, nhưng vẫn đang ở mức thử nghiệm và chưa là một phần của gate phát hành chính.
