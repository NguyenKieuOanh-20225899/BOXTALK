# Real Table QA Retrieval Results

File này tổng hợp kết quả truy xuất bảng trên hệ thống thật cho phần Table QA tiếng Việt.
Benchmark không gọi LLM và không dùng kết quả mô phỏng. Mục tiêu là đo xem retrieval top-5
có tìm đúng bảng, đúng hàng, đúng cột và đúng cell evidence hay không.

## Dữ liệu

- Corpus nguồn: `results/retrieval_index/qcdt_2025_5445_constraint_table_reconstruction/corpus.jsonl`
- Tập câu hỏi: `data/benchmarks/table_qa_vi_real_qcdt/queries.jsonl`
- Số câu hỏi: 8
- Bảng được đánh giá: `page_9_p0008_b0005`, trang 9 tài liệu QCDT
- Bảng mục tiêu: quan hệ giữa `Số lần vắng mặt` và `Điểm quá trình được cộng/trừ`

Phạm vi của benchmark này là một ablation nhỏ để kiểm tra biểu diễn bảng trong retrieval index,
không phải benchmark tổng quát cho mọi loại bảng. Các biến thể corpus được sinh từ cùng một corpus
nguồn để cô lập tác động của table-aware chunking và cell-level evidence. Vì vậy, kết quả này nên
được trình bày như một case study/ablation có kiểm soát, không dùng để claim rằng hệ thống đã giải
quyết hoàn chỉnh Table QA trên mọi PDF.

## Lệnh chạy lại

```powershell
.\.venv-gpu\Scripts\python.exe scripts\benchmark_real_table_qa_retrieval.py `
  --source-corpus results\retrieval_index\qcdt_2025_5445_constraint_table_reconstruction\corpus.jsonl `
  --queries data\benchmarks\table_qa_vi_real_qcdt\queries.jsonl `
  --out results\table_qa_vi_real_retrieval\chapter5_real `
  --top-k 5
```

## Kết quả chính

```latex
\begin{table}[H]
\centering
\caption{Kết quả truy xuất bảng trên hệ thống thật với tập Table QA tiếng Việt}
\label{tab:real-table-qa-retrieval-result}
\begin{tabular}{|p{4.2cm}|c|c|c|c|}
\hline
\textbf{Cấu hình} & \textbf{Table Hit@5} & \textbf{Row Match@5} & \textbf{Column Match@5} & \textbf{Cell Match@5} \\
\hline
Normal chunking & 1.000 & 0.000 & 1.000 & 0.000 \\
\hline
Table-aware chunking & 1.000 & 1.000 & 1.000 & 0.000 \\
\hline
Table-aware chunking + cell-level evidence & 1.000 & 1.000 & 1.000 & 0.625 \\
\hline
\end{tabular}
\end{table}
```

## Diễn giải

`Normal chunking` vẫn tìm được đúng bảng vì toàn bộ bảng ở dạng Markdown chứa các từ khóa như
`Số lần vắng mặt`, `0`, `1-2`, `3-4`, `>= 5`. Tuy nhiên cấu hình này không có metadata
`row_header`, `col_header`, `cell_text`, nên không thể tạo row-level hoặc cell-level evidence.
Nói cách khác, cấu hình này cho thấy tìm đúng bảng chưa đủ để tạo citation đúng hàng hoặc đúng ô.

`Table-aware chunking` bổ sung các chunk dạng table row. Vì vậy hệ thống tìm được đúng hàng
trong top-5, nhưng vẫn chưa có citation đến từng ô do các table cell được loại ra trong biến thể
ablation này.

`Table-aware chunking + cell-level evidence` giữ đầy đủ table summary, table structure, table row
và table cell. Cell Match@5 đạt 0.625, cho thấy cell-level evidence giúp retrieval truy xuất đúng
ô trong nhiều câu hỏi. Kết quả chưa đạt 1.000 vì một số câu reverse/cell lookup có cell tiêu đề
hàng (`cột Số lần vắng mặt`) được BM25 xếp cao hơn cell giá trị thật. Khi kiểm tra top-10, cell
đúng xuất hiện đầy đủ, tức là bottleneck nằm ở reranking/cell prioritization trong top-5.

## Phân tích lỗi top-5

| Nhóm lỗi | Quan sát | Nguyên nhân | Hướng cải thiện |
|---|---|---|---|
| Cell tiêu đề được xếp cao | Một số câu hỏi chứa cụm `số lần vắng mặt`, làm cell ở cột `Số lần vắng mặt` đứng cao hơn cell giá trị thật. | BM25 ưu tiên từ khóa xuất hiện trực tiếp, chưa phân biệt cell tiêu đề và cell giá trị. | Thêm rule/reranker ưu tiên cell có `cell_text` là giá trị trả lời, hoặc giảm điểm cell header-like. |
| Cell đúng nằm ngoài top-5 | Với một số câu reverse lookup, cell đúng xuất hiện khi tăng top-k lên 10. | Candidate recall có, nhưng thứ hạng chưa đủ tốt trong top-5. | Dùng table-aware reranking, cell prioritization hoặc query rewriting cho dạng reverse lookup. |
| Bảng nhỏ, số câu hỏi ít | Benchmark chỉ dùng 8 câu hỏi quanh một bảng QCDT. | Mục tiêu là minh họa có kiểm soát cho table-aware chunking. | Mở rộng sang nhiều bảng QCDT/PubTables/Table QA nếu cần claim tổng quát hơn. |

Lệnh kiểm tra top-10:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\benchmark_real_table_qa_retrieval.py `
  --source-corpus results\retrieval_index\qcdt_2025_5445_constraint_table_reconstruction\corpus.jsonl `
  --queries data\benchmarks\table_qa_vi_real_qcdt\queries.jsonl `
  --out results\table_qa_vi_real_retrieval\chapter5_real_top10_check `
  --top-k 10
```

Kết quả kiểm tra top-10 cho thấy cấu hình `Table-aware chunking + cell-level evidence` đạt
Cell Match@10 = 1.000. Do đó, lỗi chính ở top-5 là lỗi xếp hạng, không phải lỗi mất evidence.

## Nên viết trong báo cáo

Kết quả này chứng minh table-aware chunking không thay thế table extraction, mà biến bảng đã trích
xuất thành các đơn vị evidence phù hợp cho retrieval. Bảng phẳng có thể giúp tìm đúng bảng, nhưng
không đủ để dẫn chứng đúng hàng hoặc đúng ô. Cell-level evidence là điều kiện cần để tạo citation
chính xác, nhưng vẫn cần cơ chế reranking/boost tốt hơn để ưu tiên cell giá trị thay vì cell tiêu đề.

Nên diễn đạt an toàn:

- Có thể nói: table-aware chunking cải thiện khả năng truy xuất evidence theo hàng và ô trên case study QCDT.
- Có thể nói: cell-level metadata cho phép tạo citation ở mức ô khi retrieval chọn đúng cell.
- Không nên nói: Table QA đã hoàn hảo, hoặc cell citation luôn đúng trên mọi bảng.
- Không nên nói: kết quả này đại diện cho toàn bộ PubTables hoặc mọi PDF bảng.

## Kết quả và file sinh ra

- `results/table_qa_vi_real_retrieval/chapter5_real/summary.json`
- `results/table_qa_vi_real_retrieval/chapter5_real/per_query.json`
- `results/table_qa_vi_real_retrieval/chapter5_real/per_query.csv`
- `results/table_qa_vi_real_retrieval/chapter5_real/latex_table.tex`
- `results/table_qa_vi_real_retrieval/chapter5_real/README.md`
