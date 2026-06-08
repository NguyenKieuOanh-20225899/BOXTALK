# Ablation LaTeX Tables Fixed

Các bảng ablation nên dùng `tabularx` hoặc `adjustbox` để tránh tràn lề.
Thêm các package sau vào preamble của file LaTeX chính:

```latex
\usepackage{tabularx}
\usepackage{array}
\usepackage{makecell}
\usepackage{adjustbox}
\renewcommand\theadfont{\bfseries}
\newcolumntype{Y}{>{\centering\arraybackslash}X}
```

## 1. Ablation truy xuất

```latex
\begin{table}[H]
\centering
\small
\caption{Ablation truy xuất trên các benchmark}
\label{tab:ablation-retrieval-summary}
\begin{tabularx}{\linewidth}{|p{2.3cm}|c|c|c|X|}
\hline
\textbf{Dataset} & \textbf{Top-k} & \textbf{BM25} & \textbf{Hybrid} & \textbf{Nhận xét} \\
\hline
SciFact & 5 & 0.713 & 0.793 & Hybrid cải thiện rõ rệt so với BM25 và dense riêng lẻ. \\
\hline
QCDT & 10 & 0.550 & 0.600 & BM25 đã mạnh nhờ khớp từ khóa pháp quy, nhưng hybrid vẫn cải thiện nhẹ. \\
\hline
QASPER & 20 & 0.378 & 0.392 & Hybrid cải thiện nhỏ, nhưng bài toán vẫn khó do evidence dài và phân tán. \\
\hline
\end{tabularx}
\end{table}
```

## 2. Ablation grounding/citation

Bảng này là bảng dễ tràn nhất. Dùng `tabularx` và viết tắt tên cột.

```latex
\begin{table}[H]
\centering
\small
\caption{Ablation grounding và citation trên benchmark QA}
\label{tab:ablation-grounding-citation}
\begin{tabularx}{\linewidth}{|p{2.2cm}|X|c|c|c|c|}
\hline
\textbf{Benchmark} &
\textbf{Cấu hình} &
\thead{Answer\\match} &
\thead{Evidence\\match} &
\textbf{Grounded} &
\thead{Halluc.} \\
\hline
QCDT & \texttt{routed\_grounded} & 0.725 & 1.000 & 1.000 & 0.000 \\
\hline
QCDT & \texttt{no\_citation\_grounding} & 0.725 & 1.000 & 0.000 & 1.000 \\
\hline
Operations & \texttt{routed\_grounded} & 0.925 & 1.000 & 1.000 & 0.000 \\
\hline
Operations & \texttt{no\_evidence\_checker} & 0.775 & 0.850 & 1.000 & 0.150 \\
\hline
Operations & \texttt{no\_citation\_grounding} & 0.925 & 1.000 & 0.175 & 0.825 \\
\hline
\end{tabularx}
\end{table}
```

Nếu vẫn tràn ở bản in, dùng bản xoay ngang:

```latex
\begin{landscape}
% đặt bảng trên vào đây
\end{landscape}
```

và thêm package:

```latex
\usepackage{pdflscape}
```

## 3. Ablation trích xuất bảng

```latex
\begin{table}[H]
\centering
\small
\caption{Ablation các backend trích xuất bảng trên tập con PubTables structure}
\label{tab:ablation-table-extraction}
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{|l|c|c|c|c|c|}
\hline
\textbf{Backend} &
\thead{Detection\\F1} &
\thead{Cell\\F1@0.75} &
\thead{Structure\\F1} &
\thead{Text Assign.\\F1} &
\thead{Exact\\CSV} \\
\hline
Default & 0.940 & 0.149 & 0.199 & 0.909 & 0.000 \\
\hline
TATR & 0.987 & 0.103 & 0.010 & 0.015 & 0.000 \\
\hline
Hybrid TATR & 0.987 & 0.944 & 0.772 & 0.999 & 0.480 \\
\hline
\end{tabular}
\end{adjustbox}
\end{table}
```

## 4. Ablation table-aware chunking và cell citation

```latex
\begin{table}[H]
\centering
\small
\caption{Ablation table-aware chunking và cell-level evidence trên Table QA tiếng Việt}
\label{tab:ablation-table-aware-cell-citation}
\begin{tabularx}{\linewidth}{|X|c|c|c|c|}
\hline
\textbf{Cấu hình} &
\thead{Table\\Hit@5} &
\thead{Row\\Match@5} &
\thead{Column\\Match@5} &
\thead{Cell\\Match@5} \\
\hline
Normal chunking & 1.000 & 0.000 & 1.000 & 0.000 \\
\hline
Table-aware chunking & 1.000 & 1.000 & 1.000 & 0.000 \\
\hline
Table-aware chunking + cell-level evidence & 1.000 & 1.000 & 1.000 & 0.625 \\
\hline
\end{tabularx}
\end{table}
```

## 5. Tổng hợp đóng góp từng thành phần

```latex
\begin{table}[H]
\centering
\small
\caption{Tổng hợp vai trò của các thành phần trong pipeline}
\label{tab:ablation-component-summary}
\begin{tabularx}{\linewidth}{|p{2.6cm}|p{3.2cm}|X|X|}
\hline
\textbf{Thành phần} & \textbf{So sánh} & \textbf{Kết quả quan sát} & \textbf{Kết luận} \\
\hline
Hybrid retrieval & BM25 vs Dense vs Hybrid & Hybrid cải thiện so với từng thành phần riêng lẻ trên nhiều benchmark. & Kết hợp lexical và semantic là lựa chọn hợp lý cho pipeline chính. \\
\hline
Grounding/citation & \texttt{routed\_grounded} vs \texttt{no\_citation\_grounding} & Bỏ citation làm grounded rate giảm và hallucination tăng. & Citation là cơ chế quan trọng để kiểm soát hallucination và kiểm chứng câu trả lời. \\
\hline
Hybrid TATR & Default vs TATR vs Hybrid TATR & Hybrid TATR tăng mạnh Cell F1, Structure F1 và Exact CSV. & Tái dựng bảng cần kết hợp cấu trúc hình học và word boxes. \\
\hline
Table-aware chunking & Normal vs table-aware vs cell-level evidence & Normal tìm đúng bảng nhưng không có row/cell evidence; cell evidence cho phép citation mức ô. & Bảng cần được index theo cấu trúc để phục vụ Table QA. \\
\hline
\end{tabularx}
\end{table}
```

## Lưu ý

- Nếu bảng vẫn sát lề, đổi `\small` thành `\footnotesize`.
- Không nên dùng `\resizebox{\textwidth}{!}{...}` cho bảng nhiều chữ vì chữ bị co nhỏ khó đọc.
- Với bảng nhiều chữ, ưu tiên `tabularx` để tự xuống dòng.
