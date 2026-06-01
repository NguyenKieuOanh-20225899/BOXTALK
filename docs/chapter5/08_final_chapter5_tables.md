# Final Chapter 5 Tables

## Dataset / benchmark

```latex
\begin{table}[h]
\centering
\caption{Cac bo benchmark su dung trong Chuong 5}
\begin{tabular}{llll}
\hline
Benchmark & Nhom danh gia & So mau & Muc tieu \\
\hline
Mock ingest & Ingest regression & 5 & Kiem tra pipeline khong regression \\
DocLayNet & Layout & 49 & Danh gia phat hien layout PDF \\
PubLayNet & Layout & 100 & Danh gia layout khoa hoc \\
PubTables & Table & 25/500 & Phat hien va cau truc bang \\
QCDT & Retrieval/QA & 40 & Hoi dap PDF phap quy tieng Viet \\
Operations & QA & 40 & Hoi dap co cau hoi absence/ambiguous \\
SciFact & Retrieval/QA & 300 & Evidence va citation khoa hoc \\
QASPER & Retrieval/QA & 500 & Gioi han natural scientific QA \\
\hline
\end{tabular}
\end{table}
```

## Ingest results

```latex
\begin{table}[h]
\centering
\caption{Ket qua ingest va layout tren cac benchmark}
\begin{tabular}{lrrl}
\hline
Benchmark & So mau & Metric chinh & Ghi chu \\
\hline
Mock ingest & 5 & success = 1.000 & Chay lai trong dot Chuong 5 \\
DocLayNet & 49 & F1@0.50 = 0.849, F1@0.75 = 0.807 & Tong hop tu results cu \\
PubLayNet & 100 & F1@0.50 = 0.739, F1@0.75 = 0.708 & Tong hop tu results cu \\
PubTables detection & 500 & F1@0.50 = 0.975, F1@0.75 = 0.914 & Tong hop tu results cu \\
OCR-D PAGE-XML & 19 & token F1 = 0.725 & OCR thuc te con kho \\
\hline
\end{tabular}
\end{table}
```

## Metric definitions

```latex
\begin{table}[h]
\centering
\caption{Cac metric chinh trong thuc nghiem}
\begin{tabular}{lll}
\hline
Nhom & Metric & Y nghia \\
\hline
Ingest & F1@IoU & Do khop vung layout/table theo nguong IoU \\
Table & Cell F1 & Do khop o bang theo hinh hoc cell \\
Table & Structure F1 & Do khop cau truc hang/cot cua bang \\
Retrieval & Hit@k & Ti le cau hoi co it nhat mot evidence dung trong top-k \\
Retrieval & Recall@k & Ti le evidence dung duoc thu hoi trong top-k \\
Retrieval & MRR & Thu hang trung binh cua evidence dung dau tien \\
Retrieval & NDCG & Chat luong xep hang co tinh vi tri \\
QA & Answer match & Ti le cau tra loi khop gold theo nguong danh gia \\
QA & Evidence match & Ti le lay duoc evidence dung \\
QA & Grounded rate & Ti le cau tra loi co citation/evidence grounding \\
QA & Hallucination rate & Ti le tra loi khong duoc grounding theo benchmark \\
\hline
\end{tabular}
\end{table}
```

## Table extraction ablation

```latex
\begin{table}[h]
\centering
\caption{So sanh cac backend trich xuat bang tren PubTables subset}
\begin{tabular}{lrrrrr}
\hline
Backend & Det. F1 & Cell F1@0.75 & Structure F1 & Text F1 & Exact CSV \\
\hline
Default & 0.940 & 0.149 & 0.199 & 0.909 & 0.000 \\
TATR & 0.987 & 0.103 & 0.010 & 0.015 & 0.000 \\
Hybrid TATR & 0.987 & 0.944 & 0.772 & 0.999 & 0.480 \\
\hline
\end{tabular}
\end{table}
```

## Retrieval ablation

```latex
\begin{table}[h]
\centering
\caption{So sanh BM25, dense, hybrid va hybrid rerank}
\begin{tabular}{llrrrr}
\hline
Dataset & Strategy & Hit@k & Recall@k & MRR@k & NDCG@k \\
\hline
SciFact@5 & BM25 & 0.713 & 0.693 & 0.569 & 0.594 \\
SciFact@5 & Dense & 0.713 & 0.697 & 0.580 & 0.604 \\
SciFact@5 & Hybrid & 0.793 & 0.771 & 0.654 & 0.675 \\
SciFact@5 & Hybrid rerank & 0.780 & 0.762 & 0.654 & 0.674 \\
QCDT@10 & BM25 & 0.550 & 0.550 & 0.387 & 0.469 \\
QCDT@10 & Dense & 0.400 & 0.400 & 0.258 & 0.331 \\
QCDT@10 & Hybrid & 0.600 & 0.600 & 0.407 & 0.492 \\
QCDT@10 & Hybrid rerank & 0.575 & 0.575 & 0.413 & 0.490 \\
\hline
\end{tabular}
\end{table}
```

## QA summary

```latex
\begin{table}[h]
\centering
\caption{Ket qua routed grounded QA}
\begin{tabular}{lrrrrr}
\hline
Benchmark & Answer & Evidence & Grounded & Hallucination & E2E \\
\hline
QCDT & 0.725 & 1.000 & 1.000 & 0.000 & 0.725 \\
Operations & 0.925 & 1.000 & 1.000 & 0.000 & 0.925 \\
SciFact & 0.220 & 0.727 & 1.000 & 0.000 & 0.203 \\
QASPER & 0.084 & 0.240 & 1.000 & 0.054 & 0.050 \\
\hline
\end{tabular}
\end{table}
```

## QASPER top-k

```latex
\begin{table}[h]
\centering
\caption{Anh huong cua top-k tren QASPER retrieval}
\begin{tabular}{lrrrr}
\hline
Config & Hit@20 & Recall@20 & MRR@20 & NDCG@20 \\
\hline
BM25 & 0.378 & 0.322 & 0.187 & 0.610 \\
Dense & 0.316 & 0.273 & 0.156 & 0.565 \\
Hybrid & 0.392 & 0.344 & 0.202 & 0.617 \\
Hybrid rerank & 0.392 & 0.344 & 0.202 & 0.617 \\
\hline
\end{tabular}
\end{table}
```

## Table QA

```latex
\begin{table}[h]
\centering
\caption{Ablation table-aware retrieval va cell-level citation}
\begin{tabular}{lrrrr}
\hline
Config & Answer Acc. & Evidence & Cell Citation & Hit@k \\
\hline
Default + normal retrieval & 0.000 & 0.000 & 0.000 & 0.000 \\
Hybrid TATR + normal retrieval & 0.000 & 0.000 & 0.000 & 0.000 \\
Hybrid TATR + table-aware retrieval & 1.000 & 1.000 & 0.000 & 1.000 \\
Hybrid TATR + table-aware + cell citation & 1.000 & 1.000 & 1.000 & 1.000 \\
\hline
\end{tabular}
\end{table}
```

## Error analysis

```latex
\begin{table}[h]
\centering
\caption{Nhom loi chinh va huong cai thien}
\begin{tabular}{lll}
\hline
Nhom loi & Tac dong & Huong cai thien \\
\hline
Cau hoi rong & Retrieval dung trang nhung thieu y & Multi-evidence aggregation \\
Dense tieng Viet phap quy & Dense-only kem BM25 & Embedding domain/Vietnamese \\
Answer synthesis & Evidence dung nhung answer sai & Evidence planning, grounded generation \\
Absence check & De tra loi qua muc & Sufficiency calibration \\
Section ambiguity & Citation sai tieu muc & Heading-path scoring \\
Merged table cell & Cell citation sai & Constraint-aware reconstruction \\
QASPER free-form & E2E success thap & Multi-hop retrieval va synthesis \\
OCR scan thuc te & Text/chunk loi & Benchmark scan Viet va OCR confidence \\
\hline
\end{tabular}
\end{table}
```
