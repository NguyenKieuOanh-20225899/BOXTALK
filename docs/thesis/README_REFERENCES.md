# Tai lieu tham khao cho do an

File `references.bib` chua cac nguon hoc thuat va cong bo chinh thuc phu hop
voi pipeline BOXTALK. Khong su dung Wikipedia, blog ca nhan hoac slide bai giang.

File `references.tex` chua cung danh muc o dang LaTeX `thebibliography`, dung
cho mau bao cao khong su dung BibTeX.

## Cau hinh LaTeX

Chep `references.bib` vao thu muc goc cua do an LaTeX. Trong file chinh
`DoAn.tex`, dat cac lenh sau o vi tri in danh muc tai lieu tham khao:

```latex
\bibliographystyle{IEEEtran}
\bibliography{references}
```

Bien dich theo thu tu:

```text
pdflatex DoAn
bibtex DoAn
pdflatex DoAn
pdflatex DoAn
```

Khong nen viet ca `\cite{...}` va noi dung tai lieu tham khao bang tay trong
chuong Tai lieu tham khao. Trong noi dung bao cao chi can dung `\cite{key}`.

## Dung truc tiep file LaTeX

Neu do an khong dung BibTeX, chep `references.tex` vao thu muc chuong cua bao
cao va them vao `DoAn.tex`:

```latex
\subfile{Chapters/references}
```

Chi chon mot trong hai cach: `references.bib` hoac `references.tex`. Khong dung
dong thoi ca hai vi danh muc se bi lap.

## Goi y dat trich dan

| Noi dung | Khoa BibTeX nen dung |
| --- | --- |
| Transformer | `vaswani2017transformer` |
| DETR va nen tang cua TATR | `carion2020detr` |
| DocLayNet | `pfitzmann2022doclaynet` |
| PubLayNet | `zhong2019publaynet` |
| PubTables-1M va Table Transformer | `smock2022pubtables` |
| PaddleOCR/PP-OCR | `du2020ppocr` |
| PyMuPDF | `pymupdf2026` |
| BM25 | `robertson2009bm25` |
| Dense Passage Retrieval | `karpukhin2020dpr` |
| Sentence-BERT | `reimers2019sentencebert` |
| MiniLM | `wang2020minilm` |
| Contriever | `izacard2022contriever` |
| ColBERT | `khattab2020colbert` |
| FAISS | `johnson2021faiss` |
| Reciprocal Rank Fusion | `cormack2009rrf` |
| Retrieval-Augmented Generation | `lewis2020rag` |
| Query-aware/adaptive RAG | `jeong2024adaptiverag` |
| Evidence checking/self-reflection | `asai2024selfrag` |
| BEIR retrieval benchmark | `thakur2021beir` |
| SciFact | `wadden2020scifact` |
| QASPER | `dasigi2021qasper` |
| NDCG | `jarvelin2002ndcg` |
| Tai lieu QCDT | `hust2025trainingregulation` |

## Vi du trong bao cao

```latex
BM25 duoc su dung lam phuong phap truy xuat theo tu khoa
\cite{robertson2009bm25}. Truy xuat ngu nghia su dung bieu dien cau theo
Sentence-BERT va MiniLM \cite{reimers2019sentencebert,wang2020minilm}.

Mo-dun bang duoc xay dung dua tren Table Transformer va duoc danh gia tren
PubTables-1M \cite{smock2022pubtables}.

Pipeline hoi dap ke thua y tuong Retrieval-Augmented Generation
\cite{lewis2020rag}; viec lua chon chien luoc theo loai cau hoi co lien quan
den huong Adaptive-RAG \cite{jeong2024adaptiverag}.

SciFact va QASPER duoc su dung de danh gia truy xuat evidence va hoi dap tren
tai lieu khoa hoc \cite{wadden2020scifact,dasigi2021qasper}.
```

## Luu y ve cach claim

- `Hybrid TATR` va `table-aware chunking` la thiet ke tich hop cua do an, khong
  phai ten phuong phap trong bai bao PubTables-1M.
- Khi trich dan Adaptive-RAG hoac Self-RAG, nen viet he thong "tham khao y
  tuong", khong khang dinh da cai dat lai day du cac phuong phap do.
- QCDT la van ban chinh thuc cua Dai hoc Bach khoa Ha Noi va co the duoc trich
  dan nhu tai lieu cua to chuc.
