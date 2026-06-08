# QCDT Page 9 Text Layer -> Hybrid TATR Visualization

## Muc tieu

Thu muc nay truc quan hoa truong hop PDF co text layer ro. Hybrid TATR khong can OCR; he thong lay word boxes truc tiep tu text layer PDF, sau do dung TATR de nhan dien cau truc bang va gan word boxes vao cell.

```text
QCDT page 9 PDF text layer
-> PDF word boxes
-> Hybrid TATR cell grid
-> table Markdown/CSV
```

## Artifacts

| File | Noi dung |
| --- | --- |
| `01_page9_original.png` | Anh render goc cua page 9. |
| `02_page9_pdf_words_and_hybrid_tatr_cells.png` | Full-page overlay: PDF word boxes + Hybrid TATR cell grid. |
| `03_table1_original_crop.png` | Crop goc cua bang 1. |
| `04_table1_pdf_words_and_cell_grid.png` | Overlay word boxes va cell grid cua bang 1. |
| `05_table1_hybrid_tatr_output.csv` | CSV dau ra cua bang 1. |
| `06_table1_hybrid_tatr_output.md` | Markdown dau ra cua bang 1. |
| `03_table2_original_crop.png` | Crop goc cua bang 2. |
| `04_table2_pdf_words_and_cell_grid.png` | Overlay word boxes va cell grid cua bang 2. |
| `05_table2_hybrid_tatr_output.csv` | CSV dau ra cua bang 2. |
| `06_table2_hybrid_tatr_output.md` | Markdown dau ra cua bang 2. |
| `summary.json` | Metadata tom tat. |

## Ket qua nhanh

| Table | Backend | Text source | Rows | Cols | Cells | PDF word boxes |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | hybrid_tatr | pdf_text_words | 2 | 5 | 10 | 18 |
| 2 | hybrid_tatr | pdf_text_words | 3 | 10 | 30 | 50 |

## Dien giai cho bao cao

Day la truong hop khac voi PDF scan/image. Vi QCDT co text layer, Hybrid TATR dung `pdf_text_words` thay vi OCR. OCR chi can khi PDF la scan/image hoac text layer kem. Phan nay co the dung de minh hoa loi ich cua Hybrid TATR tren PDF text-layer: TATR cung cap grid hang/cot/cell, con PDF word boxes cung cap noi dung de sinh Markdown/CSV phuc vu chunking, retrieval va citation.
