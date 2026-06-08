# OCR Word Boxes To Hybrid TATR Extension

## Muc tieu

Huong mo rong nay danh cho PDF scan/image co bang. Trong cac file nay, lop
text PDF khong co hoac khong dang tin cay, nen Hybrid TATR khong the gan text
vao cell neu chi dua anh bang vao model. Can them mot buoc OCR de lay word
boxes, sau do dua word boxes nay vao Hybrid TATR.

Pipeline de xuat:

```text
PDF scan/image
-> OCR page/table region de lay text + word boxes
-> detect table region
-> crop anh bang
-> Hybrid TATR = TATR geometry + OCR word boxes
-> table_cells / table_csv / table_markdown
-> table-aware chunks
-> index/retrieval/QA
```

Y nghia cua tung thanh phan:

| Thanh phan | Vai tro |
| --- | --- |
| OCR | Doc chu va tao word boxes cho PDF scan/image. |
| Table region detector | Xac dinh vung nao la bang de khong xu ly toan trang nhu mot khoi text. |
| TATR | Nhan dien cau truc hinh hoc cua bang: hang, cot, cell. |
| Hybrid TATR | Gan OCR/PDF word boxes vao cac cell do TATR phat hien. |
| Table-aware chunking | Tao chunk theo summary, structure, row, cell de phuc vu retrieval va citation. |

## Trang thai hien tai trong code

Trong code hien tai, Hybrid TATR da hoat dong tot khi co word boxes tu PDF text
layer hoac tu manifest benchmark da chuan bi san. Vi du:

- PDF co text layer ro: Hybrid TATR dung `pdf_text_words`.
- PubTables structure benchmark: Hybrid TATR dung word boxes trong manifest.

Voi PDF scan/image end-to-end, pipeline production hien van thuong fallback sang
OCR. Vi du `PMC2147049_table_0.pdf` duoc full ingest voi `used_backend = ocr`.
Bang trong run nay duoc xu ly theo OCR/table-from-OCR, chua tu dong noi OCR word
boxes sang Hybrid TATR.

Do do, day nen duoc trinh bay la huong mo rong co co so tu component benchmark,
khong phai claim da tich hop hoan chinh trong production pipeline.

## Flag de xuat de bat/tat trong tuong lai

De de ablation va giai thich voi hoi dong, nen them flag production ro rang:

| Flag | Gia tri | Y nghia |
| --- | --- | --- |
| `BOXBIIBOO_ENABLE_OCR_TO_HYBRID_TATR` | `0` | Mac dinh an toan: scan PDF di theo OCR/table-from-OCR hien tai. |
| `BOXBIIBOO_ENABLE_OCR_TO_HYBRID_TATR` | `1` | Neu OCR sinh duoc word boxes va co table region, chay Hybrid TATR cho vung bang. |
| `BOXBIIBOO_OCR_TO_HYBRID_TATR_MIN_WORDS` | `5` | Chi chay Hybrid TATR khi vung bang co toi thieu so word boxes nay. |
| `BOXBIIBOO_OCR_TO_HYBRID_TATR_FALLBACK` | `ocr_table` | Neu Hybrid TATR loi/yeu, fallback ve bang tu OCR. |

Logic mong muon:

```text
if region.type == "table" and pdf_has_good_text_layer:
    run Hybrid TATR with PDF word boxes
elif region.type == "table" and scan_or_image_pdf:
    if BOXBIIBOO_ENABLE_OCR_TO_HYBRID_TATR=1:
        OCR table/page region -> OCR word boxes
        run Hybrid TATR with OCR word boxes
        if weak result: fallback to OCR table block
    else:
        run OCR/table-from-OCR baseline
```

## Cach so sanh khi co flag

Khi flag nay duoc tich hop, co the chay ablation:

```powershell
# OCR/table-from-OCR baseline
$env:BOXBIIBOO_ENABLE_OCR_TO_HYBRID_TATR='0'
.\.venv-gpu\Scripts\python.exe scripts\visualize_ingest_output.py `
  data\benchmarks\pubtables_structure\pdfs\PMC2147049_table_0.pdf `
  --out-dir docs\chapter5\ingest_visualizations\PMC2147049_table_0_ocr_only

# OCR word boxes + Hybrid TATR
$env:BOXBIIBOO_ENABLE_OCR_TO_HYBRID_TATR='1'
.\.venv-gpu\Scripts\python.exe scripts\visualize_ingest_output.py `
  data\benchmarks\pubtables_structure\pdfs\PMC2147049_table_0.pdf `
  --out-dir docs\chapter5\ingest_visualizations\PMC2147049_table_0_ocr_hybrid_tatr

Remove-Item Env:\BOXBIIBOO_ENABLE_OCR_TO_HYBRID_TATR
```

Metrics nen so sanh:

| Nhom | Metrics |
| --- | --- |
| Table structure | detection F1, cell IoU F1, structure F1, row/column error |
| Text assignment | text assignment F1, token F1 |
| Reconstruction | exact CSV/HTML, table markdown quality |
| Downstream | table retrieval hit@k, table answer accuracy, cell citation accuracy |
| Runtime | latency/page hoac latency/table region |

## Minh hoa PMC2147049_table_0

Da tao bo artifact truc quan tai:

`docs/chapter5/ingest_visualizations/PMC2147049_table_0_ocr_hybrid_tatr_demo`

No gom:

| File | Noi dung |
| --- | --- |
| `01_original_table_image.png` | Anh bang goc tu PubTables. |
| `02_ocr_word_boxes_overlay.png` | Overlay OCR word boxes tu manifest PaddleOCR. |
| `03_hybrid_tatr_cell_grid_overlay.png` | Overlay cell grid tu Hybrid TATR component benchmark. |
| `04_combined_ocr_words_and_tatr_cells.png` | Overlay ket hop OCR word boxes va Hybrid TATR cells. |
| `05_hybrid_tatr_output.csv` | CSV output tu Hybrid TATR component. |
| `06_hybrid_tatr_output_table.md` | Bang Markdown tu output Hybrid TATR. |

Can ghi ro khi dua vao bao cao: day la minh hoa component benchmark
`OCR word boxes + Hybrid TATR`, khong phai production ingest end-to-end moi.
Production pipeline hien van fallback sang OCR cho file scan/image nay; viec
noi OCR word boxes vao Hybrid TATR bang flag la huong tich hop tiep theo.

## Ly do can flag

Flag can thiet vi OCR -> Hybrid TATR co the cai thien cau truc bang scan, nhung
chi phi cao hon OCR/table-from-OCR va phu thuoc chat luong OCR word boxes. Neu
OCR sai vi tri hoac word boxes qua thua/thieu, Hybrid TATR van co the gan sai
text vao cell. Bat/tat flag giup chung minh bang ablation:

- khong dung Hybrid TATR thi scan table chi co text/row cluster tu OCR;
- dung Hybrid TATR thi co them cau truc cell/row/column ro hon;
- neu ket qua yeu, fallback ve OCR baseline de tranh lam hong pipeline.

## Cau viet an toan cho bao cao

> Voi PDF scan/image, OCR la buoc can thiet de lay text va word boxes. Mot huong
> mo rong hop ly la dua cac OCR word boxes nay vao Hybrid TATR cho cac vung bang,
> de ket hop kha nang nhan dien cau truc cua TATR voi noi dung van ban do OCR
> trich xuat. Trong phien ban hien tai, huong nay da duoc danh gia o muc
> component/benchmark co word boxes; viec tich hop thanh flag end-to-end trong
> pipeline production duoc de xuat nhu mot huong phat trien tiep theo.
