# Region Routing Large Ablation Results

## Muc tieu

Benchmark nay so sanh full ingest khi tat/bat region-level routing tren cung mot
tap PDF. Muc tieu khong phai chung minh region routing luon tot hon, ma chung
minh ba diem:

1. Khi PDF phu hop, region routing tao duoc block co trace backend ro rang.
2. Khi PDF scan/proxy khong phu hop voi region, pipeline fallback an toan sang OCR.
3. Region routing co chi phi latency, nen can trinh bay nhu trade-off giua
   trace/structure va toc do.

## Command

```powershell
.\.venv-gpu\Scripts\python.exe scripts\benchmark_region_routing_ablation.py `
  --pdf data\real_pdfs\QCDT_2025_5445_QD-DHBK.pdf `
  --glob "data/benchmarks/pubtables_structure/pdfs/*.pdf" `
  --glob "data/benchmarks/ocr_scan_25/pdfs/scan_ocr_00[1-5].pdf" `
  --out results\region_routing_ablation\qcdt_pubtables_ocr31
```

## Lenh chay lai nhanh

### 1. Tao overlay cho mot trang QCDT

```powershell
$pdf = "data\real_pdfs\QCDT_2025_5445_QD-DHBK.pdf"
.\.venv-gpu\Scripts\python.exe scripts\draw_region_overlay.py $pdf --page 6 --out docs\chapter5\figures\qcdt_page6_region_overlay.png
```

### 2. So sanh region OFF/ON tren mot trang QCDT

```powershell
$pdf = "data\real_pdfs\QCDT_2025_5445_QD-DHBK.pdf"
.\.venv-gpu\Scripts\python.exe scripts\compare_region_routing.py $pdf --page 6 --out-dir docs\chapter5\region_compare
```

Output:

```text
docs/chapter5/region_compare/QCDT_2025_5445_QD-DHBK_page6_region_compare.json
docs/chapter5/region_compare/QCDT_2025_5445_QD-DHBK_page6_region_compare.md
```

### 3. Chay lai benchmark lon 31 PDF

```powershell
.\.venv-gpu\Scripts\python.exe scripts\benchmark_region_routing_ablation.py `
  --pdf data\real_pdfs\QCDT_2025_5445_QD-DHBK.pdf `
  --glob "data/benchmarks/pubtables_structure/pdfs/*.pdf" `
  --glob "data/benchmarks/ocr_scan_25/pdfs/scan_ocr_00[1-5].pdf" `
  --out results\region_routing_ablation\qcdt_pubtables_ocr31
```

### 4. Chay lai benchmark chi rieng QCDT

```powershell
.\.venv-gpu\Scripts\python.exe scripts\benchmark_region_routing_ablation.py `
  --pdf data\real_pdfs\QCDT_2025_5445_QD-DHBK.pdf `
  --out results\region_routing_ablation\qcdt_only
```

### 5. Validation sau khi sua code region

```powershell
.\.venv-gpu\Scripts\python.exe -m compileall app scripts
.\.venv-gpu\Scripts\python.exe -m pytest tests\test_region_level_routing.py -q
```

Ket qua duoc ghi tai:

```text
results/region_routing_ablation/qcdt_pubtables_ocr31/summary.json
results/region_routing_ablation/qcdt_pubtables_ocr31/per_doc.jsonl
results/region_routing_ablation/qcdt_pubtables_ocr31/README.md
```

Tap benchmark gom 31 PDF:

- 1 PDF QCDT thuc te;
- 25 PDF PubTables structure proxy;
- 5 PDF OCR scan synthetic.

Moi PDF duoc ingest hai lan:

- `BOXBIIBOO_ENABLE_REGION_ROUTING=0`;
- `BOXBIIBOO_ENABLE_REGION_ROUTING=1`.

## Ket qua tong hop 31 PDF

| Config | Success | Backend counts | Latency mean | Blocks mean | Chunks mean | Table blocks mean | Route-traced blocks mean |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Region OFF | 1.000 | `ocr: 30`, `text: 1` | 6.189s | 65.19 | 53.48 | 0.97 | 0.00 |
| Region ON | 1.000 | `ocr: 30`, `region_routed: 1` | 16.556s | 64.45 | 53.23 | 1.26 | 15.94 |

Delta trung binh:

| Metric | Delta ON - OFF |
| --- | ---: |
| Backend changed | 1 / 31 PDF |
| Block count | -0.742 |
| Chunk count | -0.258 |
| Table block count | +0.290 |
| Route-traced block count | +15.935 |
| Latency | +10.367s |

## Dien giai tong hop

Tren 30 PDF scan/proxy, ca Region OFF va Region ON deu fallback sang `ocr`.
Dieu nay la bang chung ve fallback an toan: bat region khong lam hong pipeline,
nhung region khong duoc su dung khi probe/backend validation danh gia OCR phu hop
hon.

Tren PDF QCDT thuc te, backend thay doi ro:

| Config | Used backend | Blocks | Chunks | Table blocks | Route-traced blocks |
| --- | --- | ---: | ---: | ---: | ---: |
| Region OFF | `text` | 517 | 246 | 0 | 0 |
| Region ON | `region_routed` | 494 | 238 | 9 | 494 |

Voi QCDT:

- Region OFF dung text backend, khong tao table block nao.
- Region ON dung `region_routed`, tao 9 table blocks.
- Tat ca 494 blocks trong Region ON co route trace/region trace.
- Region ON tao route backend ro rang: `text`, `table`, `hybrid_tatr`, `placeholder`.

Day la vi du tot nhat de dua vao do an vi no cho thay neu khong dung region thi
bang bi xu ly nhu text thuong; khi dung region, bang duoc route vao table pipeline.

## Bang chi tiet QCDT

| Metric | Region OFF | Region ON |
| --- | ---: | ---: |
| Used backend | `text` | `region_routed` |
| Page count | 34 | 34 |
| Latency | 0.349s | 16.595s |
| Block count | 517 | 494 |
| Chunk count | 246 | 238 |
| Table block count | 0 | 9 |
| Route-traced block count | 0 | 494 |
| Region-traced block count | 0 | 494 |

Block type counts:

| Block type | Region OFF | Region ON |
| --- | ---: | ---: |
| heading | 240 | 220 |
| paragraph | 36 | 29 |
| list_item | 214 | 212 |
| metadata | 27 | 23 |
| table | 0 | 9 |
| figure | 0 | 1 |

Route backend counts voi Region ON:

| Route backend | Count |
| --- | ---: |
| text | 484 |
| table | 8 |
| hybrid_tatr | 1 |
| placeholder | 1 |

Region type counts voi Region ON:

| Region type | Count |
| --- | ---: |
| paragraph | 133 |
| list_item | 325 |
| heading | 25 |
| table | 9 |
| header | 1 |
| image | 1 |

## Ket luan de trinh bay voi hoi dong

Nen noi:

> Region-level routing khong thay the text/OCR/layout backend. No la lop dieu
> phoi giup pipeline chon backend theo tung vung. Ket qua tren QCDT cho thay khi
> tat region, PDF duoc xu ly bang text backend va khong sinh table block; khi bat
> region, he thong dung `region_routed`, sinh 9 table blocks va moi block deu co
> trace backend/region. Tren cac PDF scan/proxy, pipeline fallback sang OCR, cho
> thay co che validation/fallback van an toan.

Khong nen noi:

> Region routing luon nhanh hon hoac luon tot hon tat ca dataset.

Nen ghi ro trade-off:

> Region ON tang latency tren QCDT do phai detect/route tung vung va xu ly bang
> table pipeline. Doi lai, he thong co cau truc bang, route trace va kha nang
> debug/citation tot hon.

## Han che

- Benchmark nay chua co ground truth block-level cho tat ca PDF, nen khong do
  truc tiep F1 cua region detector.
- PubTables proxy va OCR scan synthetic trong tap nay chay qua duong OCR, nen
  khong phai bang chung truc tiep ve cai thien region.
- De chung minh retrieval cai thien do region, can them benchmark:

```text
region OFF ingest -> build index -> retrieval/QA
region ON ingest  -> build index -> retrieval/QA
same queries, compare Hit@k/Recall/MRR/answer/evidence.
```
