# PMC2147049_table_0 OCR -> Hybrid TATR Demo

## Muc tieu

Thu muc nay truc quan hoa pipeline mong muon cho bang scan/image:

```text
Anh bang goc
-> OCR word boxes
-> Hybrid TATR cell grid
-> Markdown/CSV output
```

Day la minh hoa dua tren ket qua component benchmark da co, khong phai production
pipeline end-to-end moi. Cu the:

| Thanh phan | Trang thai | Nguon file |
| --- | --- | --- |
| Anh bang goc | Du lieu benchmark | `data/benchmarks/pubtables_structure/images/PMC2147049_table_0.jpg` |
| OCR word boxes | Da benchmark/chuan bi bang PaddleOCR manifest | `data/benchmarks/pubtables_structure_ocr_words_pmc2147049_single/pubtables_structure_samples.jsonl` |
| Hybrid TATR cell grid | Da benchmark o table component | `docs/chapter5/ingest_visualizations/PMC2147049_table_0_backend_compare/hybrid_tatr/table_debug/PMC2147049_table_0.json` |
| Markdown/CSV output | Output tu Hybrid TATR component | `docs/chapter5/ingest_visualizations/PMC2147049_table_0_backend_compare/hybrid_tatr/predictions/PMC2147049_table_0.json` |
| Production OCR -> Hybrid TATR end-to-end | Huong tich hop tiep theo | De xuat flag `BOXBIIBOO_ENABLE_OCR_TO_HYBRID_TATR` |

## Artifacts

| File | Noi dung |
| --- | --- |
| `01_original_table_image.png` | Anh bang goc. |
| `02_ocr_word_boxes_overlay.png` | Overlay 179 OCR word boxes. |
| `03_hybrid_tatr_cell_grid_overlay.png` | Overlay 92 cell boxes tu Hybrid TATR. |
| `04_combined_ocr_words_and_tatr_cells.png` | Overlay ket hop word boxes va cell grid. |
| `05_hybrid_tatr_output.csv` | CSV output tu Hybrid TATR component. |
| `06_hybrid_tatr_output_table.md` | Markdown table output tu Hybrid TATR component. |
| `summary.json` | Metadata tom tat artifact. |

## Dien giai cho bao cao

Voi bang scan/image, OCR cung cap text va word boxes. TATR cung cap cau truc
hinh hoc cua bang. Hybrid TATR ket hop hai nguon nay de gan text vao cell va
sinh bang co cau truc. Trong BOXTALK hien tai, phan nay da duoc danh gia o muc
component benchmark voi PubTables structure co OCR word boxes. Production
pipeline end-to-end hien van fallback sang OCR cho file scan nay; viec noi truc
tiep OCR word boxes vao Hybrid TATR trong production nen duoc trinh bay la huong
mo rong co flag bat/tat de ablation.

## Ket qua nhanh cua component Hybrid TATR

- OCR word boxes: 179
- Hybrid TATR rows: 46
- Hybrid TATR columns: 2
- Hybrid TATR cells: 92
- CSV lines: 46
