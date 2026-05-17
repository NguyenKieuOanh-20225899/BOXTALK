# Region-Level Routing Improvement - 2026-05-17

## Muc tieu

Lan cai tien nay tap trung vao ingest theo cap vung:

```text
Page
-> detect regions: text / table / image / caption / header / footer
-> sort theo reading order
-> text region dung text extractor
-> table region dung table extractor, co hybrid_tatr fallback khi can
-> image region dung OCR neu can, neu khong thi tao figure placeholder
-> gom lai thanh BlockNode
```

Muc tieu chinh la xu ly PDF hon hop tot hon: mot trang co text layer tot nhung co them bang/anh/caption/header/footer thi khong bi ep di qua mot backend duy nhat.

## Thay doi ky thuat

### 1. Region detector moi

File chinh:

- `app/ingest/region/detector.py`

Detector moi tao cac region co truong:

- `kind`: `paragraph`, `heading`, `list_item`, `table`, `caption`, `header`, `footer`, `image`
- `block_type`: kieu block se chuyen thanh `BlockNode`
- `bbox`: toa do vung tren page PDF
- `route_backend`: backend nen dung cho vung do
- `detection_source`: nguon phat hien, vi du `pdf_text_block`, `native_or_text_cluster`, `pdf_image_rect`

Table region duoc phat hien bang hai nguon:

- `page.find_tables()` cua PyMuPDF neu co.
- heuristic gom cac dong text giong bang dua tren multi-column spacing, `|`, tab, va cac dong lien tiep co overlap ngang.

Text nam trong table bbox se bi loai khoi text regions de tranh duplicate.

### 2. Region-routed extractor that su route theo region

File chinh:

- `app/ingest/extract/region_routed.py`

Truoc day region-routed chu yeu cat text block va image. Sau sua:

- `table` -> `extract_table_region`
- `paragraph/heading/list_item/caption/header/footer` -> `extract_text_region`
- `image` -> OCR region neu can, hoac figure placeholder

`header` va `footer` duoc dua vao `BlockNode.block_type = metadata`, nhung van giu `meta.region_kind = header/footer`.

### 3. Fast table first, hybrid TATR when needed

File chinh:

- `app/ingest/extract/table.py`
- `app/ingest/pipeline.py`

Bang co text-layer/grid tot se dung extractor nhanh truoc:

```text
table_words_grid -> table_clip_text -> hybrid_tatr -> OCR fallback
```

Neu nguoi dung ep:

```powershell
$env:BOXBIIBOO_TABLE_BACKEND="hybrid_tatr"
```

thi hybrid_tatr van duoc goi truoc.

Trong pipeline chinh, cac table block da co cau truc on dinh se khong bi hybrid_tatr override, tru khi hybrid_tatr duoc ep bang env. Dieu nay tranh goi TATR khong can thiet cho bang text-layer don gian.

### 4. Reading order band-aware

File chinh:

- `app/ingest/reading_order.py`

Sort don gian duoc doi tu `(y, x)` thanh band-aware:

- Neu cac region nam cung mot dai ngang, sort trai sang phai.
- Neu khong cung dai ngang, sort tu tren xuong duoi.

Muc dich la giam loi voi layout co bang ben trai va anh ben phai.

### 5. Vector figure detection optional

Vector drawing detection co the bat bang:

```powershell
$env:BOXBIIBOO_ENABLE_REGION_VECTOR_FIGURES="1"
```

Mac dinh tat de tranh them placeholder `Figure` vao text cua cac benchmark text extraction. Raster image region van duoc detect qua `page.get_images()`.

## Before / After

Benchmark mock truoc khi sua:

```text
Command:
.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset mock --limit 5 --out results\ingest\mock_before_region_routing_fix_20260517 --mode all
```

| Metric | Before |
|---|---:|
| success_rate | 1.000 |
| char_accuracy | 1.000 |
| token_f1 | 1.000 |
| normalized_text_similarity | 1.000 |
| reading_order_score | 1.000 |
| table_structure F1 | 1.000 |
| table_exact_csv | 1.000 |
| latency mean | 0.0055s |

Benchmark mock sau khi sua:

```text
Command:
.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset mock --limit 5 --out results\ingest\mock_after_region_routing_no_vector_default_20260517 --mode all
```

| Metric | After |
|---|---:|
| success_rate | 1.000 |
| char_accuracy | 1.000 |
| token_f1 | 1.000 |
| normalized_text_similarity | 1.000 |
| reading_order_score | 1.000 |
| table_structure F1 | 1.000 |
| table_exact_csv | 1.000 |
| latency mean | 0.0101s |

Ket qua mock khong tang metric vi truoc do mock da dat 1.0, nhung sau sua van giu duoc cac metric chinh va co them kha nang route theo region that su.

## Kiem thu moi

Them:

- `tests/test_region_level_routing.py`

Kiem tra:

- detect duoc table region tu text-layer.
- detect duoc caption/header/footer.
- optional vector figure region hoat dong khi bat env.
- table region duoc route qua table extractor.
- dong bang khong bi duplicate thanh paragraph.

Cap nhat:

- `tests/test_hybrid_tatr_table_module.py`

Kiem tra:

- bang da co cau truc on dinh khong bi hybrid_tatr override trong auto mode.
- neu force `BOXBIIBOO_TABLE_BACKEND=hybrid_tatr` thi van goi hybrid_tatr.

## Validation

```text
.\.venv-gpu\Scripts\python.exe -m compileall app scripts
Result: pass

.\.venv-gpu\Scripts\python.exe -m pytest -q
Result: 59 passed

.\.venv-gpu\Scripts\python.exe scripts\benchmark_ingest_suite.py --dataset mock --limit 5 --out results\ingest\mock_after_region_routing_no_vector_default_20260517 --mode all
Result: success_rate 1.000, error_count 0
```

## Han che con lai

- Table bbox tu text-layer van la bbox cua noi dung text, chua phai bbox layout rong nhu annotation DocLayNet/PubLayNet.
- Vector figure detection de optional vi them figure placeholder co the lam lech benchmark text extraction.
- Header/footer repeated removal van nam o cleaner, region detector moi chi gan nhan ban dau.
- Neu table khong co text-layer va khong detect duoc qua PyMuPDF/table heuristic, pipeline van can OCR/model-layout fallback.

## Ket luan

Nen merge vao `main` vi:

- Khong lam regression tren mock benchmark.
- Full pytest pass.
- Region routing dung kien truc hon: moi vung duoc route den extractor phu hop.
- Hybrid TATR duoc goi co dieu kien hon, tranh cham pipeline voi bang text-layer don gian.

Khong nen claim la da giai quyet hoan toan document layout analysis. Nen trinh bay la cai tien deterministic region-level routing co fallback an toan.
