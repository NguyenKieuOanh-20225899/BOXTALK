# Terminal ingest demo

Thu muc `demo/` chua chuong trinh trinh dien mot phut cho buoc tiep nhan va
trich xuat mot trang PDF. Tat ca ma nguon, cau hinh, dau vao mau va ket qua
trinh dien deu nam trong thu muc nay.

## Cam ket pham vi

Demo khong thay doi ma nguon san pham, khong sua API loi va khong cai dat lai
thuat toan xu ly PDF. Cac adapter trong `demo/adapters/` chi chuyen tham so,
goi lai module co san va ghi ket qua theo dinh dang tien cho buoi bao ve.

## Thanh phan cu duoc tai su dung

| Chuc nang demo | Thanh phan ma nguon cu duoc tai su dung |
| --- | --- |
| Tham do PDF | `app.ingest.probe.probe_pdf` |
| Phat hien bo cuc/vung | `app.ingest.region.detector.detect_regions` |
| Dinh tuyen vung | `app.ingest.extract.region_routed._extract_region`, `app.ingest.region.router.route_region` |
| Trich xuat van ban | `app.ingest.extract.text.extract_text_region` thong qua region-routed extractor |
| OCR | `app.ingest.extract.ocr.extract_ocr_region` thong qua region-routed extractor |
| Trich xuat bang | `app.ingest.extract.table.extract_table_region` thong qua region-routed extractor |
| Hybrid TATR | `app.ingest.extract.hybrid_tatr_table.extract_hybrid_tatr_table_region` khi cau hinh cho phep |
| Sinh overlay | `app.ingest.region.debug.draw_regions_debug` va bang mau `REGION_COLORS` |
| Giao dien terminal | Kieu terminal stdout co san trong `app.main_ingest_demo`; demo khong them Rich/Typer/Click |
| Serialization | Schema cu `BlockNode`, `PdfProbeResult`; adapter chi anh xa sang JSON demo |

## Moi truong

Chay tu thu muc goc repository. Neu chua cai package editable, demo van nap duoc
module cu qua `demo/bootstrap.py`.

Tren PowerShell co the dung truc tiep Python cua moi truong:

```powershell
.\.venv-gpu\Scripts\python.exe demo\run_ingest_demo.py --help
```

## Chuan bi PDF

Thu muc `demo/input/` co:

- `sample_page.pdf`: trang mau cho demo ingest ngan.
- `sample_table_page.pdf`: trang mau co bang de thu `--table-extractor hybrid_tatr`.

Co the thay bang PDF khac, mien la duong dan van nam trong repository hoac duoc
truyen ro qua `--pdf`.

## Chay demo mac dinh

```powershell
.\.venv-gpu\Scripts\python.exe demo\run_ingest_demo.py `
  --pdf demo\input\sample_page.pdf `
  --page 1 `
  --save-overlay `
  --output demo\output\ingest
```

Hoac:

```powershell
demo\scripts\run_ingest.bat
```

## Trich xuat va ve overlay khi tat region routing

Lenh nay khong goi `detect_regions` va khong dinh tuyen tung vung. Script dung
text backend toan trang `app.ingest.extract.text.extract_with_text_backend` de
cho thay khi tat region routing thi PDF duoc trich xuat thanh nhung block/text
nao, dong thoi ve lop phu bbox cua cac block do. Day la output doi chung voi
`run_ingest_demo.py`, trong do region routing van duoc bat.

```powershell
.\.venv-gpu\Scripts\python.exe demo\run_ingest_demo.py `
  --pdf demo\input\sample_page.pdf `
  --page 1 `
  --region-routing off `
  --save-overlay `
  --output demo\output\no_region_overlay
```

Hoac:

```powershell
demo\scripts\run_no_region_overlay.bat
```

Ket qua:

- `demo/output/no_region_overlay/page_01_overlay.png`
- `demo/output/no_region_overlay/page_01_blocks.json`
- `demo/output/no_region_overlay/page_01_summary.json`
- `demo/output/no_region_overlay/page_01_text.md`

## Ket qua

Sau khi chay, ket qua nam trong `demo/output/ingest/`:

- `page_01_blocks.json`: ket qua day du cho trang.
- `page_01_summary.json`: tom tat ngan.
- `page_01_text.md`: noi dung trich xuat.
- `page_01_overlay.png`: lop phu vung noi dung neu dung `--save-overlay`.
- `page_01_table_01.md`: sinh them neu trang co bang.

Moi so lieu trong terminal va JSON lay tu ket qua xu ly that; demo khong
hard-code so vung, so khoi hoac thoi gian.

## OCR

```powershell
.\.venv-gpu\Scripts\python.exe demo\run_ingest_demo.py `
  --pdf demo\input\sample_page.pdf `
  --page 1 `
  --ocr-mode auto `
  --save-overlay `
  --output demo\output\ingest
```

Gia tri `--ocr-mode` duoc anh xa sang bien moi truong cu
`BOXBIIBOO_ENABLE_REGION_IMAGE_OCR` trong pham vi tien trinh demo, sau do duoc
khoi phuc.

## So sanh bo trich xuat bang

Demo co the chay cung mot trang bang voi ba che do:

- `default`: bo trich xuat bang mac dinh dua vao lop van ban va quy tac hinh hoc.
- `tatr`: TATR-only, chi du doan cau truc hinh hoc; khong gan van ban vao o.
- `hybrid_tatr`: Hybrid TATR, dung TATR cho cau truc va word boxes cua PDF de gan van ban vao o.

### Default table extractor

```powershell
.\.venv-gpu\Scripts\python.exe demo\run_ingest_demo.py `
  --pdf demo\input\sample_table_page.pdf `
  --page 1 `
  --table-extractor default `
  --save-overlay `
  --output demo\output\table_default
```

Lenh ngan:

```powershell
demo\scripts\run_table_default.bat
```

### TATR-only

```powershell
.\.venv-gpu\Scripts\python.exe demo\run_ingest_demo.py `
  --pdf demo\input\sample_table_page.pdf `
  --page 1 `
  --table-extractor tatr `
  --save-overlay `
  --output demo\output\table_tatr
```

Lenh ngan:

```powershell
demo\scripts\run_table_tatr.bat
```

### Hybrid TATR

```powershell
.\.venv-gpu\Scripts\python.exe demo\run_ingest_demo.py `
  --pdf demo\input\sample_table_page.pdf `
  --page 1 `
  --table-extractor hybrid_tatr `
  --save-overlay `
  --output demo\output\table_hybrid_tatr
```

Lenh ngan:

```powershell
demo\scripts\run_table_hybrid_tatr.bat
```

Neu mo hinh TATR/Pillow/torch chua san sang, pipeline cu se tra ve canh bao
hoac fallback theo logic co san; demo khong tao ket qua gia. Voi `tatr`,
ket qua dung de minh hoa cau truc hinh hoc nen tep `page_01_table_01.md`
co the khong co noi dung o; thong tin hang, cot va o nam trong
`page_01_blocks.json`.

## Don output

```powershell
demo\scripts\clean_output.bat
```

Lenh don xoa noi dung sinh ra trong `demo/output/ingest/`,
`demo/output/no_region_overlay/` va cac thu muc so sanh bang
`demo/output/table_*`, giu lai `.gitkeep` neu co va khong dong vao input hay ma
nguon demo.

## Loi thuong gap

- PDF khong ton tai: kiem tra tham so `--pdf`.
- Trang ngoai pham vi: kiem tra `--page`.
- PDF co mat khau: demo khong tu giai ma.
- Khong tao duoc overlay: kiem tra Pillow va kha nang render trang PDF.
- Hybrid TATR cham hoac fallback: kiem tra model TATR va moi truong torch.

## Kich ban trinh bay mot phut

1. Noi: "Day la demo mot trang PDF, chi goi lai pipeline co san."
2. Chay `demo\scripts\run_ingest.bat`.
3. Chi ra buoc 1: probe cho biet kich thuoc, text layer va anh nhung.
4. Chi ra buoc 2: so vung van ban, anh, bang.
5. Chi ra buoc 3: tung vung duoc dinh tuyen sang text/OCR/table.
6. Mo `demo/output/ingest/page_01_overlay.png` de cho thay vung tren trang.
7. Mo `page_01_blocks.json` neu can chung minh metadata, bbox va route.
