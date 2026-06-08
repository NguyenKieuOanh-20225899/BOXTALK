# Region-level Routing Defense Report

## 1. Dinh nghia region

Trong pipeline ingest, `region` la don vi xu ly nho hon `page`. Thay vi coi
mot trang PDF la mot khoi duy nhat, he thong tach trang thanh cac vung noi dung
doc lap. Moi region co:

- `bbox`: toa do hinh hoc tren trang;
- `type`/`kind`: loai noi dung, vi du `paragraph`, `heading`, `table`, `image`;
- `route_backend`: backend du kien xu ly vung do;
- `detection_source`: nguon phat hien vung;
- `confidence`: diem tin cay neu detector co cung cap.

Vi du:

```json
{
  "type": "table",
  "kind": "table",
  "bbox": [72, 210, 520, 420],
  "page_index": 2,
  "route_backend": "table",
  "detection_source": "native_or_text_cluster",
  "confidence": 1.0
}
```

Diem can nhan manh: region khong chi la text block. Region la vung ngu nghia
co vai tro dieu phoi backend.

## 2. Co che hoat dong trong code

Code chinh:

- `app/ingest/extract/region_routed.py`
- `app/ingest/region/detector.py`
- `app/ingest/region/debug.py`

Luon xu ly:

```text
PDF page
-> detect_regions(page)
-> text/table/image/caption/header/footer regions
-> route tung region sang backend phu hop
-> BlockNode kem metadata trace
```

Routing hien tai:

| Region kind | Backend |
| --- | --- |
| `table` | `extract_table_region` |
| `paragraph`, `heading`, `list_item`, `caption`, `metadata` | `extract_text_region` |
| `image` | OCR neu can, neu khong thi figure placeholder |

## 3. Metadata trace/debug

Moi block sinh ra tu region-routed backend duoc gan metadata de truy vet:

```json
{
  "backend": "region_routed",
  "region_id": "p0000_table_0000",
  "region_type": "table",
  "region_kind": "table",
  "region_bbox": [72.0, 210.0, 520.0, 420.0],
  "page_number": 3,
  "route_backend": "table",
  "confidence": 1.0,
  "source": "native_or_text_cluster",
  "detection_source": "native_or_text_cluster",
  "fallback_used": false
}
```

Khi phan bien hoi "lam sao biet backend nao xu ly phan nao?", co the tra loi:

> Moi block dau ra deu mang metadata cho biet block den tu region nao, bbox nao,
> loai noi dung gi, nam o trang nao va duoc xu ly boi backend nao.

Day la bang chung ve kha nang trace, debug va explain cua pipeline.

## 4. Anh minh hoa region overlay

Da bo sung utility ve overlay:

```text
app/ingest/region/debug.py
scripts/draw_region_overlay.py
```

Lenh tao anh minh hoa:

```powershell
python scripts/draw_region_overlay.py path/to/file.pdf --page 1 --out docs/chapter5/figures/region_overlay_page1.png
```

Mau vung:

| Region | Mau |
| --- | --- |
| `heading` | xanh duong |
| `paragraph`/`text` | xam |
| `list_item` | xanh la |
| `table` | do |
| `image` | cam |
| `caption` | tim |
| `metadata` | nau |
| `header`/`footer` | xanh cyan |

Anh overlay nen duoc dua vao Chuong 5 hoac slide bao ve de bien region routing
tu mot y tuong code thanh bang chung truc quan.

## 5. So sanh voi text/layout/OCR

| Phuong phap | Don vi xu ly | Diem manh | Han che |
| --- | --- | --- | --- |
| Text only | text block/page | Nhanh, tot voi PDF co text layer | Mat cau truc bang, khong xu ly scan |
| OCR toan trang | page/image | Doc duoc scan | Cham, de loi OCR, reading order kho |
| Layout only | page | Giu bo cuc tot hon | Khong toi uu rieng cho bang/OCR |
| Region routing | region trong page | Dung backend phu hop cho tung vung | Phu thuoc chat luong detect region |

Thong diep chinh:

> Region routing khong thay the text/OCR/layout. No la lop dieu phoi giup dung
> dung cong cu cho dung loai vung.

## 6. Bang chung benchmark hien co

Da chay ablation bat/tat region tai:

```text
docs/chapter5/10_region_routing_ablation.md
```

Ket qua an toan hien tai:

- Mock full ingest: region ON dung `region_routed`, khong gay regression;
- PubTables/FUNSD trong benchmark suite la task-specific path, khong phai bang
  chung truc tiep ve full region pipeline;
- can them paired benchmark tren PDF thuc te hon hop neu muon claim chat luong
  manh hon.

## 7. Cach trinh bay an toan trong do an

Nen viet:

> Region-level routing duoc thiet ke nhu mot lop dieu phoi trong ingest pipeline.
> He thong tach moi trang PDF thanh cac vung noi dung co bbox va loai noi dung,
> sau do dinh tuyen tung vung sang backend phu hop. Cach tiep can nay giup xu ly
> PDF hon hop linh hoat hon so voi viec ap dung mot backend duy nhat cho ca
> trang. Dong thoi, moi block dau ra giu lai metadata region de phuc vu trace,
> debug va citation.

Khong nen viet:

> Region routing luon tot hon tat ca baseline.

