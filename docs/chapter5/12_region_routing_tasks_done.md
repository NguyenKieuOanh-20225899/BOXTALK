# Region-level Routing Tasks Done

Branch: `region-routing-defense-docs`

## Da lam

- Tao nhanh rieng tu `main`, khong merge vao `main`.
- Mo rong region dict trong `app/ingest/region/detector.py`:
  - them `type`;
  - them `confidence`;
  - giu `kind`, `bbox`, `route_backend`, `detection_source`.
- Mo rong block metadata trong `app/ingest/extract/region_routed.py`:
  - `region_type`;
  - `region_bbox`;
  - `page_number`;
  - `confidence`;
  - `source`;
  - `fallback_used`.
- Them utility ve overlay:
  - `app/ingest/region/debug.py`;
  - `scripts/draw_region_overlay.py`.
- Them script so sanh region ON/OFF:
  - `scripts/compare_region_routing.py`.
- Them test cho:
  - metadata trace cua table block;
  - tao anh overlay region debug.
- Tao report phuc vu bao ve/phan bien:
  - `docs/chapter5/11_region_routing_defense_report.md`.
- Tao file checklist task:
  - `docs/chapter5/12_region_routing_tasks_done.md`.

## Cach tao anh overlay

```powershell
python scripts/draw_region_overlay.py path/to/file.pdf --page 1 --out docs/chapter5/figures/region_overlay_page1.png
```

## Cach so sanh co/khong co region

```powershell
python scripts/compare_region_routing.py path/to/file.pdf --page 1 --out-dir docs/chapter5/region_compare
```

Script se chay ingest hai lan tren cung PDF:

- `BOXBIIBOO_ENABLE_REGION_ROUTING=0`;
- `BOXBIIBOO_ENABLE_REGION_ROUTING=1`.

Output gom:

- file JSON de inspect chi tiet;
- file Markdown de dua vao docs/bao cao.

Can xem:

- `used_backend`;
- `block_type_counts`;
- `route_backend_counts`;
- `trace_meta` cua tung block;
- block/chunk count tren trang can so sanh.

## Validation can chay

```powershell
python -m compileall app scripts
python -m pytest tests/test_region_level_routing.py -q
```

## Ghi chu khi dua vao do an

- Claim chinh nen la trace/debug/explain va dieu phoi backend theo tung region.
- Khong claim region routing luon tot hon neu chua co paired benchmark full ingest
  tren PDF thuc te hon hop.
- Nen dua anh overlay vao Chuong 5 hoac slide bao ve.
