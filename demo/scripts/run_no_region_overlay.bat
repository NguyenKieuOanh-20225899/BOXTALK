@echo off
setlocal

python demo\run_ingest_demo.py ^
  --pdf demo\input\sample_page.pdf ^
  --page 1 ^
  --region-routing off ^
  --save-overlay ^
  --output demo\output\no_region_overlay
