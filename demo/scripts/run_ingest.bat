@echo off
setlocal

python demo\run_ingest_demo.py ^
  --pdf demo\input\sample_page.pdf ^
  --page 1 ^
  --save-overlay ^
  --output demo\output\ingest

