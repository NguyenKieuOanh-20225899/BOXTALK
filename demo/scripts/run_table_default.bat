@echo off
setlocal

python demo\run_ingest_demo.py ^
  --pdf demo\input\sample_table_page.pdf ^
  --page 1 ^
  --table-extractor default ^
  --save-overlay ^
  --output demo\output\table_default

