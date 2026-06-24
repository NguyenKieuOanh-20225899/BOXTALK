@echo off
setlocal

for /f "delims=" %%F in ('dir /b /a-d demo\output\ingest 2^>nul') do (
  if /I not "%%F"==".gitkeep" del /q "demo\output\ingest\%%F"
)
for /f "delims=" %%D in ('dir /b /ad demo\output\ingest 2^>nul') do (
  rmdir /s /q "demo\output\ingest\%%D"
)

if exist demo\output\no_region_overlay (
  rmdir /s /q demo\output\no_region_overlay
)
if exist demo\output\table_default (
  rmdir /s /q demo\output\table_default
)
if exist demo\output\table_tatr (
  rmdir /s /q demo\output\table_tatr
)
if exist demo\output\table_hybrid_tatr (
  rmdir /s /q demo\output\table_hybrid_tatr
)
