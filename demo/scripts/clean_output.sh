#!/usr/bin/env bash
set -e

find demo/output/ingest \
  -mindepth 1 \
  ! -name ".gitkeep" \
  -delete

rm -rf demo/output/no_region_overlay
rm -rf demo/output/table_default
rm -rf demo/output/table_tatr
rm -rf demo/output/table_hybrid_tatr
