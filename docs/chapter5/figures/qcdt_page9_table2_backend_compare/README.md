# QCDT Page 9 Table 2 Backend Visual Comparison

This directory contains Markdown and PNG renderings for comparing three table extraction backends on the same QCDT page 9 table.

## Files

| File | Meaning |
|---|---|
| `table_original.png` | Original cropped table image from `QCDT_page9_tatr_only/crops/page9_table2.png`. |
| `table_default_output.md` | Markdown table output from the default/region-routed table extractor. |
| `table_default_output.png` | Rendered image of the default Markdown output. |
| `table_tatr_output.md` | Markdown representation of TATR-only geometry output. Cells are empty because no text boxes are supplied. |
| `table_tatr_output.png` | Rendered image of the TATR-only Markdown output. |
| `table_hybrid_tatr_output.md` | Markdown table output from Hybrid TATR with PDF text word boxes. |
| `table_hybrid_tatr_output.png` | Rendered image of the Hybrid TATR Markdown output. |
| `latex_figure_snippet.tex` | LaTeX figure snippet using the copied images under `docs/chapter5/figures/`. |

## Source

- Default source: `docs\chapter5\ingest_visualizations\QCDT_page9_region_routed\blocks.jsonl`
- TATR-only source: `docs\chapter5\ingest_visualizations\QCDT_page9_tatr_only\preview.md`
- Hybrid TATR source: `docs\chapter5\ingest_visualizations\QCDT_page9_text_layer_hybrid_tatr_demo\06_table2_hybrid_tatr_output.md`
