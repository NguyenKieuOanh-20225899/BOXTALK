# Table-aware chunk type figures

This folder contains four figures generated from real QCDT table-aware chunks.

Source corpus:

```text
results/retrieval_index/qcdt_2025_5445_constraint_table_reconstruction/corpus.jsonl
```

## Reproduce Commands

Run from repository root:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\create_table_aware_chunk_type_figures.py
```

## Figures

| Figure | Chunk ID | Strategy | Purpose |
| --- | --- | --- | --- |
| `table_summary.png` | `QCDT_2025_5445_QD-DHBK.pdf:chunk_00130` | `table_summary` | Captures table-level context: caption, page, columns, and size. |
| `table_structure.png` | `QCDT_2025_5445_QD-DHBK.pdf:chunk_00131` | `table_structure` | Preserves row/column layout using Markdown table format. |
| `table_row.png` | `QCDT_2025_5445_QD-DHBK.pdf:chunk_00132` | `table_row` | Represents one full table row as retrieval evidence. |
| `table_cell.png` | `QCDT_2025_5445_QD-DHBK.pdf:chunk_00136` | `table_cell` | Represents one exact cell for fine-grained citation. |
