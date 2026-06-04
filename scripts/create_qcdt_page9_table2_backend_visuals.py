from __future__ import annotations

import json
import shutil
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "chapter5" / "figures" / "qcdt_page9_table2_backend_compare"
FIG_DIR = ROOT / "docs" / "chapter5" / "figures"

ORIGINAL_CROP = (
    ROOT
    / "docs"
    / "chapter5"
    / "ingest_visualizations"
    / "QCDT_page9_tatr_only"
    / "crops"
    / "page9_table2.png"
)
REGION_BLOCKS = (
    ROOT
    / "docs"
    / "chapter5"
    / "ingest_visualizations"
    / "QCDT_page9_region_routed"
    / "blocks.jsonl"
)
HYBRID_MD = (
    ROOT
    / "docs"
    / "chapter5"
    / "ingest_visualizations"
    / "QCDT_page9_text_layer_hybrid_tatr_demo"
    / "06_table2_hybrid_tatr_output.md"
)


def _load_default_table_markdown() -> str:
    for line in REGION_BLOCKS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        block = json.loads(line)
        if block.get("page_index") != 8 or block.get("block_type") != "table":
            continue
        bbox = block.get("bbox") or []
        # Page 9 table 2 is the lower grade-conversion table.
        if len(bbox) == 4 and bbox[1] > 600:
            meta = block.get("meta") or {}
            return meta.get("table_markdown") or block.get("markdown") or block.get("text") or ""
    raise RuntimeError("Could not find QCDT page 9 table 2 in region-routed blocks")


def _markdown_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    header = rows[0]
    sep = ["---"] * len(header)
    body = rows[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _fix_mojibake(text: str) -> str:
    try:
        fixed = text.encode("latin1").decode("utf-8")
    except UnicodeError:
        return text
    # Keep the fixed text only when it clearly repairs Vietnamese mojibake.
    if "Ä" in text or "Ã" in text or "á" in text:
        return fixed
    return text


def _render_text_png(title: str, markdown: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 24)
        title_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 30)
    except OSError:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    wrapped_lines: list[str] = []
    for raw_line in markdown.splitlines():
        if len(raw_line) <= 110:
            wrapped_lines.append(raw_line)
        else:
            wrapped_lines.extend(textwrap.wrap(raw_line, width=110, break_long_words=False))

    padding = 28
    line_height = 34
    title_height = 44
    width = 1800
    height = padding * 2 + title_height + max(1, len(wrapped_lines)) * line_height

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((padding, padding), title, fill=(25, 25, 25), font=title_font)
    y = padding + title_height
    for line in wrapped_lines:
        draw.text((padding, y), line, fill=(35, 35, 35), font=font)
        y += line_height
    img.save(output_path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    original_out = OUT_DIR / "table_original.png"
    shutil.copyfile(ORIGINAL_CROP, original_out)
    shutil.copyfile(original_out, FIG_DIR / "table_original.png")

    default_md = _fix_mojibake(_load_default_table_markdown())
    tatr_md = _markdown_table(
        [
            ["", "", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", "", ""],
            ["", "", "", "", "", "", "", "", "", ""],
        ]
    )
    hybrid_md = _fix_mojibake(HYBRID_MD.read_text(encoding="utf-8").strip())

    outputs = {
        "table_default_output": ("Default extractor", default_md),
        "table_tatr_output": ("TATR-only", tatr_md),
        "table_hybrid_tatr_output": ("Hybrid TATR", hybrid_md),
    }

    for stem, (title, markdown) in outputs.items():
        md_path = OUT_DIR / f"{stem}.md"
        md_path.write_text(markdown + "\n", encoding="utf-8")
        png_path = OUT_DIR / f"{stem}.png"
        _render_text_png(title, markdown, png_path)
        shutil.copyfile(png_path, FIG_DIR / f"{stem}.png")

    latex = r"""\begin{figure}[H]
\centering

\begin{subfigure}{0.48\textwidth}
\centering
\includegraphics[width=\linewidth]{figures/table_original.png}
\caption{Bảng gốc}
\end{subfigure}
\hfill
\begin{subfigure}{0.48\textwidth}
\centering
\includegraphics[width=\linewidth]{figures/table_default_output.png}
\caption{Default extractor}
\end{subfigure}

\vspace{0.25cm}

\begin{subfigure}{0.48\textwidth}
\centering
\includegraphics[width=\linewidth]{figures/table_tatr_output.png}
\caption{TATR-only}
\end{subfigure}
\hfill
\begin{subfigure}{0.48\textwidth}
\centering
\includegraphics[width=\linewidth]{figures/table_hybrid_tatr_output.png}
\caption{Hybrid TATR}
\end{subfigure}

\caption{Minh họa đầu ra của các backend trích xuất bảng trên cùng một bảng QCDT trang 9}
\label{fig:qcdt-page9-table-backend-visual-output}
\end{figure}
"""
    (OUT_DIR / "latex_figure_snippet.tex").write_text(latex, encoding="utf-8")

    readme = f"""# QCDT Page 9 Table 2 Backend Visual Comparison

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

- Default source: `{REGION_BLOCKS.relative_to(ROOT)}`
- TATR-only source: `{(ROOT / "docs/chapter5/ingest_visualizations/QCDT_page9_tatr_only/preview.md").relative_to(ROOT)}`
- Hybrid TATR source: `{HYBRID_MD.relative_to(ROOT)}`
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    print(f"Wrote {OUT_DIR.relative_to(ROOT)}")
    print("Copied figure aliases into docs/chapter5/figures/")


if __name__ == "__main__":
    main()
