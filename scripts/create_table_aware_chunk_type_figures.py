from __future__ import annotations

import json
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS = REPO_ROOT / "results/retrieval_index/qcdt_2025_5445_constraint_table_reconstruction/corpus.jsonl"
OUT_DIR = REPO_ROOT / "docs/chapter5/figures/table_aware_chunk_types_qcdt_page9"

CHUNK_IDS = {
    "table_summary": "QCDT_2025_5445_QD-DHBK.pdf:chunk_00130",
    "table_structure": "QCDT_2025_5445_QD-DHBK.pdf:chunk_00131",
    "table_row": "QCDT_2025_5445_QD-DHBK.pdf:chunk_00132",
    "table_cell": "QCDT_2025_5445_QD-DHBK.pdf:chunk_00136",
}

TITLES = {
    "table_summary": "Table summary chunk",
    "table_structure": "Table structure chunk",
    "table_row": "Table row chunk",
    "table_cell": "Table cell chunk",
}

NOTES = {
    "table_summary": "Captures table-level context: caption, page, columns, and size.",
    "table_structure": "Preserves row/column layout using Markdown table format.",
    "table_row": "Represents one full table row as retrieval evidence.",
    "table_cell": "Represents one exact cell for fine-grained citation.",
}


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                Path("C:/Windows/Fonts/arialbd.ttf"),
                Path("C:/Windows/Fonts/calibrib.ttf"),
            ]
        )
    candidates.extend(
        [
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/calibri.ttf"),
            Path("C:/Windows/Fonts/consola.ttf"),
        ]
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _load_chunks() -> dict[str, dict]:
    wanted = set(CHUNK_IDS.values())
    chunks: dict[str, dict] = {}
    with CORPUS.open("r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            if obj.get("chunk_id") in wanted:
                chunks[obj["chunk_id"]] = obj
    missing = wanted - set(chunks)
    if missing:
        raise RuntimeError(f"Missing chunks: {sorted(missing)}")
    return chunks


def _wrap_text(text: str, width: int) -> str:
    wrapped_lines: list[str] = []
    for raw_line in text.splitlines():
        if not raw_line:
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(
            textwrap.wrap(
                raw_line,
                width=width,
                break_long_words=False,
                replace_whitespace=False,
            )
        )
    return "\n".join(wrapped_lines)


def _draw_chunk_image(kind: str, chunk: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    width = 1450
    padding = 46
    title_font = _load_font(34, bold=True)
    meta_font = _load_font(23)
    body_font = _load_font(25)
    mono_font = _load_font(23)

    text = chunk["text"]
    if kind == "table_structure":
        wrapped_body = text
        body_font_to_use = mono_font
    elif kind == "table_row":
        simplified = (
            "Bảng Điều 5. Điểm học phần, trang 9.\n"
            "Hàng \"Điểm quá trình được cộng/trừ\":\n"
            "- Cột 0: +1\n"
            "- Cột 1-2: 0\n"
            "- Cột 3-4: -1\n"
            "- Cột ≥ 5: -2"
        )
        wrapped_body = _wrap_text(simplified, 86)
        body_font_to_use = body_font
    else:
        wrapped_body = _wrap_text(text, 86)
        body_font_to_use = body_font

    lines = wrapped_body.splitlines()
    line_height = 36
    height = padding * 2 + 95 + 78 + max(3, len(lines)) * line_height + 44

    img = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(img)

    draw.rounded_rectangle(
        [18, 18, width - 18, height - 18],
        radius=18,
        outline="#d0d7de",
        width=3,
        fill="#fbfcfe",
    )
    draw.text((padding, padding), TITLES[kind], fill="#0f172a", font=title_font)

    meta = (
        f"chunking_strategy={chunk['metadata'].get('chunking_strategy')} | "
        f"citation_target={chunk['metadata'].get('citation_target')} | "
        f"page={chunk.get('page')} | table_id={chunk['metadata'].get('table_id')}"
    )
    draw.text((padding, padding + 52), meta, fill="#475569", font=meta_font)
    draw.text((padding, padding + 88), NOTES[kind], fill="#2563eb", font=meta_font)

    body_top = padding + 138
    draw.rounded_rectangle(
        [padding, body_top, width - padding, height - padding],
        radius=12,
        outline="#e2e8f0",
        width=2,
        fill="#ffffff",
    )
    y = body_top + 26
    for line in lines:
        draw.text((padding + 24, y), line, fill="#111827", font=body_font_to_use)
        y += line_height

    img.save(OUT_DIR / f"{kind}.png")


def _write_markdown(chunks_by_id: dict[str, dict]) -> None:
    lines = [
        "# Table-aware chunk type figures",
        "",
        "This folder contains four figures generated from real QCDT table-aware chunks.",
        "",
        "Source corpus:",
        "",
        "```text",
        "results/retrieval_index/qcdt_2025_5445_constraint_table_reconstruction/corpus.jsonl",
        "```",
        "",
        "## Reproduce Commands",
        "",
        "Run from repository root:",
        "",
        "```powershell",
        r".\.venv-gpu\Scripts\python.exe scripts\create_table_aware_chunk_type_figures.py",
        "```",
        "",
        "## Figures",
        "",
        "| Figure | Chunk ID | Strategy | Purpose |",
        "| --- | --- | --- | --- |",
    ]
    for kind, chunk_id in CHUNK_IDS.items():
        chunk = chunks_by_id[chunk_id]
        lines.append(
            f"| `{kind}.png` | `{chunk_id}` | `{chunk['metadata'].get('chunking_strategy')}` | {NOTES[kind]} |"
        )
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_latex() -> None:
    tex = r"""\begin{figure}[H]
\centering

\begin{subfigure}{0.48\textwidth}
\centering
\includegraphics[width=\linewidth]{figures/table_aware_chunk_types_qcdt_page9/table_summary.png}
\caption{Table summary}
\end{subfigure}
\hfill
\begin{subfigure}{0.48\textwidth}
\centering
\includegraphics[width=\linewidth]{figures/table_aware_chunk_types_qcdt_page9/table_structure.png}
\caption{Table structure}
\end{subfigure}

\vspace{0.25cm}

\begin{subfigure}{0.48\textwidth}
\centering
\includegraphics[width=\linewidth]{figures/table_aware_chunk_types_qcdt_page9/table_row.png}
\caption{Table row}
\end{subfigure}
\hfill
\begin{subfigure}{0.48\textwidth}
\centering
\includegraphics[width=\linewidth]{figures/table_aware_chunk_types_qcdt_page9/table_cell.png}
\caption{Table cell}
\end{subfigure}

\caption{Bốn dạng table-aware chunk sinh ra từ một bảng trong tài liệu QCDT}
\label{fig:table-aware-chunk-types}
\end{figure}
"""
    (OUT_DIR / "latex_figure_snippet.tex").write_text(tex, encoding="utf-8")


def main() -> None:
    chunks_by_id = _load_chunks()
    for kind, chunk_id in CHUNK_IDS.items():
        _draw_chunk_image(kind, chunks_by_id[chunk_id])
    _write_markdown(chunks_by_id)
    _write_latex()


if __name__ == "__main__":
    main()
