from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SOURCES = {
    "DocLayNet": (
        Path("results/ingest/chapter5_doclaynet_full_rerun/summary.json"),
        ("layout_iou50", "micro_f1", "mean"),
        "F1@0.50",
    ),
    "PubLayNet": (
        Path("results/ingest/chapter5_publaynet_full_rerun/summary.json"),
        ("layout_iou50", "micro_f1", "mean"),
        "F1@0.50",
    ),
    "PubTables detection": (
        Path("results/ingest/chapter5_pubtables_detection_500_rerun_model/summary.json"),
        ("table_detection_iou50", "micro_f1", "mean"),
        "F1@0.50",
    ),
    "OCR scan": (
        Path("results/ingest/chapter5_ocr_scan_25_rerun_fixed_seq/summary.json"),
        ("ocr_token_f1", "mean"),
        "OCR token F1",
    ),
    "FUNSD OCR": (
        Path("results/ingest/chapter5_funsd_ocr_25_rerun_fixed_seq/summary.json"),
        ("ocr_token_f1", "mean"),
        "OCR token F1",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Chapter 5 ingest benchmark summary bar chart."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/chapter5/figures/ingest_benchmark_bar_chart.png"),
        help="Output image path. Use .png, .svg, or any matplotlib-supported suffix.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("docs/chapter5/figures/ingest_benchmark_bar_chart.csv"),
        help="Optional CSV companion path.",
    )
    parser.add_argument(
        "--title",
        default="Chapter 5 ingest benchmark summary",
        help="Chart title.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def nested_get(payload: dict[str, Any], keys: tuple[str, ...]) -> float:
    current: Any = payload.get("metric_summary", payload)
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(f"Missing metric path: metric_summary.{'.'.join(keys)}")
        current = current[key]
    return float(current)


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for benchmark, (path, metric_path, metric_name) in DEFAULT_SOURCES.items():
        payload = read_json(path)
        value = nested_get(payload, metric_path)
        rows.append(
            {
                "benchmark": benchmark,
                "metric": metric_name,
                "value": value,
                "source": str(path),
            }
        )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["benchmark,metric,value,source"]
    for row in rows:
        lines.append(
            f"{row['benchmark']},{row['metric']},{row['value']:.6f},{row['source']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot(rows: list[dict[str, Any]], out_path: Path, *, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        svg_path = out_path.with_suffix(".svg")
        write_svg(rows, svg_path, title=title)
        if out_path.suffix.lower() != ".svg":
            print(
                "matplotlib is not installed; wrote SVG fallback instead of "
                f"{out_path.name}: {svg_path}"
            )
        return

    labels = [str(row["benchmark"]) for row in rows]
    values = [float(row["value"]) for row in rows]
    metric_labels = [str(row["metric"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    colors = ["#3f6fb5", "#5a8f3d", "#b56a3f", "#7a5ab5", "#3f9f9b"]
    bars = ax.bar(labels, values, color=colors[: len(labels)], width=0.62)

    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, value, metric in zip(bars, values, metric_labels, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.025, 1.03),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            0.035,
            metric,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#333333",
            rotation=90,
        )

    ax.tick_params(axis="x", labelrotation=18)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_svg(rows: list[dict[str, Any]], out_path: Path, *, title: str) -> None:
    width = 980
    height = 560
    margin_left = 80
    margin_right = 40
    margin_top = 70
    margin_bottom = 115
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom
    bar_gap = 34
    bar_width = (chart_width - bar_gap * (len(rows) - 1)) / len(rows)
    colors = ["#3f6fb5", "#5a8f3d", "#b56a3f", "#7a5ab5", "#3f9f9b"]

    def esc(value: object) -> str:
        return (
            str(value)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-family="Arial, sans-serif" font-size="22" font-weight="700">{esc(title)}</text>',
        f'<line x1="{margin_left}" y1="{margin_top + chart_height}" x2="{width - margin_right}" y2="{margin_top + chart_height}" stroke="#333" stroke-width="1"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + chart_height}" stroke="#333" stroke-width="1"/>',
    ]

    for tick in range(0, 6):
        value = tick / 5
        y = margin_top + chart_height - value * chart_height
        parts.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#d9d9d9" stroke-dasharray="4 4" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="12" fill="#333">{value:.1f}</text>'
        )

    parts.append(
        f'<text x="24" y="{margin_top + chart_height / 2}" text-anchor="middle" transform="rotate(-90 24 {margin_top + chart_height / 2})" font-family="Arial, sans-serif" font-size="14" fill="#333">Score</text>'
    )

    for index, row in enumerate(rows):
        value = max(0.0, min(float(row["value"]), 1.0))
        x = margin_left + index * (bar_width + bar_gap)
        bar_height = value * chart_height
        y = margin_top + chart_height - bar_height
        center = x + bar_width / 2
        color = colors[index % len(colors)]
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{center:.1f}" y="{y - 10:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="15" font-weight="700" fill="#222">{value:.3f}</text>'
        )
        parts.append(
            f'<text x="{center:.1f}" y="{margin_top + chart_height + 28}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#222">{esc(row["benchmark"])}</text>'
        )
        parts.append(
            f'<text x="{center:.1f}" y="{margin_top + chart_height + 50}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#555">{esc(row["metric"])}</text>'
        )

    parts.append("</svg>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_rows()
    write_csv(rows, args.csv_out)
    plot(rows, args.out, title=args.title)
    print(args.out)
    print(args.csv_out)


if __name__ == "__main__":
    main()
