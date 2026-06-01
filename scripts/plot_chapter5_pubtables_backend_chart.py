from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "chapter5" / "figures"
PNG_PATH = OUT_DIR / "pubtables_backend_comparison.png"
SVG_PATH = OUT_DIR / "pubtables_backend_comparison.svg"
CSV_PATH = OUT_DIR / "pubtables_backend_comparison.csv"


BACKENDS = ["Default", "TATR", "Hybrid TATR"]
METRICS = [
    ("Detection F1", [0.940, 0.987, 0.987]),
    ("Cell F1@0.75", [0.149, 0.103, 0.944]),
    ("Structure F1", [0.199, 0.010, 0.772]),
    ("Text Assign. F1", [0.909, 0.015, 0.999]),
    ("Exact CSV", [0.000, 0.000, 0.480]),
]


def write_csv() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Backend", *[name for name, _ in METRICS]])
        for idx, backend in enumerate(BACKENDS):
            writer.writerow([backend, *[values[idx] for _, values in METRICS]])


def plot_with_matplotlib() -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(BACKENDS))
    width = 0.15
    offsets = (np.arange(len(METRICS)) - (len(METRICS) - 1) / 2) * width

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 9,
        }
    )

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]

    for offset, (metric, values), color in zip(offsets, METRICS, colors):
        bars = ax.bar(x + offset, values, width, label=metric, color=color)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.025, 1.03),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90 if value > 0 else 0,
            )

    ax.set_ylabel("Score")
    ax.set_xlabel("Backend")
    ax.set_xticks(x)
    ax.set_xticklabels(BACKENDS)
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.45)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.tight_layout()

    fig.savefig(PNG_PATH, dpi=220, bbox_inches="tight")
    fig.savefig(SVG_PATH, bbox_inches="tight")


def main() -> None:
    write_csv()
    plot_with_matplotlib()
    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {SVG_PATH}")
    print(f"Wrote {CSV_PATH}")


if __name__ == "__main__":
    main()
