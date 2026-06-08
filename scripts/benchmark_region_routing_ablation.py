from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ingest.pipeline import ingest_pdf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired full-ingest benchmark with region routing OFF and ON."
    )
    parser.add_argument("--pdf", action="append", default=[], help="PDF path. Can be repeated.")
    parser.add_argument("--glob", action="append", default=[], help="Glob pattern for PDFs. Can be repeated.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of PDFs after expansion; 0 means no limit.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory.")
    parser.add_argument("--text-preview", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdfs = _resolve_pdfs(args.pdf, args.glob, limit=args.limit)
    if not pdfs:
        raise SystemExit("No PDFs matched. Provide --pdf or --glob.")

    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, pdf in enumerate(pdfs, start=1):
        print(f"[{index}/{len(pdfs)}] {pdf}")
        rows.append(_run_pair(pdf, text_preview=args.text_preview))

    summary = _summarize(rows)
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (args.out / "per_doc.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.out / "README.md").write_text(_summary_markdown(summary, rows), encoding="utf-8")

    print(f"Wrote {args.out / 'summary.json'}")
    print(f"Wrote {args.out / 'per_doc.jsonl'}")
    print(f"Wrote {args.out / 'README.md'}")


def _resolve_pdfs(pdf_args: list[str], glob_args: list[str], *, limit: int) -> list[Path]:
    paths: list[Path] = []
    for value in pdf_args:
        path = Path(value)
        if path.exists() and path.suffix.lower() == ".pdf":
            paths.append(path)
    for pattern in glob_args:
        for value in glob.glob(pattern, recursive=True):
            path = Path(value)
            if path.is_file() and path.suffix.lower() == ".pdf":
                paths.append(path)

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    deduped.sort(key=lambda item: str(item))
    if limit and limit > 0:
        return deduped[:limit]
    return deduped


def _run_pair(pdf: Path, *, text_preview: int) -> dict[str, Any]:
    off = _run_one(pdf, region_enabled=False, text_preview=text_preview)
    on = _run_one(pdf, region_enabled=True, text_preview=text_preview)
    return {
        "pdf": str(pdf),
        "region_off": off,
        "region_on": on,
        "delta": {
            "success_changed": off["success"] != on["success"],
            "used_backend_changed": off.get("used_backend") != on.get("used_backend"),
            "block_count_delta": (on.get("block_count") or 0) - (off.get("block_count") or 0),
            "chunk_count_delta": (on.get("chunk_count") or 0) - (off.get("chunk_count") or 0),
            "table_block_delta": (on.get("table_block_count") or 0) - (off.get("table_block_count") or 0),
            "route_traced_block_delta": (on.get("route_traced_block_count") or 0)
            - (off.get("route_traced_block_count") or 0),
            "latency_sec_delta": (on.get("latency_sec") or 0.0) - (off.get("latency_sec") or 0.0),
        },
    }


def _run_one(pdf: Path, *, region_enabled: bool, text_preview: int) -> dict[str, Any]:
    previous = os.environ.get("BOXBIIBOO_ENABLE_REGION_ROUTING")
    os.environ["BOXBIIBOO_ENABLE_REGION_ROUTING"] = "1" if region_enabled else "0"
    started = time.perf_counter()
    try:
        result = ingest_pdf(pdf)
        latency = time.perf_counter() - started
        blocks = result["blocks"]
        chunks = result["chunks"]
        pages = result["pages"]
        route_counts = _count(
            (block.meta or {}).get("route_backend", "<none>")
            for block in blocks
        )
        region_types = _count(
            (block.meta or {}).get("region_type", "<none>")
            for block in blocks
            if (block.meta or {}).get("region_id")
        )
        return {
            "region_enabled": region_enabled,
            "success": True,
            "error": None,
            "latency_sec": latency,
            "used_backend": result.get("used_backend"),
            "fallback_error_count": len(result.get("errors") or []),
            "page_count": len(pages),
            "block_count": len(blocks),
            "chunk_count": len(chunks),
            "table_block_count": sum(1 for block in blocks if block.block_type == "table"),
            "ocr_block_count": sum(1 for block in blocks if block.source_mode == "ocr"),
            "route_traced_block_count": sum(1 for block in blocks if (block.meta or {}).get("route_backend")),
            "region_traced_block_count": sum(1 for block in blocks if (block.meta or {}).get("region_id")),
            "block_type_counts": _count(block.block_type for block in blocks),
            "chunk_block_type_counts": _count_many(chunk.block_types for chunk in chunks),
            "route_backend_counts": route_counts,
            "region_type_counts": region_types,
            "page_route_counts": _merge_counts((page.meta or {}).get("route_counts", {}) for page in pages),
            "sample_blocks": [_block_preview(block, text_preview) for block in blocks[:8]],
        }
    except Exception as exc:
        return {
            "region_enabled": region_enabled,
            "success": False,
            "error": str(exc),
            "latency_sec": time.perf_counter() - started,
        }
    finally:
        if previous is None:
            os.environ.pop("BOXBIIBOO_ENABLE_REGION_ROUTING", None)
        else:
            os.environ["BOXBIIBOO_ENABLE_REGION_ROUTING"] = previous


def _block_preview(block: Any, max_chars: int) -> dict[str, Any]:
    meta = dict(block.meta or {})
    return {
        "block_id": block.block_id,
        "page_index": block.page_index,
        "block_type": block.block_type,
        "source_mode": block.source_mode,
        "bbox": block.bbox,
        "text_preview": (block.text or "").replace("\n", " ")[:max_chars],
        "trace": {
            key: meta[key]
            for key in [
                "region_id",
                "region_type",
                "region_bbox",
                "route_backend",
                "route_reason",
                "fallback_used",
                "table_backend",
                "table_row_count",
                "table_col_count",
            ]
            if key in meta
        },
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    off_values = [row["region_off"] for row in rows]
    on_values = [row["region_on"] for row in rows]
    deltas = [row["delta"] for row in rows]
    return {
        "doc_count": len(rows),
        "region_off": _aggregate_config(off_values),
        "region_on": _aggregate_config(on_values),
        "delta": {
            "used_backend_changed_count": sum(1 for value in deltas if value["used_backend_changed"]),
            "success_changed_count": sum(1 for value in deltas if value["success_changed"]),
            "block_count_delta_mean": _mean(value["block_count_delta"] for value in deltas),
            "chunk_count_delta_mean": _mean(value["chunk_count_delta"] for value in deltas),
            "table_block_delta_mean": _mean(value["table_block_delta"] for value in deltas),
            "route_traced_block_delta_mean": _mean(value["route_traced_block_delta"] for value in deltas),
            "latency_sec_delta_mean": _mean(value["latency_sec_delta"] for value in deltas),
        },
    }


def _aggregate_config(values: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [value for value in values if value.get("success")]
    return {
        "success_rate": len(successes) / max(len(values), 1),
        "error_count": sum(1 for value in values if not value.get("success")),
        "used_backend_counts": _count(value.get("used_backend", "<error>") for value in values),
        "latency_sec_mean": _mean(value.get("latency_sec", 0.0) for value in values),
        "block_count_mean": _mean(value.get("block_count", 0) for value in successes),
        "chunk_count_mean": _mean(value.get("chunk_count", 0) for value in successes),
        "table_block_count_mean": _mean(value.get("table_block_count", 0) for value in successes),
        "ocr_block_count_mean": _mean(value.get("ocr_block_count", 0) for value in successes),
        "route_traced_block_count_mean": _mean(value.get("route_traced_block_count", 0) for value in successes),
        "region_traced_block_count_mean": _mean(value.get("region_traced_block_count", 0) for value in successes),
        "block_type_counts": _merge_counts(value.get("block_type_counts", {}) for value in successes),
        "route_backend_counts": _merge_counts(value.get("route_backend_counts", {}) for value in successes),
        "region_type_counts": _merge_counts(value.get("region_type_counts", {}) for value in successes),
    }


def _summary_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    off = summary["region_off"]
    on = summary["region_on"]
    delta = summary["delta"]
    lines = [
        "# Region Routing ON/OFF Full-Ingest Benchmark",
        "",
        f"- Documents: {summary['doc_count']}",
        "",
        "## Summary",
        "",
        "| Config | Success | Backend counts | Latency mean | Blocks mean | Chunks mean | Tables mean | Route-traced blocks mean |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        _config_row("Region OFF", off),
        _config_row("Region ON", on),
        "",
        "## Delta",
        "",
        f"- Backend changed in {delta['used_backend_changed_count']} / {summary['doc_count']} documents.",
        f"- Mean block delta: {delta['block_count_delta_mean']:.3f}.",
        f"- Mean chunk delta: {delta['chunk_count_delta_mean']:.3f}.",
        f"- Mean table block delta: {delta['table_block_delta_mean']:.3f}.",
        f"- Mean route-traced block delta: {delta['route_traced_block_delta_mean']:.3f}.",
        f"- Mean latency delta: {delta['latency_sec_delta_mean']:.3f}s.",
        "",
        "## Per Document",
        "",
        "| PDF | OFF backend | ON backend | OFF blocks | ON blocks | OFF tables | ON tables | OFF route trace | ON route trace |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        pdf = Path(row["pdf"]).name
        off_doc = row["region_off"]
        on_doc = row["region_on"]
        lines.append(
            f"| `{pdf}` | `{off_doc.get('used_backend')}` | `{on_doc.get('used_backend')}` | "
            f"{off_doc.get('block_count', 0)} | {on_doc.get('block_count', 0)} | "
            f"{off_doc.get('table_block_count', 0)} | {on_doc.get('table_block_count', 0)} | "
            f"{off_doc.get('route_traced_block_count', 0)} | {on_doc.get('route_traced_block_count', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Safe Interpretation",
            "",
            "- Region routing is an ingest coordination layer, not a retrieval algorithm by itself.",
            "- The key evidence is whether Region ON creates route-traced blocks and table/image/text-specific routing.",
            "- If Region OFF still extracts enough text, text-only retrieval may remain similar on simple PDFs.",
            "- Region is most valuable for mixed PDFs where text, table, image and OCR regions coexist.",
        ]
    )
    return "\n".join(lines) + "\n"


def _config_row(label: str, value: dict[str, Any]) -> str:
    return (
        f"| {label} | {value['success_rate']:.3f} | `{value['used_backend_counts']}` | "
        f"{value['latency_sec_mean']:.3f}s | {value['block_count_mean']:.2f} | "
        f"{value['chunk_count_mean']:.2f} | {value['table_block_count_mean']:.2f} | "
        f"{value['route_traced_block_count_mean']:.2f} |"
    )


def _count(values: Iterable[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _count_many(values: Iterable[Iterable[Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for nested in values:
        for value in nested:
            key = str(value)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _merge_counts(values: Iterable[dict[str, Any]]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for value in values:
        for key, count in dict(value or {}).items():
            merged[str(key)] = merged.get(str(key), 0) + int(count)
    return merged


def _mean(values: Iterable[float]) -> float:
    seq = [float(value) for value in values]
    if not seq:
        return 0.0
    return statistics.fmean(seq)


if __name__ == "__main__":
    main()

