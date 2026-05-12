from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "deep-learning-analytics/arxiv_small_nougat"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a small Nougat/arXiv academic PDF benchmark manifest")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "benchmarks" / "nougat_arxiv_small")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--timeout", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = args.out / "raw"
    pdf_dir = args.out / "pdfs"
    manifest_dir = args.out / "text"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    csv_path = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename="arxiv_small.csv",
            repo_type="dataset",
            local_dir=raw_dir,
        )
    )
    df = pd.read_csv(csv_path)

    rows: list[dict] = []
    failures: list[dict] = []
    for _, row in df.iterrows():
        if args.limit > 0 and len(rows) >= args.limit:
            break
        arxiv_id = str(row.get("id") or "").strip()
        if not arxiv_id:
            continue
        pdf_path = pdf_dir / f"{_safe_id(arxiv_id)}.pdf"
        url = _pdf_url(arxiv_id, str(row.get("source") or ""))
        try:
            _download_pdf(url, pdf_path, timeout=args.timeout)
        except Exception as exc:
            failures.append({"id": arxiv_id, "url": url, "error": str(exc)})
            continue

        gt_text = _first_text(row, ["noref_content", "content"])
        if not gt_text:
            failures.append({"id": arxiv_id, "url": url, "error": "missing ground-truth content"})
            continue

        rows.append(
            {
                "doc_id": f"nougat_arxiv_{_safe_id(arxiv_id)}",
                "pdf_path": str(pdf_path.resolve()),
                "ground_truth": {
                    "text": gt_text,
                    "ordered_text": _ordered_lines(gt_text),
                },
                "metadata": {
                    "source": args.repo_id,
                    "arxiv_id": arxiv_id,
                    "pdf_url": url,
                    "title": str(row.get("title") or ""),
                    "note": (
                        "Nougat/arXiv proxy: ground truth is markdown-like academic content. "
                        "This repo evaluates text extraction against it, not full Nougat markup generation."
                    ),
                },
            }
        )

    manifest_path = manifest_dir / "nougat_samples.jsonl"
    compat_manifest_path = manifest_dir / "bastkorzen_samples.jsonl"
    for output_path in (manifest_path, compat_manifest_path):
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "dataset": "nougat_arxiv_small",
        "repo_id": args.repo_id,
        "num_samples": len(rows),
        "manifest": str(manifest_path),
        "compat_manifest": str(compat_manifest_path),
        "pdf_dir": str(pdf_dir),
        "failures": failures[:20],
        "failure_count": len(failures),
        "note": "Use with --dataset nougat --mode text. It is a Nougat-style academic text/markup proxy.",
    }
    (args.out / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _download_pdf(url: str, output_path: Path, *, timeout: int) -> None:
    if output_path.exists() and output_path.stat().st_size > 1024:
        return
    request = urllib.request.Request(url, headers={"User-Agent": "BOXTALK-ingest-benchmark/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        payload = response.read()
    if len(payload) < 1024:
        raise RuntimeError("downloaded file is too small")
    if "pdf" not in content_type.lower() and not payload.startswith(b"%PDF"):
        raise RuntimeError(f"download did not look like a PDF: content_type={content_type}")
    output_path.write_bytes(payload)


def _pdf_url(arxiv_id: str, source: str) -> str:
    if source and "arxiv.org/pdf/" in source:
        url = source.replace("http://", "https://")
        return url if url.endswith(".pdf") else f"{url}.pdf"
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def _ordered_lines(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        return lines
    return [text.strip()] if text.strip() else []


def _first_text(row, names: list[str]) -> str:
    for name in names:
        value = row.get(name)
        if value is None or pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return ""


def _safe_id(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9.]+", "_", value).strip("_").lower() or "sample"


if __name__ == "__main__":
    main()
