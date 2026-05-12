from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ID = "nielsr/funsd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a FUNSD OCR benchmark subset as BOXTALK JSONL manifest")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "benchmarks" / "funsd")
    parser.add_argument("--limit", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    filename = f"data/{args.split}-00000-of-00001.parquet"
    raw_dir = args.out / "raw"
    image_dir = args.out / "images" / args.split
    manifest_dir = args.out / "ocr"
    raw_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=raw_dir,
        )
    )
    df = pd.read_parquet(parquet_path)
    if args.limit > 0:
        df = df.head(args.limit)

    rows: list[dict] = []
    for _, row in df.iterrows():
        doc_id = f"funsd_{args.split}_{row['id']}"
        image_payload = row["image"]
        image_bytes = image_payload["bytes"]
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_path = image_dir / f"{doc_id}.png"
        image.save(image_path)

        words = [str(word).strip() for word in list(row["words"]) if str(word).strip()]
        text = " ".join(words)
        rows.append(
            {
                "doc_id": doc_id,
                "image_path": str(image_path.resolve()),
                "ground_truth": {
                    "text": text,
                    "ordered_text": words,
                },
                "metadata": {
                    "source": args.repo_id,
                    "split": args.split,
                    "sample_id": str(row["id"]),
                    "note": "FUNSD provides word-level annotation; benchmark reports OCR text metrics, not form-field extraction.",
                },
            }
        )

    manifest_path = manifest_dir / "ocr_samples.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "dataset": "funsd",
        "repo_id": args.repo_id,
        "split": args.split,
        "num_samples": len(rows),
        "manifest": str(manifest_path),
        "image_dir": str(image_dir),
        "parquet": str(parquet_path),
    }
    (args.out / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
