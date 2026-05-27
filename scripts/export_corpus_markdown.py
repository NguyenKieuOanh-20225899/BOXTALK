from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a retrieval corpus.jsonl file to a readable Markdown preview."
    )
    parser.add_argument(
        "--index-dir",
        required=True,
        help="Retrieval index directory containing corpus.jsonl.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output Markdown path. Defaults to <index-dir>/corpus_preview.md.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=2000,
        help="Maximum text characters to include per chunk. Use 0 for full text.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of chunks to export. Use 0 for all chunks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_dir = Path(args.index_dir)
    corpus_path = index_dir / "corpus.jsonl"
    if not corpus_path.exists():
        raise SystemExit(f"Missing corpus file: {corpus_path}")

    output_path = Path(args.output) if args.output else index_dir / "corpus_preview.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exported = 0
    with corpus_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as out:
        out.write(f"# Corpus Preview\n\n")
        out.write(f"- Source: `{corpus_path.as_posix()}`\n")
        out.write(f"- Max chars per chunk: `{args.max_chars}`\n\n")

        for line_number, line in enumerate(src, start=1):
            if args.limit and exported >= args.limit:
                break
            line = line.strip()
            if not line:
                continue

            try:
                row: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as exc:
                out.write(f"\n\n## Invalid JSONL line {line_number}\n\n")
                out.write(f"```text\n{exc}\n```\n")
                continue

            chunk_id = row.get("chunk_id") or row.get("id") or f"line_{line_number}"
            page_indices = row.get("page_indices") or row.get("pages") or []
            if isinstance(page_indices, list):
                pages = [page + 1 if isinstance(page, int) else page for page in page_indices]
            else:
                pages = page_indices
            heading_path = row.get("heading_path") or []
            block_types = row.get("block_types") or row.get("block_type") or []
            text = str(row.get("text") or "")
            shown_text = text if args.max_chars <= 0 else text[: args.max_chars]
            truncated = args.max_chars > 0 and len(text) > args.max_chars

            out.write(f"\n\n## Chunk {exported + 1}: `{chunk_id}`\n\n")
            out.write(f"- Pages (1-based): `{pages}`\n")
            out.write(f"- Heading path: `{heading_path}`\n")
            out.write(f"- Block types: `{block_types}`\n")
            out.write(f"- Text length: `{len(text)}`\n\n")
            out.write("```text\n")
            out.write(shown_text)
            if truncated:
                out.write("\n... [truncated]")
            out.write("\n```\n")
            exported += 1

        out.write(f"\n\n---\n\nExported chunks: `{exported}`\n")

    print(output_path)


if __name__ == "__main__":
    main()
