from __future__ import annotations

from collections import Counter
from pathlib import Path

from app.ingest.pipeline import ingest_pdf


ROOT = Path("data/test_probe")


def main():
    labels = ["text", "layout", "ocr", "mixed"]

    for label in labels:
        folder = ROOT / label
        if not folder.exists():
            print(f"Missing folder: {folder}")
            continue

        print(f"\n===== {label.upper()} =====")

        for pdf_path in sorted(folder.glob("*.pdf")):
            try:
                out = ingest_pdf(pdf_path)

                probe = out.get("probe", {})
                blocks = out.get("blocks", [])
                chunks = out.get("chunks", [])

                pred = probe.get("probe_detected_mode") or "NA"
                backend = out.get("used_backend") or "NA"
                bbox_blocks = sum(1 for b in blocks if b.bbox is not None)
                bbox_ratio = (bbox_blocks / len(blocks)) if blocks else 0.0
                route_backends = Counter(
                    (b.meta or {}).get("route_backend", b.source_mode) for b in blocks
                )

                block_types = Counter(b.block_type for b in blocks)
                types_str = ", ".join(f"{k}:{v}" for k, v in block_types.items())
                routes_str = ", ".join(f"{k}:{v}" for k, v in route_backends.items())

                print(
                    f"[OK] gold={label:<6} "
                    f"pred={pred:<6} "
                    f"backend={backend:<6} "
                    f"pages={len(out.get('pages', [])):<3} "
                    f"blocks={len(blocks):<4} "
                    f"bbox={bbox_ratio:.2f} "
                    f"chunks={len(chunks):<4} "
                    f"routes=[{routes_str}] "
                    f"types=[{types_str}] "
                    f"path={pdf_path.name}"
                )

            except Exception as e:
                print(
                    f"[ERR] gold={label:<6} "
                    f"path={pdf_path.name} "
                    f"error={str(e)[:120]}"
                )


if __name__ == "__main__":
    main()
