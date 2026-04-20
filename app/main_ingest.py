from __future__ import annotations

import json
import sys
from pathlib import Path

from app.ingest.pipeline import ingest_pdf


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python -m app.main_ingest <pdf_path> <out_dir>")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])

    report = ingest_pdf(pdf_path, out_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
