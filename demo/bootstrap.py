from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = Path(__file__).resolve().parent


def ensure_repo_on_path() -> Path:
    """Expose the existing repository packages to demo scripts.

    The repository is not guaranteed to be installed with ``pip install -e .``
    on the defense machine. This is the only demo file that mutates
    ``sys.path`` so the rest of the demo can import existing modules normally.
    """

    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return REPO_ROOT

