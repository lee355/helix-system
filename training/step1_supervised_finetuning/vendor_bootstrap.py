"""Prefer the repository-local Helix dependency sources over site-packages."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VENDOR_PATHS = (
    PROJECT_ROOT / "third_party" / "transformers" / "src",
    PROJECT_ROOT / "third_party" / "deepspeed",
    PROJECT_ROOT / "third_party" / "thop",
)


def activate_local_dependencies() -> None:
    missing = [str(path) for path in VENDOR_PATHS if not path.is_dir()]
    if missing:
        raise RuntimeError("missing repository-local dependencies: " + ", ".join(missing))
    for path in reversed(VENDOR_PATHS):
        value = str(path)
        if value in sys.path:
            sys.path.remove(value)
        sys.path.insert(0, value)
