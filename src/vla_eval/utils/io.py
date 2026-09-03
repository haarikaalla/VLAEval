"""File and artifact I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it doesn't already exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, data: Any) -> None:
    """Write `data` as pretty-printed JSON to `path`, creating parent dirs."""
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    """Read and parse JSON from `path`."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
