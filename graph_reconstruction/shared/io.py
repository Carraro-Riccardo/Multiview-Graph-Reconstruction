"""Small JSON persistence helpers used by the active pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _json_default(value):
    """Convert common NumPy and path values to JSON-compatible objects."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Non-serializable type: {type(value)!r}")


def save_json(path: Path, payload) -> None:
    """Write an UTF-8, human-readable JSON file, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=_json_default)
