"""
src/utils.py — Shared utility helpers used across every module.
"""

import logging
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently-formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ─────────────────────────────────────────────
# JSON helpers
# ─────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    """Make NumPy scalars JSON-serialisable."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(data: Dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, cls=_NumpyEncoder)


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ─────────────────────────────────────────────
# Timing helper
# ─────────────────────────────────────────────

class Timer:
    """Context-manager stopwatch."""
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start

    def __str__(self):
        return f"{self.elapsed:.3f}s"


# ─────────────────────────────────────────────
# Filesystem helpers
# ─────────────────────────────────────────────

def find_audio_files(root: Path, extensions: set) -> list:
    """Recursively collect audio files under *root* with given extensions."""
    root = Path(root)
    if not root.exists():
        return []
    return [
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]


def safe_makedirs(path: Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)
