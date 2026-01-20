from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python + NumPy (and optionally Torch) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def unique_rows(X: np.ndarray) -> np.ndarray:
    """Return unique rows of a 2D numpy array (preserving order)."""
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    seen = set()
    out = []
    for row in X:
        key = row.tobytes()
        if key not in seen:
            seen.add(key)
            out.append(row)
    return np.array(out, dtype=X.dtype)


def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum(a != b))


def topk_unique(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pick top-k (smallest y), ensure unique rows."""
    if len(X) == 0:
        return X, y
    order = np.argsort(y)
    Xs = X[order]
    ys = y[order]
    picked = []
    picked_y = []
    seen = set()
    for xi, yi in zip(Xs, ys):
        key = xi.tobytes()
        if key in seen:
            continue
        seen.add(key)
        picked.append(xi)
        picked_y.append(float(yi))
        if len(picked) >= k:
            break
    return np.array(picked, dtype=X.dtype), np.array(picked_y, dtype=float)
