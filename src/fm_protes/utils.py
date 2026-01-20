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


def _rankdata_average_ties(x: np.ndarray) -> np.ndarray:
    """Ranks with average rank for ties (1..n)."""
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)

    xs = x[order]
    n = len(xs)
    i = 0
    while i < n:
        j = i + 1
        while j < n and xs[j] == xs[i]:
            j += 1
        # average rank for [i, j)
        r = 0.5 * ((i + 1) + j)  # 1-indexed
        ranks[order[i:j]] = r
        i = j
    return ranks


def spearmanr_np(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation (robust to ties via average ranks)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size != b.size:
        raise ValueError("spearmanr_np: size mismatch")
    if a.size < 2:
        return float("nan")

    ra = _rankdata_average_ties(a)
    rb = _rankdata_average_ties(b)
    ra = ra - np.mean(ra)
    rb = rb - np.mean(rb)

    denom = float(np.sqrt(np.sum(ra * ra) * np.sum(rb * rb)) + 1e-12)
    return float(np.sum(ra * rb) / denom)


def select_diverse_topk(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    min_hamming: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pick up to k smallest-y rows, enforcing min Hamming distance greedily."""
    if len(X) == 0 or k <= 0:
        return np.zeros((0, X.shape[1] if X.ndim == 2 else 0), dtype=getattr(X, "dtype", np.int8)), np.zeros((0,), dtype=float)

    X = np.asarray(X)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(y)
    Xs = X[order]
    ys = y[order]

    picked_X: List[np.ndarray] = []
    picked_y: List[float] = []

    for xi, yi in zip(Xs, ys):
        if not picked_X:
            picked_X.append(xi)
            picked_y.append(float(yi))
        else:
            if min_hamming > 0:
                ok = True
                for xj in picked_X:
                    if hamming_distance(xi, xj) < min_hamming:
                        ok = False
                        break
                if not ok:
                    continue
            picked_X.append(xi)
            picked_y.append(float(yi))

        if len(picked_X) >= k:
            break

    return np.array(picked_X, dtype=X.dtype), np.array(picked_y, dtype=np.float64)
