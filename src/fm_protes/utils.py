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


def try_add_quadratic_constraint_penalty_inplace(
    qubo: Dict[Tuple[int, int], float],
    constraint: Any,
    *,
    d: int,
    rho: float,
) -> Tuple[float, bool]:
    """Try to add rho * penalty(x) to a QUBO dict in-place.

    Supported (quadratic) constraints:
      - CardinalityConstraint: rho*(sum(x)-K)^2
      - OneHotGroupsConstraint: sum_g rho*(sum_{i in g} x_i - 1)^2
      - CompositeConstraint: applies supported sub-constraints

    Returns: (added_offset, supported_flag).
    """
    if constraint is None or float(rho) == 0.0:
        return 0.0, True

    # local import to avoid hard dependency / circulars at module import time
    from .constraints import CardinalityConstraint, CompositeConstraint, OneHotGroupsConstraint  # noqa: WPS433

    d = int(d)
    rho = float(rho)

    if isinstance(constraint, CompositeConstraint):
        off = 0.0
        ok_all = True
        for c in constraint.constraints:
            oi, ok = try_add_quadratic_constraint_penalty_inplace(qubo, c, d=d, rho=rho)
            off += float(oi)
            ok_all = bool(ok_all and ok)
        return float(off), bool(ok_all)

    if isinstance(constraint, CardinalityConstraint):
        K = int(constraint.K)
        # rho*(sum x - K)^2 where (sum x)^2 = sum x_i + 2*sum_{i<j} x_i x_j for binary
        lin = rho * (1.0 - 2.0 * float(K))
        for i in range(d):
            qubo[(i, i)] = float(qubo.get((i, i), 0.0) + lin)

        quad = 2.0 * rho
        if quad != 0.0:
            for i in range(d):
                for j in range(i + 1, d):
                    qubo[(i, j)] = float(qubo.get((i, j), 0.0) + quad)

        return float(rho) * float(K) * float(K), True

    if isinstance(constraint, OneHotGroupsConstraint):
        # For each group g: rho*(sum_{i in g} x_i - 1)^2
        # = rho*(-sum_i x_i + 2*sum_{i<j} x_i x_j + 1)
        off = 0.0
        for g in constraint.groups:
            gi = np.asarray(g, dtype=np.int64).reshape(-1)
            if gi.size == 0:
                continue

            for i in gi:
                ii = int(i)
                qubo[(ii, ii)] = float(qubo.get((ii, ii), 0.0) - rho)

            quad = 2.0 * rho
            if quad != 0.0:
                for a in range(int(gi.size)):
                    ia = int(gi[a])
                    for b in range(a + 1, int(gi.size)):
                        ib = int(gi[b])
                        i, j = (ia, ib) if ia < ib else (ib, ia)
                        qubo[(i, j)] = float(qubo.get((i, j), 0.0) + quad)

            off += rho  # +rho*1^2
        return float(off), True

    return 0.0, False


def build_slack_qubo_for_linear_inequalities(
    Q_upper: np.ndarray,
    const: float,
    A: np.ndarray,
    b: np.ndarray,
    *,
    rho: float,
) -> Tuple[np.ndarray, float, int]:
    """Encode A·x <= b into a QUBO by adding per-row binary slack bits s_j so that:

        a_j·x + s_j == floor(b_j)   for each row j

    and adding penalty sum_j rho*(a_j·x + s_j - floor(b_j))^2.

    Returns: (Q_ext_upper, const_ext, total_slack_bits).
    """
    Q_upper = np.asarray(Q_upper, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    d = int(Q_upper.shape[0])
    if Q_upper.shape != (d, d):
        raise ValueError("Q_upper must be square")
    if A.ndim != 2 or A.shape[1] != d:
        raise ValueError(f"A must have shape (m,{d}); got {A.shape}")
    if b.size != int(A.shape[0]):
        raise ValueError(f"b must have shape (m,); got {b.shape} for m={A.shape[0]}")

    m = int(A.shape[0])
    rho = float(rho)
    if rho == 0.0 or m == 0:
        return np.triu(Q_upper, 0), float(const), 0

    rhs = np.floor(b.astype(np.float64) + 1e-12).astype(np.int64)
    rhs = np.maximum(rhs, 0)

    # slack bits per row: s_j in [0..rhs_j]
    nbits = np.maximum(1, np.ceil(np.log2(rhs.astype(np.float64) + 1.0))).astype(np.int64)
    offsets = np.zeros(m + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(nbits)
    total_slack = int(offsets[-1])

    dext = d + total_slack
    Qe = np.zeros((dext, dext), dtype=np.float64)
    Qe[:d, :d] = np.triu(Q_upper, 0)
    const_ext = float(const)

    nz_x = np.arange(d, dtype=np.int64)  # weights can be dense; keep simple/explicit

    for row in range(m):
        a = A[row].reshape(-1)
        C = float(rhs[row])
        nb = int(nbits[row])
        s0 = d + int(offsets[row])

        # constant
        const_ext += rho * (C * C)

        # diag for x vars
        # rho*((a_i)^2 - 2*C*a_i) on diag
        Qe[np.arange(d), np.arange(d)] += rho * (a * a - 2.0 * C * a)

        # diag for slack bits
        u = (2.0 ** np.arange(nb, dtype=np.float64))  # coefficients for slack bits
        for k in range(nb):
            idxk = s0 + k
            ck = float(u[k])
            Qe[idxk, idxk] += rho * (ck * ck - 2.0 * C * ck)

        # quadratic x-x: 2*rho*a_i*a_j
        for i in range(d):
            ai = float(a[i])
            if ai == 0.0:
                continue
            for j in range(i + 1, d):
                aj = float(a[j])
                if aj == 0.0:
                    continue
                Qe[i, j] += 2.0 * rho * ai * aj

        # quadratic x-slack: 2*rho*a_i*u_k
        for i in range(d):
            ai = float(a[i])
            if ai == 0.0:
                continue
            for k in range(nb):
                idxk = s0 + k
                Qe[i, idxk] += 2.0 * rho * ai * float(u[k])

        # quadratic slack-slack: 2*rho*u_k*u_l
        for k in range(nb):
            ck = float(u[k])
            if ck == 0.0:
                continue
            for l in range(k + 1, nb):
                cl = float(u[l])
                if cl == 0.0:
                    continue
                Qe[s0 + k, s0 + l] += 2.0 * rho * ck * cl

    return np.triu(Qe, 0), float(const_ext), int(total_slack)


def build_slack_qubo_for_single_linear_inequality(
    Q_upper: np.ndarray,
    const: float,
    a: np.ndarray,
    b: float,
    *,
    rho: float,
) -> Tuple[np.ndarray, float, int]:
    """Encode a·x <= b via slack bits (single-row special case)."""
    A = np.asarray(a, dtype=np.float64).reshape(1, -1)
    bb = np.asarray([float(b)], dtype=np.float64)
    return build_slack_qubo_for_linear_inequalities(Q_upper, const, A, bb, rho=float(rho))
