from __future__ import annotations

from typing import Tuple

import numpy as np


def qubo_energy(X: np.ndarray, Q: np.ndarray, const: float = 0.0) -> np.ndarray:
    """Compute y = const + sum_{i<=j} Q[i,j] x_i x_j for a batch X.

    Assumes Q is upper-triangular (diagonal contains linear terms).
    X: (B, d) with 0/1.
    Returns (B,) float.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    B, d = X.shape
    if Q.shape != (d, d):
        raise ValueError("Q shape mismatch")

    # Efficient: compute (X @ Q) elementwise times X, but must avoid double counting
    # since Q is upper-triangular for i<=j exactly.
    # We'll do explicit upper triangle multiplication for clarity in template.
    y = np.full((B,), float(const), dtype=np.float64)
    # diagonal
    y += np.sum(X * np.diag(Q)[None, :], axis=1)
    # off diagonal
    iu = np.triu_indices(d, k=1)
    if len(iu[0]) > 0:
        y += np.sum((X[:, iu[0]] * X[:, iu[1]]) * Q[iu][None, :], axis=1)
    return y


def symmetrize_upper(Q_upper: np.ndarray) -> np.ndarray:
    """Return a symmetric matrix equivalent to the upper-triangular QUBO."""
    Q = np.array(Q_upper, copy=True)
    Q = Q + np.triu(Q, 1).T
    return Q
