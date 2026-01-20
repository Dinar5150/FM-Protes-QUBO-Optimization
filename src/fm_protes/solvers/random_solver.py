from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import ObjectiveFn, Solver, SolverResult
from ..utils import topk_unique


@dataclass
class RandomSolver(Solver):
    """Random search baseline."""

    name: str = "random"
    p_one: float = 0.5

    def solve(self, objective: ObjectiveFn, d: int, budget: int, pool_size: int, seed: int) -> SolverResult:
        rng = np.random.default_rng(seed)
        B = int(max(pool_size, budget))
        X = (rng.random((B, d)) < self.p_one).astype(np.int8)
        y = objective(X).astype(np.float64)

        idx = int(np.argmin(y))
        X_best = X[idx].copy()
        y_best = float(y[idx])

        # Keep only top unique pool_size (to bound memory / match API intent)
        if pool_size is not None and int(pool_size) > 0 and len(X) > int(pool_size):
            X_pool, y_pool = topk_unique(X, y, k=int(pool_size))
        else:
            X_pool, y_pool = X, y

        return SolverResult(
            X_pool=X_pool,
            y_pool=y_pool,
            X_best=X_best,
            y_best=y_best,
            info={"evaluations": float(B)},
        )
