from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import ObjectiveFn, Solver, SolverResult


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
        return SolverResult(
            X_pool=X,
            y_pool=y,
            X_best=X[idx].copy(),
            y_best=float(y[idx]),
            info={"evaluations": float(B)},
        )
