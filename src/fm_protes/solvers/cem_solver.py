from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import ObjectiveFn, Solver, SolverResult


@dataclass
class CEMSolver(Solver):
    """Cross-Entropy Method baseline (product Bernoulli distribution).

    This is NOT TT-based, but is a strong simple baseline and a good fallback
    when PROTES is not installed.
    """

    name: str = "cem"
    batch_size: int = 512
    elite_frac: float = 0.1
    n_iters: int = 50
    lr: float = 0.7
    init_p: float = 0.5
    min_p: float = 0.01
    max_p: float = 0.99

    def solve(self, objective: ObjectiveFn, d: int, budget: int, pool_size: int, seed: int) -> SolverResult:
        rng = np.random.default_rng(seed)
        p = np.full((d,), self.init_p, dtype=np.float64)

        X_all = []
        y_all = []

        evals = 0
        it = 0
        while evals < budget and it < self.n_iters:
            B = min(self.batch_size, budget - evals)
            X = (rng.random((B, d)) < p[None, :]).astype(np.int8)
            y = objective(X).astype(np.float64)

            X_all.append(X)
            y_all.append(y)
            evals += B

            # elite update
            elite_n = max(1, int(self.elite_frac * B))
            elite_idx = np.argsort(y)[:elite_n]
            elite = X[elite_idx]
            p_new = np.mean(elite, axis=0)
            p = (1.0 - self.lr) * p + self.lr * p_new
            p = np.clip(p, self.min_p, self.max_p)

            it += 1

        X_pool = np.concatenate(X_all, axis=0) if X_all else np.zeros((0, d), dtype=np.int8)
        y_pool = np.concatenate(y_all, axis=0) if y_all else np.zeros((0,), dtype=np.float64)

        idx = int(np.argmin(y_pool)) if len(y_pool) else 0
        X_best = X_pool[idx].copy() if len(y_pool) else np.zeros((d,), dtype=np.int8)
        y_best = float(y_pool[idx]) if len(y_pool) else float("inf")

        # If pool_size is requested smaller than total evaluations, we keep best unique subset later in loop.
        return SolverResult(
            X_pool=X_pool,
            y_pool=y_pool,
            X_best=X_best,
            y_best=y_best,
            info={"evaluations": float(evals), "iters": float(it)},
        )
