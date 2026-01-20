from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .base import ObjectiveFn, Solver, SolverResult
from ..utils import topk_unique


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

        # bounded pool (best unique)
        X_keep = np.zeros((0, d), dtype=np.int8)
        y_keep = np.zeros((0,), dtype=np.float64)

        evals = 0
        it = 0
        X_best = np.zeros((d,), dtype=np.int8)
        y_best = float("inf")

        while evals < budget and it < self.n_iters:
            B = min(self.batch_size, budget - evals)
            X = (rng.random((B, d)) < p[None, :]).astype(np.int8)
            y = objective(X).astype(np.float64)
            evals += int(B)

            # track best over all evaluations (even if pool is pruned later)
            j = int(np.argmin(y)) if len(y) else 0
            if len(y) and float(y[j]) < y_best:
                y_best = float(y[j])
                X_best = X[j].copy()

            # merge into pool + prune
            if len(X_keep) == 0:
                X_keep, y_keep = X, y
            else:
                X_keep = np.concatenate([X_keep, X], axis=0)
                y_keep = np.concatenate([y_keep, y], axis=0)

            if pool_size is not None and int(pool_size) > 0:
                cap = int(pool_size)
                if len(X_keep) > 2 * cap:
                    X_keep, y_keep = topk_unique(X_keep, y_keep, k=cap)

            # elite update
            elite_n = max(1, int(self.elite_frac * B))
            elite_idx = np.argsort(y)[:elite_n]
            elite = X[elite_idx]
            p_new = np.mean(elite, axis=0)
            p = (1.0 - self.lr) * p + self.lr * p_new
            p = np.clip(p, self.min_p, self.max_p)

            it += 1

        # final prune
        if pool_size is not None and int(pool_size) > 0 and len(X_keep) > int(pool_size):
            X_pool, y_pool = topk_unique(X_keep, y_keep, k=int(pool_size))
        else:
            X_pool, y_pool = X_keep, y_keep

        return SolverResult(
            X_pool=X_pool,
            y_pool=y_pool,
            X_best=X_best,
            y_best=float(y_best),
            info={"evaluations": float(evals), "iters": float(it)},
        )
