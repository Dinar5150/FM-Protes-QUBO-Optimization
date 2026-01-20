from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import ObjectiveFn, Solver, SolverResult
from ..utils import topk_unique


def _enumerate_binary(d: int) -> np.ndarray:
    """Return all binary vectors of length d as (2^d, d) uint8."""
    n = 1 << int(d)
    I = np.arange(n, dtype=np.uint32)[:, None]
    bits = (I >> np.arange(d, dtype=np.uint32)[None, :]) & 1
    return bits.astype(np.int8)


@dataclass
class ExactEnumSolver(Solver):
    """Exact solver by full enumeration (validation / debugging only).

    Evaluates all 2^d candidates; only feasible for small d.
    """

    name: str = "exact_enum"
    max_d: int = 24
    batch_eval: int = 8192  # objective batch size to avoid huge temporary allocations

    def solve(self, objective: ObjectiveFn, d: int, budget: int, pool_size: int, seed: int) -> SolverResult:
        d = int(d)
        if d > int(self.max_d):
            raise RuntimeError(f"ExactEnumSolver: d={d} exceeds max_d={self.max_d} (2^d too large).")

        X_all = _enumerate_binary(d)
        n = int(len(X_all))

        # budget is ignored by design (this is exact); report true eval count
        y_all = np.empty((n,), dtype=np.float64)
        for s in range(0, n, int(self.batch_eval)):
            xb = X_all[s : s + int(self.batch_eval)]
            y_all[s : s + len(xb)] = objective(xb).astype(np.float64)

        idx = int(np.argmin(y_all)) if n else 0
        X_best = X_all[idx].copy() if n else np.zeros((d,), dtype=np.int8)
        y_best = float(y_all[idx]) if n else float("inf")

        if pool_size is not None and int(pool_size) > 0 and n > int(pool_size):
            X_pool, y_pool = topk_unique(X_all, y_all, k=int(pool_size))
        else:
            X_pool, y_pool = X_all, y_all

        return SolverResult(
            X_pool=X_pool,
            y_pool=y_pool,
            X_best=X_best,
            y_best=y_best,
            info={"evaluations": float(n), "ignored_budget": float(budget)},
        )
