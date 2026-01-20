from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .base import ObjectiveFn, Solver, SolverResult
from ..utils import topk_unique


@dataclass
class RandomSolver(Solver):
    """Random search baseline (unconstrained sampling; feasibility handled by surrogate penalty / clf term)."""

    name: str = "random"
    p_one: float = 0.5

    def solve(self, objective: ObjectiveFn, d: int, budget: int, pool_size: int, seed: int) -> SolverResult:
        rng = np.random.default_rng(seed)
        B = int(budget)  # must respect evaluation budget
        X = (rng.random((B, d)) < self.p_one).astype(np.int8)
        y = objective(X).astype(np.float64)

        idx = int(np.argmin(y)) if len(y) else 0
        X_best = X[idx].copy() if len(y) else np.zeros((d,), dtype=np.int8)
        y_best = float(y[idx]) if len(y) else float("inf")

        # bounded pool
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


@dataclass
class RandomFeasibleSolver(Solver):
    """Random feasible sampling baseline (matches guide baseline: 'Random feasible sampling').

    Requires a feasible sampler (typically Benchmark.sample_feasible).
    """

    name: str = "random_feasible"
    sample_feasible: Callable[[np.random.Generator, int], np.ndarray] = None  # (rng, n) -> (n,d) feasible

    def solve(self, objective: ObjectiveFn, d: int, budget: int, pool_size: int, seed: int) -> SolverResult:
        if self.sample_feasible is None:
            raise RuntimeError("RandomFeasibleSolver requires sample_feasible(rng, n).")

        rng = np.random.default_rng(seed)
        B = int(budget)
        X = np.asarray(self.sample_feasible(rng, B), dtype=np.int8)
        if X.ndim != 2 or X.shape[1] != int(d):
            raise ValueError(f"sample_feasible must return (B,d); got {X.shape}")

        y = objective(X).astype(np.float64)

        idx = int(np.argmin(y)) if len(y) else 0
        X_best = X[idx].copy() if len(y) else np.zeros((d,), dtype=np.int8)
        y_best = float(y[idx]) if len(y) else float("inf")

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
