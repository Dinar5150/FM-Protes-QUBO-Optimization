from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import ObjectiveFn, Solver, SolverResult
from ..constraints import CardinalityConstraint
from ..utils import topk_unique

try:
    import dimod  # noqa: F401
    import neal
    _HAS_SA = True
except Exception:
    _HAS_SA = False
    neal = None


def has_sa() -> bool:
    return _HAS_SA


def _qubo_from_upper(Q: np.ndarray) -> Dict[Tuple[int, int], float]:
    """Convert upper-triangular dense Q into a dimod QUBO dict."""
    Q = np.asarray(Q, dtype=np.float64)
    d = Q.shape[0]
    qubo: Dict[Tuple[int, int], float] = {}
    for i in range(d):
        v = float(Q[i, i])
        if v != 0.0:
            qubo[(i, i)] = v
    for i in range(d):
        for j in range(i + 1, d):
            v = float(Q[i, j])
            if v != 0.0:
                qubo[(i, j)] = v
    return qubo


def _add_cardinality_penalty_inplace(
    qubo: Dict[Tuple[int, int], float],
    *,
    d: int,
    K: int,
    rho: float,
) -> float:
    """Add rho*(sum x - K)^2 to QUBO dict in-place. Returns constant offset added."""
    # (sum x - K)^2 = (sum x)^2 - 2K sum x + K^2
    # (sum x)^2 = sum x_i + 2 * sum_{i<j} x_i x_j   for binary.
    # linear: rho*(1 - 2K) per variable
    lin = float(rho) * (1.0 - 2.0 * float(K))
    for i in range(d):
        qubo[(i, i)] = float(qubo.get((i, i), 0.0) + lin)

    # quadratic: 2*rho for each i<j
    quad = 2.0 * float(rho)
    if quad != 0.0:
        for i in range(d):
            for j in range(i + 1, d):
                qubo[(i, j)] = float(qubo.get((i, j), 0.0) + quad)

    # constant offset
    return float(rho) * float(K) * float(K)


@dataclass
class SASolver(Solver):
    """Simulated annealing baseline using dwave-neal.

    Notes:
    - Requires: dimod + dwave-neal
    - Operates on a QUBO/BQM, so it can only handle objectives that are (or can be made) quadratic.
    """

    name: str = "sa"
    num_sweeps: int = 2000
    beta_range: Optional[Tuple[float, float]] = None  # e.g., (0.1, 10.0); None lets neal choose

    def solve(self, objective: ObjectiveFn, d: int, budget: int, pool_size: int, seed: int) -> SolverResult:
        if not _HAS_SA:
            raise RuntimeError("SA solver requires 'dimod' and 'dwave-neal'. Install: pip install dimod dwave-neal")

        # Expect SurrogateObjective-like object (from this repo) so we can build a QUBO
        Q = getattr(objective, "Q", None)
        const = getattr(objective, "const", 0.0)
        constraint = getattr(objective, "constraint", None)
        rho = float(getattr(objective, "rho", 0.0))
        alpha = float(getattr(objective, "alpha", 0.0))
        p_feasible = getattr(objective, "p_feasible", None)

        if Q is None:
            raise ValueError("SASolver requires an objective with attribute 'Q' (expected SurrogateObjective).")

        if p_feasible is not None and alpha != 0.0:
            raise RuntimeError("SASolver cannot include -alpha*log(p_feasible); disable feasibility_term or use another solver.")

        qubo = _qubo_from_upper(np.asarray(Q, dtype=np.float64))
        offset = float(const)

        # Only support constraint penalty if it stays quadratic (cardinality)
        if constraint is not None and rho != 0.0:
            if isinstance(constraint, CardinalityConstraint):
                offset += _add_cardinality_penalty_inplace(qubo, d=int(d), K=int(constraint.K), rho=float(rho))
            else:
                raise RuntimeError(
                    "SASolver only supports penalty constraints that are quadratic. "
                    "Supported: CardinalityConstraint with rho*(sum-K)^2. "
                    "Use CEM/PROTES for non-quadratic violations."
                )

        import dimod  # local import after dependency check

        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo, offset=offset)

        sampler = neal.SimulatedAnnealingSampler()

        num_reads = int(max(1, budget))  # interpret budget as number of returned SA samples
        ss = sampler.sample(
            bqm,
            num_reads=num_reads,
            num_sweeps=int(self.num_sweeps),
            beta_range=self.beta_range,
            seed=int(seed),
        )

        # Variables are integer-labeled 0..d-1, so SampleSet order is deterministic.
        X = ss.record.sample.astype(np.int8, copy=False)
        y = ss.record.energy.astype(np.float64, copy=False)

        # pool pruning
        if pool_size is not None and int(pool_size) > 0 and len(X) > int(pool_size):
            X_pool, y_pool = topk_unique(X, y, k=int(pool_size))
        else:
            X_pool, y_pool = X, y

        idx = int(np.argmin(y_pool)) if len(y_pool) else 0
        X_best = X_pool[idx].copy() if len(y_pool) else np.zeros((int(d),), dtype=np.int8)
        y_best = float(y_pool[idx]) if len(y_pool) else float("inf")

        return SolverResult(
            X_pool=X_pool,
            y_pool=y_pool,
            X_best=X_best,
            y_best=y_best,
            info={
                "evaluations": float(num_reads),
                "sa_num_sweeps": float(self.num_sweeps),
            },
        )
