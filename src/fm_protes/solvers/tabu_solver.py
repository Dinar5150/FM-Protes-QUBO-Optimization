from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .base import ObjectiveFn, Solver, SolverResult
from ..constraints import CardinalityConstraint
from ..utils import topk_unique, try_add_quadratic_constraint_penalty_inplace

try:
    import dimod
    from tabu import TabuSampler
    _HAS_TABU = True
except ImportError:
    _HAS_TABU = False
    TabuSampler = None


def has_tabu() -> bool:
    return _HAS_TABU


def _qubo_from_upper(Q: np.ndarray) -> Dict[Tuple[int, int], float]:
    Q = np.asarray(Q, dtype=np.float64)
    d = Q.shape[0]
    qubo: Dict[Tuple[int, int], float] = {}
    # diag
    for i in range(d):
        v = float(Q[i, i])
        if v != 0.0:
            qubo[(i, i)] = v
    # upper off-diag
    for i in range(d):
        for j in range(i + 1, d):
            v = float(Q[i, j])
            if v != 0.0:
                qubo[(i, j)] = v
    return qubo


@dataclass
class TabuSolver(Solver):
    """Tabu search baseline using dwave-tabu.

    Notes:
    - Requires: dimod + dwave-tabu
    - QUBO-only: cannot include non-quadratic violations or -alpha*log(p_feasible).
    """

    name: str = "tabu"
    timeout: int = 1000  # milliseconds per run (dwave-tabu)
    tenure: Optional[int] = None  # optional tabu tenure; None lets sampler decide

    def solve(self, objective: ObjectiveFn, d: int, budget: int, pool_size: int, seed: int) -> SolverResult:
        if not _HAS_TABU:
            raise RuntimeError("Tabu solver requires 'dimod' and 'dwave-tabu'. Install: pip install dimod dwave-tabu")

        Q = getattr(objective, "Q", None)
        const = getattr(objective, "const", 0.0)
        constraint = getattr(objective, "constraint", None)
        rho = float(getattr(objective, "rho", 0.0))
        alpha = float(getattr(objective, "alpha", 0.0))
        p_feasible = getattr(objective, "p_feasible", None)

        if Q is None:
            raise ValueError("TabuSolver requires an objective with attribute 'Q' (expected SurrogateObjective).")

        if p_feasible is not None and alpha != 0.0:
            print("[warn] TabuSolver: ignoring feasibility term (-alpha*log p_feasible); QUBO-only baseline.")

        qubo = _qubo_from_upper(np.asarray(Q, dtype=np.float64))
        offset = float(const)

        if constraint is not None and rho != 0.0:
            added, ok = try_add_quadratic_constraint_penalty_inplace(qubo, constraint, d=int(d), rho=float(rho))
            if ok:
                offset += float(added)
            else:
                print("[warn] TabuSolver: constraint penalty is not quadratic/encodable; ignoring constraint in the solver.")

        import dimod  # local import after dependency check

        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo, offset=offset)
        sampler = TabuSampler()

        num_reads = int(max(1, budget))  # interpret budget as number of returned samples
        kwargs = {"num_reads": num_reads, "timeout": int(self.timeout)}
        if self.tenure is not None:
            kwargs["tenure"] = int(self.tenure)

        ss = sampler.sample(bqm, **kwargs)

        X = ss.record.sample.astype(np.int8, copy=False)
        y = ss.record.energy.astype(np.float64, copy=False)

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
                "tabu_timeout_ms": float(self.timeout),
            },
        )
