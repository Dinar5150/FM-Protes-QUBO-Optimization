from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from .base import ObjectiveFn, Solver, SolverResult

try:
    from protes import protes as _protes
    _HAS_PROTES = True
except Exception:
    _HAS_PROTES = False
    _protes = None


def has_protes() -> bool:
    return _HAS_PROTES


@dataclass
class ProtesSolver(Solver):
    """PROTES solver wrapper.

    Requires:
        pip install protes==0.3.12

    API reference:
        https://pypi.org/project/protes/
        https://github.com/anabatsh/PROTES
    """

    name: str = "protes"
    batch_size: int = 256   # k
    elite_size: int = 20    # k_top
    k_gd: int = 1
    lr: float = 5e-2
    r: int = 5
    log: bool = False

    def solve(self, objective: ObjectiveFn, d: int, budget: int, pool_size: int, seed: int) -> SolverResult:
        if not _HAS_PROTES:
            raise RuntimeError(
                "PROTES is not installed. Install with: pip install protes==0.3.12 "
                "(or switch solver.kind to 'cem')."
            )

        X_log: List[np.ndarray] = []
        y_log: List[np.ndarray] = []
        info: Dict[str, Any] = {}

        def f_batch(I):
            # I: jax array [B, d] with entries in {0,1} for binary mode (n=2)
            X = np.array(I, dtype=np.int8)
            y = objective(X).astype(np.float64)
            X_log.append(X)
            y_log.append(y)
            return y

        i_opt, y_opt = _protes(
            f=f_batch,
            d=int(d),
            n=2,
            m=int(budget),
            k=int(self.batch_size),
            k_top=int(self.elite_size),
            k_gd=int(self.k_gd),
            lr=float(self.lr),
            r=int(self.r),
            seed=int(seed),
            log=bool(self.log),
            info=info,
        )

        # Collect pool
        if X_log:
            X_pool = np.concatenate(X_log, axis=0)
            y_pool = np.concatenate(y_log, axis=0)
        else:
            X_pool = np.zeros((0, d), dtype=np.int8)
            y_pool = np.zeros((0,), dtype=np.float64)

        # best index from return
        X_best = np.array(i_opt, dtype=np.int8).reshape((d,))
        y_best = float(y_opt)

        # Some info keys depend on protes version; keep it flexible
        out_info = {"evaluations": float(len(y_pool)), "returned_y": float(y_opt)}
        for k, v in info.items():
            if isinstance(v, (int, float, np.number)):
                out_info[f"protes_{k}"] = float(v)

        return SolverResult(
            X_pool=X_pool,
            y_pool=y_pool,
            X_best=X_best,
            y_best=y_best,
            info=out_info,
        )
