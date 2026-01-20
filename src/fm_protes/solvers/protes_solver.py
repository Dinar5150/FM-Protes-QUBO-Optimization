from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from .base import ObjectiveFn, Solver, SolverResult
from ..utils import topk_unique

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

        info: Dict[str, Any] = {}

        # Keep a bounded pool (best unique) to avoid storing all samples when budgets get large.
        X_keep = np.zeros((0, d), dtype=np.int8)
        y_keep = np.zeros((0,), dtype=np.float64)
        evals_total = 0

        def f_batch(I):
            nonlocal X_keep, y_keep, evals_total
            X = np.array(I, dtype=np.int8)
            y = objective(X).astype(np.float64)
            evals_total += int(len(X))

            if pool_size is not None and int(pool_size) > 0:
                # merge + prune (keep a bit more before pruning to amortize)
                if len(X_keep) == 0:
                    X_keep, y_keep = X, y
                else:
                    X_keep = np.concatenate([X_keep, X], axis=0)
                    y_keep = np.concatenate([y_keep, y], axis=0)

                cap = int(pool_size)
                if len(X_keep) > 2 * cap:
                    X_keep, y_keep = topk_unique(X_keep, y_keep, k=cap)
            else:
                # store everything (original behavior)
                X_keep = np.concatenate([X_keep, X], axis=0) if len(X_keep) else X
                y_keep = np.concatenate([y_keep, y], axis=0) if len(y_keep) else y

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

        # Final prune to pool_size (if requested)
        if pool_size is not None and int(pool_size) > 0 and len(X_keep) > int(pool_size):
            X_pool, y_pool = topk_unique(X_keep, y_keep, k=int(pool_size))
        else:
            X_pool, y_pool = X_keep, y_keep

        X_best = np.array(i_opt, dtype=np.int8).reshape((d,))
        y_best = float(y_opt)

        out_info = {"evaluations": float(evals_total), "returned_y": float(y_opt)}
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
