from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import ObjectiveFn, Solver, SolverResult
from ..utils import topk_unique

from ..constraints import CardinalityConstraint, CompositeConstraint, Constraint

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
    # Optional initial probability TT-tensor (PROTES' special P format).
    # If provided (e.g. as a feasibility mask TT), PROTES will sample from it.
    P_init: Optional[Any] = None
    P_init_desc: str = ""

    @staticmethod
    def _extract_cardinality(cons: Optional[Constraint]) -> Optional[CardinalityConstraint]:
        if cons is None:
            return None
        if isinstance(cons, CardinalityConstraint):
            return cons
        if isinstance(cons, CompositeConstraint):
            for c in cons.constraints:
                out = ProtesSolver._extract_cardinality(c)
                if out is not None:
                    return out
        return None

    @staticmethod
    def build_tt_mask_cardinality_P(*, d: int, K: int) -> Tuple[Any, str]:
        """Build an exact TT feasibility indicator mask for sum(x)=K.

        Returns PROTES' special P format: [Yl, Ym, Yr] with shapes:
          Yl: (1, 2, r)
          Ym: (d-2, r, 2, r)
          Yr: (r, 2, 1)
        where r = K+1.
        """
        if d < 2:
            raise ValueError("TT mask requires d>=2")
        if K < 0 or K > d:
            raise ValueError(f"Invalid K for cardinality mask: K={K}, d={d}")

        # Construct in numpy first, then convert to jax arrays (inside solve).
        r = int(K + 1)
        n = 2

        Yl = np.zeros((1, n, r), dtype=np.float32)
        # state after first var equals x0
        Yl[0, 0, 0] = 1.0
        if r > 1:
            Yl[0, 1, 1] = 1.0

        if d > 2:
            Ym = np.zeros((d - 2, r, n, r), dtype=np.float32)
            for t in range(d - 2):
                for s in range(r):
                    # x=0 keeps sum
                    Ym[t, s, 0, s] = 1.0
                    # x=1 increments sum (if possible)
                    if s + 1 < r:
                        Ym[t, s, 1, s + 1] = 1.0
        else:
            Ym = np.zeros((0, r, n, r), dtype=np.float32)

        Yr = np.zeros((r, n, 1), dtype=np.float32)
        # last var must land exactly at sum==K
        for s in range(r):
            # x=0 -> final sum = s
            if s == K:
                Yr[s, 0, 0] = 1.0
            # x=1 -> final sum = s+1
            if s + 1 == K:
                Yr[s, 1, 0] = 1.0

        desc = f"tt_mask_cardinality(sum(x)=K, K={K})"
        return [Yl, Ym, Yr], desc

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

        P = None
        if self.P_init is not None:
            # PROTES expects jax arrays in a special 3-core format.
            try:
                import jax.numpy as jnp

                Yl, Ym, Yr = self.P_init
                P = [jnp.array(Yl), jnp.array(Ym), jnp.array(Yr)]
                info["P_init_desc"] = self.P_init_desc
            except Exception as e:
                raise RuntimeError(f"Failed to prepare PROTES initial P tensor: {e}")

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
            P=P,
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
