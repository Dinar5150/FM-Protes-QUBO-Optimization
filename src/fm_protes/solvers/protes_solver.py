from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import ObjectiveFn, Solver, SolverResult
from ..utils import topk_unique

from ..constraints import CardinalityConstraint, CompositeConstraint, Constraint, OneHotGroupsConstraint

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

    @staticmethod
    def _extract_onehot_groups(cons: Optional[Constraint]) -> Optional[OneHotGroupsConstraint]:
        if cons is None:
            return None
        if isinstance(cons, OneHotGroupsConstraint):
            return cons
        if isinstance(cons, CompositeConstraint):
            for c in cons.constraints:
                out = ProtesSolver._extract_onehot_groups(c)
                if out is not None:
                    return out
        return None

    @staticmethod
    def _onehot_groups_are_contiguous_partition(groups: List[np.ndarray], *, d: int) -> Optional[List[int]]:
        """Return group sizes if groups form a contiguous partition 0..d-1.

        This is required for a simple sequential TT automaton mask.
        """
        if d <= 0:
            return None
        # Sort groups by their first index.
        gs = [np.asarray(g, dtype=np.int64).reshape(-1) for g in groups]
        if any(g.size == 0 for g in gs):
            return None
        gs.sort(key=lambda a: int(np.min(a)))

        off = 0
        sizes: List[int] = []
        seen = set()
        for g in gs:
            # No overlaps
            for idx in g.tolist():
                if idx in seen:
                    return None
                seen.add(int(idx))

            s = int(g.size)
            expected = np.arange(off, off + s, dtype=np.int64)
            if not np.array_equal(g, expected):
                return None
            sizes.append(s)
            off += s
        if off != int(d):
            return None
        return sizes

    @staticmethod
    def build_tt_mask_onehot_groups_P(*, group_sizes: List[int]) -> Tuple[Any, str]:
        """Build an exact TT feasibility indicator for one-hot groups.

        Constraint: for each group G, sum_{i in G} x_i == 1.

        Uses a small DFA with 3 states tracking ones count within the current group:
          0: seen 0 ones
          1: seen exactly 1 one
          2: invalid (>=2 ones or failed group end check)

        At the end of each group we require state==1 and reset to state 0.

        Returns PROTES' special P format [Yl, Ym, Yr].
        """
        sizes = [int(s) for s in group_sizes]
        if any(s <= 0 for s in sizes):
            raise ValueError("group_sizes must be positive")
        d = int(sum(sizes))
        if d < 2:
            raise ValueError("TT mask requires d>=2")

        n = 2
        r = 3  # DFA states

        # Mark last position of each group.
        last_pos = np.zeros((d,), dtype=bool)
        off = 0
        for s in sizes:
            last_pos[off + s - 1] = True
            off += s

        def trans_internal(state: int, bit: int) -> int:
            if state == 2:
                return 2
            if state == 0:
                return 1 if bit == 1 else 0
            # state == 1
            return 2 if bit == 1 else 1

        def trans(state: int, bit: int, *, is_group_end: bool) -> int:
            s2 = trans_internal(state, bit)
            if not is_group_end:
                return s2
            # group end: must have exactly one then reset to 0
            return 0 if s2 == 1 else 2

        # First core: (1, 2, r)
        Yl = np.zeros((1, n, r), dtype=np.float32)
        for bit in (0, 1):
            ns = trans(0, bit, is_group_end=bool(last_pos[0]))
            Yl[0, bit, ns] = 1.0

        # Middle cores: (d-2, r, 2, r)
        if d > 2:
            Ym = np.zeros((d - 2, r, n, r), dtype=np.float32)
            for t in range(1, d - 1):
                is_end = bool(last_pos[t])
                for s in range(r):
                    for bit in (0, 1):
                        ns = trans(s, bit, is_group_end=is_end)
                        Ym[t - 1, s, bit, ns] = 1.0
        else:
            Ym = np.zeros((0, r, n, r), dtype=np.float32)

        # Last core: (r, 2, 1)
        Yr = np.zeros((r, n, 1), dtype=np.float32)
        is_end = bool(last_pos[d - 1])
        for s in range(r):
            for bit in (0, 1):
                ns = trans(s, bit, is_group_end=is_end)
                if ns == 0:
                    Yr[s, bit, 0] = 1.0

        desc = f"tt_mask_onehot_groups(group_sizes={sizes})"
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
