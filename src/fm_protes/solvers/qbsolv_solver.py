from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .base import ObjectiveFn, Solver, SolverResult
from ..constraints import CardinalityConstraint
from ..utils import topk_unique, try_add_quadratic_constraint_penalty_inplace

try:
    import dimod  # noqa: F401
    from dwave.samplers import SimulatedAnnealingSampler, SteepestDescentSolver
    _HAS_QBSOLV = True
except Exception:
    _HAS_QBSOLV = False
    SimulatedAnnealingSampler = None  # type: ignore
    SteepestDescentSolver = None  # type: ignore


def has_qbsolv() -> bool:
    return _HAS_QBSOLV


def _qubo_from_upper(Q: np.ndarray) -> Dict[Tuple[int, int], float]:
    Q = np.asarray(Q, dtype=np.float64)
    d = int(Q.shape[0])
    qubo: Dict[Tuple[int, int], float] = {}
    for i in range(d):
        v = float(Q[i, i])
        if v:
            qubo[(i, i)] = v
    for i in range(d):
        for j in range(i + 1, d):
            v = float(Q[i, j])
            if v:
                qubo[(i, j)] = v
    return qubo


@dataclass(frozen=True)
class QBSolvConfig:
    subproblem_size: int = 400
    max_outer_iters: int = 50
    max_no_improve: int = 10
    tol: float = 0.0

    # subQUBO optimizer = Simulated Annealing
    sub_num_reads: int = 200
    sub_num_sweeps: int = 1000
    sub_beta_range: Optional[Tuple[float, float]] = None

    # Optional polish
    polish_with_steepest_descent: bool = True

    # How many candidate full solutions to keep per subproblem (limits memory)
    candidates_per_subproblem: int = 25


@dataclass
class QBSolvSolver(Solver):
    """QBSOLV-style decomposition baseline (NO tabu), subsolver = SA.

    Works on a QUBO/BQM; in this repo it supports:
      - pure FM QUBO
      - + cardinality penalty rho*(sum-K)^2
    Not supported:
      - feasibility classifier term (-alpha*log p)
      - non-quadratic violations (e.g., knapsack hinge)
    """

    name: str = "qbsolv"
    cfg: QBSolvConfig = QBSolvConfig()

    def solve(self, objective: ObjectiveFn, d: int, budget: int, pool_size: int, seed: int) -> SolverResult:
        if not _HAS_QBSOLV:
            raise RuntimeError("QBSolvSolver requires 'dimod' and 'dwave-samplers'. Install: pip install dimod dwave-samplers")

        # Expect SurrogateObjective-like object
        Q = getattr(objective, "Q", None)
        const = float(getattr(objective, "const", 0.0))
        constraint = getattr(objective, "constraint", None)
        rho = float(getattr(objective, "rho", 0.0))
        alpha = float(getattr(objective, "alpha", 0.0))
        p_feasible = getattr(objective, "p_feasible", None)

        if Q is None:
            raise ValueError("QBSolvSolver requires objective.Q (expected SurrogateObjective).")

        if p_feasible is not None and alpha != 0.0:
            print("[warn] QBSolvSolver: ignoring feasibility term (-alpha*log p_feasible); QUBO-only baseline.")

        import dimod  # local import

        qubo = _qubo_from_upper(np.asarray(Q, dtype=np.float64))
        offset = float(const)

        if constraint is not None and rho != 0.0:
            added, ok = try_add_quadratic_constraint_penalty_inplace(qubo, constraint, d=int(d), rho=float(rho))
            if ok:
                offset += float(added)
            else:
                print("[warn] QBSolvSolver: constraint penalty is not quadratic/encodable; ignoring constraint in the solver.")

        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo, offset=offset)

        rng = np.random.default_rng(int(seed))
        n = int(d)

        # caches for fast energy + impact ordering
        linear = np.zeros(n, dtype=np.float64)
        for v, bias in bqm.linear.items():
            linear[int(v)] = float(bias)

        adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        for (u, v), w in bqm.quadratic.items():
            iu = int(u); iv = int(v)
            ww = float(w)
            adj[iu].append((iv, ww))
            adj[iv].append((iu, ww))

        def energy_full(x: np.ndarray) -> float:
            e = float(bqm.offset) + float(np.dot(linear, x.astype(np.float64)))
            for i in range(n):
                xi = int(x[i])
                if not xi:
                    continue
                for j, w in adj[i]:
                    if j > i and x[j]:
                        e += float(w)
            return float(e)

        def order_by_impact(x: np.ndarray) -> np.ndarray:
            field = linear.copy()
            # field_i = h_i + sum_j J_ij x_j
            for i in range(n):
                if not x[i]:
                    continue
                for j, w in adj[i]:
                    field[j] += float(w)
            delta = (1.0 - 2.0 * x.astype(np.float64)) * field
            return np.argsort(-np.abs(delta))

        def decompose_subbqm(subset: np.ndarray, xbest_fixed: np.ndarray) -> "dimod.BinaryQuadraticModel":
            in_subset = np.zeros(n, dtype=bool)
            in_subset[subset] = True

            lin: Dict[int, float] = {}
            quad: Dict[Tuple[int, int], float] = {}

            for iu in subset.astype(int):
                bias_u = float(linear[iu])
                for iv, w in adj[iu]:
                    if in_subset[iv]:
                        if iu < iv:
                            quad[(iu, iv)] = quad.get((iu, iv), 0.0) + float(w)
                    else:
                        if xbest_fixed[iv]:
                            bias_u += float(w)
                lin[iu] = bias_u

            return dimod.BinaryQuadraticModel(lin, quad, 0.0, dimod.BINARY)

        sa = SimulatedAnnealingSampler()
        sd = SteepestDescentSolver() if bool(self.cfg.polish_with_steepest_descent) else None

        # Initialize Xbest
        Xbest = rng.integers(0, 2, size=n, dtype=np.int8)
        best_energy = energy_full(Xbest)

        index = order_by_impact(Xbest)

        X_pool = np.zeros((0, n), dtype=np.int8)
        y_pool = np.zeros((0,), dtype=np.float64)

        no_improve = 0
        outer_done = 0

        for outer in range(1, int(self.cfg.max_outer_iters) + 1):
            outer_done = outer
            X = Xbest.copy()
            E = float(best_energy)

            for start in range(0, n, int(self.cfg.subproblem_size)):
                subset = index[start : start + int(self.cfg.subproblem_size)]
                if subset.size == 0:
                    continue

                sub_bqm = decompose_subbqm(subset, Xbest)

                # seed initial state from Xbest restricted to sub vars
                init = {int(v): int(Xbest[int(v)]) for v in sub_bqm.variables}
                init_ss = dimod.SampleSet.from_samples(init, vartype=dimod.BINARY, energy=0.0)

                sa_kwargs = dict(
                    num_reads=int(self.cfg.sub_num_reads),
                    num_sweeps=int(self.cfg.sub_num_sweeps),
                    seed=int(seed),
                    initial_states=init_ss,
                    initial_states_generator="tile",
                )
                if self.cfg.sub_beta_range is not None:
                    sa_kwargs["beta_range"] = list(self.cfg.sub_beta_range)

                ss = sa.sample(sub_bqm, **sa_kwargs)
                if sd is not None:
                    ss = sd.sample(sub_bqm, initial_states=ss)

                # take a few best candidates from this subproblem and lift to full X for the pool
                R = int(min(int(self.cfg.candidates_per_subproblem), len(ss)))
                if R > 0 and pool_size is not None and int(pool_size) > 0:
                    # ss is already sorted by energy (lowest first) for these samplers
                    for r in range(R):
                        x_cand = X.copy()
                        sub_sample = ss.record.sample[r]
                        # ss.variables is the variable order in record.sample columns
                        for col, v in enumerate(ss.variables):
                            x_cand[int(v)] = int(sub_sample[col])
                        e_cand = energy_full(x_cand)
                        X_pool = np.concatenate([X_pool, x_cand.reshape(1, -1)], axis=0)
                        y_pool = np.concatenate([y_pool, np.array([e_cand], dtype=np.float64)], axis=0)
                    # prune pool periodically
                    cap = int(pool_size)
                    if len(X_pool) > 2 * cap:
                        X_pool, y_pool = topk_unique(X_pool, y_pool, k=cap)

                # best sub-solution -> apply to current X
                best_sub = ss.first.sample
                for v, val in best_sub.items():
                    X[int(v)] = 1 if int(val) else 0
                E = energy_full(X)

            index = order_by_impact(X)

            if E < best_energy - float(self.cfg.tol):
                Xbest = X
                best_energy = float(E)
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= int(self.cfg.max_no_improve):
                    break

        # finalize pool
        if pool_size is not None and int(pool_size) > 0 and len(X_pool) > int(pool_size):
            X_pool, y_pool = topk_unique(X_pool, y_pool, k=int(pool_size))

        # ensure non-empty pool (loop expects some candidates)
        if len(X_pool) == 0:
            X_pool = Xbest.reshape(1, -1).copy()
            y_pool = np.array([float(best_energy)], dtype=np.float64)

        idx = int(np.argmin(y_pool)) if len(y_pool) else 0
        X_best = X_pool[idx].copy()
        y_best = float(y_pool[idx])

        return SolverResult(
            X_pool=X_pool,
            y_pool=y_pool,
            X_best=X_best,
            y_best=y_best,
            info={
                "evaluations": float(len(X_pool)),  # surrogate energies computed into the returned pool
                "qbsolv_outer_iters": float(outer_done),
            },
        )
