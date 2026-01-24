from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from .benchmarks import KnapsackBenchmark, MaxCutCardinalityBenchmark, OneHotQUBOBenchmark
from .constraints import (
    CardinalityConstraint,
    Constraint,
    LinearInequalityConstraint,
    OneHotGroupsConstraint,
    batch_is_feasible,
    split_constraints,
)
from .data import DataBuffer
from .fm import FMTrainConfig, fm_predict_proba, fm_predict_reg, fm_to_qubo, train_fm_classifier, train_fm_regression
from .surrogate import SurrogateObjective
from .utils import (
    ensure_dir,
    save_json,
    set_global_seed,
    topk_unique,
    select_diverse_topk,
    spearmanr_np,
    build_slack_qubo_for_linear_inequalities,
)
from .solvers import CEMSolver, ProtesSolver, RandomSolver, RandomFeasibleSolver, has_protes
from .solvers.sa_solver import SASolver, has_sa
from .solvers.exact_enum_solver import ExactEnumSolver
from .solvers.tabu_solver import TabuSolver, has_tabu
from .solvers.qbsolv_solver import QBSolvSolver, QBSolvConfig, has_qbsolv


def build_benchmark(cfg: Dict[str, Any]):
    kind = cfg["kind"].lower()
    if kind == "maxcut_cardinality":
        return MaxCutCardinalityBenchmark(
            d=int(cfg.get("d", 60)),
            K=int(cfg.get("K", 20)),
            seed=int(cfg.get("seed", 0)),
            weight_scale=float(cfg.get("weight_scale", 1.0)),
        )
    if kind == "knapsack":
        return KnapsackBenchmark(
            d=int(cfg.get("d", 200)),
            seed=int(cfg.get("seed", 0)),
            capacity_ratio=float(cfg.get("capacity_ratio", 0.35)),
        )
    if kind == "onehot_qubo":
        return OneHotQUBOBenchmark(
            group_sizes=cfg.get("group_sizes", [5, 5, 5, 5]),
            seed=int(cfg.get("seed", 0)),
            interaction_scale=float(cfg.get("interaction_scale", 1.0)),
            planted_bias=float(cfg.get("planted_bias", 2.0)),
        )
    raise ValueError(f"Unknown benchmark kind: {kind}")


def build_solver(cfg: Dict[str, Any], *, bench=None, cons=None):
    kind = cfg["kind"].lower()

    if kind == "protes":
        if not has_protes():
            print("[warn] solver.kind=protes but protes is not installed; falling back to CEM")
            kind = "cem"
        else:
            solver = ProtesSolver(
                batch_size=int(cfg.get("batch_size", 256)),
                elite_size=int(cfg.get("elite_size", 20)),
                k_gd=int(cfg.get("k_gd", 1)),
                lr=float(cfg.get("lr", 5e-2)),
                r=int(cfg.get("r", 5)),
                log=bool(cfg.get("log", False)),
            )

            # Optional hard feasibility mask via TT for simple constraints.
            # This implements a *hard* sampling restriction for PROTES by
            # zeroing probability on infeasible assignments for supported constraints.
            if bool(cfg.get("tt_hard_mask", False)) and bench is not None and cons is not None:
                d = int(bench.n_vars())

                # Currently supported: exact CardinalityConstraint (sum(x)=K)
                card = ProtesSolver._extract_cardinality(cons)
                if card is not None:
                    P, desc = ProtesSolver.build_tt_mask_cardinality_P(d=d, K=int(card.K))
                    solver.P_init = P
                    solver.P_init_desc = desc
                else:
                    print("[warn] solver.kind=protes tt_hard_mask=true but no supported hard constraint was found (currently supports CardinalityConstraint only).")

            return solver

    if kind == "cem":
        return CEMSolver(
            batch_size=int(cfg.get("batch_size", 512)),
            elite_frac=float(cfg.get("elite_frac", 0.1)),
            n_iters=int(cfg.get("n_iters", 50)),
            lr=float(cfg.get("lr", 0.7)),
            init_p=float(cfg.get("init_p", 0.5)),
        )

    if kind == "random":
        return RandomSolver(p_one=float(cfg.get("p_one", 0.5)))

    if kind == "random_feasible":
        if bench is None:
            raise RuntimeError("solver.kind=random_feasible requires bench.sample_feasible.")
        return RandomFeasibleSolver(sample_feasible=bench.sample_feasible)

    if kind == "exact_enum":
        return ExactEnumSolver(
            max_d=int(cfg.get("max_d", 24)),
            batch_eval=int(cfg.get("batch_eval", 8192)),
        )

    if kind == "sa":
        if not has_sa():
            raise RuntimeError("solver.kind=sa requires 'dimod' and 'dwave-neal'. Install: pip install dimod dwave-neal")
        return SASolver(
            num_sweeps=int(cfg.get("num_sweeps", 2000)),
            beta_range=tuple(cfg["beta_range"]) if "beta_range" in cfg and cfg["beta_range"] is not None else None,
        )

    if kind == "tabu":
        if not has_tabu():
            raise RuntimeError("solver.kind=tabu requires 'dimod' and 'dwave-tabu'. Install: pip install dimod dwave-tabu")
        return TabuSolver(
            timeout=int(cfg.get("timeout", 1000)),
            tenure=int(cfg["tenure"]) if "tenure" in cfg and cfg["tenure"] is not None else None,
        )

    if kind == "qbsolv":
        if not has_qbsolv():
            raise RuntimeError("solver.kind=qbsolv requires 'dimod' and 'dwave-samplers'. Install: pip install dimod dwave-samplers")
        qbcfg = QBSolvConfig(
            subproblem_size=int(cfg.get("subproblem_size", 400)),
            max_outer_iters=int(cfg.get("max_outer_iters", 50)),
            max_no_improve=int(cfg.get("max_no_improve", 10)),
            tol=float(cfg.get("tol", 0.0)),
            sub_num_reads=int(cfg.get("sub_num_reads", 200)),
            sub_num_sweeps=int(cfg.get("sub_num_sweeps", 1000)),
            sub_beta_range=tuple(cfg["sub_beta_range"]) if cfg.get("sub_beta_range") is not None else None,
            polish_with_steepest_descent=bool(cfg.get("polish_with_steepest_descent", True)),
            candidates_per_subproblem=int(cfg.get("candidates_per_subproblem", 25)),
        )
        return QBSolvSolver(cfg=qbcfg)

    raise ValueError(f"Unknown solver kind: {kind}")


def init_dataset(
    bench,
    cons: Optional[Constraint],
    cfg: Dict[str, Any],
    rng: np.random.Generator,
) -> DataBuffer:
    data = DataBuffer()

    mode = cfg.get("mode", "generate").lower()
    if mode == "generate":
        n0 = int(cfg.get("n_feasible", 200))
        X0 = bench.sample_feasible(rng, n0)
        for x in X0:
            y = bench.oracle(x)
            data.add_point(x, feasible=True, y=y)

        # Add some infeasible samples for the classifier (optional but recommended)
        n_neg = int(cfg.get("n_infeasible", 200))
        d = bench.n_vars()
        for _ in range(n_neg):
            x = (rng.random(d) < 0.5).astype(np.int8)
            feasible = True if cons is None else cons.is_feasible(x)
            data.add_clf(x, 1 if feasible else 0)

    elif mode == "load_npz":
        path = Path(cfg["path"])
        arr = np.load(path)
        X = arr["X"].astype(np.int8)
        y = arr["y"].astype(np.float64)
        if X.ndim != 2:
            raise ValueError("X in npz must be 2D")
        if len(X) != len(y):
            raise ValueError("X and y sizes mismatch in npz")
        for xi, yi in zip(X, y):
            # assume loaded points are feasible
            data.add_point(xi, feasible=True, y=float(yi))

        # Optionally add negatives
        n_neg = int(cfg.get("n_infeasible", 0))
        d = X.shape[1]
        for _ in range(n_neg):
            x = (rng.random(d) < 0.5).astype(np.int8)
            feasible = True if cons is None else cons.is_feasible(x)
            data.add_clf(x, 1 if feasible else 0)

    else:
        raise ValueError(f"Unknown init_dataset.mode: {mode}")

    return data


def run_experiment(config: Dict[str, Any], out_dir: str | Path) -> Path:
    out_dir = ensure_dir(out_dir)

    seed = int(config.get("seed", 0))
    set_global_seed(seed)
    rng = np.random.default_rng(seed)

    bench = build_benchmark(config["benchmark"])
    cons_all = bench.constraint()
    d = bench.n_vars()

    # --- Hybrid constraint handling (hard-mask some constraints + penalize the rest)
    ch_cfg = config.get("constraint_handling", {})
    hard_kinds = [str(s).lower() for s in ch_cfg.get("hard", [])]

    hard_types = []
    if "cardinality" in hard_kinds:
        hard_types.append(CardinalityConstraint)
    if "onehot" in hard_kinds or "onehot_groups" in hard_kinds:
        hard_types.append(OneHotGroupsConstraint)
    if "linear_inequality" in hard_kinds or "linear" in hard_kinds:
        hard_types.append(LinearInequalityConstraint)

    cons_hard, cons_soft = split_constraints(cons_all, hard_types=hard_types)

    # Save config + benchmark info
    save_json(out_dir / "benchmark.json", bench.info())

    # Initial data
    data = init_dataset(bench, cons_all, config.get("init_dataset", {}), rng)

    # Training configs
    fm_reg_cfg = FMTrainConfig(**config.get("fm_reg", {}))
    fm_clf_cfg = FMTrainConfig(**config.get("fm_clf", {}))

    # Loop configs
    n_iters = int(config.get("n_iters", 30))
    top_k = int(config.get("top_k", 20))
    solver_budget = int(config.get("solver_budget", 5000))
    candidate_pool_k = int(config.get("candidate_pool_k", 500))  # top unique from solver pool

    rho = float(config.get("penalty", {}).get("rho", 0.0))
    penalty_cfg = config.get("penalty", {})
    rho_adapt = bool(penalty_cfg.get("adaptive", False))
    rho_target = float(penalty_cfg.get("target_feasible_rate", 0.3))
    rho_grow = float(penalty_cfg.get("grow", 2.0))
    rho_shrink = float(penalty_cfg.get("shrink", 0.9))
    rho_min = float(penalty_cfg.get("min_rho", 0.0))
    rho_max = float(penalty_cfg.get("max_rho", 1e9))

    alpha = float(config.get("feasibility_term", {}).get("alpha", 0.0))
    use_clf = bool(config.get("feasibility_term", {}).get("enabled", False))
    min_clf_points = int(config.get("feasibility_term", {}).get("min_points", 200))

    cand_cfg = config.get("candidate_selection", {})
    min_hamming = int(cand_cfg.get("min_hamming", 0))

    solver_cfg = config["solver"]
    solver = build_solver(solver_cfg, bench=bench, cons=cons_hard)
    solver_kind = str(solver_cfg.get("kind", solver.name)).lower()
    qubo_only_solver = solver_kind in {"sa", "tabu", "qbsolv"}

    # Tracking best
    best_y = float("inf")
    best_x = None
    oracle_calls = 0

    rows = []

    for it in range(n_iters):
        t0 = time.time()

        # --- Train FM regression (feasible only)
        X_reg, y_reg = data.get_reg_arrays()
        if X_reg.shape[0] < 2:
            raise RuntimeError("Not enough feasible data to train regression FM")

        fm_reg, reg_info = train_fm_regression(
            X_reg, y_reg, fm_reg_cfg, seed=seed + 1000 + it
        )
        Q, const = fm_to_qubo(fm_reg)

        # Surrogate quality (cheap diagnostics on training set)
        try:
            y_hat = fm_predict_reg(fm_reg, X_reg).astype(np.float64)
            ss_res = float(np.sum((y_reg - y_hat) ** 2))
            ss_tot = float(np.sum((y_reg - float(np.mean(y_reg))) ** 2) + 1e-12)
            reg_info = {
                **reg_info,
                "surrogate_r2_train": float(1.0 - ss_res / ss_tot),
                "surrogate_spearman_train": float(spearmanr_np(y_reg, y_hat)),
            }
        except Exception:
            pass

        # --- Train feasibility classifier (optional)
        p_feasible_fn = None
        clf_info = {}

        use_clf_effective = bool(use_clf) and not bool(qubo_only_solver)
        if bool(use_clf) and bool(qubo_only_solver) and float(alpha) != 0.0:
            print(f"[warn] solver.kind={solver_kind} is QUBO-only here; disabling feasibility_term for this run.")

        if use_clf_effective and data.size_clf() >= min_clf_points:
            X_clf, y_clf = data.get_clf_arrays()
            fm_clf, clf_info = train_fm_classifier(
                X_clf, y_clf, fm_clf_cfg, seed=seed + 2000 + it
            )
            p_feasible_fn = lambda X: fm_predict_proba(fm_clf, X)

        # --- Build surrogate used for *ranking/proposals* (original variables)
        surrogate_rank = SurrogateObjective(
            Q=Q,
            const=const,
            constraint=cons_soft if cons_soft is not None else None,
            rho=rho,
            p_feasible=p_feasible_fn if use_clf_effective else None,
            alpha=alpha if use_clf_effective else 0.0,
        )

        # --- Build objective used by the chosen solver (may be extended for QUBO-only)
        d_solve = int(d)
        project_to_x = lambda X: np.asarray(X, dtype=np.int8)  # (B,d_solve)->(B,d)

        surrogate_solve = surrogate_rank

        if qubo_only_solver and cons_soft is not None and float(rho) != 0.0 and isinstance(cons_soft, LinearInequalityConstraint):
            # Meaningful QUBO encoding for Ax<=b: per-row slack-bit blocks
            try:
                Qe, const_e, total_slack = build_slack_qubo_for_linear_inequalities(Q, const, cons_soft.A, cons_soft.b, rho=float(rho))
                d_solve = int(d + total_slack)
                project_to_x = lambda X: np.asarray(X, dtype=np.int8)[:, : int(d)]
                surrogate_solve = SurrogateObjective(
                    Q=Qe,
                    const=const_e,
                    constraint=None,   # penalty already baked into Qe
                    rho=0.0,
                    p_feasible=None,   # QUBO-only baseline
                    alpha=0.0,
                )
            except Exception as e:
                print(f"[warn] QUBO-only solver: cannot encode Ax<=b as slack QUBO ({e}); falling back to unencoded QUBO (constraint handled at oracle-time).")
                surrogate_solve = SurrogateObjective(Q=Q, const=const, constraint=None, rho=0.0, p_feasible=None, alpha=0.0)

        # --- Solve surrogate
        result = solver.solve(
            objective=surrogate_solve,
            d=int(d_solve),
            budget=solver_budget,
            pool_size=candidate_pool_k,
            seed=seed + 3000 + it,
        )

        # Project solver pool to original x and re-rank with surrogate_rank (so proposals/oracle logic stays consistent)
        X_pool_s = result.X_pool
        if len(X_pool_s) == 0:
            X_pool_s = result.X_best.reshape(1, -1)

        X_pool = project_to_x(X_pool_s)
        y_pool = surrogate_rank(X_pool).astype(np.float64)

        X_top, y_top = topk_unique(X_pool, y_pool, k=candidate_pool_k)

        # Hard-mask stage: if configured, restrict candidate set to hard-feasible points.
        if cons_hard is not None:
            hard_mask = batch_is_feasible(X_top, cons_hard)
            if bool(np.any(hard_mask)):
                X_top = X_top[hard_mask]
                y_top = y_top[hard_mask]
            else:
                print("[warn] Hard constraints removed all candidates from solver pool; falling back to benchmark.sample_feasible for this iteration.")
                try:
                    X_top = np.asarray(bench.sample_feasible(rng, candidate_pool_k), dtype=np.int8)
                    y_top = surrogate_rank(X_top).astype(np.float64)
                except Exception as e:
                    print(f"[warn] benchmark.sample_feasible fallback failed ({e}); proceeding without hard-mask for this iteration.")

        # Diverse selection for *proposals* (may be infeasible)
        if min_hamming > 0:
            X_prop, _ = select_diverse_topk(X_top, y_top, k=top_k, min_hamming=min_hamming)
        else:
            X_prop = X_top[:top_k] if len(X_top) >= top_k else X_top

        # Feasibility of proposals (for logging + classifier labels)
        feas_mask = batch_is_feasible(X_prop, cons_all)
        n_prop_feas = int(np.sum(feas_mask))
        n_prop_infeas = int(len(X_prop) - n_prop_feas)
        prop_feas_rate = float(n_prop_feas / max(1, len(X_prop)))

        # Optional adaptive penalty update (based on observed feasibility of proposals)
        if rho_adapt and cons_all is not None:
            if prop_feas_rate < rho_target:
                rho = min(rho_max, max(rho_min, rho * rho_grow))
            else:
                rho = min(rho_max, max(rho_min, rho * rho_shrink))

        # Add proposal points to classifier dataset (both feasible + infeasible)
        for x, ok in zip(X_prop, feas_mask):
            data.add_clf(x, 1 if bool(ok) else 0)

        # Build feasible oracle queries. By default we keep the historical behavior
        # (fill up to top_k using bench.sample_feasible), but this is now configurable.
        oq_cfg = config.get("oracle_query", {})
        fill_cfg = oq_cfg.get("feasible_replenishment", {})
        fill_strategy = str(fill_cfg.get("strategy", "benchmark_sample_feasible")).lower()
        fill_max_tries = int(fill_cfg.get("max_tries", 200000))

        # Always enforce feasibility for oracle calls if we have constraints.
        if cons_all is None:
            X_query = X_prop[:top_k]
        else:
            X_feas = X_prop[feas_mask]
            if len(X_feas) >= top_k:
                X_query = X_feas[:top_k]
            else:
                need = int(top_k - len(X_feas))
                X_fill = np.zeros((0, d), dtype=np.int8)

                if need > 0 and fill_strategy in {"benchmark_sample_feasible", "bench"}:
                    try:
                        X_fill = np.asarray(bench.sample_feasible(rng, need), dtype=np.int8)
                    except Exception as e:
                        print(f"[warn] feasible_replenishment.strategy=benchmark_sample_feasible failed ({e}); querying fewer feasible points this iteration.")
                        X_fill = np.zeros((0, d), dtype=np.int8)

                elif need > 0 and fill_strategy in {"random_rejection", "rejection"}:
                    # Generic feasible replenishment that doesn't require bench.sample_feasible.
                    # May be slow for tight constraints.
                    found = []
                    tries = 0
                    while len(found) < need and tries < fill_max_tries:
                        tries += 1
                        x = (rng.random(d) < 0.5).astype(np.int8)
                        if cons_all.is_feasible(x):
                            found.append(x)
                    if len(found) < need:
                        print(f"[warn] random_rejection could not find enough feasible points ({len(found)}/{need}) after {tries} tries; querying fewer feasible points.")
                    if len(found) > 0:
                        X_fill = np.stack(found, axis=0)

                elif need > 0 and fill_strategy in {"none", "off", "disabled"}:
                    # Do nothing; accept fewer feasible queries.
                    X_fill = np.zeros((0, d), dtype=np.int8)

                else:
                    if need > 0:
                        print(f"[warn] Unknown feasible_replenishment.strategy={fill_strategy!r}; querying fewer feasible points.")

                # (optional) also label fill points as feasible for clf
                for x in X_fill:
                    data.add_clf(x, 1)

                X_query = np.concatenate([X_feas, X_fill], axis=0)

        # --- Query oracle (feasible only)
        y_oracle_new = []
        for x in X_query:
            y = bench.oracle(x)
            oracle_calls += 1
            data.add_reg(x, y)
            y_oracle_new.append(float(y))
            if y < best_y:
                best_y = float(y)
                best_x = x.copy()

        dt = time.time() - t0

        row = {
            "iter": it,
            "n_reg": data.size_reg(),
            "n_clf": data.size_clf(),
            "oracle_calls": oracle_calls,
            "query_k": int(len(X_query)),
            "query_feasible": int(len(X_query)),      # enforced feasible
            "query_infeasible": 0,
            "query_feasible_rate": 1.0,
            "proposed_k": int(len(X_prop)),
            "proposed_feasible": n_prop_feas,
            "proposed_infeasible": n_prop_infeas,
            "proposed_feasible_rate": float(prop_feas_rate),
            "rho": float(rho),
            "best_y": best_y,
            "time_sec": dt,
            "solver_d": float(d_solve),
            **reg_info,
            **clf_info,
            **result.info,
        }
        rows.append(row)

        print(
            f"[iter {it:03d}] best_y={best_y:.6g} | oracle_calls={oracle_calls} | "
            f"proposed_feasible {n_prop_feas}/{len(X_prop)} | queried={len(X_query)} | "
            f"reg_n={data.size_reg()} | {solver.name}"
        )

    # Save outputs
    hist = pd.DataFrame(rows)
    hist.to_csv(out_dir / "history.csv", index=False)

    if best_x is None:
        best_x = np.zeros((d,), dtype=np.int8)

    save_json(out_dir / "best.json", {"best_y": best_y, "best_x": best_x.tolist(), "oracle_calls": oracle_calls})
    # save config copy as yaml
    (out_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    return out_dir
