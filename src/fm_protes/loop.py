from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from .benchmarks import KnapsackBenchmark, MaxCutCardinalityBenchmark
from .constraints import Constraint, batch_is_feasible
from .data import DataBuffer
from .fm import FMTrainConfig, fm_predict_proba, fm_to_qubo, train_fm_classifier, train_fm_regression
from .surrogate import SurrogateObjective
from .utils import ensure_dir, save_json, set_global_seed, topk_unique
from .solvers import CEMSolver, ProtesSolver, RandomSolver, has_protes


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
    raise ValueError(f"Unknown benchmark kind: {kind}")


def build_solver(cfg: Dict[str, Any]):
    kind = cfg["kind"].lower()
    if kind == "protes":
        if not has_protes():
            print("[warn] solver.kind=protes but protes is not installed; falling back to CEM")
            kind = "cem"
        else:
            return ProtesSolver(
                batch_size=int(cfg.get("batch_size", 256)),
                elite_size=int(cfg.get("elite_size", 20)),
                k_gd=int(cfg.get("k_gd", 1)),
                lr=float(cfg.get("lr", 5e-2)),
                r=int(cfg.get("r", 5)),
                log=bool(cfg.get("log", False)),
            )

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
    cons = bench.constraint()

    # Save config + benchmark info
    save_json(out_dir / "benchmark.json", bench.info())

    # Initial data
    data = init_dataset(bench, cons, config.get("init_dataset", {}), rng)

    d = bench.n_vars()

    # Training configs
    fm_reg_cfg = FMTrainConfig(**config.get("fm_reg", {}))
    fm_clf_cfg = FMTrainConfig(**config.get("fm_clf", {}))

    # Loop configs
    n_iters = int(config.get("n_iters", 30))
    top_k = int(config.get("top_k", 20))
    solver_budget = int(config.get("solver_budget", 5000))
    candidate_pool_k = int(config.get("candidate_pool_k", 500))  # top unique from solver pool

    rho = float(config.get("penalty", {}).get("rho", 0.0))
    alpha = float(config.get("feasibility_term", {}).get("alpha", 0.0))
    use_clf = bool(config.get("feasibility_term", {}).get("enabled", False))
    min_clf_points = int(config.get("feasibility_term", {}).get("min_points", 200))

    solver = build_solver(config["solver"])

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

        # --- Train feasibility classifier (optional)
        p_feasible_fn = None
        clf_info = {}
        if use_clf and data.size_clf() >= min_clf_points:
            X_clf, y_clf = data.get_clf_arrays()
            fm_clf, clf_info = train_fm_classifier(
                X_clf, y_clf, fm_clf_cfg, seed=seed + 2000 + it
            )
            p_feasible_fn = lambda X: fm_predict_proba(fm_clf, X)

        # --- Build surrogate objective for solver
        surrogate = SurrogateObjective(
            Q=Q,
            const=const,
            constraint=cons,
            rho=rho,
            p_feasible=p_feasible_fn,
            alpha=alpha,
        )

        # --- Solve surrogate with PROTES (or chosen solver)
        result = solver.solve(
            objective=surrogate,
            d=d,
            budget=solver_budget,
            pool_size=candidate_pool_k,
            seed=seed + 3000 + it,
        )

        # Choose top candidates from evaluated pool (unique)
        X_pool, y_pool = result.X_pool, result.y_pool
        if len(X_pool) == 0:
            # fallback: at least evaluate returned best
            X_pool = result.X_best.reshape(1, -1)
            y_pool = surrogate(X_pool)

        X_top, y_top = topk_unique(X_pool, y_pool, k=candidate_pool_k)
        X_query = X_top[:top_k] if len(X_top) >= top_k else X_top

        # --- Query oracle (feasible only), always log feasibility
        feas_mask = batch_is_feasible(X_query, cons)
        n_feas = int(np.sum(feas_mask))
        n_infeas = int(len(X_query) - n_feas)

        # Add to classifier dataset
        for x, ok in zip(X_query, feas_mask):
            data.add_clf(x, 1 if bool(ok) else 0)

        # Evaluate oracle on feasible only
        y_oracle_new = []
        for x, ok in zip(X_query, feas_mask):
            if not bool(ok):
                continue
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
            "query_feasible": n_feas,
            "query_infeasible": n_infeas,
            "best_y": best_y,
            "time_sec": dt,
            **reg_info,
            **clf_info,
            **result.info,
        }
        rows.append(row)

        print(
            f"[iter {it:03d}] best_y={best_y:.6g} | oracle_calls={oracle_calls} | "
            f"feasible {n_feas}/{len(X_query)} | reg_n={data.size_reg()} | {solver.name}"
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
