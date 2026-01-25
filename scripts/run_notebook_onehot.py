#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

import sys
from typing import Dict, Tuple
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from fm_protes.loop import build_benchmark, run_experiment
from fm_protes.qubo_solvers import DWaveQuboSolver
from fm_protes.utils import try_add_quadratic_constraint_penalty_inplace


def _qubo_from_upper(Q) -> Dict[Tuple[int, int], float]:
    Q = Q.astype(float)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/onehot_qubo.yaml", help="Path to YAML config")
    ap.add_argument("--out", default="results/notebook_onehot_qubo", help="Output directory")
    ap.add_argument("--num_reads", type=int, default=1000, help="D-Wave SA num_reads")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    bench = build_benchmark(cfg["benchmark"])

    out_dir = Path(args.out)
    out = run_experiment(cfg, out_dir)

    hist = pd.read_csv(out / "history.csv")
    plt.figure()
    plt.plot(hist["oracle_calls"], hist["best_y"], marker="o")
    plt.xlabel("oracle_calls")
    plt.ylabel("best_y (lower is better)")
    plt.title("Best feasible objective over time")
    plt.grid(True)
    plt.savefig(out / "history_plot_oracle_calls.png", dpi=150, bbox_inches="tight")

    cons = bench.constraint()
    # PROTES/FM best solution feasibility
    best_path = out / "best.json"
    if best_path.exists() and cons is not None:
        import json

        best = json.loads(best_path.read_text(encoding="utf-8"))
        best_x = np.array(best.get("best_x", []), dtype=np.int8)
        if best_x.size:
            print("protes_fm best feasible:", cons.is_feasible(best_x))
    rho = float(cfg.get("penalty", {}).get("rho", 0.0))
    qubo = _qubo_from_upper(bench.Q)
    if cons is not None and rho != 0.0:
        added, ok = try_add_quadratic_constraint_penalty_inplace(qubo, cons, d=int(bench.n_vars()), rho=rho)
        if not ok:
            print("[warn] SA comparison: constraint penalty is not quadratic/encodable; SA run is unconstrained.")

    ans = DWaveQuboSolver(qubo, num_reads=args.num_reads)
    conf = ans.record.sample[0]
    E = ans.first.energy
    print("qubo solver ans")
    print("energy:", E)
    if cons is not None:
        print("feasible:", cons.is_feasible(conf))


if __name__ == "__main__":
    main()
