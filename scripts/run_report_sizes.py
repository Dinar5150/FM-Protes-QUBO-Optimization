#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from fm_protes.loop import build_benchmark, run_experiment
from fm_protes.qubo_solvers import DWaveQuboSolver
from fm_protes.utils import try_add_quadratic_constraint_penalty_inplace


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


def _scaled_group_sizes(base_sizes: List[int], target_d: int) -> List[int]:
    base_d = int(sum(base_sizes))
    if base_d <= 0:
        raise ValueError("Base group sizes sum to 0")

    scale = float(target_d) / float(base_d)
    sizes = [max(1, int(round(s * scale))) for s in base_sizes]
    diff = target_d - int(sum(sizes))

    i = 0
    while diff != 0:
        j = i % len(sizes)
        if diff > 0:
            sizes[j] += 1
            diff -= 1
        else:
            if sizes[j] > 1:
                sizes[j] -= 1
                diff += 1
        i += 1

    return sizes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/onehot_qubo.yaml", help="Base YAML config")
    ap.add_argument("--sizes", default="200,400,600,800,1000", help="Comma-separated d sizes")
    ap.add_argument("--out_root", default="results/report_onehot_sizes", help="Output root dir")
    ap.add_argument("--num_reads", type=int, default=1000, help="SA num_reads")
    ap.add_argument("--save_json", action="store_true", help="Also save JSON report")
    args = ap.parse_args()

    base_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base_sizes = list(base_cfg["benchmark"].get("group_sizes", []))
    if not base_sizes:
        raise ValueError("benchmark.group_sizes is required in config")

    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []

    for d in sizes:
        cfg = copy.deepcopy(base_cfg)
        cfg["benchmark"]["group_sizes"] = _scaled_group_sizes(base_sizes, d)
        cfg["run_name"] = f"onehot_qubo_d{d}"

        out_dir = out_root / f"onehot_qubo_d{d}"
        out = run_experiment(cfg, out_dir)

        hist = pd.read_csv(out / "history.csv")
        protes_time = float(hist["time_sec"].sum()) if "time_sec" in hist.columns else float("nan")
        protes_best_y = float(hist["best_y"].iloc[-1])
        oracle_calls = int(hist["oracle_calls"].iloc[-1])

        bench = build_benchmark(cfg["benchmark"])
        cons = bench.constraint()

        protes_feasible = None
        protes_best_x = None
        best_path = out / "best.json"
        if best_path.exists() and cons is not None:
            best = json.loads(best_path.read_text(encoding="utf-8"))
            best_x = np.array(best.get("best_x", []), dtype=np.int8)
            if best_x.size:
                protes_best_x = best_x.tolist()
                protes_feasible = bool(cons.is_feasible(best_x))

        # SA with quadratic penalties (when supported)
        qubo = _qubo_from_upper(bench.Q)
        rho = float(cfg.get("penalty", {}).get("rho", 0.0))
        if cons is not None and rho != 0.0:
            added, ok = try_add_quadratic_constraint_penalty_inplace(qubo, cons, d=int(bench.n_vars()), rho=rho)
            if not ok:
                print(f"[warn] d={d}: constraint penalty not encodable; SA run is unconstrained.")

        t0 = time.time()
        ss = DWaveQuboSolver(qubo, num_reads=args.num_reads)
        sa_time = float(time.time() - t0)
        sa_sample = ss.record.sample[0]
        sa_energy = float(ss.first.energy)
        sa_feasible = bool(cons.is_feasible(sa_sample)) if cons is not None else None

        row = {
            "d": int(d),
            "protes_best_y": protes_best_y,
            "protes_feasible": protes_feasible,
            "oracle_calls": oracle_calls,
            "protes_time_sec": protes_time,
            "sa_energy": sa_energy,
            "sa_feasible": sa_feasible,
            "sa_time_sec": sa_time,
            "sa_num_reads": int(args.num_reads),
        }
        rows.append(row)

        # Save per-size summary with benchmark info + both model results
        summary = {
            "benchmark": bench.info(),
            "protes": {
                "best_y": protes_best_y,
                "feasible": protes_feasible,
                "oracle_calls": oracle_calls,
                "time_sec": protes_time,
                "best_x": protes_best_x,
            },
            "sa": {
                "energy": sa_energy,
                "feasible": sa_feasible,
                "time_sec": sa_time,
                "num_reads": int(args.num_reads),
                "sample": sa_sample.astype(int).tolist(),
            },
        }
        per_json = out_root / f"onehotqubo{d}.json"
        per_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        # Per-size plot (PROTES vs SA)
        plt.figure()
        plt.bar(["PROTES best_y", "SA energy"], [protes_best_y, sa_energy], color=["#4C78A8", "#F58518"])
        plt.title(f"OneHot QUBO d={d}")
        plt.ylabel("objective / energy")
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        per_png = out_root / f"onehotqubo{d}.png"
        plt.savefig(per_png, dpi=150, bbox_inches="tight")
        plt.close()

        print(
            f"[d={d}] protes best_y={protes_best_y:.6g} feasible={protes_feasible} "
            f"oracle_calls={oracle_calls} time={protes_time:.2f}s | "
            f"sa_energy={sa_energy:.6g} feasible={sa_feasible} time={sa_time:.2f}s"
        )

    df = pd.DataFrame(rows)
    out_csv = out_root / "report_sizes.csv"
    df.to_csv(out_csv, index=False)
    print(f"saved report: {out_csv}")

    # Optional aggregate plots kept in CSV only; per-size PNG/JSON are written above.

    if args.save_json:
        out_json = out_root / "report_sizes.json"
        out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"saved report: {out_json}")


if __name__ == "__main__":
    main()
