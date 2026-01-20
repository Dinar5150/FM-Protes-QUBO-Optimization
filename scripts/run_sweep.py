#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import datetime as dt
from pathlib import Path

import pandas as pd
import yaml

import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from fm_protes.loop import run_experiment
from fm_protes.utils import ensure_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Base YAML config")
    ap.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated seeds")
    ap.add_argument("--solvers", default="protes,cem,random", help="Comma-separated solver kinds")
    ap.add_argument("--out_root", default="results", help="Output root directory")
    args = ap.parse_args()

    base_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]

    out_root = Path(args.out_root)
    ensure_dir(out_root)

    runs = []
    for s in seeds:
        for solver_kind in solvers:
            cfg = copy.deepcopy(base_cfg)
            cfg["seed"] = s
            cfg["solver"]["kind"] = solver_kind
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = f"{cfg['benchmark']['kind']}_{solver_kind}_seed{s}_{stamp}"
            out_dir = out_root / run_name
            ensure_dir(out_dir)
            run_experiment(cfg, out_dir)

            hist = pd.read_csv(out_dir / "history.csv")
            best_y = float(hist["best_y"].iloc[-1])
            oracle_calls = int(hist["oracle_calls"].iloc[-1])

            runs.append(
                {
                    "run_name": run_name,
                    "solver": solver_kind,
                    "seed": s,
                    "best_y": best_y,
                    "oracle_calls": oracle_calls,
                    "out_dir": str(out_dir),
                }
            )

    df = pd.DataFrame(runs)
    summary_path = out_root / f"sweep_summary_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(summary_path, index=False)
    print(f"Saved sweep summary: {summary_path}")


if __name__ == "__main__":
    main()
