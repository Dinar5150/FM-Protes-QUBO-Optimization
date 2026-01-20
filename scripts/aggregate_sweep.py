#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def _load_one_history(out_dir: str | Path) -> pd.DataFrame:
    out_dir = Path(out_dir)
    hist_path = out_dir / "history.csv"
    if not hist_path.exists():
        raise FileNotFoundError(f"Missing history.csv at: {hist_path}")
    hist = pd.read_csv(hist_path)
    # required columns in this repo
    need = {"oracle_calls", "best_y"}
    missing = need - set(hist.columns)
    if missing:
        raise ValueError(f"{hist_path} missing columns: {sorted(missing)}")
    return hist[["oracle_calls", "best_y"]].copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="Path to sweep_summary_*.csv")
    ap.add_argument("--solver", default=None, help="Filter by solver kind (optional)")
    ap.add_argument("--out", default=None, help="Output CSV path (default: alongside summary)")
    ap.add_argument("--plot", action="store_true", help="Also save a PNG plot (median + IQR)")
    args = ap.parse_args()

    summary_path = Path(args.summary)
    df = pd.read_csv(summary_path)

    # expected columns from scripts/run_sweep.py
    need = {"out_dir", "seed", "solver"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{summary_path} missing columns: {sorted(missing)}")

    if args.solver is not None:
        df = df[df["solver"].astype(str) == str(args.solver)].copy()

    if df.empty:
        raise ValueError("No runs matched (check --solver filter / summary file).")

    rows = []
    for _, r in df.iterrows():
        hist = _load_one_history(r["out_dir"])
        hist["seed"] = int(r["seed"])
        hist["solver"] = str(r["solver"])
        rows.append(hist)

    all_hist = pd.concat(rows, axis=0, ignore_index=True)

    # aggregate per solver (even if only one)
    agg = (
        all_hist.groupby(["solver", "oracle_calls"])["best_y"]
        .agg(
            median="median",
            p25=lambda s: s.quantile(0.25),
            p75=lambda s: s.quantile(0.75),
            n="count",
        )
        .reset_index()
        .sort_values(["solver", "oracle_calls"])
    )

    out_csv = Path(args.out) if args.out else summary_path.with_suffix("").with_name(summary_path.stem + "_agg.csv")
    agg.to_csv(out_csv, index=False)
    print(f"saved: {out_csv}")

    if args.plot:
        # one plot per solver (simple, avoids clutter)
        for solver_kind, g in agg.groupby("solver"):
            x = g["oracle_calls"].to_numpy()
            med = g["median"].to_numpy()
            p25 = g["p25"].to_numpy()
            p75 = g["p75"].to_numpy()

            plt.figure()
            plt.plot(x, med, label=f"{solver_kind} median")
            plt.fill_between(x, p25, p75, alpha=0.2, label="IQR (p25..p75)")
            plt.xlabel("oracle_calls")
            plt.ylabel("best_y (lower is better)")
            plt.title(f"Best feasible objective vs oracle calls ({solver_kind})")
            plt.grid(True)
            plt.legend()

            out_png = out_csv.with_suffix("").with_name(out_csv.stem + f"_{solver_kind}_plot.png")
            plt.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"saved: {out_png}")


if __name__ == "__main__":
    main()
