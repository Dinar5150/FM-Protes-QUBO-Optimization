#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True, help="Path to history.csv")
    ap.add_argument("--x", default="oracle_calls", choices=["iter", "oracle_calls"], help="x-axis")
    args = ap.parse_args()

    hist = pd.read_csv(args.history)

    x = hist[args.x].values
    y = hist["best_y"].values

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel(args.x)
    plt.ylabel("best_y (lower is better)")
    plt.title("Best feasible objective over time")
    plt.grid(True)
    out = Path(args.history).with_suffix("").as_posix() + f"_plot_{args.x}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved plot to: {out}")


if __name__ == "__main__":
    main()
