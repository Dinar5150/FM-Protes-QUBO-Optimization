#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import yaml

# Allow running without installing package:
import sys
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from fm_protes.loop import run_experiment
from fm_protes.utils import ensure_dir
from fm_protes.solvers import has_protes, has_sa, has_tabu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=False, help="Path to YAML config")
    ap.add_argument("--out", default=None, help="Output directory (default: results/<run_name>)")
    ap.add_argument("--check_deps", action="store_true", help="Print optional dependency status and exit")

    # solver smoke-test helpers
    ap.add_argument("--solver_kind", default=None, help="Override solver.kind (e.g. sa, tabu, exact_enum)")
    ap.add_argument("--no_clf", action="store_true", help="Disable feasibility_term (required for sa/tabu here)")
    ap.add_argument("--n_iters", type=int, default=None, help="Override n_iters (for multi-point histories)")

    args = ap.parse_args()

    if args.check_deps:
        print(f"has_protes: {has_protes()}")
        print(f"has_sa (dimod+dwave-neal): {has_sa()}")
        print(f"has_tabu (dwave-tabu): {has_tabu()}")
        return

    if not args.config:
        raise SystemExit("error: --config is required unless --check_deps is set")

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    if args.solver_kind is not None:
        cfg.setdefault("solver", {})
        cfg["solver"]["kind"] = str(args.solver_kind).lower()

    if args.no_clf:
        cfg["feasibility_term"] = {"enabled": False, "alpha": 0.0, "min_points": 0}

    if args.n_iters is not None:
        cfg["n_iters"] = int(args.n_iters)

    run_name = cfg.get("run_name")
    if not run_name:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{cfg['benchmark']['kind']}_{stamp}"

    out_dir = Path(args.out) if args.out else Path("results") / run_name
    ensure_dir(out_dir)

    print(f"[run] config={cfg_path} -> out={out_dir}")
    run_experiment(cfg, out_dir)


if __name__ == "__main__":
    main()
