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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--out", default=None, help="Output directory (default: results/<run_name>)")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

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
