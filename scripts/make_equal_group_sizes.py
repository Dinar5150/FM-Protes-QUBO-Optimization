#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import List

import yaml


def _equal_group_sizes(num_groups: int, target_d: int) -> List[int]:
    if num_groups <= 0:
        raise ValueError("num_groups must be > 0")
    if target_d <= 0:
        raise ValueError("target_d must be > 0")

    base = target_d // num_groups
    rem = target_d % num_groups
    sizes = [base] * num_groups
    for i in range(rem):
        sizes[i] += 1
    return sizes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/onehot_qubo.yaml", help="Base YAML config")
    ap.add_argument("--sizes", default="200,400,600,800,1000", help="Comma-separated d sizes")
    ap.add_argument("--out_dir", default="configs/generated_equal_groups", help="Output directory for generated configs")
    args = ap.parse_args()

    base_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base_groups = base_cfg["benchmark"].get("group_sizes", [])
    if not base_groups:
        raise ValueError("benchmark.group_sizes is required in config")

    num_groups = len(base_groups)
    sizes = [int(s.strip()) for s in args.sizes.split(",") if s.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for d in sizes:
        cfg = copy.deepcopy(base_cfg)
        eq_sizes = _equal_group_sizes(num_groups, d)
        cfg["benchmark"]["group_sizes"] = eq_sizes
        cfg["run_name"] = f"onehot_qubo_equalgroups_d{d}"

        out_path = out_dir / f"onehot_qubo_equalgroups_d{d}.yaml"
        out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

        print(f"d={d} -> group_sizes={eq_sizes} (saved {out_path})")


if __name__ == "__main__":
    main()
