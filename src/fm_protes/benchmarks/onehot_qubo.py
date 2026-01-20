from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..constraints import Constraint, OneHotGroupsConstraint
from ..qubo import qubo_energy


def _build_groups(group_sizes: Sequence[int]) -> List[np.ndarray]:
    groups: List[np.ndarray] = []
    off = 0
    for s in group_sizes:
        s = int(s)
        if s <= 0:
            raise ValueError("group_sizes must be positive")
        groups.append(np.arange(off, off + s, dtype=np.int64))
        off += s
    return groups


@dataclass
class OneHotQUBOBenchmark:
    """Synthetic constrained QUBO with one-hot groups.

    Variables are partitioned into groups G1..Gm, constraint is sum_{i in Gk} x_i == 1.

    Oracle (minimize): random QUBO energy. Optionally "plant" a feasible solution by
    adding linear biases favoring one chosen index per group.
    """

    group_sizes: Sequence[int]
    seed: int = 0
    interaction_scale: float = 1.0
    planted_bias: float = 2.0  # >0 encourages planted indices (via negative linear terms)

    def __post_init__(self):
        self.groups = _build_groups(self.group_sizes)
        self.d = int(sum(int(s) for s in self.group_sizes))

        rng = np.random.default_rng(self.seed)

        # random upper-triangular Q (diagonal = linear, off-diagonal = pairwise)
        Q = rng.normal(loc=0.0, scale=float(self.interaction_scale), size=(self.d, self.d)).astype(np.float64)
        Q = np.triu(Q, 0)

        # optional planted feasible x*: pick one index per group and bias it lower
        x_star = np.zeros((self.d,), dtype=np.int8)
        for g in self.groups:
            j = int(rng.choice(g))
            x_star[j] = 1
            Q[j, j] += -abs(float(self.planted_bias))  # lower energy when x_j=1

        self.Q = Q
        self.const = 0.0
        self.x_star = x_star
        self.name = f"onehot_qubo_groups{len(self.groups)}_d{self.d}_seed{self.seed}"

    def n_vars(self) -> int:
        return self.d

    def constraint(self) -> Optional[Constraint]:
        return OneHotGroupsConstraint(groups=self.groups)

    def oracle(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.int8).reshape(1, -1)
        return float(qubo_energy(x, self.Q, self.const)[0])

    def sample_feasible(self, rng: np.random.Generator, n: int) -> np.ndarray:
        X = np.zeros((int(n), self.d), dtype=np.int8)
        for t in range(int(n)):
            for g in self.groups:
                j = int(rng.choice(g))
                X[t, j] = 1
        return X

    def info(self) -> Dict:
        return {
            "name": self.name,
            "d": self.d,
            "seed": self.seed,
            "group_sizes": [int(s) for s in self.group_sizes],
            "interaction_scale": float(self.interaction_scale),
            "planted_bias": float(self.planted_bias),
        }
