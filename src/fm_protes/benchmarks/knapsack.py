from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ..constraints import Constraint, LinearInequalityConstraint


@dataclass
class KnapsackBenchmark:
    """0-1 knapsack (minimization version).

    Items i=1..d with weights w_i and values v_i.
    Constraint: sum w_i x_i <= capacity.
    Objective (minimize): - sum v_i x_i  (i.e., maximize value)
    """

    d: int
    seed: int = 0
    capacity_ratio: float = 0.35

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        self.weights = rng.integers(low=1, high=50, size=self.d).astype(np.float64)
        self.values = rng.integers(low=1, high=100, size=self.d).astype(np.float64)
        self.capacity = float(np.sum(self.weights) * self.capacity_ratio)
        self.name = f"knapsack_d{self.d}_seed{self.seed}_cap{self.capacity_ratio:.2f}"

    def n_vars(self) -> int:
        return self.d

    def constraint(self) -> Optional[Constraint]:
        A = self.weights.reshape(1, -1)
        b = np.array([self.capacity], dtype=np.float64)
        return LinearInequalityConstraint(A=A, b=b)

    def oracle(self, x: np.ndarray) -> float:
        return -float(np.dot(self.values, x.astype(np.float64)))

    def sample_feasible(self, rng: np.random.Generator, n: int) -> np.ndarray:
        # Simple greedy+random feasible generator
        X = np.zeros((n, self.d), dtype=np.int8)
        for t in range(n):
            idx = rng.permutation(self.d)
            wsum = 0.0
            x = np.zeros(self.d, dtype=np.int8)
            for i in idx:
                if wsum + self.weights[i] <= self.capacity and rng.random() < 0.5:
                    x[i] = 1
                    wsum += self.weights[i]
            X[t] = x
        return X

    def info(self) -> Dict:
        return {"name": self.name, "d": self.d, "seed": self.seed, "capacity": self.capacity, "capacity_ratio": self.capacity_ratio}
