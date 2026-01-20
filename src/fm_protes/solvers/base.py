from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np


ObjectiveFn = Callable[[np.ndarray], np.ndarray]


@dataclass
class SolverResult:
    X_pool: np.ndarray          # evaluated candidates (may include infeasible)
    y_pool: np.ndarray          # surrogate energies for X_pool
    X_best: np.ndarray          # best candidate (surrogate)
    y_best: float
    info: Dict[str, float]


class Solver:
    name: str = "base"

    def solve(
        self,
        objective: ObjectiveFn,
        d: int,
        budget: int,
        pool_size: int,
        seed: int,
    ) -> SolverResult:
        raise NotImplementedError
