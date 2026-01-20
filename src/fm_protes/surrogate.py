from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .constraints import Constraint, batch_violation
from .qubo import qubo_energy


@dataclass
class SurrogateObjective:
    """Surrogate energy used by the solver.

    E(x) = QUBO(x) + rho * violation(x) - alpha * log(p_feasible(x)+eps)

    All terms are optional except QUBO.
    """

    Q: np.ndarray
    const: float
    constraint: Optional[Constraint] = None
    rho: float = 0.0
    p_feasible: Optional[callable] = None  # function X -> prob in (0,1)
    alpha: float = 0.0
    eps: float = 1e-6

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.int8)
        e = qubo_energy(X, self.Q, self.const)

        if self.constraint is not None and self.rho != 0.0:
            v = batch_violation(X, self.constraint)
            e = e + float(self.rho) * v

        if self.p_feasible is not None and self.alpha != 0.0:
            p = np.clip(self.p_feasible(X), self.eps, 1.0 - self.eps)
            e = e - float(self.alpha) * np.log(p)

        return e.astype(np.float64)
