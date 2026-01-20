from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ..constraints import Constraint


class Benchmark:
    """A black-box objective with constraints over binary x."""

    name: str

    def n_vars(self) -> int:
        raise NotImplementedError

    def constraint(self) -> Optional[Constraint]:
        raise NotImplementedError

    def oracle(self, x: np.ndarray) -> float:
        """Objective to minimize. Only meaningful on feasible points."""
        raise NotImplementedError

    def sample_feasible(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Return (n, d) feasible samples."""
        raise NotImplementedError

    def info(self) -> Dict:
        return {"name": getattr(self, "name", self.__class__.__name__), "d": self.n_vars()}
