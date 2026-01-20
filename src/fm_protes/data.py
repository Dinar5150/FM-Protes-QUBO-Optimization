from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DataBuffer:
    """Stores feasible (X,y) for regression and (X,label) for feasibility classification."""

    X_reg: List[np.ndarray] = field(default_factory=list)
    y_reg: List[float] = field(default_factory=list)

    X_clf: List[np.ndarray] = field(default_factory=list)
    y_clf: List[int] = field(default_factory=list)  # 1 feasible, 0 infeasible

    def add_reg(self, x: np.ndarray, y: float) -> None:
        self.X_reg.append(np.asarray(x, dtype=np.int8))
        self.y_reg.append(float(y))

    def add_clf(self, x: np.ndarray, label: int) -> None:
        self.X_clf.append(np.asarray(x, dtype=np.int8))
        self.y_clf.append(int(label))

    def add_point(self, x: np.ndarray, feasible: bool, y: Optional[float] = None) -> None:
        self.add_clf(x, 1 if feasible else 0)
        if feasible:
            if y is None:
                raise ValueError("Feasible point requires y")
            self.add_reg(x, float(y))

    def get_reg_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.X_reg:
            return np.zeros((0, 0), dtype=np.int8), np.zeros((0,), dtype=float)
        X = np.stack(self.X_reg, axis=0)
        y = np.asarray(self.y_reg, dtype=float)
        return X, y

    def get_clf_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.X_clf:
            return np.zeros((0, 0), dtype=np.int8), np.zeros((0,), dtype=np.int64)
        X = np.stack(self.X_clf, axis=0)
        y = np.asarray(self.y_clf, dtype=np.int64)
        return X, y

    def size_reg(self) -> int:
        return len(self.y_reg)

    def size_clf(self) -> int:
        return len(self.y_clf)

    def summary(self) -> Dict[str, int]:
        return {"n_reg": self.size_reg(), "n_clf": self.size_clf()}
