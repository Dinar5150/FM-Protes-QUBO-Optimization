from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Type

import numpy as np


class Constraint:
    """Base class for constraints."""

    def is_feasible(self, x: np.ndarray) -> bool:
        raise NotImplementedError

    def violation(self, x: np.ndarray) -> float:
        """Nonnegative violation magnitude; 0 if feasible."""
        raise NotImplementedError


@dataclass
class CardinalityConstraint(Constraint):
    """Enforce sum(x) == K."""

    K: int

    def is_feasible(self, x: np.ndarray) -> bool:
        return int(np.sum(x)) == int(self.K)

    def violation(self, x: np.ndarray) -> float:
        s = float(np.sum(x))
        return (s - float(self.K)) ** 2


@dataclass
class LinearInequalityConstraint(Constraint):
    """Enforce A x <= b (componentwise)."""

    A: np.ndarray  # (m, d)
    b: np.ndarray  # (m,)

    def __post_init__(self):
        self.A = np.asarray(self.A, dtype=np.float64)
        self.b = np.asarray(self.b, dtype=np.float64)

    def is_feasible(self, x: np.ndarray) -> bool:
        Ax = self.A @ x.astype(np.float64)
        return bool(np.all(Ax <= self.b + 1e-12))

    def violation(self, x: np.ndarray) -> float:
        Ax = self.A @ x.astype(np.float64)
        v = np.maximum(0.0, Ax - self.b)**2
        # guide-consistent hinge penalty (nonnegative; 0 if feasible)
        return float(np.sum(v))


@dataclass
class CompositeConstraint(Constraint):
    """Logical AND of multiple constraints."""

    constraints: List[Constraint]

    def is_feasible(self, x: np.ndarray) -> bool:
        return all(c.is_feasible(x) for c in self.constraints)

    def violation(self, x: np.ndarray) -> float:
        return float(sum(c.violation(x) for c in self.constraints))


@dataclass
class OneHotGroupsConstraint(Constraint):
    """Enforce one-hot in each group: for every G, sum_{i in G} x_i == 1."""

    groups: List[np.ndarray]  # each group is a 1D array of indices

    def __post_init__(self):
        norm_groups: List[np.ndarray] = []
        for g in self.groups:
            gi = np.asarray(g, dtype=np.int64).reshape(-1)
            if gi.size == 0:
                raise ValueError("OneHotGroupsConstraint: empty group is not allowed")
            norm_groups.append(gi)
        self.groups = norm_groups

    def is_feasible(self, x: np.ndarray) -> bool:
        x = np.asarray(x, dtype=np.int8)
        for g in self.groups:
            if int(np.sum(x[g])) != 1:
                return False
        return True

    def violation(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.int8)
        v = 0.0
        for g in self.groups:
            s = float(np.sum(x[g]))
            v += (s - 1.0) ** 2
        return float(v)


def batch_is_feasible(X: np.ndarray, cons: Optional[Constraint]) -> np.ndarray:
    if cons is None:
        return np.ones((len(X),), dtype=bool)
    return np.array([cons.is_feasible(x) for x in X], dtype=bool)


def batch_violation(X: np.ndarray, cons: Optional[Constraint]) -> np.ndarray:
    if cons is None:
        return np.zeros((len(X),), dtype=np.float64)
    return np.array([cons.violation(x) for x in X], dtype=np.float64)


def flatten_constraints(cons: Optional[Constraint]) -> List[Constraint]:
    """Flatten CompositeConstraint into a plain list.

    This is useful for hybrid constraint handling (hard-mask some constraints,
    penalize others).
    """
    if cons is None:
        return []
    if isinstance(cons, CompositeConstraint):
        out: List[Constraint] = []
        for c in cons.constraints:
            out.extend(flatten_constraints(c))
        return out
    return [cons]


def make_constraint(constraints: Sequence[Constraint]) -> Optional[Constraint]:
    cs = [c for c in constraints if c is not None]
    if len(cs) == 0:
        return None
    if len(cs) == 1:
        return cs[0]
    return CompositeConstraint(list(cs))


def split_constraints(
    cons: Optional[Constraint],
    *,
    hard_types: Sequence[Type[Constraint]],
) -> Tuple[Optional[Constraint], Optional[Constraint]]:
    """Split a (possibly composite) constraint into hard + soft parts.

    Hard constraints are those whose type is in hard_types.
    """
    hard_t = tuple(hard_types)
    hard_list: List[Constraint] = []
    soft_list: List[Constraint] = []
    for c in flatten_constraints(cons):
        if hard_t and isinstance(c, hard_t):
            hard_list.append(c)
        else:
            soft_list.append(c)
    return make_constraint(hard_list), make_constraint(soft_list)
