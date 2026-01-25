from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from ..constraints import Constraint, CardinalityConstraint

@dataclass
class PortfolioBenchmark:
    """Portfolio optimization: Select K assets maximizing return, minimizing risk with diversification."""
    
    d: int = 50
    K: int = 10
    seed: int = 0
    return_scale: float = 0.1
    risk_scale: float = 0.3
    diversity_weight: float = 0.1
    risk_weight: float = 5.0

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        
        self.returns = rng.normal(loc=0.05, scale=self.return_scale, size=self.d)
        
        self.risks = rng.uniform(low=0.1, high=self.risk_scale, size=self.d)
        
        self.correlation = self._generate_correlation_matrix(rng, self.d)
        self.covariance = np.outer(self.risks, self.risks) * self.correlation
        
        self.name = f"portfolio_d{self.d}_K{self.K}_seed{self.seed}"

    def _generate_correlation_matrix(self, rng: np.random.Generator, size: int) -> np.ndarray:
        
        correlation = rng.uniform(low=-0.5, high=0.8, size=(size, size))
        correlation = (correlation + correlation.T) / 2
        np.fill_diagonal(correlation, 1.0)
        
        return correlation

    def n_vars(self) -> int:
        return self.d

    def constraint(self) -> Optional[Constraint]:
        return CardinalityConstraint(K=self.K)

    def portfolio_return(self, x: np.ndarray) -> float:
        return float(np.dot(self.returns, x.astype(np.float64)))

    def portfolio_risk(self, x: np.ndarray) -> float:
        x_float = x.astype(np.float64)
        return float(x_float.T @ self.covariance @ x_float)

    def portfolio_diversification(self, x: np.ndarray) -> float:
        n_selected = np.sum(x)
        if n_selected <= 1:
            return 0.0
        #simple 1-1/n rule as we are putting same amount of money in each asset
        return float(1.0 - 1.0 / n_selected)

    def oracle(self, x: np.ndarray) -> float:
        
        x = np.asarray(x, dtype=np.int8)
        
        ret = self.portfolio_return(x)
        risk = self.portfolio_risk(x)
        div = self.portfolio_diversification(x)
        return -ret + self.risk_weight * risk - self.diversity_weight * div
    def select_k_from_n(n: int, k: int,rng: np.random.Generator) -> list[int]:
        if k > n:
            raise ValueError(f"Cannot select k={k} items from n={n}")
        if k < 0:
            raise ValueError(f"k must be non-negative, got {k}")

        selected = []
        remaining_slots = k

        for i in range(n):
            prob_select = remaining_slots / (n - i)
            if rng.random() < prob_select:
                selected.append(i)
                remaining_slots -= 1

                if remaining_slots == 0:
                    break
        return selected
    def sample_feasible(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Generate n feasible portfolios (exactly K assets selected)."""
        if self.K > self.d:
            raise ValueError(f"Cannot select K={self.K} assets from d={self.d}")

        X = np.zeros((int(n), self.d), dtype=np.int8)
        for i in range(int(n)):
            indices = self.select_k_from_n(self.d,self.K)
            X[i, indices] = 1
        return X
    def info(self) -> Dict:
        return {
            "name": self.name,
            "d": self.d,
            "K": self.K,
            "seed": self.seed,
            "avg_return": float(np.mean(self.returns)),
            "avg_risk": float(np.mean(self.risks)),
            "return_scale": self.return_scale,
            "risk_scale": self.risk_scale,
            "diversity_weight": self.diversity_weight,
            "risk_weight": self.risk_weight,
        }


# Add to build_benchmark function in loop.py
"""
if kind == "portfolio":
    return PortfolioBenchmark(
        d=int(cfg.get("d", 50)),
        K=int(cfg.get("K", 10)),
        seed=int(cfg.get("seed", 0)),
        return_scale=float(cfg.get("return_scale", 0.1)),
        risk_scale=float(cfg.get("risk_scale", 0.3)),
        diversity_weight=float(cfg.get("diversity_weight", 0.1)),
        risk_weight=float(cfg.get("risk_weight", 5.0)),
    )
"""

# Example config for portfolio benchmark
"""
benchmark:
  kind: "portfolio"
  d: 50
  K: 10
  seed: 0
  return_scale: 0.1
  risk_scale: 0.3
  diversity_weight: 0.1
  risk_weight: 5.0
"""