from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Optional, Tuple

import numpy as np

from ..constraints import CardinalityConstraint, Constraint
from src.fm_protes.QuboMaker import DynamicMatrix,add_equality,make_symetric

@dataclass
class MaxCutCardinalityBenchmark:
    """Constrained MaxCut variant: choose exactly K vertices in a subset S.

    Decision x in {0,1}^d indicates membership in S.
    Constraint: sum(x) == K.

    Objective (minimize): negative cut weight
        f(x) = - sum_{i<j} w_ij * [x_i != x_j]
    """

    d: int
    K: int
    seed: int = 0
    weight_scale: float = 1000.0

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        W = rng.random((self.d, self.d)) * self.weight_scale # wrong scaling can force wrong answers (you may want to scale to 1000)
        W = np.triu(W, 1)
        W = W + W.T
        np.fill_diagonal(W, 0.0)
        self.W = W
        self.name = f"maxcut_cardinality_d{self.d}_K{self.K}_seed{self.seed}"

    def n_vars(self) -> int:
        return self.d

    def constraint(self) -> Optional[Constraint]:
        return CardinalityConstraint(K=self.K)

    def cut_weight(self, x: np.ndarray) -> float:
        x = x.astype(np.int8)
        # Compute sum_{i<j} w_ij * (x_i xor x_j)
        # xor = x_i + x_j - 2 x_i x_j for binary
        # Use vectorized formula:
        # For upper triangle: sum w_ij * (x_i + x_j - 2 x_i x_j)
        iu = np.triu_indices(self.d, 1)
        xi = x[iu[0]].astype(np.float64)
        xj = x[iu[1]].astype(np.float64)
        xor = xi + xj - 2.0 * xi * xj
        return float(np.sum(self.W[iu] * xor))

    def oracle(self, x: np.ndarray) -> float:
        # minimize negative cut weight (maximize cut)
        return -self.cut_weight(x)

    def sample_feasible(self, rng: np.random.Generator, n: int) -> np.ndarray:
        X = np.zeros((n, self.d), dtype=np.int8)
        for i in range(n):
            idx = rng.choice(self.d, size=self.K, replace=False)
            X[i, idx] = 1
        return X

    def info(self) -> Dict:
        return {"name": self.name, "d": self.d, "K": self.K, "seed": self.seed, "weight_scale": self.weight_scale}

    def brute_force_optimum(self, max_d: int = 24) -> Optional[Tuple[np.ndarray, float]]:
        """Compute exact optimum by enumerating all C(d,K) subsets if small."""
        if self.d > max_d:
            return None
        best_x = None
        best_y = float("inf")
        for comb in combinations(range(self.d), self.K):
            x = np.zeros(self.d, dtype=np.int8)
            x[list(comb)] = 1
            y = self.oracle(x)
            if y < best_y:
                best_y = y
                best_x = x
        if best_x is None:
            return None
        return best_x, float(best_y)
    
    #for testing
    def print_results(self,x):
        print(x)
        W=0
        cnt=int(0)
        for i in range(self.d):
            cnt+=int(x[i])
            for j in range(i+1,self.d):
                if x[i]==x[j]:
                    continue
                W+=self.W[i][j]
        print('Weight:',W,'Vertices Count:',cnt)
    def get_qubo(self):
        n=self.d
        W=self.W
        K=self.K
        Q=DynamicMatrix(n)
        
        add_equality(Q,np.ones(n),[i for i in range(n)],K,1e5)
        for i in range(n):
            for j in range(i+1,n):
                #-(x(i)*(1-x(j))+x(j)*(1-x(i)))*W[i][j]
                Q[i,j]+=2*W[i][j]
                Q[i,i]-=W[i][j]
                Q[j,j]-=W[i][j]
        make_symetric(Q)
        return Q.get_qubo()
