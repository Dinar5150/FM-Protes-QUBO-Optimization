from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn


class FactorizationMachine(nn.Module):
    """Second-order Factorization Machine for binary/categorical (one-hot) features.

    Prediction:
        y = w0 + sum_i w_i x_i + sum_{i<j} <v_i, v_j> x_i x_j

    This is naturally a QUBO when x is binary.
    """

    def __init__(self, d: int, k: int):
        super().__init__()
        self.d = int(d)
        self.k = int(k)

        self.w0 = nn.Parameter(torch.zeros(1))
        self.w = nn.Parameter(torch.zeros(d))
        self.V = nn.Parameter(torch.randn(d, k) * 0.01)

        # Optional output scaling for regression:
        self.y_mean = 0.0
        self.y_std = 1.0

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, d), float32
        linear = self.w0 + X @ self.w
        # FM trick:
        XV = X @ self.V                # (B, k)
        XV2 = XV * XV                  # (B, k)
        X2V2 = (X * X) @ (self.V * self.V)  # (B, k)
        interactions = 0.5 * torch.sum(XV2 - X2V2, dim=1)  # (B,)
        return linear + interactions


@dataclass
class FMTrainConfig:
    k: int = 16
    lr: float = 2e-2
    weight_decay: float = 1e-4
    epochs: int = 200
    batch_size: int = 256
    patience: int = 20
    device: str = "cpu"


def _batch_iter(X: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator):
    n = len(X)
    idx = np.arange(n)
    rng.shuffle(idx)
    for s in range(0, n, batch_size):
        j = idx[s : s + batch_size]
        yield X[j], y[j]


def train_fm_regression(
    X: np.ndarray,
    y: np.ndarray,
    cfg: FMTrainConfig,
    seed: int = 0,
) -> Tuple[FactorizationMachine, Dict[str, float]]:
    """Train FM with MSE loss. Returns trained model and final losses."""
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if len(X) != len(y):
        raise ValueError("X and y length mismatch")
    if len(X) < 2:
        raise ValueError("Need at least 2 training points for regression")

    rng = np.random.default_rng(seed)

    d = X.shape[1]
    model = FactorizationMachine(d=d, k=cfg.k)
    device = torch.device(cfg.device)
    model.to(device)

    # Standardize targets for stability:
    y_mean = float(np.mean(y))
    y_std = float(np.std(y) + 1e-8)
    model.y_mean = y_mean
    model.y_std = y_std
    y_s = (y - y_mean) / y_std

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_s, dtype=torch.float32, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    bad = 0

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in _batch_iter(X, y_s, cfg.batch_size, rng):
            xb_t = torch.tensor(xb, dtype=torch.float32, device=device)
            yb_t = torch.tensor(yb, dtype=torch.float32, device=device)

            opt.zero_grad(set_to_none=True)
            pred = model(xb_t)
            loss = loss_fn(pred, yb_t)
            loss.backward()
            opt.step()

            epoch_loss += float(loss.item()) * len(xb)

        epoch_loss /= len(X)

        # Simple early stopping on training loss (template-friendly)
        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"train_mse": best_loss}


def train_fm_classifier(
    X: np.ndarray,
    y01: np.ndarray,
    cfg: FMTrainConfig,
    seed: int = 0,
) -> Tuple[FactorizationMachine, Dict[str, float]]:
    """Train FM as a binary classifier with BCEWithLogits loss."""
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if len(X) != len(y01):
        raise ValueError("X and y length mismatch")
    if len(X) < 10:
        raise ValueError("Need at least 10 points for classifier (template default)")

    rng = np.random.default_rng(seed)

    d = X.shape[1]
    model = FactorizationMachine(d=d, k=cfg.k)
    device = torch.device(cfg.device)
    model.to(device)

    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y01.astype(np.float32), dtype=torch.float32, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    best_loss = float("inf")
    best_state = None
    bad = 0

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in _batch_iter(X, y01.astype(np.float32), cfg.batch_size, rng):
            xb_t = torch.tensor(xb, dtype=torch.float32, device=device)
            yb_t = torch.tensor(yb, dtype=torch.float32, device=device)

            opt.zero_grad(set_to_none=True)
            logits = model(xb_t)
            loss = loss_fn(logits, yb_t)
            loss.backward()
            opt.step()

            epoch_loss += float(loss.item()) * len(xb)

        epoch_loss /= len(X)

        if epoch_loss < best_loss - 1e-6:
            best_loss = epoch_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, {"train_bce": best_loss}


def fm_predict_reg(model: FactorizationMachine, X: np.ndarray) -> np.ndarray:
    """Predict regression output in the original y scale."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X, dtype=torch.float32, device=device)
        y_s = model(xt).detach().cpu().numpy()
    return model.y_mean + model.y_std * y_s


def fm_predict_proba(model: FactorizationMachine, X: np.ndarray) -> np.ndarray:
    """Predict feasibility probability (sigmoid of logits)."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(X, dtype=torch.float32, device=device)
        logits = model(xt).detach().cpu().numpy()
    return 1.0 / (1.0 + np.exp(-logits))


def fm_to_qubo(model: FactorizationMachine) -> Tuple[np.ndarray, float]:
    """Convert a trained FM (regression) into an upper-triangular QUBO and constant.

    We output Q such that:
        y_hat(x) = const + sum_{i<=j} Q[i,j] x_i x_j
    for x in {0,1}^d.

    Note: linear terms are placed on the diagonal because x_i^2 = x_i for binary x.
    """
    w0 = float(model.w0.detach().cpu().numpy().reshape(()))
    w = model.w.detach().cpu().numpy().astype(np.float64)
    V = model.V.detach().cpu().numpy().astype(np.float64)

    d = len(w)
    Q = np.zeros((d, d), dtype=np.float64)

    # diagonal = linear terms
    Q[np.arange(d), np.arange(d)] = w

    # off-diagonal = dot(v_i, v_j)
    G = V @ V.T  # (d, d), where G[i,j] = <v_i, v_j>
    for i in range(d):
        for j in range(i + 1, d):
            Q[i, j] = G[i, j]

    # undo y standardization: y = mean + std * y_std
    const = model.y_mean + model.y_std * w0
    Q = model.y_std * Q

    return Q, float(const)
