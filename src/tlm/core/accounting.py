from __future__ import annotations

import math

import numpy as np


def persistent_portfolio_returns(
    positions: np.ndarray,
    actual_log_returns: np.ndarray,
    cost_bps: float,
) -> dict[str, np.ndarray | float]:
    weights = np.asarray(positions, dtype=np.float64)
    returns = np.asarray(actual_log_returns, dtype=np.float64)
    if weights.shape != returns.shape or weights.ndim != 2:
        raise ValueError("Positions and returns must share [days,assets] shape")
    if not np.isfinite(weights).all() or not np.isfinite(returns).all():
        raise ValueError("Positions and returns must be finite")
    if (weights < 0).any() or (weights.sum(axis=1) > 1.0 + 1e-12).any():
        raise ValueError("Positions violate long/cash gross-exposure contract")
    if not math.isfinite(cost_bps) or cost_bps < 0:
        raise ValueError("Cost must be finite and nonnegative")
    prior = np.vstack([np.zeros((1, weights.shape[1])), weights[:-1]])
    turnover = np.abs(weights - prior).sum(axis=1)
    if len(turnover):
        turnover[-1] += float(np.abs(weights[-1]).sum())
    gross = (weights * np.expm1(returns)).sum(axis=1)
    cost = turnover * (float(cost_bps) / 10_000.0)
    return {
        "gross_return": gross,
        "turnover": turnover,
        "cost": cost,
        "net_return": gross - cost,
        "total_turnover": float(turnover.sum()),
        "total_cost": float(cost.sum()),
    }

