from __future__ import annotations

import math
from typing import Any

import numpy as np


CASH = -1


def persistent_horizon_edges(
    gross_location: np.ndarray,
    survival_probability: np.ndarray,
    *,
    horizons: list[int] | tuple[int, ...],
    horizon_weights: list[float] | tuple[float, ...],
) -> np.ndarray:
    """Convert cumulative forecasts into survival-weighted per-day edges."""

    location = np.asarray(gross_location, dtype=np.float64)
    survival = np.asarray(survival_probability, dtype=np.float64)
    horizon = np.asarray(horizons, dtype=np.float64)
    weights = np.asarray(horizon_weights, dtype=np.float64)
    if location.ndim != 3 or location.shape != survival.shape:
        raise ValueError("Location and survival must share [days,assets,horizons]")
    if horizon.ndim != 1 or weights.shape != horizon.shape:
        raise ValueError("Horizons and weights must be matching vectors")
    if location.shape[-1] != len(horizon):
        raise ValueError("Forecast horizon width does not match the policy")
    if not np.isfinite(location).all() or not np.isfinite(survival).all():
        raise ValueError("Policy forecasts must be finite")
    if np.any(horizon <= 0.0) or np.any(weights < 0.0):
        raise ValueError("Policy horizons must be positive and weights nonnegative")
    if not math.isclose(float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Horizon weights must sum to one")
    if np.any(survival < 0.0) or np.any(survival > 1.0):
        raise ValueError("Survival probabilities must lie in [0,1]")
    return np.sum(weights * survival * (location / horizon), axis=-1)


def transition_turnover(
    incumbent: int,
    candidate: int,
    *,
    asset_count: int,
    risky_gross: float,
) -> float:
    if asset_count < 1:
        raise ValueError("At least one asset is required")
    if incumbent not in {CASH, *range(asset_count)}:
        raise ValueError("Incumbent action is outside the action space")
    if candidate not in {CASH, *range(asset_count)}:
        raise ValueError("Candidate action is outside the action space")
    if not math.isfinite(risky_gross) or risky_gross <= 0.0:
        raise ValueError("Risky gross must be finite and positive")
    if incumbent == candidate:
        return 0.0
    if incumbent == CASH or candidate == CASH:
        return float(risky_gross)
    return 2.0 * float(risky_gross)


def stateful_persistent_actions(
    persistent_edges: np.ndarray,
    *,
    base_cost: float,
    risky_gross: float = 1.0,
    initial_action: int = CASH,
    tie_tolerance: float = 1e-12,
    final_liquidation: bool = True,
) -> dict[str, Any]:
    """Apply the frozen incumbent-first, cash-second, lexical policy."""

    edges = np.asarray(persistent_edges, dtype=np.float64)
    if edges.ndim != 2 or edges.shape[1] < 1:
        raise ValueError("Persistent edges must have [days,assets] shape")
    if not np.isfinite(edges).all():
        raise ValueError("Persistent edges must be finite")
    if not math.isfinite(base_cost) or base_cost < 0.0:
        raise ValueError("Base cost must be finite and nonnegative")
    if not math.isfinite(tie_tolerance) or tie_tolerance < 0.0:
        raise ValueError("Tie tolerance must be finite and nonnegative")
    asset_count = edges.shape[1]
    if initial_action not in {CASH, *range(asset_count)}:
        raise ValueError("Initial action is outside the action space")

    incumbent = int(initial_action)
    positions = np.zeros((len(edges), asset_count), dtype=np.float64)
    actions: list[str] = []
    selected: list[int] = []
    turnover: list[float] = []
    transaction_costs: list[float] = []
    incumbent_utilities: list[float] = []
    selected_utilities: list[float] = []

    for day, edge in enumerate(edges):
        candidates = [CASH, *range(asset_count)]
        utilities: dict[int, float] = {}
        for candidate in candidates:
            candidate_edge = 0.0 if candidate == CASH else float(edge[candidate])
            candidate_turnover = transition_turnover(
                incumbent,
                candidate,
                asset_count=asset_count,
                risky_gross=risky_gross,
            )
            utilities[candidate] = candidate_edge - base_cost * candidate_turnover

        incumbent_utility = utilities[incumbent]
        maximum_utility = max(utilities.values())
        priority = [incumbent, CASH, *range(asset_count)]
        unique_priority = list(dict.fromkeys(priority))
        best = next(
            candidate
            for candidate in unique_priority
            if math.isclose(
                utilities[candidate],
                maximum_utility,
                rel_tol=0.0,
                abs_tol=tie_tolerance,
            )
        )
        if best != incumbent and not (
            utilities[best] > incumbent_utility + tie_tolerance
        ):
            best = incumbent

        day_turnover = transition_turnover(
            incumbent,
            best,
            asset_count=asset_count,
            risky_gross=risky_gross,
        )
        if incumbent == CASH and best == CASH:
            action = "cash"
        elif incumbent == CASH:
            action = "enter"
        elif best == CASH:
            action = "exit"
        elif incumbent == best:
            action = "hold"
        else:
            action = "switch"

        incumbent_utilities.append(float(incumbent_utility))
        selected_utilities.append(float(utilities[best]))
        actions.append(action)
        turnover.append(day_turnover)
        transaction_costs.append(base_cost * day_turnover)
        incumbent = int(best)
        selected.append(incumbent)
        if incumbent != CASH:
            positions[day, incumbent] = risky_gross

    liquidation_turnover = (
        float(risky_gross) if final_liquidation and incumbent != CASH else 0.0
    )
    liquidation_cost = base_cost * liquidation_turnover
    return {
        "positions": positions,
        "actions": actions,
        "selected_assets": selected,
        "turnover": turnover,
        "transaction_costs": transaction_costs,
        "incumbent_utilities": incumbent_utilities,
        "selected_utilities": selected_utilities,
        "final_action": incumbent,
        "final_liquidation_turnover": liquidation_turnover,
        "final_liquidation_cost": liquidation_cost,
        "total_turnover": float(sum(turnover) + liquidation_turnover),
        "total_transaction_cost": float(sum(transaction_costs) + liquidation_cost),
    }
