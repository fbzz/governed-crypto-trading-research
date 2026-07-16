from __future__ import annotations

import numpy as np
import pytest

from tlm.persistent_duration_policy import (
    CASH,
    persistent_horizon_edges,
    stateful_persistent_actions,
    transition_turnover,
)


def test_survival_weighted_per_day_edge_is_exact() -> None:
    per_day = np.asarray([[0.01, -0.02, 0.03]], dtype=np.float64)
    horizons = np.asarray([1.0, 3.0, 7.0])
    gross_location = per_day[:, :, None] * horizons[None, None, :]
    survival = np.asarray([[[1.0, 0.5, 0.25]]])
    observed = persistent_horizon_edges(
        gross_location,
        np.broadcast_to(survival, gross_location.shape),
        horizons=[1, 3, 7],
        horizon_weights=[0.2, 0.3, 0.5],
    )
    expected_multiplier = 0.2 + 0.3 * 0.5 + 0.5 * 0.25
    assert np.allclose(observed, per_day * expected_multiplier)


def test_frozen_stateful_policy_actions_costs_and_ties_are_exact() -> None:
    edges = np.asarray(
        [
            [0.003, 0.000, 0.000],
            [0.002, 0.004, 0.000],
            [0.001, 0.005, 0.000],
            [0.000, 0.002, 0.004],
            [0.000, -0.004, 0.000],
            [0.001, 0.000, 0.000],
            [0.003, 0.003, 0.000],
            [-0.001, 0.000, 0.000],
        ]
    )
    result = stateful_persistent_actions(edges, base_cost=0.001)
    assert result["actions"] == [
        "enter",
        "hold",
        "switch",
        "hold",
        "exit",
        "cash",
        "enter",
        "hold",
    ]
    assert result["selected_assets"] == [0, 0, 1, 1, CASH, CASH, 0, 0]
    assert result["turnover"] == [1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    assert result["final_liquidation_turnover"] == 1.0
    assert result["total_turnover"] == 6.0
    assert result["total_transaction_cost"] == pytest.approx(0.006)


def test_turnover_contract_and_validation_are_bounded() -> None:
    assert transition_turnover(CASH, 0, asset_count=3, risky_gross=1.0) == 1.0
    assert transition_turnover(0, CASH, asset_count=3, risky_gross=1.0) == 1.0
    assert transition_turnover(0, 1, asset_count=3, risky_gross=1.0) == 2.0
    assert transition_turnover(1, 1, asset_count=3, risky_gross=1.0) == 0.0
    with pytest.raises(ValueError, match="sum to one"):
        persistent_horizon_edges(
            np.zeros((1, 3, 3)),
            np.ones((1, 3, 3)),
            horizons=[1, 3, 7],
            horizon_weights=[0.2, 0.3, 0.4],
        )
    with pytest.raises(ValueError, match="outside the action space"):
        transition_turnover(4, 0, asset_count=3, risky_gross=1.0)
