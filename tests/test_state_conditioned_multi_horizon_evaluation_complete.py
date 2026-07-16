from __future__ import annotations

from pathlib import Path

import numpy as np

from tlm.state_conditioned_multi_horizon_evaluation_artifacts import (
    load_json,
    verify_self_hash,
)
from tlm.state_conditioned_multi_horizon_evaluation_complete import (
    _bootstrap_cell,
    _bootstrap_seed,
    _economic_metrics,
    _request_frame,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v59_registered_outcome_request_is_exact_and_target_free() -> None:
    request = load_json(
        ROOT
        / "artifacts/v59_state_conditioned_multi_horizon_evaluation/outcome_request.json"
    )
    verify_self_hash(request, "outcome_request_sha256", "request")
    frame = _request_frame(request)
    assert len(frame) == 20_410
    assert not frame.duplicated().any()
    assert not {"BTCUSDT", "ETHUSDT", "SOLUSDT"}.intersection(frame["symbol"])


def test_v59_economic_metrics_use_compounded_wealth_and_positive_drawdown() -> None:
    returns = np.array([0.10, -0.05, 0.02], dtype=np.float64)
    turnover = np.array([1 / 3, 0.0, 1 / 3], dtype=np.float64)
    metrics = _economic_metrics(returns, turnover)
    assert np.isclose(metrics["cumulative_return"], np.prod(1 + returns) - 1)
    assert metrics["maximum_drawdown"] > 0
    assert np.isclose(metrics["total_turnover"], 2 / 3)
    assert metrics["annualized_turnover"] > 0


def test_v59_bootstrap_is_paired_deterministic_and_preserves_controls() -> None:
    candidate = np.array([0.02, 0.01, -0.005, 0.015], dtype=np.float64)
    returns = {
        "candidate": candidate,
        "cash": np.zeros(4),
        "weekly_dual_momentum_30": candidate - 0.002,
        "weekly_equal_weight_total_gross_one_third": candidate - 0.001,
        "shared_linear_h7_q50_with_train_residual_q20": candidate - 0.003,
    }
    seed = _bootstrap_seed("origin_2024", "expanding", 1, 10, 7)
    first = _bootstrap_cell(returns, block=2, seed=seed, paths=100)
    second = _bootstrap_cell(returns, block=2, seed=seed, paths=100)
    assert first == second
    assert first["paths"] == 100
    assert set(first["distributions"]) == set(returns)
    assert set(first["candidate_minus_controls"]) == set(returns) - {"candidate"}
    assert all(
        row["p05"] > 0 for row in first["candidate_minus_controls"].values()
    )
