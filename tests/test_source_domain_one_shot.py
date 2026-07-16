from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from tlm.scientific_harness import FeatureScaler
from tlm.source_domain_one_shot import (
    _fold_inference,
    build_evaluation_spec,
    build_policy_positions,
    evaluate_registered_gates,
    performance_metrics,
)


class _FixtureModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        shape = (x.shape[0], x.shape[2])
        return {
            "return_q10": torch.full(shape, -0.01, device=x.device),
            "return_q50": torch.full(shape, 0.01, device=x.device),
            "return_q90": torch.full(shape, 0.02, device=x.device),
            "volatility_7d": torch.full(
                shape, float(np.log(0.10)), device=x.device
            ),
        }


def test_policy_positions_respect_availability_thresholds_and_controls() -> None:
    q50 = np.asarray([
        [0.01, 0.02, 0.03],
        [0.001, 0.000, -0.01],
    ])
    q10 = q50 - 0.01
    momentum = np.asarray([
        [0.10, 0.20, 0.30],
        [-0.01, -0.02, -0.03],
    ])
    availability = np.asarray([
        [True, True, False],
        [True, False, True],
    ])
    positions = build_policy_positions(
        q10, q50, momentum, availability, -0.03, 0.002
    )
    np.testing.assert_array_equal(
        positions["candidate"], [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    )
    np.testing.assert_array_equal(
        positions["dual_momentum_30"], [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    )
    np.testing.assert_allclose(
        positions["equal_weight_buy_hold"],
        [[0.5, 0.5, 0.0], [0.5, 0.0, 0.5]],
    )


def test_performance_metrics_compound_and_measure_drawdown() -> None:
    metrics = performance_metrics(
        np.asarray([0.10, -0.05, 0.02]),
        np.asarray([1.0, 0.0, 1.0]),
        np.asarray([0.001, 0.0, 0.001]),
    )
    assert np.isclose(metrics["total_return"], 1.10 * 0.95 * 1.02 - 1.0)
    assert np.isclose(metrics["max_drawdown"], -0.05)
    assert metrics["total_turnover"] == 2.0
    assert metrics["total_cost"] == 0.002


def _metric(total_return: float, sharpe: float, drawdown: float) -> dict:
    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
    }


def test_registered_gates_require_every_cost_and_bootstrap_cell() -> None:
    aggregate = {
        str(cost): {
            "candidate": _metric(0.20 - cost / 10000, 1.2, -0.10),
            "dual_momentum_30": _metric(0.10, 0.8, -0.12),
            "equal_weight_buy_hold": _metric(0.05, 0.5, -0.20),
        }
        for cost in (10, 20, 30)
    }
    bootstrap = {
        str(block): {
            "comparisons": {
                control: {"paired_total_return_delta": {"p05": 0.01}}
                for control in ("dual_momentum_30", "equal_weight_buy_hold")
            }
        }
        for block in (7, 21, 63)
    }
    gates = {
        "total_return_above_controls": [
            "dual_momentum_30", "equal_weight_buy_hold"
        ],
        "max_drawdown_tolerance_vs_primary": 0.05,
        "maximum_absolute_drawdown": 0.35,
        "paired_total_return_delta_p05_above_zero_for_controls": [
            "dual_momentum_30", "equal_weight_buy_hold"
        ],
        "failure_action": "retire",
        "pass_action": "advance",
    }
    passing = evaluate_registered_gates(aggregate, bootstrap, gates)
    assert passing["passed"]
    bootstrap["63"]["comparisons"]["dual_momentum_30"][
        "paired_total_return_delta"
    ]["p05"] = -0.001
    failing = evaluate_registered_gates(aggregate, bootstrap, gates)
    assert not failing["passed"]
    assert len(failing["cost_cells"]) == 3
    assert len(failing["bootstrap_cells"]) == 6


def test_evaluation_spec_is_deterministic_and_disables_repeat_or_training() -> None:
    contract = {
        "expected_input_sha256": {"a": "hash"},
        "evaluation": {"signal_start": "2026-01-01"},
        "policy": {"ranking_head": "return_q50"},
        "controls": {"primary": {"name": "dual_momentum_30"}},
        "accounting": {"costs_bps_per_unit_turnover": [10, 20, 30]},
        "gates": {"bootstrap_paths": 10000},
        "device": "mps",
        "inference_batch_size": 256,
    }
    first = build_evaluation_spec(contract)
    second = build_evaluation_spec(contract)
    assert first == second
    assert not first["training_allowed"]
    assert not first["repeat_evaluation_allowed_after_result"]


def test_fold_inference_averages_all_fixture_triplet_contexts() -> None:
    end = pd.Timestamp("2026-01-01", tz="UTC")
    dates = pd.date_range(end=end, periods=256, freq="D")
    symbols = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
    feature_names = [
        "log_open_to_open_return",
        "log_close_to_close_return",
        "log_high_low_range",
        "log_close_open_return",
        "log1p_quote_volume_change",
        "log1p_trade_count_change",
        "rolling_realized_volatility_7d",
        "rolling_realized_volatility_30d",
    ]
    rows = []
    for asset, symbol in enumerate(symbols):
        for date in dates:
            row = {
                "date": date,
                "symbol": symbol,
                "in_one_shot_non_target_confirmation": date == end,
                "supervised_sequence_ready": date == end,
                "target_next_open_to_next_open_log_return": 0.01 * (asset + 1),
                "target_realized_volatility_7d": 0.10,
            }
            row.update({name: 0.001 * (asset + 1) for name in feature_names})
            rows.append(row)
    panel = pd.DataFrame(rows)
    scaler = FeatureScaler(
        feature_names=tuple(feature_names),
        mean=tuple([0.0] * 8),
        scale=tuple([1.0] * 8),
        source_relative_feature_index=1,
        fit_scope="representation_train_only",
        fit_start="2021-01-01",
        fit_end="2023-12-31",
        fit_rows=100,
    )
    fold = {
        "fold": 1,
        "test_symbols": symbols,
        "test_triplets": [symbols],
    }
    calibration = {
        "offsets": {
            "return_q10": 0.0,
            "return_q50": 0.0,
            "return_q90": 0.0,
            "log_volatility": 0.0,
        },
        "member_checkpoint_sha256": ["a", "b", "c"],
        "calibration_semantic_sha256": "semantic",
    }
    result = _fold_inference(
        fold,
        panel,
        feature_names,
        [_FixtureModel(), _FixtureModel(), _FixtureModel()],
        scaler,
        calibration,
        {
            "signal_start": "2026-01-01",
            "signal_end": "2026-01-01",
            "eligibility_rule": "supervised_sequence_ready",
            "volatility_floor": 1e-6,
        },
        {"enter_if_q10_above": -0.03, "enter_if_q50_above": 0.002},
        batch_size=8,
        device=torch.device("cpu"),
    )
    assert result["diagnostics"]["triplet_prediction_count"] == 1
    assert result["diagnostics"]["asset_date_prediction_count"] == 3
    assert result["diagnostics"]["context_count_min"] == 1
    assert len(result["prediction_rows"]) == 3
    assert result["positions"]["candidate"].sum() == 1.0
