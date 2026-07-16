from copy import deepcopy

import numpy as np
import pandas as pd

from tlm.data import generate_fixture
from tlm.override import (
    build_override_features,
    build_residual_targets,
    calibrate_abstention_threshold,
    diagnose_override_predictions,
    nested_chronological_split,
    run_override_suite,
    select_override_actions,
    transition_turnover,
)


def test_residual_targets_are_relative_to_daily_control_action():
    actual = np.log1p(np.array([
        [0.10, -0.05],
        [0.03, 0.07],
        [-0.02, 0.04],
    ]))
    baseline = np.array([0, 1, -1])
    residual = build_residual_targets(actual, baseline)
    np.testing.assert_allclose(residual, np.array([
        [0.00, -0.15, -0.10],
        [-0.04, 0.00, -0.07],
        [-0.02, 0.04, 0.00],
    ]))


def test_override_policy_subtracts_exact_incremental_switch_cost():
    predictions = np.array([
        [0.0, 0.010, -0.01],
        [0.001, 0.0, -0.01],
    ])
    baseline = np.array([0, 1])
    choices, edges, overridden, final_action = select_override_actions(
        predictions, baseline, thresholds=0.0, cost_bps=10
    )
    # Day one has equal entry turnover and overrides BTC with ETH. On day two,
    # switching ETH->BTC costs 20 bps more than keeping the ETH control action.
    assert choices.tolist() == [1, 1]
    assert overridden.tolist() == [True, False]
    assert np.isclose(edges[0], 0.010)
    assert edges[1] == 0.0
    assert final_action == 1
    assert transition_turnover(0, 1, cash_action=2) == 2.0
    assert transition_turnover(2, 1, cash_action=2) == 1.0


def test_nested_split_purges_inner_and_outer_label_boundaries():
    train = np.arange(200)
    core, calibration, metadata = nested_chronological_split(
        train,
        calibration_fraction=0.20,
        min_core_samples=100,
        min_calibration_samples=30,
        purge_samples=1,
    )
    assert core[-1] + 2 == calibration[0]
    assert calibration[-1] == train[-2]
    assert train[-1] not in calibration
    assert metadata["dropped_inner_boundary"] == 1
    assert metadata["dropped_outer_boundary"] == 1


def test_override_features_cannot_be_changed_by_future_ohlcv():
    frames = generate_fixture(["BTC", "ETH", "SOL"], days=220, seed=17)
    original, _, _ = build_override_features(frames)
    cutoff = frames["BTC"].index[170]
    perturbed = {asset: frame.copy() for asset, frame in frames.items()}
    for frame in perturbed.values():
        future = frame.index > cutoff
        frame.loc[future, ["open", "high", "low", "close", "volume"]] *= 3.0
    changed, _, _ = build_override_features(perturbed)
    pd.testing.assert_frame_equal(original.loc[:cutoff], changed.loc[:cutoff])


def test_calibration_abstains_when_no_threshold_improves_control():
    rows = 30
    predictions = np.zeros((rows, 3))
    baseline = np.zeros(rows, dtype=int)
    actual = np.log1p(np.column_stack([np.full(rows, 0.001), np.zeros(rows)]))
    dates = pd.date_range("2024-01-01", periods=rows, tz="UTC")
    config = {
        "threshold_grid": [0.0, 0.001],
        "minimum_calibration_overrides": 2,
        "minimum_calibration_return_delta": 0.0,
        "max_calibration_drawdown_worsening": 0.02,
    }
    threshold, evaluations = calibrate_abstention_threshold(
        predictions,
        baseline,
        actual,
        dates,
        ("BTC", "ETH"),
        config,
        cost_bps=10,
    )
    assert np.isinf(threshold)
    assert not any(item["qualifies"] for item in evaluations)


def test_override_diagnostics_compare_selected_action_with_control():
    frame = pd.DataFrame({
        "fold": [0, 0],
        "baseline_choice": [0, 1],
        "selected_action": [1, 1],
        "selected_edge": [0.02, 0.0],
        "overridden": [True, False],
        "actual_BTC": np.log1p([0.01, 0.00]),
        "actual_ETH": np.log1p([0.03, 0.02]),
    })
    result = diagnose_override_predictions(frame, ("BTC", "ETH"))
    assert result["overall"]["override_days"] == 1
    assert np.isclose(result["overall"]["realized_gross_residual_sum"], 0.02)
    assert result["overall"]["realized_edge_hit_rate"] == 1.0


def test_fixture_override_suite_writes_audited_results(tmp_path):
    config = {
        "seed": 11,
        "data": {
            "source": "fixture",
            "interval": "1d",
            "fixture_days": 520,
            "assets": {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"},
        },
        "target": {"mode": "next_open_to_open"},
        "strategy": {
            "policy": "dual_momentum_with_residual_override_and_abstention",
            "cost_bps": 10.0,
        },
        "override": {
            "momentum_lookback": 30,
            "calibration_fraction": 0.20,
            "min_core_samples": 100,
            "min_calibration_samples": 35,
            "purge_samples": 1,
            "threshold_grid": [0.0, 0.0025, 0.01],
            "minimum_calibration_overrides": 3,
            "minimum_calibration_return_delta": 0.0,
            "max_calibration_drawdown_worsening": 0.02,
            "model": {
                "learning_rate": 0.05,
                "n_estimators": 8,
                "max_depth": 2,
                "min_samples_leaf": 15,
                "max_features": None,
            },
            "gates": {
                "fold_return_tolerance": 0.10,
                "max_drawdown_worsening": 0.05,
                "cost_sensitivity_bps": [10, 20],
                "minimum_monte_carlo_probability": 0.70,
            },
        },
        "validation_suite": {
            "monte_carlo": {
                "paths": 20,
                "block_lengths": [7],
                "seed": 99,
                "batch_size": 10,
            },
            "scenarios": [
                {
                    "name": "expanding",
                    "validation": {
                        "mode": "expanding",
                        "folds": 2,
                        "min_train_fraction": 0.50,
                    },
                },
                {
                    "name": "rolling",
                    "validation": {
                        "mode": "rolling",
                        "folds": 2,
                        "min_train_fraction": 0.50,
                        "train_window_samples": 180,
                    },
                },
            ],
        },
        "output_dir": str(tmp_path / "override"),
    }
    result = run_override_suite(deepcopy(config))
    assert result["scenario_count"] == 2
    assert result["audit"]["passed"]
    assert (tmp_path / "override" / "validation_suite.json").is_file()
    assert (tmp_path / "override" / "scenarios" / "rolling" / "predictions.parquet").is_file()
    assert (tmp_path / "override" / "scenarios" / "rolling" / "diagnostics.json").is_file()
