from copy import deepcopy

import numpy as np
import pandas as pd

from tlm.risk_off import (
    calibrate_risk_threshold,
    control_simple_returns,
    fit_quantile_models,
    predict_quantiles,
    risk_off_gate,
    run_risk_off_suite,
    select_risk_off_actions,
)


def test_control_target_uses_ranked_asset_or_zero_for_cash():
    actual = np.log1p(np.array([
        [0.02, -0.01],
        [0.03, 0.04],
        [-0.02, 0.01],
    ]))
    choices = np.array([0, 1, -1])
    np.testing.assert_allclose(
        control_simple_returns(actual, choices),
        np.array([0.02, 0.04, 0.0]),
    )


def test_risk_off_policy_uses_hysteresis_and_resets_on_control_cash():
    lower = np.array([-0.05, -0.035, -0.02, -0.08])
    baseline = np.array([0, 0, 1, -1])
    selected, risk_off, final_state = select_risk_off_actions(
        lower,
        baseline,
        thresholds=-0.04,
        n_assets=2,
        hysteresis_margin=0.01,
    )
    assert selected.tolist() == [2, 2, 1, 2]
    assert risk_off.tolist() == [True, True, False, False]
    assert final_state is False


def test_quantile_predictions_are_monotonic_after_projection():
    x = np.arange(160, dtype=float).reshape(-1, 2)
    target = np.sin(x[:, 0] / 20.0) * 0.03
    config = {
        "quantiles": [0.10, 0.50],
        "learning_rate": 0.05,
        "n_estimators": 8,
        "max_depth": 2,
        "min_samples_leaf": 10,
        "max_features": None,
    }
    models = fit_quantile_models(x, target, config, seed=3)
    predictions, quantiles = predict_quantiles(models, x[:20])
    assert quantiles == (0.1, 0.5)
    assert np.all(predictions[:, 0] <= predictions[:, 1])


def test_risk_threshold_calibration_abstains_when_cash_loses_return():
    rows = 40
    lower = np.full(rows, -0.05)
    baseline = np.zeros(rows, dtype=int)
    actual = np.log1p(np.column_stack([np.full(rows, 0.002), np.zeros(rows)]))
    dates = pd.date_range("2024-01-01", periods=rows, tz="UTC")
    config = {
        "threshold_grid": [-0.06, -0.04],
        "hysteresis_margin": 0.005,
        "minimum_calibration_risk_off_days": 3,
        "minimum_calibration_return_retention": 0.85,
        "minimum_calibration_drawdown_improvement": 0.01,
    }
    threshold, evaluations = calibrate_risk_threshold(
        lower,
        baseline,
        actual,
        dates,
        ("BTC", "ETH"),
        config,
        cost_bps=10,
    )
    assert np.isneginf(threshold)
    assert not any(item["qualifies"] for item in evaluations)


def test_risk_off_gate_requires_retention_sharpe_and_drawdown():
    control = {"total_return": 2.0, "sharpe": 1.0, "max_drawdown": -0.50}
    candidate = {"total_return": 1.9, "sharpe": 1.1, "max_drawdown": -0.45}
    gates = {
        "minimum_return_retention": 0.90,
        "minimum_drawdown_improvement": 0.03,
    }
    assert risk_off_gate(candidate, control, gates)
    candidate["total_return"] = 1.7
    assert not risk_off_gate(candidate, control, gates)


def test_fixture_risk_off_suite_writes_audited_results(tmp_path):
    config = {
        "seed": 13,
        "data": {
            "source": "fixture",
            "interval": "1d",
            "fixture_days": 520,
            "assets": {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"},
        },
        "target": {"mode": "next_open_to_open"},
        "strategy": {
            "policy": "dual_momentum_or_cash_quantile_risk_off",
            "cost_bps": 10.0,
        },
        "risk_off": {
            "momentum_lookback": 30,
            "calibration_fraction": 0.20,
            "min_core_samples": 100,
            "min_calibration_samples": 35,
            "purge_samples": 1,
            "threshold_grid": [-0.08, -0.05, -0.03],
            "hysteresis_margin": 0.005,
            "minimum_calibration_risk_off_days": 3,
            "minimum_calibration_return_retention": 0.80,
            "minimum_calibration_drawdown_improvement": 0.0,
            "model": {
                "quantiles": [0.10, 0.50],
                "learning_rate": 0.05,
                "n_estimators": 8,
                "max_depth": 2,
                "min_samples_leaf": 15,
                "max_features": None,
            },
            "gates": {
                "minimum_return_retention": 0.80,
                "minimum_drawdown_improvement": 0.0,
                "fold_return_tolerance": 0.10,
                "cost_sensitivity_bps": [10, 20],
                "minimum_monte_carlo_probability": 0.50,
            },
        },
        "validation_suite": {
            "monte_carlo": {
                "paths": 20,
                "block_lengths": [7],
                "seed": 101,
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
        "output_dir": str(tmp_path / "risk_off"),
    }
    result = run_risk_off_suite(deepcopy(config))
    assert result["scenario_count"] == 2
    assert result["audit"]["passed"]
    assert result["clean_holdout_status"].startswith("unavailable")
    assert (tmp_path / "risk_off" / "validation_suite.json").is_file()
    assert (tmp_path / "risk_off" / "scenarios" / "rolling" / "diagnostics.json").is_file()
