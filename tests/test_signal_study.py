from copy import deepcopy

import numpy as np

from tlm.data import generate_fixture
from tlm.override import make_override_dataset
from tlm.signal_study import (
    assign_registered_bins,
    block_bootstrap_risk_lift,
    build_diagnostic_signals,
    evaluate_signal_rule,
    fit_signal_rule,
    run_signal_study,
)


def test_control_signal_uses_feature_for_current_ranked_asset():
    dataset = make_override_dataset(
        generate_fixture(["BTC", "ETH", "SOL"], days=240, seed=21),
        momentum_lookback=30,
    )
    signals = build_diagnostic_signals(dataset, ["control__return_30"])
    row = int(np.flatnonzero(dataset.baseline_choices >= 0)[0])
    asset = dataset.asset_names[dataset.baseline_choices[row]]
    feature_index = dataset.feature_names.index(f"{asset}__return_30")
    assert signals.iloc[row, 0] == dataset.x[row, feature_index]


def test_registered_train_edges_are_unchanged_by_test_extremes():
    train = np.arange(1.0, 101.0)
    returns = -train / 10_000.0
    rule = fit_signal_rule(train, returns, bins=5, tail_quantile=0.10)
    original_edges = np.asarray(rule["edges"]).copy()
    assigned = assign_registered_bins(np.array([-1e9, 1e9]), original_edges)
    assert assigned.tolist() == [0, 4]
    np.testing.assert_array_equal(rule["edges"], original_edges)


def test_train_orientation_and_oos_monotonicity_detect_downside():
    train_values = np.arange(1.0, 101.0)
    train_returns = -train_values / 1_000.0
    rule = fit_signal_rule(train_values, train_returns, bins=5, tail_quantile=0.10)
    assert rule["orientation"] == 1
    assert rule["risk_bin"] == 4
    test_values = np.arange(1.5, 101.5)
    test_returns = -test_values / 1_000.0
    observations, metrics = evaluate_signal_rule(
        test_values, test_returns, rule, bins=5
    )
    assert observations["risk_flag"].any()
    assert metrics["test_downside_rho"] > 0.5


def test_block_bootstrap_recovers_stronger_registered_risk_bucket():
    rows = 300
    risk = np.zeros(rows, dtype=bool)
    risk[::5] = True
    downside = np.where(risk, 0.05, 0.01)
    tail = np.where(risk, 1.0, 0.05)
    result = block_bootstrap_risk_lift(
        downside,
        tail,
        risk,
        block_length=7,
        n_paths=200,
        seed=5,
        batch_size=50,
    )
    assert result["probability_downside_lift_above_one"] > 0.95
    assert result["probability_tail_lift_above_one"] > 0.95


def test_block_bootstrap_resamples_paths_missing_a_clustered_group():
    rows = 200
    risk = np.zeros(rows, dtype=bool)
    risk[:10] = True
    result = block_bootstrap_risk_lift(
        np.where(risk, 0.05, 0.01),
        np.where(risk, 1.0, 0.05),
        risk,
        block_length=63,
        n_paths=50,
        seed=6,
        batch_size=10,
    )
    assert result["paths"] == 50
    assert result["probability_downside_lift_above_one"] > 0.9


def test_fixture_signal_study_writes_audited_outputs(tmp_path):
    config = {
        "seed": 23,
        "data": {
            "source": "fixture",
            "interval": "1d",
            "fixture_days": 520,
            "assets": {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"},
        },
        "target": {"mode": "next_open_to_open"},
        "signal_study": {
            "momentum_lookback": 30,
            "purge_samples": 1,
            "quantile_bins": 5,
            "tail_quantile": 0.10,
            "signals": [
                "control__return_30",
                "control__volatility_21",
                "market__volatility_21",
            ],
            "minimum_test_rho": 0.50,
            "minimum_orientation_consistency": 0.50,
            "minimum_monotonic_fold_fraction": 0.50,
            "minimum_risk_coverage": 0.10,
            "maximum_risk_coverage": 0.35,
            "minimum_downside_lift": 1.10,
            "minimum_tail_lift": 1.10,
            "minimum_bootstrap_probability": 0.50,
        },
        "validation_suite": {
            "monte_carlo": {
                "paths": 20,
                "block_lengths": [7],
                "seed": 107,
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
        "output_dir": str(tmp_path / "signal_study"),
    }
    result = run_signal_study(deepcopy(config))
    assert result["signal_count"] == 3
    assert result["scenario_count"] == 2
    assert result["audit"]["passed"]
    assert (tmp_path / "signal_study" / "signal_study.json").is_file()
    assert (tmp_path / "signal_study" / "downside_lift.png").is_file()
    assert (
        tmp_path
        / "signal_study"
        / "scenarios"
        / "rolling"
        / "signals.parquet"
    ).is_file()
