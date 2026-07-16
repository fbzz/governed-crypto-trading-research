from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from tlm.derivatives_data import run_derivatives_pipeline
from tlm.derivatives_signal_study import (
    build_derivatives_diagnostic_signals,
    run_derivatives_signal_scenario,
    run_derivatives_signal_study,
)
from tlm.override import OverrideDataset


def test_derivatives_signal_transforms_preserve_control_and_complete_cases():
    dates = pd.date_range("2022-01-01", periods=3, freq="D", tz="UTC")
    dataset = OverrideDataset(
        x=np.zeros((3, 1)),
        actual_log_returns=np.zeros((3, 3)),
        baseline_choices=np.array([0, 1, -1]),
        dates=dates,
        feature_names=("placeholder",),
        asset_names=("BTC", "ETH", "SOL"),
    )
    derivatives = {}
    for offset, asset in enumerate(dataset.asset_names):
        derivatives[asset] = pd.DataFrame({
            "funding_rate_7d_sum": np.arange(3, dtype=float) + offset,
            "basis_30d_z": np.arange(3, dtype=float) + offset * 2,
        }, index=dates)
    derivatives["BTC"].loc[dates[1], "basis_30d_z"] = np.nan
    signals = build_derivatives_diagnostic_signals(dataset, derivatives, [
        "control__funding_rate_7d_sum",
        "market_mean__basis_30d_z",
        "cross_dispersion__funding_rate_7d_sum",
    ])
    assert signals.loc[dates[0], "control__funding_rate_7d_sum"] == 0.0
    assert signals.loc[dates[1], "control__funding_rate_7d_sum"] == 2.0
    assert np.isnan(signals.loc[dates[2], "control__funding_rate_7d_sum"])
    assert np.isnan(signals.loc[dates[1], "market_mean__basis_30d_z"])
    assert np.isclose(
        signals.loc[dates[0], "cross_dispersion__funding_rate_7d_sum"],
        np.std([0.0, 1.0, 2.0]),
    )


def test_derivatives_signal_registry_rejects_unregistered_columns():
    dates = pd.date_range("2022-01-01", periods=3, freq="D", tz="UTC")
    dataset = OverrideDataset(
        x=np.zeros((3, 1)),
        actual_log_returns=np.zeros((3, 3)),
        baseline_choices=np.array([0, 1, 2]),
        dates=dates,
        feature_names=("placeholder",),
        asset_names=("BTC", "ETH", "SOL"),
    )
    frames = {asset: pd.DataFrame(index=dates) for asset in dataset.asset_names}
    with pytest.raises(ValueError, match="Unregistered"):
        build_derivatives_diagnostic_signals(
            dataset, frames, ["control__future_leak"]
        )


def test_constant_registered_signal_is_recorded_as_untestable(tmp_path):
    rows = 400
    dates = pd.date_range("2022-01-01", periods=rows, freq="D", tz="UTC")
    rng = np.random.default_rng(11)
    dataset = OverrideDataset(
        x=np.zeros((rows, 1)),
        actual_log_returns=rng.normal(0.0, 0.02, (rows, 3)),
        baseline_choices=np.arange(rows) % 3,
        dates=dates,
        feature_names=("placeholder",),
        asset_names=("BTC", "ETH", "SOL"),
    )
    signal = "control__funding_rate_sum"
    config = {
        "output_dir": str(tmp_path / "constant"),
        "derivatives_signal_study": {
            "signals": [signal],
            "purge_samples": 1,
            "quantile_bins": 5,
            "tail_quantile": 0.10,
            "minimum_observed_coverage": 0.90,
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
                "seed": 5,
                "batch_size": 10,
            }
        },
    }
    scenario = {
        "name": "expanding",
        "validation": {
            "mode": "expanding",
            "folds": 2,
            "min_train_fraction": 0.50,
        },
    }
    signals = pd.DataFrame({signal: np.ones(rows)}, index=dates)
    result = run_derivatives_signal_scenario(
        dataset, signals, config, scenario, scenario_index=0
    )
    assert signal in result["untestable_signals"]
    assert not result["signal_metrics"][signal]["passes"]
    assert not result["signal_metrics"][signal]["gate_checks"]["testable"]
    assert "insufficient unique" in result["signal_metrics"][signal]["failure_reason"]


def test_fixture_derivatives_signal_study_writes_audited_outputs(tmp_path):
    assets = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}
    derivatives_output = tmp_path / "derivatives"
    derivatives_config = {
        "seed": 31,
        "derivatives": {
            "source": "fixture",
            "base_url": "https://data.binance.vision",
            "start": "2020-05-01",
            "end": "2021-12-31",
            "raw_dir": str(tmp_path / "raw"),
            "symbols": assets,
            "workers": 2,
            "timeout_seconds": 1,
            "minimum_daily_coverage": 1.0,
            "minimum_derived_coverage": 1.0,
        },
        "output_dir": str(derivatives_output),
    }
    run_derivatives_pipeline(derivatives_config)
    config = {
        "seed": 31,
        "data": {
            "source": "fixture",
            "interval": "1d",
            "fixture_days": 800,
            "assets": assets,
        },
        "target": {"mode": "next_open_to_open"},
        "derivatives_signal_study": {
            "derivatives_artifact_dir": str(derivatives_output),
            "start": "2020-05-01",
            "end": "2021-12-31",
            "momentum_lookback": 30,
            "purge_samples": 1,
            "quantile_bins": 5,
            "tail_quantile": 0.10,
            "signals": [
                "control__funding_rate_7d_sum",
                "market_mean__basis_30d_z",
                "cross_dispersion__open_interest_log_change_7d",
            ],
            "minimum_observed_coverage": 0.80,
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
                "seed": 109,
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
                        "train_window_samples": 250,
                    },
                },
            ],
        },
        "output_dir": str(tmp_path / "study"),
    }
    result = run_derivatives_signal_study(deepcopy(config))
    assert result["signal_count"] == 3
    assert result["scenario_count"] == 2
    assert result["audit"]["passed"]
    assert all(
        "gate_checks" in metric
        for scenario in result["scenarios"].values()
        for metric in scenario["signal_metrics"].values()
    )
    assert (tmp_path / "study" / "derivatives_signal_study.json").is_file()
    assert (tmp_path / "study" / "dataset_summary.json").is_file()
    assert (tmp_path / "study" / "downside_lift.png").is_file()
    observations = pd.read_parquet(
        tmp_path / "study" / "scenarios" / "rolling" / "signals.parquet"
    )
    assert not observations.duplicated(["signal", "date"]).any()
    assert np.isfinite(observations["signal_value"]).all()
