from copy import deepcopy

import numpy as np
import pandas as pd

from tlm.intraday_path_study import (
    PATH_FEATURE_COLUMNS,
    aggregate_intraday_metrics,
    run_intraday_path_pipeline,
    run_intraday_path_signal_study,
)


def test_intraday_path_features_use_only_observed_day_samples():
    timestamps = pd.date_range(
        "2022-01-01", periods=24, freq="5min", tz="UTC"
    )
    taker = np.r_[np.ones(12), np.full(12, 2.0)]
    oi = np.linspace(100.0, 123.0, 24)
    frame = pd.DataFrame({
        "create_time": timestamps.astype(str),
        "sum_open_interest": oi,
        "sum_open_interest_value": oi * 100.0,
        "sum_taker_long_short_vol_ratio": taker,
    })
    daily = aggregate_intraday_metrics(frame, minimum_samples=20)
    row = daily.iloc[0]
    assert row["metrics_samples"] == 24
    assert row["taker_ratio_last"] == 2.0
    assert np.isclose(row["taker_ratio_log_change"], np.log(2.0))
    assert np.isclose(row["taker_buy_fraction"], 0.5)
    assert np.isclose(row["taker_ratio_last_hour_mean"], 2.0)
    assert np.isclose(row["taker_ratio_reversal_1h"], np.log(2.0))
    assert np.isclose(row["oi_log_change_intraday"], np.log(123.0 / 100.0))
    assert np.isclose(row["oi_range_intraday"], 0.23)
    assert row["oi_max_drawdown_intraday"] == 0.0
    assert row["source_max_timestamp"] == timestamps[-1]
    assert row["source_max_timestamp"] < row["execution_open"]


def test_intraday_path_rejects_under_sampled_days_without_imputation():
    timestamps = pd.date_range(
        "2022-01-01", periods=10, freq="5min", tz="UTC"
    )
    frame = pd.DataFrame({
        "create_time": timestamps.astype(str),
        "sum_open_interest": np.linspace(100.0, 110.0, 10),
        "sum_open_interest_value": np.linspace(10_000.0, 11_000.0, 10),
        "sum_taker_long_short_vol_ratio": np.linspace(0.8, 1.2, 10),
    })
    daily = aggregate_intraday_metrics(frame, minimum_samples=20)
    assert daily[list(PATH_FEATURE_COLUMNS)].isna().all(axis=None)
    assert daily.iloc[0]["metrics_samples"] == 10


def fixture_config(tmp_path) -> dict:
    assets = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}
    return {
        "seed": 41,
        "data": {
            "source": "fixture",
            "interval": "1d",
            "fixture_days": 800,
            "assets": assets,
        },
        "target": {"mode": "next_open_to_open"},
        "intraday_path": {
            "source": "fixture",
            "start": "2020-05-01",
            "end": "2021-12-31",
            "symbols": assets,
            "minimum_intraday_samples": 250,
            "minimum_feature_coverage": 1.0,
        },
        "derivatives_signal_study": {
            "study_title": "TLM v12 Fixture",
            "method": "fixture_intraday_path_study",
            "positive_conclusion": "fixture_signal_exists",
            "negative_conclusion": "fixture_no_signal",
            "derivatives_artifact_dir": str(tmp_path / "study" / "path_data"),
            "start": "2020-05-01",
            "end": "2021-12-31",
            "momentum_lookback": 30,
            "purge_samples": 1,
            "quantile_bins": 5,
            "tail_quantile": 0.10,
            "signals": [
                "control__taker_ratio_slope",
                "market_mean__oi_slope_intraday",
                "cross_dispersion__oi_range_intraday",
            ],
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
                "seed": 113,
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


def test_fixture_intraday_path_pipeline_is_complete_and_causal(tmp_path):
    config = fixture_config(tmp_path)
    result = run_intraday_path_pipeline(deepcopy(config))
    output = tmp_path / "study" / "path_data"
    assert result["audit"]["passed"]
    assert result["source_record_count"] > 0
    assert (output / "intraday_path_daily.parquet").is_file()
    assert (output / "assets" / "BTC.parquet").is_file()
    assert (output / "audit.json").is_file()
    assert (output / "report.md").is_file()


def test_fixture_v12_study_reuses_registered_walk_forward_gates(tmp_path):
    config = fixture_config(tmp_path)
    result = run_intraday_path_signal_study(deepcopy(config))
    output = tmp_path / "study"
    assert result["signal_count"] == 3
    assert result["scenario_count"] == 2
    assert result["path_data"]["audit_passed"]
    assert result["audit"]["passed"]
    assert (output / "derivatives_signal_study.json").is_file()
    assert (output / "downside_lift.png").is_file()
    assert (output / "scenarios" / "rolling" / "signal_metrics.json").is_file()
