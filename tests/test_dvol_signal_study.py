from copy import deepcopy
import json
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from tlm.dvol_data import run_dvol_pipeline
from tlm.dvol_signal_study import (
    build_dvol_diagnostic_signals,
    load_audited_dvol_features,
    run_dvol_signal_study,
)


def _payload(start: str, end: str, offset: float) -> dict:
    rows = []
    for index, timestamp in enumerate(pd.date_range(start, end, freq="D", tz="UTC")):
        value = 55.0 + offset + index * 0.02 + np.sin(index / 11.0) * 3.0
        rows.append([
            int(timestamp.timestamp() * 1000), value, value + 2.0,
            value - 2.0, value + np.sin(index / 3.0),
        ])
    return {"result": {"data": rows, "continuation": None}}


def _dvol_config(tmp_path) -> dict:
    columns = []
    for currency in ("btc", "eth"):
        prefix = f"market__{currency}_dvol"
        columns.extend([
            f"{prefix}_close", f"{prefix}_log_change_1d",
            f"{prefix}_intraday_log_range", f"{prefix}_close_z10",
            f"{prefix}_range_z10", f"{prefix}_change_vol5",
        ])
    columns.extend([
        "market__dvol_mean_close", "market__dvol_mean_log_change_1d",
        "market__dvol_close_dispersion", "market__dvol_change_dispersion",
    ])
    return {
        "dvol": {
            "endpoint": "https://example.test/dvol",
            "currencies": ["BTC", "ETH"],
            "start": "2020-01-01", "end": "2021-12-31", "resolution": "1D",
            "timeout_seconds": 1, "raw_dir": str(tmp_path / "raw"),
            "minimum_daily_coverage": 1.0,
            "rolling_z_window": 10, "change_vol_window": 5,
            "feature_columns": columns,
        },
        "output_dir": str(tmp_path / "dvol"),
    }


def _fetcher(url: str, timeout: float) -> dict:
    params = parse_qs(urlparse(url).query)
    start = pd.to_datetime(int(params["start_timestamp"][0]), unit="ms", utc=True)
    end = pd.to_datetime(int(params["end_timestamp"][0]), unit="ms", utc=True)
    offset = 8.0 if params["currency"][0] == "ETH" else 0.0
    return _payload(str(start.date()), str(end.date()), offset)


def test_audited_loader_preserves_direct_market_signal_alignment(tmp_path):
    config = _dvol_config(tmp_path)
    run_dvol_pipeline(config, fetch_json=_fetcher)
    signal = "market__dvol_mean_log_change_1d"
    features, source = load_audited_dvol_features(config["output_dir"], {signal})
    dates = features.index[5:20]
    signals = build_dvol_diagnostic_signals(dates, features, [signal])
    assert source["audit"]["passed"]
    assert signals.index.equals(dates)
    assert np.isfinite(signals[signal]).all()


def test_fixture_dvol_signal_study_writes_audited_outputs(tmp_path):
    dvol_config = _dvol_config(tmp_path)
    run_dvol_pipeline(dvol_config, fetch_json=_fetcher)
    signals = [
        "market__dvol_mean_close",
        "market__dvol_mean_log_change_1d",
    ]
    config = {
        "seed": 31,
        "data": {
            "source": "fixture", "interval": "1d", "fixture_days": 800,
            "assets": {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"},
        },
        "target": {"mode": "next_open_to_open"},
        "dvol_signal_study": {
            "dvol_artifact_dir": dvol_config["output_dir"],
            "start": "2020-04-09", "end": "2021-12-31",
            "momentum_lookback": 30, "purge_samples": 1,
            "quantile_bins": 5, "tail_quantile": 0.10,
            "signals": signals, "minimum_observed_coverage": 0.90,
            "minimum_test_rho": 0.50, "minimum_orientation_consistency": 0.50,
            "minimum_monotonic_fold_fraction": 0.50,
            "minimum_risk_coverage": 0.10, "maximum_risk_coverage": 0.35,
            "minimum_downside_lift": 1.10, "minimum_tail_lift": 1.10,
            "minimum_bootstrap_probability": 0.50,
        },
        "validation_suite": {
            "monte_carlo": {
                "paths": 20, "block_lengths": [7], "seed": 9, "batch_size": 10,
            },
            "scenarios": [
                {"name": "expanding", "validation": {
                    "mode": "expanding", "folds": 2, "min_train_fraction": 0.50,
                }},
                {"name": "rolling", "validation": {
                    "mode": "rolling", "folds": 2, "min_train_fraction": 0.50,
                    "train_window_samples": 250,
                }},
            ],
        },
        "output_dir": str(tmp_path / "study"),
    }
    result = run_dvol_signal_study(deepcopy(config))
    assert result["signal_count"] == 2
    assert result["scenario_count"] == 2
    assert result["audit"]["passed"]
    assert (tmp_path / "study" / "dvol_signal_study.json").is_file()
    assert (tmp_path / "study" / "downside_lift.png").is_file()
    persisted = json.loads((tmp_path / "study" / "audit.json").read_text())
    assert persisted["checks"]["no_portfolio_metrics"]
