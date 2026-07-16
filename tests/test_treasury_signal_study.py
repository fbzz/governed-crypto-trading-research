from copy import deepcopy
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from tlm.treasury_data import run_treasury_pipeline
from tlm.treasury_signal_study import (
    build_treasury_diagnostic_signals,
    load_audited_treasury_features,
    run_treasury_signal_study,
)


def _csv(year: int) -> bytes:
    lines = ["Date,2 Yr,10 Yr"]
    start = max(pd.Timestamp(f"{year}-01-01"), pd.Timestamp("2020-01-01"))
    end = min(pd.Timestamp(f"{year}-12-31"), pd.Timestamp("2021-12-31"))
    for index, date in enumerate(pd.bdate_range(start, end)):
        absolute = (date - pd.Timestamp("2020-01-01")).days
        two = 1.0 + absolute * 0.001 + np.sin(absolute / 13) * 0.05
        ten = 2.0 + absolute * 0.0005 + np.cos(absolute / 17) * 0.08
        lines.append(f"{date.strftime('%m/%d/%Y')},{two:.4f},{ten:.4f}")
    return ("\n".join(lines) + "\n").encode()


def _data_config(tmp_path) -> dict:
    return {
        "treasury": {
            "csv_url_template": "https://example.test/{year}.csv",
            "start": "2020-01-01", "end": "2021-12-31",
            "timeout_seconds": 1, "raw_dir": str(tmp_path / "raw"),
            "source_columns": ["2 Yr", "10 Yr"],
            "minimum_weekday_coverage": 1.0,
            "execution_lag_calendar_days": 3, "maximum_carry_days": 7,
            "rolling_window": 10,
            "feature_columns": [
                "market__treasury_2y", "market__treasury_10y",
                "market__treasury_curve_10y_2y", "market__treasury_2y_change",
                "market__treasury_10y_change", "market__treasury_curve_change",
                "market__treasury_curve_z10", "market__treasury_10y_z10",
                "market__treasury_2y_change_vol10",
                "market__treasury_10y_change_vol10",
            ],
        },
        "output_dir": str(tmp_path / "treasury"),
    }


def _fetcher(url: str, timeout: float) -> bytes:
    year = int(urlparse(url).path.rsplit("/", 1)[-1].removesuffix(".csv"))
    return _csv(year)


def test_audited_treasury_signal_alignment(tmp_path):
    config = _data_config(tmp_path)
    run_treasury_pipeline(config, fetch_bytes=_fetcher)
    signal = "market__treasury_curve_10y_2y"
    features, source = load_audited_treasury_features(config["output_dir"], {signal})
    dates = features.index[10:30]
    signals = build_treasury_diagnostic_signals(dates, features, [signal])
    assert source["audit"]["passed"]
    assert signals.index.equals(dates)
    assert np.isfinite(signals[signal]).all()


def test_fixture_treasury_signal_study_writes_audited_outputs(tmp_path):
    source_config = _data_config(tmp_path)
    run_treasury_pipeline(source_config, fetch_bytes=_fetcher)
    signals = ["market__treasury_2y", "market__treasury_curve_change"]
    config = {
        "seed": 31,
        "data": {
            "source": "fixture", "interval": "1d", "fixture_days": 800,
            "assets": {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"},
        },
        "target": {"mode": "next_open_to_open"},
        "treasury_signal_study": {
            "treasury_artifact_dir": source_config["output_dir"],
            "start": "2020-04-09", "end": "2021-12-31",
            "momentum_lookback": 30, "purge_samples": 1,
            "quantile_bins": 5, "tail_quantile": 0.10, "signals": signals,
            "minimum_observed_coverage": 0.90, "minimum_test_rho": 0.50,
            "minimum_orientation_consistency": 0.50,
            "minimum_monotonic_fold_fraction": 0.50,
            "minimum_risk_coverage": 0.10, "maximum_risk_coverage": 0.35,
            "minimum_downside_lift": 1.10, "minimum_tail_lift": 1.10,
            "minimum_bootstrap_probability": 0.50,
        },
        "validation_suite": {
            "monte_carlo": {
                "paths": 20, "block_lengths": [7], "seed": 11, "batch_size": 10,
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
    result = run_treasury_signal_study(deepcopy(config))
    assert result["audit"]["passed"]
    assert result["signal_count"] == 2
    assert (tmp_path / "study" / "treasury_signal_study.json").is_file()
    assert (tmp_path / "study" / "report.md").is_file()
