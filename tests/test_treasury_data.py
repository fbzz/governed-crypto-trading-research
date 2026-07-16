from urllib.parse import urlparse

import numpy as np
import pandas as pd

from tlm.treasury_data import (
    build_treasury_source_features,
    load_or_download_treasury_year,
    materialize_causal_treasury_state,
    run_treasury_pipeline,
)


def _csv() -> bytes:
    lines = ["Date,2 Yr,10 Yr"]
    for index, date in enumerate(pd.bdate_range("2022-01-03", "2022-05-31")):
        two = 1.0 + index * 0.005 + np.sin(index / 7) * 0.05
        ten = 2.0 + index * 0.003 + np.cos(index / 9) * 0.06
        lines.append(f"{date.strftime('%m/%d/%Y')},{two:.4f},{ten:.4f}")
    return ("\n".join(lines) + "\n").encode()


def _config(tmp_path) -> dict:
    return {
        "treasury": {
            "csv_url_template": "https://example.test/{year}.csv",
            "start": "2022-01-03", "end": "2022-05-31",
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
        "output_dir": str(tmp_path / "output"),
    }


def _fetcher(url: str, timeout: float) -> bytes:
    assert urlparse(url).path.endswith("2022.csv")
    assert timeout == 1
    return _csv()


def test_treasury_cache_replay_avoids_network(tmp_path):
    config = _config(tmp_path)
    calls = []

    def counted(url: str, timeout: float) -> bytes:
        calls.append(url)
        return _fetcher(url, timeout)

    first, first_meta = load_or_download_treasury_year(
        config, 2022, fetch_bytes=counted
    )
    second, second_meta = load_or_download_treasury_year(
        config, 2022, fetch_bytes=counted
    )
    assert len(calls) == 1
    assert not first_meta["cached"] and second_meta["cached"]
    assert first.equals(second)
    assert first_meta["observations_sha256"] == second_meta["observations_sha256"]


def test_known_state_materialization_is_lagged_and_bounded(tmp_path):
    config = _config(tmp_path)
    raw = load_or_download_treasury_year(config, 2022, fetch_bytes=_fetcher)[0]
    features = build_treasury_source_features(raw, 10)
    state = materialize_causal_treasury_state(features, 3, 7)
    assert (state["source_final_at"] < state.index.to_series()).all()
    assert (state["source_eligible_at"] <= state.index.to_series()).all()
    assert state["source_age_days"].between(3, 7).all()


def test_fixture_treasury_pipeline_is_reproducible_from_cache(tmp_path):
    config = _config(tmp_path)
    first = run_treasury_pipeline(config, fetch_bytes=_fetcher)

    def forbidden(url: str, timeout: float) -> bytes:
        raise AssertionError("cache miss on replay")

    second = run_treasury_pipeline(config, fetch_bytes=forbidden)
    output = tmp_path / "output"
    assert first["audit"]["passed"] and second["audit"]["passed"]
    assert all(row["cached"] for row in second["manifest"]["records"])
    assert (output / "treasury_raw.parquet").is_file()
    assert (output / "treasury_features.parquet").is_file()
    assert (output / "catalog.json").is_file()
    assert (output / "report.md").is_file()
