from urllib.parse import parse_qs, urlparse

import pandas as pd

from tlm.dvol_data import (
    audit_dvol_dataset,
    build_causal_dvol_features,
    load_or_download_dvol,
    run_dvol_pipeline,
)


def _payload(start: str, days: int, offset: float = 0.0) -> dict:
    rows = []
    for index, timestamp in enumerate(pd.date_range(start, periods=days, freq="D", tz="UTC")):
        value = 50.0 + offset + index * 0.1 + (index % 7) * 0.2
        rows.append([
            int(timestamp.timestamp() * 1000),
            value,
            value + 2.0 + (index % 3) * 0.1,
            value - 2.0,
            value + (index % 5 - 2) * 0.15,
        ])
    return {"jsonrpc": "2.0", "result": {"data": rows, "continuation": None}}


def _config(tmp_path) -> dict:
    columns = []
    for currency in ("btc", "eth"):
        prefix = f"market__{currency}_dvol"
        columns.extend([
            f"{prefix}_close",
            f"{prefix}_log_change_1d",
            f"{prefix}_intraday_log_range",
            f"{prefix}_close_z10",
            f"{prefix}_range_z10",
            f"{prefix}_change_vol5",
        ])
    columns.extend([
        "market__dvol_mean_close",
        "market__dvol_mean_log_change_1d",
        "market__dvol_close_dispersion",
        "market__dvol_change_dispersion",
    ])
    return {
        "dvol": {
            "endpoint": "https://example.test/dvol",
            "currencies": ["BTC", "ETH"],
            "start": "2022-01-01",
            "end": "2022-02-09",
            "resolution": "1D",
            "timeout_seconds": 1,
            "raw_dir": str(tmp_path / "raw"),
            "minimum_daily_coverage": 1.0,
            "rolling_z_window": 10,
            "change_vol_window": 5,
            "feature_columns": columns,
        },
        "output_dir": str(tmp_path / "output"),
    }


def _fetcher(url: str, timeout: float) -> dict:
    params = parse_qs(urlparse(url).query)
    currency = params["currency"][0]
    return _payload("2022-01-01", 40, offset=10.0 if currency == "ETH" else 0.0)


def test_dvol_cache_prevents_second_network_fetch(tmp_path):
    config = _config(tmp_path)
    calls = []

    def counted_fetcher(url: str, timeout: float) -> dict:
        calls.append(url)
        return _fetcher(url, timeout)

    first, first_metadata = load_or_download_dvol(
        config, "BTC", fetch_json=counted_fetcher
    )
    second, second_metadata = load_or_download_dvol(
        config, "BTC", fetch_json=counted_fetcher
    )
    assert len(calls) == 1
    assert not first_metadata["cached"]
    assert second_metadata["cached"]
    assert first.equals(second)
    assert first_metadata["observations_sha256"] == second_metadata["observations_sha256"]


def test_features_have_frozen_two_day_execution_lag(tmp_path):
    config = _config(tmp_path)
    frames = {
        currency: load_or_download_dvol(config, currency, fetch_json=_fetcher)[0]
        for currency in ("BTC", "ETH")
    }
    features = build_causal_dvol_features(frames, 10, 5)
    assert (
        features.index.to_series(index=features.index)
        - features["source_candle_timestamp"]
        == pd.Timedelta(days=2)
    ).all()
    assert (features["source_final_at"] < features.index.to_series()).all()
    assert not any("sol" in column.lower() for column in features.columns)


def test_fixture_pipeline_is_audited_and_reproducible_from_cache(tmp_path):
    config = _config(tmp_path)
    first = run_dvol_pipeline(config, fetch_json=_fetcher)

    def forbidden_fetcher(url: str, timeout: float) -> dict:
        raise AssertionError("cache miss on reproducibility run")

    second = run_dvol_pipeline(config, fetch_json=forbidden_fetcher)
    output = tmp_path / "output"
    assert first["audit"]["passed"]
    assert second["audit"]["passed"]
    assert all(record["cached"] for record in second["manifest"]["records"])
    assert (output / "dvol_daily.parquet").is_file()
    assert (output / "dvol_features.parquet").is_file()
    assert (output / "manifest.json").is_file()
    assert (output / "catalog.json").is_file()
    assert (output / "audit.json").is_file()
    assert (output / "report.md").is_file()
