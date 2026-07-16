from copy import deepcopy
import hashlib
import zipfile

import numpy as np
import pandas as pd

from tlm.derivatives_data import (
    aggregate_funding,
    aggregate_metrics,
    aggregate_premium,
    assemble_daily_derivatives,
    audit_derivatives_dataset,
    build_archive_specs,
    generate_derivatives_fixture,
    parse_checksum,
    read_zip_csv,
    run_derivatives_pipeline,
)


def fixture_config(tmp_path) -> dict:
    return {
        "seed": 17,
        "derivatives": {
            "source": "fixture",
            "base_url": "https://data.binance.vision",
            "start": "2022-01-01",
            "end": "2022-03-01",
            "raw_dir": str(tmp_path / "raw"),
            "symbols": {
                "BTC": "BTCUSDT",
                "ETH": "ETHUSDT",
                "SOL": "SOLUSDT",
            },
            "workers": 2,
            "timeout_seconds": 1,
            "minimum_daily_coverage": 1.0,
            "minimum_derived_coverage": 1.0,
        },
        "output_dir": str(tmp_path / "derivatives"),
    }


def test_archive_specs_use_registered_official_paths(tmp_path):
    config = fixture_config(tmp_path)
    config["derivatives"].update({
        "source": "binance_public_archive",
        "start": "2021-12-31",
        "end": "2022-01-01",
        "symbols": {"BTC": "BTCUSDT"},
    })
    specs = build_archive_specs(config)
    assert len(specs) == 6
    assert any(
        spec.url.endswith(
            "/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2021-12.zip"
        )
        for spec in specs
    )
    assert any(
        spec.url.endswith(
            "/monthly/premiumIndexKlines/BTCUSDT/1d/BTCUSDT-1d-2022-01.zip"
        )
        for spec in specs
    )
    assert any(
        spec.url.endswith(
            "/daily/metrics/BTCUSDT/BTCUSDT-metrics-2022-01-01.zip"
        )
        for spec in specs
    )


def test_checksum_parser_accepts_sha256_and_rejects_invalid_payload():
    digest = hashlib.sha256(b"archive").hexdigest()
    assert parse_checksum(f"{digest}  file.zip\n".encode()) == digest
    try:
        parse_checksum(b"not-a-checksum")
    except ValueError as error:
        assert "Invalid SHA-256" in str(error)
    else:
        raise AssertionError("invalid checksum payload was accepted")


def test_headerless_legacy_premium_archive_uses_explicit_schema(tmp_path):
    archive_path = tmp_path / "legacy.zip"
    row = "1640995200000,0,0.001,-0.001,0.0002,0,1641081599999,0,17280,0,0,0\n"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("legacy.csv", row)
    frame = read_zip_csv(archive_path, "premiumIndexKlines")
    assert len(frame) == 1
    assert frame.loc[0, "open_time"] == 1640995200000
    assert np.isclose(frame.loc[0, "close"], 0.0002)


def test_raw_derivatives_are_aggregated_to_a_causal_daily_row():
    day = pd.Timestamp("2022-01-01", tz="UTC")
    funding_times = day + pd.to_timedelta([0, 8, 16], unit="h")
    funding = aggregate_funding(pd.DataFrame({
        "calc_time": funding_times.astype("int64") // 1_000_000,
        "last_funding_rate": [0.0001, -0.0002, 0.0003],
    }))
    premium = aggregate_premium(pd.DataFrame({
        "open_time": [day.value // 1_000_000],
        "open": [0.0010],
        "high": [0.0020],
        "low": [-0.0010],
        "close": [0.0005],
        "close_time": [
            (day + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)).value
            // 1_000_000
        ],
    }))
    metrics = aggregate_metrics(pd.DataFrame({
        "create_time": ["2022-01-01 00:00:00", "2022-01-01 23:55:00"],
        "sum_open_interest": [100.0, 110.0],
        "sum_open_interest_value": [10_000.0, 12_000.0],
        "count_toptrader_long_short_ratio": [1.0, 1.2],
        "sum_toptrader_long_short_ratio": [1.1, 1.3],
        "count_long_short_ratio": [0.9, 1.1],
        "sum_taker_long_short_vol_ratio": [0.8, 1.2],
    }))
    daily = assemble_daily_derivatives(funding, premium, metrics)
    row = daily.loc[day]
    assert np.isclose(row["funding_rate_sum"], 0.0002)
    assert row["funding_events"] == 3
    assert row["open_interest_first"] == 100.0
    assert row["open_interest_last"] == 110.0
    assert row["metrics_samples"] == 2
    assert row["basis_close"] == 0.0005
    assert row["source_max_timestamp"] < row["execution_open"]


def test_audit_rejects_a_source_observed_at_execution_open(tmp_path):
    config = fixture_config(tmp_path)
    frames = generate_derivatives_fixture(
        list(config["derivatives"]["symbols"]), days=60, seed=5
    )
    corrupted = deepcopy(frames)
    date = corrupted["BTC"].index[40]
    corrupted["BTC"].loc[date, "source_max_timestamp"] = (
        corrupted["BTC"].loc[date, "execution_open"]
    )
    records = [
        {"checksum_verified": True}
        for _ in range(len(config["derivatives"]["symbols"]) * 3)
    ]
    manifest = {"archive_count": len(records), "records": records}
    audit = audit_derivatives_dataset(corrupted, manifest, config)
    assert not audit["passed"]
    assert not audit["assets"]["BTC"]["checks"]["sources_precede_execution_open"]


def test_fixture_pipeline_writes_complete_audited_dataset(tmp_path):
    config = fixture_config(tmp_path)
    result = run_derivatives_pipeline(config)
    output = tmp_path / "derivatives"
    assert result["audit"]["passed"]
    assert result["archive_count"] == 9
    assert all(asset["coverage"] == 1.0 for asset in result["assets"].values())
    assert all(
        values["derived_coverage_after_warmup"] == 1.0
        for values in result["audit"]["assets"].values()
    )
    assert (output / "manifest.json").is_file()
    assert (output / "catalog.json").is_file()
    assert (output / "audit.json").is_file()
    assert (output / "derivatives_daily.parquet").is_file()
    assert (output / "assets" / "BTC.parquet").is_file()
    assert (output / "report.md").is_file()
