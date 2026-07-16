from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import yaml

from .data import _verified_ssl_context


@dataclass(frozen=True)
class ArchiveSpec:
    asset: str
    symbol: str
    dataset: str
    period: str
    url: str
    cache_path: Path


BASE_REQUIRED_COLUMNS = (
    "funding_rate_sum",
    "funding_rate_mean",
    "funding_rate_last",
    "funding_events",
    "basis_close",
    "basis_mean",
    "basis_range",
    "open_interest_last",
    "open_interest_value_last",
    "metrics_samples",
)

DERIVED_COLUMNS = (
    "funding_rate_7d_sum",
    "funding_rate_30d_mean",
    "basis_7d_mean",
    "basis_30d_z",
    "open_interest_log_change_1d",
    "open_interest_log_change_7d",
    "open_interest_value_log_change_1d",
)

ARCHIVE_COLUMNS = {
    "fundingRate": (
        "calc_time", "funding_interval_hours", "last_funding_rate",
    ),
    "premiumIndexKlines": (
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "count", "taker_buy_volume",
        "taker_buy_quote_volume", "ignore",
    ),
    "metrics": (
        "create_time", "symbol", "sum_open_interest",
        "sum_open_interest_value", "count_toptrader_long_short_ratio",
        "sum_toptrader_long_short_ratio", "count_long_short_ratio",
        "sum_taker_long_short_vol_ratio",
    ),
}


def _monthly_periods(start: str, end: str) -> list[str]:
    return [str(period) for period in pd.period_range(start, end, freq="M")]


def build_archive_specs(config: dict) -> list[ArchiveSpec]:
    derivatives = config["derivatives"]
    base_url = derivatives["base_url"].rstrip("/")
    raw_dir = Path(derivatives["raw_dir"])
    start = derivatives["start"]
    end = derivatives["end"]
    specs: list[ArchiveSpec] = []
    for asset, symbol in derivatives["symbols"].items():
        for month in _monthly_periods(start, end):
            funding_name = f"{symbol}-fundingRate-{month}.zip"
            specs.append(ArchiveSpec(
                asset=asset,
                symbol=symbol,
                dataset="fundingRate",
                period=month,
                url=(
                    f"{base_url}/data/futures/um/monthly/fundingRate/"
                    f"{symbol}/{funding_name}"
                ),
                cache_path=raw_dir / "fundingRate" / symbol / funding_name,
            ))
            premium_name = f"{symbol}-1d-{month}.zip"
            specs.append(ArchiveSpec(
                asset=asset,
                symbol=symbol,
                dataset="premiumIndexKlines",
                period=month,
                url=(
                    f"{base_url}/data/futures/um/monthly/premiumIndexKlines/"
                    f"{symbol}/1d/{premium_name}"
                ),
                cache_path=raw_dir / "premiumIndexKlines" / symbol / premium_name,
            ))
        for date in pd.date_range(start, end, freq="D"):
            period = str(date.date())
            metrics_name = f"{symbol}-metrics-{period}.zip"
            specs.append(ArchiveSpec(
                asset=asset,
                symbol=symbol,
                dataset="metrics",
                period=period,
                url=(
                    f"{base_url}/data/futures/um/daily/metrics/"
                    f"{symbol}/{metrics_name}"
                ),
                cache_path=raw_dir / "metrics" / symbol / metrics_name,
            ))
    return specs


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_checksum(payload: bytes) -> str:
    text = payload.decode("utf-8").strip()
    checksum = text.split()[0] if text else ""
    if len(checksum) != 64 or any(character not in "0123456789abcdefABCDEF" for character in checksum):
        raise ValueError(f"Invalid SHA-256 checksum payload: {text!r}")
    return checksum.lower()


def _http_get(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "tlm-derivatives-research/0.1"}
    )
    with urllib.request.urlopen(
        request, timeout=timeout, context=_verified_ssl_context()
    ) as response:
        return response.read()


def download_archive(
    spec: ArchiveSpec,
    force: bool,
    timeout: float,
) -> dict[str, object]:
    spec.cache_path.parent.mkdir(parents=True, exist_ok=True)
    checksum_path = spec.cache_path.with_suffix(spec.cache_path.suffix + ".CHECKSUM")
    cached = spec.cache_path.is_file() and checksum_path.is_file() and not force
    if cached:
        archive_payload = spec.cache_path.read_bytes()
        checksum_payload = checksum_path.read_bytes()
    else:
        archive_payload = _http_get(spec.url, timeout)
        checksum_payload = _http_get(spec.url + ".CHECKSUM", timeout)
    expected = parse_checksum(checksum_payload)
    actual = _sha256_bytes(archive_payload)
    if actual != expected:
        raise ValueError(f"Checksum mismatch for {spec.url}: {actual} != {expected}")
    if not cached:
        temporary_archive = spec.cache_path.with_suffix(spec.cache_path.suffix + ".tmp")
        temporary_checksum = checksum_path.with_suffix(checksum_path.suffix + ".tmp")
        temporary_archive.write_bytes(archive_payload)
        temporary_checksum.write_bytes(checksum_payload)
        temporary_archive.replace(spec.cache_path)
        temporary_checksum.replace(checksum_path)
    return {
        "asset": spec.asset,
        "symbol": spec.symbol,
        "dataset": spec.dataset,
        "period": spec.period,
        "url": spec.url,
        "cache_path": str(spec.cache_path),
        "bytes": len(archive_payload),
        "sha256": actual,
        "checksum_verified": True,
        "cached": cached,
    }


def download_archives(
    specs: list[ArchiveSpec],
    force: bool,
    workers: int,
    timeout: float,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_archive, spec, force, timeout): spec
            for spec in specs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            spec = futures[future]
            try:
                records.append(future.result())
            except urllib.error.HTTPError as error:
                raise RuntimeError(
                    f"Archive unavailable ({error.code}) for {spec.dataset} "
                    f"{spec.symbol} {spec.period}: {spec.url}"
                ) from error
            if completed % 250 == 0 or completed == len(specs):
                print(f"derivatives archives: {completed}/{len(specs)} verified", flush=True)
    return sorted(records, key=lambda row: (
        str(row["asset"]), str(row["dataset"]), str(row["period"])
    ))


def read_zip_csv(path: str | Path, dataset: str) -> pd.DataFrame:
    path = Path(path)
    if dataset not in ARCHIVE_COLUMNS:
        raise ValueError(f"Unknown archive dataset: {dataset}")
    columns = list(ARCHIVE_COLUMNS[dataset])
    with zipfile.ZipFile(path) as archive:
        csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected exactly one CSV in {path}, found {csv_names}")
        with archive.open(csv_names[0]) as handle:
            frame = pd.read_csv(handle, header=None, low_memory=False)
    if len(frame.columns) != len(columns):
        raise ValueError(
            f"Unexpected {dataset} schema in {path}: "
            f"expected {len(columns)} columns, found {len(frame.columns)}"
        )
    if frame.iloc[0].astype(str).tolist() == columns:
        frame = frame.iloc[1:].reset_index(drop=True)
    frame.columns = columns
    return frame


def _numeric_timestamp(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise")
    unit = "us" if float(numeric.abs().max()) >= 1e14 else "ms"
    return pd.to_datetime(numeric, unit=unit, utc=True)


def aggregate_funding(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"calc_time", "last_funding_rate"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Funding CSV missing columns: {sorted(required - set(frame.columns))}")
    work = frame.copy()
    work["timestamp"] = _numeric_timestamp(work["calc_time"])
    work["rate"] = pd.to_numeric(work["last_funding_rate"], errors="raise")
    work = work.sort_values("timestamp")
    work["date"] = work["timestamp"].dt.floor("D")
    grouped = work.groupby("date", sort=True)
    result = pd.DataFrame({
        "funding_rate_sum": grouped["rate"].sum(),
        "funding_rate_mean": grouped["rate"].mean(),
        "funding_rate_last": grouped["rate"].last(),
        "funding_events": grouped.size().astype(float),
        "funding_source_timestamp": grouped["timestamp"].max(),
    })
    return result


def aggregate_premium(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"open_time", "open", "high", "low", "close", "close_time"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Premium CSV missing columns: {sorted(required - set(frame.columns))}")
    work = frame.copy()
    work["timestamp"] = _numeric_timestamp(work["open_time"])
    work["source_timestamp"] = _numeric_timestamp(work["close_time"])
    for column in ("open", "high", "low", "close"):
        work[column] = pd.to_numeric(work[column], errors="raise")
    work["date"] = work["timestamp"].dt.floor("D")
    work["basis_mean_row"] = work[["open", "high", "low", "close"]].mean(axis=1)
    work["basis_range_row"] = work["high"] - work["low"]
    grouped = work.sort_values("timestamp").groupby("date", sort=True)
    return pd.DataFrame({
        "basis_close": grouped["close"].last(),
        "basis_mean": grouped["basis_mean_row"].mean(),
        "basis_range": grouped["basis_range_row"].max(),
        "premium_source_timestamp": grouped["source_timestamp"].max(),
    })


def aggregate_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "create_time", "sum_open_interest", "sum_open_interest_value",
        "count_toptrader_long_short_ratio", "sum_toptrader_long_short_ratio",
        "count_long_short_ratio", "sum_taker_long_short_vol_ratio",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"Metrics CSV missing columns: {sorted(required - set(frame.columns))}")
    work = frame.copy()
    work["timestamp"] = pd.to_datetime(work["create_time"], utc=True, errors="raise")
    numeric_columns = sorted(required - {"create_time"})
    for column in numeric_columns:
        work[column] = pd.to_numeric(work[column], errors="raise")
    work = work.sort_values("timestamp")
    work["date"] = work["timestamp"].dt.floor("D")
    grouped = work.groupby("date", sort=True)
    return pd.DataFrame({
        "open_interest_first": grouped["sum_open_interest"].first(),
        "open_interest_last": grouped["sum_open_interest"].last(),
        "open_interest_value_last": grouped["sum_open_interest_value"].last(),
        "toptrader_count_ratio_mean": grouped["count_toptrader_long_short_ratio"].mean(),
        "toptrader_position_ratio_mean": grouped["sum_toptrader_long_short_ratio"].mean(),
        "global_long_short_ratio_mean": grouped["count_long_short_ratio"].mean(),
        "taker_long_short_ratio_mean": grouped["sum_taker_long_short_vol_ratio"].mean(),
        "metrics_samples": grouped.size().astype(float),
        "metrics_source_timestamp": grouped["timestamp"].max(),
    })


def add_causal_derivatives_features(daily: pd.DataFrame) -> pd.DataFrame:
    result = daily.sort_index().copy()
    result["funding_rate_7d_sum"] = result["funding_rate_sum"].rolling(7).sum()
    result["funding_rate_30d_mean"] = result["funding_rate_mean"].rolling(30).mean()
    result["basis_7d_mean"] = result["basis_close"].rolling(7).mean()
    basis_mean = result["basis_close"].rolling(30).mean()
    basis_std = result["basis_close"].rolling(30).std().replace(0.0, np.nan)
    result["basis_30d_z"] = (result["basis_close"] - basis_mean) / basis_std
    log_oi = np.log(result["open_interest_last"].where(
        result["open_interest_last"] > 0.0
    ))
    log_oi_value = np.log(result["open_interest_value_last"].where(
        result["open_interest_value_last"] > 0.0
    ))
    result["open_interest_log_change_1d"] = log_oi.diff()
    result["open_interest_log_change_7d"] = log_oi.diff(7)
    result["open_interest_value_log_change_1d"] = log_oi_value.diff()
    source_columns = [
        "funding_source_timestamp", "premium_source_timestamp",
        "metrics_source_timestamp",
    ]
    result["source_max_timestamp"] = result[source_columns].max(axis=1)
    result["execution_open"] = result.index + pd.Timedelta(days=1)
    return result.replace([np.inf, -np.inf], np.nan)


def assemble_daily_derivatives(
    funding: pd.DataFrame,
    premium: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    daily = funding.join(premium, how="outer").join(metrics, how="outer")
    if daily.index.has_duplicates or not daily.index.is_monotonic_increasing:
        daily = daily.sort_index()
        if daily.index.has_duplicates:
            raise ValueError("Daily derivatives index contains duplicates")
    return add_causal_derivatives_features(daily)


def parse_asset_archives(specs: list[ArchiveSpec]) -> pd.DataFrame:
    by_dataset: dict[str, list[pd.DataFrame]] = {
        "fundingRate": [], "premiumIndexKlines": [], "metrics": [],
    }
    for spec in sorted(specs, key=lambda item: (item.dataset, item.period)):
        by_dataset[spec.dataset].append(read_zip_csv(spec.cache_path, spec.dataset))
    if any(not frames for frames in by_dataset.values()):
        raise ValueError("Missing one or more required archive datasets")
    return assemble_daily_derivatives(
        aggregate_funding(pd.concat(by_dataset["fundingRate"], ignore_index=True)),
        aggregate_premium(pd.concat(by_dataset["premiumIndexKlines"], ignore_index=True)),
        aggregate_metrics(pd.concat(by_dataset["metrics"], ignore_index=True)),
    )


def generate_derivatives_fixture(
    assets: list[str],
    start: str = "2022-01-01",
    days: int = 180,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=days, freq="D", tz="UTC")
    result: dict[str, pd.DataFrame] = {}
    for offset, asset in enumerate(assets):
        funding = rng.normal(0.00005, 0.0002, days)
        basis = rng.normal(0.0002, 0.0015, days)
        oi = 1_000_000.0 * (offset + 1) * np.exp(np.cumsum(rng.normal(0.0, 0.015, days)))
        daily = pd.DataFrame(index=index)
        daily["funding_rate_sum"] = funding
        daily["funding_rate_mean"] = funding / 3.0
        daily["funding_rate_last"] = funding / 3.0
        daily["funding_events"] = 3.0
        daily["funding_source_timestamp"] = index + pd.Timedelta(hours=16)
        daily["basis_close"] = basis
        daily["basis_mean"] = basis
        daily["basis_range"] = np.abs(rng.normal(0.002, 0.0005, days))
        daily["premium_source_timestamp"] = index + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)
        daily["open_interest_first"] = oi * (1.0 + rng.normal(0.0, 0.002, days))
        daily["open_interest_last"] = oi
        daily["open_interest_value_last"] = oi * (100.0 + offset * 20.0)
        daily["toptrader_count_ratio_mean"] = rng.normal(1.1, 0.1, days)
        daily["toptrader_position_ratio_mean"] = rng.normal(1.05, 0.1, days)
        daily["global_long_short_ratio_mean"] = rng.normal(1.0, 0.08, days)
        daily["taker_long_short_ratio_mean"] = rng.normal(1.0, 0.1, days)
        daily["metrics_samples"] = 288.0
        daily["metrics_source_timestamp"] = index + pd.Timedelta(hours=23, minutes=55)
        result[asset] = add_causal_derivatives_features(daily)
    return result


def audit_derivatives_dataset(
    frames: dict[str, pd.DataFrame],
    manifest: dict[str, object],
    config: dict,
) -> dict[str, object]:
    derivatives = config["derivatives"]
    expected_index = pd.date_range(
        derivatives["start"], derivatives["end"], freq="D", tz="UTC"
    )
    minimum_coverage = float(derivatives["minimum_daily_coverage"])
    minimum_derived_coverage = float(
        derivatives.get("minimum_derived_coverage", minimum_coverage)
    )
    asset_checks: dict[str, dict[str, object]] = {}
    for asset, frame in frames.items():
        expected_frame = frame.reindex(expected_index)
        complete = expected_frame[list(BASE_REQUIRED_COLUMNS)].notna().all(axis=1)
        coverage = float(complete.mean())
        warm = expected_frame.iloc[30:]
        derived_frame = warm[list(DERIVED_COLUMNS)]
        derived_complete = derived_frame.notna().all(axis=1)
        derived_coverage = float(derived_complete.mean())
        finite_derived = bool(
            derived_complete.any()
            and np.isfinite(derived_frame.loc[derived_complete].to_numpy()).all()
        )
        missing_dates = expected_frame.index[~complete].strftime("%Y-%m-%d").tolist()
        missing_columns = {
            column: int(count)
            for column, count in expected_frame.loc[
                ~complete, list(BASE_REQUIRED_COLUMNS)
            ].isna().sum().items()
            if count > 0
        }
        checks = {
            "unique_ordered_daily_index": bool(
                not frame.index.has_duplicates and frame.index.is_monotonic_increasing
            ),
            "daily_coverage_passes": coverage >= minimum_coverage,
            "derived_daily_coverage_passes": (
                derived_coverage >= minimum_derived_coverage
            ),
            "no_dates_outside_requested_range": bool(
                frame.index.difference(expected_index).empty
            ),
            "sources_precede_execution_open": bool(
                (expected_frame.loc[complete, "source_max_timestamp"]
                 < expected_frame.loc[complete, "execution_open"]).all()
            ),
            "base_values_are_finite": bool(
                np.isfinite(expected_frame.loc[
                    complete, list(BASE_REQUIRED_COLUMNS)
                ].to_numpy()).all()
            ),
            "derived_values_are_finite_after_warmup": finite_derived,
        }
        asset_checks[asset] = {
            "passed": all(checks.values()),
            "coverage": coverage,
            "derived_coverage_after_warmup": derived_coverage,
            "complete_days": int(complete.sum()),
            "expected_days": int(len(expected_index)),
            "missing_dates": missing_dates,
            "missing_base_columns": missing_columns,
            "first_complete_date": (
                str(expected_frame.index[complete].min().date()) if complete.any() else None
            ),
            "last_complete_date": (
                str(expected_frame.index[complete].max().date()) if complete.any() else None
            ),
            "checks": checks,
        }
    manifest_records = manifest.get("records", [])
    global_checks = {
        "all_assets_present": set(frames) == set(derivatives["symbols"]),
        "all_asset_audits_pass": all(values["passed"] for values in asset_checks.values()),
        "all_archive_checksums_verified": bool(manifest_records) and all(
            bool(record["checksum_verified"]) for record in manifest_records
        ),
        "manifest_count_matches": len(manifest_records) == int(manifest["archive_count"]),
    }
    audit = {
        "passed": all(global_checks.values()),
        "checks": global_checks,
        "assets": asset_checks,
    }
    return audit


def run_derivatives_pipeline(config: dict, force: bool = False) -> dict[str, object]:
    derivatives = config["derivatives"]
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    if derivatives["source"] == "fixture":
        frames = generate_derivatives_fixture(
            list(derivatives["symbols"]),
            start=derivatives["start"],
            days=len(pd.date_range(derivatives["start"], derivatives["end"], freq="D")),
            seed=int(config.get("seed", 42)),
        )
        records = [
            {
                "asset": asset,
                "symbol": symbol,
                "dataset": dataset,
                "period": "fixture",
                "url": "fixture://local",
                "cache_path": "fixture",
                "bytes": 0,
                "sha256": "0" * 64,
                "checksum_verified": True,
                "cached": True,
            }
            for asset, symbol in derivatives["symbols"].items()
            for dataset in ("fundingRate", "premiumIndexKlines", "metrics")
        ]
        specs: list[ArchiveSpec] = []
    elif derivatives["source"] == "binance_public_archive":
        specs = build_archive_specs(config)
        records = download_archives(
            specs,
            force=force,
            workers=int(derivatives["workers"]),
            timeout=float(derivatives["timeout_seconds"]),
        )
        frames = {}
        for asset in derivatives["symbols"]:
            frame = parse_asset_archives([
                spec for spec in specs if spec.asset == asset
            ])
            start = pd.Timestamp(derivatives["start"], tz="UTC")
            end = pd.Timestamp(derivatives["end"], tz="UTC")
            frames[asset] = frame.loc[(frame.index >= start) & (frame.index <= end)]
    else:
        raise ValueError(f"Unsupported derivatives source: {derivatives['source']}")

    manifest = {
        "source": derivatives["source"],
        "archive_count": len(records),
        "requested_start": derivatives["start"],
        "requested_end": derivatives["end"],
        "records": records,
    }
    with (output / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    asset_dir = output / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    for asset, frame in frames.items():
        frame.to_parquet(asset_dir / f"{asset}.parquet")
    combined = pd.concat(frames, axis=1, join="inner")
    combined.columns = [f"{asset}__{column}" for asset, column in combined.columns]
    combined.to_parquet(output / "derivatives_daily.parquet")

    catalog = {
        "source": "Binance public USD-M futures archive",
        "funding": {
            "archive_frequency": "monthly",
            "raw_frequency": "funding event, typically 8h",
            "timestamp": "calc_time",
        },
        "basis": {
            "archive_frequency": "monthly",
            "raw_frequency": "1d premium index kline",
            "timestamp": "close_time",
        },
        "open_interest": {
            "archive_frequency": "daily",
            "raw_frequency": "5m metrics sample",
            "timestamp": "create_time",
        },
        "availability_rule": "source_max_timestamp < next UTC open",
        "no_forward_fill": True,
    }
    with (output / "catalog.json").open("w", encoding="utf-8") as handle:
        json.dump(catalog, handle, indent=2, sort_keys=True)
    audit = audit_derivatives_dataset(frames, manifest, config)
    with (output / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
    if not audit["passed"]:
        raise RuntimeError(f"Derivatives dataset audit failed: {audit}")

    lines = [
        "# TLM v10 derivatives data report",
        "",
        f"- Source: **{derivatives['source']}**",
        f"- Requested range: **{derivatives['start']} to {derivatives['end']}**",
        f"- Verified archives: **{len(records)}**",
        f"- Dataset audit passed: **{audit['passed']}**",
        "- Policy/model experiment: **none**",
        "",
        "## Daily causal coverage",
        "",
        "| asset | complete days | base coverage | derived coverage | missing dates | passed |",
        "|:--|--:|--:|--:|:--|:--:|",
    ]
    for asset, values in audit["assets"].items():
        lines.append(
            f"| {asset} | {values['complete_days']}/{values['expected_days']} | "
            f"{values['coverage']:.2%} | "
            f"{values['derived_coverage_after_warmup']:.2%} | "
            f"{', '.join(values['missing_dates']) or 'none'} | "
            f"{values['passed']} |"
        )
    lines.extend([
        "",
        "## Timestamp contract",
        "",
        "Funding is aggregated from events whose `calc_time` falls inside UTC day t. Basis uses the daily premium-index candle closing at the end of t. Open interest and positioning use the final 5-minute metrics sample observed during t. Every daily row records the maximum raw source timestamp and must satisfy `source_max_timestamp < open(t+1)`.",
        "",
        "## Next gate",
        "",
        "A derivatives experiment may use only rows that clear the registered coverage thresholds. Preserve the listed source gaps as missing, keep checksums and timestamps auditable, and do not forward-fill them.",
    ])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "audit": audit,
        "archive_count": len(records),
        "assets": {
            asset: {
                "rows": int(len(frame)),
                "complete_days": audit["assets"][asset]["complete_days"],
                "coverage": audit["assets"][asset]["coverage"],
            }
            for asset, frame in frames.items()
        },
        "output_dir": str(output),
    }
