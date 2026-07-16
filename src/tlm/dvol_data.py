from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import yaml

from .data_family_audit import (
    JsonFetcher,
    _http_get_json,
    audit_dvol_frame,
    download_dvol,
    parse_dvol_response,
)


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def combine_dvol_payloads(
    payloads: list[dict], currency: str, start: str, end: str
) -> pd.DataFrame:
    if not payloads:
        raise ValueError(f"No cached DVOL payloads for {currency}")
    frames = [parse_dvol_response(payload, currency)[0] for payload in payloads]
    combined = pd.concat(frames).sort_index()
    if combined.index.has_duplicates:
        duplicates = combined.index[combined.index.duplicated()].unique()
        for timestamp in duplicates:
            rows = combined.loc[[timestamp]]
            if not rows.eq(rows.iloc[0]).all(axis=None):
                raise ValueError(f"Conflicting DVOL rows for {currency} {timestamp}")
        combined = combined.loc[~combined.index.duplicated(keep="first")]
    return combined.loc[
        pd.Timestamp(start, tz="UTC"):pd.Timestamp(end, tz="UTC")
    ]


def load_or_download_dvol(
    config: dict,
    currency: str,
    force: bool = False,
    fetch_json: JsonFetcher = _http_get_json,
) -> tuple[pd.DataFrame, dict[str, object]]:
    dvol = config["dvol"]
    cache_dir = Path(dvol["raw_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    identity = (
        f"{currency}_{dvol['start']}_{dvol['end']}_{dvol['resolution']}"
        .replace(":", "-")
    )
    cache_path = cache_dir / f"{identity}.json"
    cached = cache_path.is_file() and not force
    if cached:
        payloads = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        _, payloads = download_dvol(
            endpoint=dvol["endpoint"],
            currency=currency,
            start=dvol["start"],
            end=dvol["end"],
            resolution=dvol["resolution"],
            timeout=float(dvol["timeout_seconds"]),
            fetch_json=fetch_json,
        )
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payloads, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(cache_path)
    if not isinstance(payloads, list):
        raise ValueError(f"Invalid cached DVOL payload container for {currency}")
    frame = combine_dvol_payloads(payloads, currency, dvol["start"], dvol["end"])
    observation_rows = [
        [int(timestamp.timestamp() * 1000), *map(float, row)]
        for timestamp, row in zip(frame.index, frame.to_numpy(), strict=True)
    ]
    metadata = {
        "currency": currency,
        "endpoint": dvol["endpoint"],
        "cache_path": str(cache_path),
        "cached": cached,
        "page_count": len(payloads),
        "raw_payload_sha256": _sha256(_canonical_json_bytes(payloads)),
        "observations_sha256": _sha256(_canonical_json_bytes(observation_rows)),
        "rows": len(frame),
    }
    return frame, metadata


def build_causal_dvol_features(
    frames: dict[str, pd.DataFrame],
    rolling_z_window: int,
    change_vol_window: int,
) -> pd.DataFrame:
    required = {"BTC", "ETH"}
    if set(frames) != required:
        raise ValueError(f"DVOL market context requires exactly {sorted(required)}")
    common = frames["BTC"].index.intersection(frames["ETH"].index).sort_values()
    if len(common) < max(rolling_z_window, change_vol_window) + 2:
        raise ValueError("Insufficient common DVOL history for registered features")
    features = pd.DataFrame(index=common)
    changes: dict[str, pd.Series] = {}
    for currency in sorted(frames):
        frame = frames[currency].loc[common]
        prefix = f"market__{currency.lower()}_dvol"
        close = frame["close"]
        change = np.log(close / close.shift(1))
        intraday_range = np.log(frame["high"] / frame["low"])
        close_mean = close.rolling(rolling_z_window, min_periods=rolling_z_window).mean()
        close_std = close.rolling(rolling_z_window, min_periods=rolling_z_window).std(ddof=0)
        range_mean = intraday_range.rolling(
            rolling_z_window, min_periods=rolling_z_window
        ).mean()
        range_std = intraday_range.rolling(
            rolling_z_window, min_periods=rolling_z_window
        ).std(ddof=0)
        features[f"{prefix}_close"] = close
        features[f"{prefix}_log_change_1d"] = change
        features[f"{prefix}_intraday_log_range"] = intraday_range
        features[f"{prefix}_close_z{rolling_z_window}"] = (
            (close - close_mean) / close_std.replace(0.0, np.nan)
        )
        features[f"{prefix}_range_z{rolling_z_window}"] = (
            (intraday_range - range_mean) / range_std.replace(0.0, np.nan)
        )
        features[f"{prefix}_change_vol{change_vol_window}"] = change.rolling(
            change_vol_window, min_periods=change_vol_window
        ).std(ddof=0)
        changes[currency] = change
    features["market__dvol_mean_close"] = (
        frames["BTC"].loc[common, "close"] + frames["ETH"].loc[common, "close"]
    ) / 2.0
    features["market__dvol_mean_log_change_1d"] = (
        changes["BTC"] + changes["ETH"]
    ) / 2.0
    features["market__dvol_close_dispersion"] = (
        frames["BTC"].loc[common, "close"] - frames["ETH"].loc[common, "close"]
    ).abs()
    features["market__dvol_change_dispersion"] = (
        changes["BTC"] - changes["ETH"]
    ).abs()
    features = features.dropna(how="any")
    if features.empty:
        raise ValueError("Registered DVOL features are empty after warmup")
    source_candle_timestamp = features.index
    features.insert(0, "source_candle_timestamp", source_candle_timestamp)
    features.insert(1, "source_final_at", source_candle_timestamp + pd.Timedelta(days=1))
    features.index = source_candle_timestamp + pd.Timedelta(days=2)
    features.index.name = "execution_open"
    return features


def audit_dvol_dataset(
    raw_frames: dict[str, pd.DataFrame],
    features: pd.DataFrame,
    manifest: dict,
    config: dict,
) -> dict[str, object]:
    dvol = config["dvol"]
    asset_audits = {
        currency: audit_dvol_frame(
            frame, dvol["start"], dvol["end"], float(dvol["minimum_daily_coverage"])
        )
        for currency, frame in raw_frames.items()
    }
    feature_columns = [
        column for column in features.columns
        if column not in {"source_candle_timestamp", "source_final_at"}
    ]
    values = features[feature_columns]
    checks = {
        "exact_currency_set": set(raw_frames) == {"BTC", "ETH"},
        "all_raw_assets_pass": all(row["passed"] for row in asset_audits.values()),
        "raw_hashes_recorded": all(
            len(record["raw_payload_sha256"]) == 64
            and len(record["observations_sha256"]) == 64
            for record in manifest["records"]
        ),
        "feature_schema_matches_registration": feature_columns
        == list(dvol["feature_columns"]),
        "feature_index_unique_sorted": bool(
            features.index.is_monotonic_increasing and not features.index.has_duplicates
        ),
        "feature_values_finite": bool(np.isfinite(values.to_numpy()).all()),
        "source_final_strictly_precedes_execution": bool(
            (features["source_final_at"] < features.index.to_series(index=features.index)).all()
        ),
        "source_candle_to_execution_is_two_days": bool(
            (
                features.index.to_series(index=features.index)
                - features["source_candle_timestamp"]
                == pd.Timedelta(days=2)
            ).all()
        ),
        "no_sol_specific_series": not any("sol" in column.lower() for column in features),
    }
    return {
        "assets": asset_audits,
        "feature_rows": len(features),
        "feature_count": len(feature_columns),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _catalog(feature_columns: list[str], config: dict) -> dict[str, object]:
    return {
        "family": "options_implied_volatility",
        "source": "Deribit DVOL",
        "source_url": config["dvol"]["endpoint"],
        "currencies": ["BTC", "ETH"],
        "scope": "market_level_context_for_btc_eth_sol_decisions",
        "sol_specific_series": False,
        "timestamp_contract": {
            "raw_candle": "[t, t+1)",
            "final_at": "t+1",
            "first_allowed_execution_open": "t+2",
        },
        "feature_columns": feature_columns,
        "missing_data_policy": "no_forward_fill_complete_common_rows_only",
    }


def _report(result: dict) -> str:
    records = {row["currency"]: row for row in result["manifest"]["records"]}
    lines = [
        "# TLM v14 Options Volatility Data Layer",
        "",
        "## Result",
        "",
        f"- Audit: **{'PASS' if result['audit']['passed'] else 'FAIL'}**",
        f"- Raw window: **{result['start']} through {result['end']}**",
        f"- Causal feature rows after warmup: **{result['audit']['feature_rows']}**",
        f"- Registered feature columns: **{result['audit']['feature_count']}**",
        "",
        "## Source records",
        "",
    ]
    for currency in ("BTC", "ETH"):
        record = records[currency]
        asset = result["audit"]["assets"][currency]
        lines.append(
            f"- {currency}: {asset['rows']}/{asset['expected_rows']} rows "
            f"({asset['coverage']:.2%}); pages={record['page_count']}; "
            f"cache={'hit' if record['cached'] else 'miss'}; observations SHA-256 "
            f"`{record['observations_sha256']}`."
        )
    lines.extend([
        "",
        "## Chronology",
        "",
        "A daily DVOL candle stamped `t` covers `[t, t+1)` and is final at `t+1`. The feature table is indexed at `t+2`, so every source finalization is strictly earlier than the permitted execution open. Missing observations are never forward-filled.",
        "",
        "## Decision",
        "",
        "V14 is accepted as data infrastructure only. It creates named BTC/ETH options-volatility market context and no fabricated SOL series. It does not evaluate returns, a signal, a model, a threshold, or a policy. V15 may pre-register a policy-free signal-existence study using these frozen columns.",
        "",
    ])
    return "\n".join(lines)


def run_dvol_pipeline(
    config: dict,
    force: bool = False,
    fetch_json: JsonFetcher = _http_get_json,
) -> dict[str, object]:
    dvol = config["dvol"]
    raw_frames: dict[str, pd.DataFrame] = {}
    records: list[dict[str, object]] = []
    for currency in dvol["currencies"]:
        frame, metadata = load_or_download_dvol(
            config, currency, force=force, fetch_json=fetch_json
        )
        raw_frames[currency] = frame
        records.append(metadata)
    records = sorted(records, key=lambda row: str(row["currency"]))
    features = build_causal_dvol_features(
        raw_frames,
        rolling_z_window=int(dvol["rolling_z_window"]),
        change_vol_window=int(dvol["change_vol_window"]),
    )
    manifest = {
        "source": "deribit_public_dvol_api",
        "endpoint": dvol["endpoint"],
        "start": dvol["start"],
        "end": dvol["end"],
        "resolution": dvol["resolution"],
        "records": records,
    }
    audit = audit_dvol_dataset(raw_frames, features, manifest, config)
    if not audit["passed"]:
        raise RuntimeError(f"DVOL data audit failed: {audit['checks']}")
    output = Path(config["output_dir"])
    assets_dir = output / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    for currency, frame in raw_frames.items():
        frame.to_parquet(assets_dir / f"{currency}.parquet")
    wide = pd.concat(
        {currency: frame for currency, frame in sorted(raw_frames.items())}, axis=1
    )
    wide.to_parquet(output / "dvol_daily.parquet")
    features.to_parquet(output / "dvol_features.parquet")
    catalog = _catalog(
        [
            column for column in features.columns
            if column not in {"source_candle_timestamp", "source_final_at"}
        ],
        config,
    )
    result: dict[str, object] = {
        "version": "v14",
        "method": "cache_first_hash_audited_causal_dvol_data_layer",
        "start": dvol["start"],
        "end": dvol["end"],
        "manifest": manifest,
        "catalog": catalog,
        "audit": audit,
        "decision": "authorize_v15_policy_free_signal_existence_registration",
        "output_dir": str(output),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "catalog.json").write_text(
        json.dumps(catalog, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    return result
