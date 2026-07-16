from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .derivatives_data import read_zip_csv
from .derivatives_signal_study import run_derivatives_signal_study


PATH_FEATURE_COLUMNS = (
    "taker_ratio_last",
    "taker_ratio_log_change",
    "taker_ratio_slope",
    "taker_ratio_autocorr_1",
    "taker_buy_fraction",
    "taker_ratio_last_hour_mean",
    "taker_ratio_reversal_1h",
    "oi_log_change_intraday",
    "oi_range_intraday",
    "oi_max_drawdown_intraday",
    "oi_slope_intraday",
    "oi_last_hour_log_change",
    "oi_value_log_change_intraday",
)

TAKER_PATH_COLUMNS = PATH_FEATURE_COLUMNS[:7]
OI_PATH_COLUMNS = PATH_FEATURE_COLUMNS[7:]


def _log_ratio(last: float, first: float) -> float:
    if last <= 0.0 or first <= 0.0:
        return float("nan")
    return float(np.log(last / first))


def _normalized_slope(values: np.ndarray, log_values: bool = False) -> float:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return float("nan")
    finite = np.isfinite(values)
    if int(finite.sum()) < 2:
        return float("nan")
    full_x = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    values = values[finite]
    x = full_x[finite]
    if log_values:
        if (values <= 0.0).any():
            return float("nan")
        values = np.log(values)
    centered_x = x - x.mean()
    denominator = float(np.square(centered_x).sum())
    if denominator <= 0.0:
        return float("nan")
    return float(np.dot(centered_x, values - values.mean()) / denominator)


def _finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if len(finite) else float("nan")


def aggregate_intraday_metrics(
    frame: pd.DataFrame,
    minimum_samples: int = 250,
) -> pd.DataFrame:
    required = {
        "create_time", "sum_open_interest", "sum_open_interest_value",
        "sum_taker_long_short_vol_ratio",
    }
    if not required.issubset(frame.columns):
        raise ValueError(
            f"Metrics path CSV missing columns: {sorted(required - set(frame.columns))}"
        )
    work = frame.copy()
    work["timestamp"] = pd.to_datetime(work["create_time"], utc=True, errors="raise")
    if work["timestamp"].duplicated().any():
        raise ValueError("Metrics path data contains duplicate timestamps")
    numeric = {
        "sum_open_interest": "oi",
        "sum_open_interest_value": "oi_value",
        "sum_taker_long_short_vol_ratio": "taker_ratio",
    }
    for source, target in numeric.items():
        work[target] = pd.to_numeric(work[source], errors="coerce")
    work = work.sort_values("timestamp")
    work["date"] = work["timestamp"].dt.floor("D")
    rows: list[dict[str, object]] = []
    for date, group in work.groupby("date", sort=True):
        group = group.sort_values("timestamp")
        taker = group["taker_ratio"].to_numpy(dtype=np.float64)
        oi = group["oi"].to_numpy(dtype=np.float64)
        oi_value = group["oi_value"].to_numpy(dtype=np.float64)
        taker_valid = np.isfinite(taker)
        oi_valid = np.isfinite(oi) & np.isfinite(oi_value)
        taker_observed = taker[taker_valid]
        oi_observed = oi[oi_valid]
        oi_value_observed = oi_value[oi_valid]
        last_hour_start = max(0, len(group) - 12)
        first_hour_end = min(12, len(group))
        taker_last_hour = taker[last_hour_start:]
        taker_first_hour = taker[:first_hour_end]
        taker_std = float(taker_observed.std()) if len(taker_observed) else 0.0
        taker_autocorr = (
            float(pd.Series(taker_observed).autocorr(lag=1))
            if len(taker_observed) >= 3 and taker_std > 0.0
            else float("nan")
        )
        taker_last_hour_mean = _finite_mean(taker_last_hour)
        taker_first_hour_mean = _finite_mean(taker_first_hour)
        oi_running_max = (
            np.maximum.accumulate(oi_observed)
            if len(oi_observed) else np.array([], dtype=np.float64)
        )
        row: dict[str, object] = {
            "date": date,
            "metrics_samples": float(len(group)),
            "taker_samples": float(taker_valid.sum()),
            "oi_samples": float(oi_valid.sum()),
            "source_max_timestamp": group["timestamp"].max(),
            "execution_open": date + pd.Timedelta(days=1),
            "taker_ratio_last": (
                float(taker_observed[-1]) if len(taker_observed) else float("nan")
            ),
            "taker_ratio_log_change": (
                _log_ratio(taker_observed[-1], taker_observed[0])
                if len(taker_observed) else float("nan")
            ),
            "taker_ratio_slope": _normalized_slope(taker),
            "taker_ratio_autocorr_1": taker_autocorr,
            "taker_buy_fraction": (
                float(np.mean(taker_observed > 1.0))
                if len(taker_observed) else float("nan")
            ),
            "taker_ratio_last_hour_mean": taker_last_hour_mean,
            "taker_ratio_reversal_1h": _log_ratio(
                taker_last_hour_mean, taker_first_hour_mean
            ),
            "oi_log_change_intraday": (
                _log_ratio(oi_observed[-1], oi_observed[0])
                if len(oi_observed) else float("nan")
            ),
            "oi_range_intraday": (
                float((oi_observed.max() - oi_observed.min()) / oi_observed[0])
                if len(oi_observed) and oi_observed[0] > 0.0 else float("nan")
            ),
            "oi_max_drawdown_intraday": (
                float(np.min(oi_observed / oi_running_max - 1.0))
                if len(oi_observed) and (oi_running_max > 0.0).all()
                else float("nan")
            ),
            "oi_slope_intraday": _normalized_slope(oi, log_values=True),
            "oi_last_hour_log_change": (
                _log_ratio(oi_observed[-1], oi_observed[max(0, len(oi_observed) - 12)])
                if len(oi_observed) else float("nan")
            ),
            "oi_value_log_change_intraday": (
                _log_ratio(oi_value_observed[-1], oi_value_observed[0])
                if len(oi_value_observed) else float("nan")
            ),
        }
        rows.append(row)
    result = pd.DataFrame(rows).set_index("date").sort_index()
    result.loc[
        result["taker_samples"] < float(minimum_samples), list(TAKER_PATH_COLUMNS)
    ] = np.nan
    result.loc[
        result["oi_samples"] < float(minimum_samples), list(OI_PATH_COLUMNS)
    ] = np.nan
    return result.replace([np.inf, -np.inf], np.nan)


def generate_intraday_path_fixture(
    assets: list[str],
    start: str,
    end: str,
    seed: int,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, end, freq="D", tz="UTC")
    frames: dict[str, pd.DataFrame] = {}
    for offset, asset in enumerate(assets):
        frame = pd.DataFrame(index=index)
        for column_index, column in enumerate(PATH_FEATURE_COLUMNS):
            scale = 0.02 + column_index * 0.002
            frame[column] = rng.normal(offset * 0.01, scale, len(index))
        frame["taker_buy_fraction"] = np.clip(
            rng.normal(0.5, 0.08, len(index)), 0.0, 1.0
        )
        frame["taker_ratio_last"] = np.exp(
            rng.normal(0.0, 0.15, len(index))
        )
        frame["taker_ratio_last_hour_mean"] = np.exp(
            rng.normal(0.0, 0.08, len(index))
        )
        frame["metrics_samples"] = 288.0
        frame["taker_samples"] = 288.0
        frame["oi_samples"] = 288.0
        frame["source_max_timestamp"] = index + pd.Timedelta(hours=23, minutes=55)
        frame["execution_open"] = index + pd.Timedelta(days=1)
        frames[asset] = frame
    return frames


def audit_intraday_path_frames(
    frames: dict[str, pd.DataFrame],
    config: dict,
    source_checksums_verified: bool,
    source_record_count: int,
) -> dict[str, object]:
    path_config = config["intraday_path"]
    expected = pd.date_range(
        path_config["start"], path_config["end"], freq="D", tz="UTC"
    )
    minimum_coverage = float(path_config["minimum_feature_coverage"])
    minimum_samples = int(path_config["minimum_intraday_samples"])
    assets: dict[str, dict[str, object]] = {}
    for asset, frame in frames.items():
        aligned = frame.reindex(expected)
        feature_coverage = {
            column: float(aligned[column].notna().mean())
            for column in PATH_FEATURE_COLUMNS
        }
        complete = aligned[list(PATH_FEATURE_COLUMNS)].notna().all(axis=1)
        coverage = min(feature_coverage.values())
        taker_observed = aligned[list(TAKER_PATH_COLUMNS)].notna().any(axis=1)
        oi_observed = aligned[list(OI_PATH_COLUMNS)].notna().any(axis=1)
        finite_values = aligned[list(PATH_FEATURE_COLUMNS)].to_numpy(dtype=float)
        checks = {
            "unique_ordered_daily_index": bool(
                not frame.index.has_duplicates and frame.index.is_monotonic_increasing
            ),
            "no_dates_outside_registered_range": frame.index.difference(expected).empty,
            "all_feature_coverages_pass": all(
                value >= minimum_coverage for value in feature_coverage.values()
            ),
            "observed_taker_rows_have_minimum_samples": bool(
                (aligned.loc[taker_observed, "taker_samples"] >= minimum_samples).all()
            ),
            "observed_oi_rows_have_minimum_samples": bool(
                (aligned.loc[oi_observed, "oi_samples"] >= minimum_samples).all()
            ),
            "observed_values_are_finite": bool(
                np.isfinite(finite_values[~np.isnan(finite_values)]).all()
            ),
            "sources_precede_execution_open": bool(
                (aligned.loc[taker_observed | oi_observed, "source_max_timestamp"]
                 < aligned.loc[taker_observed | oi_observed, "execution_open"]).all()
            ),
        }
        assets[asset] = {
            "passed": all(checks.values()),
            "coverage": coverage,
            "feature_coverage": feature_coverage,
            "complete_days": int(complete.sum()),
            "expected_days": int(len(expected)),
            "missing_dates": expected[~complete].strftime("%Y-%m-%d").tolist(),
            "checks": checks,
        }
    global_checks = {
        "all_assets_present": set(frames) == set(path_config["symbols"]),
        "all_asset_audits_pass": all(result["passed"] for result in assets.values()),
        "source_checksums_verified": source_checksums_verified,
        "source_records_present": source_record_count > 0,
    }
    return {
        "passed": all(global_checks.values()),
        "checks": global_checks,
        "assets": assets,
        "source_record_count": source_record_count,
    }


def _load_verified_metrics_frames(
    config: dict,
) -> tuple[dict[str, pd.DataFrame], int]:
    path_config = config["intraday_path"]
    manifest_path = Path(path_config["source_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [
        record for record in manifest["records"]
        if record["dataset"] == "metrics"
    ]
    expected_days = len(pd.date_range(
        path_config["start"], path_config["end"], freq="D"
    ))
    expected_records = expected_days * len(path_config["symbols"])
    if len(records) != expected_records:
        raise ValueError(
            f"Expected {expected_records} metrics archives, found {len(records)}"
        )
    frames: dict[str, pd.DataFrame] = {}
    completed = 0
    for asset in path_config["symbols"]:
        asset_records = sorted(
            (record for record in records if record["asset"] == asset),
            key=lambda record: record["period"],
        )
        raw_frames: list[pd.DataFrame] = []
        for record in asset_records:
            path = Path(record["cache_path"])
            payload = path.read_bytes()
            actual = hashlib.sha256(payload).hexdigest()
            if not record["checksum_verified"] or actual != record["sha256"]:
                raise ValueError(f"Metrics archive checksum mismatch: {path}")
            raw_frames.append(read_zip_csv(path, "metrics"))
            completed += 1
            if completed % 500 == 0 or completed == len(records):
                print(
                    f"intraday metrics archives: {completed}/{len(records)} verified",
                    flush=True,
                )
        frame = aggregate_intraday_metrics(
            pd.concat(raw_frames, ignore_index=True),
            minimum_samples=int(path_config["minimum_intraday_samples"]),
        )
        start = pd.Timestamp(path_config["start"], tz="UTC")
        end = pd.Timestamp(path_config["end"], tz="UTC")
        frames[asset] = frame.loc[(frame.index >= start) & (frame.index <= end)]
    return frames, len(records)


def run_intraday_path_pipeline(config: dict) -> dict[str, object]:
    path_config = config["intraday_path"]
    output = Path(config["output_dir"]) / "path_data"
    output.mkdir(parents=True, exist_ok=True)
    if path_config["source"] == "fixture":
        frames = generate_intraday_path_fixture(
            list(path_config["symbols"]),
            path_config["start"],
            path_config["end"],
            int(config.get("seed", 42)),
        )
        source_count = len(frames) * len(pd.date_range(
            path_config["start"], path_config["end"], freq="D"
        ))
        source_verified = True
    elif path_config["source"] == "binance_metrics_archive":
        frames, source_count = _load_verified_metrics_frames(config)
        source_verified = True
    else:
        raise ValueError(f"Unsupported intraday path source: {path_config['source']}")

    asset_dir = output / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    for asset, frame in frames.items():
        frame.to_parquet(asset_dir / f"{asset}.parquet")
    combined = pd.concat(frames, axis=1, join="inner")
    combined.columns = [f"{asset}__{column}" for asset, column in combined.columns]
    combined.to_parquet(output / "intraday_path_daily.parquet")
    audit = audit_intraday_path_frames(
        frames, config, source_verified, source_count
    )
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    catalog = {
        "source": path_config["source"],
        "raw_frequency": "5m",
        "daily_features": list(PATH_FEATURE_COLUMNS),
        "minimum_intraday_samples": int(path_config["minimum_intraday_samples"]),
        "minimum_feature_coverage": float(path_config["minimum_feature_coverage"]),
        "availability_rule": "source_max_timestamp < next UTC open",
        "no_forward_fill": True,
    }
    (output / "catalog.json").write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    lines = [
        "# TLM v12 intraday path data",
        "",
        f"- Source: **{path_config['source']}**",
        f"- Source records: **{source_count}**",
        f"- Registered features: **{len(PATH_FEATURE_COLUMNS)}**",
        f"- Audit passed: **{audit['passed']}**",
        "- Forward-fill: **disabled**",
        "",
        "| asset | all-feature days | minimum feature coverage | missing days | passed |",
        "|:--|--:|--:|--:|:--:|",
    ]
    for asset, values in audit["assets"].items():
        lines.append(
            f"| {asset} | {values['complete_days']}/{values['expected_days']} | "
            f"{values['coverage']:.2%} | {len(values['missing_dates'])} | "
            f"{values['passed']} |"
        )
    lines.extend([
        "",
        "Every path summary uses only 5-minute observations within UTC day t. The maximum raw timestamp must precede the next UTC open; incomplete days remain missing.",
    ])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not audit["passed"]:
        raise RuntimeError(f"Intraday path audit failed: {audit}")
    return {
        "audit": audit,
        "output_dir": str(output),
        "source_record_count": source_count,
        "features": list(PATH_FEATURE_COLUMNS),
    }


def run_intraday_path_signal_study(config: dict) -> dict[str, object]:
    path_result = run_intraday_path_pipeline(config)
    resolved = deepcopy(config)
    resolved["derivatives_signal_study"]["derivatives_artifact_dir"] = (
        path_result["output_dir"]
    )
    result = run_derivatives_signal_study(resolved)
    result["path_data"] = {
        "audit_passed": path_result["audit"]["passed"],
        "source_record_count": path_result["source_record_count"],
        "feature_count": len(path_result["features"]),
        "artifact_dir": path_result["output_dir"],
    }
    result_path = Path(config["output_dir"]) / "derivatives_signal_study.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
