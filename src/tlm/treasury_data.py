from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .treasury_feasibility import (
    BytesFetcher,
    _http_get_bytes,
    build_treasury_year_url,
    parse_treasury_csv,
)


def _observation_hash(frame: pd.DataFrame) -> str:
    rows = [
        [int(timestamp.timestamp()), *map(float, values)]
        for timestamp, values in zip(frame.index, frame.to_numpy(), strict=True)
    ]
    payload = json.dumps(rows, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_or_download_treasury_year(
    config: dict,
    year: int,
    force: bool = False,
    fetch_bytes: BytesFetcher = _http_get_bytes,
) -> tuple[pd.DataFrame, dict[str, object]]:
    treasury = config["treasury"]
    raw_dir = Path(treasury["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_path = raw_dir / f"daily_par_yield_curve_{year}.csv"
    cached = cache_path.is_file() and not force
    url_config = {"treasury_feasibility": {
        "csv_url_template": treasury["csv_url_template"]
    }}
    url = build_treasury_year_url(url_config, year)
    if cached:
        payload = cache_path.read_bytes()
    else:
        payload = fetch_bytes(url, float(treasury["timeout_seconds"]))
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(cache_path)
    frame = parse_treasury_csv(
        payload, year, ["Date", *treasury["source_columns"]]
    )
    return frame, {
        "year": year,
        "url": url,
        "cache_path": str(cache_path),
        "cached": cached,
        "bytes": len(payload),
        "raw_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "observations_sha256": _observation_hash(frame),
        "rows": len(frame),
    }


def build_treasury_source_features(
    raw: pd.DataFrame,
    rolling_window: int,
) -> pd.DataFrame:
    frame = pd.DataFrame(index=raw.index)
    two = raw["2 Yr"].astype(float)
    ten = raw["10 Yr"].astype(float)
    curve = ten - two
    two_change = two.diff()
    ten_change = ten.diff()
    curve_change = curve.diff()

    def rolling_z(values: pd.Series) -> pd.Series:
        mean = values.rolling(rolling_window, min_periods=rolling_window).mean()
        std = values.rolling(rolling_window, min_periods=rolling_window).std(ddof=0)
        return (values - mean) / std.replace(0.0, np.nan)

    frame["market__treasury_2y"] = two
    frame["market__treasury_10y"] = ten
    frame["market__treasury_curve_10y_2y"] = curve
    frame["market__treasury_2y_change"] = two_change
    frame["market__treasury_10y_change"] = ten_change
    frame["market__treasury_curve_change"] = curve_change
    frame[f"market__treasury_curve_z{rolling_window}"] = rolling_z(curve)
    frame[f"market__treasury_10y_z{rolling_window}"] = rolling_z(ten)
    frame[f"market__treasury_2y_change_vol{rolling_window}"] = two_change.rolling(
        rolling_window, min_periods=rolling_window
    ).std(ddof=0)
    frame[f"market__treasury_10y_change_vol{rolling_window}"] = ten_change.rolling(
        rolling_window, min_periods=rolling_window
    ).std(ddof=0)
    frame = frame.dropna(how="any")
    if frame.empty:
        raise ValueError("Treasury features are empty after registered warmup")
    return frame


def materialize_causal_treasury_state(
    source_features: pd.DataFrame,
    execution_lag_days: int,
    maximum_carry_days: int,
) -> pd.DataFrame:
    source = source_features.copy()
    source.insert(0, "source_date", source.index)
    source.insert(1, "source_final_at", source.index + pd.Timedelta(hours=21))
    source.insert(
        2, "source_eligible_at",
        source.index + pd.Timedelta(days=execution_lag_days),
    )
    source = source.reset_index(drop=True).sort_values("source_eligible_at")
    execution_dates = pd.date_range(
        source["source_eligible_at"].min(),
        source["source_eligible_at"].max(),
        freq="D", tz="UTC",
    )
    daily = pd.DataFrame({"execution_open": execution_dates})
    state = pd.merge_asof(
        daily,
        source,
        left_on="execution_open",
        right_on="source_eligible_at",
        direction="backward",
        allow_exact_matches=True,
    )
    state["source_age_days"] = (
        state["execution_open"] - state["source_date"]
    ).dt.days.astype(int)
    state = state[state["source_age_days"] <= maximum_carry_days].copy()
    state = state.set_index("execution_open")
    if state.empty:
        raise ValueError("Treasury causal state is empty")
    return state


def audit_treasury_dataset(
    raw: pd.DataFrame,
    state: pd.DataFrame,
    manifest: dict,
    config: dict,
) -> dict[str, object]:
    treasury = config["treasury"]
    start = pd.Timestamp(treasury["start"], tz="UTC")
    end = pd.Timestamp(treasury["end"], tz="UTC")
    expected = pd.bdate_range(start, end, tz="UTC")
    gaps = raw.index.to_series().diff().dropna()
    feature_columns = [
        column for column in state.columns
        if column not in {
            "source_date", "source_final_at", "source_eligible_at", "source_age_days"
        }
    ]
    values = state[feature_columns].to_numpy(dtype=float)
    execution = state.index.to_series(index=state.index)
    checks = {
        "raw_weekday_coverage": bool(
            len(raw) / len(expected) >= float(treasury["minimum_weekday_coverage"])
        ),
        "raw_dates_unique_sorted": bool(
            raw.index.is_monotonic_increasing and not raw.index.has_duplicates
        ),
        "raw_values_finite_positive": bool(
            np.isfinite(raw.to_numpy()).all() and (raw.to_numpy() > 0).all()
        ),
        "raw_gap_within_carry_bound": bool(
            gaps.empty
            or gaps.max() <= pd.Timedelta(days=int(treasury["maximum_carry_days"]))
        ),
        "all_source_hashes_recorded": all(
            len(row["raw_payload_sha256"]) == 64
            and len(row["observations_sha256"]) == 64
            for row in manifest["records"]
        ),
        "feature_schema_matches_registration": feature_columns
        == list(treasury["feature_columns"]),
        "state_index_unique_sorted": bool(
            state.index.is_monotonic_increasing and not state.index.has_duplicates
        ),
        "state_values_finite": bool(np.isfinite(values).all()),
        "source_final_strictly_precedes_execution": bool(
            (state["source_final_at"] < execution).all()
        ),
        "eligibility_not_after_execution": bool(
            (state["source_eligible_at"] <= execution).all()
        ),
        "eligibility_is_frozen_t_plus_three": bool(
            (
                state["source_eligible_at"] - state["source_date"]
                == pd.Timedelta(days=int(treasury["execution_lag_calendar_days"]))
            ).all()
        ),
        "known_state_age_is_bounded": bool(
            state["source_age_days"].between(
                int(treasury["execution_lag_calendar_days"]),
                int(treasury["maximum_carry_days"]),
            ).all()
        ),
        "no_sol_specific_series": not any("sol" in column.lower() for column in state),
        "no_return_or_policy_columns": not any(
            token in column.lower()
            for column in state
            for token in ("return", "position", "signal", "pnl")
        ),
    }
    return {
        "raw_rows": len(raw),
        "state_rows": len(state),
        "feature_count": len(feature_columns),
        "maximum_state_age_days": int(state["source_age_days"].max()),
        "checks": checks,
        "passed": all(checks.values()),
    }


def _report(result: dict) -> str:
    audit = result["audit"]
    lines = [
        "# TLM v18 Treasury Curve Data Layer",
        "",
        "## Result",
        "",
        "- Audit: **PASS**",
        f"- Raw observations: **{audit['raw_rows']}**",
        f"- Daily causal state rows: **{audit['state_rows']}**",
        f"- Registered features: **{audit['feature_count']}**",
        f"- Maximum observed state age: **{audit['maximum_state_age_days']} days**",
        "",
        "## Annual sources",
        "",
    ]
    for row in result["manifest"]["records"]:
        lines.append(
            f"- {row['year']}: rows={row['rows']}; cache="
            f"{'hit' if row['cached'] else 'miss'}; observation SHA-256 "
            f"`{row['observations_sha256']}`."
        )
    lines.extend([
        "",
        "## Chronology",
        "",
        "Each source observation becomes eligible only at `source date + 3 calendar days`. Daily crypto rows use the latest already-eligible public state, retain source and eligibility timestamps, and stop carrying after seven days. This is a causal known-state join, not an unbounded imputation.",
        "",
        "## Decision",
        "",
        "V18 is accepted as data infrastructure only. V19 may pre-register a policy-free signal-existence study over the frozen ten macro-market columns. No model or portfolio is authorized by this data audit.",
        "",
    ])
    return "\n".join(lines)


def run_treasury_pipeline(
    config: dict,
    force: bool = False,
    fetch_bytes: BytesFetcher = _http_get_bytes,
) -> dict[str, object]:
    treasury = config["treasury"]
    start_year = pd.Timestamp(treasury["start"]).year
    end_year = pd.Timestamp(treasury["end"]).year
    frames = []
    records = []
    for year in range(start_year, end_year + 1):
        frame, record = load_or_download_treasury_year(
            config, year, force=force, fetch_bytes=fetch_bytes
        )
        frames.append(frame)
        records.append(record)
    raw = pd.concat(frames).sort_index()
    if raw.index.has_duplicates:
        raise ValueError("Treasury annual cache files overlap")
    raw = raw.loc[
        pd.Timestamp(treasury["start"], tz="UTC"):
        pd.Timestamp(treasury["end"], tz="UTC")
    ]
    source_features = build_treasury_source_features(
        raw, int(treasury["rolling_window"])
    )
    state = materialize_causal_treasury_state(
        source_features,
        int(treasury["execution_lag_calendar_days"]),
        int(treasury["maximum_carry_days"]),
    )
    manifest = {
        "source": "us_treasury_daily_par_yield_curve",
        "start": treasury["start"],
        "end": treasury["end"],
        "records": records,
    }
    audit = audit_treasury_dataset(raw, state, manifest, config)
    if not audit["passed"]:
        raise RuntimeError(f"Treasury data audit failed: {audit['checks']}")
    catalog = {
        "family": "official_us_rates_macro_market",
        "source_columns": treasury["source_columns"],
        "feature_columns": treasury["feature_columns"],
        "scope": "market_level_context_for_btc_eth_sol_decisions",
        "timestamp_contract": {
            "first_allowed_execution": "source date + 3 calendar days",
            "known_state_carry": "latest eligible source, maximum 7 days",
        },
        "missing_data_policy": "bounded_causal_known_state_carry",
    }
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(output / "treasury_raw.parquet")
    state.to_parquet(output / "treasury_features.parquet")
    result: dict[str, object] = {
        "version": "v18",
        "method": "cache_first_hash_audited_causal_treasury_curve_data_layer",
        "manifest": manifest,
        "catalog": catalog,
        "audit": audit,
        "decision": "authorize_v19_policy_free_treasury_signal_registration",
        "output_dir": str(output),
    }
    for name, value in (
        ("manifest.json", manifest), ("catalog.json", catalog), ("audit.json", audit)
    ):
        (output / name).write_text(
            json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
        )
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    return result
