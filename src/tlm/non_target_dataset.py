from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import yaml

from .derivatives_data import parse_checksum
from .non_target_inventory import KLINE_COLUMNS


PANEL_FEATURES = (
    "log_open_to_open_return",
    "log_close_to_close_return",
    "log_high_low_range",
    "log_close_open_return",
    "log1p_quote_volume_change",
    "log1p_trade_count_change",
    "rolling_realized_volatility_7d",
    "rolling_realized_volatility_30d",
)
TRIPLET_FEATURE = "within_triplet_relative_strength"
LABEL_COLUMNS = (
    "target_next_open_to_next_open_log_return",
    "target_realized_volatility_7d",
)
RAW_FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trade_count",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"JSONL manifest is empty: {path}")
    return rows


def _to_utc_datetime(values: pd.Series) -> pd.DatetimeIndex:
    integers = pd.to_numeric(values, errors="raise").astype("int64")
    unit = "us" if integers.abs().max() >= 100_000_000_000_000 else "ms"
    return pd.DatetimeIndex(pd.to_datetime(integers, unit=unit, utc=True))


def read_cached_archive(
    record: dict[str, object],
    raw_dir: Path,
    interval: str,
) -> pd.DataFrame:
    symbol = str(record["symbol"])
    month = str(record["month"])
    name = f"{symbol}-{interval}-{month}.zip"
    archive_path = raw_dir / symbol / interval / name
    checksum_path = archive_path.with_suffix(archive_path.suffix + ".CHECKSUM")
    if not archive_path.is_file() or not checksum_path.is_file():
        raise FileNotFoundError(f"Missing frozen cache files for {symbol} {month}")

    payload = archive_path.read_bytes()
    checksum_payload = checksum_path.read_bytes()
    actual = _sha256_bytes(payload)
    published = parse_checksum(checksum_payload)
    if actual != str(record["sha256"]) or actual != published:
        raise ValueError(f"Frozen checksum mismatch for {symbol} {month}")
    if _sha256_bytes(checksum_payload) != str(record["checksum_sha256"]):
        raise ValueError(f"Checksum sidecar hash mismatch for {symbol} {month}")

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected one CSV for {symbol} {month}")
        with archive.open(csv_names[0]) as raw:
            frame = pd.read_csv(raw, header=None, names=KLINE_COLUMNS, dtype=str)
    if frame.empty:
        raise ValueError(f"Empty archive for {symbol} {month}")
    if frame.iloc[0]["open_time"].strip().lower().replace(" ", "_") == "open_time":
        frame = frame.iloc[1:].copy()
    dates = _to_utc_datetime(frame["open_time"])
    if not dates.is_normalized:
        raise ValueError(f"Non-midnight daily timestamp for {symbol} {month}")

    parsed = pd.DataFrame(index=dates)
    for field in RAW_FIELDS:
        parsed[field] = pd.to_numeric(frame[field].to_numpy(), errors="raise")
    if not np.isfinite(parsed.to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite raw value for {symbol} {month}")
    if (parsed[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError(f"Non-positive price for {symbol} {month}")
    if (parsed[["volume", "quote_volume", "trade_count"]] < 0).any().any():
        raise ValueError(f"Negative activity value for {symbol} {month}")
    if (parsed["high"] < parsed[["open", "close"]].max(axis=1)).any():
        raise ValueError(f"High below open/close for {symbol} {month}")
    if (parsed["low"] > parsed[["open", "close"]].min(axis=1)).any():
        raise ValueError(f"Low above open/close for {symbol} {month}")
    if parsed.index.has_duplicates or not parsed.index.is_monotonic_increasing:
        raise ValueError(f"Duplicate or unsorted dates for {symbol} {month}")
    if len(parsed) != int(record["row_count"]):
        raise ValueError(f"Row-count drift for {symbol} {month}")
    if parsed.index[0].date().isoformat() != str(record["first_date"]):
        raise ValueError(f"First-date drift for {symbol} {month}")
    if parsed.index[-1].date().isoformat() != str(record["last_date"]):
        raise ValueError(f"Last-date drift for {symbol} {month}")
    return parsed.astype({"trade_count": "float64"})


def load_frozen_frames(
    records: list[dict[str, object]],
    symbols: list[str],
    raw_dir: Path,
    interval: str,
    workers: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    expected_symbols = set(symbols)
    manifest_symbols = {str(record["symbol"]) for record in records}
    if manifest_symbols != expected_symbols:
        raise ValueError("V27 manifest symbols do not match the frozen universe")

    parts: dict[str, list[pd.DataFrame]] = {symbol: [] for symbol in symbols}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(read_cached_archive, record, raw_dir, interval): record
            for record in records
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            record = futures[future]
            parts[str(record["symbol"])].append(future.result())
            if completed % 500 == 0 or completed == len(futures):
                print(f"non-target dataset archives: {completed}/{len(futures)}", flush=True)

    frames: dict[str, pd.DataFrame] = {}
    source_rows = 0
    per_symbol: list[dict[str, object]] = []
    for symbol in symbols:
        frame = pd.concat(parts[symbol]).sort_index()
        if frame.index.has_duplicates:
            raise ValueError(f"Duplicate dates across archives for {symbol}")
        source_rows += len(frame)
        frames[symbol] = frame
        per_symbol.append({
            "symbol": symbol,
            "archive_count": len(parts[symbol]),
            "raw_rows": len(frame),
            "first_date": frame.index.min().date().isoformat(),
            "last_date": frame.index.max().date().isoformat(),
        })
    return frames, {
        "archive_count": len(records),
        "source_rows": source_rows,
        "per_symbol": per_symbol,
        "all_cache_hashes_verified": True,
    }


def _root_sum_squares(values: pd.Series, window: int) -> pd.Series:
    return values.pow(2).rolling(window, min_periods=window).sum().pow(0.5)


def build_symbol_panel(
    symbol: str,
    frame: pd.DataFrame,
    expected_index: pd.DatetimeIndex,
    splits: dict[str, list[str]],
    lookback_days: int,
) -> pd.DataFrame:
    raw = frame.reindex(expected_index)
    observed = raw["open"].notna()
    log_open = np.log(raw["open"])
    log_close = np.log(raw["close"])
    open_return = log_open.diff()
    close_return = log_close.diff()

    panel = pd.DataFrame({
        "date": expected_index,
        "symbol": symbol,
        "base": symbol.removesuffix("USDT"),
        "eligible_action_date": expected_index + pd.Timedelta(days=1),
        "target_window_end_date": expected_index + pd.Timedelta(days=8),
        "raw_observation_available": observed.to_numpy(),
    })
    for field in RAW_FIELDS:
        panel[f"raw_{field}"] = raw[field].to_numpy()
    panel["log_open_to_open_return"] = open_return.to_numpy()
    panel["log_close_to_close_return"] = close_return.to_numpy()
    panel["log_high_low_range"] = np.log(raw["high"] / raw["low"]).to_numpy()
    panel["log_close_open_return"] = np.log(raw["close"] / raw["open"]).to_numpy()
    panel["log1p_quote_volume_change"] = np.log1p(raw["quote_volume"]).diff().to_numpy()
    panel["log1p_trade_count_change"] = np.log1p(raw["trade_count"]).diff().to_numpy()
    panel["rolling_realized_volatility_7d"] = _root_sum_squares(
        close_return, 7
    ).to_numpy()
    panel["rolling_realized_volatility_30d"] = _root_sum_squares(
        close_return, 30
    ).to_numpy()

    panel["target_next_open_to_next_open_log_return"] = (
        log_open.shift(-2) - log_open.shift(-1)
    ).to_numpy()
    forward_returns = pd.concat(
        [
            log_open.shift(-(offset + 1)) - log_open.shift(-offset)
            for offset in range(1, 8)
        ],
        axis=1,
    )
    panel["target_realized_volatility_7d"] = forward_returns.pow(2).sum(
        axis=1, min_count=7
    ).pow(0.5).to_numpy()

    feature_complete = np.isfinite(panel[list(PANEL_FEATURES)].to_numpy()).all(axis=1)
    label_complete = np.isfinite(panel[list(LABEL_COLUMNS)].to_numpy()).all(axis=1)
    sequence_ready = (
        pd.Series(feature_complete, index=expected_index)
        .rolling(lookback_days, min_periods=lookback_days)
        .sum()
        .eq(lookback_days)
        .to_numpy()
    )
    panel["feature_complete"] = feature_complete
    panel["label_complete"] = label_complete
    panel["sequence_ready"] = sequence_ready
    panel["supervised_sequence_ready"] = sequence_ready & label_complete
    for name, bounds in splits.items():
        start = pd.Timestamp(bounds[0], tz="UTC")
        end = pd.Timestamp(bounds[1], tz="UTC")
        panel[f"in_{name}"] = (expected_index >= start) & (expected_index <= end)
    return panel


def build_feature_schema(
    blueprint: dict,
    dataset_config: dict,
    version: str = "v28",
) -> dict[str, object]:
    registered = list(blueprint["data_contract"]["derived_features"])
    expected = [*PANEL_FEATURES, TRIPLET_FEATURE]
    if registered != expected:
        raise ValueError(f"V26 feature order drift: {registered} != {expected}")
    if dataset_config["realized_volatility_formula"] != "root_sum_squared_log_returns":
        raise ValueError("Unsupported realized-volatility formula")
    if list(dataset_config["realized_volatility_windows"]) != [7, 30]:
        raise ValueError("V28 requires the frozen 7d/30d volatility windows")
    return {
        "version": version,
        "panel_key": ["symbol", "date"],
        "raw_fields": list(RAW_FIELDS),
        "panel_features": [
            {
                "name": "log_open_to_open_return",
                "formula": "log(open[t] / open[t-1])",
                "available": "close_of_t",
            },
            {
                "name": "log_close_to_close_return",
                "formula": "log(close[t] / close[t-1])",
                "available": "close_of_t",
            },
            {
                "name": "log_high_low_range",
                "formula": "log(high[t] / low[t])",
                "available": "close_of_t",
            },
            {
                "name": "log_close_open_return",
                "formula": "log(close[t] / open[t])",
                "available": "close_of_t",
            },
            {
                "name": "log1p_quote_volume_change",
                "formula": "log1p(quote_volume[t]) - log1p(quote_volume[t-1])",
                "available": "close_of_t",
            },
            {
                "name": "log1p_trade_count_change",
                "formula": "log1p(trade_count[t]) - log1p(trade_count[t-1])",
                "available": "close_of_t",
            },
            {
                "name": "rolling_realized_volatility_7d",
                "formula": "sqrt(sum(log_close_to_close_return[t-6:t]^2))",
                "available": "close_of_t",
            },
            {
                "name": "rolling_realized_volatility_30d",
                "formula": "sqrt(sum(log_close_to_close_return[t-29:t]^2))",
                "available": "close_of_t",
            },
        ],
        "triplet_derived_feature": {
            "name": TRIPLET_FEATURE,
            "source": dataset_config["triplet_relative_strength_source"],
            "formula": dataset_config["triplet_relative_strength_formula"],
            "materialization": "computed_after_triplet_formation_for_each_date",
        },
        "labels": [
            {
                "name": LABEL_COLUMNS[0],
                "formula": "log(open[t+2] / open[t+1])",
                "action_date": "t+1",
            },
            {
                "name": LABEL_COLUMNS[1],
                "formula": "sqrt(sum(log(open[d+1]/open[d])^2 for d=t+1..t+7))",
                "window_end": "t+8",
            },
        ],
        "missing_data_policy": dataset_config["missing_data_policy"],
        "model_feature_order": expected,
        "scaling": f"not_applied_in_{version}_train_only_fit_required_later",
    }


def build_asset_folds(symbols: list[str], fold_count: int) -> dict[str, object]:
    if fold_count < 3 or len(symbols) % fold_count:
        raise ValueError("Asset folds must be at least three and equally sized")
    ordered = sorted(symbols)
    folds = []
    for fold in range(fold_count):
        test_symbols = ordered[fold::fold_count]
        train_symbols = [symbol for symbol in ordered if symbol not in set(test_symbols)]
        folds.append({
            "fold": fold + 1,
            "train_symbols": train_symbols,
            "test_symbols": test_symbols,
        })
    return {
        "method": "lexical_round_robin_performance_blind",
        "fold_count": fold_count,
        "folds": folds,
    }


def _split_audit(panel: pd.DataFrame, splits: dict[str, list[str]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name in splits:
        subset = panel.loc[panel[f"in_{name}"]]
        result[name] = {
            "start": splits[name][0],
            "end": splits[name][1],
            "panel_rows": len(subset),
            "raw_observation_rows": int(subset["raw_observation_available"].sum()),
            "feature_complete_rows": int(subset["feature_complete"].sum()),
            "label_complete_rows": int(subset["label_complete"].sum()),
            "sequence_ready_rows": int(subset["sequence_ready"].sum()),
            "supervised_sequence_ready_rows": int(
                subset["supervised_sequence_ready"].sum()
            ),
        }
    return result


def _report(result: dict[str, object]) -> str:
    manifest = result["dataset_manifest"]
    source = result["source_audit"]
    missing = manifest["expected_panel_rows"] - source["source_rows"]
    return "\n".join([
        "# TLM v28 Non-Target Dataset",
        "",
        "## Decision",
        "",
        "**DATASET PASSED; TRAINING REMAINS BLOCKED PENDING THE EUR SCOPE AMENDMENT.**",
        "",
        f"Panel rows: **{manifest['panel_rows']:,}**",
        f"Observed raw rows: **{source['source_rows']:,}**",
        f"Preserved missing raw rows: **{missing:,}**",
        f"Verified archives: **{source['archive_count']:,}**",
        f"Panel SHA-256: `{manifest['panel_sha256']}`",
        f"Cache replay byte-identical: **{manifest['reproducibility']['byte_identical']}**",
        "",
        "Eight per-asset features and two forward labels were materialized. The ninth registered model feature is computed causally only after deterministic triplet formation.",
        "",
        "No target asset was loaded. No scaler or model was fitted, no portfolio was constructed, and no return statistic, Sharpe, drawdown, or PnL was evaluated.",
        "",
        "## Causality",
        "",
        "A row stamped `t` uses raw observations only through the close of `t` and is first actionable at the open of `t+1`. The return label spans `open[t+1] -> open[t+2]`; the volatility label ends at `open[t+8]`. Missing dates remain missing and break rolling/sequence eligibility.",
        "",
        "## Asset-disjoint folds",
        "",
        "Three equal 16-asset holdout folds were assigned by lexical round-robin without inspecting labels. Triplets and performance are not materialized in v28.",
        "",
        "## Next action",
        "",
        "V29 may only resolve the EURUSDT crypto-domain mismatch through a versioned, performance-blind universe amendment and refresh the affected source manifest/dataset. Training remains unauthorized.",
        "",
    ])


def run_non_target_dataset(config: dict) -> dict[str, object]:
    dataset_config = config["non_target_dataset"]
    root = Path(dataset_config["project_root"]).resolve()
    v26_path = root / dataset_config["v26_specification_path"]
    v27_inventory_path = root / dataset_config["v27_inventory_path"]
    v27_audit_path = root / dataset_config["v27_audit_path"]
    v27_manifest_path = root / dataset_config["v27_manifest_path"]

    v26 = json.loads(v26_path.read_text(encoding="utf-8"))
    v27 = json.loads(v27_inventory_path.read_text(encoding="utf-8"))
    v27_audit = json.loads(v27_audit_path.read_text(encoding="utf-8"))
    records = _load_jsonl(v27_manifest_path)
    blueprint = v26["blueprint"]
    symbols = list(v27["universe"]["selected_symbols"])
    target_symbols = set(blueprint["target_symbols"])
    data_contract = blueprint["data_contract"]
    splits = blueprint["chronological_splits"]

    feature_schema = build_feature_schema(blueprint, dataset_config)
    feature_schema_sha256 = _canonical_sha256(feature_schema)
    frames, source_audit = load_frozen_frames(
        records,
        symbols,
        root / dataset_config["raw_dir"],
        data_contract["frequency"],
        int(dataset_config["archive_workers"]),
    )
    start = pd.Timestamp(data_contract["development_start"])
    end = pd.Timestamp(data_contract["development_cutoff"]).floor("D")
    expected_index = pd.date_range(start, end, freq="D", tz="UTC")
    for row in source_audit["per_symbol"]:
        row["expected_rows"] = len(expected_index)
        row["coverage"] = row["raw_rows"] / len(expected_index)
    panels = [
        build_symbol_panel(
            symbol,
            frames[symbol],
            expected_index,
            splits,
            int(dataset_config["lookback_days"]),
        )
        for symbol in symbols
    ]
    panel = pd.concat(panels, ignore_index=True).sort_values(
        ["date", "symbol"]
    ).reset_index(drop=True)

    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    panel_path = root / dataset_config["panel_path"]
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    prior_panel_sha256 = _sha256_file(panel_path) if panel_path.is_file() else None
    panel.to_parquet(panel_path, index=False, compression="zstd")
    panel_sha256 = _sha256_file(panel_path)
    replay_hash_matches = (
        prior_panel_sha256 is None or prior_panel_sha256 == panel_sha256
    )
    asset_folds = build_asset_folds(symbols, int(dataset_config["asset_fold_count"]))
    split_audit = _split_audit(panel, splits)

    expected_rows = len(expected_index) * len(symbols)
    raw_missing_mask = ~panel["raw_observation_available"]
    raw_columns = [f"raw_{field}" for field in RAW_FIELDS]
    feature_values = panel[list(PANEL_FEATURES)].to_numpy(dtype=float)
    label_values = panel[list(LABEL_COLUMNS)].to_numpy(dtype=float)
    checks = {
        "v26_specification_hash_matches": _sha256_file(v26_path)
        == dataset_config["expected_v26_specification_sha256"],
        "v26_blueprint_hash_matches": v26["blueprint_sha256"]
        == dataset_config["expected_v26_blueprint_sha256"],
        "v27_inventory_hash_matches": _sha256_file(v27_inventory_path)
        == dataset_config["expected_v27_inventory_sha256"],
        "v27_manifest_hash_matches": _sha256_file(v27_manifest_path)
        == dataset_config["expected_v27_manifest_sha256"],
        "v27_authorizes_dataset_only": v27["decision"]
        == "authorize_v28_non_target_dataset_build_only",
        "v27_audit_passes": bool(v27_audit["passed"]),
        "exact_universe_loaded": set(frames) == set(symbols),
        "no_target_symbol_loaded": not target_symbols.intersection(frames),
        "all_frozen_archive_hashes_verified": bool(
            source_audit["all_cache_hashes_verified"]
        ),
        "archive_count_matches_manifest": source_audit["archive_count"]
        == len(records)
        == v27["artifact_references"]["archive_manifest"]["records"],
        "panel_row_count_matches_cartesian_calendar": len(panel) == expected_rows,
        "panel_key_is_unique": not panel.duplicated(["symbol", "date"]).any(),
        "missing_raw_rows_are_preserved": panel.loc[
            raw_missing_mask, raw_columns
        ].isna().all().all(),
        "observed_raw_count_matches_sources": int(
            panel["raw_observation_available"].sum()
        ) == source_audit["source_rows"],
        "source_rows_match_manifest_rows": source_audit["source_rows"]
        == sum(int(record["row_count"]) for record in records),
        "all_source_coverage_passes": all(
            row["coverage"] >= float(v27["universe"]["minimum_daily_coverage"])
            for row in source_audit["per_symbol"]
        ),
        "finite_or_missing_features_only": not np.isinf(feature_values).any(),
        "finite_or_missing_labels_only": not np.isinf(label_values).any(),
        "registered_feature_order_matches": feature_schema["model_feature_order"]
        == list(data_contract["derived_features"]),
        "eligible_action_is_t_plus_one": (
            panel["eligible_action_date"] - panel["date"]
            == pd.Timedelta(days=1)
        ).all(),
        "target_window_ends_at_t_plus_eight": (
            panel["target_window_end_date"] - panel["date"]
            == pd.Timedelta(days=8)
        ).all(),
        "model_training_count_is_zero": True,
        "target_asset_prediction_count_is_zero": True,
        "portfolio_evaluation_count_is_zero": True,
        "performance_metric_count_is_zero": True,
        "pnl_evaluation_count_is_zero": True,
        "imputation_count_is_zero": True,
        "cache_replay_hash_matches_when_available": replay_hash_matches,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise RuntimeError(f"V28 dataset audit failed: {checks}")

    dataset_manifest = {
        "version": "v28",
        "panel_path": str(panel_path.relative_to(root)),
        "panel_sha256": panel_sha256,
        "panel_rows": len(panel),
        "expected_panel_rows": expected_rows,
        "symbols": symbols,
        "symbol_count": len(symbols),
        "calendar_start": expected_index.min().date().isoformat(),
        "calendar_end": expected_index.max().date().isoformat(),
        "calendar_days": len(expected_index),
        "panel_features": list(PANEL_FEATURES),
        "triplet_feature": TRIPLET_FEATURE,
        "labels": list(LABEL_COLUMNS),
        "feature_schema_sha256": feature_schema_sha256,
        "source_manifest_sha256": _sha256_file(v27_manifest_path),
        "source_blueprint_sha256": v26["blueprint_sha256"],
        "reproducibility": {
            "prior_panel_sha256": prior_panel_sha256,
            "current_panel_sha256": panel_sha256,
            "cache_replay_performed": prior_panel_sha256 is not None,
            "byte_identical": replay_hash_matches,
        },
    }
    result = {
        "version": "v28",
        "decision": "authorize_v29_performance_blind_scope_amendment_only",
        "dataset_manifest": dataset_manifest,
        "source_audit": source_audit,
        "split_audit": split_audit,
        "asset_folds": asset_folds,
        "feature_schema": feature_schema,
        "tested": {
            "labels_materialized": True,
            "return_label_materialized": True,
            "performance_metrics_computed": False,
            "model_trained": False,
            "portfolio_constructed": False,
            "pnl_computed": False,
            "target_assets_loaded": False,
            "cache_replay_performed": prior_panel_sha256 is not None,
            "cache_replay_byte_identical": replay_hash_matches,
            "improvement_status": "unknown_not_evaluated",
            "drawdown_status": "unknown_not_evaluated",
        },
        "scope_observations": v27["universe"]["scope_observations"],
        "audit": {"passed": True, "checks": checks},
    }
    files = {
        "dataset_manifest.json": dataset_manifest,
        "feature_schema.json": feature_schema,
        "split_audit.json": split_audit,
        "asset_folds.json": asset_folds,
        "source_audit.json": source_audit,
        "audit.json": result["audit"],
        "result.json": result,
    }
    for name, value in files.items():
        (output / name).write_text(
            json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
        )
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    return result
