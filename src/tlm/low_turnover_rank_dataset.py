from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from typing import Any, Iterator

import numpy as np
import pandas as pd
import pyarrow
import yaml

from .core import canonical_sha256, file_sha256, write_json_atomic, write_yaml_atomic
from .non_target_dataset import PANEL_FEATURES, RAW_FIELDS, load_frozen_frames
from .non_target_inventory import list_s3_objects, verify_month_archive


TARGET_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
STATIC_INPUT_NAMES = {
    "v82_user_authorization",
    "v82_r0_result",
    "v82_r0_audit",
    "v82_r0_erratum",
    "v82_r0_artifact_manifest",
    "v80_blueprint",
    "v32_result",
    "v32_audit",
    "v32_dataset_manifest",
    "v32_feature_schema",
    "v32_asset_folds",
    "v32_triplet_catalog",
}


@dataclass
class LowTurnoverDatasetLedger:
    authorized_metadata_reads: int = 0
    official_source_listings: int = 0
    verified_source_archives: int = 0
    authorized_raw_source_rows: int = 0
    parquet_writes: int = 0
    training_label_rows_materialized: int = 0
    internal_validation_label_rows_materialized: int = 0
    sealed_evaluation_outcome_rows_materialized: int = 0
    sealed_evaluation_outcome_packet_writes: int = 0
    sealed_evaluation_outcome_packet_unseals: int = 0
    scaler_fits: int = 0
    model_instantiations: int = 0
    checkpoint_reads: int = 0
    checkpoint_writes: int = 0
    optimizer_steps: int = 0
    training_epochs: int = 0
    market_inferences: int = 0
    market_predictions: int = 0
    positions_constructed: int = 0
    performance_metrics: int = 0
    pnl_evaluations: int = 0
    bootstrap_paths: int = 0
    target_asset_loads: int = 0
    missing_value_imputations: int = 0
    universe_reselections: int = 0

    def forbidden_operations_are_zero(self) -> bool:
        names = (
            "sealed_evaluation_outcome_packet_unseals",
            "scaler_fits",
            "model_instantiations",
            "checkpoint_reads",
            "checkpoint_writes",
            "optimizer_steps",
            "training_epochs",
            "market_inferences",
            "market_predictions",
            "positions_constructed",
            "performance_metrics",
            "pnl_evaluations",
            "bootstrap_paths",
            "target_asset_loads",
            "missing_value_imputations",
            "universe_reselections",
        )
        return all(getattr(self, name) == 0 for name in names)

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _project_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"V82 path escapes project root: {relative}") from exc
    return path


def _load_json(path: Path, ledger: LowTurnoverDatasetLedger) -> dict[str, Any]:
    ledger.authorized_metadata_reads += 1
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _self_hash_matches(value: dict[str, Any], field: str, expected: str) -> bool:
    payload = dict(value)
    embedded = payload.pop(field, None)
    return embedded == expected == canonical_sha256(payload)


@contextmanager
def _process_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another V82 dataset process holds the lock") from exc
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        path.unlink(missing_ok=True)


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
        for row in rows
    )
    _write_text_atomic(path, text)


def _write_parquet_with_fresh_replay(
    frame: pd.DataFrame,
    path: Path,
    *,
    engine: str,
    compression: str,
    ledger: LowTurnoverDatasetLedger,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    first = path.with_name(f".{path.name}.v82-replay-a.tmp")
    second = path.with_name(f".{path.name}.v82-replay-b.tmp")
    first.unlink(missing_ok=True)
    second.unlink(missing_ok=True)
    try:
        frame.to_parquet(first, index=False, engine=engine, compression=compression)
        frame.to_parquet(second, index=False, engine=engine, compression=compression)
        ledger.parquet_writes += 2
        first_hash = file_sha256(first)
        second_hash = file_sha256(second)
        if first_hash != second_hash:
            raise RuntimeError(f"V82 fresh Parquet replay drift: {path.name}")
        os.replace(first, path)
        return {
            "path": str(path),
            "sha256": first_hash,
            "fresh_replay_sha256": second_hash,
            "byte_identical": True,
            "rows": len(frame),
            "columns": list(frame.columns),
        }
    finally:
        first.unlink(missing_ok=True)
        second.unlink(missing_ok=True)


def _month_range(start: str, end: str) -> list[str]:
    return [str(period) for period in pd.period_range(start, end, freq="M")]


def discover_frozen_source_pairs(
    source_contract: dict[str, Any],
    *,
    timeout: float,
    workers: int,
    ledger: LowTurnoverDatasetLedger,
) -> tuple[list[tuple[str, str]], dict[str, list[str]]]:
    symbols = list(source_contract["symbols"])
    months = _month_range(
        source_contract["month_start"], source_contract["month_end"]
    )
    endpoint = str(source_contract["s3_endpoint"])
    root_prefix = str(source_contract["root_prefix"])
    interval = str(source_contract["frequency"])

    def list_symbol(symbol: str) -> tuple[str, set[str]]:
        prefix = f"{root_prefix}{symbol}/{interval}/"
        objects, _ = list_s3_objects(endpoint, prefix, timeout)
        return symbol, {Path(item.key).name for item in objects}

    names_by_symbol: dict[str, set[str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(list_symbol, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol, names = future.result()
            names_by_symbol[symbol] = names
    ledger.official_source_listings += len(symbols)

    admitted: list[tuple[str, str]] = []
    missing: dict[str, list[str]] = {}
    for symbol in symbols:
        names = names_by_symbol[symbol]
        missing[symbol] = []
        for month in months:
            archive = f"{symbol}-{interval}-{month}.zip"
            if archive in names and f"{archive}.CHECKSUM" in names:
                admitted.append((symbol, month))
            else:
                missing[symbol].append(month)
    return admitted, missing


def verify_frozen_source_archives(
    pairs: list[tuple[str, str]],
    source_contract: dict[str, Any],
    *,
    raw_dir: Path,
    timeout: float,
    workers: int,
    ledger: LowTurnoverDatasetLedger,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    interval = str(source_contract["frequency"])
    download_base = str(source_contract["download_base_url"])

    def verify(pair: tuple[str, str]) -> dict[str, Any]:
        symbol, month = pair
        record = verify_month_archive(
            symbol,
            month,
            interval,
            download_base,
            raw_dir,
            timeout,
            False,
        )
        return {
            "symbol": record["symbol"],
            "month": record["month"],
            "url": record["url"],
            "bytes": record["bytes"],
            "sha256": record["sha256"],
            "checksum_sha256": record["checksum_sha256"],
            "checksum_verified": record["checksum_verified"],
            "row_count": record["row_count"],
            "first_date": record["first_date"],
            "last_date": record["last_date"],
            "timestamp_units": record["timestamp_units"],
            "column_count": record["column_count"],
            "schema_valid": record["schema_valid"],
        }

    records: list[dict[str, Any]] = []
    rejections: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(verify, pair): pair for pair in pairs}
        for completed, future in enumerate(as_completed(futures), start=1):
            symbol, month = futures[future]
            try:
                records.append(future.result())
            except Exception as error:
                rejections.append(
                    {
                        "symbol": symbol,
                        "month": month,
                        "error_type": type(error).__name__,
                        "reason": str(error),
                    }
                )
            if completed % 500 == 0 or completed == len(futures):
                print(f"V82 verified source archives: {completed}/{len(futures)}", flush=True)
    records.sort(key=lambda row: (str(row["symbol"]), str(row["month"])))
    rejections.sort(key=lambda row: (row["symbol"], row["month"]))
    ledger.verified_source_archives += len(records)
    ledger.authorized_raw_source_rows += sum(int(row["row_count"]) for row in records)
    return records, rejections


def _root_sum_squares(values: pd.Series, window: int) -> pd.Series:
    return values.pow(2).rolling(window, min_periods=window).sum().pow(0.5)


def build_causal_feature_frame(
    symbol: str,
    raw_frame: pd.DataFrame,
    expected_index: pd.DatetimeIndex,
    *,
    lookback_days: int,
) -> pd.DataFrame:
    raw = raw_frame.reindex(expected_index)
    log_open = np.log(raw["open"])
    log_close = np.log(raw["close"])
    open_return = log_open.diff()
    close_return = log_close.diff()
    frame = pd.DataFrame(
        {
            "date": expected_index,
            "symbol": symbol,
            "raw_observation_available": raw["open"].notna().to_numpy(),
            "log_open_to_open_return": open_return.to_numpy(),
            "log_close_to_close_return": close_return.to_numpy(),
            "log_high_low_range": np.log(raw["high"] / raw["low"]).to_numpy(),
            "log_close_open_return": np.log(raw["close"] / raw["open"]).to_numpy(),
            "log1p_quote_volume_change": np.log1p(raw["quote_volume"]).diff().to_numpy(),
            "log1p_trade_count_change": np.log1p(raw["trade_count"]).diff().to_numpy(),
            "rolling_realized_volatility_7d": _root_sum_squares(
                close_return, 7
            ).to_numpy(),
            "rolling_realized_volatility_30d": _root_sum_squares(
                close_return, 30
            ).to_numpy(),
        }
    )
    feature_values = frame[list(PANEL_FEATURES)].to_numpy(dtype=np.float64)
    complete = np.isfinite(feature_values).all(axis=1)
    ready = (
        pd.Series(complete, index=expected_index)
        .rolling(lookback_days, min_periods=lookback_days)
        .sum()
        .eq(lookback_days)
        .to_numpy()
    )
    frame["feature_complete"] = complete
    frame["sequence_start_date"] = frame["date"] - pd.Timedelta(
        days=lookback_days - 1
    )
    frame["sequence_ready"] = ready
    return frame


def build_development_labels(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    source_index: pd.DatetimeIndex,
    feature_frames: dict[str, pd.DataFrame],
    role_contract: dict[str, Any],
    label_contract: dict[str, Any],
    folds: list[dict[str, Any]],
) -> pd.DataFrame:
    if label_contract["formula"] != "log(open[t+22] / open[t+1])":
        raise ValueError("V82 label formula drift")
    if int(label_contract["maturity_days"]) != 22:
        raise ValueError("V82 label maturity drift")
    role_dates: list[tuple[str, pd.DatetimeIndex]] = []
    for name in ("train", "internal_validation"):
        role = role_contract[name]
        role_dates.append(
            (
                name,
                pd.date_range(role["signal_start"], role["signal_end"], freq="D", tz="UTC"),
            )
        )
    output: list[pd.DataFrame] = []
    for symbol in symbols:
        raw = frames[symbol].reindex(source_index)
        log_open = np.log(raw["open"])
        target = log_open.shift(-22) - log_open.shift(-1)
        readiness = feature_frames[symbol].set_index("date")["sequence_ready"]
        for role_name, dates in role_dates:
            values = target.reindex(dates)
            table = pd.DataFrame(
                {
                    "signal_date": dates,
                    "symbol": symbol,
                    "execution_open_date": dates + pd.Timedelta(days=1),
                    "exit_open_date": dates + pd.Timedelta(days=22),
                    "role": role_name,
                    "target_21d_open_to_open_log_return": values.to_numpy(),
                    "label_complete": np.isfinite(values.to_numpy(dtype=np.float64)),
                    "sequence_ready": readiness.reindex(dates).fillna(False).to_numpy(dtype=bool),
                }
            )
            for fold in folds:
                fold_id = int(fold["fold"])
                in_training_assets = symbol in set(fold["train_symbols"])
                table[f"eligible_fold_{fold_id}"] = (
                    table["label_complete"]
                    & table["sequence_ready"]
                    & in_training_assets
                )
            output.append(table)
    return (
        pd.concat(output, ignore_index=True)
        .sort_values(["signal_date", "symbol"])
        .reset_index(drop=True)
    )


def build_sealed_daily_outcome_packet(
    frames: dict[str, pd.DataFrame],
    symbols: list[str],
    source_index: pd.DatetimeIndex,
    evaluation_contract: dict[str, Any],
) -> pd.DataFrame:
    starts = pd.date_range(
        evaluation_contract["daily_outcome_interval_start"],
        evaluation_contract["daily_outcome_interval_end"],
        freq="D",
        tz="UTC",
    )
    output: list[pd.DataFrame] = []
    for symbol in symbols:
        raw = frames[symbol].reindex(source_index)
        log_open = np.log(raw["open"])
        daily = log_open.shift(-1) - log_open
        values = daily.reindex(starts)
        output.append(
            pd.DataFrame(
                {
                    "interval_start_date": starts,
                    "interval_end_date": starts + pd.Timedelta(days=1),
                    "symbol": symbol,
                    "open_to_next_open_log_return": values.to_numpy(),
                    "outcome_complete": np.isfinite(values.to_numpy(dtype=np.float64)),
                }
            )
        )
    return (
        pd.concat(output, ignore_index=True)
        .sort_values(["interval_start_date", "symbol"])
        .reset_index(drop=True)
    )


def _metadata_context(
    config: dict[str, Any], ledger: LowTurnoverDatasetLedger
) -> dict[str, Any]:
    dataset = config["low_turnover_rank_dataset"]
    root = Path(dataset["project_root"]).resolve()
    phase_reference = dataset["phase_contract"]
    phase_path = _project_path(root, phase_reference["path"])
    if not phase_path.is_file() or file_sha256(phase_path) != phase_reference["file_sha256"]:
        raise RuntimeError("V82 phase contract is missing or hash-drifted")
    contract = _load_yaml(phase_path)
    if (
        contract.get("phase") != "v82"
        or contract.get("stage_revision")
        != "v082_non_target_low_turnover_rank_dataset_r1"
        or contract.get("authorized_next_action")
        != "authorize_v82_non_target_low_turnover_rank_dataset_only"
        or config.get("output_dir") != contract["access_contract"]["output_dir"]
    ):
        raise RuntimeError("V82 phase contract is inconsistent")

    input_paths = {
        name: _project_path(root, relative)
        for name, relative in dataset["inputs"].items()
    }
    if set(input_paths) != STATIC_INPUT_NAMES:
        raise RuntimeError("V82 static input-name allowlist drift")
    if set(dataset["inputs"].values()) != set(contract["access_contract"]["allowed_inputs"]):
        raise RuntimeError("V82 static input-path allowlist drift")
    expected_by_path = contract["input_contract"]["expected_static_file_sha256_by_path"]
    expected_hashes = {
        name: expected_by_path[relative] for name, relative in dataset["inputs"].items()
    }
    observed_hashes: dict[str, str] = {}
    for name, path in input_paths.items():
        if not path.is_file():
            raise RuntimeError(f"V82 static input is missing: {name}")
        observed_hashes[name] = file_sha256(path)
        if observed_hashes[name] != expected_hashes[name]:
            raise RuntimeError(f"V82 static input hash drift: {name}")
    values = {name: _load_json(path, ledger) for name, path in input_paths.items()}
    canonical = contract["input_contract"]["expected_canonical_sha256"]
    authorization = values["v82_user_authorization"]
    v82_result = values["v82_r0_result"]
    blueprint = values["v80_blueprint"]
    v32_manifest = values["v32_dataset_manifest"]
    v32_schema = values["v32_feature_schema"]
    folds = values["v32_asset_folds"]
    catalog = values["v32_triplet_catalog"]
    if (
        not _self_hash_matches(
            authorization, "authorization_sha256", canonical["v82_user_authorization"]
        )
        or authorization.get("bound_v82_r0_result_sha256")
        != canonical["v82_r0_result"]
        or authorization.get("target_assets_status") != "sealed"
        or not _self_hash_matches(v82_result, "result_sha256", canonical["v82_r0_result"])
        or v82_result.get("decision")
        != "authorize_v82_non_target_low_turnover_rank_dataset_only"
        or v82_result.get("final_evaluation_signal_end") != "2026-06-08"
        or v82_result.get("final_evaluation_signal_dates") != 159
        or v82_result.get("target_assets_loaded") != []
        or values["v82_r0_audit"].get("passed") is not True
        or not _self_hash_matches(blueprint, "blueprint_sha256", canonical["v80_blueprint"])
        or blueprint.get("chronology", {}).get("source_start") != "2018-01-01"
        or blueprint.get("input", {}).get("lookback_days") != 128
        or blueprint.get("target", {}).get("horizon_intervals") != 21
        or values["v32_audit"].get("passed") is not True
        or v32_manifest.get("symbol_count") != 30
        or list(v32_manifest.get("symbols", [])) != list(contract["source_contract"]["symbols"])
        or list(v32_manifest.get("panel_features", [])) != list(contract["feature_contract"]["columns"])
        or list(v32_schema.get("model_feature_order", []))[:8]
        != list(contract["feature_contract"]["columns"])
        or catalog.get("catalog_sha256") != canonical["v32_triplet_catalog"]
        or len(folds.get("folds", [])) != 3
        or len(catalog.get("folds", [])) != 3
    ):
        raise RuntimeError("V82 parent metadata contract drift")
    for fold, catalog_fold in zip(folds["folds"], catalog["folds"], strict=True):
        if (
            int(fold["fold"]) != int(catalog_fold["fold"])
            or fold["train_symbols"] != catalog_fold["train_symbols"]
            or fold["test_symbols"] != catalog_fold["test_symbols"]
            or len(fold["train_symbols"]) != 20
            or len(fold["test_symbols"]) != 10
            or len(catalog_fold["train_triplets"]) != math.comb(20, 3)
            or len(catalog_fold["test_triplets"]) != math.comb(10, 3)
        ):
            raise RuntimeError("V82 frozen fold or triplet ancestry drift")

    source_hashes: dict[str, str] = {}
    for relative in dataset["source_receipt_files"]:
        path = _project_path(root, relative)
        if not path.is_file():
            raise RuntimeError(f"V82 source receipt file is missing: {relative}")
        source_hashes[relative] = file_sha256(path)
    source_receipt: dict[str, Any] = {
        "schema_version": "v82-source-receipt/v1",
        "files": source_hashes,
        "bundle_sha256": canonical_sha256(source_hashes),
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
        },
    }
    source_receipt["source_receipt_sha256"] = canonical_sha256(source_receipt)
    return {
        "root": root,
        "dataset": dataset,
        "contract": contract,
        "phase_path": phase_path,
        "input_paths": input_paths,
        "expected_hashes": expected_hashes,
        "input_hashes": observed_hashes,
        "values": values,
        "source_receipt": source_receipt,
    }


def run_low_turnover_rank_dataset(config: dict[str, Any]) -> dict[str, Any]:
    ledger = LowTurnoverDatasetLedger()
    context = _metadata_context(config, ledger)
    root = context["root"]
    contract = context["contract"]
    dataset = context["dataset"]
    source_contract = contract["source_contract"]
    symbols = list(source_contract["symbols"])
    target_overlap = TARGET_SYMBOLS.intersection(symbols)
    ledger.target_asset_loads += len(target_overlap)
    if target_overlap:
        raise RuntimeError(f"V82 target assets present in source universe: {sorted(target_overlap)}")

    output = _project_path(root, config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with _process_lock(root / "data" / "processed" / ".v82-low-turnover-dataset.lock"):
        pairs, missing_months = discover_frozen_source_pairs(
            source_contract,
            timeout=float(dataset["timeout_seconds"]),
            workers=int(dataset["listing_workers"]),
            ledger=ledger,
        )
        records, archive_rejections = verify_frozen_source_archives(
            pairs,
            source_contract,
            raw_dir=_project_path(root, dataset["raw_dir"]),
            timeout=float(dataset["timeout_seconds"]),
            workers=int(dataset["archive_workers"]),
            ledger=ledger,
        )
        for rejection in archive_rejections:
            missing_months[rejection["symbol"]].append(rejection["month"])
            missing_months[rejection["symbol"]] = sorted(
                set(missing_months[rejection["symbol"]])
            )
        source_manifest_path = output / "source_manifest.jsonl"
        _write_jsonl_atomic(source_manifest_path, records)
        source_manifest_sha256 = file_sha256(source_manifest_path)

        frames, raw_audit = load_frozen_frames(
            records,
            symbols,
            _project_path(root, dataset["raw_dir"]),
            str(source_contract["frequency"]),
            int(dataset["archive_workers"]),
        )
        source_index = pd.date_range(
            source_contract["source_calendar_start"],
            source_contract["source_calendar_end"],
            freq="D",
            tz="UTC",
        )
        feature_frames = {
            symbol: build_causal_feature_frame(
                symbol,
                frames[symbol],
                source_index,
                lookback_days=int(contract["feature_contract"]["lookback_days"]),
            )
            for symbol in symbols
        }
        chronology = contract["role_contract"]
        development_end = pd.Timestamp(
            chronology["internal_validation"]["signal_end"], tz="UTC"
        )
        development_features = (
            pd.concat(
                [
                    frame.loc[frame["date"] <= development_end]
                    for frame in feature_frames.values()
                ],
                ignore_index=True,
            )
            .sort_values(["date", "symbol"])
            .reset_index(drop=True)
        )
        folds = context["values"]["v32_asset_folds"]["folds"]
        labels = build_development_labels(
            frames,
            symbols,
            source_index,
            feature_frames,
            chronology,
            contract["label_contract"],
            folds,
        )
        ledger.training_label_rows_materialized = int((labels["role"] == "train").sum())
        ledger.internal_validation_label_rows_materialized = int(
            (labels["role"] == "internal_validation").sum()
        )
        evaluation = contract["evaluation_contract"]
        evaluation_start = pd.Timestamp(evaluation["feature_lookback_start"], tz="UTC")
        evaluation_end = pd.Timestamp(evaluation["feature_end"], tz="UTC")
        evaluation_features = (
            pd.concat(
                [
                    frame.loc[frame["date"].between(evaluation_start, evaluation_end)]
                    for frame in feature_frames.values()
                ],
                ignore_index=True,
            )
            .sort_values(["date", "symbol"])
            .reset_index(drop=True)
        )
        sealed_outcomes = build_sealed_daily_outcome_packet(
            frames, symbols, source_index, evaluation
        )
        ledger.sealed_evaluation_outcome_rows_materialized = len(sealed_outcomes)

        output_contract = contract["output_contract"]
        engine = str(output_contract["parquet_engine"])
        compression = str(output_contract["compression"])
        writes = {
            "development_features": _write_parquet_with_fresh_replay(
                development_features,
                _project_path(root, output_contract["development_features_path"]),
                engine=engine,
                compression=compression,
                ledger=ledger,
            ),
            "development_labels": _write_parquet_with_fresh_replay(
                labels,
                _project_path(root, output_contract["development_labels_path"]),
                engine=engine,
                compression=compression,
                ledger=ledger,
            ),
            "evaluation_features": _write_parquet_with_fresh_replay(
                evaluation_features,
                _project_path(root, output_contract["evaluation_features_path"]),
                engine=engine,
                compression=compression,
                ledger=ledger,
            ),
            "sealed_evaluation_outcomes": _write_parquet_with_fresh_replay(
                sealed_outcomes,
                _project_path(root, output_contract["sealed_evaluation_outcomes_path"]),
                engine=engine,
                compression=compression,
                ledger=ledger,
            ),
        }
        ledger.sealed_evaluation_outcome_packet_writes = 1

    input_hashes_after = {
        name: file_sha256(path) for name, path in context["input_paths"].items()
    }
    expected_source_rows = sum(int(record["row_count"]) for record in records)
    expected_development_dates = pd.date_range(
        pd.Timestamp(source_contract["source_calendar_start"], tz="UTC"),
        development_end,
        freq="D",
    )
    expected_evaluation_dates = pd.date_range(
        evaluation_start, evaluation_end, freq="D", tz="UTC"
    )
    expected_signal_dates = pd.date_range(
        evaluation["signal_start"], evaluation["signal_end"], freq="D", tz="UTC"
    )
    expected_outcome_dates = pd.date_range(
        evaluation["daily_outcome_interval_start"],
        evaluation["daily_outcome_interval_end"],
        freq="D",
        tz="UTC",
    )
    feature_columns = list(contract["feature_contract"]["columns"])
    label_target = labels["target_21d_open_to_open_log_return"].to_numpy(dtype=np.float64)
    forbidden_evaluation_columns = {
        "target_21d_open_to_open_log_return",
        "open_to_next_open_log_return",
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "raw_volume",
        "raw_quote_volume",
        "raw_trade_count",
    }
    fold_roles_exact = all(
        len(fold["train_symbols"]) == 20 and len(fold["test_symbols"]) == 10
        for fold in folds
    )
    operation_ledger = ledger.to_dict()
    checks = {
        "all_static_input_hashes_match": input_hashes_after == context["expected_hashes"],
        "exact_user_authorization_and_v82_r0_result_hashes": _self_hash_matches(
            context["values"]["v82_user_authorization"],
            "authorization_sha256",
            contract["input_contract"]["expected_canonical_sha256"]["v82_user_authorization"],
        )
        and _self_hash_matches(
            context["values"]["v82_r0_result"],
            "result_sha256",
            contract["input_contract"]["expected_canonical_sha256"]["v82_r0_result"],
        ),
        "exact_v32_universe_fold_triplet_and_feature_ancestry": len(symbols) == 30
        and fold_roles_exact
        and context["values"]["v32_triplet_catalog"].get("catalog_sha256")
        == contract["input_contract"]["expected_canonical_sha256"]["v32_triplet_catalog"]
        and feature_columns == list(PANEL_FEATURES),
        "exact_official_source_contract_and_checksums": len(records)
        + len(archive_rejections)
        == len(pairs)
        and all(record["checksum_verified"] and record["schema_valid"] for record in records)
        and source_contract["month_start"] == "2018-01"
        and source_contract["month_end"] == "2026-06",
        "exact_non_target_universe": set(frames) == set(symbols)
        and not TARGET_SYMBOLS.intersection(symbols)
        and ledger.target_asset_loads == 0,
        "source_rows_and_manifest_are_hash_registered": raw_audit["source_rows"]
        == expected_source_rows
        == ledger.authorized_raw_source_rows
        and len(source_manifest_sha256) == 64,
        "development_calendar_is_complete_and_unique": len(development_features)
        == len(expected_development_dates) * len(symbols)
        and not development_features.duplicated(["date", "symbol"]).any(),
        "evaluation_calendar_is_complete_and_unique": len(evaluation_features)
        == len(expected_evaluation_dates) * len(symbols)
        and not evaluation_features.duplicated(["date", "symbol"]).any(),
        "missing_rows_are_preserved_without_imputation": ledger.missing_value_imputations == 0
        and all(
            len(feature_frames[symbol]) == len(source_index) for symbol in symbols
        ),
        "eight_features_are_causal_finite_or_missing_and_frozen_order": not np.isinf(
            development_features[feature_columns].to_numpy(dtype=np.float64)
        ).any()
        and feature_columns == list(PANEL_FEATURES),
        "exact_128_day_sequence_readiness": int(contract["feature_contract"]["lookback_days"])
        == 128
        and bool(
            (
                development_features["sequence_start_date"]
                == development_features["date"] - pd.Timedelta(days=127)
            ).all()
        ),
        "exact_t_plus_1_to_t_plus_22_label_geometry": not np.isinf(label_target).any()
        and bool(
            (labels["execution_open_date"] == labels["signal_date"] + pd.Timedelta(days=1)).all()
        )
        and bool((labels["exit_open_date"] == labels["signal_date"] + pd.Timedelta(days=22)).all()),
        "exact_train_validation_roles_and_embargo": set(labels["role"]) == {
            "train",
            "internal_validation",
        }
        and labels.loc[labels["role"] == "train", "signal_date"].max()
        == pd.Timestamp("2023-11-18", tz="UTC")
        and labels.loc[labels["role"] == "internal_validation", "signal_date"].min()
        == pd.Timestamp("2024-01-01", tz="UTC")
        and fold_roles_exact,
        "evaluation_features_are_outcome_free": not forbidden_evaluation_columns.intersection(
            evaluation_features.columns
        ),
        "exact_159_signal_dates_and_last_maturity": len(expected_signal_dates) == 159
        and expected_signal_dates[-1] == pd.Timestamp("2026-06-08", tz="UTC")
        and expected_signal_dates[-1] + pd.Timedelta(days=22)
        == pd.Timestamp("2026-06-30", tz="UTC"),
        "sealed_daily_outcome_packet_geometry": len(sealed_outcomes)
        == len(expected_outcome_dates) * len(symbols)
        and sealed_outcomes["interval_start_date"].min()
        == pd.Timestamp("2026-01-02", tz="UTC")
        and sealed_outcomes["interval_end_date"].max()
        == pd.Timestamp("2026-06-30", tz="UTC")
        and not sealed_outcomes.duplicated(["interval_start_date", "symbol"]).any(),
        "sealed_outcome_packet_was_not_unsealed_or_evaluated": ledger.sealed_evaluation_outcome_packet_writes
        == 1
        and ledger.sealed_evaluation_outcome_packet_unseals == 0
        and ledger.performance_metrics == ledger.pnl_evaluations == ledger.bootstrap_paths == 0,
        "no_scaler_model_checkpoint_training_inference_prediction_or_position": ledger.forbidden_operations_are_zero(),
        "all_four_parquets_replay_byte_identically": all(
            write["byte_identical"]
            and write["sha256"] == write["fresh_replay_sha256"]
            for write in writes.values()
        ),
        "phase_contract_hash_matches": file_sha256(context["phase_path"])
        == dataset["phase_contract"]["file_sha256"],
    }
    checks = {name: bool(value) for name, value in checks.items()}
    audit = {
        "schema_version": "v82-low-turnover-rank-dataset-audit/v1",
        "passed": all(checks.values()),
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "access_ledger": operation_ledger,
    }
    decision = contract["pass_action"] if audit["passed"] else contract["failure_action"]

    feature_schema: dict[str, Any] = {
        "schema_version": "v82-low-turnover-rank-feature-schema/v1",
        "feature_order": feature_columns,
        "lookback_days": 128,
        "availability": "close_of_t",
        "sequence_rule": "exact_128_consecutive_complete_daily_feature_rows",
        "triplet_relative_strength": contract["feature_contract"]["triplet_relative_strength"],
        "label": contract["label_contract"],
        "missing_policy": contract["feature_contract"]["missing_policy"],
        "scaling": "not_applied_in_v82",
    }
    feature_schema["feature_schema_sha256"] = canonical_sha256(feature_schema)
    dataset_spec: dict[str, Any] = {
        "schema_version": "v82-low-turnover-rank-dataset-spec/v1",
        "family_id": contract["family_id"],
        "phase_contract_file_sha256": dataset["phase_contract"]["file_sha256"],
        "source_contract": source_contract,
        "feature_contract": contract["feature_contract"],
        "label_contract": contract["label_contract"],
        "role_contract": contract["role_contract"],
        "evaluation_contract": contract["evaluation_contract"],
        "ancestry_contract": contract["ancestry_contract"],
        "output_contract": contract["output_contract"],
        "pass_action": contract["pass_action"],
        "failure_action": contract["failure_action"],
    }
    dataset_spec["dataset_spec_sha256"] = canonical_sha256(dataset_spec)
    input_receipt = {
        name: {
            "path": str(path.relative_to(root)),
            "sha256": context["input_hashes"][name],
        }
        for name, path in sorted(context["input_paths"].items())
    }
    source_audit = {
        "schema_version": "v82-source-audit/v1",
        "provider": source_contract["provider"],
        "source_manifest_sha256": source_manifest_sha256,
        "verified_archive_count": len(records),
        "rejected_archive_count": len(archive_rejections),
        "archive_rejections": archive_rejections,
        "verified_source_rows": expected_source_rows,
        "expected_months_per_symbol": len(
            _month_range(source_contract["month_start"], source_contract["month_end"])
        ),
        "missing_archive_months_by_symbol": missing_months,
        "missing_archive_month_count": sum(len(value) for value in missing_months.values()),
        "per_symbol": raw_audit["per_symbol"],
        "all_admitted_checksums_verified": all(
            bool(record["checksum_verified"]) for record in records
        ),
        "all_admitted_schemas_valid": all(bool(record["schema_valid"]) for record in records),
        "target_assets_loaded": [],
        "missing_rows_imputed": 0,
    }
    replay_receipt: dict[str, Any] = {
        "schema_version": "v82-dataset-replay-receipt/v1",
        "data_files": {
            name: {
                "sha256": write["sha256"],
                "fresh_replay_sha256": write["fresh_replay_sha256"],
                "byte_identical": write["byte_identical"],
            }
            for name, write in writes.items()
        },
        "byte_identical": checks["all_four_parquets_replay_byte_identically"],
    }
    replay_receipt["replay_receipt_sha256"] = canonical_sha256(replay_receipt)
    sealed_packet_receipt: dict[str, Any] = {
        "schema_version": "v82-sealed-evaluation-outcome-packet-receipt/v1",
        "path": output_contract["sealed_evaluation_outcomes_path"],
        "file_sha256": writes["sealed_evaluation_outcomes"]["sha256"],
        "rows": len(sealed_outcomes),
        "columns": list(sealed_outcomes.columns),
        "interval_start": "2026-01-02",
        "interval_end": "2026-06-29",
        "final_maturity": "2026-06-30",
        "status": "sealed",
        "unseal_count": 0,
        "outcome_values_evaluated": 0,
        "future_unseal_requires_new_exact_hash_bound_user_authorization": True,
        "target_assets_loaded": [],
    }
    sealed_packet_receipt["sealed_packet_receipt_sha256"] = canonical_sha256(
        sealed_packet_receipt
    )
    data_access = {
        "schema_version": "v82-data-access/v1",
        "authorized_static_inputs": list(contract["access_contract"]["allowed_inputs"]),
        "static_hash_verifications": len(context["input_paths"]),
        "official_source_contract": source_contract,
        "source_manifest_sha256": source_manifest_sha256,
        "loaded_symbols": symbols,
        "target_assets_loaded": [],
        "evaluation_feature_columns": list(evaluation_features.columns),
        "sealed_outcome_packet_status": "sealed",
        "operation_ledger": operation_ledger,
    }
    dataset_manifest: dict[str, Any] = {
        "schema_version": "v82-low-turnover-rank-dataset-manifest/v1",
        "source_manifest_sha256": source_manifest_sha256,
        "source_archive_count": len(records),
        "source_rows": expected_source_rows,
        "symbols": symbols,
        "feature_schema_sha256": feature_schema["feature_schema_sha256"],
        "asset_folds_file_sha256": context["input_hashes"]["v32_asset_folds"],
        "triplet_catalog_file_sha256": context["input_hashes"]["v32_triplet_catalog"],
        "data_files": {
            name: {
                **write,
                "path": output_contract[f"{name}_path"],
            }
            for name, write in writes.items()
        },
        "role_rows": {
            "train": ledger.training_label_rows_materialized,
            "internal_validation": ledger.internal_validation_label_rows_materialized,
        },
        "evaluation_signal_dates": 159,
        "sealed_evaluation_outcome_rows": len(sealed_outcomes),
    }
    dataset_manifest["dataset_manifest_sha256"] = canonical_sha256(dataset_manifest)
    result: dict[str, Any] = {
        "schema_version": "v82-low-turnover-rank-dataset-result/v1",
        "family_id": contract["family_id"],
        "decision": decision,
        "dataset_spec_sha256": dataset_spec["dataset_spec_sha256"],
        "dataset_manifest_sha256": dataset_manifest["dataset_manifest_sha256"],
        "feature_schema_sha256": feature_schema["feature_schema_sha256"],
        "source_manifest_sha256": source_manifest_sha256,
        "source_receipt_sha256": context["source_receipt"]["source_receipt_sha256"],
        "replay_receipt_sha256": replay_receipt["replay_receipt_sha256"],
        "sealed_packet_receipt_sha256": sealed_packet_receipt[
            "sealed_packet_receipt_sha256"
        ],
        "summary": {
            "symbol_count": len(symbols),
            "verified_source_archives": len(records),
            "development_feature_rows": len(development_features),
            "development_label_rows": len(labels),
            "training_label_rows": ledger.training_label_rows_materialized,
            "internal_validation_label_rows": ledger.internal_validation_label_rows_materialized,
            "evaluation_feature_rows": len(evaluation_features),
            "evaluation_signal_dates": len(expected_signal_dates),
            "sealed_evaluation_outcome_rows": len(sealed_outcomes),
            "sealed_evaluation_outcome_packet_unseals": ledger.sealed_evaluation_outcome_packet_unseals,
            "target_asset_loads": ledger.target_asset_loads,
            "scaler_fits": ledger.scaler_fits,
            "model_instantiations": ledger.model_instantiations,
            "checkpoint_reads": ledger.checkpoint_reads,
            "checkpoint_writes": ledger.checkpoint_writes,
            "optimizer_steps": ledger.optimizer_steps,
            "training_epochs": ledger.training_epochs,
            "market_predictions": ledger.market_predictions,
            "positions_constructed": ledger.positions_constructed,
            "performance_metrics": ledger.performance_metrics,
            "pnl_evaluations": ledger.pnl_evaluations,
            "bootstrap_paths": ledger.bootstrap_paths,
        },
        "audit": audit,
        "deployable": False,
        "target_assets_loaded": [],
        "v83_executed": False,
    }
    result["result_sha256"] = canonical_sha256(result)
    report = "\n".join(
        [
            "# V82 Non-target Low-turnover Rank Dataset",
            "",
            f"Decision: **{decision}**",
            "",
            f"Verified official archives: **{len(records):,}**",
            f"Development feature rows: **{len(development_features):,}**",
            f"Development label rows: **{len(labels):,}**",
            f"Evaluation feature rows: **{len(evaluation_features):,}**",
            f"Sealed 2026 outcome rows: **{len(sealed_outcomes):,}**",
            f"Source manifest SHA-256: `{source_manifest_sha256}`",
            f"Sealed outcome packet SHA-256: `{writes['sealed_evaluation_outcomes']['sha256']}`",
            "",
            "The exact V32 non-target universe, folds, triplets, eight causal features,",
            "128-day lookback, and 21-interval open-to-open target were preserved.",
            "Missing observations remain missing and break sequence eligibility.",
            "",
            "The evaluation feature table contains no outcome or target column. The",
            "separate 2026 daily outcome packet was hash-sealed with unseal count zero.",
            "No scaler, model, checkpoint, training, inference, prediction, position,",
            "performance metric, PnL, bootstrap, or BTC/ETH/SOL access occurred.",
            "A pass authorizes only a separately governed V83 training phase.",
            "",
        ]
    )

    write_yaml_atomic(output / "resolved_config.yaml", config)
    write_json_atomic(output / "input_hash_receipt.json", input_receipt)
    write_json_atomic(output / "source_receipt.json", context["source_receipt"])
    write_json_atomic(output / "source_audit.json", source_audit)
    write_json_atomic(output / "dataset_spec.json", dataset_spec)
    write_json_atomic(output / "feature_schema.json", feature_schema)
    write_json_atomic(output / "dataset_manifest.json", dataset_manifest)
    write_json_atomic(output / "data_access.json", data_access)
    write_json_atomic(output / "sealed_packet_receipt.json", sealed_packet_receipt)
    write_json_atomic(output / "replay_receipt.json", replay_receipt)
    write_json_atomic(output / "audit.json", audit)
    write_json_atomic(output / "result.json", result)
    _write_text_atomic(output / "report.md", report)
    manifest_names = [
        name
        for name in output_contract["packet_files"]
        if name != "artifact_manifest.json"
    ]
    artifact_manifest: dict[str, Any] = {
        "schema_version": "v82-artifact-manifest/v1",
        "files": {name: file_sha256(output / name) for name in manifest_names},
        "data_files": {
            output_contract[f"{name}_path"]: write["sha256"]
            for name, write in writes.items()
        },
    }
    artifact_manifest["artifact_manifest_sha256"] = canonical_sha256(
        artifact_manifest
    )
    write_json_atomic(output / "artifact_manifest.json", artifact_manifest)
    actual_files = sorted(path.name for path in output.iterdir() if path.is_file())
    if actual_files != sorted(output_contract["packet_files"]):
        raise RuntimeError(f"V82 artifact packet file-set drift: {actual_files}")
    if not audit["passed"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"V82 dataset audit failed: {failed}")
    return result
