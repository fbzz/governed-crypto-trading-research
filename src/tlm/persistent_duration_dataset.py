from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd
import pyarrow
import yaml

from .core import (
    DatasetAccessLedger,
    canonical_sha256,
    file_sha256,
    write_json_atomic,
    write_yaml_atomic,
)


TARGET_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
JSON_INPUTS = {
    "v75_result",
    "v75_audit",
    "v75_harness_spec",
    "v75_replay_receipt",
    "v75_artifact_manifest",
    "v75_source_receipt",
    "v32_result",
    "v32_audit",
    "v32_dataset_manifest",
    "v32_feature_schema",
    "v32_asset_folds",
    "v32_triplet_catalog",
}
PARQUET_INPUTS = {"panel", "sequence_index"}
HORIZON_DAYS = {"h1": 1, "h3": 3, "h7": 7}


def _project_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"V76 path escapes project root: {relative}") from exc
    return path


def _load_json(path: Path, ledger: DatasetAccessLedger) -> dict[str, Any]:
    ledger.authorized_metadata_reads += 1
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {path}")
    return value


def _load_yaml(path: Path, ledger: DatasetAccessLedger) -> dict[str, Any]:
    ledger.authorized_metadata_reads += 1
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def _registered_hash(value: dict[str, Any], field: str, expected: str) -> bool:
    payload = dict(value)
    registered = payload.pop(field, None)
    return registered == expected == canonical_sha256(payload)


@contextmanager
def _process_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another V76 dataset process holds the lock") from exc
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        path.unlink(missing_ok=True)


def build_persistent_duration_labels(
    panel: pd.DataFrame,
    label_contract: dict[str, Any],
    output_columns: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    required = {"date", "symbol", "raw_open", "raw_observation_available"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"V76 source panel is missing columns: {missing}")
    if list(label_contract["return_order"]) != ["h1", "h3", "h7"]:
        raise ValueError("V76 return horizon order drift")
    if list(label_contract["duration_support_days"]) != list(range(1, 8)):
        raise ValueError("V76 duration support drift")
    expected_definitions = {
        "h1": "log(open[t+2] / open[t+1])",
        "h3": "log(open[t+4] / open[t+1])",
        "h7": "log(open[t+8] / open[t+1])",
    }
    if label_contract["cumulative_return_definitions"] != expected_definitions:
        raise ValueError("V76 cumulative return definition drift")

    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame["symbol"] = frame["symbol"].astype(str)
    frame = frame.sort_values(["date", "symbol"]).reset_index(drop=True)
    if frame.duplicated(["symbol", "date"]).any():
        raise ValueError("V76 source panel key is not unique")
    for _, group in frame.groupby("symbol", sort=False):
        dates = pd.DatetimeIndex(group["date"])
        if len(dates) > 1 and not bool(
            (dates[1:] - dates[:-1] == pd.Timedelta(days=1)).all()
        ):
            raise ValueError("V76 source panel is not a complete daily calendar")

    grouped_open = frame.groupby("symbol", sort=False)["raw_open"]
    start_open = grouped_open.shift(-1).to_numpy(dtype=np.float64)
    cumulative = np.full((len(frame), 7), np.nan, dtype=np.float64)
    for day in range(1, 8):
        end_open = grouped_open.shift(-(day + 1)).to_numpy(dtype=np.float64)
        valid = (
            np.isfinite(start_open)
            & np.isfinite(end_open)
            & (start_open > 0.0)
            & (end_open > 0.0)
        )
        cumulative[valid, day - 1] = np.log(end_open[valid] / start_open[valid])

    output = frame[["date", "symbol"]].copy()
    output["eligible_action_date"] = output["date"] + pd.Timedelta(days=1)
    for horizon, day in HORIZON_DAYS.items():
        maturity_offset = day + 1
        output[f"target_{horizon}_maturity_date"] = output["date"] + pd.Timedelta(
            days=maturity_offset
        )
    output["target_duration_maturity_date"] = output["date"] + pd.Timedelta(
        days=8
    )

    for horizon, day in HORIZON_DAYS.items():
        column = f"target_{horizon}_open_to_open_log_return"
        output[column] = cumulative[:, day - 1]
        output[f"{horizon}_label_complete"] = np.isfinite(output[column])

    duration_complete = np.isfinite(cumulative).all(axis=1)
    duration_days = np.zeros(len(frame), dtype=np.int8)
    duration_days[duration_complete] = (
        np.argmax(cumulative[duration_complete], axis=1) + 1
    ).astype(np.int8)
    duration_values = pd.array([pd.NA] * len(frame), dtype="Int8")
    duration_values[duration_complete] = duration_days[duration_complete]
    censored_values = pd.array([pd.NA] * len(frame), dtype="boolean")
    censored_values[duration_complete] = (
        duration_days[duration_complete] == 7
    )
    output["target_duration_days"] = duration_values
    output["duration_right_censored"] = censored_values
    output["duration_label_complete"] = duration_complete
    completion_columns = [
        "h1_label_complete",
        "h3_label_complete",
        "h7_label_complete",
        "duration_label_complete",
    ]
    output["persistent_label_complete"] = output[completion_columns].all(axis=1)
    if set(output.columns) != set(output_columns) or len(output.columns) != len(
        output_columns
    ):
        raise ValueError("V76 output label schema drift")
    return output[output_columns].copy(), cumulative


def build_persistent_sequence_roles(
    sequence_index: pd.DataFrame,
    labels: pd.DataFrame,
    role_contract: dict[str, Any],
    output_contract: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    required = {"date", "sequence_start_date", "symbol"}
    missing = sorted(required - set(sequence_index.columns))
    if missing:
        raise ValueError(f"V76 sequence index is missing columns: {missing}")
    sequence = sequence_index[["date", "sequence_start_date", "symbol"]].copy()
    sequence["date"] = pd.to_datetime(sequence["date"], utc=True)
    sequence["sequence_start_date"] = pd.to_datetime(
        sequence["sequence_start_date"], utc=True
    )
    sequence["symbol"] = sequence["symbol"].astype(str)
    sequence = sequence.sort_values(["date", "symbol"]).reset_index(drop=True)
    if sequence.duplicated(["symbol", "date"]).any():
        raise ValueError("V76 source sequence key is not unique")

    completion_columns = [
        "h1_label_complete",
        "h3_label_complete",
        "h7_label_complete",
        "duration_label_complete",
        "persistent_label_complete",
    ]
    merged = sequence.merge(
        labels[["date", "symbol", *completion_columns]],
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    for column in completion_columns:
        merged[column] = merged[column].astype("boolean").fillna(False).astype(bool)

    role_audit: dict[str, dict[str, Any]] = {}
    role_names = ("train", "internal_validation")
    for role in role_names:
        window = role_contract[role]
        start = pd.Timestamp(window["signal_start"], tz="UTC")
        end = pd.Timestamp(window["signal_end"], tz="UTC")
        maturity_end = pd.Timestamp(window["maturity_end"], tz="UTC")
        if end + pd.Timedelta(days=8) != maturity_end:
            raise ValueError(f"V76 {role} boundary is not purged by eight days")
        column = f"eligible_{role}"
        flag = (
            merged["persistent_label_complete"]
            & merged["date"].between(start, end, inclusive="both")
            & (merged["date"] + pd.Timedelta(days=8) <= maturity_end)
        )
        merged[column] = flag
        eligible = merged.loc[flag]
        role_audit[column] = {
            "labels_materialized": True,
            "signal_start": window["signal_start"],
            "signal_end": window["signal_end"],
            "maturity_end": window["maturity_end"],
            "eligible_rows": int(flag.sum()),
            "eligible_dates": int(eligible["date"].nunique()),
            "first_eligible_date": (
                eligible["date"].min().date().isoformat() if len(eligible) else None
            ),
            "last_eligible_date": (
                eligible["date"].max().date().isoformat() if len(eligible) else None
            ),
            "maximum_target_maturity": (
                (eligible["date"].max() + pd.Timedelta(days=8)).date().isoformat()
                if len(eligible)
                else None
            ),
        }

    evaluation = role_contract["adaptive_development_evaluation"]
    evaluation_start = pd.Timestamp(evaluation["signal_start"], tz="UTC")
    evaluation_end = pd.Timestamp(evaluation["signal_end"], tz="UTC")
    evaluation_maturity_end = pd.Timestamp(evaluation["maturity_end"], tz="UTC")
    if evaluation_end + pd.Timedelta(days=8) != evaluation_maturity_end:
        raise ValueError("V76 evaluation boundary is not purged by eight days")
    evaluation_column = "eligible_adaptive_development_evaluation"
    evaluation_flag = merged["date"].between(
        evaluation_start, evaluation_end, inclusive="both"
    )
    merged[evaluation_column] = evaluation_flag
    eligible_evaluation = merged.loc[evaluation_flag]
    role_audit[evaluation_column] = {
        "labels_materialized": False,
        "role_source": "sequence_index_dates_only",
        "signal_start": evaluation["signal_start"],
        "signal_end": evaluation["signal_end"],
        "maturity_end": evaluation["maturity_end"],
        "eligible_rows": int(evaluation_flag.sum()),
        "eligible_dates": int(eligible_evaluation["date"].nunique()),
        "first_eligible_date": (
            eligible_evaluation["date"].min().date().isoformat()
            if len(eligible_evaluation)
            else None
        ),
        "last_eligible_date": (
            eligible_evaluation["date"].max().date().isoformat()
            if len(eligible_evaluation)
            else None
        ),
        "maximum_target_maturity": evaluation["maturity_end"],
    }
    base_columns = list(output_contract["sequence_base_columns"])
    physical_flags = list(role_contract["physical_flags"])
    if physical_flags != [
        "eligible_train",
        "eligible_internal_validation",
        evaluation_column,
    ]:
        raise ValueError("V76 physical role flag order drift")
    return merged[base_columns + physical_flags].copy(), role_audit


def _write_parquet_with_fresh_replay(
    frame: pd.DataFrame,
    final_path: Path,
    *,
    engine: str,
    compression: str,
    ledger: DatasetAccessLedger,
) -> dict[str, Any]:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    first = final_path.with_name(f".{final_path.name}.v76-replay-a.tmp")
    second = final_path.with_name(f".{final_path.name}.v76-replay-b.tmp")
    first.unlink(missing_ok=True)
    second.unlink(missing_ok=True)
    try:
        frame.to_parquet(first, index=False, engine=engine, compression=compression)
        frame.to_parquet(second, index=False, engine=engine, compression=compression)
        ledger.parquet_writes += 2
        first_hash = file_sha256(first)
        second_hash = file_sha256(second)
        if first_hash != second_hash:
            raise RuntimeError(f"V76 fresh Parquet replay drift: {final_path.name}")
        os.replace(first, final_path)
        return {
            "path": str(final_path),
            "sha256": first_hash,
            "fresh_replay_sha256": second_hash,
            "byte_identical": True,
            "rows": len(frame),
            "columns": list(frame.columns),
        }
    finally:
        first.unlink(missing_ok=True)
        second.unlink(missing_ok=True)


def _metadata_context(
    config: dict[str, Any], ledger: DatasetAccessLedger
) -> dict[str, Any]:
    dataset = config["persistent_duration_dataset"]
    root = Path(dataset["project_root"]).resolve()
    contract_reference = dataset["phase_contract"]
    contract_path = _project_path(root, contract_reference["path"])
    if (
        not contract_path.is_file()
        or file_sha256(contract_path) != contract_reference["file_sha256"]
    ):
        raise RuntimeError("V76 phase contract is missing or hash-drifted")
    contract = _load_yaml(contract_path, ledger)
    if (
        contract.get("phase") != "v76"
        or contract.get("stage_revision")
        != "v076_non_target_persistent_duration_dataset_r2"
        or contract.get("authorized_next_action")
        != "authorize_v76_non_target_persistent_duration_dataset_only"
        or config.get("output_dir") != contract["access_contract"]["output_dir"]
    ):
        raise RuntimeError("V76 frozen phase contract is inconsistent")

    input_paths = {
        name: _project_path(root, relative)
        for name, relative in dataset["inputs"].items()
    }
    if set(input_paths) != JSON_INPUTS | PARQUET_INPUTS:
        raise RuntimeError("V76 input-name allowlist drift")
    if set(dataset["inputs"].values()) != set(
        contract["access_contract"]["allowed_inputs"]
    ):
        raise RuntimeError("V76 input-path allowlist drift")
    expected_by_path = contract["input_contract"][
        "expected_static_file_sha256_by_path"
    ]
    expected_hashes = {
        name: expected_by_path[relative]
        for name, relative in dataset["inputs"].items()
    }
    if set(expected_hashes) != set(input_paths):
        raise RuntimeError("V76 expected-input hash map drift")
    observed_hashes: dict[str, str] = {}
    for name, path in input_paths.items():
        if not path.is_file():
            raise RuntimeError(f"V76 input is missing: {name}")
        observed_hashes[name] = file_sha256(path)
        if observed_hashes[name] != expected_hashes[name]:
            raise RuntimeError(f"V76 input hash drift: {name}")

    values = {name: _load_json(input_paths[name], ledger) for name in JSON_INPUTS}
    canonical = contract["input_contract"]["expected_canonical_sha256"]
    v75_result = values["v75_result"]
    v75_audit = values["v75_audit"]
    v75_harness_spec = values["v75_harness_spec"]
    v75_replay = values["v75_replay_receipt"]
    v75_manifest = values["v75_artifact_manifest"]
    v75_source = values["v75_source_receipt"]
    v32_result = values["v32_result"]
    v32_audit = values["v32_audit"]
    v32_manifest = values["v32_dataset_manifest"]
    v32_schema = values["v32_feature_schema"]
    folds = values["v32_asset_folds"]
    catalog = values["v32_triplet_catalog"]
    if (
        not _registered_hash(
            v75_harness_spec,
            "harness_spec_sha256",
            canonical["v75_harness_spec"],
        )
        or not _registered_hash(
            v75_result, "result_sha256", canonical["v75_result"]
        )
        or not _registered_hash(
            v75_replay,
            "replay_receipt_sha256",
            canonical["v75_replay_receipt"],
        )
        or not _registered_hash(
            v75_manifest,
            "artifact_manifest_sha256",
            canonical["v75_artifact_manifest"],
        )
        or not _registered_hash(
            v75_source,
            "source_receipt_sha256",
            canonical["v75_source_receipt"],
        )
        or v75_audit.get("passed") is not True
        or len(v75_audit.get("checks", {})) != 17
        or not all(v75_audit.get("checks", {}).values())
        or v75_result.get("decision")
        != "authorize_v76_non_target_persistent_duration_dataset_only"
        or v75_replay.get("byte_identical") is not True
        or v32_audit.get("passed") is not True
        or v32_result.get("audit", {}).get("checks", {}).get(
            "no_target_symbol_loaded"
        )
        is not True
        or v32_manifest.get("panel_sha256") != observed_hashes["panel"]
        or v32_manifest.get("sequence_index_sha256")
        != observed_hashes["sequence_index"]
        or v32_result.get("dataset_manifest") != v32_manifest
        or list(v32_schema.get("model_feature_order", []))
        != list(contract["feature_contract"]["model_feature_order"])
        or len(folds.get("folds", [])) != 3
        or len(catalog.get("folds", [])) != 3
    ):
        raise RuntimeError("V76 parent metadata contract drift")
    for fold, catalog_fold in zip(folds["folds"], catalog["folds"], strict=True):
        if (
            int(fold["fold"]) != int(catalog_fold["fold"])
            or fold["train_symbols"] != catalog_fold["train_symbols"]
            or fold["test_symbols"] != catalog_fold["test_symbols"]
            or len(catalog_fold["train_triplets"]) != math.comb(20, 3)
            or len(catalog_fold["test_triplets"]) != math.comb(10, 3)
        ):
            raise RuntimeError("V76 frozen fold/catalog roles drifted")

    correction = contract["source_receipt_correction"]
    hardening = contract["pre_deserialization_hardening"]
    if (
        correction["authoritative_v32_value"] != observed_hashes["sequence_index"]
        or len(correction["malformed_v74_value"]) != 61
        or len(correction["authoritative_v32_value"]) != 64
        or correction["panel_or_sequence_deserializations_during_registration"]
        != 0
        or hardening["maximum_panel_value_date"] != "2024-12-31"
        or hardening["adaptive_evaluation_label_values_materialized"] is not False
        or hardening[
            "first_v76_parquet_deserialization_completed_before_hardening"
        ]
        is not False
    ):
        raise RuntimeError("V76 source correction or hardening drift")

    source_files = list(dataset["source_receipt_files"])
    if not source_files or len(source_files) != len(set(source_files)):
        raise RuntimeError("V76 source receipt is empty or duplicated")
    source_hashes: dict[str, str] = {}
    for relative in source_files:
        path = _project_path(root, relative)
        if not path.is_file():
            raise RuntimeError(f"V76 source receipt file is missing: {relative}")
        source_hashes[relative] = file_sha256(path)
    source_receipt: dict[str, Any] = {
        "schema_version": "v76-source-receipt/v1",
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
        "contract_path": contract_path,
        "input_paths": input_paths,
        "input_hashes": observed_hashes,
        "expected_hashes": expected_hashes,
        "values": values,
        "source_receipt": source_receipt,
    }


def _read_inputs(
    context: dict[str, Any],
    ledger: DatasetAccessLedger,
    *,
    reader: Callable[..., pd.DataFrame] = pd.read_parquet,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    contract = context["contract"]
    access = contract["data_access_contract"]
    cutoff = pd.Timestamp(access["panel_filter"]["value"], tz="UTC")
    panel = reader(
        context["input_paths"]["panel"],
        engine="pyarrow",
        columns=list(access["panel_projection"]),
        filters=[("date", "<=", cutoff)],
    )
    ledger.authorized_parquet_deserializations += 1
    ledger.authorized_panel_rows += len(panel)
    sequence = reader(
        context["input_paths"]["sequence_index"],
        engine="pyarrow",
        columns=list(access["sequence_projection"]),
    )
    ledger.authorized_parquet_deserializations += 1
    ledger.authorized_sequence_rows += len(sequence)
    panel["date"] = pd.to_datetime(panel["date"], utc=True)
    sequence["date"] = pd.to_datetime(sequence["date"], utc=True)
    if panel["date"].max() > cutoff:
        raise RuntimeError("V76 loaded post-2024 panel values")
    if len(panel) != int(
        contract["input_contract"]["admitted_panel_rows_after_outcome_blind_filter"]
    ):
        raise RuntimeError("V76 outcome-blind panel row count drift")
    panel_symbols = set(panel["symbol"].astype(str))
    sequence_symbols = set(sequence["symbol"].astype(str))
    target_loads = TARGET_SYMBOLS.intersection(panel_symbols | sequence_symbols)
    ledger.target_asset_loads += len(target_loads)
    if target_loads:
        raise RuntimeError(f"V76 loaded target symbols: {sorted(target_loads)}")
    if panel_symbols != sequence_symbols:
        raise RuntimeError("V76 panel and sequence universes differ")
    return panel, sequence, sorted(panel_symbols)


def run_persistent_duration_dataset(config: dict[str, Any]) -> dict[str, Any]:
    ledger = DatasetAccessLedger()
    context = _metadata_context(config, ledger)
    root = context["root"]
    contract = context["contract"]
    output = _project_path(root, config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)

    with _process_lock(root / "data" / "processed" / ".v76-dataset.lock"):
        panel, sequence, loaded_symbols = _read_inputs(context, ledger)
        labels, cumulative_returns = build_persistent_duration_labels(
            panel,
            contract["label_contract"],
            list(contract["output_contract"]["labels_columns"]),
        )
        sequence_roles, role_audit = build_persistent_sequence_roles(
            sequence,
            labels,
            contract["role_contract"],
            contract["output_contract"],
        )
        output_contract = contract["output_contract"]
        labels_path = _project_path(root, output_contract["labels_path"])
        sequence_roles_path = _project_path(
            root, output_contract["sequence_roles_path"]
        )
        label_write = _write_parquet_with_fresh_replay(
            labels,
            labels_path,
            engine=output_contract["parquet_engine"],
            compression=output_contract["compression"],
            ledger=ledger,
        )
        sequence_write = _write_parquet_with_fresh_replay(
            sequence_roles,
            sequence_roles_path,
            engine=output_contract["parquet_engine"],
            compression=output_contract["compression"],
            ledger=ledger,
        )

    input_hashes_after = {
        name: file_sha256(path) for name, path in context["input_paths"].items()
    }
    input_contract = contract["input_contract"]
    feature_contract = contract["feature_contract"]
    label_contract = contract["label_contract"]
    v32_manifest = context["values"]["v32_dataset_manifest"]
    folds = context["values"]["v32_asset_folds"]["folds"]
    catalog = context["values"]["v32_triplet_catalog"]["folds"]
    panel_keys = set(zip(panel["date"], panel["symbol"].astype(str), strict=True))
    label_keys = set(zip(labels["date"], labels["symbol"], strict=True))
    sequence_keys = set(
        zip(sequence["date"], sequence["symbol"].astype(str), strict=True)
    )
    role_keys = set(
        zip(sequence_roles["date"], sequence_roles["symbol"], strict=True)
    )
    missing_keys = set(
        zip(
            panel.loc[~panel["raw_observation_available"], "date"],
            panel.loc[~panel["raw_observation_available"], "symbol"].astype(str),
            strict=True,
        )
    )
    exact_fold_roles = all(
        len(fold["train_symbols"]) == 20
        and len(fold["test_symbols"]) == 10
        and len(catalog_fold["train_triplets"]) == math.comb(20, 3)
        and len(catalog_fold["test_triplets"]) == math.comb(10, 3)
        for fold, catalog_fold in zip(folds, catalog, strict=True)
    )
    duration_complete = labels["duration_label_complete"].to_numpy(dtype=bool)
    complete_duration_values = labels.loc[
        duration_complete, "target_duration_days"
    ].to_numpy(dtype=np.int8)
    complete_censor_values = labels.loc[
        duration_complete, "duration_right_censored"
    ].to_numpy(dtype=bool)
    expected_duration_values = (
        np.argmax(cumulative_returns[duration_complete], axis=1) + 1
    ).astype(np.int8)
    train_audit = role_audit["eligible_train"]
    validation_audit = role_audit["eligible_internal_validation"]
    evaluation_audit = role_audit["eligible_adaptive_development_evaluation"]
    operation_ledger = ledger.to_dict()
    checks = {
        "all_input_hashes_match": input_hashes_after
        == context["expected_hashes"],
        "v74_sequence_hash_typo_is_explicitly_corrected_from_authoritative_v32_receipt": len(
            contract["source_receipt_correction"]["malformed_v74_value"]
        )
        == 61
        and len(
            contract["source_receipt_correction"]["authoritative_v32_value"]
        )
        == 64
        and contract["source_receipt_correction"]["authoritative_v32_value"]
        == input_hashes_after["sequence_index"],
        "exact_thirty_asset_universe_and_three_folds": len(loaded_symbols) == 30
        and loaded_symbols == sorted(v32_manifest["symbols"])
        and exact_fold_roles,
        "exact_nine_feature_order": list(
            context["values"]["v32_feature_schema"]["model_feature_order"]
        )
        == list(feature_contract["model_feature_order"])
        and list(v32_manifest["panel_features"])
        == list(feature_contract["panel_columns"]),
        "exact_h1_h3_h7_cumulative_return_labels": list(
            label_contract["return_order"]
        )
        == ["h1", "h3", "h7"]
        and all(
            labels[f"{horizon}_label_complete"].equals(
                labels[f"target_{horizon}_open_to_open_log_return"].notna()
            )
            for horizon in HORIZON_DAYS
        ),
        "exact_earliest_argmax_duration_and_day7_right_censoring": np.array_equal(
            complete_duration_values, expected_duration_values
        )
        and np.array_equal(
            complete_censor_values, complete_duration_values == 7
        )
        and bool(
            ((complete_duration_values >= 1) & (complete_duration_values <= 7)).all()
        ),
        "no_label_crosses_registered_role_boundary": train_audit[
            "maximum_target_maturity"
        ]
        == train_audit["maturity_end"]
        and validation_audit["maximum_target_maturity"]
        == validation_audit["maturity_end"]
        and evaluation_audit["labels_materialized"] is False
        and not bool(
            sequence_roles.loc[
                sequence_roles["eligible_adaptive_development_evaluation"],
                "persistent_label_complete",
            ].any()
        ),
        "maximum_maturity_purge_and_embargo_are_eight_days": label_contract[
            "maximum_label_maturity_days"
        ]
        == label_contract["purge_days"]
        == label_contract["embargo_days"]
        == 8
        and bool(
            (
                labels["target_duration_maturity_date"] - labels["date"]
                == pd.Timedelta(days=8)
            ).all()
        ),
        "panel_and_sequence_keys_are_unique": len(panel_keys) == len(panel)
        == int(input_contract["admitted_panel_rows_after_outcome_blind_filter"])
        and panel_keys == label_keys
        and len(sequence_keys) == len(sequence)
        == int(input_contract["sequence_rows"])
        and sequence_keys == role_keys,
        "missing_rows_are_preserved_without_imputation": missing_keys.issubset(
            label_keys
        )
        and len(labels) == len(panel)
        and len(sequence_roles) == len(sequence)
        and ledger.missing_value_imputations == 0,
        "target_assets_are_absent": not TARGET_SYMBOLS.intersection(loaded_symbols)
        and ledger.target_asset_loads == 0,
        "no_scaler_model_optimizer_checkpoint_prediction_performance_or_pnl": ledger.forbidden_operations_are_zero()
        and ledger.authorized_parquet_deserializations
        == contract["data_access_contract"][
            "authorized_parquet_deserializations_per_execution"
        ]
        and max(panel["date"]) == pd.Timestamp("2024-12-31", tz="UTC")
        and int((panel["date"] >= pd.Timestamp("2025-01-01", tz="UTC")).sum())
        == 0,
        "byte_identical_replay": label_write["byte_identical"]
        and sequence_write["byte_identical"]
        and label_write["sha256"] == label_write["fresh_replay_sha256"]
        and sequence_write["sha256"] == sequence_write["fresh_replay_sha256"],
        "phase_contract_hash_matches": file_sha256(context["contract_path"])
        == context["dataset"]["phase_contract"]["file_sha256"],
        "output_schema_is_exact": list(labels.columns)
        == list(contract["output_contract"]["labels_columns"])
        and list(sequence_roles.columns)
        == list(contract["output_contract"]["sequence_base_columns"])
        + list(contract["role_contract"]["physical_flags"]),
        "source_receipt_is_complete": bool(context["source_receipt"]["files"])
        and len(context["source_receipt"]["bundle_sha256"]) == 64,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    audit = {
        "schema_version": "v76-persistent-duration-dataset-audit/v1",
        "passed": all(checks.values()),
        "checks": checks,
        "operation_ledger": operation_ledger,
    }
    decision = contract["pass_action"] if audit["passed"] else contract["failure_action"]

    dataset_spec: dict[str, Any] = {
        "schema_version": "v76-persistent-duration-dataset-spec/v1",
        "version": "v76",
        "family_id": contract["family_id"],
        "phase_contract_file_sha256": context["dataset"]["phase_contract"][
            "file_sha256"
        ],
        "feature_contract": feature_contract,
        "label_contract": label_contract,
        "role_contract": contract["role_contract"],
        "data_access_contract": contract["data_access_contract"],
        "output_contract": contract["output_contract"],
        "source_receipt_correction": contract["source_receipt_correction"],
        "pre_deserialization_hardening": contract["pre_deserialization_hardening"],
        "pass_action": contract["pass_action"],
        "failure_action": contract["failure_action"],
    }
    dataset_spec["dataset_spec_sha256"] = canonical_sha256(dataset_spec)
    label_schema: dict[str, Any] = {
        "schema_version": "v76-persistent-duration-label-schema/v1",
        "columns": list(labels.columns),
        "dtypes": {name: str(dtype) for name, dtype in labels.dtypes.items()},
        "cumulative_return_definitions": label_contract[
            "cumulative_return_definitions"
        ],
        "duration_definition": label_contract["duration_definition"],
        "duration_tie_break": label_contract["duration_tie_break"],
        "duration_right_censor_rule": label_contract[
            "duration_right_censor_rule"
        ],
        "missing_policy": label_contract["missing_policy"],
    }
    label_schema["label_schema_sha256"] = canonical_sha256(label_schema)
    input_receipt = {
        name: {
            "path": str(path.relative_to(root)),
            "sha256": context["input_hashes"][name],
        }
        for name, path in sorted(context["input_paths"].items())
    }
    data_access = {
        "authorized_inputs": list(contract["access_contract"]["allowed_inputs"]),
        "static_hash_verifications": len(context["input_paths"]),
        "panel_projection": list(
            contract["data_access_contract"]["panel_projection"]
        ),
        "panel_filter": contract["data_access_contract"]["panel_filter"],
        "sequence_projection": list(
            contract["data_access_contract"]["sequence_projection"]
        ),
        "maximum_loaded_panel_value_date": max(panel["date"]).date().isoformat(),
        "adaptive_evaluation_label_values_loaded": False,
        "loaded_symbols": loaded_symbols,
        "operation_ledger": operation_ledger,
    }
    dataset_manifest: dict[str, Any] = {
        "schema_version": "v76-persistent-duration-dataset-manifest/v1",
        "labels": {
            **label_write,
            "path": contract["output_contract"]["labels_path"],
            "complete_h1_rows": int(labels["h1_label_complete"].sum()),
            "complete_h3_rows": int(labels["h3_label_complete"].sum()),
            "complete_h7_rows": int(labels["h7_label_complete"].sum()),
            "complete_duration_rows": int(labels["duration_label_complete"].sum()),
            "complete_persistent_rows": int(labels["persistent_label_complete"].sum()),
            "event_rows": int(
                (
                    labels["duration_label_complete"]
                    & ~labels["duration_right_censored"].fillna(False)
                ).sum()
            ),
            "right_censored_rows": int(
                labels["duration_right_censored"].fillna(False).sum()
            ),
        },
        "sequence_roles": {
            **sequence_write,
            "path": contract["output_contract"]["sequence_roles_path"],
        },
        "role_audit": role_audit,
        "symbols": loaded_symbols,
        "source_panel_sha256": context["input_hashes"]["panel"],
        "source_sequence_index_sha256": context["input_hashes"]["sequence_index"],
        "feature_schema_sha256": context["input_hashes"]["v32_feature_schema"],
        "asset_folds_sha256": context["input_hashes"]["v32_asset_folds"],
        "triplet_catalog_sha256": context["input_hashes"]["v32_triplet_catalog"],
        "label_schema_sha256": label_schema["label_schema_sha256"],
    }
    dataset_manifest["dataset_manifest_sha256"] = canonical_sha256(dataset_manifest)
    replay_receipt: dict[str, Any] = {
        "schema_version": "v76-dataset-replay-receipt/v1",
        "labels_sha256": label_write["sha256"],
        "labels_fresh_replay_sha256": label_write["fresh_replay_sha256"],
        "sequence_roles_sha256": sequence_write["sha256"],
        "sequence_roles_fresh_replay_sha256": sequence_write[
            "fresh_replay_sha256"
        ],
        "byte_identical": checks["byte_identical_replay"],
    }
    replay_receipt["replay_receipt_sha256"] = canonical_sha256(replay_receipt)
    result: dict[str, Any] = {
        "schema_version": "v76-persistent-duration-dataset-result/v1",
        "version": "v76",
        "family_id": contract["family_id"],
        "decision": decision,
        "dataset_spec_sha256": dataset_spec["dataset_spec_sha256"],
        "dataset_manifest_sha256": dataset_manifest["dataset_manifest_sha256"],
        "label_schema_sha256": label_schema["label_schema_sha256"],
        "source_receipt_sha256": context["source_receipt"][
            "source_receipt_sha256"
        ],
        "replay_receipt_sha256": replay_receipt["replay_receipt_sha256"],
        "summary": {
            "label_rows": len(labels),
            "sequence_role_rows": len(sequence_roles),
            "complete_persistent_rows": int(
                labels["persistent_label_complete"].sum()
            ),
            "duration_event_rows": dataset_manifest["labels"]["event_rows"],
            "duration_right_censored_rows": dataset_manifest["labels"][
                "right_censored_rows"
            ],
            "train_eligible_rows": train_audit["eligible_rows"],
            "internal_validation_eligible_rows": validation_audit["eligible_rows"],
            "adaptive_evaluation_role_rows": evaluation_audit["eligible_rows"],
            "authorized_parquet_deserializations": ledger.authorized_parquet_deserializations,
            "adaptive_evaluation_label_values_loaded": False,
            "target_asset_loads": ledger.target_asset_loads,
            "scaler_fits": ledger.scaler_fits,
            "model_instantiations": ledger.model_instantiations,
            "optimizer_steps": ledger.optimizer_steps,
            "checkpoint_reads": ledger.checkpoint_reads,
            "market_predictions": ledger.market_predictions,
            "performance_metrics": ledger.performance_metrics,
            "pnl_evaluations": ledger.pnl_evaluations,
        },
        "audit": audit,
    }
    result["result_sha256"] = canonical_sha256(result)
    report = "\n".join(
        [
            "# V76 Non-target Persistent-Duration Dataset",
            "",
            f"Decision: **{decision}**",
            "",
            f"Label rows: **{len(labels):,}**",
            f"Sequence-role rows: **{len(sequence_roles):,}**",
            f"Complete persistent labels: **{int(labels['persistent_label_complete'].sum()):,}**",
            f"Duration events: **{dataset_manifest['labels']['event_rows']:,}**",
            f"Right-censored durations: **{dataset_manifest['labels']['right_censored_rows']:,}**",
            f"Labels SHA-256: `{label_write['sha256']}`",
            f"Sequence roles SHA-256: `{sequence_write['sha256']}`",
            "",
            "The frozen h1/h3/h7 cumulative returns and earliest-argmax",
            "1..7-day duration target were materialized without imputation.",
            "The panel was predicate-filtered through 2024-12-31; the 2025",
            "adaptive-evaluation role was created from dates only, with no",
            "evaluation label values loaded.",
            "",
            "BTC/ETH/SOL, scalers, models, optimizers, checkpoints, predictions,",
            "performance metrics, PnL, and outcome packets remained unopened.",
            "A pass authorizes only the separately governed V77 training loop.",
            "",
        ]
    )

    write_yaml_atomic(output / "resolved_config.yaml", config)
    write_json_atomic(output / "input_hash_receipt.json", input_receipt)
    write_json_atomic(output / "source_receipt.json", context["source_receipt"])
    write_json_atomic(output / "dataset_spec.json", dataset_spec)
    write_json_atomic(output / "label_schema.json", label_schema)
    write_json_atomic(output / "dataset_manifest.json", dataset_manifest)
    write_json_atomic(output / "data_access.json", data_access)
    write_json_atomic(output / "result.json", result)
    write_json_atomic(output / "audit.json", audit)
    write_json_atomic(output / "replay_receipt.json", replay_receipt)
    (output / "report.md").write_text(report, encoding="utf-8")
    manifest_names = tuple(
        name
        for name in contract["output_contract"]["packet_files"]
        if name != "artifact_manifest.json"
    )
    artifact_manifest: dict[str, Any] = {
        "schema_version": "v76-artifact-manifest/v1",
        "files": {name: file_sha256(output / name) for name in manifest_names},
        "data_files": {
            contract["output_contract"]["labels_path"]: label_write["sha256"],
            contract["output_contract"]["sequence_roles_path"]: sequence_write[
                "sha256"
            ],
        },
    }
    artifact_manifest["artifact_manifest_sha256"] = canonical_sha256(
        artifact_manifest
    )
    write_json_atomic(output / "artifact_manifest.json", artifact_manifest)
    actual_packet_files = sorted(path.name for path in output.iterdir() if path.is_file())
    if actual_packet_files != sorted(contract["output_contract"]["packet_files"]):
        raise RuntimeError(f"V76 artifact packet file-set drift: {actual_packet_files}")
    if not audit["passed"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"V76 dataset audit failed: {failed}")
    return result
