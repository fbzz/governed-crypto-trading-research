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
    "v56_result",
    "v56_audit",
    "v56_artifact_manifest",
    "v56_completion_receipt",
    "v32_result",
    "v32_audit",
    "v32_dataset_manifest",
    "v32_feature_schema",
    "v32_asset_folds",
    "v32_triplet_catalog",
}
BINARY_INPUTS = {"panel", "sequence_index"}


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


def _project_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"V57 path escapes project root: {relative}") from exc
    return path


@contextmanager
def _process_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another V57 dataset process holds the lock") from exc
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        path.unlink(missing_ok=True)


def build_multi_horizon_labels(
    panel: pd.DataFrame,
    label_contract: dict[str, Any],
) -> pd.DataFrame:
    required = {"date", "symbol", "raw_open", "raw_observation_available"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"V57 source panel is missing columns: {missing}")
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame["symbol"] = frame["symbol"].astype(str)
    frame = frame.sort_values(["date", "symbol"]).reset_index(drop=True)
    if frame.duplicated(["symbol", "date"]).any():
        raise ValueError("V57 source panel key is not unique")
    for _, group in frame.groupby("symbol", sort=False):
        dates = pd.DatetimeIndex(group["date"])
        if len(dates) > 1 and not bool(
            (dates[1:] - dates[:-1] == pd.Timedelta(days=1)).all()
        ):
            raise ValueError("V57 source panel is not a complete daily calendar")

    output = frame[["date", "symbol"]].copy()
    output["eligible_action_date"] = output["date"] + pd.Timedelta(days=1)
    grouped_open = frame.groupby("symbol", sort=False)["raw_open"]
    start_open = grouped_open.shift(-1).to_numpy(dtype=np.float64)
    for horizon in label_contract["order"]:
        definition = label_contract["definitions"][horizon]
        offset = int(definition["maturity_offset_days"])
        end_open = grouped_open.shift(-offset).to_numpy(dtype=np.float64)
        values = np.full(len(frame), np.nan, dtype=np.float64)
        valid = (
            np.isfinite(start_open)
            & np.isfinite(end_open)
            & (start_open > 0.0)
            & (end_open > 0.0)
        )
        values[valid] = np.log(end_open[valid] / start_open[valid])
        output[f"target_{horizon}_maturity_date"] = output["date"] + pd.Timedelta(
            days=offset
        )
        output[str(definition["column"])] = values
        output[f"{horizon}_label_complete"] = np.isfinite(values)
    completion_columns = [
        f"{horizon}_label_complete" for horizon in label_contract["order"]
    ]
    output["multi_horizon_label_complete"] = output[completion_columns].all(axis=1)
    ordered = ["date", "symbol", "eligible_action_date"]
    ordered.extend(
        f"target_{horizon}_maturity_date" for horizon in label_contract["order"]
    )
    ordered.extend(
        str(label_contract["definitions"][horizon]["column"])
        for horizon in label_contract["order"]
    )
    ordered.extend(completion_columns)
    ordered.append("multi_horizon_label_complete")
    return output[ordered].copy()


def _role_specs(role_contract: dict[str, Any]) -> list[tuple[str, dict[str, str]]]:
    specs: list[tuple[str, dict[str, str]]] = []
    for origin in role_contract["origins"]:
        origin_id = str(origin["id"])
        for geometry, geometry_spec in origin["geometries"].items():
            for role, window in (
                ("train", geometry_spec["train"]),
                ("validation", origin["validation"]),
                ("development_evaluation", origin["development_evaluation"]),
            ):
                specs.append((f"eligible_{origin_id}_{geometry}_{role}", window))
    return specs


def build_sequence_role_index(
    sequence_index: pd.DataFrame,
    labels: pd.DataFrame,
    role_contract: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    required = {"date", "sequence_start_date", "symbol"}
    missing = sorted(required - set(sequence_index.columns))
    if missing:
        raise ValueError(f"V57 sequence index is missing columns: {missing}")
    sequence = sequence_index[["date", "sequence_start_date", "symbol"]].copy()
    sequence["date"] = pd.to_datetime(sequence["date"], utc=True)
    sequence["sequence_start_date"] = pd.to_datetime(
        sequence["sequence_start_date"], utc=True
    )
    sequence["symbol"] = sequence["symbol"].astype(str)
    sequence = sequence.sort_values(["date", "symbol"]).reset_index(drop=True)
    if sequence.duplicated(["symbol", "date"]).any():
        raise ValueError("V57 source sequence key is not unique")
    label_columns = [
        "date",
        "symbol",
        "target_h7_maturity_date",
        "h1_label_complete",
        "h3_label_complete",
        "h7_label_complete",
        "multi_horizon_label_complete",
    ]
    merged = sequence.merge(
        labels[label_columns],
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    if merged["multi_horizon_label_complete"].isna().any():
        raise ValueError("V57 sequence keys are missing from the label table")

    role_audit: dict[str, dict[str, Any]] = {}
    for column, window in _role_specs(role_contract):
        start = pd.Timestamp(window["signal_start"], tz="UTC")
        end = pd.Timestamp(window["signal_end"], tz="UTC")
        maturity_end = pd.Timestamp(window["maturity_end"], tz="UTC")
        if end + pd.Timedelta(
            days=int(role_contract["maximum_maturity_purge_days"])
        ) != maturity_end:
            raise ValueError(f"V57 role boundary is not purged exactly once: {column}")
        flag = (
            merged["multi_horizon_label_complete"].astype(bool)
            & merged["date"].between(start, end, inclusive="both")
            & (merged["target_h7_maturity_date"] <= maturity_end)
        )
        merged[column] = flag
        eligible = merged.loc[flag]
        role_audit[column] = {
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
                eligible["target_h7_maturity_date"].max().date().isoformat()
                if len(eligible)
                else None
            ),
        }
    base = [
        "date",
        "sequence_start_date",
        "symbol",
        "h1_label_complete",
        "h3_label_complete",
        "h7_label_complete",
        "multi_horizon_label_complete",
    ]
    flags = list(role_contract["physical_flags"])
    generated_flags = [name for name, _ in _role_specs(role_contract)]
    if generated_flags != flags:
        raise ValueError("V57 physical role-flag order drifted")
    return merged[base + flags].copy(), role_audit


def _write_parquet_with_fresh_replay(
    frame: pd.DataFrame,
    final_path: Path,
    *,
    engine: str,
    compression: str,
    ledger: DatasetAccessLedger,
) -> dict[str, Any]:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    first = final_path.with_name(f".{final_path.name}.v57-replay-a.tmp")
    second = final_path.with_name(f".{final_path.name}.v57-replay-b.tmp")
    first.unlink(missing_ok=True)
    second.unlink(missing_ok=True)
    try:
        frame.to_parquet(first, index=False, engine=engine, compression=compression)
        frame.to_parquet(second, index=False, engine=engine, compression=compression)
        ledger.parquet_writes += 2
        first_hash = file_sha256(first)
        second_hash = file_sha256(second)
        if first_hash != second_hash:
            raise RuntimeError(f"V57 fresh Parquet replay drift: {final_path.name}")
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
    dataset = config["state_conditioned_multi_horizon_dataset"]
    root = Path(dataset["project_root"]).resolve()
    contract_reference = dataset["phase_contract"]
    contract_path = _project_path(root, contract_reference["path"])
    if (
        not contract_path.is_file()
        or file_sha256(contract_path) != contract_reference["file_sha256"]
    ):
        raise RuntimeError("V57 phase contract is missing or hash-drifted")
    contract = _load_yaml(contract_path, ledger)
    if (
        contract.get("phase") != "v57"
        or contract.get("authorized_next_action")
        != "authorize_v57_non_target_multi_horizon_dataset_build_only"
        or contract.get("output_contract", {}).get("labels_path") is None
        or config.get("output_dir") != contract["access_contract"]["output_dir"]
    ):
        raise RuntimeError("V57 frozen phase contract is inconsistent")

    input_paths = {
        name: _project_path(root, relative) for name, relative in dataset["inputs"].items()
    }
    if set(input_paths) != JSON_INPUTS | BINARY_INPUTS:
        raise RuntimeError("V57 input-name allowlist drift")
    if set(dataset["inputs"].values()) != set(
        contract["access_contract"]["allowed_inputs"]
    ):
        raise RuntimeError("V57 input-path allowlist drift")
    expected_hashes = contract["input_contract"]["expected_sha256"]
    if set(expected_hashes) != set(input_paths):
        raise RuntimeError("V57 expected-input hash map drift")
    observed_hashes: dict[str, str] = {}
    for name, path in input_paths.items():
        if not path.is_file():
            raise RuntimeError(f"V57 input is missing: {name}")
        observed_hashes[name] = file_sha256(path)
        if observed_hashes[name] != expected_hashes[name]:
            raise RuntimeError(f"V57 input hash drift: {name}")

    values = {name: _load_json(input_paths[name], ledger) for name in JSON_INPUTS}
    v32_manifest = values["v32_dataset_manifest"]
    v32_schema = values["v32_feature_schema"]
    folds = values["v32_asset_folds"]
    catalog = values["v32_triplet_catalog"]
    v56_result = values["v56_result"]
    if (
        values["v56_audit"].get("passed") is not True
        or v56_result.get("decision")
        != "authorize_v57_non_target_multi_horizon_dataset_build_only"
        or values["v56_completion_receipt"].get("decision")
        != "authorize_v57_non_target_multi_horizon_dataset_build_only"
        or v56_result.get("harness_spec", {}).get("v55_blueprint_sha256")
        != contract["v55_blueprint"]["canonical_sha256"]
        or values["v32_audit"].get("passed") is not True
        or v32_manifest.get("panel_sha256") != observed_hashes["panel"]
        or v32_manifest.get("sequence_index_sha256")
        != observed_hashes["sequence_index"]
        or values["v32_result"].get("dataset_manifest") != v32_manifest
        or list(v32_schema.get("model_feature_order", []))
        != list(contract["feature_contract"]["model_feature_order"])
        or len(folds.get("folds", [])) != 3
        or len(catalog.get("folds", [])) != 3
    ):
        raise RuntimeError("V57 parent metadata contract drift")
    for fold, catalog_fold in zip(folds["folds"], catalog["folds"], strict=True):
        if (
            int(fold["fold"]) != int(catalog_fold["fold"])
            or sorted(fold["train_symbols"]) != sorted(catalog_fold["train_symbols"])
            or sorted(fold["test_symbols"]) != sorted(catalog_fold["test_symbols"])
        ):
            raise RuntimeError("V57 frozen fold/catalog roles drifted")

    source_files = list(dataset["source_receipt_files"])
    if not source_files or len(source_files) != len(set(source_files)):
        raise RuntimeError("V57 source receipt is empty or duplicated")
    source_hashes: dict[str, str] = {}
    for relative in source_files:
        path = _project_path(root, relative)
        if not path.is_file():
            raise RuntimeError(f"V57 source receipt file is missing: {relative}")
        source_hashes[relative] = file_sha256(path)
    source_receipt = {
        "files": source_hashes,
        "bundle_sha256": canonical_sha256(source_hashes),
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
        },
    }
    return {
        "root": root,
        "dataset": dataset,
        "contract": contract,
        "contract_path": contract_path,
        "input_paths": input_paths,
        "input_hashes": observed_hashes,
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
    feature_columns = list(contract["feature_contract"]["panel_columns"])
    panel_columns = [
        "date",
        "symbol",
        "raw_observation_available",
        "raw_open",
        *feature_columns,
        "target_next_open_to_next_open_log_return",
    ]
    sequence_columns = ["date", "sequence_start_date", "symbol"]
    panel = reader(
        context["input_paths"]["panel"],
        engine="pyarrow",
        columns=panel_columns,
    )
    ledger.authorized_parquet_deserializations += 1
    ledger.authorized_panel_rows += len(panel)
    sequence = reader(
        context["input_paths"]["sequence_index"],
        engine="pyarrow",
        columns=sequence_columns,
    )
    ledger.authorized_parquet_deserializations += 1
    ledger.authorized_sequence_rows += len(sequence)
    loaded_symbols = sorted(set(panel["symbol"].astype(str)))
    target_loads = TARGET_SYMBOLS.intersection(loaded_symbols)
    ledger.target_asset_loads += len(target_loads)
    if target_loads:
        raise RuntimeError(f"V57 loaded target symbols: {sorted(target_loads)}")
    if set(sequence["symbol"].astype(str)) != set(loaded_symbols):
        raise RuntimeError("V57 panel and sequence universes differ")
    return panel, sequence, loaded_symbols


def _source_h1_matches(panel: pd.DataFrame, labels: pd.DataFrame) -> bool:
    source = panel.sort_values(["date", "symbol"])[
        "target_next_open_to_next_open_log_return"
    ].to_numpy(dtype=np.float64)
    rebuilt = labels["target_h1_open_to_open_log_return"].to_numpy(dtype=np.float64)
    same_missing = np.array_equal(np.isnan(source), np.isnan(rebuilt))
    finite = np.isfinite(source) & np.isfinite(rebuilt)
    return bool(same_missing and np.allclose(source[finite], rebuilt[finite], atol=1e-15, rtol=1e-13))


def run_state_conditioned_multi_horizon_dataset(
    config: dict[str, Any],
) -> dict[str, Any]:
    ledger = DatasetAccessLedger()
    context = _metadata_context(config, ledger)
    root = context["root"]
    contract = context["contract"]
    output = _project_path(root, config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)

    with _process_lock(root / "data" / "processed" / ".v57-dataset.lock"):
        panel, sequence, loaded_symbols = _read_inputs(context, ledger)
        labels = build_multi_horizon_labels(panel, contract["label_contract"])
        sequence_roles, role_audit = build_sequence_role_index(
            sequence, labels, contract["role_contract"]
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
    v32_manifest = context["values"]["v32_dataset_manifest"]
    folds = context["values"]["v32_asset_folds"]["folds"]
    catalog = context["values"]["v32_triplet_catalog"]["folds"]
    missing_keys = set(
        zip(
            pd.to_datetime(panel.loc[~panel["raw_observation_available"], "date"], utc=True),
            panel.loc[~panel["raw_observation_available"], "symbol"].astype(str),
            strict=True,
        )
    )
    label_keys = set(zip(labels["date"], labels["symbol"], strict=True))
    sequence_keys = set(zip(sequence["date"], sequence["symbol"], strict=True))
    role_keys = set(
        zip(sequence_roles["date"], sequence_roles["symbol"], strict=True)
    )
    role_boundaries_pass = all(
        row["eligible_rows"] > 0
        and row["maximum_target_maturity"] == row["maturity_end"]
        for row in role_audit.values()
    )
    exact_fold_roles = all(
        len(fold["train_symbols"]) == 20
        and len(fold["test_symbols"]) == 10
        and math.comb(len(catalog_fold["train_symbols"]), 3) == 1140
        and math.comb(len(catalog_fold["test_symbols"]), 3) == 120
        for fold, catalog_fold in zip(folds, catalog, strict=True)
    )
    checks = {
        "all_input_hashes_match": input_hashes_after
        == input_contract["expected_sha256"],
        "exact_thirty_asset_universe_and_three_folds": len(loaded_symbols) == 30
        and loaded_symbols == sorted(v32_manifest["symbols"])
        and exact_fold_roles,
        "exact_nine_feature_order": list(
            context["values"]["v32_feature_schema"]["model_feature_order"]
        )
        == list(feature_contract["model_feature_order"])
        and list(v32_manifest["panel_features"])
        == list(feature_contract["panel_columns"])
        and "within_triplet_relative_strength" not in labels.columns,
        "exact_h1_h3_h7_label_definitions": list(
            contract["label_contract"]["order"]
        )
        == ["h1", "h3", "h7"]
        and _source_h1_matches(panel, labels)
        and all(
            labels[f"{horizon}_label_complete"].equals(
                labels[definition["column"]].notna()
            )
            for horizon, definition in contract["label_contract"][
                "definitions"
            ].items()
        ),
        "no_label_crosses_registered_role_boundary": role_boundaries_pass,
        "maximum_maturity_purge_is_eight_days": max(
            int(item["maturity_offset_days"])
            for item in contract["label_contract"]["definitions"].values()
        )
        == int(contract["role_contract"]["maximum_maturity_purge_days"])
        == 8
        and bool(
            (
                labels["target_h7_maturity_date"] - labels["date"]
                == pd.Timedelta(days=8)
            ).all()
        ),
        "panel_and_sequence_keys_are_unique": len(label_keys) == len(labels)
        == int(input_contract["panel_rows"])
        and len(sequence_keys) == len(sequence)
        == int(input_contract["sequence_rows"])
        and sequence_keys == role_keys,
        "missing_rows_are_preserved": len(missing_keys)
        == int(input_contract["preserved_missing_panel_rows"])
        and missing_keys.issubset(label_keys)
        and len(labels) == len(panel)
        and len(sequence_roles) == len(sequence)
        and ledger.missing_value_imputations == 0,
        "target_assets_are_absent": not TARGET_SYMBOLS.intersection(loaded_symbols)
        and ledger.target_asset_loads == 0,
        "no_scaler_model_optimizer_prediction_performance_or_pnl": ledger.forbidden_operations_are_zero()
        and ledger.authorized_parquet_deserializations == 2,
        "byte_identical_replay": label_write["byte_identical"]
        and sequence_write["byte_identical"]
        and label_write["sha256"] == label_write["fresh_replay_sha256"]
        and sequence_write["sha256"] == sequence_write["fresh_replay_sha256"],
        "phase_contract_hash_matches": file_sha256(context["contract_path"])
        == context["dataset"]["phase_contract"]["file_sha256"],
        "output_schema_is_exact": list(labels.columns)
        == list(output_contract["labels_columns"])
        and list(sequence_roles.columns)
        == list(output_contract["sequence_base_columns"])
        + list(contract["role_contract"]["physical_flags"]),
        "source_receipt_is_complete": bool(context["source_receipt"]["files"])
        and len(context["source_receipt"]["bundle_sha256"]) == 64,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    audit = {"passed": all(checks.values()), "checks": checks}
    decision = contract["pass_action"] if audit["passed"] else contract["failure_action"]

    dataset_spec: dict[str, Any] = {
        "version": "v57",
        "candidate_family_id": contract["family_id"],
        "phase_contract_file_sha256": context["dataset"]["phase_contract"][
            "file_sha256"
        ],
        "feature_contract": feature_contract,
        "label_contract": contract["label_contract"],
        "role_contract": contract["role_contract"],
        "output_contract": output_contract,
        "pass_action": contract["pass_action"],
        "failure_action": contract["failure_action"],
    }
    dataset_spec["dataset_spec_sha256"] = canonical_sha256(dataset_spec)
    label_schema: dict[str, Any] = {
        "version": "v57",
        "columns": list(labels.columns),
        "dtypes": {name: str(dtype) for name, dtype in labels.dtypes.items()},
        "definitions": contract["label_contract"]["definitions"],
        "missing_policy": contract["label_contract"]["missing_policy"],
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
        "panel_projection": [
            "date",
            "symbol",
            "raw_observation_available",
            "raw_open",
            *feature_contract["panel_columns"],
            "target_next_open_to_next_open_log_return",
        ],
        "sequence_projection": ["date", "sequence_start_date", "symbol"],
        "loaded_symbols": loaded_symbols,
        "operation_ledger": ledger.to_dict(),
    }
    dataset_manifest = {
        "version": "v57",
        "labels": {
            **label_write,
            "path": output_contract["labels_path"],
            "complete_h1_rows": int(labels["h1_label_complete"].sum()),
            "complete_h3_rows": int(labels["h3_label_complete"].sum()),
            "complete_h7_rows": int(labels["h7_label_complete"].sum()),
            "complete_all_horizons_rows": int(
                labels["multi_horizon_label_complete"].sum()
            ),
        },
        "sequence_roles": {
            **sequence_write,
            "path": output_contract["sequence_roles_path"],
        },
        "role_audit": role_audit,
        "symbols": loaded_symbols,
        "source_panel_sha256": context["input_hashes"]["panel"],
        "source_sequence_index_sha256": context["input_hashes"]["sequence_index"],
        "feature_schema_sha256": file_sha256(
            context["input_paths"]["v32_feature_schema"]
        ),
        "asset_folds_sha256": file_sha256(context["input_paths"]["v32_asset_folds"]),
        "triplet_catalog_sha256": file_sha256(
            context["input_paths"]["v32_triplet_catalog"]
        ),
        "label_schema_sha256": label_schema["label_schema_sha256"],
    }
    report = "\n".join(
        [
            "# V57 Non-target Multi-horizon Dataset",
            "",
            f"Decision: **{decision}**",
            "",
            f"Label rows: **{len(labels):,}**",
            f"Sequence-role rows: **{len(sequence_roles):,}**",
            f"All-horizon complete label rows: **{int(labels['multi_horizon_label_complete'].sum()):,}**",
            f"Labels SHA-256: `{label_write['sha256']}`",
            f"Sequence roles SHA-256: `{sequence_write['sha256']}`",
            "",
            "The frozen h1/h3/h7 endpoint returns and every origin/geometry role",
            "were materialized without imputation or selection. Two fresh writes of",
            "each Parquet were byte-identical.",
            "",
            "BTC/ETH/SOL, scalers, models, optimizers, checkpoints, predictions,",
            "performance metrics, and PnL remained unopened. A pass authorizes only",
            "the separately governed V58 non-target training loop.",
            "",
        ]
    )

    write_json_atomic(output / "dataset_spec.json", dataset_spec)
    write_json_atomic(output / "input_hash_receipt.json", input_receipt)
    write_json_atomic(output / "source_receipt.json", context["source_receipt"])
    write_json_atomic(output / "data_access.json", data_access)
    write_json_atomic(output / "dataset_manifest.json", dataset_manifest)
    write_json_atomic(output / "label_schema.json", label_schema)
    write_json_atomic(output / "audit.json", audit)
    write_yaml_atomic(output / "resolved_config.yaml", config)
    (output / "report.md").write_text(report, encoding="utf-8")
    result: dict[str, Any] = {
        "version": "v57",
        "candidate_family_id": contract["family_id"],
        "decision": decision,
        "dataset_spec": dataset_spec,
        "dataset_manifest": dataset_manifest,
        "data_access": data_access,
        "source_receipt": context["source_receipt"],
        "audit": audit,
    }
    result["result_sha256"] = canonical_sha256(result)
    write_json_atomic(output / "result.json", result)

    manifest_names = tuple(
        name
        for name in output_contract["packet_files"]
        if name not in {"artifact_manifest.json", "completion_receipt.json"}
    )
    artifact_manifest: dict[str, Any] = {
        "version": "v57",
        "files": {name: file_sha256(output / name) for name in manifest_names},
        "data_files": {
            output_contract["labels_path"]: label_write["sha256"],
            output_contract["sequence_roles_path"]: sequence_write["sha256"],
        },
    }
    artifact_manifest["manifest_sha256"] = canonical_sha256(artifact_manifest)
    write_json_atomic(output / "artifact_manifest.json", artifact_manifest)
    completion = {
        "version": "v57",
        "decision": decision,
        "dataset_spec_sha256": dataset_spec["dataset_spec_sha256"],
        "result_file_sha256": file_sha256(output / "result.json"),
        "audit_file_sha256": file_sha256(output / "audit.json"),
        "artifact_manifest_file_sha256": file_sha256(
            output / "artifact_manifest.json"
        ),
        "labels_sha256": label_write["sha256"],
        "sequence_roles_sha256": sequence_write["sha256"],
    }
    write_json_atomic(output / "completion_receipt.json", completion)
    actual_packet_files = sorted(path.name for path in output.iterdir() if path.is_file())
    if actual_packet_files != sorted(output_contract["packet_files"]):
        raise RuntimeError(
            f"V57 artifact packet file-set drift: {actual_packet_files}"
        )
    if not audit["passed"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"V57 dataset audit failed: {failed}")
    return result
