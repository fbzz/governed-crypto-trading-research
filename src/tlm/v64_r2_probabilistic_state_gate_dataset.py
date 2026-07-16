from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.dataset as pads
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
    "v66_result",
    "v66_audit",
    "v66_harness_spec",
    "v66_artifact_manifest",
    "v66_completion_receipt",
    "v65_specification",
    "v65_blueprint",
    "v62_result",
    "v62_audit",
    "v62_dataset_spec",
    "v62_dataset_manifest",
    "v62_label_schema",
    "v62_source_receipt",
    "v62_completion_receipt",
    "v62_artifact_manifest",
    "v62_data_access",
    "v62_triplet_derivation_smoke",
}
PARQUET_INPUTS = {"labels", "sequence_roles"}
LABEL_PROJECTION = [
    "date",
    "symbol",
    "target_h1_maturity_date",
    "target_h1_open_to_open_log_return",
    "h1_label_complete",
]
SEQUENCE_PROJECTION = [
    "date",
    "sequence_start_date",
    "symbol",
    "h1_label_complete",
    "eligible_train",
]
ROLE_COLUMNS = [
    "date",
    "sequence_start_date",
    "symbol",
    "h1_label_complete",
    "eligible_v62_train",
    "gate_role",
    "eligible_gate_train",
    "eligible_gate_internal_validation",
]
FORBIDDEN_COLUMNS = {
    "eligible_consumed_development_validation",
    "target_asset",
    "prediction",
    "portfolio",
    "pnl",
}


def _load_json(path: Path, ledger: DatasetAccessLedger) -> Any:
    ledger.authorized_metadata_reads += 1
    return json.loads(path.read_text(encoding="utf-8"))


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
        raise ValueError(f"V67 path escapes project root: {relative}") from exc
    return path


@contextmanager
def _process_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another V67 dataset process holds the lock") from exc
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        path.unlink(missing_ok=True)


def _embedded_hash_matches(value: dict[str, Any], key: str) -> bool:
    copy = dict(value)
    embedded = copy.pop(key, None)
    return isinstance(embedded, str) and embedded == canonical_sha256(copy)


def _write_parquet_with_fresh_replay(
    frame: pd.DataFrame,
    final_path: Path,
    *,
    engine: str,
    compression: str,
    ledger: DatasetAccessLedger,
) -> dict[str, Any]:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    first = final_path.with_name(f".{final_path.name}.v67-replay-a.tmp")
    second = final_path.with_name(f".{final_path.name}.v67-replay-b.tmp")
    first.unlink(missing_ok=True)
    second.unlink(missing_ok=True)
    try:
        frame.to_parquet(first, index=False, engine=engine, compression=compression)
        frame.to_parquet(second, index=False, engine=engine, compression=compression)
        ledger.parquet_writes += 2
        first_hash = file_sha256(first)
        second_hash = file_sha256(second)
        if first_hash != second_hash:
            raise RuntimeError(f"V67 fresh Parquet replay drift: {final_path.name}")
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


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"V67 expected mapping metadata: {name}")
    return value


def _metadata_context(
    config: dict[str, Any], ledger: DatasetAccessLedger
) -> dict[str, Any]:
    dataset = config["v64_r2_probabilistic_state_gate_dataset"]
    root = Path(dataset["project_root"]).resolve()
    contract_reference = dataset["phase_contract"]
    contract_path = _project_path(root, contract_reference["path"])
    if (
        not contract_path.is_file()
        or file_sha256(contract_path) != contract_reference["file_sha256"]
    ):
        raise RuntimeError("V67 phase contract is missing or hash-drifted")
    contract = _load_yaml(contract_path, ledger)
    if (
        contract.get("phase") != "v67"
        or contract.get("stage_revision")
        != "v067_non_target_v64_r2_probabilistic_state_gate_dataset_r1"
        or contract.get("authorized_next_action")
        != "authorize_v67_non_target_v64_r2_probabilistic_state_gate_dataset_only"
        or config.get("output_dir") != contract["access_contract"]["output_dir"]
    ):
        raise RuntimeError("V67 frozen phase contract is inconsistent")

    input_paths = {
        name: _project_path(root, relative)
        for name, relative in dataset["inputs"].items()
    }
    if set(input_paths) != JSON_INPUTS | PARQUET_INPUTS:
        raise RuntimeError("V67 input-name allowlist drift")
    allowed_inputs = set(contract["access_contract"]["allowed_inputs"])
    if set(dataset["inputs"].values()) != allowed_inputs:
        raise RuntimeError("V67 input-path allowlist drift")
    expected_by_path = contract["input_contract"]["expected_file_sha256_by_path"]
    if set(expected_by_path) != allowed_inputs:
        raise RuntimeError("V67 expected-input hash map drift")

    observed_hashes: dict[str, str] = {}
    for name, path in input_paths.items():
        if not path.is_file():
            raise RuntimeError(f"V67 input is missing: {name}")
        observed_hashes[name] = file_sha256(path)
        relative = dataset["inputs"][name]
        if observed_hashes[name] != expected_by_path[relative]:
            raise RuntimeError(f"V67 input hash drift: {name}")

    values = {name: _load_json(input_paths[name], ledger) for name in JSON_INPUTS}
    v66_result = _mapping(values["v66_result"], "v66_result")
    v66_audit = _mapping(values["v66_audit"], "v66_audit")
    v66_harness = _mapping(values["v66_harness_spec"], "v66_harness_spec")
    v66_manifest = _mapping(
        values["v66_artifact_manifest"], "v66_artifact_manifest"
    )
    v66_completion = _mapping(
        values["v66_completion_receipt"], "v66_completion_receipt"
    )
    if (
        v66_audit.get("passed") is not True
        or v66_result.get("decision")
        != "authorize_v67_non_target_v64_r2_probabilistic_state_gate_dataset_only"
        or not _embedded_hash_matches(v66_result, "result_sha256")
        or not _embedded_hash_matches(v66_harness, "harness_spec_sha256")
        or not _embedded_hash_matches(v66_manifest, "artifact_manifest_sha256")
        or not _embedded_hash_matches(v66_completion, "completion_receipt_sha256")
        or v66_result.get("harness_spec_sha256")
        != v66_harness.get("harness_spec_sha256")
        or v66_result.get("smoke", {}).get("ranker_requires_grad") is not False
        or v66_result.get("smoke", {}).get("ranker_optimizer_present") is not False
    ):
        raise RuntimeError("V67 V66 authorization metadata drift")

    v65_spec = _mapping(values["v65_specification"], "v65_specification")
    v65_blueprint = _mapping(values["v65_blueprint"], "v65_blueprint")
    if (
        not _embedded_hash_matches(v65_spec, "specification_sha256")
        or v65_blueprint.get("specification_sha256")
        != v65_spec.get("specification_sha256")
        or v65_spec.get("candidate_family_id") != contract["family_id"]
        or v65_spec.get("lineage_label") != "V64-R2"
        or v65_spec.get("probabilistic_gate", {}).get("distribution")
        != "student_t_location_scale"
        or v65_spec.get("probabilistic_gate", {}).get("degrees_of_freedom") != 5.0
        or v65_spec.get("policy", {}).get("abstention_probability_threshold")
        != 0.60
        or v65_spec.get("state_gate_architecture", {}).get("input_features") != 18
        or v65_spec.get("ranker_contract", {}).get("status")
        != "frozen_exactly_from_v64"
    ):
        raise RuntimeError("V67 V65 frozen probabilistic-gate contract drift")

    v62_result = _mapping(values["v62_result"], "v62_result")
    v62_audit = _mapping(values["v62_audit"], "v62_audit")
    v62_spec = _mapping(values["v62_dataset_spec"], "v62_dataset_spec")
    v62_manifest = _mapping(values["v62_dataset_manifest"], "v62_dataset_manifest")
    v62_label_schema = _mapping(values["v62_label_schema"], "v62_label_schema")
    v62_completion = _mapping(
        values["v62_completion_receipt"], "v62_completion_receipt"
    )
    v62_artifact_manifest = _mapping(
        values["v62_artifact_manifest"], "v62_artifact_manifest"
    )
    v62_data_access = _mapping(values["v62_data_access"], "v62_data_access")
    v62_smoke = values["v62_triplet_derivation_smoke"]
    if not isinstance(v62_smoke, list):
        raise RuntimeError("V67 V62 triplet smoke is not a list")
    if (
        v62_audit.get("passed") is not True
        or v62_result.get("decision")
        != "authorize_v63_frozen_non_target_decoupled_rank_state_training_only"
        or not _embedded_hash_matches(v62_result, "result_sha256")
        or not _embedded_hash_matches(v62_spec, "dataset_spec_sha256")
        or not _embedded_hash_matches(v62_label_schema, "label_schema_sha256")
        or not _embedded_hash_matches(
            v62_artifact_manifest, "artifact_manifest_sha256"
        )
        or not _embedded_hash_matches(v62_completion, "completion_receipt_sha256")
        or v62_result.get("dataset_spec") != v62_spec
        or v62_result.get("dataset_manifest") != v62_manifest
        or v62_result.get("data_access") != v62_data_access
        or v62_manifest.get("labels", {}).get("sha256")
        != observed_hashes["labels"]
        or v62_manifest.get("sequence_roles", {}).get("sha256")
        != observed_hashes["sequence_roles"]
        or v62_spec.get("data_contract", {})
        .get("action_return", {})
        .get("formula")
        != "log(open[t+2] / open[t+1])"
        or v62_spec.get("data_contract", {})
        .get("derived_state_features", {})
        .get("count")
        != 18
        or len(v62_smoke) != 6
        or not all(
            row.get("input_shape") == [256, 3, 9]
            and row.get("state_shape") == [256, 18]
            and row.get("state_permutation_invariant") is True
            for row in v62_smoke
        )
    ):
        raise RuntimeError("V67 V62 dataset metadata contract drift")

    symbols = sorted(str(symbol) for symbol in v62_manifest.get("symbols", []))
    if len(symbols) != 30 or TARGET_SYMBOLS.intersection(symbols):
        raise RuntimeError("V67 V62 non-target universe drift")

    source_files = list(dataset["source_receipt_files"])
    if not source_files or len(source_files) != len(set(source_files)):
        raise RuntimeError("V67 source receipt is empty or duplicated")
    source_hashes: dict[str, str] = {}
    for relative in source_files:
        path = _project_path(root, relative)
        if not path.is_file():
            raise RuntimeError(f"V67 source receipt file is missing: {relative}")
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
        "symbols": symbols,
        "source_receipt": source_receipt,
    }


TableScanner = Callable[[Path, list[str], pads.Expression], pd.DataFrame]


def _scan_table(
    path: Path, columns: list[str], predicate: pads.Expression
) -> pd.DataFrame:
    table = pads.dataset(path, format="parquet").to_table(
        columns=columns,
        filter=predicate,
    )
    return table.to_pandas()


def _read_authorized_parquets(
    context: dict[str, Any],
    ledger: DatasetAccessLedger,
    *,
    scanner: TableScanner = _scan_table,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    contract = context["contract"]
    parquet_contract = contract["parquet_access_contract"]
    if (
        parquet_contract["labels_projection"] != LABEL_PROJECTION
        or parquet_contract["sequence_projection"] != SEQUENCE_PROJECTION
        or set(parquet_contract["forbidden_columns"]) != FORBIDDEN_COLUMNS
        or parquet_contract["maximum_parquet_deserializations"] != 2
        or parquet_contract["predicate_pushdown_required"] is not True
        or parquet_contract["full_table_materialization_then_filtering_allowed"]
        is not False
    ):
        raise RuntimeError("V67 Parquet access contract drift")

    start = datetime(2021, 3, 1, tzinfo=timezone.utc)
    end = datetime(2024, 12, 23, 23, 59, 59, 999999, tzinfo=timezone.utc)
    labels_predicate = (pads.field("date") >= start) & (pads.field("date") <= end)
    sequence_predicate = (pads.field("eligible_train") == True) & (
        pads.field("date") <= end
    )
    labels = scanner(
        context["input_paths"]["labels"], LABEL_PROJECTION, labels_predicate
    )
    ledger.authorized_parquet_deserializations += 1
    ledger.authorized_panel_rows += len(labels)
    sequence = scanner(
        context["input_paths"]["sequence_roles"],
        SEQUENCE_PROJECTION,
        sequence_predicate,
    )
    ledger.authorized_parquet_deserializations += 1
    ledger.authorized_sequence_rows += len(sequence)

    if list(labels.columns) != LABEL_PROJECTION:
        raise RuntimeError("V67 label projection drift after predicate scan")
    if list(sequence.columns) != SEQUENCE_PROJECTION:
        raise RuntimeError("V67 sequence projection drift after predicate scan")
    for frame, date_columns in (
        (labels, ("date", "target_h1_maturity_date")),
        (sequence, ("date", "sequence_start_date")),
    ):
        for column in date_columns:
            frame[column] = pd.to_datetime(frame[column], utc=True)
        frame["symbol"] = frame["symbol"].astype(str)
    labels = labels.sort_values(["date", "symbol"]).reset_index(drop=True)
    sequence = sequence.sort_values(["date", "symbol"]).reset_index(drop=True)

    if labels.empty or sequence.empty:
        raise RuntimeError("V67 predicate scan returned an empty table")
    if labels["date"].min() < pd.Timestamp("2021-03-01", tz="UTC"):
        raise RuntimeError("V67 label scan loaded a pre-contract row")
    maximum = pd.Timestamp("2024-12-23", tz="UTC")
    if labels["date"].max() > maximum or sequence["date"].max() > maximum:
        raise RuntimeError("V67 predicate scan loaded a 2025-or-later signal")
    if not sequence["eligible_train"].astype(bool).all():
        raise RuntimeError("V67 sequence scan loaded a non-training role")
    loaded_symbols = set(labels["symbol"]) | set(sequence["symbol"])
    target_loads = TARGET_SYMBOLS.intersection(loaded_symbols)
    ledger.target_asset_loads += len(target_loads)
    if target_loads:
        raise RuntimeError(f"V67 loaded target symbols: {sorted(target_loads)}")
    if not loaded_symbols.issubset(set(context["symbols"])):
        raise RuntimeError("V67 predicate scan loaded an unregistered symbol")

    access = {
        "scanner": "pyarrow.dataset.Dataset.to_table",
        "predicate_pushdown_applied_before_materialization": True,
        "full_table_materialization_then_filtering": False,
        "labels_projection": LABEL_PROJECTION,
        "labels_predicate": str(labels_predicate),
        "sequence_projection": SEQUENCE_PROJECTION,
        "sequence_predicate": str(sequence_predicate),
        "forbidden_columns_loaded": sorted(
            (set(labels.columns) | set(sequence.columns)) & FORBIDDEN_COLUMNS
        ),
        "labels_rows_loaded": len(labels),
        "sequence_rows_loaded": len(sequence),
        "maximum_loaded_signal_date": max(
            labels["date"].max(), sequence["date"].max()
        ).date().isoformat(),
        "maximum_loaded_label_maturity": labels[
            "target_h1_maturity_date"
        ].max().date().isoformat(),
        "loaded_2025_or_later_values": 0,
    }
    return labels, sequence, access


def build_gate_roles(
    labels: pd.DataFrame,
    sequence: pd.DataFrame,
    role_contract: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if labels.duplicated(["date", "symbol"]).any():
        raise ValueError("V67 label keys are not unique")
    if sequence.duplicated(["date", "symbol"]).any():
        raise ValueError("V67 sequence keys are not unique")
    source = sequence.copy()
    source["eligible_v62_train"] = source.pop("eligible_train").astype(bool)
    label_keys = labels[
        [
            "date",
            "symbol",
            "target_h1_maturity_date",
            "target_h1_open_to_open_log_return",
            "h1_label_complete",
        ]
    ].rename(columns={"h1_label_complete": "label_h1_complete"})
    merged = source.merge(
        label_keys,
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    if merged["target_h1_maturity_date"].isna().any():
        raise ValueError("V67 eligible sequence keys are missing from filtered labels")
    complete = (
        merged["eligible_v62_train"]
        & merged["h1_label_complete"].astype(bool)
        & merged["label_h1_complete"].astype(bool)
        & np.isfinite(
            merged["target_h1_open_to_open_log_return"].to_numpy(dtype=np.float64)
        )
    )
    train = role_contract["gate_train"]
    validation = role_contract["gate_internal_validation"]
    train_start = pd.Timestamp(train["signal_start"], tz="UTC")
    train_end = pd.Timestamp(train["signal_end"], tz="UTC")
    validation_start = pd.Timestamp(validation["signal_start"], tz="UTC")
    validation_end = pd.Timestamp(validation["signal_end"], tz="UTC")
    train_maturity_end = validation_start - pd.Timedelta(days=1)
    validation_maturity_end = validation_end + pd.Timedelta(days=2)
    eligible_train = (
        complete
        & merged["date"].between(train_start, train_end, inclusive="both")
        & (merged["target_h1_maturity_date"] <= train_maturity_end)
    )
    eligible_validation = (
        complete
        & merged["date"].between(
            validation_start, validation_end, inclusive="both"
        )
        & (merged["target_h1_maturity_date"] <= validation_maturity_end)
    )
    if (eligible_train & eligible_validation).any():
        raise ValueError("V67 gate train and internal validation overlap")
    admitted = eligible_train | eligible_validation
    merged["eligible_gate_train"] = eligible_train
    merged["eligible_gate_internal_validation"] = eligible_validation
    merged["gate_role"] = np.where(
        eligible_train, "gate_train", "gate_internal_validation"
    )
    roles = merged.loc[admitted, ROLE_COLUMNS].copy()
    roles = roles.sort_values(["date", "symbol"]).reset_index(drop=True)

    audits: dict[str, dict[str, Any]] = {}
    for name, flag, window, maturity_end in (
        ("gate_train", eligible_train, train, train_maturity_end),
        (
            "gate_internal_validation",
            eligible_validation,
            validation,
            validation_maturity_end,
        ),
    ):
        eligible = merged.loc[flag]
        audits[name] = {
            "signal_start": window["signal_start"],
            "signal_end": window["signal_end"],
            "maturity_end": maturity_end.date().isoformat(),
            "eligible_rows": int(flag.sum()),
            "eligible_dates": int(eligible["date"].nunique()),
            "first_eligible_date": eligible["date"].min().date().isoformat(),
            "last_eligible_date": eligible["date"].max().date().isoformat(),
            "maximum_target_maturity": eligible[
                "target_h1_maturity_date"
            ].max().date().isoformat(),
        }
    audit = {
        "roles": audits,
        "eligible_v62_rows_scanned": len(sequence),
        "admitted_rows": len(roles),
        "excluded_split_boundary_rows": int((~admitted).sum()),
        "split_embargo_days": 2,
        "train_maturity_precedes_validation_signal_start": bool(
            train_maturity_end < validation_start
        ),
    }
    return roles, audit


def _utc_bound(frame: pd.DataFrame, column: str, kind: str) -> str:
    value = frame[column].min() if kind == "min" else frame[column].max()
    return pd.Timestamp(value).date().isoformat()


def run_v64_r2_probabilistic_state_gate_dataset(
    config: dict[str, Any],
) -> dict[str, Any]:
    ledger = DatasetAccessLedger()
    context = _metadata_context(config, ledger)
    root = context["root"]
    contract = context["contract"]
    dataset = context["dataset"]
    output = _project_path(root, config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)

    with _process_lock(root / "data" / "processed" / ".v67-dataset.lock"):
        labels, source_sequence, scan_access = _read_authorized_parquets(
            context, ledger
        )
        roles, role_audit = build_gate_roles(
            labels, source_sequence, contract["role_contract"]
        )
        parquet = dataset["parquet"]
        output_contract = contract["output_contract"]
        labels_write = _write_parquet_with_fresh_replay(
            labels,
            _project_path(root, output_contract["labels_path"]),
            engine=parquet["engine"],
            compression=parquet["compression"],
            ledger=ledger,
        )
        roles_write = _write_parquet_with_fresh_replay(
            roles,
            _project_path(root, output_contract["sequence_roles_path"]),
            engine=parquet["engine"],
            compression=parquet["compression"],
            ledger=ledger,
        )

    input_hashes_after = {
        name: file_sha256(path) for name, path in context["input_paths"].items()
    }
    v66_result = _mapping(context["values"]["v66_result"], "v66_result")
    v65_spec = _mapping(
        context["values"]["v65_specification"], "v65_specification"
    )
    v62_spec = _mapping(context["values"]["v62_dataset_spec"], "v62_dataset_spec")
    v62_manifest = _mapping(
        context["values"]["v62_dataset_manifest"], "v62_dataset_manifest"
    )
    role_rows = role_audit["roles"]
    label_missing = int((~labels["h1_label_complete"].astype(bool)).sum())
    maximum_written_year = max(
        labels["date"].max().year,
        labels["target_h1_maturity_date"].max().year,
        roles["date"].max().year,
        roles["sequence_start_date"].max().year,
    )
    role_checks = (
        len(roles) > 0
        and not (
            roles["eligible_gate_train"]
            & roles["eligible_gate_internal_validation"]
        ).any()
        and bool(
            (
                roles["eligible_gate_train"]
                | roles["eligible_gate_internal_validation"]
            ).all()
        )
        and role_rows["gate_train"]["maximum_target_maturity"]
        < role_rows["gate_internal_validation"]["first_eligible_date"]
    )
    derivation_contract: dict[str, Any] = {
        "schema_version": "v67-v64-r2-derivation-contract/v1",
        "action_return": "log(open[t+2] / open[t+1])",
        "probabilistic_gate_target": (
            "train_only_triplet_mean_next_open_log_return"
        ),
        "triplet_market_component": (
            "equal_weight_mean_of_three_asset_action_returns"
        ),
        "state_features": (
            "per_day_cross_asset_mean_and_population_std_of_nine_inputs"
        ),
        "state_feature_count": 18,
        "materialization": "derive_on_demand_inside_exact_lexical_triplet",
        "input_shape": [None, 256, 3, 9],
        "real_feature_tensors_materialized": 0,
        "universe_fold_or_triplet_reselections": 0,
        "source_v62_triplet_smoke_file_sha256": context["input_hashes"][
            "v62_triplet_derivation_smoke"
        ],
    }
    derivation_contract["derivation_contract_sha256"] = canonical_sha256(
        derivation_contract
    )
    checks = {
        "all_input_hashes_match": input_hashes_after == context["input_hashes"],
        "exact_v66_probabilistic_gate_and_abstention_contract_preserved": (
            v66_result.get("harness_spec_sha256")
            == "7f5e4f4787c38567d85870ce7722a302a42e19238a60ce4c90ce0fcc75740e25"
            and v65_spec.get("probabilistic_gate", {}).get("degrees_of_freedom")
            == 5.0
            and v65_spec.get("policy", {}).get("abstention_probability_threshold")
            == 0.60
        ),
        "exact_v62_non_target_dataset_receipts_preserved": (
            v62_manifest.get("labels", {}).get("sha256")
            == context["input_hashes"]["labels"]
            and v62_manifest.get("sequence_roles", {}).get("sha256")
            == context["input_hashes"]["sequence_roles"]
            and v62_spec.get("data_contract", {}).get("symbol_count") == 30
        ),
        "only_eligible_v62_train_rows_are_admitted": (
            bool(roles["eligible_v62_train"].all())
            and len(roles) + role_audit["excluded_split_boundary_rows"]
            == len(source_sequence)
        ),
        "gate_train_and_internal_validation_are_disjoint_and_chronological": role_checks,
        "no_2025_or_later_signal_label_or_role_value_is_loaded_or_written": (
            scan_access["loaded_2025_or_later_values"] == 0
            and scan_access["maximum_loaded_signal_date"] <= "2024-12-23"
            and maximum_written_year <= 2024
        ),
        "exact_h1_open_t_plus_1_to_open_t_plus_2_market_component_target_preserved": (
            bool(
                (
                    labels["target_h1_maturity_date"] - labels["date"]
                    == pd.Timedelta(days=2)
                ).all()
            )
            and labels["h1_label_complete"].equals(
                labels["target_h1_open_to_open_log_return"].notna()
            )
            and derivation_contract["triplet_market_component"]
            == "equal_weight_mean_of_three_asset_action_returns"
        ),
        "exact_eighteen_state_features_remain_derived_on_demand": (
            derivation_contract["state_feature_count"] == 18
            and derivation_contract["materialization"]
            == "derive_on_demand_inside_exact_lexical_triplet"
            and derivation_contract["real_feature_tensors_materialized"] == 0
        ),
        "missing_rows_are_preserved_without_imputation": (
            labels_write["rows"] == len(labels)
            and int((~labels["h1_label_complete"].astype(bool)).sum())
            == label_missing
            and ledger.missing_value_imputations == 0
        ),
        "target_assets_are_absent": (
            not TARGET_SYMBOLS.intersection(context["symbols"])
            and not TARGET_SYMBOLS.intersection(set(labels["symbol"]))
            and not TARGET_SYMBOLS.intersection(set(roles["symbol"]))
            and ledger.target_asset_loads == 0
        ),
        "no_scaler_model_optimizer_checkpoint_prediction_performance_or_pnl": (
            ledger.forbidden_operations_are_zero()
            and ledger.authorized_parquet_deserializations == 2
            and scan_access["forbidden_columns_loaded"] == []
        ),
        "byte_identical_replay": (
            labels_write["byte_identical"]
            and roles_write["byte_identical"]
            and labels_write["sha256"] == labels_write["fresh_replay_sha256"]
            and roles_write["sha256"] == roles_write["fresh_replay_sha256"]
        ),
        "output_schema_is_exact": (
            list(labels.columns) == LABEL_PROJECTION
            and list(roles.columns) == ROLE_COLUMNS
        ),
        "predicate_pushdown_is_explicit_and_bounded": (
            scan_access["predicate_pushdown_applied_before_materialization"]
            is True
            and scan_access["full_table_materialization_then_filtering"] is False
            and ledger.authorized_parquet_deserializations == 2
        ),
        "source_receipt_is_complete": (
            bool(context["source_receipt"]["files"])
            and len(context["source_receipt"]["bundle_sha256"]) == 64
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    audit = {
        "schema_version": "v67-v64-r2-probabilistic-state-gate-dataset-audit/v1",
        "passed": all(checks.values()),
        "checks": checks,
    }
    decision = contract["pass_action"] if audit["passed"] else contract["failure_action"]

    dataset_spec: dict[str, Any] = {
        "schema_version": "v67-v64-r2-probabilistic-state-gate-dataset-spec/v1",
        "family_id": contract["family_id"],
        "lineage_label": contract["lineage_label"],
        "phase_contract_file_sha256": dataset["phase_contract"]["file_sha256"],
        "parquet_access_contract": contract["parquet_access_contract"],
        "role_contract": contract["role_contract"],
        "data_contract": contract["data_contract"],
        "output_contract": contract["output_contract"],
        "parquet": dataset["parquet"],
        "pass_action": contract["pass_action"],
        "failure_action": contract["failure_action"],
    }
    dataset_spec["dataset_spec_sha256"] = canonical_sha256(dataset_spec)
    label_schema: dict[str, Any] = {
        "schema_version": "v67-v64-r2-label-schema/v1",
        "columns": list(labels.columns),
        "dtypes": {name: str(dtype) for name, dtype in labels.dtypes.items()},
        "formula": "log(open[t+2] / open[t+1])",
        "missing_policy": "preserve_without_imputation",
    }
    label_schema["label_schema_sha256"] = canonical_sha256(label_schema)
    role_schema: dict[str, Any] = {
        "schema_version": "v67-v64-r2-role-schema/v1",
        "columns": list(roles.columns),
        "dtypes": {name: str(dtype) for name, dtype in roles.dtypes.items()},
        "admitted_roles": contract["output_contract"]["admitted_roles"],
        "internal_validation_is_clean_evidence": False,
        "split_embargo_days": role_audit["split_embargo_days"],
    }
    role_schema["role_schema_sha256"] = canonical_sha256(role_schema)
    input_receipt = {
        name: {
            "path": dataset["inputs"][name],
            "sha256": context["input_hashes"][name],
        }
        for name in sorted(context["input_paths"])
    }
    data_access = {
        "authorized_inputs": list(contract["access_contract"]["allowed_inputs"]),
        "scan": scan_access,
        "loaded_symbols": sorted(set(labels["symbol"]) | set(roles["symbol"])),
        "loaded_date_bounds": {
            "labels": [
                _utc_bound(labels, "date", "min"),
                _utc_bound(labels, "date", "max"),
            ],
            "roles": [
                _utc_bound(roles, "date", "min"),
                _utc_bound(roles, "date", "max"),
            ],
        },
        "role_admission": role_audit,
        "operation_ledger": ledger.to_dict(),
    }
    dataset_manifest = {
        "schema_version": "v67-v64-r2-probabilistic-state-gate-dataset-manifest/v1",
        "labels": {
            **labels_write,
            "path": output_contract["labels_path"],
            "complete_rows": int(labels["h1_label_complete"].sum()),
            "incomplete_rows": label_missing,
        },
        "sequence_roles": {
            **roles_write,
            "path": output_contract["sequence_roles_path"],
        },
        "role_audit": role_audit,
        "symbols": context["symbols"],
        "source_v62_labels_sha256": context["input_hashes"]["labels"],
        "source_v62_sequence_roles_sha256": context["input_hashes"][
            "sequence_roles"
        ],
        "label_schema_sha256": label_schema["label_schema_sha256"],
        "role_schema_sha256": role_schema["role_schema_sha256"],
        "derivation_contract_sha256": derivation_contract[
            "derivation_contract_sha256"
        ],
    }
    replay_receipt = {
        "schema_version": "v67-v64-r2-dataset-replay-receipt/v1",
        "fresh_writes_per_output": 2,
        "labels_byte_identical": labels_write["byte_identical"],
        "labels_sha256": labels_write["sha256"],
        "sequence_roles_byte_identical": roles_write["byte_identical"],
        "sequence_roles_sha256": roles_write["sha256"],
    }
    replay_receipt["replay_receipt_sha256"] = canonical_sha256(replay_receipt)
    report = "\n".join(
        [
            "# V67 Non-target V64-R2 Gate Dataset",
            "",
            f"Decision: **{decision}**",
            "",
            f"Projected label rows: **{len(labels):,}**",
            f"Incomplete H1 rows preserved: **{label_missing:,}**",
            f"Admitted gate-role rows: **{len(roles):,}**",
            f"Gate-train rows: **{role_rows['gate_train']['eligible_rows']:,}**",
            (
                "Gate-internal-validation rows: "
                f"**{role_rows['gate_internal_validation']['eligible_rows']:,}**"
            ),
            (
                "Split-boundary rows embargoed: "
                f"**{role_audit['excluded_split_boundary_rows']:,}**"
            ),
            f"Labels SHA-256: `{labels_write['sha256']}`",
            f"Sequence roles SHA-256: `{roles_write['sha256']}`",
            "",
            "Both V62 Parquets were projected and predicate-filtered before",
            "materialization. No 2025 value or consumed-development role was read.",
            "The H1 market target and 18 state features remain exact on-demand",
            "derivations; no scaler, model, optimizer, checkpoint, prediction,",
            "metric, PnL, outcome, or target-asset operation occurred.",
            "",
            "A pass authorizes only a separately governed V68 frozen non-target",
            "gate-training phase. This packet does not train or evaluate a model.",
            "",
        ]
    )

    write_json_atomic(output / "dataset_spec.json", dataset_spec)
    write_json_atomic(output / "input_hash_receipt.json", input_receipt)
    write_json_atomic(output / "source_receipt.json", context["source_receipt"])
    write_json_atomic(output / "data_access.json", data_access)
    write_json_atomic(output / "dataset_manifest.json", dataset_manifest)
    write_json_atomic(output / "label_schema.json", label_schema)
    write_json_atomic(output / "role_schema.json", role_schema)
    write_json_atomic(output / "derivation_contract.json", derivation_contract)
    write_json_atomic(output / "replay_receipt.json", replay_receipt)
    write_json_atomic(output / "audit.json", audit)
    write_yaml_atomic(output / "resolved_config.yaml", config)
    (output / "report.md").write_text(report, encoding="utf-8")
    result: dict[str, Any] = {
        "schema_version": "v67-v64-r2-probabilistic-state-gate-dataset-result/v1",
        "family_id": contract["family_id"],
        "lineage_label": contract["lineage_label"],
        "decision": decision,
        "dataset_spec": dataset_spec,
        "dataset_manifest": dataset_manifest,
        "derivation_contract": derivation_contract,
        "data_access": data_access,
        "audit": audit,
    }
    result["result_sha256"] = canonical_sha256(result)
    write_json_atomic(output / "result.json", result)

    manifest_names = tuple(
        name
        for name in dataset["packet_files"]
        if name not in {"artifact_manifest.json", "completion_receipt.json"}
    )
    artifact_manifest: dict[str, Any] = {
        "schema_version": "v67-v64-r2-artifact-manifest/v1",
        "files": {name: file_sha256(output / name) for name in manifest_names},
        "data_files": {
            output_contract["labels_path"]: labels_write["sha256"],
            output_contract["sequence_roles_path"]: roles_write["sha256"],
        },
    }
    artifact_manifest["artifact_manifest_sha256"] = canonical_sha256(
        artifact_manifest
    )
    write_json_atomic(output / "artifact_manifest.json", artifact_manifest)
    completion: dict[str, Any] = {
        "schema_version": "v67-v64-r2-completion-receipt/v1",
        "family_id": contract["family_id"],
        "decision": decision,
        "audit_passed": audit["passed"],
        "dataset_spec_sha256": dataset_spec["dataset_spec_sha256"],
        "result_file_sha256": file_sha256(output / "result.json"),
        "result_sha256": result["result_sha256"],
        "audit_file_sha256": file_sha256(output / "audit.json"),
        "artifact_manifest_file_sha256": file_sha256(
            output / "artifact_manifest.json"
        ),
        "artifact_manifest_sha256": artifact_manifest[
            "artifact_manifest_sha256"
        ],
        "labels_sha256": labels_write["sha256"],
        "sequence_roles_sha256": roles_write["sha256"],
    }
    completion["completion_receipt_sha256"] = canonical_sha256(completion)
    write_json_atomic(output / "completion_receipt.json", completion)
    actual_packet_files = sorted(path.name for path in output.iterdir() if path.is_file())
    if actual_packet_files != sorted(dataset["packet_files"]):
        raise RuntimeError(f"V67 artifact packet file-set drift: {actual_packet_files}")
    if not audit["passed"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"V67 dataset audit failed: {failed}")
    return result
