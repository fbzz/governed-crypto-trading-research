from __future__ import annotations

from contextlib import contextmanager
from itertools import combinations
import fcntl
import hashlib
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
    "v61_result",
    "v61_audit",
    "v61_artifact_manifest",
    "v61_completion_receipt",
    "v60_specification",
    "v60_blueprint",
    "v32_result",
    "v32_audit",
    "v32_dataset_manifest",
    "v32_feature_schema",
    "v32_asset_folds",
    "v32_triplet_catalog",
}
BINARY_INPUTS = {"panel", "sequence_index"}
PANEL_FEATURES = [
    "log_open_to_open_return",
    "log_close_to_close_return",
    "log_high_low_range",
    "log_close_open_return",
    "log1p_quote_volume_change",
    "log1p_trade_count_change",
    "rolling_realized_volatility_7d",
    "rolling_realized_volatility_30d",
]
TRIPLET_FEATURE = "within_triplet_relative_strength"
LABEL_COLUMNS = [
    "date",
    "symbol",
    "eligible_action_date",
    "target_h1_maturity_date",
    "target_h1_open_to_open_log_return",
    "h1_label_complete",
]
SEQUENCE_ROLE_COLUMNS = [
    "date",
    "sequence_start_date",
    "symbol",
    "h1_label_complete",
    "eligible_train",
    "eligible_consumed_development_validation",
]


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
        raise ValueError(f"V62 path escapes project root: {relative}") from exc
    return path


@contextmanager
def _process_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another V62 dataset process holds the lock") from exc
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        path.unlink(missing_ok=True)


def build_h1_labels(panel: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "symbol", "raw_open", "raw_observation_available"}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"V62 source panel is missing columns: {missing}")
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame["symbol"] = frame["symbol"].astype(str)
    frame = frame.sort_values(["date", "symbol"]).reset_index(drop=True)
    if frame.duplicated(["symbol", "date"]).any():
        raise ValueError("V62 source panel key is not unique")
    for _, group in frame.groupby("symbol", sort=False):
        dates = pd.DatetimeIndex(group["date"])
        if len(dates) > 1 and not bool(
            (dates[1:] - dates[:-1] == pd.Timedelta(days=1)).all()
        ):
            raise ValueError("V62 source panel is not a complete daily calendar")

    grouped_open = frame.groupby("symbol", sort=False)["raw_open"]
    start_open = grouped_open.shift(-1).to_numpy(dtype=np.float64)
    end_open = grouped_open.shift(-2).to_numpy(dtype=np.float64)
    values = np.full(len(frame), np.nan, dtype=np.float64)
    valid = (
        np.isfinite(start_open)
        & np.isfinite(end_open)
        & (start_open > 0.0)
        & (end_open > 0.0)
    )
    values[valid] = np.log(end_open[valid] / start_open[valid])
    labels = frame[["date", "symbol"]].copy()
    labels["eligible_action_date"] = labels["date"] + pd.Timedelta(days=1)
    labels["target_h1_maturity_date"] = labels["date"] + pd.Timedelta(days=2)
    labels["target_h1_open_to_open_log_return"] = values
    labels["h1_label_complete"] = np.isfinite(values)
    return labels[LABEL_COLUMNS].copy()


def build_sequence_roles(
    sequence_index: pd.DataFrame,
    labels: pd.DataFrame,
    role_contract: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    required = {"date", "sequence_start_date", "symbol"}
    missing = sorted(required - set(sequence_index.columns))
    if missing:
        raise ValueError(f"V62 sequence index is missing columns: {missing}")
    sequence = sequence_index[["date", "sequence_start_date", "symbol"]].copy()
    sequence["date"] = pd.to_datetime(sequence["date"], utc=True)
    sequence["sequence_start_date"] = pd.to_datetime(
        sequence["sequence_start_date"], utc=True
    )
    sequence["symbol"] = sequence["symbol"].astype(str)
    sequence = sequence.sort_values(["date", "symbol"]).reset_index(drop=True)
    if sequence.duplicated(["symbol", "date"]).any():
        raise ValueError("V62 source sequence key is not unique")
    merged = sequence.merge(
        labels[
            [
                "date",
                "symbol",
                "target_h1_maturity_date",
                "h1_label_complete",
            ]
        ],
        on=["date", "symbol"],
        how="left",
        validate="one_to_one",
    )
    if merged["h1_label_complete"].isna().any():
        raise ValueError("V62 sequence keys are missing from the label table")

    roles = {
        "eligible_train": role_contract["train"],
        "eligible_consumed_development_validation": role_contract[
            "consumed_development_validation"
        ],
    }
    role_audit: dict[str, dict[str, Any]] = {}
    for column, window in roles.items():
        start = pd.Timestamp(window["signal_start"], tz="UTC")
        end = pd.Timestamp(window["signal_end"], tz="UTC")
        maturity_end = end + pd.Timedelta(days=2)
        flag = (
            merged["h1_label_complete"].astype(bool)
            & merged["date"].between(start, end, inclusive="both")
            & (merged["target_h1_maturity_date"] <= maturity_end)
        )
        merged[column] = flag
        eligible = merged.loc[flag]
        role_audit[column] = {
            "signal_start": window["signal_start"],
            "signal_end": window["signal_end"],
            "maturity_end": maturity_end.date().isoformat(),
            "eligible_rows": int(flag.sum()),
            "eligible_dates": int(eligible["date"].nunique()),
            "first_eligible_date": (
                eligible["date"].min().date().isoformat() if len(eligible) else None
            ),
            "last_eligible_date": (
                eligible["date"].max().date().isoformat() if len(eligible) else None
            ),
            "maximum_target_maturity": (
                eligible["target_h1_maturity_date"].max().date().isoformat()
                if len(eligible)
                else None
            ),
        }
    if (merged["eligible_train"] & merged["eligible_consumed_development_validation"]).any():
        raise ValueError("V62 train and consumed validation roles overlap")
    return merged[SEQUENCE_ROLE_COLUMNS].copy(), role_audit


def derive_triplet_contract(
    feature_tensor: np.ndarray,
    action_returns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    features = np.asarray(feature_tensor, dtype=np.float32)
    returns = np.asarray(action_returns, dtype=np.float64)
    if features.ndim != 3 or features.shape[1:] != (3, 9):
        raise ValueError("V62 triplet tensor must have shape [time, 3, 9]")
    if returns.shape != (3,) or not np.isfinite(returns).all():
        raise ValueError("V62 action returns must be a finite three-vector")
    state = np.concatenate(
        [features.mean(axis=1), features.std(axis=1, ddof=0)], axis=1
    ).astype(np.float32)
    market = float(returns.mean())
    excess = returns - market
    return features, state, market, excess


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _write_parquet_with_fresh_replay(
    frame: pd.DataFrame,
    final_path: Path,
    *,
    engine: str,
    compression: str,
    ledger: DatasetAccessLedger,
) -> dict[str, Any]:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    first = final_path.with_name(f".{final_path.name}.v62-replay-a.tmp")
    second = final_path.with_name(f".{final_path.name}.v62-replay-b.tmp")
    first.unlink(missing_ok=True)
    second.unlink(missing_ok=True)
    try:
        frame.to_parquet(first, index=False, engine=engine, compression=compression)
        frame.to_parquet(second, index=False, engine=engine, compression=compression)
        ledger.parquet_writes += 2
        first_hash = file_sha256(first)
        second_hash = file_sha256(second)
        if first_hash != second_hash:
            raise RuntimeError(f"V62 fresh Parquet replay drift: {final_path.name}")
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


def _embedded_hash_matches(value: dict[str, Any], key: str) -> bool:
    copy = dict(value)
    embedded = copy.pop(key, None)
    return isinstance(embedded, str) and embedded == canonical_sha256(copy)


def _metadata_context(
    config: dict[str, Any], ledger: DatasetAccessLedger
) -> dict[str, Any]:
    dataset = config["decoupled_rank_state_dataset"]
    root = Path(dataset["project_root"]).resolve()
    contract_reference = dataset["phase_contract"]
    contract_path = _project_path(root, contract_reference["path"])
    if (
        not contract_path.is_file()
        or file_sha256(contract_path) != contract_reference["file_sha256"]
    ):
        raise RuntimeError("V62 phase contract is missing or hash-drifted")
    contract = _load_yaml(contract_path, ledger)
    if (
        contract.get("phase") != "v62"
        or contract.get("stage_revision") != "v062_dataset_r1"
        or contract.get("authorized_next_action")
        != "authorize_v62_non_target_decoupled_rank_state_dataset_only"
        or config.get("output_dir") != contract["access_contract"]["output_dir"]
    ):
        raise RuntimeError("V62 frozen phase contract is inconsistent")

    input_paths = {
        name: _project_path(root, relative)
        for name, relative in dataset["inputs"].items()
    }
    if set(input_paths) != JSON_INPUTS | BINARY_INPUTS:
        raise RuntimeError("V62 input-name allowlist drift")
    if set(dataset["inputs"].values()) != set(
        contract["access_contract"]["allowed_inputs"]
    ):
        raise RuntimeError("V62 input-path allowlist drift")
    expected_by_path = contract["input_contract"][
        "expected_file_sha256_by_path"
    ]
    if set(expected_by_path) != set(dataset["inputs"].values()):
        raise RuntimeError("V62 expected-input hash map drift")
    observed_hashes: dict[str, str] = {}
    for name, path in input_paths.items():
        if not path.is_file():
            raise RuntimeError(f"V62 input is missing: {name}")
        observed_hashes[name] = file_sha256(path)
        relative = dataset["inputs"][name]
        if observed_hashes[name] != expected_by_path[relative]:
            raise RuntimeError(f"V62 input hash drift: {name}")

    values = {name: _load_json(input_paths[name], ledger) for name in JSON_INPUTS}
    v61_result = values["v61_result"]
    v61_completion = values["v61_completion_receipt"]
    v60_spec = values["v60_specification"]
    v60_blueprint = values["v60_blueprint"]
    v32_manifest = values["v32_dataset_manifest"]
    v32_schema = values["v32_feature_schema"]
    folds = values["v32_asset_folds"]
    catalog = values["v32_triplet_catalog"]
    if (
        values["v61_audit"].get("passed") is not True
        or v61_result.get("decision")
        != "authorize_v62_non_target_decoupled_rank_state_dataset_only"
        or not _embedded_hash_matches(v61_result, "result_sha256")
        or v61_completion.get("decision")
        != "authorize_v62_non_target_decoupled_rank_state_dataset_only"
        or v61_completion.get("audit_passed") is not True
        or not _embedded_hash_matches(
            values["v61_artifact_manifest"], "artifact_manifest_sha256"
        )
        or not _embedded_hash_matches(v61_completion, "completion_receipt_sha256")
        or v60_spec.get("specification_sha256")
        != v60_blueprint.get("specification_sha256")
        or v60_spec.get("data_contract", {}).get("source_panel_sha256")
        != observed_hashes["panel"]
        or v60_spec.get("data_contract", {}).get("source_sequence_index_sha256")
        != observed_hashes["sequence_index"]
        or values["v32_audit"].get("passed") is not True
        or v32_manifest.get("panel_sha256") != observed_hashes["panel"]
        or v32_manifest.get("sequence_index_sha256")
        != observed_hashes["sequence_index"]
        or values["v32_result"].get("dataset_manifest") != v32_manifest
        or list(v32_schema.get("model_feature_order", []))
        != list(dataset["feature_order"])
        or len(folds.get("folds", [])) != 3
        or len(catalog.get("folds", [])) != 3
    ):
        raise RuntimeError("V62 parent metadata contract drift")
    for fold, catalog_fold in zip(folds["folds"], catalog["folds"], strict=True):
        if (
            int(fold["fold"]) != int(catalog_fold["fold"])
            or sorted(fold["train_symbols"])
            != sorted(catalog_fold["train_symbols"])
            or sorted(fold["test_symbols"])
            != sorted(catalog_fold["test_symbols"])
        ):
            raise RuntimeError("V62 frozen fold/catalog roles drifted")

    source_files = list(dataset["source_receipt_files"])
    if not source_files or len(source_files) != len(set(source_files)):
        raise RuntimeError("V62 source receipt is empty or duplicated")
    source_hashes: dict[str, str] = {}
    for relative in source_files:
        path = _project_path(root, relative)
        if not path.is_file():
            raise RuntimeError(f"V62 source receipt file is missing: {relative}")
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
    panel_columns = [
        "date",
        "symbol",
        "raw_observation_available",
        "raw_open",
        *PANEL_FEATURES,
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
        raise RuntimeError(f"V62 loaded target symbols: {sorted(target_loads)}")
    if set(sequence["symbol"].astype(str)) != set(loaded_symbols):
        raise RuntimeError("V62 panel and sequence universes differ")
    return panel, sequence, loaded_symbols


def _source_h1_matches(panel: pd.DataFrame, labels: pd.DataFrame) -> bool:
    source = panel.sort_values(["date", "symbol"])[
        "target_next_open_to_next_open_log_return"
    ].to_numpy(dtype=np.float64)
    rebuilt = labels["target_h1_open_to_open_log_return"].to_numpy(dtype=np.float64)
    same_missing = np.array_equal(np.isnan(source), np.isnan(rebuilt))
    finite = np.isfinite(source) & np.isfinite(rebuilt)
    return bool(
        same_missing
        and np.allclose(source[finite], rebuilt[finite], atol=1e-15, rtol=1e-13)
    )


def _catalog_is_exact(
    folds: list[dict[str, Any]], catalog: list[dict[str, Any]]
) -> bool:
    for fold, catalog_fold in zip(folds, catalog, strict=True):
        train = sorted(fold["train_symbols"])
        test = sorted(fold["test_symbols"])
        expected_train = [list(group) for group in combinations(train, 3)]
        expected_test = [list(group) for group in combinations(test, 3)]
        if (
            catalog_fold["train_triplets"] != expected_train
            or catalog_fold["test_triplets"] != expected_test
        ):
            return False
    return True


def _materialize_smoke(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    roles: pd.DataFrame,
    catalog: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame["symbol"] = frame["symbol"].astype(str)
    label_frame = labels.set_index(["date", "symbol"])
    rows: list[dict[str, Any]] = []
    for catalog_fold in catalog:
        for catalog_key, role_flag in (
            ("train_triplets", "eligible_train"),
            ("test_triplets", "eligible_consumed_development_validation"),
        ):
            triplet = list(catalog_fold[catalog_key][0])
            eligible = roles.loc[
                roles[role_flag] & roles["symbol"].isin(triplet), ["date", "symbol"]
            ]
            counts = eligible.groupby("date")["symbol"].nunique()
            dates = counts[counts == 3].index
            materialized: dict[str, Any] | None = None
            for signal_date in dates:
                start = signal_date - pd.Timedelta(days=255)
                subset = frame.loc[
                    frame["date"].between(start, signal_date, inclusive="both")
                    & frame["symbol"].isin(triplet),
                    ["date", "symbol", *PANEL_FEATURES],
                ]
                tensors = []
                valid = True
                for symbol in triplet:
                    asset = subset.loc[subset["symbol"] == symbol].sort_values("date")
                    values = asset[PANEL_FEATURES].to_numpy(dtype=np.float32)
                    if len(asset) != 256 or not np.isfinite(values).all():
                        valid = False
                        break
                    tensors.append(values)
                if not valid:
                    continue
                base = np.stack(tensors, axis=1)
                relative = base[:, :, 1] - base[:, :, 1].mean(axis=1, keepdims=True)
                features = np.concatenate([base, relative[:, :, None]], axis=2)
                action = np.asarray(
                    [
                        label_frame.loc[
                            (signal_date, symbol),
                            "target_h1_open_to_open_log_return",
                        ]
                        for symbol in triplet
                    ],
                    dtype=np.float64,
                )
                features, state, market, excess = derive_triplet_contract(
                    features, action
                )
                permuted = features[:, ::-1, :]
                _, permuted_state, _, _ = derive_triplet_contract(permuted, action[::-1])
                materialized = {
                    "fold": int(catalog_fold["fold"]),
                    "catalog_role": catalog_key.removesuffix("_triplets"),
                    "role_flag": role_flag,
                    "signal_date": signal_date.date().isoformat(),
                    "triplet": triplet,
                    "input_shape": list(features.shape),
                    "state_shape": list(state.shape),
                    "input_dtype": str(features.dtype),
                    "state_dtype": str(state.dtype),
                    "input_sha256": _array_sha256(features),
                    "state_sha256": _array_sha256(state),
                    "market_component": market,
                    "centered_excess": excess.tolist(),
                    "centered_excess_sum": float(excess.sum()),
                    "maximum_reconstruction_error": float(
                        np.max(np.abs((market + excess) - action))
                    ),
                    "maximum_relative_feature_sum": float(
                        np.max(np.abs(features[:, :, -1].sum(axis=1)))
                    ),
                    "state_permutation_invariant": bool(
                        np.allclose(state, permuted_state, atol=1e-7, rtol=1e-6)
                    ),
                }
                break
            if materialized is None:
                raise RuntimeError(
                    f"V62 cannot materialize frozen smoke triplet: fold={catalog_fold['fold']} role={catalog_key}"
                )
            rows.append(materialized)
    return rows


def run_decoupled_rank_state_dataset(config: dict[str, Any]) -> dict[str, Any]:
    ledger = DatasetAccessLedger()
    context = _metadata_context(config, ledger)
    root = context["root"]
    contract = context["contract"]
    dataset = context["dataset"]
    output = _project_path(root, config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)

    with _process_lock(root / "data" / "processed" / ".v62-dataset.lock"):
        panel, sequence, loaded_symbols = _read_inputs(context, ledger)
        labels = build_h1_labels(panel)
        sequence_roles, role_audit = build_sequence_roles(
            sequence, labels, contract["role_contract"]
        )
        smoke = _materialize_smoke(
            panel,
            labels,
            sequence_roles,
            context["values"]["v32_triplet_catalog"]["folds"],
        )
        parquet = dataset["parquet"]
        output_contract = contract["output_contract"]
        label_write = _write_parquet_with_fresh_replay(
            labels,
            _project_path(root, output_contract["labels_path"]),
            engine=parquet["engine"],
            compression=parquet["compression"],
            ledger=ledger,
        )
        sequence_write = _write_parquet_with_fresh_replay(
            sequence_roles,
            _project_path(root, output_contract["sequence_roles_path"]),
            engine=parquet["engine"],
            compression=parquet["compression"],
            ledger=ledger,
        )

    input_hashes_after = {
        name: file_sha256(path) for name, path in context["input_paths"].items()
    }
    v32_manifest = context["values"]["v32_dataset_manifest"]
    folds = context["values"]["v32_asset_folds"]["folds"]
    catalog = context["values"]["v32_triplet_catalog"]["folds"]
    missing_keys = set(
        zip(
            pd.to_datetime(
                panel.loc[~panel["raw_observation_available"], "date"], utc=True
            ),
            panel.loc[~panel["raw_observation_available"], "symbol"].astype(str),
            strict=True,
        )
    )
    label_keys = set(zip(labels["date"], labels["symbol"], strict=True))
    sequence_keys = set(zip(sequence_roles["date"], sequence_roles["symbol"], strict=True))
    test_sets = [set(row["test_symbols"]) for row in folds]
    exact_fold_roles = (
        all(len(row["train_symbols"]) == 20 and len(row["test_symbols"]) == 10 for row in folds)
        and not any(test_sets[left] & test_sets[right] for left in range(3) for right in range(left + 1, 3))
        and set.union(*test_sets) == set(loaded_symbols)
        and _catalog_is_exact(folds, catalog)
    )
    role_boundaries_pass = (
        all(
            row["eligible_rows"] > 0
            and row["last_eligible_date"] <= row["signal_end"]
            and row["maximum_target_maturity"] <= row["maturity_end"]
            for row in role_audit.values()
        )
        and role_audit["eligible_train"]["maturity_end"]
        < contract["role_contract"]["consumed_development_validation"]["signal_start"]
        and not (
            sequence_roles["eligible_train"]
            & sequence_roles["eligible_consumed_development_validation"]
        ).any()
    )
    smoke_contract_pass = all(
        row["input_shape"] == [256, 3, 9]
        and row["state_shape"] == [256, 18]
        and row["input_dtype"] == "float32"
        and row["state_dtype"] == "float32"
        and abs(row["centered_excess_sum"]) <= 1e-15
        and row["maximum_reconstruction_error"] <= 1e-15
        and row["maximum_relative_feature_sum"] <= 1e-6
        and row["state_permutation_invariant"] is True
        for row in smoke
    )
    checks = {
        "all_input_hashes_match": input_hashes_after == context["input_hashes"],
        "exact_thirty_asset_universe_and_three_disjoint_folds": len(loaded_symbols) == 30
        and loaded_symbols == sorted(v32_manifest["symbols"])
        and exact_fold_roles,
        "exact_nine_feature_order_and_triplet_catalog": dataset["feature_order"]
        == [*PANEL_FEATURES, TRIPLET_FEATURE]
        and context["values"]["v32_feature_schema"]["model_feature_order"]
        == dataset["feature_order"]
        and all(
            len(row["train_triplets"]) == math.comb(20, 3)
            and len(row["test_triplets"]) == math.comb(10, 3)
            for row in catalog
        ),
        "exact_h1_open_t_plus_1_to_open_t_plus_2_label": _source_h1_matches(panel, labels)
        and labels["h1_label_complete"].equals(
            labels["target_h1_open_to_open_log_return"].notna()
        )
        and bool(
            (
                labels["target_h1_maturity_date"] - labels["date"]
                == pd.Timedelta(days=2)
            ).all()
        ),
        "exact_centered_excess_and_triplet_market_component_identity": smoke_contract_pass,
        "exact_eighteen_state_features_from_cross_asset_mean_and_population_std": smoke_contract_pass
        and dataset["state_feature_order"]
        == [
            "cross_asset_mean_of_nine_inputs",
            "cross_asset_population_std_of_nine_inputs",
        ],
        "no_label_crosses_registered_role_boundary": role_boundaries_pass,
        "panel_and_sequence_keys_are_unique": len(label_keys) == len(labels) == 60210
        and len(sequence_keys) == len(sequence_roles) == 49919,
        "missing_rows_are_preserved_without_imputation": len(missing_keys) == 1619
        and missing_keys.issubset(label_keys)
        and len(labels) == len(panel)
        and len(sequence_roles) == len(sequence)
        and ledger.missing_value_imputations == 0,
        "target_assets_are_absent": not TARGET_SYMBOLS.intersection(loaded_symbols)
        and ledger.target_asset_loads == 0,
        "no_scaler_model_optimizer_checkpoint_prediction_performance_or_pnl": ledger.forbidden_operations_are_zero()
        and ledger.authorized_parquet_deserializations == 2,
        "byte_identical_replay": label_write["byte_identical"]
        and sequence_write["byte_identical"]
        and label_write["sha256"] == label_write["fresh_replay_sha256"]
        and sequence_write["sha256"] == sequence_write["fresh_replay_sha256"],
        "output_schema_is_exact": list(labels.columns) == LABEL_COLUMNS
        and list(sequence_roles.columns) == SEQUENCE_ROLE_COLUMNS,
        "phase_contract_hash_matches": file_sha256(context["contract_path"])
        == dataset["phase_contract"]["file_sha256"],
        "source_receipt_is_complete": bool(context["source_receipt"]["files"])
        and len(context["source_receipt"]["bundle_sha256"]) == 64,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    audit = {
        "schema_version": "v62-decoupled-rank-state-dataset-audit/v1",
        "passed": all(checks.values()),
        "checks": checks,
    }
    decision = contract["pass_action"] if audit["passed"] else contract["failure_action"]

    dataset_spec: dict[str, Any] = {
        "schema_version": "v62-decoupled-rank-state-dataset-spec/v1",
        "family_id": contract["family_id"],
        "phase_contract_file_sha256": dataset["phase_contract"]["file_sha256"],
        "data_contract": contract["data_contract"],
        "role_contract": contract["role_contract"],
        "output_contract": contract["output_contract"],
        "feature_order": dataset["feature_order"],
        "state_feature_order": dataset["state_feature_order"],
        "parquet": dataset["parquet"],
        "pass_action": contract["pass_action"],
        "failure_action": contract["failure_action"],
    }
    dataset_spec["dataset_spec_sha256"] = canonical_sha256(dataset_spec)
    label_schema: dict[str, Any] = {
        "schema_version": "v62-label-schema/v1",
        "columns": list(labels.columns),
        "dtypes": {name: str(dtype) for name, dtype in labels.dtypes.items()},
        "formula": contract["data_contract"]["action_return"]["formula"],
        "missing_policy": contract["data_contract"]["missing_data"],
    }
    label_schema["label_schema_sha256"] = canonical_sha256(label_schema)
    input_receipt = {
        name: {
            "path": dataset["inputs"][name],
            "sha256": context["input_hashes"][name],
        }
        for name in sorted(context["input_paths"])
    }
    data_access = {
        "authorized_inputs": list(contract["access_contract"]["allowed_inputs"]),
        "panel_projection": [
            "date",
            "symbol",
            "raw_observation_available",
            "raw_open",
            *PANEL_FEATURES,
            "target_next_open_to_next_open_log_return",
        ],
        "sequence_projection": ["date", "sequence_start_date", "symbol"],
        "loaded_symbols": loaded_symbols,
        "operation_ledger": ledger.to_dict(),
    }
    dataset_manifest = {
        "schema_version": "v62-decoupled-rank-state-dataset-manifest/v1",
        "labels": {
            **label_write,
            "path": output_contract["labels_path"],
            "complete_rows": int(labels["h1_label_complete"].sum()),
        },
        "sequence_roles": {
            **sequence_write,
            "path": output_contract["sequence_roles_path"],
        },
        "role_audit": role_audit,
        "symbols": loaded_symbols,
        "source_panel_sha256": context["input_hashes"]["panel"],
        "source_sequence_index_sha256": context["input_hashes"]["sequence_index"],
        "feature_schema_sha256": context["input_hashes"]["v32_feature_schema"],
        "asset_folds_sha256": context["input_hashes"]["v32_asset_folds"],
        "triplet_catalog_sha256": context["input_hashes"]["v32_triplet_catalog"],
        "label_schema_sha256": label_schema["label_schema_sha256"],
        "triplet_smoke_sha256": canonical_sha256(smoke),
    }
    replay_receipt = {
        "schema_version": "v62-dataset-replay-receipt/v1",
        "fresh_writes_per_output": 2,
        "labels_byte_identical": label_write["byte_identical"],
        "labels_sha256": label_write["sha256"],
        "sequence_roles_byte_identical": sequence_write["byte_identical"],
        "sequence_roles_sha256": sequence_write["sha256"],
    }
    replay_receipt["replay_receipt_sha256"] = canonical_sha256(replay_receipt)
    report = "\n".join(
        [
            "# V62 Non-target Decoupled Rank/State Dataset",
            "",
            f"Decision: **{decision}**",
            "",
            f"Label rows: **{len(labels):,}**",
            f"Complete H1 rows: **{int(labels['h1_label_complete'].sum()):,}**",
            f"Sequence-role rows: **{len(sequence_roles):,}**",
            f"Triplet derivation smoke contexts: **{len(smoke)}**",
            f"Labels SHA-256: `{label_write['sha256']}`",
            f"Sequence roles SHA-256: `{sequence_write['sha256']}`",
            "",
            "The frozen H1 endpoint return and both chronological roles were",
            "materialized without imputation. Centered excess, market component,",
            "and 18 state features remain exact on-demand triplet derivations.",
            "",
            "BTC/ETH/SOL, scalers, models, optimizers, checkpoints, predictions,",
            "outcomes, performance metrics, and PnL remained unopened. A pass",
            "authorizes only a separately governed V63 non-target training phase.",
            "",
        ]
    )

    write_json_atomic(output / "dataset_spec.json", dataset_spec)
    write_json_atomic(output / "input_hash_receipt.json", input_receipt)
    write_json_atomic(output / "source_receipt.json", context["source_receipt"])
    write_json_atomic(output / "data_access.json", data_access)
    write_json_atomic(output / "dataset_manifest.json", dataset_manifest)
    write_json_atomic(output / "label_schema.json", label_schema)
    write_json_atomic(output / "triplet_derivation_smoke.json", smoke)
    write_json_atomic(output / "replay_receipt.json", replay_receipt)
    write_json_atomic(output / "audit.json", audit)
    write_yaml_atomic(output / "resolved_config.yaml", config)
    (output / "report.md").write_text(report, encoding="utf-8")
    result: dict[str, Any] = {
        "schema_version": "v62-decoupled-rank-state-dataset-result/v1",
        "family_id": contract["family_id"],
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
        for name in dataset["packet_files"]
        if name not in {"artifact_manifest.json", "completion_receipt.json"}
    )
    artifact_manifest: dict[str, Any] = {
        "schema_version": "v62-artifact-manifest/v1",
        "files": {name: file_sha256(output / name) for name in manifest_names},
        "data_files": {
            output_contract["labels_path"]: label_write["sha256"],
            output_contract["sequence_roles_path"]: sequence_write["sha256"],
        },
    }
    artifact_manifest["artifact_manifest_sha256"] = canonical_sha256(
        artifact_manifest
    )
    write_json_atomic(output / "artifact_manifest.json", artifact_manifest)
    completion: dict[str, Any] = {
        "schema_version": "v62-completion-receipt/v1",
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
        "labels_sha256": label_write["sha256"],
        "sequence_roles_sha256": sequence_write["sha256"],
    }
    completion["completion_receipt_sha256"] = canonical_sha256(completion)
    write_json_atomic(output / "completion_receipt.json", completion)
    actual_packet_files = sorted(path.name for path in output.iterdir() if path.is_file())
    if actual_packet_files != sorted(dataset["packet_files"]):
        raise RuntimeError(f"V62 artifact packet file-set drift: {actual_packet_files}")
    if not audit["passed"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"V62 dataset audit failed: {failed}")
    return result
