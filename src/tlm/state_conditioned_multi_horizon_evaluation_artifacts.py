"""Artifact and governance helpers for the frozen V59 prepare stage.

The V59 prepare packet deliberately does not authorize outcome access.  This
module implements a contract-aware v2 receipt because the legacy generic
one-shot v1 receipt conflates a passing prepare with permission to unseal.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import NamedTemporaryFile
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
import pyarrow.parquet as pq

from .core.artifacts import canonical_sha256, file_sha256
from .research_workflow import (
    V59_PHASE_CONTRACT_CANONICAL_SHA256,
    validate_research_state,
)
from .state_conditioned_multi_horizon_evaluation_data import (
    PANEL_COLUMNS,
    SEQUENCE_BASE_COLUMNS,
    TRAIN_LABEL_COLUMNS,
    _key_dnf,
    day_text,
    jsonable,
    key_sha256,
    utc_day,
)


V59_PHASE_FILE_SHA256 = (
    "321c6a805b94f73d441def62af7478337b5e33f5f23631808dba032a376df6a2"
)
PREPARE_SCHEMA = "tlm-one-shot-prepare/v2"
PREPARE_DECISION = "await_explicit_v59_registered_outcome_unseal_authorization"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_HEAD = re.compile(r"^[0-9a-f]{40,64}$")
REQUIRED_PREPARE_FILES = (
    "evaluation_spec.json",
    "source_receipt.json",
    "input_hash_receipt.json",
    "checkpoint_binding.json",
    "scaler_binding.json",
    "linear_control_receipt.json",
    "predictions.parquet",
    "candidate_positions.parquet",
    "control_positions.parquet",
    "outcome_request.json",
    "behavior_audit.json",
    "prepare_manifest.json",
    "prepare_receipt.json",
)
MANIFEST_PAYLOAD_FILES = REQUIRED_PREPARE_FILES[:-2]
FORBIDDEN_PREPARE_FILES = (
    "authorization_receipt.json",
    "outcome_packet.parquet",
    "outcome_receipt.json",
    "metrics.json",
    "bootstrap.json",
    "gate_matrix.json",
    "result.json",
    "audit.json",
    "report.md",
    "completion_receipt.json",
    "artifact_manifest.json",
    "replay.json",
)
REGISTERED_PROJECTION_KEYS = (
    "phase",
    "family_id",
    "status",
    "evidence_tier",
    "authorized_next_action",
    "authorized_command",
    "prepare_pass_action",
    "prepare_failure_action",
    "pass_action",
    "failure_action",
    "prepare_parquet_access_contract",
    "evaluation_cells",
    "feature_tensor_contract",
    "inference_contract",
    "linear_control_contract",
    "decision_contract",
    "control_contract",
    "accounting_contract",
    "metric_contract",
    "bootstrap_contract",
    "outcome_blind_gate_contract",
    "gate_contract",
    "one_shot_contract",
    "target_contract",
    "runtime_contract",
    "storage_contract",
    "artifact_contract",
    "commands",
)
V59_SOURCE_FILES = (
    "configs/v59_state_conditioned_multi_horizon_evaluation.yaml",
    "research/current.yaml",
    "research/experiments/v058.yaml",
    "research/phase_contracts/v059.yaml",
    "research/waivers/v059_local_artifacts_owner_waiver.json",
    "src/tlm/__main__.py",
    "src/tlm/config.py",
    "src/tlm/core/__init__.py",
    "src/tlm/core/artifacts.py",
    "src/tlm/research_workflow.py",
    "src/tlm/state_conditioned_multi_horizon_model.py",
    "src/tlm/state_conditioned_multi_horizon_training_data.py",
    "src/tlm/state_conditioned_multi_horizon_training_engine.py",
    "src/tlm/state_conditioned_multi_horizon_evaluation_artifacts.py",
    "src/tlm/state_conditioned_multi_horizon_evaluation_data.py",
    "src/tlm/state_conditioned_multi_horizon_evaluation.py",
    "tests/test_v59_phase_registration.py",
    "tests/test_state_conditioned_multi_horizon_evaluation.py",
    "pyproject.toml",
)


class V59PrepareError(RuntimeError):
    """Raised when the registered V59 prepare contract is violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise V59PrepareError(message)


def mapping(value: Any, name: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{name} must be an object")
    return value


def resolve_repo_path(root: str | Path, relative: str | Path, name: str) -> Path:
    repository = Path(root).resolve()
    raw = Path(relative)
    require(str(relative) not in {"", "."}, f"{name} must be a file path")
    require(not raw.is_absolute(), f"{name} must be repository-relative")
    result = (repository / raw).resolve()
    require(
        result != repository and repository in result.parents,
        f"{name} escapes the repository",
    )
    return result


def load_json(path: str | Path, name: str = "JSON") -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V59PrepareError(f"{name} is not valid JSON: {exc}") from exc
    return mapping(value, name)


def load_yaml(path: str | Path, name: str = "YAML") -> dict[str, Any]:
    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise V59PrepareError(f"{name} is not valid YAML: {exc}") from exc
    return mapping(value, name)


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def with_self_hash(value: dict[str, Any], field: str) -> dict[str, Any]:
    result = deepcopy(value)
    require(field not in result, f"{field} already exists")
    result[field] = canonical_sha256(result)
    return result


def verify_self_hash(value: Mapping[str, Any], field: str, name: str) -> str:
    registered = value.get(field)
    require(
        isinstance(registered, str) and SHA256.fullmatch(registered) is not None,
        f"{name} lacks a valid {field}",
    )
    body = deepcopy(dict(value))
    body.pop(field, None)
    require(canonical_sha256(body) == registered, f"{name} self-hash drift")
    return registered


def _git(root: Path, *arguments: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        capture_output=True,
        text=not binary,
        check=False,
    )
    stderr = result.stderr.decode(errors="replace") if binary else result.stderr
    require(
        result.returncode == 0,
        f"git {' '.join(arguments)} failed: {str(stderr).strip()}",
    )
    return result.stdout if binary else result.stdout.strip()


def source_receipt(root: Path, source_files: Sequence[str]) -> dict[str, Any]:
    files = list(source_files)
    require(files and len(files) == len(set(files)), "source file set is empty or duplicated")
    require(
        all(isinstance(item, str) and item for item in files),
        "source paths must be non-empty strings",
    )
    top = Path(str(_git(root, "rev-parse", "--show-toplevel"))).resolve()
    require(top == root, "project_root is not the Git repository root")
    head = str(_git(root, "rev-parse", "HEAD"))
    require(GIT_HEAD.fullmatch(head) is not None, "Git HEAD is invalid")
    require(
        _git(root, "status", "--porcelain", "--untracked-files=all") == "",
        "V59 prepare requires a clean committed Git worktree",
    )
    raw = _git(root, "ls-files", "-z", "--", *files, binary=True)
    assert isinstance(raw, bytes)
    tracked = {item.decode("utf-8") for item in raw.split(b"\0") if item}
    require(tracked == set(files), "source receipt includes missing or untracked files")
    hashes: dict[str, str] = {}
    for relative in files:
        path = resolve_repo_path(root, relative, f"source file {relative}")
        require(path.is_file(), f"source file is missing: {relative}")
        hashes[relative] = file_sha256(path)
    body = {
        "schema_version": "v59-source-receipt/v1",
        "git_clean": True,
        "git_head": head,
        "files": hashes,
        "bundle_sha256": canonical_sha256(hashes),
    }
    return with_self_hash(body, "source_receipt_sha256")


def registered_projection(contract: Mapping[str, Any]) -> dict[str, Any]:
    require(
        all(key in contract for key in REGISTERED_PROJECTION_KEYS),
        "V59 phase contract lacks a registered projection field",
    )
    return {key: deepcopy(contract[key]) for key in REGISTERED_PROJECTION_KEYS}


def load_live_v59_contract(
    root: Path, *, state_path: str, phase_contract_path: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    status = validate_research_state(root, state_path)
    require(status.get("passed") is True, "live research state did not pass")
    require(status.get("authorized_phase") == "v59", "live phase is not V59")
    require(
        status.get("authorized_next_action")
        == "authorize_v59_frozen_adaptive_development_evaluation_only",
        "live V59 action drift",
    )
    require(
        status.get("phase_contract_path") == phase_contract_path,
        "live V59 phase-contract path drift",
    )
    path = resolve_repo_path(root, phase_contract_path, "phase_contract")
    require(path.is_file(), "V59 phase contract is missing")
    require(file_sha256(path) == V59_PHASE_FILE_SHA256, "V59 phase file hash drift")
    contract = load_yaml(path, "V59 phase contract")
    require(
        canonical_sha256(contract) == V59_PHASE_CONTRACT_CANONICAL_SHA256,
        "V59 phase semantic hash drift",
    )
    return contract, status


def verify_input_files(
    root: Path, contract: Mapping[str, Any]
) -> dict[str, Any]:
    bindings = mapping(
        mapping(contract.get("input_contract"), "input_contract").get(
            "expected_file_sha256_by_path"
        ),
        "expected_file_sha256_by_path",
    )
    observed: dict[str, str] = {}
    for relative, expected in bindings.items():
        require(
            isinstance(relative, str)
            and isinstance(expected, str)
            and SHA256.fullmatch(expected) is not None,
            "invalid input path/hash binding",
        )
        path = resolve_repo_path(root, relative, f"input {relative}")
        require(path.is_file(), f"registered input is missing: {relative}")
        actual = file_sha256(path)
        require(actual == expected, f"registered input hash drift: {relative}")
        observed[relative] = actual
    body = {
        "schema_version": "v59-input-hash-receipt/v1",
        "files": observed,
        "file_count": len(observed),
        "development_outcome_value_reads": 0,
        "target_asset_loads": 0,
    }
    return with_self_hash(body, "input_hash_receipt_sha256")


def minimum_free_space(root: Path, gib: float) -> dict[str, Any]:
    free = shutil.disk_usage(root).free
    required = int(float(gib) * 1024**3)
    require(free >= required, f"V59 requires at least {gib:g} GiB free")
    return {
        "required_free_gib": float(gib),
        "observed_free_bytes": int(free),
        "passed": True,
    }


@contextmanager
def process_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise V59PrepareError("another V59 prepare process owns the lock") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def build_prepare_manifest(
    directory: Path,
    *,
    parquet_metadata: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for name in MANIFEST_PAYLOAD_FILES:
        path = directory / name
        require(path.is_file(), f"prepare artifact is missing: {name}")
        entry: dict[str, Any] = {
            "path": name,
            "size_bytes": path.stat().st_size,
            "file_sha256": file_sha256(path),
        }
        if name in parquet_metadata:
            entry.update(dict(parquet_metadata[name]))
        entries.append(entry)
    body = {
        "schema_version": "v59-prepare-manifest/v1",
        "files": entries,
        "file_count": len(entries),
    }
    return with_self_hash(body, "prepare_manifest_sha256")


def _manifest_map(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = manifest.get("files")
    require(isinstance(rows, list), "prepare manifest files must be an array")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        require(isinstance(row, dict), "prepare manifest row must be an object")
        name = row.get("path")
        require(isinstance(name, str) and name not in result, "manifest path drift")
        result[name] = row
    return result


def _registered_access_receipts(
    root: Path,
    phase: Mapping[str, Any],
    registered_inputs: Mapping[str, str],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[tuple[str, int], list[tuple[str, str]]],
    dict[str, dict[str, Any]],
]:
    """Rederive outcome-blind access receipts from exact projected role rows."""

    folds_path = resolve_repo_path(
        root,
        "artifacts/v32_selected_universe_dataset/asset_folds.json",
        "V32 asset folds",
    )
    dataset_spec_path = resolve_repo_path(
        root,
        "artifacts/v57_non_target_multi_horizon_dataset/dataset_spec.json",
        "V57 dataset spec",
    )
    catalog_path = resolve_repo_path(
        root,
        "artifacts/v32_selected_universe_dataset/triplet_catalog.json",
        "V32 triplet catalog",
    )
    sequence_path = resolve_repo_path(
        root,
        "data/processed/state_conditioned_multi_horizon_sequence_roles_v57.parquet",
        "V57 sequence roles",
    )
    for relative, path in (
        ("artifacts/v32_selected_universe_dataset/asset_folds.json", folds_path),
        (
            "artifacts/v57_non_target_multi_horizon_dataset/dataset_spec.json",
            dataset_spec_path,
        ),
        ("artifacts/v32_selected_universe_dataset/triplet_catalog.json", catalog_path),
    ):
        require(
            file_sha256(path) == registered_inputs.get(relative),
            f"V59 access verifier metadata hash drift: {relative}",
        )
    folds = load_json(folds_path, "V32 asset folds")
    dataset_spec = load_json(dataset_spec_path, "V57 dataset spec")
    catalog = load_json(catalog_path, "V32 triplet catalog")
    folds_by_id = {int(row["fold"]): row for row in folds.get("folds", [])}
    origins_by_id = {
        str(row["id"]): row
        for row in dataset_spec.get("role_contract", {}).get("origins", [])
    }
    catalog_by_fold = {
        int(row["fold"]): tuple(
            tuple(map(str, triplet)) for triplet in row["test_triplets"]
        )
        for row in catalog.get("folds", [])
    }
    require(
        set(folds_by_id) == {1, 2, 3}
        and set(catalog_by_fold) == {1, 2, 3}
        and set(origins_by_id) == {"origin_2024", "origin_2025"},
        "V59 access verifier fold/origin metadata drift",
    )
    access = phase["prepare_parquet_access_contract"]
    expected_by_cell: dict[str, dict[str, Any]] = {}
    outcome_keys_by_group: dict[tuple[str, int], list[tuple[str, str]]] = {}
    context_by_cell: dict[str, dict[str, Any]] = {}
    target_symbols = set(map(str, phase["target_contract"]["symbols"]))
    for origin in ("origin_2024", "origin_2025"):
        origin_row = origins_by_id[origin]
        development = phase["evaluation_cells"]["origins"][origin]
        for geometry in ("expanding", "rolling"):
            train = origin_row["geometries"][geometry]["train"]
            flags = access["exact_role_flags"][origin][geometry]
            train_flag = str(flags["train"])
            development_flag = str(flags["development"])
            sequence_columns = (
                *SEQUENCE_BASE_COLUMNS,
                train_flag,
                development_flag,
            )
            for fold in (1, 2, 3):
                fold_row = folds_by_id[fold]
                train_symbols = tuple(map(str, fold_row["train_symbols"]))
                test_symbols = tuple(map(str, fold_row["test_symbols"]))
                require(
                    len(train_symbols) == 20
                    and len(test_symbols) == 10
                    and not (set(train_symbols) | set(test_symbols)).intersection(
                        target_symbols
                    ),
                    "V59 access verifier symbol boundary drift",
                )
                sequence_filters = [
                    [
                        ("symbol", "in", train_symbols),
                        (train_flag, "==", True),
                        ("date", ">=", utc_day(train["signal_start"])),
                        ("date", "<=", utc_day(train["signal_end"])),
                    ],
                    [
                        ("symbol", "in", test_symbols),
                        (development_flag, "==", True),
                        ("date", ">=", utc_day(development["signal_start"])),
                        ("date", "<=", utc_day(development["signal_end"])),
                    ],
                ]
                reader_filters = [
                    [
                        (
                            column,
                            operator,
                            list(value) if operator == "in" else value,
                        )
                        for column, operator, value in conjunction
                    ]
                    for conjunction in sequence_filters
                ]
                sequence = pd.read_parquet(
                    sequence_path,
                    engine="pyarrow",
                    columns=list(sequence_columns),
                    filters=reader_filters,
                )
                require(
                    list(sequence.columns) == list(sequence_columns)
                    and not sequence.empty,
                    "V59 independent sequence projection drift",
                )
                for column in ("date", "sequence_start_date"):
                    sequence[column] = sequence[column].map(utc_day)
                sequence["symbol"] = sequence["symbol"].astype(str)
                require(
                    not sequence.duplicated(["date", "symbol"]).any(),
                    "V59 independent sequence keys are duplicated",
                )
                train_mask = sequence[train_flag].astype(bool)
                development_mask = sequence[development_flag].astype(bool)
                require(
                    ((train_mask ^ development_mask).all()),
                    "V59 independent sequence role isolation drift",
                )
                train_rows = sequence.loc[train_mask]
                development_rows = sequence.loc[development_mask]
                require(
                    bool(set(train_rows["symbol"]))
                    and bool(set(development_rows["symbol"]))
                    and set(train_rows["symbol"]).issubset(train_symbols)
                    and set(development_rows["symbol"]).issubset(test_symbols),
                    "V59 independent sequence role-symbol drift",
                )
                train_keys = frozenset(
                    zip(train_rows["date"], train_rows["symbol"], strict=True)
                )
                development_keys = frozenset(
                    zip(
                        development_rows["date"],
                        development_rows["symbol"],
                        strict=True,
                    )
                )
                development_context_keys: set[tuple[pd.Timestamp, str]] = set()
                for row in development_rows.itertuples(index=False):
                    dates = pd.date_range(
                        row.sequence_start_date,
                        row.date,
                        freq="D",
                        tz="UTC",
                    )
                    require(
                        len(dates) == 256,
                        "V59 independent development context length drift",
                    )
                    development_context_keys.update(
                        (date, str(row.symbol)) for date in dates
                    )
                panel_keys = train_keys | frozenset(development_context_keys)
                cell_id = f"{origin}|{geometry}|{fold}"
                sequence_start_by_key = {
                    (row.date, str(row.symbol)): row.sequence_start_date
                    for row in development_rows.itertuples(index=False)
                }
                context_by_cell[cell_id] = {
                    "origin": origin,
                    "geometry": geometry,
                    "fold": fold,
                    "development_start": utc_day(development["signal_start"]),
                    "development_end": utc_day(development["signal_end"]),
                    "test_symbols": test_symbols,
                    "test_triplets": catalog_by_fold[fold],
                    "development_keys": development_keys,
                    "sequence_start_by_key": sequence_start_by_key,
                    "panel_keys": panel_keys,
                }
                expected_by_cell[cell_id] = {
                    "cell_id": cell_id,
                    "projected_columns": {
                        "sequence_roles": list(sequence_columns),
                        "train_labels": list(TRAIN_LABEL_COLUMNS),
                        "panel": list(PANEL_COLUMNS),
                        "development_labels": [],
                        "development_outcomes": [],
                    },
                    "predicate_dnf": {
                        "sequence_roles": jsonable(sequence_filters),
                        "train_labels": jsonable(_key_dnf(train_keys)),
                        "panel": jsonable(_key_dnf(panel_keys)),
                    },
                    "roles": {
                        "train": {
                            "date_min": day_text(train_rows["date"].min()),
                            "date_max": day_text(train_rows["date"].max()),
                            "symbols": sorted(train_rows["symbol"].unique()),
                            "key_count": len(train_keys),
                            "key_sha256": key_sha256(train_keys),
                        },
                        "development": {
                            "date_min": day_text(development_rows["date"].min()),
                            "date_max": day_text(development_rows["date"].max()),
                            "symbols": sorted(development_rows["symbol"].unique()),
                            "key_count": len(development_keys),
                            "key_sha256": key_sha256(development_keys),
                        },
                        "development_context": {
                            "key_count": len(development_context_keys),
                            "key_sha256": key_sha256(development_context_keys),
                        },
                    },
                    "train_label_key_count": len(train_keys),
                    "train_label_key_sha256": key_sha256(train_keys),
                    "development_sequence_key_count": len(development_keys),
                    "development_sequence_key_sha256": key_sha256(
                        development_keys
                    ),
                    "feature_context_key_count": len(panel_keys),
                    "feature_context_key_sha256": key_sha256(panel_keys),
                    "development_outcome_value_reads": 0,
                    "development_outcome_columns_materialized": [],
                    "target_asset_loads": 0,
                    "full_table_materializations": 0,
                }
                group = (origin, fold)
                records = [
                    (day_text(date), str(symbol))
                    for date, symbol in sorted(development_keys)
                ]
                if group in outcome_keys_by_group:
                    require(
                        outcome_keys_by_group[group] == records,
                        "V59 independent development keys differ by geometry",
                    )
                else:
                    outcome_keys_by_group[group] = records
    return expected_by_cell, outcome_keys_by_group, context_by_cell


def _decision_schedule(eligible: np.ndarray) -> np.ndarray:
    active = np.asarray(eligible, dtype=bool)
    result = np.zeros(len(active), dtype=bool)
    has_decided = False
    since = 0
    for index, available in enumerate(active):
        if not available:
            continue
        if not has_decided:
            result[index] = True
            has_decided = True
            since = 0
        else:
            since += 1
            if since >= 7:
                result[index] = True
                since = 0
    return result


def _tie_best(utilities: Sequence[float]) -> int:
    best = 0
    best_value = float(utilities[0])
    for index, value in enumerate(utilities[1:], start=1):
        current = float(value)
        if current > best_value + 1.0e-12:
            best = index
            best_value = current
    return best


def _state_policy(forecasts: np.ndarray, eligible: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(forecasts, dtype=np.float64)
    active = np.asarray(eligible, dtype=bool)
    require(values.shape == (len(active), 3), "V59 verifier forecast geometry drift")
    decision = _decision_schedule(active)
    forced = np.zeros(len(active), dtype=bool)
    weights = np.zeros((len(active), 3), dtype=np.float64)
    current = np.zeros(3, dtype=np.float64)
    for day in range(len(active)):
        if not active[day]:
            if current.sum() > 0:
                forced[day] = True
            current = np.zeros(3, dtype=np.float64)
        elif decision[day]:
            candidates = [current.copy()]
            cash = np.zeros(3, dtype=np.float64)
            if not np.array_equal(current, cash):
                candidates.append(cash)
            for slot in range(3):
                candidate = np.zeros(3, dtype=np.float64)
                candidate[slot] = 1.0 / 3.0
                if not any(np.array_equal(candidate, item) for item in candidates):
                    candidates.append(candidate)
            utilities = [
                float(np.dot(candidate, values[day]))
                - 0.001 * float(np.abs(candidate - current).sum())
                for candidate in candidates
            ]
            current = candidates[_tie_best(utilities)].copy()
        weights[day] = current
    return {"weights": weights, "decision": decision, "forced": forced}


def _momentum_policy(scores: np.ndarray, eligible: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64)
    active = np.asarray(eligible, dtype=bool)
    decision = _decision_schedule(active)
    forced = np.zeros(len(active), dtype=bool)
    weights = np.zeros((len(active), 3), dtype=np.float64)
    current = np.zeros(3, dtype=np.float64)
    for day in range(len(active)):
        if not active[day]:
            if current.sum() > 0:
                forced[day] = True
            current = np.zeros(3, dtype=np.float64)
        elif decision[day]:
            maximum = float(np.max(values[day]))
            if not np.isfinite(maximum) or maximum <= 0.0:
                current = np.zeros(3, dtype=np.float64)
            else:
                tied = [
                    slot
                    for slot in range(3)
                    if abs(float(values[day, slot]) - maximum) <= 1.0e-12
                ]
                incumbent = int(np.argmax(current)) if current.sum() > 0 else None
                selected = incumbent if incumbent in tied else tied[0]
                current = np.zeros(3, dtype=np.float64)
                current[selected] = 1.0 / 3.0
        weights[day] = current
    return {"weights": weights, "decision": decision, "forced": forced}


def _cash_policy(eligible: np.ndarray) -> dict[str, np.ndarray]:
    active = np.asarray(eligible, dtype=bool)
    return {
        "weights": np.zeros((len(active), 3), dtype=np.float64),
        "decision": _decision_schedule(active),
        "forced": np.zeros(len(active), dtype=bool),
    }


def _equal_policy(eligible: np.ndarray) -> dict[str, np.ndarray]:
    active = np.asarray(eligible, dtype=bool)
    decision = _decision_schedule(active)
    forced = np.zeros(len(active), dtype=bool)
    weights = np.zeros((len(active), 3), dtype=np.float64)
    current = np.zeros(3, dtype=np.float64)
    for day in range(len(active)):
        if not active[day]:
            if current.sum() > 0:
                forced[day] = True
            current = np.zeros(3, dtype=np.float64)
        elif decision[day]:
            current = np.full(3, 1.0 / 9.0, dtype=np.float64)
        weights[day] = current
    return {"weights": weights, "decision": decision, "forced": forced}


def _registered_behavior_state(
    root: Path,
    registered_inputs: Mapping[str, str],
    cell_contexts: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Independently derive availability, missing keys, and momentum without outcomes."""

    panel_path = resolve_repo_path(
        root,
        "data/processed/selected_universe_panel_v32.parquet",
        "V32 panel",
    )
    require(
        panel_path.is_file()
        and registered_inputs.get(
            "data/processed/selected_universe_panel_v32.parquet"
        )
        is not None,
        "V59 verifier panel binding is missing",
    )
    result: dict[str, dict[str, Any]] = {}
    for cell_id, context in sorted(cell_contexts.items()):
        panel_filters = [
            [
                (
                    column,
                    operator,
                    list(value) if operator == "in" else value,
                )
                for column, operator, value in conjunction
            ]
            for conjunction in _key_dnf(context["panel_keys"])
        ]
        panel = pd.read_parquet(
            panel_path,
            engine="pyarrow",
            columns=list(PANEL_COLUMNS),
            filters=panel_filters,
        )
        require(
            list(panel.columns) == list(PANEL_COLUMNS)
            and not panel.duplicated(["date", "symbol"]).any(),
            "V59 independent panel projection drift",
        )
        panel["date"] = panel["date"].map(utc_day)
        panel["symbol"] = panel["symbol"].astype(str)
        feature_columns = list(PANEL_COLUMNS[2:])
        panel_values = {
            (row.date, str(row.symbol)): np.asarray(
                [getattr(row, name) for name in feature_columns],
                dtype=np.float64,
            )
            for row in panel.itertuples(index=False)
        }
        require(
            set(panel_values) == set(context["panel_keys"]),
            "V59 independent panel key projection drift",
        )
        dates = pd.date_range(
            context["development_start"],
            context["development_end"],
            freq="D",
            tz="UTC",
        )
        development_keys = set(context["development_keys"])
        starts = context["sequence_start_by_key"]
        valid_window: dict[tuple[pd.Timestamp, str], bool] = {}
        momentum: dict[tuple[pd.Timestamp, str], float] = {}
        for date, symbol in sorted(development_keys):
            start = starts.get((date, symbol))
            if start is None or int((date - start).days) + 1 != 256:
                valid_window[(date, symbol)] = False
                continue
            window_keys = [(day, symbol) for day in pd.date_range(start, date, freq="D")]
            rows = [panel_values.get(key) for key in window_keys]
            valid = all(row is not None for row in rows)
            if valid:
                values = np.stack(rows).astype(np.float64, copy=False)
                valid = values.shape == (256, 8) and np.isfinite(values).all()
            valid_window[(date, symbol)] = bool(valid)
            if valid:
                momentum[(date, symbol)] = float(
                    values[-30:, 0].sum(dtype=np.float64)
                )
        availability: dict[str, np.ndarray] = {}
        momentum_by_triplet: dict[str, np.ndarray] = {}
        unavailable_indexed: list[tuple[int, int, dict[str, Any]]] = []
        for triplet_index, triplet in enumerate(context["test_triplets"]):
            triplet_key = "|".join(triplet)
            active = np.zeros(len(dates), dtype=bool)
            scores = np.full((len(dates), 3), np.nan, dtype=np.float64)
            for day_index, date in enumerate(dates):
                member_keys = [(date, symbol) for symbol in triplet]
                if not all(key in development_keys for key in member_keys):
                    reason = "missing_registered_sequence_member"
                else:
                    member_starts = tuple(starts.get(key) for key in member_keys)
                    if None in member_starts or len(set(member_starts)) != 1:
                        reason = "missing_or_nonshared_sequence_start"
                    elif int((date - member_starts[0]).days) + 1 != 256:
                        reason = "sequence_calendar_length_drift"
                    elif not all(valid_window.get(key, False) for key in member_keys):
                        reason = "missing_or_nonfinite_exact_context"
                    else:
                        reason = ""
                if reason:
                    unavailable_indexed.append(
                        (
                            day_index,
                            triplet_index,
                            {
                                "date": day_text(date),
                                "triplet_key": triplet_key,
                                "reason": reason,
                            },
                        )
                    )
                    continue
                active[day_index] = True
                scores[day_index] = [momentum[key] for key in member_keys]
            availability[triplet_key] = active
            momentum_by_triplet[triplet_key] = scores
        unavailable = [
            row
            for _, _, row in sorted(
                unavailable_indexed, key=lambda item: (item[0], item[1])
            )
        ]
        result[cell_id] = {
            "dates": dates,
            "triplets": tuple("|".join(row) for row in context["test_triplets"]),
            "availability": availability,
            "momentum": momentum_by_triplet,
            "final_raw_by_key": {
                key: panel_values[key]
                for key in sorted(development_keys)
                if key in panel_values
            },
            "unavailable_contexts": unavailable,
            "eligible_triplet_dates": sum(
                int(values.sum()) for values in availability.values()
            ),
        }
    return result


def _linear_recompute_states(
    root: Path,
    scaler_binding: Mapping[str, Any],
    linear_binding: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    scaler_by_cell = {
        str(row["cell_id"]): row for row in scaler_binding.get("entries", [])
    }
    linear_by_cell = {
        str(row["cell_id"]): row for row in linear_binding.get("entries", [])
    }
    require(
        set(scaler_by_cell) == set(linear_by_cell) and len(scaler_by_cell) == 12,
        "V59 linear/scaler recomputation grid drift",
    )
    result: dict[str, dict[str, Any]] = {}
    for cell_id in sorted(scaler_by_cell):
        scaler_entry = scaler_by_cell[cell_id]
        path = resolve_repo_path(root, scaler_entry["path"], "V59 verifier scaler")
        require(
            file_sha256(path) == scaler_entry.get("file_sha256"),
            "V59 verifier scaler byte hash drift",
        )
        wrapper = load_json(path, "V59 verifier scaler")
        scaler = mapping(wrapper.get("scaler"), "V59 verifier scaler payload")
        semantic = deepcopy(scaler)
        registered_semantic = semantic.pop("scaler_sha256", None)
        require(
            registered_semantic == scaler_entry.get("scaler_sha256")
            and canonical_sha256(semantic) == registered_semantic,
            "V59 verifier scaler semantic hash drift",
        )
        mean = np.asarray(scaler.get("mean"), dtype=np.float64)
        scale = np.asarray(scaler.get("standard_deviation"), dtype=np.float64)
        linear = mapping(linear_by_cell[cell_id].get("state"), "V59 linear state")
        coefficient = np.asarray(linear.get("coefficient"), dtype=np.float64)
        require(
            mean.shape == (8,)
            and scale.shape == (8,)
            and coefficient.shape == (9,)
            and np.isfinite(mean).all()
            and np.isfinite(scale).all()
            and (scale > 0).all()
            and np.isfinite(coefficient).all(),
            "V59 verifier scaler/Ridge numeric geometry drift",
        )
        result[cell_id] = {
            "mean": mean,
            "scale": scale,
            "coefficient": coefficient,
            "intercept": float(linear["intercept"]),
            "residual_q20": float(linear["residual_q20"]),
        }
    return result


def _prediction_drivers(
    path: Path,
    states: Mapping[str, Mapping[str, Any]],
    linear_states: Mapping[str, Mapping[str, Any]],
    expected_rows: int,
    target_symbols: set[str],
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    predictions = pd.read_parquet(path, engine="pyarrow")
    require(len(predictions) == expected_rows, "V59 prediction ledger row-count drift")
    numeric_columns = [
        column
        for column in predictions.columns
        if column.startswith(("seed_", "ensemble_", "linear_"))
    ]
    require(
        numeric_columns
        and np.isfinite(
            predictions.loc[:, numeric_columns].to_numpy(dtype=np.float64)
        ).all(),
        "V59 prepared predictions contain a non-finite value",
    )
    for horizon in (1, 3, 7):
        for quantile in (20, 50, 80):
            expected = predictions[
                f"seed_42_h{horizon}_q{quantile}"
            ].to_numpy(dtype=np.float64, copy=True)
            expected += predictions[f"seed_7_h{horizon}_q{quantile}"].to_numpy(
                dtype=np.float64
            )
            expected += predictions[f"seed_123_h{horizon}_q{quantile}"].to_numpy(
                dtype=np.float64
            )
            expected /= 3
            require(
                np.array_equal(
                    expected,
                    predictions[f"ensemble_h{horizon}_q{quantile}"].to_numpy(
                        dtype=np.float64
                    ),
                ),
                "V59 ordered float64 seed aggregation drift",
            )
    primary = [
        "origin",
        "geometry",
        "fold",
        "triplet_key",
        "date",
        "asset_slot",
        "symbol",
    ]
    require(
        not predictions.duplicated(primary).any()
        and len(predictions) % 3 == 0
        and not set(predictions["symbol"].astype(str)).intersection(target_symbols),
        "V59 prediction key duplication, geometry, or target drift",
    )
    predictions["date"] = predictions["date"].map(utc_day)
    slot = predictions["asset_slot"].to_numpy(dtype=np.int64).reshape(-1, 3)
    require(
        np.array_equal(slot, np.tile(np.arange(3), (len(slot), 1))),
        "V59 prediction asset-slot order drift",
    )
    for column in ("origin", "geometry", "fold", "triplet_key", "date"):
        values = predictions[column].to_numpy().reshape(-1, 3)
        require(
            np.array_equal(values[:, 0], values[:, 1])
            and np.array_equal(values[:, 0], values[:, 2]),
            f"V59 prediction triplet grouping drift: {column}",
        )
    triplets = predictions["triplet_key"].astype(str).iloc[::3].tolist()
    expected_symbols = np.asarray(
        [symbol for triplet in triplets for symbol in triplet.split("|")],
        dtype=object,
    )
    require(
        all(parts == sorted(set(parts)) and len(parts) == 3 for parts in map(lambda value: value.split("|"), triplets))
        and np.array_equal(
            predictions["symbol"].astype(str).to_numpy(), expected_symbols
        ),
        "V59 prediction lexical triplet/symbol drift",
    )
    residual_by_cell = {
        cell_id: float(state["residual_q20"])
        for cell_id, state in linear_states.items()
    }
    for cell_id, rows in predictions.groupby(
        ["origin", "geometry", "fold"], sort=False
    ):
        key = f"{cell_id[0]}|{cell_id[1]}|{int(cell_id[2])}"
        expected_q20 = rows["linear_h7_q50"].to_numpy(dtype=np.float64, copy=True)
        expected_q20 += residual_by_cell[key]
        require(
            np.array_equal(
                expected_q20, rows["linear_h7_q20"].to_numpy(dtype=np.float64)
            ),
            "V59 linear q20 residual arithmetic drift",
        )
    drivers: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    grouped = predictions.groupby(
        ["origin", "geometry", "fold", "triplet_key", "date"], sort=False
    )
    for key, rows in grouped:
        require(len(rows) == 3, "V59 prediction key does not have three asset rows")
        cell_id = f"{key[0]}|{key[1]}|{int(key[2])}"
        triplet_key = str(key[3])
        state = states.get(cell_id)
        require(
            state is not None and triplet_key in state["availability"],
            "V59 prediction contains an unregistered cell or triplet",
        )
        linear_state = linear_states.get(cell_id)
        require(linear_state is not None, "V59 linear recomputation state is missing")
        triplet = tuple(triplet_key.split("|"))
        date = utc_day(key[4])
        raw_rows = [state["final_raw_by_key"].get((date, symbol)) for symbol in triplet]
        require(
            all(row is not None for row in raw_rows),
            "V59 linear recomputation raw final features are missing",
        )
        raw = np.stack(raw_rows).astype(np.float64, copy=False)
        mean = np.asarray(linear_state["mean"], dtype=np.float64)
        scale = np.asarray(linear_state["scale"], dtype=np.float64)
        features = np.empty((3, 9), dtype=np.float64)
        features[:, :8] = (raw - mean) / scale
        features[:, 8] = (raw[:, 1] - raw[:, 1].mean(dtype=np.float64)) / scale[1]
        # The frozen tensor contract converts the complete transformed vector to
        # float32 before the Ridge receives the final timestep, then Ridge
        # promotes that exact quantized vector back to float64 internally.
        features = features.astype(np.float32).astype(np.float64)
        expected_linear_q50 = (
            features @ np.asarray(linear_state["coefficient"], dtype=np.float64)
            + float(linear_state["intercept"])
        )
        require(
            np.allclose(
                expected_linear_q50,
                rows["linear_h7_q50"].to_numpy(dtype=np.float64),
                rtol=0.0,
                atol=5.0e-15,
            ),
            "V59 linear q50 differs from frozen scaler and fitted Ridge state",
        )
        episode_key = (cell_id, triplet_key)
        if episode_key not in drivers:
            dates = state["dates"]
            drivers[episode_key] = {
                "candidate": np.full((len(dates), 3), np.nan, dtype=np.float64),
                "linear": np.full((len(dates), 3), np.nan, dtype=np.float64),
                "present": np.zeros(len(dates), dtype=bool),
            }
        day = int((date - state["dates"][0]).days)
        require(
            0 <= day < len(state["dates"])
            and not drivers[episode_key]["present"][day],
            "V59 prediction date duplication or range drift",
        )
        drivers[episode_key]["present"][day] = True
        drivers[episode_key]["candidate"][day] = rows[
            "ensemble_h7_q20"
        ].to_numpy(dtype=np.float64)
        drivers[episode_key]["linear"][day] = rows["linear_h7_q20"].to_numpy(
            dtype=np.float64
        )
    for cell_id, state in states.items():
        for triplet_key, eligible in state["availability"].items():
            values = drivers.get((cell_id, triplet_key))
            if values is None:
                require(not eligible.any(), "V59 eligible prediction episode is missing")
                dates = state["dates"]
                drivers[(cell_id, triplet_key)] = {
                    "candidate": np.full((len(dates), 3), np.nan, dtype=np.float64),
                    "linear": np.full((len(dates), 3), np.nan, dtype=np.float64),
                    "present": np.zeros(len(dates), dtype=bool),
                }
                continue
            require(
                np.array_equal(values["present"], eligible),
                "V59 prediction presence differs from independently derived availability",
            )
    return drivers


def _verify_episode_rows(
    rows: pd.DataFrame,
    dates: pd.DatetimeIndex,
    triplet_key: str,
    eligible: np.ndarray,
    expected: Mapping[str, np.ndarray],
) -> None:
    values = rows.sort_values("date").reset_index(drop=True)
    values["date"] = values["date"].map(utc_day)
    require(
        len(values) == len(dates)
        and np.array_equal(values["date"].to_numpy(), dates.to_numpy())
        and np.array_equal(values["available"].to_numpy(dtype=bool), eligible),
        "V59 position episode calendar or availability drift",
    )
    triplet = tuple(triplet_key.split("|"))
    require(
        len(triplet) == 3
        and all(
            (values[f"symbol_{slot}"].astype(str) == triplet[slot]).all()
            for slot in range(3)
        ),
        "V59 position episode symbol drift",
    )
    weights = values[["weight_0", "weight_1", "weight_2"]].to_numpy(
        dtype=np.float64
    )
    expected_weights = np.asarray(expected["weights"], dtype=np.float64)
    require(
        np.array_equal(weights, expected_weights)
        and np.array_equal(
            values["decision"].to_numpy(dtype=bool), expected["decision"]
        )
        and np.array_equal(
            values["forced_cash"].to_numpy(dtype=bool), expected["forced"]
        ),
        "V59 position policy, decision clock, or forced cash drift",
    )
    previous = np.zeros(3, dtype=np.float64)
    base = np.zeros(len(weights), dtype=np.float64)
    for day, weight in enumerate(weights):
        base[day] = float(np.abs(weight - previous).sum())
        previous = weight
    liquidation = np.zeros(len(weights), dtype=np.float64)
    liquidation[-1] = float(np.abs(weights[-1]).sum())
    turnover = base + liquidation
    post = weights.copy()
    post[-1] = 0.0
    require(
        np.array_equal(values["base_turnover"].to_numpy(dtype=np.float64), base)
        and np.array_equal(
            values["final_liquidation_turnover"].to_numpy(dtype=np.float64),
            liquidation,
        )
        and np.array_equal(
            values["turnover"].to_numpy(dtype=np.float64), turnover
        )
        and np.array_equal(
            values[
                [
                    "post_event_weight_0",
                    "post_event_weight_1",
                    "post_event_weight_2",
                ]
            ].to_numpy(dtype=np.float64),
            post,
        )
        and np.array_equal(
            values["final_liquidation"].to_numpy(dtype=bool),
            np.arange(len(values)) == len(values) - 1,
        ),
        "V59 position turnover or final liquidation drift",
    )
    expected_actions: list[str] = []
    expected_selected: list[str] = []
    for weight in weights:
        active = np.flatnonzero(weight > 0)
        if len(active) == 0:
            expected_actions.append("cash")
            expected_selected.append("")
        elif len(active) == 1:
            expected_actions.append("long_one_asset")
            expected_selected.append(triplet[int(active[0])])
        else:
            expected_actions.append("equal_weight")
            expected_selected.append("")
    require(
        values["action"].astype(str).tolist() == expected_actions
        and values["selected_symbol"].fillna("").astype(str).tolist()
        == expected_selected,
        "V59 position action or selected-symbol drift",
    )


def _verify_position_values(
    output_dir: Path,
    states: Mapping[str, Mapping[str, Any]],
    drivers: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
    ledger: Mapping[str, Any],
) -> dict[str, int]:
    candidates = pd.read_parquet(output_dir / "candidate_positions.parquet")
    require(
        len(candidates) == ledger.get("candidate_position_rows")
        and not candidates.duplicated(
            ["origin", "geometry", "fold", "triplet_key", "date"]
        ).any(),
        "V59 candidate position key or row-count drift",
    )
    expected_episode_count = sum(len(state["triplets"]) for state in states.values())
    candidate_episodes = 0
    for key, rows in candidates.groupby(
        ["origin", "geometry", "fold", "triplet_key"], sort=False
    ):
        cell_id = f"{key[0]}|{key[1]}|{int(key[2])}"
        triplet_key = str(key[3])
        state = states.get(cell_id)
        driver = drivers.get((cell_id, triplet_key))
        require(
            state is not None
            and triplet_key in state["availability"]
            and driver is not None,
            "V59 candidate contains an unregistered episode",
        )
        eligible = state["availability"][triplet_key]
        _verify_episode_rows(
            rows,
            state["dates"],
            triplet_key,
            eligible,
            _state_policy(driver["candidate"], eligible),
        )
        candidate_episodes += 1
    require(
        candidate_episodes == expected_episode_count,
        "V59 candidate episode cardinality drift",
    )
    control_names = (
        "cash",
        "weekly_dual_momentum_30",
        "weekly_equal_weight_total_gross_one_third",
        "shared_linear_h7_q50_with_train_residual_q20",
    )
    control_rows = 0
    control_episodes = 0
    for control in control_names:
        controls = pd.read_parquet(
            output_dir / "control_positions.parquet",
            filters=[("control", "==", control)],
        )
        control_rows += len(controls)
        require(
            not controls.duplicated(
                ["origin", "geometry", "fold", "triplet_key", "date", "control"]
            ).any()
            and (controls["control"].astype(str) == control).all(),
            "V59 control position key or name drift",
        )
        episodes = 0
        for key, rows in controls.groupby(
            ["origin", "geometry", "fold", "triplet_key"], sort=False
        ):
            cell_id = f"{key[0]}|{key[1]}|{int(key[2])}"
            triplet_key = str(key[3])
            state = states.get(cell_id)
            driver = drivers.get((cell_id, triplet_key))
            require(
                state is not None
                and triplet_key in state["availability"]
                and driver is not None,
                "V59 control contains an unregistered episode",
            )
            eligible = state["availability"][triplet_key]
            if control == "cash":
                expected = _cash_policy(eligible)
            elif control == "weekly_dual_momentum_30":
                expected = _momentum_policy(
                    state["momentum"][triplet_key], eligible
                )
            elif control == "weekly_equal_weight_total_gross_one_third":
                expected = _equal_policy(eligible)
            else:
                expected = _state_policy(driver["linear"], eligible)
            _verify_episode_rows(
                rows,
                state["dates"],
                triplet_key,
                eligible,
                expected,
            )
            episodes += 1
        require(
            episodes == expected_episode_count,
            f"V59 {control} episode cardinality drift",
        )
        control_episodes += episodes
    require(
        control_rows == ledger.get("control_position_rows"),
        "V59 control position ledger row-count drift",
    )
    return {
        "candidate_episode_count": candidate_episodes,
        "control_episode_count": control_episodes,
    }


def verify_prepared_values(
    root: Path,
    output_dir: Path,
    phase: Mapping[str, Any],
    scaler_binding: Mapping[str, Any],
    linear_binding: Mapping[str, Any],
    ledger: Mapping[str, Any],
    cell_contexts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute all prepared-value gates without opening outcomes or targets."""

    registered_inputs = phase["input_contract"]["expected_file_sha256_by_path"]
    states = _registered_behavior_state(root, registered_inputs, cell_contexts)
    linear_states = _linear_recompute_states(root, scaler_binding, linear_binding)
    drivers = _prediction_drivers(
        output_dir / "predictions.parquet",
        states,
        linear_states,
        int(ledger["prediction_rows"]),
        set(map(str, phase["target_contract"]["symbols"])),
    )
    episodes = _verify_position_values(output_dir, states, drivers, ledger)
    diagnostics = {
        cell_id: {
            "eligible_triplet_dates": int(state["eligible_triplet_dates"]),
            "unavailable_context_count": len(state["unavailable_contexts"]),
            "unavailable_context_sha256": canonical_sha256(
                state["unavailable_contexts"]
            ),
        }
        for cell_id, state in sorted(states.items())
    }
    body = {
        "schema_version": "v59-independent-preoutcome-verification/v1",
        "passed": True,
        "prediction_rows": int(ledger["prediction_rows"]),
        "candidate_position_rows": int(ledger["candidate_position_rows"]),
        "control_position_rows": int(ledger["control_position_rows"]),
        **episodes,
        "cell_diagnostics": diagnostics,
        "development_outcome_value_reads": 0,
        "target_asset_loads": 0,
        "performance_metrics_computed": 0,
        "pnl_evaluations": 0,
    }
    return with_self_hash(body, "independent_verification_sha256")


def verify_prepare_packet(
    root: Path,
    output_dir: Path,
    *,
    contract: Mapping[str, Any] | None = None,
    enforce_live_git: bool = True,
    enforce_live_inputs: bool = True,
    verify_prepared_values_gate: bool = True,
    verify_source_commit: bool = True,
    allow_post_prepare_files: bool = False,
) -> dict[str, Any]:
    """Validate a complete V59 prepare packet without reading Parquet values."""

    require(output_dir.is_dir(), "V59 prepare packet directory is missing")
    for name in REQUIRED_PREPARE_FILES:
        require((output_dir / name).is_file(), f"V59 packet is missing {name}")
    if not allow_post_prepare_files:
        require(
            not any((output_dir / name).exists() for name in FORBIDDEN_PREPARE_FILES),
            "V59 prepare packet contains a post-authorization or outcome artifact",
        )
    phase = (
        dict(contract)
        if contract is not None
        else load_live_v59_contract(
            root,
            state_path="research/current.yaml",
            phase_contract_path="research/phase_contracts/v059.yaml",
        )[0]
    )
    phase_path = resolve_repo_path(
        root, "research/phase_contracts/v059.yaml", "live V59 phase contract"
    )
    require(
        phase_path.is_file()
        and file_sha256(phase_path) == V59_PHASE_FILE_SHA256
        and canonical_sha256(phase) == V59_PHASE_CONTRACT_CANONICAL_SHA256,
        "injected or live V59 phase contract hash drift",
    )
    projection = registered_projection(phase)
    projection_sha256 = canonical_sha256(projection)

    spec = load_json(output_dir / "evaluation_spec.json", "evaluation spec")
    source = load_json(output_dir / "source_receipt.json", "source receipt")
    input_receipt = load_json(
        output_dir / "input_hash_receipt.json", "input hash receipt"
    )
    checkpoint_binding = load_json(
        output_dir / "checkpoint_binding.json", "checkpoint binding"
    )
    scaler_binding = load_json(output_dir / "scaler_binding.json", "scaler binding")
    linear_binding = load_json(
        output_dir / "linear_control_receipt.json", "linear control receipt"
    )
    manifest = load_json(output_dir / "prepare_manifest.json", "prepare manifest")
    receipt = load_json(output_dir / "prepare_receipt.json", "prepare receipt")
    behavior = load_json(output_dir / "behavior_audit.json", "behavior audit")
    outcome_request = load_json(output_dir / "outcome_request.json", "outcome request")
    verify_self_hash(spec, "evaluation_spec_sha256", "evaluation spec")
    verify_self_hash(source, "source_receipt_sha256", "source receipt")
    verify_self_hash(
        input_receipt, "input_hash_receipt_sha256", "input hash receipt"
    )
    verify_self_hash(
        checkpoint_binding, "checkpoint_binding_sha256", "checkpoint binding"
    )
    verify_self_hash(scaler_binding, "scaler_binding_sha256", "scaler binding")
    verify_self_hash(
        linear_binding, "linear_control_receipt_sha256", "linear control receipt"
    )
    verify_self_hash(manifest, "prepare_manifest_sha256", "prepare manifest")
    verify_self_hash(receipt, "prepare_receipt_sha256", "prepare receipt")
    verify_self_hash(behavior, "behavior_audit_sha256", "behavior audit")
    verify_self_hash(outcome_request, "outcome_request_sha256", "outcome request")
    require(
        spec.get("phase_contract_file_sha256") == V59_PHASE_FILE_SHA256
        and spec.get("phase_contract_canonical_sha256")
        == V59_PHASE_CONTRACT_CANONICAL_SHA256
        and spec.get("registered_projection") == projection
        and spec.get("registered_projection_sha256") == projection_sha256,
        "evaluation spec does not bind the live V59 projection",
    )
    source_files = source.get("files")
    require(
        source.get("schema_version") == "v59-source-receipt/v1"
        and source.get("git_clean") is True
        and isinstance(source_files, dict)
        and set(source_files) == set(V59_SOURCE_FILES)
        and source.get("bundle_sha256") == canonical_sha256(source_files),
        "V59 source receipt structure drift",
    )
    source_head = str(source.get("git_head", ""))
    require(
        GIT_HEAD.fullmatch(source_head) is not None,
        "V59 source receipt Git head drift",
    )
    if verify_source_commit:
        for relative, expected in source_files.items():
            blob = _git(root, "show", f"{source_head}:{relative}", binary=True)
            assert isinstance(blob, bytes)
            require(
                hashlib.sha256(blob).hexdigest() == expected,
                f"V59 source receipt differs from its committed source: {relative}",
            )
    if enforce_live_git:
        live_head = str(_git(root, "rev-parse", "HEAD"))
        require(
            source.get("git_head") == live_head
            and _git(root, "status", "--porcelain", "--untracked-files=all") == "",
            "V59 source receipt does not bind the live clean Git head",
        )
        for relative, expected in source_files.items():
            path = resolve_repo_path(root, relative, f"source receipt {relative}")
            require(
                path.is_file() and file_sha256(path) == expected,
                f"V59 live source receipt file drift: {relative}",
            )
    registered_inputs = phase["input_contract"]["expected_file_sha256_by_path"]
    require(
        input_receipt.get("schema_version") == "v59-input-hash-receipt/v1"
        and input_receipt.get("files") == registered_inputs
        and input_receipt.get("file_count") == len(registered_inputs)
        and input_receipt.get("development_outcome_value_reads") == 0
        and input_receipt.get("target_asset_loads") == 0,
        "V59 input receipt content drift",
    )
    if enforce_live_inputs:
        require(
            verify_input_files(root, phase) == input_receipt,
            "V59 input receipt does not match live registered bytes",
        )
    require(
        spec.get("source_receipt_sha256") == source.get("source_receipt_sha256")
        and spec.get("input_hash_receipt_sha256")
        == input_receipt.get("input_hash_receipt_sha256"),
        "V59 evaluation spec source/input receipt binding drift",
    )
    (
        expected_access_receipts,
        expected_outcome_keys,
        registered_cell_contexts,
    ) = _registered_access_receipts(root, phase, registered_inputs)
    checkpoint_entries = checkpoint_binding.get("entries")
    require(
        checkpoint_binding.get("schema_version") == "v59-checkpoint-binding/v1"
        and checkpoint_binding.get("entry_count") == 36
        and isinstance(checkpoint_entries, list)
        and len(checkpoint_entries) == 36,
        "V59 checkpoint binding cardinality drift",
    )
    expected_grid = {
        (origin, geometry, fold, seed)
        for origin in ("origin_2024", "origin_2025")
        for geometry in ("expanding", "rolling")
        for fold in (1, 2, 3)
        for seed in (42, 7, 123)
    }
    observed_grid = {
        (
            row.get("origin"),
            row.get("geometry"),
            row.get("fold"),
            row.get("seed"),
        )
        for row in checkpoint_entries
        if isinstance(row, dict)
    }
    require(
        observed_grid == expected_grid
        and all(
            row.get("load_count") == 1
            and row.get("selected") is False
            and row.get("weight") is None
            and row.get("checkpoint_state") == "best_model_state"
            and row.get("optimizer_steps") == 0
            for row in checkpoint_entries
        ),
        "V59 checkpoint binding behavior drift",
    )
    checkpoint_manifest_path = resolve_repo_path(
        root,
        "artifacts/v58_state_conditioned_multi_horizon_training/checkpoint_manifest.json",
        "V58 checkpoint manifest",
    )
    require(
        file_sha256(checkpoint_manifest_path)
        == registered_inputs.get(
            "artifacts/v58_state_conditioned_multi_horizon_training/checkpoint_manifest.json"
        ),
        "V58 checkpoint manifest file hash drift",
    )
    checkpoint_manifest = load_json(checkpoint_manifest_path, "V58 checkpoint manifest")
    manifest_jobs = {
        (
            row["origin"],
            row["geometry"],
            int(row["fold"]),
            int(row["seed"]),
        ): row
        for row in checkpoint_manifest.get("jobs", [])
    }
    require(set(manifest_jobs) == expected_grid, "V58 checkpoint manifest grid drift")
    for entry in checkpoint_entries:
        key = (
            entry["origin"],
            entry["geometry"],
            int(entry["fold"]),
            int(entry["seed"]),
        )
        registered = manifest_jobs[key]
        require(
            entry.get("job_id") == registered.get("job_id")
            and entry.get("path") == registered.get("checkpoint_path")
            and entry.get("checkpoint_sha256") == registered.get("checkpoint_sha256")
            and entry.get("semantic_checkpoint_sha256")
            == registered.get("semantic_checkpoint_sha256")
            and entry.get("best_model_state_sha256")
            == registered.get("best_model_state_sha256"),
            "V59 checkpoint binding differs from the frozen V58 manifest",
        )
    scaler_entries = scaler_binding.get("entries")
    expected_cells = {
        f"{origin}|{geometry}|{fold}"
        for origin in ("origin_2024", "origin_2025")
        for geometry in ("expanding", "rolling")
        for fold in (1, 2, 3)
    }
    require(
        scaler_binding.get("schema_version") == "v59-scaler-binding/v1"
        and scaler_binding.get("entry_count") == 12
        and isinstance(scaler_entries, list)
        and {row.get("cell_id") for row in scaler_entries} == expected_cells
        and all(
            row.get("load_count") == 1 and row.get("fit_refit_count") == 0
            for row in scaler_entries
        ),
        "V59 scaler binding behavior drift",
    )
    scaler_manifest_path = resolve_repo_path(
        root,
        "artifacts/v58_state_conditioned_multi_horizon_training/scaler_manifest.json",
        "V58 scaler manifest",
    )
    require(
        file_sha256(scaler_manifest_path)
        == registered_inputs.get(
            "artifacts/v58_state_conditioned_multi_horizon_training/scaler_manifest.json"
        ),
        "V58 scaler manifest file hash drift",
    )
    scaler_manifest = load_json(scaler_manifest_path, "V58 scaler manifest")
    manifest_scalers = {
        f"{row['origin']}|{row['geometry']}|{int(row['fold'])}": row
        for row in scaler_manifest.get("scalers", [])
    }
    require(set(manifest_scalers) == expected_cells, "V58 scaler manifest grid drift")
    expected_scaler_files = phase["input_contract"][
        "expected_scaler_file_sha256_by_path"
    ]
    for entry in scaler_entries:
        cell_id = entry["cell_id"]
        origin, geometry, fold = cell_id.split("|")
        expected_path = (
            "data/checkpoints/v58_state_conditioned_multi_horizon_training/"
            f"{origin}/{geometry}/fold_{int(fold)}/scaler.json"
        )
        registered = manifest_scalers[cell_id]
        require(
            entry.get("path") == expected_path
            and entry.get("file_sha256") == expected_scaler_files.get(expected_path)
            and entry.get("scaler_sha256") == registered.get("scaler_sha256")
            and entry.get("fit_symbols") == registered.get("fit_symbols"),
            "V59 scaler binding differs from the frozen V58 manifest",
        )
    for entry in checkpoint_entries:
        cell_id = f"{entry['origin']}|{entry['geometry']}|{int(entry['fold'])}"
        require(
            manifest_jobs[
                (
                    entry["origin"],
                    entry["geometry"],
                    int(entry["fold"]),
                    int(entry["seed"]),
                )
            ]["scaler_sha256"]
            == manifest_scalers[cell_id]["scaler_sha256"],
            "V59 checkpoint/scaler cross-binding drift",
        )
    linear_entries = linear_binding.get("entries")
    require(
        linear_binding.get("schema_version") == "v59-linear-control-receipt/v1"
        and linear_binding.get("entry_count") == 12
        and isinstance(linear_entries, list)
        and {row.get("cell_id") for row in linear_entries} == expected_cells
        and all(
            row.get("fit_scope") == "exact_origin_geometry_fold_train_role_only"
            and row.get("validation_or_development_fit_rows") == 0
            and row.get("development_outcome_value_reads") == 0
            for row in linear_entries
        ),
        "V59 linear-control receipt behavior drift",
    )
    for row in linear_entries:
        state = row.get("state")
        require(
            isinstance(state, dict)
            and isinstance(state.get("coefficient"), list)
            and len(state["coefficient"]) == 9
            and all(isinstance(value, (int, float)) for value in state["coefficient"])
            and isinstance(state.get("intercept"), (int, float))
            and isinstance(state.get("residual_q20"), (int, float))
            and row.get("state_sha256") == canonical_sha256(state),
            "V59 linear-control fitted-state binding drift",
        )
    manifest_rows = _manifest_map(manifest)
    require(
        set(manifest_rows) == set(MANIFEST_PAYLOAD_FILES),
        "prepare manifest file set drift",
    )
    for name, row in manifest_rows.items():
        path = output_dir / name
        require(
            row.get("size_bytes") == path.stat().st_size
            and row.get("file_sha256") == file_sha256(path),
            f"prepare manifest hash/size drift: {name}",
        )
    expected_parquet_columns = {
        "predictions.parquet": spec.get("prediction_schema", {}).get("columns"),
        "candidate_positions.parquet": spec.get("position_schema", {}).get(
            "candidate_columns"
        ),
        "control_positions.parquet": spec.get("position_schema", {}).get(
            "control_columns"
        ),
    }
    for name, expected_columns in expected_parquet_columns.items():
        require(
            isinstance(expected_columns, list) and expected_columns,
            f"V59 evaluation spec lacks {name} columns",
        )
        parquet = pq.ParquetFile(output_dir / name)
        arrow_schema = parquet.schema_arrow
        schema_records = [
            {
                "name": field.name,
                "type": str(field.type),
                "nullable": field.nullable,
            }
            for field in arrow_schema
        ]
        row = manifest_rows[name]
        require(
            arrow_schema.names == expected_columns
            and row.get("row_count") == parquet.metadata.num_rows
            and row.get("columns") == expected_columns
            and row.get("arrow_schema") == schema_records
            and row.get("arrow_schema_sha256") == canonical_sha256(schema_records),
            f"V59 {name} footer/schema manifest drift",
        )
    behavior_checks = behavior.get("checks")
    registered_behavior_gates = set(phase["outcome_blind_gate_contract"]["gates"])
    ledger = behavior.get("operation_ledger")
    require(
        behavior.get("passed") is True
        and isinstance(behavior_checks, dict)
        and registered_behavior_gates.issubset(behavior_checks)
        and all(behavior_checks[name] is True for name in registered_behavior_gates)
        and isinstance(ledger, dict)
        and ledger.get("checkpoint_loads") == 36
        and ledger.get("scaler_loads") == 12
        and ledger.get("linear_control_fits") == 12
        and ledger.get("optimizer_steps") == 0
        and ledger.get("development_outcome_value_reads") == 0
        and ledger.get("target_asset_loads") == 0
        and ledger.get("performance_metrics_computed") == 0
        and ledger.get("pnl_evaluations") == 0,
        "V59 behavior audit did not pass exact registered gates",
    )
    require(
        manifest_rows["predictions.parquet"].get("row_count")
        == ledger.get("prediction_rows")
        and manifest_rows["candidate_positions.parquet"].get("row_count")
        == ledger.get("candidate_position_rows")
        and manifest_rows["control_positions.parquet"].get("row_count")
        == ledger.get("control_position_rows"),
        "V59 Parquet row counts differ from the behavior ledger",
    )
    if verify_prepared_values_gate:
        independent_verification = verify_prepared_values(
            root,
            output_dir,
            phase,
            scaler_binding,
            linear_binding,
            ledger,
            registered_cell_contexts,
        )
        require(
            behavior.get("independent_preoutcome_verification")
            == independent_verification,
            "V59 persisted independent pre-outcome verification drift",
        )
    request_keys = outcome_request.get("keys")
    require(isinstance(request_keys, list) and request_keys, "V59 outcome request is empty")
    folds_path = resolve_repo_path(
        root,
        "artifacts/v32_selected_universe_dataset/asset_folds.json",
        "V32 asset folds",
    )
    require(
        file_sha256(folds_path)
        == registered_inputs.get(
            "artifacts/v32_selected_universe_dataset/asset_folds.json"
        ),
        "V32 asset-fold file hash drift",
    )
    folds = load_json(folds_path, "V32 asset folds")
    test_symbols_by_fold = {
        int(row["fold"]): frozenset(map(str, row["test_symbols"]))
        for row in folds.get("folds", [])
    }
    require(
        set(test_symbols_by_fold) == {1, 2, 3}
        and all(len(symbols) == 10 for symbols in test_symbols_by_fold.values()),
        "V32 test-symbol fold geometry drift",
    )
    request_primary = [
        (row.get("origin"), row.get("fold"), row.get("date"), row.get("symbol"))
        for row in request_keys
        if isinstance(row, dict)
    ]
    require(
        len(request_primary) == len(request_keys)
        and request_primary == sorted(set(request_primary))
        and request_primary
        == [
            (origin, fold, date, symbol)
            for (origin, fold), records in sorted(expected_outcome_keys.items())
            for date, symbol in records
        ]
        and outcome_request.get("key_count") == len(request_primary)
        and outcome_request.get("key_sha256") == canonical_sha256(request_primary)
        and all(
            int(row[1]) in test_symbols_by_fold
            and str(row[3]) in test_symbols_by_fold[int(row[1])]
            for row in request_primary
        )
        and not {row[3] for row in request_primary}.intersection(
            phase["target_contract"]["symbols"]
        ),
        "V59 outcome request key-set drift",
    )
    groups = outcome_request.get("groups")
    require(
        isinstance(groups, list)
        and outcome_request.get("group_count") == 6
        and len(groups) == 6,
        "V59 outcome request group cardinality drift",
    )
    grouped_primary: dict[tuple[str, int], list[tuple[str, int, str, str]]] = {}
    for row in request_primary:
        grouped_primary.setdefault((str(row[0]), int(row[1])), []).append(
            (str(row[0]), int(row[1]), str(row[2]), str(row[3]))
        )
    group_map = {
        (str(row.get("origin")), int(row.get("fold", -1))): row
        for row in groups
        if isinstance(row, dict)
    }
    require(set(group_map) == set(grouped_primary), "V59 outcome request groups drift")
    for key, rows in grouped_primary.items():
        group = group_map[key]
        sequence_records = [[row[2], row[3]] for row in rows]
        require(
            group.get("key_count") == len(rows)
            and group.get("key_sha256") == canonical_sha256(rows)
            and group.get("development_sequence_key_sha256")
            == canonical_sha256(sequence_records),
            "V59 outcome request grouped key hash drift",
        )
    access_receipts = behavior.get("data_access_receipts")
    require(
        isinstance(access_receipts, list) and len(access_receipts) == 12,
        "V59 behavior audit data-access receipt count drift",
    )
    observed_access_groups: dict[tuple[str, int], tuple[int, str]] = {}
    for row in access_receipts:
        require(isinstance(row, dict), "V59 data-access receipt row drift")
        verify_self_hash(row, "access_receipt_sha256", "V59 data-access receipt")
        parts = str(row.get("cell_id", "")).split("|")
        require(len(parts) == 3, "V59 data-access cell identity drift")
        receipt_body = deepcopy(row)
        receipt_body.pop("access_receipt_sha256", None)
        require(
            receipt_body == expected_access_receipts.get(str(row.get("cell_id"))),
            "V59 data-access projection, predicate, role, or key receipt drift",
        )
        key = (parts[0], int(parts[2]))
        value = (
            int(row.get("development_sequence_key_count", -1)),
            str(row.get("development_sequence_key_sha256", "")),
        )
        require(
            key not in observed_access_groups or observed_access_groups[key] == value,
            "V59 geometry development-key receipt mismatch",
        )
        observed_access_groups[key] = value
    require(
        set(observed_access_groups) == set(group_map)
        and all(
            group_map[key].get("key_count") == value[0]
            and group_map[key].get("development_sequence_key_sha256") == value[1]
            for key, value in observed_access_groups.items()
        ),
        "V59 outcome request is not bound to exact development sequence receipts",
    )
    replay_binding = receipt.get("cached_replay_binding")
    replay_files = (
        replay_binding.get("files") if isinstance(replay_binding, dict) else None
    )
    expected_replay_names = set(REQUIRED_PREPARE_FILES) - {"prepare_receipt.json"}
    require(
        isinstance(replay_binding, dict)
        and replay_binding.get("stage")
        == "hidden_staging_before_atomic_publish"
        and isinstance(replay_files, dict)
        and set(replay_files) == expected_replay_names
        and replay_binding.get("file_count") == len(expected_replay_names)
        and replay_binding.get("file_hash_map_sha256")
        == canonical_sha256(replay_files)
        and all(
            replay_files[name] == file_sha256(output_dir / name)
            for name in expected_replay_names
        )
        and all(
            replay_binding.get(name) == 0
            for name in (
                "new_checkpoint_loads",
                "new_inference",
                "new_linear_control_fits",
                "new_position_generation",
                "new_outcome_reads",
                "files_rewritten",
            )
        ),
        "V59 cached replay hash binding or zero-work evidence drift",
    )
    require(
        receipt.get("schema_version") == PREPARE_SCHEMA
        and receipt.get("decision") == PREPARE_DECISION
        and receipt.get("pass_authorizes_unseal") is False
        and receipt.get("eligible_to_request_explicit_authorization") is True
        and receipt.get("authorization_state")
        == "awaiting_explicit_user_authorization"
        and receipt.get("next_action") == PREPARE_DECISION
        and receipt.get("required_stage_revision") == "v059_unseal_r1"
        and receipt.get("phase_contract_file_sha256") == V59_PHASE_FILE_SHA256
        and receipt.get("phase_contract_canonical_sha256")
        == V59_PHASE_CONTRACT_CANONICAL_SHA256
        and receipt.get("registered_projection_sha256") == projection_sha256
        and receipt.get("prepare_git_head") == source.get("git_head")
        and receipt.get("evaluation_spec_file_sha256")
        == file_sha256(output_dir / "evaluation_spec.json")
        and receipt.get("source_receipt_file_sha256")
        == file_sha256(output_dir / "source_receipt.json")
        and receipt.get("prepare_manifest_file_sha256")
        == file_sha256(output_dir / "prepare_manifest.json")
        and receipt.get("outcome_request_file_sha256")
        == file_sha256(output_dir / "outcome_request.json")
        and receipt.get("behavior_audit_file_sha256")
        == file_sha256(output_dir / "behavior_audit.json")
        and receipt.get("development_outcome_value_reads") == 0
        and receipt.get("target_asset_loads") == 0
        and receipt.get("performance_metrics_computed") == 0
        and receipt.get("pnl_evaluations") == 0
        and receipt.get("outcome_packet_created") is False
        and receipt.get("authorization_receipt_created") is False,
        "V59 prepare v2 receipt boundary drift",
    )
    require(
        outcome_request.get("allowed_columns")
        == phase["one_shot_contract"]["unseal"]["allowed_columns"],
        "V59 outcome request column drift",
    )
    return {
        "passed": True,
        "decision": PREPARE_DECISION,
        "evaluation_spec_sha256": spec["evaluation_spec_sha256"],
        "evaluation_spec_file_sha256": file_sha256(
            output_dir / "evaluation_spec.json"
        ),
        "prepare_manifest_sha256": manifest["prepare_manifest_sha256"],
        "prepare_manifest_file_sha256": file_sha256(
            output_dir / "prepare_manifest.json"
        ),
        "prepare_receipt_sha256": receipt["prepare_receipt_sha256"],
        "prepare_receipt_file_sha256": file_sha256(output_dir / "prepare_receipt.json"),
        "outcome_request_sha256": outcome_request["outcome_request_sha256"],
        "outcome_request_file_sha256": file_sha256(output_dir / "outcome_request.json"),
        "pass_authorizes_unseal": False,
    }
