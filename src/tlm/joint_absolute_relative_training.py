from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, fields
import fcntl
import gc
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import platform
import subprocess
from typing import Callable, Iterator

import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml

from .joint_absolute_relative_model import (
    JOINT_HEADS,
    JointAbsoluteRelativeTransformer,
    joint_absolute_relative_loss,
)
from .joint_absolute_relative_spec import (
    _canonical_sha256,
    _load_json,
    _sha256_file,
)
from .non_target_pretraining import TripletTensorStore
from .ranking_excess_pretraining import (
    TARGET_SYMBOLS,
    _atomic_torch_save,
    _availability_from_index,
    _cpu_state_dict,
    _move_optimizer_state,
    _optimizer_contract,
    _restore_rng_state,
    _rng_state,
    _seed_device,
    _semantic_state_sha256,
    _state_is_finite,
    _to_cpu,
    _validate_optimizer_resume_state,
)
from .scientific_harness import FeatureScaler
from .supervised_non_target import model_state_sha256


METADATA_INPUTS = {
    "v47_result",
    "v47_blueprint",
    "v47_audit",
    "v48_result",
    "v48_harness_spec",
    "v48_audit",
    "v32_dataset_manifest",
    "v32_feature_schema",
    "v32_asset_folds",
    "v32_triplet_catalog",
}
BINARY_INPUTS = {"panel", "sequence_index"}


@dataclass
class V49EarlyStopping:
    patience: int
    best_loss: float = math.inf
    best_epoch: int = 0
    stale_epochs: int = 0
    should_stop: bool = False

    def update(self, epoch: int, loss: float) -> bool:
        if not math.isfinite(loss):
            raise RuntimeError("V49 validation total loss must be finite")
        improved = self.best_epoch == 0 or loss < self.best_loss
        if improved:
            self.best_loss = float(loss)
            self.best_epoch = int(epoch)
            self.stale_epochs = 0
            self.should_stop = False
        else:
            self.stale_epochs += 1
            self.should_stop = self.stale_epochs >= self.patience
        return improved


@dataclass
class CellData:
    feature_panel: pd.DataFrame
    train_labels: pd.DataFrame
    validation_labels: pd.DataFrame
    train_availability: dict[pd.Timestamp, list[str]]
    validation_availability: dict[pd.Timestamp, list[str]]
    audit: dict[str, object]
    receipts: list[dict[str, object]]


class JointFeatureLabelStore:
    def __init__(
        self,
        feature_panel: pd.DataFrame,
        label_frames: list[pd.DataFrame],
        feature_names: list[str],
        lookback_days: int,
        relative_source_feature: str,
        return_column: str,
    ) -> None:
        self.feature_store = TripletTensorStore(
            feature_panel[["date", "symbol", *feature_names]],
            feature_names,
            lookback_days,
            relative_source_feature,
        )
        labels = pd.concat(label_frames, ignore_index=True)
        if labels.duplicated(["date", "symbol"]).any():
            raise RuntimeError("V49 train and validation labels overlap")
        self.labels = np.full(
            (len(self.feature_store.symbols), len(self.feature_store.dates)),
            np.nan,
            dtype=np.float32,
        )
        for symbol, frame in labels.groupby("symbol", sort=True):
            symbol_index = self.feature_store.symbol_to_index[str(symbol)]
            date_indexes = np.asarray(
                [
                    self.feature_store.date_to_index[pd.Timestamp(date)]
                    for date in frame["date"]
                ],
                dtype=np.int64,
            )
            self.labels[symbol_index, date_indexes] = frame[return_column].to_numpy(
                dtype=np.float32
            )

    def materialize_batch(
        self,
        samples: list[dict[str, object]],
        scaler: FeatureScaler,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = self.feature_store.materialize_batch(samples, scaler)
        asset_indexes = np.asarray(
            [
                [
                    self.feature_store.symbol_to_index[str(symbol)]
                    for symbol in sample["triplet"]
                ]
                for sample in samples
            ],
            dtype=np.int64,
        )
        date_indexes = np.asarray(
            [
                self.feature_store.date_to_index[pd.Timestamp(sample["date"])]
                for sample in samples
            ],
            dtype=np.int64,
        )
        y = self.labels[asset_indexes, date_indexes[:, None]]
        if y.shape != (len(samples), 3) or not np.isfinite(y).all():
            raise RuntimeError("V49 materialized invalid return labels")
        return x, y.astype(np.float32, copy=False)


class CanonicalTripletSampler:
    def __init__(
        self,
        availability: dict[pd.Timestamp, list[str]],
        role_symbols: list[str],
        *,
        master_seed: int,
        version: str,
        origin: str,
        geometry: str,
        fold: int,
        seed: int,
        role: str,
    ) -> None:
        allowed = set(role_symbols)
        entries = []
        for date, symbols in sorted(availability.items()):
            available = tuple(sorted(set(symbols).intersection(allowed)))
            if len(available) >= 3:
                entries.append(
                    (pd.Timestamp(date), available, math.comb(len(available), 3))
                )
        if not entries:
            raise RuntimeError("V49 sampler has no eligible date-triplet pairs")
        self.entries = entries
        self.triplets_by_symbols = {
            symbols: tuple(combinations(symbols, 3))
            for symbols in {entry[1] for entry in entries}
        }
        self.cumulative = np.cumsum(
            [entry[2] for entry in entries], dtype=np.int64
        )
        self.total_pairs = int(self.cumulative[-1])
        self.key_prefix = (
            int(master_seed),
            str(version),
            str(origin),
            str(geometry),
            int(fold),
            int(seed),
        )
        self.role = str(role)

    def _seed(self, epoch: int) -> int:
        payload = json.dumps(
            [*self.key_prefix, int(epoch), self.role],
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    def sample_epoch(
        self, epoch: int, sample_count: int
    ) -> tuple[list[dict[str, object]], str]:
        if sample_count < 1:
            raise ValueError("V49 sample_count must be positive")
        rng = np.random.default_rng(self._seed(epoch))
        draws = rng.integers(
            0, self.total_pairs, size=sample_count, dtype=np.int64
        )
        samples = []
        receipt = []
        for draw in draws:
            entry_index = int(np.searchsorted(self.cumulative, draw, side="right"))
            prior = int(self.cumulative[entry_index - 1]) if entry_index else 0
            date, symbols, _ = self.entries[entry_index]
            triplet = self.triplets_by_symbols[symbols][int(draw) - prior]
            samples.append(
                {"date": date, "triplet": triplet, "pair_index": int(draw)}
            )
            receipt.append([date.isoformat(), list(triplet), int(draw)])
        return samples, _canonical_sha256(receipt)


class SimulatedSmokeInterruption(RuntimeError):
    pass


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


@contextmanager
def _process_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another V49 process holds the training lock") from exc
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _git_receipt(root: Path, required: bool) -> dict[str, object]:
    if not required:
        return {"required": False, "head": "not_required", "clean": True}
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("V49 requires a clean committed Git receipt")
    return {"required": True, "head": head, "clean": True}


def _source_receipt(root: Path, source_files: list[str]) -> dict[str, object]:
    hashes = {}
    for relative in source_files:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"V49 source file is missing: {relative}")
        hashes[relative] = _sha256_file(path)
    return {
        "files": hashes,
        "bundle_sha256": _canonical_sha256(hashes),
    }


def _metadata_context(config: dict) -> dict[str, object]:
    training = config["joint_absolute_relative_training"]
    root = Path(training["project_root"]).resolve()
    paths = {
        name: (root / relative).resolve()
        for name, relative in training["inputs"].items()
    }
    if set(paths) != METADATA_INPUTS | BINARY_INPUTS:
        raise RuntimeError("V49 input allowlist drift")
    input_hashes = {}
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"V49 input is missing: {name}")
        observed = _sha256_file(path)
        if observed != training["expected_input_sha256"][name]:
            raise RuntimeError(f"V49 input hash drift: {name}")
        input_hashes[name] = observed
    values = {name: _load_json(paths[name]) for name in METADATA_INPUTS}
    blueprint = values["v47_blueprint"]
    if (
        values["v47_result"]["decision"]
        != "authorize_v48_joint_absolute_relative_synthetic_harness_only"
        or not values["v47_audit"]["passed"]
        or values["v48_result"]["decision"]
        != "authorize_v49_purged_non_target_training_only"
        or not values["v48_audit"]["passed"]
        or values["v48_harness_spec"]["v47_blueprint_sha256"]
        != blueprint["blueprint_sha256"]
    ):
        raise RuntimeError("V47/V48 do not authorize V49")
    manifest = values["v32_dataset_manifest"]
    feature_schema = values["v32_feature_schema"]
    folds = values["v32_asset_folds"]
    catalog = values["v32_triplet_catalog"]
    data_access = training["data_access"]
    if (
        manifest["panel_sha256"] != input_hashes["panel"]
        or manifest["sequence_index_sha256"] != input_hashes["sequence_index"]
        or list(feature_schema["model_feature_order"][:-1])
        != list(manifest["panel_features"])
        or list(data_access["feature_columns"])
        != ["date", "symbol", *manifest["panel_features"]]
        or list(data_access["label_columns"])
        != [
            "date",
            "symbol",
            "target_window_end_date",
            "target_next_open_to_next_open_log_return",
        ]
        or len(folds["folds"]) != 3
        or len(catalog["folds"]) != 3
        or TARGET_SYMBOLS.intersection(manifest["symbols"])
    ):
        raise RuntimeError("V49 V32 dataset contract drift")
    source = _source_receipt(root, list(training["source_files"]))
    git = _git_receipt(root, bool(training["require_clean_git_receipt"]))
    return {
        "root": root,
        "paths": paths,
        "training": training,
        "values": values,
        "blueprint": blueprint,
        "manifest": manifest,
        "feature_schema": feature_schema,
        "asset_folds": folds,
        "triplet_catalog": catalog,
        "input_hashes": input_hashes,
        "source": source,
        "git": git,
    }


def _build_contract(context: dict[str, object]) -> dict[str, object]:
    training = context["training"]
    blueprint = context["blueprint"]
    contract = {
        "format": training["training_spec_format"],
        "version": "v49",
        "candidate_family_id": blueprint["candidate_family_id"],
        "v47_blueprint_sha256": blueprint["blueprint_sha256"],
        "v48_harness_spec_sha256": context["values"]["v48_harness_spec"][
            "harness_spec_sha256"
        ],
        "architecture": blueprint["architecture"],
        "objective": training["objective"],
        "early_stopping": training["early_stopping"],
        "origins": training["origins"],
        "folds": training["folds"],
        "seeds": training["seeds"],
        "geometries": training["geometries"],
        "expected_full_job_count": training["expected_full_job_count"],
        "initialization": training["initialization"],
        "sampling": training["sampling"],
        "optimizer": training["optimizer"],
        "gradient_clip_norm": training["gradient_clip_norm"],
        "data_access": training["data_access"],
        "device": training["device"],
        "dtype": training["dtype"],
        "amp": training["amp"],
        "deterministic_algorithms": training["deterministic_algorithms"],
        "cpu_fallback_allowed": training["cpu_fallback_allowed"],
        "checkpoint_format": training["checkpoint_format"],
        "resume_format": training["resume_format"],
        "constraints": training["constraints"],
        "authorized_next_action": training["authorized_next_action"],
    }
    contract["contract_sha256"] = _canonical_sha256(contract)
    return contract


def _build_training_spec(
    context: dict[str, object], mode: str, effective: dict | None
) -> dict[str, object]:
    contract = _build_contract(context)
    spec = {
        "contract": contract,
        "contract_sha256": contract["contract_sha256"],
        "mode": mode,
        "effective_grid": effective,
        "input_hashes": context["input_hashes"],
        "source_receipt": context["source"],
        "git_receipt": context["git"],
    }
    spec["training_spec_sha256"] = _canonical_sha256(spec)
    return spec


def _origin_map(training: dict) -> dict[str, dict]:
    return {str(item["id"]): item for item in training["origins"]}


def _serialize_filters(filters: list[tuple[str, str, object]]) -> list[list[object]]:
    rows = []
    for column, operator, value in filters:
        if isinstance(value, pd.Timestamp):
            value = value.isoformat()
        elif isinstance(value, (list, tuple)):
            value = list(value)
        rows.append([column, operator, value])
    return rows


def _validate_frame(
    frame: pd.DataFrame,
    columns: list[str],
    allowed_symbols: set[str],
    label: str,
) -> None:
    if list(frame.columns) != columns:
        raise RuntimeError(f"V49 {label} projection drift")
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame["symbol"] = frame["symbol"].astype(str)
    if frame.duplicated(["date", "symbol"]).any():
        raise RuntimeError(f"V49 {label} contains duplicate keys")
    loaded = set(frame["symbol"].unique())
    if loaded != allowed_symbols or loaded.intersection(TARGET_SYMBOLS):
        raise RuntimeError(f"V49 {label} ignored its symbol isolation")


def read_cell_data(
    panel_path: Path,
    sequence_path: Path,
    fold_entry: dict,
    origin: dict,
    geometry: str,
    data_access: dict,
    *,
    reader: Callable[..., pd.DataFrame] = pd.read_parquet,
) -> CellData:
    fold = int(fold_entry["fold"])
    train_symbols = sorted(fold_entry["train_symbols"])
    test_symbols = set(fold_entry["test_symbols"])
    if len(train_symbols) != 20 or len(test_symbols) != 10:
        raise RuntimeError("V49 fold cardinality drift")
    if set(train_symbols).intersection(test_symbols | TARGET_SYMBOLS):
        raise RuntimeError("V49 fold contains forbidden overlap")
    window = origin["geometries"][geometry]
    train_start = pd.Timestamp(window["train_start"], tz="UTC")
    train_end = pd.Timestamp(window["train_end"], tz="UTC")
    validation_start = pd.Timestamp(origin["validation_start"], tz="UTC")
    validation_end = pd.Timestamp(origin["validation_end"], tz="UTC")
    sequence_columns = list(data_access["sequence_columns"])
    feature_columns = list(data_access["feature_columns"])
    label_columns = list(data_access["label_columns"])

    def eligible_filters(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple]:
        return [
            ("symbol", "in", train_symbols),
            (data_access["sequence_ready_filter"], "==", True),
            (data_access["label_complete_filter"], "==", True),
            ("date", ">=", start),
            ("date", "<=", end),
        ]

    train_filters = eligible_filters(train_start, train_end)
    validation_filters = eligible_filters(validation_start, validation_end)
    train_index = reader(
        sequence_path,
        engine="pyarrow",
        columns=sequence_columns,
        filters=train_filters,
    )
    validation_index = reader(
        sequence_path,
        engine="pyarrow",
        columns=sequence_columns,
        filters=validation_filters,
    )
    for frame, label in (
        (train_index, "train sequence index"),
        (validation_index, "validation sequence index"),
    ):
        _validate_frame(frame, sequence_columns, set(train_symbols), label)
        starts = pd.to_datetime(frame["sequence_start_date"], utc=True)
        if not bool((starts == frame["date"] - pd.Timedelta(days=255)).all()):
            raise RuntimeError("V49 sequence lookback drift")

    feature_start = min(
        pd.to_datetime(train_index["sequence_start_date"], utc=True).min(),
        pd.to_datetime(validation_index["sequence_start_date"], utc=True).min(),
    )
    feature_end = max(train_index["date"].max(), validation_index["date"].max())
    feature_filters = [
        ("symbol", "in", train_symbols),
        ("date", ">=", feature_start),
        ("date", "<=", feature_end),
    ]
    feature_panel = reader(
        panel_path,
        engine="pyarrow",
        columns=feature_columns,
        filters=feature_filters,
    )
    train_labels = reader(
        panel_path,
        engine="pyarrow",
        columns=label_columns,
        filters=train_filters,
    )
    validation_labels = reader(
        panel_path,
        engine="pyarrow",
        columns=label_columns,
        filters=validation_filters,
    )
    for frame, columns, label in (
        (feature_panel, feature_columns, "feature panel"),
        (train_labels, label_columns, "train labels"),
        (validation_labels, label_columns, "validation labels"),
    ):
        _validate_frame(frame, columns, set(train_symbols), label)

    for frame, maturity_end, label in (
        (train_labels, pd.Timestamp(window["train_maturity_end"], tz="UTC"), "train"),
        (
            validation_labels,
            pd.Timestamp(origin["validation_maturity_end"], tz="UTC"),
            "validation",
        ),
    ):
        frame["target_window_end_date"] = pd.to_datetime(
            frame["target_window_end_date"], utc=True
        )
        values = frame[data_access["return_column"]].to_numpy(dtype=np.float64)
        if (
            not np.isfinite(values).all()
            or not bool(
                (
                    frame["target_window_end_date"]
                    == frame["date"] + pd.Timedelta(days=8)
                ).all()
            )
            or frame["target_window_end_date"].max() > maturity_end
        ):
            raise RuntimeError(f"V49 {label} return-label maturity drift")
    train_keys = set(zip(train_labels["date"], train_labels["symbol"], strict=True))
    validation_keys = set(
        zip(validation_labels["date"], validation_labels["symbol"], strict=True)
    )
    if train_keys != set(zip(train_index["date"], train_index["symbol"], strict=True)):
        raise RuntimeError("V49 train label/index key drift")
    if validation_keys != set(
        zip(validation_index["date"], validation_index["symbol"], strict=True)
    ):
        raise RuntimeError("V49 validation label/index key drift")

    train_availability = _availability_from_index(train_index)
    validation_availability = _availability_from_index(validation_index)
    if not train_availability or not validation_availability:
        raise RuntimeError("V49 cell has no eligible train or validation dates")
    receipts = [
        {
            "dataset": name,
            "columns": columns,
            "filters": _serialize_filters(filters),
        }
        for name, columns, filters in (
            ("sequence_train", sequence_columns, train_filters),
            ("sequence_validation", sequence_columns, validation_filters),
            ("panel_features", feature_columns, feature_filters),
            ("panel_labels_train", label_columns, train_filters),
            ("panel_labels_validation", label_columns, validation_filters),
        )
    ]
    audit = {
        "origin": origin["id"],
        "geometry": geometry,
        "fold": fold,
        "train_symbols": train_symbols,
        "heldout_symbols_materialized": [],
        "target_symbols_materialized": [],
        "label_columns_materialized": [data_access["return_column"]],
        "forbidden_label_columns_materialized": [],
        "feature_rows": len(feature_panel),
        "train_label_rows": len(train_labels),
        "validation_label_rows": len(validation_labels),
        "train_sequence_rows": len(train_index),
        "validation_sequence_rows": len(validation_index),
        "train_eligible_dates": len(train_availability),
        "validation_eligible_dates": len(validation_availability),
        "first_feature_date": feature_panel["date"].min().date().isoformat(),
        "last_feature_date": feature_panel["date"].max().date().isoformat(),
        "first_train_signal_date": min(train_availability).date().isoformat(),
        "last_train_signal_date": max(train_availability).date().isoformat(),
        "first_validation_signal_date": min(validation_availability).date().isoformat(),
        "last_validation_signal_date": max(validation_availability).date().isoformat(),
        "maximum_train_target_maturity": train_labels[
            "target_window_end_date"
        ].max().date().isoformat(),
        "maximum_validation_target_maturity": validation_labels[
            "target_window_end_date"
        ].max().date().isoformat(),
        "physical_row_group_isolation_claimed": False,
    }
    return CellData(
        feature_panel=feature_panel,
        train_labels=train_labels,
        validation_labels=validation_labels,
        train_availability=train_availability,
        validation_availability=validation_availability,
        audit=audit,
        receipts=receipts,
    )


def fit_cell_return_scale(
    train_labels: pd.DataFrame,
    train_availability: dict[pd.Timestamp, list[str]],
    return_column: str,
    floor: float,
) -> dict[str, object]:
    returns = {
        (pd.Timestamp(date), str(symbol)): float(value)
        for date, symbol, value in zip(
            train_labels["date"],
            train_labels["symbol"],
            train_labels[return_column],
            strict=True,
        )
    }
    sum_squares = 0.0
    value_count = 0
    triplet_count = 0
    for date, symbols in sorted(train_availability.items()):
        for triplet in combinations(sorted(symbols), 3):
            values = [returns[(date, symbol)] for symbol in triplet]
            if not all(math.isfinite(value) for value in values):
                raise RuntimeError("V49 scale enumeration found non-finite return")
            sum_squares += sum(value * value for value in values)
            value_count += 3
            triplet_count += 1
    if not value_count:
        raise RuntimeError("V49 scale enumeration is empty")
    scale = max(math.sqrt(sum_squares / value_count), float(floor))
    record = {
        "fit_scope": "complete_train_only_date_triplet_raw_return_enumeration",
        "fit_start": min(train_availability).date().isoformat(),
        "fit_end": max(train_availability).date().isoformat(),
        "eligible_dates": len(train_availability),
        "enumerated_triplets": triplet_count,
        "enumerated_raw_returns": value_count,
        "scale_floor": float(floor),
        "raw_return_rms_scale": float(scale),
    }
    record["target_scale_state_sha256"] = _canonical_sha256(record)
    return record


def _scaler_from_record(record: dict) -> FeatureScaler:
    names = {field.name for field in fields(FeatureScaler)}
    values = {name: record[name] for name in names}
    values["feature_names"] = tuple(values["feature_names"])
    values["mean"] = tuple(values["mean"])
    values["scale"] = tuple(values["scale"])
    return FeatureScaler(**values)


def _run_batches(
    model: JointAbsoluteRelativeTransformer,
    store: JointFeatureLabelStore,
    scaler: FeatureScaler,
    samples: list[dict[str, object]],
    *,
    batch_size: int,
    target_scale: float,
    tie_tolerance: float,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float,
) -> tuple[dict[str, float], int]:
    training = optimizer is not None
    model.train(training)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    sample_sums = {
        "excess": 0.0,
        "market_level": 0.0,
        "absolute_level": 0.0,
        "level": 0.0,
    }
    ranking_sum = 0.0
    observations = 0
    pair_count = 0
    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        x_np, y_np = store.materialize_batch(batch_samples, scaler)
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            losses = joint_absolute_relative_loss(
                model(x), y, target_scale, tie_tolerance=tie_tolerance
            )
        if any(
            not bool(torch.isfinite(losses[name]))
            for name in (*sample_sums, "ranking", "total")
        ):
            raise RuntimeError("V49 produced a non-finite objective")
        if optimizer is not None:
            losses["total"].backward()
            if any(
                parameter.grad is None
                or not bool(torch.isfinite(parameter.grad).all())
                for parameter in parameters
            ):
                raise RuntimeError("V49 produced missing or non-finite gradients")
            gradient_norm = nn.utils.clip_grad_norm_(parameters, gradient_clip_norm)
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("V49 produced a non-finite gradient norm")
            optimizer.step()
        count = len(batch_samples)
        batch_pairs = int(losses["pair_count"].detach().cpu())
        ranking_sum += float(losses["ranking"].detach().cpu()) * batch_pairs
        for name in sample_sums:
            sample_sums[name] += float(losses[name].detach().cpu()) * count
        observations += count
        pair_count += batch_pairs
    if not observations or not pair_count:
        raise RuntimeError("V49 batches contain no observations or active pairs")
    ranking = ranking_sum / pair_count
    averaged = {
        name: value / observations for name, value in sample_sums.items()
    }
    return {
        "ranking": ranking,
        **averaged,
        "total": ranking + averaged["excess"] + averaged["level"],
        "observations": observations,
        "pair_count": pair_count,
    }, math.ceil(observations / batch_size)


def _save_resume(
    path: Path,
    *,
    model: JointAbsoluteRelativeTransformer,
    optimizer: torch.optim.Optimizer,
    early: V49EarlyStopping,
    history: list[dict[str, object]],
    completed_epoch: int,
    metadata: dict[str, object],
    best_model_state: dict[str, torch.Tensor],
    device: torch.device,
    format_version: str,
    parameter_names: list[str],
) -> None:
    optimizer_state = _to_cpu(optimizer.state_dict())
    rng = _rng_state(device)
    payload = {
        "format_version": format_version,
        "metadata": metadata,
        "completed_epoch": int(completed_epoch),
        "model_state_dict": _cpu_state_dict(model.state_dict()),
        "model_state_sha256": model_state_sha256(model.state_dict()),
        "best_model_state_dict": _cpu_state_dict(best_model_state),
        "best_model_state_sha256": model_state_sha256(best_model_state),
        "optimizer_state_dict": optimizer_state,
        "optimizer_state_sha256": _semantic_state_sha256(optimizer_state),
        "optimizer_contract": _optimizer_contract(optimizer, parameter_names),
        "early_stopping": asdict(early),
        "history": history,
        "history_sha256": _semantic_state_sha256(history),
        "rng_state": rng,
        "rng_state_sha256": _semantic_state_sha256(rng),
    }
    _atomic_torch_save(payload, path)


def _load_resume(
    path: Path,
    *,
    model: JointAbsoluteRelativeTransformer,
    optimizer: torch.optim.Optimizer,
    expected_metadata: dict[str, object],
    device: torch.device,
    format_version: str,
    patience: int,
    maximum_epochs: int,
    parameter_names: list[str],
    steps_per_epoch: int,
) -> tuple[int, V49EarlyStopping, list[dict[str, object]], dict[str, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != format_version:
        raise RuntimeError("V49 resume format drift")
    if payload.get("metadata") != expected_metadata:
        raise RuntimeError("V49 resume metadata drift")
    epoch = int(payload.get("completed_epoch", -1))
    history = list(payload.get("history", []))
    early_payload = dict(payload.get("early_stopping", {}))
    if (
        not 0 < epoch <= maximum_epochs
        or len(history) != epoch
        or [int(row["epoch"]) for row in history] != list(range(1, epoch + 1))
        or _semantic_state_sha256(history) != payload.get("history_sha256")
        or int(early_payload.get("patience", -1)) != patience
        or payload.get("model_state_sha256")
        != model_state_sha256(payload["model_state_dict"])
        or payload.get("best_model_state_sha256")
        != model_state_sha256(payload["best_model_state_dict"])
        or not _state_is_finite(payload["model_state_dict"])
        or not _state_is_finite(payload["best_model_state_dict"])
        or payload.get("optimizer_state_sha256")
        != _semantic_state_sha256(payload["optimizer_state_dict"])
        or payload.get("rng_state_sha256")
        != _semantic_state_sha256(payload["rng_state"])
    ):
        raise RuntimeError("V49 resume state drift")
    _validate_optimizer_resume_state(
        payload["optimizer_state_dict"],
        optimizer,
        parameter_names,
        epoch * steps_per_epoch,
    )
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    _move_optimizer_state(optimizer, device)
    _restore_rng_state(payload["rng_state"], device)
    return (
        epoch,
        V49EarlyStopping(**early_payload),
        history,
        _cpu_state_dict(payload["best_model_state_dict"]),
    )


def _load_final_checkpoint(
    path: Path,
    *,
    architecture: dict,
    format_version: str,
    expected_metadata: dict[str, object],
) -> tuple[JointAbsoluteRelativeTransformer, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("format_version") != format_version
        or payload.get("architecture") != architecture
        or payload.get("architecture_sha256") != _canonical_sha256(architecture)
        or payload.get("metadata") != expected_metadata
        or payload.get("model_state_sha256")
        != model_state_sha256(payload["model_state_dict"])
        or not _state_is_finite(payload["model_state_dict"])
    ):
        raise RuntimeError(f"V49 final checkpoint drift: {path}")
    model = JointAbsoluteRelativeTransformer(
        int(payload["input_features"]), architecture
    )
    model.load_state_dict(payload["model_state_dict"])
    return model, payload


def _completed_job(
    job_dir: Path,
    *,
    architecture: dict,
    checkpoint_format: str,
    expected_base_metadata: dict[str, object],
) -> dict[str, object] | None:
    complete_path = job_dir / "complete.json"
    if not complete_path.is_file():
        return None
    complete = _load_json(complete_path)
    checkpoint_path = job_dir / "checkpoint.pt"
    metadata = dict(expected_base_metadata)
    metadata.update(complete.get("final_metadata", {}))
    if (
        not checkpoint_path.is_file()
        or _sha256_file(checkpoint_path) != complete.get("checkpoint_sha256")
        or complete.get("base_metadata") != expected_base_metadata
    ):
        raise RuntimeError(f"V49 completed job drift: {job_dir}")
    model, payload = _load_final_checkpoint(
        checkpoint_path,
        architecture=architecture,
        format_version=checkpoint_format,
        expected_metadata=metadata,
    )
    if payload["model_state_sha256"] != complete.get("model_state_sha256"):
        raise RuntimeError(f"V49 completed state hash drift: {job_dir}")
    resume_path = job_dir / "resume.pt"
    if resume_path.exists():
        resume_path.unlink()
    del model, payload
    return complete


def _train_job(
    *,
    context: dict[str, object],
    contract: dict,
    run_kind: str,
    origin: dict,
    geometry: str,
    fold_entry: dict,
    seed: int,
    store: JointFeatureLabelStore,
    scaler: FeatureScaler,
    target_scale: float,
    train_availability: dict[pd.Timestamp, list[str]],
    validation_availability: dict[pd.Timestamp, list[str]],
    cell_hashes: dict[str, str],
    effective: dict,
    checkpoint_root: Path,
    device: torch.device,
    interrupt_after_epoch: int | None = None,
) -> dict[str, object]:
    training = context["training"]
    architecture = context["blueprint"]["architecture"]
    fold = int(fold_entry["fold"])
    train_symbols = sorted(fold_entry["train_symbols"])
    test_symbols = sorted(fold_entry["test_symbols"])
    job_dir = (
        checkpoint_root
        / str(origin["id"])
        / str(geometry)
        / f"fold_{fold}"
        / f"seed_{seed}"
    )
    job_dir.mkdir(parents=True, exist_ok=True)
    base_metadata = {
        "version": "v49",
        "run_kind": run_kind,
        "candidate_family_id": context["blueprint"]["candidate_family_id"],
        "contract_sha256": contract["contract_sha256"],
        "source_bundle_sha256": context["source"]["bundle_sha256"],
        "git_head": context["git"]["head"],
        "origin": origin["id"],
        "geometry": geometry,
        "fold": fold,
        "seed": int(seed),
        "train_symbols": train_symbols,
        "test_symbols": test_symbols,
        **cell_hashes,
    }
    if set(train_symbols).intersection(set(test_symbols) | TARGET_SYMBOLS):
        raise RuntimeError("V49 job contains forbidden asset overlap")

    _seed_device(seed, device)
    model = JointAbsoluteRelativeTransformer(9, architecture).to(device)
    initialization_state_sha256 = model_state_sha256(model.state_dict())
    parameter_names = [name for name, _ in model.named_parameters()]
    optimizer_config = training["optimizer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        eps=float(optimizer_config["epsilon"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        amsgrad=bool(optimizer_config["amsgrad"]),
    )
    sampler_args = {
        "master_seed": int(training["sampling"]["master_seed"]),
        "version": "v49",
        "origin": str(origin["id"]),
        "geometry": str(geometry),
        "fold": fold,
        "seed": int(seed),
    }
    train_sampler = CanonicalTripletSampler(
        train_availability, train_symbols, role="train", **sampler_args
    )
    validation_sampler = CanonicalTripletSampler(
        validation_availability, train_symbols, role="validation", **sampler_args
    )
    validation_samples, validation_sample_sha256 = validation_sampler.sample_epoch(
        int(training["validation_sampling_epoch"]),
        int(effective["validation_samples"]),
    )
    base_metadata["initialization_state_sha256"] = initialization_state_sha256
    base_metadata["validation_sample_sha256"] = validation_sample_sha256
    completed = _completed_job(
        job_dir,
        architecture=architecture,
        checkpoint_format=training["checkpoint_format"],
        expected_base_metadata=base_metadata,
    )
    if completed is not None:
        return completed
    early = V49EarlyStopping(
        patience=int(effective["early_stopping_patience"])
    )
    history: list[dict[str, object]] = []
    completed_epoch = 0
    best_model_state = _cpu_state_dict(model.state_dict())
    steps_per_epoch = math.ceil(
        int(effective["train_samples_per_epoch"]) / int(effective["batch_size"])
    )
    resume_path = job_dir / "resume.pt"
    if resume_path.is_file():
        completed_epoch, early, history, best_model_state = _load_resume(
            resume_path,
            model=model,
            optimizer=optimizer,
            expected_metadata=base_metadata,
            device=device,
            format_version=training["resume_format"],
            patience=int(effective["early_stopping_patience"]),
            maximum_epochs=int(effective["maximum_epochs"]),
            parameter_names=parameter_names,
            steps_per_epoch=steps_per_epoch,
        )

    if not early.should_stop:
        for epoch in range(completed_epoch + 1, int(effective["maximum_epochs"]) + 1):
            train_samples, train_sample_sha256 = train_sampler.sample_epoch(
                epoch, int(effective["train_samples_per_epoch"])
            )
            train_losses, train_steps = _run_batches(
                model,
                store,
                scaler,
                train_samples,
                batch_size=int(effective["batch_size"]),
                target_scale=target_scale,
                tie_tolerance=float(training["objective"]["exact_tie_tolerance"]),
                device=device,
                optimizer=optimizer,
                gradient_clip_norm=float(training["gradient_clip_norm"]),
            )
            validation_losses, validation_steps = _run_batches(
                model,
                store,
                scaler,
                validation_samples,
                batch_size=int(effective["batch_size"]),
                target_scale=target_scale,
                tie_tolerance=float(training["objective"]["exact_tie_tolerance"]),
                device=device,
                optimizer=None,
                gradient_clip_norm=float(training["gradient_clip_norm"]),
            )
            improved = early.update(epoch, validation_losses["total"])
            if improved:
                best_model_state = _cpu_state_dict(model.state_dict())
            history.append(
                {
                    "epoch": epoch,
                    "train_losses": train_losses,
                    "validation_losses": validation_losses,
                    "train_steps": train_steps,
                    "validation_steps": validation_steps,
                    "train_sample_sha256": train_sample_sha256,
                    "validation_sample_sha256": validation_sample_sha256,
                    "improved": improved,
                    "stale_epochs": early.stale_epochs,
                }
            )
            _save_resume(
                resume_path,
                model=model,
                optimizer=optimizer,
                early=early,
                history=history,
                completed_epoch=epoch,
                metadata=base_metadata,
                best_model_state=best_model_state,
                device=device,
                format_version=training["resume_format"],
                parameter_names=parameter_names,
            )
            if interrupt_after_epoch == epoch:
                raise SimulatedSmokeInterruption("V49 synthetic smoke interruption")
            if early.should_stop:
                break

    completed_epochs = len(history)
    if not completed_epochs or early.best_epoch < 1:
        raise RuntimeError("V49 job completed without a best epoch")
    model.load_state_dict(best_model_state)
    if not _state_is_finite(model.state_dict()):
        raise RuntimeError("V49 best model state is non-finite")
    train_optimizer_steps = completed_epochs * steps_per_epoch
    history_sha256 = _semantic_state_sha256(history)
    final_metadata = {
        "best_epoch": early.best_epoch,
        "best_validation_total_loss": early.best_loss,
        "completed_epochs": completed_epochs,
        "train_optimizer_steps": train_optimizer_steps,
        "history_sha256": history_sha256,
        "checkpoint_status": "job_local_best_all_jobs_retained_no_selection",
    }
    checkpoint_metadata = {**base_metadata, **final_metadata}
    checkpoint_path = job_dir / "checkpoint.pt"
    checkpoint_payload = {
        "format_version": training["checkpoint_format"],
        "architecture": architecture,
        "architecture_sha256": _canonical_sha256(architecture),
        "input_features": 9,
        "prediction_heads": list(JOINT_HEADS),
        "model_state_dict": _cpu_state_dict(model.state_dict()),
        "model_state_sha256": model_state_sha256(model.state_dict()),
        "metadata": checkpoint_metadata,
    }
    _atomic_torch_save(checkpoint_payload, checkpoint_path)
    reopened, payload = _load_final_checkpoint(
        checkpoint_path,
        architecture=architecture,
        format_version=training["checkpoint_format"],
        expected_metadata=checkpoint_metadata,
    )
    complete = {
        "version": "v49_job_complete_v1",
        "base_metadata": base_metadata,
        "final_metadata": final_metadata,
        "origin": origin["id"],
        "geometry": geometry,
        "fold": fold,
        "seed": int(seed),
        "train_symbols": train_symbols,
        "test_symbols": test_symbols,
        "target_scale": float(target_scale),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "model_state_sha256": payload["model_state_sha256"],
        "best_epoch": early.best_epoch,
        "best_validation_total_loss": early.best_loss,
        "completed_epochs": completed_epochs,
        "train_optimizer_steps": train_optimizer_steps,
        "history": history,
        "history_sha256": history_sha256,
        "heldout_assets_loaded": False,
        "target_assets_loaded": False,
        "deployment_predictions_persisted": False,
        "predictive_metrics_computed": False,
        "policy_actions_computed": False,
        "performance_metrics_computed": False,
        "pnl_computed": False,
        "seed_fold_origin_geometry_selected": False,
    }
    _atomic_write_json(job_dir / "complete.json", complete)
    del reopened, payload
    _completed_job(
        job_dir,
        architecture=architecture,
        checkpoint_format=training["checkpoint_format"],
        expected_base_metadata=base_metadata,
    )
    return complete


def _configure_device(training: dict) -> torch.device:
    torch.set_num_threads(int(training["torch_threads"]))
    torch.use_deterministic_algorithms(True)
    fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0").strip().lower()
    if training["device"] != "mps" or fallback not in {"", "0", "false", "no"}:
        raise RuntimeError("V49 requires MPS with CPU fallback disabled")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; run V49 in the host environment")
    return torch.device("mps")


def _environment(context: dict[str, object]) -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "git_receipt": context["git"],
        "source_receipt": context["source"],
    }


def _report(result: dict[str, object]) -> str:
    summary = result["summary"]
    return "\n".join(
        [
            "# TLM v49 Joint Absolute/Relative Training",
            "",
            "## Decision",
            "",
            f"**{result['decision']}**",
            "",
            f"Checkpoints: **{summary['checkpoint_count']}**",
            f"Completed epochs: **{summary['total_completed_epochs']}**",
            f"Optimizer steps: **{summary['total_optimizer_steps']}**",
            f"Contract SHA-256: `{result['training_spec']['contract_sha256']}`",
            "",
            "Every checkpoint is a job-local best state. No seed, fold, origin, or geometry was selected or discarded.",
            "",
            "BTC/ETH/SOL, held-out outcomes, deployment predictions, predictive metrics, policy actions, PnL, Sharpe, and drawdown remained sealed.",
            "",
            "## Next action",
            "",
            "Stop. V50 economic evaluation remains outside this autonomous loop.",
            "",
        ]
    )


def _write_result(
    root: Path,
    output_relative: str,
    config: dict,
    result: dict[str, object],
) -> None:
    output = root / output_relative
    output.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output / "result.json", result)
    _atomic_write_json(output / "audit.json", result["audit"])
    _atomic_write_json(output / "training_spec.json", result["training_spec"])
    _atomic_write_json(
        output / "checkpoint_manifest.json", result.get("checkpoint_manifest", [])
    )
    _atomic_write_json(
        output / "data_access_audit.json", result.get("data_access_audit", {})
    )
    _atomic_write_json(output / "target_scales.json", result.get("target_scales", []))
    _atomic_write_json(output / "environment.json", result["environment"])
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output / "report.md").write_text(_report(result), encoding="utf-8")


def _base_metadata_for_grid(
    context: dict[str, object],
    contract: dict,
    run_kind: str,
    origin: dict,
    geometry: str,
    fold_entry: dict,
    seed: int,
    cell_hashes: dict[str, str],
    initialization_hash: str,
    validation_sample_hash: str,
) -> dict[str, object]:
    return {
        "version": "v49",
        "run_kind": run_kind,
        "candidate_family_id": context["blueprint"]["candidate_family_id"],
        "contract_sha256": contract["contract_sha256"],
        "source_bundle_sha256": context["source"]["bundle_sha256"],
        "git_head": context["git"]["head"],
        "origin": origin["id"],
        "geometry": geometry,
        "fold": int(fold_entry["fold"]),
        "seed": int(seed),
        "train_symbols": sorted(fold_entry["train_symbols"]),
        "test_symbols": sorted(fold_entry["test_symbols"]),
        **cell_hashes,
        "initialization_state_sha256": initialization_hash,
        "validation_sample_sha256": validation_sample_hash,
    }


def _verify_tree(
    context: dict[str, object],
    contract: dict,
    checkpoint_root: Path,
    effective: dict,
    run_kind: str,
) -> list[dict[str, object]]:
    folds = {int(item["fold"]): item for item in context["asset_folds"]["folds"]}
    origins = _origin_map(context["training"])
    jobs = []
    for origin_id in effective["origins"]:
        for geometry in effective["geometries"]:
            for fold in effective["folds"]:
                cell_dir = checkpoint_root / origin_id / geometry / f"fold_{fold}"
                cell_paths = {
                    "data_access_sha256": cell_dir / "data_access.json",
                    "scaler_artifact_sha256": cell_dir / "scaler.json",
                    "target_scale_artifact_sha256": cell_dir / "target_scale.json",
                }
                if not all(path.is_file() for path in cell_paths.values()):
                    raise RuntimeError(f"V49 cell artifact is missing: {cell_dir}")
                cell_hashes = {
                    name: _sha256_file(path) for name, path in cell_paths.items()
                }
                for seed in effective["seeds"]:
                    # Initialization and validation hashes are persisted in complete.json;
                    # use them only after binding all immutable base fields.
                    complete_path = cell_dir / f"seed_{seed}" / "complete.json"
                    if not complete_path.is_file():
                        raise RuntimeError(f"V49 job is missing: {complete_path}")
                    raw = _load_json(complete_path)
                    base = raw.get("base_metadata", {})
                    expected = _base_metadata_for_grid(
                        context,
                        contract,
                        run_kind,
                        origins[origin_id],
                        geometry,
                        folds[int(fold)],
                        int(seed),
                        cell_hashes,
                        str(base.get("initialization_state_sha256", "")),
                        str(base.get("validation_sample_sha256", "")),
                    )
                    complete = _completed_job(
                        complete_path.parent,
                        architecture=context["blueprint"]["architecture"],
                        checkpoint_format=context["training"]["checkpoint_format"],
                        expected_base_metadata=expected,
                    )
                    if complete is None:
                        raise RuntimeError(f"V49 job did not verify: {complete_path}")
                    jobs.append(complete)
    if any(checkpoint_root.rglob("resume.pt")):
        raise RuntimeError("V49 verification found a resume artifact")
    return jobs


def run_joint_absolute_relative_training(
    config: dict, mode: str
) -> dict[str, object]:
    if mode not in {"preflight", "smoke", "full", "verify"}:
        raise ValueError("V49 mode must be preflight, smoke, full, or verify")
    context = _metadata_context(config)
    root = context["root"]
    training = context["training"]
    effective = None if mode == "preflight" else (
        training["smoke"] if mode == "smoke" else training["full_run"]
    )
    spec = _build_training_spec(context, mode, effective)
    contract = spec["contract"]
    architecture = context["blueprint"]["architecture"]
    torch.manual_seed(int(config["seed"]))
    model = JointAbsoluteRelativeTransformer(9, architecture)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    contract_checks = {
        "v47_v48_authorization_is_exact": context["values"]["v48_result"][
            "decision"
        ]
        == "authorize_v49_purged_non_target_training_only",
        "input_hashes_are_exact": context["input_hashes"]
        == training["expected_input_sha256"],
        "source_and_git_receipts_pass": context["source"]["bundle_sha256"]
        and context["git"]["clean"],
        "model_parameter_and_head_contract_is_exact": parameter_count
        == int(training["expected_total_parameters"])
        == 1_212_930
        and tuple(model.prediction_heads) == JOINT_HEADS,
        "fresh_weights_and_no_pretraining_are_frozen": training[
            "initialization"
        ]
        == {
            "source": "fresh_registered_seed",
            "prior_checkpoint_allowed": False,
            "representation_pretraining_allowed": False,
        },
        "full_grid_is_exactly_thirty_six_jobs": len(training["origins"])
        * len(training["geometries"])
        * len(training["folds"])
        * len(training["seeds"])
        == int(training["expected_full_job_count"])
        == 36,
        "runtime_contract_is_exact": training["device"] == "mps"
        and training["dtype"] == "float32"
        and training["amp"] is False
        and training["cpu_fallback_allowed"] is False
        and int(training["full_run"]["train_samples_per_epoch"]) == 8192
        and int(training["full_run"]["validation_samples"]) == 2048
        and int(training["full_run"]["batch_size"]) == 128
        and int(training["full_run"]["maximum_epochs"]) == 30
        and int(training["full_run"]["early_stopping_patience"]) == 5,
        "forbidden_operations_are_all_disabled": all(
            value is False for value in training["constraints"].values()
        ),
        "v50_remains_forbidden": training["authorized_next_action"]
        == "v49_training_complete_economic_evaluation_still_forbidden",
    }
    contract_checks = {name: bool(value) for name, value in contract_checks.items()}
    if not all(contract_checks.values()):
        raise RuntimeError(f"V49 contract audit failed: {contract_checks}")
    del model

    if mode == "preflight":
        result = {
            "version": "v49_preflight",
            "decision": "authorize_v49_one_job_mps_smoke_only",
            "training_spec": spec,
            "summary": {
                "checkpoint_count": 0,
                "total_completed_epochs": 0,
                "total_optimizer_steps": 0,
                "total_parameters": parameter_count,
                "parquet_files_deserialized": 0,
                "label_rows_materialized": 0,
                "mps_available": bool(torch.backends.mps.is_available()),
            },
            "checkpoint_manifest": [],
            "data_access_audit": {
                "parquet_files_deserialized": 0,
                "label_rows_materialized": 0,
            },
            "target_scales": [],
            "environment": _environment(context),
            "audit": {"passed": True, "checks": contract_checks},
        }
        _write_result(root, training["preflight_output_dir"], config, result)
        return result

    preflight_path = root / training["preflight_output_dir"] / "result.json"
    if not preflight_path.is_file():
        raise RuntimeError("V49 requires a passing committed preflight")
    preflight = _load_json(preflight_path)
    if (
        preflight.get("decision") != "authorize_v49_one_job_mps_smoke_only"
        or not preflight.get("audit", {}).get("passed")
        or preflight.get("training_spec", {}).get("contract_sha256")
        != contract["contract_sha256"]
        or preflight.get("training_spec", {}).get("source_receipt")
        != context["source"]
        or preflight.get("training_spec", {}).get("git_receipt") != context["git"]
    ):
        raise RuntimeError("V49 committed preflight receipt drift")

    if mode == "full":
        smoke_path = root / training["smoke_output_dir"] / "result.json"
        if not smoke_path.is_file():
            raise RuntimeError("V49 full training requires a passing MPS smoke")
        smoke = _load_json(smoke_path)
        smoke_summary = smoke.get("summary", {})
        if (
            smoke.get("decision")
            != "authorize_v49_full_thirty_six_job_training_only"
            or not smoke.get("audit", {}).get("passed")
            or smoke.get("training_spec", {}).get("contract_sha256")
            != contract["contract_sha256"]
            or smoke.get("training_spec", {}).get("source_receipt")
            != context["source"]
            or smoke.get("training_spec", {}).get("git_receipt") != context["git"]
            or int(smoke_summary.get("checkpoint_count", -1)) != 1
            or int(smoke_summary.get("resume_events_proven", -1)) != 1
        ):
            raise RuntimeError("V49 MPS smoke receipt drift")

    checkpoint_root = root / (
        training["smoke_checkpoint_dir"]
        if mode == "smoke"
        else training["checkpoint_dir"]
    )
    run_kind = "smoke" if mode == "smoke" else "full"
    if mode == "verify":
        jobs = _verify_tree(
            context, contract, checkpoint_root, training["full_run"], "full"
        )
        verification = {
            "version": "v49_verification_v1",
            "decision": "v49_training_complete_economic_evaluation_still_forbidden",
            "contract_sha256": contract["contract_sha256"],
            "source_receipt": context["source"],
            "git_receipt": context["git"],
            "checkpoint_count": len(jobs),
            "checkpoint_manifest_sha256": _canonical_sha256(
                [job["checkpoint_sha256"] for job in jobs]
            ),
            "parquet_files_deserialized": 0,
            "optimizer_steps": 0,
            "resume_artifacts": 0,
            "passed": len(jobs) == 36,
        }
        if not verification["passed"]:
            raise RuntimeError("V49 verification grid is incomplete")
        _atomic_write_json(root / config["output_dir"] / "verification.json", verification)
        existing = _load_json(root / config["output_dir"] / "result.json")
        return {**existing, "invocation": {"mode": "verify", "new_optimizer_steps": 0}}

    device = _configure_device(training)
    folds = {int(item["fold"]): item for item in context["asset_folds"]["folds"]}
    origins = _origin_map(training)
    feature_names = list(context["manifest"]["panel_features"])
    jobs: list[dict[str, object]] = []
    data_audits = []
    receipts = []
    scales = []
    new_jobs = 0
    resume_events = 0
    lock_path = checkpoint_root / training["process_lock_name"]
    with _process_lock(lock_path):
        for origin_id in effective["origins"]:
            origin = origins[origin_id]
            for geometry in effective["geometries"]:
                for fold in effective["folds"]:
                    fold_entry = folds[int(fold)]
                    cell_dir = checkpoint_root / origin_id / geometry / f"fold_{fold}"
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    data_path = cell_dir / "data_access.json"
                    scaler_path = cell_dir / "scaler.json"
                    scale_path = cell_dir / "target_scale.json"
                    all_complete = all(
                        (cell_dir / f"seed_{seed}" / "complete.json").is_file()
                        for seed in effective["seeds"]
                    )
                    if all_complete and all(
                        path.is_file() for path in (data_path, scaler_path, scale_path)
                    ):
                        access_record = _load_json(data_path)
                        scaler_record = _load_json(scaler_path)
                        scale_record = _load_json(scale_path)
                        cell_hashes = {
                            "data_access_sha256": _sha256_file(data_path),
                            "scaler_artifact_sha256": _sha256_file(scaler_path),
                            "target_scale_artifact_sha256": _sha256_file(scale_path),
                        }
                        for seed in effective["seeds"]:
                            raw = _load_json(cell_dir / f"seed_{seed}" / "complete.json")
                            base = raw["base_metadata"]
                            expected_base = _base_metadata_for_grid(
                                context,
                                contract,
                                run_kind,
                                origin,
                                geometry,
                                fold_entry,
                                int(seed),
                                cell_hashes,
                                base["initialization_state_sha256"],
                                base["validation_sample_sha256"],
                            )
                            complete = _completed_job(
                                cell_dir / f"seed_{seed}",
                                architecture=architecture,
                                checkpoint_format=training["checkpoint_format"],
                                expected_base_metadata=expected_base,
                            )
                            if complete is None:
                                raise RuntimeError("V49 completed job disappeared")
                            jobs.append(complete)
                        data_audits.append(access_record["audit"])
                        receipts.extend(access_record["receipts"])
                        scales.append(scale_record)
                        continue

                    data = read_cell_data(
                        context["paths"]["panel"],
                        context["paths"]["sequence_index"],
                        fold_entry,
                        origin,
                        geometry,
                        training["data_access"],
                    )
                    access_record = {
                        "audit": data.audit,
                        "receipts": [
                            {
                                "origin": origin_id,
                                "geometry": geometry,
                                "fold": int(fold),
                                **receipt,
                            }
                            for receipt in data.receipts
                        ],
                    }
                    _atomic_write_json(data_path, access_record)
                    scaler = FeatureScaler.fit_from_panel(
                        data.feature_panel,
                        feature_names,
                        origin["geometries"][geometry]["train_start"],
                        origin["geometries"][geometry]["train_end"],
                        origin["geometries"][geometry]["train_end"],
                        training["data_access"]["relative_source_feature"],
                    )
                    scaler_record = {
                        **asdict(scaler),
                        "origin": origin_id,
                        "geometry": geometry,
                        "fold": int(fold),
                        "train_symbols": sorted(fold_entry["train_symbols"]),
                        "scaler_state_sha256": scaler.state_sha256(),
                    }
                    _atomic_write_json(scaler_path, scaler_record)
                    scale_record = fit_cell_return_scale(
                        data.train_labels,
                        data.train_availability,
                        training["data_access"]["return_column"],
                        float(training["objective"]["scale_floor"]),
                    )
                    scale_record.update(
                        {
                            "origin": origin_id,
                            "geometry": geometry,
                            "fold": int(fold),
                            "train_symbols": sorted(fold_entry["train_symbols"]),
                        }
                    )
                    scale_record["target_scale_state_sha256"] = _canonical_sha256(
                        {
                            key: value
                            for key, value in scale_record.items()
                            if key != "target_scale_state_sha256"
                        }
                    )
                    _atomic_write_json(scale_path, scale_record)
                    cell_hashes = {
                        "data_access_sha256": _sha256_file(data_path),
                        "scaler_artifact_sha256": _sha256_file(scaler_path),
                        "target_scale_artifact_sha256": _sha256_file(scale_path),
                    }
                    store = JointFeatureLabelStore(
                        data.feature_panel,
                        [data.train_labels, data.validation_labels],
                        feature_names,
                        int(architecture["lookback_days"]),
                        training["data_access"]["relative_source_feature"],
                        training["data_access"]["return_column"],
                    )
                    for seed in effective["seeds"]:
                        job_dir = cell_dir / f"seed_{seed}"
                        already_complete = (job_dir / "complete.json").is_file()
                        try:
                            complete = _train_job(
                                context=context,
                                contract=contract,
                                run_kind=run_kind,
                                origin=origin,
                                geometry=geometry,
                                fold_entry=fold_entry,
                                seed=int(seed),
                                store=store,
                                scaler=scaler,
                                target_scale=float(scale_record["raw_return_rms_scale"]),
                                train_availability=data.train_availability,
                                validation_availability=data.validation_availability,
                                cell_hashes=cell_hashes,
                                effective=effective,
                                checkpoint_root=checkpoint_root,
                                device=device,
                                interrupt_after_epoch=(
                                    1
                                    if mode == "smoke"
                                    and training["smoke"][
                                        "require_interrupted_resume_replay"
                                    ]
                                    and not (job_dir / "resume.pt").is_file()
                                    else None
                                ),
                            )
                        except SimulatedSmokeInterruption:
                            resume_events += 1
                            complete = _train_job(
                                context=context,
                                contract=contract,
                                run_kind=run_kind,
                                origin=origin,
                                geometry=geometry,
                                fold_entry=fold_entry,
                                seed=int(seed),
                                store=store,
                                scaler=scaler,
                                target_scale=float(scale_record["raw_return_rms_scale"]),
                                train_availability=data.train_availability,
                                validation_availability=data.validation_availability,
                                cell_hashes=cell_hashes,
                                effective=effective,
                                checkpoint_root=checkpoint_root,
                                device=device,
                            )
                        if not already_complete:
                            new_jobs += 1
                        jobs.append(complete)
                    data_audits.append(data.audit)
                    receipts.extend(access_record["receipts"])
                    scales.append(scale_record)
                    del store, data
                    gc.collect()
                    torch.mps.empty_cache()

    expected_jobs = (
        len(effective["origins"])
        * len(effective["geometries"])
        * len(effective["folds"])
        * len(effective["seeds"])
    )
    verified = _verify_tree(context, contract, checkpoint_root, effective, run_kind)
    total_steps = sum(int(job["train_optimizer_steps"]) for job in jobs)
    total_epochs = sum(int(job["completed_epochs"]) for job in jobs)
    execution_checks = {
        **contract_checks,
        "checkpoint_count_and_grid_are_exact": len(jobs) == len(verified) == expected_jobs,
        "all_jobs_restore_job_local_best": all(
            int(job["best_epoch"]) >= 1
            and int(job["best_epoch"]) <= int(job["completed_epochs"])
            for job in jobs
        ),
        "all_jobs_start_fresh_and_are_retained": all(
            job["base_metadata"]["initialization_state_sha256"]
            and not job["seed_fold_origin_geometry_selected"]
            for job in jobs
        ),
        "no_forbidden_outputs_or_operations": all(
            not job[flag]
            for job in jobs
            for flag in (
                "heldout_assets_loaded",
                "target_assets_loaded",
                "deployment_predictions_persisted",
                "predictive_metrics_computed",
                "policy_actions_computed",
                "performance_metrics_computed",
                "pnl_computed",
            )
        ),
        "data_access_is_train_asset_and_return_only": all(
            not audit["heldout_symbols_materialized"]
            and not audit["target_symbols_materialized"]
            and audit["label_columns_materialized"]
            == [training["data_access"]["return_column"]]
            and not audit["forbidden_label_columns_materialized"]
            for audit in data_audits
        ),
        "one_train_only_scaler_and_scale_per_cell": len(scales)
        == len(effective["origins"])
        * len(effective["geometries"])
        * len(effective["folds"])
        and all(
            record["fit_scope"]
            == "complete_train_only_date_triplet_raw_return_enumeration"
            and int(record["enumerated_triplets"]) > 0
            for record in scales
        ),
        "smoke_performs_exact_resume_once": mode != "smoke" or resume_events == 1,
        "full_grid_is_thirty_six": mode != "full" or expected_jobs == 36,
        "optimizer_step_count_is_positive_and_bounded": 0 < total_steps
        <= expected_jobs
        * int(effective["maximum_epochs"])
        * math.ceil(
            int(effective["train_samples_per_epoch"]) / int(effective["batch_size"])
        ),
        "no_resume_artifacts_remain": not any(checkpoint_root.rglob("resume.pt")),
    }
    execution_checks = {name: bool(value) for name, value in execution_checks.items()}
    if not all(execution_checks.values()):
        raise RuntimeError(f"V49 execution audit failed: {execution_checks}")

    manifest_rows = [
        {
            key: job[key]
            for key in (
                "origin",
                "geometry",
                "fold",
                "seed",
                "train_symbols",
                "test_symbols",
                "target_scale",
                "checkpoint_path",
                "checkpoint_sha256",
                "model_state_sha256",
                "best_epoch",
                "best_validation_total_loss",
                "completed_epochs",
                "train_optimizer_steps",
            )
        }
        for job in jobs
    ]
    decision = (
        "authorize_v49_full_thirty_six_job_training_only"
        if mode == "smoke"
        else "v49_training_complete_economic_evaluation_still_forbidden"
    )
    stable_result = {
        "version": "v49_smoke" if mode == "smoke" else "v49",
        "decision": decision,
        "training_spec": spec,
        "summary": {
            "checkpoint_count": len(jobs),
            "total_completed_epochs": total_epochs,
            "total_optimizer_steps": total_steps,
            "total_parameters": parameter_count,
            "origin_count": len(effective["origins"]),
            "geometry_count": len(effective["geometries"]),
            "fold_count": len(effective["folds"]),
            "seed_count": len(effective["seeds"]),
            "resume_events_proven": 1 if mode == "smoke" else 0,
        },
        "checkpoint_manifest": manifest_rows,
        "data_access_audit": {
            "cells": data_audits,
            "read_receipts": receipts,
            "physical_row_group_isolation_claimed": False,
        },
        "target_scales": scales,
        "environment": _environment(context),
        "tested": {
            "heldout_assets_loaded": False,
            "target_assets_loaded": False,
            "deployment_predictions_persisted": False,
            "predictive_metrics_computed": False,
            "policy_actions_computed": False,
            "performance_metrics_computed": False,
            "pnl_computed": False,
            "seed_fold_origin_geometry_selection_executed": False,
        },
        "audit": {"passed": True, "checks": execution_checks},
    }
    stable_result["result_sha256"] = _canonical_sha256(stable_result)
    output_relative = training["smoke_output_dir"] if mode == "smoke" else config["output_dir"]
    _write_result(root, output_relative, config, stable_result)
    return {
        **stable_result,
        "invocation": {
            "mode": mode,
            "new_jobs": new_jobs,
            "new_optimizer_steps": sum(
                int(job["train_optimizer_steps"])
                for job in jobs
                if job in jobs[-new_jobs:]
            )
            if new_jobs
            else 0,
            "resume_events": resume_events,
        },
    }
