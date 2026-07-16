from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import gc
from itertools import combinations
import math
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml

from .non_target_pretraining import TripletTensorStore
from .patch_transformer import MultiAssetPatchTransformer
from .ranking_excess_harness import (
    RANKING_EXCESS_HEADS,
    fit_triplet_excess_rms_scale,
    ranking_excess_loss,
)
from .ranking_excess_pretraining import (
    TARGET_SYMBOLS,
    _atomic_torch_save,
    _availability_from_index,
    _cpu_state_dict,
    _eligible_pair_count,
    _move_optimizer_state,
    _optimizer_contract,
    _restore_rng_state,
    _rng_state,
    _seed_device,
    _semantic_state_sha256,
    _serialize_filters,
    _state_is_finite,
    _state_subset_sha256,
    _to_cpu,
    _validate_optimizer_resume_state,
    load_ranking_excess_pretrained_checkpoint,
)
from .ranking_excess_spec import (
    _canonical_sha256,
    _load_json,
    _sha256_file,
    _write_json,
)
from .scientific_harness import (
    DeterministicEligibleTripletSampler,
    FeatureScaler,
)
from .supervised_non_target import model_state_sha256


FROZEN_SUPERVISED_PREFIXES = ("mask_token", "reconstruction_head.")
TEMPORAL_SUPERVISED_PREFIXES = (
    "temporal_position",
    "patch_projection.",
    "temporal_encoder.",
    "temporal_norm.",
)
CROSS_HEAD_PREFIXES = (
    "cross_asset_encoder.",
    "cross_asset_norm.",
    "prediction_heads.",
)
METADATA_INPUT_NAMES = {
    "v41_specification",
    "v41_blueprint",
    "v41_audit",
    "v42_result",
    "v42_audit",
    "v43_result",
    "v43_audit",
    "v43_checkpoint_manifest",
    "v32_dataset_manifest",
    "v32_feature_schema",
    "v32_asset_folds",
}
BINARY_INPUT_NAMES = {"panel", "sequence_index"}


@dataclass
class SupervisedEarlyStopping:
    patience: int
    best_loss: float = math.inf
    best_epoch: int = -1
    stale_epochs: int = 0
    should_stop: bool = False

    def update(self, epoch: int, loss: float) -> bool:
        if not math.isfinite(loss):
            raise ValueError("V44 validation core loss must be finite")
        if loss < self.best_loss:
            self.best_loss = float(loss)
            self.best_epoch = int(epoch)
            self.stale_epochs = 0
            self.should_stop = False
        else:
            self.stale_epochs += 1
            self.should_stop = self.stale_epochs >= self.patience
        return self.should_stop


@dataclass
class FoldSupervisedData:
    feature_panel: pd.DataFrame
    train_labels: pd.DataFrame
    validation_labels: pd.DataFrame
    train_availability: dict[pd.Timestamp, list[str]]
    validation_availability: dict[pd.Timestamp, list[str]]
    audit: dict[str, object]
    receipts: list[dict[str, object]]


class SupervisedFeatureLabelStore:
    def __init__(
        self,
        feature_panel: pd.DataFrame,
        label_frames: list[pd.DataFrame],
        feature_names: list[str],
        label_names: list[str],
        lookback_days: int,
        relative_source_feature: str,
    ) -> None:
        self.feature_store = TripletTensorStore(
            feature_panel[["date", "symbol", *feature_names]],
            feature_names,
            lookback_days,
            relative_source_feature,
        )
        labels = pd.concat(label_frames, ignore_index=True)
        if labels.duplicated(["date", "symbol"]).any():
            raise RuntimeError("V44 label frames overlap")
        self.label_names = tuple(label_names)
        self.labels = np.full(
            (
                len(self.feature_store.symbols),
                len(self.feature_store.dates),
                len(label_names),
            ),
            np.nan,
            dtype=np.float32,
        )
        for symbol, frame in labels.groupby("symbol", sort=True):
            symbol_index = self.feature_store.symbol_to_index[str(symbol)]
            date_indexes = np.asarray([
                self.feature_store.date_to_index[pd.Timestamp(date)]
                for date in frame["date"]
            ], dtype=np.int64)
            self.labels[symbol_index, date_indexes] = frame[label_names].to_numpy(
                dtype=np.float32
            )

    def materialize_batch(
        self,
        samples: list[dict[str, object]],
        scaler: FeatureScaler,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = self.feature_store.materialize_batch(samples, scaler)
        asset_indexes = np.asarray([
            [
                self.feature_store.symbol_to_index[str(symbol)]
                for symbol in sample["triplet"]
            ]
            for sample in samples
        ], dtype=np.int64)
        date_indexes = np.asarray([
            self.feature_store.date_to_index[pd.Timestamp(sample["date"])]
            for sample in samples
        ], dtype=np.int64)
        y = self.labels[asset_indexes, date_indexes[:, None], :]
        expected = (len(samples), 3, len(self.label_names))
        if y.shape != expected or not np.isfinite(y).all():
            raise RuntimeError("V44 materialized an invalid supervised label batch")
        return x, y.astype(np.float32, copy=False)


def supervised_parameter_names(
    model: MultiAssetPatchTransformer,
) -> list[str]:
    return [
        name
        for name, _ in model.named_parameters()
        if not name.startswith(FROZEN_SUPERVISED_PREFIXES)
    ]


def configure_supervised_scope(
    model: MultiAssetPatchTransformer,
) -> list[nn.Parameter]:
    allowed = set(supervised_parameter_names(model))
    parameters = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in allowed)
        if name in allowed:
            parameters.append(parameter)
    if not parameters:
        raise RuntimeError("V44 selected no supervised parameters")
    if any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith(FROZEN_SUPERVISED_PREFIXES)
    ):
        raise RuntimeError("V44 enabled a frozen reconstruction parameter")
    return parameters


def _mps_fallback_is_disabled() -> bool:
    return os.environ.get(
        "PYTORCH_ENABLE_MPS_FALLBACK", "0"
    ).strip().lower() in {"", "0", "false", "no"}


def _configure_device(name: str, torch_threads: int) -> torch.device:
    torch.set_num_threads(int(torch_threads))
    torch.use_deterministic_algorithms(True)
    if name != "mps":
        raise ValueError("V44 smoke/full require MPS with no CPU fallback")
    if not _mps_fallback_is_disabled():
        raise RuntimeError(
            "V44 forbids PYTORCH_ENABLE_MPS_FALLBACK during smoke/full"
        )
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was requested but is unavailable; run V44 outside the sandbox"
        )
    return torch.device("mps")


def _validate_frame_identity(
    frame: pd.DataFrame,
    columns: list[str],
    train_symbols: list[str],
    forbidden_symbols: set[str],
    label: str,
) -> None:
    if list(frame.columns) != columns:
        raise RuntimeError(f"V44 {label} projection drift")
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame["symbol"] = frame["symbol"].astype(str)
    if frame.duplicated(["date", "symbol"]).any():
        raise RuntimeError(f"V44 {label} contains duplicate keys")
    loaded = set(frame["symbol"].unique())
    if loaded != set(train_symbols):
        raise RuntimeError(f"V44 {label} ignored the fold symbol filter")
    if loaded.intersection(forbidden_symbols):
        raise RuntimeError(f"V44 {label} materialized a forbidden symbol")


def read_fold_supervised_data(
    panel_path: Path,
    sequence_path: Path,
    fold_entry: dict,
    data_access: dict,
    *,
    reader: Callable[..., pd.DataFrame] = pd.read_parquet,
) -> FoldSupervisedData:
    fold = int(fold_entry["fold"])
    train_symbols = sorted(fold_entry["train_symbols"])
    test_symbols = set(fold_entry["test_symbols"])
    if len(train_symbols) != 20 or len(test_symbols) != 10:
        raise RuntimeError("V44 fold cardinality drift")
    forbidden_symbols = test_symbols | TARGET_SYMBOLS
    if set(train_symbols).intersection(forbidden_symbols):
        raise RuntimeError("V44 fold contains held-out or target overlap")

    feature_start = pd.Timestamp(data_access["feature_start"], tz="UTC")
    feature_end = pd.Timestamp(data_access["feature_end"], tz="UTC")
    train_start = pd.Timestamp(data_access["supervised_train_start"], tz="UTC")
    train_end = pd.Timestamp(data_access["supervised_train_end"], tz="UTC")
    validation_start = pd.Timestamp(data_access["validation_start"], tz="UTC")
    validation_end = pd.Timestamp(data_access["validation_end"], tz="UTC")
    train_maturity_end = pd.Timestamp(
        data_access["supervised_train_maturity_end"], tz="UTC"
    )
    validation_maturity_end = pd.Timestamp(
        data_access["validation_maturity_end"], tz="UTC"
    )
    feature_columns = list(data_access["feature_columns"])
    label_columns = list(data_access["label_columns"])
    sequence_columns = list(data_access["sequence_columns"])

    feature_filters = [
        ("symbol", "in", train_symbols),
        ("date", ">=", feature_start),
        ("date", "<=", feature_end),
    ]
    train_filters = [
        ("symbol", "in", train_symbols),
        ("in_supervised_train", "==", True),
        ("supervised_sequence_ready", "==", True),
        ("label_complete", "==", True),
        ("date", ">=", train_start),
        ("date", "<=", train_end),
    ]
    validation_filters = [
        ("symbol", "in", train_symbols),
        ("in_validation", "==", True),
        ("supervised_sequence_ready", "==", True),
        ("label_complete", "==", True),
        ("date", ">=", validation_start),
        ("date", "<=", validation_end),
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
    receipts = [
        {
            "dataset": name,
            "columns": columns,
            "filters": _serialize_filters(filters),
        }
        for name, columns, filters in (
            ("panel_features", feature_columns, feature_filters),
            ("panel_labels_train", label_columns, train_filters),
            ("panel_labels_validation", label_columns, validation_filters),
            ("sequence_train", sequence_columns, train_filters),
            ("sequence_validation", sequence_columns, validation_filters),
        )
    ]

    for frame, columns, label in (
        (feature_panel, feature_columns, "feature panel"),
        (train_labels, label_columns, "train labels"),
        (validation_labels, label_columns, "validation labels"),
        (train_index, sequence_columns, "train sequence index"),
        (validation_index, sequence_columns, "validation sequence index"),
    ):
        _validate_frame_identity(
            frame, columns, train_symbols, forbidden_symbols, label
        )

    if (
        feature_panel["date"].min() < feature_start
        or feature_panel["date"].max() > feature_end
        or train_labels["date"].min() < train_start
        or train_labels["date"].max() > train_end
        or validation_labels["date"].min() < validation_start
        or validation_labels["date"].max() > validation_end
    ):
        raise RuntimeError("V44 panel reader ignored a date filter")

    label_names = label_columns[-2:]
    for frame, maturity_end, label in (
        (train_labels, train_maturity_end, "train"),
        (validation_labels, validation_maturity_end, "validation"),
    ):
        frame["target_window_end_date"] = pd.to_datetime(
            frame["target_window_end_date"], utc=True
        )
        if not bool(
            (frame["target_window_end_date"] == frame["date"] + pd.Timedelta(days=8)).all()
        ) or frame["target_window_end_date"].max() > maturity_end:
            raise RuntimeError(f"V44 {label} target maturity drift")
        values = frame[label_names].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or bool((values[:, 1] < 0).any()):
            raise RuntimeError(f"V44 {label} labels are invalid")

    for frame in (train_index, validation_index):
        starts = pd.to_datetime(frame["sequence_start_date"], utc=True)
        if not bool((starts == frame["date"] - pd.Timedelta(days=255)).all()):
            raise RuntimeError("V44 sequence lookback is not exactly 256 days")
    train_keys = set(zip(train_labels["date"], train_labels["symbol"], strict=True))
    validation_keys = set(zip(
        validation_labels["date"], validation_labels["symbol"], strict=True
    ))
    if train_keys != set(zip(train_index["date"], train_index["symbol"], strict=True)):
        raise RuntimeError("V44 train label/sequence keys drifted")
    if validation_keys != set(zip(
        validation_index["date"], validation_index["symbol"], strict=True
    )):
        raise RuntimeError("V44 validation label/sequence keys drifted")

    train_availability = _availability_from_index(train_index)
    validation_availability = _availability_from_index(validation_index)
    observed = {
        "feature_rows": len(feature_panel),
        "train_label_rows": len(train_labels),
        "validation_label_rows": len(validation_labels),
        "train_sequence_rows": len(train_index),
        "validation_sequence_rows": len(validation_index),
        "train_eligible_pairs": _eligible_pair_count(train_availability),
        "validation_eligible_pairs": _eligible_pair_count(validation_availability),
    }
    expected = {
        key: int(value)
        for key, value in data_access["expected_by_fold"][str(fold)].items()
    }
    if observed != expected:
        raise RuntimeError(
            f"V44 fold {fold} filtered-data counts drifted: {observed}"
        )
    audit = {
        "fold": fold,
        "train_symbols": train_symbols,
        "heldout_symbols_materialized": [],
        "target_symbols_materialized": [],
        "label_columns_materialized": label_names,
        "post_2024_signal_rows": 0,
        "post_2024_target_maturities": 0,
        "physical_row_group_isolation_claimed": False,
        **observed,
        "first_train_signal_date": min(train_availability).date().isoformat(),
        "last_train_signal_date": max(train_availability).date().isoformat(),
        "first_validation_signal_date": min(
            validation_availability
        ).date().isoformat(),
        "last_validation_signal_date": max(
            validation_availability
        ).date().isoformat(),
        "maximum_train_target_maturity": train_labels[
            "target_window_end_date"
        ].max().date().isoformat(),
        "maximum_validation_target_maturity": validation_labels[
            "target_window_end_date"
        ].max().date().isoformat(),
    }
    return FoldSupervisedData(
        feature_panel=feature_panel,
        train_labels=train_labels,
        validation_labels=validation_labels,
        train_availability=train_availability,
        validation_availability=validation_availability,
        audit=audit,
        receipts=receipts,
    )


def fit_fold_excess_scale(
    train_labels: pd.DataFrame,
    train_availability: dict[pd.Timestamp, list[str]],
    floor: float,
) -> dict[str, object]:
    returns = {
        (pd.Timestamp(row.date), str(row.symbol)): float(
            row.target_next_open_to_next_open_log_return
        )
        for row in train_labels.itertuples(index=False)
    }
    triplet_count = _eligible_pair_count(train_availability)
    enumerated = np.empty((triplet_count, 3), dtype=np.float64)
    cursor = 0
    for date, symbols in sorted(train_availability.items()):
        for triplet in combinations(sorted(symbols), 3):
            enumerated[cursor] = [returns[(date, symbol)] for symbol in triplet]
            cursor += 1
    if cursor != triplet_count or not np.isfinite(enumerated).all():
        raise RuntimeError("V44 target-scale enumeration drift")
    tensor = torch.from_numpy(enumerated)
    scale = fit_triplet_excess_rms_scale(
        tensor,
        torch.ones(len(tensor), dtype=torch.bool),
        float(floor),
    )
    record = {
        "fit_scope": "full_eligible_lexical_triplet_enumeration_train_only",
        "fit_start": min(train_availability).date().isoformat(),
        "fit_end": max(train_availability).date().isoformat(),
        "eligible_dates": len(train_availability),
        "enumerated_triplets": triplet_count,
        "enumerated_excess_values": triplet_count * 3,
        "scale_floor": float(floor),
        "excess_rms_scale": float(scale),
    }
    record["target_scale_state_sha256"] = _canonical_sha256(record)
    return record


def _validated_loss(
    output: dict[str, torch.Tensor],
    labels: torch.Tensor,
    scale: float,
    objective: dict,
) -> dict[str, torch.Tensor]:
    if set(output) != set(RANKING_EXCESS_HEADS):
        raise RuntimeError("V44 output-head contract drift")
    if labels.ndim != 3 or tuple(labels.shape[1:]) != (3, 2):
        raise RuntimeError("V44 label shape drift")
    if any(
        output[name].shape != labels.shape[:2]
        or not bool(torch.isfinite(output[name]).all())
        for name in RANKING_EXCESS_HEADS
    ) or not bool(torch.isfinite(labels).all()):
        raise RuntimeError("V44 received non-finite or malformed tensors")
    if (
        not math.isfinite(scale)
        or scale <= 0
        or float(objective["ranking_weight"]) != 1.0
        or float(objective["excess_weight"]) != 1.0
        or float(objective["log_volatility_weight"]) != 0.1
        or float(objective["volatility_floor"]) <= 0
        or float(objective["exact_tie_tolerance"]) < 0
        or bool((labels[..., 1] < 0).any())
    ):
        raise RuntimeError("V44 objective contract drift")
    return ranking_excess_loss(
        output,
        labels,
        scale,
        tie_tolerance=float(objective["exact_tie_tolerance"]),
        volatility_floor=float(objective["volatility_floor"]),
        ranking_weight=float(objective["ranking_weight"]),
        excess_weight=float(objective["excess_weight"]),
        volatility_weight=float(objective["log_volatility_weight"]),
    )


def _run_supervised_batches(
    model: MultiAssetPatchTransformer,
    store: SupervisedFeatureLabelStore,
    scaler: FeatureScaler,
    samples: list[dict[str, object]],
    *,
    batch_size: int,
    target_scale: float,
    objective: dict,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float,
) -> tuple[dict[str, float], int]:
    training = optimizer is not None
    model.train(training)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    ranking_sum = 0.0
    pair_count = 0
    sample_sums = {"excess": 0.0, "log_volatility": 0.0}
    observations = 0
    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        x_np, y_np = store.materialize_batch(batch_samples, scaler)
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            losses = _validated_loss(model(x), y, target_scale, objective)
        scalar_names = ("ranking", "excess", "log_volatility", "core", "total")
        if any(not bool(torch.isfinite(losses[name])) for name in scalar_names):
            raise RuntimeError("V44 produced a non-finite supervised loss")
        if training:
            losses["total"].backward()
            if any(
                parameter.grad is None
                or not bool(torch.isfinite(parameter.grad).all())
                for parameter in trainable
            ):
                raise RuntimeError("V44 produced missing or non-finite gradients")
            if any(
                parameter.grad is not None
                for name, parameter in model.named_parameters()
                if name.startswith(FROZEN_SUPERVISED_PREFIXES)
            ):
                raise RuntimeError("V44 produced a frozen reconstruction gradient")
            gradient_norm = nn.utils.clip_grad_norm_(
                trainable, float(gradient_clip_norm)
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("V44 produced a non-finite gradient norm")
            optimizer.step()
        batch_pairs = int(losses["pair_count"].detach().cpu())
        ranking_sum += float(losses["ranking"].detach().cpu()) * batch_pairs
        pair_count += batch_pairs
        count = len(batch_samples)
        for name in sample_sums:
            sample_sums[name] += float(losses[name].detach().cpu()) * count
        observations += count
    if observations == 0 or pair_count == 0:
        raise RuntimeError("V44 supervised batches contain no observations or pairs")
    ranking = ranking_sum / pair_count
    excess = sample_sums["excess"] / observations
    log_volatility = sample_sums["log_volatility"] / observations
    core = ranking + excess
    total = core + float(objective["log_volatility_weight"]) * log_volatility
    return {
        "ranking": ranking,
        "excess": excess,
        "log_volatility": log_volatility,
        "core": core,
        "total": total,
        "pair_count": pair_count,
        "observations": observations,
    }, math.ceil(observations / batch_size)


def _scaler_from_record(record: dict) -> FeatureScaler:
    names = {field.name for field in fields(FeatureScaler)}
    values = {name: record[name] for name in names}
    values["feature_names"] = tuple(values["feature_names"])
    values["mean"] = tuple(values["mean"])
    values["scale"] = tuple(values["scale"])
    return FeatureScaler(**values)


def build_supervised_spec(
    blueprint: dict,
    supervised: dict,
    mode: str,
    prior_gate: dict[str, object] | None = None,
) -> dict[str, object]:
    if mode not in {"preflight", "smoke", "full"}:
        raise ValueError("V44 mode must be preflight, smoke, or full")
    effective = supervised["smoke"] if mode == "smoke" else supervised["full_run"]
    spec = {
        "version": "v44",
        "candidate_family_id": blueprint["candidate_family_id"],
        "mode": mode,
        "phase": "supervised_non_target_ranking_excess_training",
        "architecture": blueprint["architecture"],
        "folds": list(effective["folds"]),
        "seeds": list(effective["seeds"]),
        "supervised_train_window": blueprint["chronological_splits"][
            "supervised_train"
        ],
        "early_stopping_window": blueprint["chronological_splits"][
            "early_stopping_train_assets_only"
        ],
        "train_samples_per_epoch": int(effective["train_samples_per_epoch"]),
        "validation_samples": int(effective["validation_samples"]),
        "validation_sampling_epoch": int(supervised["validation_sampling_epoch"]),
        "batch_size": int(effective["batch_size"]),
        "maximum_epochs": int(effective["maximum_epochs"]),
        "early_stopping_patience": int(effective["early_stopping_patience"]),
        "optimizer": supervised["optimizer"],
        "objective": supervised["objective"],
        "parameter_scope": supervised["parameter_scope"],
        "data_access": supervised["data_access"],
        "initialization": supervised["initialization"],
        "expected_parents": supervised["expected_parents"],
        "expected_scalers": supervised["expected_scalers"],
        "sampling": supervised["sampling"],
        "early_stopping": supervised["early_stopping"],
        "device": supervised["device"],
        "dtype": supervised["dtype"],
        "amp": supervised["amp"],
        "deterministic_algorithms": supervised["deterministic_algorithms"],
        "cpu_fallback_allowed": supervised["cpu_fallback_allowed"],
        "checkpoint_format": supervised["checkpoint_format"],
        "resume_format": supervised["resume_format"],
        "best_state_format": supervised["best_state_format"],
        "seed_selection_allowed": False,
        "fold_selection_allowed": False,
        "heldout_assets_loaded": False,
        "target_assets_loaded": False,
        "development_2025_loaded": False,
        "performance_metrics_allowed": False,
        "pnl_allowed": False,
        "prior_gate": prior_gate,
    }
    spec["supervised_spec_sha256"] = _canonical_sha256(spec)
    return spec


def _metadata_context(config: dict) -> dict[str, object]:
    supervised = config["ranking_excess_supervised"]
    root = Path(supervised["project_root"]).resolve()
    paths = {
        name: (root / relative).resolve()
        for name, relative in supervised["inputs"].items()
    }
    if set(paths) != METADATA_INPUT_NAMES | BINARY_INPUT_NAMES:
        raise RuntimeError("V44 input allowlist drift")
    for name in METADATA_INPUT_NAMES:
        path = paths[name]
        if (
            not path.is_file()
            or _sha256_file(path) != supervised["expected_input_sha256"][name]
        ):
            raise RuntimeError(f"V44 metadata input drift: {name}")
    for name in BINARY_INPUT_NAMES:
        if not paths[name].is_file():
            raise RuntimeError(f"V44 binary input is missing: {name}")

    values = {name: _load_json(paths[name]) for name in METADATA_INPUT_NAMES}
    blueprint = values["v41_blueprint"]
    v43_result = values["v43_result"]
    if (
        values["v41_specification"]["decision"]
        != "authorize_v42_synthetic_ranking_excess_harness_only"
        or not values["v41_audit"]["passed"]
        or values["v42_result"]["decision"]
        != "authorize_v43_medium_non_target_pretraining_only"
        or not values["v42_audit"]["passed"]
        or v43_result["decision"]
        != "v44_ranking_excess_supervised_non_target_only"
        or not values["v43_audit"]["passed"]
        or v43_result["pretraining_spec"]["mode"] != "full"
        or int(v43_result["summary"]["checkpoint_count"]) != 9
    ):
        raise RuntimeError("V41/V42/V43 do not authorize V44")

    manifest = values["v32_dataset_manifest"]
    feature_schema = values["v32_feature_schema"]
    asset_folds = values["v32_asset_folds"]
    data_access = supervised["data_access"]
    expected_features = list(manifest["panel_features"])
    expected_labels = list(manifest["labels"])
    if (
        manifest["panel_sha256"] != supervised["expected_input_sha256"]["panel"]
        or manifest["sequence_index_sha256"]
        != supervised["expected_input_sha256"]["sequence_index"]
        or list(feature_schema["model_feature_order"][:-1]) != expected_features
        or list(data_access["feature_columns"])
        != ["date", "symbol", *expected_features]
        or list(data_access["label_columns"])
        != ["date", "symbol", "target_window_end_date", *expected_labels]
        or list(data_access["sequence_columns"])
        != ["date", "sequence_start_date", "symbol"]
    ):
        raise RuntimeError("V44 dataset projection or hash drift")
    if TARGET_SYMBOLS.intersection(manifest["symbols"]):
        raise RuntimeError("V44 source dataset contains target assets")
    if len(asset_folds["folds"]) != 3:
        raise RuntimeError("V44 requires exactly three asset folds")
    if (
        not data_access["per_fold_filtered_read_required"]
        or data_access["scaler_refit_allowed"]
        or not data_access["train_asset_labels_allowed"]
        or not data_access["validation_train_asset_labels_allowed"]
        or data_access["heldout_asset_features_allowed"]
        or data_access["heldout_asset_labels_allowed"]
        or data_access["target_assets_allowed"]
        or data_access["post_2024_signal_dates_allowed"]
        or data_access["post_2024_target_maturity_allowed"]
        or data_access["development_2025_allowed"]
        or data_access["physical_row_group_isolation_claimed"]
    ):
        raise RuntimeError("V44 data-access permission drift")
    if (
        data_access["supervised_train_start"]
        != blueprint["chronological_splits"]["supervised_train"][0]
        or data_access["supervised_train_end"]
        != blueprint["chronological_splits"]["supervised_train"][1]
        or data_access["validation_start"]
        != blueprint["chronological_splits"][
            "early_stopping_train_assets_only"
        ][0]
        or data_access["validation_end"]
        != blueprint["chronological_splits"][
            "early_stopping_train_assets_only"
        ][1]
    ):
        raise RuntimeError("V44 chronology drift")

    folds_by_number = {
        int(entry["fold"]): entry for entry in asset_folds["folds"]
    }
    parent_rows = values["v43_checkpoint_manifest"]
    if len(parent_rows) != int(supervised["expected_parent_checkpoints"]):
        raise RuntimeError("V44 parent checkpoint count drift")
    parents: dict[tuple[int, int], dict[str, object]] = {}
    expected_parent_root = (
        root / "data/checkpoints/v43_ranking_excess_pretraining"
    ).resolve()
    v43_spec_hash = v43_result["pretraining_spec"]["pretraining_spec_sha256"]
    for row in parent_rows:
        fold = int(row["fold"])
        seed = int(row["seed"])
        key = f"{fold}:{seed}"
        checkpoint_path = Path(row["checkpoint_path"]).resolve()
        if (
            key not in supervised["expected_parents"]
            or row["checkpoint_sha256"] != supervised["expected_parents"][key]
            or not checkpoint_path.is_relative_to(expected_parent_root)
            or not checkpoint_path.is_file()
            or _sha256_file(checkpoint_path) != row["checkpoint_sha256"]
            or (fold, seed) in parents
        ):
            raise RuntimeError(f"V44 parent checkpoint drift: {key}")
        model, payload = load_ranking_excess_pretrained_checkpoint(
            checkpoint_path,
            expected_architecture=blueprint["architecture"],
        )
        metadata = payload["metadata"]
        fold_entry = folds_by_number[fold]
        scaler_expected = supervised["expected_scalers"][str(fold)]
        if (
            int(metadata["fold"]) != fold
            or int(metadata["initialization_seed"]) != seed
            or payload["model_state_sha256"] != row["model_state_sha256"]
            or metadata["pretraining_spec_sha256"] != v43_spec_hash
            or metadata["checkpoint_status"]
            != "frozen_pretrained_no_seed_or_fold_selection"
            or metadata["train_symbols"] != sorted(fold_entry["train_symbols"])
            or metadata["test_symbols"] != sorted(fold_entry["test_symbols"])
            or metadata["scaler_state_sha256"]
            != scaler_expected["state_sha256"]
            or metadata["scaler_artifact_sha256"]
            != scaler_expected["artifact_sha256"]
        ):
            raise RuntimeError(f"V44 parent semantic drift: {key}")
        parents[(fold, seed)] = {
            "row": row,
            "path": checkpoint_path,
            "model_state_sha256": payload["model_state_sha256"],
            "frozen_state_sha256": _state_subset_sha256(
                model, FROZEN_SUPERVISED_PREFIXES
            ),
            "temporal_state_sha256": _state_subset_sha256(
                model, TEMPORAL_SUPERVISED_PREFIXES
            ),
            "cross_head_state_sha256": _state_subset_sha256(
                model, CROSS_HEAD_PREFIXES
            ),
        }
        del model, payload

    expected_grid = {
        (fold, seed) for fold in (1, 2, 3) for seed in (42, 7, 123)
    }
    if set(parents) != expected_grid:
        raise RuntimeError("V44 parent fold/seed grid drift")
    scalers = {}
    for fold, expected in supervised["expected_scalers"].items():
        path = expected_parent_root / f"fold_{fold}" / "scaler.json"
        if not path.is_file() or _sha256_file(path) != expected["artifact_sha256"]:
            raise RuntimeError(f"V44 scaler artifact drift: {fold}")
        record = _load_json(path)
        scaler = _scaler_from_record(record)
        if (
            record.get("scaler_state_sha256") != expected["state_sha256"]
            or scaler.state_sha256() != expected["state_sha256"]
            or scaler.fit_scope != "representation_train_only"
            or scaler.fit_start != data_access["feature_start"]
            or scaler.fit_end != "2023-12-31"
        ):
            raise RuntimeError(f"V44 scaler semantic drift: {fold}")
        scalers[int(fold)] = {"path": path, "record": record, "scaler": scaler}

    return {
        "root": root,
        "paths": paths,
        "supervised": supervised,
        "blueprint": blueprint,
        "manifest": manifest,
        "feature_schema": feature_schema,
        "asset_folds": asset_folds,
        "parents": parents,
        "scalers": scalers,
        "v43_result": v43_result,
    }


def _scope_audit(
    architecture: dict,
) -> tuple[dict[str, bool], int, int, int, list[str]]:
    model = MultiAssetPatchTransformer(
        9,
        architecture,
        expected_prediction_heads=RANKING_EXCESS_HEADS,
    )
    parameters = configure_supervised_scope(model)
    names = supervised_parameter_names(model)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in parameters)
    frozen = total - trainable
    checks = {
        "cross_asset_and_heads_are_trainable": all(
            parameter.requires_grad
            for name, parameter in model.named_parameters()
            if name.startswith(CROSS_HEAD_PREFIXES)
        ),
        "mask_and_reconstruction_are_frozen": all(
            not parameter.requires_grad
            for name, parameter in model.named_parameters()
            if name.startswith(FROZEN_SUPERVISED_PREFIXES)
        ),
    }
    return checks, total, trainable, frozen, names


def _save_resume(
    path: Path,
    *,
    model: MultiAssetPatchTransformer,
    optimizer: torch.optim.Optimizer,
    early_stopping: SupervisedEarlyStopping,
    history: list[dict[str, object]],
    completed_epoch: int,
    metadata: dict[str, object],
    best_model_state: dict[str, torch.Tensor],
    device: torch.device,
    format_version: str,
    best_state_format: str,
    optimizer_parameter_names: list[str],
) -> None:
    optimizer_state = _to_cpu(optimizer.state_dict())
    early_state = asdict(early_stopping)
    rng_state = _rng_state(device)
    _atomic_torch_save(
        {
            "format_version": format_version,
            "best_state_format": best_state_format,
            "metadata": metadata,
            "completed_epoch": int(completed_epoch),
            "model_state_dict": _cpu_state_dict(model.state_dict()),
            "model_state_sha256": model_state_sha256(model.state_dict()),
            "best_model_state_dict": _cpu_state_dict(best_model_state),
            "best_model_state_sha256": model_state_sha256(best_model_state),
            "optimizer_state_dict": optimizer_state,
            "optimizer_state_sha256": _semantic_state_sha256(optimizer_state),
            "optimizer_contract": _optimizer_contract(
                optimizer, optimizer_parameter_names
            ),
            "early_stopping": early_state,
            "early_stopping_sha256": _semantic_state_sha256(early_state),
            "history": history,
            "history_sha256": _semantic_state_sha256(history),
            "rng_state": rng_state,
            "rng_state_sha256": _semantic_state_sha256(rng_state),
        },
        path,
    )


def _load_resume(
    path: Path,
    *,
    model: MultiAssetPatchTransformer,
    optimizer: torch.optim.Optimizer,
    expected_metadata: dict[str, object],
    device: torch.device,
    format_version: str,
    expected_best_state_format: str,
    expected_patience: int,
    maximum_epochs: int,
    expected_optimizer_parameter_names: list[str],
    expected_train_observations: int,
    expected_validation_observations: int,
    expected_train_steps: int,
    expected_validation_steps: int,
    expected_volatility_weight: float,
) -> tuple[
    int,
    SupervisedEarlyStopping,
    list[dict[str, object]],
    dict[str, torch.Tensor],
]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != format_version:
        raise RuntimeError("Unsupported or old V44 resume checkpoint")
    if payload.get("best_state_format") != expected_best_state_format:
        raise RuntimeError("Unsupported or old V44 best-state checkpoint")
    if payload.get("metadata") != expected_metadata:
        raise RuntimeError("V44 resume metadata drift")
    completed_epoch = int(payload.get("completed_epoch", -1))
    history = list(payload.get("history", []))
    if _semantic_state_sha256(history) != payload.get("history_sha256"):
        raise RuntimeError("V44 resume history hash drift")
    epochs = [int(row.get("epoch", -1)) for row in history]
    validation_core = [
        float(row.get("validation_losses", {}).get("core", math.nan))
        for row in history
    ]
    optimizer_steps = [
        int(row.get("train_optimizer_steps", -1)) for row in history
    ]
    validation_steps = [
        int(row.get("validation_steps", -1)) for row in history
    ]
    loss_names = {
        "ranking",
        "excess",
        "log_volatility",
        "core",
        "total",
        "pair_count",
        "observations",
    }
    history_losses_are_exact = True
    prior_best = math.inf
    for row in history:
        train_losses = row.get("train_losses", {})
        validation_losses = row.get("validation_losses", {})
        if set(train_losses) != loss_names or set(validation_losses) != loss_names:
            history_losses_are_exact = False
            break
        for losses, expected_observations in (
            (train_losses, int(expected_train_observations)),
            (validation_losses, int(expected_validation_observations)),
        ):
            scalar_values = [
                float(losses[name])
                for name in (
                    "ranking",
                    "excess",
                    "log_volatility",
                    "core",
                    "total",
                )
            ]
            observations = int(losses["observations"])
            pairs = int(losses["pair_count"])
            if (
                not all(math.isfinite(value) for value in scalar_values)
                or observations != expected_observations
                or pairs <= 0
                or pairs > observations * 3
                or not math.isclose(
                    float(losses["core"]),
                    float(losses["ranking"]) + float(losses["excess"]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    float(losses["total"]),
                    float(losses["core"])
                    + float(expected_volatility_weight)
                    * float(losses["log_volatility"]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            ):
                history_losses_are_exact = False
                break
        validation_core_value = float(validation_losses.get("core", math.nan))
        expected_improved = validation_core_value < prior_best
        if row.get("improved") is not expected_improved:
            history_losses_are_exact = False
        if expected_improved:
            prior_best = validation_core_value
    if (
        completed_epoch < 1
        or completed_epoch > int(maximum_epochs)
        or epochs != list(range(1, completed_epoch + 1))
        or any(not math.isfinite(loss) for loss in validation_core)
        or any(steps != int(expected_train_steps) for steps in optimizer_steps)
        or any(
            steps != int(expected_validation_steps) for steps in validation_steps
        )
        or not history_losses_are_exact
    ):
        raise RuntimeError("V44 resume epoch/history coherence drift")
    early = SupervisedEarlyStopping(**payload["early_stopping"])
    if _semantic_state_sha256(payload["early_stopping"]) != payload.get(
        "early_stopping_sha256"
    ):
        raise RuntimeError("V44 resume early-stopping hash drift")
    minimum_loss = min(validation_core)
    minimum_epoch = epochs[validation_core.index(minimum_loss)]
    if (
        early.best_epoch != minimum_epoch
        or not math.isclose(early.best_loss, minimum_loss)
        or early.patience != int(expected_patience)
        or early.stale_epochs != completed_epoch - early.best_epoch
        or early.should_stop != (early.stale_epochs >= early.patience)
    ):
        raise RuntimeError("V44 resume early-stopping coherence drift")
    for state_key, hash_key in (
        ("model_state_dict", "model_state_sha256"),
        ("best_model_state_dict", "best_model_state_sha256"),
    ):
        state = payload[state_key]
        if (
            model_state_sha256(state) != payload.get(hash_key)
            or not _state_is_finite(state)
        ):
            raise RuntimeError("V44 resume model-state drift")
    expected_contract = _optimizer_contract(
        optimizer, expected_optimizer_parameter_names
    )
    if payload.get("optimizer_contract") != expected_contract:
        raise RuntimeError("V44 resume optimizer contract drift")
    _validate_optimizer_resume_state(
        payload.get("optimizer_state_dict"),
        optimizer,
        expected_optimizer_parameter_names,
        sum(optimizer_steps),
    )
    if _semantic_state_sha256(payload["optimizer_state_dict"]) != payload.get(
        "optimizer_state_sha256"
    ):
        raise RuntimeError("V44 resume optimizer semantic hash drift")
    rng_state = payload.get("rng_state")
    expected_rng_keys = {"cpu", "mps"} if device.type == "mps" else {"cpu"}
    if (
        not isinstance(rng_state, dict)
        or set(rng_state) != expected_rng_keys
        or any(
            not isinstance(state, torch.Tensor)
            or state.dtype != torch.uint8
            or state.ndim != 1
            or state.numel() == 0
            for state in rng_state.values()
        )
        or _semantic_state_sha256(rng_state)
        != payload.get("rng_state_sha256")
    ):
        raise RuntimeError("V44 resume RNG-state drift")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    _move_optimizer_state(optimizer, device)
    _restore_rng_state(rng_state, device)
    return (
        completed_epoch,
        early,
        history,
        _cpu_state_dict(payload["best_model_state_dict"]),
    )


def load_ranking_excess_supervised_checkpoint(
    path: str | Path,
    *,
    expected_architecture: dict | None = None,
    expected_metadata: dict | None = None,
) -> tuple[MultiAssetPatchTransformer, dict]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format_version") != "v44_ranking_excess_supervised_v1":
        raise RuntimeError("Unsupported or old V44 supervised checkpoint")
    architecture = payload["architecture"]
    if payload.get("architecture_sha256") != _canonical_sha256(architecture):
        raise RuntimeError("V44 checkpoint architecture hash drift")
    if expected_architecture is not None and architecture != expected_architecture:
        raise RuntimeError("V44 checkpoint architecture differs from V41")
    if int(payload.get("input_features", -1)) != 9:
        raise RuntimeError("V44 checkpoint input-feature drift")
    if expected_metadata is not None and payload.get("metadata") != expected_metadata:
        raise RuntimeError("V44 checkpoint metadata drift")
    state = payload["state_dict"]
    if (
        model_state_sha256(state) != payload.get("model_state_sha256")
        or not _state_is_finite(state)
    ):
        raise RuntimeError("V44 checkpoint model-state drift")
    metadata = payload.get("metadata", {})
    if (
        metadata.get("initialization_source")
        != "exact_matching_v43_fold_seed_checkpoint"
        or not metadata.get("parent_v43_checkpoint_sha256")
        or metadata.get("checkpoint_status")
        != "frozen_supervised_no_seed_or_fold_selection"
        or set(metadata.get("unused_during_supervised_training", []))
        != {"mask_token", "reconstruction_head"}
    ):
        raise RuntimeError("V44 checkpoint semantic metadata drift")
    model = MultiAssetPatchTransformer(
        9,
        architecture,
        expected_prediction_heads=RANKING_EXCESS_HEADS,
    )
    parameters = configure_supervised_scope(model)
    if int(metadata.get("supervised_parameter_count", -1)) != sum(
        parameter.numel() for parameter in parameters
    ):
        raise RuntimeError("V44 checkpoint supervised scope drift")
    model.load_state_dict(state)
    if _state_subset_sha256(model, FROZEN_SUPERVISED_PREFIXES) != metadata.get(
        "frozen_state_sha256"
    ):
        raise RuntimeError("V44 checkpoint frozen-state hash drift")
    return model, payload


def _load_prior_gate(
    root: Path,
    supervised: dict,
    blueprint: dict,
    mode: str,
) -> dict[str, object] | None:
    if mode == "preflight":
        return None
    if mode == "smoke":
        relative = supervised["preflight_output_dir"]
        expected_mode = "preflight"
        expected_decision = "authorize_v44_one_job_mps_smoke_only"
        nested_gate = None
    else:
        relative = supervised["smoke_output_dir"]
        expected_mode = "smoke"
        expected_decision = "authorize_v44_full_nine_job_supervised_only"
        nested_gate = _load_prior_gate(root, supervised, blueprint, "smoke")
    path = root / relative / "result.json"
    if not path.is_file():
        raise RuntimeError(f"V44 {mode} requires its prior passing gate")
    result = _load_json(path)
    expected_spec = build_supervised_spec(
        blueprint,
        supervised,
        expected_mode,
        prior_gate=nested_gate,
    )
    checks = result.get("audit", {}).get("checks", {})
    if (
        result.get("decision") != expected_decision
        or result.get("supervised_spec") != expected_spec
        or not result.get("audit", {}).get("passed")
        or not checks
        or not all(bool(value) for value in checks.values())
    ):
        raise RuntimeError(f"V44 prior {expected_mode} gate is invalid")
    summary = result.get("summary", {})
    if (
        int(summary.get("total_parameters", -1))
        != int(supervised["expected_total_parameters"])
        or int(summary.get("supervised_parameters", -1))
        != int(supervised["expected_supervised_parameters"])
        or int(summary.get("frozen_parameters", -1))
        != int(supervised["expected_frozen_parameters"])
    ):
        raise RuntimeError(f"V44 prior {expected_mode} parameter audit drift")
    if expected_mode == "preflight":
        if (
            int(summary.get("parent_checkpoint_count", -1)) != 9
            or int(summary.get("checkpoint_count", -1)) != 0
            or int(summary.get("total_optimizer_steps", -1)) != 0
            or int(summary.get("parquet_files_deserialized", -1)) != 0
            or int(summary.get("label_rows_materialized", -1)) != 0
            or result.get("tested", {}).get("optimizer_executed") is not False
            or result.get("tested", {}).get("panel_or_sequence_deserialized")
            is not False
        ):
            raise RuntimeError("V44 fabricated or contaminated preflight gate")
    else:
        manifest = result.get("checkpoint_manifest", [])
        if (
            int(summary.get("checkpoint_count", -1)) != 1
            or int(summary.get("total_optimizer_steps", -1)) != 8
            or len(manifest) != 1
            or int(manifest[0].get("fold", -1)) != 1
            or int(manifest[0].get("seed", -1)) != 42
            or int(manifest[0].get("completed_epochs", -1)) != 2
            or int(manifest[0].get("train_optimizer_steps", -1)) != 8
        ):
            raise RuntimeError("V44 smoke gate does not contain exactly one job")
        checkpoint_path = Path(manifest[0]["checkpoint_path"]).resolve()
        expected_root = (root / supervised["smoke_checkpoint_dir"]).resolve()
        if (
            not checkpoint_path.is_relative_to(expected_root)
            or not checkpoint_path.is_file()
            or _sha256_file(checkpoint_path)
            != manifest[0].get("checkpoint_sha256")
        ):
            raise RuntimeError("V44 smoke checkpoint drift")
        _, payload = load_ranking_excess_supervised_checkpoint(
            checkpoint_path,
            expected_architecture=blueprint["architecture"],
        )
        if payload["metadata"].get("supervised_spec_sha256") != expected_spec[
            "supervised_spec_sha256"
        ]:
            raise RuntimeError("V44 smoke checkpoint spec drift")
        tested = result.get("tested", {})
        if any(
            tested.get(name) is not False
            for name in (
                "heldout_assets_loaded",
                "target_assets_loaded",
                "development_2025_loaded",
                "predictions_computed",
                "performance_metrics_computed",
                "pnl_computed",
                "seed_or_fold_selection_executed",
            )
        ):
            raise RuntimeError("V44 smoke gate contains a forbidden operation")
    return {
        "mode": expected_mode,
        "decision": expected_decision,
        "result_sha256": _sha256_file(path),
        "supervised_spec_sha256": result["supervised_spec"][
            "supervised_spec_sha256"
        ],
    }


def _validate_persisted_data_access(
    record: dict,
    fold_entry: dict,
    data_access: dict,
) -> None:
    fold = int(fold_entry["fold"])
    audit = record.get("audit", {})
    receipts = record.get("receipts", [])
    expected = {
        key: int(value)
        for key, value in data_access["expected_by_fold"][str(fold)].items()
    }
    by_dataset = {receipt.get("dataset"): receipt for receipt in receipts}
    expected_columns = {
        "panel_features": list(data_access["feature_columns"]),
        "panel_labels_train": list(data_access["label_columns"]),
        "panel_labels_validation": list(data_access["label_columns"]),
        "sequence_train": list(data_access["sequence_columns"]),
        "sequence_validation": list(data_access["sequence_columns"]),
    }
    train_symbols = sorted(fold_entry["train_symbols"])
    feature_filters = _serialize_filters([
        ("symbol", "in", train_symbols),
        ("date", ">=", pd.Timestamp(data_access["feature_start"], tz="UTC")),
        ("date", "<=", pd.Timestamp(data_access["feature_end"], tz="UTC")),
    ])
    train_filters = _serialize_filters([
        ("symbol", "in", train_symbols),
        ("in_supervised_train", "==", True),
        ("supervised_sequence_ready", "==", True),
        ("label_complete", "==", True),
        (
            "date",
            ">=",
            pd.Timestamp(data_access["supervised_train_start"], tz="UTC"),
        ),
        (
            "date",
            "<=",
            pd.Timestamp(data_access["supervised_train_end"], tz="UTC"),
        ),
    ])
    validation_filters = _serialize_filters([
        ("symbol", "in", train_symbols),
        ("in_validation", "==", True),
        ("supervised_sequence_ready", "==", True),
        ("label_complete", "==", True),
        (
            "date",
            ">=",
            pd.Timestamp(data_access["validation_start"], tz="UTC"),
        ),
        (
            "date",
            "<=",
            pd.Timestamp(data_access["validation_end"], tz="UTC"),
        ),
    ])
    expected_filters = {
        "panel_features": feature_filters,
        "panel_labels_train": train_filters,
        "panel_labels_validation": validation_filters,
        "sequence_train": train_filters,
        "sequence_validation": validation_filters,
    }
    if (
        int(audit.get("fold", -1)) != fold
        or audit.get("train_symbols") != train_symbols
        or audit.get("heldout_symbols_materialized")
        or audit.get("target_symbols_materialized")
        or audit.get("label_columns_materialized")
        != list(data_access["label_columns"][-2:])
        or int(audit.get("post_2024_signal_rows", -1)) != 0
        or int(audit.get("post_2024_target_maturities", -1)) != 0
        or any(int(audit.get(key, -1)) != value for key, value in expected.items())
        or len(receipts) != 5
        or set(by_dataset) != set(expected_columns)
        or any(
            by_dataset[name].get("columns") != columns
            or int(by_dataset[name].get("fold", -1)) != fold
            or by_dataset[name].get("filters") != expected_filters[name]
            for name, columns in expected_columns.items()
        )
    ):
        raise RuntimeError(f"Persisted V44 fold data audit drifted: {fold}")


def _validate_scaler_reference(
    path: Path,
    fold: int,
    supervised: dict,
) -> dict:
    record = _load_json(path)
    expected = supervised["expected_scalers"][str(fold)]
    source = Path(str(record.get("source_path", "")))
    if (
        int(record.get("fold", -1)) != fold
        or record.get("artifact_sha256") != expected["artifact_sha256"]
        or record.get("state_sha256") != expected["state_sha256"]
        or not source.is_file()
        or _sha256_file(source) != expected["artifact_sha256"]
    ):
        raise RuntimeError(f"Persisted V44 scaler reference drifted: {fold}")
    return record


def _validate_target_scale(
    path: Path,
    fold: int,
    fold_entry: dict,
    data_access: dict,
    data_audit: dict,
    expected_scale_floor: float,
) -> dict:
    record = _load_json(path)
    expected_pairs = int(
        data_access["expected_by_fold"][str(fold)]["train_eligible_pairs"]
    )
    semantic = {
        key: value
        for key, value in record.items()
        if key != "target_scale_state_sha256"
    }
    if (
        int(record.get("fold", -1)) != fold
        or record.get("train_symbols") != sorted(fold_entry["train_symbols"])
        or record.get("fit_scope")
        != "full_eligible_lexical_triplet_enumeration_train_only"
        or record.get("fit_start") != data_audit.get("first_train_signal_date")
        or record.get("fit_end") != data_audit.get("last_train_signal_date")
        or int(record.get("eligible_dates", -1))
        != int(data_audit.get("train_label_rows", -1))
        // len(fold_entry["train_symbols"])
        or int(record.get("enumerated_triplets", -1)) != expected_pairs
        or int(record.get("enumerated_excess_values", -1)) != expected_pairs * 3
        or float(record.get("scale_floor", math.nan))
        != float(expected_scale_floor)
        or not math.isfinite(float(record.get("excess_rms_scale", math.nan)))
        or float(record.get("excess_rms_scale", 0.0)) <= 0
        or record.get("target_scale_state_sha256")
        != _canonical_sha256(semantic)
    ):
        raise RuntimeError(f"Persisted V44 target scale drifted: {fold}")
    return record


def _completed_job(
    job_dir: Path,
    *,
    expected_spec_sha256: str,
    expected_architecture: dict,
    expected_parent_sha256: str,
    expected_fold: int | None = None,
    expected_seed: int | None = None,
    expected_parent_model_state_sha256: str | None = None,
    expected_parent_frozen_state_sha256: str | None = None,
    expected_parent_temporal_state_sha256: str | None = None,
    expected_parent_cross_head_state_sha256: str | None = None,
    expected_scaler_state_sha256: str | None = None,
    expected_target_scale_state_sha256: str | None = None,
) -> dict[str, object] | None:
    complete_path = job_dir / "complete.json"
    if not complete_path.is_file():
        return None
    complete = _load_json(complete_path)
    checkpoint_path = job_dir / "checkpoint.pt"
    if (
        complete.get("supervised_spec_sha256") != expected_spec_sha256
        or complete.get("parent_v43_checkpoint_sha256")
        != expected_parent_sha256
        or not checkpoint_path.is_file()
        or _sha256_file(checkpoint_path) != complete.get("checkpoint_sha256")
    ):
        raise RuntimeError(f"Completed V44 job drifted: {job_dir}")
    model, payload = load_ranking_excess_supervised_checkpoint(
        checkpoint_path,
        expected_architecture=expected_architecture,
    )
    metadata = payload["metadata"]
    history = complete.get("history", [])
    if (
        payload["model_state_sha256"] != complete.get("model_state_sha256")
        or metadata.get("supervised_spec_sha256") != expected_spec_sha256
        or int(metadata.get("fold", -1)) != int(complete.get("fold", -2))
        or int(metadata.get("initialization_seed", -1))
        != int(complete.get("seed", -2))
        or metadata.get("parent_v43_checkpoint_sha256")
        != expected_parent_sha256
        or (
            expected_fold is not None
            and int(metadata.get("fold", -1)) != int(expected_fold)
        )
        or (
            expected_seed is not None
            and int(metadata.get("initialization_seed", -1))
            != int(expected_seed)
        )
        or metadata.get("parent_v43_model_state_sha256")
        != complete.get("parent_v43_model_state_sha256")
        or (
            expected_parent_model_state_sha256 is not None
            and metadata.get("parent_v43_model_state_sha256")
            != expected_parent_model_state_sha256
        )
        or (
            expected_parent_frozen_state_sha256 is not None
            and complete.get("frozen_parent_state_sha256")
            != expected_parent_frozen_state_sha256
        )
        or metadata.get("parent_frozen_state_sha256")
        != complete.get("frozen_parent_state_sha256")
        or (
            expected_parent_temporal_state_sha256 is not None
            and complete.get("temporal_parent_state_sha256")
            != expected_parent_temporal_state_sha256
        )
        or metadata.get("parent_temporal_state_sha256")
        != complete.get("temporal_parent_state_sha256")
        or (
            expected_parent_cross_head_state_sha256 is not None
            and complete.get("cross_head_parent_state_sha256")
            != expected_parent_cross_head_state_sha256
        )
        or metadata.get("parent_cross_head_state_sha256")
        != complete.get("cross_head_parent_state_sha256")
        or metadata.get("scaler_state_sha256")
        != complete.get("scaler_state_sha256")
        or (
            expected_scaler_state_sha256 is not None
            and metadata.get("scaler_state_sha256")
            != expected_scaler_state_sha256
        )
        or metadata.get("target_scale_state_sha256")
        != complete.get("target_scale_state_sha256")
        or (
            expected_target_scale_state_sha256 is not None
            and metadata.get("target_scale_state_sha256")
            != expected_target_scale_state_sha256
        )
        or float(metadata.get("target_scale", math.nan))
        != float(complete.get("target_scale", math.nan))
        or metadata.get("train_symbols") != complete.get("train_symbols")
        or metadata.get("test_symbols") != complete.get("test_symbols")
        or int(metadata.get("best_epoch", -1))
        != int(complete.get("best_epoch", -2))
        or float(metadata.get("best_validation_core_loss", math.nan))
        != float(complete.get("best_validation_core_loss", math.nan))
        or int(metadata.get("completed_epochs", -1))
        != int(complete.get("completed_epochs", -2))
        or int(metadata.get("train_optimizer_steps", -1))
        != int(complete.get("train_optimizer_steps", -2))
        or metadata.get("history_state_sha256")
        != complete.get("history_state_sha256")
        or metadata.get("history_state_sha256") != _canonical_sha256(history)
        or metadata.get("frozen_state_sha256")
        != complete.get("frozen_final_state_sha256")
        or _state_subset_sha256(model, FROZEN_SUPERVISED_PREFIXES)
        != complete.get("frozen_final_state_sha256")
        or metadata.get("temporal_state_sha256")
        != complete.get("temporal_final_state_sha256")
        or _state_subset_sha256(model, TEMPORAL_SUPERVISED_PREFIXES)
        != complete.get("temporal_final_state_sha256")
        or metadata.get("cross_head_state_sha256")
        != complete.get("cross_head_final_state_sha256")
        or _state_subset_sha256(model, CROSS_HEAD_PREFIXES)
        != complete.get("cross_head_final_state_sha256")
    ):
        raise RuntimeError(f"Completed V44 checkpoint state drifted: {job_dir}")
    for path_key, hash_key in (
        ("fold_data_access_path", "fold_data_access_sha256"),
        ("scaler_reference_path", "scaler_reference_sha256"),
        ("target_scale_path", "target_scale_artifact_sha256"),
    ):
        artifact_path = Path(str(complete.get(path_key, "")))
        if (
            not artifact_path.is_file()
            or _sha256_file(artifact_path) != complete.get(hash_key)
            or metadata.get(hash_key) != complete.get(hash_key)
        ):
            raise RuntimeError(f"Completed V44 fold artifact drifted: {job_dir}")
    return complete


def _train_job(
    *,
    fold_entry: dict,
    seed: int,
    parent: dict,
    architecture: dict,
    store: SupervisedFeatureLabelStore,
    scaler: FeatureScaler,
    train_availability: dict[pd.Timestamp, list[str]],
    validation_availability: dict[pd.Timestamp, list[str]],
    supervised: dict,
    effective: dict,
    supervised_spec: dict,
    artifact_hashes: dict[str, str],
    fold_data_access_path: Path,
    fold_data_access_sha256: str,
    scaler_reference_path: Path,
    scaler_reference_sha256: str,
    target_scale_path: Path,
    target_scale_artifact_sha256: str,
    target_scale_state_sha256: str,
    target_scale: float,
    checkpoint_root: Path,
    device: torch.device,
) -> dict[str, object]:
    fold = int(fold_entry["fold"])
    train_symbols = sorted(fold_entry["train_symbols"])
    test_symbols = sorted(fold_entry["test_symbols"])
    parent_row = parent["row"]
    parent_path = Path(parent["path"])
    parent_sha256 = str(parent_row["checkpoint_sha256"])
    job_dir = checkpoint_root / f"fold_{fold}" / f"seed_{seed}"
    job_dir.mkdir(parents=True, exist_ok=True)
    completed = _completed_job(
        job_dir,
        expected_spec_sha256=supervised_spec["supervised_spec_sha256"],
        expected_architecture=architecture,
        expected_parent_sha256=parent_sha256,
        expected_fold=fold,
        expected_seed=seed,
        expected_parent_model_state_sha256=parent["model_state_sha256"],
        expected_parent_frozen_state_sha256=parent["frozen_state_sha256"],
        expected_parent_temporal_state_sha256=parent["temporal_state_sha256"],
        expected_parent_cross_head_state_sha256=parent["cross_head_state_sha256"],
        expected_scaler_state_sha256=scaler.state_sha256(),
        expected_target_scale_state_sha256=target_scale_state_sha256,
    )
    if completed is not None:
        return completed
    if set(train_symbols).intersection(set(test_symbols) | TARGET_SYMBOLS):
        raise RuntimeError("V44 job has forbidden asset overlap")
    if _sha256_file(parent_path) != parent_sha256:
        raise RuntimeError("V44 parent checkpoint hash drift")

    model, parent_payload = load_ranking_excess_pretrained_checkpoint(
        parent_path,
        expected_architecture=architecture,
    )
    parent_metadata = parent_payload["metadata"]
    if (
        int(parent_metadata["fold"]) != fold
        or int(parent_metadata["initialization_seed"]) != int(seed)
        or parent_metadata["scaler_state_sha256"] != scaler.state_sha256()
        or parent_payload["model_state_sha256"]
        != parent_row["model_state_sha256"]
    ):
        raise RuntimeError("V44 parent checkpoint semantic drift")
    parameters = configure_supervised_scope(model)
    parameter_names = supervised_parameter_names(model)
    parent_model_state_sha256 = model_state_sha256(model.state_dict())
    parent_frozen_state_sha256 = _state_subset_sha256(
        model, FROZEN_SUPERVISED_PREFIXES
    )
    parent_temporal_state_sha256 = _state_subset_sha256(
        model, TEMPORAL_SUPERVISED_PREFIXES
    )
    parent_cross_head_state_sha256 = _state_subset_sha256(
        model, CROSS_HEAD_PREFIXES
    )
    _seed_device(seed, device)
    model.to(device)
    optimizer_config = supervised["optimizer"]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(optimizer_config["learning_rate"]),
        betas=tuple(float(value) for value in optimizer_config["betas"]),
        eps=float(optimizer_config["epsilon"]),
        weight_decay=float(optimizer_config["weight_decay"]),
        amsgrad=bool(optimizer_config["amsgrad"]),
    )
    metadata = {
        "version": "v44",
        "candidate_family_id": supervised_spec["candidate_family_id"],
        "fold": fold,
        "initialization_seed": int(seed),
        "initialization_source": "exact_matching_v43_fold_seed_checkpoint",
        "parent_v43_checkpoint_sha256": parent_sha256,
        "parent_v43_model_state_sha256": parent_model_state_sha256,
        "parent_frozen_state_sha256": parent_frozen_state_sha256,
        "parent_temporal_state_sha256": parent_temporal_state_sha256,
        "parent_cross_head_state_sha256": parent_cross_head_state_sha256,
        "train_symbols": train_symbols,
        "test_symbols": test_symbols,
        "scaler_state_sha256": scaler.state_sha256(),
        "fold_data_access_sha256": fold_data_access_sha256,
        "scaler_reference_sha256": scaler_reference_sha256,
        "target_scale_artifact_sha256": target_scale_artifact_sha256,
        "target_scale_state_sha256": target_scale_state_sha256,
        "target_scale": float(target_scale),
        "supervised_spec_sha256": supervised_spec["supervised_spec_sha256"],
        **artifact_hashes,
    }
    train_sampler = DeterministicEligibleTripletSampler(
        train_availability, train_symbols, seed, fold
    )
    validation_sampler = DeterministicEligibleTripletSampler(
        validation_availability, train_symbols, seed, fold
    )
    validation_samples = validation_sampler.sample_epoch(
        int(supervised["validation_sampling_epoch"]),
        int(effective["validation_samples"]),
    )
    early = SupervisedEarlyStopping(
        patience=int(effective["early_stopping_patience"])
    )
    history: list[dict[str, object]] = []
    completed_epoch = 0
    best_model_state = _cpu_state_dict(model.state_dict())
    resume_path = job_dir / "resume.pt"
    if resume_path.is_file():
        completed_epoch, early, history, best_model_state = _load_resume(
            resume_path,
            model=model,
            optimizer=optimizer,
            expected_metadata=metadata,
            device=device,
            format_version=supervised["resume_format"],
            expected_best_state_format=supervised["best_state_format"],
            expected_patience=int(effective["early_stopping_patience"]),
            maximum_epochs=int(effective["maximum_epochs"]),
            expected_optimizer_parameter_names=parameter_names,
            expected_train_observations=int(
                effective["train_samples_per_epoch"]
            ),
            expected_validation_observations=int(
                effective["validation_samples"]
            ),
            expected_train_steps=math.ceil(
                int(effective["train_samples_per_epoch"])
                / int(effective["batch_size"])
            ),
            expected_validation_steps=math.ceil(
                int(effective["validation_samples"])
                / int(effective["batch_size"])
            ),
            expected_volatility_weight=float(
                supervised["objective"]["log_volatility_weight"]
            ),
        )

    if not early.should_stop:
        for epoch in range(
            completed_epoch + 1,
            int(effective["maximum_epochs"]) + 1,
        ):
            train_samples = train_sampler.sample_epoch(
                epoch, int(effective["train_samples_per_epoch"])
            )
            train_losses, train_steps = _run_supervised_batches(
                model,
                store,
                scaler,
                train_samples,
                batch_size=int(effective["batch_size"]),
                target_scale=target_scale,
                objective=supervised["objective"],
                device=device,
                optimizer=optimizer,
                gradient_clip_norm=float(supervised["gradient_clip_norm"]),
            )
            validation_losses, validation_steps = _run_supervised_batches(
                model,
                store,
                scaler,
                validation_samples,
                batch_size=int(effective["batch_size"]),
                target_scale=target_scale,
                objective=supervised["objective"],
                device=device,
                optimizer=None,
                gradient_clip_norm=float(supervised["gradient_clip_norm"]),
            )
            prior_best = early.best_loss
            should_stop = early.update(epoch, validation_losses["core"])
            if validation_losses["core"] < prior_best:
                best_model_state = _cpu_state_dict(model.state_dict())
            history.append({
                "epoch": epoch,
                "train_losses": train_losses,
                "validation_losses": validation_losses,
                "improved": validation_losses["core"] < prior_best,
                "train_optimizer_steps": train_steps,
                "validation_steps": validation_steps,
            })
            _save_resume(
                resume_path,
                model=model,
                optimizer=optimizer,
                early_stopping=early,
                history=history,
                completed_epoch=epoch,
                metadata=metadata,
                best_model_state=best_model_state,
                device=device,
                format_version=supervised["resume_format"],
                best_state_format=supervised["best_state_format"],
                optimizer_parameter_names=parameter_names,
            )
            _write_json(job_dir / "progress.json", {
                "fold": fold,
                "seed": int(seed),
                "completed_epoch": epoch,
                "best_epoch": early.best_epoch,
                "best_validation_core_loss": early.best_loss,
                "stale_epochs": early.stale_epochs,
                "should_stop": should_stop,
                "last_epoch": history[-1],
            })
            if should_stop:
                break

    if early.best_epoch < 0 or not history:
        raise RuntimeError("V44 job produced no best supervised state")
    if not _state_is_finite(best_model_state):
        raise RuntimeError("V44 best model state contains non-finite values")
    model.load_state_dict(best_model_state)
    final_frozen_state_sha256 = _state_subset_sha256(
        model, FROZEN_SUPERVISED_PREFIXES
    )
    final_temporal_state_sha256 = _state_subset_sha256(
        model, TEMPORAL_SUPERVISED_PREFIXES
    )
    final_cross_head_state_sha256 = _state_subset_sha256(
        model, CROSS_HEAD_PREFIXES
    )
    if final_frozen_state_sha256 != parent_frozen_state_sha256:
        raise RuntimeError("V44 modified mask/reconstruction state")
    if (
        final_temporal_state_sha256 == parent_temporal_state_sha256
        or final_cross_head_state_sha256 == parent_cross_head_state_sha256
    ):
        raise RuntimeError("V44 did not update the full inference path")
    if any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if name.startswith(FROZEN_SUPERVISED_PREFIXES)
    ):
        raise RuntimeError("V44 left a frozen reconstruction gradient")

    state = _cpu_state_dict(model.state_dict())
    state_sha256 = model_state_sha256(state)
    train_optimizer_steps = int(
        sum(row["train_optimizer_steps"] for row in history)
    )
    history_state_sha256 = _canonical_sha256(history)
    checkpoint_metadata = {
        **metadata,
        "checkpoint_status": "frozen_supervised_no_seed_or_fold_selection",
        "best_epoch": int(early.best_epoch),
        "best_validation_core_loss": float(early.best_loss),
        "completed_epochs": len(history),
        "train_optimizer_steps": train_optimizer_steps,
        "history_state_sha256": history_state_sha256,
        "supervised_parameter_names": parameter_names,
        "supervised_parameter_count": int(
            sum(parameter.numel() for parameter in parameters)
        ),
        "unused_during_supervised_training": [
            "mask_token",
            "reconstruction_head",
        ],
        "frozen_state_sha256": final_frozen_state_sha256,
        "temporal_state_sha256": final_temporal_state_sha256,
        "cross_head_state_sha256": final_cross_head_state_sha256,
    }
    checkpoint_payload = {
        "format_version": supervised["checkpoint_format"],
        "input_features": 9,
        "architecture": architecture,
        "architecture_sha256": _canonical_sha256(architecture),
        "metadata": checkpoint_metadata,
        "model_state_sha256": state_sha256,
        "state_dict": state,
    }
    checkpoint_path = job_dir / "checkpoint.pt"
    _atomic_torch_save(checkpoint_payload, checkpoint_path)
    restored, restored_payload = load_ranking_excess_supervised_checkpoint(
        checkpoint_path,
        expected_architecture=architecture,
        expected_metadata=checkpoint_metadata,
    )
    if model_state_sha256(restored.state_dict()) != state_sha256:
        raise RuntimeError("V44 checkpoint semantic roundtrip drift")
    del restored, restored_payload
    complete = {
        "version": "v44",
        "fold": fold,
        "seed": int(seed),
        "train_symbols": train_symbols,
        "test_symbols": test_symbols,
        "initialization_source": "exact_matching_v43_fold_seed_checkpoint",
        "parent_v43_checkpoint_sha256": parent_sha256,
        "parent_v43_model_state_sha256": parent_model_state_sha256,
        "scaler_state_sha256": scaler.state_sha256(),
        "target_scale": float(target_scale),
        "target_scale_state_sha256": target_scale_state_sha256,
        "fold_data_access_path": str(fold_data_access_path),
        "fold_data_access_sha256": fold_data_access_sha256,
        "scaler_reference_path": str(scaler_reference_path),
        "scaler_reference_sha256": scaler_reference_sha256,
        "target_scale_path": str(target_scale_path),
        "target_scale_artifact_sha256": target_scale_artifact_sha256,
        "supervised_spec_sha256": supervised_spec["supervised_spec_sha256"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "model_state_sha256": state_sha256,
        "frozen_parent_state_sha256": parent_frozen_state_sha256,
        "frozen_final_state_sha256": final_frozen_state_sha256,
        "temporal_parent_state_sha256": parent_temporal_state_sha256,
        "temporal_final_state_sha256": final_temporal_state_sha256,
        "cross_head_parent_state_sha256": parent_cross_head_state_sha256,
        "cross_head_final_state_sha256": final_cross_head_state_sha256,
        "best_epoch": int(early.best_epoch),
        "best_validation_core_loss": float(early.best_loss),
        "completed_epochs": len(history),
        "train_optimizer_steps": train_optimizer_steps,
        "history_state_sha256": history_state_sha256,
        "history": history,
        "seed_selected": False,
        "fold_selected": False,
        "heldout_assets_loaded": False,
        "target_assets_loaded": False,
        "development_2025_loaded": False,
        "predictions_computed": False,
        "performance_metrics_computed": False,
        "pnl_computed": False,
    }
    _write_json(job_dir / "complete.json", complete)
    resume_path.unlink(missing_ok=True)
    return complete


def _report(result: dict[str, object]) -> str:
    mode = result["supervised_spec"]["mode"]
    if mode == "preflight":
        status = "METADATA/CHECKPOINT PREFLIGHT PASSED; ONE-JOB MPS SMOKE IS NEXT."
    elif mode == "smoke":
        status = "ONE-JOB MPS SMOKE PASSED; FULL NINE-JOB SUPERVISION IS NEXT."
    else:
        status = "ALL NINE RANKING/EXCESS SUPERVISED JOBS PASSED."
    summary = result["summary"]
    return "\n".join([
        "# TLM v44 Ranking/Excess Supervised Non-Target Training",
        "",
        "## Decision",
        "",
        f"**{status}**",
        "",
        f"Mode: **{mode}**",
        f"Supervised spec SHA-256: `{result['supervised_spec']['supervised_spec_sha256']}`",
        f"Checkpoints: **{summary['checkpoint_count']}**",
        f"Optimizer steps: **{summary['total_optimizer_steps']:,}**",
        "",
        "V44 initializes each job from its exact V43 fold/seed parent, reuses the immutable fold scaler, and fits the excess scale only by full enumeration of supervised-train triplets.",
        "",
        "No held-out asset, BTC/ETH/SOL, 2025 signal/outcome, prediction metric, portfolio, performance statistic, or PnL was loaded or computed.",
        "",
        "## Next action",
        "",
        "A passing full run authorizes only the frozen V45 one-shot 2025 asset-disjoint development screen. It does not establish alpha or economic value.",
        "",
    ])


def _write_result(
    root: Path,
    output_relative: str,
    config: dict,
    result: dict[str, object],
) -> None:
    output = root / output_relative
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "result.json", result)
    _write_json(output / "supervised_spec.json", result["supervised_spec"])
    _write_json(output / "audit.json", result["audit"])
    if "checkpoint_manifest" in result:
        _write_json(output / "checkpoint_manifest.json", result["checkpoint_manifest"])
    if "data_access_audit" in result:
        _write_json(output / "data_access_audit.json", result["data_access_audit"])
    if "target_scales" in result:
        _write_json(output / "target_scales.json", result["target_scales"])
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output / "report.md").write_text(_report(result), encoding="utf-8")


def run_ranking_excess_supervised(
    config: dict,
    mode: str,
) -> dict[str, object]:
    context = _metadata_context(config)
    root = context["root"]
    paths = context["paths"]
    supervised = context["supervised"]
    blueprint = context["blueprint"]
    manifest = context["manifest"]
    asset_folds = context["asset_folds"]
    parents = context["parents"]
    scalers = context["scalers"]
    if mode not in {"preflight", "smoke", "full"}:
        raise ValueError("V44 mode must be preflight, smoke, or full")

    prior_gate = _load_prior_gate(root, supervised, blueprint, mode)
    spec = build_supervised_spec(
        blueprint, supervised, mode, prior_gate=prior_gate
    )
    effective = supervised["smoke"] if mode == "smoke" else supervised["full_run"]
    seeds = [int(seed) for seed in effective["seeds"]]
    folds = [int(fold) for fold in effective["folds"]]
    scope_checks, total_parameters, trainable_parameters, frozen_parameters, parameter_names = (
        _scope_audit(blueprint["architecture"])
    )

    objective = supervised["objective"]
    blueprint_objective = blueprint["objective"]
    optimizer_config = supervised["optimizer"]
    blueprint_training = blueprint["training"]
    expected_grid = {
        (fold, seed) for fold in (1, 2, 3) for seed in (42, 7, 123)
    }
    contract_checks = {
        **scope_checks,
        "prior_gate_is_exact_for_execution_mode": (
            mode == "preflight" and prior_gate is None
        ) or (mode != "preflight" and prior_gate is not None),
        "input_allowlist_is_exact": set(paths)
        == METADATA_INPUT_NAMES | BINARY_INPUT_NAMES,
        "parent_grid_and_hashes_are_exact": set(parents) == expected_grid
        and all(
            parents[(fold, seed)]["row"]["checkpoint_sha256"]
            == supervised["expected_parents"][f"{fold}:{seed}"]
            for fold, seed in expected_grid
        ),
        "scaler_grid_and_hashes_are_exact": set(scalers) == {1, 2, 3}
        and all(
            scalers[fold]["scaler"].state_sha256()
            == supervised["expected_scalers"][str(fold)]["state_sha256"]
            for fold in (1, 2, 3)
        ),
        "total_parameter_count_is_frozen": total_parameters
        == int(supervised["expected_total_parameters"]),
        "supervised_parameter_count_is_frozen": trainable_parameters
        == int(supervised["expected_supervised_parameters"]),
        "frozen_parameter_count_is_frozen": frozen_parameters
        == int(supervised["expected_frozen_parameters"]),
        "parameter_scope_is_exact": supervised["parameter_scope"]
        == {
            "train_all_except": ["mask_token", "reconstruction_head"],
            "frozen_state_must_remain_parent_exact": True,
        }
        and all(
            not name.startswith(FROZEN_SUPERVISED_PREFIXES)
            for name in parameter_names
        ),
        "runtime_is_mps_float32_deterministic_without_amp_or_fallback": supervised[
            "device"
        ]
        == "mps"
        and supervised["dtype"] == "float32"
        and supervised["deterministic_algorithms"] is True
        and supervised["amp"] is False
        and supervised["cpu_fallback_allowed"] is False,
        "execution_environment_disables_mps_fallback": mode == "preflight"
        or _mps_fallback_is_disabled(),
        "objective_matches_frozen_blueprint": objective
        == {
            "ranking_loss": blueprint_objective["ranking_loss"],
            "excess_loss": blueprint_objective["excess_loss"],
            "log_volatility_loss": blueprint_objective["log_volatility_loss"],
            "early_stopping_monitor": blueprint_objective[
                "early_stopping_monitor"
            ],
            "ranking_weight": blueprint_objective["weights"]["ranking"],
            "excess_weight": blueprint_objective["weights"]["excess"],
            "log_volatility_weight": blueprint_objective["weights"][
                "log_volatility"
            ],
            "exact_tie_tolerance": blueprint_objective["exact_tie_tolerance"],
            "volatility_floor": blueprint_objective["volatility_floor"],
            "scale_floor": blueprint_objective["scale_floor"],
            "target_clipping": False,
            "outcome_weighting": False,
            "target_scale_fit": "full_eligible_lexical_triplet_enumeration_train_only",
        },
        "optimizer_matches_frozen_blueprint": optimizer_config
        == {
            "name": "AdamW",
            "learning_rate": blueprint_training["learning_rate"],
            "weight_decay": blueprint_training["weight_decay"],
            "betas": [0.9, 0.999],
            "epsilon": 1.0e-8,
            "amsgrad": False,
            "scheduler": None,
        },
        "early_stopping_and_sampling_are_exact": supervised["early_stopping"]
        == {
            "improvement_rule": "strict_less",
            "consecutive_non_improvements": True,
            "restore_best_state": True,
        }
        and supervised["sampling"]
        == {
            "train": "deterministic_uniform_date_triplet_with_replacement_seed_fold_epoch",
            "validation": "deterministic_fixed_sampling_epoch_zero_per_fold_seed",
        }
        and int(supervised["validation_sampling_epoch"]) == 0,
        "checkpoint_formats_are_new_and_exact": supervised["checkpoint_format"]
        == "v44_ranking_excess_supervised_v1"
        and supervised["resume_format"]
        == "v44_ranking_excess_supervised_resume_v1"
        and supervised["best_state_format"]
        == "v44_ranking_excess_supervised_best_state_v1",
        "full_fold_seed_grid_is_exact": list(supervised["full_run"]["folds"])
        == [1, 2, 3]
        and list(supervised["full_run"]["seeds"]) == [42, 7, 123],
        "smoke_fold_seed_grid_is_exact": list(supervised["smoke"]["folds"])
        == [1]
        and list(supervised["smoke"]["seeds"]) == [42],
        "full_training_contract_matches_blueprint": int(
            supervised["full_run"]["train_samples_per_epoch"]
        )
        == int(blueprint_training["supervised_samples_per_epoch"])
        and int(supervised["full_run"]["validation_samples"])
        == int(blueprint_training["fixed_validation_samples"])
        and int(supervised["full_run"]["batch_size"])
        == int(blueprint_training["batch_size"])
        and int(supervised["full_run"]["maximum_epochs"])
        == int(blueprint_training["maximum_supervised_epochs"])
        and int(supervised["full_run"]["early_stopping_patience"])
        == int(blueprint_training["early_stopping_patience"])
        and float(supervised["gradient_clip_norm"])
        == float(blueprint_training["gradient_clip_norm"]),
        "initialization_is_exact_parent_only": supervised["initialization"]
        == {
            "source": "exact_matching_v43_fold_seed_checkpoint",
            "v42_synthetic_checkpoint_allowed": False,
            "v35_v36_checkpoint_allowed": False,
            "fresh_reinitialization_allowed": False,
        },
        "chronology_is_exact": supervised["data_access"][
            "supervised_train_start"
        ]
        == blueprint["chronological_splits"]["supervised_train"][0]
        and supervised["data_access"]["supervised_train_end"]
        == blueprint["chronological_splits"]["supervised_train"][1]
        and supervised["data_access"]["validation_start"]
        == blueprint["chronological_splits"][
            "early_stopping_train_assets_only"
        ][0]
        and supervised["data_access"]["validation_end"]
        == blueprint["chronological_splits"][
            "early_stopping_train_assets_only"
        ][1],
        "authorized_next_action_is_exact": supervised["authorized_next_action"]
        == "v45_asset_disjoint_2025_development_screen_only",
    }
    contract_checks = {key: bool(value) for key, value in contract_checks.items()}
    if not all(contract_checks.values()):
        raise RuntimeError(f"V44 contract audit failed: {contract_checks}")

    mps_available = bool(torch.backends.mps.is_available())
    if mode != "preflight" and not mps_available:
        raise RuntimeError("V44 requires an available MPS device and forbids CPU fallback")
    if mode == "preflight":
        result = {
            "version": "v44_preflight",
            "decision": "authorize_v44_one_job_mps_smoke_only",
            "supervised_spec": spec,
            "summary": {
                "checkpoint_count": 0,
                "parent_checkpoint_count": len(parents),
                "total_optimizer_steps": 0,
                "total_parameters": total_parameters,
                "supervised_parameters": trainable_parameters,
                "frozen_parameters": frozen_parameters,
                "mps_available": mps_available,
                "parquet_files_deserialized": 0,
                "label_rows_materialized": 0,
            },
            "tested": {
                "metadata_and_parent_checkpoints_loaded": True,
                "panel_or_sequence_deserialized": False,
                "optimizer_executed": False,
                "labels_loaded": False,
                "heldout_assets_loaded": False,
                "target_assets_loaded": False,
                "development_2025_loaded": False,
                "predictions_computed": False,
                "performance_metrics_computed": False,
                "pnl_computed": False,
                "seed_or_fold_selection_executed": False,
            },
            "audit": {"passed": True, "checks": contract_checks},
        }
        _write_result(root, supervised["preflight_output_dir"], config, result)
        return result

    for name in BINARY_INPUT_NAMES:
        if _sha256_file(paths[name]) != supervised["expected_input_sha256"][name]:
            raise RuntimeError(f"V44 binary input hash drift: {name}")
    device = _configure_device(supervised["device"], supervised["torch_threads"])
    feature_names = list(manifest["panel_features"])
    label_names = list(manifest["labels"])
    folds_by_number = {
        int(entry["fold"]): entry for entry in asset_folds["folds"]
    }
    checkpoint_root = root / (
        supervised["smoke_checkpoint_dir"]
        if mode == "smoke"
        else supervised["checkpoint_dir"]
    )
    artifact_hashes = {
        "v41_blueprint_artifact_sha256": supervised["expected_input_sha256"][
            "v41_blueprint"
        ],
        "v43_result_sha256": supervised["expected_input_sha256"]["v43_result"],
        "v43_checkpoint_manifest_sha256": supervised["expected_input_sha256"][
            "v43_checkpoint_manifest"
        ],
        "dataset_manifest_sha256": supervised["expected_input_sha256"][
            "v32_dataset_manifest"
        ],
        "feature_schema_sha256": supervised["expected_input_sha256"][
            "v32_feature_schema"
        ],
        "asset_folds_sha256": supervised["expected_input_sha256"][
            "v32_asset_folds"
        ],
        "panel_sha256": supervised["expected_input_sha256"]["panel"],
        "sequence_index_sha256": supervised["expected_input_sha256"][
            "sequence_index"
        ],
    }
    jobs: list[dict[str, object]] = []
    data_audits: list[dict[str, object]] = []
    read_receipts: list[dict[str, object]] = []
    target_scale_records: list[dict[str, object]] = []
    for fold in folds:
        fold_entry = folds_by_number[fold]
        fold_dir = checkpoint_root / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_data_access_path = fold_dir / "data_access.json"
        scaler_reference_path = fold_dir / "scaler_reference.json"
        target_scale_path = fold_dir / "target_scale.json"
        completed_jobs = [
            _completed_job(
                fold_dir / f"seed_{seed}",
                expected_spec_sha256=spec["supervised_spec_sha256"],
                expected_architecture=blueprint["architecture"],
                expected_parent_sha256=parents[(fold, seed)]["row"][
                    "checkpoint_sha256"
                ],
                expected_fold=fold,
                expected_seed=seed,
                expected_parent_model_state_sha256=parents[(fold, seed)][
                    "model_state_sha256"
                ],
                expected_parent_frozen_state_sha256=parents[(fold, seed)][
                    "frozen_state_sha256"
                ],
                expected_parent_temporal_state_sha256=parents[(fold, seed)][
                    "temporal_state_sha256"
                ],
                expected_parent_cross_head_state_sha256=parents[(fold, seed)][
                    "cross_head_state_sha256"
                ],
                expected_scaler_state_sha256=scalers[fold][
                    "scaler"
                ].state_sha256(),
            )
            for seed in seeds
        ]
        if all(job is not None for job in completed_jobs):
            if not all(
                path.is_file()
                for path in (
                    fold_data_access_path,
                    scaler_reference_path,
                    target_scale_path,
                )
            ):
                raise RuntimeError(f"Completed V44 fold lacks an artifact: {fold}")
            prior_access = _load_json(fold_data_access_path)
            _validate_persisted_data_access(
                prior_access, fold_entry, supervised["data_access"]
            )
            _validate_scaler_reference(scaler_reference_path, fold, supervised)
            target_scale_record = _validate_target_scale(
                target_scale_path,
                fold,
                fold_entry,
                supervised["data_access"],
                prior_access["audit"],
                float(objective["scale_floor"]),
            )
            data_audits.append(prior_access["audit"])
            read_receipts.extend(prior_access["receipts"])
            target_scale_records.append(target_scale_record)
            jobs.extend(job for job in completed_jobs if job is not None)
            continue

        data = read_fold_supervised_data(
            paths["panel"],
            paths["sequence_index"],
            fold_entry,
            supervised["data_access"],
        )
        fold_receipts = [{"fold": fold, **receipt} for receipt in data.receipts]
        fold_access = {"audit": data.audit, "receipts": fold_receipts}
        _write_json(fold_data_access_path, fold_access)
        _validate_persisted_data_access(
            fold_access, fold_entry, supervised["data_access"]
        )
        fold_data_access_sha256 = _sha256_file(fold_data_access_path)
        data_audits.append(data.audit)
        read_receipts.extend(fold_receipts)

        scaler_info = scalers[fold]
        scaler = scaler_info["scaler"]
        scaler_reference = {
            "version": "v44_scaler_reference_v1",
            "fold": fold,
            "source_path": str(scaler_info["path"]),
            "artifact_sha256": supervised["expected_scalers"][str(fold)][
                "artifact_sha256"
            ],
            "state_sha256": supervised["expected_scalers"][str(fold)][
                "state_sha256"
            ],
            "scaler_refit_performed": False,
        }
        _write_json(scaler_reference_path, scaler_reference)
        _validate_scaler_reference(scaler_reference_path, fold, supervised)
        scaler_reference_sha256 = _sha256_file(scaler_reference_path)

        target_scale_record = fit_fold_excess_scale(
            data.train_labels,
            data.train_availability,
            float(objective["scale_floor"]),
        )
        target_scale_record.update({
            "fold": fold,
            "train_symbols": sorted(fold_entry["train_symbols"]),
        })
        target_scale_record["target_scale_state_sha256"] = _canonical_sha256({
            key: value
            for key, value in target_scale_record.items()
            if key != "target_scale_state_sha256"
        })
        _write_json(target_scale_path, target_scale_record)
        _validate_target_scale(
            target_scale_path,
            fold,
            fold_entry,
            supervised["data_access"],
            data.audit,
            float(objective["scale_floor"]),
        )
        target_scale_records.append(target_scale_record)
        target_scale_artifact_sha256 = _sha256_file(target_scale_path)
        target_scale = float(target_scale_record["excess_rms_scale"])
        target_scale_state_sha256 = str(
            target_scale_record["target_scale_state_sha256"]
        )

        store = SupervisedFeatureLabelStore(
            data.feature_panel,
            [data.train_labels, data.validation_labels],
            feature_names,
            label_names,
            int(blueprint["architecture"]["lookback_days"]),
            "log_close_to_close_return",
        )
        for seed, prior_complete in zip(seeds, completed_jobs, strict=True):
            if prior_complete is not None:
                jobs.append(prior_complete)
                continue
            jobs.append(_train_job(
                fold_entry=fold_entry,
                seed=seed,
                parent=parents[(fold, seed)],
                architecture=blueprint["architecture"],
                store=store,
                scaler=scaler,
                train_availability=data.train_availability,
                validation_availability=data.validation_availability,
                supervised=supervised,
                effective=effective,
                supervised_spec=spec,
                artifact_hashes=artifact_hashes,
                fold_data_access_path=fold_data_access_path,
                fold_data_access_sha256=fold_data_access_sha256,
                scaler_reference_path=scaler_reference_path,
                scaler_reference_sha256=scaler_reference_sha256,
                target_scale_path=target_scale_path,
                target_scale_artifact_sha256=target_scale_artifact_sha256,
                target_scale_state_sha256=target_scale_state_sha256,
                target_scale=target_scale,
                checkpoint_root=checkpoint_root,
                device=device,
            ))
        del store, data
        gc.collect()
        torch.mps.empty_cache()

    expected_jobs = len(folds) * len(seeds)
    combinations_seen = {(int(job["fold"]), int(job["seed"])) for job in jobs}
    parent_hashes_by_job = {
        (int(job["fold"]), int(job["seed"])): job[
            "parent_v43_checkpoint_sha256"
        ]
        for job in jobs
    }
    scaler_hashes = {
        fold: {
            job["scaler_state_sha256"]
            for job in jobs
            if int(job["fold"]) == fold
        }
        for fold in folds
    }
    scale_hashes = {
        fold: {
            job["target_scale_state_sha256"]
            for job in jobs
            if int(job["fold"]) == fold
        }
        for fold in folds
    }
    total_optimizer_steps = int(
        sum(int(job["train_optimizer_steps"]) for job in jobs)
    )
    target_scale_by_fold = {
        int(record["fold"]): record for record in target_scale_records
    }
    revalidated_jobs = [
        _completed_job(
            Path(str(job["checkpoint_path"])).parent,
            expected_spec_sha256=spec["supervised_spec_sha256"],
            expected_architecture=blueprint["architecture"],
            expected_parent_sha256=supervised["expected_parents"][
                f"{int(job['fold'])}:{int(job['seed'])}"
            ],
            expected_fold=int(job["fold"]),
            expected_seed=int(job["seed"]),
            expected_parent_model_state_sha256=parents[
                (int(job["fold"]), int(job["seed"]))
            ]["model_state_sha256"],
            expected_parent_frozen_state_sha256=parents[
                (int(job["fold"]), int(job["seed"]))
            ]["frozen_state_sha256"],
            expected_parent_temporal_state_sha256=parents[
                (int(job["fold"]), int(job["seed"]))
            ]["temporal_state_sha256"],
            expected_parent_cross_head_state_sha256=parents[
                (int(job["fold"]), int(job["seed"]))
            ]["cross_head_state_sha256"],
            expected_scaler_state_sha256=scalers[int(job["fold"])][
                "scaler"
            ].state_sha256(),
            expected_target_scale_state_sha256=target_scale_by_fold[
                int(job["fold"])
            ]["target_scale_state_sha256"],
        )
        for job in jobs
    ]
    execution_checks = {
        **contract_checks,
        "checkpoint_count_and_grid_are_exact": len(jobs) == expected_jobs
        and len(combinations_seen) == expected_jobs,
        "all_checkpoint_files_and_states_match": all(
            Path(str(job["checkpoint_path"])).is_file()
            and _sha256_file(Path(str(job["checkpoint_path"])))
            == job["checkpoint_sha256"]
            and bool(job["model_state_sha256"])
            for job in jobs
        ) and all(job is not None for job in revalidated_jobs),
        "each_job_uses_exact_matching_v43_parent": all(
            parent_hashes_by_job[(fold, seed)]
            == supervised["expected_parents"][f"{fold}:{seed}"]
            for fold, seed in combinations_seen
        ),
        "one_scaler_and_scale_per_fold_are_shared_across_seeds": all(
            len(scaler_hashes[fold]) == 1 and len(scale_hashes[fold]) == 1
            for fold in folds
        ),
        "frozen_reconstruction_state_never_changes": all(
            job["frozen_parent_state_sha256"]
            == job["frozen_final_state_sha256"]
            for job in jobs
        ),
        "temporal_and_cross_head_paths_are_really_updated": all(
            job["temporal_parent_state_sha256"]
            != job["temporal_final_state_sha256"]
            and job["cross_head_parent_state_sha256"]
            != job["cross_head_final_state_sha256"]
            for job in jobs
        ),
        "no_seed_or_fold_selection": all(
            not job["seed_selected"] and not job["fold_selected"] for job in jobs
        ),
        "no_heldout_target_2025_prediction_performance_or_pnl": all(
            not job["heldout_assets_loaded"]
            and not job["target_assets_loaded"]
            and not job["development_2025_loaded"]
            and not job["predictions_computed"]
            and not job["performance_metrics_computed"]
            and not job["pnl_computed"]
            for job in jobs
        ),
        "all_materialized_fold_data_passed_audit": all(
            not audit["heldout_symbols_materialized"]
            and not audit["target_symbols_materialized"]
            and int(audit["post_2024_signal_rows"]) == 0
            and int(audit["post_2024_target_maturities"]) == 0
            for audit in data_audits
        ),
        "every_fold_scale_uses_full_train_enumeration": all(
            record["fit_scope"]
            == "full_eligible_lexical_triplet_enumeration_train_only"
            and int(record["enumerated_triplets"])
            == int(supervised["data_access"]["expected_by_fold"][str(fold)][
                "train_eligible_pairs"
            ])
            for fold, record in zip(folds, target_scale_records, strict=True)
        ),
        "smoke_has_exactly_eight_optimizer_steps": mode != "smoke"
        or total_optimizer_steps == 8,
        "full_run_has_all_nine_jobs_within_step_budget": mode == "smoke"
        or (len(jobs) == 9 and 0 < total_optimizer_steps <= 17_280),
        "no_resume_artifacts_remain": not any(
            checkpoint_root.rglob("resume.pt")
        ),
    }
    execution_checks = {key: bool(value) for key, value in execution_checks.items()}
    if not all(execution_checks.values()):
        raise RuntimeError(f"V44 execution audit failed: {execution_checks}")

    manifest_rows = [
        {
            key: job[key]
            for key in (
                "fold",
                "seed",
                "train_symbols",
                "test_symbols",
                "parent_v43_checkpoint_sha256",
                "parent_v43_model_state_sha256",
                "scaler_state_sha256",
                "target_scale",
                "target_scale_state_sha256",
                "checkpoint_path",
                "checkpoint_sha256",
                "model_state_sha256",
                "best_epoch",
                "best_validation_core_loss",
                "completed_epochs",
                "train_optimizer_steps",
            )
        }
        for job in jobs
    ]
    decision = (
        "authorize_v44_full_nine_job_supervised_only"
        if mode == "smoke"
        else supervised["authorized_next_action"]
    )
    result = {
        "version": "v44_smoke" if mode == "smoke" else "v44",
        "decision": decision,
        "supervised_spec": spec,
        "summary": {
            "checkpoint_count": len(jobs),
            "parent_checkpoint_count": len(parents),
            "total_optimizer_steps": total_optimizer_steps,
            "total_completed_epochs": int(
                sum(int(job["completed_epochs"]) for job in jobs)
            ),
            "total_parameters": total_parameters,
            "supervised_parameters": trainable_parameters,
            "frozen_parameters": frozen_parameters,
            "mps_available": mps_available,
            "folds_covered": [
                int(audit["fold"]) for audit in data_audits
            ],
            "label_rows_covered": int(sum(
                int(audit["train_label_rows"])
                + int(audit["validation_label_rows"])
                for audit in data_audits
            )),
        },
        "checkpoint_manifest": manifest_rows,
        "data_access_audit": {
            "folds": data_audits,
            "read_receipts": read_receipts,
            "physical_row_group_isolation_claimed": False,
        },
        "target_scales": target_scale_records,
        "tested": {
            "training_artifacts_cover_real_non_target_features": True,
            "training_artifacts_cover_train_and_validation_labels": True,
            "supervised_training_artifacts_validated": True,
            "fixed_2024_validation_artifacts_validated": True,
            "heldout_assets_loaded": False,
            "target_assets_loaded": False,
            "development_2025_loaded": False,
            "predictions_computed": False,
            "performance_metrics_computed": False,
            "pnl_computed": False,
            "seed_or_fold_selection_executed": False,
        },
        "audit": {"passed": True, "checks": execution_checks},
    }
    output_relative = (
        supervised["smoke_output_dir"] if mode == "smoke" else config["output_dir"]
    )
    _write_result(root, output_relative, config, result)
    return result
