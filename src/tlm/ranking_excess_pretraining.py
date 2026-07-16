from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import gc
import hashlib
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml

from .non_target_pretraining import TripletTensorStore
from .patch_transformer import MultiAssetPatchTransformer
from .ranking_excess_harness import RANKING_EXCESS_HEADS
from .ranking_excess_spec import (
    _canonical_sha256,
    _load_json,
    _sha256_file,
    _write_json,
)
from .scientific_harness import (
    DeterministicEligibleTripletSampler,
    FeatureScaler,
    deterministic_patch_mask,
    masked_reconstruction_loss,
)
from .supervised_non_target import model_state_sha256


TARGET_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
PRETRAINING_PREFIXES = (
    "temporal_position",
    "mask_token",
    "patch_projection.",
    "temporal_encoder.",
    "temporal_norm.",
    "reconstruction_head.",
)
FORBIDDEN_PRETRAINING_PREFIXES = (
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
    "v32_dataset_manifest",
    "v32_feature_schema",
    "v32_asset_folds",
}
BINARY_INPUT_NAMES = {"panel", "sequence_index"}


@dataclass
class PretrainingEarlyStopping:
    patience: int
    best_loss: float = math.inf
    best_epoch: int = -1
    stale_epochs: int = 0
    should_stop: bool = False

    def update(self, epoch: int, loss: float) -> bool:
        if not math.isfinite(loss):
            raise ValueError("Validation loss must be finite")
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
class FoldFeatureData:
    panel: pd.DataFrame
    train_availability: dict[pd.Timestamp, list[str]]
    validation_availability: dict[pd.Timestamp, list[str]]
    audit: dict[str, object]
    receipts: list[dict[str, object]]


def pretraining_parameter_names(
    model: MultiAssetPatchTransformer,
) -> list[str]:
    return [
        name
        for name, _ in model.named_parameters()
        if name.startswith(PRETRAINING_PREFIXES)
    ]


def configure_pretraining_scope(
    model: MultiAssetPatchTransformer,
) -> list[nn.Parameter]:
    allowed = set(pretraining_parameter_names(model))
    parameters = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in allowed)
        if name in allowed:
            parameters.append(parameter)
    if not parameters:
        raise RuntimeError("V43 selected no pretraining parameters")
    if any(
        parameter.requires_grad
        for name, parameter in model.named_parameters()
        if name.startswith(FORBIDDEN_PRETRAINING_PREFIXES)
    ):
        raise RuntimeError("V43 enabled a forbidden parameter")
    return parameters


def _state_subset_sha256(
    model: MultiAssetPatchTransformer,
    prefixes: tuple[str, ...],
) -> str:
    subset = {
        name: value
        for name, value in model.state_dict().items()
        if name.startswith(prefixes)
    }
    if not subset:
        raise RuntimeError("State subset is empty")
    return model_state_sha256(subset)


def _cpu_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in state_dict.items()
    }


def _to_cpu(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary, _use_new_zipfile_serialization=False)
    temporary.replace(path)


def _seed_device(seed: int, device: torch.device) -> None:
    torch.manual_seed(int(seed))
    if device.type == "mps":
        torch.mps.manual_seed(int(seed))


def _rng_state(device: torch.device) -> dict[str, torch.Tensor]:
    state = {"cpu": torch.get_rng_state().cpu()}
    if device.type == "mps":
        state["mps"] = torch.mps.get_rng_state().cpu()
    return state


def _restore_rng_state(
    state: dict[str, torch.Tensor],
    device: torch.device,
) -> None:
    torch.set_rng_state(state["cpu"].cpu())
    if device.type == "mps":
        if "mps" not in state:
            raise RuntimeError("V43 MPS resume is missing MPS RNG state")
        torch.mps.set_rng_state(state["mps"].cpu())


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _state_is_finite(state_dict: dict[str, torch.Tensor]) -> bool:
    return all(
        not value.is_floating_point() or bool(torch.isfinite(value).all())
        for value in state_dict.values()
    )


def _nested_state_is_finite(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_nested_state_is_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_nested_state_is_finite(item) for item in value)
    return True


def _semantic_state_sha256(value: object) -> str:
    digest = hashlib.sha256()

    def emit(tag: str, payload: bytes = b"") -> None:
        encoded_tag = tag.encode("utf-8")
        digest.update(len(encoded_tag).to_bytes(4, "big"))
        digest.update(encoded_tag)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    def visit(item: object) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            emit("tensor_dtype", str(tensor.dtype).encode("utf-8"))
            emit("tensor_shape", repr(tuple(tensor.shape)).encode("utf-8"))
            emit(
                "tensor_bytes",
                tensor.reshape(-1).view(torch.uint8).numpy().tobytes(),
            )
        elif isinstance(item, dict):
            emit("dict_start", str(len(item)).encode("ascii"))
            for key in sorted(
                item,
                key=lambda value: (type(value).__name__, repr(value)),
            ):
                visit(key)
                visit(item[key])
            emit("dict_end")
        elif isinstance(item, (list, tuple)):
            emit(type(item).__name__ + "_start", str(len(item)).encode("ascii"))
            for nested in item:
                visit(nested)
            emit(type(item).__name__ + "_end")
        elif item is None:
            emit("none")
        elif isinstance(item, bool):
            emit("bool", b"1" if item else b"0")
        elif isinstance(item, int):
            emit("int", str(item).encode("ascii"))
        elif isinstance(item, float):
            emit("float", item.hex().encode("ascii"))
        elif isinstance(item, str):
            emit("str", item.encode("utf-8"))
        else:
            raise TypeError(
                f"Unsupported V43 semantic-state type: {type(item).__name__}"
            )

    visit(value)
    return digest.hexdigest()


def _optimizer_contract(
    optimizer: torch.optim.Optimizer,
    parameter_names: list[str],
) -> dict[str, object]:
    param_groups = optimizer.state_dict()["param_groups"]
    parameter_count = sum(len(group["params"]) for group in param_groups)
    if parameter_count != len(parameter_names):
        raise RuntimeError("V43 optimizer parameter-name cardinality drift")
    return {
        "optimizer_class": type(optimizer).__name__,
        "parameter_names": list(parameter_names),
        "param_groups": param_groups,
    }


def _validate_optimizer_resume_state(
    optimizer_state: object,
    optimizer: torch.optim.Optimizer,
    parameter_names: list[str],
    expected_optimizer_steps: int,
) -> None:
    if not isinstance(optimizer_state, dict) or set(optimizer_state) != {
        "state",
        "param_groups",
    }:
        raise RuntimeError("V43 resume optimizer structure drift")
    expected_groups = optimizer.state_dict()["param_groups"]
    if optimizer_state["param_groups"] != expected_groups:
        raise RuntimeError("V43 resume optimizer hyperparameter/group drift")
    if not _nested_state_is_finite(optimizer_state):
        raise RuntimeError("V43 resume optimizer contains non-finite state")

    parameter_ids = [
        parameter_id
        for group in expected_groups
        for parameter_id in group["params"]
    ]
    if len(parameter_ids) != len(parameter_names) or len(set(parameter_ids)) != len(
        parameter_ids
    ):
        raise RuntimeError("V43 resume optimizer parameter-group drift")
    state = optimizer_state["state"]
    if not isinstance(state, dict) or set(state) != set(parameter_ids):
        raise RuntimeError("V43 resume optimizer state coverage drift")

    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    for parameter_id, parameter in zip(parameter_ids, parameters, strict=True):
        parameter_state = state[parameter_id]
        if not isinstance(parameter_state, dict):
            raise RuntimeError("V43 resume optimizer parameter state drift")
        for name in ("exp_avg", "exp_avg_sq"):
            value = parameter_state.get(name)
            if not isinstance(value, torch.Tensor) or value.shape != parameter.shape:
                raise RuntimeError("V43 resume optimizer moment-shape drift")
        step = parameter_state.get("step")
        if (
            not isinstance(step, torch.Tensor)
            or step.numel() != 1
            or float(step.detach().cpu().item()) != float(expected_optimizer_steps)
        ):
            raise RuntimeError("V43 resume optimizer step-state drift")


def _configure_device(name: str, torch_threads: int) -> torch.device:
    torch.set_num_threads(int(torch_threads))
    torch.use_deterministic_algorithms(True)
    if name != "mps":
        raise ValueError("V43 smoke/full require MPS with no CPU fallback")
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was requested but is unavailable; run V43 outside the sandbox"
        )
    return torch.device("mps")


def _serialize_filters(
    filters: list[tuple[str, str, object]],
) -> list[list[object]]:
    serialized = []
    for column, operation, value in filters:
        if isinstance(value, pd.Timestamp):
            value = value.isoformat()
        elif isinstance(value, (list, tuple)):
            value = list(value)
        serialized.append([column, operation, value])
    return serialized


def _availability_from_index(
    index: pd.DataFrame,
) -> dict[pd.Timestamp, list[str]]:
    return {
        pd.Timestamp(date): sorted(frame["symbol"].tolist())
        for date, frame in index.groupby("date", sort=True)
    }


def _eligible_pair_count(
    availability: dict[pd.Timestamp, list[str]],
) -> int:
    return int(sum(math.comb(len(symbols), 3) for symbols in availability.values()))


def read_fold_feature_data(
    panel_path: Path,
    sequence_path: Path,
    fold_entry: dict,
    data_access: dict,
    *,
    reader: Callable[..., pd.DataFrame] = pd.read_parquet,
) -> FoldFeatureData:
    train_symbols = sorted(fold_entry["train_symbols"])
    test_symbols = set(fold_entry["test_symbols"])
    fold = int(fold_entry["fold"])
    if len(train_symbols) != 20 or len(test_symbols) != 10:
        raise RuntimeError("V43 fold cardinality drift")
    if set(train_symbols).intersection(test_symbols | TARGET_SYMBOLS):
        raise RuntimeError("V43 fold contains held-out or target overlap")

    minimum_date = pd.Timestamp(
        data_access["representation_train_start"], tz="UTC"
    )
    maximum_date = pd.Timestamp(data_access["maximum_loaded_date"], tz="UTC")
    train_end = pd.Timestamp(data_access["representation_train_end"], tz="UTC")
    validation_start = pd.Timestamp(
        data_access["feature_only_validation_start"], tz="UTC"
    )
    validation_end = pd.Timestamp(
        data_access["feature_only_validation_end"], tz="UTC"
    )
    panel_columns = list(data_access["panel_columns"])
    sequence_columns = list(data_access["sequence_columns"])
    forbidden_columns = {
        column
        for column in panel_columns + sequence_columns
        if column.startswith(("target_", "label_"))
    }
    if forbidden_columns:
        raise RuntimeError(f"V43 requested forbidden columns: {forbidden_columns}")

    panel_filters = [
        ("symbol", "in", train_symbols),
        ("date", ">=", minimum_date),
        ("date", "<=", maximum_date),
    ]
    train_filters = [
        ("symbol", "in", train_symbols),
        ("in_representation_train", "==", True),
        ("date", ">=", minimum_date),
        ("date", "<=", train_end),
    ]
    validation_filters = [
        ("symbol", "in", train_symbols),
        ("in_validation", "==", True),
        ("date", ">=", validation_start),
        ("date", "<=", validation_end),
    ]
    panel = reader(
        panel_path,
        engine="pyarrow",
        columns=panel_columns,
        filters=panel_filters,
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
            "dataset": "panel",
            "columns": panel_columns,
            "filters": _serialize_filters(panel_filters),
        },
        {
            "dataset": "sequence_train",
            "columns": sequence_columns,
            "filters": _serialize_filters(train_filters),
        },
        {
            "dataset": "sequence_validation",
            "columns": sequence_columns,
            "filters": _serialize_filters(validation_filters),
        },
    ]

    for frame, columns, label in (
        (panel, panel_columns, "panel"),
        (train_index, sequence_columns, "train index"),
        (validation_index, sequence_columns, "validation index"),
    ):
        if list(frame.columns) != columns:
            raise RuntimeError(f"V43 {label} projection drift")
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        frame["symbol"] = frame["symbol"].astype(str)
        if frame.duplicated(["date", "symbol"]).any():
            raise RuntimeError(f"V43 {label} contains duplicate keys")
        loaded_symbols = set(frame["symbol"].unique())
        if loaded_symbols != set(train_symbols):
            raise RuntimeError(f"V43 {label} ignored the fold symbol filter")
        if loaded_symbols.intersection(test_symbols | TARGET_SYMBOLS):
            raise RuntimeError(f"V43 {label} materialized a forbidden symbol")

    if panel["date"].min() < minimum_date or panel["date"].max() > maximum_date:
        raise RuntimeError("V43 panel ignored the date filter")
    if (
        train_index["date"].min() < minimum_date
        or train_index["date"].max() > train_end
    ):
        raise RuntimeError("V43 train index exceeds representation train")
    if (
        validation_index["date"].min() < validation_start
        or validation_index["date"].max() > validation_end
    ):
        raise RuntimeError("V43 validation index exceeds the frozen window")
    for frame in (train_index, validation_index):
        starts = pd.to_datetime(frame["sequence_start_date"], utc=True)
        expected_starts = frame["date"] - pd.Timedelta(days=255)
        if not bool((starts == expected_starts).all()):
            raise RuntimeError("V43 sequence lookback is not exactly 256 days")

    train_availability = _availability_from_index(train_index)
    validation_availability = _availability_from_index(validation_index)
    expected = data_access["expected_by_fold"][str(fold)]
    base_features = panel_columns[2:]
    scaler_frame = panel.loc[
        panel["date"] <= train_end,
        base_features,
    ].to_numpy(dtype=np.float64)
    finite_scaler_rows = int(np.isfinite(scaler_frame).all(axis=1).sum())
    observed = {
        "panel_rows": len(panel),
        "scaler_finite_train_rows": finite_scaler_rows,
        "train_sequence_rows": len(train_index),
        "validation_sequence_rows": len(validation_index),
        "train_eligible_pairs": _eligible_pair_count(train_availability),
        "validation_eligible_pairs": _eligible_pair_count(validation_availability),
    }
    if observed != {key: int(value) for key, value in expected.items()}:
        raise RuntimeError(
            f"V43 fold {fold} filtered-data counts drifted: {observed}"
        )
    audit = {
        "fold": fold,
        "train_symbols": train_symbols,
        "heldout_symbols_materialized": [],
        "target_symbols_materialized": [],
        "label_column_read_count": 0,
        "post_2024_market_row_count": int(
            (panel["date"] > maximum_date).sum()
        ),
        "physical_row_group_isolation_claimed": False,
        **observed,
        "minimum_panel_date": panel["date"].min().date().isoformat(),
        "maximum_panel_date": panel["date"].max().date().isoformat(),
    }
    return FoldFeatureData(
        panel=panel,
        train_availability=train_availability,
        validation_availability=validation_availability,
        audit=audit,
        receipts=receipts,
    )


def _register_forbidden_forward_guards(
    model: MultiAssetPatchTransformer,
) -> list[torch.utils.hooks.RemovableHandle]:
    def reject_forward(_module: nn.Module, _inputs: tuple[object, ...]) -> None:
        raise RuntimeError("V43 invoked a forbidden cross-asset/head forward")

    modules: list[nn.Module] = [
        model.cross_asset_encoder,
        model.cross_asset_norm,
        *model.prediction_heads.values(),
    ]
    return [module.register_forward_pre_hook(reject_forward) for module in modules]


def _run_reconstruction_batches(
    model: MultiAssetPatchTransformer,
    store: TripletTensorStore,
    scaler: FeatureScaler,
    samples: list[dict[str, object]],
    *,
    batch_size: int,
    seed: int,
    fold: int,
    mask_epoch: int,
    mask_fraction: float,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float,
) -> tuple[float, int]:
    training = optimizer is not None
    model.train(training)
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    total_loss = 0.0
    observations = 0
    for batch_index, start in enumerate(range(0, len(samples), batch_size)):
        batch_samples = samples[start : start + batch_size]
        values = store.materialize_batch(batch_samples, scaler)
        x = torch.from_numpy(values).to(device)
        mask = deterministic_patch_mask(
            len(batch_samples),
            model.triplet_size,
            model.patch_count,
            mask_fraction,
            seed,
            fold,
            mask_epoch,
            batch_index,
        ).to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            targets = model.extract_patches(x)
            reconstruction = model.reconstruct_masked_patches(x, mask)
            loss = masked_reconstruction_loss(reconstruction, targets, mask)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("V43 produced non-finite reconstruction loss")
        if training:
            loss.backward()
            if any(
                parameter.grad is None
                or not bool(torch.isfinite(parameter.grad).all())
                for parameter in parameters
            ):
                raise RuntimeError("V43 produced missing or non-finite gradients")
            if any(
                parameter.grad is not None
                for name, parameter in model.named_parameters()
                if name.startswith(FORBIDDEN_PRETRAINING_PREFIXES)
            ):
                raise RuntimeError("V43 produced a forbidden gradient")
            gradient_norm = nn.utils.clip_grad_norm_(
                parameters, float(gradient_clip_norm)
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("V43 produced non-finite gradient norm")
            optimizer.step()
        count = len(batch_samples)
        total_loss += float(loss.detach().cpu()) * count
        observations += count
    if observations == 0:
        raise RuntimeError("V43 reconstruction received no samples")
    return total_loss / observations, math.ceil(observations / batch_size)


def _save_resume(
    path: Path,
    *,
    model: MultiAssetPatchTransformer,
    optimizer: torch.optim.Optimizer,
    early_stopping: PretrainingEarlyStopping,
    history: list[dict[str, object]],
    completed_epoch: int,
    metadata: dict[str, object],
    best_model_state: dict[str, torch.Tensor],
    device: torch.device,
    format_version: str,
    best_state_format: str,
    optimizer_parameter_names: list[str],
) -> None:
    best_hash = model_state_sha256(best_model_state)
    optimizer_state = _to_cpu(optimizer.state_dict())
    _atomic_torch_save(
        {
            "format_version": format_version,
            "metadata": metadata,
            "completed_epoch": int(completed_epoch),
            "model_state_dict": _cpu_state_dict(model.state_dict()),
            "model_state_sha256": model_state_sha256(model.state_dict()),
            "optimizer_state_dict": optimizer_state,
            "optimizer_state_sha256": _semantic_state_sha256(optimizer_state),
            "optimizer_contract": _optimizer_contract(
                optimizer, optimizer_parameter_names
            ),
            "early_stopping": asdict(early_stopping),
            "history": history,
            "best_model_state_dict": _cpu_state_dict(best_model_state),
            "best_model_state_sha256": best_hash,
            "best_state_format": best_state_format,
            "rng_state": _rng_state(device),
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
) -> tuple[
    int,
    PretrainingEarlyStopping,
    list[dict[str, object]],
    dict[str, torch.Tensor],
]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != format_version:
        raise RuntimeError("Unsupported or old V43 resume checkpoint")
    if payload.get("best_state_format") != expected_best_state_format:
        raise RuntimeError("Unsupported or old V43 best-state checkpoint")
    if payload.get("metadata") != expected_metadata:
        raise RuntimeError("V43 resume metadata drift")
    completed_epoch = int(payload.get("completed_epoch", -1))
    history = list(payload.get("history", []))
    early_stopping = PretrainingEarlyStopping(**payload["early_stopping"])
    expected_epochs = list(range(1, completed_epoch + 1))
    observed_epochs = [int(row["epoch"]) for row in history]
    validation_losses = [float(row["validation_loss"]) for row in history]
    optimizer_steps = [int(row.get("train_optimizer_steps", -1)) for row in history]
    if (
        completed_epoch < 1
        or completed_epoch > int(maximum_epochs)
        or observed_epochs != expected_epochs
        or any(not math.isfinite(loss) for loss in validation_losses)
        or any(steps <= 0 for steps in optimizer_steps)
    ):
        raise RuntimeError("V43 resume epoch/history coherence drift")
    minimum_loss = min(validation_losses)
    minimum_epoch = observed_epochs[validation_losses.index(minimum_loss)]
    if (
        early_stopping.best_epoch != minimum_epoch
        or not math.isclose(early_stopping.best_loss, minimum_loss)
        or early_stopping.patience != int(expected_patience)
        or early_stopping.stale_epochs
        != completed_epoch - early_stopping.best_epoch
        or early_stopping.should_stop
        != (early_stopping.stale_epochs >= early_stopping.patience)
    ):
        raise RuntimeError("V43 resume early-stopping coherence drift")
    if model_state_sha256(payload["best_model_state_dict"]) != payload.get(
        "best_model_state_sha256"
    ):
        raise RuntimeError("V43 resume best-state hash drift")
    if model_state_sha256(payload["model_state_dict"]) != payload.get(
        "model_state_sha256"
    ):
        raise RuntimeError("V43 resume current-state hash drift")
    if not _state_is_finite(payload["model_state_dict"]) or not _state_is_finite(
        payload["best_model_state_dict"]
    ):
        raise RuntimeError("V43 resume contains non-finite model state")
    expected_optimizer_contract = _optimizer_contract(
        optimizer, expected_optimizer_parameter_names
    )
    if payload.get("optimizer_contract") != expected_optimizer_contract:
        raise RuntimeError("V43 resume optimizer contract drift")
    _validate_optimizer_resume_state(
        payload.get("optimizer_state_dict"),
        optimizer,
        expected_optimizer_parameter_names,
        sum(optimizer_steps),
    )
    if _semantic_state_sha256(payload["optimizer_state_dict"]) != payload.get(
        "optimizer_state_sha256"
    ):
        raise RuntimeError("V43 resume optimizer semantic hash drift")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    _move_optimizer_state(optimizer, device)
    _restore_rng_state(payload["rng_state"], device)
    return (
        completed_epoch,
        early_stopping,
        history,
        _cpu_state_dict(payload["best_model_state_dict"]),
    )


def load_ranking_excess_pretrained_checkpoint(
    path: str | Path,
    *,
    expected_architecture: dict | None = None,
    expected_metadata: dict | None = None,
) -> tuple[MultiAssetPatchTransformer, dict]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format_version") != "v43_ranking_excess_pretraining_v1":
        raise RuntimeError("Unsupported or old ranking/excess pretraining checkpoint")
    architecture = payload["architecture"]
    if payload.get("architecture_sha256") != _canonical_sha256(architecture):
        raise RuntimeError("V43 checkpoint architecture hash drift")
    if expected_architecture is not None and architecture != expected_architecture:
        raise RuntimeError("V43 checkpoint architecture does not match V41")
    if int(payload.get("input_features", -1)) != 9:
        raise RuntimeError("V43 checkpoint input-feature drift")
    if expected_metadata is not None and payload.get("metadata") != expected_metadata:
        raise RuntimeError("V43 checkpoint metadata drift")
    if model_state_sha256(payload["state_dict"]) != payload.get(
        "model_state_sha256"
    ):
        raise RuntimeError("V43 checkpoint model-state hash drift")
    if not _state_is_finite(payload["state_dict"]):
        raise RuntimeError("V43 checkpoint contains non-finite model state")
    metadata = payload.get("metadata", {})
    if (
        metadata.get("initialization_source") != "fresh_registered_seed"
        or metadata.get("parent_checkpoint_sha256") is not None
        or metadata.get("checkpoint_status")
        != "frozen_pretrained_no_seed_or_fold_selection"
        or set(metadata.get("unused_during_pretraining", []))
        != {"cross_asset_encoder", "cross_asset_norm", "prediction_heads"}
    ):
        raise RuntimeError("V43 checkpoint semantic metadata drift")
    model = MultiAssetPatchTransformer(
        9,
        architecture,
        expected_prediction_heads=RANKING_EXCESS_HEADS,
    )
    expected_pretraining_parameters = sum(
        parameter.numel() for parameter in configure_pretraining_scope(model)
    )
    if int(metadata.get("pretraining_parameter_count", -1)) != int(
        expected_pretraining_parameters
    ):
        raise RuntimeError("V43 checkpoint pretraining scope drift")
    model.load_state_dict(payload["state_dict"])
    return model, payload


def build_pretraining_spec(
    blueprint: dict,
    pretraining: dict,
    mode: str,
    prior_gate: dict[str, object] | None = None,
) -> dict[str, object]:
    if mode not in {"preflight", "smoke", "full"}:
        raise ValueError("V43 mode must be preflight, smoke, or full")
    effective = (
        pretraining["smoke"]
        if mode == "smoke"
        else pretraining["full_run"]
    )
    spec = {
        "version": "v43",
        "candidate_family_id": blueprint["candidate_family_id"],
        "mode": mode,
        "phase": "masked_patch_non_target_representation_pretraining",
        "architecture": blueprint["architecture"],
        "folds": list(effective["folds"]),
        "seeds": list(effective["seeds"]),
        "training_window": blueprint["chronological_splits"][
            "representation_train"
        ],
        "feature_only_validation_window": [
            pretraining["data_access"]["feature_only_validation_start"],
            pretraining["data_access"]["feature_only_validation_end"],
        ],
        "train_samples_per_epoch": int(effective["train_samples_per_epoch"]),
        "validation_samples": int(effective["validation_samples"]),
        "validation_sampling_epoch": int(
            pretraining["validation_sampling_epoch"]
        ),
        "batch_size": int(effective["batch_size"]),
        "maximum_epochs": int(effective["maximum_epochs"]),
        "early_stopping_patience": int(
            effective["early_stopping_patience"]
        ),
        "optimizer": {
            "name": "AdamW",
            "learning_rate": blueprint["training"]["learning_rate"],
            "weight_decay": blueprint["training"]["weight_decay"],
            "gradient_clip_norm": pretraining["gradient_clip_norm"],
            "scheduler": None,
        },
        "mask_fraction": pretraining["mask_fraction"],
        "loss": pretraining["loss"],
        "validation_monitor": pretraining["validation_monitor"],
        "device": pretraining["device"],
        "dtype": "float32",
        "amp": False,
        "fresh_initialization": pretraining["initialization"],
        "data_access": pretraining["data_access"],
        "checkpoint_format": pretraining["checkpoint_format"],
        "resume_format": pretraining["resume_format"],
        "seed_selection_allowed": False,
        "supervised_heads_used": False,
        "labels_loaded": False,
        "heldout_assets_loaded": False,
        "target_assets_loaded": False,
        "performance_metrics_allowed": False,
        "prior_gate": prior_gate,
    }
    spec["pretraining_spec_sha256"] = _canonical_sha256(spec)
    return spec


def _metadata_context(config: dict) -> dict[str, object]:
    pretraining = config["ranking_excess_pretraining"]
    root = Path(pretraining["project_root"]).resolve()
    paths = {
        name: root / relative for name, relative in pretraining["inputs"].items()
    }
    if set(paths) != METADATA_INPUT_NAMES | BINARY_INPUT_NAMES:
        raise RuntimeError("V43 input allowlist drift")
    for name in METADATA_INPUT_NAMES:
        path = paths[name]
        if not path.is_file() or _sha256_file(path) != pretraining[
            "expected_input_sha256"
        ][name]:
            raise RuntimeError(f"V43 metadata input drift: {name}")
    for name in BINARY_INPUT_NAMES:
        if not paths[name].is_file():
            raise RuntimeError(f"V43 binary input is missing: {name}")

    values = {name: _load_json(paths[name]) for name in METADATA_INPUT_NAMES}
    blueprint = values["v41_blueprint"]
    if (
        values["v41_specification"]["decision"]
        != "authorize_v42_synthetic_ranking_excess_harness_only"
        or not values["v41_audit"]["passed"]
        or values["v42_result"]["decision"]
        != "authorize_v43_medium_non_target_pretraining_only"
        or not values["v42_audit"]["passed"]
    ):
        raise RuntimeError("V41/V42 do not authorize V43")
    manifest = values["v32_dataset_manifest"]
    feature_schema = values["v32_feature_schema"]
    asset_folds = values["v32_asset_folds"]
    if manifest["panel_sha256"] != pretraining["expected_input_sha256"]["panel"]:
        raise RuntimeError("V43 panel hash differs from V32 manifest")
    if manifest["sequence_index_sha256"] != pretraining[
        "expected_input_sha256"
    ]["sequence_index"]:
        raise RuntimeError("V43 sequence hash differs from V32 manifest")
    if list(feature_schema["model_feature_order"][:-1]) != list(
        manifest["panel_features"]
    ):
        raise RuntimeError("V43 feature-order drift")
    if list(pretraining["data_access"]["panel_columns"][2:]) != list(
        manifest["panel_features"]
    ):
        raise RuntimeError("V43 panel projection differs from V32 features")
    data_access = pretraining["data_access"]
    if (
        list(data_access["panel_columns"][:2]) != ["date", "symbol"]
        or list(data_access["sequence_columns"])
        != ["date", "sequence_start_date", "symbol"]
        or not data_access["per_fold_filtered_read_required"]
        or data_access["labels_allowed"]
        or data_access["heldout_assets_allowed"]
        or data_access["target_assets_allowed"]
        or data_access["post_validation_dates_allowed"]
        or data_access["physical_row_group_isolation_claimed"]
    ):
        raise RuntimeError("V43 data-access permission or projection drift")
    if TARGET_SYMBOLS.intersection(manifest["symbols"]):
        raise RuntimeError("V43 source dataset contains target assets")
    if len(asset_folds["folds"]) != 3:
        raise RuntimeError("V43 requires exactly three asset folds")
    if pretraining["initialization"]["synthetic_v42_checkpoint_allowed"]:
        raise RuntimeError("V43 may not reuse the synthetic V42 checkpoint")
    if any("checkpoint" in name for name in paths):
        raise RuntimeError("V43 input list contains a checkpoint")
    return {
        "root": root,
        "paths": paths,
        "pretraining": pretraining,
        "blueprint": blueprint,
        "manifest": manifest,
        "feature_schema": feature_schema,
        "asset_folds": asset_folds,
    }


def _fresh_initialization_audit(
    architecture: dict,
    seeds: list[int],
    folds: list[int],
) -> tuple[dict[int, str], dict[str, bool], int, int]:
    hashes_by_seed_and_fold: dict[int, set[str]] = {
        int(seed): set() for seed in seeds
    }
    total_parameters = -1
    pretraining_parameters = -1
    for fold in folds:
        for seed in seeds:
            torch.manual_seed(int(seed))
            model = MultiAssetPatchTransformer(
                9,
                architecture,
                expected_prediction_heads=RANKING_EXCESS_HEADS,
            )
            configure_pretraining_scope(model)
            hashes_by_seed_and_fold[int(seed)].add(
                model_state_sha256(model.state_dict())
            )
            total_parameters = sum(
                parameter.numel() for parameter in model.parameters()
            )
            pretraining_parameters = sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            del model
    seed_hashes = {
        seed: next(iter(hashes))
        for seed, hashes in hashes_by_seed_and_fold.items()
        if len(hashes) == 1
    }
    checks = {
        "same_seed_initializes_identically_across_folds": len(seed_hashes)
        == len(seeds),
        "different_seeds_have_distinct_initializations": len(
            set(seed_hashes.values())
        )
        == len(seeds),
    }
    return seed_hashes, checks, total_parameters, pretraining_parameters


def _load_prior_gate(
    root: Path,
    pretraining: dict,
    blueprint: dict,
    mode: str,
) -> dict[str, object] | None:
    if mode == "preflight":
        return None
    if mode == "smoke":
        relative = pretraining["preflight_output_dir"]
        expected_mode = "preflight"
        expected_decision = "authorize_v43_one_job_mps_smoke_only"
        nested_gate = None
    else:
        relative = pretraining["smoke_output_dir"]
        expected_mode = "smoke"
        expected_decision = "authorize_v43_full_nine_job_pretraining_only"
        nested_gate = _load_prior_gate(root, pretraining, blueprint, "smoke")
    path = root / relative / "result.json"
    if not path.is_file():
        raise RuntimeError(f"V43 {mode} requires its prior passing gate")
    result = _load_json(path)
    expected_spec = build_pretraining_spec(
        blueprint,
        pretraining,
        expected_mode,
        prior_gate=nested_gate,
    )
    audit = result.get("audit", {})
    audit_checks = audit.get("checks", {})
    if (
        result.get("decision") != expected_decision
        or result.get("pretraining_spec") != expected_spec
        or not audit.get("passed")
        or not audit_checks
        or not all(bool(value) for value in audit_checks.values())
    ):
        raise RuntimeError(f"V43 prior {expected_mode} gate is invalid")
    summary = result.get("summary", {})
    if (
        int(summary.get("total_parameters", -1))
        != int(pretraining["expected_total_parameters"])
        or int(summary.get("pretraining_parameters", -1))
        != int(pretraining["expected_pretraining_parameters"])
    ):
        raise RuntimeError(f"V43 prior {expected_mode} parameter audit drifted")
    if expected_mode == "preflight":
        if (
            int(summary.get("checkpoint_count", -1)) != 0
            or int(summary.get("total_optimizer_steps", -1)) != 0
            or int(summary.get("parquet_files_deserialized", -1)) != 0
            or result.get("tested", {}).get("panel_or_sequence_deserialized")
            is not False
            or result.get("tested", {}).get("optimizer_executed") is not False
        ):
            raise RuntimeError("V43 fabricated or contaminated preflight gate")
    else:
        manifest = result.get("checkpoint_manifest", [])
        if (
            int(summary.get("checkpoint_count", -1)) != 1
            or int(summary.get("total_optimizer_steps", 0)) <= 0
            or len(manifest) != 1
            or int(manifest[0].get("fold", -1)) != 1
            or int(manifest[0].get("seed", -1)) != 42
        ):
            raise RuntimeError("V43 smoke gate does not contain exactly one job")
        checkpoint_path = Path(str(manifest[0].get("checkpoint_path", ""))).resolve()
        expected_root = (root / pretraining["smoke_checkpoint_dir"]).resolve()
        if (
            not checkpoint_path.is_relative_to(expected_root)
            or not checkpoint_path.is_file()
            or _sha256_file(checkpoint_path)
            != manifest[0].get("checkpoint_sha256")
        ):
            raise RuntimeError("V43 smoke gate checkpoint drifted")
        _, payload = load_ranking_excess_pretrained_checkpoint(
            checkpoint_path,
            expected_architecture=blueprint["architecture"],
        )
        if payload["metadata"].get("pretraining_spec_sha256") != expected_spec[
            "pretraining_spec_sha256"
        ]:
            raise RuntimeError("V43 smoke checkpoint spec drifted")
        tested = result.get("tested", {})
        if any(
            tested.get(name) is not False
            for name in (
                "labels_loaded",
                "heldout_assets_loaded",
                "target_assets_loaded",
                "supervised_predictions_computed",
                "performance_metrics_computed",
                "pnl_computed",
                "seed_or_fold_selection_executed",
            )
        ):
            raise RuntimeError("V43 smoke gate contains a forbidden operation")
    return {
        "mode": expected_mode,
        "decision": expected_decision,
        "result_sha256": _sha256_file(path),
        "pretraining_spec_sha256": result["pretraining_spec"][
            "pretraining_spec_sha256"
        ],
    }


def _completed_job(
    job_dir: Path,
    expected_spec_sha256: str,
    expected_architecture: dict,
) -> dict[str, object] | None:
    complete_path = job_dir / "complete.json"
    if not complete_path.is_file():
        return None
    complete = _load_json(complete_path)
    checkpoint_path = job_dir / "checkpoint.pt"
    if (
        complete.get("pretraining_spec_sha256") != expected_spec_sha256
        or not checkpoint_path.is_file()
        or _sha256_file(checkpoint_path) != complete.get("checkpoint_sha256")
    ):
        raise RuntimeError(f"Completed V43 job drifted: {job_dir}")
    _, payload = load_ranking_excess_pretrained_checkpoint(
        checkpoint_path, expected_architecture=expected_architecture
    )
    if (
        payload["model_state_sha256"] != complete.get("model_state_sha256")
        or payload["metadata"].get("pretraining_spec_sha256")
        != expected_spec_sha256
        or int(payload["metadata"].get("fold", -1))
        != int(complete.get("fold", -2))
        or int(payload["metadata"].get("initialization_seed", -1))
        != int(complete.get("seed", -2))
    ):
        raise RuntimeError(f"Completed V43 state drifted: {job_dir}")
    for path_key, hash_key in (
        ("fold_data_access_path", "fold_data_access_sha256"),
        ("scaler_path", "scaler_artifact_sha256"),
    ):
        artifact_path = Path(str(complete.get(path_key, "")))
        if (
            not artifact_path.is_file()
            or _sha256_file(artifact_path) != complete.get(hash_key)
            or payload["metadata"].get(hash_key) != complete.get(hash_key)
        ):
            raise RuntimeError(f"Completed V43 fold artifact drifted: {job_dir}")
    return complete


def _validate_persisted_fold_access(
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
    receipts_by_dataset = {
        receipt.get("dataset"): receipt for receipt in receipts
    }
    if (
        int(audit.get("fold", -1)) != fold
        or audit.get("train_symbols") != sorted(fold_entry["train_symbols"])
        or audit.get("heldout_symbols_materialized")
        or audit.get("target_symbols_materialized")
        or int(audit.get("label_column_read_count", -1)) != 0
        or int(audit.get("post_2024_market_row_count", -1)) != 0
        or any(int(audit.get(key, -1)) != value for key, value in expected.items())
        or len(receipts) != 3
        or any(int(receipt.get("fold", -1)) != fold for receipt in receipts)
        or set(receipts_by_dataset)
        != {"panel", "sequence_train", "sequence_validation"}
        or receipts_by_dataset.get("panel", {}).get("columns")
        != list(data_access["panel_columns"])
        or receipts_by_dataset.get("sequence_train", {}).get("columns")
        != list(data_access["sequence_columns"])
        or receipts_by_dataset.get("sequence_validation", {}).get("columns")
        != list(data_access["sequence_columns"])
        or any(not receipt.get("filters") for receipt in receipts)
    ):
        raise RuntimeError(f"Persisted V43 fold data audit drifted: {fold}")


def _validate_persisted_scaler(
    path: Path,
    fold_entry: dict,
    data_access: dict,
) -> None:
    record = _load_json(path)
    names = {field.name for field in fields(FeatureScaler)}
    scaler = FeatureScaler(**{name: record[name] for name in names})
    fold = int(fold_entry["fold"])
    expected_rows = int(
        data_access["expected_by_fold"][str(fold)]["scaler_finite_train_rows"]
    )
    if (
        int(record.get("fold", -1)) != fold
        or record.get("train_symbols") != sorted(fold_entry["train_symbols"])
        or record.get("scaler_state_sha256") != scaler.state_sha256()
        or scaler.fit_scope != "representation_train_only"
        or scaler.fit_start != data_access["representation_train_start"]
        or scaler.fit_end != data_access["representation_train_end"]
        or scaler.fit_rows != expected_rows
    ):
        raise RuntimeError(f"Persisted V43 scaler drifted: {fold}")


def _train_job(
    *,
    fold_entry: dict,
    seed: int,
    architecture: dict,
    feature_names: list[str],
    store: TripletTensorStore,
    scaler: FeatureScaler,
    train_availability: dict[pd.Timestamp, list[str]],
    validation_availability: dict[pd.Timestamp, list[str]],
    blueprint: dict,
    pretraining: dict,
    effective: dict,
    pretraining_spec: dict,
    artifact_hashes: dict[str, str],
    fold_data_access_path: Path,
    fold_data_access_sha256: str,
    scaler_path: Path,
    scaler_artifact_sha256: str,
    checkpoint_root: Path,
    device: torch.device,
    expected_initialization_sha256: str,
) -> dict[str, object]:
    fold = int(fold_entry["fold"])
    train_symbols = sorted(fold_entry["train_symbols"])
    test_symbols = sorted(fold_entry["test_symbols"])
    job_dir = checkpoint_root / f"fold_{fold}" / f"seed_{seed}"
    job_dir.mkdir(parents=True, exist_ok=True)
    completed = _completed_job(
        job_dir,
        pretraining_spec["pretraining_spec_sha256"],
        architecture,
    )
    if completed is not None:
        return completed
    if set(train_symbols).intersection(set(test_symbols) | TARGET_SYMBOLS):
        raise RuntimeError("V43 job has forbidden asset overlap")

    _seed_device(seed, device)
    model = MultiAssetPatchTransformer(
        9,
        architecture,
        expected_prediction_heads=RANKING_EXCESS_HEADS,
    )
    initialization_sha256 = model_state_sha256(model.state_dict())
    if initialization_sha256 != expected_initialization_sha256:
        raise RuntimeError("V43 fresh initialization hash drift")
    parameters = configure_pretraining_scope(model)
    parameter_names = pretraining_parameter_names(model)
    initial_forbidden_state_sha256 = _state_subset_sha256(
        model, FORBIDDEN_PRETRAINING_PREFIXES
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(blueprint["training"]["learning_rate"]),
        weight_decay=float(blueprint["training"]["weight_decay"]),
    )
    metadata = {
        "version": "v43",
        "candidate_family_id": blueprint["candidate_family_id"],
        "fold": fold,
        "initialization_seed": int(seed),
        "initialization_source": "fresh_registered_seed",
        "initialization_state_sha256": initialization_sha256,
        "parent_checkpoint_sha256": None,
        "train_symbols": train_symbols,
        "test_symbols": test_symbols,
        "scaler_state_sha256": scaler.state_sha256(),
        "fold_data_access_sha256": fold_data_access_sha256,
        "scaler_artifact_sha256": scaler_artifact_sha256,
        "pretraining_spec_sha256": pretraining_spec["pretraining_spec_sha256"],
        **artifact_hashes,
    }
    train_sampler = DeterministicEligibleTripletSampler(
        train_availability, train_symbols, seed, fold
    )
    validation_sampler = DeterministicEligibleTripletSampler(
        validation_availability, train_symbols, seed, fold
    )
    validation_epoch = int(pretraining["validation_sampling_epoch"])
    validation_samples = validation_sampler.sample_epoch(
        validation_epoch, int(effective["validation_samples"])
    )
    early_stopping = PretrainingEarlyStopping(
        patience=int(effective["early_stopping_patience"])
    )
    history: list[dict[str, object]] = []
    completed_epoch = 0
    best_model_state = _cpu_state_dict(model.state_dict())
    resume_path = job_dir / "resume.pt"
    if resume_path.is_file():
        (
            completed_epoch,
            early_stopping,
            history,
            best_model_state,
        ) = _load_resume(
            resume_path,
            model=model,
            optimizer=optimizer,
            expected_metadata=metadata,
            device=device,
            format_version=pretraining["resume_format"],
            expected_best_state_format=pretraining["best_state_format"],
            expected_patience=int(effective["early_stopping_patience"]),
            maximum_epochs=int(effective["maximum_epochs"]),
            expected_optimizer_parameter_names=parameter_names,
        )

    hooks = _register_forbidden_forward_guards(model)
    try:
        if not early_stopping.should_stop:
            for epoch in range(
                completed_epoch + 1,
                int(effective["maximum_epochs"]) + 1,
            ):
                train_samples = train_sampler.sample_epoch(
                    epoch, int(effective["train_samples_per_epoch"])
                )
                train_loss, train_steps = _run_reconstruction_batches(
                    model,
                    store,
                    scaler,
                    train_samples,
                    batch_size=int(effective["batch_size"]),
                    seed=seed,
                    fold=fold,
                    mask_epoch=epoch,
                    mask_fraction=float(pretraining["mask_fraction"]),
                    device=device,
                    optimizer=optimizer,
                    gradient_clip_norm=float(pretraining["gradient_clip_norm"]),
                )
                validation_loss, validation_steps = _run_reconstruction_batches(
                    model,
                    store,
                    scaler,
                    validation_samples,
                    batch_size=int(effective["batch_size"]),
                    seed=seed,
                    fold=fold,
                    mask_epoch=validation_epoch,
                    mask_fraction=float(pretraining["mask_fraction"]),
                    device=device,
                    optimizer=None,
                    gradient_clip_norm=float(pretraining["gradient_clip_norm"]),
                )
                prior_best = early_stopping.best_loss
                should_stop = early_stopping.update(epoch, validation_loss)
                if validation_loss < prior_best:
                    best_model_state = _cpu_state_dict(model.state_dict())
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "validation_loss": validation_loss,
                        "improved": validation_loss < prior_best,
                        "train_optimizer_steps": train_steps,
                        "validation_steps": validation_steps,
                    }
                )
                _save_resume(
                    resume_path,
                    model=model,
                    optimizer=optimizer,
                    early_stopping=early_stopping,
                    history=history,
                    completed_epoch=epoch,
                    metadata=metadata,
                    best_model_state=best_model_state,
                    device=device,
                    format_version=pretraining["resume_format"],
                    best_state_format=pretraining["best_state_format"],
                    optimizer_parameter_names=parameter_names,
                )
                _write_json(
                    job_dir / "progress.json",
                    {
                        "fold": fold,
                        "seed": seed,
                        "completed_epoch": epoch,
                        "best_epoch": early_stopping.best_epoch,
                        "best_validation_loss": early_stopping.best_loss,
                        "stale_epochs": early_stopping.stale_epochs,
                        "should_stop": should_stop,
                        "last_epoch": history[-1],
                    },
                )
                if should_stop:
                    break
    finally:
        for hook in hooks:
            hook.remove()

    if early_stopping.best_epoch < 0 or not history:
        raise RuntimeError("V43 job produced no best reconstruction state")
    if any(
        value.is_floating_point() and not bool(torch.isfinite(value).all())
        for value in best_model_state.values()
    ):
        raise RuntimeError("V43 best state contains non-finite values")
    model.load_state_dict(best_model_state)
    final_forbidden_state_sha256 = _state_subset_sha256(
        model, FORBIDDEN_PRETRAINING_PREFIXES
    )
    if final_forbidden_state_sha256 != initial_forbidden_state_sha256:
        raise RuntimeError("V43 modified a forbidden pretraining module")
    if any(
        parameter.grad is not None
        for name, parameter in model.named_parameters()
        if name.startswith(FORBIDDEN_PRETRAINING_PREFIXES)
    ):
        raise RuntimeError("V43 left a forbidden gradient")

    state_dict = _cpu_state_dict(model.state_dict())
    state_sha256 = model_state_sha256(state_dict)
    checkpoint_metadata = {
        **metadata,
        "checkpoint_status": "frozen_pretrained_no_seed_or_fold_selection",
        "best_epoch": int(early_stopping.best_epoch),
        "best_validation_loss": float(early_stopping.best_loss),
        "completed_epochs": len(history),
        "pretraining_parameter_names": parameter_names,
        "pretraining_parameter_count": int(
            sum(parameter.numel() for parameter in parameters)
        ),
        "unused_during_pretraining": [
            "cross_asset_encoder",
            "cross_asset_norm",
            "prediction_heads",
        ],
        "forbidden_state_sha256": final_forbidden_state_sha256,
    }
    checkpoint_payload = {
        "format_version": pretraining["checkpoint_format"],
        "input_features": 9,
        "architecture": architecture,
        "architecture_sha256": _canonical_sha256(architecture),
        "metadata": checkpoint_metadata,
        "model_state_sha256": state_sha256,
        "state_dict": state_dict,
    }
    checkpoint_path = job_dir / "checkpoint.pt"
    _atomic_torch_save(checkpoint_payload, checkpoint_path)
    restored_model, restored_payload = load_ranking_excess_pretrained_checkpoint(
        checkpoint_path,
        expected_architecture=architecture,
        expected_metadata=checkpoint_metadata,
    )
    if model_state_sha256(restored_model.state_dict()) != state_sha256:
        raise RuntimeError("V43 checkpoint semantic roundtrip drift")
    del restored_model, restored_payload
    complete = {
        "version": "v43",
        "fold": fold,
        "seed": int(seed),
        "train_symbols": train_symbols,
        "test_symbols": test_symbols,
        "initialization_source": "fresh_registered_seed",
        "initialization_state_sha256": initialization_sha256,
        "parent_checkpoint_sha256": None,
        "scaler_state_sha256": scaler.state_sha256(),
        "scaler_fit_rows": scaler.fit_rows,
        "fold_data_access_path": str(fold_data_access_path),
        "fold_data_access_sha256": fold_data_access_sha256,
        "scaler_path": str(scaler_path),
        "scaler_artifact_sha256": scaler_artifact_sha256,
        "pretraining_spec_sha256": pretraining_spec["pretraining_spec_sha256"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "model_state_sha256": state_sha256,
        "forbidden_initial_state_sha256": initial_forbidden_state_sha256,
        "forbidden_final_state_sha256": final_forbidden_state_sha256,
        "best_epoch": int(early_stopping.best_epoch),
        "best_validation_loss": float(early_stopping.best_loss),
        "completed_epochs": len(history),
        "train_optimizer_steps": int(
            sum(row["train_optimizer_steps"] for row in history)
        ),
        "history": history,
        "seed_selected": False,
        "fold_selected": False,
        "labels_loaded": False,
        "heldout_assets_loaded": False,
        "target_assets_loaded": False,
        "supervised_heads_used": False,
        "performance_metrics_computed": False,
        "pnl_computed": False,
    }
    _write_json(job_dir / "complete.json", complete)
    resume_path.unlink(missing_ok=True)
    return complete


def _report(result: dict[str, object]) -> str:
    mode = result["pretraining_spec"]["mode"]
    if mode == "preflight":
        status = "METADATA-ONLY PREFLIGHT PASSED; ONE-JOB MPS SMOKE IS NEXT."
    elif mode == "smoke":
        status = "ONE-JOB MPS SMOKE PASSED; FULL NINE-JOB PRETRAINING IS NEXT."
    else:
        status = "ALL NINE MEDIUM NON-TARGET PRETRAINING JOBS PASSED."
    summary = result["summary"]
    return "\n".join(
        [
            "# TLM v43 Ranking/Excess Medium Pretraining",
            "",
            "## Decision",
            "",
            f"**{status}**",
            "",
            f"Mode: **{mode}**",
            f"Pretraining spec SHA-256: `{result['pretraining_spec']['pretraining_spec_sha256']}`",
            f"Checkpoints: **{summary['checkpoint_count']}**",
            f"Optimizer steps: **{summary['total_optimizer_steps']:,}**",
            "",
            "Each fold is loaded through projected feature-only columns and train-symbol/date filters. The scaler sees only 2021-2023 train-asset rows; fixed 2024 samples monitor reconstruction only.",
            "",
            "No forward label, held-out asset row, BTC/ETH/SOL row, supervised prediction, portfolio, performance metric, or PnL was loaded or computed.",
            "",
            "## Next action",
            "",
            "A passing preflight authorizes only the smoke. A passing smoke authorizes only the full V43 run. A passing full run authorizes only V44 supervised non-target training.",
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
    _write_json(output / "result.json", result)
    _write_json(output / "pretraining_spec.json", result["pretraining_spec"])
    _write_json(output / "audit.json", result["audit"])
    if "checkpoint_manifest" in result:
        _write_json(output / "checkpoint_manifest.json", result["checkpoint_manifest"])
    if "data_access_audit" in result:
        _write_json(output / "data_access_audit.json", result["data_access_audit"])
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output / "report.md").write_text(_report(result), encoding="utf-8")


def run_ranking_excess_pretraining(
    config: dict,
    mode: str,
) -> dict[str, object]:
    context = _metadata_context(config)
    root = context["root"]
    paths = context["paths"]
    pretraining = context["pretraining"]
    blueprint = context["blueprint"]
    asset_folds = context["asset_folds"]
    if mode not in {"preflight", "smoke", "full"}:
        raise ValueError("V43 mode must be preflight, smoke, or full")
    prior_gate = _load_prior_gate(root, pretraining, blueprint, mode)
    spec = build_pretraining_spec(
        blueprint, pretraining, mode, prior_gate=prior_gate
    )
    effective = pretraining["smoke"] if mode == "smoke" else pretraining["full_run"]
    seeds = [int(seed) for seed in effective["seeds"]]
    folds = [int(fold) for fold in effective["folds"]]
    seed_hashes, initialization_checks, total_parameters, trainable_parameters = (
        _fresh_initialization_audit(
            blueprint["architecture"], seeds, folds
        )
    )
    contract_checks = {
        **initialization_checks,
        "prior_gate_is_exact_for_execution_mode": (
            mode == "preflight" and prior_gate is None
        )
        or (mode != "preflight" and prior_gate is not None),
        "input_allowlist_excludes_every_checkpoint": not any(
            "checkpoint" in name for name in paths
        ),
        "total_parameter_count_is_frozen": total_parameters
        == int(pretraining["expected_total_parameters"]),
        "pretraining_parameter_count_is_frozen": trainable_parameters
        == int(pretraining["expected_pretraining_parameters"]),
        "registered_device_is_mps_without_fallback": pretraining["device"] == "mps",
        "registered_loss_and_mask_are_exact": pretraining["loss"]
        == "smooth_l1_beta_1_masked_patches_only"
        and pretraining["validation_monitor"]
        == "fixed_2024_feature_only_masked_reconstruction_loss"
        and float(pretraining["mask_fraction"]) == 0.15
        and float(pretraining["gradient_clip_norm"])
        == float(blueprint["training"]["gradient_clip_norm"])
        and int(pretraining["validation_sampling_epoch"]) == 0,
        "checkpoint_formats_are_new_and_exact": pretraining["checkpoint_format"]
        == "v43_ranking_excess_pretraining_v1"
        and pretraining["resume_format"]
        == "v43_ranking_excess_resume_v1"
        and pretraining["best_state_format"]
        == "v43_ranking_excess_best_state_v1",
        "full_fold_seed_grid_is_exact": list(pretraining["full_run"]["folds"])
        == [1, 2, 3]
        and list(pretraining["full_run"]["seeds"]) == [42, 7, 123],
        "smoke_fold_seed_grid_is_exact": list(pretraining["smoke"]["folds"])
        == [1]
        and list(pretraining["smoke"]["seeds"]) == [42],
        "blueprint_training_contract_matches": int(
            pretraining["full_run"]["train_samples_per_epoch"]
        )
        == int(blueprint["training"]["pretrain_samples_per_epoch"])
        and int(pretraining["full_run"]["validation_samples"])
        == int(blueprint["training"]["fixed_validation_samples"])
        and int(pretraining["full_run"]["batch_size"])
        == int(blueprint["training"]["batch_size"])
        and int(pretraining["full_run"]["maximum_epochs"])
        == int(blueprint["training"]["maximum_pretrain_epochs"])
        and int(pretraining["full_run"]["early_stopping_patience"])
        == int(blueprint["training"]["early_stopping_patience"]),
        "synthetic_and_old_checkpoint_reuse_is_forbidden": not pretraining[
            "initialization"
        ]["old_checkpoint_reuse_allowed"]
        and not pretraining["initialization"][
            "synthetic_v42_checkpoint_allowed"
        ]
        and pretraining["initialization"]["parent_checkpoint_sha256"] is None,
        "fresh_initialization_source_is_exact": pretraining["initialization"][
            "source"
        ]
        == "fresh_registered_seed",
        "chronology_is_exact": pretraining["data_access"][
            "representation_train_start"
        ]
        == blueprint["chronological_splits"]["representation_train"][0]
        and pretraining["data_access"]["representation_train_end"]
        == blueprint["chronological_splits"]["representation_train"][1]
        and pretraining["data_access"]["feature_only_validation_start"]
        == blueprint["chronological_splits"][
            "early_stopping_train_assets_only"
        ][0]
        and pretraining["data_access"]["feature_only_validation_end"]
        == blueprint["chronological_splits"][
            "early_stopping_train_assets_only"
        ][1]
        and pretraining["data_access"]["maximum_loaded_date"]
        == blueprint["chronological_splits"][
            "early_stopping_train_assets_only"
        ][1],
        "authorized_next_action_is_exact": pretraining["authorized_next_action"]
        == "v44_ranking_excess_supervised_non_target_only",
    }
    contract_checks = {key: bool(value) for key, value in contract_checks.items()}
    if not all(contract_checks.values()):
        raise RuntimeError(f"V43 contract audit failed: {contract_checks}")

    mps_available = bool(torch.backends.mps.is_available())
    if mode != "preflight" and not mps_available:
        raise RuntimeError(
            "V43 requires an available MPS device and forbids CPU fallback"
        )
    if mode == "preflight":
        result = {
            "version": "v43_preflight",
            "decision": "authorize_v43_one_job_mps_smoke_only",
            "pretraining_spec": spec,
            "summary": {
                "checkpoint_count": 0,
                "total_optimizer_steps": 0,
                "total_parameters": total_parameters,
                "pretraining_parameters": trainable_parameters,
                "registered_initialization_state_sha256": {
                    str(seed): value for seed, value in seed_hashes.items()
                },
                "mps_available": mps_available,
                "parquet_files_deserialized": 0,
            },
            "tested": {
                "metadata_inputs_loaded": True,
                "panel_or_sequence_deserialized": False,
                "optimizer_executed": False,
                "labels_loaded": False,
                "target_assets_loaded": False,
                "performance_metrics_computed": False,
                "pnl_computed": False,
            },
            "audit": {"passed": True, "checks": contract_checks},
        }
        _write_result(
            root,
            pretraining["preflight_output_dir"],
            config,
            result,
        )
        return result

    for name in BINARY_INPUT_NAMES:
        if _sha256_file(paths[name]) != pretraining["expected_input_sha256"][name]:
            raise RuntimeError(f"V43 binary input hash drift: {name}")
    device = _configure_device(pretraining["device"], pretraining["torch_threads"])
    feature_names = list(context["manifest"]["panel_features"])
    folds_by_number = {
        int(entry["fold"]): entry for entry in asset_folds["folds"]
    }
    checkpoint_root = root / (
        pretraining["smoke_checkpoint_dir"]
        if mode == "smoke"
        else pretraining["checkpoint_dir"]
    )
    artifact_hashes = {
        "v41_blueprint_sha256": context["blueprint"]["blueprint_sha256"],
        "v42_result_sha256": pretraining["expected_input_sha256"]["v42_result"],
        "dataset_manifest_sha256": pretraining["expected_input_sha256"][
            "v32_dataset_manifest"
        ],
        "feature_schema_sha256": pretraining["expected_input_sha256"][
            "v32_feature_schema"
        ],
        "asset_folds_sha256": pretraining["expected_input_sha256"][
            "v32_asset_folds"
        ],
        "panel_sha256": pretraining["expected_input_sha256"]["panel"],
        "sequence_index_sha256": pretraining["expected_input_sha256"][
            "sequence_index"
        ],
    }
    jobs: list[dict[str, object]] = []
    data_audits: list[dict[str, object]] = []
    read_receipts: list[dict[str, object]] = []
    for fold in folds:
        fold_entry = folds_by_number[fold]
        fold_dir = checkpoint_root / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        fold_data_access_path = fold_dir / "data_access.json"
        fold_jobs = [
            fold_dir / f"seed_{seed}"
            for seed in seeds
        ]
        completed_jobs = [
            _completed_job(
                path,
                spec["pretraining_spec_sha256"],
                blueprint["architecture"],
            )
            for path in fold_jobs
        ]
        if all(job is not None for job in completed_jobs):
            if not fold_data_access_path.is_file():
                raise RuntimeError(
                    f"Completed V43 fold lacks data-access audit: {fold}"
                )
            prior_access = _load_json(fold_data_access_path)
            _validate_persisted_fold_access(
                prior_access, fold_entry, pretraining["data_access"]
            )
            _validate_persisted_scaler(
                fold_dir / "scaler.json",
                fold_entry,
                pretraining["data_access"],
            )
            data_audits.append(prior_access["audit"])
            read_receipts.extend(prior_access["receipts"])
            jobs.extend(job for job in completed_jobs if job is not None)
            continue
        data = read_fold_feature_data(
            paths["panel"],
            paths["sequence_index"],
            fold_entry,
            pretraining["data_access"],
        )
        data_audits.append(data.audit)
        fold_receipts = [
            {"fold": fold, **receipt} for receipt in data.receipts
        ]
        read_receipts.extend(fold_receipts)
        _write_json(
            fold_data_access_path,
            {"audit": data.audit, "receipts": fold_receipts},
        )
        fold_data_access_sha256 = _sha256_file(fold_data_access_path)
        scaler = FeatureScaler.fit_from_panel(
            data.panel,
            feature_names,
            pretraining["data_access"]["representation_train_start"],
            pretraining["data_access"]["representation_train_end"],
            pretraining["data_access"]["representation_train_end"],
            "log_close_to_close_return",
        )
        expected_rows = int(
            pretraining["data_access"]["expected_by_fold"][str(fold)][
                "scaler_finite_train_rows"
            ]
        )
        if scaler.fit_rows != expected_rows:
            raise RuntimeError("V43 scaler row count drift")
        scaler_path = fold_dir / "scaler.json"
        _write_json(
            scaler_path,
            {
                **asdict(scaler),
                "fold": fold,
                "train_symbols": sorted(fold_entry["train_symbols"]),
                "scaler_state_sha256": scaler.state_sha256(),
            },
        )
        scaler_artifact_sha256 = _sha256_file(scaler_path)
        _validate_persisted_scaler(
            scaler_path, fold_entry, pretraining["data_access"]
        )
        store = TripletTensorStore(
            data.panel,
            feature_names,
            int(blueprint["architecture"]["lookback_days"]),
            "log_close_to_close_return",
        )
        for seed, prior_complete in zip(seeds, completed_jobs, strict=True):
            if prior_complete is not None:
                jobs.append(prior_complete)
                continue
            jobs.append(
                _train_job(
                    fold_entry=fold_entry,
                    seed=seed,
                    architecture=blueprint["architecture"],
                    feature_names=feature_names,
                    store=store,
                    scaler=scaler,
                    train_availability=data.train_availability,
                    validation_availability=data.validation_availability,
                    blueprint=blueprint,
                    pretraining=pretraining,
                    effective=effective,
                    pretraining_spec=spec,
                    artifact_hashes=artifact_hashes,
                    fold_data_access_path=fold_data_access_path,
                    fold_data_access_sha256=fold_data_access_sha256,
                    scaler_path=scaler_path,
                    scaler_artifact_sha256=scaler_artifact_sha256,
                    checkpoint_root=checkpoint_root,
                    device=device,
                    expected_initialization_sha256=seed_hashes[seed],
                )
            )
        del store, data
        gc.collect()
        torch.mps.empty_cache()

    expected_jobs = len(folds) * len(seeds)
    combinations = {(int(job["fold"]), int(job["seed"])) for job in jobs}
    scaler_hashes = {
        fold: {
            job["scaler_state_sha256"]
            for job in jobs
            if int(job["fold"]) == fold
        }
        for fold in folds
    }
    execution_checks = {
        **contract_checks,
        "checkpoint_count_and_grid_are_exact": len(jobs) == expected_jobs
        and len(combinations) == expected_jobs,
        "all_checkpoint_files_and_states_match": all(
            Path(job["checkpoint_path"]).is_file()
            and _sha256_file(Path(job["checkpoint_path"]))
            == job["checkpoint_sha256"]
            and bool(job["model_state_sha256"])
            for job in jobs
        ),
        "one_scaler_per_fold_is_shared_across_seeds": all(
            len(values) == 1 for values in scaler_hashes.values()
        ),
        "all_initializations_are_fresh_and_registered": all(
            job["initialization_source"] == "fresh_registered_seed"
            and job["parent_checkpoint_sha256"] is None
            and job["initialization_state_sha256"]
            == seed_hashes[int(job["seed"])]
            for job in jobs
        ),
        "forbidden_modules_never_change": all(
            job["forbidden_initial_state_sha256"]
            == job["forbidden_final_state_sha256"]
            for job in jobs
        ),
        "no_seed_or_fold_selection": all(
            not job["seed_selected"] and not job["fold_selected"]
            for job in jobs
        ),
        "no_labels_heldout_targets_predictions_or_pnl": all(
            not job["labels_loaded"]
            and not job["heldout_assets_loaded"]
            and not job["target_assets_loaded"]
            and not job["supervised_heads_used"]
            and not job["performance_metrics_computed"]
            and not job["pnl_computed"]
            for job in jobs
        ),
        "full_run_has_all_nine_jobs": mode == "smoke" or len(jobs) == 9,
        "all_materialized_fold_data_passed_audit": all(
            not audit["heldout_symbols_materialized"]
            and not audit["target_symbols_materialized"]
            and audit["label_column_read_count"] == 0
            and audit["post_2024_market_row_count"] == 0
            for audit in data_audits
        ),
    }
    execution_checks = {
        key: bool(value) for key, value in execution_checks.items()
    }
    if not all(execution_checks.values()):
        raise RuntimeError(f"V43 execution audit failed: {execution_checks}")

    manifest = [
        {
            key: job[key]
            for key in (
                "fold",
                "seed",
                "train_symbols",
                "test_symbols",
                "initialization_state_sha256",
                "scaler_state_sha256",
                "checkpoint_path",
                "checkpoint_sha256",
                "model_state_sha256",
                "best_epoch",
                "best_validation_loss",
                "completed_epochs",
                "train_optimizer_steps",
            )
        }
        for job in jobs
    ]
    decision = (
        "authorize_v43_full_nine_job_pretraining_only"
        if mode == "smoke"
        else pretraining["authorized_next_action"]
    )
    result = {
        "version": "v43_smoke" if mode == "smoke" else "v43",
        "decision": decision,
        "pretraining_spec": spec,
        "summary": {
            "checkpoint_count": len(jobs),
            "total_optimizer_steps": int(
                sum(int(job["train_optimizer_steps"]) for job in jobs)
            ),
            "total_completed_epochs": int(
                sum(int(job["completed_epochs"]) for job in jobs)
            ),
            "total_parameters": total_parameters,
            "pretraining_parameters": trainable_parameters,
            "registered_initialization_state_sha256": {
                str(seed): value for seed, value in seed_hashes.items()
            },
            "mps_available": mps_available,
            "folds_materialized_this_invocation": [
                int(audit["fold"]) for audit in data_audits
            ],
        },
        "checkpoint_manifest": manifest,
        "data_access_audit": {
            "folds": data_audits,
            "read_receipts": read_receipts,
            "physical_row_group_isolation_claimed": False,
        },
        "tested": {
            "real_non_target_features_loaded": True,
            "masked_pretraining_executed": True,
            "feature_only_validation_executed": True,
            "labels_loaded": False,
            "heldout_assets_loaded": False,
            "target_assets_loaded": False,
            "supervised_predictions_computed": False,
            "performance_metrics_computed": False,
            "pnl_computed": False,
            "seed_or_fold_selection_executed": False,
        },
        "audit": {"passed": True, "checks": execution_checks},
    }
    output_relative = (
        pretraining["smoke_output_dir"] if mode == "smoke" else config["output_dir"]
    )
    _write_result(root, output_relative, config, result)
    return result
