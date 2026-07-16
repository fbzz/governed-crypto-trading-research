"""Deterministic V58 optimization and checkpoint mechanics.

This module deliberately has no data-loader dependency.  Callers supply
already-authorized, deterministic batches and their registered sampler
receipts; the engine owns only model optimization, exact loss accounting,
epoch-boundary checkpointing, and same-job resume validation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import math
import os
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias

import torch
from torch import nn
from torch.nn import functional as F

from .state_conditioned_multi_horizon_model import (
    PAIR_INDEXES,
    QUANTILES,
    StateConditionedMultiHorizonTransformer,
    multi_horizon_quantile_loss,
)


FINAL_FORMAT_VERSION = "v58_state_conditioned_multi_horizon_checkpoint_v1"
RESUME_FORMAT_VERSION = "v58_state_conditioned_multi_horizon_resume_v1"
PINBALL_WEIGHT = 1.0
RANKING_WEIGHT = 0.5
CROSSING_WEIGHT = 0.1
RANKNET_TIE_TOLERANCE = 1.0e-12
EXPECTED_PARAMETER_COUNT = 465_513


@dataclass(frozen=True)
class V58Batch:
    """One already-authorized model batch."""

    features: torch.Tensor
    targets: torch.Tensor
    target_mask: torch.Tensor | None = None


BatchLike: TypeAlias = (
    V58Batch
    | tuple[torch.Tensor, torch.Tensor]
    | tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]
    | Mapping[str, Any]
)


@dataclass(frozen=True)
class V58BatchStream:
    """An epoch stream paired with its registered ordered-draw receipt."""

    batches: Iterable[BatchLike]
    sampler_receipt: str


class V58BatchProvider(Protocol):
    def __call__(
        self, role: Literal["train", "validation"], epoch: int
    ) -> V58BatchStream | tuple[Iterable[BatchLike], str] | Mapping[str, Any]: ...


@dataclass(frozen=True)
class V58CheckpointContext:
    """Hash-bound identity that must match exactly on every checkpoint load."""

    scaler_sha256: str
    data_access_sha256: str
    phase_contract_sha256: str
    source_bundle_sha256: str
    job_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in (
            "scaler_sha256",
            "data_access_sha256",
            "phase_contract_sha256",
            "source_bundle_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if not isinstance(self.job_metadata, Mapping) or not self.job_metadata:
            raise ValueError("V58 job_metadata must be a non-empty mapping")

    def payload_fields(self) -> dict[str, Any]:
        return {
            "scaler_sha256": self.scaler_sha256,
            "data_access_sha256": self.data_access_sha256,
            "phase_contract_sha256": self.phase_contract_sha256,
            "source_bundle_sha256": self.source_bundle_sha256,
            "job_metadata": deepcopy(dict(self.job_metadata)),
        }


@dataclass
class StrictEarlyStopping:
    """V58's strict, earliest-best early-stopping state machine."""

    patience: int
    minimum_delta: float = 0.0
    best_validation_total_loss: float = math.inf
    best_epoch: int = 0
    consecutive_non_improvements: int = 0
    should_stop: bool = False

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError("Early-stopping patience must be positive")
        if self.minimum_delta != 0.0:
            raise ValueError("V58 freezes early-stopping minimum_delta to 0.0")

    def update(self, epoch: int, validation_total_loss: float) -> bool:
        if epoch < 1 or not math.isfinite(validation_total_loss):
            raise ValueError("Epoch and validation total loss must be finite")
        improved = validation_total_loss < self.best_validation_total_loss
        if improved:
            self.best_validation_total_loss = float(validation_total_loss)
            self.best_epoch = int(epoch)
            self.consecutive_non_improvements = 0
            self.should_stop = False
        else:
            self.consecutive_non_improvements += 1
            self.should_stop = (
                self.consecutive_non_improvements >= self.patience
            )
        return improved


@dataclass
class GlobalLossAccumulator:
    """Accumulate raw component sums and exact denominators across batches."""

    pinball_sum: float = 0.0
    pinball_count: int = 0
    ranking_sum: float = 0.0
    ranking_pair_count: int = 0
    crossing_sum: float = 0.0
    crossing_count: int = 0
    batch_count: int = 0

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> None:
        sums = v58_loss_component_sums(
            predictions, targets, target_mask=target_mask
        )
        self.pinball_sum += float(sums["pinball_sum"])
        self.pinball_count += int(sums["pinball_count"])
        self.ranking_sum += float(sums["ranking_sum"])
        self.ranking_pair_count += int(sums["ranking_pair_count"])
        self.crossing_sum += float(sums["crossing_sum"])
        self.crossing_count += int(sums["crossing_count"])
        self.batch_count += 1

    def finalize(self) -> dict[str, float | int]:
        if self.batch_count < 1 or self.pinball_count < 1 or self.crossing_count < 1:
            raise RuntimeError("V58 loss aggregation received no active cells")
        pinball = self.pinball_sum / self.pinball_count
        ranking = (
            self.ranking_sum / self.ranking_pair_count
            if self.ranking_pair_count
            else 0.0
        )
        crossing = self.crossing_sum / self.crossing_count
        total = (
            PINBALL_WEIGHT * pinball
            + RANKING_WEIGHT * ranking
            + CROSSING_WEIGHT * crossing
        )
        values = (pinball, ranking, crossing, total)
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("V58 aggregate loss is non-finite")
        return {
            "pinball": pinball,
            "ranking": ranking,
            "crossing": crossing,
            "total": total,
            "pinball_count": self.pinball_count,
            "ranking_pair_count": self.ranking_pair_count,
            "crossing_count": self.crossing_count,
            "batch_count": self.batch_count,
        }


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _to_cpu(value: object) -> object:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return deepcopy(value)


def clone_model_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def semantic_state_sha256(value: object) -> str:
    """Hash nested tensor state by semantics rather than pickle bytes."""

    digest = hashlib.sha256()

    def emit(tag: str, payload: bytes = b"") -> None:
        encoded = tag.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
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
        elif isinstance(item, Mapping):
            emit("dict_start", str(len(item)).encode("ascii"))
            for key in sorted(
                item, key=lambda candidate: (type(candidate).__name__, repr(candidate))
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
                f"Unsupported V58 semantic-state type: {type(item).__name__}"
            )

    visit(value)
    return digest.hexdigest()


def _cpu_float64_sum(values: torch.Tensor) -> float:
    """Sum float32 device values in float64 after moving them to CPU.

    Apple MPS cannot materialize float64 tensors. Moving first preserves the
    frozen float64 diagnostic accumulation without changing model/loss dtype.
    """

    return float(values.detach().cpu().double().sum().item())


def v58_loss_component_sums(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    target_mask: torch.Tensor | None = None,
) -> dict[str, float | int]:
    """Return V56-equivalent raw loss sums and their exact global counts."""

    # Reuse the frozen V56 implementation as the shape/finite/semantic guard.
    multi_horizon_quantile_loss(
        predictions,
        targets,
        target_mask=target_mask,
        tie_tolerance=RANKNET_TIE_TOLERANCE,
        pinball_weight=PINBALL_WEIGHT,
        ranking_weight=RANKING_WEIGHT,
        crossing_weight=CROSSING_WEIGHT,
    )
    if target_mask is None:
        target_mask = torch.isfinite(targets)
    safe_targets = torch.where(target_mask, targets, torch.zeros_like(targets))
    error = safe_targets.unsqueeze(-1) - predictions
    quantiles = predictions.new_tensor(QUANTILES).view(1, 1, 1, 3)
    pinball_cells = torch.maximum(
        quantiles * error, (quantiles - 1.0) * error
    )
    pinball_mask = target_mask.unsqueeze(-1).expand_as(pinball_cells)
    pinball_values = pinball_cells[pinball_mask]

    h7_q50 = predictions[:, :, 2, 1]
    h7_targets = safe_targets[:, :, 2]
    h7_mask = target_mask[:, :, 2]
    ranking_values: list[torch.Tensor] = []
    for left, right in PAIR_INDEXES:
        difference = h7_targets[:, left] - h7_targets[:, right]
        active = h7_mask[:, left] & h7_mask[:, right] & (
            difference.abs() > RANKNET_TIE_TOLERANCE
        )
        if bool(active.any()):
            sign = difference[active].sign()
            prediction_difference = (h7_q50[:, left] - h7_q50[:, right])[active]
            ranking_values.append(F.softplus(-sign * prediction_difference))

    crossing_values = (
        F.relu(predictions[..., 0] - predictions[..., 1])
        + F.relu(predictions[..., 1] - predictions[..., 2])
    )
    ranking_count = sum(int(values.numel()) for values in ranking_values)
    return {
        "pinball_sum": _cpu_float64_sum(pinball_values),
        "pinball_count": int(pinball_values.numel()),
        "ranking_sum": float(
            sum(
                (
                    values.detach().cpu().double().sum()
                    for values in ranking_values
                ),
                start=torch.tensor(0.0, dtype=torch.float64),
            )
        ),
        "ranking_pair_count": ranking_count,
        "crossing_sum": _cpu_float64_sum(crossing_values),
        "crossing_count": int(crossing_values.numel()),
    }


def configure_v58_runtime(
    device: str | torch.device,
    *,
    seed: int,
    torch_threads: int = 1,
) -> torch.device:
    """Configure deterministic CPU tests or the frozen no-fallback MPS runtime."""

    resolved = torch.device(device)
    if resolved.type not in {"cpu", "mps"}:
        raise ValueError("V58 supports only deterministic CPU tests or MPS")
    if torch_threads != 1:
        raise ValueError("V58 freezes torch_threads to 1")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if resolved.type == "mps":
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
            raise RuntimeError("V58 MPS requires PYTORCH_ENABLE_MPS_FALLBACK=0")
        if not torch.backends.mps.is_available():
            raise RuntimeError("V58 MPS runtime is unavailable")
    seed_v58_torch(seed, resolved)
    return resolved


def seed_v58_torch(seed: int, device: str | torch.device) -> None:
    resolved = torch.device(device)
    torch.manual_seed(int(seed))
    if resolved.type == "mps":
        torch.mps.manual_seed(int(seed))


def capture_v58_rng_state(device: str | torch.device) -> dict[str, torch.Tensor | None]:
    resolved = torch.device(device)
    return {
        "cpu_rng_state": torch.get_rng_state().cpu().clone(),
        "mps_rng_state": (
            torch.mps.get_rng_state().cpu().clone()
            if resolved.type == "mps"
            else None
        ),
    }


def restore_v58_rng_state(
    *,
    cpu_rng_state: torch.Tensor,
    mps_rng_state: torch.Tensor | None,
    device: str | torch.device,
) -> None:
    resolved = torch.device(device)
    torch.set_rng_state(cpu_rng_state.detach().cpu())
    if resolved.type == "mps":
        if mps_rng_state is None:
            raise RuntimeError("V58 MPS resume is missing its RNG state")
        torch.mps.set_rng_state(mps_rng_state.detach().cpu())
    elif mps_rng_state is not None:
        raise RuntimeError("V58 CPU checkpoint unexpectedly contains MPS RNG state")


def instantiate_v58_model(
    architecture: dict[str, Any], device: str | torch.device
) -> StateConditionedMultiHorizonTransformer:
    model = StateConditionedMultiHorizonTransformer(architecture)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("V58 model parameter-count drift")
    return model.to(device=torch.device(device), dtype=torch.float32)


def build_v58_adamw(model: nn.Module) -> torch.optim.AdamW:
    """Build AdamW with every V58 option explicit and frozen."""

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("V58 model has no trainable parameters")
    return torch.optim.AdamW(
        parameters,
        lr=0.0003,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=0.0001,
        amsgrad=False,
        foreach=False,
        fused=False,
        capturable=False,
        maximize=False,
        differentiable=False,
    )


def clip_v58_gradients(model: nn.Module) -> float:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if any(
        parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
        for parameter in parameters
    ):
        raise RuntimeError("V58 produced missing or non-finite gradients")
    norm = nn.utils.clip_grad_norm_(
        parameters,
        max_norm=1.0,
        norm_type=2.0,
        error_if_nonfinite=True,
        foreach=False,
    )
    value = float(norm.detach().cpu())
    if not math.isfinite(value):
        raise RuntimeError("V58 produced a non-finite gradient norm")
    return value


def _trainable_named_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]


def optimizer_contract(
    optimizer: torch.optim.Optimizer, model: nn.Module
) -> dict[str, Any]:
    named = _trainable_named_parameters(model)
    state = optimizer.state_dict()
    parameter_ids = [
        parameter_id
        for group in state["param_groups"]
        for parameter_id in group["params"]
    ]
    if len(parameter_ids) != len(named) or len(set(parameter_ids)) != len(named):
        raise RuntimeError("V58 optimizer parameter cardinality drift")
    return {
        "optimizer_class": type(optimizer).__name__,
        "parameter_names": [name for name, _ in named],
        "parameter_shapes": [list(parameter.shape) for _, parameter in named],
        "parameter_dtypes": [str(parameter.dtype) for _, parameter in named],
        "param_groups": _to_cpu(state["param_groups"]),
    }


def model_state_sha256(state_or_model: Mapping[str, torch.Tensor] | nn.Module) -> str:
    state = (
        state_or_model.state_dict()
        if isinstance(state_or_model, nn.Module)
        else state_or_model
    )
    return semantic_state_sha256(state)


def optimizer_state_sha256(
    state_or_optimizer: Mapping[str, Any] | torch.optim.Optimizer,
) -> str:
    state = (
        state_or_optimizer.state_dict()
        if isinstance(state_or_optimizer, torch.optim.Optimizer)
        else state_or_optimizer
    )
    return semantic_state_sha256(_to_cpu(state))


def _state_is_finite(value: object) -> bool:
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_state_is_finite(nested) for nested in value.values())
    if isinstance(value, (list, tuple)):
        return all(_state_is_finite(nested) for nested in value)
    return True


def _validate_model_state(
    state: object, model: nn.Module, *, label: str
) -> dict[str, torch.Tensor]:
    if not isinstance(state, Mapping):
        raise RuntimeError(f"V58 {label} model state structure drift")
    expected = model.state_dict()
    if set(state) != set(expected):
        raise RuntimeError(f"V58 {label} model state key drift")
    result: dict[str, torch.Tensor] = {}
    for name, expected_tensor in expected.items():
        tensor = state[name]
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.shape != expected_tensor.shape
            or tensor.dtype != expected_tensor.dtype
            or (tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()))
        ):
            raise RuntimeError(f"V58 {label} model tensor drift: {name}")
        result[name] = tensor
    return result


def _validate_optimizer_state(
    state: object,
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    *,
    expected_step_count: int,
) -> dict[str, Any]:
    if not isinstance(state, dict) or set(state) != {"state", "param_groups"}:
        raise RuntimeError("V58 optimizer state structure drift")
    expected_groups = optimizer.state_dict()["param_groups"]
    if state["param_groups"] != expected_groups:
        raise RuntimeError("V58 optimizer hyperparameter/group drift")
    if not _state_is_finite(state):
        raise RuntimeError("V58 optimizer contains non-finite state")

    named = _trainable_named_parameters(model)
    parameter_ids = [
        parameter_id
        for group in expected_groups
        for parameter_id in group["params"]
    ]
    if len(parameter_ids) != len(named) or len(set(parameter_ids)) != len(named):
        raise RuntimeError("V58 optimizer parameter-group drift")
    expected_state_ids = set(parameter_ids) if expected_step_count > 0 else set()
    if not isinstance(state["state"], dict) or set(state["state"]) != expected_state_ids:
        raise RuntimeError("V58 optimizer parameter-state coverage drift")
    for parameter_id, (name, parameter) in zip(parameter_ids, named, strict=True):
        if expected_step_count == 0:
            continue
        parameter_state = state["state"][parameter_id]
        if not isinstance(parameter_state, dict):
            raise RuntimeError(f"V58 optimizer state drift: {name}")
        if set(parameter_state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise RuntimeError(f"V58 optimizer moment-key drift: {name}")
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = parameter_state[moment_name]
            if (
                not isinstance(moment, torch.Tensor)
                or moment.shape != parameter.shape
                or moment.dtype != parameter.dtype
            ):
                raise RuntimeError(
                    f"V58 optimizer moment shape/dtype drift: {name}/{moment_name}"
                )
        step = parameter_state["step"]
        if isinstance(step, torch.Tensor):
            if step.numel() != 1:
                raise RuntimeError(f"V58 optimizer step shape drift: {name}")
            step_value = int(step.detach().cpu().item())
        else:
            step_value = int(step)
        if step_value != expected_step_count:
            raise RuntimeError(f"V58 optimizer step-count drift: {name}")
    return state


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                state[name] = value.to(device)


def _validate_loss_record(value: object, *, label: str) -> dict[str, Any]:
    required = {
        "pinball",
        "ranking",
        "crossing",
        "total",
        "pinball_count",
        "ranking_pair_count",
        "crossing_count",
        "batch_count",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise RuntimeError(f"V58 {label} loss record drift")
    scalars = [float(value[name]) for name in ("pinball", "ranking", "crossing", "total")]
    if not all(math.isfinite(item) for item in scalars):
        raise RuntimeError(f"V58 {label} loss contains non-finite values")
    expected_total = scalars[0] + RANKING_WEIGHT * scalars[1] + CROSSING_WEIGHT * scalars[2]
    if not math.isclose(scalars[3], expected_total, rel_tol=1.0e-15, abs_tol=1.0e-15):
        raise RuntimeError(f"V58 {label} total-loss accounting drift")
    counts = [
        int(value["pinball_count"]),
        int(value["ranking_pair_count"]),
        int(value["crossing_count"]),
        int(value["batch_count"]),
    ]
    if counts[0] < 1 or counts[1] < 0 or counts[2] < 1 or counts[3] < 1:
        raise RuntimeError(f"V58 {label} loss-count drift")
    return value


def _validate_history_and_early_stopping(
    history: object,
    early_state: object,
    *,
    completed_epoch: int,
    maximum_epochs: int,
    patience: int,
    optimizer_step_count: int,
) -> tuple[list[dict[str, Any]], StrictEarlyStopping]:
    if (
        not isinstance(history, list)
        or completed_epoch < 1
        or completed_epoch > maximum_epochs
        or len(history) != completed_epoch
    ):
        raise RuntimeError("V58 epoch/history boundary drift")
    replay = StrictEarlyStopping(patience=patience)
    cumulative_steps = 0
    validation_receipt: str | None = None
    for expected_epoch, row in enumerate(history, start=1):
        if not isinstance(row, dict) or int(row.get("epoch", -1)) != expected_epoch:
            raise RuntimeError("V58 history epoch order drift")
        _validate_loss_record(row.get("train_losses"), label="train")
        validation = _validate_loss_record(
            row.get("validation_losses"), label="validation"
        )
        steps = int(row.get("train_optimizer_steps", -1))
        cumulative_steps += steps
        if steps < 1 or int(row.get("optimizer_step_count", -1)) != cumulative_steps:
            raise RuntimeError("V58 history optimizer-step drift")
        train_receipt = row.get("train_sampler_receipt")
        current_validation_receipt = row.get("validation_sampler_receipt")
        _require_sha256(train_receipt, "train_sampler_receipt")
        _require_sha256(current_validation_receipt, "validation_sampler_receipt")
        if validation_receipt is None:
            validation_receipt = current_validation_receipt
        elif current_validation_receipt != validation_receipt:
            raise RuntimeError("V58 validation sampler receipt changed across epochs")
        improved = replay.update(expected_epoch, float(validation["total"]))
        if row.get("improved") is not improved:
            raise RuntimeError("V58 strict-improvement history drift")
        if not math.isfinite(float(row.get("maximum_gradient_norm", math.nan))):
            raise RuntimeError("V58 gradient-norm history drift")
    if cumulative_steps != optimizer_step_count:
        raise RuntimeError("V58 checkpoint optimizer-step total drift")
    if not isinstance(early_state, dict):
        raise RuntimeError("V58 early-stopping state structure drift")
    try:
        restored = StrictEarlyStopping(**early_state)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("V58 early-stopping state drift") from exc
    if asdict(restored) != asdict(replay):
        raise RuntimeError("V58 early-stopping/history coherence drift")
    return history, restored


def _atomic_torch_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary, _use_new_zipfile_serialization=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _checkpoint_format(kind: Literal["resume", "final"]) -> str:
    if kind == "resume":
        return RESUME_FORMAT_VERSION
    if kind == "final":
        return FINAL_FORMAT_VERSION
    raise ValueError("V58 checkpoint kind must be resume or final")


def save_v58_checkpoint(
    path: str | Path,
    *,
    kind: Literal["resume", "final"],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_model_state: Mapping[str, torch.Tensor],
    completed_epoch: int,
    optimizer_step_count: int,
    sampler_epoch: int,
    early_stopping: StrictEarlyStopping,
    full_history: list[dict[str, Any]],
    context: V58CheckpointContext,
    device: str | torch.device,
) -> dict[str, Any]:
    """Atomically persist a complete V58 epoch-boundary checkpoint."""

    if completed_epoch < 1 or sampler_epoch != completed_epoch:
        raise ValueError("V58 checkpoints are completed-epoch boundaries only")
    current_state = clone_model_state(model)
    best_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in best_model_state.items()
    }
    _validate_model_state(current_state, model, label="current")
    _validate_model_state(best_state, model, label="best")
    optimizer_state = _to_cpu(optimizer.state_dict())
    rng = capture_v58_rng_state(device)
    early_state = asdict(early_stopping)
    history = deepcopy(full_history)
    payload: dict[str, Any] = {
        "format_version": _checkpoint_format(kind),
        "checkpoint_kind": kind,
        "current_model_state": current_state,
        "current_model_state_sha256": model_state_sha256(current_state),
        "best_model_state": best_state,
        "best_model_state_sha256": model_state_sha256(best_state),
        "optimizer_state": optimizer_state,
        "optimizer_state_sha256": optimizer_state_sha256(optimizer_state),
        "optimizer_contract": optimizer_contract(optimizer, model),
        "completed_epoch": int(completed_epoch),
        "next_epoch": int(completed_epoch) + 1,
        "optimizer_step_count": int(optimizer_step_count),
        "cpu_rng_state": rng["cpu_rng_state"],
        "mps_rng_state": rng["mps_rng_state"],
        "rng_state_sha256": semantic_state_sha256(rng),
        "sampler_epoch": int(sampler_epoch),
        "early_stopping_state": early_state,
        "early_stopping_state_sha256": semantic_state_sha256(early_state),
        "full_history": history,
        "full_history_sha256": semantic_state_sha256(history),
        **context.payload_fields(),
    }
    payload["semantic_checkpoint_sha256"] = semantic_state_sha256(payload)
    _atomic_torch_save(payload, Path(path))
    return payload


def _validate_checkpoint_payload(
    payload: object,
    *,
    expected_kind: Literal["resume", "final"],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    context: V58CheckpointContext,
    maximum_epochs: int,
    patience: int,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError("V58 checkpoint payload structure drift")
    required = {
        "format_version",
        "checkpoint_kind",
        "current_model_state",
        "current_model_state_sha256",
        "best_model_state",
        "best_model_state_sha256",
        "optimizer_state",
        "optimizer_state_sha256",
        "optimizer_contract",
        "completed_epoch",
        "next_epoch",
        "optimizer_step_count",
        "cpu_rng_state",
        "mps_rng_state",
        "rng_state_sha256",
        "sampler_epoch",
        "early_stopping_state",
        "early_stopping_state_sha256",
        "full_history",
        "full_history_sha256",
        "scaler_sha256",
        "data_access_sha256",
        "phase_contract_sha256",
        "source_bundle_sha256",
        "job_metadata",
        "semantic_checkpoint_sha256",
    }
    if set(payload) != required:
        raise RuntimeError("V58 checkpoint required-state drift")
    if (
        payload["checkpoint_kind"] != expected_kind
        or payload["format_version"] != _checkpoint_format(expected_kind)
    ):
        raise RuntimeError("V58 checkpoint kind/format drift")
    expected_context = context.payload_fields()
    if any(payload.get(name) != value for name, value in expected_context.items()):
        raise RuntimeError("V58 same-job metadata or contract-hash drift")
    for name in (
        "scaler_sha256",
        "data_access_sha256",
        "phase_contract_sha256",
        "source_bundle_sha256",
    ):
        _require_sha256(payload[name], name)

    completed_epoch = int(payload["completed_epoch"])
    if (
        int(payload["next_epoch"]) != completed_epoch + 1
        or int(payload["sampler_epoch"]) != completed_epoch
    ):
        raise RuntimeError("V58 checkpoint epoch-boundary drift")
    optimizer_steps = int(payload["optimizer_step_count"])
    history, early = _validate_history_and_early_stopping(
        payload["full_history"],
        payload["early_stopping_state"],
        completed_epoch=completed_epoch,
        maximum_epochs=maximum_epochs,
        patience=patience,
        optimizer_step_count=optimizer_steps,
    )
    if semantic_state_sha256(history) != payload["full_history_sha256"]:
        raise RuntimeError("V58 checkpoint history semantic hash drift")
    if semantic_state_sha256(asdict(early)) != payload["early_stopping_state_sha256"]:
        raise RuntimeError("V58 checkpoint early-stopping semantic hash drift")

    current_state = _validate_model_state(
        payload["current_model_state"], model, label="current"
    )
    best_state = _validate_model_state(
        payload["best_model_state"], model, label="best"
    )
    if model_state_sha256(current_state) != payload["current_model_state_sha256"]:
        raise RuntimeError("V58 current model semantic hash drift")
    if model_state_sha256(best_state) != payload["best_model_state_sha256"]:
        raise RuntimeError("V58 best model semantic hash drift")
    optimizer_state = _validate_optimizer_state(
        payload["optimizer_state"],
        optimizer,
        model,
        expected_step_count=optimizer_steps,
    )
    if optimizer_state_sha256(optimizer_state) != payload["optimizer_state_sha256"]:
        raise RuntimeError("V58 optimizer semantic hash drift")
    if payload["optimizer_contract"] != optimizer_contract(optimizer, model):
        raise RuntimeError("V58 optimizer parameter contract drift")

    rng = {
        "cpu_rng_state": payload["cpu_rng_state"],
        "mps_rng_state": payload["mps_rng_state"],
    }
    if (
        not isinstance(rng["cpu_rng_state"], torch.Tensor)
        or rng["cpu_rng_state"].dtype != torch.uint8
        or not _state_is_finite(rng)
        or semantic_state_sha256(rng) != payload["rng_state_sha256"]
    ):
        raise RuntimeError("V58 RNG state/hash drift")
    without_checkpoint_hash = dict(payload)
    registered_hash = without_checkpoint_hash.pop("semantic_checkpoint_sha256")
    if semantic_state_sha256(without_checkpoint_hash) != registered_hash:
        raise RuntimeError("V58 checkpoint semantic hash drift")
    return payload


def load_v58_checkpoint(
    path: str | Path,
    *,
    expected_kind: Literal["resume", "final"],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    context: V58CheckpointContext,
    device: str | torch.device,
    maximum_epochs: int,
    patience: int,
    restore_runtime_state: bool = True,
) -> dict[str, Any]:
    """Validate a hash-bound same-job checkpoint before optionally restoring it."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    payload = _validate_checkpoint_payload(
        payload,
        expected_kind=expected_kind,
        model=model,
        optimizer=optimizer,
        context=context,
        maximum_epochs=maximum_epochs,
        patience=patience,
    )
    if restore_runtime_state:
        model.load_state_dict(payload["current_model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        _move_optimizer_state(optimizer, torch.device(device))
        restore_v58_rng_state(
            cpu_rng_state=payload["cpu_rng_state"],
            mps_rng_state=payload["mps_rng_state"],
            device=device,
        )
    return payload


def _normalize_batch(value: BatchLike) -> V58Batch:
    if isinstance(value, V58Batch):
        batch = value
    elif isinstance(value, Mapping):
        unknown = set(value) - {"features", "targets", "target_mask"}
        if unknown or not {"features", "targets"}.issubset(value):
            raise ValueError("V58 batch mapping contract drift")
        batch = V58Batch(
            features=value["features"],
            targets=value["targets"],
            target_mask=value.get("target_mask"),
        )
    elif isinstance(value, tuple) and len(value) in {2, 3}:
        batch = V58Batch(
            features=value[0],
            targets=value[1],
            target_mask=value[2] if len(value) == 3 else None,
        )
    else:
        raise TypeError("V58 batch must be V58Batch, mapping, or 2/3-tuple")
    if not isinstance(batch.features, torch.Tensor) or not isinstance(
        batch.targets, torch.Tensor
    ):
        raise TypeError("V58 batch features and targets must be torch tensors")
    if batch.features.dtype != torch.float32 or batch.targets.dtype != torch.float32:
        raise ValueError("V58 feature and target tensor dtypes must be float32")
    if batch.target_mask is not None and (
        not isinstance(batch.target_mask, torch.Tensor)
        or batch.target_mask.dtype != torch.bool
    ):
        raise ValueError("V58 target mask must be a boolean tensor")
    return batch


def _call_batch_provider(
    provider: V58BatchProvider | object,
    role: Literal["train", "validation"],
    epoch: int,
) -> V58BatchStream:
    if callable(provider):
        supplied = provider(role, epoch)
    elif hasattr(provider, "batches") and callable(getattr(provider, "batches")):
        supplied = provider.batches(role, epoch)  # type: ignore[attr-defined]
    else:
        raise TypeError("V58 batch provider must be callable or expose batches()")

    if isinstance(supplied, V58BatchStream):
        stream = supplied
    elif (
        isinstance(supplied, tuple)
        and len(supplied) == 2
        and isinstance(supplied[1], str)
    ):
        stream = V58BatchStream(supplied[0], supplied[1])
    elif isinstance(supplied, Mapping) and set(supplied) == {
        "batches",
        "sampler_receipt",
    }:
        stream = V58BatchStream(
            supplied["batches"], supplied["sampler_receipt"]
        )
    else:
        receipt_method = getattr(provider, "sampler_receipt", None)
        if not callable(receipt_method):
            raise TypeError(
                "V58 provider output must include batches and sampler_receipt"
            )
        stream = V58BatchStream(supplied, receipt_method(role, epoch))
    _require_sha256(stream.sampler_receipt, f"{role}_sampler_receipt")
    return stream


def _run_epoch(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: Iterable[BatchLike],
    device: torch.device,
    training: bool,
) -> tuple[dict[str, float | int], int, float]:
    accumulator = GlobalLossAccumulator()
    optimizer_steps = 0
    maximum_gradient_norm = 0.0
    model.train(training)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for raw_batch in batches:
            batch = _normalize_batch(raw_batch)
            features = batch.features.to(device=device, dtype=torch.float32)
            targets = batch.targets.to(device=device, dtype=torch.float32)
            target_mask = (
                batch.target_mask.to(device=device)
                if batch.target_mask is not None
                else None
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            predictions = model(features)
            if not isinstance(predictions, torch.Tensor):
                raise RuntimeError("V58 model must return one prediction tensor")
            losses = multi_horizon_quantile_loss(
                predictions,
                targets,
                target_mask=target_mask,
                tie_tolerance=RANKNET_TIE_TOLERANCE,
                pinball_weight=PINBALL_WEIGHT,
                ranking_weight=RANKING_WEIGHT,
                crossing_weight=CROSSING_WEIGHT,
            )
            accumulator.update(
                predictions.detach(), targets.detach(), target_mask=target_mask
            )
            if training:
                losses["total"].backward()
                gradient_norm = clip_v58_gradients(model)
                optimizer.step()
                optimizer_steps += 1
                maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
    if training and optimizer_steps < 1:
        raise RuntimeError("V58 training epoch received no batches")
    return accumulator.finalize(), optimizer_steps, maximum_gradient_norm


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _result_from_checkpoint(
    payload: dict[str, Any],
    *,
    status: str,
    new_optimizer_steps: int,
    checkpoint_path: Path,
    resume_path: Path,
) -> dict[str, Any]:
    completed = payload["checkpoint_kind"] == "final"
    early = payload["early_stopping_state"]
    history = deepcopy(payload["full_history"])
    return {
        "status": status,
        "completed": completed,
        "job_metadata": deepcopy(payload["job_metadata"]),
        "completed_epoch": int(payload["completed_epoch"]),
        "next_epoch": int(payload["next_epoch"]),
        "optimizer_step_count": int(payload["optimizer_step_count"]),
        "new_optimizer_steps": int(new_optimizer_steps),
        "best_epoch": int(early["best_epoch"]),
        "best_validation_total_loss": float(
            early["best_validation_total_loss"]
        ),
        "early_stopping_state": deepcopy(early),
        "history": history,
        "sampler_receipts": [
            {
                "epoch": int(row["epoch"]),
                "train": row["train_sampler_receipt"],
                "validation": row["validation_sampler_receipt"],
            }
            for row in history
        ],
        "current_model_state_sha256": payload["current_model_state_sha256"],
        "best_model_state_sha256": payload["best_model_state_sha256"],
        "optimizer_state_sha256": payload["optimizer_state_sha256"],
        "rng_state_sha256": payload["rng_state_sha256"],
        "early_stopping_state_sha256": payload[
            "early_stopping_state_sha256"
        ],
        "full_history_sha256": payload["full_history_sha256"],
        "semantic_checkpoint_sha256": payload["semantic_checkpoint_sha256"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_file_sha256": _file_sha256(checkpoint_path),
        "resume_path": str(resume_path) if resume_path.is_file() else None,
        "scaler_sha256": payload["scaler_sha256"],
        "data_access_sha256": payload["data_access_sha256"],
        "phase_contract_sha256": payload["phase_contract_sha256"],
        "source_bundle_sha256": payload["source_bundle_sha256"],
    }


def _finalize_epoch_boundary(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_model_state: Mapping[str, torch.Tensor],
    completed_epoch: int,
    optimizer_step_count: int,
    early_stopping: StrictEarlyStopping,
    history: list[dict[str, Any]],
    context: V58CheckpointContext,
    device: torch.device,
    final_path: Path,
    resume_path: Path,
    status: str,
    new_optimizer_steps: int,
) -> dict[str, Any]:
    payload = save_v58_checkpoint(
        final_path,
        kind="final",
        model=model,
        optimizer=optimizer,
        best_model_state=best_model_state,
        completed_epoch=completed_epoch,
        optimizer_step_count=optimizer_step_count,
        sampler_epoch=completed_epoch,
        early_stopping=early_stopping,
        full_history=history,
        context=context,
        device=device,
    )
    resume_path.unlink(missing_ok=True)
    model.load_state_dict(best_model_state, strict=True)
    return _result_from_checkpoint(
        payload,
        status=status,
        new_optimizer_steps=new_optimizer_steps,
        checkpoint_path=final_path,
        resume_path=resume_path,
    )


def run_v58_training_job(
    *,
    model_factory: Callable[[], nn.Module],
    batch_provider: V58BatchProvider | object,
    job_seed: int,
    context: V58CheckpointContext,
    resume_path: str | Path,
    final_path: str | Path,
    device: str | torch.device = "mps",
    maximum_epochs: int = 30,
    patience: int = 5,
    interrupt_after_completed_epoch: int | None = None,
) -> dict[str, Any]:
    """Train or resume exactly one job from an atomic completed-epoch boundary."""

    if maximum_epochs < 1 or patience < 1:
        raise ValueError("V58 maximum_epochs and patience must be positive")
    if interrupt_after_completed_epoch is not None and not (
        1 <= interrupt_after_completed_epoch <= maximum_epochs
    ):
        raise ValueError("V58 interruption epoch is outside the training window")
    resume = Path(resume_path)
    final = Path(final_path)
    if resume.resolve() == final.resolve():
        raise ValueError("V58 resume and final checkpoint paths must differ")
    if resume.exists() and final.exists():
        raise RuntimeError("V58 completed job has an orphan resume checkpoint")

    resolved_device = configure_v58_runtime(device, seed=job_seed)
    model = model_factory().to(device=resolved_device, dtype=torch.float32)
    optimizer = build_v58_adamw(model)

    if final.is_file():
        payload = load_v58_checkpoint(
            final,
            expected_kind="final",
            model=model,
            optimizer=optimizer,
            context=context,
            device=resolved_device,
            maximum_epochs=maximum_epochs,
            patience=patience,
            restore_runtime_state=False,
        )
        return _result_from_checkpoint(
            payload,
            status="already_complete",
            new_optimizer_steps=0,
            checkpoint_path=final,
            resume_path=resume,
        )

    if resume.is_file():
        payload = load_v58_checkpoint(
            resume,
            expected_kind="resume",
            model=model,
            optimizer=optimizer,
            context=context,
            device=resolved_device,
            maximum_epochs=maximum_epochs,
            patience=patience,
        )
        completed_epoch = int(payload["completed_epoch"])
        optimizer_step_count = int(payload["optimizer_step_count"])
        history = deepcopy(payload["full_history"])
        early_stopping = StrictEarlyStopping(**payload["early_stopping_state"])
        best_model_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in payload["best_model_state"].items()
        }
        resumed = True
    else:
        completed_epoch = 0
        optimizer_step_count = 0
        history: list[dict[str, Any]] = []
        early_stopping = StrictEarlyStopping(patience=patience)
        best_model_state: dict[str, torch.Tensor] = {}
        resumed = False

    new_optimizer_steps = 0
    if completed_epoch >= maximum_epochs or early_stopping.should_stop:
        if not best_model_state:
            raise RuntimeError("V58 terminal resume is missing its best state")
        return _finalize_epoch_boundary(
            model=model,
            optimizer=optimizer,
            best_model_state=best_model_state,
            completed_epoch=completed_epoch,
            optimizer_step_count=optimizer_step_count,
            early_stopping=early_stopping,
            history=history,
            context=context,
            device=resolved_device,
            final_path=final,
            resume_path=resume,
            status="completed_from_resume_boundary",
            new_optimizer_steps=0,
        )

    for epoch in range(completed_epoch + 1, maximum_epochs + 1):
        train_stream = _call_batch_provider(batch_provider, "train", epoch)
        train_losses, epoch_steps, maximum_gradient_norm = _run_epoch(
            model=model,
            optimizer=optimizer,
            batches=train_stream.batches,
            device=resolved_device,
            training=True,
        )
        optimizer_step_count += epoch_steps
        new_optimizer_steps += epoch_steps

        validation_stream = _call_batch_provider(batch_provider, "validation", 0)
        validation_losses, validation_steps, _ = _run_epoch(
            model=model,
            optimizer=optimizer,
            batches=validation_stream.batches,
            device=resolved_device,
            training=False,
        )
        if validation_steps != 0:
            raise RuntimeError("V58 validation performed optimizer steps")
        improved = early_stopping.update(epoch, float(validation_losses["total"]))
        if improved:
            best_model_state = clone_model_state(model)
        if not best_model_state:
            raise RuntimeError("V58 failed to register a strict best model state")
        history.append(
            {
                "epoch": epoch,
                "train_losses": train_losses,
                "validation_losses": validation_losses,
                "train_optimizer_steps": epoch_steps,
                "optimizer_step_count": optimizer_step_count,
                "train_sampler_receipt": train_stream.sampler_receipt,
                "validation_sampler_receipt": validation_stream.sampler_receipt,
                "maximum_gradient_norm": maximum_gradient_norm,
                "improved": improved,
            }
        )
        resume_payload = save_v58_checkpoint(
            resume,
            kind="resume",
            model=model,
            optimizer=optimizer,
            best_model_state=best_model_state,
            completed_epoch=epoch,
            optimizer_step_count=optimizer_step_count,
            sampler_epoch=epoch,
            early_stopping=early_stopping,
            full_history=history,
            context=context,
            device=resolved_device,
        )
        if interrupt_after_completed_epoch == epoch:
            return _result_from_checkpoint(
                resume_payload,
                status="interrupted",
                new_optimizer_steps=new_optimizer_steps,
                checkpoint_path=resume,
                resume_path=resume,
            )
        if early_stopping.should_stop or epoch == maximum_epochs:
            return _finalize_epoch_boundary(
                model=model,
                optimizer=optimizer,
                best_model_state=best_model_state,
                completed_epoch=epoch,
                optimizer_step_count=optimizer_step_count,
                early_stopping=early_stopping,
                history=history,
                context=context,
                device=resolved_device,
                final_path=final,
                resume_path=resume,
                status="completed_after_resume" if resumed else "completed",
                new_optimizer_steps=new_optimizer_steps,
            )
    raise AssertionError("V58 training loop exited without a terminal receipt")


def verify_v58_checkpoint_roundtrip(
    path: str | Path,
    *,
    model_factory: Callable[[], nn.Module],
    job_seed: int,
    context: V58CheckpointContext,
    checkpoint_kind: Literal["resume", "final"],
    device: str | torch.device = "mps",
    maximum_epochs: int = 30,
    patience: int = 5,
) -> dict[str, Any]:
    """Load model/optimizer/RNG state and verify semantic roundtrip hashes."""

    resolved_device = configure_v58_runtime(device, seed=job_seed)
    model = model_factory().to(device=resolved_device, dtype=torch.float32)
    optimizer = build_v58_adamw(model)
    payload = load_v58_checkpoint(
        path,
        expected_kind=checkpoint_kind,
        model=model,
        optimizer=optimizer,
        context=context,
        device=resolved_device,
        maximum_epochs=maximum_epochs,
        patience=patience,
    )
    current_hash = model_state_sha256(model)
    optimizer_hash = optimizer_state_sha256(optimizer)
    model.load_state_dict(payload["best_model_state"], strict=True)
    best_hash = model_state_sha256(model)
    rng_hash = semantic_state_sha256(
        {
            "cpu_rng_state": payload["cpu_rng_state"],
            "mps_rng_state": payload["mps_rng_state"],
        }
    )
    early_stopping_hash = semantic_state_sha256(
        payload["early_stopping_state"]
    )
    history_hash = semantic_state_sha256(payload["full_history"])
    states_use_distinct_storage = all(
        payload["current_model_state"][name].data_ptr()
        != payload["best_model_state"][name].data_ptr()
        for name in payload["current_model_state"]
    )
    checks = {
        "current_model_roundtrip": current_hash
        == payload["current_model_state_sha256"],
        "best_model_roundtrip": best_hash == payload["best_model_state_sha256"],
        "optimizer_roundtrip": optimizer_hash == payload["optimizer_state_sha256"],
        "current_and_best_state_are_distinct": states_use_distinct_storage,
        "same_job_metadata": payload["job_metadata"] == dict(context.job_metadata),
        "semantic_checkpoint_hash": semantic_state_sha256(
            {
                key: value
                for key, value in payload.items()
                if key != "semantic_checkpoint_sha256"
            }
        )
        == payload["semantic_checkpoint_sha256"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "checkpoint_path": str(Path(path)),
        "checkpoint_file_sha256": _file_sha256(Path(path)),
        "semantic_checkpoint_sha256": payload["semantic_checkpoint_sha256"],
        "current_model_state_sha256": current_hash,
        "best_model_state_sha256": best_hash,
        "optimizer_state_sha256": optimizer_hash,
        "rng_state_sha256": rng_hash,
        "early_stopping_state_sha256": early_stopping_hash,
        "full_history_sha256": history_hash,
        "scaler_sha256": payload["scaler_sha256"],
        "data_access_sha256": payload["data_access_sha256"],
        "phase_contract_sha256": payload["phase_contract_sha256"],
        "source_bundle_sha256": payload["source_bundle_sha256"],
        "job_metadata": deepcopy(payload["job_metadata"]),
        "completed_epoch": int(payload["completed_epoch"]),
        "optimizer_step_count": int(payload["optimizer_step_count"]),
    }


def prove_v58_interrupted_resume_equivalence(
    *,
    model_factory: Callable[[], nn.Module],
    batch_provider: V58BatchProvider | object,
    job_seed: int,
    context: V58CheckpointContext,
    work_dir: str | Path,
    device: str | torch.device = "cpu",
    maximum_epochs: int = 2,
    patience: int = 2,
    interrupt_after_completed_epoch: int = 1,
) -> dict[str, Any]:
    """Prove uninterrupted and epoch-1-interrupt/resume executions are identical."""

    directory = Path(work_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "uninterrupted_resume": directory / "uninterrupted.resume.pt",
        "uninterrupted_final": directory / "uninterrupted.final.pt",
        "resumed_resume": directory / "interrupted.resume.pt",
        "resumed_final": directory / "interrupted.final.pt",
    }
    existing = sorted(str(path) for path in paths.values() if path.exists())
    if existing:
        raise FileExistsError(f"V58 equivalence work directory is not fresh: {existing}")

    uninterrupted = run_v58_training_job(
        model_factory=model_factory,
        batch_provider=batch_provider,
        job_seed=job_seed,
        context=context,
        resume_path=paths["uninterrupted_resume"],
        final_path=paths["uninterrupted_final"],
        device=device,
        maximum_epochs=maximum_epochs,
        patience=patience,
    )
    interrupted = run_v58_training_job(
        model_factory=model_factory,
        batch_provider=batch_provider,
        job_seed=job_seed,
        context=context,
        resume_path=paths["resumed_resume"],
        final_path=paths["resumed_final"],
        device=device,
        maximum_epochs=maximum_epochs,
        patience=patience,
        interrupt_after_completed_epoch=interrupt_after_completed_epoch,
    )
    resumed = run_v58_training_job(
        model_factory=model_factory,
        batch_provider=batch_provider,
        job_seed=job_seed,
        context=context,
        resume_path=paths["resumed_resume"],
        final_path=paths["resumed_final"],
        device=device,
        maximum_epochs=maximum_epochs,
        patience=patience,
    )
    compared_fields = (
        "current_model_state_sha256",
        "best_model_state_sha256",
        "optimizer_state_sha256",
        "rng_state_sha256",
        "early_stopping_state_sha256",
        "full_history_sha256",
        "semantic_checkpoint_sha256",
        "optimizer_step_count",
        "completed_epoch",
        "history",
        "sampler_receipts",
    )
    comparisons = {
        name: uninterrupted[name] == resumed[name] for name in compared_fields
    }
    uninterrupted_roundtrip = verify_v58_checkpoint_roundtrip(
        paths["uninterrupted_final"],
        model_factory=model_factory,
        job_seed=job_seed,
        context=context,
        checkpoint_kind="final",
        device=device,
        maximum_epochs=maximum_epochs,
        patience=patience,
    )
    resumed_roundtrip = verify_v58_checkpoint_roundtrip(
        paths["resumed_final"],
        model_factory=model_factory,
        job_seed=job_seed,
        context=context,
        checkpoint_kind="final",
        device=device,
        maximum_epochs=maximum_epochs,
        patience=patience,
    )
    passed = (
        uninterrupted["completed"]
        and interrupted["status"] == "interrupted"
        and not interrupted["completed"]
        and resumed["completed"]
        and all(comparisons.values())
        and uninterrupted_roundtrip["passed"]
        and resumed_roundtrip["passed"]
        and not paths["uninterrupted_resume"].exists()
        and not paths["resumed_resume"].exists()
    )
    return {
        "passed": passed,
        "comparisons": comparisons,
        "uninterrupted": uninterrupted,
        "interrupted": interrupted,
        "resumed": resumed,
        "uninterrupted_roundtrip": uninterrupted_roundtrip,
        "resumed_roundtrip": resumed_roundtrip,
    }
