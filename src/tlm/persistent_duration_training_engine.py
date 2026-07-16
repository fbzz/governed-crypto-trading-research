"""Optimization, resume, and checkpoint mechanics for frozen V77 training."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .core.artifacts import canonical_sha256, file_sha256
from .persistent_duration_training_data import (
    V77FoldTrainingData,
    V77SampleDraw,
)
from .persistent_multi_horizon_duration_model import (
    PersistentMultiHorizonDurationTransformer,
    persistent_multi_task_loss,
)
from .state_conditioned_multi_horizon_training_engine import (
    capture_v58_rng_state,
    clone_model_state,
    restore_v58_rng_state,
    semantic_state_sha256,
)


FINAL_FORMAT = "v77_persistent_duration_checkpoint_v1"
RESUME_FORMAT = "v77_persistent_duration_resume_v1"


@dataclass
class V77EarlyStopping:
    patience: int
    minimum_delta: float
    best_validation_joint_objective: float = math.inf
    best_epoch: int = 0
    consecutive_non_improvements: int = 0
    should_stop: bool = False

    def __post_init__(self) -> None:
        if self.patience < 1 or self.minimum_delta < 0.0:
            raise ValueError("V77 early-stopping contract is invalid")

    def update(self, epoch: int, validation_joint_objective: float) -> bool:
        if epoch < 1 or not math.isfinite(validation_joint_objective):
            raise ValueError("V77 early-stopping observation is invalid")
        improved = (
            validation_joint_objective
            < self.best_validation_joint_objective - self.minimum_delta
        )
        if improved:
            self.best_validation_joint_objective = float(
                validation_joint_objective
            )
            self.best_epoch = int(epoch)
            self.consecutive_non_improvements = 0
            self.should_stop = False
        else:
            self.consecutive_non_improvements += 1
            self.should_stop = self.consecutive_non_improvements >= self.patience
        return improved


def configure_v77_runtime(device: str, *, seed: int) -> torch.device:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }:
        raise RuntimeError("V77 forbids PYTORCH_ENABLE_MPS_FALLBACK")
    torch.set_num_threads(10)
    torch.use_deterministic_algorithms(True)
    resolved = torch.device(device)
    if resolved.type == "mps":
        if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
            raise RuntimeError("V77 requires operational Apple MPS")
        probe = torch.ones(4, device=resolved, dtype=torch.float32)
        if float((probe * 2.0).sum().cpu()) != 8.0:
            raise RuntimeError("V77 MPS probe failed")
    elif resolved.type != "cpu":
        raise RuntimeError("V77 supports only MPS training and CPU verification")
    torch.manual_seed(int(seed))
    if resolved.type == "mps":
        torch.mps.manual_seed(int(seed))
    return resolved


def instantiate_v77_model(
    blueprint: dict[str, Any], device: torch.device, *, seed: int
) -> PersistentMultiHorizonDurationTransformer:
    torch.manual_seed(int(seed))
    model = PersistentMultiHorizonDurationTransformer(
        blueprint["architecture"]
    ).to(device=device, dtype=torch.float32)
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != 1_083_155 or count != int(blueprint["parameter_count"]):
        raise RuntimeError(f"V77 model parameter-count drift: {count}")
    return model


def _optimizer(
    model: nn.Module, contract: dict[str, Any]
) -> torch.optim.AdamW:
    frozen = contract["grid_optimizer_and_runtime_contract"]["optimizer"]
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(frozen["learning_rate"]),
        betas=tuple(float(value) for value in frozen["betas"]),
        eps=float(frozen["epsilon"]),
        weight_decay=float(frozen["weight_decay"]),
        amsgrad=False,
        foreach=False,
        fused=False,
        capturable=False,
        maximize=False,
        differentiable=False,
    )


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return deepcopy(value)


def _move_optimizer(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary, _use_new_zipfile_serialization=False)
    temporary.replace(path)


def _draw_receipt(draws: list[V77SampleDraw]) -> str:
    return canonical_sha256(
        [
            {
                "date": draw.date.isoformat(),
                "triplet": list(draw.triplet),
                "pair_index": draw.pair_index,
            }
            for draw in draws
        ]
    )


def _epoch(
    *,
    model: PersistentMultiHorizonDurationTransformer,
    optimizer: torch.optim.Optimizer | None,
    data: V77FoldTrainingData,
    draws: list[V77SampleDraw],
    batch_size: int,
    device: torch.device,
    gradient_clip: float,
    objective: dict[str, Any],
) -> tuple[dict[str, float], int, float, int]:
    training = optimizer is not None
    model.train(training)
    sums = {
        "return_nll": 0.0,
        "ranking": 0.0,
        "duration_nll": 0.0,
        "joint_objective": 0.0,
    }
    observations = 0
    pair_count = 0
    steps = 0
    maximum_gradient = 0.0
    weights = objective["objective_weights"]
    for start in range(0, len(draws), int(batch_size)):
        features, returns, durations, censored = data.store.materialize(
            draws[start : start + int(batch_size)], data.scale.feature_scaler
        )
        x = torch.from_numpy(features).to(device=device, dtype=torch.float32)
        y = torch.from_numpy(returns).to(device=device, dtype=torch.float32)
        d = torch.from_numpy(durations).to(device=device, dtype=torch.long)
        c = torch.from_numpy(censored).to(device=device, dtype=torch.bool)
        with torch.set_grad_enabled(training):
            output = model(x, round_trip_cost=0.0)
            losses = persistent_multi_task_loss(
                output,
                y,
                d,
                c,
                degrees_of_freedom=float(
                    objective["student_t_degrees_of_freedom"]
                ),
                return_nll_weight=float(weights["return_nll"]),
                ranking_weight=float(weights["pairwise_ranking"]),
                duration_weight=float(weights["duration_nll"]),
            )
        if not bool(torch.isfinite(losses["total"])):
            raise RuntimeError("V77 joint objective is non-finite")
        if training:
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            parameters = [
                parameter for parameter in model.parameters() if parameter.requires_grad
            ]
            if any(
                parameter.grad is None
                or not bool(torch.isfinite(parameter.grad).all())
                for parameter in parameters
            ):
                raise RuntimeError("V77 produced missing or non-finite gradients")
            norm = nn.utils.clip_grad_norm_(
                parameters,
                float(gradient_clip),
                error_if_nonfinite=True,
                foreach=False,
            )
            maximum_gradient = max(
                maximum_gradient, float(norm.detach().cpu())
            )
            optimizer.step()
            steps += 1
        count = len(x)
        sums["return_nll"] += float(losses["return_nll"].detach().cpu()) * count
        sums["ranking"] += float(losses["ranking"].detach().cpu()) * count
        sums["duration_nll"] += (
            float(losses["duration_nll"].detach().cpu()) * count
        )
        sums["joint_objective"] += float(losses["total"].detach().cpu()) * count
        observations += count
        pair_count += int(losses["pair_count"].detach().cpu())
    if observations != len(draws):
        raise RuntimeError("V77 epoch observation accounting drift")
    return (
        {key: value / observations for key, value in sums.items()},
        steps,
        maximum_gradient,
        pair_count,
    )


def _payload(
    *,
    kind: str,
    model: PersistentMultiHorizonDurationTransformer,
    optimizer: torch.optim.Optimizer,
    best_model_state: dict[str, torch.Tensor],
    early: V77EarlyStopping,
    completed_epoch: int,
    optimizer_steps: int,
    history: list[dict[str, Any]],
    context: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    rng = capture_v58_rng_state(device)
    value: dict[str, Any] = {
        "format_version": FINAL_FORMAT if kind == "final" else RESUME_FORMAT,
        "kind": kind,
        "stage": "complete" if kind == "final" else "training",
        "completed_epoch": int(completed_epoch),
        "model_current_state": clone_model_state(model),
        "model_best_state": _to_cpu(best_model_state),
        "optimizer_state": _to_cpu(optimizer.state_dict()),
        "early_stopping": asdict(early),
        "optimizer_steps": int(optimizer_steps),
        "cpu_rng_state": rng["cpu_rng_state"],
        "mps_rng_state": rng["mps_rng_state"],
        "history": deepcopy(history),
        "phase_contract_sha256": context["phase_contract_sha256"],
        "source_bundle_sha256": context["source_bundle_sha256"],
        "fold_feature_scaler_sha256": context["fold_feature_scaler_sha256"],
        "data_access_sha256": context["data_access_sha256"],
        "job_context": deepcopy(context),
        "prior_checkpoint_reused": False,
    }
    value["semantic_checkpoint_sha256"] = semantic_state_sha256(value)
    return value


def _validate_payload(
    payload: dict[str, Any], *, kind: str, context: dict[str, Any]
) -> None:
    expected_format = FINAL_FORMAT if kind == "final" else RESUME_FORMAT
    if payload.get("format_version") != expected_format or payload.get("kind") != kind:
        raise RuntimeError("V77 checkpoint format/kind drift")
    registered = payload.get("semantic_checkpoint_sha256")
    body = {
        key: value
        for key, value in payload.items()
        if key != "semantic_checkpoint_sha256"
    }
    if semantic_state_sha256(body) != registered:
        raise RuntimeError("V77 semantic checkpoint hash drift")
    if payload.get("job_context") != context:
        raise RuntimeError("V77 checkpoint job-context drift")
    if payload.get("prior_checkpoint_reused") is not False:
        raise RuntimeError("V77 checkpoint claims forbidden prior-state reuse")


def _result(
    path: Path, payload: dict[str, Any], *, status: str, new_steps: int
) -> dict[str, Any]:
    return {
        "job_id": payload["job_context"]["job_id"],
        "fold": payload["job_context"]["fold"],
        "seed": payload["job_context"]["seed"],
        "status": status,
        "completed": payload["stage"] == "complete",
        "completed_epoch": payload["completed_epoch"],
        "best_epoch": payload["early_stopping"]["best_epoch"],
        "checkpoint_path": str(path),
        "checkpoint_file_sha256": file_sha256(path),
        "semantic_checkpoint_sha256": payload["semantic_checkpoint_sha256"],
        "model_state_sha256": semantic_state_sha256(
            payload["model_current_state"]
        ),
        "optimizer_state_sha256": semantic_state_sha256(
            payload["optimizer_state"]
        ),
        "optimizer_steps": payload["optimizer_steps"],
        "new_optimizer_steps": int(new_steps),
        "history": deepcopy(payload["history"]),
    }


def run_v77_training_job(
    *,
    blueprint: dict[str, Any],
    contract: dict[str, Any],
    data: V77FoldTrainingData,
    seed: int,
    context: dict[str, Any],
    resume_path: Path,
    final_path: Path,
    device: str = "mps",
    train_samples: int = 8192,
    validation_samples: int = 2048,
    batch_size: int = 128,
    maximum_epochs: int = 40,
    patience: int = 6,
    minimum_delta: float = 1.0e-6,
    interrupt_after_epoch: int | None = None,
) -> dict[str, Any]:
    resolved = configure_v77_runtime(device, seed=int(seed))
    model = instantiate_v77_model(blueprint, resolved, seed=int(seed))
    optimizer = _optimizer(model, contract)
    if final_path.is_file():
        payload = torch.load(final_path, map_location="cpu", weights_only=False)
        _validate_payload(payload, kind="final", context=context)
        model.load_state_dict(payload["model_current_state"], strict=True)
        return _result(final_path, payload, status="already_complete", new_steps=0)

    if resume_path.is_file():
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        _validate_payload(payload, kind="resume", context=context)
        model.load_state_dict(payload["model_current_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        _move_optimizer(optimizer, resolved)
        early = V77EarlyStopping(**payload["early_stopping"])
        best_state = _to_cpu(payload["model_best_state"])
        history = deepcopy(payload["history"])
        steps = int(payload["optimizer_steps"])
        start_epoch = int(payload["completed_epoch"])
        restore_v58_rng_state(
            cpu_rng_state=payload["cpu_rng_state"],
            mps_rng_state=payload["mps_rng_state"],
            device=resolved,
        )
    else:
        early = V77EarlyStopping(
            patience=int(patience), minimum_delta=float(minimum_delta)
        )
        best_state: dict[str, torch.Tensor] = {}
        history: list[dict[str, Any]] = []
        steps = 0
        start_epoch = 0

    train_sampler = data.sampler(seed=int(seed), role="train")
    validation_sampler = data.sampler(
        seed=int(seed), role="internal_validation"
    )
    validation_draws = validation_sampler.sample(0, int(validation_samples))
    new_steps = 0
    gradient_clip = float(
        contract["grid_optimizer_and_runtime_contract"]["gradient_clip_norm"]
    )
    objective = contract["model_and_objective_contract"]
    for epoch in range(start_epoch + 1, int(maximum_epochs) + 1):
        if early.should_stop:
            break
        train_draws = train_sampler.sample(epoch, int(train_samples))
        train_metrics, epoch_steps, maximum_gradient, train_pairs = _epoch(
            model=model,
            optimizer=optimizer,
            data=data,
            draws=train_draws,
            batch_size=int(batch_size),
            device=resolved,
            gradient_clip=gradient_clip,
            objective=objective,
        )
        validation_metrics, _, _, validation_pairs = _epoch(
            model=model,
            optimizer=None,
            data=data,
            draws=validation_draws,
            batch_size=int(batch_size),
            device=resolved,
            gradient_clip=gradient_clip,
            objective=objective,
        )
        improved = early.update(
            epoch, validation_metrics["joint_objective"]
        )
        if improved:
            best_state = clone_model_state(model)
        steps += epoch_steps
        new_steps += epoch_steps
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "internal_validation": validation_metrics,
                "optimizer_steps": epoch_steps,
                "maximum_gradient_norm": maximum_gradient,
                "train_pair_count": train_pairs,
                "internal_validation_pair_count": validation_pairs,
                "train_draw_sha256": _draw_receipt(train_draws),
                "internal_validation_draw_sha256": _draw_receipt(
                    validation_draws
                ),
                "improved": improved,
            }
        )
        print(
            json.dumps(
                {
                    "v77_event": "epoch_complete",
                    "job_id": context["job_id"],
                    "epoch": epoch,
                    "optimizer_steps": steps,
                    "train_joint_objective": train_metrics["joint_objective"],
                    "internal_validation_joint_objective": validation_metrics[
                        "joint_objective"
                    ],
                    "improved": improved,
                    "early_stop": early.should_stop,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        payload = _payload(
            kind="resume",
            model=model,
            optimizer=optimizer,
            best_model_state=best_state,
            early=early,
            completed_epoch=epoch,
            optimizer_steps=steps,
            history=history,
            context=context,
            device=resolved,
        )
        _atomic_torch_save(payload, resume_path)
        if interrupt_after_epoch == epoch:
            return _result(
                resume_path, payload, status="interrupted", new_steps=new_steps
            )
    if not best_state:
        raise RuntimeError("V77 training has no strict best model state")
    model.load_state_dict(best_state, strict=True)
    final_payload = _payload(
        kind="final",
        model=model,
        optimizer=optimizer,
        best_model_state=best_state,
        early=early,
        completed_epoch=len(history),
        optimizer_steps=steps,
        history=history,
        context=context,
        device=resolved,
    )
    _atomic_torch_save(final_payload, final_path)
    resume_path.unlink(missing_ok=True)
    return _result(
        final_path, final_payload, status="completed", new_steps=new_steps
    )


def verify_v77_checkpoint(
    path: Path,
    *,
    blueprint: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _validate_payload(payload, kind="final", context=context)
    model = instantiate_v77_model(
        blueprint, torch.device("cpu"), seed=int(context["seed"])
    )
    model.load_state_dict(payload["model_current_state"], strict=True)
    optimizer = _optimizer(model, {
        "grid_optimizer_and_runtime_contract": context["optimizer_contract"]
    })
    optimizer.load_state_dict(payload["optimizer_state"])
    checks = {
        "complete_stage": payload["stage"] == "complete",
        "model_state_finite": all(
            not tensor.is_floating_point() or bool(torch.isfinite(tensor).all())
            for tensor in model.state_dict().values()
        ),
        "strict_best_state_present": bool(payload["model_best_state"]),
        "strict_best_epoch_registered": int(
            payload["early_stopping"]["best_epoch"]
        ) >= 1,
        "optimizer_steps_nonzero": int(payload["optimizer_steps"]) > 0,
        "prior_checkpoint_reused_false": payload["prior_checkpoint_reused"] is False,
        "history_matches_completed_epoch": len(payload["history"])
        == int(payload["completed_epoch"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "job_id": context["job_id"],
        "checkpoint_path": str(path),
        "checkpoint_file_sha256": file_sha256(path),
        "semantic_checkpoint_sha256": payload["semantic_checkpoint_sha256"],
        "model_state_sha256": semantic_state_sha256(
            payload["model_current_state"]
        ),
        "optimizer_state_sha256": semantic_state_sha256(
            payload["optimizer_state"]
        ),
        "optimizer_steps": int(payload["optimizer_steps"]),
        "completed_epoch": int(payload["completed_epoch"]),
        "best_epoch": int(payload["early_stopping"]["best_epoch"]),
    }
