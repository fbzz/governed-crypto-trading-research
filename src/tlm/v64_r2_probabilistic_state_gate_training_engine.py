"""Gate-only optimization and checkpoint mechanics for frozen V68 training."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .core.artifacts import canonical_sha256, file_sha256
from .decoupled_rank_state_harness import _build_ranker
from .state_conditioned_multi_horizon_training_engine import (
    StrictEarlyStopping,
    capture_v58_rng_state,
    clone_model_state,
    restore_v58_rng_state,
    semantic_state_sha256,
)
from .v64_r2_probabilistic_state_gate_harness import (
    ProbabilisticStateGate,
    student_t_negative_log_likelihood,
)
from .v64_r2_probabilistic_state_gate_training_data import (
    V68FoldTrainingData,
)


FINAL_FORMAT = "v68_v64_r2_probabilistic_state_gate_checkpoint_v1"
RESUME_FORMAT = "v68_v64_r2_probabilistic_state_gate_resume_v1"


def configure_v68_runtime(device: str, *, seed: int) -> torch.device:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0").strip().lower() not in {
        "", "0", "false", "no", "off"
    }:
        raise RuntimeError("V68 forbids PYTORCH_ENABLE_MPS_FALLBACK")
    torch.set_num_threads(10)
    torch.use_deterministic_algorithms(True)
    resolved = torch.device(device)
    if resolved.type == "mps":
        if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
            raise RuntimeError("V68 requires operational Apple MPS")
        probe = torch.ones(4, device=resolved, dtype=torch.float32)
        if float((probe * 2.0).sum().cpu()) != 8.0:
            raise RuntimeError("V68 MPS probe failed")
    elif resolved.type != "cpu":
        raise RuntimeError("V68 supports only MPS and CPU test execution")
    torch.manual_seed(int(seed))
    if resolved.type == "mps":
        torch.mps.manual_seed(int(seed))
    return resolved


def instantiate_v68_models(
    blueprint: dict[str, Any], device: torch.device, *, seed: int
) -> tuple[nn.Module, ProbabilisticStateGate]:
    torch.manual_seed(int(seed))
    ranker = _build_ranker(blueprint["ranker_contract"]["architecture"])
    gate = ProbabilisticStateGate(
        blueprint["state_gate_architecture"],
        degrees_of_freedom=float(blueprint["probabilistic_gate"]["degrees_of_freedom"]),
        scale_floor=float(blueprint["probabilistic_gate"]["scale_floor"]),
    )
    ranker.to(device=device, dtype=torch.float32)
    gate.to(device=device, dtype=torch.float32)
    return ranker, gate


def load_frozen_v63_ranker(
    path: Path,
    *,
    blueprint: dict[str, Any],
    device: torch.device,
    seed: int,
    expected_file_sha256: str,
    expected_ranker_state_sha256: str,
    expected_job_id: str,
) -> tuple[nn.Module, ProbabilisticStateGate, dict[str, Any]]:
    if file_sha256(path) != expected_file_sha256:
        raise RuntimeError("V68 source V63 checkpoint file hash drift")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("format_version") != "v63_decoupled_rank_state_checkpoint_v1"
        or payload.get("kind") != "final"
        or payload.get("context", {}).get("job_id") != expected_job_id
    ):
        raise RuntimeError("V68 source V63 checkpoint format/job drift")
    ranker_state = payload.get("ranker_current_state")
    if not isinstance(ranker_state, dict):
        raise RuntimeError("V68 source checkpoint lacks ranker state")
    if semantic_state_sha256(ranker_state) != expected_ranker_state_sha256:
        raise RuntimeError("V68 source ranker semantic identity drift")
    # The legacy container necessarily deserializes old gate tensors. They are
    # never accessed, hashed, applied to a model, selected, or reused.
    del payload
    ranker, gate = instantiate_v68_models(blueprint, device, seed=seed)
    ranker.load_state_dict(ranker_state, strict=True)
    for parameter in ranker.parameters():
        parameter.requires_grad_(False)
    ranker.eval()
    receipt = {
        "source_checkpoint_path": str(path),
        "source_checkpoint_sha256": expected_file_sha256,
        "source_ranker_state_sha256": expected_ranker_state_sha256,
        "legacy_gate_container_tensors_deserialized": True,
        "old_gate_substate_loaded_into_model": False,
        "old_gate_substate_values_inspected_or_hashed": False,
        "old_gate_substate_selected_or_reused": False,
        "ranker_requires_grad": any(p.requires_grad for p in ranker.parameters()),
        "ranker_optimizer_present": False,
    }
    return ranker, gate, receipt


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


def _atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _optimizer(gate: ProbabilisticStateGate, contract: dict[str, Any]) -> torch.optim.AdamW:
    config = contract["grid_optimizer_and_runtime_contract"]["optimizer"]
    return torch.optim.AdamW(
        gate.parameters(),
        lr=float(config["learning_rate"]),
        betas=tuple(float(value) for value in config["betas"]),
        eps=float(config["epsilon"]),
        weight_decay=float(config["weight_decay"]),
    )


def _draw_receipt(draws: list[Any]) -> str:
    return canonical_sha256(
        [
            {"date": draw.date.isoformat(), "triplet": list(draw.triplet), "pair_index": draw.pair_index}
            for draw in draws
        ]
    )


def _epoch(
    *,
    gate: ProbabilisticStateGate,
    optimizer: torch.optim.Optimizer | None,
    data: V68FoldTrainingData,
    draws: list[Any],
    batch_size: int,
    device: torch.device,
    gradient_clip: float,
) -> tuple[float, int, float]:
    training = optimizer is not None
    gate.train(training)
    weighted = 0.0
    observations = 0
    steps = 0
    maximum_gradient = 0.0
    for start in range(0, len(draws), int(batch_size)):
        state, target = data.store.materialize(
            draws[start : start + int(batch_size)],
            data.scale.feature_scaler,
            market_target_rms=data.scale.market_target_rms,
        )
        state_tensor = torch.from_numpy(state).to(device=device, dtype=torch.float32)
        target_tensor = torch.from_numpy(target).to(device=device, dtype=torch.float32)
        with torch.set_grad_enabled(training):
            loss = student_t_negative_log_likelihood(
                gate(state_tensor), target_tensor, degrees_of_freedom=5.0
            )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("V68 Student-t NLL is non-finite")
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = nn.utils.clip_grad_norm_(gate.parameters(), float(gradient_clip))
            if not bool(torch.isfinite(gradient)):
                raise RuntimeError("V68 gate gradient norm is non-finite")
            maximum_gradient = max(maximum_gradient, float(gradient.detach().cpu()))
            optimizer.step()
            steps += 1
        weighted += float(loss.detach().cpu()) * len(state)
        observations += len(state)
    return weighted / observations, steps, maximum_gradient


def _payload(
    *,
    kind: str,
    ranker: nn.Module,
    gate: ProbabilisticStateGate,
    optimizer: torch.optim.Optimizer,
    best_gate_state: dict[str, torch.Tensor],
    early: StrictEarlyStopping,
    completed_epoch: int,
    optimizer_steps: int,
    history: list[dict[str, Any]],
    context: dict[str, Any],
    source_receipt: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    rng = capture_v58_rng_state(device)
    value: dict[str, Any] = {
        "format_version": FINAL_FORMAT if kind == "final" else RESUME_FORMAT,
        "kind": kind,
        "stage": "complete" if kind == "final" else "gate_training",
        "completed_epoch": int(completed_epoch),
        "ranker_state": clone_model_state(ranker),
        "gate_current_state": clone_model_state(gate),
        "gate_best_state": _to_cpu(best_gate_state),
        "gate_optimizer_state": _to_cpu(optimizer.state_dict()),
        "gate_early_stopping": asdict(early),
        "optimizer_steps": int(optimizer_steps),
        "cpu_rng_state": rng["cpu_rng_state"],
        "mps_rng_state": rng["mps_rng_state"],
        "history": deepcopy(history),
        "source_ranker_checkpoint_sha256": source_receipt["source_checkpoint_sha256"],
        "source_ranker_state_sha256": source_receipt["source_ranker_state_sha256"],
        "fold_feature_scaler_sha256": context["fold_feature_scaler_sha256"],
        "market_target_scaler_sha256": context["market_target_scaler_sha256"],
        "phase_contract_sha256": context["phase_contract_sha256"],
        "source_bundle_sha256": context["source_bundle_sha256"],
        "job_context": deepcopy(context),
        "ranker_optimizer_present": False,
        "old_gate_state_present": False,
    }
    value["semantic_checkpoint_sha256"] = semantic_state_sha256(value)
    return value


def _validate_payload(
    payload: dict[str, Any], *, kind: str, context: dict[str, Any]
) -> None:
    expected_format = FINAL_FORMAT if kind == "final" else RESUME_FORMAT
    if payload.get("format_version") != expected_format or payload.get("kind") != kind:
        raise RuntimeError("V68 checkpoint format/kind drift")
    registered = payload.get("semantic_checkpoint_sha256")
    body = {key: value for key, value in payload.items() if key != "semantic_checkpoint_sha256"}
    if semantic_state_sha256(body) != registered:
        raise RuntimeError("V68 semantic checkpoint hash drift")
    if payload.get("job_context") != context:
        raise RuntimeError("V68 checkpoint job context drift")
    if payload.get("ranker_optimizer_present") is not False or payload.get("old_gate_state_present") is not False:
        raise RuntimeError("V68 checkpoint contains forbidden optimizer/gate state")


def _result(path: Path, payload: dict[str, Any], *, status: str, new_steps: int) -> dict[str, Any]:
    return {
        "job_id": payload["job_context"]["job_id"],
        "fold": payload["job_context"]["fold"],
        "seed": payload["job_context"]["seed"],
        "status": status,
        "completed": payload["stage"] == "complete",
        "completed_epoch": payload["completed_epoch"],
        "checkpoint_path": str(path),
        "checkpoint_file_sha256": file_sha256(path),
        "semantic_checkpoint_sha256": payload["semantic_checkpoint_sha256"],
        "ranker_state_sha256": semantic_state_sha256(payload["ranker_state"]),
        "gate_state_sha256": semantic_state_sha256(payload["gate_current_state"]),
        "optimizer_state_sha256": semantic_state_sha256(payload["gate_optimizer_state"]),
        "optimizer_steps": payload["optimizer_steps"],
        "new_optimizer_steps": int(new_steps),
        "history": deepcopy(payload["history"]),
    }


def run_v68_training_job(
    *,
    blueprint: dict[str, Any],
    contract: dict[str, Any],
    data: V68FoldTrainingData,
    seed: int,
    context: dict[str, Any],
    source_checkpoint_path: Path,
    source_checkpoint_file_sha256: str,
    source_ranker_state_sha256: str,
    resume_path: Path,
    final_path: Path,
    device: str = "mps",
    train_samples: int = 8192,
    validation_samples: int = 2048,
    batch_size: int = 128,
    maximum_epochs: int = 30,
    patience: int = 5,
    interrupt_after_epoch: int | None = None,
) -> dict[str, Any]:
    resolved = configure_v68_runtime(device, seed=int(seed))
    if final_path.is_file():
        payload = torch.load(final_path, map_location="cpu", weights_only=False)
        _validate_payload(payload, kind="final", context=context)
        return _result(final_path, payload, status="already_complete", new_steps=0)

    if resume_path.is_file():
        ranker, gate = instantiate_v68_models(blueprint, resolved, seed=int(seed))
        for parameter in ranker.parameters():
            parameter.requires_grad_(False)
        optimizer = _optimizer(gate, contract)
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        _validate_payload(payload, kind="resume", context=context)
        ranker.load_state_dict(payload["ranker_state"], strict=True)
        gate.load_state_dict(payload["gate_current_state"], strict=True)
        optimizer.load_state_dict(payload["gate_optimizer_state"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(resolved)
        early = StrictEarlyStopping(**payload["gate_early_stopping"])
        best_gate = _to_cpu(payload["gate_best_state"])
        history = deepcopy(payload["history"])
        steps = int(payload["optimizer_steps"])
        start_epoch = int(payload["completed_epoch"])
        source_receipt = {
            "source_checkpoint_sha256": payload["source_ranker_checkpoint_sha256"],
            "source_ranker_state_sha256": payload["source_ranker_state_sha256"],
        }
        restore_v58_rng_state(
            cpu_rng_state=payload["cpu_rng_state"],
            mps_rng_state=payload["mps_rng_state"],
            device=resolved,
        )
    else:
        ranker, gate, source_receipt = load_frozen_v63_ranker(
            source_checkpoint_path,
            blueprint=blueprint,
            device=resolved,
            seed=int(seed),
            expected_file_sha256=source_checkpoint_file_sha256,
            expected_ranker_state_sha256=source_ranker_state_sha256,
            expected_job_id=context["job_id"],
        )
        optimizer = _optimizer(gate, contract)
        early = StrictEarlyStopping(patience=int(patience))
        best_gate: dict[str, torch.Tensor] = {}
        history: list[dict[str, Any]] = []
        steps = 0
        start_epoch = 0

    if any(parameter.requires_grad for parameter in ranker.parameters()):
        raise RuntimeError("V68 frozen ranker has a gradient path")
    train_sampler = data.sampler(seed=int(seed), role="gate_train")
    validation_sampler = data.sampler(seed=int(seed), role="gate_internal_validation")
    validation_draws = validation_sampler.sample(0, int(validation_samples))
    new_steps = 0
    gradient_clip = float(contract["grid_optimizer_and_runtime_contract"]["gradient_clip_norm"])
    for epoch in range(start_epoch + 1, int(maximum_epochs) + 1):
        if early.should_stop:
            break
        train_draws = train_sampler.sample(epoch, int(train_samples))
        train_loss, epoch_steps, maximum_gradient = _epoch(
            gate=gate, optimizer=optimizer, data=data, draws=train_draws,
            batch_size=int(batch_size), device=resolved, gradient_clip=gradient_clip,
        )
        validation_loss, _, _ = _epoch(
            gate=gate, optimizer=None, data=data, draws=validation_draws,
            batch_size=int(batch_size), device=resolved, gradient_clip=gradient_clip,
        )
        improved = early.update(epoch, validation_loss)
        if improved:
            best_gate = clone_model_state(gate)
        steps += epoch_steps
        new_steps += epoch_steps
        history.append(
            {
                "epoch": epoch,
                "train_negative_log_likelihood": train_loss,
                "validation_negative_log_likelihood": validation_loss,
                "optimizer_steps": epoch_steps,
                "maximum_gradient_norm": maximum_gradient,
                "train_draw_sha256": _draw_receipt(train_draws),
                "validation_draw_sha256": _draw_receipt(validation_draws),
                "improved": improved,
            }
        )
        payload = _payload(
            kind="resume", ranker=ranker, gate=gate, optimizer=optimizer,
            best_gate_state=best_gate, early=early, completed_epoch=epoch,
            optimizer_steps=steps, history=history, context=context,
            source_receipt=source_receipt, device=resolved,
        )
        _atomic_torch_save(payload, resume_path)
        if interrupt_after_epoch == epoch:
            return _result(resume_path, payload, status="interrupted", new_steps=new_steps)
    if not best_gate:
        raise RuntimeError("V68 training has no strict best gate state")
    gate.load_state_dict(best_gate, strict=True)
    final_payload = _payload(
        kind="final", ranker=ranker, gate=gate, optimizer=optimizer,
        best_gate_state=best_gate, early=early, completed_epoch=len(history),
        optimizer_steps=steps, history=history, context=context,
        source_receipt=source_receipt, device=resolved,
    )
    _atomic_torch_save(final_payload, final_path)
    resume_path.unlink(missing_ok=True)
    return _result(final_path, final_payload, status="completed", new_steps=new_steps)


def verify_v68_checkpoint(
    path: Path,
    *,
    blueprint: dict[str, Any],
    context: dict[str, Any],
    expected_ranker_state_sha256: str,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _validate_payload(payload, kind="final", context=context)
    ranker, gate = instantiate_v68_models(blueprint, torch.device("cpu"), seed=int(context["seed"]))
    ranker.load_state_dict(payload["ranker_state"], strict=True)
    gate.load_state_dict(payload["gate_current_state"], strict=True)
    for parameter in ranker.parameters():
        parameter.requires_grad_(False)
    checks = {
        "complete_stage": payload["stage"] == "complete",
        "ranker_state_identity": semantic_state_sha256(payload["ranker_state"]) == expected_ranker_state_sha256,
        "ranker_requires_grad_false": not any(p.requires_grad for p in ranker.parameters()),
        "ranker_optimizer_absent": payload["ranker_optimizer_present"] is False,
        "old_gate_state_absent": payload["old_gate_state_present"] is False,
        "gate_state_finite": all(not x.is_floating_point() or bool(torch.isfinite(x).all()) for x in gate.state_dict().values()),
        "gate_optimizer_steps_nonzero": int(payload["optimizer_steps"]) > 0,
        "strict_best_gate_present": bool(payload["gate_best_state"]),
    }
    return {
        "passed": all(checks.values()), "checks": checks,
        "job_id": context["job_id"], "checkpoint_path": str(path),
        "checkpoint_file_sha256": file_sha256(path),
        "semantic_checkpoint_sha256": payload["semantic_checkpoint_sha256"],
        "ranker_state_sha256": semantic_state_sha256(payload["ranker_state"]),
        "gate_state_sha256": semantic_state_sha256(payload["gate_current_state"]),
        "optimizer_state_sha256": semantic_state_sha256(payload["gate_optimizer_state"]),
        "optimizer_steps": int(payload["optimizer_steps"]),
    }
