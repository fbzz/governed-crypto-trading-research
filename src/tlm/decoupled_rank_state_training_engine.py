"""Optimization, resume, and checkpoint mechanics for frozen V63 training."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math
import os
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch.nn import functional as F

from .core.artifacts import canonical_sha256, file_sha256
from .decoupled_rank_state_harness import IndependentStateGate, derive_state_features
from .decoupled_rank_state_training_data import FoldTrainingData, SampleDraw
from .patch_transformer import MultiAssetPatchTransformer
from .ranking_excess_harness import RANKING_EXCESS_HEADS, ranking_excess_loss
from .ranking_excess_pretraining import configure_pretraining_scope
from .ranking_excess_supervised import configure_supervised_scope
from .scientific_harness import deterministic_patch_mask, masked_reconstruction_loss
from .state_conditioned_multi_horizon_training_engine import (
    StrictEarlyStopping,
    capture_v58_rng_state,
    clone_model_state,
    restore_v58_rng_state,
    semantic_state_sha256,
)


FINAL_FORMAT = "v63_decoupled_rank_state_checkpoint_v1"
RESUME_FORMAT = "v63_decoupled_rank_state_resume_v1"
Stage = Literal["pretraining", "supervised", "complete"]


def configure_v63_runtime(device: str, *, seed: int) -> torch.device:
    resolved = torch.device(device)
    if resolved.type not in {"cpu", "mps"}:
        raise ValueError("V63 supports only CPU verification or MPS training")
    torch.set_num_threads(10)
    torch.use_deterministic_algorithms(True)
    if resolved.type == "mps":
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
            raise RuntimeError("V63 MPS requires PYTORCH_ENABLE_MPS_FALLBACK=0")
        if not torch.backends.mps.is_available():
            raise RuntimeError("V63 MPS runtime is unavailable")
    torch.manual_seed(int(seed))
    if resolved.type == "mps":
        torch.mps.manual_seed(int(seed))
    return resolved


def _ranker_architecture(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "input_triplet_size": 3}


def instantiate_models(
    blueprint: dict[str, Any], device: torch.device
) -> tuple[MultiAssetPatchTransformer, IndependentStateGate]:
    ranker = MultiAssetPatchTransformer(
        9,
        _ranker_architecture(blueprint["architecture"]["ranker"]),
        expected_prediction_heads=RANKING_EXCESS_HEADS,
    ).to(device=device, dtype=torch.float32)
    gate = IndependentStateGate(
        blueprint["architecture"]["state_gate"]
    ).to(device=device, dtype=torch.float32)
    counts = (
        sum(parameter.numel() for parameter in ranker.parameters()),
        sum(parameter.numel() for parameter in gate.parameters()),
    )
    expected = blueprint["architecture"]["parameter_counts"]
    if counts != (int(expected["ranker"]), int(expected["state_gate"])):
        raise RuntimeError(f"V63 model parameter-count drift: {counts}")
    if set(map(id, ranker.parameters())).intersection(map(id, gate.parameters())):
        raise RuntimeError("V63 ranker and state gate unexpectedly share parameters")
    return ranker, gate


def _optimizer(parameters: list[nn.Parameter], contract: dict[str, Any]) -> torch.optim.AdamW:
    optimizer = contract["grid_optimizer_and_runtime_contract"]["optimizer"]
    return torch.optim.AdamW(
        parameters,
        lr=float(optimizer["learning_rate"]),
        betas=tuple(float(value) for value in optimizer["betas"]),
        eps=float(optimizer["epsilon"]),
        weight_decay=float(optimizer["weight_decay"]),
        amsgrad=False,
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


def _move_optimizer(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary, _use_new_zipfile_serialization=False)
    temporary.replace(path)


def _draw_receipt(draws: list[SampleDraw]) -> str:
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


def _batches(
    data: FoldTrainingData,
    draws: list[SampleDraw],
    batch_size: int,
    *,
    require_targets: bool = True,
):
    for start in range(0, len(draws), batch_size):
        features, targets = data.store.materialize(
            draws[start : start + batch_size],
            data.scale.feature_scaler,
            require_targets=require_targets,
        )
        yield start // batch_size, torch.from_numpy(features), torch.from_numpy(targets)


def _pretraining_epoch(
    *,
    ranker: MultiAssetPatchTransformer,
    optimizer: torch.optim.Optimizer | None,
    data: FoldTrainingData,
    draws: list[SampleDraw],
    batch_size: int,
    epoch: int,
    seed: int,
    device: torch.device,
    mask_fraction: float,
    gradient_clip: float,
) -> tuple[float, int, float]:
    training = optimizer is not None
    ranker.train(training)
    parameters = [parameter for parameter in ranker.parameters() if parameter.requires_grad]
    weighted = 0.0
    observations = 0
    steps = 0
    max_gradient = 0.0
    for batch_index, x, _ in _batches(
        data, draws, batch_size, require_targets=False
    ):
        x = x.to(device=device, dtype=torch.float32)
        mask = deterministic_patch_mask(
            len(x), 3, ranker.patch_count, mask_fraction, seed, data.fold,
            epoch, batch_index,
        ).to(device)
        with torch.set_grad_enabled(training):
            target = ranker.extract_patches(x)
            reconstruction = ranker.reconstruct_masked_patches(x, mask)
            loss = masked_reconstruction_loss(reconstruction, target, mask, beta=1.0)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("V63 masked reconstruction loss is non-finite")
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = nn.utils.clip_grad_norm_(parameters, gradient_clip)
            if not bool(torch.isfinite(gradient)):
                raise RuntimeError("V63 pretraining gradient norm is non-finite")
            max_gradient = max(max_gradient, float(gradient.detach().cpu()))
            optimizer.step()
            steps += 1
        weighted += float(loss.detach().cpu()) * len(x)
        observations += len(x)
    return weighted / observations, steps, max_gradient


def _supervised_epoch(
    *,
    ranker: MultiAssetPatchTransformer,
    gate: IndependentStateGate,
    ranker_optimizer: torch.optim.Optimizer | None,
    gate_optimizer: torch.optim.Optimizer | None,
    data: FoldTrainingData,
    draws: list[SampleDraw],
    batch_size: int,
    device: torch.device,
    gradient_clip: float,
) -> tuple[dict[str, float], int, int, float, float]:
    training = ranker_optimizer is not None and gate_optimizer is not None
    if (ranker_optimizer is None) != (gate_optimizer is None):
        raise RuntimeError("V63 ranker/gate training mode diverged")
    ranker.train(training)
    gate.train(training)
    ranker_parameters = [p for p in ranker.parameters() if p.requires_grad]
    gate_parameters = [p for p in gate.parameters() if p.requires_grad]
    sums = {name: 0.0 for name in ("ranking", "excess", "log_volatility", "ranker_core", "ranker_total", "gate")}
    observations = 0
    ranker_steps = 0
    gate_steps = 0
    max_ranker_gradient = 0.0
    max_gate_gradient = 0.0
    for _, x, labels in _batches(data, draws, batch_size):
        x = x.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.float32)
        with torch.set_grad_enabled(training):
            ranker_output = ranker(x)
            losses = ranking_excess_loss(
                ranker_output,
                labels,
                data.scale.excess_rms,
                tie_tolerance=1.0e-12,
                volatility_floor=1.0e-8,
                ranking_weight=1.0,
                excess_weight=1.0,
                volatility_weight=0.1,
            )
        if not bool(torch.isfinite(losses["total"])):
            raise RuntimeError("V63 ranker supervised loss is non-finite")
        if training:
            ranker_optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            gradient = nn.utils.clip_grad_norm_(ranker_parameters, gradient_clip)
            if not bool(torch.isfinite(gradient)):
                raise RuntimeError("V63 ranker gradient norm is non-finite")
            max_ranker_gradient = max(
                max_ranker_gradient, float(gradient.detach().cpu())
            )
            ranker_optimizer.step()
            ranker_steps += 1

        state_features = derive_state_features(x.detach())
        market_target = labels[..., 0].mean(dim=1) / float(data.scale.market_rms)
        with torch.set_grad_enabled(training):
            gate_output = gate(state_features)
            gate_loss = F.smooth_l1_loss(
                gate_output, market_target, beta=1.0
            )
        if not bool(torch.isfinite(gate_loss)):
            raise RuntimeError("V63 gate supervised loss is non-finite")
        if training:
            gate_optimizer.zero_grad(set_to_none=True)
            gate_loss.backward()
            gradient = nn.utils.clip_grad_norm_(gate_parameters, gradient_clip)
            if not bool(torch.isfinite(gradient)):
                raise RuntimeError("V63 gate gradient norm is non-finite")
            max_gate_gradient = max(max_gate_gradient, float(gradient.detach().cpu()))
            gate_optimizer.step()
            gate_steps += 1

        count = len(x)
        sums["ranking"] += float(losses["ranking"].detach().cpu()) * count
        sums["excess"] += float(losses["excess"].detach().cpu()) * count
        sums["log_volatility"] += float(losses["log_volatility"].detach().cpu()) * count
        sums["ranker_core"] += float(losses["core"].detach().cpu()) * count
        sums["ranker_total"] += float(losses["total"].detach().cpu()) * count
        sums["gate"] += float(gate_loss.detach().cpu()) * count
        observations += count
    return (
        {name: value / observations for name, value in sums.items()},
        ranker_steps,
        gate_steps,
        max_ranker_gradient,
        max_gate_gradient,
    )


def _checkpoint_payload(
    *,
    kind: Literal["resume", "final"],
    stage: Stage,
    stage_epoch: int,
    ranker: MultiAssetPatchTransformer,
    gate: IndependentStateGate,
    pretraining_optimizer: torch.optim.Optimizer,
    ranker_optimizer: torch.optim.Optimizer,
    gate_optimizer: torch.optim.Optimizer,
    pretraining_best: dict[str, torch.Tensor],
    ranker_best: dict[str, torch.Tensor],
    gate_best: dict[str, torch.Tensor],
    pretraining_early: StrictEarlyStopping,
    ranker_early: StrictEarlyStopping,
    gate_early: StrictEarlyStopping,
    history: dict[str, list[dict[str, Any]]],
    optimizer_steps: dict[str, int],
    context: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    rng = capture_v58_rng_state(device)
    payload: dict[str, Any] = {
        "format_version": FINAL_FORMAT if kind == "final" else RESUME_FORMAT,
        "kind": kind,
        "stage": stage,
        "stage_epoch": int(stage_epoch),
        "ranker_current_state": clone_model_state(ranker),
        "gate_current_state": clone_model_state(gate),
        "pretraining_best_state": _to_cpu(pretraining_best),
        "ranker_best_state": _to_cpu(ranker_best),
        "gate_best_state": _to_cpu(gate_best),
        "pretraining_optimizer_state": _to_cpu(pretraining_optimizer.state_dict()),
        "ranker_optimizer_state": _to_cpu(ranker_optimizer.state_dict()),
        "gate_optimizer_state": _to_cpu(gate_optimizer.state_dict()),
        "pretraining_early_stopping": asdict(pretraining_early),
        "ranker_early_stopping": asdict(ranker_early),
        "gate_early_stopping": asdict(gate_early),
        "history": deepcopy(history),
        "optimizer_steps": dict(optimizer_steps),
        "cpu_rng_state": rng["cpu_rng_state"],
        "mps_rng_state": rng["mps_rng_state"],
        "context": deepcopy(context),
    }
    payload["semantic_checkpoint_sha256"] = semantic_state_sha256(payload)
    return payload


def _load_checkpoint(
    path: Path,
    *,
    expected_kind: Literal["resume", "final"],
    ranker: MultiAssetPatchTransformer,
    gate: IndependentStateGate,
    pretraining_optimizer: torch.optim.Optimizer,
    ranker_optimizer: torch.optim.Optimizer,
    gate_optimizer: torch.optim.Optimizer,
    context: dict[str, Any],
    device: torch.device,
    restore_rng: bool,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected_format = FINAL_FORMAT if expected_kind == "final" else RESUME_FORMAT
    if payload.get("format_version") != expected_format or payload.get("kind") != expected_kind:
        raise RuntimeError("V63 checkpoint format/kind drift")
    registered = payload.get("semantic_checkpoint_sha256")
    body = {key: value for key, value in payload.items() if key != "semantic_checkpoint_sha256"}
    if semantic_state_sha256(body) != registered:
        raise RuntimeError("V63 semantic checkpoint hash drift")
    if payload.get("context") != context:
        raise RuntimeError("V63 checkpoint context drift")
    ranker.load_state_dict(payload["ranker_current_state"], strict=True)
    gate.load_state_dict(payload["gate_current_state"], strict=True)
    pretraining_optimizer.load_state_dict(payload["pretraining_optimizer_state"])
    ranker_optimizer.load_state_dict(payload["ranker_optimizer_state"])
    gate_optimizer.load_state_dict(payload["gate_optimizer_state"])
    for optimizer in (pretraining_optimizer, ranker_optimizer, gate_optimizer):
        _move_optimizer(optimizer, device)
    if restore_rng:
        restore_v58_rng_state(
            cpu_rng_state=payload["cpu_rng_state"],
            mps_rng_state=payload["mps_rng_state"],
            device=device,
        )
    return payload


def _result(path: Path, payload: dict[str, Any], *, status: str, new_steps: int) -> dict[str, Any]:
    return {
        "job_id": payload["context"]["job_id"],
        "fold": payload["context"]["fold"],
        "seed": payload["context"]["seed"],
        "status": status,
        "completed": payload["stage"] == "complete",
        "stage": payload["stage"],
        "stage_epoch": payload["stage_epoch"],
        "checkpoint_path": str(path),
        "checkpoint_file_sha256": file_sha256(path),
        "semantic_checkpoint_sha256": payload["semantic_checkpoint_sha256"],
        "ranker_state_sha256": semantic_state_sha256(payload["ranker_current_state"]),
        "gate_state_sha256": semantic_state_sha256(payload["gate_current_state"]),
        "optimizer_steps": dict(payload["optimizer_steps"]),
        "new_optimizer_steps": int(new_steps),
        "history": deepcopy(payload["history"]),
    }


def run_training_job(
    *,
    blueprint: dict[str, Any],
    contract: dict[str, Any],
    data: FoldTrainingData,
    seed: int,
    context: dict[str, Any],
    resume_path: Path,
    final_path: Path,
    device: str = "mps",
    pretraining_samples: int = 8192,
    supervised_samples: int = 8192,
    validation_samples: int = 2048,
    batch_size: int = 128,
    pretraining_epochs: int = 50,
    supervised_epochs: int = 30,
    patience: int = 5,
    interrupt_at: tuple[str, int] | None = None,
) -> dict[str, Any]:
    resolved = configure_v63_runtime(device, seed=int(seed))
    ranker, gate = instantiate_models(blueprint, resolved)
    pretraining_parameters = configure_pretraining_scope(ranker)
    pretraining_optimizer = _optimizer(pretraining_parameters, contract)
    supervised_parameters = configure_supervised_scope(ranker)
    ranker_optimizer = _optimizer(supervised_parameters, contract)
    gate_parameters = list(gate.parameters())
    gate_optimizer = _optimizer(gate_parameters, contract)
    if set(map(id, supervised_parameters)).intersection(map(id, gate_parameters)):
        raise RuntimeError("V63 optimizer parameter sets overlap")

    if final_path.is_file():
        payload = _load_checkpoint(
            final_path,
            expected_kind="final",
            ranker=ranker,
            gate=gate,
            pretraining_optimizer=pretraining_optimizer,
            ranker_optimizer=ranker_optimizer,
            gate_optimizer=gate_optimizer,
            context=context,
            device=resolved,
            restore_rng=False,
        )
        return _result(final_path, payload, status="already_complete", new_steps=0)

    pretraining_early = StrictEarlyStopping(patience=patience)
    ranker_early = StrictEarlyStopping(patience=patience)
    gate_early = StrictEarlyStopping(patience=patience)
    pretraining_best: dict[str, torch.Tensor] = {}
    ranker_best: dict[str, torch.Tensor] = {}
    gate_best: dict[str, torch.Tensor] = {}
    history: dict[str, list[dict[str, Any]]] = {"pretraining": [], "supervised": []}
    steps = {"pretraining": 0, "ranker": 0, "gate": 0}
    stage: Stage = "pretraining"
    stage_epoch = 0
    if resume_path.is_file():
        payload = _load_checkpoint(
            resume_path,
            expected_kind="resume",
            ranker=ranker,
            gate=gate,
            pretraining_optimizer=pretraining_optimizer,
            ranker_optimizer=ranker_optimizer,
            gate_optimizer=gate_optimizer,
            context=context,
            device=resolved,
            restore_rng=True,
        )
        stage = payload["stage"]
        stage_epoch = int(payload["stage_epoch"])
        pretraining_best = payload["pretraining_best_state"]
        ranker_best = payload["ranker_best_state"]
        gate_best = payload["gate_best_state"]
        pretraining_early = StrictEarlyStopping(**payload["pretraining_early_stopping"])
        ranker_early = StrictEarlyStopping(**payload["ranker_early_stopping"])
        gate_early = StrictEarlyStopping(**payload["gate_early_stopping"])
        history = deepcopy(payload["history"])
        steps = dict(payload["optimizer_steps"])
    new_steps = 0
    pretraining_train_sampler = data.sampler(
        seed=int(seed), role="pretraining_train"
    )
    pretraining_validation_sampler = data.sampler(
        seed=int(seed), role="pretraining_validation"
    )
    supervised_train_sampler = data.sampler(
        seed=int(seed), role="supervised_train"
    )
    supervised_validation_sampler = data.sampler(
        seed=int(seed), role="supervised_validation"
    )
    pretraining_validation_draws = pretraining_validation_sampler.sample(
        0, validation_samples
    )
    supervised_validation_draws = supervised_validation_sampler.sample(
        0, validation_samples
    )
    gradient_clip = float(
        contract["grid_optimizer_and_runtime_contract"]["supervised_training"][
            "gradient_clip_norm"
        ]
    )
    mask_fraction = float(
        contract["model_and_objective_contract"]["ranker"]["pretraining"]["mask_fraction"]
    )

    if stage == "pretraining":
        configure_pretraining_scope(ranker)
        for epoch in range(stage_epoch + 1, pretraining_epochs + 1):
            train_draws = pretraining_train_sampler.sample(
                epoch, pretraining_samples
            )
            train_loss, epoch_steps, maximum_gradient = _pretraining_epoch(
                ranker=ranker,
                optimizer=pretraining_optimizer,
                data=data,
                draws=train_draws,
                batch_size=batch_size,
                epoch=epoch,
                seed=int(seed),
                device=resolved,
                mask_fraction=mask_fraction,
                gradient_clip=gradient_clip,
            )
            validation_loss, _, _ = _pretraining_epoch(
                ranker=ranker,
                optimizer=None,
                data=data,
                draws=pretraining_validation_draws,
                batch_size=batch_size,
                epoch=0,
                seed=int(seed),
                device=resolved,
                mask_fraction=mask_fraction,
                gradient_clip=gradient_clip,
            )
            improved = pretraining_early.update(epoch, validation_loss)
            if improved:
                pretraining_best = clone_model_state(ranker)
            steps["pretraining"] += epoch_steps
            new_steps += epoch_steps
            history["pretraining"].append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "optimizer_steps": epoch_steps,
                    "maximum_gradient_norm": maximum_gradient,
                    "train_draw_sha256": _draw_receipt(train_draws),
                    "validation_draw_sha256": _draw_receipt(
                        pretraining_validation_draws
                    ),
                    "improved": improved,
                }
            )
            payload = _checkpoint_payload(
                kind="resume", stage="pretraining", stage_epoch=epoch,
                ranker=ranker, gate=gate,
                pretraining_optimizer=pretraining_optimizer,
                ranker_optimizer=ranker_optimizer, gate_optimizer=gate_optimizer,
                pretraining_best=pretraining_best, ranker_best=ranker_best,
                gate_best=gate_best, pretraining_early=pretraining_early,
                ranker_early=ranker_early, gate_early=gate_early,
                history=history, optimizer_steps=steps, context=context,
                device=resolved,
            )
            _atomic_torch_save(payload, resume_path)
            if interrupt_at == ("pretraining", epoch):
                return _result(resume_path, payload, status="interrupted", new_steps=new_steps)
            if pretraining_early.should_stop:
                break
        if not pretraining_best:
            raise RuntimeError("V63 pretraining has no strict best state")
        ranker.load_state_dict(pretraining_best, strict=True)
        configure_supervised_scope(ranker)
        stage = "supervised"
        stage_epoch = 0
        payload = _checkpoint_payload(
            kind="resume", stage=stage, stage_epoch=0, ranker=ranker, gate=gate,
            pretraining_optimizer=pretraining_optimizer,
            ranker_optimizer=ranker_optimizer, gate_optimizer=gate_optimizer,
            pretraining_best=pretraining_best, ranker_best=ranker_best,
            gate_best=gate_best, pretraining_early=pretraining_early,
            ranker_early=ranker_early, gate_early=gate_early, history=history,
            optimizer_steps=steps, context=context, device=resolved,
        )
        _atomic_torch_save(payload, resume_path)

    if stage == "supervised":
        configure_supervised_scope(ranker)
        for epoch in range(stage_epoch + 1, supervised_epochs + 1):
            if ranker_early.should_stop and gate_early.should_stop:
                break
            train_draws = supervised_train_sampler.sample(
                10_000 + epoch, supervised_samples
            )
            train_losses, ranker_steps, gate_steps, max_ranker, max_gate = _supervised_epoch(
                ranker=ranker, gate=gate,
                ranker_optimizer=None if ranker_early.should_stop else ranker_optimizer,
                gate_optimizer=None if gate_early.should_stop else gate_optimizer,
                data=data, draws=train_draws, batch_size=batch_size,
                device=resolved, gradient_clip=gradient_clip,
            ) if ranker_early.should_stop == gate_early.should_stop else _supervised_epoch_independent(
                ranker=ranker, gate=gate, ranker_optimizer=ranker_optimizer,
                gate_optimizer=gate_optimizer, ranker_active=not ranker_early.should_stop,
                gate_active=not gate_early.should_stop, data=data, draws=train_draws,
                batch_size=batch_size, device=resolved, gradient_clip=gradient_clip,
            )
            validation_losses, _, _, _, _ = _supervised_epoch(
                ranker=ranker, gate=gate, ranker_optimizer=None, gate_optimizer=None,
                data=data, draws=supervised_validation_draws, batch_size=batch_size,
                device=resolved, gradient_clip=gradient_clip,
            )
            ranker_improved = False
            gate_improved = False
            if not ranker_early.should_stop:
                ranker_improved = ranker_early.update(epoch, validation_losses["ranker_core"])
                if ranker_improved:
                    ranker_best = clone_model_state(ranker)
            if not gate_early.should_stop:
                gate_improved = gate_early.update(epoch, validation_losses["gate"])
                if gate_improved:
                    gate_best = clone_model_state(gate)
            steps["ranker"] += ranker_steps
            steps["gate"] += gate_steps
            new_steps += ranker_steps + gate_steps
            history["supervised"].append(
                {
                    "epoch": epoch,
                    "train_losses": train_losses,
                    "validation_losses": validation_losses,
                    "ranker_optimizer_steps": ranker_steps,
                    "gate_optimizer_steps": gate_steps,
                    "maximum_ranker_gradient_norm": max_ranker,
                    "maximum_gate_gradient_norm": max_gate,
                    "train_draw_sha256": _draw_receipt(train_draws),
                    "validation_draw_sha256": _draw_receipt(
                        supervised_validation_draws
                    ),
                    "ranker_improved": ranker_improved,
                    "gate_improved": gate_improved,
                }
            )
            payload = _checkpoint_payload(
                kind="resume", stage="supervised", stage_epoch=epoch,
                ranker=ranker, gate=gate,
                pretraining_optimizer=pretraining_optimizer,
                ranker_optimizer=ranker_optimizer, gate_optimizer=gate_optimizer,
                pretraining_best=pretraining_best, ranker_best=ranker_best,
                gate_best=gate_best, pretraining_early=pretraining_early,
                ranker_early=ranker_early, gate_early=gate_early,
                history=history, optimizer_steps=steps, context=context,
                device=resolved,
            )
            _atomic_torch_save(payload, resume_path)
            if interrupt_at == ("supervised", epoch):
                return _result(resume_path, payload, status="interrupted", new_steps=new_steps)
            if ranker_early.should_stop and gate_early.should_stop:
                break
        if not ranker_best or not gate_best:
            raise RuntimeError("V63 supervised training lacks strict best states")
        ranker.load_state_dict(ranker_best, strict=True)
        gate.load_state_dict(gate_best, strict=True)

    final_payload = _checkpoint_payload(
        kind="final", stage="complete", stage_epoch=len(history["supervised"]),
        ranker=ranker, gate=gate, pretraining_optimizer=pretraining_optimizer,
        ranker_optimizer=ranker_optimizer, gate_optimizer=gate_optimizer,
        pretraining_best=pretraining_best, ranker_best=ranker_best,
        gate_best=gate_best, pretraining_early=pretraining_early,
        ranker_early=ranker_early, gate_early=gate_early, history=history,
        optimizer_steps=steps, context=context, device=resolved,
    )
    _atomic_torch_save(final_payload, final_path)
    resume_path.unlink(missing_ok=True)
    return _result(final_path, final_payload, status="completed", new_steps=new_steps)


def _supervised_epoch_independent(
    *,
    ranker: MultiAssetPatchTransformer,
    gate: IndependentStateGate,
    ranker_optimizer: torch.optim.Optimizer,
    gate_optimizer: torch.optim.Optimizer,
    ranker_active: bool,
    gate_active: bool,
    data: FoldTrainingData,
    draws: list[SampleDraw],
    batch_size: int,
    device: torch.device,
    gradient_clip: float,
) -> tuple[dict[str, float], int, int, float, float]:
    """Train one still-active component after the other has early-stopped."""
    # The common epoch implementation deliberately requires both modules to be
    # active together.  A frozen component is evaluated with gradients disabled
    # while the other keeps its own optimizer and graph.
    ranker.train(ranker_active)
    gate.train(gate_active)
    ranker_parameters = [p for p in ranker.parameters() if p.requires_grad]
    gate_parameters = list(gate.parameters())
    sums = {name: 0.0 for name in ("ranking", "excess", "log_volatility", "ranker_core", "ranker_total", "gate")}
    observations = ranker_steps = gate_steps = 0
    max_ranker = max_gate = 0.0
    for _, x, labels in _batches(data, draws, batch_size):
        x = x.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.float32)
        with torch.set_grad_enabled(ranker_active):
            losses = ranking_excess_loss(
                ranker(x), labels, data.scale.excess_rms,
                tie_tolerance=1.0e-12, volatility_floor=1.0e-8,
                ranking_weight=1.0, excess_weight=1.0, volatility_weight=0.1,
            )
        if ranker_active:
            ranker_optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            gradient = nn.utils.clip_grad_norm_(ranker_parameters, gradient_clip)
            max_ranker = max(max_ranker, float(gradient.detach().cpu()))
            ranker_optimizer.step()
            ranker_steps += 1
        market = labels[..., 0].mean(dim=1) / data.scale.market_rms
        with torch.set_grad_enabled(gate_active):
            gate_loss = F.smooth_l1_loss(
                gate(derive_state_features(x.detach())), market, beta=1.0
            )
        if gate_active:
            gate_optimizer.zero_grad(set_to_none=True)
            gate_loss.backward()
            gradient = nn.utils.clip_grad_norm_(gate_parameters, gradient_clip)
            max_gate = max(max_gate, float(gradient.detach().cpu()))
            gate_optimizer.step()
            gate_steps += 1
        count = len(x)
        for name, tensor in (
            ("ranking", losses["ranking"]), ("excess", losses["excess"]),
            ("log_volatility", losses["log_volatility"]),
            ("ranker_core", losses["core"]), ("ranker_total", losses["total"]),
            ("gate", gate_loss),
        ):
            sums[name] += float(tensor.detach().cpu()) * count
        observations += count
    return ({name: value / observations for name, value in sums.items()}, ranker_steps, gate_steps, max_ranker, max_gate)


def verify_checkpoint(
    path: Path,
    *,
    blueprint: dict[str, Any],
    contract: dict[str, Any],
    context: dict[str, Any],
    device: str = "cpu",
) -> dict[str, Any]:
    resolved = configure_v63_runtime(device, seed=int(context["seed"]))
    ranker, gate = instantiate_models(blueprint, resolved)
    pretraining_optimizer = _optimizer(configure_pretraining_scope(ranker), contract)
    ranker_optimizer = _optimizer(configure_supervised_scope(ranker), contract)
    gate_optimizer = _optimizer(list(gate.parameters()), contract)
    payload = _load_checkpoint(
        path, expected_kind="final", ranker=ranker, gate=gate,
        pretraining_optimizer=pretraining_optimizer,
        ranker_optimizer=ranker_optimizer, gate_optimizer=gate_optimizer,
        context=context, device=resolved, restore_rng=False,
    )
    checks = {
        "complete_stage": payload["stage"] == "complete",
        "ranker_state_finite": all(
            not value.is_floating_point() or bool(torch.isfinite(value).all())
            for value in ranker.state_dict().values()
        ),
        "gate_state_finite": all(
            not value.is_floating_point() or bool(torch.isfinite(value).all())
            for value in gate.state_dict().values()
        ),
        "independent_optimizer_steps_nonzero": all(
            int(payload["optimizer_steps"][key]) > 0
            for key in ("pretraining", "ranker", "gate")
        ),
        "best_states_present": all(
            bool(payload[key])
            for key in ("pretraining_best_state", "ranker_best_state", "gate_best_state")
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "job_id": context["job_id"],
        "checkpoint_path": str(path),
        "checkpoint_file_sha256": file_sha256(path),
        "semantic_checkpoint_sha256": payload["semantic_checkpoint_sha256"],
        "ranker_state_sha256": semantic_state_sha256(payload["ranker_current_state"]),
        "gate_state_sha256": semantic_state_sha256(payload["gate_current_state"]),
        "optimizer_steps": dict(payload["optimizer_steps"]),
    }
