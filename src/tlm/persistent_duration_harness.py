from __future__ import annotations

from copy import deepcopy
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

from .core import (
    SyntheticAccessLedger,
    canonical_sha256,
    file_sha256,
    write_json_atomic,
    write_yaml_atomic,
)
from .persistent_duration_policy import (
    CASH,
    persistent_horizon_edges,
    stateful_persistent_actions,
)
from .persistent_multi_horizon_duration_model import (
    PersistentMultiHorizonDurationTransformer,
    explicit_duration_negative_log_likelihood,
    persistent_multi_task_loss,
)


EXPECTED_INPUT_NAMES = {
    "v74_specification",
    "v74_blueprint",
    "v74_audit",
    "v74_result",
    "v74_artifact_manifest",
    "v74_source_receipt",
}
FORBIDDEN_INPUT_SUFFIXES = {".parquet", ".pt", ".pth", ".ckpt"}


def _load_allowed_json(
    path: Path,
    allowed_paths: set[Path],
    ledger: SyntheticAccessLedger,
) -> dict[str, Any]:
    resolved = path.resolve()
    if resolved not in allowed_paths or path.suffix != ".json":
        raise ValueError(f"Unauthorized V75 metadata read: {path}")
    ledger.authorized_metadata_reads += 1
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _canonical_self_hash(value: dict[str, Any], field: str) -> bool:
    payload = dict(value)
    registered = payload.pop(field, None)
    return isinstance(registered, str) and registered == canonical_sha256(payload)


def _clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _state_dict_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _update_tensor_digest(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(b"tensor")
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
        return
    if isinstance(value, dict):
        digest.update(b"dict")
        for key in sorted(value, key=lambda item: str(item)):
            digest.update(str(key).encode("utf-8"))
            _update_tensor_digest(digest, value[key])
        return
    if isinstance(value, (list, tuple)):
        digest.update(b"sequence")
        for item in value:
            _update_tensor_digest(digest, item)
        return
    digest.update(repr(value).encode("utf-8"))


def _optimizer_sha256(optimizer: torch.optim.Optimizer) -> str:
    digest = hashlib.sha256()
    _update_tensor_digest(digest, optimizer.state_dict())
    return digest.hexdigest()


def _optimizer_step_count(optimizer: torch.optim.Optimizer) -> int:
    steps: list[int] = []
    for state in optimizer.state.values():
        step = state.get("step", 0)
        steps.append(int(step.item()) if isinstance(step, torch.Tensor) else int(step))
    return max(steps, default=0)


def _serialize_checkpoint(checkpoint: dict[str, Any]) -> bytes:
    buffer = BytesIO()
    torch.save(checkpoint, buffer)
    return buffer.getvalue()


def _load_checkpoint(
    checkpoint_bytes: bytes,
    *,
    expected_format: str,
    expected_architecture: dict[str, Any],
    expected_metadata: dict[str, Any],
) -> dict[str, Any]:
    value = torch.load(BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise ValueError("Synthetic checkpoint must contain a dictionary")
    if value.get("format_version") != expected_format:
        raise ValueError("Synthetic checkpoint format mismatch")
    if value.get("architecture") != expected_architecture:
        raise ValueError("Synthetic checkpoint architecture mismatch")
    if value.get("metadata") != expected_metadata:
        raise ValueError("Synthetic checkpoint metadata mismatch")
    return value


def _train_step(
    model: PersistentMultiHorizonDurationTransformer,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    return_targets: torch.Tensor,
    duration_days: torch.Tensor,
    duration_censored: torch.Tensor,
    *,
    round_trip_cost: float,
    objective: dict[str, Any],
    gradient_clip_norm: float,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(features, round_trip_cost=round_trip_cost)
    weights = objective["weights"]
    losses = persistent_multi_task_loss(
        output,
        return_targets,
        duration_days,
        duration_censored,
        degrees_of_freedom=float(model.degrees_of_freedom),
        return_nll_weight=float(weights["return_nll"]),
        ranking_weight=float(weights["pairwise_ranking"]),
        duration_weight=float(weights["duration_nll"]),
    )
    losses["total"].backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), gradient_clip_norm
    )
    gradients_finite = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    optimizer.step()
    return {
        "return_nll": float(losses["return_nll"].detach().cpu()),
        "ranking": float(losses["ranking"].detach().cpu()),
        "duration_nll": float(losses["duration_nll"].detach().cpu()),
        "total": float(losses["total"].detach().cpu()),
        "pair_count": int(losses["pair_count"].detach().cpu()),
        "gradient_norm": float(gradient_norm.detach().cpu()),
        "gradients_finite": gradients_finite,
    }


def _build_harness_spec(
    blueprint: dict[str, Any], harness: dict[str, Any]
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": "v75-persistent-duration-harness-spec/v1",
        "version": "v75",
        "candidate_family_id": blueprint["candidate_family_id"],
        "v74_specification_sha256": blueprint["specification_sha256"],
        "v74_blueprint_sha256": blueprint["blueprint_sha256"],
        "architecture": blueprint["architecture"],
        "objective": blueprint["objective"],
        "policy": blueprint["policy"],
        "training_contract": blueprint["training_contract"],
        "synthetic": harness["synthetic"],
        "optimizer": harness["optimizer"],
        "checkpoint": harness["checkpoint"],
        "policy_fixture": harness["policy_fixture"],
        "constraints": harness["constraints"],
        "source_receipt_files": harness["source_receipt_files"],
        "authorized_next_action": harness["authorized_next_action"],
    }
    value["harness_spec_sha256"] = canonical_sha256(value)
    return value


def _policy_fixture(
    policy: dict[str, Any], fixture: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    edges = np.asarray(
        [
            [0.003, 0.000, 0.000],
            [0.002, 0.004, 0.000],
            [0.001, 0.005, 0.000],
            [0.000, 0.002, 0.004],
            [0.000, -0.004, 0.000],
            [0.001, 0.000, 0.000],
            [0.003, 0.003, 0.000],
            [-0.001, 0.000, 0.000],
        ],
        dtype=np.float64,
    )
    horizons = np.asarray(fixture["horizons"], dtype=np.float64)
    gross_location = edges[:, :, None] * horizons[None, None, :]
    survival = np.ones_like(gross_location)
    observed_edges = persistent_horizon_edges(
        gross_location,
        survival,
        horizons=fixture["horizons"],
        horizon_weights=fixture["horizon_weights"],
    )
    policy_result = stateful_persistent_actions(
        observed_edges,
        base_cost=float(policy["base_cost_bps"]) / 10_000.0,
        risky_gross=float(policy["risky_gross"]),
        initial_action=CASH,
        tie_tolerance=float(fixture["tie_tolerance"]),
        final_liquidation=bool(policy["final_liquidation"]),
    )
    return edges, observed_edges, policy_result


def _execute_synthetic(
    blueprint: dict[str, Any],
    harness: dict[str, Any],
    harness_spec: dict[str, Any],
    *,
    seed: int,
    authorized_metadata_reads: int,
) -> dict[str, Any]:
    architecture = blueprint["architecture"]
    objective = blueprint["objective"]
    policy = blueprint["policy"]
    synthetic = harness["synthetic"]
    optimizer_config = harness["optimizer"]
    checkpoint_config = harness["checkpoint"]
    fixture = dict(harness["policy_fixture"])
    fixture["tie_tolerance"] = synthetic["tie_tolerance"]
    ledger = SyntheticAccessLedger(authorized_metadata_reads=authorized_metadata_reads)

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    prototype = PersistentMultiHorizonDurationTransformer(architecture)
    base_state = _clone_state_dict(prototype)
    parameter_count = sum(parameter.numel() for parameter in prototype.parameters())

    feature_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    target_generator = torch.Generator(device="cpu").manual_seed(seed + 2)
    batch_size = int(synthetic["batch_size"])
    features = torch.randn(
        batch_size,
        int(architecture["lookback_days"]),
        int(architecture["input_triplet_size"]),
        int(architecture["input_features"]),
        generator=feature_generator,
        dtype=torch.float32,
    )
    return_targets = (
        torch.randn(
            batch_size,
            int(architecture["input_triplet_size"]),
            len(architecture["output_horizons"]),
            generator=target_generator,
            dtype=torch.float32,
        )
        * 0.01
    )
    duration_days = torch.tensor([[1, 3, 7], [2, 5, 7]], dtype=torch.long)
    duration_censored = torch.tensor(
        [[False, False, True], [False, True, True]], dtype=torch.bool
    )
    ledger.synthetic_tensor_generations += 4

    model = PersistentMultiHorizonDurationTransformer(architecture)
    model.load_state_dict(base_state)
    model.eval()
    with torch.no_grad():
        reference = model(
            features, round_trip_cost=float(synthetic["round_trip_cost"])
        )
        permutation = torch.tensor(synthetic["asset_permutation"], dtype=torch.long)
        permuted = model(
            features[:, :, permutation, :],
            round_trip_cost=float(synthetic["round_trip_cost"]),
        )
        temporal = model.encode_temporal_patches(features)
        altered_features = features.clone()
        altered_features[:, -1, :, :] += 100.0
        altered_temporal = model.encode_temporal_patches(altered_features)
    ledger.synthetic_tensor_generations += 1

    cpu_model = PersistentMultiHorizonDurationTransformer(architecture)
    cpu_model.load_state_dict(base_state)
    cpu_optimizer = torch.optim.AdamW(
        cpu_model.parameters(),
        lr=float(optimizer_config["learning_rate"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    cpu_loss = _train_step(
        cpu_model,
        cpu_optimizer,
        features,
        return_targets,
        duration_days,
        duration_censored,
        round_trip_cost=float(synthetic["round_trip_cost"]),
        objective=objective,
        gradient_clip_norm=float(optimizer_config["gradient_clip_norm"]),
    )
    ledger.synthetic_optimizer_steps += 1

    optimizer_cycles = int(synthetic["optimizer_cycles"])
    if optimizer_cycles != 2:
        raise ValueError("V75 interrupted-resume fixture is frozen to two cycles")
    optimizer_kwargs = {
        "lr": float(optimizer_config["learning_rate"]),
        "weight_decay": float(optimizer_config["weight_decay"]),
    }
    torch.manual_seed(seed + 3)
    training_rng = torch.get_rng_state().clone()
    uninterrupted_model = PersistentMultiHorizonDurationTransformer(architecture)
    uninterrupted_model.load_state_dict(base_state)
    uninterrupted_optimizer = torch.optim.AdamW(
        uninterrupted_model.parameters(), **optimizer_kwargs
    )
    torch.set_rng_state(training_rng)
    uninterrupted_history = []
    for _ in range(optimizer_cycles):
        uninterrupted_history.append(
            _train_step(
                uninterrupted_model,
                uninterrupted_optimizer,
                features,
                return_targets,
                duration_days,
                duration_censored,
                round_trip_cost=float(synthetic["round_trip_cost"]),
                objective=objective,
                gradient_clip_norm=float(optimizer_config["gradient_clip_norm"]),
            )
        )
        ledger.synthetic_optimizer_steps += 1

    interrupted_model = PersistentMultiHorizonDurationTransformer(architecture)
    interrupted_model.load_state_dict(base_state)
    interrupted_optimizer = torch.optim.AdamW(
        interrupted_model.parameters(), **optimizer_kwargs
    )
    torch.set_rng_state(training_rng)
    first_interrupted_loss = _train_step(
        interrupted_model,
        interrupted_optimizer,
        features,
        return_targets,
        duration_days,
        duration_censored,
        round_trip_cost=float(synthetic["round_trip_cost"]),
        objective=objective,
        gradient_clip_norm=float(optimizer_config["gradient_clip_norm"]),
    )
    ledger.synthetic_optimizer_steps += 1
    checkpoint_metadata = {
        "candidate_family_id": blueprint["candidate_family_id"],
        "v74_blueprint_sha256": blueprint["blueprint_sha256"],
        "v75_harness_spec_sha256": harness_spec["harness_spec_sha256"],
        "seed": seed,
        "status": checkpoint_config["status"],
        "completed_optimizer_cycles": 1,
    }
    saved_rng = torch.get_rng_state().clone()
    checkpoint = {
        "format_version": checkpoint_config["format_version"],
        "architecture": architecture,
        "metadata": checkpoint_metadata,
        "model_state": _clone_state_dict(interrupted_model),
        "optimizer_state": deepcopy(interrupted_optimizer.state_dict()),
        "cpu_rng_state": saved_rng,
        "history": [first_interrupted_loss],
    }
    checkpoint_bytes = _serialize_checkpoint(checkpoint)
    ledger.synthetic_checkpoint_writes += 1
    loaded_checkpoint = _load_checkpoint(
        checkpoint_bytes,
        expected_format=checkpoint_config["format_version"],
        expected_architecture=architecture,
        expected_metadata=checkpoint_metadata,
    )
    ledger.synthetic_checkpoint_reads += 1
    resumed_model = PersistentMultiHorizonDurationTransformer(architecture)
    resumed_model.load_state_dict(loaded_checkpoint["model_state"])
    resumed_optimizer = torch.optim.AdamW(
        resumed_model.parameters(), **optimizer_kwargs
    )
    resumed_optimizer.load_state_dict(loaded_checkpoint["optimizer_state"])
    torch.set_rng_state(loaded_checkpoint["cpu_rng_state"])
    second_resumed_loss = _train_step(
        resumed_model,
        resumed_optimizer,
        features,
        return_targets,
        duration_days,
        duration_censored,
        round_trip_cost=float(synthetic["round_trip_cost"]),
        objective=objective,
        gradient_clip_norm=float(optimizer_config["gradient_clip_norm"]),
    )
    ledger.synthetic_optimizer_steps += 1

    zero_logits = torch.zeros(1, 2, int(architecture["maximum_duration_days"]))
    duration_fixture = torch.tensor([[3, 3]], dtype=torch.long)
    event_censor_fixture = torch.tensor([[False, True]], dtype=torch.bool)
    duration_fixture_loss = explicit_duration_negative_log_likelihood(
        zero_logits, duration_fixture, event_censor_fixture
    )
    duration_event_loss = explicit_duration_negative_log_likelihood(
        zero_logits[:, :1], duration_fixture[:, :1], event_censor_fixture[:, :1]
    )
    duration_censor_loss = explicit_duration_negative_log_likelihood(
        zero_logits[:, 1:], duration_fixture[:, 1:], event_censor_fixture[:, 1:]
    )
    ledger.synthetic_tensor_generations += 3

    expected_edges, observed_edges, policy_result = _policy_fixture(policy, fixture)
    ledger.synthetic_tensor_generations += 3

    mps_built = bool(torch.backends.mps.is_built())
    mps_available = bool(torch.backends.mps.is_available())
    mps_loss_finite = False
    mps_gradients_finite = False
    mps_optimizer_step_completed = False
    if mps_built and mps_available:
        mps_device = torch.device("mps")
        mps_model = PersistentMultiHorizonDurationTransformer(architecture).to(
            mps_device
        )
        mps_model.load_state_dict(base_state)
        mps_optimizer = torch.optim.AdamW(
            mps_model.parameters(), **optimizer_kwargs
        )
        mps_result = _train_step(
            mps_model,
            mps_optimizer,
            features.to(mps_device),
            return_targets.to(mps_device),
            duration_days.to(mps_device),
            duration_censored.to(mps_device),
            round_trip_cost=float(synthetic["round_trip_cost"]),
            objective=objective,
            gradient_clip_norm=float(optimizer_config["gradient_clip_norm"]),
        )
        ledger.synthetic_tensor_generations += 4
        ledger.synthetic_optimizer_steps += 1
        mps_loss_finite = all(
            math.isfinite(mps_result[name])
            for name in ("return_nll", "ranking", "duration_nll", "total")
        )
        mps_gradients_finite = bool(mps_result["gradients_finite"])
        mps_optimizer_step_completed = (
            _optimizer_step_count(mps_optimizer)
            == int(synthetic["mps_backward_steps"])
            == 1
        )
        del mps_result, mps_optimizer, mps_model
        torch.mps.empty_cache()

    expected_output_shapes = {
        "excess_location": [batch_size, 3, 3],
        "excess_scale": [batch_size, 3, 3],
        "market_location": [batch_size, 3],
        "market_scale": [batch_size, 3],
        "gross_location": [batch_size, 3, 3],
        "gross_scale": [batch_size, 3, 3],
        "net_location": [batch_size, 3, 3],
        "hazard_logits": [batch_size, 3, 7],
        "survival_probability": [batch_size, 3, 7],
        "horizon_survival_probability": [batch_size, 3, 3],
        "persistent_net_score": [batch_size, 3, 3],
    }
    output_shapes = {
        name: list(reference[name].shape) for name in expected_output_shapes
    }
    atol = float(synthetic["comparison_atol"])
    rtol = float(synthetic["comparison_rtol"])
    horizon_indexes = [int(value) - 1 for value in architecture["output_horizons"]]
    asset_keys = (
        "excess_location",
        "excess_scale",
        "gross_location",
        "gross_scale",
        "net_location",
        "hazard_logits",
        "survival_probability",
        "horizon_survival_probability",
        "persistent_net_score",
    )
    market_keys = ("market_location", "market_scale")
    finite_cpu_losses = all(
        math.isfinite(cpu_loss[name])
        for name in ("return_nll", "ranking", "duration_nll", "total")
    )
    loss_history_finite = all(
        math.isfinite(item[name])
        for item in uninterrupted_history
        + [first_interrupted_loss, second_resumed_loss]
        for name in ("return_nll", "ranking", "duration_nll", "total", "gradient_norm")
    )
    uninterrupted_state_hash = _state_dict_sha256(uninterrupted_model)
    resumed_state_hash = _state_dict_sha256(resumed_model)
    uninterrupted_optimizer_hash = _optimizer_sha256(uninterrupted_optimizer)
    resumed_optimizer_hash = _optimizer_sha256(resumed_optimizer)

    expected_actions = list(fixture["expected_actions"])
    expected_selected = [int(value) for value in fixture["expected_selected_assets"]]
    expected_turnover = [
        float(value) for value in fixture["expected_transition_turnover"]
    ]
    expected_total_turnover = float(fixture["expected_total_turnover"])
    base_cost = float(policy["base_cost_bps"]) / 10_000.0
    policy_checks = (
        np.allclose(observed_edges, expected_edges, atol=1e-15, rtol=0.0)
        and policy_result["actions"] == expected_actions
        and policy_result["selected_assets"] == expected_selected
        and np.allclose(policy_result["turnover"], expected_turnover)
        and math.isclose(
            float(policy_result["final_liquidation_turnover"]),
            float(fixture["expected_final_liquidation_turnover"]),
        )
        and math.isclose(float(policy_result["total_turnover"]), expected_total_turnover)
        and math.isclose(
            float(policy_result["total_transaction_cost"]),
            base_cost * expected_total_turnover,
        )
    )
    constraints = harness["constraints"]
    constraints_exact = (
        constraints["synthetic_only"] is True
        and all(
            constraints[name] is False
            for name in (
                "parquet_or_real_data_access_allowed",
                "prior_or_real_checkpoint_access_allowed",
                "real_training_or_inference_allowed",
                "real_prediction_or_position_allowed",
                "performance_metric_or_pnl_allowed",
                "outcome_source_read_allowed",
                "target_asset_access_allowed",
                "architecture_objective_policy_or_hyperparameter_change_allowed",
                "v76_implementation_allowed",
            )
        )
    )
    checks = {
        "exact_1083155_parameter_count_and_output_shapes": parameter_count
        == int(architecture["expected_parameter_count"])
        == 1_083_155
        and output_shapes == expected_output_shapes,
        "causal_temporal_prefix_invariance": torch.allclose(
            temporal[:, :, :-1], altered_temporal[:, :, :-1], atol=atol, rtol=rtol
        ),
        "asset_permutation_equivariance_and_market_invariance": all(
            torch.allclose(
                permuted[name], reference[name][:, permutation], atol=atol, rtol=rtol
            )
            for name in asset_keys
        )
        and all(
            torch.allclose(permuted[name], reference[name], atol=atol, rtol=rtol)
            for name in market_keys
        ),
        "centered_excess_positive_scales_and_monotone_survival": torch.allclose(
            reference["excess_location"].sum(dim=1),
            torch.zeros_like(reference["market_location"]),
            atol=atol,
            rtol=0.0,
        )
        and bool((reference["excess_scale"] > 0).all())
        and bool((reference["market_scale"] > 0).all())
        and bool((reference["gross_scale"] > 0).all())
        and bool(
            (
                reference["survival_probability"][..., 1:]
                <= reference["survival_probability"][..., :-1] + atol
            ).all()
        )
        and torch.allclose(
            reference["horizon_survival_probability"],
            reference["survival_probability"][..., horizon_indexes],
            atol=atol,
            rtol=rtol,
        ),
        "finite_student_t_return_pairwise_ranking_and_duration_losses": finite_cpu_losses
        and int(cpu_loss["pair_count"]) > 0
        and cpu_loss["gradients_finite"]
        and loss_history_finite,
        "event_and_right_censor_duration_likelihood": bool(
            torch.isfinite(duration_fixture_loss)
        )
        and math.isclose(
            float(duration_event_loss), 3.0 * math.log(2.0), abs_tol=1e-6
        )
        and math.isclose(
            float(duration_censor_loss), 3.0 * math.log(2.0), abs_tol=1e-6
        ),
        "finite_cpu_and_mps_joint_backward": finite_cpu_losses
        and cpu_loss["gradients_finite"]
        and mps_built
        and mps_available
        and mps_loss_finite
        and mps_gradients_finite
        and mps_optimizer_step_completed,
        "exact_entry_hold_exit_switch_costs_and_incumbent_cash_lexical_ties": policy_checks,
        "synthetic_checkpoint_roundtrip_and_interrupted_resume_equivalence": uninterrupted_state_hash
        == resumed_state_hash
        and uninterrupted_optimizer_hash == resumed_optimizer_hash
        and uninterrupted_history == [first_interrupted_loss, second_resumed_loss]
        and _optimizer_step_count(uninterrupted_optimizer)
        == _optimizer_step_count(resumed_optimizer)
        == optimizer_cycles
        and torch.equal(loaded_checkpoint["cpu_rng_state"], saved_rng),
        "zero_real_data_prior_checkpoint_outcome_metric_pnl_or_target_access": constraints_exact
        and ledger.forbidden_operations_are_zero(),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    operation_ledger = ledger.to_dict()
    smoke = {
        "parameter_count": parameter_count,
        "output_shapes": output_shapes,
        "cpu_joint_backward_finite": finite_cpu_losses
        and bool(cpu_loss["gradients_finite"]),
        "mps_built": mps_built,
        "mps_available": mps_available,
        "mps_joint_backward_finite": mps_loss_finite and mps_gradients_finite,
        "optimizer_steps_executed": operation_ledger["synthetic_optimizer_steps"],
        "resume_equivalent": checks[
            "synthetic_checkpoint_roundtrip_and_interrupted_resume_equivalence"
        ],
        "policy_actions": policy_result["actions"],
        "policy_selected_assets": policy_result["selected_assets"],
        "policy_transition_turnover": policy_result["turnover"],
        "policy_final_liquidation_turnover": policy_result[
            "final_liquidation_turnover"
        ],
        "policy_total_turnover": policy_result["total_turnover"],
        "policy_total_transaction_cost": policy_result["total_transaction_cost"],
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "devices": ["cpu", "mps"],
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
    }
    replay_payload = {
        "checks": checks,
        "smoke": smoke,
        "operation_ledger": operation_ledger,
        "checkpoint_metadata": checkpoint_metadata,
        "model_state_sha256": resumed_state_hash,
        "optimizer_state_sha256": resumed_optimizer_hash,
        "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
    }
    return {
        "checks": checks,
        "smoke": smoke,
        "operation_ledger": operation_ledger,
        "checkpoint_metadata": checkpoint_metadata,
        "checkpoint_bytes": checkpoint_bytes,
        "replay_payload": replay_payload,
    }


def run_persistent_duration_harness(config: dict[str, Any]) -> dict[str, Any]:
    harness = config["persistent_duration_harness"]
    root = Path(harness.get("project_root", ".")).resolve()
    configured_output = Path(config["output_dir"])
    output = configured_output if configured_output.is_absolute() else root / configured_output
    output.mkdir(parents=True, exist_ok=True)

    paths = {name: root / relative for name, relative in harness["inputs"].items()}
    if set(paths) != EXPECTED_INPUT_NAMES:
        raise ValueError("V75 runtime input allowlist drift")
    if any(path.suffix in FORBIDDEN_INPUT_SUFFIXES for path in paths.values()):
        raise ValueError("V75 input allowlist contains data or checkpoint")
    observed_before = {name: file_sha256(path) for name, path in paths.items()}
    if observed_before != harness["expected_input_sha256"]:
        failed = sorted(
            name
            for name, digest in observed_before.items()
            if digest != harness["expected_input_sha256"].get(name)
        )
        raise ValueError(f"V75 input receipt mismatch: {failed}")

    metadata_ledger = SyntheticAccessLedger()
    allowed_paths = {path.resolve() for path in paths.values()}
    loaded = {
        name: _load_allowed_json(path, allowed_paths, metadata_ledger)
        for name, path in paths.items()
    }
    specification = loaded["v74_specification"]
    blueprint = loaded["v74_blueprint"]
    v74_audit = loaded["v74_audit"]
    v74_result = loaded["v74_result"]
    v74_manifest = loaded["v74_artifact_manifest"]
    v74_source = loaded["v74_source_receipt"]
    canonical_expected = harness["expected_canonical_sha256"]
    metadata_contract_passes = (
        _canonical_self_hash(specification, "specification_sha256")
        and specification["specification_sha256"]
        == canonical_expected["specification"]
        and _canonical_self_hash(blueprint, "blueprint_sha256")
        and blueprint["blueprint_sha256"] == canonical_expected["blueprint"]
        and _canonical_self_hash(v74_result, "result_sha256")
        and v74_result["result_sha256"] == canonical_expected["result"]
        and _canonical_self_hash(v74_manifest, "artifact_manifest_sha256")
        and v74_manifest["artifact_manifest_sha256"]
        == canonical_expected["artifact_manifest"]
        and _canonical_self_hash(v74_source, "source_receipt_sha256")
        and v74_source["source_receipt_sha256"]
        == canonical_expected["source_receipt"]
        and v74_audit.get("passed") is True
        and v74_result.get("decision")
        == "authorize_v75_synthetic_persistent_duration_harness_only"
        and blueprint["candidate_family_id"] == harness["candidate_family_id"]
        and blueprint["specification_sha256"]
        == specification["specification_sha256"]
    )
    if not metadata_contract_passes:
        raise ValueError("V75 V74 authorization packet is not canonical")

    source_files = [str(value) for value in harness["source_receipt_files"]]
    if not source_files or len(source_files) != len(set(source_files)):
        raise ValueError("V75 source receipt paths must be unique")
    source_hashes: dict[str, str] = {}
    for relative in source_files:
        source_path = (root / relative).resolve()
        try:
            source_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Source receipt path escapes project root: {relative}") from exc
        if not source_path.is_file():
            raise ValueError(f"Missing V75 source receipt file: {relative}")
        source_hashes[relative] = file_sha256(source_path)
    source_receipt: dict[str, Any] = {
        "schema_version": "v75-source-receipt/v1",
        "files": source_hashes,
        "bundle_sha256": canonical_sha256(source_hashes),
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "devices": ["cpu", "mps"],
        },
    }
    source_receipt["source_receipt_sha256"] = canonical_sha256(source_receipt)

    harness_spec = _build_harness_spec(blueprint, harness)
    first = _execute_synthetic(
        blueprint,
        harness,
        harness_spec,
        seed=int(config["seed"]),
        authorized_metadata_reads=metadata_ledger.authorized_metadata_reads,
    )
    replay = _execute_synthetic(
        blueprint,
        harness,
        harness_spec,
        seed=int(config["seed"]),
        authorized_metadata_reads=metadata_ledger.authorized_metadata_reads,
    )
    byte_identical_replay = (
        first["checkpoint_bytes"] == replay["checkpoint_bytes"]
        and canonical_sha256(first["replay_payload"])
        == canonical_sha256(replay["replay_payload"])
    )
    checks = {
        "all_v74_input_hashes_match": observed_before
        == harness["expected_input_sha256"],
        "input_allowlist_is_exactly_six_v74_metadata_artifacts": set(paths)
        == EXPECTED_INPUT_NAMES,
        "v74_authorization_packet_is_canonical": metadata_contract_passes,
        "source_receipt_is_complete": len(source_hashes) == len(source_files)
        and all(len(value) == 64 for value in source_hashes.values()),
        **first["checks"],
        "byte_identical_replay": byte_identical_replay,
        "input_hashes_still_match_after_harness": all(
            file_sha256(paths[name]) == digest
            for name, digest in observed_before.items()
        ),
        "only_v76_dataset_phase_is_authorized": harness["authorized_next_action"]
        == "authorize_v76_non_target_persistent_duration_dataset_only",
    }
    checks = {name: bool(value) for name, value in checks.items()}
    audit = {
        "schema_version": "v75-persistent-duration-audit/v1",
        "passed": all(checks.values()),
        "checks": checks,
        "operation_ledger": first["operation_ledger"],
    }
    decision = (
        harness["authorized_next_action"]
        if audit["passed"]
        else "keep_v76_and_later_unauthorized"
    )
    input_receipt = {
        name: {
            "path": str(paths[name].relative_to(root)),
            "sha256": observed_before[name],
        }
        for name in sorted(paths)
    }
    replay_receipt: dict[str, Any] = {
        "schema_version": "v75-internal-replay-receipt/v1",
        "core_execution_sha256": canonical_sha256(first["replay_payload"]),
        "checkpoint_sha256": hashlib.sha256(first["checkpoint_bytes"]).hexdigest(),
        "byte_identical": byte_identical_replay,
    }
    replay_receipt["replay_receipt_sha256"] = canonical_sha256(replay_receipt)
    result: dict[str, Any] = {
        "schema_version": "v75-persistent-duration-result/v1",
        "version": "v75",
        "family_id": blueprint["candidate_family_id"],
        "decision": decision,
        "harness_spec_sha256": harness_spec["harness_spec_sha256"],
        "input_hash_receipt": input_receipt,
        "source_receipt_sha256": source_receipt["source_receipt_sha256"],
        "smoke": first["smoke"],
        "operation_ledger": first["operation_ledger"],
        "replay_receipt": replay_receipt,
        "audit": audit,
    }
    result["result_sha256"] = canonical_sha256(result)
    report = "\n".join(
        [
            "# V75 Synthetic Persistent-Duration Harness",
            "",
            f"Decision: **{decision}**",
            "",
            f"Harness SHA-256: `{harness_spec['harness_spec_sha256']}`",
            f"Parameters: **{first['smoke']['parameter_count']:,}**",
            "",
            "The exact V74 architecture passed shape, causal-prefix, asset-",
            "permutation, Student-t, duration event/censor, CPU/MPS backward,",
            "stateful transition-cost, checkpoint-resume, and deterministic replay",
            "checks on synthetic tensors only.",
            "",
            "No Parquet, real label, prior checkpoint, prediction, position,",
            "performance/PnL, outcome source, or BTC/ETH/SOL data was opened.",
            "V76 is dataset-construction only and has not been implemented.",
            "",
        ]
    )

    (output / "synthetic_checkpoint.pt").write_bytes(first["checkpoint_bytes"])
    write_yaml_atomic(output / "resolved_config.yaml", config)
    write_json_atomic(output / "input_hash_receipt.json", input_receipt)
    write_json_atomic(output / "source_receipt.json", source_receipt)
    write_json_atomic(output / "harness_spec.json", harness_spec)
    write_json_atomic(output / "result.json", result)
    write_json_atomic(output / "audit.json", audit)
    write_json_atomic(output / "replay_receipt.json", replay_receipt)
    (output / "report.md").write_text(report, encoding="utf-8")
    manifest_names = (
        "audit.json",
        "harness_spec.json",
        "input_hash_receipt.json",
        "replay_receipt.json",
        "report.md",
        "resolved_config.yaml",
        "result.json",
        "source_receipt.json",
        "synthetic_checkpoint.pt",
    )
    manifest: dict[str, Any] = {
        "schema_version": "v75-artifact-manifest/v1",
        "files": {name: file_sha256(output / name) for name in manifest_names},
    }
    manifest["artifact_manifest_sha256"] = canonical_sha256(manifest)
    write_json_atomic(output / "artifact_manifest.json", manifest)

    if not audit["passed"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"V75 synthetic harness failed: {failed}")
    return result
