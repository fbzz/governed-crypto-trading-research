from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import yaml

from .core import (
    SyntheticAccessLedger,
    canonical_sha256,
    file_sha256,
    persistent_portfolio_returns,
    write_json_atomic,
    write_yaml_atomic,
)
from .state_conditioned_multi_horizon_model import (
    StateConditionedMultiHorizonTransformer,
    load_state_conditioned_checkpoint,
    multi_horizon_quantile_loss,
    save_state_conditioned_checkpoint,
    state_conditioned_weekly_policy,
)
from .state_conditioned_multi_horizon_spec import analytic_parameter_count


def _load_json(path: Path, ledger: SyntheticAccessLedger) -> dict[str, Any]:
    ledger.authorized_metadata_reads += 1
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _state_dict_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _optimizer_step_count(optimizer: torch.optim.Optimizer) -> int:
    steps = []
    for state in optimizer.state.values():
        step = state.get("step", 0)
        steps.append(int(step.item()) if isinstance(step, torch.Tensor) else int(step))
    return max(steps, default=0)


def _train_step(
    model: StateConditionedMultiHorizonTransformer,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    tie_tolerance: float,
    gradient_clip_norm: float,
) -> tuple[float, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = multi_horizon_quantile_loss(
        model(features), targets, tie_tolerance=tie_tolerance
    )
    losses["total"].backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), gradient_clip_norm
    )
    optimizer.step()
    return float(losses["total"].detach()), float(gradient_norm)


def _build_harness_spec(
    blueprint: dict[str, Any],
    harness: dict[str, Any],
) -> dict[str, Any]:
    spec = {
        "version": "v56",
        "candidate_family_id": blueprint["candidate_family_id"],
        "v55_blueprint_sha256": blueprint["blueprint_sha256"],
        "architecture": blueprint["architecture"],
        "objective": blueprint["objective"],
        "policy": blueprint["policy"],
        "synthetic": harness["synthetic"],
        "mechanics": harness["mechanics"],
        "checkpoint": harness["checkpoint"],
        "source_receipt_files": harness["source_receipt"]["files"],
        "constraints": harness["constraints"],
        "authorized_next_action": harness["authorized_next_action"],
    }
    spec["harness_spec_sha256"] = canonical_sha256(spec)
    return spec


def _main_policy_fixture(policy: dict[str, Any], days: int) -> dict[str, np.ndarray]:
    forecasts = np.full((days, 3), [-0.01, -0.02, -0.03], dtype=np.float64)
    forecasts[0] = [0.006, 0.001, 0.000]
    forecasts[7] = [0.000, 0.009, 0.001]
    forecasts[14] = [0.000, 0.008, 0.012]
    return state_conditioned_weekly_policy(
        forecasts,
        np.ones_like(forecasts, dtype=bool),
        risky_weight=float(policy["risky_gross"]),
        base_cost=float(policy["base_cost_bps"]) / 10_000.0,
        decision_interval=7,
    )


def run_state_conditioned_multi_horizon_harness(
    config: dict[str, Any],
) -> dict[str, Any]:
    harness = config["state_conditioned_multi_horizon_harness"]
    root = Path(harness.get("project_root", ".")).resolve()
    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    ledger = SyntheticAccessLedger()

    paths = {
        name: root / relative for name, relative in harness["inputs"].items()
    }
    if set(paths) != {"v55_result", "v55_blueprint", "v55_audit"}:
        raise ValueError("V56 runtime input allowlist drift")
    input_checks = {
        name: path.is_file()
        and file_sha256(path) == harness["expected_input_sha256"][name]
        for name, path in paths.items()
    }
    if not all(input_checks.values()):
        failed = sorted(name for name, passed in input_checks.items() if not passed)
        raise ValueError(f"V56 input receipt mismatch: {failed}")
    v55_result = _load_json(paths["v55_result"], ledger)
    blueprint = _load_json(paths["v55_blueprint"], ledger)
    v55_audit = _load_json(paths["v55_audit"], ledger)
    blueprint_without_hash = dict(blueprint)
    registered_blueprint_hash = blueprint_without_hash.pop("blueprint_sha256", None)
    blueprint_hash_valid = (
        registered_blueprint_hash
        == harness["expected_blueprint_sha256"]
        == canonical_sha256(blueprint_without_hash)
    )
    harness_spec = _build_harness_spec(blueprint, harness)
    source_files = [str(path) for path in harness["source_receipt"]["files"]]
    if len(source_files) != len(set(source_files)) or not source_files:
        raise ValueError("V56 source receipt must contain unique source paths")
    source_hashes = {}
    for relative in source_files:
        source_path = (root / relative).resolve()
        try:
            source_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Source receipt path escapes project root: {relative}") from exc
        if not source_path.is_file():
            raise ValueError(f"Missing V56 source receipt file: {relative}")
        source_hashes[relative] = file_sha256(source_path)
    source_receipt = {
        "files": source_hashes,
        "bundle_sha256": canonical_sha256(source_hashes),
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "deterministic_algorithms": True,
        },
    }

    architecture = blueprint["architecture"]
    objective = blueprint["objective"]
    policy = blueprint["policy"]
    training = blueprint["training"]
    synthetic = harness["synthetic"]
    mechanics = harness["mechanics"]
    seed = int(config["seed"])
    if synthetic["device"] != "cpu":
        raise ValueError("V56 synthetic harness is frozen to deterministic CPU")
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    prototype = StateConditionedMultiHorizonTransformer(architecture)
    base_state = _clone_state_dict(prototype)
    parameter_count = sum(parameter.numel() for parameter in prototype.parameters())

    feature_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    target_generator = torch.Generator(device="cpu").manual_seed(seed + 2)
    features = torch.randn(
        int(synthetic["batch_size"]), 256, 3, int(synthetic["input_features"]),
        generator=feature_generator, dtype=torch.float32,
    )
    targets = torch.randn(
        int(synthetic["batch_size"]), 3, 3,
        generator=target_generator, dtype=torch.float32,
    ) * 0.01
    ledger.synthetic_tensor_generations += 2
    torch.manual_seed(seed + 3)
    start_rng = torch.get_rng_state().clone()

    full_model = StateConditionedMultiHorizonTransformer(architecture)
    full_model.load_state_dict(base_state)
    full_optimizer = torch.optim.AdamW(
        full_model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    torch.set_rng_state(start_rng)
    full_history: list[float] = []
    full_gradient_norms: list[float] = []
    for _ in range(int(synthetic["optimizer_steps"])):
        loss, gradient_norm = _train_step(
            full_model,
            full_optimizer,
            features,
            targets,
            tie_tolerance=float(mechanics["ranknet_tie_tolerance"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
        )
        full_history.append(loss)
        full_gradient_norms.append(gradient_norm)

    interrupted_model = StateConditionedMultiHorizonTransformer(architecture)
    interrupted_model.load_state_dict(base_state)
    interrupted_optimizer = torch.optim.AdamW(
        interrupted_model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    torch.set_rng_state(start_rng)
    first_loss, first_gradient_norm = _train_step(
        interrupted_model,
        interrupted_optimizer,
        features,
        targets,
        tie_tolerance=float(mechanics["ranknet_tie_tolerance"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
    )
    ledger.synthetic_optimizer_steps += 1
    checkpoint_path = output / "synthetic_checkpoint.pt"
    checkpoint_metadata = {
        "candidate_family_id": blueprint["candidate_family_id"],
        "v55_blueprint_sha256": blueprint["blueprint_sha256"],
        "v56_harness_spec_sha256": harness_spec["harness_spec_sha256"],
        "initialization_seed": seed,
        "job_key": "synthetic/v56/fold_1/seed_20260714",
        "status": harness["checkpoint"]["status"],
    }
    early_stopping_state = {
        "best_validation_total_loss": first_loss,
        "best_epoch": 1,
        "non_improvement_count": 0,
        "patience": int(training["early_stopping_patience"]),
        "minimum_delta": 0.0,
    }
    saved_rng = torch.get_rng_state().clone()
    save_state_conditioned_checkpoint(
        checkpoint_path,
        {
            "model_state": _clone_state_dict(interrupted_model),
            "best_model_state": _clone_state_dict(interrupted_model),
            "optimizer_state": deepcopy(interrupted_optimizer.state_dict()),
            "cpu_rng_state": saved_rng,
            "mps_rng_state": None,
            "early_stopping_state": early_stopping_state,
            "history": [first_loss],
            "metadata": checkpoint_metadata,
            "architecture": architecture,
        },
        format_version=harness["checkpoint"]["format_version"],
    )
    ledger.synthetic_checkpoint_writes += 1
    checkpoint = load_state_conditioned_checkpoint(
        checkpoint_path,
        expected_format_version=harness["checkpoint"]["format_version"],
        expected_architecture=architecture,
        expected_metadata=checkpoint_metadata,
    )
    ledger.synthetic_checkpoint_reads += 1
    resumed_model = StateConditionedMultiHorizonTransformer(architecture)
    resumed_model.load_state_dict(checkpoint["model_state"])
    resumed_optimizer = torch.optim.AdamW(
        resumed_model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    resumed_optimizer.load_state_dict(checkpoint["optimizer_state"])
    torch.set_rng_state(checkpoint["cpu_rng_state"])
    second_loss, second_gradient_norm = _train_step(
        resumed_model,
        resumed_optimizer,
        features,
        targets,
        tie_tolerance=float(mechanics["ranknet_tie_tolerance"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
    )
    ledger.synthetic_optimizer_steps += 1

    resumed_model.eval()
    with torch.no_grad():
        reference = resumed_model(features)
        permutation = torch.tensor([2, 0, 1])
        permuted = resumed_model(features[:, :, permutation, :])
        temporal = resumed_model.encode_temporal_patches(features)
        altered = features.clone()
        altered[:, 200:, :, :] += 100.0
        altered_temporal = resumed_model.encode_temporal_patches(altered)
    early_patch_count = (200 - resumed_model.patch_length) // resumed_model.patch_stride + 1

    loss_fixture = multi_horizon_quantile_loss(
        reference,
        targets,
        tie_tolerance=float(mechanics["ranknet_tie_tolerance"]),
    )
    ordered = torch.zeros(1, 3, 3, 3)
    ordered[..., 0] = -0.01
    ordered[..., 2] = 0.01
    crossed = ordered.clone()
    crossed[..., 0] = 0.02
    crossed[..., 2] = -0.02
    zero_targets = torch.zeros(1, 3, 3)
    ordered_loss = multi_horizon_quantile_loss(ordered, zero_targets)
    crossed_loss = multi_horizon_quantile_loss(crossed, zero_targets)

    main_policy = _main_policy_fixture(policy, int(synthetic["policy_days"]))
    positions = main_policy["positions"]
    exit_forecasts = np.full((8, 3), [-0.01, -0.02, -0.03])
    exit_forecasts[0] = [0.006, 0.001, 0.000]
    exit_policy = state_conditioned_weekly_policy(
        exit_forecasts,
        np.ones_like(exit_forecasts, dtype=bool),
        risky_weight=float(policy["risky_gross"]),
        base_cost=float(policy["base_cost_bps"]) / 10_000.0,
    )
    missing_forecasts = np.full((8, 3), [0.006, 0.005, 0.004])
    missing_eligible = np.ones_like(missing_forecasts, dtype=bool)
    missing_eligible[3, 0] = False
    missing_forecasts[3, 1] = 0.02
    missing_policy = state_conditioned_weekly_policy(
        missing_forecasts,
        missing_eligible,
        risky_weight=float(policy["risky_gross"]),
        base_cost=float(policy["base_cost_bps"]) / 10_000.0,
    )
    entry_tie = state_conditioned_weekly_policy(
        np.array([[0.001, 0.0, 0.0]]),
        np.ones((1, 3), dtype=bool),
        risky_weight=float(policy["risky_gross"]),
        base_cost=float(policy["base_cost_bps"]) / 10_000.0,
    )
    lexical_tie = state_conditioned_weekly_policy(
        np.array([[0.006, 0.006, 0.0]]),
        np.ones((1, 3), dtype=bool),
        risky_weight=float(policy["risky_gross"]),
        base_cost=float(policy["base_cost_bps"]) / 10_000.0,
    )
    ledger.synthetic_tensor_generations += 6

    return_generator = np.random.default_rng(seed + 4)
    actual_returns = return_generator.normal(0.0, 0.01, size=positions.shape)
    accounting = {
        str(cost_bps): persistent_portfolio_returns(
            positions, actual_returns, float(cost_bps)
        )
        for cost_bps in policy["reporting_cost_bps"]
    }
    ledger.synthetic_tensor_generations += 1
    base_accounting = accounting[str(policy["base_cost_bps"])]
    expected_turnover = np.zeros(len(positions))
    expected_turnover[0] = 1.0 / 3.0
    expected_turnover[7] = 2.0 / 3.0
    expected_turnover[14] = 1.0

    full_state_hash = _state_dict_sha256(full_model)
    resumed_state_hash = _state_dict_sha256(resumed_model)
    input_hashes_after = {name: file_sha256(path) for name, path in paths.items()}
    constraints_all_false = not any(bool(value) for value in harness["constraints"].values())
    checks = {
        "all_v55_input_hashes_match": all(input_checks.values()),
        "input_hashes_still_match_after_harness": all(
            input_hashes_after[name] == harness["expected_input_sha256"][name]
            for name in paths
        ),
        "input_allowlist_contains_only_v55_metadata": set(paths)
        == {"v55_result", "v55_blueprint", "v55_audit"},
        "v55_authorizes_v56": v55_audit.get("passed") is True
        and v55_result.get("decision")
        == "authorize_v56_synthetic_state_policy_harness_only",
        "v55_blueprint_hash_is_canonical": blueprint_hash_valid,
        "source_receipt_is_complete": len(source_hashes) == len(source_files)
        and all(len(value) == 64 for value in source_hashes.values()),
        "deterministic_cpu_runtime_is_active": synthetic["device"] == "cpu"
        and torch.are_deterministic_algorithms_enabled(),
        "parameter_count_matches_frozen_value": parameter_count
        == analytic_parameter_count(architecture)
        == int(architecture["expected_parameter_count"])
        == 465_513,
        "model_output_shape_and_mapping_are_exact": tuple(reference.shape)
        == (int(synthetic["batch_size"]), 3, 3, 3),
        "asset_permutation_equivariance_passes": torch.allclose(
            permuted, reference[:, permutation], atol=1e-5, rtol=1e-5
        ),
        "causal_temporal_prefix_is_invariant": torch.allclose(
            temporal[:, :, :early_patch_count],
            altered_temporal[:, :, :early_patch_count],
            atol=1e-5,
            rtol=1e-5,
        ),
        "pinball_ranknet_crossing_losses_are_finite": all(
            bool(torch.isfinite(loss_fixture[name]))
            for name in ("pinball", "ranking", "crossing", "total")
        )
        and int(loss_fixture["pair_count"]) > 0,
        "quantile_ordering_diagnostic_is_exact": float(ordered_loss["crossing"]) == 0.0
        and float(crossed_loss["crossing"]) > 0.0,
        "all_model_parameters_receive_finite_gradients": all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in resumed_model.parameters()
        ),
        "gradient_norms_and_histories_are_finite": all(
            math.isfinite(value)
            for value in full_history
            + full_gradient_norms
            + [first_loss, first_gradient_norm, second_loss, second_gradient_norm]
        ),
        "interrupted_resume_matches_uninterrupted_training": full_state_hash
        == resumed_state_hash
        and full_history == [first_loss, second_loss]
        and _optimizer_step_count(full_optimizer)
        == _optimizer_step_count(resumed_optimizer)
        == int(synthetic["optimizer_steps"]),
        "checkpoint_contains_exact_resume_contract": checkpoint["history"]
        == [first_loss]
        and checkpoint["early_stopping_state"] == early_stopping_state
        and torch.equal(checkpoint["cpu_rng_state"], saved_rng)
        and checkpoint["mps_rng_state"] is None,
        "checkpoint_model_roundtrip_is_exact": all(
            torch.equal(interrupted_model.state_dict()[name], tensor)
            for name, tensor in checkpoint["model_state"].items()
        ),
        "seven_eligible_date_clock_is_exact": np.flatnonzero(
            main_policy["decision_mask"]
        ).tolist()
        == [0, 7, 14],
        "state_conditioned_entry_hold_switch_is_exact": np.argmax(positions[0]) == 0
        and np.argmax(positions[6]) == 0
        and np.argmax(positions[7]) == 1
        and np.argmax(positions[13]) == 1
        and np.argmax(positions[14]) == 2,
        "scheduled_exit_to_cash_is_exact": exit_policy["positions"][0].sum() > 0
        and exit_policy["positions"][7].sum() == 0,
        "missing_incumbent_forces_cash_without_replacement": missing_policy[
            "forced_cash_mask"
        ][3]
        and missing_policy["positions"][3].sum() == 0
        and missing_policy["positions"][4].sum() == 0,
        "tie_priority_is_current_cash_then_lexical": entry_tie["positions"].sum() == 0
        and int(np.argmax(lexical_tie["positions"][0])) == 0,
        "turnover_and_final_liquidation_are_exact": np.allclose(
            base_accounting["turnover"], expected_turnover
        )
        and math.isclose(float(base_accounting["total_turnover"]), 2.0),
        "positions_are_frozen_across_reporting_costs": all(
            np.allclose(values["turnover"], base_accounting["turnover"])
            and np.allclose(values["gross_return"], base_accounting["gross_return"])
            for values in accounting.values()
        ),
        "cost_accounting_identities_pass": all(
            np.allclose(values["net_return"], values["gross_return"] - values["cost"])
            and np.allclose(
                values["cost"], values["turnover"] * (float(cost) / 10_000.0)
            )
            for cost, values in accounting.items()
        ),
        "configuration_forbids_every_real_operation": constraints_all_false,
        "real_and_target_operations_remain_zero": ledger.forbidden_operations_are_zero(),
        "exactly_two_synthetic_optimizer_steps": ledger.synthetic_optimizer_steps
        == int(synthetic["optimizer_steps"]),
        "only_v57_dataset_build_is_authorized": harness["authorized_next_action"]
        == "authorize_v57_non_target_multi_horizon_dataset_build_only",
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    audit = {"passed": all(checks.values()), "checks": checks}
    decision = (
        harness["authorized_next_action"]
        if audit["passed"]
        else "keep_v57_and_later_unauthorized"
    )
    smoke = {
        "parameter_count": parameter_count,
        "output_shape": list(reference.shape),
        "optimizer_steps": ledger.synthetic_optimizer_steps,
        "resume_equivalent": full_state_hash == resumed_state_hash,
        "decision_indexes": np.flatnonzero(main_policy["decision_mask"]).tolist(),
        "base_total_turnover": float(base_accounting["total_turnover"]),
        "operation_ledger": ledger.to_dict(),
    }
    input_receipt = {
        name: {
            "path": str(paths[name].relative_to(root)),
            "sha256": input_hashes_after[name],
        }
        for name in sorted(paths)
    }
    report = "\n".join(
        [
            "# V56 Synthetic State/Policy Harness",
            "",
            f"Decision: **{decision}**",
            "",
            f"Harness SHA-256: `{harness_spec['harness_spec_sha256']}`",
            f"Parameters: **{parameter_count:,}**",
            "",
            "The exact V55 model, multi-horizon quantile loss, seven-date state-",
            "conditioned policy, accounting, checkpoint roundtrip, and deterministic",
            "resume passed on synthetic CPU tensors only.",
            "",
            "No Parquet, real label, previous checkpoint, market prediction,",
            "performance/PnL outcome, or BTC/ETH/SOL data was opened.",
            "V57 remains a dataset-construction phase only; training is not authorized.",
            "",
        ]
    )

    write_json_atomic(output / "harness_spec.json", harness_spec)
    write_json_atomic(output / "input_hash_receipt.json", input_receipt)
    write_json_atomic(output / "source_receipt.json", source_receipt)
    write_json_atomic(output / "smoke.json", smoke)
    write_json_atomic(output / "checkpoint_metadata.json", checkpoint_metadata)
    write_json_atomic(output / "audit.json", audit)
    write_yaml_atomic(output / "resolved_config.yaml", config)
    (output / "report.md").write_text(report, encoding="utf-8")
    result: dict[str, Any] = {
        "version": "v56",
        "candidate_family_id": blueprint["candidate_family_id"],
        "decision": decision,
        "harness_spec": harness_spec,
        "input_hash_receipt": input_receipt,
        "source_receipt": source_receipt,
        "smoke": smoke,
        "audit": audit,
    }
    result["result_sha256"] = canonical_sha256(result)
    write_json_atomic(output / "result.json", result)

    manifest_names = (
        "audit.json",
        "checkpoint_metadata.json",
        "harness_spec.json",
        "input_hash_receipt.json",
        "report.md",
        "resolved_config.yaml",
        "result.json",
        "smoke.json",
        "source_receipt.json",
        "synthetic_checkpoint.pt",
    )
    manifest = {
        "version": "v56",
        "files": {
            name: file_sha256(output / name) for name in manifest_names
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    write_json_atomic(output / "artifact_manifest.json", manifest)
    completion = {
        "version": "v56",
        "decision": decision,
        "harness_spec_sha256": harness_spec["harness_spec_sha256"],
        "result_file_sha256": file_sha256(output / "result.json"),
        "artifact_manifest_file_sha256": file_sha256(
            output / "artifact_manifest.json"
        ),
    }
    write_json_atomic(output / "completion_receipt.json", completion)
    if not audit["passed"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"V56 synthetic harness failed: {failed}")
    return result
