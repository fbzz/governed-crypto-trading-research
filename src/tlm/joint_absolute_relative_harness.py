from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml

from .joint_absolute_relative_model import (
    JOINT_HEADS,
    JointAbsoluteRelativeTransformer,
    _state_dict_sha256,
    fit_raw_return_rms_scale,
    joint_absolute_relative_loss,
    joint_triplet_positions,
    load_joint_checkpoint,
    reconstruct_joint_predictions,
    save_joint_checkpoint,
)
from .joint_absolute_relative_spec import (
    _canonical_sha256,
    _load_json,
    _sha256_file,
    _write_json,
    analytic_joint_parameter_count,
)
from .scientific_harness import FeatureScaler, persistent_portfolio_returns


@dataclass
class V48OperationLedger:
    authorized_metadata_reads: int = 0
    synthetic_feature_scaler_fits: int = 0
    synthetic_target_scale_fits: int = 0
    synthetic_tensor_generations: int = 0
    synthetic_optimizer_steps: int = 0
    synthetic_checkpoint_writes: int = 0
    synthetic_checkpoint_reads: int = 0
    real_panel_or_label_reads: int = 0
    previous_checkpoint_reads: int = 0
    real_training_epochs: int = 0
    real_predictions: int = 0
    real_performance_metrics: int = 0
    real_pnl_evaluations: int = 0
    target_asset_loads: int = 0


def _load_authorized_metadata(
    path: Path,
    allowed_paths: set[Path],
    ledger: V48OperationLedger,
) -> dict:
    resolved = path.resolve()
    if resolved not in allowed_paths:
        raise PermissionError(f"V48 metadata read is not allowlisted: {resolved}")
    payload = _load_json(resolved)
    ledger.authorized_metadata_reads += 1
    return payload


def _clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _optimizer_step_count(optimizer: torch.optim.Optimizer) -> int:
    steps = []
    for state in optimizer.state.values():
        value = state.get("step", 0)
        if torch.is_tensor(value):
            value = int(value.item())
        steps.append(int(value))
    return max(steps, default=0)


def _train_step(
    model: JointAbsoluteRelativeTransformer,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    returns: torch.Tensor,
    scale: float,
    tie_tolerance: float,
    clip_norm: float,
) -> tuple[float, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = joint_absolute_relative_loss(
        model(features), returns, scale, tie_tolerance=tie_tolerance
    )
    losses["total"].backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    optimizer.step()
    return float(losses["total"].detach()), float(grad_norm)


def _risk_matched_equal_weight(eligible: np.ndarray, gross: float) -> np.ndarray:
    eligibility = np.asarray(eligible, dtype=bool)
    positions = np.zeros(eligibility.shape, dtype=np.float64)
    for day in range(len(positions)):
        indexes = np.flatnonzero(eligibility[day])
        if len(indexes):
            positions[day, indexes] = gross / len(indexes)
    return positions


def _risk_matched_dual_momentum(
    momentum: np.ndarray, eligible: np.ndarray, gross: float
) -> np.ndarray:
    values = np.asarray(momentum, dtype=np.float64)
    eligibility = np.asarray(eligible, dtype=bool)
    positions = np.zeros(values.shape, dtype=np.float64)
    for day in range(len(positions)):
        indexes = np.flatnonzero(eligibility[day] & np.isfinite(values[day]))
        if not len(indexes):
            continue
        best_value = np.max(values[day, indexes])
        best = int(indexes[np.flatnonzero(values[day, indexes] == best_value)[0]])
        if best_value > 0:
            positions[day, best] = gross
    return positions


def _build_harness_spec(blueprint: dict, harness: dict) -> dict[str, object]:
    spec = {
        "version": "v48",
        "candidate_family_id": blueprint["candidate_family_id"],
        "v47_blueprint_sha256": blueprint["blueprint_sha256"],
        "architecture": blueprint["architecture"],
        "objective": blueprint["objective"],
        "early_stopping": blueprint["early_stopping"],
        "policy": blueprint["policy"],
        "later_evaluation": blueprint["later_evaluation"],
        "synthetic": harness["synthetic"],
        "checkpoint": harness["checkpoint"],
        "constraints": harness["constraints"],
    }
    spec["harness_spec_sha256"] = _canonical_sha256(spec)
    return spec


def _report(result: dict[str, object]) -> str:
    smoke = result["smoke"]
    return "\n".join(
        [
            "# TLM v48 Joint Absolute/Relative Synthetic Harness",
            "",
            "## Decision",
            "",
            "**SYNTHETIC HARNESS PASSED; ONLY THE FROZEN V49 TRAINING WORKFLOW IS AUTHORIZED.**",
            "",
            f"Harness SHA-256: `{result['harness_spec']['harness_spec_sha256']}`",
            f"Parameters: **{smoke['parameter_count']:,}**",
            f"Synthetic optimizer steps: **{smoke['optimizer_steps']}**",
            f"Synthetic return scale: **{smoke['return_scale']:.6f}**",
            f"Candidate turnover including liquidation: **{smoke['candidate_turnover']:.6f}**",
            "",
            "The exact model, mu=m+e loss, causal/permutation contracts, train-only scaling, cost-aware one-third policy, risk-matched controls, checkpoint schema, and interrupted resume passed on deterministic synthetic data.",
            "",
            "No real panel, label, previous checkpoint, target asset, held-out prediction, performance metric, or PnL was read or produced.",
            "",
            "## Next action",
            "",
            "V49 may implement the frozen metadata preflight and non-target training workflow. Real labels remain forbidden until the V49 source, config, tests, and clean Git receipt are committed.",
            "",
        ]
    )


def run_joint_absolute_relative_harness(config: dict) -> dict[str, object]:
    harness = config["joint_absolute_relative_harness"]
    root = Path(harness["project_root"]).resolve()
    paths = {name: root / relative for name, relative in harness["inputs"].items()}
    ledger = V48OperationLedger()
    allowed_paths = {path.resolve() for path in paths.values()}
    input_checks = {
        name: path.is_file()
        and _sha256_file(path) == harness["expected_input_sha256"][name]
        for name, path in paths.items()
    }
    if not all(input_checks.values()):
        raise RuntimeError(f"V48 input missing or hash drifted: {input_checks}")
    v47_result = _load_authorized_metadata(paths["v47_result"], allowed_paths, ledger)
    blueprint = _load_authorized_metadata(
        paths["v47_blueprint"], allowed_paths, ledger
    )
    v47_audit = _load_authorized_metadata(paths["v47_audit"], allowed_paths, ledger)
    if (
        v47_result["decision"]
        != "authorize_v48_joint_absolute_relative_synthetic_harness_only"
        or not v47_audit["passed"]
        or v47_result["blueprint_sha256"] != blueprint["blueprint_sha256"]
    ):
        raise RuntimeError("V47 does not authorize the V48 synthetic harness")

    harness_spec = _build_harness_spec(blueprint, harness)
    architecture = blueprint["architecture"]
    objective = blueprint["objective"]
    training = blueprint["training"]
    synthetic = harness["synthetic"]
    seed = int(config["seed"])
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    input_features = int(synthetic["input_features"])
    base_feature_names = [
        f"synthetic_feature_{index}" for index in range(input_features - 1)
    ]
    scaler_dates = pd.date_range("2021-01-01", periods=12, freq="D", tz="UTC")
    scaler_values = rng.normal(size=(len(scaler_dates), len(base_feature_names)))
    clean_panel = pd.DataFrame(scaler_values, columns=base_feature_names)
    clean_panel.insert(0, "date", scaler_dates)
    contaminated_panel = clean_panel.copy()
    contaminated_panel.loc[8:, base_feature_names] = 1e9
    feature_scaler = FeatureScaler.fit_from_panel(
        contaminated_panel,
        base_feature_names,
        "2021-01-01",
        "2021-01-08",
        "2021-01-08",
        base_feature_names[0],
    )
    clean_scaler = FeatureScaler.fit_from_panel(
        clean_panel,
        base_feature_names,
        "2021-01-01",
        "2021-01-08",
        "2021-01-08",
        base_feature_names[0],
    )
    ledger.synthetic_feature_scaler_fits += 2

    scale_count = int(synthetic["scale_fit_triplets"])
    train_returns = torch.from_numpy(
        rng.normal(0.0, 0.02, size=(scale_count, 3)).astype(np.float32)
    )
    validation_returns = torch.from_numpy(
        rng.normal(0.0, 0.02, size=(8, 3)).astype(np.float32)
    )
    all_returns = torch.cat([train_returns, validation_returns])
    train_mask = torch.zeros(len(all_returns), dtype=torch.bool)
    train_mask[:scale_count] = True
    return_scale = fit_raw_return_rms_scale(
        all_returns, train_mask, float(objective["scale_floor"])
    )
    replay_scale = fit_raw_return_rms_scale(
        all_returns, train_mask, float(objective["scale_floor"])
    )
    outlier_returns = all_returns.clone()
    outlier_returns[~train_mask] = 1e9
    outlier_scale = fit_raw_return_rms_scale(
        outlier_returns, train_mask, float(objective["scale_floor"])
    )
    zero_scale = fit_raw_return_rms_scale(
        torch.zeros(4, 3), torch.ones(4, dtype=torch.bool), float(objective["scale_floor"])
    )
    ledger.synthetic_target_scale_fits += 4

    batch_size = int(synthetic["batch_size"])
    raw_features = rng.normal(
        size=(batch_size, 256, 3, input_features)
    ).astype(np.float32)
    features = torch.from_numpy(feature_scaler.transform_triplet_tensor(raw_features))
    labels = torch.from_numpy(
        rng.normal(0.0, 0.02, size=(batch_size, 3)).astype(np.float32)
    )
    ledger.synthetic_tensor_generations += 2

    torch.manual_seed(seed)
    base_model = JointAbsoluteRelativeTransformer(input_features, architecture)
    base_state = _clone_state_dict(base_model)
    parameter_count = sum(parameter.numel() for parameter in base_model.parameters())

    # Uninterrupted two-step reference.
    full_model = JointAbsoluteRelativeTransformer(input_features, architecture)
    full_model.load_state_dict(base_state)
    full_optimizer = torch.optim.AdamW(
        full_model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    torch.manual_seed(seed + 1)
    full_history = []
    full_grad_norms = []
    for _ in range(int(synthetic["optimizer_steps"])):
        loss_value, grad_norm = _train_step(
            full_model,
            full_optimizer,
            features,
            labels,
            return_scale,
            float(objective["exact_tie_tolerance"]),
            float(training["gradient_clip_norm"]),
        )
        full_history.append(loss_value)
        full_grad_norms.append(grad_norm)

    # Interrupted after one step, persisted, reopened, RNG-restored, then resumed.
    interrupted_model = JointAbsoluteRelativeTransformer(input_features, architecture)
    interrupted_model.load_state_dict(base_state)
    interrupted_optimizer = torch.optim.AdamW(
        interrupted_model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    torch.manual_seed(seed + 1)
    first_loss, first_grad_norm = _train_step(
        interrupted_model,
        interrupted_optimizer,
        features,
        labels,
        return_scale,
        float(objective["exact_tie_tolerance"]),
        float(training["gradient_clip_norm"]),
    )
    ledger.synthetic_optimizer_steps += 1
    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "synthetic_checkpoint.pt"
    checkpoint_metadata = {
        "candidate_family_id": blueprint["candidate_family_id"],
        "v47_blueprint_sha256": blueprint["blueprint_sha256"],
        "initialization_seed": seed,
        "checkpoint_status": harness["checkpoint"]["status"],
        "job_key": "synthetic/fold_1/seed_20260713",
    }
    early_state = {
        "best_validation_total_loss": first_loss,
        "best_epoch": 1,
        "non_improvement_count": 0,
        "patience": 5,
        "minimum_delta": 0.0,
    }
    saved_cpu_rng = torch.get_rng_state().clone()
    save_joint_checkpoint(
        checkpoint_path,
        {
            "model_state": _clone_state_dict(interrupted_model),
            "best_model_state": _clone_state_dict(interrupted_model),
            "optimizer_state": deepcopy(interrupted_optimizer.state_dict()),
            "cpu_rng_state": saved_cpu_rng,
            "mps_rng_state": None,
            "early_stopping_state": early_state,
            "history": [first_loss],
            "metadata": checkpoint_metadata,
            "architecture": architecture,
            "input_features": input_features,
        },
        harness["checkpoint"]["format_version"],
    )
    ledger.synthetic_checkpoint_writes += 1
    checkpoint = load_joint_checkpoint(
        checkpoint_path,
        expected_format_version=harness["checkpoint"]["format_version"],
        expected_architecture=architecture,
        expected_metadata=checkpoint_metadata,
    )
    ledger.synthetic_checkpoint_reads += 1
    resumed_model = JointAbsoluteRelativeTransformer(input_features, architecture)
    resumed_model.load_state_dict(checkpoint["model_state"])
    resumed_optimizer = torch.optim.AdamW(
        resumed_model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    resumed_optimizer.load_state_dict(checkpoint["optimizer_state"])
    torch.set_rng_state(checkpoint["cpu_rng_state"])
    second_loss, second_grad_norm = _train_step(
        resumed_model,
        resumed_optimizer,
        features,
        labels,
        return_scale,
        float(objective["exact_tie_tolerance"]),
        float(training["gradient_clip_norm"]),
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
    predictions = reconstruct_joint_predictions(reference, return_scale)
    early_patch_count = (200 - resumed_model.patch_length) // resumed_model.patch_stride + 1

    loss_fixture = joint_absolute_relative_loss(
        reference,
        labels,
        return_scale,
        tie_tolerance=float(objective["exact_tie_tolerance"]),
    )
    tied_labels = torch.tensor(
        [[0.01, 0.01, -0.01], [0.00, 0.00, 0.00]], dtype=torch.float32
    )
    tie_losses = joint_absolute_relative_loss(
        {name: reference[name].detach().clone() for name in JOINT_HEADS},
        tied_labels,
        return_scale,
        tie_tolerance=float(objective["exact_tie_tolerance"]),
    )

    mu = np.array(
        [
            [0.0030, 0.0000, -0.0010],
            [0.0020, 0.0000, -0.0010],
            [0.0020, 0.0035, -0.0010],
            [0.0010, 0.0045, -0.0010],
            [0.0000, 0.0020, 0.0010],
            [-0.0030, -0.0040, -0.0050],
            [np.nan, np.nan, np.nan],
            [0.0000, 0.0000, 0.0030],
            [0.0000, 0.0030, 0.0000],
            [0.0020, 0.0000, -0.0010],
        ]
    )
    excess = np.array(
        [
            [3.0, 2.0, 1.0],
            [3.0, 2.0, 1.0],
            [2.0, 3.0, 1.0],
            [2.0, 3.0, 1.0],
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0],
            [np.nan, np.nan, np.nan],
            [1.0, 2.0, 3.0],
            [1.0, 3.0, 2.0],
            [3.0, 2.0, 1.0],
        ]
    )
    eligible = np.ones_like(mu, dtype=bool)
    eligible[6] = False
    eligible[8, 2] = False
    positions = joint_triplet_positions(
        mu,
        excess,
        eligible,
        risky_weight=float(blueprint["policy"]["risky_gross"]),
        base_cost=float(blueprint["policy"]["base_cost_bps"]) / 10_000.0,
    )
    expected_assets = [0, 0, 0, 1, 1, None, None, 2, 1, 1]
    observed_assets = [
        int(np.argmax(row)) if row.sum() else None for row in positions
    ]
    entry_tie = joint_triplet_positions(
        np.array([[0.001, 0.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0]]),
        np.ones((1, 3), dtype=bool),
    )
    lexical_tie = joint_triplet_positions(
        np.array([[0.003, 0.003, 0.0]]),
        np.array([[1.0, 1.0, 0.0]]),
        np.ones((1, 3), dtype=bool),
    )
    momentum = rng.normal(size=positions.shape)
    equal_positions = _risk_matched_equal_weight(eligible, 1.0 / 3.0)
    dual_positions = _risk_matched_dual_momentum(momentum, eligible, 1.0 / 3.0)
    cash_positions = np.zeros_like(positions)
    actual_returns = rng.normal(0.0, 0.01, size=positions.shape)
    ledger.synthetic_tensor_generations += 4
    accounting = {}
    for cost_bps in blueprint["policy"]["reporting_cost_bps"]:
        accounting[str(cost_bps)] = {
            "candidate": persistent_portfolio_returns(
                positions, actual_returns, float(cost_bps)
            ),
            "cash": persistent_portfolio_returns(
                cash_positions, actual_returns, float(cost_bps)
            ),
            "dual_momentum_30": persistent_portfolio_returns(
                dual_positions, actual_returns, float(cost_bps)
            ),
            "equal_weight": persistent_portfolio_returns(
                equal_positions, actual_returns, float(cost_bps)
            ),
        }

    base_accounting = accounting[str(blueprint["policy"]["base_cost_bps"])]
    operation_record = asdict(ledger)
    prohibited = (
        "real_panel_or_label_reads",
        "previous_checkpoint_reads",
        "real_training_epochs",
        "real_predictions",
        "real_performance_metrics",
        "real_pnl_evaluations",
        "target_asset_loads",
    )
    full_state_hash = _state_dict_sha256(full_model)
    resumed_state_hash = _state_dict_sha256(resumed_model)
    all_cost_identities = all(
        np.allclose(values["net_return"], values["gross_return"] - values["cost"])
        and np.allclose(
            values["cost"], values["turnover"] * (float(cost) / 10_000.0)
        )
        for cost, strategies in accounting.items()
        for values in strategies.values()
    )
    checks = {
        "all_v47_input_hashes_match": all(input_checks.values()),
        "v47_blueprint_and_audit_authorize_v48": v47_audit["passed"]
        and v47_result["decision"]
        == "authorize_v48_joint_absolute_relative_synthetic_harness_only",
        "input_allowlist_contains_only_v47_metadata": set(paths)
        == {"v47_result", "v47_blueprint", "v47_audit"},
        "parameter_count_matches_analytic_frozen_value": parameter_count
        == analytic_joint_parameter_count(architecture, input_features)
        == 1_212_930,
        "model_has_only_registered_heads": set(reference) == set(JOINT_HEADS)
        and all(reference[name].shape == (batch_size, 3) for name in JOINT_HEADS)
        and not hasattr(resumed_model, "mask_token")
        and not hasattr(resumed_model, "reconstruction_head"),
        "asset_permutation_equivariance_passes": all(
            torch.allclose(
                permuted[name], reference[name][:, permutation], atol=1e-5, rtol=1e-5
            )
            for name in JOINT_HEADS
        ),
        "market_prediction_is_permutation_invariant": torch.allclose(
            reconstruct_joint_predictions(permuted, return_scale)["m_hat_z"],
            predictions["m_hat_z"],
            atol=1e-5,
            rtol=1e-5,
        ),
        "causal_temporal_prefix_is_invariant": torch.allclose(
            temporal[:, :, :early_patch_count],
            altered_temporal[:, :, :early_patch_count],
            atol=1e-5,
            rtol=1e-5,
        ),
        "feature_scaler_is_train_only_and_validation_outlier_safe": feature_scaler.state_sha256()
        == clean_scaler.state_sha256()
        and feature_scaler.fit_rows == 8
        and np.isfinite(features.numpy()).all(),
        "raw_return_scale_replays_and_ignores_validation": math.isfinite(return_scale)
        and return_scale == replay_scale == outlier_scale,
        "zero_variance_scale_uses_registered_floor": zero_scale
        == float(objective["scale_floor"]),
        "centered_excess_and_reconstruction_identities_pass": torch.allclose(
            predictions["e_hat_z"].sum(dim=1), torch.zeros(batch_size), atol=1e-6
        )
        and torch.allclose(
            predictions["mu_hat_z"].mean(dim=1), predictions["m_hat_z"], atol=1e-6
        )
        and torch.allclose(
            predictions["mu_hat_z"],
            predictions["m_hat_z"][:, None] + predictions["e_hat_z"],
        ),
        "all_registered_losses_are_finite": all(
            bool(torch.isfinite(loss_fixture[name]))
            for name in (
                "ranking",
                "excess",
                "market_level",
                "absolute_level",
                "level",
                "total",
            )
        ),
        "ranknet_uses_pairs_and_excludes_ties": int(loss_fixture["pair_count"])
        == batch_size * 3
        and int(tie_losses["pair_count"]) == 2,
        "all_model_parameters_receive_finite_gradients": all(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in resumed_model.parameters()
        ),
        "gradient_norms_and_histories_are_finite": all(
            math.isfinite(value)
            for value in full_history + full_grad_norms + [
                first_loss,
                first_grad_norm,
                second_loss,
                second_grad_norm,
            ]
        ),
        "interrupted_resume_matches_uninterrupted_training": full_state_hash
        == resumed_state_hash
        and full_history == [first_loss, second_loss]
        and _optimizer_step_count(full_optimizer)
        == _optimizer_step_count(resumed_optimizer)
        == 2,
        "checkpoint_contains_exact_resume_contract": checkpoint[
            "early_stopping_state"
        ]
        == early_state
        and checkpoint["history"] == [first_loss]
        and torch.equal(checkpoint["cpu_rng_state"], saved_cpu_rng)
        and checkpoint["cpu_rng_state"].dtype == torch.uint8
        and checkpoint["mps_rng_state"] is None,
        "checkpoint_model_roundtrip_is_exact": all(
            torch.equal(interrupted_model.state_dict()[name], tensor)
            for name, tensor in checkpoint["model_state"].items()
        ),
        "policy_fixture_covers_registered_transitions": observed_assets
        == expected_assets,
        "entry_cost_tie_prefers_cash": not bool(entry_tie.any()),
        "lexical_excess_tie_prefers_first_asset": int(np.argmax(lexical_tie[0]))
        == 0,
        "turnover_units_and_final_liquidation_are_exact": np.allclose(
            base_accounting["candidate"]["turnover"],
            [1 / 3, 0, 0, 2 / 3, 0, 1 / 3, 0, 1 / 3, 2 / 3, 1 / 3],
        )
        and math.isclose(
            float(base_accounting["candidate"]["total_turnover"]), 8.0 / 3.0
        ),
        "candidate_and_controls_are_risk_matched": np.all(
            positions.sum(axis=1) <= 1.0 / 3.0 + 1e-12
        )
        and np.all(equal_positions.sum(axis=1) <= 1.0 / 3.0 + 1e-12)
        and np.all(dual_positions.sum(axis=1) <= 1.0 / 3.0 + 1e-12),
        "missing_and_nonfinite_assets_never_receive_weight": not bool(
            positions[~eligible].any()
        )
        and positions[6].sum() == 0,
        "all_cost_and_control_cells_are_present": set(accounting)
        == {"10", "20", "30", "50"}
        and all(
            set(strategies)
            == {"candidate", "cash", "dual_momentum_30", "equal_weight"}
            for strategies in accounting.values()
        ),
        "cost_accounting_and_cash_identity_pass": all_cost_identities
        and all(
            np.allclose(cell["cash"]["net_return"], 0.0)
            for cell in accounting.values()
        ),
        "higher_reporting_cost_never_improves_same_positions": all(
            accounting[str(left)]["candidate"]["total_cost"]
            <= accounting[str(right)]["candidate"]["total_cost"]
            for left, right in ((10, 20), (20, 30), (30, 50))
        ),
        "exactly_two_synthetic_optimizer_steps": ledger.synthetic_optimizer_steps
        == int(synthetic["optimizer_steps"])
        == 2,
        "authorized_metadata_reads_are_counted": ledger.authorized_metadata_reads
        == len(paths)
        == 3,
        "real_and_target_operations_remain_zero": all(
            operation_record[name] == 0 for name in prohibited
        ),
        "configuration_forbids_every_real_operation": all(
            value is False for value in harness["constraints"].values()
        ),
        "input_hashes_still_match_after_harness": all(
            _sha256_file(paths[name]) == expected
            for name, expected in harness["expected_input_sha256"].items()
        ),
        "only_v49_training_is_authorized": harness["authorized_next_action"]
        == "authorize_v49_purged_non_target_training_only",
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError(f"V48 synthetic harness audit failed: {failed}")

    smoke = {
        "parameter_count": parameter_count,
        "optimizer_steps": ledger.synthetic_optimizer_steps,
        "return_scale": return_scale,
        "pair_count": int(loss_fixture["pair_count"]),
        "tie_pair_count": int(tie_losses["pair_count"]),
        "losses": {
            name: float(loss_fixture[name].detach())
            for name in (
                "ranking",
                "excess",
                "market_level",
                "absolute_level",
                "level",
                "total",
            )
        },
        "candidate_turnover": float(
            base_accounting["candidate"]["total_turnover"]
        ),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_state_sha256": resumed_state_hash,
        "resume_equivalent": True,
        "synthetic_only": True,
    }
    result = {
        "version": "v48",
        "decision": "authorize_v49_purged_non_target_training_only",
        "harness_spec": harness_spec,
        "smoke": smoke,
        "operation_ledger": operation_record,
        "tested": {
            **{name: operation_record[name] for name in prohibited},
            "improvement_status": "unknown_synthetic_only",
        },
        "audit": {"passed": True, "checks": checks},
    }
    _write_json(output / "result.json", result)
    _write_json(output / "harness_spec.json", harness_spec)
    _write_json(output / "smoke.json", smoke)
    _write_json(output / "checkpoint_metadata.json", checkpoint_metadata)
    _write_json(output / "audit.json", result["audit"])
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    return result
