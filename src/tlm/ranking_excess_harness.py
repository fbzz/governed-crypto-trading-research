from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from .patch_transformer import MultiAssetPatchTransformer
from .ranking_excess_spec import (
    _canonical_sha256,
    _load_json,
    _sha256_file,
    _write_json,
    analytic_parameter_count,
)
from .scientific_harness import (
    FeatureScaler,
    deterministic_patch_mask,
    masked_reconstruction_loss,
    persistent_portfolio_returns,
)


RANKING_EXCESS_HEADS = ("excess_return_z", "log_volatility_7d")
PAIR_INDEXES = ((0, 1), (0, 2), (1, 2))


@dataclass(frozen=True)
class SharedAssetRidgeModel:
    coefficient: np.ndarray
    intercept: float
    solution_form: str


@dataclass
class V42OperationLedger:
    authorized_metadata_reads: int = 0
    synthetic_feature_scaler_fits: int = 0
    synthetic_target_scale_fits: int = 0
    synthetic_tensor_generations: int = 0
    synthetic_optimizer_steps: int = 0
    synthetic_checkpoint_writes: int = 0
    synthetic_checkpoint_reads: int = 0
    real_panel_or_label_reads: int = 0
    real_training_epochs: int = 0
    real_predictions: int = 0
    real_performance_metrics: int = 0
    real_pnl_evaluations: int = 0
    target_asset_loads: int = 0


def _load_authorized_metadata(
    path: Path,
    allowed_paths: set[Path],
    ledger: V42OperationLedger,
) -> dict:
    resolved = path.resolve()
    if resolved not in allowed_paths:
        raise PermissionError(f"V42 metadata read is not allowlisted: {resolved}")
    payload = _load_json(resolved)
    ledger.authorized_metadata_reads += 1
    return payload


def fit_triplet_excess_rms_scale(
    observed_log_returns: torch.Tensor,
    train_mask: torch.Tensor,
    floor: float,
) -> float:
    if observed_log_returns.ndim != 2 or observed_log_returns.shape[1] != 3:
        raise ValueError("Returns must have shape [triplets, 3]")
    if floor <= 0:
        raise ValueError("Scale floor must be positive")
    if not bool(torch.isfinite(observed_log_returns).all()):
        raise ValueError("Returns must be finite")
    if train_mask.ndim != 1 or train_mask.shape[0] != observed_log_returns.shape[0]:
        raise ValueError("Train mask must select the triplet rows")
    if train_mask.dtype != torch.bool or not bool(train_mask.any()):
        raise ValueError("Train mask must be boolean and select at least one row")
    train_returns = observed_log_returns[train_mask]
    excess = train_returns - train_returns.mean(dim=1, keepdim=True)
    scale = torch.sqrt(torch.mean(excess.square()))
    return max(float(scale), float(floor))


def normalized_triplet_excess(
    observed_log_returns: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    if observed_log_returns.ndim != 2 or observed_log_returns.shape[1] != 3:
        raise ValueError("Returns must have shape [batch, 3]")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Excess scale must be finite and positive")
    excess = observed_log_returns - observed_log_returns.mean(dim=1, keepdim=True)
    return excess / float(scale)


def ranking_excess_loss(
    output: dict[str, torch.Tensor],
    labels: torch.Tensor,
    scale: float,
    *,
    tie_tolerance: float,
    volatility_floor: float,
    ranking_weight: float,
    excess_weight: float,
    volatility_weight: float,
) -> dict[str, torch.Tensor]:
    if labels.ndim != 3 or tuple(labels.shape[1:]) != (3, 2):
        raise ValueError("Labels must have shape [batch, 3, 2]")
    if set(output) != set(RANKING_EXCESS_HEADS):
        raise ValueError("Ranking/excess output-head contract drift")
    if output["excess_return_z"].shape != labels.shape[:2]:
        raise ValueError("Score shape does not match labels")
    if tie_tolerance < 0:
        raise ValueError("Tie tolerance cannot be negative")

    returns = labels[..., 0]
    target_z = normalized_triplet_excess(returns, scale)
    raw_score = output["excess_return_z"]
    centered_score = raw_score - raw_score.mean(dim=1, keepdim=True)
    pair_losses = []
    pair_count = 0
    pair_correct = 0
    for left, right in PAIR_INDEXES:
        return_difference = returns[:, left] - returns[:, right]
        target_difference = target_z[:, left] - target_z[:, right]
        score_difference = centered_score[:, left] - centered_score[:, right]
        active = return_difference.abs() > tie_tolerance
        if bool(active.any()):
            sign = target_difference[active].sign()
            difference = score_difference[active]
            pair_losses.append(F.softplus(-sign * difference))
            pair_count += int(active.sum())
            pair_correct += int(((difference * sign) > 0).sum())
    if pair_losses:
        ranking = torch.cat(pair_losses).mean()
    else:
        ranking = centered_score.sum() * 0.0
    excess = F.smooth_l1_loss(centered_score, target_z, beta=1.0)
    observed_volatility = labels[..., 1].clamp_min(volatility_floor).log()
    log_volatility = F.smooth_l1_loss(
        output["log_volatility_7d"],
        observed_volatility,
        beta=1.0,
    )
    core = ranking_weight * ranking + excess_weight * excess
    total = core + volatility_weight * log_volatility
    return {
        "ranking": ranking,
        "excess": excess,
        "log_volatility": log_volatility,
        "core": core,
        "total": total,
        "centered_score": centered_score,
        "target_z": target_z,
        "pair_count": torch.tensor(pair_count, device=labels.device),
        "pair_accuracy": torch.tensor(
            pair_correct / pair_count if pair_count else 0.0,
            dtype=labels.dtype,
            device=labels.device,
        ),
    }


def aggregate_raw_excess_predictions(
    predicted_excess_z: np.ndarray,
    fold_scales: np.ndarray,
) -> np.ndarray:
    scores = np.asarray(predicted_excess_z, dtype=np.float64)
    scales = np.asarray(fold_scales, dtype=np.float64)
    if scores.ndim != 3:
        raise ValueError("Predicted z-scores must have [members, days, assets] shape")
    if scales.shape != (scores.shape[0],):
        raise ValueError("One train-only scale is required per ensemble member")
    if not np.isfinite(scores).all() or not np.isfinite(scales).all():
        raise ValueError("Predicted z-scores and fold scales must be finite")
    if np.any(scales <= 0):
        raise ValueError("Fold scales must be positive")
    centered = scores - scores.mean(axis=2, keepdims=True)
    raw_by_member = centered * scales[:, None, None]
    return raw_by_member.mean(axis=0)


def ranking_excess_positions(
    predicted_raw_excess: np.ndarray,
    momentum_30: np.ndarray,
    eligible: np.ndarray,
    switch_hurdle: float,
) -> np.ndarray:
    scores = np.asarray(predicted_raw_excess, dtype=np.float64)
    momentum = np.asarray(momentum_30, dtype=np.float64)
    eligibility = np.asarray(eligible, dtype=bool)
    if (
        scores.shape != momentum.shape
        or scores.shape != eligibility.shape
        or scores.ndim != 2
    ):
        raise ValueError(
            "Scores, momentum, and eligibility must share [days, assets] shape"
        )
    if not np.isfinite(scores[eligibility]).all() or not np.isfinite(
        momentum[eligibility]
    ).all():
        raise ValueError("Scores and momentum must be finite for eligible assets")
    if switch_hurdle < 0:
        raise ValueError("Switch hurdle cannot be negative")
    positions = np.zeros_like(scores)
    incumbent: int | None = None
    for day in range(len(scores)):
        eligible_assets = np.flatnonzero(eligibility[day])
        if incumbent is not None and not eligibility[day, incumbent]:
            incumbent = None
        if (
            len(eligible_assets) == 0
            or np.all(momentum[day, eligible_assets] <= 0)
        ):
            incumbent = None
            continue
        challenger = int(
            eligible_assets[np.argmax(scores[day, eligible_assets])]
        )
        if incumbent is None:
            incumbent = challenger
        elif challenger != incumbent and (
            scores[day, challenger] - scores[day, incumbent] > switch_hurdle
        ):
            incumbent = challenger
        positions[day, incumbent] = 1.0
    return positions


def momentum_gated_equal_weight_positions(
    momentum_30: np.ndarray,
    eligible: np.ndarray,
) -> np.ndarray:
    momentum = np.asarray(momentum_30, dtype=np.float64)
    eligibility = np.asarray(eligible, dtype=bool)
    if momentum.ndim != 2 or momentum.shape != eligibility.shape:
        raise ValueError("Momentum and eligibility must share [days, assets] shape")
    if not np.isfinite(momentum[eligibility]).all():
        raise ValueError("Momentum must be finite for eligible assets")
    positions = np.zeros_like(momentum)
    for day in range(len(momentum)):
        eligible_assets = np.flatnonzero(eligibility[day])
        if len(eligible_assets) and np.any(momentum[day, eligible_assets] > 0):
            positions[day, eligible_assets] = 1.0 / len(eligible_assets)
    return positions


def eligible_dual_momentum_positions(
    momentum_30: np.ndarray,
    eligible: np.ndarray,
) -> np.ndarray:
    momentum = np.asarray(momentum_30, dtype=np.float64)
    eligibility = np.asarray(eligible, dtype=bool)
    if momentum.ndim != 2 or momentum.shape != eligibility.shape:
        raise ValueError("Momentum and eligibility must share [days, assets] shape")
    if not np.isfinite(momentum[eligibility]).all():
        raise ValueError("Momentum must be finite for eligible assets")
    positions = np.zeros_like(momentum)
    for day in range(len(momentum)):
        eligible_assets = np.flatnonzero(eligibility[day])
        if not len(eligible_assets):
            continue
        best = int(eligible_assets[np.argmax(momentum[day, eligible_assets])])
        if momentum[day, best] > 0:
            positions[day, best] = 1.0
    return positions


def fit_shared_asset_ridge(
    tensor: np.ndarray,
    target_z: np.ndarray,
    alpha: float,
) -> SharedAssetRidgeModel:
    values = np.asarray(tensor, dtype=np.float64)
    targets = np.asarray(target_z, dtype=np.float64)
    if values.ndim != 4 or values.shape[2] != 3:
        raise ValueError("Tensor must have [batch, time, 3, features] shape")
    if targets.shape != (values.shape[0], 3):
        raise ValueError("Ridge targets must have [batch, 3] shape")
    if not np.isfinite(values).all() or not np.isfinite(targets).all():
        raise ValueError("Ridge inputs and targets must be finite")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Ridge alpha must be finite and positive")
    design = values.transpose(0, 2, 1, 3).reshape(
        values.shape[0] * 3,
        values.shape[1] * values.shape[3],
    )
    response = targets.reshape(-1)
    design_mean = design.mean(axis=0)
    response_mean = float(response.mean())
    centered_design = design - design_mean
    centered_response = response - response_mean
    if centered_design.shape[0] <= centered_design.shape[1]:
        gram = np.einsum(
            "ik,jk->ij",
            centered_design,
            centered_design,
            optimize=False,
        )
        gram.flat[:: gram.shape[0] + 1] += float(alpha)
        dual = np.linalg.solve(gram, centered_response)
        coefficient = np.einsum(
            "ij,i->j",
            centered_design,
            dual,
            optimize=False,
        )
        solution_form = "dual"
    else:
        gram = np.einsum(
            "ij,ik->jk",
            centered_design,
            centered_design,
            optimize=False,
        )
        gram.flat[:: gram.shape[0] + 1] += float(alpha)
        right_hand_side = np.einsum(
            "ij,i->j",
            centered_design,
            centered_response,
            optimize=False,
        )
        coefficient = np.linalg.solve(gram, right_hand_side)
        solution_form = "primal"
    intercept = response_mean - float(
        np.einsum("j,j->", design_mean, coefficient, optimize=False)
    )
    if not np.isfinite(coefficient).all() or not math.isfinite(intercept):
        raise FloatingPointError("Ridge fit produced non-finite parameters")
    return SharedAssetRidgeModel(
        coefficient=coefficient,
        intercept=intercept,
        solution_form=solution_form,
    )


def predict_shared_asset_ridge(
    model: SharedAssetRidgeModel,
    tensor: np.ndarray,
) -> np.ndarray:
    values = np.asarray(tensor, dtype=np.float64)
    if values.ndim != 4 or values.shape[2] != 3:
        raise ValueError("Tensor must have [batch, time, 3, features] shape")
    if not np.isfinite(values).all():
        raise ValueError("Ridge inputs must be finite")
    design = values.transpose(0, 2, 1, 3).reshape(
        values.shape[0] * 3,
        values.shape[1] * values.shape[3],
    )
    if design.shape[1] != model.coefficient.shape[0]:
        raise ValueError("Ridge feature shape drift")
    prediction = (
        np.einsum(
            "ij,j->i",
            design,
            model.coefficient,
            optimize=False,
        )
        + model.intercept
    ).reshape(values.shape[0], 3)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Ridge prediction produced non-finite values")
    return prediction - prediction.mean(axis=1, keepdims=True)


def _state_dict_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def save_ranking_excess_checkpoint(
    model: MultiAssetPatchTransformer,
    path: Path,
    architecture: dict,
    metadata: dict,
    format_version: str,
) -> None:
    required = {
        "candidate_family_id",
        "v41_blueprint_sha256",
        "initialization_seed",
        "checkpoint_status",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"Checkpoint metadata missing: {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": format_version,
        "input_features": model.input_features,
        "architecture": architecture,
        "architecture_sha256": _canonical_sha256(architecture),
        "prediction_heads": list(RANKING_EXCESS_HEADS),
        "state_dict": model.state_dict(),
        "state_dict_sha256": _state_dict_sha256(model),
        "metadata": metadata,
    }, path)


def load_ranking_excess_checkpoint(
    path: Path,
    expected_format_version: str,
    expected_input_features: int,
    expected_architecture: dict,
    expected_metadata: dict,
) -> tuple[MultiAssetPatchTransformer, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["format_version"] != expected_format_version:
        raise ValueError("Ranking/excess checkpoint format drift")
    if int(payload["input_features"]) != int(expected_input_features):
        raise ValueError("Ranking/excess checkpoint input-feature drift")
    if payload["architecture_sha256"] != _canonical_sha256(
        payload["architecture"]
    ):
        raise ValueError("Ranking/excess checkpoint architecture hash drift")
    if payload["architecture"] != expected_architecture:
        raise ValueError("Ranking/excess checkpoint does not match V41 architecture")
    if tuple(payload["prediction_heads"]) != RANKING_EXCESS_HEADS:
        raise ValueError("Ranking/excess checkpoint head drift")
    if payload["metadata"] != expected_metadata:
        raise ValueError("Ranking/excess checkpoint metadata does not match V41")
    model = MultiAssetPatchTransformer(
        int(payload["input_features"]),
        payload["architecture"],
        expected_prediction_heads=RANKING_EXCESS_HEADS,
    )
    model.load_state_dict(payload["state_dict"])
    if _state_dict_sha256(model) != payload["state_dict_sha256"]:
        raise ValueError("Ranking/excess checkpoint state hash drift")
    return model, payload


def _build_harness_spec(blueprint: dict, harness: dict) -> dict[str, object]:
    spec = {
        "version": "v42",
        "candidate_family_id": blueprint["candidate_family_id"],
        "v41_blueprint_sha256": blueprint["blueprint_sha256"],
        "architecture": blueprint["architecture"],
        "objective": blueprint["objective"],
        "policy": blueprint["policy"],
        "baselines": blueprint["baselines"],
        "synthetic": harness["synthetic"],
        "checkpoint": harness["checkpoint"],
        "constraints": harness["constraints"],
    }
    spec["harness_spec_sha256"] = _canonical_sha256(spec)
    return spec


def _report(result: dict[str, object]) -> str:
    smoke = result["smoke"]
    return "\n".join([
        "# TLM v42 Synthetic Ranking/Excess Harness",
        "",
        "## Decision",
        "",
        "**SYNTHETIC HARNESS PASSED; MEDIUM NON-TARGET PRETRAINING IS AUTHORIZED NEXT.**",
        "",
        f"Harness SHA-256: `{result['harness_spec']['harness_spec_sha256']}`",
        f"Actual parameters: **{smoke['parameter_count']:,}**",
        f"Synthetic optimizer steps: **{smoke['optimizer_steps']}**",
        f"Synthetic excess scale: **{smoke['excess_scale']:.6f}**",
        f"Synthetic policy turnover: **{smoke['candidate_turnover']:.1f}**",
        "",
        "The Medium model passed output-shape, causal-prefix, asset-permutation, reconstruction, ranking/excess loss, finite-gradient, shared-Ridge, turnover-hurdle, cost-accounting, and checkpoint-roundtrip checks.",
        "",
        "All tensors and returns were generated from a fixed random seed. No real panel, label, checkpoint, market result, BTC/ETH/SOL value, performance claim, or deployable strategy was loaded or produced.",
        "",
        "## Next action",
        "",
        "V43 may run masked-patch pretraining for all nine Medium fold/seed jobs on the frozen non-target representation-train window. It may not read forward labels, train the ranking heads, inspect held-out assets, or evaluate performance.",
        "",
    ])


def run_ranking_excess_harness(config: dict) -> dict[str, object]:
    harness = config["ranking_excess_harness"]
    root = Path(harness["project_root"]).resolve()
    paths = {name: root / relative for name, relative in harness["inputs"].items()}
    ledger = V42OperationLedger()
    allowed_metadata_paths = {path.resolve() for path in paths.values()}
    input_checks = {
        name: path.is_file()
        and _sha256_file(path) == harness["expected_input_sha256"][name]
        for name, path in paths.items()
    }
    if not all(input_checks.values()):
        raise RuntimeError(f"V42 input missing or hash drifted: {input_checks}")
    v41_result = _load_authorized_metadata(
        paths["v41_specification"], allowed_metadata_paths, ledger
    )
    blueprint = _load_authorized_metadata(
        paths["v41_blueprint"], allowed_metadata_paths, ledger
    )
    v41_audit = _load_authorized_metadata(
        paths["v41_audit"], allowed_metadata_paths, ledger
    )
    if (
        v41_result["decision"]
        != "authorize_v42_synthetic_ranking_excess_harness_only"
        or not v41_audit["passed"]
        or v41_result["blueprint_sha256"] != blueprint["blueprint_sha256"]
    ):
        raise RuntimeError("V41 does not authorize the V42 synthetic harness")

    harness_spec = _build_harness_spec(blueprint, harness)
    architecture = blueprint["architecture"]
    objective = blueprint["objective"]
    synthetic = harness["synthetic"]
    seed = int(config["seed"])
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    input_features = int(synthetic["input_features"])
    base_feature_names = [
        f"synthetic_feature_{index}" for index in range(input_features - 1)
    ]
    if len(base_feature_names) < 1:
        raise ValueError("V42 requires base features plus one relative feature")
    scaler_dates = pd.date_range("2021-01-01", periods=12, freq="D", tz="UTC")
    scaler_values = rng.normal(size=(len(scaler_dates), len(base_feature_names)))
    clean_scaler_panel = pd.DataFrame(
        scaler_values,
        columns=base_feature_names,
    )
    clean_scaler_panel.insert(0, "date", scaler_dates)
    contaminated_scaler_panel = clean_scaler_panel.copy()
    contaminated_scaler_panel.loc[8:, base_feature_names] = 1e6
    feature_scaler = FeatureScaler.fit_from_panel(
        contaminated_scaler_panel,
        base_feature_names,
        "2021-01-01",
        "2021-01-08",
        "2021-01-08",
        base_feature_names[0],
    )
    ledger.synthetic_feature_scaler_fits += 1
    clean_feature_scaler = FeatureScaler.fit_from_panel(
        clean_scaler_panel,
        base_feature_names,
        "2021-01-01",
        "2021-01-08",
        "2021-01-08",
        base_feature_names[0],
    )
    ledger.synthetic_feature_scaler_fits += 1

    scale_fit_triplets = int(synthetic["scale_fit_triplets"])
    train_scale_returns = torch.from_numpy(
        rng.normal(0.0, 0.02, size=(scale_fit_triplets, 3)).astype(np.float32)
    )
    validation_scale_returns = torch.from_numpy(
        rng.normal(0.0, 0.02, size=(8, 3)).astype(np.float32)
    )
    all_scale_returns = torch.cat(
        [train_scale_returns, validation_scale_returns], dim=0
    )
    train_scale_mask = torch.zeros(len(all_scale_returns), dtype=torch.bool)
    train_scale_mask[:scale_fit_triplets] = True
    excess_scale = fit_triplet_excess_rms_scale(
        all_scale_returns,
        train_scale_mask,
        float(objective["scale_floor"]),
    )
    ledger.synthetic_target_scale_fits += 1
    scale_replay = fit_triplet_excess_rms_scale(
        all_scale_returns,
        train_scale_mask,
        float(objective["scale_floor"]),
    )
    ledger.synthetic_target_scale_fits += 1
    validation_outlier_returns = all_scale_returns.clone()
    validation_outlier_returns[~train_scale_mask] = 1e6
    scale_after_unseen_outlier = fit_triplet_excess_rms_scale(
        validation_outlier_returns,
        train_scale_mask,
        float(objective["scale_floor"]),
    )
    ledger.synthetic_target_scale_fits += 1
    ledger.synthetic_tensor_generations += 3
    model = MultiAssetPatchTransformer(
        input_features,
        architecture,
        expected_prediction_heads=RANKING_EXCESS_HEADS,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(blueprint["training"]["learning_rate"]),
        weight_decay=float(blueprint["training"]["weight_decay"]),
    )
    batch_size = int(synthetic["batch_size"])
    raw_fixture = rng.normal(
        size=(batch_size, 256, 3, input_features)
    ).astype(np.float32)
    fixture = torch.from_numpy(
        feature_scaler.transform_triplet_tensor(raw_fixture)
    )
    ledger.synthetic_tensor_generations += 1

    model.train()
    patch_mask = deterministic_patch_mask(
        batch_size,
        3,
        model.patch_count,
        0.15,
        seed=42,
        fold=1,
        epoch=0,
        batch_index=0,
    )
    patch_mask_replay = deterministic_patch_mask(
        batch_size,
        3,
        model.patch_count,
        0.15,
        seed=42,
        fold=1,
        epoch=0,
        batch_index=0,
    )
    target_patches = model.extract_patches(fixture)
    optimizer.zero_grad(set_to_none=True)
    pretrain_output = model(
        fixture,
        patch_mask=patch_mask,
        return_reconstruction=True,
    )
    pretrain_loss = masked_reconstruction_loss(
        pretrain_output["patch_reconstruction"],
        target_patches,
        patch_mask,
    )
    pretrain_loss.backward()
    pretrain_gradient_passes = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for name, parameter in model.named_parameters()
        if name.startswith((
            "patch_projection",
            "temporal_encoder",
            "reconstruction_head",
            "mask_token",
        ))
    )
    pretrain_grad_norm = nn.utils.clip_grad_norm_(
        model.parameters(),
        float(blueprint["training"]["gradient_clip_norm"]),
    )
    optimizer.step()
    ledger.synthetic_optimizer_steps += 1

    labels = torch.from_numpy(np.stack([
        rng.normal(0.0, 0.02, size=(batch_size, 3)),
        rng.uniform(0.01, 0.20, size=(batch_size, 3)),
    ], axis=-1).astype(np.float32))
    optimizer.zero_grad(set_to_none=True)
    supervised_output = model(fixture)
    losses = ranking_excess_loss(
        supervised_output,
        labels,
        excess_scale,
        tie_tolerance=float(objective["exact_tie_tolerance"]),
        volatility_floor=float(objective["volatility_floor"]),
        ranking_weight=float(objective["weights"]["ranking"]),
        excess_weight=float(objective["weights"]["excess"]),
        volatility_weight=float(objective["weights"]["log_volatility"]),
    )
    losses["total"].backward()
    supervised_gradient_passes = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for name, parameter in model.named_parameters()
        if name.startswith((
            "patch_projection",
            "temporal_encoder",
            "cross_asset_encoder",
            "prediction_heads",
        ))
    )
    supervised_grad_norm = nn.utils.clip_grad_norm_(
        model.parameters(),
        float(blueprint["training"]["gradient_clip_norm"]),
    )
    optimizer.step()
    ledger.synthetic_optimizer_steps += 1

    model.eval()
    with torch.no_grad():
        reference_output = model(fixture)
        permutation = torch.tensor([2, 0, 1])
        permuted_output = model(fixture[:, :, permutation, :])
        temporal = model.encode_temporal_patches(fixture)
        altered = fixture.clone()
        altered[:, 200:, :, :] += 100.0
        altered_temporal = model.encode_temporal_patches(altered)
    early_patch_count = (200 - model.patch_length) // model.patch_stride + 1
    permutation_passes = all(torch.allclose(
        permuted_output[name],
        reference_output[name][:, permutation],
        atol=1e-5,
        rtol=1e-5,
    ) for name in RANKING_EXCESS_HEADS)

    raw_ridge_tensor = rng.normal(
        size=(16, 256, 3, input_features)
    ).astype(np.float32)
    ridge_tensor = feature_scaler.transform_triplet_tensor(raw_ridge_tensor)
    ridge_returns = torch.from_numpy(
        rng.normal(0.0, 0.02, size=(16, 3)).astype(np.float32)
    )
    ledger.synthetic_tensor_generations += 2
    ridge_target = normalized_triplet_excess(
        ridge_returns,
        excess_scale,
    ).numpy()
    ridge = fit_shared_asset_ridge(
        ridge_tensor,
        ridge_target,
        float(synthetic["ridge_alpha"]),
    )
    ridge_prediction = predict_shared_asset_ridge(ridge, ridge_tensor)

    desired_raw_scores = np.array([
        [0.0010, 0.0000, -0.0010],
        [0.0015, 0.0010, 0.0000],
        [0.0010, 0.0025, 0.0000],
        [0.0010, 0.0031, 0.0000],
        [0.0010, 0.0030, 0.0040],
        [0.0030, 0.0020, 0.0010],
        [0.0040, 0.0010, 0.0000],
        [0.0010, 0.0040, 0.0000],
    ])
    policy_member_z = desired_raw_scores[None, :, :] / excess_scale
    scores = aggregate_raw_excess_predictions(
        policy_member_z,
        np.array([excess_scale]),
    )
    model_raw_excess = aggregate_raw_excess_predictions(
        reference_output["excess_return_z"].detach().numpy()[None, :, :],
        np.array([excess_scale]),
    )
    momentum = np.array([
        [0.10, 0.05, -0.02],
        [0.09, 0.04, -0.03],
        [0.08, 0.03, -0.04],
        [0.07, 0.02, -0.05],
        [0.06, 0.01, -0.06],
        [-0.01, -0.02, -0.03],
        [0.03, 0.02, -0.01],
        [0.02, 0.01, -0.02],
    ])
    eligibility = np.ones_like(momentum, dtype=bool)
    eligibility[4, 1] = False
    eligibility[6, 0] = False
    scores[6, 0] = np.nan
    momentum[6, 0] = np.nan
    candidate_positions = ranking_excess_positions(
        scores,
        momentum,
        eligibility,
        float(blueprint["policy"]["switch_hurdle"]),
    )
    expected_assets = [0, 0, 0, 1, 2, None, 1, 1]
    observed_assets = [
        int(np.argmax(row)) if row.sum() else None for row in candidate_positions
    ]
    dual_positions = eligible_dual_momentum_positions(momentum, eligibility)
    equal_positions = momentum_gated_equal_weight_positions(
        momentum, eligibility
    )
    actual_log_returns = rng.normal(0.0, 0.01, size=scores.shape)
    ledger.synthetic_tensor_generations += 2
    cost_results = {}
    for cost_bps in blueprint["policy"]["reporting_cost_bps"]:
        cost_results[str(cost_bps)] = {
            "candidate": persistent_portfolio_returns(
                candidate_positions,
                actual_log_returns,
                float(cost_bps),
            ),
            "dual_momentum_30": persistent_portfolio_returns(
                dual_positions,
                actual_log_returns,
                float(cost_bps),
            ),
            "momentum_gated_equal_weight": persistent_portfolio_returns(
                equal_positions,
                actual_log_returns,
                float(cost_bps),
            ),
        }

    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "synthetic_checkpoint.pt"
    checkpoint_metadata = {
        "candidate_family_id": blueprint["candidate_family_id"],
        "v41_blueprint_sha256": blueprint["blueprint_sha256"],
        "initialization_seed": seed,
        "checkpoint_status": harness["checkpoint"]["status"],
    }
    save_ranking_excess_checkpoint(
        model,
        checkpoint_path,
        architecture,
        checkpoint_metadata,
        harness["checkpoint"]["format_version"],
    )
    ledger.synthetic_checkpoint_writes += 1
    loaded_model, checkpoint = load_ranking_excess_checkpoint(
        checkpoint_path,
        harness["checkpoint"]["format_version"],
        input_features,
        architecture,
        checkpoint_metadata,
    )
    ledger.synthetic_checkpoint_reads += 1

    base_cost = cost_results[str(blueprint["policy"]["base_cost_bps"])]
    all_cost_identities = all(
        np.allclose(
            values["net_return"],
            values["gross_return"] - values["cost"],
        )
        and np.allclose(
            values["cost"],
            values["turnover"] * (float(cost) / 10_000.0),
        )
        for cost, strategies in cost_results.items()
        for values in strategies.values()
    )
    operation_record = asdict(ledger)
    prohibited_operation_names = (
        "real_panel_or_label_reads",
        "real_training_epochs",
        "real_predictions",
        "real_performance_metrics",
        "real_pnl_evaluations",
        "target_asset_loads",
    )
    expected_raw_policy_scores = desired_raw_scores - desired_raw_scores.mean(
        axis=1, keepdims=True
    )
    expected_model_raw_excess = (
        reference_output["excess_return_z"].detach().numpy()
        - reference_output["excess_return_z"]
        .detach()
        .numpy()
        .mean(axis=1, keepdims=True)
    ) * excess_scale
    checks = {
        "all_v41_input_hashes_match": all(input_checks.values()),
        "v41_blueprint_and_audit_authorize_v42": v41_audit["passed"]
        and v41_result["decision"]
        == "authorize_v42_synthetic_ranking_excess_harness_only",
        "input_allowlist_contains_only_v41_metadata": set(paths)
        == {"v41_specification", "v41_blueprint", "v41_audit"},
        "actual_parameter_count_matches_analytic_and_frozen": parameter_count
        == analytic_parameter_count(architecture, input_features)
        == architecture["expected_parameter_count_for_nine_features"],
        "prediction_heads_and_shapes_are_exact": set(reference_output)
        == set(RANKING_EXCESS_HEADS)
        and all(
            tuple(reference_output[name].shape) == (batch_size, 3)
            for name in RANKING_EXCESS_HEADS
        ),
        "asset_permutation_equivariance_passes": permutation_passes,
        "causal_temporal_prefix_is_invariant": torch.allclose(
            temporal[:, :, :early_patch_count],
            altered_temporal[:, :, :early_patch_count],
            atol=1e-5,
            rtol=1e-5,
        ),
        "patch_mask_is_deterministic_and_exact": torch.equal(
            patch_mask, patch_mask_replay
        )
        and bool((patch_mask.flatten(1).sum(dim=1) == 14).all()),
        "reconstruction_shape_is_exact": tuple(
            pretrain_output["patch_reconstruction"].shape
        )
        == (batch_size, 3, 31, 16, input_features),
        "feature_scaler_is_train_only_and_replays": feature_scaler.state_sha256()
        == clean_feature_scaler.state_sha256()
        and feature_scaler.fit_scope == "representation_train_only"
        and feature_scaler.fit_rows == 8
        and np.isfinite(fixture.numpy()).all(),
        "train_only_excess_scale_is_finite_and_replays": math.isfinite(
            excess_scale
        )
        and excess_scale == scale_replay == scale_after_unseen_outlier
        and validation_outlier_returns.shape == all_scale_returns.shape
        and bool((validation_outlier_returns[~train_scale_mask] == 1e6).all()),
        "normalized_targets_are_zero_sum": torch.allclose(
            losses["target_z"].sum(dim=1),
            torch.zeros(batch_size),
            atol=1e-6,
        ),
        "centered_scores_are_zero_sum": torch.allclose(
            losses["centered_score"].sum(dim=1),
            torch.zeros(batch_size),
            atol=1e-6,
        ),
        "all_registered_losses_are_finite": all(
            bool(torch.isfinite(losses[name]))
            for name in ("ranking", "excess", "log_volatility", "core", "total")
        ),
        "all_three_pairs_are_used_per_nontied_sample": int(losses["pair_count"])
        == batch_size * 3,
        "critical_gradients_are_finite": pretrain_gradient_passes
        and supervised_gradient_passes,
        "gradient_norms_are_finite": math.isfinite(float(pretrain_grad_norm))
        and math.isfinite(float(supervised_grad_norm)),
        "exactly_two_synthetic_optimizer_steps": ledger.synthetic_optimizer_steps
        == int(synthetic["optimizer_steps"])
        == 2,
        "shared_asset_ridge_is_centered_and_finite": ridge_prediction.shape
        == (16, 3)
        and np.isfinite(ridge_prediction).all()
        and np.allclose(ridge_prediction.sum(axis=1), 0.0, atol=1e-10),
        "normalized_scores_convert_to_raw_before_policy": np.allclose(
            scores[eligibility],
            expected_raw_policy_scores[eligibility],
        )
        and np.allclose(model_raw_excess, expected_model_raw_excess)
        and np.isfinite(model_raw_excess).all(),
        "turnover_hurdle_and_cash_gate_match_fixture": observed_assets
        == expected_assets,
        "all_policies_respect_current_eligibility": not bool(
            candidate_positions[~eligibility].any()
        )
        and not bool(dual_positions[~eligibility].any())
        and not bool(equal_positions[~eligibility].any()),
        "candidate_turnover_includes_final_liquidation": math.isclose(
            float(base_cost["candidate"]["total_turnover"]), 8.0
        ),
        "positions_are_long_or_cash": bool((candidate_positions >= 0).all())
        and bool((candidate_positions.sum(axis=1) <= 1.0).all()),
        "all_cost_and_control_cells_are_present": set(cost_results)
        == {"10", "20", "30"}
        and all(
            set(strategies)
            == {"candidate", "dual_momentum_30", "momentum_gated_equal_weight"}
            for strategies in cost_results.values()
        ),
        "cost_and_net_identities_pass": all_cost_identities,
        "checkpoint_metadata_is_exact": checkpoint["metadata"]
        == checkpoint_metadata,
        "checkpoint_state_roundtrip_is_exact": _state_dict_sha256(loaded_model)
        == _state_dict_sha256(model)
        == checkpoint["state_dict_sha256"],
        "authorized_metadata_reads_are_runtime_counted": ledger.authorized_metadata_reads
        == len(paths)
        == 3,
        "real_data_and_target_operations_remain_zero": all(
            operation_record[name] == 0 for name in prohibited_operation_names
        ),
        "configuration_forbids_every_real_operation": all(
            value is False for value in harness["constraints"].values()
        ),
        "input_hashes_still_match_after_smoke": all(
            _sha256_file(paths[name]) == expected
            for name, expected in harness["expected_input_sha256"].items()
        ),
        "only_v43_pretraining_is_authorized": harness["authorized_next_action"]
        == "v43_medium_non_target_pretraining_only",
    }
    checks = {name: bool(passed) for name, passed in checks.items()}
    if not all(checks.values()):
        raise RuntimeError(f"V42 ranking/excess harness audit failed: {checks}")

    smoke = {
        "parameter_count": parameter_count,
        "optimizer_steps": ledger.synthetic_optimizer_steps,
        "excess_scale": excess_scale,
        "pair_count": int(losses["pair_count"]),
        "pair_accuracy": float(losses["pair_accuracy"]),
        "losses": {
            name: float(losses[name].detach())
            for name in ("ranking", "excess", "log_volatility", "core", "total")
        },
        "candidate_turnover": float(base_cost["candidate"]["total_turnover"]),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_state_sha256": checkpoint["state_dict_sha256"],
        "synthetic_only": True,
    }
    result = {
        "version": "v42",
        "decision": "authorize_v43_medium_non_target_pretraining_only",
        "harness_spec": harness_spec,
        "smoke": smoke,
        "operation_ledger": operation_record,
        "tested": {
            **{
                name: operation_record[name]
                for name in prohibited_operation_names
            },
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
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    return result
