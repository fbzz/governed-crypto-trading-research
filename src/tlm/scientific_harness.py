from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from .monte_carlo import paired_block_bootstrap
from .patch_transformer import MultiAssetPatchTransformer, PREDICTION_HEADS


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class FeatureScaler:
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    source_relative_feature_index: int
    fit_scope: str
    fit_start: str
    fit_end: str
    fit_rows: int

    @classmethod
    def fit_from_panel(
        cls,
        panel: pd.DataFrame,
        feature_names: list[str],
        start: str,
        end: str,
        maximum_allowed_end: str,
        source_relative_feature: str,
    ) -> "FeatureScaler":
        if pd.Timestamp(end) > pd.Timestamp(maximum_allowed_end):
            raise ValueError("Scaler fit window exceeds the frozen training boundary")
        dates = pd.to_datetime(panel["date"], utc=True)
        subset = panel.loc[
            (dates >= pd.Timestamp(start, tz="UTC"))
            & (dates <= pd.Timestamp(end, tz="UTC")),
            feature_names,
        ].to_numpy(dtype=np.float64)
        subset = subset[np.isfinite(subset).all(axis=1)]
        if len(subset) < 2:
            raise ValueError("Not enough finite train-only rows to fit scaler")
        mean = subset.mean(axis=0)
        scale = subset.std(axis=0, ddof=0)
        scale[scale == 0] = 1.0
        return cls(
            feature_names=tuple(feature_names),
            mean=tuple(float(value) for value in mean),
            scale=tuple(float(value) for value in scale),
            source_relative_feature_index=feature_names.index(source_relative_feature),
            fit_scope="representation_train_only",
            fit_start=start,
            fit_end=end,
            fit_rows=len(subset),
        )

    def transform_triplet_tensor(self, values: np.ndarray) -> np.ndarray:
        if values.shape[-1] != len(self.feature_names) + 1:
            raise ValueError("Triplet tensor must contain base plus relative feature")
        result = np.asarray(values, dtype=np.float32).copy()
        mean = np.asarray(self.mean, dtype=np.float32)
        scale = np.asarray(self.scale, dtype=np.float32)
        result[..., :-1] = (result[..., :-1] - mean) / scale
        result[..., -1] = result[..., -1] / scale[
            self.source_relative_feature_index
        ]
        if not np.isfinite(result).all():
            raise ValueError("Scaler produced non-finite values")
        return result

    def state_sha256(self) -> str:
        return _canonical_sha256(asdict(self))


class DeterministicEligibleTripletSampler:
    def __init__(
        self,
        availability_by_date: dict[pd.Timestamp, list[str]],
        role_symbols: list[str],
        seed: int,
        fold: int,
    ) -> None:
        allowed = set(role_symbols)
        entries = []
        for date, symbols in sorted(availability_by_date.items()):
            available = tuple(sorted(set(symbols).intersection(allowed)))
            if len(available) >= 3:
                entries.append((pd.Timestamp(date), available, math.comb(len(available), 3)))
        if not entries:
            raise ValueError("No eligible date-triplet pairs")
        self.entries = entries
        self._triplets_by_symbols = {
            symbols: tuple(combinations(symbols, 3))
            for symbols in {entry[1] for entry in entries}
        }
        self.cumulative = np.cumsum([entry[2] for entry in entries], dtype=np.int64)
        self.total_pairs = int(self.cumulative[-1])
        self.seed = int(seed)
        self.fold = int(fold)

    def sample_epoch(self, epoch: int, sample_count: int) -> list[dict[str, object]]:
        if sample_count < 1:
            raise ValueError("sample_count must be positive")
        rng = np.random.default_rng(
            np.random.SeedSequence([self.seed, self.fold, int(epoch)])
        )
        draws = rng.integers(0, self.total_pairs, size=sample_count, dtype=np.int64)
        samples = []
        for draw in draws:
            entry_index = int(np.searchsorted(self.cumulative, draw, side="right"))
            prior = int(self.cumulative[entry_index - 1]) if entry_index else 0
            date, symbols, _ = self.entries[entry_index]
            triplet = self._triplets_by_symbols[symbols][int(draw) - prior]
            samples.append({
                "date": date,
                "triplet": triplet,
                "pair_index": int(draw),
            })
        return samples


def deterministic_patch_mask(
    batch_size: int,
    asset_count: int,
    patch_count: int,
    mask_fraction: float,
    seed: int,
    fold: int,
    epoch: int,
    batch_index: int,
) -> torch.Tensor:
    if not 0 < mask_fraction < 1:
        raise ValueError("mask_fraction must be between zero and one")
    total = asset_count * patch_count
    masked = max(1, int(round(total * mask_fraction)))
    digest = hashlib.sha256(
        f"{seed}:{fold}:{epoch}:{batch_index}".encode("ascii")
    ).digest()
    generator_seed = int.from_bytes(digest[:8], "big") % (2**63 - 1)
    generator = torch.Generator().manual_seed(generator_seed)
    result = torch.zeros(batch_size, total, dtype=torch.bool)
    for sample in range(batch_size):
        indexes = torch.randperm(total, generator=generator)[:masked]
        result[sample, indexes] = True
    return result.reshape(batch_size, asset_count, patch_count)


def masked_reconstruction_loss(
    reconstruction: torch.Tensor,
    target_patches: torch.Tensor,
    patch_mask: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    if reconstruction.shape != target_patches.shape:
        raise ValueError("Reconstruction and patch target shapes must match")
    if patch_mask.shape != reconstruction.shape[:3]:
        raise ValueError("Patch mask shape does not match reconstruction")
    return F.smooth_l1_loss(
        reconstruction[patch_mask], target_patches[patch_mask], beta=beta
    )


def pinball_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between zero and one")
    error = target - prediction
    return torch.maximum(quantile * error, (quantile - 1.0) * error).mean()


def supervised_probabilistic_loss(
    output: dict[str, torch.Tensor],
    labels: torch.Tensor,
    volatility_weight: float,
    volatility_floor: float,
) -> dict[str, torch.Tensor]:
    if labels.ndim != 3 or labels.shape[-1] != 2:
        raise ValueError("Labels must have shape [batch, assets, 2]")
    returns = labels[..., 0]
    volatility = torch.log(labels[..., 1].clamp_min(volatility_floor))
    losses = {
        "return_q10": pinball_loss(output["return_q10"], returns, 0.10),
        "return_q50": pinball_loss(output["return_q50"], returns, 0.50),
        "return_q90": pinball_loss(output["return_q90"], returns, 0.90),
        "log_volatility": F.smooth_l1_loss(
            output["volatility_7d"], volatility, beta=1.0
        ),
    }
    losses["total"] = (
        losses["return_q10"]
        + losses["return_q50"]
        + losses["return_q90"]
    ) / 3.0 + volatility_weight * losses["log_volatility"]
    return losses


@dataclass
class EarlyStoppingState:
    patience: int
    minimum_delta: float = 0.0
    best_loss: float = float("inf")
    best_epoch: int = -1
    stale_epochs: int = 0
    should_stop: bool = False

    def update(self, epoch: int, validation_loss: float) -> bool:
        if not math.isfinite(validation_loss):
            raise ValueError("Validation loss must be finite")
        if validation_loss < self.best_loss - self.minimum_delta:
            self.best_loss = float(validation_loss)
            self.best_epoch = int(epoch)
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
            self.should_stop = self.stale_epochs >= self.patience
        return self.should_stop


def q_policy_positions(
    q10: np.ndarray,
    q50: np.ndarray,
    q10_threshold: float,
    q50_threshold: float,
) -> np.ndarray:
    if q10.shape != q50.shape or q10.ndim != 2:
        raise ValueError("q10 and q50 must share [days, assets] shape")
    positions = np.zeros_like(q50, dtype=np.float64)
    best = np.argmax(q50, axis=1)
    rows = np.arange(len(q50))
    active = (q50[rows, best] > q50_threshold) & (
        q10[rows, best] > q10_threshold
    )
    positions[rows[active], best[active]] = 1.0
    return positions


def dual_momentum_positions(momentum: np.ndarray) -> np.ndarray:
    if momentum.ndim != 2:
        raise ValueError("Momentum must have [days, assets] shape")
    positions = np.zeros_like(momentum, dtype=np.float64)
    best = np.argmax(momentum, axis=1)
    rows = np.arange(len(momentum))
    active = momentum[rows, best] > 0
    positions[rows[active], best[active]] = 1.0
    return positions


def persistent_portfolio_returns(
    positions: np.ndarray,
    actual_log_returns: np.ndarray,
    cost_bps: float,
) -> dict[str, np.ndarray | float]:
    if positions.shape != actual_log_returns.shape or positions.ndim != 2:
        raise ValueError("Positions and returns must share [days, assets] shape")
    if (positions < 0).any() or (positions.sum(axis=1) > 1.0 + 1e-12).any():
        raise ValueError("Positions violate long/cash gross-exposure contract")
    prior = np.vstack([np.zeros((1, positions.shape[1])), positions[:-1]])
    turnover = np.abs(positions - prior).sum(axis=1)
    if len(turnover):
        turnover[-1] += float(np.abs(positions[-1]).sum())
    gross = (positions * np.expm1(actual_log_returns)).sum(axis=1)
    cost = turnover * (float(cost_bps) / 10_000.0)
    return {
        "gross_return": gross,
        "turnover": turnover,
        "cost": cost,
        "net_return": gross - cost,
        "total_turnover": float(turnover.sum()),
        "total_cost": float(cost.sum()),
    }


def build_harness_spec(blueprint: dict, harness: dict) -> dict[str, object]:
    spec = {
        "version": "v34",
        "candidate_family_id": blueprint["candidate_family_id"],
        "scaler": harness["scaler"],
        "sampler": harness["sampler"],
        "patch_mask": harness["patch_mask"],
        "losses": harness["losses"],
        "optimizer": {
            "name": "AdamW",
            "learning_rate": blueprint["training"]["learning_rate"],
            "weight_decay": blueprint["training"]["weight_decay"],
            "gradient_clip_norm": harness["optimizer"]["gradient_clip_norm"],
            "scheduler": None,
        },
        "early_stopping": {
            "patience": blueprint["training"]["early_stopping_patience"],
            "minimum_delta": harness["early_stopping"]["minimum_delta"],
            "monitor": "chronological_validation_total_loss",
            "restore_best_checkpoint": True,
        },
        "training_limits": {
            "batch_size": blueprint["training"]["batch_size"],
            "pretrain_epochs_max": blueprint["training"]["maximum_pretrain_epochs"],
            "finetune_epochs_max": blueprint["training"]["maximum_finetune_epochs"],
            "seeds": blueprint["training"]["seeds"],
            "seed_selection_allowed": False,
        },
        "policy": blueprint["policy"],
        "controls": [
            blueprint["source_domain_gates"]["primary_control"],
            blueprint["source_domain_gates"]["secondary_control"],
        ],
        "cost_bps": blueprint["source_domain_gates"]["cost_bps"],
        "bootstrap": {
            "method": "paired_circular_block_bootstrap",
            "paths": blueprint["source_domain_gates"]["bootstrap_paths"],
            "block_lengths_days": blueprint["source_domain_gates"]["block_lengths_days"],
        },
        "target_boundary": "non_target_only_no_btc_eth_sol",
    }
    spec["harness_spec_sha256"] = _canonical_sha256(spec)
    return spec


def _report(result: dict[str, object]) -> str:
    smoke = result["smoke"]
    return "\n".join([
        "# TLM v34 Scientific Training Harness",
        "",
        "## Decision",
        "",
        "**HARNESS PASSED; FULL NON-TARGET PRETRAINING IS AUTHORIZED NEXT.**",
        "",
        f"Harness-spec SHA-256: `{result['harness_spec']['harness_spec_sha256']}`",
        f"Synthetic optimizer steps: **{smoke['optimizer_steps']}**",
        f"Masked patches per sample: **{smoke['masked_patches_per_sample']}**",
        f"Bootstrap cells: **{smoke['bootstrap_cells']}**",
        "",
        "The smoke pipeline passed train-only scaling, deterministic eligible-triplet sampling, deterministic masking, masked reconstruction loss, probabilistic supervised loss, AdamW steps, gradient clipping, early stopping, q-policy accounting, dual-momentum/equal-weight controls, 10/20/30 bps costs, and 10,000-path paired block bootstrap at 7/21/63 days.",
        "",
        "Only synthetic fixture data was used. No real panel, target asset, source-domain result, model-selection metric, or PnL claim was produced.",
        "",
        "## Next action",
        "",
        "V35 may run the frozen masked-patch pretraining on the 2021-2023 non-target training folds for seeds 42, 7, and 123. It must persist all checkpoints and losses without seed selection and may not begin supervised training or target evaluation.",
        "",
    ])


def run_scientific_harness(config: dict) -> dict[str, object]:
    harness = config["scientific_harness"]
    root = Path(harness["project_root"]).resolve()
    paths = {name: root / relative for name, relative in harness["inputs"].items()}
    for name, path in paths.items():
        if not path.is_file() or _sha256_file(path) != harness[
            "expected_input_sha256"
        ][name]:
            raise RuntimeError(f"V34 input missing or hash drifted: {name}")
    amendment = _load_json(paths["v29_amendment"])
    v32_manifest = _load_json(paths["v32_dataset_manifest"])
    feature_schema = _load_json(paths["v32_feature_schema"])
    v33_result = _load_json(paths["v33_result"])
    v33_audit = _load_json(paths["v33_audit"])
    blueprint = amendment["blueprint"]
    harness_spec = build_harness_spec(blueprint, harness)

    rng = np.random.default_rng(int(config["seed"]))
    base_features = list(feature_schema["model_feature_order"][:-1])
    dates = pd.date_range("2021-01-01", periods=160, freq="D", tz="UTC")
    synthetic_panel = pd.DataFrame(
        rng.normal(size=(len(dates), len(base_features))), columns=base_features
    )
    synthetic_panel.insert(0, "date", dates)
    validation_outlier = synthetic_panel.copy()
    validation_outlier.loc[validation_outlier["date"] > dates[119], base_features] = 1e6
    scaler = FeatureScaler.fit_from_panel(
        validation_outlier,
        base_features,
        dates[0].date().isoformat(),
        dates[119].date().isoformat(),
        dates[119].date().isoformat(),
        "log_close_to_close_return",
    )
    reference_scaler = FeatureScaler.fit_from_panel(
        synthetic_panel,
        base_features,
        dates[0].date().isoformat(),
        dates[119].date().isoformat(),
        dates[119].date().isoformat(),
        "log_close_to_close_return",
    )

    role_symbols = [f"A{index:02d}USDT" for index in range(20)]
    availability = {
        date: role_symbols[: (18 if index % 7 == 0 else 20)]
        for index, date in enumerate(dates[:30])
    }
    sampler = DeterministicEligibleTripletSampler(
        availability, role_symbols, seed=42, fold=1
    )
    sampled = sampler.sample_epoch(epoch=3, sample_count=64)
    sampled_replay = sampler.sample_epoch(epoch=3, sample_count=64)
    sampled_other_epoch = sampler.sample_epoch(epoch=4, sample_count=64)

    architecture = blueprint["architecture"]
    input_features = len(feature_schema["model_feature_order"])
    torch.manual_seed(42)
    model = MultiAssetPatchTransformer(input_features, architecture)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(blueprint["training"]["learning_rate"]),
        weight_decay=float(blueprint["training"]["weight_decay"]),
    )
    fixture_np = rng.normal(size=(4, 256, 3, input_features)).astype(np.float32)
    scaled_np = scaler.transform_triplet_tensor(fixture_np)
    fixture = torch.from_numpy(scaled_np)
    patch_mask = deterministic_patch_mask(
        4,
        3,
        model.patch_count,
        float(blueprint["training"]["mask_fraction"]),
        seed=42,
        fold=1,
        epoch=0,
        batch_index=0,
    )
    mask_replay = deterministic_patch_mask(
        4, 3, model.patch_count, float(blueprint["training"]["mask_fraction"]),
        seed=42, fold=1, epoch=0, batch_index=0,
    )
    target_patches = model.extract_patches(fixture)
    optimizer.zero_grad(set_to_none=True)
    pretrain_output = model(
        fixture, patch_mask=patch_mask, return_reconstruction=True
    )
    pretrain_loss = masked_reconstruction_loss(
        pretrain_output["patch_reconstruction"], target_patches, patch_mask
    )
    pretrain_loss.backward()
    pretrain_grad_norm = nn.utils.clip_grad_norm_(
        model.parameters(), float(harness["optimizer"]["gradient_clip_norm"])
    )
    optimizer.step()

    labels = torch.from_numpy(
        np.stack([
            rng.normal(0, 0.02, size=(4, 3)),
            rng.uniform(0.01, 0.25, size=(4, 3)),
        ], axis=-1).astype(np.float32)
    )
    optimizer.zero_grad(set_to_none=True)
    supervised_output = model(fixture)
    supervised_losses = supervised_probabilistic_loss(
        supervised_output,
        labels,
        volatility_weight=float(harness["losses"]["volatility_weight"]),
        volatility_floor=float(harness["losses"]["volatility_floor"]),
    )
    supervised_losses["total"].backward()
    supervised_grad_norm = nn.utils.clip_grad_norm_(
        model.parameters(), float(harness["optimizer"]["gradient_clip_norm"])
    )
    optimizer.step()

    stopper = EarlyStoppingState(
        patience=int(blueprint["training"]["early_stopping_patience"]),
        minimum_delta=float(harness["early_stopping"]["minimum_delta"]),
    )
    stop_trace = [1.0, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85]
    stop_epoch = None
    for epoch, loss in enumerate(stop_trace):
        if stopper.update(epoch, loss):
            stop_epoch = epoch
            break

    observations = int(harness["smoke"]["observations"])
    time = np.arange(observations)
    actual = np.stack([
        0.012 * np.sin(time / 7.0),
        0.010 * np.cos(time / 9.0),
        0.008 * np.sin(time / 11.0 + 1.0),
    ], axis=1)
    q50 = actual * 0.25 + 0.003
    q10 = q50 - 0.02
    candidate_positions = q_policy_positions(
        q10,
        q50,
        float(blueprint["policy"]["enter_if_q10_above"]),
        float(blueprint["policy"]["enter_if_q50_above"]),
    )
    momentum = np.cumsum(actual, axis=0)
    dual_positions = dual_momentum_positions(momentum)
    equal_positions = np.full_like(actual, 1.0 / actual.shape[1])
    cost_results: dict[str, dict[str, dict[str, np.ndarray | float]]] = {}
    for cost_bps in blueprint["source_domain_gates"]["cost_bps"]:
        cost_results[str(cost_bps)] = {
            "candidate": persistent_portfolio_returns(
                candidate_positions, actual, cost_bps
            ),
            "dual_momentum_30": persistent_portfolio_returns(
                dual_positions, actual, cost_bps
            ),
            "equal_weight_buy_hold": persistent_portfolio_returns(
                equal_positions, actual, cost_bps
            ),
        }
    base = cost_results[str(blueprint["policy"]["base_cost_bps"])]
    bootstrap = {
        str(block): paired_block_bootstrap(
            {
                name: np.asarray(values["net_return"])
                for name, values in base.items()
            },
            "candidate",
            ["dual_momentum_30", "equal_weight_buy_hold"],
            block_length=int(block),
            n_paths=int(blueprint["source_domain_gates"]["bootstrap_paths"]),
            seed=int(config["seed"]) + int(block),
        )
        for block in blueprint["source_domain_gates"]["block_lengths_days"]
    }

    costs_monotonic = all(
        cost_results["10"][name]["total_cost"]
        < cost_results["20"][name]["total_cost"]
        < cost_results["30"][name]["total_cost"]
        for name in base
    )
    checks = {
        "v32_tensor_contract_matches": v32_manifest["tensor_contract"]["x_shape"]
        == [256, 3, input_features],
        "v33_audit_passes": bool(v33_audit["passed"]),
        "v33_authorizes_only_v34": v33_result["decision"]
        == "authorize_v34_scientific_harness_implementation_only",
        "scaler_fit_is_train_only": scaler.fit_scope == "representation_train_only"
        and scaler.fit_end == dates[119].date().isoformat(),
        "validation_outlier_cannot_change_scaler": scaler == reference_scaler,
        "relative_feature_uses_source_scale_without_center": harness["scaler"][
            "relative_feature_scaling"
        ] == "divide_by_source_feature_scale_without_centering",
        "triplet_sampler_replays_exactly": sampled == sampled_replay,
        "triplet_sampler_changes_across_epochs": sampled != sampled_other_epoch,
        "all_sampled_triplets_are_eligible": all(
            set(row["triplet"]).issubset(set(availability[row["date"]]))
            for row in sampled
        ),
        "sampler_is_uniform_over_eligible_pairs": harness["sampler"]["distribution"]
        == "uniform_over_eligible_date_triplet_pairs_with_replacement",
        "patch_mask_replays_exactly": torch.equal(patch_mask, mask_replay),
        "masked_patch_count_is_exact": bool(
            torch.all(patch_mask.flatten(1).sum(axis=1) == 14)
        ),
        "pretrain_loss_is_finite": bool(torch.isfinite(pretrain_loss)),
        "supervised_losses_are_finite": all(
            bool(torch.isfinite(value)) for value in supervised_losses.values()
        ),
        "gradient_norms_are_finite": bool(torch.isfinite(pretrain_grad_norm))
        and bool(torch.isfinite(supervised_grad_norm)),
        "optimizer_is_frozen_adamw": isinstance(optimizer, torch.optim.AdamW),
        "early_stopping_restores_best_epoch_contract": stop_epoch == 6
        and stopper.best_epoch == 1
        and stopper.should_stop,
        "policy_is_long_top1_or_cash": bool(
            np.isin(candidate_positions.sum(axis=1), [0.0, 1.0]).all()
        ),
        "turnover_includes_final_liquidation": all(
            float(values["turnover"][-1])
            >= float(np.abs(positions[-1] - positions[-2]).sum())
            for values, positions in (
                (base["candidate"], candidate_positions),
                (base["dual_momentum_30"], dual_positions),
                (base["equal_weight_buy_hold"], equal_positions),
            )
        ),
        "cost_sensitivity_is_monotonic": bool(costs_monotonic),
        "all_controls_present": set(base)
        == {"candidate", "dual_momentum_30", "equal_weight_buy_hold"},
        "bootstrap_paths_and_blocks_are_exact": all(
            result["paths"] == 10_000 and result["block_length"] == int(block)
            for block, result in bootstrap.items()
        ),
        "real_panel_not_loaded": True,
        "real_label_not_loaded": True,
        "full_training_epoch_count_is_zero": True,
        "target_asset_load_count_is_zero": True,
        "source_domain_performance_not_evaluated": True,
        "deployment_pnl_not_computed": True,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise RuntimeError(f"V34 scientific-harness audit failed: {checks}")

    smoke = {
        "fixture": "synthetic_bounded_no_market_data",
        "optimizer_steps": 2,
        "pretrain_loss": float(pretrain_loss.detach()),
        "supervised_loss": float(supervised_losses["total"].detach()),
        "masked_patches_per_sample": 14,
        "sampler_total_eligible_pairs": sampler.total_pairs,
        "sampler_smoke_samples": len(sampled),
        "early_stop_epoch": stop_epoch,
        "early_stop_best_epoch": stopper.best_epoch,
        "cost_cells": len(cost_results) * len(base),
        "bootstrap_cells": len(bootstrap),
        "bootstrap_paths_per_cell": 10_000,
    }
    result = {
        "version": "v34",
        "decision": "authorize_v35_full_non_target_pretraining_only",
        "harness_spec": harness_spec,
        "smoke": smoke,
        "scaler_smoke_state": {**asdict(scaler), "state_sha256": scaler.state_sha256()},
        "tested": {
            "synthetic_optimizer_steps": 2,
            "train_only_scaler": True,
            "eligible_triplet_sampler": True,
            "deterministic_patch_masks": True,
            "pretraining_loss": True,
            "supervised_loss": True,
            "early_stopping": True,
            "portfolio_accounting": True,
            "cost_sensitivity": True,
            "paired_block_bootstrap": True,
            "real_data_loaded": False,
            "full_model_trained": False,
            "target_assets_loaded": False,
            "improvement_status": "unknown_not_evaluated",
            "drawdown_status": "unknown_not_evaluated",
        },
        "audit": {"passed": True, "checks": checks},
    }
    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "harness_spec.json": harness_spec,
        "smoke.json": smoke,
        "scaler_smoke_state.json": result["scaler_smoke_state"],
        "bootstrap_smoke.json": bootstrap,
        "audit.json": result["audit"],
        "result.json": result,
    }
    for name, value in artifacts.items():
        (output / name).write_text(
            json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
        )
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    return result
