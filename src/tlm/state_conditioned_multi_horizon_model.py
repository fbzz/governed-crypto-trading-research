from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .core import canonical_sha256


HORIZONS = (1, 3, 7)
QUANTILES = (0.2, 0.5, 0.8)
PAIR_INDEXES = ((0, 1), (0, 2), (1, 2))


class StateConditionedMultiHorizonTransformer(nn.Module):
    def __init__(self, architecture: dict[str, Any]) -> None:
        super().__init__()
        self.input_features = int(architecture["input_features"])
        self.lookback_days = int(architecture["lookback_days"])
        self.triplet_size = int(architecture["input_triplet_size"])
        self.patch_length = int(architecture["patch_length_days"])
        self.patch_stride = int(architecture["patch_stride_days"])
        self.d_model = int(architecture["d_model"])
        self.horizons = tuple(int(value) for value in architecture["output_horizons"])
        self.quantiles = tuple(float(value) for value in architecture["output_quantiles"])
        if architecture.get("variant_count") != 1:
            raise ValueError("V55 freezes exactly one architecture variant")
        if architecture.get("shared_asset_encoder") is not True:
            raise ValueError("V55 requires a shared asset encoder")
        if architecture.get("causal_inference_mask") is not True:
            raise ValueError("V55 requires causal temporal attention")
        if architecture.get("asset_slot_embedding") is not False:
            raise ValueError("V55 forbids asset-slot embeddings")
        if architecture.get("output_head") != "shared_per_asset_nine_quantiles":
            raise ValueError("V55 output-head contract drift")
        if self.horizons != HORIZONS or self.quantiles != QUANTILES:
            raise ValueError("V55 output horizon/quantile contract drift")
        if self.triplet_size != 3:
            raise ValueError("V55 requires an exact three-asset triplet")
        self.patch_count = (
            (self.lookback_days - self.patch_length) // self.patch_stride + 1
        )
        heads = int(architecture["attention_heads"])
        if self.patch_count < 1 or self.d_model % heads:
            raise ValueError("Invalid frozen patch or attention geometry")

        patch_width = self.patch_length * self.input_features
        self.patch_projection = nn.Linear(patch_width, self.d_model)
        self.temporal_position = nn.Parameter(
            torch.zeros(1, self.patch_count, self.d_model)
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=heads,
            dim_feedforward=int(architecture["feed_forward_width"]),
            dropout=float(architecture["dropout"]),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=int(architecture["temporal_encoder_layers"]),
            enable_nested_tensor=False,
        )
        cross_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=heads,
            dim_feedforward=int(architecture["feed_forward_width"]),
            dropout=float(architecture["dropout"]),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross_asset_encoder = nn.TransformerEncoder(
            cross_layer,
            num_layers=int(architecture["cross_asset_attention_layers"]),
            enable_nested_tensor=False,
        )
        self.temporal_norm = nn.LayerNorm(self.d_model)
        self.cross_asset_norm = nn.LayerNorm(self.d_model)
        self.quantile_head = nn.Linear(
            self.d_model, len(self.horizons) * len(self.quantiles)
        )
        causal_mask = torch.triu(
            torch.ones(self.patch_count, self.patch_count, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_patch_mask", causal_mask, persistent=False)
        nn.init.normal_(self.temporal_position, mean=0.0, std=0.02)

    def _validate_input(self, features: torch.Tensor) -> None:
        expected = (
            self.lookback_days,
            self.triplet_size,
            self.input_features,
        )
        if features.ndim != 4 or tuple(features.shape[1:]) != expected:
            raise ValueError(
                "Input must have frozen [batch,256,3,9] shape; "
                f"received {tuple(features.shape)}"
            )
        if features.dtype != torch.float32:
            raise ValueError("V55 input dtype must be float32")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("Model inputs must be finite")

    def extract_patches(self, features: torch.Tensor) -> torch.Tensor:
        self._validate_input(features)
        patches = features.unfold(1, self.patch_length, self.patch_stride)
        return patches.permute(0, 2, 1, 4, 3).contiguous()

    def encode_temporal_patches(self, features: torch.Tensor) -> torch.Tensor:
        patches = self.extract_patches(features)
        batch, assets, patch_count, _, _ = patches.shape
        tokens = self.patch_projection(patches.flatten(start_dim=3))
        tokens = tokens + self.temporal_position[:, :patch_count].unsqueeze(1)
        encoded = self.temporal_encoder(
            tokens.reshape(batch * assets, patch_count, self.d_model),
            mask=self.causal_patch_mask[:patch_count, :patch_count],
        )
        return self.temporal_norm(encoded).reshape(
            batch, assets, patch_count, self.d_model
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        temporal = self.encode_temporal_patches(features)
        cross = self.cross_asset_encoder(temporal[:, :, -1, :])
        cross = self.cross_asset_norm(cross)
        return self.quantile_head(cross).reshape(
            features.shape[0],
            self.triplet_size,
            len(self.horizons),
            len(self.quantiles),
        )


def multi_horizon_quantile_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    target_mask: torch.Tensor | None = None,
    tie_tolerance: float = 1e-12,
    pinball_weight: float = 1.0,
    ranking_weight: float = 0.5,
    crossing_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    if predictions.ndim != 4 or tuple(predictions.shape[1:]) != (3, 3, 3):
        raise ValueError("Predictions must have [batch,3,3,3] shape")
    if targets.shape != predictions.shape[:-1]:
        raise ValueError("Targets must have [batch,3,3] shape")
    if tie_tolerance < 0:
        raise ValueError("Tie tolerance cannot be negative")
    if target_mask is None:
        target_mask = torch.isfinite(targets)
    if target_mask.shape != targets.shape or target_mask.dtype != torch.bool:
        raise ValueError("Target mask must be boolean [batch,3,3]")
    if not bool(target_mask.any()):
        raise ValueError("At least one synthetic target must be active")
    if not bool(torch.isfinite(predictions).all()):
        raise ValueError("Predictions must be finite")
    if not bool(torch.isfinite(targets[target_mask]).all()):
        raise ValueError("Active targets must be finite")

    safe_targets = torch.where(target_mask, targets, torch.zeros_like(targets))
    error = safe_targets.unsqueeze(-1) - predictions
    quantile_values = predictions.new_tensor(QUANTILES).view(1, 1, 1, 3)
    pinball_cells = torch.maximum(
        quantile_values * error, (quantile_values - 1.0) * error
    )
    pinball_mask = target_mask.unsqueeze(-1).expand_as(pinball_cells)
    pinball = pinball_cells[pinball_mask].mean()

    h7_q50 = predictions[:, :, 2, 1]
    h7_target = safe_targets[:, :, 2]
    h7_mask = target_mask[:, :, 2]
    pair_losses: list[torch.Tensor] = []
    pair_count = 0
    for left, right in PAIR_INDEXES:
        difference = h7_target[:, left] - h7_target[:, right]
        active = h7_mask[:, left] & h7_mask[:, right] & (
            difference.abs() > tie_tolerance
        )
        if bool(active.any()):
            sign = difference[active].sign()
            predicted_difference = (
                h7_q50[:, left] - h7_q50[:, right]
            )[active]
            pair_losses.append(F.softplus(-sign * predicted_difference))
            pair_count += int(active.sum())
    ranking = (
        torch.cat(pair_losses).mean()
        if pair_losses
        else predictions.sum() * 0.0
    )
    crossing = (
        F.relu(predictions[..., 0] - predictions[..., 1])
        + F.relu(predictions[..., 1] - predictions[..., 2])
    ).mean()
    total = (
        float(pinball_weight) * pinball
        + float(ranking_weight) * ranking
        + float(crossing_weight) * crossing
    )
    return {
        "pinball": pinball,
        "ranking": ranking,
        "crossing": crossing,
        "total": total,
        "pair_count": torch.tensor(pair_count, device=predictions.device),
    }


def state_conditioned_weekly_policy(
    h7_q20: np.ndarray,
    eligible: np.ndarray,
    *,
    risky_weight: float = 1.0 / 3.0,
    base_cost: float = 0.001,
    decision_interval: int = 7,
) -> dict[str, np.ndarray]:
    forecasts = np.asarray(h7_q20, dtype=np.float64)
    eligibility = np.asarray(eligible, dtype=bool)
    if forecasts.ndim != 2 or forecasts.shape[1] != 3:
        raise ValueError("Forecasts must have [eligible dates,3] shape")
    if eligibility.shape != forecasts.shape:
        raise ValueError("Eligibility must match forecast shape")
    if not 0 < risky_weight <= 1:
        raise ValueError("Risky weight must be in (0,1]")
    if not math.isfinite(base_cost) or base_cost < 0:
        raise ValueError("Base cost must be finite and nonnegative")
    if decision_interval < 1:
        raise ValueError("Decision interval must be positive")

    positions = np.zeros_like(forecasts)
    decision_mask = np.zeros(len(forecasts), dtype=bool)
    forced_cash_mask = np.zeros(len(forecasts), dtype=bool)
    current = np.zeros(3, dtype=np.float64)
    eligible_dates_since_decision = 0
    has_decided = False

    for day in range(len(forecasts)):
        valid = eligibility[day] & np.isfinite(forecasts[day])
        incumbent = int(np.argmax(current)) if current.sum() > 0 else None
        if incumbent is not None and not valid[incumbent]:
            current = np.zeros(3, dtype=np.float64)
            forced_cash_mask[day] = True

        # V56 freezes an eligible signal date as one exact triplet-ready row.
        eligible_date = bool(valid.all())
        should_decide = False
        if eligible_date:
            if not has_decided:
                should_decide = True
            else:
                eligible_dates_since_decision += 1
                should_decide = eligible_dates_since_decision >= decision_interval

        if should_decide and not forced_cash_mask[day]:
            candidates: list[np.ndarray] = [current.copy()]
            cash = np.zeros(3, dtype=np.float64)
            if not np.array_equal(current, cash):
                candidates.append(cash)
            for asset in np.flatnonzero(valid):
                action = np.zeros(3, dtype=np.float64)
                action[int(asset)] = risky_weight
                if not any(np.array_equal(action, item) for item in candidates):
                    candidates.append(action)
            best = candidates[0]
            best_utility = float(np.dot(best, forecasts[day])) - base_cost * float(
                np.abs(best - current).sum()
            )
            for candidate in candidates[1:]:
                utility = float(np.dot(candidate, forecasts[day])) - base_cost * float(
                    np.abs(candidate - current).sum()
                )
                if utility > best_utility and not math.isclose(
                    utility, best_utility, rel_tol=0.0, abs_tol=1e-15
                ):
                    best = candidate
                    best_utility = utility
            current = best.copy()
            decision_mask[day] = True
            has_decided = True
            eligible_dates_since_decision = 0
        positions[day] = current

    return {
        "positions": positions,
        "decision_mask": decision_mask,
        "forced_cash_mask": forced_cash_mask,
    }


def save_state_conditioned_checkpoint(
    path: str | Path,
    payload: dict[str, Any],
    *,
    format_version: str,
) -> None:
    required = {
        "model_state",
        "best_model_state",
        "optimizer_state",
        "cpu_rng_state",
        "mps_rng_state",
        "early_stopping_state",
        "history",
        "metadata",
        "architecture",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Checkpoint payload missing: {missing}")
    serializable = dict(payload)
    serializable["format_version"] = format_version
    serializable["architecture_sha256"] = canonical_sha256(payload["architecture"])
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(serializable, destination)


def load_state_conditioned_checkpoint(
    path: str | Path,
    *,
    expected_format_version: str,
    expected_architecture: dict[str, Any],
    expected_metadata: dict[str, Any],
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format_version") != expected_format_version:
        raise ValueError("Checkpoint format drift")
    if payload.get("architecture_sha256") != canonical_sha256(expected_architecture):
        raise ValueError("Checkpoint architecture hash drift")
    if payload.get("architecture") != expected_architecture:
        raise ValueError("Checkpoint architecture mismatch")
    if payload.get("metadata") != expected_metadata:
        raise ValueError("Checkpoint metadata mismatch")
    return payload
