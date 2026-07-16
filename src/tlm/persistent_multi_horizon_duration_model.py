from __future__ import annotations

from itertools import combinations
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


CANDIDATE_FAMILY_ID = "tlm_persistent_multi_horizon_duration_v1"
DEFAULT_HORIZONS = (1, 3, 7)


def default_persistent_duration_architecture() -> dict[str, Any]:
    """Return the unregistered post-V72 candidate architecture."""

    return {
        "input_features": 9,
        "lookback_days": 256,
        "input_triplet_size": 3,
        "patch_length_days": 16,
        "patch_stride_days": 8,
        "d_model": 128,
        "temporal_encoder_layers": 4,
        "cross_asset_attention_layers": 1,
        "attention_heads": 8,
        "feed_forward_width": 512,
        "dropout": 0.15,
        "output_horizons": list(DEFAULT_HORIZONS),
        "maximum_duration_days": 7,
        "student_t_degrees_of_freedom": 5.0,
        "scale_floor": 1e-4,
    }


class PersistentMultiHorizonDurationTransformer(nn.Module):
    """Causal multi-asset forecaster with an explicit holding-duration head.

    The shared temporal encoder and cross-asset encoder have no asset-slot
    embeddings. Asset-specific outputs are therefore permutation equivariant,
    while the market outputs are permutation invariant.
    """

    def __init__(self, architecture: dict[str, Any]) -> None:
        super().__init__()
        self.input_features = int(architecture["input_features"])
        self.lookback_days = int(architecture["lookback_days"])
        self.triplet_size = int(architecture["input_triplet_size"])
        self.patch_length = int(architecture["patch_length_days"])
        self.patch_stride = int(architecture["patch_stride_days"])
        self.d_model = int(architecture["d_model"])
        self.horizons = tuple(int(value) for value in architecture["output_horizons"])
        self.maximum_duration_days = int(architecture["maximum_duration_days"])
        self.degrees_of_freedom = float(
            architecture["student_t_degrees_of_freedom"]
        )
        self.scale_floor = float(architecture["scale_floor"])
        self.patch_count = (
            (self.lookback_days - self.patch_length) // self.patch_stride + 1
        )
        attention_heads = int(architecture["attention_heads"])

        if self.triplet_size < 2:
            raise ValueError("The persistent-duration model needs at least two assets")
        if self.patch_count < 1:
            raise ValueError("Patch geometry does not fit the lookback")
        if self.d_model % attention_heads:
            raise ValueError("d_model must be divisible by attention_heads")
        if not self.horizons or tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("Output horizons must be unique and increasing")
        if self.horizons[0] < 1 or self.horizons[-1] > self.maximum_duration_days:
            raise ValueError("Output horizons must fit the explicit-duration support")
        if self.degrees_of_freedom <= 2.0:
            raise ValueError("Student-t degrees of freedom must exceed two")
        if self.scale_floor <= 0.0:
            raise ValueError("Scale floor must be positive")

        patch_width = self.patch_length * self.input_features
        self.patch_projection = nn.Linear(patch_width, self.d_model)
        self.temporal_position = nn.Parameter(
            torch.zeros(1, self.patch_count, self.d_model)
        )
        self.patch_norm = nn.LayerNorm(self.d_model)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=attention_heads,
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
        cross_asset_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=attention_heads,
            dim_feedforward=int(architecture["feed_forward_width"]),
            dropout=float(architecture["dropout"]),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross_asset_encoder = nn.TransformerEncoder(
            cross_asset_layer,
            num_layers=int(architecture["cross_asset_attention_layers"]),
            enable_nested_tensor=False,
        )
        self.temporal_norm = nn.LayerNorm(self.d_model)
        self.cross_asset_norm = nn.LayerNorm(self.d_model)

        fusion_width = 2 * self.d_model
        self.fusion_gate = nn.Linear(fusion_width, self.d_model)
        self.fusion_candidate = nn.Linear(fusion_width, self.d_model)
        self.fusion_norm = nn.LayerNorm(self.d_model)

        horizon_count = len(self.horizons)
        self.excess_location_head = nn.Linear(self.d_model, horizon_count)
        self.excess_scale_head = nn.Linear(self.d_model, horizon_count)
        self.market_distribution_head = nn.Linear(self.d_model, 2 * horizon_count)
        self.duration_hazard_head = nn.Linear(
            self.d_model, self.maximum_duration_days
        )

        causal_mask = torch.triu(
            torch.ones(self.patch_count, self.patch_count, dtype=torch.bool),
            diagonal=1,
        )
        horizon_indexes = torch.tensor(
            [horizon - 1 for horizon in self.horizons], dtype=torch.long
        )
        self.register_buffer("causal_patch_mask", causal_mask, persistent=False)
        self.register_buffer("horizon_indexes", horizon_indexes, persistent=False)
        nn.init.normal_(self.temporal_position, mean=0.0, std=0.02)

    def _validate_input(self, features: torch.Tensor) -> None:
        expected = (
            self.lookback_days,
            self.triplet_size,
            self.input_features,
        )
        if features.ndim != 4 or tuple(features.shape[1:]) != expected:
            raise ValueError(
                "Input must have [batch,time,assets,features] shape; "
                f"expected [batch,{expected[0]},{expected[1]},{expected[2]}], "
                f"received {tuple(features.shape)}"
            )
        if features.dtype != torch.float32:
            raise ValueError("Model input dtype must be float32")
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
        tokens = self.patch_norm(
            tokens + self.temporal_position[:, :patch_count].unsqueeze(1)
        )
        encoded = self.temporal_encoder(
            tokens.reshape(batch * assets, patch_count, self.d_model),
            mask=self.causal_patch_mask[:patch_count, :patch_count],
        )
        return self.temporal_norm(encoded).reshape(
            batch, assets, patch_count, self.d_model
        )

    def encode_assets(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        temporal = self.encode_temporal_patches(features)
        assets = self.cross_asset_encoder(temporal[:, :, -1, :])
        assets = self.cross_asset_norm(assets)
        market = assets.mean(dim=1)
        market_per_asset = market.unsqueeze(1).expand_as(assets)
        fusion_input = torch.cat((assets, market_per_asset), dim=-1)
        gate = torch.sigmoid(self.fusion_gate(fusion_input))
        candidate = F.gelu(self.fusion_candidate(fusion_input))
        fused = self.fusion_norm(gate * candidate + (1.0 - gate) * assets)
        return fused, market

    @staticmethod
    def _broadcast_cost(
        reference: torch.Tensor, round_trip_cost: float | torch.Tensor
    ) -> torch.Tensor:
        cost = torch.as_tensor(
            round_trip_cost, dtype=reference.dtype, device=reference.device
        )
        if not bool(torch.isfinite(cost).all()) or bool((cost < 0).any()):
            raise ValueError("Round-trip cost must be finite and nonnegative")
        try:
            return torch.broadcast_to(cost, reference.shape)
        except RuntimeError as exc:
            raise ValueError(
                "Round-trip cost must broadcast to [batch,assets,horizons]"
            ) from exc

    def forward(
        self,
        features: torch.Tensor,
        *,
        round_trip_cost: float | torch.Tensor = 0.0,
    ) -> dict[str, torch.Tensor]:
        asset_state, market_state = self.encode_assets(features)

        raw_excess_location = self.excess_location_head(asset_state)
        excess_location = raw_excess_location - raw_excess_location.mean(
            dim=1, keepdim=True
        )
        raw_excess_scale = self.excess_scale_head(asset_state)
        excess_scale = F.softplus(raw_excess_scale) + self.scale_floor

        raw_market_distribution = self.market_distribution_head(market_state)
        market_location, raw_market_scale = raw_market_distribution.chunk(2, dim=-1)
        market_scale = F.softplus(raw_market_scale) + self.scale_floor

        gross_location = market_location.unsqueeze(1) + excess_location
        gross_scale = torch.sqrt(
            market_scale.unsqueeze(1).square() + excess_scale.square()
        )
        cost = self._broadcast_cost(gross_location, round_trip_cost)
        net_location = gross_location - cost

        hazard_logits = self.duration_hazard_head(asset_state)
        hazard_probability = torch.sigmoid(hazard_logits)
        survival_probability = torch.cumprod(
            1.0 - hazard_probability, dim=-1
        )
        horizon_survival_probability = survival_probability.index_select(
            dim=-1, index=self.horizon_indexes
        )

        return {
            "excess_location": excess_location,
            "raw_excess_scale": raw_excess_scale,
            "excess_scale": excess_scale,
            "market_location": market_location,
            "raw_market_scale": raw_market_scale,
            "market_scale": market_scale,
            "gross_location": gross_location,
            "gross_scale": gross_scale,
            "net_location": net_location,
            "hazard_logits": hazard_logits,
            "hazard_probability": hazard_probability,
            "survival_probability": survival_probability,
            "horizon_survival_probability": horizon_survival_probability,
            "expected_holding_days": survival_probability.sum(dim=-1),
            "persistent_net_score": net_location * horizon_survival_probability,
        }


def explicit_duration_negative_log_likelihood(
    hazard_logits: torch.Tensor,
    duration_days: torch.Tensor,
    censored: torch.Tensor,
    *,
    duration_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Discrete-time event likelihood with right censoring.

    ``duration_days`` is the first exit day for observed events. For censored
    rows it is the final day through which the state remained alive.
    """

    if hazard_logits.ndim != 3:
        raise ValueError("Hazard logits must have [batch,assets,duration] shape")
    if duration_days.shape != hazard_logits.shape[:-1]:
        raise ValueError("Duration targets must have [batch,assets] shape")
    if censored.shape != duration_days.shape or censored.dtype != torch.bool:
        raise ValueError("Censor flags must be boolean [batch,assets]")
    if duration_mask is None:
        duration_mask = torch.ones_like(censored)
    if duration_mask.shape != duration_days.shape or duration_mask.dtype != torch.bool:
        raise ValueError("Duration mask must be boolean [batch,assets]")
    if not bool(duration_mask.any()):
        raise ValueError("At least one duration target must be active")

    maximum_duration = hazard_logits.shape[-1]
    active_durations = duration_days[duration_mask]
    if bool((active_durations < 1).any()) or bool(
        (active_durations > maximum_duration).any()
    ):
        raise ValueError("Active durations must fit the modeled duration support")

    day = torch.arange(
        1, maximum_duration + 1, device=hazard_logits.device
    ).view(1, 1, -1)
    target_day = duration_days.unsqueeze(-1)
    survived = torch.where(censored.unsqueeze(-1), day <= target_day, day < target_day)
    exited = (~censored).unsqueeze(-1) & (day == target_day)
    per_row = (
        F.softplus(hazard_logits) * survived
        + F.softplus(-hazard_logits) * exited
    ).sum(dim=-1)
    return per_row[duration_mask].mean()


def persistent_multi_task_loss(
    output: dict[str, torch.Tensor],
    return_targets: torch.Tensor,
    duration_days: torch.Tensor,
    duration_censored: torch.Tensor,
    *,
    return_mask: torch.Tensor | None = None,
    duration_mask: torch.Tensor | None = None,
    degrees_of_freedom: float = 5.0,
    return_nll_weight: float = 1.0,
    ranking_weight: float = 0.25,
    duration_weight: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Joint probabilistic-return, relative-ranking and duration objective."""

    location = output["gross_location"]
    scale = output["gross_scale"]
    if return_targets.shape != location.shape or scale.shape != location.shape:
        raise ValueError("Return tensors must share [batch,assets,horizons] shape")
    if return_mask is None:
        return_mask = torch.isfinite(return_targets)
    if return_mask.shape != return_targets.shape or return_mask.dtype != torch.bool:
        raise ValueError("Return mask must be boolean [batch,assets,horizons]")
    if not bool(return_mask.any()):
        raise ValueError("At least one return target must be active")
    if not bool(torch.isfinite(return_targets[return_mask]).all()):
        raise ValueError("Active return targets must be finite")

    safe_targets = torch.where(
        return_mask, return_targets, torch.zeros_like(return_targets)
    )
    distribution = torch.distributions.StudentT(
        df=float(degrees_of_freedom), loc=location, scale=scale
    )
    return_nll = -distribution.log_prob(safe_targets)[return_mask].mean()

    pair_losses: list[torch.Tensor] = []
    pair_count = 0
    for left, right in combinations(range(location.shape[1]), 2):
        active = return_mask[:, left] & return_mask[:, right]
        target_difference = safe_targets[:, left] - safe_targets[:, right]
        active = active & (target_difference.abs() > 1e-12)
        if bool(active.any()):
            predicted_difference = location[:, left] - location[:, right]
            pair_losses.append(
                F.softplus(
                    -target_difference[active].sign()
                    * predicted_difference[active]
                )
            )
            pair_count += int(active.sum())
    ranking = (
        torch.cat(pair_losses).mean()
        if pair_losses
        else location.sum() * 0.0
    )

    duration_nll = explicit_duration_negative_log_likelihood(
        output["hazard_logits"],
        duration_days,
        duration_censored,
        duration_mask=duration_mask,
    )
    total = (
        float(return_nll_weight) * return_nll
        + float(ranking_weight) * ranking
        + float(duration_weight) * duration_nll
    )
    return {
        "return_nll": return_nll,
        "ranking": ranking,
        "duration_nll": duration_nll,
        "total": total,
        "pair_count": torch.tensor(pair_count, device=location.device),
    }
