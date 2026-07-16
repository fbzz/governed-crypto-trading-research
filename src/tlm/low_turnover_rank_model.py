from __future__ import annotations

from itertools import combinations
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def deterministic_feature_layer_norm(
    values: torch.Tensor, norm: nn.LayerNorm
) -> torch.Tensor:
    """Algebraic LayerNorm avoiding a non-finite deterministic MPS backward."""

    mean = values.mean(dim=-1, keepdim=True)
    centered = values - mean
    variance = centered.square().mean(dim=-1, keepdim=True)
    scaled = centered * torch.rsqrt(variance + norm.eps)
    if norm.elementwise_affine:
        scaled = scaled * norm.weight + norm.bias
    return scaled


class CausalDepthwiseBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.left_padding,
            groups=channels,
            bias=True,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        transformed = self.depthwise(values)
        transformed = transformed[..., : values.shape[-1]]
        transformed = self.pointwise(transformed)
        transformed = F.gelu(transformed)
        transformed = self.dropout(transformed)
        return self.norm((residual + transformed).transpose(1, 2)).transpose(1, 2)


class LowTurnoverRankModel(nn.Module):
    def __init__(
        self,
        *,
        feature_count: int = 8,
        channels: int = 32,
        dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
        kernel_size: int = 3,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.feature_count = feature_count
        self.channels = channels
        self.feature_norm = nn.LayerNorm(feature_count)
        self.input_projection = nn.Linear(feature_count, channels)
        self.temporal = nn.ModuleList(
            CausalDepthwiseBlock(
                channels,
                kernel_size=kernel_size,
                dilation=dilation,
                dropout=dropout,
            )
            for dilation in dilations
        )
        self.rank_head = nn.Sequential(
            nn.Linear(3 * channels, channels),
            nn.LayerNorm(channels),
            nn.GELU(),
            nn.Linear(channels, 1),
        )

    def forward_sequence(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4:
            raise ValueError("Expected [batch, time, assets, features]")
        batch, time, assets, features = values.shape
        if assets != 3 or features != self.feature_count:
            raise ValueError("Expected exactly three assets and the frozen feature count")
        encoded = self.input_projection(
            deterministic_feature_layer_norm(values, self.feature_norm)
        )
        encoded = encoded.permute(0, 2, 3, 1).reshape(
            batch * assets, self.channels, time
        )
        for block in self.temporal:
            encoded = block(encoded)
        encoded = encoded.reshape(batch, assets, self.channels, time).permute(
            0, 3, 1, 2
        )
        context_mean = encoded.mean(dim=2, keepdim=True)
        context = torch.cat(
            [encoded, context_mean.expand_as(encoded), encoded - context_mean],
            dim=-1,
        )
        raw_scores = self.rank_head(context).squeeze(-1)
        return raw_scores - raw_scores.mean(dim=2, keepdim=True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.forward_sequence(values)[:, -1]


def low_turnover_rank_loss(
    scores: torch.Tensor,
    scaled_excess_targets: torch.Tensor,
    *,
    pairwise_weight: float = 0.50,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if scores.shape != scaled_excess_targets.shape or scores.ndim != 2:
        raise ValueError("Scores and targets must share [batch, assets] shape")
    scores = scores.contiguous()
    scaled_excess_targets = scaled_excess_targets.contiguous()
    centered_targets = scaled_excess_targets - scaled_excess_targets.mean(
        dim=1, keepdim=True
    )
    point = F.smooth_l1_loss(scores, centered_targets, beta=1.0)
    pair_terms: list[torch.Tensor] = []
    for left, right in combinations(range(scores.shape[1]), 2):
        target_difference = centered_targets[:, left] - centered_targets[:, right]
        non_tie = target_difference.abs() > 1.0e-12
        if torch.any(non_tie):
            score_difference = scores[:, left] - scores[:, right]
            sign = torch.sign(target_difference[non_tie])
            pair_terms.append(F.softplus(-score_difference[non_tie] * sign).mean())
    pairwise = (
        torch.stack(pair_terms).mean()
        if pair_terms
        else torch.zeros((), dtype=scores.dtype, device=scores.device)
    )
    total = point + pairwise_weight * pairwise
    return total, {"point": point, "pairwise": pairwise, "total": total}


def apply_low_turnover_policy(
    centered_scores: torch.Tensor,
    market_gate: torch.Tensor,
    *,
    decision_interval: int = 21,
    switch_margin: float = 0.25,
) -> dict[str, Any]:
    if centered_scores.ndim != 2 or centered_scores.shape[1] != 3:
        raise ValueError("Policy scores must be [signal_dates, 3]")
    if market_gate.shape != (centered_scores.shape[0],):
        raise ValueError("Market gate must match signal dates")
    if decision_interval <= 0:
        raise ValueError("Decision interval must be positive")

    incumbent: int | None = None
    previous = torch.zeros(3, dtype=torch.float64)
    turnover = 0.0
    decisions = 0
    actions = {"cash": 0, "enter": 0, "exit": 0, "hold": 0, "switch": 0}
    positions: list[list[float]] = []
    for index in range(centered_scores.shape[0]):
        if index % decision_interval == 0:
            decisions += 1
            scores = centered_scores[index]
            candidate = int(torch.argmax(scores).item())
            candidate_is_valid = bool(
                market_gate[index].item() and scores[candidate].item() > 0.0
            )
            desired = incumbent
            if not candidate_is_valid:
                desired = None
            elif incumbent is None:
                desired = candidate
            elif candidate == incumbent:
                desired = incumbent
            elif scores[candidate].item() - scores[incumbent].item() >= switch_margin:
                desired = candidate

            if incumbent is None and desired is None:
                actions["cash"] += 1
            elif incumbent is None and desired is not None:
                actions["enter"] += 1
            elif incumbent is not None and desired is None:
                actions["exit"] += 1
            elif incumbent == desired:
                actions["hold"] += 1
            else:
                actions["switch"] += 1
            incumbent = desired

        current = torch.zeros(3, dtype=torch.float64)
        if incumbent is not None:
            current[incumbent] = 1.0
        turnover += float(torch.abs(current - previous).sum().item())
        positions.append(current.tolist())
        previous = current

    turnover += float(torch.abs(previous).sum().item())
    return {
        "positions": positions,
        "turnover": turnover,
        "decisions": decisions,
        "actions": actions,
        "final_liquidation_turnover": float(torch.abs(previous).sum().item()),
    }
