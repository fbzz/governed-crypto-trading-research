from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .joint_absolute_relative_spec import _canonical_sha256


JOINT_HEADS = ("excess_score_z", "market_component_z")
PAIR_INDEXES = ((0, 1), (0, 2), (1, 2))


class JointAbsoluteRelativeTransformer(nn.Module):
    def __init__(self, input_features: int, architecture: dict) -> None:
        super().__init__()
        self.input_features = int(input_features)
        self.lookback_days = int(architecture["lookback_days"])
        self.triplet_size = int(architecture["input_triplet_size"])
        self.patch_length = int(architecture["patch_length_days"])
        self.patch_stride = int(architecture["patch_stride_days"])
        self.d_model = int(architecture["d_model"])
        self.patch_count = (
            (self.lookback_days - self.patch_length) // self.patch_stride + 1
        )
        heads = int(architecture["attention_heads"])
        if self.d_model % heads:
            raise ValueError("d_model must be divisible by attention_heads")
        if self.patch_count < 1:
            raise ValueError("Patch geometry does not fit the lookback")
        if tuple(architecture["prediction_heads"]) != JOINT_HEADS:
            raise ValueError("Joint prediction-head contract drift")
        if architecture.get("mask_token") is not False:
            raise ValueError("V47 forbids a mask token")
        if architecture.get("reconstruction_head") is not False:
            raise ValueError("V47 forbids a reconstruction head")

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
            num_layers=int(architecture["encoder_layers"]),
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
        self.prediction_heads = nn.ModuleDict(
            {name: nn.Linear(self.d_model, 1) for name in JOINT_HEADS}
        )
        causal_mask = torch.triu(
            torch.ones(self.patch_count, self.patch_count, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_patch_mask", causal_mask, persistent=False)
        nn.init.normal_(self.temporal_position, mean=0.0, std=0.02)

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError("Input must have [batch,time,assets,features] shape")
        if tuple(x.shape[1:]) != (
            self.lookback_days,
            self.triplet_size,
            self.input_features,
        ):
            raise ValueError(
                "Input contract drift: expected "
                f"[batch,{self.lookback_days},{self.triplet_size},"
                f"{self.input_features}]"
            )
        if x.dtype != torch.float32:
            raise ValueError("V47 input dtype must be float32")

    def extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        patches = x.unfold(1, self.patch_length, self.patch_stride)
        return patches.permute(0, 2, 1, 4, 3).contiguous()

    def encode_temporal_patches(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.extract_patches(x)
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

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        temporal = self.encode_temporal_patches(x)
        cross = self.cross_asset_encoder(temporal[:, :, -1, :])
        cross = self.cross_asset_norm(cross)
        return {
            name: head(cross).squeeze(-1)
            for name, head in self.prediction_heads.items()
        }


def fit_raw_return_rms_scale(
    observed_log_returns: torch.Tensor,
    train_mask: torch.Tensor,
    floor: float,
) -> float:
    if observed_log_returns.ndim != 2 or observed_log_returns.shape[1] != 3:
        raise ValueError("Returns must have [triplets,3] shape")
    if not bool(torch.isfinite(observed_log_returns).all()):
        raise ValueError("Returns must be finite")
    if (
        train_mask.ndim != 1
        or train_mask.shape[0] != observed_log_returns.shape[0]
        or train_mask.dtype != torch.bool
        or not bool(train_mask.any())
    ):
        raise ValueError("Train mask must be boolean and select triplet rows")
    if not math.isfinite(floor) or floor <= 0:
        raise ValueError("Scale floor must be finite and positive")
    rms = torch.sqrt(torch.mean(observed_log_returns[train_mask].square()))
    return max(float(rms), float(floor))


def decompose_return_targets(
    observed_log_returns: torch.Tensor,
    scale: float,
) -> dict[str, torch.Tensor]:
    if observed_log_returns.ndim != 2 or observed_log_returns.shape[1] != 3:
        raise ValueError("Returns must have [batch,3] shape")
    if not bool(torch.isfinite(observed_log_returns).all()):
        raise ValueError("Returns must be finite")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Return scale must be finite and positive")
    market = observed_log_returns.mean(dim=1)
    excess = observed_log_returns - market[:, None]
    return {
        "r": observed_log_returns,
        "m": market,
        "e": excess,
        "z_r": observed_log_returns / float(scale),
        "z_m": market / float(scale),
        "z_e": excess / float(scale),
    }


def reconstruct_joint_predictions(
    output: dict[str, torch.Tensor],
    scale: float,
) -> dict[str, torch.Tensor]:
    if set(output) != set(JOINT_HEADS):
        raise ValueError("Joint output-head contract drift")
    excess_score = output["excess_score_z"]
    market_component = output["market_component_z"]
    if excess_score.ndim != 2 or excess_score.shape[1] != 3:
        raise ValueError("Joint outputs must have [batch,3] shape")
    if market_component.shape != excess_score.shape:
        raise ValueError("Joint head shapes must match")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Return scale must be finite and positive")
    e_hat_z = excess_score - excess_score.mean(dim=1, keepdim=True)
    m_hat_z = market_component.mean(dim=1)
    mu_hat_z = m_hat_z[:, None] + e_hat_z
    return {
        "e_hat_z": e_hat_z,
        "m_hat_z": m_hat_z,
        "mu_hat_z": mu_hat_z,
        "e_hat": e_hat_z * float(scale),
        "m_hat": m_hat_z * float(scale),
        "mu_hat": mu_hat_z * float(scale),
    }


def joint_absolute_relative_loss(
    output: dict[str, torch.Tensor],
    observed_log_returns: torch.Tensor,
    scale: float,
    *,
    tie_tolerance: float,
) -> dict[str, torch.Tensor]:
    if tie_tolerance < 0:
        raise ValueError("Tie tolerance cannot be negative")
    target = decompose_return_targets(observed_log_returns, scale)
    predicted = reconstruct_joint_predictions(output, scale)
    pair_losses = []
    pair_count = 0
    for left, right in PAIR_INDEXES:
        raw_difference = target["r"][:, left] - target["r"][:, right]
        active = raw_difference.abs() > tie_tolerance
        if bool(active.any()):
            sign = (target["z_r"][:, left] - target["z_r"][:, right])[active].sign()
            predicted_difference = (
                predicted["e_hat_z"][:, left]
                - predicted["e_hat_z"][:, right]
            )[active]
            pair_losses.append(F.softplus(-sign * predicted_difference))
            pair_count += int(active.sum())
    ranking = (
        torch.cat(pair_losses).mean()
        if pair_losses
        else predicted["e_hat_z"].sum() * 0.0
    )
    excess = F.smooth_l1_loss(
        predicted["e_hat_z"], target["z_e"], beta=1.0
    )
    market_level = F.smooth_l1_loss(
        predicted["m_hat_z"], target["z_m"], beta=1.0
    )
    absolute_level = F.smooth_l1_loss(
        predicted["mu_hat_z"], target["z_r"], beta=1.0
    )
    level = 0.5 * market_level + 0.5 * absolute_level
    total = ranking + excess + level
    return {
        "ranking": ranking,
        "excess": excess,
        "market_level": market_level,
        "absolute_level": absolute_level,
        "level": level,
        "total": total,
        "pair_count": torch.tensor(pair_count, device=observed_log_returns.device),
        **target,
        **predicted,
    }


def joint_triplet_positions(
    predicted_mu: np.ndarray,
    predicted_excess: np.ndarray,
    eligible: np.ndarray,
    *,
    risky_weight: float = 1.0 / 3.0,
    base_cost: float = 0.001,
) -> np.ndarray:
    mu = np.asarray(predicted_mu, dtype=np.float64)
    excess = np.asarray(predicted_excess, dtype=np.float64)
    eligibility = np.asarray(eligible, dtype=bool)
    if mu.ndim != 2 or mu.shape[1] != 3:
        raise ValueError("Predictions must have [days,3] shape")
    if excess.shape != mu.shape or eligibility.shape != mu.shape:
        raise ValueError("Prediction and eligibility shapes must match")
    if not 0 < risky_weight <= 1:
        raise ValueError("Risky weight must be in (0,1]")
    if not math.isfinite(base_cost) or base_cost < 0:
        raise ValueError("Base cost must be finite and nonnegative")

    positions = np.zeros_like(mu)
    current = np.zeros(3, dtype=np.float64)
    for day in range(len(mu)):
        valid = eligibility[day] & np.isfinite(mu[day]) & np.isfinite(excess[day])
        valid_indexes = np.flatnonzero(valid)
        candidates: list[np.ndarray] = []
        incumbent = int(np.argmax(current)) if current.sum() > 0 else None
        if incumbent is not None and valid[incumbent]:
            candidates.append(current.copy())
        candidates.append(np.zeros(3, dtype=np.float64))
        if len(valid_indexes):
            best_excess = np.max(excess[day, valid_indexes])
            challenger = int(
                valid_indexes[np.flatnonzero(excess[day, valid_indexes] == best_excess)[0]]
            )
            action = np.zeros(3, dtype=np.float64)
            action[challenger] = risky_weight
            if not any(np.array_equal(action, candidate) for candidate in candidates):
                candidates.append(action)
        best = candidates[0]
        best_utility = float(np.dot(best, np.where(np.isfinite(mu[day]), mu[day], 0.0)))
        best_utility -= base_cost * float(np.abs(best - current).sum())
        for candidate in candidates[1:]:
            utility = float(
                np.dot(candidate, np.where(np.isfinite(mu[day]), mu[day], 0.0))
            ) - base_cost * float(np.abs(candidate - current).sum())
            if utility > best_utility and not math.isclose(
                utility, best_utility, rel_tol=0.0, abs_tol=1e-15
            ):
                best = candidate
                best_utility = utility
        current = best.copy()
        positions[day] = current
    return positions


def _state_dict_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def save_joint_checkpoint(
    path: Path,
    payload: dict,
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
        "input_features",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Joint checkpoint payload missing: {missing}")
    serializable = dict(payload)
    serializable["format_version"] = format_version
    serializable["architecture_sha256"] = _canonical_sha256(payload["architecture"])
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(serializable, path)


def load_joint_checkpoint(
    path: Path,
    *,
    expected_format_version: str,
    expected_architecture: dict,
    expected_metadata: dict,
) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != expected_format_version:
        raise ValueError("Joint checkpoint format drift")
    if payload.get("architecture_sha256") != _canonical_sha256(
        expected_architecture
    ):
        raise ValueError("Joint checkpoint architecture hash drift")
    if payload.get("architecture") != expected_architecture:
        raise ValueError("Joint checkpoint architecture mismatch")
    if payload.get("metadata") != expected_metadata:
        raise ValueError("Joint checkpoint metadata mismatch")
    return payload
