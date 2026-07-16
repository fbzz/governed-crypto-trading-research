from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .core import (
    SyntheticAccessLedger,
    canonical_sha256,
    file_sha256,
    persistent_portfolio_returns,
    write_json_atomic,
    write_yaml_atomic,
)
from .decoupled_rank_state_spec import (
    ranker_parameter_count,
    state_gate_parameter_count,
)
from .patch_transformer import MultiAssetPatchTransformer
from .ranking_excess_harness import ranking_excess_loss


RANKER_HEADS = ("excess_return_z", "log_volatility_7d")
EXPECTED_INPUT_NAMES = {
    "v60_specification",
    "v60_blueprint",
    "v60_audit",
    "v60_result",
    "v60_artifact_manifest",
    "v60_completion_receipt",
}


class IndependentStateGate(nn.Module):
    """The exact independent 27,489-parameter V60 market-state encoder."""

    def __init__(self, architecture: dict[str, Any]) -> None:
        super().__init__()
        self.input_features = int(architecture["input_features"])
        self.lookback_days = int(architecture["lookback_days"])
        self.patch_length = int(architecture["patch_length_days"])
        self.patch_stride = int(architecture["patch_stride_days"])
        self.d_model = int(architecture["d_model"])
        self.patch_count = (
            (self.lookback_days - self.patch_length) // self.patch_stride + 1
        )
        heads = int(architecture["attention_heads"])
        if self.d_model % heads:
            raise ValueError("State-gate d_model must divide attention heads")
        if self.patch_count < 1:
            raise ValueError("State-gate patch geometry does not fit lookback")
        if architecture.get("independent_encoder") is not True:
            raise ValueError("V60 state gate must be independent")
        if architecture.get("ranker_representation_input") != "none":
            raise ValueError("V60 state gate cannot consume ranker representations")

        patch_width = self.patch_length * self.input_features
        self.patch_projection = nn.Linear(patch_width, self.d_model)
        self.temporal_position = nn.Parameter(
            torch.zeros(1, self.patch_count, self.d_model)
        )
        self.patch_norm = nn.LayerNorm(self.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=heads,
            dim_feedforward=int(architecture["feed_forward_width"]),
            dropout=float(architecture["dropout"]),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer,
            num_layers=int(architecture["encoder_layers"]),
            enable_nested_tensor=False,
        )
        self.temporal_norm = nn.LayerNorm(self.d_model)
        self.output_head = nn.Linear(self.d_model, 1)
        causal_mask = torch.triu(
            torch.ones(self.patch_count, self.patch_count, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_patch_mask", causal_mask, persistent=False)
        nn.init.normal_(self.temporal_position, mean=0.0, std=0.02)

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 3:
            raise ValueError("State input must have [batch,time,features] shape")
        if tuple(x.shape[1:]) != (self.lookback_days, self.input_features):
            raise ValueError(
                "State input contract drift: expected "
                f"[batch,{self.lookback_days},{self.input_features}]"
            )
        if x.dtype != torch.float32:
            raise ValueError("State input dtype must be float32")

    def extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        return x.unfold(1, self.patch_length, self.patch_stride).permute(
            0, 1, 3, 2
        ).contiguous()

    def encode_temporal_patches(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.extract_patches(x)
        tokens = self.patch_projection(patches.flatten(start_dim=2))
        tokens = self.patch_norm(tokens + self.temporal_position)
        encoded = self.temporal_encoder(tokens, mask=self.causal_patch_mask)
        return self.temporal_norm(encoded)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_head(self.encode_temporal_patches(x)[:, -1]).squeeze(-1)


def derive_state_features(triplet_features: torch.Tensor) -> torch.Tensor:
    if triplet_features.ndim != 4 or triplet_features.shape[2:] != (3, 9):
        raise ValueError("Triplet features must have [batch,time,3,9] shape")
    if triplet_features.dtype != torch.float32:
        raise ValueError("Triplet features must be float32")
    mean = triplet_features.mean(dim=2)
    population_std = triplet_features.std(dim=2, unbiased=False)
    return torch.cat([mean, population_std], dim=-1)


def reconstruct_decoupled_returns(
    predicted_excess_z: torch.Tensor,
    predicted_market_z: torch.Tensor,
    *,
    excess_scale: float,
    market_scale: float,
) -> dict[str, torch.Tensor]:
    if predicted_excess_z.ndim != 2 or predicted_excess_z.shape[1] != 3:
        raise ValueError("Ranker output must have [batch,3] shape")
    if predicted_market_z.shape != predicted_excess_z.shape[:1]:
        raise ValueError("State-gate output must have [batch] shape")
    if not math.isfinite(excess_scale) or excess_scale <= 0:
        raise ValueError("Excess scale must be finite and positive")
    if not math.isfinite(market_scale) or market_scale <= 0:
        raise ValueError("Market scale must be finite and positive")
    centered_excess_z = predicted_excess_z - predicted_excess_z.mean(
        dim=1, keepdim=True
    )
    raw_excess = centered_excess_z * float(excess_scale)
    market = predicted_market_z * float(market_scale)
    absolute = market[:, None] + raw_excess
    return {
        "centered_excess_z": centered_excess_z,
        "raw_excess": raw_excess,
        "market": market,
        "absolute": absolute,
    }


def decoupled_rank_state_positions(
    predicted_raw_excess: np.ndarray,
    predicted_market_component: np.ndarray,
    momentum_30: np.ndarray,
    eligible: np.ndarray,
    *,
    base_cost: float,
    switch_hurdle: float,
    risky_weight: float = 1.0,
) -> dict[str, Any]:
    excess = np.asarray(predicted_raw_excess, dtype=np.float64)
    market = np.asarray(predicted_market_component, dtype=np.float64)
    momentum = np.asarray(momentum_30, dtype=np.float64)
    eligibility = np.asarray(eligible, dtype=bool)
    if excess.ndim != 2 or excess.shape[1] != 3:
        raise ValueError("Excess forecasts must have [days,3] shape")
    if momentum.shape != excess.shape or eligibility.shape != excess.shape:
        raise ValueError("Excess, momentum, and eligibility shapes must match")
    if market.shape != excess.shape[:1]:
        raise ValueError("Market component must have [days] shape")
    if not math.isfinite(base_cost) or base_cost < 0:
        raise ValueError("Base cost must be finite and nonnegative")
    if not math.isfinite(switch_hurdle) or switch_hurdle < 0:
        raise ValueError("Switch hurdle must be finite and nonnegative")
    if not 0 < risky_weight <= 1:
        raise ValueError("Risky weight must be in (0,1]")

    positions = np.zeros_like(excess)
    actions: list[str] = []
    selected_assets: list[int | None] = []
    incumbent: int | None = None
    entry_cost = base_cost * risky_weight
    switch_cost = base_cost * 2.0 * risky_weight
    for day in range(len(excess)):
        valid = (
            eligibility[day]
            & np.isfinite(excess[day])
            & np.isfinite(momentum[day])
        )
        valid_indexes = np.flatnonzero(valid)
        if incumbent is not None and not valid[incumbent]:
            incumbent = None
            actions.append("forced_exit")
            selected_assets.append(None)
            continue
        if (
            not np.isfinite(market[day])
            or len(valid_indexes) == 0
            or np.all(momentum[day, valid_indexes] <= 0)
        ):
            actions.append("momentum_exit" if incumbent is not None else "cash")
            incumbent = None
            selected_assets.append(None)
            continue

        best_value = np.max(excess[day, valid_indexes])
        challenger = int(
            valid_indexes[np.flatnonzero(excess[day, valid_indexes] == best_value)[0]]
        )
        challenger_edge = float(market[day] + excess[day, challenger])
        if incumbent is None:
            if challenger_edge > entry_cost:
                incumbent = challenger
                actions.append("entry")
            else:
                actions.append("cash")
        else:
            incumbent_edge = float(market[day] + excess[day, incumbent])
            if incumbent_edge <= 0.0:
                incumbent = None
                actions.append("edge_exit")
            elif challenger != incumbent and (
                excess[day, challenger] - excess[day, incumbent] > switch_hurdle
                and challenger_edge > switch_cost
            ):
                incumbent = challenger
                actions.append("switch")
            else:
                actions.append("hold")
        if incumbent is not None:
            positions[day, incumbent] = risky_weight
        selected_assets.append(incumbent)
    return {
        "positions": positions,
        "actions": actions,
        "selected_assets": selected_assets,
    }


def _clone_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def _state_dict_sha256(model: nn.Module) -> str:
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


def _ranker_architecture(registered: dict[str, Any]) -> dict[str, Any]:
    return {**registered, "input_triplet_size": 3}


def _build_ranker(architecture: dict[str, Any]) -> MultiAssetPatchTransformer:
    return MultiAssetPatchTransformer(
        9,
        _ranker_architecture(architecture),
        expected_prediction_heads=RANKER_HEADS,
    )


def _ranker_step(
    model: MultiAssetPatchTransformer,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    excess_scale: float,
    tie_tolerance: float,
    gradient_clip_norm: float,
) -> tuple[float, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = ranking_excess_loss(
        model(features),
        labels,
        excess_scale,
        tie_tolerance=tie_tolerance,
        volatility_floor=1.0e-6,
        ranking_weight=1.0,
        excess_weight=1.0,
        volatility_weight=0.1,
    )
    losses["total"].backward()
    norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    optimizer.step()
    return float(losses["total"].detach()), float(norm)


def _gate_step(
    model: IndependentStateGate,
    optimizer: torch.optim.Optimizer,
    state_features: torch.Tensor,
    target_market_z: torch.Tensor,
    *,
    gradient_clip_norm: float,
) -> tuple[float, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = F.smooth_l1_loss(
        model(state_features), target_market_z, beta=1.0
    )
    loss.backward()
    norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    optimizer.step()
    return float(loss.detach()), float(norm)


def _training_cycle(
    ranker: MultiAssetPatchTransformer,
    gate: IndependentStateGate,
    ranker_optimizer: torch.optim.Optimizer,
    gate_optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    state_features: torch.Tensor,
    labels: torch.Tensor,
    target_market_z: torch.Tensor,
    *,
    excess_scale: float,
    tie_tolerance: float,
    gradient_clip_norm: float,
    ledger: SyntheticAccessLedger,
) -> tuple[dict[str, float], bool]:
    gate_before = _state_dict_sha256(gate)
    ranker_loss, ranker_norm = _ranker_step(
        ranker,
        ranker_optimizer,
        features,
        labels,
        excess_scale=excess_scale,
        tie_tolerance=tie_tolerance,
        gradient_clip_norm=gradient_clip_norm,
    )
    ledger.synthetic_optimizer_steps += 1
    gate_unchanged = gate_before == _state_dict_sha256(gate)

    ranker_before = _state_dict_sha256(ranker)
    gate_loss, gate_norm = _gate_step(
        gate,
        gate_optimizer,
        state_features,
        target_market_z,
        gradient_clip_norm=gradient_clip_norm,
    )
    ledger.synthetic_optimizer_steps += 1
    ranker_unchanged = ranker_before == _state_dict_sha256(ranker)
    return {
        "ranker_loss": ranker_loss,
        "ranker_gradient_norm": ranker_norm,
        "state_gate_loss": gate_loss,
        "state_gate_gradient_norm": gate_norm,
    }, gate_unchanged and ranker_unchanged


def _serialize_checkpoint(payload: dict[str, Any], format_version: str) -> bytes:
    value = dict(payload)
    value["format_version"] = format_version
    value["architecture_sha256"] = canonical_sha256(payload["architecture"])
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _load_checkpoint(
    checkpoint_bytes: bytes,
    *,
    expected_format_version: str,
    expected_architecture: dict[str, Any],
    expected_metadata: dict[str, Any],
) -> dict[str, Any]:
    payload = torch.load(
        io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False
    )
    if payload.get("format_version") != expected_format_version:
        raise ValueError("V61 synthetic checkpoint format drift")
    if payload.get("architecture") != expected_architecture:
        raise ValueError("V61 synthetic checkpoint architecture drift")
    if payload.get("architecture_sha256") != canonical_sha256(
        expected_architecture
    ):
        raise ValueError("V61 synthetic checkpoint architecture hash drift")
    if payload.get("metadata") != expected_metadata:
        raise ValueError("V61 synthetic checkpoint metadata drift")
    return payload


def _canonical_self_hash(value: dict[str, Any], key: str) -> bool:
    payload = dict(value)
    registered = payload.pop(key, None)
    return isinstance(registered, str) and registered == canonical_sha256(payload)


def _load_allowed_json(
    path: Path,
    allowed_paths: set[Path],
    ledger: SyntheticAccessLedger,
) -> dict[str, Any]:
    if path.resolve() not in allowed_paths:
        raise PermissionError(f"V61 metadata read is not allowlisted: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    ledger.authorized_metadata_reads += 1
    return value


def _build_harness_spec(
    blueprint: dict[str, Any], harness: dict[str, Any]
) -> dict[str, Any]:
    value = {
        "schema_version": "v61-decoupled-rank-state-harness-spec/v1",
        "version": harness["version"],
        "candidate_family_id": blueprint["candidate_family_id"],
        "v60_blueprint_sha256": blueprint["blueprint_sha256"],
        "architecture": blueprint["architecture"],
        "objective": blueprint["objective"],
        "policy": blueprint["policy"],
        "synthetic": harness["synthetic"],
        "mechanics": harness["mechanics"],
        "checkpoint": harness["checkpoint"],
        "constraints": harness["constraints"],
        "authorized_next_action": harness["authorized_next_action"],
    }
    value["harness_spec_sha256"] = canonical_sha256(value)
    return value


def _policy_fixture(
    policy: dict[str, Any], days: int
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray | float]], bool]:
    if days != 7:
        raise ValueError("V61 policy fixture is frozen to seven synthetic days")
    excess = np.array(
        [
            [0.0010, 0.0000, -0.0010],
            [0.0010, 0.0020, -0.0030],
            [0.0010, 0.0040, -0.0050],
            [-0.0080, 0.0030, 0.0050],
            [0.0000, 0.0010, -0.0010],
            [0.0010, 0.0000, -0.0010],
            [-0.0010, -0.0015, 0.0025],
        ],
        dtype=np.float64,
    )
    market = np.array(
        [0.0020, 0.0005, 0.0040, 0.0040, -0.0040, 0.0020, 0.0020],
        dtype=np.float64,
    )
    momentum = np.ones_like(excess)
    momentum[5] = -1.0
    eligible = np.ones_like(excess, dtype=bool)
    base_cost = float(policy["base_cost_bps"]) / 10_000.0
    result = decoupled_rank_state_positions(
        excess,
        market,
        momentum,
        eligible,
        base_cost=base_cost,
        switch_hurdle=float(policy["switch_hurdle"]),
        risky_weight=float(policy["risky_gross"]),
    )
    accounting = {
        str(cost): persistent_portfolio_returns(
            result["positions"], np.zeros_like(excess), float(cost)
        )
        for cost in policy["reporting_cost_bps"]
    }
    strict_entry = decoupled_rank_state_positions(
        np.zeros((1, 3), dtype=np.float64),
        np.array([base_cost], dtype=np.float64),
        np.ones((1, 3), dtype=np.float64),
        np.ones((1, 3), dtype=bool),
        base_cost=base_cost,
        switch_hurdle=float(policy["switch_hurdle"]),
    )
    strict_entry_stays_cash = strict_entry["positions"].sum() == 0.0
    return result, accounting, strict_entry_stays_cash


def _execute_synthetic(
    blueprint: dict[str, Any],
    harness: dict[str, Any],
    harness_spec: dict[str, Any],
    *,
    seed: int,
    authorized_metadata_reads: int,
) -> dict[str, Any]:
    synthetic = harness["synthetic"]
    mechanics = harness["mechanics"]
    checkpoint_contract = harness["checkpoint"]
    ranker_architecture = blueprint["architecture"]["ranker"]
    gate_architecture = blueprint["architecture"]["state_gate"]
    training = blueprint["training_contract"]
    policy = blueprint["policy"]
    if synthetic["device"] != "cpu":
        raise ValueError("V61 harness is frozen to deterministic CPU")
    if int(synthetic["input_features"]) != 9 or int(
        synthetic["state_features"]
    ) != 18:
        raise ValueError("V61 synthetic feature count drift")

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    ranker_prototype = _build_ranker(ranker_architecture)
    gate_prototype = IndependentStateGate(gate_architecture)
    ranker_base_state = _clone_state_dict(ranker_prototype)
    gate_base_state = _clone_state_dict(gate_prototype)
    ranker_parameters = sum(p.numel() for p in ranker_prototype.parameters())
    gate_parameters = sum(p.numel() for p in gate_prototype.parameters())

    batch_size = int(synthetic["batch_size"])
    feature_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    return_generator = torch.Generator(device="cpu").manual_seed(seed + 2)
    volatility_generator = torch.Generator(device="cpu").manual_seed(seed + 3)
    features = torch.randn(
        batch_size,
        256,
        3,
        9,
        generator=feature_generator,
        dtype=torch.float32,
    )
    returns = torch.randn(
        batch_size, 3, generator=return_generator, dtype=torch.float32
    ) * 0.02
    volatility = 0.01 + torch.rand(
        batch_size, 3, generator=volatility_generator, dtype=torch.float32
    ) * 0.10
    labels = torch.stack([returns, volatility], dim=-1)
    state_features = derive_state_features(features)
    market_target = returns.mean(dim=1)
    excess_target = returns - market_target[:, None]
    excess_scale = max(
        float(torch.sqrt(excess_target.square().mean())),
        float(mechanics["excess_scale_floor"]),
    )
    market_scale = max(
        float(torch.sqrt(market_target.square().mean())),
        float(mechanics["market_scale_floor"]),
    )
    target_market_z = market_target / market_scale
    ledger = SyntheticAccessLedger(
        authorized_metadata_reads=authorized_metadata_reads,
        synthetic_tensor_generations=4,
    )

    optimizer_kwargs = {
        "lr": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
    }
    torch.manual_seed(seed + 4)
    start_rng = torch.get_rng_state().clone()
    full_ranker = _build_ranker(ranker_architecture)
    full_gate = IndependentStateGate(gate_architecture)
    full_ranker.load_state_dict(ranker_base_state)
    full_gate.load_state_dict(gate_base_state)
    full_ranker_optimizer = torch.optim.AdamW(
        full_ranker.parameters(), **optimizer_kwargs
    )
    full_gate_optimizer = torch.optim.AdamW(
        full_gate.parameters(), **optimizer_kwargs
    )
    torch.set_rng_state(start_rng)
    full_history: list[dict[str, float]] = []
    optimizer_isolation: list[bool] = []
    for _ in range(int(synthetic["optimizer_cycles_per_component"])):
        history, isolated = _training_cycle(
            full_ranker,
            full_gate,
            full_ranker_optimizer,
            full_gate_optimizer,
            features,
            state_features,
            labels,
            target_market_z,
            excess_scale=excess_scale,
            tie_tolerance=float(mechanics["ranknet_tie_tolerance"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            ledger=ledger,
        )
        full_history.append(history)
        optimizer_isolation.append(isolated)

    interrupted_ranker = _build_ranker(ranker_architecture)
    interrupted_gate = IndependentStateGate(gate_architecture)
    interrupted_ranker.load_state_dict(ranker_base_state)
    interrupted_gate.load_state_dict(gate_base_state)
    interrupted_ranker_optimizer = torch.optim.AdamW(
        interrupted_ranker.parameters(), **optimizer_kwargs
    )
    interrupted_gate_optimizer = torch.optim.AdamW(
        interrupted_gate.parameters(), **optimizer_kwargs
    )
    torch.set_rng_state(start_rng)
    first_history, first_isolated = _training_cycle(
        interrupted_ranker,
        interrupted_gate,
        interrupted_ranker_optimizer,
        interrupted_gate_optimizer,
        features,
        state_features,
        labels,
        target_market_z,
        excess_scale=excess_scale,
        tie_tolerance=float(mechanics["ranknet_tie_tolerance"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
        ledger=ledger,
    )
    optimizer_isolation.append(first_isolated)
    saved_rng = torch.get_rng_state().clone()
    checkpoint_metadata = {
        "candidate_family_id": blueprint["candidate_family_id"],
        "v60_blueprint_sha256": blueprint["blueprint_sha256"],
        "v61_harness_spec_sha256": harness_spec["harness_spec_sha256"],
        "initialization_seed": seed,
        "status": checkpoint_contract["status"],
    }
    checkpoint_architecture = {
        "ranker": ranker_architecture,
        "state_gate": gate_architecture,
    }
    checkpoint_payload = {
        "architecture": checkpoint_architecture,
        "ranker_state": _clone_state_dict(interrupted_ranker),
        "state_gate_state": _clone_state_dict(interrupted_gate),
        "ranker_optimizer_state": deepcopy(
            interrupted_ranker_optimizer.state_dict()
        ),
        "state_gate_optimizer_state": deepcopy(
            interrupted_gate_optimizer.state_dict()
        ),
        "cpu_rng_state": saved_rng,
        "history": [first_history],
        "metadata": checkpoint_metadata,
    }
    checkpoint_bytes = _serialize_checkpoint(
        checkpoint_payload, checkpoint_contract["format_version"]
    )
    ledger.synthetic_checkpoint_writes += 1
    checkpoint = _load_checkpoint(
        checkpoint_bytes,
        expected_format_version=checkpoint_contract["format_version"],
        expected_architecture=checkpoint_architecture,
        expected_metadata=checkpoint_metadata,
    )
    ledger.synthetic_checkpoint_reads += 1

    resumed_ranker = _build_ranker(ranker_architecture)
    resumed_gate = IndependentStateGate(gate_architecture)
    resumed_ranker.load_state_dict(checkpoint["ranker_state"])
    resumed_gate.load_state_dict(checkpoint["state_gate_state"])
    roundtrip_ranker_hash = _state_dict_sha256(resumed_ranker)
    roundtrip_gate_hash = _state_dict_sha256(resumed_gate)
    resumed_ranker_optimizer = torch.optim.AdamW(
        resumed_ranker.parameters(), **optimizer_kwargs
    )
    resumed_gate_optimizer = torch.optim.AdamW(
        resumed_gate.parameters(), **optimizer_kwargs
    )
    resumed_ranker_optimizer.load_state_dict(checkpoint["ranker_optimizer_state"])
    resumed_gate_optimizer.load_state_dict(
        checkpoint["state_gate_optimizer_state"]
    )
    torch.set_rng_state(checkpoint["cpu_rng_state"])
    resumed_history = [first_history]
    for _ in range(1, int(synthetic["optimizer_cycles_per_component"])):
        history, isolated = _training_cycle(
            resumed_ranker,
            resumed_gate,
            resumed_ranker_optimizer,
            resumed_gate_optimizer,
            features,
            state_features,
            labels,
            target_market_z,
            excess_scale=excess_scale,
            tie_tolerance=float(mechanics["ranknet_tie_tolerance"]),
            gradient_clip_norm=float(training["gradient_clip_norm"]),
            ledger=ledger,
        )
        resumed_history.append(history)
        optimizer_isolation.append(isolated)

    resumed_ranker.eval()
    resumed_gate.eval()
    permutation = torch.tensor(mechanics["asset_permutation"])
    cutoff = int(mechanics["causal_cutoff_day"])
    with torch.no_grad():
        ranker_output = resumed_ranker(features)
        permuted_ranker_output = resumed_ranker(features[:, :, permutation, :])
        state_output = resumed_gate(state_features)
        permuted_state_features = derive_state_features(
            features[:, :, permutation, :]
        )
        permuted_state_output = resumed_gate(permuted_state_features)
        ranker_temporal = resumed_ranker.encode_temporal_patches(features)
        altered_features = features.clone()
        altered_features[:, cutoff:] += 100.0
        altered_ranker_temporal = resumed_ranker.encode_temporal_patches(
            altered_features
        )
        gate_temporal = resumed_gate.encode_temporal_patches(state_features)
        altered_state_features = state_features.clone()
        altered_state_features[:, cutoff:] += 100.0
        altered_gate_temporal = resumed_gate.encode_temporal_patches(
            altered_state_features
        )
        patch_mask = torch.zeros(
            batch_size, 3, resumed_ranker.patch_count, dtype=torch.bool
        )
        patch_mask[:, :, ::2] = True
        reconstruction = resumed_ranker(
            features, patch_mask=patch_mask, return_reconstruction=True
        )["patch_reconstruction"]
    early_patch_count = (
        (cutoff - resumed_ranker.patch_length) // resumed_ranker.patch_stride + 1
    )
    reconstructed = reconstruct_decoupled_returns(
        ranker_output["excess_return_z"],
        state_output,
        excess_scale=excess_scale,
        market_scale=market_scale,
    )

    for parameter in resumed_ranker.parameters():
        parameter.grad = None
    for parameter in resumed_gate.parameters():
        parameter.grad = None
    gate_isolation_loss = F.smooth_l1_loss(
        resumed_gate(state_features), target_market_z, beta=1.0
    )
    gate_isolation_loss.backward()
    gate_gradients_finite = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in resumed_gate.parameters()
    )
    ranker_gradients_after_gate = any(
        parameter.grad is not None for parameter in resumed_ranker.parameters()
    )
    for parameter in resumed_ranker.parameters():
        parameter.grad = None
    for parameter in resumed_gate.parameters():
        parameter.grad = None
    ranker_isolation_losses = ranking_excess_loss(
        resumed_ranker(features),
        labels,
        excess_scale,
        tie_tolerance=float(mechanics["ranknet_tie_tolerance"]),
        volatility_floor=1.0e-6,
        ranking_weight=1.0,
        excess_weight=1.0,
        volatility_weight=0.1,
    )
    ranker_isolation_losses["total"].backward()
    active_ranker_parameters = [
        parameter
        for name, parameter in resumed_ranker.named_parameters()
        if not name.startswith(("mask_token", "reconstruction_head"))
    ]
    ranker_gradients_finite = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in active_ranker_parameters
    )
    gate_gradients_after_ranker = any(
        parameter.grad is not None for parameter in resumed_gate.parameters()
    )

    policy_result, accounting, strict_entry_stays_cash = _policy_fixture(
        policy, int(synthetic["policy_days"])
    )
    base_accounting = accounting[str(policy["base_cost_bps"])]
    expected_turnover = np.array([1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0])
    expected_actions = [
        "entry",
        "hold",
        "switch",
        "hold",
        "edge_exit",
        "cash",
        "entry",
    ]
    parameter_identity_ranker = {id(p) for p in resumed_ranker.parameters()}
    parameter_identity_gate = {id(p) for p in resumed_gate.parameters()}
    ranker_optimizer_parameters = {
        id(p)
        for group in resumed_ranker_optimizer.param_groups
        for p in group["params"]
    }
    gate_optimizer_parameters = {
        id(p)
        for group in resumed_gate_optimizer.param_groups
        for p in group["params"]
    }
    operation_ledger = ledger.to_dict()
    component_counts = blueprint["architecture"]["parameter_counts"]
    checks = {
        "exact_component_and_total_parameter_counts": ranker_parameters
        == ranker_parameter_count(ranker_architecture)
        == int(ranker_architecture["expected_parameter_count"])
        == int(component_counts["ranker"])
        and gate_parameters
        == state_gate_parameter_count(gate_architecture)
        == int(gate_architecture["expected_parameter_count"])
        == int(component_counts["state_gate"])
        and ranker_parameters + gate_parameters == int(component_counts["total"]),
        "exact_output_shapes": set(ranker_output) == set(RANKER_HEADS)
        and all(tuple(ranker_output[name].shape) == (batch_size, 3) for name in RANKER_HEADS)
        and tuple(state_output.shape) == (batch_size,)
        and tuple(reconstruction.shape) == (batch_size, 3, 31, 16, 9),
        "causal_temporal_prefix_for_both_encoders": torch.allclose(
            ranker_temporal[:, :, :early_patch_count],
            altered_ranker_temporal[:, :, :early_patch_count],
            atol=1.0e-5,
            rtol=1.0e-5,
        )
        and torch.allclose(
            gate_temporal[:, :early_patch_count],
            altered_gate_temporal[:, :early_patch_count],
            atol=1.0e-5,
            rtol=1.0e-5,
        ),
        "asset_permutation_equivariance": all(
            torch.allclose(
                permuted_ranker_output[name],
                ranker_output[name][:, permutation],
                atol=1.0e-5,
                rtol=1.0e-5,
            )
            for name in RANKER_HEADS
        ),
        "state_feature_permutation_invariance": torch.allclose(
            permuted_state_features, state_features, atol=1.0e-6, rtol=1.0e-6
        )
        and torch.allclose(
            permuted_state_output, state_output, atol=1.0e-5, rtol=1.0e-5
        ),
        "centered_excess_and_absolute_return_decomposition": torch.allclose(
            reconstructed["raw_excess"].sum(dim=1),
            torch.zeros(batch_size),
            atol=1.0e-6,
        )
        and torch.allclose(
            reconstructed["absolute"],
            reconstructed["market"][:, None] + reconstructed["raw_excess"],
            atol=0.0,
            rtol=0.0,
        ),
        "zero_shared_parameter_identity": parameter_identity_ranker.isdisjoint(
            parameter_identity_gate
        )
        and ranker_optimizer_parameters == parameter_identity_ranker
        and gate_optimizer_parameters == parameter_identity_gate
        and ranker_optimizer_parameters.isdisjoint(gate_optimizer_parameters),
        "gate_backward_does_not_change_ranker_gradients": gate_gradients_finite
        and not ranker_gradients_after_gate,
        "ranker_backward_does_not_change_gate_gradients": ranker_gradients_finite
        and not gate_gradients_after_ranker,
        "independent_optimizer_state_and_steps": all(optimizer_isolation)
        and _optimizer_step_count(resumed_ranker_optimizer)
        == int(synthetic["optimizer_cycles_per_component"])
        and _optimizer_step_count(resumed_gate_optimizer)
        == int(synthetic["optimizer_cycles_per_component"])
        and math.isfinite(
            sum(row["ranker_loss"] + row["state_gate_loss"] for row in full_history)
        ),
        "checkpoint_roundtrip_and_interrupted_resume": roundtrip_ranker_hash
        == _state_dict_sha256(interrupted_ranker)
        and roundtrip_gate_hash == _state_dict_sha256(interrupted_gate)
        and _state_dict_sha256(full_ranker) == _state_dict_sha256(resumed_ranker)
        and _state_dict_sha256(full_gate) == _state_dict_sha256(resumed_gate)
        and full_history == resumed_history
        and torch.equal(checkpoint["cpu_rng_state"], saved_rng),
        "exact_cost_aware_entry_hold_switch_exit_and_final_liquidation": policy_result[
            "actions"
        ]
        == expected_actions
        and np.allclose(base_accounting["turnover"], expected_turnover)
        and math.isclose(float(base_accounting["total_turnover"]), 6.0)
        and strict_entry_stays_cash
        and all(
            np.allclose(values["turnover"], base_accounting["turnover"])
            and np.allclose(
                values["cost"], values["turnover"] * (float(cost) / 10_000.0)
            )
            and np.allclose(values["net_return"], values["gross_return"] - values["cost"])
            for cost, values in accounting.items()
        ),
        "zero_real_data_checkpoint_outcome_or_target_access": ledger.forbidden_operations_are_zero()
        and not any(bool(value) for value in harness["constraints"].values()),
        "bounded_synthetic_optimizer_steps": ledger.synthetic_optimizer_steps
        == int(synthetic["optimizer_cycles_per_component"]) * 4,
        "only_v62_dataset_phase_is_authorized": harness["authorized_next_action"]
        == "authorize_v62_non_target_decoupled_rank_state_dataset_only",
    }
    checks = {name: bool(value) for name, value in checks.items()}
    smoke = {
        "ranker_parameters": ranker_parameters,
        "state_gate_parameters": gate_parameters,
        "total_parameters": ranker_parameters + gate_parameters,
        "ranker_output_shapes": {
            name: list(ranker_output[name].shape) for name in RANKER_HEADS
        },
        "state_gate_output_shape": list(state_output.shape),
        "reconstruction_shape": list(reconstruction.shape),
        "excess_scale": excess_scale,
        "market_scale": market_scale,
        "optimizer_steps_executed": ledger.synthetic_optimizer_steps,
        "logical_optimizer_steps_per_component": int(
            synthetic["optimizer_cycles_per_component"]
        ),
        "resume_equivalent": checks[
            "checkpoint_roundtrip_and_interrupted_resume"
        ],
        "policy_actions": policy_result["actions"],
        "base_total_turnover": float(base_accounting["total_turnover"]),
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
    }
    replay_payload = {
        "checks": checks,
        "smoke": smoke,
        "operation_ledger": operation_ledger,
        "checkpoint_metadata": checkpoint_metadata,
        "ranker_state_sha256": _state_dict_sha256(resumed_ranker),
        "state_gate_state_sha256": _state_dict_sha256(resumed_gate),
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


def run_decoupled_rank_state_harness(config: dict[str, Any]) -> dict[str, Any]:
    harness = config["decoupled_rank_state_harness"]
    root = Path(harness.get("project_root", ".")).resolve()
    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: root / relative for name, relative in harness["inputs"].items()}
    if set(paths) != EXPECTED_INPUT_NAMES:
        raise ValueError("V61 runtime input allowlist drift")
    if any(path.suffix in {".parquet", ".pt", ".pth", ".ckpt"} for path in paths.values()):
        raise ValueError("V61 input allowlist contains data or checkpoint")
    observed_before = {name: file_sha256(path) for name, path in paths.items()}
    if observed_before != harness["expected_input_sha256"]:
        failed = sorted(
            name
            for name, digest in observed_before.items()
            if digest != harness["expected_input_sha256"].get(name)
        )
        raise ValueError(f"V61 input receipt mismatch: {failed}")
    metadata_ledger = SyntheticAccessLedger()
    allowed_paths = {path.resolve() for path in paths.values()}
    loaded = {
        name: _load_allowed_json(path, allowed_paths, metadata_ledger)
        for name, path in paths.items()
    }
    specification = loaded["v60_specification"]
    blueprint = loaded["v60_blueprint"]
    v60_audit = loaded["v60_audit"]
    v60_result = loaded["v60_result"]
    v60_manifest = loaded["v60_artifact_manifest"]
    v60_completion = loaded["v60_completion_receipt"]
    canonical_expected = harness["expected_canonical_sha256"]
    metadata_contract_passes = (
        _canonical_self_hash(specification, "specification_sha256")
        and specification["specification_sha256"]
        == canonical_expected["specification"]
        and _canonical_self_hash(blueprint, "blueprint_sha256")
        and blueprint["blueprint_sha256"] == canonical_expected["blueprint"]
        and _canonical_self_hash(v60_result, "result_sha256")
        and v60_result["result_sha256"] == canonical_expected["result"]
        and _canonical_self_hash(v60_manifest, "artifact_manifest_sha256")
        and v60_manifest["artifact_manifest_sha256"]
        == canonical_expected["artifact_manifest"]
        and _canonical_self_hash(v60_completion, "completion_receipt_sha256")
        and v60_completion["completion_receipt_sha256"]
        == canonical_expected["completion_receipt"]
        and v60_audit.get("passed") is True
        and v60_result.get("decision")
        == "authorize_v61_synthetic_decoupled_rank_state_harness_only"
        and v60_completion.get("decision")
        == "authorize_v61_synthetic_decoupled_rank_state_harness_only"
        and v60_completion.get("audit_passed") is True
    )
    if not metadata_contract_passes:
        raise ValueError("V61 V60 authorization packet is not canonical")

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
        "all_v60_input_hashes_match": observed_before
        == harness["expected_input_sha256"],
        "input_allowlist_is_exactly_six_v60_metadata_artifacts": set(paths)
        == EXPECTED_INPUT_NAMES,
        "v60_authorization_packet_is_canonical": metadata_contract_passes,
        **first["checks"],
        "byte_identical_replay": byte_identical_replay,
        "input_hashes_still_match_after_harness": all(
            file_sha256(paths[name]) == digest
            for name, digest in observed_before.items()
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    audit = {
        "schema_version": "v61-decoupled-rank-state-audit/v1",
        "passed": all(checks.values()),
        "checks": checks,
    }
    decision = (
        harness["authorized_next_action"]
        if audit["passed"]
        else "keep_v62_and_later_unauthorized"
    )
    input_receipt = {
        name: {
            "path": str(paths[name].relative_to(root)),
            "sha256": observed_before[name],
        }
        for name in sorted(paths)
    }
    replay_receipt = {
        "schema_version": "v61-internal-replay-receipt/v1",
        "core_execution_sha256": canonical_sha256(first["replay_payload"]),
        "checkpoint_sha256": hashlib.sha256(first["checkpoint_bytes"]).hexdigest(),
        "byte_identical": byte_identical_replay,
    }
    replay_receipt["replay_receipt_sha256"] = canonical_sha256(replay_receipt)
    result: dict[str, Any] = {
        "schema_version": "v61-decoupled-rank-state-result/v1",
        "version": "v61",
        "family_id": blueprint["candidate_family_id"],
        "decision": decision,
        "harness_spec_sha256": harness_spec["harness_spec_sha256"],
        "input_hash_receipt": input_receipt,
        "smoke": first["smoke"],
        "operation_ledger": first["operation_ledger"],
        "replay_receipt": replay_receipt,
        "audit": audit,
    }
    result["result_sha256"] = canonical_sha256(result)
    report = "\n".join(
        [
            "# V61 Synthetic Decoupled Rank/State Harness",
            "",
            f"Decision: **{decision}**",
            "",
            f"Harness SHA-256: `{harness_spec['harness_spec_sha256']}`",
            f"Ranker parameters: **{first['smoke']['ranker_parameters']:,}**",
            f"State-gate parameters: **{first['smoke']['state_gate_parameters']:,}**",
            f"Total parameters: **{first['smoke']['total_parameters']:,}**",
            "",
            "The exact V60 ranker and independent state gate passed causal-prefix,",
            "asset-permutation, state-invariance, decomposition, gradient-isolation,",
            "independent-optimizer, interrupted-resume, cost-policy, liquidation, and",
            "byte-identical synthetic replay checks on deterministic CPU tensors.",
            "",
            "No Parquet, real label, prior checkpoint, outcome, performance/PnL,",
            "or BTC/ETH/SOL data was opened. V62 is dataset-construction only; real",
            "training, evaluation, and target assets remain unauthorized.",
            "",
        ]
    )

    (output / "synthetic_checkpoint.pt").write_bytes(first["checkpoint_bytes"])
    write_json_atomic(output / "harness_spec.json", harness_spec)
    write_json_atomic(output / "input_hash_receipt.json", input_receipt)
    write_json_atomic(output / "checkpoint_metadata.json", first["checkpoint_metadata"])
    write_json_atomic(output / "replay_receipt.json", replay_receipt)
    write_json_atomic(output / "smoke.json", first["smoke"])
    write_json_atomic(output / "audit.json", audit)
    write_json_atomic(output / "result.json", result)
    write_yaml_atomic(output / "resolved_config.yaml", config)
    (output / "report.md").write_text(report, encoding="utf-8")
    manifest_names = (
        "audit.json",
        "checkpoint_metadata.json",
        "harness_spec.json",
        "input_hash_receipt.json",
        "replay_receipt.json",
        "report.md",
        "resolved_config.yaml",
        "result.json",
        "smoke.json",
        "synthetic_checkpoint.pt",
    )
    manifest = {
        "schema_version": "v61-artifact-manifest/v1",
        "files": {name: file_sha256(output / name) for name in manifest_names},
    }
    manifest["artifact_manifest_sha256"] = canonical_sha256(manifest)
    write_json_atomic(output / "artifact_manifest.json", manifest)
    completion = {
        "schema_version": "v61-completion-receipt/v1",
        "decision": decision,
        "family_id": blueprint["candidate_family_id"],
        "harness_spec_sha256": harness_spec["harness_spec_sha256"],
        "result_sha256": result["result_sha256"],
        "result_file_sha256": file_sha256(output / "result.json"),
        "artifact_manifest_sha256": manifest["artifact_manifest_sha256"],
        "artifact_manifest_file_sha256": file_sha256(
            output / "artifact_manifest.json"
        ),
        "audit_passed": audit["passed"],
    }
    completion["completion_receipt_sha256"] = canonical_sha256(completion)
    write_json_atomic(output / "completion_receipt.json", completion)
    if not audit["passed"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"V61 synthetic harness failed: {failed}")
    return result
