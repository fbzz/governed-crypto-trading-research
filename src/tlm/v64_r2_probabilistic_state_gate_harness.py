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
from .decoupled_rank_state_harness import (
    RANKER_HEADS,
    _build_ranker,
    _clone_state_dict,
    _optimizer_step_count,
    _state_dict_sha256,
    derive_state_features,
)
from .v64_r2_probabilistic_state_gate_spec import (
    ranker_parameter_count,
    state_gate_parameter_count,
)


EXPECTED_INPUT_NAMES = {
    "v65_specification",
    "v65_blueprint",
    "v65_audit",
    "v65_result",
    "v65_artifact_manifest",
    "v65_completion_receipt",
}


class ProbabilisticStateGate(nn.Module):
    """The exact independent 27,522-parameter V64-R2 state encoder."""

    def __init__(
        self,
        architecture: dict[str, Any],
        *,
        degrees_of_freedom: float,
        scale_floor: float,
    ) -> None:
        super().__init__()
        self.input_features = int(architecture["input_features"])
        self.lookback_days = int(architecture["lookback_days"])
        self.patch_length = int(architecture["patch_length_days"])
        self.patch_stride = int(architecture["patch_stride_days"])
        self.d_model = int(architecture["d_model"])
        self.output_width = int(architecture["output_width"])
        self.patch_count = (
            (self.lookback_days - self.patch_length) // self.patch_stride + 1
        )
        self.degrees_of_freedom = float(degrees_of_freedom)
        self.scale_floor = float(scale_floor)
        heads = int(architecture["attention_heads"])
        if self.d_model % heads:
            raise ValueError("State-gate d_model must divide attention heads")
        if self.patch_count < 1:
            raise ValueError("State-gate patch geometry does not fit lookback")
        if self.output_width != 2:
            raise ValueError("V64-R2 state gate must emit location and raw scale")
        if architecture.get("independent_encoder") is not True:
            raise ValueError("V64-R2 state gate must be independent")
        if architecture.get("ranker_representation_input") != "none":
            raise ValueError("V64-R2 state gate cannot consume ranker representations")
        if self.degrees_of_freedom != 5.0:
            raise ValueError("V64-R2 Student-t degrees of freedom are frozen at five")
        if not math.isfinite(self.scale_floor) or self.scale_floor <= 0:
            raise ValueError("V64-R2 scale floor must be finite and positive")

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
        self.output_head = nn.Linear(self.d_model, self.output_width)
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
        return (
            x.unfold(1, self.patch_length, self.patch_stride)
            .permute(0, 1, 3, 2)
            .contiguous()
        )

    def encode_temporal_patches(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.extract_patches(x)
        tokens = self.patch_projection(patches.flatten(start_dim=2))
        tokens = self.patch_norm(tokens + self.temporal_position)
        encoded = self.temporal_encoder(tokens, mask=self.causal_patch_mask)
        return self.temporal_norm(encoded)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.output_head(self.encode_temporal_patches(x)[:, -1])
        raw_scale = raw[:, 1]
        return {
            "location": raw[:, 0],
            "raw_scale": raw_scale,
            "scale": F.softplus(raw_scale) + self.scale_floor,
        }


def student_t_df5_cdf_standardized(value: np.ndarray | float) -> np.ndarray:
    """Closed-form CDF for a standardized Student-t with five degrees of freedom."""

    z = np.asarray(value, dtype=np.float64)
    u = z / math.sqrt(5.0)
    one_plus_square = 1.0 + np.square(u)
    return 0.5 + (
        np.arctan(u)
        + u / one_plus_square
        + (2.0 * u) / (3.0 * np.square(one_plus_square))
    ) / math.pi


def probability_of_clearing_cost(
    market_location: np.ndarray,
    market_scale: np.ndarray,
    *,
    asset_excess: float,
    transition_cost: float,
    degrees_of_freedom: float,
) -> float:
    location = np.asarray(market_location, dtype=np.float64)
    scale = np.asarray(market_scale, dtype=np.float64)
    if location.ndim != 1 or scale.shape != location.shape or not len(location):
        raise ValueError("Market mixture location and scale must share [components]")
    if not np.isfinite(location).all() or not np.isfinite(scale).all():
        raise ValueError("Market mixture parameters must be finite")
    if (scale <= 0).any():
        raise ValueError("Market mixture scale must be positive")
    if degrees_of_freedom != 5.0:
        raise ValueError("Only the frozen Student-t df=5 distribution is allowed")
    if not math.isfinite(asset_excess) or not math.isfinite(transition_cost):
        raise ValueError("Asset excess and transition cost must be finite")
    standardized = (float(transition_cost) - (location + float(asset_excess))) / scale
    survival = 1.0 - student_t_df5_cdf_standardized(standardized)
    return float(np.mean(survival))


def passes_abstention(probability: float, threshold: float) -> bool:
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("Probability must be finite and in [0,1]")
    if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("Abstention threshold must be in (0,1)")
    return probability >= threshold


def student_t_negative_log_likelihood(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    *,
    degrees_of_freedom: float,
) -> torch.Tensor:
    if set(output) != {"location", "raw_scale", "scale"}:
        raise ValueError("State-gate output contract drift")
    if output["location"].shape != target.shape or output["scale"].shape != target.shape:
        raise ValueError("State-gate output and target shapes must match")
    distribution = torch.distributions.StudentT(
        df=float(degrees_of_freedom),
        loc=output["location"],
        scale=output["scale"],
    )
    return -distribution.log_prob(target).mean()


def _manual_student_t_negative_log_likelihood(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    *,
    degrees_of_freedom: float,
) -> torch.Tensor:
    df = float(degrees_of_freedom)
    location = output["location"]
    scale = output["scale"]
    constant = (
        math.lgamma((df + 1.0) / 2.0)
        - math.lgamma(df / 2.0)
        - 0.5 * math.log(df * math.pi)
    )
    standardized_square = torch.square((target - location) / scale)
    log_probability = (
        constant
        - torch.log(scale)
        - ((df + 1.0) / 2.0) * torch.log1p(standardized_square / df)
    )
    return -log_probability.mean()


def probabilistic_rank_state_positions(
    predicted_raw_excess: np.ndarray,
    market_location: np.ndarray,
    market_scale: np.ndarray,
    momentum_30: np.ndarray,
    eligible: np.ndarray,
    *,
    base_cost: float,
    switch_hurdle: float,
    probability_threshold: float,
    degrees_of_freedom: float,
    risky_weight: float = 1.0,
) -> dict[str, Any]:
    excess = np.asarray(predicted_raw_excess, dtype=np.float64)
    location = np.asarray(market_location, dtype=np.float64)
    scale = np.asarray(market_scale, dtype=np.float64)
    momentum = np.asarray(momentum_30, dtype=np.float64)
    eligibility = np.asarray(eligible, dtype=bool)
    if excess.ndim != 2 or excess.shape[1] != 3:
        raise ValueError("Excess forecasts must have [days,3] shape")
    if location.ndim != 2 or scale.shape != location.shape:
        raise ValueError("Market mixture parameters must have [days,components] shape")
    if location.shape[0] != excess.shape[0] or location.shape[1] < 1:
        raise ValueError("Market mixture day count must match excess forecasts")
    if momentum.shape != excess.shape or eligibility.shape != excess.shape:
        raise ValueError("Excess, momentum, and eligibility shapes must match")
    if not np.isfinite(location).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Market mixture parameters must be finite with positive scale")
    if not math.isfinite(base_cost) or base_cost < 0:
        raise ValueError("Base cost must be finite and nonnegative")
    if not math.isfinite(switch_hurdle) or switch_hurdle < 0:
        raise ValueError("Switch hurdle must be finite and nonnegative")
    if not 0 < risky_weight <= 1:
        raise ValueError("Risky weight must be in (0,1]")

    positions = np.zeros_like(excess)
    actions: list[str] = []
    selected_assets: list[int | None] = []
    event_probabilities: list[float | None] = []
    transition_costs: list[float | None] = []
    incumbent: int | None = None
    entry_cost = base_cost * risky_weight
    switch_cost = base_cost * 2.0 * risky_weight
    for day in range(len(excess)):
        valid = eligibility[day] & np.isfinite(excess[day]) & np.isfinite(momentum[day])
        valid_indexes = np.flatnonzero(valid)
        if incumbent is not None and not valid[incumbent]:
            incumbent = None
            actions.append("forced_exit")
            selected_assets.append(None)
            event_probabilities.append(None)
            transition_costs.append(None)
            continue
        if len(valid_indexes) == 0 or np.all(momentum[day, valid_indexes] <= 0):
            actions.append("momentum_exit" if incumbent is not None else "cash")
            incumbent = None
            selected_assets.append(None)
            event_probabilities.append(None)
            transition_costs.append(None)
            continue

        best_value = np.max(excess[day, valid_indexes])
        challenger = int(
            valid_indexes[np.flatnonzero(excess[day, valid_indexes] == best_value)[0]]
        )
        if incumbent is None:
            probability = probability_of_clearing_cost(
                location[day],
                scale[day],
                asset_excess=float(excess[day, challenger]),
                transition_cost=entry_cost,
                degrees_of_freedom=degrees_of_freedom,
            )
            if passes_abstention(probability, probability_threshold):
                incumbent = challenger
                actions.append("entry")
            else:
                actions.append("cash")
            event_probabilities.append(probability)
            transition_costs.append(entry_cost)
        else:
            hold_probability = probability_of_clearing_cost(
                location[day],
                scale[day],
                asset_excess=float(excess[day, incumbent]),
                transition_cost=0.0,
                degrees_of_freedom=degrees_of_freedom,
            )
            if not passes_abstention(hold_probability, probability_threshold):
                incumbent = None
                actions.append("probability_exit")
                event_probabilities.append(hold_probability)
                transition_costs.append(0.0)
            elif challenger != incumbent and (
                excess[day, challenger] - excess[day, incumbent] > switch_hurdle
            ):
                switch_probability = probability_of_clearing_cost(
                    location[day],
                    scale[day],
                    asset_excess=float(excess[day, challenger]),
                    transition_cost=switch_cost,
                    degrees_of_freedom=degrees_of_freedom,
                )
                if passes_abstention(switch_probability, probability_threshold):
                    incumbent = challenger
                    actions.append("switch")
                else:
                    actions.append("hold")
                event_probabilities.append(switch_probability)
                transition_costs.append(switch_cost)
            else:
                actions.append("hold")
                event_probabilities.append(hold_probability)
                transition_costs.append(0.0)
        if incumbent is not None:
            positions[day, incumbent] = risky_weight
        selected_assets.append(incumbent)
    return {
        "positions": positions,
        "actions": actions,
        "selected_assets": selected_assets,
        "event_probabilities": event_probabilities,
        "transition_costs": transition_costs,
    }


def _standardized_quantile_df5(cdf_probability: float) -> float:
    if not 0.0 < cdf_probability < 1.0:
        raise ValueError("CDF probability must be in (0,1)")
    lower = -100.0
    upper = 100.0
    for _ in range(200):
        midpoint = (lower + upper) / 2.0
        if float(student_t_df5_cdf_standardized(midpoint)) < cdf_probability:
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / 2.0


def _market_location_for_probability(
    event_probability: float,
    *,
    asset_excess: float,
    transition_cost: float,
    scale: float,
) -> float:
    standardized = _standardized_quantile_df5(1.0 - event_probability)
    return float(transition_cost - standardized * scale - asset_excess)


def _clone_optimizer_state(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    return deepcopy(optimizer.state_dict())


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
        raise ValueError("V66 synthetic checkpoint format drift")
    if payload.get("architecture") != expected_architecture:
        raise ValueError("V66 synthetic checkpoint architecture drift")
    if payload.get("architecture_sha256") != canonical_sha256(expected_architecture):
        raise ValueError("V66 synthetic checkpoint architecture hash drift")
    if payload.get("metadata") != expected_metadata:
        raise ValueError("V66 synthetic checkpoint metadata drift")
    if "ranker_optimizer_state" in payload:
        raise ValueError("V66 synthetic checkpoint must not contain ranker optimizer")
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
        raise PermissionError(f"V66 metadata read is not allowlisted: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    ledger.authorized_metadata_reads += 1
    return value


def _build_harness_spec(
    blueprint: dict[str, Any], harness: dict[str, Any]
) -> dict[str, Any]:
    value = {
        "schema_version": "v66-v64-r2-probabilistic-state-gate-harness-spec/v1",
        "version": harness["version"],
        "lineage_label": blueprint["lineage_label"],
        "candidate_family_id": blueprint["candidate_family_id"],
        "v65_blueprint_sha256": blueprint["blueprint_sha256"],
        "ranker_contract": blueprint["ranker_contract"],
        "ranker_identity_receipts": blueprint["ranker_identity_receipts"],
        "state_gate_architecture": blueprint["state_gate_architecture"],
        "probabilistic_gate": blueprint["probabilistic_gate"],
        "decomposition": blueprint["decomposition"],
        "policy": blueprint["policy"],
        "synthetic": harness["synthetic"],
        "mechanics": harness["mechanics"],
        "checkpoint": harness["checkpoint"],
        "constraints": harness["constraints"],
        "authorized_next_action": harness["authorized_next_action"],
    }
    value["harness_spec_sha256"] = canonical_sha256(value)
    return value


def _gate_step(
    model: ProbabilisticStateGate,
    optimizer: torch.optim.Optimizer,
    state_features: torch.Tensor,
    target: torch.Tensor,
    *,
    degrees_of_freedom: float,
    gradient_clip_norm: float,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(state_features)
    loss = student_t_negative_log_likelihood(
        output, target, degrees_of_freedom=degrees_of_freedom
    )
    loss.backward()
    gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
    optimizer.step()
    return {
        "state_gate_nll": float(loss.detach()),
        "state_gate_gradient_norm": float(gradient_norm),
    }


def _policy_fixture(
    policy: dict[str, Any], mechanics: dict[str, Any], synthetic: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if int(synthetic["policy_days"]) != 7:
        raise ValueError("V66 policy fixture is frozen to seven synthetic days")
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
    momentum = np.ones_like(excess)
    momentum[5] = -1.0
    eligible = np.ones_like(excess, dtype=bool)
    components = int(synthetic["mixture_components"])
    # Keep the synthetic mixture broad enough that the incumbent hold gate can
    # pass before the challenger is assessed against the stricter switch cost.
    component_scale = 0.01
    market_scale = np.full((7, components), component_scale, dtype=np.float64)
    base_cost = float(policy["base_cost_bps"]) / 10_000.0
    switch_cost = base_cost * 2.0 * float(policy["risky_gross"])
    target_probability = [
        float(mechanics["entry_probability_fixture"]),
        float(mechanics["hold_probability_fixture"]),
        float(mechanics["switch_probability_fixture"]),
        float(mechanics["hold_probability_fixture"]),
        float(mechanics["exit_probability_fixture"]),
        0.50,
        float(mechanics["entry_probability_fixture"]),
    ]
    tested_asset = [0, 0, 1, 1, 1, 0, 2]
    transition_cost = [base_cost, 0.0, switch_cost, 0.0, 0.0, 0.0, base_cost]
    market_location = np.empty((7, components), dtype=np.float64)
    for day in range(7):
        value = _market_location_for_probability(
            target_probability[day],
            asset_excess=float(excess[day, tested_asset[day]]),
            transition_cost=float(transition_cost[day]),
            scale=component_scale,
        )
        market_location[day] = value
    result = probabilistic_rank_state_positions(
        excess,
        market_location,
        market_scale,
        momentum,
        eligible,
        base_cost=base_cost,
        switch_hurdle=float(policy["switch_hurdle"]),
        probability_threshold=float(policy["abstention_probability_threshold"]),
        degrees_of_freedom=float(mechanics["degrees_of_freedom"]),
        risky_weight=float(policy["risky_gross"]),
    )
    accounting = {
        str(cost): persistent_portfolio_returns(
            result["positions"], np.zeros_like(excess), float(cost)
        )
        for cost in policy["reporting_cost_bps"]
    }
    return result, {
        "accounting": accounting,
        "target_probabilities": target_probability,
        "transition_costs": transition_cost,
    }


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
    ranker_architecture = blueprint["ranker_contract"]["architecture"]
    gate_architecture = blueprint["state_gate_architecture"]
    probabilistic = blueprint["probabilistic_gate"]
    training = blueprint["future_training_contract"]
    policy = blueprint["policy"]
    if synthetic["device"] != "cpu":
        raise ValueError("V66 harness is frozen to deterministic CPU")
    if int(synthetic["input_features"]) != 9 or int(synthetic["state_features"]) != 18:
        raise ValueError("V66 synthetic feature count drift")
    if float(mechanics["degrees_of_freedom"]) != float(
        probabilistic["degrees_of_freedom"]
    ):
        raise ValueError("V66 degrees-of-freedom contract drift")
    if float(mechanics["scale_floor"]) != float(probabilistic["scale_floor"]):
        raise ValueError("V66 scale-floor contract drift")

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    ranker = _build_ranker(ranker_architecture)
    for parameter in ranker.parameters():
        parameter.requires_grad_(False)
    ranker.eval()
    ranker_initial_sha256 = _state_dict_sha256(ranker)
    gate_prototype = ProbabilisticStateGate(
        gate_architecture,
        degrees_of_freedom=float(mechanics["degrees_of_freedom"]),
        scale_floor=float(mechanics["scale_floor"]),
    )
    gate_base_state = _clone_state_dict(gate_prototype)
    ranker_parameters = sum(parameter.numel() for parameter in ranker.parameters())
    gate_parameters = sum(parameter.numel() for parameter in gate_prototype.parameters())

    batch_size = int(synthetic["batch_size"])
    feature_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    target_generator = torch.Generator(device="cpu").manual_seed(seed + 2)
    features = torch.randn(
        batch_size,
        256,
        3,
        9,
        generator=feature_generator,
        dtype=torch.float32,
    )
    state_features = derive_state_features(features)
    target_market_z = torch.randn(
        batch_size, generator=target_generator, dtype=torch.float32
    )
    ledger = SyntheticAccessLedger(
        authorized_metadata_reads=authorized_metadata_reads,
        synthetic_tensor_generations=2,
    )
    optimizer_kwargs = {
        "lr": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
    }
    cycles = int(synthetic["gate_optimizer_cycles"])

    torch.manual_seed(seed + 3)
    start_rng = torch.get_rng_state().clone()
    full_gate = ProbabilisticStateGate(
        gate_architecture,
        degrees_of_freedom=float(mechanics["degrees_of_freedom"]),
        scale_floor=float(mechanics["scale_floor"]),
    )
    full_gate.load_state_dict(gate_base_state)
    full_optimizer = torch.optim.AdamW(full_gate.parameters(), **optimizer_kwargs)
    torch.set_rng_state(start_rng)
    full_history = []
    for _ in range(cycles):
        full_history.append(
            _gate_step(
                full_gate,
                full_optimizer,
                state_features,
                target_market_z,
                degrees_of_freedom=float(mechanics["degrees_of_freedom"]),
                gradient_clip_norm=float(training["gradient_clip_norm"]),
            )
        )
        ledger.synthetic_optimizer_steps += 1

    interrupted_gate = ProbabilisticStateGate(
        gate_architecture,
        degrees_of_freedom=float(mechanics["degrees_of_freedom"]),
        scale_floor=float(mechanics["scale_floor"]),
    )
    interrupted_gate.load_state_dict(gate_base_state)
    interrupted_optimizer = torch.optim.AdamW(
        interrupted_gate.parameters(), **optimizer_kwargs
    )
    torch.set_rng_state(start_rng)
    first_history = _gate_step(
        interrupted_gate,
        interrupted_optimizer,
        state_features,
        target_market_z,
        degrees_of_freedom=float(mechanics["degrees_of_freedom"]),
        gradient_clip_norm=float(training["gradient_clip_norm"]),
    )
    ledger.synthetic_optimizer_steps += 1
    saved_rng = torch.get_rng_state().clone()
    checkpoint_metadata = {
        "candidate_family_id": blueprint["candidate_family_id"],
        "lineage_label": blueprint["lineage_label"],
        "v65_blueprint_sha256": blueprint["blueprint_sha256"],
        "v66_harness_spec_sha256": harness_spec["harness_spec_sha256"],
        "initialization_seed": seed,
        "status": checkpoint_contract["status"],
        "ranker_frozen": True,
        "ranker_optimizer_present": False,
    }
    checkpoint_architecture = {
        "ranker": ranker_architecture,
        "state_gate": gate_architecture,
        "probabilistic_gate": probabilistic,
    }
    checkpoint_payload = {
        "architecture": checkpoint_architecture,
        "ranker_state": _clone_state_dict(ranker),
        "state_gate_state": _clone_state_dict(interrupted_gate),
        "state_gate_optimizer_state": _clone_optimizer_state(interrupted_optimizer),
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
    resumed_ranker.load_state_dict(checkpoint["ranker_state"])
    for parameter in resumed_ranker.parameters():
        parameter.requires_grad_(False)
    resumed_ranker.eval()
    resumed_gate = ProbabilisticStateGate(
        gate_architecture,
        degrees_of_freedom=float(mechanics["degrees_of_freedom"]),
        scale_floor=float(mechanics["scale_floor"]),
    )
    resumed_gate.load_state_dict(checkpoint["state_gate_state"])
    roundtrip_gate_sha256 = _state_dict_sha256(resumed_gate)
    resumed_optimizer = torch.optim.AdamW(resumed_gate.parameters(), **optimizer_kwargs)
    resumed_optimizer.load_state_dict(checkpoint["state_gate_optimizer_state"])
    torch.set_rng_state(checkpoint["cpu_rng_state"])
    resumed_history = [first_history]
    for _ in range(1, cycles):
        resumed_history.append(
            _gate_step(
                resumed_gate,
                resumed_optimizer,
                state_features,
                target_market_z,
                degrees_of_freedom=float(mechanics["degrees_of_freedom"]),
                gradient_clip_norm=float(training["gradient_clip_norm"]),
            )
        )
        ledger.synthetic_optimizer_steps += 1

    resumed_gate.eval()
    permutation = torch.tensor(mechanics["asset_permutation"])
    cutoff = int(mechanics["causal_cutoff_day"])
    with torch.no_grad():
        ranker_output = resumed_ranker(features)
        permuted_ranker_output = resumed_ranker(features[:, :, permutation, :])
        gate_output = resumed_gate(state_features)
        permuted_state_features = derive_state_features(features[:, :, permutation, :])
        permuted_gate_output = resumed_gate(permuted_state_features)
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
    early_patch_count = (
        (cutoff - resumed_ranker.patch_length) // resumed_ranker.patch_stride + 1
    )

    for parameter in resumed_ranker.parameters():
        parameter.grad = None
    for parameter in resumed_gate.parameters():
        parameter.grad = None
    isolation_output = resumed_gate(state_features)
    isolation_loss = student_t_negative_log_likelihood(
        isolation_output,
        target_market_z,
        degrees_of_freedom=float(mechanics["degrees_of_freedom"]),
    )
    manual_nll = _manual_student_t_negative_log_likelihood(
        isolation_output,
        target_market_z,
        degrees_of_freedom=float(mechanics["degrees_of_freedom"]),
    )
    isolation_loss.backward()
    gate_gradients_finite = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in resumed_gate.parameters()
    )
    ranker_gradients_after_gate = any(
        parameter.grad is not None for parameter in resumed_ranker.parameters()
    )

    policy_result, policy_fixture = _policy_fixture(policy, mechanics, synthetic)
    accounting = policy_fixture["accounting"]
    base_accounting = accounting[str(policy["base_cost_bps"])]
    expected_actions = [
        "entry",
        "hold",
        "switch",
        "hold",
        "probability_exit",
        "cash",
        "entry",
    ]
    expected_turnover = np.array([1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0])
    probability_tolerance = float(mechanics["probability_absolute_tolerance"])
    observed_policy_probabilities = [
        probability
        for probability in policy_result["event_probabilities"]
        if probability is not None
    ]
    expected_policy_probabilities = [
        policy_fixture["target_probabilities"][index]
        for index in (0, 1, 2, 3, 4, 6)
    ]
    optimizer_parameters = {
        id(parameter)
        for group in resumed_optimizer.param_groups
        for parameter in group["params"]
    }
    ranker_parameter_ids = {id(parameter) for parameter in resumed_ranker.parameters()}
    gate_parameter_ids = {id(parameter) for parameter in resumed_gate.parameters()}
    operation_ledger = ledger.to_dict()
    checks = {
        "exact_component_and_total_parameter_counts": ranker_parameters
        == ranker_parameter_count(ranker_architecture)
        == int(ranker_architecture["expected_parameter_count"])
        == 1_231_634
        and gate_parameters
        == state_gate_parameter_count(gate_architecture)
        == int(gate_architecture["expected_parameter_count"])
        == 27_522
        and ranker_parameters + gate_parameters == 1_259_156,
        "exact_output_shapes_and_positive_scale": set(ranker_output) == set(RANKER_HEADS)
        and all(
            tuple(ranker_output[name].shape) == (batch_size, 3)
            for name in RANKER_HEADS
        )
        and all(tuple(gate_output[name].shape) == (batch_size,) for name in gate_output)
        and bool(torch.all(gate_output["scale"] > float(mechanics["scale_floor"]))),
        "student_t_nll_matches_closed_form": torch.allclose(
            isolation_loss.detach(),
            manual_nll.detach(),
            atol=float(mechanics["nll_absolute_tolerance"]),
            rtol=0.0,
        )
        and math.isfinite(float(isolation_loss.detach())),
        "student_t_df5_cdf_matches_registered_reference": math.isclose(
            float(student_t_df5_cdf_standardized(0.0)), 0.5, abs_tol=1.0e-15
        )
        and math.isclose(
            float(student_t_df5_cdf_standardized(1.0)),
            0.8183912661754387,
            abs_tol=1.0e-15,
        )
        and math.isclose(
            float(student_t_df5_cdf_standardized(-1.0)),
            1.0 - float(student_t_df5_cdf_standardized(1.0)),
            abs_tol=1.0e-15,
        ),
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
        "asset_permutation_equivariance_and_state_invariance": all(
            torch.allclose(
                permuted_ranker_output[name],
                ranker_output[name][:, permutation],
                atol=1.0e-5,
                rtol=1.0e-5,
            )
            for name in RANKER_HEADS
        )
        and torch.allclose(
            permuted_state_features, state_features, atol=1.0e-6, rtol=1.0e-6
        )
        and all(
            torch.allclose(
                permuted_gate_output[name],
                gate_output[name],
                atol=1.0e-5,
                rtol=1.0e-5,
            )
            for name in gate_output
        ),
        "ranker_is_frozen_without_gradient_or_optimizer": all(
            not parameter.requires_grad for parameter in resumed_ranker.parameters()
        )
        and not ranker_gradients_after_gate
        and ranker_initial_sha256 == _state_dict_sha256(resumed_ranker)
        and ranker_parameter_ids.isdisjoint(optimizer_parameters),
        "gate_has_finite_gradients_and_only_gate_optimizer": gate_gradients_finite
        and optimizer_parameters == gate_parameter_ids
        and ranker_parameter_ids.isdisjoint(gate_parameter_ids),
        "gate_checkpoint_roundtrip_and_interrupted_resume": roundtrip_gate_sha256
        == _state_dict_sha256(interrupted_gate)
        and _state_dict_sha256(full_gate) == _state_dict_sha256(resumed_gate)
        and full_history == resumed_history
        and _optimizer_step_count(full_optimizer) == cycles
        and _optimizer_step_count(resumed_optimizer) == cycles
        and torch.equal(checkpoint["cpu_rng_state"], saved_rng)
        and "ranker_optimizer_state" not in checkpoint,
        "abstention_threshold_boundary_is_exact": passes_abstention(0.60, 0.60)
        and not passes_abstention(float(np.nextafter(0.60, 0.0)), 0.60),
        "transition_specific_probabilities_are_exact": np.allclose(
            observed_policy_probabilities,
            expected_policy_probabilities,
            atol=probability_tolerance,
            rtol=0.0,
        )
        and policy_result["transition_costs"]
        == [
            policy_fixture["transition_costs"][0],
            policy_fixture["transition_costs"][1],
            policy_fixture["transition_costs"][2],
            policy_fixture["transition_costs"][3],
            policy_fixture["transition_costs"][4],
            None,
            policy_fixture["transition_costs"][6],
        ],
        "exact_probability_entry_hold_switch_exit_and_final_liquidation": policy_result[
            "actions"
        ]
        == expected_actions
        and np.allclose(base_accounting["turnover"], expected_turnover)
        and math.isclose(float(base_accounting["total_turnover"]), 6.0)
        and all(
            np.allclose(values["turnover"], base_accounting["turnover"])
            and np.allclose(
                values["cost"], values["turnover"] * (float(cost) / 10_000.0)
            )
            and np.allclose(
                values["net_return"], values["gross_return"] - values["cost"]
            )
            for cost, values in accounting.items()
        ),
        "ranker_relative_policy_and_risk_contract_are_preserved": policy[
            "desired_asset"
        ]
        == "highest_context_and_seed_averaged_raw_excess"
        and policy["momentum_cash_gate"]
        == "all_currently_eligible_momentum_30_nonpositive"
        and policy["switch_hurdle"] == 0.002
        and policy["action_space"] == ["long_one_asset", "cash"]
        and policy["final_liquidation"] is True
        and policy["leverage"] is False
        and policy["shorting"] is False,
        "zero_real_data_checkpoint_outcome_or_target_access": ledger.forbidden_operations_are_zero()
        and not any(bool(value) for value in harness["constraints"].values()),
        "bounded_synthetic_gate_optimizer_steps": ledger.synthetic_optimizer_steps
        == cycles * 2,
        "only_v67_dataset_phase_is_authorized": harness["authorized_next_action"]
        == "authorize_v67_non_target_v64_r2_probabilistic_state_gate_dataset_only",
    }
    checks = {name: bool(value) for name, value in checks.items()}
    smoke = {
        "ranker_parameters": ranker_parameters,
        "state_gate_parameters": gate_parameters,
        "total_parameters": ranker_parameters + gate_parameters,
        "ranker_output_shapes": {
            name: list(ranker_output[name].shape) for name in RANKER_HEADS
        },
        "state_gate_output_shapes": {
            name: list(gate_output[name].shape) for name in gate_output
        },
        "minimum_state_scale": float(gate_output["scale"].min()),
        "student_t_degrees_of_freedom": float(mechanics["degrees_of_freedom"]),
        "state_gate_nll": float(isolation_loss.detach()),
        "optimizer_steps_executed": ledger.synthetic_optimizer_steps,
        "logical_gate_optimizer_steps": cycles,
        "ranker_optimizer_present": False,
        "ranker_requires_grad": False,
        "resume_equivalent": checks[
            "gate_checkpoint_roundtrip_and_interrupted_resume"
        ],
        "policy_actions": policy_result["actions"],
        "policy_probabilities": policy_result["event_probabilities"],
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


def run_v64_r2_probabilistic_state_gate_harness(
    config: dict[str, Any],
) -> dict[str, Any]:
    harness = config["v64_r2_probabilistic_state_gate_harness"]
    root = Path(harness.get("project_root", ".")).resolve()
    configured_output = Path(config["output_dir"])
    output = configured_output if configured_output.is_absolute() else root / configured_output
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: root / relative for name, relative in harness["inputs"].items()}
    if set(paths) != EXPECTED_INPUT_NAMES:
        raise ValueError("V66 runtime input allowlist drift")
    if any(path.suffix in {".parquet", ".pt", ".pth", ".ckpt"} for path in paths.values()):
        raise ValueError("V66 input allowlist contains data or checkpoint")
    observed_before = {name: file_sha256(path) for name, path in paths.items()}
    if observed_before != harness["expected_input_sha256"]:
        failed = sorted(
            name
            for name, digest in observed_before.items()
            if digest != harness["expected_input_sha256"].get(name)
        )
        raise ValueError(f"V66 input receipt mismatch: {failed}")

    metadata_ledger = SyntheticAccessLedger()
    allowed_paths = {path.resolve() for path in paths.values()}
    loaded = {
        name: _load_allowed_json(path, allowed_paths, metadata_ledger)
        for name, path in paths.items()
    }
    specification = loaded["v65_specification"]
    blueprint = loaded["v65_blueprint"]
    v65_audit = loaded["v65_audit"]
    v65_result = loaded["v65_result"]
    v65_manifest = loaded["v65_artifact_manifest"]
    v65_completion = loaded["v65_completion_receipt"]
    canonical_expected = harness["expected_canonical_sha256"]
    metadata_contract_passes = (
        _canonical_self_hash(specification, "specification_sha256")
        and specification["specification_sha256"]
        == canonical_expected["specification"]
        and _canonical_self_hash(blueprint, "blueprint_sha256")
        and blueprint["blueprint_sha256"] == canonical_expected["blueprint"]
        and _canonical_self_hash(v65_result, "result_sha256")
        and v65_result["result_sha256"] == canonical_expected["result"]
        and _canonical_self_hash(v65_manifest, "artifact_manifest_sha256")
        and v65_manifest["artifact_manifest_sha256"]
        == canonical_expected["artifact_manifest"]
        and _canonical_self_hash(v65_completion, "completion_receipt_sha256")
        and v65_completion["completion_receipt_sha256"]
        == canonical_expected["completion_receipt"]
        and v65_audit.get("passed") is True
        and v65_result.get("decision")
        == "authorize_v66_synthetic_v64_r2_probabilistic_state_gate_harness_only"
        and v65_completion.get("decision")
        == "authorize_v66_synthetic_v64_r2_probabilistic_state_gate_harness_only"
        and v65_completion.get("audit_passed") is True
        and specification["ranker_contract"] == blueprint["ranker_contract"]
        and specification["state_gate_architecture"]
        == blueprint["state_gate_architecture"]
        and specification["probabilistic_gate"] == blueprint["probabilistic_gate"]
        and specification["policy"] == blueprint["policy"]
    )
    if not metadata_contract_passes:
        raise ValueError("V66 V65 authorization packet is not canonical")

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
        "all_v65_input_hashes_match": observed_before
        == harness["expected_input_sha256"],
        "input_allowlist_is_exactly_six_v65_metadata_artifacts": set(paths)
        == EXPECTED_INPUT_NAMES,
        "v65_authorization_packet_is_canonical": metadata_contract_passes,
        "exact_nine_ranker_identity_receipts_are_preserved": len(
            blueprint["ranker_identity_receipts"]
        )
        == 9,
        **first["checks"],
        "byte_identical_replay": byte_identical_replay,
        "input_hashes_still_match_after_harness": all(
            file_sha256(paths[name]) == digest
            for name, digest in observed_before.items()
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    audit = {
        "schema_version": "v66-v64-r2-probabilistic-state-gate-audit/v1",
        "passed": all(checks.values()),
        "checks": checks,
    }
    decision = (
        harness["authorized_next_action"]
        if audit["passed"]
        else "retire_v64_r2_without_data_training_or_evaluation"
    )
    input_receipt = {
        name: {
            "path": str(paths[name].relative_to(root)),
            "sha256": observed_before[name],
        }
        for name in sorted(paths)
    }
    replay_receipt = {
        "schema_version": "v66-internal-replay-receipt/v1",
        "core_execution_sha256": canonical_sha256(first["replay_payload"]),
        "checkpoint_sha256": hashlib.sha256(first["checkpoint_bytes"]).hexdigest(),
        "byte_identical": byte_identical_replay,
    }
    replay_receipt["replay_receipt_sha256"] = canonical_sha256(replay_receipt)
    result: dict[str, Any] = {
        "schema_version": "v66-v64-r2-probabilistic-state-gate-result/v1",
        "version": "v66",
        "lineage_label": blueprint["lineage_label"],
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
            "# V66 Synthetic V64-R2 Probabilistic State-Gate Harness",
            "",
            f"Decision: **{decision}**",
            "",
            f"Harness SHA-256: `{harness_spec['harness_spec_sha256']}`",
            f"Frozen ranker parameters: **{first['smoke']['ranker_parameters']:,}**",
            f"Probabilistic gate parameters: **{first['smoke']['state_gate_parameters']:,}**",
            f"Total parameters: **{first['smoke']['total_parameters']:,}**",
            "",
            "The exact ranker architecture remained gradient-free and had no",
            "optimizer. The Student-t gate passed location/scale, positive-scale,",
            "closed-form NLL/CDF, causal-prefix, permutation, gate-only optimizer,",
            "interrupted-resume, 60% abstention, transition-cost, accounting, and",
            "byte-identical synthetic replay checks on deterministic CPU tensors.",
            "",
            "No real data, prior checkpoint, V64 gate state, real prediction,",
            "performance/PnL, outcome source/packet, or BTC/ETH/SOL data was opened.",
            "V67 is dataset-construction only and has not been implemented.",
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
        "schema_version": "v66-artifact-manifest/v1",
        "files": {name: file_sha256(output / name) for name in manifest_names},
    }
    manifest["artifact_manifest_sha256"] = canonical_sha256(manifest)
    write_json_atomic(output / "artifact_manifest.json", manifest)
    completion = {
        "schema_version": "v66-completion-receipt/v1",
        "decision": decision,
        "family_id": blueprint["candidate_family_id"],
        "lineage_label": blueprint["lineage_label"],
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
        raise RuntimeError(f"V66 synthetic harness failed: {failed}")
    return result
