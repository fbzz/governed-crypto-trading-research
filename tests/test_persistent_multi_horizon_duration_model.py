from __future__ import annotations

import math

import pytest
import torch

from tlm.persistent_multi_horizon_duration_model import (
    PersistentMultiHorizonDurationTransformer,
    default_persistent_duration_architecture,
    explicit_duration_negative_log_likelihood,
    persistent_multi_task_loss,
)


def _small_architecture() -> dict:
    architecture = default_persistent_duration_architecture()
    architecture.update({
        "input_features": 4,
        "lookback_days": 32,
        "patch_length_days": 8,
        "patch_stride_days": 4,
        "d_model": 32,
        "temporal_encoder_layers": 1,
        "cross_asset_attention_layers": 1,
        "attention_heads": 4,
        "feed_forward_width": 64,
        "dropout": 0.0,
    })
    return architecture


def test_forward_contract_joint_loss_and_backward_are_finite() -> None:
    torch.manual_seed(74)
    model = PersistentMultiHorizonDurationTransformer(_small_architecture())
    features = torch.randn(2, 32, 3, 4)
    output = model(features, round_trip_cost=0.001)

    assert output["gross_location"].shape == (2, 3, 3)
    assert output["market_location"].shape == (2, 3)
    assert output["survival_probability"].shape == (2, 3, 7)
    assert output["expected_holding_days"].shape == (2, 3)
    assert torch.allclose(
        output["excess_location"].sum(dim=1), torch.zeros(2, 3), atol=1e-6
    )
    assert bool((output["gross_scale"] > 0).all())
    assert bool(
        (
            output["survival_probability"][..., 1:]
            <= output["survival_probability"][..., :-1]
        ).all()
    )
    assert torch.allclose(
        output["gross_location"] - output["net_location"],
        torch.full((2, 3, 3), 0.001),
        atol=1e-6,
    )

    returns = torch.randn(2, 3, 3) * 0.02
    duration_days = torch.tensor([[2, 4, 7], [1, 3, 6]])
    censored = torch.tensor([[False, False, True], [False, True, False]])
    losses = persistent_multi_task_loss(
        output, returns, duration_days, censored
    )
    losses["total"].backward()
    assert losses["pair_count"].item() > 0
    assert bool(torch.isfinite(losses["total"]))
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def test_asset_permutation_and_temporal_prefix_contracts_hold() -> None:
    torch.manual_seed(75)
    model = PersistentMultiHorizonDurationTransformer(_small_architecture()).eval()
    features = torch.randn(2, 32, 3, 4)
    permutation = torch.tensor([2, 0, 1])

    with torch.no_grad():
        original = model(features, round_trip_cost=0.002)
        permuted = model(features[:, :, permutation], round_trip_cost=0.002)

    asset_outputs = {
        "excess_location",
        "gross_location",
        "gross_scale",
        "net_location",
        "survival_probability",
        "expected_holding_days",
        "persistent_net_score",
    }
    for name in asset_outputs:
        assert torch.allclose(
            permuted[name], original[name][:, permutation], atol=1e-5, rtol=1e-5
        )
    for name in {"market_location", "market_scale"}:
        assert torch.allclose(
            permuted[name], original[name], atol=1e-5, rtol=1e-5
        )

    changed = features.clone()
    changed[:, -1] += 10.0
    with torch.no_grad():
        encoded = model.encode_temporal_patches(features)
        changed_encoded = model.encode_temporal_patches(changed)
    assert torch.allclose(encoded[:, :, :-1], changed_encoded[:, :, :-1], atol=1e-6)
    assert not torch.allclose(encoded[:, :, -1], changed_encoded[:, :, -1])


def test_explicit_duration_likelihood_respects_event_and_censoring() -> None:
    logits = torch.zeros(1, 2, 7)
    durations = torch.tensor([[1, 3]])
    censored = torch.tensor([[False, True]])
    loss = explicit_duration_negative_log_likelihood(logits, durations, censored)
    assert loss.item() == pytest.approx(2.0 * math.log(2.0))

    with pytest.raises(ValueError, match="duration support"):
        explicit_duration_negative_log_likelihood(
            logits, torch.tensor([[0, 3]]), censored
        )


def test_default_candidate_capacity_is_deliberately_moderate() -> None:
    model = PersistentMultiHorizonDurationTransformer(
        default_persistent_duration_architecture()
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert 1_000_000 < parameter_count < 2_000_000
