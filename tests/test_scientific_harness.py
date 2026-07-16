import numpy as np
import pandas as pd
import torch

from tlm.scientific_harness import (
    DeterministicEligibleTripletSampler,
    EarlyStoppingState,
    FeatureScaler,
    deterministic_patch_mask,
    persistent_portfolio_returns,
    q_policy_positions,
    supervised_probabilistic_loss,
)


def test_scaler_is_train_only_and_relative_uses_source_scale():
    dates = pd.date_range("2021-01-01", periods=6, freq="D", tz="UTC")
    panel = pd.DataFrame({
        "date": dates,
        "a": [1, 2, 3, 4, 999, 999],
        "b": [2, 4, 6, 8, -999, -999],
    })
    scaler = FeatureScaler.fit_from_panel(
        panel, ["a", "b"], "2021-01-01", "2021-01-04", "2021-01-04", "a"
    )
    assert scaler.mean == (2.5, 5.0)
    x = np.array([[[[2.5, 5.0, scaler.scale[0]]]]], dtype=np.float32)
    transformed = scaler.transform_triplet_tensor(x)
    np.testing.assert_allclose(transformed[..., :-1], 0.0)
    np.testing.assert_allclose(transformed[..., -1], 1.0)


def test_sampler_is_reproducible_and_only_emits_available_assets():
    dates = pd.date_range("2021-01-01", periods=3, freq="D", tz="UTC")
    availability = {
        dates[0]: ["A", "B", "C", "D"],
        dates[1]: ["A", "B", "C"],
        dates[2]: ["A", "B"],
    }
    sampler = DeterministicEligibleTripletSampler(
        availability, ["A", "B", "C", "D"], seed=9, fold=1
    )
    first = sampler.sample_epoch(2, 20)
    second = sampler.sample_epoch(2, 20)
    assert first == second
    assert all(set(row["triplet"]).issubset(availability[row["date"]]) for row in first)
    assert all(row["date"] != dates[2] for row in first)


def test_patch_mask_has_exact_count_and_replays():
    first = deterministic_patch_mask(5, 3, 31, 0.15, 42, 1, 2, 3)
    second = deterministic_patch_mask(5, 3, 31, 0.15, 42, 1, 2, 3)
    assert torch.equal(first, second)
    assert torch.all(first.flatten(1).sum(axis=1) == 14)


def test_probabilistic_losses_are_finite_and_differentiable():
    output = {
        name: torch.randn(4, 3, requires_grad=True)
        for name in ("return_q10", "return_q50", "return_q90", "volatility_7d")
    }
    labels = torch.stack([
        torch.randn(4, 3) * 0.02,
        torch.rand(4, 3) * 0.2 + 0.01,
    ], dim=-1)
    losses = supervised_probabilistic_loss(output, labels, 0.1, 1e-6)
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    assert all(value.grad is not None for value in output.values())


def test_policy_accounting_charges_switch_and_final_liquidation():
    q50 = np.array([[0.01, 0.0], [0.0, 0.02], [-0.01, -0.02]])
    q10 = q50 - 0.005
    positions = q_policy_positions(q10, q50, -0.03, 0.002)
    result = persistent_portfolio_returns(positions, np.zeros_like(q50), 10)
    np.testing.assert_allclose(result["turnover"], [1.0, 2.0, 1.0])
    assert result["total_cost"] == 0.004


def test_early_stopping_counts_consecutive_stale_epochs():
    state = EarlyStoppingState(patience=3)
    assert not state.update(0, 1.0)
    assert not state.update(1, 0.8)
    assert not state.update(2, 0.81)
    assert not state.update(3, 0.82)
    assert state.update(4, 0.83)
    assert state.best_epoch == 1
