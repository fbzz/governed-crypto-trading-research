from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch
import yaml

from tlm.state_conditioned_multi_horizon_model import (
    StateConditionedMultiHorizonTransformer,
    load_state_conditioned_checkpoint,
    multi_horizon_quantile_loss,
    save_state_conditioned_checkpoint,
    state_conditioned_weekly_policy,
)
from tlm.state_conditioned_multi_horizon_spec import analytic_parameter_count


def _architecture() -> dict:
    config = yaml.safe_load(
        open("configs/v55_state_conditioned_multi_horizon_spec.yaml", encoding="utf-8")
    )
    return deepcopy(config["state_conditioned_multi_horizon_spec"]["architecture"])


def test_model_matches_parameter_shape_causality_and_permutation_contract() -> None:
    torch.manual_seed(11)
    architecture = _architecture()
    model = StateConditionedMultiHorizonTransformer(architecture).eval()
    features = torch.randn(2, 256, 3, 9)
    with torch.no_grad():
        output = model(features)
        permutation = torch.tensor([2, 0, 1])
        permuted = model(features[:, :, permutation, :])
        temporal = model.encode_temporal_patches(features)
        altered = features.clone()
        altered[:, 200:, :, :] += 50.0
        altered_temporal = model.encode_temporal_patches(altered)
    early_patch_count = (200 - model.patch_length) // model.patch_stride + 1
    assert output.shape == (2, 3, 3, 3)
    assert torch.allclose(permuted, output[:, permutation], atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        temporal[:, :, :early_patch_count],
        altered_temporal[:, :, :early_patch_count],
        atol=1e-5,
        rtol=1e-5,
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 465_513
    assert analytic_parameter_count(architecture) == 465_513


def test_loss_freezes_pinball_ranknet_crossing_and_finite_gradients() -> None:
    torch.manual_seed(12)
    model = StateConditionedMultiHorizonTransformer(_architecture())
    predictions = model(torch.randn(2, 256, 3, 9))
    targets = torch.tensor(
        [
            [[0.01, 0.02, 0.03], [0.00, 0.01, 0.01], [-0.01, -0.02, -0.03]],
            [[0.01, 0.02, 0.02], [0.01, 0.02, 0.02], [0.00, 0.01, -0.01]],
        ],
        dtype=torch.float32,
    )
    losses = multi_horizon_quantile_loss(predictions, targets)
    losses["total"].backward()
    assert int(losses["pair_count"]) == 5
    assert all(torch.isfinite(losses[name]) for name in ("pinball", "ranking", "crossing", "total"))
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    ordered = torch.tensor([[[[-1.0, 0.0, 1.0]] * 3] * 3])
    crossed = ordered.clone()
    crossed[..., 0] = 2.0
    ordered_loss = multi_horizon_quantile_loss(ordered, torch.zeros(1, 3, 3))
    crossed_loss = multi_horizon_quantile_loss(crossed, torch.zeros(1, 3, 3))
    assert float(ordered_loss["crossing"]) == 0.0
    assert float(crossed_loss["crossing"]) > 0.0


def test_loss_masks_only_inactive_nonfinite_targets() -> None:
    predictions = torch.zeros(1, 3, 3, 3)
    targets = torch.zeros(1, 3, 3)
    targets[0, 0, 0] = float("nan")
    mask = torch.ones_like(targets, dtype=torch.bool)
    mask[0, 0, 0] = False
    assert torch.isfinite(
        multi_horizon_quantile_loss(predictions, targets, target_mask=mask)["total"]
    )
    mask[0, 0, 0] = True
    with pytest.raises(ValueError, match="Active targets"):
        multi_horizon_quantile_loss(predictions, targets, target_mask=mask)


def test_policy_freezes_zero_seven_clock_forced_cash_and_transition_costs() -> None:
    forecasts = np.full((15, 3), [-0.01, -0.02, -0.03], dtype=float)
    forecasts[0] = [0.006, 0.001, 0.000]
    forecasts[7] = [0.000, 0.009, 0.001]
    forecasts[14] = [0.000, 0.008, 0.012]
    eligible = np.ones_like(forecasts, dtype=bool)
    result = state_conditioned_weekly_policy(forecasts, eligible)
    positions = result["positions"]
    assert np.flatnonzero(result["decision_mask"]).tolist() == [0, 7, 14]
    assert np.argmax(positions[0]) == 0
    assert np.argmax(positions[6]) == 0
    assert np.argmax(positions[7]) == 1
    assert np.argmax(positions[14]) == 2
    assert np.allclose(positions.sum(axis=1)[positions.sum(axis=1) > 0], 1 / 3)


def test_missing_incumbent_forces_cash_without_same_day_replacement() -> None:
    forecasts = np.full((8, 3), [0.006, 0.005, 0.004], dtype=float)
    eligible = np.ones_like(forecasts, dtype=bool)
    eligible[3, 0] = False
    forecasts[3, 1] = 0.02
    result = state_conditioned_weekly_policy(forecasts, eligible)
    assert np.argmax(result["positions"][0]) == 0
    assert result["forced_cash_mask"][3]
    assert result["positions"][3].sum() == 0
    assert result["positions"][4].sum() == 0


def test_partial_triplet_does_not_advance_decision_clock() -> None:
    forecasts = np.full((9, 3), [0.006, 0.001, 0.000], dtype=float)
    eligible = np.ones_like(forecasts, dtype=bool)
    eligible[4, 2] = False
    result = state_conditioned_weekly_policy(forecasts, eligible)
    assert np.flatnonzero(result["decision_mask"]).tolist() == [0, 8]


def test_checkpoint_roundtrip_rejects_contract_drift(tmp_path) -> None:
    architecture = _architecture()
    model = StateConditionedMultiHorizonTransformer(architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    metadata = {"job_key": "synthetic/v56"}
    path = tmp_path / "checkpoint.pt"
    save_state_conditioned_checkpoint(
        path,
        {
            "model_state": model.state_dict(),
            "best_model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cpu_rng_state": torch.get_rng_state(),
            "mps_rng_state": None,
            "early_stopping_state": {},
            "history": [],
            "metadata": metadata,
            "architecture": architecture,
        },
        format_version="v56_test_v1",
    )
    restored = load_state_conditioned_checkpoint(
        path,
        expected_format_version="v56_test_v1",
        expected_architecture=architecture,
        expected_metadata=metadata,
    )
    assert all(
        torch.equal(model.state_dict()[name], tensor)
        for name, tensor in restored["model_state"].items()
    )
    changed = deepcopy(architecture)
    changed["dropout"] = 0.1
    with pytest.raises(ValueError, match="architecture hash"):
        load_state_conditioned_checkpoint(
            path,
            expected_format_version="v56_test_v1",
            expected_architecture=changed,
            expected_metadata=metadata,
        )
