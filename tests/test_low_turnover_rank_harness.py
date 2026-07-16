from __future__ import annotations

from pathlib import Path

import torch

from tlm.low_turnover_rank_model import (
    LowTurnoverRankModel,
    apply_low_turnover_policy,
    low_turnover_rank_loss,
)


def test_low_turnover_rank_model_shape_count_causality_and_permutation() -> None:
    torch.manual_seed(81)
    model = LowTurnoverRankModel()
    model.eval()
    values = torch.randn(4, 128, 3, 8)
    sequence = model.forward_sequence(values)
    scores = sequence[:, -1]

    assert sum(parameter.numel() for parameter in model.parameters()) == 10993
    assert sequence.shape == (4, 128, 3)
    assert scores.shape == (4, 3)
    assert torch.allclose(scores.sum(dim=1), torch.zeros(4), atol=1.0e-6)

    permutation = torch.tensor([2, 0, 1])
    assert torch.allclose(
        model(values[:, :, permutation]), scores[:, permutation], atol=1.0e-6
    )

    changed = values.clone()
    changed[:, 64:] += 100.0
    changed_sequence = model.forward_sequence(changed)
    assert torch.allclose(sequence[:, 63], changed_sequence[:, 63], atol=1.0e-6)


def test_low_turnover_rank_loss_is_finite_and_backward_safe() -> None:
    torch.manual_seed(82)
    model = LowTurnoverRankModel()
    values = torch.randn(4, 128, 3, 8)
    targets = torch.randn(4, 3)
    scores = model(values)
    loss, components = low_turnover_rank_loss(scores, targets)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in components.values())
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_structural_policy_turnover_bound_is_exact() -> None:
    scores = torch.zeros(160, 3)
    for decision_number, index in enumerate(range(0, 160, 21)):
        scores[index] = -0.5
        scores[index, decision_number % 3] = 1.0
    result = apply_low_turnover_policy(scores, torch.ones(160, dtype=torch.bool))
    assert result["decisions"] == 8
    assert result["turnover"] == 16.0
    assert result["final_liquidation_turnover"] == 1.0
    assert result["actions"] == {
        "cash": 0,
        "enter": 1,
        "exit": 0,
        "hold": 0,
        "switch": 7,
    }
