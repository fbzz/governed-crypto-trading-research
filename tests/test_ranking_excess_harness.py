from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from tlm.patch_transformer import MultiAssetPatchTransformer
from tlm.ranking_excess_harness import (
    RANKING_EXCESS_HEADS,
    _sha256_file,
    aggregate_raw_excess_predictions,
    fit_shared_asset_ridge,
    fit_triplet_excess_rms_scale,
    load_ranking_excess_checkpoint,
    normalized_triplet_excess,
    predict_shared_asset_ridge,
    ranking_excess_loss,
    ranking_excess_positions,
    run_ranking_excess_harness,
    save_ranking_excess_checkpoint,
)


def _architecture() -> dict:
    config = yaml.safe_load(
        Path("configs/v41_ranking_excess_spec.yaml").read_text(encoding="utf-8")
    )
    return config["ranking_excess_spec"]["architecture"]


def test_excess_scale_and_targets_are_train_scaled_and_zero_sum() -> None:
    returns = torch.tensor([
        [0.03, 0.01, -0.02],
        [0.00, 0.02, 0.01],
    ])
    train_mask = torch.tensor([True, True])
    scale = fit_triplet_excess_rms_scale(returns, train_mask, floor=1e-6)
    target = normalized_triplet_excess(returns, scale)
    assert scale > 0
    assert torch.allclose(target.sum(dim=1), torch.zeros(2), atol=1e-6)
    assert torch.isclose(torch.sqrt(target.square().mean()), torch.tensor(1.0))


def test_ranking_excess_loss_rewards_order_and_is_score_shift_invariant() -> None:
    returns = torch.tensor([
        [0.03, 0.01, -0.02],
        [-0.01, 0.04, 0.00],
    ])
    scale = fit_triplet_excess_rms_scale(
        returns, torch.tensor([True, True]), floor=1e-6
    )
    target = normalized_triplet_excess(returns, scale)
    volatility = torch.full_like(returns, 0.05)
    labels = torch.stack([returns, volatility], dim=-1)

    def calculate(scores: torch.Tensor) -> dict[str, torch.Tensor]:
        return ranking_excess_loss(
            {
                "excess_return_z": scores,
                "log_volatility_7d": volatility.log(),
            },
            labels,
            scale,
            tie_tolerance=1e-12,
            volatility_floor=1e-6,
            ranking_weight=1.0,
            excess_weight=1.0,
            volatility_weight=0.1,
        )

    aligned = calculate(target)
    shifted = calculate(target + 10.0)
    reversed_order = calculate(-target)
    assert torch.allclose(aligned["core"], shifted["core"])
    assert aligned["ranking"] < reversed_order["ranking"]
    assert int(aligned["pair_count"]) == 6


def test_pairwise_tie_tolerance_is_applied_in_raw_return_units() -> None:
    returns = torch.tensor([[0.0, 5e-13, 0.02]], dtype=torch.float64)
    volatility = torch.full_like(returns, 0.05)
    labels = torch.stack([returns, volatility], dim=-1)
    result = ranking_excess_loss(
        {
            "excess_return_z": torch.zeros_like(returns),
            "log_volatility_7d": volatility.log(),
        },
        labels,
        scale=0.01,
        tie_tolerance=1e-12,
        volatility_floor=1e-6,
        ranking_weight=1.0,
        excess_weight=1.0,
        volatility_weight=0.1,
    )
    assert int(result["pair_count"]) == 2


def test_normalized_member_scores_convert_to_centered_raw_excess() -> None:
    z_scores = np.array([
        [[2.0, 1.0, 0.0]],
        [[0.0, 1.0, 2.0]],
    ])
    result = aggregate_raw_excess_predictions(
        z_scores,
        np.array([0.01, 0.02]),
    )
    np.testing.assert_allclose(result, [[-0.005, 0.0, 0.005]])


def test_turnover_hurdle_holds_switches_and_absolute_gate_returns_to_cash() -> None:
    scores = np.array([
        [0.0030, 0.0010, 0.0000],
        [0.0010, 0.0025, 0.0000],
        [0.0010, 0.0031, 0.0000],
        [0.0030, 0.0020, 0.0010],
    ])
    momentum = np.array([
        [0.10, 0.05, -0.01],
        [0.09, 0.04, -0.02],
        [0.08, 0.03, -0.03],
        [-0.01, -0.02, -0.03],
    ])
    eligible = np.ones_like(momentum, dtype=bool)
    positions = ranking_excess_positions(
        scores, momentum, eligible, switch_hurdle=0.002
    )
    selected = [int(np.argmax(row)) if row.sum() else None for row in positions]
    assert selected == [0, 0, 1, None]


def test_policy_ignores_unavailable_values_and_drops_ineligible_incumbent() -> None:
    scores = np.array([
        [0.003, 0.001, 0.000],
        [np.nan, 0.002, 0.001],
    ])
    momentum = np.array([
        [0.10, 0.05, 0.01],
        [np.nan, 0.05, 0.01],
    ])
    eligible = np.array([
        [True, True, True],
        [False, True, True],
    ])
    positions = ranking_excess_positions(
        scores, momentum, eligible, switch_hurdle=0.002
    )
    np.testing.assert_array_equal(positions.argmax(axis=1), [0, 1])
    assert not positions[~eligible].any()


def test_medium_model_accepts_only_the_frozen_ranking_heads() -> None:
    architecture = _architecture()
    model = MultiAssetPatchTransformer(
        9,
        architecture,
        expected_prediction_heads=RANKING_EXCESS_HEADS,
    )
    output = model(torch.zeros(2, 256, 3, 9))
    assert set(output) == set(RANKING_EXCESS_HEADS)
    assert all(value.shape == (2, 3) for value in output.values())
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_231_634


def test_shared_asset_ridge_predictions_are_triplet_centered() -> None:
    rng = np.random.default_rng(7)
    tensor = rng.normal(size=(8, 16, 3, 2))
    target = rng.normal(size=(8, 3))
    target -= target.mean(axis=1, keepdims=True)
    model = fit_shared_asset_ridge(tensor, target, alpha=10.0)
    prediction = predict_shared_asset_ridge(model, tensor)
    assert model.solution_form == "dual"
    assert prediction.shape == target.shape
    assert np.allclose(prediction.sum(axis=1), 0.0, atol=1e-10)


def test_shared_asset_ridge_uses_primal_form_when_rows_exceed_features() -> None:
    rng = np.random.default_rng(11)
    tensor = rng.normal(size=(20, 2, 3, 2))
    target = rng.normal(size=(20, 3))
    target -= target.mean(axis=1, keepdims=True)
    model = fit_shared_asset_ridge(tensor, target, alpha=10.0)
    prediction = predict_shared_asset_ridge(model, tensor)
    assert model.solution_form == "primal"
    assert prediction.shape == target.shape
    assert np.isfinite(prediction).all()
    assert np.allclose(prediction.sum(axis=1), 0.0, atol=1e-10)


def test_checkpoint_loader_rejects_semantically_different_v41_contract(
    tmp_path: Path,
) -> None:
    architecture = _architecture()
    model = MultiAssetPatchTransformer(
        9,
        architecture,
        expected_prediction_heads=RANKING_EXCESS_HEADS,
    )
    metadata = {
        "candidate_family_id": "tlm_cross_sectional_rank_excess_medium_v1",
        "v41_blueprint_sha256": "fixture-blueprint",
        "initialization_seed": 7,
        "checkpoint_status": "synthetic_smoke_only_not_trained_candidate",
    }
    path = tmp_path / "checkpoint.pt"
    save_ranking_excess_checkpoint(
        model,
        path,
        architecture,
        metadata,
        "fixture-format",
    )
    wrong_metadata = {**metadata, "initialization_seed": 42}
    try:
        load_ranking_excess_checkpoint(
            path,
            "fixture-format",
            9,
            architecture,
            wrong_metadata,
        )
    except ValueError as error:
        assert "metadata does not match V41" in str(error)
    else:
        raise AssertionError("Semantically different checkpoint was accepted")


def test_v42_end_to_end_smoke_uses_only_synthetic_inputs(tmp_path: Path) -> None:
    v42 = yaml.safe_load(
        Path("configs/v42_ranking_excess_harness.yaml").read_text(encoding="utf-8")
    )
    config = deepcopy(v42)
    blueprint = {
        "candidate_family_id": "tlm_cross_sectional_rank_excess_medium_v1",
        "blueprint_sha256": "fixture-blueprint",
        "architecture": _architecture(),
        "objective": {
            "scale_floor": 1e-6,
            "exact_tie_tolerance": 1e-12,
            "volatility_floor": 1e-6,
            "weights": {
                "ranking": 1.0,
                "excess": 1.0,
                "log_volatility": 0.1,
            },
        },
        "training": {
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 1.0,
        },
        "policy": {
            "switch_hurdle": 0.002,
            "reporting_cost_bps": [10, 20, 30],
            "base_cost_bps": 10,
        },
        "baselines": {"fixture": True},
    }
    payloads = {
        "v41_specification": {
            "decision": "authorize_v42_synthetic_ranking_excess_harness_only",
            "blueprint_sha256": "fixture-blueprint",
        },
        "v41_blueprint": blueprint,
        "v41_audit": {"passed": True},
    }
    harness = config["ranking_excess_harness"]
    harness["project_root"] = str(tmp_path)
    for name, payload in payloads.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        harness["inputs"][name] = path.name
        harness["expected_input_sha256"][name] = _sha256_file(path)
    config["output_dir"] = "output"

    result = run_ranking_excess_harness(config)
    assert result["decision"] == "authorize_v43_medium_non_target_pretraining_only"
    assert result["audit"]["passed"]
    assert all(result["audit"]["checks"].values())
    assert result["tested"]["real_panel_or_label_reads"] == 0
    assert result["tested"]["target_asset_loads"] == 0
    assert result["operation_ledger"]["authorized_metadata_reads"] == 3
    assert result["operation_ledger"]["synthetic_optimizer_steps"] == 2
