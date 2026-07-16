from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from tlm.joint_absolute_relative_harness import (
    run_joint_absolute_relative_harness,
)
from tlm.joint_absolute_relative_model import (
    JOINT_HEADS,
    JointAbsoluteRelativeTransformer,
    fit_raw_return_rms_scale,
    joint_absolute_relative_loss,
    joint_triplet_positions,
    load_joint_checkpoint,
    save_joint_checkpoint,
)
from tlm.joint_absolute_relative_spec import (
    _canonical_sha256,
    _sha256_file,
    analytic_joint_parameter_count,
)


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _fixture_config(tmp_path: Path) -> dict:
    v47_config = yaml.safe_load(
        Path("configs/v47_joint_absolute_relative_triplet_spec.yaml").read_text(
            encoding="utf-8"
        )
    )["joint_absolute_relative_spec"]
    blueprint = {
        "version": "v47",
        "candidate_family_id": v47_config["candidate_family_id"],
        "architecture": v47_config["architecture"],
        "objective": v47_config["objective"],
        "early_stopping": v47_config["early_stopping"],
        "training": v47_config["training"],
        "policy": v47_config["policy"],
        "later_evaluation": v47_config["later_evaluation"],
    }
    blueprint["blueprint_sha256"] = _canonical_sha256(blueprint)
    payloads = {
        "v47_result": {
            "decision": "authorize_v48_joint_absolute_relative_synthetic_harness_only",
            "blueprint_sha256": blueprint["blueprint_sha256"],
        },
        "v47_blueprint": blueprint,
        "v47_audit": {"passed": True},
    }

    config = deepcopy(
        yaml.safe_load(
            Path("configs/v48_joint_absolute_relative_harness.yaml").read_text(
                encoding="utf-8"
            )
        )
    )
    harness = config["joint_absolute_relative_harness"]
    harness["project_root"] = str(tmp_path)
    for name, payload in payloads.items():
        path = tmp_path / f"{name}.json"
        _write(path, payload)
        harness["inputs"][name] = path.name
        harness["expected_input_sha256"][name] = _sha256_file(path)
    config["output_dir"] = "output"
    return config


def test_joint_model_matches_frozen_parameter_and_head_contract() -> None:
    spec = yaml.safe_load(
        Path("configs/v47_joint_absolute_relative_triplet_spec.yaml").read_text(
            encoding="utf-8"
        )
    )["joint_absolute_relative_spec"]
    model = JointAbsoluteRelativeTransformer(9, spec["architecture"])
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_212_930
    assert analytic_joint_parameter_count(spec["architecture"], 9) == 1_212_930
    assert tuple(model.prediction_heads) == JOINT_HEADS
    assert not hasattr(model, "mask_token")
    assert not hasattr(model, "reconstruction_head")


def test_joint_loss_centers_excess_reconstructs_absolute_and_excludes_ties() -> None:
    output = {
        "excess_score_z": torch.tensor([[2.0, 2.0, -1.0]]),
        "market_component_z": torch.tensor([[0.3, 0.6, 0.0]]),
    }
    returns = torch.tensor([[0.01, 0.01, -0.02]])
    loss = joint_absolute_relative_loss(
        output, returns, 0.01, tie_tolerance=1e-12
    )
    assert int(loss["pair_count"]) == 2
    assert torch.allclose(loss["e_hat_z"].sum(dim=1), torch.zeros(1))
    assert torch.allclose(loss["mu_hat_z"].mean(dim=1), loss["m_hat_z"])
    assert torch.allclose(
        loss["mu_hat_z"], loss["m_hat_z"][:, None] + loss["e_hat_z"]
    )
    assert torch.isfinite(loss["total"])


def test_raw_return_scale_is_train_only_and_floored() -> None:
    returns = torch.tensor([[0.01, -0.02, 0.03], [999.0, 999.0, 999.0]])
    mask = torch.tensor([True, False])
    expected = float(torch.sqrt(torch.mean(returns[:1].square())))
    assert fit_raw_return_rms_scale(returns, mask, 1e-6) == expected
    assert fit_raw_return_rms_scale(
        torch.zeros(2, 3), torch.ones(2, dtype=torch.bool), 1e-6
    ) == 1e-6


def test_cost_aware_policy_uses_one_third_and_tie_priority() -> None:
    mu = np.array(
        [
            [0.003, 0.0, 0.0],
            [0.001, 0.003, 0.0],
            [-0.003, -0.004, -0.005],
        ]
    )
    excess = np.array([[3, 2, 1], [2, 3, 1], [3, 2, 1]], dtype=float)
    positions = joint_triplet_positions(mu, excess, np.ones_like(mu, dtype=bool))
    assert np.argmax(positions[0]) == 0
    assert np.argmax(positions[1]) == 0  # switch edge equals cost: keep incumbent
    assert positions[2].sum() == 0
    assert np.all(positions.sum(axis=1) <= 1 / 3 + 1e-12)


def test_joint_checkpoint_rejects_metadata_drift(tmp_path: Path) -> None:
    spec = yaml.safe_load(
        Path("configs/v47_joint_absolute_relative_triplet_spec.yaml").read_text(
            encoding="utf-8"
        )
    )["joint_absolute_relative_spec"]
    model = JointAbsoluteRelativeTransformer(9, spec["architecture"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    metadata = {"job": "synthetic"}
    path = tmp_path / "checkpoint.pt"
    save_joint_checkpoint(
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
            "architecture": spec["architecture"],
            "input_features": 9,
        },
        "test_v1",
    )
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_joint_checkpoint(
            path,
            expected_format_version="test_v1",
            expected_architecture=spec["architecture"],
            expected_metadata={"job": "changed"},
        )


def test_v48_harness_passes_and_replays_bytes(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    first = run_joint_absolute_relative_harness(config)
    output = tmp_path / "output"
    first_bytes = {path.name: path.read_bytes() for path in output.iterdir()}
    second = run_joint_absolute_relative_harness(config)
    second_bytes = {path.name: path.read_bytes() for path in output.iterdir()}
    assert first["decision"] == "authorize_v49_purged_non_target_training_only"
    assert first["audit"]["passed"]
    assert first["smoke"]["parameter_count"] == 1_212_930
    assert first["smoke"]["resume_equivalent"]
    assert first_bytes == second_bytes
