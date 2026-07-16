from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from tlm.research_workflow import (
    ResearchStateError,
    _validate_v63_training_boundary,
    _validate_v64_evaluation_boundary,
    validate_research_state,
)


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    value = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _v63_state() -> dict:
    state = deepcopy(_load("research/current.yaml"))
    state.update(
        {
            "authorized_phase": "v63",
            "authorized_next_action": "authorize_v63_frozen_non_target_decoupled_rank_state_training_only",
            "active_family_status": "dataset_passed_training_authorized",
            "last_completed_phase": "v62_non_target_decoupled_rank_state_dataset",
            "last_completed_result": "artifacts/v62_non_target_decoupled_rank_state_dataset/result.json",
        }
    )
    return state


def _v64_evaluation_state() -> dict:
    state = deepcopy(_load("research/current.yaml"))
    state.update(
        {
            "authorized_phase": "v64",
            "authorized_next_action": (
                "authorize_v64_frozen_adaptive_development_evaluation_only"
            ),
            "active_family_status": (
                "trained_adaptive_development_evaluation_authorized"
            ),
            "last_completed_phase": (
                "v63_frozen_non_target_decoupled_rank_state_training"
            ),
        }
    )
    return state


def test_v63_completion_registers_only_v64_adaptive_evaluation() -> None:
    status = validate_research_state(ROOT)
    assert status["passed"] is True
    assert int(status["authorized_phase"].removeprefix("v").split("-", 1)[0]) >= 64
    assert status["family_count"] >= 5
    assert status["trained_family_count"] == 8
    assert status["target_asset_status"] == "sealed"


def test_v64_evaluation_boundary_accepts_registered_contract() -> None:
    _validate_v64_evaluation_boundary(
        ROOT,
        _v64_evaluation_state(),
        _load("research/experiments/v063.yaml"),
        _load("research/phase_contracts/v064.yaml"),
    )


def test_v63_training_boundary_accepts_registered_contract() -> None:
    _validate_v63_training_boundary(
        ROOT,
        _v63_state(),
        _load("research/experiments/v062.yaml"),
        _load("research/phase_contracts/v063.yaml"),
    )


def test_v63_training_boundary_rejects_input_drift() -> None:
    contract = deepcopy(_load("research/phase_contracts/v063.yaml"))
    contract["input_contract"]["expected_file_sha256_by_path"][
        "data/processed/decoupled_rank_state_labels_v62.parquet"
    ] = "0" * 64
    with pytest.raises(ResearchStateError, match="input drift"):
        _validate_v63_training_boundary(
            ROOT,
            _v63_state(),
            _load("research/experiments/v062.yaml"),
            contract,
        )


def test_v63_training_boundary_rejects_target_unseal() -> None:
    contract = deepcopy(_load("research/phase_contracts/v063.yaml"))
    contract["target_contract"]["status"] = "open"
    with pytest.raises(ResearchStateError, match="target/deployment"):
        _validate_v63_training_boundary(
            ROOT,
            _v63_state(),
            _load("research/experiments/v062.yaml"),
            contract,
        )


def test_v63_training_boundary_rejects_grid_expansion() -> None:
    contract = deepcopy(_load("research/phase_contracts/v063.yaml"))
    contract["grid_optimizer_and_runtime_contract"]["seeds"].append(999)
    with pytest.raises(ResearchStateError, match="model/grid/operator"):
        _validate_v63_training_boundary(
            ROOT,
            _v63_state(),
            _load("research/experiments/v062.yaml"),
            contract,
        )
