from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from tlm.research_workflow import (
    ResearchStateError,
    _validate_v83_low_turnover_rank_training_boundary,
    validate_research_state,
)


ROOT = Path(__file__).resolve().parents[1]


def _yaml(relative: str) -> dict:
    return yaml.safe_load((ROOT / relative).read_text())


def _v83_staging_state() -> dict:
    state = _yaml("research/current.yaml")
    state.update(
        {
            "active_family_status": "dataset_passed_training_authorized",
            "last_completed_phase": "v82_non_target_low_turnover_rank_dataset",
            "last_completed_result": "artifacts/v82_low_turnover_rank_dataset/result.json",
            "authorized_next_action": "authorize_v83_frozen_non_target_low_turnover_rank_training_only",
            "authorized_phase": "v83",
            "authorized_command": (
                "PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src python3 -m tlm "
                "low-turnover-rank-training --config configs/v83_low_turnover_rank_training.yaml"
            ),
            "evidence_tier": "causal_non_target_training_only",
        }
    )
    return state


def test_current_state_authorizes_only_v84_prepare() -> None:
    status = validate_research_state(ROOT, "research/current.yaml")
    assert status["passed"] is True
    assert status["authorized_phase"] == "v85"
    assert status["authorized_next_action"] == (
        "execute_v85_exactly_one_registered_non_target_outcome_unseal_and_complete_evaluation"
    )
    assert status["target_asset_status"] == "sealed"
    assert status["deployable_strategy"] is False


def test_v83_boundary_excludes_evaluation_and_target_data() -> None:
    contract = _yaml("research/phase_contracts/v083.yaml")
    allowed = set(contract["access_contract"]["allowed_inputs"])
    assert "data/processed/low_turnover_rank_evaluation_features_v82.parquet" not in allowed
    assert "data/processed/low_turnover_rank_evaluation_outcomes_v82.parquet" not in allowed
    _validate_v83_low_turnover_rank_training_boundary(
        ROOT, _v83_staging_state(), _yaml("research/experiments/v083.yaml"), contract
    )

    drift = deepcopy(contract)
    drift["sampling_contract"]["train_samples_per_epoch"] += 1
    with pytest.raises(ResearchStateError, match="frozen training contract drift"):
        _validate_v83_low_turnover_rank_training_boundary(
            ROOT, _v83_staging_state(), _yaml("research/experiments/v083.yaml"), drift
        )
