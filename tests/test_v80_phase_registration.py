from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from tlm.research_workflow import (
    ResearchStateError,
    _validate_v80_low_turnover_rank_specification_boundary,
    validate_research_state,
)


ROOT = Path(__file__).resolve().parents[1]


def _yaml(relative: str) -> dict:
    value = yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _v80_staging_state() -> dict:
    state = deepcopy(_yaml("research/current.yaml"))
    state.update(
        {
            "authorized_phase": "v80",
            "authorized_next_action": (
                "authorize_v80_low_turnover_cross_sectional_rank_specification_only"
            ),
            "authorized_command": (
                "PYTHONPATH=src python3 -m tlm low-turnover-rank-spec "
                "--config configs/v80_low_turnover_rank_spec.yaml"
            ),
            "active_family_status": (
                "outcome_blind_specification_authorized_not_started"
            ),
            "last_completed_phase": "v79_metadata_only_v78_terminal_record",
            "last_completed_result": (
                "artifacts/v79_v78_terminal_record/result.json"
            ),
            "evidence_tier": "metadata_only_final_family_specification",
        }
    )
    state["safety"].update(
        {
            "v80_metadata_only_specification_phase": True,
            "v80_data_checkpoint_model_training_inference_or_outcome_allowed": False,
            "v80_target_assets_remain_sealed": True,
            "v80_single_final_family": True,
        }
    )
    return state


def test_current_state_authorizes_only_v84_prepare() -> None:
    status = validate_research_state(ROOT, "research/current.yaml")
    assert status["passed"] is True
    assert status["family_count"] == 8
    assert status["trained_family_count"] == 8
    assert status["retired_family_count"] == 6
    assert status["authorized_phase"] == "v85"
    assert status["authorized_next_action"] == (
        "execute_v85_exactly_one_registered_non_target_outcome_unseal_and_complete_evaluation"
    )
    assert status["target_asset_status"] == "sealed"
    assert status["deployable_strategy"] is False


def test_v80_boundary_is_new_metadata_only_and_single_variant() -> None:
    state = _v80_staging_state()
    experiment = _yaml("research/experiments/v080.yaml")
    contract = _yaml("research/phase_contracts/v080.yaml")
    _validate_v80_low_turnover_rank_specification_boundary(
        ROOT, state, experiment, contract
    )

    scope = contract["specification_scope"]
    assert scope["family_is_new"] is True
    assert scope["single_final_family"] is True
    assert scope["single_frozen_variant_required"] is True
    assert scope["turnover_budget_must_be_structural"] is True
    assert scope["parameter_increase_as_default_direction"] is False
    assert scope["prior_checkpoint_weight_scaler_or_optimizer_reuse_allowed"] is False
    assert contract["target_contract"]["status"] == "sealed"


def test_v80_boundary_rejects_scientific_or_checkpoint_input() -> None:
    state = _v80_staging_state()
    experiment = _yaml("research/experiments/v080.yaml")
    contract = deepcopy(_yaml("research/phase_contracts/v080.yaml"))
    forbidden = "data/processed/selected_universe_panel_v32.parquet"
    contract["access_contract"]["allowed_inputs"].append(forbidden)
    contract["input_contract"]["expected_static_file_sha256_by_path"][forbidden] = (
        "0" * 64
    )
    with pytest.raises(ResearchStateError, match="metadata-only input contract"):
        _validate_v80_low_turnover_rank_specification_boundary(
            ROOT, state, experiment, contract
        )
