from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from tlm.research_workflow import (
    ResearchStateError,
    _validate_backup_receipt,
    _validate_v77_persistent_duration_training_boundary,
    validate_research_state,
)


ROOT = Path(__file__).resolve().parents[1]


def _yaml(relative: str) -> dict:
    value = yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _registered_v77_state() -> dict:
    state = _yaml("research/current.yaml")
    state.update(
        {
            "authorized_phase": "v77",
            "authorized_next_action": (
                "authorize_v77_frozen_non_target_persistent_duration_training_only"
            ),
            "authorized_command": (
                "PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src python3 -m tlm "
                "persistent-duration-training "
                "--config configs/v77_persistent_duration_training.yaml"
            ),
            "active_family_status": "dataset_passed_training_authorized",
            "last_completed_phase": "v76_non_target_persistent_duration_dataset",
            "last_completed_result": (
                "artifacts/v76_non_target_persistent_duration_dataset/result.json"
            ),
            "evidence_tier": "causal_non_target_training_only",
        }
    )
    state["safety"]["dataset_only_phase"] = False
    state["safety"]["training_only_phase"] = True
    return state


def test_current_state_advances_only_to_v84_prepare() -> None:
    status = validate_research_state(ROOT, "research/current.yaml")
    assert status["passed"] is True
    assert status["active_family_id"] == "tlm_low_turnover_cross_sectional_rank_v1"
    assert status["active_family_status"] == (
        "retrospective_non_target_economic_evaluation_exact_unseal_authorized"
    )
    assert status["authorized_phase"] == "v85"
    assert status["authorized_next_action"] == (
        "execute_v85_exactly_one_registered_non_target_outcome_unseal_and_complete_evaluation"
    )
    assert status["target_asset_status"] == "sealed"
    assert status["deployable_strategy"] is False


def test_v77_training_grid_and_outcome_blind_boundary_are_frozen() -> None:
    contract = _yaml("research/phase_contracts/v077.yaml")
    _validate_v77_persistent_duration_training_boundary(
        ROOT,
        _registered_v77_state(),
        _yaml("research/experiments/v077.yaml"),
        contract,
    )
    training = contract["grid_optimizer_and_runtime_contract"]
    assert training["folds"] == [1, 2, 3]
    assert training["seeds"] == [42, 7, 123]
    assert training["expected_jobs"] == 9
    assert training["prior_checkpoint_reuse"] == "none"
    assert contract["model_and_objective_contract"]["architecture"][
        "expected_parameter_count"
    ] == 1_083_155
    roles = contract["data_and_role_contract"]
    assert roles["any_2025_or_later_value_allowed"] is False
    assert roles["forbidden_role_columns"] == [
        "eligible_adaptive_development_evaluation"
    ]
    assert contract["stage_revision"] == (
        "v077_frozen_non_target_persistent_duration_training_r3"
    )
    assert contract["contract_repair"]["supersedes_revision"] == (
        "v077_frozen_non_target_persistent_duration_training_r2"
    )
    assert contract["contract_repair"][
        "scientific_architecture_objective_grid_roles_and_hyperparameters_changed"
    ] is False
    assert contract["runtime_contract"]["backup_policy"]["mode"] == "owner_waiver"
    assert "smoke_data_access.json" in contract["artifact_contract"][
        "required_files"
    ]
    safety = _yaml("research/current.yaml")["safety"]
    assert safety[
        "v77_r3_smoke_data_access_receipt_registered_before_data_access"
    ] is True
    assert safety["v77_r3_scientific_contract_changed"] is False


def test_v77_boundary_rejects_2025_role_or_hyperparameter_drift() -> None:
    state = _registered_v77_state()
    experiment = _yaml("research/experiments/v077.yaml")

    role_drift = deepcopy(_yaml("research/phase_contracts/v077.yaml"))
    role_drift["data_and_role_contract"]["any_2025_or_later_value_allowed"] = True
    with pytest.raises(ResearchStateError, match="training, target, or access"):
        _validate_v77_persistent_duration_training_boundary(
            ROOT, state, experiment, role_drift
        )

    parameter_drift = deepcopy(_yaml("research/phase_contracts/v077.yaml"))
    parameter_drift["model_and_objective_contract"]["architecture"]["d_model"] = 256
    with pytest.raises(ResearchStateError, match="training, target, or access"):
        _validate_v77_persistent_duration_training_boundary(
            ROOT, state, experiment, parameter_drift
        )


def test_v77_owner_storage_waiver_is_bound_for_live_doctor() -> None:
    receipt = _validate_backup_receipt(
        ROOT,
        "v77",
        _yaml("research/phase_contracts/v077.yaml"),
        "test-git-head",
    )
    assert receipt["passed"] is True
    assert receipt["mode"] == "owner_waiver"
    assert receipt["waiver_verified"] is True
    assert receipt["waiver_path"] == (
        "research/waivers/v077_external_backup_owner_waiver.json"
    )
