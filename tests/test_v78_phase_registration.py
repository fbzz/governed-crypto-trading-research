from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from tlm.research_workflow import (
    ResearchStateError,
    _validate_v78_persistent_duration_evaluation_prepare_boundary,
    validate_research_state,
)


ROOT = Path(__file__).resolve().parents[1]


def _yaml(relative: str) -> dict:
    value = yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _v78_staging_state() -> dict:
    state = _yaml("research/current.yaml")
    state.update(
        {
            "active_family_id": "tlm_persistent_multi_horizon_duration_v1",
            "active_family_status": (
                "trained_outcome_blind_evaluation_prepare_authorized"
            ),
            "last_completed_phase": (
                "v77_frozen_non_target_persistent_duration_training"
            ),
            "last_completed_result": (
                "artifacts/v77_persistent_duration_training/result.json"
            ),
            "authorized_next_action": (
                "authorize_v78_outcome_blind_persistent_duration_evaluation_prepare_only"
            ),
            "authorized_phase": "v78",
            "authorized_command": (
                "PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src python3 -m tlm "
                "persistent-duration-evaluation-prepare "
                "--config configs/v78_persistent_duration_evaluation.yaml"
            ),
            "evidence_tier": (
                "adaptive_consumed_2025_non_target_development_prepare_outcomes_sealed"
            ),
        }
    )
    return state


def test_current_state_authorizes_only_v84_prepare() -> None:
    status = validate_research_state(ROOT, "research/current.yaml")
    assert status["passed"] is True
    assert status["trained_family_count"] == 8
    assert status["family_count"] == 8
    assert status["authorized_phase"] == "v85"
    assert status["authorized_next_action"] == (
        "execute_v85_exactly_one_registered_non_target_outcome_unseal_and_complete_evaluation"
    )
    assert status["target_asset_status"] == "sealed"
    assert status["deployable_strategy"] is False


def test_v78_freezes_exact_checkpoint_policy_and_one_shot_boundary() -> None:
    state = _v78_staging_state()
    experiment = _yaml("research/experiments/v078.yaml")
    contract = _yaml("research/phase_contracts/v078.yaml")
    _validate_v78_persistent_duration_evaluation_prepare_boundary(
        ROOT, state, experiment, contract
    )
    evaluation = contract["evaluation_contract"]
    assert evaluation["folds"] == [1, 2, 3]
    assert evaluation["seeds"] == [42, 7, 123]
    assert evaluation["inference"]["checkpoint_state"] == (
        "model_best_state_at_registered_early_stopping_best_epoch"
    )
    assert evaluation["triplet_scope"].startswith("exact_120_lexical")
    assert evaluation["fold_triplet_calendar"] == (
        "every_triplet_has_all_357_registered_signal_dates"
    )
    assert contract["policy_contract"]["reporting_cost_bps"] == [10, 20, 30]
    assert len(contract["outcome_blind_gate_contract"]["gates"]) == 12
    assert contract["one_shot_contract"]["current_stage"] == (
        "outcome_blind_prepare"
    )
    assert contract["one_shot_contract"]["unseal"][
        "generic_continue_is_not_authorization"
    ] is True
    assert contract["one_shot_contract"]["unseal"]["maximum_unseal_count"] == 1
    assert contract["target_contract"]["status"] == "sealed"


def test_v78_rejects_outcome_input_or_checkpoint_selection_drift() -> None:
    state = _v78_staging_state()
    experiment = _yaml("research/experiments/v078.yaml")

    outcome_drift = deepcopy(_yaml("research/phase_contracts/v078.yaml"))
    outcome_drift["input_contract"]["expected_file_sha256_by_path"][
        "data/processed/persistent_duration_labels_v76.parquet"
    ] = "0" * 64
    outcome_drift["access_contract"]["allowed_inputs"].append(
        "data/processed/persistent_duration_labels_v76.parquet"
    )
    with pytest.raises(ResearchStateError, match="allowlist|outcome"):
        _validate_v78_persistent_duration_evaluation_prepare_boundary(
            ROOT, state, experiment, outcome_drift
        )

    seed_drift = deepcopy(_yaml("research/phase_contracts/v078.yaml"))
    seed_drift["evaluation_contract"]["seeds"] = [42]
    with pytest.raises(ResearchStateError, match="evaluation"):
        _validate_v78_persistent_duration_evaluation_prepare_boundary(
            ROOT, state, experiment, seed_drift
        )
