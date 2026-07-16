from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from tlm.research_workflow import (
    ResearchStateError,
    _validate_v84_low_turnover_rank_evaluation_prepare_boundary,
)


ROOT = Path(__file__).resolve().parents[1]


def _yaml(relative: str) -> dict:
    value = yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _v84_staging_state() -> dict:
    state = deepcopy(_yaml("research/current.yaml"))
    state.update({
        "authorized_phase": "v84",
        "authorized_next_action": (
            "authorize_v84_outcome_blind_low_turnover_rank_evaluation_prepare_only"
        ),
        "authorized_command": (
            "PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src python3 -m tlm "
            "low-turnover-rank-evaluation-prepare "
            "--config configs/v84_low_turnover_rank_evaluation.yaml"
        ),
        "active_family_status": "trained_outcome_blind_evaluation_prepare_authorized",
        "last_completed_phase": "v83_frozen_non_target_low_turnover_rank_training",
        "last_completed_result": "artifacts/v83_low_turnover_rank_training/result.json",
        "evidence_tier": (
            "retrospective_non_target_first_use_2026_prepare_outcomes_sealed"
        ),
    })
    return state


def test_v84_freezes_exact_checkpoint_policy_and_one_shot_boundary() -> None:
    state = _v84_staging_state()
    experiment = _yaml("research/experiments/v084.yaml")
    contract = _yaml("research/phase_contracts/v084.yaml")
    _validate_v84_low_turnover_rank_evaluation_prepare_boundary(
        ROOT, state, experiment, contract
    )
    allowed = set(contract["access_contract"]["allowed_inputs"])
    assert "data/processed/low_turnover_rank_evaluation_features_v82.parquet" in allowed
    assert "data/processed/low_turnover_rank_evaluation_outcomes_v82.parquet" not in allowed
    assert contract["evaluation_contract"]["window"]["signal_dates"] == 159
    assert contract["evaluation_contract"]["seeds"] == [42, 7, 123]
    assert contract["policy_contract"]["structural_maximum_turnover"] == 16.0
    assert len(contract["outcome_blind_gate_contract"]["gates"]) == 12
    assert contract["one_shot_contract"]["unseal"][
        "generic_continue_is_not_authorization"
    ] is True
    assert contract["target_contract"]["status"] == "sealed"


def test_v84_rejects_outcome_allowlist_or_checkpoint_selection_drift() -> None:
    state = _v84_staging_state()
    experiment = _yaml("research/experiments/v084.yaml")

    outcome_drift = deepcopy(_yaml("research/phase_contracts/v084.yaml"))
    path = "data/processed/low_turnover_rank_evaluation_outcomes_v82.parquet"
    outcome_drift["access_contract"]["allowed_inputs"].append(path)
    outcome_drift["input_contract"]["expected_file_sha256_by_path"][path] = "0" * 64
    with pytest.raises(ResearchStateError, match="allowlist"):
        _validate_v84_low_turnover_rank_evaluation_prepare_boundary(
            ROOT, state, experiment, outcome_drift
        )

    seed_drift = deepcopy(_yaml("research/phase_contracts/v084.yaml"))
    seed_drift["evaluation_contract"]["seeds"] = [42]
    with pytest.raises(ResearchStateError, match="evaluation"):
        _validate_v84_low_turnover_rank_evaluation_prepare_boundary(
            ROOT, state, experiment, seed_drift
        )
