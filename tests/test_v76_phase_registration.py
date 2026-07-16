from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from tlm.research_workflow import (
    ResearchStateError,
    _validate_v76_persistent_duration_dataset_boundary,
    validate_research_state,
)


ROOT = Path(__file__).resolve().parents[1]


def _yaml(relative: str) -> dict:
    return yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))


def _v76_staging_state() -> dict:
    state = deepcopy(_yaml("research/current.yaml"))
    state.update(
        {
            "authorized_phase": "v76",
            "authorized_next_action": (
                "authorize_v76_non_target_persistent_duration_dataset_only"
            ),
            "authorized_command": (
                "PYTHONPATH=src python3 -m tlm persistent-duration-dataset "
                "--config configs/v76_persistent_duration_dataset.yaml"
            ),
            "active_family_status": "dataset_authorized_not_started",
            "last_completed_phase": "v75_synthetic_persistent_duration_harness",
            "evidence_tier": "causal_non_target_dataset_construction_only",
        }
    )
    state["safety"]["synthetic_only_phase"] = False
    state["safety"]["dataset_only_phase"] = True
    state["safety"]["training_only_phase"] = False
    return state


def test_current_state_registers_only_v84_prepare() -> None:
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


def test_v76_contract_records_malformed_v74_hash_before_data_access() -> None:
    state = _v76_staging_state()
    experiment = _yaml("research/experiments/v076.yaml")
    contract = _yaml("research/phase_contracts/v076.yaml")
    _validate_v76_persistent_duration_dataset_boundary(
        ROOT, state, experiment, contract
    )
    correction = contract["source_receipt_correction"]
    assert correction["malformed_v74_length"] == 61
    assert correction["authoritative_v32_length"] == 64
    assert correction["scientific_semantics_changed"] is False
    assert correction["source_rows_changed"] is False
    assert correction["source_values_changed"] is False
    assert correction["panel_or_sequence_deserializations_during_registration"] == 0
    assert correction["post_v75_gate_hash_only_reads"] == 2


def test_v76_boundary_rejects_checkpoint_or_target_access() -> None:
    state = _v76_staging_state()
    experiment = _yaml("research/experiments/v076.yaml")
    contract = deepcopy(_yaml("research/phase_contracts/v076.yaml"))
    contract["access_contract"]["allowed_inputs"].append(
        "data/checkpoints/v75_real_checkpoint.pt"
    )
    with pytest.raises(ResearchStateError, match="input contract drift"):
        _validate_v76_persistent_duration_dataset_boundary(
            ROOT, state, experiment, contract
        )
