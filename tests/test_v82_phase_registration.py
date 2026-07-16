from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from tlm.research_workflow import (
    ResearchStateError,
    _validate_v82_low_turnover_rank_dataset_boundary,
    validate_research_state,
)


ROOT = Path(__file__).resolve().parents[1]


def _yaml(relative: str) -> dict:
    return yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))


def _v82_staging_state() -> dict:
    state = deepcopy(_yaml("research/current.yaml"))
    state.update({
        "authorized_phase": "v82",
        "authorized_next_action": "authorize_v82_non_target_low_turnover_rank_dataset_only",
        "authorized_command": (
            "PYTHONPATH=src python3 -m tlm low-turnover-rank-dataset "
            "--config configs/v82_low_turnover_rank_dataset.yaml"
        ),
        "active_family_status": "chronology_erratum_passed_dataset_authorized",
        "last_completed_phase": "v82_r0_metadata_only_chronology_erratum",
        "last_completed_result": "artifacts/v82_r0_low_turnover_rank_chronology_erratum/result.json",
        "evidence_tier": "causal_non_target_dataset_and_sealed_evaluation_packet_only",
    })
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


def test_v82_contract_is_bound_to_exact_user_and_v82_r0_result_hashes() -> None:
    contract = _yaml("research/phase_contracts/v082.yaml")
    assert contract["authorization_receipt"]["registered_result_sha256"] == (
        "56ef31fcb37a5566e3b0badbf1e2681862e9bc86a1affa898950c823dc490b96"
    )
    assert contract["explicit_user_authorization"][
        "registered_authorization_sha256"
    ] == "91155a7dc02bba8958fd74fff5d684802db54c76c592d371280eb9ad1a5acbdf"
    assert contract["evaluation_contract"]["signal_dates"] == 159
    assert contract["evaluation_contract"]["final_outcome_maturity"] == (
        "2026-06-30"
    )


def test_v82_boundary_rejects_target_asset_or_unseal_scope() -> None:
    state = _v82_staging_state()
    experiment = _yaml("research/experiments/v082.yaml")
    contract = deepcopy(_yaml("research/phase_contracts/v082.yaml"))
    contract["source_contract"]["symbols"][0] = "BTCUSDT"
    with pytest.raises(ResearchStateError, match="ancestry drift"):
        _validate_v82_low_turnover_rank_dataset_boundary(
            ROOT, state, experiment, contract
        )

    contract = deepcopy(_yaml("research/phase_contracts/v082.yaml"))
    contract["evaluation_contract"]["outcome_packet_unseals_during_v82"] = 1
    with pytest.raises(ResearchStateError, match="source, data, output"):
        _validate_v82_low_turnover_rank_dataset_boundary(
            ROOT, state, experiment, contract
        )
