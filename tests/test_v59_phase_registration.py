from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from tlm.core.artifacts import canonical_sha256
from tlm.research_workflow import (
    ResearchStateError,
    _validate_v59_terminal_boundary,
    _validate_v59_unseal_boundary,
    validate_research_state,
)


ROOT = Path(__file__).resolve().parents[1]
CURRENT_PATH = ROOT / "research/current.yaml"
EXPERIMENT_PATH = ROOT / "research/experiments/v059_prepare.yaml"
FINAL_EXPERIMENT_PATH = ROOT / "research/experiments/v059.yaml"
BASE_PATH = ROOT / "research/phase_contracts/v059.yaml"
STAGE_PATH = ROOT / "research/phase_contracts/v059_unseal_r1.yaml"
TERMINAL_PATH = ROOT / "research/phase_contracts/v059_terminal_r1.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _validate(current: dict, stage: dict) -> None:
    _validate_v59_unseal_boundary(
        ROOT, current, _load_yaml(EXPERIMENT_PATH), stage
    )


def _terminal_current() -> dict:
    return {
        "schema_version": 1,
        "project": "TLM",
        "as_of_date": "2026-07-14",
        "current_experiment": "research/experiments/v059.yaml",
        "phase_contract": {
            "path": "research/phase_contracts/v059_terminal_r1.yaml",
            "file_sha256": _sha256(TERMINAL_PATH),
        },
        "active_family_id": "tlm_state_conditioned_multi_horizon_quantile_small_v1",
        "active_family_status": "retired",
        "last_completed_phase": "v59_state_conditioned_multi_horizon_evaluation",
        "last_completed_result": (
            "artifacts/v59_state_conditioned_multi_horizon_evaluation/result.json"
        ),
        "authorized_next_action": "retire_family_without_tuning",
        "authorized_phase": "v59",
        "target_assets": {
            "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            "status": "sealed",
        },
        "deployable_strategy": False,
        "evidence_tier": "adaptive_historical_non_target_evaluation_negative",
        "families": [
            {
                "family_id": "tlm_multi_asset_target_transfer_v2",
                "trained": True,
                "status": "retired",
                "terminal_phase": "v37",
            },
            {
                "family_id": "tlm_cross_sectional_rank_excess_medium_v1",
                "trained": True,
                "status": "retired",
                "terminal_phase": "v45",
            },
            {
                "family_id": "tlm_joint_absolute_relative_triplet_medium_v1",
                "trained": True,
                "status": "retired",
                "terminal_phase": "v50",
            },
            {
                "family_id": "tlm_state_conditioned_multi_horizon_quantile_small_v1",
                "trained": True,
                "status": "retired",
                "terminal_phase": "v59",
            },
        ],
        "forbidden_capabilities": [
            "access_target_assets",
            "second_development_outcome_unseal_or_source_reread",
            "refit_candidate_model_or_scaler",
            "train_or_modify_any_checkpoint",
            "select_discard_or_weight_checkpoints",
            "regenerate_predictions_or_positions",
            "change_v59_costs_metrics_bootstrap_gates_or_decision",
            "implement_v60_or_register_new_family_without_new_explicit_authorization",
            "paper_shadow_live_or_real_money_trading",
        ],
        "safety": {
            "target_assets_remain_sealed": True,
            "terminal_retirement_decision_is_immutable": True,
            "completed_development_outcome_unseals": 1,
            "maximum_development_outcome_unseals": 1,
            "additional_source_outcome_reads_forbidden": True,
            "replay_source_outcome_reads": 0,
            "retuning_or_regeneration_forbidden": True,
        },
    }


def _unseal_current() -> dict:
    current = deepcopy(_terminal_current())
    current.update(
        {
            "current_experiment": "research/experiments/v059_prepare.yaml",
            "phase_contract": {
                "path": "research/phase_contracts/v059_unseal_r1.yaml",
                "file_sha256": _sha256(STAGE_PATH),
            },
            "active_family_status": (
                "development_evaluation_prepared_outcomes_sealed_"
                "exactly_one_unseal_authorized"
            ),
            "authorized_phase": "v59",
            "last_completed_phase": "v59_outcome_blind_prepare",
            "last_completed_result": (
                "artifacts/v59_state_conditioned_multi_horizon_evaluation/"
                "prepare_receipt.json"
            ),
            "authorized_next_action": (
                "execute_v59_exactly_one_registered_outcome_unseal_and_"
                "complete_evaluation"
            ),
            "authorized_command": (
                "PYTHONPATH=src python3 -m tlm "
                "state-conditioned-multi-horizon-evaluation-unseal "
                "--config configs/v59_state_conditioned_multi_horizon_evaluation.yaml"
            ),
            "evidence_tier": (
                "adaptive_historical_development_evaluation_prepared_outcomes_sealed"
            ),
            "forbidden_capabilities": [
                "access_target_assets",
                "second_development_outcome_unseal_or_source_reread",
                "refit_candidate_model_or_scaler",
                "train_or_modify_any_checkpoint",
                "select_discard_or_weight_checkpoints",
                "regenerate_predictions_or_positions",
                "implement_v60_or_later",
                "paper_shadow_live_or_real_money_trading",
            ],
        }
    )
    current["families"][-1].update(
        {
            "status": current["active_family_status"],
            "terminal_phase": None,
        }
    )
    return current


def test_v59_terminal_registration_is_hash_bound_and_retired() -> None:
    status = validate_research_state(ROOT)
    current = _terminal_current()
    experiment = _load_yaml(FINAL_EXPERIMENT_PATH)
    terminal = _load_yaml(TERMINAL_PATH)

    assert status["passed"] is True
    assert int(status["authorized_phase"].removeprefix("v").split("-", 1)[0]) >= 64
    assert status["trained_family_count"] == 8
    assert status["target_asset_status"] == "sealed"
    assert current["phase_contract"] == {
        "path": "research/phase_contracts/v059_terminal_r1.yaml",
        "file_sha256": _sha256(TERMINAL_PATH),
    }
    assert terminal["parent_experiment"] == {
        "path": "research/experiments/v059.yaml",
        "file_sha256": _sha256(FINAL_EXPERIMENT_PATH),
    }
    assert experiment["result"]["file_sha256"] == _sha256(
        ROOT / experiment["result"]["path"]
    )
    _validate_v59_terminal_boundary(ROOT, current, experiment, terminal)


def test_v59_user_authorization_binds_exact_prepare_hashes() -> None:
    stage = _load_yaml(STAGE_PATH)
    explicit = stage["explicit_user_authorization"]
    payload = explicit["payload"]
    assert canonical_sha256(payload) == explicit["canonical_sha256"]
    assert payload["evaluation_spec_sha256"] == (
        "31bd555f333e0b3c039f39e09b269b30446c0b372df55946e8bf473283308291"
    )
    assert payload["prepare_manifest_sha256"] == (
        "27e171d3d540f0edc0142ad06c3e892aa73d3c8fb39a1020727ca16c17dd1eb5"
    )
    assert payload["prepare_receipt_sha256"] == (
        "be930d7a5166bbba597748c6e8a4a7144ef66d3fffb213d097d77bf64e39e30e"
    )
    assert payload["maximum_unseal_count"] == 1
    assert payload["no_retuning_or_regeneration"] is True
    assert payload["target_assets"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def test_v59_stage_preserves_frozen_science_by_base_contract_hash() -> None:
    stage = _load_yaml(STAGE_PATH)
    base = _load_yaml(BASE_PATH)
    assert stage["base_phase_contract"] == {
        "path": "research/phase_contracts/v059.yaml",
        "file_sha256": _sha256(BASE_PATH),
        "canonical_sha256": canonical_sha256(base),
    }
    assert base["evaluation_cells"]["checkpoint_count"] == 36
    assert base["evaluation_cells"]["selected_jobs"] == []
    assert base["accounting_contract"]["mandatory_cost_bps"] == [10, 20, 30]
    assert base["bootstrap_contract"]["paths"] == 10000
    assert base["bootstrap_contract"]["block_lengths"] == [7, 21, 63]
    assert base["gate_contract"]["aggregate_rescue_allowed"] is False
    assert base["target_contract"]["status"] == "sealed"


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("explicit_user_authorization", "canonical_sha256"), "0" * 64, "authorization"),
        (("outcome_access_contract", "maximum_source_reads"), 2, "outcome access"),
        (("target_contract", "status"), "open", "target seal"),
        (("commands", "unseal"), "echo unsafe", "unseal command"),
    ],
)
def test_v59_unseal_boundary_rejects_scope_drift(
    path: tuple[str, str], value: object, message: str
) -> None:
    current = _unseal_current()
    stage = deepcopy(_load_yaml(STAGE_PATH))
    stage[path[0]][path[1]] = value
    if path == ("commands", "unseal"):
        stage["authorized_command"] = value
        current["authorized_command"] = value
    with pytest.raises(ResearchStateError, match=message):
        _validate(current, stage)


def test_v59_unseal_boundary_rejects_current_state_drift() -> None:
    current = _unseal_current()
    current["deployable_strategy"] = True
    with pytest.raises(ResearchStateError, match="cannot mark"):
        _validate(current, _load_yaml(STAGE_PATH))


def test_v59_prepare_packet_self_hashes_remain_bound() -> None:
    stage = _load_yaml(STAGE_PATH)
    fields = {
        "evaluation_spec": "evaluation_spec_sha256",
        "prepare_manifest": "prepare_manifest_sha256",
        "prepare_receipt": "prepare_receipt_sha256",
        "outcome_request": "outcome_request_sha256",
    }
    for name, field in fields.items():
        reference = stage["prepare_packet"][name]
        path = ROOT / reference["path"]
        assert _sha256(path) == reference["file_sha256"]
        value = json.loads(path.read_text(encoding="utf-8"))
        registered = value.pop(field)
        assert registered == reference["canonical_sha256"] == canonical_sha256(value)
