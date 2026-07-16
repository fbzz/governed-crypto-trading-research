from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from tlm.research_workflow import (
    ResearchStateError,
    _validate_v65_v64_r2_specification_boundary,
    _validate_v66_v64_r2_harness_boundary,
    _validate_v67_v64_r2_dataset_boundary,
    _validate_v68_v64_r2_gate_training_boundary,
    _validate_v69_v64_r2_prospective_prepare_boundary,
    _validate_v70_v64_r2_prospective_capture_boundary,
    _validate_v71_posthoc_retrospective_prepare_boundary,
    validate_research_state,
)


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    value = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _v65_staging_state() -> dict:
    state = deepcopy(_load("research/current.yaml"))
    state.update(
        {
            "authorized_phase": "v65",
            "authorized_next_action": (
                "execute_v65_metadata_only_v64_r2_probabilistic_state_gate_specification"
            ),
            "authorized_command": (
                "PYTHONPATH=src python3 -m tlm "
                "v64-r2-probabilistic-state-gate-spec "
                "--config configs/v65_v64_r2_probabilistic_state_gate_spec.yaml"
            ),
            "active_family_status": "specification_authorized_not_started",
            "last_completed_phase": "v64_adaptive_development_evaluation",
            "evidence_tier": "owner_authorized_metadata_only_design",
        }
    )
    return state


def _v66_staging_state() -> dict:
    state = deepcopy(_load("research/current.yaml"))
    state.update(
        {
            "authorized_phase": "v66",
            "authorized_next_action": (
                "authorize_v66_synthetic_v64_r2_probabilistic_state_gate_harness_only"
            ),
            "authorized_command": (
                "PYTHONPATH=src python3 -m tlm "
                "v64-r2-probabilistic-state-gate-harness "
                "--config configs/v66_v64_r2_probabilistic_state_gate_harness.yaml"
            ),
            "active_family_status": "specification_frozen_harness_authorized",
            "last_completed_phase": (
                "v65_v64_r2_probabilistic_state_gate_specification"
            ),
            "evidence_tier": "metadata_specification_passed_synthetic_only",
        }
    )
    return state


def _v67_staging_state() -> dict:
    state = deepcopy(_load("research/current.yaml"))
    state.update(
        {
            "authorized_phase": "v67",
            "authorized_next_action": (
                "authorize_v67_non_target_v64_r2_probabilistic_state_gate_dataset_only"
            ),
            "authorized_command": (
                "PYTHONPATH=src python3 -m tlm "
                "v64-r2-probabilistic-state-gate-dataset "
                "--config configs/v67_v64_r2_probabilistic_state_gate_dataset.yaml"
            ),
            "active_family_status": "synthetic_harness_passed_dataset_authorized",
            "last_completed_phase": (
                "v66_synthetic_v64_r2_probabilistic_state_gate_harness"
            ),
            "evidence_tier": "causal_non_target_gate_dataset_construction_only",
        }
    )
    return state


def _v68_staging_state() -> dict:
    state = deepcopy(_load("research/current.yaml"))
    state.update(
        {
            "authorized_phase": "v68",
            "authorized_next_action": (
                "authorize_v68_frozen_non_target_v64_r2_gate_training_only"
            ),
            "authorized_command": (
                "PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src python3 -m tlm "
                "v64-r2-probabilistic-state-gate-training "
                "--config configs/v68_v64_r2_probabilistic_state_gate_training.yaml"
            ),
            "active_family_status": "dataset_passed_training_authorized",
            "last_completed_phase": (
                "v67_non_target_v64_r2_probabilistic_state_gate_dataset"
            ),
            "evidence_tier": "causal_non_target_gate_training_only",
        }
    )
    return state


def _v69_staging_state() -> dict:
    state = deepcopy(_load("research/current.yaml"))
    state.update(
        {
            "authorized_phase": "v69",
            "authorized_next_action": (
                "authorize_v69_outcome_blind_non_target_prospective_confirmation_prepare_only"
            ),
            "authorized_command": (
                "PYTHONPATH=src python3 -m tlm "
                "v64-r2-prospective-confirmation-prepare "
                "--config configs/v69_v64_r2_prospective_confirmation_prepare.yaml"
            ),
            "active_family_status": "training_passed_prospective_prepare_authorized",
            "last_completed_phase": (
                "v68_frozen_non_target_v64_r2_probabilistic_state_gate_training"
            ),
            "evidence_tier": (
                "ex_ante_metadata_only_prospective_confirmation_preparation_authorized"
            ),
        }
    )
    return state


def _v70_staging_state() -> dict:
    state = deepcopy(_load("research/current.yaml"))
    state.update(
        {
            "authorized_phase": "v70",
            "authorized_next_action": (
                "authorize_v70_prospective_non_target_capture_and_prediction_freeze_only"
            ),
            "authorized_command": (
                "PYTHONPATH=src python3 -m tlm v64-r2-prospective-capture "
                "--config configs/v70_v64_r2_prospective_capture.yaml"
            ),
            "active_family_status": (
                "prospective_capture_prediction_freeze_authorized_not_started"
            ),
            "last_completed_phase": "v69_outcome_blind_prospective_confirmation_prepare",
            "evidence_tier": (
                "prospective_non_target_outcome_blind_prediction_accumulation_authorized"
            ),
        }
    )
    return state


def _v71_staging_state() -> dict:
    state = deepcopy(_load("research/current.yaml"))
    state.update(
        {
            "authorized_phase": "v71",
            "authorized_next_action": (
                "authorize_v71_posthoc_consumed_2025_diagnostic_prepare_only"
            ),
            "authorized_command": (
                "PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src python3 -m tlm "
                "v64-r2-retrospective-diagnostic-prepare "
                "--config configs/v71_v64_r2_retrospective_diagnostic.yaml"
            ),
            "active_family_status": (
                "posthoc_consumed_2025_diagnostic_prepare_authorized_not_started"
            ),
            "evidence_tier": (
                "posthoc_consumed_2025_diagnostic_prepare_only_not_confirmation"
            ),
        }
    )
    state["safety"]["v71_exact_hash_bound_unseal_authorization_present"] = False
    return state


def test_v66_completion_registers_only_v67_non_target_dataset() -> None:
    _validate_v67_v64_r2_dataset_boundary(
        ROOT,
        _v67_staging_state(),
        _load("research/experiments/v066.yaml"),
        _load("research/phase_contracts/v067.yaml"),
    )


def test_v67_completion_registers_only_v68_gate_training() -> None:
    _validate_v68_v64_r2_gate_training_boundary(
        ROOT,
        _v68_staging_state(),
        _load("research/experiments/v067.yaml"),
        _load("research/phase_contracts/v068.yaml"),
    )


def test_v68_completion_registers_only_v69_metadata_prepare() -> None:
    _validate_v69_v64_r2_prospective_prepare_boundary(
        ROOT,
        _v69_staging_state(),
        _load("research/experiments/v068.yaml"),
        _load("research/phase_contracts/v069.yaml"),
    )


def test_current_state_registers_only_v84_prepare() -> None:
    status = validate_research_state(ROOT, "research/current.yaml")
    assert status["passed"] is True
    assert status["authorized_phase"] == "v85"
    assert status["authorized_next_action"] == (
        "execute_v85_exactly_one_registered_non_target_outcome_unseal_and_complete_evaluation"
    )


def test_v70_contract_remains_reconstructable_after_owner_override() -> None:
    _validate_v70_v64_r2_prospective_capture_boundary(
        ROOT,
        _v70_staging_state(),
        _load("research/experiments/v069.yaml"),
        _load("research/phase_contracts/v070.yaml"),
    )


def test_v70_rejects_outcome_or_metric_access() -> None:
    contract = deepcopy(_load("research/phase_contracts/v070.yaml"))
    contract["artifact_contract"]["no_outcome_packet_during_v70"] = False
    with pytest.raises(ResearchStateError, match="outcome, target, or deployment"):
        _validate_v70_v64_r2_prospective_capture_boundary(
            ROOT,
            _v70_staging_state(),
            _load("research/experiments/v069.yaml"),
            contract,
        )


def test_v71_rejects_outcome_access_during_prepare() -> None:
    contract = deepcopy(_load("research/phase_contracts/v071.yaml"))
    contract["sealed_outcome_contract"]["may_be_opened_during_v71_prepare"] = True
    with pytest.raises(ResearchStateError, match="sealed outcome boundary"):
        _validate_v71_posthoc_retrospective_prepare_boundary(
            ROOT,
            _v71_staging_state(),
            _load("research/experiments/v071.yaml"),
            contract,
        )


def test_v66_staging_boundary_accepts_registered_contract() -> None:
    _validate_v66_v64_r2_harness_boundary(
        ROOT,
        _v66_staging_state(),
        _load("research/experiments/v065.yaml"),
        _load("research/phase_contracts/v066.yaml"),
    )


def test_v67_rejects_2025_role_access() -> None:
    contract = deepcopy(_load("research/phase_contracts/v067.yaml"))
    contract["role_contract"]["consumed_v64_2025_role_created"] = True
    with pytest.raises(ResearchStateError, match="chronology"):
        _validate_v67_v64_r2_dataset_boundary(
            ROOT,
            _v67_staging_state(),
            _load("research/experiments/v066.yaml"),
            contract,
        )


def test_v68_rejects_old_gate_state_reuse() -> None:
    contract = deepcopy(_load("research/phase_contracts/v068.yaml"))
    contract["ranker_and_scaler_reuse_contract"]["old_gate_substate_loaded_into_model"] = True
    with pytest.raises(ResearchStateError, match="model/grid/role"):
        _validate_v68_v64_r2_gate_training_boundary(
            ROOT,
            _v68_staging_state(),
            _load("research/experiments/v067.yaml"),
            contract,
        )


def test_v68_rejects_any_2025_value_access() -> None:
    contract = deepcopy(_load("research/phase_contracts/v068.yaml"))
    contract["data_and_role_contract"]["any_2025_or_later_value_allowed"] = True
    with pytest.raises(ResearchStateError, match="model/grid/role"):
        _validate_v68_v64_r2_gate_training_boundary(
            ROOT,
            _v68_staging_state(),
            _load("research/experiments/v067.yaml"),
            contract,
        )


def test_v65_staging_boundary_is_hash_bound() -> None:
    _validate_v65_v64_r2_specification_boundary(
        ROOT,
        _v65_staging_state(),
        _load("research/experiments/v065_authorized.yaml"),
        _load("research/phase_contracts/v065.yaml"),
    )


def test_v65_rejects_checkpoint_or_parquet_allowlist_drift() -> None:
    contract = deepcopy(_load("research/phase_contracts/v065.yaml"))
    contract["access_contract"]["allowed_inputs"].append(
        "data/checkpoints/v63_decoupled_rank_state_training/fold_1/seed_42.final.pt"
    )
    with pytest.raises(ResearchStateError, match="allowlist"):
        _validate_v65_v64_r2_specification_boundary(
            ROOT,
            _v65_staging_state(),
            _load("research/experiments/v065_authorized.yaml"),
            contract,
        )


def test_v65_rejects_ranker_unfreeze() -> None:
    config_path = ROOT / "configs/v65_v64_r2_probabilistic_state_gate_spec.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    spec = config["v64_r2_probabilistic_state_gate_spec"]
    assert spec["ranker_contract"]["status"] == "frozen_exactly_from_v64"
    assert spec["ranker_contract"]["weights"][
        "checkpoint_deserialization_during_v65"
    ] is False
