from __future__ import annotations

from pathlib import Path

import yaml

from tlm.research_workflow import (
    _validate_v63_training_boundary,
    _validate_v64_evaluation_boundary,
)


ROOT = Path(__file__).resolve().parents[1]


def test_v63_boundary_accepts_runtime_checkpoint_and_owner_waiver_contract() -> None:
    state = yaml.safe_load((ROOT / "research/current.yaml").read_text())
    state.update(
        {
            "authorized_phase": "v63",
            "authorized_next_action": "authorize_v63_frozen_non_target_decoupled_rank_state_training_only",
            "active_family_status": "dataset_passed_training_authorized",
            "last_completed_phase": "v62_non_target_decoupled_rank_state_dataset",
        }
    )
    experiment = yaml.safe_load((ROOT / "research/experiments/v062.yaml").read_text())
    contract = yaml.safe_load((ROOT / "research/phase_contracts/v063.yaml").read_text())
    _validate_v63_training_boundary(ROOT, state, experiment, contract)
    assert contract["runtime_contract"]["backup_policy"]["mode"] == "owner_waiver"
    assert contract["checkpoint_contract"]["cross_job_resume_allowed"] is False
    assert contract["artifact_contract"][
        "checkpoint_manifest_requires_all_nine_file_and_semantic_hashes"
    ] is True


def test_v64_boundary_is_adaptive_only_and_keeps_targets_sealed() -> None:
    state = yaml.safe_load((ROOT / "research/current.yaml").read_text())
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
    experiment = yaml.safe_load((ROOT / "research/experiments/v063.yaml").read_text())
    contract = yaml.safe_load((ROOT / "research/phase_contracts/v064.yaml").read_text())
    _validate_v64_evaluation_boundary(ROOT, state, experiment, contract)
    assert contract["evidence_tier"] == "adaptive_development_only_not_confirmation"
    assert contract["evaluation_contract"]["clean_holdout_claim"] is False
    assert contract["target_contract"]["status"] == "sealed"
