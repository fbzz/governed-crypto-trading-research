from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest
import yaml

from tlm.research_workflow import (
    ResearchStateError,
    _validate_v60_specification_boundary,
    _validate_v61_harness_boundary,
    validate_research_state,
)


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    value = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _v60_staging_state() -> dict:
    state = deepcopy(_load("research/current.yaml"))
    state.update(
        {
            "current_experiment": "research/experiments/v060_authorized.yaml",
            "phase_contract": {
                "path": "research/phase_contracts/v060.yaml",
                "file_sha256": _sha256("research/phase_contracts/v060.yaml"),
            },
            "active_family_status": "specification_authorized_not_started",
            "last_completed_phase": "v59_state_conditioned_multi_horizon_evaluation",
            "last_completed_result": (
                "artifacts/v59_state_conditioned_multi_horizon_evaluation/result.json"
            ),
            "authorized_next_action": (
                "execute_v60_metadata_only_decoupled_rank_state_family_specification"
            ),
            "authorized_phase": "v60",
            "authorized_command": (
                "PYTHONPATH=src python3 -m tlm decoupled-rank-state-spec "
                "--config configs/v60_decoupled_rank_state_spec.yaml"
            ),
            "evidence_tier": "owner_authorized_metadata_only_design",
        }
    )
    state["families"][-1]["status"] = "specification_authorized_not_started"
    return state


def _v61_staging_state() -> dict:
    state = deepcopy(_load("research/current.yaml"))
    state.update(
        {
            "current_experiment": "research/experiments/v060.yaml",
            "phase_contract": {
                "path": "research/phase_contracts/v061.yaml",
                "file_sha256": _sha256("research/phase_contracts/v061.yaml"),
            },
            "active_family_status": "specification_frozen_harness_authorized",
            "last_completed_phase": "v60_decoupled_rank_state_specification",
            "last_completed_result": (
                "artifacts/v60_decoupled_rank_state_spec/result.json"
            ),
            "authorized_next_action": (
                "authorize_v61_synthetic_decoupled_rank_state_harness_only"
            ),
            "authorized_phase": "v61",
            "authorized_command": (
                "PYTHONPATH=src python3 -m tlm decoupled-rank-state-harness "
                "--config configs/v61_decoupled_rank_state_harness.yaml"
            ),
            "evidence_tier": "metadata_specification_passed_synthetic_only",
        }
    )
    state["families"][-1]["status"] = "specification_frozen_harness_authorized"
    return state


def test_v60_completion_registers_only_v61_synthetic_harness() -> None:
    experiment = _load("research/experiments/v060.yaml")
    contract = _load("research/phase_contracts/v061.yaml")
    assert experiment["authorized_next_action"] == (
        "authorize_v61_synthetic_decoupled_rank_state_harness_only"
    )
    assert contract["authorized_next_action"] == experiment["authorized_next_action"]
    assert contract["parent_experiment"] == {
        "path": "research/experiments/v060.yaml",
        "file_sha256": _sha256("research/experiments/v060.yaml"),
    }
    _validate_v61_harness_boundary(
        ROOT,
        _v61_staging_state(),
        experiment,
        contract,
    )


def test_v60_staging_boundary_is_hash_bound() -> None:
    _validate_v60_specification_boundary(
        ROOT,
        _v60_staging_state(),
        _load("research/experiments/v060_authorized.yaml"),
        _load("research/phase_contracts/v060.yaml"),
    )


def test_v60_staging_rejects_input_allowlist_drift() -> None:
    contract = deepcopy(_load("research/phase_contracts/v060.yaml"))
    contract["access_contract"]["allowed_inputs"].append("data/processed/unsafe.parquet")
    with pytest.raises(ResearchStateError, match="allowlist"):
        _validate_v60_specification_boundary(
            ROOT,
            _v60_staging_state(),
            _load("research/experiments/v060_authorized.yaml"),
            contract,
        )


def test_v61_rejects_target_unseal() -> None:
    contract = deepcopy(_load("research/phase_contracts/v061.yaml"))
    contract["target_contract"]["status"] = "open"
    with pytest.raises(ResearchStateError, match="target/deployment"):
        _validate_v61_harness_boundary(
            ROOT,
            _v61_staging_state(),
            _load("research/experiments/v060.yaml"),
            contract,
        )
