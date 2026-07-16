from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from tlm.research_workflow import (
    ResearchStateError,
    _validate_v62_dataset_boundary,
)


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    value = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _v62_staging_state() -> dict:
    state = deepcopy(_load("research/current.yaml"))
    state.update(
        {
            "current_experiment": "research/experiments/v061.yaml",
            "active_family_status": "synthetic_harness_passed_dataset_authorized",
            "last_completed_phase": "v61_decoupled_rank_state_harness",
            "last_completed_result": (
                "artifacts/v61_decoupled_rank_state_harness/result.json"
            ),
            "authorized_next_action": (
                "authorize_v62_non_target_decoupled_rank_state_dataset_only"
            ),
            "authorized_phase": "v62",
            "authorized_command": (
                "PYTHONPATH=src python3 -m tlm decoupled-rank-state-dataset "
                "--config configs/v62_decoupled_rank_state_dataset.yaml"
            ),
            "evidence_tier": (
                "synthetic_harness_passed_non_target_dataset_only_authorized"
            ),
        }
    )
    state["families"][-1]["status"] = (
        "synthetic_harness_passed_dataset_authorized"
    )
    return state


def test_v61_completion_registers_only_v62_non_target_dataset() -> None:
    experiment = _load("research/experiments/v061.yaml")
    contract = _load("research/phase_contracts/v062.yaml")
    assert experiment["authorized_next_action"] == (
        "authorize_v62_non_target_decoupled_rank_state_dataset_only"
    )
    assert contract["authorized_next_action"] == experiment["authorized_next_action"]
    _validate_v62_dataset_boundary(ROOT, _v62_staging_state(), experiment, contract)


def test_v62_dataset_boundary_accepts_registered_contract() -> None:
    _validate_v62_dataset_boundary(
        ROOT,
        _v62_staging_state(),
        _load("research/experiments/v061.yaml"),
        _load("research/phase_contracts/v062.yaml"),
    )


def test_v62_dataset_boundary_rejects_input_drift() -> None:
    contract = deepcopy(_load("research/phase_contracts/v062.yaml"))
    contract["input_contract"]["expected_file_sha256_by_path"][
        "data/processed/selected_universe_panel_v32.parquet"
    ] = "0" * 64
    with pytest.raises(ResearchStateError, match="input drift"):
        _validate_v62_dataset_boundary(
            ROOT,
            _v62_staging_state(),
            _load("research/experiments/v061.yaml"),
            contract,
        )


def test_v62_dataset_boundary_rejects_target_unseal() -> None:
    contract = deepcopy(_load("research/phase_contracts/v062.yaml"))
    contract["target_contract"]["status"] = "open"
    with pytest.raises(ResearchStateError, match="target/deployment"):
        _validate_v62_dataset_boundary(
            ROOT,
            _v62_staging_state(),
            _load("research/experiments/v061.yaml"),
            contract,
        )


def test_v62_dataset_boundary_requires_training_to_remain_forbidden() -> None:
    contract = deepcopy(_load("research/phase_contracts/v062.yaml"))
    contract["access_contract"]["forbidden_capabilities"].remove(
        "optimizer_step_or_training"
    )
    with pytest.raises(ResearchStateError, match="forbidden boundary"):
        _validate_v62_dataset_boundary(
            ROOT,
            _v62_staging_state(),
            _load("research/experiments/v061.yaml"),
            contract,
        )
