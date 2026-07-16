from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from tlm.research_workflow import (
    ResearchStateError,
    _validate_v81_low_turnover_rank_harness_boundary,
    validate_research_state,
)


ROOT = Path(__file__).resolve().parents[1]


def _yaml(relative: str) -> dict:
    value = yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _v81_staging_state() -> dict:
    state = deepcopy(_yaml("research/current.yaml"))
    state.update(
        {
            "authorized_phase": "v81",
            "authorized_next_action": (
                "authorize_v81_synthetic_low_turnover_rank_harness_only"
            ),
            "authorized_command": (
                "PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src python3 -m tlm "
                "low-turnover-rank-harness "
                "--config configs/v81_low_turnover_rank_harness.yaml"
            ),
            "active_family_status": (
                "specification_frozen_synthetic_harness_authorized"
            ),
            "last_completed_phase": (
                "v80_low_turnover_cross_sectional_rank_specification"
            ),
            "last_completed_result": (
                "artifacts/v80_low_turnover_rank_spec/result.json"
            ),
            "evidence_tier": "deterministic_synthetic_harness_only",
        }
    )
    return state


def test_current_state_authorizes_only_v84_prepare() -> None:
    status = validate_research_state(ROOT, "research/current.yaml")
    assert status["passed"] is True
    assert status["family_count"] == 8
    assert status["trained_family_count"] == 8
    assert status["retired_family_count"] == 6
    assert status["active_family_id"] == "tlm_low_turnover_cross_sectional_rank_v1"
    assert status["authorized_phase"] == "v85"
    assert status["authorized_next_action"] == (
        "execute_v85_exactly_one_registered_non_target_outcome_unseal_and_complete_evaluation"
    )
    assert status["target_asset_status"] == "sealed"
    assert status["deployable_strategy"] is False


def test_v81_boundary_is_exact_and_synthetic_only() -> None:
    state = _v81_staging_state()
    experiment = _yaml("research/experiments/v081.yaml")
    contract = _yaml("research/phase_contracts/v081.yaml")
    _validate_v81_low_turnover_rank_harness_boundary(
        ROOT, state, experiment, contract
    )
    synthetic = contract["synthetic_contract"]
    assert synthetic["input_shape"] == [4, 128, 3, 8]
    assert synthetic["parameter_count"] == 10993
    assert synthetic["structural_maximum_turnover"] == 16.0
    assert contract["target_contract"]["status"] == "sealed"


def test_v81_boundary_rejects_real_data_or_architecture_drift() -> None:
    state = _v81_staging_state()
    experiment = _yaml("research/experiments/v081.yaml")

    data_drift = deepcopy(_yaml("research/phase_contracts/v081.yaml"))
    data_drift["access_contract"]["allowed_inputs"].append(
        "data/processed/selected_universe_panel_v32.parquet"
    )
    with pytest.raises(ResearchStateError, match="metadata input contract"):
        _validate_v81_low_turnover_rank_harness_boundary(
            ROOT, state, experiment, data_drift
        )

    architecture_drift = deepcopy(_yaml("research/phase_contracts/v081.yaml"))
    architecture_drift["synthetic_contract"]["parameter_count"] = 10994
    with pytest.raises(ResearchStateError, match="synthetic or target"):
        _validate_v81_low_turnover_rank_harness_boundary(
            ROOT, state, experiment, architecture_drift
        )
