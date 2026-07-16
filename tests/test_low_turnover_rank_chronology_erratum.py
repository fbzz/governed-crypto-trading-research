from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import yaml

from tlm.low_turnover_rank_chronology_erratum import (
    run_low_turnover_rank_chronology_erratum,
)
from tlm.research_workflow import (
    ResearchStateError,
    _validate_v82_r0_low_turnover_rank_chronology_erratum_boundary,
    validate_research_state,
)


ROOT = Path(__file__).resolve().parents[1]


def _yaml(relative: str) -> dict:
    value = yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _json(relative: str) -> dict:
    value = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _file_hashes(path: Path) -> dict[str, str]:
    return {
        item.name: hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(path.iterdir())
        if item.is_file()
    }


def _v82_r0_staging_state() -> dict:
    state = _yaml("research/current.yaml")
    state.update(
        {
            "authorized_phase": "v82-r0",
            "authorized_next_action": (
                "record_v82_r0_low_turnover_rank_chronology_erratum_metadata_only"
            ),
            "authorized_command": (
                "PYTHONPATH=src python3 -m tlm low-turnover-rank-chronology-erratum "
                "--config configs/v82_r0_low_turnover_rank_chronology_erratum.yaml"
            ),
            "active_family_status": "metadata_only_chronology_erratum_authorized",
            "last_completed_phase": "v81_synthetic_low_turnover_rank_harness",
            "last_completed_result": "artifacts/v81_low_turnover_rank_harness/result.json",
            "evidence_tier": "metadata_only_chronology_erratum",
        }
    )
    return state


def test_v82_r0_boundary_remains_reconstructable_after_v82_registration() -> None:
    status = validate_research_state(ROOT, "research/current.yaml")
    assert status["passed"] is True
    assert status["authorized_phase"] == "v85"
    assert status["authorized_next_action"] == (
        "execute_v85_exactly_one_registered_non_target_outcome_unseal_and_complete_evaluation"
    )
    assert status["active_family_id"] == "tlm_low_turnover_cross_sectional_rank_v1"
    assert status["target_asset_status"] == "sealed"
    assert status["deployable_strategy"] is False

    _validate_v82_r0_low_turnover_rank_chronology_erratum_boundary(
        ROOT,
        _v82_r0_staging_state(),
        _yaml("research/experiments/v082_r0.yaml"),
        _yaml("research/phase_contracts/v082_r0.yaml"),
    )


def test_v82_r0_runner_changes_only_two_chronology_fields_and_replays() -> None:
    config = _yaml("configs/v82_r0_low_turnover_rank_chronology_erratum.yaml")
    with TemporaryDirectory(dir=ROOT / "artifacts") as directory:
        output = Path(directory)
        config["output_dir"] = str(output.relative_to(ROOT))
        first = run_low_turnover_rank_chronology_erratum(config)
        first_hashes = _file_hashes(output)
        second = run_low_turnover_rank_chronology_erratum(config)
        second_hashes = _file_hashes(output)

    assert first_hashes == second_hashes
    assert len(first_hashes) == 8
    assert first["decision"] == (
        "authorize_v82_non_target_low_turnover_rank_dataset_only"
    )
    assert first == second
    assert first["audit"]["passed"] is True
    assert first["audit"]["checks_passed"] == first["audit"]["checks_total"] == 14
    assert first["erratum"]["arithmetic_evidence"] == {
        "changed_fields": [
            "final_evaluation_signal_dates",
            "final_evaluation_signal_end",
        ],
        "maximum_decisions": 8,
        "new_last_maturity": "2026-06-30",
        "new_signal_dates": 159,
        "old_last_maturity": "2026-07-01",
        "old_signal_dates": 160,
        "structural_maximum_turnover": 16.0,
    }
    assert first["erratum"]["scientific_change_count"] == 0
    assert first["erratum"]["target_assets_loaded"] == []
    assert all(
        value == 0
        for key, value in first["audit"]["access_ledger"].items()
        if key not in {"json_metadata_reads", "target_assets_loaded"}
    )
    assert first["audit"]["access_ledger"]["json_metadata_reads"] == 8


def test_v82_r0_runner_rejects_extra_input_or_scientific_change() -> None:
    config = _yaml("configs/v82_r0_low_turnover_rank_chronology_erratum.yaml")
    extra = deepcopy(config)
    extra["low_turnover_rank_chronology_erratum"]["inputs"]["panel"] = (
        "data/processed/selected_universe_panel_v32.parquet"
    )
    extra["low_turnover_rank_chronology_erratum"]["expected_input_sha256"][
        "panel"
    ] = "0" * 64
    with pytest.raises(ValueError, match="allowlist drift"):
        run_low_turnover_rank_chronology_erratum(extra)

    changed = deepcopy(config)
    changed["low_turnover_rank_chronology_erratum"]["frozen_invariants"][
        "structural_maximum_turnover"
    ] = 15.0
    with pytest.raises(ValueError, match="semantic gate failed"):
        run_low_turnover_rank_chronology_erratum(changed)


def test_v82_r0_boundary_rejects_data_input_or_scope_drift() -> None:
    state = _v82_r0_staging_state()
    experiment = _yaml("research/experiments/v082_r0.yaml")
    contract = deepcopy(_yaml("research/phase_contracts/v082_r0.yaml"))
    forbidden = "data/processed/selected_universe_panel_v32.parquet"
    contract["access_contract"]["allowed_inputs"].append(forbidden)
    contract["input_contract"]["expected_static_file_sha256_by_path"][forbidden] = (
        "0" * 64
    )
    with pytest.raises(ResearchStateError, match="metadata-only input contract"):
        _validate_v82_r0_low_turnover_rank_chronology_erratum_boundary(
            ROOT, state, experiment, contract
        )


def test_v82_r0_user_authorization_is_self_hash_valid() -> None:
    authorization = _json(
        "research/authorizations/v082_r0_chronology_erratum.json"
    )
    registered = authorization.pop("authorization_sha256")
    canonical = hashlib.sha256(
        json.dumps(
            authorization,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert registered == canonical == (
        "7120faa7e6267e2234bb34b576a65aa4834bc1b527a9012ab100b7d3c409394c"
    )
