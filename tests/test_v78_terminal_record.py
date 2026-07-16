from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil

import pytest
import yaml

from tlm.research_workflow import (
    ResearchStateError,
    _validate_v79_v78_terminal_record_boundary,
)
from tlm.v78_terminal_record import run_v78_terminal_record


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/v78_persistent_duration_evaluation"


def _yaml(relative: str) -> dict:
    value = yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _v79_staging_state() -> dict:
    return {
        "authorized_phase": "v79",
        "authorized_next_action": "record_v79_v78_terminal_failure_metadata_only",
        "authorized_command": (
            "PYTHONPATH=src python3 -m tlm v78-terminal-record "
            "--config configs/v79_v78_terminal_record.yaml"
        ),
        "active_family_id": "tlm_persistent_multi_horizon_duration_v1",
        "active_family_status": (
            "outcome_blind_prepare_failed_metadata_record_authorized"
        ),
        "last_completed_phase": (
            "v78_outcome_blind_persistent_duration_evaluation_prepare_failed"
        ),
        "last_completed_result": (
            "artifacts/v78_persistent_duration_evaluation/result.json"
        ),
        "evidence_tier": (
            "adaptive_consumed_2025_non_target_development_prepare_outcomes_sealed"
        ),
        "target_assets": {"status": "sealed"},
        "deployable_strategy": False,
        "safety": {
            "v79_metadata_only_terminal_recording_phase": True,
            "v79_parquet_checkpoint_model_outcome_or_target_access_allowed": False,
            "v79_scientific_recomputation_allowed": False,
            "v79_v78_family_retirement_authorized": True,
            "v79_v80_specification_execution_allowed": False,
            "v79_target_assets_remain_sealed": True,
        },
    }


def test_v79_boundary_accepts_only_four_hash_bound_json_inputs() -> None:
    _validate_v79_v78_terminal_record_boundary(
        ROOT,
        _v79_staging_state(),
        _yaml("research/experiments/v079.yaml"),
        _yaml("research/phase_contracts/v079.yaml"),
    )

    drift = deepcopy(_yaml("research/phase_contracts/v079.yaml"))
    drift["access_contract"]["allowed_inputs"].append(
        "data/predictions/v78_persistent_duration_predictions.parquet"
    )
    with pytest.raises(ResearchStateError, match="metadata input contract"):
        _validate_v79_v78_terminal_record_boundary(
            ROOT,
            _v79_staging_state(),
            _yaml("research/experiments/v079.yaml"),
            drift,
        )


def test_v79_runner_is_metadata_only_and_byte_identical(tmp_path: Path) -> None:
    inputs = {
        "result": "result.json",
        "audit": "audit.json",
        "prepare_failure_receipt": "prepare_failure_receipt.json",
        "replay_receipt": "replay_receipt.json",
    }
    expected = {
        "result": "ce50db6966f3ebcbcde63994603a62bd2f20545ecd6e1ffd6f5aef2047db9bba",
        "audit": "6003e7e358d52e3e83692bf1499f39551510aff1060d96a3b052dc859a30e818",
        "prepare_failure_receipt": (
            "3269eb9ceaa6d92bc96970793a69ac96f6e66bf7608d52740e199c8558e64f3a"
        ),
        "replay_receipt": (
            "651e8f1814be682c077dc5526c5b4bac90b12de77f4fdade7c541cfb18e134fc"
        ),
    }
    for name, destination in inputs.items():
        shutil.copyfile(SOURCE / f"{name}.json", tmp_path / destination)
    (tmp_path / "source.py").write_text("VALUE = 79\n", encoding="utf-8")
    config = {
        "v78_terminal_record": {
            "project_root": str(tmp_path),
            "retired_family_id": "tlm_persistent_multi_horizon_duration_v1",
            "successor_family_id": "tlm_low_turnover_cross_sectional_rank_v1",
            "evidence_tier": (
                "adaptive_consumed_2025_non_target_development_prepare_outcomes_sealed"
            ),
            "owner_authorization_sha256": (
                "bf24b776fe7df07132d6b16beaea636a40221ee98fb9c6f45e9157023779b85c"
            ),
            "inputs": inputs,
            "expected_input_sha256": expected,
            "source_receipt_files": ["source.py"],
        },
        "output_dir": "output",
    }

    first = run_v78_terminal_record(config)
    output = tmp_path / "output"
    first_bytes = {path.name: path.read_bytes() for path in output.iterdir()}
    second = run_v78_terminal_record(config)
    second_bytes = {path.name: path.read_bytes() for path in output.iterdir()}

    assert first["decision"] == (
        "authorize_v80_low_turnover_cross_sectional_rank_specification_only"
    )
    assert first["record"]["family_status_after"] == "retired"
    assert first["record"]["successor_specification_executed"] is False
    assert first["audit"]["passed"] is True
    assert first["audit"]["access_ledger"]["json_metadata_reads"] == 4
    assert first["audit"]["access_ledger"]["parquet_deserializations"] == 0
    assert first["audit"]["access_ledger"]["checkpoint_loads"] == 0
    assert first["audit"]["access_ledger"]["outcome_packet_reads"] == 0
    assert first["audit"]["access_ledger"]["target_assets_loaded"] == []
    assert first_bytes == second_bytes
    assert second["record"]["record_sha256"] == first["record"]["record_sha256"]
