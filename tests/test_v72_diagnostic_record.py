from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil

import pytest
import yaml

from tlm.research_workflow import (
    ResearchStateError,
    _validate_v73_v72_diagnostic_record_boundary,
    validate_research_state,
)
from tlm.v72_diagnostic_record import run_v72_diagnostic_record


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/v72_v64_r2_posthoc_retrospective_evaluation"


def _yaml(path: str) -> dict:
    value = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _v73_staging_state() -> dict:
    state = _yaml("research/current.yaml")
    state.update(
        {
            "active_family_id": "tlm_decoupled_rank_state_probabilistic_gate_v2",
            "active_family_status": "posthoc_diagnostic_failed_metadata_record_authorized",
            "last_completed_phase": "v72_posthoc_consumed_2025_diagnostic_evaluation",
            "last_completed_result": "artifacts/v72_v64_r2_posthoc_retrospective_evaluation/result.json",
            "authorized_next_action": "record_v73_v72_posthoc_diagnostic_result_metadata_only",
            "authorized_phase": "v73",
            "authorized_command": (
                "PYTHONPATH=src python3 -m tlm v72-diagnostic-record "
                "--config configs/v73_v72_diagnostic_record.yaml"
            ),
            "evidence_tier": "posthoc_consumed_2025_diagnostic_only_not_confirmation",
        }
    )
    return state


def test_current_state_registers_only_v84_prepare() -> None:
    status = validate_research_state(ROOT, "research/current.yaml")
    assert status["passed"] is True
    assert status["authorized_phase"] == "v85"
    assert status["authorized_next_action"] == (
        "execute_v85_exactly_one_registered_non_target_outcome_unseal_and_complete_evaluation"
    )


def test_v73_boundary_rejects_any_parquet_input() -> None:
    contract = deepcopy(_yaml("research/phase_contracts/v073.yaml"))
    contract["access_contract"]["allowed_inputs"].append(
        "artifacts/v72_v64_r2_posthoc_retrospective_evaluation/daily_returns.parquet"
    )
    with pytest.raises(ResearchStateError, match="metadata input contract"):
        _validate_v73_v72_diagnostic_record_boundary(
            ROOT,
            _v73_staging_state(),
            _yaml("research/experiments/v073.yaml"),
            contract,
        )


def test_v73_runner_is_metadata_only_and_byte_identical(tmp_path: Path) -> None:
    inputs = {
        "result": "result.json",
        "audit": "audit.json",
        "completion_receipt": "completion_receipt.json",
        "replay": "replay.json",
    }
    expected = {
        "result": "3a0ff56c77445ac05bf37fc8c020b392792007afd861475123d6ebcf95417c8f",
        "audit": "b55010b7f9294d49d95342835d6e1976f10c428f29532fe68ebf7666ab0f2d8b",
        "completion_receipt": "025edb0ec18d9f17efce6e0c5b627aef1f3de286ffa2f3aa2fa8cd955c9a3395",
        "replay": "b102d3968daf85e92ece8d05294d298e25994be29706141d363e5ba8702c4f20",
    }
    for name, destination in inputs.items():
        shutil.copyfile(SOURCE / f"{name}.json", tmp_path / destination)
    (tmp_path / "source.py").write_text("VALUE = 73\n", encoding="utf-8")
    config = {
        "v72_diagnostic_record": {
            "project_root": str(tmp_path),
            "inputs": inputs,
            "expected_input_sha256": expected,
            "source_receipt_files": ["source.py"],
        },
        "output_dir": "output",
    }

    first = run_v72_diagnostic_record(config)
    output = tmp_path / "output"
    first_bytes = {path.name: path.read_bytes() for path in output.iterdir()}
    second = run_v72_diagnostic_record(config)
    second_bytes = {path.name: path.read_bytes() for path in output.iterdir()}

    assert first["decision"] == (
        "authorize_v74_persistent_duration_family_specification_only"
    )
    assert first["audit"]["passed"] is True
    assert first["audit"]["access_ledger"] == {
        "json_metadata_reads": 4,
        "parquet_deserializations": 0,
        "outcome_packet_reads": 0,
        "checkpoint_loads": 0,
        "model_instantiations": 0,
        "optimizer_steps": 0,
        "predictions_generated": 0,
        "positions_generated": 0,
        "target_assets_loaded": [],
    }
    assert first_bytes == second_bytes
    assert second["record"]["record_sha256"] == first["record"]["record_sha256"]
