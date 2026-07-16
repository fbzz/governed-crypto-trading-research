from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tlm.__main__ import build_parser
from tlm.config import load_config
from tlm.v64_r2_prospective_confirmation_prepare import (
    run_v64_r2_prospective_confirmation_prepare,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v69_v64_r2_prospective_confirmation_prepare.yaml"
EXPECTED_FILES = {
    "specification.json",
    "protocol.json",
    "audit.json",
    "result.json",
    "report.md",
    "source_receipt.json",
    "artifact_manifest.json",
    "completion_receipt.json",
}


def _config(tmp_path: Path) -> dict:
    config = copy.deepcopy(load_config(CONFIG))
    config["v64_r2_prospective_confirmation_prepare"]["project_root"] = str(ROOT)
    config["output_dir"] = str(tmp_path / "packet")
    return config


def _hashes(path: Path) -> dict[str, str]:
    return {
        item.name: hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(path.iterdir())
        if item.is_file()
    }


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_v69_packet_replays_byte_identically_and_is_metadata_only(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    first = run_v64_r2_prospective_confirmation_prepare(config)
    first_hashes = _hashes(tmp_path / "packet")
    second = run_v64_r2_prospective_confirmation_prepare(config)
    second_hashes = _hashes(tmp_path / "packet")

    assert first == second
    assert first_hashes == second_hashes
    assert set(first_hashes) == EXPECTED_FILES
    assert first["decision"] == (
        "authorize_v70_prospective_non_target_capture_and_prediction_freeze_only"
    )
    assert first["audit"]["passed"] is True
    assert first["summary"]["metadata_json_reads"] == 13
    assert first["summary"]["checkpoint_identity_receipts"] == 9
    for field in (
        "parquet_deserializations",
        "raw_market_data_reads",
        "checkpoint_deserializations",
        "model_instantiations",
        "predictions",
        "positions",
        "performance_metrics",
        "pnl_computations",
        "outcome_source_reads",
        "target_asset_rows",
    ):
        assert first["summary"][field] == 0


def test_v69_protocol_freezes_clock_accounting_and_all_36_gates(
    tmp_path: Path,
) -> None:
    run_v64_r2_prospective_confirmation_prepare(_config(tmp_path))
    protocol = json.loads(
        (tmp_path / "packet/protocol.json").read_text(encoding="utf-8")
    )

    assert protocol["registration"]["first_admissible_signal_date_rule"] == (
        "strictly_after_v69_completion_receipt_commit"
    )
    assert protocol["evidence_window"]["minimum_calendar_days"] == 120
    assert protocol["evidence_window"][
        "minimum_fully_matured_signal_dates_per_fold"
    ] == 90
    assert protocol["policy"]["reporting_cost_bps"] == [10, 20, 30]
    assert protocol["accounting"][
        "charge_entry_switch_exit_forced_exit_and_final_liquidation"
    ] is True
    assert len(protocol["gate_matrix"]) == 36
    assert all(row["mandatory"] for row in protocol["gate_matrix"])
    assert protocol["outcome_dependent_gates"]["aggregate_rescue_allowed"] is False
    assert protocol["prediction_freeze"]["regeneration_after_freeze_allowed"] is False
    assert protocol["one_shot"]["evaluation_count"] == 1
    assert protocol["one_shot"]["outcome_source_read_count"] == 1
    assert all(value is False for value in protocol["operational_boundary"].values())
    assert protocol["target_contract"]["status"] == "sealed"


def test_v69_manifest_and_receipts_are_self_consistent(tmp_path: Path) -> None:
    run_v64_r2_prospective_confirmation_prepare(_config(tmp_path))
    output = tmp_path / "packet"
    manifest = json.loads((output / "artifact_manifest.json").read_text())
    completion = json.loads((output / "completion_receipt.json").read_text())
    manifest_body = dict(manifest)
    manifest_hash = manifest_body.pop("artifact_manifest_sha256")
    completion_body = dict(completion)
    completion_hash = completion_body.pop("completion_receipt_sha256")

    assert manifest_hash == _canonical_sha256(manifest_body)
    assert completion_hash == _canonical_sha256(completion_body)
    assert manifest["files"] == {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in (
            "specification.json",
            "protocol.json",
            "audit.json",
            "result.json",
            "report.md",
            "source_receipt.json",
        )
    }
    assert completion["audit_passed"] is True
    assert completion["first_admissible_signal_date_rule"] == (
        "strictly_after_v69_completion_receipt_commit"
    )


def test_v69_rejects_non_json_input_and_hash_drift(tmp_path: Path) -> None:
    config = _config(tmp_path)
    spec = config["v64_r2_prospective_confirmation_prepare"]
    original = spec["inputs"]["v65_blueprint"]
    spec["inputs"]["v65_blueprint"] = "data/forbidden.parquet"
    expected = spec["expected_file_sha256_by_path"].pop(original)
    spec["expected_file_sha256_by_path"]["data/forbidden.parquet"] = expected
    with pytest.raises(RuntimeError, match="JSON metadata inputs only"):
        run_v64_r2_prospective_confirmation_prepare(config)

    config = _config(tmp_path)
    config["v64_r2_prospective_confirmation_prepare"][
        "expected_file_sha256_by_path"
    ]["artifacts/v65_v64_r2_probabilistic_state_gate_spec/blueprint.json"] = "0" * 64
    with pytest.raises(RuntimeError, match="immutable metadata input drift"):
        run_v64_r2_prospective_confirmation_prepare(config)


def test_v69_cli_command_is_registered() -> None:
    args = build_parser().parse_args(
        [
            "v64-r2-prospective-confirmation-prepare",
            "--config",
            str(CONFIG),
        ]
    )
    assert args.command == "v64-r2-prospective-confirmation-prepare"
