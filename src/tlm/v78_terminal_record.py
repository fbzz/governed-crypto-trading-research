from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping

from .core.artifacts import canonical_sha256, file_sha256, write_json_atomic, write_yaml_atomic


EXPECTED_INPUTS = {
    "result",
    "audit",
    "prepare_failure_receipt",
    "replay_receipt",
}
V78_FAMILY_ID = "tlm_persistent_multi_horizon_duration_v1"
V80_FAMILY_ID = "tlm_low_turnover_cross_sectional_rank_v1"
V78_FAILURE_DECISION = (
    "pivot_away_from_current_family_without_target_evaluation_or_retuning"
)
V80_SPEC_ACTION = "authorize_v80_low_turnover_cross_sectional_rank_specification_only"


def _mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing V79 {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V79 {label} must be a JSON object")
    return value


def _inside(root: Path, relative: str, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"V79 {label} escapes the repository") from exc
    return path


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.replace(path)


def _verify_inputs(
    root: Path, section: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    inputs = section.get("inputs")
    expected_hashes = section.get("expected_input_sha256")
    if not isinstance(inputs, Mapping) or not isinstance(expected_hashes, Mapping):
        raise ValueError("V79 input paths and hashes must be mappings")
    if set(inputs) != EXPECTED_INPUTS or set(expected_hashes) != EXPECTED_INPUTS:
        raise ValueError("V79 metadata input allowlist drift")

    payloads: dict[str, dict[str, Any]] = {}
    observed_hashes: dict[str, str] = {}
    for name in sorted(EXPECTED_INPUTS):
        relative = inputs[name]
        expected = expected_hashes[name]
        if not isinstance(relative, str) or not relative.endswith(".json"):
            raise ValueError("V79 may read only registered JSON metadata")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"Invalid V79 hash for {name}")
        path = _inside(root, relative, f"input {name}")
        observed = file_sha256(path)
        if observed != expected:
            raise ValueError(f"V79 input hash drift: {name}")
        payloads[name] = _mapping(path, name)
        observed_hashes[name] = observed
    return payloads, observed_hashes


def _semantic_checks(payloads: Mapping[str, dict[str, Any]]) -> dict[str, bool]:
    result = payloads["result"]
    audit = payloads["audit"]
    failure = payloads["prepare_failure_receipt"]
    replay = payloads["replay_receipt"]
    checks = audit.get("checks", {})
    summary = result.get("summary", {})
    target = result.get("target_contract", {})
    failed = [
        name
        for name, passed in checks.items()
        if passed is not True
    ] if isinstance(checks, Mapping) else []
    return {
        "v78_terminal_decision_is_exact": (
            result.get("decision") == V78_FAILURE_DECISION
            and audit.get("decision") == V78_FAILURE_DECISION
            and failure.get("decision") == V78_FAILURE_DECISION
            and replay.get("decision") == V78_FAILURE_DECISION
        ),
        "only_registered_turnover_gate_failed": (
            result.get("audit", {}).get("passed") is False
            and result.get("audit", {}).get("failed_checks")
            == ["aggregate_turnover_within_registered_ceiling"]
            and audit.get("passed") is False
            and audit.get("failed_checks")
            == ["aggregate_turnover_within_registered_ceiling"]
            and failed == ["aggregate_turnover_within_registered_ceiling"]
        ),
        "turnover_failure_is_exact_and_not_accounting_bug": (
            summary.get("aggregate_candidate_turnover") == 59.55
            and summary.get("registered_turnover_ceiling") == 45.0
            and summary.get("aggregate_candidate_turnover")
            > summary.get("registered_turnover_ceiling")
            and audit.get("failure_is_accounting_bug") is False
            and audit.get("independent_turnover_audit", {}).get(
                "maximum_daily_turnover_error"
            )
            == 0.0
            and audit.get("independent_turnover_audit", {}).get(
                "maximum_final_liquidation_error"
            )
            == 0.0
            and audit.get("independent_turnover_audit", {}).get(
                "maximum_total_turnover_error"
            )
            == 0.0
        ),
        "outcomes_metrics_pnl_and_targets_remained_zero": (
            summary.get("outcome_rows_read") == 0
            and summary.get("performance_metrics") == 0
            and summary.get("pnl_evaluations") == 0
            and summary.get("target_assets_loaded") == 0
            and audit.get("outcome_rows_read") == 0
            and audit.get("performance_metrics_computed") == 0
            and audit.get("pnl_evaluations") == 0
            and audit.get("target_assets_loaded") == []
            and target.get("status") == "sealed"
            and target.get("target_assets_loaded") == []
            and target.get("target_predictions") == 0
            and target.get("target_pnl_evaluations") == 0
        ),
        "no_unseal_retuning_or_regeneration": (
            result.get("one_shot_packet_created") is False
            and result.get("one_shot_unseal_authorized") is False
            and failure.get("authorizes_unseal") is False
            and failure.get("outcome_rows_read") == 0
            and failure.get("retuning_or_policy_change") is False
            and failure.get("prediction_or_position_regeneration") is False
        ),
        "source_free_replay_matches_without_scientific_access": (
            replay.get("passed") is True
            and replay.get("frozen_artifact_hashes_match") is True
            and replay.get("scientific_source_parquet_deserializations") == 0
            and replay.get("checkpoint_container_deserializations") == 0
            and replay.get("model_instantiations") == 0
            and replay.get("prediction_or_position_regeneration") is False
            and replay.get("outcome_rows_read") == 0
            and replay.get("target_assets_loaded") == []
        ),
        "family_and_successor_ids_are_exact": (
            result.get("family_id") == V78_FAMILY_ID
            and target.get("symbols") == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        ),
    }


def run_v78_terminal_record(config: dict[str, Any]) -> dict[str, Any]:
    """Register the frozen V78 prepare failure without scientific-data access."""

    section = config.get("v78_terminal_record")
    if not isinstance(section, Mapping):
        raise ValueError("Missing v78_terminal_record config section")
    if section.get("retired_family_id") != V78_FAMILY_ID:
        raise ValueError("V79 retired family drift")
    if section.get("successor_family_id") != V80_FAMILY_ID:
        raise ValueError("V79 successor family drift")
    root = Path(section.get("project_root", ".")).resolve()
    output_value = config.get("output_dir")
    if not isinstance(output_value, str):
        raise ValueError("V79 output_dir must be a repository-relative path")
    output = _inside(root, output_value, "output directory")
    payloads, input_hashes = _verify_inputs(root, section)

    checks = _semantic_checks(payloads)
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"V79 semantic gate failed: {failed}")

    result = payloads["result"]
    summary = result["summary"]
    record = {
        "schema_version": "v79-v78-terminal-record/v1",
        "evidence_tier": section["evidence_tier"],
        "owner_authorization_sha256": section["owner_authorization_sha256"],
        "retired_family_id": V78_FAMILY_ID,
        "family_status_before": "trained_outcome_blind_evaluation_prepare_authorized",
        "family_status_after": "retired",
        "terminal_phase": "v78",
        "terminal_decision": V78_FAILURE_DECISION,
        "failed_gate": "aggregate_turnover_within_registered_ceiling",
        "aggregate_candidate_turnover": summary["aggregate_candidate_turnover"],
        "registered_turnover_ceiling": summary["registered_turnover_ceiling"],
        "failure_is_accounting_bug": False,
        "outcome_rows_read": 0,
        "performance_metrics_computed": 0,
        "pnl_evaluations": 0,
        "target_assets_status": "sealed",
        "target_assets_loaded": [],
        "successor_family_id": V80_FAMILY_ID,
        "successor_authorized_phase": "v80_outcome_blind_specification_only",
        "successor_specification_executed": False,
        "deployable": False,
        "record_only": True,
    }
    record["record_sha256"] = canonical_sha256(record)
    phase_result = {
        "schema_version": "v79-result/v1",
        "decision": V80_SPEC_ACTION,
        "record_sha256": record["record_sha256"],
        "retired_family_id": V78_FAMILY_ID,
        "retired_family_status": "retired",
        "successor_family_id": V80_FAMILY_ID,
        "successor_specification_executed": False,
        "json_metadata_reads": 4,
        "outcome_rows_read": 0,
        "scientific_metrics_recomputed": 0,
        "models_or_checkpoints_loaded": 0,
        "target_assets_loaded": [],
        "deployable": False,
    }
    phase_result["result_sha256"] = canonical_sha256(phase_result)
    access_ledger = {
        "json_metadata_reads": 4,
        "parquet_deserializations": 0,
        "market_panel_reads": 0,
        "outcome_packet_reads": 0,
        "checkpoint_loads": 0,
        "model_instantiations": 0,
        "scaler_fits": 0,
        "optimizer_steps": 0,
        "training_runs": 0,
        "inference_runs": 0,
        "predictions_generated": 0,
        "positions_generated": 0,
        "scientific_metrics_recomputed": 0,
        "pnl_evaluations": 0,
        "target_assets_loaded": [],
    }
    audit = {
        "schema_version": "v79-audit/v1",
        "passed": True,
        "checks": checks,
        "access_ledger": access_ledger,
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    input_receipt = {
        "schema_version": "v79-input-receipt/v1",
        "inputs": input_hashes,
    }
    input_receipt["input_receipt_sha256"] = canonical_sha256(input_receipt)

    source_files = section.get("source_receipt_files")
    if not isinstance(source_files, list) or not source_files:
        raise ValueError("V79 source receipt file list is required")
    source_hashes: dict[str, str] = {}
    for relative in source_files:
        if not isinstance(relative, str):
            raise ValueError("V79 source receipt paths must be strings")
        source_hashes[relative] = file_sha256(_inside(root, relative, "source file"))
    source_receipt = {
        "schema_version": "v79-source-receipt/v1",
        "files": source_hashes,
    }
    source_receipt["source_receipt_sha256"] = canonical_sha256(source_receipt)

    report = "\n".join([
        "# V79 metadata-only V78 terminal record",
        "",
        "V78 failed only its preregistered turnover behavior gate: 59.55",
        "aggregate turnover against a ceiling of 45.0. Independent accounting",
        "checks had zero error, so the persistent-duration family is retired",
        "without outcome unseal, target evaluation, retuning, or regeneration.",
        "",
        "This phase read four hash-bound JSON metadata files and performed no",
        "Parquet, checkpoint, model, training, inference, prediction, position,",
        "metric/PnL, outcome-packet, or target-asset operation. It authorizes only",
        "the separate V80 outcome-blind specification for the final successor",
        "family; that specification was not executed in V79.",
        "",
    ])

    output.mkdir(parents=True, exist_ok=True)
    write_yaml_atomic(output / "resolved_config.yaml", config)
    write_json_atomic(output / "input_hash_receipt.json", input_receipt)
    write_json_atomic(output / "source_receipt.json", source_receipt)
    write_json_atomic(output / "terminal_record.json", record)
    write_json_atomic(output / "result.json", phase_result)
    write_json_atomic(output / "audit.json", audit)
    _write_text_atomic(output / "report.md", report)

    manifest_files = [
        "resolved_config.yaml",
        "input_hash_receipt.json",
        "source_receipt.json",
        "terminal_record.json",
        "result.json",
        "audit.json",
        "report.md",
    ]
    manifest = {
        "schema_version": "v79-artifact-manifest/v1",
        "files": {name: file_sha256(output / name) for name in manifest_files},
    }
    manifest["artifact_manifest_sha256"] = canonical_sha256(manifest)
    write_json_atomic(output / "artifact_manifest.json", manifest)
    return {
        "decision": phase_result["decision"],
        "record": record,
        "result": phase_result,
        "audit": audit,
        "artifact_manifest": manifest,
    }
