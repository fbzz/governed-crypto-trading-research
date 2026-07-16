from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping

from .core.artifacts import canonical_sha256, file_sha256, write_json_atomic, write_yaml_atomic


EXPECTED_INPUTS = {
    "result",
    "audit",
    "completion_receipt",
    "replay",
}


def _mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing V73 {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V73 {label} must be a JSON object")
    return value


def _inside(root: Path, relative: str, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"V73 {label} escapes the repository") from exc
    return path


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def _verify_inputs(
    root: Path, section: Mapping[str, Any]
) -> tuple[dict[str, Path], dict[str, dict[str, Any]], dict[str, str]]:
    inputs = section.get("inputs")
    expected_hashes = section.get("expected_input_sha256")
    if not isinstance(inputs, Mapping) or not isinstance(expected_hashes, Mapping):
        raise ValueError("V73 input paths and hashes must be mappings")
    if set(inputs) != EXPECTED_INPUTS or set(expected_hashes) != EXPECTED_INPUTS:
        raise ValueError("V73 metadata input allowlist drift")

    paths: dict[str, Path] = {}
    payloads: dict[str, dict[str, Any]] = {}
    observed_hashes: dict[str, str] = {}
    for name in sorted(EXPECTED_INPUTS):
        relative = inputs[name]
        expected = expected_hashes[name]
        if not isinstance(relative, str) or not relative.endswith(".json"):
            raise ValueError("V73 may read only registered JSON metadata")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"Invalid V73 hash for {name}")
        path = _inside(root, relative, f"input {name}")
        observed = file_sha256(path)
        if observed != expected:
            raise ValueError(f"V73 input hash drift: {name}")
        paths[name] = path
        payloads[name] = _mapping(path, name)
        observed_hashes[name] = observed
    return paths, payloads, observed_hashes


def _semantic_checks(payloads: Mapping[str, dict[str, Any]]) -> dict[str, bool]:
    result = payloads["result"]
    audit = payloads["audit"]
    completion = payloads["completion_receipt"]
    replay = payloads["replay"]
    return {
        "v72_diagnostic_failed_without_family_status_change": (
            result.get("diagnostic_outcome") == "fail"
            and result.get("one_shot_decision") == "retire"
            and result.get("family_status_changed") is False
        ),
        "all_registered_gate_cells_preserved": (
            result.get("mandatory_gate_count") == 24
            and result.get("passed_gate_count") == 13
            and result.get("failed_gate_count") == 11
        ),
        "one_packet_unseal_and_zero_source_rereads": (
            result.get("unseal_count") == 1
            and result.get("sealed_packet_deserializations") == 1
            and result.get("underlying_source_outcome_reads") == 0
            and completion.get("unseal_count") == 1
            and completion.get("sealed_packet_deserializations") == 1
            and completion.get("source_outcome_reads") == 0
        ),
        "no_retuning_or_regeneration": (
            result.get("retuning_performed") is False
            and result.get("prediction_or_position_regeneration") is False
            and replay.get("new_inference") == 0
            and replay.get("new_position_generation") == 0
            and replay.get("new_checkpoint_loads") == 0
        ),
        "source_free_replay_matches": (
            replay.get("result_hashes_match") is True
            and replay.get("sealed_source_packet_deserializations") == 0
            and replay.get("source_outcome_rows_read") == 0
        ),
        "repository_audit_passed_scientific_gate_failed": (
            audit.get("passed") is True
            and audit.get("scientific_gates_all_passed") is False
        ),
        "target_assets_remained_sealed": (
            result.get("target_assets_loaded") == []
            and result.get("target_predictions") == 0
            and result.get("target_pnl_evaluations") == 0
            and completion.get("target_assets_status") == "sealed"
            and replay.get("target_assets_loaded") == []
        ),
    }


def run_v72_diagnostic_record(config: dict[str, Any]) -> dict[str, Any]:
    """Record the completed V72 result without opening scientific data."""

    section = config.get("v72_diagnostic_record")
    if not isinstance(section, Mapping):
        raise ValueError("Missing v72_diagnostic_record config section")
    root = Path(section.get("project_root", ".")).resolve()
    output_value = config.get("output_dir")
    if not isinstance(output_value, str):
        raise ValueError("V73 output_dir must be a repository-relative path")
    output = _inside(root, output_value, "output directory")
    _, payloads, input_hashes = _verify_inputs(root, section)

    checks = _semantic_checks(payloads)
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"V73 semantic gate failed: {failed}")

    result = payloads["result"]
    record = {
        "schema_version": "v73-v72-diagnostic-record/v1",
        "family_id": "tlm_decoupled_rank_state_probabilistic_gate_v2",
        "lineage_label": "V64-R2",
        "evidence_tier": "posthoc_consumed_2025_diagnostic_only_not_confirmation",
        "diagnostic_outcome": result["diagnostic_outcome"],
        "one_shot_decision": result["one_shot_decision"],
        "family_status_changed": result["family_status_changed"],
        "mandatory_gate_count": result["mandatory_gate_count"],
        "passed_gate_count": result["passed_gate_count"],
        "failed_gate_count": result["failed_gate_count"],
        "candidate_aggregate": result["candidate_aggregate"],
        "target_assets_status": "sealed",
        "deployable": False,
        "record_only": True,
    }
    record["record_sha256"] = canonical_sha256(record)
    phase_result = {
        "schema_version": "v73-result/v1",
        "decision": "authorize_v74_persistent_duration_family_specification_only",
        "record_sha256": record["record_sha256"],
        "family_status_changed": False,
        "outcomes_opened": 0,
        "models_or_checkpoints_loaded": 0,
        "target_assets_loaded": [],
        "deployable": False,
    }
    audit = {
        "schema_version": "v73-audit/v1",
        "passed": True,
        "checks": checks,
        "access_ledger": {
            "json_metadata_reads": 4,
            "parquet_deserializations": 0,
            "outcome_packet_reads": 0,
            "checkpoint_loads": 0,
            "model_instantiations": 0,
            "optimizer_steps": 0,
            "predictions_generated": 0,
            "positions_generated": 0,
            "target_assets_loaded": [],
        },
    }
    input_receipt = {
        "schema_version": "v73-input-receipt/v1",
        "inputs": input_hashes,
    }
    input_receipt["input_receipt_sha256"] = canonical_sha256(input_receipt)

    source_files = section.get("source_receipt_files")
    if not isinstance(source_files, list) or not source_files:
        raise ValueError("V73 source receipt file list is required")
    source_hashes: dict[str, str] = {}
    for relative in source_files:
        if not isinstance(relative, str):
            raise ValueError("V73 source receipt paths must be strings")
        source_hashes[relative] = file_sha256(_inside(root, relative, "source file"))
    source_receipt = {
        "schema_version": "v73-source-receipt/v1",
        "files": source_hashes,
    }
    source_receipt["source_receipt_sha256"] = canonical_sha256(source_receipt)

    report = "\n".join([
        "# V73 metadata-only V72 diagnostic record",
        "",
        "V72 failed its scientific diagnostic: 13 of 24 registered gates passed.",
        "The V64-R2 family status is unchanged and no deployable claim is made.",
        "",
        "This phase read four hash-bound JSON metadata files, opened no outcome",
        "packet, model, checkpoint, prediction, position, or target-asset data, and",
        "authorizes only a separate V74 family specification.",
        "",
    ])

    output.mkdir(parents=True, exist_ok=True)
    write_yaml_atomic(output / "resolved_config.yaml", config)
    write_json_atomic(output / "input_hash_receipt.json", input_receipt)
    write_json_atomic(output / "source_receipt.json", source_receipt)
    write_json_atomic(output / "diagnostic_record.json", record)
    write_json_atomic(output / "result.json", phase_result)
    write_json_atomic(output / "audit.json", audit)
    _write_text_atomic(output / "report.md", report)

    manifest_files = [
        "resolved_config.yaml",
        "input_hash_receipt.json",
        "source_receipt.json",
        "diagnostic_record.json",
        "result.json",
        "audit.json",
        "report.md",
    ]
    manifest = {
        "schema_version": "v73-artifact-manifest/v1",
        "files": {
            name: file_sha256(output / name) for name in manifest_files
        },
    }
    manifest["artifact_manifest_sha256"] = canonical_sha256(manifest)
    write_json_atomic(output / "artifact_manifest.json", manifest)
    return {
        "decision": phase_result["decision"],
        "record": record,
        "audit": audit,
        "artifact_manifest": manifest,
    }
