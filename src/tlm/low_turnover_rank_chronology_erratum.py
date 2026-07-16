from __future__ import annotations

from copy import deepcopy
from datetime import date, timedelta
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping

from .core.artifacts import canonical_sha256, file_sha256, write_json_atomic, write_yaml_atomic


EXPECTED_INPUTS = {
    "authorization",
    "v80_specification",
    "v80_blueprint",
    "v80_result",
    "v81_harness_spec",
    "v81_audit",
    "v81_result",
    "v81_artifact_manifest",
}
FAMILY_ID = "tlm_low_turnover_cross_sectional_rank_v1"
V82_DATASET_ACTION = "authorize_v82_non_target_low_turnover_rank_dataset_only"


def _inside(root: Path, relative: str, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"V82-R0 {label} escapes the repository") from exc
    return path


def _mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing V82-R0 {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V82-R0 {label} must be a JSON object")
    return value


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


def _registered_hash(
    value: Mapping[str, Any], field: str, expected: str
) -> bool:
    payload = dict(value)
    registered = payload.pop(field, None)
    return registered == expected == canonical_sha256(payload)


def _verify_inputs(
    root: Path, section: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    inputs = section.get("inputs")
    expected_hashes = section.get("expected_input_sha256")
    if not isinstance(inputs, Mapping) or not isinstance(expected_hashes, Mapping):
        raise ValueError("V82-R0 input paths and hashes must be mappings")
    if set(inputs) != EXPECTED_INPUTS or set(expected_hashes) != EXPECTED_INPUTS:
        raise ValueError("V82-R0 metadata input allowlist drift")

    payloads: dict[str, dict[str, Any]] = {}
    observed: dict[str, str] = {}
    for name in sorted(EXPECTED_INPUTS):
        relative = inputs[name]
        expected = expected_hashes[name]
        if not isinstance(relative, str) or not relative.endswith(".json"):
            raise ValueError("V82-R0 may read only registered JSON metadata")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"Invalid V82-R0 input hash for {name}")
        path = _inside(root, relative, f"input {name}")
        digest = file_sha256(path)
        if digest != expected:
            raise ValueError(f"V82-R0 input hash drift: {name}")
        payloads[name] = _mapping(path, name)
        observed[name] = digest
    return payloads, observed


def _semantic_checks(
    payloads: Mapping[str, dict[str, Any]], section: Mapping[str, Any]
) -> tuple[dict[str, bool], dict[str, Any]]:
    authorization = payloads["authorization"]
    specification = payloads["v80_specification"]
    blueprint = payloads["v80_blueprint"]
    v80_result = payloads["v80_result"]
    harness = payloads["v81_harness_spec"]
    v81_audit = payloads["v81_audit"]
    v81_result = payloads["v81_result"]
    v81_manifest = payloads["v81_artifact_manifest"]
    registered = section["expected_registered_sha256"]
    correction = section["correction"]
    frozen = section["frozen_invariants"]

    chronology = blueprint.get("chronology", {})
    target = blueprint.get("target", {})
    architecture = blueprint.get("architecture", {})
    policy = blueprint.get("policy", {})
    evaluation = blueprint.get("evaluation", {})
    training = blueprint.get("training", {})

    old_end = date.fromisoformat(correction["final_evaluation_signal_end_before"])
    new_end = date.fromisoformat(correction["final_evaluation_signal_end_after"])
    start = date.fromisoformat(correction["final_evaluation_signal_start"])
    maturity_end = date.fromisoformat(
        correction["final_evaluation_outcome_maturity_end"]
    )
    maturity_offset = int(correction["maturity_offset_days"])
    old_maturity = old_end + timedelta(days=maturity_offset)
    new_maturity = new_end + timedelta(days=maturity_offset)
    old_count = (old_end - start).days + 1
    new_count = (new_end - start).days + 1
    decisions = ((new_count - 1) // int(frozen["decision_interval_days"])) + 1
    structural_turnover = 1.0 + 2.0 * (decisions - 1) + 1.0

    effective = deepcopy(chronology)
    effective["final_evaluation_signal_end"] = new_end.isoformat()
    effective["final_evaluation_signal_dates"] = new_count
    changed_fields = sorted(
        key for key in set(chronology) | set(effective)
        if chronology.get(key) != effective.get(key)
    )

    expected_authorization_text = (
        "Autorizo a V82-R0 metadata-only para corrigir exclusivamente "
        "2026-06-09 para 2026-06-08 e 160 para 159 sinais, preservando "
        "target, arquitetura, política, custos, gates e turnover, sem "
        "acessar dados, outcomes, checkpoints ou BTC/ETH/SOL."
    )
    v81_files = v81_manifest.get("files", {})
    checks = {
        "authorization_is_exact_and_self_hash_valid": (
            authorization.get("authorization_text") == expected_authorization_text
            and authorization.get("authorized_phase") == "v82-r0"
            and authorization.get("target_assets_status") == "sealed"
            and _registered_hash(
                authorization,
                "authorization_sha256",
                registered["authorization"],
            )
        ),
        "authorization_forbids_scientific_and_target_access": (
            set(authorization.get("forbidden_accesses", []))
            == {
                "parquet_or_market_panel",
                "outcome_or_target_value",
                "checkpoint_or_model",
                "training_or_inference",
                "prediction_position_metric_pnl_or_bootstrap",
                "btc_eth_sol",
            }
        ),
        "v80_registered_hashes_are_exact": (
            _registered_hash(
                specification,
                "specification_sha256",
                registered["v80_specification"],
            )
            and _registered_hash(
                blueprint,
                "blueprint_sha256",
                registered["v80_blueprint"],
            )
            and _registered_hash(
                v80_result, "result_sha256", registered["v80_result"]
            )
        ),
        "v80_family_and_old_chronology_are_exact": (
            specification.get("family_id") == FAMILY_ID
            and chronology.get("final_evaluation_signal_start")
            == start.isoformat()
            and chronology.get("final_evaluation_signal_end")
            == old_end.isoformat()
            and chronology.get("final_evaluation_signal_dates") == old_count == 160
            and chronology.get("final_evaluation_outcome_maturity_end")
            == maturity_end.isoformat()
        ),
        "v81_registered_hashes_are_exact": (
            _registered_hash(
                harness,
                "harness_spec_sha256",
                registered["v81_harness_spec"],
            )
            and _registered_hash(
                v81_audit, "audit_sha256", registered["v81_audit"]
            )
            and _registered_hash(
                v81_result, "result_sha256", registered["v81_result"]
            )
            and _registered_hash(
                v81_manifest,
                "artifact_manifest_sha256",
                registered["v81_artifact_manifest"],
            )
        ),
        "v81_harness_passed_without_scientific_access": (
            v81_audit.get("passed") is True
            and v81_audit.get("checks_passed") == 15
            and v81_audit.get("checks_total") == 15
            and all(v81_audit.get("checks", {}).values())
            and v81_result.get("decision") == V82_DATASET_ACTION
            and v81_result.get("v82_executed") is False
            and v81_result.get("outcome_rows_read") == 0
            and v81_result.get("prior_checkpoint_loads") == 0
            and v81_result.get("real_data_reads") == 0
            and v81_result.get("target_assets_loaded") == []
        ),
        "v81_manifest_binds_exact_gate_files": (
            v81_files.get("harness_spec.json")
            == section["expected_input_sha256"]["v81_harness_spec"]
            and v81_files.get("audit.json")
            == section["expected_input_sha256"]["v81_audit"]
            and v81_files.get("result.json")
            == section["expected_input_sha256"]["v81_result"]
            and len(v81_files) == 8
        ),
        "old_last_signal_matures_after_frozen_end": (
            old_maturity.isoformat() == "2026-07-01"
            and old_maturity > maturity_end
        ),
        "corrected_signal_count_is_exact": (
            new_count
            == correction["final_evaluation_signal_dates_after"]
            == 159
        ),
        "corrected_last_maturity_is_exact": (
            new_maturity == maturity_end
            and new_maturity.isoformat() == "2026-06-30"
        ),
        "only_two_chronology_fields_change": (
            changed_fields
            == ["final_evaluation_signal_dates", "final_evaluation_signal_end"]
        ),
        "target_architecture_policy_costs_and_gates_are_frozen": (
            target.get("asset_return") == frozen["target_asset_return"]
            and target.get("execution_open") == frozen["execution_open"]
            and target.get("exit_open") == frozen["exit_open"]
            and target.get("horizon_intervals") == frozen["horizon_intervals"]
            and architecture.get("expected_total_parameters")
            == frozen["expected_total_parameters"]
            and architecture.get("architecture_variant_count")
            == frozen["architecture_variant_count"]
            and policy.get("decision_interval_days")
            == frozen["decision_interval_days"]
            and policy.get("maximum_evaluation_decisions")
            == frozen["maximum_evaluation_decisions"]
            and policy.get("structural_maximum_turnover")
            == frozen["structural_maximum_turnover"]
            and evaluation.get("costs_bps") == frozen["costs_bps"]
            and len(evaluation.get("mandatory_gates", []))
            == frozen["mandatory_financial_gate_count"]
            and training.get("folds") == frozen["folds"]
            and training.get("seeds") == frozen["seeds"]
        ),
        "maximum_decisions_remains_eight": (
            decisions == frozen["maximum_evaluation_decisions"] == 8
        ),
        "structural_turnover_remains_sixteen": (
            structural_turnover
            == frozen["structural_maximum_turnover"]
            == 16.0
        ),
    }
    evidence = {
        "old_last_maturity": old_maturity.isoformat(),
        "new_last_maturity": new_maturity.isoformat(),
        "old_signal_dates": old_count,
        "new_signal_dates": new_count,
        "maximum_decisions": decisions,
        "structural_maximum_turnover": structural_turnover,
        "changed_fields": changed_fields,
        "effective_chronology": effective,
    }
    return checks, evidence


def run_low_turnover_rank_chronology_erratum(
    config: dict[str, Any],
) -> dict[str, Any]:
    """Register the exact V82-R0 chronology overlay without scientific access."""

    section = config.get("low_turnover_rank_chronology_erratum")
    if not isinstance(section, Mapping):
        raise ValueError("Missing low_turnover_rank_chronology_erratum config")
    if section.get("family_id") != FAMILY_ID:
        raise ValueError("V82-R0 family drift")
    constraints = section.get("constraints")
    if not isinstance(constraints, Mapping) or constraints.get("metadata_only") is not True:
        raise ValueError("V82-R0 must remain metadata-only")
    if any(
        constraints.get(name) is not False
        for name in (
            "parquet_market_panel_or_raw_data_access_allowed",
            "outcome_or_target_value_access_allowed",
            "checkpoint_model_or_scaler_access_allowed",
            "training_or_inference_allowed",
            "prediction_position_metric_pnl_or_bootstrap_allowed",
            "target_asset_access_allowed",
            "v80_or_v81_artifact_rewrite_allowed",
            "target_architecture_objective_policy_cost_gate_or_turnover_change_allowed",
            "v82_dataset_execution_allowed",
        )
    ):
        raise ValueError("V82-R0 forbidden capability drift")

    root = Path(section.get("project_root", ".")).resolve()
    output = _inside(root, config["output_dir"], "output directory")
    payloads, observed_hashes = _verify_inputs(root, section)
    checks, evidence = _semantic_checks(payloads, section)
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"V82-R0 semantic gate failed: {failed}")

    erratum = {
        "schema_version": "v82-r0-chronology-erratum/v1",
        "family_id": FAMILY_ID,
        "evidence_tier": section["evidence_tier"],
        "immutable_base": {
            "specification_sha256": section["expected_registered_sha256"][
                "v80_specification"
            ],
            "blueprint_sha256": section["expected_registered_sha256"][
                "v80_blueprint"
            ],
            "harness_spec_sha256": section["expected_registered_sha256"][
                "v81_harness_spec"
            ],
        },
        "corrected_fields_exactly": {
            "final_evaluation_signal_end": {
                "before": "2026-06-09",
                "after": "2026-06-08",
            },
            "final_evaluation_signal_dates": {"before": 160, "after": 159},
        },
        "arithmetic_evidence": {
            key: evidence[key]
            for key in (
                "old_last_maturity",
                "new_last_maturity",
                "old_signal_dates",
                "new_signal_dates",
                "maximum_decisions",
                "structural_maximum_turnover",
                "changed_fields",
            )
        },
        "effective_chronology": evidence["effective_chronology"],
        "frozen_unchanged": list(
            payloads["authorization"]["frozen_unchanged"]
        ),
        "scientific_change_count": 0,
        "target_assets_status": "sealed",
        "target_assets_loaded": [],
        "v82_dataset_executed": False,
    }
    erratum["erratum_sha256"] = canonical_sha256(erratum)

    access_ledger = {
        "json_metadata_reads": 8,
        "parquet_deserializations": 0,
        "market_panel_or_raw_data_reads": 0,
        "outcome_or_target_value_reads": 0,
        "checkpoint_model_or_scaler_loads": 0,
        "model_instantiations": 0,
        "scaler_fits": 0,
        "optimizer_steps": 0,
        "training_runs": 0,
        "inference_runs": 0,
        "predictions_generated": 0,
        "positions_generated": 0,
        "performance_metrics_computed": 0,
        "pnl_evaluations": 0,
        "bootstrap_runs": 0,
        "target_assets_loaded": [],
    }
    audit = {
        "schema_version": "v82-r0-audit/v1",
        "passed": True,
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "access_ledger": access_ledger,
    }
    audit["audit_sha256"] = canonical_sha256(audit)

    result = {
        "schema_version": "v82-r0-result/v1",
        "decision": V82_DATASET_ACTION,
        "family_id": FAMILY_ID,
        "erratum_sha256": erratum["erratum_sha256"],
        "final_evaluation_signal_end": "2026-06-08",
        "final_evaluation_signal_dates": 159,
        "final_evaluation_outcome_maturity_end": "2026-06-30",
        "maximum_evaluation_decisions": 8,
        "structural_maximum_turnover": 16.0,
        "scientific_change_count": 0,
        "json_metadata_reads": 8,
        "outcome_rows_read": 0,
        "models_or_checkpoints_loaded": 0,
        "target_assets_loaded": [],
        "v82_dataset_executed": False,
        "deployable": False,
    }
    result["result_sha256"] = canonical_sha256(result)

    input_receipt = {
        "schema_version": "v82-r0-input-receipt/v1",
        "inputs": observed_hashes,
    }
    input_receipt["input_receipt_sha256"] = canonical_sha256(input_receipt)

    source_files = section.get("source_receipt_files")
    if not isinstance(source_files, list) or not source_files:
        raise ValueError("V82-R0 source receipt file list is required")
    source_hashes: dict[str, str] = {}
    for relative in source_files:
        if not isinstance(relative, str):
            raise ValueError("V82-R0 source receipt paths must be strings")
        source_hashes[relative] = file_sha256(
            _inside(root, relative, "source receipt file")
        )
    source_receipt = {
        "schema_version": "v82-r0-source-receipt/v1",
        "files": source_hashes,
    }
    source_receipt["source_receipt_sha256"] = canonical_sha256(source_receipt)

    report = "\n".join([
        "# V82-R0 metadata-only chronology erratum",
        "",
        "The immutable V80/V81 artifacts were not rewritten. This overlay",
        "changes only the final evaluation signal end from 2026-06-09 to",
        "2026-06-08 and its inclusive date count from 160 to 159.",
        "",
        "The corrected final signal plus the frozen 22-day maturity offset",
        "lands exactly on 2026-06-30. Eight decisions and the structural",
        "turnover ceiling of 16 remain unchanged.",
        "",
        "Eight hash-bound JSON metadata files were read. No Parquet, market",
        "panel, raw data, outcome, target value, checkpoint, model, scaler,",
        "training, inference, prediction, position, metric, PnL, bootstrap,",
        "or BTC/ETH/SOL access occurred.",
        "",
        "The packet authorizes only a separately registered V82 non-target",
        "dataset phase. That dataset was not implemented or executed here.",
        "",
    ])

    output.mkdir(parents=True, exist_ok=True)
    write_yaml_atomic(output / "resolved_config.yaml", config)
    write_json_atomic(output / "input_hash_receipt.json", input_receipt)
    write_json_atomic(output / "source_receipt.json", source_receipt)
    write_json_atomic(output / "chronology_erratum.json", erratum)
    write_json_atomic(output / "audit.json", audit)
    write_json_atomic(output / "result.json", result)
    _write_text_atomic(output / "report.md", report)

    manifest_files = [
        "resolved_config.yaml",
        "input_hash_receipt.json",
        "source_receipt.json",
        "chronology_erratum.json",
        "audit.json",
        "result.json",
        "report.md",
    ]
    manifest = {
        "schema_version": "v82-r0-artifact-manifest/v1",
        "files": {name: file_sha256(output / name) for name in manifest_files},
    }
    manifest["artifact_manifest_sha256"] = canonical_sha256(manifest)
    write_json_atomic(output / "artifact_manifest.json", manifest)
    return {
        "decision": result["decision"],
        "erratum": erratum,
        "result": result,
        "audit": audit,
        "artifact_manifest": manifest,
    }
