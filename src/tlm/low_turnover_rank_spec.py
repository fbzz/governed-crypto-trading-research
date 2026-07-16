from __future__ import annotations

import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping

from .core.artifacts import canonical_sha256, file_sha256, write_json_atomic, write_yaml_atomic


EXPECTED_INPUTS = {"result", "audit", "terminal_record", "input_hash_receipt"}
FAMILY_ID = "tlm_low_turnover_cross_sectional_rank_v1"
V80_ACTION = "authorize_v80_low_turnover_cross_sectional_rank_specification_only"
V81_ACTION = "authorize_v81_synthetic_low_turnover_rank_harness_only"
EXPECTED_FEATURES = [
    "log_open_to_open_return",
    "log_close_to_close_return",
    "log_high_low_range",
    "log_close_open_return",
    "log1p_quote_volume_change",
    "log1p_trade_count_change",
    "rolling_realized_volatility_7d",
    "rolling_realized_volatility_30d",
]


def _mapping(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing V80 {label}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V80 {label} must be a JSON object")
    return value


def _inside(root: Path, relative: str, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"V80 {label} escapes the repository") from exc
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
        raise ValueError("V80 input paths and hashes must be mappings")
    if set(inputs) != EXPECTED_INPUTS or set(expected_hashes) != EXPECTED_INPUTS:
        raise ValueError("V80 metadata input allowlist drift")

    payloads: dict[str, dict[str, Any]] = {}
    observed_hashes: dict[str, str] = {}
    for name in sorted(EXPECTED_INPUTS):
        relative = inputs[name]
        expected = expected_hashes[name]
        if not isinstance(relative, str) or not relative.endswith(".json"):
            raise ValueError("V80 may read only registered JSON metadata")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"Invalid V80 hash for {name}")
        path = _inside(root, relative, f"input {name}")
        observed = file_sha256(path)
        if observed != expected:
            raise ValueError(f"V80 input hash drift: {name}")
        payloads[name] = _mapping(path, name)
        observed_hashes[name] = observed
    return payloads, observed_hashes


def _v79_checks(payloads: Mapping[str, dict[str, Any]]) -> dict[str, bool]:
    result = payloads["result"]
    audit = payloads["audit"]
    record = payloads["terminal_record"]
    receipt = payloads["input_hash_receipt"]
    ledger = audit.get("access_ledger", {})
    numeric_zero_keys = {
        "parquet_deserializations",
        "market_panel_reads",
        "outcome_packet_reads",
        "checkpoint_loads",
        "model_instantiations",
        "scaler_fits",
        "optimizer_steps",
        "training_runs",
        "inference_runs",
        "predictions_generated",
        "positions_generated",
        "scientific_metrics_recomputed",
        "pnl_evaluations",
    }
    return {
        "v79_authorization_is_exact": (
            result.get("decision") == V80_ACTION
            and result.get("successor_family_id") == FAMILY_ID
            and result.get("successor_specification_executed") is False
        ),
        "v78_family_is_terminal_and_retired": (
            result.get("retired_family_id")
            == "tlm_persistent_multi_horizon_duration_v1"
            and result.get("retired_family_status") == "retired"
            and record.get("family_status_after") == "retired"
            and record.get("terminal_phase") == "v78"
        ),
        "v79_was_metadata_only": (
            result.get("json_metadata_reads") == 4
            and result.get("outcome_rows_read") == 0
            and result.get("scientific_metrics_recomputed") == 0
            and result.get("models_or_checkpoints_loaded") == 0
            and result.get("target_assets_loaded") == []
            and audit.get("passed") is True
            and ledger.get("json_metadata_reads") == 4
            and all(ledger.get(key) == 0 for key in numeric_zero_keys)
            and ledger.get("target_assets_loaded") == []
        ),
        "v79_input_receipt_is_metadata_only": (
            receipt.get("schema_version") == "v79-input-receipt/v1"
            and isinstance(receipt.get("inputs"), Mapping)
            and len(receipt.get("inputs", {})) == 4
            and all(
                isinstance(digest, str) and len(digest) == 64
                for digest in receipt.get("inputs", {}).values()
            )
        ),
        "targets_remained_sealed": (
            record.get("target_assets_status") == "sealed"
            and record.get("target_assets_loaded") == []
            and record.get("successor_family_id") == FAMILY_ID
        ),
    }


def _parameter_count(design: Mapping[str, Any]) -> int:
    architecture = design["architecture"]
    feature_count = design["input"]["feature_count"]
    channels = architecture["temporal_channels"]
    blocks = architecture["temporal_blocks"]
    kernel = architecture["kernel_size"]
    head_input, head_hidden, head_output = architecture["rank_head"]
    input_layer_norm = 2 * feature_count
    input_projection = feature_count * channels + channels
    depthwise = channels * kernel + channels
    pointwise = channels * channels + channels
    block_layer_norm = 2 * channels
    temporal = blocks * (depthwise + pointwise + block_layer_norm)
    head = head_input * head_hidden + head_hidden
    head += 2 * head_hidden
    head += head_hidden * head_output + head_output
    return input_layer_norm + input_projection + temporal + head


def _design_checks(section: Mapping[str, Any]) -> dict[str, bool]:
    constraints = section.get("constraints", {})
    design = section.get("frozen_design", {})
    input_spec = design.get("input", {})
    architecture = design.get("architecture", {})
    target = design.get("target", {})
    objective = design.get("objective", {})
    policy = design.get("policy", {})
    universe = design.get("universe", {})
    chronology = design.get("chronology", {})
    training = design.get("training", {})
    evaluation = design.get("evaluation", {})
    terminal = design.get("terminal_decision", {})
    signal_dates = chronology.get("final_evaluation_signal_dates")
    interval = policy.get("decision_interval_days")
    decision_count = (
        math.ceil(signal_dates / interval)
        if isinstance(signal_dates, int) and isinstance(interval, int) and interval > 0
        else -1
    )
    parameter_count = _parameter_count(design) if design else -1
    return {
        "new_single_final_family_without_state_reuse": (
            section.get("family_id") == FAMILY_ID
            and constraints.get("family_is_new") is True
            and constraints.get("single_final_family") is True
            and constraints.get("prior_checkpoint_weight_scaler_or_optimizer_reuse_allowed")
            is False
            and universe.get("prior_checkpoint_or_scaler_reuse") is False
        ),
        "exact_causal_compact_architecture": (
            input_spec.get("lookback_days") == 128
            and input_spec.get("assets_per_context") == 3
            and input_spec.get("features") == EXPECTED_FEATURES
            and architecture.get("family")
            == "shared_causal_depthwise_tcn_deepsets_ranker"
            and architecture.get("dilations") == [1, 2, 4, 8, 16, 32]
            and architecture.get("receptive_field_days") == 127
            and architecture.get("architecture_variant_count") == 1
        ),
        "parameter_accounting_is_exact": (
            parameter_count == 10993
            and architecture.get("expected_total_parameters") == parameter_count
            and architecture.get("rank_head") == [96, 32, 1]
        ),
        "relative_rank_target_and_loss_are_exact": (
            target.get("execution_open") == "t_plus_1"
            and target.get("exit_open") == "t_plus_22"
            and target.get("horizon_intervals") == 21
            and target.get("excess_return")
            == "asset_return_minus_triplet_mean_asset_return"
            and objective.get("point_loss")
            == "smooth_l1_centered_score_to_scaled_excess_beta_1"
            and objective.get("pairwise_weight") == 0.50
            and objective.get("pnl_loss_weight") == 0.0
            and objective.get("turnover_loss_weight") == 0.0
            and objective.get("auxiliary_absolute_or_state_head") is False
        ),
        "turnover_is_structurally_bounded": (
            interval == 21
            and decision_count == 8
            and policy.get("maximum_evaluation_decisions") == decision_count
            and policy.get("structural_maximum_turnover") == 2.0 * decision_count
            and policy.get("structural_maximum_turnover") == 16.0
            and policy.get("threshold_or_interval_tuning_allowed") is False
            and policy.get("transition_turnover")
            == {"enter": 1.0, "exit": 1.0, "hold": 0.0, "switch": 2.0}
            and policy.get("final_liquidation") is True
        ),
        "chronology_excludes_consumed_2025_outcomes": (
            chronology.get("training_signal_end") == "2023-11-18"
            and chronology.get("internal_validation_signal_start") == "2024-01-01"
            and chronology.get("internal_validation_signal_end") == "2024-11-18"
            and chronology.get("consumed_2025_outcomes_role") == "forbidden"
            and chronology.get("final_evaluation_signal_start") == "2026-01-01"
            and chronology.get("final_evaluation_signal_end") == "2026-06-09"
            and chronology.get("final_evaluation_outcome_maturity_end")
            == "2026-06-30"
        ),
        "training_grid_is_single_and_frozen": (
            training.get("device") == "mps"
            and training.get("mps_fallback_allowed") is False
            and training.get("folds") == [1, 2, 3]
            and training.get("seeds") == [42, 7, 123]
            and training.get("future_job_count") == 9
            and training.get("fresh_initialization_only") is True
            and training.get("model_or_hyperparameter_selection") is False
        ),
        "evaluation_and_kill_criteria_are_frozen": (
            evaluation.get("costs_bps") == [10, 20, 30]
            and len(evaluation.get("mandatory_gates", [])) == 9
            and evaluation.get("aggregate_rescue_for_failed_fold") is False
            and evaluation.get("missing_cell_pass_allowed") is False
            and terminal.get("any_mandatory_gate_failure")
            == "retire_final_family_without_target_evaluation_or_retuning"
            and terminal.get("all_mandatory_gates_pass")
            == "retain_family_and_authorize_separate_target_transfer_specification_only"
            and terminal.get("deployable_from_non_target_result") is False
            and terminal.get("second_variant_or_rescue_allowed") is False
        ),
        "no_sweep_data_model_or_outcome_access_in_v80": (
            constraints.get("single_frozen_variant_required") is True
            and constraints.get("parameter_increase_as_default_direction") is False
            and constraints.get("hyperparameter_or_threshold_sweep_allowed") is False
            and constraints.get(
                "data_checkpoint_model_training_inference_prediction_or_position_allowed"
            )
            is False
            and constraints.get("outcome_metric_pnl_or_target_access_allowed") is False
            and constraints.get("target_assets_status") == "sealed"
        ),
    }


def run_low_turnover_rank_spec(config: dict[str, Any]) -> dict[str, Any]:
    """Freeze the final-family specification using metadata only."""

    section = config.get("low_turnover_rank_spec")
    if not isinstance(section, Mapping):
        raise ValueError("Missing low_turnover_rank_spec config section")
    root = Path(section.get("project_root", ".")).resolve()
    output_value = config.get("output_dir")
    if not isinstance(output_value, str):
        raise ValueError("V80 output_dir must be a repository-relative path")
    output = _inside(root, output_value, "output directory")
    payloads, input_hashes = _verify_inputs(root, section)

    checks = {**_v79_checks(payloads), **_design_checks(section)}
    if not all(checks.values()):
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"V80 specification gate failed: {failed}")

    design = section["frozen_design"]
    hypothesis = {
        "primary": (
            "A compact permutation-equivariant 21-day excess-rank model paired "
            "with a fixed 21-day decision clock can preserve relative signal "
            "while bounding evaluation turnover by construction."
        ),
        "falsifier": (
            "Any mandatory behavior or financial gate failure retires the final "
            "family without target evaluation, retuning, or a second variant."
        ),
    }
    specification = {
        "schema_version": "v80-low-turnover-rank-specification/v1",
        "family_id": FAMILY_ID,
        "evidence_tier": section["evidence_tier"],
        "hypothesis": hypothesis,
        "design": design,
        "parameter_count": 10993,
        "future_training_jobs": 9,
        "structural_maximum_evaluation_turnover": 16.0,
        "consumed_2025_outcomes_allowed": False,
        "target_assets_status": "sealed",
        "deployable": False,
    }
    specification["specification_sha256"] = canonical_sha256(specification)
    blueprint = {
        "schema_version": "v80-low-turnover-rank-blueprint/v1",
        "family_id": FAMILY_ID,
        "input": design["input"],
        "architecture": design["architecture"],
        "target": design["target"],
        "objective": design["objective"],
        "policy": design["policy"],
        "chronology": design["chronology"],
        "training": design["training"],
        "evaluation": design["evaluation"],
        "terminal_decision": design["terminal_decision"],
    }
    blueprint["blueprint_sha256"] = canonical_sha256(blueprint)
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
        "performance_metrics_computed": 0,
        "pnl_evaluations": 0,
        "target_assets_loaded": [],
    }
    audit = {
        "schema_version": "v80-audit/v1",
        "passed": True,
        "checks": checks,
        "checks_passed": len(checks),
        "checks_total": len(checks),
        "access_ledger": access_ledger,
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    result = {
        "schema_version": "v80-result/v1",
        "decision": V81_ACTION,
        "family_id": FAMILY_ID,
        "specification_sha256": specification["specification_sha256"],
        "blueprint_sha256": blueprint["blueprint_sha256"],
        "parameter_count": 10993,
        "future_training_jobs": 9,
        "structural_maximum_evaluation_turnover": 16.0,
        "audit_checks_passed": len(checks),
        "v81_executed": False,
        "scientific_data_reads": 0,
        "models_or_checkpoints_loaded": 0,
        "outcome_rows_read": 0,
        "target_assets_loaded": [],
        "deployable": False,
    }
    result["result_sha256"] = canonical_sha256(result)
    input_receipt = {
        "schema_version": "v80-input-receipt/v1",
        "inputs": input_hashes,
    }
    input_receipt["input_receipt_sha256"] = canonical_sha256(input_receipt)

    source_files = section.get("source_receipt_files")
    if not isinstance(source_files, list) or not source_files:
        raise ValueError("V80 source receipt file list is required")
    source_hashes: dict[str, str] = {}
    for relative in source_files:
        if not isinstance(relative, str):
            raise ValueError("V80 source receipt paths must be strings")
        source_hashes[relative] = file_sha256(_inside(root, relative, "source file"))
    source_receipt = {
        "schema_version": "v80-source-receipt/v1",
        "files": source_hashes,
    }
    source_receipt["source_receipt_sha256"] = canonical_sha256(source_receipt)

    report = "\n".join([
        "# V80 final low-turnover rank specification",
        "",
        "The final family is a 10,993-parameter shared causal depthwise TCN with",
        "DeepSets-style cross-asset context and one centered 21-day excess-rank",
        "score per asset. It has no learned absolute-return or state-gate head.",
        "",
        "The policy decides only every 21 eligible signal dates. Across the frozen",
        "160-date 2026 non-target evaluation window this permits eight decisions",
        "and at most 16.0 turnover including final liquidation, by construction.",
        "",
        "V80 read four hash-bound V79 JSON metadata files and performed no data,",
        "model, training, inference, prediction, position, metric/PnL, outcome, or",
        "target operation. It authorizes only the separate V81 synthetic harness.",
        "",
    ])

    output.mkdir(parents=True, exist_ok=True)
    write_yaml_atomic(output / "resolved_config.yaml", config)
    write_json_atomic(output / "input_hash_receipt.json", input_receipt)
    write_json_atomic(output / "source_receipt.json", source_receipt)
    write_json_atomic(output / "specification.json", specification)
    write_json_atomic(output / "blueprint.json", blueprint)
    write_json_atomic(output / "audit.json", audit)
    write_json_atomic(output / "result.json", result)
    _write_text_atomic(output / "report.md", report)

    manifest_files = [
        "resolved_config.yaml",
        "input_hash_receipt.json",
        "source_receipt.json",
        "specification.json",
        "blueprint.json",
        "audit.json",
        "result.json",
        "report.md",
    ]
    manifest = {
        "schema_version": "v80-artifact-manifest/v1",
        "files": {name: file_sha256(output / name) for name in manifest_files},
    }
    manifest["artifact_manifest_sha256"] = canonical_sha256(manifest)
    write_json_atomic(output / "artifact_manifest.json", manifest)
    return {
        "decision": result["decision"],
        "specification": specification,
        "blueprint": blueprint,
        "audit": audit,
        "result": result,
        "artifact_manifest": manifest,
    }
