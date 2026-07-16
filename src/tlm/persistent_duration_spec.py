from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


TARGET_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def persistent_duration_parameter_count(architecture: dict[str, Any]) -> int:
    """Analytic count without importing torch or instantiating the model."""

    d_model = int(architecture["d_model"])
    feed_forward = int(architecture["feed_forward_width"])
    input_features = int(architecture["input_features"])
    patch_length = int(architecture["patch_length_days"])
    patch_stride = int(architecture["patch_stride_days"])
    lookback = int(architecture["lookback_days"])
    temporal_layers = int(architecture["temporal_encoder_layers"])
    cross_asset_layers = int(architecture["cross_asset_attention_layers"])
    horizon_count = len(architecture["output_horizons"])
    duration_days = int(architecture["maximum_duration_days"])
    patch_width = patch_length * input_features
    patch_count = (lookback - patch_length) // patch_stride + 1
    transformer_layer = (
        4 * d_model * d_model
        + 2 * d_model * feed_forward
        + feed_forward
        + 9 * d_model
    )
    return int(
        patch_width * d_model
        + d_model
        + patch_count * d_model
        + 2 * d_model
        + (temporal_layers + cross_asset_layers) * transformer_layer
        + 4 * d_model
        + 2 * (2 * d_model * d_model + d_model)
        + 2 * d_model
        + 2 * (d_model * horizon_count + horizon_count)
        + d_model * (2 * horizon_count)
        + 2 * horizon_count
        + d_model * duration_days
        + duration_days
    )


def run_persistent_duration_spec(config: dict[str, Any]) -> dict[str, Any]:
    """Freeze the V74 family contract using metadata and source hashes only."""

    spec = config["persistent_duration_spec"]
    root = Path(spec["project_root"]).resolve()
    inputs = {name: root / value for name, value in spec["inputs"].items()}
    observed_before = {name: _sha256_file(path) for name, path in inputs.items()}
    if observed_before != spec["expected_input_sha256"]:
        raise RuntimeError("V74 immutable metadata/source input drift")

    loaded = {
        name: _load_json(inputs[name]) for name in spec["json_metadata_inputs"]
    }
    v73_result = loaded["v73_result"]
    v73_audit = loaded["v73_audit"]
    v73_record = loaded["v73_diagnostic_record"]
    v73_manifest = loaded["v73_artifact_manifest"]
    architecture = spec["architecture"]
    parameter_count = persistent_duration_parameter_count(architecture)
    training = spec["training_contract"]
    objective = spec["objective"]
    policy = spec["policy"]
    gates = spec["financial_evaluation_contract"]["mandatory_gates"]
    constraints = spec["constraints"]

    specification = {
        "schema_version": "v74-persistent-duration-specification/v1",
        "version": spec["version"],
        "candidate_family_id": spec["candidate_family_id"],
        "state": "ex_ante_metadata_only_design_frozen_not_trained_or_evaluated",
        "lineage": spec["lineage"],
        "target_contract": spec["target_contract"],
        "data_and_label_contract": spec["data_and_label_contract"],
        "architecture": architecture,
        "capacity_contract": spec["capacity_contract"],
        "objective": objective,
        "policy": policy,
        "training_contract": training,
        "financial_evaluation_contract": spec["financial_evaluation_contract"],
        "evidence_contract": spec["evidence_contract"],
        "constraints": constraints,
        "parameter_count": parameter_count,
        "registered_training_jobs": len(training["folds"]) * len(training["seeds"]),
        "source_receipts": spec["expected_input_sha256"],
        "authorized_next_action": spec["authorized_next_action"],
    }
    specification["specification_sha256"] = _canonical_sha256(specification)

    checks = {
        "all_metadata_and_source_hashes_match_before_read": observed_before
        == spec["expected_input_sha256"],
        "v73_authorizes_only_the_v74_specification": v73_result.get("decision")
        == "authorize_v74_persistent_duration_family_specification_only"
        and v73_result.get("family_status_changed") is False
        and v73_result.get("outcomes_opened") == 0
        and v73_result.get("models_or_checkpoints_loaded") == 0,
        "v73_record_is_valid_and_non_deployable": v73_audit.get("passed") is True
        and v73_record.get("diagnostic_outcome") == "fail"
        and v73_record.get("passed_gate_count") == 13
        and v73_record.get("mandatory_gate_count") == 24
        and v73_record.get("target_assets_status") == "sealed"
        and v73_record.get("deployable") is False,
        "v73_packet_manifest_is_frozen": v73_manifest.get("schema_version")
        == "v73-artifact-manifest/v1"
        and "result.json" in v73_manifest.get("files", {})
        and "audit.json" in v73_manifest.get("files", {}),
        "family_is_new_and_uses_no_prior_weights": spec["lineage"]["family_is_new"]
        is True
        and spec["lineage"]["checkpoint_reuse"] == "none"
        and spec["lineage"]["prior_family_weights_reused"] is False,
        "exact_input_and_patch_contract_is_frozen": architecture["input_shape"]
        == [None, 256, 3, 9]
        and architecture["patch_length_days"] == 16
        and architecture["patch_stride_days"] == 8
        and architecture["patch_count"] == 31
        and architecture["asset_slot_embedding"] is False,
        "exact_single_capacity_is_under_ceiling": parameter_count
        == architecture["expected_parameter_count"]
        == spec["capacity_contract"]["expected_total_parameter_count"]
        and parameter_count <= spec["capacity_contract"]["parameter_ceiling"]
        and spec["capacity_contract"]["variant_count"] == 1
        and spec["capacity_contract"]["size_sweep_allowed"] is False,
        "joint_objective_is_frozen": objective["weights"]
        == {"return_nll": 1.0, "pairwise_ranking": 0.25, "duration_nll": 0.5}
        and objective["pnl_loss"] is False
        and objective["outcome_weighting"] is False,
        "duration_label_and_maximum_maturity_are_frozen": spec[
            "data_and_label_contract"
        ]["duration_target"]
        == "earliest_argmax_day_of_cumulative_gross_open_to_open_log_return_days_1_through_7"
        and spec["data_and_label_contract"]["duration_right_censor_rule"]
        == "censored_when_earliest_argmax_is_day_7"
        and spec["data_and_label_contract"]["maximum_label_maturity_days"] == 8,
        "cost_conditioned_stateful_policy_is_frozen": policy["action_space"]
        == ["long_one_asset", "cash"]
        and policy["horizon_weights"] == [0.2, 0.3, 0.5]
        and policy["transition_turnover"]["switch"] == 2.0
        and policy["threshold_sweep_allowed"] is False
        and policy["leverage"] is False
        and policy["shorting"] is False,
        "nine_job_mps_grid_is_frozen_without_selection": training[
            "expected_job_count"
        ]
        == 9
        and specification["registered_training_jobs"] == 9
        and training["fold_selection_allowed"] is False
        and training["seed_selection_allowed"] is False
        and training["hyperparameter_search_allowed"] is False
        and training["device"] == "mps"
        and training["dtype"] == "float32",
        "financial_gates_are_strict_and_have_no_aggregate_rescue": gates[
            "aggregate_net_total_return_positive_at_cost_bps"
        ]
        == [10, 20, 30]
        and gates["each_fold_net_total_return_positive_at_cost_bps"] == [10]
        and gates["aggregate_sharpe_strictly_positive_at_cost_bps"] == [10]
        and gates["bootstrap_block_days"] == [7, 21, 63]
        and gates["aggregate_rescue_for_failed_fold"] is False,
        "targets_remain_sealed": spec["target_contract"]["status"] == "sealed"
        and set(spec["target_contract"]["target_symbols"]) == TARGET_SYMBOLS
        and spec["target_contract"]["target_data_allowed"] is False
        and spec["target_contract"]["target_prediction_count"] == 0
        and spec["target_contract"]["target_pnl_evaluation_count"] == 0,
        "v74_is_strictly_metadata_only": constraints["metadata_only"] is True
        and not any(value for key, value in constraints.items() if key != "metadata_only"),
        "v74_authorizes_only_v75_synthetic_harness": spec[
            "authorized_next_action"
        ]
        == "authorize_v75_synthetic_persistent_duration_harness_only",
    }
    observed_after = {name: _sha256_file(path) for name, path in inputs.items()}
    checks["all_metadata_and_source_hashes_match_after_read"] = (
        observed_after == observed_before
    )
    audit = {
        "schema_version": "v74-persistent-duration-audit/v1",
        "passed": all(checks.values()),
        "checks": checks,
    }
    if not audit["passed"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"V74 specification audit failed: {failed}")

    blueprint = {
        "schema_version": "v74-persistent-duration-blueprint/v1",
        "candidate_family_id": spec["candidate_family_id"],
        "architecture": architecture,
        "parameter_count": parameter_count,
        "data_and_label_contract": spec["data_and_label_contract"],
        "objective": objective,
        "policy": policy,
        "training_contract": training,
        "financial_evaluation_contract": spec["financial_evaluation_contract"],
        "target_contract": spec["target_contract"],
        "specification_sha256": specification["specification_sha256"],
        "authorized_next_action": spec["authorized_next_action"],
    }
    blueprint["blueprint_sha256"] = _canonical_sha256(blueprint)
    source_receipt = {
        "schema_version": "v74-source-receipt/v1",
        "files": {
            path: _sha256_file(root / path) for path in spec["source_receipt_files"]
        },
    }
    source_receipt["source_receipt_sha256"] = _canonical_sha256(source_receipt)
    result = {
        "schema_version": "v74-persistent-duration-result/v1",
        "decision": spec["authorized_next_action"],
        "family_id": spec["candidate_family_id"],
        "specification_sha256": specification["specification_sha256"],
        "blueprint_sha256": blueprint["blueprint_sha256"],
        "audit": audit,
        "summary": {
            "total_parameters": parameter_count,
            "registered_training_jobs": specification["registered_training_jobs"],
            "json_metadata_reads": len(spec["json_metadata_inputs"]),
            "parquet_deserializations": 0,
            "checkpoint_reads": 0,
            "model_instantiations": 0,
            "optimizer_steps": 0,
            "predictions": 0,
            "positions": 0,
            "performance_metrics": 0,
            "pnl_computations": 0,
            "outcome_source_reads": 0,
            "target_asset_rows": 0,
        },
    }
    result["result_sha256"] = _canonical_sha256(result)

    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "specification.json", specification)
    _write_json(output / "blueprint.json", blueprint)
    _write_json(output / "audit.json", audit)
    _write_json(output / "input_hash_receipt.json", observed_after)
    _write_json(output / "source_receipt.json", source_receipt)
    _write_json(output / "result.json", result)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output / "report.md").write_text(
        "\n".join(
            [
                "# V74 Persistent Multi-Horizon Duration Specification",
                "",
                f"Decision: **{result['decision']}**",
                f"Frozen parameters: **{parameter_count:,}**",
                "",
                "One fresh multi-asset model jointly forecasts 1/3/7-day returns,",
                "cross-asset rank, and an explicit 1..7-day duration distribution.",
                "The policy prices exact transition turnover before changing state.",
                "No data, checkpoint, model, prediction, outcome, PnL, or target asset",
                "was opened. Only the V75 synthetic harness is authorized next.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    packet_files = [
        "audit.json",
        "blueprint.json",
        "input_hash_receipt.json",
        "report.md",
        "resolved_config.yaml",
        "result.json",
        "source_receipt.json",
        "specification.json",
    ]
    artifact_manifest = {
        "schema_version": "v74-artifact-manifest/v1",
        "files": {name: _sha256_file(output / name) for name in packet_files},
    }
    artifact_manifest["artifact_manifest_sha256"] = _canonical_sha256(
        artifact_manifest
    )
    _write_json(output / "artifact_manifest.json", artifact_manifest)
    return result
