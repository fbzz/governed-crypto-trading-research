from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


TARGET_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def ranker_parameter_count(architecture: dict[str, Any]) -> int:
    d_model = int(architecture["d_model"])
    feed_forward = int(architecture["feed_forward_width"])
    patch_width = int(architecture["patch_length_days"]) * 9
    patch_count = (
        (int(architecture["lookback_days"]) - int(architecture["patch_length_days"]))
        // int(architecture["patch_stride_days"])
        + 1
    )
    layer_count = int(architecture["encoder_layers"]) + int(
        architecture["cross_asset_attention_layers"]
    )
    output_width = 2
    return int(
        patch_width * d_model
        + d_model
        + patch_count * d_model
        + d_model
        + layer_count
        * (
            4 * d_model * d_model
            + 2 * d_model * feed_forward
            + feed_forward
            + 9 * d_model
        )
        + 4 * d_model
        + d_model * output_width
        + output_width
        + d_model * patch_width
        + patch_width
    )


def state_gate_parameter_count(architecture: dict[str, Any]) -> int:
    d_model = int(architecture["d_model"])
    feed_forward = int(architecture["feed_forward_width"])
    patch_width = int(architecture["patch_length_days"]) * int(
        architecture["input_features"]
    )
    patch_count = (
        (int(architecture["lookback_days"]) - int(architecture["patch_length_days"]))
        // int(architecture["patch_stride_days"])
        + 1
    )
    layer_count = int(architecture["encoder_layers"])
    return int(
        patch_width * d_model
        + d_model
        + patch_count * d_model
        + layer_count
        * (
            4 * d_model * d_model
            + 2 * d_model * feed_forward
            + feed_forward
            + 9 * d_model
        )
        + 4 * d_model
        + d_model
        + 1
    )


def _registered_self_hash(value: dict[str, Any], key: str) -> bool:
    payload = dict(value)
    registered = payload.pop(key, None)
    return registered == _canonical_sha256(payload)


def run_decoupled_rank_state_spec(config: dict[str, Any]) -> dict[str, Any]:
    spec = config["decoupled_rank_state_spec"]
    root = Path(spec["project_root"]).resolve()
    paths = {name: root / value for name, value in spec["inputs"].items()}
    observed_before = {name: _sha256_file(path) for name, path in paths.items()}
    if observed_before != spec["expected_input_sha256"]:
        raise RuntimeError("V60 immutable metadata input drift")

    loaded = {name: _load_json(path) for name, path in paths.items()}
    authorization = loaded["user_authorization_blueprint"]
    authorization_audit = loaded["user_authorization_audit"]
    authorization_result = loaded["user_authorization_result"]
    v32_result = loaded["v32_result"]
    v32_audit = loaded["v32_audit"]
    manifest = loaded["v32_dataset_manifest"]
    feature_schema = loaded["v32_feature_schema"]
    asset_folds = loaded["v32_asset_folds"]
    triplet_catalog = loaded["v32_triplet_catalog"]
    v41_specification = loaded["v41_specification"]
    v41_blueprint = loaded["v41_blueprint"]
    v41_audit = loaded["v41_audit"]
    v44_result = loaded["v44_training_result"]
    v45_result = loaded["v45_result"]
    v45_gate = loaded["v45_gate_result"]
    v45_audit = loaded["v45_audit"]
    v46_attribution = loaded["v46_failure_attribution"]
    v54_attribution = loaded["v54_failure_attribution"]
    v59_attribution = loaded["v59_failure_attribution"]

    ranker_parameters = ranker_parameter_count(spec["ranker_architecture"])
    gate_parameters = state_gate_parameter_count(spec["state_gate_architecture"])
    total_parameters = ranker_parameters + gate_parameters
    source_blueprint = v41_specification["blueprint"]
    source_architecture = source_blueprint["architecture"]
    source_objective = source_blueprint["objective"]
    best_epochs = [int(row["best_epoch"]) for row in v44_result["checkpoint_manifest"]]
    selected_symbols = set(manifest["symbols"])
    gradient_contract = spec["objective"]["gradient_contract"]
    constraints = spec["constraints"]

    specification = {
        "schema_version": "v60-decoupled-rank-state-specification/v1",
        "version": spec["version"],
        "candidate_family_id": spec["candidate_family_id"],
        "state": "ex_ante_metadata_only_design_frozen_not_implemented_or_trained",
        "lineage": spec["lineage"],
        "target_contract": spec["target_contract"],
        "data_contract": spec["data_contract"],
        "ranker_architecture": spec["ranker_architecture"],
        "state_gate_architecture": spec["state_gate_architecture"],
        "capacity_contract": spec["capacity_contract"],
        "objective": spec["objective"],
        "policy": spec["policy"],
        "training_contract": spec["training_contract"],
        "evidence_contract": spec["evidence_contract"],
        "constraints": constraints,
        "authorized_next_action": spec["authorized_next_action"],
        "source_receipts": spec["expected_input_sha256"],
        "parameter_counts": {
            "ranker": ranker_parameters,
            "state_gate": gate_parameters,
            "total": total_parameters,
        },
        "registered_training_jobs": len(spec["training_contract"]["folds"])
        * len(spec["training_contract"]["seeds"]),
    }
    specification["specification_sha256"] = _canonical_sha256(specification)

    checks = {
        "all_metadata_hashes_match_before_read": observed_before
        == spec["expected_input_sha256"],
        "explicit_user_authorization_is_hash_valid": _registered_self_hash(
            authorization, "blueprint_sha256"
        )
        and _registered_self_hash(authorization_result, "result_sha256")
        and authorization_audit["passed"] is True,
        "authorization_is_exactly_v60_metadata_only": authorization["source_user_message"]
        == "Ok faça com a v45"
        and authorization["authorized_action"]
        == "execute_v60_metadata_only_decoupled_rank_state_family_specification"
        and authorization_result["decision"]
        == "execute_v60_metadata_only_decoupled_rank_state_family_specification",
        "v32_metadata_contract_passes": v32_audit["passed"] is True
        and v32_result["audit"]["passed"] is True,
        "exact_non_target_universe_preserved": len(selected_symbols) == 30
        and not TARGET_SYMBOLS.intersection(selected_symbols),
        "exact_source_receipts_preserved": manifest["panel_sha256"]
        == spec["data_contract"]["source_panel_sha256"]
        and manifest["sequence_index_sha256"]
        == spec["data_contract"]["source_sequence_index_sha256"],
        "exact_input_shape_and_features": spec["data_contract"]["input_shape"]
        == [None, 256, 3, 9]
        and len(feature_schema["model_feature_order"]) == 9,
        "three_asset_disjoint_folds_and_catalog_preserved": len(asset_folds["folds"])
        == 3
        and len(triplet_catalog["folds"]) == 3
        and all(
            not set(fold["train_symbols"]).intersection(fold["test_symbols"])
            for fold in asset_folds["folds"]
        ),
        "v41_source_spec_and_audit_are_valid": v41_audit["passed"] is True
        and _registered_self_hash(v41_blueprint, "blueprint_sha256")
        and v41_specification["decision"]
        == "authorize_v42_synthetic_ranking_excess_harness_only",
        "v45_ranker_architecture_is_preserved_exactly": {
            "lookback_days": source_architecture["lookback_days"],
            "patch_length_days": source_architecture["patch_length_days"],
            "patch_stride_days": source_architecture["patch_stride_days"],
            "d_model": source_architecture["d_model"],
            "encoder_layers": source_architecture["encoder_layers"],
            "cross_asset_attention_layers": source_architecture[
                "cross_asset_attention_layers"
            ],
            "attention_heads": source_architecture["attention_heads"],
            "feed_forward_width": source_architecture["feed_forward_width"],
            "dropout": source_architecture["dropout"],
            "shared_asset_encoder": source_architecture["shared_asset_encoder"],
            "asset_slot_embedding": source_architecture["asset_slot_embedding"],
            "prediction_heads": source_architecture["prediction_heads"],
        }
        == {key: value for key, value in spec["ranker_architecture"].items() if key not in {"variant_count", "expected_parameter_count"}},
        "v45_ranker_objective_is_preserved": source_objective["ranking_loss"]
        == "mean_softplus_negative_pair_sign_times_centered_score_difference"
        and source_objective["weights"] == spec["objective"]["ranker"]["weights"]
        and source_objective["early_stopping_monitor"]
        == spec["objective"]["ranker"]["early_stopping_monitor"],
        "v44_has_no_empirical_capacity_scaling_signal": len(best_epochs) == 9
        and max(best_epochs) <= 2
        and v44_result["summary"]["total_parameters"] == ranker_parameters,
        "v45_predictive_signal_and_retirement_are_preserved": v45_audit["passed"]
        is True
        and v45_result["decision"]
        == "retire_family_without_target_evaluation_or_parameter_tuning"
        and v45_gate["passed"] is False
        and sum(bool(cell["passed"]) for group in ("predictive_cells", "bootstrap_cells", "economic_cells") for cell in v45_gate[group])
        == 36
        and v45_gate["cell_count"] == 39,
        "v46_identifies_relative_absolute_conversion_gap": v46_attribution[
            "all_registered_predictive_gates_passed"
        ]
        and v46_attribution["relative_ranking_absolute_return_gap_observed"]
        and v46_attribution["fold_3_gross_return_was_negative"]
        and v46_attribution["costs_worsened_fold_3_but_did_not_create_its_gross_loss"]
        and v46_attribution["family_remains_retired"],
        "v54_supports_separate_calibration_problem": v54_attribution[
            "ranking_signal_survived"
        ]
        and v54_attribution["absolute_calibration_unstable"]
        and v54_attribution["structural_turnover_failure"],
        "v59_supports_ordinal_absolute_separation": v59_attribution["attribution"][
            "primary"
        ]
        == "weak_ordinal_signal_failed_policy_and_absolute_return_conversion"
        and "transaction_costs"
        in v59_attribution["attribution"]["not_supported_as_primary"],
        "new_family_does_not_reopen_v45": spec["lineage"]["family_is_new"] is True
        and spec["candidate_family_id"]
        != spec["lineage"]["scientific_parent_family"]
        and spec["lineage"]["parent_weights_reused"] is False
        and spec["lineage"]["parent_checkpoints_opened"] is False,
        "ranker_capacity_is_exact": ranker_parameters
        == spec["ranker_architecture"]["expected_parameter_count"],
        "independent_gate_capacity_is_exact": gate_parameters
        == spec["state_gate_architecture"]["expected_parameter_count"],
        "total_capacity_is_frozen_under_ceiling": total_parameters
        == spec["capacity_contract"]["expected_total_parameter_count"]
        and total_parameters <= spec["capacity_contract"]["parameter_ceiling"]
        and spec["capacity_contract"]["size_sweep_allowed"] is False
        and spec["capacity_contract"]["larger_model_allowed"] is False,
        "gradient_isolation_is_structural": spec["state_gate_architecture"][
            "independent_encoder"
        ]
        and spec["state_gate_architecture"]["ranker_representation_input"] == "none"
        and not any(
            [
                gradient_contract["shared_parameters"],
                gradient_contract["combined_scalar_loss"],
                gradient_contract["gate_gradients_enter_ranker"],
                gradient_contract["ranker_gradients_enter_gate"],
            ]
        ),
        "fresh_nine_job_training_is_preregistered_not_executed": spec[
            "training_contract"
        ]["expected_job_count"]
        == 9
        and specification["registered_training_jobs"] == 9
        and spec["training_contract"]["checkpoint_reuse"] == "none"
        and spec["training_contract"]["implementation_allowed_during_v60"] is False,
        "historical_evidence_is_adaptive_only": spec["evidence_contract"][
            "historical_windows_are_adaptive_development_only"
        ]
        and spec["training_contract"]["validation_role_is_clean_evidence"] is False
        and spec["evidence_contract"]["historical_outcome_reread_allowed"] is False,
        "targets_remain_sealed": spec["target_contract"]["status"] == "sealed"
        and spec["target_contract"]["target_data_allowed"] is False
        and spec["target_contract"]["target_prediction_count"] == 0
        and spec["target_contract"]["target_pnl_evaluation_count"] == 0,
        "v60_is_strictly_metadata_only": constraints["metadata_only"] is True
        and not any(value for key, value in constraints.items() if key != "metadata_only"),
        "v60_authorizes_only_v61_synthetic_harness": spec["authorized_next_action"]
        == "authorize_v61_synthetic_decoupled_rank_state_harness_only",
    }
    observed_after = {name: _sha256_file(path) for name, path in paths.items()}
    checks["all_metadata_hashes_match_after_read"] = observed_after == observed_before
    audit = {
        "schema_version": "v60-decoupled-rank-state-audit/v1",
        "passed": all(checks.values()),
        "checks": checks,
    }
    if not audit["passed"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"V60 specification audit failed: {failed}")

    blueprint = {
        "schema_version": "v60-decoupled-rank-state-blueprint/v1",
        "version": spec["version"],
        "candidate_family_id": spec["candidate_family_id"],
        "state": specification["state"],
        "scientific_parent": spec["lineage"],
        "architecture": {
            "ranker": spec["ranker_architecture"],
            "state_gate": spec["state_gate_architecture"],
            "gradient_contract": gradient_contract,
            "parameter_counts": specification["parameter_counts"],
        },
        "objective": spec["objective"],
        "policy": spec["policy"],
        "training_contract": spec["training_contract"],
        "evidence_contract": spec["evidence_contract"],
        "target_contract": spec["target_contract"],
        "authorized_next_action": spec["authorized_next_action"],
        "specification_sha256": specification["specification_sha256"],
    }
    blueprint["blueprint_sha256"] = _canonical_sha256(blueprint)
    result = {
        "schema_version": "v60-decoupled-rank-state-result/v1",
        "version": spec["version"],
        "decision": spec["authorized_next_action"],
        "family_id": spec["candidate_family_id"],
        "specification_sha256": specification["specification_sha256"],
        "blueprint_sha256": blueprint["blueprint_sha256"],
        "audit": audit,
        "input_hash_receipt": observed_after,
        "summary": {
            "ranker_parameters": ranker_parameters,
            "state_gate_parameters": gate_parameters,
            "total_parameters": total_parameters,
            "registered_training_jobs": specification["registered_training_jobs"],
            "parquet_deserializations": 0,
            "checkpoint_reads": 0,
            "model_instantiations": 0,
            "optimizer_steps": 0,
            "predictions": 0,
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
    _write_json(output / "result.json", result)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    report = "\n".join(
        [
            "# V60 Decoupled Rank-State Family Specification",
            "",
            f"Decision: **{result['decision']}**",
            "",
            f"Scientific parent: **{spec['lineage']['scientific_parent_family']} (V45)**",
            f"Ranker parameters: **{ranker_parameters:,}**",
            f"Independent state-gate parameters: **{gate_parameters:,}**",
            f"Total parameters: **{total_parameters:,}**",
            "",
            "V60 preserves the V45 ranker hypothesis and adds a separately trained",
            "absolute market-state gate with no shared parameters or gradient path.",
            "No Parquet, checkpoint, model, prediction, metric, PnL, outcome source,",
            "or target asset was opened. V61 may run only the synthetic harness.",
            "",
        ]
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    packet_files = [
        "audit.json",
        "blueprint.json",
        "input_hash_receipt.json",
        "report.md",
        "resolved_config.yaml",
        "result.json",
        "specification.json",
    ]
    artifact_manifest = {
        "schema_version": "v60-artifact-manifest/v1",
        "files": {name: _sha256_file(output / name) for name in packet_files},
    }
    artifact_manifest["artifact_manifest_sha256"] = _canonical_sha256(
        artifact_manifest
    )
    _write_json(output / "artifact_manifest.json", artifact_manifest)
    completion_receipt = {
        "schema_version": "v60-completion-receipt/v1",
        "decision": result["decision"],
        "family_id": spec["candidate_family_id"],
        "artifact_manifest_file_sha256": _sha256_file(
            output / "artifact_manifest.json"
        ),
        "artifact_manifest_sha256": artifact_manifest["artifact_manifest_sha256"],
        "result_sha256": result["result_sha256"],
        "audit_passed": audit["passed"],
    }
    completion_receipt["completion_receipt_sha256"] = _canonical_sha256(
        completion_receipt
    )
    _write_json(output / "completion_receipt.json", completion_receipt)
    return result
