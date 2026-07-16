from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import yaml


TARGET_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}


def _load_json(path: Path) -> dict:
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
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def analytic_joint_parameter_count(architecture: dict, input_features: int) -> int:
    """Count the V47 model without instantiating torch or touching checkpoints."""
    d_model = int(architecture["d_model"])
    feed_forward = int(architecture["feed_forward_width"])
    patch_width = int(architecture["patch_length_days"]) * int(input_features)
    patch_count = (
        (
            int(architecture["lookback_days"])
            - int(architecture["patch_length_days"])
        )
        // int(architecture["patch_stride_days"])
        + 1
    )
    encoder_layers = int(architecture["encoder_layers"]) + int(
        architecture["cross_asset_attention_layers"]
    )
    prediction_heads = len(architecture["prediction_heads"])
    patch_projection = patch_width * d_model + d_model
    temporal_position = patch_count * d_model
    transformer_layer = (
        4 * d_model * d_model
        + 2 * d_model * feed_forward
        + feed_forward
        + 9 * d_model
    )
    final_norms = 4 * d_model
    output_heads = prediction_heads * (d_model + 1)
    return int(
        patch_projection
        + temporal_position
        + encoder_layers * transformer_layer
        + final_norms
        + output_heads
    )


def _fold_contract_checks(asset_folds: dict, triplet_catalog: dict) -> dict[str, bool]:
    folds = asset_folds["folds"]
    catalogs = {int(item["fold"]): item for item in triplet_catalog["folds"]}
    return {
        "exact_three_asset_folds": asset_folds["fold_count"] == 3
        and len(folds) == 3
        and set(catalogs) == {1, 2, 3},
        "each_fold_has_twenty_train_and_ten_test_assets": all(
            len(fold["train_symbols"]) == 20
            and len(fold["test_symbols"]) == 10
            for fold in folds
        ),
        "train_and_test_assets_are_disjoint": all(
            not set(fold["train_symbols"]).intersection(fold["test_symbols"])
            for fold in folds
        ),
        "catalog_symbols_match_asset_folds": all(
            catalogs[int(fold["fold"])]["train_symbols"]
            == fold["train_symbols"]
            and catalogs[int(fold["fold"])]["test_symbols"]
            == fold["test_symbols"]
            for fold in folds
        ),
        "catalog_triplet_counts_are_exact": all(
            len(catalogs[int(fold["fold"])]["train_triplets"])
            == math.comb(20, 3)
            and len(catalogs[int(fold["fold"])]["test_triplets"])
            == math.comb(10, 3)
            for fold in folds
        ),
    }


def build_joint_absolute_relative_spec(config: dict) -> dict[str, object]:
    spec = config["joint_absolute_relative_spec"]
    root = Path(spec["project_root"]).resolve()
    paths = {name: root / relative for name, relative in spec["inputs"].items()}
    input_checks = {
        name: path.is_file()
        and _sha256_file(path) == spec["expected_input_sha256"][name]
        for name, path in paths.items()
    }
    if not all(input_checks.values()):
        raise RuntimeError(f"V47 input missing or hash drifted: {input_checks}")

    v32_result = _load_json(paths["v32_result"])
    v32_audit = _load_json(paths["v32_audit"])
    dataset_manifest = _load_json(paths["v32_dataset_manifest"])
    feature_schema = _load_json(paths["v32_feature_schema"])
    asset_folds = _load_json(paths["v32_asset_folds"])
    triplet_catalog = _load_json(paths["v32_triplet_catalog"])
    v45_result = _load_json(paths["v45_result"])
    v45_gate = _load_json(paths["v45_gate"])
    v45_audit = _load_json(paths["v45_audit"])
    failure_attribution = _load_json(paths["v46_failure_attribution"])
    v46_audit = _load_json(paths["v46_audit"])
    v46_receipt = _load_json(paths["v46_completion_receipt"])
    v46_autopsy_spec = _load_json(paths["v46_autopsy_spec"])

    architecture = dict(spec["architecture"])
    feature_order = feature_schema["model_feature_order"]
    parameter_count = analytic_joint_parameter_count(
        architecture, len(feature_order)
    )
    training = spec["training"]
    job_count = (
        len(training["origins"])
        * len(training["geometries"])
        * len(training["folds"])
        * len(training["seeds"])
    )
    blueprint = {
        "version": spec["version"],
        "candidate_family_id": spec["candidate_family_id"],
        "state": "ex_ante_design_frozen_not_implemented_or_trained",
        "lineage": spec["lineage"],
        "target_contract": spec["target_contract"],
        "data_contract": spec["data_contract"],
        "architecture": architecture,
        "objective": spec["objective"],
        "early_stopping": spec["early_stopping"],
        "training": training,
        "policy": spec["policy"],
        "later_evaluation": spec["later_evaluation"],
        "lifecycle": spec["lifecycle"],
        "source_dataset_contract": {
            "dataset_manifest_sha256": spec["expected_input_sha256"][
                "v32_dataset_manifest"
            ],
            "feature_schema_sha256": spec["expected_input_sha256"][
                "v32_feature_schema"
            ],
            "asset_folds_sha256": spec["expected_input_sha256"][
                "v32_asset_folds"
            ],
            "triplet_catalog_sha256": spec["expected_input_sha256"][
                "v32_triplet_catalog"
            ],
            "panel_sha256": dataset_manifest["panel_sha256"],
            "sequence_index_sha256": dataset_manifest["sequence_index_sha256"],
            "symbol_count": dataset_manifest["symbol_count"],
            "feature_count": len(feature_order),
            "panel_or_label_loaded": False,
        },
        "parameter_count_analytic": parameter_count,
        "registered_job_count": job_count,
    }
    blueprint["blueprint_sha256"] = _canonical_sha256(blueprint)

    fold_checks = _fold_contract_checks(asset_folds, triplet_catalog)
    selected_symbols = set(dataset_manifest["symbols"])
    objective = spec["objective"]
    early = spec["early_stopping"]
    policy = spec["policy"]
    later = spec["later_evaluation"]
    origin_ids = [item["id"] for item in training["origins"]]
    checks = {
        "all_frozen_input_hashes_match": all(input_checks.values()),
        "v32_dataset_contract_passes": bool(v32_audit["passed"])
        and bool(v32_result["audit"]["passed"])
        and v32_result["decision"]
        == "authorize_v33_patch_transformer_implementation_only",
        "exact_v32_artifact_hashes_reused": dataset_manifest["panel_sha256"]
        == spec["data_contract"]["panel_sha256"]
        and dataset_manifest["sequence_index_sha256"]
        == spec["data_contract"]["sequence_index_sha256"],
        "exact_tensor_and_feature_contract_reused": dataset_manifest[
            "tensor_contract"
        ]["x_shape"]
        == [256, 3, 9]
        and spec["data_contract"]["input_shape"] == [None, 256, 3, 9]
        and len(feature_order) == 9
        and feature_order[-1] == "within_triplet_relative_strength",
        "return_label_is_the_only_numeric_target": spec["data_contract"][
            "permitted_label_columns"
        ]
        == ["target_window_end_date", "target_next_open_to_next_open_log_return"]
        and spec["data_contract"]["forbidden_label_columns"]
        == ["target_realized_volatility_7d"],
        "exact_thirty_non_target_symbols_reused": len(selected_symbols) == 30
        and not TARGET_SYMBOLS.intersection(selected_symbols),
        "target_assets_remain_sealed": spec["target_contract"][
            "target_data_allowed"
        ]
        is False
        and spec["target_contract"]["target_prediction_count"] == 0
        and spec["target_contract"]["target_pnl_evaluation_count"] == 0,
        "v45_retirement_is_immutable": bool(v45_audit["passed"])
        and not bool(v45_gate["passed"])
        and v45_result["decision"] == spec["lineage"]["retired_parent_decision"],
        "v46_diagnostic_requires_joint_absolute_relative_family": bool(
            v46_audit["passed"]
        )
        and v46_receipt["decision"]
        == spec["lineage"]["v46_completion_decision"]
        and failure_attribution["family_remains_retired"]
        and failure_attribution["relative_ranking_absolute_return_gap_observed"]
        and v46_autopsy_spec["constraints"]["v45_decision_mutable"] is False,
        "new_family_has_no_parent_weights": spec["lineage"]["family_is_new"]
        and spec["lineage"]["previous_checkpoint_reuse_allowed"] is False
        and spec["lineage"]["masked_pretraining_allowed"] is False,
        "exactly_one_registered_architecture": architecture["variant_count"] == 1
        and architecture["d_model"] == 128
        and architecture["encoder_layers"] == 4
        and architecture["cross_asset_attention_layers"] == 2
        and architecture["attention_heads"] == 4
        and architecture["feed_forward_width"] == 512,
        "architecture_is_permutation_equivariant": architecture[
            "shared_asset_encoder"
        ]
        and architecture["asset_slot_embedding"] is False,
        "heads_and_removed_components_are_exact": architecture[
            "prediction_heads"
        ]
        == ["excess_score_z", "market_component_z"]
        and architecture["mask_token"] is False
        and architecture["reconstruction_head"] is False,
        "analytic_parameter_count_matches_and_is_below_v41": parameter_count
        == architecture["expected_parameter_count_for_nine_features"]
        and parameter_count < architecture["parameter_ceiling"],
        "objective_decomposes_absolute_and_relative_return": objective[
            "market_target"
        ]
        == "m_equals_triplet_mean_r"
        and objective["excess_target"] == "e_i_equals_r_i_minus_m"
        and objective["absolute_prediction"]
        == "mu_hat_z_equals_m_hat_z_plus_e_hat_z",
        "objective_loss_is_exact": objective["weights"]
        == {"ranking": 1.0, "excess": 1.0, "level": 1.0}
        and objective["pairs"] == [[0, 1], [0, 2], [1, 2]]
        and objective["outcome_weighting"] is False
        and objective["target_clipping"] is False
        and objective["pnl_loss"] is False,
        "raw_return_scale_is_train_only": objective["scale_estimator"]
        == "sqrt_mean_squared_raw_return_over_complete_train_enumeration"
        and objective["scale_floor"] == 0.000001
        and "train" in objective["scale_scope"],
        "early_stopping_is_fully_frozen": early["monitor"]
        == "validation_total_loss"
        and early["first_completed_epoch_initializes_best"]
        and early["comparator"] == "strictly_lower"
        and early["min_delta"] == 0.0
        and early["equal_is_non_improvement"]
        and early["patience_consecutive_non_improvements"] == 5
        and early["maximum_epochs"] == 30
        and early["restore_job_local_best"]
        and early["cross_job_selection_allowed"] is False,
        "walk_forward_grid_is_exactly_thirty_six_jobs": origin_ids
        == ["origin_2024", "origin_2025"]
        and training["geometries"] == ["expanding", "rolling"]
        and training["folds"] == [1, 2, 3]
        and training["seeds"] == [42, 7, 123]
        and job_count == 36
        and job_count == training["expected_job_count"],
        "all_jobs_start_fresh_without_selection": training["initialization"]
        == "fresh_registered_seed"
        and training["pretraining"] == "none"
        and not any(
            training[key]
            for key in (
                "seed_selection_allowed",
                "fold_selection_allowed",
                "origin_selection_allowed",
                "geometry_selection_allowed",
            )
        ),
        "mps_training_contract_is_exact": training["device"] == "mps"
        and training["dtype"] == "float32"
        and training["amp"] is False
        and training["cpu_fallback_allowed"] is False
        and training["train_samples_per_epoch"] == 8192
        and training["fixed_validation_samples"] == 2048
        and training["batch_size"] == 128,
        "policy_is_cost_aware_one_third_long_or_cash": policy["action_space"]
        == ["cash", "long_one_asset"]
        and math.isclose(policy["risky_gross"], 1.0 / 3.0)
        and policy["base_cost_bps"] == 10
        and policy["momentum_gate"] == "none"
        and policy["execution_during_v47_v49"] is False,
        "later_evaluation_is_frozen_but_disabled": later[
            "execution_during_v47_v49"
        ]
        is False
        and later["cross_context_asset_averaging_allowed"] is False
        and later["bootstrap_paths"] == 10_000
        and later["bootstrap_blocks"] == [7, 21, 63]
        and later["finalization_refit_allowed"] is False,
        "v47_is_metadata_only": all(
            value is False for value in spec["constraints"].values()
        ),
        "only_v48_is_authorized": spec["authorized_next_action"]
        == "authorize_v48_joint_absolute_relative_synthetic_harness_only",
        **fold_checks,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V47 specification audit failed: {checks}")

    input_hashes_after = {name: _sha256_file(path) for name, path in paths.items()}
    if input_hashes_after != spec["expected_input_sha256"]:
        raise RuntimeError("V47 inputs changed while specification was built")

    return {
        "version": "v47",
        "method": "metadata_only_joint_absolute_relative_triplet_specification",
        "decision": "authorize_v48_joint_absolute_relative_synthetic_harness_only",
        "blueprint": blueprint,
        "blueprint_sha256": blueprint["blueprint_sha256"],
        "tested": {
            "json_metadata_reads": len(paths),
            "model_instantiations": 0,
            "parquet_deserializations": 0,
            "panel_or_label_reads": 0,
            "checkpoint_tensor_reads": 0,
            "optimizer_steps": 0,
            "predictions": 0,
            "performance_metrics": 0,
            "pnl_evaluations": 0,
            "target_asset_loads": 0,
            "improvement_status": "unknown_not_evaluated",
        },
        "source_hashes": input_hashes_after,
        "audit": {"passed": True, "checks": checks},
    }


def _report(result: dict[str, object]) -> str:
    blueprint = result["blueprint"]
    return "\n".join(
        [
            "# TLM v47 Joint Absolute/Relative Triplet Specification",
            "",
            "## Decision",
            "",
            "**SPECIFICATION PASSED; ONLY THE SYNTHETIC V48 HARNESS IS AUTHORIZED.**",
            "",
            f"Family: `{blueprint['candidate_family_id']}`",
            f"Blueprint SHA-256: `{result['blueprint_sha256']}`",
            f"Analytic parameters: **{blueprint['parameter_count_analytic']:,}**",
            f"Registered V49 jobs: **{blueprint['registered_job_count']}**",
            "",
            "No model, Parquet, label, checkpoint tensor, prediction, performance metric, PnL, or target asset was loaded or produced.",
            "",
            "## Frozen change",
            "",
            "One exact triplet now predicts centered relative excess and a shared absolute market component, reconstructing each asset forecast as mu = m + e.",
            "",
            "## Training boundary",
            "",
            "V49 is pre-registered as 2 origins x 2 geometries x 3 folds x 3 seeds, all fresh and all retained. V47 does not authorize that data access yet.",
            "",
            "## Next action",
            "",
            "V48 may instantiate the exact model and verify losses, symmetry, policy accounting, checkpointing, and resume behavior on deterministic synthetic data only.",
            "",
        ]
    )


def run_joint_absolute_relative_spec(config: dict) -> dict[str, object]:
    result = build_joint_absolute_relative_spec(config)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "result.json", result)
    _write_json(output / "specification.json", result)
    _write_json(output / "blueprint.json", result["blueprint"])
    _write_json(output / "audit.json", result["audit"])
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    return result
