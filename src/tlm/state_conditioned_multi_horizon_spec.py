from __future__ import annotations

import hashlib
import json
import math
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


def analytic_parameter_count(architecture: dict[str, Any]) -> int:
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
    layer_count = int(architecture["temporal_encoder_layers"]) + int(
        architecture["cross_asset_attention_layers"]
    )
    head_width = len(architecture["output_horizons"]) * len(
        architecture["output_quantiles"]
    )
    patch_projection = patch_width * d_model + d_model
    temporal_position = patch_count * d_model
    transformer_layer = (
        4 * d_model * d_model
        + 2 * d_model * feed_forward
        + feed_forward
        + 9 * d_model
    )
    final_norms = 4 * d_model
    shared_output_head = d_model * head_width + head_width
    return int(
        patch_projection
        + temporal_position
        + layer_count * transformer_layer
        + final_norms
        + shared_output_head
    )


def _fold_checks(asset_folds: dict[str, Any], triplet_catalog: dict[str, Any]) -> dict[str, bool]:
    folds = asset_folds["folds"]
    catalogs = {int(item["fold"]): item for item in triplet_catalog["folds"]}
    return {
        "exact_three_folds": len(folds) == 3 and set(catalogs) == {1, 2, 3},
        "twenty_train_ten_test_per_fold": all(
            len(fold["train_symbols"]) == 20 and len(fold["test_symbols"]) == 10
            for fold in folds
        ),
        "fold_assets_are_disjoint": all(
            not set(fold["train_symbols"]).intersection(fold["test_symbols"])
            for fold in folds
        ),
        "exact_triplet_counts": all(
            len(catalogs[int(fold["fold"])]["train_triplets"]) == math.comb(20, 3)
            and len(catalogs[int(fold["fold"])]["test_triplets"]) == math.comb(10, 3)
            for fold in folds
        ),
    }


def run_state_conditioned_multi_horizon_spec(config: dict[str, Any]) -> dict[str, Any]:
    spec = config["state_conditioned_multi_horizon_spec"]
    root = Path(spec["project_root"]).resolve()
    paths = {name: root / value for name, value in spec["inputs"].items()}
    observed = {name: _sha256_file(path) for name, path in paths.items()}
    if observed != spec["expected_input_sha256"]:
        raise RuntimeError("V55 immutable metadata input drift")

    v32_result = _load_json(paths["v32_result"])
    v32_audit = _load_json(paths["v32_audit"])
    manifest = _load_json(paths["v32_dataset_manifest"])
    feature_schema = _load_json(paths["v32_feature_schema"])
    asset_folds = _load_json(paths["v32_asset_folds"])
    triplet_catalog = _load_json(paths["v32_triplet_catalog"])
    v50_result = _load_json(paths["v50_result"])
    v50_audit = _load_json(paths["v50_audit"])
    v54_result = _load_json(paths["v54_result"])
    attribution = _load_json(paths["v54_failure_attribution"])
    v54_audit = _load_json(paths["v54_audit"])

    architecture = spec["architecture"]
    parameter_count = analytic_parameter_count(architecture)
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
        "policy": spec["policy"],
        "training": training,
        "controls": spec["controls"],
        "later_evaluation": spec["later_evaluation"],
        "lifecycle": spec["lifecycle"],
        "source_dataset_contract": {
            "dataset_manifest_sha256": spec["expected_input_sha256"]["v32_dataset_manifest"],
            "feature_schema_sha256": spec["expected_input_sha256"]["v32_feature_schema"],
            "asset_folds_sha256": spec["expected_input_sha256"]["v32_asset_folds"],
            "triplet_catalog_sha256": spec["expected_input_sha256"]["v32_triplet_catalog"],
            "panel_sha256": manifest["panel_sha256"],
            "sequence_index_sha256": manifest["sequence_index_sha256"],
            "symbol_count": manifest["symbol_count"],
            "feature_count": len(feature_schema["model_feature_order"]),
            "panel_or_label_loaded": False,
        },
        "diagnostic_basis": {
            "v54_result_sha256": v54_result["result_sha256"],
            "absolute_calibration_unstable": attribution["absolute_calibration_unstable"],
            "structural_turnover_failure": attribution["structural_turnover_failure"],
            "ranking_signal_survived": attribution["ranking_signal_survived"],
        },
        "parameter_count_analytic": parameter_count,
        "registered_job_count": job_count,
    }
    blueprint["blueprint_sha256"] = _canonical_sha256(blueprint)

    selected_symbols = set(manifest["symbols"])
    folds = _fold_checks(asset_folds, triplet_catalog)
    constraints = spec["constraints"]
    checks = {
        "all_metadata_hashes_match": observed == spec["expected_input_sha256"],
        "v32_metadata_contract_passes": v32_audit["passed"] is True
        and v32_result["audit"]["passed"] is True,
        "exact_non_target_universe_preserved": len(selected_symbols) == 30
        and not TARGET_SYMBOLS.intersection(selected_symbols),
        "exact_source_panel_receipts_preserved": manifest["panel_sha256"]
        == spec["data_contract"]["source_panel_sha256"]
        and manifest["sequence_index_sha256"]
        == spec["data_contract"]["source_sequence_index_sha256"],
        "exact_input_shape_and_features": spec["data_contract"]["input_shape"]
        == [None, 256, 3, 9]
        and len(feature_schema["model_feature_order"]) == 9,
        **folds,
        "v50_retirement_is_binding": v50_audit["passed"] is True
        and v50_result["decision"] == "retire_family_without_tuning",
        "v54_diagnostic_is_complete": v54_audit["passed"] is True
        and v54_result["decision"] == "v50_retirement_confirmed_diagnostic_only",
        "v54_failure_modes_drive_new_family": attribution["ranking_signal_survived"]
        and attribution["economic_conversion_failed"]
        and attribution["structural_turnover_failure"]
        and attribution["absolute_calibration_unstable"],
        "family_identity_is_new": spec["lineage"]["family_is_new"] is True
        and spec["candidate_family_id"] != spec["lineage"]["retired_parent_family"],
        "no_checkpoint_reuse": spec["lineage"]["previous_checkpoint_reuse_allowed"]
        is False
        and training["checkpoint_reuse"] == "none",
        "one_architecture_under_ceiling": architecture["variant_count"] == 1
        and parameter_count == architecture["expected_parameter_count"]
        and parameter_count <= architecture["parameter_ceiling"],
        "multi_horizon_quantile_contract_is_exact": architecture["output_horizons"]
        == [1, 3, 7]
        and architecture["output_quantiles"] == [0.2, 0.5, 0.8]
        and spec["objective"]["absolute_calibration_is_primary"] is True,
        "policy_is_state_conditioned_and_structurally_slow": spec["policy"][
            "state_conditioning"
        ]
        == "deterministic_transition_cost_from_current_action"
        and spec["policy"]["decision_clock"]
        == "every_seventh_eligible_signal_date_per_pseudo_deployment"
        and spec["policy"]["forecast_used"] == "h7_q20",
        "full_training_grid_is_registered_without_selection": job_count == 36
        and job_count == training["expected_job_count"]
        and training["seed_fold_origin_geometry_selection_allowed"] is False,
        "consumed_windows_are_not_clean_confirmation": spec["later_evaluation"][
            "evidence_status"
        ]
        == "adaptive_historical_development_only"
        and spec["later_evaluation"]["consumed_windows"] == [2024, 2025],
        "target_assets_remain_sealed": spec["target_contract"]["target_data_allowed"]
        is False
        and spec["later_evaluation"]["target_assets_allowed"] is False
        and spec["lifecycle"]["target_assets_remain_sealed_through_v61"] is True,
        "v55_is_metadata_only": constraints["metadata_only"] is True
        and not any(
            constraints[name]
            for name in [
                "parquet_deserialization_allowed",
                "model_instantiation_allowed",
                "optimizer_step_allowed",
                "prediction_allowed",
                "performance_metric_allowed",
                "pnl_allowed",
                "target_asset_allowed",
            ]
        ),
        "v55_authorizes_only_synthetic_harness": spec["authorized_next_action"]
        == "authorize_v56_synthetic_state_policy_harness_only",
    }
    audit = {"passed": all(checks.values()), "checks": checks}
    if not audit["passed"]:
        raise RuntimeError(f"V55 specification audit failed: {checks}")

    result = {
        "version": "v55",
        "decision": spec["authorized_next_action"],
        "blueprint": blueprint,
        "blueprint_sha256": blueprint["blueprint_sha256"],
        "audit": audit,
        "input_hash_receipt": observed,
        "summary": {
            "parameter_count": parameter_count,
            "registered_job_count": job_count,
            "parquet_deserializations": 0,
            "model_instantiations": 0,
            "optimizer_steps": 0,
            "target_asset_rows": 0,
        },
    }
    result["result_sha256"] = _canonical_sha256(result)
    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "blueprint.json", blueprint)
    _write_json(output / "audit.json", audit)
    _write_json(output / "input_hash_receipt.json", observed)
    _write_json(output / "result.json", result)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    report = "\n".join(
        [
            "# V55 State-Conditioned Multi-Horizon Family Specification",
            "",
            f"Decision: **{result['decision']}**",
            "",
            f"Blueprint SHA-256: `{blueprint['blueprint_sha256']}`",
            f"Analytic parameters: **{parameter_count:,}**",
            "",
            "This is a metadata-only ex-ante specification. No Parquet, model,",
            "prediction, performance statistic, PnL, or target asset was opened.",
            "V56 may exercise only the synthetic scientific harness.",
            "",
        ]
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    return result
