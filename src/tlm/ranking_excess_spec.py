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
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def analytic_parameter_count(architecture: dict, input_features: int) -> int:
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
    mask_token = d_model
    transformer_layer = (
        4 * d_model * d_model
        + 2 * d_model * feed_forward
        + feed_forward
        + 9 * d_model
    )
    final_norms = 4 * d_model
    output_heads = prediction_heads * (d_model + 1)
    reconstruction_head = d_model * patch_width + patch_width
    return int(
        patch_projection
        + temporal_position
        + mask_token
        + encoder_layers * transformer_layer
        + final_norms
        + output_heads
        + reconstruction_head
    )


def _fold_contract_checks(asset_folds: dict, triplet_catalog: dict) -> dict[str, bool]:
    folds = asset_folds["folds"]
    catalogs = {
        int(fold["fold"]): fold for fold in triplet_catalog["folds"]
    }
    checks = {
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
    return checks


def build_ranking_excess_spec(config: dict) -> dict[str, object]:
    spec = config["ranking_excess_spec"]
    root = Path(spec["project_root"]).resolve()
    paths = {name: root / relative for name, relative in spec["inputs"].items()}
    input_checks = {
        name: path.is_file()
        and _sha256_file(path) == spec["expected_input_sha256"][name]
        for name, path in paths.items()
    }
    if not all(input_checks.values()):
        raise RuntimeError(f"V41 input missing or hash drifted: {input_checks}")

    v32_result = _load_json(paths["v32_result"])
    v32_audit = _load_json(paths["v32_audit"])
    dataset_manifest = _load_json(paths["v32_dataset_manifest"])
    feature_schema = _load_json(paths["v32_feature_schema"])
    asset_folds = _load_json(paths["v32_asset_folds"])
    triplet_catalog = _load_json(paths["v32_triplet_catalog"])
    v37_result = _load_json(paths["v37_result"])
    v37_gate = _load_json(paths["v37_gate"])
    autopsy = _load_json(paths["v37_autopsy_result"])
    autopsy_audit = _load_json(paths["v37_autopsy_audit"])

    architecture = dict(spec["architecture"])
    input_features = len(feature_schema["model_feature_order"])
    parameter_count = analytic_parameter_count(architecture, input_features)
    blueprint = {
        "version": spec["version"],
        "candidate_family_id": spec["candidate_family_id"],
        "state": "ex_ante_design_frozen_not_implemented_or_trained",
        "lineage": spec["lineage"],
        "target_contract": spec["target_contract"],
        "data_contract": spec["data_contract"],
        "chronological_splits": spec["chronological_splits"],
        "architecture": architecture,
        "objective": spec["objective"],
        "training": spec["training"],
        "baselines": spec["baselines"],
        "policy": spec["policy"],
        "development_screen": spec["development_screen"],
        "constraints": spec["constraints"],
        "source_dataset_contract": {
            "manifest_sha256": spec["expected_input_sha256"][
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
            "symbol_count": dataset_manifest["symbol_count"],
            "feature_count": input_features,
            "panel_or_label_loaded": False,
        },
        "parameter_count_analytic": parameter_count,
    }
    blueprint["blueprint_sha256"] = _canonical_sha256(blueprint)

    fold_checks = _fold_contract_checks(asset_folds, triplet_catalog)
    selected_symbols = set(dataset_manifest["symbols"])
    splits = spec["chronological_splits"]
    checks = {
        "all_frozen_input_hashes_match": all(input_checks.values()),
        "v32_dataset_audit_passes": bool(v32_audit["passed"])
        and bool(v32_result["audit"]["passed"]),
        "v32_authorized_only_model_implementation": v32_result["decision"]
        == "authorize_v33_patch_transformer_implementation_only",
        "exact_v32_tensor_contract_reused": dataset_manifest["tensor_contract"]
        == {
            "dtype": "float32",
            "x_shape": [256, 3, 9],
            "y_shape": [3, 2],
        }
        and spec["data_contract"]["input_shape"] == [None, 256, 3, 9],
        "exact_nine_feature_schema_reused": input_features == 9
        and feature_schema["model_feature_order"][-1]
        == "within_triplet_relative_strength",
        "exact_thirty_non_target_symbols_reused": dataset_manifest["symbol_count"]
        == 30
        and len(selected_symbols) == 30
        and not TARGET_SYMBOLS.intersection(selected_symbols),
        "target_assets_remain_sealed": spec["target_contract"][
            "target_data_allowed"
        ]
        is False
        and spec["target_contract"]["target_prediction_count"] == 0
        and spec["target_contract"]["target_pnl_evaluation_count"] == 0,
        "parent_family_is_retired": v37_result["decision"]
        == spec["lineage"]["retired_parent_decision"]
        and not v37_gate["passed"],
        "autopsy_requires_new_ranking_family": bool(autopsy_audit["passed"])
        and autopsy["decision"]
        == spec["lineage"]["new_family_required_decision"]
        and autopsy["recommendation"]["next_family_primary_change"]
        == "train_cross_sectional_ranking_or_excess_return_objective",
        "candidate_family_id_is_new": spec["candidate_family_id"]
        != spec["lineage"]["retired_parent_family"],
        "exactly_one_medium_architecture": architecture["variant_count"] == 1
        and architecture["d_model"] == 128
        and architecture["encoder_layers"] == 4
        and architecture["cross_asset_attention_layers"] == 2
        and architecture["feed_forward_width"] == 512,
        "heads_divide_width": architecture["d_model"]
        % architecture["attention_heads"]
        == 0,
        "prediction_heads_are_exact": architecture["prediction_heads"]
        == ["excess_return_z", "log_volatility_7d"],
        "analytic_parameter_count_matches_frozen_value": parameter_count
        == architecture["expected_parameter_count_for_nine_features"],
        "ranking_excess_objective_is_exact": spec["objective"]["weights"]
        == {"ranking": 1.0, "excess": 1.0, "log_volatility": 0.1}
        and spec["objective"]["pairs"] == [[0, 1], [0, 2], [1, 2]]
        and spec["objective"]["outcome_weighting"] is False
        and spec["objective"]["target_clipping"] is False,
        "no_hyperparameter_or_seed_selection": spec["training"][
            "hyperparameter_search_allowed"
        ]
        is False
        and spec["training"]["seed_selection_allowed"] is False
        and spec["training"]["fold_selection_allowed"] is False
        and spec["training"]["seeds"] == [42, 7, 123],
        "old_checkpoints_cannot_be_reused": spec["lineage"][
            "old_checkpoints_compatible"
        ]
        is False
        and spec["training"]["old_checkpoint_reuse_allowed"] is False,
        "cost_hurdle_is_exactly_two_base_cost_legs": math.isclose(
            spec["policy"]["switch_hurdle"],
            2.0 * spec["policy"]["base_cost_bps"] / 10_000.0,
        )
        and spec["policy"]["switch_hurdle_cost_scenario_specific"] is False,
        "development_screen_is_2025_asset_disjoint_only": splits[
            "asset_disjoint_development_screen"
        ]
        == ["2025-01-01", "2025-12-23"]
        and spec["development_screen"]["role"]
        == "held_out_assets_per_fold_only",
        "consumed_2026_window_is_forbidden": splits["forbidden_consumed_window"]
        == ["2026-01-01", "2026-06-30"]
        and spec["lineage"]["v37_2026_window_status"]
        == "consumed_forbidden_for_new_family_evaluation",
        "future_confirmation_is_prospective_and_long_enough": splits[
            "prospective_confirmation_not_before"
        ]
        >= "2026-07-14"
        and splits["prospective_confirmation_minimum_mature_signal_dates"] >= 180,
        "v41_is_specification_only": all(
            value is False for value in spec["constraints"].values()
        ),
        "only_v42_synthetic_harness_is_authorized": spec[
            "authorized_next_action"
        ]
        == "v42_synthetic_ranking_excess_harness_only",
        **fold_checks,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V41 specification audit failed: {checks}")

    input_hashes_after = {
        name: _sha256_file(path) for name, path in paths.items()
    }
    if input_hashes_after != spec["expected_input_sha256"]:
        raise RuntimeError("V41 inputs changed while specification was built")

    return {
        "version": "v41",
        "method": "ex_ante_cross_sectional_ranking_excess_family_specification",
        "decision": "authorize_v42_synthetic_ranking_excess_harness_only",
        "blueprint": blueprint,
        "blueprint_sha256": blueprint["blueprint_sha256"],
        "tested": {
            "model_instantiations": 0,
            "panel_or_label_reads": 0,
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
    architecture = blueprint["architecture"]
    splits = blueprint["chronological_splits"]
    return "\n".join([
        "# TLM v41 Ranking/Excess Family Specification",
        "",
        "## Decision",
        "",
        "**NEW FAMILY FROZEN; ONLY THE SYNTHETIC V42 HARNESS IS AUTHORIZED.**",
        "",
        f"Family: `{blueprint['candidate_family_id']}`",
        f"Blueprint SHA-256: `{result['blueprint_sha256']}`",
        f"Analytic parameters: **{blueprint['parameter_count_analytic']:,}**",
        "",
        "No model, real panel, label, prediction, performance metric, PnL, or target asset was loaded or produced.",
        "",
        "## Primary change",
        "",
        "The old winner-classification/q50 family is replaced by one shared permutation-equivariant Medium scorer trained on both pairwise ordering and continuous triplet excess magnitude. There is no architecture sweep.",
        "",
        "## Frozen objective",
        "",
        "For each triplet, subtract its mean next-open log return, divide by the train-only fold RMS excess scale, center the three model scores, and optimize RankNet plus Smooth L1 excess. Log seven-day volatility remains auxiliary at weight 0.1.",
        "",
        "## Frozen policy",
        "",
        "Remain in cash only when every eligible 30-day momentum is non-positive. Otherwise rank by context/seed-averaged raw excess and switch only when the challenger exceeds the incumbent by 0.002, exactly two base-cost legs.",
        "",
        "## Evidence boundaries",
        "",
        f"Train through {splits['supervised_train'][1]}, early-stop on 2024 train assets, and use the 2025 held-out assets exactly once as a development screen. The consumed 2026 v37 interval is forbidden. A clean future source confirmation requires at least {splits['prospective_confirmation_minimum_mature_signal_dates']} mature dates beginning no earlier than {splits['prospective_confirmation_not_before']}.",
        "",
        "BTC/ETH/SOL remain sealed. Passing the 2025 development screen would only freeze the family to wait for future source confirmation.",
        "",
        "## Next action",
        "",
        "V42 may implement and smoke-test the exact Medium model, losses, normalization, policy, controls, and checkpoint contract on synthetic data only.",
        "",
    ])


def run_ranking_excess_spec(config: dict) -> dict[str, object]:
    result = build_ranking_excess_spec(config)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "specification.json", result)
    _write_json(output / "blueprint.json", result["blueprint"])
    _write_json(output / "audit.json", result["audit"])
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    return result
