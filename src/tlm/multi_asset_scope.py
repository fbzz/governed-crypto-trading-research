from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_multi_asset_scope_amendment(config: dict) -> dict[str, object]:
    amendment = config["multi_asset_scope_amendment"]
    root = Path(amendment["project_root"]).resolve()
    input_paths = {
        name: root / relative
        for name, relative in amendment["inputs"].items()
    }
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V29 inputs are missing: {missing}")
    inputs = {name: _load_json(path) for name, path in input_paths.items()}
    input_hashes = {name: _sha256_file(path) for name, path in input_paths.items()}

    v26 = inputs["v26_specification"]
    v27 = inputs["v27_inventory"]
    v28 = inputs["v28_result"]
    v28_audit = inputs["v28_audit"]
    old_blueprint = v26["blueprint"]
    universe = dict(amendment["training_universe"])
    target_assets = list(amendment["target_assets"])
    target_symbols = list(amendment["target_symbols"])

    selection_contract = {
        "state": "policy_frozen_universe_not_yet_selected",
        "selected_symbols": [],
        "selected_asset_count": int(universe["selected_asset_count"]),
        "asset_fold_count": int(universe["asset_fold_count"]),
        "venue": universe["venue"],
        "quote_asset": universe["quote_asset"],
        "listed_on_or_before": universe["listed_on_or_before"],
        "observation_window": [
            universe["selection_observation_start"],
            universe["selection_observation_end"],
        ],
        "minimum_daily_coverage": float(universe["minimum_daily_coverage"]),
        "minimum_nonzero_quote_volume_fraction": float(
            universe["minimum_nonzero_quote_volume_fraction"]
        ),
        "ranking": {
            "metric": universe["ranking_metric"],
            "direction": universe["ranking_direction"],
            "tie_breaker": universe["tie_breaker"],
            "input_columns_allowed": list(universe["input_columns_allowed"]),
        },
        "future_window_usage": {
            "validation_calibration_confirmation_used_for_selection": bool(
                universe[
                    "validation_calibration_confirmation_used_for_selection"
                ]
            ),
            "full_future_availability_used_for_selection": bool(
                universe["full_future_availability_used_for_selection"]
            ),
        },
        "exclusions": {
            "target_bases": list(universe["excluded_bases"]),
            "target_proxy_bases": list(universe["target_proxy_bases"]),
            "fiat_bases": list(universe["fiat_bases"]),
            "stablecoin_bases": list(universe["stablecoin_bases"]),
            "fan_token_bases": list(universe["fan_token_bases"]),
            "token_suffixes": list(universe["excluded_token_suffixes"]),
        },
    }
    blueprint = {
        "candidate_family_id": amendment["candidate_family_id"],
        "state": "scope_amended_not_inventoried_not_trained",
        "training_mode": "shared_representation_from_non_target_crypto_triplets",
        "inference_mode": "target_only_zero_shot_after_source_domain_gates",
        "training_universe": selection_contract,
        "target_contract": {
            "assets": target_assets,
            "symbols": target_symbols,
            "only_inference_assets": target_symbols,
            "only_tradable_assets": target_symbols,
            "development_data_allowed": False,
            "development_labels_allowed": False,
            "development_prediction_count": 0,
            "development_performance_evaluation_count": 0,
            "allowed_target_evaluation": (
                "v22_one_shot_after_candidate_registration_and_maturity_only"
            ),
        },
        "data_contract": old_blueprint["data_contract"],
        "chronological_splits": old_blueprint["chronological_splits"],
        "architecture": old_blueprint["architecture"],
        "training": old_blueprint["training"],
        "policy": old_blueprint["policy"],
        "source_domain_gates": {
            **old_blueprint["source_domain_gates"],
            "asset_fold_count": int(universe["asset_fold_count"]),
            "assets_per_holdout_fold": int(universe["selected_asset_count"])
            // int(universe["asset_fold_count"]),
        },
        "historical_artifact_status": {
            "v27_lexical_universe": "superseded_not_deleted",
            "v28_lexical_dataset": "superseded_for_training_not_deleted",
            "reason": (
                "replace_lexical_cap_with_performance_blind_crypto_liquidity_policy"
            ),
        },
    }
    blueprint_sha256 = _canonical_sha256(blueprint)

    exclusion_sets = selection_contract["exclusions"]
    all_excluded = set().union(
        *(set(values) for values in exclusion_sets.values())
    )
    representation_window = old_blueprint["chronological_splits"][
        "representation_train"
    ]
    expected_hashes = amendment["expected_input_sha256"]
    checks = {
        "all_inputs_exist": not missing,
        "all_input_hashes_match": all(
            input_hashes[name] == expected_hashes[name]
            for name in input_paths
        ),
        "v26_family_is_expected_predecessor": old_blueprint[
            "candidate_family_id"
        ] == amendment["supersedes_candidate_family_id"],
        "v27_is_data_only": v27["decision"]
        == "authorize_v28_non_target_dataset_build_only",
        "v28_audit_passes": bool(v28_audit["passed"]),
        "v28_has_no_model_or_performance": not v28["tested"]["model_trained"]
        and not v28["tested"]["performance_metrics_computed"]
        and not v28["tested"]["pnl_computed"],
        "targets_are_exactly_btc_eth_sol": target_assets == ["BTC", "ETH", "SOL"]
        and target_symbols == ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "targets_are_excluded_from_training": set(target_assets).issubset(
            set(universe["excluded_bases"])
        ),
        "only_targets_are_tradable": blueprint["target_contract"][
            "only_tradable_assets"
        ] == target_symbols,
        "training_asset_count_is_30": int(universe["selected_asset_count"])
        == 30,
        "asset_folds_are_equal": int(universe["selected_asset_count"])
        % int(universe["asset_fold_count"])
        == 0,
        "selection_uses_training_window_only": universe[
            "selection_observation_start"
        ] == representation_window[0]
        and universe[
            "selection_observation_end"
        ]
        == representation_window[1]
        and not selection_contract["future_window_usage"][
            "validation_calibration_confirmation_used_for_selection"
        ]
        and not selection_contract["future_window_usage"][
            "full_future_availability_used_for_selection"
        ],
        "ranking_is_quote_volume_only": selection_contract["ranking"] == {
            "metric": "median_daily_quote_volume_usdt",
            "direction": "descending",
            "tie_breaker": "lexical_symbol_ascending",
            "input_columns_allowed": ["symbol", "date", "quote_volume"],
        },
        "crypto_exclusions_are_explicit": set(target_assets).issubset(all_excluded)
        and "EUR" in all_excluded
        and "USDC" in all_excluded
        and "ASR" in all_excluded
        and "UP" in all_excluded,
        "no_universe_is_selected_prematurely": not selection_contract[
            "selected_symbols"
        ],
        "no_target_development_or_evaluation": not blueprint[
            "target_contract"
        ]["development_data_allowed"]
        and not blueprint["target_contract"]["development_labels_allowed"]
        and blueprint["target_contract"]["development_prediction_count"] == 0
        and blueprint["target_contract"][
            "development_performance_evaluation_count"
        ]
        == 0,
        "only_v30_inventory_is_authorized": amendment["authorized_next_action"]
        == "v30_training_universe_liquidity_inventory_only",
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise RuntimeError(f"V29 scope-amendment audit failed: {checks}")

    return {
        "version": "v29",
        "method": "performance_blind_multi_asset_training_scope_amendment",
        "decision": "authorize_v30_training_universe_liquidity_inventory_only",
        "supersession": {
            "superseded_candidate_family_id": amendment[
                "supersedes_candidate_family_id"
            ],
            "replacement_candidate_family_id": amendment["candidate_family_id"],
            "performance_observed_before_amendment": False,
            "historical_artifacts_deleted": False,
        },
        "blueprint": blueprint,
        "blueprint_sha256": blueprint_sha256,
        "tested": {
            "universe_selected": False,
            "raw_market_data_loaded": False,
            "label_columns_read": False,
            "returns_computed": False,
            "model_trained": False,
            "performance_metrics_computed": False,
            "pnl_computed": False,
            "target_assets_loaded": False,
            "improvement_status": "unknown_not_evaluated",
            "drawdown_status": "unknown_not_evaluated",
        },
        "input_hashes": input_hashes,
        "audit": {"passed": True, "checks": checks},
    }


def _report(result: dict[str, object]) -> str:
    blueprint = result["blueprint"]
    universe = blueprint["training_universe"]
    targets = blueprint["target_contract"]
    return "\n".join([
        "# TLM v29 Multi-Asset Training Scope Amendment",
        "",
        "## Decision",
        "",
        "**MULTI-ASSET TRAINING RETAINED; BTC/ETH/SOL ARE THE ONLY TARGETS AND TRADABLE ASSETS.**",
        "",
        "The previous 48-symbol lexical universe and its dataset remain immutable historical artifacts but are superseded for future training. No performance result caused this amendment because no model had been trained.",
        "",
        "## Frozen training-universe policy",
        "",
        f"- Select exactly {universe['selected_asset_count']} non-target cryptoassets.",
        f"- Use {universe['asset_fold_count']} equal asset-disjoint folds.",
        f"- Require {universe['minimum_daily_coverage']:.1%} coverage in {universe['observation_window'][0]} through {universe['observation_window'][1]}.",
        "- Rank only by median daily USDT quote volume inside that training window; ties are lexical.",
        "- Exclude target/proxy, fiat, stablecoin, fan-token, and leveraged-token bases.",
        "- Do not use 2024 validation, 2025 calibration, 2026 confirmation, labels, returns, or PnL for selection.",
        "",
        "## Target boundary",
        "",
        f"Training excludes {targets['symbols']}. After source-domain gates, these are also the only permitted inference and trading assets. Their development prediction and performance counts remain zero.",
        "",
        "## Next action",
        "",
        "V30 may inventory official archives and select the 30-symbol training universe using only the frozen coverage and quote-volume columns. It may not read labels, train a model, or evaluate performance.",
        "",
    ])


def run_multi_asset_scope_amendment(config: dict) -> dict[str, object]:
    result = build_multi_asset_scope_amendment(config)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "amendment.json": result,
        "blueprint.json": result["blueprint"],
        "audit.json": result["audit"],
    }
    for name, value in files.items():
        (output / name).write_text(
            json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
        )
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    return result
