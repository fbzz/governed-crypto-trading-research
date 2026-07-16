import json

from tlm.multi_asset_scope import (
    build_multi_asset_scope_amendment,
    run_multi_asset_scope_amendment,
)


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _config(tmp_path):
    v26 = tmp_path / "v26.json"
    _write(v26, {
        "blueprint": {
            "candidate_family_id": "old",
            "data_contract": {"derived_features": ["return"]},
            "chronological_splits": {
                "representation_train": ["2021-01-01", "2023-12-31"]
            },
            "architecture": {"d_model": 32},
            "training": {"seeds": [1, 2, 3]},
            "policy": {"action_space": ["long_top1", "cash"]},
            "source_domain_gates": {"minimum_asset_folds": 3},
        }
    })
    v27 = tmp_path / "v27.json"
    _write(v27, {"decision": "authorize_v28_non_target_dataset_build_only"})
    v28 = tmp_path / "v28.json"
    _write(v28, {"tested": {
        "model_trained": False,
        "performance_metrics_computed": False,
        "pnl_computed": False,
    }})
    audit = tmp_path / "audit.json"
    _write(audit, {"passed": True})
    paths = {
        "v26_specification": v26,
        "v27_inventory": v27,
        "v28_result": v28,
        "v28_audit": audit,
    }
    hashes = {
        name: __import__("hashlib").sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }
    return {
        "multi_asset_scope_amendment": {
            "project_root": str(tmp_path),
            "inputs": {name: path.name for name, path in paths.items()},
            "expected_input_sha256": hashes,
            "candidate_family_id": "new",
            "supersedes_candidate_family_id": "old",
            "target_assets": ["BTC", "ETH", "SOL"],
            "target_symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            "training_universe": {
                "venue": "Binance spot",
                "quote_asset": "USDT",
                "selected_asset_count": 30,
                "asset_fold_count": 3,
                "listed_on_or_before": "2021-01-01",
                "selection_observation_start": "2021-01-01",
                "selection_observation_end": "2023-12-31",
                "minimum_daily_coverage": 0.995,
                "minimum_nonzero_quote_volume_fraction": 0.99,
                "ranking_metric": "median_daily_quote_volume_usdt",
                "ranking_direction": "descending",
                "tie_breaker": "lexical_symbol_ascending",
                "input_columns_allowed": ["symbol", "date", "quote_volume"],
                "validation_calibration_confirmation_used_for_selection": False,
                "full_future_availability_used_for_selection": False,
                "excluded_bases": ["BTC", "ETH", "SOL"],
                "target_proxy_bases": ["WBTC"],
                "fiat_bases": ["EUR"],
                "stablecoin_bases": ["USDC"],
                "fan_token_bases": ["ASR"],
                "excluded_token_suffixes": ["UP"],
            },
            "authorized_next_action": (
                "v30_training_universe_liquidity_inventory_only"
            ),
        },
        "output_dir": str(tmp_path / "output"),
    }


def test_amendment_keeps_multi_asset_training_and_target_only_execution(tmp_path):
    result = build_multi_asset_scope_amendment(_config(tmp_path))
    blueprint = result["blueprint"]
    assert result["decision"] == (
        "authorize_v30_training_universe_liquidity_inventory_only"
    )
    assert blueprint["training_universe"]["selected_asset_count"] == 30
    assert blueprint["source_domain_gates"]["assets_per_holdout_fold"] == 10
    assert blueprint["target_contract"]["only_tradable_assets"] == [
        "BTCUSDT", "ETHUSDT", "SOLUSDT"
    ]
    assert not result["tested"]["model_trained"]
    assert not result["tested"]["label_columns_read"]
    assert result["audit"]["passed"]


def test_amendment_writes_blueprint_audit_and_report(tmp_path):
    config = _config(tmp_path)
    result = run_multi_asset_scope_amendment(config)
    output = tmp_path / "output"
    assert result["blueprint_sha256"]
    assert (output / "amendment.json").is_file()
    assert (output / "blueprint.json").is_file()
    assert (output / "audit.json").is_file()
    assert "BTC/ETH/SOL" in (output / "report.md").read_text()
