from tlm.non_target_dataset import build_asset_folds
from tlm.selected_source_manifest import build_selected_manifest_result


def test_v31_preserves_v30_universe_and_authorizes_dataset_only():
    symbols = [f"A{index:02d}USDT" for index in range(30)]
    amendment = {
        "blueprint_sha256": "blueprint",
        "blueprint": {
            "target_contract": {"symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"]}
        },
    }
    v30 = {
        "decision": "authorize_v31_selected_universe_manifest_refresh_only",
        "audit": {"passed": True},
        "universe": {"selected_symbols": symbols},
        "asset_folds": build_asset_folds(symbols, 3),
    }
    audits = [
        {"symbol": symbol, "coverage": 1.0, "eligible": True}
        for symbol in symbols
    ]
    records = [
        {
            "symbol": symbol,
            "month": "2021-01",
            "row_count": 1,
            "checksum_verified": True,
            "schema_valid": True,
        }
        for symbol in symbols
    ]
    discovery = {
        "development_start": "2021-01-01",
        "development_end": "2026-06-30",
        "development_months": ["2021-01"],
        "expected_calendar_days": 1,
        "expected_archive_count": 30,
        "symbol_listing_pages": [{"sha256": "source"}],
    }
    config = {
        "selected_source_manifest": {
            "expected_v29_blueprint_sha256": "blueprint",
            "expected_v30_inventory_sha256": "inventory",
            "observed_v30_inventory_sha256": "inventory",
            "full_window_coverage_policy": "audit_only_no_reselection_or_replacement",
        }
    }
    result = build_selected_manifest_result(
        amendment, v30, audits, records, [], discovery, config
    )
    assert result["universe"]["selected_symbols"] == symbols
    assert result["decision"] == "authorize_v32_selected_universe_dataset_only"
    assert result["manifest_summary"]["accepted_archive_count"] == 30
    assert not result["tested"]["universe_reselected"]
    assert not result["tested"]["model_trained"]
    assert not result["tested"]["target_assets_loaded"]
