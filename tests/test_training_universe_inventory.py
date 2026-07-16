import io
import zipfile

from tlm.training_universe_inventory import (
    audit_liquidity_candidate,
    build_training_universe_result,
    inspect_quote_volume_zip,
    scope_exclusion_reason,
)


def _contract():
    return {
        "selected_asset_count": 30,
        "asset_fold_count": 3,
        "quote_asset": "USDT",
        "listed_on_or_before": "2021-01-01",
        "observation_window": ["2021-01-01", "2023-12-31"],
        "minimum_daily_coverage": 0.995,
        "minimum_nonzero_quote_volume_fraction": 0.99,
        "ranking": {
            "metric": "median_daily_quote_volume_usdt",
            "direction": "descending",
            "tie_breaker": "lexical_symbol_ascending",
            "input_columns_allowed": ["symbol", "date", "quote_volume"],
        },
        "future_window_usage": {
            "full_future_availability_used_for_selection": False,
            "validation_calibration_confirmation_used_for_selection": False,
        },
        "exclusions": {
            "target_bases": ["BTC", "ETH", "SOL"],
            "target_proxy_bases": ["WBTC"],
            "fiat_bases": ["EUR"],
            "stablecoin_bases": ["USDC"],
            "fan_token_bases": ["ASR"],
            "token_suffixes": ["UP", "DOWN", "BULL", "BEAR"],
        },
    }


def _zip_payload(rows):
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("fixture.csv", "\n".join(",".join(row) for row in rows) + "\n")
    return target.getvalue()


def _row(timestamp, quote_volume):
    return [
        str(timestamp), "1", "1.1", "0.9", "1.05", "100",
        str(timestamp + 86_399_999), str(quote_volume), "10", "50", "52", "0",
    ]


def test_scope_exclusions_cover_every_frozen_category():
    contract = _contract()
    assert scope_exclusion_reason("BTCUSDT", "USDT", contract) == "target_base"
    assert scope_exclusion_reason("WBTCUSDT", "USDT", contract) == "target_proxy"
    assert scope_exclusion_reason("EURUSDT", "USDT", contract) == "fiat_base"
    assert scope_exclusion_reason("USDCUSDT", "USDT", contract) == "stablecoin_base"
    assert scope_exclusion_reason("ASRUSDT", "USDT", contract) == "fan_token_base"
    assert scope_exclusion_reason("AAVEUPUSDT", "USDT", contract) == "excluded_suffix"
    assert scope_exclusion_reason("ADAUSDT", "USDT", contract) is None


def test_quote_volume_parser_reads_only_registered_selection_fields():
    payload = _zip_payload([
        _row(1_609_459_200_000, 100),
        _row(1_609_545_600_000, 200),
    ])
    result = inspect_quote_volume_zip(payload)
    assert result["dates"] == ["2021-01-01", "2021-01-02"]
    assert result["quote_volumes"] == [100.0, 200.0]
    assert result["selection_columns_read"] == ["date", "quote_volume"]
    assert result["timestamp_units"] == ["ms"]


def test_candidate_uses_expected_calendar_denominator_without_imputation():
    record = {
        "dates": ["2021-01-01", "2021-01-02", "2021-01-03"],
        "quote_volumes": [100.0, 0.0, 300.0],
        "checksum_verified": True,
        "schema_valid": True,
        "selection_columns_read": ["date", "quote_volume"],
        "timestamp_units": ["ms"],
    }
    contract = _contract() | {
        "minimum_daily_coverage": 0.75,
        "minimum_nonzero_quote_volume_fraction": 0.50,
    }
    result = audit_liquidity_candidate(
        {"symbol": "AAAUSDT", "base": "AAA"},
        [record],
        {"2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"},
        contract,
    )
    assert result["coverage"] == 0.75
    assert result["nonzero_quote_volume_fraction"] == 0.50
    assert result["median_daily_quote_volume_usdt"] == 100.0
    assert result["eligible"]


def test_result_ranks_liquidity_with_lexical_tie_break_and_three_folds():
    contract = _contract()
    amendment = {
        "decision": "authorize_v30_training_universe_liquidity_inventory_only",
        "blueprint_sha256": "frozen",
        "blueprint": {
            "training_universe": contract,
            "target_contract": {"symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"]},
        },
    }
    audits = []
    records = []
    for index in range(32):
        symbol = f"A{index:02d}USDT"
        volume = 1000.0 - index // 2
        audits.append({
            "symbol": symbol,
            "base": f"A{index:02d}",
            "eligible": True,
            "coverage": 1.0,
            "nonzero_quote_volume_fraction": 1.0,
            "median_daily_quote_volume_usdt": volume,
        })
        records.append({
            "symbol": symbol,
            "month": "2021-01",
            "checksum_verified": True,
            "selection_columns_read": ["date", "quote_volume"],
        })
    discovery = {
        "root_pages": [{"sha256": "root"}],
        "symbol_listing_pages": [{"sha256": "symbol"}],
        "selection_listing": [{"selection_object_sha256": "selection"}],
        "expected_archive_job_count": len(records),
    }
    config = {
        "training_universe_inventory": {
            "expected_v29_blueprint_sha256": "frozen"
        }
    }
    result = build_training_universe_result(
        amendment, discovery, audits, records, config
    )
    assert result["universe"]["selected_count"] == 30
    assert result["universe"]["selected_symbols"][:4] == [
        "A00USDT", "A01USDT", "A02USDT", "A03USDT"
    ]
    assert all(len(fold["test_symbols"]) == 10 for fold in result["asset_folds"]["folds"])
    assert result["decision"] == "authorize_v31_selected_universe_manifest_refresh_only"
    assert not result["tested"]["model_trained"]
    assert not result["tested"]["target_assets_loaded"]
