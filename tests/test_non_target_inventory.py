import io
import urllib.parse
import zipfile

from tlm.non_target_inventory import (
    audit_candidate,
    build_inventory_result,
    exclusion_reason,
    inspect_kline_zip,
    list_common_prefixes,
)


def _zip_row(open_timestamp, close_timestamp):
    return [
        str(open_timestamp),
        "1.0",
        "1.1",
        "0.9",
        "1.05",
        "100",
        str(close_timestamp),
        "105",
        "12",
        "50",
        "52.5",
        "0",
    ]


def _zip_payload(rows):
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        content = "\n".join(",".join(row) for row in rows) + "\n"
        archive.writestr("fixture.csv", content)
    return target.getvalue()


def test_prefix_listing_follows_s3_marker():
    first = b"""<?xml version="1.0"?>
    <ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
      <IsTruncated>true</IsTruncated><NextMarker>root/BB/</NextMarker>
      <CommonPrefixes><Prefix>root/AA/</Prefix></CommonPrefixes>
      <CommonPrefixes><Prefix>root/BB/</Prefix></CommonPrefixes>
    </ListBucketResult>"""
    second = b"""<?xml version="1.0"?>
    <ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
      <IsTruncated>false</IsTruncated>
      <CommonPrefixes><Prefix>root/CC/</Prefix></CommonPrefixes>
    </ListBucketResult>"""

    def fetch(url, _timeout):
        marker = urllib.parse.parse_qs(urllib.parse.urlparse(url).query).get("marker")
        return second if marker else first

    prefixes, pages = list_common_prefixes(
        "https://fixture", "root/", 1.0, fetch_bytes=fetch
    )
    assert prefixes == ["root/AA/", "root/BB/", "root/CC/"]
    assert len(pages) == 2
    assert all(page["sha256"] for page in pages)


def test_kline_parser_handles_official_ms_to_us_transition():
    ms = _zip_payload([
        _zip_row(1_609_459_200_000, 1_609_545_599_999),
        _zip_row(1_609_545_600_000, 1_609_631_999_999),
    ])
    us = _zip_payload([
        _zip_row(1_735_689_600_000_000, 1_735_775_999_999_999),
        _zip_row(1_735_776_000_000_000, 1_735_862_399_999_999),
    ])
    assert inspect_kline_zip(ms)["timestamp_units"] == ["ms"]
    assert inspect_kline_zip(us)["timestamp_units"] == ["us"]


def test_exclusion_contract_blocks_targets_proxies_stables_and_suffixes():
    universe = {
        "excluded_bases": ["BTC", "ETH", "SOL"],
        "target_proxy_bases": ["WBTC", "WETH"],
        "excluded_stablecoin_bases": ["USDC"],
        "excluded_token_suffixes": ["UP", "DOWN", "BULL", "BEAR"],
    }
    assert exclusion_reason("BTCUSDT", "USDT", universe) == "target_base"
    assert exclusion_reason("WBTCUSDT", "USDT", universe) == "target_proxy"
    assert exclusion_reason("USDCUSDT", "USDT", universe) == "stablecoin_base"
    assert exclusion_reason("AAVEUPUSDT", "USDT", universe) == "excluded_suffix"
    assert exclusion_reason("ADAUSDT", "USDT", universe) is None


def test_result_selects_lexical_eligible_universe_without_performance():
    blueprint = {
        "decision": "authorize_v27_non_target_universe_data_audit_only",
        "blueprint_sha256": "frozen",
        "blueprint": {
            "target_symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            "development_universe": {
                "minimum_assets": 1,
                "maximum_assets": 2,
                "minimum_daily_coverage": 0.98,
                "selection_rule": "lexical",
                "excluded_bases": ["BTC", "ETH", "SOL"],
                "target_proxy_bases": ["WBTC"],
                "excluded_stablecoin_bases": ["USDC"],
            },
        },
    }
    candidate_audits = [
        {"symbol": "BBBUSDT", "base": "BBB", "coverage": 1.0, "eligible": True},
        {"symbol": "AAAUSDT", "base": "AAA", "coverage": 0.99, "eligible": True},
        {"symbol": "CCCUSDT", "base": "CCC", "coverage": 0.50, "eligible": False},
    ]
    records = [
        {"symbol": symbol, "checksum_verified": True}
        for symbol in ("AAAUSDT", "BBBUSDT")
    ]
    discovery = {
        "root_pages": [{"sha256": "root"}],
        "symbol_listing_pages": [{"sha256": "symbol"}],
    }
    config = {
        "non_target_inventory": {"expected_blueprint_sha256": "frozen"}
    }
    result = build_inventory_result(
        blueprint, discovery, candidate_audits, records, config
    )
    assert result["universe"]["selected_symbols"] == ["AAAUSDT", "BBBUSDT"]
    assert result["decision"] == "authorize_v28_non_target_dataset_build_only"
    assert not result["tested"]["returns_computed"]
    assert not result["tested"]["pnl_computed"]
    assert result["audit"]["passed"]
    assert result["universe"]["scope_observations"] == []


def test_candidate_records_rejected_archive_without_silently_using_it():
    record = {
        "dates": ["2021-01-01", "2021-01-02"],
        "checksum_verified": True,
        "schema_valid": True,
        "timestamp_units": ["ms"],
    }
    rejection = {
        "symbol": "AAAUSDT",
        "month": "2021-02",
        "error_type": "ValueError",
        "error": "duplicate UTC dates",
    }
    result = audit_candidate(
        {"symbol": "AAAUSDT", "base": "AAA"},
        [record],
        {"2021-01-01", "2021-01-02"},
        "2021-01-01",
        0.98,
        [rejection],
    )
    assert result["eligible"]
    assert result["rejected_archive_count"] == 1
    assert result["archive_rejections"] == [rejection]
    assert result["checks"]["all_used_archive_schemas_valid"]
