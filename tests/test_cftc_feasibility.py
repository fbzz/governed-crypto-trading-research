import json
from urllib.parse import parse_qs, urlparse

import pandas as pd

from tlm.cftc_feasibility import (
    audit_cftc_rows,
    build_cftc_probe_url,
    run_cftc_feasibility,
)


def _config(tmp_path) -> dict:
    return {
        "cftc_feasibility": {
            "endpoint": "https://example.test/cot.json",
            "start": "2022-01-01", "end": "2022-02-01",
            "timeout_seconds": 1, "limit": 5000,
            "minimum_weekly_coverage": 1.0,
            "contract_codes": {"BTC": "BTC1", "ETH": "ETH1"},
            "required_fields": [
                "commodity_name", "contract_market_name",
                "cftc_contract_market_code", "report_date_as_yyyy_mm_dd",
                "open_interest_all", "lev_money_positions_long",
                "lev_money_positions_short",
            ],
            "known_delay_exception_url": "https://example.test/delays",
            "evidence_urls": [
                "https://example.test/1", "https://example.test/2",
                "https://example.test/3", "https://example.test/4",
            ],
        },
        "output_dir": str(tmp_path / "audit"),
    }


def _rows() -> list[dict]:
    rows = []
    for asset, code in (("BTC", "BTC1"), ("ETH", "ETH1")):
        for index, date in enumerate(pd.date_range(
            "2022-01-04", "2022-02-01", freq="W-TUE"
        )):
            rows.append({
                "commodity_name": asset,
                "contract_market_name": asset,
                "cftc_contract_market_code": code,
                "report_date_as_yyyy_mm_dd": date.isoformat(),
                "open_interest_all": str(1000 + index),
                "lev_money_positions_long": str(300 + index),
                "lev_money_positions_short": str(200 + index),
            })
    return rows


def test_probe_url_registers_contracts_window_and_fields(tmp_path):
    url = build_cftc_probe_url(_config(tmp_path))
    params = parse_qs(urlparse(url).query)
    assert "BTC1" in params["$where"][0]
    assert "2022-01-01" in params["$where"][0]
    assert "lev_money_positions_long" in params["$select"][0]


def test_weekly_coverage_audit_uses_tuesday_as_of_dates(tmp_path):
    result = audit_cftc_rows(_rows(), _config(tmp_path))
    assert result["all_contract_probes_passed"]
    assert result["contracts"]["BTC"]["coverage"] == 1.0
    assert result["contracts"]["ETH"]["rows"] == 5


def test_feasibility_rejects_family_despite_complete_live_coverage(tmp_path):
    config = _config(tmp_path)

    def fetcher(url: str, timeout: float) -> list[dict]:
        assert timeout == 1
        return _rows()

    result = run_cftc_feasibility(config, fetch_json=fetcher)
    output = tmp_path / "audit"
    assert result["audit"]["passed"]
    assert not result["selected"]
    assert not result["hard_gates"]["historical_release_timestamps_available"]
    assert (output / "feasibility.json").is_file()
    assert (output / "report.md").is_file()
    persisted = json.loads((output / "audit.json").read_text())
    assert persisted["passed"]
