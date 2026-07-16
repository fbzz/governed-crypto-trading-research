import json
from urllib.parse import parse_qs, urlparse

import pandas as pd

from tlm.data_family_audit import (
    audit_dvol_frame,
    download_dvol,
    parse_dvol_response,
    run_data_family_audit,
    score_and_select_candidates,
)


def _payload(start: str, days: int) -> dict:
    rows = []
    for timestamp in pd.date_range(start, periods=days, freq="D", tz="UTC"):
        milliseconds = int(timestamp.timestamp() * 1000)
        value = 50.0 + len(rows)
        rows.append([milliseconds, value, value + 2, value - 2, value + 1])
    return {"jsonrpc": "2.0", "result": {"data": rows, "continuation": None}}


def _fixture_config(tmp_path) -> dict:
    return {
        "data_family_audit": {
            "weights": {
                "causal_timestamp_semantics": 0.25,
                "historical_coverage": 0.25,
                "public_reproducibility": 0.20,
                "solo_project_cost": 0.15,
                "information_independence": 0.15,
            },
            "deribit_dvol": {
                "endpoint": "https://example.test/dvol",
                "currencies": ["BTC", "ETH"],
                "start": "2022-01-01",
                "end": "2022-01-05",
                "resolution": "1D",
                "timeout_seconds": 1,
                "minimum_daily_coverage": 1.0,
            },
            "candidates": {
                "deribit_dvol": {
                    "principal_constraint": "fixture",
                    "evidence_urls": ["https://example.test/docs"],
                    "criteria": {
                        "causal_timestamp_semantics": 1.0,
                        "historical_coverage": 0.0,
                        "public_reproducibility": 1.0,
                        "solo_project_cost": 1.0,
                        "information_independence": 1.0,
                    },
                    "hard_gates": {
                        "genuinely_independent": True,
                        "historical_coverage": False,
                        "strict_causal_contract": False,
                    },
                },
                "unavailable": {
                    "principal_constraint": "no history",
                    "evidence_urls": ["https://example.test/unavailable"],
                    "criteria": {
                        "causal_timestamp_semantics": 1.0,
                        "historical_coverage": 0.0,
                        "public_reproducibility": 0.0,
                        "solo_project_cost": 1.0,
                        "information_independence": 1.0,
                    },
                    "hard_gates": {"historical_coverage": False},
                },
            },
        },
        "output_dir": str(tmp_path / "audit"),
    }


def test_dvol_parser_and_causal_contract_require_post_close_buffer():
    frame, continuation = parse_dvol_response(_payload("2022-01-01", 5), "BTC")
    assert continuation is None
    audit = audit_dvol_frame(frame, "2022-01-01", "2022-01-05", 1.0)
    assert audit["passed"]
    assert audit["coverage"] == 1.0
    assert audit["timestamp_contract"]["first_strict_execution_open"] == (
        "timestamp + 2 days"
    )


def test_candidate_selection_never_selects_failed_hard_gate():
    candidates = {
        "high_but_blocked": {
            "criteria": {"quality": 1.0},
            "hard_gates": {"history": False},
        },
        "eligible": {
            "criteria": {"quality": 0.4},
            "hard_gates": {"history": True},
        },
    }
    evaluated, selected = score_and_select_candidates(candidates, {"quality": 1.0})
    assert selected == ["eligible"]
    assert not evaluated["high_but_blocked"]["selected"]


def test_dvol_pagination_moves_end_timestamp_backward():
    calls = []

    def fetcher(url: str, timeout: float) -> dict:
        params = parse_qs(urlparse(url).query)
        calls.append(params)
        end = int(params["end_timestamp"][0])
        boundary = int(pd.Timestamp("2022-01-03", tz="UTC").timestamp() * 1000)
        if end > boundary:
            payload = _payload("2022-01-04", 2)
            payload["result"]["continuation"] = boundary
            return payload
        return _payload("2022-01-01", 3)

    frame, payloads = download_dvol(
        "https://example.test/dvol", "BTC", "2022-01-01", "2022-01-05",
        "1D", 1.0, fetch_json=fetcher,
    )
    assert len(frame) == 5
    assert len(payloads) == 2
    assert calls[0]["start_timestamp"] == calls[1]["start_timestamp"]
    assert int(calls[1]["end_timestamp"][0]) < int(calls[0]["end_timestamp"][0])


def test_fixture_feasibility_audit_writes_policy_free_outputs(tmp_path):
    config = _fixture_config(tmp_path)

    def fetcher(url: str, timeout: float) -> dict:
        assert timeout == 1
        params = parse_qs(urlparse(url).query)
        assert params["resolution"] == ["1D"]
        return _payload("2022-01-01", 5)

    result = run_data_family_audit(config, fetch_json=fetcher)
    output = tmp_path / "audit"
    assert result["selected"] == ["deribit_dvol"]
    assert result["audit"]["passed"]
    assert (output / "feasibility.json").is_file()
    assert (output / "audit.json").is_file()
    assert (output / "report.md").is_file()
    assert (output / "probes" / "BTC.parquet").is_file()
    persisted = json.loads((output / "feasibility.json").read_text())
    assert persisted["decision"] == "authorize_v14_deribit_dvol_data_layer_only"
