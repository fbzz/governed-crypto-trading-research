import json

from tlm.holdout_protocol import build_holdout_protocol, run_holdout_protocol


def _config(tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({
        "decision": "halt_new_historical_model_search",
        "synthesis": {"active_candidate_versions": []},
    }))
    ledger_audit = tmp_path / "ledger_audit.json"
    ledger_audit.write_text(json.dumps({"passed": True}))
    control = tmp_path / "control.json"
    control.write_text(json.dumps({
        "benchmark_status": "certified_research_control",
        "deployment_status": "not_authorized",
        "control": {"name": "dual_momentum_30"},
    }))
    control_audit = tmp_path / "control_audit.json"
    control_audit.write_text(json.dumps({"passed": True}))
    return {
        "holdout_protocol": {
            "evidence_ledger_path": str(ledger),
            "evidence_audit_path": str(ledger_audit),
            "control_certificate_path": str(control),
            "control_audit_path": str(control_audit),
            "last_exposed_return_at_utc": "2026-07-10T00:00:00Z",
            "earliest_candidate_registration_at_utc": "2026-07-13T23:59:59Z",
            "earliest_holdout_signal_at_utc": "2026-07-14T00:00:00Z",
            "initial_state": "dormant_no_registered_candidate",
            "registered_candidate": None,
            "minimum_calendar_days": 365,
            "minimum_daily_observations": 365,
            "minimum_active_position_days": 60,
            "minimum_regime_observations": {"bull": 60, "bear": 60, "high_volatility": 30},
            "regime_definitions": {"bull": "fixture", "bear": "fixture", "high_volatility": "fixture"},
            "outcome_source": "fixture",
            "outcome_source_verification": "fixture",
            "allowed_integrity_metadata": ["count"],
            "forbidden_until_unseal": ["returns"],
            "evaluation_cost_bps": [10, 20, 30],
            "bootstrap_block_lengths_days": [7, 21, 63],
            "bootstrap_paths": 10000,
            "max_drawdown_tolerance": 0.05,
        },
        "output_dir": str(tmp_path / "output"),
    }


def test_holdout_is_dormant_and_has_no_execution(tmp_path):
    result = build_holdout_protocol(_config(tmp_path))
    assert result["state"] == "dormant_no_registered_candidate"
    assert result["clean_holdout_status"] == "not_started"
    assert not result["evaluation"]["run_candidate_during_accumulation"]
    assert not result["evaluation"]["simulate_daily_orders"]
    assert result["audit"]["passed"]


def test_holdout_writes_hashed_protocol_and_schema(tmp_path):
    config = _config(tmp_path)
    run_holdout_protocol(config)
    output = tmp_path / "output"
    assert (output / "protocol.json").is_file()
    assert (output / "protocol.sha256").is_file()
    assert (output / "candidate_registration.schema.json").is_file()
    assert (output / "report.md").is_file()
