import json

import yaml

from tlm.control_certificate import build_control_certificate, run_control_certificate


def _metric(observations=100):
    return {
        "observations": observations,
        "total_return": 0.5,
        "sharpe": 0.8,
        "max_drawdown": -0.4,
    }


def _config(tmp_path):
    names = ["expanding_6", "reference_expanding_3", "rolling_6_730d"]
    scenarios = {}
    for name in names:
        monte_carlo = {}
        for block in (7, 21, 63):
            monte_carlo[str(block)] = {
                "paths": 3000,
                "block_length": block,
                "distributions": {
                    "dual_momentum": {
                        "total_return": {"median": 0.5, "p05": -0.1},
                        "sharpe": {"median": 0.8, "p05": 0.1},
                        "max_drawdown": {"median": -0.45, "p05": -0.7},
                    }
                },
            }
        scenarios[name] = {
            "validation": {"mode": "rolling" if name.startswith("rolling") else "expanding"},
            "metrics": {"dual_momentum_30": _metric()},
            "monte_carlo": monte_carlo,
        }
    validation = {
        "dual_momentum_beats_buy_hold_in_all_scenarios": True,
        "scenarios": scenarios,
    }
    ledger = {
        "decision": "halt_new_historical_model_search",
        "synthesis": {"active_candidate_versions": []},
    }
    files = {
        "validation_result": tmp_path / "validation.json",
        "validation_audit": tmp_path / "validation_audit.json",
        "validation_config": tmp_path / "v6.yaml",
        "evidence_ledger": tmp_path / "ledger.json",
        "evidence_audit": tmp_path / "ledger_audit.json",
        "control_implementation": tmp_path / "consensus.py",
    }
    files["validation_result"].write_text(json.dumps(validation))
    files["validation_audit"].write_text(json.dumps({"passed": True}))
    files["validation_config"].write_text(yaml.safe_dump({
        "data": {"assets": {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}},
        "target": {"mode": "next_open_to_open"},
        "strategy": {"cost_bps": 10.0},
        "consensus": {"momentum_lookback": 30},
    }))
    files["evidence_ledger"].write_text(json.dumps(ledger))
    files["evidence_audit"].write_text(json.dumps({"passed": True}))
    files["control_implementation"].write_text("# fixture\n")
    return {
        "control_certificate": {
            **{f"{name}_path": str(path) for name, path in files.items()},
            "frozen_control_name": "dual_momentum_30",
            "frozen_lookback_days": 30,
            "expected_scenarios": names,
            "expected_blocks": [7, 21, 63],
            "expected_paths": 3000,
        },
        "output_dir": str(tmp_path / "output"),
    }


def test_control_certificate_freezes_research_role_and_risk(tmp_path):
    result = build_control_certificate(_config(tmp_path))
    assert result["benchmark_status"] == "certified_research_control"
    assert result["deployment_status"] == "not_authorized"
    assert result["risk_summary"]["bootstrap_cells_with_negative_p05_return"] == 9
    assert result["audit"]["passed"]


def test_control_certificate_writes_report_and_machine_outputs(tmp_path):
    config = _config(tmp_path)
    run_control_certificate(config)
    output = tmp_path / "output"
    assert (output / "certificate.json").is_file()
    assert (output / "audit.json").is_file()
    assert (output / "report.md").is_file()
