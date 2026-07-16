import json

from tlm.zero_shot_spec import build_zero_shot_spec, run_zero_shot_spec


def _config(tmp_path):
    completion = tmp_path / "completion.json"
    completion.write_text(json.dumps({
        "decision": "complete_research_framework_no_deployable_tlm",
        "system_status": {"deployable_tlm": "not_available"},
    }))
    completion_audit = tmp_path / "completion_audit.json"
    completion_audit.write_text(json.dumps({"passed": True}))
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "state": "dormant_no_registered_candidate",
        "registered_candidate": None,
        "clean_holdout_status": "not_started",
    }))
    protocol_audit = tmp_path / "protocol_audit.json"
    protocol_audit.write_text(json.dumps({"passed": True}))
    return {
        "zero_shot_spec": {
            "project_root": str(tmp_path),
            "inputs": {
                "v25_completion": "completion.json",
                "v25_audit": "completion_audit.json",
                "v22_protocol": "protocol.json",
                "v22_audit": "protocol_audit.json",
            },
            "release_tag": "tlm-v25",
            "release_commit": "fixture",
            "verify_git_release": False,
            "candidate_family_id": "tlm_zero_shot_cross_asset_v1",
            "target_assets": ["BTC", "ETH", "SOL"],
            "target_symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            "universe": {
                "minimum_assets": 24,
                "maximum_assets": 48,
                "excluded_bases": ["BTC", "ETH", "SOL"],
                "target_proxy_bases": ["WBTC", "WETH", "STETH", "CBETH", "WBETH", "BETH"],
            },
            "data_contract": {
                "source": "fixture",
                "frequency": "1d",
                "development_start": "2021-01-01",
                "development_cutoff": "2026-06-30",
                "raw_fields": ["open", "close"],
                "derived_features": ["return"],
                "target": "next_open_to_open",
                "timestamp_rule": "t to t+1",
            },
            "chronological_splits": {"train": ["2021", "2023"]},
            "architecture": {
                "lookback_days": 256,
                "patch_length_days": 16,
                "patch_stride_days": 8,
                "d_model": 96,
                "encoder_layers": 3,
                "attention_heads": 4,
                "prediction_heads": ["q10", "q50", "q90"],
            },
            "training": {
                "seeds": [42, 7, 123],
                "seed_selection_allowed": False,
                "hyperparameter_search_allowed": False,
            },
            "policy": {"base_cost_bps": 10},
            "source_domain_gates": {
                "cost_bps": [10, 20, 30],
                "primary_control": "dual_momentum_30",
                "block_lengths_days": [7, 21, 63],
                "bootstrap_paths": 10000,
            },
            "authorized_next_action": "v27_non_target_universe_data_audit_only",
        },
        "output_dir": str(tmp_path / "output"),
    }


def test_spec_freezes_design_without_target_evaluation(tmp_path):
    result = build_zero_shot_spec(_config(tmp_path))
    assert result["decision"] == "authorize_v27_non_target_universe_data_audit_only"
    assert not result["tested"]["model_trained"]
    assert result["blueprint"]["target_domain_contract"]["prediction_count"] == 0
    assert result["registration_draft"]["status"] == "incomplete_not_valid_for_v22_registration"
    assert result["audit"]["passed"]


def test_spec_writes_blueprint_registration_draft_and_report(tmp_path):
    config = _config(tmp_path)
    result = run_zero_shot_spec(config)
    output = tmp_path / "output"
    assert result["blueprint_sha256"]
    assert (output / "specification.json").is_file()
    assert (output / "blueprint.json").is_file()
    assert (output / "registration_draft.json").is_file()
    assert (output / "report.md").is_file()
