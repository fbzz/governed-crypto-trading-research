import json

from tlm.final_audit import build_final_audit, run_final_audit
from tlm.reproducibility import build_reproducibility_manifest


def _config(tmp_path):
    for directory in ("src", "tests", "configs", "evidence"):
        (tmp_path / directory).mkdir()
    (tmp_path / "src" / "code.py").write_text("VALUE = 1\n")
    (tmp_path / "tests" / "test.py").write_text("def test_ok(): assert True\n")
    (tmp_path / "configs" / "base.yaml").write_text("seed: 1\n")
    for name in ("AGENTS.md", "TASKS.md", "README.md", "Makefile", "pyproject.toml"):
        (tmp_path / name).write_text(f"{name}\n")

    evidence = tmp_path / "evidence"
    versions = [
        {
            "version": f"v{index}",
            "kind": "fixture",
            "status": "rejected",
            "decision": "fixture",
        }
        for index in range(1, 20)
    ]
    payloads = {
        "evidence_ledger": {
            "decision": "halt_new_historical_model_search",
            "versions": versions,
            "synthesis": {
                "active_candidate_versions": [],
                "registered_signal_count": 61,
                "signal_scenario_evaluations": 183,
                "robust_signal_count": 0,
                "clean_holdout_decision_versions": 0,
            },
        },
        "evidence_audit": {"passed": True},
        "control_certificate": {
            "benchmark_status": "certified_research_control",
            "deployment_status": "not_authorized",
            "decision": "certify_as_research_control_only",
        },
        "control_audit": {"passed": True},
        "holdout_protocol": {
            "state": "dormant_no_registered_candidate",
            "clean_holdout_status": "not_started",
            "registered_candidate": None,
            "decision": "freeze_protocol_but_do_not_start_without_candidate",
        },
        "holdout_audit": {"passed": True},
        "reproducibility_audit": {"passed": True},
        "research_review": {
            "decision": "no_model_promotion_research_framework_only",
            "deployment_readiness": {
                "status": "not_ready",
                "passed_gates": 0,
                "total_gates": 4,
            },
        },
        "research_review_audit": {"passed": True},
    }
    paths = {}
    for name, payload in payloads.items():
        path = evidence / f"{name}.json"
        path.write_text(json.dumps(payload))
        paths[name] = str(path.relative_to(tmp_path))

    manifest_config = {
        "reproducibility_bundle": {
            "project_root": str(tmp_path),
            "include_roots": ["src", "tests", "configs"],
            "include_files": [str((evidence / "evidence_audit.json").relative_to(tmp_path))],
            "exclude_suffixes": [],
            "packages": ["pytest"],
        }
    }
    manifest = build_reproducibility_manifest(manifest_config)
    manifest_path = evidence / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    paths["reproducibility_manifest"] = str(manifest_path.relative_to(tmp_path))
    return {
        "final_audit": {
            "project_root": str(tmp_path),
            "inputs": paths,
            "project_contract_files": [
                "AGENTS.md", "TASKS.md", "README.md", "Makefile", "pyproject.toml"
            ],
            "test_command": ["python3", "-m", "pytest"],
            "run_tests": False,
            "fixture_test_count": 10,
        },
        "output_dir": "output",
    }


def test_final_audit_distinguishes_engineering_from_trading(tmp_path):
    config = _config(tmp_path)
    result = build_final_audit(config, {"passed": True, "passed_test_count": 10})
    assert result["decision"] == "complete_research_framework_no_deployable_tlm"
    assert result["system_status"]["engineering_goal"] == "complete_through_v25"
    assert result["system_status"]["deployable_tlm"] == "not_available"
    assert result["audit"]["passed"]


def test_final_audit_writes_completion_and_hashes(tmp_path):
    config = _config(tmp_path)
    result = run_final_audit(config)
    output = tmp_path / "output"
    assert result["evidence_summary"]["decision_versions_before_v25"] == 24
    assert (output / "completion.json").is_file()
    assert (output / "final_hashes.json").is_file()
    assert (output / "report.md").is_file()
