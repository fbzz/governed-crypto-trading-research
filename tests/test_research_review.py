import json

from tlm.reproducibility import build_reproducibility_manifest
from tlm.research_review import build_research_review, run_research_review


def _config(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "src" / "code.py").write_text("VALUE = 1\n")
    (tmp_path / "tests" / "test.py").write_text("def test_ok(): assert True\n")
    (tmp_path / "configs" / "base.yaml").write_text("seed: 1\n")
    evidence = tmp_path / "evidence"
    evidence.mkdir()

    payloads = {
        "evidence_ledger": {
            "decision": "halt_new_historical_model_search",
            "synthesis": {
                "active_candidate_versions": [],
                "clean_holdout_decision_versions": 0,
            },
        },
        "evidence_audit": {"passed": True},
        "control_certificate": {
            "benchmark_status": "certified_research_control",
            "deployment_status": "not_authorized",
            "control": {"name": "dual_momentum_30"},
            "risk_summary": {
                "worst_observed_max_drawdown": -0.5,
                "bootstrap_cells_with_negative_p05_return": 9,
            },
        },
        "control_audit": {"passed": True},
        "holdout_protocol": {
            "state": "dormant_no_registered_candidate",
            "clean_holdout_status": "not_started",
            "registered_candidate": None,
        },
        "holdout_audit": {"passed": True},
        "reproducibility_audit": {"passed": True},
        "test_result": {"passed": True, "passed_test_count": 10},
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
            "include_files": list(paths.values()),
            "exclude_suffixes": [],
            "packages": ["pytest"],
        }
    }
    manifest = build_reproducibility_manifest(manifest_config)
    manifest_path = evidence / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    paths["reproducibility_manifest"] = str(manifest_path.relative_to(tmp_path))
    return {
        "research_review": {"project_root": str(tmp_path), "inputs": paths},
        "output_dir": str(tmp_path / "output"),
    }


def test_review_blocks_promotion_and_grades_evidence(tmp_path):
    result = build_research_review(_config(tmp_path))
    assert result["decision"] == "no_model_promotion_research_framework_only"
    assert result["deployment_readiness"]["passed_gates"] == 0
    assert result["grades"]["model_deployment_readiness"] == "F"
    assert result["audit"]["passed"]


def test_review_writes_findings_and_report(tmp_path):
    config = _config(tmp_path)
    result = run_research_review(config)
    output = tmp_path / "output"
    assert result["findings"][0]["priority"] == "P0"
    assert (output / "review.json").is_file()
    assert (output / "findings.json").is_file()
    assert (output / "report.md").is_file()
