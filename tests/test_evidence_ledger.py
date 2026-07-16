import json

from tlm.evidence_ledger import build_evidence_ledger, run_evidence_ledger


def _config(tmp_path) -> dict:
    versions = {}
    signal_versions = {"v9", "v11", "v12", "v15", "v19"}
    counts = {"v9": 9, "v11": 18, "v12": 18, "v15": 8, "v19": 8}
    for index in range(1, 20):
        version = f"v{index}"
        audit = tmp_path / f"{version}_audit.json"
        evidence = tmp_path / f"{version}_evidence.json"
        audit.write_text(json.dumps({"passed": True}))
        evidence.write_text(json.dumps({"version": version}))
        entry = {
            "kind": "signal_study" if version in signal_versions else "data_infrastructure",
            "status": "no_robust_signal" if version in signal_versions else "accepted_data_only",
            "decision": "fixture",
            "history_exposed": version in signal_versions,
            "clean_holdout": False,
            "evidence_paths": [str(evidence)],
            "audit_paths": [str(audit)],
        }
        if version in signal_versions:
            result_path = tmp_path / f"{version}_study.json"
            result_path.write_text(json.dumps({
                "signal_count": counts[version], "scenario_count": 3,
                "robust_signals": [], "conclusion": "none",
                "clean_holdout_status": "adaptive_research_only",
            }))
            entry.update({
                "family": version,
                "signal_result_path": str(result_path),
            })
            entry["evidence_paths"].append(str(result_path))
        versions[version] = entry
    return {
        "evidence_ledger": {
            "illustrative_nominal_alpha": 0.05,
            "expected_registered_signals": 61,
            "versions": versions,
        },
        "output_dir": str(tmp_path / "output"),
    }


def test_ledger_counts_all_registered_signal_scenarios(tmp_path):
    result = build_evidence_ledger(_config(tmp_path))
    assert result["synthesis"]["version_count"] == 19
    assert result["synthesis"]["registered_signal_count"] == 61
    assert result["synthesis"]["signal_scenario_evaluations"] == 183
    assert result["decision"] == "halt_new_historical_model_search"


def test_ledger_writes_machine_readable_and_visual_outputs(tmp_path):
    config = _config(tmp_path)
    result = run_evidence_ledger(config)
    output = tmp_path / "output"
    assert result["audit"]["passed"]
    assert (output / "ledger.json").is_file()
    assert (output / "ledger.csv").is_file()
    assert (output / "signal_evidence.png").is_file()
    assert (output / "report.md").is_file()
