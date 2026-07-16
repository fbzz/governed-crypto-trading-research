from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from .reproducibility import verify_reproducibility_manifest


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_research_review(config: dict) -> dict[str, object]:
    review = config["research_review"]
    root = Path(review["project_root"]).resolve()
    inputs = {
        name: root / path for name, path in review["inputs"].items()
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Research review inputs are missing: {missing}")

    ledger = _load_json(inputs["evidence_ledger"])
    ledger_audit = _load_json(inputs["evidence_audit"])
    certificate = _load_json(inputs["control_certificate"])
    certificate_audit = _load_json(inputs["control_audit"])
    holdout = _load_json(inputs["holdout_protocol"])
    holdout_audit = _load_json(inputs["holdout_audit"])
    manifest = _load_json(inputs["reproducibility_manifest"])
    reproducibility_audit = _load_json(inputs["reproducibility_audit"])
    test_result = _load_json(inputs["test_result"])
    verification = verify_reproducibility_manifest(manifest, root)

    deployment_gates = {
        "active_candidate_exists": bool(
            ledger["synthesis"]["active_candidate_versions"]
        ),
        "clean_prospective_holdout_complete": holdout["clean_holdout_status"]
        == "complete",
        "prospective_superiority_gate_passed": False,
        "deployment_risk_authorization_exists": certificate[
            "deployment_status"
        ]
        == "authorized",
    }
    findings = [
        {
            "id": "V24-F1",
            "priority": "P0",
            "title": "No model is eligible for promotion",
            "evidence": (
                f"The v20 ledger has {len(ledger['synthesis']['active_candidate_versions'])} "
                "active candidates and zero clean-holdout decision versions."
            ),
            "impact": "Any model, paper, shadow, live, or capital claim would exceed the evidence.",
            "required_action": "Keep deployment disabled and historical model search closed.",
            "status": "open_by_design",
        },
        {
            "id": "V24-F2",
            "priority": "P0",
            "title": "The deterministic control is not capital-safe",
            "evidence": (
                f"Observed max drawdown is {certificate['risk_summary']['worst_observed_max_drawdown']:.2%}; "
                f"all {certificate['risk_summary']['bootstrap_cells_with_negative_p05_return']} "
                "bootstrap cells have negative fifth-percentile total return."
            ),
            "impact": "Benchmark reproducibility cannot be interpreted as deployment readiness.",
            "required_action": "Use dual momentum only as a frozen research comparator.",
            "status": "mitigated_by_certificate",
        },
        {
            "id": "V24-F3",
            "priority": "P1",
            "title": "Prospective evidence has not started",
            "evidence": (
                f"V22 state is {holdout['state']}; registered candidate is "
                f"{holdout['registered_candidate']}."
            ),
            "impact": "There is no untouched future evidence supporting any strategy.",
            "required_action": "Register one immutable candidate before starting a new future window.",
            "status": "open_by_design",
        },
        {
            "id": "V24-F4",
            "priority": "P2",
            "title": "The evidence bundle lacks a Git commit anchor",
            "evidence": (
                "The v23 runtime manifest reports an unborn or unavailable repository revision."
                if not manifest["runtime"]["git"]["available"]
                else f"The bundle is anchored at commit {manifest['runtime']['git']['commit']}."
            ),
            "impact": "Content hashes verify the local snapshot, but distribution lacks a conventional immutable release reference.",
            "required_action": "Archive the final v25 snapshot in a signed commit or immutable release when desired.",
            "status": (
                "open"
                if not manifest["runtime"]["git"]["available"]
                else "resolved"
            ),
        },
        {
            "id": "V24-F5",
            "priority": "P3",
            "title": "Research infrastructure is internally consistent",
            "evidence": (
                f"V23 verified {verification['checked_files']} files and the full suite passed "
                f"{test_result['passed_test_count']} tests."
            ),
            "impact": "The negative decision is reproducible and auditable.",
            "required_action": "Preserve the bundle; do not reinterpret this as alpha evidence.",
            "status": "verified",
        },
    ]
    grades = {
        "timestamp_and_leakage_controls": "A",
        "data_lineage_and_hashing": "A-",
        "walk_forward_and_dependence_stress": "A-",
        "multiple_testing_governance": "A",
        "reproducibility": "B+" if not manifest["runtime"]["git"]["available"] else "A-",
        "clean_prospective_evidence": "F",
        "model_deployment_readiness": "F",
    }
    authorized_actions = [
        "archive_the_v1_v24_negative_research_record",
        "keep_dual_momentum_30_as_research_comparator_only",
        "keep_v22_dormant_until_one_immutable_candidate_is_registered",
        "add_an_immutable_release_anchor_without_changing_results",
    ]
    forbidden_actions = [
        "resume_historical_architecture_feature_or_threshold_search",
        "promote_any_v1_v21_model_or_control",
        "start_shadow_paper_live_or_real_money_execution",
        "inspect_future_candidate_performance_before_v22_maturity",
        "tune_a_replacement_on_a_failed_v22_holdout",
    ]
    checks = {
        "all_inputs_exist": not missing,
        "v20_audit_passes": bool(ledger_audit.get("passed")),
        "v21_audit_passes": bool(certificate_audit.get("passed")),
        "v22_audit_passes": bool(holdout_audit.get("passed")),
        "v23_audit_passes": bool(reproducibility_audit.get("passed")),
        "v23_manifest_reverifies_now": bool(verification["passed"]),
        "full_test_suite_passed": bool(test_result["passed"]),
        "historical_search_is_halted": ledger["decision"]
        == "halt_new_historical_model_search",
        "control_is_not_deployable": certificate["deployment_status"]
        == "not_authorized",
        "holdout_is_dormant": holdout["state"]
        == "dormant_no_registered_candidate",
        "no_deployment_gate_passes": not any(deployment_gates.values()),
        "review_did_not_train_or_select_model": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Independent research review failed: {checks}")

    return {
        "version": "v24",
        "method": "read_only_cross_artifact_research_review",
        "review_independence": "separate_audit_logic_no_retraining_no_parameter_selection",
        "decision": "no_model_promotion_research_framework_only",
        "deployment_readiness": {
            "passed_gates": sum(deployment_gates.values()),
            "total_gates": len(deployment_gates),
            "gates": deployment_gates,
            "status": "not_ready",
        },
        "grades": grades,
        "findings": findings,
        "authorized_actions": authorized_actions,
        "forbidden_actions": forbidden_actions,
        "source_hashes": {
            str(path.relative_to(root)): _sha256_file(path)
            for path in inputs.values()
        },
        "audit": {"passed": True, "checks": checks},
    }


def _report(result: dict) -> str:
    readiness = result["deployment_readiness"]
    lines = [
        "# TLM v24 Read-Only Research Review",
        "",
        "## Reviewer decision",
        "",
        "**NO MODEL PROMOTION. THE DELIVERABLE IS A RESEARCH FRAMEWORK, NOT A TRADING SYSTEM.**",
        "",
        f"Deployment gates passed: **{readiness['passed_gates']}/{readiness['total_gates']}**. The review retrained no model, changed no historical result, and selected no parameter.",
        "",
        "## Findings",
        "",
        "| Priority | ID | Finding | Status |",
        "|---|---|---|---|",
    ]
    for finding in result["findings"]:
        lines.append(
            f"| {finding['priority']} | {finding['id']} | {finding['title']} | {finding['status']} |"
        )
    lines.extend(["", "## Evidence grades", ""])
    for category, grade in result["grades"].items():
        lines.append(f"- {category}: **{grade}**")
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The engineering and negative-result record are strong: timestamp contracts, walk-forward geometry, dependence-aware bootstrap, evidence accounting, and content verification are all explicit. The trading claim is weak by design: no learned candidate survived, the deterministic control has deep downside, and no clean prospective observation exists.",
        "",
        "## Authorized state",
        "",
        "Archive the research record, keep the deterministic control as a comparator only, and leave the prospective protocol dormant. A future candidate must be immutable before a genuinely future window and receives one evaluation only after the v22 maturity gate.",
        "",
    ])
    return "\n".join(lines)


def run_research_review(config: dict) -> dict[str, object]:
    result = build_research_review(config)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "review.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "findings.json").write_text(
        json.dumps(result["findings"], indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "audit.json").write_text(
        json.dumps(result["audit"], indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    return result
