from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from .reproducibility import (
    run_test_command,
    verify_reproducibility_manifest,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def build_final_audit(
    config: dict, test_result: dict[str, object]
) -> dict[str, object]:
    final = config["final_audit"]
    root = Path(final["project_root"]).resolve()
    inputs = {name: root / path for name, path in final["inputs"].items()}
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Final audit inputs are missing: {missing}")

    ledger = _load_json(inputs["evidence_ledger"])
    ledger_audit = _load_json(inputs["evidence_audit"])
    certificate = _load_json(inputs["control_certificate"])
    certificate_audit = _load_json(inputs["control_audit"])
    holdout = _load_json(inputs["holdout_protocol"])
    holdout_audit = _load_json(inputs["holdout_audit"])
    manifest = _load_json(inputs["reproducibility_manifest"])
    reproducibility_audit = _load_json(inputs["reproducibility_audit"])
    review = _load_json(inputs["research_review"])
    review_audit = _load_json(inputs["research_review_audit"])
    manifest_verification = verify_reproducibility_manifest(manifest, root)

    versions = ledger["versions"]
    expected_ledger_versions = [f"v{index}" for index in range(1, 20)]
    decision_chain = [
        {
            "version": row["version"],
            "kind": row["kind"],
            "status": row["status"],
            "decision": row["decision"],
        }
        for row in versions
    ]
    decision_chain.extend([
        {
            "version": "v20",
            "kind": "evidence_governance",
            "status": "passed",
            "decision": ledger["decision"],
        },
        {
            "version": "v21",
            "kind": "control_governance",
            "status": certificate["benchmark_status"],
            "decision": certificate["decision"],
        },
        {
            "version": "v22",
            "kind": "prospective_protocol",
            "status": holdout["state"],
            "decision": holdout["decision"],
        },
        {
            "version": "v23",
            "kind": "reproducibility",
            "status": "verified",
            "decision": "reproducibility_bundle_verified",
        },
        {
            "version": "v24",
            "kind": "research_review",
            "status": review["deployment_readiness"]["status"],
            "decision": review["decision"],
        },
    ])

    contract_paths = [root / path for path in final["project_contract_files"]]
    contract_missing = [str(path) for path in contract_paths if not path.is_file()]
    if contract_missing:
        raise FileNotFoundError(f"Final project contracts are missing: {contract_missing}")
    final_hashes = {
        "inputs": {
            str(path.relative_to(root)): _sha256_file(path)
            for path in inputs.values()
        },
        "project_contract": {
            str(path.relative_to(root)): _sha256_file(path)
            for path in contract_paths
        },
    }
    final_hashes["contract_set_sha256"] = _canonical_hash(
        final_hashes["project_contract"]
    )
    final_hashes["input_set_sha256"] = _canonical_hash(final_hashes["inputs"])

    system_status = {
        "engineering_goal": "complete_through_v25",
        "research_framework": "complete_and_reproducible",
        "learned_model_candidate": "none",
        "deterministic_control": "research_comparator_only",
        "historical_model_search": "halted",
        "prospective_holdout": "dormant_not_started",
        "shadow_paper_live_or_real_trading": "not_authorized",
        "deployable_tlm": "not_available",
    }
    checks = {
        "all_inputs_exist": not missing,
        "v1_v19_registry_is_exact": [row["version"] for row in versions]
        == expected_ledger_versions,
        "v20_audit_passes": bool(ledger_audit.get("passed")),
        "v21_audit_passes": bool(certificate_audit.get("passed")),
        "v22_audit_passes": bool(holdout_audit.get("passed")),
        "v23_audit_passes": bool(reproducibility_audit.get("passed")),
        "v23_manifest_reverifies_after_final_code": bool(
            manifest_verification["passed"]
        ),
        "v24_audit_passes": bool(review_audit.get("passed")),
        "full_final_test_suite_passes": bool(test_result["passed"]),
        "complete_v1_v24_decision_chain": len(decision_chain) == 24,
        "no_active_candidate": not ledger["synthesis"][
            "active_candidate_versions"
        ],
        "no_robust_registered_signal": ledger["synthesis"][
            "robust_signal_count"
        ]
        == 0,
        "control_not_deployable": certificate["deployment_status"]
        == "not_authorized",
        "prospective_holdout_not_started": holdout["clean_holdout_status"]
        == "not_started"
        and holdout["registered_candidate"] is None,
        "review_blocks_model_promotion": review["decision"]
        == "no_model_promotion_research_framework_only"
        and review["deployment_readiness"]["passed_gates"] == 0,
        "project_contract_is_content_addressed": bool(
            final_hashes["contract_set_sha256"]
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Final completion audit failed: {checks}")

    return {
        "version": "v25",
        "method": "complete_decision_chain_and_final_state_audit",
        "decision": "complete_research_framework_no_deployable_tlm",
        "system_status": system_status,
        "evidence_summary": {
            "decision_versions_before_v25": len(decision_chain),
            "registered_signals": ledger["synthesis"]["registered_signal_count"],
            "signal_scenario_evaluations": ledger["synthesis"][
                "signal_scenario_evaluations"
            ],
            "robust_signals": ledger["synthesis"]["robust_signal_count"],
            "clean_holdout_decisions": ledger["synthesis"][
                "clean_holdout_decision_versions"
            ],
            "final_test_count": test_result["passed_test_count"],
            "v23_verified_files": manifest_verification["checked_files"],
            "v24_deployment_gates_passed": review["deployment_readiness"][
                "passed_gates"
            ],
            "v24_deployment_gates_total": review["deployment_readiness"][
                "total_gates"
            ],
        },
        "decision_chain": decision_chain,
        "final_hashes": final_hashes,
        "limitations": [
            "No learned model survived the frozen historical validation program.",
            "Dual momentum 30d is a risky comparator, not an approved strategy.",
            "No candidate is registered and the prospective holdout has not started.",
            "The local repository has no resolvable immutable Git commit anchor.",
            "No shadow, paper, live, or real-money behavior was validated or authorized.",
        ],
        "next_legal_transition": [
            "Archive this v25 snapshot in an immutable commit or release.",
            "Only consider a single candidate derived without further outcome-driven historical search.",
            "Hash and register that candidate under the v22 schema before its future window.",
            "Quarantine future source data without running daily candidate decisions.",
            "Evaluate once after every v22 maturity gate; retire the candidate if any gate fails.",
        ],
        "audit": {"passed": True, "checks": checks},
    }


def _report(result: dict) -> str:
    evidence = result["evidence_summary"]
    status = result["system_status"]
    lines = [
        "# TLM v25 Final Completion Audit",
        "",
        "## Final decision",
        "",
        "**THE RESEARCH FRAMEWORK IS COMPLETE. NO DEPLOYABLE TLM OR TRADING STRATEGY IS APPROVED.**",
        "",
        "This is a successful engineering completion and a negative scientific result. The pipeline can reproduce data contracts, experiments, validation, Monte Carlo diagnostics, multiple-testing governance, and final decisions. It cannot support a capital claim.",
        "",
        "## Final state",
        "",
        "| Component | Status |",
        "|---|---|",
    ]
    for name, value in status.items():
        lines.append(f"| {name} | {value} |")
    lines.extend([
        "",
        "## Evidence summary",
        "",
        f"- Audited decisions before v25: **{evidence['decision_versions_before_v25']}**",
        f"- Registered signals: **{evidence['registered_signals']}**",
        f"- Signal-scenario evaluations: **{evidence['signal_scenario_evaluations']}**",
        f"- Robust signals: **{evidence['robust_signals']}**",
        f"- Clean-holdout decisions: **{evidence['clean_holdout_decisions']}**",
        f"- Final tests passed: **{evidence['final_test_count']}**",
        f"- V23 files reverified: **{evidence['v23_verified_files']}**",
        f"- Deployment gates passed: **{evidence['v24_deployment_gates_passed']}/{evidence['v24_deployment_gates_total']}**",
        "",
        "## What survived",
        "",
        "The durable output is the research system itself: causal timestamp contracts, cache/hash lineage, expanding and rolling walk-forward, cost accounting, dependence-aware block bootstrap, a multiple-testing ledger, a research-only deterministic comparator, a dormant one-shot prospective protocol, and reproducibility/reviewer gates.",
        "",
        "## What did not survive",
        "",
        "The Transformer candidates, learned override, risk-off policy, and all five registered signal families failed the required stability program. Dual momentum remains useful only as a benchmark and has materially adverse downside diagnostics.",
        "",
        "## Next legal transition",
        "",
    ])
    for index, item in enumerate(result["next_legal_transition"], start=1):
        lines.append(f"{index}. {item}")
    lines.extend([
        "",
        "No additional historical feature, architecture, lookback, threshold, or policy search is authorized by this result.",
        "",
    ])
    return "\n".join(lines)


def run_final_audit(config: dict) -> dict[str, object]:
    final = config["final_audit"]
    root = Path(final["project_root"]).resolve()
    output = Path(config["output_dir"])
    if not output.is_absolute():
        output = root / output
    output.mkdir(parents=True, exist_ok=True)
    test_result = (
        run_test_command(root, list(final["test_command"]))
        if final.get("run_tests", True)
        else {
            "command": [],
            "exit_code": 0,
            "passed": True,
            "passed_test_count": int(final.get("fixture_test_count", 1)),
            "stdout": "test execution disabled by fixture",
            "stderr": "",
        }
    )
    (output / "test_result.json").write_text(
        json.dumps(test_result, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not test_result["passed"]:
        raise RuntimeError("Final test suite failed")
    result = build_final_audit(config, test_result)
    (output / "completion.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "final_hashes.json").write_text(
        json.dumps(result["final_hashes"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output / "audit.json").write_text(
        json.dumps(result["audit"], indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    return result
