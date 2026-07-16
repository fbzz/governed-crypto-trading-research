from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tlm-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml


ACTIVE_STATUSES = {"accepted_candidate", "active_candidate", "promoted"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_passed(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "passed" in payload:
        return bool(payload["passed"])
    if isinstance(payload.get("audit"), dict):
        return bool(payload["audit"].get("passed"))
    raise ValueError(f"Unrecognized audit schema: {path}")


def build_evidence_ledger(config: dict) -> dict[str, object]:
    registry = config["evidence_ledger"]["versions"]
    expected_versions = [f"v{index}" for index in range(1, 20)]
    if list(registry) != expected_versions:
        raise ValueError("Evidence registry must list v1 through v19 in order")
    versions: list[dict[str, object]] = []
    signal_families: list[dict[str, object]] = []
    missing_files: list[str] = []
    failed_audits: list[str] = []
    for version, entry in registry.items():
        evidence_paths = [Path(path) for path in entry["evidence_paths"]]
        audit_paths = [Path(path) for path in entry["audit_paths"]]
        all_paths = evidence_paths + audit_paths
        for path in all_paths:
            if not path.is_file():
                missing_files.append(str(path))
        if missing_files:
            continue
        for path in audit_paths:
            if not _audit_passed(path):
                failed_audits.append(str(path))
        hashes = {str(path): _sha256_file(path) for path in all_paths}
        row: dict[str, object] = {
            "version": version,
            "kind": entry["kind"],
            "status": entry["status"],
            "decision": entry["decision"],
            "history_exposed": bool(entry["history_exposed"]),
            "clean_holdout": bool(entry["clean_holdout"]),
            "evidence_hashes": hashes,
        }
        if entry.get("superseded_by"):
            row["superseded_by"] = entry["superseded_by"]
        if entry["kind"] == "signal_study":
            result_path = Path(entry["signal_result_path"])
            result = json.loads(result_path.read_text(encoding="utf-8"))
            family = {
                "version": version,
                "family": entry["family"],
                "signal_count": int(result["signal_count"]),
                "scenario_count": int(result["scenario_count"]),
                "signal_scenario_evaluations": int(
                    result["signal_count"] * result["scenario_count"]
                ),
                "robust_signal_count": len(result["robust_signals"]),
                "robust_signals": result["robust_signals"],
                "conclusion": result["conclusion"],
                "clean_holdout_status": result.get(
                    "clean_holdout_status", "not_recorded"
                ),
            }
            signal_families.append(family)
            row["signal_family"] = family
        versions.append(row)
    if missing_files:
        raise FileNotFoundError(f"Evidence ledger is missing files: {missing_files}")
    if failed_audits:
        raise RuntimeError(f"Evidence ledger found failed audits: {failed_audits}")

    registered_signals = sum(row["signal_count"] for row in signal_families)
    signal_scenario_evaluations = sum(
        row["signal_scenario_evaluations"] for row in signal_families
    )
    robust_signals = sum(row["robust_signal_count"] for row in signal_families)
    model_policy_versions = sum(
        row["kind"] in {"model_experiment", "validation_suite"} for row in versions
    )
    infrastructure_versions = sum(
        row["kind"] in {"data_infrastructure", "source_feasibility"}
        for row in versions
    )
    exposed_versions = sum(bool(row["history_exposed"]) for row in versions)
    clean_holdout_versions = sum(bool(row["clean_holdout"]) for row in versions)
    active_versions = [
        row["version"] for row in versions if row["status"] in ACTIVE_STATUSES
    ]
    alpha = float(config["evidence_ledger"]["illustrative_nominal_alpha"])
    independence_warning = 1.0 - (1.0 - alpha) ** registered_signals
    synthesis = {
        "version_count": len(versions),
        "model_or_policy_versions": model_policy_versions,
        "source_or_data_versions": infrastructure_versions,
        "signal_family_count": len(signal_families),
        "registered_signal_count": registered_signals,
        "signal_scenario_evaluations": signal_scenario_evaluations,
        "robust_signal_count": robust_signals,
        "history_exposed_decision_versions": exposed_versions,
        "clean_holdout_decision_versions": clean_holdout_versions,
        "active_candidate_versions": active_versions,
        "illustrative_any_false_positive_probability_if_independent": (
            independence_warning
        ),
        "illustrative_probability_is_inferential": False,
    }
    decision = (
        "halt_new_historical_model_search"
        if robust_signals == 0 and not active_versions
        else "review_registered_survivor_before_any_new_search"
    )
    checks = {
        "all_v1_v19_registered": len(versions) == 19,
        "all_evidence_files_exist": not missing_files,
        "all_registered_audits_pass": not failed_audits,
        "signal_counts_match_frozen_results": registered_signals == int(
            config["evidence_ledger"]["expected_registered_signals"]
        ),
        "no_unrecorded_robust_signal": robust_signals == 0,
        "no_active_candidate": not active_versions,
        "historical_search_halted_after_negative_ledger": decision
        == "halt_new_historical_model_search",
    }
    if not all(checks.values()):
        raise RuntimeError(f"Evidence ledger audit failed: {checks}")
    return {
        "version": "v20",
        "method": "artifact_verified_multiple_testing_and_exposure_ledger",
        "decision": decision,
        "synthesis": synthesis,
        "signal_families": signal_families,
        "versions": versions,
        "audit": {"passed": True, "checks": checks},
    }


def _plot_signal_families(result: dict, output: Path) -> None:
    families = result["signal_families"]
    labels = [row["family"] for row in families]
    registered = [row["signal_count"] for row in families]
    robust = [row["robust_signal_count"] for row in families]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(labels, registered, label="registered", color="#4472C4")
    axis.bar(labels, robust, label="robust", color="#C00000")
    axis.set_ylabel("signals")
    axis.set_title("TLM v20 registered versus robust signals")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "signal_evidence.png", dpi=150)
    plt.close(figure)


def _report(result: dict) -> str:
    summary = result["synthesis"]
    lines = [
        "# TLM v20 Evidence and Multiple-Testing Ledger",
        "",
        "## Decision",
        "",
        "**HALT NEW HISTORICAL MODEL SEARCH.** No registered signal family produced a robust survivor, and no learned candidate remains active after extended validation.",
        "",
        "## Exposure summary",
        "",
        f"- Versioned decisions audited: **{summary['version_count']}**",
        f"- Model/policy experiment versions: **{summary['model_or_policy_versions']}**",
        f"- Source/data infrastructure versions: **{summary['source_or_data_versions']}**",
        f"- Policy-free signal families: **{summary['signal_family_count']}**",
        f"- Registered signals: **{summary['registered_signal_count']}**",
        f"- Signal-scenario evaluations: **{summary['signal_scenario_evaluations']}**",
        f"- Robust signals: **{summary['robust_signal_count']}**",
        f"- History-exposed decision versions: **{summary['history_exposed_decision_versions']}**",
        f"- Clean-holdout decision versions: **{summary['clean_holdout_decision_versions']}**",
        "",
        "At a nominal 5% per-signal rate, 61 independent trials would imply a {:.1%} chance of at least one false positive. This is only an illustration: the signals are correlated, so it is not used as a p-value or decision statistic.".format(
            summary["illustrative_any_false_positive_probability_if_independent"]
        ),
        "",
        "## Signal-family ledger",
        "",
        "| Version | Family | Signals | Scenarios | Evaluations | Robust | Conclusion |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in result["signal_families"]:
        lines.append(
            f"| {row['version']} | {row['family']} | {row['signal_count']} | "
            f"{row['scenario_count']} | {row['signal_scenario_evaluations']} | "
            f"{row['robust_signal_count']} | {row['conclusion']} |"
        )
    lines.extend([
        "",
        "## Scientific interpretation",
        "",
        "The original Transformer passed one sparse three-fold slice and was suspended when six-fold expanding and rolling validation were added. Subsequent learned overrides failed. Five independently versioned signal families then tested OHLCV, daily derivatives, intraday derivatives paths, options volatility, and Treasury rates; all produced zero robust signals under the same stability and block-bootstrap gates.",
        "",
        "Dual momentum remains a deterministic research control, not a promoted candidate. The same historical return window has informed many choices, so another historical architecture or threshold experiment would increase selection bias without new clean evidence.",
        "",
        "## Authorized next step",
        "",
        "V21 may characterize and certify the deterministic control as a benchmark only. It must not tune the lookback or claim deployability. V22 must freeze a prospective holdout protocol; no historical model search resumes before that protocol produces enough untouched observations.",
        "",
    ])
    return "\n".join(lines)


def run_evidence_ledger(config: dict) -> dict[str, object]:
    result = build_evidence_ledger(config)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    _plot_signal_families(result, output)
    rows = []
    for version in result["versions"]:
        rows.append({
            "version": version["version"],
            "kind": version["kind"],
            "status": version["status"],
            "history_exposed": version["history_exposed"],
            "clean_holdout": version["clean_holdout"],
            "decision": version["decision"],
        })
    pd.DataFrame(rows).to_csv(output / "ledger.csv", index=False)
    (output / "ledger.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "audit.json").write_text(
        json.dumps(result["audit"], indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    return result
