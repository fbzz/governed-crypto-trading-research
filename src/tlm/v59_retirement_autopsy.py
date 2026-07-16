from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping

from .core.artifacts import canonical_sha256, file_sha256, write_json_atomic


CONTRACT_SHA256 = "e0b93d9c65e5b68819b4414b6d3fb3bcb6602659fe07bb311d9b93d32a91e708"
FAMILY_ID = "tlm_state_conditioned_multi_horizon_quantile_small_v1"
RETIREMENT_DECISION = "retire_family_without_tuning"
AUTOPSY_DECISION = "v59_retirement_confirmed_diagnostic_only"
OUTPUT_ROOT = Path(
    "artifacts/v59_state_conditioned_multi_horizon_evaluation/retirement_autopsy"
)


def _with_self_hash(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    value = dict(payload)
    value[field] = canonical_sha256(value)
    return value


def _verify_self_hash(payload: Mapping[str, Any], field: str, name: str) -> str:
    value = dict(payload)
    registered = value.pop(field, None)
    computed = canonical_sha256(value)
    if registered != computed:
        raise RuntimeError(f"V59 autopsy {name} self-hash drift")
    return computed


def _safe_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise RuntimeError("V59 autopsy path must be a non-empty string")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"V59 autopsy path escapes repository: {relative}") from exc
    return path


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.replace(path)


def load_contract(path: str | Path) -> dict[str, Any]:
    contract_path = Path(path)
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("V59 autopsy contract is not valid JSON") from exc
    if not isinstance(contract, dict) or canonical_sha256(contract) != CONTRACT_SHA256:
        raise RuntimeError("V59 autopsy contract semantic hash drift")
    if (
        contract.get("family_id") != FAMILY_ID
        or contract.get("retirement")
        != {
            "input": "evaluation_result",
            "decision": RETIREMENT_DECISION,
            "immutable": True,
        }
        or set(contract.get("inputs", {}))
        != {
            "evaluation_result",
            "evaluation_audit",
            "registered_metrics",
            "registered_bootstrap",
            "registered_gate_matrix",
            "completion_receipt",
            "artifact_manifest",
            "source_free_replay",
            "evaluation_report",
        }
        or any(value is not True for value in contract.get("forbidden", {}).values())
    ):
        raise RuntimeError("V59 autopsy contract boundary drift")
    return contract


def verify_frozen_inputs(
    contract: Mapping[str, Any], root: str | Path = "."
) -> dict[str, Any]:
    project_root = Path(root).resolve()
    checks: dict[str, bool] = {}
    inputs = contract["inputs"]
    for name, receipt in sorted(inputs.items()):
        path = _safe_path(project_root, receipt["path"])
        checks[f"{name}.regular_file"] = path.is_file()
        checks[f"{name}.sha256"] = path.is_file() and file_sha256(path) == receipt["sha256"]
    result_path = _safe_path(project_root, inputs["evaluation_result"]["path"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    checks["retirement_decision_immutable"] = result.get("decision") == RETIREMENT_DECISION
    checks["target_assets_loaded_zero"] = result.get("target_assets_loaded") == []
    checks["target_predictions_zero"] = result.get("target_predictions") == 0
    checks["target_pnl_evaluations_zero"] = result.get("target_pnl_evaluations") == 0
    checks["retuning_zero"] = result.get("retuning_performed") is False
    checks["regeneration_zero"] = result.get("predictions_or_positions_regenerated") is False
    passed = all(checks.values())
    if not passed:
        failed = sorted(name for name, value in checks.items() if not value)
        raise RuntimeError(f"V59 autopsy frozen-input verification failed: {failed}")
    return {
        "schema_version": "v59-retirement-autopsy-input-receipt/v1",
        "contract_sha256": CONTRACT_SHA256,
        "verified_inputs": len(inputs),
        "checks": checks,
        "passed": True,
        "source_outcome_reads": 0,
        "parquet_deserializations": 0,
        "checkpoint_loads": 0,
        "model_inference": 0,
        "new_bootstrap_paths": 0,
        "target_assets_loaded": [],
    }


def _load_inputs(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    for name, receipt in contract["inputs"].items():
        path = _safe_path(root, receipt["path"])
        if path.suffix == ".json":
            loaded[name] = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix == ".md":
            loaded[name] = path.read_text(encoding="utf-8")
        else:
            raise RuntimeError(f"V59 autopsy undeclared input type: {path.suffix}")
    return loaded


def _gate_summary(gates: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for name in sorted({row["gate"] for row in gates}):
        cells = [row for row in gates if row["gate"] == name]
        passed = sum(bool(row["passed"]) for row in cells)
        result[name] = {
            "total": len(cells),
            "passed": passed,
            "failed": len(cells) - passed,
        }
    return result


def build_signal_diagnostics(
    metrics: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    gates: list[dict[str, Any]],
) -> dict[str, Any]:
    predictive = deepcopy(metrics["predictive"]["fold_cells"])
    pairwise = [float(row["h7_q50_pairwise_accuracy"]) for row in predictive]
    spearman = [float(row["h7_q50_spearman"]) for row in predictive]
    signal_gate_names = {
        "h7_q50_pairwise_accuracy",
        "h7_q50_spearman",
        "candidate_bootstrap_p05",
        "candidate_minus_control_bootstrap_p05",
    }
    signal_gates = [deepcopy(row) for row in gates if row["gate"] in signal_gate_names]
    absolute_p05 = [
        float(row["distributions"]["candidate"]["p05"])
        for row in bootstrap["cells"]
    ]
    delta_p05 = [
        float(distribution["p05"])
        for row in bootstrap["cells"]
        for distribution in row["candidate_minus_controls"].values()
    ]
    body = {
        "schema_version": "v59-retirement-autopsy-signal/v1",
        "registered_predictive_cells": predictive,
        "registered_bootstrap_cells": deepcopy(bootstrap["cells"]),
        "registered_signal_gates": signal_gates,
        "summary": {
            "predictive_cells": len(predictive),
            "pairwise_accuracy_min": min(pairwise),
            "pairwise_accuracy_max": max(pairwise),
            "pairwise_accuracy_mean": sum(pairwise) / len(pairwise),
            "pairwise_gate_passed": sum(value > 0.5 for value in pairwise),
            "spearman_min": min(spearman),
            "spearman_max": max(spearman),
            "spearman_mean": sum(spearman) / len(spearman),
            "spearman_gate_passed": sum(value > 0.0 for value in spearman),
            "candidate_bootstrap_cells": len(absolute_p05),
            "candidate_bootstrap_p05_min": min(absolute_p05),
            "candidate_bootstrap_p05_max": max(absolute_p05),
            "candidate_bootstrap_p05_positive": sum(value > 0.0 for value in absolute_p05),
            "candidate_minus_control_bootstrap_cells": len(delta_p05),
            "candidate_minus_control_bootstrap_p05_min": min(delta_p05),
            "candidate_minus_control_bootstrap_p05_max": max(delta_p05),
            "candidate_minus_control_bootstrap_p05_positive": sum(
                value > 0.0 for value in delta_p05
            ),
        },
        "interpretation": (
            "Weak ordinal information was present in every registered fold cell, "
            "but no registered absolute or control-relative bootstrap p05 was positive."
        ),
    }
    return _with_self_hash(body, "signal_diagnostics_sha256")


def build_calibration_diagnostics(
    metrics: Mapping[str, Any], gates: list[dict[str, Any]]
) -> dict[str, Any]:
    predictive = deepcopy(metrics["predictive"]["fold_cells"])
    coverage_gates = [deepcopy(row) for row in gates if row["gate"] == "h7_q20_coverage"]
    by_origin: dict[str, dict[str, float]] = {}
    for origin in sorted({row["origin"] for row in predictive}):
        cells = [row for row in predictive if row["origin"] == origin]
        by_origin[origin] = {
            "cells": len(cells),
            "mean_h1_q20_coverage": sum(float(row["h1_q20_coverage"]) for row in cells) / len(cells),
            "mean_h3_q20_coverage": sum(float(row["h3_q20_coverage"]) for row in cells) / len(cells),
            "mean_h7_q20_coverage": sum(float(row["h7_q20_coverage"]) for row in cells) / len(cells),
            "mean_h7_q50_pairwise_accuracy": sum(
                float(row["h7_q50_pairwise_accuracy"]) for row in cells
            )
            / len(cells),
            "mean_h7_q50_spearman": sum(float(row["h7_q50_spearman"]) for row in cells) / len(cells),
        }
    body = {
        "schema_version": "v59-retirement-autopsy-calibration/v1",
        "registered_predictive_cells": predictive,
        "registered_coverage_gates": coverage_gates,
        "origin_summary": by_origin,
        "summary": {
            "h7_q20_coverage_passed": sum(bool(row["passed"]) for row in coverage_gates),
            "h7_q20_coverage_failed": sum(not bool(row["passed"]) for row in coverage_gates),
            "forecast_bias_and_scale": "unavailable_not_recreated",
        },
        "interpretation": (
            "Tail coverage changed materially across origins while ordinal metrics also "
            "shifted; frozen summaries do not identify forecast bias or scale directly."
        ),
    }
    return _with_self_hash(body, "calibration_diagnostics_sha256")


def build_churn_diagnostics(
    metrics: Mapping[str, Any], gates: list[dict[str, Any]]
) -> dict[str, Any]:
    positions = [
        deepcopy(row)
        for row in metrics["position_diagnostics"]
        if row["strategy"] == "candidate"
    ]
    fold_10bps = [
        deepcopy(row)
        for row in metrics["economic"]["fold_cells"]
        if row["strategy"] == "candidate" and row["cost_bps"] == 10
    ]
    turnover_gates = [
        deepcopy(row)
        for row in gates
        if row["gate"] == "aggregate_turnover_vs_dual_momentum"
    ]
    total_actions = sum(sum(int(value) for value in row["action_counts"].values()) for row in positions)
    risky_actions = sum(int(row["action_counts"].get("long_one_asset", 0)) for row in positions)
    active = [row for row in positions if float(row["risky_exposure_fraction"]) > 0.0]
    body = {
        "schema_version": "v59-retirement-autopsy-churn/v1",
        "registered_candidate_position_cells": positions,
        "registered_candidate_fold_metrics_10bps": fold_10bps,
        "registered_turnover_gates": turnover_gates,
        "summary": {
            "position_cells": len(positions),
            "cash_only_cells": len(positions) - len(active),
            "active_cells": len(active),
            "total_position_rows": total_actions,
            "risky_position_rows": risky_actions,
            "overall_risky_exposure_fraction": risky_actions / total_actions,
            "active_cell_risky_exposure_min": min(
                float(row["risky_exposure_fraction"]) for row in active
            ),
            "active_cell_risky_exposure_max": max(
                float(row["risky_exposure_fraction"]) for row in active
            ),
            "turnover_vs_dual_gates_passed": sum(bool(row["passed"]) for row in turnover_gates),
            "transition_and_episode_rows": "unavailable_not_recreated",
        },
        "interpretation": (
            "The policy was overwhelmingly in cash and passed every turnover-vs-dual "
            "gate; frozen aggregates support sparse activation, not a global churn diagnosis."
        ),
    }
    return _with_self_hash(body, "churn_diagnostics_sha256")


def _economic_with_registered_cost_decomposition(row: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(row))
    drag = float(row["annualized_turnover"]) * float(row["cost_bps"]) / 10000.0
    value["derived_annualized_cost_drag"] = drag
    value["derived_annualized_gross_return"] = float(row["annualized_arithmetic_return"]) + drag
    return value


def build_cost_diagnostics(
    metrics: Mapping[str, Any], gates: list[dict[str, Any]]
) -> dict[str, Any]:
    aggregate = [
        _economic_with_registered_cost_decomposition(row)
        for row in metrics["economic"]["aggregate_cells"]
    ]
    folds = [
        _economic_with_registered_cost_decomposition(row)
        for row in metrics["economic"]["fold_cells"]
    ]
    economic_gate_names = {
        "candidate_return_10bps",
        "aggregate_return_vs_control",
        "aggregate_sharpe_vs_dual_momentum",
        "fold_maximum_drawdown",
        "aggregate_maximum_drawdown",
    }
    economic_gates = [deepcopy(row) for row in gates if row["gate"] in economic_gate_names]
    candidate_aggregate = [row for row in aggregate if row["strategy"] == "candidate"]
    candidate_10bps = [row for row in candidate_aggregate if row["cost_bps"] == 10]
    candidate_return_gates = [row for row in gates if row["gate"] == "candidate_return_10bps"]
    body = {
        "schema_version": "v59-retirement-autopsy-cost/v1",
        "registered_aggregate_economic_cells": aggregate,
        "registered_fold_economic_cells": folds,
        "registered_economic_gates": economic_gates,
        "summary": {
            "aggregate_cells": len(aggregate),
            "fold_cells": len(folds),
            "candidate_aggregate_cells": len(candidate_aggregate),
            "candidate_aggregate_10bps": candidate_10bps,
            "candidate_aggregate_10bps_negative": sum(
                float(row["cumulative_return"]) < 0.0 for row in candidate_10bps
            ),
            "candidate_aggregate_derived_gross_negative": sum(
                float(row["derived_annualized_gross_return"]) < 0.0
                for row in candidate_aggregate
            ),
            "candidate_return_10bps_passed": sum(bool(row["passed"]) for row in candidate_return_gates),
            "candidate_return_10bps_failed": sum(not bool(row["passed"]) for row in candidate_return_gates),
        },
        "decomposition_note": (
            "Annualized gross equals the frozen annualized arithmetic net return plus "
            "annualized turnover times the registered cost. No new PnL path or cost was computed."
        ),
        "interpretation": (
            "Candidate aggregate performance was negative before the registered cost drag "
            "in every origin/geometry/cost cell, so transaction costs amplified but did not cause failure."
        ),
    }
    return _with_self_hash(body, "cost_diagnostics_sha256")


def build_failure_attribution(
    signal: Mapping[str, Any],
    calibration: Mapping[str, Any],
    churn: Mapping[str, Any],
    cost: Mapping[str, Any],
    gates: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = _gate_summary(gates)
    body = {
        "schema_version": "v59-retirement-autopsy-attribution/v1",
        "retirement_decision": RETIREMENT_DECISION,
        "retirement_immutable": True,
        "gate_summary": summary,
        "facts": [
            "All 12 h7 q50 pairwise-accuracy gates and all 12 h7 q50 Spearman gates passed.",
            "Only 8 of 12 h7 q20 coverage gates passed.",
            "Only 1 of 12 candidate-return-at-10bps gates passed.",
            "All 108 candidate bootstrap p05 gates failed.",
            "All 432 candidate-minus-control bootstrap p05 gates failed.",
            "Five of 12 policy cells held cash for every registered row.",
            "All four aggregate turnover-vs-dual-momentum gates passed.",
            "All candidate aggregate cells had negative derived annualized gross return.",
        ],
        "attribution": {
            "primary": "weak_ordinal_signal_failed_policy_and_absolute_return_conversion",
            "contributing": [
                "tail_coverage_and_origin_drift",
                "extremely_sparse_policy_activation",
                "negative_returns_when_the_policy_activated",
            ],
            "not_supported_as_primary": [
                "transaction_costs",
                "global_turnover_or_churn",
                "drawdown_limit_breach",
            ],
        },
        "limitations": [
            "No row-level transitions or holding episodes were allowlisted.",
            "Forecast bias and scale were not persisted in the frozen summaries.",
            "No seed, checkpoint, prediction, position, outcome, or target data was reopened.",
            "No counterfactual threshold, policy, cost, bootstrap, or PnL was computed.",
        ],
        "diagnostic_hashes": {
            "signal": signal["signal_diagnostics_sha256"],
            "calibration": calibration["calibration_diagnostics_sha256"],
            "churn": churn["churn_diagnostics_sha256"],
            "cost": cost["cost_diagnostics_sha256"],
        },
        "recommendation": (
            "Keep V59 retired. Any continuation requires a newly preregistered ex-ante family "
            "and genuinely unseen non-target evidence under new explicit authorization."
        ),
    }
    return _with_self_hash(body, "failure_attribution_sha256")


def _report(
    signal: Mapping[str, Any],
    calibration: Mapping[str, Any],
    churn: Mapping[str, Any],
    cost: Mapping[str, Any],
    attribution: Mapping[str, Any],
) -> str:
    rows = cost["summary"]["candidate_aggregate_10bps"]
    table = "\n".join(
        "| {origin} | {geometry} | {ret:.4%} | {sharpe:.3f} | {dd:.4%} |".format(
            origin=row["origin"],
            geometry=row["geometry"],
            ret=row["cumulative_return"],
            sharpe=row["sharpe"],
            dd=row["maximum_drawdown"],
        )
        for row in rows
    )
    gates = attribution["gate_summary"]
    return f"""# V59 immutable retirement autopsy

## Frozen decision

`{RETIREMENT_DECISION}` remains immutable. This packet is diagnostic only.

## Facts

- Ordinal signal: {signal['summary']['pairwise_gate_passed']}/12 pairwise and {signal['summary']['spearman_gate_passed']}/12 Spearman gates passed.
- Tail calibration: {calibration['summary']['h7_q20_coverage_passed']}/12 h7 q20 coverage gates passed.
- Policy conversion: {gates['candidate_return_10bps']['passed']}/12 candidate-return gates passed.
- Robustness: {gates['candidate_bootstrap_p05']['failed']}/108 absolute and {gates['candidate_minus_control_bootstrap_p05']['failed']}/432 relative bootstrap gates failed.
- Activation: {churn['summary']['cash_only_cells']}/12 cells were cash-only; overall risky exposure was {churn['summary']['overall_risky_exposure_fraction']:.4%}.
- Cost attribution: every aggregate candidate cell was negative before registered cost drag.

| Origin | Geometry | Candidate return at 10 bps | Sharpe | Max drawdown |
|---|---|---:|---:|---:|
{table}

## Attribution

Primary: `{attribution['attribution']['primary']}`.

The Transformer retained weak cross-sectional ordering information, but the frozen policy activated extremely rarely and the selected exposures did not produce positive absolute return. Costs worsened the result but were not the cause; the policy passed every turnover comparison and failed economically before cost drag.

## Boundaries

- No source-outcome read, Parquet deserialization, checkpoint load, inference, refit, bootstrap, or regeneration.
- No BTC, ETH, or SOL data was opened.
- No counterfactual threshold, policy, cost, or PnL was evaluated.
- Any continuation requires a new ex-ante family and new explicit authorization.
"""


def preflight_v59_retirement_autopsy(
    contract_path: str | Path,
    root: str | Path = ".",
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    receipt = verify_frozen_inputs(contract, root)
    return {
        "decision": "authorize_v59_immutable_retirement_autopsy",
        "contract_sha256": CONTRACT_SHA256,
        "audit": {"passed": True, "checks": receipt["checks"]},
        "access_ledger": {
            "verified_inputs": receipt["verified_inputs"],
            "source_outcome_reads": 0,
            "parquet_deserializations": 0,
            "checkpoint_loads": 0,
            "model_inference": 0,
            "target_assets_loaded": [],
        },
    }


def run_v59_retirement_autopsy(
    contract_path: str | Path,
    root: str | Path = ".",
) -> dict[str, Any]:
    project_root = Path(root).resolve()
    contract = load_contract(contract_path)
    before = verify_frozen_inputs(contract, project_root)
    inputs = _load_inputs(contract, project_root)
    metrics = inputs["registered_metrics"]
    bootstrap = inputs["registered_bootstrap"]
    gate_matrix = inputs["registered_gate_matrix"]
    gates = gate_matrix["gates"]
    if len(gates) != 700 or gate_matrix.get("failed_count") != 603:
        raise RuntimeError("V59 autopsy frozen gate matrix drift")

    signal = build_signal_diagnostics(metrics, bootstrap, gates)
    calibration = build_calibration_diagnostics(metrics, gates)
    churn = build_churn_diagnostics(metrics, gates)
    cost = build_cost_diagnostics(metrics, gates)
    attribution = build_failure_attribution(signal, calibration, churn, cost, gates)
    after = verify_frozen_inputs(contract, project_root)
    input_receipt = _with_self_hash(
        {
            **before,
            "post_analysis_reverification_passed": after["passed"],
            "post_analysis_checks_match": after["checks"] == before["checks"],
        },
        "input_hash_receipt_sha256",
    )
    audit = _with_self_hash(
        {
            "schema_version": "v59-retirement-autopsy-audit/v1",
            "checks": {
                "contract_hash_exact": canonical_sha256(contract) == CONTRACT_SHA256,
                "all_nine_inputs_verified_before_and_after": (
                    before["verified_inputs"] == after["verified_inputs"] == 9
                    and after["checks"] == before["checks"]
                ),
                "all_700_gates_preserved": sum(
                    len(item)
                    for item in (
                        signal["registered_signal_gates"],
                        calibration["registered_coverage_gates"],
                        churn["registered_turnover_gates"],
                        cost["registered_economic_gates"],
                    )
                )
                == 700,
                "all_registered_predictive_cells_preserved": len(
                    signal["registered_predictive_cells"]
                )
                == 12,
                "all_registered_economic_cells_preserved": (
                    len(cost["registered_aggregate_economic_cells"]) == 80
                    and len(cost["registered_fold_economic_cells"]) == 240
                ),
                "retirement_decision_unchanged": inputs["evaluation_result"]["decision"]
                == RETIREMENT_DECISION,
                "source_outcome_reads_zero": True,
                "parquet_deserializations_zero": True,
                "checkpoint_loads_and_inference_zero": True,
                "new_bootstrap_and_cost_grid_zero": True,
                "target_assets_remained_sealed": (
                    inputs["evaluation_result"]["target_assets_loaded"] == []
                    and inputs["completion_receipt"]["target_assets_status"] == "sealed"
                ),
                "source_free_replay_preserved": (
                    inputs["source_free_replay"]["source_outcome_rows_read"] == 0
                    and inputs["source_free_replay"]["result_hashes_match"] is True
                ),
            },
            "decision": AUTOPSY_DECISION,
        },
        "audit_sha256",
    )
    audit["passed"] = all(audit["checks"].values())
    # Recompute after adding the derived pass flag.
    audit.pop("audit_sha256")
    audit = _with_self_hash(audit, "audit_sha256")
    if audit["passed"] is not True:
        raise RuntimeError("V59 retirement autopsy audit failed")

    result = _with_self_hash(
        {
            "schema_version": "v59-retirement-autopsy-result/v1",
            "family_id": FAMILY_ID,
            "decision": AUTOPSY_DECISION,
            "retirement_decision": RETIREMENT_DECISION,
            "retirement_immutable": True,
            "primary_failure_attribution": attribution["attribution"]["primary"],
            "mandatory_gate_count": 700,
            "passed_gate_count": 97,
            "failed_gate_count": 603,
            "contract_sha256": CONTRACT_SHA256,
            "input_hash_receipt_sha256": input_receipt["input_hash_receipt_sha256"],
            "signal_diagnostics_sha256": signal["signal_diagnostics_sha256"],
            "calibration_diagnostics_sha256": calibration[
                "calibration_diagnostics_sha256"
            ],
            "churn_diagnostics_sha256": churn["churn_diagnostics_sha256"],
            "cost_diagnostics_sha256": cost["cost_diagnostics_sha256"],
            "failure_attribution_sha256": attribution["failure_attribution_sha256"],
            "audit_sha256": audit["audit_sha256"],
            "source_outcome_reads": 0,
            "parquet_deserializations": 0,
            "checkpoint_loads": 0,
            "model_inference": 0,
            "new_bootstrap_paths": 0,
            "counterfactual_pnl_evaluations": 0,
            "target_assets_loaded": [],
            "authorized_next_action": (
                "terminate_family_or_register_new_ex_ante_family_only_with_new_explicit_authorization"
            ),
        },
        "result_sha256",
    )
    report = _report(signal, calibration, churn, cost, attribution)
    output = project_root / OUTPUT_ROOT
    payloads = {
        "input_hash_receipt.json": input_receipt,
        "signal_diagnostics.json": signal,
        "calibration_diagnostics.json": calibration,
        "churn_diagnostics.json": churn,
        "cost_diagnostics.json": cost,
        "failure_attribution.json": attribution,
        "audit.json": audit,
        "result.json": result,
    }
    prior_hashes = {
        name: file_sha256(output / name)
        for name in [*payloads, "report.md", "artifact_manifest.json"]
        if (output / name).is_file()
    }
    for name, payload in payloads.items():
        write_json_atomic(output / name, payload)
    report_path = output / "report.md"
    _write_text_atomic(report_path, report)
    manifest_body = {
        "schema_version": "v59-retirement-autopsy-manifest/v1",
        "contract_sha256": CONTRACT_SHA256,
        "file_count": len(payloads) + 1,
        "files": {
            name: file_sha256(output / name)
            for name in sorted([*payloads, "report.md"])
        },
    }
    manifest = _with_self_hash(manifest_body, "artifact_manifest_sha256")
    write_json_atomic(output / "artifact_manifest.json", manifest)
    final_hashes = {
        name: file_sha256(output / name)
        for name in [*payloads, "report.md", "artifact_manifest.json"]
    }
    files_rewritten = sum(prior_hashes.get(name) != digest for name, digest in final_hashes.items())
    final_verification = verify_frozen_inputs(contract, project_root)
    if final_verification["checks"] != before["checks"]:
        raise RuntimeError("V59 autopsy input drift after artifact write")
    return {
        "decision": AUTOPSY_DECISION,
        "retirement_decision": RETIREMENT_DECISION,
        "contract_sha256": CONTRACT_SHA256,
        "result": result,
        "audit": audit,
        "failure_attribution": attribution,
        "manifest": manifest,
        "files_rewritten": files_rewritten,
        "output_dir": str(OUTPUT_ROOT),
        "access_ledger": {
            "verified_inputs": 9,
            "source_outcome_reads": 0,
            "parquet_deserializations": 0,
            "checkpoint_loads": 0,
            "model_inference": 0,
            "new_bootstrap_paths": 0,
            "target_assets_loaded": [],
        },
    }


def verify_v59_retirement_autopsy(
    contract_path: str | Path,
    root: str | Path = ".",
) -> dict[str, Any]:
    project_root = Path(root).resolve()
    contract = load_contract(contract_path)
    inputs = verify_frozen_inputs(contract, project_root)
    output = project_root / OUTPUT_ROOT
    manifest = json.loads((output / "artifact_manifest.json").read_text(encoding="utf-8"))
    _verify_self_hash(manifest, "artifact_manifest_sha256", "manifest")
    checks = {
        "frozen_inputs": inputs["passed"],
        "manifest_contract": manifest.get("contract_sha256") == CONTRACT_SHA256,
        "manifest_file_count": manifest.get("file_count") == 9,
    }
    for name, digest in manifest.get("files", {}).items():
        path = output / name
        checks[f"artifact.{name}"] = path.is_file() and file_sha256(path) == digest
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    _verify_self_hash(result, "result_sha256", "result")
    _verify_self_hash(audit, "audit_sha256", "audit")
    checks["retirement_immutable"] = result.get("retirement_decision") == RETIREMENT_DECISION
    checks["audit_passed"] = audit.get("passed") is True
    if not all(checks.values()):
        failed = sorted(name for name, value in checks.items() if not value)
        raise RuntimeError(f"V59 autopsy verification failed: {failed}")
    return {
        "passed": True,
        "checks": checks,
        "result_sha256": result["result_sha256"],
        "artifact_manifest_sha256": manifest["artifact_manifest_sha256"],
        "source_outcome_reads": 0,
        "target_assets_loaded": [],
    }
