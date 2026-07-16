from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


TARGET_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
PARQUET_INPUTS = {
    "daily_returns",
    "outcomes",
    "context_predictions",
    "triplet_positions",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _correlation(x: pd.Series, y: pd.Series, method: str) -> float | None:
    value = x.corr(y, method=method)
    return _finite_or_none(float(value))


def _git_receipt(root: Path) -> dict[str, object]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"head": head, "clean": not bool(dirty)}


def build_autopsy_spec(config: dict[str, Any]) -> dict[str, Any]:
    autopsy = config["v50_economic_failure_autopsy"]
    spec = {
        "version": autopsy["version"],
        "phase": "v50_economic_failure_autopsy_read_only",
        "expected_input_sha256": autopsy["expected_input_sha256"],
        "expected_lineage": autopsy["expected_lineage"],
        "constraints": autopsy["constraints"],
        "data_contract": autopsy["data_contract"],
        "diagnostics": autopsy["diagnostics"],
        "lifecycle": autopsy["lifecycle"],
        "artifact_contract": autopsy["artifact_contract"],
    }
    spec["autopsy_spec_sha256"] = _canonical_sha256(spec)
    return spec


def _context(config: dict[str, Any]) -> dict[str, Any]:
    autopsy = config["v50_economic_failure_autopsy"]
    root = Path(autopsy["project_root"]).resolve()
    paths = {name: root / value for name, value in autopsy["inputs"].items()}
    expected = autopsy["expected_input_sha256"]
    if set(paths) != set(expected):
        raise RuntimeError("V54 input/hash key mismatch")
    observed = {name: _sha256_file(path) for name, path in paths.items()}
    mismatches = {
        name: {"expected": expected[name], "observed": value}
        for name, value in observed.items()
        if value != expected[name]
    }
    if mismatches:
        raise RuntimeError(f"V54 immutable input drift: {mismatches}")

    result = _load_json(paths["v50_result"])
    audit = _load_json(paths["v50_audit"])
    outcome_manifest = _load_json(paths["v50_outcome_manifest"])
    lineage = autopsy["expected_lineage"]
    lineage_checks = {
        "v50_audit_passed": audit["passed"] is True,
        "v50_decision_is_retirement": result["decision"] == lineage["v50_decision"],
        "v50_spec_is_exact": result["evaluation_spec"]["evaluation_spec_sha256"]
        == lineage["v50_spec_sha256"],
        "v50_result_is_exact": result["result_sha256"]
        == lineage["v50_result_sha256"],
        "gate_counts_are_exact": result["summary"]["gate_cells"]
        == lineage["gate_cells"]
        and result["summary"]["passed_gate_cells"]
        == lineage["passed_gate_cells"]
        and result["summary"]["failed_gate_cells"]
        == lineage["failed_gate_cells"],
        "target_assets_remained_sealed": outcome_manifest["target_assets_loaded"]
        is False
        and not TARGET_SYMBOLS.intersection(
            symbol
            for receipt in outcome_manifest["receipts"]
            for symbol in receipt["symbols"]
        ),
        "post_2025_outcomes_remained_sealed": outcome_manifest[
            "post_2025_outcomes_loaded"
        ]
        is False,
    }
    if not all(lineage_checks.values()):
        raise RuntimeError(f"V54 lineage audit failed: {lineage_checks}")
    return {
        "root": root,
        "paths": paths,
        "observed_hashes": observed,
        "lineage_checks": lineage_checks,
        "autopsy_spec": build_autopsy_spec(config),
        "v50_result": result,
    }


def classify_transition(previous: str, current: str) -> str:
    if previous == "cash" and current == "cash":
        return "cash_hold"
    if previous == "cash" and current != "cash":
        return "entry"
    if previous != "cash" and current == "cash":
        return "exit"
    if previous == current:
        return "hold"
    return "switch"


def _position_actions(positions: pd.DataFrame) -> pd.DataFrame:
    frame = positions.copy()
    actions: list[str] = []
    for row in frame.itertuples(index=False):
        active = [
            getattr(row, f"symbol_{slot}")
            for slot in range(3)
            if float(getattr(row, f"candidate_weight_{slot}")) > 0.0
        ]
        if len(active) > 1:
            raise RuntimeError("V54 found more than one candidate asset per triplet")
        actions.append(active[0] if active else "cash")
    frame["action"] = actions
    keys = ["origin", "geometry", "fold", "triplet_key"]
    frame = frame.sort_values(keys + ["date"], kind="mergesort").reset_index(drop=True)
    previous = frame.groupby(keys, sort=False)["action"].shift(1).fillna("cash")
    frame["previous_action"] = previous
    frame["transition_state"] = [
        classify_transition(str(left), str(right))
        for left, right in zip(previous, frame["action"], strict=True)
    ]
    frame["transition_turnover"] = np.select(
        [
            frame["transition_state"].isin(["entry", "exit"]),
            frame["transition_state"].eq("switch"),
        ],
        [1.0 / 3.0, 2.0 / 3.0],
        default=0.0,
    )
    return frame


def summarize_transitions(actions: pd.DataFrame) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for (origin, geometry, fold), group in actions.groupby(
        ["origin", "geometry", "fold"], sort=True
    ):
        counts = group["transition_state"].value_counts()
        total = len(group)
        output[f"{origin}|{geometry}|fold{fold}"] = {
            "rows": total,
            "active_fraction": float(group["action"].ne("cash").mean()),
            "mean_transition_turnover": float(group["transition_turnover"].mean()),
            "total_transition_turnover": float(group["transition_turnover"].sum()),
            "states": {
                state: {
                    "count": int(counts.get(state, 0)),
                    "fraction": float(counts.get(state, 0) / total),
                }
                for state in ["cash_hold", "entry", "hold", "switch", "exit"]
            },
        }
    return output


def summarize_consensus(actions: pd.DataFrame, expected_triplets: int) -> dict[str, Any]:
    daily_rows: list[dict[str, Any]] = []
    for keys, group in actions.groupby(
        ["origin", "geometry", "fold", "date"], sort=True
    ):
        counts = group.loc[group["action"].ne("cash"), "action"].value_counts()
        active = int(counts.sum())
        if len(group) != expected_triplets:
            raise RuntimeError(f"V54 expected {expected_triplets} triplets, found {len(group)}")
        probabilities = counts.to_numpy(dtype=float) / active if active else np.array([])
        entropy = float(-(probabilities * np.log(probabilities)).sum()) if active else 0.0
        symbols = max(int(group["action"].nunique() - ("cash" in set(group["action"]))), 1)
        normalized = entropy / math.log(symbols) if symbols > 1 else 0.0
        daily_rows.append(
            {
                "origin": keys[0],
                "geometry": keys[1],
                "fold": int(keys[2]),
                "active_fraction": active / expected_triplets,
                "top_action_fraction": float(counts.max() / active) if active else 0.0,
                "normalized_entropy": normalized,
                "effective_asset_count": float(math.exp(entropy)),
            }
        )
    daily = pd.DataFrame(daily_rows)
    output: dict[str, Any] = {}
    for (origin, geometry, fold), group in daily.groupby(
        ["origin", "geometry", "fold"], sort=True
    ):
        output[f"{origin}|{geometry}|fold{fold}"] = {
            column: float(group[column].mean())
            for column in [
                "active_fraction",
                "top_action_fraction",
                "normalized_entropy",
                "effective_asset_count",
            ]
        }
    return output


def _prediction_long(predictions: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    keys = ["date", "origin", "geometry", "fold", "triplet_key"]
    for slot in range(3):
        piece = predictions[keys].copy()
        piece["symbol"] = predictions[f"symbol_{slot}"].to_numpy()
        piece["predicted_absolute"] = predictions[
            f"transformer_raw_absolute_{slot}"
        ].to_numpy(dtype=float)
        piece["predicted_excess"] = predictions[
            f"transformer_raw_excess_{slot}"
        ].to_numpy(dtype=float)
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def summarize_calibration(
    predictions: pd.DataFrame, outcomes: pd.DataFrame
) -> dict[str, Any]:
    long = _prediction_long(predictions)
    merged = long.merge(
        outcomes[["date", "origin", "fold", "symbol", "action_log_return"]],
        on=["date", "origin", "fold", "symbol"],
        how="left",
        validate="many_to_one",
    )
    if merged["action_log_return"].isna().any():
        raise RuntimeError("V54 calibration merge found missing registered outcomes")
    output: dict[str, Any] = {}
    for (origin, geometry, fold), group in merged.groupby(
        ["origin", "geometry", "fold"], sort=True
    ):
        predicted = group["predicted_absolute"]
        actual = group["action_log_return"]
        output[f"{origin}|{geometry}|fold{fold}"] = {
            "rows": len(group),
            "pearson": _correlation(predicted, actual, "pearson"),
            "spearman": _correlation(predicted, actual, "spearman"),
            "sign_accuracy": float(
                np.equal(predicted.to_numpy() > 0.0, actual.to_numpy() > 0.0).mean()
            ),
            "mean_prediction": float(predicted.mean()),
            "mean_outcome": float(actual.mean()),
            "mae": float(np.abs(predicted - actual).mean()),
        }
    return output


def _gate_category(name: str) -> str:
    if name.endswith(("_fold1_spearman", "_fold2_spearman", "_fold3_spearman")):
        return "fold_spearman"
    if name.endswith(("_fold1_pairwise", "_fold2_pairwise", "_fold3_pairwise")):
        return "fold_pairwise"
    if name.endswith(("_fold1_top1_excess", "_fold2_top1_excess", "_fold3_top1_excess")):
        return "fold_top1_excess"
    if name.endswith(("_fold1_return_10bps", "_fold2_return_10bps", "_fold3_return_10bps")):
        return "fold_return_10bps"
    if "candidate_above_ridge_mean" in name:
        return "predictive_vs_ridge"
    if "return_above_ridge" in name:
        return "return_vs_ridge"
    if "return_above_dual_momentum" in name:
        return "return_vs_dual"
    if "return_above_equal_weight" in name:
        return "return_vs_equal"
    if "sharpe_above_dual" in name:
        return "sharpe_vs_dual"
    if "absolute_drawdown" in name:
        return "absolute_drawdown"
    if "drawdown_vs_dual" in name:
        return "drawdown_vs_dual"
    if "turnover_vs_dual" in name:
        return "turnover_vs_dual"
    if "bootstrap" in name or "_p05_block" in name:
        return "bootstrap"
    raise RuntimeError(f"V54 unknown gate category: {name}")


def summarize_gates(gates: dict[str, Any]) -> dict[str, Any]:
    rows: dict[str, dict[str, int]] = {}
    for cell in gates["cells"]:
        category = _gate_category(str(cell["gate"]))
        row = rows.setdefault(category, {"total": 0, "passed": 0, "failed": 0})
        row["total"] += 1
        row["passed" if cell["passed"] else "failed"] += 1
    return rows


def economic_decomposition(
    portfolio: dict[str, Any],
    stresses: dict[str, Any],
    daily_returns: pd.DataFrame,
    primary_cost_bps: int,
) -> dict[str, Any]:
    metrics = portfolio["aggregate_metrics"]
    output: dict[str, Any] = {}
    base = daily_returns.loc[
        (daily_returns["cost_bps"] == primary_cost_bps)
        & (daily_returns["strategy"] == "candidate")
    ]
    for (origin, geometry), group in base.groupby(["origin", "geometry"], sort=True):
        registered = metrics[origin][geometry][str(primary_cost_bps)]
        candidate = registered["candidate"]
        dual = registered["dual_momentum_30"]
        equal = registered["equal_weight"]
        ridge = registered["ridge"]
        gross_compound = float(np.prod(1.0 + group["gross_return"].to_numpy()) - 1.0)
        net_compound = float(np.prod(1.0 + group["net_return"].to_numpy()) - 1.0)
        delayed = stresses["one_day_extra_signal_delay"][origin][geometry][
            str(primary_cost_bps)
        ]["candidate"]
        output[f"{origin}|{geometry}"] = {
            "candidate": candidate,
            "registered_gross_compound_return": gross_compound,
            "registered_net_compound_return": net_compound,
            "gross_to_net_return_gap": gross_compound - net_compound,
            "turnover_multiple_vs_dual": float(
                candidate["total_turnover"] / dual["total_turnover"]
            ),
            "return_delta_vs_ridge": float(
                candidate["total_return"] - ridge["total_return"]
            ),
            "return_delta_vs_dual": float(
                candidate["total_return"] - dual["total_return"]
            ),
            "return_delta_vs_equal_weight": float(
                candidate["total_return"] - equal["total_return"]
            ),
            "registered_one_day_delay_candidate": delayed,
            "one_day_delay_return_delta": float(
                delayed["total_return"] - candidate["total_return"]
            ),
        }
    return output


def _manifest(output: Path, required: list[str], spec_sha: str) -> dict[str, Any]:
    files = {
        name: _sha256_file(output / name)
        for name in required
        if name != "artifact_manifest.json"
    }
    manifest = {
        "version": "v54_artifact_manifest_v1",
        "autopsy_spec_sha256": spec_sha,
        "files": files,
    }
    manifest["artifact_manifest_sha256"] = _canonical_sha256(manifest)
    return manifest


def _write_report(path: Path, result: dict[str, Any]) -> None:
    attribution = result.get("failure_attribution", {})
    lines = [
        "# V54 V50 Economic Failure Autopsy",
        "",
        "## Decision",
        "",
        f"**{result['decision']}**",
        "",
        "This is retrospective diagnosis of consumed development evidence. It does not",
        "revive V50, authorize V51-V53, or permit tuning on 2024/2025.",
        "",
    ]
    if attribution:
        lines.extend(
            [
                "## Attribution",
                "",
                f"- Ranking signal survived: `{attribution['ranking_signal_survived']}`",
                f"- Economic conversion failed: `{attribution['economic_conversion_failed']}`",
                f"- Turnover was structural: `{attribution['structural_turnover_failure']}`",
                f"- Absolute calibration was unstable: `{attribution['absolute_calibration_unstable']}`",
                "",
                "## Next legal action",
                "",
                "Freeze a metadata-only V55 state-conditioned multi-horizon family.",
                "BTC/ETH/SOL remain sealed.",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _validate_cached(output: Path, required: list[str], spec_sha: str) -> dict[str, Any] | None:
    if not all((output / name).is_file() for name in required):
        return None
    manifest = _load_json(output / "artifact_manifest.json")
    expected_manifest_sha = manifest.get("artifact_manifest_sha256")
    candidate = dict(manifest)
    candidate.pop("artifact_manifest_sha256", None)
    if expected_manifest_sha != _canonical_sha256(candidate):
        return None
    if manifest.get("autopsy_spec_sha256") != spec_sha:
        return None
    for name, expected in manifest["files"].items():
        if _sha256_file(output / name) != expected:
            return None
    return _load_json(output / "result.json")


def preflight_v50_economic_failure_autopsy(config: dict[str, Any]) -> dict[str, Any]:
    context = _context(config)
    autopsy = config["v50_economic_failure_autopsy"]
    output = context["root"] / autopsy["preflight_output_dir"]
    required = autopsy["artifact_contract"]["preflight_required_files"]
    spec = context["autopsy_spec"]
    cached = _validate_cached(output, required, spec["autopsy_spec_sha256"])
    if cached is not None:
        return cached
    output.mkdir(parents=True, exist_ok=True)
    receipt = {
        "input_sha256": context["observed_hashes"],
        "parquet_inputs_hashed": sorted(PARQUET_INPUTS),
        "parquet_inputs_deserialized": [],
    }
    checks = {
        **context["lineage_checks"],
        "all_input_hashes_match": context["observed_hashes"]
        == autopsy["expected_input_sha256"],
        "zero_parquet_deserializations": True,
        "v50_decision_immutable": autopsy["constraints"]["v50_decision_mutable"]
        is False,
        "no_training_inference_or_counterfactual_policy": not any(
            [
                autopsy["constraints"]["model_instantiation_or_inference_allowed"],
                autopsy["constraints"]["training_recalibration_or_finetuning_allowed"],
                autopsy["constraints"]["alternative_policy_threshold_or_hurdle_allowed"],
                autopsy["constraints"]["counterfactual_pnl_allowed"],
            ]
        ),
    }
    audit = {"passed": all(checks.values()), "checks": checks}
    result = {
        "version": "v54_preflight",
        "decision": "authorize_v54_deterministic_read_only_autopsy",
        "autopsy_spec": spec,
        "audit": audit,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_receipt": _git_receipt(context["root"]),
        },
        "summary": {"parquet_deserializations": 0, "input_count": len(receipt["input_sha256"])},
    }
    _write_json(output / "autopsy_spec.json", spec)
    _write_json(output / "input_hash_receipt.json", receipt)
    _write_json(output / "audit.json", audit)
    _write_json(output / "result.json", result)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    _write_report(output / "report.md", result)
    manifest = _manifest(output, required, spec["autopsy_spec_sha256"])
    _write_json(output / "artifact_manifest.json", manifest)
    return result


def run_v50_economic_failure_autopsy(config: dict[str, Any]) -> dict[str, Any]:
    context = _context(config)
    autopsy = config["v50_economic_failure_autopsy"]
    spec = context["autopsy_spec"]
    preflight_output = context["root"] / autopsy["preflight_output_dir"]
    preflight_required = autopsy["artifact_contract"]["preflight_required_files"]
    preflight = _validate_cached(
        preflight_output, preflight_required, spec["autopsy_spec_sha256"]
    )
    if preflight is None or not preflight["audit"]["passed"]:
        raise RuntimeError("V54 requires a passing exact preflight")

    output = context["root"] / config["output_dir"]
    required = autopsy["artifact_contract"]["run_required_files"]
    cached = _validate_cached(output, required, spec["autopsy_spec_sha256"])
    if cached is not None:
        return cached

    paths = context["paths"]
    predictions = pd.read_parquet(paths["context_predictions"])
    positions = pd.read_parquet(paths["triplet_positions"])
    outcomes = pd.read_parquet(paths["outcomes"])
    daily_returns = pd.read_parquet(paths["daily_returns"])
    contract = autopsy["data_contract"]
    row_checks = {
        "context_prediction_rows_exact": len(predictions)
        == contract["expected_context_prediction_rows"],
        "triplet_position_rows_exact": len(positions)
        == contract["expected_triplet_position_rows"],
        "outcome_rows_exact": len(outcomes) == contract["expected_outcome_rows"],
        "daily_return_rows_exact": len(daily_returns)
        == contract["expected_daily_return_rows"],
    }
    symbols = set(outcomes["symbol"])
    target_checks = {
        "target_assets_absent_from_outcomes": not TARGET_SYMBOLS.intersection(symbols),
        "target_assets_absent_from_predictions": not TARGET_SYMBOLS.intersection(
            set(predictions[["symbol_0", "symbol_1", "symbol_2"]].stack())
        ),
        "target_assets_absent_from_positions": not TARGET_SYMBOLS.intersection(
            set(positions[["symbol_0", "symbol_1", "symbol_2"]].stack())
        ),
    }
    if not all({**row_checks, **target_checks}.values()):
        raise RuntimeError(f"V54 data audit failed: {row_checks}, {target_checks}")

    gates = summarize_gates(_load_json(paths["v50_gate_result"]))
    actions = _position_actions(positions)
    transitions = summarize_transitions(actions)
    consensus = summarize_consensus(actions, contract["expected_triplets_per_fold"])
    calibration = summarize_calibration(predictions, outcomes)
    economics = economic_decomposition(
        _load_json(paths["v50_portfolio_metrics"]),
        _load_json(paths["v50_stresses"]),
        daily_returns,
        int(contract["primary_cost_bps"]),
    )

    calibration_signs = [cell["sign_accuracy"] for cell in calibration.values()]
    calibration_pearson = [cell["pearson"] or 0.0 for cell in calibration.values()]
    turnover_multiples = [cell["turnover_multiple_vs_dual"] for cell in economics.values()]
    failure_attribution = {
        "ranking_signal_survived": gates["fold_spearman"]["passed"] == 12
        and gates["fold_pairwise"]["passed"] == 12,
        "economic_conversion_failed": gates["fold_return_10bps"]["failed"] == 9
        and gates["return_vs_dual"]["failed"] == 12
        and gates["return_vs_equal"]["failed"] == 12,
        "structural_turnover_failure": gates["turnover_vs_dual"]["failed"] == 4
        and min(turnover_multiples) > 2.0,
        "absolute_calibration_unstable": min(calibration_signs) < 0.5
        and min(calibration_pearson) < 0.0,
        "calibration_sign_accuracy_range": [
            float(min(calibration_signs)),
            float(max(calibration_signs)),
        ],
        "calibration_pearson_range": [
            float(min(calibration_pearson)),
            float(max(calibration_pearson)),
        ],
        "turnover_multiple_vs_dual_range": [
            float(min(turnover_multiples)),
            float(max(turnover_multiples)),
        ],
        "v50_decision_remains": "retire_family_without_tuning",
        "recommendation": autopsy["lifecycle"]["recommendation"],
        "authorized_next_action": autopsy["lifecycle"]["authorized_next_action"],
    }
    checks = {
        **context["lineage_checks"],
        **row_checks,
        **target_checks,
        "all_thirteen_gate_categories_preserved": len(gates) == 13
        and sum(item["total"] for item in gates.values()) == 180,
        "all_twelve_calibration_cells_reported": len(calibration) == 12,
        "all_twelve_transition_cells_reported": len(transitions) == 12,
        "all_twelve_consensus_cells_reported": len(consensus) == 12,
        "all_four_economic_cells_reported": len(economics) == 4,
        "no_counterfactual_policy_or_new_bootstrap": True,
        "v50_retirement_unchanged": failure_attribution["v50_decision_remains"]
        == "retire_family_without_tuning",
        "input_hashes_unchanged_after_run": {
            name: _sha256_file(path) for name, path in paths.items()
        }
        == context["observed_hashes"],
    }
    audit = {"passed": all(checks.values()), "checks": checks}
    if not audit["passed"]:
        raise RuntimeError(f"V54 autopsy audit failed: {checks}")
    receipt = {
        "input_sha256": context["observed_hashes"],
        "parquet_inputs_hashed": sorted(PARQUET_INPUTS),
        "parquet_inputs_deserialized": sorted(PARQUET_INPUTS),
    }
    result = {
        "version": "v54",
        "decision": autopsy["lifecycle"]["decision"],
        "autopsy_spec": spec,
        "failure_attribution": failure_attribution,
        "audit": audit,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "git_receipt": _git_receipt(context["root"]),
        },
        "summary": {
            "gate_categories": len(gates),
            "calibration_cells": len(calibration),
            "transition_cells": len(transitions),
            "consensus_cells": len(consensus),
            "economic_cells": len(economics),
        },
    }
    result["result_sha256"] = _canonical_sha256(result)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "autopsy_spec.json", spec)
    _write_json(output / "input_hash_receipt.json", receipt)
    _write_json(output / "gate_summary.json", gates)
    _write_json(output / "calibration_by_cell.json", calibration)
    _write_json(output / "transition_by_cell.json", transitions)
    _write_json(output / "consensus_by_cell.json", consensus)
    _write_json(output / "economic_decomposition.json", economics)
    _write_json(output / "failure_attribution.json", failure_attribution)
    _write_json(output / "audit.json", audit)
    _write_json(output / "result.json", result)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    _write_report(output / "report.md", result)
    manifest = _manifest(output, required, spec["autopsy_spec_sha256"])
    _write_json(output / "artifact_manifest.json", manifest)
    return result
