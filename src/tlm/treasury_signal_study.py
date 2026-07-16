from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .data import load_market_data
from .derivatives_signal_study import (
    _slice_dataset,
    _write_lift_plot,
    run_derivatives_signal_scenario,
)
from .dvol_signal_study import _near_signals
from .override import _finite_nested, make_override_dataset


def load_audited_treasury_features(
    artifact_dir: str | Path,
    required_columns: set[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    root = Path(artifact_dir)
    paths = {
        "audit": root / "audit.json",
        "catalog": root / "catalog.json",
        "features": root / "treasury_features.parquet",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing audited Treasury artifact: {path}")
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    catalog = json.loads(paths["catalog"].read_text(encoding="utf-8"))
    if not audit.get("passed"):
        raise ValueError("The Treasury source audit did not pass")
    if not required_columns.issubset(set(catalog["feature_columns"])):
        raise ValueError("Treasury study requested columns outside the frozen catalog")
    frame = pd.read_parquet(paths["features"])
    frame.index = pd.DatetimeIndex(frame.index)
    if frame.index.tz is None:
        raise ValueError("Treasury execution timestamps must be timezone-aware")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Treasury execution timestamps must be unique and sorted")
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Missing registered Treasury features: {sorted(missing)}")
    execution = frame.index.to_series(index=frame.index)
    if not (frame["source_final_at"] < execution).all():
        raise ValueError("Non-causal Treasury source-final timestamps")
    if not (frame["source_eligible_at"] <= execution).all():
        raise ValueError("Treasury state used before eligibility")
    if not frame["source_age_days"].between(3, 7).all():
        raise ValueError("Treasury state age violates the frozen carry contract")
    if not np.isfinite(frame[list(sorted(required_columns))].to_numpy(dtype=float)).all():
        raise ValueError("Registered Treasury features contain non-finite values")
    return frame, {"audit": audit, "catalog": catalog}


def build_treasury_diagnostic_signals(
    dates: pd.DatetimeIndex,
    features: pd.DataFrame,
    signal_names: list[str],
) -> pd.DataFrame:
    result = features.reindex(dates)[signal_names].apply(
        pd.to_numeric, errors="coerce"
    )
    result.index = dates
    return result


def _report(result: dict, study: dict) -> str:
    scenarios = result["scenarios"]
    scenario_names = list(scenarios)
    lines = [
        "# TLM v19 Treasury Curve Signal Existence Study",
        "",
        f"- Conclusion: **{result['conclusion']}**",
        f"- Robust signals: **{', '.join(result['robust_signals']) or 'none'}**",
        f"- Registered signals: {result['signal_count']}",
        f"- Frozen scenarios: {result['scenario_count']}",
        f"- Study window: **{study['start']} to {study['end']}**",
        f"- Audit passed: **{result['audit']['passed']}**",
        "- Source state: **t+3 eligible, bounded carry, complete observed rows**",
        "- Interpretation: **adaptive research only; no portfolio was evaluated**",
        "",
        "## Out-of-sample signal gates",
        "",
        "| signal | exp3 lift (risk/observed) | exp6 lift (risk/observed) | rolling lift (risk/observed) | passes | robust |",
        "|:--|--:|--:|--:|:--:|:--:|",
    ]
    for signal in study["signals"]:
        cells = []
        passes = []
        for name in scenario_names:
            metric = scenarios[name]["signal_metrics"][signal]
            cells.append(
                f"{metric['downside_lift']:.2f}x/{metric['tail_lift']:.2f}x "
                f"({metric['risk_coverage']:.1%}/{metric['observed_coverage']:.1%})"
            )
            passes.append("Y" if metric["passes"] else "N")
        lines.append(
            f"| {signal} | {' | '.join(cells)} | {'/'.join(passes)} | "
            f"{result['signal_summary'][signal]['robust']} |"
        )
    lines.extend([
        "",
        "## Near signals (not accepted)",
        "",
        "| signal | scenarios passed | same orientation | failed gates elsewhere |",
        "|:--|:--|:--:|:--|",
    ])
    if result["near_signals"]:
        for row in result["near_signals"]:
            failed = "; ".join(
                f"{name}: {', '.join(gates)}"
                for name, gates in row["failed_gates_by_scenario"].items()
            )
            lines.append(
                f"| {row['signal']} | {', '.join(row['passed_scenarios'])} | "
                f"{row['same_orientation_across_scenarios']} | {failed} |"
            )
    else:
        lines.append("| none | none | - | - |")
    lines.extend([
        "",
        "## Registered existence rule",
        "",
        "A signal exists only if one train-derived orientation passes observed coverage, downside lift, tail lift, fold monotonicity, orientation consistency, risk coverage, and all circular block-bootstrap gates in every scenario.",
        "",
        "## Decision",
        "",
        (
            "At least one Treasury signal survived. This authorizes at most one separately pre-registered bounded policy experiment; it does not authorize promotion or live trading."
            if result["robust_signals"]
            else "No Treasury signal survived. Close this macro-market policy branch without tuning. The next version must synthesize the accumulated evidence and decide whether any further historical model experiment is scientifically justified."
        ),
        "",
    ])
    return "\n".join(lines)


def run_treasury_signal_study(config: dict) -> dict[str, object]:
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    study = config["treasury_signal_study"]
    signal_names = list(study["signals"])
    dataset = make_override_dataset(
        load_market_data(config), momentum_lookback=int(study["momentum_lookback"])
    )
    dataset = _slice_dataset(dataset, study["start"], study["end"])
    features, source = load_audited_treasury_features(
        study["treasury_artifact_dir"], set(signal_names)
    )
    signals = build_treasury_diagnostic_signals(
        dataset.dates, features, signal_names
    )
    active = dataset.baseline_choices >= 0
    dataset_summary = {
        "start": str(dataset.dates.min()),
        "end": str(dataset.dates.max()),
        "rows": int(len(dataset.dates)),
        "active_control_rows": int(active.sum()),
        "signal_observed_coverage_on_active_control": {
            signal: float(np.isfinite(signals.loc[active, signal]).mean())
            for signal in signal_names
        },
        "source_audit_passed": bool(source["audit"]["passed"]),
        "source_timestamp_contract": source["catalog"]["timestamp_contract"],
    }
    (root / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    scenario_config = deepcopy(config)
    scenario_config["derivatives_signal_study"] = deepcopy(study)
    scenarios: dict[str, dict[str, object]] = {}
    for index, scenario in enumerate(config["validation_suite"]["scenarios"]):
        scenarios[scenario["name"]] = run_derivatives_signal_scenario(
            dataset, signals, scenario_config, scenario, scenario_index=index
        )
    robust_signals = []
    signal_summary = {}
    for signal in signal_names:
        passes = {
            name: bool(scenario["signal_metrics"][signal]["passes"])
            for name, scenario in scenarios.items()
        }
        orientations = {
            name: int(scenario["signal_metrics"][signal]["majority_orientation"])
            for name, scenario in scenarios.items()
        }
        same_orientation = len(set(orientations.values())) == 1
        robust = all(passes.values()) and same_orientation
        if robust:
            robust_signals.append(signal)
        signal_summary[signal] = {
            "robust": robust,
            "same_orientation_across_scenarios": same_orientation,
            "scenario_passes": passes,
            "scenario_orientations": orientations,
        }
    conclusion = (
        "candidate_treasury_downside_signal_exists_research_only"
        if robust_signals else "no_stable_treasury_downside_signal"
    )
    near_signals = _near_signals(
        signal_names, scenarios, robust_signals, signal_summary
    )
    _write_lift_plot(
        root, scenarios, signal_names, float(study["minimum_downside_lift"]),
        "TLM v19 Treasury Curve Signal Existence Study",
    )
    common = features.index.intersection(dataset.dates)
    checks = {
        "source_treasury_audit_passed": bool(source["audit"]["passed"]),
        "scenario_count_matches_config": len(scenarios)
        == len(config["validation_suite"]["scenarios"]),
        "contains_expanding_and_rolling": {
            scenario["validation"].get("mode", "expanding")
            for scenario in scenarios.values()
        } == {"expanding", "rolling"},
        "all_results_finite": _finite_nested(scenarios),
        "all_registered_signals_have_metrics": all(
            set(scenario["signal_metrics"]) == set(signal_names)
            for scenario in scenarios.values()
        ),
        "all_signal_rows_unique": all(
            not pd.read_parquet(
                Path(scenario["artifact_dir"]) / "signals.parquet"
            ).duplicated(["signal", "date"]).any()
            for scenario in scenarios.values()
        ),
        "all_train_windows_precede_test": all(
            pd.Timestamp(fold["train_end"]) < pd.Timestamp(fold["test_start"])
            for scenario in scenarios.values()
            for metric in scenario["signal_metrics"].values()
            for fold in metric["folds"].values()
        ),
        "study_window_matches_registered_range": bool(
            dataset.dates.min() == pd.Timestamp(study["start"], tz="UTC")
            and dataset.dates.max() == pd.Timestamp(study["end"], tz="UTC")
        ),
        "source_final_strictly_precedes_signal_date": bool(
            (features.loc[common, "source_final_at"] < common).all()
        ),
        "eligibility_not_after_signal_date": bool(
            (features.loc[common, "source_eligible_at"] <= common).all()
        ),
        "state_age_remains_bounded": bool(
            features.loc[common, "source_age_days"].between(3, 7).all()
        ),
        "lift_plot_present": (root / "downside_lift.png").is_file(),
        "no_portfolio_metrics": not any(
            key in config for key in ("strategy", "costs", "model")
        ),
    }
    audit = {"passed": all(checks.values()), "checks": checks}
    (root / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not audit["passed"]:
        raise RuntimeError(f"Treasury signal-study audit failed: {checks}")
    result: dict[str, object] = {
        "version": "v19",
        "method": "complete_case_train_oriented_treasury_downside_study",
        "conclusion": conclusion,
        "robust_signals": robust_signals,
        "near_signals": near_signals,
        "signal_count": len(signal_names),
        "scenario_count": len(scenarios),
        "clean_holdout_status": "adaptive_research_only",
        "dataset": dataset_summary,
        "signal_summary": signal_summary,
        "scenarios": scenarios,
        "audit": audit,
    }
    (root / "treasury_signal_study.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    (root / "report.md").write_text(_report(result, study), encoding="utf-8")
    return result
