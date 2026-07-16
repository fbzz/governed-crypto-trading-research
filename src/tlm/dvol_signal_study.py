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
from .override import _finite_nested, make_override_dataset


def load_audited_dvol_features(
    artifact_dir: str | Path,
    required_columns: set[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    root = Path(artifact_dir)
    audit_path = root / "audit.json"
    catalog_path = root / "catalog.json"
    features_path = root / "dvol_features.parquet"
    for path in (audit_path, catalog_path, features_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing audited DVOL artifact: {path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not audit.get("passed"):
        raise ValueError("The DVOL source audit did not pass")
    if catalog.get("sol_specific_series") is not False:
        raise ValueError("DVOL catalog must not claim a SOL-specific series")
    frame = pd.read_parquet(features_path)
    frame.index = pd.DatetimeIndex(frame.index)
    if frame.index.tz is None:
        raise ValueError("DVOL execution timestamps must be timezone-aware")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("DVOL execution timestamps must be unique and sorted")
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Missing registered DVOL features: {sorted(missing)}")
    if not (frame["source_final_at"] < frame.index.to_series(index=frame.index)).all():
        raise ValueError("Non-causal DVOL feature timestamps")
    if not (
        frame.index.to_series(index=frame.index) - frame["source_candle_timestamp"]
        == pd.Timedelta(days=2)
    ).all():
        raise ValueError("DVOL features do not preserve the frozen t+2 lag")
    values = frame[list(sorted(required_columns))].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Registered DVOL features contain non-finite values")
    return frame, {"audit": audit, "catalog": catalog}


def build_dvol_diagnostic_signals(
    dates: pd.DatetimeIndex,
    features: pd.DataFrame,
    signal_names: list[str],
) -> pd.DataFrame:
    aligned = features.reindex(dates)
    result = aligned[signal_names].apply(pd.to_numeric, errors="coerce")
    result.index = dates
    return result


def _near_signals(
    signal_names: list[str],
    scenarios: dict[str, dict[str, object]],
    robust_signals: list[str],
    signal_summary: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for signal in signal_names:
        passed = [
            name for name, scenario in scenarios.items()
            if scenario["signal_metrics"][signal]["passes"]
        ]
        if not passed or signal in robust_signals:
            continue
        rows.append({
            "signal": signal,
            "scenario_pass_count": len(passed),
            "passed_scenarios": passed,
            "same_orientation_across_scenarios": signal_summary[signal][
                "same_orientation_across_scenarios"
            ],
            "failed_gates_by_scenario": {
                name: [
                    gate for gate, ok in scenario["signal_metrics"][signal][
                        "gate_checks"
                    ].items() if not ok
                ]
                for name, scenario in scenarios.items()
                if not scenario["signal_metrics"][signal]["passes"]
            },
        })
    return sorted(
        rows,
        key=lambda row: (
            int(row["scenario_pass_count"]),
            bool(row["same_orientation_across_scenarios"]),
        ),
        reverse=True,
    )


def _report(result: dict, study: dict) -> str:
    scenarios = result["scenarios"]
    scenario_names = list(scenarios)
    lines = [
        "# TLM v15 Options Volatility Signal Existence Study",
        "",
        f"- Conclusion: **{result['conclusion']}**",
        f"- Robust signals: **{', '.join(result['robust_signals']) or 'none'}**",
        f"- Registered signals: {result['signal_count']}",
        f"- Frozen scenarios: {result['scenario_count']}",
        f"- Study window: **{study['start']} to {study['end']}**",
        f"- Audit passed: **{result['audit']['passed']}**",
        "- Missing-data policy: **complete observed rows; no imputation**",
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
        for scenario_name in scenario_names:
            metric = scenarios[scenario_name]["signal_metrics"][signal]
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
            failures = "; ".join(
                f"{scenario}: {', '.join(gates)}"
                for scenario, gates in row["failed_gates_by_scenario"].items()
            )
            lines.append(
                f"| {row['signal']} | {', '.join(row['passed_scenarios'])} | "
                f"{row['same_orientation_across_scenarios']} | {failures} |"
            )
    else:
        lines.append("| none | none | - | - |")
    lines.extend([
        "",
        "## Registered existence rule",
        "",
        "A signal exists only if one train-derived orientation passes observed coverage, downside lift, tail lift, fold monotonicity, orientation consistency, risk coverage, and every circular block-bootstrap gate in all expanding and rolling scenarios.",
        "",
        "## Decision",
        "",
        (
            "At least one options-volatility signal survived. This authorizes at most one separately pre-registered bounded policy experiment; it does not authorize promotion or live trading."
            if result["robust_signals"]
            else "No options-volatility signal survived. Close the DVOL policy branch without tuning signals or gates; continue only with a clean future holdout or a new independently audited information family."
        ),
        "",
    ])
    return "\n".join(lines)


def run_dvol_signal_study(config: dict) -> dict[str, object]:
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    (root / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    study = config["dvol_signal_study"]
    signal_names = list(study["signals"])
    dataset = make_override_dataset(
        load_market_data(config),
        momentum_lookback=int(study["momentum_lookback"]),
    )
    dataset = _slice_dataset(dataset, study["start"], study["end"])
    features, source = load_audited_dvol_features(
        study["dvol_artifact_dir"], set(signal_names)
    )
    signals = build_dvol_diagnostic_signals(dataset.dates, features, signal_names)
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
    robust_signals: list[str] = []
    signal_summary: dict[str, dict[str, object]] = {}
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
        "candidate_dvol_downside_signal_exists_research_only"
        if robust_signals else "no_stable_dvol_downside_signal"
    )
    near_signals = _near_signals(
        signal_names, scenarios, robust_signals, signal_summary
    )
    _write_lift_plot(
        root, scenarios, signal_names, float(study["minimum_downside_lift"]),
        "TLM v15 Options Volatility Signal Existence Study",
    )
    checks = {
        "source_dvol_audit_passed": bool(source["audit"]["passed"]),
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
            (
                features.loc[features.index.intersection(dataset.dates), "source_final_at"]
                < features.index.intersection(dataset.dates)
            ).all()
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
        raise RuntimeError(f"DVOL signal-study audit failed: {checks}")
    result: dict[str, object] = {
        "version": "v15",
        "method": "complete_case_train_oriented_dvol_downside_study",
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
    (root / "dvol_signal_study.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    (root / "report.md").write_text(_report(result, study), encoding="utf-8")
    return result
