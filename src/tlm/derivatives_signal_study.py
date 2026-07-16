from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tlm-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .data import load_market_data
from .dataset import walk_forward_splits
from .override import OverrideDataset, _finite_nested, make_override_dataset
from .risk_off import control_simple_returns
from .signal_study import (
    _aggregate_signal,
    _asset_regimes,
    evaluate_signal_rule,
    fit_signal_rule,
)


ALLOWED_DERIVATIVE_SIGNALS = {
    "funding_rate_sum",
    "funding_rate_7d_sum",
    "funding_rate_30d_mean",
    "basis_close",
    "basis_7d_mean",
    "basis_30d_z",
    "open_interest_log_change_1d",
    "open_interest_log_change_7d",
    "open_interest_value_log_change_1d",
    "toptrader_count_ratio_mean",
    "toptrader_position_ratio_mean",
    "global_long_short_ratio_mean",
    "taker_long_short_ratio_mean",
    "taker_ratio_last",
    "taker_ratio_log_change",
    "taker_ratio_slope",
    "taker_ratio_autocorr_1",
    "taker_buy_fraction",
    "taker_ratio_last_hour_mean",
    "taker_ratio_reversal_1h",
    "oi_log_change_intraday",
    "oi_range_intraday",
    "oi_max_drawdown_intraday",
    "oi_slope_intraday",
    "oi_last_hour_log_change",
    "oi_value_log_change_intraday",
}


def _slice_dataset(
    dataset: OverrideDataset,
    start: str,
    end: str,
) -> OverrideDataset:
    start_timestamp = pd.Timestamp(start, tz="UTC")
    end_timestamp = pd.Timestamp(end, tz="UTC")
    mask = (dataset.dates >= start_timestamp) & (dataset.dates <= end_timestamp)
    if int(mask.sum()) < 300:
        raise ValueError("Derivatives study window has fewer than 300 control rows")
    return OverrideDataset(
        x=dataset.x[mask],
        actual_log_returns=dataset.actual_log_returns[mask],
        baseline_choices=dataset.baseline_choices[mask],
        dates=dataset.dates[mask],
        feature_names=dataset.feature_names,
        asset_names=dataset.asset_names,
    )


def load_audited_derivatives(
    artifact_dir: str | Path,
    asset_names: tuple[str, ...],
    required_columns: set[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    root = Path(artifact_dir)
    audit_path = root / "audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Missing derivatives audit: {audit_path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not audit.get("passed"):
        raise ValueError("The derivatives source audit did not pass")
    frames: dict[str, pd.DataFrame] = {}
    for asset in asset_names:
        path = root / "assets" / f"{asset}.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"Missing audited derivatives asset: {path}")
        frame = pd.read_parquet(path)
        frame.index = pd.DatetimeIndex(frame.index)
        if frame.index.tz is None:
            raise ValueError(f"{asset}: derivatives timestamps must be timezone-aware")
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            raise ValueError(f"{asset}: derivatives timestamps must be unique and sorted")
        missing = (required_columns or ALLOWED_DERIVATIVE_SIGNALS) - set(frame.columns)
        if missing:
            raise ValueError(f"{asset}: missing derivatives columns {sorted(missing)}")
        observed = frame["source_max_timestamp"].notna()
        if not (
            frame.loc[observed, "source_max_timestamp"]
            < frame.loc[observed, "execution_open"]
        ).all():
            raise ValueError(f"{asset}: non-causal derivatives timestamps")
        frames[asset] = frame
    return frames, audit


def _signal_parts(signal: str) -> tuple[str, str]:
    try:
        transform, column = signal.split("__", maxsplit=1)
    except ValueError as error:
        raise ValueError(f"Invalid derivatives signal name: {signal}") from error
    if transform not in {"control", "market_mean", "cross_dispersion"}:
        raise ValueError(f"Unsupported derivatives signal transform: {transform}")
    if column not in ALLOWED_DERIVATIVE_SIGNALS:
        raise ValueError(f"Unregistered derivatives signal column: {column}")
    return transform, column


def build_derivatives_diagnostic_signals(
    dataset: OverrideDataset,
    derivatives: dict[str, pd.DataFrame],
    signal_names: list[str],
) -> pd.DataFrame:
    if set(derivatives) != set(dataset.asset_names):
        raise ValueError("Derivatives assets do not match the control dataset")
    result = pd.DataFrame(index=dataset.dates)
    aligned = {
        asset: derivatives[asset].reindex(dataset.dates)
        for asset in dataset.asset_names
    }
    for signal in signal_names:
        transform, column = _signal_parts(signal)
        matrix = np.column_stack([
            pd.to_numeric(aligned[asset][column], errors="coerce").to_numpy(
                dtype=np.float64
            )
            for asset in dataset.asset_names
        ])
        values = np.full(len(dataset.dates), np.nan, dtype=np.float64)
        if transform == "control":
            for asset_index in range(len(dataset.asset_names)):
                selected = dataset.baseline_choices == asset_index
                values[selected] = matrix[selected, asset_index]
        else:
            complete = np.isfinite(matrix).all(axis=1)
            if transform == "market_mean":
                values[complete] = matrix[complete].mean(axis=1)
            else:
                values[complete] = matrix[complete].std(axis=1, ddof=0)
        result[signal] = values
    return result


def _rejected_signal_metric(
    fold_rules: dict[str, dict[str, object]],
    reason: str,
) -> dict[str, object]:
    orientations = [int(rule["orientation"]) for rule in fold_rules.values()]
    if orientations:
        majority = 1 if orientations.count(1) >= orientations.count(-1) else -1
        consistency = max(
            orientations.count(1), orientations.count(-1)
        ) / len(orientations)
    else:
        majority = 0
        consistency = 0.0
    return {
        "passes": False,
        "failure_reason": reason,
        "observations": 0,
        "risk_observations": 0,
        "risk_coverage": 0.0,
        "orientation_consistency": consistency,
        "majority_orientation": majority,
        "monotonic_fold_fraction": 0.0,
        "downside_lift": 0.0,
        "tail_lift": 0.0,
        "risk_mean_return": 0.0,
        "rest_mean_return": 0.0,
        "risk_loss_rate": 0.0,
        "rest_loss_rate": 0.0,
        "folds": fold_rules,
        "monte_carlo": {},
    }


def run_derivatives_signal_scenario(
    dataset: OverrideDataset,
    signals: pd.DataFrame,
    config: dict,
    scenario: dict,
    scenario_index: int,
) -> dict[str, object]:
    output = Path(config["output_dir"]) / "scenarios" / scenario["name"]
    output.mkdir(parents=True, exist_ok=True)
    validation = scenario["validation"]
    study = config["derivatives_signal_study"]
    splits = walk_forward_splits(
        len(dataset.x),
        folds=int(validation["folds"]),
        min_train_fraction=float(validation["min_train_fraction"]),
        mode=validation.get("mode", "expanding"),
        train_window_samples=validation.get("train_window_samples"),
    )
    target = control_simple_returns(
        dataset.actual_log_returns, dataset.baseline_choices
    )
    signal_frames: list[pd.DataFrame] = []
    fold_rules_by_signal: dict[str, dict[str, dict[str, object]]] = {
        signal: {} for signal in study["signals"]
    }
    observed_counts = {
        signal: {"observed": 0, "eligible": 0}
        for signal in study["signals"]
    }
    signal_failures: dict[str, str] = {}
    base_rows: list[pd.DataFrame] = []
    purge = int(study["purge_samples"])

    for split in splits:
        train = split.train[:-purge]
        active_train_base = train[dataset.baseline_choices[train] >= 0]
        active_test_base = split.test[dataset.baseline_choices[split.test] >= 0]
        if len(active_train_base) < 100 or len(active_test_base) < 10:
            raise ValueError("Insufficient active control observations")
        base_rows.append(pd.DataFrame({
            "date": dataset.dates[active_test_base],
            "fold": split.fold,
            "baseline_choice": dataset.baseline_choices[active_test_base],
            "control_return": target[active_test_base],
        }))

        for signal in study["signals"]:
            train_values_all = signals.iloc[active_train_base][signal].to_numpy(
                dtype=float
            )
            test_values_all = signals.iloc[active_test_base][signal].to_numpy(
                dtype=float
            )
            train_available = np.isfinite(train_values_all)
            test_available = np.isfinite(test_values_all)
            active_train = active_train_base[train_available]
            active_test = active_test_base[test_available]
            if len(active_train) < 100 or len(active_test) < 10:
                raise ValueError(f"Insufficient complete cases for {signal}")
            observed_counts[signal]["observed"] += int(len(active_test))
            observed_counts[signal]["eligible"] += int(len(active_test_base))
            if signal in signal_failures:
                continue
            train_values = signals.iloc[active_train][signal].to_numpy(dtype=float)
            test_values = signals.iloc[active_test][signal].to_numpy(dtype=float)
            try:
                rule = fit_signal_rule(
                    train_values,
                    target[active_train],
                    bins=int(study["quantile_bins"]),
                    tail_quantile=float(study["tail_quantile"]),
                )
            except ValueError as error:
                signal_failures[signal] = f"fold {split.fold}: {error}"
                continue
            observations, test_metrics = evaluate_signal_rule(
                test_values,
                target[active_test],
                rule,
                bins=int(study["quantile_bins"]),
            )
            observations.insert(0, "date", dataset.dates[active_test])
            observations.insert(1, "fold", split.fold)
            observations.insert(2, "signal", signal)
            observations["orientation"] = int(rule["orientation"])
            observations["train_downside_rho"] = float(
                rule["train_downside_rho"]
            )
            observations["tail_cutoff"] = float(rule["tail_cutoff"])
            signal_frames.append(observations)
            fold_rules_by_signal[signal][str(split.fold)] = {
                "train_start": str(dataset.dates[active_train[0]]),
                "train_end": str(dataset.dates[active_train[-1]]),
                "test_start": str(dataset.dates[active_test[0]]),
                "test_end": str(dataset.dates[active_test[-1]]),
                "train_samples": int(len(active_train)),
                "test_samples": int(len(active_test)),
                "train_observed_coverage": float(train_available.mean()),
                "test_observed_coverage": float(test_available.mean()),
                "purge_samples": purge,
                "orientation": int(rule["orientation"]),
                "risk_bin": int(rule["risk_bin"]),
                "train_downside_rho": float(rule["train_downside_rho"]),
                "tail_cutoff": float(rule["tail_cutoff"]),
                **test_metrics,
            }

    if signal_frames:
        observations = pd.concat(signal_frames, ignore_index=True).sort_values(
            ["signal", "date"]
        ).reset_index(drop=True)
    else:
        observations = pd.DataFrame(columns=[
            "date", "fold", "signal", "signal_value", "bin", "risk_flag",
            "control_return", "downside", "tail_event", "orientation",
            "train_downside_rho", "tail_cutoff",
        ])
    observations.to_parquet(output / "signals.parquet", index=False)
    base = pd.concat(base_rows, ignore_index=True).sort_values("date").reset_index(
        drop=True
    )
    base.to_parquet(output / "control_observations.parquet", index=False)

    metrics: dict[str, object] = {}
    minimum_observed = float(study["minimum_observed_coverage"])
    for signal_index, signal in enumerate(study["signals"]):
        signal_frame = observations[observations["signal"] == signal].sort_values(
            "date"
        )
        if signal in signal_failures:
            metric = _rejected_signal_metric(
                fold_rules_by_signal[signal], signal_failures[signal]
            )
        else:
            try:
                metric = _aggregate_signal(
                    signal_frame,
                    fold_rules_by_signal[signal],
                    study,
                    config["validation_suite"]["monte_carlo"],
                    scenario_index,
                    signal_index,
                )
            except ValueError as error:
                signal_failures[signal] = f"aggregate: {error}"
                metric = _rejected_signal_metric(
                    fold_rules_by_signal[signal], signal_failures[signal]
                )
        counts = observed_counts[signal]
        observed_coverage = counts["observed"] / counts["eligible"]
        metric["observed_coverage"] = observed_coverage
        metric["eligible_observations"] = int(counts["eligible"])
        metric["missing_observations"] = int(
            counts["eligible"] - counts["observed"]
        )
        metric["observed_coverage_passes"] = observed_coverage >= minimum_observed
        gate_checks = {
            "testable": signal not in signal_failures,
            "observed_coverage": metric["observed_coverage_passes"],
            "orientation_consistency": metric["orientation_consistency"]
            >= float(study["minimum_orientation_consistency"]),
            "fold_monotonicity": metric["monotonic_fold_fraction"]
            >= float(study["minimum_monotonic_fold_fraction"]),
            "risk_coverage": float(study["minimum_risk_coverage"])
            <= metric["risk_coverage"]
            <= float(study["maximum_risk_coverage"]),
            "downside_lift": metric["downside_lift"]
            >= float(study["minimum_downside_lift"]),
            "tail_lift": metric["tail_lift"]
            >= float(study["minimum_tail_lift"]),
            "all_block_bootstraps": bool(metric["monte_carlo"])
            and all(
                bool(result["passes"])
                for result in metric["monte_carlo"].values()
            ),
        }
        metric["gate_checks"] = gate_checks
        metric["passes"] = all(gate_checks.values())
        metrics[signal] = metric

    passing = [signal for signal, values in metrics.items() if values["passes"]]
    result = {
        "validation": deepcopy(validation),
        "artifact_dir": str(output),
        "passing_signals": passing,
        "untestable_signals": signal_failures,
        "signal_metrics": metrics,
        "asset_regimes": _asset_regimes(base, dataset.asset_names),
    }
    with (output / "signal_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    with (output / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        scenario_config = deepcopy(config)
        scenario_config["validation"] = deepcopy(validation)
        yaml.safe_dump(scenario_config, handle, sort_keys=False)
    return result


def _write_lift_plot(
    root: Path,
    scenarios: dict[str, dict[str, object]],
    signals: list[str],
    minimum_lift: float,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(15, 7))
    x = np.arange(len(signals), dtype=float)
    width = 0.8 / len(scenarios)
    for scenario_index, (name, scenario) in enumerate(scenarios.items()):
        values = [
            scenario["signal_metrics"][signal]["downside_lift"]
            for signal in signals
        ]
        axis.bar(
            x + (scenario_index - (len(scenarios) - 1) / 2) * width,
            values,
            width=width,
            label=name,
        )
    axis.axhline(
        minimum_lift, color="black", linestyle="--", linewidth=1.0,
        label="registered gate",
    )
    axis.set_xticks(x)
    axis.set_xticklabels(signals, rotation=45, ha="right")
    axis.set_ylabel("OOS downside lift: risk quintile / remaining observations")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(root / "downside_lift.png", dpi=150)
    plt.close(figure)


def run_derivatives_signal_study(config: dict) -> dict[str, object]:
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    study = config["derivatives_signal_study"]
    study_title = study.get(
        "study_title", "TLM v11 Derivatives Signal Existence Study"
    )
    positive_conclusion = study.get(
        "positive_conclusion", "candidate_derivatives_signals_exist_research_only"
    )
    negative_conclusion = study.get(
        "negative_conclusion", "no_stable_derivatives_downside_signal"
    )
    required_columns = {
        _signal_parts(signal)[1] for signal in study["signals"]
    }
    dataset = make_override_dataset(
        load_market_data(config),
        momentum_lookback=int(study["momentum_lookback"]),
    )
    dataset = _slice_dataset(dataset, study["start"], study["end"])
    derivatives, source_audit = load_audited_derivatives(
        study["derivatives_artifact_dir"], dataset.asset_names, required_columns
    )
    signals = build_derivatives_diagnostic_signals(
        dataset, derivatives, list(study["signals"])
    )
    active = dataset.baseline_choices >= 0
    dataset_summary = {
        "start": str(dataset.dates.min()),
        "end": str(dataset.dates.max()),
        "rows": int(len(dataset.dates)),
        "active_control_rows": int(active.sum()),
        "signal_observed_coverage_on_active_control": {
            signal: float(np.isfinite(signals.loc[active, signal]).mean())
            for signal in study["signals"]
        },
        "source_audit_passed": bool(source_audit["passed"]),
    }
    with (root / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(dataset_summary, handle, indent=2, sort_keys=True)

    scenarios: dict[str, dict[str, object]] = {}
    for index, scenario in enumerate(config["validation_suite"]["scenarios"]):
        scenarios[scenario["name"]] = run_derivatives_signal_scenario(
            dataset, signals, config, scenario, scenario_index=index
        )

    robust_signals: list[str] = []
    signal_summary: dict[str, dict[str, object]] = {}
    for signal in study["signals"]:
        scenario_passes = {
            name: bool(scenario["signal_metrics"][signal]["passes"])
            for name, scenario in scenarios.items()
        }
        orientations = {
            name: int(
                scenario["signal_metrics"][signal]["majority_orientation"]
            )
            for name, scenario in scenarios.items()
        }
        same_orientation = len(set(orientations.values())) == 1
        robust = all(scenario_passes.values()) and same_orientation
        if robust:
            robust_signals.append(signal)
        signal_summary[signal] = {
            "robust": robust,
            "same_orientation_across_scenarios": same_orientation,
            "scenario_passes": scenario_passes,
            "scenario_orientations": orientations,
        }

    conclusion = (
        positive_conclusion
        if robust_signals
        else negative_conclusion
    )
    untestable_signals = {
        signal: {
            name: scenario["untestable_signals"][signal]
            for name, scenario in scenarios.items()
            if signal in scenario["untestable_signals"]
        }
        for signal in study["signals"]
        if any(
            signal in scenario["untestable_signals"]
            for scenario in scenarios.values()
        )
    }
    near_signals = []
    for signal in study["signals"]:
        if signal in untestable_signals:
            continue
        passed_scenarios = [
            name
            for name, scenario in scenarios.items()
            if scenario["signal_metrics"][signal]["passes"]
        ]
        if not passed_scenarios or signal in robust_signals:
            continue
        near_signals.append({
            "signal": signal,
            "scenario_pass_count": len(passed_scenarios),
            "passed_scenarios": passed_scenarios,
            "same_orientation_across_scenarios": signal_summary[signal][
                "same_orientation_across_scenarios"
            ],
            "failed_gates_by_scenario": {
                name: [
                    gate
                    for gate, passed in scenario["signal_metrics"][signal][
                        "gate_checks"
                    ].items()
                    if not passed
                ]
                for name, scenario in scenarios.items()
                if not scenario["signal_metrics"][signal]["passes"]
            },
        })
    near_signals.sort(
        key=lambda row: (
            int(row["scenario_pass_count"]),
            bool(row["same_orientation_across_scenarios"]),
        ),
        reverse=True,
    )
    _write_lift_plot(
        root,
        scenarios,
        list(study["signals"]),
        float(study["minimum_downside_lift"]),
        study_title,
    )
    checks = {
        "source_derivatives_audit_passed": bool(source_audit["passed"]),
        "scenario_count_matches_config": len(scenarios)
        == len(config["validation_suite"]["scenarios"]),
        "contains_expanding_and_rolling": {
            scenario["validation"].get("mode", "expanding")
            for scenario in scenarios.values()
        }
        == {"expanding", "rolling"},
        "all_results_finite": _finite_nested(scenarios),
        "all_registered_signals_have_metrics": all(
            set(scenario["signal_metrics"]) == set(study["signals"])
            for scenario in scenarios.values()
        ),
        "all_required_outputs_present": all(
            all(
                (Path(scenario["artifact_dir"]) / filename).is_file()
                for filename in (
                    "signals.parquet", "control_observations.parquet",
                    "signal_metrics.json", "resolved_config.yaml",
                )
            )
            for scenario in scenarios.values()
        ),
        "all_signal_rows_unique": all(
            not pd.read_parquet(
                Path(scenario["artifact_dir"]) / "signals.parquet"
            ).duplicated(["signal", "date"]).any()
            for scenario in scenarios.values()
        ),
        "all_observed_signal_values_finite": all(
            np.isfinite(pd.read_parquet(
                Path(scenario["artifact_dir"]) / "signals.parquet"
            )["signal_value"].to_numpy()).all()
            for scenario in scenarios.values()
        ),
        "all_train_windows_precede_test": all(
            pd.Timestamp(fold["train_end"]) < pd.Timestamp(fold["test_start"])
            for scenario in scenarios.values()
            for signal in scenario["signal_metrics"].values()
            for fold in signal["folds"].values()
        ),
        "study_window_matches_registered_range": bool(
            dataset.dates.min() == pd.Timestamp(study["start"], tz="UTC")
            and dataset.dates.max() == pd.Timestamp(study["end"], tz="UTC")
        ),
        "lift_plot_present": (root / "downside_lift.png").is_file(),
    }
    audit = {"passed": all(checks.values()), "checks": checks}
    with (root / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
    if not audit["passed"]:
        raise RuntimeError(f"Derivatives signal-study audit failed: {checks}")

    result = {
        "method": study.get(
            "method", "complete_case_train_oriented_derivatives_downside_study"
        ),
        "conclusion": conclusion,
        "robust_signals": robust_signals,
        "untestable_signals": untestable_signals,
        "near_signals": near_signals,
        "signal_count": len(study["signals"]),
        "scenario_count": len(scenarios),
        "clean_holdout_status": "adaptive_research_only",
        "audit": audit,
        "dataset": dataset_summary,
        "signal_summary": signal_summary,
        "scenarios": scenarios,
    }
    with (root / "derivatives_signal_study.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    lines = [
        f"# {study_title}",
        "",
        f"- Conclusion: **{conclusion}**",
        f"- Robust signals: **{', '.join(robust_signals) if robust_signals else 'none'}**",
        f"- Untestable registrations: **{', '.join(untestable_signals) if untestable_signals else 'none'}**",
        f"- Registered signals: {len(study['signals'])}",
        f"- Frozen scenarios: {len(scenarios)}",
        f"- Study window: **{study['start']} to {study['end']}**",
        f"- Audit passed: **{audit['passed']}**",
        "- Missing-data policy: **complete-case per signal; no forward-fill**",
        "- Interpretation: **adaptive research only; historical windows were already exposed**",
        "",
        "## Out-of-sample signal gates",
        "",
        "| signal | exp3 lift (risk/observed) | exp6 lift (risk/observed) | rolling lift (risk/observed) | passes | robust |",
        "|:--|--:|--:|--:|:--:|:--:|",
    ]
    scenario_names = list(scenarios)
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
            f"{signal_summary[signal]['robust']} |"
        )
    lines.extend([
        "",
        "## Near signals (not accepted)",
        "",
        "| signal | scenarios passed | same orientation | failed gates elsewhere |",
        "|:--|:--|:--:|:--|",
        *(
            [
                f"| {row['signal']} | "
                f"{', '.join(row['passed_scenarios'])} | "
                f"{row['same_orientation_across_scenarios']} | "
                + "; ".join(
                    f"{scenario}: {', '.join(gates)}"
                    for scenario, gates in row["failed_gates_by_scenario"].items()
                )
                + " |"
                for row in near_signals
            ]
            if near_signals
            else ["| none | none | - | - |"]
        ),
        "",
        "## Untestable registrations",
        "",
        *(
            [
                f"- `{signal}`: "
                + "; ".join(f"{scenario}: {reason}" for scenario, reason in failures.items())
                for signal, failures in untestable_signals.items()
            ]
            if untestable_signals
            else ["- None."]
        ),
        "",
        "## Registered existence rule",
        "",
        "A registered signal exists only if the same train-derived orientation passes observed coverage, downside lift, tail lift, fold monotonicity, orientation consistency, risk coverage, and every circular block-bootstrap gate in all expanding and rolling scenarios.",
        "",
        "## Decision",
        "",
        (
            study.get(
                "positive_decision",
                "At least one derivatives signal survived. It may justify one separately pre-registered policy experiment, but does not authorize promotion or live trading.",
            )
            if robust_signals
            else study.get(
                "negative_decision",
                "No derivatives signal survived the registered study. Do not tune the v11 bins or gates and do not design a policy from these variables. The next bounded experiment should remain policy-free and pre-register intraday-path features from the existing 5-minute metrics, especially taker-flow persistence/reversal and open-interest path shape, before rerunning the same existence gates.",
            )
        ),
    ])
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
