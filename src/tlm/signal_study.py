from __future__ import annotations

from copy import deepcopy
import json
import math
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
from .monte_carlo import circular_block_indices
from .override import OverrideDataset, _finite_nested, make_override_dataset
from .risk_off import control_simple_returns


def build_diagnostic_signals(
    dataset: OverrideDataset,
    signal_names: list[str],
) -> pd.DataFrame:
    """Extract fixed market features and current-control-asset features."""
    feature_index = {name: index for index, name in enumerate(dataset.feature_names)}
    frame = pd.DataFrame(index=dataset.dates)
    for signal in signal_names:
        if signal.startswith("control__"):
            suffix = signal.removeprefix("control__")
            values = np.full(len(dataset.x), np.nan, dtype=np.float64)
            active = dataset.baseline_choices >= 0
            for asset_index, asset in enumerate(dataset.asset_names):
                mask = active & (dataset.baseline_choices == asset_index)
                feature_name = f"{asset}__{suffix}"
                if feature_name not in feature_index:
                    raise KeyError(f"Unknown control feature: {feature_name}")
                values[mask] = dataset.x[mask, feature_index[feature_name]]
            frame[signal] = values
        else:
            if signal not in feature_index:
                raise KeyError(f"Unknown diagnostic feature: {signal}")
            frame[signal] = dataset.x[:, feature_index[signal]]
    return frame


def _rank_correlation(bin_indexes: np.ndarray, values: np.ndarray) -> float:
    if len(values) < 2 or np.allclose(values, values[0]):
        return 0.0
    ranks = pd.Series(values).rank(method="average").to_numpy(dtype=float)
    correlation = float(np.corrcoef(bin_indexes.astype(float), ranks)[0, 1])
    return correlation if np.isfinite(correlation) else 0.0


def _quantile_edges(values: np.ndarray, bins: int) -> np.ndarray:
    if bins < 3:
        raise ValueError("At least three quantile bins are required")
    internal = np.quantile(values, np.arange(1, bins) / bins)
    if len(np.unique(internal)) != len(internal):
        raise ValueError("Signal has insufficient unique values for registered bins")
    return np.r_[-np.inf, internal, np.inf]


def assign_registered_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Assign values using train-only edges; test extremes cannot change bins."""
    return np.digitize(values, edges[1:-1], right=True).astype(np.int64)


def fit_signal_rule(
    train_values: np.ndarray,
    train_returns: np.ndarray,
    bins: int,
    tail_quantile: float,
) -> dict[str, object]:
    edges = _quantile_edges(train_values, bins)
    assigned = assign_registered_bins(train_values, edges)
    downside = np.maximum(-train_returns, 0.0)
    means = np.array([
        float(downside[assigned == bucket].mean())
        for bucket in range(bins)
    ])
    train_rho = _rank_correlation(np.arange(bins), means)
    orientation = 1 if train_rho >= 0.0 else -1
    risk_bin = bins - 1 if orientation > 0 else 0
    return {
        "edges": edges,
        "orientation": orientation,
        "risk_bin": risk_bin,
        "train_downside_rho": train_rho,
        "train_bin_downside": means,
        "tail_cutoff": float(np.quantile(train_returns, tail_quantile)),
    }


def evaluate_signal_rule(
    values: np.ndarray,
    returns: np.ndarray,
    rule: dict[str, object],
    bins: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    assigned = assign_registered_bins(values, np.asarray(rule["edges"]))
    downside = np.maximum(-returns, 0.0)
    tail = returns <= float(rule["tail_cutoff"])
    risk = assigned == int(rule["risk_bin"])
    bin_rows: list[dict[str, float | int]] = []
    for bucket in range(bins):
        mask = assigned == bucket
        if not mask.any():
            continue
        bin_rows.append({
            "bin": bucket,
            "risk_order": bucket if int(rule["orientation"]) > 0 else bins - 1 - bucket,
            "observations": int(mask.sum()),
            "mean_return": float(returns[mask].mean()),
            "mean_downside": float(downside[mask].mean()),
            "loss_rate": float((returns[mask] < 0.0).mean()),
            "tail_rate": float(tail[mask].mean()),
        })
    bin_frame = pd.DataFrame(bin_rows)
    test_rho = _rank_correlation(
        bin_frame["risk_order"].to_numpy(dtype=float),
        bin_frame["mean_downside"].to_numpy(dtype=float),
    )
    observations = pd.DataFrame({
        "signal_value": values,
        "bin": assigned,
        "risk_flag": risk,
        "control_return": returns,
        "downside": downside,
        "tail_event": tail,
    })
    return observations, {
        "test_downside_rho": test_rho,
        "risk_coverage": float(risk.mean()),
        "bins": bin_rows,
    }


def _safe_lift(risk_values: np.ndarray, rest_values: np.ndarray) -> float:
    if not len(risk_values) or not len(rest_values):
        raise ValueError("Risk and non-risk groups must both be observed out of sample")
    denominator = float(rest_values.mean())
    numerator = float(risk_values.mean())
    if denominator <= 1e-12:
        return 1.0 if numerator <= 1e-12 else 1_000_000.0
    return numerator / denominator


def block_bootstrap_risk_lift(
    downside: np.ndarray,
    tail_event: np.ndarray,
    risk_flag: np.ndarray,
    block_length: int,
    n_paths: int,
    seed: int,
    batch_size: int = 250,
) -> dict[str, object]:
    downside = np.asarray(downside, dtype=np.float64)
    tail_event = np.asarray(tail_event, dtype=np.float64)
    risk_flag = np.asarray(risk_flag, dtype=bool)
    if not (len(downside) == len(tail_event) == len(risk_flag)):
        raise ValueError("Bootstrap inputs are not aligned")
    if not risk_flag.any() or risk_flag.all():
        raise ValueError("Risk flag must contain risk and non-risk observations")
    rng = np.random.default_rng(seed)
    downside_lift = np.empty(n_paths, dtype=np.float64)
    tail_lift = np.empty(n_paths, dtype=np.float64)
    cursor = 0
    attempts = 0
    while cursor < n_paths:
        needed = min(batch_size, n_paths - cursor)
        draw_count = max(needed * 2, needed)
        indexes = circular_block_indices(
            len(downside), block_length, draw_count, rng
        )
        sampled_risk = risk_flag[indexes]
        risk_count = sampled_risk.sum(axis=1)
        rest_count = sampled_risk.shape[1] - risk_count
        valid = (risk_count > 0) & (rest_count > 0)
        indexes = indexes[valid][:needed]
        sampled_risk = sampled_risk[valid][:needed]
        risk_count = risk_count[valid][:needed]
        rest_count = rest_count[valid][:needed]
        batch = len(indexes)
        attempts += 1
        if batch == 0:
            if attempts > 100:
                raise RuntimeError("Unable to sample a bootstrap path with both risk groups")
            continue
        sampled_downside = downside[indexes]
        sampled_tail = tail_event[indexes]
        risk_downside = (sampled_downside * sampled_risk).sum(axis=1) / risk_count
        rest_downside = (sampled_downside * ~sampled_risk).sum(axis=1) / rest_count
        risk_tail = (sampled_tail * sampled_risk).sum(axis=1) / risk_count
        rest_tail = (sampled_tail * ~sampled_risk).sum(axis=1) / rest_count
        downside_lift[cursor : cursor + batch] = np.divide(
            risk_downside,
            rest_downside,
            out=np.full(batch, 1_000_000.0),
            where=rest_downside > 1e-12,
        )
        tail_lift[cursor : cursor + batch] = np.divide(
            risk_tail,
            rest_tail,
            out=np.full(batch, 1_000_000.0),
            where=rest_tail > 1e-12,
        )
        cursor += batch

    def summary(values: np.ndarray) -> dict[str, float]:
        quantiles = np.quantile(values, [0.05, 0.5, 0.95])
        return {
            "mean": float(values.mean()),
            "p05": float(quantiles[0]),
            "median": float(quantiles[1]),
            "p95": float(quantiles[2]),
        }

    return {
        "method": "paired_circular_block_bootstrap_risk_lift",
        "block_length": block_length,
        "paths": n_paths,
        "seed": seed,
        "downside_lift": summary(downside_lift),
        "tail_lift": summary(tail_lift),
        "probability_downside_lift_above_one": float((downside_lift > 1.0).mean()),
        "probability_tail_lift_above_one": float((tail_lift > 1.0).mean()),
    }


def _aggregate_signal(
    signal_frame: pd.DataFrame,
    fold_rules: dict[str, dict[str, object]],
    study_config: dict,
    monte_carlo_config: dict,
    scenario_index: int,
    signal_index: int,
) -> dict[str, object]:
    risk = signal_frame["risk_flag"].to_numpy(dtype=bool)
    downside = signal_frame["downside"].to_numpy(dtype=float)
    tail = signal_frame["tail_event"].to_numpy(dtype=bool)
    returns = signal_frame["control_return"].to_numpy(dtype=float)
    orientations = [int(rule["orientation"]) for rule in fold_rules.values()]
    orientation_consistency = max(
        orientations.count(1), orientations.count(-1)
    ) / len(orientations)
    majority_orientation = 1 if orientations.count(1) >= orientations.count(-1) else -1
    fold_rhos = [float(rule["test_downside_rho"]) for rule in fold_rules.values()]
    monotonic_fraction = float(
        np.mean(np.asarray(fold_rhos) >= float(study_config["minimum_test_rho"]))
    )
    downside_lift = _safe_lift(downside[risk], downside[~risk])
    tail_lift = _safe_lift(tail[risk].astype(float), tail[~risk].astype(float))
    monte_carlo: dict[str, object] = {}
    all_bootstrap_pass = True
    minimum_probability = float(study_config["minimum_bootstrap_probability"])
    for block in monte_carlo_config["block_lengths"]:
        result = block_bootstrap_risk_lift(
            downside,
            tail,
            risk,
            block_length=int(block),
            n_paths=int(monte_carlo_config["paths"]),
            seed=(
                int(monte_carlo_config["seed"])
                + scenario_index * 100_000
                + signal_index * 1_000
                + int(block)
            ),
            batch_size=int(monte_carlo_config.get("batch_size", 250)),
        )
        passes = bool(
            result["probability_downside_lift_above_one"] >= minimum_probability
            and result["probability_tail_lift_above_one"] >= minimum_probability
        )
        result["passes"] = passes
        all_bootstrap_pass = all_bootstrap_pass and passes
        monte_carlo[str(block)] = result
    risk_coverage = float(risk.mean())
    passes = bool(
        orientation_consistency >= float(study_config["minimum_orientation_consistency"])
        and monotonic_fraction >= float(study_config["minimum_monotonic_fold_fraction"])
        and float(study_config["minimum_risk_coverage"])
        <= risk_coverage
        <= float(study_config["maximum_risk_coverage"])
        and downside_lift >= float(study_config["minimum_downside_lift"])
        and tail_lift >= float(study_config["minimum_tail_lift"])
        and all_bootstrap_pass
    )
    return {
        "passes": passes,
        "observations": int(len(signal_frame)),
        "risk_observations": int(risk.sum()),
        "risk_coverage": risk_coverage,
        "orientation_consistency": orientation_consistency,
        "majority_orientation": majority_orientation,
        "monotonic_fold_fraction": monotonic_fraction,
        "downside_lift": downside_lift,
        "tail_lift": tail_lift,
        "risk_mean_return": float(returns[risk].mean()),
        "rest_mean_return": float(returns[~risk].mean()),
        "risk_loss_rate": float((returns[risk] < 0.0).mean()),
        "rest_loss_rate": float((returns[~risk] < 0.0).mean()),
        "folds": fold_rules,
        "monte_carlo": monte_carlo,
    }


def _asset_regimes(
    rows: pd.DataFrame,
    asset_names: tuple[str, ...],
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for asset_index, asset in enumerate(asset_names):
        mask = rows["baseline_choice"].to_numpy(dtype=int) == asset_index
        values = rows.loc[mask, "control_return"].to_numpy(dtype=float)
        if not len(values):
            continue
        result[asset] = {
            "observations": int(len(values)),
            "mean_return": float(values.mean()),
            "mean_downside": float(np.maximum(-values, 0.0).mean()),
            "loss_rate": float((values < 0.0).mean()),
        }
    return result


def run_signal_scenario(
    dataset: OverrideDataset,
    signals: pd.DataFrame,
    config: dict,
    scenario: dict,
    scenario_index: int,
) -> dict[str, object]:
    output = Path(config["output_dir"]) / "scenarios" / scenario["name"]
    output.mkdir(parents=True, exist_ok=True)
    validation = scenario["validation"]
    study = config["signal_study"]
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
    base_rows: list[pd.DataFrame] = []
    purge = int(study["purge_samples"])

    for split in splits:
        train = split.train[:-purge]
        active_train = train[dataset.baseline_choices[train] >= 0]
        active_test = split.test[dataset.baseline_choices[split.test] >= 0]
        if len(active_train) < 100 or len(active_test) < 10:
            raise ValueError("Insufficient active observations for signal study")
        base_rows.append(pd.DataFrame({
            "date": dataset.dates[active_test],
            "fold": split.fold,
            "baseline_choice": dataset.baseline_choices[active_test],
            "control_return": target[active_test],
        }))
        for signal in study["signals"]:
            train_values = signals.iloc[active_train][signal].to_numpy(dtype=float)
            test_values = signals.iloc[active_test][signal].to_numpy(dtype=float)
            if not np.isfinite(train_values).all() or not np.isfinite(test_values).all():
                raise ValueError(f"Non-finite diagnostic signal: {signal}")
            rule = fit_signal_rule(
                train_values,
                target[active_train],
                bins=int(study["quantile_bins"]),
                tail_quantile=float(study["tail_quantile"]),
            )
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
            observations["train_downside_rho"] = float(rule["train_downside_rho"])
            observations["tail_cutoff"] = float(rule["tail_cutoff"])
            signal_frames.append(observations)
            fold_rules_by_signal[signal][str(split.fold)] = {
                "train_start": str(dataset.dates[active_train[0]]),
                "train_end": str(dataset.dates[active_train[-1]]),
                "test_start": str(dataset.dates[active_test[0]]),
                "test_end": str(dataset.dates[active_test[-1]]),
                "train_samples": int(len(active_train)),
                "test_samples": int(len(active_test)),
                "purge_samples": purge,
                "orientation": int(rule["orientation"]),
                "risk_bin": int(rule["risk_bin"]),
                "train_downside_rho": float(rule["train_downside_rho"]),
                "tail_cutoff": float(rule["tail_cutoff"]),
                **test_metrics,
            }

    observations = pd.concat(signal_frames, ignore_index=True).sort_values(
        ["signal", "date"]
    ).reset_index(drop=True)
    observations.to_parquet(output / "signals.parquet", index=False)
    base = pd.concat(base_rows, ignore_index=True).sort_values("date").reset_index(drop=True)
    base.to_parquet(output / "control_observations.parquet", index=False)

    metrics: dict[str, object] = {}
    for signal_index, signal in enumerate(study["signals"]):
        signal_frame = observations[observations["signal"] == signal].sort_values("date")
        metrics[signal] = _aggregate_signal(
            signal_frame,
            fold_rules_by_signal[signal],
            study,
            config["validation_suite"]["monte_carlo"],
            scenario_index,
            signal_index,
        )
    passing = [signal for signal, values in metrics.items() if values["passes"]]
    result = {
        "validation": deepcopy(validation),
        "artifact_dir": str(output),
        "passing_signals": passing,
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
) -> None:
    figure, axis = plt.subplots(figsize=(12, 6))
    x = np.arange(len(signals), dtype=float)
    width = 0.8 / len(scenarios)
    for scenario_index, (name, scenario) in enumerate(scenarios.items()):
        values = [scenario["signal_metrics"][signal]["downside_lift"] for signal in signals]
        axis.bar(
            x + (scenario_index - (len(scenarios) - 1) / 2) * width,
            values,
            width=width,
            label=name,
        )
    axis.axhline(minimum_lift, color="black", linestyle="--", linewidth=1.0, label="registered gate")
    axis.set_xticks(x)
    axis.set_xticklabels(signals, rotation=35, ha="right")
    axis.set_ylabel("OOS downside lift: risk quintile / remaining observations")
    axis.set_title("TLM v9 signal-existence study")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(root / "downside_lift.png", dpi=150)
    plt.close(figure)


def run_signal_study(config: dict) -> dict[str, object]:
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    study = config["signal_study"]
    dataset = make_override_dataset(
        load_market_data(config),
        momentum_lookback=int(study["momentum_lookback"]),
    )
    signals = build_diagnostic_signals(dataset, list(study["signals"]))
    scenarios: dict[str, dict[str, object]] = {}
    for index, scenario in enumerate(config["validation_suite"]["scenarios"]):
        scenarios[scenario["name"]] = run_signal_scenario(
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
            name: int(scenario["signal_metrics"][signal]["majority_orientation"])
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
        "candidate_ohlcv_signals_exist_research_only"
        if robust_signals
        else "no_stable_ohlcv_downside_signal"
    )
    _write_lift_plot(
        root,
        scenarios,
        list(study["signals"]),
        float(study["minimum_downside_lift"]),
    )
    checks = {
        "scenario_count_matches_config": len(scenarios)
        == len(config["validation_suite"]["scenarios"]),
        "contains_expanding_and_rolling": {
            scenario["validation"].get("mode", "expanding")
            for scenario in scenarios.values()
        }
        == {"expanding", "rolling"},
        "all_results_finite": _finite_nested(scenarios),
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
        "all_train_windows_precede_test": all(
            pd.Timestamp(fold["train_end"]) < pd.Timestamp(fold["test_start"])
            for scenario in scenarios.values()
            for signal in scenario["signal_metrics"].values()
            for fold in signal["folds"].values()
        ),
        "lift_plot_present": (root / "downside_lift.png").is_file(),
    }
    audit = {"passed": all(checks.values()), "checks": checks}
    with (root / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
    if not audit["passed"]:
        raise RuntimeError(f"Signal-study audit failed: {checks}")
    result = {
        "method": "train_oriented_oos_quantile_regime_downside_study",
        "conclusion": conclusion,
        "robust_signals": robust_signals,
        "signal_count": len(study["signals"]),
        "scenario_count": len(scenarios),
        "clean_holdout_status": "adaptive_research_only",
        "audit": audit,
        "signal_summary": signal_summary,
        "scenarios": scenarios,
    }
    with (root / "signal_study.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    lines = [
        "# TLM v9 Signal Existence Study",
        "",
        f"- Conclusion: **{conclusion}**",
        f"- Robust signals: **{', '.join(robust_signals) if robust_signals else 'none'}**",
        f"- Registered signals: {len(study['signals'])}",
        f"- Frozen scenarios: {len(scenarios)}",
        f"- Audit passed: **{audit['passed']}**",
        "- Interpretation: **adaptive research only; historical outer windows were already exposed**",
        "",
        "## Out-of-sample signal gates",
        "",
        "| signal | exp3 lift (coverage) | exp6 lift (coverage) | rolling lift (coverage) | scenario passes | robust |",
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
                f"({metric['risk_coverage']:.1%})"
            )
            passes.append("Y" if metric["passes"] else "N")
        lines.append(
            f"| {signal} | {' | '.join(cells)} | {'/'.join(passes)} | "
            f"{signal_summary[signal]['robust']} |"
        )
    lines.extend([
        "",
        "## Registered existence rule",
        "",
        "A signal exists only if the same train-derived orientation passes downside lift, tail lift, fold monotonicity, orientation consistency, risk coverage, and all circular block-bootstrap gates in every expanding and rolling scenario.",
        "",
        "## Decision",
        "",
        (
            "At least one OHLCV signal survived the registered study. It may justify one new pre-registered policy experiment, but not promotion."
            if robust_signals
            else "No OHLCV-only signal survived. Stop adding model complexity to the current feature family; the next experiment must add genuinely new, correctly timestamped information such as funding, open interest, or basis."
        ),
    ])
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
