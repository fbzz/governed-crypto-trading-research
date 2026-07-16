from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor

from .backtest import run_equal_weight_buy_hold
from .data import load_market_data
from .dataset import walk_forward_splits
from .monte_carlo import paired_block_bootstrap
from .override import (
    OverrideDataset,
    _baseline_action_choices,
    _choices_to_scores,
    _finite_nested,
    _run_choices_backtest,
    make_override_dataset,
    nested_chronological_split,
)
from .report import write_artifacts


def control_simple_returns(
    actual_log_returns: np.ndarray,
    baseline_choices: np.ndarray,
) -> np.ndarray:
    """Realized simple return of the dual-momentum action, including cash."""
    if actual_log_returns.ndim != 2 or len(actual_log_returns) != len(baseline_choices):
        raise ValueError("Actual returns and baseline choices are not aligned")
    simple = np.expm1(actual_log_returns)
    result = np.zeros(len(simple), dtype=np.float64)
    active = baseline_choices >= 0
    rows = np.arange(len(simple))[active]
    result[active] = simple[rows, baseline_choices[active]]
    return result


def fit_quantile_models(
    x: np.ndarray,
    target: np.ndarray,
    model_config: dict,
    seed: int,
) -> dict[float, GradientBoostingRegressor]:
    models: dict[float, GradientBoostingRegressor] = {}
    for offset, quantile in enumerate(model_config["quantiles"]):
        quantile = float(quantile)
        model = GradientBoostingRegressor(
            loss="quantile",
            alpha=quantile,
            learning_rate=float(model_config["learning_rate"]),
            n_estimators=int(model_config["n_estimators"]),
            max_depth=int(model_config["max_depth"]),
            min_samples_leaf=int(model_config["min_samples_leaf"]),
            max_features=model_config.get("max_features"),
            subsample=1.0,
            random_state=seed + offset,
        )
        model.fit(x, target)
        models[quantile] = model
    return models


def predict_quantiles(
    models: dict[float, GradientBoostingRegressor],
    x: np.ndarray,
) -> tuple[np.ndarray, tuple[float, ...]]:
    quantiles = tuple(sorted(models))
    predictions = np.column_stack([models[quantile].predict(x) for quantile in quantiles])
    # Independent quantile fits can cross. Sorting is a deterministic monotonic
    # projection and does not use labels from calibration or test.
    predictions = np.sort(predictions, axis=1)
    if not np.isfinite(predictions).all():
        raise FloatingPointError("Quantile models produced non-finite predictions")
    return predictions, quantiles


def select_risk_off_actions(
    lower_quantile: np.ndarray,
    baseline_choices: np.ndarray,
    thresholds: float | np.ndarray,
    n_assets: int,
    hysteresis_margin: float,
    initial_risk_off: bool = False,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Choose control or cash using a lower-quantile entry/exit band."""
    if len(lower_quantile) != len(baseline_choices):
        raise ValueError("Quantiles and baseline choices are not aligned")
    threshold_values = np.broadcast_to(
        np.asarray(thresholds, dtype=np.float64), (len(lower_quantile),)
    )
    cash_action = n_assets
    selected = np.empty(len(lower_quantile), dtype=np.int64)
    risk_off = np.zeros(len(lower_quantile), dtype=bool)
    state = bool(initial_risk_off)

    for row in range(len(lower_quantile)):
        baseline = int(baseline_choices[row])
        if baseline < 0:
            selected[row] = cash_action
            state = False
            continue
        entry = threshold_values[row]
        exit_level = entry + float(hysteresis_margin)
        if state:
            if lower_quantile[row] >= exit_level:
                state = False
        elif lower_quantile[row] <= entry:
            state = True
        selected[row] = cash_action if state else baseline
        risk_off[row] = state
    return selected, risk_off, state


def _return_retention(candidate: dict, baseline: dict) -> float:
    baseline_return = float(baseline["total_return"])
    candidate_return = float(candidate["total_return"])
    if baseline_return > 0.0:
        return candidate_return / baseline_return
    return 1.0 if candidate_return >= baseline_return else 0.0


def calibrate_risk_threshold(
    lower_quantile: np.ndarray,
    baseline_choices: np.ndarray,
    actual_log_returns: np.ndarray,
    dates: pd.DatetimeIndex,
    assets: tuple[str, ...],
    risk_config: dict,
    cost_bps: float,
) -> tuple[float, list[dict[str, object]]]:
    baseline_actions = _baseline_action_choices(baseline_choices, len(assets))
    _, baseline_metrics = _run_choices_backtest(
        baseline_actions, actual_log_returns, dates, assets, cost_bps
    )
    candidates = [float(value) for value in risk_config["threshold_grid"]]
    candidates.append(float("-inf"))
    evaluations: list[dict[str, object]] = []
    passing: list[tuple[float, float, float, float]] = []
    for threshold in candidates:
        choices, risk_off, _ = select_risk_off_actions(
            lower_quantile,
            baseline_choices,
            threshold,
            len(assets),
            float(risk_config["hysteresis_margin"]),
        )
        _, metrics = _run_choices_backtest(
            choices, actual_log_returns, dates, assets, cost_bps
        )
        retention = _return_retention(metrics, baseline_metrics)
        drawdown_improvement = float(
            metrics["max_drawdown"] - baseline_metrics["max_drawdown"]
        )
        qualifies = bool(
            np.isfinite(threshold)
            and risk_off.sum() >= int(risk_config["minimum_calibration_risk_off_days"])
            and retention >= float(risk_config["minimum_calibration_return_retention"])
            and metrics["sharpe"] >= baseline_metrics["sharpe"]
            and drawdown_improvement
            >= float(risk_config["minimum_calibration_drawdown_improvement"])
        )
        evaluations.append({
            "threshold": None if not np.isfinite(threshold) else threshold,
            "risk_off_days": int(risk_off.sum()),
            "risk_off_fraction": float(risk_off.mean()),
            "return_retention": retention,
            "drawdown_improvement": drawdown_improvement,
            "qualifies": qualifies,
            "candidate": metrics,
            "dual_momentum": baseline_metrics,
        })
        if qualifies:
            passing.append((
                float(metrics["sharpe"]),
                float(metrics["max_drawdown"]),
                float(metrics["total_return"]),
                threshold,
            ))
    selected = max(passing)[-1] if passing else float("-inf")
    return selected, evaluations


def risk_off_gate(candidate: dict, baseline: dict, gate_config: dict) -> bool:
    return bool(
        _return_retention(candidate, baseline)
        >= float(gate_config["minimum_return_retention"])
        and candidate["sharpe"] > baseline["sharpe"]
        and candidate["max_drawdown"] - baseline["max_drawdown"]
        >= float(gate_config["minimum_drawdown_improvement"])
    )


def diagnose_risk_off_predictions(
    predictions: pd.DataFrame,
    asset_names: tuple[str, ...],
) -> dict[str, object]:
    actual = np.expm1(
        predictions[[f"actual_{asset}" for asset in asset_names]].to_numpy()
    )
    control = np.zeros(len(predictions), dtype=np.float64)
    baseline = predictions["baseline_choice"].to_numpy(dtype=np.int64)
    active = baseline >= 0
    rows = np.arange(len(predictions))[active]
    control[active] = actual[rows, baseline[active]]
    risk_off = predictions["risk_off"].to_numpy(dtype=bool)

    def summarize(mask: np.ndarray) -> dict[str, float | int]:
        selected_control = control[mask]
        if not len(selected_control):
            return {
                "risk_off_days": 0,
                "loss_capture_precision": 0.0,
                "avoided_loss_sum": 0.0,
                "missed_gain_sum": 0.0,
                "gross_benefit_sum": 0.0,
            }
        return {
            "risk_off_days": int(len(selected_control)),
            "loss_capture_precision": float((selected_control < 0.0).mean()),
            "avoided_loss_sum": float(-np.minimum(selected_control, 0.0).sum()),
            "missed_gain_sum": float(np.maximum(selected_control, 0.0).sum()),
            "gross_benefit_sum": float(-selected_control.sum()),
        }

    lower = predictions["predicted_q10"].to_numpy(dtype=float)
    by_fold = {
        str(int(fold)): summarize(
            risk_off & (predictions["fold"].to_numpy(dtype=int) == int(fold))
        )
        for fold in sorted(predictions["fold"].unique())
    }
    return {
        "overall": summarize(risk_off),
        "by_fold": by_fold,
        "q10_empirical_lower_tail_coverage": float((control[active] <= lower[active]).mean()),
        "active_control_days": int(active.sum()),
    }


def _run_risk_off_scenario(
    dataset: OverrideDataset,
    config: dict,
    scenario: dict,
    scenario_index: int,
) -> dict[str, object]:
    output = Path(config["output_dir"]) / "scenarios" / scenario["name"]
    output.mkdir(parents=True, exist_ok=True)
    validation = scenario["validation"]
    splits = walk_forward_splits(
        len(dataset.x),
        folds=int(validation["folds"]),
        min_train_fraction=float(validation["min_train_fraction"]),
        mode=validation.get("mode", "expanding"),
        train_window_samples=validation.get("train_window_samples"),
    )
    risk_config = config["risk_off"]
    target = control_simple_returns(
        dataset.actual_log_returns, dataset.baseline_choices
    )
    base_cost = float(config["strategy"]["cost_bps"])
    prediction_frames: list[pd.DataFrame] = []
    fold_details: dict[str, dict[str, object]] = {}
    previous_risk_state = False

    for split in splits:
        core, calibration, purge = nested_chronological_split(
            split.train,
            calibration_fraction=float(risk_config["calibration_fraction"]),
            min_core_samples=int(risk_config["min_core_samples"]),
            min_calibration_samples=int(risk_config["min_calibration_samples"]),
            purge_samples=int(risk_config["purge_samples"]),
        )
        active_core = core[dataset.baseline_choices[core] >= 0]
        if len(active_core) < int(risk_config["min_core_samples"]) // 2:
            raise ValueError("Too few active control samples for quantile training")
        models = fit_quantile_models(
            dataset.x[active_core],
            target[active_core],
            risk_config["model"],
            seed=int(config.get("seed", 42)) + split.fold,
        )
        calibration_predictions, quantiles = predict_quantiles(
            models, dataset.x[calibration]
        )
        lower_column = quantiles.index(min(quantiles))
        median_quantile = min(quantiles, key=lambda value: abs(value - 0.5))
        median_column = quantiles.index(median_quantile)
        threshold, calibration_grid = calibrate_risk_threshold(
            calibration_predictions[:, lower_column],
            dataset.baseline_choices[calibration],
            dataset.actual_log_returns[calibration],
            dataset.dates[calibration],
            dataset.asset_names,
            risk_config,
            base_cost,
        )
        test_predictions, test_quantiles = predict_quantiles(models, dataset.x[split.test])
        if test_quantiles != quantiles:
            raise AssertionError("Quantile columns changed between calibration and test")
        choices, risk_off, previous_risk_state = select_risk_off_actions(
            test_predictions[:, lower_column],
            dataset.baseline_choices[split.test],
            threshold,
            len(dataset.asset_names),
            float(risk_config["hysteresis_margin"]),
            initial_risk_off=previous_risk_state,
        )
        scores = _choices_to_scores(choices, len(dataset.asset_names))
        frame = pd.DataFrame({
            "date": dataset.dates[split.test],
            "fold": split.fold,
            "model": "risk_off_meta",
            "baseline_choice": dataset.baseline_choices[split.test],
            "selected_action": choices,
            "risk_off": risk_off,
            "threshold": threshold,
            "predicted_q10": test_predictions[:, lower_column],
            "predicted_q50": test_predictions[:, median_column],
        })
        for index, asset in enumerate(dataset.asset_names):
            frame[f"pred_{asset}"] = scores[:, index]
            frame[f"actual_{asset}"] = dataset.actual_log_returns[split.test, index]
        prediction_frames.append(frame)
        fold_details[str(split.fold)] = {
            "core_start": str(dataset.dates[core[0]]),
            "core_end": str(dataset.dates[core[-1]]),
            "calibration_start": str(dataset.dates[calibration[0]]),
            "calibration_end": str(dataset.dates[calibration[-1]]),
            "test_start": str(dataset.dates[split.test[0]]),
            "test_end": str(dataset.dates[split.test[-1]]),
            "core_samples": int(len(core)),
            "active_core_samples": int(len(active_core)),
            "calibration_samples": int(len(calibration)),
            "test_samples": int(len(split.test)),
            "threshold": None if not np.isfinite(threshold) else threshold,
            "test_risk_off_days": int(risk_off.sum()),
            "purge": purge,
            "calibration_grid": calibration_grid,
        }

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    actual_columns = [f"actual_{asset}" for asset in dataset.asset_names]
    actual = predictions[actual_columns].to_numpy()
    dates = pd.DatetimeIndex(predictions["date"])
    candidate_choices = predictions["selected_action"].to_numpy(dtype=np.int64)
    baseline_choices = _baseline_action_choices(
        predictions["baseline_choice"].to_numpy(dtype=np.int64), len(dataset.asset_names)
    )
    candidate_curve, candidate_metrics = _run_choices_backtest(
        candidate_choices, actual, dates, dataset.asset_names, base_cost
    )
    control_curve, control_metrics = _run_choices_backtest(
        baseline_choices, actual, dates, dataset.asset_names, base_cost
    )
    buy_hold_curve, buy_hold_metrics = run_equal_weight_buy_hold(actual, dates, base_cost)
    control_name = f"dual_momentum_{int(risk_config['momentum_lookback'])}"
    metrics = {
        "risk_off_meta": candidate_metrics,
        control_name: control_metrics,
        "equal_weight_buy_hold": buy_hold_metrics,
    }

    fold_tolerance = float(risk_config["gates"]["fold_return_tolerance"])
    all_folds_ok = True
    for fold, details in fold_details.items():
        fold_frame = predictions[predictions["fold"] == int(fold)]
        fold_actual = fold_frame[actual_columns].to_numpy()
        fold_dates = pd.DatetimeIndex(fold_frame["date"])
        _, fold_candidate = _run_choices_backtest(
            fold_frame["selected_action"].to_numpy(dtype=np.int64),
            fold_actual,
            fold_dates,
            dataset.asset_names,
            base_cost,
        )
        fold_baseline = _baseline_action_choices(
            fold_frame["baseline_choice"].to_numpy(dtype=np.int64), len(dataset.asset_names)
        )
        _, fold_control = _run_choices_backtest(
            fold_baseline, fold_actual, fold_dates, dataset.asset_names, base_cost
        )
        delta = float(fold_candidate["total_return"] - fold_control["total_return"])
        within_tolerance = delta >= -fold_tolerance
        all_folds_ok = all_folds_ok and within_tolerance
        details.update({
            "candidate": fold_candidate,
            "dual_momentum": fold_control,
            "return_delta": delta,
            "within_tolerance": within_tolerance,
        })

    cost_sensitivity: dict[str, dict[str, object]] = {}
    all_costs_ok = True
    for level in risk_config["gates"]["cost_sensitivity_bps"]:
        level = float(level)
        _, candidate = _run_choices_backtest(
            candidate_choices, actual, dates, dataset.asset_names, level
        )
        _, control = _run_choices_backtest(
            baseline_choices, actual, dates, dataset.asset_names, level
        )
        passes = risk_off_gate(candidate, control, risk_config["gates"])
        all_costs_ok = all_costs_ok and passes
        cost_sensitivity[str(level)] = {
            "candidate": candidate,
            "dual_momentum": control,
            "passes": passes,
        }

    diagnostics = diagnose_risk_off_predictions(predictions, dataset.asset_names)
    delayed_q10 = np.full(len(predictions), np.inf, dtype=np.float64)
    delayed_q10[1:] = predictions["predicted_q10"].to_numpy()[:-1]
    delayed_choices, delayed_risk_off, _ = select_risk_off_actions(
        delayed_q10,
        predictions["baseline_choice"].to_numpy(dtype=np.int64),
        predictions["threshold"].to_numpy(dtype=float),
        len(dataset.asset_names),
        float(risk_config["hysteresis_margin"]),
    )
    _, delayed_metrics = _run_choices_backtest(
        delayed_choices, actual, dates, dataset.asset_names, base_cost
    )
    delayed_stress = {
        "metrics": delayed_metrics,
        "risk_off_days": int(delayed_risk_off.sum()),
    }

    daily = pd.DataFrame({"date": dates, "fold": predictions["fold"]})
    curves = {
        "risk_off_meta": candidate_curve,
        control_name: control_curve,
        "equal_weight_buy_hold": buy_hold_curve,
    }
    for name, curve in curves.items():
        for column in ("net_return", "gross_return", "turnover", "equity"):
            daily[f"{name}__{column}"] = curve[column].to_numpy()
    daily.to_parquet(output / "daily_returns.parquet", index=False)

    monte_carlo: dict[str, object] = {}
    mc_config = config["validation_suite"]["monte_carlo"]
    all_monte_carlo_ok = True
    minimum_mc = float(risk_config["gates"]["minimum_monte_carlo_probability"])
    for block in mc_config["block_lengths"]:
        result = paired_block_bootstrap(
            {
                "risk_off_meta": candidate_curve["net_return"].to_numpy(),
                "dual_momentum": control_curve["net_return"].to_numpy(),
            },
            candidate_name="risk_off_meta",
            baseline_names=["dual_momentum"],
            block_length=int(block),
            n_paths=int(mc_config["paths"]),
            seed=int(mc_config["seed"]) + scenario_index * 10_000 + int(block),
            batch_size=int(mc_config.get("batch_size", 250)),
        )
        comparison = result["comparisons"]["dual_momentum"]
        block_passes = bool(
            comparison["probability_higher_sharpe"] >= minimum_mc
            and comparison["probability_better_max_drawdown"] >= minimum_mc
        )
        result["risk_off_gate_passed"] = block_passes
        all_monte_carlo_ok = all_monte_carlo_ok and block_passes
        monte_carlo[str(block)] = result

    primary_gate = risk_off_gate(candidate_metrics, control_metrics, risk_config["gates"])
    accepted = bool(primary_gate and all_folds_ok and all_costs_ok and all_monte_carlo_ok)
    robustness = {
        "offline_gates_passed": accepted,
        "research_only_due_to_adaptive_history_reuse": True,
        "primary_gate_passed": primary_gate,
        "all_folds_within_return_tolerance": all_folds_ok,
        "all_cost_gates_passed": all_costs_ok,
        "all_monte_carlo_gates_passed": all_monte_carlo_ok,
        "risk_off_days": int(predictions["risk_off"].sum()),
        "risk_off_fraction": float(predictions["risk_off"].mean()),
        "return_retention": _return_retention(candidate_metrics, control_metrics),
        "drawdown_improvement": float(
            candidate_metrics["max_drawdown"] - control_metrics["max_drawdown"]
        ),
        "folds": fold_details,
        "cost_sensitivity": cost_sensitivity,
        "monte_carlo": monte_carlo,
        "one_day_signal_delay": delayed_stress,
        "diagnostics": diagnostics,
        "validation": deepcopy(validation),
    }
    with (output / "robustness.json").open("w", encoding="utf-8") as handle:
        json.dump(robustness, handle, indent=2, sort_keys=True)
    with (output / "diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, sort_keys=True)
    with (output / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        scenario_config = deepcopy(config)
        scenario_config["validation"] = deepcopy(validation)
        yaml.safe_dump(scenario_config, handle, sort_keys=False)
    write_artifacts(
        output,
        predictions,
        metrics,
        curves,
        context={
            "data_source": config["data"]["source"],
            "assets": list(dataset.asset_names),
            "start": str(dates.min().date()),
            "end": str(dates.max().date()),
            "sequences": len(dates),
            "folds": len(splits),
            "cost_bps": base_cost,
            "target_mode": "next_open_to_open",
            "policy": "dual_momentum_or_cash_quantile_risk_off",
            "model_objective": "q10_q50_control_return_quantiles",
            "model_name": "Risk-Off quantile meta-labeler",
            "candidate_key": "risk_off_meta",
            "acceptance_mode": "risk_off",
            "robustness_verified": accepted,
            "walk_forward_mode": validation.get("mode", "expanding"),
        },
    )
    return {
        "artifact_dir": str(output),
        "validation": validation,
        "metrics": metrics,
        "robustness": robustness,
    }


def run_risk_off_suite(config: dict) -> dict[str, object]:
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    dataset = make_override_dataset(
        load_market_data(config),
        momentum_lookback=int(config["risk_off"]["momentum_lookback"]),
    )
    scenarios: dict[str, object] = {}
    for index, scenario in enumerate(config["validation_suite"]["scenarios"]):
        scenarios[scenario["name"]] = _run_risk_off_scenario(
            dataset, config, scenario, scenario_index=index
        )

    all_gates_passed = all(
        bool(scenario["robustness"]["offline_gates_passed"])
        for scenario in scenarios.values()
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
        "all_nested_boundaries_ordered": all(
            pd.Timestamp(fold["core_end"]) < pd.Timestamp(fold["calibration_start"])
            < pd.Timestamp(fold["calibration_end"]) < pd.Timestamp(fold["test_start"])
            for scenario in scenarios.values()
            for fold in scenario["robustness"]["folds"].values()
        ),
        "all_required_outputs_present": all(
            all(
                (Path(scenario["artifact_dir"]) / filename).is_file()
                for filename in (
                    "predictions.parquet", "metrics.json", "daily_returns.parquet",
                    "equity_curve.png", "report.md", "robustness.json", "diagnostics.json",
                )
            )
            for scenario in scenarios.values()
        ),
        "all_prediction_values_finite": all(
            (
                lambda frame: np.isfinite(
                    frame[[
                        column for column in frame.columns
                        if column.startswith(("pred_", "actual_", "predicted_"))
                    ]].to_numpy()
                ).all()
            )(pd.read_parquet(Path(scenario["artifact_dir"]) / "predictions.parquet"))
            for scenario in scenarios.values()
        ),
    }
    audit = {"passed": all(checks.values()), "checks": checks}
    with (root / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
    if not audit["passed"]:
        raise RuntimeError(f"Risk-off suite audit failed: {checks}")

    status = (
        "offline_gates_passed_research_only"
        if all_gates_passed
        else "rejected_or_continue_research"
    )
    result = {
        "method": "nested_walk_forward_quantile_risk_off_meta_labeler",
        "all_offline_gates_passed": all_gates_passed,
        "candidate_status": status,
        "control": f"dual_momentum_{int(config['risk_off']['momentum_lookback'])}",
        "clean_holdout_status": "unavailable_due_to_adaptive_reuse_of_historical_outer_windows",
        "scenario_count": len(scenarios),
        "audit": audit,
        "scenarios": scenarios,
    }
    with (root / "validation_suite.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    lines = [
        "# Risk-Off Meta-Labeler v1 research report",
        "",
        f"- Candidate status: **{status}**",
        f"- Control: **{result['control']}**",
        f"- All offline gates passed: **{all_gates_passed}**",
        f"- Suite audit passed: **{audit['passed']}**",
        "- Clean holdout: **unavailable; v6/v7 already exposed these historical windows**",
        "",
        "## Frozen scenarios",
        "",
        "| scenario | mode | return | control | retained | Sharpe | control | max DD | improvement | risk-off days | gates |",
        "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--:|",
    ]
    for name, scenario in scenarios.items():
        candidate = scenario["metrics"]["risk_off_meta"]
        control_key = next(
            key for key in scenario["metrics"] if key.startswith("dual_momentum_")
        )
        control = scenario["metrics"][control_key]
        robustness = scenario["robustness"]
        lines.append(
            f"| {name} | {scenario['validation'].get('mode', 'expanding')} | "
            f"{candidate['total_return']:.2%} | {control['total_return']:.2%} | "
            f"{robustness['return_retention']:.1%} | {candidate['sharpe']:.3f} | "
            f"{control['sharpe']:.3f} | {candidate['max_drawdown']:.2%} | "
            f"{robustness['drawdown_improvement']:.2%} | {robustness['risk_off_days']} | "
            f"{robustness['offline_gates_passed']} |"
        )
    lines.extend([
        "",
        "## Risk-off diagnostics",
        "",
        "| scenario | loss precision | avoided losses | missed gains | gross benefit | q10 coverage | delayed return |",
        "|:--|--:|--:|--:|--:|--:|--:|",
    ])
    for name, scenario in scenarios.items():
        diagnostics = scenario["robustness"]["diagnostics"]
        overall = diagnostics["overall"]
        delayed = scenario["robustness"]["one_day_signal_delay"]["metrics"]
        lines.append(
            f"| {name} | {overall['loss_capture_precision']:.1%} | "
            f"{overall['avoided_loss_sum']:.2%} | {overall['missed_gain_sum']:.2%} | "
            f"{overall['gross_benefit_sum']:.2%} | "
            f"{diagnostics['q10_empirical_lower_tail_coverage']:.1%} | "
            f"{delayed['total_return']:.2%} |"
        )
    lines.extend([
        "",
        "## Acceptance contract",
        "",
        "Every scenario must retain the configured share of dual-momentum return, improve Sharpe, improve maximum drawdown by the configured minimum, keep every fold inside the return tolerance, pass all cost levels, and reach the paired block-bootstrap Sharpe and drawdown probabilities for every block length.",
        "",
        "## Decision",
        "",
        (
            "The candidate passed the adaptive offline gates, but cannot be promoted without a newly frozen external dataset or future period."
            if all_gates_passed
            else "Reject the current risk-off candidate. Preserve dual momentum as the deterministic research control and do not tune v8 against these observed outer results."
        ),
    ])
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
