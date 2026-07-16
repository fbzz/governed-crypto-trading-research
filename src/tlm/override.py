from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingRegressor

from .backtest import run_equal_weight_buy_hold, run_persistent_long_cash_backtest
from .data import load_market_data
from .dataset import walk_forward_splits
from .monte_carlo import paired_block_bootstrap
from .report import write_artifacts


@dataclass(frozen=True)
class OverrideDataset:
    x: np.ndarray
    actual_log_returns: np.ndarray
    baseline_choices: np.ndarray
    dates: pd.DatetimeIndex
    feature_names: tuple[str, ...]
    asset_names: tuple[str, ...]


def build_override_features(
    frames: Mapping[str, pd.DataFrame],
    momentum_lookback: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Build causal regime/ranking features and the next-open return target."""
    assets = list(frames)
    if not assets:
        raise ValueError("No assets supplied")
    close = pd.DataFrame({asset: frames[asset]["close"] for asset in assets})
    log_close = np.log(close)
    returns_1 = log_close.diff()
    momentum = log_close.diff(momentum_lookback)
    feature_parts: list[pd.DataFrame] = []

    for asset in assets:
        frame = frames[asset]
        part = pd.DataFrame(index=frame.index)
        for horizon in (1, 3, 7, 14, 30, 60, 90):
            part[f"{asset}__return_{horizon}"] = log_close[asset].diff(horizon)
        for horizon in (7, 21, 63):
            part[f"{asset}__volatility_{horizon}"] = returns_1[asset].rolling(horizon).std()
        annualized_30_vol = returns_1[asset].rolling(30).std() * np.sqrt(30.0)
        part[f"{asset}__momentum_vol_adjusted_30"] = (
            momentum[asset] / annualized_30_vol.replace(0.0, np.nan)
        )
        for horizon in (20, 50, 100):
            rolling_mean = frame["close"].rolling(horizon).mean()
            part[f"{asset}__trend_{horizon}"] = np.log(frame["close"] / rolling_mean)
        for horizon in (30, 90):
            part[f"{asset}__drawdown_{horizon}"] = (
                frame["close"] / frame["close"].rolling(horizon).max() - 1.0
            )
        part[f"{asset}__range"] = (frame["high"] - frame["low"]) / frame["open"]
        candle_range = (frame["high"] - frame["low"]).replace(0.0, np.nan)
        part[f"{asset}__close_position"] = (
            (frame["close"] - frame["low"]) / candle_range
        )
        log_volume = np.log(frame["volume"])
        volume_std = log_volume.rolling(21).std().replace(0.0, np.nan)
        part[f"{asset}__volume_z21"] = (
            log_volume - log_volume.rolling(21).mean()
        ) / volume_std
        market_return = returns_1.mean(axis=1)
        covariance = returns_1[asset].rolling(30).cov(market_return)
        market_variance = market_return.rolling(30).var().replace(0.0, np.nan)
        part[f"{asset}__beta_market_30"] = covariance / market_variance
        part[f"{asset}__correlation_market_30"] = (
            returns_1[asset].rolling(30).corr(market_return)
        )
        feature_parts.append(part)

    cross_features = pd.DataFrame(index=close.index)
    for horizon in (1, 7, 30):
        cross_features[f"market__return_{horizon}"] = log_close.diff(horizon).mean(axis=1)
    cross_features["market__volatility_21"] = returns_1.mean(axis=1).rolling(21).std()
    cross_features["cross__momentum_dispersion_30"] = momentum.std(axis=1)
    cross_features["cross__correlation_30"] = returns_1.rolling(30).corr().groupby(level=0).mean().mean(axis=1)

    momentum_choice = momentum.to_numpy().argmax(axis=1)
    active = momentum.max(axis=1).to_numpy() > 0.0
    baseline_values = np.where(active, momentum_choice, -1)
    baseline = pd.Series(baseline_values, index=close.index, name="baseline_choice")
    baseline_one_hot = pd.DataFrame(index=close.index)
    for index, asset in enumerate(assets):
        baseline_one_hot[f"baseline__{asset}"] = (baseline == index).astype(float)
    baseline_one_hot["baseline__CASH"] = (baseline == -1).astype(float)

    targets = pd.DataFrame({
        asset: np.log(frames[asset]["open"].shift(-2) / frames[asset]["open"].shift(-1))
        for asset in assets
    })
    features = pd.concat([*feature_parts, cross_features, baseline_one_hot], axis=1)
    return features.replace([np.inf, -np.inf], np.nan), targets, baseline


def make_override_dataset(
    frames: Mapping[str, pd.DataFrame],
    momentum_lookback: int = 30,
) -> OverrideDataset:
    features, targets, baseline = build_override_features(frames, momentum_lookback)
    valid = (
        np.isfinite(features.to_numpy()).all(axis=1)
        & np.isfinite(targets.to_numpy()).all(axis=1)
    )
    if int(valid.sum()) < 100:
        raise ValueError("Not enough finite observations for the override dataset")
    return OverrideDataset(
        x=features.loc[valid].to_numpy(dtype=np.float64),
        actual_log_returns=targets.loc[valid].to_numpy(dtype=np.float64),
        baseline_choices=baseline.loc[valid].to_numpy(dtype=np.int64),
        dates=pd.DatetimeIndex(features.index[valid]),
        feature_names=tuple(features.columns),
        asset_names=tuple(targets.columns),
    )


def build_residual_targets(
    actual_log_returns: np.ndarray,
    baseline_choices: np.ndarray,
) -> np.ndarray:
    """Return gross action edge versus the same day's dual-momentum action."""
    if actual_log_returns.ndim != 2 or len(actual_log_returns) != len(baseline_choices):
        raise ValueError("Actual returns and baseline choices are not aligned")
    simple_returns = np.expm1(actual_log_returns)
    rows = np.arange(len(simple_returns))
    baseline_returns = np.zeros(len(simple_returns), dtype=np.float64)
    active = baseline_choices >= 0
    baseline_returns[active] = simple_returns[rows[active], baseline_choices[active]]
    risky_residuals = simple_returns - baseline_returns[:, None]
    cash_residual = -baseline_returns[:, None]
    return np.concatenate([risky_residuals, cash_residual], axis=1)


def nested_chronological_split(
    train_indexes: np.ndarray,
    calibration_fraction: float,
    min_core_samples: int,
    min_calibration_samples: int,
    purge_samples: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Split an outer train window into core/calibration with both boundary purges."""
    if not 0.05 <= calibration_fraction <= 0.4 or purge_samples < 1:
        raise ValueError("Invalid nested validation configuration")
    if len(train_indexes) <= purge_samples:
        raise ValueError("Outer train window is too short for purge")
    outer_available = train_indexes[:-purge_samples]
    calibration_size = max(
        min_calibration_samples,
        int(len(outer_available) * calibration_fraction),
    )
    calibration_start = len(outer_available) - calibration_size
    core_end = calibration_start - purge_samples
    if core_end < min_core_samples or calibration_size < min_calibration_samples:
        raise ValueError("Train window is too short for nested chronological validation")
    core = outer_available[:core_end]
    calibration = outer_available[calibration_start:]
    if core[-1] >= calibration[0] - purge_samples:
        raise AssertionError("Inner purge was not preserved")
    return core, calibration, {
        "inner_purge_samples": purge_samples,
        "outer_purge_samples": purge_samples,
        "dropped_inner_boundary": int(calibration[0] - core[-1] - 1),
        "dropped_outer_boundary": int(train_indexes[-1] - outer_available[-1]),
    }


def fit_residual_models(
    x: np.ndarray,
    residual_targets: np.ndarray,
    model_config: dict,
    seed: int,
) -> list[GradientBoostingRegressor]:
    models: list[GradientBoostingRegressor] = []
    for action in range(residual_targets.shape[1]):
        model = GradientBoostingRegressor(
            loss="huber",
            learning_rate=float(model_config["learning_rate"]),
            n_estimators=int(model_config["n_estimators"]),
            max_depth=int(model_config["max_depth"]),
            min_samples_leaf=int(model_config["min_samples_leaf"]),
            max_features=model_config.get("max_features"),
            subsample=1.0,
            random_state=seed + action,
        )
        model.fit(x, residual_targets[:, action])
        models.append(model)
    return models


def predict_residual_models(
    models: list[GradientBoostingRegressor], x: np.ndarray
) -> np.ndarray:
    predictions = np.column_stack([model.predict(x) for model in models])
    if not np.isfinite(predictions).all():
        raise FloatingPointError("Residual models produced non-finite predictions")
    return predictions


def transition_turnover(previous_action: int, next_action: int, cash_action: int) -> float:
    if previous_action == next_action:
        return 0.0
    if previous_action == cash_action or next_action == cash_action:
        return 1.0
    return 2.0


def select_override_actions(
    predicted_residuals: np.ndarray,
    baseline_choices: np.ndarray,
    thresholds: float | np.ndarray,
    cost_bps: float,
    initial_previous_action: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Apply abstention after subtracting exact incremental transition costs."""
    n_rows, n_actions = predicted_residuals.shape
    cash_action = n_actions - 1
    if len(baseline_choices) != n_rows:
        raise ValueError("Predictions and baseline choices are not aligned")
    threshold_values = np.broadcast_to(np.asarray(thresholds, dtype=float), (n_rows,))
    previous = cash_action if initial_previous_action is None else int(initial_previous_action)
    selected = np.empty(n_rows, dtype=np.int64)
    selected_edges = np.empty(n_rows, dtype=np.float64)
    overridden = np.zeros(n_rows, dtype=bool)
    rate = float(cost_bps) / 10_000.0

    for row in range(n_rows):
        baseline_action = (
            int(baseline_choices[row]) if baseline_choices[row] >= 0 else cash_action
        )
        baseline_turnover = transition_turnover(previous, baseline_action, cash_action)
        edges = predicted_residuals[row].astype(np.float64, copy=True)
        for action in range(n_actions):
            incremental_turnover = (
                transition_turnover(previous, action, cash_action) - baseline_turnover
            )
            edges[action] -= incremental_turnover * rate
        # Keeping the deterministic control has known incremental edge zero.
        edges[baseline_action] = 0.0
        best_action = int(np.argmax(edges))
        if best_action != baseline_action and edges[best_action] > threshold_values[row]:
            action = best_action
            overridden[row] = True
            selected_edges[row] = edges[best_action]
        else:
            action = baseline_action
            selected_edges[row] = 0.0
        selected[row] = action
        previous = action
    return selected, selected_edges, overridden, previous


def _choices_to_scores(choices: np.ndarray, n_assets: int) -> np.ndarray:
    scores = np.full((len(choices), n_assets), -1.0, dtype=np.float64)
    risky = choices < n_assets
    scores[np.arange(len(choices))[risky], choices[risky]] = 1.0
    return scores


def _baseline_action_choices(baseline_choices: np.ndarray, n_assets: int) -> np.ndarray:
    return np.where(baseline_choices >= 0, baseline_choices, n_assets).astype(np.int64)


def _run_choices_backtest(
    choices: np.ndarray,
    actual_log_returns: np.ndarray,
    dates: pd.DatetimeIndex,
    assets: tuple[str, ...],
    cost_bps: float,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    return run_persistent_long_cash_backtest(
        _choices_to_scores(choices, len(assets)),
        actual_log_returns,
        dates,
        assets,
        threshold=0.0,
        cost_bps=cost_bps,
    )


def calibrate_abstention_threshold(
    predicted_residuals: np.ndarray,
    baseline_choices: np.ndarray,
    actual_log_returns: np.ndarray,
    dates: pd.DatetimeIndex,
    assets: tuple[str, ...],
    override_config: dict,
    cost_bps: float,
) -> tuple[float, list[dict[str, object]]]:
    baseline_actions = _baseline_action_choices(baseline_choices, len(assets))
    _, baseline_metrics = _run_choices_backtest(
        baseline_actions, actual_log_returns, dates, assets, cost_bps
    )
    candidates = [float(value) for value in override_config["threshold_grid"]]
    candidates.append(float("inf"))
    evaluations: list[dict[str, object]] = []
    passing: list[tuple[float, float]] = []
    for threshold in candidates:
        choices, _, overridden, _ = select_override_actions(
            predicted_residuals,
            baseline_choices,
            threshold,
            cost_bps,
        )
        _, metrics = _run_choices_backtest(
            choices, actual_log_returns, dates, assets, cost_bps
        )
        return_delta = float(metrics["total_return"] - baseline_metrics["total_return"])
        qualifies = bool(
            np.isfinite(threshold)
            and overridden.sum() >= int(override_config["minimum_calibration_overrides"])
            and return_delta >= float(override_config["minimum_calibration_return_delta"])
            and metrics["sharpe"] >= baseline_metrics["sharpe"]
            and metrics["max_drawdown"]
            >= baseline_metrics["max_drawdown"]
            - float(override_config["max_calibration_drawdown_worsening"])
        )
        evaluations.append({
            "threshold": None if not np.isfinite(threshold) else threshold,
            "override_days": int(overridden.sum()),
            "override_fraction": float(overridden.mean()),
            "return_delta": return_delta,
            "qualifies": qualifies,
            "candidate": metrics,
            "dual_momentum": baseline_metrics,
        })
        if qualifies:
            passing.append((float(metrics["total_return"]), threshold))
    selected = max(passing, key=lambda item: (item[0], item[1]))[1] if passing else float("inf")
    return selected, evaluations


def _gate_metrics(candidate: dict, baseline: dict, max_drawdown_worsening: float) -> bool:
    return bool(
        candidate["total_return"] > baseline["total_return"]
        and candidate["sharpe"] > baseline["sharpe"]
        and candidate["max_drawdown"]
        >= baseline["max_drawdown"] - max_drawdown_worsening
    )


def _scenario_predictions_at_cost(
    predictions: pd.DataFrame,
    n_assets: int,
    cost_bps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residual_columns = [f"residual_hat_{name}" for name in [*predictions.attrs["assets"], "CASH"]]
    choices, edges, overridden, _ = select_override_actions(
        predictions[residual_columns].to_numpy(),
        predictions["baseline_choice"].to_numpy(dtype=np.int64),
        predictions["threshold"].to_numpy(dtype=float),
        cost_bps,
    )
    if np.any((choices < 0) | (choices > n_assets)):
        raise AssertionError("Policy emitted an invalid action")
    return choices, edges, overridden


def diagnose_override_predictions(
    predictions: pd.DataFrame,
    asset_names: tuple[str, ...],
) -> dict[str, object]:
    """Describe whether selected OOS overrides delivered their forecast edge."""
    actual = np.expm1(
        predictions[[f"actual_{asset}" for asset in asset_names]].to_numpy()
    )
    actual = np.column_stack([actual, np.zeros(len(predictions))])
    cash_action = len(asset_names)
    baseline = np.where(
        predictions["baseline_choice"].to_numpy() >= 0,
        predictions["baseline_choice"].to_numpy(),
        cash_action,
    ).astype(np.int64)
    selected = predictions["selected_action"].to_numpy(dtype=np.int64)
    rows = np.arange(len(predictions))
    realized_residual = actual[rows, selected] - actual[rows, baseline]
    overridden = predictions["overridden"].to_numpy(dtype=bool)

    def summarize(mask: np.ndarray) -> dict[str, float | int]:
        values = realized_residual[mask]
        if not len(values):
            return {
                "override_days": 0,
                "predicted_edge_mean": 0.0,
                "realized_gross_residual_sum": 0.0,
                "realized_gross_residual_mean": 0.0,
                "realized_edge_hit_rate": 0.0,
            }
        return {
            "override_days": int(len(values)),
            "predicted_edge_mean": float(predictions.loc[mask, "selected_edge"].mean()),
            "realized_gross_residual_sum": float(values.sum()),
            "realized_gross_residual_mean": float(values.mean()),
            "realized_edge_hit_rate": float((values > 0.0).mean()),
        }

    by_fold = {
        str(int(fold)): summarize(overridden & (predictions["fold"].to_numpy() == fold))
        for fold in sorted(predictions["fold"].unique())
    }
    action_names = [*asset_names, "CASH"]
    by_action = {
        action_names[action]: summarize(overridden & (selected == action))
        for action in range(len(action_names))
        if np.any(overridden & (selected == action))
    }
    return {
        "overall": summarize(overridden),
        "by_fold": by_fold,
        "by_selected_action": by_action,
    }


def run_override_scenario(
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
    override_config = config["override"]
    residual_targets = build_residual_targets(
        dataset.actual_log_returns, dataset.baseline_choices
    )
    prediction_frames: list[pd.DataFrame] = []
    fold_details: dict[str, dict[str, object]] = {}
    previous_test_action: int | None = None
    base_cost = float(config["strategy"]["cost_bps"])

    for split in splits:
        core, calibration, purge = nested_chronological_split(
            split.train,
            calibration_fraction=float(override_config["calibration_fraction"]),
            min_core_samples=int(override_config["min_core_samples"]),
            min_calibration_samples=int(override_config["min_calibration_samples"]),
            purge_samples=int(override_config["purge_samples"]),
        )
        models = fit_residual_models(
            dataset.x[core],
            residual_targets[core],
            override_config["model"],
            seed=int(config.get("seed", 42)) + split.fold,
        )
        calibration_predictions = predict_residual_models(models, dataset.x[calibration])
        threshold, calibration_results = calibrate_abstention_threshold(
            calibration_predictions,
            dataset.baseline_choices[calibration],
            dataset.actual_log_returns[calibration],
            dataset.dates[calibration],
            dataset.asset_names,
            override_config,
            base_cost,
        )
        test_predictions = predict_residual_models(models, dataset.x[split.test])
        test_choices, selected_edges, overridden, previous_test_action = select_override_actions(
            test_predictions,
            dataset.baseline_choices[split.test],
            threshold,
            base_cost,
            initial_previous_action=previous_test_action,
        )
        frame = pd.DataFrame({
            "date": dataset.dates[split.test],
            "fold": split.fold,
            "model": "override_net",
            "baseline_choice": dataset.baseline_choices[split.test],
            "selected_action": test_choices,
            "selected_edge": selected_edges,
            "overridden": overridden,
            "threshold": threshold,
        })
        action_names = [*dataset.asset_names, "CASH"]
        for index, name in enumerate(action_names):
            frame[f"residual_hat_{name}"] = test_predictions[:, index]
        for index, asset in enumerate(dataset.asset_names):
            frame[f"pred_{asset}"] = _choices_to_scores(test_choices, len(dataset.asset_names))[:, index]
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
            "calibration_samples": int(len(calibration)),
            "test_samples": int(len(split.test)),
            "threshold": None if not np.isfinite(threshold) else threshold,
            "test_override_days": int(overridden.sum()),
            "purge": purge,
            "calibration_grid": calibration_results,
        }

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    predictions.attrs["assets"] = list(dataset.asset_names)
    diagnostics = diagnose_override_predictions(predictions, dataset.asset_names)
    candidate_choices = predictions["selected_action"].to_numpy(dtype=np.int64)
    baseline_choices = _baseline_action_choices(
        predictions["baseline_choice"].to_numpy(dtype=np.int64), len(dataset.asset_names)
    )
    actual_columns = [f"actual_{asset}" for asset in dataset.asset_names]
    actual = predictions[actual_columns].to_numpy()
    dates = pd.DatetimeIndex(predictions["date"])
    candidate_curve, candidate_metrics = _run_choices_backtest(
        candidate_choices, actual, dates, dataset.asset_names, base_cost
    )
    momentum_curve, momentum_metrics = _run_choices_backtest(
        baseline_choices, actual, dates, dataset.asset_names, base_cost
    )
    buy_hold_curve, buy_hold_metrics = run_equal_weight_buy_hold(actual, dates, base_cost)
    momentum_name = f"dual_momentum_{int(override_config['momentum_lookback'])}"
    metrics = {
        "override_net": candidate_metrics,
        momentum_name: momentum_metrics,
        "equal_weight_buy_hold": buy_hold_metrics,
    }

    fold_tolerance = float(override_config["gates"]["fold_return_tolerance"])
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
        fold_baseline_choices = _baseline_action_choices(
            fold_frame["baseline_choice"].to_numpy(dtype=np.int64), len(dataset.asset_names)
        )
        _, fold_baseline = _run_choices_backtest(
            fold_baseline_choices,
            fold_actual,
            fold_dates,
            dataset.asset_names,
            base_cost,
        )
        delta = float(fold_candidate["total_return"] - fold_baseline["total_return"])
        within_tolerance = delta >= -fold_tolerance
        all_folds_ok = all_folds_ok and within_tolerance
        details.update({
            "candidate": fold_candidate,
            "dual_momentum": fold_baseline,
            "return_delta": delta,
            "within_tolerance": within_tolerance,
        })

    cost_sensitivity: dict[str, dict[str, object]] = {}
    max_dd_worsening = float(override_config["gates"]["max_drawdown_worsening"])
    all_costs_ok = True
    for level in override_config["gates"]["cost_sensitivity_bps"]:
        level = float(level)
        choices, _, overridden = _scenario_predictions_at_cost(
            predictions, len(dataset.asset_names), level
        )
        _, candidate = _run_choices_backtest(
            choices, actual, dates, dataset.asset_names, level
        )
        _, baseline = _run_choices_backtest(
            baseline_choices, actual, dates, dataset.asset_names, level
        )
        passes = _gate_metrics(candidate, baseline, max_dd_worsening)
        all_costs_ok = all_costs_ok and passes
        cost_sensitivity[str(level)] = {
            "candidate": candidate,
            "dual_momentum": baseline,
            "override_days": int(overridden.sum()),
            "passes": passes,
        }

    daily = pd.DataFrame({"date": dates, "fold": predictions["fold"]})
    curves = {
        "override_net": candidate_curve,
        momentum_name: momentum_curve,
        "equal_weight_buy_hold": buy_hold_curve,
    }
    for name, curve in curves.items():
        for column in ("net_return", "gross_return", "turnover", "equity"):
            daily[f"{name}__{column}"] = curve[column].to_numpy()
    daily.to_parquet(output / "daily_returns.parquet", index=False)

    monte_carlo: dict[str, object] = {}
    mc_config = config["validation_suite"]["monte_carlo"]
    returns_by_strategy = {
        "override_net": candidate_curve["net_return"].to_numpy(),
        "dual_momentum": momentum_curve["net_return"].to_numpy(),
    }
    all_monte_carlo_ok = True
    minimum_probability = float(override_config["gates"]["minimum_monte_carlo_probability"])
    for block in mc_config["block_lengths"]:
        result = paired_block_bootstrap(
            returns_by_strategy,
            candidate_name="override_net",
            baseline_names=["dual_momentum"],
            block_length=int(block),
            n_paths=int(mc_config["paths"]),
            seed=int(mc_config["seed"]) + scenario_index * 10_000 + int(block),
            batch_size=int(mc_config.get("batch_size", 250)),
        )
        monte_carlo[str(block)] = result
        all_monte_carlo_ok = all_monte_carlo_ok and (
            result["comparisons"]["dual_momentum"]["probability_higher_total_return"]
            >= minimum_probability
        )

    primary_gate = _gate_metrics(candidate_metrics, momentum_metrics, max_dd_worsening)
    accepted = bool(primary_gate and all_folds_ok and all_costs_ok and all_monte_carlo_ok)
    robustness = {
        "accepted": accepted,
        "primary_gate_passed": primary_gate,
        "all_folds_within_return_tolerance": all_folds_ok,
        "all_cost_gates_passed": all_costs_ok,
        "all_monte_carlo_gates_passed": all_monte_carlo_ok,
        "override_days": int(predictions["overridden"].sum()),
        "override_fraction": float(predictions["overridden"].mean()),
        "folds": fold_details,
        "cost_sensitivity": cost_sensitivity,
        "monte_carlo": monte_carlo,
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
            "policy": "dual_momentum_with_residual_override_and_abstention",
            "model_objective": "huber_gradient_boosting_residual_edge",
            "model_name": "Gradient Boosting residual override",
            "candidate_key": "override_net",
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


def _finite_nested(value: object) -> bool:
    if isinstance(value, dict):
        return all(_finite_nested(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_nested(item) for item in value)
    if isinstance(value, (int, float)):
        return bool(np.isfinite(value))
    return True


def run_override_suite(config: dict) -> dict[str, object]:
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    frames = load_market_data(config)
    dataset = make_override_dataset(
        frames, momentum_lookback=int(config["override"]["momentum_lookback"])
    )
    scenarios: dict[str, object] = {}
    for index, scenario in enumerate(config["validation_suite"]["scenarios"]):
        scenarios[scenario["name"]] = run_override_scenario(
            dataset, config, scenario, scenario_index=index
        )

    all_accepted = all(
        bool(result["robustness"]["accepted"]) for result in scenarios.values()
    )
    checks = {
        "scenario_count_matches_config": len(scenarios)
        == len(config["validation_suite"]["scenarios"]),
        "contains_expanding_and_rolling": {
            result["validation"].get("mode", "expanding")
            for result in scenarios.values()
        }
        == {"expanding", "rolling"},
        "all_results_finite": _finite_nested(scenarios),
        "all_prediction_dates_unique_and_ordered": all(
            (
                lambda frame: not frame["date"].duplicated().any()
                and frame["date"].is_monotonic_increasing
            )(pd.read_parquet(Path(result["artifact_dir"]) / "predictions.parquet"))
            for result in scenarios.values()
        ),
        "all_prediction_values_finite": all(
            (
                lambda frame: np.isfinite(
                    frame[[
                        column for column in frame.columns
                        if column.startswith(("pred_", "actual_", "residual_hat_"))
                    ]].to_numpy()
                ).all()
            )(pd.read_parquet(Path(result["artifact_dir"]) / "predictions.parquet"))
            for result in scenarios.values()
        ),
        "all_required_scenario_outputs_present": all(
            all(
                (Path(result["artifact_dir"]) / filename).is_file()
                for filename in (
                    "predictions.parquet", "metrics.json", "daily_returns.parquet",
                    "equity_curve.png", "report.md", "robustness.json", "diagnostics.json",
                )
            )
            for result in scenarios.values()
        ),
        "all_nested_boundaries_ordered": all(
            pd.Timestamp(fold["core_end"]) < pd.Timestamp(fold["calibration_start"])
            < pd.Timestamp(fold["calibration_end"]) < pd.Timestamp(fold["test_start"])
            for result in scenarios.values()
            for fold in result["robustness"]["folds"].values()
        ),
    }
    audit = {"passed": all(checks.values()), "checks": checks}
    with (root / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
    if not audit["passed"]:
        raise RuntimeError(f"Override suite audit failed: {checks}")

    result = {
        "method": "nested_walk_forward_residual_gradient_boosting_with_abstention",
        "all_scenarios_accepted": all_accepted,
        "candidate_status": "accepted" if all_accepted else "rejected_or_continue_research",
        "control": f"dual_momentum_{int(config['override']['momentum_lookback'])}",
        "scenario_count": len(scenarios),
        "audit": audit,
        "scenarios": scenarios,
    }
    with (root / "validation_suite.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    lines = [
        "# OverrideNet v1 research report",
        "",
        f"- Candidate status: **{result['candidate_status']}**",
        f"- Control: **{result['control']}**",
        f"- Scenarios: {result['scenario_count']}",
        f"- Suite audit passed: **{audit['passed']}**",
        "",
        "## Frozen scenarios",
        "",
        "| scenario | mode | folds | candidate return | Sharpe | max DD | control return | override days | accepted |",
        "|:--|:--|--:|--:|--:|--:|--:|--:|:--:|",
    ]
    for name, scenario in scenarios.items():
        candidate = scenario["metrics"]["override_net"]
        control_key = next(key for key in scenario["metrics"] if key.startswith("dual_momentum_"))
        control = scenario["metrics"][control_key]
        robustness = scenario["robustness"]
        lines.append(
            f"| {name} | {scenario['validation'].get('mode', 'expanding')} | "
            f"{scenario['validation']['folds']} | {candidate['total_return']:.2%} | "
            f"{candidate['sharpe']:.3f} | {candidate['max_drawdown']:.2%} | "
            f"{control['total_return']:.2%} | {robustness['override_days']} | "
            f"{robustness['accepted']} |"
        )
    lines.extend([
        "",
        "## Out-of-sample override calibration",
        "",
        "| scenario | predicted edge/override | realized residual/override | residual sum | edge hit rate |",
        "|:--|--:|--:|--:|--:|",
    ])
    for name, scenario in scenarios.items():
        diagnostic = scenario["robustness"]["diagnostics"]["overall"]
        lines.append(
            f"| {name} | {diagnostic['predicted_edge_mean']:.3%} | "
            f"{diagnostic['realized_gross_residual_mean']:.3%} | "
            f"{diagnostic['realized_gross_residual_sum']:.2%} | "
            f"{diagnostic['realized_edge_hit_rate']:.1%} |"
        )
    lines.extend([
        "",
        "## Acceptance contract",
        "",
        "Every scenario must beat dual momentum on return and Sharpe, keep max drawdown within the configured tolerance, keep every fold within the return tolerance, preserve the edge at all configured cost levels, and reach the minimum paired block-bootstrap probability at every block length.",
        "",
        "## Decision",
        "",
        (
            "The residual override clears the frozen offline research gates."
            if all_accepted
            else "Do not promote the residual override. Keep dual momentum as the deterministic research control and diagnose the failed gates before increasing model complexity."
        ),
    ])
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
