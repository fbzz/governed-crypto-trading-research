from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .monte_carlo import circular_block_indices
from .scientific_harness import persistent_portfolio_returns
from .source_domain_one_shot import performance_metrics


MODEL_NAMES = ("transformer", "ridge")
PAIR_INDEXES = ((0, 1), (0, 2), (1, 2))
STRATEGY_WEIGHT_COLUMNS = {
    "candidate": "candidate_weight",
    "dual_momentum_30": "dual_momentum_30_weight",
    "momentum_gated_equal_weight": "momentum_gated_equal_weight_weight",
}
V45_COSTS_BPS = (10, 20, 30)
V45_BLOCK_LENGTHS = (7, 21, 63)


def _finite_vector(values: object, label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite one-dimensional vector")
    return array


def _average_ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(values, copy=False).rank(method="average").to_numpy(
        dtype=np.float64
    )


def average_rank_spearman(scores: object, actual: object) -> float:
    """Spearman correlation using exact-equality average ranks.

    A constant score or outcome vector is an undefined correlation and returns
    ``nan``. The V45 gate treats any such value as a screen failure.
    """

    predicted = _finite_vector(scores, "scores")
    observed = _finite_vector(actual, "actual")
    if predicted.shape != observed.shape:
        raise ValueError("scores and actual must share one-dimensional shape")
    predicted_rank = _average_ranks(predicted)
    observed_rank = _average_ranks(observed)
    predicted_centered = predicted_rank - predicted_rank.mean()
    observed_centered = observed_rank - observed_rank.mean()
    denominator = math.sqrt(
        float(np.dot(predicted_centered, predicted_centered))
        * float(np.dot(observed_centered, observed_centered))
    )
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(predicted_centered, observed_centered) / denominator)


def _normalized_context_inputs(
    context_predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    context_columns = {
        "date",
        "fold",
        "triplet_key",
        *(f"symbol_{index}" for index in range(3)),
        *(
            f"{model}_raw_excess_{index}"
            for model in MODEL_NAMES
            for index in range(3)
        ),
    }
    outcome_columns = {"date", "fold", "symbol", "action_log_return"}
    missing_context = sorted(context_columns - set(context_predictions.columns))
    missing_outcome = sorted(outcome_columns - set(outcomes.columns))
    if missing_context or missing_outcome:
        raise ValueError(
            "V45 predictive columns missing: "
            f"context={missing_context}, outcomes={missing_outcome}"
        )

    context = context_predictions.copy()
    observed = outcomes.copy()
    context["date"] = pd.to_datetime(context["date"], utc=True)
    observed["date"] = pd.to_datetime(observed["date"], utc=True)
    context["fold"] = context["fold"].astype(int)
    observed["fold"] = observed["fold"].astype(int)
    observed["symbol"] = observed["symbol"].astype(str)
    if context.duplicated(["date", "fold", "triplet_key"]).any():
        raise ValueError("V45 context predictions contain duplicate keys")
    if observed.duplicated(["date", "fold", "symbol"]).any():
        raise ValueError("V45 outcomes contain duplicate keys")
    if not np.isfinite(
        observed["action_log_return"].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("V45 outcomes contain non-finite action returns")

    for index in range(3):
        context[f"symbol_{index}"] = context[f"symbol_{index}"].astype(str)
        for model in MODEL_NAMES:
            values = context[f"{model}_raw_excess_{index}"].to_numpy(
                dtype=np.float64
            )
            if not np.isfinite(values).all():
                raise ValueError(f"V45 {model} predictions are non-finite")
    for row in context.itertuples(index=False):
        symbols = tuple(str(getattr(row, f"symbol_{index}")) for index in range(3))
        if len(set(symbols)) != 3 or list(symbols) != sorted(symbols):
            raise ValueError("V45 triplet symbols must be distinct and lexical")

    used_keys = {
        (pd.Timestamp(row.date), int(row.fold), str(getattr(row, f"symbol_{index}")))
        for row in context.itertuples(index=False)
        for index in range(3)
    }
    outcome_keys = set(
        zip(
            observed["date"],
            observed["fold"],
            observed["symbol"],
            strict=True,
        )
    )
    if used_keys != outcome_keys:
        raise ValueError("V45 context and outcome asset-date keys differ")
    return (
        context.sort_values(["date", "fold", "triplet_key"]).reset_index(drop=True),
        observed.sort_values(["date", "fold", "symbol"]).reset_index(drop=True),
    )


def _context_model_metrics(
    symbols: tuple[str, str, str],
    scores: np.ndarray,
    actual: np.ndarray,
    tie_tolerance: float,
) -> dict[str, object]:
    spearman = average_rank_spearman(scores, actual)
    pair_correct = 0
    pair_active = 0
    for left, right in PAIR_INDEXES:
        actual_difference = float(actual[left] - actual[right])
        if abs(actual_difference) <= tie_tolerance:
            continue
        pair_active += 1
        score_difference = float(scores[left] - scores[right])
        pair_correct += int(score_difference * actual_difference > 0.0)
    selected = min(range(3), key=lambda index: (-float(scores[index]), symbols[index]))
    maximum_return = float(actual.max())
    selected_return = float(actual[selected])
    return {
        "spearman": spearman,
        "pair_correct": pair_correct,
        "pair_active": pair_active,
        "pairwise_accuracy": (
            float(pair_correct / pair_active) if pair_active else float("nan")
        ),
        "top1_symbol": symbols[selected],
        "top1_hit": bool(abs(selected_return - maximum_return) <= tie_tolerance),
        "top1_excess": float(selected_return - actual.mean()),
    }


def _all_finite_mean(values: pd.Series) -> float:
    array = values.to_numpy(dtype=np.float64)
    return float(array.mean()) if len(array) and np.isfinite(array).all() else float("nan")


def compute_predictive_metrics(
    context_predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    tie_tolerance: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Compute the frozen V45 triplet, fold-date, and equal-fold metrics."""

    if not math.isfinite(tie_tolerance) or tie_tolerance < 0.0:
        raise ValueError("V45 tie tolerance must be finite and non-negative")
    context, observed = _normalized_context_inputs(context_predictions, outcomes)
    outcome_lookup = {
        (pd.Timestamp(row.date), int(row.fold), str(row.symbol)): float(
            row.action_log_return
        )
        for row in observed.itertuples(index=False)
    }
    context_rows: list[dict[str, object]] = []
    for row in context.itertuples(index=False):
        date = pd.Timestamp(row.date)
        fold = int(row.fold)
        symbols = tuple(str(getattr(row, f"symbol_{index}")) for index in range(3))
        actual = np.asarray(
            [outcome_lookup[(date, fold, symbol)] for symbol in symbols],
            dtype=np.float64,
        )
        record: dict[str, object] = {
            "date": date,
            "fold": fold,
            "triplet_key": str(row.triplet_key),
            **{f"symbol_{index}": symbol for index, symbol in enumerate(symbols)},
            **{
                f"action_log_return_{index}": float(actual[index])
                for index in range(3)
            },
        }
        for model in MODEL_NAMES:
            scores = np.asarray(
                [
                    float(getattr(row, f"{model}_raw_excess_{index}"))
                    for index in range(3)
                ],
                dtype=np.float64,
            )
            metrics = _context_model_metrics(
                symbols, scores, actual, float(tie_tolerance)
            )
            record.update(
                {f"{model}_{name}": value for name, value in metrics.items()}
            )
        context_rows.append(record)
    context_frame = pd.DataFrame(context_rows).sort_values(
        ["date", "fold", "triplet_key"]
    ).reset_index(drop=True)

    daily_rows: list[dict[str, object]] = []
    for (date, fold), frame in context_frame.groupby(["date", "fold"], sort=True):
        record = {
            "date": pd.Timestamp(date),
            "fold": int(fold),
            "context_count": int(len(frame)),
        }
        for model in MODEL_NAMES:
            spearman_values = frame[f"{model}_spearman"]
            active = int(frame[f"{model}_pair_active"].sum())
            correct = int(frame[f"{model}_pair_correct"].sum())
            record.update(
                {
                    f"{model}_undefined_spearman_contexts": int(
                        (~np.isfinite(spearman_values.to_numpy(dtype=np.float64))).sum()
                    ),
                    f"{model}_spearman": _all_finite_mean(spearman_values),
                    f"{model}_pair_correct": correct,
                    f"{model}_pair_active": active,
                    f"{model}_pairwise_accuracy": (
                        float(correct / active) if active else float("nan")
                    ),
                    f"{model}_top1_hit_rate": float(
                        frame[f"{model}_top1_hit"].mean()
                    ),
                    f"{model}_top1_excess": float(
                        frame[f"{model}_top1_excess"].mean()
                    ),
                }
            )
        daily_rows.append(record)
    daily_frame = pd.DataFrame(daily_rows).sort_values(
        ["date", "fold"]
    ).reset_index(drop=True)

    folds = sorted(int(value) for value in daily_frame["fold"].unique())
    date_sets = {
        fold: tuple(daily_frame.loc[daily_frame["fold"] == fold, "date"])
        for fold in folds
    }
    if len({values for values in date_sets.values()}) != 1:
        raise ValueError("V45 predictive folds do not share the same dates")

    model_summaries: dict[str, object] = {}
    for model in MODEL_NAMES:
        fold_summaries: dict[str, object] = {}
        for fold in folds:
            frame = daily_frame.loc[daily_frame["fold"] == fold]
            fold_summaries[str(fold)] = {
                "fold_date_count": int(len(frame)),
                "context_count": int(frame["context_count"].sum()),
                "undefined_spearman_contexts": int(
                    frame[f"{model}_undefined_spearman_contexts"].sum()
                ),
                "zero_active_pair_fold_dates": int(
                    (frame[f"{model}_pair_active"] == 0).sum()
                ),
                "mean_spearman": _all_finite_mean(frame[f"{model}_spearman"]),
                "mean_pairwise_accuracy": _all_finite_mean(
                    frame[f"{model}_pairwise_accuracy"]
                ),
                "mean_top1_hit_rate": float(
                    frame[f"{model}_top1_hit_rate"].mean()
                ),
                "mean_top1_excess": float(
                    frame[f"{model}_top1_excess"].mean()
                ),
            }
        aggregate_columns = [
            f"{model}_spearman",
            f"{model}_pairwise_accuracy",
            f"{model}_top1_hit_rate",
            f"{model}_top1_excess",
        ]
        aggregate_daily = daily_frame.groupby("date", sort=True)[
            aggregate_columns
        ].agg(_all_finite_mean)
        model_summaries[model] = {
            "folds": fold_summaries,
            "aggregate": {
                "date_count": int(len(aggregate_daily)),
                "mean_spearman": _all_finite_mean(
                    aggregate_daily[f"{model}_spearman"]
                ),
                "mean_pairwise_accuracy": _all_finite_mean(
                    aggregate_daily[f"{model}_pairwise_accuracy"]
                ),
                "mean_top1_hit_rate": _all_finite_mean(
                    aggregate_daily[f"{model}_top1_hit_rate"]
                ),
                "mean_top1_excess": _all_finite_mean(
                    aggregate_daily[f"{model}_top1_excess"]
                ),
            },
        }

    undefined = int(
        sum(
            int(daily_frame[f"{model}_undefined_spearman_contexts"].sum())
            for model in MODEL_NAMES
        )
    )
    zero_active = int((daily_frame["transformer_pair_active"] == 0).sum())
    summary = {
        "folds": folds,
        "fold_date_count": int(len(daily_frame)),
        "unique_date_count": int(daily_frame["date"].nunique()),
        "context_count": int(len(context_frame)),
        "models": model_summaries,
        "integrity": {
            "undefined_spearman_contexts": undefined,
            "zero_active_pair_fold_dates": zero_active,
            "passed": bool(undefined == 0 and zero_active == 0),
        },
    }
    return context_frame, daily_frame, summary


def top1_excess_block_bootstrap(
    equal_fold_daily_series: pd.Series | np.ndarray | Sequence[float],
    block_lengths: Sequence[int],
    paths: int,
    base_seed: int,
    batch_size: int,
) -> dict[str, dict[str, object]]:
    values = _finite_vector(equal_fold_daily_series, "equal-fold top1 series")
    blocks = tuple(int(value) for value in block_lengths)
    if len(set(blocks)) != len(blocks) or any(value < 1 for value in blocks):
        raise ValueError("V45 block lengths must be unique positive integers")
    if paths < 1 or batch_size < 1:
        raise ValueError("V45 bootstrap paths and batch size must be positive")
    if isinstance(equal_fold_daily_series, pd.Series):
        index = equal_fold_daily_series.index
        if not index.is_unique or not index.is_monotonic_increasing:
            raise ValueError("V45 equal-fold top1 series index must be unique and sorted")

    result: dict[str, dict[str, object]] = {}
    for block_length in blocks:
        seed = int(base_seed) + 1000 + block_length
        rng = np.random.default_rng(seed)
        statistics = np.empty(int(paths), dtype=np.float64)
        cursor = 0
        while cursor < paths:
            current = min(int(batch_size), int(paths) - cursor)
            indexes = circular_block_indices(
                len(values), block_length, current, rng
            )
            statistics[cursor : cursor + current] = values[indexes].mean(axis=1)
            cursor += current
        quantiles = np.quantile(statistics, [0.01, 0.05, 0.5, 0.95, 0.99])
        result[str(block_length)] = {
            "method": "circular_block_bootstrap_equal_fold_daily_mean",
            "block_length": block_length,
            "paths": int(paths),
            "seed": seed,
            "observations_per_path": int(len(values)),
            "mean": float(statistics.mean()),
            "p01": float(quantiles[0]),
            "p05": float(quantiles[1]),
            "median": float(quantiles[2]),
            "p95": float(quantiles[3]),
            "p99": float(quantiles[4]),
        }
    return result


def _normalize_fold_symbols(
    fold_symbols: Mapping[int | str, Sequence[str]],
) -> dict[int, tuple[str, ...]]:
    normalized = {
        int(fold): tuple(str(symbol) for symbol in symbols)
        for fold, symbols in fold_symbols.items()
    }
    if set(normalized) != {1, 2, 3}:
        raise ValueError("V45 portfolio evaluation requires folds 1, 2, and 3")
    all_symbols: list[str] = []
    for fold, symbols in normalized.items():
        if not symbols or len(set(symbols)) != len(symbols) or list(symbols) != sorted(symbols):
            raise ValueError(f"V45 fold {fold} symbols must be unique and lexical")
        all_symbols.extend(symbols)
    if len(set(all_symbols)) != len(all_symbols):
        raise ValueError("V45 fold symbol sets must be asset-disjoint")
    return normalized


def build_portfolio_evaluation(
    positions: pd.DataFrame,
    outcomes: pd.DataFrame,
    fold_symbols: Mapping[int | str, Sequence[str]],
    costs: Sequence[int],
    annualization_days: int,
) -> dict[str, object]:
    """Apply frozen positions to outcomes and produce fold/equal-fold metrics."""

    required_positions = {
        "date",
        "fold",
        "symbol",
        "eligible",
        *STRATEGY_WEIGHT_COLUMNS.values(),
    }
    required_outcomes = {"date", "fold", "symbol", "action_log_return"}
    missing_positions = sorted(required_positions - set(positions.columns))
    missing_outcomes = sorted(required_outcomes - set(outcomes.columns))
    if missing_positions or missing_outcomes:
        raise ValueError(
            "V45 portfolio columns missing: "
            f"positions={missing_positions}, outcomes={missing_outcomes}"
        )
    cost_cells = tuple(int(value) for value in costs)
    if cost_cells != V45_COSTS_BPS:
        raise ValueError("V45 portfolio costs must be exactly 10, 20, and 30 bps")
    if int(annualization_days) != 365:
        raise ValueError("V45 annualization must be exactly 365 days")
    symbols_by_fold = _normalize_fold_symbols(fold_symbols)

    position_frame = positions.copy()
    outcome_frame = outcomes.copy()
    for frame in (position_frame, outcome_frame):
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        frame["fold"] = frame["fold"].astype(int)
        frame["symbol"] = frame["symbol"].astype(str)
    if not pd.api.types.is_bool_dtype(position_frame["eligible"]):
        raise ValueError("V45 eligible column must be boolean")
    if position_frame.duplicated(["date", "fold", "symbol"]).any():
        raise ValueError("V45 positions contain duplicate keys")
    if outcome_frame.duplicated(["date", "fold", "symbol"]).any():
        raise ValueError("V45 portfolio outcomes contain duplicate keys")
    if set(position_frame["fold"].unique()) != {1, 2, 3}:
        raise ValueError("V45 positions do not cover all three folds")
    if set(outcome_frame["fold"].unique()) != {1, 2, 3}:
        raise ValueError("V45 outcomes do not cover all three folds")

    for row in position_frame.itertuples(index=False):
        if str(row.symbol) not in symbols_by_fold[int(row.fold)]:
            raise ValueError("V45 position symbol entered the wrong fold")
    eligible_positions = position_frame.loc[position_frame["eligible"]]
    eligible_keys = set(
        zip(
            eligible_positions["date"],
            eligible_positions["fold"],
            eligible_positions["symbol"],
            strict=True,
        )
    )
    outcome_keys = set(
        zip(
            outcome_frame["date"],
            outcome_frame["fold"],
            outcome_frame["symbol"],
            strict=True,
        )
    )
    if eligible_keys != outcome_keys:
        raise ValueError("V45 eligible position and outcome keys differ")
    returns = outcome_frame["action_log_return"].to_numpy(dtype=np.float64)
    if not np.isfinite(returns).all():
        raise ValueError("V45 portfolio outcomes are non-finite")

    for column in STRATEGY_WEIGHT_COLUMNS.values():
        weights = position_frame[column].to_numpy(dtype=np.float64)
        if not np.isfinite(weights).all() or bool((weights < 0.0).any()):
            raise ValueError(f"V45 weights are invalid: {column}")
        if bool((position_frame.loc[~position_frame["eligible"], column] != 0.0).any()):
            raise ValueError("V45 ineligible assets must have zero weight")

    date_sets = {
        fold: tuple(
            sorted(position_frame.loc[position_frame["fold"] == fold, "date"].unique())
        )
        for fold in (1, 2, 3)
    }
    if len({values for values in date_sets.values()}) != 1:
        raise ValueError("V45 portfolio folds do not share the same dates")
    dates = pd.DatetimeIndex(date_sets[1])
    if len(dates) < 2:
        raise ValueError("V45 portfolio evaluation requires at least two dates")

    fold_arrays: dict[int, dict[str, object]] = {}
    outcome_lookup = {
        (pd.Timestamp(row.date), int(row.fold), str(row.symbol)): float(
            row.action_log_return
        )
        for row in outcome_frame.itertuples(index=False)
    }
    position_lookup = {
        (pd.Timestamp(row.date), int(row.fold), str(row.symbol)): row
        for row in position_frame.itertuples(index=False)
    }
    for fold in (1, 2, 3):
        symbols = symbols_by_fold[fold]
        actual = np.zeros((len(dates), len(symbols)), dtype=np.float64)
        strategy_positions = {
            strategy: np.zeros_like(actual) for strategy in STRATEGY_WEIGHT_COLUMNS
        }
        for day, date in enumerate(dates):
            for asset, symbol in enumerate(symbols):
                row = position_lookup.get((pd.Timestamp(date), fold, symbol))
                if row is None:
                    continue
                if bool(row.eligible):
                    actual[day, asset] = outcome_lookup[(pd.Timestamp(date), fold, symbol)]
                for strategy, column in STRATEGY_WEIGHT_COLUMNS.items():
                    strategy_positions[strategy][day, asset] = float(
                        getattr(row, column)
                    )
        for strategy, weights in strategy_positions.items():
            if bool((weights.sum(axis=1) > 1.0 + 1e-12).any()):
                raise ValueError(f"V45 {strategy} exceeds maximum gross exposure")
        fold_arrays[fold] = {
            "symbols": symbols,
            "actual": actual,
            "positions": strategy_positions,
        }

    fold_metrics: dict[str, dict[str, dict[str, object]]] = {}
    aggregate_metrics: dict[str, dict[str, object]] = {}
    daily_rows: list[dict[str, object]] = []
    for cost_bps in cost_cells:
        cost_key = str(cost_bps)
        fold_metrics[cost_key] = {}
        curves_by_strategy: dict[str, list[dict[str, object]]] = {
            strategy: [] for strategy in STRATEGY_WEIGHT_COLUMNS
        }
        for fold in (1, 2, 3):
            fold_key = str(fold)
            fold_metrics[cost_key][fold_key] = {}
            actual = fold_arrays[fold]["actual"]
            for strategy in STRATEGY_WEIGHT_COLUMNS:
                curve = persistent_portfolio_returns(
                    fold_arrays[fold]["positions"][strategy],
                    actual,
                    float(cost_bps),
                )
                curves_by_strategy[strategy].append(curve)
                metrics = performance_metrics(
                    np.asarray(curve["net_return"]),
                    np.asarray(curve["turnover"]),
                    np.asarray(curve["cost"]),
                    annualization_days=int(annualization_days),
                )
                fold_metrics[cost_key][fold_key][strategy] = metrics
                equity = np.cumprod(1.0 + np.asarray(curve["net_return"]))
                for day, date in enumerate(dates):
                    daily_rows.append(
                        {
                            "date": date,
                            "cost_bps": cost_bps,
                            "scope": f"fold_{fold}",
                            "strategy": strategy,
                            "gross_return": float(curve["gross_return"][day]),
                            "turnover": float(curve["turnover"][day]),
                            "cost": float(curve["cost"][day]),
                            "net_return": float(curve["net_return"][day]),
                            "equity": float(equity[day]),
                        }
                    )
        aggregate_metrics[cost_key] = {}
        for strategy, curves in curves_by_strategy.items():
            aggregate = {
                name: np.mean(
                    np.stack([np.asarray(curve[name]) for curve in curves]), axis=0
                )
                for name in ("gross_return", "turnover", "cost", "net_return")
            }
            aggregate_metrics[cost_key][strategy] = performance_metrics(
                aggregate["net_return"],
                aggregate["turnover"],
                aggregate["cost"],
                annualization_days=int(annualization_days),
            )
            equity = np.cumprod(1.0 + aggregate["net_return"])
            for day, date in enumerate(dates):
                daily_rows.append(
                    {
                        "date": date,
                        "cost_bps": cost_bps,
                        "scope": "aggregate_equal_fold_capital",
                        "strategy": strategy,
                        "gross_return": float(aggregate["gross_return"][day]),
                        "turnover": float(aggregate["turnover"][day]),
                        "cost": float(aggregate["cost"][day]),
                        "net_return": float(aggregate["net_return"][day]),
                        "equity": float(equity[day]),
                    }
                )
    daily_frame = pd.DataFrame(daily_rows).sort_values(
        ["date", "cost_bps", "scope", "strategy"]
    ).reset_index(drop=True)
    return {
        "daily_frame": daily_frame,
        "fold_metrics": fold_metrics,
        "aggregate_metrics": aggregate_metrics,
    }


def _required_metric(
    mapping: Mapping[str, object],
    key: str,
    label: str,
) -> float:
    if key not in mapping:
        raise ValueError(f"V45 gate input missing {label}: {key}")
    value = float(mapping[key])
    if not math.isfinite(value):
        return float("nan")
    return value


def evaluate_v45_gates(
    predictive_summary: Mapping[str, object],
    fold_metrics: Mapping[str, object],
    aggregate_metrics: Mapping[str, object],
    bootstrap: Mapping[str, object],
    gates: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate every frozen V45 gate cell without tolerance or epsilon."""

    required_true = (
        "mean_spearman_strictly_positive_each_fold",
        "mean_top1_excess_strictly_positive_each_fold",
        "transformer_beats_ridge_on_aggregate_spearman_and_top1_excess",
        "top1_excess_bootstrap_p05_strictly_positive_all_blocks",
        "positive_net_return_each_fold_at_base_cost",
        "aggregate_return_above_both_policy_controls_all_costs",
        "aggregate_sharpe_above_dual_momentum_all_costs",
        "drawdown_gates_apply_all_costs",
        "turnover_not_above_dual_momentum_at_base_cost",
        "paired_return_delta_p05_above_zero_all_controls_and_blocks",
        "all_cells_required",
    )
    if any(gates.get(name) is not True for name in required_true):
        raise ValueError("V45 gate enablement contract drift")
    if float(gates.get("floating_comparison_epsilon", math.nan)) != 0.0:
        raise ValueError("V45 gates forbid a floating comparison epsilon")
    pairwise_threshold = float(
        gates["aggregate_pairwise_accuracy_strictly_above"]
    )
    absolute_drawdown = float(gates["maximum_absolute_drawdown"])
    drawdown_tolerance = float(
        gates["maximum_drawdown_tolerance_vs_dual_momentum"]
    )
    if pairwise_threshold != 0.5 or absolute_drawdown != 0.35 or drawdown_tolerance != 0.05:
        raise ValueError("V45 numeric gate contract drift")

    models = predictive_summary.get("models", {})
    if set(models) != set(MODEL_NAMES):
        raise ValueError("V45 predictive summary model grid drift")
    transformer = models["transformer"]
    ridge = models["ridge"]
    if set(transformer.get("folds", {})) != {"1", "2", "3"}:
        raise ValueError("V45 transformer fold summary drift")
    if set(ridge.get("folds", {})) != {"1", "2", "3"}:
        raise ValueError("V45 Ridge fold summary drift")

    predictive_cells: list[dict[str, object]] = []
    integrity = predictive_summary.get("integrity", {})
    predictive_cells.extend(
        [
            {
                "gate": "no_undefined_spearman_contexts",
                "value": int(integrity.get("undefined_spearman_contexts", -1)),
                "operator": "==",
                "threshold": 0,
                "passed": int(integrity.get("undefined_spearman_contexts", -1)) == 0,
            },
            {
                "gate": "no_zero_active_pair_fold_dates",
                "value": int(integrity.get("zero_active_pair_fold_dates", -1)),
                "operator": "==",
                "threshold": 0,
                "passed": int(integrity.get("zero_active_pair_fold_dates", -1)) == 0,
            },
        ]
    )
    for fold in (1, 2, 3):
        metrics = transformer["folds"][str(fold)]
        for name in ("mean_spearman", "mean_top1_excess"):
            value = _required_metric(metrics, name, f"transformer fold {fold}")
            predictive_cells.append(
                {
                    "gate": f"transformer_{name}_fold_{fold}",
                    "value": value,
                    "operator": ">",
                    "threshold": 0.0,
                    "passed": bool(math.isfinite(value) and value > 0.0),
                }
            )
    transformer_aggregate = transformer["aggregate"]
    ridge_aggregate = ridge["aggregate"]
    pairwise = _required_metric(
        transformer_aggregate,
        "mean_pairwise_accuracy",
        "transformer aggregate",
    )
    predictive_cells.append(
        {
            "gate": "transformer_aggregate_pairwise_accuracy",
            "value": pairwise,
            "operator": ">",
            "threshold": pairwise_threshold,
            "passed": bool(math.isfinite(pairwise) and pairwise > pairwise_threshold),
        }
    )
    for name in ("mean_spearman", "mean_top1_excess"):
        candidate = _required_metric(
            transformer_aggregate, name, "transformer aggregate"
        )
        control = _required_metric(ridge_aggregate, name, "Ridge aggregate")
        predictive_cells.append(
            {
                "gate": f"transformer_above_ridge_{name}",
                "value": candidate,
                "operator": ">",
                "threshold": control,
                "passed": bool(
                    math.isfinite(candidate)
                    and math.isfinite(control)
                    and candidate > control
                ),
            }
        )

    if set(fold_metrics) != {"10", "20", "30"} or set(aggregate_metrics) != {
        "10",
        "20",
        "30",
    }:
        raise ValueError("V45 economic cost grid drift")
    strategies = set(STRATEGY_WEIGHT_COLUMNS)
    economic_cells: list[dict[str, object]] = []
    base_folds = fold_metrics["10"]
    if set(base_folds) != {"1", "2", "3"}:
        raise ValueError("V45 base-cost fold metric grid drift")
    for fold in (1, 2, 3):
        metrics = base_folds[str(fold)]
        if set(metrics) != strategies:
            raise ValueError("V45 fold strategy metric grid drift")
        value = _required_metric(metrics["candidate"], "total_return", "candidate")
        economic_cells.append(
            {
                "gate": f"candidate_positive_return_fold_{fold}_10bps",
                "value": value,
                "operator": ">",
                "threshold": 0.0,
                "passed": bool(value > 0.0),
            }
        )
    for cost in V45_COSTS_BPS:
        metrics = aggregate_metrics[str(cost)]
        if set(metrics) != strategies:
            raise ValueError("V45 aggregate strategy metric grid drift")
        candidate = metrics["candidate"]
        dual = metrics["dual_momentum_30"]
        for control in ("dual_momentum_30", "momentum_gated_equal_weight"):
            candidate_return = _required_metric(
                candidate, "total_return", f"candidate {cost}bps"
            )
            control_return = _required_metric(
                metrics[control], "total_return", f"{control} {cost}bps"
            )
            economic_cells.append(
                {
                    "gate": f"candidate_return_above_{control}_{cost}bps",
                    "value": candidate_return,
                    "operator": ">",
                    "threshold": control_return,
                    "passed": bool(candidate_return > control_return),
                }
            )
        candidate_sharpe = _required_metric(candidate, "sharpe", "candidate")
        dual_sharpe = _required_metric(dual, "sharpe", "dual momentum")
        candidate_drawdown = _required_metric(
            candidate, "max_drawdown", "candidate"
        )
        dual_drawdown = _required_metric(dual, "max_drawdown", "dual momentum")
        economic_cells.extend(
            [
                {
                    "gate": f"candidate_sharpe_above_dual_{cost}bps",
                    "value": candidate_sharpe,
                    "operator": ">",
                    "threshold": dual_sharpe,
                    "passed": bool(candidate_sharpe > dual_sharpe),
                },
                {
                    "gate": f"candidate_absolute_drawdown_{cost}bps",
                    "value": candidate_drawdown,
                    "operator": ">=",
                    "threshold": -absolute_drawdown,
                    "passed": bool(candidate_drawdown >= -absolute_drawdown),
                },
                {
                    "gate": f"candidate_drawdown_vs_dual_{cost}bps",
                    "value": candidate_drawdown,
                    "operator": ">=",
                    "threshold": dual_drawdown - drawdown_tolerance,
                    "passed": bool(
                        candidate_drawdown >= dual_drawdown - drawdown_tolerance
                    ),
                },
            ]
        )
    candidate_turnover = _required_metric(
        aggregate_metrics["10"]["candidate"], "total_turnover", "candidate"
    )
    dual_turnover = _required_metric(
        aggregate_metrics["10"]["dual_momentum_30"],
        "total_turnover",
        "dual momentum",
    )
    economic_cells.append(
        {
            "gate": "candidate_turnover_not_above_dual_10bps",
            "value": candidate_turnover,
            "operator": "<=",
            "threshold": dual_turnover,
            "passed": bool(candidate_turnover <= dual_turnover),
        }
    )

    if set(bootstrap) != {"top1_excess", "economic"}:
        raise ValueError("V45 bootstrap sections drift")
    top1 = bootstrap["top1_excess"]
    economic_bootstrap = bootstrap["economic"]
    if set(top1) != {"7", "21", "63"} or set(economic_bootstrap) != {
        "7",
        "21",
        "63",
    }:
        raise ValueError("V45 bootstrap block grid drift")
    bootstrap_cells: list[dict[str, object]] = []
    for block in V45_BLOCK_LENGTHS:
        top1_p05 = _required_metric(top1[str(block)], "p05", "top1 bootstrap")
        bootstrap_cells.append(
            {
                "gate": f"top1_excess_p05_block_{block}",
                "value": top1_p05,
                "operator": ">",
                "threshold": 0.0,
                "passed": bool(top1_p05 > 0.0),
            }
        )
        comparisons = economic_bootstrap[str(block)].get("comparisons", {})
        if set(comparisons) != {
            "dual_momentum_30",
            "momentum_gated_equal_weight",
        }:
            raise ValueError("V45 economic bootstrap comparison grid drift")
        for control, comparison in comparisons.items():
            delta = comparison.get("paired_total_return_delta", {})
            p05 = _required_metric(delta, "p05", f"{control} bootstrap")
            bootstrap_cells.append(
                {
                    "gate": f"paired_return_delta_p05_{control}_block_{block}",
                    "value": p05,
                    "operator": ">",
                    "threshold": 0.0,
                    "passed": bool(p05 > 0.0),
                }
            )

    all_cells = [*predictive_cells, *economic_cells, *bootstrap_cells]
    return {
        "passed": bool(all(bool(cell["passed"]) for cell in all_cells)),
        "predictive_cells": predictive_cells,
        "economic_cells": economic_cells,
        "bootstrap_cells": bootstrap_cells,
        "cell_count": len(all_cells),
    }
