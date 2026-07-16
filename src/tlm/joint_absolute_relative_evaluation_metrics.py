from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .monte_carlo import paired_block_bootstrap
from .scientific_harness import persistent_portfolio_returns
from .source_domain_one_shot import performance_metrics


STRATEGIES = (
    "candidate",
    "ridge",
    "dual_momentum_30",
    "equal_weight",
    "cash",
)
WEIGHT_COLUMNS = {
    strategy: tuple(f"{strategy}_weight_{slot}" for slot in range(3))
    for strategy in STRATEGIES
}


def _required_float(mapping: Mapping[str, object], key: str) -> float:
    value = float(mapping[key])
    return value if math.isfinite(value) else float("nan")


def _normalize_positions(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date",
        "origin",
        "geometry",
        "fold",
        "triplet_key",
        *(f"symbol_{slot}" for slot in range(3)),
        *(column for columns in WEIGHT_COLUMNS.values() for column in columns),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"V50 position columns missing: {missing}")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], utc=True)
    result["fold"] = result["fold"].astype(int)
    if result.duplicated(
        ["date", "origin", "geometry", "fold", "triplet_key"]
    ).any():
        raise ValueError("V50 positions contain duplicate triplet-date keys")
    weights = result[
        [column for columns in WEIGHT_COLUMNS.values() for column in columns]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(weights).all() or bool((weights < 0).any()):
        raise ValueError("V50 positions contain invalid weights")
    for row in result.itertuples(index=False):
        symbols = tuple(str(getattr(row, f"symbol_{slot}")) for slot in range(3))
        if len(set(symbols)) != 3 or list(symbols) != sorted(symbols):
            raise ValueError("V50 triplets must be distinct and lexical")
        if str(row.triplet_key) != "|".join(symbols):
            raise ValueError("V50 triplet key drift")
        for columns in WEIGHT_COLUMNS.values():
            gross = sum(float(getattr(row, column)) for column in columns)
            if gross > 1.0 / 3.0 + 1e-12:
                raise ValueError("V50 triplet gross exceeds one third")
    return result.sort_values(
        ["origin", "geometry", "fold", "triplet_key", "date"]
    ).reset_index(drop=True)


def _normalize_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "origin", "fold", "symbol", "action_log_return"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"V50 outcome columns missing: {missing}")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], utc=True)
    result["fold"] = result["fold"].astype(int)
    result["symbol"] = result["symbol"].astype(str)
    if result.duplicated(["date", "origin", "fold", "symbol"]).any():
        raise ValueError("V50 outcomes contain duplicate keys")
    values = result["action_log_return"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("V50 outcomes contain non-finite returns")
    return result


def shift_triplet_positions_one_day(positions: pd.DataFrame) -> pd.DataFrame:
    result = _normalize_positions(positions)
    columns = [column for values in WEIGHT_COLUMNS.values() for column in values]
    result[columns] = (
        result.groupby(
            ["origin", "geometry", "fold", "triplet_key"], sort=False
        )[columns]
        .shift(1)
        .fillna(0.0)
    )
    return result


def build_exact_triplet_portfolio(
    positions: pd.DataFrame,
    outcomes: pd.DataFrame,
    costs_bps: Sequence[int],
    *,
    annualization_days: int = 365,
) -> dict[str, object]:
    """Evaluate fixed independent triplet deployments and equal-weight them."""

    position_frame = _normalize_positions(positions)
    outcome_frame = _normalize_outcomes(outcomes)
    costs = tuple(int(value) for value in costs_bps)
    if not costs or len(set(costs)) != len(costs) or any(value < 0 for value in costs):
        raise ValueError("V50 costs must be unique nonnegative integers")
    if int(annualization_days) != 365:
        raise ValueError("V50 annualization must be 365")
    origins = tuple(sorted(position_frame["origin"].unique()))
    geometries = tuple(sorted(position_frame["geometry"].unique()))
    if origins != ("origin_2024", "origin_2025"):
        raise ValueError("V50 origin grid drift")
    if geometries != ("expanding", "rolling"):
        raise ValueError("V50 geometry grid drift")

    outcome_lookup = {
        (pd.Timestamp(row.date), str(row.origin), int(row.fold), str(row.symbol)):
        float(row.action_log_return)
        for row in outcome_frame.itertuples(index=False)
    }
    cell_metrics: dict[str, object] = {}
    aggregate_metrics: dict[str, object] = {}
    daily_rows: list[dict[str, object]] = []
    triplet_count_audit: list[dict[str, object]] = []

    for origin in origins:
        cell_metrics[origin] = {}
        aggregate_metrics[origin] = {}
        for geometry in geometries:
            cell_metrics[origin][geometry] = {}
            aggregate_metrics[origin][geometry] = {}
            fold_curves: dict[int, dict[int, dict[str, dict[str, np.ndarray]]]] = {}
            shared_dates: pd.DatetimeIndex | None = None
            for fold in (1, 2, 3):
                subset = position_frame.loc[
                    (position_frame["origin"] == origin)
                    & (position_frame["geometry"] == geometry)
                    & (position_frame["fold"] == fold)
                ]
                dates = pd.DatetimeIndex(sorted(subset["date"].unique()))
                triplets = tuple(sorted(subset["triplet_key"].unique()))
                if len(triplets) < 1 or len(dates) < 2:
                    raise ValueError("V50 portfolio cell is empty")
                if shared_dates is None:
                    shared_dates = dates
                elif not shared_dates.equals(dates):
                    raise ValueError("V50 folds do not share evaluation dates")
                triplet_count_audit.append(
                    {
                        "origin": origin,
                        "geometry": geometry,
                        "fold": fold,
                        "triplet_count": len(triplets),
                        "date_count": len(dates),
                    }
                )
                date_index = {date: index for index, date in enumerate(dates)}
                curves = {
                    cost: {
                        strategy: {
                            name: np.zeros(len(dates), dtype=np.float64)
                            for name in ("gross_return", "turnover", "cost", "net_return")
                        }
                        for strategy in STRATEGIES
                    }
                    for cost in costs
                }
                for triplet_key, triplet_frame in subset.groupby(
                    "triplet_key", sort=True
                ):
                    current = triplet_frame.sort_values("date")
                    if len(current) != len(dates) or not pd.DatetimeIndex(
                        current["date"]
                    ).equals(dates):
                        raise ValueError("V50 triplet does not span the complete calendar")
                    symbols = tuple(str(current.iloc[0][f"symbol_{slot}"]) for slot in range(3))
                    actual = np.zeros((len(dates), 3), dtype=np.float64)
                    for day, date in enumerate(dates):
                        for slot, symbol in enumerate(symbols):
                            actual[day, slot] = outcome_lookup.get(
                                (date, origin, fold, symbol), 0.0
                            )
                    for strategy, columns in WEIGHT_COLUMNS.items():
                        weights = current[list(columns)].to_numpy(dtype=np.float64)
                        for cost in costs:
                            curve = persistent_portfolio_returns(weights, actual, cost)
                            for name in curves[cost][strategy]:
                                curves[cost][strategy][name] += np.asarray(
                                    curve[name], dtype=np.float64
                                ) / len(triplets)
                fold_curves[fold] = curves
                cell_metrics[origin][geometry][str(fold)] = {}
                for cost in costs:
                    cell_metrics[origin][geometry][str(fold)][str(cost)] = {}
                    for strategy in STRATEGIES:
                        curve = curves[cost][strategy]
                        cell_metrics[origin][geometry][str(fold)][str(cost)][strategy] = (
                            performance_metrics(
                                curve["net_return"],
                                curve["turnover"],
                                curve["cost"],
                                annualization_days,
                            )
                        )

            assert shared_dates is not None
            for cost in costs:
                aggregate_metrics[origin][geometry][str(cost)] = {}
                for strategy in STRATEGIES:
                    aggregate = {
                        name: np.mean(
                            np.stack(
                                [fold_curves[fold][cost][strategy][name] for fold in (1, 2, 3)]
                            ),
                            axis=0,
                        )
                        for name in ("gross_return", "turnover", "cost", "net_return")
                    }
                    aggregate_metrics[origin][geometry][str(cost)][strategy] = (
                        performance_metrics(
                            aggregate["net_return"],
                            aggregate["turnover"],
                            aggregate["cost"],
                            annualization_days,
                        )
                    )
                    equity = np.cumprod(1.0 + aggregate["net_return"])
                    for day, date in enumerate(shared_dates):
                        daily_rows.append(
                            {
                                "date": date,
                                "origin": origin,
                                "geometry": geometry,
                                "cost_bps": cost,
                                "strategy": strategy,
                                "gross_return": float(aggregate["gross_return"][day]),
                                "turnover": float(aggregate["turnover"][day]),
                                "cost": float(aggregate["cost"][day]),
                                "net_return": float(aggregate["net_return"][day]),
                                "equity": float(equity[day]),
                            }
                        )
    return {
        "daily_frame": pd.DataFrame(daily_rows).sort_values(
            ["origin", "geometry", "cost_bps", "strategy", "date"]
        ).reset_index(drop=True),
        "cell_metrics": cell_metrics,
        "aggregate_metrics": aggregate_metrics,
        "triplet_count_audit": triplet_count_audit,
    }


def build_v50_bootstrap(
    daily_frame: pd.DataFrame,
    contract: Mapping[str, object],
) -> dict[str, object]:
    result: dict[str, object] = {}
    base = daily_frame.loc[daily_frame["cost_bps"] == 10].copy()
    for origin in ("origin_2024", "origin_2025"):
        result[origin] = {}
        for geometry in ("expanding", "rolling"):
            subset = base.loc[
                (base["origin"] == origin) & (base["geometry"] == geometry)
            ]
            series = {}
            for strategy in STRATEGIES:
                current = subset.loc[subset["strategy"] == strategy].sort_values("date")
                series[strategy] = current["net_return"].to_numpy(dtype=np.float64)
            result[origin][geometry] = {
                str(block): paired_block_bootstrap(
                    series,
                    "candidate",
                    ["ridge", "dual_momentum_30", "equal_weight"],
                    int(block),
                    int(contract["paths"]),
                    int(contract["base_seed"])
                    + (0 if origin == "origin_2024" else 100_000)
                    + (0 if geometry == "expanding" else 10_000)
                    + int(block),
                    int(contract["batch_size"]),
                )
                for block in contract["block_lengths_days"]
            }
    return result


def evaluate_v50_gates(
    predictive: Mapping[str, object],
    cell_metrics: Mapping[str, object],
    aggregate_metrics: Mapping[str, object],
    bootstrap: Mapping[str, object],
    gates: Mapping[str, object],
) -> dict[str, object]:
    if float(gates["floating_comparison_epsilon"]) != 0.0:
        raise ValueError("V50 floating comparison epsilon must be zero")
    cells: list[dict[str, object]] = []

    def add(name: str, value: float, operator: str, threshold: float) -> None:
        passed = {
            ">": value > threshold,
            ">=": value >= threshold,
            "<=": value <= threshold,
        }[operator]
        cells.append(
            {
                "gate": name,
                "value": value,
                "operator": operator,
                "threshold": threshold,
                "passed": bool(math.isfinite(value) and passed),
            }
        )

    for origin in ("origin_2024", "origin_2025"):
        for geometry in ("expanding", "rolling"):
            summary = predictive[origin][geometry]
            transformer = summary["models"]["transformer"]
            ridge = summary["models"]["ridge"]
            for fold in (1, 2, 3):
                metrics = transformer["folds"][str(fold)]
                add(f"{origin}_{geometry}_fold{fold}_spearman", _required_float(metrics, "mean_spearman"), ">", 0.0)
                add(f"{origin}_{geometry}_fold{fold}_pairwise", _required_float(metrics, "mean_pairwise_accuracy"), ">", 0.5)
                add(f"{origin}_{geometry}_fold{fold}_top1_excess", _required_float(metrics, "mean_top1_excess"), ">", 0.0)
                add(
                    f"{origin}_{geometry}_fold{fold}_return_10bps",
                    _required_float(cell_metrics[origin][geometry][str(fold)]["10"]["candidate"], "total_return"),
                    ">",
                    0.0,
                )
            for metric in ("mean_spearman", "mean_top1_excess"):
                add(
                    f"{origin}_{geometry}_candidate_above_ridge_{metric}",
                    _required_float(transformer["aggregate"], metric),
                    ">",
                    _required_float(ridge["aggregate"], metric),
                )
            for cost in (10, 20, 30):
                metrics = aggregate_metrics[origin][geometry][str(cost)]
                candidate = metrics["candidate"]
                for control in ("ridge", "dual_momentum_30", "equal_weight"):
                    add(
                        f"{origin}_{geometry}_return_above_{control}_{cost}bps",
                        _required_float(candidate, "total_return"),
                        ">",
                        _required_float(metrics[control], "total_return"),
                    )
                add(
                    f"{origin}_{geometry}_sharpe_above_dual_{cost}bps",
                    _required_float(candidate, "sharpe"),
                    ">",
                    _required_float(metrics["dual_momentum_30"], "sharpe"),
                )
                add(
                    f"{origin}_{geometry}_absolute_drawdown_{cost}bps",
                    _required_float(candidate, "max_drawdown"),
                    ">=",
                    -float(gates["maximum_absolute_drawdown"]),
                )
                add(
                    f"{origin}_{geometry}_drawdown_vs_dual_{cost}bps",
                    _required_float(candidate, "max_drawdown"),
                    ">=",
                    _required_float(metrics["dual_momentum_30"], "max_drawdown")
                    - float(gates["maximum_drawdown_tolerance_vs_dual"]),
                )
            base = aggregate_metrics[origin][geometry]["10"]
            add(
                f"{origin}_{geometry}_turnover_vs_dual_10bps",
                _required_float(base["candidate"], "total_turnover"),
                "<=",
                _required_float(base["dual_momentum_30"], "total_turnover"),
            )
            for block in (7, 21, 63):
                current = bootstrap[origin][geometry][str(block)]
                add(
                    f"{origin}_{geometry}_absolute_bootstrap_p05_block{block}",
                    _required_float(current["distributions"]["candidate"]["total_return"], "p05"),
                    ">",
                    0.0,
                )
                for control in ("ridge", "dual_momentum_30", "equal_weight"):
                    add(
                        f"{origin}_{geometry}_delta_{control}_p05_block{block}",
                        _required_float(current["comparisons"][control]["paired_total_return_delta"], "p05"),
                        ">",
                        0.0,
                    )
    return {
        "passed": bool(all(cell["passed"] for cell in cells)),
        "cell_count": len(cells),
        "passed_count": sum(bool(cell["passed"]) for cell in cells),
        "failed_count": sum(not bool(cell["passed"]) for cell in cells),
        "cells": cells,
    }
