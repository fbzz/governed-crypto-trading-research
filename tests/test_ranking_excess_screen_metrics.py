from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from tlm.ranking_excess_screen_metrics import (
    average_rank_spearman,
    build_portfolio_evaluation,
    compute_predictive_metrics,
    evaluate_v45_gates,
    top1_excess_block_bootstrap,
)


def test_average_rank_spearman_uses_average_ties_and_marks_undefined() -> None:
    value = average_rank_spearman(
        [3.0, 2.0, 1.0],
        [4.0, 4.0, 1.0],
    )
    expected = pd.Series([3.0, 2.0, 1.0]).rank(method="average").corr(
        pd.Series([4.0, 4.0, 1.0]).rank(method="average")
    )
    assert value == pytest.approx(expected)
    assert np.isnan(average_rank_spearman([1.0, 1.0, 1.0], [3.0, 2.0, 1.0]))
    with pytest.raises(ValueError, match="share one-dimensional shape"):
        average_rank_spearman([1.0, 2.0], [1.0, 2.0, 3.0])


def _context_row(
    date: pd.Timestamp,
    fold: int,
    key: str,
    symbols: tuple[str, str, str],
    transformer: tuple[float, float, float],
    ridge: tuple[float, float, float],
) -> dict[str, object]:
    return {
        "date": date,
        "fold": fold,
        "triplet_key": key,
        **{f"symbol_{index}": symbol for index, symbol in enumerate(symbols)},
        **{
            f"transformer_raw_excess_{index}": transformer[index]
            for index in range(3)
        },
        **{f"ridge_raw_excess_{index}": ridge[index] for index in range(3)},
    }


def test_predictive_metrics_pool_pairs_within_fold_date_before_averaging() -> None:
    dates = pd.date_range("2025-01-01", periods=2, freq="D", tz="UTC")
    contexts = []
    outcomes = []
    actual = {"A": 0.03, "B": 0.03, "C": 0.01, "D": -0.01}
    for fold in (1, 2, 3):
        for date in dates:
            contexts.extend(
                [
                    _context_row(
                        date,
                        fold,
                        "A|B|C",
                        ("A", "B", "C"),
                        (3.0, 2.0, 1.0),
                        (2.0, 3.0, 1.0),
                    ),
                    _context_row(
                        date,
                        fold,
                        "A|C|D",
                        ("A", "C", "D"),
                        (1.0, 2.0, 3.0),
                        (3.0, 2.0, 1.0),
                    ),
                ]
            )
            outcomes.extend(
                {
                    "date": date,
                    "fold": fold,
                    "symbol": symbol,
                    "action_log_return": value,
                }
                for symbol, value in actual.items()
            )

    context_frame, daily_frame, summary = compute_predictive_metrics(
        pd.DataFrame(contexts),
        pd.DataFrame(outcomes),
        tie_tolerance=1e-12,
    )

    first = daily_frame.iloc[0]
    assert first["transformer_pair_correct"] == 2
    assert first["transformer_pair_active"] == 5
    assert first["transformer_pairwise_accuracy"] == pytest.approx(0.4)
    assert summary["models"]["transformer"]["aggregate"][
        "mean_pairwise_accuracy"
    ] == pytest.approx(0.4)
    assert summary["fold_date_count"] == 6
    assert summary["context_count"] == 12
    assert summary["integrity"]["passed"]
    assert len(context_frame) == 12


def test_predictive_top1_uses_lexical_score_tie_and_tolerant_actual_hit() -> None:
    date = pd.Timestamp("2025-01-01", tz="UTC")
    contexts = pd.DataFrame(
        [
            _context_row(
                date,
                1,
                "A|B|C",
                ("A", "B", "C"),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, -1.0),
            )
        ]
    )
    outcomes = pd.DataFrame(
        [
            {"date": date, "fold": 1, "symbol": "A", "action_log_return": 0.03},
            {
                "date": date,
                "fold": 1,
                "symbol": "B",
                "action_log_return": 0.03 + 5e-13,
            },
            {"date": date, "fold": 1, "symbol": "C", "action_log_return": -0.01},
        ]
    )
    context_frame, _, _ = compute_predictive_metrics(
        contexts, outcomes, tie_tolerance=1e-12
    )
    row = context_frame.iloc[0]
    assert row["transformer_top1_symbol"] == "A"
    assert bool(row["transformer_top1_hit"])
    assert row["transformer_pair_active"] == 2
    assert row["transformer_top1_excess"] == pytest.approx(
        0.03 - np.mean([0.03, 0.03 + 5e-13, -0.01])
    )


def test_predictive_metrics_preserve_undefined_spearman_and_zero_pair_failure() -> None:
    date = pd.Timestamp("2025-01-01", tz="UTC")
    context = pd.DataFrame(
        [
            _context_row(
                date,
                1,
                "A|B|C",
                ("A", "B", "C"),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            )
        ]
    )
    outcomes = pd.DataFrame(
        [
            {"date": date, "fold": 1, "symbol": symbol, "action_log_return": 0.01}
            for symbol in ("A", "B", "C")
        ]
    )
    _, daily, summary = compute_predictive_metrics(context, outcomes, 1e-12)
    assert np.isnan(daily.iloc[0]["transformer_spearman"])
    assert np.isnan(daily.iloc[0]["transformer_pairwise_accuracy"])
    assert summary["integrity"] == {
        "undefined_spearman_contexts": 2,
        "zero_active_pair_fold_dates": 1,
        "passed": False,
    }


def test_predictive_aggregate_propagates_an_undefined_fold_date() -> None:
    date = pd.Timestamp("2025-01-01", tz="UTC")
    contexts = []
    outcomes = []
    for fold in (1, 2, 3):
        symbols = tuple(f"F{fold}{suffix}" for suffix in ("A", "B", "C"))
        contexts.append(
            _context_row(
                date,
                fold,
                "|".join(symbols),
                symbols,
                (3.0, 2.0, 1.0),
                (3.0, 2.0, 1.0),
            )
        )
        actual = (0.01, 0.01, 0.01) if fold == 1 else (0.03, 0.02, 0.01)
        outcomes.extend(
            {
                "date": date,
                "fold": fold,
                "symbol": symbol,
                "action_log_return": value,
            }
            for symbol, value in zip(symbols, actual, strict=True)
        )

    _, daily, summary = compute_predictive_metrics(
        pd.DataFrame(contexts), pd.DataFrame(outcomes), 1e-12
    )

    invalid_fold = daily.loc[daily["fold"] == 1].iloc[0]
    assert np.isnan(invalid_fold["transformer_spearman"])
    assert np.isnan(invalid_fold["transformer_pairwise_accuracy"])
    aggregate = summary["models"]["transformer"]["aggregate"]
    assert np.isnan(aggregate["mean_spearman"])
    assert np.isnan(aggregate["mean_pairwise_accuracy"])


def test_top1_bootstrap_resamples_daily_series_with_registered_seed_rule() -> None:
    series = pd.Series(
        np.full(20, 0.01),
        index=pd.date_range("2025-01-01", periods=20, freq="D", tz="UTC"),
    )
    first = top1_excess_block_bootstrap(
        series,
        [7, 21, 63],
        paths=100,
        base_seed=20260713,
        batch_size=17,
    )
    second = top1_excess_block_bootstrap(
        series,
        [7, 21, 63],
        paths=100,
        base_seed=20260713,
        batch_size=17,
    )
    assert first == second
    assert set(first) == {"7", "21", "63"}
    for block in (7, 21, 63):
        assert first[str(block)]["seed"] == 20260713 + 1000 + block
        assert first[str(block)]["p05"] == pytest.approx(0.01)


def _portfolio_fixture() -> tuple[pd.DataFrame, pd.DataFrame, dict[int, list[str]]]:
    dates = pd.date_range("2025-01-01", periods=2, freq="D", tz="UTC")
    positions = []
    outcomes = []
    fold_symbols = {}
    for fold in (1, 2, 3):
        symbols = [f"F{fold}A", f"F{fold}B"]
        fold_symbols[fold] = symbols
        for day, date in enumerate(dates):
            for asset, symbol in enumerate(symbols):
                positions.append(
                    {
                        "date": date,
                        "fold": fold,
                        "symbol": symbol,
                        "eligible": True,
                        "candidate_weight": float(asset == day),
                        "dual_momentum_30_weight": float(asset == 0),
                        "momentum_gated_equal_weight_weight": 0.5,
                    }
                )
                outcomes.append(
                    {
                        "date": date,
                        "fold": fold,
                        "symbol": symbol,
                        "action_log_return": 0.0,
                    }
                )
    return pd.DataFrame(positions), pd.DataFrame(outcomes), fold_symbols


def test_portfolio_evaluation_uses_frozen_cost_and_liquidation_accounting() -> None:
    positions, outcomes, fold_symbols = _portfolio_fixture()
    result = build_portfolio_evaluation(
        positions,
        outcomes,
        fold_symbols,
        costs=[10, 20, 30],
        annualization_days=365,
    )
    fold_candidate = result["fold_metrics"]["10"]["1"]["candidate"]
    assert fold_candidate["total_turnover"] == pytest.approx(4.0)
    assert fold_candidate["total_cost"] == pytest.approx(0.004)
    candidate_daily = result["daily_frame"].loc[
        (result["daily_frame"]["cost_bps"] == 10)
        & (result["daily_frame"]["scope"] == "fold_1")
        & (result["daily_frame"]["strategy"] == "candidate")
    ]
    np.testing.assert_allclose(candidate_daily["turnover"], [1.0, 3.0])
    np.testing.assert_allclose(candidate_daily["net_return"], [-0.001, -0.003])
    aggregate = result["aggregate_metrics"]["10"]["candidate"]
    assert aggregate["total_turnover"] == pytest.approx(4.0)
    assert set(result["fold_metrics"]) == {"10", "20", "30"}


def _passing_predictive_summary() -> dict[str, object]:
    transformer_folds = {
        str(fold): {
            "mean_spearman": 0.20,
            "mean_top1_excess": 0.01,
        }
        for fold in (1, 2, 3)
    }
    ridge_folds = {
        str(fold): {
            "mean_spearman": 0.10,
            "mean_top1_excess": 0.005,
        }
        for fold in (1, 2, 3)
    }
    return {
        "models": {
            "transformer": {
                "folds": transformer_folds,
                "aggregate": {
                    "mean_spearman": 0.20,
                    "mean_pairwise_accuracy": 0.60,
                    "mean_top1_excess": 0.01,
                },
            },
            "ridge": {
                "folds": ridge_folds,
                "aggregate": {
                    "mean_spearman": 0.10,
                    "mean_pairwise_accuracy": 0.55,
                    "mean_top1_excess": 0.005,
                },
            },
        },
        "integrity": {
            "undefined_spearman_contexts": 0,
            "zero_active_pair_fold_dates": 0,
            "passed": True,
        },
    }


def _metric(
    total_return: float,
    sharpe: float,
    drawdown: float,
    turnover: float,
) -> dict[str, float]:
    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
        "total_turnover": turnover,
    }


def _passing_economic_metrics() -> tuple[dict, dict]:
    folds = {
        str(cost): {
            str(fold): {
                "candidate": _metric(0.20, 1.2, -0.10, 1.0),
                "dual_momentum_30": _metric(0.10, 0.8, -0.12, 2.0),
                "momentum_gated_equal_weight": _metric(0.05, 0.5, -0.20, 2.5),
            }
            for fold in (1, 2, 3)
        }
        for cost in (10, 20, 30)
    }
    aggregate = {
        str(cost): {
            "candidate": _metric(0.20, 1.2, -0.10, 1.0),
            "dual_momentum_30": _metric(0.10, 0.8, -0.12, 2.0),
            "momentum_gated_equal_weight": _metric(0.05, 0.5, -0.20, 2.5),
        }
        for cost in (10, 20, 30)
    }
    return folds, aggregate


def _passing_bootstrap() -> dict[str, object]:
    return {
        "top1_excess": {
            str(block): {"p05": 0.001} for block in (7, 21, 63)
        },
        "economic": {
            str(block): {
                "comparisons": {
                    control: {"paired_total_return_delta": {"p05": 0.001}}
                    for control in (
                        "dual_momentum_30",
                        "momentum_gated_equal_weight",
                    )
                }
            }
            for block in (7, 21, 63)
        },
    }


def _gates() -> dict[str, object]:
    return {
        "mean_spearman_strictly_positive_each_fold": True,
        "mean_top1_excess_strictly_positive_each_fold": True,
        "aggregate_pairwise_accuracy_strictly_above": 0.50,
        "transformer_beats_ridge_on_aggregate_spearman_and_top1_excess": True,
        "top1_excess_bootstrap_p05_strictly_positive_all_blocks": True,
        "positive_net_return_each_fold_at_base_cost": True,
        "aggregate_return_above_both_policy_controls_all_costs": True,
        "aggregate_sharpe_above_dual_momentum_all_costs": True,
        "maximum_absolute_drawdown": 0.35,
        "maximum_drawdown_tolerance_vs_dual_momentum": 0.05,
        "drawdown_gates_apply_all_costs": True,
        "turnover_not_above_dual_momentum_at_base_cost": True,
        "paired_return_delta_p05_above_zero_all_controls_and_blocks": True,
        "all_cells_required": True,
        "floating_comparison_epsilon": 0.0,
    }


def test_v45_gate_evaluator_preserves_all_cells_and_exact_operators() -> None:
    fold_metrics, aggregate_metrics = _passing_economic_metrics()
    result = evaluate_v45_gates(
        _passing_predictive_summary(),
        fold_metrics,
        aggregate_metrics,
        _passing_bootstrap(),
        _gates(),
    )
    assert result["passed"]
    assert result["cell_count"] == (
        len(result["predictive_cells"])
        + len(result["economic_cells"])
        + len(result["bootstrap_cells"])
    )
    assert len(result["bootstrap_cells"]) == 9

    boundary_summary = _passing_predictive_summary()
    boundary_summary["models"]["transformer"]["aggregate"][
        "mean_pairwise_accuracy"
    ] = 0.50
    failed = evaluate_v45_gates(
        boundary_summary,
        fold_metrics,
        aggregate_metrics,
        _passing_bootstrap(),
        _gates(),
    )
    assert not failed["passed"]
    pair_cell = next(
        cell
        for cell in failed["predictive_cells"]
        if cell["gate"] == "transformer_aggregate_pairwise_accuracy"
    )
    assert not pair_cell["passed"]

    inclusive = deepcopy(aggregate_metrics)
    for cost in (10, 20, 30):
        inclusive[str(cost)]["candidate"]["max_drawdown"] = -0.35
        inclusive[str(cost)]["dual_momentum_30"]["max_drawdown"] = -0.30
    inclusive["10"]["candidate"]["total_turnover"] = inclusive["10"][
        "dual_momentum_30"
    ]["total_turnover"]
    boundary_pass = evaluate_v45_gates(
        _passing_predictive_summary(),
        fold_metrics,
        inclusive,
        _passing_bootstrap(),
        _gates(),
    )
    assert boundary_pass["passed"]


def test_v45_gate_evaluator_fails_any_single_bootstrap_cell() -> None:
    fold_metrics, aggregate_metrics = _passing_economic_metrics()
    bootstrap = _passing_bootstrap()
    bootstrap["economic"]["63"]["comparisons"]["dual_momentum_30"][
        "paired_total_return_delta"
    ]["p05"] = 0.0
    result = evaluate_v45_gates(
        _passing_predictive_summary(),
        fold_metrics,
        aggregate_metrics,
        bootstrap,
        _gates(),
    )
    assert not result["passed"]
    assert any(
        not cell["passed"]
        and cell["gate"]
        == "paired_return_delta_p05_dual_momentum_30_block_63"
        for cell in result["bootstrap_cells"]
    )
