from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tlm.__main__ import build_parser
from tlm.ranking_excess_failure_autopsy import (
    _seal_packet,
    _v46_report,
    _validate_packet,
    build_context_stability,
    build_fold_date_diagnostics,
    extract_drawdown_episodes,
    extract_holding_episodes,
    loss_concentration,
    reconcile_v45_ledger,
)
from tlm.ranking_excess_screen_metrics import build_portfolio_evaluation


TOLERANCE = 1e-12
BASE_COST_RATE = 10 / 10_000


def _context_stability_fixture() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    date = pd.Timestamp("2025-01-02", tz="UTC")
    context_predictions = pd.DataFrame(
        [
            {
                "date": date,
                "fold": 1,
                "triplet_key": "AAAUSDT|BBBUSDT|CCCUSDT",
                "symbol_0": "AAAUSDT",
                "symbol_1": "BBBUSDT",
                "symbol_2": "CCCUSDT",
                "transformer_raw_excess_0": 1.0,
                "transformer_raw_excess_1": 0.0,
                "transformer_raw_excess_2": -1.0,
                "ridge_raw_excess_0": 0.5,
                "ridge_raw_excess_1": 0.0,
                "ridge_raw_excess_2": -0.5,
            },
            {
                "date": date,
                "fold": 1,
                "triplet_key": "AAAUSDT|BBBUSDT|DDDUSDT",
                "symbol_0": "AAAUSDT",
                "symbol_1": "BBBUSDT",
                "symbol_2": "DDDUSDT",
                "transformer_raw_excess_0": 3.0,
                "transformer_raw_excess_1": -1.0,
                "transformer_raw_excess_2": -2.0,
                "ridge_raw_excess_0": 1.5,
                "ridge_raw_excess_1": -0.5,
                "ridge_raw_excess_2": -1.0,
            },
        ]
    )
    asset_predictions = pd.DataFrame(
        [
            {
                "date": date,
                "fold": 1,
                "symbol": "AAAUSDT",
                "context_count": 2,
                "transformer_raw_excess": 2.0,
                "ridge_raw_excess": 1.0,
                "momentum_30": 0.10,
            },
            {
                "date": date,
                "fold": 1,
                "symbol": "BBBUSDT",
                "context_count": 2,
                "transformer_raw_excess": -0.5,
                "ridge_raw_excess": -0.25,
                "momentum_30": 0.05,
            },
            {
                "date": date,
                "fold": 1,
                "symbol": "CCCUSDT",
                "context_count": 1,
                "transformer_raw_excess": -1.0,
                "ridge_raw_excess": -0.5,
                "momentum_30": -0.01,
            },
            {
                "date": date,
                "fold": 1,
                "symbol": "DDDUSDT",
                "context_count": 1,
                "transformer_raw_excess": -2.0,
                "ridge_raw_excess": -1.0,
                "momentum_30": -0.02,
            },
        ]
    )
    outcomes = pd.DataFrame(
        [
            {
                "date": date,
                "fold": 1,
                "symbol": symbol,
                "action_log_return": value,
            }
            for symbol, value in (
                ("AAAUSDT", 0.02),
                ("BBBUSDT", 0.01),
                ("CCCUSDT", -0.01),
                ("DDDUSDT", -0.02),
            )
        ]
    )
    return context_predictions, asset_predictions, outcomes


def test_context_stability_uses_all_contexts_and_population_std() -> None:
    context, assets, outcomes = _context_stability_fixture()

    stability = build_context_stability(context, assets, outcomes, TOLERANCE)

    aaa = stability.loc[stability["symbol"] == "AAAUSDT"].iloc[0]
    assert aaa["context_count"] == 2
    assert aaa["context_score_mean"] == pytest.approx(2.0)
    # Context scores are [1, 3].  ddof=0 gives 1.0; ddof=1 would give sqrt(2).
    assert aaa["context_score_std_ddof0"] == pytest.approx(1.0)
    assert aaa["context_score_min"] == pytest.approx(1.0)
    assert aaa["context_score_max"] == pytest.approx(3.0)
    assert aaa["positive_score_fraction"] == pytest.approx(1.0)
    assert aaa["triplet_top1_fraction"] == pytest.approx(1.0)

    bbb = stability.loc[stability["symbol"] == "BBBUSDT"].iloc[0]
    assert bbb["context_score_mean"] == pytest.approx(-0.5)
    assert bbb["context_score_std_ddof0"] == pytest.approx(0.5)
    assert bbb["positive_score_fraction"] == pytest.approx(0.0)
    assert bbb["triplet_top1_fraction"] == pytest.approx(0.0)


def test_context_stability_rejects_asset_mean_drift() -> None:
    context, assets, outcomes = _context_stability_fixture()
    assets.loc[assets["symbol"] == "AAAUSDT", "transformer_raw_excess"] = 2.1

    with pytest.raises(ValueError, match="mean|aggregate|asset"):
        build_context_stability(context, assets, outcomes, TOLERANCE)


def test_context_target_is_average_triplet_excess_not_fold_mean_excess() -> None:
    context, assets, outcomes = _context_stability_fixture()

    stability = build_context_stability(context, assets, outcomes, TOLERANCE)

    aaa = stability.loc[stability["symbol"] == "AAAUSDT"].iloc[0]
    # AAA's triplet excess is 0.013333... against [AAA, BBB, CCC] and
    # 0.016666... against [AAA, BBB, DDD], so its registered target is 0.015.
    # Across all four fold assets the mean return is zero, which would instead
    # produce a 0.02 fold-mean excess.  Keeping both columns makes the target
    # distinction explicit and protects the V45 training/evaluation semantics.
    assert aaa["context_averaged_realized_excess_log_return"] == pytest.approx(
        0.015
    )
    assert aaa["fold_excess_log_return"] == pytest.approx(0.02)
    assert aaa["context_averaged_realized_excess_log_return"] != pytest.approx(
        aaa["fold_excess_log_return"]
    )
    assert aaa["transformer_prediction_error"] == pytest.approx(2.0 - 0.015)


def _ranking_fixture() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    dates = pd.date_range("2025-02-01", periods=3, freq="D", tz="UTC")
    symbols = ("AAAUSDT", "BBBUSDT", "CCCUSDT")
    transformer_scores = (
        (-0.003, 0.012, -0.009),
        (0.011, 0.010, -0.021),
        (0.020, 0.005, -0.025),
    )
    simple_returns = (
        (0.00, 0.02, -0.01),
        (0.10, -0.04, 0.00),
        (0.03, -0.02, 0.01),
    )
    held_symbols = ("BBBUSDT", "BBBUSDT", "AAAUSDT")

    context_rows: list[dict[str, object]] = []
    asset_rows: list[dict[str, object]] = []
    position_rows: list[dict[str, object]] = []
    outcome_rows: list[dict[str, object]] = []
    for day, date in enumerate(dates):
        context_rows.append(
            {
                "date": date,
                "fold": 1,
                "triplet_key": "AAAUSDT|BBBUSDT|CCCUSDT",
                "symbol_0": symbols[0],
                "symbol_1": symbols[1],
                "symbol_2": symbols[2],
                **{
                    f"transformer_raw_excess_{index}": transformer_scores[day][
                        index
                    ]
                    for index in range(3)
                },
                "ridge_raw_excess_0": 0.001,
                "ridge_raw_excess_1": 0.000,
                "ridge_raw_excess_2": -0.001,
            }
        )
        for asset, symbol in enumerate(symbols):
            asset_rows.append(
                {
                    "date": date,
                    "fold": 1,
                    "symbol": symbol,
                    "context_count": 1,
                    "transformer_raw_excess": transformer_scores[day][asset],
                    "ridge_raw_excess": (0.001, 0.000, -0.001)[asset],
                    "momentum_30": (0.10, 0.08, -0.02)[asset],
                }
            )
            position_rows.append(
                {
                    "date": date,
                    "fold": 1,
                    "symbol": symbol,
                    "eligible": True,
                    "momentum_30": (0.10, 0.08, -0.02)[asset],
                    "candidate_weight": float(symbol == held_symbols[day]),
                    "dual_momentum_30_weight": float(symbol == "AAAUSDT"),
                    "momentum_gated_equal_weight_weight": 1.0 / 3.0,
                }
            )
            outcome_rows.append(
                {
                    "date": date,
                    "fold": 1,
                    "symbol": symbol,
                    "action_log_return": float(
                        np.log1p(simple_returns[day][asset])
                    ),
                }
            )

    gross = np.asarray([0.02, -0.04, 0.03], dtype=np.float64)
    turnover = np.asarray([1.0, 0.0, 3.0], dtype=np.float64)
    daily_rows: list[dict[str, object]] = []
    for day, date in enumerate(dates):
        for strategy in (
            "candidate",
            "dual_momentum_30",
            "momentum_gated_equal_weight",
        ):
            strategy_gross = gross[day] if strategy == "candidate" else 0.0
            strategy_turnover = turnover[day] if strategy == "candidate" else 0.0
            cost = strategy_turnover * BASE_COST_RATE
            daily_rows.append(
                {
                    "date": date,
                    "cost_bps": 10,
                    "scope": "fold_1",
                    "strategy": strategy,
                    "gross_return": strategy_gross,
                    "turnover": strategy_turnover,
                    "cost": cost,
                    "net_return": strategy_gross - cost,
                    "equity": 1.0,
                }
            )

    context = pd.DataFrame(context_rows)
    assets = pd.DataFrame(asset_rows)
    positions = pd.DataFrame(position_rows)
    outcomes = pd.DataFrame(outcome_rows)
    daily = pd.DataFrame(daily_rows)
    stability = build_context_stability(context, assets, outcomes, TOLERANCE)
    return assets, positions, outcomes, daily, stability


def test_fold_date_diagnostics_distinguish_desired_top_from_held_incumbent() -> None:
    assets, positions, outcomes, daily, stability = _ranking_fixture()

    diagnostics = build_fold_date_diagnostics(
        assets,
        positions,
        outcomes,
        daily,
        stability,
        base_cost_bps=10,
        tie_tolerance=TOLERANCE,
    )

    assert list(diagnostics["position_state"]) == [
        "entry_from_cash",
        "hold_same_non_top1",
        "switch_hurdle",
    ]
    hysteresis_day = diagnostics.iloc[1]
    assert hysteresis_day["desired_top_symbol"] == "AAAUSDT"
    assert hysteresis_day["held_symbol"] == "BBBUSDT"
    # The desired top made +10%, but the frozen incumbent actually held made -4%.
    assert hysteresis_day["candidate_gross_return"] == pytest.approx(-0.04)
    assert hysteresis_day["candidate_gross_return"] != pytest.approx(0.10)


def _position_state_fixture() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    dates = pd.date_range("2025-02-10", periods=7, freq="D", tz="UTC")
    symbols = ("AAAUSDT", "BBBUSDT", "CCCUSDT")
    # Day 6 removes BBB from the eligible asset-prediction universe.  The
    # frozen positions table still records its zero/ineligible row, as V45 did.
    eligible_by_day = (
        symbols,
        symbols,
        symbols,
        symbols,
        symbols,
        ("AAAUSDT", "CCCUSDT"),
        ("AAAUSDT", "CCCUSDT"),
    )
    scores_by_day = (
        {"AAAUSDT": 0.02, "BBBUSDT": 0.00, "CCCUSDT": -0.02},
        {"AAAUSDT": 0.02, "BBBUSDT": 0.00, "CCCUSDT": -0.02},
        {"AAAUSDT": 0.02, "BBBUSDT": 0.00, "CCCUSDT": -0.02},
        {"AAAUSDT": 0.00, "BBBUSDT": 0.02, "CCCUSDT": -0.02},
        {"AAAUSDT": 0.00, "BBBUSDT": 0.02, "CCCUSDT": -0.02},
        {"AAAUSDT": 0.02, "CCCUSDT": -0.02},
        {"AAAUSDT": 0.001, "CCCUSDT": 0.010},
    )
    momentum_by_day = (
        {"AAAUSDT": -0.01, "BBBUSDT": -0.02, "CCCUSDT": -0.03},
        {"AAAUSDT": 0.01, "BBBUSDT": -0.02, "CCCUSDT": -0.03},
        {"AAAUSDT": -0.01, "BBBUSDT": -0.02, "CCCUSDT": -0.03},
        {"AAAUSDT": 0.01, "BBBUSDT": 0.02, "CCCUSDT": -0.03},
        {"AAAUSDT": 0.01, "BBBUSDT": 0.02, "CCCUSDT": -0.03},
        {"AAAUSDT": 0.01, "CCCUSDT": -0.03},
        {"AAAUSDT": 0.01, "CCCUSDT": 0.02},
    )
    held_by_day = (
        None,
        "AAAUSDT",
        None,
        "BBBUSDT",
        "BBBUSDT",
        "AAAUSDT",
        "CCCUSDT",
    )
    turnover_by_day = (0.0, 1.0, 1.0, 1.0, 0.0, 2.0, 3.0)

    asset_rows: list[dict[str, object]] = []
    position_rows: list[dict[str, object]] = []
    outcome_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    daily_rows: list[dict[str, object]] = []
    for day, date in enumerate(dates):
        eligible = set(eligible_by_day[day])
        for symbol in symbols:
            position_rows.append(
                {
                    "date": date,
                    "fold": 1,
                    "symbol": symbol,
                    "eligible": symbol in eligible,
                    "momentum_30": momentum_by_day[day].get(symbol, np.nan),
                    "candidate_weight": float(symbol == held_by_day[day]),
                    "dual_momentum_30_weight": 0.0,
                    "momentum_gated_equal_weight_weight": 0.0,
                }
            )
            if symbol not in eligible:
                continue
            score = scores_by_day[day][symbol]
            momentum = momentum_by_day[day][symbol]
            asset_rows.append(
                {
                    "date": date,
                    "fold": 1,
                    "symbol": symbol,
                    "context_count": 1,
                    "transformer_raw_excess": score,
                    "ridge_raw_excess": score,
                    "momentum_30": momentum,
                }
            )
            outcome_rows.append(
                {
                    "date": date,
                    "fold": 1,
                    "symbol": symbol,
                    "action_log_return": 0.0,
                }
            )
            stability_rows.append(
                {
                    "date": date,
                    "fold": 1,
                    "symbol": symbol,
                    "context_score_std_ddof0": 0.0,
                    "triplet_top1_fraction": float(
                        score == max(scores_by_day[day].values())
                    ),
                }
            )
        for strategy in (
            "candidate",
            "dual_momentum_30",
            "momentum_gated_equal_weight",
        ):
            turnover = turnover_by_day[day] if strategy == "candidate" else 0.0
            cost = turnover * BASE_COST_RATE
            daily_rows.append(
                {
                    "date": date,
                    "cost_bps": 10,
                    "scope": "fold_1",
                    "strategy": strategy,
                    "gross_return": 0.0,
                    "turnover": turnover,
                    "cost": cost,
                    "net_return": -cost,
                    "equity": 1.0,
                }
            )
    return (
        pd.DataFrame(asset_rows),
        pd.DataFrame(position_rows),
        pd.DataFrame(outcome_rows),
        pd.DataFrame(daily_rows),
        pd.DataFrame(stability_rows),
    )


def test_fold_date_states_validate_cash_exit_forced_switch_and_hurdle() -> None:
    assets, positions, outcomes, daily, stability = _position_state_fixture()

    diagnostics = build_fold_date_diagnostics(
        assets,
        positions,
        outcomes,
        daily,
        stability,
        base_cost_bps=10,
        tie_tolerance=TOLERANCE,
    )

    assert diagnostics["position_state"].to_list() == [
        "cash_gate",
        "entry_from_cash",
        "exit_to_cash",
        "entry_from_cash",
        "hold_same_top1",
        "switch_forced_ineligible",
        "switch_hurdle",
    ]
    forced = diagnostics.loc[
        diagnostics["position_state"] == "switch_forced_ineligible"
    ].iloc[0]
    assert forced["previous_held_symbol"] == "BBBUSDT"
    assert forced["held_symbol"] == "AAAUSDT"
    hurdle = diagnostics.loc[
        diagnostics["position_state"] == "switch_hurdle"
    ].iloc[0]
    assert hurdle["previous_held_symbol"] == "AAAUSDT"
    assert hurdle["held_symbol"] == "CCCUSDT"
    assert hurdle["challenger_minus_incumbent_score"] > 0.002


def _episode_fixture() -> pd.DataFrame:
    dates = pd.date_range("2025-03-01", periods=5, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "date": dates,
            "fold": [1] * 5,
            "candidate_active": [False, True, True, True, True],
            "held_symbol": [None, "AAAUSDT", "AAAUSDT", "BBBUSDT", "BBBUSDT"],
            "position_state": [
                "cash_gate",
                "entry_from_cash",
                "hold_same_top1",
                "switch_hurdle",
                "hold_same_top1",
            ],
            "candidate_gross_return": [0.0, 0.01, 0.02, -0.03, 0.04],
            # The switch has one exit and one entry leg.  The final row includes
            # liquidation of BBB, so total frozen turnover is four legs.
            "candidate_turnover": [0.0, 1.0, 0.0, 2.0, 1.0],
            "candidate_cost": [0.0, 0.001, 0.0, 0.002, 0.001],
            "candidate_net_return": [0.0, 0.009, 0.02, -0.032, 0.039],
        }
    )


def test_holding_episodes_allocate_cash_switch_and_final_liquidation() -> None:
    episodes = extract_holding_episodes(_episode_fixture(), BASE_COST_RATE)

    assert list(episodes["symbol"]) == ["AAAUSDT", "BBBUSDT"]
    assert list(episodes["duration_signal_days"]) == [2, 2]
    assert episodes.iloc[0]["gross_compounded_return"] == pytest.approx(
        1.01 * 1.02 - 1.0
    )
    assert episodes.iloc[0]["gross_additive_return"] == pytest.approx(0.03)
    assert episodes.iloc[1]["gross_compounded_return"] == pytest.approx(
        0.97 * 1.04 - 1.0
    )
    assert episodes["allocated_cost"].to_list() == pytest.approx([0.002, 0.002])
    assert episodes.iloc[0]["net_additive_return"] == pytest.approx(0.03 - 0.002)
    assert episodes.iloc[1]["net_additive_return"] == pytest.approx(0.01 - 0.002)
    assert episodes["allocated_cost"].sum() == pytest.approx(
        _episode_fixture()["candidate_cost"].sum()
    )
    assert {
        "duration_signal_days",
        "gross_compounded_return",
        "gross_additive_return",
        "allocated_entry_cost",
        "allocated_exit_cost",
        "allocated_cost",
        "net_additive_return",
        "entry_state",
        "exit_reason",
    }.issubset(episodes.columns)
    assert {"duration_days", "gross_return", "net_return"}.isdisjoint(
        episodes.columns
    )
    assert episodes["exit_reason"].to_list() == [
        "switch_hurdle",
        "final_liquidation",
    ]


def test_loss_concentration_preserves_registered_top_n_cells() -> None:
    result = loss_concentration(
        pd.Series([-0.05, 0.04, -0.03, -0.02]),
        [1, 3, 5],
    )

    assert result["losing_observation_count"] == 3
    assert result["total_loss_magnitude"] == pytest.approx(0.10)
    assert result["shares"] == pytest.approx(
        {"1": 0.50, "3": 1.00, "5": 1.00}
    )


def test_drawdown_episodes_include_initial_capital_as_first_peak() -> None:
    dates = pd.date_range("2025-04-01", periods=5, freq="D", tz="UTC")
    returns = np.asarray([-0.10, 0.05, 0.06, -0.20, 0.25], dtype=np.float64)

    episodes = extract_drawdown_episodes(dates, returns, top_n=2)

    assert len(episodes) == 2
    worst, first = episodes
    assert worst["max_drawdown"] == pytest.approx(-0.20)
    assert first["max_drawdown"] == pytest.approx(-0.10)
    # Without a virtual initial-capital peak, the first -10% day disappears.
    assert first["peak_equity"] == pytest.approx(1.0)
    assert first["trough_equity"] == pytest.approx(0.9)
    assert pd.Timestamp(first["trough_date"]) == dates[0]
    assert pd.Timestamp(first["recovery_date"]) == dates[2]


def _ledger_fixture() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[int, list[str]],
    pd.DataFrame,
]:
    dates = pd.date_range("2025-05-01", periods=2, freq="D", tz="UTC")
    fold_symbols = {
        1: ["AAAUSDT"],
        2: ["BBBUSDT"],
        3: ["CCCUSDT"],
    }
    weights = {
        1: (1.0, 1.0),  # final row carries one liquidation leg
        2: (1.0, 0.0),  # ordinary exit on the final row
        3: (0.0, 1.0),  # entry plus final liquidation on the final row
    }
    simple_returns = {
        1: (0.01, 0.02),
        2: (-0.01, 0.03),
        3: (0.00, -0.02),
    }
    position_rows: list[dict[str, object]] = []
    outcome_rows: list[dict[str, object]] = []
    for fold, symbols in fold_symbols.items():
        symbol = symbols[0]
        for day, date in enumerate(dates):
            weight = weights[fold][day]
            position_rows.append(
                {
                    "date": date,
                    "fold": fold,
                    "symbol": symbol,
                    "eligible": True,
                    "candidate_weight": weight,
                    "dual_momentum_30_weight": weight,
                    "momentum_gated_equal_weight_weight": weight,
                }
            )
            outcome_rows.append(
                {
                    "date": date,
                    "fold": fold,
                    "symbol": symbol,
                    "action_log_return": float(
                        np.log1p(simple_returns[fold][day])
                    ),
                }
            )
    positions = pd.DataFrame(position_rows)
    outcomes = pd.DataFrame(outcome_rows)
    portfolio = build_portfolio_evaluation(
        positions,
        outcomes,
        fold_symbols,
        costs=[10, 20, 30],
        annualization_days=365,
    )
    return positions, outcomes, fold_symbols, portfolio["daily_frame"]


def test_ledger_reconciliation_covers_final_liquidation_and_equal_fold_capital() -> None:
    positions, outcomes, fold_symbols, daily = _ledger_fixture()

    reconciliation = reconcile_v45_ledger(
        positions,
        outcomes,
        daily,
        fold_symbols,
        costs=[10, 20, 30],
        annualization_days=365,
        tolerance=TOLERANCE,
    )

    assert all(reconciliation["checks"].values())
    recomputed = reconciliation["portfolio"]["daily_frame"]
    base_candidate = recomputed.loc[
        (recomputed["cost_bps"] == 10)
        & (recomputed["strategy"] == "candidate")
    ]
    final_fold_turnover = base_candidate.loc[
        base_candidate["scope"].isin(["fold_1", "fold_2", "fold_3"])
        & base_candidate["date"].eq(base_candidate["date"].max())
    ].sort_values("scope")["turnover"]
    assert final_fold_turnover.to_list() == pytest.approx([1.0, 1.0, 2.0])
    aggregate_final = base_candidate.loc[
        base_candidate["scope"].eq("aggregate_equal_fold_capital")
        & base_candidate["date"].eq(base_candidate["date"].max()),
        "turnover",
    ].iloc[0]
    assert aggregate_final == pytest.approx((1.0 + 1.0 + 2.0) / 3.0)


def test_ledger_reconciliation_detects_tampered_equal_fold_row() -> None:
    positions, outcomes, fold_symbols, daily = _ledger_fixture()
    tampered = daily.copy()
    mask = (
        tampered["scope"].eq("aggregate_equal_fold_capital")
        & tampered["strategy"].eq("candidate")
        & tampered["cost_bps"].eq(10)
    )
    tampered.loc[mask, "net_return"] += 0.01

    reconciliation = reconcile_v45_ledger(
        positions,
        outcomes,
        tampered,
        fold_symbols,
        costs=[10, 20, 30],
        annualization_days=365,
        tolerance=TOLERANCE,
    )

    assert not all(reconciliation["checks"].values())


@pytest.mark.parametrize(
    "command",
    [
        "ranking-excess-failure-autopsy-preflight",
        "ranking-excess-failure-autopsy",
    ],
)
def test_parser_registers_both_v46_commands(command: str) -> None:
    parsed = build_parser().parse_args(
        [command, "--config", "configs/v46_ranking_excess_failure_autopsy.yaml"]
    )

    assert parsed.command == command
    assert parsed.config == "configs/v46_ranking_excess_failure_autopsy.yaml"


def _report_fixture() -> dict[str, object]:
    fold_ranking = {
        str(fold): {
            "transformer": {"mean_asset_spearman": 0.01 * fold},
            "held_candidate": {
                "mean_excess_log_return_active": 0.002 * fold,
                "mean_absolute_log_return_active": -0.001 * fold,
            },
        }
        for fold in (1, 2, 3)
    }
    return {
        "economic_diagnostics": {
            "base_cost_candidate_aggregate": {
                "gross_compounded_return": 0.12,
                "net_compounded_return": 0.10,
                "registered_metrics": {
                    "sharpe": 0.75,
                    "max_drawdown": -0.20,
                },
                "turnover_sum": 20.0,
                "additive_cost_sum": 0.02,
            },
            "base_cost_candidate_by_fold": [
                {
                    "fold": fold,
                    "gross_compounded_return": 0.04 - 0.02 * fold,
                    "net_compounded_return": 0.03 - 0.02 * fold,
                    "turnover_sum": float(fold),
                }
                for fold in (1, 2, 3)
            ],
        },
        "concentration": {
            "held_asset_exposure_by_fold": {
                str(fold): {
                    "dominant_symbol": f"ASSET{fold}USDT",
                    "dominant_active_day_share": 0.25 * fold,
                    "effective_assets": 4.0 / fold,
                }
                for fold in (1, 2, 3)
            },
            "aggregate_candidate_losing_days": {
                "shares": {"1": 0.40, "5": 0.85}
            },
        },
        "failure_attribution": {
            "relative_ranking_absolute_return_gap_observed": True,
            "worst_day_explains_majority_of_losing_day_magnitude": False,
            "fold_3_return_gate_failed": True,
            "fold_3_gross_return_was_negative": True,
            "momentum_regime_association_sign_stable": False,
            "counterfactual_policy_tested": False,
        },
        "drawdown_diagnostics": {
            "base_cost_candidate_worst_drawdown_attribution": {
                "episode": {
                    "peak_date": "2025-01-01T00:00:00+00:00",
                    "trough_date": "2025-02-01T00:00:00+00:00",
                    "recovery_date": None,
                    "max_drawdown": -0.20,
                }
            }
        },
        "ranking_summary": {
            "context_averaged_asset": {"by_fold": fold_ranking}
        },
        "context_stability": {
            "pooled_asset_dates_descriptive": {
                "transformer": {"mean_score_std_ddof0": 0.003},
                "fixed_correlations": {
                    "transformer_std_vs_absolute_error": {"pearson": 0.25}
                },
            }
        },
    }


def test_v46_report_renders_additive_cost_and_conditional_attribution() -> None:
    result = _report_fixture()

    report = _v46_report(result)

    assert "additive cost **2.00%**" in report
    assert "The central failure is visible in fold 3" in report
    assert "The failure was not a one-observation collapse." in report
    assert "recovery occurred on **not recovered**" in report

    alternative = deepcopy(result)
    alternative["failure_attribution"][
        "relative_ranking_absolute_return_gap_observed"
    ] = False
    alternative["failure_attribution"][
        "worst_day_explains_majority_of_losing_day_magnitude"
    ] = True
    alternative_report = _v46_report(alternative)
    assert (
        "did not satisfy the fixed definition of a relative-ranking/absolute-return gap"
        in alternative_report
    )
    assert "The failure was dominated by one observation." in alternative_report


def test_packet_seal_cache_exact_grid_and_artifact_tamper(tmp_path: Path) -> None:
    output = tmp_path / "v46_packet"
    output.mkdir()
    (output / "audit.json").write_text('{"passed": true}', encoding="utf-8")
    (output / "report.md").write_text("frozen report\n", encoding="utf-8")
    spec_sha = "a" * 64
    result = {
        "mode": "run",
        "decision": "v45_retirement_confirmed_diagnostic_only",
        "autopsy_spec": {"autopsy_spec_sha256": spec_sha},
        "audit": {"passed": True},
    }
    core_files = ("audit.json", "report.md")
    required_files = (
        *core_files,
        "artifact_manifest.json",
        "result.json",
        "completion_receipt.json",
    )

    _seal_packet(output, result, core_files)
    first = _validate_packet(output, spec_sha, "run", required_files)
    second = _validate_packet(output, spec_sha, "run", required_files)

    assert first == second == result
    with pytest.raises(RuntimeError, match="grid contains duplicates"):
        _validate_packet(
            output,
            spec_sha,
            "run",
            (*required_files, "report.md"),
        )
    with pytest.raises(RuntimeError, match="manifest file grid drift"):
        _validate_packet(
            output,
            spec_sha,
            "run",
            tuple(path for path in required_files if path != "audit.json"),
        )

    (output / "report.md").write_text("tampered report\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="cached artifact drift: report.md"):
        _validate_packet(output, spec_sha, "run", required_files)
