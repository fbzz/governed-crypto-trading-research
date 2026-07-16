from __future__ import annotations

import numpy as np
import pandas as pd

from tlm.v37_failure_autopsy import (
    audit_frozen_ledger,
    build_fold_date_diagnostics,
    extract_holding_episodes,
    loss_concentration,
    summarize_ranking,
    summarize_signal_time_regimes,
)


def _fixture_predictions() -> pd.DataFrame:
    dates = pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True)
    rows = []
    q50_by_date = ([0.03, 0.02, 0.01], [0.03, 0.02, 0.01])
    realized_by_date = ([0.01, 0.03, -0.01], [0.04, 0.02, -0.02])
    for day, date in enumerate(dates):
        for index, symbol in enumerate(("AAAUSDT", "BBBUSDT", "CCCUSDT")):
            q50 = q50_by_date[day][index]
            rows.append({
                "date": date,
                "fold": 1,
                "symbol": symbol,
                "context_count": 1,
                "calibrated_q10": q50 - 0.02,
                "calibrated_q50": q50,
                "calibrated_q90": q50 + 0.02,
                "calibrated_log_volatility": -2.0,
                "observed_log_return": realized_by_date[day][index],
                "momentum_30": (
                    0.1 - 0.05 * index
                    if day == 0
                    else -0.1 - 0.05 * index
                ),
                "candidate_weight": 1.0 if index == 0 else 0.0,
            })
    return pd.DataFrame(rows)


def _fixture_daily() -> pd.DataFrame:
    rows = []
    dates = pd.to_datetime(["2026-01-01", "2026-01-02"], utc=True)
    for day, date in enumerate(dates):
        candidate_gross = float(np.expm1([0.01, 0.04][day]))
        for scope in ("fold_1", "aggregate_equal_fold_capital"):
            for strategy, gross, turnover in (
                ("candidate", candidate_gross, 1.0),
                ("dual_momentum_30", [0.01, 0.02][day], 0.0),
                ("equal_weight_buy_hold", [0.01, 0.01][day], 0.0),
            ):
                cost = turnover * 0.001
                rows.append({
                    "date": date,
                    "cost_bps": 10,
                    "scope": scope,
                    "strategy": strategy,
                    "gross_return": gross,
                    "turnover": turnover,
                    "cost": cost,
                    "net_return": gross - cost,
                    "equity": 1.0,
                })
    return pd.DataFrame(rows)


def test_fold_date_diagnostics_measure_frozen_top1_ranking() -> None:
    diagnostics = build_fold_date_diagnostics(
        _fixture_predictions(), _fixture_daily(), cost_bps=10
    )
    assert len(diagnostics) == 2
    assert diagnostics["candidate_active"].all()
    assert diagnostics.iloc[0]["predicted_top1_actual_rank"] == 2
    assert diagnostics.iloc[1]["predicted_top1_actual_rank"] == 1
    summary = summarize_ranking(diagnostics)
    assert summary["top1_hit_rate"] == 0.5
    assert np.isclose(summary["random_top1_expectation"], 1.0 / 3.0)


def test_episode_extraction_uses_consecutive_same_asset_rule() -> None:
    diagnostics = build_fold_date_diagnostics(
        _fixture_predictions(), _fixture_daily(), cost_bps=10
    )
    episodes = extract_holding_episodes(diagnostics, round_trip_cost_rate=0.001)
    assert len(episodes) == 1
    assert episodes.iloc[0]["duration_days"] == 2
    assert episodes.iloc[0]["symbol"] == "AAAUSDT"
    assert np.isclose(
        episodes.iloc[0]["net_additive_return"],
        np.expm1(0.01) + np.expm1(0.04) - 0.002,
    )


def test_loss_concentration_preserves_fixed_top_n() -> None:
    result = loss_concentration(pd.Series([-0.05, -0.03, -0.02, 0.04]), [1, 3, 5])
    assert result["losing_observation_count"] == 3
    assert np.isclose(result["shares"]["1"], 0.5)
    assert np.isclose(result["shares"]["3"], 1.0)
    assert np.isclose(result["shares"]["5"], 1.0)


def test_signal_time_regimes_cover_all_fold_dates_and_controls() -> None:
    diagnostics = build_fold_date_diagnostics(
        _fixture_predictions(), _fixture_daily(), cost_bps=10
    )
    regimes = summarize_signal_time_regimes(
        diagnostics,
        active_only=False,
        fold_capital_weight=1.0,
    )
    assert sum(row["fold_date_count"] for row in regimes) == len(diagnostics)
    assert {row["signal_time_regime"] for row in regimes} == {"risk_off", "risk_on"}
    assert all("dual_momentum_net_mean" in row for row in regimes)
    assert all("equal_weight_net_mean" in row for row in regimes)


def test_frozen_ledger_reconciles_predictions_costs_and_equal_fold_aggregate() -> None:
    daily = _fixture_daily()
    diagnostics = build_fold_date_diagnostics(
        _fixture_predictions(), daily, cost_bps=10
    )
    checks = audit_frozen_ledger(daily, diagnostics, cost_bps=10)
    assert all(checks.values())

    broken = daily.copy()
    mask = (
        (broken["scope"] == "aggregate_equal_fold_capital")
        & (broken["strategy"] == "candidate")
    )
    broken.loc[mask, "gross_return"] += 0.01
    broken_checks = audit_frozen_ledger(broken, diagnostics, cost_bps=10)
    assert not broken_checks["daily_net_equals_gross_minus_cost"]
    assert not broken_checks["aggregate_gross_is_equal_fold_mean"]
