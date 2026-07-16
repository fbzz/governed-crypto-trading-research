import numpy as np
import pandas as pd

from tlm.backtest import (
    run_equal_weight_buy_hold,
    run_long_cash_backtest,
    run_persistent_long_cash_backtest,
)


def test_backtest_selects_top_positive_prediction_or_cash():
    predictions = np.array([[0.02, 0.01], [-0.01, -0.02], [0.01, 0.03]])
    actual = np.log1p(np.array([[0.01, -0.01], [0.05, 0.05], [-0.02, 0.04]]))
    dates = pd.date_range("2024-01-01", periods=3, tz="UTC")
    daily, metrics = run_long_cash_backtest(
        predictions, actual, dates, ["BTC", "ETH"], threshold=0.0, cost_bps=10
    )
    assert daily["asset"].tolist() == ["BTC", "CASH", "ETH"]
    assert daily["turnover"].tolist() == [2.0, 0.0, 2.0]
    assert daily["cost"].sum() == 0.004
    assert metrics["trade_days"] == 2
    assert metrics["trade_count"] == 2
    assert metrics["position_changes"] == 4


def test_each_intraday_trade_has_entry_and_exit_turnover():
    predictions = np.array([[0.02, 0.01], [0.01, 0.02]])
    actual = np.zeros_like(predictions)
    dates = pd.date_range("2024-01-01", periods=2, tz="UTC")
    daily, _ = run_long_cash_backtest(predictions, actual, dates, ["BTC", "ETH"])
    assert daily["turnover"].tolist() == [2.0, 2.0]


def test_buy_hold_charges_only_one_initial_entry():
    log_returns = np.log1p(np.array([[0.01, 0.02], [-0.01, 0.03]]))
    dates = pd.date_range("2024-01-01", periods=2, tz="UTC")
    daily, metrics = run_equal_weight_buy_hold(log_returns, dates, cost_bps=10)
    assert daily["turnover"].tolist() == [1.0, 0.0]
    assert metrics["trade_count"] == 1
    assert metrics["cost_paid"] == 0.001


def test_persistent_backtest_charges_only_transitions_and_final_exit():
    predictions = np.array([[0.02, 0.01], [0.03, 0.01], [0.01, 0.03]])
    actual = np.log1p(np.array([[0.01, 0.0], [0.02, 0.0], [0.0, 0.03]]))
    dates = pd.date_range("2024-01-01", periods=3, tz="UTC")
    daily, metrics = run_persistent_long_cash_backtest(
        predictions, actual, dates, ["BTC", "ETH"], cost_bps=10
    )
    assert daily["asset"].tolist() == ["BTC", "BTC", "ETH"]
    assert daily["turnover"].tolist() == [1.0, 0.0, 3.0]
    assert metrics["trade_count"] == 2
    assert metrics["cost_paid"] == 0.004


def test_always_invested_ignores_negative_score_but_keeps_top_rank():
    predictions = np.array([[-0.02, -0.01], [-0.03, -0.04]])
    actual = np.zeros_like(predictions)
    dates = pd.date_range("2024-01-01", periods=2, tz="UTC")
    daily, _ = run_persistent_long_cash_backtest(
        predictions,
        actual,
        dates,
        ["BTC", "ETH"],
        always_invested=True,
    )
    assert daily["asset"].tolist() == ["ETH", "BTC"]


def test_drawdown_is_measured_from_initial_capital():
    predictions = np.array([[0.1, 0.0]])
    actual = np.log1p(np.array([[-0.2, 0.0]]))
    dates = pd.date_range("2024-01-01", periods=1, tz="UTC")
    _, metrics = run_long_cash_backtest(
        predictions, actual, dates, ["BTC", "ETH"], cost_bps=0
    )
    assert np.isclose(metrics["max_drawdown"], -0.2)
