import numpy as np
import pandas as pd

from tlm.validation_suite import (
    _beats_on_required_metrics,
    _delayed_signal_stress,
    _finite_numbers,
)


def test_signal_delay_moves_first_decision_to_cash():
    predictions = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=3, tz="UTC"),
        "pred_BTC": [1.0, -1.0, -1.0],
        "pred_ETH": [-1.0, 1.0, -1.0],
        "actual_BTC": np.log1p([0.1, 0.0, 0.0]),
        "actual_ETH": np.log1p([0.0, 0.1, 0.0]),
    })
    metrics = _delayed_signal_stress(predictions, ["BTC", "ETH"], cost_bps=0)
    assert metrics["trade_days"] == 2
    assert metrics["observations"] == 3


def test_required_metrics_gate_needs_return_sharpe_and_drawdown():
    baseline = {"total_return": 1.0, "sharpe": 0.8, "max_drawdown": -0.5}
    candidate = {"total_return": 1.2, "sharpe": 0.9, "max_drawdown": -0.4}
    assert _beats_on_required_metrics(candidate, baseline)
    candidate["max_drawdown"] = -0.6
    assert not _beats_on_required_metrics(candidate, baseline)


def test_recursive_finite_number_check_rejects_nan():
    assert _finite_numbers({"a": [1.0, {"b": -2.0}]})
    assert not _finite_numbers({"a": [1.0, {"b": float("nan")}]})
