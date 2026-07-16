from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _metrics(daily: pd.DataFrame) -> dict[str, float | int]:
    returns = daily["net_return"]
    equity = daily["equity"]
    total_return = float(equity.iloc[-1] - 1.0)
    years = max(len(daily) / 365.0, 1.0 / 365.0)
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if equity.iloc[-1] > 0 else -1.0
    std = float(returns.std(ddof=1))
    sharpe = float(math.sqrt(365.0) * returns.mean() / std) if std > 0 else 0.0
    running_peak = equity.cummax().clip(lower=1.0)
    drawdown = equity / running_peak - 1.0
    active = daily["asset"] != "CASH"
    trade_started = daily.get("trade_started", daily["turnover"] > 0)
    trades = int(trade_started.sum())
    changes = int(round(daily["turnover"].sum()))
    return {
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "turnover": float(daily["turnover"].sum()),
        "cost_paid": float(daily["cost"].sum()),
        "hit_rate": float((daily.loc[active, "gross_return"] > 0).mean()) if active.any() else 0.0,
        "trade_days": int(active.sum()),
        "trade_count": trades,
        "position_changes": changes,
        "pnl_per_position_change": float(returns.sum() / max(changes, 1)),
        "pnl_per_trade": float(returns.sum() / max(trades, 1)),
        "observations": int(len(daily)),
    }


def run_long_cash_backtest(
    predictions: np.ndarray,
    actual_log_returns: np.ndarray,
    dates: pd.DatetimeIndex,
    assets: list[str] | tuple[str, ...],
    threshold: float = 0.0,
    cost_bps: float = 10.0,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    if predictions.shape != actual_log_returns.shape:
        raise ValueError("Prediction and actual arrays must have the same shape")
    if len(predictions) != len(dates):
        raise ValueError("Dates must match predictions")
    assets = list(assets)
    records: list[dict[str, object]] = []
    rate = cost_bps / 10_000.0

    for index, date in enumerate(dates):
        best = int(np.argmax(predictions[index]))
        selected = best if predictions[index, best] > threshold else -1
        # This target is next-day open-to-close, so each active observation is
        # an intraday round trip: enter from cash at open, exit to cash at close.
        turnover = 2.0 if selected >= 0 else 0.0
        cost = turnover * rate
        gross = float(np.expm1(actual_log_returns[index, selected])) if selected >= 0 else 0.0
        net = gross - cost
        records.append({
            "date": date,
            "asset": assets[selected] if selected >= 0 else "CASH",
            "prediction": float(predictions[index, best]),
            "gross_return": gross,
            "turnover": turnover,
            "cost": cost,
            "net_return": net,
            "trade_started": selected >= 0,
        })
    daily = pd.DataFrame.from_records(records).set_index("date")
    daily["equity"] = (1.0 + daily["net_return"]).cumprod()
    return daily, _metrics(daily)


def run_equal_weight_intraday_benchmark(
    actual_log_returns: np.ndarray,
    dates: pd.DatetimeIndex,
    cost_bps: float,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    gross = np.expm1(actual_log_returns).mean(axis=1)
    daily = pd.DataFrame(index=dates)
    daily["asset"] = "EQUAL_WEIGHT_INTRADAY"
    daily["prediction"] = np.nan
    daily["gross_return"] = gross
    daily["turnover"] = 2.0
    daily["cost"] = daily["turnover"] * (cost_bps / 10_000.0)
    daily["net_return"] = daily["gross_return"] - daily["cost"]
    daily["trade_started"] = True
    daily["equity"] = (1.0 + daily["net_return"]).cumprod()
    metrics = _metrics(daily)
    metrics["hit_rate"] = float((gross > 0).mean())
    metrics["trade_days"] = int(len(daily))
    return daily, metrics


def run_equal_weight_buy_hold(
    close_to_close_log_returns: np.ndarray,
    dates: pd.DatetimeIndex,
    cost_bps: float,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Fixed-share equal-weight portfolio with a single initial entry."""
    asset_growth = np.exp(close_to_close_log_returns).cumprod(axis=0)
    gross_equity = asset_growth.mean(axis=1)
    prior_equity = np.r_[1.0, gross_equity[:-1]]
    gross_return = gross_equity / prior_equity - 1.0
    daily = pd.DataFrame(index=dates)
    daily["asset"] = "EQUAL_WEIGHT_BUY_HOLD"
    daily["prediction"] = np.nan
    daily["gross_return"] = gross_return
    daily["turnover"] = 0.0
    daily.iloc[0, daily.columns.get_loc("turnover")] = 1.0
    daily["cost"] = daily["turnover"] * (cost_bps / 10_000.0)
    daily["net_return"] = daily["gross_return"] - daily["cost"]
    daily["trade_started"] = False
    daily.iloc[0, daily.columns.get_loc("trade_started")] = True
    daily["equity"] = (1.0 + daily["net_return"]).cumprod()
    metrics = _metrics(daily)
    metrics["hit_rate"] = float((gross_return > 0).mean())
    metrics["trade_days"] = int(len(daily))
    return daily, metrics


def run_persistent_long_cash_backtest(
    predictions: np.ndarray,
    actual_log_returns: np.ndarray,
    dates: pd.DatetimeIndex,
    assets: list[str] | tuple[str, ...],
    threshold: float = 0.0,
    cost_bps: float = 10.0,
    always_invested: bool = False,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Hold the chosen asset from next open to the following open."""
    if predictions.shape != actual_log_returns.shape:
        raise ValueError("Prediction and actual arrays must have the same shape")
    if len(predictions) != len(dates):
        raise ValueError("Dates must match predictions")
    assets = list(assets)
    previous_weights = np.zeros(len(assets), dtype=float)
    previous_selected = -1
    rate = cost_bps / 10_000.0
    records: list[dict[str, object]] = []

    for index, date in enumerate(dates):
        best = int(np.argmax(predictions[index]))
        selected = best if always_invested or predictions[index, best] > threshold else -1
        weights = np.zeros(len(assets), dtype=float)
        if selected >= 0:
            weights[selected] = 1.0
        # Sum risky-asset trades: cash->asset=1, switch=2, asset->cash=1.
        turnover = float(np.abs(weights - previous_weights).sum())
        cost = turnover * rate
        gross = float(np.expm1(actual_log_returns[index, selected])) if selected >= 0 else 0.0
        records.append({
            "date": date,
            "asset": assets[selected] if selected >= 0 else "CASH",
            "prediction": float(predictions[index, best]),
            "gross_return": gross,
            "turnover": turnover,
            "cost": cost,
            "net_return": gross - cost,
            "trade_started": selected >= 0 and selected != previous_selected,
        })
        previous_weights = weights
        previous_selected = selected

    # Liquidate the final position at the final target's exit open.
    if records and previous_weights.sum() > 0:
        final_turnover = float(previous_weights.sum())
        records[-1]["turnover"] = float(records[-1]["turnover"]) + final_turnover
        records[-1]["cost"] = float(records[-1]["cost"]) + final_turnover * rate
        records[-1]["net_return"] = float(records[-1]["net_return"]) - final_turnover * rate
    daily = pd.DataFrame.from_records(records).set_index("date")
    daily["equity"] = (1.0 + daily["net_return"]).cumprod()
    return daily, _metrics(daily)
