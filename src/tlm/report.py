from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/tlm-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METRIC_COLUMNS = [
    "total_return", "cagr", "sharpe", "max_drawdown", "turnover",
    "cost_paid", "hit_rate", "position_changes", "pnl_per_position_change",
    "trade_count", "pnl_per_trade",
]


def write_artifacts(
    output_dir: str | Path,
    predictions: pd.DataFrame,
    metrics: dict[str, dict],
    equity_curves: dict[str, pd.DataFrame],
    context: dict,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output / "predictions.parquet", index=False)
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    figure, axis = plt.subplots(figsize=(10, 5))
    for name, curve in equity_curves.items():
        axis.plot(curve.index, curve["equity"], label=name)
    axis.set_title("TLM walk-forward equity curves")
    axis.set_ylabel("Equity (initial = 1.0)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "equity_curve.png", dpi=150)
    plt.close(figure)

    table = pd.DataFrame(metrics).T[[column for column in METRIC_COLUMNS if column in next(iter(metrics.values()))]]
    candidate_key = context.get("candidate_key", "transformer")
    candidate = metrics.get(candidate_key)
    baseline = metrics.get("equal_weight_buy_hold")
    additional_baselines = [
        values for name, values in metrics.items() if name.startswith("dual_momentum_")
    ]
    if candidate and baseline and context.get("acceptance_mode") == "risk_off":
        if context.get("robustness_verified", False):
            verdict = "OFFLINE RISK GATES PASSED - the candidate improved the registered risk metrics, but still requires a clean external holdout."
            next_action = "Freeze the implementation and validate it on a new exchange dataset or a future untouched period before any promotion."
        else:
            verdict = "REJECT CURRENT RISK-OFF CANDIDATE - it failed at least one registered return-retention, Sharpe, drawdown, fold, cost, or Monte Carlo gate."
            next_action = "Keep dual momentum as the deterministic control and do not tune this candidate against the observed outer windows."
    elif candidate and baseline:
        comparisons = [baseline, *additional_baselines]
        passes = all(
            candidate["total_return"] > item["total_return"]
            and candidate["sharpe"] > item["sharpe"]
            and candidate["max_drawdown"] > item["max_drawdown"]
            for item in comparisons
        )
        if passes and context.get("robustness_verified", False):
            verdict = "ACCEPT FOR PAPER TRADING - the candidate clears performance, cost-sensitivity, and fold-stability gates."
            next_action = "Run it in shadow mode; roll back to dual momentum if rolling 90-day return lags or drawdown breaches the research bound."
        elif passes:
            verdict = "TEST MORE - the candidate clears the first performance gate, but robustness is not yet verified."
            next_action = "Test multiple seeds, higher costs, and fold stability before paper trading."
        else:
            verdict = "REJECT CURRENT CANDIDATE - it does not beat all required baselines on return, Sharpe, and drawdown together."
            next_action = "Keep the model offline; add an explicit volatility/risk policy and rerun the same frozen walk-forward evaluation."
    else:
        verdict = "BASELINE ONLY - no Transformer comparison was executed."
        next_action = "Run the complete Ridge plus Transformer experiment."
    target_description = {
        "next_open_to_close": "next-day open-to-close (intraday round trip)",
        "next_open_to_open": "next-open-to-following-open (persistent daily position)",
    }.get(context["target_mode"], context["target_mode"])
    lines = [
        "# TLM MVP experiment report",
        "",
        f"- Data source: `{context['data_source']}`",
        f"- Assets: {', '.join(context['assets'])}",
        f"- Date range: {context['start']} to {context['end']}",
        f"- Valid sequences: {context['sequences']}",
        f"- Walk-forward folds: {context['folds']}",
        f"- Walk-forward mode: `{context.get('walk_forward_mode', 'expanding')}`",
        f"- Trading cost: {context['cost_bps']:.2f} bps per one-way unit of turnover",
        f"- Target/execution: {target_description}",
        f"- Portfolio policy: `{context['policy']}`",
        f"- Model: `{context.get('model_name', 'Transformer')}`",
        f"- Model objective: `{context['model_objective']}`",
        "",
        "## Baseline versus candidate",
        "",
        table.to_markdown(floatfmt=".4f"),
        "",
        "## Decision",
        "",
        f"**{verdict}**",
        "",
        f"Next action: {next_action}",
        "",
        "## Interpretation",
        "",
        "Results are research-only. A candidate is not accepted on hit rate alone; it must improve net return or risk-adjusted return without an unacceptable drawdown increase.",
        "",
        f"Signals use close-of-day features at `t` and target {target_description}. Scalers and models are fitted only on observations preceding each test window.",
        "",
        "## Limitations",
        "",
        "- Spot OHLCV only; spread and slippage are represented by a configurable one-way turnover cost.",
        "- No overnight return, shorting, leverage, funding, liquidity sizing, or live execution.",
        "- This short history is insufficient evidence for capital deployment.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
