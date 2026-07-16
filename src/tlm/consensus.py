from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .audit import audit_artifacts
from .backtest import run_equal_weight_buy_hold, run_persistent_long_cash_backtest
from .data import load_market_data
from .pipeline import run_experiment
from .report import write_artifacts


def _policy_scores(choice: np.ndarray, active: np.ndarray, n_assets: int) -> np.ndarray:
    scores = np.full((len(choice), n_assets), -1.0, dtype=np.float32)
    scores[np.arange(len(choice)), choice] = 1.0
    scores[~active] = -1.0
    return scores


def _passes(candidate: dict, baselines: list[dict]) -> bool:
    return all(
        candidate["total_return"] > baseline["total_return"]
        and candidate["sharpe"] > baseline["sharpe"]
        and candidate["max_drawdown"] > baseline["max_drawdown"]
        for baseline in baselines
    )


def run_consensus_experiment(config: dict, retrain_members: bool = False) -> dict[str, dict]:
    """Train seed members and combine them with a causal dual-momentum prior."""
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    seeds = [int(seed) for seed in config["consensus"]["member_seeds"]]
    assets = list(config["data"]["assets"])
    pred_columns = [f"pred_{asset}" for asset in assets]
    actual_columns = [f"actual_{asset}" for asset in assets]
    member_frames: list[pd.DataFrame] = []

    for seed in seeds:
        member_output = output / "members" / f"seed_{seed}"
        prediction_path = member_output / "predictions.parquet"
        if retrain_members or not prediction_path.exists():
            member_config = deepcopy(config)
            member_config["seed"] = seed
            member_config["output_dir"] = str(member_output)
            member_config["strategy"]["policy"] = "always_long_top1"
            run_experiment(member_config, models=("transformer",))
        frame = pd.read_parquet(prediction_path)
        frame = frame[frame["model"] == "transformer"].sort_values("date").reset_index(drop=True)
        frame["seed"] = seed
        member_frames.append(frame)

    reference = member_frames[0]
    for frame in member_frames[1:]:
        if not reference["date"].equals(frame["date"]):
            raise ValueError("Consensus members do not share identical dates")
        if not np.allclose(reference[actual_columns], frame[actual_columns]):
            raise ValueError("Consensus members do not share identical labels")

    member_predictions = np.stack([frame[pred_columns].to_numpy() for frame in member_frames])
    member_choices = member_predictions.argmax(axis=2)
    unanimous = np.all(member_choices == member_choices[[0]], axis=0)
    unanimous_choice = member_choices[0]
    dates = pd.DatetimeIndex(reference["date"])
    actual = reference[actual_columns].to_numpy()

    frames = load_market_data(config)
    close = pd.DataFrame({asset: frame["close"] for asset, frame in frames.items()})[assets]
    lookback = int(config["consensus"]["momentum_lookback"])
    momentum = np.log(close / close.shift(lookback)).reindex(dates).to_numpy()
    if not np.isfinite(momentum).all():
        raise ValueError("Momentum prior is not finite on consensus dates")
    momentum_choice = momentum.argmax(axis=1)
    active = momentum.max(axis=1) > 0.0
    consensus_choice = momentum_choice.copy()
    consensus_choice[unanimous] = unanimous_choice[unanimous]
    consensus_scores = _policy_scores(consensus_choice, active, len(assets))
    momentum_scores = _policy_scores(momentum_choice, active, len(assets))

    predictions = pd.DataFrame({
        "date": dates,
        "fold": reference["fold"].to_numpy(),
        "model": "transformer",
    })
    for index, asset in enumerate(assets):
        predictions[f"pred_{asset}"] = consensus_scores[:, index]
        predictions[f"actual_{asset}"] = actual[:, index]

    cost_bps = float(config["strategy"]["cost_bps"])
    candidate_curve, candidate_metrics = run_persistent_long_cash_backtest(
        consensus_scores, actual, dates, assets, threshold=0.0, cost_bps=cost_bps
    )
    momentum_curve, momentum_metrics = run_persistent_long_cash_backtest(
        momentum_scores, actual, dates, assets, threshold=0.0, cost_bps=cost_bps
    )
    buy_hold_curve, buy_hold_metrics = run_equal_weight_buy_hold(actual, dates, cost_bps)
    metrics = {
        "transformer": candidate_metrics,
        f"dual_momentum_{lookback}": momentum_metrics,
        "equal_weight_buy_hold": buy_hold_metrics,
    }
    curves = {
        "tlm_consensus": candidate_curve,
        f"dual_momentum_{lookback}": momentum_curve,
        "equal_weight_buy_hold": buy_hold_curve,
    }
    daily_returns = pd.DataFrame({
        "date": dates,
        "fold": reference["fold"].to_numpy(),
    })
    for strategy, curve in curves.items():
        daily_returns[f"{strategy}__net_return"] = curve["net_return"].to_numpy()
        daily_returns[f"{strategy}__gross_return"] = curve["gross_return"].to_numpy()
        daily_returns[f"{strategy}__turnover"] = curve["turnover"].to_numpy()
        daily_returns[f"{strategy}__equity"] = curve["equity"].to_numpy()
    daily_returns.to_parquet(output / "daily_returns.parquet", index=False)

    cost_results: dict[str, dict] = {}
    cost_levels = [float(value) for value in config["consensus"]["cost_sensitivity_bps"]]
    for level in cost_levels:
        _, candidate = run_persistent_long_cash_backtest(
            consensus_scores, actual, dates, assets, threshold=0.0, cost_bps=level
        )
        _, momentum_baseline = run_persistent_long_cash_backtest(
            momentum_scores, actual, dates, assets, threshold=0.0, cost_bps=level
        )
        _, buy_hold = run_equal_weight_buy_hold(actual, dates, level)
        cost_results[str(level)] = {
            "candidate": candidate,
            "dual_momentum": momentum_baseline,
            "buy_hold": buy_hold,
            "passes": _passes(candidate, [momentum_baseline, buy_hold]),
        }

    fold_results: dict[str, dict] = {}
    fold_passes = True
    tolerance = float(config["consensus"]["fold_return_tolerance"])
    folds = reference["fold"].to_numpy()
    for fold in sorted(reference["fold"].unique()):
        indexes = np.flatnonzero(folds == fold)
        _, candidate = run_persistent_long_cash_backtest(
            consensus_scores[indexes], actual[indexes], dates[indexes], assets,
            threshold=0.0, cost_bps=cost_bps,
        )
        _, momentum_baseline = run_persistent_long_cash_backtest(
            momentum_scores[indexes], actual[indexes], dates[indexes], assets,
            threshold=0.0, cost_bps=cost_bps,
        )
        return_delta = candidate["total_return"] - momentum_baseline["total_return"]
        fold_ok = return_delta >= -tolerance
        fold_passes = fold_passes and fold_ok
        fold_results[str(int(fold))] = {
            "start_date": str(dates[indexes].min().date()),
            "end_date": str(dates[indexes].max().date()),
            "observations": int(len(indexes)),
            "candidate": candidate,
            "dual_momentum": momentum_baseline,
            "return_delta": return_delta,
            "within_tolerance": fold_ok,
        }

    primary_pass = _passes(candidate_metrics, [momentum_metrics, buy_hold_metrics])
    robustness = {
        "accepted_for_paper_trading": bool(
            primary_pass
            and all(item["passes"] for item in cost_results.values())
            and fold_passes
        ),
        "primary_gate_passed": primary_pass,
        "all_cost_gates_passed": all(item["passes"] for item in cost_results.values()),
        "all_folds_within_return_tolerance": fold_passes,
        "fold_return_tolerance": tolerance,
        "member_seeds": seeds,
        "unanimous_days": int(unanimous.sum()),
        "unanimous_fraction": float(unanimous.mean()),
        "cost_sensitivity": cost_results,
        "folds": fold_results,
        "walk_forward": deepcopy(config["validation"]),
    }
    with (output / "robustness.json").open("w", encoding="utf-8") as handle:
        json.dump(robustness, handle, indent=2, sort_keys=True)
    robustness_lines = [
        "# TLM consensus robustness",
        "",
        f"- Accepted for paper trading: **{robustness['accepted_for_paper_trading']}**",
        f"- Member seeds: {', '.join(map(str, seeds))}",
        f"- Unanimous override days: {robustness['unanimous_days']} ({robustness['unanimous_fraction']:.1%})",
        f"- Fold return tolerance versus dual momentum: {tolerance:.1%}",
        f"- Walk-forward mode: {config['validation'].get('mode', 'expanding')}",
        "",
        "## Cost sensitivity",
        "",
        "| bps/side | candidate return | Sharpe | max DD | passes |",
        "|---:|---:|---:|---:|:---:|",
    ]
    for level, item in cost_results.items():
        values = item["candidate"]
        robustness_lines.append(
            f"| {float(level):.0f} | {values['total_return']:.2%} | {values['sharpe']:.3f} | {values['max_drawdown']:.2%} | {item['passes']} |"
        )
    robustness_lines.extend([
        "",
        "## Rollback condition",
        "",
        "During shadow/paper operation, fall back to dual momentum if the consensus trails it over a rolling 90-day window or breaches the research max-drawdown bound.",
    ])
    (output / "robustness.md").write_text("\n".join(robustness_lines) + "\n", encoding="utf-8")
    pd.concat(member_frames, ignore_index=True).to_parquet(
        output / "member_predictions.parquet", index=False
    )

    write_artifacts(
        output,
        predictions,
        metrics,
        curves,
        context={
            "data_source": config["data"]["source"],
            "assets": assets,
            "start": str(dates.min().date()),
            "end": str(dates.max().date()),
            "sequences": len(dates),
            "folds": len(np.unique(folds)),
            "cost_bps": cost_bps,
            "target_mode": "next_open_to_open",
            "policy": "dual_momentum_with_unanimous_tlm_override",
            "model_objective": config["transformer"].get("objective", "huber_regression"),
            "robustness_verified": robustness["accepted_for_paper_trading"],
            "walk_forward_mode": config["validation"].get("mode", "expanding"),
        },
    )
    audit_artifacts(output)
    return metrics
