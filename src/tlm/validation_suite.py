from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .backtest import run_persistent_long_cash_backtest
from .consensus import run_consensus_experiment
from .monte_carlo import paired_block_bootstrap


def _load_scenario(output: Path) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    with (output / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    with (output / "robustness.json").open("r", encoding="utf-8") as handle:
        robustness = json.load(handle)
    daily = pd.read_parquet(output / "daily_returns.parquet")
    predictions = pd.read_parquet(output / "predictions.parquet")
    return metrics, robustness, daily, predictions


def _scenario_monte_carlo(
    daily: pd.DataFrame,
    momentum_lookback: int,
    monte_carlo_config: dict,
    scenario_index: int,
) -> dict[str, object]:
    strategy_columns = {
        "tlm_consensus": daily["tlm_consensus__net_return"].to_numpy(),
        "dual_momentum": daily[
            f"dual_momentum_{momentum_lookback}__net_return"
        ].to_numpy(),
        "buy_hold": daily["equal_weight_buy_hold__net_return"].to_numpy(),
    }
    results: dict[str, object] = {}
    for block_length in monte_carlo_config["block_lengths"]:
        block_seed = (
            int(monte_carlo_config["seed"])
            + scenario_index * 10_000
            + int(block_length)
        )
        results[str(block_length)] = paired_block_bootstrap(
            strategy_columns,
            candidate_name="tlm_consensus",
            baseline_names=["dual_momentum", "buy_hold"],
            block_length=int(block_length),
            n_paths=int(monte_carlo_config["paths"]),
            seed=block_seed,
            batch_size=int(monte_carlo_config.get("batch_size", 250)),
        )
    return results


def _delayed_signal_stress(
    predictions: pd.DataFrame,
    assets: list[str],
    cost_bps: float,
) -> dict[str, float | int]:
    pred_columns = [f"pred_{asset}" for asset in assets]
    actual_columns = [f"actual_{asset}" for asset in assets]
    scores = predictions[pred_columns].to_numpy()
    delayed = np.full_like(scores, -1.0)
    delayed[1:] = scores[:-1]
    _, metrics = run_persistent_long_cash_backtest(
        delayed,
        predictions[actual_columns].to_numpy(),
        pd.DatetimeIndex(predictions["date"]),
        assets,
        threshold=0.0,
        cost_bps=cost_bps,
    )
    return metrics


def _beats_on_required_metrics(candidate: dict, baseline: dict) -> bool:
    return bool(
        candidate["total_return"] > baseline["total_return"]
        and candidate["sharpe"] > baseline["sharpe"]
        and candidate["max_drawdown"] > baseline["max_drawdown"]
    )


def _finite_numbers(value: object) -> bool:
    if isinstance(value, dict):
        return all(_finite_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_numbers(item) for item in value)
    if isinstance(value, (int, float)):
        return bool(np.isfinite(value))
    return True


def run_validation_suite(config: dict, retrain_scenarios: bool = False) -> dict:
    suite_config = config["validation_suite"]
    root = Path(config["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    results: dict[str, dict] = {}
    for scenario_index, scenario in enumerate(suite_config["scenarios"]):
        name = scenario["name"]
        if "artifact_dir" in scenario:
            scenario_output = Path(scenario["artifact_dir"])
            if not (scenario_output / "daily_returns.parquet").exists():
                reference_config = deepcopy(config)
                reference_config["output_dir"] = str(scenario_output)
                reference_config["validation"] = deepcopy(scenario["validation"])
                run_consensus_experiment(reference_config, retrain_members=False)
        else:
            scenario_output = root / "scenarios" / name
            scenario_config = deepcopy(config)
            scenario_config["output_dir"] = str(scenario_output)
            scenario_config["validation"] = deepcopy(scenario["validation"])
            run_consensus_experiment(
                scenario_config,
                retrain_members=retrain_scenarios,
            )

        metrics, robustness, daily, predictions = _load_scenario(scenario_output)
        monte_carlo = _scenario_monte_carlo(
            daily,
            momentum_lookback=int(config["consensus"]["momentum_lookback"]),
            monte_carlo_config=suite_config["monte_carlo"],
            scenario_index=scenario_index,
        )
        delayed = _delayed_signal_stress(
            predictions.sort_values("date").reset_index(drop=True),
            assets=list(config["data"]["assets"]),
            cost_bps=float(config["strategy"]["cost_bps"]),
        )
        results[name] = {
            "validation": scenario["validation"],
            "artifact_dir": str(scenario_output),
            "metrics": metrics,
            "robustness": robustness,
            "monte_carlo": monte_carlo,
            "one_day_signal_delay": delayed,
        }

    all_scenarios_accepted = all(
        result["robustness"]["accepted_for_paper_trading"]
        for result in results.values()
    )
    dual_momentum_robust = all(
        _beats_on_required_metrics(
            next(
                values
                for key, values in result["metrics"].items()
                if key.startswith("dual_momentum_")
            ),
            result["metrics"]["equal_weight_buy_hold"],
        )
        for result in results.values()
    )
    expected_blocks = {
        str(int(value)) for value in suite_config["monte_carlo"]["block_lengths"]
    }
    suite_audit = {
        "scenario_count_matches_config": len(results) == len(suite_config["scenarios"]),
        "all_scenario_audits_pass": all(
            json.loads(
                (Path(result["artifact_dir"]) / "audit.json").read_text(encoding="utf-8")
            )["passed"]
            for result in results.values()
        ),
        "all_monte_carlo_blocks_present": all(
            set(result["monte_carlo"]) == expected_blocks
            for result in results.values()
        ),
        "all_monte_carlo_path_counts_match": all(
            item["paths"] == int(suite_config["monte_carlo"]["paths"])
            for result in results.values()
            for item in result["monte_carlo"].values()
        ),
        "all_results_are_finite": _finite_numbers(results),
        "contains_expanding_and_rolling": {
            result["validation"].get("mode", "expanding")
            for result in results.values()
        } == {"expanding", "rolling"},
    }
    suite_audit["passed"] = all(suite_audit.values())
    with (root / "validation_suite_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(suite_audit, handle, indent=2, sort_keys=True)

    suite_result = {
        "method": "pre_registered_multi_geometry_walk_forward_and_block_bootstrap",
        "scenario_count": len(results),
        "all_scenarios_accepted": all_scenarios_accepted,
        "dual_momentum_beats_buy_hold_in_all_scenarios": dual_momentum_robust,
        "tlm_consensus_status": (
            "accepted_for_paper_trading"
            if all_scenarios_accepted
            else "suspended_after_extended_validation"
        ),
        "recommended_shadow_control": (
            f"dual_momentum_{config['consensus']['momentum_lookback']}"
            if dual_momentum_robust
            else "none"
        ),
        "audit": suite_audit,
        "scenarios": results,
    }
    with (root / "validation_suite.json").open("w", encoding="utf-8") as handle:
        json.dump(suite_result, handle, indent=2, sort_keys=True)

    lines = [
        "# TLM extended validation suite",
        "",
        f"- Scenarios: {len(results)}",
        f"- All scenario acceptance gates passed: **{suite_result['all_scenarios_accepted']}**",
        f"- TLM consensus status: **{suite_result['tlm_consensus_status']}**",
        f"- Recommended shadow control: **{suite_result['recommended_shadow_control']}**",
        f"- Monte Carlo paths per block/scenario: {suite_config['monte_carlo']['paths']}",
        "",
        "## Walk-forward scenarios",
        "",
        "| scenario | mode | folds | candidate return | Sharpe | max DD | dual momentum return | accepted |",
        "|:--|:--|--:|--:|--:|--:|--:|:--:|",
    ]
    for name, result in results.items():
        validation = result["validation"]
        candidate = result["metrics"]["transformer"]
        momentum_key = next(
            key for key in result["metrics"] if key.startswith("dual_momentum_")
        )
        momentum = result["metrics"][momentum_key]
        lines.append(
            f"| {name} | {validation.get('mode', 'expanding')} | {validation['folds']} | "
            f"{candidate['total_return']:.2%} | {candidate['sharpe']:.3f} | "
            f"{candidate['max_drawdown']:.2%} | {momentum['total_return']:.2%} | "
            f"{result['robustness']['accepted_for_paper_trading']} |"
        )

    lines.extend([
        "",
        "## Monte Carlo versus dual momentum",
        "",
        "| scenario | block | return p05 | median | p95 | P(loss) | P(higher return) | P(win all 3) |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|",
    ])
    for name, result in results.items():
        for block, monte_carlo in result["monte_carlo"].items():
            distribution = monte_carlo["distributions"]["tlm_consensus"]["total_return"]
            comparison = monte_carlo["comparisons"]["dual_momentum"]
            lines.append(
                f"| {name} | {block} | {distribution['p05']:.2%} | "
                f"{distribution['median']:.2%} | {distribution['p95']:.2%} | "
                f"{monte_carlo['candidate_probability_of_loss']:.2%} | "
                f"{comparison['probability_higher_total_return']:.2%} | "
                f"{comparison['probability_wins_all_three']:.2%} |"
            )

    lines.extend([
        "",
        "## One-day signal delay stress",
        "",
        "| scenario | delayed return | Sharpe | max DD |",
        "|:--|--:|--:|--:|",
    ])
    for name, result in results.items():
        delayed = result["one_day_signal_delay"]
        lines.append(
            f"| {name} | {delayed['total_return']:.2%} | "
            f"{delayed['sharpe']:.3f} | {delayed['max_drawdown']:.2%} |"
        )
    lines.extend([
        "",
        "## Interpretation rule",
        "",
        "No scenario is removed after seeing its result. Paper-trading status must be reconsidered if alternative walk-forward geometries fail materially or if the Monte Carlo advantage is weak across block lengths.",
        "",
        "## Decision",
        "",
        (
            "The TLM consensus remains accepted for paper trading."
            if all_scenarios_accepted
            else "Suspend the TLM consensus override. Continue research with dual momentum as the shadow control; the learned override must demonstrate stability under denser and rolling retraining before reconsideration."
        ),
    ])
    (root / "validation_suite.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return suite_result
