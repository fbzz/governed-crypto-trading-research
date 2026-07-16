from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tlm.joint_absolute_relative_evaluation import (
    preflight_joint_absolute_relative_evaluation,
)
from tlm.joint_absolute_relative_evaluation_metrics import (
    STRATEGIES,
    build_exact_triplet_portfolio,
    evaluate_v50_gates,
    shift_triplet_positions_one_day,
)


def _config() -> dict:
    return deepcopy(
        yaml.safe_load(
            Path("configs/v50_joint_absolute_relative_evaluation.yaml").read_text(
                encoding="utf-8"
            )
        )
    )


def _portfolio_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    positions = []
    outcomes = []
    for origin in ("origin_2024", "origin_2025"):
        for fold in (1, 2, 3):
            symbols = tuple(f"F{fold}_{slot}" for slot in range(3))
            for date in dates:
                for slot, symbol in enumerate(symbols):
                    outcomes.append(
                        {
                            "date": date,
                            "origin": origin,
                            "fold": fold,
                            "symbol": symbol,
                            "action_log_return": 0.03 if slot == 0 else 0.0,
                        }
                    )
                for geometry in ("expanding", "rolling"):
                    row = {
                        "date": date,
                        "origin": origin,
                        "geometry": geometry,
                        "fold": fold,
                        "triplet_key": "|".join(symbols),
                        **{f"symbol_{slot}": symbol for slot, symbol in enumerate(symbols)},
                    }
                    for strategy in STRATEGIES:
                        for slot in range(3):
                            row[f"{strategy}_weight_{slot}"] = (
                                1.0 / 3.0
                                if strategy in {"candidate", "ridge", "dual_momentum_30"}
                                and slot == 0
                                else 1.0 / 9.0
                                if strategy == "equal_weight"
                                else 0.0
                            )
                    positions.append(row)
    return pd.DataFrame(positions), pd.DataFrame(outcomes)


def test_exact_triplet_portfolio_charges_entry_and_final_liquidation() -> None:
    positions, outcomes = _portfolio_fixture()
    result = build_exact_triplet_portfolio(positions, outcomes, [10])
    metric = result["aggregate_metrics"]["origin_2024"]["expanding"]["10"][
        "candidate"
    ]
    expected_daily_gross = np.expm1(0.03) / 3.0
    expected = 1.0 + expected_daily_gross - 1.0 / 3.0 * 0.001
    expected *= 1.0 + expected_daily_gross
    expected *= 1.0 + expected_daily_gross - 1.0 / 3.0 * 0.001
    assert metric["total_return"] == pytest.approx(expected - 1.0)
    assert metric["total_turnover"] == pytest.approx(2.0 / 3.0)
    assert metric["total_cost"] == pytest.approx(2.0 / 3.0 * 0.001)
    assert len(result["triplet_count_audit"]) == 12


def test_one_day_delay_is_group_local_and_starts_in_cash() -> None:
    positions, _ = _portfolio_fixture()
    delayed = shift_triplet_positions_one_day(positions)
    columns = [f"candidate_weight_{slot}" for slot in range(3)]
    first = delayed.sort_values("date").groupby(
        ["origin", "geometry", "fold", "triplet_key"], sort=False
    ).head(1)
    assert np.allclose(first[columns].to_numpy(dtype=float), 0.0)
    assert delayed["candidate_weight_0"].sum() < positions[
        "candidate_weight_0"
    ].sum()


def _passing_gate_inputs() -> tuple[dict, dict, dict, dict, dict]:
    predictive = {}
    cell = {}
    aggregate = {}
    bootstrap = {}
    for origin in ("origin_2024", "origin_2025"):
        predictive[origin] = {}
        cell[origin] = {}
        aggregate[origin] = {}
        bootstrap[origin] = {}
        for geometry in ("expanding", "rolling"):
            predictive[origin][geometry] = {
                "models": {
                    "transformer": {
                        "folds": {
                            str(fold): {
                                "mean_spearman": 0.1,
                                "mean_pairwise_accuracy": 0.6,
                                "mean_top1_excess": 0.01,
                            }
                            for fold in (1, 2, 3)
                        },
                        "aggregate": {
                            "mean_spearman": 0.1,
                            "mean_pairwise_accuracy": 0.6,
                            "mean_top1_excess": 0.01,
                        },
                    },
                    "ridge": {
                        "folds": {str(fold): {} for fold in (1, 2, 3)},
                        "aggregate": {
                            "mean_spearman": 0.05,
                            "mean_pairwise_accuracy": 0.55,
                            "mean_top1_excess": 0.005,
                        },
                    },
                }
            }
            cell[origin][geometry] = {
                str(fold): {
                    "10": {"candidate": {"total_return": 0.1}}
                }
                for fold in (1, 2, 3)
            }
            aggregate[origin][geometry] = {}
            for cost in (10, 20, 30):
                aggregate[origin][geometry][str(cost)] = {
                    "candidate": {
                        "total_return": 0.2,
                        "sharpe": 2.0,
                        "max_drawdown": -0.1,
                        "total_turnover": 1.0,
                    },
                    "ridge": {"total_return": 0.1},
                    "dual_momentum_30": {
                        "total_return": 0.1,
                        "sharpe": 1.0,
                        "max_drawdown": -0.12,
                        "total_turnover": 2.0,
                    },
                    "equal_weight": {"total_return": 0.1},
                }
            bootstrap[origin][geometry] = {
                str(block): {
                    "distributions": {
                        "candidate": {"total_return": {"p05": 0.01}}
                    },
                    "comparisons": {
                        control: {"paired_total_return_delta": {"p05": 0.01}}
                        for control in ("ridge", "dual_momentum_30", "equal_weight")
                    },
                }
                for block in (7, 21, 63)
            }
    gates = _config()["joint_absolute_relative_evaluation"]["gates"]
    return predictive, cell, aggregate, bootstrap, gates


def test_v50_gates_preserve_all_cells_and_any_failure_retires() -> None:
    inputs = _passing_gate_inputs()
    passed = evaluate_v50_gates(*inputs)
    assert passed["passed"]
    assert passed["cell_count"] == passed["passed_count"]
    inputs[0]["origin_2024"]["rolling"]["models"]["transformer"]["folds"][
        "2"
    ]["mean_spearman"] = -0.01
    failed = evaluate_v50_gates(*inputs)
    assert not failed["passed"]
    assert failed["failed_count"] == 1


def test_v50_preflight_reopens_all_checkpoints_without_parquet(tmp_path: Path) -> None:
    config = _config()
    evaluation = config["joint_absolute_relative_evaluation"]
    evaluation["require_clean_git_receipt"] = False
    evaluation["source_files"] = []
    evaluation["artifact_contract"]["preflight_output_dir"] = str(
        tmp_path / "preflight"
    )
    result = preflight_joint_absolute_relative_evaluation(config)
    assert result["audit"]["passed"]
    assert result["summary"]["checkpoint_count"] == 36
    assert result["summary"]["parquet_files_deserialized"] == 0
    assert result["summary"]["outcome_rows_materialized"] == 0
    assert result["summary"]["optimizer_steps"] == 0
