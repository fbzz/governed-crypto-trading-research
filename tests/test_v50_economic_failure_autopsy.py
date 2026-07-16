from __future__ import annotations

import pandas as pd

from tlm.v50_economic_failure_autopsy import (
    _gate_category,
    classify_transition,
    summarize_calibration,
    summarize_consensus,
    summarize_gates,
    summarize_transitions,
)


def test_transition_classifier_covers_registered_states() -> None:
    assert classify_transition("cash", "cash") == "cash_hold"
    assert classify_transition("cash", "A") == "entry"
    assert classify_transition("A", "A") == "hold"
    assert classify_transition("A", "B") == "switch"
    assert classify_transition("A", "cash") == "exit"


def test_gate_categories_preserve_v50_contract() -> None:
    names = {
        "fold_spearman": "origin_2024_expanding_fold1_spearman",
        "fold_pairwise": "origin_2024_expanding_fold1_pairwise",
        "fold_top1_excess": "origin_2024_expanding_fold1_top1_excess",
        "fold_return_10bps": "origin_2024_expanding_fold1_return_10bps",
        "predictive_vs_ridge": "origin_2024_expanding_candidate_above_ridge_mean_spearman",
        "return_vs_ridge": "origin_2024_expanding_return_above_ridge_10bps",
        "return_vs_dual": "origin_2024_expanding_return_above_dual_momentum_30_10bps",
        "return_vs_equal": "origin_2024_expanding_return_above_equal_weight_10bps",
        "sharpe_vs_dual": "origin_2024_expanding_sharpe_above_dual_10bps",
        "absolute_drawdown": "origin_2024_expanding_absolute_drawdown_10bps",
        "drawdown_vs_dual": "origin_2024_expanding_drawdown_vs_dual_10bps",
        "turnover_vs_dual": "origin_2024_expanding_turnover_vs_dual_10bps",
        "bootstrap": "origin_2024_expanding_absolute_bootstrap_p05_block7",
    }
    assert {_gate_category(value) for value in names.values()} == set(names)
    summary = summarize_gates(
        {"cells": [{"gate": value, "passed": True} for value in names.values()]}
    )
    assert set(summary) == set(names)
    assert all(item == {"total": 1, "passed": 1, "failed": 0} for item in summary.values())


def test_transition_and_consensus_summaries_are_deterministic() -> None:
    dates = pd.to_datetime(["2025-01-01", "2025-01-02"], utc=True)
    rows = []
    for triplet, actions in [("A|B|C", ["cash", "A"]), ("A|B|D", ["B", "B"])]:
        for date, action in zip(dates, actions, strict=True):
            previous = "cash" if date == dates[0] else ("cash" if triplet == "A|B|C" else "B")
            rows.append(
                {
                    "origin": "origin_2025",
                    "geometry": "expanding",
                    "fold": 1,
                    "triplet_key": triplet,
                    "date": date,
                    "action": action,
                    "previous_action": previous,
                    "transition_state": classify_transition(previous, action),
                    "transition_turnover": 0.0 if previous == action else 1.0 / 3.0,
                }
            )
    actions = pd.DataFrame(rows)
    transitions = summarize_transitions(actions)
    consensus = summarize_consensus(actions, expected_triplets=2)
    key = "origin_2025|expanding|fold1"
    assert transitions[key]["rows"] == 4
    assert transitions[key]["states"]["entry"]["count"] == 2
    assert consensus[key]["active_fraction"] == 0.75


def test_calibration_uses_only_registered_context_outcomes() -> None:
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01"], utc=True),
            "origin": ["origin_2025"],
            "geometry": ["expanding"],
            "fold": [1],
            "triplet_key": ["A|B|C"],
            "symbol_0": ["A"],
            "symbol_1": ["B"],
            "symbol_2": ["C"],
            "transformer_raw_absolute_0": [0.03],
            "transformer_raw_absolute_1": [-0.02],
            "transformer_raw_absolute_2": [0.01],
            "transformer_raw_excess_0": [0.02],
            "transformer_raw_excess_1": [-0.03],
            "transformer_raw_excess_2": [0.01],
        }
    )
    outcomes = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01"] * 3, utc=True),
            "origin": ["origin_2025"] * 3,
            "fold": [1] * 3,
            "symbol": ["A", "B", "C"],
            "action_log_return": [0.04, -0.01, 0.02],
        }
    )
    result = summarize_calibration(predictions, outcomes)[
        "origin_2025|expanding|fold1"
    ]
    assert result["rows"] == 3
    assert result["sign_accuracy"] == 1.0
    assert result["pearson"] is not None
