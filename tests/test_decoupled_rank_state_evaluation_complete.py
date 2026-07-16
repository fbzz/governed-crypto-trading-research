from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tlm.core.artifacts import canonical_sha256
from tlm.decoupled_rank_state_evaluation_complete import (
    _economic_metrics,
    _gate_matrix,
    _series_metrics,
    _spearman,
)
from tlm.research_workflow import (
    ResearchStateError,
    _validate_v64_unseal_boundary,
)


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    value = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _authorized_state() -> dict:
    state = deepcopy(_load("research/current.yaml"))
    state.update(
        {
            "current_experiment": "research/experiments/v064_prepare.yaml",
            "active_family_status": (
                "adaptive_development_evaluation_prepared_exact_unseal_authorized"
            ),
            "last_completed_phase": "v64_outcome_blind_prepare",
            "last_completed_result": (
                "artifacts/v64_decoupled_rank_state_evaluation/result.json"
            ),
            "authorized_next_action": (
                "execute_v64_exactly_one_registered_non_target_outcome_unseal_and_complete_evaluation"
            ),
            "authorized_phase": "v64",
            "authorized_command": (
                "PYTHONPATH=src python3 -m tlm "
                "decoupled-rank-state-evaluation-unseal "
                "--config configs/v64_decoupled_rank_state_evaluation.yaml"
            ),
            "evidence_tier": (
                "adaptive_development_evaluation_prepared_outcomes_sealed"
            ),
        }
    )
    state["families"][-1]["status"] = state["active_family_status"]
    return state


def test_user_authorization_is_exactly_hash_bound_and_target_sealed() -> None:
    stage = _load("research/phase_contracts/v064_unseal_r1.yaml")
    authorization = stage["explicit_user_authorization"]
    assert canonical_sha256(authorization["payload"]) == authorization[
        "canonical_sha256"
    ]
    assert authorization["payload"]["maximum_unseal_count"] == 1
    assert authorization["payload"]["evaluation_spec_sha256"] == (
        "f6fbf371b5e33efdaaf0b0d1622acefce4938efe1e22d63ffa6e086e0a45d134"
    )
    assert authorization["payload"]["prepare_receipt_sha256"] == (
        "18429f83790cd16b57dbb4208c4b211c3f2600aed1adaadaef722ca4bded4e4e"
    )
    assert authorization["payload"]["one_shot_packet_sha256"] == (
        "a18379581eabf694338837f5f7bd5c00faa73ac5a1be92ca17ee1e23eb94d5f3"
    )
    assert stage["target_contract"]["status"] == "sealed"


def test_unseal_boundary_accepts_registered_contract() -> None:
    _validate_v64_unseal_boundary(
        ROOT,
        _authorized_state(),
        _load("research/experiments/v064_prepare.yaml"),
        _load("research/phase_contracts/v064_unseal_r1.yaml"),
    )


def test_unseal_boundary_rejects_second_source_read() -> None:
    stage = deepcopy(_load("research/phase_contracts/v064_unseal_r1.yaml"))
    stage["outcome_access_contract"]["maximum_source_reads"] = 2
    with pytest.raises(ResearchStateError, match="outcome access"):
        _validate_v64_unseal_boundary(
            ROOT,
            _authorized_state(),
            _load("research/experiments/v064_prepare.yaml"),
            stage,
        )


def test_rank_and_series_metrics_are_deterministic() -> None:
    assert _spearman(np.array([3.0, 1.0, 2.0]), np.array([30.0, 10.0, 20.0])) == pytest.approx(1.0)
    assert np.isnan(_spearman(np.ones(3), np.arange(3.0)))
    metrics = _series_metrics(np.array([0.10, -0.05]))
    assert metrics["total_return"] == pytest.approx(0.045)
    assert metrics["maximum_drawdown"] == pytest.approx(-0.05)


def test_economic_metrics_include_entry_and_final_liquidation() -> None:
    dates = pd.date_range("2025-01-01", periods=2, tz="UTC")
    positions = pd.DataFrame(
        [
            {
                "date": date,
                "fold": 1,
                "cost_bps": 10,
                "symbol": symbol,
                "candidate_weight": float(symbol == "AAVEUSDT"),
                "total_turnover": 1.0,
            }
            for date in dates
            for symbol in ("AAVEUSDT", "ATOMUSDT")
        ]
    )
    outcomes = pd.DataFrame(
        [
            {
                "date": date,
                "fold": 1,
                "symbol": symbol,
                "target_h1_open_to_open_log_return": 0.0,
            }
            for date in dates
            for symbol in ("AAVEUSDT", "ATOMUSDT")
        ]
    )
    metrics, daily, attribution = _economic_metrics(positions, outcomes)
    assert daily["turnover"].tolist() == [1.0, 1.0]
    assert daily["net_return"].tolist() == pytest.approx([-0.001, -0.001])
    assert metrics["folds"]["1"]["10"]["turnover"] == pytest.approx(2.0)
    assert metrics["aggregate"]["10"]["total_return"] == pytest.approx(
        (1.0 - 0.001) ** 2 - 1.0
    )
    assert attribution["episode_count"] == 1


def test_gate_matrix_preserves_failed_cells_without_aggregate_rescue() -> None:
    predictive = {
        "folds": {
            "1": {"spearman": -0.01, "top1_centered_excess": 0.01},
            "2": {"spearman": 0.01, "top1_centered_excess": 0.01},
            "3": {"spearman": 0.01, "top1_centered_excess": 0.01},
        },
        "aggregate": {
            "pairwise_accuracy": 0.51,
            "state_direction_accuracy": 0.51,
            "absolute_direction_accuracy": 0.51,
        },
    }
    economic = {
        "folds": {
            str(fold): {
                str(cost): {
                    "total_return": 0.01,
                    "sharpe": 1.0,
                    "maximum_drawdown": -0.1,
                }
                for cost in (10, 20, 30)
            }
            for fold in (1, 2, 3)
        },
        "aggregate": {
            str(cost): {
                "total_return": 0.01,
                "sharpe": 1.0,
                "maximum_drawdown": -0.1,
            }
            for cost in (10, 20, 30)
        },
    }
    bootstrap = {
        "cells": [
            {"kind": kind, "block_length_days": block, "p05": 0.001}
            for kind in ("top1_centered_excess", "economic_total_return_10bps")
            for block in (7, 21, 63)
        ]
    }
    contract = {
        "aggregate_pairwise_accuracy_strictly_above": 0.5,
        "aggregate_state_direction_accuracy_strictly_above": 0.5,
        "aggregate_absolute_direction_accuracy_strictly_above": 0.5,
        "maximum_absolute_drawdown": 0.35,
    }
    gates = _gate_matrix(
        {"predictive": predictive, "economic": economic}, bootstrap, contract
    )
    assert gates["mandatory_gate_count"] == 36
    assert gates["failed_gate_count"] == 1
    assert gates["all_passed"] is False
