from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tlm.core.artifacts import canonical_sha256
from tlm.research_workflow import (
    ResearchStateError,
    _validate_v72_posthoc_outcome_unseal_boundary,
)
from tlm.v64_r2_retrospective_evaluation import (
    _cash_daily,
    _economic_metrics,
    _gate_matrix,
    _portfolio_daily,
    _series_metrics,
)


ROOT = Path(__file__).resolve().parents[1]


def _yaml(path: str) -> dict:
    value = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _v72_staging_state() -> dict:
    state = deepcopy(_yaml("research/current.yaml"))
    state.update({
        "authorized_phase": "v72",
        "authorized_next_action": (
            "execute_v72_exactly_one_hash_bound_posthoc_outcome_unseal_and_complete_diagnostic"
        ),
        "authorized_command": (
            "PYTHONPATH=src python3 -m tlm v64-r2-retrospective-diagnostic-unseal "
            "--config configs/v72_v64_r2_retrospective_evaluation.yaml"
        ),
        "active_family_status": (
            "posthoc_consumed_2025_diagnostic_exact_unseal_authorized"
        ),
        "evidence_tier": (
            "posthoc_consumed_2025_diagnostic_only_not_confirmation"
        ),
    })
    state["safety"].update({
        "v72_completed_diagnostic_unseal_count": 0,
        "v72_source_packet_deserialization_count": 0,
        "v72_underlying_source_outcome_read_count": 0,
    })
    return state


def test_exact_user_authorization_is_hash_bound_and_target_sealed() -> None:
    path = ROOT / "research/authorizations/v072_posthoc_outcome_unseal.json"
    authorization = json.loads(path.read_text(encoding="utf-8"))
    registered = authorization.pop("authorization_sha256")
    assert canonical_sha256(authorization) == registered
    assert registered == "817fed1b19ca1ab5b040e964e24676ac6c768ade0038d09c73b67f1f86fc8acb"
    assert authorization["maximum_unseal_count"] == 1
    assert authorization["source_outcome_reread_allowed"] is False
    assert authorization["target_assets_status"] == "sealed"


def test_v72_boundary_accepts_the_registered_contract() -> None:
    _validate_v72_posthoc_outcome_unseal_boundary(
        ROOT,
        _v72_staging_state(),
        _yaml("research/experiments/v072.yaml"),
        _yaml("research/phase_contracts/v072.yaml"),
    )


def test_v72_runtime_bootstrap_matches_the_frozen_contract() -> None:
    config = _yaml("configs/v72_v64_r2_retrospective_evaluation.yaml")
    stage = _yaml("research/phase_contracts/v072.yaml")
    expected = deepcopy(stage["evaluation_contract"]["bootstrap"])
    expected.pop("portfolios")
    assert config["bootstrap"] == expected


def test_v72_boundary_rejects_second_packet_open() -> None:
    stage = deepcopy(_yaml("research/phase_contracts/v072.yaml"))
    stage["outcome_access_contract"]["maximum_sealed_packet_deserializations"] = 2
    with pytest.raises(ResearchStateError, match="outcome or evaluation"):
        _validate_v72_posthoc_outcome_unseal_boundary(
            ROOT,
            _v72_staging_state(),
            _yaml("research/experiments/v072.yaml"),
            stage,
        )


def _positions() -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=2, tz="UTC")
    rows = []
    for date_index, date in enumerate(dates):
        for symbol in ("AAVEUSDT", "ATOMUSDT"):
            rows.append(
                {
                    "date": date,
                    "fold": 1,
                    "cost_bps": 10,
                    "symbol": symbol,
                    "eligible": True,
                    "candidate_weight": float(symbol == "AAVEUSDT"),
                    "selected_symbol": "AAVEUSDT",
                    "action": "entry" if date_index == 0 else "hold",
                    "transition_turnover": 1.0 if date_index == 0 else 0.0,
                    "final_liquidation_turnover": 1.0 if date_index == 1 else 0.0,
                    "total_turnover": 1.0,
                    "gross_exposure": 1.0,
                }
            )
    return pd.DataFrame(rows)


def _outcomes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": date,
                "fold": 1,
                "symbol": symbol,
                "target_h1_open_to_open_log_return": 0.0,
            }
            for date in pd.date_range("2025-01-01", periods=2, tz="UTC")
            for symbol in ("AAVEUSDT", "ATOMUSDT")
        ]
    )


def test_economic_accounting_includes_entry_cost_and_final_liquidation() -> None:
    daily = _portfolio_daily("candidate", _positions(), _outcomes())
    assert daily["turnover"].tolist() == [1.0, 1.0]
    assert daily["net_return"].tolist() == pytest.approx([-0.001, -0.001])
    cash = _cash_daily(daily)
    metrics = _economic_metrics(pd.concat([daily, cash], ignore_index=True))
    candidate = metrics["portfolios"]["candidate"]["aggregate"]["10"]
    assert candidate["total_return"] == pytest.approx((1.0 - 0.001) ** 2 - 1.0)
    assert candidate["turnover"] == pytest.approx(2.0)
    assert metrics["portfolios"]["cash"]["aggregate"]["10"]["total_return"] == 0.0


def test_series_metrics_preserve_compounding_and_drawdown() -> None:
    metrics = _series_metrics(np.array([0.10, -0.05], dtype=np.float64))
    assert metrics["total_return"] == pytest.approx(0.045)
    assert metrics["maximum_drawdown"] == pytest.approx(-0.05)


def _passing_metrics() -> dict:
    folds = {
        str(fold): {
            str(cost): {
                "total_return": 0.05,
                "sharpe": 1.0,
                "maximum_drawdown": -0.10,
            }
            for cost in (10, 20, 30)
        }
        for fold in (1, 2, 3)
    }
    aggregate = {
        str(cost): {
            "total_return": 0.05,
            "sharpe": 1.0,
            "maximum_drawdown": -0.10,
        }
        for cost in (10, 20, 30)
    }
    return {"portfolios": {"candidate": {"folds": folds, "aggregate": aggregate}}}


def _passing_bootstrap() -> dict:
    return {
        "cells": [
            {
                "portfolio": "candidate",
                "block_length_days": block,
                "p05": 0.01,
            }
            for block in (7, 21, 63)
        ]
    }


def test_gate_matrix_has_exactly_24_mandatory_cells() -> None:
    gates = _gate_matrix(_passing_metrics(), _passing_bootstrap(), 0.35)
    assert gates["mandatory_gate_count"] == 24
    assert gates["passed_gate_count"] == 24
    assert gates["all_passed"] is True


def test_gate_matrix_preserves_fold_failure_without_aggregate_rescue() -> None:
    metrics = _passing_metrics()
    metrics["portfolios"]["candidate"]["folds"]["2"]["10"]["total_return"] = -0.01
    gates = _gate_matrix(metrics, _passing_bootstrap(), 0.35)
    failed = [row for row in gates["gates"] if not row["passed"]]
    assert len(failed) == 1
    assert failed[0]["scope"] == "fold_2_10bps"
    assert gates["all_passed"] is False
