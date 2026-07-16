from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tlm.core.artifacts import canonical_sha256
from tlm.low_turnover_rank_evaluation_complete import (
    CONTROL_NAMES,
    _bootstrap,
    _gate_matrix,
    _series_metrics,
)
from tlm.research_workflow import _validate_v85_low_turnover_rank_unseal_boundary


ROOT = Path(__file__).resolve().parents[1]


def _yaml(relative: str) -> dict:
    value = yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_v85_authorization_is_exact_hash_bound_and_target_sealed() -> None:
    authorization = json.loads(
        (ROOT / "research/authorizations/v085_low_turnover_rank_outcome_unseal.json")
        .read_text(encoding="utf-8")
    )
    registered = authorization.pop("authorization_sha256")
    assert canonical_sha256(authorization) == registered
    assert registered == "9400f8987a596237fce6d09ac98019f02d5b1c804ce7706dcdd9e869690009ca"
    assert authorization["maximum_unseal_count"] == 1
    assert authorization["source_outcome_reread_allowed"] is False
    assert authorization["target_assets_status"] == "sealed"


def test_v85_boundary_accepts_the_registered_contract() -> None:
    _validate_v85_low_turnover_rank_unseal_boundary(
        ROOT,
        _yaml("research/current.yaml"),
        _yaml("research/experiments/v085.yaml"),
        _yaml("research/phase_contracts/v085.yaml"),
    )


def test_series_metrics_compound_and_include_starting_equity_in_drawdown() -> None:
    metrics = _series_metrics(np.asarray([0.10, -0.05], dtype=np.float64))
    assert metrics["total_return"] == pytest.approx(0.045)
    assert metrics["maximum_drawdown"] == pytest.approx(-0.05)
    assert metrics["observations"] == 2


def _daily_return_fixture(value_by_name: dict[str, float]) -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=179, freq="D", tz="UTC")
    rows = []
    for portfolio, value in value_by_name.items():
        for fold in (1, 2, 3):
            for date in dates:
                rows.append({
                    "portfolio": portfolio,
                    "date": date,
                    "fold": fold,
                    "cost_bps": 10,
                    "gross_return": value,
                    "net_return": value,
                    "turnover": 0.0,
                    "exposure": float(portfolio != "cash"),
                    "unavailable_active_assets": 0,
                    "triplet_count": 120,
                })
    return pd.DataFrame(rows)


def test_bootstrap_is_synchronized_and_deterministic() -> None:
    daily = _daily_return_fixture({
        "candidate": 0.001,
        "cash": 0.0,
        CONTROL_NAMES[1]: 0.0002,
        CONTROL_NAMES[2]: 0.0003,
    })
    contract = _yaml("configs/v85_low_turnover_rank_evaluation.yaml")["bootstrap"]
    first = _bootstrap(daily, contract)
    second = _bootstrap(daily, contract)
    assert first == second
    assert [cell["seed"] for cell in first["cells"]] == [20260723, 20260737, 20260758]
    assert all(cell["candidate_total_return"]["p10"] > 0.0 for cell in first["cells"])
    assert all(cell["candidate_minus_best_control"]["p10"] > 0.0 for cell in first["cells"])


def _passing_metrics() -> dict:
    candidate_folds = {
        str(fold): {"10": {"total_return": 0.04}}
        for fold in (1, 2, 3)
    }
    aggregate = {
        "10": {"total_return": 0.05, "sharpe": 1.0, "maximum_drawdown": -0.10},
        "20": {"total_return": 0.04},
        "30": {"total_return": 0.03},
    }
    portfolios = {
        "candidate": {"folds": candidate_folds, "aggregate": aggregate},
    }
    for control in CONTROL_NAMES:
        portfolios[control] = {"aggregate": {"10": {"total_return": 0.01}}}
    return {"portfolios": portfolios}


def test_gate_matrix_preserves_all_nineteen_cells_without_rescue() -> None:
    bootstrap = {
        "cells": [
            {
                "block_length_days": block,
                "candidate_total_return": {"p10": 0.01},
                "candidate_minus_best_control": {"p10": 0.005},
            }
            for block in (7, 21, 42)
        ]
    }
    behavior = {
        "candidate_turnover": {"aggregate_turnover": 2.0},
        "candidate_exposure_fraction": 0.25,
    }
    gates = _gate_matrix(_passing_metrics(), bootstrap, behavior)
    assert gates["mandatory_category_count"] == 9
    assert gates["mandatory_gate_count"] == 19
    assert gates["passed_gate_count"] == 19
    assert gates["aggregate_rescue_used"] is False
    assert gates["missing_cell_pass_used"] is False


def test_fold_failure_cannot_be_rescued_by_positive_aggregate() -> None:
    metrics = _passing_metrics()
    metrics["portfolios"]["candidate"]["folds"]["2"]["10"]["total_return"] = -0.01
    bootstrap = {
        "cells": [
            {
                "block_length_days": block,
                "candidate_total_return": {"p10": 0.01},
                "candidate_minus_best_control": {"p10": 0.005},
            }
            for block in (7, 21, 42)
        ]
    }
    gates = _gate_matrix(
        metrics,
        bootstrap,
        {
            "candidate_turnover": {"aggregate_turnover": 2.0},
            "candidate_exposure_fraction": 0.25,
        },
    )
    failed = [row for row in gates["gates"] if not row["passed"]]
    assert len(failed) == 1
    assert failed[0]["scope"] == "fold_2_10bps"
    assert gates["all_passed"] is False
