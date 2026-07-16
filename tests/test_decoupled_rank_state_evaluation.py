from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path

import pandas as pd
import numpy as np
import pytest
import yaml

from tlm.decoupled_rank_state_evaluation import (
    V64PrepareError,
    _behavior_gates,
    _fold_scope,
    _fold_policy_positions,
    _read_projection,
    _registered_contract,
)
from tlm.decoupled_rank_state_harness import decoupled_rank_state_positions


ROOT = Path(__file__).resolve().parents[1]


def _evaluation() -> dict:
    value = yaml.safe_load(
        (ROOT / "configs/v64_decoupled_rank_state_evaluation.yaml").read_text()
    )
    return value["decoupled_rank_state_evaluation"]


def test_registered_contract_binds_all_frozen_science() -> None:
    registered = _registered_contract(_evaluation())
    assert registered["cost_bps"] == [10, 20, 30]
    assert registered["controls"] == {"cash": "all_zero_weights"}
    assert len(registered["outcome_blind_gate_names"]) == 12
    assert len(registered["sha256"]) == 64


def test_fold_scope_requires_exact_lexical_test_triplets() -> None:
    symbols = [f"A{index:02d}USDT" for index in range(10)]
    train = [f"Z{index:02d}USDT" for index in range(20)]
    context = {
        "metadata": {
            "asset_folds": {
                "folds": [
                    {"fold": fold, "test_symbols": symbols, "train_symbols": train}
                    for fold in (1, 2, 3)
                ]
            },
            "triplet_catalog": {
                "folds": [
                    {
                        "fold": fold,
                        "test_triplets": [list(value) for value in combinations(symbols, 3)],
                    }
                    for fold in (1, 2, 3)
                ]
            },
        }
    }
    observed_symbols, observed_triplets = _fold_scope(context, 1)
    assert observed_symbols == symbols
    assert observed_triplets == list(combinations(symbols, 3))

    context["metadata"]["triplet_catalog"]["folds"][0]["test_triplets"].pop()
    with pytest.raises(V64PrepareError, match="not exact lexical"):
        _fold_scope(context, 1)


def test_projected_reader_rejects_target_symbol(tmp_path: Path) -> None:
    path = tmp_path / "panel.parquet"
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01"], utc=True),
            "symbol": ["BTCUSDT"],
            "feature": [1.0],
        }
    ).to_parquet(path, index=False)
    with pytest.raises(V64PrepareError, match="target symbol"):
        _read_projection(
            path,
            ["date", "symbol", "feature"],
            ["BTCUSDT"],
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-01", tz="UTC"),
        )


def test_readiness_projection_allows_registered_missing_symbol(tmp_path: Path) -> None:
    path = tmp_path / "roles.parquet"
    pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01"], utc=True),
            "symbol": ["AAVEUSDT"],
            "ready": [True],
        }
    ).to_parquet(path, index=False)
    frame = _read_projection(
        path,
        ["date", "symbol", "ready"],
        ["AAVEUSDT", "MATICUSDT"],
        pd.Timestamp("2025-01-01", tz="UTC"),
        pd.Timestamp("2025-01-01", tz="UTC"),
        require_exact_symbols=False,
    )
    assert frame["symbol"].tolist() == ["AAVEUSDT"]


def test_behavior_gates_accept_complete_outcome_blind_cash_packet() -> None:
    evaluation = _evaluation()
    dates = pd.date_range("2025-01-01", periods=357, freq="D", tz="UTC")
    positions = []
    for fold in (1, 2, 3):
        for cost in (10, 20, 30):
            for date in dates:
                for asset in range(10):
                    positions.append(
                        {
                            "date": date,
                            "fold": fold,
                            "cost_bps": cost,
                            "symbol": f"A{asset:02d}USDT",
                            "eligible": True,
                            "candidate_weight": 0.0,
                            "selected_symbol": None,
                            "action": "cash",
                            "transition_turnover": 0.0,
                            "final_liquidation_turnover": 0.0,
                            "total_turnover": 0.0,
                            "gross_exposure": 0.0,
                        }
                    )
    positions = pd.DataFrame(positions)
    contexts = pd.DataFrame(
        {
            "raw_excess": [-0.1, 0.1],
            "market_component": [-0.2, 0.2],
            "absolute_edge": [-0.3, 0.3],
            "log_volatility_z": [0.0, 0.1],
        }
    )
    assets = pd.DataFrame(
        {
            "symbol": ["A00USDT", "A01USDT", "A02USDT"],
            "eligible": [True, True, True],
            "excess_seed_disagreement": [0.0, 0.1, 0.2],
        }
    )
    diagnostics = [
        {
            "signal_dates": 357,
            "test_symbols": [f"A{asset:02d}USDT" for asset in range(10)],
            "eligible_asset_dates": 1,
            "prediction_raw_excess_std": 0.1,
            "prediction_market_std": 0.1,
            "outcome_rows_read": 0,
            "target_assets_loaded": [],
        }
        for _ in (1, 2, 3)
    ]
    checkpoints = [
        {
            "fold": fold,
            "seed": seed,
            "used_for_inference": True,
            "selected_or_discarded": False,
        }
        for fold in (1, 2, 3)
        for seed in (42, 7, 123)
    ]
    result = _behavior_gates(
        {"evaluation": evaluation},
        contexts,
        assets,
        positions,
        diagnostics,
        checkpoints,
    )
    assert result["passed"] is True
    assert set(result["gates"]) == set(evaluation["outcome_blind_gate_names"])


def test_v64_contract_stops_before_explicit_unseal() -> None:
    phase = yaml.safe_load((ROOT / "research/phase_contracts/v064.yaml").read_text())
    lifecycle = phase["evaluation_contract"]["one_shot_lifecycle"]
    assert lifecycle["generic_continue_is_not_unseal_authorization"] is True
    assert lifecycle["explicit_user_authorization_after_prepare_required"] is True
    assert lifecycle["maximum_unseal_count"] == 1
    assert phase["target_contract"]["status"] == "sealed"


def test_v63_checkpoint_manifest_exposes_exact_nine_jobs() -> None:
    manifest = json.loads(
        (
            ROOT
            / "artifacts/v63_decoupled_rank_state_training/checkpoint_manifest.json"
        ).read_text()
    )
    assert "jobs" in manifest
    assert len(manifest["jobs"]) == 9
    assert {(row["fold"], row["seed"]) for row in manifest["jobs"]} == {
        (fold, seed) for fold in (1, 2, 3) for seed in (42, 7, 123)
    }


def test_fold_policy_matches_registered_triplet_policy_and_supports_ten_assets() -> None:
    excess = np.asarray([[0.01, 0.00, -0.01], [0.00, 0.02, -0.02]])
    market = np.asarray([0.01, 0.01])
    momentum = np.ones_like(excess)
    eligible = np.ones_like(excess, dtype=bool)
    expected = decoupled_rank_state_positions(
        excess,
        market,
        momentum,
        eligible,
        base_cost=0.001,
        switch_hurdle=0.002,
        risky_weight=1.0,
    )
    observed = _fold_policy_positions(
        excess,
        market,
        momentum,
        eligible,
        base_cost=0.001,
        switch_hurdle=0.002,
        risky_weight=1.0,
    )
    np.testing.assert_array_equal(observed["positions"], expected["positions"])
    assert observed["actions"] == expected["actions"]
    assert observed["selected_assets"] == expected["selected_assets"]

    wide = _fold_policy_positions(
        np.tile(excess[:, :1], (1, 10)),
        market,
        np.ones((2, 10)),
        np.ones((2, 10), dtype=bool),
        base_cost=0.001,
        switch_hurdle=0.002,
        risky_weight=1.0,
    )
    assert wide["positions"].shape == (2, 10)
    assert (wide["positions"].sum(axis=1) <= 1.0).all()
