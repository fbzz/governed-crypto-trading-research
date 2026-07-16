from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tlm.core import canonical_sha256
from tlm.persistent_duration_evaluation import (
    SIGNAL_DATES,
    _candidate_for_triplet,
    _control_for_triplet,
    _independent_turnover_audit,
    _prediction_variant_frame,
    _registered_packet_contract,
    _transition_ledger,
)


ROOT = Path(__file__).resolve().parents[1]


class _MomentumStore:
    def momentum_30(
        self, date: pd.Timestamp, triplet: tuple[str, str, str]
    ) -> np.ndarray:
        del date, triplet
        return np.asarray([0.01, 0.02, -0.01], dtype=np.float64)


def test_transition_ledger_counts_entry_switch_exit_and_liquidation() -> None:
    positions = np.asarray(
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    turnover, liquidation, total = _transition_ledger(positions)
    np.testing.assert_array_equal(turnover, [1.0, 0.0, 2.0])
    assert liquidation == 1.0
    assert total == 4.0


def test_candidate_forces_cash_on_unavailable_dates_with_exact_turnover() -> None:
    triplet = ("AUSDT", "BUSDT", "CUSDT")
    eligible = {pd.Timestamp(SIGNAL_DATES[0]), pd.Timestamp(SIGNAL_DATES[2])}
    edges = {
        pd.Timestamp(SIGNAL_DATES[0]): np.asarray([0.02, 0.01, -0.01]),
        pd.Timestamp(SIGNAL_DATES[2]): np.asarray([0.02, 0.01, -0.01]),
    }
    frame, check = _candidate_for_triplet(1, "F1-T000", triplet, eligible, edges)
    daily = frame.drop_duplicates("signal_date")
    assert check["unavailable_cash_exact"] is True
    assert check["turnover_exact"] is True
    assert daily.iloc[0]["action"] == "enter"
    assert daily.iloc[1]["action"] == "exit"
    assert daily.iloc[2]["action"] == "enter"
    assert daily.iloc[1]["selected_symbol"] == "CASH"


def test_never_eligible_triplet_remains_fixed_capital_cash_subaccount() -> None:
    triplet = ("AUSDT", "BUSDT", "CUSDT")
    frame, check = _candidate_for_triplet(
        1, "F1-T000", triplet, set(), {}
    )
    assert check["eligible_days"] == 0
    assert check["total_turnover"] == 0.0
    assert check["unavailable_cash_exact"] is True
    assert frame["weight"].sum() == 0.0
    assert frame["selected_symbol"].unique().tolist() == ["CASH"]


def test_registered_controls_are_cash_or_bounded_long_only() -> None:
    triplet = ("AUSDT", "BUSDT", "CUSDT")
    eligible = {pd.Timestamp(value) for value in SIGNAL_DATES[:10]}
    cash, cash_check = _control_for_triplet(
        1, "F1-T000", triplet, eligible, _MomentumStore(), "cash"
    )
    weekly, weekly_check = _control_for_triplet(
        1,
        "F1-T000",
        triplet,
        eligible,
        _MomentumStore(),
        "weekly_dual_momentum_30_long_one_or_cash",
    )
    equal, equal_check = _control_for_triplet(
        1,
        "F1-T000",
        triplet,
        eligible,
        _MomentumStore(),
        "daily_equal_weight_eligible_assets",
    )
    assert cash_check["weights_exact"] is True
    assert weekly_check["weights_exact"] is True
    assert equal_check["weights_exact"] is True
    assert cash["weight"].sum() == 0.0
    assert weekly.groupby("signal_date")["weight"].sum().max() <= 1.0
    assert equal.groupby("signal_date")["weight"].sum().max() <= 1.0


def test_prediction_variant_projection_is_exact_and_finite() -> None:
    episodes = (
        ("F1-T000", pd.Timestamp(SIGNAL_DATES[0]), ("A", "B", "C")),
        ("F1-T000", pd.Timestamp(SIGNAL_DATES[1]), ("A", "B", "C")),
    )
    shape = (2, 3, 3)
    location = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) / 1000.0
    scale = np.full(shape, 0.02, dtype=np.float64)
    survival = np.asarray([0.9, 0.7, 0.4], dtype=np.float64)[None, None, :]
    survival = np.broadcast_to(survival, shape).copy()
    holding = np.full((2, 3), 3.0)
    edge = np.full((2, 3), 0.01)
    disagreement = np.full((2, 3), 0.001)
    frame = _prediction_variant_frame(
        1,
        episodes,
        "ensemble",
        location,
        scale,
        survival,
        holding,
        edge,
        disagreement,
    )
    assert len(frame) == 6
    assert frame["seed"].unique().tolist() == ["ensemble"]
    assert np.isfinite(frame.select_dtypes(include=[np.number])).all().all()


def test_registered_one_shot_hash_binds_costs_controls_accounting_and_gates() -> None:
    contract = yaml.safe_load(
        (ROOT / "research/phase_contracts/v078.yaml").read_text(encoding="utf-8")
    )
    registered = _registered_packet_contract(contract)
    body = {key: value for key, value in registered.items() if key != "sha256"}
    assert registered["cost_bps"] == [10, 20, 30]
    assert registered["sha256"] == canonical_sha256(body)
    assert len(registered["outcome_blind_gate_names"]) == 12


def test_independent_turnover_audit_reconciles_frozen_position_rows() -> None:
    rows = []
    positions = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    turnover = [1.0, 2.0]
    for day, date in enumerate(SIGNAL_DATES[:2]):
        for slot, symbol in enumerate(("A", "B", "C")):
            rows.append(
                {
                    "fold": 1,
                    "triplet_id": "F1-T000",
                    "signal_date": date,
                    "symbol": symbol,
                    "eligible": True,
                    "weight": positions[day, slot],
                    "action": "enter" if day == 0 else "switch",
                    "transition_turnover": turnover[day],
                    "final_liquidation_turnover": 1.0 if day == 1 else 0.0,
                }
            )
    audit = _independent_turnover_audit(pd.DataFrame(rows))
    assert audit["aggregate_candidate_turnover"] == 4.0
    assert audit["maximum_total_turnover_error"] == 0.0
    assert audit["maximum_daily_turnover_error"] == 0.0
    assert audit["maximum_final_liquidation_error"] == 0.0
