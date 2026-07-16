from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tlm.core import canonical_sha256
from tlm.low_turnover_rank_evaluation import (
    CONTROL_NAMES,
    POSITION_SIGNAL_DATES,
    SIGNAL_DATES,
    _FeatureStore,
    _policy_frame,
    _registered_packet_contract,
    _turnover_audit,
)
from tlm.low_turnover_rank_training_data import BASE_FEATURES


ROOT = Path(__file__).resolve().parents[1]


def _store() -> _FeatureStore:
    rows = []
    dates = pd.date_range("2025-08-26", "2026-06-08", freq="D", tz="UTC")
    for symbol_index, symbol in enumerate(("AUSDT", "BUSDT", "CUSDT")):
        for date in dates:
            row = {
                "date": date,
                "symbol": symbol,
                "sequence_ready": True,
                "log_close_to_close_return": 0.001 + symbol_index * 0.0001,
            }
            for feature in BASE_FEATURES:
                row.setdefault(feature, 0.01 + symbol_index * 0.001)
            rows.append(row)
    return _FeatureStore(pd.DataFrame(rows))


def _scores() -> dict[pd.Timestamp, np.ndarray]:
    return {
        pd.Timestamp(value, tz="UTC"): np.asarray([0.4, 0.0, -0.4])
        for value in SIGNAL_DATES
    }


def test_candidate_policy_has_eight_decisions_and_structural_turnover_at_most_16() -> None:
    frame = _policy_frame(
        fold=1,
        triplet_id="F1-T000",
        triplet=("AUSDT", "BUSDT", "CUSDT"),
        store=_store(),
        scores=_scores(),
        control=None,
    )
    daily = frame.drop_duplicates("signal_date")
    assert len(daily) == len(POSITION_SIGNAL_DATES) == 179
    assert int(daily["decision"].sum()) == 8
    audit = _turnover_audit(frame, ["fold", "triplet_id"])
    assert audit["maximum_daily_turnover_error"] == 0.0
    assert audit["maximum_final_liquidation_error"] == 0.0
    assert audit["maximum_triplet_turnover"] <= 16.0
    assert daily.iloc[-1]["final_liquidation_turnover"] == 1.0


def test_registered_controls_are_cash_or_bounded_long_only() -> None:
    frames = []
    for control in CONTROL_NAMES:
        frames.append(
            _policy_frame(
                fold=1,
                triplet_id="F1-T000",
                triplet=("AUSDT", "BUSDT", "CUSDT"),
                store=_store(),
                scores=_scores(),
                control=control,
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    daily = frame.groupby(["control", "signal_date"])["weight"].sum()
    assert (daily >= 0.0).all()
    assert (daily <= 1.0 + 1.0e-12).all()
    cash = frame.loc[frame["control"] == "cash", "weight"]
    assert float(cash.sum()) == 0.0


def test_registered_one_shot_hash_binds_costs_controls_accounting_and_gates() -> None:
    contract = yaml.safe_load(
        (ROOT / "research/phase_contracts/v084.yaml").read_text(encoding="utf-8")
    )
    registered = _registered_packet_contract(contract)
    body = {key: value for key, value in registered.items() if key != "sha256"}
    assert registered["cost_bps"] == [10, 20, 30]
    assert len(registered["outcome_blind_gate_names"]) == 12
    assert registered["sha256"] == canonical_sha256(body)
