from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tlm.core import DatasetAccessLedger, file_sha256
from tlm.state_conditioned_multi_horizon_dataset import (
    _write_parquet_with_fresh_replay,
    build_multi_horizon_labels,
    build_sequence_role_index,
)


def _phase_contract() -> dict:
    return yaml.safe_load(
        Path("research/phase_contracts/v057.yaml").read_text(encoding="utf-8")
    )


def _panel(symbols: tuple[str, ...] = ("AAVEUSDT",)) -> pd.DataFrame:
    dates = pd.date_range("2022-12-15", periods=20, freq="D", tz="UTC")
    rows = []
    for symbol_index, symbol in enumerate(symbols):
        for index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "raw_open": 100.0 + symbol_index + index,
                    "raw_observation_available": True,
                }
            )
    return pd.DataFrame(rows)


def test_multi_horizon_labels_use_exact_endpoints_and_preserve_keys() -> None:
    contract = _phase_contract()
    panel = _panel(("AAVEUSDT", "ADAUSDT"))
    labels = build_multi_horizon_labels(panel, contract["label_contract"])
    first = labels.loc[
        (labels["symbol"] == "AAVEUSDT")
        & (labels["date"] == pd.Timestamp("2022-12-15", tz="UTC"))
    ].iloc[0]
    assert np.isclose(
        first["target_h1_open_to_open_log_return"], np.log(102.0 / 101.0)
    )
    assert np.isclose(
        first["target_h3_open_to_open_log_return"], np.log(104.0 / 101.0)
    )
    assert np.isclose(
        first["target_h7_open_to_open_log_return"], np.log(108.0 / 101.0)
    )
    assert first["target_h7_maturity_date"] == pd.Timestamp(
        "2022-12-23", tz="UTC"
    )
    assert len(labels) == len(panel)
    assert not labels.duplicated(["symbol", "date"]).any()
    assert list(labels.columns) == contract["output_contract"]["labels_columns"]


def test_missing_endpoint_retains_row_and_disables_joint_eligibility() -> None:
    contract = _phase_contract()
    panel = _panel()
    panel.loc[panel["date"] == pd.Timestamp("2022-12-23", tz="UTC"), "raw_open"] = np.nan
    labels = build_multi_horizon_labels(panel, contract["label_contract"])
    first = labels.iloc[0]
    assert np.isnan(first["target_h7_open_to_open_log_return"])
    assert not bool(first["h7_label_complete"])
    assert not bool(first["multi_horizon_label_complete"])
    assert len(labels) == len(panel)


def test_role_index_purges_once_at_registered_maturity_boundary() -> None:
    contract = _phase_contract()
    panel = _panel()
    labels = build_multi_horizon_labels(panel, contract["label_contract"])
    sequence = pd.DataFrame(
        {
            "date": pd.date_range("2022-12-22", periods=3, freq="D", tz="UTC"),
            "sequence_start_date": pd.date_range(
                "2022-04-11", periods=3, freq="D", tz="UTC"
            ),
            "symbol": ["AAVEUSDT"] * 3,
        }
    )
    roles, audit = build_sequence_role_index(
        sequence, labels, contract["role_contract"]
    )
    flag = "eligible_origin_2024_expanding_train"
    assert roles.loc[roles["date"] == pd.Timestamp("2022-12-23", tz="UTC"), flag].item()
    assert not roles.loc[
        roles["date"] == pd.Timestamp("2022-12-24", tz="UTC"), flag
    ].item()
    assert audit[flag]["last_eligible_date"] == "2022-12-23"
    assert audit[flag]["maximum_target_maturity"] == "2022-12-31"
    assert list(roles.columns) == contract["output_contract"][
        "sequence_base_columns"
    ] + contract["role_contract"]["physical_flags"]


def test_fresh_parquet_replay_is_byte_identical(tmp_path: Path) -> None:
    ledger = DatasetAccessLedger()
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, tz="UTC"),
            "symbol": ["A", "B", "C"],
            "value": [1.0, np.nan, 3.0],
        }
    )
    path = tmp_path / "replay.parquet"
    receipt = _write_parquet_with_fresh_replay(
        frame, path, engine="pyarrow", compression="zstd", ledger=ledger
    )
    assert receipt["byte_identical"]
    assert receipt["sha256"] == file_sha256(path)
    assert receipt["sha256"] == receipt["fresh_replay_sha256"]
    assert ledger.parquet_writes == 2
    assert not list(tmp_path.glob("*.tmp"))
