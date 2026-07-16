from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tlm.core import DatasetAccessLedger, file_sha256
from tlm.persistent_duration_dataset import (
    _read_inputs,
    _write_parquet_with_fresh_replay,
    build_persistent_duration_labels,
    build_persistent_sequence_roles,
    run_persistent_duration_dataset,
)


def _contract() -> dict:
    return yaml.safe_load(
        Path("research/phase_contracts/v076.yaml").read_text(encoding="utf-8")
    )


def _panel(opens: list[float], *, start: str = "2023-01-01") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range(start, periods=len(opens), freq="D", tz="UTC"),
            "symbol": ["AAVEUSDT"] * len(opens),
            "raw_open": opens,
            "raw_observation_available": [True] * len(opens),
        }
    )


def test_return_endpoints_and_earliest_duration_event_are_exact() -> None:
    contract = _contract()
    panel = _panel(
        [99.0, 100.0, 101.0, 102.0, 110.0, 105.0, 106.0, 107.0, 108.0, 109.0]
    )
    labels, cumulative = build_persistent_duration_labels(
        panel,
        contract["label_contract"],
        contract["output_contract"]["labels_columns"],
    )
    first = labels.iloc[0]
    assert first["target_h1_open_to_open_log_return"] == pytest.approx(
        np.log(101.0 / 100.0)
    )
    assert first["target_h3_open_to_open_log_return"] == pytest.approx(
        np.log(110.0 / 100.0)
    )
    assert first["target_h7_open_to_open_log_return"] == pytest.approx(
        np.log(108.0 / 100.0)
    )
    assert int(first["target_duration_days"]) == 3
    assert not bool(first["duration_right_censored"])
    assert np.argmax(cumulative[0]) + 1 == 3
    assert list(labels.columns) == contract["output_contract"]["labels_columns"]


def test_day7_argmax_is_right_censored_and_ties_choose_earliest() -> None:
    contract = _contract()
    rising = _panel([99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0])
    rising_labels, _ = build_persistent_duration_labels(
        rising,
        contract["label_contract"],
        contract["output_contract"]["labels_columns"],
    )
    assert int(rising_labels.iloc[0]["target_duration_days"]) == 7
    assert bool(rising_labels.iloc[0]["duration_right_censored"])

    tied = _panel([99.0, 100.0, 110.0, 110.0, 105.0, 104.0, 103.0, 102.0, 101.0])
    tied_labels, _ = build_persistent_duration_labels(
        tied,
        contract["label_contract"],
        contract["output_contract"]["labels_columns"],
    )
    assert int(tied_labels.iloc[0]["target_duration_days"]) == 1
    assert not bool(tied_labels.iloc[0]["duration_right_censored"])


def test_missing_duration_endpoint_preserves_row_and_masks_label() -> None:
    contract = _contract()
    panel = _panel([99.0, 100.0, 101.0, 102.0, 103.0, np.nan, 105.0, 106.0, 107.0])
    labels, _ = build_persistent_duration_labels(
        panel,
        contract["label_contract"],
        contract["output_contract"]["labels_columns"],
    )
    first = labels.iloc[0]
    assert not bool(first["duration_label_complete"])
    assert pd.isna(first["target_duration_days"])
    assert pd.isna(first["duration_right_censored"])
    assert not bool(first["persistent_label_complete"])
    assert len(labels) == len(panel)


def test_roles_keep_2025_evaluation_dates_but_no_label_values() -> None:
    contract = _contract()
    dates = pd.to_datetime(
        ["2023-12-23", "2024-12-23", "2025-01-01", "2025-12-23"],
        utc=True,
    )
    completion = [True, True]
    labels = pd.DataFrame(
        {
            "date": dates[:2],
            "symbol": ["AAVEUSDT", "AAVEUSDT"],
            "h1_label_complete": completion,
            "h3_label_complete": completion,
            "h7_label_complete": completion,
            "duration_label_complete": completion,
            "persistent_label_complete": completion,
        }
    )
    sequence = pd.DataFrame(
        {
            "date": dates,
            "sequence_start_date": dates - pd.Timedelta(days=255),
            "symbol": ["AAVEUSDT"] * len(dates),
        }
    )
    roles, audit = build_persistent_sequence_roles(
        sequence,
        labels,
        contract["role_contract"],
        contract["output_contract"],
    )
    evaluation = roles["eligible_adaptive_development_evaluation"]
    assert evaluation.tolist() == [False, False, True, True]
    assert not roles.loc[evaluation, "persistent_label_complete"].any()
    assert audit["eligible_adaptive_development_evaluation"][
        "labels_materialized"
    ] is False


def test_input_reader_enforces_outcome_blind_panel_predicate() -> None:
    contract = deepcopy(_contract())
    contract["input_contract"]["admitted_panel_rows_after_outcome_blind_filter"] = 2
    calls: list[dict] = []

    def reader(path: Path, **kwargs: object) -> pd.DataFrame:
        calls.append({"path": str(path), **kwargs})
        if "panel" in str(path):
            return pd.DataFrame(
                {
                    "date": pd.to_datetime(["2024-12-30", "2024-12-31"], utc=True),
                    "symbol": ["AAVEUSDT", "AAVEUSDT"],
                }
            )
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-01"], utc=True),
                "sequence_start_date": pd.to_datetime(["2024-04-21"], utc=True),
                "symbol": ["AAVEUSDT"],
            }
        )

    context = {
        "contract": contract,
        "input_paths": {
            "panel": Path("panel.parquet"),
            "sequence_index": Path("sequence.parquet"),
        },
    }
    ledger = DatasetAccessLedger()
    panel, sequence, symbols = _read_inputs(context, ledger, reader=reader)
    assert len(panel) == 2
    assert len(sequence) == 1
    assert symbols == ["AAVEUSDT"]
    assert calls[0]["filters"] == [
        ("date", "<=", pd.Timestamp("2024-12-31", tz="UTC"))
    ]
    assert ledger.authorized_parquet_deserializations == 2


def test_fresh_parquet_replay_is_byte_identical(tmp_path: Path) -> None:
    ledger = DatasetAccessLedger()
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, tz="UTC"),
            "symbol": ["A", "B", "C"],
            "duration": pd.array([1, pd.NA, 7], dtype="Int8"),
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


def test_runner_rejects_phase_hash_drift_before_data_access(tmp_path: Path) -> None:
    config = yaml.safe_load(
        Path("configs/v76_persistent_duration_dataset.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["output_dir"] = str(tmp_path / "output")
    config["persistent_duration_dataset"]["phase_contract"]["file_sha256"] = (
        "0" * 64
    )
    with pytest.raises(RuntimeError, match="phase contract"):
        run_persistent_duration_dataset(config)
    assert not (tmp_path / "output").exists()
