from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tlm.core import file_sha256
from tlm.low_turnover_rank_dataset import (
    LowTurnoverDatasetLedger,
    _write_parquet_with_fresh_replay,
    build_causal_feature_frame,
    build_development_labels,
    build_sealed_daily_outcome_packet,
    run_low_turnover_rank_dataset,
)


def _raw(periods: int = 220, start: str = "2023-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="D", tz="UTC")
    price = np.exp(np.linspace(np.log(100.0), np.log(160.0), periods))
    return pd.DataFrame(
        {
            "open": price,
            "high": price * 1.01,
            "low": price * 0.99,
            "close": price * 1.002,
            "volume": np.arange(periods, dtype=float) + 1000.0,
            "quote_volume": np.arange(periods, dtype=float) + 10000.0,
            "trade_count": np.arange(periods, dtype=float) + 500.0,
        },
        index=index,
    )


def test_features_are_causal_and_sequence_requires_128_complete_days() -> None:
    raw = _raw()
    index = raw.index
    before = build_causal_feature_frame(
        "AAVEUSDT", raw, index, lookback_days=128
    )
    changed = raw.copy()
    changed.iloc[180, changed.columns.get_loc("close")] *= 20.0
    after = build_causal_feature_frame(
        "AAVEUSDT", changed, index, lookback_days=128
    )
    feature_columns = [
        column
        for column in before.columns
        if column
        not in {
            "date",
            "symbol",
            "raw_observation_available",
            "feature_complete",
            "sequence_start_date",
            "sequence_ready",
        }
    ]
    pd.testing.assert_series_equal(
        before.loc[179, feature_columns], after.loc[179, feature_columns]
    )
    assert not before.loc[:156, "sequence_ready"].any()
    assert bool(before.loc[157, "sequence_ready"])
    assert before.loc[157, "sequence_start_date"] == before.loc[30, "date"]


def test_21_interval_target_uses_only_exact_t_plus_1_and_t_plus_22_opens() -> None:
    raw = _raw(periods=70)
    index = raw.index
    features = build_causal_feature_frame(
        "AAVEUSDT", raw, index, lookback_days=2
    )
    role_contract = {
        "train": {"signal_start": "2023-02-05", "signal_end": "2023-02-05"},
        "internal_validation": {
            "signal_start": "2023-02-06",
            "signal_end": "2023-02-06",
        },
    }
    folds = [
        {"fold": fold, "train_symbols": ["AAVEUSDT"], "test_symbols": []}
        for fold in (1, 2, 3)
    ]
    labels = build_development_labels(
        {"AAVEUSDT": raw},
        ["AAVEUSDT"],
        index,
        {"AAVEUSDT": features},
        role_contract,
        {"formula": "log(open[t+22] / open[t+1])", "maturity_days": 22},
        folds,
    )
    first = labels.iloc[0]
    expected = np.log(raw.iloc[57]["open"] / raw.iloc[36]["open"])
    assert first["target_21d_open_to_open_log_return"] == pytest.approx(expected)
    assert first["execution_open_date"] == first["signal_date"] + pd.Timedelta(days=1)
    assert first["exit_open_date"] == first["signal_date"] + pd.Timedelta(days=22)
    assert bool(first["eligible_fold_1"])


def test_missing_exact_target_endpoint_is_preserved_as_ineligible() -> None:
    raw = _raw(periods=70)
    raw.iloc[57, raw.columns.get_loc("open")] = np.nan
    index = raw.index
    features = build_causal_feature_frame(
        "AAVEUSDT", raw, index, lookback_days=2
    )
    roles = {
        "train": {"signal_start": "2023-02-05", "signal_end": "2023-02-05"},
        "internal_validation": {
            "signal_start": "2023-02-06",
            "signal_end": "2023-02-06",
        },
    }
    folds = [
        {"fold": fold, "train_symbols": ["AAVEUSDT"], "test_symbols": []}
        for fold in (1, 2, 3)
    ]
    labels = build_development_labels(
        {"AAVEUSDT": raw},
        ["AAVEUSDT"],
        index,
        {"AAVEUSDT": features},
        roles,
        {"formula": "log(open[t+22] / open[t+1])", "maturity_days": 22},
        folds,
    )
    first = labels.iloc[0]
    assert pd.isna(first["target_21d_open_to_open_log_return"])
    assert not bool(first["label_complete"])
    assert not bool(first["eligible_fold_1"])


def test_sealed_packet_contains_exact_daily_open_intervals_only() -> None:
    raw = _raw(periods=8, start="2026-01-01")
    packet = build_sealed_daily_outcome_packet(
        {"AAVEUSDT": raw},
        ["AAVEUSDT"],
        raw.index,
        {
            "daily_outcome_interval_start": "2026-01-02",
            "daily_outcome_interval_end": "2026-01-04",
        },
    )
    assert list(packet.columns) == [
        "interval_start_date",
        "interval_end_date",
        "symbol",
        "open_to_next_open_log_return",
        "outcome_complete",
    ]
    assert len(packet) == 3
    assert packet.iloc[0]["open_to_next_open_log_return"] == pytest.approx(
        np.log(raw.iloc[2]["open"] / raw.iloc[1]["open"])
    )


def test_all_v82_parquet_writes_have_fresh_byte_identical_replay(
    tmp_path: Path,
) -> None:
    ledger = LowTurnoverDatasetLedger()
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, tz="UTC"),
            "symbol": ["A", "B", "C"],
            "value": [1.0, np.nan, 3.0],
        }
    )
    path = tmp_path / "v82.parquet"
    receipt = _write_parquet_with_fresh_replay(
        frame, path, engine="pyarrow", compression="zstd", ledger=ledger
    )
    assert receipt["byte_identical"]
    assert receipt["sha256"] == receipt["fresh_replay_sha256"]
    assert receipt["sha256"] == file_sha256(path)
    assert ledger.parquet_writes == 2


def test_runner_rejects_phase_hash_drift_before_source_access(tmp_path: Path) -> None:
    config = yaml.safe_load(
        Path("configs/v82_low_turnover_rank_dataset.yaml").read_text(
            encoding="utf-8"
        )
    )
    config = deepcopy(config)
    config["output_dir"] = str(tmp_path / "output")
    config["low_turnover_rank_dataset"]["phase_contract"]["file_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="phase contract"):
        run_low_turnover_rank_dataset(config)
    assert not (tmp_path / "output").exists()
