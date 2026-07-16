from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tlm.joint_absolute_relative_training import (
    CanonicalTripletSampler,
    V49EarlyStopping,
    fit_cell_return_scale,
    read_cell_data,
    run_joint_absolute_relative_training,
)


def _training_config() -> dict:
    return deepcopy(
        yaml.safe_load(
            Path("configs/v49_joint_absolute_relative_training.yaml").read_text(
                encoding="utf-8"
            )
        )
    )


def test_canonical_sampler_is_exact_and_key_separated() -> None:
    date = pd.Timestamp("2022-06-01", tz="UTC")
    availability = {date: ["D", "B", "A", "C"]}
    common = {
        "master_seed": 20260713,
        "version": "v49",
        "origin": "origin_2024",
        "geometry": "expanding",
        "fold": 1,
        "seed": 42,
    }
    sampler = CanonicalTripletSampler(
        availability, ["A", "B", "C", "D"], role="train", **common
    )
    payload = json.dumps(
        [20260713, "v49", "origin_2024", "expanding", 1, 42, 3, "train"],
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    assert sampler._seed(3) == int.from_bytes(
        hashlib.sha256(payload).digest()[:8], "big"
    )
    first, first_hash = sampler.sample_epoch(3, 32)
    replay, replay_hash = sampler.sample_epoch(3, 32)
    validation = CanonicalTripletSampler(
        availability, ["A", "B", "C", "D"], role="validation", **common
    )
    _, validation_hash = validation.sample_epoch(3, 32)
    assert first == replay
    assert first_hash == replay_hash
    assert first_hash != validation_hash


def test_early_stopping_uses_strict_loss_and_exact_patience() -> None:
    state = V49EarlyStopping(patience=2)
    assert state.update(1, 1.0)
    assert state.best_epoch == 1
    assert not state.update(2, 1.0)
    assert not state.should_stop
    assert state.update(3, 0.9)
    assert state.stale_epochs == 0
    assert not state.update(4, 0.91)
    assert not state.should_stop
    assert not state.update(5, 0.91)
    assert state.should_stop


def test_return_scale_enumerates_raw_triplet_values_without_centering() -> None:
    dates = [
        pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2022-01-02", tz="UTC"),
    ]
    symbols = ["A", "B", "C", "D"]
    rows = []
    for date_index, date in enumerate(dates):
        for symbol_index, symbol in enumerate(symbols):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "return": float(1 + date_index * 4 + symbol_index),
                }
            )
    labels = pd.DataFrame(rows)
    availability = {date: symbols for date in dates}
    result = fit_cell_return_scale(labels, availability, "return", 1e-6)
    enumerated = []
    for date in dates:
        values = labels.loc[labels["date"] == date].set_index("symbol")["return"]
        for triplet in (("A", "B", "C"), ("A", "B", "D"), ("A", "C", "D"), ("B", "C", "D")):
            enumerated.extend(float(values[symbol]) for symbol in triplet)
    assert result["enumerated_triplets"] == 8
    assert result["enumerated_raw_returns"] == 24
    assert result["raw_return_rms_scale"] == pytest.approx(
        float(np.sqrt(np.mean(np.square(enumerated))))
    )


def test_cell_reader_projects_only_train_asset_features_and_return_labels() -> None:
    config = _training_config()["joint_absolute_relative_training"]
    data_access = config["data_access"]
    train_symbols = [f"TRAIN_{index:02d}" for index in range(20)]
    fold = {
        "fold": 1,
        "train_symbols": train_symbols,
        "test_symbols": [f"TEST_{index:02d}" for index in range(10)],
    }
    origin = deepcopy(config["origins"][0])
    origin["geometries"]["expanding"] = {
        "train_start": "2022-12-23",
        "train_end": "2022-12-23",
        "train_maturity_end": "2022-12-31",
    }
    origin["validation_start"] = "2023-12-23"
    origin["validation_end"] = "2023-12-23"
    origin["validation_maturity_end"] = "2023-12-31"
    calls: list[dict[str, object]] = []

    def reader(path: Path, *, engine: str, columns: list[str], filters: list[tuple]):
        calls.append({"path": str(path), "columns": list(columns), "filters": filters})
        dates = [value for column, operator, value in filters if column == "date" and operator == ">="]
        start = pd.Timestamp(dates[0])
        if columns == data_access["sequence_columns"]:
            return pd.DataFrame(
                [
                    {
                        "date": start,
                        "sequence_start_date": start - pd.Timedelta(days=255),
                        "symbol": symbol,
                    }
                    for symbol in train_symbols
                ],
                columns=columns,
            )
        if columns == data_access["label_columns"]:
            return pd.DataFrame(
                [
                    {
                        "date": start,
                        "symbol": symbol,
                        "target_window_end_date": start + pd.Timedelta(days=8),
                        data_access["return_column"]: 0.001,
                    }
                    for symbol in train_symbols
                ],
                columns=columns,
            )
        rows = []
        for symbol in train_symbols:
            row = {column: 0.0 for column in columns}
            row.update({"date": start, "symbol": symbol})
            rows.append(row)
        return pd.DataFrame(rows, columns=columns)

    result = read_cell_data(
        Path("panel.parquet"),
        Path("sequence.parquet"),
        fold,
        origin,
        "expanding",
        data_access,
        reader=reader,
    )
    assert result.audit["heldout_symbols_materialized"] == []
    assert result.audit["target_symbols_materialized"] == []
    assert result.audit["label_columns_materialized"] == [
        data_access["return_column"]
    ]
    assert len(calls) == 5
    assert all(call["columns"] != ["target_realized_volatility_7d"] for call in calls)
    assert all(
        next(value for column, operator, value in call["filters"] if column == "symbol")
        == train_symbols
        for call in calls
    )


def test_preflight_reads_metadata_only_and_rejects_input_drift(tmp_path: Path) -> None:
    config = _training_config()
    training = config["joint_absolute_relative_training"]
    training["require_clean_git_receipt"] = False
    training["source_files"] = []
    training["preflight_output_dir"] = str(tmp_path / "preflight")
    result = run_joint_absolute_relative_training(config, "preflight")
    assert result["decision"] == "authorize_v49_one_job_mps_smoke_only"
    assert result["audit"]["passed"]
    assert result["summary"]["checkpoint_count"] == 0
    assert result["summary"]["total_optimizer_steps"] == 0
    assert result["summary"]["parquet_files_deserialized"] == 0
    assert result["summary"]["total_parameters"] == 1_212_930

    drifted = deepcopy(config)
    drifted["joint_absolute_relative_training"]["expected_input_sha256"][
        "panel"
    ] = "0" * 64
    with pytest.raises(RuntimeError, match="input hash drift: panel"):
        run_joint_absolute_relative_training(drifted, "preflight")
