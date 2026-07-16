from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from tlm.state_conditioned_multi_horizon_training_data import (
    BASE_FEATURE_COLUMNS,
    LABEL_COLUMNS,
    PANEL_COLUMNS,
    JobCell,
    ProjectedDNFRead,
    SampleDraw,
    UniformDateTripletSampler,
    derive_sampling_seed,
    fit_job_train_only_scaler,
    fit_train_only_scaler,
    materialize_triplet_batch,
    read_job_training_data,
    sampling_seed_components,
)


SYMBOLS = ("AUSDT", "BUSDT", "CUSDT")
TRAIN_DATE = pd.Timestamp("2022-12-23", tz="UTC")
VALIDATION_DATE = pd.Timestamp("2023-01-01", tz="UTC")


def _cell() -> JobCell:
    return JobCell(
        origin="origin_2024",
        geometry="expanding",
        fold=1,
        train_symbols=SYMBOLS,
        heldout_symbols=("ZUSDT",),
        train_triplets=(SYMBOLS,),
        train_flag="eligible_origin_2024_expanding_train",
        validation_flag="eligible_origin_2024_expanding_validation",
        train_signal_start=pd.Timestamp("2021-03-01", tz="UTC"),
        train_signal_end=TRAIN_DATE,
        validation_signal_start=VALIDATION_DATE,
        validation_signal_end=pd.Timestamp("2023-12-23", tz="UTC"),
    )


def _synthetic_frames() -> dict[str, pd.DataFrame]:
    cell = _cell()
    sequence_rows = []
    for role, signal_date in (("train", TRAIN_DATE), ("validation", VALIDATION_DATE)):
        for symbol in SYMBOLS:
            sequence_rows.append(
                {
                    "date": signal_date,
                    "sequence_start_date": signal_date - pd.Timedelta(days=255),
                    "symbol": symbol,
                    cell.train_flag: role == "train",
                    cell.validation_flag: role == "validation",
                }
            )
    sequence = pd.DataFrame(sequence_rows)[
        [
            "date",
            "sequence_start_date",
            "symbol",
            cell.train_flag,
            cell.validation_flag,
        ]
    ]
    label_rows = []
    for date in (TRAIN_DATE, VALIDATION_DATE):
        for symbol_index, symbol in enumerate(SYMBOLS):
            label_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "target_h7_maturity_date": date + pd.Timedelta(days=8),
                    "target_h1_open_to_open_log_return": 0.01 + symbol_index,
                    "target_h3_open_to_open_log_return": 0.03 + symbol_index,
                    "target_h7_open_to_open_log_return": 0.07 + symbol_index,
                    "multi_horizon_label_complete": True,
                }
            )
    labels = pd.DataFrame(label_rows)[list(LABEL_COLUMNS)]
    context_dates = sorted(
        set(pd.date_range(TRAIN_DATE - pd.Timedelta(days=255), TRAIN_DATE, freq="D"))
        | set(
            pd.date_range(
                VALIDATION_DATE - pd.Timedelta(days=255), VALIDATION_DATE, freq="D"
            )
        )
    )
    panel_rows = []
    for date_index, date in enumerate(context_dates):
        for symbol_index, symbol in enumerate(SYMBOLS):
            row = {"date": date, "symbol": symbol}
            for feature_index, feature in enumerate(BASE_FEATURE_COLUMNS):
                row[feature] = (
                    date_index * 0.01 + symbol_index * 0.1 + feature_index * 0.001
                )
            panel_rows.append(row)
    panel = pd.DataFrame(panel_rows)[list(PANEL_COLUMNS)]
    return {"sequence": sequence, "labels": labels, "panel": panel}


class _Reader:
    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(self, path: object, **kwargs: object) -> pd.DataFrame:
        name = str(path)
        self.calls.append((name, kwargs))
        return self.frames[name].copy()


def _read_data():
    reader = _Reader(_synthetic_frames())
    data = read_job_training_data(
        _cell(),
        sequence_path="sequence",
        labels_path="labels",
        panel_path="panel",
        reader=reader,
    )
    return data, reader


def test_projected_dnf_reader_materializes_only_exact_job_keys() -> None:
    data, reader = _read_data()
    assert [name for name, _ in reader.calls] == ["sequence", "labels", "panel"]
    sequence_kwargs = reader.calls[0][1]
    assert sequence_kwargs["columns"] == [
        "date",
        "sequence_start_date",
        "symbol",
        _cell().train_flag,
        _cell().validation_flag,
    ]
    assert len(sequence_kwargs["filters"]) == 2
    assert all("development_evaluation" not in column for column in sequence_kwargs["columns"])
    assert len(data.train_signal_keys) == 3
    assert len(data.validation_signal_keys) == 3
    assert len(data.labels_by_key) == 6
    assert len(data.context_keys) == len(data.panel)
    assert data.train_availability[TRAIN_DATE] == SYMBOLS
    assert data.validation_availability[VALIDATION_DATE] == SYMBOLS
    assert len(data.scaler_source_rows) > len(data.train_signal_keys)
    assert data.access_receipt["feature_context_key_count"] == len(data.context_keys)
    assert data.access_receipt["forbidden_column_count_zero"] == 0
    assert data.access_receipt[
        "job_relative_development_evaluation_value_count_zero"
    ] == 0
    assert data.access_receipt["target_asset_load_count_zero"] == 0
    assert all(
        isinstance(request, ProjectedDNFRead)
        for request in (
            data.read_plan.sequence,
            data.read_plan.labels,
            data.read_plan.panel,
        )
    )
    assert data.read_plan.labels.expected_logical_key_count == 6
    assert data.read_plan.panel.expected_logical_key_count == len(data.context_keys)


def test_reader_rejects_extra_materialized_development_column() -> None:
    frames = _synthetic_frames()
    frames["sequence"][
        "eligible_origin_2024_expanding_development_evaluation"
    ] = False
    with pytest.raises(ValueError, match="projection drift"):
        read_job_training_data(
            _cell(),
            sequence_path="sequence",
            labels_path="labels",
            panel_path="panel",
            reader=_Reader(frames),
        )


def test_scaler_is_float64_ddof0_unique_finite_train_range_only() -> None:
    dates = pd.date_range("2022-01-01", periods=4, freq="D", tz="UTC")
    rows = []
    for symbol_index, symbol in enumerate(SYMBOLS):
        for date_index, date in enumerate(dates):
            row = {"date": date, "symbol": symbol}
            for feature_index, feature in enumerate(BASE_FEATURE_COLUMNS):
                row[feature] = float(symbol_index * 10 + date_index + feature_index)
            row[BASE_FEATURE_COLUMNS[-1]] = 5.0
            rows.append(row)
    panel = pd.DataFrame(rows)
    panel.loc[
        (panel["symbol"] == "AUSDT") & (panel["date"] == dates[1]),
        BASE_FEATURE_COLUMNS[0],
    ] = np.nan
    panel.loc[panel["date"] == dates[-1], BASE_FEATURE_COLUMNS] = 1.0e12
    scaler = fit_train_only_scaler(
        panel,
        train_symbols=SYMBOLS,
        train_start=dates[0],
        train_end=dates[2],
        origin="origin_2024",
        geometry="expanding",
        fold=1,
    )
    population = panel.loc[panel["date"] <= dates[2], BASE_FEATURE_COLUMNS].to_numpy(
        dtype=np.float64
    )
    population = population[np.isfinite(population).all(axis=1)]
    expected_mean = population.mean(axis=0, dtype=np.float64)
    expected_scale = population.std(axis=0, ddof=0, dtype=np.float64)
    expected_scale[expected_scale == 0] = 1.0
    np.testing.assert_array_equal(np.asarray(scaler.mean), expected_mean)
    np.testing.assert_array_equal(
        np.asarray(scaler.standard_deviation), expected_scale
    )
    assert scaler.fit_unique_symbol_date_count == len(population)
    assert scaler.fit_symbol_count == 3
    assert scaler.zero_scale_replacements == 1
    assert len(scaler.scaler_sha256) == 64


def test_scaler_rejects_duplicate_weighting() -> None:
    frames = _synthetic_frames()
    panel = pd.concat([frames["panel"], frames["panel"].iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate context weighting"):
        fit_train_only_scaler(
            panel,
            train_symbols=SYMBOLS,
            train_start=pd.Timestamp("2022-01-01", tz="UTC"),
            train_end=TRAIN_DATE,
            origin="origin_2024",
            geometry="expanding",
            fold=1,
        )


def test_explicit_registered_calendar_materializes_float32_features_and_targets() -> None:
    data, _ = _read_data()
    scaler = fit_job_train_only_scaler(data)
    sample = SampleDraw(TRAIN_DATE, SYMBOLS, 0)
    batch = materialize_triplet_batch(data, [sample], scaler, role="train")
    assert batch.features.shape == (1, 256, 3, 9)
    assert batch.targets.shape == (1, 3, 3)
    assert batch.features.dtype == np.float32
    assert batch.targets.dtype == np.float32
    np.testing.assert_allclose(
        batch.targets[0, 0], np.asarray([0.01, 0.03, 0.07], dtype=np.float32)
    )
    close_index = BASE_FEATURE_COLUMNS.index("log_close_to_close_return")
    raw = np.stack(
        [
            data.tensor_store.panel_by_key[(TRAIN_DATE, symbol)]
            for symbol in SYMBOLS
        ]
    )
    relative = raw[:, close_index] - raw[:, close_index].mean()
    np.testing.assert_allclose(
        batch.features[0, -1, :, -1],
        relative / scaler.standard_deviation[close_index],
        rtol=1e-6,
    )
    data.tensor_store.sequence_start_by_key[(TRAIN_DATE, "AUSDT")] += pd.Timedelta(
        days=1
    )
    with pytest.raises(ValueError, match="shared registered sequence_start"):
        materialize_triplet_batch(data, [sample], scaler, role="train")


def test_sampler_uses_exact_sha_arrays_and_validation_is_seed_invariant() -> None:
    date_a = pd.Timestamp("2022-01-01", tz="UTC")
    date_b = pd.Timestamp("2022-01-02", tz="UTC")
    triplets = (
        ("A", "B", "C"),
        ("A", "B", "D"),
        ("A", "C", "D"),
        ("B", "C", "D"),
    )
    sampler = UniformDateTripletSampler(
        {date_a: ("A", "B", "C", "D"), date_b: ("A", "B", "C")}, triplets
    )
    train_components = (
        20260714,
        "v58",
        "origin_2024",
        "expanding",
        1,
        42,
        "train",
        3,
    )
    canonical = json.dumps(
        train_components,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    expected_seed = int.from_bytes(hashlib.sha256(canonical).digest()[:8], "big")
    kwargs = dict(
        origin="origin_2024",
        geometry="expanding",
        fold=1,
        job_seed=42,
        role="train",
        epoch=3,
    )
    assert sampling_seed_components(**kwargs) == train_components
    assert derive_sampling_seed(**kwargs) == expected_seed
    first = sampler.sample(100, **kwargs)
    replay = sampler.sample(100, **kwargs)
    assert first == replay
    assert first.generator_seed == expected_seed
    assert len(first.ordered_draw_list_sha256) == 64
    assert all(0 <= draw.pair_index < 5 for draw in first.draws)
    validation_42 = sampler.sample(
        20,
        origin="origin_2024",
        geometry="expanding",
        fold=1,
        job_seed=42,
        role="validation",
        epoch=0,
    )
    validation_123 = sampler.sample(
        20,
        origin="origin_2024",
        geometry="expanding",
        fold=1,
        job_seed=123,
        role="validation",
        epoch=0,
    )
    assert validation_42.seed_components == (
        20260714,
        "v58",
        "origin_2024",
        "expanding",
        1,
        20260714,
        "validation",
        0,
    )
    assert validation_42.draws == validation_123.draws
    assert (
        validation_42.ordered_draw_list_sha256
        == validation_123.ordered_draw_list_sha256
    )
    with pytest.raises(ValueError, match="epoch zero"):
        sampler.sample(
            1,
            origin="origin_2024",
            geometry="expanding",
            fold=1,
            job_seed=42,
            role="validation",
            epoch=1,
        )
