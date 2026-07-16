from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from tlm.persistent_duration_training_data import (
    BASE_FEATURES,
    ROLE_COLUMNS,
    V77ExactTripletSampler,
    V77FeatureScaler,
    V77FoldScale,
    V77FoldTrainingData,
    V77TensorStore,
)


def tiny_v77_fold() -> V77FoldTrainingData:
    symbols = ("ADAUSDT", "AVAXUSDT", "BNBUSDT")
    dates = pd.date_range("2020-01-01", periods=258, tz="UTC")
    panel_rows: list[dict[str, object]] = []
    for symbol_index, symbol in enumerate(symbols):
        for date_index, date in enumerate(dates):
            row: dict[str, object] = {"date": date, "symbol": symbol}
            for feature_index, feature in enumerate(BASE_FEATURES):
                row[feature] = (
                    0.001 * (date_index + 1)
                    + 0.01 * symbol_index
                    + 0.0001 * feature_index
                )
            panel_rows.append(row)
    label_rows: list[dict[str, object]] = []
    for date in dates[-2:]:
        for symbol_index, symbol in enumerate(symbols):
            label_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "target_h1_open_to_open_log_return": 0.01 + symbol_index * 0.001,
                    "target_h3_open_to_open_log_return": 0.02 + symbol_index * 0.001,
                    "target_h7_open_to_open_log_return": 0.03 + symbol_index * 0.001,
                    "target_duration_days": 2 + symbol_index,
                    "duration_right_censored": symbol_index == 2,
                    "persistent_label_complete": True,
                }
            )
    panel = pd.DataFrame(panel_rows)
    labels = pd.DataFrame(label_rows)
    store = V77TensorStore(panel, labels, lookback_days=256)
    values = panel[list(BASE_FEATURES)].to_numpy(dtype=np.float64)
    scaler = V77FeatureScaler(
        feature_names=BASE_FEATURES,
        mean=tuple(float(value) for value in values.mean(axis=0)),
        scale=tuple(float(value) for value in values.std(axis=0)),
        source_relative_feature_index=1,
        fit_scope="eligible_train_unique_symbol_date_cells_only",
        fit_start=str(dates[0].date()),
        fit_end=str(dates[-2].date()),
        fit_rows=len(panel),
    )
    train = {dates[-2]: symbols}
    validation = {dates[-1]: symbols}
    triplets = tuple(combinations(symbols, 3))
    return V77FoldTrainingData(
        fold=1,
        train_symbols=symbols,
        heldout_symbols=("DOTUSDT",),
        registered_triplets=triplets,
        store=store,
        train_availability=train,
        validation_availability=validation,
        scale=V77FoldScale(fold=1, feature_scaler=scaler),
        access_receipt={
            "access_sha256": "a" * 64,
            "target_assets_loaded": [],
            "heldout_symbols_loaded": [],
            "rows_from_2025_or_later": 0,
            "adaptive_evaluation_role_column_loaded": False,
        },
    )


def test_sampler_and_materialization_are_exact_and_deterministic() -> None:
    data = tiny_v77_fold()
    sampler = data.sampler(seed=42, role="train")
    first = sampler.sample(epoch=1, sample_count=3)
    second = sampler.sample(epoch=1, sample_count=3)
    assert first == second
    assert all(draw.triplet == data.train_symbols for draw in first)

    features, returns, durations, censored = data.store.materialize(
        first, data.scale.feature_scaler
    )
    assert features.shape == (3, 256, 3, 9)
    assert returns.shape == (3, 3, 3)
    assert durations.shape == (3, 3)
    assert censored.shape == (3, 3)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
    assert np.allclose(features[..., -1].sum(axis=2), 0.0, atol=1e-6)


def test_role_projection_excludes_adaptive_evaluation_column() -> None:
    assert "eligible_train" in ROLE_COLUMNS
    assert "eligible_internal_validation" in ROLE_COLUMNS
    assert "eligible_adaptive_development_evaluation" not in ROLE_COLUMNS

