from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from tlm.low_turnover_rank_training_data import (
    BASE_FEATURES,
    V83BalancedRotationSampler,
    V83FeatureScaler,
    V83FoldScale,
    V83FoldTrainingData,
    V83TensorStore,
)


def tiny_v83_fold() -> V83FoldTrainingData:
    symbols = ("ADAUSDT", "AVAXUSDT", "BNBUSDT")
    dates = pd.date_range("2020-01-01", periods=129, tz="UTC")
    panel = pd.DataFrame([
        {
            "date": date,
            "symbol": symbol,
            **{
                feature: 0.001 * (date_index + 1) + 0.01 * symbol_index
                + 0.0001 * feature_index
                for feature_index, feature in enumerate(BASE_FEATURES)
            },
        }
        for symbol_index, symbol in enumerate(symbols)
        for date_index, date in enumerate(dates)
    ])
    labels = pd.DataFrame([
        {
            "signal_date": date,
            "symbol": symbol,
            "target_21d_open_to_open_log_return": 0.01 + 0.002 * symbol_index,
            "label_complete": True,
            "sequence_ready": True,
        }
        for date in dates[-2:]
        for symbol_index, symbol in enumerate(symbols)
    ])
    values = panel[list(BASE_FEATURES)].to_numpy(np.float64)
    median = np.median(values, axis=0)
    q75, q25 = np.percentile(values, [75, 25], axis=0)
    scaler = V83FeatureScaler(
        BASE_FEATURES,
        tuple(median),
        tuple(np.maximum(q75 - q25, 1.0e-6)),
        "unit_test_train_only",
        str(dates[0].date()),
        str(dates[-2].date()),
        len(panel),
    )
    triplets = tuple(combinations(symbols, 3))
    train = {dates[-2]: symbols}
    validation = {dates[-1]: symbols}
    return V83FoldTrainingData(
        1, symbols, ("DOTUSDT",), triplets,
        V83TensorStore(panel, labels, 128), train, validation,
        V83FoldScale(1, scaler, 0.01, 3),
        {
            "access_sha256": "a" * 64,
            "target_assets_loaded": [],
            "heldout_symbols_loaded": [],
            "rows_from_2025_or_later": 0,
            "adaptive_evaluation_role_column_loaded": False,
        },
    )


def test_rotation_and_materialization_are_deterministic() -> None:
    data = tiny_v83_fold()
    sampler = data.sampler(seed=42, role="train")
    first = sampler.sample(1, 3)
    assert first == sampler.sample(1, 3)
    features, targets = data.store.materialize(first, data.scale.feature_scaler)
    assert features.shape == (3, 128, 3, 8)
    assert targets.shape == (3, 3)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_validation_rotation_is_seed_independent_and_registered() -> None:
    data = tiny_v83_fold()
    left = data.sampler(seed=7, role="internal_validation").sample(0, 10)
    right = data.sampler(seed=123, role="internal_validation").sample(0, 10)
    assert left == right
    assert all(draw.triplet in data.registered_triplets for draw in left)
