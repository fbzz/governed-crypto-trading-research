from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from tlm.decoupled_rank_state_training_data import (
    BASE_FEATURES,
    ExactTripletSampler,
    FoldTensorStore,
)
from tlm.scientific_harness import FeatureScaler


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2020-01-01", periods=260, tz="UTC")
    symbols = ("AAAUSDT", "BBBUSDT", "CCCUSDT")
    panel_rows = []
    label_rows = []
    for symbol_index, symbol in enumerate(symbols):
        for date_index, date in enumerate(dates):
            features = {
                name: 0.001 * (date_index + feature_index + symbol_index)
                for feature_index, name in enumerate(BASE_FEATURES)
            }
            panel_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    **features,
                    "target_realized_volatility_7d": 0.02 + symbol_index * 0.001,
                }
            )
            label_rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "target_h1_maturity_date": date + pd.Timedelta(days=2),
                    "target_h1_open_to_open_log_return": 0.01 * (symbol_index - 1),
                    "h1_label_complete": True,
                }
            )
    return pd.DataFrame(panel_rows), pd.DataFrame(label_rows)


def test_exact_sampler_is_deterministic_and_uniform_over_registered_pairs() -> None:
    symbols = ("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT")
    dates = pd.date_range("2024-01-01", periods=2, tz="UTC")
    registered = tuple(combinations(symbols, 3))
    sampler = ExactTripletSampler(
        {date: symbols for date in dates}, registered, seed=42, fold=1, role_code=11
    )
    first = sampler.sample(3, 32)
    second = sampler.sample(3, 32)
    assert first == second
    assert sampler.total_pairs == 8
    assert all(draw.triplet in registered for draw in first)


def test_fold_tensor_store_materializes_exact_ranker_and_label_shapes() -> None:
    panel, labels = _frames()
    store = FoldTensorStore(panel, labels, lookback_days=256)
    scaler = FeatureScaler(
        feature_names=BASE_FEATURES,
        mean=tuple(0.0 for _ in BASE_FEATURES),
        scale=tuple(1.0 for _ in BASE_FEATURES),
        source_relative_feature_index=1,
        fit_scope="eligible_train_unique_symbol_date_cells_only",
        fit_start="2020-01-01",
        fit_end="2020-09-16",
        fit_rows=len(panel),
    )
    sampler = ExactTripletSampler(
        {pd.Timestamp("2020-09-16", tz="UTC"): ("AAAUSDT", "BBBUSDT", "CCCUSDT")},
        (("AAAUSDT", "BBBUSDT", "CCCUSDT"),),
        seed=7,
        fold=1,
        role_code=11,
    )
    features, targets = store.materialize(sampler.sample(1, 2), scaler)
    assert features.shape == (2, 256, 3, 9)
    assert targets.shape == (2, 3, 2)
    assert features.dtype == np.float32
    assert targets.dtype == np.float32
    np.testing.assert_allclose(features[..., -1].mean(axis=2), 0.0, atol=1e-7)
