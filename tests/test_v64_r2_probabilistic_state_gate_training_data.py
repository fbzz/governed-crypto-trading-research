from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from tlm.decoupled_rank_state_training_data import BASE_FEATURES, ExactTripletSampler
from tlm.scientific_harness import FeatureScaler
from tlm.v64_r2_probabilistic_state_gate_training_data import (
    V68FoldScale,
    V68FoldTrainingData,
    V68TensorStore,
)


def tiny_v68_fold() -> V68FoldTrainingData:
    dates = pd.date_range("2020-01-01", periods=260, tz="UTC")
    symbols = ("AAAUSDT", "BBBUSDT", "CCCUSDT")
    rng = np.random.default_rng(68)
    panel_rows = []
    label_rows = []
    for symbol_index, symbol in enumerate(symbols):
        for date in dates:
            panel_rows.append(
                {"date": date, "symbol": symbol, **dict(zip(BASE_FEATURES, rng.normal(0, 0.02, len(BASE_FEATURES)), strict=True))}
            )
            label_rows.append(
                {
                    "date": date, "symbol": symbol,
                    "target_h1_maturity_date": date + pd.Timedelta(days=2),
                    "target_h1_open_to_open_log_return": 0.001 * (symbol_index + 1),
                    "h1_label_complete": True,
                }
            )
    panel = pd.DataFrame(panel_rows)
    labels = pd.DataFrame(label_rows)
    scaler = FeatureScaler(
        feature_names=BASE_FEATURES,
        mean=tuple(float(x) for x in panel[list(BASE_FEATURES)].mean()),
        scale=tuple(float(x) for x in panel[list(BASE_FEATURES)].std(ddof=0)),
        source_relative_feature_index=1,
        fit_scope="v63_eligible_train_unique_symbol_date_cells_only",
        fit_start="2020-01-01", fit_end="2020-09-16", fit_rows=len(panel),
    )
    store = V68TensorStore(panel, labels, lookback_days=256)
    scale = V68FoldScale(
        fold=1, feature_scaler=scaler,
        source_v63_feature_scaler_state_sha256=scaler.state_sha256(),
        source_v63_fold_scale_sha256="a" * 64,
        market_target_rms=0.002,
        exact_gate_train_triplet_pairs=3,
    )
    return V68FoldTrainingData(
        fold=1, train_symbols=symbols, heldout_symbols=("DDDUSDT",),
        registered_triplets=tuple(combinations(symbols, 3)), store=store,
        train_availability={date: symbols for date in dates[255:258]},
        validation_availability={date: symbols for date in dates[258:]},
        scale=scale,
        access_receipt={"access_sha256": "b" * 64, "target_assets_loaded": [], "heldout_symbols_loaded": []},
    )


def test_v68_materializes_exact_state_and_normalized_market_target() -> None:
    data = tiny_v68_fold()
    draws = data.sampler(seed=42, role="gate_train").sample(epoch=1, sample_count=2)
    state, target = data.store.materialize(
        draws, data.scale.feature_scaler, market_target_rms=data.scale.market_target_rms
    )
    assert state.shape == (2, 256, 18)
    assert state.dtype == np.float32
    assert target.shape == (2,)
    assert np.allclose(target, 1.0)
    assert np.isfinite(state).all()
    assert data.sampler(seed=42, role="gate_train").sample(1, 2) == draws

