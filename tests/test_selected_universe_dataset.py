import numpy as np
import pandas as pd

from tlm.non_target_dataset import PANEL_FEATURES, build_asset_folds, build_symbol_panel
from tlm.selected_universe_dataset import (
    build_sequence_index,
    build_triplet_catalog,
    materialize_triplet_sequence,
)


def _frame(days=300, drift=0.01):
    index = pd.date_range("2021-01-01", periods=days, freq="D", tz="UTC")
    opened = np.exp(np.arange(days) * drift)
    closed = opened * np.exp(drift / 2)
    return pd.DataFrame({
        "open": opened,
        "high": closed * 1.01,
        "low": opened * 0.99,
        "close": closed,
        "volume": np.arange(days, dtype=float) + 100,
        "quote_volume": np.arange(days, dtype=float) + 1000,
        "trade_count": np.arange(days, dtype=float) + 10,
    }, index=index)


def _splits():
    return {
        "representation_train": ["2021-01-01", "2021-10-27"],
        "supervised_train": ["2021-01-01", "2021-10-27"],
    }


def test_sequence_index_preserves_exact_lookback_boundary():
    frame = _frame()
    panel = build_symbol_panel("AAAUSDT", frame, frame.index, _splits(), 256)
    index = build_sequence_index(panel, _splits(), 256)
    assert index.iloc[0]["date"] == frame.index[285]
    assert index.iloc[0]["sequence_start_date"] == frame.index[30]
    assert (index["date"] - index["sequence_start_date"] == pd.Timedelta(days=255)).all()


def test_triplet_catalog_is_fold_disjoint_and_excludes_targets():
    symbols = [f"A{index:02d}USDT" for index in range(30)]
    catalog = build_triplet_catalog(
        build_asset_folds(symbols, 3), {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
    )
    assert len(catalog["folds"]) == 3
    assert all(len(fold["train_triplets"]) == 1140 for fold in catalog["folds"])
    assert all(len(fold["test_triplets"]) == 120 for fold in catalog["folds"])
    assert catalog["catalog_sha256"]


def test_triplet_loader_materializes_256_by_3_by_9_without_future_data():
    frames = {
        "AAAUSDT": _frame(drift=0.010),
        "BBBUSDT": _frame(drift=0.015),
        "CCCUSDT": _frame(drift=0.020),
    }
    panels = [
        build_symbol_panel(symbol, frame, frame.index, _splits(), 256)
        for symbol, frame in frames.items()
    ]
    panel = pd.concat(panels, ignore_index=True)
    end = frames["AAAUSDT"].index[290]
    x, y = materialize_triplet_sequence(
        panel, ["AAAUSDT", "BBBUSDT", "CCCUSDT"], end, 256
    )
    assert x.shape == (256, 3, len(PANEL_FEATURES) + 1)
    assert y.shape == (3, 2)
    assert x.dtype == np.float32
    assert np.allclose(x[:, :, -1].sum(axis=1), 0.0, atol=1e-6)

    changed = panel.copy()
    changed.loc[changed["date"] > end, list(PANEL_FEATURES)] = 999.0
    changed_x, changed_y = materialize_triplet_sequence(
        changed, ["AAAUSDT", "BBBUSDT", "CCCUSDT"], end, 256
    )
    np.testing.assert_array_equal(x, changed_x)
    np.testing.assert_array_equal(y, changed_y)
