import pandas as pd

from tlm.data import generate_fixture
from tlm.features import build_features_and_targets


def test_future_ohlcv_change_cannot_modify_past_features():
    frames = generate_fixture(["BTC", "ETH", "SOL"], days=180, seed=9)
    original, _ = build_features_and_targets(frames)
    cutoff = frames["BTC"].index[120]
    perturbed = {asset: frame.copy() for asset, frame in frames.items()}
    for frame in perturbed.values():
        future = frame.index > cutoff
        frame.loc[future, ["open", "high", "low", "close", "volume"]] *= 4.0
    changed, _ = build_features_and_targets(perturbed)
    pd.testing.assert_frame_equal(original.loc[:cutoff], changed.loc[:cutoff])


def test_target_at_t_is_allowed_to_depend_on_t_plus_one_only():
    frames = generate_fixture(["BTC", "ETH", "SOL"], days=180, seed=10)
    _, targets = build_features_and_targets(frames)
    t = frames["BTC"].index[100]
    modified = {asset: frame.copy() for asset, frame in frames.items()}
    modified["BTC"].loc[modified["BTC"].index[101], "close"] *= 1.1
    # Preserve candle validity after moving the close.
    modified["BTC"].loc[modified["BTC"].index[101], "high"] *= 1.1
    _, changed_targets = build_features_and_targets(modified)
    assert targets.loc[t, "BTC"] != changed_targets.loc[t, "BTC"]
    pd.testing.assert_series_equal(targets.loc[: t].iloc[:-1]["BTC"], changed_targets.loc[: t].iloc[:-1]["BTC"])
