import numpy as np

from tlm.data import generate_fixture
from tlm.features import build_features_and_targets


def test_target_is_next_day_open_to_close_log_return():
    frames = generate_fixture(["BTC", "ETH", "SOL"], days=160, seed=3)
    _, targets = build_features_and_targets(frames)
    t = frames["BTC"].index[80]
    tomorrow = frames["BTC"].iloc[81]
    expected = np.log(tomorrow["close"] / tomorrow["open"])
    assert targets.loc[t, "BTC"] == expected


def test_expected_causal_feature_set_is_finite_after_warmup():
    frames = generate_fixture(["BTC", "ETH", "SOL"], days=160)
    features, targets = build_features_and_targets(frames)
    assert len(features.columns) == 30
    assert list(targets.columns) == ["BTC", "ETH", "SOL"]
    assert np.isfinite(features.iloc[30:].to_numpy()).all()


def test_open_to_open_target_enters_after_signal_day():
    frames = generate_fixture(["BTC", "ETH", "SOL"], days=160, seed=4)
    _, targets = build_features_and_targets(frames, target_mode="next_open_to_open")
    t = frames["BTC"].index[80]
    expected = np.log(frames["BTC"].iloc[82]["open"] / frames["BTC"].iloc[81]["open"])
    assert targets.loc[t, "BTC"] == expected
