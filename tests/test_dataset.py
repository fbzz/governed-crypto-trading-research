import numpy as np

from tlm.data import generate_fixture
from tlm.dataset import expanding_walk_forward_splits, make_sequences, walk_forward_splits
from tlm.features import build_features_and_targets


def test_sequences_have_expected_shapes():
    features, targets = build_features_and_targets(
        generate_fixture(["BTC", "ETH", "SOL"], days=180)
    )
    dataset = make_sequences(features, targets, lookback=32)
    assert dataset.x.ndim == 3
    assert dataset.x.shape[1:] == (32, 30)
    assert dataset.y.shape == (len(dataset.x), 3)
    assert np.isfinite(dataset.x).all()


def test_walk_forward_is_ordered_and_disjoint():
    splits = expanding_walk_forward_splits(300, folds=3, min_train_fraction=0.5)
    assert len(splits) == 3
    previous_test_end = 149
    for split in splits:
        assert split.train[-1] < split.test[0]
        assert not set(split.train).intersection(split.test)
        assert split.test[0] == previous_test_end + 1
        previous_test_end = split.test[-1]
    assert previous_test_end == 299


def test_rolling_walk_forward_keeps_fixed_training_window():
    splits = walk_forward_splits(
        500,
        folds=5,
        min_train_fraction=0.4,
        mode="rolling",
        train_window_samples=150,
    )
    assert len(splits) == 5
    assert all(len(split.train) == 150 for split in splits)
    assert all(split.train[-1] + 1 == split.test[0] for split in splits)
    assert splits[0].train[0] == 50
    assert splits[-1].train[0] > splits[0].train[0]


def test_walk_forward_rejects_invalid_mode_or_window():
    import pytest

    with pytest.raises(ValueError, match="Unsupported"):
        walk_forward_splits(500, mode="random")
    with pytest.raises(ValueError, match="Rolling train window"):
        walk_forward_splits(
            500,
            min_train_fraction=0.4,
            mode="rolling",
            train_window_samples=250,
        )
