from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SequenceDataset:
    x: np.ndarray
    y: np.ndarray
    dates: pd.DatetimeIndex
    feature_names: tuple[str, ...]
    asset_names: tuple[str, ...]


@dataclass(frozen=True)
class WalkForwardSplit:
    fold: int
    train: np.ndarray
    test: np.ndarray


def make_sequences(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    lookback: int,
) -> SequenceDataset:
    if lookback < 2:
        raise ValueError("lookback must be at least 2")
    if not features.index.equals(targets.index):
        raise ValueError("Feature and target indexes must match")

    x_values: list[np.ndarray] = []
    y_values: list[np.ndarray] = []
    dates: list[pd.Timestamp] = []
    feature_array = features.to_numpy(dtype=np.float32)
    target_array = targets.to_numpy(dtype=np.float32)
    for end in range(lookback - 1, len(features)):
        start = end - lookback + 1
        window = feature_array[start : end + 1]
        label = target_array[end]
        if not np.isfinite(window).all() or not np.isfinite(label).all():
            continue
        x_values.append(window)
        y_values.append(label)
        dates.append(features.index[end])
    if not x_values:
        raise ValueError("No valid sequences after rolling-feature warmup")
    return SequenceDataset(
        x=np.stack(x_values),
        y=np.stack(y_values),
        dates=pd.DatetimeIndex(dates),
        feature_names=tuple(features.columns),
        asset_names=tuple(targets.columns),
    )


def expanding_walk_forward_splits(
    n_samples: int,
    folds: int = 3,
    min_train_fraction: float = 0.5,
) -> list[WalkForwardSplit]:
    return walk_forward_splits(
        n_samples=n_samples,
        folds=folds,
        min_train_fraction=min_train_fraction,
        mode="expanding",
    )


def walk_forward_splits(
    n_samples: int,
    folds: int = 3,
    min_train_fraction: float = 0.5,
    mode: str = "expanding",
    train_window_samples: int | None = None,
) -> list[WalkForwardSplit]:
    if folds < 1 or not 0.2 <= min_train_fraction < 0.9:
        raise ValueError("Invalid walk-forward configuration")
    if mode not in {"expanding", "rolling"}:
        raise ValueError(f"Unsupported walk-forward mode: {mode}")
    min_train = int(n_samples * min_train_fraction)
    if mode == "rolling":
        train_window_samples = train_window_samples or min_train
        if train_window_samples < 30 or train_window_samples > min_train:
            raise ValueError("Rolling train window must be between 30 and initial train size")
    remaining = n_samples - min_train
    test_size = remaining // folds
    if min_train < 30 or test_size < 10:
        raise ValueError("Not enough samples for requested walk-forward splits")
    splits: list[WalkForwardSplit] = []
    for fold in range(folds):
        test_start = min_train + fold * test_size
        test_end = n_samples if fold == folds - 1 else test_start + test_size
        train_start = 0 if mode == "expanding" else test_start - int(train_window_samples)
        splits.append(WalkForwardSplit(
            fold=fold,
            train=np.arange(train_start, test_start),
            test=np.arange(test_start, test_end),
        ))
    return splits
