from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def build_features_and_targets(
    frames: Mapping[str, pd.DataFrame],
    target_mode: str = "next_open_to_close",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build close-of-t features and causally executable future labels."""
    assets = list(frames)
    if not assets:
        raise ValueError("No assets supplied")

    feature_parts: list[pd.DataFrame] = []
    target_parts: list[pd.Series] = []
    returns_7: dict[str, pd.Series] = {}
    for asset, frame in frames.items():
        close = frame["close"]
        log_close = np.log(close)
        log_volume = np.log(frame["volume"])
        part = pd.DataFrame(index=frame.index)
        for horizon in (1, 3, 7, 14):
            part[f"{asset}__return_{horizon}"] = log_close.diff(horizon)
        returns_7[asset] = part[f"{asset}__return_7"]
        part[f"{asset}__volatility_7"] = log_close.diff().rolling(7).std()
        part[f"{asset}__volatility_21"] = log_close.diff().rolling(21).std()
        part[f"{asset}__range"] = (frame["high"] - frame["low"]) / frame["open"]
        candle_range = (frame["high"] - frame["low"]).replace(0.0, np.nan)
        part[f"{asset}__close_position"] = (frame["close"] - frame["low"]) / candle_range
        volume_mean = log_volume.rolling(21).mean()
        volume_std = log_volume.rolling(21).std().replace(0.0, np.nan)
        part[f"{asset}__volume_z21"] = (log_volume - volume_mean) / volume_std
        feature_parts.append(part)
        if target_mode == "next_open_to_close":
            target = np.log(frame["close"].shift(-1) / frame["open"].shift(-1))
        elif target_mode == "next_open_to_open":
            target = np.log(frame["open"].shift(-2) / frame["open"].shift(-1))
        else:
            raise ValueError(f"Unsupported target mode: {target_mode}")
        target_parts.append(target.rename(asset))

    cross_mean = pd.concat(returns_7, axis=1).mean(axis=1)
    relative = pd.DataFrame(
        {f"{asset}__relative_strength_7": values - cross_mean for asset, values in returns_7.items()}
    )
    features = pd.concat([*feature_parts, relative], axis=1).sort_index(axis=1)
    targets = pd.concat(target_parts, axis=1)[assets]
    return features.replace([np.inf, -np.inf], np.nan), targets
