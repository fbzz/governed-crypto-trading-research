from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .audit import audit_artifacts
from .backtest import (
    run_equal_weight_buy_hold,
    run_equal_weight_intraday_benchmark,
    run_long_cash_backtest,
    run_persistent_long_cash_backtest,
)
from .data import load_market_data
from .dataset import SequenceDataset, make_sequences, walk_forward_splits
from .features import build_features_and_targets
from .models import predict_transformer, save_checkpoint, seed_everything, train_transformer
from .report import write_artifacts


def _fit_sequence_scaler(x_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(x_train.reshape(-1, x_train.shape[-1]))
    return scaler


def _transform_sequences(scaler: StandardScaler, x: np.ndarray) -> np.ndarray:
    shape = x.shape
    transformed = scaler.transform(x.reshape(-1, shape[-1])).reshape(shape)
    return transformed.astype(np.float32)


def _prediction_frame(
    dataset: SequenceDataset,
    indexes: np.ndarray,
    predictions: np.ndarray,
    fold: int,
    model: str,
) -> pd.DataFrame:
    frame = pd.DataFrame({"date": dataset.dates[indexes], "fold": fold, "model": model})
    for asset_index, asset in enumerate(dataset.asset_names):
        frame[f"pred_{asset}"] = predictions[:, asset_index]
        frame[f"actual_{asset}"] = dataset.y[indexes, asset_index]
    return frame


def run_experiment(
    config: dict,
    models: Iterable[str] = ("ridge", "transformer"),
    force_data: bool = False,
) -> dict[str, dict]:
    models = tuple(models)
    unsupported = set(models) - {"ridge", "transformer"}
    if unsupported:
        raise ValueError(f"Unsupported models: {sorted(unsupported)}")
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    frames = load_market_data(config, force=force_data)
    target_mode = config.get("target", {}).get("mode", "next_open_to_close")
    features, targets = build_features_and_targets(frames, target_mode=target_mode)
    dataset = make_sequences(features, targets, int(config["features"]["lookback"]))
    validation_config = config["validation"]
    splits = walk_forward_splits(
        len(dataset.x),
        folds=int(validation_config["folds"]),
        min_train_fraction=float(validation_config["min_train_fraction"]),
        mode=validation_config.get("mode", "expanding"),
        train_window_samples=validation_config.get("train_window_samples"),
    )
    prediction_frames: list[pd.DataFrame] = []

    for split in splits:
        if "ridge" in models:
            scaler = _fit_sequence_scaler(dataset.x[split.train])
            x_train = _transform_sequences(scaler, dataset.x[split.train]).reshape(len(split.train), -1).astype(np.float64)
            x_test = _transform_sequences(scaler, dataset.x[split.test]).reshape(len(split.test), -1).astype(np.float64)
            ridge = Ridge(alpha=float(config["ridge"]["alpha"]), solver="lsqr")
            ridge.fit(x_train, dataset.y[split.train])
            # NumPy 2.x can emit spurious overflow warnings in sklearn's
            # `X @ coef.T` path on some BLAS builds even when all operands and
            # outputs are small and finite. Einsum is equivalent and stable.
            ridge_prediction = np.einsum("ij,kj->ik", x_test, ridge.coef_) + ridge.intercept_
            if not np.isfinite(ridge_prediction).all():
                raise FloatingPointError("Ridge produced non-finite predictions")
            prediction_frames.append(_prediction_frame(
                dataset, split.test, ridge_prediction, split.fold, "ridge"
            ))

        if "transformer" in models:
            validation_fraction = float(config["transformer"]["validation_fraction"])
            validation_size = max(16, int(len(split.train) * validation_fraction))
            if validation_size >= len(split.train) - 20:
                raise ValueError("Training window is too small for chronological validation")
            core_train = split.train[:-validation_size]
            validation = split.train[-validation_size:]
            scaler = _fit_sequence_scaler(dataset.x[core_train])
            x_train = _transform_sequences(scaler, dataset.x[core_train])
            x_validation = _transform_sequences(scaler, dataset.x[validation])
            x_test = _transform_sequences(scaler, dataset.x[split.test])
            result = train_transformer(
                x_train,
                dataset.y[core_train],
                x_validation,
                dataset.y[validation],
                config["transformer"],
                seed + split.fold,
            )
            save_checkpoint(
                result,
                output / f"transformer_fold_{split.fold}.pt",
                {
                    "fold": split.fold,
                    "feature_names": dataset.feature_names,
                    "asset_names": dataset.asset_names,
                    "train_end": str(dataset.dates[core_train[-1]]),
                    "validation_end": str(dataset.dates[validation[-1]]),
                    "test_start": str(dataset.dates[split.test[0]]),
                    "scaler_mean": scaler.mean_.tolist(),
                    "scaler_scale": scaler.scale_.tolist(),
                },
            )
            prediction_frames.append(_prediction_frame(
                dataset,
                split.test,
                predict_transformer(result.model, x_test),
                split.fold,
                "transformer",
            ))

    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(["model", "date"])
    metrics: dict[str, dict] = {}
    curves: dict[str, pd.DataFrame] = {}
    pred_columns = [f"pred_{asset}" for asset in dataset.asset_names]
    actual_columns = [f"actual_{asset}" for asset in dataset.asset_names]
    for model in models:
        model_frame = predictions[predictions["model"] == model].sort_values("date")
        backtest = (
            run_persistent_long_cash_backtest
            if target_mode == "next_open_to_open"
            else run_long_cash_backtest
        )
        curve, model_metrics = backtest(
            model_frame[pred_columns].to_numpy(),
            model_frame[actual_columns].to_numpy(),
            pd.DatetimeIndex(model_frame["date"]),
            dataset.asset_names,
            threshold=float(config["strategy"]["prediction_threshold"]),
            cost_bps=float(config["strategy"]["cost_bps"]),
            **({
                "always_invested": config["strategy"].get("policy") == "always_long_top1"
            } if target_mode == "next_open_to_open" else {}),
        )
        metrics[model] = model_metrics
        curves[model] = curve

    reference = predictions[predictions["model"] == models[0]].sort_values("date")
    if target_mode == "next_open_to_close":
        benchmark_curve, benchmark_metrics = run_equal_weight_intraday_benchmark(
            reference[actual_columns].to_numpy(),
            pd.DatetimeIndex(reference["date"]),
            cost_bps=float(config["strategy"]["cost_bps"]),
        )
        metrics["equal_weight_intraday"] = benchmark_metrics
        curves["equal_weight_intraday"] = benchmark_curve
        benchmark_returns = pd.DataFrame({
            asset: np.log(frame["close"].shift(-1) / frame["close"])
            for asset, frame in frames.items()
        }).reindex(pd.DatetimeIndex(reference["date"]))
    else:
        benchmark_returns = reference[actual_columns].copy()
        benchmark_returns.columns = list(dataset.asset_names)
    if benchmark_returns.isna().any().any():
        raise ValueError("Buy-and-hold benchmark dates are not aligned with market data")
    buy_hold_curve, buy_hold_metrics = run_equal_weight_buy_hold(
        benchmark_returns.to_numpy(),
        pd.DatetimeIndex(reference["date"]),
        cost_bps=float(config["strategy"]["cost_bps"]),
    )
    metrics["equal_weight_buy_hold"] = buy_hold_metrics
    curves["equal_weight_buy_hold"] = buy_hold_curve
    write_artifacts(
        output,
        predictions,
        metrics,
        curves,
        context={
            "data_source": config["data"]["source"],
            "assets": list(dataset.asset_names),
            "start": str(dataset.dates.min().date()),
            "end": str(dataset.dates.max().date()),
            "sequences": len(dataset.x),
            "folds": len(splits),
            "cost_bps": float(config["strategy"]["cost_bps"]),
            "target_mode": target_mode,
            "policy": config["strategy"].get("policy", "long_cash"),
            "model_objective": config["transformer"].get("objective", "huber_regression"),
            "walk_forward_mode": validation_config.get("mode", "expanding"),
        },
    )
    audit_artifacts(output)
    return metrics
