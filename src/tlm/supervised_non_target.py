from __future__ import annotations

from dataclasses import asdict, fields
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
import yaml

from .non_target_pretraining import (
    TARGET_SYMBOLS,
    TripletTensorStore,
    _atomic_torch_save,
    _canonical_sha256,
    _sha256_file,
    _write_json,
    load_pretrained_checkpoint,
)
from .patch_transformer import MultiAssetPatchTransformer, PREDICTION_HEADS
from .scientific_harness import (
    DeterministicEligibleTripletSampler,
    EarlyStoppingState,
    FeatureScaler,
    supervised_probabilistic_loss,
)


RETURN_HEADS = ("return_q10", "return_q50", "return_q90")
RETURN_QUANTILES = (0.10, 0.50, 0.90)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def model_state_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(
            json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def calibration_semantic_sha256(state: dict[str, object]) -> str:
    """Hash calibration meaning without serialization-specific checkpoint hashes."""
    semantic = {
        key: value
        for key, value in state.items()
        if key not in {
            "calibration_semantic_sha256",
            "calibration_state_sha256",
            "member_checkpoint_sha256",
        }
    }
    return _canonical_sha256(semantic)


def calibration_state_sha256(state: dict[str, object]) -> str:
    payload = {
        key: value
        for key, value in state.items()
        if key != "calibration_state_sha256"
    }
    return _canonical_sha256(payload)


class SupervisedTripletTensorStore:
    """Fold-local feature and label store for supervised training."""

    def __init__(
        self,
        panel: pd.DataFrame,
        feature_names: list[str],
        label_names: list[str],
        lookback_days: int,
        relative_source_feature: str,
    ) -> None:
        required = {"date", "symbol", *feature_names, *label_names}
        if not required.issubset(panel.columns):
            raise ValueError(f"Panel columns missing: {sorted(required - set(panel.columns))}")
        self.feature_store = TripletTensorStore(
            panel[["date", "symbol", *feature_names]],
            feature_names,
            lookback_days,
            relative_source_feature,
        )
        self.label_names = tuple(label_names)
        labels = np.full(
            (
                len(self.feature_store.symbols),
                len(self.feature_store.dates),
                len(label_names),
            ),
            np.nan,
            dtype=np.float32,
        )
        for symbol, frame in panel.groupby("symbol", sort=True):
            frame = frame.sort_values("date")
            symbol_index = self.feature_store.symbol_to_index[symbol]
            date_indexes = [
                self.feature_store.date_to_index[pd.Timestamp(date)]
                for date in frame["date"]
            ]
            labels[symbol_index, date_indexes] = frame[label_names].to_numpy(
                dtype=np.float32
            )
        self.labels = labels

    def materialize_batch(
        self,
        samples: list[dict[str, object]],
        scaler: FeatureScaler,
    ) -> tuple[np.ndarray, np.ndarray]:
        x = self.feature_store.materialize_batch(samples, scaler)
        asset_indexes = np.asarray([
            [
                self.feature_store.symbol_to_index[str(symbol)]
                for symbol in sample["triplet"]
            ]
            for sample in samples
        ], dtype=np.int64)
        date_indexes = np.asarray([
            self.feature_store.date_to_index[pd.Timestamp(sample["date"])]
            for sample in samples
        ], dtype=np.int64)
        y = self.labels[asset_indexes, date_indexes[:, None], :]
        if y.shape != (len(samples), 3, len(self.label_names)):
            raise RuntimeError("Supervised label shape drift")
        if not np.isfinite(y).all():
            raise ValueError("Eligible supervised sample contains a non-finite label")
        return x, y.astype(np.float32, copy=False)


def eligible_supervised_availability(
    panel: pd.DataFrame,
    split_column: str,
    split_end: str,
    role_symbols: list[str],
) -> tuple[dict[pd.Timestamp, list[str]], dict[str, object]]:
    end = pd.Timestamp(split_end, tz="UTC")
    dates = pd.to_datetime(panel["date"], utc=True)
    maturity = pd.to_datetime(panel["target_window_end_date"], utc=True)
    subset = panel.loc[
        panel[split_column]
        & panel["supervised_sequence_ready"]
        & panel["label_complete"]
        & (maturity <= end)
        & panel["symbol"].isin(role_symbols),
        ["date", "symbol", "target_window_end_date"],
    ].copy()
    availability = {
        pd.Timestamp(date): sorted(frame["symbol"].tolist())
        for date, frame in subset.groupby("date", sort=True)
    }
    eligible_dates = [date for date, symbols in availability.items() if len(symbols) >= 3]
    if not eligible_dates:
        raise ValueError(f"No eligible supervised dates for {split_column}")
    audit = {
        "split_column": split_column,
        "maturity_boundary": end.date().isoformat(),
        "first_eligible_signal_date": min(eligible_dates).date().isoformat(),
        "last_eligible_signal_date": max(eligible_dates).date().isoformat(),
        "maximum_target_maturity": pd.to_datetime(
            subset["target_window_end_date"], utc=True
        ).max().date().isoformat(),
        "eligible_dates": len(eligible_dates),
        "eligible_symbol_date_rows": len(subset),
    }
    return availability, audit


def supervised_parameter_names(model: MultiAssetPatchTransformer) -> list[str]:
    excluded = ("mask_token", "reconstruction_head.")
    return [
        name for name, _ in model.named_parameters()
        if not name.startswith(excluded)
    ]


def _supervised_parameters(model: MultiAssetPatchTransformer) -> list[nn.Parameter]:
    allowed = set(supervised_parameter_names(model))
    parameters = [parameter for name, parameter in model.named_parameters() if name in allowed]
    if not parameters:
        raise RuntimeError("No supervised parameters selected")
    return parameters


def _seed_device(seed: int, device: torch.device) -> None:
    torch.manual_seed(int(seed))
    if device.type == "mps":
        torch.mps.manual_seed(int(seed))


def _rng_state(device: torch.device) -> dict[str, torch.Tensor]:
    result = {"cpu": torch.get_rng_state()}
    if device.type == "mps":
        result["mps"] = torch.mps.get_rng_state()
    return result


def _restore_rng_state(state: dict[str, torch.Tensor], device: torch.device) -> None:
    torch.set_rng_state(state["cpu"])
    if device.type == "mps":
        torch.mps.set_rng_state(state["mps"])


def _configure_device(name: str, torch_threads: int) -> torch.device:
    torch.set_num_threads(int(torch_threads))
    torch.use_deterministic_algorithms(True)
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was requested but is unavailable; run v36 outside the sandbox"
        )
    if name not in {"cpu", "mps"}:
        raise ValueError("V36 device must be cpu or mps")
    return torch.device(name)


def _run_supervised_batches(
    model: MultiAssetPatchTransformer,
    store: SupervisedTripletTensorStore,
    scaler: FeatureScaler,
    samples: list[dict[str, object]],
    batch_size: int,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float,
    volatility_weight: float,
    volatility_floor: float,
) -> tuple[dict[str, float], int]:
    training = optimizer is not None
    model.train(training)
    trainable_parameters = _supervised_parameters(model) if training else []
    totals = {
        "total": 0.0,
        "return_q10": 0.0,
        "return_q50": 0.0,
        "return_q90": 0.0,
        "log_volatility": 0.0,
    }
    observations = 0
    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        x_np, y_np = store.materialize_batch(batch_samples, scaler)
        x = torch.from_numpy(x_np).to(device)
        y = torch.from_numpy(y_np).to(device)
        with torch.set_grad_enabled(training):
            output = model(x)
            losses = supervised_probabilistic_loss(
                output, y, volatility_weight, volatility_floor
            )
        if not all(bool(torch.isfinite(value)) for value in losses.values()):
            raise RuntimeError("Non-finite supervised loss")
        if training:
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                trainable_parameters, gradient_clip_norm
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("Non-finite supervised gradient norm")
            optimizer.step()
        count = len(batch_samples)
        for name in totals:
            totals[name] += float(losses[name].detach().cpu()) * count
        observations += count
    return (
        {name: value / observations for name, value in totals.items()},
        math.ceil(observations / batch_size),
    )


def _scaler_from_record(record: dict) -> FeatureScaler:
    names = {field.name for field in fields(FeatureScaler)}
    values = {name: record[name] for name in names}
    values["feature_names"] = tuple(values["feature_names"])
    values["mean"] = tuple(values["mean"])
    values["scale"] = tuple(values["scale"])
    return FeatureScaler(**values)


def build_supervised_spec(
    blueprint: dict,
    supervised: dict,
    smoke: bool,
) -> dict[str, object]:
    effective = supervised["smoke"] if smoke else supervised["full_run"]
    splits = blueprint["chronological_splits"]
    purge_days = int(supervised["label_boundary_purge_days"])
    boundaries = {}
    for name in ("supervised_train", "validation", "calibration"):
        start, end = splits[name]
        boundaries[name] = {
            "window_start": start,
            "window_end": end,
            "last_eligible_signal_date": (
                pd.Timestamp(end) - pd.Timedelta(days=purge_days)
            ).date().isoformat(),
            "maximum_target_maturity": end,
        }
    spec = {
        "version": "v36",
        "candidate_family_id": blueprint["candidate_family_id"],
        "phase": "supervised_non_target_quantile_volatility_training",
        "folds": effective["folds"],
        "seeds": effective["seeds"],
        "chronological_boundaries": boundaries,
        "label_boundary_purge_days": purge_days,
        "train_samples_per_epoch": effective["train_samples_per_epoch"],
        "validation_samples": effective["validation_samples"],
        "calibration_samples": effective["calibration_samples"],
        "batch_size": effective["batch_size"],
        "maximum_epochs": effective["maximum_epochs"],
        "early_stopping_patience": effective["early_stopping_patience"],
        "objective": "mean_pinball_q10_q50_q90_plus_0.1_huber_log_volatility",
        "optimizer": {
            "name": "AdamW",
            "learning_rate": blueprint["training"]["learning_rate"],
            "weight_decay": blueprint["training"]["weight_decay"],
            "gradient_clip_norm": supervised["gradient_clip_norm"],
            "scheduler": None,
        },
        "calibration": {
            **supervised["calibration_method"],
            "seed": supervised["calibration_seed"],
            "sampling_epoch": supervised["calibration_sampling_epoch"],
            "ensemble_rule": blueprint["training"]["ensemble_rule"],
        },
        "device": supervised["device"],
        "torch_threads": supervised["torch_threads"],
        "resume_granularity": "completed_epoch",
        "seed_selection_allowed": False,
        "held_out_fold_assets_allowed": False,
        "target_assets_loaded": False,
        "performance_metrics_allowed": False,
        "smoke": bool(smoke),
    }
    spec["supervised_spec_sha256"] = _canonical_sha256(spec)
    return spec


def _save_resume(
    path: Path,
    model: MultiAssetPatchTransformer,
    optimizer: torch.optim.Optimizer,
    early_stopping: EarlyStoppingState,
    history: list[dict[str, object]],
    completed_epoch: int,
    metadata: dict[str, object],
    device: torch.device,
) -> None:
    _atomic_torch_save({
        "format_version": "v36_resume_v1",
        "metadata": metadata,
        "completed_epoch": completed_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "early_stopping": asdict(early_stopping),
        "history": history,
        "rng_state": _rng_state(device),
    }, path)


def _load_resume(
    path: Path,
    model: MultiAssetPatchTransformer,
    optimizer: torch.optim.Optimizer,
    metadata: dict[str, object],
    device: torch.device,
) -> tuple[int, EarlyStoppingState, list[dict[str, object]]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format_version") != "v36_resume_v1":
        raise RuntimeError("Unsupported v36 resume checkpoint")
    if payload.get("metadata") != metadata:
        raise RuntimeError("V36 resume metadata drift")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    _restore_rng_state(payload["rng_state"], device)
    return (
        int(payload["completed_epoch"]),
        EarlyStoppingState(**payload["early_stopping"]),
        list(payload["history"]),
    )


def load_supervised_checkpoint(
    path: str | Path,
) -> tuple[MultiAssetPatchTransformer, dict]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format_version") != "v36_supervised_non_target_v1":
        raise RuntimeError("Unsupported v36 supervised checkpoint")
    if payload.get("architecture_sha256") != _canonical_sha256(
        payload["architecture"]
    ):
        raise RuntimeError("V36 checkpoint architecture hash mismatch")
    model = MultiAssetPatchTransformer(
        int(payload["input_features"]), payload["architecture"]
    )
    model.load_state_dict(payload["state_dict"])
    if any(
        not bool(torch.isfinite(value).all())
        for value in model.state_dict().values()
        if value.is_floating_point()
    ):
        raise RuntimeError("V36 checkpoint contains non-finite model state")
    expected_state_hash = payload.get("metadata", {}).get("model_state_sha256")
    if expected_state_hash and model_state_sha256(model.state_dict()) != expected_state_hash:
        raise RuntimeError("V36 checkpoint model-state hash mismatch")
    return model, payload


def _train_job(
    fold_entry: dict,
    seed: int,
    parent_entry: dict,
    parent_checkpoint: Path,
    store: SupervisedTripletTensorStore,
    scaler: FeatureScaler,
    train_availability: dict[pd.Timestamp, list[str]],
    validation_availability: dict[pd.Timestamp, list[str]],
    blueprint: dict,
    supervised: dict,
    effective: dict,
    supervised_spec: dict,
    artifact_hashes: dict[str, str],
    checkpoint_root: Path,
    device: torch.device,
) -> dict[str, object]:
    fold = int(fold_entry["fold"])
    train_symbols = list(fold_entry["train_symbols"])
    test_symbols = list(fold_entry["test_symbols"])
    if TARGET_SYMBOLS.intersection(train_symbols + test_symbols):
        raise RuntimeError("Target symbol entered v36 fold")
    job_dir = checkpoint_root / f"fold_{fold}" / f"seed_{seed}"
    job_dir.mkdir(parents=True, exist_ok=True)
    complete_path = job_dir / "complete.json"
    if complete_path.is_file():
        complete = _load_json(complete_path)
        checkpoint = job_dir / "checkpoint.pt"
        if (
            complete.get("supervised_spec_sha256")
            != supervised_spec["supervised_spec_sha256"]
            or not checkpoint.is_file()
            or _sha256_file(checkpoint) != complete.get("checkpoint_sha256")
            or not complete.get("model_state_sha256")
        ):
            raise RuntimeError(f"Completed v36 job drifted: fold={fold}, seed={seed}")
        return complete

    if _sha256_file(parent_checkpoint) != parent_entry["checkpoint_sha256"]:
        raise RuntimeError("V35 parent checkpoint hash drift")
    model, parent_payload = load_pretrained_checkpoint(parent_checkpoint)
    parent_metadata = parent_payload["metadata"]
    if (
        int(parent_metadata["fold"]) != fold
        or int(parent_metadata["initialization_seed"]) != int(seed)
        or parent_metadata["scaler_state_sha256"] != scaler.state_sha256()
    ):
        raise RuntimeError("V35 parent checkpoint metadata drift")
    model.to(device)
    optimizer = torch.optim.AdamW(
        _supervised_parameters(model),
        lr=float(blueprint["training"]["learning_rate"]),
        weight_decay=float(blueprint["training"]["weight_decay"]),
    )
    early_stopping = EarlyStoppingState(
        patience=int(effective["early_stopping_patience"]), minimum_delta=0.0
    )
    metadata = {
        "version": "v36",
        "candidate_family_id": blueprint["candidate_family_id"],
        "fold": fold,
        "initialization_seed": int(seed),
        "train_symbols": train_symbols,
        "test_symbols": test_symbols,
        "parent_v35_checkpoint_sha256": parent_entry["checkpoint_sha256"],
        "scaler_state_sha256": scaler.state_sha256(),
        "supervised_spec_sha256": supervised_spec["supervised_spec_sha256"],
        **artifact_hashes,
    }
    train_sampler = DeterministicEligibleTripletSampler(
        train_availability, train_symbols, seed, fold
    )
    validation_sampler = DeterministicEligibleTripletSampler(
        validation_availability, train_symbols, seed, fold
    )
    validation_samples = validation_sampler.sample_epoch(
        int(supervised["validation_sampling_epoch"]),
        int(effective["validation_samples"]),
    )
    _seed_device(int(seed), device)
    history: list[dict[str, object]] = []
    completed_epoch = 0
    resume_path = job_dir / "resume.pt"
    best_path = job_dir / "best_state.pt"
    if resume_path.is_file():
        completed_epoch, early_stopping, history = _load_resume(
            resume_path, model, optimizer, metadata, device
        )
    started = time.perf_counter()
    for epoch in range(completed_epoch + 1, int(effective["maximum_epochs"]) + 1):
        epoch_started = time.perf_counter()
        train_samples = train_sampler.sample_epoch(
            epoch, int(effective["train_samples_per_epoch"])
        )
        train_losses, train_steps = _run_supervised_batches(
            model,
            store,
            scaler,
            train_samples,
            int(effective["batch_size"]),
            device,
            optimizer,
            float(supervised["gradient_clip_norm"]),
            float(supervised["volatility_weight"]),
            float(supervised["volatility_floor"]),
        )
        validation_losses, validation_steps = _run_supervised_batches(
            model,
            store,
            scaler,
            validation_samples,
            int(effective["batch_size"]),
            device,
            None,
            float(supervised["gradient_clip_norm"]),
            float(supervised["volatility_weight"]),
            float(supervised["volatility_floor"]),
        )
        prior_best = early_stopping.best_loss
        should_stop = early_stopping.update(epoch, validation_losses["total"])
        history.append({
            "epoch": epoch,
            "train_losses": train_losses,
            "validation_losses": validation_losses,
            "improved": validation_losses["total"] < prior_best,
            "epoch_seconds": time.perf_counter() - epoch_started,
            "train_optimizer_steps": train_steps,
            "validation_steps": validation_steps,
        })
        if validation_losses["total"] < prior_best:
            _atomic_torch_save({
                "format_version": "v36_best_state_v1",
                "metadata": metadata,
                "best_epoch": epoch,
                "best_validation_loss": validation_losses["total"],
                "model_state_dict": model.state_dict(),
            }, best_path)
        _save_resume(
            resume_path,
            model,
            optimizer,
            early_stopping,
            history,
            epoch,
            metadata,
            device,
        )
        _write_json(job_dir / "progress.json", {
            "fold": fold,
            "seed": int(seed),
            "completed_epoch": epoch,
            "best_epoch": early_stopping.best_epoch,
            "best_validation_loss": early_stopping.best_loss,
            "stale_epochs": early_stopping.stale_epochs,
            "should_stop": should_stop,
            "last_epoch": history[-1],
        })
        if should_stop:
            break
    if not best_path.is_file():
        raise RuntimeError("V36 training did not produce a best state")
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    state_hash = model_state_sha256(model.state_dict())
    final_payload = {
        "format_version": "v36_supervised_non_target_v1",
        "input_features": model.input_features,
        "architecture": parent_payload["architecture"],
        "architecture_sha256": _canonical_sha256(parent_payload["architecture"]),
        "metadata": {
            **metadata,
            "checkpoint_status": "frozen_supervised_no_seed_selection",
            "best_epoch": int(best["best_epoch"]),
            "best_validation_loss": float(best["best_validation_loss"]),
            "completed_epochs": len(history),
            "model_state_sha256": state_hash,
            "supervised_parameter_names": supervised_parameter_names(model),
            "unused_during_supervised_training": [
                "mask_token",
                "reconstruction_head",
            ],
        },
        "state_dict": model.state_dict(),
    }
    checkpoint_path = job_dir / "checkpoint.pt"
    _atomic_torch_save(final_payload, checkpoint_path)
    complete = {
        "version": "v36",
        "fold": fold,
        "seed": int(seed),
        "train_symbols": train_symbols,
        "test_symbols": test_symbols,
        "parent_v35_checkpoint_sha256": parent_entry["checkpoint_sha256"],
        "scaler_state_sha256": scaler.state_sha256(),
        "supervised_spec_sha256": supervised_spec["supervised_spec_sha256"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "model_state_sha256": state_hash,
        "best_epoch": int(best["best_epoch"]),
        "best_validation_loss": float(best["best_validation_loss"]),
        "completed_epochs": len(history),
        "train_optimizer_steps": int(sum(
            row["train_optimizer_steps"] for row in history
        )),
        "elapsed_seconds_current_invocation": time.perf_counter() - started,
        "history": history,
        "seed_selected": False,
        "held_out_assets_used": False,
        "target_assets_loaded": False,
        "performance_metrics_computed": False,
    }
    _write_json(complete_path, complete)
    resume_path.unlink(missing_ok=True)
    best_path.unlink(missing_ok=True)
    if device.type == "mps":
        torch.mps.empty_cache()
    return complete


def _collect_ensemble_predictions(
    models: list[MultiAssetPatchTransformer],
    store: SupervisedTripletTensorStore,
    scaler: FeatureScaler,
    samples: list[dict[str, object]],
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    for model in models:
        model.eval()
    collected = {name: [] for name in PREDICTION_HEADS}
    collected_labels = []
    for start in range(0, len(samples), batch_size):
        batch_samples = samples[start : start + batch_size]
        x_np, y_np = store.materialize_batch(batch_samples, scaler)
        x = torch.from_numpy(x_np).to(device)
        with torch.no_grad():
            member_outputs = [model(x) for model in models]
        for name in PREDICTION_HEADS:
            mean = torch.stack([output[name] for output in member_outputs]).mean(0)
            collected[name].append(mean.cpu().numpy())
        collected_labels.append(y_np)
    return (
        {name: np.concatenate(values) for name, values in collected.items()},
        np.concatenate(collected_labels),
    )


def compute_calibration_parameters(
    predictions: dict[str, np.ndarray],
    labels: np.ndarray,
    volatility_floor: float,
) -> dict[str, object]:
    if labels.ndim != 3 or labels.shape[1:] != (3, 2):
        raise ValueError("Calibration labels must have shape [samples, 3, 2]")
    if any(predictions[name].shape != labels.shape[:2] for name in PREDICTION_HEADS):
        raise ValueError("Calibration prediction shape drift")
    observed_return = labels[..., 0]
    observed_log_vol = np.log(np.maximum(labels[..., 1], volatility_floor))
    raw_quantiles = np.stack([predictions[name] for name in RETURN_HEADS], axis=-1)
    offsets = np.asarray([
        np.quantile(
            observed_return - predictions[name], quantile, method="linear"
        )
        for name, quantile in zip(RETURN_HEADS, RETURN_QUANTILES, strict=True)
    ])
    volatility_offset = float(np.quantile(
        observed_log_vol - predictions["volatility_7d"], 0.50, method="linear"
    ))
    calibrated_quantiles = np.sort(
        raw_quantiles + offsets, axis=-1, kind="stable"
    )
    raw_crossing = (raw_quantiles[..., 0] > raw_quantiles[..., 1]) | (
        raw_quantiles[..., 1] > raw_quantiles[..., 2]
    )
    calibrated_crossing = (
        calibrated_quantiles[..., 0] > calibrated_quantiles[..., 1]
    ) | (calibrated_quantiles[..., 1] > calibrated_quantiles[..., 2])
    return {
        "offsets": {
            "return_q10": float(offsets[0]),
            "return_q50": float(offsets[1]),
            "return_q90": float(offsets[2]),
            "log_volatility": volatility_offset,
        },
        "diagnostics": {
            "raw_return_coverage": {
                name: float(np.mean(observed_return <= raw_quantiles[..., index]))
                for index, name in enumerate(RETURN_HEADS)
            },
            "calibrated_return_coverage": {
                name: float(np.mean(
                    observed_return <= calibrated_quantiles[..., index]
                ))
                for index, name in enumerate(RETURN_HEADS)
            },
            "raw_quantile_crossing_rate": float(np.mean(raw_crossing)),
            "calibrated_quantile_crossing_rate": float(
                np.mean(calibrated_crossing)
            ),
            "raw_q50_mae": float(np.mean(np.abs(
                observed_return - raw_quantiles[..., 1]
            ))),
            "calibrated_q50_mae": float(np.mean(np.abs(
                observed_return - calibrated_quantiles[..., 1]
            ))),
            "raw_log_vol_mae": float(np.mean(np.abs(
                observed_log_vol - predictions["volatility_7d"]
            ))),
            "calibrated_log_vol_mae": float(np.mean(np.abs(
                observed_log_vol
                - (predictions["volatility_7d"] + volatility_offset)
            ))),
        },
    }


def calibrate_fold_ensemble(
    fold: int,
    jobs: list[dict[str, object]],
    store: SupervisedTripletTensorStore,
    scaler: FeatureScaler,
    calibration_availability: dict[pd.Timestamp, list[str]],
    train_symbols: list[str],
    supervised: dict,
    effective: dict,
    supervised_spec: dict,
    checkpoint_root: Path,
    device: torch.device,
) -> dict[str, object]:
    calibration_path = checkpoint_root / f"fold_{fold}" / "calibration.json"
    if calibration_path.is_file():
        state = _load_json(calibration_path)
        recorded_state_hash = state.get("calibration_state_sha256")
        if recorded_state_hash and recorded_state_hash != calibration_state_sha256(
            state
        ):
            raise RuntimeError(f"V36 calibration state hash drifted for fold {fold}")
        if state.get("supervised_spec_sha256") != supervised_spec[
            "supervised_spec_sha256"
        ]:
            raise RuntimeError(f"V36 calibration drifted for fold {fold}")
        expected_member_seeds = sorted(int(job["seed"]) for job in jobs)
        expected_checkpoint_hashes = sorted(
            str(job["checkpoint_sha256"]) for job in jobs
        )
        expected_model_state_hashes = sorted(
            str(job["model_state_sha256"]) for job in jobs
        )
        if (
            state.get("member_seeds") != expected_member_seeds
            or state.get("member_checkpoint_sha256") != expected_checkpoint_hashes
            or state.get("member_model_state_sha256") != expected_model_state_hashes
            or state.get("sample_count") != int(effective["calibration_samples"])
        ):
            raise RuntimeError(f"V36 calibration members drifted for fold {fold}")
        state["calibration_semantic_sha256"] = calibration_semantic_sha256(state)
        state["calibration_state_sha256"] = calibration_state_sha256(state)
        _write_json(calibration_path, state)
        return state
    models = []
    for job in sorted(jobs, key=lambda row: int(row["seed"])):
        model, _ = load_supervised_checkpoint(job["checkpoint_path"])
        model.to(device)
        models.append(model)
    sampler = DeterministicEligibleTripletSampler(
        calibration_availability,
        train_symbols,
        int(supervised["calibration_seed"]),
        int(fold),
    )
    samples = sampler.sample_epoch(
        int(supervised["calibration_sampling_epoch"]),
        int(effective["calibration_samples"]),
    )
    predictions, labels = _collect_ensemble_predictions(
        models,
        store,
        scaler,
        samples,
        int(effective["batch_size"]),
        device,
    )
    calibration = compute_calibration_parameters(
        predictions, labels, float(supervised["volatility_floor"])
    )
    state = {
        "version": "v36",
        "fold": int(fold),
        "supervised_spec_sha256": supervised_spec["supervised_spec_sha256"],
        "train_symbols": train_symbols,
        "member_seeds": sorted(int(job["seed"]) for job in jobs),
        "member_checkpoint_sha256": sorted(
            job["checkpoint_sha256"] for job in jobs
        ),
        "member_model_state_sha256": sorted(
            job["model_state_sha256"] for job in jobs
        ),
        "sample_count": len(samples),
        "asset_prediction_count": int(labels[..., 0].size),
        "sampling_seed": int(supervised["calibration_seed"]),
        "sampling_epoch": int(supervised["calibration_sampling_epoch"]),
        **calibration,
        "quantile_projection": "stable_ascending_sort",
        "model_weights_updated": False,
        "policy_thresholds_changed": False,
        "seed_or_checkpoint_selected": False,
        "held_out_assets_used": False,
        "target_assets_loaded": False,
        "performance_metrics_computed": False,
    }
    state["calibration_semantic_sha256"] = calibration_semantic_sha256(state)
    state["calibration_state_sha256"] = calibration_state_sha256(state)
    _write_json(calibration_path, state)
    for model in models:
        model.to("cpu")
    if device.type == "mps":
        torch.mps.empty_cache()
    return state


def _report(result: dict[str, object]) -> str:
    summary = result["summary"]
    smoke = result["supervised_spec"]["smoke"]
    status = (
        "SUPERVISED SMOKE PASSED; FULL NINE-CHECKPOINT RUN IS AUTHORIZED."
        if smoke
        else "ALL NINE SUPERVISED CHECKPOINTS AND THREE CALIBRATIONS PASSED."
    )
    return "\n".join([
        "# TLM v36 Supervised Non-Target Training",
        "",
        "## Decision",
        "",
        f"**{status}**",
        "",
        f"Completed checkpoints: **{summary['checkpoint_count']}**",
        f"Fold calibrations: **{summary['calibration_count']}**",
        f"Total optimizer steps: **{summary['total_optimizer_steps']:,}**",
        f"Supervised-spec SHA-256: `{result['supervised_spec']['supervised_spec_sha256']}`",
        "",
        "Every model was initialized from its exact v35 checkpoint. Training, validation, and calibration label maturities were purged at chronological boundaries. All seeds are retained and averaged; no checkpoint was selected.",
        "",
        "The 2025 calibration applies frozen residual offsets and monotone quantile projection only. It does not update model weights or policy thresholds. Held-out fold assets, BTC/ETH/SOL, portfolios, PnL, Sharpe, and drawdown remain unobserved.",
        "",
        "## Next action",
        "",
        "V37 may run the one-shot 2026 asset-disjoint source-domain evaluation using each fold's ten held-out assets, three-member ensemble, frozen scaler, and calibration state. It may not retrain or inspect BTC/ETH/SOL.",
        "",
    ])


def run_supervised_non_target(config: dict, smoke: bool = False) -> dict[str, object]:
    supervised = config["supervised_non_target"]
    root = Path(supervised["project_root"]).resolve()
    paths = {name: root / relative for name, relative in supervised["inputs"].items()}
    for name, path in paths.items():
        if (
            not path.is_file()
            or _sha256_file(path) != supervised["expected_input_sha256"][name]
        ):
            raise RuntimeError(f"V36 input missing or hash drifted: {name}")
    amendment = _load_json(paths["v29_amendment"])
    dataset_manifest = _load_json(paths["v32_dataset_manifest"])
    feature_schema = _load_json(paths["v32_feature_schema"])
    triplet_catalog = _load_json(paths["v32_triplet_catalog"])
    v35_result = _load_json(paths["v35_result"])
    v35_audit = _load_json(paths["v35_audit"])
    parent_manifest = _load_json(paths["v35_checkpoint_manifest"])
    if (
        v35_result["decision"] != "authorize_v36_supervised_non_target_training_only"
        or not v35_audit.get("passed")
    ):
        raise RuntimeError("V35 does not authorize v36")
    blueprint = amendment["blueprint"]
    feature_names = list(dataset_manifest["panel_features"])
    label_names = list(dataset_manifest["labels"])
    if feature_schema["model_feature_order"][:-1] != feature_names:
        raise RuntimeError("V36 feature-order drift")
    if TARGET_SYMBOLS.intersection(dataset_manifest["symbols"]):
        raise RuntimeError("Target symbols entered v36 dataset")
    if int(blueprint["training"]["maximum_finetune_epochs"]) != int(
        supervised["full_run"]["maximum_epochs"]
    ):
        raise RuntimeError("V36 maximum-epoch drift")
    if list(blueprint["training"]["seeds"]) != list(
        supervised["full_run"]["seeds"]
    ):
        raise RuntimeError("V36 seed drift")
    effective = supervised["smoke"] if smoke else supervised["full_run"]
    supervised_spec = build_supervised_spec(blueprint, supervised, smoke)
    device = _configure_device(supervised["device"], supervised["torch_threads"])
    folds_by_number = {int(fold["fold"]): fold for fold in triplet_catalog["folds"]}
    parents_by_key = {
        (int(row["fold"]), int(row["seed"])): row for row in parent_manifest
    }
    artifact_hashes = {
        "dataset_manifest_sha256": _sha256_file(paths["v32_dataset_manifest"]),
        "feature_schema_sha256": _sha256_file(paths["v32_feature_schema"]),
        "harness_spec_sha256": _load_json(paths["v34_harness_spec"])[
            "harness_spec_sha256"
        ],
        "v35_pretraining_spec_sha256": v35_result["pretraining_spec"][
            "pretraining_spec_sha256"
        ],
        "panel_sha256": _sha256_file(paths["panel"]),
    }
    checkpoint_root = root / (
        supervised["smoke_checkpoint_dir"] if smoke else supervised["checkpoint_dir"]
    )
    parent_root = root / supervised["v35_checkpoint_dir"]
    panel_columns = [
        "date",
        "symbol",
        "target_window_end_date",
        "label_complete",
        "supervised_sequence_ready",
        "in_supervised_train",
        "in_validation",
        "in_calibration",
        *feature_names,
        *label_names,
    ]
    jobs: list[dict[str, object]] = []
    calibrations: list[dict[str, object]] = []
    eligibility_audits: list[dict[str, object]] = []
    for fold_number in effective["folds"]:
        fold = folds_by_number[int(fold_number)]
        train_symbols = list(fold["train_symbols"])
        panel = pd.read_parquet(
            paths["panel"],
            columns=panel_columns,
            filters=[("symbol", "in", train_symbols)],
        )
        if set(panel["symbol"].unique()) != set(train_symbols):
            raise RuntimeError("V36 fold-local panel symbol drift")
        store = SupervisedTripletTensorStore(
            panel,
            feature_names,
            label_names,
            int(blueprint["architecture"]["lookback_days"]),
            "log_close_to_close_return",
        )
        scaler_records = []
        for seed in effective["seeds"]:
            scaler_records.append(_load_json(
                parent_root / f"fold_{fold_number}" / f"seed_{seed}" / "scaler.json"
            ))
        scaler = _scaler_from_record(scaler_records[0])
        if (
            any(_scaler_from_record(record).state_sha256() != scaler.state_sha256()
                for record in scaler_records)
            or scaler.state_sha256() != scaler_records[0]["scaler_state_sha256"]
        ):
            raise RuntimeError("V36 fold scaler drift")
        split_config = blueprint["chronological_splits"]
        train_availability, train_audit = eligible_supervised_availability(
            panel, "in_supervised_train", split_config["supervised_train"][1], train_symbols
        )
        validation_availability, validation_audit = eligible_supervised_availability(
            panel, "in_validation", split_config["validation"][1], train_symbols
        )
        calibration_availability, calibration_audit = eligible_supervised_availability(
            panel, "in_calibration", split_config["calibration"][1], train_symbols
        )
        for split_name, audit in (
            ("supervised_train", train_audit),
            ("validation", validation_audit),
            ("calibration", calibration_audit),
        ):
            eligibility_audits.append({"fold": int(fold_number), "split": split_name, **audit})
        fold_jobs = []
        for seed in effective["seeds"]:
            parent_entry = parents_by_key[(int(fold_number), int(seed))]
            parent_checkpoint = (
                parent_root / f"fold_{fold_number}" / f"seed_{seed}" / "checkpoint.pt"
            )
            job = _train_job(
                fold,
                int(seed),
                parent_entry,
                parent_checkpoint,
                store,
                scaler,
                train_availability,
                validation_availability,
                blueprint,
                supervised,
                effective,
                supervised_spec,
                artifact_hashes,
                checkpoint_root,
                device,
            )
            jobs.append(job)
            fold_jobs.append(job)
        calibrations.append(calibrate_fold_ensemble(
            int(fold_number),
            fold_jobs,
            store,
            scaler,
            calibration_availability,
            train_symbols,
            supervised,
            effective,
            supervised_spec,
            checkpoint_root,
            device,
        ))
    expected_jobs = len(effective["folds"]) * len(effective["seeds"])
    loaded = [load_supervised_checkpoint(job["checkpoint_path"])[1] for job in jobs]
    fold_scaler_hashes = {
        int(fold): {job["scaler_state_sha256"] for job in jobs if job["fold"] == int(fold)}
        for fold in effective["folds"]
    }
    checks = {
        "v35_authorizes_v36": True,
        "checkpoint_count_is_exact": len(jobs) == expected_jobs,
        "fold_seed_combinations_are_unique": len({
            (job["fold"], job["seed"]) for job in jobs
        }) == expected_jobs,
        "all_checkpoint_hashes_match": all(
            _sha256_file(Path(job["checkpoint_path"])) == job["checkpoint_sha256"]
            for job in jobs
        ),
        "all_checkpoint_metadata_roundtrips": all(
            payload["metadata"]["fold"] == job["fold"]
            and payload["metadata"]["initialization_seed"] == job["seed"]
            and payload["metadata"]["parent_v35_checkpoint_sha256"]
            == job["parent_v35_checkpoint_sha256"]
            and payload["metadata"]["supervised_spec_sha256"]
            == supervised_spec["supervised_spec_sha256"]
            and payload["metadata"]["model_state_sha256"]
            == job["model_state_sha256"]
            and model_state_sha256(payload["state_dict"])
            == job["model_state_sha256"]
            for payload, job in zip(loaded, jobs, strict=True)
        ),
        "all_losses_are_finite": all(
            math.isfinite(float(job["best_validation_loss"])) for job in jobs
        ),
        "one_identical_scaler_per_fold": all(
            len(hashes) == 1 for hashes in fold_scaler_hashes.values()
        ),
        "all_parent_v35_hashes_preserved": all(
            job["parent_v35_checkpoint_sha256"]
            == parents_by_key[(job["fold"], job["seed"])]["checkpoint_sha256"]
            for job in jobs
        ),
        "no_seed_or_checkpoint_selection": all(not job["seed_selected"] for job in jobs)
        and all(not row["seed_or_checkpoint_selected"] for row in calibrations),
        "held_out_assets_never_used": all(not job["held_out_assets_used"] for job in jobs)
        and all(not row["held_out_assets_used"] for row in calibrations),
        "target_assets_never_loaded": all(not job["target_assets_loaded"] for job in jobs)
        and all(not row["target_assets_loaded"] for row in calibrations),
        "label_maturities_respect_boundaries": all(
            row["maximum_target_maturity"] == row["maturity_boundary"]
            and pd.Timestamp(row["last_eligible_signal_date"])
            <= pd.Timestamp(row["maturity_boundary"]) - pd.Timedelta(days=8)
            for row in eligibility_audits
        ),
        "one_calibration_per_fold": len(calibrations) == len(effective["folds"]),
        "calibration_sample_count_is_exact": all(
            row["sample_count"] == int(effective["calibration_samples"])
            for row in calibrations
        ),
        "calibration_is_weight_and_policy_free": all(
            not row["model_weights_updated"]
            and not row["policy_thresholds_changed"]
            for row in calibrations
        ),
        "calibrated_quantiles_are_monotone": all(
            row["diagnostics"]["calibrated_quantile_crossing_rate"] == 0.0
            for row in calibrations
        ),
        "calibration_state_hashes_match": all(
            row["calibration_state_sha256"] == calibration_state_sha256(row)
            and row["calibration_semantic_sha256"]
            == calibration_semantic_sha256(row)
            for row in calibrations
        ),
        "calibration_members_match_fold_checkpoints": all(
            row["member_seeds"]
            == sorted(job["seed"] for job in jobs if job["fold"] == row["fold"])
            and row["member_checkpoint_sha256"]
            == sorted(
                job["checkpoint_sha256"]
                for job in jobs
                if job["fold"] == row["fold"]
            )
            and row["member_model_state_sha256"]
            == sorted(
                job["model_state_sha256"]
                for job in jobs
                if job["fold"] == row["fold"]
            )
            for row in calibrations
        ),
        "no_performance_metrics_computed": all(
            not job["performance_metrics_computed"] for job in jobs
        ) and all(not row["performance_metrics_computed"] for row in calibrations),
        "full_run_has_nine_checkpoints_and_three_calibrations": bool(smoke)
        or (len(jobs) == 9 and len(calibrations) == 3),
        "mps_device_used": bool(smoke) or supervised["device"] == "mps",
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise RuntimeError(f"V36 supervised audit failed: {checks}")
    manifest = [{
        key: job[key]
        for key in (
            "fold",
            "seed",
            "train_symbols",
            "test_symbols",
            "parent_v35_checkpoint_sha256",
            "scaler_state_sha256",
            "checkpoint_path",
            "checkpoint_sha256",
            "model_state_sha256",
            "best_epoch",
            "best_validation_loss",
            "completed_epochs",
            "train_optimizer_steps",
        )
    } for job in jobs]
    summary = {
        "checkpoint_count": len(jobs),
        "calibration_count": len(calibrations),
        "total_completed_epochs": int(sum(job["completed_epochs"] for job in jobs)),
        "total_optimizer_steps": int(sum(job["train_optimizer_steps"] for job in jobs)),
        "best_validation_loss_range": [
            min(job["best_validation_loss"] for job in jobs),
            max(job["best_validation_loss"] for job in jobs),
        ],
        "device": supervised["device"],
    }
    result = {
        "version": "v36_smoke" if smoke else "v36",
        "decision": (
            "authorize_v36_full_run" if smoke
            else "authorize_v37_one_shot_asset_disjoint_source_test_only"
        ),
        "supervised_spec": supervised_spec,
        "summary": summary,
        "checkpoint_manifest": manifest,
        "calibrations": calibrations,
        "eligibility_audit": eligibility_audits,
        "tested": {
            "supervised_labels_loaded": True,
            "supervised_training_executed": True,
            "feature_only_validation_replaced_by_label_validation": True,
            "fold_calibration_executed": True,
            "held_out_fold_assets_used": False,
            "target_assets_loaded": False,
            "portfolio_constructed": False,
            "performance_metrics_computed": False,
            "seed_selection_executed": False,
        },
        "audit": {"passed": True, "checks": checks},
    }
    output = root / (
        supervised["smoke_output_dir"] if smoke else config["output_dir"]
    )
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "supervised_spec.json", supervised_spec)
    _write_json(output / "checkpoint_manifest.json", manifest)
    _write_json(output / "calibration_states.json", calibrations)
    _write_json(output / "eligibility_audit.json", eligibility_audits)
    _write_json(output / "training_histories.json", jobs)
    _write_json(output / "audit.json", result["audit"])
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    _write_json(output / "result.json", result)
    return result
