from __future__ import annotations

from dataclasses import asdict
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

from .patch_transformer import MultiAssetPatchTransformer
from .scientific_harness import (
    DeterministicEligibleTripletSampler,
    EarlyStoppingState,
    FeatureScaler,
    deterministic_patch_mask,
    masked_reconstruction_loss,
)


TARGET_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary, _use_new_zipfile_serialization=False)
    temporary.replace(path)


def load_pretrained_checkpoint(
    path: str | Path,
) -> tuple[MultiAssetPatchTransformer, dict]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format_version") != "v35_non_target_pretraining_v1":
        raise RuntimeError("Unsupported v35 pretrained checkpoint")
    architecture = payload["architecture"]
    if payload.get("architecture_sha256") != _canonical_sha256(architecture):
        raise RuntimeError("V35 checkpoint architecture hash mismatch")
    model = MultiAssetPatchTransformer(int(payload["input_features"]), architecture)
    model.load_state_dict(payload["state_dict"])
    if any(
        not bool(torch.isfinite(value).all())
        for value in model.state_dict().values()
        if value.is_floating_point()
    ):
        raise RuntimeError("V35 checkpoint contains non-finite model state")
    return model, payload


class TripletTensorStore:
    """Compact calendar-aligned feature store; labels are never loaded."""

    def __init__(
        self,
        panel: pd.DataFrame,
        feature_names: list[str],
        lookback_days: int,
        relative_source_feature: str,
    ) -> None:
        required = {"date", "symbol", *feature_names}
        if not required.issubset(panel.columns):
            raise ValueError(f"Panel columns missing: {sorted(required - set(panel.columns))}")
        self.feature_names = tuple(feature_names)
        self.lookback_days = int(lookback_days)
        self.relative_source_index = feature_names.index(relative_source_feature)
        self.symbols = tuple(sorted(panel["symbol"].unique()))
        self.symbol_to_index = {symbol: index for index, symbol in enumerate(self.symbols)}
        self.dates = pd.DatetimeIndex(sorted(pd.to_datetime(panel["date"], utc=True).unique()))
        self.date_to_index = {date: index for index, date in enumerate(self.dates)}
        values = np.full(
            (len(self.symbols), len(self.dates), len(feature_names)),
            np.nan,
            dtype=np.float32,
        )
        for symbol, frame in panel.groupby("symbol", sort=True):
            frame = frame.sort_values("date")
            indexes = [self.date_to_index[pd.Timestamp(date)] for date in frame["date"]]
            values[self.symbol_to_index[symbol], indexes] = frame[
                feature_names
            ].to_numpy(dtype=np.float32)
        self.values = values

    def materialize_batch(
        self,
        samples: list[dict[str, object]],
        scaler: FeatureScaler,
    ) -> np.ndarray:
        if not samples:
            raise ValueError("Cannot materialize an empty sample batch")
        asset_indexes = np.asarray(
            [
                [self.symbol_to_index[str(symbol)] for symbol in sample["triplet"]]
                for sample in samples
            ],
            dtype=np.int64,
        )
        end_indexes = np.asarray(
            [self.date_to_index[pd.Timestamp(sample["date"])] for sample in samples],
            dtype=np.int64,
        )
        if int(end_indexes.min()) < self.lookback_days - 1:
            raise ValueError("Sample does not have the frozen lookback")
        time_indexes = end_indexes[:, None] + np.arange(
            -self.lookback_days + 1, 1, dtype=np.int64
        )[None, :]
        base = self.values[
            asset_indexes[:, None, :], time_indexes[:, :, None], :
        ]
        if base.shape[1:] != (
            self.lookback_days,
            3,
            len(self.feature_names),
        ):
            raise RuntimeError("Materialized batch shape drift")
        if not np.isfinite(base).all():
            raise ValueError("Eligible sample contains a non-finite feature")
        source = base[..., self.relative_source_index]
        relative = source - source.mean(axis=2, keepdims=True)
        values = np.concatenate([base, relative[..., None]], axis=3)
        return scaler.transform_triplet_tensor(values)


def _availability_by_date(
    sequence_index: pd.DataFrame,
    split_column: str,
    role_symbols: list[str],
) -> dict[pd.Timestamp, list[str]]:
    subset = sequence_index.loc[
        sequence_index[split_column]
        & sequence_index["symbol"].isin(role_symbols),
        ["date", "symbol"],
    ]
    return {
        pd.Timestamp(date): sorted(frame["symbol"].tolist())
        for date, frame in subset.groupby("date", sort=True)
    }


def pretraining_parameter_names(model: MultiAssetPatchTransformer) -> list[str]:
    prefixes = (
        "temporal_position",
        "mask_token",
        "patch_projection.",
        "temporal_encoder.",
        "temporal_norm.",
        "reconstruction_head.",
    )
    return [name for name, _ in model.named_parameters() if name.startswith(prefixes)]


def _pretraining_parameters(model: MultiAssetPatchTransformer) -> list[nn.Parameter]:
    allowed = set(pretraining_parameter_names(model))
    parameters = [parameter for name, parameter in model.named_parameters() if name in allowed]
    if not parameters:
        raise RuntimeError("No pretraining parameters selected")
    return parameters


def _run_batches(
    model: MultiAssetPatchTransformer,
    store: TripletTensorStore,
    scaler: FeatureScaler,
    samples: list[dict[str, object]],
    batch_size: int,
    seed: int,
    fold: int,
    mask_epoch: int,
    mask_fraction: float,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float,
) -> tuple[float, int]:
    training = optimizer is not None
    model.train(training)
    trainable_parameters = _pretraining_parameters(model) if training else []
    weighted_loss = 0.0
    observations = 0
    for batch_index, start in enumerate(range(0, len(samples), batch_size)):
        batch_samples = samples[start : start + batch_size]
        x = torch.from_numpy(store.materialize_batch(batch_samples, scaler)).to(device)
        mask = deterministic_patch_mask(
            len(batch_samples),
            model.triplet_size,
            model.patch_count,
            mask_fraction,
            seed,
            fold,
            mask_epoch,
            batch_index,
        ).to(device)
        with torch.set_grad_enabled(training):
            target_patches = model.extract_patches(x)
            reconstruction = model.reconstruct_masked_patches(x, mask)
            loss = masked_reconstruction_loss(reconstruction, target_patches, mask)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Non-finite masked reconstruction loss")
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(
                trainable_parameters, gradient_clip_norm
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("Non-finite gradient norm")
            optimizer.step()
        count = len(batch_samples)
        weighted_loss += float(loss.detach().cpu()) * count
        observations += count
    return weighted_loss / observations, math.ceil(observations / batch_size)


def build_pretraining_spec(
    blueprint: dict,
    pretraining: dict,
    smoke: bool,
) -> dict[str, object]:
    effective = pretraining["smoke"] if smoke else pretraining["full_run"]
    spec = {
        "version": "v35",
        "candidate_family_id": blueprint["candidate_family_id"],
        "phase": "masked_patch_non_target_representation_pretraining",
        "folds": effective["folds"],
        "seeds": effective["seeds"],
        "training_window": blueprint["chronological_splits"]["representation_train"],
        "feature_only_validation_window": blueprint["chronological_splits"]["validation"],
        "train_samples_per_epoch": effective["train_samples_per_epoch"],
        "validation_samples": effective["validation_samples"],
        "validation_sampling_epoch": pretraining["validation_sampling_epoch"],
        "batch_size": effective["batch_size"],
        "maximum_epochs": effective["maximum_epochs"],
        "early_stopping_patience": effective["early_stopping_patience"],
        "optimizer": {
            "name": "AdamW",
            "learning_rate": blueprint["training"]["learning_rate"],
            "weight_decay": blueprint["training"]["weight_decay"],
            "gradient_clip_norm": pretraining["gradient_clip_norm"],
            "scheduler": None,
        },
        "mask_fraction": blueprint["training"]["mask_fraction"],
        "loss": "smooth_l1_beta_1_masked_patches_only",
        "device": pretraining["device"],
        "torch_threads": pretraining["torch_threads"],
        "resume_granularity": "completed_epoch",
        "seed_selection_allowed": False,
        "labels_loaded": False,
        "target_assets_loaded": False,
        "performance_metrics_allowed": False,
        "smoke": bool(smoke),
    }
    spec["pretraining_spec_sha256"] = _canonical_sha256(spec)
    return spec


def _save_resume(
    path: Path,
    model: MultiAssetPatchTransformer,
    optimizer: torch.optim.Optimizer,
    early_stopping: EarlyStoppingState,
    history: list[dict[str, object]],
    completed_epoch: int,
    metadata: dict[str, object],
) -> None:
    _atomic_torch_save({
        "format_version": "v35_resume_v1",
        "metadata": metadata,
        "completed_epoch": completed_epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "early_stopping": asdict(early_stopping),
        "history": history,
        "torch_rng_state": torch.get_rng_state(),
    }, path)


def _load_resume(
    path: Path,
    model: MultiAssetPatchTransformer,
    optimizer: torch.optim.Optimizer,
    expected_metadata: dict[str, object],
) -> tuple[int, EarlyStoppingState, list[dict[str, object]]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != "v35_resume_v1":
        raise RuntimeError("Unsupported v35 resume checkpoint")
    if payload.get("metadata") != expected_metadata:
        raise RuntimeError("V35 resume metadata drift")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    state = EarlyStoppingState(**payload["early_stopping"])
    torch.set_rng_state(payload["torch_rng_state"])
    return int(payload["completed_epoch"]), state, list(payload["history"])


def _train_job(
    fold_entry: dict,
    seed: int,
    architecture: dict,
    feature_names: list[str],
    panel: pd.DataFrame,
    sequence_index: pd.DataFrame,
    store: TripletTensorStore,
    blueprint: dict,
    pretraining: dict,
    effective: dict,
    pretraining_spec: dict,
    artifact_hashes: dict[str, str],
    checkpoint_root: Path,
) -> dict[str, object]:
    fold = int(fold_entry["fold"])
    train_symbols = list(fold_entry["train_symbols"])
    if TARGET_SYMBOLS.intersection(train_symbols):
        raise RuntimeError("Target symbol entered v35 training")
    job_dir = checkpoint_root / f"fold_{fold}" / f"seed_{seed}"
    job_dir.mkdir(parents=True, exist_ok=True)
    complete_path = job_dir / "complete.json"
    if complete_path.is_file():
        complete = _load_json(complete_path)
        checkpoint = job_dir / "checkpoint.pt"
        if (
            complete.get("pretraining_spec_sha256")
            != pretraining_spec["pretraining_spec_sha256"]
            or not checkpoint.is_file()
            or _sha256_file(checkpoint) != complete.get("checkpoint_sha256")
        ):
            raise RuntimeError(f"Completed v35 job drifted: fold={fold}, seed={seed}")
        return complete

    train_start, train_end = blueprint["chronological_splits"]["representation_train"]
    scaler_panel = panel.loc[
        panel["symbol"].isin(train_symbols), ["date", *feature_names]
    ]
    scaler = FeatureScaler.fit_from_panel(
        scaler_panel,
        feature_names,
        train_start,
        train_end,
        train_end,
        "log_close_to_close_return",
    )
    scaler_record = {
        **asdict(scaler),
        "fold": fold,
        "train_symbols": train_symbols,
        "scaler_state_sha256": scaler.state_sha256(),
    }
    _write_json(job_dir / "scaler.json", scaler_record)

    train_availability = _availability_by_date(
        sequence_index, "in_representation_train", train_symbols
    )
    validation_availability = _availability_by_date(
        sequence_index, "in_validation", train_symbols
    )
    train_sampler = DeterministicEligibleTripletSampler(
        train_availability, train_symbols, seed, fold
    )
    validation_sampler = DeterministicEligibleTripletSampler(
        validation_availability, train_symbols, seed, fold
    )
    validation_epoch = int(pretraining["validation_sampling_epoch"])
    validation_samples = validation_sampler.sample_epoch(
        validation_epoch, int(effective["validation_samples"])
    )

    torch.manual_seed(int(seed))
    model = MultiAssetPatchTransformer(len(feature_names) + 1, architecture)
    device = torch.device(pretraining["device"])
    model.to(device)
    optimizer = torch.optim.AdamW(
        _pretraining_parameters(model),
        lr=float(blueprint["training"]["learning_rate"]),
        weight_decay=float(blueprint["training"]["weight_decay"]),
    )
    early_stopping = EarlyStoppingState(
        patience=int(effective["early_stopping_patience"]),
        minimum_delta=0.0,
    )
    metadata = {
        "version": "v35",
        "candidate_family_id": blueprint["candidate_family_id"],
        "fold": fold,
        "initialization_seed": int(seed),
        "train_symbols": train_symbols,
        "test_symbols": list(fold_entry["test_symbols"]),
        "scaler_state_sha256": scaler.state_sha256(),
        "pretraining_spec_sha256": pretraining_spec["pretraining_spec_sha256"],
        **artifact_hashes,
    }
    resume_path = job_dir / "resume.pt"
    best_path = job_dir / "best_state.pt"
    history: list[dict[str, object]] = []
    completed_epoch = 0
    if resume_path.is_file():
        completed_epoch, early_stopping, history = _load_resume(
            resume_path, model, optimizer, metadata
        )

    started = time.perf_counter()
    train_steps = 0
    validation_steps = 0
    for epoch in range(completed_epoch + 1, int(effective["maximum_epochs"]) + 1):
        epoch_started = time.perf_counter()
        train_samples = train_sampler.sample_epoch(
            epoch, int(effective["train_samples_per_epoch"])
        )
        train_loss, epoch_train_steps = _run_batches(
            model,
            store,
            scaler,
            train_samples,
            int(effective["batch_size"]),
            seed,
            fold,
            epoch,
            float(blueprint["training"]["mask_fraction"]),
            device,
            optimizer,
            float(pretraining["gradient_clip_norm"]),
        )
        validation_loss, epoch_validation_steps = _run_batches(
            model,
            store,
            scaler,
            validation_samples,
            int(effective["batch_size"]),
            seed,
            fold,
            validation_epoch,
            float(blueprint["training"]["mask_fraction"]),
            device,
            None,
            float(pretraining["gradient_clip_norm"]),
        )
        train_steps += epoch_train_steps
        validation_steps += epoch_validation_steps
        prior_best = early_stopping.best_loss
        should_stop = early_stopping.update(epoch, validation_loss)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "improved": validation_loss < prior_best,
            "epoch_seconds": time.perf_counter() - epoch_started,
            "train_optimizer_steps": epoch_train_steps,
            "validation_steps": epoch_validation_steps,
        })
        if validation_loss < prior_best:
            _atomic_torch_save({
                "format_version": "v35_best_state_v1",
                "metadata": metadata,
                "best_epoch": epoch,
                "best_validation_loss": validation_loss,
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
        )
        _write_json(job_dir / "progress.json", {
            "fold": fold,
            "seed": seed,
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
        raise RuntimeError("V35 training did not produce a best state")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    model.load_state_dict(best["model_state_dict"])
    final_payload = {
        "format_version": "v35_non_target_pretraining_v1",
        "input_features": len(feature_names) + 1,
        "architecture": architecture,
        "architecture_sha256": _canonical_sha256(architecture),
        "metadata": {
            **metadata,
            "checkpoint_status": "frozen_pretrained_no_seed_selection",
            "best_epoch": int(best["best_epoch"]),
            "best_validation_loss": float(best["best_validation_loss"]),
            "completed_epochs": len(history),
            "pretraining_parameter_names": pretraining_parameter_names(model),
            "unused_during_pretraining": [
                "cross_asset_encoder",
                "cross_asset_norm",
                "prediction_heads",
            ],
        },
        "state_dict": model.state_dict(),
    }
    checkpoint_path = job_dir / "checkpoint.pt"
    _atomic_torch_save(final_payload, checkpoint_path)
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    complete = {
        "version": "v35",
        "fold": fold,
        "seed": int(seed),
        "train_symbols": train_symbols,
        "test_symbols": list(fold_entry["test_symbols"]),
        "scaler_state_sha256": scaler.state_sha256(),
        "scaler_fit_rows": scaler.fit_rows,
        "pretraining_spec_sha256": pretraining_spec["pretraining_spec_sha256"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "best_epoch": int(best["best_epoch"]),
        "best_validation_loss": float(best["best_validation_loss"]),
        "completed_epochs": len(history),
        "train_optimizer_steps": int(sum(row["train_optimizer_steps"] for row in history)),
        "validation_steps": int(sum(row["validation_steps"] for row in history)),
        "elapsed_seconds_current_invocation": time.perf_counter() - started,
        "history": history,
        "seed_selected": False,
        "labels_loaded": False,
        "target_assets_loaded": False,
        "performance_metrics_computed": False,
    }
    _write_json(complete_path, complete)
    resume_path.unlink(missing_ok=True)
    best_path.unlink(missing_ok=True)
    return complete


def _report(result: dict[str, object]) -> str:
    summary = result["summary"]
    status = (
        "SMOKE TRAINING PASSED; FULL NINE-CHECKPOINT RUN REMAINS NEXT."
        if result["pretraining_spec"]["smoke"]
        else "ALL NINE NON-TARGET PRETRAINING CHECKPOINTS PASSED."
    )
    return "\n".join([
        "# TLM v35 Non-Target Masked Pretraining",
        "",
        "## Decision",
        "",
        f"**{status}**",
        "",
        f"Completed checkpoints: **{summary['checkpoint_count']}**",
        f"Fold-seed jobs: **{summary['fold_seed_jobs']}**",
        f"Total optimizer steps: **{summary['total_optimizer_steps']:,}**",
        f"Pretraining-spec SHA-256: `{result['pretraining_spec']['pretraining_spec_sha256']}`",
        "",
        "Each fold scaler was fit only on its 2021-2023 training assets. Early stopping used fixed feature-only samples from 2024. Every checkpoint is retained; no seed or fold was selected using validation loss.",
        "",
        "Forward labels, BTC, ETH, SOL, portfolios, PnL, Sharpe, and drawdown were not loaded or computed.",
        "",
        "## Next action",
        "",
        "V36 may initialize supervised non-target training from every frozen fold-seed checkpoint. It must preserve the asset-disjoint folds and may not inspect BTC/ETH/SOL.",
        "",
    ])


def run_non_target_pretraining(config: dict, smoke: bool = False) -> dict[str, object]:
    pretraining = config["non_target_pretraining"]
    root = Path(pretraining["project_root"]).resolve()
    paths = {name: root / relative for name, relative in pretraining["inputs"].items()}
    for name, path in paths.items():
        expected = pretraining["expected_input_sha256"][name]
        if not path.is_file() or _sha256_file(path) != expected:
            raise RuntimeError(f"V35 input missing or hash drifted: {name}")

    amendment = _load_json(paths["v29_amendment"])
    dataset_manifest = _load_json(paths["v32_dataset_manifest"])
    feature_schema = _load_json(paths["v32_feature_schema"])
    triplet_catalog = _load_json(paths["v32_triplet_catalog"])
    v33_result = _load_json(paths["v33_result"])
    v34_result = _load_json(paths["v34_result"])
    v34_audit = _load_json(paths["v34_audit"])
    if v34_result["decision"] != "authorize_v35_full_non_target_pretraining_only":
        raise RuntimeError("V34 does not authorize v35 pretraining")
    if not v34_audit.get("passed"):
        raise RuntimeError("V34 audit does not pass")

    blueprint = amendment["blueprint"]
    architecture = blueprint["architecture"]
    feature_names = list(dataset_manifest["panel_features"])
    if feature_schema["model_feature_order"][:-1] != feature_names:
        raise RuntimeError("V35 feature-order drift")
    if TARGET_SYMBOLS.intersection(dataset_manifest["symbols"]):
        raise RuntimeError("Target symbols entered v35 dataset")
    if int(blueprint["training"]["maximum_pretrain_epochs"]) != int(
        pretraining["full_run"]["maximum_epochs"]
    ):
        raise RuntimeError("V35 maximum-epoch drift from frozen blueprint")
    if list(blueprint["training"]["seeds"]) != list(pretraining["full_run"]["seeds"]):
        raise RuntimeError("V35 seed drift from frozen blueprint")

    effective = pretraining["smoke"] if smoke else pretraining["full_run"]
    pretraining_spec = build_pretraining_spec(blueprint, pretraining, smoke)
    torch.set_num_threads(int(pretraining["torch_threads"]))
    torch.use_deterministic_algorithms(True)

    panel_columns = ["date", "symbol", *feature_names]
    sequence_columns = [
        "date",
        "symbol",
        "in_representation_train",
        "in_validation",
    ]
    panel = pd.read_parquet(paths["panel"], columns=panel_columns)
    sequence_index = pd.read_parquet(paths["sequence_index"], columns=sequence_columns)
    if TARGET_SYMBOLS.intersection(panel["symbol"].unique()):
        raise RuntimeError("Target symbols were loaded into v35")
    store = TripletTensorStore(
        panel,
        feature_names,
        int(architecture["lookback_days"]),
        "log_close_to_close_return",
    )
    folds_by_number = {int(fold["fold"]): fold for fold in triplet_catalog["folds"]}
    artifact_hashes = {
        "dataset_manifest_sha256": _sha256_file(paths["v32_dataset_manifest"]),
        "feature_schema_sha256": _sha256_file(paths["v32_feature_schema"]),
        "model_spec_sha256": v33_result["model_spec"]["model_spec_sha256"],
        "harness_spec_sha256": v34_result["harness_spec"]["harness_spec_sha256"],
        "panel_sha256": _sha256_file(paths["panel"]),
        "sequence_index_sha256": _sha256_file(paths["sequence_index"]),
    }
    checkpoint_root = root / (
        pretraining["smoke_checkpoint_dir"] if smoke else pretraining["checkpoint_dir"]
    )
    jobs = []
    for fold in effective["folds"]:
        for seed in effective["seeds"]:
            jobs.append(_train_job(
                folds_by_number[int(fold)],
                int(seed),
                architecture,
                feature_names,
                panel,
                sequence_index,
                store,
                blueprint,
                pretraining,
                effective,
                pretraining_spec,
                artifact_hashes,
                checkpoint_root,
            ))

    expected_jobs = len(effective["folds"]) * len(effective["seeds"])
    combinations = {(job["fold"], job["seed"]) for job in jobs}
    loaded_checkpoints = [
        load_pretrained_checkpoint(job["checkpoint_path"])[1] for job in jobs
    ]
    scaler_hashes_by_fold = {
        fold: {
            job["scaler_state_sha256"] for job in jobs if job["fold"] == fold
        }
        for fold in effective["folds"]
    }
    checks = {
        "v34_authorizes_pretraining": True,
        "checkpoint_count_is_exact": len(jobs) == expected_jobs,
        "fold_seed_combinations_are_unique": len(combinations) == expected_jobs,
        "all_checkpoints_exist_and_match": all(
            Path(job["checkpoint_path"]).is_file()
            and _sha256_file(Path(job["checkpoint_path"])) == job["checkpoint_sha256"]
            for job in jobs
        ),
        "all_losses_are_finite": all(
            math.isfinite(float(job["best_validation_loss"])) for job in jobs
        ),
        "all_scalers_are_fold_train_only": all(job["scaler_fit_rows"] > 0 for job in jobs),
        "one_identical_scaler_per_fold": all(
            len(hashes) == 1 for hashes in scaler_hashes_by_fold.values()
        ),
        "checkpoint_metadata_roundtrips": all(
            payload["metadata"]["fold"] == job["fold"]
            and payload["metadata"]["initialization_seed"] == job["seed"]
            and payload["metadata"]["pretraining_spec_sha256"]
            == pretraining_spec["pretraining_spec_sha256"]
            and payload["metadata"]["checkpoint_status"]
            == "frozen_pretrained_no_seed_selection"
            for payload, job in zip(loaded_checkpoints, jobs, strict=True)
        ),
        "checkpoint_training_scope_is_representation_only": all(
            set(payload["metadata"]["unused_during_pretraining"])
            == {"cross_asset_encoder", "cross_asset_norm", "prediction_heads"}
            and not any(
                name.startswith(("cross_asset_encoder.", "prediction_heads."))
                for name in payload["metadata"]["pretraining_parameter_names"]
            )
            for payload in loaded_checkpoints
        ),
        "no_seed_or_fold_selection": all(not job["seed_selected"] for job in jobs),
        "no_target_assets_loaded": all(not job["target_assets_loaded"] for job in jobs),
        "no_labels_loaded": all(not job["labels_loaded"] for job in jobs),
        "no_performance_metrics_computed": all(
            not job["performance_metrics_computed"] for job in jobs
        ),
        "full_run_has_nine_checkpoints": bool(smoke) or len(jobs) == 9,
        "full_run_has_three_folds_and_seeds": bool(smoke)
        or (
            {job["fold"] for job in jobs} == {1, 2, 3}
            and {job["seed"] for job in jobs} == {7, 42, 123}
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise RuntimeError(f"V35 pretraining audit failed: {checks}")

    manifest = [{
        key: job[key]
        for key in (
            "fold",
            "seed",
            "train_symbols",
            "test_symbols",
            "scaler_state_sha256",
            "checkpoint_path",
            "checkpoint_sha256",
            "best_epoch",
            "best_validation_loss",
            "completed_epochs",
            "train_optimizer_steps",
        )
    } for job in jobs]
    summary = {
        "checkpoint_count": len(jobs),
        "fold_seed_jobs": [f"fold_{job['fold']}/seed_{job['seed']}" for job in jobs],
        "total_optimizer_steps": int(sum(job["train_optimizer_steps"] for job in jobs)),
        "total_completed_epochs": int(sum(job["completed_epochs"] for job in jobs)),
        "best_validation_loss_range": [
            min(job["best_validation_loss"] for job in jobs),
            max(job["best_validation_loss"] for job in jobs),
        ],
    }
    result = {
        "version": "v35_smoke" if smoke else "v35",
        "decision": (
            "authorize_v35_full_run" if smoke else "authorize_v36_supervised_non_target_training_only"
        ),
        "pretraining_spec": pretraining_spec,
        "summary": summary,
        "checkpoint_manifest": manifest,
        "tested": {
            "real_non_target_features_loaded": True,
            "fold_scalers_fitted": True,
            "masked_pretraining_executed": True,
            "feature_only_validation_executed": True,
            "forward_labels_loaded": False,
            "supervised_training_executed": False,
            "target_assets_loaded": False,
            "portfolio_constructed": False,
            "performance_metrics_computed": False,
            "seed_selection_executed": False,
        },
        "audit": {"passed": True, "checks": checks},
    }
    output = root / (
        pretraining["smoke_output_dir"] if smoke else config["output_dir"]
    )
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "pretraining_spec.json", pretraining_spec)
    _write_json(output / "checkpoint_manifest.json", manifest)
    _write_json(output / "training_histories.json", jobs)
    _write_json(output / "audit.json", result["audit"])
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    _write_json(output / "result.json", result)
    return result
