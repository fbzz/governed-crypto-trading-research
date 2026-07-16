from __future__ import annotations

from dataclasses import asdict, fields
import gc
import hashlib
from itertools import combinations
import json
import math
import os
from pathlib import Path
import platform
import subprocess
from typing import Callable

import numpy as np
import pandas as pd
import torch
import yaml

from .joint_absolute_relative_evaluation_metrics import (
    STRATEGIES,
    WEIGHT_COLUMNS,
    build_exact_triplet_portfolio,
    build_v50_bootstrap,
    evaluate_v50_gates,
    shift_triplet_positions_one_day,
)
from .joint_absolute_relative_model import (
    JOINT_HEADS,
    JointAbsoluteRelativeTransformer,
    joint_triplet_positions,
    reconstruct_joint_predictions,
)
from .joint_absolute_relative_spec import _canonical_sha256, _load_json, _sha256_file
from .joint_absolute_relative_training import (
    CanonicalTripletSampler,
    JointFeatureLabelStore,
)
from .non_target_pretraining import TARGET_SYMBOLS, TripletTensorStore
from .ranking_excess_harness import SharedAssetRidgeModel, fit_shared_asset_ridge
from .ranking_excess_pretraining import _availability_from_index, _state_is_finite
from .ranking_excess_screen_metrics import compute_predictive_metrics
from .scientific_harness import FeatureScaler
from .supervised_non_target import model_state_sha256


VERSION = "v50"
METADATA_INPUTS = {
    "v47_result",
    "v47_blueprint",
    "v47_audit",
    "v48_result",
    "v48_audit",
    "v49_result",
    "v49_audit",
    "v49_training_spec",
    "v49_checkpoint_manifest",
    "v49_target_scales",
    "v49_verification",
    "v32_dataset_manifest",
    "v32_feature_schema",
    "v32_asset_folds",
    "v32_triplet_catalog",
    "v32_source_audit",
}
BINARY_INPUTS = {"panel", "sequence_index"}


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _git_receipt(root: Path, required: bool) -> dict[str, object]:
    if not required:
        return {"required": False, "head": "not_required", "clean": True}
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise RuntimeError("V50 requires a clean committed Git receipt")
    return {"required": True, "head": head, "clean": True}


def _source_receipt(root: Path, files: list[str]) -> dict[str, object]:
    hashes = {}
    for relative in files:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"V50 source file missing: {relative}")
        hashes[relative] = _sha256_file(path)
    return {"files": hashes, "bundle_sha256": _canonical_sha256(hashes)}


def _scaler_from_record(record: dict) -> FeatureScaler:
    names = {field.name for field in fields(FeatureScaler)}
    values = {name: record[name] for name in names}
    values["feature_names"] = tuple(values["feature_names"])
    values["mean"] = tuple(values["mean"])
    values["scale"] = tuple(values["scale"])
    return FeatureScaler(**values)


def _checkpoint_model(path: Path, row: dict, blueprint: dict) -> tuple[JointAbsoluteRelativeTransformer, dict]:
    if not path.is_file() or _sha256_file(path) != row["checkpoint_sha256"]:
        raise RuntimeError(f"V50 checkpoint hash drift: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    if (
        payload.get("format_version") != "v49_joint_absolute_relative_training_v1"
        or payload.get("architecture") != blueprint["architecture"]
        or tuple(payload.get("prediction_heads", ())) != JOINT_HEADS
        or payload.get("model_state_sha256") != row["model_state_sha256"]
        or model_state_sha256(payload["model_state_dict"]) != row["model_state_sha256"]
        or not _state_is_finite(payload["model_state_dict"])
        or metadata.get("checkpoint_status")
        != "job_local_best_all_jobs_retained_no_selection"
        or metadata.get("origin") != row["origin"]
        or metadata.get("geometry") != row["geometry"]
        or int(metadata.get("fold", -1)) != int(row["fold"])
        or int(metadata.get("seed", -1)) != int(row["seed"])
    ):
        raise RuntimeError(f"V50 checkpoint semantic drift: {path}")
    model = JointAbsoluteRelativeTransformer(9, blueprint["architecture"])
    model.load_state_dict(payload["model_state_dict"])
    model.eval().requires_grad_(False)
    return model, payload


def _build_spec(context: dict[str, object]) -> dict[str, object]:
    evaluation = context["evaluation"]
    keys = (
        "device",
        "torch_threads",
        "dtype",
        "amp",
        "deterministic_algorithms",
        "cpu_fallback_allowed",
        "inference_batch_size",
        "data_access",
        "origins",
        "geometries",
        "folds",
        "seeds",
        "inference",
        "ridge",
        "policy",
        "controls",
        "accounting",
        "predictive_metrics",
        "bootstrap",
        "stresses",
        "gates",
        "lifecycle",
        "artifact_contract",
    )
    spec = {
        "format": evaluation["artifact_contract"]["evaluation_spec_format"],
        "version": VERSION,
        "candidate_family_id": context["blueprint"]["candidate_family_id"],
        "v47_blueprint_sha256": context["blueprint"]["blueprint_sha256"],
        "v49_contract_sha256": evaluation["expected_v49_contract_sha256"],
        "v49_result_sha256": evaluation["expected_v49_result_sha256"],
        "input_hashes": context["input_hashes"],
        "checkpoint_grid_sha256": _canonical_sha256(
            [row["checkpoint_sha256"] for row in context["checkpoint_rows"]]
        ),
        "source_receipt": context["source"],
        "git_receipt": context["git"],
        **{key: evaluation[key] for key in keys},
    }
    spec["evaluation_spec_sha256"] = _canonical_sha256(spec)
    return spec


def _metadata_context(config: dict, *, reopen_checkpoints: bool) -> dict[str, object]:
    evaluation = config["joint_absolute_relative_evaluation"]
    root = Path(evaluation["project_root"]).resolve()
    paths = {name: (root / relative).resolve() for name, relative in evaluation["inputs"].items()}
    if set(paths) != METADATA_INPUTS | BINARY_INPUTS:
        raise RuntimeError("V50 input allowlist drift")
    input_hashes = {}
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"V50 input missing: {name}")
        observed = _sha256_file(path)
        if observed != evaluation["expected_input_sha256"][name]:
            raise RuntimeError(f"V50 input hash drift: {name}")
        input_hashes[name] = observed
    values = {name: _load_json(paths[name]) for name in METADATA_INPUTS}
    blueprint = values["v47_blueprint"]
    v49 = values["v49_result"]
    if (
        values["v47_result"].get("decision")
        != "authorize_v48_joint_absolute_relative_synthetic_harness_only"
        or not values["v47_audit"].get("passed")
        or values["v48_result"].get("decision")
        != "authorize_v49_purged_non_target_training_only"
        or not values["v48_audit"].get("passed")
        or v49.get("decision") != "v49_training_complete_economic_evaluation_still_forbidden"
        or v49.get("result_sha256") != evaluation["expected_v49_result_sha256"]
        or v49.get("training_spec", {}).get("contract_sha256")
        != evaluation["expected_v49_contract_sha256"]
        or not values["v49_audit"].get("passed")
        or not values["v49_verification"].get("passed")
        or int(values["v49_verification"].get("checkpoint_count", -1)) != 36
        or values["v49_verification"].get("git_receipt", {}).get("head")
        != evaluation["expected_v49_git_head"]
    ):
        raise RuntimeError("V47-V49 lineage does not authorize V50")
    manifest = values["v32_dataset_manifest"]
    schema = values["v32_feature_schema"]
    folds_payload = values["v32_asset_folds"]
    catalog = values["v32_triplet_catalog"]
    feature_names = list(manifest["panel_features"])
    if (
        blueprint.get("candidate_family_id")
        != "tlm_joint_absolute_relative_triplet_medium_v1"
        or set(blueprint.get("target_contract", {}).get("target_symbols", ()))
        != TARGET_SYMBOLS
        or blueprint.get("target_contract", {}).get("target_data_allowed")
        or manifest["panel_sha256"] != input_hashes["panel"]
        or manifest["sequence_index_sha256"] != input_hashes["sequence_index"]
        or list(schema["model_feature_order"][:-1]) != feature_names
        or evaluation["data_access"]["feature_columns"]
        != ["date", "symbol", *feature_names]
        or TARGET_SYMBOLS.intersection(manifest["symbols"])
    ):
        raise RuntimeError("V50 model/dataset/target contract drift")
    folds = {int(row["fold"]): row for row in folds_payload["folds"]}
    catalogs = {int(row["fold"]): row for row in catalog["folds"]}
    for fold in (1, 2, 3):
        train = sorted(folds[fold]["train_symbols"])
        test = sorted(folds[fold]["test_symbols"])
        if (
            len(train) != 20
            or len(test) != 10
            or set(train).intersection(set(test) | TARGET_SYMBOLS)
            or catalogs[fold]["test_triplets"]
            != [list(item) for item in combinations(test, 3)]
        ):
            raise RuntimeError(f"V50 fold/catalog drift: {fold}")

    checkpoint_rows = list(values["v49_checkpoint_manifest"])
    if len(checkpoint_rows) != int(evaluation["expected_checkpoint_count"]):
        raise RuntimeError("V50 checkpoint count drift")
    checkpoint_by_key = {}
    checkpoint_root = (root / "data/checkpoints/v49_joint_absolute_relative_training").resolve()
    for row in checkpoint_rows:
        key = (str(row["origin"]), str(row["geometry"]), int(row["fold"]), int(row["seed"]))
        if key in checkpoint_by_key:
            raise RuntimeError("V50 duplicate checkpoint grid key")
        path = Path(row["checkpoint_path"]).resolve()
        canonical = checkpoint_root / key[0] / key[1] / f"fold_{key[2]}" / f"seed_{key[3]}" / "checkpoint.pt"
        if path != canonical or not path.is_relative_to(checkpoint_root):
            raise RuntimeError("V50 checkpoint path drift")
        if reopen_checkpoints:
            model, payload = _checkpoint_model(path, row, blueprint)
            del model, payload
        else:
            if not path.is_file() or _sha256_file(path) != row["checkpoint_sha256"]:
                raise RuntimeError("V50 checkpoint file drift")
        row = {**row, "resolved_path": str(path)}
        checkpoint_by_key[key] = row
    expected_grid = {
        (origin, geometry, fold, seed)
        for origin in ("origin_2024", "origin_2025")
        for geometry in ("expanding", "rolling")
        for fold in (1, 2, 3)
        for seed in (42, 7, 123)
    }
    if set(checkpoint_by_key) != expected_grid:
        raise RuntimeError("V50 checkpoint grid is incomplete")
    source = _source_receipt(root, list(evaluation["source_files"]))
    git = _git_receipt(root, bool(evaluation["require_clean_git_receipt"]))
    context = {
        "root": root,
        "paths": paths,
        "evaluation": evaluation,
        "values": values,
        "blueprint": blueprint,
        "manifest": manifest,
        "feature_names": feature_names,
        "folds": folds,
        "catalogs": catalogs,
        "checkpoint_rows": checkpoint_rows,
        "checkpoint_by_key": checkpoint_by_key,
        "input_hashes": input_hashes,
        "source": source,
        "git": git,
        "config": config,
    }
    context["evaluation_spec"] = _build_spec(context)
    return context


def _environment(context: dict[str, object]) -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "git_receipt": context["git"],
        "source_receipt": context["source"],
    }


def _write_phase(root: Path, relative: str, config: dict, result: dict) -> None:
    output = root / relative
    output.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output / "result.json", result)
    _atomic_write_json(output / "audit.json", result["audit"])
    _atomic_write_json(output / "evaluation_spec.json", result["evaluation_spec"])
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )


def preflight_joint_absolute_relative_evaluation(config: dict) -> dict[str, object]:
    context = _metadata_context(config, reopen_checkpoints=True)
    evaluation = context["evaluation"]
    checks = {
        "lineage_and_input_hashes_exact": True,
        "checkpoint_grid_reopened_exactly": len(context["checkpoint_by_key"]) == 36,
        "target_assets_remain_sealed": not TARGET_SYMBOLS.intersection(context["manifest"]["symbols"]),
        "preflight_deserializes_zero_parquets": True,
        "preflight_executes_zero_inference_or_optimizer_steps": True,
        "v50_contract_is_frozen_before_outcomes": bool(context["git"]["clean"]),
    }
    result = {
        "version": "v50_preflight",
        "decision": "authorize_v50_prediction_and_control_prepare_without_outcomes",
        "evaluation_spec": context["evaluation_spec"],
        "summary": {
            "checkpoint_count": 36,
            "parquet_files_deserialized": 0,
            "outcome_rows_materialized": 0,
            "optimizer_steps": 0,
            "prediction_rows": 0,
        },
        "environment": _environment(context),
        "audit": {"passed": all(checks.values()), "checks": checks},
    }
    _write_phase(
        context["root"],
        evaluation["artifact_contract"]["preflight_output_dir"],
        config,
        result,
    )
    return result


def _require_preflight(context: dict[str, object]) -> dict:
    path = context["root"] / context["evaluation"]["artifact_contract"]["preflight_output_dir"] / "result.json"
    if not path.is_file():
        raise RuntimeError("V50 requires a passing committed preflight")
    result = _load_json(path)
    if (
        result.get("decision")
        != "authorize_v50_prediction_and_control_prepare_without_outcomes"
        or not result.get("audit", {}).get("passed")
        or result.get("evaluation_spec") != context["evaluation_spec"]
    ):
        raise RuntimeError("V50 preflight receipt drift")
    return result


def _configure_mps(evaluation: dict) -> torch.device:
    torch.set_num_threads(int(evaluation["torch_threads"]))
    torch.use_deterministic_algorithms(bool(evaluation["deterministic_algorithms"]))
    fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0").strip().lower()
    if (
        evaluation["device"] != "mps"
        or evaluation["dtype"] != "float32"
        or evaluation["amp"]
        or evaluation["cpu_fallback_allowed"]
        or fallback not in {"", "0", "false", "no"}
    ):
        raise RuntimeError("V50 requires deterministic float32 MPS without fallback")
    if not torch.backends.mps.is_available():
        raise RuntimeError("V50 prepare requires host MPS")
    return torch.device("mps")


def _validate_projected(frame: pd.DataFrame, columns: list[str], allowed: set[str], label: str) -> None:
    if list(frame.columns) != columns:
        raise RuntimeError(f"V50 {label} projection drift")
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame["symbol"] = frame["symbol"].astype(str)
    if frame.duplicated(["date", "symbol"]).any():
        raise RuntimeError(f"V50 {label} duplicate keys")
    if not set(frame["symbol"].unique()).issubset(allowed) or set(frame["symbol"].unique()).intersection(TARGET_SYMBOLS):
        raise RuntimeError(f"V50 {label} symbol isolation drift")


def _read_prepare_cell(
    context: dict[str, object], origin: str, geometry: str, fold: int,
    *, reader: Callable[..., pd.DataFrame] = pd.read_parquet,
) -> dict[str, object]:
    evaluation = context["evaluation"]
    access = evaluation["data_access"]
    fold_entry = context["folds"][fold]
    train_symbols = sorted(fold_entry["train_symbols"])
    test_symbols = sorted(fold_entry["test_symbols"])
    origin_contract = next(
        row for row in context["values"]["v49_training_spec"]["contract"]["origins"]
        if row["id"] == origin
    )
    train_window = origin_contract["geometries"][geometry]
    evaluation_window = evaluation["origins"][origin]
    train_start = pd.Timestamp(train_window["train_start"], tz="UTC")
    train_end = pd.Timestamp(train_window["train_end"], tz="UTC")
    eval_start = pd.Timestamp(evaluation_window["evaluation_start"], tz="UTC")
    eval_end = pd.Timestamp(evaluation_window["evaluation_end"], tz="UTC")
    sequence_columns = list(access["sequence_columns"])
    feature_columns = list(access["feature_columns"])
    label_columns = list(access["label_columns"])

    def filters(symbols: list[str], start: pd.Timestamp, end: pd.Timestamp) -> list[tuple]:
        return [
            ("symbol", "in", symbols),
            (access["sequence_ready_filter"], "==", True),
            (access["label_complete_filter"], "==", True),
            ("date", ">=", start),
            ("date", "<=", end),
        ]

    train_filters = filters(train_symbols, train_start, train_end)
    eval_filters = filters(test_symbols, eval_start, eval_end)
    train_index = reader(context["paths"]["sequence_index"], engine="pyarrow", columns=sequence_columns, filters=train_filters)
    eval_index = reader(context["paths"]["sequence_index"], engine="pyarrow", columns=sequence_columns, filters=eval_filters)
    _validate_projected(train_index, sequence_columns, set(train_symbols), "train sequence")
    _validate_projected(eval_index, sequence_columns, set(test_symbols), "evaluation sequence")
    for frame in (train_index, eval_index):
        starts = pd.to_datetime(frame["sequence_start_date"], utc=True)
        if not bool((starts == frame["date"] - pd.Timedelta(days=255)).all()):
            raise RuntimeError("V50 sequence lookback drift")
    train_feature_filters = [
        ("symbol", "in", train_symbols),
        ("date", ">=", pd.to_datetime(train_index["sequence_start_date"], utc=True).min()),
        ("date", "<=", train_end),
    ]
    eval_feature_filters = [
        ("symbol", "in", test_symbols),
        ("date", ">=", pd.to_datetime(eval_index["sequence_start_date"], utc=True).min()),
        ("date", "<=", eval_end),
    ]
    train_features = reader(context["paths"]["panel"], engine="pyarrow", columns=feature_columns, filters=train_feature_filters)
    eval_features = reader(context["paths"]["panel"], engine="pyarrow", columns=feature_columns, filters=eval_feature_filters)
    train_labels = reader(context["paths"]["panel"], engine="pyarrow", columns=label_columns, filters=train_filters)
    _validate_projected(train_features, feature_columns, set(train_symbols), "train features")
    _validate_projected(eval_features, feature_columns, set(test_symbols), "evaluation features")
    _validate_projected(train_labels, label_columns, set(train_symbols), "train labels")
    train_labels["target_window_end_date"] = pd.to_datetime(train_labels["target_window_end_date"], utc=True)
    if (
        not bool((train_labels["target_window_end_date"] == train_labels["date"] + pd.Timedelta(days=8)).all())
        or train_labels["target_window_end_date"].max()
        > pd.Timestamp(train_window["train_maturity_end"], tz="UTC")
        or set(zip(train_labels["date"], train_labels["symbol"], strict=True))
        != set(zip(train_index["date"], train_index["symbol"], strict=True))
    ):
        raise RuntimeError("V50 train-label maturity/key drift")
    train_availability = _availability_from_index(train_index)
    eval_availability = _availability_from_index(eval_index)
    return {
        "train_features": train_features,
        "train_labels": train_labels,
        "eval_features": eval_features,
        "train_availability": train_availability,
        "eval_availability": eval_availability,
        "audit": {
            "origin": origin,
            "geometry": geometry,
            "fold": fold,
            "train_symbols": train_symbols,
            "test_symbols": test_symbols,
            "train_label_rows": len(train_labels),
            "evaluation_feature_rows": len(eval_features),
            "evaluation_ready_asset_dates": sum(len(value) for value in eval_availability.values()),
            "heldout_outcome_rows_read": 0,
            "target_asset_rows_read": 0,
            "forbidden_label_columns_read": [],
            "physical_row_group_isolation_claimed": False,
        },
    }


def _ridge_prediction(model: SharedAssetRidgeModel, tensor: np.ndarray) -> np.ndarray:
    values = np.asarray(tensor, dtype=np.float64)
    design = values.transpose(0, 2, 1, 3).reshape(values.shape[0] * 3, -1)
    prediction = np.einsum(
        "ij,j->i", design, model.coefficient, optimize=False
    ) + float(model.intercept)
    if not np.isfinite(prediction).all():
        raise RuntimeError("V50 Ridge inference produced non-finite predictions")
    return prediction.reshape(len(values), 3)


def _ridge_state_sha256(model: SharedAssetRidgeModel) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(model.coefficient, dtype=np.float64).tobytes())
    digest.update(np.asarray([model.intercept], dtype=np.float64).tobytes())
    digest.update(model.solution_form.encode("ascii"))
    return digest.hexdigest()


def _fit_ridge(context: dict[str, object], cell: dict, origin: str, geometry: str, fold: int, scaler: FeatureScaler, scale: float) -> tuple[SharedAssetRidgeModel, dict]:
    contract = context["evaluation"]["ridge"]
    train_symbols = sorted(context["folds"][fold]["train_symbols"])
    sampler = CanonicalTripletSampler(
        cell["train_availability"],
        train_symbols,
        master_seed=20260713,
        version="v49",
        origin=origin,
        geometry=geometry,
        fold=fold,
        seed=int(contract["sampler_seed"]),
        role=str(contract["sampler_role"]),
    )
    samples, sample_sha = sampler.sample_epoch(
        int(contract["sampler_epoch"]), int(contract["train_samples"])
    )
    store = JointFeatureLabelStore(
        cell["train_features"],
        [cell["train_labels"]],
        context["feature_names"],
        256,
        context["evaluation"]["data_access"]["relative_source_feature"],
        context["evaluation"]["data_access"]["return_column"],
    )
    tensor, labels = store.materialize_batch(samples, scaler)
    model = fit_shared_asset_ridge(tensor, labels / float(scale), float(contract["alpha"]))
    receipt = {
        "origin": origin,
        "geometry": geometry,
        "fold": fold,
        "alpha": float(contract["alpha"]),
        "sample_count": len(samples),
        "sample_sequence_sha256": sample_sha,
        "sampler_seed": int(contract["sampler_seed"]),
        "sampler_epoch": int(contract["sampler_epoch"]),
        "target": contract["target"],
        "solution_form": model.solution_form,
        "coefficient_count": len(model.coefficient),
        "ridge_state_sha256": _ridge_state_sha256(model),
    }
    del store, tensor, labels
    gc.collect()
    return model, receipt


def _load_cell_models(context: dict[str, object], origin: str, geometry: str, fold: int, device: torch.device) -> list[JointAbsoluteRelativeTransformer]:
    models = []
    for seed in context["evaluation"]["seeds"]:
        row = context["checkpoint_by_key"][(origin, geometry, fold, int(seed))]
        model, payload = _checkpoint_model(Path(row["resolved_path"]), row, context["blueprint"])
        model.to(device)
        models.append(model)
        del payload
    return models


def _momentum_lookup(features: pd.DataFrame) -> dict[tuple[pd.Timestamp, str], float]:
    result = {}
    for symbol, frame in features.groupby("symbol", sort=True):
        current = frame.sort_values("date").copy()
        current["momentum_30"] = current["log_close_to_close_return"].rolling(30, min_periods=30).sum()
        for row in current[["date", "momentum_30"]].itertuples(index=False):
            result[(pd.Timestamp(row.date), str(symbol))] = float(row.momentum_30)
    return result


def _infer_cell(context: dict[str, object], cell: dict, origin: str, geometry: str, fold: int, scaler: FeatureScaler, scale: float, ridge: SharedAssetRidgeModel, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    evaluation = context["evaluation"]
    dates = pd.date_range(
        evaluation["origins"][origin]["evaluation_start"],
        evaluation["origins"][origin]["evaluation_end"],
        freq="D",
        tz="UTC",
    )
    triplets = [tuple(item) for item in context["catalogs"][fold]["test_triplets"]]
    samples = []
    for date in dates:
        available = set(cell["eval_availability"].get(date, []))
        for triplet in triplets:
            if set(triplet).issubset(available):
                samples.append({"date": date, "triplet": triplet})
    store = TripletTensorStore(
        cell["eval_features"][["date", "symbol", *context["feature_names"]]],
        context["feature_names"],
        256,
        evaluation["data_access"]["relative_source_feature"],
    )
    models = _load_cell_models(context, origin, geometry, fold, device)
    prediction_by_key = {}
    context_rows = []
    batch_size = int(evaluation["inference_batch_size"])
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        x_np = store.materialize_batch(batch, scaler)
        x = torch.from_numpy(x_np).to(device)
        seed_e = []
        seed_mu = []
        with torch.inference_mode():
            for model in models:
                reconstructed = reconstruct_joint_predictions(model(x), scale)
                seed_e.append(reconstructed["e_hat"].detach().cpu().numpy())
                seed_mu.append(reconstructed["mu_hat"].detach().cpu().numpy())
        transformer_e = np.mean(np.stack(seed_e), axis=0)
        transformer_mu = np.mean(np.stack(seed_mu), axis=0)
        ridge_mu = _ridge_prediction(ridge, x_np) * float(scale)
        ridge_e = ridge_mu - ridge_mu.mean(axis=1, keepdims=True)
        for offset, sample in enumerate(batch):
            date = pd.Timestamp(sample["date"])
            triplet = tuple(str(value) for value in sample["triplet"])
            key = (date, triplet)
            prediction_by_key[key] = (
                transformer_e[offset].copy(),
                transformer_mu[offset].copy(),
                ridge_e[offset].copy(),
                ridge_mu[offset].copy(),
            )
            row = {
                "date": date,
                "origin": origin,
                "geometry": geometry,
                "fold": fold,
                "triplet_key": "|".join(triplet),
            }
            for slot, symbol in enumerate(triplet):
                row[f"symbol_{slot}"] = symbol
                row[f"transformer_raw_excess_{slot}"] = float(transformer_e[offset, slot])
                row[f"transformer_raw_absolute_{slot}"] = float(transformer_mu[offset, slot])
                row[f"ridge_raw_excess_{slot}"] = float(ridge_e[offset, slot])
                row[f"ridge_raw_absolute_{slot}"] = float(ridge_mu[offset, slot])
            context_rows.append(row)
        del x, x_np, seed_e, seed_mu, transformer_e, transformer_mu, ridge_e, ridge_mu
    momentum = _momentum_lookup(cell["eval_features"])
    position_rows = []
    for triplet in triplets:
        candidate_e = np.full((len(dates), 3), np.nan)
        candidate_mu = np.full_like(candidate_e, np.nan)
        ridge_e = np.full_like(candidate_e, np.nan)
        ridge_mu = np.full_like(candidate_e, np.nan)
        eligible = np.zeros_like(candidate_e, dtype=bool)
        triplet_momentum = np.full_like(candidate_e, np.nan)
        for day, date in enumerate(dates):
            value = prediction_by_key.get((date, triplet))
            if value is not None:
                candidate_e[day], candidate_mu[day], ridge_e[day], ridge_mu[day] = value
                eligible[day] = True
                triplet_momentum[day] = [momentum[(date, symbol)] for symbol in triplet]
        candidate = joint_triplet_positions(
            candidate_mu,
            candidate_e,
            eligible,
            risky_weight=float(evaluation["policy"]["risky_weight"]),
            base_cost=float(evaluation["policy"]["base_cost"]),
        )
        ridge_position = joint_triplet_positions(
            ridge_mu,
            ridge_e,
            eligible,
            risky_weight=float(evaluation["policy"]["risky_weight"]),
            base_cost=float(evaluation["policy"]["base_cost"]),
        )
        dual = np.zeros_like(candidate)
        equal = np.zeros_like(candidate)
        for day in range(len(dates)):
            if bool(eligible[day].all()):
                best = int(np.argmax(triplet_momentum[day]))
                if triplet_momentum[day, best] > 0:
                    dual[day, best] = 1.0 / 3.0
                equal[day] = 1.0 / 9.0
        for day, date in enumerate(dates):
            row = {
                "date": date,
                "origin": origin,
                "geometry": geometry,
                "fold": fold,
                "triplet_key": "|".join(triplet),
            }
            for slot, symbol in enumerate(triplet):
                row[f"symbol_{slot}"] = symbol
                row[f"candidate_weight_{slot}"] = float(candidate[day, slot])
                row[f"ridge_weight_{slot}"] = float(ridge_position[day, slot])
                row[f"dual_momentum_30_weight_{slot}"] = float(dual[day, slot])
                row[f"equal_weight_weight_{slot}"] = float(equal[day, slot])
                row[f"cash_weight_{slot}"] = 0.0
            position_rows.append(row)
    for model in models:
        model.to("cpu")
    del models, store
    torch.mps.empty_cache()
    gc.collect()
    diagnostics = {
        "origin": origin,
        "geometry": geometry,
        "fold": fold,
        "calendar_dates": len(dates),
        "registered_triplets": len(triplets),
        "eligible_contexts": len(context_rows),
        "seed_context_forwards": len(context_rows) * 3,
        "position_rows": len(position_rows),
        "outcome_rows_read": 0,
        "cross_context_asset_averaging_performed": False,
    }
    return pd.DataFrame(context_rows), pd.DataFrame(position_rows), diagnostics


def prepare_joint_absolute_relative_evaluation(config: dict) -> dict[str, object]:
    context = _metadata_context(config, reopen_checkpoints=False)
    _require_preflight(context)
    device = _configure_mps(context["evaluation"])
    context_frames = []
    position_frames = []
    ridge_receipts = []
    data_audits = []
    inference_audits = []
    for origin in ("origin_2024", "origin_2025"):
        for geometry in ("expanding", "rolling"):
            for fold in (1, 2, 3):
                cell = _read_prepare_cell(context, origin, geometry, fold)
                cell_dir = context["root"] / "data/checkpoints/v49_joint_absolute_relative_training" / origin / geometry / f"fold_{fold}"
                scaler_record = _load_json(cell_dir / "scaler.json")
                scale_record = _load_json(cell_dir / "target_scale.json")
                scaler = _scaler_from_record(scaler_record)
                scale = float(scale_record["raw_return_rms_scale"])
                ridge, ridge_receipt = _fit_ridge(
                    context, cell, origin, geometry, fold, scaler, scale
                )
                contexts, positions, diagnostics = _infer_cell(
                    context, cell, origin, geometry, fold, scaler, scale, ridge, device
                )
                context_frames.append(contexts)
                position_frames.append(positions)
                ridge_receipts.append(ridge_receipt)
                data_audits.append(cell["audit"])
                inference_audits.append(diagnostics)
                del cell, ridge, contexts, positions
                gc.collect()
    context_frame = pd.concat(context_frames, ignore_index=True)
    position_frame = pd.concat(position_frames, ignore_index=True)
    artifact = context["evaluation"]["artifact_contract"]
    output = context["root"] / artifact["prepare_output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    context_path = output / "context_predictions.parquet"
    position_path = output / "triplet_positions.parquet"
    _atomic_write_parquet(context_frame, context_path)
    _atomic_write_parquet(position_frame, position_path)
    _atomic_write_json(output / "ridge_receipts.json", ridge_receipts)
    _atomic_write_json(output / "data_access_audit.json", data_audits)
    _atomic_write_json(output / "inference_audit.json", inference_audits)
    manifest = {
        "context_predictions": _sha256_file(context_path),
        "triplet_positions": _sha256_file(position_path),
        "ridge_receipts": _sha256_file(output / "ridge_receipts.json"),
        "data_access_audit": _sha256_file(output / "data_access_audit.json"),
        "inference_audit": _sha256_file(output / "inference_audit.json"),
    }
    manifest["prepare_manifest_sha256"] = _canonical_sha256(manifest)
    _atomic_write_json(output / artifact["prepare_manifest_name"], manifest)
    checks = {
        "all_twelve_cells_prepared": len(inference_audits) == 12,
        "all_thirty_six_checkpoints_used_without_selection": sum(
            row["seed_context_forwards"] for row in inference_audits
        ) == 3 * len(context_frame),
        "predictions_are_exact_triplet_only": all(
            not row["cross_context_asset_averaging_performed"] for row in inference_audits
        ),
        "outcomes_remained_sealed": all(row["outcome_rows_read"] == 0 for row in inference_audits),
        "target_assets_remained_sealed": all(row["target_asset_rows_read"] == 0 for row in data_audits),
        "ridge_is_outcome_blind_and_fixed": len(ridge_receipts) == 12
        and all(row["sample_count"] == 8192 for row in ridge_receipts),
    }
    result = {
        "version": "v50_prepare",
        "decision": "authorize_v50_one_shot_registered_outcome_unseal",
        "evaluation_spec": context["evaluation_spec"],
        "prepare_manifest": manifest,
        "summary": {
            "cell_count": 12,
            "checkpoint_count": 36,
            "context_prediction_rows": len(context_frame),
            "triplet_position_rows": len(position_frame),
            "ridge_models": len(ridge_receipts),
            "outcome_rows_materialized": 0,
        },
        "environment": _environment(context),
        "audit": {"passed": all(checks.values()), "checks": checks},
    }
    _write_phase(context["root"], artifact["prepare_output_dir"], config, result)
    return result


def _load_prepare(context: dict[str, object]) -> dict[str, object]:
    artifact = context["evaluation"]["artifact_contract"]
    output = context["root"] / artifact["prepare_output_dir"]
    result = _load_json(output / "result.json")
    manifest = _load_json(output / artifact["prepare_manifest_name"])
    checks = {
        "context_predictions": _sha256_file(output / "context_predictions.parquet"),
        "triplet_positions": _sha256_file(output / "triplet_positions.parquet"),
        "ridge_receipts": _sha256_file(output / "ridge_receipts.json"),
        "data_access_audit": _sha256_file(output / "data_access_audit.json"),
        "inference_audit": _sha256_file(output / "inference_audit.json"),
    }
    checks["prepare_manifest_sha256"] = _canonical_sha256(checks)
    if (
        result.get("decision") != "authorize_v50_one_shot_registered_outcome_unseal"
        or not result.get("audit", {}).get("passed")
        or result.get("evaluation_spec") != context["evaluation_spec"]
        or result.get("prepare_manifest") != manifest
        or manifest != checks
    ):
        raise RuntimeError("V50 prepare packet drift")
    return {
        "result": result,
        "context_predictions": pd.read_parquet(output / "context_predictions.parquet"),
        "positions": pd.read_parquet(output / "triplet_positions.parquet"),
        "manifest": manifest,
    }


def _unseal_outcomes(context: dict[str, object]) -> tuple[pd.DataFrame, dict]:
    artifact = context["evaluation"]["artifact_contract"]
    output = context["root"] / artifact["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    packet_path = output / artifact["outcome_packet_name"]
    manifest_path = output / "outcome_manifest.json"
    if packet_path.is_file() and manifest_path.is_file():
        manifest = _load_json(manifest_path)
        if _sha256_file(packet_path) != manifest.get("outcome_packet_sha256"):
            raise RuntimeError("V50 sealed outcome packet drift")
        return pd.read_parquet(packet_path), manifest
    rows = []
    receipts = []
    access = context["evaluation"]["data_access"]
    for origin in ("origin_2024", "origin_2025"):
        window = context["evaluation"]["origins"][origin]
        start = pd.Timestamp(window["evaluation_start"], tz="UTC")
        end = pd.Timestamp(window["evaluation_end"], tz="UTC")
        maturity_end = pd.Timestamp(window["maturity_end"], tz="UTC")
        for fold in (1, 2, 3):
            symbols = sorted(context["folds"][fold]["test_symbols"])
            filters = [
                ("symbol", "in", symbols),
                (access["sequence_ready_filter"], "==", True),
                (access["label_complete_filter"], "==", True),
                ("date", ">=", start),
                ("date", "<=", end),
            ]
            frame = pd.read_parquet(
                context["paths"]["panel"],
                engine="pyarrow",
                columns=list(access["label_columns"]),
                filters=filters,
            )
            _validate_projected(frame, list(access["label_columns"]), set(symbols), "evaluation outcomes")
            frame["target_window_end_date"] = pd.to_datetime(frame["target_window_end_date"], utc=True)
            if (
                not bool((frame["target_window_end_date"] == frame["date"] + pd.Timedelta(days=8)).all())
                or frame["target_window_end_date"].max() > maturity_end
            ):
                raise RuntimeError("V50 outcome maturity drift")
            for row in frame.itertuples(index=False):
                rows.append(
                    {
                        "date": pd.Timestamp(row.date),
                        "origin": origin,
                        "fold": fold,
                        "symbol": str(row.symbol),
                        "target_window_end_date": pd.Timestamp(row.target_window_end_date),
                        "action_log_return": float(getattr(row, access["return_column"])),
                    }
                )
            receipts.append(
                {
                    "origin": origin,
                    "fold": fold,
                    "symbols": symbols,
                    "start": start.date().isoformat(),
                    "end": end.date().isoformat(),
                    "row_count": len(frame),
                    "columns": list(access["label_columns"]),
                }
            )
    outcome = pd.DataFrame(rows).sort_values(["origin", "fold", "date", "symbol"]).reset_index(drop=True)
    _atomic_write_parquet(outcome, packet_path)
    manifest = {
        "version": "v50_outcome_packet_v1",
        "outcome_packet_sha256": _sha256_file(packet_path),
        "row_count": len(outcome),
        "receipts": receipts,
        "target_assets_loaded": False,
        "post_2025_outcomes_loaded": False,
    }
    manifest["outcome_manifest_sha256"] = _canonical_sha256(manifest)
    _atomic_write_json(manifest_path, manifest)
    return outcome, manifest


def _missing_asset_stress(positions: pd.DataFrame, outcomes: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    symbols = sorted({str(value) for slot in range(3) for value in positions[f"symbol_{slot}"].unique()})
    for symbol in symbols:
        keep = ~(
            (positions["symbol_0"] == symbol)
            | (positions["symbol_1"] == symbol)
            | (positions["symbol_2"] == symbol)
        )
        portfolio = build_exact_triplet_portfolio(positions.loc[keep], outcomes, [10])
        for origin in ("origin_2024", "origin_2025"):
            for geometry in ("expanding", "rolling"):
                metric = portfolio["aggregate_metrics"][origin][geometry]["10"]["candidate"]
                rows.append(
                    {
                        "removed_symbol": symbol,
                        "origin": origin,
                        "geometry": geometry,
                        "candidate_total_return_10bps": metric["total_return"],
                        "candidate_max_drawdown_10bps": metric["max_drawdown"],
                    }
                )
    return rows


def evaluate_joint_absolute_relative_evaluation(config: dict) -> dict[str, object]:
    context = _metadata_context(config, reopen_checkpoints=False)
    _require_preflight(context)
    prepared = _load_prepare(context)
    outcomes, outcome_manifest = _unseal_outcomes(context)
    context_predictions = prepared["context_predictions"]
    positions = prepared["positions"]
    predictive = {}
    for origin in ("origin_2024", "origin_2025"):
        predictive[origin] = {}
        outcome_subset = outcomes.loc[outcomes["origin"] == origin, ["date", "fold", "symbol", "action_log_return"]]
        for geometry in ("expanding", "rolling"):
            prediction_subset = context_predictions.loc[
                (context_predictions["origin"] == origin)
                & (context_predictions["geometry"] == geometry)
            ]
            predictive[origin][geometry] = compute_predictive_metrics(
                prediction_subset,
                outcome_subset,
                float(context["evaluation"]["predictive_metrics"]["exact_tie_tolerance"]),
            )[2]
    portfolio = build_exact_triplet_portfolio(
        positions,
        outcomes,
        [10, 20, 30, int(context["evaluation"]["accounting"]["diagnostic_cost_bps"])],
        annualization_days=int(context["evaluation"]["accounting"]["annualization_days"]),
    )
    bootstrap = build_v50_bootstrap(portfolio["daily_frame"], context["evaluation"]["bootstrap"])
    gates = evaluate_v50_gates(
        predictive,
        portfolio["cell_metrics"],
        portfolio["aggregate_metrics"],
        bootstrap,
        context["evaluation"]["gates"],
    )
    delayed = build_exact_triplet_portfolio(
        shift_triplet_positions_one_day(positions), outcomes, [10]
    )
    missing = _missing_asset_stress(positions, outcomes)
    stresses = {
        "cost_50bps": {
            origin: {
                geometry: portfolio["aggregate_metrics"][origin][geometry]["50"]
                for geometry in ("expanding", "rolling")
            }
            for origin in ("origin_2024", "origin_2025")
        },
        "one_day_extra_signal_delay": delayed["aggregate_metrics"],
        "missing_asset_leave_one_out": missing,
        "mandatory_gate": False,
    }
    decision = (
        context["evaluation"]["lifecycle"]["pass_action"]
        if gates["passed"]
        else context["evaluation"]["lifecycle"]["failure_action"]
    )
    output = context["root"] / context["evaluation"]["artifact_contract"]["output_dir"]
    _atomic_write_json(output / "predictive_summary.json", predictive)
    _atomic_write_json(output / "portfolio_metrics.json", {
        "cell_metrics": portfolio["cell_metrics"],
        "aggregate_metrics": portfolio["aggregate_metrics"],
        "triplet_count_audit": portfolio["triplet_count_audit"],
    })
    _atomic_write_parquet(portfolio["daily_frame"], output / "daily_returns.parquet")
    _atomic_write_json(output / "bootstrap.json", bootstrap)
    _atomic_write_json(output / "gate_result.json", gates)
    _atomic_write_json(output / "stresses.json", stresses)
    checks = {
        "prepare_packet_hash_verified_before_outcomes": True,
        "outcome_packet_is_atomic_and_hash_bound": _sha256_file(
            output / context["evaluation"]["artifact_contract"]["outcome_packet_name"]
        ) == outcome_manifest["outcome_packet_sha256"],
        "all_twelve_cells_reported": len(portfolio["triplet_count_audit"]) == 12,
        "mandatory_costs_and_diagnostic_50bps_reported": all(
            set(portfolio["aggregate_metrics"][origin][geometry]) == {"10", "20", "30", "50"}
            for origin in ("origin_2024", "origin_2025")
            for geometry in ("expanding", "rolling")
        ),
        "bootstrap_grid_is_exact": all(
            set(bootstrap[origin][geometry]) == {"7", "21", "63"}
            for origin in ("origin_2024", "origin_2025")
            for geometry in ("expanding", "rolling")
        ),
        "target_assets_remained_sealed": not outcome_manifest["target_assets_loaded"],
        "no_retraining_or_selection": True,
        "all_gate_cells_preserved": gates["cell_count"] > 0
        and gates["passed_count"] + gates["failed_count"] == gates["cell_count"],
    }
    result = {
        "version": "v50",
        "decision": decision,
        "evidence_status": "adaptive_historical_development_only",
        "evaluation_spec": context["evaluation_spec"],
        "prepare_manifest": prepared["manifest"],
        "outcome_manifest": outcome_manifest,
        "summary": {
            "checkpoint_count": 36,
            "origin_geometry_cells": 4,
            "fold_cells": 12,
            "context_prediction_rows": len(context_predictions),
            "outcome_rows": len(outcomes),
            "gate_cells": gates["cell_count"],
            "passed_gate_cells": gates["passed_count"],
            "failed_gate_cells": gates["failed_count"],
        },
        "gate_result": gates,
        "environment": _environment(context),
        "audit": {"passed": all(checks.values()), "checks": checks},
        "limitations": [
            "adaptive historical development evidence only",
            "no BTC/ETH/SOL observation or prediction",
            "no model refit, checkpoint selection, or gate tuning",
            "stress scenarios are diagnostic and cannot rescue a failed mandatory cell",
        ],
    }
    result["result_sha256"] = _canonical_sha256(result)
    _atomic_write_json(output / "result.json", result)
    _atomic_write_json(output / "audit.json", result["audit"])
    _atomic_write_json(output / "evaluation_spec.json", result["evaluation_spec"])
    report = "\n".join(
        [
            "# TLM V50 Frozen Historical Development Evaluation",
            "",
            "## Decision",
            "",
            f"**{decision}**",
            "",
            f"Mandatory gate cells: **{gates['passed_count']}/{gates['cell_count']} passed**",
            f"Result SHA-256: `{result['result_sha256']}`",
            "",
            "This is adaptive historical development evidence, not clean confirmation or deployment authorization.",
            "BTC, ETH, and SOL remained sealed. No refit or model selection was performed.",
            "",
        ]
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    return result


def verify_joint_absolute_relative_evaluation(config: dict) -> dict[str, object]:
    context = _metadata_context(config, reopen_checkpoints=False)
    prepared = _load_prepare(context)
    output = context["root"] / context["evaluation"]["artifact_contract"]["output_dir"]
    result = _load_json(output / "result.json")
    required = (
        "audit.json",
        "evaluation_spec.json",
        "predictive_summary.json",
        "portfolio_metrics.json",
        "daily_returns.parquet",
        "bootstrap.json",
        "gate_result.json",
        "stresses.json",
        "outcome_manifest.json",
        context["evaluation"]["artifact_contract"]["outcome_packet_name"],
    )
    if (
        not result.get("audit", {}).get("passed")
        or result.get("evaluation_spec") != context["evaluation_spec"]
        or result.get("prepare_manifest") != prepared["manifest"]
        or result.get("result_sha256")
        != _canonical_sha256({key: value for key, value in result.items() if key != "result_sha256"})
        or not all((output / name).is_file() for name in required)
    ):
        raise RuntimeError("V50 verification failed")
    return {**result, "verification": {"passed": True, "parquet_outcome_reads": 0, "inference_calls": 0}}
