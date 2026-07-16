"""Frozen V59 outcome-blind evaluation preparation.

This module performs inference and freezes behavior only.  It contains no
outcome-unseal, PnL, performance-metric, bootstrap, or gate-evaluation path.
"""

from __future__ import annotations

from copy import deepcopy
import gc
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.linear_model import Ridge
import sklearn
import torch

from .core.artifacts import canonical_sha256, file_sha256
from .state_conditioned_multi_horizon_evaluation_artifacts import (
    PREPARE_DECISION,
    PREPARE_SCHEMA,
    REQUIRED_PREPARE_FILES,
    V59_PHASE_CONTRACT_CANONICAL_SHA256,
    V59_PHASE_FILE_SHA256,
    V59_SOURCE_FILES,
    V59PrepareError,
    _registered_access_receipts,
    build_prepare_manifest,
    load_json,
    load_live_v59_contract,
    minimum_free_space,
    process_lock,
    registered_projection,
    require,
    resolve_repo_path,
    source_receipt,
    verify_input_files,
    verify_prepared_values,
    verify_prepare_packet,
    with_self_hash,
    write_json,
)
from .state_conditioned_multi_horizon_evaluation_data import (
    CellData,
    EvaluationCell,
    ScalerValues,
    build_evaluation_cell,
    classify_development_samples,
    day_text,
    materialize_development_batch,
    read_cell_data,
    ridge_training_arrays,
    scaler_from_wrapper,
)
from .state_conditioned_multi_horizon_training_data import TARGET_SYMBOLS
from .state_conditioned_multi_horizon_training_engine import (
    V58CheckpointContext,
    build_v58_adamw,
    configure_v58_runtime,
    instantiate_v58_model,
    load_v58_checkpoint,
    model_state_sha256,
)


VERSION = "v59"
SEEDS = (42, 7, 123)
HORIZONS = (1, 3, 7)
QUANTILES = (20, 50, 80)
CONTROLS = (
    "cash",
    "weekly_dual_momentum_30",
    "weekly_equal_weight_total_gross_one_third",
    "shared_linear_h7_q50_with_train_residual_q20",
)
INPUT_NAMES = {
    "blueprint",
    "dataset_spec",
    "dataset_manifest",
    "label_schema",
    "asset_folds",
    "triplet_catalog",
    "v58_result",
    "v58_training_spec",
    "checkpoint_manifest",
    "scaler_manifest",
    "panel",
    "labels",
    "sequence_roles",
}
EXPECTED_INPUT_PATHS = {
    "blueprint": "artifacts/v55_state_conditioned_multi_horizon_spec/blueprint.json",
    "dataset_spec": "artifacts/v57_non_target_multi_horizon_dataset/dataset_spec.json",
    "dataset_manifest": "artifacts/v57_non_target_multi_horizon_dataset/dataset_manifest.json",
    "label_schema": "artifacts/v57_non_target_multi_horizon_dataset/label_schema.json",
    "asset_folds": "artifacts/v32_selected_universe_dataset/asset_folds.json",
    "triplet_catalog": "artifacts/v32_selected_universe_dataset/triplet_catalog.json",
    "v58_result": "artifacts/v58_state_conditioned_multi_horizon_training/result.json",
    "v58_training_spec": "artifacts/v58_state_conditioned_multi_horizon_training/training_spec.json",
    "checkpoint_manifest": "artifacts/v58_state_conditioned_multi_horizon_training/checkpoint_manifest.json",
    "scaler_manifest": "artifacts/v58_state_conditioned_multi_horizon_training/scaler_manifest.json",
    "panel": "data/processed/selected_universe_panel_v32.parquet",
    "labels": "data/processed/state_conditioned_multi_horizon_labels_v57.parquet",
    "sequence_roles": "data/processed/state_conditioned_multi_horizon_sequence_roles_v57.parquet",
}


def prediction_value_columns() -> list[str]:
    result = [
        f"seed_{seed}_h{horizon}_q{quantile}"
        for seed in SEEDS
        for horizon in HORIZONS
        for quantile in QUANTILES
    ]
    result.extend(
        f"ensemble_h{horizon}_q{quantile}"
        for horizon in HORIZONS
        for quantile in QUANTILES
    )
    return result


PREDICTION_COLUMNS = (
    "origin",
    "geometry",
    "fold",
    "triplet_key",
    "date",
    "asset_slot",
    "symbol",
    *prediction_value_columns(),
    "linear_h7_q50",
    "linear_h7_q20",
)
POSITION_COLUMNS = (
    "origin",
    "geometry",
    "fold",
    "triplet_key",
    "date",
    "symbol_0",
    "symbol_1",
    "symbol_2",
    "available",
    "decision",
    "forced_cash",
    "final_liquidation",
    "action",
    "selected_symbol",
    "weight_0",
    "weight_1",
    "weight_2",
    "post_event_weight_0",
    "post_event_weight_1",
    "post_event_weight_2",
    "base_turnover",
    "final_liquidation_turnover",
    "turnover",
)
CONTROL_POSITION_COLUMNS = (*POSITION_COLUMNS[:5], "control", *POSITION_COLUMNS[5:])


class ParquetAccumulator:
    def __init__(self, path: Path, columns: Sequence[str]) -> None:
        self.path = path
        self.columns = tuple(columns)
        self.writer: pq.ParquetWriter | None = None
        self.schema: pa.Schema | None = None
        self.rows = 0

    def write(self, frame: pd.DataFrame) -> None:
        if list(frame.columns) != list(self.columns):
            raise V59PrepareError(f"Parquet schema column drift: {self.path.name}")
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.schema = table.schema
            self.writer = pq.ParquetWriter(
                self.path,
                table.schema,
                compression="zstd",
                use_dictionary=True,
            )
        elif table.schema != self.schema:
            raise V59PrepareError(f"Parquet Arrow schema drift: {self.path.name}")
        self.writer.write_table(table)
        self.rows += len(frame)

    def close(self) -> None:
        if self.writer is None:
            raise V59PrepareError(f"no rows written to {self.path.name}")
        self.writer.close()
        self.writer = None

    def metadata(self) -> dict[str, Any]:
        if self.schema is None or self.writer is not None:
            raise V59PrepareError(f"Parquet writer not finalized: {self.path.name}")
        schema_records = [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in self.schema
        ]
        return {
            "row_count": self.rows,
            "columns": list(self.columns),
            "arrow_schema": schema_records,
            "arrow_schema_sha256": canonical_sha256(schema_records),
        }

    def __enter__(self) -> "ParquetAccumulator":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def _evaluation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    require(
        config.get("_invoked_config_path")
        == "configs/v59_state_conditioned_multi_horizon_evaluation.yaml",
        "V59 must be invoked with the exact authorized --config path",
    )
    value = config.get("state_conditioned_multi_horizon_evaluation")
    require(isinstance(value, dict), "config lacks state_conditioned_multi_horizon_evaluation")
    required = {
        "version",
        "project_root",
        "config_path",
        "research_state",
        "experiment_contract",
        "phase_contract",
        "inputs",
        "source_receipt_files",
        "output_dir",
        "require_clean_git",
    }
    require(set(value) == required, "V59 config field set drift")
    require(value["version"] == VERSION, "V59 config version drift")
    require(value["project_root"] == ".", "V59 project_root drift")
    require(value["require_clean_git"] is True, "V59 clean-Git requirement drift")
    inputs = value["inputs"]
    require(isinstance(inputs, dict) and set(inputs) == INPUT_NAMES, "V59 input alias set drift")
    require(inputs == EXPECTED_INPUT_PATHS, "V59 input alias/path binding drift")
    require(
        value["config_path"]
        == "configs/v59_state_conditioned_multi_horizon_evaluation.yaml"
        and value["research_state"] == "research/current.yaml"
        and value["experiment_contract"] == "research/experiments/v058.yaml"
        and value["phase_contract"] == "research/phase_contracts/v059.yaml",
        "V59 config governance path drift",
    )
    require(
        tuple(value["source_receipt_files"]) == V59_SOURCE_FILES,
        "V59 source receipt file set/order drift",
    )
    return dict(value)


def _metadata_context(config: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = _evaluation_config(config)
    root = Path(evaluation["project_root"]).resolve()
    contract, live_status = load_live_v59_contract(
        root,
        state_path=evaluation["research_state"],
        phase_contract_path=evaluation["phase_contract"],
    )
    require(
        evaluation["output_dir"] == contract["artifact_contract"]["output_dir"],
        "V59 output directory differs from the phase contract",
    )
    context_path = resolve_repo_path(root, evaluation["config_path"], "config_path")
    require(context_path.is_file(), "V59 config file is missing")
    require(
        contract["commands"]["prepare"]
        == live_status["authorized_command"],
        "V59 prepare command differs from live authorization",
    )
    source = source_receipt(root, evaluation["source_receipt_files"])
    input_receipt = verify_input_files(root, contract)
    paths = {
        name: resolve_repo_path(root, relative, f"inputs.{name}")
        for name, relative in evaluation["inputs"].items()
    }
    for name, path in paths.items():
        require(path.is_file(), f"V59 input is missing: {name}")
    values = {
        name: load_json(path, f"V59 {name}")
        for name, path in paths.items()
        if name not in {"panel", "labels", "sequence_roles"}
    }
    blueprint = values["blueprint"]
    result = values["v58_result"]
    checkpoint_manifest = values["checkpoint_manifest"]
    scaler_manifest = values["scaler_manifest"]
    require(
        blueprint.get("candidate_family_id") == contract["family_id"]
        and set(blueprint["target_contract"]["target_symbols"]) == set(TARGET_SYMBOLS)
        and blueprint["target_contract"]["target_data_allowed"] is False,
        "V59 blueprint family/target boundary drift",
    )
    require(
        result.get("decision")
        == "authorize_v59_frozen_adaptive_development_evaluation_only"
        and result.get("result_sha256")
        == contract["authorization_receipt"]["registered_result_sha256"],
        "V58 result does not authorize V59",
    )
    require(
        checkpoint_manifest.get("checkpoint_count") == 36
        and checkpoint_manifest.get("selected_jobs") == []
        and len(checkpoint_manifest.get("jobs", [])) == 36,
        "V59 checkpoint manifest grid drift",
    )
    require(
        scaler_manifest.get("scaler_count") == 12
        and len(scaler_manifest.get("scalers", [])) == 12,
        "V59 scaler manifest grid drift",
    )
    folds = values["asset_folds"]
    catalog = values["triplet_catalog"]
    all_symbols = {
        str(symbol)
        for row in folds["folds"]
        for symbol in (*row["train_symbols"], *row["test_symbols"])
    }
    require(not all_symbols.intersection(TARGET_SYMBOLS), "V59 metadata contains target assets")
    context = {
        "root": root,
        "evaluation": evaluation,
        "contract": contract,
        "live_status": live_status,
        "source_receipt": source,
        "input_receipt": input_receipt,
        "paths": paths,
        "values": values,
        "blueprint": blueprint,
        "checkpoint_rows": checkpoint_manifest["jobs"],
        "scaler_rows": scaler_manifest["scalers"],
        "folds": folds,
        "catalog": catalog,
    }
    context["evaluation_spec"] = _build_evaluation_spec(context)
    return context


def _build_evaluation_spec(context: Mapping[str, Any]) -> dict[str, Any]:
    contract = context["contract"]
    projection = registered_projection(contract)
    prediction_schema = {
        "layout": "wide_one_row_per_available_triplet_date_asset",
        "primary_key": [
            "origin",
            "geometry",
            "fold",
            "triplet_key",
            "date",
            "asset_slot",
            "symbol",
        ],
        "columns": list(PREDICTION_COLUMNS),
        "seed_value_count_per_key": 27,
        "ensemble_value_count_per_key": 9,
    }
    position_schema = {
        "candidate_columns": list(POSITION_COLUMNS),
        "control_columns": list(CONTROL_POSITION_COLUMNS),
        "gross_weights_apply_to_same_row_return": True,
        "post_event_weights_apply_after_same_row_final_liquidation": True,
    }
    body = {
        "schema_version": "v59-evaluation-spec/v1",
        "version": VERSION,
        "family_id": contract["family_id"],
        "phase_contract_path": context["evaluation"]["phase_contract"],
        "phase_contract_file_sha256": V59_PHASE_FILE_SHA256,
        "phase_contract_canonical_sha256": V59_PHASE_CONTRACT_CANONICAL_SHA256,
        "registered_projection": projection,
        "registered_projection_sha256": canonical_sha256(projection),
        "source_receipt_sha256": context["source_receipt"]["source_receipt_sha256"],
        "input_hash_receipt_sha256": context["input_receipt"][
            "input_hash_receipt_sha256"
        ],
        "prediction_schema": prediction_schema,
        "position_schema": position_schema,
        "outcome_request_schema": {
            "primary_key": ["origin", "fold", "date", "symbol"],
            "allowed_columns": contract["one_shot_contract"]["unseal"][
                "allowed_columns"
            ],
        },
        "runtime_environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "mps_available": bool(torch.backends.mps.is_available()),
        },
        "prepare_only": True,
        "pass_authorizes_unseal": False,
        "performance_metrics_computed": 0,
        "pnl_evaluations": 0,
    }
    return with_self_hash(body, "evaluation_spec_sha256")


def _checkpoint_row(
    context: Mapping[str, Any], origin: str, geometry: str, fold: int, seed: int
) -> dict[str, Any]:
    rows = [
        row
        for row in context["checkpoint_rows"]
        if row["origin"] == origin
        and row["geometry"] == geometry
        and int(row["fold"]) == int(fold)
        and int(row["seed"]) == int(seed)
    ]
    require(len(rows) == 1, "V59 checkpoint grid key is not unique")
    return rows[0]


def _scaler_row(
    context: Mapping[str, Any], origin: str, geometry: str, fold: int
) -> dict[str, Any]:
    rows = [
        row
        for row in context["scaler_rows"]
        if row["origin"] == origin
        and row["geometry"] == geometry
        and int(row["fold"]) == int(fold)
    ]
    require(len(rows) == 1, "V59 scaler grid key is not unique")
    return rows[0]


def _load_scaler(
    context: Mapping[str, Any], cell: EvaluationCell
) -> tuple[ScalerValues, dict[str, Any]]:
    relative = (
        f"data/checkpoints/v58_state_conditioned_multi_horizon_training/"
        f"{cell.origin}/{cell.geometry}/fold_{cell.fold}/scaler.json"
    )
    path = resolve_repo_path(context["root"], relative, "V59 scaler path")
    expected_files = context["contract"]["input_contract"][
        "expected_scaler_file_sha256_by_path"
    ]
    require(relative in expected_files, "V59 scaler path is not registered")
    observed = file_sha256(path)
    require(observed == expected_files[relative], "V59 scaler file hash drift")
    wrapper = load_json(path, "V59 scaler wrapper")
    scaler = scaler_from_wrapper(wrapper)
    row = _scaler_row(context, cell.origin, cell.geometry, cell.fold)
    manifest_payload = {key: value for key, value in row.items() if key != "scaler_id"}
    require(
        (scaler.origin, scaler.geometry, scaler.fold)
        == (cell.origin, cell.geometry, cell.fold)
        and scaler.semantic_sha256 == row["scaler_sha256"]
        and wrapper["scaler_id"] == row["scaler_id"]
        and wrapper["scaler"] == manifest_payload,
        "V59 scaler wrapper/manifest semantic drift",
    )
    require(
        row["fit_symbols"] == list(cell.train_symbols)
        and int(row["fit_symbol_count"]) == 20
        and row["fit_min_date"] == day_text(cell.train_start)
        and row["fit_max_date"] == day_text(cell.train_end),
        "V59 scaler fit population/window differs from the exact cell train role",
    )
    receipt = {
        "cell_id": cell.cell_id,
        "path": relative,
        "file_sha256": observed,
        "scaler_sha256": scaler.semantic_sha256,
        "fit_symbols": list(cell.train_symbols),
        "fit_refit_count": 0,
        "load_count": 1,
    }
    return scaler, receipt


def _load_cell_models(
    context: Mapping[str, Any],
    cell: EvaluationCell,
    scaler_sha256: str,
    device: torch.device,
) -> tuple[list[torch.nn.Module], list[dict[str, Any]]]:
    architecture = context["blueprint"]["architecture"]
    models: list[torch.nn.Module] = []
    receipts: list[dict[str, Any]] = []
    for seed in SEEDS:
        row = _checkpoint_row(context, cell.origin, cell.geometry, cell.fold, seed)
        require(
            row["scaler_sha256"] == scaler_sha256,
            "V59 checkpoint is not bound to the loaded cell scaler",
        )
        relative = str(row["checkpoint_path"])
        expected_relative = (
            f"data/checkpoints/v58_state_conditioned_multi_horizon_training/"
            f"{cell.origin}/{cell.geometry}/fold_{cell.fold}/seed_{seed}/final.pt"
        )
        require(relative == expected_relative, "V59 checkpoint path drift")
        path = resolve_repo_path(context["root"], relative, "V59 checkpoint path")
        observed = file_sha256(path)
        require(observed == row["checkpoint_sha256"], "V59 checkpoint byte hash drift")
        model = instantiate_v58_model(architecture, "cpu")
        optimizer = build_v58_adamw(model)
        checkpoint_context = V58CheckpointContext(
            scaler_sha256=str(row["scaler_sha256"]),
            data_access_sha256=str(row["data_access_sha256"]),
            phase_contract_sha256=str(row["phase_contract_sha256"]),
            source_bundle_sha256=str(row["source_bundle_sha256"]),
            job_metadata=dict(row["job_metadata"]),
        )
        payload = load_v58_checkpoint(
            path,
            expected_kind="final",
            model=model,
            optimizer=optimizer,
            context=checkpoint_context,
            device="cpu",
            maximum_epochs=int(context["blueprint"]["training"]["maximum_epochs"]),
            patience=int(context["blueprint"]["training"]["early_stopping_patience"]),
            restore_runtime_state=False,
        )
        require(
            payload["best_model_state_sha256"] == row["best_model_state_sha256"]
            and model_state_sha256(payload["best_model_state"])
            == row["best_model_state_sha256"]
            and payload["semantic_checkpoint_sha256"]
            == row["semantic_checkpoint_sha256"],
            "V59 checkpoint best-state semantic drift",
        )
        model.load_state_dict(payload["best_model_state"], strict=True)
        model.requires_grad_(False).eval().to(device=device, dtype=torch.float32)
        models.append(model)
        receipts.append(
            {
                "job_id": row["job_id"],
                "origin": cell.origin,
                "geometry": cell.geometry,
                "fold": cell.fold,
                "seed": seed,
                "path": relative,
                "checkpoint_sha256": observed,
                "semantic_checkpoint_sha256": row["semantic_checkpoint_sha256"],
                "best_model_state_sha256": row["best_model_state_sha256"],
                "checkpoint_state": "best_model_state",
                "selected": False,
                "weight": None,
                "load_count": 1,
                "optimizer_steps": 0,
            }
        )
        del optimizer, payload
    return models, receipts


def _fit_linear_control(
    data: CellData, scaler: ScalerValues
) -> tuple[Ridge, float, dict[str, Any]]:
    features, targets, population = ridge_training_arrays(data, scaler)
    model = Ridge(
        alpha=1.0,
        fit_intercept=True,
        solver="svd",
        copy_X=True,
        tol=1.0e-4,
    )
    model.fit(features, targets)
    fitted = model.predict(features).astype(np.float64)
    residuals = targets - fitted
    residual_q20 = float(np.quantile(residuals, 0.2, method="linear"))
    require(
        np.isfinite(model.coef_).all()
        and math.isfinite(float(model.intercept_))
        and math.isfinite(residual_q20),
        "V59 linear control produced non-finite state",
    )
    state = {
        "coefficient": [float(value) for value in np.asarray(model.coef_, dtype=np.float64)],
        "intercept": float(model.intercept_),
        "residual_q20": residual_q20,
    }
    receipt = {
        "cell_id": data.cell.cell_id,
        "name": "shared_linear_h7_q50_with_train_residual_q20",
        "estimator": "sklearn.linear_model.Ridge",
        "sklearn_version": sklearn.__version__,
        "alpha": 1.0,
        "fit_intercept": True,
        "solver": "svd",
        "target": "target_h7_open_to_open_log_return",
        "residual_quantile": 0.2,
        "residual_quantile_method": "linear",
        "fit_scope": "exact_origin_geometry_fold_train_role_only",
        "validation_or_development_fit_rows": 0,
        "development_outcome_value_reads": 0,
        "population": population,
        "state": state,
        "state_sha256": canonical_sha256(state),
    }
    del features, targets, fitted, residuals
    gc.collect()
    return model, residual_q20, receipt


def _decision_schedule(eligible: np.ndarray, interval: int = 7) -> np.ndarray:
    values = np.asarray(eligible, dtype=bool)
    result = np.zeros(len(values), dtype=bool)
    has_decided = False
    since = 0
    for index, active in enumerate(values):
        if not active:
            continue
        if not has_decided:
            result[index] = True
            has_decided = True
            since = 0
        else:
            since += 1
            if since >= interval:
                result[index] = True
                since = 0
    return result


def _tie_best(
    utilities: Sequence[float], *, tolerance: float = 1.0e-12
) -> int:
    best = 0
    best_value = float(utilities[0])
    for index, value in enumerate(utilities[1:], start=1):
        current = float(value)
        if current > best_value + tolerance:
            best = index
            best_value = current
    return best


def _state_conditioned_positions(
    forecasts: np.ndarray,
    eligible: np.ndarray,
    *,
    risky_weight: float = 1.0 / 3.0,
    cost_decimal: float = 0.001,
) -> dict[str, np.ndarray]:
    values = np.asarray(forecasts, dtype=np.float64)
    active = np.asarray(eligible, dtype=bool)
    if values.shape != (len(active), 3):
        raise ValueError("V59 policy forecast geometry drift")
    schedule = _decision_schedule(active)
    weights = np.zeros((len(active), 3), dtype=np.float64)
    forced = np.zeros(len(active), dtype=bool)
    current = np.zeros(3, dtype=np.float64)
    for day in range(len(active)):
        if not active[day]:
            if current.sum() > 0:
                forced[day] = True
            current = np.zeros(3, dtype=np.float64)
        elif schedule[day]:
            candidates = [current.copy()]
            cash = np.zeros(3, dtype=np.float64)
            if not np.array_equal(current, cash):
                candidates.append(cash)
            for slot in range(3):
                candidate = np.zeros(3, dtype=np.float64)
                candidate[slot] = risky_weight
                if not any(np.array_equal(candidate, item) for item in candidates):
                    candidates.append(candidate)
            utilities = [
                float(np.dot(candidate, values[day]))
                - cost_decimal * float(np.abs(candidate - current).sum())
                for candidate in candidates
            ]
            current = candidates[_tie_best(utilities)].copy()
        weights[day] = current
    return {"weights": weights, "decision": schedule, "forced": forced}


def _momentum_positions(
    scores: np.ndarray, eligible: np.ndarray, *, risky_weight: float = 1.0 / 3.0
) -> dict[str, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64)
    active = np.asarray(eligible, dtype=bool)
    schedule = _decision_schedule(active)
    weights = np.zeros((len(active), 3), dtype=np.float64)
    forced = np.zeros(len(active), dtype=bool)
    current = np.zeros(3, dtype=np.float64)
    for day in range(len(active)):
        if not active[day]:
            if current.sum() > 0:
                forced[day] = True
            current = np.zeros(3, dtype=np.float64)
        elif schedule[day]:
            maximum = float(np.max(values[day]))
            if not math.isfinite(maximum) or maximum <= 0.0:
                current = np.zeros(3, dtype=np.float64)
            else:
                tied = [
                    slot
                    for slot in range(3)
                    if abs(float(values[day, slot]) - maximum) <= 1.0e-12
                ]
                incumbent = int(np.argmax(current)) if current.sum() > 0 else None
                selected = incumbent if incumbent in tied else tied[0]
                current = np.zeros(3, dtype=np.float64)
                current[selected] = risky_weight
        weights[day] = current
    return {"weights": weights, "decision": schedule, "forced": forced}


def _equal_weight_positions(eligible: np.ndarray) -> dict[str, np.ndarray]:
    active = np.asarray(eligible, dtype=bool)
    schedule = _decision_schedule(active)
    weights = np.zeros((len(active), 3), dtype=np.float64)
    forced = np.zeros(len(active), dtype=bool)
    current = np.zeros(3, dtype=np.float64)
    for day in range(len(active)):
        if not active[day]:
            if current.sum() > 0:
                forced[day] = True
            current = np.zeros(3, dtype=np.float64)
        elif schedule[day]:
            current = np.full(3, 1.0 / 9.0, dtype=np.float64)
        weights[day] = current
    return {"weights": weights, "decision": schedule, "forced": forced}


def _cash_positions(eligible: np.ndarray) -> dict[str, np.ndarray]:
    active = np.asarray(eligible, dtype=bool)
    return {
        "weights": np.zeros((len(active), 3), dtype=np.float64),
        "decision": _decision_schedule(active),
        "forced": np.zeros(len(active), dtype=bool),
    }


def _prediction_batch_frame(
    cell: EvaluationCell,
    samples: Sequence[tuple[pd.Timestamp, tuple[str, str, str]]],
    seed_outputs: Sequence[np.ndarray],
    ensemble: np.ndarray,
    linear_q50: np.ndarray,
    linear_q20: np.ndarray,
) -> pd.DataFrame:
    size = len(samples)
    rows = size * 3
    triplets = ["|".join(triplet) for _, triplet in samples]
    dates = pd.DatetimeIndex([date for date, _ in samples])
    symbols = [symbol for _, triplet in samples for symbol in triplet]
    values: dict[str, Any] = {
        "origin": np.full(rows, cell.origin, dtype=object),
        "geometry": np.full(rows, cell.geometry, dtype=object),
        "fold": np.full(rows, cell.fold, dtype=np.int8),
        "triplet_key": np.repeat(np.asarray(triplets, dtype=object), 3),
        "date": dates.repeat(3),
        "asset_slot": np.tile(np.arange(3, dtype=np.int8), size),
        "symbol": np.asarray(symbols, dtype=object),
    }
    for seed_index, seed in enumerate(SEEDS):
        output = seed_outputs[seed_index]
        for horizon_index, horizon in enumerate(HORIZONS):
            for quantile_index, quantile in enumerate(QUANTILES):
                values[f"seed_{seed}_h{horizon}_q{quantile}"] = output[
                    :, :, horizon_index, quantile_index
                ].reshape(-1)
    for horizon_index, horizon in enumerate(HORIZONS):
        for quantile_index, quantile in enumerate(QUANTILES):
            values[f"ensemble_h{horizon}_q{quantile}"] = ensemble[
                :, :, horizon_index, quantile_index
            ].reshape(-1)
    values["linear_h7_q50"] = linear_q50.reshape(-1)
    values["linear_h7_q20"] = linear_q20.reshape(-1)
    frame = pd.DataFrame(values, columns=PREDICTION_COLUMNS)
    numeric = frame.loc[:, prediction_value_columns() + ["linear_h7_q50", "linear_h7_q20"]]
    if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
        raise V59PrepareError("V59 prediction frame contains non-finite values")
    if frame.duplicated(
        ["origin", "geometry", "fold", "triplet_key", "date", "asset_slot", "symbol"]
    ).any():
        raise V59PrepareError("V59 prediction primary-key duplication")
    return frame


def _infer_cell(
    context: Mapping[str, Any],
    data: CellData,
    scaler: ScalerValues,
    linear: Ridge,
    residual_q20: float,
    device: torch.device,
) -> tuple[
    pd.DataFrame,
    dict[tuple[pd.Timestamp, tuple[str, str, str]], tuple[np.ndarray, np.ndarray]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    samples, unavailable_contexts = classify_development_samples(data)
    require(samples, f"V59 cell has no eligible inference samples: {data.cell.cell_id}")
    models, checkpoint_receipts = _load_cell_models(
        context, data.cell, scaler.semantic_sha256, device
    )
    batch_frames: list[pd.DataFrame] = []
    prediction_map: dict[
        tuple[pd.Timestamp, tuple[str, str, str]], tuple[np.ndarray, np.ndarray]
    ] = {}
    batch_size = int(context["contract"]["inference_contract"]["batch_size"])
    require(batch_size == 128, "V59 inference batch-size drift")
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        features = materialize_development_batch(data, batch, scaler)
        tensor = torch.from_numpy(features).to(device=device, dtype=torch.float32)
        seed_outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for model in models:
                output = model(tensor)
                seed_outputs.append(
                    output.detach().cpu().numpy().astype(np.float64, copy=False)
                )
        ensemble = np.zeros_like(seed_outputs[0], dtype=np.float64)
        ensemble += seed_outputs[0]
        ensemble += seed_outputs[1]
        ensemble += seed_outputs[2]
        ensemble /= 3
        final_features = features[:, -1, :, :].astype(np.float64)
        linear_q50 = linear.predict(final_features.reshape(-1, 9)).reshape(-1, 3)
        linear_q20 = linear_q50 + float(residual_q20)
        frame = _prediction_batch_frame(
            data.cell,
            batch,
            seed_outputs,
            ensemble,
            linear_q50,
            linear_q20,
        )
        batch_frames.append(frame)
        for offset, sample in enumerate(batch):
            prediction_map[sample] = (
                ensemble[offset, :, 2, 0].copy(),
                linear_q20[offset].copy(),
            )
        del features, tensor, seed_outputs, ensemble, final_features, linear_q50, linear_q20
    predictions = pd.concat(batch_frames, ignore_index=True)
    expected_rows = len(samples) * 3
    require(len(predictions) == expected_rows, "V59 prediction row-count drift")
    for model in models:
        model.to("cpu")
    del models, batch_frames
    if device.type == "mps":
        torch.mps.empty_cache()
    gc.collect()
    diagnostics = {
        "cell_id": data.cell.cell_id,
        "registered_test_triplets": 120,
        "eligible_triplet_dates": len(samples),
        "unavailable_context_count": len(unavailable_contexts),
        "unavailable_contexts": unavailable_contexts,
        "prediction_rows": len(predictions),
        "seed_level_values_per_prediction_key": 27,
        "ensemble_values_per_prediction_key": 9,
        "checkpoint_loads": 3,
        "seed_context_forwards": len(samples) * 3,
        "seed_aggregation_order": [42, 7, 123],
        "seed_aggregation_dtype": "float64_cpu",
        "development_outcome_value_reads": 0,
        "target_asset_loads": 0,
        "optimizer_steps": 0,
    }
    return predictions, prediction_map, checkpoint_receipts, diagnostics


def _accounting_fields(weights: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(weights, dtype=np.float64)
    base = np.zeros(len(values), dtype=np.float64)
    previous = np.zeros(3, dtype=np.float64)
    for day in range(len(values)):
        base[day] = float(np.abs(values[day] - previous).sum())
        previous = values[day]
    liquidation = np.zeros(len(values), dtype=np.float64)
    liquidation[-1] = float(np.abs(values[-1]).sum())
    post = values.copy()
    post[-1] = 0.0
    return {
        "base_turnover": base,
        "final_liquidation_turnover": liquidation,
        "turnover": base + liquidation,
        "post": post,
    }


def _position_frame(
    cell: EvaluationCell,
    dates: pd.DatetimeIndex,
    triplet: tuple[str, str, str],
    eligible: np.ndarray,
    policy: Mapping[str, np.ndarray],
    *,
    control: str | None = None,
) -> pd.DataFrame:
    weights = np.asarray(policy["weights"], dtype=np.float64)
    decision = np.asarray(policy["decision"], dtype=bool)
    forced = np.asarray(policy["forced"], dtype=bool)
    if weights.shape != (len(dates), 3):
        raise V59PrepareError("V59 position weight geometry drift")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise V59PrepareError("V59 position weights are invalid")
    accounting = _accounting_fields(weights)
    actions: list[str] = []
    selected: list[str] = []
    for row in weights:
        active = np.flatnonzero(row > 0)
        if not len(active):
            actions.append("cash")
            selected.append("")
        elif len(active) == 1:
            actions.append("long_one_asset")
            selected.append(triplet[int(active[0])])
        else:
            actions.append("equal_weight")
            selected.append("")
    values: dict[str, Any] = {
        "origin": np.full(len(dates), cell.origin, dtype=object),
        "geometry": np.full(len(dates), cell.geometry, dtype=object),
        "fold": np.full(len(dates), cell.fold, dtype=np.int8),
        "triplet_key": np.full(len(dates), "|".join(triplet), dtype=object),
        "date": dates,
        "symbol_0": np.full(len(dates), triplet[0], dtype=object),
        "symbol_1": np.full(len(dates), triplet[1], dtype=object),
        "symbol_2": np.full(len(dates), triplet[2], dtype=object),
        "available": np.asarray(eligible, dtype=bool),
        "decision": decision,
        "forced_cash": forced,
        "final_liquidation": np.arange(len(dates)) == len(dates) - 1,
        "action": np.asarray(actions, dtype=object),
        "selected_symbol": np.asarray(selected, dtype=object),
        "weight_0": weights[:, 0],
        "weight_1": weights[:, 1],
        "weight_2": weights[:, 2],
        "post_event_weight_0": accounting["post"][:, 0],
        "post_event_weight_1": accounting["post"][:, 1],
        "post_event_weight_2": accounting["post"][:, 2],
        "base_turnover": accounting["base_turnover"],
        "final_liquidation_turnover": accounting["final_liquidation_turnover"],
        "turnover": accounting["turnover"],
    }
    frame = pd.DataFrame(values, columns=POSITION_COLUMNS)
    gross = weights.sum(axis=1)
    changed = accounting["base_turnover"] > 1.0e-15
    if not np.all(~changed | decision | forced):
        raise V59PrepareError("V59 action changed outside a decision or forced exit")
    if not np.all(~decision | np.asarray(eligible, dtype=bool)):
        raise V59PrepareError("V59 decision occurred on an ineligible date")
    if control == "cash":
        valid_weights = bool(np.all(weights == 0.0))
    elif control == "weekly_equal_weight_total_gross_one_third":
        valid_weights = bool(
            np.all(
                np.isclose(gross, 0.0, rtol=0.0, atol=1.0e-15)
                | (
                    np.isclose(weights[:, 0], 1.0 / 9.0, rtol=0.0, atol=1.0e-15)
                    & np.isclose(weights[:, 1], 1.0 / 9.0, rtol=0.0, atol=1.0e-15)
                    & np.isclose(weights[:, 2], 1.0 / 9.0, rtol=0.0, atol=1.0e-15)
                )
            )
        )
    else:
        valid_weights = bool(
            np.all(
                np.isclose(gross, 0.0, rtol=0.0, atol=1.0e-15)
                | np.isclose(gross, 1.0 / 3.0, rtol=0.0, atol=1.0e-15)
            )
            and np.all((weights > 0).sum(axis=1) <= 1)
        )
    if not valid_weights:
        raise V59PrepareError("V59 candidate/control weight contract drift")
    expected_turnover = accounting["base_turnover"] + accounting[
        "final_liquidation_turnover"
    ]
    if not np.array_equal(accounting["turnover"], expected_turnover):
        raise V59PrepareError("V59 turnover accounting drift")
    if control is not None:
        frame.insert(5, "control", control)
        frame = frame.loc[:, CONTROL_POSITION_COLUMNS]
    return frame


def _momentum_lookup(data: CellData) -> dict[tuple[pd.Timestamp, str], float]:
    result: dict[tuple[pd.Timestamp, str], float] = {}
    for date, symbols in sorted(data.development_availability.items()):
        for symbol in symbols:
            start = data.sequence_start_by_key.get((date, symbol))
            if start is None:
                continue
            window = data.panel_index.window(start, date, symbol)
            if window is None:
                continue
            tail = window[-30:, 0]
            if len(tail) == 30 and np.isfinite(tail).all():
                result[(date, symbol)] = float(tail.sum(dtype=np.float64))
    return result


def _build_cell_positions(
    data: CellData,
    prediction_map: Mapping[
        tuple[pd.Timestamp, tuple[str, str, str]], tuple[np.ndarray, np.ndarray]
    ],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dates = pd.date_range(
        data.cell.development_start,
        data.cell.development_end,
        freq="D",
        tz="UTC",
    )
    momentum = _momentum_lookup(data)
    candidate_frames: list[pd.DataFrame] = []
    control_frames: list[pd.DataFrame] = []
    eligible_count = 0
    for triplet in data.cell.test_triplets:
        candidate_forecast = np.full((len(dates), 3), np.nan, dtype=np.float64)
        linear_forecast = np.full((len(dates), 3), np.nan, dtype=np.float64)
        momentum_scores = np.full((len(dates), 3), np.nan, dtype=np.float64)
        eligible = np.zeros(len(dates), dtype=bool)
        for day, date in enumerate(dates):
            value = prediction_map.get((date, triplet))
            if value is None:
                continue
            candidate_forecast[day] = value[0]
            linear_forecast[day] = value[1]
            scores = [momentum.get((date, symbol), math.nan) for symbol in triplet]
            if not np.isfinite(scores).all():
                raise V59PrepareError("V59 eligible triplet lacks exact 30-day momentum")
            momentum_scores[day] = scores
            eligible[day] = True
        eligible_count += int(eligible.sum())
        candidate = _state_conditioned_positions(candidate_forecast, eligible)
        linear = _state_conditioned_positions(linear_forecast, eligible)
        dual = _momentum_positions(momentum_scores, eligible)
        equal = _equal_weight_positions(eligible)
        cash = _cash_positions(eligible)
        candidate_frames.append(
            _position_frame(data.cell, dates, triplet, eligible, candidate)
        )
        for name, policy in (
            ("cash", cash),
            ("weekly_dual_momentum_30", dual),
            ("weekly_equal_weight_total_gross_one_third", equal),
            ("shared_linear_h7_q50_with_train_residual_q20", linear),
        ):
            control_frames.append(
                _position_frame(
                    data.cell,
                    dates,
                    triplet,
                    eligible,
                    policy,
                    control=name,
                )
            )
    candidates = pd.concat(candidate_frames, ignore_index=True)
    controls = pd.concat(control_frames, ignore_index=True)
    expected = len(dates) * 120
    require(len(candidates) == expected, "V59 candidate position row-count drift")
    require(len(controls) == expected * 4, "V59 control position row-count drift")
    candidate_gross = candidates[["weight_0", "weight_1", "weight_2"]].sum(axis=1)
    require(
        np.isin(
            np.round(candidate_gross.to_numpy(), 15),
            [0.0, round(1.0 / 3.0, 15)],
        ).all(),
        "V59 candidate gross exposure drift",
    )
    diagnostics = {
        "cell_id": data.cell.cell_id,
        "calendar_dates": len(dates),
        "registered_triplets": 120,
        "eligible_triplet_dates": eligible_count,
        "candidate_position_rows": len(candidates),
        "control_position_rows": len(controls),
        "control_count": 4,
        "positions_frozen_at_cost_bps": 10,
        "final_liquidation_rows_candidate": int(candidates["final_liquidation"].sum()),
        "final_liquidation_rows_controls": int(controls["final_liquidation"].sum()),
    }
    return candidates, controls, diagnostics


def _outcome_request(
    contract: Mapping[str, Any],
    keys_by_origin_fold: Mapping[tuple[str, int], frozenset[tuple[pd.Timestamp, str]]],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for (origin, fold), keys in sorted(keys_by_origin_fold.items()):
        group_primary = [
            (origin, int(fold), day_text(date), str(symbol))
            for date, symbol in sorted(keys)
        ]
        groups.append(
            {
                "origin": origin,
                "fold": int(fold),
                "key_count": len(group_primary),
                "key_sha256": canonical_sha256(group_primary),
                "development_sequence_key_sha256": canonical_sha256(
                    [[day_text(date), str(symbol)] for date, symbol in sorted(keys)]
                ),
            }
        )
        for date, symbol in sorted(keys):
            records.append(
                {
                    "origin": origin,
                    "fold": int(fold),
                    "date": day_text(date),
                    "symbol": str(symbol),
                }
            )
    primary = [(row["origin"], row["fold"], row["date"], row["symbol"]) for row in records]
    require(primary == sorted(set(primary)), "V59 outcome request keys are not sorted unique")
    require(
        not {row["symbol"] for row in records}.intersection(TARGET_SYMBOLS),
        "V59 outcome request contains a sealed target",
    )
    body = {
        "schema_version": "v59-outcome-request/v1",
        "primary_key": ["origin", "fold", "date", "symbol"],
        "allowed_columns": contract["one_shot_contract"]["unseal"]["allowed_columns"],
        "keys": records,
        "groups": groups,
        "group_count": len(groups),
        "key_count": len(records),
        "key_sha256": canonical_sha256(primary),
        "development_outcome_value_reads_during_prepare": 0,
        "target_asset_loads": 0,
    }
    return with_self_hash(body, "outcome_request_sha256")


def _binding_packet(
    schema: str, entries: Sequence[Mapping[str, Any]], hash_field: str
) -> dict[str, Any]:
    body = {
        "schema_version": schema,
        "entries": [dict(row) for row in entries],
        "entry_count": len(entries),
    }
    return with_self_hash(body, hash_field)


def _behavior_audit(
    context: Mapping[str, Any],
    *,
    data_receipts: Sequence[Mapping[str, Any]],
    inference_diagnostics: Sequence[Mapping[str, Any]],
    position_diagnostics: Sequence[Mapping[str, Any]],
    checkpoint_receipts: Sequence[Mapping[str, Any]],
    scaler_receipts: Sequence[Mapping[str, Any]],
    linear_receipts: Sequence[Mapping[str, Any]],
    outcome_request: Mapping[str, Any],
    independent_verification: Mapping[str, Any],
) -> dict[str, Any]:
    prediction_rows = sum(int(row["prediction_rows"]) for row in inference_diagnostics)
    eligible = sum(int(row["eligible_triplet_dates"]) for row in inference_diagnostics)
    candidate_rows = sum(int(row["candidate_position_rows"]) for row in position_diagnostics)
    control_rows = sum(int(row["control_position_rows"]) for row in position_diagnostics)
    request_groups = {
        (row["origin"], int(row["fold"])): row
        for row in outcome_request["groups"]
    }
    access_groups: dict[tuple[str, int], tuple[int, str]] = {}
    access_groups_match = True
    for row in data_receipts:
        origin, _, fold_text = str(row["cell_id"]).split("|")
        key = (origin, int(fold_text))
        value = (
            int(row["development_sequence_key_count"]),
            str(row["development_sequence_key_sha256"]),
        )
        if key in access_groups and access_groups[key] != value:
            access_groups_match = False
        access_groups[key] = value
    request_groups_match = access_groups_match and set(request_groups) == set(access_groups)
    if request_groups_match:
        for key, (count, digest) in access_groups.items():
            request_groups_match = request_groups_match and (
                request_groups[key]["key_count"] == count
                and request_groups[key]["development_sequence_key_sha256"] == digest
            )
    independent_cells = independent_verification.get("cell_diagnostics", {})
    independent_diagnostics_match = (
        independent_verification.get("passed") is True
        and set(independent_cells)
        == {str(row["cell_id"]) for row in inference_diagnostics}
        and all(
            independent_cells[str(row["cell_id"])]["eligible_triplet_dates"]
            == int(row["eligible_triplet_dates"])
            and independent_cells[str(row["cell_id"])]["unavailable_context_count"]
            == int(row["unavailable_context_count"])
            and independent_cells[str(row["cell_id"])][
                "unavailable_context_sha256"
            ]
            == canonical_sha256(row["unavailable_contexts"])
            for row in inference_diagnostics
        )
    )
    checks = {
        "input_bindings_exact": context["input_receipt"]["file_count"] == 29,
        "parquet_access_exact": len(data_receipts) == 12
        and all(
            row["development_outcome_value_reads"] == 0
            and row["development_outcome_columns_materialized"] == []
            and row["full_table_materializations"] == 0
            for row in data_receipts
        ),
        "checkpoint_grid_exact": len(checkpoint_receipts) == 36
        and len({row["job_id"] for row in checkpoint_receipts}) == 36
        and all(
            row["load_count"] == 1
            and row["selected"] is False
            and row["weight"] is None
            and row["optimizer_steps"] == 0
            for row in checkpoint_receipts
        ),
        "scaler_grid_exact": len(scaler_receipts) == 12
        and len({row["cell_id"] for row in scaler_receipts}) == 12
        and all(row["fit_refit_count"] == 0 for row in scaler_receipts),
        "targets_absent": all(row["target_asset_loads"] == 0 for row in data_receipts)
        and set(context["contract"]["target_contract"]["symbols"]) == set(TARGET_SYMBOLS),
        "development_outcomes_sealed": all(
            row["development_outcome_value_reads"] == 0 for row in data_receipts
        ),
        "prediction_keys_exact": len(inference_diagnostics) == 12
        and independent_diagnostics_match
        and prediction_rows == eligible * 3
        and all(
            row["seed_level_values_per_prediction_key"] == 27
            and row["ensemble_values_per_prediction_key"] == 9
            for row in inference_diagnostics
        ),
        "predictions_finite": independent_verification.get("passed") is True,
        "seed_aggregation_exact": all(
            row["seed_aggregation_order"] == [42, 7, 123]
            and row["seed_aggregation_dtype"] == "float64_cpu"
            for row in inference_diagnostics
        )
        and independent_verification.get("passed") is True,
        "candidate_positions_exact": len(position_diagnostics) == 12
        and independent_verification.get("passed") is True
        and candidate_rows
        == sum(int(row["calendar_dates"]) * 120 for row in position_diagnostics),
        "control_positions_exact": control_rows == candidate_rows * 4
        and independent_verification.get("passed") is True,
        "turnover_structure_exact": all(
            row["final_liquidation_rows_candidate"] == 120
            and row["final_liquidation_rows_controls"] == 480
            for row in position_diagnostics
        )
        and independent_verification.get("passed") is True,
        "outcome_request_exact": outcome_request["key_count"] > 0
        and request_groups_match
        and outcome_request["allowed_columns"]
        == context["contract"]["one_shot_contract"]["unseal"]["allowed_columns"],
        "prepare_packet_complete_atomic": True,
        "prepare_replay_exact": True,
        "linear_control_train_only": len(linear_receipts) == 12
        and all(
            row["validation_or_development_fit_rows"] == 0
            and row["development_outcome_value_reads"] == 0
            for row in linear_receipts
        ),
        "no_performance_or_pnl_during_prepare": True,
        "prepare_stops_before_unseal_stage": context["contract"]["one_shot_contract"][
            "prepare"
        ]["pass_authorizes_unseal"]
        is False,
    }
    registered_gate_names = set(
        context["contract"]["outcome_blind_gate_contract"]["gates"]
    )
    require(
        registered_gate_names.issubset(checks),
        "V59 behavior audit omits a registered outcome-blind gate",
    )
    body = {
        "schema_version": "v59-behavior-audit/v1",
        "passed": all(checks.values()),
        "checks": checks,
        "operation_ledger": {
            "cells": 12,
            "checkpoint_loads": len(checkpoint_receipts),
            "scaler_loads": len(scaler_receipts),
            "linear_control_fits": len(linear_receipts),
            "optimizer_steps": 0,
            "prediction_rows": prediction_rows,
            "candidate_position_rows": candidate_rows,
            "control_position_rows": control_rows,
            "development_outcome_value_reads": 0,
            "target_asset_loads": 0,
            "performance_metrics_computed": 0,
            "pnl_evaluations": 0,
        },
        "data_access_receipts": [dict(row) for row in data_receipts],
        "inference_diagnostics": [dict(row) for row in inference_diagnostics],
        "position_diagnostics": [dict(row) for row in position_diagnostics],
        "independent_preoutcome_verification": dict(independent_verification),
        "procedural_evidence": {
            "prediction_finiteness": "every_written_batch_rejected_on_any_nonfinite_value",
            "position_semantics": "every_triplet_policy_validated_before_parquet_write",
            "atomic_packet": "hidden_sibling_staging_then_single_directory_os_replace_after_v2_validation",
            "cached_replay": "published_packet_is_reentered_through_hash_only_cached_path_before_success_return",
            "no_outcome_or_pnl": "prepare_module_has_no_unseal_or_metric_entrypoint_and_ledger_is_zero",
        },
    }
    require(body["passed"], "V59 pre-outcome behavior audit failed")
    return with_self_hash(body, "behavior_audit_sha256")


def _cached_result(context: Mapping[str, Any], output: Path) -> dict[str, Any]:
    verification = verify_prepare_packet(
        context["root"],
        output,
        contract=context["contract"],
        enforce_live_inputs=False,
    )
    stored_source = load_json(output / "source_receipt.json", "stored source receipt")
    require(
        stored_source == context["source_receipt"],
        "cached V59 packet source receipt differs from the live clean head",
    )
    behavior = load_json(output / "behavior_audit.json", "behavior audit")
    ledger = behavior["operation_ledger"]
    return {
        "version": VERSION,
        "decision": PREPARE_DECISION,
        "audit": {"passed": True},
        "summary": {
            "cell_count": ledger["cells"],
            "checkpoint_count": ledger["checkpoint_loads"],
            "prediction_rows": ledger["prediction_rows"],
            "candidate_position_rows": ledger["candidate_position_rows"],
            "control_position_rows": ledger["control_position_rows"],
            "development_outcome_value_reads": 0,
            "performance_metrics_computed": 0,
            "pnl_evaluations": 0,
        },
        "invocation": {
            "cached": True,
            "new_checkpoint_loads": 0,
            "new_inference": 0,
            "new_linear_control_fits": 0,
            "new_position_generation": 0,
            "new_outcome_reads": 0,
            "files_rewritten": 0,
        },
        "verification": verification,
    }


def _stable_prepare_file_hashes(output: Path) -> dict[str, str]:
    """Hash every frozen prepare file without deserializing outcome values."""

    return {
        name: file_sha256(output / name)
        for name in REQUIRED_PREPARE_FILES
    }


def prepare_state_conditioned_multi_horizon_evaluation(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    context = _metadata_context(config)
    root = context["root"]
    output = resolve_repo_path(root, context["evaluation"]["output_dir"], "output_dir")
    if output.exists():
        return _cached_result(context, output)
    runtime = context["contract"]["runtime_contract"]
    minimum_free_space(root, float(runtime["minimum_free_gib"]))
    lock_path = resolve_repo_path(root, runtime["process_lock"], "process_lock")
    with process_lock(lock_path):
        if output.exists():
            return _cached_result(context, output)
        device = configure_v58_runtime(
            "mps", seed=20260714, torch_threads=1
        )
        staging = output.with_name(f".{output.name}.prepare-{os.getpid()}")
        require(not staging.exists(), "V59 staging directory already exists")
        staging.mkdir(parents=True)
        try:
            write_json(staging / "evaluation_spec.json", context["evaluation_spec"])
            write_json(staging / "source_receipt.json", context["source_receipt"])
            write_json(staging / "input_hash_receipt.json", context["input_receipt"])

            data_receipts: list[dict[str, Any]] = []
            inference_diagnostics: list[dict[str, Any]] = []
            position_diagnostics: list[dict[str, Any]] = []
            checkpoint_receipts: list[dict[str, Any]] = []
            scaler_receipts: list[dict[str, Any]] = []
            linear_receipts: list[dict[str, Any]] = []
            development_keys: dict[
                tuple[str, int], frozenset[tuple[pd.Timestamp, str]]
            ] = {}

            predictions_writer = ParquetAccumulator(
                staging / "predictions.parquet", PREDICTION_COLUMNS
            )
            candidates_writer = ParquetAccumulator(
                staging / "candidate_positions.parquet", POSITION_COLUMNS
            )
            controls_writer = ParquetAccumulator(
                staging / "control_positions.parquet", CONTROL_POSITION_COLUMNS
            )
            with predictions_writer, candidates_writer, controls_writer:
                completed = 0
                for origin in ("origin_2024", "origin_2025"):
                    for geometry in ("expanding", "rolling"):
                        for fold in (1, 2, 3):
                            cell = build_evaluation_cell(
                                context["contract"],
                                context["values"]["dataset_spec"],
                                context["folds"],
                                context["catalog"],
                                origin=origin,
                                geometry=geometry,
                                fold=fold,
                            )
                            print(
                                f"[V59 prepare] cell {completed + 1}/12 {cell.cell_id}: projected reads",
                                file=sys.stderr,
                                flush=True,
                            )
                            data = read_cell_data(
                                cell,
                                sequence_path=context["paths"]["sequence_roles"],
                                labels_path=context["paths"]["labels"],
                                panel_path=context["paths"]["panel"],
                            )
                            key = (origin, fold)
                            if key in development_keys:
                                require(
                                    development_keys[key] == data.development_keys,
                                    "V59 development keys differ across geometries",
                                )
                            else:
                                development_keys[key] = data.development_keys
                            scaler, scaler_receipt = _load_scaler(context, cell)
                            print(
                                f"[V59 prepare] cell {completed + 1}/12 {cell.cell_id}: train-only Ridge",
                                file=sys.stderr,
                                flush=True,
                            )
                            linear, residual_q20, linear_receipt = _fit_linear_control(
                                data, scaler
                            )
                            print(
                                f"[V59 prepare] cell {completed + 1}/12 {cell.cell_id}: three-seed MPS inference",
                                file=sys.stderr,
                                flush=True,
                            )
                            predictions, prediction_map, checkpoint_rows, inference = _infer_cell(
                                context,
                                data,
                                scaler,
                                linear,
                                residual_q20,
                                device,
                            )
                            candidates, controls, positions = _build_cell_positions(
                                data, prediction_map
                            )
                            predictions_writer.write(predictions)
                            candidates_writer.write(candidates)
                            controls_writer.write(controls)
                            data_receipts.append(data.access_receipt)
                            inference_diagnostics.append(inference)
                            position_diagnostics.append(positions)
                            checkpoint_receipts.extend(checkpoint_rows)
                            scaler_receipts.append(scaler_receipt)
                            linear_receipts.append(linear_receipt)
                            completed += 1
                            print(
                                f"[V59 prepare] cell {completed}/12 {cell.cell_id}: frozen",
                                file=sys.stderr,
                                flush=True,
                            )
                            del (
                                data,
                                linear,
                                predictions,
                                prediction_map,
                                candidates,
                                controls,
                            )
                            gc.collect()

            checkpoint_binding = _binding_packet(
                "v59-checkpoint-binding/v1",
                checkpoint_receipts,
                "checkpoint_binding_sha256",
            )
            scaler_binding = _binding_packet(
                "v59-scaler-binding/v1", scaler_receipts, "scaler_binding_sha256"
            )
            linear_binding = _binding_packet(
                "v59-linear-control-receipt/v1",
                linear_receipts,
                "linear_control_receipt_sha256",
            )
            request = _outcome_request(context["contract"], development_keys)
            print(
                "[V59 prepare] independent outcome-blind semantic verification",
                file=sys.stderr,
                flush=True,
            )
            _, _, registered_cell_contexts = _registered_access_receipts(
                root,
                context["contract"],
                context["contract"]["input_contract"][
                    "expected_file_sha256_by_path"
                ],
            )
            independent_verification = verify_prepared_values(
                root,
                staging,
                context["contract"],
                scaler_binding,
                linear_binding,
                {
                    "prediction_rows": sum(
                        int(row["prediction_rows"])
                        for row in inference_diagnostics
                    ),
                    "candidate_position_rows": sum(
                        int(row["candidate_position_rows"])
                        for row in position_diagnostics
                    ),
                    "control_position_rows": sum(
                        int(row["control_position_rows"])
                        for row in position_diagnostics
                    ),
                },
                registered_cell_contexts,
            )
            behavior = _behavior_audit(
                context,
                data_receipts=data_receipts,
                inference_diagnostics=inference_diagnostics,
                position_diagnostics=position_diagnostics,
                checkpoint_receipts=checkpoint_receipts,
                scaler_receipts=scaler_receipts,
                linear_receipts=linear_receipts,
                outcome_request=request,
                independent_verification=independent_verification,
            )
            write_json(staging / "checkpoint_binding.json", checkpoint_binding)
            write_json(staging / "scaler_binding.json", scaler_binding)
            write_json(staging / "linear_control_receipt.json", linear_binding)
            write_json(staging / "outcome_request.json", request)
            write_json(staging / "behavior_audit.json", behavior)

            parquet_metadata = {
                "predictions.parquet": predictions_writer.metadata(),
                "candidate_positions.parquet": candidates_writer.metadata(),
                "control_positions.parquet": controls_writer.metadata(),
            }
            manifest = build_prepare_manifest(
                staging, parquet_metadata=parquet_metadata
            )
            write_json(staging / "prepare_manifest.json", manifest)
            replay_file_hashes = {
                name: file_sha256(staging / name)
                for name in REQUIRED_PREPARE_FILES
                if name != "prepare_receipt.json"
            }
            receipt_body = {
                "schema_version": PREPARE_SCHEMA,
                "decision": PREPARE_DECISION,
                "pass_authorizes_unseal": False,
                "eligible_to_request_explicit_authorization": True,
                "authorization_state": "awaiting_explicit_user_authorization",
                "next_action": PREPARE_DECISION,
                "required_stage_revision": "v059_unseal_r1",
                "phase_contract_file_sha256": V59_PHASE_FILE_SHA256,
                "phase_contract_canonical_sha256": V59_PHASE_CONTRACT_CANONICAL_SHA256,
                "registered_projection_sha256": context["evaluation_spec"][
                    "registered_projection_sha256"
                ],
                "prepare_git_head": context["source_receipt"]["git_head"],
                "source_receipt_file_sha256": file_sha256(
                    staging / "source_receipt.json"
                ),
                "evaluation_spec_file_sha256": file_sha256(
                    staging / "evaluation_spec.json"
                ),
                "prepare_manifest_file_sha256": file_sha256(
                    staging / "prepare_manifest.json"
                ),
                "outcome_request_file_sha256": file_sha256(
                    staging / "outcome_request.json"
                ),
                "behavior_audit_file_sha256": file_sha256(
                    staging / "behavior_audit.json"
                ),
                "development_outcome_value_reads": 0,
                "target_asset_loads": 0,
                "performance_metrics_computed": 0,
                "pnl_evaluations": 0,
                "outcome_packet_created": False,
                "authorization_receipt_created": False,
                "cached_replay_binding": {
                    "stage": "hidden_staging_before_atomic_publish",
                    "files": replay_file_hashes,
                    "file_count": len(replay_file_hashes),
                    "file_hash_map_sha256": canonical_sha256(replay_file_hashes),
                    "new_checkpoint_loads": 0,
                    "new_inference": 0,
                    "new_linear_control_fits": 0,
                    "new_position_generation": 0,
                    "new_outcome_reads": 0,
                    "files_rewritten": 0,
                },
            }
            receipt = with_self_hash(receipt_body, "prepare_receipt_sha256")
            write_json(staging / "prepare_receipt.json", receipt)
            verify_prepare_packet(
                root,
                staging,
                contract=context["contract"],
                enforce_live_inputs=False,
            )
            staging_hashes_before = _stable_prepare_file_hashes(staging)
            staging_cached_probe = _cached_result(context, staging)
            staging_hashes_after = _stable_prepare_file_hashes(staging)
            require(
                staging_hashes_after == staging_hashes_before
                and all(
                    staging_cached_probe["invocation"][name] == 0
                    for name in (
                        "new_checkpoint_loads",
                        "new_inference",
                        "new_linear_control_fits",
                        "new_position_generation",
                        "new_outcome_reads",
                        "files_rewritten",
                    )
                ),
                "V59 hidden-staging cached replay was not byte-identical and zero-work",
            )
            output.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, output)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise

    hashes_before_replay = _stable_prepare_file_hashes(output)
    cached_probe = _cached_result(context, output)
    hashes_after_replay = _stable_prepare_file_hashes(output)
    require(
        hashes_after_replay == hashes_before_replay,
        "V59 cached prepare replay rewrote or changed a frozen artifact",
    )
    verification = cached_probe["verification"]
    verification["cached_replay_probe"] = {
        "passed": True,
        "file_count": len(hashes_before_replay),
        "file_hash_map_sha256": canonical_sha256(hashes_before_replay),
        "new_checkpoint_loads": cached_probe["invocation"]["new_checkpoint_loads"],
        "new_inference": cached_probe["invocation"]["new_inference"],
        "new_linear_control_fits": cached_probe["invocation"][
            "new_linear_control_fits"
        ],
        "new_position_generation": cached_probe["invocation"][
            "new_position_generation"
        ],
        "new_outcome_reads": cached_probe["invocation"]["new_outcome_reads"],
        "files_rewritten": cached_probe["invocation"]["files_rewritten"],
    }
    behavior = load_json(output / "behavior_audit.json", "behavior audit")
    ledger = behavior["operation_ledger"]
    return {
        "version": VERSION,
        "decision": PREPARE_DECISION,
        "audit": {"passed": True},
        "summary": {
            "cell_count": ledger["cells"],
            "checkpoint_count": ledger["checkpoint_loads"],
            "prediction_rows": ledger["prediction_rows"],
            "candidate_position_rows": ledger["candidate_position_rows"],
            "control_position_rows": ledger["control_position_rows"],
            "development_outcome_value_reads": 0,
            "performance_metrics_computed": 0,
            "pnl_evaluations": 0,
        },
        "invocation": {
            "cached": False,
            "new_checkpoint_loads": 36,
            "new_inference": ledger["prediction_rows"] // 3 * 3,
            "new_linear_control_fits": 12,
            "new_position_generation": ledger["candidate_position_rows"]
            + ledger["control_position_rows"],
            "new_outcome_reads": 0,
            "files_rewritten": 13,
        },
        "verification": verification,
    }
