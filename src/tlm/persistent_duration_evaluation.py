"""Outcome-blind V78 persistent-duration evaluation preparation.

This module may read only registered feature/readiness projections and the nine
V77 checkpoint containers. It freezes predictions and positions, but never
reads raw opens, labels, realized returns, PnL, or target assets.
"""

from __future__ import annotations

from dataclasses import asdict
from itertools import combinations
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import torch
import yaml

from .core import canonical_sha256, file_sha256, write_json_atomic, write_yaml_atomic
from .persistent_duration_policy import CASH, persistent_horizon_edges, stateful_persistent_actions
from .persistent_duration_training_data import (
    BASE_FEATURES,
    RELATIVE_SOURCE,
    TARGET_SYMBOLS,
    V77FeatureScaler,
)
from .persistent_duration_training_engine import FINAL_FORMAT, instantiate_v77_model
from .research_workflow import validate_research_state
from .state_conditioned_multi_horizon_training_engine import semantic_state_sha256


PREPARE_ACTION = "authorize_v78_outcome_blind_persistent_duration_evaluation_prepare_only"
WAIT_ACTION = "await_explicit_v79_registered_non_target_outcome_unseal_authorization"
SIGNAL_START = pd.Timestamp("2025-01-01", tz="UTC")
SIGNAL_END = pd.Timestamp("2025-12-23", tz="UTC")
SIGNAL_DATES = pd.date_range(SIGNAL_START, SIGNAL_END, freq="D", tz="UTC")
SEEDS = (42, 7, 123)
HORIZONS = (1, 3, 7)
HORIZON_WEIGHTS = (0.2, 0.3, 0.5)
BASE_COST = 0.001
TARGETS = frozenset(TARGET_SYMBOLS)
PANEL_COLUMNS = ("date", "symbol", *BASE_FEATURES)
ROLE_COLUMNS = (
    "date",
    "sequence_start_date",
    "symbol",
    "eligible_adaptive_development_evaluation",
)
PREDICTION_COLUMNS = (
    "fold",
    "triplet_id",
    "signal_date",
    "symbol",
    "seed",
    "gross_location_h1",
    "gross_location_h3",
    "gross_location_h7",
    "gross_scale_h1",
    "gross_scale_h3",
    "gross_scale_h7",
    "survival_h1",
    "survival_h3",
    "survival_h7",
    "expected_holding_days",
    "persistent_edge",
    "ensemble_seed_disagreement",
)


class V78EvaluationError(RuntimeError):
    pass


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V78EvaluationError(f"Unable to read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise V78EvaluationError(f"{label} must be a JSON object")
    return value


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _registered_hash(value: dict[str, Any], field: str) -> str:
    body = dict(value)
    registered = body.pop(field, None)
    observed = canonical_sha256(body)
    if registered != observed:
        raise V78EvaluationError(f"Registered canonical hash drift: {field}")
    return str(registered)


def _git_receipt(root: Path, require_clean: bool) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if require_clean and status:
        raise V78EvaluationError("V78 requires a clean committed source tree")
    return {"git_head": head, "git_clean": not bool(status)}


def _context(config: dict[str, Any]) -> dict[str, Any]:
    spec = config.get("persistent_duration_evaluation")
    if not isinstance(spec, dict) or spec.get("version") != "v78":
        raise V78EvaluationError("Missing frozen V78 evaluation config")
    root = Path(spec.get("project_root", ".")).resolve()
    status = validate_research_state(root, spec["research_state"])
    if (
        status.get("passed") is not True
        or status.get("authorized_phase") != "v78"
        or status.get("authorized_next_action") != PREPARE_ACTION
    ):
        raise V78EvaluationError("V78 prepare authorization is not active")
    contract_path = root / spec["phase_contract"]
    contract_hash = file_sha256(contract_path)
    current_path = root / spec["research_state"]
    current = yaml.safe_load(current_path.read_text(encoding="utf-8"))
    if current.get("phase_contract") != {
        "path": spec["phase_contract"],
        "file_sha256": contract_hash,
    }:
        raise V78EvaluationError("V78 live phase-contract reference drift")
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if (
        contract.get("phase") != "v78"
        or contract.get("stage_revision")
        != "v078_outcome_blind_persistent_duration_evaluation_prepare_r3"
    ):
        raise V78EvaluationError("V78 phase revision drift")

    expected = contract["input_contract"]["expected_file_sha256_by_path"]
    if set(expected) != set(contract["access_contract"]["allowed_inputs"]):
        raise V78EvaluationError("V78 input allowlist drift")
    observed: dict[str, str] = {}
    for relative, expected_hash in expected.items():
        path = root / relative
        observed[relative] = file_sha256(path)
        if observed[relative] != expected_hash:
            raise V78EvaluationError(f"V78 input hash drift: {relative}")
    if "data/processed/persistent_duration_labels_v76.parquet" in observed:
        raise V78EvaluationError("V78 prepare cannot register a label table")

    metadata = {
        name: _load_json(root / relative, name)
        for name, relative in spec["inputs"].items()
        if Path(relative).suffix == ".json"
    }
    blueprint = metadata["v74_blueprint"]
    if (
        blueprint.get("candidate_family_id") != contract["family_id"]
        or _registered_hash(blueprint, "blueprint_sha256")
        != contract["input_contract"]["expected_canonical_sha256"]["v74_blueprint"]
    ):
        raise V78EvaluationError("V78 blueprint identity drift")
    v77_result = metadata["v77_result"]
    v77_audit = metadata["v77_audit"]
    v77_completion = metadata["v77_completion_receipt"]
    if (
        _registered_hash(v77_result, "result_sha256")
        != contract["input_contract"]["expected_canonical_sha256"]["v77_result"]
        or _registered_hash(v77_audit, "audit_sha256")
        != contract["input_contract"]["expected_canonical_sha256"]["v77_audit"]
        or _registered_hash(v77_completion, "completion_receipt_sha256")
        != contract["input_contract"]["expected_canonical_sha256"][
            "v77_completion_receipt"
        ]
        or v77_result.get("decision") != PREPARE_ACTION
        or v77_audit.get("passed") is not True
    ):
        raise V78EvaluationError("V78 V77 terminal receipt drift")

    git = _git_receipt(root, bool(spec.get("require_clean_git", True)))
    source_files = list(spec["source_receipt_files"])
    if not source_files or len(source_files) != len(set(source_files)):
        raise V78EvaluationError("V78 source receipt list is empty or duplicated")
    source_hashes = {relative: file_sha256(root / relative) for relative in source_files}
    source_receipt = {
        "schema_version": "v78-source-receipt/v1",
        **git,
        "files": source_hashes,
        "bundle_sha256": canonical_sha256(source_hashes),
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "pyarrow": pa.__version__,
        },
    }
    return {
        "root": root,
        "spec": spec,
        "contract": contract,
        "contract_path": contract_path,
        "contract_hash": contract_hash,
        "current": current,
        "current_path": current_path,
        "status": status,
        "metadata": metadata,
        "blueprint": blueprint,
        "input_hashes": observed,
        "source_receipt": source_receipt,
        "output": root / config["output_dir"],
        "prediction_path": root / config["prediction_path"],
        "candidate_path": root / config["candidate_position_path"],
        "control_path": root / config["control_position_path"],
        "resolved_config": config,
    }


class V78TensorStore:
    def __init__(self, panel: pd.DataFrame, *, lookback_days: int = 256) -> None:
        self.lookback_days = int(lookback_days)
        self.symbols = tuple(sorted(str(value) for value in panel["symbol"].unique()))
        self.symbol_to_index = {value: index for index, value in enumerate(self.symbols)}
        self.dates = pd.DatetimeIndex(sorted(pd.to_datetime(panel["date"], utc=True).unique()))
        self.date_to_index = {pd.Timestamp(value): index for index, value in enumerate(self.dates)}
        self.values = np.full(
            (len(self.symbols), len(self.dates), len(BASE_FEATURES)),
            np.nan,
            dtype=np.float32,
        )
        for symbol, frame in panel.groupby("symbol", sort=True):
            indexes = np.asarray(
                [self.date_to_index[pd.Timestamp(value)] for value in frame["date"]],
                dtype=np.int64,
            )
            self.values[self.symbol_to_index[str(symbol)], indexes] = frame[
                list(BASE_FEATURES)
            ].to_numpy(dtype=np.float32)

    def context_ready(self, date: pd.Timestamp, symbol: str) -> bool:
        end = self.date_to_index.get(pd.Timestamp(date))
        if end is None or end < self.lookback_days - 1 or symbol not in self.symbol_to_index:
            return False
        start = end - self.lookback_days + 1
        values = self.values[self.symbol_to_index[symbol], start : end + 1]
        return len(values) == self.lookback_days and bool(np.isfinite(values).all())

    def materialize(
        self,
        episodes: Iterable[tuple[str, pd.Timestamp, tuple[str, str, str]]],
        scaler: V77FeatureScaler,
    ) -> np.ndarray:
        values = list(episodes)
        if not values:
            raise V78EvaluationError("Cannot materialize an empty V78 batch")
        assets = np.asarray(
            [[self.symbol_to_index[symbol] for symbol in row[2]] for row in values],
            dtype=np.int64,
        )
        ends = np.asarray([self.date_to_index[pd.Timestamp(row[1])] for row in values])
        times = ends[:, None] + np.arange(-self.lookback_days + 1, 1)[None, :]
        base = self.values[assets[:, None, :], times[:, :, None], :]
        if base.shape[1:] != (256, 3, 8) or not np.isfinite(base).all():
            raise V78EvaluationError("V78 eligible context is incomplete or non-finite")
        relative = base[..., BASE_FEATURES.index(RELATIVE_SOURCE)]
        relative = relative - relative.mean(axis=2, keepdims=True)
        return scaler.transform(
            np.concatenate([base, relative[..., None]], axis=-1)
        ).astype(np.float32, copy=False)

    def momentum_30(self, date: pd.Timestamp, triplet: tuple[str, str, str]) -> np.ndarray:
        end = self.date_to_index.get(pd.Timestamp(date))
        if end is None or end < 29:
            return np.full(3, np.nan, dtype=np.float64)
        source = BASE_FEATURES.index("log_close_to_close_return")
        result = np.asarray(
            [
                self.values[self.symbol_to_index[symbol], end - 29 : end + 1, source].sum(
                    dtype=np.float64
                )
                for symbol in triplet
            ],
            dtype=np.float64,
        )
        return result


def _read_projected(
    path: Path,
    columns: tuple[str, ...],
    symbols: tuple[str, ...],
    *,
    minimum_date: pd.Timestamp,
    maximum_date: pd.Timestamp,
) -> pd.DataFrame:
    predicate = (
        ds.field("symbol").isin(list(symbols))
        & (ds.field("date") >= minimum_date.to_pydatetime())
        & (ds.field("date") <= maximum_date.to_pydatetime())
    )
    table = ds.dataset(path, format="parquet").to_table(
        columns=list(columns), filter=predicate, use_threads=False
    )
    frame = table.to_pandas()
    if list(frame.columns) != list(columns):
        raise V78EvaluationError(f"V78 projection drift: {path}")
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame["symbol"] = frame["symbol"].astype(str)
    if frame.duplicated(["date", "symbol"]).any():
        raise V78EvaluationError(f"Duplicate V78 date/symbol key: {path}")
    loaded_symbols = set(frame["symbol"].unique())
    if not loaded_symbols or not loaded_symbols.issubset(set(symbols)):
        raise V78EvaluationError(f"V78 fold symbol predicate drift: {path}")
    if set(frame["symbol"]).intersection(TARGETS):
        raise V78EvaluationError("V78 target asset entered a projected frame")
    if not frame.empty and (
        frame["date"].min() < minimum_date or frame["date"].max() > maximum_date
    ):
        raise V78EvaluationError(f"V78 date predicate drift: {path}")
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def _scaler_from_record(value: dict[str, Any]) -> V77FeatureScaler:
    return V77FeatureScaler(
        feature_names=tuple(value["feature_names"]),
        mean=tuple(float(item) for item in value["mean"]),
        scale=tuple(float(item) for item in value["scale"]),
        source_relative_feature_index=int(value["source_relative_feature_index"]),
        fit_scope=str(value["fit_scope"]),
        fit_start=str(value["fit_start"]),
        fit_end=str(value["fit_end"]),
        fit_rows=int(value["fit_rows"]),
    )


def _read_fold_data(context: dict[str, Any], fold: int) -> dict[str, Any]:
    root = context["root"]
    asset_entry = next(
        row for row in context["metadata"]["v32_asset_folds"]["folds"]
        if int(row["fold"]) == int(fold)
    )
    catalog_entry = next(
        row for row in context["metadata"]["v32_triplet_catalog"]["folds"]
        if int(row["fold"]) == int(fold)
    )
    symbols = tuple(sorted(str(value) for value in asset_entry["test_symbols"]))
    triplets = tuple(tuple(str(symbol) for symbol in row) for row in catalog_entry["test_triplets"])
    if (
        len(symbols) != 10
        or len(triplets) != 120
        or triplets != tuple(combinations(symbols, 3))
        or set(symbols).intersection(TARGETS)
    ):
        raise V78EvaluationError(f"V78 fold {fold} heldout triplet scope drift")

    roles_path = root / "data/processed/persistent_duration_sequence_roles_v76.parquet"
    roles = _read_projected(
        roles_path,
        ROLE_COLUMNS,
        symbols,
        minimum_date=SIGNAL_START,
        maximum_date=SIGNAL_END,
    )
    roles["sequence_start_date"] = pd.to_datetime(roles["sequence_start_date"], utc=True)
    roles = roles.loc[roles["eligible_adaptive_development_evaluation"]].reset_index(drop=True)
    if roles.empty or not roles["date"].isin(SIGNAL_DATES).all():
        raise V78EvaluationError(f"V78 fold {fold} readiness role drift")
    minimum_context = pd.Timestamp(roles["sequence_start_date"].min())
    panel_path = root / "data/processed/selected_universe_panel_v32.parquet"
    panel = _read_projected(
        panel_path,
        PANEL_COLUMNS,
        symbols,
        minimum_date=minimum_context,
        maximum_date=SIGNAL_END,
    )
    store = V78TensorStore(panel, lookback_days=256)
    ready = {(pd.Timestamp(row.date), str(row.symbol)) for row in roles.itertuples(index=False)}
    episodes: list[tuple[str, pd.Timestamp, tuple[str, str, str]]] = []
    availability: dict[str, set[pd.Timestamp]] = {}
    for index, triplet in enumerate(triplets):
        triplet_id = f"F{fold}-T{index:03d}"
        dates: set[pd.Timestamp] = set()
        for date in SIGNAL_DATES:
            if all(
                (pd.Timestamp(date), symbol) in ready
                and store.context_ready(pd.Timestamp(date), symbol)
                for symbol in triplet
            ):
                dates.add(pd.Timestamp(date))
                episodes.append((triplet_id, pd.Timestamp(date), triplet))
        availability[triplet_id] = dates
    if not episodes:
        raise V78EvaluationError(f"V78 fold {fold} has no eligible triplet episodes")

    scale_record = _load_json(
        root / f"data/checkpoints/v77_persistent_duration_training/fold_{fold}/fold_scale.json",
        f"fold {fold} scaler",
    )
    scaler = _scaler_from_record(scale_record["feature_scaler"])
    if (
        scaler.state_sha256() != scale_record["feature_scaler_state_sha256"]
        or scale_record.get("shared_across_seeds") != list(SEEDS)
        or scale_record.get("fit_role") != "eligible_train_only"
    ):
        raise V78EvaluationError(f"V78 fold {fold} scaler identity drift")
    access = {
        "fold": int(fold),
        "symbols": list(symbols),
        "triplets": len(triplets),
        "signal_dates": len(SIGNAL_DATES),
        "eligible_triplet_dates": len(episodes),
        "roles_rows": int(len(roles)),
        "panel_rows": int(len(panel)),
        "minimum_context_date": str(minimum_context.date()),
        "maximum_loaded_feature_date": str(panel["date"].max().date()),
        "panel_columns": list(PANEL_COLUMNS),
        "role_columns": list(ROLE_COLUMNS),
        "raw_open_loaded": False,
        "label_columns_loaded": [],
        "outcome_rows_read": 0,
        "target_assets_loaded": [],
        "parquet_deserializations": 2,
        "scaler_fit_performed": False,
    }
    return {
        "fold": int(fold),
        "symbols": symbols,
        "triplets": triplets,
        "episodes": tuple(episodes),
        "availability": availability,
        "store": store,
        "scaler": scaler,
        "access": access,
    }


def _validate_checkpoint_payload(
    payload: dict[str, Any], job: dict[str, Any], expected_file_hash: str
) -> None:
    body = {key: value for key, value in payload.items() if key != "semantic_checkpoint_sha256"}
    if (
        payload.get("format_version") != FINAL_FORMAT
        or payload.get("kind") != "final"
        or payload.get("stage") != "complete"
        or semantic_state_sha256(body) != payload.get("semantic_checkpoint_sha256")
        or payload.get("semantic_checkpoint_sha256") != job.get("semantic_checkpoint_sha256")
        or payload.get("job_context") != job.get("context")
        or int(payload.get("completed_epoch", 0)) != int(job.get("completed_epoch", -1))
        or int(payload.get("early_stopping", {}).get("best_epoch", 0))
        != int(job.get("best_epoch", -1))
        or not payload.get("model_best_state")
        or job.get("file_sha256") != expected_file_hash
        or payload.get("prior_checkpoint_reused") is not False
    ):
        raise V78EvaluationError(f"V78 checkpoint semantic drift: {job.get('job_id')}")


def _configure_mps() -> torch.device:
    fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0").strip().lower()
    if fallback not in {"", "0", "false", "no", "off"}:
        raise V78EvaluationError("V78 forbids MPS fallback")
    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise V78EvaluationError("V78 requires Apple MPS")
    torch.set_num_threads(10)
    torch.use_deterministic_algorithms(True)
    device = torch.device("mps")
    probe = torch.ones(4, dtype=torch.float32, device=device)
    if float((probe * 2).sum().cpu()) != 8.0:
        raise V78EvaluationError("V78 MPS probe failed")
    return device


def _load_model(
    context: dict[str, Any], fold: int, seed: int, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    manifest = context["metadata"]["v77_checkpoint_manifest"]
    job = next(
        row for row in manifest["jobs"]
        if int(row["fold"]) == int(fold) and int(row["seed"]) == int(seed)
    )
    path = context["root"] / job["path"]
    expected = context["input_hashes"][job["path"]]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise V78EvaluationError(f"V78 checkpoint is not a mapping: {job['job_id']}")
    _validate_checkpoint_payload(payload, job, expected)
    model = instantiate_v77_model(context["blueprint"], device, seed=int(seed))
    model.load_state_dict(payload["model_best_state"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    receipt = {
        "job_id": job["job_id"],
        "fold": int(fold),
        "seed": int(seed),
        "path": job["path"],
        "file_sha256": expected,
        "semantic_checkpoint_sha256": payload["semantic_checkpoint_sha256"],
        "checkpoint_state": "model_best_state",
        "best_epoch": int(job["best_epoch"]),
        "selected_or_discarded": False,
    }
    del payload
    return model, receipt


def _batched(values: tuple[Any, ...], size: int) -> Iterable[tuple[Any, ...]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _infer_fold(
    context: dict[str, Any], data: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    episodes = data["episodes"]
    count = len(episodes)
    location = np.empty((len(SEEDS), count, 3, 3), dtype=np.float64)
    scale = np.empty_like(location)
    survival = np.empty_like(location)
    expected_holding = np.empty((len(SEEDS), count, 3), dtype=np.float64)
    checkpoint_receipts: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(SEEDS):
        model, receipt = _load_model(context, data["fold"], seed, device)
        checkpoint_receipts.append(receipt)
        offset = 0
        with torch.inference_mode():
            for batch in _batched(episodes, 128):
                features = data["store"].materialize(batch, data["scaler"])
                tensor = torch.from_numpy(features).to(device=device, dtype=torch.float32)
                output = model(tensor, round_trip_cost=0.0)
                width = len(batch)
                location[seed_index, offset : offset + width] = (
                    output["gross_location"].detach().cpu().numpy().astype(np.float64)
                )
                scale[seed_index, offset : offset + width] = (
                    output["gross_scale"].detach().cpu().numpy().astype(np.float64)
                )
                survival[seed_index, offset : offset + width] = (
                    output["horizon_survival_probability"]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                expected_holding[seed_index, offset : offset + width] = (
                    output["expected_holding_days"]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                offset += width
        if offset != count:
            raise V78EvaluationError("V78 inference batch accounting drift")
        del model
        if hasattr(torch, "mps"):
            torch.mps.empty_cache()

    ensemble_location = location.mean(axis=0, dtype=np.float64)
    ensemble_scale = scale.mean(axis=0, dtype=np.float64)
    ensemble_survival = survival.mean(axis=0, dtype=np.float64)
    ensemble_expected_holding = expected_holding.mean(axis=0, dtype=np.float64)
    seed_edges = np.stack(
        [
            persistent_horizon_edges(
                location[index],
                survival[index],
                horizons=HORIZONS,
                horizon_weights=HORIZON_WEIGHTS,
            )
            for index in range(len(SEEDS))
        ],
        axis=0,
    )
    ensemble_edge = persistent_horizon_edges(
        ensemble_location,
        ensemble_survival,
        horizons=HORIZONS,
        horizon_weights=HORIZON_WEIGHTS,
    )
    disagreement = seed_edges.std(axis=0, ddof=0)
    return {
        "episodes": episodes,
        "location": location,
        "scale": scale,
        "survival": survival,
        "expected_holding": expected_holding,
        "ensemble_location": ensemble_location,
        "ensemble_scale": ensemble_scale,
        "ensemble_survival": ensemble_survival,
        "ensemble_expected_holding": ensemble_expected_holding,
        "seed_edges": seed_edges,
        "ensemble_edge": ensemble_edge,
        "disagreement": disagreement,
        "checkpoint_receipts": checkpoint_receipts,
    }


def _prediction_variant_frame(
    fold: int,
    episodes: tuple[tuple[str, pd.Timestamp, tuple[str, str, str]], ...],
    seed: str,
    location: np.ndarray,
    scale: np.ndarray,
    survival: np.ndarray,
    expected_holding: np.ndarray,
    edge: np.ndarray,
    disagreement: np.ndarray,
) -> pd.DataFrame:
    count = len(episodes)
    triplet_ids = np.repeat(np.asarray([row[0] for row in episodes], dtype=object), 3)
    dates = np.repeat(np.asarray([row[1].to_datetime64() for row in episodes]), 3)
    symbols = np.asarray([symbol for row in episodes for symbol in row[2]], dtype=object)
    data: dict[str, Any] = {
        "fold": np.full(count * 3, int(fold), dtype=np.int16),
        "triplet_id": triplet_ids,
        "signal_date": pd.to_datetime(dates, utc=True),
        "symbol": symbols,
        "seed": np.full(count * 3, str(seed), dtype=object),
    }
    for index, horizon in enumerate(HORIZONS):
        data[f"gross_location_h{horizon}"] = location[:, :, index].reshape(-1)
        data[f"gross_scale_h{horizon}"] = scale[:, :, index].reshape(-1)
        data[f"survival_h{horizon}"] = survival[:, :, index].reshape(-1)
    data["expected_holding_days"] = expected_holding.reshape(-1)
    data["persistent_edge"] = edge.reshape(-1)
    data["ensemble_seed_disagreement"] = disagreement.reshape(-1)
    frame = pd.DataFrame(data)
    return frame.loc[:, list(PREDICTION_COLUMNS)]


def _prediction_frame(fold: int, inferred: dict[str, Any]) -> pd.DataFrame:
    frames = []
    for index, seed in enumerate(SEEDS):
        frames.append(
            _prediction_variant_frame(
                fold,
                inferred["episodes"],
                str(seed),
                inferred["location"][index],
                inferred["scale"][index],
                inferred["survival"][index],
                inferred["expected_holding"][index],
                inferred["seed_edges"][index],
                inferred["disagreement"],
            )
        )
    frames.append(
        _prediction_variant_frame(
            fold,
            inferred["episodes"],
            "ensemble",
            inferred["ensemble_location"],
            inferred["ensemble_scale"],
            inferred["ensemble_survival"],
            inferred["ensemble_expected_holding"],
            inferred["ensemble_edge"],
            inferred["disagreement"],
        )
    )
    result = pd.concat(frames, ignore_index=True)
    order = {"42": 0, "7": 1, "123": 2, "ensemble": 3}
    result["_seed_order"] = result["seed"].map(order).astype(np.int8)
    return result.sort_values(
        ["fold", "triplet_id", "signal_date", "symbol", "_seed_order"],
        kind="mergesort",
    ).drop(columns="_seed_order").reset_index(drop=True)


def _transition_ledger(positions: np.ndarray) -> tuple[np.ndarray, float, float]:
    value = np.asarray(positions, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("Positions must be finite [days,assets]")
    prior = np.vstack([np.zeros((1, value.shape[1])), value[:-1]])
    turnover = np.abs(value - prior).sum(axis=1)
    liquidation = float(np.abs(value[-1]).sum()) if len(value) else 0.0
    return turnover, liquidation, float(turnover.sum() + liquidation)


def _action_labels(positions: np.ndarray) -> tuple[list[str], list[str]]:
    prior = np.zeros(positions.shape[1], dtype=np.float64)
    labels: list[str] = []
    selected: list[str] = []
    for current in positions:
        prior_index = int(np.argmax(prior)) if prior.sum() > 0 else CASH
        current_index = int(np.argmax(current)) if current.sum() > 0 else CASH
        if prior_index == CASH and current_index == CASH:
            labels.append("cash")
        elif prior_index == CASH:
            labels.append("enter")
        elif current_index == CASH:
            labels.append("exit")
        elif prior_index == current_index:
            labels.append("hold")
        else:
            labels.append("switch")
        selected.append("CASH" if current_index == CASH else str(current_index))
        prior = current
    return labels, selected


def _candidate_for_triplet(
    fold: int,
    triplet_id: str,
    triplet: tuple[str, str, str],
    eligible_dates: set[pd.Timestamp],
    edge_by_date: dict[pd.Timestamp, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    edges = np.full((len(SIGNAL_DATES), 3), -1.0e9, dtype=np.float64)
    eligible = np.zeros(len(SIGNAL_DATES), dtype=np.bool_)
    for index, date in enumerate(SIGNAL_DATES):
        key = pd.Timestamp(date)
        if key in eligible_dates:
            edge = edge_by_date.get(key)
            if edge is None or edge.shape != (3,) or not np.isfinite(edge).all():
                raise V78EvaluationError(f"Missing V78 edge: {triplet_id} {key.date()}")
            edges[index] = edge
            eligible[index] = True
    policy = stateful_persistent_actions(
        edges,
        base_cost=BASE_COST,
        risky_gross=1.0,
        initial_action=CASH,
        tie_tolerance=1e-12,
        final_liquidation=True,
    )
    positions = np.asarray(policy["positions"], dtype=np.float64)
    if np.any(positions[~eligible] != 0.0):
        raise V78EvaluationError("V78 unavailable triplet did not force cash")
    selected_symbols = [
        "CASH" if index == CASH else triplet[int(index)]
        for index in policy["selected_assets"]
    ]
    frame = pd.DataFrame(
        {
            "fold": np.repeat(np.int16(fold), len(SIGNAL_DATES) * 3),
            "triplet_id": np.repeat(triplet_id, len(SIGNAL_DATES) * 3),
            "signal_date": np.repeat(SIGNAL_DATES.to_numpy(), 3),
            "symbol": np.tile(np.asarray(triplet, dtype=object), len(SIGNAL_DATES)),
            "eligible": np.repeat(eligible, 3),
            "weight": positions.reshape(-1),
            "action": np.repeat(np.asarray(policy["actions"], dtype=object), 3),
            "selected_symbol": np.repeat(np.asarray(selected_symbols, dtype=object), 3),
            "transition_turnover": np.repeat(
                np.asarray(policy["turnover"], dtype=np.float64), 3
            ),
            "transaction_cost_10bps": np.repeat(
                np.asarray(policy["transaction_costs"], dtype=np.float64), 3
            ),
            "final_liquidation_turnover": np.repeat(
                np.asarray(
                    [0.0] * (len(SIGNAL_DATES) - 1)
                    + [float(policy["final_liquidation_turnover"])],
                    dtype=np.float64,
                ),
                3,
            ),
        }
    )
    transition, liquidation, total = _transition_ledger(positions)
    check = {
        "fold": int(fold),
        "triplet_id": triplet_id,
        "days": len(SIGNAL_DATES),
        "eligible_days": int(eligible.sum()),
        "unavailable_cash_exact": bool(np.all(positions[~eligible] == 0.0)),
        "one_hot_or_cash": bool(
            np.all((positions == 0.0) | (positions == 1.0))
            and np.all(positions.sum(axis=1) <= 1.0)
        ),
        "turnover_exact": bool(
            np.allclose(transition, policy["turnover"], rtol=0.0, atol=1e-12)
            and math.isclose(
                liquidation,
                float(policy["final_liquidation_turnover"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(total, float(policy["total_turnover"]), rel_tol=0.0, abs_tol=1e-12)
        ),
        "total_turnover": total,
    }
    return frame, check


def _control_for_triplet(
    fold: int,
    triplet_id: str,
    triplet: tuple[str, str, str],
    eligible_dates: set[pd.Timestamp],
    store: V78TensorStore,
    control: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    positions = np.zeros((len(SIGNAL_DATES), 3), dtype=np.float64)
    if control == "daily_equal_weight_eligible_assets":
        for index, date in enumerate(SIGNAL_DATES):
            if pd.Timestamp(date) in eligible_dates:
                positions[index] = 1.0 / 3.0
    elif control == "weekly_dual_momentum_30_long_one_or_cash":
        desired = CASH
        eligible_index = 0
        for index, date in enumerate(SIGNAL_DATES):
            key = pd.Timestamp(date)
            if key not in eligible_dates:
                continue
            if eligible_index % 7 == 0:
                momentum = store.momentum_30(key, triplet)
                if np.isfinite(momentum).all():
                    best = int(np.argmax(momentum))
                    desired = best if float(momentum[best]) > 0.0 else CASH
                else:
                    desired = CASH
            if desired != CASH:
                positions[index, desired] = 1.0
            eligible_index += 1
    elif control != "cash":
        raise ValueError(f"Unknown V78 control: {control}")
    turnover, liquidation, total = _transition_ledger(positions)
    actions, selected_indexes = _action_labels(positions)
    selected = [
        "CASH" if value == "CASH" else triplet[int(value)]
        for value in selected_indexes
    ]
    frame = pd.DataFrame(
        {
            "fold": np.repeat(np.int16(fold), len(SIGNAL_DATES) * 3),
            "triplet_id": np.repeat(triplet_id, len(SIGNAL_DATES) * 3),
            "control": np.repeat(control, len(SIGNAL_DATES) * 3),
            "signal_date": np.repeat(SIGNAL_DATES.to_numpy(), 3),
            "symbol": np.tile(np.asarray(triplet, dtype=object), len(SIGNAL_DATES)),
            "eligible": np.repeat(
                np.asarray([pd.Timestamp(value) in eligible_dates for value in SIGNAL_DATES]),
                3,
            ),
            "weight": positions.reshape(-1),
            "action": np.repeat(np.asarray(actions, dtype=object), 3),
            "selected_symbol": np.repeat(np.asarray(selected, dtype=object), 3),
            "transition_turnover": np.repeat(turnover, 3),
            "final_liquidation_turnover": np.repeat(
                np.asarray([0.0] * (len(SIGNAL_DATES) - 1) + [liquidation]), 3
            ),
        }
    )
    check = {
        "control": control,
        "fold": int(fold),
        "triplet_id": triplet_id,
        "unavailable_cash_exact": bool(
            np.all(
                positions[
                    np.asarray(
                        [pd.Timestamp(value) not in eligible_dates for value in SIGNAL_DATES]
                    )
                ]
                == 0.0
            )
        ),
        "weights_exact": bool(
            np.isfinite(positions).all()
            and np.all(positions >= 0.0)
            and np.all(positions.sum(axis=1) <= 1.0 + 1e-12)
        ),
        "total_turnover": total,
    }
    if control == "cash":
        check["weights_exact"] = bool(np.all(positions == 0.0))
    if control == "daily_equal_weight_eligible_assets":
        eligible_mask = np.asarray(
            [pd.Timestamp(value) in eligible_dates for value in SIGNAL_DATES]
        )
        check["weights_exact"] = bool(
            np.allclose(positions[eligible_mask], 1.0 / 3.0)
            and np.all(positions[~eligible_mask] == 0.0)
        )
    return frame, check


def _build_position_frames(
    fold_data: list[dict[str, Any]], inferred: list[dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    candidate_frames: list[pd.DataFrame] = []
    control_frames: list[pd.DataFrame] = []
    candidate_checks: list[dict[str, Any]] = []
    control_checks: list[dict[str, Any]] = []
    for data, output in zip(fold_data, inferred, strict=True):
        edge_maps: dict[str, dict[pd.Timestamp, np.ndarray]] = {
            f"F{data['fold']}-T{index:03d}": {}
            for index in range(len(data["triplets"]))
        }
        for episode_index, episode in enumerate(output["episodes"]):
            edge_maps[episode[0]][pd.Timestamp(episode[1])] = output["ensemble_edge"][
                episode_index
            ]
        for index, triplet in enumerate(data["triplets"]):
            triplet_id = f"F{data['fold']}-T{index:03d}"
            candidate, candidate_check = _candidate_for_triplet(
                data["fold"],
                triplet_id,
                triplet,
                data["availability"][triplet_id],
                edge_maps[triplet_id],
            )
            candidate_frames.append(candidate)
            candidate_checks.append(candidate_check)
            for control in (
                "cash",
                "weekly_dual_momentum_30_long_one_or_cash",
                "daily_equal_weight_eligible_assets",
            ):
                frame, check = _control_for_triplet(
                    data["fold"],
                    triplet_id,
                    triplet,
                    data["availability"][triplet_id],
                    data["store"],
                    control,
                )
                control_frames.append(frame)
                control_checks.append(check)
    candidate_frame = pd.concat(candidate_frames, ignore_index=True).sort_values(
        ["fold", "triplet_id", "signal_date", "symbol"], kind="mergesort"
    ).reset_index(drop=True)
    control_frame = pd.concat(control_frames, ignore_index=True).sort_values(
        ["control", "fold", "triplet_id", "signal_date", "symbol"],
        kind="mergesort",
    ).reset_index(drop=True)
    return candidate_frame, control_frame, {
        "candidate_checks": candidate_checks,
        "control_checks": control_checks,
        "aggregate_turnover": float(
            np.mean([row["total_turnover"] for row in candidate_checks], dtype=np.float64)
        ),
    }


def _behavior_gates(
    context: dict[str, Any],
    fold_data: list[dict[str, Any]],
    inferred: list[dict[str, Any]],
    predictions: pd.DataFrame,
    candidate: pd.DataFrame,
    controls: pd.DataFrame,
    position_checks: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_receipts = [
        receipt for output in inferred for receipt in output["checkpoint_receipts"]
    ]
    numeric = predictions.loc[
        :, [column for column in PREDICTION_COLUMNS if column not in {
            "fold", "triplet_id", "signal_date", "symbol", "seed"
        }]
    ].to_numpy(dtype=np.float64)
    survival = predictions[["survival_h1", "survival_h3", "survival_h7"]].to_numpy(
        dtype=np.float64
    )
    scale = predictions[["gross_scale_h1", "gross_scale_h3", "gross_scale_h7"]].to_numpy(
        dtype=np.float64
    )
    expected_candidate_rows = 3 * 120 * len(SIGNAL_DATES) * 3
    expected_control_rows = expected_candidate_rows * 3
    exact_triplets = all(
        data["triplets"] == tuple(combinations(data["symbols"], 3))
        for data in fold_data
    )
    candidate_checks = position_checks["candidate_checks"]
    control_checks = position_checks["control_checks"]
    gate_values = {
        "all_registered_checkpoints_used_without_selection": (
            len(checkpoint_receipts) == 9
            and {(row["fold"], row["seed"]) for row in checkpoint_receipts}
            == {(fold, seed) for fold in (1, 2, 3) for seed in SEEDS}
            and all(row["selected_or_discarded"] is False for row in checkpoint_receipts)
        ),
        "exact_fold_triplet_date_scope": (
            len(fold_data) == 3
            and all(len(data["triplets"]) == 120 for data in fold_data)
            and len(candidate) == expected_candidate_rows
            and len(controls) == expected_control_rows
            and all(row["days"] == 357 for row in candidate_checks)
        ),
        "missingness_matches_registered_readiness": (
            all(row["unavailable_cash_exact"] for row in candidate_checks)
            and all(row["unavailable_cash_exact"] for row in control_checks)
            and all(data["access"]["maximum_loaded_feature_date"] == "2025-12-23" for data in fold_data)
        ),
        "prediction_distribution_finite_and_nonconstant": (
            bool(np.isfinite(numeric).all())
            and float(predictions["persistent_edge"].std(ddof=0)) > 0.0
            and float(predictions["gross_location_h1"].std(ddof=0)) > 0.0
        ),
        "positive_scale_and_valid_survival": (
            bool(np.all(scale > 0.0))
            and bool(np.all((survival >= 0.0) & (survival <= 1.0)))
            and bool(np.all(survival[:, 0] >= survival[:, 1]))
            and bool(np.all(survival[:, 1] >= survival[:, 2]))
        ),
        "ensemble_disagreement_finite": (
            bool(np.isfinite(predictions["ensemble_seed_disagreement"]).all())
            and bool((predictions["ensemble_seed_disagreement"] >= 0.0).all())
        ),
        "permutation_and_lexical_structure_complete": (
            exact_triplets
            and context["blueprint"]["architecture"]["asset_slot_embedding"] is False
            and context["blueprint"]["architecture"]["asset_permutation_contract"]
            == "equivariant_asset_outputs_invariant_market_outputs"
        ),
        "action_space_and_state_transitions_exact": (
            all(row["one_hot_or_cash"] for row in candidate_checks)
            and set(candidate["action"].unique()).issubset(
                {"cash", "enter", "hold", "exit", "switch"}
            )
            and set(candidate["selected_symbol"]).issubset(
                set().union(*(set(data["symbols"]) for data in fold_data), {"CASH"})
            )
        ),
        "turnover_and_final_liquidation_exact": all(
            row["turnover_exact"] for row in candidate_checks
        ),
        "control_positions_exact": (
            len(control_checks) == 1080
            and all(row["weights_exact"] for row in control_checks)
            and set(controls["control"].unique())
            == {
                "cash",
                "weekly_dual_momentum_30_long_one_or_cash",
                "daily_equal_weight_eligible_assets",
            }
        ),
        "aggregate_turnover_within_registered_ceiling": position_checks[
            "aggregate_turnover"
        ]
        <= 45.0,
        "zero_outcome_and_target_access": (
            all(data["access"]["outcome_rows_read"] == 0 for data in fold_data)
            and all(data["access"]["target_assets_loaded"] == [] for data in fold_data)
            and not set(predictions["symbol"]).intersection(TARGETS)
            and not set(candidate["symbol"]).intersection(TARGETS)
            and not set(controls["symbol"]).intersection(TARGETS)
        ),
    }
    expected_names = context["contract"]["outcome_blind_gate_contract"]["gates"]
    if list(gate_values) != list(expected_names):
        raise V78EvaluationError("V78 behavior-gate name/order drift")
    return {
        "schema_version": "v78-behavior-audit/v1",
        "passed": all(gate_values.values()),
        "gates": gate_values,
        "diagnostics": {
            "checkpoint_count": len(checkpoint_receipts),
            "prediction_rows": int(len(predictions)),
            "candidate_position_rows": int(len(candidate)),
            "control_position_rows": int(len(controls)),
            "eligible_triplet_dates_by_fold": {
                str(data["fold"]): len(data["episodes"]) for data in fold_data
            },
            "aggregate_candidate_turnover": position_checks["aggregate_turnover"],
            "prediction_persistent_edge_std": float(
                predictions["persistent_edge"].std(ddof=0)
            ),
            "ensemble_disagreement_mean": float(
                predictions["ensemble_seed_disagreement"].mean()
            ),
            "outcome_rows_read": 0,
            "target_assets_loaded": [],
        },
        "checkpoint_receipts": checkpoint_receipts,
    }


def _write_parquet(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(
        table,
        temporary,
        compression="zstd",
        version="2.6",
        use_dictionary=False,
        write_statistics=True,
    )
    temporary.replace(path)
    return {
        "path": str(path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "sha256": file_sha256(path),
    }


def _artifact_ref(root: Path, path: Path, kind: str) -> dict[str, Any]:
    return {"kind": kind, "path": _relative(root, path), "sha256": file_sha256(path)}


def _registered_packet_contract(contract: dict[str, Any]) -> dict[str, Any]:
    bound = {
        "cost_bps": list(contract["policy_contract"]["reporting_cost_bps"]),
        "accounting": contract["registered_accounting"],
        "controls": contract["registered_controls"],
        "gates": contract["registered_outcome_dependent_gates"],
        "outcome_blind_gate_names": contract["outcome_blind_gate_contract"]["gates"],
    }
    return {**bound, "sha256": canonical_sha256(bound)}


def _validate_one_shot_packet(context: dict[str, Any], packet_path: Path) -> dict[str, Any]:
    script = (
        context["root"]
        / ".agents/skills/tlm-one-shot-evaluator/scripts/validate_evaluation_packet.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--repo-root",
            str(context["root"]),
            "--packet",
            str(packet_path),
        ],
        cwd=context["root"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise V78EvaluationError(
            f"V78 one-shot packet validation failed: {result.stderr.strip()}"
        )
    value = json.loads(result.stdout)
    if value.get("valid") is not True or value.get("outcomes_sealed") is not True:
        raise V78EvaluationError("V78 one-shot validator did not preserve sealing")
    return value


def prepare_persistent_duration_evaluation(config: dict[str, Any]) -> dict[str, Any]:
    context = _context(config)
    output = context["output"]
    output.mkdir(parents=True, exist_ok=True)
    for path in (context["prediction_path"], context["candidate_path"], context["control_path"]):
        if path.exists():
            raise V78EvaluationError(
                f"V78 frozen data already exists; use hash-only replay, not regeneration: {path}"
            )

    registered = _registered_packet_contract(context["contract"])
    evaluation_spec = {
        "schema_version": "v78-evaluation-spec/v1",
        "frozen": True,
        "family_id": context["contract"]["family_id"],
        "evidence_tier": context["contract"]["evidence_tier"],
        "phase_contract": {
            "path": _relative(context["root"], context["contract_path"]),
            "sha256": context["contract_hash"],
        },
        "evaluation": context["contract"]["evaluation_contract"],
        "policy": context["contract"]["policy_contract"],
        "registered": registered,
        "bootstrap": context["contract"]["registered_bootstrap"],
        "outcome_request": context["contract"]["outcome_request_contract"],
        "one_shot": context["contract"]["one_shot_contract"],
        "target_contract": context["contract"]["target_contract"],
    }
    evaluation_spec["evaluation_spec_sha256"] = canonical_sha256(evaluation_spec)
    write_json_atomic(output / "evaluation_spec.json", evaluation_spec)
    write_json_atomic(output / "source_receipt.json", context["source_receipt"])
    input_receipt = {
        "schema_version": "v78-input-hash-receipt/v1",
        "files": context["input_hashes"],
        "bundle_sha256": canonical_sha256(context["input_hashes"]),
        "outcome_table_registered_for_prepare": False,
    }
    input_receipt["receipt_sha256"] = canonical_sha256(input_receipt)
    write_json_atomic(output / "input_hash_receipt.json", input_receipt)
    write_yaml_atomic(output / "resolved_config.yaml", config)

    device = _configure_mps()
    fold_data: list[dict[str, Any]] = []
    inferred: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for fold in (1, 2, 3):
        data = _read_fold_data(context, fold)
        output_values = _infer_fold(context, data, device)
        fold_data.append(data)
        inferred.append(output_values)
        prediction_frames.append(_prediction_frame(fold, output_values))
    predictions = pd.concat(prediction_frames, ignore_index=True).sort_values(
        ["fold", "triplet_id", "signal_date", "symbol", "seed"], kind="mergesort"
    ).reset_index(drop=True)
    candidate, controls, position_checks = _build_position_frames(fold_data, inferred)
    behavior = _behavior_gates(
        context, fold_data, inferred, predictions, candidate, controls, position_checks
    )

    prediction_manifest = _write_parquet(context["prediction_path"], predictions)
    candidate_manifest = _write_parquet(context["candidate_path"], candidate)
    control_manifest = _write_parquet(context["control_path"], controls)
    for name, value in (
        ("predictions_manifest.json", prediction_manifest),
        ("candidate_positions_manifest.json", candidate_manifest),
        ("control_positions_manifest.json", control_manifest),
    ):
        relative_value = dict(value)
        relative_value["path"] = _relative(context["root"], Path(value["path"]))
        relative_value["manifest_sha256"] = canonical_sha256(relative_value)
        write_json_atomic(output / name, relative_value)
    write_json_atomic(output / "behavior_audit.json", behavior)
    data_access = {
        "schema_version": "v78-data-access/v1",
        "folds": [data["access"] for data in fold_data],
        "parquet_deserializations": 6,
        "checkpoint_container_deserializations": 9,
        "checkpoint_state": "model_best_state",
        "checkpoint_writes": 0,
        "scaler_fits": 0,
        "optimizer_steps": 0,
        "predictions_written": int(len(predictions)),
        "candidate_positions_written": int(len(candidate)),
        "control_positions_written": int(len(controls)),
        "performance_metrics_computed": 0,
        "pnl_evaluations": 0,
        "bootstrap_paths": 0,
        "outcome_rows_read": 0,
        "raw_open_columns_read": 0,
        "target_assets_loaded": [],
    }
    data_access["access_sha256"] = canonical_sha256(data_access)
    write_json_atomic(output / "data_access.json", data_access)
    outcome_request = {
        "schema_version": "v78-outcome-request/v1",
        "status": "sealed_not_authorized",
        "evaluation_spec_path": _relative(context["root"], output / "evaluation_spec.json"),
        "evaluation_spec_sha256": file_sha256(output / "evaluation_spec.json"),
        "prepare_source_read_count": 0,
        "request": context["contract"]["outcome_request_contract"],
        "explicit_user_authorization_required": True,
        "generic_continue_is_not_authorization": True,
    }
    outcome_request["outcome_request_sha256"] = canonical_sha256(outcome_request)
    write_json_atomic(output / "outcome_request.json", outcome_request)

    if not behavior["passed"]:
        result = {
            "schema_version": "v78-prepare-result/v1",
            "decision": "keep_v78_outcomes_sealed_behavior_gate_failed",
            "audit": behavior,
            "outcome_rows_read": 0,
            "target_assets_loaded": [],
        }
        result["result_sha256"] = canonical_sha256(result)
        write_json_atomic(output / "result.json", result)
        raise V78EvaluationError(
            "V78 outcome-blind behavior gate failed: "
            + json.dumps(
                {name: passed for name, passed in behavior["gates"].items() if not passed},
                sort_keys=True,
            )
        )

    artifacts = [
        _artifact_ref(context["root"], context["prediction_path"], "predictions"),
        _artifact_ref(context["root"], context["candidate_path"], "positions"),
        _artifact_ref(context["root"], context["control_path"], "positions"),
        _artifact_ref(context["root"], output / "behavior_audit.json", "behavior_audit"),
    ]
    artifact_hashes = {row["path"]: row["sha256"] for row in artifacts}
    prepare_receipt = {
        "schema_version": "tlm-one-shot-prepare/v1",
        "evaluation_spec_sha256": file_sha256(output / "evaluation_spec.json"),
        "registered_sha256": registered["sha256"],
        "artifact_hashes": artifact_hashes,
        "outcome_rows_read": 0,
        "outcome_blind_gates_passed": True,
        "authorizes_unseal": True,
        "predictions_frozen": True,
        "positions_frozen": True,
        "all_checkpoints_used_without_selection": True,
        "target_assets_loaded": [],
    }
    prepare_receipt["prepare_receipt_sha256"] = canonical_sha256(prepare_receipt)
    write_json_atomic(output / "prepare_receipt.json", prepare_receipt)

    packet = {
        "schema_version": "tlm-one-shot-evaluator/v1",
        "phase": "prepare",
        "research_state": {
            "path": _relative(context["root"], context["current_path"]),
            "sha256": file_sha256(context["current_path"]),
            "authorized_phase": context["status"]["authorized_phase"],
            "authorized_next_action": context["status"]["authorized_next_action"],
            "authorized_command": context["status"]["authorized_command"],
        },
        "evaluation_spec": {
            "path": _relative(context["root"], output / "evaluation_spec.json"),
            "sha256": file_sha256(output / "evaluation_spec.json"),
            "frozen": True,
        },
        "source_receipt": context["source_receipt"],
        "registered": registered,
        "prepare": {
            "receipt": {
                "path": _relative(context["root"], output / "prepare_receipt.json"),
                "sha256": file_sha256(output / "prepare_receipt.json"),
            },
            "artifacts": artifacts,
            "outcome_rows_read": 0,
            "outcome_artifacts_present": False,
            "outcome_blind_gates": behavior["gates"],
            "predictions_frozen": True,
            "positions_frozen": True,
            "all_checkpoints_used_without_selection": True,
            "authorizes_unseal": True,
        },
        "safety": {
            "target_assets_loaded": [],
            "retuning_performed": False,
            "thresholds_changed": False,
            "costs_or_accounting_changed": False,
            "second_unseal_attempted": False,
        },
        "authorization": {
            "explicit_user_authorization": False,
            "exact_registered_unseal": False,
        },
        "unseal": None,
        "completion": None,
        "replay": None,
    }
    write_json_atomic(output / "one_shot_packet.json", packet)
    validator = _validate_one_shot_packet(context, output / "one_shot_packet.json")

    result = {
        "schema_version": "v78-prepare-result/v1",
        "decision": WAIT_ACTION,
        "family_id": context["contract"]["family_id"],
        "evidence_tier": context["contract"]["evidence_tier"],
        "evaluation_spec_sha256": file_sha256(output / "evaluation_spec.json"),
        "prepare_receipt_sha256": file_sha256(output / "prepare_receipt.json"),
        "one_shot_packet_sha256": file_sha256(output / "one_shot_packet.json"),
        "registered_sha256": registered["sha256"],
        "audit": {"passed": True, "checks": behavior["gates"]},
        "summary": {
            "checkpoints_used": 9,
            "folds": 3,
            "triplets_per_fold": 120,
            "signal_dates": 357,
            "prediction_rows": int(len(predictions)),
            "candidate_position_rows": int(len(candidate)),
            "control_position_rows": int(len(controls)),
            "aggregate_candidate_turnover": position_checks["aggregate_turnover"],
            "outcome_blind_gates": len(behavior["gates"]),
            "outcome_rows_read": 0,
            "performance_metrics": 0,
            "pnl_evaluations": 0,
            "target_assets_loaded": 0,
        },
        "validator": validator,
        "target_contract": context["contract"]["target_contract"],
    }
    result["result_sha256"] = canonical_sha256(result)
    write_json_atomic(output / "result.json", result)
    report = "\n".join(
        [
            "# V78 outcome-blind persistent-duration prepare",
            "",
            "All nine V77 checkpoints were used without selection. Predictions,",
            "candidate positions, and registered controls are frozen for the exact",
            "2025 non-target adaptive-development window.",
            "",
            f"- Prediction rows: {len(predictions):,}",
            f"- Candidate position rows: {len(candidate):,}",
            f"- Control position rows: {len(controls):,}",
            f"- Aggregate candidate turnover: {position_checks['aggregate_turnover']:.6f}",
            "- Outcome rows read: 0",
            "- Financial metrics / PnL computed: 0",
            "- BTC/ETH/SOL loaded: none",
            "",
            "The one-shot outcome packet remains sealed. A new exact hash-bound",
            "user authorization is required before the single registered unseal.",
            "",
        ]
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    replay_receipt = replay_persistent_duration_evaluation_prepare(
        config, _prepared_context=context
    )
    artifact_files = [
        "evaluation_spec.json",
        "predictions_manifest.json",
        "candidate_positions_manifest.json",
        "control_positions_manifest.json",
        "outcome_request.json",
        "behavior_audit.json",
        "data_access.json",
        "input_hash_receipt.json",
        "source_receipt.json",
        "prepare_receipt.json",
        "one_shot_packet.json",
        "replay_receipt.json",
        "result.json",
        "report.md",
        "resolved_config.yaml",
    ]
    manifest = {
        "schema_version": "v78-artifact-manifest/v1",
        "files": {name: file_sha256(output / name) for name in artifact_files},
        "frozen_data": artifact_hashes,
        "outcome_rows_read": 0,
    }
    manifest["artifact_manifest_sha256"] = canonical_sha256(manifest)
    write_json_atomic(output / "artifact_manifest.json", manifest)
    result["replay_receipt_sha256"] = replay_receipt["replay_receipt_sha256"]
    return result


def replay_persistent_duration_evaluation_prepare(
    config: dict[str, Any], *, _prepared_context: dict[str, Any] | None = None
) -> dict[str, Any]:
    context = _prepared_context or _context(config)
    output = context["output"]
    packet = _load_json(output / "one_shot_packet.json", "V78 one-shot packet")
    prepare = packet.get("prepare", {})
    artifacts = prepare.get("artifacts", [])
    observed = {
        row["path"]: file_sha256(context["root"] / row["path"])
        for row in artifacts
    }
    expected = {row["path"]: row["sha256"] for row in artifacts}
    checks = {
        "artifact_hashes_match": observed == expected,
        "evaluation_spec_hash_matches": file_sha256(output / "evaluation_spec.json")
        == packet["evaluation_spec"]["sha256"],
        "prepare_receipt_hash_matches": file_sha256(output / "prepare_receipt.json")
        == packet["prepare"]["receipt"]["sha256"],
        "outcome_rows_read": packet["prepare"]["outcome_rows_read"] == 0,
        "outcomes_remain_sealed": packet.get("unseal") is None,
        "no_model_or_scientific_source_read": True,
    }
    if not all(checks.values()):
        raise V78EvaluationError(f"V78 hash-only replay failed: {checks}")
    receipt = {
        "schema_version": "v78-prepare-replay/v1",
        "passed": True,
        "checks": checks,
        "artifact_hashes": observed,
        "parquet_source_deserializations": 0,
        "checkpoint_container_deserializations": 0,
        "model_instantiations": 0,
        "inference_batches": 0,
        "outcome_rows_read": 0,
        "target_assets_loaded": [],
        "decision": WAIT_ACTION,
    }
    receipt["replay_receipt_sha256"] = canonical_sha256(receipt)
    write_json_atomic(output / "replay_receipt.json", receipt)
    return receipt


def _independent_turnover_audit(candidate: pd.DataFrame) -> dict[str, Any]:
    required = {
        "fold",
        "triplet_id",
        "signal_date",
        "symbol",
        "eligible",
        "weight",
        "action",
        "transition_turnover",
        "final_liquidation_turnover",
    }
    if set(candidate.columns) != required:
        raise V78EvaluationError("V78 frozen candidate-position projection drift")
    candidate = candidate.copy()
    candidate["signal_date"] = pd.to_datetime(candidate["signal_date"], utc=True)
    candidate = candidate.sort_values(
        ["fold", "triplet_id", "signal_date", "symbol"], kind="mergesort"
    )
    daily = candidate.drop_duplicates(["fold", "triplet_id", "signal_date"])
    rows: list[dict[str, Any]] = []
    for (fold, triplet_id), frame in candidate.groupby(
        ["fold", "triplet_id"], sort=True
    ):
        positions = (
            frame.pivot(index="signal_date", columns="symbol", values="weight")
            .sort_index()
            .to_numpy(dtype=np.float64)
        )
        transition, liquidation, total = _transition_ledger(positions)
        recorded = daily.loc[
            (daily["fold"] == fold) & (daily["triplet_id"] == triplet_id)
        ].sort_values("signal_date")
        recorded_total = float(
            recorded["transition_turnover"].sum()
            + recorded["final_liquidation_turnover"].sum()
        )
        rows.append(
            {
                "fold": int(fold),
                "triplet_id": str(triplet_id),
                "recomputed_total_turnover": total,
                "recorded_total_turnover": recorded_total,
                "maximum_daily_turnover_error": float(
                    np.max(
                        np.abs(
                            transition
                            - recorded["transition_turnover"].to_numpy(
                                dtype=np.float64
                            )
                        )
                    )
                ),
                "final_liquidation_error": abs(
                    liquidation
                    - float(recorded["final_liquidation_turnover"].sum())
                ),
                "eligible_days": int(recorded["eligible"].sum()),
                "exposure_fraction": float((positions.sum(axis=1) > 0.0).mean()),
            }
        )
    frame = pd.DataFrame(rows)
    aggregate = float(frame["recomputed_total_turnover"].mean())
    by_fold = {
        str(int(fold)): {
            "mean_turnover": float(group["recomputed_total_turnover"].mean()),
            "mean_eligible_days": float(group["eligible_days"].mean()),
            "mean_exposure_fraction": float(group["exposure_fraction"].mean()),
            "never_eligible_triplets": int((group["eligible_days"] == 0).sum()),
        }
        for fold, group in frame.groupby("fold", sort=True)
    }
    action_counts = {
        str(key): int(value) for key, value in daily["action"].value_counts().items()
    }
    turnover_by_action = {
        str(key): float(value)
        for key, value in daily.groupby("action")["transition_turnover"].sum().items()
    }
    unavailable_exit = float(
        daily.loc[
            (~daily["eligible"]) & (daily["action"] == "exit"),
            "transition_turnover",
        ].sum()
        / len(frame)
    )
    return {
        "triplet_subaccounts": int(len(frame)),
        "aggregate_candidate_turnover": aggregate,
        "registered_turnover_ceiling": 45.0,
        "excess_over_ceiling": aggregate - 45.0,
        "maximum_total_turnover_error": float(
            np.max(
                np.abs(
                    frame["recomputed_total_turnover"]
                    - frame["recorded_total_turnover"]
                )
            )
        ),
        "maximum_daily_turnover_error": float(
            frame["maximum_daily_turnover_error"].max()
        ),
        "maximum_final_liquidation_error": float(
            frame["final_liquidation_error"].max()
        ),
        "by_fold": by_fold,
        "action_counts": action_counts,
        "turnover_by_action": turnover_by_action,
        "mean_unavailable_exit_turnover": unavailable_exit,
        "position_implied_transaction_cost_debit": {
            str(cost): aggregate * cost / 10_000.0 for cost in (10, 20, 30)
        },
    }


def finalize_failed_persistent_duration_evaluation_prepare(
    config: dict[str, Any]
) -> dict[str, Any]:
    """Finalize a genuine V78 behavior-gate failure without model/source replay."""

    context = _context(config)
    output = context["output"]
    behavior = _load_json(output / "behavior_audit.json", "V78 behavior audit")
    failed = [name for name, passed in behavior.get("gates", {}).items() if not passed]
    if behavior.get("passed") is not False or failed != [
        "aggregate_turnover_within_registered_ceiling"
    ]:
        raise V78EvaluationError("V78 is not the exact turnover-only prepare failure")
    if (output / "one_shot_packet.json").exists():
        raise V78EvaluationError("V78 failure finalizer refuses an unseal-capable packet")
    manifests = {
        "predictions": _load_json(
            output / "predictions_manifest.json", "V78 prediction manifest"
        ),
        "candidate_positions": _load_json(
            output / "candidate_positions_manifest.json", "V78 candidate manifest"
        ),
        "control_positions": _load_json(
            output / "control_positions_manifest.json", "V78 control manifest"
        ),
    }
    for manifest in manifests.values():
        path = context["root"] / manifest["path"]
        if file_sha256(path) != manifest["sha256"]:
            raise V78EvaluationError(f"V78 frozen artifact hash drift: {path}")

    candidate_path = context["root"] / manifests["candidate_positions"]["path"]
    columns = [
        "fold",
        "triplet_id",
        "signal_date",
        "symbol",
        "eligible",
        "weight",
        "action",
        "transition_turnover",
        "final_liquidation_turnover",
    ]
    candidate = ds.dataset(candidate_path, format="parquet").to_table(
        columns=columns, use_threads=False
    ).to_pandas()
    independent = _independent_turnover_audit(candidate)
    if (
        not math.isclose(
            independent["aggregate_candidate_turnover"],
            float(behavior["diagnostics"]["aggregate_candidate_turnover"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or independent["maximum_total_turnover_error"] != 0.0
        or independent["maximum_daily_turnover_error"] != 0.0
        or independent["maximum_final_liquidation_error"] != 0.0
    ):
        raise V78EvaluationError("V78 independent turnover audit did not reconcile")

    audit = {
        "schema_version": "v78-failed-prepare-audit/v1",
        "passed": False,
        "checks": behavior["gates"],
        "failed_checks": failed,
        "independent_turnover_audit": independent,
        "frozen_artifacts": {
            name: {"path": value["path"], "sha256": value["sha256"]}
            for name, value in manifests.items()
        },
        "outcome_rows_read": 0,
        "performance_metrics_computed": 0,
        "pnl_evaluations": 0,
        "target_assets_loaded": [],
        "one_shot_unseal_authorized": False,
        "failure_is_accounting_bug": False,
        "decision": "pivot_away_from_current_family_without_target_evaluation_or_retuning",
        "finalization_source": context["source_receipt"],
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    write_json_atomic(output / "audit.json", audit)

    failure_receipt = {
        "schema_version": "v78-failed-prepare-receipt/v1",
        "evaluation_spec_sha256": file_sha256(output / "evaluation_spec.json"),
        "behavior_audit_sha256": file_sha256(output / "behavior_audit.json"),
        "audit_sha256": file_sha256(output / "audit.json"),
        "frozen_artifact_hashes": {
            value["path"]: value["sha256"] for value in manifests.values()
        },
        "outcome_rows_read": 0,
        "authorizes_unseal": False,
        "prediction_or_position_regeneration": False,
        "retuning_or_policy_change": False,
        "decision": audit["decision"],
    }
    failure_receipt["failure_receipt_sha256"] = canonical_sha256(failure_receipt)
    write_json_atomic(output / "prepare_failure_receipt.json", failure_receipt)
    replay = {
        "schema_version": "v78-failed-prepare-hash-replay/v1",
        "passed": True,
        "frozen_artifact_hashes_match": all(
            file_sha256(context["root"] / value["path"]) == value["sha256"]
            for value in manifests.values()
        ),
        "prediction_or_position_regeneration": False,
        "model_instantiations": 0,
        "checkpoint_container_deserializations": 0,
        "scientific_source_parquet_deserializations": 0,
        "outcome_rows_read": 0,
        "target_assets_loaded": [],
        "decision": audit["decision"],
    }
    replay["replay_receipt_sha256"] = canonical_sha256(replay)
    write_json_atomic(output / "replay_receipt.json", replay)
    result = {
        "schema_version": "v78-prepare-result/v2",
        "decision": audit["decision"],
        "family_id": context["contract"]["family_id"],
        "evidence_tier": context["contract"]["evidence_tier"],
        "audit": {"passed": False, "failed_checks": failed},
        "summary": {
            "checkpoints_used": 9,
            "prediction_rows": int(manifests["predictions"]["rows"]),
            "candidate_position_rows": int(manifests["candidate_positions"]["rows"]),
            "control_position_rows": int(manifests["control_positions"]["rows"]),
            "aggregate_candidate_turnover": independent[
                "aggregate_candidate_turnover"
            ],
            "registered_turnover_ceiling": 45.0,
            "position_implied_transaction_cost_debit": independent[
                "position_implied_transaction_cost_debit"
            ],
            "outcome_rows_read": 0,
            "performance_metrics": 0,
            "pnl_evaluations": 0,
            "target_assets_loaded": 0,
        },
        "behavior_audit_sha256": file_sha256(output / "behavior_audit.json"),
        "audit_sha256": file_sha256(output / "audit.json"),
        "prepare_failure_receipt_sha256": file_sha256(
            output / "prepare_failure_receipt.json"
        ),
        "replay_receipt_sha256": file_sha256(output / "replay_receipt.json"),
        "one_shot_packet_created": False,
        "one_shot_unseal_authorized": False,
        "target_contract": context["contract"]["target_contract"],
    }
    result["result_sha256"] = canonical_sha256(result)
    write_json_atomic(output / "result.json", result)
    report = "\n".join(
        [
            "# V78 outcome-blind prepare failure",
            "",
            "Eleven of twelve behavior gates passed. The preregistered aggregate",
            "turnover ceiling failed before any realized return was opened.",
            "",
            f"- Aggregate turnover: {independent['aggregate_candidate_turnover']:.2f}",
            "- Registered ceiling: 45.00",
            f"- Excess: {independent['excess_over_ceiling']:.2f}",
            "- Implied transaction-cost debit at 10/20/30 bps: "
            + ", ".join(
                f"{100.0 * independent['position_implied_transaction_cost_debit'][str(cost)]:.3f}%"
                for cost in (10, 20, 30)
            ),
            "- Outcome rows read: 0",
            "- Return, Sharpe, drawdown, PnL, and bootstrap computed: no",
            "- BTC/ETH/SOL loaded: none",
            "",
            "An independent position-only recomputation matched every daily and",
            "total turnover cell exactly. The failure is genuine, not an accounting",
            "or missingness bug. The one-shot outcome unseal is not authorized.",
            "",
        ]
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    packet_files = [
        "evaluation_spec.json",
        "predictions_manifest.json",
        "candidate_positions_manifest.json",
        "control_positions_manifest.json",
        "outcome_request.json",
        "behavior_audit.json",
        "audit.json",
        "data_access.json",
        "input_hash_receipt.json",
        "source_receipt.json",
        "prepare_failure_receipt.json",
        "replay_receipt.json",
        "result.json",
        "report.md",
        "resolved_config.yaml",
    ]
    manifest = {
        "schema_version": "v78-failed-artifact-manifest/v1",
        "files": {name: file_sha256(output / name) for name in packet_files},
        "frozen_data": failure_receipt["frozen_artifact_hashes"],
        "one_shot_packet_created": False,
        "outcome_rows_read": 0,
        "decision": audit["decision"],
    }
    manifest["artifact_manifest_sha256"] = canonical_sha256(manifest)
    write_json_atomic(output / "artifact_manifest.json", manifest)
    return result
