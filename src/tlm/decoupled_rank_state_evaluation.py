"""Outcome-blind V64 preparation for the frozen decoupled rank/state family.

This module deliberately stops before outcomes.  It verifies the V63 lineage,
uses every registered fold/seed checkpoint, freezes context and asset forecasts,
freezes the exact cost-specific policy positions, and emits the generic TLM
one-shot evaluator packet.  The labels Parquet is never opened here.
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from itertools import combinations
import gc
import json
import math
import os
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
import yaml

from .core.artifacts import canonical_sha256, file_sha256, write_json_atomic
from .decoupled_rank_state_harness import (
    derive_state_features,
    reconstruct_decoupled_returns,
)
from .decoupled_rank_state_training_data import BASE_FEATURES
from .decoupled_rank_state_training_engine import (
    FINAL_FORMAT,
    configure_v63_runtime,
    instantiate_models,
)
from .non_target_pretraining import TripletTensorStore
from .scientific_harness import FeatureScaler
from .state_conditioned_multi_horizon_training_engine import semantic_state_sha256


TARGET_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
PREPARE_ARTIFACTS = (
    ("predictions", "context_predictions.parquet"),
    ("predictions", "asset_predictions.parquet"),
    ("positions", "positions.parquet"),
    ("behavior_gates", "behavior_gates.json"),
    ("data_access", "data_access_receipt.json"),
)


class V64PrepareError(RuntimeError):
    """Fail-closed V64 preparation error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V64PrepareError(message)


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V64PrepareError(f"cannot load {name}: {path}") from exc
    _require(isinstance(value, dict), f"{name} must be a JSON object")
    return value


def _load_yaml(path: Path, name: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise V64PrepareError(f"cannot load {name}: {path}") from exc
    _require(isinstance(value, dict), f"{name} must be a YAML object")
    return value


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, text=True, capture_output=True, check=False
    )
    _require(result.returncode == 0, f"git {' '.join(arguments)} failed")
    return result.stdout.strip()


def _source_receipt(root: Path, files: list[str], require_clean: bool) -> dict[str, Any]:
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    if require_clean:
        _require(status == "", "V64 source tree must be clean before prepare")
    hashes: dict[str, str] = {}
    for relative in files:
        path = (root / relative).resolve()
        _require(root == path or root in path.parents, f"source path escapes root: {relative}")
        _require(path.is_file(), f"missing source receipt file: {relative}")
        hashes[relative] = file_sha256(path)
    return {
        "git_clean": status == "",
        "git_head": _git(root, "rev-parse", "HEAD"),
        "files": hashes,
        "bundle_sha256": canonical_sha256(hashes),
    }


def _utc_day(value: object) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        return result.tz_localize("UTC")
    return result.tz_convert("UTC")


def _fold_rows(value: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    rows = value.get("folds")
    _require(isinstance(rows, list), "fold artifact lacks rows")
    result = {int(row["fold"]): dict(row) for row in rows}
    _require(set(result) == {1, 2, 3}, "fold artifact must contain folds 1, 2, 3")
    return result


def _scaler(record: Mapping[str, Any]) -> FeatureScaler:
    value = record["feature_scaler"]
    scaler = FeatureScaler(
        feature_names=tuple(str(item) for item in value["feature_names"]),
        mean=tuple(float(item) for item in value["mean"]),
        scale=tuple(float(item) for item in value["scale"]),
        source_relative_feature_index=int(value["source_relative_feature_index"]),
        fit_scope=str(value["fit_scope"]),
        fit_start=str(value["fit_start"]),
        fit_end=str(value["fit_end"]),
        fit_rows=int(value["fit_rows"]),
    )
    _require(
        scaler.state_sha256() == record["feature_scaler_state_sha256"],
        "V63 feature scaler semantic hash drift",
    )
    return scaler


def _registered_contract(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    bound = {
        "cost_bps": [int(value) for value in evaluation["policy"]["reporting_cost_bps"]],
        "accounting": dict(evaluation["accounting"]),
        "controls": dict(evaluation["policy"]["controls"]),
        "gates": dict(evaluation["outcome_dependent_gates"]),
        "outcome_blind_gate_names": list(evaluation["outcome_blind_gate_names"]),
    }
    return {**bound, "sha256": canonical_sha256(bound)}


def _fold_policy_positions(
    predicted_raw_excess: np.ndarray,
    predicted_market_component: np.ndarray,
    momentum_30: np.ndarray,
    eligible: np.ndarray,
    *,
    base_cost: float,
    switch_hurdle: float,
    risky_weight: float,
) -> dict[str, Any]:
    """Apply the exact V60 policy to an arbitrary heldout fold width."""
    excess = np.asarray(predicted_raw_excess, dtype=np.float64)
    market = np.asarray(predicted_market_component, dtype=np.float64)
    momentum = np.asarray(momentum_30, dtype=np.float64)
    eligibility = np.asarray(eligible, dtype=bool)
    _require(excess.ndim == 2 and excess.shape[1] >= 1, "invalid fold excess shape")
    _require(momentum.shape == excess.shape and eligibility.shape == excess.shape, "fold policy shape drift")
    _require(market.shape == excess.shape[:1], "fold market-component shape drift")
    _require(math.isfinite(base_cost) and base_cost >= 0, "invalid fold policy cost")
    _require(math.isfinite(switch_hurdle) and switch_hurdle >= 0, "invalid switch hurdle")
    _require(0 < risky_weight <= 1, "invalid risky weight")

    positions = np.zeros_like(excess)
    actions: list[str] = []
    selected_assets: list[int | None] = []
    incumbent: int | None = None
    entry_cost = base_cost * risky_weight
    switch_cost = base_cost * 2.0 * risky_weight
    for day in range(len(excess)):
        valid = eligibility[day] & np.isfinite(excess[day]) & np.isfinite(momentum[day])
        valid_indexes = np.flatnonzero(valid)
        if incumbent is not None and not valid[incumbent]:
            incumbent = None
            actions.append("forced_exit")
            selected_assets.append(None)
            continue
        if (
            not np.isfinite(market[day])
            or len(valid_indexes) == 0
            or np.all(momentum[day, valid_indexes] <= 0)
        ):
            actions.append("momentum_exit" if incumbent is not None else "cash")
            incumbent = None
            selected_assets.append(None)
            continue
        best_value = np.max(excess[day, valid_indexes])
        challenger = int(
            valid_indexes[np.flatnonzero(excess[day, valid_indexes] == best_value)[0]]
        )
        challenger_edge = float(market[day] + excess[day, challenger])
        if incumbent is None:
            if challenger_edge > entry_cost:
                incumbent = challenger
                actions.append("entry")
            else:
                actions.append("cash")
        else:
            incumbent_edge = float(market[day] + excess[day, incumbent])
            if incumbent_edge <= 0.0:
                incumbent = None
                actions.append("edge_exit")
            elif challenger != incumbent and (
                excess[day, challenger] - excess[day, incumbent] > switch_hurdle
                and challenger_edge > switch_cost
            ):
                incumbent = challenger
                actions.append("switch")
            else:
                actions.append("hold")
        if incumbent is not None:
            positions[day, incumbent] = risky_weight
        selected_assets.append(incumbent)
    return {
        "positions": positions,
        "actions": actions,
        "selected_assets": selected_assets,
    }


def _metadata_context(config: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = dict(config["decoupled_rank_state_evaluation"])
    root = Path(evaluation["project_root"]).resolve()
    current_path = root / evaluation["research_state"]
    phase_path = root / evaluation["phase_contract"]
    current = _load_yaml(current_path, "research state")
    phase = _load_yaml(phase_path, "V64 phase contract")
    command = (
        "PYTHONPATH=src python3 -m tlm decoupled-rank-state-evaluation-prepare "
        "--config configs/v64_decoupled_rank_state_evaluation.yaml"
    )
    _require(current.get("authorized_phase") == "v64", "live phase is not V64")
    _require(
        current.get("authorized_next_action")
        == "authorize_v64_frozen_adaptive_development_evaluation_only",
        "live V64 action drift",
    )
    _require(current.get("authorized_command") == command, "live V64 command drift")
    _require(current.get("target_assets", {}).get("status") == "sealed", "targets unsealed")
    _require(
        current.get("phase_contract", {}).get("path") == evaluation["phase_contract"]
        and current.get("phase_contract", {}).get("file_sha256") == file_sha256(phase_path),
        "live V64 phase-contract binding drift",
    )
    _require(phase.get("phase") == "v64", "wrong phase contract")
    _require(phase.get("family_id") == evaluation["family_id"], "V64 family drift")

    expected_hashes = phase["input_contract"]["expected_file_sha256_by_path"]
    input_receipt = {}
    for relative, expected in expected_hashes.items():
        path = root / relative
        _require(path.is_file(), f"missing registered V64 input: {relative}")
        observed = file_sha256(path)
        _require(observed == expected, f"registered V64 input hash drift: {relative}")
        input_receipt[relative] = observed

    paths = {name: root / relative for name, relative in evaluation["inputs"].items()}
    metadata = {
        name: _load_json(paths[name], name)
        for name in (
            "blueprint",
            "dataset_spec",
            "dataset_manifest",
            "asset_folds",
            "triplet_catalog",
            "training_result",
            "training_spec",
            "checkpoint_manifest",
            "scaler_manifest",
        )
    }
    _require(
        metadata["training_result"].get("decision")
        == "authorize_v64_frozen_adaptive_development_evaluation_only",
        "V63 receipt does not authorize V64",
    )
    _require(
        metadata["blueprint"].get("candidate_family_id") == evaluation["family_id"],
        "blueprint family drift",
    )
    _require(
        set(metadata["dataset_manifest"].get("symbols", [])).isdisjoint(TARGET_SYMBOLS),
        "target symbols appear in V62 manifest",
    )
    source = _source_receipt(
        root, list(evaluation["source_receipt_files"]), bool(evaluation["require_clean_git"])
    )
    registered = _registered_contract(evaluation)
    spec_body = {
        "schema_version": "v64-decoupled-rank-state-evaluation-spec/v1",
        "frozen": True,
        "family_id": evaluation["family_id"],
        "phase_contract_path": evaluation["phase_contract"],
        "phase_contract_sha256": file_sha256(phase_path),
        "authorization_receipt": phase["authorization_receipt"],
        "evidence_tier": evaluation["lifecycle"]["evidence_tier"],
        "data_access": evaluation["data_access"],
        "inference": evaluation["inference"],
        "policy": evaluation["policy"],
        "accounting": evaluation["accounting"],
        "predictive_metrics": evaluation["predictive_metrics"],
        "bootstrap": evaluation["bootstrap"],
        "outcome_dependent_gates": evaluation["outcome_dependent_gates"],
        "outcome_blind_gate_names": evaluation["outcome_blind_gate_names"],
        "lifecycle": evaluation["lifecycle"],
        "target_contract": evaluation["target_contract"],
        "registered_sha256": registered["sha256"],
    }
    spec = {**spec_body, "evaluation_spec_semantic_sha256": canonical_sha256(spec_body)}
    return {
        "root": root,
        "evaluation": evaluation,
        "current": current,
        "current_path": current_path,
        "phase": phase,
        "phase_path": phase_path,
        "paths": paths,
        "metadata": metadata,
        "input_receipt": input_receipt,
        "source": source,
        "registered": registered,
        "spec": spec,
    }


def _read_projection(
    path: Path,
    columns: list[str],
    symbols: list[str],
    minimum: pd.Timestamp,
    maximum: pd.Timestamp,
    *,
    require_exact_symbols: bool = True,
) -> pd.DataFrame:
    predicate = (
        ds.field("symbol").isin(symbols)
        & (ds.field("date") >= minimum.to_pydatetime())
        & (ds.field("date") <= maximum.to_pydatetime())
    )
    table = ds.dataset(path, format="parquet").to_table(
        columns=columns, filter=predicate, use_threads=False
    )
    frame = table.to_pandas()
    _require(list(frame.columns) == columns, f"projection drift: {path}")
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame["symbol"] = frame["symbol"].astype(str)
    _require(not frame.duplicated(["date", "symbol"]).any(), f"duplicate keys: {path}")
    loaded_symbols = set(frame["symbol"].unique())
    if require_exact_symbols:
        _require(loaded_symbols == set(symbols), f"symbol filter drift: {path}")
    else:
        _require(
            bool(loaded_symbols) and loaded_symbols.issubset(symbols),
            f"readiness symbol filter drift: {path}",
        )
    _require(set(frame["symbol"]).isdisjoint(TARGET_SYMBOLS), "target symbol loaded")
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def _fold_scope(context: Mapping[str, Any], fold: int) -> tuple[list[str], list[tuple[str, str, str]]]:
    folds = _fold_rows(context["metadata"]["asset_folds"])
    catalogs = _fold_rows(context["metadata"]["triplet_catalog"])
    row = folds[fold]
    symbols = sorted(str(value) for value in row["test_symbols"])
    train = set(str(value) for value in row["train_symbols"])
    _require(len(symbols) == 10, "V64 requires ten heldout symbols per fold")
    _require(train.isdisjoint(symbols) and set(symbols).isdisjoint(TARGET_SYMBOLS), "fold isolation drift")
    triplets = [tuple(str(value) for value in item) for item in catalogs[fold]["test_triplets"]]
    _require(triplets == list(combinations(symbols, 3)), "test triplet catalog is not exact lexical set")
    return symbols, triplets


def _fold_data(context: Mapping[str, Any], fold: int) -> dict[str, Any]:
    evaluation = context["evaluation"]
    access = evaluation["data_access"]
    symbols, triplets = _fold_scope(context, fold)
    minimum = _utc_day(access["feature_start"])
    maximum = _utc_day(access["signal_end"])
    panel = _read_projection(
        context["paths"]["panel"], list(access["feature_columns"]), symbols, minimum, maximum
    )
    roles = _read_projection(
        context["paths"]["sequence_roles"], list(access["readiness_columns"]), symbols,
        _utc_day(access["signal_start"]), maximum,
        require_exact_symbols=False,
    )
    roles["sequence_start_date"] = pd.to_datetime(roles["sequence_start_date"], utc=True)
    ready = roles.loc[
        roles[access["readiness_flag"]].astype(bool)
        & roles["h1_label_complete"].astype(bool),
        ["date", "symbol"],
    ]
    availability = {
        pd.Timestamp(date): tuple(sorted(frame["symbol"].unique()))
        for date, frame in ready.groupby("date", sort=True)
    }
    dates = pd.date_range(access["signal_start"], access["signal_end"], freq="D", tz="UTC")
    _require(set(availability) == set(dates), f"fold {fold} readiness date drift")
    samples: list[dict[str, Any]] = []
    allowed = set(triplets)
    for date in dates:
        current = availability[pd.Timestamp(date)]
        for triplet in combinations(current, 3):
            if triplet in allowed:
                samples.append({"date": pd.Timestamp(date), "triplet": triplet})
    _require(samples, f"fold {fold} has no eligible triplet contexts")
    return {
        "symbols": symbols,
        "triplets": triplets,
        "dates": dates,
        "panel": panel,
        "roles": roles,
        "availability": availability,
        "samples": samples,
    }


def _load_models(
    context: Mapping[str, Any], fold: int, device: torch.device
) -> tuple[list[tuple[int, torch.nn.Module, torch.nn.Module]], list[dict[str, Any]]]:
    manifest_rows = {
        (int(row["fold"]), int(row["seed"])): row
        for row in context["metadata"]["checkpoint_manifest"]["jobs"]
    }
    models = []
    receipts = []
    for seed in context["evaluation"]["inference"]["seeds"]:
        key = (fold, int(seed))
        _require(key in manifest_rows, f"missing V63 checkpoint cell {key}")
        row = manifest_rows[key]
        path = context["root"] / row["path"]
        _require(file_sha256(path) == row["file_sha256"], f"checkpoint file hash drift: {key}")
        ranker, gate = instantiate_models(context["metadata"]["blueprint"], device)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _require(
            payload.get("format_version") == FINAL_FORMAT
            and payload.get("kind") == "final"
            and payload.get("stage") == "complete",
            f"checkpoint final-state drift: {key}",
        )
        registered_semantic = payload.get("semantic_checkpoint_sha256")
        body = {k: v for k, v in payload.items() if k != "semantic_checkpoint_sha256"}
        _require(semantic_state_sha256(body) == registered_semantic, f"checkpoint semantic drift: {key}")
        _require(registered_semantic == row["semantic_checkpoint_sha256"], f"manifest semantic drift: {key}")
        _require(payload.get("context") == row["context"], f"checkpoint context drift: {key}")
        ranker.load_state_dict(payload["ranker_current_state"], strict=True)
        gate.load_state_dict(payload["gate_current_state"], strict=True)
        _require(
            semantic_state_sha256(payload["ranker_current_state"])
            == row["ranker_state_sha256"]
            and semantic_state_sha256(payload["gate_current_state"])
            == row["gate_state_sha256"],
            f"checkpoint component-state hash drift: {key}",
        )
        ranker.eval().requires_grad_(False)
        gate.eval().requires_grad_(False)
        models.append((int(seed), ranker, gate))
        receipts.append({
            "fold": fold,
            "seed": int(seed),
            "path": row["path"],
            "file_sha256": row["file_sha256"],
            "semantic_checkpoint_sha256": registered_semantic,
            "ranker_state_sha256": semantic_state_sha256(payload["ranker_current_state"]),
            "gate_state_sha256": semantic_state_sha256(payload["gate_current_state"]),
            "used_for_inference": True,
            "selected_or_discarded": False,
        })
        del payload
    return models, receipts


def _infer_fold(
    context: Mapping[str, Any], fold: int, device: torch.device
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    data = _fold_data(context, fold)
    evaluation = context["evaluation"]
    feature_names = list(evaluation["inference"]["feature_order"])
    scale_rows = _fold_rows(context["metadata"]["scaler_manifest"])
    scale_row = scale_rows[fold]
    scaler = _scaler(scale_row)
    _require(tuple(feature_names) == scaler.feature_names == tuple(BASE_FEATURES), "feature order drift")
    store = TripletTensorStore(
        data["panel"], feature_names, int(evaluation["inference"]["lookback_days"]),
        evaluation["inference"]["relative_source_feature"],
    )
    models, checkpoint_receipts = _load_models(context, fold, device)
    symbols = data["symbols"]
    dates = data["dates"]
    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
    date_index = {pd.Timestamp(date): index for index, date in enumerate(dates)}
    shape = (len(dates), len(symbols))
    excess_sum = np.zeros(shape, dtype=np.float64)
    excess_count = np.zeros(shape, dtype=np.int64)
    market_sum = np.zeros(len(dates), dtype=np.float64)
    market_count = np.zeros(len(dates), dtype=np.int64)
    disagreement_sum = np.zeros(shape, dtype=np.float64)
    context_rows: list[dict[str, Any]] = []
    batch_size = int(evaluation["runtime"]["inference_batch_size"])
    excess_scale = float(scale_row["ranker_excess_rms"])
    market_scale = float(scale_row["state_market_rms"])
    for start in range(0, len(data["samples"]), batch_size):
        batch = data["samples"][start : start + batch_size]
        x_np = store.materialize_batch(batch, scaler)
        x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
        seed_excess = []
        seed_market = []
        seed_volatility = []
        with torch.inference_mode():
            state_features = derive_state_features(x)
            for seed, ranker, gate in models:
                output = ranker(x)
                _require(set(output) == {"excess_return_z", "log_volatility_7d"}, "ranker head drift")
                reconstructed = reconstruct_decoupled_returns(
                    output["excess_return_z"], gate(state_features),
                    excess_scale=excess_scale, market_scale=market_scale,
                )
                seed_excess.append(reconstructed["raw_excess"].cpu().numpy())
                seed_market.append(reconstructed["market"].cpu().numpy())
                seed_volatility.append(output["log_volatility_7d"].cpu().numpy())
        excess_values = np.stack(seed_excess, axis=0)
        market_values = np.stack(seed_market, axis=0)
        volatility_values = np.stack(seed_volatility, axis=0)
        _require(
            np.isfinite(excess_values).all()
            and np.isfinite(market_values).all()
            and np.isfinite(volatility_values).all(),
            "non-finite V64 prediction",
        )
        context_excess = excess_values.mean(axis=0)
        context_market = market_values.mean(axis=0)
        context_disagreement = excess_values.std(axis=0, ddof=0)
        for offset, sample in enumerate(batch):
            date = pd.Timestamp(sample["date"])
            day = date_index[date]
            triplet = tuple(str(value) for value in sample["triplet"])
            market_sum[day] += float(context_market[offset])
            market_count[day] += 1
            for slot, symbol in enumerate(triplet):
                asset = symbol_index[symbol]
                excess_sum[day, asset] += float(context_excess[offset, slot])
                excess_count[day, asset] += 1
                disagreement_sum[day, asset] += float(context_disagreement[offset, slot])
                for seed_offset, seed in enumerate(evaluation["inference"]["seeds"]):
                    raw_excess = float(excess_values[seed_offset, offset, slot])
                    market = float(market_values[seed_offset, offset])
                    context_rows.append({
                        "date": date,
                        "fold": fold,
                        "seed": int(seed),
                        "triplet_key": "|".join(triplet),
                        "slot": slot,
                        "symbol": symbol,
                        "raw_excess": raw_excess,
                        "market_component": market,
                        "absolute_edge": market + raw_excess,
                        "log_volatility_z": float(volatility_values[seed_offset, offset, slot]),
                    })
        del x, x_np, state_features, excess_values, market_values, volatility_values

    eligibility = np.zeros(shape, dtype=bool)
    expected_counts = np.zeros(shape, dtype=np.int64)
    for day, date in enumerate(dates):
        ready = data["availability"][pd.Timestamp(date)]
        for symbol in ready:
            asset = symbol_index[symbol]
            eligibility[day, asset] = True
            expected_counts[day, asset] = math.comb(len(ready) - 1, 2)
    _require(np.array_equal(excess_count, expected_counts), f"fold {fold} context count drift")
    raw_excess = np.divide(
        excess_sum, excess_count, out=np.full(shape, np.nan), where=excess_count > 0
    )
    disagreement = np.divide(
        disagreement_sum, excess_count, out=np.full(shape, np.nan), where=excess_count > 0
    )
    market = np.divide(
        market_sum, market_count, out=np.full(len(dates), np.nan), where=market_count > 0
    )
    _require(
        np.isfinite(raw_excess[eligibility]).all()
        and np.isfinite(disagreement[eligibility]).all()
        and np.isfinite(market).all(),
        f"fold {fold} aggregate prediction drift",
    )

    momentum_lookup: dict[tuple[pd.Timestamp, str], float] = {}
    for symbol, frame in data["panel"].groupby("symbol", sort=True):
        current = frame.sort_values("date").copy()
        current["momentum_30"] = current["log_close_to_close_return"].rolling(30, min_periods=30).sum()
        momentum_lookup.update({
            (pd.Timestamp(row.date), str(symbol)): float(row.momentum_30)
            for row in current[["date", "momentum_30"]].itertuples(index=False)
        })
    momentum = np.full(shape, np.nan, dtype=np.float64)
    for day, date in enumerate(dates):
        for asset, symbol in enumerate(symbols):
            if eligibility[day, asset]:
                momentum[day, asset] = momentum_lookup[(pd.Timestamp(date), symbol)]
    _require(np.isfinite(momentum[eligibility]).all(), f"fold {fold} momentum drift")

    asset_rows = []
    for day, date in enumerate(dates):
        for asset, symbol in enumerate(symbols):
            if eligibility[day, asset]:
                asset_rows.append({
                    "date": pd.Timestamp(date),
                    "fold": fold,
                    "symbol": symbol,
                    "eligible": True,
                    "context_count": int(excess_count[day, asset]),
                    "raw_excess": float(raw_excess[day, asset]),
                    "market_component": float(market[day]),
                    "absolute_edge": float(market[day] + raw_excess[day, asset]),
                    "momentum_30": float(momentum[day, asset]),
                    "excess_seed_disagreement": float(disagreement[day, asset]),
                })

    position_rows = []
    policy_summaries = []
    for cost_bps in evaluation["policy"]["reporting_cost_bps"]:
        policy = _fold_policy_positions(
            raw_excess, market, momentum, eligibility,
            base_cost=float(cost_bps) / 10000.0,
            switch_hurdle=float(evaluation["policy"]["switch_hurdle"]),
            risky_weight=float(evaluation["policy"]["risky_weight"]),
        )
        weights = np.asarray(policy["positions"], dtype=np.float64)
        previous = np.vstack([np.zeros((1, len(symbols))), weights[:-1]])
        transition_turnover = np.abs(weights - previous).sum(axis=1)
        liquidation = np.zeros(len(dates), dtype=np.float64)
        if bool(evaluation["policy"]["final_liquidation"]):
            liquidation[-1] = float(np.abs(weights[-1]).sum())
        total_turnover = transition_turnover + liquidation
        for day, date in enumerate(dates):
            selected = policy["selected_assets"][day]
            selected_symbol = symbols[selected] if selected is not None else None
            for asset, symbol in enumerate(symbols):
                position_rows.append({
                    "date": pd.Timestamp(date),
                    "fold": fold,
                    "cost_bps": int(cost_bps),
                    "symbol": symbol,
                    "eligible": bool(eligibility[day, asset]),
                    "candidate_weight": float(weights[day, asset]),
                    "selected_symbol": selected_symbol,
                    "action": str(policy["actions"][day]),
                    "transition_turnover": float(transition_turnover[day]),
                    "final_liquidation_turnover": float(liquidation[day]),
                    "total_turnover": float(total_turnover[day]),
                    "gross_exposure": float(weights[day].sum()),
                })
        policy_summaries.append({
            "cost_bps": int(cost_bps),
            "risky_days": int((weights.sum(axis=1) > 0).sum()),
            "cash_days": int((weights.sum(axis=1) == 0).sum()),
            "transition_count": int((transition_turnover > 0).sum()),
            "turnover": float(total_turnover.sum()),
            "actions": {name: policy["actions"].count(name) for name in sorted(set(policy["actions"]))},
        })

    for _, ranker, gate in models:
        ranker.to("cpu")
        gate.to("cpu")
    if device.type == "mps":
        torch.mps.empty_cache()
    gc.collect()
    diagnostics = {
        "fold": fold,
        "test_symbols": symbols,
        "signal_dates": len(dates),
        "eligible_asset_dates": int(eligibility.sum()),
        "triplet_contexts": len(data["samples"]),
        "seed_context_asset_predictions": len(context_rows),
        "minimum_ready_assets": int(eligibility.sum(axis=1).min()),
        "maximum_ready_assets": int(eligibility.sum(axis=1).max()),
        "minimum_context_count": int(excess_count[eligibility].min()),
        "maximum_context_count": int(excess_count[eligibility].max()),
        "prediction_raw_excess_std": float(np.std(raw_excess[eligibility], ddof=0)),
        "prediction_market_std": float(np.std(market, ddof=0)),
        "mean_seed_disagreement": float(np.mean(disagreement[eligibility])),
        "policy": policy_summaries,
        "outcome_rows_read": 0,
        "target_assets_loaded": [],
    }
    return (
        pd.DataFrame(context_rows),
        pd.DataFrame(asset_rows),
        pd.DataFrame(position_rows),
        diagnostics,
        checkpoint_receipts,
    )


def _behavior_gates(
    context: Mapping[str, Any],
    contexts: pd.DataFrame,
    assets: pd.DataFrame,
    positions: pd.DataFrame,
    diagnostics: list[dict[str, Any]],
    checkpoints: list[dict[str, Any]],
) -> dict[str, Any]:
    names = list(context["evaluation"]["outcome_blind_gate_names"])
    expected_dates = int(context["evaluation"]["inference"]["expected_signal_dates"])
    cost_count = len(context["evaluation"]["policy"]["reporting_cost_bps"])
    action_rows = positions.drop_duplicates(["date", "fold", "cost_bps"])
    selected_consistent = []
    for _, frame in positions.groupby(["date", "fold", "cost_bps"], sort=False):
        selected = frame["selected_symbol"].iloc[0]
        selected_consistent.append(
            int((frame["candidate_weight"] > 0).sum()) <= 1
            and (
                selected is None
                or (
                    bool((frame["symbol"] == selected).any())
                    and bool(
                        (
                            frame.loc[
                                frame["symbol"] == selected, "candidate_weight"
                            ]
                            > 0
                        ).all()
                    )
                )
            )
        )
    values = {
        "all_registered_checkpoints_used_without_selection": len(checkpoints) == 9
        and {(row["fold"], row["seed"]) for row in checkpoints}
        == {(fold, seed) for fold in (1, 2, 3) for seed in (42, 7, 123)}
        and all(row["used_for_inference"] and not row["selected_or_discarded"] for row in checkpoints),
        "exact_fold_asset_and_triplet_scope": all(
            row["signal_dates"] == expected_dates and len(row["test_symbols"]) == 10
            for row in diagnostics
        ),
        "missingness_matches_registered_readiness": len(assets)
        == sum(row["eligible_asset_dates"] for row in diagnostics)
        and bool(assets["eligible"].all()),
        "prediction_distribution_finite_and_nonconstant": bool(
            np.isfinite(contexts[["raw_excess", "market_component", "absolute_edge", "log_volatility_z"]].to_numpy()).all()
        ) and all(row["prediction_raw_excess_std"] > 0 and row["prediction_market_std"] > 0 for row in diagnostics),
        "ensemble_disagreement_finite": bool(np.isfinite(assets["excess_seed_disagreement"]).all())
        and bool((assets["excess_seed_disagreement"] >= 0).all()),
        "action_coverage_complete": len(action_rows) == 3 * expected_dates * cost_count
        and set(action_rows["action"]).issubset({"entry", "cash", "hold", "switch", "edge_exit", "momentum_exit", "forced_exit"}),
        "turnover_accounting_complete": bool(
            np.isfinite(action_rows[["transition_turnover", "final_liquidation_turnover", "total_turnover"]].to_numpy()).all()
        ) and bool((action_rows["total_turnover"] >= 0).all()),
        "concentration_at_most_one_asset": bool(
            (positions.groupby(["date", "fold", "cost_bps"])["candidate_weight"].apply(lambda x: int((x > 0).sum())) <= 1).all()
        ),
        "exposure_bounded_zero_to_one": bool((positions["candidate_weight"] >= 0).all())
        and bool((action_rows["gross_exposure"] <= 1.0 + 1.0e-12).all()),
        "episode_and_churn_structure_complete": all(selected_consistent),
        "quantile_crossing_not_applicable": True,
        "zero_outcome_and_target_access": all(row["outcome_rows_read"] == 0 and not row["target_assets_loaded"] for row in diagnostics)
        and set(assets["symbol"]).isdisjoint(TARGET_SYMBOLS),
    }
    _require(set(values) == set(names), "outcome-blind gate registry drift")
    return {
        "schema_version": "v64-outcome-blind-behavior-gates/v1",
        "gates": values,
        "passed": all(values.values()),
        "diagnostics": diagnostics,
    }


def _cached_prepare(context: Mapping[str, Any], output: Path) -> dict[str, Any] | None:
    packet_path = output / "one_shot_packet.json"
    result_path = output / "result.json"
    if not packet_path.is_file() and not result_path.is_file():
        return None
    _require(packet_path.is_file() and result_path.is_file(), "partial V64 prepare cache")
    packet = _load_json(packet_path, "V64 one-shot packet")
    for artifact in packet["prepare"]["artifacts"]:
        path = context["root"] / artifact["path"]
        _require(path.is_file() and file_sha256(path) == artifact["sha256"], "V64 cached artifact drift")
    receipt_ref = packet["prepare"]["receipt"]
    receipt_path = context["root"] / receipt_ref["path"]
    _require(receipt_path.is_file() and file_sha256(receipt_path) == receipt_ref["sha256"], "V64 cached receipt drift")
    _require(packet["source_receipt"] == context["source"], "V64 cached source drift")
    result = _load_json(result_path, "V64 prepare result")
    result["replay"] = {"reused_frozen_prepare": True, "new_predictions": 0, "new_positions": 0}
    return result


def prepare_decoupled_rank_state_evaluation(config: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze the V64 prepare packet without opening labels or outcomes."""
    context = _metadata_context(config)
    output = context["root"] / context["evaluation"]["output_dir"]
    cached = _cached_prepare(context, output)
    if cached is not None:
        return cached
    output.mkdir(parents=True, exist_ok=True)
    for forbidden in (
        "unseal_authorization_receipt.json",
        "outcome_packet.parquet",
        "outcome_receipt.json",
        "metrics.json",
        "completion_receipt.json",
    ):
        _require(
            not (output / forbidden).exists(),
            f"post-prepare V64 artifact exists before prepare: {forbidden}",
        )
    device_name = str(context["evaluation"]["runtime"]["device"])
    _require(os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0" if device_name == "mps" else True, "V64 MPS fallback must be disabled")
    device = configure_v63_runtime(device_name, seed=20260715)

    context_frames = []
    asset_frames = []
    position_frames = []
    diagnostics = []
    checkpoints = []
    for fold in context["evaluation"]["inference"]["folds"]:
        current_contexts, current_assets, current_positions, current_diagnostics, current_checkpoints = _infer_fold(
            context, int(fold), device
        )
        context_frames.append(current_contexts)
        asset_frames.append(current_assets)
        position_frames.append(current_positions)
        diagnostics.append(current_diagnostics)
        checkpoints.extend(current_checkpoints)

    contexts = pd.concat(context_frames, ignore_index=True).sort_values(
        ["date", "fold", "triplet_key", "seed", "slot"]
    ).reset_index(drop=True)
    assets = pd.concat(asset_frames, ignore_index=True).sort_values(
        ["date", "fold", "symbol"]
    ).reset_index(drop=True)
    positions = pd.concat(position_frames, ignore_index=True).sort_values(
        ["date", "fold", "cost_bps", "symbol"]
    ).reset_index(drop=True)
    behavior = _behavior_gates(context, contexts, assets, positions, diagnostics, checkpoints)
    _require(behavior["passed"], "V64 outcome-blind behavior gate failed")

    write_json_atomic(output / "evaluation_spec.json", context["spec"])
    write_json_atomic(output / "input_hash_receipt.json", {
        "schema_version": "v64-input-hash-receipt/v1",
        "files": context["input_receipt"],
        "bundle_sha256": canonical_sha256(context["input_receipt"]),
    })
    write_json_atomic(output / "source_receipt.json", context["source"])
    write_json_atomic(output / "checkpoint_receipt.json", {
        "schema_version": "v64-checkpoint-receipt/v1",
        "checkpoints": checkpoints,
        "checkpoint_count": len(checkpoints),
        "all_used_without_selection": True,
    })
    data_access = {
        "schema_version": "v64-prepare-data-access/v1",
        "panel_reads": 3,
        "readiness_reads": 3,
        "outcome_source_reads": 0,
        "outcome_rows_read": 0,
        "outcome_columns_requested": [],
        "labels_path_opened": False,
        "target_assets_loaded": [],
        "feature_columns": context["evaluation"]["data_access"]["feature_columns"],
        "readiness_columns": context["evaluation"]["data_access"]["readiness_columns"],
        "folds": diagnostics,
    }
    write_json_atomic(output / "data_access_receipt.json", data_access)
    _atomic_parquet(contexts, output / "context_predictions.parquet")
    _atomic_parquet(assets, output / "asset_predictions.parquet")
    _atomic_parquet(positions, output / "positions.parquet")
    write_json_atomic(output / "behavior_gates.json", behavior)

    artifact_hashes = {
        (output / name).relative_to(context["root"]).as_posix(): file_sha256(output / name)
        for _, name in PREPARE_ARTIFACTS
    }
    spec_file_hash = file_sha256(output / "evaluation_spec.json")
    prepare_receipt = {
        "schema_version": "tlm-one-shot-prepare/v1",
        "evaluation_spec_sha256": spec_file_hash,
        "registered_sha256": context["registered"]["sha256"],
        "artifact_hashes": artifact_hashes,
        "outcome_rows_read": 0,
        "outcome_blind_gates_passed": True,
        "authorizes_unseal": True,
    }
    write_json_atomic(output / "prepare_receipt.json", prepare_receipt)
    packet = {
        "schema_version": "tlm-one-shot-evaluator/v1",
        "phase": "prepare",
        "research_state": {
            "path": context["evaluation"]["research_state"],
            "sha256": file_sha256(context["current_path"]),
            "authorized_phase": context["current"]["authorized_phase"],
            "authorized_next_action": context["current"]["authorized_next_action"],
            "authorized_command": context["current"]["authorized_command"],
        },
        "evaluation_spec": {
            "path": (output / "evaluation_spec.json").relative_to(context["root"]).as_posix(),
            "sha256": spec_file_hash,
            "frozen": True,
        },
        "source_receipt": context["source"],
        "registered": context["registered"],
        "prepare": {
            "receipt": {
                "path": (output / "prepare_receipt.json").relative_to(context["root"]).as_posix(),
                "sha256": file_sha256(output / "prepare_receipt.json"),
            },
            "artifacts": [
                {
                    "kind": kind,
                    "path": (output / name).relative_to(context["root"]).as_posix(),
                    "sha256": artifact_hashes[(output / name).relative_to(context["root"]).as_posix()],
                }
                for kind, name in PREPARE_ARTIFACTS
            ],
            "outcome_rows_read": 0,
            "outcome_artifacts_present": False,
            "outcome_blind_gates": behavior["gates"],
            "predictions_frozen": True,
            "positions_frozen": True,
            "all_checkpoints_used_without_selection": True,
            "authorizes_unseal": True,
        },
        "authorization": {
            "explicit_user_authorization": False,
            "exact_registered_unseal": False,
        },
        "unseal": None,
        "safety": {
            "target_assets_loaded": [],
            "retuning_performed": False,
            "thresholds_changed": False,
            "costs_or_accounting_changed": False,
            "second_unseal_attempted": False,
        },
        "completion": None,
        "replay": None,
    }
    write_json_atomic(output / "one_shot_packet.json", packet)
    audit = {
        "schema_version": "v64-prepare-audit/v1",
        "checks": behavior["gates"],
        "passed": True,
        "outcomes_remain_sealed": True,
        "target_assets_remain_sealed": True,
    }
    write_json_atomic(output / "audit.json", audit)
    result = {
        "schema_version": "v64-prepare-result/v1",
        "decision": "await_explicit_v64_one_shot_unseal_authorization",
        "evaluation_spec_sha256": spec_file_hash,
        "prepare_receipt_sha256": file_sha256(output / "prepare_receipt.json"),
        "one_shot_packet_sha256": file_sha256(output / "one_shot_packet.json"),
        "summary": {
            "checkpoint_count": len(checkpoints),
            "context_prediction_rows": len(contexts),
            "asset_prediction_rows": len(assets),
            "position_rows": len(positions),
            "outcome_rows_read": 0,
            "target_asset_rows": 0,
            "behavior_gate_count": len(behavior["gates"]),
            "behavior_gates_passed": True,
        },
        "audit": audit,
        "outcomes_sealed": True,
        "target_assets_sealed": True,
        "only_next_action": "obtain_explicit_user_authorization_for_exact_registered_v64_unseal",
    }
    write_json_atomic(output / "result.json", result)
    (output / "report.md").write_text(
        "# TLM V64 outcome-blind prepare\n\n"
        "Predictions and cost-specific positions are frozen. Outcomes were not opened.\n\n"
        "The only next action is explicit user authorization for the exact registered V64 unseal.\n",
        encoding="utf-8",
    )
    return result
