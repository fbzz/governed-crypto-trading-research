"""Outcome-blind V71 post-hoc preparation for the frozen V64-R2 family."""

from __future__ import annotations

from itertools import combinations
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
import yaml

from .core.artifacts import canonical_sha256, file_sha256, write_json_atomic
from .decoupled_rank_state_harness import derive_state_features
from .non_target_dataset import PANEL_FEATURES
from .non_target_pretraining import TripletTensorStore
from .research_workflow import validate_research_state
from .v64_r2_probabilistic_state_gate_training_engine import configure_v68_runtime
from .v64_r2_prospective_capture import (
    EXPECTED_FOLDS,
    EXPECTED_SEEDS,
    TARGET_SYMBOLS,
    _load_inference_models,
    _policy_step,
    _scaler,
)


ACTION = "authorize_v71_posthoc_consumed_2025_diagnostic_prepare_only"
COMMAND = (
    "PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src python3 -m tlm "
    "v64-r2-retrospective-diagnostic-prepare "
    "--config configs/v71_v64_r2_retrospective_diagnostic.yaml"
)
SOURCE_FILES = (
    "src/tlm/v64_r2_retrospective_diagnostic.py",
    "src/tlm/v64_r2_prospective_capture.py",
    "src/tlm/__main__.py",
    "src/tlm/research_workflow.py",
    "configs/v71_v64_r2_retrospective_diagnostic.yaml",
    "research/current.yaml",
    "research/phase_contracts/v071.yaml",
    "research/incidents/v071_prepare_schema_probe_projection_gap.json",
)
POSITION_COLUMNS = (
    "date",
    "fold",
    "cost_bps",
    "symbol",
    "eligible",
    "candidate_weight",
    "selected_symbol",
    "action",
    "transition_turnover",
    "final_liquidation_turnover",
    "total_turnover",
    "gross_exposure",
)
BLIND_GATE_NAMES = (
    "all_registered_checkpoints_used_without_selection",
    "exact_fold_asset_triplet_and_date_scope",
    "missingness_matches_registered_readiness",
    "prediction_distribution_finite_and_nonconstant",
    "probabilistic_scale_strictly_positive",
    "action_coverage_complete",
    "turnover_accounting_complete",
    "concentration_at_most_one_asset",
    "exposure_bounded_zero_to_one",
    "exact_v64_control_positions_reused",
    "equal_weight_control_complete",
    "post_anchor_feature_projection_only",
    "zero_evaluation_outcome_and_target_access",
)


class V71DiagnosticError(RuntimeError):
    """Raised when the frozen V71 preparation contract cannot be preserved."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V71DiagnosticError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V71DiagnosticError(f"Unable to load JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise V71DiagnosticError(f"Expected JSON object: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise V71DiagnosticError(f"Unable to load YAML {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise V71DiagnosticError(f"Expected YAML mapping: {path}")
    return value


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise V71DiagnosticError(
            f"git {' '.join(args)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def resolve_prepare_anchor(
    root: Path, anchor_contract: Mapping[str, Any]
) -> dict[str, Any]:
    relative = str(anchor_contract["incident_path"])
    expected = str(anchor_contract["incident_file_sha256"])
    commits = _git(root, "rev-list", "--reverse", "HEAD", "--", relative).splitlines()
    for commit in commits:
        payload = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=root,
            capture_output=True,
            check=False,
        )
        if payload.returncode != 0:
            continue
        observed = hashlib.sha256(payload.stdout).hexdigest()
        if observed == expected:
            return {
                "commit": commit,
                "commit_timestamp_utc": _git(root, "show", "-s", "--format=%cI", commit),
                "incident_path": relative,
                "incident_file_sha256": expected,
            }
    raise V71DiagnosticError("V71-R1 prepare anchor commit was not found")


def _source_receipt(root: Path, anchor: Mapping[str, Any]) -> dict[str, Any]:
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    _require(status == "", "V71 source tree must be clean before preparation")
    files = {relative: file_sha256(root / relative) for relative in SOURCE_FILES}
    return {
        "schema_version": "v71-source-receipt/v1",
        "git_clean": True,
        "git_head": _git(root, "rev-parse", "HEAD"),
        "files": files,
        "bundle_sha256": canonical_sha256(files),
        "prepare_registration_anchor": dict(anchor),
    }


def _project_root(config: Mapping[str, Any]) -> Path:
    value = Path(str(config.get("project_root", ".")))
    return value.resolve()


def _context(config: Mapping[str, Any]) -> dict[str, Any]:
    root = _project_root(config)
    state_path = root / str(config["research_state"])
    contract_path = root / str(config["phase_contract"])
    status = validate_research_state(root, state_path.relative_to(root))
    _require(
        status["authorized_phase"] == "v71"
        and status["authorized_next_action"] == ACTION
        and status["authorized_command"] == COMMAND,
        "V71 live authorization drift",
    )
    contract = _load_yaml(contract_path)
    _require(
        contract["stage_revision"]
        == "v071_posthoc_consumed_2025_diagnostic_prepare_r2",
        "V71 phase revision drift",
    )
    configured_output = str(config["output_dir"])
    _require(
        configured_output == contract["access_contract"]["output_dir"],
        "V71 output directory drift",
    )
    output = root / configured_output
    hashes = contract["input_contract"]["expected_static_file_sha256_by_path"]
    _require(
        set(hashes) == set(contract["access_contract"]["allowed_inputs"]),
        "V71 input allowlist/hash registry drift",
    )
    for relative, expected in hashes.items():
        path = root / relative
        _require(
            path.is_file() and file_sha256(path) == expected,
            f"V71 static input drift: {relative}",
        )
    anchor = resolve_prepare_anchor(
        root, contract["prepare_registration_anchor_contract"]
    )
    source = _source_receipt(root, anchor)
    inputs = config["inputs"]
    metadata = {
        "blueprint": _load_json(root / str(inputs["blueprint"])),
        "asset_folds": _load_json(root / str(inputs["asset_folds"])),
        "triplet_catalog": _load_json(root / str(inputs["triplet_catalog"])),
        "checkpoint_manifest": _load_json(root / str(inputs["checkpoint_manifest"])),
        "scaler_manifest": _load_json(root / str(inputs["scaler_manifest"])),
        "ranker_scale_receipt": _load_json(root / str(inputs["ranker_scale_receipt"])),
        "v64_spec": _load_json(root / str(inputs["v64_evaluation_spec"])),
        "v64_prepare": _load_json(root / str(inputs["v64_prepare_receipt"])),
        "outcome_receipt": _load_json(root / str(inputs["sealed_outcome_receipt"])),
    }
    sealed = contract["sealed_outcome_contract"]
    _require(
        metadata["outcome_receipt"].get("outcome_packet_sha256")
        == sealed["packet_sha256"]
        and metadata["outcome_receipt"].get("unseal_count") == 1,
        "V71 registered consumed outcome receipt drift",
    )
    _require(
        metadata["blueprint"]["policy"]["reporting_cost_bps"]
        == contract["diagnostic_contract"]["reporting_cost_bps"],
        "V71 policy/cost registry drift",
    )
    return {
        "root": root,
        "state_path": state_path,
        "contract_path": contract_path,
        "status": status,
        "contract": contract,
        "config": dict(config),
        "inputs": dict(inputs),
        "output": output,
        "source_receipt": source,
        **metadata,
    }


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    result = pd.Timestamp(value)
    return result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")


def read_projected_parquet(
    path: Path,
    columns: Sequence[str],
    symbols: Sequence[str],
    minimum: pd.Timestamp,
    maximum: pd.Timestamp,
    *,
    require_exact_symbols: bool = True,
    forbidden_tokens: Sequence[str] = ("target", "label", "outcome", "raw_"),
) -> pd.DataFrame:
    """Read only an explicit column projection and bounded non-target key range."""

    _require(set(symbols).isdisjoint(TARGET_SYMBOLS), "target symbol in V71 predicate")
    predicate = (
        ds.field("symbol").isin(list(symbols))
        & (ds.field("date") >= minimum.to_pydatetime())
        & (ds.field("date") <= maximum.to_pydatetime())
    )
    table = ds.dataset(path, format="parquet").to_table(
        columns=list(columns), filter=predicate, use_threads=False
    )
    frame = table.to_pandas()
    _require(list(frame.columns) == list(columns), f"V71 projection drift: {path}")
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame["symbol"] = frame["symbol"].astype(str)
    _require(
        not frame.duplicated(["date", "symbol"]).any(),
        f"V71 duplicate keys: {path}",
    )
    loaded = set(frame["symbol"].unique())
    if require_exact_symbols:
        _require(loaded == set(symbols), f"V71 symbol predicate drift: {path}")
    else:
        _require(
            bool(loaded) and loaded.issubset(symbols),
            f"V71 readiness predicate drift: {path}",
        )
    _require(loaded.isdisjoint(TARGET_SYMBOLS), "target asset reached V71 frame")
    _require(
        not any(
            any(token in column.lower() for token in forbidden_tokens)
            for column in frame.columns
        ),
        f"V71 forbidden projected column: {path}",
    )
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def _fold_rows(value: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["fold"]): dict(row) for row in value["folds"]}


def _fold_data(context: Mapping[str, Any], fold: int) -> dict[str, Any]:
    contract = context["contract"]
    projection = contract["projection_contract"]
    diagnostic = contract["diagnostic_contract"]
    folds = _fold_rows(context["asset_folds"])
    catalogs = _fold_rows(context["triplet_catalog"])
    symbols = sorted(str(value) for value in folds[fold]["test_symbols"])
    triplets = [tuple(str(value) for value in row) for row in catalogs[fold]["test_triplets"]]
    _require(
        len(symbols) == 10
        and set(symbols).isdisjoint(TARGET_SYMBOLS)
        and triplets == list(combinations(symbols, 3)),
        f"V71 fold {fold} scope drift",
    )
    start = _utc(diagnostic["signal_start"])
    end = _utc(diagnostic["signal_end"])
    feature_start = start - pd.Timedelta(days=int(diagnostic["lookback_days"]) - 1)
    panel = read_projected_parquet(
        context["root"] / context["inputs"]["feature_panel"],
        projection["feature_panel_columns"],
        symbols,
        feature_start,
        end,
    )
    roles = read_projected_parquet(
        context["root"] / context["inputs"]["readiness_roles"],
        projection["readiness_columns"],
        symbols,
        start,
        end,
        require_exact_symbols=False,
        forbidden_tokens=("target", "outcome", "raw_"),
    )
    roles["sequence_start_date"] = pd.to_datetime(
        roles["sequence_start_date"], utc=True
    )
    ready = roles.loc[
        roles[str(context["config"]["evaluation"]["readiness_flag"])].astype(bool)
        & roles["h1_label_complete"].astype(bool),
        ["date", "symbol"],
    ]
    availability = {
        pd.Timestamp(date): tuple(sorted(frame["symbol"].unique()))
        for date, frame in ready.groupby("date", sort=True)
    }
    dates = pd.date_range(start, end, freq="D", tz="UTC")
    _require(
        len(dates) == int(diagnostic["expected_signal_dates"])
        and set(availability) == set(dates),
        f"V71 fold {fold} readiness date drift",
    )
    registered = set(triplets)
    samples: list[dict[str, Any]] = []
    for date in dates:
        current = availability[pd.Timestamp(date)]
        for triplet in combinations(current, 3):
            if triplet in registered:
                samples.append({"date": pd.Timestamp(date), "triplet": triplet})
    _require(samples, f"V71 fold {fold} has no eligible contexts")
    return {
        "symbols": symbols,
        "dates": dates,
        "panel": panel,
        "roles": roles,
        "availability": availability,
        "samples": samples,
    }


def _equal_weight_positions(
    fold: int,
    symbols: Sequence[str],
    dates: Sequence[pd.Timestamp],
    availability: Mapping[pd.Timestamp, Sequence[str]],
    costs: Sequence[int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cost in costs:
        previous = {symbol: 0.0 for symbol in symbols}
        for date_index, raw_date in enumerate(dates):
            date = pd.Timestamp(raw_date)
            eligible = tuple(availability[date])
            weight = 1.0 / len(eligible)
            weights = {symbol: weight if symbol in eligible else 0.0 for symbol in symbols}
            transition = float(
                sum(abs(weights[symbol] - previous[symbol]) for symbol in symbols)
            )
            liquidation = float(sum(abs(value) for value in weights.values())) if date_index == len(dates) - 1 else 0.0
            total = transition + liquidation
            for symbol in symbols:
                rows.append(
                    {
                        "date": date,
                        "fold": int(fold),
                        "cost_bps": int(cost),
                        "symbol": symbol,
                        "eligible": symbol in eligible,
                        "candidate_weight": float(weights[symbol]),
                        "selected_symbol": None,
                        "action": "equal_weight",
                        "transition_turnover": transition,
                        "final_liquidation_turnover": liquidation,
                        "total_turnover": total,
                        "gross_exposure": float(sum(weights.values())),
                    }
                )
            previous = weights
    return pd.DataFrame(rows, columns=POSITION_COLUMNS)


def _infer_fold(
    context: Mapping[str, Any],
    fold: int,
    models: Sequence[tuple[int, torch.nn.Module, torch.nn.Module]],
    device: torch.device,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
]:
    data = _fold_data(context, fold)
    symbols = data["symbols"]
    dates = data["dates"]
    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
    date_index = {pd.Timestamp(date): index for index, date in enumerate(dates)}
    shape = (len(dates), len(symbols))
    excess_sum = np.zeros(shape, dtype=np.float64)
    excess_count = np.zeros(shape, dtype=np.int64)
    mixture_rows: list[dict[str, Any]] = []
    mixture_by_date: dict[pd.Timestamp, list[dict[str, Any]]] = {
        pd.Timestamp(date): [] for date in dates
    }
    scale_row = _fold_rows(context["scaler_manifest"])[fold]
    ranker_scale_row = _fold_rows(context["ranker_scale_receipt"])[fold]
    scaler = _scaler(scale_row)
    _require(tuple(PANEL_FEATURES) == scaler.feature_names, "V71 feature scaler drift")
    store = TripletTensorStore(
        data["panel"],
        list(PANEL_FEATURES),
        int(context["contract"]["diagnostic_contract"]["lookback_days"]),
        "log_close_to_close_return",
    )
    batch_size = int(context["config"]["runtime"]["inference_batch_size"])
    excess_rms = float(ranker_scale_row["ranker_excess_rms"])
    market_rms = float(scale_row["market_target_rms"])
    for start in range(0, len(data["samples"]), batch_size):
        batch = data["samples"][start : start + batch_size]
        x_np = store.materialize_batch(batch, scaler)
        x = torch.from_numpy(x_np).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            state = derive_state_features(x)
            for seed, ranker, gate in models:
                ranker_output = ranker(x)
                gate_output = gate(state)
                centered = ranker_output["excess_return_z"] - ranker_output[
                    "excess_return_z"
                ].mean(dim=1, keepdim=True)
                raw_excess = (centered * excess_rms).detach().cpu().numpy()
                locations = (gate_output["location"] * market_rms).detach().cpu().numpy()
                scales = (gate_output["scale"] * market_rms).detach().cpu().numpy()
                _require(
                    np.isfinite(raw_excess).all()
                    and np.isfinite(locations).all()
                    and np.isfinite(scales).all()
                    and bool((scales > 0.0).all()),
                    f"V71 non-finite prediction in fold {fold}",
                )
                for offset, sample in enumerate(batch):
                    date = pd.Timestamp(sample["date"])
                    triplet = tuple(str(value) for value in sample["triplet"])
                    mixture = {
                        "date": date,
                        "fold": int(fold),
                        "triplet_key": "|".join(triplet),
                        "seed": int(seed),
                        "market_location": float(locations[offset]),
                        "market_scale": float(scales[offset]),
                    }
                    mixture_rows.append(mixture)
                    mixture_by_date[date].append(mixture)
                    day = date_index[date]
                    for slot, symbol in enumerate(triplet):
                        asset = symbol_index[symbol]
                        excess_sum[day, asset] += float(raw_excess[offset, slot])
                        excess_count[day, asset] += 1
        del x, x_np, state

    eligible = np.zeros(shape, dtype=bool)
    expected_count = np.zeros(shape, dtype=np.int64)
    for day, date in enumerate(dates):
        current = data["availability"][pd.Timestamp(date)]
        for symbol in current:
            asset = symbol_index[symbol]
            eligible[day, asset] = True
            expected_count[day, asset] = math.comb(len(current) - 1, 2) * len(models)
        _require(
            len(mixture_by_date[pd.Timestamp(date)])
            == math.comb(len(current), 3) * len(models),
            f"V71 fold {fold} mixture count drift",
        )
    _require(
        np.array_equal(excess_count, expected_count),
        f"V71 fold {fold} ranker aggregation drift",
    )
    raw_excess = np.divide(
        excess_sum,
        excess_count,
        out=np.full(shape, np.nan),
        where=excess_count > 0,
    )
    _require(
        np.isfinite(raw_excess[eligible]).all(),
        f"V71 fold {fold} asset prediction drift",
    )

    momentum_lookup: dict[tuple[pd.Timestamp, str], float] = {}
    for symbol, frame in data["panel"].groupby("symbol", sort=True):
        current = frame.sort_values("date").copy()
        current["momentum_30"] = current["log_close_to_close_return"].rolling(
            30, min_periods=30
        ).sum()
        momentum_lookup.update(
            {
                (pd.Timestamp(row.date), str(symbol)): float(row.momentum_30)
                for row in current[["date", "momentum_30"]].itertuples(index=False)
            }
        )

    asset_rows: list[dict[str, Any]] = []
    assets_by_date: dict[pd.Timestamp, list[dict[str, Any]]] = {}
    for day, raw_date in enumerate(dates):
        date = pd.Timestamp(raw_date)
        mixtures = mixture_by_date[date]
        market_mean = float(np.mean([row["market_location"] for row in mixtures]))
        market_scale_mean = float(np.mean([row["market_scale"] for row in mixtures]))
        current_rows = []
        for asset, symbol in enumerate(symbols):
            if not eligible[day, asset]:
                continue
            row = {
                "date": date,
                "fold": int(fold),
                "symbol": symbol,
                "eligible": True,
                "context_seed_count": int(excess_count[day, asset]),
                "raw_excess": float(raw_excess[day, asset]),
                "market_location_mean": market_mean,
                "market_scale_mean": market_scale_mean,
                "absolute_location": float(market_mean + raw_excess[day, asset]),
                "momentum_30": float(momentum_lookup[(date, symbol)]),
            }
            _require(math.isfinite(row["momentum_30"]), "V71 momentum drift")
            asset_rows.append(row)
            current_rows.append(row)
        assets_by_date[date] = current_rows

    costs = [int(value) for value in context["blueprint"]["policy"]["reporting_cost_bps"]]
    previous = {(fold, cost): None for cost in costs}
    position_rows: list[dict[str, Any]] = []
    action_counts = {cost: {} for cost in costs}
    for raw_date in dates:
        date = pd.Timestamp(raw_date)
        registered_symbols = list(symbols)
        eligible_symbols = list(data["availability"][date])
        fold_prediction = {
            "fold": int(fold),
            "registered_symbols": registered_symbols,
            "eligible_symbols": eligible_symbols,
            "assets": assets_by_date[date],
            "market_mixture": mixture_by_date[date],
        }
        for cost in costs:
            cell = _policy_step(
                fold_prediction,
                previous[(fold, cost)],
                cost_bps=cost,
                policy=context["blueprint"]["policy"],
            )
            previous[(fold, cost)] = cell["selected_symbol"]
            action_counts[cost][cell["action"]] = (
                action_counts[cost].get(cell["action"], 0) + 1
            )
            for symbol in symbols:
                position_rows.append(
                    {
                        "date": date,
                        "fold": int(fold),
                        "cost_bps": int(cost),
                        "symbol": symbol,
                        "eligible": symbol in eligible_symbols,
                        "candidate_weight": float(cell["weights"][symbol]),
                        "selected_symbol": cell["selected_symbol"],
                        "action": cell["action"],
                        "transition_turnover": float(cell["transition_turnover"]),
                        "final_liquidation_turnover": 0.0,
                        "total_turnover": float(cell["transition_turnover"]),
                        "gross_exposure": float(cell["gross_exposure"]),
                    }
                )
    positions = pd.DataFrame(position_rows, columns=POSITION_COLUMNS)
    final_date = pd.Timestamp(dates[-1])
    for cost in costs:
        mask = (positions["date"] == final_date) & (positions["cost_bps"] == cost)
        liquidation = float(positions.loc[mask, "gross_exposure"].iloc[0])
        positions.loc[mask, "final_liquidation_turnover"] = liquidation
        positions.loc[mask, "total_turnover"] = (
            positions.loc[mask, "transition_turnover"] + liquidation
        )
    equal_weight = _equal_weight_positions(
        fold, symbols, list(dates), data["availability"], costs
    )
    diagnostics = {
        "fold": int(fold),
        "signal_dates": len(dates),
        "test_symbols": list(symbols),
        "eligible_asset_dates": int(eligible.sum()),
        "triplet_contexts": len(data["samples"]),
        "market_mixture_rows": len(mixture_rows),
        "minimum_ready_assets": int(eligible.sum(axis=1).min()),
        "maximum_ready_assets": int(eligible.sum(axis=1).max()),
        "minimum_context_seed_count": int(excess_count[eligible].min()),
        "maximum_context_seed_count": int(excess_count[eligible].max()),
        "raw_excess_std": float(np.std(raw_excess[eligible], ddof=0)),
        "market_location_std": float(
            np.std([row["market_location"] for row in mixture_rows], ddof=0)
        ),
        "market_scale_minimum": float(
            min(row["market_scale"] for row in mixture_rows)
        ),
        "action_counts": {str(cost): values for cost, values in action_counts.items()},
        "panel_projected_rows": len(data["panel"]),
        "readiness_projected_rows": len(data["roles"]),
        "evaluation_outcome_rows_read": 0,
        "target_assets_loaded": [],
    }
    return (
        pd.DataFrame(asset_rows),
        pd.DataFrame(mixture_rows),
        positions,
        equal_weight,
        diagnostics,
    )


def _turnover_matches(frame: pd.DataFrame) -> bool:
    for (_, _, _), current in frame.groupby(
        ["fold", "cost_bps", "date"], sort=True
    ):
        if current["total_turnover"].nunique() != 1:
            return False
    for (fold, cost), current in frame.groupby(["fold", "cost_bps"], sort=True):
        pivot = current.pivot(
            index="date", columns="symbol", values="candidate_weight"
        ).sort_index()
        values = pivot.to_numpy(dtype=np.float64)
        previous = np.vstack([np.zeros((1, values.shape[1])), values[:-1]])
        observed = np.abs(values - previous).sum(axis=1)
        observed[-1] += np.abs(values[-1]).sum()
        registered = (
            current.groupby("date", sort=True)["total_turnover"].first().to_numpy()
        )
        if not np.allclose(observed, registered, atol=1.0e-12):
            return False
    return True


def _behavior_gates(
    context: Mapping[str, Any],
    assets: pd.DataFrame,
    mixtures: pd.DataFrame,
    positions: pd.DataFrame,
    equal_weight: pd.DataFrame,
    control: pd.DataFrame,
    checkpoint_receipts: Sequence[Mapping[str, Any]],
    diagnostics: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    expected_dates = int(context["contract"]["diagnostic_contract"]["expected_signal_dates"])
    costs = context["contract"]["diagnostic_contract"]["reporting_cost_bps"]
    decision_rows = positions.drop_duplicates(["date", "fold", "cost_bps"])
    gates = {
        "all_registered_checkpoints_used_without_selection": len(checkpoint_receipts) == 9
        and {(int(row["fold"]), int(row["seed"])) for row in checkpoint_receipts}
        == {(fold, seed) for fold in EXPECTED_FOLDS for seed in EXPECTED_SEEDS}
        and all(row["used_without_selection"] for row in checkpoint_receipts),
        "exact_fold_asset_triplet_and_date_scope": all(
            row["signal_dates"] == expected_dates
            and len(row["test_symbols"]) == 10
            and row["triplet_contexts"] > 0
            for row in diagnostics
        ),
        "missingness_matches_registered_readiness": len(assets)
        == sum(int(row["eligible_asset_dates"]) for row in diagnostics)
        and bool(assets["eligible"].all()),
        "prediction_distribution_finite_and_nonconstant": bool(
            np.isfinite(
                assets[
                    [
                        "raw_excess",
                        "market_location_mean",
                        "market_scale_mean",
                        "absolute_location",
                        "momentum_30",
                    ]
                ].to_numpy(dtype=np.float64)
            ).all()
        )
        and all(
            float(row["raw_excess_std"]) > 0.0
            and float(row["market_location_std"]) > 0.0
            for row in diagnostics
        ),
        "probabilistic_scale_strictly_positive": bool(
            np.isfinite(mixtures["market_scale"].to_numpy(dtype=np.float64)).all()
            and (mixtures["market_scale"] > 0.0).all()
        ),
        "action_coverage_complete": len(decision_rows)
        == len(EXPECTED_FOLDS) * expected_dates * len(costs)
        and set(decision_rows["action"]).issubset(
            {
                "cash",
                "entry",
                "hold",
                "switch",
                "momentum_exit",
                "probability_exit",
                "forced_exit",
            }
        ),
        "turnover_accounting_complete": _turnover_matches(positions),
        "concentration_at_most_one_asset": bool(
            (
                positions.groupby(["date", "fold", "cost_bps"])[
                    "candidate_weight"
                ].apply(lambda values: int((values > 0.0).sum()))
                <= 1
            ).all()
        ),
        "exposure_bounded_zero_to_one": bool(
            (positions["candidate_weight"] >= 0.0).all()
            and (decision_rows["gross_exposure"] <= 1.0 + 1.0e-12).all()
        ),
        "exact_v64_control_positions_reused": file_sha256(
            context["root"] / context["inputs"]["v64_control_positions"]
        )
        == "7722c68e522fba1a3bb708b803d08230677920998255fc3c37d697c1096cd88f"
        and list(control.columns) == list(POSITION_COLUMNS)
        and len(control) == len(positions),
        "equal_weight_control_complete": len(equal_weight) == len(positions)
        and _turnover_matches(equal_weight)
        and bool(
            np.allclose(
                equal_weight.drop_duplicates(["date", "fold", "cost_bps"])[
                    "gross_exposure"
                ].to_numpy(dtype=np.float64),
                1.0,
                atol=1.0e-12,
            )
        ),
        "post_anchor_feature_projection_only": all(
            row["panel_projected_rows"] > 0
            and row["readiness_projected_rows"] > 0
            for row in diagnostics
        ),
        "zero_evaluation_outcome_and_target_access": all(
            row["evaluation_outcome_rows_read"] == 0
            and row["target_assets_loaded"] == []
            for row in diagnostics
        )
        and set(assets["symbol"]).isdisjoint(TARGET_SYMBOLS)
        and set(positions["symbol"]).isdisjoint(TARGET_SYMBOLS),
    }
    _require(set(gates) == set(BLIND_GATE_NAMES), "V71 blind gate registry drift")
    return {
        "schema_version": "v71-outcome-blind-behavior-gates/v1",
        "gates": gates,
        "passed": all(gates.values()),
        "diagnostics": list(diagnostics),
    }


def _evaluation_spec(context: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = context["contract"]
    accounting = context["v64_spec"]["accounting"]
    controls = {
        "candidate": "exact_v65_probabilistic_abstention_policy",
        "v64_control_positions": {
            "path": context["inputs"]["v64_control_positions"],
            "sha256": "7722c68e522fba1a3bb708b803d08230677920998255fc3c37d697c1096cd88f",
        },
        "cash": "all_zero_weights",
        "equal_weight": "equal_weight_currently_eligible_fold_assets",
    }
    registered_body = {
        "cost_bps": list(contract["diagnostic_contract"]["reporting_cost_bps"]),
        "accounting": accounting,
        "controls": controls,
        "gates": contract["diagnostic_contract"]["outcome_dependent_gates"],
        "outcome_blind_gate_names": list(BLIND_GATE_NAMES),
    }
    registered = {**registered_body, "sha256": canonical_sha256(registered_body)}
    spec = {
        "schema_version": "v71-v64-r2-posthoc-diagnostic-spec/v1",
        "phase": "v71",
        "family_id": contract["family_id"],
        "lineage_label": contract["lineage_label"],
        "frozen": True,
        "evidence_tier": contract["evidence_tier"],
        "interpretation": contract["diagnostic_contract"]["interpretation"],
        "window": {
            "signal_start": contract["diagnostic_contract"]["signal_start"],
            "signal_end": contract["diagnostic_contract"]["signal_end"],
            "expected_signal_dates": contract["diagnostic_contract"][
                "expected_signal_dates"
            ],
            "consumed_by_scientific_parent": True,
        },
        "inference": {
            "checkpoint_count": 9,
            "folds": list(EXPECTED_FOLDS),
            "seeds": list(EXPECTED_SEEDS),
            "lookback_days": contract["diagnostic_contract"]["lookback_days"],
            "context_aggregation": contract["diagnostic_contract"][
                "context_aggregation"
            ],
            "seed_aggregation": contract["diagnostic_contract"]["seed_aggregation"],
        },
        "policy": context["blueprint"]["policy"],
        "accounting": accounting,
        "controls": controls,
        "bootstrap": contract["diagnostic_contract"]["bootstrap"],
        "outcome_dependent_gates": contract["diagnostic_contract"][
            "outcome_dependent_gates"
        ],
        "outcome_blind_gate_names": list(BLIND_GATE_NAMES),
        "registered_sha256": registered["sha256"],
        "sealed_outcome": contract["sealed_outcome_contract"],
        "projection_contract": contract["projection_contract"],
        "access_incident": contract["access_incident"],
        "target_contract": contract["target_contract"],
        "lifecycle": {
            "prepare_is_outcome_blind": True,
            "explicit_hash_bound_authorization_after_prepare_required": True,
            "maximum_diagnostic_unseal_count": 1,
            "retuning_or_regeneration_after_unseal": False,
            "clean_holdout_prospective_deployable_or_target_claim": False,
        },
    }
    return spec, registered


def _read_control(context: Mapping[str, Any]) -> pd.DataFrame:
    path = context["root"] / context["inputs"]["v64_control_positions"]
    _require(
        file_sha256(path)
        == context["contract"]["input_contract"][
            "expected_static_file_sha256_by_path"
        ][context["inputs"]["v64_control_positions"]],
        "V71 V64 control hash drift",
    )
    diagnostic = context["contract"]["diagnostic_contract"]
    symbols = sorted(
        {
            str(symbol)
            for row in context["asset_folds"]["folds"]
            for symbol in row["test_symbols"]
        }
    )
    predicate = (
        ds.field("symbol").isin(symbols)
        & (ds.field("date") >= _utc(diagnostic["signal_start"]).to_pydatetime())
        & (ds.field("date") <= _utc(diagnostic["signal_end"]).to_pydatetime())
    )
    table = ds.dataset(path, format="parquet").to_table(
        columns=list(POSITION_COLUMNS), filter=predicate, use_threads=False
    )
    frame = table.to_pandas()
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame["symbol"] = frame["symbol"].astype(str)
    _require(
        list(frame.columns) == list(POSITION_COLUMNS)
        and len(frame) == 32130
        and set(frame["symbol"]).isdisjoint(TARGET_SYMBOLS),
        "V71 V64 control scope drift",
    )
    return frame.sort_values(["date", "fold", "cost_bps", "symbol"]).reset_index(
        drop=True
    )


def _cached(context: Mapping[str, Any]) -> dict[str, Any] | None:
    output = context["output"]
    manifest_path = output / "artifact_manifest.json"
    result_path = output / "result.json"
    if not manifest_path.is_file() and not result_path.is_file():
        return None
    _require(manifest_path.is_file() and result_path.is_file(), "partial V71 cache")
    manifest = _load_json(manifest_path)
    for relative, expected in manifest["artifact_hashes"].items():
        path = context["root"] / relative
        _require(path.is_file() and file_sha256(path) == expected, f"V71 cache drift: {relative}")
    result = _load_json(result_path)
    return {
        **result,
        "replay": {
            "reused_frozen_prepare": True,
            "new_predictions": 0,
            "new_positions": 0,
            "checkpoint_deserializations": 0,
            "feature_parquet_deserializations": 0,
            "outcome_packet_deserializations": 0,
        },
    }


def run_v64_r2_retrospective_diagnostic_prepare(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the post-hoc V71 predictions and positions without outcomes."""

    context = _context(config)
    cached = _cached(context)
    if cached is not None:
        return cached
    output = context["output"]
    _require(not output.exists(), "V71 output exists without a complete cache")
    output.mkdir(parents=True)
    spec, registered = _evaluation_spec(context)
    device_name = str(context["config"]["runtime"]["device"])
    device = configure_v68_runtime(device_name, seed=20260715)
    models, checkpoint_receipts = _load_inference_models(context, device)

    asset_frames = []
    mixture_frames = []
    position_frames = []
    equal_frames = []
    diagnostics = []
    for fold in EXPECTED_FOLDS:
        assets, mixtures, positions, equal_weight, current_diagnostics = _infer_fold(
            context, fold, models[fold], device
        )
        asset_frames.append(assets)
        mixture_frames.append(mixtures)
        position_frames.append(positions)
        equal_frames.append(equal_weight)
        diagnostics.append(current_diagnostics)
        for _, ranker, gate in models[fold]:
            ranker.to("cpu")
            gate.to("cpu")
        if device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()

    assets = pd.concat(asset_frames, ignore_index=True).sort_values(
        ["date", "fold", "symbol"]
    ).reset_index(drop=True)
    mixtures = pd.concat(mixture_frames, ignore_index=True).sort_values(
        ["date", "fold", "triplet_key", "seed"]
    ).reset_index(drop=True)
    positions = pd.concat(position_frames, ignore_index=True).sort_values(
        ["date", "fold", "cost_bps", "symbol"]
    ).reset_index(drop=True)
    equal_weight = pd.concat(equal_frames, ignore_index=True).sort_values(
        ["date", "fold", "cost_bps", "symbol"]
    ).reset_index(drop=True)
    control = _read_control(context)
    behavior = _behavior_gates(
        context,
        assets,
        mixtures,
        positions,
        equal_weight,
        control,
        checkpoint_receipts,
        diagnostics,
    )
    _require(behavior["passed"], "V71 outcome-blind behavior gate failed")

    write_json_atomic(output / "evaluation_spec.json", spec)
    write_json_atomic(output / "source_receipt.json", context["source_receipt"])
    write_json_atomic(
        output / "input_hash_receipt.json",
        {
            "schema_version": "v71-input-hash-receipt/v1",
            "files": context["contract"]["input_contract"][
                "expected_static_file_sha256_by_path"
            ],
            "bundle_sha256": canonical_sha256(
                context["contract"]["input_contract"][
                    "expected_static_file_sha256_by_path"
                ]
            ),
        },
    )
    write_json_atomic(
        output / "checkpoint_receipt.json",
        {
            "schema_version": "v71-checkpoint-receipt/v1",
            "checkpoints": checkpoint_receipts,
            "checkpoint_count": len(checkpoint_receipts),
            "all_used_without_selection": True,
            "optimizer_created": False,
        },
    )
    data_access = {
        "schema_version": "v71-prepare-data-access/v1",
        "feature_panel_reads": 3,
        "readiness_role_reads": 3,
        "control_position_reads": 1,
        "outcome_receipt_metadata_reads": 1,
        "sealed_outcome_packet_reads": 0,
        "evaluation_outcome_rows_read": 0,
        "post_anchor_forbidden_columns_projected": [],
        "target_assets_loaded": [],
        "registered_pre_anchor_incident": {
            "training_era_rows_displayed": 2,
            "evaluation_window_outcome_rows_read": 0,
            "path": context["contract"]["access_incident"]["path"],
            "sha256": context["contract"]["access_incident"]["file_sha256"],
        },
        "feature_projection": context["contract"]["projection_contract"][
            "feature_panel_columns"
        ],
        "readiness_projection": context["contract"]["projection_contract"][
            "readiness_columns"
        ],
        "folds": diagnostics,
    }
    write_json_atomic(output / "data_access_receipt.json", data_access)
    write_json_atomic(output / "behavior_gates.json", behavior)
    _atomic_parquet(assets, output / "asset_predictions.parquet")
    _atomic_parquet(mixtures, output / "market_mixture.parquet")
    _atomic_parquet(positions, output / "candidate_positions.parquet")
    _atomic_parquet(equal_weight, output / "equal_weight_positions.parquet")

    prepare_artifacts = [
        ("predictions", output / "asset_predictions.parquet"),
        ("predictions", output / "market_mixture.parquet"),
        ("positions", output / "candidate_positions.parquet"),
        (
            "control_positions",
            context["root"] / context["inputs"]["v64_control_positions"],
        ),
        ("control_positions", output / "equal_weight_positions.parquet"),
        ("behavior", output / "behavior_gates.json"),
        ("access", output / "data_access_receipt.json"),
        ("checkpoint", output / "checkpoint_receipt.json"),
        ("inputs", output / "input_hash_receipt.json"),
        ("source", output / "source_receipt.json"),
    ]
    artifact_hashes = {
        path.relative_to(context["root"]).as_posix(): file_sha256(path)
        for _, path in prepare_artifacts
    }
    spec_hash = file_sha256(output / "evaluation_spec.json")
    prepare_receipt = {
        "schema_version": "tlm-one-shot-prepare/v1",
        "evaluation_spec_sha256": spec_hash,
        "registered_sha256": registered["sha256"],
        "artifact_hashes": artifact_hashes,
        "outcome_rows_read": 0,
        "outcome_blind_gates_passed": True,
        "authorizes_unseal": True,
    }
    write_json_atomic(output / "prepare_receipt.json", prepare_receipt)
    one_shot = {
        "schema_version": "tlm-one-shot-evaluator/v1",
        "phase": "prepare",
        "research_state": {
            "path": context["state_path"].relative_to(context["root"]).as_posix(),
            "sha256": file_sha256(context["state_path"]),
            "authorized_phase": "v71",
            "authorized_next_action": ACTION,
            "authorized_command": COMMAND,
        },
        "evaluation_spec": {
            "path": (output / "evaluation_spec.json").relative_to(
                context["root"]
            ).as_posix(),
            "sha256": spec_hash,
            "frozen": True,
        },
        "source_receipt": context["source_receipt"],
        "registered": registered,
        "prepare": {
            "receipt": {
                "path": (output / "prepare_receipt.json").relative_to(
                    context["root"]
                ).as_posix(),
                "sha256": file_sha256(output / "prepare_receipt.json"),
            },
            "artifacts": [
                {
                    "kind": kind,
                    "path": path.relative_to(context["root"]).as_posix(),
                    "sha256": artifact_hashes[
                        path.relative_to(context["root"]).as_posix()
                    ],
                }
                for kind, path in prepare_artifacts
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
            "evaluation_outcome_rows_read": 0,
            "registered_pre_anchor_training_rows_displayed": 2,
            "clean_holdout_or_prospective_claim": False,
        },
        "completion": None,
        "replay": None,
    }
    write_json_atomic(output / "one_shot_packet.json", one_shot)
    validator_command = (
        "python3 .agents/skills/tlm-one-shot-evaluator/scripts/"
        "validate_evaluation_packet.py --repo-root . --packet "
        "artifacts/v71_v64_r2_posthoc_retrospective_diagnostic/one_shot_packet.json"
    )
    authorization_text = (
        "Autorizo exatamente uma abertura V72 do packet de outcomes não-target "
        "já imutável de 2025 para o diagnóstico post-hoc V64-R2, vinculada ao "
        f"evaluation spec {spec_hash}, prepare receipt "
        f"{file_sha256(output / 'prepare_receipt.json')} e registered contract "
        f"{registered['sha256']}, sem regeneração, retuning ou mudança de "
        "política/custos, mantendo BTC, ETH e SOL selados."
    )
    result = {
        "schema_version": "v71-prepare-result/v1",
        "decision": "authorize_v72_exact_hash_bound_posthoc_outcome_unseal_only",
        "evidence_tier": context["contract"]["evidence_tier"],
        "evaluation_spec_sha256": spec_hash,
        "prepare_receipt_sha256": file_sha256(output / "prepare_receipt.json"),
        "registered_sha256": registered["sha256"],
        "one_shot_packet_sha256": file_sha256(output / "one_shot_packet.json"),
        "source_receipt_sha256": file_sha256(output / "source_receipt.json"),
        "outcomes_remain_sealed": True,
        "outcome_packet_reads": 0,
        "performance_or_pnl_computed": False,
        "target_assets_loaded": [],
        "retuning_or_retraining_performed": False,
        "checkpoint_count": len(checkpoint_receipts),
        "signal_dates": int(
            context["contract"]["diagnostic_contract"]["expected_signal_dates"]
        ),
        "required_exact_user_authorization": authorization_text,
        "validator_command": validator_command,
    }
    audit = {
        "schema_version": "v71-prepare-audit/v1",
        "checks": behavior["gates"],
        "passed": True,
        "outcomes_remain_sealed": True,
        "evaluation_outcome_rows_read": 0,
        "sealed_outcome_packet_reads": 0,
        "target_assets_loaded": [],
        "posthoc_not_confirmation": True,
    }
    report = "\n".join(
        [
            "# TLM V71 Post-hoc V64-R2 Diagnostic Prepare",
            "",
            "**Decision:** `authorize_v72_exact_hash_bound_posthoc_outcome_unseal_only`",
            "",
            "This packet freezes an immediate consumed-2025 diagnostic. It is not clean prospective evidence.",
            "",
            f"- Signal dates: {result['signal_dates']}",
            f"- Checkpoints used without selection: {result['checkpoint_count']}",
            f"- Candidate position rows: {len(positions)}",
            f"- V64 control position rows: {len(control)}",
            f"- Equal-weight control rows: {len(equal_weight)}",
            "- Outcome packet reads: 0",
            "- Financial metrics/PnL: not computed during prepare",
            "- BTC/ETH/SOL: sealed",
            "",
        ]
    )
    write_json_atomic(output / "audit.json", audit)
    write_json_atomic(output / "result.json", result)
    _atomic_text(output / "report.md", report)
    manifest_paths = [
        path
        for path in output.iterdir()
        if path.is_file() and path.name != "artifact_manifest.json"
    ]
    manifest = {
        "schema_version": "v71-artifact-manifest/v1",
        "artifact_hashes": {
            path.relative_to(context["root"]).as_posix(): file_sha256(path)
            for path in sorted(manifest_paths)
        },
        "outcome_artifacts": [],
        "outcome_packet_reads": 0,
        "target_assets_loaded": [],
    }
    write_json_atomic(output / "artifact_manifest.json", manifest)
    return {**result, "audit": audit, "summary": diagnostics}
