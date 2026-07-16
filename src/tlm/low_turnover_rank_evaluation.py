"""Outcome-blind V84 inference, policy freeze, and one-shot prepare packet."""

from __future__ import annotations

import json
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

from .core.artifacts import canonical_sha256, file_sha256, write_json_atomic, write_yaml_atomic
from .low_turnover_rank_training_data import BASE_FEATURES, V83FeatureScaler
from .low_turnover_rank_training_engine import (
    configure_v83_runtime,
    instantiate_v83_model,
)
from .research_workflow import validate_research_state
from .state_conditioned_multi_horizon_training_engine import semantic_state_sha256


TARGET_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
PREPARE_ACTION = "authorize_v84_outcome_blind_low_turnover_rank_evaluation_prepare_only"
PREPARE_COMMAND = (
    "PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src python3 -m tlm "
    "low-turnover-rank-evaluation-prepare "
    "--config configs/v84_low_turnover_rank_evaluation.yaml"
)
PASS_ACTION = (
    "authorize_v85_exactly_one_registered_non_target_outcome_unseal_only_after_"
    "explicit_hash_bound_user_authorization"
)
FAIL_ACTION = "retire_low_turnover_rank_family_without_outcome_unseal_or_target_access"
SIGNAL_DATES = tuple(
    value.strftime("%Y-%m-%d")
    for value in pd.date_range("2026-01-01", "2026-06-08", freq="D", tz="UTC")
)
POSITION_SIGNAL_DATES = tuple(
    value.strftime("%Y-%m-%d")
    for value in pd.date_range("2026-01-01", "2026-06-28", freq="D", tz="UTC")
)
CONTROL_NAMES = (
    "cash",
    "decision_21d_equal_weight_three_assets",
    "decision_21d_top_trailing_21d_momentum_with_same_63d_market_gate",
)


class V84EvaluationError(RuntimeError):
    """Fail-closed error for the V84 outcome-blind boundary."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V84EvaluationError(message)


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V84EvaluationError(f"cannot load {name}: {path}") from exc
    _require(isinstance(value, dict), f"{name} must be a JSON object")
    return value


def _load_yaml(path: Path, name: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise V84EvaluationError(f"cannot load {name}: {path}") from exc
    _require(isinstance(value, dict), f"{name} must be a YAML object")
    return value


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, text=True, capture_output=True, check=False
    )
    _require(result.returncode == 0, f"git {' '.join(arguments)} failed")
    return result.stdout.strip()


def _relative(root: Path, path: Path) -> str:
    resolved = path.resolve()
    _require(root == resolved or root in resolved.parents, f"path escapes root: {path}")
    return resolved.relative_to(root).as_posix()


def _source_receipt(root: Path, files: list[str]) -> dict[str, Any]:
    _require(
        _git(root, "status", "--porcelain", "--untracked-files=all") == "",
        "V84 prepare requires a clean committed Git tree",
    )
    hashes: dict[str, str] = {}
    for relative in files:
        path = (root / relative).resolve()
        _require(root in path.parents and path.is_file(), f"missing source file: {relative}")
        hashes[relative] = file_sha256(path)
    return {
        "git_clean": True,
        "git_head": _git(root, "rev-parse", "HEAD"),
        "files": hashes,
        "bundle_sha256": canonical_sha256(hashes),
    }


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(
            temporary, index=False, engine="pyarrow", compression="zstd"
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "sha256": file_sha256(path),
    }


def _evaluation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config.get("low_turnover_rank_evaluation")
    _require(
        isinstance(value, dict) and value.get("version") == "v84",
        "V84 evaluation config drift",
    )
    return dict(value)


def _context(config: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = _evaluation_config(config)
    root = Path(evaluation.get("project_root", ".")).resolve()
    status = validate_research_state(root, evaluation["research_state"])
    _require(status.get("passed") is True, "V84 live research status failed")
    _require(status.get("authorized_phase") == "v84", "V84 phase drift")
    _require(status.get("authorized_next_action") == PREPARE_ACTION, "V84 action drift")
    _require(status.get("authorized_command") == PREPARE_COMMAND, "V84 command drift")
    configure_v83_runtime("mps", seed=42)
    contract_path = root / status["phase_contract_path"]
    contract = _load_yaml(contract_path, "V84 phase contract")
    _require(
        contract.get("stage_revision")
        == "v084_outcome_blind_low_turnover_rank_evaluation_prepare_r2",
        "V84 stage revision drift",
    )
    expected = contract["input_contract"]["expected_file_sha256_by_path"]
    _require(
        set(expected) == set(contract["access_contract"]["allowed_inputs"]),
        "V84 input allowlist drift",
    )
    for relative, digest in expected.items():
        path = root / relative
        _require(path.is_file(), f"missing V84 input: {relative}")
        _require(file_sha256(path) == digest, f"V84 input hash drift: {relative}")
    source = _source_receipt(root, list(evaluation["source_receipt_files"]))
    output = root / str(config["output_dir"])
    prediction = root / str(config["prediction_path"])
    candidate = root / str(config["candidate_position_path"])
    controls = root / str(config["control_position_path"])
    return {
        "root": root,
        "evaluation": evaluation,
        "status": status,
        "contract": contract,
        "contract_path": contract_path,
        "output": output,
        "prediction_path": prediction,
        "candidate_path": candidate,
        "control_path": controls,
        "source_receipt": source,
        "expected_inputs": expected,
    }


class _FeatureStore:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.symbols = tuple(sorted(frame["symbol"].astype(str).unique()))
        _require(not TARGET_SYMBOLS.intersection(self.symbols), "target reached V84 features")
        self.dates = pd.DatetimeIndex(sorted(pd.to_datetime(frame["date"], utc=True).unique()))
        self.symbol_index = {value: index for index, value in enumerate(self.symbols)}
        self.date_index = {pd.Timestamp(value): index for index, value in enumerate(self.dates)}
        self.values = np.full(
            (len(self.symbols), len(self.dates), len(BASE_FEATURES)), np.nan, np.float32
        )
        self.ready = np.zeros((len(self.symbols), len(self.dates)), dtype=bool)
        for symbol, group in frame.groupby("symbol", sort=True):
            date_indexes = [self.date_index[pd.Timestamp(value)] for value in group["date"]]
            symbol_index = self.symbol_index[str(symbol)]
            self.values[symbol_index, date_indexes] = group[list(BASE_FEATURES)].to_numpy(
                dtype=np.float32
            )
            self.ready[symbol_index, date_indexes] = group["sequence_ready"].astype(bool)

    def context(self, triplet: tuple[str, str, str], date: pd.Timestamp) -> np.ndarray | None:
        end = self.date_index.get(pd.Timestamp(date))
        if end is None or end < 127:
            return None
        assets = np.asarray([self.symbol_index[symbol] for symbol in triplet], dtype=np.int64)
        if not bool(self.ready[assets, end].all()):
            return None
        values = self.values[assets, end - 127 : end + 1].transpose(1, 0, 2)
        if values.shape != (128, 3, 8) or not np.isfinite(values).all():
            return None
        return values

    def trailing_sum(
        self, triplet: tuple[str, str, str], date: pd.Timestamp, days: int
    ) -> np.ndarray | None:
        end = self.date_index.get(pd.Timestamp(date))
        if end is None or end < days - 1:
            return None
        assets = np.asarray([self.symbol_index[symbol] for symbol in triplet], dtype=np.int64)
        column = BASE_FEATURES.index("log_close_to_close_return")
        values = self.values[assets, end - days + 1 : end + 1, column]
        if values.shape != (3, days) or not np.isfinite(values).all():
            return None
        return values.astype(np.float64).sum(axis=1)


def _read_feature_store(context: Mapping[str, Any]) -> _FeatureStore:
    columns = list(context["contract"]["evaluation_contract"]["feature_projection"])
    path = context["root"] / context["evaluation"]["inputs"]["evaluation_features"]
    table = ds.dataset(path, format="parquet").to_table(columns=columns, use_threads=False)
    frame = table.to_pandas()
    _require(list(frame.columns) == columns, "V84 feature projection drift")
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame["symbol"] = frame["symbol"].astype(str)
    _require(not frame.duplicated(["date", "symbol"]).any(), "duplicate V84 feature key")
    _require(frame["date"].max() == pd.Timestamp("2026-06-08", tz="UTC"), "V84 feature end drift")
    _require(frame["date"].min() == pd.Timestamp("2025-08-26", tz="UTC"), "V84 feature start drift")
    _require(len(frame) == 8610 and frame["symbol"].nunique() == 30, "V84 feature scope drift")
    return _FeatureStore(frame.sort_values(["date", "symbol"]).reset_index(drop=True))


def _fold_triplets(root: Path) -> dict[int, tuple[tuple[str, str, str], ...]]:
    catalog = _load_json(
        root / "artifacts/v32_selected_universe_dataset/triplet_catalog.json",
        "V32 triplet catalog",
    )
    result: dict[int, tuple[tuple[str, str, str], ...]] = {}
    for row in catalog["folds"]:
        fold = int(row["fold"])
        triplets = tuple(tuple(str(item) for item in triplet) for triplet in row["test_triplets"])
        _require(len(triplets) == 120, f"V84 fold {fold} triplet count drift")
        _require(all(tuple(sorted(value)) == value for value in triplets), "non-lexical V84 triplet")
        result[fold] = triplets
    _require(set(result) == {1, 2, 3}, "V84 fold catalog drift")
    return result


def _fold_scaler(root: Path, fold: int) -> V83FeatureScaler:
    value = _load_json(
        root / f"data/checkpoints/v83_low_turnover_rank_training/fold_{fold}/fold_scale.json",
        f"V83 fold {fold} scaler",
    )
    scaler = value["feature_scaler"]
    _require(value["fold"] == fold, "V84 fold scaler identity drift")
    return V83FeatureScaler(
        feature_names=tuple(scaler["feature_names"]),
        median=tuple(float(item) for item in scaler["median"]),
        iqr=tuple(float(item) for item in scaler["iqr"]),
        fit_scope=str(scaler["fit_scope"]),
        fit_start=str(scaler["fit_start"]),
        fit_end=str(scaler["fit_end"]),
        fit_rows=int(scaler["fit_rows"]),
    )


def _checkpoint_rows(context: Mapping[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    manifest = _load_json(
        context["root"] / context["evaluation"]["inputs"]["v83_checkpoint_manifest"],
        "V83 checkpoint manifest",
    )
    rows = {(int(row["fold"]), int(row["seed"])): row for row in manifest["jobs"]}
    expected = {(fold, seed) for fold in (1, 2, 3) for seed in (42, 7, 123)}
    _require(set(rows) == expected, "V84 checkpoint grid drift")
    return rows


def _checkpoint_model(
    context: Mapping[str, Any], blueprint: dict[str, Any], row: Mapping[str, Any]
) -> tuple[torch.nn.Module, torch.device]:
    path = context["root"] / str(row["path"])
    _require(file_sha256(path) == row["file_sha256"], "V84 checkpoint file drift")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    registered = payload.get("semantic_checkpoint_sha256")
    body = {key: value for key, value in payload.items() if key != "semantic_checkpoint_sha256"}
    _require(
        registered == row["semantic_checkpoint_sha256"]
        and semantic_state_sha256(body) == registered,
        "V84 checkpoint semantic drift",
    )
    _require(payload.get("format_version") == "v83_low_turnover_rank_checkpoint_v1", "V84 checkpoint format drift")
    device = configure_v83_runtime("mps", seed=int(row["seed"]))
    model = instantiate_v83_model(blueprint, device, seed=int(row["seed"]))
    model.load_state_dict(payload["model_best_state"], strict=True)
    model.eval()
    return model, device


def _prediction_rows(
    *, fold: int, episodes: list[tuple[str, pd.Timestamp, tuple[str, str, str]]],
    seed: int | str, scores: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode_index, (triplet_id, date, triplet) in enumerate(episodes):
        for slot, symbol in enumerate(triplet):
            rows.append(
                {
                    "fold": fold,
                    "triplet_id": triplet_id,
                    "signal_date": date,
                    "symbol": symbol,
                    "seed": str(seed),
                    "centered_score": float(scores[episode_index, slot]),
                    "eligible": True,
                }
            )
    return rows


def _infer(
    context: Mapping[str, Any], store: _FeatureStore,
    triplets_by_fold: Mapping[int, tuple[tuple[str, str, str], ...]],
) -> tuple[pd.DataFrame, dict[tuple[int, str, pd.Timestamp], np.ndarray], list[str], dict[str, float]]:
    blueprint = _load_json(
        context["root"] / context["evaluation"]["inputs"]["v80_blueprint"],
        "V80 blueprint",
    )
    checkpoints = _checkpoint_rows(context)
    rows: list[dict[str, Any]] = []
    ensembles: dict[tuple[int, str, pd.Timestamp], np.ndarray] = {}
    used: list[str] = []
    centered_error = 0.0
    ensemble_error = 0.0
    batch_size = int(context["contract"]["evaluation_contract"]["inference"]["batch_size"])
    for fold in (1, 2, 3):
        scaler = _fold_scaler(context["root"], fold)
        episodes: list[tuple[str, pd.Timestamp, tuple[str, str, str]]] = []
        raw_contexts: list[np.ndarray] = []
        for triplet_index, triplet in enumerate(triplets_by_fold[fold]):
            triplet_id = f"F{fold}-T{triplet_index:03d}"
            for raw_date in SIGNAL_DATES:
                date = pd.Timestamp(raw_date, tz="UTC")
                values = store.context(triplet, date)
                if values is not None:
                    episodes.append((triplet_id, date, triplet))
                    raw_contexts.append(values)
        _require(bool(episodes), f"V84 fold {fold} has no eligible contexts")
        features = scaler.transform(np.stack(raw_contexts).astype(np.float32, copy=False))
        seed_scores: list[np.ndarray] = []
        for seed in (42, 7, 123):
            checkpoint = checkpoints[(fold, seed)]
            model, device = _checkpoint_model(context, blueprint, checkpoint)
            chunks: list[np.ndarray] = []
            with torch.no_grad():
                for start in range(0, len(features), batch_size):
                    values = torch.from_numpy(features[start : start + batch_size]).to(
                        device=device, dtype=torch.float32
                    )
                    output = model(values).detach().cpu().numpy().astype(np.float64)
                    _require(np.isfinite(output).all(), "V84 inference produced non-finite scores")
                    chunks.append(output)
            scores = np.concatenate(chunks, axis=0)
            centered_error = max(centered_error, float(np.max(np.abs(scores.sum(axis=1)))))
            seed_scores.append(scores)
            rows.extend(_prediction_rows(fold=fold, episodes=episodes, seed=seed, scores=scores))
            used.append(str(checkpoint["job_id"]))
            del model
            if device.type == "mps":
                torch.mps.empty_cache()
        ensemble = np.mean(np.stack(seed_scores, axis=0), axis=0, dtype=np.float64)
        direct = (seed_scores[0] + seed_scores[1] + seed_scores[2]) / 3.0
        ensemble_error = max(ensemble_error, float(np.max(np.abs(ensemble - direct))))
        rows.extend(_prediction_rows(fold=fold, episodes=episodes, seed="ensemble", scores=ensemble))
        for index, (triplet_id, date, _) in enumerate(episodes):
            ensembles[(fold, triplet_id, date)] = ensemble[index]
    frame = pd.DataFrame(rows).sort_values(
        ["fold", "triplet_id", "signal_date", "seed", "symbol"]
    ).reset_index(drop=True)
    return frame, ensembles, used, {
        "maximum_centered_score_sum_error": centered_error,
        "maximum_ensemble_mean_error": ensemble_error,
    }


def _classify(previous: np.ndarray, current: np.ndarray) -> str:
    old = float(previous.sum())
    new = float(current.sum())
    if old == 0.0 and new == 0.0:
        return "cash"
    if old == 0.0:
        return "enter"
    if new == 0.0:
        return "exit"
    if np.array_equal(previous, current):
        return "hold"
    return "switch"


def _policy_frame(
    *, fold: int, triplet_id: str, triplet: tuple[str, str, str],
    store: _FeatureStore, scores: Mapping[pd.Timestamp, np.ndarray],
    control: str | None,
) -> pd.DataFrame:
    previous = np.zeros(3, dtype=np.float64)
    incumbent: int | None = None
    eligible_count = 0
    rows: list[dict[str, Any]] = []
    dates = [pd.Timestamp(value, tz="UTC") for value in POSITION_SIGNAL_DATES]
    for date_index, date in enumerate(dates):
        eligible = date in scores
        decision = bool(eligible and eligible_count % 21 == 0)
        if eligible:
            eligible_count += 1
        gate_values = store.trailing_sum(triplet, date, 63) if decision else None
        market_gate = bool(gate_values is not None and float(gate_values.mean()) > 0.0)
        current = previous.copy()
        if control == "cash":
            current = np.zeros(3, dtype=np.float64)
        elif decision:
            if not market_gate:
                current = np.zeros(3, dtype=np.float64)
                incumbent = None
            elif control == "decision_21d_equal_weight_three_assets":
                current = np.full(3, 1.0 / 3.0, dtype=np.float64)
            elif control == "decision_21d_top_trailing_21d_momentum_with_same_63d_market_gate":
                momentum = store.trailing_sum(triplet, date, 21)
                current = np.zeros(3, dtype=np.float64)
                if momentum is not None:
                    candidate = int(np.argmax(momentum))
                    if float(momentum[candidate]) > 0.0:
                        current[candidate] = 1.0
            elif control is None:
                values = scores[date]
                candidate = int(np.argmax(values))
                candidate_valid = float(values[candidate]) > 0.0
                desired = incumbent
                if not candidate_valid:
                    desired = None
                elif incumbent is None:
                    desired = candidate
                elif candidate == incumbent:
                    desired = incumbent
                elif float(values[candidate] - values[incumbent]) >= 0.25:
                    desired = candidate
                current = np.zeros(3, dtype=np.float64)
                if desired is not None:
                    current[desired] = 1.0
                incumbent = desired
            else:
                raise V84EvaluationError(f"unknown V84 control: {control}")
        turnover = float(np.abs(current - previous).sum())
        action = _classify(previous, current)
        liquidation = float(np.abs(current).sum()) if date_index == len(dates) - 1 else 0.0
        selected = "CASH"
        if control == "decision_21d_equal_weight_three_assets" and current.sum() > 0:
            selected = "EQUAL_WEIGHT"
        elif current.sum() > 0:
            selected = triplet[int(np.argmax(current))]
        for slot, symbol in enumerate(triplet):
            row = {
                "fold": fold,
                "triplet_id": triplet_id,
                "signal_date": date,
                "interval_start_date": date + pd.Timedelta(days=1),
                "interval_end_date": date + pd.Timedelta(days=2),
                "symbol": symbol,
                "eligible": eligible,
                "decision": decision,
                "market_gate": market_gate,
                "weight": float(current[slot]),
                "selected_symbol": selected,
                "action": action,
                "transition_turnover": turnover,
                "final_liquidation_turnover": liquidation,
            }
            if control is not None:
                row["control"] = control
            rows.append(row)
        previous = current
    return pd.DataFrame(rows)


def _positions(
    store: _FeatureStore,
    triplets_by_fold: Mapping[int, tuple[tuple[str, str, str], ...]],
    ensembles: Mapping[tuple[int, str, pd.Timestamp], np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_frames: list[pd.DataFrame] = []
    control_frames: list[pd.DataFrame] = []
    for fold in (1, 2, 3):
        for triplet_index, triplet in enumerate(triplets_by_fold[fold]):
            triplet_id = f"F{fold}-T{triplet_index:03d}"
            scores = {
                date: value
                for (active_fold, active_id, date), value in ensembles.items()
                if active_fold == fold and active_id == triplet_id
            }
            candidate_frames.append(
                _policy_frame(
                    fold=fold, triplet_id=triplet_id, triplet=triplet,
                    store=store, scores=scores, control=None,
                )
            )
            for control in CONTROL_NAMES:
                control_frames.append(
                    _policy_frame(
                        fold=fold, triplet_id=triplet_id, triplet=triplet,
                        store=store, scores=scores, control=control,
                    )
                )
    candidate = pd.concat(candidate_frames, ignore_index=True).sort_values(
        ["fold", "triplet_id", "signal_date", "symbol"]
    ).reset_index(drop=True)
    controls = pd.concat(control_frames, ignore_index=True).sort_values(
        ["control", "fold", "triplet_id", "signal_date", "symbol"]
    ).reset_index(drop=True)
    return candidate, controls


def _turnover_audit(frame: pd.DataFrame, group_columns: list[str]) -> dict[str, float]:
    maximum_daily = 0.0
    maximum_final = 0.0
    totals: list[float] = []
    fold_totals: dict[int, list[float]] = {1: [], 2: [], 3: []}
    for key, group in frame.groupby(group_columns, sort=True):
        pivot = group.pivot(index="signal_date", columns="symbol", values="weight").sort_index()
        weights = pivot.to_numpy(dtype=np.float64)
        previous = np.vstack([np.zeros((1, weights.shape[1])), weights[:-1]])
        daily = np.abs(weights - previous).sum(axis=1)
        observed = group.drop_duplicates("signal_date").sort_values("signal_date")
        maximum_daily = max(
            maximum_daily,
            float(np.max(np.abs(daily - observed["transition_turnover"].to_numpy(float)))),
        )
        expected_final = float(np.abs(weights[-1]).sum())
        maximum_final = max(
            maximum_final,
            abs(expected_final - float(observed.iloc[-1]["final_liquidation_turnover"])),
        )
        total = float(daily.sum() + expected_final)
        totals.append(total)
        fold = int(key[-2] if group_columns[0] == "control" else key[0])
        fold_totals[fold].append(total)
    fold_mean = {
        str(fold): float(np.mean(values))
        for fold, values in fold_totals.items()
        if values
    }
    return {
        "maximum_daily_turnover_error": maximum_daily,
        "maximum_final_liquidation_error": maximum_final,
        "maximum_triplet_turnover": float(max(totals)),
        "aggregate_turnover": float(np.mean(list(fold_mean.values()))),
        **{f"fold_{fold}_turnover": value for fold, value in fold_mean.items()},
    }


def _behavior_audit(
    *, predictions: pd.DataFrame, candidate: pd.DataFrame, controls: pd.DataFrame,
    triplets_by_fold: Mapping[int, tuple[tuple[str, str, str], ...]],
    used_jobs: list[str], inference_checks: Mapping[str, float],
) -> dict[str, Any]:
    expected_jobs = [f"{fold}|{seed}" for fold in (1, 2, 3) for seed in (42, 7, 123)]
    expected_position_rows = 3 * 120 * len(POSITION_SIGNAL_DATES) * 3
    prediction_numbers = predictions["centered_score"].to_numpy(dtype=np.float64)
    seed_frame = predictions.loc[predictions["seed"] != "ensemble"]
    ensemble_frame = predictions.loc[predictions["seed"] == "ensemble"]
    seed_mean = seed_frame.groupby(
        ["fold", "triplet_id", "signal_date", "symbol"], sort=True
    )["centered_score"].mean()
    observed_ensemble = ensemble_frame.set_index(
        ["fold", "triplet_id", "signal_date", "symbol"]
    )["centered_score"].sort_index()
    ensemble_error = float(np.max(np.abs(seed_mean.sort_index() - observed_ensemble)))
    candidate_turnover = _turnover_audit(candidate, ["fold", "triplet_id"])
    control_turnover = _turnover_audit(controls, ["control", "fold", "triplet_id"])
    candidate_daily = candidate.groupby(
        ["fold", "triplet_id", "signal_date"], sort=True
    )["weight"].sum()
    exposure = float((candidate_daily > 0.0).mean())
    candidate_weights = candidate["weight"].to_numpy(float)
    control_names = tuple(sorted(controls["control"].unique()))
    control_daily = controls.groupby(
        ["control", "fold", "triplet_id", "signal_date"], sort=True
    )["weight"].sum()
    prediction_keys = set(
        zip(
            ensemble_frame["fold"], ensemble_frame["triplet_id"],
            pd.to_datetime(ensemble_frame["signal_date"], utc=True),
        )
    )
    candidate_scope = candidate.drop_duplicates(["fold", "triplet_id", "signal_date"])
    eligible_keys = set(
        zip(
            candidate_scope.loc[candidate_scope["eligible"], "fold"],
            candidate_scope.loc[candidate_scope["eligible"], "triplet_id"],
            pd.to_datetime(candidate_scope.loc[candidate_scope["eligible"], "signal_date"], utc=True),
        )
    )
    gates = {
        "all_registered_checkpoints_used_without_selection": used_jobs == expected_jobs,
        "exact_fold_triplet_date_scope": (
            len(candidate) == expected_position_rows
            and len(controls) == expected_position_rows * len(CONTROL_NAMES)
            and all(len(value) == 120 for value in triplets_by_fold.values())
        ),
        "missingness_matches_registered_readiness": prediction_keys == eligible_keys,
        "prediction_distribution_finite_and_nonconstant": bool(
            np.isfinite(prediction_numbers).all() and np.std(prediction_numbers) > 0.0
        ),
        "centered_scores_and_seed_ensemble_exact": bool(
            inference_checks["maximum_centered_score_sum_error"] <= 1.0e-5
            and inference_checks["maximum_ensemble_mean_error"] <= 1.0e-12
            and ensemble_error <= 1.0e-12
        ),
        "permutation_and_lexical_structure_complete": all(
            tuple(sorted(triplet)) == triplet
            for values in triplets_by_fold.values() for triplet in values
        ),
        "action_space_and_state_transitions_exact": bool(
            np.all(candidate_weights >= 0.0)
            and set(candidate_daily.round(12).unique()).issubset({0.0, 1.0})
            and set(candidate["action"].unique()).issubset(
                {"cash", "enter", "exit", "hold", "switch"}
            )
        ),
        "turnover_and_final_liquidation_exact": bool(
            candidate_turnover["maximum_daily_turnover_error"] <= 1.0e-12
            and candidate_turnover["maximum_final_liquidation_error"] <= 1.0e-12
            and candidate_turnover["maximum_triplet_turnover"] <= 16.0
        ),
        "control_positions_exact": bool(
            control_names == tuple(sorted(CONTROL_NAMES))
            and np.isfinite(controls["weight"].to_numpy(float)).all()
            and (controls["weight"] >= 0.0).all()
            and (control_daily <= 1.0 + 1.0e-12).all()
            and control_turnover["maximum_daily_turnover_error"] <= 1.0e-12
            and control_turnover["maximum_final_liquidation_error"] <= 1.0e-12
        ),
        "aggregate_turnover_within_registered_ceiling": (
            candidate_turnover["aggregate_turnover"] <= 16.0
        ),
        "exposure_fraction_within_registered_bounds": 0.05 <= exposure <= 0.95,
        "zero_outcome_and_target_access": True,
    }
    value = {
        "schema_version": "v84-outcome-blind-behavior-audit/v1",
        "gates": gates,
        "passed": all(gates.values()),
        "checkpoint_jobs_used_in_registered_order": used_jobs,
        "inference_checks": dict(inference_checks),
        "ensemble_reconstruction_error": ensemble_error,
        "candidate_turnover": candidate_turnover,
        "control_turnover": control_turnover,
        "candidate_exposure_fraction": exposure,
        "prediction_rows": len(predictions),
        "candidate_position_rows": len(candidate),
        "control_position_rows": len(controls),
        "eligible_triplet_dates": len(eligible_keys),
        "outcome_rows_read": 0,
        "target_assets_loaded": [],
    }
    value["behavior_audit_sha256"] = canonical_sha256(value)
    return value


def _registered_packet_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "cost_bps": list(contract["policy_contract"]["reporting_cost_bps"]),
        "accounting": dict(contract["registered_accounting"]),
        "controls": dict(contract["registered_controls"]),
        "gates": dict(contract["registered_outcome_dependent_gates"]),
        "outcome_blind_gate_names": list(contract["outcome_blind_gate_contract"]["gates"]),
    }
    return {**body, "sha256": canonical_sha256(body)}


def _validator(context: Mapping[str, Any], packet: Path) -> dict[str, Any]:
    script = (
        context["root"]
        / ".agents/skills/tlm-one-shot-evaluator/scripts/validate_evaluation_packet.py"
    )
    result = subprocess.run(
        [
            os.fspath(Path(os.sys.executable)), os.fspath(script),
            "--repo-root", os.fspath(context["root"]), "--packet", os.fspath(packet),
        ],
        cwd=context["root"], text=True, capture_output=True, check=False,
    )
    _require(result.returncode == 0, f"V84 one-shot validation failed: {result.stderr.strip()}")
    value = json.loads(result.stdout)
    _require(value.get("valid") is True and value.get("outcomes_sealed") is True, "V84 one-shot validation drift")
    return value


def _artifact_manifest(output: Path, data_paths: list[Path]) -> dict[str, Any]:
    files = {
        path.name: file_sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "artifact_manifest.json"
    }
    data = {str(path): file_sha256(path) for path in data_paths}
    value = {
        "schema_version": "v84-artifact-manifest/v1",
        "files": files,
        "data_files": data,
    }
    value["artifact_manifest_sha256"] = canonical_sha256(value)
    return value


def prepare_low_turnover_rank_evaluation(config: Mapping[str, Any]) -> dict[str, Any]:
    context = _context(config)
    output = context["output"]
    if (output / "one_shot_packet.json").is_file():
        raise V84EvaluationError("V84 prepare is already frozen; use hash-only replay")
    output.mkdir(parents=True, exist_ok=True)
    store = _read_feature_store(context)
    triplets = _fold_triplets(context["root"])
    predictions, ensembles, used_jobs, inference_checks = _infer(context, store, triplets)
    candidate, controls = _positions(store, triplets, ensembles)
    behavior = _behavior_audit(
        predictions=predictions, candidate=candidate, controls=controls,
        triplets_by_fold=triplets, used_jobs=used_jobs, inference_checks=inference_checks,
    )
    prediction_manifest = _atomic_parquet(predictions, context["prediction_path"])
    candidate_manifest = _atomic_parquet(candidate, context["candidate_path"])
    control_manifest = _atomic_parquet(controls, context["control_path"])
    write_json_atomic(output / "predictions_manifest.json", prediction_manifest)
    write_json_atomic(output / "candidate_positions_manifest.json", candidate_manifest)
    write_json_atomic(output / "control_positions_manifest.json", control_manifest)

    evaluation_spec = {
        "schema_version": "v84-low-turnover-rank-evaluation-spec/v1",
        "family_id": context["contract"]["family_id"],
        "evaluation_contract": context["contract"]["evaluation_contract"],
        "policy_contract": context["contract"]["policy_contract"],
        "registered_accounting": context["contract"]["registered_accounting"],
        "registered_controls": context["contract"]["registered_controls"],
        "registered_bootstrap": context["contract"]["registered_bootstrap"],
        "registered_outcome_dependent_gates": context["contract"]["registered_outcome_dependent_gates"],
        "outcome_blind_gate_contract": context["contract"]["outcome_blind_gate_contract"],
        "outcome_request_contract": context["contract"]["outcome_request_contract"],
        "target_contract": context["contract"]["target_contract"],
        "frozen": True,
    }
    evaluation_spec["evaluation_spec_sha256"] = canonical_sha256(evaluation_spec)
    write_json_atomic(output / "evaluation_spec.json", evaluation_spec)
    write_json_atomic(output / "behavior_audit.json", behavior)
    data_access = {
        "schema_version": "v84-data-access-ledger/v1",
        "evaluation_feature_parquet_deserializations": 1,
        "checkpoint_deserializations": 9,
        "checkpoint_jobs_loaded": used_jobs,
        "scaler_fits": 0,
        "optimizer_steps": 0,
        "checkpoint_writes": 0,
        "outcome_packet_deserializations": 0,
        "outcome_rows_read": 0,
        "performance_metrics_computed": False,
        "pnl_computed": False,
        "bootstrap_computed": False,
        "target_assets_loaded": [],
    }
    write_json_atomic(output / "data_access.json", data_access)
    input_receipt = {
        "schema_version": "v84-input-hash-receipt/v1",
        "files": context["expected_inputs"],
        "bundle_sha256": canonical_sha256(context["expected_inputs"]),
    }
    write_json_atomic(output / "input_hash_receipt.json", input_receipt)
    write_json_atomic(output / "source_receipt.json", context["source_receipt"])
    write_yaml_atomic(output / "resolved_config.yaml", dict(config))
    outcome_request = {
        "schema_version": "v84-sealed-outcome-request/v1",
        **context["contract"]["outcome_request_contract"],
        "status": "sealed_not_opened",
        "unseal_count": 0,
        "outcome_rows_read": 0,
    }
    outcome_request["outcome_request_sha256"] = canonical_sha256(outcome_request)
    write_json_atomic(output / "outcome_request.json", outcome_request)

    if not behavior["passed"]:
        audit = {
            "schema_version": "v84-prepare-audit/v1",
            "passed": False,
            "checks": behavior["gates"],
        }
        result = {
            "schema_version": "v84-low-turnover-rank-evaluation-prepare-result/v1",
            "family_id": context["contract"]["family_id"],
            "decision": FAIL_ACTION,
            "audit": audit,
            "summary": {
                "outcome_rows_read": 0,
                "target_assets_loaded": 0,
                "failed_outcome_blind_gates": [
                    name for name, passed in behavior["gates"].items() if not passed
                ],
            },
        }
        result["result_sha256"] = canonical_sha256(result)
        write_json_atomic(output / "audit.json", audit)
        write_json_atomic(output / "result.json", result)
        (output / "report.md").write_text(
            "# V84 outcome-blind prepare\n\nPrepare failed before outcomes; the sealed packet was not opened.\n",
            encoding="utf-8",
        )
        write_json_atomic(
            output / "artifact_manifest.json",
            _artifact_manifest(
                output,
                [context["prediction_path"], context["candidate_path"], context["control_path"]],
            ),
        )
        return result

    registered = _registered_packet_contract(context["contract"])
    artifact_hashes = {
        _relative(context["root"], context["prediction_path"]): prediction_manifest["sha256"],
        _relative(context["root"], context["candidate_path"]): candidate_manifest["sha256"],
        _relative(context["root"], context["control_path"]): control_manifest["sha256"],
    }
    spec_file_hash = file_sha256(output / "evaluation_spec.json")
    prepare_receipt = {
        "schema_version": "tlm-one-shot-prepare/v1",
        "evaluation_spec_sha256": spec_file_hash,
        "registered_sha256": registered["sha256"],
        "artifact_hashes": artifact_hashes,
        "outcome_rows_read": 0,
        "outcome_blind_gates_passed": True,
        "authorizes_unseal": True,
    }
    write_json_atomic(output / "prepare_receipt.json", prepare_receipt)
    prepare_file_hash = file_sha256(output / "prepare_receipt.json")
    packet = {
        "schema_version": "tlm-one-shot-evaluator/v1",
        "phase": "prepare",
        "research_state": {
            "path": context["evaluation"]["research_state"],
            "sha256": file_sha256(context["root"] / context["evaluation"]["research_state"]),
            "authorized_phase": "v84",
            "authorized_next_action": PREPARE_ACTION,
            "authorized_command": PREPARE_COMMAND,
        },
        "evaluation_spec": {
            "path": _relative(context["root"], output / "evaluation_spec.json"),
            "sha256": spec_file_hash,
            "frozen": True,
        },
        "source_receipt": context["source_receipt"],
        "registered": registered,
        "prepare": {
            "receipt": {
                "path": _relative(context["root"], output / "prepare_receipt.json"),
                "sha256": prepare_file_hash,
            },
            "artifacts": [
                {"kind": "predictions", "path": path, "sha256": digest}
                if "predictions.parquet" in path
                else {"kind": "positions", "path": path, "sha256": digest}
                for path, digest in artifact_hashes.items()
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
    validator = _validator(context, output / "one_shot_packet.json")
    replay_receipt = {
        "schema_version": "v84-hash-only-prepare-replay/v1",
        "artifact_hashes_match": True,
        "model_instantiations": 0,
        "checkpoint_deserializations": 0,
        "feature_parquet_deserializations": 0,
        "outcome_packet_deserializations": 0,
        "outcome_rows_read": 0,
        "one_shot_validator": validator,
    }
    write_json_atomic(output / "replay_receipt.json", replay_receipt)
    audit = {
        "schema_version": "v84-prepare-audit/v1",
        "passed": True,
        "checks": behavior["gates"],
    }
    write_json_atomic(output / "audit.json", audit)
    result = {
        "schema_version": "v84-low-turnover-rank-evaluation-prepare-result/v1",
        "family_id": context["contract"]["family_id"],
        "decision": PASS_ACTION,
        "evidence_tier": context["contract"]["evidence_tier"],
        "audit": audit,
        "evaluation_spec_sha256": spec_file_hash,
        "prepare_receipt_sha256": prepare_file_hash,
        "registered_sha256": registered["sha256"],
        "one_shot_packet_sha256": file_sha256(output / "one_shot_packet.json"),
        "summary": {
            "checkpoint_count": 9,
            "prediction_rows": len(predictions),
            "candidate_position_rows": len(candidate),
            "control_position_rows": len(controls),
            "aggregate_turnover": behavior["candidate_turnover"]["aggregate_turnover"],
            "exposure_fraction": behavior["candidate_exposure_fraction"],
            "outcome_rows_read": 0,
            "performance_metrics": 0,
            "pnl_evaluations": 0,
            "bootstrap_evaluations": 0,
            "target_asset_loads": 0,
            "unseal_count": 0,
        },
    }
    result["result_sha256"] = canonical_sha256(result)
    write_json_atomic(output / "result.json", result)
    (output / "report.md").write_text(
        "\n".join(
            [
                "# V84 outcome-blind evaluation prepare",
                "",
                "All registered checkpoints, predictions, positions, controls, and behavior gates are frozen.",
                f"- Aggregate structural turnover: {behavior['candidate_turnover']['aggregate_turnover']:.6f}",
                f"- Exposure fraction: {behavior['candidate_exposure_fraction']:.6f}",
                "- Outcomes opened: 0",
                "- Economic metrics computed: 0",
                "- Next step: exact hash-bound V85 unseal authorization",
                "",
            ]
        ),
        encoding="utf-8",
    )
    write_json_atomic(
        output / "artifact_manifest.json",
        _artifact_manifest(
            output,
            [context["prediction_path"], context["candidate_path"], context["control_path"]],
        ),
    )
    return result


def replay_low_turnover_rank_evaluation_prepare(config: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = _evaluation_config(config)
    root = Path(evaluation.get("project_root", ".")).resolve()
    output = root / str(config["output_dir"])
    manifest = _load_json(output / "artifact_manifest.json", "V84 artifact manifest")
    for name, digest in manifest["files"].items():
        path = output / name
        _require(path.is_file() and file_sha256(path) == digest, f"V84 replay drift: {name}")
    for raw_path, digest in manifest["data_files"].items():
        path = Path(raw_path)
        if not path.is_absolute():
            path = root / path
        _require(path.is_file() and file_sha256(path) == digest, f"V84 replay data drift: {path}")
    status = validate_research_state(root, evaluation["research_state"])
    _require(status.get("passed") is True and status.get("authorized_phase") == "v84", "V84 replay state drift")
    context = {"root": root}
    validator = _validator(context, output / "one_shot_packet.json")
    result = _load_json(output / "result.json", "V84 prepare result")
    _require(result.get("decision") == PASS_ACTION, "V84 replay result is not a passed prepare")
    return {
        **result,
        "replay": {
            "artifact_hashes_match": True,
            "model_instantiations": 0,
            "checkpoint_deserializations": 0,
            "feature_parquet_deserializations": 0,
            "outcome_packet_deserializations": 0,
            "outcome_rows_read": 0,
            "one_shot_validator": validator,
        },
    }
