"""Exactly-once V85 economic evaluation for the frozen V84 rank policy."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd
import yaml

from .core.artifacts import canonical_sha256, file_sha256, write_json_atomic
from .monte_carlo import circular_block_indices
from .research_workflow import validate_research_state


ACTION = (
    "execute_v85_exactly_one_registered_non_target_outcome_unseal_and_complete_"
    "evaluation"
)
COMMAND = (
    "PYTHONPATH=src python3 -m tlm low-turnover-rank-evaluation-unseal "
    "--config configs/v85_low_turnover_rank_evaluation.yaml"
)
REPLAY_COMMAND = (
    "PYTHONPATH=src python3 -m tlm low-turnover-rank-evaluation-replay "
    "--config configs/v85_low_turnover_rank_evaluation.yaml"
)
PASS_ACTION = "retain_family_and_authorize_separate_target_transfer_specification_only"
FAIL_ACTION = "retire_final_family_without_target_evaluation_or_retuning"
TARGET_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
CONTROL_NAMES = (
    "cash",
    "decision_21d_equal_weight_three_assets",
    "decision_21d_top_trailing_21d_momentum_with_same_63d_market_gate",
)
AUTHORIZATION_FILE = "unseal_authorization_receipt.json"
OUTCOME_FILE = "outcome_packet.parquet"
OUTCOME_RECEIPT_FILE = "outcome_receipt.json"
COMPLETE_PACKET_FILE = "one_shot_complete_packet.json"
REPLAY_PACKET_FILE = "one_shot_replay_packet.json"
CORE_FILES = (
    "economic_metrics.json",
    "daily_returns.parquet",
    "bootstrap.json",
    "gate_matrix.json",
    "evaluation_result.json",
    "evaluation_audit.json",
    "evaluation_report.md",
    "result.json",
    "audit.json",
)
CANDIDATE_COLUMNS = (
    "fold",
    "triplet_id",
    "signal_date",
    "interval_start_date",
    "interval_end_date",
    "symbol",
    "eligible",
    "decision",
    "market_gate",
    "weight",
    "selected_symbol",
    "action",
    "transition_turnover",
    "final_liquidation_turnover",
)


class V85EvaluationError(RuntimeError):
    """Fail-closed V85 evaluation error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V85EvaluationError(message)


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V85EvaluationError(f"cannot load {name}: {path}") from exc
    _require(isinstance(value, dict), f"{name} must be a JSON object")
    return value


def _load_yaml(path: Path, name: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise V85EvaluationError(f"cannot load {name}: {path}") from exc
    _require(isinstance(value, dict), f"{name} must be a YAML object")
    return value


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, text=True, capture_output=True, check=False
    )
    _require(result.returncode == 0, f"git {' '.join(arguments)} failed")
    return result.stdout.strip()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(
            temporary, index=False, engine="pyarrow", compression="zstd"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    os.replace(temporary, path)


@contextmanager
def _process_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise V85EvaluationError("another V85 evaluation process owns the lock") from exc
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
        yield
    finally:
        path.unlink(missing_ok=True)


def _verify_file(root: Path, relative: str, digest: str, name: str) -> Path:
    path = (root / relative).resolve()
    _require(root == path or root in path.parents, f"{name} path escapes repository")
    _require(path.is_file(), f"missing {name}: {relative}")
    _require(file_sha256(path) == digest, f"{name} hash drift")
    return path


def _source_receipt(root: Path, files: list[str]) -> dict[str, Any]:
    _require(
        _git(root, "status", "--porcelain", "--untracked-files=all") == "",
        "V85 unseal requires a clean committed Git tree",
    )
    hashes: dict[str, str] = {}
    for relative in files:
        path = (root / relative).resolve()
        _require(root in path.parents and path.is_file(), f"missing source: {relative}")
        hashes[relative] = file_sha256(path)
    return {
        "schema_version": "v85-source-receipt/v1",
        "git_clean": True,
        "git_head": _git(root, "rev-parse", "HEAD"),
        "files": hashes,
        "bundle_sha256": canonical_sha256(hashes),
    }


def _context(config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(config.get("project_root", "."))).resolve()
    state_path = root / str(config["research_state"])
    status = validate_research_state(root, state_path.relative_to(root))
    _require(status.get("passed") is True, "V85 research state failed")
    _require(status.get("authorized_phase") == "v85", "V85 phase drift")
    _require(status.get("authorized_next_action") == ACTION, "V85 action drift")
    _require(status.get("authorized_command") == COMMAND, "V85 command drift")

    stage_path = root / str(config["phase_contract"])
    stage = _load_yaml(stage_path, "V85 phase contract")
    _require(
        stage.get("stage_revision") == "v085_hash_bound_low_turnover_rank_unseal_r1",
        "V85 stage revision drift",
    )
    _require(stage.get("authorized_command") == COMMAND, "V85 command contract drift")

    prepare_dir = root / str(config["prepare_dir"])
    output = root / str(config["output_dir"])
    packet_path = prepare_dir / "one_shot_packet.json"
    prepare_packet = _load_json(packet_path, "V84 one-shot prepare packet")
    evaluation_spec = _load_json(
        prepare_dir / "evaluation_spec.json", "V84 evaluation spec"
    )
    authorization_path = root / stage["explicit_user_authorization"]["path"]
    authorization = _load_json(authorization_path, "V85 explicit authorization")
    registered_authorization = authorization.pop("authorization_sha256", None)
    _require(
        registered_authorization
        == stage["explicit_user_authorization"]["canonical_sha256"]
        and canonical_sha256(authorization) == registered_authorization,
        "V85 explicit authorization canonical hash drift",
    )
    _require(authorization.get("authorized_action") == ACTION, "authorization action drift")
    _require(authorization.get("maximum_unseal_count") == 1, "authorization count drift")
    _require(authorization.get("target_assets_status") == "sealed", "target authorization drift")
    _require(
        file_sha256(packet_path) == authorization["one_shot_prepare_packet_sha256"]
        and prepare_packet["evaluation_spec"]["sha256"]
        == authorization["evaluation_spec_sha256"]
        and prepare_packet["prepare"]["receipt"]["sha256"]
        == authorization["prepare_receipt_sha256"]
        and prepare_packet["registered"]["sha256"]
        == authorization["registered_sha256"],
        "authorization does not bind the live V84 packet",
    )
    _require(
        prepare_packet.get("phase") == "prepare"
        and prepare_packet["prepare"]["authorizes_unseal"] is True
        and prepare_packet["prepare"]["outcome_rows_read"] == 0
        and all(prepare_packet["prepare"]["outcome_blind_gates"].values()),
        "V84 prepare packet does not authorize unseal",
    )
    _require(
        evaluation_spec.get("frozen") is True
        and evaluation_spec["registered_outcome_dependent_gates"]
        == prepare_packet["registered"]["gates"],
        "V85 frozen gate contract drift",
    )

    for key in ("prediction_path", "candidate_position_path", "control_position_path"):
        relative = str(config[key])
        expected = stage["input_contract"]["expected_file_sha256_by_path"][relative]
        _verify_file(root, relative, expected, key)
    source_relative = stage["outcome_access_contract"]["source_packet"]
    source_packet = _verify_file(
        root,
        source_relative,
        stage["outcome_access_contract"]["source_packet_sha256"],
        "sealed V82 outcome source",
    )
    source = _source_receipt(root, list(stage["source_receipt_files"]))
    _require(dict(config["bootstrap"]) == stage["evaluation_contract"]["bootstrap"], "bootstrap drift")
    return {
        "root": root,
        "state_path": state_path,
        "stage": stage,
        "stage_path": stage_path,
        "prepare_dir": prepare_dir,
        "output": output,
        "prepare_packet": prepare_packet,
        "evaluation_spec": evaluation_spec,
        "authorization": authorization,
        "authorization_sha256": registered_authorization,
        "source_packet": source_packet,
        "source_receipt": source,
        "bootstrap": dict(config["bootstrap"]),
        "prediction_path": root / str(config["prediction_path"]),
        "candidate_path": root / str(config["candidate_position_path"]),
        "control_path": root / str(config["control_position_path"]),
    }


def _prepare_positions(context: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    candidate = pd.read_parquet(context["candidate_path"])
    controls = pd.read_parquet(context["control_path"])
    _require(list(candidate.columns) == list(CANDIDATE_COLUMNS), "candidate schema drift")
    _require(
        list(controls.columns) == [*CANDIDATE_COLUMNS, "control"],
        "control schema drift",
    )
    _require(len(candidate) == 193320 and len(controls) == 579960, "position row drift")
    for name, frame in (("candidate", candidate), ("controls", controls)):
        for column in ("signal_date", "interval_start_date", "interval_end_date"):
            frame[column] = pd.to_datetime(frame[column], utc=True)
        frame["fold"] = frame["fold"].astype(int)
        frame["symbol"] = frame["symbol"].astype(str)
        _require(set(frame["symbol"]).isdisjoint(TARGET_SYMBOLS), f"target reached {name}")
        _require(np.isfinite(frame["weight"].to_numpy(dtype=np.float64)).all(), f"nonfinite {name}")
        _require(set(frame["fold"]) == {1, 2, 3}, f"{name} fold drift")
        _require(frame["triplet_id"].nunique() == 360, f"{name} triplet drift")
        _require(frame["signal_date"].nunique() == 179, f"{name} date drift")
    _require(tuple(sorted(controls["control"].unique())) == tuple(sorted(CONTROL_NAMES)), "control identity drift")
    return {"candidate": candidate, "controls": controls}


def _authorization_receipt(context: Mapping[str, Any]) -> dict[str, Any]:
    prepare = context["prepare_packet"]
    return {
        "schema_version": "tlm-one-shot-unseal-authorization/v1",
        "phase": "v85",
        "stage_revision": "v085_hash_bound_low_turnover_rank_unseal_r1",
        "family_id": context["stage"]["family_id"],
        "explicit_user_authorization": True,
        "exact_registered_unseal": True,
        "unseal_count": 1,
        "authorized_command": COMMAND,
        "authorization_payload": context["authorization"],
        "authorization_payload_sha256": context["authorization_sha256"],
        "evaluation_spec_sha256": prepare["evaluation_spec"]["sha256"],
        "prepare_receipt_sha256": prepare["prepare"]["receipt"]["sha256"],
        "registered_sha256": prepare["registered"]["sha256"],
        "source_git_head": context["source_receipt"]["git_head"],
        "source_outcome_reread_allowed": False,
        "target_assets_loaded": [],
    }


def _read_source_once(
    context: Mapping[str, Any], symbols: set[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    access = context["stage"]["outcome_access_contract"]
    columns = list(access["allowed_columns"])
    frame = pd.read_parquet(context["source_packet"], columns=columns)
    _require(list(frame.columns) == columns, "V85 outcome projection drift")
    frame["interval_start_date"] = pd.to_datetime(frame["interval_start_date"], utc=True)
    frame["interval_end_date"] = pd.to_datetime(frame["interval_end_date"], utc=True)
    frame["symbol"] = frame["symbol"].astype(str)
    frame["outcome_complete"] = frame["outcome_complete"].astype(bool)
    _require(len(frame) == int(access["expected_rows"]), "V85 outcome row drift")
    _require(
        not frame.duplicated(["interval_start_date", "interval_end_date", "symbol"]).any(),
        "duplicate V85 outcome key",
    )
    _require(set(frame["symbol"]) == symbols, "V85 outcome symbol scope drift")
    _require(set(frame["symbol"]).isdisjoint(TARGET_SYMBOLS), "target reached V85 outcomes")
    _require(
        frame["interval_start_date"].min() == pd.Timestamp(access["interval_start"], tz="UTC")
        and frame["interval_start_date"].max() == pd.Timestamp(access["interval_end"], tz="UTC"),
        "V85 outcome date scope drift",
    )
    _require(
        (frame["interval_end_date"] == frame["interval_start_date"] + pd.Timedelta(days=1)).all(),
        "V85 outcome interval drift",
    )
    packet = frame.sort_values(
        ["interval_start_date", "interval_end_date", "symbol"]
    ).reset_index(drop=True)
    return packet, {
        "source_packet": access["source_packet"],
        "source_packet_sha256": access["source_packet_sha256"],
        "requested_columns": columns,
        "exact_rows": len(packet),
        "complete_rows": int(packet["outcome_complete"].sum()),
        "incomplete_rows": int((~packet["outcome_complete"]).sum()),
        "source_packet_deserializations": 1,
        "source_outcome_rows_read": len(packet),
        "target_assets_loaded": [],
    }


def _validate_cached_outcome(context: Mapping[str, Any]) -> pd.DataFrame:
    output = context["output"]
    packet_path = output / OUTCOME_FILE
    receipt_path = output / OUTCOME_RECEIPT_FILE
    _require(packet_path.is_file() and receipt_path.is_file(), "V85 outcome packet incomplete")
    receipt = _load_json(receipt_path, "V85 outcome receipt")
    _require(receipt.get("schema_version") == "tlm-one-shot-outcome/v1", "outcome receipt schema drift")
    _require(receipt.get("unseal_count") == 1, "V85 unseal count drift")
    _require(receipt.get("outcome_packet_sha256") == file_sha256(packet_path), "outcome packet hash drift")
    _require(
        receipt.get("authorization_receipt_sha256")
        == file_sha256(output / AUTHORIZATION_FILE),
        "outcome authorization binding drift",
    )
    _require(receipt.get("written_atomically") is True and receipt.get("immutable") is True, "outcome immutability drift")
    frame = pd.read_parquet(packet_path)
    frame["interval_start_date"] = pd.to_datetime(frame["interval_start_date"], utc=True)
    frame["interval_end_date"] = pd.to_datetime(frame["interval_end_date"], utc=True)
    frame["symbol"] = frame["symbol"].astype(str)
    frame["outcome_complete"] = frame["outcome_complete"].astype(bool)
    _require(len(frame) == 5370, "cached V85 outcome row drift")
    _require(set(frame["symbol"]).isdisjoint(TARGET_SYMBOLS), "target in cached outcomes")
    return frame


def _portfolio_fold_daily(
    portfolio: str, positions: pd.DataFrame, outcomes: pd.DataFrame
) -> pd.DataFrame:
    keys = ["interval_start_date", "interval_end_date", "symbol"]
    merged = positions.merge(outcomes, on=keys, how="left", validate="many_to_one")
    complete = merged["outcome_complete"].fillna(False).to_numpy(dtype=bool)
    log_returns = merged["open_to_next_open_log_return"].to_numpy(dtype=np.float64)
    available = complete & np.isfinite(log_returns)
    merged["simple_return"] = np.where(available, np.expm1(np.where(available, log_returns, 0.0)), 0.0)
    merged["gross_contribution"] = merged["weight"].to_numpy(dtype=np.float64) * merged["simple_return"]
    merged["unavailable_active"] = ((merged["weight"] > 0.0) & ~available).astype(int)
    group_columns = ["fold", "triplet_id", "signal_date", "interval_start_date"]
    for column in ("transition_turnover", "final_liquidation_turnover"):
        _require(
            int(merged.groupby(group_columns, sort=False)[column].nunique().max()) == 1,
            f"{portfolio} repeated turnover drift",
        )
    subaccounts = merged.groupby(group_columns, sort=True).agg(
        gross_return=("gross_contribution", "sum"),
        transition_turnover=("transition_turnover", "first"),
        final_liquidation_turnover=("final_liquidation_turnover", "first"),
        exposure=("weight", "sum"),
        unavailable_active_assets=("unavailable_active", "sum"),
    ).reset_index()
    subaccounts["turnover"] = (
        subaccounts["transition_turnover"] + subaccounts["final_liquidation_turnover"]
    )
    rows: list[pd.DataFrame] = []
    for cost_bps in (10, 20, 30):
        current = subaccounts.copy()
        current["cost_bps"] = cost_bps
        current["net_return"] = current["gross_return"] - current["turnover"] * cost_bps / 10000.0
        rows.append(current)
    all_costs = pd.concat(rows, ignore_index=True)
    fold_daily = all_costs.groupby(
        ["interval_start_date", "fold", "cost_bps"], sort=True
    ).agg(
        gross_return=("gross_return", "mean"),
        net_return=("net_return", "mean"),
        turnover=("turnover", "mean"),
        exposure=("exposure", "mean"),
        unavailable_active_assets=("unavailable_active_assets", "sum"),
        triplet_count=("triplet_id", "nunique"),
    ).reset_index().rename(columns={"interval_start_date": "date"})
    _require(len(fold_daily) == 179 * 3 * 3, f"{portfolio} fold daily row drift")
    _require((fold_daily["triplet_count"] == 120).all(), f"{portfolio} triplet aggregation drift")
    fold_daily.insert(0, "portfolio", portfolio)
    return fold_daily


def _daily_returns(
    positions: Mapping[str, pd.DataFrame], outcomes: pd.DataFrame
) -> pd.DataFrame:
    frames = [_portfolio_fold_daily("candidate", positions["candidate"], outcomes)]
    for control in CONTROL_NAMES:
        frame = positions["controls"].loc[positions["controls"]["control"] == control].drop(columns="control")
        frames.append(_portfolio_fold_daily(control, frame, outcomes))
    result = pd.concat(frames, ignore_index=True).sort_values(
        ["portfolio", "date", "fold", "cost_bps"]
    ).reset_index(drop=True)
    _require(len(result) == 4 * 179 * 3 * 3, "V85 daily return row drift")
    return result


def _series_metrics(values: np.ndarray) -> dict[str, float]:
    returns = np.asarray(values, dtype=np.float64)
    _require(len(returns) > 1 and np.isfinite(returns).all(), "invalid metric series")
    equity = np.cumprod(1.0 + returns)
    total_return = float(equity[-1] - 1.0)
    standard_deviation = float(np.std(returns, ddof=1))
    sharpe = 0.0 if standard_deviation == 0.0 else float(np.sqrt(365.0) * np.mean(returns) / standard_deviation)
    peak = np.maximum.accumulate(np.maximum(equity, 1.0))
    maximum_drawdown = float(np.min(equity / peak - 1.0))
    years = len(returns) / 365.0
    cagr = -1.0 if equity[-1] <= 0.0 else float(equity[-1] ** (1.0 / years) - 1.0)
    return {
        "observations": int(len(returns)),
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "annualized_volatility": float(np.sqrt(365.0) * standard_deviation),
        "maximum_drawdown": maximum_drawdown,
        "mean_daily_return": float(np.mean(returns)),
    }


def _economic_metrics(daily: pd.DataFrame) -> dict[str, Any]:
    portfolios: dict[str, Any] = {}
    for portfolio, portfolio_frame in daily.groupby("portfolio", sort=True):
        folds: dict[str, Any] = {}
        for (fold, cost), frame in portfolio_frame.groupby(["fold", "cost_bps"], sort=True):
            current = frame.sort_values("date")
            cell = _series_metrics(current["net_return"].to_numpy(dtype=np.float64))
            cell["gross_total_return"] = float(np.prod(1.0 + current["gross_return"]) - 1.0)
            cell["turnover"] = float(current["turnover"].sum())
            cell["exposure_fraction"] = float((current["exposure"] > 0.0).mean())
            folds.setdefault(str(int(fold)), {})[str(int(cost))] = cell
        aggregate: dict[str, Any] = {}
        for cost, frame in portfolio_frame.groupby("cost_bps", sort=True):
            current = frame.groupby("date", sort=True).agg(
                net_return=("net_return", "mean"),
                gross_return=("gross_return", "mean"),
                turnover=("turnover", "mean"),
                exposure=("exposure", "mean"),
            )
            cell = _series_metrics(current["net_return"].to_numpy(dtype=np.float64))
            cell["gross_total_return"] = float(np.prod(1.0 + current["gross_return"]) - 1.0)
            cell["turnover"] = float(current["turnover"].sum())
            cell["exposure_fraction"] = float((current["exposure"] > 0.0).mean())
            aggregate[str(int(cost))] = cell
        portfolios[str(portfolio)] = {"folds": folds, "aggregate": aggregate}
    return {
        "schema_version": "v85-economic-metrics/v1",
        "evidence_tier": "retrospective_non_target_first_use_2026_not_prospective_confirmation",
        "portfolios": portfolios,
    }


def _distribution(values: np.ndarray) -> dict[str, float]:
    quantiles = np.quantile(values, [0.01, 0.10, 0.50, 0.90, 0.99])
    return {
        "mean": float(np.mean(values)),
        "p01": float(quantiles[0]),
        "p10": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p90": float(quantiles[3]),
        "p99": float(quantiles[4]),
    }


def _bootstrap(daily: pd.DataFrame, contract: Mapping[str, Any]) -> dict[str, Any]:
    cost = int(contract["economic_cost_bps"])
    names = ("candidate", *CONTROL_NAMES)
    series: dict[str, np.ndarray] = {}
    dates: pd.DatetimeIndex | None = None
    for name in names:
        frame = daily.loc[(daily["portfolio"] == name) & (daily["cost_bps"] == cost)]
        current = frame.groupby("date", sort=True)["net_return"].mean()
        _require(len(current) == 179, f"V85 {name} bootstrap length drift")
        if dates is None:
            dates = pd.DatetimeIndex(current.index)
        else:
            _require(pd.DatetimeIndex(current.index).equals(dates), "bootstrap date alignment drift")
        series[name] = current.to_numpy(dtype=np.float64)
    cells: list[dict[str, Any]] = []
    paths = int(contract["paths"])
    for raw_block in contract["block_lengths_days"]:
        block = int(raw_block)
        seed = int(contract["base_seed"]) + block
        rng = np.random.default_rng(seed)
        indexes = circular_block_indices(179, block, paths, rng)
        path_returns = {
            name: np.prod(1.0 + values[indexes], axis=1) - 1.0
            for name, values in series.items()
        }
        best_control = np.maximum.reduce([path_returns[name] for name in CONTROL_NAMES])
        comparison = path_returns["candidate"] - best_control
        cells.append({
            "block_length_days": block,
            "paths": paths,
            "seed": seed,
            "candidate_total_return": _distribution(path_returns["candidate"]),
            "candidate_minus_best_control": _distribution(comparison),
            "controls": {name: _distribution(path_returns[name]) for name in CONTROL_NAMES},
        })
    return {
        "schema_version": "v85-bootstrap/v1",
        "method": contract["method"],
        "synchronized": True,
        "confidence_level": float(contract["confidence_level"]),
        "economic_cost_bps": cost,
        "cells": cells,
        "cell_count": len(cells),
    }


def _gate_matrix(
    metrics: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    behavior: Mapping[str, Any],
) -> dict[str, Any]:
    candidate = metrics["portfolios"]["candidate"]
    rows: list[dict[str, Any]] = []

    def add(category: str, scope: str, observed: float, operator: str, threshold: float) -> None:
        if operator == "strictly_greater_than":
            passed = observed > threshold
        elif operator == "greater_than_or_equal":
            passed = observed >= threshold
        elif operator == "less_than_or_equal":
            passed = observed <= threshold
        else:
            raise V85EvaluationError(f"unknown gate operator: {operator}")
        rows.append({
            "category": category,
            "scope": scope,
            "observed": float(observed),
            "operator": operator,
            "threshold": float(threshold),
            "passed": bool(passed),
        })

    for cost in (10, 20, 30):
        add("aggregate_net_total_return_positive", f"aggregate_{cost}bps", candidate["aggregate"][str(cost)]["total_return"], "strictly_greater_than", 0.0)
    for fold in (1, 2, 3):
        add("every_fold_net_total_return_positive", f"fold_{fold}_10bps", candidate["folds"][str(fold)]["10"]["total_return"], "strictly_greater_than", 0.0)
    candidate_10 = candidate["aggregate"]["10"]["total_return"]
    for control in CONTROL_NAMES:
        control_return = metrics["portfolios"][control]["aggregate"]["10"]["total_return"]
        add("aggregate_net_total_return_above_every_control", f"candidate_minus_{control}_10bps", candidate_10 - control_return, "strictly_greater_than", 0.0)
    add("aggregate_sharpe_positive", "aggregate_10bps", candidate["aggregate"]["10"]["sharpe"], "strictly_greater_than", 0.0)
    add("aggregate_maximum_drawdown", "aggregate_10bps", candidate["aggregate"]["10"]["maximum_drawdown"], "greater_than_or_equal", -0.20)
    add("aggregate_turnover", "aggregate", float(behavior["candidate_turnover"]["aggregate_turnover"]), "less_than_or_equal", 16.0)
    exposure = float(behavior["candidate_exposure_fraction"])
    rows.append({
        "category": "exposure_fraction_between",
        "scope": "aggregate",
        "observed": exposure,
        "operator": "between_inclusive",
        "threshold": [0.05, 0.95],
        "passed": bool(0.05 <= exposure <= 0.95),
    })
    for cell in bootstrap["cells"]:
        block = int(cell["block_length_days"])
        add("bootstrap_candidate_total_return_p10_positive", f"block_{block}", cell["candidate_total_return"]["p10"], "strictly_greater_than", 0.0)
        add("bootstrap_candidate_minus_best_control_p10_positive", f"block_{block}", cell["candidate_minus_best_control"]["p10"], "strictly_greater_than", 0.0)
    _require(len(rows) == 19, "V85 mandatory gate cell count drift")
    categories = sorted({row["category"] for row in rows})
    category_pass = {
        category: all(row["passed"] for row in rows if row["category"] == category)
        for category in categories
    }
    return {
        "schema_version": "v85-gate-matrix/v1",
        "gates": rows,
        "category_pass": category_pass,
        "mandatory_category_count": len(category_pass),
        "mandatory_gate_count": len(rows),
        "passed_gate_count": sum(row["passed"] for row in rows),
        "failed_gate_count": sum(not row["passed"] for row in rows),
        "all_passed": all(row["passed"] for row in rows),
        "aggregate_rescue_used": False,
        "missing_cell_pass_used": False,
    }


def _compute_core(
    context: Mapping[str, Any], outcomes: pd.DataFrame, positions: Mapping[str, pd.DataFrame]
) -> dict[str, Any]:
    daily = _daily_returns(positions, outcomes)
    metrics = _economic_metrics(daily)
    bootstrap = _bootstrap(daily, context["bootstrap"])
    behavior = _load_json(context["prepare_dir"] / "behavior_audit.json", "V84 behavior audit")
    gates = _gate_matrix(metrics, bootstrap, behavior)
    decision = PASS_ACTION if gates["all_passed"] else FAIL_ACTION
    one_shot = "pass" if gates["all_passed"] else "retire"
    candidate = metrics["portfolios"]["candidate"]
    controls = {
        name: metrics["portfolios"][name]["aggregate"] for name in CONTROL_NAMES
    }
    result = {
        "schema_version": "v85-evaluation-result/v1",
        "family_id": context["stage"]["family_id"],
        "decision": decision,
        "one_shot_decision": one_shot,
        "evidence_tier": "retrospective_non_target_first_use_2026_not_prospective_confirmation",
        "mandatory_category_count": gates["mandatory_category_count"],
        "mandatory_gate_count": gates["mandatory_gate_count"],
        "passed_gate_count": gates["passed_gate_count"],
        "failed_gate_count": gates["failed_gate_count"],
        "candidate_aggregate": candidate["aggregate"],
        "candidate_folds": candidate["folds"],
        "control_aggregate": controls,
        "unseal_count": 1,
        "source_outcome_rows_read": 5370,
        "source_packet_deserializations": 1,
        "retuning_performed": False,
        "prediction_or_position_regeneration": False,
        "target_assets_loaded": [],
        "target_predictions": 0,
        "target_pnl_evaluations": 0,
        "deployable": False,
    }
    audit = {
        "schema_version": "v85-evaluation-audit/v1",
        "checks": {
            "exactly_one_hash_bound_source_unseal": True,
            "immutable_v84_predictions_positions_and_controls_reused": True,
            "all_registered_cost_accounting_control_bootstrap_and_gate_cells_preserved": True,
            "no_retuning_selection_regeneration_or_target_access": True,
            "aggregate_rescue_not_used": True,
            "missing_cell_pass_not_used": True,
            "target_assets_remain_sealed": True,
            "retrospective_non_target_evidence_label_preserved": True,
        },
        "passed": True,
        "scientific_gates_all_passed": gates["all_passed"],
    }
    candidate_10 = candidate["aggregate"]["10"]
    report = "\n".join([
        "# TLM V85 Low-Turnover Rank Economic Evaluation",
        "",
        f"**Decision:** `{decision}`",
        "",
        "Retrospective non-target 2026 evidence; not target transfer, prospective confirmation, or deployable evidence.",
        "",
        f"- Mandatory categories: {gates['mandatory_category_count']}",
        f"- Mandatory cells passed: {gates['passed_gate_count']}/{gates['mandatory_gate_count']}",
        f"- Candidate 10 bps total return: {candidate_10['total_return']:.6f}",
        f"- Candidate 10 bps CAGR: {candidate_10['cagr']:.6f}",
        f"- Candidate 10 bps Sharpe: {candidate_10['sharpe']:.6f}",
        f"- Candidate 10 bps maximum drawdown: {candidate_10['maximum_drawdown']:.6f}",
        f"- Candidate aggregate turnover: {candidate_10['turnover']:.6f}",
        "- Exactly one immutable non-target source opening",
        "- BTC/ETH/SOL remain sealed",
        "",
    ])
    return {
        "economic_metrics.json": metrics,
        "daily_returns.parquet": daily,
        "bootstrap.json": bootstrap,
        "gate_matrix.json": gates,
        "evaluation_result.json": result,
        "evaluation_audit.json": audit,
        "evaluation_report.md": report,
        "result.json": result,
        "audit.json": audit,
    }


def _write_core(output: Path, core: Mapping[str, Any]) -> None:
    for name, value in core.items():
        if name.endswith(".json"):
            write_json_atomic(output / name, value)
        elif name.endswith(".parquet"):
            _atomic_parquet(value, output / name)
        else:
            _atomic_text(output / name, str(value))


def _complete_packet(context: Mapping[str, Any], completion_path: Path) -> dict[str, Any]:
    packet = dict(context["prepare_packet"])
    packet["phase"] = "complete"
    packet["research_state"] = {
        "path": context["state_path"].relative_to(context["root"]).as_posix(),
        "sha256": file_sha256(context["state_path"]),
        "authorized_phase": "v85",
        "authorized_next_action": ACTION,
        "authorized_command": COMMAND,
    }
    packet["source_receipt"] = context["source_receipt"]
    packet["authorization"] = {
        "explicit_user_authorization": True,
        "exact_registered_unseal": True,
    }
    output = context["output"]
    packet["unseal"] = {
        "authorization_receipt": {
            "path": (output / AUTHORIZATION_FILE).relative_to(context["root"]).as_posix(),
            "sha256": file_sha256(output / AUTHORIZATION_FILE),
        },
        "outcome_packet": {
            "path": (output / OUTCOME_FILE).relative_to(context["root"]).as_posix(),
            "sha256": file_sha256(output / OUTCOME_FILE),
        },
        "outcome_receipt": {
            "path": (output / OUTCOME_RECEIPT_FILE).relative_to(context["root"]).as_posix(),
            "sha256": file_sha256(output / OUTCOME_RECEIPT_FILE),
        },
    }
    packet["completion"] = {
        "path": completion_path.relative_to(context["root"]).as_posix(),
        "sha256": file_sha256(completion_path),
    }
    packet["safety"] = {
        "target_assets_loaded": [],
        "retuning_performed": False,
        "thresholds_changed": False,
        "costs_or_accounting_changed": False,
        "second_unseal_attempted": False,
        "prospective_or_deployable_claim": False,
    }
    packet["replay"] = None
    return packet


def unseal_low_turnover_rank_evaluation(config: Mapping[str, Any]) -> dict[str, Any]:
    """Open the exact V82 source once and complete the frozen V85 evaluation."""

    context = _context(config)
    output = context["output"]
    lock = context["root"] / context["stage"]["runtime_contract"]["process_lock"]
    with _process_lock(lock):
        output.mkdir(parents=True, exist_ok=True)
        auth_path = output / AUTHORIZATION_FILE
        packet_path = output / OUTCOME_FILE
        outcome_receipt_path = output / OUTCOME_RECEIPT_FILE
        completion_path = output / "completion_receipt.json"
        if completion_path.is_file():
            result = _load_json(output / "evaluation_result.json", "V85 result")
            return {**result, "cached": True, "source_packet_reads_this_invocation": 0}
        if auth_path.exists() and not packet_path.exists():
            raise V85EvaluationError(
                "V85 authorization exists without an atomic outcome packet; fail closed"
            )
        if packet_path.exists() != outcome_receipt_path.exists():
            raise V85EvaluationError("V85 outcome packet/receipt is incomplete; fail closed")

        positions = _prepare_positions(context)
        source_reads = 0
        if not auth_path.exists():
            write_json_atomic(auth_path, _authorization_receipt(context))
        _require(
            _load_json(auth_path, "V85 authorization receipt")
            == _authorization_receipt(context),
            "V85 authorization receipt drift",
        )
        if not packet_path.exists():
            symbols = set(positions["candidate"]["symbol"])
            outcomes, access = _read_source_once(context, symbols)
            source_reads = 1
            _atomic_parquet(outcomes, packet_path)
            write_json_atomic(outcome_receipt_path, {
                "schema_version": "tlm-one-shot-outcome/v1",
                "evaluation_spec_sha256": context["prepare_packet"]["evaluation_spec"]["sha256"],
                "prepare_receipt_sha256": context["prepare_packet"]["prepare"]["receipt"]["sha256"],
                "registered_sha256": context["prepare_packet"]["registered"]["sha256"],
                "authorization_receipt_sha256": file_sha256(auth_path),
                "outcome_packet_sha256": file_sha256(packet_path),
                "unseal_count": 1,
                "source_outcome_rows_read": 5370,
                "source_packet_deserializations": 1,
                "written_atomically": True,
                "immutable": True,
                "access_receipt": access,
                "target_assets_loaded": [],
            })
        else:
            outcomes = _validate_cached_outcome(context)

        core = _compute_core(context, outcomes, positions)
        _write_core(output, core)
        result_hashes = {
            (output / name).relative_to(context["root"]).as_posix(): file_sha256(output / name)
            for name in CORE_FILES
        }
        result = core["evaluation_result.json"]
        completion = {
            "schema_version": "tlm-one-shot-completion/v1",
            "evaluation_spec_sha256": context["prepare_packet"]["evaluation_spec"]["sha256"],
            "prepare_receipt_sha256": context["prepare_packet"]["prepare"]["receipt"]["sha256"],
            "registered_sha256": context["prepare_packet"]["registered"]["sha256"],
            "outcome_receipt_sha256": file_sha256(outcome_receipt_path),
            "decision": result["one_shot_decision"],
            "result_artifacts": result_hashes,
            "unseal_count": 1,
            "source_outcome_rows_read": 5370,
            "source_packet_deserializations": 1,
            "target_assets_status": "sealed",
        }
        write_json_atomic(completion_path, completion)
        manifest_names = [
            AUTHORIZATION_FILE,
            OUTCOME_FILE,
            OUTCOME_RECEIPT_FILE,
            *CORE_FILES,
            "completion_receipt.json",
        ]
        manifest = {
            "schema_version": "v85-artifact-manifest/v1",
            "files": {name: file_sha256(output / name) for name in manifest_names},
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        write_json_atomic(output / "artifact_manifest.json", manifest)
        write_json_atomic(output / COMPLETE_PACKET_FILE, _complete_packet(context, completion_path))
        return {
            **result,
            "cached": False,
            "source_packet_reads_this_invocation": source_reads,
            "outcome_packet_sha256": file_sha256(packet_path),
            "outcome_receipt_sha256": file_sha256(outcome_receipt_path),
            "completion_receipt_sha256": file_sha256(completion_path),
        }


def replay_low_turnover_rank_evaluation(config: Mapping[str, Any]) -> dict[str, Any]:
    """Replay V85 from its immutable local packet without reopening V82."""

    context = _context(config)
    output = context["output"]
    _require((output / "completion_receipt.json").is_file(), "V85 completion is missing")
    outcomes = _validate_cached_outcome(context)
    positions = _prepare_positions(context)
    observed = {name: file_sha256(output / name) for name in CORE_FILES}
    recomputed = _compute_core(context, outcomes, positions)
    with TemporaryDirectory(dir=output, prefix=".v85-replay-") as temporary:
        replay_dir = Path(temporary)
        _write_core(replay_dir, recomputed)
        expected = {name: file_sha256(replay_dir / name) for name in CORE_FILES}
    _require(observed == expected, "V85 replay result hashes differ")
    replay = {
        "schema_version": "v85-replay/v1",
        "reused_existing_outcome_packet": True,
        "new_unseal_receipts": 0,
        "source_outcome_rows_read": 0,
        "source_packet_deserializations": 0,
        "result_hashes_match": True,
        "new_checkpoint_loads": 0,
        "new_inference": 0,
        "new_prediction_or_position_generation": 0,
        "core_file_hashes": expected,
        "target_assets_loaded": [],
    }
    write_json_atomic(output / "replay.json", replay)
    complete = _load_json(output / COMPLETE_PACKET_FILE, "V85 complete packet")
    packet = dict(complete)
    packet["phase"] = "replay"
    packet["replay"] = {
        "reused_existing_outcome_packet": True,
        "new_unseal_receipts": 0,
        "source_outcome_rows_read": 0,
        "result_hashes_match": True,
    }
    write_json_atomic(output / REPLAY_PACKET_FILE, packet)
    result = recomputed["evaluation_result.json"]
    return {
        "decision": result["decision"],
        "one_shot_decision": result["one_shot_decision"],
        "cached": True,
        "new_unseal_authorization_receipts": 0,
        "new_outcome_packets": 0,
        "source_outcome_rows_read": 0,
        "source_packet_deserializations": 0,
        "new_checkpoint_loads": 0,
        "new_inference": 0,
        "new_prediction_or_position_generation": 0,
        "result_hashes_match": True,
        "target_assets_loaded": [],
    }
