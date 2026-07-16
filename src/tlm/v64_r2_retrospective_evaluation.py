"""Exactly-once V72 post-hoc economic evaluation for the frozen V64-R2 policy."""

from __future__ import annotations

from contextlib import contextmanager
import json
import math
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


ACTION = "execute_v72_exactly_one_hash_bound_posthoc_outcome_unseal_and_complete_diagnostic"
COMMAND = (
    "PYTHONPATH=src python3 -m tlm v64-r2-retrospective-diagnostic-unseal "
    "--config configs/v72_v64_r2_retrospective_evaluation.yaml"
)
REPLAY_COMMAND = (
    "PYTHONPATH=src python3 -m tlm v64-r2-retrospective-diagnostic-replay "
    "--config configs/v72_v64_r2_retrospective_evaluation.yaml"
)
TARGET_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
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


class V72EvaluationError(RuntimeError):
    """Fail-closed V72 evaluation error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V72EvaluationError(message)


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V72EvaluationError(f"cannot load {name}: {path}") from exc
    _require(isinstance(value, dict), f"{name} must be a JSON object")
    return value


def _load_yaml(path: Path, name: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise V72EvaluationError(f"cannot load {name}: {path}") from exc
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
        frame.to_parquet(temporary, index=False, engine="pyarrow", compression="zstd")
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
        raise V72EvaluationError("another V72 evaluation process owns the lock") from exc
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
        yield
    finally:
        path.unlink(missing_ok=True)


def _verify_ref(root: Path, reference: Mapping[str, Any], name: str) -> Path:
    path = (root / str(reference["path"])).resolve()
    _require(root == path or root in path.parents, f"{name} path escapes repository")
    _require(path.is_file(), f"missing {name}: {path}")
    _require(file_sha256(path) == str(reference["file_sha256"]), f"{name} hash drift")
    return path


def _source_receipt(root: Path, files: list[str]) -> dict[str, Any]:
    _require(
        _git(root, "status", "--porcelain", "--untracked-files=all") == "",
        "V72 unseal requires a clean committed Git tree",
    )
    hashes = {}
    for relative in files:
        path = (root / relative).resolve()
        _require(root == path or root in path.parents, f"source path escapes root: {relative}")
        _require(path.is_file(), f"missing source file: {relative}")
        hashes[relative] = file_sha256(path)
    return {
        "schema_version": "v72-source-receipt/v1",
        "git_clean": True,
        "git_head": _git(root, "rev-parse", "HEAD"),
        "files": hashes,
        "bundle_sha256": canonical_sha256(hashes),
    }


def _context(config: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(str(config.get("project_root", "."))).resolve()
    state_path = root / str(config["research_state"])
    status = validate_research_state(root, state_path.relative_to(root))
    _require(status.get("passed") is True, "V72 research state failed")
    _require(status.get("authorized_phase") == "v72", "V72 phase drift")
    _require(status.get("authorized_next_action") == ACTION, "V72 action drift")
    _require(status.get("authorized_command") == COMMAND, "V72 command drift")

    stage_path = root / str(config["phase_contract"])
    stage = _load_yaml(stage_path, "V72 phase contract")
    _require(
        stage.get("stage_revision") == "v072_hash_bound_posthoc_outcome_unseal_r1",
        "V72 stage revision drift",
    )
    _require(stage.get("authorized_command") == COMMAND, "V72 stage command drift")
    _verify_ref(root, stage["parent_experiment"], "V71 parent experiment")
    _verify_ref(root, stage["base_phase_contract"], "V71 base phase")

    for name, reference in stage["prepare_packet"].items():
        if isinstance(reference, dict) and "path" in reference:
            _verify_ref(root, reference, f"V71 prepare {name}")
    authorization_path = _verify_ref(
        root, stage["explicit_user_authorization"], "V72 explicit authorization"
    )
    authorization = _load_json(authorization_path, "V72 explicit authorization")
    authorization_hash = authorization.pop("authorization_sha256", None)
    _require(
        authorization_hash == stage["explicit_user_authorization"]["canonical_sha256"]
        and canonical_sha256(authorization) == authorization_hash,
        "V72 explicit authorization canonical hash drift",
    )
    _require(authorization.get("authorized_action") == ACTION, "V72 authorization action drift")
    _require(authorization.get("maximum_unseal_count") == 1, "V72 authorization count drift")
    _require(authorization.get("target_assets_status") == "sealed", "V72 target authorization drift")

    prepare_dir = root / str(config["prepare_dir"])
    output = root / str(config["output_dir"])
    _require(
        str(config["prepare_dir"]) == stage["prepare_packet"]["output_dir"],
        "V72 prepare directory drift",
    )
    _require(
        str(config["output_dir"]) == stage["access_contract"]["output_dir"],
        "V72 output directory drift",
    )
    prepare_packet = _load_json(prepare_dir / "one_shot_packet.json", "V71 prepare packet")
    evaluation_spec = _load_json(prepare_dir / "evaluation_spec.json", "V71 evaluation spec")
    _require(prepare_packet.get("phase") == "prepare", "V71 packet is not prepare-phase")
    _require(
        prepare_packet["evaluation_spec"]["sha256"] == authorization["evaluation_spec_sha256"]
        and prepare_packet["prepare"]["receipt"]["sha256"]
        == authorization["prepare_receipt_sha256"]
        and prepare_packet["registered"]["sha256"] == authorization["registered_sha256"]
        and file_sha256(prepare_dir / "one_shot_packet.json")
        == authorization["one_shot_prepare_packet_sha256"],
        "V72 authorization does not bind the live V71 prepare packet",
    )
    _require(
        prepare_packet["prepare"]["authorizes_unseal"] is True
        and prepare_packet["prepare"]["outcome_rows_read"] == 0
        and all(prepare_packet["prepare"]["outcome_blind_gates"].values()),
        "V71 prepare packet does not authorize unseal",
    )
    _require(
        evaluation_spec["outcome_dependent_gates"]
        == prepare_packet["registered"]["gates"]
        and evaluation_spec["outcome_dependent_gates"]["maximum_absolute_drawdown"]
        == 0.35,
        "V72 outcome-dependent gate contract drift",
    )
    behavior = _load_json(prepare_dir / "behavior_gates.json", "V71 behavior gates")
    access = _load_json(prepare_dir / "data_access_receipt.json", "V71 data access")
    _require(behavior.get("passed") is True, "V71 behavior gates failed")
    _require(
        access.get("sealed_outcome_packet_reads") == 0
        and access.get("evaluation_outcome_rows_read") == 0
        and access.get("target_assets_loaded") == [],
        "V71 prepare outcome access drift",
    )

    outcome_access = stage["outcome_access_contract"]
    source_packet = root / str(outcome_access["source_packet"])
    source_receipt_path = root / str(outcome_access["source_receipt"])
    _require(
        source_packet.is_file()
        and file_sha256(source_packet) == outcome_access["source_packet_sha256"],
        "V72 sealed source packet hash drift",
    )
    _require(
        source_receipt_path.is_file()
        and file_sha256(source_receipt_path) == outcome_access["source_receipt_sha256"],
        "V72 sealed source receipt hash drift",
    )
    sealed_receipt = _load_json(source_receipt_path, "sealed V64 outcome receipt metadata")
    _require(
        sealed_receipt.get("outcome_packet_sha256") == outcome_access["source_packet_sha256"]
        and sealed_receipt.get("unseal_count") == 1
        and sealed_receipt.get("immutable") is True,
        "V72 sealed source receipt semantic drift",
    )
    bootstrap = dict(config["bootstrap"])
    expected_bootstrap = dict(stage["evaluation_contract"]["bootstrap"])
    expected_bootstrap.pop("portfolios")
    _require(bootstrap == expected_bootstrap, "V72 bootstrap config drift")
    source = _source_receipt(root, list(stage["source_receipt_files"]))
    return {
        "root": root,
        "state_path": state_path,
        "status": status,
        "stage": stage,
        "stage_path": stage_path,
        "prepare_dir": prepare_dir,
        "output": output,
        "prepare_packet": prepare_packet,
        "evaluation_spec": evaluation_spec,
        "authorization": authorization,
        "source_packet": source_packet,
        "source_receipt": source,
        "bootstrap": bootstrap,
    }


def _prepare_frames(context: Mapping[str, Any]) -> dict[str, pd.DataFrame]:
    prepare = context["stage"]["prepare_packet"]
    frames = {
        "assets": pd.read_parquet(context["root"] / prepare["asset_predictions"]["path"]),
        "candidate": pd.read_parquet(context["root"] / prepare["candidate_positions"]["path"]),
        "v64_control": pd.read_parquet(context["root"] / prepare["v64_control_positions"]["path"]),
        "equal_weight": pd.read_parquet(context["root"] / prepare["equal_weight_positions"]["path"]),
    }
    assets = frames["assets"]
    _require(len(assets) == 9794, "V72 asset prediction row drift")
    _require(
        {"date", "fold", "symbol", "eligible"}.issubset(assets.columns),
        "V72 asset prediction schema drift",
    )
    for name, frame in frames.items():
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        frame["fold"] = frame["fold"].astype(int)
        frame["symbol"] = frame["symbol"].astype(str)
        _require(set(frame["symbol"]).isdisjoint(TARGET_SYMBOLS), f"target reached {name}")
    _require(assets["eligible"].astype(bool).all(), "V72 asset keys contain ineligible rows")
    for name in ("candidate", "v64_control", "equal_weight"):
        frame = frames[name]
        _require(list(frame.columns) == list(POSITION_COLUMNS), f"V72 {name} schema drift")
        _require(len(frame) == 32130, f"V72 {name} row drift")
        _require(set(frame["cost_bps"].astype(int)) == {10, 20, 30}, f"V72 {name} cost drift")
    return frames


def _authorization_receipt(context: Mapping[str, Any]) -> dict[str, Any]:
    prepare = context["prepare_packet"]
    return {
        "schema_version": "tlm-one-shot-unseal-authorization/v1",
        "phase": "v72",
        "stage_revision": "v072_hash_bound_posthoc_outcome_unseal_r1",
        "family_id": context["stage"]["family_id"],
        "explicit_user_authorization": True,
        "exact_registered_unseal": True,
        "unseal_count": 1,
        "authorized_command": COMMAND,
        "authorization_payload": context["authorization"],
        "authorization_payload_sha256": context["stage"]["explicit_user_authorization"]["canonical_sha256"],
        "evaluation_spec_sha256": prepare["evaluation_spec"]["sha256"],
        "prepare_receipt_sha256": prepare["prepare"]["receipt"]["sha256"],
        "registered_sha256": prepare["registered"]["sha256"],
        "source_git_head": context["source_receipt"]["git_head"],
        "source_outcome_reread_allowed": False,
        "target_assets_loaded": [],
    }


def _verify_authorization(context: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    _require(dict(receipt) == _authorization_receipt(context), "V72 authorization receipt drift")


def _read_sealed_outcome_once(
    context: Mapping[str, Any], assets: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    access = context["stage"]["outcome_access_contract"]
    columns = list(access["allowed_columns"])
    frame = pd.read_parquet(context["source_packet"], columns=columns)
    _require(list(frame.columns) == columns, "V72 outcome projection drift")
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame["target_h1_maturity_date"] = pd.to_datetime(frame["target_h1_maturity_date"], utc=True)
    frame["fold"] = frame["fold"].astype(int)
    frame["symbol"] = frame["symbol"].astype(str)
    _require(not frame.duplicated(["date", "fold", "symbol"]).any(), "duplicate V72 outcome keys")
    keys = assets[["date", "fold", "symbol"]].sort_values(["date", "fold", "symbol"])
    outcome = keys.merge(frame, on=["date", "fold", "symbol"], how="left", validate="one_to_one")
    _require(len(outcome) == int(access["exact_key_count"]), "V72 outcome key count drift")
    values = outcome["target_h1_open_to_open_log_return"].to_numpy(dtype=np.float64)
    _require(np.isfinite(values).all(), "V72 outcome contains missing or non-finite returns")
    _require(
        outcome["date"].min() >= pd.Timestamp(access["signal_start"], tz="UTC")
        and outcome["date"].max() <= pd.Timestamp(access["signal_end"], tz="UTC"),
        "V72 outcome date scope drift",
    )
    _require(
        outcome["target_h1_maturity_date"].max()
        <= pd.Timestamp(access["maximum_maturity"], tz="UTC"),
        "V72 outcome maturity drift",
    )
    _require(set(outcome["symbol"]).isdisjoint(TARGET_SYMBOLS), "target reached V72 outcome")
    packet = outcome[columns].sort_values(["date", "fold", "symbol"]).reset_index(drop=True)
    receipt = {
        "source_packet": access["source_packet"],
        "source_packet_sha256": access["source_packet_sha256"],
        "requested_columns": columns,
        "exact_key_count": len(packet),
        "sealed_packet_deserializations": 1,
        "underlying_source_outcome_reads": 0,
        "target_assets_loaded": [],
    }
    return packet, receipt


def _validate_v72_outcome_packet(
    context: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = context["output"]
    packet_path = output / OUTCOME_FILE
    receipt_path = output / OUTCOME_RECEIPT_FILE
    _require(packet_path.is_file() and receipt_path.is_file(), "V72 outcome packet is incomplete")
    receipt = _load_json(receipt_path, "V72 outcome receipt")
    _require(receipt.get("schema_version") == "tlm-one-shot-outcome/v1", "V72 outcome schema drift")
    _require(receipt.get("unseal_count") == 1, "V72 unseal count drift")
    _require(
        receipt.get("authorization_receipt_sha256")
        == file_sha256(output / AUTHORIZATION_FILE),
        "V72 outcome authorization binding drift",
    )
    _require(receipt.get("outcome_packet_sha256") == file_sha256(packet_path), "V72 packet hash drift")
    _require(receipt.get("written_atomically") is True and receipt.get("immutable") is True, "V72 immutability drift")
    packet = pd.read_parquet(packet_path)
    packet["date"] = pd.to_datetime(packet["date"], utc=True)
    packet["target_h1_maturity_date"] = pd.to_datetime(packet["target_h1_maturity_date"], utc=True)
    packet["fold"] = packet["fold"].astype(int)
    packet["symbol"] = packet["symbol"].astype(str)
    _require(len(packet) == 9794, "V72 cached outcome row drift")
    _require(set(packet["symbol"]).isdisjoint(TARGET_SYMBOLS), "target in cached V72 packet")
    return packet, receipt


def _series_metrics(values: np.ndarray) -> dict[str, float]:
    returns = np.asarray(values, dtype=np.float64)
    equity = np.cumprod(1.0 + returns)
    total_return = float(equity[-1] - 1.0)
    standard_deviation = float(np.std(returns, ddof=1))
    sharpe = 0.0 if standard_deviation == 0.0 else float(np.sqrt(365.0) * np.mean(returns) / standard_deviation)
    peak = np.maximum.accumulate(np.maximum(equity, 1.0))
    maximum_drawdown = float(np.min(equity / peak - 1.0))
    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "maximum_drawdown": maximum_drawdown,
    }


def _portfolio_daily(
    portfolio: str, positions: pd.DataFrame, outcomes: pd.DataFrame
) -> pd.DataFrame:
    merged = positions.merge(
        outcomes[["date", "fold", "symbol", "target_h1_open_to_open_log_return"]],
        on=["date", "fold", "symbol"],
        how="left",
        validate="many_to_one",
    )
    missing_active = (merged["candidate_weight"] > 0.0) & merged[
        "target_h1_open_to_open_log_return"
    ].isna()
    _require(not missing_active.any(), f"active {portfolio} position lacks outcome")
    merged["simple_return"] = np.expm1(
        merged["target_h1_open_to_open_log_return"].fillna(0.0).to_numpy(dtype=np.float64)
    )
    merged["gross_contribution"] = merged["candidate_weight"] * merged["simple_return"]
    rows = []
    for (fold, cost), frame in merged.groupby(["fold", "cost_bps"], sort=True):
        current = frame.sort_values(["date", "symbol"])
        dates = sorted(current["date"].unique())
        symbols = sorted(current["symbol"].unique())
        weight = current.pivot(index="date", columns="symbol", values="candidate_weight").reindex(index=dates, columns=symbols)
        previous = np.vstack([np.zeros((1, len(symbols))), weight.to_numpy(dtype=np.float64)[:-1]])
        turnover = np.abs(weight.to_numpy(dtype=np.float64) - previous)
        turnover[-1] += np.abs(weight.to_numpy(dtype=np.float64)[-1])
        observed_turnover = turnover.sum(axis=1)
        registered_turnover = (
            current.groupby("date", sort=True)["total_turnover"]
            .first()
            .reindex(dates)
            .to_numpy(dtype=np.float64)
        )
        _require(
            np.allclose(observed_turnover, registered_turnover, atol=1.0e-12),
            f"{portfolio} turnover accounting drift",
        )
        gross = (
            current.groupby("date", sort=True)["gross_contribution"]
            .sum()
            .reindex(dates)
            .to_numpy(dtype=np.float64)
        )
        net = gross - registered_turnover * float(cost) / 10000.0
        for index, date in enumerate(dates):
            rows.append(
                {
                    "portfolio": portfolio,
                    "date": pd.Timestamp(date),
                    "fold": int(fold),
                    "cost_bps": int(cost),
                    "gross_return": float(gross[index]),
                    "turnover": float(registered_turnover[index]),
                    "net_return": float(net[index]),
                }
            )
    return pd.DataFrame(rows)


def _cash_daily(reference: pd.DataFrame) -> pd.DataFrame:
    frame = reference[["date", "fold", "cost_bps"]].drop_duplicates().copy()
    frame.insert(0, "portfolio", "cash")
    frame["gross_return"] = 0.0
    frame["turnover"] = 0.0
    frame["net_return"] = 0.0
    return frame


def _economic_metrics(daily: pd.DataFrame) -> dict[str, Any]:
    portfolios: dict[str, Any] = {}
    for portfolio, portfolio_frame in daily.groupby("portfolio", sort=True):
        folds: dict[str, Any] = {}
        for (fold, cost), frame in portfolio_frame.groupby(["fold", "cost_bps"], sort=True):
            current = frame.sort_values("date")
            cell = _series_metrics(current["net_return"].to_numpy(dtype=np.float64))
            cell["gross_total_return"] = float(
                np.prod(1.0 + current["gross_return"].to_numpy(dtype=np.float64)) - 1.0
            )
            cell["turnover"] = float(current["turnover"].sum())
            folds.setdefault(str(int(fold)), {})[str(int(cost))] = cell
        aggregate: dict[str, Any] = {}
        for cost, frame in portfolio_frame.groupby("cost_bps", sort=True):
            current = frame.groupby("date", sort=True).agg(
                net_return=("net_return", "mean"),
                gross_return=("gross_return", "mean"),
                turnover=("turnover", "mean"),
            )
            cell = _series_metrics(current["net_return"].to_numpy(dtype=np.float64))
            cell["gross_total_return"] = float(
                np.prod(1.0 + current["gross_return"].to_numpy(dtype=np.float64)) - 1.0
            )
            cell["turnover"] = float(current["turnover"].sum())
            aggregate[str(int(cost))] = cell
        portfolios[str(portfolio)] = {"folds": folds, "aggregate": aggregate}
    return {
        "schema_version": "v72-economic-metrics/v1",
        "evidence_tier": "posthoc_consumed_2025_diagnostic_only_not_confirmation",
        "portfolios": portfolios,
    }


def _bootstrap(daily: pd.DataFrame, contract: Mapping[str, Any]) -> dict[str, Any]:
    cells = []
    paths = int(contract["paths"])
    batch_size = int(contract["batch_size"])
    cost = int(contract["economic_cost_bps"])
    portfolio_order = ("candidate", "v64_control", "equal_weight")
    for portfolio_index, portfolio in enumerate(portfolio_order):
        values = (
            daily.loc[(daily["portfolio"] == portfolio) & (daily["cost_bps"] == cost)]
            .groupby("date", sort=True)["net_return"]
            .mean()
            .to_numpy(dtype=np.float64)
        )
        _require(len(values) == 357, f"V72 {portfolio} bootstrap length drift")
        for block in contract["block_lengths_days"]:
            seed = int(contract["base_seed"]) + portfolio_index * 10000 + int(block)
            rng = np.random.default_rng(seed)
            store = np.empty(paths, dtype=np.float64)
            cursor = 0
            while cursor < paths:
                size = min(batch_size, paths - cursor)
                indexes = circular_block_indices(len(values), int(block), size, rng)
                sampled = values[indexes]
                store[cursor : cursor + size] = np.prod(1.0 + sampled, axis=1) - 1.0
                cursor += size
            quantiles = np.quantile(store, [0.01, 0.05, 0.5, 0.95, 0.99])
            cells.append(
                {
                    "portfolio": portfolio,
                    "cost_bps": cost,
                    "block_length_days": int(block),
                    "paths": paths,
                    "seed": seed,
                    "mean": float(store.mean()),
                    "p01": float(quantiles[0]),
                    "p05": float(quantiles[1]),
                    "median": float(quantiles[2]),
                    "p95": float(quantiles[3]),
                    "p99": float(quantiles[4]),
                }
            )
    return {
        "schema_version": "v72-bootstrap/v1",
        "method": "circular_block",
        "cells": cells,
        "cell_count": len(cells),
    }


def _gate_matrix(
    metrics: Mapping[str, Any], bootstrap: Mapping[str, Any], maximum_drawdown: float
) -> dict[str, Any]:
    candidate = metrics["portfolios"]["candidate"]
    rows: list[dict[str, Any]] = []

    def add(name: str, scope: str, observed: float, operator: str, threshold: float) -> None:
        passed = observed > threshold if operator == "strictly_greater_than" else observed >= threshold
        rows.append(
            {
                "gate": name,
                "scope": scope,
                "observed": float(observed),
                "operator": operator,
                "threshold": float(threshold),
                "passed": bool(passed),
            }
        )

    for fold, cells in candidate["folds"].items():
        add(
            "positive_net_return_each_fold_at_10bps",
            f"fold_{fold}_10bps",
            cells["10"]["total_return"],
            "strictly_greater_than",
            0.0,
        )
    for cost, values in candidate["aggregate"].items():
        add(
            "aggregate_net_return_strictly_positive_all_costs",
            f"aggregate_{cost}bps",
            values["total_return"],
            "strictly_greater_than",
            0.0,
        )
        add(
            "aggregate_sharpe_strictly_positive_all_costs",
            f"aggregate_{cost}bps",
            values["sharpe"],
            "strictly_greater_than",
            0.0,
        )
    threshold = -float(maximum_drawdown)
    for fold, cells in candidate["folds"].items():
        for cost, values in cells.items():
            add(
                "maximum_absolute_drawdown",
                f"fold_{fold}_{cost}bps",
                values["maximum_drawdown"],
                "greater_than_or_equal",
                threshold,
            )
    for cost, values in candidate["aggregate"].items():
        add(
            "maximum_absolute_drawdown",
            f"aggregate_{cost}bps",
            values["maximum_drawdown"],
            "greater_than_or_equal",
            threshold,
        )
    for cell in bootstrap["cells"]:
        if cell["portfolio"] == "candidate":
            add(
                "economic_bootstrap_p05_strictly_positive_all_blocks",
                f"block_{cell['block_length_days']}",
                cell["p05"],
                "strictly_greater_than",
                0.0,
            )
    _require(len(rows) == 24, "V72 mandatory gate count drift")
    return {
        "schema_version": "v72-gate-matrix/v1",
        "gates": rows,
        "mandatory_gate_count": len(rows),
        "passed_gate_count": sum(row["passed"] for row in rows),
        "failed_gate_count": sum(not row["passed"] for row in rows),
        "all_passed": all(row["passed"] for row in rows),
    }


def _compute_core(
    context: Mapping[str, Any], outcomes: pd.DataFrame, frames: Mapping[str, pd.DataFrame]
) -> dict[str, Any]:
    daily = pd.concat(
        [
            _portfolio_daily("candidate", frames["candidate"], outcomes),
            _portfolio_daily("v64_control", frames["v64_control"], outcomes),
            _portfolio_daily("equal_weight", frames["equal_weight"], outcomes),
        ],
        ignore_index=True,
    )
    daily = pd.concat([daily, _cash_daily(daily)], ignore_index=True)
    daily = daily.sort_values(["portfolio", "date", "fold", "cost_bps"]).reset_index(drop=True)
    metrics = _economic_metrics(daily)
    bootstrap = _bootstrap(daily, context["bootstrap"])
    gates = _gate_matrix(
        metrics,
        bootstrap,
        float(
            context["evaluation_spec"]["outcome_dependent_gates"][
                "maximum_absolute_drawdown"
            ]
        ),
    )
    diagnostic = "pass" if gates["all_passed"] else "fail"
    next_action = "authorize_v73_record_posthoc_diagnostic_result_only"
    candidate = metrics["portfolios"]["candidate"]["aggregate"]
    controls = {
        name: metrics["portfolios"][name]["aggregate"]
        for name in ("v64_control", "equal_weight", "cash")
    }
    result = {
        "schema_version": "v72-evaluation-result/v1",
        "decision": next_action,
        "diagnostic_outcome": diagnostic,
        "one_shot_decision": "pass" if gates["all_passed"] else "retire",
        "family_status_changed": False,
        "evidence_tier": "posthoc_consumed_2025_diagnostic_only_not_confirmation",
        "mandatory_gate_count": gates["mandatory_gate_count"],
        "passed_gate_count": gates["passed_gate_count"],
        "failed_gate_count": gates["failed_gate_count"],
        "candidate_aggregate": candidate,
        "control_aggregate": controls,
        "unseal_count": 1,
        "sealed_packet_deserializations": 1,
        "underlying_source_outcome_reads": 0,
        "retuning_performed": False,
        "prediction_or_position_regeneration": False,
        "target_assets_loaded": [],
        "target_predictions": 0,
        "target_pnl_evaluations": 0,
        "deployable": False,
    }
    audit = {
        "schema_version": "v72-evaluation-audit/v1",
        "checks": {
            "exactly_one_hash_bound_packet_unseal": True,
            "underlying_source_outcome_reads_equal_zero": True,
            "immutable_v71_predictions_positions_and_controls_reused": True,
            "all_registered_cost_bootstrap_and_gate_cells_preserved": True,
            "no_retuning_selection_or_regeneration": True,
            "family_status_unchanged": True,
            "target_assets_remain_sealed": True,
            "posthoc_consumed_evidence_label_preserved": True,
        },
        "passed": True,
        "scientific_gates_all_passed": gates["all_passed"],
    }
    candidate_10 = candidate["10"]
    report = "\n".join(
        [
            "# TLM V72 V64-R2 Post-hoc Retrospective Evaluation",
            "",
            f"**Diagnostic outcome:** `{diagnostic}`",
            "",
            "This is consumed-2025 post-hoc evidence, not clean confirmation or deployable evidence.",
            "",
            f"- Mandatory candidate gates: {gates['mandatory_gate_count']}",
            f"- Passed: {gates['passed_gate_count']}",
            f"- Failed: {gates['failed_gate_count']}",
            f"- Candidate 10 bps total return: {candidate_10['total_return']:.6f}",
            f"- Candidate 10 bps Sharpe: {candidate_10['sharpe']:.6f}",
            f"- Candidate 10 bps max drawdown: {candidate_10['maximum_drawdown']:.6f}",
            f"- Frozen V64 control 10 bps return: {controls['v64_control']['10']['total_return']:.6f}",
            f"- Equal-weight 10 bps return: {controls['equal_weight']['10']['total_return']:.6f}",
            "- Exactly one immutable non-target packet opening; zero underlying source rereads",
            "- BTC/ETH/SOL: sealed",
            "- Family status: unchanged",
            "",
        ]
    )
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
        "authorized_phase": "v72",
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
        "clean_holdout_or_prospective_claim": False,
    }
    packet["replay"] = None
    return packet


def unseal_v64_r2_retrospective_diagnostic(config: Mapping[str, Any]) -> dict[str, Any]:
    """Open the exact immutable non-target packet once and complete V72."""

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
            result = _load_json(output / "evaluation_result.json", "V72 result")
            return {**result, "cached": True, "source_packet_reads_this_invocation": 0}
        if auth_path.exists() and not packet_path.exists():
            raise V72EvaluationError(
                "V72 authorization exists without an atomic outcome packet; fail closed"
            )
        if packet_path.exists() != outcome_receipt_path.exists():
            raise V72EvaluationError("V72 outcome packet/receipt is incomplete; fail closed")

        frames = _prepare_frames(context)
        source_reads = 0
        if not auth_path.exists():
            write_json_atomic(auth_path, _authorization_receipt(context))
        authorization = _load_json(auth_path, "V72 authorization receipt")
        _verify_authorization(context, authorization)
        if not packet_path.exists():
            outcomes, access_receipt = _read_sealed_outcome_once(context, frames["assets"])
            source_reads = 1
            _atomic_parquet(outcomes, packet_path)
            outcome_receipt = {
                "schema_version": "tlm-one-shot-outcome/v1",
                "evaluation_spec_sha256": context["prepare_packet"]["evaluation_spec"]["sha256"],
                "prepare_receipt_sha256": context["prepare_packet"]["prepare"]["receipt"]["sha256"],
                "registered_sha256": context["prepare_packet"]["registered"]["sha256"],
                "authorization_receipt_sha256": file_sha256(auth_path),
                "outcome_packet_sha256": file_sha256(packet_path),
                "unseal_count": 1,
                "source_outcome_reads": 0,
                "sealed_packet_deserializations": 1,
                "written_atomically": True,
                "immutable": True,
                "access_receipt": access_receipt,
                "target_assets_loaded": [],
            }
            write_json_atomic(outcome_receipt_path, outcome_receipt)
        else:
            outcomes, _ = _validate_v72_outcome_packet(context)
        core = _compute_core(context, outcomes, frames)
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
            "family_status_changed": False,
            "result_artifacts": result_hashes,
            "unseal_count": 1,
            "sealed_packet_deserializations": 1,
            "source_outcome_reads": 0,
            "target_assets_status": "sealed",
        }
        write_json_atomic(completion_path, completion)
        manifest_files = [
            AUTHORIZATION_FILE,
            OUTCOME_FILE,
            OUTCOME_RECEIPT_FILE,
            *CORE_FILES,
            "completion_receipt.json",
        ]
        manifest = {
            "schema_version": "v72-artifact-manifest/v1",
            "files": {name: file_sha256(output / name) for name in manifest_files},
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


def replay_v64_r2_retrospective_diagnostic(config: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute V72 from its immutable packet without reopening V64 outcomes."""

    context = _context(config)
    output = context["output"]
    _require((output / "completion_receipt.json").is_file(), "V72 completion is missing")
    outcomes, _ = _validate_v72_outcome_packet(context)
    frames = _prepare_frames(context)
    observed = {name: file_sha256(output / name) for name in CORE_FILES}
    recomputed = _compute_core(context, outcomes, frames)
    with TemporaryDirectory(dir=output, prefix=".v72-replay-") as temporary:
        replay_dir = Path(temporary)
        _write_core(replay_dir, recomputed)
        expected = {name: file_sha256(replay_dir / name) for name in CORE_FILES}
    _require(observed == expected, "V72 replay result hashes differ")
    replay = {
        "schema_version": "v72-replay/v1",
        "reused_existing_outcome_packet": True,
        "new_unseal_receipts": 0,
        "source_outcome_rows_read": 0,
        "sealed_source_packet_deserializations": 0,
        "result_hashes_match": True,
        "new_checkpoint_loads": 0,
        "new_inference": 0,
        "new_position_generation": 0,
        "core_file_hashes": expected,
        "target_assets_loaded": [],
    }
    write_json_atomic(output / "replay.json", replay)
    complete = _load_json(output / COMPLETE_PACKET_FILE, "V72 complete packet")
    packet = dict(complete)
    packet["phase"] = "replay"
    packet["replay"] = {
        "reused_existing_outcome_packet": True,
        "new_unseal_receipts": 0,
        "source_outcome_rows_read": 0,
        "result_hashes_match": True,
    }
    write_json_atomic(output / REPLAY_PACKET_FILE, packet)
    return {
        "decision": recomputed["evaluation_result.json"]["decision"],
        "diagnostic_outcome": recomputed["evaluation_result.json"]["diagnostic_outcome"],
        "cached": True,
        "files_rewritten": 0,
        "new_unseal_authorization_receipts": 0,
        "new_outcome_packets": 0,
        "source_outcome_rows_read": 0,
        "sealed_source_packet_deserializations": 0,
        "new_checkpoint_loads": 0,
        "new_inference": 0,
        "new_position_generation": 0,
        "result_hashes_match": True,
        "target_assets_loaded": [],
    }
