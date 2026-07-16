"""Exactly-once V64 outcome unseal, frozen evaluation, and source-free replay."""

from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import yaml

from .core.artifacts import canonical_sha256, file_sha256, write_json_atomic
from .monte_carlo import circular_block_indices
from .research_workflow import validate_research_state


TARGET_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
AUTHORIZATION_FILE = "unseal_authorization_receipt.json"
OUTCOME_FILE = "outcome_packet.parquet"
OUTCOME_RECEIPT_FILE = "outcome_receipt.json"
COMPLETE_PACKET_FILE = "one_shot_complete_packet.json"
REPLAY_PACKET_FILE = "one_shot_replay_packet.json"
CORE_FILES = (
    "metrics.json",
    "bootstrap.json",
    "gate_matrix.json",
    "attribution.json",
    "evaluation_result.json",
    "evaluation_audit.json",
    "evaluation_report.md",
)


class V64EvaluationError(RuntimeError):
    """Fail-closed V64 evaluation error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise V64EvaluationError(message)


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V64EvaluationError(f"cannot load {name}: {path}") from exc
    _require(isinstance(value, dict), f"{name} must be a JSON object")
    return value


def _load_yaml(path: Path, name: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise V64EvaluationError(f"cannot load {name}: {path}") from exc
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


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False,
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.replace(path)


@contextmanager
def _process_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise V64EvaluationError("another V64 evaluation process owns the lock") from exc
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
        yield
    finally:
        path.unlink(missing_ok=True)


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=root, text=True, capture_output=True, check=False
    )
    _require(result.returncode == 0, f"git {' '.join(arguments)} failed")
    return result.stdout.strip()


def _source_receipt(root: Path, files: list[str]) -> dict[str, Any]:
    _require(
        _git(root, "status", "--porcelain", "--untracked-files=all") == "",
        "V64 unseal requires a clean committed Git tree",
    )
    hashes = {}
    for relative in files:
        path = (root / relative).resolve()
        _require(root == path or root in path.parents, f"source path escapes root: {relative}")
        _require(path.is_file(), f"missing source file: {relative}")
        hashes[relative] = file_sha256(path)
    return {
        "git_clean": True,
        "git_head": _git(root, "rev-parse", "HEAD"),
        "files": hashes,
        "bundle_sha256": canonical_sha256(hashes),
    }


def _evaluation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config.get("decoupled_rank_state_evaluation")
    _require(isinstance(value, dict) and value.get("version") == "v64", "V64 config drift")
    return dict(value)


def _verify_ref(root: Path, reference: Mapping[str, Any], name: str) -> Path:
    path = root / str(reference["path"])
    _require(path.is_file(), f"missing {name}: {path}")
    _require(file_sha256(path) == reference["file_sha256"], f"{name} hash drift")
    return path


def _context(config: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = _evaluation_config(config)
    root = Path(evaluation.get("project_root", ".")).resolve()
    status = validate_research_state(root, evaluation["research_state"])
    _require(status.get("passed") is True, "V64 research state failed")
    expected_action = (
        "execute_v64_exactly_one_registered_non_target_outcome_unseal_and_complete_evaluation"
    )
    expected_command = (
        "PYTHONPATH=src python3 -m tlm decoupled-rank-state-evaluation-unseal "
        "--config configs/v64_decoupled_rank_state_evaluation.yaml"
    )
    _require(status.get("authorized_phase") == "v64", "V64 phase drift")
    _require(status.get("authorized_next_action") == expected_action, "V64 action drift")
    _require(status.get("authorized_command") == expected_command, "V64 command drift")
    stage_path = root / status["phase_contract_path"]
    stage = _load_yaml(stage_path, "V64 unseal stage")
    _require(stage.get("stage_revision") == "v064_unseal_r1", "V64 stage revision drift")
    _require(stage.get("authorized_command") == expected_command, "V64 stage command drift")
    base_path = _verify_ref(root, stage["base_phase_contract"], "V64 base phase")
    base = _load_yaml(base_path, "V64 base phase")
    output = root / stage["prepare_packet"]["output_dir"]
    for name, reference in stage["prepare_packet"].items():
        if isinstance(reference, dict) and "path" in reference:
            _verify_ref(root, reference, f"V64 prepare {name}")
    payload = stage["explicit_user_authorization"]["payload"]
    _require(
        canonical_sha256(payload) == stage["explicit_user_authorization"]["canonical_sha256"],
        "V64 user authorization hash drift",
    )
    _require(payload["authorized_action"] == expected_action, "V64 user authorization action drift")
    _require(payload["maximum_unseal_count"] == 1, "V64 unseal count authorization drift")
    _require(payload["target_assets_status"] == "sealed", "V64 target authorization drift")
    prepare_packet = _load_json(output / "one_shot_packet.json", "V64 prepare packet")
    _require(
        prepare_packet["evaluation_spec"]["sha256"] == payload["evaluation_spec_sha256"]
        and prepare_packet["prepare"]["receipt"]["sha256"] == payload["prepare_receipt_sha256"]
        and file_sha256(output / "one_shot_packet.json") == payload["one_shot_packet_sha256"],
        "V64 explicit authorization does not bind the live prepare packet",
    )
    source = _source_receipt(root, list(stage["source_receipt_files"]))
    return {
        "root": root,
        "evaluation": evaluation,
        "status": status,
        "stage": stage,
        "stage_path": stage_path,
        "base": base,
        "output": output,
        "prepare_packet": prepare_packet,
        "source_receipt": source,
    }


def _key_records(assets: pd.DataFrame) -> list[dict[str, Any]]:
    frame = assets.loc[assets["eligible"].astype(bool), ["date", "fold", "symbol"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.strftime("%Y-%m-%d")
    frame["fold"] = frame["fold"].astype(int)
    frame["symbol"] = frame["symbol"].astype(str)
    return frame.sort_values(["date", "fold", "symbol"]).to_dict("records")


def _prepare_frames(context: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output = context["output"]
    predictions = pd.read_parquet(output / "context_predictions.parquet")
    assets = pd.read_parquet(output / "asset_predictions.parquet")
    positions = pd.read_parquet(output / "positions.parquet")
    expected = context["stage"]["prepare_packet"]
    _require(len(predictions) == expected["context_predictions"]["rows"], "context row drift")
    _require(len(assets) == expected["asset_predictions"]["rows"], "asset row drift")
    _require(len(positions) == expected["positions"]["rows"], "position row drift")
    for frame in (predictions, assets, positions):
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        frame["fold"] = frame["fold"].astype(int)
        frame["symbol"] = frame["symbol"].astype(str)
        _require(set(frame["symbol"]).isdisjoint(TARGET_SYMBOLS), "target reached V64 prepare frame")
    keys = _key_records(assets)
    access = context["stage"]["outcome_access_contract"]
    _require(len(keys) == access["exact_key_count"], "V64 outcome key count drift")
    _require(canonical_sha256(keys) == access["exact_key_sha256"], "V64 outcome key hash drift")
    return predictions, assets, positions


def _authorization_receipt(context: Mapping[str, Any]) -> dict[str, Any]:
    stage = context["stage"]
    prepare = context["prepare_packet"]
    return {
        "schema_version": "tlm-one-shot-unseal-authorization/v1",
        "phase": "v64",
        "stage_revision": "v064_unseal_r1",
        "family_id": stage["family_id"],
        "explicit_user_authorization": True,
        "exact_registered_unseal": True,
        "unseal_count": 1,
        "authorized_command": stage["authorized_command"],
        "authorization_payload": stage["explicit_user_authorization"]["payload"],
        "authorization_payload_sha256": stage["explicit_user_authorization"]["canonical_sha256"],
        "evaluation_spec_sha256": prepare["evaluation_spec"]["sha256"],
        "prepare_receipt_sha256": prepare["prepare"]["receipt"]["sha256"],
        "registered_sha256": prepare["registered"]["sha256"],
        "source_git_head": context["source_receipt"]["git_head"],
        "target_assets_loaded": [],
    }


def _verify_authorization(context: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    expected = _authorization_receipt(context)
    _require(dict(receipt) == expected, "V64 authorization receipt drift")


def _outcome_expression(keys: list[dict[str, Any]]) -> ds.Expression:
    by_symbol: dict[str, list[pd.Timestamp]] = {}
    for row in keys:
        timestamp = pd.Timestamp(row["date"])
        timestamp = (
            timestamp.tz_localize("UTC")
            if timestamp.tzinfo is None
            else timestamp.tz_convert("UTC")
        )
        by_symbol.setdefault(row["symbol"], []).append(timestamp)
    expression: ds.Expression | None = None
    for symbol, dates in sorted(by_symbol.items()):
        current = (ds.field("symbol") == symbol) & ds.field("date").isin(
            [date.to_pydatetime() for date in sorted(set(dates))]
        )
        expression = current if expression is None else expression | current
    _require(expression is not None, "empty V64 outcome expression")
    return expression


def _read_source_outcomes(
    context: Mapping[str, Any], assets: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    access = context["stage"]["outcome_access_contract"]
    source = context["root"] / access["source"]
    _require(file_sha256(source) == access["source_file_sha256"], "V64 outcome source hash drift")
    keys = _key_records(assets)
    table = ds.dataset(source, format="parquet").to_table(
        columns=list(access["allowed_columns"]),
        filter=_outcome_expression(keys),
        use_threads=False,
    )
    frame = table.to_pandas()
    _require(list(frame.columns) == list(access["allowed_columns"]), "V64 outcome projection drift")
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame["target_h1_maturity_date"] = pd.to_datetime(frame["target_h1_maturity_date"], utc=True)
    frame["symbol"] = frame["symbol"].astype(str)
    _require(not frame.duplicated(["date", "symbol"]).any(), "duplicate V64 outcome keys")
    key_frame = assets.loc[assets["eligible"].astype(bool), ["date", "fold", "symbol"]].copy()
    outcome = key_frame.merge(frame, on=["date", "symbol"], how="left", validate="one_to_one")
    _require(len(outcome) == access["exact_key_count"], "V64 outcome merge count drift")
    _require(outcome["h1_label_complete"].astype(bool).all(), "V64 outcome contains incomplete labels")
    values = outcome["target_h1_open_to_open_log_return"].to_numpy(dtype=np.float64)
    _require(np.isfinite(values).all(), "V64 outcome contains non-finite returns")
    _require(
        outcome["target_h1_maturity_date"].max()
        <= pd.Timestamp(access["maximum_maturity"], tz="UTC"),
        "V64 outcome exceeds maturity boundary",
    )
    _require(set(outcome["symbol"]).isdisjoint(TARGET_SYMBOLS), "target reached outcome packet")
    packet = outcome[
        ["date", "fold", "symbol", "target_h1_maturity_date", "target_h1_open_to_open_log_return"]
    ].sort_values(["date", "fold", "symbol"]).reset_index(drop=True)
    packet["fold"] = packet["fold"].astype(int)
    observed_keys = packet[["date", "fold", "symbol"]].copy()
    observed_keys["date"] = observed_keys["date"].dt.strftime("%Y-%m-%d")
    _require(canonical_sha256(observed_keys.to_dict("records")) == access["exact_key_sha256"], "V64 outcome packet key drift")
    receipt = {
        "source": access["source"],
        "source_file_sha256": access["source_file_sha256"],
        "requested_columns": list(access["allowed_columns"]),
        "exact_key_count": len(packet),
        "exact_key_sha256": access["exact_key_sha256"],
        "source_outcome_reads": 1,
        "target_assets_loaded": [],
    }
    return packet, receipt


def _validate_outcome_packet(context: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = context["output"]
    packet_path = output / OUTCOME_FILE
    receipt_path = output / OUTCOME_RECEIPT_FILE
    _require(packet_path.is_file() and receipt_path.is_file(), "V64 outcome packet is incomplete")
    receipt = _load_json(receipt_path, "V64 outcome receipt")
    authorization_path = output / AUTHORIZATION_FILE
    _require(receipt.get("schema_version") == "tlm-one-shot-outcome/v1", "V64 outcome receipt schema drift")
    _require(receipt.get("unseal_count") == 1, "V64 outcome unseal count drift")
    _require(receipt.get("authorization_receipt_sha256") == file_sha256(authorization_path), "V64 outcome authorization binding drift")
    _require(receipt.get("outcome_packet_sha256") == file_sha256(packet_path), "V64 outcome packet hash drift")
    _require(receipt.get("written_atomically") is True and receipt.get("immutable") is True, "V64 outcome immutability drift")
    packet = pd.read_parquet(packet_path)
    packet["date"] = pd.to_datetime(packet["date"], utc=True)
    packet["target_h1_maturity_date"] = pd.to_datetime(packet["target_h1_maturity_date"], utc=True)
    _require(len(packet) == context["stage"]["outcome_access_contract"]["exact_key_count"], "V64 cached outcome row drift")
    _require(set(packet["symbol"]).isdisjoint(TARGET_SYMBOLS), "target in cached outcome packet")
    return packet, receipt


def _spearman(predicted: np.ndarray, actual: np.ndarray) -> float:
    pred_rank = pd.Series(predicted).rank(method="average").to_numpy(dtype=np.float64)
    actual_rank = pd.Series(actual).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(pred_rank) == 0 or np.std(actual_rank) == 0:
        return math.nan
    return float(np.corrcoef(pred_rank, actual_rank)[0, 1])


def _predictive_metrics(
    predictions: pd.DataFrame, outcomes: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame]:
    grouped = predictions.groupby(
        ["date", "fold", "triplet_key", "slot", "symbol"], sort=True, as_index=False
    ).agg(
        predicted_excess=("raw_excess", "mean"),
        predicted_market=("market_component", "mean"),
        predicted_absolute=("absolute_edge", "mean"),
    )
    joined = grouped.merge(
        outcomes[["date", "fold", "symbol", "target_h1_open_to_open_log_return"]],
        on=["date", "fold", "symbol"], how="left", validate="many_to_one",
    )
    _require(joined["target_h1_open_to_open_log_return"].notna().all(), "missing context outcome")
    rows = []
    tolerance = 1.0e-12
    for (date, fold, triplet), frame in joined.groupby(
        ["date", "fold", "triplet_key"], sort=True
    ):
        current = frame.sort_values("slot")
        _require(len(current) == 3, "V64 context width drift")
        predicted = current["predicted_excess"].to_numpy(dtype=np.float64)
        actual = current["target_h1_open_to_open_log_return"].to_numpy(dtype=np.float64)
        centered = actual - actual.mean()
        correct = 0
        active = 0
        for left, right in ((0, 1), (0, 2), (1, 2)):
            actual_difference = actual[left] - actual[right]
            if abs(actual_difference) <= tolerance:
                continue
            active += 1
            correct += int((predicted[left] - predicted[right]) * actual_difference > 0)
        _require(active > 0, "V64 context has zero active pairs")
        top = int(np.argmax(predicted))
        market_prediction = float(current["predicted_market"].mean())
        rows.append({
            "date": pd.Timestamp(date),
            "fold": int(fold),
            "triplet_key": str(triplet),
            "spearman": _spearman(predicted, centered),
            "pairwise_accuracy": correct / active,
            "top1_centered_excess": float(centered[top]),
            "state_absolute_error": abs(market_prediction - float(actual.mean())),
            "state_direction_correct": float(market_prediction * float(actual.mean()) > 0),
            "absolute_error": float(np.mean(np.abs(current["predicted_absolute"].to_numpy(dtype=np.float64) - actual))),
            "absolute_direction_accuracy": float(np.mean(current["predicted_absolute"].to_numpy(dtype=np.float64) * actual > 0)),
        })
    context_metrics = pd.DataFrame(rows)
    _require(np.isfinite(context_metrics.drop(columns=["date", "triplet_key"]).to_numpy(dtype=np.float64)).all(), "non-finite V64 predictive metric")
    daily = context_metrics.groupby(["date", "fold"], sort=True, as_index=False).agg(
        spearman=("spearman", "mean"),
        pairwise_accuracy=("pairwise_accuracy", "mean"),
        top1_centered_excess=("top1_centered_excess", "mean"),
        state_mae=("state_absolute_error", "mean"),
        state_direction_accuracy=("state_direction_correct", "mean"),
        absolute_mae=("absolute_error", "mean"),
        absolute_direction_accuracy=("absolute_direction_accuracy", "mean"),
    )
    folds = {}
    for fold, frame in daily.groupby("fold", sort=True):
        folds[str(int(fold))] = {
            name: float(frame[name].mean())
            for name in (
                "spearman", "pairwise_accuracy", "top1_centered_excess",
                "state_mae", "state_direction_accuracy", "absolute_mae",
                "absolute_direction_accuracy",
            )
        }
    aggregate = {
        name: float(np.mean([row[name] for row in folds.values()]))
        for name in next(iter(folds.values()))
    }
    return {
        "folds": folds,
        "aggregate": aggregate,
        "context_count": len(context_metrics),
        "daily_fold_count": len(daily),
    }, daily


def _series_metrics(values: np.ndarray) -> dict[str, float]:
    returns = np.asarray(values, dtype=np.float64)
    equity = np.cumprod(1.0 + returns)
    total_return = float(equity[-1] - 1.0)
    standard_deviation = float(np.std(returns, ddof=1))
    sharpe = 0.0 if standard_deviation == 0 else float(np.sqrt(365.0) * np.mean(returns) / standard_deviation)
    peak = np.maximum.accumulate(np.maximum(equity, 1.0))
    maximum_drawdown = float(np.min(equity / peak - 1.0))
    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "maximum_drawdown": maximum_drawdown,
    }


def _economic_metrics(
    positions: pd.DataFrame, outcomes: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    merged = positions.merge(
        outcomes[["date", "fold", "symbol", "target_h1_open_to_open_log_return"]],
        on=["date", "fold", "symbol"], how="left", validate="many_to_one",
    )
    missing_active = (merged["candidate_weight"] > 0) & merged["target_h1_open_to_open_log_return"].isna()
    _require(not missing_active.any(), "active V64 position lacks outcome")
    merged["simple_return"] = np.expm1(
        merged["target_h1_open_to_open_log_return"].fillna(0.0).to_numpy(dtype=np.float64)
    )
    merged["gross_contribution"] = merged["candidate_weight"] * merged["simple_return"]
    daily_rows = []
    asset_attribution = []
    fold_metrics: dict[str, Any] = {}
    episode_rows = []
    for (fold, cost), frame in merged.groupby(["fold", "cost_bps"], sort=True):
        current = frame.sort_values(["date", "symbol"]).copy()
        dates = sorted(current["date"].unique())
        symbols = sorted(current["symbol"].unique())
        weight = current.pivot(index="date", columns="symbol", values="candidate_weight").reindex(index=dates, columns=symbols)
        contribution = current.pivot(index="date", columns="symbol", values="gross_contribution").reindex(index=dates, columns=symbols)
        previous = np.vstack([np.zeros((1, len(symbols))), weight.to_numpy(dtype=np.float64)[:-1]])
        turnover_component = np.abs(weight.to_numpy(dtype=np.float64) - previous)
        turnover_component[-1] += np.abs(weight.to_numpy(dtype=np.float64)[-1])
        daily_turnover = turnover_component.sum(axis=1)
        registered_turnover = current.groupby("date", sort=True)["total_turnover"].first().reindex(dates).to_numpy(dtype=np.float64)
        _require(np.allclose(daily_turnover, registered_turnover, atol=1.0e-12), "V64 turnover accounting drift")
        gross = contribution.sum(axis=1).to_numpy(dtype=np.float64)
        net = gross - daily_turnover * float(cost) / 10000.0
        for index, date in enumerate(dates):
            daily_rows.append({
                "date": pd.Timestamp(date), "fold": int(fold), "cost_bps": int(cost),
                "gross_return": float(gross[index]), "turnover": float(daily_turnover[index]),
                "net_return": float(net[index]),
            })
        for symbol_index, symbol in enumerate(symbols):
            net_contribution = contribution[symbol].to_numpy(dtype=np.float64) - turnover_component[:, symbol_index] * float(cost) / 10000.0
            asset_attribution.append({
                "fold": int(fold), "cost_bps": int(cost), "symbol": symbol,
                "gross_contribution": float(contribution[symbol].sum()),
                "turnover": float(turnover_component[:, symbol_index].sum()),
                "net_contribution": float(net_contribution.sum()),
            })
            active = weight[symbol].to_numpy(dtype=np.float64) > 0
            start = None
            for index, is_active in enumerate(np.append(active, False)):
                if is_active and start is None:
                    start = index
                elif not is_active and start is not None:
                    stop = index
                    values = net_contribution[start:stop]
                    episode_rows.append({
                        "fold": int(fold), "cost_bps": int(cost), "symbol": symbol,
                        "start": str(pd.Timestamp(dates[start]).date()),
                        "end": str(pd.Timestamp(dates[stop - 1]).date()),
                        "duration_days": stop - start,
                        "net_return": float(np.prod(1.0 + values) - 1.0),
                    })
                    start = None
        fold_metrics.setdefault(str(int(fold)), {})[str(int(cost))] = {
            **_series_metrics(net),
            "gross_total_return": float(np.prod(1.0 + gross) - 1.0),
            "turnover": float(daily_turnover.sum()),
            "transition_days": int((daily_turnover > 0).sum()),
        }
    daily = pd.DataFrame(daily_rows).sort_values(["date", "fold", "cost_bps"]).reset_index(drop=True)
    aggregate_metrics = {}
    month_rows = []
    for cost, frame in daily.groupby("cost_bps", sort=True):
        aggregate = frame.groupby("date", sort=True)["net_return"].mean()
        aggregate_metrics[str(int(cost))] = _series_metrics(aggregate.to_numpy(dtype=np.float64))
        aggregate_metrics[str(int(cost))]["turnover"] = float(
            frame.groupby("date", sort=True)["turnover"].mean().sum()
        )
        month_index = aggregate.index.tz_localize(None).to_period("M")
        month = aggregate.groupby(month_index).apply(
            lambda values: float(np.prod(1.0 + values.to_numpy(dtype=np.float64)) - 1.0)
        )
        for period, value in month.items():
            month_rows.append({"cost_bps": int(cost), "month": str(period), "net_return": float(value)})
    attribution = {
        "asset": asset_attribution,
        "month": month_rows,
        "holding_episodes": episode_rows,
        "asset_cells": len(asset_attribution),
        "month_cells": len(month_rows),
        "episode_count": len(episode_rows),
    }
    return {"folds": fold_metrics, "aggregate": aggregate_metrics}, daily, attribution


def _bootstrap(
    predictive_daily: pd.DataFrame,
    economic_daily: pd.DataFrame,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    paths = int(contract["paths"])
    base_seed = int(contract["base_seed"])
    batch_size = int(contract["batch_size"])
    top1 = predictive_daily.groupby("date", sort=True)["top1_centered_excess"].mean().to_numpy(dtype=np.float64)
    economic = economic_daily.loc[economic_daily["cost_bps"] == int(contract["economic_cost_bps"])].groupby("date", sort=True)["net_return"].mean().to_numpy(dtype=np.float64)
    _require(len(top1) == len(economic) == 357, "V64 bootstrap series length drift")
    cells = []
    for block in contract["block_lengths_days"]:
        for kind, values, seed in (
            ("top1_centered_excess", top1, base_seed + 1000 + int(block)),
            ("economic_total_return_10bps", economic, base_seed + int(block)),
        ):
            rng = np.random.default_rng(seed)
            store = np.empty(paths, dtype=np.float64)
            cursor = 0
            while cursor < paths:
                size = min(batch_size, paths - cursor)
                indexes = circular_block_indices(len(values), int(block), size, rng)
                sampled = values[indexes]
                store[cursor : cursor + size] = (
                    sampled.mean(axis=1)
                    if kind == "top1_centered_excess"
                    else np.prod(1.0 + sampled, axis=1) - 1.0
                )
                cursor += size
            quantiles = np.quantile(store, [0.01, 0.05, 0.5, 0.95, 0.99])
            cells.append({
                "kind": kind, "block_length_days": int(block), "paths": paths,
                "seed": seed, "mean": float(store.mean()), "p01": float(quantiles[0]),
                "p05": float(quantiles[1]), "median": float(quantiles[2]),
                "p95": float(quantiles[3]), "p99": float(quantiles[4]),
            })
    return {"method": "circular_block", "cells": cells, "cell_count": len(cells)}


def _gate_matrix(
    metrics: Mapping[str, Any], bootstrap: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    rows = []
    def add(name: str, scope: str, observed: float, operator: str, threshold: float) -> None:
        passed = observed > threshold if operator == "strictly_greater_than" else observed >= threshold
        rows.append({
            "gate": name, "scope": scope, "observed": float(observed),
            "operator": operator, "threshold": float(threshold), "passed": bool(passed),
        })
    predictive = metrics["predictive"]
    economic = metrics["economic"]
    for fold, values in predictive["folds"].items():
        add("mean_spearman_strictly_positive_each_fold", f"fold_{fold}", values["spearman"], "strictly_greater_than", 0.0)
        add("mean_top1_centered_excess_strictly_positive_each_fold", f"fold_{fold}", values["top1_centered_excess"], "strictly_greater_than", 0.0)
    add("aggregate_pairwise_accuracy_strictly_above", "aggregate", predictive["aggregate"]["pairwise_accuracy"], "strictly_greater_than", float(contract["aggregate_pairwise_accuracy_strictly_above"]))
    add("aggregate_state_direction_accuracy_strictly_above", "aggregate", predictive["aggregate"]["state_direction_accuracy"], "strictly_greater_than", float(contract["aggregate_state_direction_accuracy_strictly_above"]))
    add("aggregate_absolute_direction_accuracy_strictly_above", "aggregate", predictive["aggregate"]["absolute_direction_accuracy"], "strictly_greater_than", float(contract["aggregate_absolute_direction_accuracy_strictly_above"]))
    for fold, cells in economic["folds"].items():
        add("positive_net_return_each_fold_at_10bps", f"fold_{fold}_10bps", cells["10"]["total_return"], "strictly_greater_than", 0.0)
    for cost, values in economic["aggregate"].items():
        add("aggregate_net_return_strictly_positive_all_costs", f"aggregate_{cost}bps", values["total_return"], "strictly_greater_than", 0.0)
        add("aggregate_sharpe_strictly_positive_all_costs", f"aggregate_{cost}bps", values["sharpe"], "strictly_greater_than", 0.0)
    drawdown_threshold = -float(contract["maximum_absolute_drawdown"])
    for fold, cells in economic["folds"].items():
        for cost, values in cells.items():
            add("maximum_absolute_drawdown", f"fold_{fold}_{cost}bps", values["maximum_drawdown"], "greater_than_or_equal", drawdown_threshold)
    for cost, values in economic["aggregate"].items():
        add("maximum_absolute_drawdown", f"aggregate_{cost}bps", values["maximum_drawdown"], "greater_than_or_equal", drawdown_threshold)
    for cell in bootstrap["cells"]:
        name = (
            "top1_bootstrap_p05_strictly_positive_all_blocks"
            if cell["kind"] == "top1_centered_excess"
            else "economic_bootstrap_p05_strictly_positive_all_blocks"
        )
        add(name, f"block_{cell['block_length_days']}", cell["p05"], "strictly_greater_than", 0.0)
    return {
        "gates": rows,
        "mandatory_gate_count": len(rows),
        "passed_gate_count": sum(row["passed"] for row in rows),
        "failed_gate_count": sum(not row["passed"] for row in rows),
        "all_passed": all(row["passed"] for row in rows),
    }


def _compute_core(
    context: Mapping[str, Any], outcomes: pd.DataFrame
) -> dict[str, Any]:
    predictions, assets, positions = _prepare_frames(context)
    predictive, predictive_daily = _predictive_metrics(predictions, outcomes)
    economic, economic_daily, attribution = _economic_metrics(positions, outcomes)
    spec = _load_json(context["output"] / "evaluation_spec.json", "V64 evaluation spec")
    bootstrap = _bootstrap(predictive_daily, economic_daily, spec["bootstrap"])
    metrics = {
        "schema_version": "v64-metrics/v1",
        "evidence_tier": "consumed_adaptive_development_only",
        "predictive": predictive,
        "economic": economic,
    }
    gates = _gate_matrix(metrics, bootstrap, spec["outcome_dependent_gates"])
    passed = gates["all_passed"]
    decision = (
        "authorize_future_prospective_non_target_specification_only"
        if passed
        else "retire_family_without_target_evaluation_or_retuning"
    )
    result_body = {
        "schema_version": "v64-result/v1",
        "decision": decision,
        "one_shot_decision": "pass" if passed else "retire",
        "evidence_tier": "consumed_adaptive_development_only_not_confirmation",
        "mandatory_gate_count": gates["mandatory_gate_count"],
        "passed_gate_count": gates["passed_gate_count"],
        "failed_gate_count": gates["failed_gate_count"],
        "aggregate_predictive": predictive["aggregate"],
        "aggregate_economic": economic["aggregate"],
        "unseal_count": 1,
        "source_outcome_reads": 1,
        "retuning_performed": False,
        "prediction_or_position_regeneration": False,
        "target_assets_loaded": [],
        "target_predictions": 0,
        "target_pnl_evaluations": 0,
        "deployable": False,
    }
    result = {**result_body, "result_sha256": canonical_sha256(result_body)}
    audit = {
        "schema_version": "v64-evaluation-audit/v1",
        "checks": {
            "exactly_one_outcome_unseal": True,
            "source_outcome_reads_equal_one": True,
            "immutable_prepare_predictions_and_positions_reused": True,
            "all_registered_metric_and_gate_cells_preserved": True,
            "no_retuning_or_regeneration": True,
            "target_assets_remain_sealed": True,
            "adaptive_development_evidence_label_preserved": True,
        },
        "passed": True,
        "scientific_gates_all_passed": passed,
    }
    report = "\n".join([
        "# TLM V64 Decoupled Rank/State Evaluation",
        "",
        f"**Decision:** `{decision}`",
        "",
        "This is consumed adaptive-development evidence, not clean holdout or target evidence.",
        "",
        f"- Mandatory gates: {gates['mandatory_gate_count']}",
        f"- Passed: {gates['passed_gate_count']}",
        f"- Failed: {gates['failed_gate_count']}",
        f"- Aggregate Spearman: {predictive['aggregate']['spearman']:.6f}",
        f"- Aggregate pairwise accuracy: {predictive['aggregate']['pairwise_accuracy']:.6f}",
        f"- Aggregate 10 bps return: {economic['aggregate']['10']['total_return']:.6f}",
        f"- Aggregate 10 bps Sharpe: {economic['aggregate']['10']['sharpe']:.6f}",
        f"- Aggregate 10 bps max drawdown: {economic['aggregate']['10']['maximum_drawdown']:.6f}",
        "- BTC/ETH/SOL: sealed",
        "",
    ])
    return {
        "metrics.json": metrics,
        "bootstrap.json": bootstrap,
        "gate_matrix.json": gates,
        "attribution.json": attribution,
        "evaluation_result.json": result,
        "evaluation_audit.json": audit,
        "evaluation_report.md": report,
    }


def _core_hashes(core: Mapping[str, Any]) -> dict[str, str]:
    hashes = {}
    for name, value in core.items():
        if name.endswith(".json"):
            payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
        else:
            payload = str(value)
        import hashlib
        hashes[name] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return hashes


def _write_core(output: Path, core: Mapping[str, Any]) -> None:
    for name, value in core.items():
        if name.endswith(".json"):
            write_json_atomic(output / name, value)
        else:
            _atomic_text(output / name, str(value))


def _complete_packet(
    context: Mapping[str, Any], outcome_receipt: Mapping[str, Any], completion_path: Path
) -> dict[str, Any]:
    packet = dict(context["prepare_packet"])
    packet["phase"] = "complete"
    packet["research_state"] = {
        "path": context["evaluation"]["research_state"],
        "sha256": file_sha256(context["root"] / context["evaluation"]["research_state"]),
        "authorized_phase": context["status"]["authorized_phase"],
        "authorized_next_action": context["status"]["authorized_next_action"],
        "authorized_command": context["status"]["authorized_command"],
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
    packet["replay"] = None
    return packet


def unseal_decoupled_rank_state_evaluation(config: Mapping[str, Any]) -> dict[str, Any]:
    context = _context(config)
    output = context["output"]
    lock = context["root"] / context["stage"]["runtime_contract"]["process_lock"]
    with _process_lock(lock):
        auth_path = output / AUTHORIZATION_FILE
        packet_path = output / OUTCOME_FILE
        outcome_receipt_path = output / OUTCOME_RECEIPT_FILE
        completion_path = output / "completion_receipt.json"
        if completion_path.is_file():
            result = _load_json(output / "evaluation_result.json", "V64 result")
            return {**result, "cached": True, "source_outcome_reads_this_invocation": 0}
        if auth_path.exists() and not packet_path.exists():
            raise V64EvaluationError(
                "V64 authorization exists without an atomic outcome packet; fail closed"
            )
        if packet_path.exists() != outcome_receipt_path.exists():
            raise V64EvaluationError("V64 outcome packet/receipt is incomplete; fail closed")
        predictions, assets, positions = _prepare_frames(context)
        del predictions, positions
        source_reads = 0
        if not auth_path.exists():
            write_json_atomic(auth_path, _authorization_receipt(context))
        authorization = _load_json(auth_path, "V64 authorization receipt")
        _verify_authorization(context, authorization)
        if not packet_path.exists():
            outcomes, access_receipt = _read_source_outcomes(context, assets)
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
                "source_outcome_reads": 1,
                "written_atomically": True,
                "immutable": True,
                "access_receipt": access_receipt,
                "target_assets_loaded": [],
            }
            write_json_atomic(outcome_receipt_path, outcome_receipt)
        outcomes, outcome_receipt = _validate_outcome_packet(context)
        core = _compute_core(context, outcomes)
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
            "source_outcome_reads": 1,
            "target_assets_status": "sealed",
        }
        write_json_atomic(completion_path, completion)
        manifest_files = [
            AUTHORIZATION_FILE, OUTCOME_FILE, OUTCOME_RECEIPT_FILE,
            *CORE_FILES, "completion_receipt.json",
        ]
        manifest = {
            "schema_version": "v64-artifact-manifest/v1",
            "files": {name: file_sha256(output / name) for name in manifest_files},
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        write_json_atomic(output / "artifact_manifest.json", manifest)
        complete_packet = _complete_packet(context, outcome_receipt, completion_path)
        write_json_atomic(output / COMPLETE_PACKET_FILE, complete_packet)
        return {
            **result,
            "cached": False,
            "source_outcome_reads_this_invocation": source_reads,
            "outcome_packet_sha256": file_sha256(packet_path),
            "outcome_receipt_sha256": file_sha256(outcome_receipt_path),
            "completion_receipt_sha256": file_sha256(completion_path),
        }


def replay_decoupled_rank_state_evaluation(config: Mapping[str, Any]) -> dict[str, Any]:
    context = _context(config)
    output = context["output"]
    _require((output / "completion_receipt.json").is_file(), "V64 completion is missing")
    outcomes, _ = _validate_outcome_packet(context)
    observed = {name: file_sha256(output / name) for name in CORE_FILES}
    recomputed = _compute_core(context, outcomes)
    expected = _core_hashes(recomputed)
    _require(observed == expected, "V64 replay result hashes differ")
    replay = {
        "schema_version": "v64-replay/v1",
        "reused_existing_outcome_packet": True,
        "new_unseal_receipts": 0,
        "source_outcome_rows_read": 0,
        "result_hashes_match": True,
        "new_checkpoint_loads": 0,
        "new_inference": 0,
        "new_position_generation": 0,
        "core_file_hashes": expected,
        "target_assets_loaded": [],
    }
    write_json_atomic(output / "replay.json", replay)
    complete = _load_json(output / COMPLETE_PACKET_FILE, "V64 complete packet")
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
        "cached": True,
        "files_rewritten": 0,
        "new_unseal_authorization_receipts": 0,
        "new_outcome_packets": 0,
        "source_outcome_rows_read": 0,
        "new_checkpoint_loads": 0,
        "new_inference": 0,
        "new_position_generation": 0,
        "result_hashes_match": True,
        "target_assets_loaded": [],
    }
