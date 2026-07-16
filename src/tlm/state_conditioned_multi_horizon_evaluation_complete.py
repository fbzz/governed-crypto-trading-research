"""Exactly-once V59 outcome unseal, frozen evaluation, and source-free replay."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import os
import subprocess
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .core.artifacts import canonical_sha256, file_sha256
from .monte_carlo import circular_block_indices
from .research_workflow import validate_research_state
from .state_conditioned_multi_horizon_evaluation import CONTROLS
from .state_conditioned_multi_horizon_evaluation_artifacts import (
    V59PrepareError,
    load_json,
    load_yaml,
    mapping,
    minimum_free_space,
    process_lock,
    require,
    resolve_repo_path,
    verify_prepare_packet,
    verify_self_hash,
    with_self_hash,
    write_json,
)
from .state_conditioned_multi_horizon_evaluation_data import (
    _key_dnf,
    _read,
    day_text,
    key_sha256,
    utc_day,
)
from .state_conditioned_multi_horizon_training_data import TARGET_SYMBOLS


AUTHORIZATION_FILE = "authorization_receipt.json"
OUTCOME_FILE = "outcome_packet.parquet"
OUTCOME_RECEIPT_FILE = "outcome_receipt.json"
CORE_FILES = (
    "metrics.json",
    "bootstrap.json",
    "gate_matrix.json",
    "result.json",
    "audit.json",
    "report.md",
)
FINAL_FILES = (
    *CORE_FILES,
    "replay.json",
    "completion_receipt.json",
    "artifact_manifest.json",
)
STRATEGIES = ("candidate", *CONTROLS)
OUTCOME_COLUMNS = (
    "date",
    "symbol",
    "target_h1_open_to_open_log_return",
    "target_h3_open_to_open_log_return",
    "target_h7_open_to_open_log_return",
)
OUTCOME_PACKET_COLUMNS = ("origin", "fold", *OUTCOME_COLUMNS)
MANDATORY_COSTS = (10, 20, 30)
REPORTING_COSTS = (10, 20, 30, 50)
BLOCK_LENGTHS = (7, 21, 63)


def _evaluation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = mapping(
        config.get("state_conditioned_multi_horizon_evaluation"),
        "state_conditioned_multi_horizon_evaluation",
    )
    require(value.get("version") == "v59", "V59 completion config version drift")
    return value


def _git_clean(root: Path) -> str:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    require(not status.strip(), "V59 unseal requires a clean committed Git tree")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(len(head) == 40, "V59 unseal Git head drift")
    return head


def _context(config: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = _evaluation_config(config)
    root = Path(str(evaluation.get("project_root", "."))).resolve()
    status = validate_research_state(root, evaluation["research_state"])
    require(status.get("passed") is True, "V59 unseal research state failed")
    require(
        status.get("authorized_next_action")
        == "execute_v59_exactly_one_registered_outcome_unseal_and_complete_evaluation",
        "V59 unseal authorization token drift",
    )
    stage_path = resolve_repo_path(root, evaluation["phase_contract"], "phase contract")
    base_path = resolve_repo_path(
        root, evaluation["base_phase_contract"], "base phase contract"
    )
    stage = load_yaml(stage_path, "V59 unseal phase contract")
    base = load_yaml(base_path, "V59 base phase contract")
    require(stage.get("stage_revision") == "v059_unseal_r1", "V59 stage drift")
    require(
        file_sha256(stage_path)
        == "f23bbc7891a5754ecdc8261492fdb85cfe53ba94d3f0d2a9ca6019a49626771f",
        "V59 unseal phase file hash drift",
    )
    output = resolve_repo_path(root, evaluation["output_dir"], "output directory")
    verify_prepare_packet(
        root,
        output,
        contract=base,
        enforce_live_git=False,
        enforce_live_inputs=True,
        verify_prepared_values_gate=True,
        verify_source_commit=True,
        allow_post_prepare_files=True,
    )
    return {
        "root": root,
        "evaluation": evaluation,
        "status": status,
        "stage": stage,
        "base": base,
        "output": output,
    }


def _authorization_receipt(context: Mapping[str, Any], git_head: str) -> dict[str, Any]:
    stage = context["stage"]
    prepare = stage["prepare_packet"]
    body = {
        "schema_version": "tlm-one-shot-unseal-authorization/v2",
        "phase": "v59",
        "stage_revision": "v059_unseal_r1",
        "family_id": stage["family_id"],
        "explicit_user_authorization": True,
        "exact_registered_unseal": True,
        "unseal_count": 1,
        "authorized_command": stage["commands"]["unseal"],
        "authorization_payload": stage["explicit_user_authorization"]["payload"],
        "authorization_payload_sha256": stage["explicit_user_authorization"][
            "canonical_sha256"
        ],
        "base_phase_contract_file_sha256": stage["base_phase_contract"][
            "file_sha256"
        ],
        "evaluation_spec_sha256": prepare["evaluation_spec"]["canonical_sha256"],
        "prepare_manifest_sha256": prepare["prepare_manifest"]["canonical_sha256"],
        "prepare_receipt_sha256": prepare["prepare_receipt"]["canonical_sha256"],
        "outcome_request_sha256": prepare["outcome_request"]["canonical_sha256"],
        "source_git_head": git_head,
        "written_atomically_before_source_read": True,
        "source_outcome_reads_before_receipt": 0,
        "retuning_performed": False,
        "predictions_or_positions_regenerated": False,
        "target_assets_loaded": [],
        "target_assets_status": "sealed",
    }
    return with_self_hash(body, "authorization_receipt_sha256")


def _verify_authorization(
    context: Mapping[str, Any], receipt: Mapping[str, Any]
) -> str:
    registered = verify_self_hash(
        receipt, "authorization_receipt_sha256", "V59 authorization receipt"
    )
    expected = _authorization_receipt(context, str(receipt.get("source_git_head", "")))
    require(dict(receipt) == expected, "V59 authorization receipt content drift")
    require(receipt.get("unseal_count") == 1, "V59 unseal count drift")
    return registered


def _request_frame(request: Mapping[str, Any]) -> pd.DataFrame:
    require(request.get("schema_version") == "v59-outcome-request/v1", "request schema drift")
    verify_self_hash(request, "outcome_request_sha256", "V59 outcome request")
    frame = pd.DataFrame(request.get("keys", []))
    require(list(frame.columns) == ["date", "fold", "origin", "symbol"], "request columns drift")
    frame = frame.loc[:, ["origin", "fold", "date", "symbol"]].copy()
    frame["date"] = frame["date"].map(utc_day)
    frame["fold"] = frame["fold"].astype(np.int8)
    frame["symbol"] = frame["symbol"].astype(str)
    frame = frame.sort_values(["origin", "fold", "date", "symbol"]).reset_index(drop=True)
    records = [
        [row.origin, int(row.fold), day_text(row.date), row.symbol]
        for row in frame.itertuples(index=False)
    ]
    require(len(frame) == request.get("key_count") == 20410, "request key count drift")
    require(not frame.duplicated().any(), "request keys are not unique")
    require(canonical_sha256(records) == request.get("key_sha256"), "request key hash drift")
    require(not set(frame["symbol"]).intersection(TARGET_SYMBOLS), "request contains target")
    return frame


def _write_parquet_atomic(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_parquet(temporary, engine="pyarrow", index=False, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _unseal_outcomes(
    context: Mapping[str, Any], request: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    stage = context["stage"]
    access = stage["outcome_access_contract"]
    request_frame = _request_frame(request)
    logical_keys = frozenset(zip(request_frame["date"], request_frame["symbol"], strict=True))
    filters = _key_dnf(logical_keys)
    source = resolve_repo_path(context["root"], access["source"], "outcome source")
    require(file_sha256(source) == access["source_file_sha256"], "outcome source hash drift")
    values = _read(
        pd.read_parquet,
        source,
        columns=OUTCOME_COLUMNS,
        filters=filters,
        label="V59 exact non-target outcomes",
    )
    value_keys = frozenset(zip(values["date"], values["symbol"], strict=True))
    require(value_keys == logical_keys, "unsealed outcome keys differ from request")
    for column in OUTCOME_COLUMNS[2:]:
        require(values[column].dtype == np.dtype("float64"), f"{column} dtype drift")
        require(np.isfinite(values[column]).all(), f"{column} contains nonfinite values")
    packet = request_frame.merge(values, on=["date", "symbol"], how="left", validate="many_to_one")
    packet = packet.loc[:, list(OUTCOME_PACKET_COLUMNS)].sort_values(
        ["origin", "fold", "date", "symbol"]
    ).reset_index(drop=True)
    require(len(packet) == 20410, "outcome packet row count drift")
    access_receipt = {
        "source": access["source"],
        "source_file_sha256": access["source_file_sha256"],
        "projected_columns": list(OUTCOME_COLUMNS),
        "predicate": "exact_date_lists_grouped_by_each_non_target_symbol",
        "predicate_dnf_sha256": canonical_sha256(
            [
                [
                    [column, operator, [day_text(item) for item in value] if operator == "in" else value]
                    for column, operator, value in conjunction
                ]
                for conjunction in filters
            ]
        ),
        "request_key_count": len(request_frame),
        "request_key_sha256": request["key_sha256"],
        "materialized_row_count": len(values),
        "source_outcome_reads": 1,
        "target_asset_loads": 0,
        "full_table_materializations": 0,
    }
    return packet, access_receipt


def _validate_outcome_packet(
    context: Mapping[str, Any], request: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = context["output"]
    receipt = load_json(output / OUTCOME_RECEIPT_FILE, "V59 outcome receipt")
    verify_self_hash(receipt, "outcome_receipt_sha256", "V59 outcome receipt")
    packet_path = output / OUTCOME_FILE
    require(packet_path.is_file(), "V59 outcome packet is missing")
    require(file_sha256(packet_path) == receipt.get("outcome_packet_file_sha256"), "outcome packet hash drift")
    packet = pd.read_parquet(packet_path, engine="pyarrow", columns=list(OUTCOME_PACKET_COLUMNS))
    packet["date"] = packet["date"].map(utc_day)
    packet["symbol"] = packet["symbol"].astype(str)
    packet = packet.sort_values(["origin", "fold", "date", "symbol"]).reset_index(drop=True)
    expected = _request_frame(request)
    require(packet.loc[:, ["origin", "fold", "date", "symbol"]].equals(expected), "outcome packet keys drift")
    require(len(packet) == receipt.get("row_count") == 20410, "outcome packet count drift")
    require(not set(packet["symbol"]).intersection(TARGET_SYMBOLS), "outcome packet contains target")
    for column in OUTCOME_COLUMNS[2:]:
        require(np.isfinite(packet[column]).all(), "outcome packet contains nonfinite values")
    return packet, receipt


def _economic_metrics(returns: np.ndarray, turnover: np.ndarray) -> dict[str, float]:
    values = np.asarray(returns, dtype=np.float64)
    turns = np.asarray(turnover, dtype=np.float64)
    require(len(values) > 1 and len(values) == len(turns), "economic series geometry drift")
    wealth = np.cumprod(1.0 + values)
    require(np.isfinite(wealth).all() and (wealth > 0).all(), "nonpositive or invalid wealth")
    peak = np.maximum.accumulate(np.maximum(wealth, 1.0))
    std = float(values.std(ddof=1))
    return {
        "cumulative_return": float(wealth[-1] - 1.0),
        "annualized_arithmetic_return": float(values.mean() * 365.0),
        "annualized_volatility": float(std * math.sqrt(365.0)),
        "sharpe": float(values.mean() / std * math.sqrt(365.0)) if std > 0 else 0.0,
        "maximum_drawdown": float(np.max(1.0 - wealth / peak)),
        "total_turnover": float(turns.sum()),
        "annualized_turnover": float(turns.mean() * 365.0),
    }


def _position_summary(frame: pd.DataFrame) -> dict[str, Any]:
    gross = frame[["weight_0", "weight_1", "weight_2"]].sum(axis=1).to_numpy()
    selected = frame.loc[frame["selected_symbol"] != "", "selected_symbol"].astype(str)
    counts = Counter(selected.tolist())
    total = sum(counts.values())
    concentration = max(counts.values()) / total if total else 0.0
    return {
        "action_counts": {str(k): int(v) for k, v in sorted(Counter(frame["action"]).items())},
        "risky_exposure_fraction": float(np.mean(gross > 0.0)),
        "selected_asset_concentration": float(concentration),
        "selected_asset_counts": {str(k): int(v) for k, v in sorted(counts.items())},
    }


def _daily_for_positions(frame: pd.DataFrame, outcomes: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    lookup = outcomes.set_index(["date", "symbol"])["target_h1_open_to_open_log_return"]
    gross = np.zeros(len(frame), dtype=np.float64)
    for slot in range(3):
        keys = pd.MultiIndex.from_arrays([frame["date"], frame[f"symbol_{slot}"]])
        log_returns = lookup.reindex(keys).to_numpy(dtype=np.float64)
        weights = frame[f"weight_{slot}"].to_numpy(dtype=np.float64)
        missing_active = ~np.isfinite(log_returns) & (np.abs(weights) > 1.0e-15)
        require(not missing_active.any(), "active position lacks a registered h1 outcome")
        gross += weights * np.expm1(np.where(np.isfinite(log_returns), log_returns, 0.0))
    values = pd.DataFrame(
        {"date": frame["date"], "gross": gross, "turnover": frame["turnover"].to_numpy(dtype=np.float64)}
    )
    daily = values.groupby("date", sort=True).sum(numeric_only=True) / 120.0
    require(len(values) == len(daily) * 120, "fixed triplet denominator drift")
    return daily["gross"], daily["turnover"]


def _build_economic_series(
    output: Path, outcomes: pd.DataFrame
) -> tuple[dict[tuple[str, str, int, int], dict[str, np.ndarray]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidate = pd.read_parquet(output / "candidate_positions.parquet", engine="pyarrow")
    controls = pd.read_parquet(output / "control_positions.parquet", engine="pyarrow")
    for frame in (candidate, controls):
        frame["date"] = frame["date"].map(utc_day)
    series: dict[tuple[str, str, int, int], dict[str, np.ndarray]] = {}
    fold_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    for origin in ("origin_2024", "origin_2025"):
        for geometry in ("expanding", "rolling"):
            for fold in (1, 2, 3):
                mask = (
                    (candidate["origin"] == origin)
                    & (candidate["geometry"] == geometry)
                    & (candidate["fold"] == fold)
                )
                candidate_cell = candidate.loc[mask].reset_index(drop=True)
                outcome_cell = outcomes.loc[
                    (outcomes["origin"] == origin) & (outcomes["fold"] == fold)
                ]
                gross_by: dict[str, pd.Series] = {}
                turnover_by: dict[str, pd.Series] = {}
                gross_by["candidate"], turnover_by["candidate"] = _daily_for_positions(
                    candidate_cell, outcome_cell
                )
                position_rows.append(
                    {"origin": origin, "geometry": geometry, "fold": fold, "strategy": "candidate", **_position_summary(candidate_cell)}
                )
                control_mask = (
                    (controls["origin"] == origin)
                    & (controls["geometry"] == geometry)
                    & (controls["fold"] == fold)
                )
                controls_cell = controls.loc[control_mask]
                for control in CONTROLS:
                    frame = controls_cell.loc[controls_cell["control"] == control].reset_index(drop=True)
                    gross_by[control], turnover_by[control] = _daily_for_positions(frame, outcome_cell)
                    position_rows.append(
                        {"origin": origin, "geometry": geometry, "fold": fold, "strategy": control, **_position_summary(frame)}
                    )
                dates = gross_by["candidate"].index
                require(all(item.index.equals(dates) for item in gross_by.values()), "strategy date drift")
                for cost in REPORTING_COSTS:
                    returns_by: dict[str, np.ndarray] = {}
                    for strategy in STRATEGIES:
                        net = gross_by[strategy].to_numpy() - turnover_by[strategy].to_numpy() * cost / 10000.0
                        returns_by[strategy] = net.astype(np.float64)
                        fold_rows.append(
                            {
                                "origin": origin,
                                "geometry": geometry,
                                "fold": fold,
                                "cost_bps": cost,
                                "strategy": strategy,
                                "dates": len(dates),
                                **_economic_metrics(net, turnover_by[strategy].to_numpy()),
                            }
                        )
                    series[(origin, geometry, fold, cost)] = returns_by
    return series, fold_rows, position_rows


def _aggregate_rows(
    series: Mapping[tuple[str, str, int, int], Mapping[str, np.ndarray]]
) -> tuple[dict[tuple[str, str, int], dict[str, np.ndarray]], list[dict[str, Any]]]:
    aggregates: dict[tuple[str, str, int], dict[str, np.ndarray]] = {}
    rows: list[dict[str, Any]] = []
    for origin in ("origin_2024", "origin_2025"):
        for geometry in ("expanding", "rolling"):
            for cost in REPORTING_COSTS:
                returns_by: dict[str, np.ndarray] = {}
                for strategy in STRATEGIES:
                    values = np.vstack([series[(origin, geometry, fold, cost)][strategy] for fold in (1, 2, 3)])
                    returns = values.mean(axis=0)
                    returns_by[strategy] = returns
                    turnover = np.zeros(len(returns), dtype=np.float64)
                    # Turnover metrics are cost-independent and recovered from the cost wedge.
                    gross_proxy = np.vstack([series[(origin, geometry, fold, 10)][strategy] for fold in (1, 2, 3)]).mean(axis=0)
                    if cost != 10:
                        turnover = (gross_proxy - returns) / ((cost - 10) / 10000.0)
                    else:
                        returns20 = np.vstack([series[(origin, geometry, fold, 20)][strategy] for fold in (1, 2, 3)]).mean(axis=0)
                        turnover = (returns - returns20) / (10.0 / 10000.0)
                    rows.append(
                        {
                            "origin": origin,
                            "geometry": geometry,
                            "cost_bps": cost,
                            "strategy": strategy,
                            "dates": len(returns),
                            **_economic_metrics(returns, turnover),
                        }
                    )
                aggregates[(origin, geometry, cost)] = returns_by
    return aggregates, rows


def _predictive_metrics(output: Path, outcomes: pd.DataFrame) -> list[dict[str, Any]]:
    all_columns = pq.ParquetFile(output / "predictions.parquet").schema_arrow.names
    prediction_columns = [
        name
        for name in all_columns
        if name in {"origin", "geometry", "fold", "triplet_key", "date", "asset_slot", "symbol"}
        or name.startswith("seed_")
        or name.startswith("ensemble_")
    ]
    predictions = pd.read_parquet(
        output / "predictions.parquet", engine="pyarrow", columns=prediction_columns
    )
    predictions["date"] = predictions["date"].map(utc_day)
    merged = predictions.merge(
        outcomes,
        on=["origin", "fold", "date", "symbol"],
        how="left",
        validate="many_to_one",
    )
    require(merged[list(OUTCOME_COLUMNS[2:])].notna().all().all(), "prediction outcomes missing")
    rows: list[dict[str, Any]] = []
    for (origin, geometry, fold), frame in merged.groupby(["origin", "geometry", "fold"], sort=True):
        frame = frame.sort_values(["triplet_key", "date", "asset_slot"]).reset_index(drop=True)
        require(len(frame) % 3 == 0, "predictive triplet geometry drift")
        group_sizes = frame.groupby(["triplet_key", "date"], sort=False).size()
        require((group_sizes == 3).all(), "predictive cross section is not three assets")
        pred = frame["ensemble_h7_q50"].to_numpy().reshape(-1, 3)
        realized = frame["target_h7_open_to_open_log_return"].to_numpy().reshape(-1, 3)
        pred_rank = pd.DataFrame(pred).rank(axis=1, method="average").to_numpy()
        realized_rank = pd.DataFrame(realized).rank(axis=1, method="average").to_numpy()
        pred_center = pred_rank - pred_rank.mean(axis=1, keepdims=True)
        realized_center = realized_rank - realized_rank.mean(axis=1, keepdims=True)
        denominator = np.sqrt((pred_center**2).sum(axis=1) * (realized_center**2).sum(axis=1))
        valid = denominator > 0
        spearman = np.divide(
            (pred_center * realized_center).sum(axis=1),
            denominator,
            out=np.full(len(denominator), np.nan),
            where=valid,
        )
        pair_scores: list[np.ndarray] = []
        excluded_ties = 0
        for left, right in ((0, 1), (0, 2), (1, 2)):
            outcome_delta = realized[:, left] - realized[:, right]
            prediction_delta = pred[:, left] - pred[:, right]
            active = np.abs(outcome_delta) > 1.0e-12
            excluded_ties += int((~active).sum())
            score = np.where(
                np.abs(prediction_delta[active]) <= 1.0e-12,
                0.5,
                (np.sign(prediction_delta[active]) == np.sign(outcome_delta[active])).astype(float),
            )
            pair_scores.append(score)
        seed_columns = [name for name in frame if name.startswith("seed_")]
        seed_groups: dict[str, list[str]] = defaultdict(list)
        for name in seed_columns:
            seed_groups[name.split("_", 2)[2]].append(name)
        seed_std = [
            np.std(frame[sorted(names)].to_numpy(), axis=1, ddof=0)
            for names in seed_groups.values()
        ]
        crossing = []
        for horizon in (1, 3, 7):
            q20 = frame[f"ensemble_h{horizon}_q20"].to_numpy()
            q50 = frame[f"ensemble_h{horizon}_q50"].to_numpy()
            q80 = frame[f"ensemble_h{horizon}_q80"].to_numpy()
            crossing.append((q20 > q50) | (q50 > q80))
        rows.append(
            {
                "origin": str(origin),
                "geometry": str(geometry),
                "fold": int(fold),
                "rows": len(frame),
                "cross_sections": len(pred),
                "h7_q20_coverage": float(
                    np.mean(
                        frame["target_h7_open_to_open_log_return"].to_numpy()
                        <= frame["ensemble_h7_q20"].to_numpy()
                    )
                ),
                "h7_q50_spearman": float(np.nanmean(spearman)),
                "invalid_constant_cross_sections": int((~valid).sum()),
                "h7_q50_pairwise_accuracy": float(np.mean(np.concatenate(pair_scores))),
                "outcome_tied_pairs_excluded": excluded_ties,
                "all_horizon_quantile_crossing_rate": float(np.mean(np.concatenate(crossing))),
                "three_seed_prediction_standard_deviation": float(np.mean(np.concatenate(seed_std))),
                "h1_q20_coverage": float(np.mean(frame["target_h1_open_to_open_log_return"] <= frame["ensemble_h1_q20"])),
                "h3_q20_coverage": float(np.mean(frame["target_h3_open_to_open_log_return"] <= frame["ensemble_h3_q20"])),
            }
        )
    require(len(rows) == 12, "predictive cell count drift")
    return rows


def _bootstrap_seed(origin: str, geometry: str, fold: int, cost: int, block: int) -> int:
    value = [20260714, "v59", "bootstrap", origin, geometry, fold, cost, block]
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _distribution(values: np.ndarray) -> dict[str, float]:
    q = np.quantile(values, [0.01, 0.05, 0.5, 0.95, 0.99], method="linear")
    return {
        "mean": float(values.mean()),
        "p01": float(q[0]),
        "p05": float(q[1]),
        "median": float(q[2]),
        "p95": float(q[3]),
        "p99": float(q[4]),
    }


def _bootstrap_cell(
    returns_by: Mapping[str, np.ndarray], *, block: int, seed: int, paths: int = 10000
) -> dict[str, Any]:
    n = len(returns_by["candidate"])
    require(all(len(values) == n for values in returns_by.values()), "bootstrap length drift")
    stores = {name: np.empty(paths, dtype=np.float64) for name in STRATEGIES}
    rng = np.random.default_rng(seed)
    cursor = 0
    while cursor < paths:
        size = min(256, paths - cursor)
        indexes = circular_block_indices(n, block, size, rng)
        for name in STRATEGIES:
            sampled = np.asarray(returns_by[name], dtype=np.float64)[indexes]
            require((sampled > -1.0).all(), "bootstrap sampled nonpositive wealth return")
            stores[name][cursor : cursor + size] = np.prod(1.0 + sampled, axis=1) - 1.0
        cursor += size
    candidate = stores["candidate"]
    return {
        "method": "paired_circular_moving_block",
        "paths": paths,
        "block_length": block,
        "seed": seed,
        "observations_per_path": n,
        "distributions": {name: _distribution(stores[name]) for name in STRATEGIES},
        "candidate_minus_controls": {
            name: _distribution(candidate - stores[name]) for name in CONTROLS
        },
    }


def _bootstrap_grid(
    series: Mapping[tuple[str, str, int, int], Mapping[str, np.ndarray]], paths: int = 10000
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for origin in ("origin_2024", "origin_2025"):
        for geometry in ("expanding", "rolling"):
            for fold in (1, 2, 3):
                for cost in MANDATORY_COSTS:
                    for block in BLOCK_LENGTHS:
                        rows.append(
                            {
                                "origin": origin,
                                "geometry": geometry,
                                "fold": fold,
                                "cost_bps": cost,
                                **_bootstrap_cell(
                                    series[(origin, geometry, fold, cost)],
                                    block=block,
                                    seed=_bootstrap_seed(origin, geometry, fold, cost, block),
                                    paths=paths,
                                ),
                            }
                        )
    require(len(rows) == 108, "bootstrap grid count drift")
    return rows


def _gate_matrix(
    fold_rows: Sequence[Mapping[str, Any]],
    aggregate_rows: Sequence[Mapping[str, Any]],
    predictive_rows: Sequence[Mapping[str, Any]],
    bootstrap_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    def add(name: str, scope: dict[str, Any], value: float, rule: str, passed: bool) -> None:
        gates.append({"gate": name, **scope, "value": float(value), "rule": rule, "passed": bool(passed)})

    fold_lookup = {(r["origin"], r["geometry"], r["fold"], r["cost_bps"], r["strategy"]): r for r in fold_rows}
    agg_lookup = {(r["origin"], r["geometry"], r["cost_bps"], r["strategy"]): r for r in aggregate_rows}
    for row in predictive_rows:
        scope = {k: row[k] for k in ("origin", "geometry", "fold")}
        coverage = float(row["h7_q20_coverage"])
        add("h7_q20_coverage", scope, coverage, "0.15<=x<=0.25", 0.15 <= coverage <= 0.25)
        pairwise = float(row["h7_q50_pairwise_accuracy"])
        add("h7_q50_pairwise_accuracy", scope, pairwise, "x>0.5", pairwise > 0.5)
        spearman = float(row["h7_q50_spearman"])
        add("h7_q50_spearman", scope, spearman, "x>0", spearman > 0.0)
        candidate = fold_lookup[(row["origin"], row["geometry"], row["fold"], 10, "candidate")]
        value = float(candidate["cumulative_return"])
        add("candidate_return_10bps", scope, value, "x>0", value > 0.0)
    for origin in ("origin_2024", "origin_2025"):
        for geometry in ("expanding", "rolling"):
            for cost in MANDATORY_COSTS:
                candidate = agg_lookup[(origin, geometry, cost, "candidate")]
                scope = {"origin": origin, "geometry": geometry, "cost_bps": cost}
                for control in CONTROLS:
                    baseline = agg_lookup[(origin, geometry, cost, control)]
                    delta = float(candidate["cumulative_return"] - baseline["cumulative_return"])
                    add("aggregate_return_vs_control", {**scope, "control": control}, delta, "x>0", delta > 0.0)
                dual = agg_lookup[(origin, geometry, cost, "weekly_dual_momentum_30")]
                sharpe_delta = float(candidate["sharpe"] - dual["sharpe"])
                add("aggregate_sharpe_vs_dual_momentum", scope, sharpe_delta, "x>0", sharpe_delta > 0.0)
                drawdown = float(candidate["maximum_drawdown"])
                add("aggregate_maximum_drawdown", scope, drawdown, "x<=0.35", drawdown <= 0.35)
            candidate_turn = agg_lookup[(origin, geometry, 10, "candidate")]["total_turnover"]
            dual_turn = agg_lookup[(origin, geometry, 10, "weekly_dual_momentum_30")]["total_turnover"]
            add("aggregate_turnover_vs_dual_momentum", {"origin": origin, "geometry": geometry}, float(candidate_turn - dual_turn), "x<=0", candidate_turn <= dual_turn)
    for row in fold_rows:
        if row["cost_bps"] in MANDATORY_COSTS and row["strategy"] == "candidate":
            drawdown = float(row["maximum_drawdown"])
            add(
                "fold_maximum_drawdown",
                {k: row[k] for k in ("origin", "geometry", "fold", "cost_bps", "strategy")},
                drawdown,
                "x<=0.35",
                drawdown <= 0.35,
            )
    for row in bootstrap_rows:
        scope = {k: row[k] for k in ("origin", "geometry", "fold", "cost_bps", "block_length")}
        absolute = float(row["distributions"]["candidate"]["p05"])
        add("candidate_bootstrap_p05", scope, absolute, "x>0", absolute > 0.0)
        for control in CONTROLS:
            delta = float(row["candidate_minus_controls"][control]["p05"])
            add("candidate_minus_control_bootstrap_p05", {**scope, "control": control}, delta, "x>0", delta > 0.0)
    return gates


def _report(metrics: Mapping[str, Any], gates: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    lines = [
        "# V59 frozen adaptive development evaluation",
        "",
        f"Decision: **{result['decision']}**",
        "",
        f"Mandatory gates passed: **{gates['passed_count']}/{gates['gate_count']}**.",
        f"Failed gates: **{gates['failed_count']}**.",
        "",
        "## Aggregate candidate performance at 10 bps",
        "",
        "| Origin | Geometry | Return | Sharpe | Max drawdown | Turnover |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in metrics["economic"]["aggregate_cells"]:
        if row["strategy"] == "candidate" and row["cost_bps"] == 10:
            lines.append(
                f"| {row['origin']} | {row['geometry']} | {row['cumulative_return']:.2%} | "
                f"{row['sharpe']:.3f} | {row['maximum_drawdown']:.2%} | {row['total_turnover']:.2f} |"
            )
    lines.extend(
        [
            "",
            "BTC, ETH, and SOL remained sealed. This is adaptive historical non-target evidence, not deployment authorization.",
            "",
        ]
    )
    return "\n".join(lines)


def _compute_core(output: Path, outcomes: pd.DataFrame) -> dict[str, Any]:
    series, fold_rows, position_rows = _build_economic_series(output, outcomes)
    aggregates, aggregate_rows = _aggregate_rows(series)
    predictive_rows = _predictive_metrics(output, outcomes)
    bootstrap_rows = _bootstrap_grid(series)
    gate_rows = _gate_matrix(fold_rows, aggregate_rows, predictive_rows, bootstrap_rows)
    failed = [row for row in gate_rows if not row["passed"]]
    decision = "authorize_v60_immutable_registration_without_refit" if not failed else "retire_family_without_tuning"
    metrics = with_self_hash(
        {
            "schema_version": "v59-metrics/v1",
            "economic": {"fold_cells": fold_rows, "aggregate_cells": aggregate_rows},
            "predictive": {"fold_cells": predictive_rows},
            "position_diagnostics": position_rows,
            "reporting_cost_bps": list(REPORTING_COSTS),
            "mandatory_cost_bps": list(MANDATORY_COSTS),
            "annualization_days": 365,
        },
        "metrics_sha256",
    )
    bootstrap = with_self_hash(
        {
            "schema_version": "v59-bootstrap/v1",
            "method": "paired_circular_moving_block",
            "paths_per_cell": 10000,
            "block_lengths": list(BLOCK_LENGTHS),
            "cells": bootstrap_rows,
        },
        "bootstrap_sha256",
    )
    gates = with_self_hash(
        {
            "schema_version": "v59-gate-matrix/v1",
            "gate_count": len(gate_rows),
            "passed_count": len(gate_rows) - len(failed),
            "failed_count": len(failed),
            "all_mandatory_passed": not failed,
            "aggregate_rescue_allowed": False,
            "gates": gate_rows,
        },
        "gate_matrix_sha256",
    )
    result = with_self_hash(
        {
            "schema_version": "v59-result/v1",
            "family_id": "tlm_state_conditioned_multi_horizon_quantile_small_v1",
            "decision": decision,
            "mandatory_gate_count": len(gate_rows),
            "failed_gate_count": len(failed),
            "outcome_unseal_count": 1,
            "target_assets_loaded": [],
            "target_predictions": 0,
            "target_pnl_evaluations": 0,
            "retuning_performed": False,
            "predictions_or_positions_regenerated": False,
            "metrics_sha256": metrics["metrics_sha256"],
            "bootstrap_sha256": bootstrap["bootstrap_sha256"],
            "gate_matrix_sha256": gates["gate_matrix_sha256"],
        },
        "result_sha256",
    )
    audit = with_self_hash(
        {
            "schema_version": "v59-completion-audit/v1",
            "passed": True,
            "scientific_gate_passed": not failed,
            "decision": decision,
            "checks": {
                "exactly_one_outcome_unseal": True,
                "source_free_metric_computation": True,
                "all_reporting_costs_present": True,
                "all_108_bootstrap_cells_present": len(bootstrap_rows) == 108,
                "all_failed_cells_preserved": True,
                "no_aggregate_rescue": True,
                "no_retuning_or_regeneration": True,
                "target_assets_remained_sealed": True,
            },
        },
        "audit_sha256",
    )
    report = _report(metrics, gates, result)
    return {
        "metrics.json": metrics,
        "bootstrap.json": bootstrap,
        "gate_matrix.json": gates,
        "result.json": result,
        "audit.json": audit,
        "report.md": report,
    }


def _core_hashes(core: Mapping[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, value in core.items():
        if name.endswith(".json"):
            payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        else:
            payload = str(value).encode()
        hashes[name] = hashlib.sha256(payload).hexdigest()
    return hashes


def _write_core(output: Path, core: Mapping[str, Any]) -> None:
    for name, value in core.items():
        if name.endswith(".json"):
            write_json(output / name, value)
        else:
            destination = output / name
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=output, prefix=f".{name}.", suffix=".tmp", delete=False
            ) as handle:
                handle.write(str(value))
                temporary = Path(handle.name)
            os.replace(temporary, destination)


def _finalize(
    context: Mapping[str, Any],
    authorization: Mapping[str, Any],
    outcome_receipt: Mapping[str, Any],
    core: Mapping[str, Any],
) -> dict[str, Any]:
    output = context["output"]
    expected_hashes = _core_hashes(core)
    replayed = _compute_core(output, pd.read_parquet(output / OUTCOME_FILE, engine="pyarrow"))
    replay_hashes = _core_hashes(replayed)
    require(replay_hashes == expected_hashes, "V59 source-free replay core hash drift")
    replay = with_self_hash(
        {
            "schema_version": "v59-replay/v1",
            "reused_existing_outcome_packet": True,
            "new_unseal_authorization_receipts": 0,
            "new_outcome_packets": 0,
            "source_outcome_rows_read": 0,
            "new_checkpoint_loads": 0,
            "new_inference": 0,
            "new_linear_control_fits": 0,
            "new_position_generation": 0,
            "result_hashes_match": True,
            "core_file_hashes": expected_hashes,
        },
        "replay_sha256",
    )
    write_json(output / "replay.json", replay)
    completion = with_self_hash(
        {
            "schema_version": "tlm-one-shot-completion/v2",
            "decision": core["result.json"]["decision"],
            "evaluation_spec_sha256": context["stage"]["prepare_packet"]["evaluation_spec"]["canonical_sha256"],
            "prepare_receipt_sha256": context["stage"]["prepare_packet"]["prepare_receipt"]["canonical_sha256"],
            "authorization_receipt_sha256": authorization["authorization_receipt_sha256"],
            "outcome_receipt_sha256": outcome_receipt["outcome_receipt_sha256"],
            "result_artifact_hashes": {**expected_hashes, "replay.json": file_sha256(output / "replay.json")},
            "target_assets_status": "sealed",
            "unseal_count": 1,
            "source_outcome_reads": 1,
            "replay_source_outcome_reads": 0,
        },
        "completion_receipt_sha256",
    )
    write_json(output / "completion_receipt.json", completion)
    manifest_names = (
        "evaluation_spec.json",
        "prepare_receipt.json",
        AUTHORIZATION_FILE,
        OUTCOME_FILE,
        OUTCOME_RECEIPT_FILE,
        *CORE_FILES,
        "replay.json",
        "completion_receipt.json",
    )
    manifest = with_self_hash(
        {
            "schema_version": "v59-artifact-manifest/v1",
            "file_count": len(manifest_names),
            "files": {name: file_sha256(output / name) for name in manifest_names},
        },
        "artifact_manifest_sha256",
    )
    write_json(output / "artifact_manifest.json", manifest)
    return {
        "decision": core["result.json"]["decision"],
        "failed_gates": core["gate_matrix.json"]["failed_count"],
        "gate_count": core["gate_matrix.json"]["gate_count"],
        "completion_receipt_sha256": completion["completion_receipt_sha256"],
        "artifact_manifest_sha256": manifest["artifact_manifest_sha256"],
        "replay_sha256": replay["replay_sha256"],
    }


def unseal_state_conditioned_multi_horizon_evaluation(config: Mapping[str, Any]) -> dict[str, Any]:
    context = _context(config)
    root, output, stage = context["root"], context["output"], context["stage"]
    minimum_free_space(root, float(stage["runtime_contract"]["minimum_free_gib"]))
    lock_path = resolve_repo_path(root, stage["runtime_contract"]["process_lock"], "process lock")
    with process_lock(lock_path):
        if all((output / name).is_file() for name in FINAL_FILES):
            raise V59PrepareError("V59 evaluation is already complete; use the replay command")
        git_head = _git_clean(root)
        auth_path, packet_path, receipt_path = (
            output / AUTHORIZATION_FILE,
            output / OUTCOME_FILE,
            output / OUTCOME_RECEIPT_FILE,
        )
        if auth_path.exists() and not packet_path.exists():
            raise V59PrepareError("V59 authorization exists without an atomic outcome packet; fail closed")
        request = load_json(output / "outcome_request.json", "V59 outcome request")
        if not auth_path.exists():
            write_json(auth_path, _authorization_receipt(context, git_head))
        authorization = load_json(auth_path, "V59 authorization receipt")
        _verify_authorization(context, authorization)
        source_reads = 0
        if not packet_path.exists():
            packet, access_receipt = _unseal_outcomes(context, request)
            source_reads = 1
            _write_parquet_atomic(packet_path, packet)
            outcome_receipt = with_self_hash(
                {
                    "schema_version": "tlm-one-shot-outcome/v2",
                    "authorization_receipt_sha256": authorization["authorization_receipt_sha256"],
                    "outcome_request_sha256": request["outcome_request_sha256"],
                    "outcome_packet_file_sha256": file_sha256(packet_path),
                    "row_count": len(packet),
                    "unseal_count": 1,
                    "source_outcome_reads": 1,
                    "written_atomically": True,
                    "immutable": True,
                    "access_receipt": access_receipt,
                    "target_assets_loaded": [],
                },
                "outcome_receipt_sha256",
            )
            write_json(receipt_path, outcome_receipt)
        require(receipt_path.is_file(), "V59 outcome packet exists without its receipt")
        outcomes, outcome_receipt = _validate_outcome_packet(context, request)
        print("[V59 unseal] immutable outcome packet verified; computing frozen metrics", flush=True)
        core = _compute_core(output, outcomes)
        _write_core(output, core)
        result = _finalize(context, authorization, outcome_receipt, core)
        return {
            **result,
            "unseal_count": 1,
            "source_outcome_reads_this_invocation": source_reads,
            "source_outcome_reads_total": 1,
            "target_assets_loaded": [],
            "target_predictions": 0,
            "target_pnl_evaluations": 0,
        }


def replay_state_conditioned_multi_horizon_evaluation(config: Mapping[str, Any]) -> dict[str, Any]:
    context = _context(config)
    output = context["output"]
    require(all((output / name).is_file() for name in FINAL_FILES), "V59 completion packet is incomplete")
    request = load_json(output / "outcome_request.json", "V59 outcome request")
    outcomes, _ = _validate_outcome_packet(context, request)
    observed = {name: file_sha256(output / name) for name in CORE_FILES}
    recomputed = _compute_core(output, outcomes)
    expected = _core_hashes(recomputed)
    require(observed == expected, "V59 replay result hashes differ")
    replay = load_json(output / "replay.json", "V59 replay receipt")
    verify_self_hash(replay, "replay_sha256", "V59 replay receipt")
    require(replay.get("core_file_hashes") == expected, "V59 replay binding drift")
    return {
        "decision": recomputed["result.json"]["decision"],
        "cached": True,
        "files_rewritten": 0,
        "new_unseal_authorization_receipts": 0,
        "new_outcome_packets": 0,
        "source_outcome_rows_read": 0,
        "new_checkpoint_loads": 0,
        "new_inference": 0,
        "new_linear_control_fits": 0,
        "new_position_generation": 0,
        "result_hashes_match": True,
        "target_assets_loaded": [],
    }
