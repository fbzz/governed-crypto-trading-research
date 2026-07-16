from __future__ import annotations

from dataclasses import fields
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .monte_carlo import paired_block_bootstrap
from .non_target_pretraining import (
    TARGET_SYMBOLS,
    TripletTensorStore,
    _canonical_sha256,
    _sha256_file,
    _write_json,
)
from .patch_transformer import PREDICTION_HEADS
from .scientific_harness import FeatureScaler, persistent_portfolio_returns
from .supervised_non_target import (
    calibration_semantic_sha256,
    calibration_state_sha256,
    load_supervised_checkpoint,
    model_state_sha256,
)


STRATEGIES = ("candidate", "dual_momentum_30", "equal_weight_buy_hold")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _scaler_from_record(record: dict) -> FeatureScaler:
    names = {field.name for field in fields(FeatureScaler)}
    values = {name: record[name] for name in names}
    values["feature_names"] = tuple(values["feature_names"])
    values["mean"] = tuple(values["mean"])
    values["scale"] = tuple(values["scale"])
    return FeatureScaler(**values)


def build_evaluation_spec(one_shot: dict) -> dict[str, object]:
    spec = {
        "version": "v37",
        "phase": "one_shot_asset_disjoint_source_domain_confirmation",
        "expected_input_sha256": one_shot["expected_input_sha256"],
        "evaluation": one_shot["evaluation"],
        "policy": one_shot["policy"],
        "controls": one_shot["controls"],
        "accounting": one_shot["accounting"],
        "gates": one_shot["gates"],
        "device": one_shot["device"],
        "inference_batch_size": one_shot["inference_batch_size"],
        "training_allowed": False,
        "recalibration_allowed": False,
        "checkpoint_or_seed_selection_allowed": False,
        "target_assets_allowed": False,
        "repeat_evaluation_allowed_after_result": False,
    }
    spec["evaluation_spec_sha256"] = _canonical_sha256(spec)
    return spec


def _verify_protocol_contract(
    one_shot: dict,
    v26_blueprint: dict,
    amended_blueprint: dict,
    harness_spec: dict,
) -> dict[str, bool]:
    policy = one_shot["policy"]
    accounting = one_shot["accounting"]
    gates = one_shot["gates"]
    registered_policy = v26_blueprint["policy"]
    registered_gates = v26_blueprint["source_domain_gates"]
    return {
        "candidate_family_is_amended_v2": amended_blueprint[
            "candidate_family_id"
        ] == "tlm_multi_asset_target_transfer_v2",
        "policy_thresholds_match_v26": (
            policy["enter_if_q10_above"]
            == registered_policy["enter_if_q10_above"]
            and policy["enter_if_q50_above"]
            == registered_policy["enter_if_q50_above"]
            and policy["ranking_head"] == registered_policy["ranking_score"]
        ),
        "costs_match_v26_and_v34": (
            accounting["costs_bps_per_unit_turnover"]
            == registered_gates["cost_bps"]
            == harness_spec["cost_bps"]
        ),
        "controls_match_v26_and_v34": (
            one_shot["controls"]["primary"]["name"]
            == registered_gates["primary_control"]
            == harness_spec["controls"][0]
            and one_shot["controls"]["secondary"]["name"]
            == registered_gates["secondary_control"]
            == harness_spec["controls"][1]
        ),
        "bootstrap_matches_v26_and_v34": (
            gates["bootstrap_paths"] == registered_gates["bootstrap_paths"]
            == harness_spec["bootstrap"]["paths"]
            and gates["bootstrap_block_lengths_days"]
            == registered_gates["block_lengths_days"]
            == harness_spec["bootstrap"]["block_lengths_days"]
        ),
        "drawdown_gates_match_v26": (
            gates["max_drawdown_tolerance_vs_primary"]
            == registered_gates["max_drawdown_tolerance_vs_primary"]
            and gates["maximum_absolute_drawdown"]
            == registered_gates["maximum_absolute_drawdown"]
        ),
        "three_asset_folds_required": gates["minimum_asset_folds"]
        == registered_gates["minimum_asset_folds"]
        == 3,
        "failure_action_matches_v26": gates["failure_action"]
        == registered_gates["failure_action"],
        "target_boundary_matches_v34": harness_spec["target_boundary"]
        == "non_target_only_no_btc_eth_sol",
        "volatility_floor_matches_v34": one_shot["evaluation"][
            "volatility_floor"
        ] == harness_spec["losses"]["volatility_floor"],
    }


def preflight_source_domain_one_shot(config: dict) -> dict[str, object]:
    one_shot = config["source_domain_one_shot"]
    root = Path(one_shot["project_root"]).resolve()
    paths = {name: root / value for name, value in one_shot["inputs"].items()}
    input_hash_checks = {
        name: path.is_file()
        and _sha256_file(path) == one_shot["expected_input_sha256"][name]
        for name, path in paths.items()
    }
    if not all(input_hash_checks.values()):
        raise RuntimeError(f"V37 input hash drift: {input_hash_checks}")

    v26_blueprint = _load_json(paths["v26_blueprint"])
    amendment = _load_json(paths["v29_amendment"])
    dataset_manifest = _load_json(paths["v32_dataset_manifest"])
    feature_schema = _load_json(paths["v32_feature_schema"])
    triplet_catalog = _load_json(paths["v32_triplet_catalog"])
    triplet_availability = _load_json(paths["v32_triplet_availability"])
    harness_spec = _load_json(paths["v34_harness_spec"])
    v36_result = _load_json(paths["v36_result"])
    v36_audit = _load_json(paths["v36_audit"])
    checkpoint_manifest = _load_json(paths["v36_checkpoint_manifest"])
    calibrations = _load_json(paths["v36_calibration_states"])
    supervised_spec = _load_json(paths["v36_supervised_spec"])
    protocol_checks = _verify_protocol_contract(
        one_shot, v26_blueprint, amendment["blueprint"], harness_spec
    )

    checkpoint_checks = []
    for row in checkpoint_manifest:
        checkpoint_path = root / row["checkpoint_path"]
        model, payload = load_supervised_checkpoint(checkpoint_path)
        checkpoint_checks.append(
            checkpoint_path.is_file()
            and _sha256_file(checkpoint_path) == row["checkpoint_sha256"]
            and payload["metadata"]["model_state_sha256"]
            == row["model_state_sha256"]
            and model_state_sha256(model.state_dict())
            == row["model_state_sha256"]
        )
    calibration_checks = [
        row["calibration_state_sha256"] == calibration_state_sha256(row)
        and row["calibration_semantic_sha256"]
        == calibration_semantic_sha256(row)
        for row in calibrations
    ]
    folds = triplet_catalog["folds"]
    all_test_symbols = {
        symbol for fold in folds for symbol in fold["test_symbols"]
    }
    availability_rows = [
        row
        for row in triplet_availability
        if row["role"] == "test"
        and row["split"] == one_shot["evaluation"]["split"]
    ]
    structural_checks = {
        "v36_authorizes_v37": v36_result["decision"]
        == "authorize_v37_one_shot_asset_disjoint_source_test_only"
        and bool(v36_audit["passed"]),
        "v36_is_full_run": not supervised_spec["smoke"],
        "nine_checkpoints_are_present": len(checkpoint_manifest) == 9
        and all(checkpoint_checks),
        "three_calibrations_are_present": len(calibrations) == 3
        and all(calibration_checks),
        "three_disjoint_test_folds_cover_thirty_assets": len(folds) == 3
        and len(all_test_symbols) == 30
        and sum(len(fold["test_symbols"]) for fold in folds) == 30,
        "target_assets_are_absent": not TARGET_SYMBOLS.intersection(
            dataset_manifest["symbols"]
        )
        and not TARGET_SYMBOLS.intersection(all_test_symbols),
        "feature_order_is_frozen": feature_schema["model_feature_order"][:-1]
        == dataset_manifest["panel_features"],
        "availability_metadata_has_exact_confirmation_window": len(
            availability_rows
        ) == 3
        and all(
            row["eligible_dates"]
            == one_shot["evaluation"]["required_signal_dates"]
            and row["first_eligible_date"]
            == one_shot["evaluation"]["signal_start"]
            and row["last_eligible_date"]
            == one_shot["evaluation"]["signal_end"]
            for row in availability_rows
        ),
        "protocol_contract_matches_registered_gates": all(
            protocol_checks.values()
        ),
        "preflight_does_not_load_panel_or_labels": True,
        "training_and_recalibration_are_disabled": True,
    }
    if not all(structural_checks.values()):
        raise RuntimeError(f"V37 preflight failed: {structural_checks}")
    return {
        "version": "v37_preflight",
        "decision": "authorize_v37_one_shot_execution",
        "evaluation_spec": build_evaluation_spec(one_shot),
        "input_hash_checks": input_hash_checks,
        "protocol_checks": protocol_checks,
        "structural_checks": structural_checks,
        "loaded": {
            "checkpoint_metadata": True,
            "checkpoint_weights_for_integrity_verification": True,
            "calibration_metadata": True,
            "panel": False,
            "panel_bytes_hashed_but_values_not_parsed": True,
            "held_out_label_values": False,
            "model_predictions": False,
            "target_assets": False,
        },
        "paths": {name: str(path) for name, path in paths.items()},
        "checkpoint_manifest": checkpoint_manifest,
        "calibrations": calibrations,
        "triplet_catalog": triplet_catalog,
        "dataset_manifest": dataset_manifest,
        "feature_schema": feature_schema,
        "amended_blueprint": amendment["blueprint"],
    }


def build_policy_positions(
    q10: np.ndarray,
    q50: np.ndarray,
    momentum: np.ndarray,
    availability: np.ndarray,
    q10_threshold: float,
    q50_threshold: float,
) -> dict[str, np.ndarray]:
    expected = q10.shape
    if (
        q50.shape != expected
        or momentum.shape != expected
        or availability.shape != expected
        or q10.ndim != 2
    ):
        raise ValueError("V37 policy arrays must share [days, assets] shape")
    if availability.dtype != bool:
        availability = availability.astype(bool)
    if (availability.sum(axis=1) < 1).any():
        raise ValueError("Every V37 date must have an eligible asset")
    rows = np.arange(len(q50))

    candidate_scores = np.where(availability, q50, -np.inf)
    candidate_best = np.argmax(candidate_scores, axis=1)
    candidate_active = (
        q50[rows, candidate_best] > q50_threshold
    ) & (q10[rows, candidate_best] > q10_threshold)
    candidate = np.zeros_like(q50, dtype=np.float64)
    candidate[rows[candidate_active], candidate_best[candidate_active]] = 1.0

    momentum_scores = np.where(availability, momentum, -np.inf)
    momentum_best = np.argmax(momentum_scores, axis=1)
    momentum_active = momentum[rows, momentum_best] > 0.0
    dual = np.zeros_like(momentum, dtype=np.float64)
    dual[rows[momentum_active], momentum_best[momentum_active]] = 1.0

    equal_weight = availability.astype(np.float64)
    equal_weight /= equal_weight.sum(axis=1, keepdims=True)
    return {
        "candidate": candidate,
        "dual_momentum_30": dual,
        "equal_weight_buy_hold": equal_weight,
    }


def performance_metrics(
    net_return: np.ndarray,
    turnover: np.ndarray,
    cost: np.ndarray,
    annualization_days: int = 365,
) -> dict[str, float | int]:
    net = np.asarray(net_return, dtype=np.float64)
    if net.ndim != 1 or len(net) < 2 or not np.isfinite(net).all():
        raise ValueError("V37 metrics require finite one-dimensional daily returns")
    if (net <= -1.0).any():
        raise ValueError("V37 daily return reached total loss")
    equity = np.cumprod(1.0 + net)
    years = len(net) / float(annualization_days)
    standard_deviation = float(net.std(ddof=1))
    sharpe = (
        math.sqrt(float(annualization_days)) * float(net.mean())
        / standard_deviation
        if standard_deviation > 0
        else 0.0
    )
    peak = np.maximum.accumulate(np.maximum(equity, 1.0))
    drawdown = equity / peak - 1.0
    return {
        "observations": int(len(net)),
        "total_return": float(equity[-1] - 1.0),
        "cagr": float(equity[-1] ** (1.0 / years) - 1.0)
        if equity[-1] > 0
        else -1.0,
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "total_turnover": float(np.asarray(turnover).sum()),
        "total_cost": float(np.asarray(cost).sum()),
        "positive_day_rate": float(np.mean(net > 0)),
    }


def evaluate_registered_gates(
    aggregate_metrics: dict[str, dict[str, dict[str, float]]],
    bootstrap: dict[str, dict[str, object]],
    gates: dict,
) -> dict[str, object]:
    primary = "dual_momentum_30"
    controls = list(gates["total_return_above_controls"])
    cost_cells = []
    for cost, metrics in sorted(aggregate_metrics.items(), key=lambda item: int(item[0])):
        candidate = metrics["candidate"]
        cell_checks = {
            **{
                f"total_return_above_{control}": candidate["total_return"]
                > metrics[control]["total_return"]
                for control in controls
            },
            "sharpe_above_primary": candidate["sharpe"]
            > metrics[primary]["sharpe"],
            "drawdown_within_primary_tolerance": candidate["max_drawdown"]
            >= metrics[primary]["max_drawdown"]
            - float(gates["max_drawdown_tolerance_vs_primary"]),
            "absolute_drawdown_within_limit": candidate["max_drawdown"]
            >= -float(gates["maximum_absolute_drawdown"]),
        }
        cost_cells.append({
            "cost_bps": int(cost),
            "checks": {name: bool(value) for name, value in cell_checks.items()},
            "passed": bool(all(cell_checks.values())),
        })
    bootstrap_cells = []
    for block, result in sorted(bootstrap.items(), key=lambda item: int(item[0])):
        for control in gates[
            "paired_total_return_delta_p05_above_zero_for_controls"
        ]:
            p05 = result["comparisons"][control][
                "paired_total_return_delta"
            ]["p05"]
            bootstrap_cells.append({
                "block_length_days": int(block),
                "control": control,
                "paired_total_return_delta_p05": float(p05),
                "passed": bool(p05 > 0.0),
            })
    passed = all(row["passed"] for row in cost_cells) and all(
        row["passed"] for row in bootstrap_cells
    )
    return {
        "passed": bool(passed),
        "cost_cells": cost_cells,
        "bootstrap_cells": bootstrap_cells,
        "failure_action": gates["failure_action"],
        "pass_action": gates["pass_action"],
    }


def _fold_inference(
    fold: dict,
    panel: pd.DataFrame,
    feature_names: list[str],
    models: list,
    scaler: FeatureScaler,
    calibration: dict,
    evaluation: dict,
    policy: dict,
    batch_size: int,
    device: torch.device,
) -> dict[str, object]:
    fold_number = int(fold["fold"])
    symbols = list(fold["test_symbols"])
    symbol_to_index = {symbol: index for index, symbol in enumerate(symbols)}
    dates = pd.date_range(
        evaluation["signal_start"], evaluation["signal_end"], freq="D", tz="UTC"
    )
    date_to_index = {date: index for index, date in enumerate(dates)}
    eligible = panel.loc[
        panel["symbol"].isin(symbols)
        & panel["in_one_shot_non_target_confirmation"]
        & panel[evaluation["eligibility_rule"]]
        & panel["date"].between(dates[0], dates[-1]),
        ["date", "symbol"],
    ]
    available_by_date = {
        pd.Timestamp(date): sorted(frame["symbol"].tolist())
        for date, frame in eligible.groupby("date", sort=True)
    }
    if set(available_by_date) != set(dates):
        raise RuntimeError(f"V37 fold {fold_number} date coverage drift")
    frozen_triplets = {tuple(row) for row in fold["test_triplets"]}
    samples = []
    availability = np.zeros((len(dates), len(symbols)), dtype=bool)
    for date in dates:
        current = available_by_date[date]
        availability[date_to_index[date], [symbol_to_index[s] for s in current]] = True
        for triplet in sorted(
            group for group in frozen_triplets if set(group).issubset(current)
        ):
            samples.append({"date": date, "triplet": triplet})

    store = TripletTensorStore(
        panel[["date", "symbol", *feature_names]],
        feature_names,
        lookback_days=256,
        relative_source_feature="log_close_to_close_return",
    )
    sums = {
        name: np.zeros((len(dates), len(symbols)), dtype=np.float64)
        for name in PREDICTION_HEADS
    }
    context_counts = np.zeros((len(dates), len(symbols)), dtype=np.int64)
    for model in models:
        model.eval()
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        x = torch.from_numpy(store.materialize_batch(batch, scaler)).to(device)
        with torch.no_grad():
            outputs = [model(x) for model in models]
        date_indexes = np.asarray(
            [date_to_index[pd.Timestamp(row["date"])] for row in batch],
            dtype=np.int64,
        )
        asset_indexes = np.asarray([
            [symbol_to_index[symbol] for symbol in row["triplet"]]
            for row in batch
        ], dtype=np.int64)
        flat_dates = np.repeat(date_indexes, 3)
        flat_assets = asset_indexes.reshape(-1)
        np.add.at(context_counts, (flat_dates, flat_assets), 1)
        for name in PREDICTION_HEADS:
            mean_output = torch.stack([output[name] for output in outputs]).mean(0)
            np.add.at(
                sums[name],
                (flat_dates, flat_assets),
                mean_output.detach().cpu().numpy().reshape(-1),
            )
    expected_counts = np.zeros_like(context_counts)
    for day, count in enumerate(availability.sum(axis=1)):
        expected_counts[day, availability[day]] = math.comb(int(count) - 1, 2)
    if not np.array_equal(context_counts, expected_counts):
        raise RuntimeError(f"V37 fold {fold_number} context aggregation drift")
    raw = {
        name: np.divide(
            values,
            context_counts,
            out=np.full_like(values, np.nan),
            where=context_counts > 0,
        )
        for name, values in sums.items()
    }

    return_label = "target_next_open_to_next_open_log_return"
    volatility_label = "target_realized_volatility_7d"
    actual = np.full((len(dates), len(symbols)), np.nan, dtype=np.float64)
    observed_volatility = np.full_like(actual, np.nan)
    momentum = np.full_like(actual, np.nan)
    for symbol in symbols:
        frame = panel.loc[panel["symbol"] == symbol].sort_values("date").copy()
        frame["momentum_30"] = frame["log_close_to_close_return"].rolling(
            30, min_periods=30
        ).sum()
        current = frame.loc[frame["date"].isin(dates)].set_index("date")
        asset_index = symbol_to_index[symbol]
        for date in dates[availability[:, asset_index]]:
            row = current.loc[date]
            actual[date_to_index[date], asset_index] = float(row[return_label])
            observed_volatility[date_to_index[date], asset_index] = float(
                row[volatility_label]
            )
            momentum[date_to_index[date], asset_index] = float(row["momentum_30"])
    if (
        not np.isfinite(actual[availability]).all()
        or not np.isfinite(observed_volatility[availability]).all()
        or not np.isfinite(momentum[availability]).all()
    ):
        raise RuntimeError(f"V37 fold {fold_number} eligible values are non-finite")

    offsets = calibration["offsets"]
    calibrated_quantiles = np.sort(
        np.stack([
            raw["return_q10"] + float(offsets["return_q10"]),
            raw["return_q50"] + float(offsets["return_q50"]),
            raw["return_q90"] + float(offsets["return_q90"]),
        ], axis=-1),
        axis=-1,
        kind="stable",
    )
    calibrated_log_volatility = raw["volatility_7d"] + float(
        offsets["log_volatility"]
    )
    q10, q50, q90 = (
        calibrated_quantiles[..., index] for index in range(3)
    )
    positions = build_policy_positions(
        q10,
        q50,
        momentum,
        availability,
        float(policy["enter_if_q10_above"]),
        float(policy["enter_if_q50_above"]),
    )
    observed_return = actual[availability]
    pinball = {}
    for name, quantile, predicted in (
        ("q10", 0.10, q10[availability]),
        ("q50", 0.50, q50[availability]),
        ("q90", 0.90, q90[availability]),
    ):
        error = observed_return - predicted
        pinball[name] = float(np.mean(np.maximum(
            quantile * error, (quantile - 1.0) * error
        )))
    diagnostics = {
        "asset_date_prediction_count": int(availability.sum()),
        "triplet_prediction_count": len(samples),
        "available_assets_min": int(availability.sum(axis=1).min()),
        "available_assets_max": int(availability.sum(axis=1).max()),
        "context_count_min": int(context_counts[availability].min()),
        "context_count_max": int(context_counts[availability].max()),
        "q10_coverage": float(np.mean(observed_return <= q10[availability])),
        "q50_coverage": float(np.mean(observed_return <= q50[availability])),
        "q90_coverage": float(np.mean(observed_return <= q90[availability])),
        "pinball_q10": pinball["q10"],
        "pinball_q50": pinball["q50"],
        "pinball_q90": pinball["q90"],
        "mean_pinball": float(np.mean(list(pinball.values()))),
        "q50_mae": float(np.mean(np.abs(observed_return - q50[availability]))),
        "log_volatility_mae": float(np.mean(np.abs(
            np.log(np.maximum(
                observed_volatility[availability],
                float(evaluation["volatility_floor"]),
            ))
            - calibrated_log_volatility[availability]
        ))),
        "calibrated_quantile_crossing_rate": float(np.mean(
            (q10[availability] > q50[availability])
            | (q50[availability] > q90[availability])
        )),
    }
    prediction_rows = []
    for day, date in enumerate(dates):
        for asset_index, symbol in enumerate(symbols):
            if not availability[day, asset_index]:
                continue
            prediction_rows.append({
                "date": date,
                "fold": fold_number,
                "symbol": symbol,
                "context_count": int(context_counts[day, asset_index]),
                "raw_q10": float(raw["return_q10"][day, asset_index]),
                "raw_q50": float(raw["return_q50"][day, asset_index]),
                "raw_q90": float(raw["return_q90"][day, asset_index]),
                "calibrated_q10": float(q10[day, asset_index]),
                "calibrated_q50": float(q50[day, asset_index]),
                "calibrated_q90": float(q90[day, asset_index]),
                "calibrated_log_volatility": float(
                    calibrated_log_volatility[day, asset_index]
                ),
                "observed_log_return": float(actual[day, asset_index]),
                "observed_volatility_7d": float(
                    observed_volatility[day, asset_index]
                ),
                "momentum_30": float(momentum[day, asset_index]),
                "candidate_weight": float(positions["candidate"][day, asset_index]),
                "dual_momentum_weight": float(
                    positions["dual_momentum_30"][day, asset_index]
                ),
                "equal_weight_control": float(
                    positions["equal_weight_buy_hold"][day, asset_index]
                ),
            })
    return {
        "fold": fold_number,
        "dates": dates,
        "symbols": symbols,
        "actual": np.nan_to_num(actual, nan=0.0),
        "availability": availability,
        "positions": positions,
        "diagnostics": diagnostics,
        "prediction_rows": prediction_rows,
        "checkpoint_sha256": [
            calibration_hash for calibration_hash in calibration[
                "member_checkpoint_sha256"
            ]
        ],
        "calibration_semantic_sha256": calibration[
            "calibration_semantic_sha256"
        ],
    }


def _portfolio_outputs(
    folds: list[dict[str, object]],
    costs: list[int],
    annualization_days: int,
) -> tuple[dict, dict, dict, list[dict[str, object]]]:
    fold_metrics: dict[str, dict] = {}
    aggregate_metrics: dict[str, dict] = {}
    aggregate_curves: dict[str, dict] = {}
    daily_rows: list[dict[str, object]] = []
    for cost_bps in costs:
        cost_key = str(cost_bps)
        fold_metrics[cost_key] = {}
        curves_by_strategy = {name: [] for name in STRATEGIES}
        for fold in folds:
            fold_key = str(fold["fold"])
            fold_metrics[cost_key][fold_key] = {}
            for strategy in STRATEGIES:
                curve = persistent_portfolio_returns(
                    fold["positions"][strategy], fold["actual"], cost_bps
                )
                metrics = performance_metrics(
                    curve["net_return"], curve["turnover"], curve["cost"],
                    annualization_days,
                )
                fold_metrics[cost_key][fold_key][strategy] = metrics
                curves_by_strategy[strategy].append(curve)
                equity = np.cumprod(1.0 + np.asarray(curve["net_return"]))
                for index, date in enumerate(fold["dates"]):
                    daily_rows.append({
                        "date": date,
                        "cost_bps": cost_bps,
                        "scope": f"fold_{fold_key}",
                        "strategy": strategy,
                        "gross_return": float(curve["gross_return"][index]),
                        "turnover": float(curve["turnover"][index]),
                        "cost": float(curve["cost"][index]),
                        "net_return": float(curve["net_return"][index]),
                        "equity": float(equity[index]),
                    })
        aggregate_metrics[cost_key] = {}
        aggregate_curves[cost_key] = {}
        dates = folds[0]["dates"]
        for strategy in STRATEGIES:
            aggregate = {
                name: np.mean(
                    np.stack([np.asarray(curve[name]) for curve in curves_by_strategy[strategy]]),
                    axis=0,
                )
                for name in ("gross_return", "turnover", "cost", "net_return")
            }
            aggregate_curves[cost_key][strategy] = aggregate
            aggregate_metrics[cost_key][strategy] = performance_metrics(
                aggregate["net_return"],
                aggregate["turnover"],
                aggregate["cost"],
                annualization_days,
            )
            equity = np.cumprod(1.0 + aggregate["net_return"])
            for index, date in enumerate(dates):
                daily_rows.append({
                    "date": date,
                    "cost_bps": cost_bps,
                    "scope": "aggregate_equal_fold_capital",
                    "strategy": strategy,
                    "gross_return": float(aggregate["gross_return"][index]),
                    "turnover": float(aggregate["turnover"][index]),
                    "cost": float(aggregate["cost"][index]),
                    "net_return": float(aggregate["net_return"][index]),
                    "equity": float(equity[index]),
                })
    return fold_metrics, aggregate_metrics, aggregate_curves, daily_rows


def _report(result: dict[str, object]) -> str:
    gate = result["gate_result"]
    base = result["aggregate_metrics"]["10"]
    candidate = base["candidate"]
    primary = base["dual_momentum_30"]
    secondary = base["equal_weight_buy_hold"]
    status = (
        "SOURCE-DOMAIN GATE PASSED."
        if gate["passed"]
        else "SOURCE-DOMAIN GATE FAILED; CANDIDATE FAMILY IS RETIRED."
    )
    return "\n".join([
        "# TLM v37 One-Shot Source-Domain Evaluation",
        "",
        "## Decision",
        "",
        f"**{status}**",
        "",
        f"Decision: `{result['decision']}`",
        f"Evaluation-spec SHA-256: `{result['evaluation_spec']['evaluation_spec_sha256']}`",
        f"Signal dates: **{result['summary']['signal_dates']}**",
        f"Asset folds: **{result['summary']['fold_count']}**",
        "",
        "## Base-cost aggregate (10 bps)",
        "",
        "| Strategy | Total return | Sharpe | Max drawdown |",
        "|---|---:|---:|---:|",
        f"| Candidate | {candidate['total_return']:.2%} | {candidate['sharpe']:.3f} | {candidate['max_drawdown']:.2%} |",
        f"| Dual momentum 30 | {primary['total_return']:.2%} | {primary['sharpe']:.3f} | {primary['max_drawdown']:.2%} |",
        f"| Equal weight | {secondary['total_return']:.2%} | {secondary['sharpe']:.3f} | {secondary['max_drawdown']:.2%} |",
        "",
        "All 10/20/30 bps and 7/21/63-day bootstrap cells are preserved in the machine artifacts. No seed, fold, checkpoint, threshold, or failed cell was discarded.",
        "",
        "BTC, ETH, and SOL remained sealed. This result does not authorize target evaluation, paper trading, shadow trading, or execution.",
        "",
    ])


def run_source_domain_one_shot(
    config: dict,
    preflight_only: bool = False,
) -> dict[str, object]:
    preflight = preflight_source_domain_one_shot(config)
    if preflight_only:
        return preflight
    one_shot = config["source_domain_one_shot"]
    root = Path(one_shot["project_root"]).resolve()
    output = root / config["output_dir"]
    result_path = output / "result.json"
    if result_path.is_file():
        cached = _load_json(result_path)
        if cached["evaluation_spec"]["evaluation_spec_sha256"] != preflight[
            "evaluation_spec"
        ]["evaluation_spec_sha256"]:
            raise RuntimeError("V37 cached result uses a different evaluation spec")
        return cached

    torch.set_num_threads(int(one_shot["torch_threads"]))
    torch.use_deterministic_algorithms(True)
    device_name = one_shot["device"]
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("V37 requires MPS outside the sandbox")
    device = torch.device(device_name)
    paths = {name: Path(path) for name, path in preflight["paths"].items()}
    panel_columns = [
        "date",
        "symbol",
        "target_window_end_date",
        "supervised_sequence_ready",
        "in_one_shot_non_target_confirmation",
        *preflight["dataset_manifest"]["panel_features"],
        *preflight["dataset_manifest"]["labels"],
    ]
    test_symbols = sorted({
        symbol
        for fold in preflight["triplet_catalog"]["folds"]
        for symbol in fold["test_symbols"]
    })
    panel = pd.read_parquet(
        paths["panel"],
        columns=panel_columns,
        filters=[("symbol", "in", test_symbols)],
    )
    if TARGET_SYMBOLS.intersection(panel["symbol"].unique()):
        raise RuntimeError("Target asset entered v37 panel")
    evaluation = one_shot["evaluation"]
    eligible_maturities = pd.to_datetime(
        panel.loc[
            panel["in_one_shot_non_target_confirmation"]
            & panel[evaluation["eligibility_rule"]]
            & panel["date"].between(
                pd.Timestamp(evaluation["signal_start"], tz="UTC"),
                pd.Timestamp(evaluation["signal_end"], tz="UTC"),
            ),
            "target_window_end_date",
        ],
        utc=True,
    )
    maturity_boundary = pd.Timestamp(
        evaluation["maturity_boundary"], tz="UTC"
    )
    if eligible_maturities.empty or eligible_maturities.max() != maturity_boundary:
        raise RuntimeError("V37 held-out label maturity boundary drift")

    manifest_by_fold = {
        fold: sorted(
            [row for row in preflight["checkpoint_manifest"] if row["fold"] == fold],
            key=lambda row: int(row["seed"]),
        )
        for fold in (1, 2, 3)
    }
    calibration_by_fold = {
        int(row["fold"]): row for row in preflight["calibrations"]
    }
    fold_outputs = []
    for fold in preflight["triplet_catalog"]["folds"]:
        fold_number = int(fold["fold"])
        jobs = manifest_by_fold[fold_number]
        models = []
        for job in jobs:
            model, _ = load_supervised_checkpoint(root / job["checkpoint_path"])
            model.to(device)
            models.append(model)
        scaler_records = [
            _load_json(
                root
                / one_shot["v35_checkpoint_dir"]
                / f"fold_{fold_number}"
                / f"seed_{job['seed']}"
                / "scaler.json"
            )
            for job in jobs
        ]
        scaler = _scaler_from_record(scaler_records[0])
        if any(
            _scaler_from_record(row).state_sha256() != scaler.state_sha256()
            for row in scaler_records
        ):
            raise RuntimeError(f"V37 scaler drift in fold {fold_number}")
        fold_outputs.append(_fold_inference(
            fold,
            panel,
            preflight["dataset_manifest"]["panel_features"],
            models,
            scaler,
            calibration_by_fold[fold_number],
            one_shot["evaluation"],
            one_shot["policy"],
            int(one_shot["inference_batch_size"]),
            device,
        ))
        for model in models:
            model.to("cpu")
        if device.type == "mps":
            torch.mps.empty_cache()

    costs = list(one_shot["accounting"]["costs_bps_per_unit_turnover"])
    fold_metrics, aggregate_metrics, aggregate_curves, daily_rows = _portfolio_outputs(
        fold_outputs,
        costs,
        int(one_shot["accounting"]["annualization_days"]),
    )
    base_cost = str(one_shot["gates"]["bootstrap_cost_bps"])
    bootstrap = {
        str(block): paired_block_bootstrap(
            {
                strategy: np.asarray(
                    aggregate_curves[base_cost][strategy]["net_return"]
                )
                for strategy in STRATEGIES
            },
            "candidate",
            ["dual_momentum_30", "equal_weight_buy_hold"],
            block_length=int(block),
            n_paths=int(one_shot["gates"]["bootstrap_paths"]),
            seed=int(config["seed"]) + int(block),
        )
        for block in one_shot["gates"]["bootstrap_block_lengths_days"]
    }
    gate_result = evaluate_registered_gates(
        aggregate_metrics, bootstrap, one_shot["gates"]
    )
    decision = (
        one_shot["gates"]["pass_action"]
        if gate_result["passed"]
        else one_shot["gates"]["failure_action"]
    )
    prediction_frame = pd.DataFrame([
        row for fold in fold_outputs for row in fold["prediction_rows"]
    ]).sort_values(["date", "fold", "symbol"]).reset_index(drop=True)
    daily_frame = pd.DataFrame(daily_rows).sort_values(
        ["date", "cost_bps", "scope", "strategy"]
    ).reset_index(drop=True)
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "predictions.parquet"
    daily_path = output / "daily_returns.parquet"
    prediction_frame.to_parquet(prediction_path, index=False)
    daily_frame.to_parquet(daily_path, index=False)
    diagnostics = [
        {"fold": fold["fold"], **fold["diagnostics"]}
        for fold in fold_outputs
    ]
    audit_checks = {
        "preflight_passed_before_label_load": all(
            preflight["structural_checks"].values()
        ),
        "exactly_three_asset_folds_evaluated": len(fold_outputs) == 3,
        "exactly_173_signal_dates_per_fold": all(
            len(fold["dates"]) == 173 for fold in fold_outputs
        ),
        "all_held_out_labels_mature_by_boundary": bool(
            (eligible_maturities <= maturity_boundary).all()
            and eligible_maturities.max() == maturity_boundary
        ),
        "all_three_seed_members_used_per_fold": all(
            len(manifest_by_fold[int(fold["fold"])]) == 3
            for fold in fold_outputs
        ),
        "all_eligible_triplet_contexts_aggregated": all(
            fold["diagnostics"]["triplet_prediction_count"] > 0
            and fold["diagnostics"]["context_count_min"] > 0
            for fold in fold_outputs
        ),
        "calibrated_quantiles_are_monotone": all(
            fold["diagnostics"]["calibrated_quantile_crossing_rate"] == 0.0
            for fold in fold_outputs
        ),
        "all_costs_and_strategies_are_present": set(aggregate_metrics)
        == {str(value) for value in costs}
        and all(set(cell) == set(STRATEGIES) for cell in aggregate_metrics.values()),
        "bootstrap_paths_and_blocks_are_exact": all(
            result["paths"] == one_shot["gates"]["bootstrap_paths"]
            and result["block_length"] == int(block)
            for block, result in bootstrap.items()
        ),
        "registered_gate_cells_are_all_preserved": len(
            gate_result["cost_cells"]
        ) == 3 and len(gate_result["bootstrap_cells"]) == 6,
        "checkpoint_files_remain_unchanged": all(
            _sha256_file(root / row["checkpoint_path"])
            == row["checkpoint_sha256"]
            for row in preflight["checkpoint_manifest"]
        ),
        "no_training_recalibration_or_selection": True,
        "target_assets_never_loaded": not TARGET_SYMBOLS.intersection(
            panel["symbol"].unique()
        ),
        "one_shot_result_will_be_cached": True,
    }
    if not all(audit_checks.values()):
        raise RuntimeError(f"V37 audit failed: {audit_checks}")
    result = {
        "version": "v37",
        "decision": decision,
        "evaluation_execution_count": 1,
        "evaluation_spec": preflight["evaluation_spec"],
        "summary": {
            "fold_count": len(fold_outputs),
            "signal_dates": len(fold_outputs[0]["dates"]),
            "asset_date_predictions": int(sum(
                row["asset_date_prediction_count"] for row in diagnostics
            )),
            "triplet_predictions": int(sum(
                row["triplet_prediction_count"] for row in diagnostics
            )),
            "bootstrap_paths_per_block": one_shot["gates"]["bootstrap_paths"],
            "device": device_name,
            "gate_passed": gate_result["passed"],
        },
        "prediction_diagnostics": diagnostics,
        "fold_metrics": fold_metrics,
        "aggregate_metrics": aggregate_metrics,
        "bootstrap": bootstrap,
        "gate_result": gate_result,
        "artifacts": {
            "predictions_parquet_sha256": _sha256_file(prediction_path),
            "daily_returns_parquet_sha256": _sha256_file(daily_path),
        },
        "tested": {
            "held_out_non_target_labels_loaded": True,
            "held_out_non_target_predictions_computed": True,
            "source_domain_performance_computed": True,
            "training_or_recalibration_executed": False,
            "checkpoint_seed_or_fold_selected": False,
            "target_assets_loaded": False,
            "target_domain_performance_computed": False,
        },
        "audit": {"passed": True, "checks": audit_checks},
    }
    _write_json(output / "evaluation_spec.json", preflight["evaluation_spec"])
    _write_json(output / "prediction_diagnostics.json", diagnostics)
    _write_json(output / "metrics.json", {
        "fold_metrics": fold_metrics,
        "aggregate_metrics": aggregate_metrics,
    })
    _write_json(output / "bootstrap.json", bootstrap)
    _write_json(output / "gate_result.json", gate_result)
    _write_json(output / "audit.json", result["audit"])
    _write_json(output / "evaluation_receipt.json", {
        "evaluation_execution_count": 1,
        "evaluation_spec_sha256": preflight["evaluation_spec"][
            "evaluation_spec_sha256"
        ],
        "input_sha256": one_shot["expected_input_sha256"],
        "output_sha256": result["artifacts"],
        "target_assets_loaded": False,
        "retraining_or_recalibration_executed": False,
    })
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    _write_json(result_path, result)
    return result
