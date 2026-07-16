from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import platform
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow
import torch
import yaml

from .ranking_excess_screen_metrics import (
    STRATEGY_WEIGHT_COLUMNS,
    average_rank_spearman,
    build_portfolio_evaluation,
    compute_predictive_metrics,
)
from .ranking_excess_spec import _canonical_sha256, _load_json, _sha256_file


VERSION = "v46"
TARGET_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
JSON_INPUTS = {
    "v45_result",
    "v45_gate_result",
    "v45_audit",
    "v45_artifact_manifest",
    "v45_completion_receipt",
    "v45_unseal_receipt",
    "v45_outcome_receipt",
    "v45_prepare_result",
    "v45_prepare_artifact_manifest",
    "v45_prepare_completion_receipt",
    "v45_prepare_data_audit",
    "v45_prepare_data_access_receipt",
}
YAML_INPUTS = {"v45_resolved_config"}
TABLE_INPUTS = {
    "context_predictions",
    "asset_predictions",
    "positions",
    "outcomes",
    "predictive_context_metrics",
    "predictive_daily_metrics",
    "daily_returns",
}
IMPLEMENTATION_SOURCE_FILES = (
    "monte_carlo.py",
    "ranking_excess_failure_autopsy.py",
    "ranking_excess_screen_metrics.py",
    "ranking_excess_spec.py",
    "scientific_harness.py",
    "source_domain_one_shot.py",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    _atomic_write_text(
        path,
        json.dumps(_json_ready(value), indent=2, sort_keys=True, allow_nan=False),
    )


def _write_yaml_atomic(path: Path, value: object) -> None:
    _atomic_write_text(path, yaml.safe_dump(value, sort_keys=False))


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    frame.to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def _implementation_provenance() -> dict[str, object]:
    package_root = Path(__file__).resolve().parent
    source_hashes = {}
    for relative in IMPLEMENTATION_SOURCE_FILES:
        path = package_root / relative
        if not path.is_file():
            raise RuntimeError(f"V46 implementation source is missing: {relative}")
        source_hashes[f"src/tlm/{relative}"] = _sha256_file(path)
    return {
        "source_sha256": source_hashes,
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "torch": str(torch.__version__),
            "pyyaml": yaml.__version__,
        },
    }


def build_autopsy_spec(config: dict) -> dict[str, object]:
    autopsy = config["ranking_excess_failure_autopsy"]
    spec: dict[str, object] = {
        "version": VERSION,
        "phase": "ranking_excess_failure_autopsy_read_only",
        "resolved_config_semantic_sha256": _canonical_sha256(config),
        "expected_input_sha256": autopsy["expected_input_sha256"],
        "expected_lineage": autopsy["expected_lineage"],
        "constraints": autopsy["constraints"],
        "data_contract": autopsy["data_contract"],
        "diagnostics": autopsy["diagnostics"],
        "limitations": autopsy["limitations"],
        "lifecycle": autopsy["lifecycle"],
        "artifact_contract": autopsy["artifact_contract"],
        "implementation_provenance": _implementation_provenance(),
    }
    spec["autopsy_spec_sha256"] = _canonical_sha256(spec)
    return spec


def _artifact_manifest(
    output: Path,
    files: Iterable[str],
    autopsy_spec_sha256: str,
) -> dict[str, object]:
    rows = []
    for relative in sorted(set(files)):
        path = output / relative
        if not path.is_file():
            raise RuntimeError(f"V46 output is missing before sealing: {relative}")
        rows.append({
            "path": relative,
            "bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        })
    manifest: dict[str, object] = {
        "version": "v46_artifact_manifest_v1",
        "autopsy_spec_sha256": autopsy_spec_sha256,
        "files": rows,
    }
    manifest["manifest_semantic_sha256"] = _canonical_sha256(manifest)
    return manifest


def _seal_packet(
    output: Path,
    result: dict[str, object],
    core_files: Iterable[str],
) -> None:
    spec_sha = str(result["autopsy_spec"]["autopsy_spec_sha256"])
    manifest = _artifact_manifest(output, core_files, spec_sha)
    _write_json_atomic(output / "artifact_manifest.json", manifest)
    _write_json_atomic(output / "result.json", result)
    completion = {
        "version": "v46_completion_receipt_v1",
        "mode": result["mode"],
        "decision": result["decision"],
        "autopsy_spec_sha256": spec_sha,
        "artifact_manifest_sha256": _sha256_file(
            output / "artifact_manifest.json"
        ),
        "result_sha256": _sha256_file(output / "result.json"),
    }
    _write_json_atomic(output / "completion_receipt.json", completion)


def _validate_packet(
    output: Path,
    expected_spec_sha256: str,
    mode: str,
    required_files: Iterable[str],
) -> dict[str, object]:
    required = tuple(str(value) for value in required_files)
    if len(required) != len(set(required)):
        raise RuntimeError("V46 required artifact grid contains duplicates")
    for relative in required:
        if not (output / relative).is_file():
            raise RuntimeError(f"V46 cached {mode} packet is incomplete")
    completion = _load_json(output / "completion_receipt.json")
    if (
        completion.get("version") != "v46_completion_receipt_v1"
        or completion.get("mode") != mode
        or completion.get("autopsy_spec_sha256") != expected_spec_sha256
        or completion.get("artifact_manifest_sha256")
        != _sha256_file(output / "artifact_manifest.json")
        or completion.get("result_sha256") != _sha256_file(output / "result.json")
    ):
        raise RuntimeError(f"V46 cached {mode} completion receipt drift")
    manifest = _load_json(output / "artifact_manifest.json")
    semantic = dict(manifest)
    claimed_semantic = semantic.pop("manifest_semantic_sha256", None)
    rows = manifest.get("files", [])
    if (
        manifest.get("version") != "v46_artifact_manifest_v1"
        or manifest.get("autopsy_spec_sha256") != expected_spec_sha256
        or claimed_semantic != _canonical_sha256(semantic)
        or not isinstance(rows, list)
    ):
        raise RuntimeError(f"V46 cached {mode} manifest drift")
    envelope = {"artifact_manifest.json", "completion_receipt.json", "result.json"}
    expected_paths = set(required) - envelope
    observed_paths = [str(row.get("path", "")) for row in rows]
    if (
        len(observed_paths) != len(set(observed_paths))
        or set(observed_paths) != expected_paths
    ):
        raise RuntimeError(f"V46 cached {mode} manifest file grid drift")
    for row in rows:
        relative = str(row["path"])
        path = (output / relative).resolve()
        if not path.is_relative_to(output.resolve()):
            raise RuntimeError("V46 manifest path escaped output directory")
        if (
            not path.is_file()
            or int(row.get("bytes", -1)) != path.stat().st_size
            or row.get("sha256") != _sha256_file(path)
        ):
            raise RuntimeError(f"V46 cached artifact drift: {relative}")
    result = _load_json(output / "result.json")
    if (
        result.get("mode") != mode
        or result.get("autopsy_spec", {}).get("autopsy_spec_sha256")
        != expected_spec_sha256
        or not result.get("audit", {}).get("passed")
    ):
        raise RuntimeError(f"V46 cached {mode} result drift")
    return result


def _manifest_hashes(value: dict) -> dict[str, str]:
    rows = value.get("files", [])
    if not isinstance(rows, list):
        raise RuntimeError("V46 source manifest rows are invalid")
    result = {str(row["path"]): str(row["sha256"]) for row in rows}
    if len(result) != len(rows):
        raise RuntimeError("V46 source manifest contains duplicate paths")
    return result


def _preflight_context(config: dict) -> dict[str, object]:
    if "ranking_excess_failure_autopsy" not in config:
        raise RuntimeError("V46 config section is missing")
    autopsy = config["ranking_excess_failure_autopsy"]
    root = Path(autopsy["project_root"]).resolve()
    paths = {
        name: (root / relative).resolve()
        for name, relative in autopsy["inputs"].items()
    }
    if any(not path.is_relative_to(root) for path in paths.values()):
        raise RuntimeError("V46 input path escaped the project root")
    allowed = JSON_INPUTS | YAML_INPUTS | TABLE_INPUTS
    if set(paths) != allowed or set(autopsy["expected_input_sha256"]) != allowed:
        raise RuntimeError("V46 input allowlist drift")
    hashes = {}
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"V46 input is missing: {name}")
        observed = _sha256_file(path)
        hashes[name] = observed
        if observed != autopsy["expected_input_sha256"][name]:
            raise RuntimeError(f"V46 input hash drift: {name}")
    values = {name: _load_json(paths[name]) for name in JSON_INPUTS}
    resolved_v45 = yaml.safe_load(
        paths["v45_resolved_config"].read_text(encoding="utf-8")
    )
    expected = autopsy["expected_lineage"]
    result = values["v45_result"]
    gate = values["v45_gate_result"]
    prepare = values["v45_prepare_result"]
    completion = values["v45_completion_receipt"]
    prepare_completion = values["v45_prepare_completion_receipt"]
    unseal = values["v45_unseal_receipt"]
    outcome_receipt = values["v45_outcome_receipt"]
    failed_cells = [
        row
        for section in ("predictive_cells", "economic_cells", "bootstrap_cells")
        for row in gate[section]
        if not row["passed"]
    ]
    if (
        result.get("decision") != expected["v45_decision"]
        or result.get("evaluation_spec", {}).get("evaluation_spec_sha256")
        != expected["v45_evaluation_spec_sha256"]
        or int(result.get("evaluation_execution_count", -1))
        != int(expected["v45_evaluation_execution_count"])
        or int(result.get("summary", {}).get("gate_cells", -1))
        != int(expected["v45_gate_cells"])
        or bool(gate.get("passed"))
        or int(gate.get("cell_count", -1)) != int(expected["v45_gate_cells"])
        or len(failed_cells) != int(expected["v45_failed_gate_cells"])
        or int(expected["v45_gate_cells"]) - len(failed_cells)
        != int(expected["v45_passed_gate_cells"])
        or not values["v45_audit"].get("passed")
        or completion.get("result_sha256") != hashes["v45_result"]
        or completion.get("decision") != expected["v45_decision"]
        or prepare.get("decision") != "authorize_v45_one_shot_outcome_unseal_only"
        or int(prepare.get("summary", {}).get("heldout_outcome_rows", -1)) != 0
        or prepare_completion.get("result_sha256") != hashes["v45_prepare_result"]
        or hashes["v45_prepare_completion_receipt"]
        != expected["v45_prepare_completion_receipt_sha256"]
        or unseal.get("prepare_completion_receipt_sha256")
        != expected["v45_prepare_completion_receipt_sha256"]
        or int(unseal.get("evaluation_execution_count", -1)) != 1
        or hashes["v45_outcome_receipt"]
        != expected["v45_outcome_receipt_sha256"]
        or outcome_receipt.get("outcomes_parquet_sha256") != hashes["outcomes"]
        or int(outcome_receipt.get("outcome_rows", -1))
        != int(autopsy["data_contract"]["expected_asset_dates"])
        or int(outcome_receipt.get("evaluation_execution_count", -1)) != 1
        or values["v45_prepare_data_audit"].get("totals", {}).get(
            "heldout_outcome_rows"
        ) != 0
        or values["v45_prepare_data_access_receipt"].get("heldout_label_reads") != 0
        or _canonical_sha256(resolved_v45)
        != result["evaluation_spec"]["resolved_config_semantic_sha256"]
    ):
        raise RuntimeError("V46 frozen V45 lineage drift")

    evaluate_manifest = _manifest_hashes(values["v45_artifact_manifest"])
    prepare_manifest = _manifest_hashes(values["v45_prepare_artifact_manifest"])
    evaluate_table_paths = {
        "outcomes": "outcomes.parquet",
        "predictive_context_metrics": "predictive_context_metrics.parquet",
        "predictive_daily_metrics": "predictive_daily_metrics.parquet",
        "daily_returns": "daily_returns.parquet",
    }
    prepare_table_paths = {
        "context_predictions": "context_predictions.parquet",
        "asset_predictions": "asset_predictions.parquet",
        "positions": "positions.parquet",
    }
    for name, relative in evaluate_table_paths.items():
        if evaluate_manifest.get(relative) != hashes[name]:
            raise RuntimeError(f"V46 evaluate manifest binding drift: {name}")
    for name, relative in prepare_table_paths.items():
        if prepare_manifest.get(relative) != hashes[name]:
            raise RuntimeError(f"V46 prepare manifest binding drift: {name}")

    spec = build_autopsy_spec(config)
    return {
        "root": root,
        "paths": paths,
        "hashes": hashes,
        "values": values,
        "resolved_v45": resolved_v45,
        "autopsy_spec": spec,
        "failed_cells": failed_cells,
    }


def _preflight_report(result: dict[str, object]) -> str:
    return "\n".join([
        "# TLM V46 Ranking/Excess Failure Autopsy Preflight",
        "",
        "## Decision",
        "",
        "**HASH-EXACT READ-ONLY AUTOPSY AUTHORIZED.**",
        "",
        f"Autopsy-spec SHA-256: `{result['autopsy_spec']['autopsy_spec_sha256']}`",
        "Parquet files deserialized: **0**",
        "Models/checkpoints opened: **0**",
        "V45 retirement remains immutable.",
        "",
    ])


def preflight_ranking_excess_failure_autopsy(config: dict) -> dict[str, object]:
    context = _preflight_context(config)
    autopsy = config["ranking_excess_failure_autopsy"]
    output = context["root"] / autopsy["preflight_output_dir"]
    spec = context["autopsy_spec"]
    required = autopsy["artifact_contract"]["preflight_required_files"]
    if (output / "completion_receipt.json").is_file():
        return _validate_packet(
            output, spec["autopsy_spec_sha256"], "preflight", required
        )
    output.mkdir(parents=True, exist_ok=True)
    input_receipt = {
        "version": "v46_input_hash_receipt_v1",
        "hashes": context["hashes"],
        "json_files_deserialized": len(JSON_INPUTS),
        "yaml_files_deserialized": len(YAML_INPUTS),
        "parquet_files_raw_byte_hashed": len(TABLE_INPUTS),
        "parquet_files_deserialized": 0,
    }
    checks = {
        "all_allowlisted_input_hashes_match": True,
        "v45_retirement_and_three_failed_gates_are_exact": True,
        "v45_one_shot_receipts_are_exact": True,
        "prepare_packet_had_zero_heldout_outcomes": True,
        "source_manifests_bind_every_table_input": True,
        "target_assets_and_post_2025_remain_forbidden": True,
        "zero_parquet_deserializations": True,
        "zero_model_checkpoint_or_raw_panel_access": True,
    }
    audit = {"checks": checks, "passed": bool(all(checks.values()))}
    result: dict[str, object] = {
        "version": VERSION,
        "mode": "preflight",
        "decision": "authorize_v46_deterministic_read_only_autopsy",
        "autopsy_spec": spec,
        "input_hash_receipt": input_receipt,
        "summary": {
            "input_files_hashed": len(context["hashes"]),
            "parquet_files_deserialized": 0,
            "models_or_checkpoints_opened": 0,
            "new_predictions_or_positions": 0,
            "v45_failed_gate_cells": len(context["failed_cells"]),
            "target_asset_rows": 0,
            "post_2025_rows": 0,
        },
        "audit": audit,
    }
    _write_json_atomic(output / "autopsy_spec.json", spec)
    _write_json_atomic(output / "input_hash_receipt.json", input_receipt)
    _write_json_atomic(output / "audit.json", audit)
    _write_yaml_atomic(output / "resolved_config.yaml", config)
    _atomic_write_text(output / "report.md", _preflight_report(result))
    _seal_packet(
        output,
        result,
        (
            "autopsy_spec.json",
            "input_hash_receipt.json",
            "audit.json",
            "resolved_config.yaml",
            "report.md",
        ),
    )
    return _validate_packet(
        output, spec["autopsy_spec_sha256"], "preflight", required
    )


def _normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], utc=True)
    if "target_window_end_date" in result:
        result["target_window_end_date"] = pd.to_datetime(
            result["target_window_end_date"], utc=True
        )
    if "fold" in result:
        result["fold"] = result["fold"].astype(int)
    for column in ("symbol", "symbol_0", "symbol_1", "symbol_2"):
        if column in result:
            result[column] = result[column].astype(str)
    return result


def _finite_correlation(
    left: pd.Series | np.ndarray,
    right: pd.Series | np.ndarray,
    method: str,
) -> float | None:
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if (
        len(frame) < 3
        or frame["left"].nunique() < 2
        or frame["right"].nunique() < 2
    ):
        return None
    if method == "pearson":
        value = frame["left"].corr(frame["right"])
    elif method == "spearman":
        value = frame["left"].rank(method="average").corr(
            frame["right"].rank(method="average")
        )
    else:
        raise ValueError(f"Unsupported V46 correlation: {method}")
    return float(value) if pd.notna(value) else None


def _compound(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all() or bool((array <= -1.0).any()):
        raise ValueError("V46 compounding requires finite returns above -100%")
    return float(np.prod(1.0 + array) - 1.0)


def _max_drawdown(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    equity = np.cumprod(1.0 + array)
    peaks = np.maximum.accumulate(np.maximum(equity, 1.0))
    return float(np.min(equity / peaks - 1.0))


def build_context_stability(
    context_predictions: pd.DataFrame,
    asset_predictions: pd.DataFrame,
    outcomes: pd.DataFrame,
    tolerance: float,
) -> pd.DataFrame:
    required_context = {
        "date",
        "fold",
        "triplet_key",
        *(f"symbol_{slot}" for slot in range(3)),
        *(f"transformer_raw_excess_{slot}" for slot in range(3)),
        *(f"ridge_raw_excess_{slot}" for slot in range(3)),
    }
    required_asset = {
        "date",
        "fold",
        "symbol",
        "context_count",
        "transformer_raw_excess",
        "ridge_raw_excess",
        "momentum_30",
    }
    required_outcome = {"date", "fold", "symbol", "action_log_return"}
    if (
        not required_context.issubset(context_predictions)
        or not required_asset.issubset(asset_predictions)
        or not required_outcome.issubset(outcomes)
    ):
        raise ValueError("V46 context-stability columns are incomplete")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("V46 context-stability tolerance is invalid")
    context = _normalize_dates(context_predictions)
    asset = _normalize_dates(asset_predictions)
    observed = _normalize_dates(outcomes)
    if (
        context.duplicated(["date", "fold", "triplet_key"]).any()
        or asset.duplicated(["date", "fold", "symbol"]).any()
        or observed.duplicated(["date", "fold", "symbol"]).any()
    ):
        raise ValueError("V46 context-stability inputs contain duplicate keys")
    outcome_lookup = {
        (pd.Timestamp(row.date), int(row.fold), str(row.symbol)): float(
            row.action_log_return
        )
        for row in observed.itertuples(index=False)
    }
    exploded = []
    for slot in range(3):
        current = context[
            [
                "date",
                "fold",
                "triplet_key",
                f"symbol_{slot}",
                f"transformer_raw_excess_{slot}",
                f"ridge_raw_excess_{slot}",
            ]
        ].copy()
        current = current.rename(
            columns={
                f"symbol_{slot}": "symbol",
                f"transformer_raw_excess_{slot}": "context_score",
                f"ridge_raw_excess_{slot}": "ridge_context_score",
            }
        )
        transformer_scores = context[
            [f"transformer_raw_excess_{index}" for index in range(3)]
        ].to_numpy(dtype=np.float64)
        ridge_scores = context[
            [f"ridge_raw_excess_{index}" for index in range(3)]
        ].to_numpy(dtype=np.float64)
        symbols = context[
            [f"symbol_{index}" for index in range(3)]
        ].to_numpy(dtype=str)
        transformer_top = np.asarray(
            [
                min(
                    range(3),
                    key=lambda index: (-float(scores[index]), names[index]),
                )
                for scores, names in zip(transformer_scores, symbols, strict=True)
            ],
            dtype=np.int64,
        )
        ridge_top = np.asarray(
            [
                min(
                    range(3),
                    key=lambda index: (-float(scores[index]), names[index]),
                )
                for scores, names in zip(ridge_scores, symbols, strict=True)
            ],
            dtype=np.int64,
        )
        current["is_triplet_top1"] = transformer_top == slot
        current["ridge_is_triplet_top1"] = ridge_top == slot
        realized_excess = []
        for row in context.itertuples(index=False):
            date = pd.Timestamp(row.date)
            fold = int(row.fold)
            triplet_symbols = tuple(
                str(getattr(row, f"symbol_{index}")) for index in range(3)
            )
            triplet_returns = np.asarray(
                [outcome_lookup[(date, fold, symbol)] for symbol in triplet_symbols],
                dtype=np.float64,
            )
            realized_excess.append(
                float(triplet_returns[slot] - triplet_returns.mean())
            )
        current["context_realized_excess_log_return"] = realized_excess
        exploded.append(
            current[
                [
                    "date",
                    "fold",
                    "triplet_key",
                    "symbol",
                    "context_score",
                    "ridge_context_score",
                    "is_triplet_top1",
                    "ridge_is_triplet_top1",
                    "context_realized_excess_log_return",
                ]
            ]
        )
    long = pd.concat(exploded, ignore_index=True)
    numeric = long[
        ["context_score", "ridge_context_score"]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("V46 context predictions contain non-finite scores")
    long["context_score_sq"] = long["context_score"] ** 2
    long["ridge_context_score_sq"] = long["ridge_context_score"] ** 2
    long["positive_score"] = long["context_score"] > 0.0
    long["ridge_positive_score"] = long["ridge_context_score"] > 0.0
    grouped = long.groupby(["date", "fold", "symbol"], sort=True).agg(
        context_count=("context_score", "size"),
        context_score_mean=("context_score", "mean"),
        context_score_sq_mean=("context_score_sq", "mean"),
        context_score_min=("context_score", "min"),
        context_score_max=("context_score", "max"),
        positive_score_fraction=("positive_score", "mean"),
        triplet_top1_fraction=("is_triplet_top1", "mean"),
        ridge_context_score_mean=("ridge_context_score", "mean"),
        ridge_context_score_sq_mean=("ridge_context_score_sq", "mean"),
        ridge_context_score_min=("ridge_context_score", "min"),
        ridge_context_score_max=("ridge_context_score", "max"),
        ridge_positive_score_fraction=("ridge_positive_score", "mean"),
        ridge_triplet_top1_fraction=("ridge_is_triplet_top1", "mean"),
        context_averaged_realized_excess_log_return=(
            "context_realized_excess_log_return",
            "mean",
        ),
    ).reset_index()
    grouped["context_score_std_ddof0"] = np.sqrt(
        np.maximum(
            grouped["context_score_sq_mean"]
            - grouped["context_score_mean"] ** 2,
            0.0,
        )
    )
    grouped["ridge_context_score_std_ddof0"] = np.sqrt(
        np.maximum(
            grouped["ridge_context_score_sq_mean"]
            - grouped["ridge_context_score_mean"] ** 2,
            0.0,
        )
    )
    grouped = grouped.drop(
        columns=["context_score_sq_mean", "ridge_context_score_sq_mean"]
    )
    merged = grouped.merge(
        asset,
        on=["date", "fold", "symbol"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if (
        not merged["_merge"].eq("both").all()
        or not np.array_equal(
            merged["context_count_x"].to_numpy(dtype=np.int64),
            merged["context_count_y"].to_numpy(dtype=np.int64),
        )
        or not np.allclose(
            merged["context_score_mean"],
            merged["transformer_raw_excess"],
            rtol=0.0,
            atol=tolerance,
        )
        or not np.allclose(
            merged["ridge_context_score_mean"],
            merged["ridge_raw_excess"],
            rtol=0.0,
            atol=tolerance,
        )
    ):
        raise ValueError("V46 context-to-asset aggregation drift")
    merged = merged.drop(columns=["_merge", "context_count_y"]).rename(
        columns={"context_count_x": "context_count"}
    )
    merged = merged.merge(
        observed[["date", "fold", "symbol", "action_log_return"]],
        on=["date", "fold", "symbol"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise RuntimeError("V46 context-stability outcome keys drift")
    merged = merged.drop(columns="_merge")
    fold_mean = merged.groupby(["date", "fold"])["action_log_return"].transform(
        "mean"
    )
    merged["fold_excess_log_return"] = (
        merged["action_log_return"] - fold_mean
    )
    merged["transformer_prediction_error"] = (
        merged["transformer_raw_excess"]
        - merged["context_averaged_realized_excess_log_return"]
    )
    merged["transformer_absolute_prediction_error"] = np.abs(
        merged["transformer_prediction_error"]
    )
    merged["ridge_prediction_error"] = (
        merged["ridge_raw_excess"]
        - merged["context_averaged_realized_excess_log_return"]
    )
    merged["ridge_absolute_prediction_error"] = np.abs(
        merged["ridge_prediction_error"]
    )
    return merged.sort_values(["date", "fold", "symbol"]).reset_index(drop=True)


def _actual_rank(values: np.ndarray, index: int) -> float:
    ranks = pd.Series(values).rank(method="average", ascending=False)
    return float(ranks.iloc[index])


def _breadth_regime(positive: int, eligible: int) -> str:
    fraction = positive / eligible
    if positive == 0:
        return "cash_gate_closed"
    if fraction <= 1.0 / 3.0:
        return "narrow"
    if fraction <= 2.0 / 3.0:
        return "medium"
    return "broad"


def _pairwise_accuracy(
    scores: np.ndarray,
    actual: np.ndarray,
    tie_tolerance: float,
) -> tuple[int, int, float]:
    correct = 0
    active = 0
    for left in range(len(scores)):
        for right in range(left + 1, len(scores)):
            actual_difference = float(actual[left] - actual[right])
            if abs(actual_difference) <= tie_tolerance:
                continue
            active += 1
            correct += int(
                float(scores[left] - scores[right]) * actual_difference > 0.0
            )
    return correct, active, float(correct / active) if active else math.nan


def _top_k_hit(actual: np.ndarray, index: int, k: int, tolerance: float) -> bool:
    ordered = np.sort(actual)[::-1]
    boundary = float(ordered[min(int(k), len(ordered)) - 1])
    return bool(float(actual[index]) >= boundary - tolerance)


def build_fold_date_diagnostics(
    asset_predictions: pd.DataFrame,
    positions: pd.DataFrame,
    outcomes: pd.DataFrame,
    daily_returns: pd.DataFrame,
    context_stability: pd.DataFrame,
    base_cost_bps: int,
    tie_tolerance: float,
    switch_hurdle: float = 0.002,
) -> pd.DataFrame:
    asset = _normalize_dates(asset_predictions)
    position = _normalize_dates(positions)
    observed = _normalize_dates(outcomes)
    daily = _normalize_dates(daily_returns)
    stability = _normalize_dates(context_stability)
    candidate_daily = daily.loc[
        (daily["cost_bps"].astype(int) == int(base_cost_bps))
        & daily["scope"].str.startswith("fold_")
        & (daily["strategy"] == "candidate")
    ].copy()
    candidate_daily["fold"] = candidate_daily["scope"].str.split("_").str[1].astype(int)
    control_daily = daily.loc[
        (daily["cost_bps"].astype(int) == int(base_cost_bps))
        & daily["scope"].str.startswith("fold_")
    ].copy()
    control_daily["fold"] = control_daily["scope"].str.split("_").str[1].astype(int)
    daily_lookup = {
        (pd.Timestamp(row.date), int(row.fold), str(row.strategy)): row
        for row in control_daily.itertuples(index=False)
    }
    outcome_lookup = {
        (pd.Timestamp(row.date), int(row.fold), str(row.symbol)): float(
            row.action_log_return
        )
        for row in observed.itertuples(index=False)
    }
    stability_lookup = {
        (pd.Timestamp(row.date), int(row.fold), str(row.symbol)): row
        for row in stability.itertuples(index=False)
    }
    position_groups = {
        (pd.Timestamp(date), int(fold)): frame.sort_values("symbol")
        for (date, fold), frame in position.groupby(["date", "fold"], sort=True)
    }
    records: list[dict[str, object]] = []
    for (date, fold), frame in asset.groupby(["date", "fold"], sort=True):
        date = pd.Timestamp(date)
        fold = int(fold)
        current = frame.sort_values("symbol").reset_index(drop=True)
        symbols = current["symbol"].astype(str).tolist()
        transformer = current["transformer_raw_excess"].to_numpy(dtype=np.float64)
        ridge = current["ridge_raw_excess"].to_numpy(dtype=np.float64)
        actual = np.asarray(
            [outcome_lookup[(date, fold, symbol)] for symbol in symbols],
            dtype=np.float64,
        )
        simple = np.expm1(actual)
        transformer_top = min(
            range(len(symbols)),
            key=lambda index: (-float(transformer[index]), symbols[index]),
        )
        ridge_top = min(
            range(len(symbols)),
            key=lambda index: (-float(ridge[index]), symbols[index]),
        )
        ordered_scores = sorted(
            ((float(score), symbol) for score, symbol in zip(transformer, symbols, strict=True)),
            key=lambda item: (-item[0], item[1]),
        )
        desired_symbol = symbols[transformer_top]
        position_frame = position_groups[(date, fold)]
        held_rows = position_frame.loc[position_frame["candidate_weight"] > 0.5]
        if len(held_rows) > 1:
            raise RuntimeError("V46 candidate held multiple assets in one fold-date")
        held_symbol = str(held_rows.iloc[0]["symbol"]) if len(held_rows) else None
        held_index = symbols.index(held_symbol) if held_symbol is not None else None
        momentum = current["momentum_30"].to_numpy(dtype=np.float64)
        positive_count = int((momentum > 0.0).sum())
        candidate = daily_lookup[(date, fold, "candidate")]
        dual = daily_lookup[(date, fold, "dual_momentum_30")]
        equal = daily_lookup[(date, fold, "momentum_gated_equal_weight")]
        expected_gross = float(simple[held_index]) if held_index is not None else 0.0
        if not math.isclose(
            float(candidate.gross_return), expected_gross, rel_tol=0.0, abs_tol=1e-12
        ):
            raise RuntimeError("V46 candidate ledger used a non-held asset return")
        desired_stability = stability_lookup[(date, fold, desired_symbol)]
        held_stability = (
            stability_lookup[(date, fold, held_symbol)]
            if held_symbol is not None
            else None
        )
        maximum_actual = float(actual.max())
        transformer_selected_return = float(actual[transformer_top])
        ridge_selected_return = float(actual[ridge_top])
        transformer_pair_correct, transformer_pair_active, transformer_pairwise = (
            _pairwise_accuracy(transformer, actual, tie_tolerance)
        )
        ridge_pair_correct, ridge_pair_active, ridge_pairwise = _pairwise_accuracy(
            ridge, actual, tie_tolerance
        )
        record: dict[str, object] = {
            "date": date,
            "fold": fold,
            "calendar_month": date.strftime("%Y-%m"),
            "eligible_assets": len(symbols),
            "desired_top_symbol": desired_symbol,
            "ridge_top_symbol": symbols[ridge_top],
            "held_symbol": held_symbol,
            "candidate_active": held_symbol is not None,
            "held_is_desired_top": held_symbol == desired_symbol,
            "desired_top_score": float(transformer[transformer_top]),
            "desired_score_margin": float(ordered_scores[0][0] - ordered_scores[1][0]),
            "held_score": float(transformer[held_index]) if held_index is not None else math.nan,
            "desired_minus_held_score": (
                float(transformer[transformer_top] - transformer[held_index])
                if held_index is not None
                else math.nan
            ),
            "transformer_asset_spearman": average_rank_spearman(transformer, actual),
            "transformer_asset_pair_correct": transformer_pair_correct,
            "transformer_asset_pair_active": transformer_pair_active,
            "transformer_asset_pairwise_accuracy": transformer_pairwise,
            "ridge_asset_spearman": average_rank_spearman(ridge, actual),
            "ridge_asset_pair_correct": ridge_pair_correct,
            "ridge_asset_pair_active": ridge_pair_active,
            "ridge_asset_pairwise_accuracy": ridge_pairwise,
            "desired_top_actual_rank": _actual_rank(actual, transformer_top),
            "desired_top1_hit": bool(
                abs(transformer_selected_return - maximum_actual) <= tie_tolerance
            ),
            "desired_top3_hit": _top_k_hit(
                actual, transformer_top, 3, tie_tolerance
            ),
            "desired_top_excess_log_return": float(
                transformer_selected_return - actual.mean()
            ),
            "ridge_top_actual_rank": _actual_rank(actual, ridge_top),
            "ridge_top1_hit": bool(
                abs(ridge_selected_return - maximum_actual) <= tie_tolerance
            ),
            "ridge_top3_hit": _top_k_hit(actual, ridge_top, 3, tie_tolerance),
            "ridge_top_excess_log_return": float(ridge_selected_return - actual.mean()),
            "held_actual_rank": (
                _actual_rank(actual, held_index) if held_index is not None else math.nan
            ),
            "held_top1_hit": (
                _top_k_hit(actual, held_index, 1, tie_tolerance)
                if held_index is not None
                else False
            ),
            "held_top3_hit": (
                _top_k_hit(actual, held_index, 3, tie_tolerance)
                if held_index is not None
                else False
            ),
            "held_excess_log_return": (
                float(actual[held_index] - actual.mean())
                if held_index is not None
                else math.nan
            ),
            "held_absolute_log_return": (
                float(actual[held_index]) if held_index is not None else math.nan
            ),
            "fold_mean_log_return": float(actual.mean()),
            "fold_mean_simple_return": float(simple.mean()),
            "median_momentum_30": float(np.median(momentum)),
            "positive_momentum_assets": positive_count,
            "positive_momentum_fraction": float(positive_count / len(symbols)),
            "momentum_regime": "risk_on" if np.median(momentum) > 0.0 else "risk_off",
            "breadth_regime": _breadth_regime(positive_count, len(symbols)),
            "candidate_gross_return": float(candidate.gross_return),
            "candidate_net_return": float(candidate.net_return),
            "candidate_turnover": float(candidate.turnover),
            "candidate_cost": float(candidate.cost),
            "dual_momentum_net_return": float(dual.net_return),
            "momentum_gated_equal_weight_net_return": float(equal.net_return),
            "desired_context_score_std_ddof0": float(
                desired_stability.context_score_std_ddof0
            ),
            "desired_triplet_top1_fraction": float(
                desired_stability.triplet_top1_fraction
            ),
            "held_context_score_std_ddof0": (
                float(held_stability.context_score_std_ddof0)
                if held_stability is not None
                else math.nan
            ),
            "held_triplet_top1_fraction": (
                float(held_stability.triplet_top1_fraction)
                if held_stability is not None
                else math.nan
            ),
        }
        records.append(record)
    result = pd.DataFrame(records).sort_values(["fold", "date"]).reset_index(drop=True)
    result["position_state"] = ""
    result["previous_held_symbol"] = None
    result["challenger_minus_incumbent_score"] = math.nan
    result["final_liquidation_turnover"] = 0.0
    for fold, indexes in result.groupby("fold", sort=True).groups.items():
        previous: str | None = None
        ordered_indexes = list(indexes)
        for offset, index in enumerate(ordered_indexes):
            row = result.loc[index]
            held = row["held_symbol"] if pd.notna(row["held_symbol"]) else None
            eligible_symbols = set(
                asset.loc[
                    (asset["date"] == row["date"]) & (asset["fold"] == fold),
                    "symbol",
                ].astype(str)
            )
            gate_open = int(row["positive_momentum_assets"]) > 0
            desired = str(row["desired_top_symbol"])
            current_scores = asset.loc[
                (asset["date"] == row["date"]) & (asset["fold"] == fold)
            ].set_index("symbol")["transformer_raw_excess"]
            prior_is_eligible = previous is not None and previous in eligible_symbols
            challenger_gap = (
                float(current_scores.loc[desired] - current_scores.loc[previous])
                if prior_is_eligible and desired != previous
                else math.nan
            )
            result.at[index, "challenger_minus_incumbent_score"] = challenger_gap
            if not gate_open:
                expected_held = None
                state = "exit_to_cash" if previous is not None else "cash_gate"
            elif not prior_is_eligible:
                expected_held = desired
                state = (
                    "entry_from_cash"
                    if previous is None
                    else "switch_forced_ineligible"
                )
            elif desired == previous:
                expected_held = previous
                state = "hold_same_top1"
            elif challenger_gap > switch_hurdle:
                expected_held = desired
                state = "switch_hurdle"
            else:
                expected_held = previous
                state = "hold_same_non_top1"
            if held != expected_held:
                raise RuntimeError(
                    "V46 frozen candidate state does not reproduce the registered "
                    f"hysteresis policy on fold={fold}, date={row['date']}"
                )
            result.at[index, "position_state"] = state
            result.at[index, "previous_held_symbol"] = previous
            previous = held
            if offset == len(ordered_indexes) - 1 and held is not None:
                result.at[index, "final_liquidation_turnover"] = 1.0
    return result.sort_values(["date", "fold"]).reset_index(drop=True)


def reconcile_v45_ledger(
    positions: pd.DataFrame,
    outcomes: pd.DataFrame,
    daily_returns: pd.DataFrame,
    fold_symbols: dict[int | str, list[str]],
    costs: list[int],
    annualization_days: int,
    tolerance: float,
) -> dict[str, object]:
    persisted = _normalize_dates(daily_returns).sort_values(
        ["date", "cost_bps", "scope", "strategy"]
    ).reset_index(drop=True)
    portfolio = build_portfolio_evaluation(
        positions,
        outcomes,
        fold_symbols,
        costs,
        annualization_days,
    )
    recomputed = portfolio["daily_frame"].sort_values(
        ["date", "cost_bps", "scope", "strategy"]
    ).reset_index(drop=True)
    key_columns = ["date", "cost_bps", "scope", "strategy"]
    value_columns = [
        "gross_return",
        "turnover",
        "cost",
        "net_return",
        "equity",
    ]
    checks = {
        "daily_row_count_matches": len(persisted) == len(recomputed),
        "daily_keys_match": bool(
            persisted[key_columns].equals(recomputed[key_columns])
        ),
        "daily_values_match": bool(
            len(persisted) == len(recomputed)
            and np.allclose(
                persisted[value_columns].to_numpy(dtype=np.float64),
                recomputed[value_columns].to_numpy(dtype=np.float64),
                rtol=0.0,
                atol=tolerance,
            )
        ),
        "persisted_net_equals_gross_minus_cost": bool(
            np.allclose(
                persisted["net_return"],
                persisted["gross_return"] - persisted["cost"],
                rtol=0.0,
                atol=tolerance,
            )
        ),
        "registered_cost_grid_is_exact": set(persisted["cost_bps"].astype(int))
        == set(int(value) for value in costs),
    }
    return {"checks": checks, "portfolio": portfolio}


def extract_holding_episodes(
    fold_dates: pd.DataFrame,
    cost_rate: float,
) -> pd.DataFrame:
    if not math.isfinite(cost_rate) or cost_rate < 0.0:
        raise ValueError("V46 episode cost rate is invalid")
    required = {
        "date",
        "fold",
        "held_symbol",
        "candidate_gross_return",
        "position_state",
    }
    if not required.issubset(fold_dates):
        raise ValueError("V46 episode input columns are incomplete")
    source = _normalize_dates(fold_dates)
    episodes: list[dict[str, object]] = []
    for fold, frame in source.groupby("fold", sort=True):
        frame = frame.sort_values("date")
        current: list[pd.Series] = []
        prior_date: pd.Timestamp | None = None
        prior_symbol: str | None = None

        def close_episode(exit_reason: str) -> None:
            nonlocal current
            if not current:
                return
            gross_values = np.asarray(
                [float(row["candidate_gross_return"]) for row in current],
                dtype=np.float64,
            )
            allocated_cost = 2.0 * cost_rate
            gross_additive = float(gross_values.sum())
            episodes.append({
                "episode_id": len(episodes) + 1,
                "fold": int(fold),
                "symbol": str(current[0]["held_symbol"]),
                "start_date": pd.Timestamp(current[0]["date"]),
                "end_date": pd.Timestamp(current[-1]["date"]),
                "duration_signal_days": len(current),
                "entry_state": str(current[0]["position_state"]),
                "exit_reason": exit_reason,
                "momentum_regime_at_entry": (
                    str(current[0]["momentum_regime"])
                    if "momentum_regime" in current[0]
                    else None
                ),
                "breadth_regime_at_entry": (
                    str(current[0]["breadth_regime"])
                    if "breadth_regime" in current[0]
                    else None
                ),
                "gross_compounded_return": _compound(gross_values),
                "gross_additive_return": gross_additive,
                "allocated_entry_cost": cost_rate,
                "allocated_exit_cost": cost_rate,
                "allocated_cost": allocated_cost,
                "net_additive_return": gross_additive - allocated_cost,
                "winner_after_cost": bool(gross_additive - allocated_cost > 0.0),
            })
            current = []

        for _, row in frame.iterrows():
            date = pd.Timestamp(row["date"])
            symbol_value = row["held_symbol"]
            symbol = str(symbol_value) if pd.notna(symbol_value) else None
            continues = (
                symbol is not None
                and symbol == prior_symbol
                and prior_date is not None
                and date - prior_date == pd.Timedelta(days=1)
            )
            if symbol is None:
                close_episode(str(row["position_state"]))
            elif continues:
                current.append(row)
            else:
                if current:
                    reason = (
                        "calendar_gap"
                        if prior_date is not None
                        and date - prior_date != pd.Timedelta(days=1)
                        else str(row["position_state"])
                    )
                    close_episode(reason)
                current = [row]
            prior_date = date
            prior_symbol = symbol
        close_episode("final_liquidation")
    columns = [
        "episode_id",
        "fold",
        "symbol",
        "start_date",
        "end_date",
        "duration_signal_days",
        "entry_state",
        "exit_reason",
        "momentum_regime_at_entry",
        "breadth_regime_at_entry",
        "gross_compounded_return",
        "gross_additive_return",
        "allocated_entry_cost",
        "allocated_exit_cost",
        "allocated_cost",
        "net_additive_return",
        "winner_after_cost",
    ]
    return pd.DataFrame(episodes, columns=columns)


def loss_concentration(
    values: pd.Series | np.ndarray,
    top_n: list[int],
) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("V46 loss concentration requires a finite vector")
    if not top_n or any(int(value) < 1 for value in top_n):
        raise ValueError("V46 loss concentration top-n grid is invalid")
    losses = np.sort(-array[array < 0.0])[::-1]
    total = float(losses.sum())
    return {
        "losing_observation_count": int(len(losses)),
        "total_loss_magnitude": total,
        "shares": {
            str(int(count)): float(losses[: int(count)].sum() / total)
            if total > 0.0
            else 0.0
            for count in top_n
        },
    }


def extract_drawdown_episodes(
    dates: pd.Series | pd.DatetimeIndex | np.ndarray,
    net_returns: pd.Series | np.ndarray,
    top_n: int,
) -> list[dict[str, object]]:
    date_index = pd.DatetimeIndex(pd.to_datetime(dates, utc=True))
    values = np.asarray(net_returns, dtype=np.float64)
    if (
        len(date_index) != len(values)
        or len(values) < 1
        or not date_index.is_monotonic_increasing
        or not date_index.is_unique
        or not np.isfinite(values).all()
        or bool((values <= -1.0).any())
        or top_n < 1
    ):
        raise ValueError("V46 drawdown inputs are invalid")
    equity = np.cumprod(1.0 + values)
    peak_equity = 1.0
    peak_date = date_index[0] - pd.Timedelta(days=1)
    active: dict[str, object] | None = None
    episodes: list[dict[str, object]] = []
    for index, (date, current_equity) in enumerate(
        zip(date_index, equity, strict=True)
    ):
        if current_equity >= peak_equity:
            if active is not None:
                active["recovery_date"] = date
                active["end_date"] = date
                active["recovered"] = True
                active["duration_observations"] = index - int(active["start_index"]) + 1
                episodes.append(active)
                active = None
            peak_equity = float(current_equity)
            peak_date = date
            continue
        drawdown = float(current_equity / peak_equity - 1.0)
        if active is None:
            active = {
                "peak_date": peak_date,
                "start_date": date,
                "start_index": index,
                "peak_equity": peak_equity,
                "trough_date": date,
                "trough_equity": float(current_equity),
                "max_drawdown": drawdown,
                "recovery_date": None,
                "end_date": date_index[-1],
                "recovered": False,
            }
        elif drawdown < float(active["max_drawdown"]):
            active["trough_date"] = date
            active["trough_equity"] = float(current_equity)
            active["max_drawdown"] = drawdown
    if active is not None:
        active["duration_observations"] = len(values) - int(active["start_index"])
        episodes.append(active)
    ranked = sorted(episodes, key=lambda row: float(row["max_drawdown"]))[:top_n]
    result = []
    for rank, row in enumerate(ranked, start=1):
        current = dict(row)
        current.pop("start_index", None)
        current["rank"] = rank
        current["duration_calendar_days"] = int(
            (pd.Timestamp(current["end_date"]) - pd.Timestamp(current["start_date"])).days
            + 1
        )
        result.append(current)
    return result


def _mean_finite(values: pd.Series | np.ndarray | list[object]) -> float | None:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce")
    numeric = numeric[np.isfinite(numeric.to_numpy(dtype=np.float64))]
    return float(numeric.mean()) if len(numeric) else None


def _frames_reconcile(
    persisted: pd.DataFrame,
    recomputed: pd.DataFrame,
    key_columns: list[str],
    tolerance: float,
) -> bool:
    if list(persisted.columns) != list(recomputed.columns):
        return False
    left = _normalize_dates(persisted)
    right = _normalize_dates(recomputed)
    if len(left) != len(right) or any(
        column not in left or column not in right for column in key_columns
    ):
        return False
    left = left.sort_values(key_columns).reset_index(drop=True)
    right = right.sort_values(key_columns).reset_index(drop=True)
    if not left[key_columns].equals(right[key_columns]):
        return False
    for column in sorted(set(left.columns) - set(key_columns)):
        left_values = left[column]
        right_values = right[column]
        if pd.api.types.is_numeric_dtype(left_values) and pd.api.types.is_numeric_dtype(
            right_values
        ):
            if not np.allclose(
                left_values.to_numpy(dtype=np.float64),
                right_values.to_numpy(dtype=np.float64),
                rtol=0.0,
                atol=tolerance,
                equal_nan=True,
            ):
                return False
        elif not left_values.equals(right_values):
            return False
    return True


def _fold_symbols_from_positions(
    positions: pd.DataFrame,
) -> dict[int, list[str]]:
    normalized = _normalize_dates(positions)
    result = {
        int(fold): sorted(frame["symbol"].astype(str).unique().tolist())
        for fold, frame in normalized.groupby("fold", sort=True)
    }
    if set(result) != {1, 2, 3}:
        raise RuntimeError("V46 position folds are not exactly 1, 2, and 3")
    flattened = [symbol for symbols in result.values() for symbol in symbols]
    if len(flattened) != len(set(flattened)):
        raise RuntimeError("V46 fold universes are not asset-disjoint")
    return result


def _ranking_block(frame: pd.DataFrame) -> dict[str, object]:
    active = frame.loc[frame["candidate_active"]]
    return {
        "fold_date_count": int(len(frame)),
        "active_fold_date_count": int(len(active)),
        "transformer": {
            "defined_asset_spearman_fold_dates": int(
                np.isfinite(
                    frame["transformer_asset_spearman"].to_numpy(dtype=float)
                ).sum()
            ),
            "undefined_asset_spearman_fold_dates": int(
                (~np.isfinite(
                    frame["transformer_asset_spearman"].to_numpy(dtype=float)
                )).sum()
            ),
            "zero_active_pair_fold_dates": int(
                (frame["transformer_asset_pair_active"] == 0).sum()
            ),
            "mean_asset_spearman": _mean_finite(
                frame["transformer_asset_spearman"]
            ),
            "mean_asset_pairwise_accuracy": _mean_finite(
                frame["transformer_asset_pairwise_accuracy"]
            ),
            "top1_hit_rate": _mean_finite(frame["desired_top1_hit"]),
            "top3_hit_rate": _mean_finite(frame["desired_top3_hit"]),
            "mean_top1_actual_rank": _mean_finite(
                frame["desired_top_actual_rank"]
            ),
            "mean_top1_excess_log_return": _mean_finite(
                frame["desired_top_excess_log_return"]
            ),
        },
        "ridge": {
            "defined_asset_spearman_fold_dates": int(
                np.isfinite(frame["ridge_asset_spearman"].to_numpy(dtype=float)).sum()
            ),
            "undefined_asset_spearman_fold_dates": int(
                (~np.isfinite(
                    frame["ridge_asset_spearman"].to_numpy(dtype=float)
                )).sum()
            ),
            "zero_active_pair_fold_dates": int(
                (frame["ridge_asset_pair_active"] == 0).sum()
            ),
            "mean_asset_spearman": _mean_finite(frame["ridge_asset_spearman"]),
            "mean_asset_pairwise_accuracy": _mean_finite(
                frame["ridge_asset_pairwise_accuracy"]
            ),
            "top1_hit_rate": _mean_finite(frame["ridge_top1_hit"]),
            "top3_hit_rate": _mean_finite(frame["ridge_top3_hit"]),
            "mean_top1_actual_rank": _mean_finite(frame["ridge_top_actual_rank"]),
            "mean_top1_excess_log_return": _mean_finite(
                frame["ridge_top_excess_log_return"]
            ),
        },
        "held_candidate": {
            "active_denominator": int(len(active)),
            "active_rate": float(frame["candidate_active"].mean()),
            "held_is_desired_top_rate_active": (
                float(active["held_is_desired_top"].mean()) if len(active) else None
            ),
            "mean_actual_rank_active": _mean_finite(active["held_actual_rank"]),
            "top1_hit_rate_active": _mean_finite(active["held_top1_hit"]),
            "top3_hit_rate_active": _mean_finite(active["held_top3_hit"]),
            "mean_excess_log_return_active": _mean_finite(
                active["held_excess_log_return"]
            ),
            "mean_absolute_log_return_active": _mean_finite(
                active["held_absolute_log_return"]
            ),
        },
    }


def _build_ranking_daily(
    predictive_daily: pd.DataFrame,
    fold_dates: pd.DataFrame,
) -> pd.DataFrame:
    predictive = _normalize_dates(predictive_daily)
    selected = fold_dates[
        [
            "date",
            "fold",
            "calendar_month",
            "eligible_assets",
            "transformer_asset_spearman",
            "transformer_asset_pairwise_accuracy",
            "desired_top1_hit",
            "desired_top3_hit",
            "desired_top_actual_rank",
            "desired_top_excess_log_return",
            "ridge_asset_spearman",
            "ridge_asset_pairwise_accuracy",
            "ridge_top1_hit",
            "ridge_top3_hit",
            "ridge_top_actual_rank",
            "ridge_top_excess_log_return",
            "candidate_active",
            "held_is_desired_top",
            "held_actual_rank",
            "held_top1_hit",
            "held_top3_hit",
            "held_excess_log_return",
        ]
    ].copy()
    renamed = predictive.rename(
        columns={
            "transformer_spearman": "transformer_context_spearman",
            "transformer_pairwise_accuracy": (
                "transformer_context_pairwise_accuracy"
            ),
            "transformer_top1_hit_rate": "transformer_context_top1_hit_rate",
            "transformer_top1_excess": "transformer_context_top1_excess",
            "ridge_spearman": "ridge_context_spearman",
            "ridge_pairwise_accuracy": "ridge_context_pairwise_accuracy",
            "ridge_top1_hit_rate": "ridge_context_top1_hit_rate",
            "ridge_top1_excess": "ridge_context_top1_excess",
        }
    )
    result = renamed.merge(
        selected,
        on=["date", "fold"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not result["_merge"].eq("both").all():
        raise RuntimeError("V46 ranking-daily key reconciliation failed")
    return result.drop(columns="_merge").sort_values(["date", "fold"]).reset_index(
        drop=True
    )


def _build_asset_fold_metrics(
    context_stability: pd.DataFrame,
    fold_dates: pd.DataFrame,
    episodes: pd.DataFrame,
    fold_symbols: dict[int, list[str]],
    fold_capital_weight: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in (1, 2, 3):
        fold_daily = fold_dates.loc[fold_dates["fold"] == fold]
        active_days = int(fold_daily["candidate_active"].sum())
        for symbol in fold_symbols[fold]:
            stability = context_stability.loc[
                (context_stability["fold"] == fold)
                & (context_stability["symbol"] == symbol)
            ]
            desired = fold_daily.loc[fold_daily["desired_top_symbol"] == symbol]
            ridge_desired = fold_daily.loc[fold_daily["ridge_top_symbol"] == symbol]
            held = fold_daily.loc[fold_daily["held_symbol"] == symbol]
            symbol_episodes = episodes.loc[
                (episodes["fold"] == fold) & (episodes["symbol"] == symbol)
            ]
            gross_additive = float(held["candidate_gross_return"].sum())
            allocated_cost = float(symbol_episodes["allocated_cost"].sum())
            net_additive = gross_additive - allocated_cost
            rows.append({
                "fold": fold,
                "symbol": symbol,
                "eligible_days": int(len(stability)),
                "desired_top_days": int(len(desired)),
                "ridge_top_days": int(len(ridge_desired)),
                "held_days": int(len(held)),
                "held_share_of_active_days": (
                    float(len(held) / active_days) if active_days else 0.0
                ),
                "holding_episodes": int(len(symbol_episodes)),
                "held_positive_days": int(
                    (held["candidate_gross_return"] > 0.0).sum()
                ),
                "gross_compounded_held_return": _compound(
                    held["candidate_gross_return"]
                ),
                "gross_additive_held_return": gross_additive,
                "allocated_turnover": float(2 * len(symbol_episodes)),
                "allocated_cost": allocated_cost,
                "net_additive_contribution": net_additive,
                "equal_fold_capital_net_contribution": (
                    net_additive * fold_capital_weight
                ),
                "mean_held_log_return": _mean_finite(
                    held["held_absolute_log_return"]
                ),
                "mean_held_fold_excess_log_return": _mean_finite(
                    held["held_excess_log_return"]
                ),
                "mean_context_score_std_ddof0": _mean_finite(
                    stability["context_score_std_ddof0"]
                ),
                "mean_triplet_top1_fraction": _mean_finite(
                    stability["triplet_top1_fraction"]
                ),
                "transformer_score_realized_excess_pearson": _finite_correlation(
                    stability["transformer_raw_excess"],
                    stability["context_averaged_realized_excess_log_return"],
                    "pearson",
                ),
                "transformer_score_realized_excess_spearman": _finite_correlation(
                    stability["transformer_raw_excess"],
                    stability["context_averaged_realized_excess_log_return"],
                    "spearman",
                ),
                "ridge_score_realized_excess_pearson": _finite_correlation(
                    stability["ridge_raw_excess"],
                    stability["context_averaged_realized_excess_log_return"],
                    "pearson",
                ),
                "ridge_score_realized_excess_spearman": _finite_correlation(
                    stability["ridge_raw_excess"],
                    stability["context_averaged_realized_excess_log_return"],
                    "spearman",
                ),
            })
    return pd.DataFrame(rows).sort_values(["fold", "symbol"]).reset_index(drop=True)


def _build_monthly_metrics(daily_returns: pd.DataFrame) -> pd.DataFrame:
    source = _normalize_dates(daily_returns)
    source["calendar_month"] = source["date"].dt.strftime("%Y-%m")
    rows = []
    for (month, cost, scope, strategy), frame in source.groupby(
        ["calendar_month", "cost_bps", "scope", "strategy"], sort=True
    ):
        ordered = frame.sort_values("date")
        rows.append({
            "calendar_month": str(month),
            "cost_bps": int(cost),
            "scope": str(scope),
            "strategy": str(strategy),
            "observations": int(len(ordered)),
            "gross_compounded_return": _compound(ordered["gross_return"]),
            "net_compounded_return": _compound(ordered["net_return"]),
            "turnover_sum": float(ordered["turnover"].sum()),
            "cost_sum": float(ordered["cost"].sum()),
            "positive_net_day_rate": float((ordered["net_return"] > 0.0).mean()),
            "within_month_max_drawdown": _max_drawdown(ordered["net_return"]),
        })
    return pd.DataFrame(rows)


def _build_position_state_metrics(
    fold_dates: pd.DataFrame,
    states: list[str],
    fold_capital_weight: float,
) -> pd.DataFrame:
    scopes: list[tuple[str, int | None, pd.DataFrame, float]] = [
        (f"fold_{fold}", fold, fold_dates.loc[fold_dates["fold"] == fold], 1.0)
        for fold in (1, 2, 3)
    ]
    scopes.append(
        (
            "all_folds_equal_capital_contribution",
            None,
            fold_dates,
            fold_capital_weight,
        )
    )
    rows = []
    for scope, fold, source, weight in scopes:
        for state in states:
            frame = source.loc[source["position_state"] == state]
            rows.append({
                "scope": scope,
                "fold": fold,
                "position_state": state,
                "fold_date_count": int(len(frame)),
                "candidate_active_count": int(frame["candidate_active"].sum()),
                "held_is_desired_count": int(frame["held_is_desired_top"].sum()),
                "gross_additive_return": float(
                    frame["candidate_gross_return"].sum() * weight
                ),
                "net_additive_return": float(
                    frame["candidate_net_return"].sum() * weight
                ),
                "turnover_sum": float(frame["candidate_turnover"].sum() * weight),
                "cost_sum": float(frame["candidate_cost"].sum() * weight),
                "equal_fold_capital_net_contribution": float(
                    frame["candidate_net_return"].sum() * fold_capital_weight
                ),
                "mean_held_excess_log_return": _mean_finite(
                    frame["held_excess_log_return"]
                ),
                "mean_challenger_minus_incumbent_score": _mean_finite(
                    frame["challenger_minus_incumbent_score"]
                ),
            })
    return pd.DataFrame(rows)


def _build_regime_metrics(
    fold_dates: pd.DataFrame,
    daily_returns: pd.DataFrame,
    base_cost_bps: int,
    fold_capital_weight: float,
) -> pd.DataFrame:
    daily = _normalize_dates(daily_returns)
    fold_daily = daily.loc[
        (daily["cost_bps"].astype(int) == int(base_cost_bps))
        & daily["scope"].str.startswith("fold_")
    ].copy()
    fold_daily["fold"] = fold_daily["scope"].str.split("_").str[1].astype(int)
    joined = fold_daily.merge(
        fold_dates[
            [
                "date",
                "fold",
                "candidate_active",
                "momentum_regime",
                "breadth_regime",
            ]
        ],
        on=["date", "fold"],
        how="left",
        validate="many_to_one",
    )
    if joined[["momentum_regime", "breadth_regime"]].isna().any().any():
        raise RuntimeError("V46 regime metric join failed")
    axes = {
        "momentum": ("momentum_regime", ["risk_off", "risk_on"]),
        "breadth": (
            "breadth_regime",
            ["cash_gate_closed", "narrow", "medium", "broad"],
        ),
    }
    rows = []
    for scope_name, fold, source, weight in [
        *(
            (
                f"fold_{current_fold}",
                current_fold,
                joined.loc[joined["fold"] == current_fold],
                1.0,
            )
            for current_fold in (1, 2, 3)
        ),
        ("all_folds_equal_capital_contribution", None, joined, fold_capital_weight),
    ]:
        for family, (column, regimes) in axes.items():
            for regime in regimes:
                for strategy in sorted(STRATEGY_WEIGHT_COLUMNS):
                    frame = source.loc[
                        (source[column] == regime) & (source["strategy"] == strategy)
                    ]
                    rows.append({
                        "regime_family": family,
                        "regime": regime,
                        "scope": scope_name,
                        "fold": fold,
                        "strategy": strategy,
                        "fold_date_count": int(len(frame)),
                        "candidate_active_count": int(
                            frame["candidate_active"].sum()
                        ),
                        "gross_additive_return": float(
                            frame["gross_return"].sum() * weight
                        ),
                        "net_additive_return": float(
                            frame["net_return"].sum() * weight
                        ),
                        "turnover_sum": float(frame["turnover"].sum() * weight),
                        "cost_sum": float(frame["cost"].sum() * weight),
                        "gross_compounded_return": (
                            _compound(frame.sort_values("date")["gross_return"])
                            if fold is not None and len(frame)
                            else None
                        ),
                        "net_compounded_return": (
                            _compound(frame.sort_values("date")["net_return"])
                            if fold is not None and len(frame)
                            else None
                        ),
                        "equal_fold_capital_net_contribution": float(
                            frame["net_return"].sum() * fold_capital_weight
                        ),
                        "mean_net_return": _mean_finite(frame["net_return"]),
                        "positive_net_day_rate": (
                            float((frame["net_return"] > 0.0).mean())
                            if len(frame)
                            else None
                        ),
                    })
    return pd.DataFrame(rows)


def _context_stability_summary(
    context_stability: pd.DataFrame,
    fold_dates: pd.DataFrame,
) -> dict[str, object]:
    def summarize(frame: pd.DataFrame) -> dict[str, object]:
        score_range = frame["context_score_max"] - frame["context_score_min"]
        ridge_range = (
            frame["ridge_context_score_max"]
            - frame["ridge_context_score_min"]
        )
        result: dict[str, object] = {
            "asset_date_count": int(len(frame)),
            "mean_context_count": _mean_finite(frame["context_count"]),
            "transformer": {
                "mean_score_std_ddof0": _mean_finite(
                    frame["context_score_std_ddof0"]
                ),
                "mean_score_range": _mean_finite(score_range),
                "mean_positive_score_fraction": _mean_finite(
                    frame["positive_score_fraction"]
                ),
                "mean_triplet_top1_fraction": _mean_finite(
                    frame["triplet_top1_fraction"]
                ),
                "mean_absolute_prediction_error": _mean_finite(
                    frame["transformer_absolute_prediction_error"]
                ),
            },
            "ridge": {
                "mean_score_std_ddof0": _mean_finite(
                    frame["ridge_context_score_std_ddof0"]
                ),
                "mean_score_range": _mean_finite(ridge_range),
                "mean_positive_score_fraction": _mean_finite(
                    frame["ridge_positive_score_fraction"]
                ),
                "mean_triplet_top1_fraction": _mean_finite(
                    frame["ridge_triplet_top1_fraction"]
                ),
                "mean_absolute_prediction_error": _mean_finite(
                    frame["ridge_absolute_prediction_error"]
                ),
            },
            "fixed_correlations": {},
        }
        correlation_pairs = {
            "transformer_std_vs_absolute_error": (
                frame["context_score_std_ddof0"],
                frame["transformer_absolute_prediction_error"],
            ),
            "transformer_range_vs_absolute_error": (
                score_range,
                frame["transformer_absolute_prediction_error"],
            ),
            "transformer_top1_fraction_vs_absolute_error": (
                frame["triplet_top1_fraction"],
                frame["transformer_absolute_prediction_error"],
            ),
            "ridge_std_vs_absolute_error": (
                frame["ridge_context_score_std_ddof0"],
                frame["ridge_absolute_prediction_error"],
            ),
            "ridge_range_vs_absolute_error": (
                ridge_range,
                frame["ridge_absolute_prediction_error"],
            ),
        }
        for name, (left, right) in correlation_pairs.items():
            result["fixed_correlations"][name] = {
                method: _finite_correlation(left, right, method)
                for method in ("pearson", "spearman")
            }
        return result

    by_fold = {
        str(fold): summarize(
            context_stability.loc[context_stability["fold"] == fold]
        )
        for fold in (1, 2, 3)
    }
    desired_correlations = {}
    for fold_label, frame in [
        *(
            (str(fold), fold_dates.loc[fold_dates["fold"] == fold])
            for fold in (1, 2, 3)
        ),
        ("equal_fold_aggregate", fold_dates),
    ]:
        desired_correlations[fold_label] = {
            "desired_score_margin_vs_top1_excess": {
                method: _finite_correlation(
                    frame["desired_score_margin"],
                    frame["desired_top_excess_log_return"],
                    method,
                )
                for method in ("pearson", "spearman")
            },
            "held_context_std_vs_absolute_held_return": {
                method: _finite_correlation(
                    frame.loc[frame["candidate_active"], "held_context_score_std_ddof0"],
                    frame.loc[frame["candidate_active"], "held_absolute_log_return"].abs(),
                    method,
                )
                for method in ("pearson", "spearman")
            },
        }
    return {
        "target_definition": (
            "mean_over_asset_contexts_of_asset_return_minus_triplet_mean_return"
        ),
        "population_standard_deviation_ddof": 0,
        "by_fold": by_fold,
        "pooled_asset_dates_descriptive": summarize(context_stability),
        "policy_translation_correlations": desired_correlations,
        "limitations": {
            "seed_disagreement": (
                "unavailable_seed_outputs_were_averaged_before_persistence"
            ),
            "predicted_volatility": "unavailable_head_not_persisted_in_v45_screen",
            "causal_interpretation": False,
        },
    }


def _cost_decomposition(
    daily_returns: pd.DataFrame,
    positions: pd.DataFrame,
) -> dict[str, object]:
    daily = _normalize_dates(daily_returns)
    position = _normalize_dates(positions)
    liquidation: dict[tuple[str, str], float] = {}
    for fold in (1, 2, 3):
        frame = position.loc[position["fold"] == fold]
        final = frame.loc[frame["date"] == frame["date"].max()]
        for strategy, column in STRATEGY_WEIGHT_COLUMNS.items():
            liquidation[(f"fold_{fold}", strategy)] = float(final[column].sum())
    for strategy in STRATEGY_WEIGHT_COLUMNS:
        liquidation[("aggregate_equal_fold_capital", strategy)] = float(
            np.mean(
                [liquidation[(f"fold_{fold}", strategy)] for fold in (1, 2, 3)]
            )
        )
    cells = []
    for (cost, scope, strategy), frame in daily.groupby(
        ["cost_bps", "scope", "strategy"], sort=True
    ):
        ordered = frame.sort_values("date")
        gross = _compound(ordered["gross_return"])
        net = _compound(ordered["net_return"])
        cells.append({
            "cost_bps": int(cost),
            "scope": str(scope),
            "strategy": str(strategy),
            "observations": int(len(ordered)),
            "gross_compounded_return": gross,
            "net_compounded_return": net,
            "compounded_cost_drag": float(gross - net),
            "turnover_sum": float(ordered["turnover"].sum()),
            "additive_cost_sum": float(ordered["cost"].sum()),
            "final_liquidation_turnover": liquidation[(str(scope), str(strategy))],
        })
    frame = pd.DataFrame(cells)
    invariant_checks: dict[str, bool] = {}
    for (scope, strategy), group in frame.groupby(["scope", "strategy"]):
        ordered = group.sort_values("cost_bps")
        invariant_checks[f"{scope}:{strategy}:gross_invariant"] = bool(
            np.allclose(
                ordered["gross_compounded_return"],
                ordered["gross_compounded_return"].iloc[0],
                rtol=0.0,
                atol=1e-12,
            )
        )
        invariant_checks[f"{scope}:{strategy}:turnover_invariant"] = bool(
            np.allclose(
                ordered["turnover_sum"],
                ordered["turnover_sum"].iloc[0],
                rtol=0.0,
                atol=1e-12,
            )
        )
        invariant_checks[f"{scope}:{strategy}:net_nonincreasing"] = bool(
            (np.diff(ordered["net_compounded_return"].to_numpy()) <= 1e-12).all()
        )
        invariant_checks[f"{scope}:{strategy}:cost_formula"] = bool(
            np.allclose(
                ordered["additive_cost_sum"],
                ordered["turnover_sum"] * ordered["cost_bps"] / 10_000.0,
                rtol=0.0,
                atol=1e-12,
            )
        )
    return {"cells": cells, "invariant_checks": invariant_checks}


def _drawdown_diagnostics(
    daily_returns: pd.DataFrame,
    registered_metrics: dict[str, object],
    top_n: int,
    base_cost_bps: int,
) -> dict[str, object]:
    daily = _normalize_dates(daily_returns)
    cells = []
    for (cost, scope, strategy), frame in daily.groupby(
        ["cost_bps", "scope", "strategy"], sort=True
    ):
        ordered = frame.sort_values("date")
        registered = (
            registered_metrics[str(int(cost))][strategy]
            if scope == "aggregate_equal_fold_capital"
            else None
        )
        if scope.startswith("fold_"):
            fold = scope.split("_")[1]
            registered = registered_metrics["folds"][str(int(cost))][fold][strategy]
        episodes = extract_drawdown_episodes(
            ordered["date"], ordered["net_return"], top_n
        )
        cells.append({
            "cost_bps": int(cost),
            "scope": str(scope),
            "strategy": str(strategy),
            "max_drawdown": _max_drawdown(ordered["net_return"]),
            "registered_max_drawdown": float(registered["max_drawdown"]),
            "top_episodes": episodes,
        })
    aggregate_candidate = daily.loc[
        (daily["cost_bps"].astype(int) == int(base_cost_bps))
        & (daily["scope"] == "aggregate_equal_fold_capital")
        & (daily["strategy"] == "candidate")
    ].sort_values("date")
    worst = extract_drawdown_episodes(
        aggregate_candidate["date"], aggregate_candidate["net_return"], 1
    )[0]
    decline_start = pd.Timestamp(worst["start_date"])
    trough = pd.Timestamp(worst["trough_date"])
    attribution = []
    for scope in ("fold_1", "fold_2", "fold_3", "aggregate_equal_fold_capital"):
        frame = daily.loc[
            (daily["cost_bps"].astype(int) == int(base_cost_bps))
            & (daily["scope"] == scope)
            & (daily["strategy"] == "candidate")
            & (daily["date"] >= decline_start)
            & (daily["date"] <= trough)
        ].sort_values("date")
        attribution.append({
            "scope": scope,
            "start_date": decline_start,
            "trough_date": trough,
            "observations": int(len(frame)),
            "gross_compounded_return": _compound(frame["gross_return"]),
            "net_compounded_return": _compound(frame["net_return"]),
            "turnover_sum": float(frame["turnover"].sum()),
            "cost_sum": float(frame["cost"].sum()),
        })
    return {
        "cells": cells,
        "base_cost_candidate_worst_drawdown_attribution": {
            "episode": worst,
            "decline_only_contributions": attribution,
        },
    }


def _concentration_diagnostics(
    daily_returns: pd.DataFrame,
    fold_dates: pd.DataFrame,
    episodes: pd.DataFrame,
    asset_fold_metrics: pd.DataFrame,
    base_cost_bps: int,
    top_n: list[int],
    fold_capital_weight: float,
) -> dict[str, object]:
    daily = _normalize_dates(daily_returns)
    aggregate = daily.loc[
        (daily["cost_bps"].astype(int) == int(base_cost_bps))
        & (daily["scope"] == "aggregate_equal_fold_capital")
        & (daily["strategy"] == "candidate")
    ].sort_values("date")
    fold_candidate = daily.loc[
        (daily["cost_bps"].astype(int) == int(base_cost_bps))
        & daily["scope"].str.startswith("fold_")
        & (daily["strategy"] == "candidate")
    ].sort_values(["date", "scope"])

    def worst_records(frame: pd.DataFrame, value: str, count: int = 10) -> list[dict]:
        current = frame.nsmallest(count, value).copy()
        if "date" in current:
            current["date"] = pd.to_datetime(current["date"], utc=True).dt.strftime(
                "%Y-%m-%d"
            )
        return current.to_dict(orient="records")

    episode_values = episodes["net_additive_return"] * fold_capital_weight
    fold_contribution = fold_dates["candidate_net_return"] * fold_capital_weight
    asset_contribution = asset_fold_metrics["equal_fold_capital_net_contribution"]
    exposure = {}
    for fold in (1, 2, 3):
        frame = asset_fold_metrics.loc[asset_fold_metrics["fold"] == fold]
        active = int(frame["held_days"].sum())
        shares = (
            frame.set_index("symbol")["held_days"] / active
            if active
            else pd.Series(dtype=float)
        )
        hhi = float(np.square(shares).sum()) if active else 0.0
        dominant_symbol = str(shares.idxmax()) if active else None
        exposure[str(fold)] = {
            "active_fold_days": active,
            "dominant_symbol": dominant_symbol,
            "dominant_active_day_share": (
                float(shares.max()) if active else 0.0
            ),
            "herfindahl_index": hhi,
            "effective_assets": float(1.0 / hhi) if hhi > 0.0 else 0.0,
        }
    return {
        "base_cost_bps": int(base_cost_bps),
        "aggregate_candidate_losing_days": {
            **loss_concentration(aggregate["net_return"], top_n),
            "worst_records": worst_records(
                aggregate[["date", "net_return", "gross_return", "cost"]],
                "net_return",
            ),
        },
        "fold_candidate_losing_days": {
            **loss_concentration(fold_candidate["net_return"], top_n),
            "worst_records": worst_records(
                fold_candidate[["date", "scope", "net_return", "gross_return", "cost"]],
                "net_return",
            ),
        },
        "fold_day_equal_capital_contribution_losses": loss_concentration(
            fold_contribution, top_n
        ),
        "holding_episode_losses": loss_concentration(episode_values, top_n),
        "asset_additive_contribution_losses": loss_concentration(
            asset_contribution, top_n
        ),
        "held_asset_exposure_by_fold": exposure,
    }


def _v46_report(result: dict[str, object]) -> str:
    economic = result["economic_diagnostics"]
    base = economic["base_cost_candidate_aggregate"]
    folds = economic["base_cost_candidate_by_fold"]
    concentration = result["concentration"]
    failure = result["failure_attribution"]
    drawdown = result["drawdown_diagnostics"][
        "base_cost_candidate_worst_drawdown_attribution"
    ]["episode"]
    ranking = result["ranking_summary"]["context_averaged_asset"]
    stability = result["context_stability"]["pooled_asset_dates_descriptive"]
    exposure = concentration["held_asset_exposure_by_fold"]
    loss_shares = concentration["aggregate_candidate_losing_days"]["shares"]
    central_failure_text = (
        "The central failure is visible in fold 3: relative ranking remained "
        "positive and the held asset beat its fold cross-section on average, "
        "but its absolute long return was negative. Costs deepened that loss; "
        "they did not create the gross loss."
        if failure["relative_ranking_absolute_return_gap_observed"]
        else "The registered evidence did not satisfy the fixed definition of a relative-ranking/absolute-return gap."
    )
    outlier_text = (
        "The failure was dominated by one observation."
        if failure["worst_day_explains_majority_of_losing_day_magnitude"]
        else "The failure was not a one-observation collapse."
    )
    lines = [
        "# TLM V46 Ranking/Excess Failure Autopsy",
        "",
        "## Decision",
        "",
        "**V45 RETIREMENT CONFIRMED. THIS IS DIAGNOSTIC EVIDENCE ONLY.**",
        "",
        "No model was trained or instantiated, no prediction or position was regenerated, no counterfactual policy was tested, and BTC/ETH/SOL plus post-2025 observations remained sealed.",
        "",
        "## Objective and frozen evidence",
        "",
        "Explain why the one-shot V45 ranking/excess family failed 3 of 39 preregistered gates, using only the hash-locked V45 predictions, positions, outcomes, metrics, and ledger.",
        "",
        "## Economic result",
        "",
        f"At 10 bps, the frozen candidate returned **{base['gross_compounded_return']:.2%} gross** and **{base['net_compounded_return']:.2%} net**, with Sharpe **{base['registered_metrics']['sharpe']:.3f}**, maximum drawdown **{base['registered_metrics']['max_drawdown']:.2%}**, turnover **{base['turnover_sum']:.2f}**, and additive cost **{base['additive_cost_sum']:.2%}**.",
        "",
        "| Fold | Gross | Net | Turnover | Asset Spearman | Held excess | Held absolute return |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in folds:
        fold_ranking = ranking["by_fold"][str(row["fold"])]
        lines.append(
            f"| {row['fold']} | {row['gross_compounded_return']:.2%} | {row['net_compounded_return']:.2%} | {row['turnover_sum']:.2f} | {fold_ranking['transformer']['mean_asset_spearman']:.4f} | {fold_ranking['held_candidate']['mean_excess_log_return_active']:.3%} | {fold_ranking['held_candidate']['mean_absolute_log_return_active']:.3%} |"
        )
    lines.extend([
        "",
        central_failure_text,
        "",
        "## Concentration, costs, and drawdown",
        "",
        "| Fold | Dominant held asset | Active-day share | Effective assets |",
        "|---:|---|---:|---:|",
    ])
    for fold in (1, 2, 3):
        row = exposure[str(fold)]
        lines.append(
            f"| {fold} | {row['dominant_symbol']} | {row['dominant_active_day_share']:.2%} | {row['effective_assets']:.2f} |"
        )
    lines.extend([
        "",
        f"The worst aggregate drawdown ran from **{pd.Timestamp(drawdown['peak_date']).strftime('%Y-%m-%d')}** to **{pd.Timestamp(drawdown['trough_date']).strftime('%Y-%m-%d')}**, reaching **{drawdown['max_drawdown']:.2%}**; recovery occurred on **{pd.Timestamp(drawdown['recovery_date']).strftime('%Y-%m-%d') if drawdown['recovery_date'] is not None else 'not recovered'}**.",
        f"The worst day represented **{loss_shares['1']:.2%}** of total losing-day magnitude and the worst five represented **{loss_shares['5']:.2%}**. {outlier_text}",
        "",
        "## Context stability",
        "",
        f"Mean context dispersion (population standard deviation) was **{stability['transformer']['mean_score_std_ddof0']:.6f}**. Its Pearson association with absolute prediction error was **{stability['fixed_correlations']['transformer_std_vs_absolute_error']['pearson']:.4f}**. This is descriptive only and does not authorize a confidence filter.",
        "Per-seed disagreement and the predicted-volatility head cannot be recovered from the V45 packet because they were not persisted at that granularity.",
        "",
        "## Failure attribution",
        "",
        f"- Fold-3 return gate failed: **{str(failure['fold_3_return_gate_failed']).lower()}**",
        f"- Gross fold-3 loss existed before costs: **{str(failure['fold_3_gross_return_was_negative']).lower()}**",
        f"- Relative-ranking/absolute-return mismatch observed: **{str(failure['relative_ranking_absolute_return_gap_observed']).lower()}**",
        f"- Momentum-regime association had a stable sign across folds: **{str(failure['momentum_regime_association_sign_stable']).lower()}**",
        f"- Counterfactual policy tested: **{str(failure['counterfactual_policy_tested']).lower()}**",
        "",
        "## Leakage and accounting checks",
        "",
        "All 20 V45 inputs matched their preregistered hashes before and after analysis. The 96,864 context rows, 9,782 eligible asset-dates, 10,710 position rows, 1,071 fold-dates, 357 dates, and the complete 10/20/30 bps ledger reconciled exactly, including equal-fold capital and final liquidation.",
        "",
        "## Next legal research action",
        "",
        "If research continues, preregister a genuinely new family that combines relative ranking with an explicit absolute-return or market-state objective, ex-ante concentration control, and turnover-sensitive training. It may be evaluated only on genuinely new non-target data. These V46 slices cannot be converted into thresholds, filters, exclusions, or a revived V45 policy.",
        "",
    ])
    return "\n".join(lines)


def run_ranking_excess_failure_autopsy(config: dict) -> dict[str, object]:
    context = _preflight_context(config)
    autopsy = config["ranking_excess_failure_autopsy"]
    root = context["root"]
    spec = context["autopsy_spec"]
    preflight_output = root / autopsy["preflight_output_dir"]
    _validate_packet(
        preflight_output,
        str(spec["autopsy_spec_sha256"]),
        "preflight",
        autopsy["artifact_contract"]["preflight_required_files"],
    )
    output = (root / config["output_dir"]).resolve()
    if not output.is_relative_to(root):
        raise RuntimeError("V46 output path escaped the project root")
    required = autopsy["artifact_contract"]["run_required_files"]
    if (output / "completion_receipt.json").is_file():
        return _validate_packet(
            output, str(spec["autopsy_spec_sha256"]), "run", required
        )

    tables = {
        name: pd.read_parquet(context["paths"][name], engine="pyarrow")
        for name in sorted(TABLE_INPUTS)
    }
    context_predictions = _normalize_dates(tables["context_predictions"])
    asset_predictions = _normalize_dates(tables["asset_predictions"])
    positions = _normalize_dates(tables["positions"])
    outcomes = _normalize_dates(tables["outcomes"])
    persisted_context_metrics = _normalize_dates(
        tables["predictive_context_metrics"]
    )
    persisted_predictive_daily = _normalize_dates(
        tables["predictive_daily_metrics"]
    )
    persisted_daily_returns = _normalize_dates(tables["daily_returns"])

    expected_schemas = {
        "context_predictions": [
            "date", "fold", "triplet_key",
            *(f"symbol_{slot}" for slot in range(3)),
            *(f"transformer_raw_excess_{slot}" for slot in range(3)),
            *(f"ridge_raw_excess_{slot}" for slot in range(3)),
        ],
        "asset_predictions": [
            "date", "fold", "symbol", "context_count",
            "transformer_raw_excess", "ridge_raw_excess", "momentum_30",
        ],
        "positions": [
            "date", "fold", "symbol", "eligible", "momentum_30",
            *STRATEGY_WEIGHT_COLUMNS.values(),
        ],
        "outcomes": [
            "date", "fold", "symbol", "target_window_end_date",
            "action_log_return",
        ],
        "daily_returns": [
            "date", "cost_bps", "scope", "strategy", "gross_return",
            "turnover", "cost", "net_return", "equity",
        ],
    }
    schema_checks = {
        name: list(tables[name].columns) == columns
        for name, columns in expected_schemas.items()
    }
    if not all(schema_checks.values()):
        raise RuntimeError(f"V46 frozen input schema drift: {schema_checks}")
    dtype_and_finiteness_exact = bool(
        all(
            isinstance(frame["date"].dtype, pd.DatetimeTZDtype)
            and pd.api.types.is_integer_dtype(frame["fold"])
            for frame in (
                context_predictions,
                asset_predictions,
                positions,
                outcomes,
                persisted_context_metrics,
                persisted_predictive_daily,
            )
        )
        and isinstance(
            outcomes["target_window_end_date"].dtype, pd.DatetimeTZDtype
        )
        and pd.api.types.is_bool_dtype(positions["eligible"])
        and pd.api.types.is_integer_dtype(asset_predictions["context_count"])
        and pd.api.types.is_integer_dtype(persisted_daily_returns["cost_bps"])
        and np.isfinite(
            context_predictions[
                [
                    *(f"transformer_raw_excess_{slot}" for slot in range(3)),
                    *(f"ridge_raw_excess_{slot}" for slot in range(3)),
                ]
            ].to_numpy(dtype=float)
        ).all()
        and np.isfinite(
            asset_predictions[
                ["transformer_raw_excess", "ridge_raw_excess", "momentum_30"]
            ].to_numpy(dtype=float)
        ).all()
        and np.isfinite(outcomes["action_log_return"].to_numpy(dtype=float)).all()
        and np.isfinite(
            positions[list(STRATEGY_WEIGHT_COLUMNS.values())].to_numpy(dtype=float)
        ).all()
        and np.isfinite(
            persisted_daily_returns[
                ["gross_return", "turnover", "cost", "net_return", "equity"]
            ].to_numpy(dtype=float)
        ).all()
    )

    data_contract = autopsy["data_contract"]
    tolerance = float(autopsy["diagnostics"]["numerical_tolerance"])
    tie_tolerance = float(autopsy["diagnostics"]["actual_tie_tolerance"])
    costs = [int(value) for value in data_contract["registered_cost_bps"]]
    base_cost = int(autopsy["diagnostics"]["primary_cost_bps"])
    annualization_days = int(autopsy["diagnostics"]["annualization_days"])
    fold_weight = float(autopsy["diagnostics"]["fold_capital_weight"])
    switch_hurdle = float(
        context["resolved_v45"]["ranking_excess_screen"]["policy"][
            "switch_hurdle"
        ]
    )
    fold_symbols = _fold_symbols_from_positions(positions)
    all_symbols = {
        symbol for symbols in fold_symbols.values() for symbol in symbols
    }
    context_symbols = {
        str(symbol)
        for slot in range(3)
        for symbol in context_predictions[f"symbol_{slot}"].unique()
    }
    target_absence = not bool(
        (all_symbols | context_symbols | set(outcomes["symbol"])) & TARGET_SYMBOLS
    )
    context_keys_exact = bool(
        (
            context_predictions["triplet_key"]
            == context_predictions[
                [f"symbol_{slot}" for slot in range(3)]
            ].agg("|".join, axis=1)
        ).all()
    )
    transformer_context_centered = bool(
        np.allclose(
            context_predictions[
                [f"transformer_raw_excess_{slot}" for slot in range(3)]
            ].sum(axis=1),
            0.0,
            rtol=0.0,
            atol=tolerance,
        )
    )
    ridge_context_centered = bool(
        np.allclose(
            context_predictions[
                [f"ridge_raw_excess_{slot}" for slot in range(3)]
            ].sum(axis=1),
            0.0,
            rtol=0.0,
            atol=tolerance,
        )
    )
    signal_start = pd.Timestamp(data_contract["signal_start"], tz="UTC")
    signal_end = pd.Timestamp(data_contract["signal_end"], tz="UTC")
    maturity_end = pd.Timestamp(data_contract["maturity_end"], tz="UTC")
    scheduled_frames = (
        context_predictions,
        asset_predictions,
        positions,
        outcomes,
        persisted_context_metrics,
        persisted_predictive_daily,
        persisted_daily_returns,
    )
    post_2025_absence = all(
        int(frame["date"].dt.year.max()) <= 2025
        for frame in scheduled_frames
    ) and int(outcomes["target_window_end_date"].dt.year.max()) <= 2025
    signal_schedule_exact = all(
        frame["date"].min() == signal_start
        and frame["date"].max() == signal_end
        for frame in scheduled_frames
    )
    maturity_schedule_exact = bool(
        outcomes["target_window_end_date"].max() == maturity_end
        and (
            outcomes["target_window_end_date"]
            == outcomes["date"] + pd.Timedelta(days=8)
        ).all()
    )
    expected_dates = pd.date_range(signal_start, signal_end, freq="D")
    complete_signal_calendars = all(
        pd.DatetimeIndex(sorted(frame["date"].unique())).equals(expected_dates)
        for frame in scheduled_frames
    )

    recomputed_context_metrics, recomputed_predictive_daily, predictive_summary = (
        compute_predictive_metrics(
            context_predictions,
            outcomes,
            tie_tolerance,
        )
    )
    ledger = reconcile_v45_ledger(
        positions,
        outcomes,
        persisted_daily_returns,
        fold_symbols,
        costs,
        annualization_days,
        tolerance,
    )
    if not all(ledger["checks"].values()):
        raise RuntimeError(
            f"V46 frozen ledger reconciliation failed: {ledger['checks']}"
        )
    portfolio = ledger["portfolio"]
    stability = build_context_stability(
        context_predictions, asset_predictions, outcomes, tolerance
    )
    fold_dates = build_fold_date_diagnostics(
        asset_predictions,
        positions,
        outcomes,
        persisted_daily_returns,
        stability,
        base_cost,
        tie_tolerance,
        switch_hurdle,
    )
    fold_calendars_exact = all(
        pd.DatetimeIndex(
            fold_dates.loc[fold_dates["fold"] == fold, "date"]
            .sort_values()
            .unique()
        ).equals(expected_dates)
        for fold in (1, 2, 3)
    )
    episodes = extract_holding_episodes(fold_dates, base_cost / 10_000.0)
    ranking_daily = _build_ranking_daily(
        persisted_predictive_daily, fold_dates
    )
    asset_fold_metrics = _build_asset_fold_metrics(
        stability, fold_dates, episodes, fold_symbols, fold_weight
    )
    monthly_metrics = _build_monthly_metrics(persisted_daily_returns)
    position_state_metrics = _build_position_state_metrics(
        fold_dates,
        list(autopsy["diagnostics"]["position_states"]),
        fold_weight,
    )
    regime_metrics = _build_regime_metrics(
        fold_dates, persisted_daily_returns, base_cost, fold_weight
    )

    ranking_summary = {
        "aggregation_contract": {
            "triplet_context": "frozen_v45_equal_date_equal_fold_summary",
            "context_averaged_asset": "one_cross_section_per_fold_date",
            "aggregate": "equal_weight_per_date_and_fold_never_raw_context_pooling",
        },
        "triplet_context_frozen_v45": predictive_summary,
        "context_averaged_asset": {
            "by_fold": {
                str(fold): _ranking_block(
                    fold_dates.loc[fold_dates["fold"] == fold]
                )
                for fold in (1, 2, 3)
            },
            "equal_fold_aggregate": _ranking_block(fold_dates),
        },
    }
    stability_summary = _context_stability_summary(stability, fold_dates)
    cost_decomposition = _cost_decomposition(persisted_daily_returns, positions)
    registered_metrics = {
        "folds": portfolio["fold_metrics"],
        **portfolio["aggregate_metrics"],
    }
    drawdowns = _drawdown_diagnostics(
        persisted_daily_returns,
        registered_metrics,
        int(autopsy["diagnostics"]["drawdown_top_n"]),
        base_cost,
    )
    concentration = _concentration_diagnostics(
        persisted_daily_returns,
        fold_dates,
        episodes,
        asset_fold_metrics,
        base_cost,
        [int(value) for value in autopsy["diagnostics"]["loss_concentration_top_n"]],
        fold_weight,
    )

    cost_cells = cost_decomposition["cells"]
    def cost_cell(scope: str, strategy: str, cost: int) -> dict[str, object]:
        return next(
            row for row in cost_cells
            if row["scope"] == scope
            and row["strategy"] == strategy
            and int(row["cost_bps"]) == int(cost)
        )

    base_fold_rows = []
    for fold in (1, 2, 3):
        row = dict(cost_cell(f"fold_{fold}", "candidate", base_cost))
        row["fold"] = fold
        row["registered_metrics"] = portfolio["fold_metrics"][str(base_cost)][
            str(fold)
        ]["candidate"]
        base_fold_rows.append(row)
    base_aggregate = dict(
        cost_cell("aggregate_equal_fold_capital", "candidate", base_cost)
    )
    base_aggregate["registered_metrics"] = portfolio["aggregate_metrics"][
        str(base_cost)
    ]["candidate"]
    economic = {
        "base_cost_bps": base_cost,
        "registered_failed_gate_cells": context["failed_cells"],
        "registered_fold_metrics": portfolio["fold_metrics"],
        "registered_aggregate_metrics": portfolio["aggregate_metrics"],
        "base_cost_candidate_by_fold": base_fold_rows,
        "base_cost_candidate_aggregate": base_aggregate,
        "activity": {
            "fold_dates": int(len(fold_dates)),
            "active_fold_dates": int(fold_dates["candidate_active"].sum()),
            "active_rate": float(fold_dates["candidate_active"].mean()),
            "all_cash_calendar_dates": int(
                (~fold_dates.groupby("date")["candidate_active"].any()).sum()
            ),
        },
        "episodes": {
            "episode_count": int(len(episodes)),
            "active_signal_days": int(episodes["duration_signal_days"].sum()),
            "winning_episode_count": int(episodes["winner_after_cost"].sum()),
            "winner_rate": float(episodes["winner_after_cost"].mean()),
            "allocated_cost_sum": float(episodes["allocated_cost"].sum()),
        },
        "policy_translation": {
            "position_state_counts": {
                str(key): int(value)
                for key, value in fold_dates["position_state"].value_counts(
                    sort=False
                ).sort_index().items()
            },
            "held_is_desired_top_active_rate": float(
                fold_dates.loc[
                    fold_dates["candidate_active"], "held_is_desired_top"
                ].mean()
            ),
        },
        "ledger_reconciliation": ledger["checks"],
    }

    fold_regime_evidence = {}
    for fold in (1, 2, 3):
        frame = fold_dates.loc[fold_dates["fold"] == fold].sort_values("date")
        fold_regime_evidence[str(fold)] = {
            regime: {
                "observations": int(len(group)),
                "mean_daily_net_return": float(
                    group["candidate_net_return"].mean()
                ),
                "net_compounded_return": _compound(group["candidate_net_return"]),
            }
            for regime, group in frame.groupby("momentum_regime", sort=True)
        }
    regime_differences = []
    for fold in (1, 2, 3):
        values = fold_regime_evidence[str(fold)]
        regime_differences.append(
            float(
                values["risk_on"]["mean_daily_net_return"]
                - values["risk_off"]["mean_daily_net_return"]
            )
        )
    fold3_ranking = ranking_summary["context_averaged_asset"]["by_fold"]["3"]
    fold3_economic = base_fold_rows[2]
    failure_attribution = {
        "registered_failure_cells": context["failed_cells"],
        "fold_3_return_gate_failed": bool(
            fold3_economic["net_compounded_return"] <= 0.0
        ),
        "fold_3_gross_return_was_negative": bool(
            fold3_economic["gross_compounded_return"] < 0.0
        ),
        "costs_worsened_fold_3_but_did_not_create_its_gross_loss": bool(
            fold3_economic["gross_compounded_return"] < 0.0
            and fold3_economic["net_compounded_return"]
            < fold3_economic["gross_compounded_return"]
        ),
        "absolute_drawdown_gate_failed_at_20bps": bool(
            portfolio["aggregate_metrics"]["20"]["candidate"]["max_drawdown"]
            < -0.35
        ),
        "absolute_drawdown_gate_failed_at_30bps": bool(
            portfolio["aggregate_metrics"]["30"]["candidate"]["max_drawdown"]
            < -0.35
        ),
        "all_registered_predictive_gates_passed": bool(
            all(row["passed"] for row in context["values"]["v45_gate_result"]["predictive_cells"])
        ),
        "relative_ranking_absolute_return_gap_observed": bool(
            fold3_ranking["transformer"]["mean_asset_spearman"] > 0.0
            and fold3_ranking["held_candidate"]["mean_excess_log_return_active"]
            > 0.0
            and fold3_ranking["held_candidate"]["mean_absolute_log_return_active"]
            < 0.0
        ),
        "hysteresis_non_top_hold_observation_count": int(
            (fold_dates["position_state"] == "hold_same_non_top1").sum()
        ),
        "momentum_regime_net_compounded_returns_by_fold": fold_regime_evidence,
        "momentum_regime_association_sign_stable": bool(
            all(value > 0.0 for value in regime_differences)
            or all(value < 0.0 for value in regime_differences)
        ),
        "worst_day_explains_majority_of_losing_day_magnitude": bool(
            concentration["aggregate_candidate_losing_days"]["shares"]["1"]
            > 0.5
        ),
        "family_remains_retired": True,
        "counterfactual_policy_tested": False,
        "causal_claim_made_from_diagnostic_slice": False,
    }

    eligible_positions = positions.loc[positions["eligible"]].merge(
        asset_predictions,
        on=["date", "fold", "symbol"],
        how="outer",
        validate="one_to_one",
        suffixes=("_position", "_asset"),
        indicator=True,
    )
    expected_context_counts = stability.merge(
        stability.groupby(["date", "fold"], as_index=False).size().rename(
            columns={"size": "eligible_assets"}
        ),
        on=["date", "fold"],
        how="left",
        validate="many_to_one",
    )
    context_count_formula = np.asarray([
        math.comb(int(value) - 1, 2)
        for value in expected_context_counts["eligible_assets"]
    ], dtype=np.int64)
    input_hashes_after = {
        name: _sha256_file(path) for name, path in context["paths"].items()
    }
    transition_turnover = {
        "cash_gate": 0.0,
        "entry_from_cash": 1.0,
        "hold_same_top1": 0.0,
        "hold_same_non_top1": 0.0,
        "switch_hurdle": 2.0,
        "switch_forced_ineligible": 2.0,
        "exit_to_cash": 1.0,
    }
    expected_turnover = fold_dates["position_state"].map(transition_turnover) + (
        fold_dates["final_liquidation_turnover"]
    )
    turnover_by_fold_reconciles = all(
        math.isclose(
            float(2 * (episodes["fold"] == fold).sum()),
            float(
                fold_dates.loc[
                    fold_dates["fold"] == fold, "candidate_turnover"
                ].sum()
            ),
            rel_tol=0.0,
            abs_tol=tolerance,
        )
        for fold in (1, 2, 3)
    )
    asset_contributions_reconcile_by_fold = all(
        math.isclose(
            float(
                asset_fold_metrics.loc[
                    asset_fold_metrics["fold"] == fold,
                    "net_additive_contribution",
                ].sum()
            ),
            float(
                fold_dates.loc[
                    fold_dates["fold"] == fold, "candidate_net_return"
                ].sum()
            ),
            rel_tol=0.0,
            abs_tol=tolerance,
        )
        for fold in (1, 2, 3)
    )
    held_days_reconcile_by_fold = all(
        int(
            asset_fold_metrics.loc[
                asset_fold_metrics["fold"] == fold, "held_days"
            ].sum()
        )
        == int(
            fold_dates.loc[fold_dates["fold"] == fold, "candidate_active"].sum()
        )
        for fold in (1, 2, 3)
    )
    monthly_reconciles = True
    for cell in cost_cells:
        months = monthly_metrics.loc[
            (monthly_metrics["cost_bps"] == int(cell["cost_bps"]))
            & (monthly_metrics["scope"] == cell["scope"])
            & (monthly_metrics["strategy"] == cell["strategy"])
        ].sort_values("calendar_month")
        monthly_reconciles = monthly_reconciles and len(months) == 12
        monthly_reconciles = monthly_reconciles and math.isclose(
            _compound(months["gross_compounded_return"]),
            float(cell["gross_compounded_return"]),
            rel_tol=0.0,
            abs_tol=tolerance,
        )
        monthly_reconciles = monthly_reconciles and math.isclose(
            _compound(months["net_compounded_return"]),
            float(cell["net_compounded_return"]),
            rel_tol=0.0,
            abs_tol=tolerance,
        )
    registered_months = {
        date.strftime("%Y-%m") for date in expected_dates
    }
    registered_scopes = {
        "fold_1", "fold_2", "fold_3", "aggregate_equal_fold_capital"
    }
    registered_strategies = set(STRATEGY_WEIGHT_COLUMNS)
    expected_monthly_keys = {
        (month, cost, scope, strategy)
        for month in registered_months
        for cost in costs
        for scope in registered_scopes
        for strategy in registered_strategies
    }
    observed_monthly_keys = set(zip(
        monthly_metrics["calendar_month"],
        monthly_metrics["cost_bps"].astype(int),
        monthly_metrics["scope"],
        monthly_metrics["strategy"],
        strict=True,
    ))
    state_scopes = {
        "fold_1", "fold_2", "fold_3", "all_folds_equal_capital_contribution"
    }
    expected_state_keys = {
        (scope, state)
        for scope in state_scopes
        for state in autopsy["diagnostics"]["position_states"]
    }
    observed_state_keys = set(zip(
        position_state_metrics["scope"],
        position_state_metrics["position_state"],
        strict=True,
    ))
    regime_cells = [
        *( ("momentum", regime) for regime in ("risk_off", "risk_on") ),
        *(
            ("breadth", regime)
            for regime in ("cash_gate_closed", "narrow", "medium", "broad")
        ),
    ]
    expected_regime_keys = {
        (family, regime, scope, strategy)
        for family, regime in regime_cells
        for scope in state_scopes
        for strategy in registered_strategies
    }
    observed_regime_keys = set(zip(
        regime_metrics["regime_family"],
        regime_metrics["regime"],
        regime_metrics["scope"],
        regime_metrics["strategy"],
        strict=True,
    ))
    expected_cost_keys = {
        (cost, scope, strategy)
        for cost in costs
        for scope in registered_scopes
        for strategy in registered_strategies
    }
    observed_cost_keys = {
        (int(row["cost_bps"]), str(row["scope"]), str(row["strategy"]))
        for row in cost_cells
    }
    expected_asset_keys = {
        (fold, symbol)
        for fold, symbols in fold_symbols.items()
        for symbol in symbols
    }
    observed_asset_keys = set(zip(
        asset_fold_metrics["fold"].astype(int),
        asset_fold_metrics["symbol"],
        strict=True,
    ))
    v45_result = context["values"]["v45_result"]
    checks = {
        "preflight_packet_was_hash_exact_before_first_parquet_read": True,
        "frozen_input_schemas_are_exact": bool(all(schema_checks.values())),
        "frozen_input_dtypes_and_finiteness_are_exact": (
            dtype_and_finiteness_exact
        ),
        "context_count_is_exact": len(context_predictions)
        == int(data_contract["expected_contexts"]),
        "asset_date_count_is_exact": len(asset_predictions)
        == int(data_contract["expected_asset_dates"]),
        "outcome_count_is_exact": len(outcomes)
        == int(data_contract["expected_asset_dates"]),
        "position_row_count_is_exact": len(positions)
        == int(data_contract["expected_position_rows"]),
        "fold_date_count_is_exact": len(fold_dates)
        == int(data_contract["expected_fold_dates"]),
        "unique_date_count_is_exact": fold_dates["date"].nunique()
        == int(data_contract["expected_unique_dates"]),
        "non_target_symbol_count_is_exact": len(all_symbols)
        == int(data_contract["expected_non_target_symbols"]),
        "target_assets_are_absent": target_absence,
        "triplet_keys_match_lexical_symbols": context_keys_exact,
        "transformer_context_scores_are_centered": transformer_context_centered,
        "ridge_context_scores_are_centered": ridge_context_centered,
        "post_2025_observations_are_absent": post_2025_absence,
        "signal_schedule_is_exact": signal_schedule_exact,
        "target_maturity_schedule_is_exact": maturity_schedule_exact,
        "every_input_table_covers_the_complete_signal_calendar": (
            complete_signal_calendars
        ),
        "every_fold_covers_the_complete_signal_calendar": fold_calendars_exact,
        "context_metrics_reconcile_exactly": _frames_reconcile(
            persisted_context_metrics,
            recomputed_context_metrics,
            ["date", "fold", "triplet_key"],
            tolerance,
        ),
        "predictive_daily_metrics_reconcile_exactly": _frames_reconcile(
            persisted_predictive_daily,
            recomputed_predictive_daily,
            ["date", "fold"],
            tolerance,
        ),
        "predictive_summary_reconciles_v45_result": _canonical_sha256(
            _json_ready(predictive_summary)
        ) == _canonical_sha256(v45_result["predictive_metrics"]),
        "fold_metrics_reconcile_v45_result": _canonical_sha256(
            _json_ready(portfolio["fold_metrics"])
        ) == _canonical_sha256(v45_result["fold_metrics"]),
        "aggregate_metrics_reconcile_v45_result": _canonical_sha256(
            _json_ready(portfolio["aggregate_metrics"])
        ) == _canonical_sha256(v45_result["aggregate_metrics"]),
        **{f"ledger_{name}": bool(value) for name, value in ledger["checks"].items()},
        "candidate_weights_are_binary": bool(
            np.isin(positions["candidate_weight"].to_numpy(dtype=float), [0.0, 1.0]).all()
        ),
        "candidate_holds_at_most_one_asset": bool(
            (positions.groupby(["date", "fold"])["candidate_weight"].sum() <= 1.0).all()
        ),
        "candidate_turnover_matches_every_policy_transition": bool(
            np.allclose(
                fold_dates["candidate_turnover"],
                expected_turnover,
                rtol=0.0,
                atol=tolerance,
            )
        ),
        "eligible_position_keys_match_asset_predictions": bool(
            eligible_positions["_merge"].eq("both").all()
        ),
        "eligible_momentum_matches_asset_predictions": bool(
            eligible_positions["_merge"].eq("both").all()
            and np.allclose(
                eligible_positions["momentum_30_position"],
                eligible_positions["momentum_30_asset"],
                rtol=0.0,
                atol=tolerance,
            )
        ),
        "per_asset_context_count_formula_is_exact": bool(
            np.array_equal(
                expected_context_counts["context_count"].to_numpy(dtype=np.int64),
                context_count_formula,
            )
        ),
        "episode_active_days_reconcile": int(
            episodes["duration_signal_days"].sum()
        ) == int(fold_dates["candidate_active"].sum()),
        "episode_turnover_reconciles": math.isclose(
            float(2 * len(episodes)),
            float(fold_dates["candidate_turnover"].sum()),
            rel_tol=0.0,
            abs_tol=tolerance,
        ),
        "episode_turnover_reconciles_by_fold": turnover_by_fold_reconciles,
        "episode_cost_reconciles": math.isclose(
            float(episodes["allocated_cost"].sum()),
            float(fold_dates["candidate_cost"].sum()),
            rel_tol=0.0,
            abs_tol=tolerance,
        ),
        "episode_gross_additive_reconciles": math.isclose(
            float(episodes["gross_additive_return"].sum()),
            float(fold_dates["candidate_gross_return"].sum()),
            rel_tol=0.0,
            abs_tol=tolerance,
        ),
        "all_registered_assets_materialized_once": bool(
            observed_asset_keys == expected_asset_keys
            and len(asset_fold_metrics) == len(observed_asset_keys)
            and len(asset_fold_metrics)
            == int(data_contract["expected_non_target_symbols"])
        ),
        "ranking_daily_keys_are_complete_and_unique": bool(
            len(ranking_daily) == int(data_contract["expected_fold_dates"])
            and not ranking_daily.duplicated(["date", "fold"]).any()
        ),
        "context_stability_keys_are_complete_and_unique": bool(
            len(stability) == int(data_contract["expected_asset_dates"])
            and not stability.duplicated(["date", "fold", "symbol"]).any()
        ),
        "every_fold_has_exactly_ten_registered_symbols": all(
            len(fold_symbols[fold]) == 10 for fold in (1, 2, 3)
        ),
        "asset_net_contributions_reconcile_by_fold": (
            asset_contributions_reconcile_by_fold
        ),
        "asset_equal_fold_contributions_reconcile_aggregate": math.isclose(
            float(
                asset_fold_metrics[
                    "equal_fold_capital_net_contribution"
                ].sum()
            ),
            float(
                persisted_daily_returns.loc[
                    (persisted_daily_returns["cost_bps"] == base_cost)
                    & (
                        persisted_daily_returns["scope"]
                        == "aggregate_equal_fold_capital"
                    )
                    & (persisted_daily_returns["strategy"] == "candidate"),
                    "net_return",
                ].sum()
            ),
            rel_tol=0.0,
            abs_tol=tolerance,
        ),
        "held_days_reconcile_by_fold": held_days_reconcile_by_fold,
        "monthly_grid_is_complete_and_unique": bool(
            observed_monthly_keys == expected_monthly_keys
            and len(monthly_metrics) == len(observed_monthly_keys)
        ),
        "monthly_returns_reconcile_every_annual_cell": monthly_reconciles,
        "position_state_grid_is_complete_and_unique": bool(
            observed_state_keys == expected_state_keys
            and len(position_state_metrics) == len(observed_state_keys)
        ),
        "regime_grid_is_complete_and_unique": bool(
            observed_regime_keys == expected_regime_keys
            and len(regime_metrics) == len(observed_regime_keys)
        ),
        "cost_grid_is_complete_and_unique": bool(
            observed_cost_keys == expected_cost_keys
            and len(cost_cells) == len(observed_cost_keys)
        ),
        "cost_invariants_pass": all(
            cost_decomposition["invariant_checks"].values()
        ),
        "all_input_hashes_match_after_analysis": input_hashes_after
        == context["hashes"],
        "v45_retirement_decision_is_unchanged": v45_result["decision"]
        == autopsy["expected_lineage"]["v45_decision"],
        "zero_models_checkpoints_training_or_inference": True,
        "zero_counterfactual_pnl_or_policy_tuning": True,
    }
    audit = {"checks": checks, "passed": bool(all(checks.values()))}
    if not audit["passed"]:
        failed = [name for name, value in checks.items() if not value]
        raise RuntimeError(f"V46 autopsy audit failed: {failed}")

    input_receipt = {
        "version": "v46_run_input_hash_receipt_v1",
        "preflight_completion_receipt_sha256": _sha256_file(
            preflight_output / "completion_receipt.json"
        ),
        "hashes_before": context["hashes"],
        "hashes_after": input_hashes_after,
        "parquet_files_deserialized": len(TABLE_INPUTS),
        "model_or_checkpoint_files_opened": 0,
        "new_predictions_positions_or_bootstraps": 0,
    }
    result: dict[str, object] = {
        "version": VERSION,
        "mode": "run",
        "decision": "v45_retirement_confirmed_diagnostic_only",
        "autopsy_spec": spec,
        "input_hash_receipt": input_receipt,
        "summary": {
            "unique_dates": int(fold_dates["date"].nunique()),
            "fold_dates": int(len(fold_dates)),
            "asset_dates": int(len(asset_predictions)),
            "position_rows": int(len(positions)),
            "contexts": int(len(context_predictions)),
            "non_target_assets": int(len(all_symbols)),
            "holding_episodes": int(len(episodes)),
            "registered_gate_cells": int(
                context["values"]["v45_gate_result"]["cell_count"]
            ),
            "failed_gate_cells": int(len(context["failed_cells"])),
            "target_asset_rows": 0,
            "post_2025_rows": 0,
        },
        "ranking_summary": ranking_summary,
        "economic_diagnostics": economic,
        "context_stability": stability_summary,
        "concentration": concentration,
        "cost_decomposition": cost_decomposition,
        "drawdown_diagnostics": drawdowns,
        "failure_attribution": failure_attribution,
        "limitations": {
            **autopsy["limitations"],
            "v45_window_status": "consumed_development_evidence",
            "diagnostic_groups_are_not_candidate_filters": True,
        },
        "recommendation": {
            "enum": "new_ex_ante_family",
            "relative_ranking_plus_absolute_return_or_market_state": True,
            "ex_ante_concentration_control": True,
            "turnover_sensitive_objective": True,
            "requires_genuinely_new_non_target_observations": True,
            "target_assets": "remain_sealed",
        },
        "audit": audit,
    }

    output.mkdir(parents=True, exist_ok=True)
    parquet_outputs = {
        "fold_date_diagnostics.parquet": fold_dates,
        "asset_fold_metrics.parquet": asset_fold_metrics,
        "monthly_metrics.parquet": monthly_metrics,
        "holding_episodes.parquet": episodes,
        "position_state_metrics.parquet": position_state_metrics,
        "ranking_daily.parquet": ranking_daily,
        "context_stability.parquet": stability,
        "regime_metrics.parquet": regime_metrics,
    }
    for relative, frame in parquet_outputs.items():
        _write_parquet_atomic(frame, output / relative)
    _write_json_atomic(output / "autopsy_spec.json", spec)
    _write_json_atomic(output / "input_hash_receipt.json", input_receipt)
    _write_json_atomic(output / "ranking_summary.json", ranking_summary)
    _write_json_atomic(output / "economic_diagnostics.json", economic)
    _write_json_atomic(output / "context_stability.json", stability_summary)
    _write_json_atomic(output / "concentration.json", concentration)
    _write_json_atomic(output / "cost_decomposition.json", cost_decomposition)
    _write_json_atomic(output / "drawdown_diagnostics.json", drawdowns)
    _write_json_atomic(output / "failure_attribution.json", failure_attribution)
    _write_json_atomic(output / "audit.json", audit)
    _write_yaml_atomic(output / "resolved_config.yaml", config)
    _atomic_write_text(output / "report.md", _v46_report(result))
    core_files = [
        relative
        for relative in required
        if relative
        not in {"artifact_manifest.json", "completion_receipt.json", "result.json"}
    ]
    _seal_packet(output, result, core_files)
    return _validate_packet(
        output, str(spec["autopsy_spec_sha256"]), "run", required
    )
