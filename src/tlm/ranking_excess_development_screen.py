from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
import gc
import json
import math
import os
import platform
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import pyarrow
import torch
import yaml

from .monte_carlo import paired_block_bootstrap
from .non_target_pretraining import TripletTensorStore
from .ranking_excess_harness import (
    RANKING_EXCESS_HEADS,
    SharedAssetRidgeModel,
    aggregate_raw_excess_predictions,
    eligible_dual_momentum_positions,
    fit_shared_asset_ridge,
    momentum_gated_equal_weight_positions,
    normalized_triplet_excess,
    predict_shared_asset_ridge,
    ranking_excess_positions,
)
from .ranking_excess_pretraining import (
    TARGET_SYMBOLS,
    _availability_from_index,
    _eligible_pair_count,
    _serialize_filters,
)
from .ranking_excess_screen_metrics import (
    build_portfolio_evaluation,
    compute_predictive_metrics,
    evaluate_v45_gates,
    top1_excess_block_bootstrap,
)
from .ranking_excess_spec import (
    _canonical_sha256,
    _load_json,
    _sha256_file,
)
from .ranking_excess_supervised import (
    SupervisedFeatureLabelStore,
    load_ranking_excess_supervised_checkpoint,
)
from .scientific_harness import (
    DeterministicEligibleTripletSampler,
    FeatureScaler,
)


VERSION = "v45"
PREPARE_STRATEGIES = (
    "candidate",
    "dual_momentum_30",
    "momentum_gated_equal_weight",
)
METADATA_INPUT_NAMES = {
    "v41_specification",
    "v41_blueprint",
    "v41_audit",
    "v42_result",
    "v42_audit",
    "v44_result",
    "v44_audit",
    "v44_checkpoint_manifest",
    "v44_target_scales",
    "v44_supervised_spec",
    "v32_dataset_manifest",
    "v32_feature_schema",
    "v32_asset_folds",
    "v32_triplet_catalog",
    "v32_triplet_availability",
    "v32_source_audit",
    "v43_scaler_fold_1",
    "v43_scaler_fold_2",
    "v43_scaler_fold_3",
}
BINARY_INPUT_NAMES = {"panel", "sequence_index"}
EXPECTED_GRID = {(fold, seed) for fold in (1, 2, 3) for seed in (42, 7, 123)}
IMPLEMENTATION_SOURCE_FILES = (
    "monte_carlo.py",
    "non_target_pretraining.py",
    "patch_transformer.py",
    "ranking_excess_development_screen.py",
    "ranking_excess_harness.py",
    "ranking_excess_pretraining.py",
    "ranking_excess_screen_metrics.py",
    "ranking_excess_spec.py",
    "ranking_excess_supervised.py",
    "scientific_harness.py",
    "source_domain_one_shot.py",
    "supervised_non_target.py",
)


@dataclass
class FoldPrepareData:
    ridge_feature_panel: pd.DataFrame
    ridge_labels: pd.DataFrame
    ridge_availability: dict[pd.Timestamp, list[str]]
    screen_feature_panel: pd.DataFrame
    screen_availability: dict[pd.Timestamp, list[str]]
    audit: dict[str, object]
    receipts: list[dict[str, object]]


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


def _write_npy_atomic(values: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(values), allow_pickle=False)
    temporary.replace(path)


def _scaler_from_record(record: dict) -> FeatureScaler:
    names = {field.name for field in fields(FeatureScaler)}
    if not names.issubset(record):
        raise RuntimeError("V45 scaler record is incomplete")
    values = {name: record[name] for name in names}
    values["feature_names"] = tuple(str(name) for name in values["feature_names"])
    values["mean"] = tuple(float(value) for value in values["mean"])
    values["scale"] = tuple(float(value) for value in values["scale"])
    values["source_relative_feature_index"] = int(
        values["source_relative_feature_index"]
    )
    values["fit_rows"] = int(values["fit_rows"])
    scaler = FeatureScaler(**values)
    if (
        not np.isfinite(np.asarray(scaler.mean, dtype=np.float64)).all()
        or not np.isfinite(np.asarray(scaler.scale, dtype=np.float64)).all()
        or bool((np.asarray(scaler.scale, dtype=np.float64) <= 0).any())
    ):
        raise RuntimeError("V45 scaler contains invalid values")
    return scaler


def _implementation_provenance() -> dict[str, object]:
    package_root = Path(__file__).resolve().parent
    source_hashes = {}
    for relative in IMPLEMENTATION_SOURCE_FILES:
        path = package_root / relative
        if not path.is_file():
            raise RuntimeError(f"V45 implementation source is missing: {relative}")
        source_hashes[f"src/tlm/{relative}"] = _sha256_file(path)
    return {
        "source_sha256": source_hashes,
        "runtime_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "torch": torch.__version__,
            "pyyaml": yaml.__version__,
        },
    }


def _artifact_manifest(
    output: Path,
    files: Iterable[str],
    evaluation_spec_sha256: str,
) -> dict[str, object]:
    rows = []
    for relative in sorted(set(files)):
        path = output / relative
        if not path.is_file():
            raise RuntimeError(f"V45 artifact is missing before sealing: {relative}")
        rows.append({
            "path": relative,
            "bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
        })
    payload: dict[str, object] = {
        "version": "v45_artifact_manifest_v1",
        "evaluation_spec_sha256": evaluation_spec_sha256,
        "files": rows,
    }
    payload["manifest_semantic_sha256"] = _canonical_sha256(payload)
    return payload


def _seal_result_packet(
    output: Path,
    result: dict[str, object],
    core_files: Iterable[str],
) -> dict[str, object]:
    spec_sha = str(result["evaluation_spec"]["evaluation_spec_sha256"])
    manifest = _artifact_manifest(output, core_files, spec_sha)
    _write_json_atomic(output / "artifact_manifest.json", manifest)
    _write_json_atomic(output / "result.json", result)
    completion = {
        "version": "v45_completion_receipt_v1",
        "mode": result["mode"],
        "decision": result["decision"],
        "evaluation_spec_sha256": spec_sha,
        "artifact_manifest_sha256": _sha256_file(
            output / "artifact_manifest.json"
        ),
        "result_sha256": _sha256_file(output / "result.json"),
    }
    _write_json_atomic(output / "completion_receipt.json", completion)
    return completion


def _validate_result_packet(
    output: Path,
    expected_spec_sha256: str,
    expected_mode: str,
    required_files: Iterable[str],
) -> dict[str, object]:
    required_paths = tuple(str(relative) for relative in required_files)
    if len(required_paths) != len(set(required_paths)):
        raise RuntimeError(f"V45 cached {expected_mode} required file grid drift")
    for relative in required_paths:
        if not (output / relative).is_file():
            raise RuntimeError(f"V45 cached {expected_mode} packet is incomplete")
    completion = _load_json(output / "completion_receipt.json")
    if (
        completion.get("version") != "v45_completion_receipt_v1"
        or completion.get("mode") != expected_mode
        or completion.get("evaluation_spec_sha256") != expected_spec_sha256
        or completion.get("result_sha256") != _sha256_file(output / "result.json")
        or completion.get("artifact_manifest_sha256")
        != _sha256_file(output / "artifact_manifest.json")
    ):
        raise RuntimeError(f"V45 cached {expected_mode} completion receipt drift")
    manifest = _load_json(output / "artifact_manifest.json")
    semantic = dict(manifest)
    claimed_semantic = semantic.pop("manifest_semantic_sha256", None)
    if (
        manifest.get("version") != "v45_artifact_manifest_v1"
        or manifest.get("evaluation_spec_sha256") != expected_spec_sha256
        or claimed_semantic != _canonical_sha256(semantic)
    ):
        raise RuntimeError(f"V45 cached {expected_mode} manifest drift")
    manifest_rows = manifest.get("files", [])
    if not isinstance(manifest_rows, list):
        raise RuntimeError(f"V45 cached {expected_mode} manifest file grid drift")
    envelope_files = {
        "artifact_manifest.json",
        "completion_receipt.json",
        "result.json",
    }
    expected_manifest_paths = set(required_paths) - envelope_files
    observed_manifest_paths = [str(row.get("path", "")) for row in manifest_rows]
    if (
        len(observed_manifest_paths) != len(set(observed_manifest_paths))
        or set(observed_manifest_paths) != expected_manifest_paths
    ):
        raise RuntimeError(f"V45 cached {expected_mode} manifest file grid drift")
    for row in manifest_rows:
        relative = str(row.get("path", ""))
        path = (output / relative).resolve()
        if not path.is_relative_to(output.resolve()):
            raise RuntimeError("V45 manifest path escaped its output directory")
        if (
            not path.is_file()
            or int(row.get("bytes", -1)) != path.stat().st_size
            or row.get("sha256") != _sha256_file(path)
        ):
            raise RuntimeError(f"V45 cached artifact drift: {relative}")
    result = _load_json(output / "result.json")
    if (
        result.get("mode") != expected_mode
        or result.get("evaluation_spec", {}).get("evaluation_spec_sha256")
        != expected_spec_sha256
        or not result.get("audit", {}).get("passed")
    ):
        raise RuntimeError(f"V45 cached {expected_mode} result drift")
    return result


def build_evaluation_spec(
    config: dict,
    blueprint: dict,
) -> dict[str, object]:
    screen = config["ranking_excess_screen"]
    spec: dict[str, object] = {
        "version": VERSION,
        "phase": "asset_disjoint_ranking_excess_development_screen",
        "candidate_family_id": blueprint["candidate_family_id"],
        "v41_blueprint_sha256": blueprint["blueprint_sha256"],
        "resolved_config_semantic_sha256": _canonical_sha256(config),
        "expected_input_sha256": screen["expected_input_sha256"],
        "expected_checkpoints": screen["expected_checkpoints"],
        "expected_scalers": screen["expected_scalers"],
        "expected_target_scales": screen["expected_target_scales"],
        "implementation_provenance": _implementation_provenance(),
        "runtime": {
            name: screen[name]
            for name in (
                "device",
                "torch_threads",
                "dtype",
                "deterministic_algorithms",
                "amp",
                "cpu_fallback_allowed",
                "inference_batch_size",
            )
        },
        "ridge": screen["ridge"],
        "data_access": screen["data_access"],
        "inference": screen["inference"],
        "predictive_metrics": screen["predictive_metrics"],
        "policy": screen["policy"],
        "accounting": screen["accounting"],
        "bootstrap": screen["bootstrap"],
        "gates": screen["gates"],
        "lifecycle": screen["lifecycle"],
        "artifact_contract": screen["artifact_contract"],
    }
    spec["evaluation_spec_sha256"] = _canonical_sha256(spec)
    return spec


def _validate_ready_schedule(
    fold_entry: dict,
    expected: dict,
    source_last_date: dict[str, pd.Timestamp],
    signal_start: pd.Timestamp,
    signal_end: pd.Timestamp,
) -> dict[str, int]:
    schedule: dict[pd.Timestamp, tuple[str, ...]] = {}
    for segment in expected["ready_segments"]:
        start = pd.Timestamp(segment["start"], tz="UTC")
        end = pd.Timestamp(segment["end"], tz="UTC")
        absent = tuple(sorted(str(symbol) for symbol in segment["absent"]))
        if start < signal_start or end > signal_end or start > end:
            raise RuntimeError("V45 configured readiness segment is outside screen")
        for date in pd.date_range(start, end, freq="D", tz="UTC"):
            if date in schedule:
                raise RuntimeError("V45 readiness segments overlap")
            schedule[date] = absent
            if int(segment["ready_assets"]) != len(fold_entry["test_symbols"]) - len(
                absent
            ):
                raise RuntimeError("V45 readiness segment count drift")
    expected_dates = pd.date_range(signal_start, signal_end, freq="D", tz="UTC")
    if set(schedule) != set(expected_dates):
        raise RuntimeError("V45 readiness segments do not cover the exact screen")
    asset_dates = 0
    contexts = 0
    for date in expected_dates:
        derived_absent = tuple(sorted(
            symbol
            for symbol in fold_entry["test_symbols"]
            if date > source_last_date[symbol] - pd.Timedelta(days=8)
        ))
        if schedule[date] != derived_absent:
            raise RuntimeError("V45 readiness schedule disagrees with source audit")
        ready = len(fold_entry["test_symbols"]) - len(derived_absent)
        asset_dates += ready
        contexts += math.comb(ready, 3)
    if (
        len(expected_dates) != int(expected["screen_signal_dates"])
        or asset_dates != int(expected["screen_asset_date_rows"])
        or contexts != int(expected["screen_triplet_contexts"])
    ):
        raise RuntimeError("V45 readiness schedule arithmetic drift")
    return {
        "signal_dates": len(expected_dates),
        "asset_dates": asset_dates,
        "triplet_contexts": contexts,
    }


def _metadata_context(
    config: dict,
    *,
    reopen_checkpoints: bool,
) -> dict[str, object]:
    if "ranking_excess_screen" not in config:
        raise RuntimeError("V45 config section is missing")
    screen = config["ranking_excess_screen"]
    root = Path(screen["project_root"]).resolve()
    paths = {
        name: (root / relative).resolve()
        for name, relative in screen["inputs"].items()
    }
    expected_names = METADATA_INPUT_NAMES | BINARY_INPUT_NAMES
    if set(paths) != expected_names or set(screen["expected_input_sha256"]) != expected_names:
        raise RuntimeError("V45 input allowlist drift")
    input_hashes = {}
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"V45 input is missing: {name}")
        observed = _sha256_file(path)
        expected = screen["expected_input_sha256"][name]
        input_hashes[name] = observed
        if observed != expected:
            raise RuntimeError(f"V45 input hash drift: {name}")
    values = {name: _load_json(paths[name]) for name in METADATA_INPUT_NAMES}
    blueprint = values["v41_blueprint"]
    v44_result = values["v44_result"]
    if (
        values["v41_specification"].get("decision")
        != "authorize_v42_synthetic_ranking_excess_harness_only"
        or not values["v41_audit"].get("passed")
        or values["v42_result"].get("decision")
        != "authorize_v43_medium_non_target_pretraining_only"
        or not values["v42_audit"].get("passed")
        or v44_result.get("decision")
        != "v45_asset_disjoint_2025_development_screen_only"
        or not values["v44_audit"].get("passed")
        or not v44_result.get("audit", {}).get("passed")
        or v44_result.get("supervised_spec", {}).get("mode") != "full"
        or int(v44_result.get("summary", {}).get("checkpoint_count", -1)) != 9
    ):
        raise RuntimeError("V41/V42/V44 lineage does not authorize V45")
    if (
        blueprint.get("candidate_family_id")
        != "tlm_cross_sectional_rank_excess_medium_v1"
        or blueprint.get("blueprint_sha256")
        != "dc28004a9419424f6d9e437b9ac8a7bf42f73ec9ceb1892494e280d9240fdf5e"
        or blueprint.get("target_contract", {}).get("target_data_allowed")
        or set(blueprint.get("target_contract", {}).get("target_symbols", []))
        != TARGET_SYMBOLS
    ):
        raise RuntimeError("V45 blueprint identity or target seal drift")

    manifest = values["v32_dataset_manifest"]
    feature_schema = values["v32_feature_schema"]
    data_access = screen["data_access"]
    feature_names = list(manifest["panel_features"])
    if (
        manifest["panel_sha256"] != input_hashes["panel"]
        or manifest["sequence_index_sha256"] != input_hashes["sequence_index"]
        or list(feature_schema["model_feature_order"][:-1]) != feature_names
        or list(data_access["ridge_feature_columns"])
        != ["date", "symbol", *feature_names]
        or list(data_access["screen_feature_columns"])
        != ["date", "symbol", *feature_names]
        or list(data_access["ridge_label_columns"])
        != ["date", "symbol", "target_next_open_to_next_open_log_return"]
        or list(data_access["screen_label_columns"])
        != [
            "date",
            "symbol",
            "target_window_end_date",
            "target_next_open_to_next_open_log_return",
        ]
        or list(data_access["sequence_columns"])
        != ["date", "sequence_start_date", "symbol"]
        or TARGET_SYMBOLS.intersection(manifest["symbols"])
    ):
        raise RuntimeError("V45 dataset projection, hash, or target seal drift")

    asset_folds = values["v32_asset_folds"]
    triplet_catalog = values["v32_triplet_catalog"]
    if len(asset_folds.get("folds", [])) != 3 or len(triplet_catalog.get("folds", [])) != 3:
        raise RuntimeError("V45 requires exactly three registered folds")
    folds = {int(row["fold"]): row for row in asset_folds["folds"]}
    catalog_folds = {int(row["fold"]): row for row in triplet_catalog["folds"]}
    all_test_symbols: list[str] = []
    for fold in (1, 2, 3):
        fold_entry = folds[fold]
        catalog_entry = catalog_folds[fold]
        train = sorted(fold_entry["train_symbols"])
        test = sorted(fold_entry["test_symbols"])
        expected_test_triplets = [list(row) for row in combinations(test, 3)]
        expected_train_triplets = [list(row) for row in combinations(train, 3)]
        if (
            len(train) != 20
            or len(test) != 10
            or set(train).intersection(test)
            or set(train).union(test) != set(manifest["symbols"])
            or catalog_entry["train_symbols"] != train
            or catalog_entry["test_symbols"] != test
            or catalog_entry["train_triplets"] != expected_train_triplets
            or catalog_entry["test_triplets"] != expected_test_triplets
        ):
            raise RuntimeError(f"V45 fold/catalog drift: {fold}")
        all_test_symbols.extend(test)
    if sorted(all_test_symbols) != sorted(manifest["symbols"]):
        raise RuntimeError("V45 test folds are not an exact asset partition")

    source_rows = values["v32_source_audit"].get("per_symbol", [])
    source_last_date = {
        str(row["symbol"]): pd.Timestamp(row["last_date"], tz="UTC")
        for row in source_rows
    }
    if set(source_last_date) != set(manifest["symbols"]):
        raise RuntimeError("V45 source audit symbol coverage drift")
    signal_start = pd.Timestamp(data_access["screen_signal_start"], tz="UTC")
    signal_end = pd.Timestamp(data_access["screen_signal_end"], tz="UTC")
    schedule_audits = []
    for fold in (1, 2, 3):
        schedule_audits.append({
            "fold": fold,
            **_validate_ready_schedule(
                folds[fold],
                data_access["expected_by_fold"][str(fold)],
                source_last_date,
                signal_start,
                signal_end,
            ),
        })
    if (
        sum(int(row["asset_dates"]) for row in schedule_audits)
        != int(data_access["expected_total_asset_dates"])
        or sum(int(row["triplet_contexts"]) for row in schedule_audits)
        != int(data_access["expected_total_triplet_contexts"])
        or int(data_access["expected_total_seed_context_forwards"])
        != 3 * int(data_access["expected_total_triplet_contexts"])
        or int(data_access["expected_unique_signal_dates"]) != len(
            pd.date_range(signal_start, signal_end, freq="D", tz="UTC")
        )
        or int(data_access["expected_total_fold_signal_dates"])
        != 3 * int(data_access["expected_unique_signal_dates"])
    ):
        raise RuntimeError("V45 aggregate schedule counts drift")

    scalers: dict[int, FeatureScaler] = {}
    scaler_receipt = []
    for fold in (1, 2, 3):
        name = f"v43_scaler_fold_{fold}"
        record = values[name]
        scaler = _scaler_from_record(record)
        expected = screen["expected_scalers"][str(fold)]
        if (
            input_hashes[name] != expected["artifact_sha256"]
            or scaler.state_sha256() != expected["state_sha256"]
            or list(scaler.feature_names) != feature_names
            or scaler.source_relative_feature_index
            != feature_names.index("log_close_to_close_return")
            or scaler.fit_start != "2021-01-01"
            or scaler.fit_end != "2023-12-31"
        ):
            raise RuntimeError(f"V45 immutable scaler drift: fold {fold}")
        scalers[fold] = scaler
        scaler_receipt.append({
            "fold": fold,
            "artifact_sha256": input_hashes[name],
            "state_sha256": scaler.state_sha256(),
        })

    target_scale_rows = values["v44_target_scales"]
    if not isinstance(target_scale_rows, list) or len(target_scale_rows) != 3:
        raise RuntimeError("V45 target scale manifest drift")
    target_scales: dict[int, dict[str, object]] = {}
    for row in target_scale_rows:
        fold = int(row["fold"])
        expected = screen["expected_target_scales"][str(fold)]
        if (
            fold in target_scales
            or float(row["excess_rms_scale"]) != float(expected["value"])
            or row["target_scale_state_sha256"] != expected["state_sha256"]
            or int(row["eligible_dates"]) != 802
            or int(row["enumerated_triplets"]) != 914_280
            or row["fit_end"] != "2023-12-23"
        ):
            raise RuntimeError(f"V45 target scale drift: fold {fold}")
        target_scales[fold] = row
    if set(target_scales) != {1, 2, 3}:
        raise RuntimeError("V45 target scale fold grid drift")

    checkpoint_rows = values["v44_checkpoint_manifest"]
    if not isinstance(checkpoint_rows, list) or len(checkpoint_rows) != 9:
        raise RuntimeError("V45 checkpoint manifest cardinality drift")
    checkpoint_by_key: dict[tuple[int, int], dict[str, object]] = {}
    checkpoint_receipt = []
    checkpoint_root = (
        root / "data/checkpoints/v44_ranking_excess_supervised"
    ).resolve()
    v44_spec = values["v44_supervised_spec"]
    for row in checkpoint_rows:
        fold = int(row["fold"])
        seed = int(row["seed"])
        key = (fold, seed)
        key_text = f"{fold}:{seed}"
        path = Path(row["checkpoint_path"]).resolve()
        canonical = checkpoint_root / f"fold_{fold}" / f"seed_{seed}" / "checkpoint.pt"
        if (
            key not in EXPECTED_GRID
            or key in checkpoint_by_key
            or path != canonical
            or not path.is_relative_to(checkpoint_root)
            or not path.is_file()
            or row["checkpoint_sha256"] != screen["expected_checkpoints"][key_text]
            or _sha256_file(path) != row["checkpoint_sha256"]
        ):
            raise RuntimeError(f"V45 checkpoint file/grid drift: {key_text}")
        fold_entry = folds[fold]
        target_expected = screen["expected_target_scales"][str(fold)]
        scaler_expected = screen["expected_scalers"][str(fold)]
        receipt = {
            "fold": fold,
            "seed": seed,
            "checkpoint_path": str(path),
            "checkpoint_sha256": row["checkpoint_sha256"],
            "model_state_sha256": row["model_state_sha256"],
            "scaler_state_sha256": row["scaler_state_sha256"],
            "target_scale": row["target_scale"],
            "target_scale_state_sha256": row["target_scale_state_sha256"],
        }
        if (
            row["train_symbols"] != sorted(fold_entry["train_symbols"])
            or row["test_symbols"] != sorted(fold_entry["test_symbols"])
            or row["scaler_state_sha256"] != scaler_expected["state_sha256"]
            or float(row["target_scale"]) != float(target_expected["value"])
            or row["target_scale_state_sha256"] != target_expected["state_sha256"]
        ):
            raise RuntimeError(f"V45 checkpoint manifest semantics drift: {key_text}")
        if reopen_checkpoints:
            model, payload = load_ranking_excess_supervised_checkpoint(
                path,
                expected_architecture=blueprint["architecture"],
            )
            metadata = payload["metadata"]
            if (
                payload["model_state_sha256"] != row["model_state_sha256"]
                or int(metadata.get("fold", -1)) != fold
                or int(metadata.get("initialization_seed", -1)) != seed
                or metadata.get("candidate_family_id")
                != blueprint["candidate_family_id"]
                or metadata.get("supervised_spec_sha256")
                != v44_spec["supervised_spec_sha256"]
                or metadata.get("checkpoint_status")
                != "frozen_supervised_no_seed_or_fold_selection"
                or metadata.get("train_symbols")
                != sorted(fold_entry["train_symbols"])
                or metadata.get("test_symbols")
                != sorted(fold_entry["test_symbols"])
                or metadata.get("scaler_state_sha256")
                != scaler_expected["state_sha256"]
                or metadata.get("target_scale_state_sha256")
                != target_expected["state_sha256"]
                or float(metadata.get("target_scale", math.nan))
                != float(target_expected["value"])
            ):
                raise RuntimeError(f"V45 checkpoint payload semantics drift: {key_text}")
            model.eval()
            model.requires_grad_(False)
            if any(parameter.requires_grad for parameter in model.parameters()):
                raise RuntimeError("V45 could not freeze a preflight checkpoint")
            receipt["checkpoint_semantically_reopened"] = True
            del model, payload
            gc.collect()
        else:
            receipt["checkpoint_semantically_reopened"] = False
        checkpoint_receipt.append(receipt)
        checkpoint_by_key[key] = {**row, "resolved_path": str(path)}
    if set(checkpoint_by_key) != EXPECTED_GRID:
        raise RuntimeError("V45 checkpoint grid is incomplete")

    evaluation_spec = build_evaluation_spec(config, blueprint)
    return {
        "root": root,
        "paths": paths,
        "values": values,
        "blueprint": blueprint,
        "evaluation_spec": evaluation_spec,
        "input_hashes": input_hashes,
        "feature_names": feature_names,
        "folds": folds,
        "catalog_folds": catalog_folds,
        "scalers": scalers,
        "scaler_receipt": scaler_receipt,
        "target_scales": target_scales,
        "checkpoint_by_key": checkpoint_by_key,
        "checkpoint_receipt": checkpoint_receipt,
        "schedule_audits": schedule_audits,
    }


def _preflight_report(result: dict[str, object]) -> str:
    return "\n".join([
        "# TLM v45 Ranking/Excess Development Screen Preflight",
        "",
        "## Decision",
        "",
        "**PREFLIGHT PASSED; LABEL-FREE PREDICTION PREPARATION IS AUTHORIZED.**",
        "",
        f"Evaluation-spec SHA-256: `{result['evaluation_spec']['evaluation_spec_sha256']}`",
        "Checkpoint files semantically reopened: **9**",
        "Parquet files deserialized: **0**",
        "Held-out outcome rows read: **0**",
        "",
        "The panel and sequence files were raw-byte hashed only. No table value, prediction, policy return, performance metric, or PnL was computed.",
        "",
    ])


def preflight_ranking_excess_development_screen(
    config: dict,
) -> dict[str, object]:
    context = _metadata_context(config, reopen_checkpoints=True)
    screen = config["ranking_excess_screen"]
    output = context["root"] / screen["preflight_output_dir"]
    required = screen["artifact_contract"]["preflight"]["required_files"]
    spec = context["evaluation_spec"]
    if (output / "completion_receipt.json").is_file():
        return _validate_result_packet(
            output,
            spec["evaluation_spec_sha256"],
            "preflight",
            required,
        )
    output.mkdir(parents=True, exist_ok=True)
    input_receipt = {
        "version": "v45_input_hash_receipt_v1",
        "hashes": context["input_hashes"],
        "panel_and_sequence_raw_bytes_hashed": True,
        "parquet_deserializations": 0,
    }
    checkpoint_receipt = {
        "version": "v45_checkpoint_receipt_v1",
        "checkpoints": context["checkpoint_receipt"],
        "checkpoint_count": len(context["checkpoint_receipt"]),
        "seed_or_fold_selection": False,
    }
    checks = {
        "all_input_file_hashes_match": True,
        "lineage_authorizes_only_v45": True,
        "target_assets_remain_sealed": True,
        "forbidden_2026_window_remains_sealed": True,
        "three_asset_disjoint_folds_and_catalogs_are_exact": True,
        "nine_checkpoint_files_and_payloads_are_exact": True,
        "three_scalers_and_target_scales_are_exact": True,
        "readiness_schedule_and_counts_are_exact": True,
        "panel_and_sequence_were_raw_byte_hashed_only": True,
        "zero_parquet_deserializations": True,
        "zero_heldout_outcomes": True,
        "zero_predictions_positions_metrics_or_pnl": True,
    }
    audit = {"checks": checks, "passed": bool(all(checks.values()))}
    result: dict[str, object] = {
        "version": VERSION,
        "mode": "preflight",
        "decision": "authorize_v45_prepare_without_heldout_outcomes_only",
        "evaluation_spec": spec,
        "input_hash_receipt": input_receipt,
        "checkpoint_receipt": checkpoint_receipt,
        "schedule_audits": context["schedule_audits"],
        "summary": {
            "metadata_json_files_loaded": len(METADATA_INPUT_NAMES),
            "binary_files_raw_byte_hashed": len(BINARY_INPUT_NAMES),
            "parquet_files_deserialized": 0,
            "checkpoint_files_semantically_reopened": 9,
            "model_forwards": 0,
            "ridge_fits": 0,
            "heldout_outcome_rows": 0,
            "performance_metrics": 0,
            "pnl_evaluations": 0,
            "target_asset_rows": 0,
        },
        "audit": audit,
    }
    _write_json_atomic(output / "evaluation_spec.json", spec)
    _write_json_atomic(output / "input_hash_receipt.json", input_receipt)
    _write_json_atomic(output / "checkpoint_receipt.json", checkpoint_receipt)
    _write_json_atomic(output / "audit.json", audit)
    _write_yaml_atomic(output / "resolved_config.yaml", config)
    _atomic_write_text(output / "report.md", _preflight_report(result))
    _seal_result_packet(
        output,
        result,
        (
            "evaluation_spec.json",
            "input_hash_receipt.json",
            "checkpoint_receipt.json",
            "audit.json",
            "resolved_config.yaml",
            "report.md",
        ),
    )
    return _validate_result_packet(
        output,
        spec["evaluation_spec_sha256"],
        "preflight",
        required,
    )


def _validate_projected_frame(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    label: str,
) -> pd.DataFrame:
    if list(frame.columns) != columns:
        raise RuntimeError(f"V45 {label} projection drift")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], utc=True)
    result["symbol"] = result["symbol"].astype(str)
    if result.duplicated(["date", "symbol"]).any():
        raise RuntimeError(f"V45 {label} contains duplicate keys")
    return result


def _validate_daily_cartesian_feature_panel(
    frame: pd.DataFrame,
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    label: str,
) -> None:
    expected_dates = pd.date_range(start, end, freq="D", tz="UTC")
    if (
        set(frame["symbol"].unique()) != set(symbols)
        or len(frame) != len(symbols) * len(expected_dates)
        or frame["date"].min() != start
        or frame["date"].max() != end
    ):
        raise RuntimeError(f"V45 {label} symbol/date/cartesian coverage drift")
    for symbol, current in frame.groupby("symbol", sort=True):
        observed = pd.DatetimeIndex(current["date"].sort_values().unique())
        if not observed.equals(expected_dates):
            raise RuntimeError(f"V45 {label} compressed calendar for {symbol}")


def _availability_segments(
    availability: dict[pd.Timestamp, list[str]],
    all_symbols: list[str],
) -> list[dict[str, object]]:
    dates = sorted(availability)
    if not dates:
        return []
    segments: list[dict[str, object]] = []
    start = dates[0]
    prior = dates[0]
    prior_ready = tuple(sorted(availability[dates[0]]))
    for date in dates[1:]:
        ready = tuple(sorted(availability[date]))
        if ready != prior_ready or date != prior + pd.Timedelta(days=1):
            segments.append({
                "start": start.date().isoformat(),
                "end": prior.date().isoformat(),
                "ready_assets": len(prior_ready),
                "absent": sorted(set(all_symbols) - set(prior_ready)),
            })
            start = date
            prior_ready = ready
        prior = date
    segments.append({
        "start": start.date().isoformat(),
        "end": prior.date().isoformat(),
        "ready_assets": len(prior_ready),
        "absent": sorted(set(all_symbols) - set(prior_ready)),
    })
    return segments


def read_fold_prepare_data(
    panel_path: Path,
    sequence_path: Path,
    fold_entry: dict,
    data_access: dict,
    *,
    reader: Callable[..., pd.DataFrame] = pd.read_parquet,
) -> FoldPrepareData:
    fold = int(fold_entry["fold"])
    train_symbols = sorted(str(symbol) for symbol in fold_entry["train_symbols"])
    test_symbols = sorted(str(symbol) for symbol in fold_entry["test_symbols"])
    if (
        len(train_symbols) != 20
        or len(test_symbols) != 10
        or set(train_symbols).intersection(test_symbols)
        or TARGET_SYMBOLS.intersection(train_symbols + test_symbols)
    ):
        raise RuntimeError("V45 prepare fold identity drift")

    ridge_feature_start = pd.Timestamp(data_access["ridge_feature_start"], tz="UTC")
    ridge_feature_end = pd.Timestamp(data_access["ridge_feature_end"], tz="UTC")
    ridge_signal_start = pd.Timestamp(data_access["ridge_signal_start"], tz="UTC")
    ridge_signal_end = pd.Timestamp(data_access["ridge_signal_end"], tz="UTC")
    screen_feature_start = pd.Timestamp(data_access["screen_feature_start"], tz="UTC")
    screen_feature_end = pd.Timestamp(data_access["screen_feature_end"], tz="UTC")
    screen_signal_start = pd.Timestamp(data_access["screen_signal_start"], tz="UTC")
    screen_signal_end = pd.Timestamp(data_access["screen_signal_end"], tz="UTC")
    ridge_feature_columns = list(data_access["ridge_feature_columns"])
    ridge_label_columns = list(data_access["ridge_label_columns"])
    screen_feature_columns = list(data_access["screen_feature_columns"])
    sequence_columns = list(data_access["sequence_columns"])
    readiness = list(data_access["readiness_flags"])

    ridge_feature_filters = [
        ("symbol", "in", train_symbols),
        ("date", ">=", ridge_feature_start),
        ("date", "<=", ridge_feature_end),
    ]
    ridge_signal_filters = [
        ("symbol", "in", train_symbols),
        (data_access["ridge_split_flag"], "==", True),
        *[(name, "==", True) for name in readiness],
        ("date", ">=", ridge_signal_start),
        ("date", "<=", ridge_signal_end),
    ]
    screen_feature_filters = [
        ("symbol", "in", test_symbols),
        ("date", ">=", screen_feature_start),
        ("date", "<=", screen_feature_end),
    ]
    screen_sequence_filters = [
        ("symbol", "in", test_symbols),
        (data_access["screen_split_flag"], "==", True),
        *[(name, "==", True) for name in readiness],
        ("date", ">=", screen_signal_start),
        ("date", "<=", screen_signal_end),
    ]

    ridge_features = reader(
        panel_path,
        engine="pyarrow",
        columns=ridge_feature_columns,
        filters=ridge_feature_filters,
    )
    ridge_labels = reader(
        panel_path,
        engine="pyarrow",
        columns=ridge_label_columns,
        filters=ridge_signal_filters,
    )
    ridge_sequence = reader(
        sequence_path,
        engine="pyarrow",
        columns=sequence_columns,
        filters=ridge_signal_filters,
    )
    screen_features = reader(
        panel_path,
        engine="pyarrow",
        columns=screen_feature_columns,
        filters=screen_feature_filters,
    )
    screen_sequence = reader(
        sequence_path,
        engine="pyarrow",
        columns=sequence_columns,
        filters=screen_sequence_filters,
    )
    receipts = [
        {
            "fold": fold,
            "dataset": name,
            "columns": columns,
            "filters": _serialize_filters(filters),
        }
        for name, columns, filters in (
            ("ridge_features", ridge_feature_columns, ridge_feature_filters),
            ("ridge_labels", ridge_label_columns, ridge_signal_filters),
            ("ridge_sequence", sequence_columns, ridge_signal_filters),
            ("screen_features", screen_feature_columns, screen_feature_filters),
            ("screen_sequence", sequence_columns, screen_sequence_filters),
        )
    ]
    ridge_features = _validate_projected_frame(
        ridge_features, ridge_feature_columns, label="ridge feature panel"
    )
    ridge_labels = _validate_projected_frame(
        ridge_labels, ridge_label_columns, label="ridge action-return labels"
    )
    ridge_sequence = _validate_projected_frame(
        ridge_sequence, sequence_columns, label="ridge sequence index"
    )
    screen_features = _validate_projected_frame(
        screen_features, screen_feature_columns, label="screen feature panel"
    )
    screen_sequence = _validate_projected_frame(
        screen_sequence, sequence_columns, label="screen sequence index"
    )

    _validate_daily_cartesian_feature_panel(
        ridge_features,
        train_symbols,
        ridge_feature_start,
        ridge_feature_end,
        label="ridge feature panel",
    )
    _validate_daily_cartesian_feature_panel(
        screen_features,
        test_symbols,
        screen_feature_start,
        screen_feature_end,
        label="screen feature panel",
    )
    if (
        set(ridge_labels["symbol"].unique()) != set(train_symbols)
        or set(ridge_sequence["symbol"].unique()) != set(train_symbols)
        or not set(screen_sequence["symbol"].unique()).issubset(test_symbols)
        or set(screen_sequence["symbol"].unique()).intersection(train_symbols)
        or TARGET_SYMBOLS.intersection(screen_sequence["symbol"].unique())
    ):
        raise RuntimeError("V45 prepare reader ignored a symbol filter")
    if (
        ridge_labels["date"].min() < ridge_signal_start
        or ridge_labels["date"].max() > ridge_signal_end
        or ridge_sequence["date"].min() < ridge_signal_start
        or ridge_sequence["date"].max() > ridge_signal_end
        or screen_sequence["date"].min() < screen_signal_start
        or screen_sequence["date"].max() > screen_signal_end
    ):
        raise RuntimeError("V45 prepare reader ignored a date filter")
    ridge_values = ridge_labels[
        "target_next_open_to_next_open_log_return"
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(ridge_values).all():
        raise RuntimeError("V45 Ridge train-only labels are non-finite")
    for frame, label in (
        (ridge_sequence, "ridge"),
        (screen_sequence, "screen"),
    ):
        starts = pd.to_datetime(frame["sequence_start_date"], utc=True)
        if not bool((starts == frame["date"] - pd.Timedelta(days=255)).all()):
            raise RuntimeError(f"V45 {label} sequence lookback drift")
    ridge_keys = set(zip(ridge_labels["date"], ridge_labels["symbol"], strict=True))
    ridge_sequence_keys = set(zip(
        ridge_sequence["date"], ridge_sequence["symbol"], strict=True
    ))
    if ridge_keys != ridge_sequence_keys:
        raise RuntimeError("V45 Ridge label/sequence keys drift")

    ridge_availability = _availability_from_index(ridge_sequence)
    screen_availability = _availability_from_index(screen_sequence)
    expected = data_access["expected_by_fold"][str(fold)]
    observed = {
        "ridge_feature_rows": len(ridge_features),
        "ridge_label_rows": len(ridge_labels),
        "ridge_sequence_rows": len(ridge_sequence),
        "ridge_eligible_pairs": _eligible_pair_count(ridge_availability),
        "ridge_eligible_dates": len(ridge_availability),
        "ridge_first_ready_signal": min(ridge_availability).date().isoformat(),
        "ridge_last_ready_signal": max(ridge_availability).date().isoformat(),
        "screen_feature_rows": len(screen_features),
        "screen_signal_dates": len(screen_availability),
        "screen_asset_date_rows": sum(
            len(symbols) for symbols in screen_availability.values()
        ),
        "screen_sequence_rows": len(screen_sequence),
        "screen_triplet_contexts": _eligible_pair_count(screen_availability),
        "minimum_ready_assets": min(
            len(symbols) for symbols in screen_availability.values()
        ),
        "maximum_ready_assets": max(
            len(symbols) for symbols in screen_availability.values()
        ),
        "ready_segments": _availability_segments(
            screen_availability, test_symbols
        ),
    }
    expected_prepare = {
        name: expected[name]
        for name in observed
    }
    if observed != expected_prepare:
        raise RuntimeError(
            f"V45 fold {fold} prepare counts/schedule drift: {observed}"
        )
    expected_screen_dates = pd.date_range(
        screen_signal_start, screen_signal_end, freq="D", tz="UTC"
    )
    if set(screen_availability) != set(expected_screen_dates):
        raise RuntimeError("V45 screen sequence has a missing or extra signal date")
    feature_names = ridge_feature_columns[2:]
    for date, symbols in ridge_availability.items():
        if len(symbols) < 3:
            raise RuntimeError("V45 Ridge availability has fewer than three assets")
    # Prove every eligible lookback is finite without opening a held-out outcome.
    for feature_panel, availability, role_symbols, label in (
        (ridge_features, ridge_availability, train_symbols, "ridge"),
        (screen_features, screen_availability, test_symbols, "screen"),
    ):
        store = TripletTensorStore(
            feature_panel[["date", "symbol", *feature_names]],
            feature_names,
            256,
            "log_close_to_close_return",
        )
        if not store.dates.equals(pd.date_range(
            store.dates.min(), store.dates.max(), freq="D", tz="UTC"
        )):
            raise RuntimeError(f"V45 {label} tensor store calendar compressed")
        for date, symbols in availability.items():
            if not set(symbols).issubset(role_symbols):
                raise RuntimeError(f"V45 {label} availability symbol escaped fold")
            end = store.date_to_index[pd.Timestamp(date)]
            start = end - 255
            indexes = [store.symbol_to_index[symbol] for symbol in symbols]
            if start < 0 or not np.isfinite(
                store.values[indexes, start : end + 1, :]
            ).all():
                raise RuntimeError(f"V45 {label} readiness contains invalid lookback")

    audit = {
        "fold": fold,
        **observed,
        "train_symbols_materialized_for_ridge": train_symbols,
        "heldout_symbols_materialized_for_screen": sorted(
            screen_features["symbol"].unique()
        ),
        "heldout_label_columns_materialized": [],
        "validation_2024_label_rows_materialized": 0,
        "post_2025_rows_materialized": 0,
        "target_symbols_materialized": [],
        "physical_row_group_isolation_claimed": False,
    }
    return FoldPrepareData(
        ridge_feature_panel=ridge_features,
        ridge_labels=ridge_labels,
        ridge_availability=ridge_availability,
        screen_feature_panel=screen_features,
        screen_availability=screen_availability,
        audit=audit,
        receipts=receipts,
    )


def _sample_sequence_sha256(samples: list[dict[str, object]]) -> str:
    serializable = [
        {
            "date": pd.Timestamp(sample["date"]).date().isoformat(),
            "triplet": list(sample["triplet"]),
            "pair_index": int(sample["pair_index"]),
        }
        for sample in samples
    ]
    return _canonical_sha256(serializable)


def fit_frozen_fold_ridge(
    fold: int,
    data: FoldPrepareData,
    train_symbols: list[str],
    feature_names: list[str],
    scaler: FeatureScaler,
    target_scale: float,
    target_scale_state_sha256: str,
    ridge_contract: dict,
) -> tuple[SharedAssetRidgeModel, dict[str, object]]:
    sampler = DeterministicEligibleTripletSampler(
        data.ridge_availability,
        train_symbols,
        int(ridge_contract["sampling_seed"]),
        fold,
    )
    if sampler.total_pairs != 914_280:
        raise RuntimeError("V45 Ridge sampler population drift")
    samples = sampler.sample_epoch(
        int(ridge_contract["sampling_epoch"]),
        int(ridge_contract["train_samples_per_fold"]),
    )
    store = SupervisedFeatureLabelStore(
        data.ridge_feature_panel,
        [data.ridge_labels],
        feature_names,
        ["target_next_open_to_next_open_log_return"],
        256,
        "log_close_to_close_return",
    )
    tensor, labels = store.materialize_batch(samples, scaler)
    returns = torch.from_numpy(labels[..., 0])
    target_z = normalized_triplet_excess(returns, target_scale).numpy()
    model = fit_shared_asset_ridge(
        tensor,
        target_z,
        alpha=float(ridge_contract["alpha"]),
    )
    if (
        model.solution_form != "primal"
        or model.coefficient.shape != (256 * 9,)
        or not np.isfinite(model.coefficient).all()
        or not math.isfinite(model.intercept)
    ):
        raise RuntimeError("V45 frozen Ridge state drift")
    receipt = {
        "fold": fold,
        "implementation": ridge_contract["implementation"],
        "alpha": float(ridge_contract["alpha"]),
        "solution_form": model.solution_form,
        "population_pairs": sampler.total_pairs,
        "sample_count": len(samples),
        "sample_sequence_sha256": _sample_sequence_sha256(samples),
        "sample_unique_pair_indexes": len({
            int(sample["pair_index"]) for sample in samples
        }),
        "sampling_seed": int(ridge_contract["sampling_seed"]),
        "sampling_epoch": int(ridge_contract["sampling_epoch"]),
        "coefficient_count": len(model.coefficient),
        "intercept": float(model.intercept),
        "scaler_state_sha256": scaler.state_sha256(),
        "target_scale": float(target_scale),
        "target_scale_state_sha256": target_scale_state_sha256,
    }
    receipt["ridge_state_semantic_sha256"] = _canonical_sha256(receipt)
    del tensor, labels, returns, target_z, store
    gc.collect()
    return model, receipt


def _configure_prepare_device(screen: dict) -> torch.device:
    torch.set_num_threads(int(screen["torch_threads"]))
    torch.use_deterministic_algorithms(bool(screen["deterministic_algorithms"]))
    fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0").strip().lower()
    if (
        screen["device"] != "mps"
        or screen["dtype"] != "float32"
        or bool(screen["amp"])
        or bool(screen["cpu_fallback_allowed"])
    ):
        raise RuntimeError("V45 prepare runtime contract drift")
    if fallback not in {"", "0", "false", "no"}:
        raise RuntimeError("V45 prepare forbids PYTORCH_ENABLE_MPS_FALLBACK")
    if not torch.backends.mps.is_available():
        raise RuntimeError("V45 prepare requires the registered MPS device")
    return torch.device("mps")


def _load_fold_models(
    context: dict[str, object],
    fold: int,
    device: torch.device,
) -> list[torch.nn.Module]:
    models = []
    fold_entry = context["folds"][fold]
    v44_spec_hash = context["values"]["v44_supervised_spec"][
        "supervised_spec_sha256"
    ]
    for seed in (42, 7, 123):
        row = context["checkpoint_by_key"][(fold, seed)]
        path = Path(row["resolved_path"])
        if _sha256_file(path) != row["checkpoint_sha256"]:
            raise RuntimeError("V45 prepare checkpoint changed after preflight")
        model, payload = load_ranking_excess_supervised_checkpoint(
            path,
            expected_architecture=context["blueprint"]["architecture"],
        )
        metadata = payload["metadata"]
        if (
            payload["model_state_sha256"] != row["model_state_sha256"]
            or int(metadata["fold"]) != fold
            or int(metadata["initialization_seed"]) != seed
            or metadata["supervised_spec_sha256"] != v44_spec_hash
            or metadata["train_symbols"] != sorted(fold_entry["train_symbols"])
            or metadata["test_symbols"] != sorted(fold_entry["test_symbols"])
        ):
            raise RuntimeError("V45 prepare checkpoint association drift")
        model.eval()
        model.requires_grad_(False)
        model.to(device)
        models.append(model)
    return models


def _momentum_30_by_symbol(
    feature_panel: pd.DataFrame,
) -> dict[tuple[pd.Timestamp, str], float]:
    result: dict[tuple[pd.Timestamp, str], float] = {}
    for symbol, frame in feature_panel.groupby("symbol", sort=True):
        current = frame.sort_values("date").copy()
        current["momentum_30"] = current[
            "log_close_to_close_return"
        ].rolling(30, min_periods=30).sum()
        for row in current[["date", "momentum_30"]].itertuples(index=False):
            result[(pd.Timestamp(row.date), str(symbol))] = float(row.momentum_30)
    return result


def infer_frozen_fold_predictions(
    context: dict[str, object],
    fold: int,
    data: FoldPrepareData,
    ridge_model: SharedAssetRidgeModel,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    screen = context["config"]["ranking_excess_screen"]
    fold_entry = context["folds"][fold]
    catalog_entry = context["catalog_folds"][fold]
    symbols = sorted(fold_entry["test_symbols"])
    symbol_to_index = {symbol: index for index, symbol in enumerate(symbols)}
    dates = pd.date_range(
        screen["data_access"]["screen_signal_start"],
        screen["data_access"]["screen_signal_end"],
        freq="D",
        tz="UTC",
    )
    date_to_index = {date: index for index, date in enumerate(dates)}
    frozen_triplets = {tuple(row) for row in catalog_entry["test_triplets"]}
    samples: list[dict[str, object]] = []
    eligibility = np.zeros((len(dates), len(symbols)), dtype=bool)
    for date in dates:
        current = sorted(data.screen_availability[date])
        eligibility[
            date_to_index[date],
            [symbol_to_index[symbol] for symbol in current],
        ] = True
        for triplet in sorted(
            row for row in frozen_triplets if set(row).issubset(current)
        ):
            samples.append({"date": date, "triplet": triplet})
    expected = screen["data_access"]["expected_by_fold"][str(fold)]
    if len(samples) != int(expected["screen_triplet_contexts"]):
        raise RuntimeError("V45 full triplet enumeration drift")

    feature_names = context["feature_names"]
    store = TripletTensorStore(
        data.screen_feature_panel[["date", "symbol", *feature_names]],
        feature_names,
        256,
        "log_close_to_close_return",
    )
    scaler = context["scalers"][fold]
    target_scale = float(
        context["target_scales"][fold]["excess_rms_scale"]
    )
    models = _load_fold_models(context, fold, device)
    context_rows: list[dict[str, object]] = []
    transformer_sums = np.zeros_like(eligibility, dtype=np.float64)
    ridge_sums = np.zeros_like(eligibility, dtype=np.float64)
    context_counts = np.zeros_like(eligibility, dtype=np.int64)
    batch_size = int(screen["inference_batch_size"])
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        x_np = store.materialize_batch(batch, scaler)
        x = torch.from_numpy(x_np).to(device)
        seed_outputs = []
        with torch.inference_mode():
            for model in models:
                output = model(x)
                if set(output) != set(RANKING_EXCESS_HEADS) or any(
                    value.shape != (len(batch), 3)
                    or not bool(torch.isfinite(value).all())
                    for value in output.values()
                ):
                    raise RuntimeError("V45 Transformer output contract drift")
                seed_outputs.append(
                    output["excess_return_z"].detach().cpu().numpy()
                )
        transformer_raw = aggregate_raw_excess_predictions(
            np.stack(seed_outputs, axis=0),
            np.repeat(target_scale, 3),
        )
        ridge_z = predict_shared_asset_ridge(ridge_model, x_np)
        ridge_raw = aggregate_raw_excess_predictions(
            ridge_z[None, ...], np.asarray([target_scale])
        )
        if (
            not np.isfinite(transformer_raw).all()
            or not np.isfinite(ridge_raw).all()
            or not np.allclose(transformer_raw.sum(axis=1), 0.0, atol=1e-9)
            or not np.allclose(ridge_raw.sum(axis=1), 0.0, atol=1e-9)
        ):
            raise RuntimeError("V45 context raw-excess aggregation drift")
        for offset, sample in enumerate(batch):
            date = pd.Timestamp(sample["date"])
            triplet = tuple(str(symbol) for symbol in sample["triplet"])
            date_index = date_to_index[date]
            for slot, symbol in enumerate(triplet):
                asset_index = symbol_to_index[symbol]
                transformer_sums[date_index, asset_index] += float(
                    transformer_raw[offset, slot]
                )
                ridge_sums[date_index, asset_index] += float(
                    ridge_raw[offset, slot]
                )
                context_counts[date_index, asset_index] += 1
            context_rows.append({
                "date": date,
                "fold": fold,
                "triplet_key": "|".join(triplet),
                "symbol_0": triplet[0],
                "symbol_1": triplet[1],
                "symbol_2": triplet[2],
                "transformer_raw_excess_0": float(transformer_raw[offset, 0]),
                "transformer_raw_excess_1": float(transformer_raw[offset, 1]),
                "transformer_raw_excess_2": float(transformer_raw[offset, 2]),
                "ridge_raw_excess_0": float(ridge_raw[offset, 0]),
                "ridge_raw_excess_1": float(ridge_raw[offset, 1]),
                "ridge_raw_excess_2": float(ridge_raw[offset, 2]),
            })
        del x, x_np, seed_outputs, transformer_raw, ridge_z, ridge_raw

    expected_counts = np.zeros_like(context_counts)
    for day, ready in enumerate(eligibility.sum(axis=1)):
        expected_counts[day, eligibility[day]] = math.comb(int(ready) - 1, 2)
    if not np.array_equal(context_counts, expected_counts):
        raise RuntimeError("V45 per-asset context count drift")
    transformer_asset = np.divide(
        transformer_sums,
        context_counts,
        out=np.full_like(transformer_sums, np.nan),
        where=context_counts > 0,
    )
    ridge_asset = np.divide(
        ridge_sums,
        context_counts,
        out=np.full_like(ridge_sums, np.nan),
        where=context_counts > 0,
    )
    momentum_lookup = _momentum_30_by_symbol(data.screen_feature_panel)
    momentum = np.full_like(transformer_asset, np.nan)
    for day, date in enumerate(dates):
        for asset, symbol in enumerate(symbols):
            if eligibility[day, asset]:
                momentum[day, asset] = momentum_lookup[(date, symbol)]
    if (
        not np.isfinite(transformer_asset[eligibility]).all()
        or not np.isfinite(ridge_asset[eligibility]).all()
        or not np.isfinite(momentum[eligibility]).all()
    ):
        raise RuntimeError("V45 eligible prediction/momentum value is non-finite")

    positions = {
        "candidate": ranking_excess_positions(
            transformer_asset,
            momentum,
            eligibility,
            float(screen["policy"]["switch_hurdle"]),
        ),
        "dual_momentum_30": eligible_dual_momentum_positions(
            momentum, eligibility
        ),
        "momentum_gated_equal_weight": momentum_gated_equal_weight_positions(
            momentum, eligibility
        ),
    }
    if any(
        position.shape != eligibility.shape
        or bool((position < 0).any())
        or bool((position.sum(axis=1) > 1.0 + 1e-12).any())
        or bool(position[~eligibility].any())
        for position in positions.values()
    ):
        raise RuntimeError("V45 frozen position contract drift")

    asset_rows = []
    position_rows = []
    for day, date in enumerate(dates):
        for asset, symbol in enumerate(symbols):
            position_rows.append({
                "date": date,
                "fold": fold,
                "symbol": symbol,
                "eligible": bool(eligibility[day, asset]),
                "momentum_30": float(momentum[day, asset])
                if eligibility[day, asset]
                else math.nan,
                "candidate_weight": float(positions["candidate"][day, asset]),
                "dual_momentum_30_weight": float(
                    positions["dual_momentum_30"][day, asset]
                ),
                "momentum_gated_equal_weight_weight": float(
                    positions["momentum_gated_equal_weight"][day, asset]
                ),
            })
            if eligibility[day, asset]:
                asset_rows.append({
                    "date": date,
                    "fold": fold,
                    "symbol": symbol,
                    "context_count": int(context_counts[day, asset]),
                    "transformer_raw_excess": float(
                        transformer_asset[day, asset]
                    ),
                    "ridge_raw_excess": float(ridge_asset[day, asset]),
                    "momentum_30": float(momentum[day, asset]),
                })
    for model in models:
        model.to("cpu")
    del models, store
    if device.type == "mps":
        torch.mps.empty_cache()
    gc.collect()
    context_frame = pd.DataFrame(context_rows)
    asset_frame = pd.DataFrame(asset_rows)
    position_frame = pd.DataFrame(position_rows)
    diagnostics = {
        "fold": fold,
        "signal_dates": len(dates),
        "eligible_asset_dates": int(eligibility.sum()),
        "triplet_contexts": len(samples),
        "seed_context_forwards": len(samples) * 3,
        "minimum_ready_assets": int(eligibility.sum(axis=1).min()),
        "maximum_ready_assets": int(eligibility.sum(axis=1).max()),
        "minimum_context_count": int(context_counts[eligibility].min()),
        "maximum_context_count": int(context_counts[eligibility].max()),
        "position_rows": len(position_frame),
        "heldout_outcome_rows_read": 0,
    }
    return context_frame, asset_frame, position_frame, diagnostics


def _prepare_report(result: dict[str, object]) -> str:
    summary = result["summary"]
    return "\n".join([
        "# TLM v45 Ranking/Excess Development Screen Prepare",
        "",
        "## Decision",
        "",
        "**PREDICTIONS AND POLICIES FROZEN; ONE-SHOT OUTCOME UNSEAL IS AUTHORIZED.**",
        "",
        f"Evaluation-spec SHA-256: `{result['evaluation_spec']['evaluation_spec_sha256']}`",
        f"Triplet contexts: **{summary['triplet_contexts']:,}**",
        f"Eligible asset-dates: **{summary['eligible_asset_dates']:,}**",
        f"Transformer seed-context forwards: **{summary['seed_context_forwards']:,}**",
        "Ridge models fit train-only: **3**",
        "Held-out outcome rows read: **0**",
        "",
        "All context predictions, asset scores, momentum values, and candidate/control positions were hash-sealed before any held-out 2025 return was opened.",
        "",
    ])


def prepare_ranking_excess_development_screen(
    config: dict,
    *,
    reader: Callable[..., pd.DataFrame] = pd.read_parquet,
) -> dict[str, object]:
    context = _metadata_context(config, reopen_checkpoints=False)
    context["config"] = config
    screen = config["ranking_excess_screen"]
    spec = context["evaluation_spec"]
    preflight_output = context["root"] / screen["preflight_output_dir"]
    preflight_required = screen["artifact_contract"]["preflight"]["required_files"]
    preflight = _validate_result_packet(
        preflight_output,
        spec["evaluation_spec_sha256"],
        "preflight",
        preflight_required,
    )
    if preflight["decision"] != "authorize_v45_prepare_without_heldout_outcomes_only":
        raise RuntimeError("V45 preflight does not authorize prepare")
    output = context["root"] / screen["prepare_output_dir"]
    required = screen["artifact_contract"]["prepare"]["required_files"]
    if (output / "completion_receipt.json").is_file():
        return _validate_result_packet(
            output,
            spec["evaluation_spec_sha256"],
            "prepare",
            required,
        )

    device = _configure_prepare_device(screen)
    output.mkdir(parents=True, exist_ok=True)
    data_audits = []
    data_receipts = []
    ridge_manifest = []
    context_frames = []
    asset_frames = []
    position_frames = []
    inference_audits = []
    coefficient_files = []
    for fold in (1, 2, 3):
        fold_entry = context["folds"][fold]
        data = read_fold_prepare_data(
            context["paths"]["panel"],
            context["paths"]["sequence_index"],
            fold_entry,
            screen["data_access"],
            reader=reader,
        )
        data_audits.append(data.audit)
        data_receipts.extend(data.receipts)
        target_scale = float(
            context["target_scales"][fold]["excess_rms_scale"]
        )
        ridge_model, ridge_receipt = fit_frozen_fold_ridge(
            fold,
            data,
            sorted(fold_entry["train_symbols"]),
            context["feature_names"],
            context["scalers"][fold],
            target_scale,
            str(
                context["target_scales"][fold][
                    "target_scale_state_sha256"
                ]
            ),
            screen["ridge"],
        )
        coefficient_name = f"ridge_fold_{fold}_coefficient.npy"
        coefficient_path = output / coefficient_name
        _write_npy_atomic(ridge_model.coefficient, coefficient_path)
        ridge_receipt["coefficient_file"] = coefficient_name
        ridge_receipt["coefficient_file_sha256"] = _sha256_file(
            coefficient_path
        )
        semantic = dict(ridge_receipt)
        semantic.pop("ridge_state_semantic_sha256", None)
        ridge_receipt["ridge_state_semantic_sha256"] = _canonical_sha256(
            semantic
        )
        coefficient_files.append(coefficient_name)
        ridge_manifest.append(ridge_receipt)
        context_frame, asset_frame, position_frame, inference_audit = (
            infer_frozen_fold_predictions(
                context,
                fold,
                data,
                ridge_model,
                device,
            )
        )
        context_frames.append(context_frame)
        asset_frames.append(asset_frame)
        position_frames.append(position_frame)
        inference_audits.append(inference_audit)
        del data, ridge_model, context_frame, asset_frame, position_frame
        gc.collect()

    context_frame = pd.concat(context_frames, ignore_index=True).sort_values(
        ["date", "fold", "triplet_key"]
    ).reset_index(drop=True)
    asset_frame = pd.concat(asset_frames, ignore_index=True).sort_values(
        ["date", "fold", "symbol"]
    ).reset_index(drop=True)
    position_frame = pd.concat(position_frames, ignore_index=True).sort_values(
        ["date", "fold", "symbol"]
    ).reset_index(drop=True)
    context_schema = screen["artifact_contract"]["prepare"][
        "context_prediction_schema"
    ]
    asset_schema = screen["artifact_contract"]["prepare"][
        "asset_prediction_schema"
    ]
    position_schema = screen["artifact_contract"]["prepare"]["position_schema"]
    if (
        list(context_frame.columns) != context_schema
        or list(asset_frame.columns) != asset_schema
        or list(position_frame.columns) != position_schema
        or len(context_frame)
        != int(screen["data_access"]["expected_total_triplet_contexts"])
        or len(asset_frame)
        != int(screen["data_access"]["expected_total_asset_dates"])
        or len(position_frame)
        != int(screen["data_access"]["expected_total_fold_signal_dates"]) * 10
    ):
        raise RuntimeError("V45 prepared artifact schema/count drift")

    data_access_receipt = {
        "version": "v45_prepare_data_access_receipt_v1",
        "reads": data_receipts,
        "read_count": len(data_receipts),
        "heldout_label_reads": 0,
        "requested_heldout_label_columns": [],
    }
    data_audit = {
        "folds": data_audits,
        "inference": inference_audits,
        "totals": {
            "triplet_contexts": len(context_frame),
            "eligible_asset_dates": len(asset_frame),
            "position_rows": len(position_frame),
            "seed_context_forwards": sum(
                int(row["seed_context_forwards"]) for row in inference_audits
            ),
            "heldout_outcome_rows": 0,
            "target_asset_rows": 0,
            "post_2025_rows": 0,
        },
    }
    data_audit["data_access_and_eligibility_schedule_sha256"] = (
        _canonical_sha256(data_audit)
    )
    _write_json_atomic(output / "evaluation_spec.json", spec)
    _write_json_atomic(output / "data_access_receipt.json", data_access_receipt)
    _write_json_atomic(output / "data_audit.json", data_audit)
    _write_json_atomic(output / "ridge_manifest.json", ridge_manifest)
    _write_parquet_atomic(context_frame, output / "context_predictions.parquet")
    _write_parquet_atomic(asset_frame, output / "asset_predictions.parquet")
    _write_parquet_atomic(position_frame, output / "positions.parquet")
    checks = {
        "preflight_packet_is_hash_valid": True,
        "all_prepare_reads_are_separate_and_filtered": True,
        "ridge_uses_only_train_assets_and_action_returns": True,
        "ridge_sample_population_and_sequence_are_frozen": True,
        "all_nine_registered_checkpoints_are_used_without_selection": True,
        "all_eligible_lexical_2025_triplets_are_scored": True,
        "context_counts_match_choose_n_minus_1_2": True,
        "predictions_and_positions_are_frozen_before_outcomes": True,
        "zero_heldout_label_columns_or_values_read": True,
        "zero_validation_labels_or_2026_rows": True,
        "zero_target_assets": True,
        "zero_performance_metrics_or_pnl": True,
    }
    audit = {"checks": checks, "passed": bool(all(checks.values()))}
    result: dict[str, object] = {
        "version": VERSION,
        "mode": "prepare",
        "decision": "authorize_v45_one_shot_outcome_unseal_only",
        "evaluation_spec": spec,
        "preflight_completion_receipt_sha256": _sha256_file(
            preflight_output / "completion_receipt.json"
        ),
        "checkpoint_receipt": context["checkpoint_receipt"],
        "scaler_receipt": context["scaler_receipt"],
        "target_scale_receipt": [
            context["target_scales"][fold] for fold in (1, 2, 3)
        ],
        "ridge_manifest": ridge_manifest,
        "data_audit": data_audit,
        "summary": {
            "ridge_models_fit": 3,
            "ridge_train_samples": 3 * int(
                screen["ridge"]["train_samples_per_fold"]
            ),
            "triplet_contexts": len(context_frame),
            "eligible_asset_dates": len(asset_frame),
            "position_rows": len(position_frame),
            "seed_context_forwards": int(
                screen["data_access"]["expected_total_seed_context_forwards"]
            ),
            "heldout_outcome_rows": 0,
            "performance_metrics": 0,
            "pnl_evaluations": 0,
            "target_asset_rows": 0,
        },
        "audit": audit,
    }
    _write_json_atomic(output / "audit.json", audit)
    _write_yaml_atomic(output / "resolved_config.yaml", config)
    _atomic_write_text(output / "report.md", _prepare_report(result))
    core_files = [
        "evaluation_spec.json",
        "data_access_receipt.json",
        "data_audit.json",
        "ridge_manifest.json",
        "context_predictions.parquet",
        "asset_predictions.parquet",
        "positions.parquet",
        "audit.json",
        "resolved_config.yaml",
        "report.md",
        *coefficient_files,
    ]
    _seal_result_packet(output, result, core_files)
    return _validate_result_packet(
        output,
        spec["evaluation_spec_sha256"],
        "prepare",
        required,
    )


def _load_and_validate_prepare_artifacts(
    context: dict[str, object],
    config: dict,
) -> dict[str, object]:
    screen = config["ranking_excess_screen"]
    spec = context["evaluation_spec"]
    output = context["root"] / screen["prepare_output_dir"]
    required = screen["artifact_contract"]["prepare"]["required_files"]
    prepare_result = _validate_result_packet(
        output,
        spec["evaluation_spec_sha256"],
        "prepare",
        required,
    )
    if prepare_result["decision"] != "authorize_v45_one_shot_outcome_unseal_only":
        raise RuntimeError("V45 prepare packet does not authorize unseal")
    context_frame = pd.read_parquet(
        output / "context_predictions.parquet", engine="pyarrow"
    )
    asset_frame = pd.read_parquet(
        output / "asset_predictions.parquet", engine="pyarrow"
    )
    position_frame = pd.read_parquet(output / "positions.parquet", engine="pyarrow")
    context_schema = screen["artifact_contract"]["prepare"][
        "context_prediction_schema"
    ]
    asset_schema = screen["artifact_contract"]["prepare"][
        "asset_prediction_schema"
    ]
    position_schema = screen["artifact_contract"]["prepare"]["position_schema"]
    if (
        list(context_frame.columns) != context_schema
        or list(asset_frame.columns) != asset_schema
        or list(position_frame.columns) != position_schema
    ):
        raise RuntimeError("V45 hash-valid prepare artifact schema drift")
    for frame in (context_frame, asset_frame, position_frame):
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        frame["fold"] = frame["fold"].astype(int)
    for column in ("symbol_0", "symbol_1", "symbol_2"):
        context_frame[column] = context_frame[column].astype(str)
    asset_frame["symbol"] = asset_frame["symbol"].astype(str)
    position_frame["symbol"] = position_frame["symbol"].astype(str)
    if not pd.api.types.is_bool_dtype(position_frame["eligible"]):
        raise RuntimeError("V45 prepared eligibility type drift")
    if (
        context_frame.duplicated(["date", "fold", "triplet_key"]).any()
        or asset_frame.duplicated(["date", "fold", "symbol"]).any()
        or position_frame.duplicated(["date", "fold", "symbol"]).any()
    ):
        raise RuntimeError("V45 prepare artifact contains duplicate keys")
    data_access = screen["data_access"]
    signal_start = pd.Timestamp(data_access["screen_signal_start"], tz="UTC")
    signal_end = pd.Timestamp(data_access["screen_signal_end"], tz="UTC")
    if (
        len(context_frame) != int(data_access["expected_total_triplet_contexts"])
        or len(asset_frame) != int(data_access["expected_total_asset_dates"])
        or len(position_frame)
        != int(data_access["expected_total_fold_signal_dates"]) * 10
        or context_frame["date"].min() != signal_start
        or context_frame["date"].max() != signal_end
        or asset_frame["date"].min() != signal_start
        or asset_frame["date"].max() != signal_end
        or position_frame["date"].min() != signal_start
        or position_frame["date"].max() != signal_end
    ):
        raise RuntimeError("V45 prepare artifact count/date drift")
    if any(
        TARGET_SYMBOLS.intersection(frame[column].unique())
        for frame, column in (
            (context_frame, "symbol_0"),
            (context_frame, "symbol_1"),
            (context_frame, "symbol_2"),
            (asset_frame, "symbol"),
            (position_frame, "symbol"),
        )
    ):
        raise RuntimeError("V45 target asset entered prepared artifacts")

    aggregates: dict[tuple[pd.Timestamp, int, str], list[float]] = {}
    prediction_columns = [
        *(f"transformer_raw_excess_{slot}" for slot in range(3)),
        *(f"ridge_raw_excess_{slot}" for slot in range(3)),
    ]
    if not np.isfinite(
        context_frame[prediction_columns].to_numpy(dtype=np.float64)
    ).all():
        raise RuntimeError("V45 context prediction artifact is non-finite")
    for row in context_frame.itertuples(index=False):
        symbols = tuple(str(getattr(row, f"symbol_{slot}")) for slot in range(3))
        if (
            tuple(sorted(symbols)) != symbols
            or str(row.triplet_key) != "|".join(symbols)
            or len(set(symbols)) != 3
        ):
            raise RuntimeError("V45 prepared triplet identity drift")
        if any(symbol not in context["folds"][int(row.fold)]["test_symbols"] for symbol in symbols):
            raise RuntimeError("V45 prepared triplet escaped its held-out fold")
        for slot, symbol in enumerate(symbols):
            key = (pd.Timestamp(row.date), int(row.fold), symbol)
            state = aggregates.setdefault(key, [0.0, 0.0, 0.0])
            state[0] += 1.0
            state[1] += float(
                getattr(row, f"transformer_raw_excess_{slot}")
            )
            state[2] += float(getattr(row, f"ridge_raw_excess_{slot}"))
    asset_lookup = {
        (pd.Timestamp(row.date), int(row.fold), str(row.symbol)): row
        for row in asset_frame.itertuples(index=False)
    }
    ready_counts: dict[tuple[pd.Timestamp, int], int] = {}
    for date, fold, _ in asset_lookup:
        ready_counts[(date, fold)] = ready_counts.get((date, fold), 0) + 1
    if set(aggregates) != set(asset_lookup):
        raise RuntimeError("V45 context and asset prediction keys drift")
    for key, (count, transformer_sum, ridge_sum) in aggregates.items():
        row = asset_lookup[key]
        fold = key[1]
        date = key[0]
        ready = ready_counts[(date, fold)]
        expected_count = math.comb(ready - 1, 2)
        if (
            int(count) != int(row.context_count)
            or int(row.context_count) != expected_count
            or not math.isclose(
                transformer_sum / count,
                float(row.transformer_raw_excess),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or not math.isclose(
                ridge_sum / count,
                float(row.ridge_raw_excess),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            or not math.isfinite(float(row.momentum_30))
        ):
            raise RuntimeError("V45 prepared context-to-asset aggregation drift")

    expected_position_keys = {
        (date, fold, symbol)
        for fold in (1, 2, 3)
        for date in pd.date_range(signal_start, signal_end, freq="D", tz="UTC")
        for symbol in sorted(context["folds"][fold]["test_symbols"])
    }
    position_lookup = {
        (pd.Timestamp(row.date), int(row.fold), str(row.symbol)): row
        for row in position_frame.itertuples(index=False)
    }
    if set(position_lookup) != expected_position_keys:
        raise RuntimeError("V45 prepared position grid drift")
    for key, row in position_lookup.items():
        eligible = key in asset_lookup
        if bool(row.eligible) != eligible:
            raise RuntimeError("V45 prepared eligibility/asset prediction drift")
        if eligible:
            asset = asset_lookup[key]
            if not math.isclose(
                float(row.momentum_30),
                float(asset.momentum_30),
                rel_tol=0.0,
                abs_tol=0.0,
            ):
                raise RuntimeError("V45 prepared position momentum drift")
        elif not math.isnan(float(row.momentum_30)):
            raise RuntimeError("V45 ineligible prepared momentum must be NaN")

    dates = pd.date_range(signal_start, signal_end, freq="D", tz="UTC")
    weight_columns = {
        "candidate": "candidate_weight",
        "dual_momentum_30": "dual_momentum_30_weight",
        "momentum_gated_equal_weight": "momentum_gated_equal_weight_weight",
    }
    for fold in (1, 2, 3):
        symbols = sorted(context["folds"][fold]["test_symbols"])
        scores = np.full((len(dates), len(symbols)), np.nan, dtype=np.float64)
        momentum = np.full_like(scores, np.nan)
        eligibility = np.zeros_like(scores, dtype=bool)
        persisted = {
            name: np.zeros_like(scores) for name in weight_columns
        }
        for day, date in enumerate(dates):
            for asset, symbol in enumerate(symbols):
                key = (date, fold, symbol)
                row = position_lookup[key]
                eligibility[day, asset] = bool(row.eligible)
                if row.eligible:
                    asset_row = asset_lookup[key]
                    scores[day, asset] = float(
                        asset_row.transformer_raw_excess
                    )
                    momentum[day, asset] = float(asset_row.momentum_30)
                for name, column in weight_columns.items():
                    persisted[name][day, asset] = float(getattr(row, column))
        recomputed = {
            "candidate": ranking_excess_positions(
                scores,
                momentum,
                eligibility,
                float(screen["policy"]["switch_hurdle"]),
            ),
            "dual_momentum_30": eligible_dual_momentum_positions(
                momentum, eligibility
            ),
            "momentum_gated_equal_weight": momentum_gated_equal_weight_positions(
                momentum, eligibility
            ),
        }
        if any(
            not np.array_equal(recomputed[name], persisted[name])
            for name in weight_columns
        ):
            raise RuntimeError("V45 prepared frozen policy drift")

    ridge_manifest = _load_json(output / "ridge_manifest.json")
    if not isinstance(ridge_manifest, list) or len(ridge_manifest) != 3:
        raise RuntimeError("V45 prepared Ridge manifest drift")
    ridge_folds = set()
    for row in ridge_manifest:
        fold = int(row["fold"])
        coefficient_path = output / row["coefficient_file"]
        semantic = dict(row)
        claimed_semantic = semantic.pop("ridge_state_semantic_sha256", None)
        coefficient = np.load(coefficient_path, allow_pickle=False)
        if (
            fold not in {1, 2, 3}
            or fold in ridge_folds
            or row["coefficient_file"] != f"ridge_fold_{fold}_coefficient.npy"
            or row["sample_count"] != int(screen["ridge"]["train_samples_per_fold"])
            or row["population_pairs"] != 914_280
            or not row.get("sample_sequence_sha256")
            or row["coefficient_file_sha256"] != _sha256_file(coefficient_path)
            or claimed_semantic != _canonical_sha256(semantic)
            or coefficient.shape != (256 * 9,)
            or not np.isfinite(coefficient).all()
        ):
            raise RuntimeError("V45 prepared Ridge state/sample receipt drift")
        ridge_folds.add(fold)
    if ridge_folds != {1, 2, 3}:
        raise RuntimeError("V45 prepared Ridge fold grid drift")
    return {
        "result": prepare_result,
        "output": output,
        "completion_receipt_sha256": _sha256_file(
            output / "completion_receipt.json"
        ),
        "context_predictions": context_frame,
        "asset_predictions": asset_frame,
        "positions": position_frame,
        "ridge_manifest": ridge_manifest,
    }


def _outcome_read_contract(
    fold_entry: dict,
    data_access: dict,
) -> tuple[list[str], list[tuple], dict[str, object]]:
    fold = int(fold_entry["fold"])
    test_symbols = sorted(str(symbol) for symbol in fold_entry["test_symbols"])
    columns = list(data_access["screen_label_columns"])
    signal_start = pd.Timestamp(data_access["screen_signal_start"], tz="UTC")
    signal_end = pd.Timestamp(data_access["screen_signal_end"], tz="UTC")
    maturity_end = pd.Timestamp(data_access["screen_maturity_end"], tz="UTC")
    filters = [
        ("symbol", "in", test_symbols),
        (data_access["screen_split_flag"], "==", True),
        *[(name, "==", True) for name in data_access["readiness_flags"]],
        ("date", ">=", signal_start),
        ("date", "<=", signal_end),
        ("target_window_end_date", "<=", maturity_end),
    ]
    expected = data_access["expected_by_fold"][str(fold)]
    receipt = {
        "fold": fold,
        "dataset": "heldout_2025_action_return",
        "columns": columns,
        "filters": _serialize_filters(filters),
        "rows": int(expected["screen_label_rows"]),
        "first_signal_date": signal_start.date().isoformat(),
        "last_signal_date": signal_end.date().isoformat(),
        "maximum_target_maturity": maturity_end.date().isoformat(),
    }
    return columns, filters, receipt


def _read_fold_heldout_outcomes(
    panel_path: Path,
    fold_entry: dict,
    data_access: dict,
    *,
    reader: Callable[..., pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, object]]:
    fold = int(fold_entry["fold"])
    test_symbols = sorted(str(symbol) for symbol in fold_entry["test_symbols"])
    signal_start = pd.Timestamp(data_access["screen_signal_start"], tz="UTC")
    signal_end = pd.Timestamp(data_access["screen_signal_end"], tz="UTC")
    maturity_end = pd.Timestamp(data_access["screen_maturity_end"], tz="UTC")
    columns, filters, receipt = _outcome_read_contract(
        fold_entry, data_access
    )
    frame = reader(
        panel_path,
        engine="pyarrow",
        columns=columns,
        filters=filters,
    )
    frame = _validate_projected_frame(
        frame, columns, label=f"fold {fold} held-out outcomes"
    )
    frame["target_window_end_date"] = pd.to_datetime(
        frame["target_window_end_date"], utc=True
    )
    if (
        frame["date"].min() < signal_start
        or frame["date"].max() > signal_end
        or frame["target_window_end_date"].max() > maturity_end
        or not bool((
            frame["target_window_end_date"]
            == frame["date"] + pd.Timedelta(days=8)
        ).all())
        or not set(frame["symbol"].unique()).issubset(test_symbols)
        or set(frame["symbol"].unique()).intersection(
            set(fold_entry["train_symbols"]) | TARGET_SYMBOLS
        )
    ):
        raise RuntimeError("V45 held-out outcome reader ignored a frozen filter")
    values = frame[
        "target_next_open_to_next_open_log_return"
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("V45 held-out outcomes contain non-finite returns")
    expected = data_access["expected_by_fold"][str(fold)]
    if len(frame) != int(expected["screen_label_rows"]):
        raise RuntimeError("V45 held-out outcome row count drift")
    result = frame.rename(columns={
        "target_next_open_to_next_open_log_return": "action_log_return"
    })
    result.insert(1, "fold", fold)
    result = result[[
        "date",
        "fold",
        "symbol",
        "target_window_end_date",
        "action_log_return",
    ]]
    return result, receipt


def _validate_outcome_packet(
    output: Path,
    expected_spec_sha256: str,
    prepare_completion_receipt_sha256: str,
    expected_schema: list[str],
    expected_rows: int,
    signal_start: pd.Timestamp,
    signal_end: pd.Timestamp,
    maturity_end: pd.Timestamp,
    expected_source_reads: list[dict[str, object]],
    expected_bindings: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    unseal_path = output / "unseal_receipt.json"
    outcome_path = output / "outcomes.parquet"
    receipt_path = output / "outcome_receipt.json"
    if not (unseal_path.is_file() and outcome_path.is_file() and receipt_path.is_file()):
        raise RuntimeError("V45 unseal exists without a complete atomic outcome packet")
    unseal = _load_json(unseal_path)
    receipt = _load_json(receipt_path)
    expected_binding_names = {
        "prepare_result_sha256",
        "context_predictions_sha256",
        "asset_predictions_sha256",
        "positions_sha256",
        "source_panel_sha256",
        "evaluation_execution_count",
    }
    if set(expected_bindings) != expected_binding_names:
        raise RuntimeError("V45 expected outcome binding contract drift")
    expected_unseal_keys = {
        "version",
        "started_at_utc",
        "evaluation_spec_sha256",
        "prepare_completion_receipt_sha256",
        *expected_binding_names,
    }
    expected_receipt_keys = {
        "version",
        "completed_at_utc",
        "evaluation_spec_sha256",
        "prepare_completion_receipt_sha256",
        "unseal_receipt_sha256",
        "source_panel_sha256",
        "source_reads",
        "outcome_rows",
        "outcome_schema",
        "outcomes_parquet_sha256",
        "evaluation_execution_count",
    }
    if (
        set(unseal) != expected_unseal_keys
        or set(receipt) != expected_receipt_keys
        or unseal.get("version") != "v45_unseal_receipt_v1"
        or unseal.get("evaluation_spec_sha256") != expected_spec_sha256
        or unseal.get("prepare_completion_receipt_sha256")
        != prepare_completion_receipt_sha256
        or any(
            unseal.get(name) != value
            for name, value in expected_bindings.items()
        )
        or receipt.get("version") != "v45_outcome_packet_receipt_v1"
        or receipt.get("evaluation_spec_sha256") != expected_spec_sha256
        or receipt.get("prepare_completion_receipt_sha256")
        != prepare_completion_receipt_sha256
        or receipt.get("unseal_receipt_sha256") != _sha256_file(unseal_path)
        or receipt.get("outcomes_parquet_sha256") != _sha256_file(outcome_path)
        or int(receipt.get("outcome_rows", -1)) != expected_rows
        or receipt.get("outcome_schema") != expected_schema
        or receipt.get("source_reads") != expected_source_reads
        or receipt.get("source_panel_sha256")
        != expected_bindings["source_panel_sha256"]
        or receipt.get("evaluation_execution_count")
        != expected_bindings["evaluation_execution_count"]
    ):
        raise RuntimeError("V45 outcome packet cryptographic binding drift")
    frame = pd.read_parquet(outcome_path, engine="pyarrow")
    if list(frame.columns) != expected_schema or len(frame) != expected_rows:
        raise RuntimeError("V45 outcome packet schema/count drift")
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame["target_window_end_date"] = pd.to_datetime(
        frame["target_window_end_date"], utc=True
    )
    frame["fold"] = frame["fold"].astype(int)
    frame["symbol"] = frame["symbol"].astype(str)
    if (
        frame.duplicated(["date", "fold", "symbol"]).any()
        or not np.isfinite(frame["action_log_return"].to_numpy(dtype=np.float64)).all()
        or TARGET_SYMBOLS.intersection(frame["symbol"].unique())
        or frame["date"].min() != signal_start
        or frame["date"].max() != signal_end
        or frame["target_window_end_date"].max() != maturity_end
        or not bool((
            frame["target_window_end_date"]
            == frame["date"] + pd.Timedelta(days=8)
        ).all())
    ):
        raise RuntimeError("V45 outcome packet semantic drift")
    return frame, receipt


def _unseal_or_load_outcomes(
    context: dict[str, object],
    config: dict,
    prepared: dict[str, object],
    *,
    reader: Callable[..., pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, object]]:
    screen = config["ranking_excess_screen"]
    spec_sha = context["evaluation_spec"]["evaluation_spec_sha256"]
    output = context["root"] / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    expected_schema = screen["artifact_contract"]["evaluate"]["outcome_schema"]
    expected_rows = int(screen["data_access"]["expected_total_asset_dates"])
    signal_start = pd.Timestamp(
        screen["data_access"]["screen_signal_start"], tz="UTC"
    )
    signal_end = pd.Timestamp(
        screen["data_access"]["screen_signal_end"], tz="UTC"
    )
    maturity_end = pd.Timestamp(
        screen["data_access"]["screen_maturity_end"], tz="UTC"
    )
    expected_source_reads = [
        _outcome_read_contract(
            context["folds"][fold], screen["data_access"]
        )[2]
        for fold in (1, 2, 3)
    ]
    unseal_path = output / "unseal_receipt.json"
    outcome_path = output / "outcomes.parquet"
    receipt_path = output / "outcome_receipt.json"
    packet_exists = (
        unseal_path.exists(),
        outcome_path.exists(),
        receipt_path.exists(),
    )
    if any(packet_exists) and not all(packet_exists):
        raise RuntimeError("V45 unseal exists without a complete atomic outcome packet")
    expected_bindings = {
        "prepare_result_sha256": _sha256_file(
            prepared["output"] / "result.json"
        ),
        "context_predictions_sha256": _sha256_file(
            prepared["output"] / "context_predictions.parquet"
        ),
        "asset_predictions_sha256": _sha256_file(
            prepared["output"] / "asset_predictions.parquet"
        ),
        "positions_sha256": _sha256_file(
            prepared["output"] / "positions.parquet"
        ),
        "source_panel_sha256": context["input_hashes"]["panel"],
        "evaluation_execution_count": 1,
    }
    if all(packet_exists):
        outcomes, receipt = _validate_outcome_packet(
            output,
            spec_sha,
            prepared["completion_receipt_sha256"],
            expected_schema,
            expected_rows,
            signal_start,
            signal_end,
            maturity_end,
            expected_source_reads,
            expected_bindings,
        )
        expected_keys = set(zip(
            prepared["asset_predictions"]["date"],
            prepared["asset_predictions"]["fold"],
            prepared["asset_predictions"]["symbol"],
            strict=True,
        ))
        observed_keys = set(zip(
            outcomes["date"], outcomes["fold"], outcomes["symbol"], strict=True
        ))
        if observed_keys != expected_keys:
            raise RuntimeError("V45 cached outcome keys differ from prepare packet")
        return outcomes, receipt

    unseal = {
        "version": "v45_unseal_receipt_v1",
        "started_at_utc": _utc_now(),
        "evaluation_spec_sha256": spec_sha,
        "prepare_completion_receipt_sha256": prepared[
            "completion_receipt_sha256"
        ],
        **expected_bindings,
    }
    _write_json_atomic(unseal_path, unseal)

    outcome_frames = []
    read_receipts = []
    for fold in (1, 2, 3):
        frame, receipt = _read_fold_heldout_outcomes(
            context["paths"]["panel"],
            context["folds"][fold],
            screen["data_access"],
            reader=reader,
        )
        outcome_frames.append(frame)
        read_receipts.append(receipt)
    outcomes = pd.concat(outcome_frames, ignore_index=True).sort_values(
        ["date", "fold", "symbol"]
    ).reset_index(drop=True)
    if list(outcomes.columns) != expected_schema or len(outcomes) != expected_rows:
        raise RuntimeError("V45 combined held-out outcome packet drift")
    expected_keys = set(zip(
        prepared["asset_predictions"]["date"],
        prepared["asset_predictions"]["fold"],
        prepared["asset_predictions"]["symbol"],
        strict=True,
    ))
    outcome_keys = set(zip(
        outcomes["date"], outcomes["fold"], outcomes["symbol"], strict=True
    ))
    if expected_keys != outcome_keys:
        raise RuntimeError("V45 outcome keys differ from frozen prediction keys")
    _write_parquet_atomic(outcomes, outcome_path)
    receipt = {
        "version": "v45_outcome_packet_receipt_v1",
        "completed_at_utc": _utc_now(),
        "evaluation_spec_sha256": spec_sha,
        "prepare_completion_receipt_sha256": prepared[
            "completion_receipt_sha256"
        ],
        "unseal_receipt_sha256": _sha256_file(unseal_path),
        "source_panel_sha256": context["input_hashes"]["panel"],
        "source_reads": read_receipts,
        "outcome_rows": len(outcomes),
        "outcome_schema": expected_schema,
        "outcomes_parquet_sha256": _sha256_file(outcome_path),
        "evaluation_execution_count": 1,
    }
    _write_json_atomic(receipt_path, receipt)
    return _validate_outcome_packet(
        output,
        spec_sha,
        prepared["completion_receipt_sha256"],
        expected_schema,
        expected_rows,
        signal_start,
        signal_end,
        maturity_end,
        expected_source_reads,
        expected_bindings,
    )


def _evaluation_report(result: dict[str, object]) -> str:
    gate = result["gate_result"]
    aggregate = result["aggregate_metrics"]["10"]
    candidate = aggregate["candidate"]
    dual = aggregate["dual_momentum_30"]
    equal = aggregate["momentum_gated_equal_weight"]
    if gate["passed"]:
        status = "DEVELOPMENT SCREEN PASSED; FAMILY FROZEN FOR LATER PROSPECTIVE NON-TARGET CONFIRMATION."
    else:
        status = "DEVELOPMENT SCREEN FAILED; FAMILY RETIRED WITHOUT TARGET EVALUATION OR TUNING."
    return "\n".join([
        "# TLM v45 Ranking/Excess Asset-Disjoint Development Screen",
        "",
        "## Decision",
        "",
        f"**{status}**",
        "",
        f"Decision: `{result['decision']}`",
        f"Evaluation-spec SHA-256: `{result['evaluation_spec']['evaluation_spec_sha256']}`",
        "Evaluation execution count: **1**",
        "Held-out signal dates: **357**",
        "",
        "## Base-cost aggregate (10 bps)",
        "",
        "| Strategy | Total return | Sharpe | Max drawdown | Turnover |",
        "|---|---:|---:|---:|---:|",
        f"| Candidate | {candidate['total_return']:.2%} | {candidate['sharpe']:.3f} | {candidate['max_drawdown']:.2%} | {candidate['total_turnover']:.1f} |",
        f"| Dual momentum 30 | {dual['total_return']:.2%} | {dual['sharpe']:.3f} | {dual['max_drawdown']:.2%} | {dual['total_turnover']:.1f} |",
        f"| Momentum-gated equal weight | {equal['total_return']:.2%} | {equal['sharpe']:.3f} | {equal['max_drawdown']:.2%} | {equal['total_turnover']:.1f} |",
        "",
        f"Registered gate cells: **{gate['cell_count']}**",
        "",
        "BTC, ETH, SOL and every 2026 outcome remained sealed. No failed cell may be tuned or rerun on this window.",
        "",
    ])


def evaluate_ranking_excess_development_screen(
    config: dict,
    *,
    outcome_reader: Callable[..., pd.DataFrame] = pd.read_parquet,
) -> dict[str, object]:
    context = _metadata_context(config, reopen_checkpoints=False)
    screen = config["ranking_excess_screen"]
    spec = context["evaluation_spec"]
    output = context["root"] / config["output_dir"]
    required = screen["artifact_contract"]["evaluate"]["required_result_files"]
    if (output / "completion_receipt.json").is_file():
        return _validate_result_packet(
            output,
            spec["evaluation_spec_sha256"],
            "evaluate",
            required,
        )

    # Every operation through this point is outcome-free. Validate all frozen
    # predictions and policies before creating the irreversible unseal receipt.
    prepared = _load_and_validate_prepare_artifacts(context, config)
    outcomes, outcome_receipt = _unseal_or_load_outcomes(
        context,
        config,
        prepared,
        reader=outcome_reader,
    )
    context_metrics, predictive_daily, predictive_summary = (
        compute_predictive_metrics(
            prepared["context_predictions"],
            outcomes,
            float(screen["predictive_metrics"]["exact_tie_tolerance"]),
        )
    )
    if (
        len(context_metrics)
        != int(screen["data_access"]["expected_total_triplet_contexts"])
        or len(predictive_daily)
        != int(screen["data_access"]["expected_total_fold_signal_dates"])
        or int(predictive_summary["unique_date_count"])
        != int(screen["data_access"]["expected_unique_signal_dates"])
    ):
        raise RuntimeError("V45 predictive metric observation count drift")

    fold_symbols = {
        fold: sorted(context["folds"][fold]["test_symbols"])
        for fold in (1, 2, 3)
    }
    portfolio = build_portfolio_evaluation(
        prepared["positions"],
        outcomes,
        fold_symbols,
        list(screen["accounting"]["reporting_cost_bps"]),
        int(screen["accounting"]["annualization_days"]),
    )
    daily_returns = portfolio["daily_frame"]
    expected_daily_rows = (
        int(screen["data_access"]["expected_unique_signal_dates"])
        * len(screen["accounting"]["reporting_cost_bps"])
        * 4
        * len(PREPARE_STRATEGIES)
    )
    if len(daily_returns) != expected_daily_rows:
        raise RuntimeError("V45 portfolio daily artifact row count drift")

    top1_series = (
        predictive_daily.groupby("date", sort=True)[
            "transformer_top1_excess"
        ]
        .mean()
        .sort_index()
    )
    bootstrap_contract = screen["bootstrap"]
    top1_bootstrap = top1_excess_block_bootstrap(
        top1_series,
        bootstrap_contract["block_lengths_days"],
        int(bootstrap_contract["paths"]),
        int(bootstrap_contract["base_seed"]),
        int(bootstrap_contract["batch_size"]),
    )
    base_daily = daily_returns.loc[
        (daily_returns["cost_bps"] == int(bootstrap_contract["economic_cost_bps"]))
        & (daily_returns["scope"] == "aggregate_equal_fold_capital")
    ]
    economic_series = {}
    for strategy in PREPARE_STRATEGIES:
        current = base_daily.loc[
            base_daily["strategy"] == strategy
        ].sort_values("date")
        if (
            len(current)
            != int(screen["data_access"]["expected_unique_signal_dates"])
            or current["date"].duplicated().any()
        ):
            raise RuntimeError("V45 economic bootstrap series drift")
        economic_series[strategy] = current["net_return"].to_numpy(
            dtype=np.float64
        )
    economic_bootstrap = {
        str(block): paired_block_bootstrap(
            economic_series,
            "candidate",
            ["dual_momentum_30", "momentum_gated_equal_weight"],
            block_length=int(block),
            n_paths=int(bootstrap_contract["paths"]),
            seed=int(bootstrap_contract["base_seed"]) + int(block),
            batch_size=int(bootstrap_contract["batch_size"]),
        )
        for block in bootstrap_contract["block_lengths_days"]
    }
    bootstrap = {
        "top1_excess": top1_bootstrap,
        "economic": economic_bootstrap,
    }
    gate_result = evaluate_v45_gates(
        predictive_summary,
        portfolio["fold_metrics"],
        portfolio["aggregate_metrics"],
        bootstrap,
        screen["gates"],
    )
    decision = (
        screen["lifecycle"]["pass_action"]
        if gate_result["passed"]
        else screen["lifecycle"]["failure_action"]
    )
    output.mkdir(parents=True, exist_ok=True)
    _write_parquet_atomic(
        context_metrics, output / "predictive_context_metrics.parquet"
    )
    _write_parquet_atomic(
        predictive_daily, output / "predictive_daily_metrics.parquet"
    )
    _write_json_atomic(output / "predictive_metrics.json", predictive_summary)
    _write_parquet_atomic(daily_returns, output / "daily_returns.parquet")
    _write_json_atomic(output / "fold_metrics.json", portfolio["fold_metrics"])
    _write_json_atomic(
        output / "aggregate_metrics.json", portfolio["aggregate_metrics"]
    )
    _write_json_atomic(output / "bootstrap.json", bootstrap)
    _write_json_atomic(output / "gate_result.json", gate_result)
    checks = {
        "prepare_completion_packet_is_hash_valid": True,
        "prepare_predictions_and_positions_were_not_recomputed": True,
        "unseal_receipt_preceded_first_heldout_outcome_read": True,
        "outcome_packet_is_atomic_and_hash_bound": True,
        "outcome_keys_equal_frozen_prediction_keys": True,
        "all_triplet_context_predictive_metrics_are_preserved": True,
        "equal_date_and_equal_fold_aggregation_is_exact": True,
        "all_three_cost_cells_and_fold_metrics_are_preserved": True,
        "all_10000_path_7_21_63_bootstraps_are_preserved": True,
        "every_registered_gate_cell_is_preserved": True,
        "decision_matches_all_gate_conjunction": True,
        "no_training_recalibration_or_checkpoint_selection": True,
        "no_target_assets_or_2026_outcomes": True,
        "evaluation_execution_count_is_exactly_one": True,
    }
    audit = {"checks": checks, "passed": bool(all(checks.values()))}
    result: dict[str, object] = {
        "version": VERSION,
        "mode": "evaluate",
        "decision": decision,
        "evaluation_spec": spec,
        "prepare_completion_receipt_sha256": prepared[
            "completion_receipt_sha256"
        ],
        "outcome_receipt_sha256": _sha256_file(
            output / "outcome_receipt.json"
        ),
        "evaluation_execution_count": 1,
        "predictive_metrics": predictive_summary,
        "fold_metrics": portfolio["fold_metrics"],
        "aggregate_metrics": portfolio["aggregate_metrics"],
        "bootstrap": bootstrap,
        "gate_result": gate_result,
        "summary": {
            "unique_signal_dates": int(
                screen["data_access"]["expected_unique_signal_dates"]
            ),
            "fold_signal_dates": len(predictive_daily),
            "triplet_contexts": len(context_metrics),
            "eligible_asset_dates": len(outcomes),
            "gate_cells": int(gate_result["cell_count"]),
            "gate_passed": bool(gate_result["passed"]),
            "target_asset_rows": 0,
            "post_2025_rows": 0,
        },
        "audit": audit,
    }
    _write_json_atomic(output / "audit.json", audit)
    _write_yaml_atomic(output / "resolved_config.yaml", config)
    _atomic_write_text(output / "report.md", _evaluation_report(result))
    core_files = [
        "unseal_receipt.json",
        "outcomes.parquet",
        "outcome_receipt.json",
        "predictive_context_metrics.parquet",
        "predictive_daily_metrics.parquet",
        "predictive_metrics.json",
        "daily_returns.parquet",
        "fold_metrics.json",
        "aggregate_metrics.json",
        "bootstrap.json",
        "gate_result.json",
        "audit.json",
        "resolved_config.yaml",
        "report.md",
    ]
    _seal_result_packet(output, result, core_files)
    return _validate_result_packet(
        output,
        spec["evaluation_spec_sha256"],
        "evaluate",
        required,
    )


def run_ranking_excess_development_screen(
    config: dict,
    mode: str,
) -> dict[str, object]:
    if mode == "preflight":
        return preflight_ranking_excess_development_screen(config)
    if mode == "prepare":
        return prepare_ranking_excess_development_screen(config)
    if mode == "evaluate":
        return evaluate_ranking_excess_development_screen(config)
    raise ValueError("V45 mode must be preflight, prepare, or evaluate")
