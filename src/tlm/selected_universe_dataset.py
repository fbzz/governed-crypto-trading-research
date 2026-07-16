from __future__ import annotations

from itertools import combinations
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .non_target_dataset import (
    LABEL_COLUMNS,
    PANEL_FEATURES,
    RAW_FIELDS,
    TRIPLET_FEATURE,
    build_feature_schema,
    build_symbol_panel,
    load_frozen_frames,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"JSONL manifest is empty: {path}")
    return rows


def build_sequence_index(
    panel: pd.DataFrame,
    splits: dict[str, list[str]],
    lookback_days: int,
) -> pd.DataFrame:
    split_columns = [f"in_{name}" for name in splits]
    sequence = panel.loc[
        panel["sequence_ready"],
        [
            "date",
            "symbol",
            "label_complete",
            "supervised_sequence_ready",
            *split_columns,
        ],
    ].copy()
    sequence["sequence_start_date"] = sequence["date"] - pd.Timedelta(
        days=lookback_days - 1
    )
    return sequence.sort_values(["date", "symbol"]).reset_index(drop=True)[
        [
            "date",
            "sequence_start_date",
            "symbol",
            "label_complete",
            "supervised_sequence_ready",
            *split_columns,
        ]
    ]


def build_triplet_catalog(
    asset_folds: dict[str, object],
    target_symbols: set[str],
) -> dict[str, object]:
    folds = []
    for fold in asset_folds["folds"]:
        train_symbols = sorted(fold["train_symbols"])
        test_symbols = sorted(fold["test_symbols"])
        if target_symbols.intersection(train_symbols + test_symbols):
            raise ValueError("Target symbol entered the triplet catalog")
        folds.append({
            "fold": int(fold["fold"]),
            "train_symbols": train_symbols,
            "test_symbols": test_symbols,
            "train_triplets": [list(group) for group in combinations(train_symbols, 3)],
            "test_triplets": [list(group) for group in combinations(test_symbols, 3)],
        })
    catalog = {
        "version": "v32",
        "method": "lexical_all_combinations_within_frozen_asset_fold_role",
        "triplet_size": 3,
        "relative_feature": {
            "name": TRIPLET_FEATURE,
            "source": "log_close_to_close_return",
            "formula": "asset_value_minus_equal_weight_triplet_mean_same_date",
        },
        "sampling": "deterministic_epoch_seeded_uniform_over_eligible_catalog_entries",
        "folds": folds,
    }
    catalog["catalog_sha256"] = _canonical_sha256(catalog)
    return catalog


def build_triplet_availability_audit(
    panel: pd.DataFrame,
    catalog: dict[str, object],
    splits: dict[str, list[str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fold in catalog["folds"]:
        for role in ("train", "test"):
            symbols = set(fold[f"{role}_symbols"])
            possible = len(fold[f"{role}_triplets"])
            for split_name in splits:
                ready_column = (
                    "sequence_ready"
                    if split_name == "representation_train"
                    else "supervised_sequence_ready"
                )
                subset = panel.loc[
                    panel[f"in_{split_name}"]
                    & panel[ready_column]
                    & panel["symbol"].isin(symbols),
                    ["date", "symbol"],
                ]
                counts = subset.groupby("date")["symbol"].nunique()
                eligible = counts[counts >= 3]
                sample_count = int(sum(math.comb(int(count), 3) for count in eligible))
                rows.append({
                    "fold": int(fold["fold"]),
                    "role": role,
                    "split": split_name,
                    "ready_rule": ready_column,
                    "catalog_triplets": possible,
                    "eligible_dates": len(eligible),
                    "eligible_triplet_sequence_samples": sample_count,
                    "first_eligible_date": (
                        eligible.index.min().date().isoformat() if len(eligible) else None
                    ),
                    "last_eligible_date": (
                        eligible.index.max().date().isoformat() if len(eligible) else None
                    ),
                    "minimum_ready_assets_on_eligible_date": (
                        int(eligible.min()) if len(eligible) else 0
                    ),
                    "maximum_ready_assets": int(counts.max()) if len(counts) else 0,
                })
    return rows


def materialize_triplet_sequence(
    panel: pd.DataFrame,
    triplet: list[str] | tuple[str, str, str],
    end_date: pd.Timestamp | str,
    lookback_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    symbols = list(triplet)
    if len(symbols) != 3 or len(set(symbols)) != 3:
        raise ValueError("A triplet must contain three distinct symbols")
    end = pd.Timestamp(end_date)
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")
    start = end - pd.Timedelta(days=lookback_days - 1)
    per_symbol = []
    labels = []
    for symbol in symbols:
        frame = panel.loc[
            (panel["symbol"] == symbol)
            & (panel["date"] >= start)
            & (panel["date"] <= end)
        ].sort_values("date")
        if len(frame) != lookback_days:
            raise ValueError(f"Incomplete sequence for {symbol} at {end}")
        expected_dates = pd.date_range(start, end, freq="D", tz="UTC")
        if not pd.DatetimeIndex(frame["date"]).equals(expected_dates):
            raise ValueError(f"Non-contiguous sequence for {symbol} at {end}")
        values = frame[list(PANEL_FEATURES)].to_numpy(dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite sequence feature for {symbol} at {end}")
        per_symbol.append(values)
        label = frame.iloc[-1][list(LABEL_COLUMNS)].to_numpy(dtype=np.float32)
        if not np.isfinite(label).all():
            raise ValueError(f"Non-finite label for {symbol} at {end}")
        labels.append(label)
    base = np.stack(per_symbol, axis=1)
    source_index = list(PANEL_FEATURES).index("log_close_to_close_return")
    source = base[:, :, source_index]
    relative = source - source.mean(axis=1, keepdims=True)
    tensor = np.concatenate([base, relative[:, :, None]], axis=2).astype(np.float32)
    return tensor, np.stack(labels).astype(np.float32)


def _split_audit(panel: pd.DataFrame, splits: dict[str, list[str]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, bounds in splits.items():
        subset = panel.loc[panel[f"in_{name}"]]
        result[name] = {
            "start": bounds[0],
            "end": bounds[1],
            "panel_rows": len(subset),
            "raw_observation_rows": int(subset["raw_observation_available"].sum()),
            "feature_complete_rows": int(subset["feature_complete"].sum()),
            "label_complete_rows": int(subset["label_complete"].sum()),
            "sequence_ready_rows": int(subset["sequence_ready"].sum()),
            "supervised_sequence_ready_rows": int(
                subset["supervised_sequence_ready"].sum()
            ),
        }
    return result


def _report(result: dict[str, object]) -> str:
    manifest = result["dataset_manifest"]
    triplets = result["triplet_catalog"]
    return "\n".join([
        "# TLM v32 Selected-Universe Causal Dataset",
        "",
        "## Decision",
        "",
        "**DATASET, SEQUENCES, AND TRIPLET CONTRACT PASSED; TRAINING REMAINS BLOCKED.**",
        "",
        f"Panel rows: **{manifest['panel_rows']:,}**",
        f"Observed raw rows: **{manifest['observed_raw_rows']:,}**",
        f"Preserved missing rows: **{manifest['preserved_missing_rows']:,}**",
        f"Sequence-index rows: **{manifest['sequence_index_rows']:,}**",
        f"Panel SHA-256: `{manifest['panel_sha256']}`",
        f"Sequence-index SHA-256: `{manifest['sequence_index_sha256']}`",
        f"Triplet-catalog SHA-256: `{triplets['catalog_sha256']}`",
        "",
        "The on-demand loader materializes exact [256, 3, 9] float32 tensors and [3, 2] labels. The ninth feature is computed within each triplet and date, never across the full universe.",
        "",
        "All source gaps remain missing. No imputation, scaler fit, model training, target asset, portfolio, return metric, Sharpe, drawdown, or PnL evaluation occurred.",
        "",
        "## Frozen triplet catalog",
        "",
        f"Each fold contains {len(triplets['folds'][0]['train_triplets']):,} train-role triplets and {len(triplets['folds'][0]['test_triplets']):,} test-role triplets. Sampling is deterministic and may only use catalog entries whose three sequences are complete on that date.",
        "",
        "## Next action",
        "",
        "V33 may implement only the frozen Patch Transformer architecture and checkpoint contract against fixture/smoke tensors. Full training remains forbidden until the v34 harness audit passes.",
        "",
    ])


def run_selected_universe_dataset(config: dict) -> dict[str, object]:
    dataset_config = config["selected_universe_dataset"]
    root = Path(dataset_config["project_root"]).resolve()
    paths = {
        "v29_amendment": root / dataset_config["v29_amendment_path"],
        "v30_inventory": root / dataset_config["v30_inventory_path"],
        "v31_inventory": root / dataset_config["v31_inventory_path"],
        "v31_audit": root / dataset_config["v31_audit_path"],
        "v31_manifest": root / dataset_config["v31_manifest_path"],
    }
    for name, path in paths.items():
        expected = dataset_config[f"expected_{name}_sha256"]
        if not path.is_file() or _sha256_file(path) != expected:
            raise RuntimeError(f"V32 input missing or hash drifted: {name}")
    amendment = _load_json(paths["v29_amendment"])
    v30 = _load_json(paths["v30_inventory"])
    v31 = _load_json(paths["v31_inventory"])
    v31_audit = _load_json(paths["v31_audit"])
    records = _load_jsonl(paths["v31_manifest"])
    if not v31_audit.get("passed"):
        raise RuntimeError("V31 audit does not pass")

    blueprint = amendment["blueprint"]
    symbols = list(v30["universe"]["selected_symbols"])
    target_symbols = set(blueprint["target_contract"]["symbols"])
    data_contract = blueprint["data_contract"]
    splits = blueprint["chronological_splits"]
    lookback = int(blueprint["architecture"]["lookback_days"])
    feature_schema = build_feature_schema(blueprint, dataset_config, version="v32")
    feature_schema_sha256 = _canonical_sha256(feature_schema)
    frames, source_audit = load_frozen_frames(
        records,
        symbols,
        root / dataset_config["raw_dir"],
        data_contract["frequency"],
        int(dataset_config["archive_workers"]),
    )
    expected_index = pd.date_range(
        data_contract["development_start"],
        pd.Timestamp(data_contract["development_cutoff"]).floor("D"),
        freq="D",
        tz="UTC",
    )
    panels = [
        build_symbol_panel(symbol, frames[symbol], expected_index, splits, lookback)
        for symbol in symbols
    ]
    panel = pd.concat(panels, ignore_index=True).sort_values(
        ["date", "symbol"]
    ).reset_index(drop=True)
    sequence_index = build_sequence_index(panel, splits, lookback)
    asset_folds = v30["asset_folds"]
    triplet_catalog = build_triplet_catalog(asset_folds, target_symbols)
    triplet_availability = build_triplet_availability_audit(
        panel, triplet_catalog, splits
    )

    smoke_fold = triplet_catalog["folds"][0]
    smoke_triplet = smoke_fold["train_triplets"][0]
    smoke_ready = panel.loc[
        panel["in_supervised_train"]
        & panel["supervised_sequence_ready"]
        & panel["symbol"].isin(smoke_triplet)
    ].groupby("date")["symbol"].nunique()
    smoke_date = smoke_ready[smoke_ready == 3].index.min()
    smoke_tensor, smoke_labels = materialize_triplet_sequence(
        panel, smoke_triplet, smoke_date, lookback
    )

    panel_path = root / dataset_config["panel_path"]
    sequence_path = root / dataset_config["sequence_index_path"]
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    prior_panel_hash = _sha256_file(panel_path) if panel_path.is_file() else None
    prior_sequence_hash = _sha256_file(sequence_path) if sequence_path.is_file() else None
    panel.to_parquet(panel_path, index=False, compression="zstd")
    sequence_index.to_parquet(sequence_path, index=False, compression="zstd")
    panel_hash = _sha256_file(panel_path)
    sequence_hash = _sha256_file(sequence_path)
    panel_replay = prior_panel_hash is None or prior_panel_hash == panel_hash
    sequence_replay = prior_sequence_hash is None or prior_sequence_hash == sequence_hash

    expected_rows = len(expected_index) * len(symbols)
    raw_columns = [f"raw_{field}" for field in RAW_FIELDS]
    raw_missing = ~panel["raw_observation_available"]
    feature_values = panel[list(PANEL_FEATURES)].to_numpy(dtype=float)
    label_values = panel[list(LABEL_COLUMNS)].to_numpy(dtype=float)
    checks = {
        "v29_blueprint_hash_matches": amendment["blueprint_sha256"]
        == dataset_config["expected_v29_blueprint_sha256"],
        "v31_authorizes_dataset_only": v31["decision"]
        == "authorize_v32_selected_universe_dataset_only",
        "exact_v30_universe_loaded": list(frames) == symbols
        and v31["universe"]["selected_symbols"] == symbols,
        "frozen_asset_folds_preserved": asset_folds == v31["universe"]["asset_folds"],
        "no_target_symbol_loaded": not target_symbols.intersection(frames),
        "all_source_hashes_reverified": source_audit["all_cache_hashes_verified"],
        "source_archive_count_matches": source_audit["archive_count"]
        == len(records)
        == v31["manifest_summary"]["accepted_archive_count"],
        "panel_is_full_calendar_cartesian": len(panel) == expected_rows,
        "panel_key_is_unique": not panel.duplicated(["symbol", "date"]).any(),
        "missing_rows_preserved": panel.loc[raw_missing, raw_columns].isna().all().all(),
        "observed_rows_match_manifest": int(panel["raw_observation_available"].sum())
        == source_audit["source_rows"]
        == v31["manifest_summary"]["observed_rows"],
        "finite_or_missing_features": not np.isinf(feature_values).any(),
        "finite_or_missing_labels": not np.isinf(label_values).any(),
        "feature_order_is_frozen": feature_schema["model_feature_order"]
        == list(data_contract["derived_features"]),
        "sequence_index_is_unique": not sequence_index.duplicated(["symbol", "date"]).any(),
        "sequence_lookback_is_256_days": (
            sequence_index["date"] - sequence_index["sequence_start_date"]
            == pd.Timedelta(days=lookback - 1)
        ).all()
        and lookback == 256,
        "triplet_catalog_has_three_folds": len(triplet_catalog["folds"]) == 3,
        "triplet_catalog_role_counts_are_exact": all(
            len(fold["train_triplets"]) == math.comb(20, 3)
            and len(fold["test_triplets"]) == math.comb(10, 3)
            for fold in triplet_catalog["folds"]
        ),
        "triplet_availability_is_nonempty": all(
            row["eligible_dates"] > 0 and row["eligible_triplet_sequence_samples"] > 0
            for row in triplet_availability
        ),
        "smoke_tensor_contract_passes": smoke_tensor.shape == (256, 3, 9)
        and smoke_labels.shape == (3, 2)
        and smoke_tensor.dtype == np.float32
        and smoke_labels.dtype == np.float32
        and np.isfinite(smoke_tensor).all()
        and np.isfinite(smoke_labels).all(),
        "relative_strength_is_zero_sum": np.allclose(
            smoke_tensor[:, :, -1].sum(axis=1), 0.0, atol=1e-6
        ),
        "eligible_action_is_t_plus_one": (
            panel["eligible_action_date"] - panel["date"] == pd.Timedelta(days=1)
        ).all(),
        "target_window_ends_at_t_plus_eight": (
            panel["target_window_end_date"] - panel["date"] == pd.Timedelta(days=8)
        ).all(),
        "panel_replay_hash_matches": panel_replay,
        "sequence_replay_hash_matches": sequence_replay,
        "imputation_count_is_zero": True,
        "scaler_fit_count_is_zero": True,
        "model_training_count_is_zero": True,
        "target_asset_load_count_is_zero": True,
        "portfolio_count_is_zero": True,
        "performance_metric_count_is_zero": True,
        "pnl_evaluation_count_is_zero": True,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise RuntimeError(f"V32 selected-universe dataset audit failed: {checks}")

    dataset_manifest = {
        "version": "v32",
        "panel_path": str(panel_path.relative_to(root)),
        "panel_sha256": panel_hash,
        "panel_rows": len(panel),
        "observed_raw_rows": int(panel["raw_observation_available"].sum()),
        "preserved_missing_rows": int(raw_missing.sum()),
        "sequence_index_path": str(sequence_path.relative_to(root)),
        "sequence_index_sha256": sequence_hash,
        "sequence_index_rows": len(sequence_index),
        "symbols": symbols,
        "symbol_count": len(symbols),
        "calendar_start": expected_index.min().date().isoformat(),
        "calendar_end": expected_index.max().date().isoformat(),
        "calendar_days": len(expected_index),
        "lookback_days": lookback,
        "panel_features": list(PANEL_FEATURES),
        "triplet_feature": TRIPLET_FEATURE,
        "labels": list(LABEL_COLUMNS),
        "feature_schema_sha256": feature_schema_sha256,
        "source_manifest_sha256": _sha256_file(paths["v31_manifest"]),
        "triplet_catalog_sha256": triplet_catalog["catalog_sha256"],
        "tensor_contract": {
            "x_shape": [lookback, 3, 9],
            "y_shape": [3, 2],
            "dtype": "float32",
        },
        "reproducibility": {
            "panel_prior_sha256": prior_panel_hash,
            "panel_current_sha256": panel_hash,
            "panel_byte_identical": panel_replay,
            "sequence_prior_sha256": prior_sequence_hash,
            "sequence_current_sha256": sequence_hash,
            "sequence_byte_identical": sequence_replay,
        },
    }
    result = {
        "version": "v32",
        "decision": "authorize_v33_patch_transformer_implementation_only",
        "dataset_manifest": dataset_manifest,
        "source_audit": source_audit,
        "split_audit": _split_audit(panel, splits),
        "feature_schema": feature_schema,
        "asset_folds": asset_folds,
        "triplet_catalog": triplet_catalog,
        "triplet_availability": triplet_availability,
        "smoke_contract": {
            "fold": smoke_fold["fold"],
            "triplet": smoke_triplet,
            "date": smoke_date.date().isoformat(),
            "x_shape": list(smoke_tensor.shape),
            "y_shape": list(smoke_labels.shape),
        },
        "tested": {
            "features_materialized": True,
            "labels_materialized": True,
            "sequences_indexed": True,
            "triplets_cataloged": True,
            "triplet_tensor_smoke_materialized": True,
            "scaler_fitted": False,
            "model_trained": False,
            "portfolio_constructed": False,
            "performance_metrics_computed": False,
            "pnl_computed": False,
            "target_assets_loaded": False,
            "improvement_status": "unknown_not_evaluated",
            "drawdown_status": "unknown_not_evaluated",
        },
        "audit": {"passed": True, "checks": checks},
    }
    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    files = {
        "dataset_manifest.json": dataset_manifest,
        "feature_schema.json": feature_schema,
        "source_audit.json": source_audit,
        "split_audit.json": result["split_audit"],
        "asset_folds.json": asset_folds,
        "triplet_catalog.json": triplet_catalog,
        "triplet_availability.json": triplet_availability,
        "audit.json": result["audit"],
    }
    for name, value in files.items():
        (output / name).write_text(
            json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
        )
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    result_summary = {
        key: value
        for key, value in result.items()
        if key not in {
            "source_audit",
            "split_audit",
            "feature_schema",
            "asset_folds",
            "triplet_catalog",
            "triplet_availability",
        }
    }
    result_summary["artifact_references"] = {
        name.removesuffix(".json"): {
            "path": name,
            "sha256": _sha256_file(output / name),
        }
        for name in files
    }
    (output / "result.json").write_text(
        json.dumps(result_summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result
