"""Exact fold-local data access for frozen V68 gate-only training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch

from .core.artifacts import canonical_sha256
from .decoupled_rank_state_harness import derive_state_features
from .decoupled_rank_state_training_data import (
    BASE_FEATURES,
    RELATIVE_SOURCE,
    ExactTripletSampler,
    SampleDraw,
)
from .scientific_harness import FeatureScaler


TARGET_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
PANEL_COLUMNS = ("date", "symbol", *BASE_FEATURES)
LABEL_COLUMNS = (
    "date",
    "symbol",
    "target_h1_maturity_date",
    "target_h1_open_to_open_log_return",
    "h1_label_complete",
)
ROLE_COLUMNS = (
    "date",
    "sequence_start_date",
    "symbol",
    "h1_label_complete",
    "eligible_v62_train",
    "gate_role",
    "eligible_gate_train",
    "eligible_gate_internal_validation",
)


@dataclass(frozen=True)
class V68FoldScale:
    fold: int
    feature_scaler: FeatureScaler
    source_v63_feature_scaler_state_sha256: str
    source_v63_fold_scale_sha256: str
    market_target_rms: float
    exact_gate_train_triplet_pairs: int

    def record(self) -> dict[str, Any]:
        value = {
            "schema_version": "v68-v64-r2-fold-scale/v1",
            "fold": self.fold,
            "feature_scaler": asdict(self.feature_scaler),
            "feature_scaler_state_sha256": self.feature_scaler.state_sha256(),
            "source_v63_feature_scaler_state_sha256": (
                self.source_v63_feature_scaler_state_sha256
            ),
            "source_v63_fold_scale_sha256": self.source_v63_fold_scale_sha256,
            "market_target_rms": self.market_target_rms,
            "market_target_fit_role": "gate_train_only",
            "exact_gate_train_triplet_pairs": self.exact_gate_train_triplet_pairs,
        }
        value["fold_scale_sha256"] = canonical_sha256(value)
        return value


class V68TensorStore:
    def __init__(
        self, panel: pd.DataFrame, labels: pd.DataFrame, *, lookback_days: int
    ) -> None:
        self.lookback_days = int(lookback_days)
        self.symbols = tuple(sorted(str(value) for value in panel["symbol"].unique()))
        self.symbol_to_index = {symbol: index for index, symbol in enumerate(self.symbols)}
        self.dates = pd.DatetimeIndex(
            sorted(pd.to_datetime(panel["date"], utc=True).unique())
        )
        self.date_to_index = {
            pd.Timestamp(date): index for index, date in enumerate(self.dates)
        }
        self.values = np.full(
            (len(self.symbols), len(self.dates), len(BASE_FEATURES)),
            np.nan,
            dtype=np.float32,
        )
        for symbol, frame in panel.groupby("symbol", sort=True):
            indexes = np.asarray(
                [self.date_to_index[pd.Timestamp(value)] for value in frame["date"]],
                dtype=np.int64,
            )
            self.values[self.symbol_to_index[str(symbol)], indexes] = frame[
                list(BASE_FEATURES)
            ].to_numpy(dtype=np.float32)
        self.returns = {
            (pd.Timestamp(row.date), str(row.symbol)): float(
                row.target_h1_open_to_open_log_return
            )
            for row in labels.itertuples(index=False)
            if bool(row.h1_label_complete)
            and math.isfinite(float(row.target_h1_open_to_open_log_return))
        }

    def triplet_market_return(self, date: pd.Timestamp, triplet: tuple[str, str, str]) -> float:
        values = [self.returns[(pd.Timestamp(date), symbol)] for symbol in triplet]
        return float(np.mean(values, dtype=np.float64))

    def materialize(
        self,
        draws: Iterable[SampleDraw],
        scaler: FeatureScaler,
        *,
        market_target_rms: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        draw_list = list(draws)
        if not draw_list:
            raise ValueError("V68 cannot materialize an empty batch")
        assets = np.asarray(
            [[self.symbol_to_index[symbol] for symbol in draw.triplet] for draw in draw_list],
            dtype=np.int64,
        )
        ends = np.asarray(
            [self.date_to_index[pd.Timestamp(draw.date)] for draw in draw_list],
            dtype=np.int64,
        )
        if int(ends.min()) < self.lookback_days - 1:
            raise RuntimeError("V68 draw lacks exact 256-day context")
        times = ends[:, None] + np.arange(
            -self.lookback_days + 1, 1, dtype=np.int64
        )[None, :]
        base = self.values[assets[:, None, :], times[:, :, None], :]
        if not np.isfinite(base).all():
            raise RuntimeError("V68 eligible context contains non-finite features")
        relative = base[..., BASE_FEATURES.index(RELATIVE_SOURCE)]
        relative = relative - relative.mean(axis=2, keepdims=True)
        features = scaler.transform_triplet_tensor(
            np.concatenate([base, relative[..., None]], axis=-1)
        ).astype(np.float32, copy=False)
        state = derive_state_features(torch.from_numpy(features)).numpy()
        targets = np.asarray(
            [self.triplet_market_return(draw.date, draw.triplet) for draw in draw_list],
            dtype=np.float32,
        ) / np.float32(market_target_rms)
        if not np.isfinite(state).all() or not np.isfinite(targets).all():
            raise RuntimeError("V68 materialized non-finite state or target")
        return state.astype(np.float32, copy=False), targets.astype(np.float32, copy=False)


@dataclass
class V68FoldTrainingData:
    fold: int
    train_symbols: tuple[str, ...]
    heldout_symbols: tuple[str, ...]
    registered_triplets: tuple[tuple[str, str, str], ...]
    store: V68TensorStore
    train_availability: dict[pd.Timestamp, tuple[str, ...]]
    validation_availability: dict[pd.Timestamp, tuple[str, ...]]
    scale: V68FoldScale
    access_receipt: dict[str, Any]

    def sampler(self, *, seed: int, role: str) -> ExactTripletSampler:
        if role == "gate_train":
            availability, role_code = self.train_availability, 41
        elif role == "gate_internal_validation":
            availability, role_code = self.validation_availability, 59
        else:
            raise ValueError(f"Unsupported V68 sampler role: {role}")
        return ExactTripletSampler(
            availability,
            self.registered_triplets,
            seed=int(seed),
            fold=self.fold,
            role_code=role_code,
        )


def _read_projected(
    path: Path,
    columns: tuple[str, ...],
    symbols: tuple[str, ...],
    *,
    minimum_date: pd.Timestamp | None,
    maximum_date: pd.Timestamp,
) -> pd.DataFrame:
    predicate = ds.field("symbol").isin(list(symbols)) & (
        ds.field("date") <= maximum_date.to_pydatetime()
    )
    if minimum_date is not None:
        predicate &= ds.field("date") >= minimum_date.to_pydatetime()
    table = ds.dataset(path, format="parquet").to_table(
        columns=list(columns), filter=predicate, use_threads=False
    )
    frame = table.to_pandas()
    if list(frame.columns) != list(columns):
        raise RuntimeError(f"V68 projected column drift: {path}")
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame["symbol"] = frame["symbol"].astype(str)
    if frame.duplicated(["date", "symbol"]).any():
        raise RuntimeError(f"V68 duplicate date/symbol keys: {path}")
    loaded = set(frame["symbol"].unique())
    if loaded != set(symbols) or loaded.intersection(TARGET_SYMBOLS):
        raise RuntimeError(f"V68 fold-local symbol predicate drift: {path}")
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def _availability(roles: pd.DataFrame, column: str) -> dict[pd.Timestamp, tuple[str, ...]]:
    subset = roles.loc[roles[column], ["date", "symbol"]]
    result = {
        pd.Timestamp(date): tuple(sorted(frame["symbol"].unique()))
        for date, frame in subset.groupby("date", sort=True)
        if len(frame["symbol"].unique()) >= 3
    }
    if not result:
        raise RuntimeError(f"V68 has no eligible pairs for {column}")
    return result


def _market_rms(
    store: V68TensorStore,
    availability: dict[pd.Timestamp, tuple[str, ...]],
    registered_triplets: tuple[tuple[str, str, str], ...],
) -> tuple[float, int]:
    registered = set(registered_triplets)
    sum_squares = 0.0
    pair_count = 0
    for date, symbols in sorted(availability.items()):
        for triplet in combinations(symbols, 3):
            if triplet not in registered:
                continue
            value = store.triplet_market_return(date, triplet)
            sum_squares += value * value
            pair_count += 1
    if pair_count < 1:
        raise RuntimeError("V68 market RMS fit has no gate-train pairs")
    rms = math.sqrt(sum_squares / pair_count)
    return (rms if rms > 0.0 else 1.0), pair_count


def _feature_scaler(record: dict[str, Any]) -> FeatureScaler:
    scaler = FeatureScaler(**record["feature_scaler"])
    if scaler.state_sha256() != record["feature_scaler_state_sha256"]:
        raise RuntimeError("V68 source V63 feature-scaler identity drift")
    return scaler


def read_v68_fold_training_data(
    *,
    root: Path,
    phase_contract: dict[str, Any],
    asset_folds: dict[str, Any],
    triplet_catalog: dict[str, Any],
    scaler_manifest: dict[str, Any],
    fold: int,
) -> V68FoldTrainingData:
    fold_entry = next(x for x in asset_folds["folds"] if int(x["fold"]) == int(fold))
    catalog = next(x for x in triplet_catalog["folds"] if int(x["fold"]) == int(fold))
    scaler_record = next(x for x in scaler_manifest["folds"] if int(x["fold"]) == int(fold))
    train_symbols = tuple(sorted(str(x) for x in fold_entry["train_symbols"]))
    heldout_symbols = tuple(sorted(str(x) for x in fold_entry["test_symbols"]))
    if set(train_symbols).intersection(heldout_symbols) or set(train_symbols).intersection(TARGET_SYMBOLS):
        raise RuntimeError("V68 fold isolation drift")
    triplets = tuple(tuple(str(symbol) for symbol in x) for x in catalog["train_triplets"])
    if triplets != tuple(combinations(train_symbols, 3)):
        raise RuntimeError("V68 triplet catalog is not exact lexical train set")

    paths = {
        "panel": root / "data/processed/selected_universe_panel_v32.parquet",
        "labels": root / "data/processed/v67_v64_r2_gate_labels.parquet",
        "roles": root / "data/processed/v67_v64_r2_gate_sequence_roles.parquet",
    }
    maximum = pd.Timestamp("2024-12-23", tz="UTC")
    roles = _read_projected(
        paths["roles"], ROLE_COLUMNS, train_symbols, minimum_date=None, maximum_date=maximum
    )
    roles["sequence_start_date"] = pd.to_datetime(roles["sequence_start_date"], utc=True)
    roles = roles.loc[
        roles["eligible_gate_train"] | roles["eligible_gate_internal_validation"]
    ].reset_index(drop=True)
    if roles.empty or set(roles["gate_role"].unique()) - {
        "gate_train", "gate_internal_validation"
    }:
        raise RuntimeError("V68 role admission drift")
    minimum = pd.Timestamp(roles["sequence_start_date"].min())
    panel = _read_projected(
        paths["panel"], PANEL_COLUMNS, train_symbols,
        minimum_date=minimum, maximum_date=maximum,
    )
    labels = _read_projected(
        paths["labels"], LABEL_COLUMNS, train_symbols,
        minimum_date=pd.Timestamp(roles["date"].min()), maximum_date=maximum,
    )
    labels["target_h1_maturity_date"] = pd.to_datetime(
        labels["target_h1_maturity_date"], utc=True
    )
    if labels["target_h1_maturity_date"].max() > pd.Timestamp("2024-12-25", tz="UTC"):
        raise RuntimeError("V68 loaded a post-contract H1 maturity")
    train = _availability(roles, "eligible_gate_train")
    validation = _availability(roles, "eligible_gate_internal_validation")
    if set(train).intersection(validation):
        raise RuntimeError("V68 gate roles overlap")
    store = V68TensorStore(panel, labels, lookback_days=256)
    scaler = _feature_scaler(scaler_record)
    market_rms, pair_count = _market_rms(store, train, triplets)
    scale = V68FoldScale(
        fold=int(fold),
        feature_scaler=scaler,
        source_v63_feature_scaler_state_sha256=scaler_record[
            "feature_scaler_state_sha256"
        ],
        source_v63_fold_scale_sha256=scaler_record["fold_scale_sha256"],
        market_target_rms=market_rms,
        exact_gate_train_triplet_pairs=pair_count,
    )
    bindings = phase_contract["input_contract"]["expected_file_sha256_by_path"]
    access = {
        "schema_version": "v68-v64-r2-fold-access/v1",
        "fold": int(fold),
        "train_symbols": list(train_symbols),
        "heldout_symbols_loaded": [],
        "target_assets_loaded": [],
        "projected_columns": {
            "panel": list(PANEL_COLUMNS), "labels": list(LABEL_COLUMNS), "roles": list(ROLE_COLUMNS)
        },
        "rows": {"panel": len(panel), "labels": len(labels), "roles": len(roles)},
        "eligible_pairs": {
            "gate_train": ExactTripletSampler(train, triplets, seed=0, fold=int(fold), role_code=41).total_pairs,
            "gate_internal_validation": ExactTripletSampler(validation, triplets, seed=0, fold=int(fold), role_code=59).total_pairs,
        },
        "maximum_signal_date": str(roles["date"].max().date()),
        "parquet_deserializations": 3,
        "market_target_scaler_fit_role": "gate_train_only",
        "outcome_rows_read": 0,
        "forbidden_columns_loaded": [],
        "predictions_written": False,
        "policy_actions_emitted": False,
        "performance_metrics_computed": False,
        "pnl_computed": False,
        "hyperparameters_changed": False,
        "input_file_sha256": {
            path.relative_to(root).as_posix(): bindings[path.relative_to(root).as_posix()]
            for path in paths.values()
        },
    }
    access["access_sha256"] = canonical_sha256(access)
    return V68FoldTrainingData(
        fold=int(fold), train_symbols=train_symbols, heldout_symbols=heldout_symbols,
        registered_triplets=triplets, store=store,
        train_availability=train, validation_availability=validation,
        scale=scale, access_receipt=access,
    )
