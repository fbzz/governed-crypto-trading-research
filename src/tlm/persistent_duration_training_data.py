"""Fold-local, outcome-blind data access for frozen V77 training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from .core.artifacts import canonical_sha256


TARGET_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
BASE_FEATURES = (
    "log_open_to_open_return",
    "log_close_to_close_return",
    "log_high_low_range",
    "log_close_open_return",
    "log1p_quote_volume_change",
    "log1p_trade_count_change",
    "rolling_realized_volatility_7d",
    "rolling_realized_volatility_30d",
)
RELATIVE_SOURCE = "log_close_to_close_return"
PANEL_COLUMNS = ("date", "symbol", *BASE_FEATURES)
LABEL_COLUMNS = (
    "date",
    "symbol",
    "target_h1_open_to_open_log_return",
    "target_h3_open_to_open_log_return",
    "target_h7_open_to_open_log_return",
    "target_duration_days",
    "duration_right_censored",
    "h1_label_complete",
    "h3_label_complete",
    "h7_label_complete",
    "duration_label_complete",
    "persistent_label_complete",
)
ROLE_COLUMNS = (
    "date",
    "sequence_start_date",
    "symbol",
    "persistent_label_complete",
    "eligible_train",
    "eligible_internal_validation",
)


@dataclass(frozen=True)
class V77FeatureScaler:
    feature_names: tuple[str, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    source_relative_feature_index: int
    fit_scope: str
    fit_start: str
    fit_end: str
    fit_rows: int

    def transform(self, values: np.ndarray) -> np.ndarray:
        if values.shape[-1] != len(self.feature_names) + 1:
            raise ValueError("V77 feature tensor must contain eight base plus relative")
        result = np.asarray(values, dtype=np.float32).copy()
        mean = np.asarray(self.mean, dtype=np.float32)
        scale = np.asarray(self.scale, dtype=np.float32)
        result[..., :-1] = (result[..., :-1] - mean) / scale
        result[..., -1] = result[..., -1] / scale[
            self.source_relative_feature_index
        ]
        if not np.isfinite(result).all():
            raise RuntimeError("V77 scaler produced non-finite values")
        return result

    def state_sha256(self) -> str:
        return canonical_sha256(asdict(self))


@dataclass(frozen=True)
class V77FoldScale:
    fold: int
    feature_scaler: V77FeatureScaler

    def record(self) -> dict[str, Any]:
        value = {
            "schema_version": "v77-persistent-duration-fold-scale/v1",
            "fold": self.fold,
            "feature_scaler": asdict(self.feature_scaler),
            "feature_scaler_state_sha256": self.feature_scaler.state_sha256(),
            "fit_role": "eligible_train_only",
            "shared_across_seeds": [42, 7, 123],
        }
        value["fold_scale_sha256"] = canonical_sha256(value)
        return value


@dataclass(frozen=True)
class V77SampleDraw:
    date: pd.Timestamp
    triplet: tuple[str, str, str]
    pair_index: int


class V77ExactTripletSampler:
    """Deterministically sample the exact lexical role-date/triplet population."""

    def __init__(
        self,
        availability: dict[pd.Timestamp, tuple[str, ...]],
        registered_triplets: tuple[tuple[str, str, str], ...],
        *,
        seed: int,
        fold: int,
        role_code: int,
    ) -> None:
        registered = set(registered_triplets)
        entries: list[tuple[pd.Timestamp, tuple[tuple[str, str, str], ...]]] = []
        counts: list[int] = []
        for date, symbols in sorted(availability.items()):
            triplets = tuple(
                value
                for value in combinations(tuple(sorted(symbols)), 3)
                if value in registered
            )
            if triplets:
                entries.append((pd.Timestamp(date), triplets))
                counts.append(len(triplets))
        if not entries:
            raise RuntimeError("V77 role has no exact eligible triplets")
        self.entries = tuple(entries)
        self.cumulative = np.cumsum(counts, dtype=np.int64)
        self.total_pairs = int(self.cumulative[-1])
        self.seed = int(seed)
        self.fold = int(fold)
        self.role_code = int(role_code)

    def sample(self, epoch: int, sample_count: int) -> list[V77SampleDraw]:
        if sample_count < 1:
            raise ValueError("V77 sample_count must be positive")
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [self.seed, self.fold, self.role_code, int(epoch)]
            )
        )
        indexes = rng.integers(
            0, self.total_pairs, size=int(sample_count), dtype=np.int64
        )
        draws: list[V77SampleDraw] = []
        for raw in indexes:
            pair_index = int(raw)
            entry_index = int(
                np.searchsorted(self.cumulative, pair_index, side="right")
            )
            prior = int(self.cumulative[entry_index - 1]) if entry_index else 0
            date, triplets = self.entries[entry_index]
            draws.append(
                V77SampleDraw(
                    date=date,
                    triplet=triplets[pair_index - prior],
                    pair_index=pair_index,
                )
            )
        return draws


class V77TensorStore:
    def __init__(
        self, panel: pd.DataFrame, labels: pd.DataFrame, *, lookback_days: int
    ) -> None:
        self.lookback_days = int(lookback_days)
        self.symbols = tuple(sorted(str(value) for value in panel["symbol"].unique()))
        self.symbol_to_index = {
            symbol: index for index, symbol in enumerate(self.symbols)
        }
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
        self.targets: dict[
            tuple[pd.Timestamp, str], tuple[np.ndarray, int, bool]
        ] = {}
        for row in labels.itertuples(index=False):
            if not bool(row.persistent_label_complete):
                continue
            returns = np.asarray(
                [
                    row.target_h1_open_to_open_log_return,
                    row.target_h3_open_to_open_log_return,
                    row.target_h7_open_to_open_log_return,
                ],
                dtype=np.float32,
            )
            duration = int(row.target_duration_days)
            censored = bool(row.duration_right_censored)
            if not np.isfinite(returns).all() or not 1 <= duration <= 7:
                raise RuntimeError("V77 admitted an invalid persistent target")
            self.targets[(pd.Timestamp(row.date), str(row.symbol))] = (
                returns,
                duration,
                censored,
            )

    def materialize(
        self,
        draws: Iterable[V77SampleDraw],
        scaler: V77FeatureScaler,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        draw_list = list(draws)
        if not draw_list:
            raise ValueError("V77 cannot materialize an empty batch")
        assets = np.asarray(
            [
                [self.symbol_to_index[symbol] for symbol in draw.triplet]
                for draw in draw_list
            ],
            dtype=np.int64,
        )
        ends = np.asarray(
            [self.date_to_index[pd.Timestamp(draw.date)] for draw in draw_list],
            dtype=np.int64,
        )
        if int(ends.min()) < self.lookback_days - 1:
            raise RuntimeError("V77 draw lacks its exact 256-day context")
        times = ends[:, None] + np.arange(
            -self.lookback_days + 1, 1, dtype=np.int64
        )[None, :]
        base = self.values[assets[:, None, :], times[:, :, None], :]
        if not np.isfinite(base).all():
            raise RuntimeError("V77 eligible context contains non-finite features")
        relative = base[..., BASE_FEATURES.index(RELATIVE_SOURCE)]
        relative = relative - relative.mean(axis=2, keepdims=True)
        features = scaler.transform(
            np.concatenate([base, relative[..., None]], axis=-1)
        ).astype(np.float32, copy=False)
        returns = np.empty((len(draw_list), 3, 3), dtype=np.float32)
        durations = np.empty((len(draw_list), 3), dtype=np.int64)
        censored = np.empty((len(draw_list), 3), dtype=np.bool_)
        for sample, draw in enumerate(draw_list):
            for slot, symbol in enumerate(draw.triplet):
                target = self.targets.get((pd.Timestamp(draw.date), symbol))
                if target is None:
                    raise RuntimeError("V77 sampled a missing persistent target")
                returns[sample, slot] = target[0]
                durations[sample, slot] = target[1]
                censored[sample, slot] = target[2]
        return features, returns, durations, censored


@dataclass
class V77FoldTrainingData:
    fold: int
    train_symbols: tuple[str, ...]
    heldout_symbols: tuple[str, ...]
    registered_triplets: tuple[tuple[str, str, str], ...]
    store: V77TensorStore
    train_availability: dict[pd.Timestamp, tuple[str, ...]]
    validation_availability: dict[pd.Timestamp, tuple[str, ...]]
    scale: V77FoldScale
    access_receipt: dict[str, Any]

    def sampler(self, *, seed: int, role: str) -> V77ExactTripletSampler:
        if role == "train":
            availability, role_code = self.train_availability, 77
        elif role == "internal_validation":
            availability, role_code = self.validation_availability, 97
        else:
            raise ValueError(f"Unsupported V77 role: {role}")
        return V77ExactTripletSampler(
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
        raise RuntimeError(f"V77 projected column drift: {path}")
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame["symbol"] = frame["symbol"].astype(str)
    if frame.duplicated(["date", "symbol"]).any():
        raise RuntimeError(f"V77 duplicate date/symbol keys: {path}")
    loaded = set(frame["symbol"].unique())
    if loaded != set(symbols) or loaded.intersection(TARGET_SYMBOLS):
        raise RuntimeError(f"V77 fold-local symbol predicate drift: {path}")
    if not frame.empty and frame["date"].max() > maximum_date:
        raise RuntimeError(f"V77 predicate admitted a post-boundary row: {path}")
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def _availability(
    roles: pd.DataFrame, column: str, store: V77TensorStore
) -> dict[pd.Timestamp, tuple[str, ...]]:
    subset = roles.loc[roles[column] & roles["persistent_label_complete"]]
    result: dict[pd.Timestamp, tuple[str, ...]] = {}
    for date, frame in subset.groupby("date", sort=True):
        symbols = tuple(
            sorted(
                symbol
                for symbol in frame["symbol"].unique()
                if (pd.Timestamp(date), str(symbol)) in store.targets
            )
        )
        if len(symbols) >= 3:
            result[pd.Timestamp(date)] = symbols
    if not result:
        raise RuntimeError(f"V77 has no eligible role population: {column}")
    return result


def _fit_scaler(panel: pd.DataFrame, roles: pd.DataFrame) -> V77FeatureScaler:
    keys = roles.loc[roles["eligible_train"], ["date", "symbol"]]
    cells = panel.merge(keys, on=["date", "symbol"], how="inner")
    values = cells[list(BASE_FEATURES)].to_numpy(dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise RuntimeError("V77 train-only feature-scaler population is invalid")
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    scale[scale == 0.0] = 1.0
    return V77FeatureScaler(
        feature_names=BASE_FEATURES,
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        source_relative_feature_index=BASE_FEATURES.index(RELATIVE_SOURCE),
        fit_scope="eligible_train_unique_symbol_date_cells_only",
        fit_start=str(cells["date"].min().date()),
        fit_end=str(cells["date"].max().date()),
        fit_rows=int(len(cells)),
    )


def read_v77_fold_training_data(
    *,
    root: Path,
    phase_contract: dict[str, Any],
    asset_folds: dict[str, Any],
    triplet_catalog: dict[str, Any],
    fold: int,
) -> V77FoldTrainingData:
    fold_entry = next(
        value for value in asset_folds["folds"] if int(value["fold"]) == int(fold)
    )
    catalog_entry = next(
        value
        for value in triplet_catalog["folds"]
        if int(value["fold"]) == int(fold)
    )
    train_symbols = tuple(sorted(str(value) for value in fold_entry["train_symbols"]))
    heldout_symbols = tuple(sorted(str(value) for value in fold_entry["test_symbols"]))
    if (
        set(train_symbols).intersection(heldout_symbols)
        or set(train_symbols + heldout_symbols).intersection(TARGET_SYMBOLS)
    ):
        raise RuntimeError("V77 fold isolation drift")
    triplets = tuple(
        tuple(str(symbol) for symbol in value)
        for value in catalog_entry["train_triplets"]
    )
    if triplets != tuple(combinations(train_symbols, 3)):
        raise RuntimeError("V77 triplet catalog is not the exact lexical train set")

    paths = {
        "panel": root / "data/processed/selected_universe_panel_v32.parquet",
        "labels": root / "data/processed/persistent_duration_labels_v76.parquet",
        "roles": root
        / "data/processed/persistent_duration_sequence_roles_v76.parquet",
    }
    maximum = pd.Timestamp("2024-12-23", tz="UTC")
    roles = _read_projected(
        paths["roles"], ROLE_COLUMNS, train_symbols,
        minimum_date=None, maximum_date=maximum,
    )
    roles["sequence_start_date"] = pd.to_datetime(
        roles["sequence_start_date"], utc=True
    )
    roles = roles.loc[
        roles["eligible_train"] | roles["eligible_internal_validation"]
    ].reset_index(drop=True)
    if roles.empty or (roles["eligible_train"] & roles["eligible_internal_validation"]).any():
        raise RuntimeError("V77 role admission drift")
    minimum_context = pd.Timestamp(roles["sequence_start_date"].min())
    minimum_signal = pd.Timestamp(roles["date"].min())
    panel = _read_projected(
        paths["panel"], PANEL_COLUMNS, train_symbols,
        minimum_date=minimum_context, maximum_date=maximum,
    )
    labels = _read_projected(
        paths["labels"], LABEL_COLUMNS, train_symbols,
        minimum_date=minimum_signal, maximum_date=maximum,
    )
    store = V77TensorStore(panel, labels, lookback_days=256)
    train = _availability(roles, "eligible_train", store)
    validation = _availability(roles, "eligible_internal_validation", store)
    if set(train).intersection(validation):
        raise RuntimeError("V77 train and internal-validation dates overlap")
    scaler = _fit_scaler(panel, roles)
    scale = V77FoldScale(fold=int(fold), feature_scaler=scaler)
    train_sampler = V77ExactTripletSampler(
        train, triplets, seed=0, fold=int(fold), role_code=77
    )
    validation_sampler = V77ExactTripletSampler(
        validation, triplets, seed=0, fold=int(fold), role_code=97
    )
    bindings = phase_contract["input_contract"]["expected_file_sha256_by_path"]
    access = {
        "schema_version": "v77-persistent-duration-fold-access/v1",
        "fold": int(fold),
        "train_symbols": list(train_symbols),
        "heldout_symbols_loaded": [],
        "target_assets_loaded": [],
        "projected_columns": {
            "panel": list(PANEL_COLUMNS),
            "labels": list(LABEL_COLUMNS),
            "roles": list(ROLE_COLUMNS),
        },
        "rows": {
            "panel": int(len(panel)),
            "labels": int(len(labels)),
            "roles": int(len(roles)),
            "scaler_fit": int(scaler.fit_rows),
        },
        "eligible_pairs": {
            "train": train_sampler.total_pairs,
            "internal_validation": validation_sampler.total_pairs,
        },
        "signal_dates": {
            "train_start": str(min(train).date()),
            "train_end": str(max(train).date()),
            "internal_validation_start": str(min(validation).date()),
            "internal_validation_end": str(max(validation).date()),
        },
        "maximum_loaded_value_date": str(
            max(panel["date"].max(), labels["date"].max(), roles["date"].max()).date()
        ),
        "rows_from_2025_or_later": 0,
        "adaptive_evaluation_role_column_loaded": False,
        "parquet_deserializations": 3,
        "feature_scaler_fit_role": "eligible_train_only",
        "outcome_rows_read": 0,
        "forbidden_columns_loaded": [],
        "previous_checkpoints_loaded": [],
        "predictions_written": False,
        "policy_actions_emitted": False,
        "performance_metrics_computed": False,
        "pnl_computed": False,
        "hyperparameters_changed": False,
        "input_file_sha256": {
            path.relative_to(root).as_posix(): bindings[
                path.relative_to(root).as_posix()
            ]
            for path in paths.values()
        },
    }
    access["access_sha256"] = canonical_sha256(access)
    return V77FoldTrainingData(
        fold=int(fold),
        train_symbols=train_symbols,
        heldout_symbols=heldout_symbols,
        registered_triplets=triplets,
        store=store,
        train_availability=train,
        validation_availability=validation,
        scale=scale,
        access_receipt=access,
    )
