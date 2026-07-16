"""Exact fold-local data access for frozen V63 non-target training.

Only the projections registered in ``research/phase_contracts/v063.yaml`` are
materialized.  Every read is predicate-pushed to the fold's train symbols and
the frozen train/development-validation signal range; target assets and heldout
fold assets never enter a returned frame.
"""

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
from .scientific_harness import FeatureScaler


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
PANEL_COLUMNS = (
    "date",
    "symbol",
    *BASE_FEATURES,
    "target_realized_volatility_7d",
)
LABEL_COLUMNS = (
    "date",
    "symbol",
    "target_h1_maturity_date",
    "target_h1_open_to_open_log_return",
    "h1_label_complete",
)
SEQUENCE_COLUMNS = (
    "date",
    "sequence_start_date",
    "symbol",
    "h1_label_complete",
    "eligible_train",
    "eligible_consumed_development_validation",
)


@dataclass(frozen=True)
class FoldScale:
    fold: int
    feature_scaler: FeatureScaler
    excess_rms: float
    market_rms: float
    exact_train_triplet_pairs: int
    exact_train_excess_cells: int

    def record(self) -> dict[str, Any]:
        value = {
            "schema_version": "v63-decoupled-rank-state-fold-scale/v1",
            "fold": self.fold,
            "feature_scaler": asdict(self.feature_scaler),
            "feature_scaler_state_sha256": self.feature_scaler.state_sha256(),
            "ranker_excess_rms": self.excess_rms,
            "state_market_rms": self.market_rms,
            "exact_train_triplet_pairs": self.exact_train_triplet_pairs,
            "exact_train_excess_cells": self.exact_train_excess_cells,
        }
        value["fold_scale_sha256"] = canonical_sha256(value)
        return value


@dataclass(frozen=True)
class SampleDraw:
    date: pd.Timestamp
    triplet: tuple[str, str, str]
    pair_index: int


class ExactTripletSampler:
    """Uniform sampling over the exact flattened date/triplet pair set."""

    def __init__(
        self,
        availability_by_date: dict[pd.Timestamp, tuple[str, ...]],
        registered_triplets: tuple[tuple[str, str, str], ...],
        *,
        seed: int,
        fold: int,
        role_code: int,
    ) -> None:
        self.seed = int(seed)
        self.fold = int(fold)
        self.role_code = int(role_code)
        triplet_set = set(registered_triplets)
        entries: list[tuple[pd.Timestamp, tuple[tuple[str, str, str], ...]]] = []
        counts: list[int] = []
        for date, symbols in sorted(availability_by_date.items()):
            eligible = tuple(
                triplet
                for triplet in combinations(tuple(sorted(symbols)), 3)
                if triplet in triplet_set
            )
            if eligible:
                entries.append((pd.Timestamp(date), eligible))
                counts.append(len(eligible))
        if not entries:
            raise ValueError("V63 found no eligible date/triplet pairs")
        self.entries = tuple(entries)
        self.cumulative = np.cumsum(counts, dtype=np.int64)
        self.total_pairs = int(self.cumulative[-1])

    def sample(self, epoch: int, sample_count: int) -> list[SampleDraw]:
        if sample_count < 1:
            raise ValueError("V63 sample_count must be positive")
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [self.seed, self.fold, self.role_code, int(epoch)]
            )
        )
        indexes = rng.integers(
            0, self.total_pairs, size=int(sample_count), dtype=np.int64
        )
        result: list[SampleDraw] = []
        for raw in indexes:
            pair_index = int(raw)
            entry_index = int(
                np.searchsorted(self.cumulative, pair_index, side="right")
            )
            prior = int(self.cumulative[entry_index - 1]) if entry_index else 0
            date, triplets = self.entries[entry_index]
            result.append(
                SampleDraw(
                    date=date,
                    triplet=triplets[pair_index - prior],
                    pair_index=pair_index,
                )
            )
        return result


class FoldTensorStore:
    """Calendar-aligned fold tensor store with exact label lookup."""

    def __init__(
        self,
        panel: pd.DataFrame,
        labels: pd.DataFrame,
        *,
        lookback_days: int,
    ) -> None:
        self.lookback_days = int(lookback_days)
        self.symbols = tuple(sorted(panel["symbol"].unique()))
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
        self.volatility = np.full(
            (len(self.symbols), len(self.dates)), np.nan, dtype=np.float32
        )
        for symbol, frame in panel.groupby("symbol", sort=True):
            indexes = np.asarray(
                [self.date_to_index[pd.Timestamp(value)] for value in frame["date"]],
                dtype=np.int64,
            )
            symbol_index = self.symbol_to_index[str(symbol)]
            self.values[symbol_index, indexes] = frame[list(BASE_FEATURES)].to_numpy(
                dtype=np.float32
            )
            self.volatility[symbol_index, indexes] = frame[
                "target_realized_volatility_7d"
            ].to_numpy(dtype=np.float32)
        self.returns: dict[tuple[pd.Timestamp, str], float] = {
            (pd.Timestamp(row.date), str(row.symbol)): float(
                row.target_h1_open_to_open_log_return
            )
            for row in labels.itertuples(index=False)
            if bool(row.h1_label_complete)
            and math.isfinite(float(row.target_h1_open_to_open_log_return))
        }

    def materialize(
        self,
        draws: Iterable[SampleDraw],
        scaler: FeatureScaler,
        *,
        require_targets: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        draw_list = list(draws)
        if not draw_list:
            raise ValueError("V63 cannot materialize an empty batch")
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
            raise RuntimeError("V63 draw lacks the exact 256-row context")
        times = ends[:, None] + np.arange(
            -self.lookback_days + 1, 1, dtype=np.int64
        )[None, :]
        base = self.values[assets[:, None, :], times[:, :, None], :]
        if not np.isfinite(base).all():
            raise RuntimeError("V63 eligible context contains non-finite features")
        source_index = BASE_FEATURES.index(RELATIVE_SOURCE)
        relative = base[..., source_index]
        relative = relative - relative.mean(axis=2, keepdims=True)
        features = scaler.transform_triplet_tensor(
            np.concatenate([base, relative[..., None]], axis=-1)
        )
        targets = np.zeros((len(draw_list), 3, 2), dtype=np.float32)
        if not require_targets:
            return features.astype(np.float32, copy=False), targets
        for sample_index, draw in enumerate(draw_list):
            date_index = self.date_to_index[pd.Timestamp(draw.date)]
            for asset_index, symbol in enumerate(draw.triplet):
                key = (pd.Timestamp(draw.date), symbol)
                if key not in self.returns:
                    raise RuntimeError("V63 sampled a missing H1 label")
                targets[sample_index, asset_index, 0] = self.returns[key]
                targets[sample_index, asset_index, 1] = self.volatility[
                    self.symbol_to_index[symbol], date_index
                ]
        if not np.isfinite(targets).all():
            raise RuntimeError("V63 sampled non-finite supervised targets")
        return features.astype(np.float32, copy=False), targets


@dataclass
class FoldTrainingData:
    fold: int
    train_symbols: tuple[str, ...]
    heldout_symbols: tuple[str, ...]
    registered_triplets: tuple[tuple[str, str, str], ...]
    panel: pd.DataFrame
    labels: pd.DataFrame
    roles: pd.DataFrame
    store: FoldTensorStore
    train_availability: dict[pd.Timestamp, tuple[str, ...]]
    validation_availability: dict[pd.Timestamp, tuple[str, ...]]
    supervised_train_availability: dict[pd.Timestamp, tuple[str, ...]]
    supervised_validation_availability: dict[pd.Timestamp, tuple[str, ...]]
    scale: FoldScale
    access_receipt: dict[str, Any]

    def sampler(self, *, seed: int, role: str) -> ExactTripletSampler:
        if role == "pretraining_train":
            availability = self.train_availability
            role_code = 11
        elif role == "pretraining_validation":
            availability = self.validation_availability
            role_code = 29
        elif role == "supervised_train":
            availability = self.supervised_train_availability
            role_code = 11
        elif role == "supervised_validation":
            availability = self.supervised_validation_availability
            role_code = 29
        else:
            raise ValueError(f"Unsupported V63 sampler role: {role}")
        return ExactTripletSampler(
            availability,
            self.registered_triplets,
            seed=seed,
            fold=self.fold,
            role_code=role_code,
        )


def _read_projected(
    path: Path,
    columns: tuple[str, ...],
    symbols: tuple[str, ...],
    *,
    minimum_date: pd.Timestamp | None = None,
    maximum_date: pd.Timestamp,
) -> pd.DataFrame:
    predicate = ds.field("symbol").isin(list(symbols)) & (
        ds.field("date") <= maximum_date.to_pydatetime()
    )
    if minimum_date is not None:
        predicate = predicate & (
            ds.field("date") >= minimum_date.to_pydatetime()
        )
    table = ds.dataset(path, format="parquet").to_table(
        columns=list(columns), filter=predicate, use_threads=False
    )
    frame = table.to_pandas()
    if list(frame.columns) != list(columns):
        raise RuntimeError(f"V63 projected column drift for {path}")
    frame["date"] = pd.to_datetime(frame["date"], utc=True)
    frame["symbol"] = frame["symbol"].astype(str)
    if frame.duplicated(["date", "symbol"]).any():
        raise RuntimeError(f"V63 duplicate date/symbol keys in {path}")
    loaded = set(frame["symbol"].unique())
    if loaded != set(symbols):
        raise RuntimeError(f"V63 predicate did not load exact fold symbols: {path}")
    if loaded.intersection(TARGET_SYMBOLS):
        raise RuntimeError("V63 target symbol reached fold-local data")
    return frame.sort_values(["date", "symbol"]).reset_index(drop=True)


def _availability(
    roles: pd.DataFrame, column: str
) -> dict[pd.Timestamp, tuple[str, ...]]:
    subset = roles.loc[roles[column], ["date", "symbol"]]
    return {
        pd.Timestamp(date): tuple(sorted(frame["symbol"].unique()))
        for date, frame in subset.groupby("date", sort=True)
    }


def _supervised_availability(
    availability: dict[pd.Timestamp, tuple[str, ...]],
    store: FoldTensorStore,
) -> dict[pd.Timestamp, tuple[str, ...]]:
    result: dict[pd.Timestamp, tuple[str, ...]] = {}
    for date, symbols in availability.items():
        date_index = store.date_to_index[pd.Timestamp(date)]
        eligible = tuple(
            symbol
            for symbol in symbols
            if (pd.Timestamp(date), symbol) in store.returns
            and math.isfinite(
                float(
                    store.volatility[
                        store.symbol_to_index[symbol], date_index
                    ]
                )
            )
        )
        if len(eligible) >= 3:
            result[pd.Timestamp(date)] = eligible
    if not result:
        raise RuntimeError("V63 has no finite supervised date/triplet pairs")
    return result


def _fit_feature_scaler(
    panel: pd.DataFrame,
    roles: pd.DataFrame,
    *,
    train_symbols: tuple[str, ...],
) -> FeatureScaler:
    train_keys = roles.loc[roles["eligible_train"], ["date", "symbol"]]
    cells = panel.merge(train_keys, on=["date", "symbol"], how="inner")
    values = cells[list(BASE_FEATURES)].to_numpy(dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise RuntimeError("V63 train-only scaler population is invalid")
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    scale[scale == 0.0] = 1.0
    return FeatureScaler(
        feature_names=BASE_FEATURES,
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        source_relative_feature_index=BASE_FEATURES.index(RELATIVE_SOURCE),
        fit_scope="eligible_train_unique_symbol_date_cells_only",
        fit_start=str(cells["date"].min().date()),
        fit_end=str(cells["date"].max().date()),
        fit_rows=int(len(cells)),
    )


def _fit_target_scales(
    store: FoldTensorStore,
    availability: dict[pd.Timestamp, tuple[str, ...]],
    registered_triplets: tuple[tuple[str, str, str], ...],
    *,
    floor: float,
) -> tuple[float, float, int, int]:
    allowed = set(registered_triplets)
    excess_sum_squares = 0.0
    market_sum_squares = 0.0
    pair_count = 0
    for date, symbols in sorted(availability.items()):
        for triplet in combinations(symbols, 3):
            if triplet not in allowed:
                continue
            values = np.asarray(
                [store.returns[(pd.Timestamp(date), symbol)] for symbol in triplet],
                dtype=np.float64,
            )
            market = float(values.mean())
            excess_sum_squares += float(np.square(values - market).sum())
            market_sum_squares += market * market
            pair_count += 1
    if pair_count < 1:
        raise RuntimeError("V63 target-scale fit found no exact train pairs")
    excess_cells = pair_count * 3
    excess = max(math.sqrt(excess_sum_squares / excess_cells), float(floor))
    market = max(math.sqrt(market_sum_squares / pair_count), float(floor))
    return excess, market, pair_count, excess_cells


def read_fold_training_data(
    *,
    root: Path,
    phase_contract: dict[str, Any],
    asset_folds: dict[str, Any],
    triplet_catalog: dict[str, Any],
    fold: int,
) -> FoldTrainingData:
    fold_entry = next(
        entry for entry in asset_folds["folds"] if int(entry["fold"]) == int(fold)
    )
    catalog_entry = next(
        entry
        for entry in triplet_catalog["folds"]
        if int(entry["fold"]) == int(fold)
    )
    train_symbols = tuple(sorted(str(value) for value in fold_entry["train_symbols"]))
    heldout_symbols = tuple(sorted(str(value) for value in fold_entry["test_symbols"]))
    if set(train_symbols).intersection(heldout_symbols) or set(
        train_symbols + heldout_symbols
    ).intersection(TARGET_SYMBOLS):
        raise RuntimeError("V63 fold symbol isolation drift")
    registered_triplets = tuple(
        tuple(str(symbol) for symbol in triplet)
        for triplet in catalog_entry["train_triplets"]
    )
    expected_triplets = tuple(combinations(train_symbols, 3))
    if registered_triplets != expected_triplets:
        raise RuntimeError("V63 train-triplet catalog is not the exact lexical set")

    bindings = phase_contract["input_contract"]["expected_file_sha256_by_path"]
    paths = {
        "panel": root / "data/processed/selected_universe_panel_v32.parquet",
        "labels": root / "data/processed/decoupled_rank_state_labels_v62.parquet",
        "roles": root
        / "data/processed/decoupled_rank_state_sequence_roles_v62.parquet",
    }
    maximum = pd.Timestamp("2025-12-23", tz="UTC")
    roles = _read_projected(
        paths["roles"], SEQUENCE_COLUMNS, train_symbols, maximum_date=maximum
    )
    roles["sequence_start_date"] = pd.to_datetime(
        roles["sequence_start_date"], utc=True
    )
    roles = roles.loc[
        roles["eligible_train"]
        | roles["eligible_consumed_development_validation"]
    ].reset_index(drop=True)
    if roles.empty:
        raise RuntimeError("V63 fold roles are empty")
    minimum = pd.Timestamp(roles["sequence_start_date"].min())
    panel = _read_projected(
        paths["panel"],
        PANEL_COLUMNS,
        train_symbols,
        minimum_date=minimum,
        maximum_date=maximum,
    )
    labels = _read_projected(
        paths["labels"],
        LABEL_COLUMNS,
        train_symbols,
        minimum_date=pd.Timestamp(roles["date"].min()),
        maximum_date=maximum,
    )
    labels["target_h1_maturity_date"] = pd.to_datetime(
        labels["target_h1_maturity_date"], utc=True
    )
    if labels["target_h1_maturity_date"].max() > pd.Timestamp(
        "2025-12-25", tz="UTC"
    ):
        raise RuntimeError("V63 loaded a post-contract H1 maturity")

    store = FoldTensorStore(panel, labels, lookback_days=256)
    train_availability = _availability(roles, "eligible_train")
    validation_availability = _availability(
        roles, "eligible_consumed_development_validation"
    )
    supervised_train_availability = _supervised_availability(
        train_availability, store
    )
    supervised_validation_availability = _supervised_availability(
        validation_availability, store
    )
    scaler = _fit_feature_scaler(
        panel, roles, train_symbols=train_symbols
    )
    excess, market, pair_count, excess_cells = _fit_target_scales(
        store,
        train_availability,
        registered_triplets,
        floor=1.0e-6,
    )
    scale = FoldScale(
        fold=int(fold),
        feature_scaler=scaler,
        excess_rms=excess,
        market_rms=market,
        exact_train_triplet_pairs=pair_count,
        exact_train_excess_cells=excess_cells,
    )
    access = {
        "schema_version": "v63-decoupled-rank-state-fold-access/v1",
        "fold": int(fold),
        "train_symbols": list(train_symbols),
        "heldout_symbols_loaded": [],
        "target_assets_loaded": [],
        "projected_columns": {
            "panel": list(PANEL_COLUMNS),
            "labels": list(LABEL_COLUMNS),
            "roles": list(SEQUENCE_COLUMNS),
        },
        "rows": {
            "panel": int(len(panel)),
            "labels": int(len(labels)),
            "roles": int(len(roles)),
            "scaler_fit": int(scaler.fit_rows),
        },
        "eligible_pairs": {
            "pretraining_train": int(
                ExactTripletSampler(
                    train_availability,
                    registered_triplets,
                    seed=0,
                    fold=int(fold),
                    role_code=11,
                ).total_pairs
            ),
            "pretraining_validation": int(
                ExactTripletSampler(
                    validation_availability,
                    registered_triplets,
                    seed=0,
                    fold=int(fold),
                    role_code=29,
                ).total_pairs
            ),
            "supervised_train": int(
                ExactTripletSampler(
                    supervised_train_availability,
                    registered_triplets,
                    seed=0,
                    fold=int(fold),
                    role_code=11,
                ).total_pairs
            ),
            "supervised_validation": int(
                ExactTripletSampler(
                    supervised_validation_availability,
                    registered_triplets,
                    seed=0,
                    fold=int(fold),
                    role_code=29,
                ).total_pairs
            ),
        },
        "maximum_signal_date": str(roles["date"].max().date()),
        "outcome_rows_read": 0,
        "forbidden_columns_loaded": [],
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
    return FoldTrainingData(
        fold=int(fold),
        train_symbols=train_symbols,
        heldout_symbols=heldout_symbols,
        registered_triplets=registered_triplets,
        panel=panel,
        labels=labels,
        roles=roles,
        store=store,
        train_availability=train_availability,
        validation_availability=validation_availability,
        supervised_train_availability=supervised_train_availability,
        supervised_validation_availability=supervised_validation_availability,
        scale=scale,
        access_receipt=access,
    )
