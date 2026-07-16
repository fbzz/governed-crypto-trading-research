"""Fold-local, outcome-blind data access for frozen V83 training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
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
FEATURE_COLUMNS = (
    "date", "symbol", *BASE_FEATURES, "sequence_start_date", "sequence_ready"
)
LABEL_COLUMNS = (
    "signal_date", "symbol", "role", "target_21d_open_to_open_log_return",
    "label_complete", "sequence_ready", "eligible_fold_1", "eligible_fold_2",
    "eligible_fold_3",
)


@dataclass(frozen=True)
class V83FeatureScaler:
    feature_names: tuple[str, ...]
    median: tuple[float, ...]
    iqr: tuple[float, ...]
    fit_scope: str
    fit_start: str
    fit_end: str
    fit_rows: int

    def transform(self, values: np.ndarray) -> np.ndarray:
        if values.shape[-1] != len(self.feature_names):
            raise ValueError("V83 feature tensor must contain exactly eight features")
        result = (
            np.asarray(values, dtype=np.float32)
            - np.asarray(self.median, dtype=np.float32)
        ) / np.asarray(self.iqr, dtype=np.float32)
        if not np.isfinite(result).all():
            raise RuntimeError("V83 scaler produced non-finite values")
        return result.astype(np.float32, copy=False)

    def state_sha256(self) -> str:
        return canonical_sha256(asdict(self))


@dataclass(frozen=True)
class V83FoldScale:
    fold: int
    feature_scaler: V83FeatureScaler
    excess_rms_scale: float
    excess_values: int

    def record(self) -> dict[str, Any]:
        value = {
            "schema_version": "v83-low-turnover-rank-fold-scale/v1",
            "fold": self.fold,
            "feature_scaler": asdict(self.feature_scaler),
            "feature_scaler_state_sha256": self.feature_scaler.state_sha256(),
            "excess_rms_scale": self.excess_rms_scale,
            "excess_values": self.excess_values,
            "fit_role": "eligible_train_only",
            "shared_across_seeds": [42, 7, 123],
        }
        value["fold_scale_sha256"] = canonical_sha256(value)
        return value

    def state_sha256(self) -> str:
        return self.record()["fold_scale_sha256"]


@dataclass(frozen=True)
class V83SampleDraw:
    date: pd.Timestamp
    triplet: tuple[str, str, str]
    pair_index: int


class V83BalancedRotationSampler:
    """Frozen balanced partitions; independent of model seed and outcomes."""

    def __init__(
        self,
        availability: dict[pd.Timestamp, tuple[str, ...]],
        registered_triplets: tuple[tuple[str, str, str], ...],
        *,
        role: str,
    ) -> None:
        if role not in {"train", "internal_validation"}:
            raise ValueError(f"Unsupported V83 role: {role}")
        registered = set(registered_triplets)
        entries: list[tuple[pd.Timestamp, tuple[str, ...]]] = []
        for date, symbols in sorted(availability.items()):
            ordered = tuple(sorted(symbols))
            if len(ordered) >= 3:
                entries.append((pd.Timestamp(date), ordered))
        if not entries:
            raise RuntimeError("V83 role has no eligible dates")
        self.entries = tuple(entries)
        self.registered = registered
        self.role = role
        self.total_pairs = sum(
            len(tuple(combinations(symbols, 3))) for _, symbols in self.entries
        )

    def _base(self, epoch: int) -> list[V83SampleDraw]:
        draws: list[V83SampleDraw] = []
        pair_index = 0
        shifts = range(10) if self.role == "internal_validation" else None
        for date_index, (date, symbols) in enumerate(self.entries):
            active_shifts = shifts if shifts is not None else (
                (date_index + int(epoch)) % 10,
            )
            for shift in active_shifts:
                rotated = symbols[shift:] + symbols[:shift]
                usable = len(rotated) - (len(rotated) % 3)
                for start in range(0, usable, 3):
                    triplet = tuple(sorted(rotated[start : start + 3]))
                    if triplet not in self.registered:
                        raise RuntimeError("V83 rotation emitted an unregistered triplet")
                    draws.append(V83SampleDraw(date, triplet, pair_index))
                    pair_index += 1
        if not draws:
            raise RuntimeError("V83 deterministic rotation emitted no draws")
        return draws

    def sample(self, epoch: int, sample_count: int) -> list[V83SampleDraw]:
        if sample_count < 1:
            raise ValueError("V83 sample_count must be positive")
        base = self._base(0 if self.role == "internal_validation" else epoch)
        if sample_count <= len(base):
            return base[:sample_count]
        repeats, remainder = divmod(sample_count, len(base))
        return base * repeats + base[:remainder]


class V83TensorStore:
    def __init__(self, panel: pd.DataFrame, labels: pd.DataFrame, lookback: int) -> None:
        self.lookback = int(lookback)
        self.symbols = tuple(sorted(panel["symbol"].unique()))
        self.symbol_to_index = {value: index for index, value in enumerate(self.symbols)}
        self.dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
        self.date_to_index = {pd.Timestamp(value): index for index, value in enumerate(self.dates)}
        self.values = np.full(
            (len(self.symbols), len(self.dates), len(BASE_FEATURES)), np.nan, np.float32
        )
        for symbol, frame in panel.groupby("symbol", sort=True):
            indexes = [self.date_to_index[pd.Timestamp(value)] for value in frame["date"]]
            self.values[self.symbol_to_index[symbol], indexes] = frame[
                list(BASE_FEATURES)
            ].to_numpy(np.float32)
        self.targets = {
            (pd.Timestamp(row.signal_date), str(row.symbol)): float(
                row.target_21d_open_to_open_log_return
            )
            for row in labels.itertuples(index=False)
            if bool(row.label_complete) and bool(row.sequence_ready)
        }

    def materialize(
        self, draws: Iterable[V83SampleDraw], scaler: V83FeatureScaler
    ) -> tuple[np.ndarray, np.ndarray]:
        rows = list(draws)
        assets = np.asarray(
            [[self.symbol_to_index[s] for s in draw.triplet] for draw in rows],
            dtype=np.int64,
        )
        ends = np.asarray(
            [self.date_to_index[pd.Timestamp(draw.date)] for draw in rows], dtype=np.int64
        )
        if len(rows) == 0 or int(ends.min()) < self.lookback - 1:
            raise RuntimeError("V83 draw lacks its exact 128-day context")
        times = ends[:, None] + np.arange(-self.lookback + 1, 1)[None, :]
        base = self.values[assets[:, None, :], times[:, :, None], :]
        if not np.isfinite(base).all():
            raise RuntimeError("V83 eligible context contains non-finite features")
        targets = np.asarray(
            [
                [self.targets[(pd.Timestamp(draw.date), symbol)] for symbol in draw.triplet]
                for draw in rows
            ],
            dtype=np.float32,
        )
        if not np.isfinite(targets).all():
            raise RuntimeError("V83 sampled a missing or non-finite target")
        return scaler.transform(base), targets


@dataclass
class V83FoldTrainingData:
    fold: int
    train_symbols: tuple[str, ...]
    heldout_symbols: tuple[str, ...]
    registered_triplets: tuple[tuple[str, str, str], ...]
    store: V83TensorStore
    train_availability: dict[pd.Timestamp, tuple[str, ...]]
    validation_availability: dict[pd.Timestamp, tuple[str, ...]]
    scale: V83FoldScale
    access_receipt: dict[str, Any]

    def sampler(self, *, seed: int, role: str) -> V83BalancedRotationSampler:
        del seed  # frozen schedule is deliberately seed-independent
        availability = (
            self.train_availability if role == "train" else self.validation_availability
        )
        return V83BalancedRotationSampler(
            availability, self.registered_triplets, role=role
        )


def _read(path: Path, columns: tuple[str, ...], symbols: tuple[str, ...], date_name: str) -> pd.DataFrame:
    table = ds.dataset(path, format="parquet").to_table(
        columns=list(columns),
        filter=ds.field("symbol").isin(list(symbols)),
        use_threads=False,
    )
    frame = table.to_pandas()
    frame[date_name] = pd.to_datetime(frame[date_name], utc=True)
    frame["symbol"] = frame["symbol"].astype(str)
    if list(frame.columns) != list(columns) or frame.duplicated([date_name, "symbol"]).any():
        raise RuntimeError(f"V83 projected schema/key drift: {path}")
    loaded = set(frame["symbol"].unique())
    if loaded != set(symbols) or loaded.intersection(TARGET_SYMBOLS):
        raise RuntimeError(f"V83 fold-local symbol predicate drift: {path}")
    return frame.sort_values([date_name, "symbol"]).reset_index(drop=True)


def _availability(labels: pd.DataFrame, role: str) -> dict[pd.Timestamp, tuple[str, ...]]:
    subset = labels.loc[
        (labels["role"] == role) & labels["label_complete"] & labels["sequence_ready"]
    ]
    result = {
        pd.Timestamp(date): tuple(sorted(frame["symbol"].unique()))
        for date, frame in subset.groupby("signal_date", sort=True)
        if len(frame["symbol"].unique()) >= 3
    }
    if not result:
        raise RuntimeError(f"V83 has no eligible {role} population")
    return result


def _fit_feature_scaler(
    panel: pd.DataFrame, train: dict[pd.Timestamp, tuple[str, ...]], lookback: int
) -> V83FeatureScaler:
    keys: set[tuple[pd.Timestamp, str]] = set()
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    date_to_index = {pd.Timestamp(value): index for index, value in enumerate(dates)}
    for signal_date, symbols in train.items():
        end = date_to_index[signal_date]
        for date in dates[end - lookback + 1 : end + 1]:
            keys.update((pd.Timestamp(date), symbol) for symbol in symbols)
    mask = [
        (pd.Timestamp(date), symbol) in keys
        for date, symbol in zip(panel["date"], panel["symbol"], strict=True)
    ]
    cells = panel.loc[mask]
    values = cells[list(BASE_FEATURES)].to_numpy(np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise RuntimeError("V83 train-only scaler population is invalid")
    median = np.median(values, axis=0)
    q75, q25 = np.percentile(values, [75, 25], axis=0)
    iqr = np.maximum(q75 - q25, 1.0e-6)
    return V83FeatureScaler(
        BASE_FEATURES,
        tuple(float(value) for value in median),
        tuple(float(value) for value in iqr),
        "unique_train_sequence_feature_cells_only",
        str(cells["date"].min().date()),
        str(cells["date"].max().date()),
        int(len(cells)),
    )


def _fit_excess_rms(
    labels: pd.DataFrame,
    train: dict[pd.Timestamp, tuple[str, ...]],
    registered: set[tuple[str, str, str]],
) -> tuple[float, int]:
    targets = {
        (pd.Timestamp(row.signal_date), str(row.symbol)): float(
            row.target_21d_open_to_open_log_return
        )
        for row in labels.itertuples(index=False)
    }
    squared = 0.0
    count = 0
    for date, symbols in sorted(train.items()):
        for triplet in combinations(sorted(symbols), 3):
            if triplet not in registered:
                continue
            values = np.asarray([targets[(date, symbol)] for symbol in triplet], np.float64)
            excess = values - values.mean()
            squared += float(np.square(excess).sum())
            count += 3
    if count == 0:
        raise RuntimeError("V83 target-scale enumeration is empty")
    return max(float(np.sqrt(squared / count)), 1.0e-6), count


def read_v83_fold_training_data(
    *, root: Path, phase_contract: dict[str, Any], asset_folds: dict[str, Any],
    triplet_catalog: dict[str, Any], fold: int,
) -> V83FoldTrainingData:
    fold_entry = next(row for row in asset_folds["folds"] if int(row["fold"]) == fold)
    catalog = next(row for row in triplet_catalog["folds"] if int(row["fold"]) == fold)
    train_symbols = tuple(sorted(fold_entry["train_symbols"]))
    heldout = tuple(sorted(fold_entry["test_symbols"]))
    triplets = tuple(tuple(value) for value in catalog["train_triplets"])
    if triplets != tuple(combinations(train_symbols, 3)):
        raise RuntimeError("V83 triplet catalog drift")
    if set(train_symbols + heldout).intersection(TARGET_SYMBOLS):
        raise RuntimeError("V83 target asset entered a fold")
    feature_path = root / "data/processed/low_turnover_rank_development_features_v82.parquet"
    label_path = root / "data/processed/low_turnover_rank_development_labels_v82.parquet"
    panel = _read(feature_path, FEATURE_COLUMNS, train_symbols, "date")
    labels = _read(label_path, LABEL_COLUMNS, train_symbols, "signal_date")
    eligible = f"eligible_fold_{fold}"
    labels = labels.loc[labels[eligible]].reset_index(drop=True)
    if labels.empty or labels["signal_date"].max() > pd.Timestamp("2024-11-18", tz="UTC"):
        raise RuntimeError("V83 label chronology drift")
    train = _availability(labels, "train")
    validation = _availability(labels, "internal_validation")
    if set(train).intersection(validation):
        raise RuntimeError("V83 train and validation dates overlap")
    scaler = _fit_feature_scaler(panel, train, 128)
    excess_rms, excess_values = _fit_excess_rms(labels, train, set(triplets))
    scale = V83FoldScale(fold, scaler, excess_rms, excess_values)
    store = V83TensorStore(panel, labels, 128)
    train_sampler = V83BalancedRotationSampler(train, triplets, role="train")
    val_sampler = V83BalancedRotationSampler(
        validation, triplets, role="internal_validation"
    )
    bindings = phase_contract["input_contract"]["expected_file_sha256_by_path"]
    access = {
        "schema_version": "v83-low-turnover-rank-fold-access/v1",
        "fold": fold,
        "train_symbols": list(train_symbols),
        "heldout_symbols_loaded": [],
        "target_assets_loaded": [],
        "projected_columns": {"features": list(FEATURE_COLUMNS), "labels": list(LABEL_COLUMNS)},
        "rows": {"panel": len(panel), "labels": len(labels), "roles": len(labels), "scaler_fit": scaler.fit_rows},
        "eligible_pairs": {"train": train_sampler.total_pairs, "internal_validation": val_sampler.total_pairs},
        "signal_dates": {
            "train_start": str(min(train).date()), "train_end": str(max(train).date()),
            "internal_validation_start": str(min(validation).date()),
            "internal_validation_end": str(max(validation).date()),
        },
        "maximum_loaded_value_date": str(max(panel["date"].max(), labels["signal_date"].max()).date()),
        "rows_from_2025_or_later": 0,
        "adaptive_evaluation_role_column_loaded": False,
        "parquet_deserializations": 2,
        "feature_scaler_fit_role": "eligible_train_only",
        "target_scale_fit_role": "complete_lexical_train_triplet_enumeration_only",
        "target_scale_excess_values": excess_values,
        "outcome_rows_read": 0,
        "forbidden_columns_loaded": [],
        "previous_checkpoints_loaded": [],
        "predictions_written": False,
        "policy_actions_emitted": False,
        "performance_metrics_computed": False,
        "pnl_computed": False,
        "hyperparameters_changed": False,
        "input_file_sha256": {
            str(feature_path.relative_to(root)): bindings[str(feature_path.relative_to(root))],
            str(label_path.relative_to(root)): bindings[str(label_path.relative_to(root))],
        },
    }
    access["access_sha256"] = canonical_sha256(access)
    return V83FoldTrainingData(
        fold, train_symbols, heldout, triplets, store, train, validation, scale, access
    )
