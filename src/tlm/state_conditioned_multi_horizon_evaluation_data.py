"""Exact projected data access and tensor construction for V59 prepare."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .core.artifacts import canonical_sha256
from .state_conditioned_multi_horizon_training_data import (
    BASE_FEATURE_COLUMNS,
    LOOKBACK_DAYS,
    TARGET_SYMBOLS,
)


SEQUENCE_BASE_COLUMNS = ("date", "sequence_start_date", "symbol")
TRAIN_LABEL_COLUMNS = (
    "date",
    "symbol",
    "target_h7_maturity_date",
    "target_h7_open_to_open_log_return",
    "multi_horizon_label_complete",
)
PANEL_COLUMNS = ("date", "symbol", *BASE_FEATURE_COLUMNS)
LogicalKey = tuple[pd.Timestamp, str]
Reader = Callable[..., pd.DataFrame]


def utc_day(value: object) -> pd.Timestamp:
    result = pd.Timestamp(value)
    result = result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")
    if result != result.normalize():
        raise ValueError(f"expected UTC daily value, got {value!r}")
    return result


def day_text(value: object) -> str:
    return utc_day(value).date().isoformat()


def jsonable(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return day_text(value)
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def key_records(keys: Collection[LogicalKey]) -> list[list[str]]:
    return [[day_text(date), symbol] for date, symbol in sorted(keys)]


def key_sha256(keys: Collection[LogicalKey]) -> str:
    return canonical_sha256(key_records(keys))


def _key_dnf(keys: Collection[LogicalKey]) -> list[list[tuple[str, str, object]]]:
    dates_by_symbol: dict[str, set[pd.Timestamp]] = defaultdict(set)
    for date, symbol in keys:
        dates_by_symbol[str(symbol)].add(utc_day(date))
    if not dates_by_symbol:
        raise ValueError("V59 projected key DNF cannot be empty")
    return [
        [
            ("symbol", "==", symbol),
            ("date", "in", sorted(dates_by_symbol[symbol])),
        ]
        for symbol in sorted(dates_by_symbol)
    ]


def _reader_filters(
    filters: Sequence[Sequence[tuple[str, str, object]]],
) -> list[list[tuple[str, str, object]]]:
    return [
        [
            (column, operator, list(value) if operator == "in" else value)
            for column, operator, value in conjunction
        ]
        for conjunction in filters
    ]


def _read(
    reader: Reader,
    path: str | Path,
    *,
    columns: Sequence[str],
    filters: Sequence[Sequence[tuple[str, str, object]]],
    label: str,
) -> pd.DataFrame:
    frame = reader(
        path,
        engine="pyarrow",
        columns=list(columns),
        filters=_reader_filters(filters),
    )
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{label} reader did not return a DataFrame")
    if list(frame.columns) != list(columns):
        raise ValueError(f"{label} projection drift")
    frame = frame.copy()
    for column in ("date", "sequence_start_date", "target_h7_maturity_date"):
        if column in frame:
            frame[column] = frame[column].map(utc_day)
    if "symbol" in frame:
        if frame["symbol"].isna().any():
            raise ValueError(f"{label} contains a null symbol")
        frame["symbol"] = frame["symbol"].astype(str)
    if frame.duplicated(["date", "symbol"]).any():
        raise ValueError(f"{label} contains duplicate logical keys")
    return frame


def _strict_bool(series: pd.Series, name: str) -> pd.Series:
    if series.isna().any() or not series.map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all():
        raise ValueError(f"{name} must contain only non-null booleans")
    return series.astype(bool)


def _frame_keys(frame: pd.DataFrame) -> frozenset[LogicalKey]:
    return frozenset(zip(frame["date"], frame["symbol"], strict=True))


def _origin_record(dataset_spec: Mapping[str, Any], origin: str) -> dict[str, Any]:
    matches = [
        row
        for row in dataset_spec["role_contract"]["origins"]
        if row["id"] == origin
    ]
    if len(matches) != 1:
        raise ValueError(f"V57 origin record is not unique: {origin}")
    return matches[0]


@dataclass(frozen=True)
class EvaluationCell:
    origin: str
    geometry: str
    fold: int
    train_symbols: tuple[str, ...]
    test_symbols: tuple[str, ...]
    train_triplets: tuple[tuple[str, str, str], ...]
    test_triplets: tuple[tuple[str, str, str], ...]
    train_flag: str
    development_flag: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    development_start: pd.Timestamp
    development_end: pd.Timestamp

    @property
    def cell_id(self) -> str:
        return f"{self.origin}|{self.geometry}|{self.fold}"


def build_evaluation_cell(
    phase_contract: Mapping[str, Any],
    dataset_spec: Mapping[str, Any],
    asset_folds: Mapping[str, Any],
    triplet_catalog: Mapping[str, Any],
    *,
    origin: str,
    geometry: str,
    fold: int,
) -> EvaluationCell:
    fold_rows = [row for row in asset_folds["folds"] if int(row["fold"]) == int(fold)]
    catalog_rows = [
        row for row in triplet_catalog["folds"] if int(row["fold"]) == int(fold)
    ]
    if len(fold_rows) != 1 or len(catalog_rows) != 1:
        raise ValueError("V32 fold metadata is not unique")
    fold_row, catalog = fold_rows[0], catalog_rows[0]
    train_symbols = tuple(map(str, fold_row["train_symbols"]))
    test_symbols = tuple(map(str, fold_row["test_symbols"]))
    if train_symbols != tuple(sorted(set(train_symbols))) or len(train_symbols) != 20:
        raise ValueError("V59 train symbols are not 20 lexical unique values")
    if test_symbols != tuple(sorted(set(test_symbols))) or len(test_symbols) != 10:
        raise ValueError("V59 test symbols are not 10 lexical unique values")
    if set(train_symbols).intersection(set(test_symbols) | TARGET_SYMBOLS):
        raise ValueError("V59 fold contains overlap or a sealed target")
    train_triplets = tuple(tuple(map(str, values)) for values in catalog["train_triplets"])
    test_triplets = tuple(tuple(map(str, values)) for values in catalog["test_triplets"])
    if train_triplets != tuple(combinations(train_symbols, 3)):
        raise ValueError("V59 train triplet catalog drift")
    if test_triplets != tuple(combinations(test_symbols, 3)) or len(test_triplets) != 120:
        raise ValueError("V59 test triplet catalog drift")
    origin_row = _origin_record(dataset_spec, origin)
    train = origin_row["geometries"][geometry]["train"]
    development = phase_contract["evaluation_cells"]["origins"][origin]
    flags = phase_contract["prepare_parquet_access_contract"]["exact_role_flags"][origin][geometry]
    expected_train = f"eligible_{origin}_{geometry}_train"
    expected_development = f"eligible_{origin}_{geometry}_development_evaluation"
    if flags != {"train": expected_train, "development": expected_development}:
        raise ValueError("V59 physical role flag drift")
    return EvaluationCell(
        origin=origin,
        geometry=geometry,
        fold=int(fold),
        train_symbols=train_symbols,
        test_symbols=test_symbols,
        train_triplets=train_triplets,
        test_triplets=test_triplets,
        train_flag=expected_train,
        development_flag=expected_development,
        train_start=utc_day(train["signal_start"]),
        train_end=utc_day(train["signal_end"]),
        development_start=utc_day(development["signal_start"]),
        development_end=utc_day(development["signal_end"]),
    )


class DensePanelIndex:
    """Calendar-addressed raw feature store that preserves absent rows as NaN."""

    def __init__(self, panel: pd.DataFrame) -> None:
        self._values: dict[str, tuple[pd.Timestamp, np.ndarray]] = {}
        for symbol, rows in panel.groupby("symbol", sort=True):
            rows = rows.sort_values("date")
            start = rows["date"].min()
            end = rows["date"].max()
            dates = pd.date_range(start, end, freq="D", tz="UTC")
            values = np.full((len(dates), len(BASE_FEATURE_COLUMNS)), np.nan, dtype=np.float64)
            offsets = (rows["date"] - start).dt.days.to_numpy(dtype=np.int64)
            values[offsets] = rows.loc[:, BASE_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
            self._values[str(symbol)] = (start, values)

    def final(self, date: pd.Timestamp, symbol: str) -> np.ndarray | None:
        entry = self._values.get(str(symbol))
        if entry is None:
            return None
        start, values = entry
        offset = int((utc_day(date) - start).days)
        if offset < 0 or offset >= len(values):
            return None
        result = values[offset]
        return result if np.isfinite(result).all() else None

    def window(
        self, start_date: pd.Timestamp, end_date: pd.Timestamp, symbol: str
    ) -> np.ndarray | None:
        entry = self._values.get(str(symbol))
        if entry is None:
            return None
        base, values = entry
        start = int((utc_day(start_date) - base).days)
        end = int((utc_day(end_date) - base).days) + 1
        if start < 0 or end > len(values) or end - start != LOOKBACK_DAYS:
            return None
        result = values[start:end]
        return result if np.isfinite(result).all() else None


@dataclass
class CellData:
    cell: EvaluationCell
    sequence: pd.DataFrame
    train_labels: pd.DataFrame
    panel: pd.DataFrame
    train_keys: frozenset[LogicalKey]
    development_keys: frozenset[LogicalKey]
    development_context_keys: frozenset[LogicalKey]
    sequence_start_by_key: dict[LogicalKey, pd.Timestamp]
    train_availability: dict[pd.Timestamp, tuple[str, ...]]
    development_availability: dict[pd.Timestamp, tuple[str, ...]]
    h7_by_key: dict[LogicalKey, float]
    panel_index: DensePanelIndex
    access_receipt: dict[str, Any]


def read_cell_data(
    cell: EvaluationCell,
    *,
    sequence_path: str | Path,
    labels_path: str | Path,
    panel_path: str | Path,
    reader: Reader = pd.read_parquet,
) -> CellData:
    sequence_columns = (*SEQUENCE_BASE_COLUMNS, cell.train_flag, cell.development_flag)
    sequence_filters = [
        [
            ("symbol", "in", cell.train_symbols),
            (cell.train_flag, "==", True),
            ("date", ">=", cell.train_start),
            ("date", "<=", cell.train_end),
        ],
        [
            ("symbol", "in", cell.test_symbols),
            (cell.development_flag, "==", True),
            ("date", ">=", cell.development_start),
            ("date", "<=", cell.development_end),
        ],
    ]
    sequence = _read(
        reader,
        sequence_path,
        columns=sequence_columns,
        filters=sequence_filters,
        label="V59 sequence roles",
    )
    if sequence.empty:
        raise ValueError("V59 sequence projection returned no rows")
    train_mask = _strict_bool(sequence[cell.train_flag], cell.train_flag)
    development_mask = _strict_bool(sequence[cell.development_flag], cell.development_flag)
    if (train_mask & development_mask).any() or (~train_mask & ~development_mask).any():
        raise ValueError("each V59 sequence row must have exactly one projected role")
    train_rows = sequence.loc[train_mask]
    development_rows = sequence.loc[development_mask]
    if train_rows.empty or development_rows.empty:
        raise ValueError("V59 train and development roles must both be non-empty")
    if set(train_rows["symbol"]) - set(cell.train_symbols):
        raise ValueError("V59 train-role symbol isolation drift")
    if set(development_rows["symbol"]) - set(cell.test_symbols):
        raise ValueError("V59 development-role symbol isolation drift")
    if set(sequence["symbol"]).intersection(TARGET_SYMBOLS):
        raise ValueError("V59 sequence projection loaded a sealed target")
    if not train_rows["date"].between(cell.train_start, cell.train_end).all():
        raise ValueError("V59 train sequence date drift")
    if not development_rows["date"].between(
        cell.development_start, cell.development_end
    ).all():
        raise ValueError("V59 development sequence date drift")
    lengths = (sequence["date"] - sequence["sequence_start_date"]).dt.days + 1
    if not (lengths == LOOKBACK_DAYS).all():
        raise ValueError("V59 sequence context is not exactly 256 calendar rows")
    train_keys = _frame_keys(train_rows)
    development_keys = _frame_keys(development_rows)
    if train_keys.intersection(development_keys):
        raise ValueError("V59 train and development logical keys overlap")
    sequence_start = {
        (row.date, row.symbol): row.sequence_start_date
        for row in sequence.itertuples(index=False)
    }
    context_keys: set[LogicalKey] = set()
    for row in development_rows.itertuples(index=False):
        dates = pd.date_range(row.sequence_start_date, row.date, freq="D", tz="UTC")
        if len(dates) != LOOKBACK_DAYS:
            raise ValueError("V59 development context calendar drift")
        context_keys.update((date, row.symbol) for date in dates)
    development_context_keys = frozenset(context_keys)

    label_filters = _key_dnf(train_keys)
    labels = _read(
        reader,
        labels_path,
        columns=TRAIN_LABEL_COLUMNS,
        filters=label_filters,
        label="V59 train-only h7 labels",
    )
    if _frame_keys(labels) != train_keys:
        raise ValueError("V59 train-label keys differ from exact train sequence keys")
    complete = _strict_bool(labels["multi_horizon_label_complete"], "multi_horizon_label_complete")
    h7 = labels["target_h7_open_to_open_log_return"]
    if h7.dtype != np.dtype("float64") or not complete.all() or not np.isfinite(h7).all():
        raise ValueError("V59 train h7 labels are not complete physical float64")
    if not labels["target_h7_maturity_date"].equals(labels["date"] + pd.Timedelta(days=8)):
        raise ValueError("V59 train h7 maturity drift")

    panel_keys = train_keys | development_context_keys
    panel_filters = _key_dnf(panel_keys)
    panel = _read(
        reader,
        panel_path,
        columns=PANEL_COLUMNS,
        filters=panel_filters,
        label="V59 train-final plus development-context panel",
    )
    if _frame_keys(panel) != panel_keys:
        raise ValueError("V59 panel keys differ from the exact projected union")
    if set(panel["symbol"]).intersection(TARGET_SYMBOLS):
        raise ValueError("V59 panel projection loaded a sealed target")

    def availability(frame: pd.DataFrame) -> dict[pd.Timestamp, tuple[str, ...]]:
        return {
            date: tuple(sorted(rows["symbol"].tolist()))
            for date, rows in frame.groupby("date", sort=True)
        }

    train_availability = availability(train_rows)
    development_availability = availability(development_rows)
    h7_by_key = {
        (row.date, row.symbol): float(row.target_h7_open_to_open_log_return)
        for row in labels.itertuples(index=False)
    }
    receipt = {
        "cell_id": cell.cell_id,
        "projected_columns": {
            "sequence_roles": list(sequence_columns),
            "train_labels": list(TRAIN_LABEL_COLUMNS),
            "panel": list(PANEL_COLUMNS),
            "development_labels": [],
            "development_outcomes": [],
        },
        "predicate_dnf": {
            "sequence_roles": jsonable(sequence_filters),
            "train_labels": jsonable(label_filters),
            "panel": jsonable(panel_filters),
        },
        "roles": {
            "train": {
                "date_min": day_text(train_rows["date"].min()),
                "date_max": day_text(train_rows["date"].max()),
                "symbols": sorted(train_rows["symbol"].unique()),
                "key_count": len(train_keys),
                "key_sha256": key_sha256(train_keys),
            },
            "development": {
                "date_min": day_text(development_rows["date"].min()),
                "date_max": day_text(development_rows["date"].max()),
                "symbols": sorted(development_rows["symbol"].unique()),
                "key_count": len(development_keys),
                "key_sha256": key_sha256(development_keys),
            },
            "development_context": {
                "key_count": len(development_context_keys),
                "key_sha256": key_sha256(development_context_keys),
            },
        },
        "train_label_key_count": len(train_keys),
        "train_label_key_sha256": key_sha256(train_keys),
        "development_sequence_key_count": len(development_keys),
        "development_sequence_key_sha256": key_sha256(development_keys),
        "feature_context_key_count": len(panel_keys),
        "feature_context_key_sha256": key_sha256(panel_keys),
        "development_outcome_value_reads": 0,
        "development_outcome_columns_materialized": [],
        "target_asset_loads": 0,
        "full_table_materializations": 0,
    }
    receipt["access_receipt_sha256"] = canonical_sha256(receipt)
    return CellData(
        cell=cell,
        sequence=sequence,
        train_labels=labels,
        panel=panel,
        train_keys=train_keys,
        development_keys=development_keys,
        development_context_keys=development_context_keys,
        sequence_start_by_key=sequence_start,
        train_availability=train_availability,
        development_availability=development_availability,
        h7_by_key=h7_by_key,
        panel_index=DensePanelIndex(panel),
        access_receipt=receipt,
    )


@dataclass(frozen=True)
class ScalerValues:
    origin: str
    geometry: str
    fold: int
    feature_names: tuple[str, ...]
    mean: np.ndarray
    scale: np.ndarray
    semantic_sha256: str

    def transform(self, raw: np.ndarray) -> np.ndarray:
        values = np.asarray(raw, dtype=np.float64)
        if values.shape[-1] != len(BASE_FEATURE_COLUMNS):
            raise ValueError("V59 raw tensor feature count drift")
        base = (values - self.mean) / self.scale
        source = values[..., 1]
        relative = source - source.mean(axis=-1, keepdims=True)
        if not np.allclose(relative.sum(axis=-1), 0.0, rtol=0.0, atol=1.0e-12):
            raise ValueError("V59 relative feature does not sum to zero")
        relative = relative / self.scale[1]
        result = np.concatenate([base, relative[..., None]], axis=-1)
        if not np.isfinite(result).all():
            raise ValueError("V59 scaler produced a non-finite tensor")
        return result.astype(np.float32)


def scaler_from_wrapper(wrapper: Mapping[str, Any]) -> ScalerValues:
    if wrapper.get("version") != "v58_train_only_scaler_v1":
        raise ValueError("V59 scaler wrapper version drift")
    value = wrapper.get("scaler")
    if not isinstance(value, dict):
        raise ValueError("V59 scaler wrapper lacks scaler payload")
    identity = f"{value.get('origin')}|{value.get('geometry')}|{int(value.get('fold', -1))}"
    if wrapper.get("scaler_id") != identity:
        raise ValueError("V59 scaler identity drift")
    body = dict(value)
    registered = body.pop("scaler_sha256", None)
    if canonical_sha256(body) != registered:
        raise ValueError("V59 scaler semantic hash drift")
    feature_names = tuple(map(str, value["feature_names"]))
    mean = np.asarray(value["mean"], dtype=np.float64)
    scale = np.asarray(value["standard_deviation"], dtype=np.float64)
    if feature_names != tuple(BASE_FEATURE_COLUMNS) or mean.shape != (8,) or scale.shape != (8,):
        raise ValueError("V59 scaler feature geometry drift")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("V59 scaler statistics are invalid")
    return ScalerValues(
        origin=str(value["origin"]),
        geometry=str(value["geometry"]),
        fold=int(value["fold"]),
        feature_names=feature_names,
        mean=mean,
        scale=scale,
        semantic_sha256=str(registered),
    )


def eligible_development_samples(
    data: CellData,
) -> list[tuple[pd.Timestamp, tuple[str, str, str]]]:
    available, _ = classify_development_samples(data)
    return available


def classify_development_samples(
    data: CellData,
) -> tuple[
    list[tuple[pd.Timestamp, tuple[str, str, str]]],
    list[dict[str, Any]],
]:
    result: list[tuple[pd.Timestamp, tuple[str, str, str]]] = []
    unavailable: list[dict[str, Any]] = []
    for date in pd.date_range(
        data.cell.development_start,
        data.cell.development_end,
        freq="D",
        tz="UTC",
    ):
        available = set(data.development_availability.get(date, ()))
        for triplet in data.cell.test_triplets:
            if not set(triplet).issubset(available):
                unavailable.append(
                    {
                        "date": day_text(date),
                        "triplet_key": "|".join(triplet),
                        "reason": "missing_registered_sequence_member",
                    }
                )
                continue
            starts = tuple(data.sequence_start_by_key.get((date, symbol)) for symbol in triplet)
            if None in starts or len(set(starts)) != 1:
                unavailable.append(
                    {
                        "date": day_text(date),
                        "triplet_key": "|".join(triplet),
                        "reason": "missing_or_nonshared_sequence_start",
                    }
                )
                continue
            if int((date - starts[0]).days) + 1 != LOOKBACK_DAYS:
                unavailable.append(
                    {
                        "date": day_text(date),
                        "triplet_key": "|".join(triplet),
                        "reason": "sequence_calendar_length_drift",
                    }
                )
                continue
            windows = [
                data.panel_index.window(starts[slot], date, symbol)
                for slot, symbol in enumerate(triplet)
            ]
            if any(window is None for window in windows):
                unavailable.append(
                    {
                        "date": day_text(date),
                        "triplet_key": "|".join(triplet),
                        "reason": "missing_or_nonfinite_exact_context",
                    }
                )
                continue
            result.append((date, triplet))
    if len(unavailable) != len(
        {(row["date"], row["triplet_key"]) for row in unavailable}
    ):
        raise RuntimeError("V59 unavailable context keys were recorded more than once")
    return result, unavailable


def materialize_development_batch(
    data: CellData,
    samples: Sequence[tuple[pd.Timestamp, tuple[str, str, str]]],
    scaler: ScalerValues,
) -> np.ndarray:
    if not samples:
        raise ValueError("V59 cannot materialize an empty inference batch")
    raw = np.empty((len(samples), LOOKBACK_DAYS, 3, 8), dtype=np.float64)
    for index, (date, triplet) in enumerate(samples):
        starts = [data.sequence_start_by_key.get((date, symbol)) for symbol in triplet]
        if None in starts or len(set(starts)) != 1:
            raise ValueError("V59 sample lacks a shared exact sequence start")
        for slot, symbol in enumerate(triplet):
            window = data.panel_index.window(starts[slot], date, symbol)
            if window is None:
                raise ValueError("V59 sample has unavailable exact context")
            raw[index, :, slot, :] = window
    transformed = scaler.transform(raw)
    if transformed.shape != (len(samples), LOOKBACK_DAYS, 3, 9):
        raise RuntimeError("V59 materialized tensor shape drift")
    return transformed


def exact_momentum_30(
    data: CellData, date: pd.Timestamp, triplet: Sequence[str]
) -> np.ndarray:
    start = utc_day(date) - pd.Timedelta(days=29)
    scores = np.full(3, np.nan, dtype=np.float64)
    for slot, symbol in enumerate(triplet):
        entry = data.panel_index.window(
            start - pd.Timedelta(days=LOOKBACK_DAYS - 30), date, str(symbol)
        )
        if entry is None:
            continue
        tail = entry[-30:, 0]
        if len(tail) == 30 and np.isfinite(tail).all():
            scores[slot] = float(tail.sum(dtype=np.float64))
    return scores


def ridge_training_arrays(
    data: CellData, scaler: ScalerValues
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    triplet_indexes = np.asarray(
        [
            [data.cell.train_symbols.index(symbol) for symbol in triplet]
            for triplet in data.cell.train_triplets
        ],
        dtype=np.int16,
    )
    chunks_x: list[np.ndarray] = []
    chunks_y: list[np.ndarray] = []
    used_triplets = 0
    for date in sorted(data.train_availability):
        available = set(data.train_availability[date])
        feature_rows: list[np.ndarray | None] = [
            data.panel_index.final(date, symbol) for symbol in data.cell.train_symbols
        ]
        target_rows = [
            data.h7_by_key.get((date, symbol)) for symbol in data.cell.train_symbols
        ]
        ready = np.asarray(
            [
                symbol in available
                and feature_rows[index] is not None
                and target_rows[index] is not None
                and np.isfinite(target_rows[index])
                for index, symbol in enumerate(data.cell.train_symbols)
            ],
            dtype=bool,
        )
        active = ready[triplet_indexes].all(axis=1)
        if not active.any():
            continue
        indexes = triplet_indexes[active]
        base = np.full((20, len(BASE_FEATURE_COLUMNS)), np.nan, dtype=np.float64)
        for index, row in enumerate(feature_rows):
            if row is not None:
                base[index] = row
        target = np.asarray(target_rows, dtype=np.float64)
        raw = base[indexes]
        transformed = scaler.transform(raw[:, None, :, :])[:, 0]
        chunks_x.append(transformed.reshape(-1, 9).astype(np.float64))
        chunks_y.append(target[indexes].reshape(-1))
        used_triplets += int(active.sum())
    if not chunks_x:
        raise ValueError("V59 Ridge fit population is empty")
    features = np.concatenate(chunks_x, axis=0)
    targets = np.concatenate(chunks_y, axis=0)
    if features.shape[0] != targets.shape[0] or features.shape[1] != 9:
        raise RuntimeError("V59 Ridge design geometry drift")
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise ValueError("V59 Ridge train population is non-finite")
    digest = hashlib.sha256()
    digest.update(features.astype("<f8", copy=False).tobytes(order="C"))
    digest.update(targets.astype("<f8", copy=False).tobytes(order="C"))
    receipt = {
        "train_signal_key_count": len(data.train_keys),
        "train_signal_key_sha256": key_sha256(data.train_keys),
        "finite_triplet_count": used_triplets,
        "finite_asset_row_count": len(targets),
        "design_target_bytes_sha256": digest.hexdigest(),
    }
    return features, targets, receipt
