"""Contract-bound V58 non-target training-data helpers.

This module contains no model, optimizer, checkpoint, or artifact logic.  Real
Parquet access is isolated behind an injectable projected/filter reader so the
same access contract can be proved with synthetic frames.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


TARGET_SYMBOLS = frozenset({"BTCUSDT", "ETHUSDT", "SOLUSDT"})
LOOKBACK_DAYS = 256
MASTER_SEED = 20260714
REGISTERED_JOB_SEEDS = frozenset({42, 7, 123})
BASE_FEATURE_COLUMNS = (
    "log_open_to_open_return",
    "log_close_to_close_return",
    "log_high_low_range",
    "log_close_open_return",
    "log1p_quote_volume_change",
    "log1p_trade_count_change",
    "rolling_realized_volatility_7d",
    "rolling_realized_volatility_30d",
)
RELATIVE_FEATURE_COLUMN = "within_triplet_relative_strength"
MODEL_FEATURE_COLUMNS = (*BASE_FEATURE_COLUMNS, RELATIVE_FEATURE_COLUMN)
LABEL_COLUMNS = (
    "date",
    "symbol",
    "target_h7_maturity_date",
    "target_h1_open_to_open_log_return",
    "target_h3_open_to_open_log_return",
    "target_h7_open_to_open_log_return",
    "multi_horizon_label_complete",
)
TARGET_COLUMNS = LABEL_COLUMNS[3:6]
PANEL_COLUMNS = ("date", "symbol", *BASE_FEATURE_COLUMNS)
FORBIDDEN_COLUMN_PATTERNS = (
    "development_evaluation",
    "target_next_open_to_next_open_log_return",
    "target_realized_volatility_7d",
)

LogicalKey = tuple[pd.Timestamp, str]
FilterClause = tuple[str, str, Any]
FilterDNF = tuple[tuple[FilterClause, ...], ...]
ParquetReader = Callable[..., pd.DataFrame]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _utc_day(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    timestamp = (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )
    if timestamp != timestamp.normalize():
        raise ValueError(f"Expected a UTC daily calendar value, got {value!r}")
    return timestamp


def _day_text(value: object) -> str:
    return _utc_day(value).date().isoformat()


def _jsonable(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return _day_text(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _forbidden_columns(columns: Collection[str]) -> list[str]:
    return sorted(
        column
        for column in columns
        if any(pattern in column for pattern in FORBIDDEN_COLUMN_PATTERNS)
    )


@dataclass(frozen=True)
class JobCell:
    origin: str
    geometry: str
    fold: int
    train_symbols: tuple[str, ...]
    heldout_symbols: tuple[str, ...]
    train_triplets: tuple[tuple[str, str, str], ...]
    train_flag: str
    validation_flag: str
    train_signal_start: pd.Timestamp
    train_signal_end: pd.Timestamp
    validation_signal_start: pd.Timestamp
    validation_signal_end: pd.Timestamp

    @property
    def key(self) -> str:
        return f"{self.origin}_{self.geometry}_fold_{self.fold}"


@dataclass(frozen=True)
class ProjectedDNFRead:
    input_name: Literal["sequence", "labels", "panel"]
    path: str | Path
    columns: tuple[str, ...]
    filters: FilterDNF
    expected_logical_key_count: int | None

    def __post_init__(self) -> None:
        forbidden = _forbidden_columns(self.columns)
        if forbidden:
            raise ValueError(f"Forbidden V58 projection columns: {forbidden}")
        if not self.filters:
            raise ValueError(f"{self.input_name} DNF filter cannot be empty")

    def reader_kwargs(self) -> dict[str, object]:
        filters = []
        for conjunction in self.filters:
            filters.append([
                (column, operator, list(value) if operator == "in" else value)
                for column, operator, value in conjunction
            ])
        return {"columns": list(self.columns), "filters": filters}

    def receipt(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "projected_columns": list(self.columns),
            "filters_dnf": _jsonable(self.filters),
            "expected_logical_key_count": self.expected_logical_key_count,
        }


@dataclass(frozen=True)
class JobReadPlan:
    sequence: ProjectedDNFRead
    labels: ProjectedDNFRead
    panel: ProjectedDNFRead

    def receipt(self) -> dict[str, object]:
        return {
            "sequence": self.sequence.receipt(),
            "labels": self.labels.receipt(),
            "panel": self.panel.receipt(),
        }


def build_job_cell(
    phase_contract: Mapping[str, Any],
    asset_folds: Mapping[str, Any],
    triplet_catalog: Mapping[str, Any],
    *,
    origin: str,
    geometry: str,
    fold: int,
) -> JobCell:
    """Resolve one V58 origin/geometry/fold cell from frozen V32 metadata."""

    if phase_contract.get("phase") != "v58":
        raise ValueError("Job cell requires the frozen V58 phase contract")
    role_contract = phase_contract["grid_contract"]["origin_roles"]
    if origin not in role_contract or geometry not in role_contract[origin]:
        raise ValueError(f"Unregistered V58 origin/geometry: {origin}/{geometry}")
    fold = int(fold)
    folds = [entry for entry in asset_folds["folds"] if int(entry["fold"]) == fold]
    catalogs = [
        entry for entry in triplet_catalog["folds"] if int(entry["fold"]) == fold
    ]
    if len(folds) != 1 or len(catalogs) != 1:
        raise ValueError(f"Fold {fold} is not unique in the frozen V32 metadata")
    fold_entry, catalog_entry = folds[0], catalogs[0]
    train_symbols = tuple(str(value) for value in fold_entry["train_symbols"])
    heldout_symbols = tuple(str(value) for value in fold_entry["test_symbols"])
    if train_symbols != tuple(sorted(set(train_symbols))):
        raise ValueError("V32 train symbols are not unique lexical ascending")
    if heldout_symbols != tuple(sorted(set(heldout_symbols))):
        raise ValueError("V32 held-out symbols are not unique lexical ascending")
    if set(train_symbols).intersection(heldout_symbols):
        raise ValueError("V32 train and held-out symbols overlap")
    if TARGET_SYMBOLS.intersection(train_symbols):
        raise ValueError("A sealed target symbol entered a V58 train fold")
    if tuple(catalog_entry["train_symbols"]) != train_symbols:
        raise ValueError("V32 fold/catalog train-symbol drift")
    if tuple(catalog_entry["test_symbols"]) != heldout_symbols:
        raise ValueError("V32 fold/catalog held-out-symbol drift")
    train_triplets = tuple(
        tuple(str(symbol) for symbol in triplet)
        for triplet in catalog_entry["train_triplets"]
    )
    expected_triplets = tuple(combinations(train_symbols, 3))
    if train_triplets != expected_triplets:
        raise ValueError("V32 train triplets are not the exact lexical catalog")
    role = role_contract[origin][geometry]
    train_flag = str(role["train_flag"])
    validation_flag = str(role["validation_flag"])
    if train_flag != f"eligible_{origin}_{geometry}_train":
        raise ValueError("V58 train-role flag drift")
    if validation_flag != f"eligible_{origin}_{geometry}_validation":
        raise ValueError("V58 validation-role flag drift")
    if _forbidden_columns((train_flag, validation_flag)):
        raise ValueError("Development-evaluation flag entered a V58 job cell")
    train_start, train_end = map(_utc_day, role["train_signal_range"])
    validation_start, validation_end = map(
        _utc_day, role["validation_signal_range"]
    )
    if not train_start <= train_end < validation_start <= validation_end:
        raise ValueError("V58 train/validation chronology drift")
    return JobCell(
        origin=origin,
        geometry=geometry,
        fold=fold,
        train_symbols=train_symbols,
        heldout_symbols=heldout_symbols,
        train_triplets=train_triplets,
        train_flag=train_flag,
        validation_flag=validation_flag,
        train_signal_start=train_start,
        train_signal_end=train_end,
        validation_signal_start=validation_start,
        validation_signal_end=validation_end,
    )


def _key_filter_dnf(keys: Collection[LogicalKey]) -> FilterDNF:
    dates_by_symbol: dict[str, set[pd.Timestamp]] = defaultdict(set)
    for date, symbol in keys:
        dates_by_symbol[str(symbol)].add(_utc_day(date))
    return tuple(
        (
            ("symbol", "==", symbol),
            ("date", "in", tuple(sorted(dates_by_symbol[symbol]))),
        )
        for symbol in sorted(dates_by_symbol)
    )


def build_job_read_plan(
    cell: JobCell,
    *,
    sequence_path: str | Path,
    labels_path: str | Path,
    panel_path: str | Path,
    signal_keys: Collection[LogicalKey] | None = None,
    context_keys: Collection[LogicalKey] | None = None,
) -> JobReadPlan | ProjectedDNFRead:
    """Build canonical DNF projections, incrementally around the sequence read.

    With no keys, returns the sequence request.  Once exact signal/context keys
    have been learned from that projected result, returns the complete plan.
    """

    sequence = ProjectedDNFRead(
        input_name="sequence",
        path=sequence_path,
        columns=(
            "date",
            "sequence_start_date",
            "symbol",
            cell.train_flag,
            cell.validation_flag,
        ),
        filters=(
            (
                ("symbol", "in", cell.train_symbols),
                (cell.train_flag, "==", True),
                ("date", ">=", cell.train_signal_start),
                ("date", "<=", cell.train_signal_end),
            ),
            (
                ("symbol", "in", cell.train_symbols),
                (cell.validation_flag, "==", True),
                ("date", ">=", cell.validation_signal_start),
                ("date", "<=", cell.validation_signal_end),
            ),
        ),
        expected_logical_key_count=None,
    )
    if signal_keys is None and context_keys is None:
        return sequence
    if signal_keys is None or context_keys is None:
        raise ValueError("Both signal_keys and context_keys are required")
    signal_keys = frozenset((_utc_day(date), str(symbol)) for date, symbol in signal_keys)
    context_keys = frozenset((_utc_day(date), str(symbol)) for date, symbol in context_keys)
    if not signal_keys or not context_keys:
        raise ValueError("V58 signal/context key unions cannot be empty")
    return JobReadPlan(
        sequence=sequence,
        labels=ProjectedDNFRead(
            "labels",
            labels_path,
            LABEL_COLUMNS,
            _key_filter_dnf(signal_keys),
            len(signal_keys),
        ),
        panel=ProjectedDNFRead(
            "panel",
            panel_path,
            PANEL_COLUMNS,
            _key_filter_dnf(context_keys),
            len(context_keys),
        ),
    )


@dataclass
class JobTrainingData:
    cell: JobCell
    sequence: pd.DataFrame
    labels: pd.DataFrame
    panel: pd.DataFrame
    train_signal_keys: frozenset[LogicalKey]
    validation_signal_keys: frozenset[LogicalKey]
    context_keys: frozenset[LogicalKey]
    train_availability: Mapping[pd.Timestamp, tuple[str, ...]]
    validation_availability: Mapping[pd.Timestamp, tuple[str, ...]]
    sequence_start_by_key: Mapping[LogicalKey, pd.Timestamp]
    labels_by_key: Mapping[LogicalKey, tuple[float, float, float]]
    scaler_source_rows: pd.DataFrame
    read_plan: JobReadPlan
    access_receipt: dict[str, object]
    tensor_store: ExplicitCalendarTensorStore


def _read_projected(request: ProjectedDNFRead, reader: ParquetReader) -> pd.DataFrame:
    frame = reader(request.path, **request.reader_kwargs())
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{request.input_name} reader did not return a DataFrame")
    if list(frame.columns) != list(request.columns):
        raise ValueError(
            f"{request.input_name} reader projection drift: "
            f"expected {list(request.columns)}, got {list(frame.columns)}"
        )
    forbidden = _forbidden_columns(frame.columns)
    if forbidden:
        raise ValueError(
            f"{request.input_name} materialized forbidden columns: {forbidden}"
        )
    frame = frame.copy()
    if "date" in frame:
        frame["date"] = frame["date"].map(_utc_day)
    if "sequence_start_date" in frame:
        frame["sequence_start_date"] = frame["sequence_start_date"].map(_utc_day)
    if "target_h7_maturity_date" in frame:
        frame["target_h7_maturity_date"] = frame["target_h7_maturity_date"].map(
            _utc_day
        )
    if "symbol" in frame:
        if frame["symbol"].isna().any():
            raise ValueError(f"{request.input_name} contains a null symbol")
        frame["symbol"] = frame["symbol"].astype(str)
    return frame


def _frame_keys(frame: pd.DataFrame, name: str) -> frozenset[LogicalKey]:
    if frame.duplicated(["date", "symbol"]).any():
        raise ValueError(f"{name} contains duplicate symbol/date keys")
    return frozenset(zip(frame["date"], frame["symbol"], strict=True))


def _strict_bool(series: pd.Series, name: str) -> pd.Series:
    if series.isna().any() or not series.map(
        lambda value: isinstance(value, (bool, np.bool_))
    ).all():
        raise ValueError(f"{name} must contain only non-null booleans")
    return series.astype(bool)


def _availability(
    sequence: pd.DataFrame, mask: pd.Series
) -> dict[pd.Timestamp, tuple[str, ...]]:
    subset = sequence.loc[mask, ["date", "symbol"]]
    return {
        date: tuple(sorted(frame["symbol"].tolist()))
        for date, frame in subset.groupby("date", sort=True)
    }


def _role_receipt(
    keys: frozenset[LogicalKey],
    *,
    registered_start: pd.Timestamp,
    registered_end: pd.Timestamp,
) -> dict[str, object]:
    dates = sorted({date for date, _ in keys})
    symbols = sorted({symbol for _, symbol in keys})
    return {
        "registered_signal_start": _day_text(registered_start),
        "registered_signal_end": _day_text(registered_end),
        "materialized_signal_date_min": _day_text(dates[0]),
        "materialized_signal_date_max": _day_text(dates[-1]),
        "materialized_symbols": symbols,
        "materialized_symbol_count": len(symbols),
        "logical_key_count": len(keys),
    }


def read_job_training_data(
    cell: JobCell,
    *,
    sequence_path: str | Path,
    labels_path: str | Path,
    panel_path: str | Path,
    reader: ParquetReader = pd.read_parquet,
) -> JobTrainingData:
    """Read one job cell through exact projections and predicate DNF only."""

    sequence_request = build_job_read_plan(
        cell,
        sequence_path=sequence_path,
        labels_path=labels_path,
        panel_path=panel_path,
    )
    assert isinstance(sequence_request, ProjectedDNFRead)
    sequence = _read_projected(sequence_request, reader)
    if sequence.empty:
        raise ValueError("V58 sequence read returned no authorized rows")
    sequence_keys = _frame_keys(sequence, "sequence")
    loaded_symbols = set(sequence["symbol"])
    if not loaded_symbols.issubset(cell.train_symbols):
        raise ValueError("Sequence read materialized a non-train symbol")
    if loaded_symbols.intersection(cell.heldout_symbols):
        raise ValueError("Sequence read materialized a held-out fold symbol")
    if loaded_symbols.intersection(TARGET_SYMBOLS):
        raise ValueError("Sequence read materialized a sealed target symbol")
    train_mask = _strict_bool(sequence[cell.train_flag], cell.train_flag)
    validation_mask = _strict_bool(
        sequence[cell.validation_flag], cell.validation_flag
    )
    if (train_mask & validation_mask).any() or (~train_mask & ~validation_mask).any():
        raise ValueError("Each projected sequence row must have exactly one job role")
    train_dates = sequence.loc[train_mask, "date"]
    validation_dates = sequence.loc[validation_mask, "date"]
    if train_dates.empty or validation_dates.empty:
        raise ValueError("Both V58 train and validation roles must be non-empty")
    if not train_dates.between(
        cell.train_signal_start, cell.train_signal_end, inclusive="both"
    ).all():
        raise ValueError("Train-role sequence row is outside its registered range")
    if not validation_dates.between(
        cell.validation_signal_start,
        cell.validation_signal_end,
        inclusive="both",
    ).all():
        raise ValueError("Validation-role sequence row is outside its registered range")
    calendar_lengths = (
        sequence["date"] - sequence["sequence_start_date"]
    ).dt.days + 1
    if not (calendar_lengths == LOOKBACK_DAYS).all():
        raise ValueError("Registered sequence_start does not define 256 calendar days")
    train_signal_keys = frozenset(
        zip(
            sequence.loc[train_mask, "date"],
            sequence.loc[train_mask, "symbol"],
            strict=True,
        )
    )
    validation_signal_keys = frozenset(
        zip(
            sequence.loc[validation_mask, "date"],
            sequence.loc[validation_mask, "symbol"],
            strict=True,
        )
    )
    if train_signal_keys.intersection(validation_signal_keys):
        raise ValueError("Train and validation signal keys overlap")
    signal_keys = train_signal_keys | validation_signal_keys
    if signal_keys != sequence_keys:
        raise RuntimeError("Sequence logical-key accounting drift")
    sequence_start_by_key = {
        (row.date, row.symbol): row.sequence_start_date
        for row in sequence.itertuples(index=False)
    }
    context_keys: set[LogicalKey] = set()
    for key, start in sequence_start_by_key.items():
        signal_date, symbol = key
        calendar = pd.date_range(start, signal_date, freq="D", tz="UTC")
        if len(calendar) != LOOKBACK_DAYS or calendar[0] != start:
            raise ValueError("Sequence context is not the exact registered calendar")
        context_keys.update((date, symbol) for date in calendar)
    frozen_context_keys = frozenset(context_keys)
    plan = build_job_read_plan(
        cell,
        sequence_path=sequence_path,
        labels_path=labels_path,
        panel_path=panel_path,
        signal_keys=signal_keys,
        context_keys=frozen_context_keys,
    )
    assert isinstance(plan, JobReadPlan)
    labels = _read_projected(plan.labels, reader)
    label_keys = _frame_keys(labels, "labels")
    if label_keys != signal_keys:
        raise ValueError("Label read did not materialize the exact signal-key union")
    completeness = _strict_bool(
        labels["multi_horizon_label_complete"], "multi_horizon_label_complete"
    )
    label_values = labels.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float64)
    if any(labels[column].dtype != np.dtype("float64") for column in TARGET_COLUMNS):
        raise ValueError("V58 label source columns must be physical float64 values")
    if not completeness.all() or not np.isfinite(label_values).all():
        raise ValueError("An authorized role key lacks complete finite h1/h3/h7 labels")
    expected_maturity = labels["date"] + pd.Timedelta(days=8)
    if not labels["target_h7_maturity_date"].equals(expected_maturity):
        raise ValueError("H7 maturity is not exactly signal date plus eight days")
    panel = _read_projected(plan.panel, reader)
    panel_keys = _frame_keys(panel, "panel")
    if panel_keys != frozen_context_keys:
        raise ValueError("Panel read did not materialize the exact 256-day context union")
    if not set(panel["symbol"]).issubset(cell.train_symbols):
        raise ValueError("Panel read materialized a non-train symbol")
    panel_values = panel.loc[:, BASE_FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(panel_values).all():
        raise ValueError("Registered feature context contains a non-finite base value")
    labels_by_key = {
        (row.date, row.symbol): (
            float(row.target_h1_open_to_open_log_return),
            float(row.target_h3_open_to_open_log_return),
            float(row.target_h7_open_to_open_log_return),
        )
        for row in labels.itertuples(index=False)
    }
    scaler_mask = panel["date"].between(
        cell.train_signal_start, cell.train_signal_end, inclusive="both"
    ) & panel["symbol"].isin(cell.train_symbols)
    scaler_source_rows = (
        panel.loc[scaler_mask, PANEL_COLUMNS]
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )
    scaler_source_keys = _frame_keys(scaler_source_rows, "scaler source")
    if not scaler_source_keys:
        raise RuntimeError("Scaler source has no finite train-range feature cells")
    train_availability = _availability(sequence, train_mask)
    validation_availability = _availability(sequence, validation_mask)
    receipt = {
        "job_cell": cell.key,
        "projected_columns_per_input": {
            name: list(getattr(plan, name).columns)
            for name in ("sequence", "labels", "panel")
        },
        "logical_filters_per_input": {
            name: _jsonable(getattr(plan, name).filters)
            for name in ("sequence", "labels", "panel")
        },
        "role_bounds": {
            "train": _role_receipt(
                train_signal_keys,
                registered_start=cell.train_signal_start,
                registered_end=cell.train_signal_end,
            ),
            "validation": _role_receipt(
                validation_signal_keys,
                registered_start=cell.validation_signal_start,
                registered_end=cell.validation_signal_end,
            ),
        },
        "train_validation_signal_key_counts": {
            "train": len(train_signal_keys),
            "validation": len(validation_signal_keys),
        },
        "label_logical_key_count": len(label_keys),
        "feature_context_key_count": len(panel_keys),
        "scaler_source_unique_symbol_date_count": len(scaler_source_keys),
        "scaler_source_date_min": _day_text(scaler_source_rows["date"].min()),
        "scaler_source_date_max": _day_text(scaler_source_rows["date"].max()),
        "scaler_source_symbols": sorted(scaler_source_rows["symbol"].unique()),
        "authorized_sequence_rows": len(sequence),
        "authorized_label_rows": len(labels),
        "authorized_panel_rows": len(panel),
        "forbidden_column_count_zero": 0,
        "forbidden_columns_loaded": [],
        "job_relative_development_evaluation_value_count_zero": 0,
        "job_relative_development_evaluation_values_read": 0,
        "development_evaluation_outcome_rows_read": 0,
        "target_asset_load_count_zero": 0,
        "target_asset_load_count": 0,
        "target_assets_loaded": [],
        "heldout_fold_symbols_loaded": [],
        "heldout_fold_symbols_loaded_by_job": [],
    }
    tensor_store = ExplicitCalendarTensorStore(
        cell=cell,
        panel=panel,
        sequence_start_by_key=sequence_start_by_key,
        labels_by_key=labels_by_key,
        train_availability=train_availability,
        validation_availability=validation_availability,
    )
    return JobTrainingData(
        cell=cell,
        sequence=sequence,
        labels=labels,
        panel=panel,
        train_signal_keys=train_signal_keys,
        validation_signal_keys=validation_signal_keys,
        context_keys=frozen_context_keys,
        train_availability=train_availability,
        validation_availability=validation_availability,
        sequence_start_by_key=sequence_start_by_key,
        labels_by_key=labels_by_key,
        scaler_source_rows=scaler_source_rows,
        read_plan=plan,
        access_receipt=receipt,
        tensor_store=tensor_store,
    )


@dataclass(frozen=True)
class TrainOnlyScaler:
    origin: str
    geometry: str
    fold: int
    feature_names: tuple[str, ...]
    fit_symbols: tuple[str, ...]
    fit_symbol_count: int
    fit_unique_symbol_date_count: int
    fit_min_date: str
    fit_max_date: str
    mean: tuple[float, ...]
    standard_deviation: tuple[float, ...]
    zero_scale_replacements: int
    scaler_sha256: str

    def transform(self, raw_base_and_relative: np.ndarray) -> np.ndarray:
        values = np.asarray(raw_base_and_relative, dtype=np.float64)
        if values.shape[-1] != len(self.feature_names) + 1:
            raise ValueError("V58 tensor must contain eight base plus one relative feature")
        if not np.isfinite(values).all():
            raise ValueError("V58 tensor contains a non-finite active value")
        mean = np.asarray(self.mean, dtype=np.float64)
        scale = np.asarray(self.standard_deviation, dtype=np.float64)
        transformed = np.empty(values.shape, dtype=np.float64)
        transformed[..., :-1] = (values[..., :-1] - mean) / scale
        source_index = self.feature_names.index("log_close_to_close_return")
        transformed[..., -1] = values[..., -1] / scale[source_index]
        if not np.isfinite(transformed).all():
            raise ValueError("V58 scaler produced a non-finite value")
        return transformed.astype(np.float32)

    def receipt(self) -> dict[str, object]:
        return {
            "origin": self.origin,
            "geometry": self.geometry,
            "fold": self.fold,
            "fit_symbol_count": self.fit_symbol_count,
            "fit_unique_symbol_date_count": self.fit_unique_symbol_date_count,
            "fit_min_date": self.fit_min_date,
            "fit_max_date": self.fit_max_date,
            "mean": list(self.mean),
            "standard_deviation": list(self.standard_deviation),
            "zero_scale_replacements": self.zero_scale_replacements,
            "scaler_sha256": self.scaler_sha256,
        }


def fit_train_only_scaler(
    panel: pd.DataFrame,
    *,
    train_symbols: Sequence[str],
    train_start: object,
    train_end: object,
    origin: str,
    geometry: str,
    fold: int,
    feature_names: Sequence[str] = BASE_FEATURE_COLUMNS,
) -> TrainOnlyScaler:
    """Fit float64 ddof=0 statistics on unique finite train signal cells."""

    feature_names = tuple(feature_names)
    if feature_names != BASE_FEATURE_COLUMNS:
        raise ValueError("V58 base-feature order drift")
    train_symbols = tuple(str(symbol) for symbol in train_symbols)
    if train_symbols != tuple(sorted(set(train_symbols))):
        raise ValueError("V58 scaler symbols must be unique lexical ascending")
    if TARGET_SYMBOLS.intersection(train_symbols):
        raise ValueError("A sealed target symbol entered the V58 scaler")
    required = {"date", "symbol", *feature_names}
    missing = sorted(required - set(panel.columns))
    if missing:
        raise ValueError(f"V58 scaler panel is missing columns: {missing}")
    working = panel.loc[:, ["date", "symbol", *feature_names]].copy()
    working["date"] = working["date"].map(_utc_day)
    working["symbol"] = working["symbol"].astype(str)
    if not set(working["symbol"]).issubset(train_symbols):
        raise ValueError("A non-train symbol was materialized for the V58 scaler")
    if working.duplicated(["date", "symbol"]).any():
        raise ValueError("Duplicate context weighting is forbidden for the V58 scaler")
    train_start = _utc_day(train_start)
    train_end = _utc_day(train_end)
    if train_start > train_end:
        raise ValueError("V58 scaler train range is reversed")
    _frame_keys(working, "scaler panel")
    in_population = working["date"].between(
        train_start, train_end, inclusive="both"
    ) & working["symbol"].isin(train_symbols)
    working = working.loc[in_population].reset_index(drop=True)
    if working.empty:
        raise ValueError("V58 scaler has no cells in its exact train range")
    raw = working.loc[:, feature_names].to_numpy(dtype=np.float64)
    finite = np.isfinite(raw).all(axis=1)
    finite_rows = working.loc[finite].reset_index(drop=True)
    values = raw[finite]
    if not len(values):
        raise ValueError("V58 scaler has no finite complete train signal cells")
    fitted_symbols = tuple(sorted(finite_rows["symbol"].unique()))
    if fitted_symbols != train_symbols:
        raise ValueError("V58 scaler did not fit every exact fold train symbol")
    mean = values.mean(axis=0, dtype=np.float64)
    raw_scale = values.std(axis=0, ddof=0, dtype=np.float64)
    if not np.isfinite(mean).all() or not np.isfinite(raw_scale).all():
        raise ValueError("V58 scaler statistic is non-finite")
    zero_mask = raw_scale == 0.0
    scale = raw_scale.copy()
    scale[zero_mask] = 1.0
    payload = {
        "origin": str(origin),
        "geometry": str(geometry),
        "fold": int(fold),
        "feature_names": list(feature_names),
        "fit_symbols": list(fitted_symbols),
        "fit_symbol_count": len(fitted_symbols),
        "fit_unique_symbol_date_count": len(finite_rows),
        "fit_min_date": _day_text(finite_rows["date"].min()),
        "fit_max_date": _day_text(finite_rows["date"].max()),
        "mean": [float(value) for value in mean],
        "standard_deviation": [float(value) for value in scale],
        "zero_scale_replacements": int(zero_mask.sum()),
    }
    return TrainOnlyScaler(
        origin=payload["origin"],
        geometry=payload["geometry"],
        fold=payload["fold"],
        feature_names=feature_names,
        fit_symbols=fitted_symbols,
        fit_symbol_count=payload["fit_symbol_count"],
        fit_unique_symbol_date_count=payload["fit_unique_symbol_date_count"],
        fit_min_date=payload["fit_min_date"],
        fit_max_date=payload["fit_max_date"],
        mean=tuple(payload["mean"]),
        standard_deviation=tuple(payload["standard_deviation"]),
        zero_scale_replacements=payload["zero_scale_replacements"],
        scaler_sha256=_canonical_sha256(payload),
    )


def fit_job_train_only_scaler(data: JobTrainingData) -> TrainOnlyScaler:
    return fit_train_only_scaler(
        data.panel,
        train_symbols=data.cell.train_symbols,
        train_start=data.cell.train_signal_start,
        train_end=data.cell.train_signal_end,
        origin=data.cell.origin,
        geometry=data.cell.geometry,
        fold=data.cell.fold,
    )


@dataclass(frozen=True)
class SampleDraw:
    date: pd.Timestamp
    triplet: tuple[str, str, str]
    pair_index: int

    def canonical_record(self) -> dict[str, object]:
        return {
            "date": _day_text(self.date),
            "triplet": list(self.triplet),
            "pair_index": self.pair_index,
        }


@dataclass(frozen=True)
class SamplingBatch:
    role: Literal["train", "validation"]
    epoch: int
    seed_components: tuple[object, ...]
    generator_seed: int
    draws: tuple[SampleDraw, ...]
    ordered_draw_list_sha256: str


def sampling_seed_components(
    *,
    origin: str,
    geometry: str,
    fold: int,
    job_seed: int,
    role: Literal["train", "validation"],
    epoch: int,
) -> tuple[object, ...]:
    if origin not in {"origin_2024", "origin_2025"}:
        raise ValueError("Unregistered V58 sampling origin")
    if geometry not in {"expanding", "rolling"}:
        raise ValueError("Unregistered V58 sampling geometry")
    if int(fold) not in {1, 2, 3}:
        raise ValueError("Unregistered V58 sampling fold")
    if int(job_seed) not in REGISTERED_JOB_SEEDS:
        raise ValueError("Unregistered V58 job seed")
    if int(epoch) < 0:
        raise ValueError("Sampling epoch cannot be negative")
    if role == "train":
        return (
            MASTER_SEED,
            "v58",
            origin,
            geometry,
            int(fold),
            int(job_seed),
            "train",
            int(epoch),
        )
    if role == "validation":
        if int(epoch) != 0:
            raise ValueError("V58 validation sampling is frozen at epoch zero")
        return (
            MASTER_SEED,
            "v58",
            origin,
            geometry,
            int(fold),
            MASTER_SEED,
            "validation",
            0,
        )
    raise ValueError(f"Unsupported V58 sampling role: {role}")


def derive_sampling_seed(**kwargs: Any) -> int:
    components = sampling_seed_components(**kwargs)
    return int.from_bytes(hashlib.sha256(_canonical_json(components)).digest()[:8], "big")


class UniformDateTripletSampler:
    """Uniform-with-replacement sampler over the canonical flattened pair list."""

    def __init__(
        self,
        availability_by_date: Mapping[object, Sequence[str]],
        train_triplets: Sequence[Sequence[str]],
    ) -> None:
        triplets = tuple(tuple(map(str, triplet)) for triplet in train_triplets)
        if not triplets or any(len(triplet) != 3 for triplet in triplets):
            raise ValueError("V58 sampler requires non-empty lexical triplets of size three")
        if any(
            triplet != tuple(sorted(set(triplet))) for triplet in triplets
        ):
            raise ValueError("Every V58 triplet must contain three lexical unique symbols")
        if triplets != tuple(sorted(set(triplets))):
            raise ValueError("V58 sampler triplets must be unique lexical ascending")
        catalog_symbols = frozenset(symbol for triplet in triplets for symbol in triplet)
        if TARGET_SYMBOLS.intersection(catalog_symbols):
            raise ValueError("A sealed target symbol entered the V58 sampler catalog")
        self.train_triplets = triplets
        cached: dict[tuple[str, ...], tuple[tuple[str, str, str], ...]] = {}
        entries = []
        for date_value, symbols_value in sorted(
            availability_by_date.items(), key=lambda item: _utc_day(item[0])
        ):
            date = _utc_day(date_value)
            symbols = tuple(sorted(set(map(str, symbols_value))))
            if not set(symbols).issubset(catalog_symbols):
                raise ValueError("V58 availability contains a symbol outside the catalog")
            if symbols not in cached:
                allowed = set(symbols)
                cached[symbols] = tuple(
                    triplet for triplet in triplets if set(triplet).issubset(allowed)
                )
            eligible = cached[symbols]
            if eligible:
                entries.append((date, eligible))
        if not entries:
            raise ValueError("No eligible V58 date-triplet pairs")
        self._entries = tuple(entries)
        self._cumulative = np.cumsum(
            [len(triplets_for_date) for _, triplets_for_date in self._entries],
            dtype=np.int64,
        )
        self.total_pairs = int(self._cumulative[-1])

    def sample(
        self,
        sample_count: int,
        *,
        origin: str,
        geometry: str,
        fold: int,
        job_seed: int,
        role: Literal["train", "validation"],
        epoch: int,
    ) -> SamplingBatch:
        if int(sample_count) < 1:
            raise ValueError("sample_count must be positive")
        seed_kwargs = {
            "origin": origin,
            "geometry": geometry,
            "fold": int(fold),
            "job_seed": int(job_seed),
            "role": role,
            "epoch": int(epoch),
        }
        components = sampling_seed_components(**seed_kwargs)
        generator_seed = derive_sampling_seed(**seed_kwargs)
        rng = np.random.default_rng(generator_seed)
        indexes = rng.integers(
            0, self.total_pairs, size=int(sample_count), dtype=np.int64
        )
        draws = []
        for value in indexes:
            pair_index = int(value)
            entry_index = int(
                np.searchsorted(self._cumulative, pair_index, side="right")
            )
            prior = int(self._cumulative[entry_index - 1]) if entry_index else 0
            date, triplets_for_date = self._entries[entry_index]
            draws.append(
                SampleDraw(
                    date=date,
                    triplet=triplets_for_date[pair_index - prior],
                    pair_index=pair_index,
                )
            )
        draw_tuple = tuple(draws)
        receipt = _canonical_sha256(
            [draw.canonical_record() for draw in draw_tuple]
        )
        return SamplingBatch(
            role=role,
            epoch=int(epoch),
            seed_components=components,
            generator_seed=generator_seed,
            draws=draw_tuple,
            ordered_draw_list_sha256=receipt,
        )


class ExplicitCalendarTensorStore:
    """Exact-key store; never derives a sequence by positional tail selection."""

    def __init__(
        self,
        *,
        cell: JobCell,
        panel: pd.DataFrame,
        sequence_start_by_key: Mapping[LogicalKey, pd.Timestamp],
        labels_by_key: Mapping[LogicalKey, tuple[float, float, float]],
        train_availability: Mapping[pd.Timestamp, tuple[str, ...]],
        validation_availability: Mapping[pd.Timestamp, tuple[str, ...]],
    ) -> None:
        self.cell = cell
        self.sequence_start_by_key = dict(sequence_start_by_key)
        self.labels_by_key = dict(labels_by_key)
        self.train_availability = dict(train_availability)
        self.validation_availability = dict(validation_availability)
        self.catalog = frozenset(cell.train_triplets)
        self.panel_by_key = {
            (row.date, row.symbol): np.asarray(
                [getattr(row, column) for column in BASE_FEATURE_COLUMNS],
                dtype=np.float64,
            )
            for row in panel.itertuples(index=False)
        }
        if len(self.panel_by_key) != len(panel):
            raise ValueError("Explicit calendar store received duplicate panel keys")

    def materialize_raw(
        self,
        sample: SampleDraw,
        *,
        role: Literal["train", "validation"],
    ) -> tuple[np.ndarray, np.ndarray]:
        date = _utc_day(sample.date)
        triplet = tuple(map(str, sample.triplet))
        if triplet != tuple(sorted(triplet)) or triplet not in self.catalog:
            raise ValueError("Sample triplet is outside the exact lexical V32 catalog")
        availability = (
            self.train_availability
            if role == "train"
            else self.validation_availability
        )
        if date not in availability or not set(triplet).issubset(availability[date]):
            raise ValueError(f"Sample is not eligible for the exact {role} role")
        starts = tuple(
            self.sequence_start_by_key.get((date, symbol)) for symbol in triplet
        )
        if None in starts or len(set(starts)) != 1:
            raise ValueError("Triplet lacks one shared registered sequence_start")
        start = starts[0]
        calendar = pd.date_range(start, date, freq="D", tz="UTC")
        if len(calendar) != LOOKBACK_DAYS or calendar[0] != start or calendar[-1] != date:
            raise ValueError("Triplet does not have the exact registered 256-day calendar")
        raw = np.empty(
            (LOOKBACK_DAYS, 3, len(BASE_FEATURE_COLUMNS)), dtype=np.float64
        )
        for time_index, context_date in enumerate(calendar):
            for asset_index, symbol in enumerate(triplet):
                key = (context_date, symbol)
                if key not in self.panel_by_key:
                    raise ValueError("Missing exact registered panel context key")
                raw[time_index, asset_index] = self.panel_by_key[key]
        if not np.isfinite(raw).all():
            raise ValueError("Materialized V58 context contains a non-finite value")
        active_targets = []
        for symbol in triplet:
            key = (date, symbol)
            if key not in self.labels_by_key:
                raise ValueError("Missing exact active label key")
            active_targets.append(self.labels_by_key[key])
        target = np.asarray(active_targets, dtype=np.float64)
        if target.shape != (3, 3) or not np.isfinite(target).all():
            raise ValueError("Active V58 target is not finite [asset,horizon]")
        return raw, target


@dataclass(frozen=True)
class MaterializedBatch:
    features: np.ndarray
    targets: np.ndarray


def materialize_triplet_batch(
    data: JobTrainingData,
    samples: Sequence[SampleDraw],
    scaler: TrainOnlyScaler,
    *,
    role: Literal["train", "validation"],
) -> MaterializedBatch:
    """Materialize exact registered 256-calendar sequences and h1/h3/h7 labels."""

    if not samples:
        raise ValueError("Cannot materialize an empty V58 batch")
    if role not in {"train", "validation"}:
        raise ValueError(f"Unsupported V58 materialization role: {role}")
    if (scaler.origin, scaler.geometry, scaler.fold) != (
        data.cell.origin,
        data.cell.geometry,
        data.cell.fold,
    ):
        raise ValueError("V58 scaler/job-cell identity mismatch")
    features = []
    targets = []
    for sample in samples:
        raw, target = data.tensor_store.materialize_raw(sample, role=role)
        source_index = BASE_FEATURE_COLUMNS.index("log_close_to_close_return")
        source = raw[..., source_index]
        relative = source - source.mean(axis=1, keepdims=True)
        raw_with_relative = np.concatenate([raw, relative[..., None]], axis=2)
        features.append(scaler.transform(raw_with_relative))
        targets.append(target.astype(np.float32))
    feature_batch = np.stack(features)
    target_batch = np.stack(targets)
    if feature_batch.shape != (len(samples), LOOKBACK_DAYS, 3, 9):
        raise RuntimeError("V58 materialized feature shape drift")
    if target_batch.shape != (len(samples), 3, 3):
        raise RuntimeError("V58 materialized target shape drift")
    return MaterializedBatch(feature_batch, target_batch)
