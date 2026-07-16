import hashlib
import io
import zipfile

import numpy as np
import pandas as pd

from tlm.non_target_dataset import (
    LABEL_COLUMNS,
    PANEL_FEATURES,
    build_asset_folds,
    build_feature_schema,
    build_symbol_panel,
    read_cached_archive,
)


def _frame(days=50):
    index = pd.date_range("2021-01-01", periods=days, freq="D", tz="UTC")
    opened = np.exp(np.arange(days) * 0.01)
    closed = opened * np.exp(0.002)
    return pd.DataFrame({
        "open": opened,
        "high": closed * 1.01,
        "low": opened * 0.99,
        "close": closed,
        "volume": np.arange(days, dtype=float) + 100.0,
        "quote_volume": np.arange(days, dtype=float) + 1000.0,
        "trade_count": np.arange(days, dtype=float) + 10.0,
    }, index=index)


def _splits():
    return {
        "representation_train": ["2021-01-01", "2021-01-20"],
        "supervised_train": ["2021-01-05", "2021-01-20"],
        "validation": ["2021-01-21", "2021-01-30"],
        "calibration": ["2021-01-31", "2021-02-09"],
        "one_shot_non_target_confirmation": ["2021-02-10", "2021-02-19"],
    }


def _dataset_config():
    return {
        "realized_volatility_formula": "root_sum_squared_log_returns",
        "realized_volatility_windows": [7, 30],
        "triplet_relative_strength_source": "log_close_to_close_return",
        "triplet_relative_strength_formula": (
            "asset_value_minus_equal_weight_triplet_mean_same_date"
        ),
        "missing_data_policy": "preserve_nan_no_imputation",
    }


def test_panel_has_exact_causal_features_and_forward_labels():
    frame = _frame()
    panel = build_symbol_panel("AAAUSDT", frame, frame.index, _splits(), 4)
    row = panel.iloc[10]
    assert np.isclose(row["log_open_to_open_return"], 0.01)
    assert np.isclose(row["log_close_to_close_return"], 0.01)
    assert np.isclose(row["target_next_open_to_next_open_log_return"], 0.01)
    assert np.isclose(row["target_realized_volatility_7d"], np.sqrt(7) * 0.01)
    assert row["eligible_action_date"] == frame.index[11]
    assert row["target_window_end_date"] == frame.index[18]
    assert panel.iloc[-9]["label_complete"]
    assert not panel.iloc[-8]["label_complete"]


def test_future_raw_change_cannot_modify_past_panel_features():
    frame = _frame()
    changed = frame.copy()
    cutoff = frame.index[25]
    changed.loc[changed.index > cutoff, ["open", "high", "low", "close"]] *= 3
    original_panel = build_symbol_panel("AAAUSDT", frame, frame.index, _splits(), 4)
    changed_panel = build_symbol_panel("AAAUSDT", changed, frame.index, _splits(), 4)
    columns = list(PANEL_FEATURES)
    pd.testing.assert_frame_equal(
        original_panel.loc[original_panel["date"] <= cutoff, columns],
        changed_panel.loc[changed_panel["date"] <= cutoff, columns],
    )


def test_missing_day_is_preserved_and_breaks_returns_and_sequences():
    full = _frame()
    missing_date = full.index[10]
    frame = full.drop(missing_date)
    panel = build_symbol_panel("AAAUSDT", frame, full.index, _splits(), 4)
    missing_row = panel.loc[panel["date"] == missing_date].iloc[0]
    following = panel.loc[panel["date"] == missing_date + pd.Timedelta(days=1)].iloc[0]
    assert not missing_row["raw_observation_available"]
    assert np.isnan(missing_row["raw_open"])
    assert np.isnan(following["log_open_to_open_return"])
    assert not following["sequence_ready"]
    assert not panel.loc[
        panel["date"] == missing_date + pd.Timedelta(days=30), "sequence_ready"
    ].item()
    assert panel.loc[
        panel["date"] == missing_date + pd.Timedelta(days=34), "sequence_ready"
    ].item()


def test_schema_preserves_v26_feature_order_and_triplet_transform():
    registered = [*PANEL_FEATURES, "within_triplet_relative_strength"]
    blueprint = {"data_contract": {"derived_features": registered}}
    schema = build_feature_schema(blueprint, _dataset_config())
    assert schema["model_feature_order"] == registered
    assert len(schema["panel_features"]) == 8
    assert schema["triplet_derived_feature"]["source"] == (
        "log_close_to_close_return"
    )
    assert [row["name"] for row in schema["labels"]] == list(LABEL_COLUMNS)


def test_asset_folds_are_equal_disjoint_and_deterministic():
    symbols = [f"A{index:02d}USDT" for index in range(12)]
    first = build_asset_folds(symbols, 3)
    second = build_asset_folds(list(reversed(symbols)), 3)
    assert first == second
    tests = [set(fold["test_symbols"]) for fold in first["folds"]]
    assert all(len(group) == 4 for group in tests)
    assert not tests[0].intersection(tests[1])
    assert set.union(*tests) == set(symbols)


def test_cached_archive_must_match_manifest_and_checksum(tmp_path):
    symbol = "AAAUSDT"
    month = "2021-01"
    name = f"{symbol}-1d-{month}.zip"
    target = io.BytesIO()
    row = (
        "1609459200000,1,1.1,0.9,1.05,100,1609545599999,105,12,50,52.5,0\n"
    )
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(name.replace(".zip", ".csv"), row)
    payload = target.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    checksum_payload = f"{digest}  {name}\n".encode()
    directory = tmp_path / symbol / "1d"
    directory.mkdir(parents=True)
    archive_path = directory / name
    archive_path.write_bytes(payload)
    archive_path.with_suffix(".zip.CHECKSUM").write_bytes(checksum_payload)
    record = {
        "symbol": symbol,
        "month": month,
        "sha256": digest,
        "checksum_sha256": hashlib.sha256(checksum_payload).hexdigest(),
        "row_count": 1,
        "first_date": "2021-01-01",
        "last_date": "2021-01-01",
    }
    frame = read_cached_archive(record, tmp_path, "1d")
    assert len(frame) == 1
    assert frame.index[0] == pd.Timestamp("2021-01-01", tz="UTC")
