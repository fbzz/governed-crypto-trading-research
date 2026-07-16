from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tlm.ranking_excess_development_screen as screen_module
from tlm.ranking_excess_development_screen import (
    _json_ready,
    _unseal_or_load_outcomes,
    _validate_outcome_packet,
    _validate_result_packet,
    build_evaluation_spec,
    read_fold_prepare_data,
)


def _evaluation_config() -> dict:
    return {
        "seed": 17,
        "ranking_excess_screen": {
            "expected_input_sha256": {"metadata": "a" * 64},
            "expected_checkpoints": {"1:42": "b" * 64},
            "expected_scalers": {"1": {"state_sha256": "c" * 64}},
            "expected_target_scales": {
                "1": {"value": 0.02, "state_sha256": "d" * 64}
            },
            "device": "mps",
            "torch_threads": 1,
            "dtype": "float32",
            "deterministic_algorithms": True,
            "amp": False,
            "cpu_fallback_allowed": False,
            "inference_batch_size": 8,
            "ridge": {"alpha": 10.0, "train_samples_per_fold": 4},
            "data_access": {
                "screen_signal_start": "2025-01-01",
                "screen_signal_end": "2025-01-02",
            },
            "inference": {"seeds": [42, 7, 123]},
            "predictive_metrics": {"unit": "triplet_context"},
            "policy": {"switch_hurdle": 0.002},
            "accounting": {"reporting_cost_bps": [10, 20, 30]},
            "bootstrap": {"paths": 10, "block_lengths_days": [2]},
            "gates": {"all_cells_required": True},
            "lifecycle": {"repeat_evaluation_after_result": False},
            "artifact_contract": {"format": "synthetic_v1"},
        },
        "output_dir": "unused",
    }


def test_build_evaluation_spec_is_deterministic_and_contract_sensitive() -> None:
    config = _evaluation_config()
    blueprint = {
        "candidate_family_id": "synthetic_ranker",
        "blueprint_sha256": "e" * 64,
    }

    first = build_evaluation_spec(deepcopy(config), deepcopy(blueprint))
    second = build_evaluation_spec(deepcopy(config), deepcopy(blueprint))
    assert first == second
    assert first["evaluation_spec_sha256"] == second["evaluation_spec_sha256"]

    changed = deepcopy(config)
    changed["ranking_excess_screen"]["policy"]["switch_hurdle"] = 0.003
    changed_spec = build_evaluation_spec(changed, deepcopy(blueprint))
    assert changed_spec["evaluation_spec_sha256"] != first[
        "evaluation_spec_sha256"
    ]
    assert changed_spec["resolved_config_semantic_sha256"] != first[
        "resolved_config_semantic_sha256"
    ]


def test_build_evaluation_spec_changes_with_implementation_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _evaluation_config()
    blueprint = {
        "candidate_family_id": "synthetic_ranker",
        "blueprint_sha256": "e" * 64,
    }
    first_provenance = {
        "source_sha256": {
            "src/tlm/ranking_excess_development_screen.py": "1" * 64,
            "src/tlm/ranking_excess_screen_metrics.py": "2" * 64,
            "src/tlm/ranking_excess_harness.py": "3" * 64,
        },
        "runtime_versions": {"python": "synthetic-a"},
    }
    monkeypatch.setattr(
        screen_module,
        "_implementation_provenance",
        lambda: deepcopy(first_provenance),
    )
    first = build_evaluation_spec(deepcopy(config), deepcopy(blueprint))
    assert first["implementation_provenance"] == first_provenance

    changed_provenance = deepcopy(first_provenance)
    changed_provenance["source_sha256"][
        "src/tlm/ranking_excess_screen_metrics.py"
    ] = "4" * 64
    monkeypatch.setattr(
        screen_module,
        "_implementation_provenance",
        lambda: deepcopy(changed_provenance),
    )
    changed = build_evaluation_spec(deepcopy(config), deepcopy(blueprint))

    assert changed["implementation_provenance"] == changed_provenance
    assert changed["resolved_config_semantic_sha256"] == first[
        "resolved_config_semantic_sha256"
    ]
    assert changed["evaluation_spec_sha256"] != first[
        "evaluation_spec_sha256"
    ]


def _symbols(prefix: str, count: int) -> list[str]:
    return [f"{prefix}{index:02d}USDT" for index in range(count)]


def _feature_frame(
    symbols: list[str], dates: pd.DatetimeIndex, columns: list[str]
) -> pd.DataFrame:
    rows = []
    for symbol_index, symbol in enumerate(symbols):
        for day_index, current in enumerate(dates):
            rows.append(
                {
                    "date": current,
                    "symbol": symbol,
                    **{
                        name: 0.001 * (1 + symbol_index + day_index)
                        for name in columns[2:]
                    },
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _sequence_frame(symbols: list[str], signal: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": signal,
                "sequence_start_date": signal - pd.Timedelta(days=255),
                "symbol": symbol,
            }
            for symbol in symbols
        ],
        columns=["date", "sequence_start_date", "symbol"],
    )


def _prepare_fixture() -> tuple[dict, dict, dict[str, pd.DataFrame]]:
    train_symbols = _symbols("T", 20)
    test_symbols = _symbols("H", 10)
    ridge_dates = pd.date_range("2022-01-01", periods=256, freq="D", tz="UTC")
    screen_dates = pd.date_range("2024-04-21", periods=256, freq="D", tz="UTC")
    ridge_signal = ridge_dates[-1]
    screen_signal = screen_dates[-1]
    feature_columns = ["date", "symbol", "log_close_to_close_return"]
    label_columns = [
        "date",
        "symbol",
        "target_next_open_to_next_open_log_return",
    ]
    sequence_columns = ["date", "sequence_start_date", "symbol"]
    access = {
        "ridge_feature_columns": feature_columns,
        "ridge_label_columns": label_columns,
        "screen_feature_columns": feature_columns,
        "screen_label_columns": [
            "date",
            "symbol",
            "target_window_end_date",
            "secret_heldout_outcome",
        ],
        "sequence_columns": sequence_columns,
        "ridge_feature_start": ridge_dates[0].date().isoformat(),
        "ridge_feature_end": ridge_dates[-1].date().isoformat(),
        "ridge_signal_start": ridge_signal.date().isoformat(),
        "ridge_signal_end": ridge_signal.date().isoformat(),
        "screen_feature_start": screen_dates[0].date().isoformat(),
        "screen_feature_end": screen_dates[-1].date().isoformat(),
        "screen_signal_start": screen_signal.date().isoformat(),
        "screen_signal_end": screen_signal.date().isoformat(),
        "ridge_split_flag": "in_synthetic_train",
        "screen_split_flag": "in_synthetic_screen",
        "readiness_flags": ["supervised_sequence_ready", "label_complete"],
        "expected_by_fold": {
            "1": {
                "ridge_feature_rows": len(ridge_dates) * len(train_symbols),
                "ridge_label_rows": len(train_symbols),
                "ridge_sequence_rows": len(train_symbols),
                "ridge_eligible_pairs": math.comb(len(train_symbols), 3),
                "ridge_eligible_dates": 1,
                "ridge_first_ready_signal": ridge_signal.date().isoformat(),
                "ridge_last_ready_signal": ridge_signal.date().isoformat(),
                "screen_feature_rows": len(screen_dates) * len(test_symbols),
                "screen_signal_dates": 1,
                "screen_asset_date_rows": len(test_symbols),
                "screen_sequence_rows": len(test_symbols),
                "screen_triplet_contexts": math.comb(len(test_symbols), 3),
                "minimum_ready_assets": len(test_symbols),
                "maximum_ready_assets": len(test_symbols),
                "ready_segments": [
                    {
                        "start": screen_signal.date().isoformat(),
                        "end": screen_signal.date().isoformat(),
                        "ready_assets": len(test_symbols),
                        "absent": [],
                    }
                ],
            }
        },
    }
    fold = {
        "fold": 1,
        "train_symbols": train_symbols,
        "test_symbols": test_symbols,
    }
    frames = {
        "ridge_features": _feature_frame(train_symbols, ridge_dates, feature_columns),
        "ridge_labels": pd.DataFrame(
            [
                {
                    "date": ridge_signal,
                    "symbol": symbol,
                    "target_next_open_to_next_open_log_return": index / 10_000,
                }
                for index, symbol in enumerate(train_symbols)
            ],
            columns=label_columns,
        ),
        "ridge_sequence": _sequence_frame(train_symbols, ridge_signal),
        "screen_features": _feature_frame(test_symbols, screen_dates, feature_columns),
        "screen_sequence": _sequence_frame(test_symbols, screen_signal),
    }
    return fold, access, frames


class _RecordingReader:
    def __init__(self, access: dict, fold: dict, frames: dict[str, pd.DataFrame]):
        self.access = access
        self.fold = fold
        self.frames = frames
        self.calls: list[dict[str, object]] = []

    def __call__(self, path, *, engine, columns, filters):
        filter_columns = {name for name, _, _ in filters}
        requested_symbols = next(
            list(value) for name, operation, value in filters
            if name == "symbol" and operation == "in"
        )
        if str(path).endswith("sequence.parquet"):
            kind = (
                "screen_sequence"
                if self.access["screen_split_flag"] in filter_columns
                else "ridge_sequence"
            )
        elif list(columns) == self.access["ridge_label_columns"]:
            kind = "ridge_labels"
        elif requested_symbols == sorted(self.fold["test_symbols"]):
            kind = "screen_features"
        else:
            kind = "ridge_features"
        self.calls.append(
            {
                "kind": kind,
                "path": str(path),
                "engine": engine,
                "columns": list(columns),
                "filters": list(filters),
            }
        )
        return self.frames[kind].copy()


def _run_prepare_reader(
    fold: dict, access: dict, frames: dict[str, pd.DataFrame]
) -> tuple[object, _RecordingReader]:
    reader = _RecordingReader(access, fold, frames)
    result = read_fold_prepare_data(
        Path("synthetic-panel.parquet"),
        Path("synthetic-sequence.parquet"),
        fold,
        access,
        reader=reader,
    )
    return result, reader


def _filters_by_name(call: dict[str, object]) -> dict[str, tuple[str, object]]:
    return {
        str(name): (str(operation), value)
        for name, operation, value in call["filters"]
    }


def test_prepare_reader_uses_five_separate_filtered_reads_without_heldout_labels() -> None:
    fold, access, frames = _prepare_fixture()
    result, reader = _run_prepare_reader(fold, access, frames)

    assert [call["kind"] for call in reader.calls] == [
        "ridge_features",
        "ridge_labels",
        "ridge_sequence",
        "screen_features",
        "screen_sequence",
    ]
    assert [call["path"] for call in reader.calls] == [
        "synthetic-panel.parquet",
        "synthetic-panel.parquet",
        "synthetic-sequence.parquet",
        "synthetic-panel.parquet",
        "synthetic-sequence.parquet",
    ]
    assert all(call["engine"] == "pyarrow" for call in reader.calls)
    assert all(
        "secret_heldout_outcome" not in call["columns"] for call in reader.calls
    )
    label_calls = [
        call
        for call in reader.calls
        if "target_next_open_to_next_open_log_return" in call["columns"]
    ]
    assert len(label_calls) == 1
    assert _filters_by_name(label_calls[0])["symbol"] == (
        "in",
        sorted(fold["train_symbols"]),
    )

    ridge_signal_filters = _filters_by_name(reader.calls[1])
    assert ridge_signal_filters[access["ridge_split_flag"]] == ("==", True)
    assert ridge_signal_filters["supervised_sequence_ready"] == ("==", True)
    assert ridge_signal_filters["label_complete"] == ("==", True)
    assert ridge_signal_filters["date"] == (
        "<=",
        pd.Timestamp(access["ridge_signal_end"], tz="UTC"),
    )
    screen_signal_filters = _filters_by_name(reader.calls[4])
    assert screen_signal_filters[access["screen_split_flag"]] == ("==", True)
    assert screen_signal_filters["symbol"] == (
        "in",
        sorted(fold["test_symbols"]),
    )
    assert result.audit["heldout_label_columns_materialized"] == []
    assert result.audit["screen_triplet_contexts"] == math.comb(10, 3)
    assert len(result.receipts) == 5


def test_prepare_reader_rejects_ignored_symbol_filter() -> None:
    fold, access, frames = _prepare_fixture()
    extra = frames["screen_sequence"].iloc[[0]].copy()
    extra["symbol"] = fold["train_symbols"][0]
    frames["screen_sequence"] = pd.concat(
        [frames["screen_sequence"], extra], ignore_index=True
    )
    with pytest.raises(RuntimeError, match="ignored a symbol filter"):
        _run_prepare_reader(fold, access, frames)


def test_prepare_reader_rejects_ignored_date_filter() -> None:
    fold, access, frames = _prepare_fixture()
    frames["ridge_labels"].loc[0, "date"] += pd.Timedelta(days=1)
    with pytest.raises(RuntimeError, match="ignored a date filter"):
        _run_prepare_reader(fold, access, frames)


def test_prepare_reader_rejects_duplicate_keys() -> None:
    fold, access, frames = _prepare_fixture()
    frames["ridge_labels"] = pd.concat(
        [frames["ridge_labels"], frames["ridge_labels"].iloc[[0]]],
        ignore_index=True,
    )
    with pytest.raises(RuntimeError, match="contains duplicate keys"):
        _run_prepare_reader(fold, access, frames)


def test_prepare_reader_rejects_compressed_feature_calendar() -> None:
    fold, access, frames = _prepare_fixture()
    current = frames["ridge_features"]
    symbol = fold["train_symbols"][0]
    dates = current.loc[current["symbol"] == symbol, "date"].sort_values()
    original = dates.iloc[100]
    index = current.index[
        (current["symbol"] == symbol) & (current["date"] == original)
    ][0]
    current.loc[index, "date"] = original + pd.Timedelta(hours=12)
    with pytest.raises(RuntimeError, match="compressed calendar"):
        _run_prepare_reader(fold, access, frames)


def test_validate_result_packet_detects_artifact_tamper(tmp_path: Path) -> None:
    output = tmp_path / "packet"
    output.mkdir()
    core = output / "core.txt"
    core.write_text("sealed", encoding="utf-8")
    spec_sha = "f" * 64
    result = {
        "mode": "prepare",
        "decision": "synthetic",
        "evaluation_spec": {"evaluation_spec_sha256": spec_sha},
        "audit": {"passed": True},
    }
    screen_module._seal_result_packet(output, result, ["core.txt"])
    required = [
        "core.txt",
        "artifact_manifest.json",
        "result.json",
        "completion_receipt.json",
    ]
    assert _validate_result_packet(output, spec_sha, "prepare", required) == result

    core.write_text("tampered", encoding="utf-8")
    with pytest.raises(RuntimeError, match="cached artifact drift"):
        _validate_result_packet(output, spec_sha, "prepare", required)


@pytest.mark.parametrize("mutation", ["missing", "unexpected"])
def test_validate_result_packet_requires_exact_manifest_coverage(
    tmp_path: Path, mutation: str
) -> None:
    output = tmp_path / "packet"
    output.mkdir()
    (output / "core.txt").write_text("sealed", encoding="utf-8")
    (output / "audit.txt").write_text("also sealed", encoding="utf-8")
    spec_sha = "0" * 64
    result = {
        "mode": "prepare",
        "decision": "synthetic",
        "evaluation_spec": {"evaluation_spec_sha256": spec_sha},
        "audit": {"passed": True},
    }
    sealed_files = ["core.txt", "audit.txt"]
    required = [
        *sealed_files,
        "artifact_manifest.json",
        "result.json",
        "completion_receipt.json",
    ]
    screen_module._seal_result_packet(output, result, sealed_files)
    assert _validate_result_packet(output, spec_sha, "prepare", required) == result

    manifest_path = output / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "missing":
        manifest["files"] = [
            row for row in manifest["files"] if row["path"] != "audit.txt"
        ]
    else:
        extra = output / "unexpected.txt"
        extra.write_text("not registered", encoding="utf-8")
        manifest["files"].append(
            {
                "path": extra.name,
                "bytes": extra.stat().st_size,
                "sha256": screen_module._sha256_file(extra),
            }
        )
    manifest.pop("manifest_semantic_sha256")
    manifest["manifest_semantic_sha256"] = screen_module._canonical_sha256(
        manifest
    )
    screen_module._write_json_atomic(manifest_path, manifest)
    completion_path = output / "completion_receipt.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["artifact_manifest_sha256"] = screen_module._sha256_file(
        manifest_path
    )
    screen_module._write_json_atomic(completion_path, completion)

    with pytest.raises(RuntimeError, match="manifest file grid drift"):
        _validate_result_packet(output, spec_sha, "prepare", required)


OUTCOME_SCHEMA = [
    "date",
    "fold",
    "symbol",
    "target_window_end_date",
    "action_log_return",
]


def _outcome_frame(*, maturity_days: int = 8) -> pd.DataFrame:
    signal = pd.Timestamp("2025-01-01", tz="UTC")
    return pd.DataFrame(
        [
            {
                "date": signal,
                "fold": 1,
                "symbol": "AUSDT",
                "target_window_end_date": signal
                + pd.Timedelta(days=maturity_days),
                "action_log_return": 0.01,
            }
        ],
        columns=OUTCOME_SCHEMA,
    )


def _write_synthetic_outcome_packet(
    output: Path, spec_sha: str, prepare_sha: str
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    unseal_path = output / "unseal_receipt.json"
    outcome_path = output / "outcomes.parquet"
    expected_bindings: dict[str, object] = {
        "prepare_result_sha256": "a" * 64,
        "context_predictions_sha256": "b" * 64,
        "asset_predictions_sha256": "c" * 64,
        "positions_sha256": "d" * 64,
        "source_panel_sha256": "e" * 64,
        "evaluation_execution_count": 1,
    }
    unseal = {
        "version": "v45_unseal_receipt_v1",
        "started_at_utc": "2026-07-13T00:00:00+00:00",
        "evaluation_spec_sha256": spec_sha,
        "prepare_completion_receipt_sha256": prepare_sha,
        **expected_bindings,
    }
    screen_module._write_json_atomic(unseal_path, unseal)
    outcome_path.write_bytes(b"synthetic parquet bytes")
    receipt = {
        "version": "v45_outcome_packet_receipt_v1",
        "completed_at_utc": "2026-07-13T00:00:01+00:00",
        "evaluation_spec_sha256": spec_sha,
        "prepare_completion_receipt_sha256": prepare_sha,
        "unseal_receipt_sha256": screen_module._sha256_file(unseal_path),
        "outcomes_parquet_sha256": screen_module._sha256_file(outcome_path),
        "outcome_rows": 1,
        "outcome_schema": OUTCOME_SCHEMA,
        "source_reads": [],
        "source_panel_sha256": expected_bindings["source_panel_sha256"],
        "evaluation_execution_count": expected_bindings[
            "evaluation_execution_count"
        ],
    }
    screen_module._write_json_atomic(output / "outcome_receipt.json", receipt)
    return expected_bindings


def test_validate_outcome_packet_rejects_prepare_receipt_hash_drift(
    tmp_path: Path,
) -> None:
    spec_sha = "1" * 64
    prepare_sha = "2" * 64
    expected_bindings = _write_synthetic_outcome_packet(
        tmp_path, spec_sha, prepare_sha
    )
    receipt_path = tmp_path / "outcome_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["prepare_completion_receipt_sha256"] = "3" * 64
    screen_module._write_json_atomic(receipt_path, receipt)

    with pytest.raises(RuntimeError, match="cryptographic binding drift"):
        _validate_outcome_packet(
            tmp_path,
            spec_sha,
            prepare_sha,
            OUTCOME_SCHEMA,
            1,
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-09", tz="UTC"),
            [],
            expected_bindings,
        )


def test_validate_outcome_packet_rejects_target_maturity_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec_sha = "4" * 64
    prepare_sha = "5" * 64
    expected_bindings = _write_synthetic_outcome_packet(
        tmp_path, spec_sha, prepare_sha
    )
    monkeypatch.setattr(
        screen_module.pd,
        "read_parquet",
        lambda *_args, **_kwargs: _outcome_frame(maturity_days=7),
    )

    with pytest.raises(RuntimeError, match="outcome packet semantic drift"):
        _validate_outcome_packet(
            tmp_path,
            spec_sha,
            prepare_sha,
            OUTCOME_SCHEMA,
            1,
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-01", tz="UTC"),
            pd.Timestamp("2025-01-09", tz="UTC"),
            [],
            expected_bindings,
        )


@pytest.mark.parametrize(
    ("packet_file", "binding"),
    [
        ("unseal_receipt.json", "prepare_result_sha256"),
        ("unseal_receipt.json", "context_predictions_sha256"),
        ("unseal_receipt.json", "asset_predictions_sha256"),
        ("unseal_receipt.json", "positions_sha256"),
        ("unseal_receipt.json", "source_panel_sha256"),
        ("unseal_receipt.json", "evaluation_execution_count"),
        ("outcome_receipt.json", "source_panel_sha256"),
        ("outcome_receipt.json", "evaluation_execution_count"),
    ],
)
def test_validate_outcome_packet_rejects_each_frozen_binding_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    packet_file: str,
    binding: str,
) -> None:
    spec_sha = "6" * 64
    prepare_sha = "7" * 64
    expected_bindings = _write_synthetic_outcome_packet(
        tmp_path, spec_sha, prepare_sha
    )
    monkeypatch.setattr(
        screen_module.pd,
        "read_parquet",
        lambda *_args, **_kwargs: _outcome_frame(),
    )
    validation_args = (
        tmp_path,
        spec_sha,
        prepare_sha,
        OUTCOME_SCHEMA,
        1,
        pd.Timestamp("2025-01-01", tz="UTC"),
        pd.Timestamp("2025-01-01", tz="UTC"),
        pd.Timestamp("2025-01-09", tz="UTC"),
        [],
        expected_bindings,
    )
    frame, _ = _validate_outcome_packet(*validation_args)
    pd.testing.assert_frame_equal(frame, _outcome_frame())

    packet_path = tmp_path / packet_file
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet[binding] = (
        2 if binding == "evaluation_execution_count" else "8" * 64
    )
    screen_module._write_json_atomic(packet_path, packet)
    if packet_file == "unseal_receipt.json":
        receipt_path = tmp_path / "outcome_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["unseal_receipt_sha256"] = screen_module._sha256_file(
            packet_path
        )
        screen_module._write_json_atomic(receipt_path, receipt)

    with pytest.raises(RuntimeError, match="cryptographic binding drift"):
        _validate_outcome_packet(*validation_args)


def test_existing_unseal_without_outcome_packet_fails_closed(
    tmp_path: Path,
) -> None:
    output = tmp_path / "evaluate"
    output.mkdir()
    screen_module._write_json_atomic(output / "unseal_receipt.json", {})
    reader_called = False

    def forbidden_reader(*_args, **_kwargs):
        nonlocal reader_called
        reader_called = True
        raise AssertionError("reader must not run for an incomplete cached packet")

    config = {
        "ranking_excess_screen": {
            "artifact_contract": {
                "evaluate": {"outcome_schema": OUTCOME_SCHEMA}
            },
            "data_access": {
                "expected_total_asset_dates": 1,
                "screen_signal_start": "2025-01-01",
                "screen_signal_end": "2025-01-01",
                "screen_maturity_end": "2025-01-09",
                "screen_label_columns": [
                    "date",
                    "symbol",
                    "target_window_end_date",
                    "target_next_open_to_next_open_log_return",
                ],
                "screen_split_flag": "in_synthetic_screen",
                "readiness_flags": [
                    "supervised_sequence_ready",
                    "label_complete",
                ],
                "expected_by_fold": {
                    str(fold): {"screen_label_rows": 1}
                    for fold in (1, 2, 3)
                },
            },
        },
        "output_dir": "evaluate",
    }
    context = {
        "root": tmp_path,
        "evaluation_spec": {"evaluation_spec_sha256": "6" * 64},
        "folds": {
            fold: {
                "fold": fold,
                "train_symbols": [],
                "test_symbols": [f"F{fold}USDT"],
            }
            for fold in (1, 2, 3)
        },
    }
    prepared = {"completion_receipt_sha256": "7" * 64}
    with pytest.raises(RuntimeError, match="without a complete atomic outcome packet"):
        _unseal_or_load_outcomes(
            context,
            config,
            prepared,
            reader=forbidden_reader,
        )
    assert not reader_called


def test_json_ready_preserves_undefined_values_as_json_null() -> None:
    value = {
        "nan": float("nan"),
        "positive_infinity": np.float64(np.inf),
        "nested": [np.float32(-np.inf), {"defined": 1.0}],
        "timestamp": pd.Timestamp("2025-01-01", tz="UTC"),
    }
    ready = _json_ready(value)
    assert ready["nan"] is None
    assert ready["positive_infinity"] is None
    assert ready["nested"][0] is None
    assert ready["nested"][1]["defined"] == 1.0
    assert ready["timestamp"] == "2025-01-01T00:00:00+00:00"
    encoded = json.dumps(ready, allow_nan=False, sort_keys=True)
    assert '"nan": null' in encoded
    assert '"positive_infinity": null' in encoded
