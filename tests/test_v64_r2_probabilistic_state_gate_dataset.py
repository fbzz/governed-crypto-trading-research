from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tlm.core import DatasetAccessLedger, canonical_sha256, file_sha256
from tlm.v64_r2_probabilistic_state_gate_dataset import (
    FORBIDDEN_COLUMNS,
    LABEL_PROJECTION,
    ROLE_COLUMNS,
    SEQUENCE_PROJECTION,
    _metadata_context,
    _read_authorized_parquets,
    build_gate_roles,
)


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    value = yaml.safe_load(
        (
            ROOT
            / "configs/v67_v64_r2_probabilistic_state_gate_dataset.yaml"
        ).read_text(encoding="utf-8")
    )
    assert isinstance(value, dict)
    return value


def _boundary_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(
        ["2024-06-28", "2024-06-29", "2024-07-01", "2024-12-23"],
        utc=True,
    )
    labels = pd.DataFrame(
        {
            "date": dates,
            "symbol": ["AUSDT"] * len(dates),
            "target_h1_maturity_date": dates + pd.Timedelta(days=2),
            "target_h1_open_to_open_log_return": [0.01, 0.02, -0.01, 0.03],
            "h1_label_complete": [True] * len(dates),
        }
    )
    sequence = pd.DataFrame(
        {
            "date": dates,
            "sequence_start_date": dates - pd.Timedelta(days=255),
            "symbol": ["AUSDT"] * len(dates),
            "h1_label_complete": [True] * len(dates),
            "eligible_train": [True] * len(dates),
        }
    )
    return labels, sequence


def test_v67_split_embargo_prevents_train_label_crossing_validation() -> None:
    labels, sequence = _boundary_frames()
    roles, audit = build_gate_roles(
        labels,
        sequence,
        {
            "gate_train": {
                "signal_start": "2021-03-01",
                "signal_end": "2024-06-30",
            },
            "gate_internal_validation": {
                "signal_start": "2024-07-01",
                "signal_end": "2024-12-23",
            },
        },
    )
    assert list(roles.columns) == ROLE_COLUMNS
    assert roles["date"].dt.date.astype(str).tolist() == [
        "2024-06-28",
        "2024-07-01",
        "2024-12-23",
    ]
    assert roles["gate_role"].tolist() == [
        "gate_train",
        "gate_internal_validation",
        "gate_internal_validation",
    ]
    assert audit["excluded_split_boundary_rows"] == 1
    assert audit["split_embargo_days"] == 2
    assert (
        audit["roles"]["gate_train"]["maximum_target_maturity"]
        < audit["roles"]["gate_internal_validation"]["first_eligible_date"]
    )


def test_v67_predicate_and_projection_are_passed_to_scanner_before_rows() -> None:
    labels, sequence = _boundary_frames()
    calls: list[tuple[Path, list[str], str]] = []

    def scanner(path: Path, columns: list[str], predicate: object) -> pd.DataFrame:
        calls.append((path, list(columns), str(predicate)))
        if path.name == "labels.parquet":
            return labels[LABEL_PROJECTION].copy()
        return sequence[SEQUENCE_PROJECTION].copy()

    context = {
        "contract": {
            "parquet_access_contract": {
                "maximum_parquet_deserializations": 2,
                "full_table_materialization_then_filtering_allowed": False,
                "predicate_pushdown_required": True,
                "labels_projection": LABEL_PROJECTION,
                "sequence_projection": SEQUENCE_PROJECTION,
                "forbidden_columns": sorted(FORBIDDEN_COLUMNS),
            }
        },
        "input_paths": {
            "labels": Path("labels.parquet"),
            "sequence_roles": Path("roles.parquet"),
        },
        "symbols": ["AUSDT"],
    }
    ledger = DatasetAccessLedger()
    read_labels, read_sequence, access = _read_authorized_parquets(
        context, ledger, scanner=scanner
    )
    assert len(calls) == 2
    assert calls[0][1] == LABEL_PROJECTION
    assert calls[1][1] == SEQUENCE_PROJECTION
    assert "2024-12-23" in calls[0][2]
    assert "eligible_train" in calls[1][2]
    assert not (set(read_labels.columns) | set(read_sequence.columns)) & FORBIDDEN_COLUMNS
    assert access["predicate_pushdown_applied_before_materialization"] is True
    assert access["full_table_materialization_then_filtering"] is False
    assert ledger.authorized_parquet_deserializations == 2


def test_v67_missing_labels_are_retained_without_imputation() -> None:
    labels, sequence = _boundary_frames()
    labels.loc[0, "target_h1_open_to_open_log_return"] = np.nan
    labels.loc[0, "h1_label_complete"] = False
    sequence.loc[0, "h1_label_complete"] = False
    roles, audit = build_gate_roles(
        labels,
        sequence,
        {
            "gate_train": {
                "signal_start": "2021-03-01",
                "signal_end": "2024-06-30",
            },
            "gate_internal_validation": {
                "signal_start": "2024-07-01",
                "signal_end": "2024-12-23",
            },
        },
    )
    assert len(labels) == 4
    assert labels["target_h1_open_to_open_log_return"].isna().sum() == 1
    assert "2024-06-28" not in roles["date"].dt.date.astype(str).tolist()
    assert audit["admitted_rows"] == 2


def test_v67_rejects_input_allowlist_drift_before_any_parquet_scan() -> None:
    config = deepcopy(_config())
    config["v64_r2_probabilistic_state_gate_dataset"]["inputs"]["raw_panel"] = (
        "data/processed/selected_universe_panel_v32.parquet"
    )
    with pytest.raises(RuntimeError, match="input-name allowlist drift"):
        _metadata_context(config, DatasetAccessLedger())


def test_v67_registered_packet_is_self_consistent_when_present() -> None:
    output = ROOT / "artifacts/v67_v64_r2_probabilistic_state_gate_dataset"
    if not (output / "result.json").is_file():
        pytest.skip("V67 official packet has not been executed yet")
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    audit = json.loads((output / "audit.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (output / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    completion = json.loads(
        (output / "completion_receipt.json").read_text(encoding="utf-8")
    )
    assert audit["passed"] is True
    assert result["decision"] == (
        "authorize_v68_frozen_non_target_v64_r2_gate_training_only"
    )
    result_copy = dict(result)
    assert result_copy.pop("result_sha256") == canonical_sha256(result_copy)
    manifest_copy = dict(manifest)
    assert manifest_copy.pop("artifact_manifest_sha256") == canonical_sha256(
        manifest_copy
    )
    completion_copy = dict(completion)
    assert completion_copy.pop("completion_receipt_sha256") == canonical_sha256(
        completion_copy
    )
    for path, expected in manifest["data_files"].items():
        assert file_sha256(ROOT / path) == expected
    ledger = result["data_access"]["operation_ledger"]
    assert ledger["authorized_parquet_deserializations"] == 2
    assert ledger["scaler_fits"] == 0
    assert ledger["model_instantiations"] == 0
    assert ledger["optimizer_steps"] == 0
    assert ledger["checkpoint_reads"] == 0
    assert ledger["target_asset_loads"] == 0
