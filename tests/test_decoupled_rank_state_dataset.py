from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tlm.core import canonical_sha256, file_sha256
from tlm.decoupled_rank_state_dataset import (
    LABEL_COLUMNS,
    SEQUENCE_ROLE_COLUMNS,
    _metadata_context,
    build_h1_labels,
    build_sequence_roles,
    derive_triplet_contract,
)
from tlm.core import DatasetAccessLedger


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    value = yaml.safe_load(
        (ROOT / "configs/v62_decoupled_rank_state_dataset.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(value, dict)
    return value


def _panel() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=8, freq="D", tz="UTC")
    rows = []
    for symbol_index, symbol in enumerate(("AUSDT", "BUSDT")):
        for index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "raw_open": 100.0 + symbol_index + index,
                    "raw_observation_available": True,
                }
            )
    return pd.DataFrame(rows)


def test_h1_labels_use_exact_open_endpoints_and_preserve_every_key() -> None:
    panel = _panel()
    labels = build_h1_labels(panel)
    first = labels.loc[
        (labels["date"] == pd.Timestamp("2024-01-01", tz="UTC"))
        & (labels["symbol"] == "AUSDT")
    ].iloc[0]
    assert np.isclose(
        first["target_h1_open_to_open_log_return"], np.log(102.0 / 101.0)
    )
    assert first["eligible_action_date"] == pd.Timestamp("2024-01-02", tz="UTC")
    assert first["target_h1_maturity_date"] == pd.Timestamp(
        "2024-01-03", tz="UTC"
    )
    assert len(labels) == len(panel)
    assert list(labels.columns) == LABEL_COLUMNS


def test_missing_endpoint_is_retained_and_marked_incomplete() -> None:
    panel = _panel()
    panel.loc[
        (panel["symbol"] == "AUSDT")
        & (panel["date"] == pd.Timestamp("2024-01-03", tz="UTC")),
        "raw_open",
    ] = np.nan
    labels = build_h1_labels(panel)
    first = labels.loc[
        (labels["symbol"] == "AUSDT")
        & (labels["date"] == pd.Timestamp("2024-01-01", tz="UTC"))
    ].iloc[0]
    assert np.isnan(first["target_h1_open_to_open_log_return"])
    assert not bool(first["h1_label_complete"])
    assert len(labels) == len(panel)


def test_sequence_roles_are_disjoint_and_maturity_bounded() -> None:
    panel = _panel()
    labels = build_h1_labels(panel)
    sequence = panel[["date", "symbol"]].copy()
    sequence["sequence_start_date"] = sequence["date"] - pd.Timedelta(days=255)
    roles, audit = build_sequence_roles(
        sequence,
        labels,
        {
            "train": {"signal_start": "2024-01-01", "signal_end": "2024-01-02"},
            "consumed_development_validation": {
                "signal_start": "2024-01-05",
                "signal_end": "2024-01-06",
            },
        },
    )
    assert not (roles["eligible_train"] & roles["eligible_consumed_development_validation"]).any()
    assert audit["eligible_train"]["maximum_target_maturity"] == "2024-01-04"
    assert audit["eligible_consumed_development_validation"][
        "maximum_target_maturity"
    ] == "2024-01-08"
    assert list(roles.columns) == SEQUENCE_ROLE_COLUMNS


def test_triplet_decomposition_and_state_features_are_exact() -> None:
    features = np.arange(6 * 3 * 9, dtype=np.float32).reshape(6, 3, 9) / 100.0
    returns = np.asarray([0.03, -0.01, 0.01], dtype=np.float64)
    _, state, market, excess = derive_triplet_contract(features, returns)
    assert state.shape == (6, 18)
    np.testing.assert_allclose(state[:, :9], features.mean(axis=1))
    np.testing.assert_allclose(state[:, 9:], features.std(axis=1, ddof=0))
    np.testing.assert_allclose(market + excess, returns, atol=1e-15)
    assert abs(excess.sum()) <= 1e-15
    _, permuted_state, _, _ = derive_triplet_contract(
        features[:, ::-1, :], returns[::-1]
    )
    np.testing.assert_allclose(state, permuted_state, atol=1e-7, rtol=1e-6)


def test_v62_rejects_input_hash_drift_before_parquet_read() -> None:
    config = deepcopy(_config())
    config["decoupled_rank_state_dataset"]["inputs"]["panel"] = (
        "data/processed/decoupled_rank_state_labels_v62.parquet"
    )
    with pytest.raises(RuntimeError, match="allowlist drift"):
        _metadata_context(config, DatasetAccessLedger())


def test_v62_registered_packet_is_self_consistent_and_read_only() -> None:
    output = ROOT / "artifacts/v62_non_target_decoupled_rank_state_dataset"
    experiment = yaml.safe_load(
        (ROOT / "research/experiments/v062.yaml").read_text(encoding="utf-8")
    )
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
        "authorize_v63_frozen_non_target_decoupled_rank_state_training_only"
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
    for reference in ("result", "audit", "artifact_manifest", "completion_receipt"):
        assert file_sha256(ROOT / experiment[reference]["path"]) == experiment[
            reference
        ]["file_sha256"]
    for path, expected in manifest["data_files"].items():
        assert file_sha256(ROOT / path) == expected
    data_access = json.loads((output / "data_access.json").read_text(encoding="utf-8"))
    ledger = data_access["operation_ledger"]
    assert ledger["authorized_parquet_deserializations"] == 2
    assert ledger["scaler_fits"] == 0
    assert ledger["model_instantiations"] == 0
    assert ledger["optimizer_steps"] == 0
    assert ledger["checkpoint_reads"] == 0
    assert ledger["target_asset_loads"] == 0
