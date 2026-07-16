from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from tlm.non_target_pretraining import _atomic_torch_save, _canonical_sha256
from tlm.patch_transformer import MultiAssetPatchTransformer, PREDICTION_HEADS
from tlm.scientific_harness import FeatureScaler
from tlm.supervised_non_target import (
    SupervisedTripletTensorStore,
    build_supervised_spec,
    calibration_semantic_sha256,
    calibration_state_sha256,
    compute_calibration_parameters,
    eligible_supervised_availability,
    load_supervised_checkpoint,
    model_state_sha256,
    supervised_parameter_names,
)


def _architecture() -> dict:
    return {
        "lookback_days": 8,
        "input_triplet_size": 3,
        "patch_length_days": 4,
        "patch_stride_days": 2,
        "d_model": 16,
        "attention_heads": 4,
        "encoder_layers": 1,
        "cross_asset_attention_layers": 1,
        "feed_forward_width": 32,
        "dropout": 0.0,
        "prediction_heads": list(PREDICTION_HEADS),
    }


def test_supervised_store_materializes_features_and_end_date_labels() -> None:
    dates = pd.date_range("2021-01-01", periods=5, freq="D", tz="UTC")
    rows = []
    for asset, symbol in enumerate(("AAAUSDT", "BBBUSDT", "CCCUSDT")):
        for day, date in enumerate(dates):
            rows.append({
                "date": date,
                "symbol": symbol,
                "feature_a": float(asset + day),
                "feature_b": float(10 * asset + day),
                "target_return": float(asset + day) / 100.0,
                "target_volatility": float(asset + day + 1) / 10.0,
            })
    panel = pd.DataFrame(rows)
    store = SupervisedTripletTensorStore(
        panel,
        ["feature_a", "feature_b"],
        ["target_return", "target_volatility"],
        lookback_days=3,
        relative_source_feature="feature_b",
    )
    scaler = FeatureScaler(
        feature_names=("feature_a", "feature_b"),
        mean=(0.0, 0.0),
        scale=(1.0, 1.0),
        source_relative_feature_index=1,
        fit_scope="representation_train_only",
        fit_start="2021-01-01",
        fit_end="2021-01-05",
        fit_rows=15,
    )
    x, y = store.materialize_batch([{
        "date": dates[-1],
        "triplet": ("AAAUSDT", "BBBUSDT", "CCCUSDT"),
    }], scaler)
    assert x.shape == (1, 3, 3, 3)
    assert y.shape == (1, 3, 2)
    np.testing.assert_allclose(y[0, :, 0], [0.04, 0.05, 0.06])
    assert np.allclose(x[..., -1].sum(axis=2), 0.0)


def test_eligibility_purges_labels_that_mature_after_boundary() -> None:
    dates = pd.date_range("2024-12-20", periods=12, freq="D", tz="UTC")
    rows = []
    for symbol in ("AAAUSDT", "BBBUSDT", "CCCUSDT"):
        for date in dates:
            rows.append({
                "date": date,
                "symbol": symbol,
                "target_window_end_date": date + pd.Timedelta(days=8),
                "in_validation": date.year == 2024,
                "supervised_sequence_ready": True,
                "label_complete": True,
            })
    availability, audit = eligible_supervised_availability(
        pd.DataFrame(rows),
        "in_validation",
        "2024-12-31",
        ["AAAUSDT", "BBBUSDT", "CCCUSDT"],
    )
    assert max(availability) == pd.Timestamp("2024-12-23", tz="UTC")
    assert audit["maximum_target_maturity"] == "2024-12-31"


def test_supervised_parameters_include_cross_asset_heads_but_not_reconstruction() -> None:
    model = MultiAssetPatchTransformer(3, _architecture())
    names = supervised_parameter_names(model)
    assert any(name.startswith("cross_asset_encoder.") for name in names)
    assert any(name.startswith("prediction_heads.") for name in names)
    assert not any(name.startswith("reconstruction_head.") for name in names)
    assert "mask_token" not in names


def test_calibration_offsets_and_projection_are_deterministic_and_monotone() -> None:
    labels = np.asarray([
        [[0.0, 0.2], [0.1, 0.3], [-0.1, 0.1]],
        [[0.2, 0.4], [-0.2, 0.2], [0.05, 0.5]],
    ], dtype=np.float32)
    predictions = {
        "return_q10": np.full((2, 3), 0.20),
        "return_q50": np.full((2, 3), 0.00),
        "return_q90": np.full((2, 3), -0.20),
        "volatility_7d": np.zeros((2, 3)),
    }
    first = compute_calibration_parameters(predictions, labels, 1e-6)
    second = compute_calibration_parameters(predictions, labels, 1e-6)
    assert first == second
    assert first["diagnostics"]["raw_quantile_crossing_rate"] == 1.0
    assert first["diagnostics"]["calibrated_quantile_crossing_rate"] == 0.0
    assert set(first["offsets"]) == {
        "return_q10", "return_q50", "return_q90", "log_volatility"
    }


def test_calibration_semantic_hash_ignores_checkpoint_container_hashes() -> None:
    first = {
        "fold": 1,
        "member_checkpoint_sha256": ["container-a"],
        "member_model_state_sha256": ["tensor-state"],
        "offsets": {"return_q50": 0.01},
    }
    second = {**first, "member_checkpoint_sha256": ["container-b"]}
    assert calibration_semantic_sha256(first) == calibration_semantic_sha256(second)
    assert calibration_state_sha256(first) != calibration_state_sha256(second)
    first["calibration_semantic_sha256"] = calibration_semantic_sha256(first)
    first["calibration_state_sha256"] = calibration_state_sha256(first)
    assert first["calibration_state_sha256"] == calibration_state_sha256(first)


def test_supervised_spec_freezes_eight_day_boundary_purge() -> None:
    blueprint = {
        "candidate_family_id": "fixture",
        "chronological_splits": {
            "supervised_train": ["2021-03-01", "2023-12-31"],
            "validation": ["2024-01-01", "2024-12-31"],
            "calibration": ["2025-01-01", "2025-12-31"],
        },
        "training": {
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "ensemble_rule": "mean",
        },
    }
    supervised = {
        "full_run": {
            "folds": [1, 2, 3],
            "seeds": [42, 7, 123],
            "train_samples_per_epoch": 8192,
            "validation_samples": 2048,
            "calibration_samples": 8192,
            "batch_size": 128,
            "maximum_epochs": 30,
            "early_stopping_patience": 5,
        },
        "smoke": {},
        "label_boundary_purge_days": 8,
        "gradient_clip_norm": 1.0,
        "calibration_method": {
            "return_offsets": "fixture",
            "volatility_offset": "fixture",
            "quantile_projection": "fixture",
            "updates_model_weights": False,
            "changes_policy_thresholds": False,
        },
        "calibration_seed": 20260713,
        "calibration_sampling_epoch": 0,
        "device": "cpu",
        "torch_threads": 1,
    }
    spec = build_supervised_spec(blueprint, supervised, smoke=False)
    assert spec["chronological_boundaries"]["supervised_train"][
        "last_eligible_signal_date"
    ] == "2023-12-23"
    assert spec["chronological_boundaries"]["validation"][
        "last_eligible_signal_date"
    ] == "2024-12-23"
    assert spec["chronological_boundaries"]["calibration"][
        "last_eligible_signal_date"
    ] == "2025-12-23"


def test_final_supervised_checkpoint_roundtrip(tmp_path) -> None:
    architecture = _architecture()
    model = MultiAssetPatchTransformer(3, architecture)
    path = tmp_path / "checkpoint.pt"
    state_hash = model_state_sha256(model.state_dict())
    _atomic_torch_save({
        "format_version": "v36_supervised_non_target_v1",
        "input_features": 3,
        "architecture": architecture,
        "architecture_sha256": _canonical_sha256(architecture),
        "metadata": {
            "fold": 1,
            "initialization_seed": 42,
            "model_state_sha256": state_hash,
        },
        "state_dict": model.state_dict(),
    }, path)
    restored, payload = load_supervised_checkpoint(path)
    assert payload["metadata"]["fold"] == 1
    assert model_state_sha256(restored.state_dict()) == state_hash
    assert all(
        torch.equal(model.state_dict()[name], restored.state_dict()[name])
        for name in model.state_dict()
    )
