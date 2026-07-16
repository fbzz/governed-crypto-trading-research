from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

from tlm.non_target_pretraining import (
    TripletTensorStore,
    _atomic_torch_save,
    _canonical_sha256,
    _load_resume,
    _pretraining_parameters,
    _save_resume,
    load_pretrained_checkpoint,
    pretraining_parameter_names,
)
from tlm.patch_transformer import MultiAssetPatchTransformer
from tlm.scientific_harness import EarlyStoppingState, FeatureScaler


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
        "prediction_heads": [
            "return_q10",
            "return_q50",
            "return_q90",
            "volatility_7d",
        ],
    }


def test_triplet_store_materializes_scaled_feature_only_tensor() -> None:
    dates = pd.date_range("2021-01-01", periods=5, freq="D", tz="UTC")
    rows = []
    for asset_number, symbol in enumerate(("AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT")):
        for day, date in enumerate(dates):
            rows.append({
                "date": date,
                "symbol": symbol,
                "feature_a": float(asset_number + day),
                "feature_b": float(10 * asset_number + day),
                "forward_label_that_must_not_be_used": 999.0,
            })
    panel = pd.DataFrame(rows)
    store = TripletTensorStore(
        panel[["date", "symbol", "feature_a", "feature_b"]],
        ["feature_a", "feature_b"],
        lookback_days=3,
        relative_source_feature="feature_b",
    )
    scaler = FeatureScaler(
        feature_names=("feature_a", "feature_b"),
        mean=(1.0, 2.0),
        scale=(2.0, 4.0),
        source_relative_feature_index=1,
        fit_scope="representation_train_only",
        fit_start="2021-01-01",
        fit_end="2021-01-05",
        fit_rows=20,
    )
    values = store.materialize_batch([{
        "date": dates[-1],
        "triplet": ("AAAUSDT", "BBBUSDT", "CCCUSDT"),
    }], scaler)
    assert values.shape == (1, 3, 3, 3)
    assert values.dtype == np.float32
    assert np.isfinite(values).all()
    assert np.allclose(values[..., -1].sum(axis=2), 0.0, atol=1e-6)
    assert values[0, 0, 0, 0] == (2.0 - 1.0) / 2.0


def test_pretraining_path_skips_cross_asset_and_prediction_heads() -> None:
    model = MultiAssetPatchTransformer(3, _architecture())
    names = pretraining_parameter_names(model)
    assert names
    assert any(name.startswith("temporal_encoder.") for name in names)
    assert any(name.startswith("reconstruction_head.") for name in names)
    assert not any(name.startswith("cross_asset_encoder.") for name in names)
    assert not any(name.startswith("prediction_heads.") for name in names)
    assert len(_pretraining_parameters(model)) == len(names)


def test_direct_reconstruction_matches_full_forward_reconstruction() -> None:
    torch.manual_seed(11)
    model = MultiAssetPatchTransformer(3, _architecture()).eval()
    x = torch.randn(2, 8, 3, 3)
    mask = torch.zeros(2, 3, model.patch_count, dtype=torch.bool)
    mask[:, :, 1] = True
    direct = model.reconstruct_masked_patches(x, mask)
    full = model(x, patch_mask=mask, return_reconstruction=True)[
        "patch_reconstruction"
    ]
    assert torch.equal(direct, full)


def test_epoch_resume_restores_model_optimizer_early_stop_and_rng(tmp_path) -> None:
    torch.manual_seed(17)
    model = MultiAssetPatchTransformer(3, _architecture())
    optimizer = torch.optim.AdamW(_pretraining_parameters(model), lr=0.001)
    early = EarlyStoppingState(patience=3)
    early.update(1, 0.75)
    metadata = {"fold": 1, "seed": 17}
    history = [{"epoch": 1, "validation_loss": 0.75}]
    expected_state = {
        name: tensor.detach().clone() for name, tensor in model.state_dict().items()
    }
    expected_rng = torch.get_rng_state().clone()
    path = tmp_path / "resume.pt"
    _save_resume(path, model, optimizer, early, history, 1, metadata)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1.0)
    completed, restored_early, restored_history = _load_resume(
        path, model, optimizer, metadata
    )
    assert completed == 1
    assert asdict(restored_early) == asdict(early)
    assert restored_history == history
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert all(
        torch.equal(model.state_dict()[name], value)
        for name, value in expected_state.items()
    )


def test_final_pretrained_checkpoint_roundtrip(tmp_path) -> None:
    torch.manual_seed(23)
    architecture = _architecture()
    model = MultiAssetPatchTransformer(3, architecture)
    path = tmp_path / "checkpoint.pt"
    payload = {
        "format_version": "v35_non_target_pretraining_v1",
        "input_features": 3,
        "architecture": architecture,
        "architecture_sha256": _canonical_sha256(architecture),
        "metadata": {"fold": 2, "initialization_seed": 23},
        "state_dict": model.state_dict(),
    }
    _atomic_torch_save(payload, path)
    restored, restored_payload = load_pretrained_checkpoint(path)
    assert restored_payload["metadata"] == payload["metadata"]
    assert all(
        torch.equal(model.state_dict()[name], restored.state_dict()[name])
        for name in model.state_dict()
    )
