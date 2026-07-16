from pathlib import Path

import torch

from tlm.patch_transformer import (
    MultiAssetPatchTransformer,
    PREDICTION_HEADS,
    load_model_checkpoint,
    save_model_checkpoint,
)


def _architecture():
    return {
        "lookback_days": 32,
        "input_triplet_size": 3,
        "patch_length_days": 8,
        "patch_stride_days": 4,
        "d_model": 24,
        "attention_heads": 4,
        "feed_forward_width": 48,
        "dropout": 0.0,
        "encoder_layers": 1,
        "cross_asset_attention_layers": 1,
        "prediction_heads": list(PREDICTION_HEADS),
    }


def test_patch_model_shapes_reconstruction_and_gradients():
    model = MultiAssetPatchTransformer(5, _architecture())
    x = torch.randn(2, 32, 3, 5)
    mask = torch.zeros(2, 3, 7, dtype=torch.bool)
    mask[:, :, ::2] = True
    output = model(x, patch_mask=mask, return_reconstruction=True)
    assert all(output[name].shape == (2, 3) for name in PREDICTION_HEADS)
    assert output["patch_reconstruction"].shape == (2, 3, 7, 8, 5)
    sum(value.mean() for value in output.values()).backward()
    assert model.patch_projection.weight.grad is not None
    assert model.reconstruction_head.weight.grad is not None


def test_causal_prefix_and_asset_permutation_equivariance():
    torch.manual_seed(4)
    model = MultiAssetPatchTransformer(5, _architecture()).eval()
    x = torch.randn(2, 32, 3, 5)
    changed = x.clone()
    changed[:, 24:] += 50
    with torch.no_grad():
        original_temporal = model.encode_temporal_patches(x)
        changed_temporal = model.encode_temporal_patches(changed)
        original = model(x)
        perm = torch.tensor([2, 0, 1])
        permuted = model(x[:, :, perm])
    assert torch.equal(original_temporal[:, :, :5], changed_temporal[:, :, :5])
    for name in PREDICTION_HEADS:
        torch.testing.assert_close(permuted[name], original[name][:, perm])


def test_smoke_checkpoint_roundtrip(tmp_path: Path):
    torch.manual_seed(7)
    architecture = _architecture()
    model = MultiAssetPatchTransformer(5, architecture).eval()
    path = tmp_path / "smoke.pt"
    metadata = {
        "candidate_family_id": "fixture",
        "feature_schema_sha256": "a" * 64,
        "dataset_manifest_sha256": "b" * 64,
        "initialization_seed": 7,
        "checkpoint_status": "synthetic_smoke_only_not_trained",
    }
    save_model_checkpoint(model, path, architecture, metadata)
    loaded, payload = load_model_checkpoint(path)
    loaded.eval()
    x = torch.randn(1, 32, 3, 5)
    with torch.no_grad():
        first = model(x)
        second = loaded(x)
    assert payload["metadata"] == metadata
    for name in PREDICTION_HEADS:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
