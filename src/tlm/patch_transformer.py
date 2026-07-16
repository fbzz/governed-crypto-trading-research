from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn
import yaml


PREDICTION_HEADS = ("return_q10", "return_q50", "return_q90", "volatility_7d")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class MultiAssetPatchTransformer(nn.Module):
    def __init__(
        self,
        input_features: int,
        architecture: dict,
        expected_prediction_heads: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        self.input_features = int(input_features)
        self.lookback_days = int(architecture["lookback_days"])
        self.triplet_size = int(architecture["input_triplet_size"])
        self.patch_length = int(architecture["patch_length_days"])
        self.patch_stride = int(architecture["patch_stride_days"])
        self.d_model = int(architecture["d_model"])
        self.patch_count = (
            (self.lookback_days - self.patch_length) // self.patch_stride + 1
        )
        heads = int(architecture["attention_heads"])
        if self.d_model % heads:
            raise ValueError("d_model must be divisible by attention_heads")
        if self.patch_count < 1:
            raise ValueError("Patch geometry does not fit the lookback")
        prediction_heads = tuple(architecture["prediction_heads"])
        expected_heads = (
            PREDICTION_HEADS
            if expected_prediction_heads is None
            else tuple(expected_prediction_heads)
        )
        if prediction_heads != expected_heads:
            raise ValueError("Prediction-head contract drift")
        self.prediction_head_names = prediction_heads

        patch_width = self.patch_length * self.input_features
        self.patch_projection = nn.Linear(patch_width, self.d_model)
        self.temporal_position = nn.Parameter(
            torch.zeros(1, self.patch_count, self.d_model)
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.d_model))
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=heads,
            dim_feedforward=int(architecture["feed_forward_width"]),
            dropout=float(architecture["dropout"]),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            temporal_layer,
            num_layers=int(architecture["encoder_layers"]),
            enable_nested_tensor=False,
        )
        cross_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=heads,
            dim_feedforward=int(architecture["feed_forward_width"]),
            dropout=float(architecture["dropout"]),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross_asset_encoder = nn.TransformerEncoder(
            cross_layer,
            num_layers=int(architecture["cross_asset_attention_layers"]),
            enable_nested_tensor=False,
        )
        self.temporal_norm = nn.LayerNorm(self.d_model)
        self.cross_asset_norm = nn.LayerNorm(self.d_model)
        self.prediction_heads = nn.ModuleDict({
            name: nn.Linear(self.d_model, 1) for name in prediction_heads
        })
        self.reconstruction_head = nn.Linear(self.d_model, patch_width)
        causal_mask = torch.triu(
            torch.ones(self.patch_count, self.patch_count, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_patch_mask", causal_mask, persistent=False)
        nn.init.normal_(self.temporal_position, mean=0.0, std=0.02)
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

    def extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input(x)
        patches = x.unfold(1, self.patch_length, self.patch_stride)
        return patches.permute(0, 2, 1, 4, 3).contiguous()

    def _validate_input(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError("Input must have shape [batch, time, assets, features]")
        if tuple(x.shape[1:]) != (
            self.lookback_days,
            self.triplet_size,
            self.input_features,
        ):
            raise ValueError(
                "Input contract drift: expected "
                f"[batch,{self.lookback_days},{self.triplet_size},{self.input_features}]"
            )

    def encode_temporal_patches(
        self,
        x: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        patches = self.extract_patches(x)
        batch, assets, patch_count, _, _ = patches.shape
        flattened = patches.flatten(start_dim=3)
        tokens = self.patch_projection(flattened)
        tokens = tokens + self.temporal_position[:, :patch_count].unsqueeze(1)
        if patch_mask is not None:
            if patch_mask.shape != (batch, assets, patch_count):
                raise ValueError("Patch mask must have shape [batch, assets, patches]")
            tokens = torch.where(
                patch_mask[..., None], self.mask_token.expand_as(tokens), tokens
            )
        encoded = self.temporal_encoder(
            tokens.reshape(batch * assets, patch_count, self.d_model),
            mask=self.causal_patch_mask[:patch_count, :patch_count],
        )
        return self.temporal_norm(encoded).reshape(
            batch, assets, patch_count, self.d_model
        )

    def forward(
        self,
        x: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
        return_reconstruction: bool = False,
    ) -> dict[str, torch.Tensor]:
        temporal = self.encode_temporal_patches(x, patch_mask=patch_mask)
        cross = self.cross_asset_encoder(temporal[:, :, -1, :])
        cross = self.cross_asset_norm(cross)
        output = {
            name: head(cross).squeeze(-1)
            for name, head in self.prediction_heads.items()
        }
        if return_reconstruction:
            reconstructed = self.reconstruction_head(temporal)
            output["patch_reconstruction"] = reconstructed.reshape(
                x.shape[0],
                self.triplet_size,
                self.patch_count,
                self.patch_length,
                self.input_features,
            )
        return output

    def reconstruct_masked_patches(
        self,
        x: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run only the representation path used by masked pretraining."""
        temporal = self.encode_temporal_patches(x, patch_mask=patch_mask)
        reconstructed = self.reconstruction_head(temporal)
        return reconstructed.reshape(
            x.shape[0],
            self.triplet_size,
            self.patch_count,
            self.patch_length,
            self.input_features,
        )


def save_model_checkpoint(
    model: MultiAssetPatchTransformer,
    path: str | Path,
    architecture: dict,
    metadata: dict,
) -> None:
    required = {
        "candidate_family_id",
        "feature_schema_sha256",
        "dataset_manifest_sha256",
        "initialization_seed",
        "checkpoint_status",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"Checkpoint metadata is missing: {missing}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": "v33_smoke_only",
        "input_features": model.input_features,
        "architecture": architecture,
        "architecture_sha256": _canonical_sha256(architecture),
        "state_dict": model.state_dict(),
        "metadata": metadata,
    }, path)


def load_model_checkpoint(
    path: str | Path,
) -> tuple[MultiAssetPatchTransformer, dict]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload["format_version"] != "v33_smoke_only":
        raise ValueError("Unsupported checkpoint format")
    if payload["architecture_sha256"] != _canonical_sha256(payload["architecture"]):
        raise ValueError("Checkpoint architecture hash mismatch")
    model = MultiAssetPatchTransformer(
        int(payload["input_features"]), payload["architecture"]
    )
    model.load_state_dict(payload["state_dict"])
    return model, payload


def build_model_spec(architecture: dict, input_features: int) -> dict[str, object]:
    patch_count = (
        (int(architecture["lookback_days"]) - int(architecture["patch_length_days"]))
        // int(architecture["patch_stride_days"])
        + 1
    )
    spec = {
        "version": "v33",
        "candidate_family_id": "tlm_multi_asset_target_transfer_v2",
        "input_contract": {
            "layout": "batch_time_asset_feature",
            "shape": [None, 256, 3, input_features],
            "dtype": "float32",
        },
        "patch_contract": {
            "length_days": architecture["patch_length_days"],
            "stride_days": architecture["patch_stride_days"],
            "patch_count": patch_count,
            "flatten_order": "time_then_feature_per_asset",
        },
        "temporal_encoder": {
            "shared_across_assets": True,
            "layers": architecture["encoder_layers"],
            "causal_patch_mask": True,
            "activation": "gelu",
            "norm_first": True,
            "asset_slot_embedding": False,
        },
        "cross_asset_encoder": {
            "layers": architecture["cross_asset_attention_layers"],
            "permutation_equivariant": True,
            "asset_slot_embedding": False,
        },
        "dimensions": {
            "d_model": architecture["d_model"],
            "attention_heads": architecture["attention_heads"],
            "feed_forward_width": architecture["feed_forward_width"],
            "dropout": architecture["dropout"],
        },
        "prediction_heads": {
            name: {"shape": [None, 3], "raw_output": True}
            for name in PREDICTION_HEADS
        },
        "pretraining_interface": {
            "objective": "masked_past_patch_reconstruction",
            "mask_shape": [None, 3, patch_count],
            "reconstruction_shape": [
                None, 3, patch_count, architecture["patch_length_days"], input_features
            ],
        },
    }
    spec["model_spec_sha256"] = _canonical_sha256(spec)
    return spec


def _report(result: dict[str, object]) -> str:
    smoke = result["smoke"]
    return "\n".join([
        "# TLM v33 Patch Transformer Implementation",
        "",
        "## Decision",
        "",
        "**FROZEN ARCHITECTURE IMPLEMENTED AND SMOKE-VALIDATED; REAL TRAINING REMAINS BLOCKED.**",
        "",
        f"Parameters: **{smoke['parameter_count']:,}**",
        f"Patch count: **{smoke['patch_count']}**",
        f"Model-spec SHA-256: `{result['model_spec']['model_spec_sha256']}`",
        f"Smoke-checkpoint SHA-256: `{smoke['checkpoint_sha256']}`",
        "",
        "Synthetic checks passed for output shapes, causal temporal attention, asset-permutation equivariance, masked-patch reconstruction, gradients, and checkpoint roundtrip.",
        "",
        "No real panel, sequence, label, scaler, optimizer step, target asset, portfolio, performance metric, or PnL was loaded or executed.",
        "",
        "## Next action",
        "",
        "V34 may implement and freeze the scientific harness: train-only scaler, deterministic triplet sampler/masks, losses, early stopping, controls, costs, and paired Monte Carlo on fixtures/smoke data. Full v35 pretraining remains forbidden until that audit passes.",
        "",
    ])


def run_patch_transformer_implementation(config: dict) -> dict[str, object]:
    implementation = config["patch_transformer_implementation"]
    root = Path(implementation["project_root"]).resolve()
    paths = {
        name: root / relative for name, relative in implementation["inputs"].items()
    }
    for name, path in paths.items():
        expected = implementation["expected_input_sha256"][name]
        if not path.is_file() or _sha256_file(path) != expected:
            raise RuntimeError(f"V33 input missing or hash drifted: {name}")
    amendment = _load_json(paths["v29_amendment"])
    v32_result = _load_json(paths["v32_result"])
    v32_audit = _load_json(paths["v32_audit"])
    dataset_manifest = _load_json(paths["v32_dataset_manifest"])
    feature_schema = _load_json(paths["v32_feature_schema"])
    architecture = amendment["blueprint"]["architecture"]
    input_features = len(feature_schema["model_feature_order"])
    model_spec = build_model_spec(architecture, input_features)

    initialization_seed = int(implementation["initialization_seed"])
    torch.manual_seed(initialization_seed)
    model = MultiAssetPatchTransformer(input_features, architecture)
    model.eval()
    fixture = torch.linspace(
        -1.0,
        1.0,
        steps=2 * 256 * 3 * input_features,
        dtype=torch.float32,
    ).reshape(2, 256, 3, input_features)
    with torch.no_grad():
        predictions = model(fixture)
        perm = torch.tensor([2, 0, 1])
        permuted = model(fixture[:, :, perm, :])
        temporal = model.encode_temporal_patches(fixture)
        altered = fixture.clone()
        altered[:, 200:, :, :] += 100.0
        altered_temporal = model.encode_temporal_patches(altered)
    early_patch_count = (200 - model.patch_length) // model.patch_stride + 1
    permutation_passes = all(
        torch.allclose(permuted[name], predictions[name][:, perm], atol=1e-5, rtol=1e-5)
        for name in PREDICTION_HEADS
    )
    causal_passes = torch.allclose(
        temporal[:, :, :early_patch_count],
        altered_temporal[:, :, :early_patch_count],
        atol=1e-6,
        rtol=1e-6,
    )

    mask = torch.zeros(2, 3, model.patch_count, dtype=torch.bool)
    mask[:, :, ::2] = True
    model.train()
    reconstructed = model(fixture, patch_mask=mask, return_reconstruction=True)
    loss = sum(reconstructed[name].mean() for name in PREDICTION_HEADS)
    loss = loss + reconstructed["patch_reconstruction"][mask].mean()
    loss.backward()
    gradients_pass = all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for name, parameter in model.named_parameters()
        if any(token in name for token in (
            "patch_projection", "temporal_encoder", "cross_asset_encoder",
            "prediction_heads", "reconstruction_head", "mask_token",
        ))
    )

    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "smoke_checkpoint.pt"
    metadata = {
        "candidate_family_id": amendment["blueprint"]["candidate_family_id"],
        "feature_schema_sha256": _sha256_file(paths["v32_feature_schema"]),
        "dataset_manifest_sha256": _sha256_file(paths["v32_dataset_manifest"]),
        "initialization_seed": initialization_seed,
        "checkpoint_status": "synthetic_smoke_only_not_trained",
    }
    save_model_checkpoint(model, checkpoint_path, architecture, metadata)
    loaded, checkpoint = load_model_checkpoint(checkpoint_path)
    loaded.eval()
    model.eval()
    with torch.no_grad():
        before = model(fixture)
        after = loaded(fixture)
    roundtrip_passes = all(
        torch.equal(before[name], after[name]) for name in PREDICTION_HEADS
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    checks = {
        "v29_blueprint_hash_matches": amendment["blueprint_sha256"]
        == implementation["expected_v29_blueprint_sha256"],
        "v32_audit_passes": bool(v32_audit["passed"]),
        "v32_authorizes_only_v33": v32_result["decision"]
        == "authorize_v33_patch_transformer_implementation_only",
        "input_contract_matches_v32": dataset_manifest["tensor_contract"]["x_shape"]
        == [256, 3, input_features],
        "architecture_matches_frozen_blueprint": model.lookback_days == 256
        and model.triplet_size == 3
        and model.patch_length == 16
        and model.patch_stride == 8
        and model.patch_count == 31
        and model.d_model == 96,
        "all_prediction_shapes_pass": all(
            predictions[name].shape == (2, 3) for name in PREDICTION_HEADS
        ),
        "causal_temporal_prefix_is_invariant": bool(causal_passes),
        "asset_permutation_equivariance_passes": bool(permutation_passes),
        "reconstruction_shape_passes": reconstructed["patch_reconstruction"].shape
        == (2, 3, 31, 16, input_features),
        "mask_shape_passes": mask.shape == (2, 3, 31),
        "critical_gradients_are_finite": bool(gradients_pass),
        "checkpoint_roundtrip_is_exact": bool(roundtrip_passes),
        "checkpoint_metadata_is_complete": checkpoint["metadata"] == metadata,
        "real_panel_not_loaded": True,
        "real_sequence_not_loaded": True,
        "label_columns_not_read": True,
        "scaler_fit_count_is_zero": True,
        "optimizer_step_count_is_zero": True,
        "pretraining_epoch_count_is_zero": True,
        "supervised_training_epoch_count_is_zero": True,
        "target_asset_load_count_is_zero": True,
        "portfolio_count_is_zero": True,
        "performance_metric_count_is_zero": True,
        "pnl_evaluation_count_is_zero": True,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise RuntimeError(f"V33 Patch Transformer audit failed: {checks}")
    smoke = {
        "fixture": "synthetic_linspace_no_market_data",
        "initialization_seed": initialization_seed,
        "parameter_count": parameter_count,
        "patch_count": model.patch_count,
        "prediction_shapes": {
            name: list(predictions[name].shape) for name in PREDICTION_HEADS
        },
        "reconstruction_shape": list(reconstructed["patch_reconstruction"].shape),
        "causal_unchanged_patch_count": early_patch_count,
        "checkpoint_path": str(checkpoint_path.relative_to(root)),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
    }
    result = {
        "version": "v33",
        "decision": "authorize_v34_scientific_harness_implementation_only",
        "model_spec": model_spec,
        "smoke": smoke,
        "checkpoint_metadata": metadata,
        "tested": {
            "synthetic_forward": True,
            "synthetic_backward": True,
            "causal_mask": True,
            "asset_permutation_equivariance": True,
            "masked_patch_reconstruction": True,
            "checkpoint_roundtrip": True,
            "real_data_loaded": False,
            "scaler_fitted": False,
            "model_trained": False,
            "performance_metrics_computed": False,
            "pnl_computed": False,
            "target_assets_loaded": False,
            "improvement_status": "unknown_not_evaluated",
            "drawdown_status": "unknown_not_evaluated",
        },
        "audit": {"passed": True, "checks": checks},
    }
    for name, value in {
        "model_spec.json": model_spec,
        "smoke.json": smoke,
        "checkpoint_metadata.json": metadata,
        "audit.json": result["audit"],
        "result.json": result,
    }.items():
        (output / name).write_text(
            json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
        )
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    return result
