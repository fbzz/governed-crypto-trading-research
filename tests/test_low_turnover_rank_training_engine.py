from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from tlm.low_turnover_rank_model import deterministic_feature_layer_norm
from tlm.low_turnover_rank_training_engine import (
    V83EarlyStopping,
    instantiate_v83_model,
    run_v83_training_job,
    verify_v83_checkpoint,
)
from test_low_turnover_rank_training_data import tiny_v83_fold


ROOT = Path(__file__).resolve().parents[1]


def test_deterministic_feature_layer_norm_matches_torch_on_cpu() -> None:
    torch.manual_seed(42)
    norm = torch.nn.LayerNorm(8)
    left = torch.randn(4, 16, 3, 8, requires_grad=True)
    right = left.detach().clone().requires_grad_(True)
    expected = norm(left)
    observed = deterministic_feature_layer_norm(right, norm)
    assert torch.allclose(observed, expected, atol=1.0e-6, rtol=1.0e-6)
    expected.square().mean().backward()
    observed.square().mean().backward()
    assert torch.allclose(right.grad, left.grad, atol=1.0e-6, rtol=1.0e-5)


def test_exact_capacity_and_early_stopping_delta() -> None:
    blueprint = json.loads(
        (ROOT / "artifacts/v80_low_turnover_rank_spec/blueprint.json").read_text()
    )
    model = instantiate_v83_model(blueprint, torch.device("cpu"), seed=42)
    assert sum(parameter.numel() for parameter in model.parameters()) == 10_993
    early = V83EarlyStopping(patience=2, minimum_delta=1.0e-6)
    assert early.update(1, 1.0)
    assert not early.update(2, 1.0 - 0.5e-6)
    assert not early.should_stop
    assert not early.update(3, 1.0 - 0.75e-6)
    assert early.should_stop


def test_cpu_checkpoint_roundtrip_and_zero_step_replay(tmp_path: Path) -> None:
    blueprint = json.loads(
        (ROOT / "artifacts/v80_low_turnover_rank_spec/blueprint.json").read_text()
    )
    contract = yaml.safe_load((ROOT / "research/phase_contracts/v083.yaml").read_text())
    data = tiny_v83_fold()
    context = {
        "phase": "v83", "family_id": contract["family_id"], "job_id": "1|42",
        "fold": 1, "seed": 42, "phase_contract_sha256": "b" * 64,
        "source_bundle_sha256": "c" * 64,
        "fold_feature_scaler_sha256": data.scale.feature_scaler.state_sha256(),
        "fold_scale_sha256": data.scale.state_sha256(),
        "excess_rms_scale": data.scale.excess_rms_scale,
        "data_access_sha256": data.access_receipt["access_sha256"],
        "optimizer_contract": contract["grid_optimizer_and_runtime_contract"],
        "train_symbols": list(data.train_symbols), "heldout_symbols_loaded": [],
        "target_assets_loaded": [], "prior_checkpoint_reused": False,
    }
    kwargs = dict(
        blueprint=blueprint, contract=contract, data=data, seed=42, context=context,
        resume_path=tmp_path / "job.resume.pt", final_path=tmp_path / "job.final.pt",
        device="cpu", train_samples=1, validation_samples=1, batch_size=1,
        maximum_epochs=1, patience=1,
    )
    completed = run_v83_training_job(**kwargs)
    assert completed["completed"] is True
    assert completed["optimizer_steps"] == 1
    assert verify_v83_checkpoint(
        tmp_path / "job.final.pt", blueprint=blueprint, context=context
    )["passed"]
    replay = run_v83_training_job(**kwargs)
    assert replay["status"] == "already_complete"
    assert replay["new_optimizer_steps"] == 0
