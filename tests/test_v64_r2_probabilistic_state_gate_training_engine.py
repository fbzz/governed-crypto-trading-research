from __future__ import annotations

import json
from pathlib import Path

import torch

from tlm.core import file_sha256
from tlm.state_conditioned_multi_horizon_training_engine import (
    clone_model_state,
    semantic_state_sha256,
)
from tlm.v64_r2_probabilistic_state_gate_training_engine import (
    instantiate_v68_models,
    run_v68_training_job,
    verify_v68_checkpoint,
)

from test_v64_r2_probabilistic_state_gate_training_data import tiny_v68_fold


ROOT = Path(__file__).resolve().parents[1]


def test_v68_gate_only_checkpoint_resume_and_zero_step_replay(tmp_path: Path) -> None:
    blueprint = json.loads(
        (ROOT / "artifacts/v65_v64_r2_probabilistic_state_gate_spec/blueprint.json").read_text()
    )
    contract = __import__("yaml").safe_load(
        (ROOT / "research/phase_contracts/v068.yaml").read_text()
    )
    ranker, _ = instantiate_v68_models(blueprint, torch.device("cpu"), seed=42)
    ranker_state = clone_model_state(ranker)
    ranker_hash = semantic_state_sha256(ranker_state)
    source = tmp_path / "v63.final.pt"
    torch.save(
        {
            "format_version": "v63_decoupled_rank_state_checkpoint_v1",
            "kind": "final", "context": {"job_id": "1|42"},
            "ranker_current_state": ranker_state,
            "gate_current_state": {"legacy_gate_must_not_be_reused": torch.tensor([9.0])},
        },
        source,
    )
    data = tiny_v68_fold()
    context = {
        "phase": "v68", "family_id": contract["family_id"], "job_id": "1|42",
        "fold": 1, "seed": 42, "phase_contract_sha256": "c" * 64,
        "source_bundle_sha256": "d" * 64,
        "fold_feature_scaler_sha256": data.scale.feature_scaler.state_sha256(),
        "market_target_scaler_sha256": "e" * 64, "data_access_sha256": "b" * 64,
        "train_symbols": list(data.train_symbols), "heldout_symbols_loaded": [],
        "target_assets_loaded": [],
    }
    kwargs = dict(
        blueprint=blueprint, contract=contract, data=data, seed=42, context=context,
        source_checkpoint_path=source,
        source_checkpoint_file_sha256=file_sha256(source),
        source_ranker_state_sha256=ranker_hash,
        resume_path=tmp_path / "job.resume.pt", final_path=tmp_path / "job.final.pt",
        device="cpu", train_samples=2, validation_samples=2, batch_size=2,
        maximum_epochs=1, patience=1,
    )
    completed = run_v68_training_job(**kwargs)
    assert completed["completed"] is True
    assert completed["optimizer_steps"] == 1
    assert completed["ranker_state_sha256"] == ranker_hash
    payload = torch.load(tmp_path / "job.final.pt", map_location="cpu", weights_only=False)
    assert payload["ranker_optimizer_present"] is False
    assert payload["old_gate_state_present"] is False
    assert "legacy_gate_must_not_be_reused" not in str(payload.keys())
    assert verify_v68_checkpoint(
        tmp_path / "job.final.pt", blueprint=blueprint, context=context,
        expected_ranker_state_sha256=ranker_hash,
    )["passed"]
    replay = run_v68_training_job(**kwargs)
    assert replay["status"] == "already_complete"
    assert replay["new_optimizer_steps"] == 0
