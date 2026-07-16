from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from tlm.config import load_config
from tlm.decoupled_rank_state_spec import (
    ranker_parameter_count,
    run_decoupled_rank_state_spec,
    state_gate_parameter_count,
)


ROOT = Path(__file__).resolve().parents[1]


def _file_hashes(path: Path) -> dict[str, str]:
    return {
        item.name: hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(path.iterdir())
        if item.is_file()
    }


def test_v60_parameter_counts_are_frozen() -> None:
    config = load_config(ROOT / "configs/v60_decoupled_rank_state_spec.yaml")
    spec = config["decoupled_rank_state_spec"]
    ranker = ranker_parameter_count(spec["ranker_architecture"])
    gate = state_gate_parameter_count(spec["state_gate_architecture"])
    assert ranker == 1_231_634
    assert gate == 27_489
    assert ranker + gate == 1_259_123
    assert ranker + gate <= spec["capacity_contract"]["parameter_ceiling"]
    assert spec["capacity_contract"]["size_sweep_allowed"] is False
    assert spec["capacity_contract"]["larger_model_allowed"] is False


def test_v60_spec_builder_is_metadata_only_and_byte_deterministic(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/v60_decoupled_rank_state_spec.yaml")
    config = copy.deepcopy(config)
    config["decoupled_rank_state_spec"]["project_root"] = str(ROOT)
    config["output_dir"] = str(tmp_path / "packet")
    first = run_decoupled_rank_state_spec(config)
    first_hashes = _file_hashes(tmp_path / "packet")
    second = run_decoupled_rank_state_spec(config)
    second_hashes = _file_hashes(tmp_path / "packet")

    assert first == second
    assert first_hashes == second_hashes
    assert first["decision"] == "authorize_v61_synthetic_decoupled_rank_state_harness_only"
    assert first["audit"]["passed"] is True
    zero_fields = {
        "parquet_deserializations",
        "checkpoint_reads",
        "model_instantiations",
        "optimizer_steps",
        "predictions",
        "performance_metrics",
        "pnl_computations",
        "outcome_source_reads",
        "target_asset_rows",
    }
    assert all(first["summary"][field] == 0 for field in zero_fields)


def test_v60_gate_is_independent_by_contract() -> None:
    config = load_config(ROOT / "configs/v60_decoupled_rank_state_spec.yaml")
    spec = config["decoupled_rank_state_spec"]
    gate = spec["state_gate_architecture"]
    gradients = spec["objective"]["gradient_contract"]
    assert gate["independent_encoder"] is True
    assert gate["ranker_representation_input"] == "none"
    assert gradients == {
        "shared_parameters": False,
        "combined_scalar_loss": False,
        "gate_gradients_enter_ranker": False,
        "ranker_gradients_enter_gate": False,
        "synthetic_gradient_isolation_gate_required": True,
    }
