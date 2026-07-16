from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from tlm.config import load_config
from tlm.v64_r2_probabilistic_state_gate_spec import (
    ranker_parameter_count,
    run_v64_r2_probabilistic_state_gate_spec,
    state_gate_parameter_count,
)


ROOT = Path(__file__).resolve().parents[1]


def _file_hashes(path: Path) -> dict[str, str]:
    return {
        item.name: hashlib.sha256(item.read_bytes()).hexdigest()
        for item in sorted(path.iterdir())
        if item.is_file()
    }


def test_v65_parameter_counts_are_frozen() -> None:
    config = load_config(ROOT / "configs/v65_v64_r2_probabilistic_state_gate_spec.yaml")
    spec = config["v64_r2_probabilistic_state_gate_spec"]
    ranker = ranker_parameter_count(spec["ranker_contract"]["architecture"])
    gate = state_gate_parameter_count(spec["state_gate_architecture"])
    assert ranker == 1_231_634
    assert gate == 27_522
    assert ranker + gate == 1_259_156
    assert ranker + gate <= spec["capacity_contract"]["parameter_ceiling"]
    assert spec["capacity_contract"]["size_sweep_allowed"] is False
    assert spec["capacity_contract"]["state_gate_variant_count"] == 1


def test_v65_spec_builder_is_metadata_only_and_byte_deterministic(
    tmp_path: Path,
) -> None:
    config = load_config(ROOT / "configs/v65_v64_r2_probabilistic_state_gate_spec.yaml")
    config = copy.deepcopy(config)
    config["v64_r2_probabilistic_state_gate_spec"]["project_root"] = str(ROOT)
    config["output_dir"] = str(tmp_path / "packet")
    first = run_v64_r2_probabilistic_state_gate_spec(config)
    first_hashes = _file_hashes(tmp_path / "packet")
    second = run_v64_r2_probabilistic_state_gate_spec(config)
    second_hashes = _file_hashes(tmp_path / "packet")

    assert first == second
    assert first_hashes == second_hashes
    assert first["decision"] == (
        "authorize_v66_synthetic_v64_r2_probabilistic_state_gate_harness_only"
    )
    assert first["audit"]["passed"] is True
    assert first["summary"]["frozen_ranker_state_receipts"] == 9
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


def test_v65_changes_only_gate_and_abstention_contract() -> None:
    config = load_config(ROOT / "configs/v65_v64_r2_probabilistic_state_gate_spec.yaml")
    spec = config["v64_r2_probabilistic_state_gate_spec"]
    v60 = load_config(ROOT / "configs/v60_decoupled_rank_state_spec.yaml")[
        "decoupled_rank_state_spec"
    ]
    assert spec["ranker_contract"]["architecture"] == v60["ranker_architecture"]
    assert spec["ranker_contract"]["objective"] == v60["objective"]["ranker"]
    assert spec["ranker_contract"]["weights"]["gate_state_reuse"] == "forbidden"
    assert spec["state_gate_architecture"]["output_width"] == 2
    assert spec["probabilistic_gate"]["distribution"] == "student_t_location_scale"
    assert spec["probabilistic_gate"]["degrees_of_freedom"] == 5.0
    assert spec["policy"]["abstention_probability_threshold"] == 0.60
    assert spec["policy"]["threshold_sweep_allowed"] is False
    assert spec["policy"]["action_space"] == ["long_one_asset", "cash"]
    assert spec["target_contract"]["status"] == "sealed"
