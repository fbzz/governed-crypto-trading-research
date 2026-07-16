from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import numpy as np
import pytest

from tlm.config import load_config
from tlm.v64_r2_probabilistic_state_gate_harness import (
    ProbabilisticStateGate,
    passes_abstention,
    probability_of_clearing_cost,
    run_v64_r2_probabilistic_state_gate_harness,
    student_t_df5_cdf_standardized,
)


ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path) -> dict:
    config = deepcopy(
        load_config(ROOT / "configs/v66_v64_r2_probabilistic_state_gate_harness.yaml")
    )
    config["v64_r2_probabilistic_state_gate_harness"]["project_root"] = str(ROOT)
    config["output_dir"] = str(tmp_path / "packet")
    return config


def test_v66_probabilistic_gate_has_exact_frozen_capacity() -> None:
    config = load_config(ROOT / "configs/v65_v64_r2_probabilistic_state_gate_spec.yaml")
    spec = config["v64_r2_probabilistic_state_gate_spec"]
    model = ProbabilisticStateGate(
        spec["state_gate_architecture"],
        degrees_of_freedom=spec["probabilistic_gate"]["degrees_of_freedom"],
        scale_floor=spec["probabilistic_gate"]["scale_floor"],
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 27_522


def test_v66_df5_cdf_and_probability_boundary_are_exact() -> None:
    assert math.isclose(
        float(student_t_df5_cdf_standardized(1.0)),
        0.8183912661754387,
        abs_tol=1e-15,
    )
    probability = probability_of_clearing_cost(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0]),
        asset_excess=0.0,
        transition_cost=0.0,
        degrees_of_freedom=5.0,
    )
    assert probability == 0.5
    assert passes_abstention(0.60, 0.60) is True
    assert passes_abstention(float(np.nextafter(0.60, 0.0)), 0.60) is False


def test_v66_harness_passes_and_full_packet_replays_bytes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = run_v64_r2_probabilistic_state_gate_harness(config)
    output = tmp_path / "packet"
    first_bytes = {path.name: path.read_bytes() for path in output.iterdir()}
    second = run_v64_r2_probabilistic_state_gate_harness(config)
    second_bytes = {path.name: path.read_bytes() for path in output.iterdir()}

    assert first == second
    assert first_bytes == second_bytes
    assert first["decision"] == (
        "authorize_v67_non_target_v64_r2_probabilistic_state_gate_dataset_only"
    )
    assert first["audit"]["passed"] is True
    assert first["audit"]["checks"]["byte_identical_replay"] is True
    assert first["smoke"]["ranker_parameters"] == 1_231_634
    assert first["smoke"]["state_gate_parameters"] == 27_522
    assert first["smoke"]["total_parameters"] == 1_259_156
    assert first["smoke"]["ranker_optimizer_present"] is False
    assert first["smoke"]["ranker_requires_grad"] is False
    assert first["smoke"]["resume_equivalent"] is True
    assert first["operation_ledger"]["parquet_deserializations"] == 0
    assert first["operation_ledger"]["previous_checkpoint_reads"] == 0
    assert first["operation_ledger"]["target_asset_loads"] == 0
    assert set(first_bytes) == {
        "artifact_manifest.json",
        "audit.json",
        "checkpoint_metadata.json",
        "completion_receipt.json",
        "harness_spec.json",
        "input_hash_receipt.json",
        "replay_receipt.json",
        "report.md",
        "resolved_config.yaml",
        "result.json",
        "smoke.json",
        "synthetic_checkpoint.pt",
    }


def test_v66_harness_rejects_input_hash_drift_before_checkpoint(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config["v64_r2_probabilistic_state_gate_harness"]["expected_input_sha256"][
        "v65_result"
    ] = "0" * 64
    with pytest.raises(ValueError, match="input receipt mismatch"):
        run_v64_r2_probabilistic_state_gate_harness(config)
    assert not (tmp_path / "packet" / "synthetic_checkpoint.pt").exists()


def test_v66_harness_rejects_extra_runtime_input(tmp_path: Path) -> None:
    config = _config(tmp_path)
    harness = config["v64_r2_probabilistic_state_gate_harness"]
    harness["inputs"]["real_panel"] = "data/processed/unsafe.parquet"
    harness["expected_input_sha256"]["real_panel"] = "0" * 64
    with pytest.raises(ValueError, match="allowlist drift"):
        run_v64_r2_probabilistic_state_gate_harness(config)
