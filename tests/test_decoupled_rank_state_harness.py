from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from tlm.config import load_config
from tlm.decoupled_rank_state_harness import (
    IndependentStateGate,
    decoupled_rank_state_positions,
    run_decoupled_rank_state_harness,
)


ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path) -> dict:
    config = deepcopy(load_config(ROOT / "configs/v61_decoupled_rank_state_harness.yaml"))
    config["decoupled_rank_state_harness"]["project_root"] = str(ROOT)
    config["output_dir"] = str(tmp_path / "packet")
    return config


def test_v61_state_gate_has_exact_frozen_capacity() -> None:
    config = load_config(ROOT / "configs/v60_decoupled_rank_state_spec.yaml")
    architecture = config["decoupled_rank_state_spec"]["state_gate_architecture"]
    model = IndependentStateGate(architecture)
    assert sum(parameter.numel() for parameter in model.parameters()) == 27_489


def test_v61_policy_strict_entry_switch_exit_and_liquidation_fixture() -> None:
    excess = np.array(
        [
            [0.0010, 0.0000, -0.0010],
            [0.0010, 0.0040, -0.0050],
            [0.0000, 0.0010, -0.0010],
        ]
    )
    market = np.array([0.0020, 0.0040, -0.0040])
    momentum = np.ones_like(excess)
    eligible = np.ones_like(excess, dtype=bool)
    result = decoupled_rank_state_positions(
        excess,
        market,
        momentum,
        eligible,
        base_cost=0.001,
        switch_hurdle=0.002,
    )
    assert result["actions"] == ["entry", "switch", "edge_exit"]
    assert result["selected_assets"] == [0, 1, None]


def test_v61_harness_passes_and_full_packet_replays_bytes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first = run_decoupled_rank_state_harness(config)
    output = tmp_path / "packet"
    first_bytes = {path.name: path.read_bytes() for path in output.iterdir()}
    second = run_decoupled_rank_state_harness(config)
    second_bytes = {path.name: path.read_bytes() for path in output.iterdir()}

    assert first == second
    assert first_bytes == second_bytes
    assert first["decision"] == (
        "authorize_v62_non_target_decoupled_rank_state_dataset_only"
    )
    assert first["audit"]["passed"] is True
    assert first["audit"]["checks"]["byte_identical_replay"] is True
    assert first["smoke"]["ranker_parameters"] == 1_231_634
    assert first["smoke"]["state_gate_parameters"] == 27_489
    assert first["smoke"]["total_parameters"] == 1_259_123
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


def test_v61_harness_rejects_input_hash_drift_before_checkpoint(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["decoupled_rank_state_harness"]["expected_input_sha256"][
        "v60_result"
    ] = "0" * 64
    with pytest.raises(ValueError, match="input receipt mismatch"):
        run_decoupled_rank_state_harness(config)
    assert not (tmp_path / "packet" / "synthetic_checkpoint.pt").exists()


def test_v61_harness_rejects_extra_runtime_input(tmp_path: Path) -> None:
    config = _config(tmp_path)
    harness = config["decoupled_rank_state_harness"]
    harness["inputs"]["real_panel"] = "data/processed/unsafe.parquet"
    harness["expected_input_sha256"]["real_panel"] = "0" * 64
    with pytest.raises(ValueError, match="allowlist drift"):
        run_decoupled_rank_state_harness(config)
