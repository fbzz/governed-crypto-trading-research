from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_v68_config_contract_and_operator_sequence_are_frozen() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/v68_v64_r2_probabilistic_state_gate_training.yaml").read_text()
    )
    contract = yaml.safe_load((ROOT / "research/phase_contracts/v068.yaml").read_text())
    training = config["v64_r2_probabilistic_state_gate_training"]
    assert training["version"] == "v68"
    assert set(training["inputs"].values()) == set(contract["access_contract"]["allowed_inputs"])
    assert contract["operator_enforcement_contract"]["operation_order"] == [
        "doctor", "smoke", "full", "verify", "replay"
    ]
    assert contract["grid_optimizer_and_runtime_contract"]["expected_jobs"] == 9
    assert contract["checkpoint_contract"]["cross_job_resume_allowed"] is False
    assert all((ROOT / path).is_file() for path in training["source_receipt_files"])

