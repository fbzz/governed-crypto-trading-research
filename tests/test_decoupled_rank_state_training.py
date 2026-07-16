from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_v63_config_and_contract_freeze_training_only_operator_sequence() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/v63_decoupled_rank_state_training.yaml").read_text()
    )
    contract = yaml.safe_load(
        (ROOT / "research/phase_contracts/v063.yaml").read_text()
    )
    training = config["decoupled_rank_state_training"]
    assert training["version"] == "v63"
    assert contract["runtime_contract"]["full_phase_order"] == [
        "doctor", "preflight", "smoke", "full", "verify", "replay"
    ]
    assert contract["grid_optimizer_and_runtime_contract"]["expected_jobs"] == 9
    assert contract["model_and_objective_contract"]["shared_parameters"] is False
    assert contract["model_and_objective_contract"]["combined_scalar_loss"] is False
    assert contract["target_contract"]["status"] == "sealed"
    assert "performance_metric_or_pnl" in contract["access_contract"][
        "forbidden_capabilities"
    ]
    assert set(training["inputs"].values()) == set(
        contract["access_contract"]["allowed_inputs"]
    )
