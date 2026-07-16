from __future__ import annotations

from pathlib import Path

import yaml

from tlm.low_turnover_rank_training import _context


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_config_and_contract_match() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/v83_low_turnover_rank_training.yaml").read_text()
    )
    contract = yaml.safe_load((ROOT / "research/phase_contracts/v083.yaml").read_text())
    assert config["low_turnover_rank_training"]["version"] == "v83"
    assert contract["stage_revision"] == (
        "v083_frozen_non_target_low_turnover_rank_training_r2"
    )
    assert contract["grid_optimizer_and_runtime_contract"]["expected_jobs"] == 9
    assert contract["model_and_objective_contract"]["architecture"][
        "expected_parameter_count"
    ] == 10993
    assert contract["target_contract"]["status"] == "sealed"
