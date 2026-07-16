from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tlm.core.artifacts import canonical_sha256
from tlm.v64_r2_retrospective_diagnostic import (
    BLIND_GATE_NAMES,
    _equal_weight_positions,
    _evaluation_spec,
    _turnover_matches,
    read_projected_parquet,
    resolve_prepare_anchor,
)


ROOT = Path(__file__).resolve().parents[1]


def _json(relative: str) -> dict:
    value = json.loads((ROOT / relative).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _yaml(relative: str) -> dict:
    value = yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_projected_reader_never_materializes_forbidden_panel_columns(
    tmp_path: Path,
) -> None:
    dates = pd.date_range("2025-01-01", periods=2, freq="D", tz="UTC")
    source = pd.DataFrame(
        {
            "date": [dates[0], dates[1], dates[0], dates[1]],
            "symbol": ["AUSDT", "AUSDT", "BUSDT", "BUSDT"],
            "log_close_to_close_return": [0.1, 0.2, 0.3, 0.4],
            "target_next_open_to_next_open_log_return": [9.0, 9.0, 9.0, 9.0],
            "raw_close": [100.0, 101.0, 200.0, 201.0],
        }
    )
    path = tmp_path / "panel.parquet"
    source.to_parquet(path, index=False)
    result = read_projected_parquet(
        path,
        ["date", "symbol", "log_close_to_close_return"],
        ["AUSDT", "BUSDT"],
        dates[0],
        dates[1],
    )
    assert list(result.columns) == [
        "date",
        "symbol",
        "log_close_to_close_return",
    ]
    assert "target_next_open_to_next_open_log_return" not in result
    assert "raw_close" not in result


def test_equal_weight_control_has_exact_turnover_and_final_liquidation() -> None:
    dates = pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC")
    availability = {
        dates[0]: ("AUSDT", "BUSDT"),
        dates[1]: ("AUSDT", "BUSDT", "CUSDT"),
        dates[2]: ("BUSDT", "CUSDT"),
    }
    result = _equal_weight_positions(
        1,
        ["AUSDT", "BUSDT", "CUSDT"],
        list(dates),
        availability,
        [10, 20, 30],
    )
    assert len(result) == 27
    assert _turnover_matches(result)
    daily = result.drop_duplicates(["date", "fold", "cost_bps"])
    assert np.allclose(daily["gross_exposure"], 1.0)
    assert set(daily.loc[daily["date"] == dates[-1], "final_liquidation_turnover"]) == {1.0}


def test_v71_registered_hash_covers_costs_controls_gates_and_blind_gates() -> None:
    context = {
        "contract": _yaml("research/phase_contracts/v071.yaml"),
        "v64_spec": _json(
            "artifacts/v64_decoupled_rank_state_evaluation/evaluation_spec.json"
        ),
        "blueprint": _json(
            "artifacts/v65_v64_r2_probabilistic_state_gate_spec/blueprint.json"
        ),
        "inputs": {
            "v64_control_positions": (
                "artifacts/v64_decoupled_rank_state_evaluation/positions.parquet"
            )
        },
    }
    spec, registered = _evaluation_spec(context)
    bound = {
        "cost_bps": registered["cost_bps"],
        "accounting": registered["accounting"],
        "controls": registered["controls"],
        "gates": registered["gates"],
        "outcome_blind_gate_names": registered["outcome_blind_gate_names"],
    }
    assert registered["sha256"] == canonical_sha256(bound)
    assert spec["registered_sha256"] == registered["sha256"]
    assert registered["outcome_blind_gate_names"] == list(BLIND_GATE_NAMES)
    assert spec["lifecycle"]["clean_holdout_prospective_deployable_or_target_claim"] is False


def test_v71_prepare_anchor_is_the_incident_registration_commit() -> None:
    contract = _yaml("research/phase_contracts/v071.yaml")
    anchor = resolve_prepare_anchor(
        ROOT, contract["prepare_registration_anchor_contract"]
    )
    assert len(anchor["commit"]) == 40
    assert anchor["incident_file_sha256"] == (
        contract["access_incident"]["file_sha256"]
    )
