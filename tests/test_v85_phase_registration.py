from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from tlm.research_workflow import validate_research_state


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_v85_is_the_only_live_authorized_phase() -> None:
    status = validate_research_state(ROOT)
    assert status["passed"] is True
    assert status["authorized_phase"] == "v85"
    assert status["authorized_next_action"] == (
        "execute_v85_exactly_one_registered_non_target_outcome_unseal_and_complete_evaluation"
    )
    assert status["target_asset_status"] == "sealed"


def test_current_state_binds_the_exact_v85_contract() -> None:
    current = yaml.safe_load((ROOT / "research/current.yaml").read_text(encoding="utf-8"))
    reference = current["phase_contract"]
    assert reference["path"] == "research/phase_contracts/v085.yaml"
    assert _sha256(ROOT / reference["path"]) == reference["file_sha256"]
    assert current["deployable_strategy"] is False
    assert current["safety"]["v85_maximum_unseal_count"] == 1
    assert current["safety"]["v85_target_assets_remain_sealed"] is True


def test_v85_contract_freezes_exact_source_and_replay_counts() -> None:
    contract = yaml.safe_load(
        (ROOT / "research/phase_contracts/v085.yaml").read_text(encoding="utf-8")
    )
    outcome = contract["outcome_access_contract"]
    one_shot = contract["one_shot_contract"]
    assert outcome["expected_rows"] == 5370
    assert outcome["maximum_source_packet_deserializations"] == 1
    assert outcome["replay_source_packet_deserializations"] == 0
    assert one_shot["maximum_unseal_count"] == 1
    assert one_shot["source_outcome_rows_read_first_execution"] == 5370
    assert one_shot["source_outcome_rows_read_replay"] == 0
