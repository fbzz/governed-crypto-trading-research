from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from tlm.research_workflow import validate_research_state


ROOT = Path(__file__).resolve().parents[1]


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def test_v71_owner_authorization_and_projection_incident_are_hash_bound() -> None:
    authorization = json.loads(
        (ROOT / "research/authorizations/v071_posthoc_retrospective_diagnostic.json")
        .read_text(encoding="utf-8")
    )
    incident = json.loads(
        (ROOT / "research/incidents/v071_prepare_schema_probe_projection_gap.json")
        .read_text(encoding="utf-8")
    )
    assert authorization.pop("authorization_sha256") == _canonical(authorization)
    assert incident.pop("incident_sha256") == _canonical(incident)
    assert incident["observed"]["rows_displayed"] == 2
    assert incident["observed"]["evaluation_window_rows_read"] == 0
    assert incident["observed"]["sealed_v64_outcome_packet_reads"] == 0
    assert incident["assessment"]["2025_evaluation_outcome_contamination"] is False


def test_v71_frozen_contract_requires_feature_only_projection_and_sealed_outcomes() -> None:
    status = validate_research_state(ROOT, "research/current.yaml")
    contract = yaml.safe_load(
        (ROOT / "research/phase_contracts/v071.yaml").read_text(encoding="utf-8")
    )
    assert status["passed"] is True
    assert status["authorized_phase"] == "v85"
    assert contract["stage_revision"] == (
        "v071_posthoc_consumed_2025_diagnostic_prepare_r2"
    )
    assert contract["projection_contract"]["post_anchor_forbidden_column_projection_allowed"] is False
    assert contract["sealed_outcome_contract"]["may_be_opened_during_v71_prepare"] is False
    assert len(contract["access_contract"]["allowed_inputs"]) == 25
    assert set(contract["access_contract"]["allowed_inputs"]) == set(
        contract["input_contract"]["expected_static_file_sha256_by_path"]
    )
