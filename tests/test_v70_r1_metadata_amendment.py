from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import yaml

from tlm.research_workflow import validate_research_state


ROOT = Path(__file__).resolve().parents[1]


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _json(path: str) -> dict:
    value = json.loads((ROOT / path).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _yaml(path: str) -> dict:
    value = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_v70_r1_receipts_are_self_hash_valid_and_metadata_only() -> None:
    authorization = _json(
        "research/authorizations/v070_r1_ranker_scale_amendment.json"
    )
    incident = _json("research/incidents/v070_ranker_scale_allowlist_gap.json")
    receipt = _json("research/receipts/v070_ranker_excess_scale_receipt.json")

    assert authorization.pop("authorization_sha256") == _canonical_sha256(
        authorization
    )
    assert incident.pop("incident_sha256") == _canonical_sha256(incident)
    assert receipt.pop("receipt_sha256") == _canonical_sha256(receipt)
    assert incident["scientific_evidence_admitted_under_original_anchor"] is False
    assert incident["impact"] == {
        "checkpoints_mutated": 0,
        "market_data_rows_read": 0,
        "outcome_rows_read": 0,
        "performance_metrics_computed": 0,
        "pnl_computations": 0,
        "positions_frozen": 0,
        "predictions_frozen": 0,
        "target_assets_loaded": [],
        "training_or_refit_performed": False,
    }
    assert receipt["training_or_refit_performed"] is False
    assert receipt["target_assets_loaded"] == []


def test_v70_r1_scale_projection_matches_v63_and_v68_identities() -> None:
    source = _json("artifacts/v63_decoupled_rank_state_training/scaler_manifest.json")
    v68 = _json(
        "artifacts/v68_v64_r2_probabilistic_state_gate_training/scaler_manifest.json"
    )
    receipt = _json("research/receipts/v070_ranker_excess_scale_receipt.json")
    source_by_fold = {int(row["fold"]): row for row in source["folds"]}
    v68_by_fold = {int(row["fold"]): row for row in v68["folds"]}

    assert [int(row["fold"]) for row in receipt["folds"]] == [1, 2, 3]
    for row in receipt["folds"]:
        fold = int(row["fold"])
        assert row["ranker_excess_rms"] == source_by_fold[fold][
            "ranker_excess_rms"
        ]
        assert row["source_fold_scale_sha256"] == source_by_fold[fold][
            "fold_scale_sha256"
        ]
        assert row["source_fold_scale_sha256"] == v68_by_fold[fold][
            "source_v63_fold_scale_sha256"
        ]
        assert row["feature_scaler_state_sha256"] == v68_by_fold[fold][
            "feature_scaler_state_sha256"
        ]


def test_v70_r1_changes_no_scientific_policy_and_is_the_only_anchor() -> None:
    amendment = _yaml("research/amendments/v070_r1_metadata_only.yaml")
    contract = _yaml("research/phase_contracts/v070.yaml")
    protocol = _json(
        "artifacts/v69_v64_r2_prospective_confirmation_prepare/protocol.json"
    )
    blueprint = _json(
        "artifacts/v65_v64_r2_probabilistic_state_gate_spec/blueprint.json"
    )

    assert all(value is False for value in amendment["scientific_contract"].values())
    assert protocol["policy"] == blueprint["policy"]
    assert contract["stage_revision"] == (
        "v070_prospective_non_target_capture_prediction_freeze_r2"
    )
    assert contract["registration_anchor_contract"] == {
        "amendment_path": "research/amendments/v070_r1_metadata_only.yaml",
        "amendment_file_sha256": hashlib.sha256(
            (ROOT / "research/amendments/v070_r1_metadata_only.yaml").read_bytes()
        ).hexdigest(),
        "resolver": "first_ancestor_commit_containing_exact_amendment_file_sha256",
        "commit_timestamp_timezone": "UTC",
        "first_admissible_feature_close": (
            "strictly_after_registration_commit_timestamp"
        ),
        "pre_registration_lookback_allowed_as_features": True,
        "pre_registration_signal_position_or_scored_outcome_allowed": False,
    }
    assert len(contract["access_contract"]["allowed_inputs"]) == 26
    assert set(contract["access_contract"]["allowed_inputs"]) == set(
        contract["input_contract"]["expected_static_file_sha256_by_path"]
    )


def test_v70_r1_anchor_remains_registered_after_v82_dataset_registration() -> None:
    status = validate_research_state(ROOT, "research/current.yaml")
    assert status["passed"] is True
    assert status["authorized_phase"] == "v85"

    if subprocess.run(
        ["git", "status", "--porcelain", "--", "research/amendments/v070_r1_metadata_only.yaml"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip():
        return
    contract = _yaml("research/phase_contracts/v070.yaml")
    anchor = contract["registration_anchor_contract"]
    commits = subprocess.run(
        ["git", "rev-list", "--reverse", "HEAD", "--", anchor["amendment_path"]],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    assert commits
    content = subprocess.run(
        ["git", "show", f"{commits[0]}:{anchor['amendment_path']}"],
        cwd=ROOT,
        capture_output=True,
        check=True,
    ).stdout
    assert hashlib.sha256(content).hexdigest() == anchor["amendment_file_sha256"]
