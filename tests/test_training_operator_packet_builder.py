from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
import yaml


def _load_builder() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / ".agents"
        / "skills"
        / "tlm-training-operator"
        / "scripts"
        / "build_training_packet.py"
    )
    spec = importlib.util.spec_from_file_location("tlm_training_packet_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _load_builder()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _jobs() -> list[str]:
    return [
        f"{origin}|{geometry}|{fold}|{seed}"
        for origin in ["origin_2024", "origin_2025"]
        for geometry in ["expanding", "rolling"]
        for fold in [1, 2, 3]
        for seed in [42, 7, 123]
    ]


def _write_json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> dict:
    root = tmp_path / "repo"
    (root / "research" / "phase_contracts").mkdir(parents=True)
    (root / "artifacts").mkdir()
    source_file = root / "src" / "training.py"
    source_file.parent.mkdir()
    source_file.write_text("# frozen\n", encoding="utf-8")
    operations = ["doctor", "smoke", "full", "verify", "replay"]
    phase = {
        "phase": "v58",
        "operator_enforcement_contract": {"operation_order": operations},
        "grid_contract": {
            "origins": ["origin_2024", "origin_2025"],
            "geometries": ["expanding", "rolling"],
            "folds": [1, 2, 3],
            "seeds": [42, 7, 123],
            "expected_jobs": 36,
        },
        "checkpoint_contract": {"cross_job_resume_allowed": False},
        "runtime_contract": {
            "process_lock": "data/checkpoints/.v58_state_conditioned_multi_horizon_training.lock"
        },
        "data_access_ledger_contract": {
            "development_evaluation_outcome_rows_read": 0,
            "target_assets_loaded": [],
            "forbidden_columns_loaded": [],
            "predictions_written": False,
            "policy_actions_emitted": False,
            "performance_metrics_computed": False,
            "pnl_computed": False,
            "hyperparameters_changed": False,
        },
    }
    phase_path = root / "research" / "phase_contracts" / "v058.yaml"
    phase_path.write_text(yaml.safe_dump(phase), encoding="utf-8")
    phase_reference = {
        "path": "research/phase_contracts/v058.yaml",
        "file_sha256": _sha(phase_path),
    }
    state = {"authorized_phase": "v58", "phase_contract": phase_reference}
    state_path = root / "research" / "current.yaml"
    state_path.write_text(yaml.safe_dump(state), encoding="utf-8")
    spec_path = _write_json(
        root / "artifacts" / "training_spec.json",
        {
            "phase_contract": phase_reference,
            "contract": phase,
            "source_receipt_files": ["src/training.py"],
        },
    )
    files = {"src/training.py": _sha(source_file)}
    source_receipt_path = _write_json(
        root / "artifacts" / "source_receipt.json",
        {
            "source_receipt": {
                "git": {"clean": True, "head": "a" * 40},
                "files": files,
                "bundle_sha256": _canonical(files),
            }
        },
    )
    data_access = {
        "development_evaluation_outcome_rows_read": 0,
        "outcome_rows_read": 0,
        "target_assets_loaded": [],
        "forbidden_columns_loaded": [],
        "predictions_written": False,
        "policy_actions_emitted": False,
        "performance_metrics_computed": False,
        "pnl_computed": False,
        "hyperparameters_changed": False,
    }
    access_path = _write_json(
        root / "artifacts" / "data_access.json", {"data_access": data_access}
    )
    manifest_path = _write_json(
        root / "artifacts" / "checkpoint_manifest.json",
        {
            "checkpoint_manifest": {
                "expected_jobs": _jobs(),
                "jobs": [
                    {"job_id": _jobs()[0], "status": "completed"},
                    {"job_id": _jobs()[1], "status": "pending"},
                ],
                "selected_jobs": [],
                "resume": {
                    "pending_resume_artifacts": [],
                    "pending_resume_job": None,
                    "active_resume_artifacts": [],
                    "orphan_resume_artifacts": [],
                },
            }
        },
    )
    status = {
        "passed": True,
        "authorized_phase": "v58",
        "authorized_next_action": "authorize_v58_frozen_non_target_training_only",
        "authorized_command": "run-v58",
        "phase_contract_path": phase_reference["path"],
    }
    doctor = {
        "passed": True,
        "full_training_ready": True,
        "runtime": {
            "python_ok": True,
            "torch_ok": True,
            "mps_available": True,
            "mps_operational": True,
            "device": "mps",
            "dtype": "float32",
            "deterministic_algorithms": True,
            "fallback_enabled": False,
        },
        "disk": {
            "free_bytes": 60 * 1024**3,
            "required_free_bytes": 50 * 1024**3,
        },
        "process_lock": {
            "active_job_count": 0,
            "available": True,
            "path": str(
                root
                / "data"
                / "checkpoints"
                / ".v58_state_conditioned_multi_horizon_training.lock"
            ),
        },
        "backup": {
            "receipt_sha256": "b" * 64,
            "objects_verified": 19,
            "code_backup_verified": True,
        },
    }
    return {
        "root": root,
        "spec": spec_path,
        "source": source_receipt_path,
        "access": access_path,
        "manifest": manifest_path,
        "status": status,
        "doctor": doctor,
    }


def _build(fixture: dict, operation: str, evidence: dict[str, Path] | None = None):
    return BUILDER.build_packet(
        repo_root=fixture["root"],
        operation=operation,
        training_spec_path=fixture["spec"].relative_to(fixture["root"]),
        source_receipt_path=fixture["source"].relative_to(fixture["root"]),
        evidence_paths={
            name: path.relative_to(fixture["root"])
            for name, path in (evidence or {}).items()
        },
        live_status=fixture["status"],
        live_doctor=fixture["doctor"],
    )


def test_doctor_packet_derives_grid_source_access_and_enriched_doctor(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    packet = _build(fixture, "doctor")
    assert packet["grid"]["expected_jobs"] == _jobs()
    assert packet["grid"]["completed_jobs"] == []
    assert packet["source_receipt"]["git_head"] == "a" * 40
    assert packet["data_access"]["outcome_rows_read"] == 0
    assert packet["doctor"]["mps_operational"] is True
    assert packet["doctor"]["backup_receipt_sha256"] == "b" * 64
    assert packet["doctor"]["required_free_bytes"] == 50 * 1024**3


def test_full_packet_derives_completed_grid_and_wrapped_access_evidence(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    packet = _build(
        fixture,
        "full",
        {
            "data_access": fixture["access"],
            "checkpoint_manifest": fixture["manifest"],
        },
    )
    assert packet["grid"]["completed_jobs"] == [_jobs()[0]]
    assert packet["grid"]["active_jobs"] == []
    assert set(packet["evidence"]) == {"data_access", "checkpoint_manifest"}
    assert packet["evidence"]["data_access"]["sha256"] == _sha(
        fixture["access"]
    )


def test_smoke_builder_never_invents_interrupted_resume_success(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    failed_smoke = _write_json(
        fixture["root"] / "artifacts" / "smoke.json",
        {"smoke": {"interrupted_resume_matched": False}},
    )
    with pytest.raises(BUILDER.PacketBuildError, match="does not prove"):
        _build(
            fixture,
            "smoke",
            {"smoke": failed_smoke, "data_access": fixture["access"]},
        )

    passed_smoke = _write_json(
        failed_smoke,
        {"smoke": {"interrupted_resume_matched": True}},
    )
    packet = _build(
        fixture,
        "smoke",
        {"smoke": passed_smoke, "data_access": fixture["access"]},
    )
    assert packet["resume"]["interrupted_resume_matched"] is True


def test_replay_packet_copies_verification_and_replay_without_synthesis(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    complete_manifest = _write_json(
        fixture["manifest"],
        {
            "checkpoint_manifest": {
                "expected_jobs": _jobs(),
                "jobs": [{"job_id": job, "status": "completed"} for job in _jobs()],
                "selected_jobs": [],
            }
        },
    )
    verification = _write_json(
        fixture["root"] / "artifacts" / "verification.json",
        {
            "verification": {
                "checkpoint_jobs_verified": _jobs(),
                "all_checkpoints_retained": True,
                "checkpoint_roundtrip_passed": True,
            }
        },
    )
    replay = _write_json(
        fixture["root"] / "artifacts" / "replay.json",
        {
            "replay": {
                "new_jobs": 0,
                "new_optimizer_steps": 0,
                "artifact_hashes_match": True,
            }
        },
    )
    checkpoint_backup = _write_json(
        fixture["root"] / "artifacts" / "checkpoint_backup_receipt.json",
        {"schema_version": "synthetic-test-placeholder"},
    )
    packet = _build(
        fixture,
        "replay",
        {
            "data_access": fixture["access"],
            "checkpoint_manifest": complete_manifest,
            "verification": verification,
            "replay": replay,
            "checkpoint_backup": checkpoint_backup,
        },
    )
    assert packet["grid"]["completed_jobs"] == _jobs()
    assert packet["verification"]["checkpoint_jobs_verified"] == _jobs()
    assert packet["verification"]["all_checkpoints_retained"] is True
    assert packet["replay"] == {
        "new_jobs": 0,
        "new_optimizer_steps": 0,
        "artifact_hashes_match": True,
    }


def test_atomic_write_self_validates_and_removes_invalid_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path)
    packet = _build(fixture, "doctor")
    output = Path("artifacts/operator_packet_doctor.json")
    monkeypatch.setattr(
        BUILDER.VALIDATOR,
        "validate_packet",
        lambda root, value: {"valid": True, "operation": value["operation"]},
    )
    result = BUILDER.write_and_validate_packet(fixture["root"], output, packet)
    assert result == {"valid": True, "operation": "doctor"}
    assert json.loads((fixture["root"] / output).read_text()) == packet

    monkeypatch.setattr(
        BUILDER.VALIDATOR,
        "validate_packet",
        lambda *_: (_ for _ in ()).throw(BUILDER.VALIDATOR.ValidationError("bad")),
    )
    invalid_output = Path("artifacts/invalid.json")
    with pytest.raises(BUILDER.VALIDATOR.ValidationError, match="bad"):
        BUILDER.write_and_validate_packet(
            fixture["root"], invalid_output, packet
        )
    assert not (fixture["root"] / invalid_output).exists()


def test_owner_waiver_builder_requires_bound_policy_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    phase_path = fixture["root"] / "research/phase_contracts/v058.yaml"
    phase = yaml.safe_load(phase_path.read_text(encoding="utf-8"))
    waiver_ref = {
        "path": "research/waivers/v058r1.json",
        "file_sha256": "c" * 64,
    }
    scopes = [
        "external_input_copy",
        "external_code_copy",
        "external_checkpoint_copy",
    ]
    phase["runtime_contract"].update(
        {
            "external_backup_receipt_required": False,
            "backup_policy": {
                "mode": "owner_waiver",
                "waiver": waiver_ref,
                "waived_safeguards": scopes,
            },
        }
    )
    phase_path.write_text(yaml.safe_dump(phase), encoding="utf-8")
    phase_ref = {
        "path": "research/phase_contracts/v058.yaml",
        "file_sha256": _sha(phase_path),
    }
    state_path = fixture["root"] / "research/current.yaml"
    state_path.write_text(
        yaml.safe_dump({"authorized_phase": "v58", "phase_contract": phase_ref}),
        encoding="utf-8",
    )
    spec = json.loads(fixture["spec"].read_text(encoding="utf-8"))
    spec["phase_contract"] = phase_ref
    spec["contract"] = phase
    fixture["spec"].write_text(json.dumps(spec), encoding="utf-8")
    fixture["status"]["phase_contract_path"] = phase_ref["path"]
    fixture["doctor"]["backup"] = {
        "mode": "owner_waiver",
        "required": False,
        "passed": True,
        "waiver_path": waiver_ref["path"],
        "waiver_sha256": waiver_ref["file_sha256"],
        "waiver_verified": True,
        "objects_verified": 0,
        "code_backup_verified": False,
    }
    fixture["doctor"]["authorized_phase"] = "v58"
    complete_manifest = _write_json(
        fixture["manifest"],
        {
            "checkpoint_manifest": {
                "expected_jobs": _jobs(),
                "jobs": [{"job_id": job, "status": "completed"} for job in _jobs()],
                "selected_jobs": [],
            }
        },
    )
    verification = _write_json(
        fixture["root"] / "artifacts/verification.json",
        {
            "verification": {
                "checkpoint_jobs_verified": _jobs(),
                "all_checkpoints_retained": True,
                "checkpoint_roundtrip_passed": True,
            }
        },
    )
    policy_body = {
        "version": "v58_backup_policy_receipt_v1",
        "phase": "v58",
        "mode": "owner_waiver",
        "verified": True,
        "waiver_path": waiver_ref["path"],
        "waiver_sha256": waiver_ref["file_sha256"],
        "waived_safeguards": scopes,
        "external_input_backup_created": False,
        "external_code_backup_created": False,
        "external_checkpoint_backup_created": False,
    }
    policy = {**policy_body, "policy_receipt_sha256": _canonical(policy_body)}
    policy_path = _write_json(
        fixture["root"] / "artifacts/backup_policy_receipt.json", policy
    )
    packet = _build(
        fixture,
        "verify",
        {
            "data_access": fixture["access"],
            "checkpoint_manifest": complete_manifest,
            "verification": verification,
            "backup_policy": policy_path,
        },
    )
    with pytest.raises(BUILDER.PacketBuildError, match="contradicts"):
        _build(
            fixture,
            "verify",
            {
                "data_access": fixture["access"],
                "checkpoint_manifest": complete_manifest,
                "verification": verification,
                "backup_policy": policy_path,
                "checkpoint_backup": policy_path,
            },
        )
    assert packet["doctor"]["backup_mode"] == "owner_waiver"
    assert packet["doctor"]["backup_waiver_verified"] is True
    assert set(packet["evidence"]) == {
        "data_access",
        "checkpoint_manifest",
        "verification",
        "backup_policy",
    }
    BUILDER.VALIDATOR.validate_backup_policy_receipt(policy, phase)

    monkeypatch.setattr(
        BUILDER.VALIDATOR,
        "workflow_json",
        lambda _root, command: (
            fixture["status"]
            if command == "research-status"
            else fixture["doctor"]
        ),
    )
    monkeypatch.setattr(
        BUILDER.VALIDATOR,
        "git",
        lambda _root, *args: "a" * 40 if args == ("rev-parse", "HEAD") else "",
    )
    assert BUILDER.VALIDATOR.validate_packet(fixture["root"], packet)["valid"] is True

    contradictory_packet = deepcopy(packet)
    contradictory_packet["evidence"]["checkpoint_backup"] = {
        "path": policy_path.relative_to(fixture["root"]).as_posix(),
        "sha256": _sha(policy_path),
    }
    with pytest.raises(BUILDER.VALIDATOR.ValidationError, match="contradicts"):
        BUILDER.VALIDATOR.validate_packet(
            fixture["root"], contradictory_packet
        )

    copied_live = deepcopy(fixture["doctor"])
    copied_live["backup"]["code_backup_verified"] = True
    monkeypatch.setattr(
        BUILDER.VALIDATOR,
        "workflow_json",
        lambda _root, command: (
            fixture["status"] if command == "research-status" else copied_live
        ),
    )
    with pytest.raises(BUILDER.VALIDATOR.ValidationError, match="must not claim"):
        BUILDER.VALIDATOR.validate_packet(fixture["root"], packet)

    wrong_hash_packet = deepcopy(packet)
    wrong_hash_packet["doctor"]["backup_waiver_sha256"] = "d" * 64
    wrong_hash_live = deepcopy(fixture["doctor"])
    wrong_hash_live["backup"]["waiver_sha256"] = "d" * 64
    monkeypatch.setattr(
        BUILDER.VALIDATOR,
        "workflow_json",
        lambda _root, command: (
            fixture["status"] if command == "research-status" else wrong_hash_live
        ),
    )
    with pytest.raises(BUILDER.VALIDATOR.ValidationError, match="frozen contract"):
        BUILDER.VALIDATOR.validate_packet(fixture["root"], wrong_hash_packet)

    policy["external_checkpoint_backup_created"] = True
    policy_body = dict(policy)
    policy_body.pop("policy_receipt_sha256")
    policy["policy_receipt_sha256"] = _canonical(policy_body)
    with pytest.raises(BUILDER.VALIDATOR.ValidationError, match="must not claim"):
        BUILDER.VALIDATOR.validate_backup_policy_receipt(policy, phase)

    policy["external_checkpoint_backup_created"] = False
    policy["contradictory_claim"] = True
    policy_body = dict(policy)
    policy_body.pop("policy_receipt_sha256")
    policy["policy_receipt_sha256"] = _canonical(policy_body)
    with pytest.raises(BUILDER.VALIDATOR.ValidationError, match="key set drift"):
        BUILDER.VALIDATOR.validate_backup_policy_receipt(policy, phase)
