from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest
import yaml


def _load_validator() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / ".agents"
        / "skills"
        / "tlm-training-operator"
        / "scripts"
        / "validate_training_packet.py"
    )
    spec = importlib.util.spec_from_file_location("tlm_training_packet_validator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_validator()


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


def _packet_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    (tmp_path / "research").mkdir()
    (tmp_path / "research" / "phase_contracts").mkdir()
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "src").mkdir()
    source_path = tmp_path / "src" / "training.py"
    source_path.write_text("# frozen\n", encoding="utf-8")
    operations = ["doctor", "smoke", "full", "verify", "replay"]
    phase_contract = {
        "phase": "v58",
        "operator_enforcement_contract": {"operation_order": operations},
        "grid_contract": {
            "origins": ["origin_2024", "origin_2025"],
            "geometries": ["expanding", "rolling"],
            "folds": [1, 2, 3],
            "seeds": [42, 7, 123],
            "expected_jobs": 36,
        },
        "runtime_contract": {
            "process_lock": "data/checkpoints/.v58_state_conditioned_multi_horizon_training.lock"
        },
    }
    phase_path = tmp_path / "research" / "phase_contracts" / "v058.yaml"
    phase_path.write_text(yaml.safe_dump(phase_contract), encoding="utf-8")
    phase_reference = {
        "path": "research/phase_contracts/v058.yaml",
        "file_sha256": _sha(phase_path),
    }
    state_path = tmp_path / "research" / "current.yaml"
    state_path.write_text(
        yaml.safe_dump(
            {"authorized_phase": "v58", "phase_contract": phase_reference}
        ),
        encoding="utf-8",
    )
    contract_value = {
        "phase_contract": phase_reference,
        "contract": phase_contract,
        "source_receipt_files": ["src/training.py"],
    }
    contract_path = tmp_path / "artifacts" / "training_spec.json"
    contract_path.write_text(json.dumps(contract_value), encoding="utf-8")
    access = {
        "outcome_rows_read": 0,
        "target_assets_loaded": [],
        "forbidden_columns_loaded": [],
        "predictions_written": False,
        "policy_actions_emitted": False,
        "performance_metrics_computed": False,
        "pnl_computed": False,
        "hyperparameters_changed": False,
    }
    data_access_path = tmp_path / "artifacts" / "data_access.json"
    data_access_path.write_text(json.dumps({"data_access": access}), encoding="utf-8")
    manifest = {
        "checkpoint_manifest": {
            "expected_jobs": _jobs(),
            "jobs": [],
            "selected_jobs": [],
        }
    }
    manifest_path = tmp_path / "artifacts" / "checkpoint_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    head = "a" * 40
    live_status = {
        "passed": True,
        "authorized_phase": "v58",
        "authorized_next_action": "authorize_v58_frozen_non_target_training_only",
        "authorized_command": "run-v58",
        "phase_contract_path": phase_reference["path"],
    }
    live_doctor = {
        "passed": True,
        "full_training_ready": True,
        "authorized_phase": "v58",
        "disk": {
            "free_bytes": 60 * 1024**3,
            "required_free_bytes": 50 * 1024**3,
        },
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
        "process_lock": {
            "path": str(
                tmp_path
                / "data"
                / "checkpoints"
                / ".v58_state_conditioned_multi_horizon_training.lock"
            ),
            "available": True,
            "active_job_count": 0,
        },
        "backup": {
            "passed": True,
            "receipt_sha256": "b" * 64,
            "objects_verified": 19,
            "code_backup_verified": True,
        },
    }
    monkeypatch.setattr(
        VALIDATOR,
        "workflow_json",
        lambda _root, command: live_status if command == "research-status" else live_doctor,
    )
    monkeypatch.setattr(
        VALIDATOR,
        "git",
        lambda _root, *args: head if args == ("rev-parse", "HEAD") else "",
    )
    files = {"src/training.py": _sha(source_path)}
    return {
        "schema_version": "tlm-training-operator/v1",
        "operation": "full",
        "research_state": {
            "path": "research/current.yaml",
            "sha256": _sha(state_path),
            "authorized_phase": "v58",
            "authorized_next_action": live_status["authorized_next_action"],
            "authorized_command": live_status["authorized_command"],
        },
        "contract": {
            "path": "artifacts/training_spec.json",
            "sha256": _sha(contract_path),
            "frozen": True,
            "authorized_operations": operations,
        },
        "source_receipt": {
            "git_clean": True,
            "git_head": head,
            "files": files,
            "bundle_sha256": _canonical(files),
        },
        "doctor": {
            "passed": True,
            "python_ok": True,
            "torch_ok": True,
            "mps_available": True,
            "mps_operational": True,
            "device": "mps",
            "dtype": "float32",
            "deterministic_algorithms": True,
            "fallback_enabled": False,
            "disk_free_bytes": 60 * 1024**3,
            "required_free_bytes": 50 * 1024**3,
            "active_job_count": 0,
            "process_lock_path": str(
                tmp_path
                / "data"
                / "checkpoints"
                / ".v58_state_conditioned_multi_horizon_training.lock"
            ),
            "backup_receipt_sha256": "b" * 64,
            "backup_objects_verified": 19,
            "code_backup_verified": True,
            "full_training_ready": True,
        },
        "data_access": access,
        "grid": {
            "expected_jobs": _jobs(),
            "completed_jobs": [],
            "active_jobs": [],
            "selected_jobs": [],
        },
        "resume": {
            "granularity": "epoch_boundary",
            "cross_job_resume_allowed": False,
            "active_resume_artifacts": [],
            "pending_resume_artifacts": [],
            "pending_resume_job": None,
            "orphan_resume_artifacts": [],
            "interrupted_resume_matched": False,
        },
        "verification": {
            "checkpoint_jobs_verified": [],
            "all_checkpoints_retained": False,
            "checkpoint_roundtrip_passed": False,
        },
        "replay": {
            "new_jobs": 0,
            "new_optimizer_steps": 0,
            "artifact_hashes_match": False,
        },
        "evidence": {
            "data_access": {
                "path": "artifacts/data_access.json",
                "sha256": _sha(data_access_path),
            },
            "checkpoint_manifest": {
                "path": "artifacts/checkpoint_manifest.json",
                "sha256": _sha(manifest_path),
            },
        },
    }


def test_v58_full_packet_binds_empty_preflight_evidence_and_exact_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = _packet_fixture(tmp_path, monkeypatch)
    result = VALIDATOR.validate_packet(tmp_path, packet)
    assert result["valid"] is True
    assert result["expected_jobs"] == 36
    assert result["completed_jobs"] == 0


def test_v58_packet_rejects_grid_not_derived_from_frozen_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = _packet_fixture(tmp_path, monkeypatch)
    packet["grid"]["expected_jobs"] = packet["grid"]["expected_jobs"][:-1]
    with pytest.raises(VALIDATOR.ValidationError, match="exact frozen 36-job grid"):
        VALIDATOR.validate_packet(tmp_path, packet)


def test_v58_training_spec_must_project_live_phase_contract_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = _packet_fixture(tmp_path, monkeypatch)
    spec_path = tmp_path / packet["contract"]["path"]
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["contract"]["runtime_contract"]["process_lock"] = "wrong.lock"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    packet["contract"]["sha256"] = _sha(spec_path)
    with pytest.raises(VALIDATOR.ValidationError, match="exact projection"):
        VALIDATOR.validate_packet(tmp_path, packet)


def test_v58_source_receipt_file_set_is_frozen_by_training_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = _packet_fixture(tmp_path, monkeypatch)
    extra = tmp_path / "src" / "unregistered.py"
    extra.write_text("# extra\n", encoding="utf-8")
    packet["source_receipt"]["files"]["src/unregistered.py"] = _sha(extra)
    packet["source_receipt"]["bundle_sha256"] = _canonical(
        packet["source_receipt"]["files"]
    )
    with pytest.raises(VALIDATOR.ValidationError, match="file set differs"):
        VALIDATOR.validate_packet(tmp_path, packet)


def test_pending_resume_is_bound_to_incomplete_job_without_active_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = _packet_fixture(tmp_path, monkeypatch)
    packet["operation"] = "doctor"
    packet["evidence"] = {}
    resume_path = tmp_path / "data" / "checkpoints" / "v58" / "resume.pt"
    resume_path.parent.mkdir(parents=True)
    resume_path.write_bytes(b"resume")
    packet["resume"]["pending_resume_artifacts"] = [
        "data/checkpoints/v58/resume.pt"
    ]
    packet["resume"]["pending_resume_job"] = _jobs()[0]
    result = VALIDATOR.validate_packet(tmp_path, packet)
    assert result["active_jobs"] == 0

    invalid = deepcopy(packet)
    invalid["resume"]["pending_resume_job"] = "outside|grid|1|42"
    with pytest.raises(VALIDATOR.ValidationError, match="not bound"):
        VALIDATOR.validate_packet(tmp_path, invalid)


def test_smoke_binds_interrupted_resume_and_data_access_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = _packet_fixture(tmp_path, monkeypatch)
    packet["operation"] = "smoke"
    packet["resume"]["interrupted_resume_matched"] = True
    smoke_path = tmp_path / "artifacts" / "smoke.json"
    smoke_path.write_text(
        json.dumps({"smoke": {"interrupted_resume_matched": True}}),
        encoding="utf-8",
    )
    packet["evidence"] = {
        "smoke": {
            "path": "artifacts/smoke.json",
            "sha256": _sha(smoke_path),
        },
        "data_access": packet["evidence"]["data_access"],
    }
    assert VALIDATOR.validate_packet(tmp_path, packet)["valid"] is True

    missing_access = deepcopy(packet)
    del missing_access["evidence"]["data_access"]
    with pytest.raises(VALIDATOR.ValidationError, match="missing bound evidence"):
        VALIDATOR.validate_packet(tmp_path, missing_access)


def test_verify_requires_bound_external_checkpoint_backup_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = _packet_fixture(tmp_path, monkeypatch)
    packet["operation"] = "verify"
    packet["grid"]["completed_jobs"] = _jobs()
    packet["verification"] = {
        "checkpoint_jobs_verified": _jobs(),
        "all_checkpoints_retained": True,
        "checkpoint_roundtrip_passed": True,
    }
    with pytest.raises(VALIDATOR.ValidationError, match="missing bound evidence"):
        VALIDATOR.validate_packet(tmp_path, packet)


def test_checkpoint_backup_receipt_verifies_all_36_external_copies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    backup_root = tmp_path / "external"
    root.mkdir()
    backup_root.mkdir()
    entries = []
    for index, job_id in enumerate(_jobs()):
        source = root / "checkpoints" / f"job_{index}.pt"
        copied = backup_root / "checkpoints" / f"job_{index}.pt"
        source.parent.mkdir(parents=True, exist_ok=True)
        copied.parent.mkdir(parents=True, exist_ok=True)
        payload = f"checkpoint-{index}".encode()
        source.write_bytes(payload)
        copied.write_bytes(payload)
        entries.append(
            {
                "job_id": job_id,
                "source_path": f"checkpoints/job_{index}.pt",
                "backup_path": f"checkpoints/job_{index}.pt",
                "sha256": _sha(source),
                "size_bytes": len(payload),
            }
        )
    manifest_body = {
        "version": "v58_checkpoint_manifest_v1",
        "expected_jobs": _jobs(),
        "jobs": [],
        "selected_jobs": [],
        "active_jobs": [],
        "checkpoint_count": 36,
    }
    manifest_body["manifest_sha256"] = _canonical(manifest_body)
    manifest_source = (
        root
        / "artifacts"
        / "v58_state_conditioned_multi_horizon_training"
        / "checkpoint_manifest.json"
    )
    manifest_backup = backup_root / "checkpoints" / "checkpoint_manifest.json"
    manifest_source.parent.mkdir(parents=True, exist_ok=True)
    manifest_backup.parent.mkdir(parents=True, exist_ok=True)
    manifest_payload = (
        json.dumps(manifest_body, indent=2, sort_keys=True) + "\n"
    ).encode()
    manifest_source.write_bytes(manifest_payload)
    manifest_backup.write_bytes(manifest_payload)
    original_stat = Path.stat

    def fake_stat(path: Path, *args: object, **kwargs: object):
        value = original_stat(path, *args, **kwargs)
        resolved = path.resolve()
        device = 10 if resolved == root.resolve() else 20 if resolved == backup_root.resolve() else value.st_dev
        values = list(value)
        values[2] = device
        return type(value)(values)

    monkeypatch.setattr(Path, "stat", fake_stat)
    receipt = {
        "schema_version": "tlm-checkpoint-backup-receipt/v1",
        "phase": "v58",
        "backup_root": str(backup_root),
        "source_device": 10,
        "backup_device": 20,
        "different_device": True,
        "verified": True,
        "checkpoint_manifest": {
            "source_path": (
                "artifacts/v58_state_conditioned_multi_horizon_training/"
                "checkpoint_manifest.json"
            ),
            "backup_path": "checkpoints/checkpoint_manifest.json",
            "sha256": _sha(manifest_source),
            "size_bytes": len(manifest_payload),
            "registered_manifest_sha256": manifest_body["manifest_sha256"],
        },
        "checkpoints": entries,
    }
    VALIDATOR.validate_checkpoint_backup(root, receipt, _jobs())

    manifest_backup.write_bytes(manifest_payload[:-1])
    with pytest.raises(VALIDATOR.ValidationError, match="manifest backup size drift"):
        VALIDATOR.validate_checkpoint_backup(root, receipt, _jobs())
    manifest_backup.write_bytes(manifest_payload)

    same_size_drift = b" " + manifest_payload[1:]
    manifest_backup.write_bytes(same_size_drift)
    with pytest.raises(VALIDATOR.ValidationError, match="manifest backup hash drift"):
        VALIDATOR.validate_checkpoint_backup(root, receipt, _jobs())
    manifest_backup.write_bytes(manifest_payload)

    manifest_source.write_bytes(same_size_drift)
    with pytest.raises(VALIDATOR.ValidationError, match="manifest backup hash drift"):
        VALIDATOR.validate_checkpoint_backup(root, receipt, _jobs())
    manifest_source.write_bytes(manifest_payload)

    receipt["checkpoints"][0]["sha256"] = "0" * 64
    with pytest.raises(VALIDATOR.ValidationError, match="hash drift"):
        VALIDATOR.validate_checkpoint_backup(root, receipt, _jobs())


def test_checkpoint_backup_receipt_requires_manifest_copy_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    backup_root = tmp_path / "external"
    root.mkdir()
    backup_root.mkdir()

    original_stat = Path.stat

    def fake_stat(path: Path, *args: object, **kwargs: object):
        value = original_stat(path, *args, **kwargs)
        resolved = path.resolve()
        device = (
            10
            if resolved == root.resolve()
            else 20
            if resolved == backup_root.resolve()
            else value.st_dev
        )
        values = list(value)
        values[2] = device
        return type(value)(values)

    monkeypatch.setattr(Path, "stat", fake_stat)
    receipt = {
        "schema_version": "tlm-checkpoint-backup-receipt/v1",
        "phase": "v58",
        "backup_root": str(backup_root),
        "source_device": 10,
        "backup_device": 20,
        "different_device": True,
        "verified": True,
        "checkpoints": [],
    }

    with pytest.raises(VALIDATOR.ValidationError, match="manifest"):
        VALIDATOR.validate_checkpoint_backup(root, receipt, _jobs())
