from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import ModuleType

import pytest
import yaml


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1]
BACKUP = _load_module(
    ROOT
    / ".agents"
    / "skills"
    / "tlm-training-operator"
    / "scripts"
    / "create_v58_backup.py",
    "create_v58_backup",
)
VALIDATOR = _load_module(
    ROOT
    / ".agents"
    / "skills"
    / "tlm-training-operator"
    / "scripts"
    / "validate_training_packet.py",
    "validate_v58_backup_receipt",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def _repo_fixture(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    external = tmp_path / "external"
    repo.mkdir()
    external.mkdir()
    expected: dict[str, str] = {}
    for index, (relative, key) in enumerate(
        zip(BACKUP.EXPECTED_ALLOWED_INPUTS, BACKUP.EXPECTED_INPUT_KEYS, strict=True)
    ):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"v58-input-{index}-{key}\n".encode())
        expected[key] = _sha(path)
    contract = {
        "schema_version": 1,
        "phase": "v58",
        "family_id": "tlm_state_conditioned_multi_horizon_quantile_small_v1",
        "authorized_next_action": "authorize_v58_frozen_non_target_training_only",
        "access_contract": {
            "allowed_inputs": list(BACKUP.EXPECTED_ALLOWED_INPUTS),
            "checkpoint_dir": "data/checkpoints/v58_state_conditioned_multi_horizon_training",
        },
        "input_contract": {"expected_sha256": expected},
    }
    contract_path = repo / BACKUP.PHASE_CONTRACT
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    contract_path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    current_path = repo / BACKUP.CURRENT_STATE
    current_path.parent.mkdir(parents=True, exist_ok=True)
    current_path.write_text(
        yaml.safe_dump(
            {
                "authorized_phase": "v58",
                "phase_contract": {
                    "path": BACKUP.PHASE_CONTRACT.as_posix(),
                    "file_sha256": _sha(contract_path),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (repo / ".gitignore").write_text(
        "research/backups/*.yaml\n"
        "artifacts/v58_state_conditioned_multi_horizon_training/\n"
        "data/checkpoints/\n",
        encoding="utf-8",
    )
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "TLM Test")
    _git(repo, "config", "user.email", "tlm-test@example.invalid")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "fixture")
    assert _git(repo, "status", "--porcelain", "--untracked-files=all") == ""
    return repo, external


def _different_devices(
    monkeypatch: pytest.MonkeyPatch, repo: Path, external: Path
) -> None:
    def device(path: Path) -> int:
        resolved = path.resolve()
        if resolved == repo.resolve():
            return 101
        if resolved == external.resolve():
            return 202
        return int(path.stat().st_dev)

    monkeypatch.setattr(BACKUP, "_device_id", device)


def test_inputs_backup_copies_exact_objects_and_git_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, external = _repo_fixture(tmp_path)
    _different_devices(monkeypatch, repo, external)

    receipt = BACKUP.create_inputs_backup(repo, external.resolve())

    assert receipt["schema_version"] == "tlm-external-backup-receipt/v1"
    assert receipt["verified"] is True
    assert [item["source_path"] for item in receipt["objects"]] == list(
        BACKUP.EXPECTED_ALLOWED_INPUTS
    )
    assert len(receipt["objects"]) == 19
    for item in receipt["objects"]:
        source = repo / item["source_path"]
        copied = external / item["backup_path"]
        assert copied.read_bytes() == source.read_bytes()
        assert copied.stat().st_size == item["size_bytes"]
        assert _sha(copied) == item["sha256"]
    bundle = external / receipt["code_backup"]["backup_path"]
    assert bundle.is_file()
    assert _sha(bundle) == receipt["code_backup"]["sha256"]
    assert (
        yaml.safe_load((repo / BACKUP.INPUT_RECEIPT).read_text(encoding="utf-8"))
        == receipt
    )
    assert _git(repo, "status", "--porcelain", "--untracked-files=all") == ""


def test_inputs_backup_refuses_same_device_dirty_git_and_overwrite_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, external = _repo_fixture(tmp_path)
    inside_repo = repo / "backup"
    inside_repo.mkdir()
    with pytest.raises(BACKUP.BackupError, match="repository"):
        BACKUP.create_inputs_backup(repo, inside_repo.resolve())
    inside_repo.rmdir()

    with pytest.raises(BACKUP.BackupError, match="must be absolute"):
        BACKUP.create_inputs_backup(repo, Path("relative-backup"))

    monkeypatch.setattr(BACKUP, "_device_id", lambda _path: 1)
    with pytest.raises(BACKUP.BackupError, match="different st_dev"):
        BACKUP.create_inputs_backup(repo, external.resolve())

    _different_devices(monkeypatch, repo, external)
    dirty = repo / "dirty.txt"
    dirty.write_text("dirty", encoding="utf-8")
    with pytest.raises(BACKUP.BackupError, match="worktree must be clean"):
        BACKUP.create_inputs_backup(repo, external.resolve())
    dirty.unlink()

    receipt = BACKUP.create_inputs_backup(repo, external.resolve())
    first_backup = external / receipt["objects"][0]["backup_path"]
    first_backup.write_bytes(b"drift")
    with pytest.raises(BACKUP.BackupError, match="overwrite drift"):
        BACKUP.create_inputs_backup(repo, external.resolve())


def _write_checkpoint_manifest(repo: Path, mode: str = "complete") -> dict:
    jobs: list[dict] = []
    checkpoint_root = (
        repo / "data/checkpoints/v58_state_conditioned_multi_horizon_training"
    )
    for job_id in BACKUP.EXPECTED_JOBS:
        origin, geometry, fold, seed = job_id.split("|")
        checkpoint = (
            checkpoint_root
            / origin
            / geometry
            / f"fold_{fold}"
            / f"seed_{seed}"
            / "checkpoint.pt"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-{job_id}\n".encode())
        state_hash = hashlib.sha256(f"state-{job_id}".encode()).hexdigest()
        jobs.append(
            {
                "job_id": job_id,
                "origin": origin,
                "geometry": geometry,
                "fold": int(fold),
                "seed": int(seed),
                "status": "completed",
                "checkpoint_path": checkpoint.relative_to(repo).as_posix(),
                "checkpoint_sha256": _sha(checkpoint),
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "model_state_sha256": state_hash,
                "best_model_state_sha256": state_hash,
            }
        )
    if mode == "incomplete":
        jobs.pop()
    elif mode == "reordered":
        jobs[0], jobs[1] = jobs[1], jobs[0]
    manifest = {
        "version": "v58_checkpoint_manifest_v1",
        "expected_jobs": list(BACKUP.EXPECTED_JOBS),
        "jobs": jobs,
        "selected_jobs": [],
        "active_jobs": [],
        "checkpoint_count": len(jobs),
    }
    manifest["manifest_sha256"] = BACKUP.canonical_sha256(manifest)
    path = repo / BACKUP.CHECKPOINT_MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def test_checkpoint_backup_copies_exact_order_and_matches_operator_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, external = _repo_fixture(tmp_path)
    _different_devices(monkeypatch, repo, external)
    manifest = _write_checkpoint_manifest(repo)

    receipt = BACKUP.create_checkpoints_backup(repo, external.resolve())

    assert receipt["schema_version"] == "tlm-checkpoint-backup-receipt/v1"
    assert receipt["verified"] is True
    assert [item["job_id"] for item in receipt["checkpoints"]] == list(
        BACKUP.EXPECTED_JOBS
    )
    assert len(receipt["checkpoints"]) == 36
    assert receipt["checkpoint_manifest"]["registered_manifest_sha256"] == manifest[
        "manifest_sha256"
    ]
    manifest_record = receipt["checkpoint_manifest"]
    assert manifest_record["source_path"] == BACKUP.CHECKPOINT_MANIFEST.as_posix()
    manifest_source = repo / manifest_record["source_path"]
    manifest_copy = external / manifest_record["backup_path"]
    assert manifest_copy.read_bytes() == manifest_source.read_bytes()
    assert manifest_copy.stat().st_size == manifest_record["size_bytes"]
    assert _sha(manifest_source) == manifest_record["sha256"]
    assert _sha(manifest_copy) == manifest_record["sha256"]
    for item in receipt["checkpoints"]:
        source = repo / item["source_path"]
        copied = external / item["backup_path"]
        assert copied.read_bytes() == source.read_bytes()
        assert copied.stat().st_size == item["size_bytes"]
        assert _sha(copied) == item["sha256"]
    on_disk = json.loads((repo / BACKUP.CHECKPOINT_RECEIPT).read_text(encoding="utf-8"))
    assert on_disk == receipt

    original_stat = Path.stat

    def cross_device_stat(path: Path, *args: object, **kwargs: object):
        value = original_stat(path, *args, **kwargs)
        resolved = path.resolve()
        values = list(value)
        if resolved == repo.resolve():
            values[2] = 101
        elif resolved == external.resolve():
            values[2] = 202
        return type(value)(values)

    monkeypatch.setattr(Path, "stat", cross_device_stat)
    VALIDATOR.validate_checkpoint_backup(
        repo.resolve(), receipt, list(BACKUP.EXPECTED_JOBS)
    )


@pytest.mark.parametrize("mode", ["incomplete", "reordered"])
def test_checkpoint_backup_refuses_incomplete_or_reordered_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    repo, external = _repo_fixture(tmp_path)
    _different_devices(monkeypatch, repo, external)
    _write_checkpoint_manifest(repo, mode)

    with pytest.raises(BACKUP.BackupError, match="36 jobs|incomplete or out of order"):
        BACKUP.create_checkpoints_backup(repo, external.resolve())
    assert not (repo / BACKUP.CHECKPOINT_RECEIPT).exists()


def test_checkpoint_backup_refuses_existing_copy_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, external = _repo_fixture(tmp_path)
    _different_devices(monkeypatch, repo, external)
    _write_checkpoint_manifest(repo)
    receipt = BACKUP.create_checkpoints_backup(repo, external.resolve())
    first_backup = external / receipt["checkpoints"][0]["backup_path"]
    first_backup.write_bytes(b"drift")

    with pytest.raises(BACKUP.BackupError, match="overwrite drift"):
        BACKUP.create_checkpoints_backup(repo, external.resolve())


def test_checkpoint_backup_refuses_existing_manifest_copy_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, external = _repo_fixture(tmp_path)
    _different_devices(monkeypatch, repo, external)
    _write_checkpoint_manifest(repo)
    receipt = BACKUP.create_checkpoints_backup(repo, external.resolve())
    manifest_backup = external / receipt["checkpoint_manifest"]["backup_path"]
    manifest_backup.write_bytes(b"drift")

    with pytest.raises(BACKUP.BackupError, match="overwrite drift"):
        BACKUP.create_checkpoints_backup(repo, external.resolve())
