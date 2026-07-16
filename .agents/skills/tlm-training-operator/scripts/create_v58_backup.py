#!/usr/bin/env python3
"""Create hash-verified, cross-device V58 input and checkpoint backups."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Iterator
import uuid

import yaml


PHASE = "v58"
PHASE_CONTRACT = Path("research/phase_contracts/v058.yaml")
CURRENT_STATE = Path("research/current.yaml")
INPUT_RECEIPT = Path("research/backups/v058.yaml")
CHECKPOINT_MANIFEST = Path(
    "artifacts/v58_state_conditioned_multi_horizon_training/checkpoint_manifest.json"
)
CHECKPOINT_RECEIPT = Path(
    "artifacts/v58_state_conditioned_multi_horizon_training/"
    "checkpoint_backup_receipt.json"
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_ALLOWED_INPUTS = (
    "artifacts/v55_state_conditioned_multi_horizon_spec/blueprint.json",
    "artifacts/v56_state_conditioned_multi_horizon_harness/result.json",
    "artifacts/v56_state_conditioned_multi_horizon_harness/audit.json",
    "artifacts/v56_state_conditioned_multi_horizon_harness/harness_spec.json",
    "artifacts/v57_non_target_multi_horizon_dataset/result.json",
    "artifacts/v57_non_target_multi_horizon_dataset/audit.json",
    "artifacts/v57_non_target_multi_horizon_dataset/dataset_spec.json",
    "artifacts/v57_non_target_multi_horizon_dataset/dataset_manifest.json",
    "artifacts/v57_non_target_multi_horizon_dataset/label_schema.json",
    "artifacts/v57_non_target_multi_horizon_dataset/source_receipt.json",
    "artifacts/v57_non_target_multi_horizon_dataset/completion_receipt.json",
    "artifacts/v57_non_target_multi_horizon_dataset/artifact_manifest.json",
    "artifacts/v57_non_target_multi_horizon_dataset/data_access.json",
    "artifacts/v32_selected_universe_dataset/feature_schema.json",
    "artifacts/v32_selected_universe_dataset/asset_folds.json",
    "artifacts/v32_selected_universe_dataset/triplet_catalog.json",
    "data/processed/selected_universe_panel_v32.parquet",
    "data/processed/state_conditioned_multi_horizon_labels_v57.parquet",
    "data/processed/state_conditioned_multi_horizon_sequence_roles_v57.parquet",
)
EXPECTED_INPUT_KEYS = (
    "v55_blueprint",
    "v56_result",
    "v56_audit",
    "v56_harness_spec",
    "v57_result",
    "v57_audit",
    "v57_dataset_spec",
    "v57_dataset_manifest",
    "v57_label_schema",
    "v57_source_receipt",
    "v57_completion_receipt",
    "v57_artifact_manifest",
    "v57_data_access",
    "v32_feature_schema",
    "v32_asset_folds",
    "v32_triplet_catalog",
    "feature_panel",
    "labels",
    "sequence_roles",
)
EXPECTED_JOBS = tuple(
    f"{origin}|{geometry}|{fold}|{seed}"
    for origin in ("origin_2024", "origin_2025")
    for geometry in ("expanding", "rolling")
    for fold in (1, 2, 3)
    for seed in (42, 7, 123)
)


class BackupError(RuntimeError):
    """Raised when a backup safety invariant fails."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BackupError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{name} must be a mapping")
    return value


def _load_yaml(path: Path, name: str) -> dict[str, Any]:
    require(path.is_file(), f"missing {name}: {path}")
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), name)


def _load_json(path: Path, name: str) -> dict[str, Any]:
    require(path.is_file(), f"missing {name}: {path}")
    return _mapping(json.loads(path.read_text(encoding="utf-8")), name)


def _inside(root: Path, raw: str | Path, name: str) -> Path:
    relative = Path(raw)
    require(not relative.is_absolute(), f"{name} must be relative")
    candidate = (root / relative).resolve()
    require(candidate == root or root in candidate.parents, f"{name} escapes its root")
    return candidate


def _source_inside_repo(repo: Path, raw: str | Path, name: str) -> tuple[Path, str]:
    candidate = Path(raw)
    resolved = candidate.resolve() if candidate.is_absolute() else (repo / candidate).resolve()
    try:
        relative = resolved.relative_to(repo)
    except ValueError as exc:
        raise BackupError(f"{name} escapes repository") from exc
    return resolved, relative.as_posix()


def _device_id(path: Path) -> int:
    return int(path.stat().st_dev)


def _validate_roots(repo_root: Path, backup_root: Path) -> tuple[Path, Path, int, int]:
    repo = repo_root.resolve()
    require((repo / ".git").exists(), f"not a Git worktree: {repo}")
    require(backup_root.is_absolute(), "backup root must be absolute")
    require(backup_root.exists() and backup_root.is_dir(), "backup root must already exist")
    backup = backup_root.resolve()
    require(
        backup != repo and repo not in backup.parents and backup not in repo.parents,
        "backup root must not be the repository or one of its ancestors/descendants",
    )
    source_device = _device_id(repo)
    backup_device = _device_id(backup)
    require(source_device != backup_device, "backup root must be on a different st_dev")
    return repo, backup, source_device, backup_device


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=False
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise BackupError(f"git {' '.join(args)} failed: {detail}")
    return result


def _clean_git_head(repo: Path) -> str:
    status = _git(repo, "status", "--porcelain", "--untracked-files=all").stdout
    require(status.strip() == "", "Git worktree must be clean before backup")
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    require(re.fullmatch(r"[0-9a-f]{40,64}", head) is not None, "invalid Git HEAD")
    return head


def _require_ignored(repo: Path, relative: Path) -> None:
    result = _git(repo, "check-ignore", "-q", "--", relative.as_posix(), check=False)
    require(result.returncode == 0, f"receipt path is not ignored: {relative.as_posix()}")


def _temporary_sibling(destination: Path) -> Path:
    return destination.with_name(f".{destination.name}.tmp-{uuid.uuid4().hex}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _copy_verified(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> None:
    require(source.is_file() and not source.is_symlink(), f"invalid source file: {source}")
    require(SHA256.fullmatch(expected_sha256) is not None, f"invalid source hash: {source}")
    require(source.stat().st_size == expected_size, f"source size drift: {source}")
    require(sha256_file(source) == expected_sha256, f"source hash drift: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        require(destination.is_file() and not destination.is_symlink(), f"invalid backup object: {destination}")
        require(
            destination.stat().st_size == expected_size
            and sha256_file(destination) == expected_sha256,
            f"refusing overwrite drift: {destination}",
        )
        return
    temporary = _temporary_sibling(destination)
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        require(temporary.stat().st_size == expected_size, f"temporary copy size drift: {source}")
        require(sha256_file(temporary) == expected_sha256, f"temporary copy hash drift: {source}")
        require(not destination.exists(), f"refusing concurrent overwrite: {destination}")
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            temporary.unlink()
    require(
        destination.stat().st_size == expected_size
        and sha256_file(destination) == expected_sha256,
        f"installed backup verification failed: {destination}",
    )


def _atomic_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        require(path.is_file() and not path.is_symlink(), f"invalid receipt path: {path}")
        existing = _load_yaml(path, "existing backup receipt")
        require(existing == value, f"refusing receipt overwrite drift: {path}")
        return
    temporary = _temporary_sibling(path)
    payload = yaml.safe_dump(value, sort_keys=False, allow_unicode=False)
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        require(not path.exists(), f"refusing concurrent receipt overwrite: {path}")
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        require(path.is_file() and not path.is_symlink(), f"invalid receipt path: {path}")
        existing = _load_json(path, "existing backup receipt")
        require(existing == value, f"refusing receipt overwrite drift: {path}")
        return
    temporary = _temporary_sibling(path)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        require(not path.exists(), f"refusing concurrent receipt overwrite: {path}")
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def _backup_lock(backup_root: Path) -> Iterator[None]:
    lock_path = _inside(backup_root, ".tlm_v58_backup.lock", "backup lock")
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BackupError("another V58 backup process holds the external lock") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _bound_phase_contract(repo: Path) -> tuple[dict[str, Any], str]:
    state = _load_yaml(repo / CURRENT_STATE, "current research state")
    require(state.get("authorized_phase") == PHASE, "current phase is not V58")
    reference = _mapping(state.get("phase_contract"), "current phase_contract")
    require(reference.get("path") == PHASE_CONTRACT.as_posix(), "V58 phase-contract path drift")
    expected_hash = reference.get("file_sha256")
    require(isinstance(expected_hash, str) and SHA256.fullmatch(expected_hash) is not None, "invalid V58 phase-contract hash")
    contract_path = repo / PHASE_CONTRACT
    require(contract_path.is_file(), "missing V58 phase contract")
    require(sha256_file(contract_path) == expected_hash, "V58 phase-contract hash drift")
    contract = _load_yaml(contract_path, "V58 phase contract")
    require(contract.get("phase") == PHASE, "phase contract is not V58")
    require(
        contract.get("authorized_next_action")
        == "authorize_v58_frozen_non_target_training_only",
        "V58 authorization drift",
    )
    return contract, expected_hash


def _input_records(
    repo: Path, contract: dict[str, Any]
) -> list[tuple[str, Path, str, int]]:
    access = _mapping(contract.get("access_contract"), "V58 access contract")
    allowed = access.get("allowed_inputs")
    require(
        isinstance(allowed, list) and tuple(allowed) == EXPECTED_ALLOWED_INPUTS,
        "V58 allowed_inputs are not the exact ordered 19-object contract",
    )
    expected = _mapping(
        _mapping(contract.get("input_contract"), "V58 input contract").get(
            "expected_sha256"
        ),
        "V58 expected input hashes",
    )
    require(
        tuple(expected) == EXPECTED_INPUT_KEYS,
        "V58 expected input hash keys or ordering drifted",
    )
    records: list[tuple[str, Path, str, int]] = []
    for relative, key in zip(allowed, EXPECTED_INPUT_KEYS, strict=True):
        path = _inside(repo, relative, f"allowed input {relative}")
        expected_hash = expected[key]
        require(
            isinstance(expected_hash, str)
            and SHA256.fullmatch(expected_hash) is not None,
            f"invalid registered input hash: {relative}",
        )
        require(path.is_file() and not path.is_symlink(), f"missing input: {relative}")
        size = path.stat().st_size
        require(sha256_file(path) == expected_hash, f"registered input hash drift: {relative}")
        records.append((relative, path, expected_hash, size))
    require(len(records) == 19, "V58 input backup must contain exactly 19 objects")
    return records


def _install_git_bundle(
    repo: Path, backup_root: Path, destination: Path, git_head: str
) -> tuple[str, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_sibling(destination)
    try:
        _git(repo, "bundle", "create", str(temporary), "HEAD")
        _git(repo, "bundle", "verify", str(temporary))
        heads = _git(repo, "bundle", "list-heads", str(temporary)).stdout.splitlines()
        require(
            any(line.split(maxsplit=1)[0] == git_head for line in heads if line.strip()),
            "Git bundle does not contain the registered HEAD",
        )
        bundle_hash = sha256_file(temporary)
        bundle_size = temporary.stat().st_size
        if destination.exists() or destination.is_symlink():
            require(
                destination.is_file() and not destination.is_symlink(),
                f"invalid existing Git bundle: {destination}",
            )
            require(
                destination.stat().st_size == bundle_size
                and sha256_file(destination) == bundle_hash,
                f"refusing Git bundle overwrite drift: {destination}",
            )
        else:
            require(not destination.exists(), f"refusing concurrent bundle overwrite: {destination}")
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        _git(repo, "bundle", "verify", str(destination))
        installed_heads = _git(
            repo, "bundle", "list-heads", str(destination)
        ).stdout.splitlines()
        require(
            any(
                line.split(maxsplit=1)[0] == git_head
                for line in installed_heads
                if line.strip()
            ),
            "installed Git bundle does not contain the registered HEAD",
        )
        require(
            destination.stat().st_size == bundle_size
            and sha256_file(destination) == bundle_hash,
            "installed Git bundle verification failed",
        )
        return bundle_hash, bundle_size
    finally:
        if temporary.exists():
            temporary.unlink()


def create_inputs_backup(repo_root: Path, backup_root: Path) -> dict[str, Any]:
    repo, backup, source_device, backup_device = _validate_roots(
        repo_root, backup_root
    )
    git_head = _clean_git_head(repo)
    contract, phase_contract_hash = _bound_phase_contract(repo)
    records = _input_records(repo, contract)
    receipt_path = repo / INPUT_RECEIPT
    _require_ignored(repo, INPUT_RECEIPT)

    with _backup_lock(backup):
        objects: list[dict[str, Any]] = []
        for relative, source, expected_hash, size in records:
            backup_relative = Path("tlm/v58/inputs") / relative
            destination = _inside(
                backup, backup_relative, f"backup destination for {relative}"
            )
            _copy_verified(
                source,
                destination,
                expected_sha256=expected_hash,
                expected_size=size,
            )
            objects.append(
                {
                    "source_path": relative,
                    "backup_path": backup_relative.as_posix(),
                    "sha256": expected_hash,
                    "size_bytes": size,
                }
            )

        bundle_relative = Path("tlm/v58/code") / f"source-{git_head}.bundle"
        bundle_path = _inside(backup, bundle_relative, "Git bundle destination")
        bundle_hash, bundle_size = _install_git_bundle(
            repo, backup, bundle_path, git_head
        )
        require(_clean_git_head(repo) == git_head, "Git state changed during input backup")
        receipt = {
            "schema_version": "tlm-external-backup-receipt/v1",
            "phase": PHASE,
            "backup_root": str(backup),
            "source_device": source_device,
            "backup_device": backup_device,
            "different_device": True,
            "verified": True,
            "phase_contract": {
                "path": PHASE_CONTRACT.as_posix(),
                "sha256": phase_contract_hash,
            },
            "objects": objects,
            "code_backup": {
                "kind": "git_bundle",
                "git_head": git_head,
                "backup_path": bundle_relative.as_posix(),
                "sha256": bundle_hash,
                "size_bytes": bundle_size,
            },
        }
        _atomic_yaml(receipt_path, receipt)
        require(_clean_git_head(repo) == git_head, "Git state changed after receipt write")
    return receipt


def _checkpoint_records(
    repo: Path, contract: dict[str, Any]
) -> tuple[dict[str, Any], str, list[tuple[str, Path, str, int]]]:
    manifest_path = repo / CHECKPOINT_MANIFEST
    manifest = _load_json(manifest_path, "V58 checkpoint manifest")
    version = manifest.get("version")
    require(
        isinstance(version, str) and version.startswith("v58"),
        "checkpoint manifest version is not V58",
    )
    expected_jobs = manifest.get("expected_jobs")
    require(
        isinstance(expected_jobs, list) and tuple(expected_jobs) == EXPECTED_JOBS,
        "checkpoint manifest does not contain the exact ordered 36-job grid",
    )
    jobs = manifest.get("jobs")
    require(isinstance(jobs, list) and len(jobs) == 36, "checkpoint manifest must contain 36 jobs")
    require(manifest.get("checkpoint_count") == 36, "checkpoint manifest checkpoint_count drift")
    registered_manifest_hash = manifest.get("manifest_sha256")
    require(
        isinstance(registered_manifest_hash, str)
        and SHA256.fullmatch(registered_manifest_hash) is not None,
        "checkpoint manifest lacks a valid manifest_sha256",
    )
    without_hash = dict(manifest)
    without_hash.pop("manifest_sha256")
    require(
        canonical_sha256(without_hash) == registered_manifest_hash,
        "checkpoint manifest canonical hash drift",
    )
    require(manifest.get("selected_jobs", []) == [], "checkpoint selection is forbidden")
    require(manifest.get("active_jobs", []) == [], "checkpoint manifest still has active jobs")

    checkpoint_root_raw = _mapping(
        contract.get("access_contract"), "V58 access contract"
    ).get("checkpoint_dir")
    require(isinstance(checkpoint_root_raw, str), "V58 checkpoint_dir is missing")
    checkpoint_root = _inside(repo, checkpoint_root_raw, "V58 checkpoint root")
    records: list[tuple[str, Path, str, int]] = []
    seen_paths: set[str] = set()
    for expected_job, raw in zip(EXPECTED_JOBS, jobs, strict=True):
        entry = _mapping(raw, f"checkpoint manifest job {expected_job}")
        require(entry.get("job_id") == expected_job, "checkpoint jobs are incomplete or out of order")
        origin, geometry, fold_raw, seed_raw = expected_job.split("|")
        require(
            entry.get("origin") == origin
            and entry.get("geometry") == geometry
            and int(entry.get("fold", -1)) == int(fold_raw)
            and int(entry.get("seed", -1)) == int(seed_raw),
            f"checkpoint job axes drift: {expected_job}",
        )
        require(entry.get("status", "completed") == "completed", f"checkpoint job is incomplete: {expected_job}")
        source, relative = _source_inside_repo(
            repo, entry.get("checkpoint_path", ""), f"checkpoint path {expected_job}"
        )
        require(
            source == checkpoint_root or checkpoint_root in source.parents,
            f"checkpoint is outside frozen V58 checkpoint_dir: {expected_job}",
        )
        require(relative not in seen_paths, "checkpoint manifest contains duplicate paths")
        seen_paths.add(relative)
        expected_hash = entry.get("checkpoint_sha256")
        expected_size = entry.get("checkpoint_size_bytes")
        require(
            isinstance(expected_hash, str) and SHA256.fullmatch(expected_hash) is not None,
            f"invalid checkpoint hash: {expected_job}",
        )
        require(isinstance(expected_size, int) and expected_size >= 0, f"invalid checkpoint size: {expected_job}")
        model_hashes = [
            value
            for key, value in entry.items()
            if key.endswith("model_state_sha256")
        ]
        require(model_hashes, f"checkpoint manifest lacks model hashes: {expected_job}")
        require(
            all(isinstance(value, str) and SHA256.fullmatch(value) is not None for value in model_hashes),
            f"invalid checkpoint model hash: {expected_job}",
        )
        require(source.is_file() and not source.is_symlink(), f"missing checkpoint: {expected_job}")
        require(source.stat().st_size == expected_size, f"checkpoint size drift: {expected_job}")
        require(sha256_file(source) == expected_hash, f"checkpoint hash drift: {expected_job}")
        records.append((expected_job, source, expected_hash, expected_size))
    require(len(records) == 36, "checkpoint backup grid is incomplete")
    return manifest, sha256_file(manifest_path), records


def create_checkpoints_backup(
    repo_root: Path, backup_root: Path
) -> dict[str, Any]:
    repo, backup, source_device, backup_device = _validate_roots(
        repo_root, backup_root
    )
    git_head = _clean_git_head(repo)
    contract, phase_contract_hash = _bound_phase_contract(repo)
    manifest, manifest_file_hash, records = _checkpoint_records(repo, contract)
    receipt_path = repo / CHECKPOINT_RECEIPT
    _require_ignored(repo, CHECKPOINT_RECEIPT)

    with _backup_lock(backup):
        checkpoints: list[dict[str, Any]] = []
        for job_id, source, expected_hash, expected_size in records:
            origin, geometry, fold, seed = job_id.split("|")
            backup_relative = (
                Path("tlm/v58/checkpoints")
                / origin
                / geometry
                / f"fold_{fold}"
                / f"seed_{seed}"
                / "checkpoint.pt"
            )
            destination = _inside(
                backup, backup_relative, f"backup destination for {job_id}"
            )
            _copy_verified(
                source,
                destination,
                expected_sha256=expected_hash,
                expected_size=expected_size,
            )
            relative_source = source.relative_to(repo).as_posix()
            checkpoints.append(
                {
                    "job_id": job_id,
                    "source_path": relative_source,
                    "backup_path": backup_relative.as_posix(),
                    "sha256": expected_hash,
                    "size_bytes": expected_size,
                }
            )
        manifest_source = repo / CHECKPOINT_MANIFEST
        manifest_size = manifest_source.stat().st_size
        manifest_backup_relative = Path("tlm/v58/checkpoints/checkpoint_manifest.json")
        manifest_destination = _inside(
            backup, manifest_backup_relative, "checkpoint manifest backup destination"
        )
        _copy_verified(
            manifest_source,
            manifest_destination,
            expected_sha256=manifest_file_hash,
            expected_size=manifest_size,
        )
        require(_clean_git_head(repo) == git_head, "Git state changed during checkpoint backup")
        receipt = {
            "schema_version": "tlm-checkpoint-backup-receipt/v1",
            "phase": PHASE,
            "backup_root": str(backup),
            "source_device": source_device,
            "backup_device": backup_device,
            "different_device": True,
            "verified": True,
            "git_head": git_head,
            "phase_contract": {
                "path": PHASE_CONTRACT.as_posix(),
                "sha256": phase_contract_hash,
            },
            "checkpoint_manifest": {
                "source_path": CHECKPOINT_MANIFEST.as_posix(),
                "backup_path": manifest_backup_relative.as_posix(),
                "sha256": manifest_file_hash,
                "size_bytes": manifest_size,
                "registered_manifest_sha256": manifest["manifest_sha256"],
            },
            "checkpoints": checkpoints,
        }
        _atomic_json(receipt_path, receipt)
        require(_clean_git_head(repo) == git_head, "Git state changed after checkpoint receipt write")
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for operation in ("inputs", "checkpoints"):
        command = subparsers.add_parser(operation)
        command.add_argument("--repo-root", type=Path, default=Path.cwd())
        command.add_argument("--backup-root", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.operation == "inputs":
            receipt = create_inputs_backup(args.repo_root, args.backup_root)
            summary = {
                "operation": "inputs",
                "verified": receipt["verified"],
                "objects": len(receipt["objects"]),
                "receipt": INPUT_RECEIPT.as_posix(),
            }
        else:
            receipt = create_checkpoints_backup(args.repo_root, args.backup_root)
            summary = {
                "operation": "checkpoints",
                "verified": receipt["verified"],
                "checkpoints": len(receipt["checkpoints"]),
                "receipt": CHECKPOINT_RECEIPT.as_posix(),
            }
    except (BackupError, OSError, ValueError, TypeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(json.dumps({"verified": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
