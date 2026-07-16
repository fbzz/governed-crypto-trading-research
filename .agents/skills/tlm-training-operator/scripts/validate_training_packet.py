#!/usr/bin/env python3
"""Validate a frozen TLM training operator packet without running training."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_HEAD = re.compile(r"^[0-9a-f]{40,64}$")
OPERATIONS = {"doctor", "smoke", "full", "verify", "replay"}
TARGET_ASSETS = {"BTC", "ETH", "SOL", "BTCUSDT", "ETHUSDT", "SOLUSDT"}
V58_CHECKPOINT_MANIFEST = (
    "artifacts/v58_state_conditioned_multi_horizon_training/"
    "checkpoint_manifest.json"
)


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def mapping(value: Any, name: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{name} must be an object")
    return value


def string_list(value: Any, name: str) -> list[str]:
    require(isinstance(value, list), f"{name} must be an array")
    require(all(isinstance(item, str) and item for item in value), f"{name} must contain non-empty strings")
    return value


def inside(root: Path, relative: Any, name: str) -> Path:
    require(isinstance(relative, str) and relative, f"{name} must be a relative path")
    candidate = (root / relative).resolve()
    require(candidate == root or root in candidate.parents, f"{name} escapes repo root")
    return candidate


def load_json_object(path: Path, name: str) -> dict[str, Any]:
    require(path.suffix == ".json", f"{name} must reference JSON evidence")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{name} is not valid JSON: {exc}") from exc
    return mapping(value, name)


def load_yaml_object(path: Path, name: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValidationError(f"{name} is not valid YAML: {exc}") from exc
    return mapping(value, name)


def evidence_object(
    root: Path, evidence: dict[str, Any], name: str
) -> dict[str, Any]:
    reference = mapping(evidence.get(name), f"evidence.{name}")
    path = inside(root, reference.get("path"), f"evidence.{name}.path")
    expected = reference.get("sha256")
    require(
        isinstance(expected, str) and SHA256.fullmatch(expected) is not None,
        f"invalid evidence hash: {name}",
    )
    require(path.is_file(), f"missing evidence: {name}")
    require(sha256_file(path) == expected, f"evidence hash drift: {name}")
    return load_json_object(path, f"evidence.{name}")


def nested_snapshot(value: dict[str, Any], name: str) -> dict[str, Any]:
    candidate = value.get(name, value)
    return mapping(candidate, f"{name} evidence snapshot")


def frozen_contract_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
    contract = spec.get("contract", spec)
    contract = mapping(contract, "frozen training contract")
    nested = contract.get("v58_contract")
    return mapping(nested, "frozen V58 contract") if nested is not None else contract


def exact_job_grid(contract: dict[str, Any]) -> list[str]:
    grid = mapping(
        contract.get(
            "grid_contract",
            contract.get("grid_optimizer_and_runtime_contract", contract.get("grid")),
        ),
        "frozen contract grid",
    )
    axis_names = (
        ("origins", "geometries", "folds", "seeds")
        if "origins" in grid or "geometries" in grid
        else ("folds", "seeds")
    )
    axes: list[list[Any]] = []
    for key in axis_names:
        values = grid.get(key)
        require(isinstance(values, list) and values, f"frozen grid.{key} is empty")
        require(len(values) == len(set(values)), f"frozen grid.{key} has duplicates")
        axes.append(values)
    jobs = ["|".join(str(value) for value in cell) for cell in itertools.product(*axes)]
    expected = grid.get("expected_jobs")
    require(
        isinstance(expected, int) and expected == len(jobs),
        "frozen expected_jobs contract drift",
    )
    return jobs


def manifest_job_ids(value: dict[str, Any]) -> tuple[list[str], list[str]]:
    raw_jobs = value.get("jobs", value.get("checkpoint_manifest", []))
    require(isinstance(raw_jobs, list), "checkpoint manifest jobs must be an array")
    completed: list[str] = []
    active: list[str] = []
    for item in raw_jobs:
        if isinstance(item, str):
            completed.append(item)
            continue
        entry = mapping(item, "checkpoint manifest job")
        job_id = entry.get("job_id")
        if not isinstance(job_id, str) or not job_id:
            fields = [entry.get(key) for key in ("origin", "geometry", "fold", "seed")]
            require(all(field is not None for field in fields), "manifest job lacks identity")
            job_id = "|".join(str(field) for field in fields)
        status = entry.get("status", "completed")
        if status == "completed":
            completed.append(job_id)
        elif status == "active":
            active.append(job_id)
        else:
            require(status in {"pending"}, f"unsupported manifest job status: {status}")
    return completed, active


def external_path(root: Path, relative: Any, name: str) -> Path:
    require(isinstance(relative, str) and relative, f"{name} must be relative")
    candidate = (root / relative).resolve()
    require(candidate == root or root in candidate.parents, f"{name} escapes backup root")
    return candidate


def validate_checkpoint_backup(
    root: Path,
    receipt: dict[str, Any],
    expected_jobs: list[str],
) -> None:
    require(
        receipt.get("schema_version") == "tlm-checkpoint-backup-receipt/v1",
        "unsupported checkpoint backup receipt schema",
    )
    require(receipt.get("phase") == "v58", "checkpoint backup phase drift")
    require(receipt.get("verified") is True, "checkpoint backup is not verified")
    backup_root_raw = receipt.get("backup_root")
    require(
        isinstance(backup_root_raw, str) and Path(backup_root_raw).is_absolute(),
        "checkpoint backup_root must be absolute",
    )
    backup_root = Path(backup_root_raw).resolve()
    require(backup_root.is_dir(), "checkpoint backup_root is missing")
    require(not (backup_root == root or root in backup_root.parents), "checkpoint backup_root is inside repository")
    source_device = int(root.stat().st_dev)
    backup_device = int(backup_root.stat().st_dev)
    require(receipt.get("different_device") is True, "checkpoint backup is not declared cross-device")
    require(source_device != backup_device, "checkpoint backup is on the source device")
    require(receipt.get("source_device") == source_device, "checkpoint backup source_device drift")
    require(receipt.get("backup_device") == backup_device, "checkpoint backup backup_device drift")

    manifest = mapping(
        receipt.get("checkpoint_manifest"), "checkpoint backup manifest"
    )
    require(
        manifest.get("source_path") == V58_CHECKPOINT_MANIFEST,
        "checkpoint manifest source path drift",
    )
    manifest_source = inside(
        root, manifest.get("source_path"), "checkpoint manifest source"
    )
    manifest_backup = external_path(
        backup_root, manifest.get("backup_path"), "checkpoint manifest backup"
    )
    manifest_hash = manifest.get("sha256")
    manifest_size = manifest.get("size_bytes")
    require(
        isinstance(manifest_hash, str) and SHA256.fullmatch(manifest_hash) is not None,
        "invalid checkpoint manifest backup hash",
    )
    require(
        manifest_source.is_file()
        and not manifest_source.is_symlink()
        and manifest_backup.is_file()
        and not manifest_backup.is_symlink(),
        "missing checkpoint manifest backup",
    )
    require(
        isinstance(manifest_size, int)
        and manifest_size >= 0
        and manifest_source.stat().st_size == manifest_size
        and manifest_backup.stat().st_size == manifest_size,
        "checkpoint manifest backup size drift",
    )
    require(
        sha256_file(manifest_source) == manifest_hash
        and sha256_file(manifest_backup) == manifest_hash,
        "checkpoint manifest backup hash drift",
    )
    manifest_value = load_json_object(
        manifest_source, "checkpoint manifest backup source"
    )
    registered_manifest_hash = manifest.get("registered_manifest_sha256")
    require(
        isinstance(registered_manifest_hash, str)
        and SHA256.fullmatch(registered_manifest_hash) is not None,
        "invalid registered checkpoint manifest hash",
    )
    manifest_without_hash = dict(manifest_value)
    require(
        manifest_without_hash.pop("manifest_sha256", None)
        == registered_manifest_hash,
        "registered checkpoint manifest hash drift",
    )
    require(
        canonical_sha256(manifest_without_hash) == registered_manifest_hash,
        "checkpoint manifest canonical hash drift",
    )

    checkpoints = receipt.get("checkpoints")
    require(isinstance(checkpoints, list), "checkpoint backup list is missing")
    by_job: dict[str, dict[str, Any]] = {}
    for raw in checkpoints:
        entry = mapping(raw, "checkpoint backup entry")
        job_id = entry.get("job_id")
        require(isinstance(job_id, str) and job_id not in by_job, "checkpoint backup jobs must be unique")
        by_job[job_id] = entry
    require(list(by_job) == expected_jobs, "checkpoint backup does not contain the exact ordered 36-job grid")
    for job_id in expected_jobs:
        entry = by_job[job_id]
        source_path = inside(root, entry.get("source_path"), f"checkpoint source {job_id}")
        backup_path = external_path(
            backup_root, entry.get("backup_path"), f"checkpoint backup {job_id}"
        )
        expected_hash = entry.get("sha256")
        expected_size = entry.get("size_bytes")
        require(
            isinstance(expected_hash, str) and SHA256.fullmatch(expected_hash) is not None,
            f"invalid checkpoint backup hash: {job_id}",
        )
        require(
            source_path.is_file()
            and not source_path.is_symlink()
            and backup_path.is_file()
            and not backup_path.is_symlink(),
            f"missing checkpoint backup: {job_id}",
        )
        require(
            isinstance(expected_size, int)
            and source_path.stat().st_size == expected_size
            and backup_path.stat().st_size == expected_size,
            f"checkpoint backup size drift: {job_id}",
        )
        require(
            sha256_file(source_path) == expected_hash
            and sha256_file(backup_path) == expected_hash,
            f"checkpoint backup hash drift: {job_id}",
        )


def validate_backup_policy_receipt(
    receipt: dict[str, Any], contract: dict[str, Any]
) -> None:
    expected_keys = {
        "version",
        "phase",
        "mode",
        "verified",
        "waiver_path",
        "waiver_sha256",
        "waived_safeguards",
        "external_input_backup_created",
        "external_code_backup_created",
        "external_checkpoint_backup_created",
        "policy_receipt_sha256",
    }
    require(
        set(receipt) == expected_keys,
        "backup policy receipt key set drift",
    )
    body = dict(receipt)
    registered = body.pop("policy_receipt_sha256", None)
    require(
        isinstance(registered, str)
        and SHA256.fullmatch(registered) is not None
        and canonical_sha256(body) == registered,
        "backup policy receipt canonical hash drift",
    )
    runtime = mapping(contract.get("runtime_contract"), "runtime contract")
    policy = mapping(runtime.get("backup_policy"), "runtime backup policy")
    waiver = mapping(policy.get("waiver"), "runtime backup waiver")
    phase = contract.get("phase")
    require(
        isinstance(phase, str)
        and receipt.get("version") == f"{phase}_backup_policy_receipt_v1"
        and receipt.get("phase") == phase
        and receipt.get("mode") == "owner_waiver"
        and receipt.get("verified") is True,
        "invalid owner-waiver backup policy receipt",
    )
    require(
        receipt.get("waiver_path") == waiver.get("path")
        and receipt.get("waiver_sha256") == waiver.get("file_sha256")
        and receipt.get("waived_safeguards")
        == policy.get("waived_safeguards"),
        "backup policy receipt differs from frozen owner waiver",
    )
    require(
        receipt.get("external_input_backup_created") is False
        and receipt.get("external_code_backup_created") is False
        and receipt.get("external_checkpoint_backup_created") is False,
        "owner-waiver receipt must not claim external copies",
    )


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=False
    )
    require(result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def workflow_json(root: Path, command: str) -> dict[str, Any]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(root / "src") if not existing else f"{root / 'src'}{os.pathsep}{existing}"
    result = subprocess.run(
        [sys.executable, "-m", "tlm", command],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    require(result.returncode == 0, f"{command} failed: {result.stderr.strip()}")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{command} did not emit JSON: {exc}") from exc
    return mapping(value, command)


def validate_packet(root: Path, packet: dict[str, Any]) -> dict[str, Any]:
    root = root.resolve()
    require(packet.get("schema_version") == "tlm-training-operator/v1", "unsupported schema_version")
    operation = packet.get("operation")
    require(operation in OPERATIONS, "operation must be doctor, smoke, full, verify, or replay")

    state = mapping(packet.get("research_state"), "research_state")
    state_path = inside(root, state.get("path"), "research_state.path")
    state_hash = state.get("sha256")
    require(isinstance(state_hash, str) and SHA256.fullmatch(state_hash) is not None, "invalid research state sha256")
    require(state_path.is_file() and sha256_file(state_path) == state_hash, "research state hash drift")
    current_state = load_yaml_object(state_path, "research state")
    live_status = workflow_json(root, "research-status")
    require(live_status.get("passed") is True, "live research status failed")
    for key in ("authorized_phase", "authorized_next_action", "authorized_command"):
        require(isinstance(state.get(key), str) and state.get(key), f"research_state.{key} must be non-empty")
        require(live_status.get(key) == state.get(key), f"live {key} differs from operator packet")

    contract = mapping(packet.get("contract"), "contract")
    require(contract.get("frozen") is True, "training contract is not frozen")
    authorized = string_list(contract.get("authorized_operations"), "contract.authorized_operations")
    require(operation in authorized, f"contract does not authorize {operation}")
    contract_hash = contract.get("sha256")
    require(isinstance(contract_hash, str) and SHA256.fullmatch(contract_hash) is not None, "invalid contract sha256")
    contract_path = inside(root, contract.get("path"), "contract.path")
    require(contract_path.is_file(), f"missing frozen contract: {contract_path}")
    require(sha256_file(contract_path) == contract_hash, "frozen contract hash drift")
    frozen_spec = load_json_object(contract_path, "frozen contract")
    frozen_contract = frozen_contract_from_spec(frozen_spec)
    strict_frozen = "phase_contract" in current_state and "phase_contract" in frozen_spec
    frozen_authorized = frozen_contract.get("authorized_operations")
    if strict_frozen:
        phase_reference = mapping(
            current_state.get("phase_contract"), "research state phase_contract"
        )
        phase_path = inside(
            root, phase_reference.get("path"), "research state phase_contract.path"
        )
        phase_hash = phase_reference.get("file_sha256")
        require(
            isinstance(phase_hash, str) and SHA256.fullmatch(phase_hash) is not None,
            "invalid live phase contract hash",
        )
        require(phase_path.is_file(), "live phase contract is missing")
        require(sha256_file(phase_path) == phase_hash, "live phase contract hash drift")
        require(
            live_status.get("phase_contract_path", live_status.get("phase_contract"))
            == phase_reference.get("path"),
            "research-status phase contract path drift",
        )
        spec_phase_reference = mapping(
            frozen_spec.get("phase_contract"), "training spec phase_contract"
        )
        require(
            spec_phase_reference == phase_reference,
            "training spec is not bound to the live phase contract reference",
        )
        live_phase_contract = load_yaml_object(phase_path, "live phase contract")
        require(
            frozen_spec.get("contract") == live_phase_contract,
            "training spec is not an exact projection of the live phase contract",
        )
        frozen_contract = live_phase_contract
        enforcement = mapping(
            frozen_contract.get("operator_enforcement_contract"),
            "operator enforcement contract",
        )
        frozen_authorized = string_list(
            enforcement.get("operation_order"),
            "frozen contract authorized operations",
        )
        require(
            set(frozen_authorized) == OPERATIONS,
            "frozen contract must authorize exactly doctor, smoke, full, verify, and replay",
        )
        require(
            authorized == frozen_authorized,
            "packet authorized_operations differ from frozen contract",
        )
        frozen_jobs = exact_job_grid(frozen_contract)
        runtime_contract = mapping(
            frozen_contract.get("runtime_contract"), "runtime contract"
        )
        expected_lock_path = runtime_contract.get("process_lock")
        require(
            isinstance(expected_lock_path, str) and expected_lock_path,
            "frozen contract process_lock is missing",
        )
        frozen_backup_policy = runtime_contract.get("backup_policy")
        backup_mode = (
            frozen_backup_policy.get("mode")
            if isinstance(frozen_backup_policy, dict)
            else "external"
        )
        require(
            backup_mode in {"external", "owner_waiver"},
            "unsupported frozen backup policy mode",
        )
        require(
            runtime_contract.get("process_lock") == expected_lock_path,
            "frozen contract global process_lock_path drift",
        )
    else:
        frozen_jobs = []
        expected_lock_path = None
        backup_mode = "external"

    source = mapping(packet.get("source_receipt"), "source_receipt")
    require(source.get("git_clean") is True, "source receipt is not clean")
    receipt_head = source.get("git_head")
    require(isinstance(receipt_head, str) and GIT_HEAD.fullmatch(receipt_head) is not None, "invalid source git_head")
    require(git(root, "rev-parse", "HEAD") == receipt_head, "live Git head differs from source receipt")
    require(git(root, "status", "--porcelain", "--untracked-files=all") == "", "tracked or untracked source is dirty")
    files = mapping(source.get("files"), "source_receipt.files")
    require(bool(files), "source receipt files cannot be empty")
    if strict_frozen:
        frozen_source_files = string_list(
            frozen_spec.get("source_receipt_files"),
            "training spec source_receipt_files",
        )
        require(
            len(frozen_source_files) == len(set(frozen_source_files)),
            "training spec source_receipt_files contains duplicates",
        )
        require(
            set(files) == set(frozen_source_files),
            "source receipt file set differs from frozen training spec",
        )
    for relative, expected in files.items():
        require(isinstance(relative, str), "source receipt path must be a string")
        require(isinstance(expected, str) and SHA256.fullmatch(expected) is not None, f"invalid source hash: {relative}")
        path = inside(root, relative, f"source_receipt.files[{relative}]")
        require(path.is_file(), f"missing source receipt file: {relative}")
        require(sha256_file(path) == expected, f"source hash drift: {relative}")
    bundle_hash = source.get("bundle_sha256")
    require(isinstance(bundle_hash, str) and SHA256.fullmatch(bundle_hash) is not None, "invalid source bundle_sha256")
    require(canonical_sha256(files) == bundle_hash, "source bundle hash drift")

    doctor = mapping(packet.get("doctor"), "doctor")
    for key in ("passed", "python_ok", "torch_ok", "mps_available", "deterministic_algorithms"):
        require(doctor.get(key) is True, f"doctor.{key} must be true")
    require(doctor.get("device") == "mps", "training device must be mps")
    require(doctor.get("dtype") == "float32", "training dtype must be float32")
    require(doctor.get("fallback_enabled") is False, "MPS fallback must be disabled")
    require(doctor.get("full_training_ready") is True, "operator packet is not full-training ready")
    live_doctor = workflow_json(root, "research-doctor")
    require(live_doctor.get("passed") is True, "live research doctor failed")
    require(live_doctor.get("full_training_ready") is True, "live research doctor blocks full training")
    require(live_doctor.get("authorized_phase") == state.get("authorized_phase"), "doctor/status authorized phase drift")
    require(mapping(live_doctor.get("runtime"), "research-doctor.runtime").get("mps_available") is True, "live doctor reports MPS unavailable")
    live_runtime = mapping(live_doctor.get("runtime"), "research-doctor.runtime")
    if strict_frozen:
        for key in (
            "python_ok",
            "torch_ok",
            "mps_available",
            "mps_operational",
            "deterministic_algorithms",
        ):
            require(live_runtime.get(key) is True, f"live doctor runtime.{key} must be true")
        require(live_runtime.get("device") == "mps", "live doctor device drift")
        require(live_runtime.get("dtype") == "float32", "live doctor dtype drift")
        require(live_runtime.get("fallback_enabled") is False, "live doctor fallback is enabled")
        require(doctor.get("mps_operational") is True, "packet doctor MPS probe failed")
        live_backup = mapping(live_doctor.get("backup"), "research-doctor.backup")
        require(live_backup.get("passed") is True, "live doctor backup failed")
        packet_backup_mode = doctor.get("backup_mode", "external")
        require(
            packet_backup_mode == backup_mode
            and live_backup.get("mode", "external") == backup_mode,
            "backup policy mode differs from frozen contract or live doctor",
        )
        require(doctor.get("backup_passed", True) is True, "packet backup policy failed")
        require(
            doctor.get("backup_objects_verified")
            == live_backup.get("objects_verified"),
            "backup object count differs from live doctor",
        )
        if backup_mode == "external":
            require(
                doctor.get("backup_receipt_sha256")
                == live_backup.get("receipt_sha256"),
                "backup receipt hash differs from live doctor",
            )
            require(
                doctor.get("code_backup_verified") is True
                and live_backup.get("code_backup_verified") is True,
                "code backup is not verified",
            )
        else:
            owner_policy = mapping(
                frozen_backup_policy, "frozen owner storage policy"
            )
            frozen_waiver = mapping(
                owner_policy.get("waiver"), "frozen owner storage waiver"
            )
            frozen_waiver_path = frozen_waiver.get("path")
            frozen_waiver_sha256 = frozen_waiver.get("file_sha256")
            require(
                isinstance(frozen_waiver_path, str)
                and isinstance(frozen_waiver_sha256, str)
                and SHA256.fullmatch(frozen_waiver_sha256) is not None,
                "frozen owner waiver reference is invalid",
            )
            require(
                runtime_contract.get("external_backup_receipt_required") is False
                and doctor.get("backup_required", False) is False
                and live_backup.get("required", False) is False,
                "owner waiver cannot require an external backup",
            )
            require(
                doctor.get("backup_waiver_path")
                == live_backup.get("waiver_path")
                == frozen_waiver_path
                and doctor.get("backup_waiver_sha256")
                == live_backup.get("waiver_sha256")
                == frozen_waiver_sha256,
                "owner waiver reference differs from frozen contract or live doctor",
            )
            require(
                doctor.get("backup_waiver_verified") is True
                and live_backup.get("waiver_verified") is True,
                "owner storage waiver is not verified",
            )
            require(
                doctor.get("backup_objects_verified") == 0
                and doctor.get("code_backup_verified") is False
                and live_backup.get("objects_verified") == 0
                and live_backup.get("code_backup_verified") is False,
                "owner waiver must not claim copied inputs or code",
            )
    live_fallback = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0").strip().lower()
    require(live_fallback in {"", "0", "false", "no", "off"}, "live PYTORCH_ENABLE_MPS_FALLBACK is enabled")
    free_bytes = doctor.get("disk_free_bytes")
    required_bytes = doctor.get("required_free_bytes")
    require(isinstance(free_bytes, int) and free_bytes >= 0, "doctor.disk_free_bytes must be a non-negative integer")
    require(isinstance(required_bytes, int) and required_bytes > 0, "doctor.required_free_bytes must be positive")
    require(free_bytes >= required_bytes, "doctor reports insufficient free disk")
    if strict_frozen:
        live_disk = mapping(live_doctor.get("disk"), "research-doctor.disk")
        require(live_disk.get("free_bytes", 0) >= required_bytes, "live doctor reports insufficient free disk")
        require(live_disk.get("required_free_bytes") == required_bytes, "doctor required disk drift")
    active_job_count = doctor.get("active_job_count")
    require(isinstance(active_job_count, int) and 0 <= active_job_count <= 1, "doctor must report zero or one active training job")
    if strict_frozen:
        live_lock = mapping(live_doctor.get("process_lock"), "research-doctor.process_lock")
        require(
            Path(str(live_lock.get("path"))).resolve()
            == (root / str(expected_lock_path)).resolve(),
            "live doctor process lock path differs from frozen global lock",
        )
        require(
            Path(str(doctor.get("process_lock_path"))).resolve()
            == Path(str(live_lock.get("path"))).resolve(),
            "packet process lock path differs from live doctor",
        )
        require(live_lock.get("active_job_count") == active_job_count, "live process lock count differs from packet")
        require(live_lock.get("available") is True, "another training process holds the global lock")
        require(active_job_count == 0, "operator phases must start with no active optimizer process")

    access = mapping(packet.get("data_access"), "data_access")
    require(access.get("outcome_rows_read") == 0, "training read evaluation outcomes")
    target_assets = string_list(access.get("target_assets_loaded"), "data_access.target_assets_loaded")
    require(not target_assets, f"training loaded target assets: {sorted(set(target_assets) & TARGET_ASSETS)}")
    require(string_list(access.get("forbidden_columns_loaded"), "data_access.forbidden_columns_loaded") == [], "training loaded forbidden columns")
    for key in (
        "predictions_written",
        "policy_actions_emitted",
        "performance_metrics_computed",
        "pnl_computed",
        "hyperparameters_changed",
    ):
        require(access.get(key) is False, f"data_access.{key} must be false")

    grid = mapping(packet.get("grid"), "grid")
    expected = string_list(grid.get("expected_jobs"), "grid.expected_jobs")
    completed = string_list(grid.get("completed_jobs"), "grid.completed_jobs")
    active = string_list(grid.get("active_jobs"), "grid.active_jobs")
    selected = string_list(grid.get("selected_jobs"), "grid.selected_jobs")
    require(expected and len(expected) == len(set(expected)), "expected job grid is empty or contains duplicates")
    if strict_frozen:
        grid_label = "exact frozen 36-job grid" if state.get("authorized_phase") == "v58" else "exact frozen job grid"
        require(expected == frozen_jobs, f"packet job grid differs from {grid_label}")
    require(len(completed) == len(set(completed)), "completed jobs contain duplicates")
    require(set(completed) <= set(expected), "completed job is outside frozen grid")
    require(len(active) <= 1 and set(active) <= set(expected), "active jobs violate the one-job frozen grid")
    require(active_job_count == len(active), "doctor active_job_count differs from grid.active_jobs")
    require(not selected, "checkpoint/job selection is forbidden")

    evidence = mapping(packet.get("evidence", {}), "evidence")
    protection_evidence = (
        "backup_policy" if backup_mode == "owner_waiver" else "checkpoint_backup"
    )
    required_evidence: dict[str, set[str]] = {
        "doctor": set(),
        "smoke": {"smoke", "data_access"},
        "full": {"data_access", "checkpoint_manifest"},
        "verify": {
            "data_access",
            "checkpoint_manifest",
            "verification",
            protection_evidence,
        },
        "replay": {
            "data_access",
            "checkpoint_manifest",
            "verification",
            "replay",
            protection_evidence,
        },
    }
    loaded_evidence: dict[str, dict[str, Any]] = {}
    if strict_frozen:
        missing_evidence = sorted(required_evidence[operation] - set(evidence))
        require(not missing_evidence, f"missing bound evidence: {missing_evidence}")
        contradictory = (
            "checkpoint_backup"
            if backup_mode == "owner_waiver"
            else "backup_policy"
        )
        require(
            contradictory not in evidence,
            f"{contradictory} evidence contradicts frozen {backup_mode} storage mode",
        )
        for name in required_evidence[operation]:
            loaded_evidence[name] = evidence_object(root, evidence, name)
        evidence_paths = [mapping(evidence[name], f"evidence.{name}").get("path") for name in required_evidence[operation]]
        require(len(evidence_paths) == len(set(evidence_paths)), "evidence paths must be unique")

    if "data_access" in loaded_evidence:
        access_snapshot = nested_snapshot(loaded_evidence["data_access"], "data_access")
        for key, value in access.items():
            require(
                access_snapshot.get(key) == value,
                f"data_access packet differs from bound evidence: {key}",
            )
    if "checkpoint_manifest" in loaded_evidence:
        manifest_snapshot = nested_snapshot(
            loaded_evidence["checkpoint_manifest"], "checkpoint_manifest"
        )
        manifest_expected = manifest_snapshot.get("expected_jobs")
        if manifest_expected is not None:
            require(manifest_expected == expected, "manifest expected grid drift")
        manifest_completed, manifest_active = manifest_job_ids(manifest_snapshot)
        require(manifest_completed == completed, "completed jobs differ from checkpoint manifest")
        require(manifest_active == active, "active jobs differ from checkpoint manifest")
        require(
            manifest_snapshot.get("selected_jobs", []) == selected,
            "selected jobs differ from checkpoint manifest",
        )
    if "checkpoint_backup" in loaded_evidence:
        validate_checkpoint_backup(root, loaded_evidence["checkpoint_backup"], expected)
    if "backup_policy" in loaded_evidence:
        validate_backup_policy_receipt(loaded_evidence["backup_policy"], frozen_contract)

    resume = mapping(packet.get("resume"), "resume")
    require(resume.get("granularity") == "epoch_boundary", "resume must be epoch-boundary only")
    require(resume.get("cross_job_resume_allowed") is False, "cross-job resume is forbidden")
    active_resumes = string_list(resume.get("active_resume_artifacts"), "resume.active_resume_artifacts")
    pending_resumes = string_list(
        resume.get("pending_resume_artifacts", []),
        "resume.pending_resume_artifacts",
    )
    orphan_resumes = string_list(resume.get("orphan_resume_artifacts"), "resume.orphan_resume_artifacts")
    require(len(active_resumes) <= 1, "more than one resume artifact is active")
    require(len(pending_resumes) <= 1, "more than one resume artifact is pending")
    require(not (active_resumes and pending_resumes), "resume artifact cannot be active and pending")
    if strict_frozen:
        require(not orphan_resumes, "frozen packet contains orphan resume artifacts")
        for relative in [*active_resumes, *pending_resumes]:
            resume_path = inside(root, relative, "resume artifact")
            require(resume_path.is_file(), f"resume artifact is missing: {relative}")
    if active_resumes:
        require(len(active) == 1, "resume artifact exists without exactly one active job")
    pending_resume_job = resume.get("pending_resume_job")
    if pending_resumes:
        require(not active, "pending resume requires no active optimizer job")
        require(
            isinstance(pending_resume_job, str) and pending_resume_job in set(expected),
            "pending resume is not bound to one frozen job",
        )
        require(
            pending_resume_job not in set(completed),
            "pending resume belongs to an already completed job",
        )
    else:
        require(
            pending_resume_job in {None, ""},
            "pending_resume_job exists without a pending artifact",
        )
    if "smoke" in loaded_evidence:
        smoke_snapshot = nested_snapshot(loaded_evidence["smoke"], "smoke")
        bound_resume = smoke_snapshot.get("resume", smoke_snapshot)
        require(isinstance(bound_resume, dict), "smoke resume evidence must be an object")
        require(
            bound_resume.get("interrupted_resume_matched")
            == resume.get("interrupted_resume_matched"),
            "smoke resume result differs from bound evidence",
        )

    verification = mapping(packet.get("verification"), "verification")
    verified = string_list(verification.get("checkpoint_jobs_verified"), "verification.checkpoint_jobs_verified")
    require(len(verified) == len(set(verified)) and set(verified) <= set(expected), "verified checkpoints drift from frozen grid")
    if "verification" in loaded_evidence:
        verification_snapshot = nested_snapshot(
            loaded_evidence["verification"], "verification"
        )
        for key, value in verification.items():
            require(
                verification_snapshot.get(key) == value,
                f"verification packet differs from bound evidence: {key}",
            )

    if operation == "smoke":
        require(resume.get("interrupted_resume_matched") is True, "smoke did not prove deterministic interrupted resume")
    if operation in {"verify", "replay"}:
        require(set(completed) == set(expected), "complete grid is not present")
        require(set(verified) == set(expected), "not every checkpoint is verified")
        require(verification.get("all_checkpoints_retained") is True, "not every checkpoint was retained")
        require(verification.get("checkpoint_roundtrip_passed") is True, "checkpoint roundtrip failed")
        require(
            not active
            and not active_resumes
            and not pending_resumes
            and not orphan_resumes,
            "verification found an active/pending/orphan resume or job",
        )

    replay = mapping(packet.get("replay"), "replay")
    if "replay" in loaded_evidence:
        replay_snapshot = nested_snapshot(loaded_evidence["replay"], "replay")
        for key, value in replay.items():
            require(
                replay_snapshot.get(key) == value,
                f"replay packet differs from bound evidence: {key}",
            )
    if operation == "replay":
        require(replay.get("new_jobs") == 0, "replay created a new job")
        require(replay.get("new_optimizer_steps") == 0, "replay executed optimizer steps")
        require(replay.get("artifact_hashes_match") is True, "replay artifact hashes drifted")

    return {
        "valid": True,
        "operation": operation,
        "contract_sha256": contract_hash,
        "git_head": receipt_head,
        "completed_jobs": len(completed),
        "expected_jobs": len(expected),
        "active_jobs": len(active),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    args = parser.parse_args()
    try:
        root = args.repo_root.resolve()
        require((root / ".git").exists(), "repo root is not a Git worktree")
        packet = json.loads(args.packet.read_text(encoding="utf-8"))
        result = validate_packet(root, mapping(packet, "packet"))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
