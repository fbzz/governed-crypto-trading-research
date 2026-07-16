#!/usr/bin/env python3
"""Build a V58 operator packet only from frozen contracts and bound receipts."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from types import ModuleType
from typing import Any


OPERATIONS = ("doctor", "smoke", "full", "verify", "replay")
EVIDENCE_NAMES = (
    "data_access",
    "checkpoint_manifest",
    "smoke",
    "verification",
    "replay",
    "checkpoint_backup",
    "backup_policy",
    "resume",
)


class PacketBuildError(RuntimeError):
    pass


def _load_validator() -> ModuleType:
    path = Path(__file__).with_name("validate_training_packet.py")
    spec = importlib.util.spec_from_file_location("tlm_training_packet_validator", path)
    if spec is None or spec.loader is None:
        raise PacketBuildError(f"cannot import packet validator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATOR = _load_validator()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PacketBuildError(f"{name} must be an object")
    return value


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), name)
    except json.JSONDecodeError as exc:
        raise PacketBuildError(f"{name} is not valid JSON: {exc}") from exc


def _inside(root: Path, relative: str | Path, name: str) -> Path:
    raw = str(relative)
    if not raw:
        raise PacketBuildError(f"{name} must be a relative path")
    candidate = (root / raw).resolve()
    if candidate != root and root not in candidate.parents:
        raise PacketBuildError(f"{name} escapes repository root")
    return candidate


def _snapshot(value: dict[str, Any], name: str) -> dict[str, Any]:
    candidate = value.get(name, value)
    return _mapping(candidate, f"{name} snapshot")


def _required(mapping: dict[str, Any], key: str, name: str) -> Any:
    if key not in mapping:
        raise PacketBuildError(f"{name}.{key} is missing")
    return mapping[key]


def _bool(mapping: dict[str, Any], key: str, name: str) -> bool:
    value = _required(mapping, key, name)
    if not isinstance(value, bool):
        raise PacketBuildError(f"{name}.{key} must be boolean")
    return value


def _list(mapping: dict[str, Any], key: str, name: str) -> list[Any]:
    value = _required(mapping, key, name)
    if not isinstance(value, list):
        raise PacketBuildError(f"{name}.{key} must be an array")
    return value


def _first(mapping: dict[str, Any], keys: tuple[str, ...], name: str) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    raise PacketBuildError(f"{name} lacks all accepted fields: {', '.join(keys)}")


def _source_snapshot(value: dict[str, Any]) -> dict[str, Any]:
    source = _snapshot(value, "source_receipt")
    git_value = source.get("git", source.get("git_receipt", {}))
    git_snapshot = git_value if isinstance(git_value, dict) else {}
    clean = source.get("git_clean", git_snapshot.get("clean"))
    head = source.get("git_head", git_snapshot.get("head"))
    if clean is not True:
        raise PacketBuildError("source receipt does not prove a clean Git tree")
    if not isinstance(head, str) or not head:
        raise PacketBuildError("source receipt lacks git_head")
    files = _mapping(_required(source, "files", "source_receipt"), "source_receipt.files")
    bundle = _required(source, "bundle_sha256", "source_receipt")
    return {
        "git_clean": True,
        "git_head": head,
        "files": files,
        "bundle_sha256": bundle,
    }


def _doctor_snapshot(value: dict[str, Any]) -> dict[str, Any]:
    runtime = _mapping(_required(value, "runtime", "doctor"), "doctor.runtime")
    disk = _mapping(_required(value, "disk", "doctor"), "doctor.disk")
    lock = _mapping(
        _required(value, "process_lock", "doctor"), "doctor.process_lock"
    )
    backup = _mapping(_required(value, "backup", "doctor"), "doctor.backup")
    backup_mode = backup.get("mode", "external")
    if backup_mode not in {"external", "owner_waiver"}:
        raise PacketBuildError(f"unsupported doctor backup mode: {backup_mode}")
    backup_required = backup.get("required", backup_mode == "external")
    backup_passed = backup.get("passed", True)
    if not isinstance(backup_required, bool) or not isinstance(backup_passed, bool):
        raise PacketBuildError("doctor backup required/passed fields must be boolean")
    snapshot = {
        "passed": _bool(value, "passed", "doctor"),
        "python_ok": _bool(runtime, "python_ok", "doctor.runtime"),
        "torch_ok": _bool(runtime, "torch_ok", "doctor.runtime"),
        "mps_available": _bool(runtime, "mps_available", "doctor.runtime"),
        "mps_operational": _bool(runtime, "mps_operational", "doctor.runtime"),
        "device": _required(runtime, "device", "doctor.runtime"),
        "dtype": _required(runtime, "dtype", "doctor.runtime"),
        "deterministic_algorithms": _bool(
            runtime, "deterministic_algorithms", "doctor.runtime"
        ),
        "fallback_enabled": _bool(
            runtime, "fallback_enabled", "doctor.runtime"
        ),
        "disk_free_bytes": _required(disk, "free_bytes", "doctor.disk"),
        "required_free_bytes": _required(
            disk, "required_free_bytes", "doctor.disk"
        ),
        "active_job_count": _required(
            lock, "active_job_count", "doctor.process_lock"
        ),
        "process_lock_path": _required(lock, "path", "doctor.process_lock"),
        "backup_mode": backup_mode,
        "backup_required": backup_required,
        "backup_passed": backup_passed,
        "backup_receipt_sha256": backup.get("receipt_sha256"),
        "backup_waiver_path": backup.get("waiver_path"),
        "backup_waiver_sha256": backup.get("waiver_sha256"),
        "backup_waiver_verified": bool(backup.get("waiver_verified", False)),
        "backup_objects_verified": _required(
            backup, "objects_verified", "doctor.backup"
        ),
        "code_backup_verified": _bool(
            backup, "code_backup_verified", "doctor.backup"
        ),
        "full_training_ready": _bool(
            value, "full_training_ready", "doctor"
        ),
    }
    if backup_mode == "external":
        if not isinstance(snapshot["backup_receipt_sha256"], str):
            raise PacketBuildError("external backup receipt hash is missing")
        if snapshot["code_backup_verified"] is not True:
            raise PacketBuildError("external code backup is not verified")
    else:
        if (
            not isinstance(snapshot["backup_waiver_path"], str)
            or not isinstance(snapshot["backup_waiver_sha256"], str)
        ):
            raise PacketBuildError("owner-waiver receipt hash is missing")
        if snapshot["backup_waiver_verified"] is not True:
            raise PacketBuildError("owner backup waiver is not verified")
    return snapshot


def _data_access_snapshot(value: dict[str, Any]) -> dict[str, Any]:
    access = _snapshot(value, "data_access")
    return {
        "outcome_rows_read": _first(
            access,
            ("outcome_rows_read", "development_evaluation_outcome_rows_read"),
            "data_access",
        ),
        "target_assets_loaded": _list(
            access, "target_assets_loaded", "data_access"
        ),
        "forbidden_columns_loaded": _list(
            access, "forbidden_columns_loaded", "data_access"
        ),
        "predictions_written": _bool(
            access, "predictions_written", "data_access"
        ),
        "policy_actions_emitted": _bool(
            access, "policy_actions_emitted", "data_access"
        ),
        "performance_metrics_computed": _bool(
            access, "performance_metrics_computed", "data_access"
        ),
        "pnl_computed": _bool(access, "pnl_computed", "data_access"),
        "hyperparameters_changed": _bool(
            access, "hyperparameters_changed", "data_access"
        ),
    }


def _resume_snapshot(
    operation: str,
    contract: dict[str, Any],
    evidence: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for name in ("resume", "smoke", "checkpoint_manifest"):
        if name in evidence:
            snapshot = _snapshot(evidence[name], name)
            nested = snapshot.get("resume", snapshot)
            if isinstance(nested, dict):
                candidates.append(nested)
    value = candidates[0] if candidates else {}
    checkpoint = contract.get("checkpoint_contract", {})
    cross_job = (
        checkpoint.get("cross_job_resume_allowed")
        if isinstance(checkpoint, dict)
        else None
    )
    if cross_job is not False:
        raise PacketBuildError("frozen contract does not forbid cross-job resume")
    interrupted = value.get("interrupted_resume_matched", False)
    if not isinstance(interrupted, bool):
        raise PacketBuildError("resume interrupted_resume_matched must be boolean")
    if operation == "smoke" and interrupted is not True:
        raise PacketBuildError("smoke receipt does not prove interrupted resume")
    return {
        "granularity": "epoch_boundary",
        "cross_job_resume_allowed": False,
        "active_resume_artifacts": list(value.get("active_resume_artifacts", [])),
        "pending_resume_artifacts": list(value.get("pending_resume_artifacts", [])),
        "pending_resume_job": value.get("pending_resume_job"),
        "orphan_resume_artifacts": list(value.get("orphan_resume_artifacts", [])),
        "interrupted_resume_matched": interrupted,
    }


def _verification_snapshot(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {
            "checkpoint_jobs_verified": [],
            "all_checkpoints_retained": False,
            "checkpoint_roundtrip_passed": False,
        }
    snapshot = _snapshot(value, "verification")
    return {
        "checkpoint_jobs_verified": _list(
            snapshot, "checkpoint_jobs_verified", "verification"
        ),
        "all_checkpoints_retained": _bool(
            snapshot, "all_checkpoints_retained", "verification"
        ),
        "checkpoint_roundtrip_passed": _bool(
            snapshot, "checkpoint_roundtrip_passed", "verification"
        ),
    }


def _replay_snapshot(value: dict[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {
            "new_jobs": 0,
            "new_optimizer_steps": 0,
            "artifact_hashes_match": False,
        }
    snapshot = _snapshot(value, "replay")
    return {
        "new_jobs": _required(snapshot, "new_jobs", "replay"),
        "new_optimizer_steps": _required(
            snapshot, "new_optimizer_steps", "replay"
        ),
        "artifact_hashes_match": _bool(
            snapshot, "artifact_hashes_match", "replay"
        ),
    }


def _evidence_ref(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": str(path.resolve().relative_to(root)),
        "sha256": VALIDATOR.sha256_file(path),
    }


def build_packet(
    *,
    repo_root: Path,
    operation: str,
    training_spec_path: Path,
    source_receipt_path: Path,
    evidence_paths: dict[str, Path] | None = None,
    state_path: Path = Path("research/current.yaml"),
    live_status: dict[str, Any] | None = None,
    live_doctor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Purely derive a packet from frozen files and supplied live snapshots."""

    root = repo_root.resolve()
    if operation not in OPERATIONS:
        raise PacketBuildError(f"unsupported operation: {operation}")
    state_file = _inside(root, state_path, "state_path")
    spec_file = _inside(root, training_spec_path, "training_spec_path")
    source_file = _inside(root, source_receipt_path, "source_receipt_path")
    for name, path in (
        ("research state", state_file),
        ("training spec", spec_file),
        ("source receipt", source_file),
    ):
        if not path.is_file():
            raise PacketBuildError(f"missing {name}: {path}")

    status = live_status or VALIDATOR.workflow_json(root, "research-status")
    doctor_live = live_doctor or VALIDATOR.workflow_json(root, "research-doctor")
    spec = _load_json(spec_file, "training_spec")
    contract = VALIDATOR.frozen_contract_from_spec(spec)
    runtime_contract = _mapping(
        contract.get("runtime_contract"), "runtime_contract"
    )
    backup_policy = runtime_contract.get("backup_policy")
    backup_mode = (
        backup_policy.get("mode")
        if isinstance(backup_policy, dict)
        else "external"
    )
    if backup_mode not in {"external", "owner_waiver"}:
        raise PacketBuildError(f"unsupported backup mode: {backup_mode}")
    exact_jobs = VALIDATOR.exact_job_grid(contract)
    enforcement = _mapping(
        contract.get("operator_enforcement_contract"),
        "operator_enforcement_contract",
    )
    operations = _list(enforcement, "operation_order", "operator_enforcement_contract")

    resolved_evidence: dict[str, Path] = {}
    evidence_values: dict[str, dict[str, Any]] = {}
    for name, raw_path in (evidence_paths or {}).items():
        if name not in EVIDENCE_NAMES:
            raise PacketBuildError(f"unsupported evidence name: {name}")
        path = _inside(root, raw_path, f"evidence {name}")
        if not path.is_file():
            raise PacketBuildError(f"missing evidence {name}: {path}")
        resolved_evidence[name] = path
        evidence_values[name] = _load_json(path, f"evidence {name}")

    protection_evidence = (
        "backup_policy" if backup_mode == "owner_waiver" else "checkpoint_backup"
    )
    required_evidence = {
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
            protection_evidence,
            "replay",
        },
    }
    missing = sorted(required_evidence[operation] - set(resolved_evidence))
    if missing:
        raise PacketBuildError(f"missing operation evidence: {missing}")
    contradictory = (
        "checkpoint_backup" if backup_mode == "owner_waiver" else "backup_policy"
    )
    if contradictory in resolved_evidence:
        raise PacketBuildError(
            f"{contradictory} evidence contradicts frozen {backup_mode} storage mode"
        )

    if "data_access" in evidence_values:
        data_access = _data_access_snapshot(evidence_values["data_access"])
    else:
        data_contract = _mapping(
            contract.get("data_access_ledger_contract"),
            "data_access_ledger_contract",
        )
        data_access = _data_access_snapshot(data_contract)

    completed: list[str] = []
    active: list[str] = []
    selected: list[str] = []
    if "checkpoint_manifest" in evidence_values:
        manifest = _snapshot(
            evidence_values["checkpoint_manifest"], "checkpoint_manifest"
        )
        completed, active = VALIDATOR.manifest_job_ids(manifest)
        raw_selected = manifest.get("selected_jobs", [])
        if not isinstance(raw_selected, list):
            raise PacketBuildError("checkpoint manifest selected_jobs must be an array")
        selected = list(raw_selected)

    packet_evidence = {
        name: _evidence_ref(root, path)
        for name, path in resolved_evidence.items()
    }
    packet = {
        "schema_version": "tlm-training-operator/v1",
        "operation": operation,
        "research_state": {
            "path": str(state_file.relative_to(root)),
            "sha256": VALIDATOR.sha256_file(state_file),
            "authorized_phase": _required(
                status, "authorized_phase", "research-status"
            ),
            "authorized_next_action": _required(
                status, "authorized_next_action", "research-status"
            ),
            "authorized_command": _required(
                status, "authorized_command", "research-status"
            ),
        },
        "contract": {
            "path": str(spec_file.relative_to(root)),
            "sha256": VALIDATOR.sha256_file(spec_file),
            "frozen": True,
            "authorized_operations": operations,
        },
        "source_receipt": _source_snapshot(
            _load_json(source_file, "source_receipt")
        ),
        "doctor": _doctor_snapshot(doctor_live),
        "data_access": data_access,
        "grid": {
            "expected_jobs": exact_jobs,
            "completed_jobs": completed,
            "active_jobs": active,
            "selected_jobs": selected,
        },
        "resume": _resume_snapshot(operation, contract, evidence_values),
        "verification": _verification_snapshot(
            evidence_values.get("verification")
        ),
        "replay": _replay_snapshot(evidence_values.get("replay")),
        "evidence": packet_evidence,
    }
    return packet


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def write_and_validate_packet(
    root: Path, output_path: Path, packet: dict[str, Any]
) -> dict[str, Any]:
    output = _inside(root.resolve(), output_path, "output")
    _atomic_json(output, packet)
    try:
        loaded = _load_json(output, "written operator packet")
        return VALIDATOR.validate_packet(root.resolve(), loaded)
    except Exception:
        output.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--operation", choices=OPERATIONS, required=True)
    parser.add_argument("--training-spec", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--state", type=Path, default=Path("research/current.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    for name in EVIDENCE_NAMES:
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path)
    args = parser.parse_args()
    evidence = {
        name: getattr(args, name)
        for name in EVIDENCE_NAMES
        if getattr(args, name) is not None
    }
    try:
        packet = build_packet(
            repo_root=args.repo_root,
            operation=args.operation,
            training_spec_path=args.training_spec,
            source_receipt_path=args.source_receipt,
            evidence_paths=evidence,
            state_path=args.state,
        )
        validation = write_and_validate_packet(
            args.repo_root.resolve(), args.output, packet
        )
    except (OSError, PacketBuildError, VALIDATOR.ValidationError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(validation, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
