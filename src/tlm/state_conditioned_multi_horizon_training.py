"""Frozen V58 non-target training orchestration.

The runner deliberately separates metadata authorization, exact projected data
access, optimization/checkpoint mechanics, and operator evidence.  No mode in
this module is allowed to create predictions, policy actions, performance
metrics, PnL, or target-asset tensors.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Iterator, Literal, Mapping

import torch

from .core.artifacts import canonical_sha256, file_sha256
from .research_workflow import research_doctor
from .state_conditioned_multi_horizon_training_artifacts import (
    V58_OWNER_STORAGE_WAIVER,
    V58_WAIVED_STORAGE_SAFEGUARDS,
    build_artifact_manifest,
    build_checkpoint_manifest,
    build_data_access_manifest,
    build_grid_manifest,
    build_history_manifest,
    build_scaler_manifest,
    prepare_training_metadata,
    stable_replay_hashes,
    write_json_atomic,
    write_report_atomic,
    write_yaml_atomic,
)
from .state_conditioned_multi_horizon_training_data import (
    JobTrainingData,
    TrainOnlyScaler,
    UniformDateTripletSampler,
    build_job_cell,
    fit_job_train_only_scaler,
    materialize_triplet_batch,
    read_job_training_data,
)
from .state_conditioned_multi_horizon_training_engine import (
    V58Batch,
    V58BatchStream,
    V58CheckpointContext,
    prove_v58_interrupted_resume_equivalence,
    run_v58_training_job,
    verify_v58_checkpoint_roundtrip,
)
from .state_conditioned_multi_horizon_model import (
    StateConditionedMultiHorizonTransformer,
)


MODES = ("preflight", "smoke", "full", "verify", "replay")
PACKET_SCRIPT = Path(
    ".agents/skills/tlm-training-operator/scripts/build_training_packet.py"
)
VALIDATOR_SCRIPT = Path(
    ".agents/skills/tlm-training-operator/scripts/validate_training_packet.py"
)
EMPTY_RESUME = {
    "active_resume_artifacts": [],
    "pending_resume_artifacts": [],
    "pending_resume_job": None,
    "orphan_resume_artifacts": [],
    "interrupted_resume_matched": False,
}


class V58TrainingError(RuntimeError):
    """Raised when a frozen V58 gate or receipt is not satisfied."""


def _json_value(value: Any) -> Any:
    """Normalize tuples/numpy-like scalars through strict JSON semantics."""

    return json.loads(json.dumps(value, allow_nan=False))


def _load_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise V58TrainingError(f"Missing {name}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise V58TrainingError(f"Invalid {name}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise V58TrainingError(f"{name} must be a JSON object: {path}")
    return value


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise V58TrainingError(f"Path escapes V58 repository: {path}") from exc


def _write_or_verify_json(path: Path, value: Mapping[str, Any]) -> None:
    normalized = _json_value(dict(value))
    if path.is_file():
        if _load_json(path, path.name) != normalized:
            raise V58TrainingError(f"Immutable V58 receipt drift: {path}")
        return
    write_json_atomic(path, normalized)


@contextmanager
def _process_lock(path: Path, operation: str) -> Iterator[None]:
    """Hold the one frozen advisory lock for smoke/full/verify/replay."""

    if operation not in {"smoke", "full", "verify", "replay"}:
        raise ValueError("V58 process lock operation is not authorized")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise V58TrainingError(
                f"Another V58 process holds the global lock: {path}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {"operation": operation, "pid": os.getpid()}, sort_keys=True
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


class V58MaterializedBatchProvider:
    """Build registered deterministic streams from one authorized job cell."""

    def __init__(
        self,
        data: JobTrainingData,
        scaler: TrainOnlyScaler,
        *,
        origin: str,
        geometry: str,
        fold: int,
        job_seed: int,
        train_samples: int,
        validation_samples: int,
        batch_size: int,
    ) -> None:
        if min(train_samples, validation_samples, batch_size) < 1:
            raise ValueError("V58 sample and batch counts must be positive")
        if (origin, geometry, int(fold)) != (
            data.cell.origin,
            data.cell.geometry,
            data.cell.fold,
        ):
            raise ValueError("V58 batch-provider job-cell identity drift")
        self.data = data
        self.scaler = scaler
        self.origin = origin
        self.geometry = geometry
        self.fold = int(fold)
        self.job_seed = int(job_seed)
        self.train_samples = int(train_samples)
        self.validation_samples = int(validation_samples)
        self.batch_size = int(batch_size)
        self.train_sampler = UniformDateTripletSampler(
            data.train_availability, data.cell.train_triplets
        )
        self.validation_sampler = UniformDateTripletSampler(
            data.validation_availability, data.cell.train_triplets
        )

    def __call__(
        self, role: Literal["train", "validation"], epoch: int
    ) -> V58BatchStream:
        if role == "train":
            sample_count = self.train_samples
            sampler = self.train_sampler
            registered_epoch = int(epoch)
        elif role == "validation":
            sample_count = self.validation_samples
            sampler = self.validation_sampler
            registered_epoch = 0
            if int(epoch) != 0:
                raise ValueError("V58 validation stream is frozen at epoch zero")
        else:
            raise ValueError(f"Unsupported V58 batch role: {role}")
        sampled = sampler.sample(
            sample_count,
            origin=self.origin,
            geometry=self.geometry,
            fold=self.fold,
            job_seed=self.job_seed,
            role=role,
            epoch=registered_epoch,
        )

        def batches() -> Iterable[V58Batch]:
            for offset in range(0, len(sampled.draws), self.batch_size):
                materialized = materialize_triplet_batch(
                    self.data,
                    sampled.draws[offset : offset + self.batch_size],
                    self.scaler,
                    role=role,
                )
                yield V58Batch(
                    features=torch.from_numpy(materialized.features),
                    targets=torch.from_numpy(materialized.targets),
                )

        return V58BatchStream(
            batches=batches(),
            sampler_receipt=sampled.ordered_draw_list_sha256,
        )


def _doctor_or_raise(metadata: Mapping[str, Any]) -> dict[str, Any]:
    doctor = research_doctor(metadata["root"], metadata["training"]["research_state"])
    if doctor.get("passed") is not True or doctor.get("full_training_ready") is not True:
        raise V58TrainingError(
            "V58 runtime doctor blocks execution: "
            + json.dumps(
                {
                    "warnings": doctor.get("warnings", []),
                    "disk": doctor.get("disk", {}),
                    "backup": doctor.get("backup", {}),
                    "runtime": doctor.get("runtime", {}),
                },
                sort_keys=True,
            )
        )
    return doctor


def _storage_protection_mode(metadata: Mapping[str, Any]) -> str:
    contract = metadata["contract"]
    runtime = metadata["contract"]["runtime_contract"]
    policy = runtime.get("backup_policy")
    if policy is None:
        if (
            runtime.get("external_backup_receipt_required") is not True
            or not isinstance(runtime.get("external_backup_receipt"), str)
            or not runtime["external_backup_receipt"]
        ):
            raise V58TrainingError(
                "V58 external storage mode requires its frozen backup receipt"
            )
        return "external"
    if not isinstance(policy, Mapping) or policy.get("mode") != "owner_waiver":
        raise V58TrainingError("Unsupported V58 storage protection policy")
    if (
        contract.get("revision") != "v058r1"
        or runtime.get("external_backup_receipt_required") is not False
        or policy.get("waiver") != V58_OWNER_STORAGE_WAIVER
        or policy.get("waived_safeguards") != V58_WAIVED_STORAGE_SAFEGUARDS
    ):
        raise V58TrainingError("V58r1 owner storage policy drift")
    return "owner_waiver"


def _owner_waiver_receipt(
    metadata: Mapping[str, Any], doctor: Mapping[str, Any]
) -> dict[str, Any]:
    backup = doctor.get("backup")
    runtime = metadata["contract"]["runtime_contract"]
    policy = runtime.get("backup_policy")
    if not isinstance(backup, Mapping) or not isinstance(policy, Mapping):
        raise V58TrainingError("V58 owner-waiver storage policy is missing")
    waiver = policy.get("waiver")
    if (
        backup.get("mode") != "owner_waiver"
        or backup.get("passed") is not True
        or backup.get("waiver_verified") is not True
        or not isinstance(waiver, Mapping)
        or backup.get("waiver_path") != waiver.get("path")
        or backup.get("waiver_sha256") != waiver.get("file_sha256")
    ):
        raise V58TrainingError("V58 owner storage waiver did not validate")
    receipt = {
        "version": "v58_backup_policy_receipt_v1",
        "phase": "v58",
        "mode": "owner_waiver",
        "verified": True,
        "waiver_path": waiver["path"],
        "waiver_sha256": waiver["file_sha256"],
        "waived_safeguards": list(policy["waived_safeguards"]),
        "external_input_backup_created": False,
        "external_code_backup_created": False,
        "external_checkpoint_backup_created": False,
    }
    receipt["policy_receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def _require_owner_waiver_receipt(
    metadata: Mapping[str, Any], doctor: Mapping[str, Any]
) -> dict[str, Any]:
    output: Path = metadata["output_dir"]
    observed = _load_json(
        output / "backup_policy_receipt.json", "V58 backup policy receipt"
    )
    expected = _owner_waiver_receipt(metadata, doctor)
    if observed != expected:
        raise V58TrainingError("V58 backup policy receipt drift")
    return observed


def _storage_evidence(
    metadata: Mapping[str, Any], doctor: Mapping[str, Any]
) -> tuple[str, Path, dict[str, Any]]:
    output: Path = metadata["output_dir"]
    if _storage_protection_mode(metadata) == "owner_waiver":
        receipt = _require_owner_waiver_receipt(metadata, doctor)
        return "backup_policy", output / "backup_policy_receipt.json", receipt
    receipt = _checkpoint_backup(metadata)
    return "checkpoint_backup", output / "checkpoint_backup_receipt.json", receipt


def _script_command(root: Path, relative: Path, *args: str) -> subprocess.CompletedProcess[str]:
    script = root / relative
    if not script.is_file():
        raise V58TrainingError(f"Missing V58 operator script: {script}")
    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(root / "src")
        if not existing
        else f"{root / 'src'}{os.pathsep}{existing}"
    )
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _require_operator_packet(
    output_dir: Path, name: str, operation: str
) -> dict[str, Any]:
    root = output_dir.resolve().parents[1]
    packet = output_dir / name
    if not packet.is_file():
        raise V58TrainingError(
            f"V58 {operation} requires its preceding operator packet: {packet}"
        )
    local = _load_json(packet, f"V58 {operation} operator packet")
    if local.get("operation") != operation:
        raise V58TrainingError(
            f"V58 operator packet operation differs from {operation}: {packet}"
        )
    result = _script_command(
        root,
        VALIDATOR_SCRIPT,
        "--repo-root",
        str(root),
        "--packet",
        str(packet),
    )
    if result.returncode != 0:
        raise V58TrainingError(
            f"Invalid V58 operator packet {packet.name}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    value = json.loads(result.stdout)
    if value.get("valid") is not True or value.get("operation") != operation:
        raise V58TrainingError(f"Operator packet operation drift: {packet}")
    return value


def _build_operator_packet(
    metadata: Mapping[str, Any],
    operation: str,
    *,
    evidence: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    root: Path = metadata["root"]
    output: Path = metadata["output_dir"]
    destination = output / f"operator_packet_{operation}.json"
    arguments = [
        "--repo-root",
        str(root),
        "--operation",
        operation,
        "--training-spec",
        str(output / "training_spec.json"),
        "--source-receipt",
        str(output / "source_receipt.json"),
        "--state",
        str(metadata["training"]["research_state"]),
        "--output",
        str(destination),
    ]
    for name, path in (evidence or {}).items():
        arguments.extend([f"--{name.replace('_', '-')}", str(path)])
    result = _script_command(root, PACKET_SCRIPT, *arguments)
    if result.returncode != 0:
        raise V58TrainingError(
            f"Failed to build V58 {operation} operator packet: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return _load_json(destination, f"V58 {operation} operator packet")


def _preflight_receipt(output: Path) -> dict[str, Any]:
    receipt = _load_json(output / "preflight.json", "V58 preflight")
    if receipt.get("passed") is not True:
        raise V58TrainingError("V58 preflight did not pass")
    return receipt


def _smoke_receipt(output: Path) -> dict[str, Any]:
    receipt = _load_json(output / "smoke.json", "V58 smoke")
    if receipt.get("passed") is not True:
        raise V58TrainingError("V58 smoke did not pass")
    return receipt


def _training_receipt(output: Path) -> dict[str, Any]:
    receipt = _load_json(output / "training_result.json", "V58 training result")
    if receipt.get("passed") is not True or receipt.get("checkpoint_count") != 36:
        raise V58TrainingError("V58 full training is not complete")
    return receipt


def _verification_receipt(output: Path) -> dict[str, Any]:
    receipt = _load_json(output / "verification.json", "V58 verification")
    snapshot = receipt.get("verification", receipt)
    if not isinstance(snapshot, dict) or snapshot.get("passed") is not True:
        raise V58TrainingError("V58 checkpoint verification did not pass")
    return receipt


def _base_access_ledger() -> dict[str, Any]:
    return {
        "development_evaluation_outcome_rows_read": 0,
        "outcome_rows_read": 0,
        "target_assets_loaded": [],
        "heldout_fold_symbols_loaded_by_job": {},
        "forbidden_columns_loaded": [],
        "previous_checkpoints_loaded": [],
        "predictions_written": False,
        "policy_actions_emitted": False,
        "performance_metrics_computed": False,
        "pnl_computed": False,
        "hyperparameters_changed": False,
        "authorized_panel_rows_by_job": {},
        "authorized_label_rows_by_job_and_role": {},
        "authorized_sequence_rows_by_job_and_role": {},
        "scaler_fit_rows_by_origin_geometry_fold": {},
        "optimizer_steps_by_job": {},
        "access_receipts_by_origin_geometry_fold": {},
    }


def _run_preflight(metadata: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    doctor = _doctor_or_raise(metadata)
    output: Path = metadata["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    input_receipt = {
        "version": "v58_input_hash_receipt_v1",
        "inputs": metadata["input_hashes"],
    }
    input_receipt["input_hash_receipt_sha256"] = canonical_sha256(input_receipt)
    preflight = {
        "version": "v58_preflight_v1",
        "passed": True,
        "decision": "authorize_v58_one_job_mps_smoke_only",
        "phase_contract_sha256": metadata["phase_contract"]["file_sha256"],
        "source_bundle_sha256": metadata["source_receipt"]["bundle_sha256"],
        "input_hash_receipt_sha256": input_receipt["input_hash_receipt_sha256"],
        "expected_jobs": metadata["job_ids"],
        "parameter_count": 465_513,
        "parquet_deserializations": 0,
        "model_instantiations": 0,
        "optimizer_steps": 0,
        "target_asset_loads": 0,
        "doctor": doctor,
    }
    write_json_atomic(output / "training_spec.json", metadata["training_spec"])
    write_json_atomic(output / "source_receipt.json", metadata["source_receipt"])
    write_json_atomic(output / "input_hash_receipt.json", input_receipt)
    if _storage_protection_mode(metadata) == "owner_waiver":
        write_json_atomic(
            output / "backup_policy_receipt.json",
            _owner_waiver_receipt(metadata, doctor),
        )
    write_yaml_atomic(output / "resolved_config.yaml", dict(config))
    write_json_atomic(output / "preflight.json", preflight)
    write_json_atomic(
        output / "grid_manifest.json",
        build_grid_manifest(metadata["contract"], []),
    )
    _build_operator_packet(metadata, "doctor")
    return {
        "decision": preflight["decision"],
        "audit": {"passed": True},
        "summary": {
            "expected_jobs": 36,
            "checkpoint_count": 0,
            "optimizer_steps": 0,
            "parquet_deserializations": 0,
        },
        "invocation": {"mode": "preflight", "new_jobs": 0, "new_optimizer_steps": 0},
    }


def _split_job_id(job_id: str) -> tuple[str, str, int, int]:
    origin, geometry, fold, seed = job_id.split("|")
    return origin, geometry, int(fold), int(seed)


def _cell_id(origin: str, geometry: str, fold: int) -> str:
    return f"{origin}|{geometry}|{fold}"


def _cell_directory(root: Path, origin: str, geometry: str, fold: int) -> Path:
    return root / origin / geometry / f"fold_{fold}"


def _job_directory(root: Path, origin: str, geometry: str, fold: int, seed: int) -> Path:
    return _cell_directory(root, origin, geometry, fold) / f"seed_{seed}"


def _scan_checkpoint_tree(
    checkpoint_root: Path,
    expected_job_ids: Iterable[str],
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Bind a completed prefix and at most one next-job resume artifact."""

    expected = list(expected_job_ids)
    if len(expected) != 36 or len(expected) != len(set(expected)):
        raise V58TrainingError("V58 checkpoint scan requires the exact 36-job grid")
    checkpoint_root = checkpoint_root.resolve()
    repository = (repo_root or checkpoint_root).resolve()
    expected_final = {
        job_id: _job_directory(checkpoint_root, *_split_job_id(job_id)) / "final.pt"
        for job_id in expected
    }
    expected_resume = {
        job_id: _job_directory(checkpoint_root, *_split_job_id(job_id)) / "resume.pt"
        for job_id in expected
    }
    known_final_paths = {path.resolve(): job_id for job_id, path in expected_final.items()}
    known_resume_paths = {path.resolve(): job_id for job_id, path in expected_resume.items()}
    observed_finals = sorted(checkpoint_root.rglob("final.pt")) if checkpoint_root.exists() else []
    observed_resumes = sorted(checkpoint_root.rglob("resume.pt")) if checkpoint_root.exists() else []
    unknown_finals = [
        path for path in observed_finals if path.resolve() not in known_final_paths
    ]
    unknown_resumes = [
        path for path in observed_resumes if path.resolve() not in known_resume_paths
    ]
    symlinks = [
        path for path in [*observed_finals, *observed_resumes] if path.is_symlink()
    ]
    temporary = sorted(checkpoint_root.rglob("*.tmp")) if checkpoint_root.exists() else []
    if unknown_finals or unknown_resumes or symlinks or temporary:
        raise V58TrainingError(
            "V58 checkpoint tree contains orphan artifacts: "
            + json.dumps(
                {
                    "final": [str(path) for path in unknown_finals],
                    "resume": [str(path) for path in unknown_resumes],
                    "symlink": [str(path) for path in symlinks],
                    "temporary": [str(path) for path in temporary],
                },
                sort_keys=True,
            )
        )
    if len(observed_resumes) > 1:
        raise V58TrainingError("V58 checkpoint tree contains multiple resume artifacts")
    completed = [job_id for job_id in expected if expected_final[job_id].is_file()]
    completed_count = len(completed)
    if completed != expected[:completed_count]:
        raise V58TrainingError("V58 completed checkpoints are not an exact grid prefix")
    pending_job: str | None = None
    pending_paths: list[str] = []
    if observed_resumes:
        resume_path = observed_resumes[0].resolve()
        pending_job = known_resume_paths[resume_path]
        if completed_count >= len(expected) or pending_job != expected[completed_count]:
            raise V58TrainingError(
                "V58 resume checkpoint is not bound to the next incomplete job"
            )
        if expected_final[pending_job].is_file():
            raise V58TrainingError("V58 completed job also has a resume checkpoint")
        try:
            pending_paths = [resume_path.relative_to(repository).as_posix()]
        except ValueError:
            pending_paths = [str(resume_path)]
    return {
        "completed_jobs": completed,
        "active_resume_artifacts": [],
        "pending_resume_artifacts": pending_paths,
        "pending_resume_job": pending_job,
        "orphan_resume_artifacts": [],
        "interrupted_resume_matched": False,
    }


def _model_factory(architecture: dict[str, Any]):
    def factory() -> StateConditionedMultiHorizonTransformer:
        model = StateConditionedMultiHorizonTransformer(architecture)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count != 465_513:
            raise V58TrainingError(
                f"V58 model parameter-count drift: {parameter_count}"
            )
        return model

    return factory


def _job_metadata(
    *,
    run_kind: str,
    job_id: str,
    origin: str,
    geometry: str,
    fold: int,
    seed: int,
    train_symbols: Iterable[str],
) -> dict[str, Any]:
    return {
        "version": "v58",
        "run_kind": run_kind,
        "job_id": job_id,
        "origin": origin,
        "geometry": geometry,
        "fold": int(fold),
        "seed": int(seed),
        "train_symbols": list(train_symbols),
        "prior_checkpoint_or_representation_reuse": False,
        "selected": False,
    }


def _read_and_bind_cell(
    metadata: Mapping[str, Any],
    *,
    origin: str,
    geometry: str,
    fold: int,
    checkpoint_root: Path,
) -> tuple[JobTrainingData, TrainOnlyScaler, dict[str, Any], dict[str, Any]]:
    contract = metadata["contract"]
    values = metadata["input_values"]
    paths = metadata["input_paths"]
    cell = build_job_cell(
        contract,
        values["v32_asset_folds"],
        values["v32_triplet_catalog"],
        origin=origin,
        geometry=geometry,
        fold=fold,
    )
    data = read_job_training_data(
        cell,
        sequence_path=paths["sequence_roles"],
        labels_path=paths["labels"],
        panel_path=paths["feature_panel"],
    )
    scaler = fit_job_train_only_scaler(data)
    cell_key = _cell_id(origin, geometry, fold)
    access = {
        "version": "v58_cell_data_access_v1",
        "cell_id": cell_key,
        "access_receipt": data.access_receipt,
    }
    access["data_access_sha256"] = canonical_sha256(access)
    scaler_payload = {
        "version": "v58_train_only_scaler_v1",
        "scaler_id": cell_key,
        "scaler": asdict(scaler),
    }
    cell_dir = _cell_directory(checkpoint_root, origin, geometry, fold)
    _write_or_verify_json(cell_dir / "data_access.json", access)
    _write_or_verify_json(cell_dir / "scaler.json", scaler_payload)
    return data, scaler, access, scaler_payload


def _load_bound_cell(
    checkpoint_root: Path, origin: str, geometry: str, fold: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    cell_dir = _cell_directory(checkpoint_root, origin, geometry, fold)
    access = _load_json(cell_dir / "data_access.json", "V58 cell data access")
    scaler = _load_json(cell_dir / "scaler.json", "V58 cell scaler")
    expected_access = dict(access)
    registered_access_hash = expected_access.pop("data_access_sha256", None)
    if registered_access_hash != canonical_sha256(expected_access):
        raise V58TrainingError(f"V58 cell data-access hash drift: {cell_dir}")
    registered_scaler = scaler.get("scaler", {}).get("scaler_sha256")
    if not isinstance(registered_scaler, str) or len(registered_scaler) != 64:
        raise V58TrainingError(f"V58 cell scaler hash missing: {cell_dir}")
    return access, scaler


def _expected_job_binding(
    metadata: Mapping[str, Any],
    checkpoint_root: Path,
    job_id: str,
) -> tuple[V58CheckpointContext, Path, dict[str, Any], dict[str, Any]]:
    origin, geometry, fold, seed = _split_job_id(job_id)
    access, scaler_payload = _load_bound_cell(
        checkpoint_root, origin, geometry, fold
    )
    train_symbols = scaler_payload["scaler"]["fit_symbols"]
    job_metadata = _job_metadata(
        run_kind="full",
        job_id=job_id,
        origin=origin,
        geometry=geometry,
        fold=fold,
        seed=seed,
        train_symbols=train_symbols,
    )
    context = V58CheckpointContext(
        scaler_sha256=scaler_payload["scaler"]["scaler_sha256"],
        data_access_sha256=access["data_access_sha256"],
        phase_contract_sha256=metadata["phase_contract"]["file_sha256"],
        source_bundle_sha256=metadata["source_receipt"]["bundle_sha256"],
        job_metadata=job_metadata,
    )
    final_path = _job_directory(
        checkpoint_root, origin, geometry, fold, seed
    ) / "final.pt"
    return context, final_path, access, scaler_payload


def _validate_checkpoint_row_binding(
    metadata: Mapping[str, Any],
    checkpoint_root: Path,
    row: Mapping[str, Any],
    job_id: str,
) -> tuple[V58CheckpointContext, Path, dict[str, Any], dict[str, Any]]:
    context, final_path, access, scaler_payload = _expected_job_binding(
        metadata, checkpoint_root, job_id
    )
    origin, geometry, fold, seed = _split_job_id(job_id)
    expected = {
        "job_id": job_id,
        "origin": origin,
        "geometry": geometry,
        "fold": fold,
        "seed": seed,
        "status": "completed",
        "checkpoint_path": _relative(metadata["root"], final_path),
        "scaler_sha256": context.scaler_sha256,
        "data_access_sha256": context.data_access_sha256,
        "phase_contract_sha256": context.phase_contract_sha256,
        "source_bundle_sha256": context.source_bundle_sha256,
        "job_metadata": dict(context.job_metadata),
    }
    drift = {
        key: {"expected": value, "observed": row.get(key)}
        for key, value in expected.items()
        if row.get(key) != value
    }
    if drift:
        raise V58TrainingError(
            f"V58 checkpoint manifest binding drift for {job_id}: "
            + json.dumps(drift, sort_keys=True)
        )
    if not final_path.is_file() or row.get("checkpoint_sha256") != file_sha256(
        final_path
    ):
        raise V58TrainingError(f"V58 checkpoint file/hash drift: {job_id}")
    if row.get("checkpoint_size_bytes") != final_path.stat().st_size:
        raise V58TrainingError(f"V58 checkpoint size drift: {job_id}")
    return context, final_path, access, scaler_payload


def _checkpoint_row(
    root: Path,
    result: Mapping[str, Any],
    *,
    job_id: str,
    origin: str,
    geometry: str,
    fold: int,
    seed: int,
    job_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = Path(str(result["checkpoint_path"]))
    if not checkpoint.is_file():
        raise V58TrainingError(f"V58 final checkpoint is missing: {checkpoint}")
    return {
        "job_id": job_id,
        "origin": origin,
        "geometry": geometry,
        "fold": int(fold),
        "seed": int(seed),
        "status": "completed",
        "checkpoint_path": _relative(root, checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "current_model_state_sha256": result["current_model_state_sha256"],
        "best_model_state_sha256": result["best_model_state_sha256"],
        "optimizer_state_sha256": result["optimizer_state_sha256"],
        "semantic_checkpoint_sha256": result["semantic_checkpoint_sha256"],
        "completed_epoch": int(result["completed_epoch"]),
        "optimizer_step_count": int(result["optimizer_step_count"]),
        "best_epoch": int(result["best_epoch"]),
        "best_validation_total_loss": float(
            result["best_validation_total_loss"]
        ),
        "scaler_sha256": result["scaler_sha256"],
        "data_access_sha256": result["data_access_sha256"],
        "phase_contract_sha256": result["phase_contract_sha256"],
        "source_bundle_sha256": result["source_bundle_sha256"],
        "job_metadata": dict(job_metadata),
    }


def _history_row(job_id: str, result: Mapping[str, Any]) -> dict[str, Any]:
    history = result["history"]
    return {
        "job_id": job_id,
        "history": history,
        "history_sha256": canonical_sha256(history),
        "sampler_receipts": result["sampler_receipts"],
    }


def _run_completed_grid_noop(
    metadata: Mapping[str, Any],
    checkpoint_manifest: Mapping[str, Any],
    *,
    device: str = "mps",
) -> dict[str, Any]:
    """Traverse the exact complete grid through the real zero-step skip path."""

    expected = metadata["job_ids"]
    rows = checkpoint_manifest.get("jobs")
    if (
        checkpoint_manifest.get("expected_jobs") != expected
        or not isinstance(rows, list)
        or [row.get("job_id") for row in rows] != expected
        or checkpoint_manifest.get("checkpoint_count") != 36
    ):
        raise V58TrainingError("V58 no-op replay requires the exact completed grid")
    contract = metadata["contract"]
    checkpoint_root = metadata["root"] / contract["access_contract"]["checkpoint_dir"]
    architecture = metadata["input_values"]["v55_blueprint"]["architecture"]
    optimizer_contract = contract["optimizer_and_early_stopping_contract"]
    receipts: list[dict[str, Any]] = []
    for job_id, row in zip(expected, rows, strict=True):
        context, final_path, _, _ = _validate_checkpoint_row_binding(
            metadata, checkpoint_root, row, job_id
        )
        before = file_sha256(final_path)
        result = run_v58_training_job(
            model_factory=_model_factory(architecture),
            batch_provider=_empty_provider,
            job_seed=int(row["seed"]),
            context=context,
            resume_path=final_path.with_name("resume.pt"),
            final_path=final_path,
            device=device,
            maximum_epochs=int(optimizer_contract["maximum_epochs"]),
            patience=int(optimizer_contract["early_stopping_patience"]),
        )
        after = file_sha256(final_path)
        if (
            result.get("status") != "already_complete"
            or result.get("completed") is not True
            or int(result.get("new_optimizer_steps", -1)) != 0
            or before != after
            or after != row.get("checkpoint_sha256")
            or result.get("current_model_state_sha256")
            != row.get("current_model_state_sha256")
            or result.get("best_model_state_sha256")
            != row.get("best_model_state_sha256")
            or result.get("optimizer_state_sha256")
            != row.get("optimizer_state_sha256")
            or result.get("semantic_checkpoint_sha256")
            != row.get("semantic_checkpoint_sha256")
        ):
            raise V58TrainingError(f"V58 completed-grid no-op drift: {job_id}")
        receipts.append(
            {
                "job_id": job_id,
                "checkpoint_sha256_before": before,
                "checkpoint_sha256_after": after,
                "new_optimizer_steps": 0,
                "status": result["status"],
            }
        )
    return {
        "new_jobs": 0,
        "new_optimizer_steps": 0,
        "rewritten_checkpoints": 0,
        "jobs": receipts,
    }


def _scaler_row(scaler_payload: Mapping[str, Any]) -> dict[str, Any]:
    scaler = dict(scaler_payload["scaler"])
    scaler["scaler_id"] = scaler_payload["scaler_id"]
    return scaler


def _empty_provider(*_args: Any, **_kwargs: Any) -> V58BatchStream:
    raise V58TrainingError("A completed V58 job unexpectedly requested data")


def _run_smoke(metadata: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    del config
    output: Path = metadata["output_dir"]
    _preflight_receipt(output)
    _require_operator_packet(output, "operator_packet_doctor.json", "doctor")
    _doctor_or_raise(metadata)
    existing = output / "smoke.json"
    if existing.is_file():
        smoke = _smoke_receipt(output)
        _require_operator_packet(output, "operator_packet_smoke.json", "smoke")
        return {
            "decision": smoke["decision"],
            "audit": {"passed": True},
            "summary": smoke["summary"],
            "invocation": {"mode": "smoke", "new_jobs": 0, "new_optimizer_steps": 0},
        }

    contract = metadata["contract"]
    smoke_contract = contract["smoke_contract"]
    origin = smoke_contract["origin"]
    geometry = smoke_contract["geometry"]
    fold = int(smoke_contract["fold"])
    seed = int(smoke_contract["seed"])
    checkpoint_root = metadata["root"] / contract["access_contract"][
        "smoke_checkpoint_dir"
    ]
    lock_path = metadata["root"] / contract["runtime_contract"]["process_lock"]
    work_dir = checkpoint_root / "equivalence"
    with _process_lock(lock_path, "smoke"):
        data, scaler, access, _ = _read_and_bind_cell(
            metadata,
            origin=origin,
            geometry=geometry,
            fold=fold,
            checkpoint_root=checkpoint_root,
        )
        job_id = f"{origin}|{geometry}|{fold}|{seed}"
        job_metadata = _job_metadata(
            run_kind="smoke",
            job_id=job_id,
            origin=origin,
            geometry=geometry,
            fold=fold,
            seed=seed,
            train_symbols=data.cell.train_symbols,
        )
        context = V58CheckpointContext(
            scaler_sha256=scaler.scaler_sha256,
            data_access_sha256=access["data_access_sha256"],
            phase_contract_sha256=metadata["phase_contract"]["file_sha256"],
            source_bundle_sha256=metadata["source_receipt"]["bundle_sha256"],
            job_metadata=job_metadata,
        )
        provider = V58MaterializedBatchProvider(
            data,
            scaler,
            origin=origin,
            geometry=geometry,
            fold=fold,
            job_seed=seed,
            train_samples=int(smoke_contract["train_samples_per_epoch"]),
            validation_samples=int(smoke_contract["fixed_validation_samples"]),
            batch_size=int(smoke_contract["batch_size"]),
        )
        equivalence = prove_v58_interrupted_resume_equivalence(
            model_factory=_model_factory(
                metadata["input_values"]["v55_blueprint"]["architecture"]
            ),
            batch_provider=provider,
            job_seed=seed,
            context=context,
            work_dir=work_dir,
            device="mps",
            maximum_epochs=int(smoke_contract["maximum_epochs"]),
            patience=int(smoke_contract["early_stopping_patience"]),
            interrupt_after_completed_epoch=int(
                smoke_contract["interrupt_after_completed_epoch"]
            ),
        )
        if equivalence.get("passed") is not True:
            raise V58TrainingError("V58 MPS interrupted-resume smoke failed")
        resume = {
            **EMPTY_RESUME,
            "interrupted_resume_matched": True,
        }
        smoke = {
            "version": "v58_smoke_v1",
            "passed": True,
            "decision": "authorize_v58_full_thirty_six_job_training_only",
            "job_id": job_id,
            "resume": resume,
            "equivalence": equivalence,
            "summary": {
                "checkpoint_executions": 2,
                "registered_jobs_covered": 1,
                "new_optimizer_steps": sum(
                    int(equivalence[name]["new_optimizer_steps"])
                    for name in ("uninterrupted", "interrupted", "resumed")
                ),
                "interrupted_resume_matched": True,
            },
        }
        smoke_access = _base_access_ledger()
        smoke_access["heldout_fold_symbols_loaded_by_job"] = {job_id: []}
        smoke_access["authorized_panel_rows_by_job"] = {
            job_id: access["access_receipt"]["authorized_panel_rows"]
        }
        smoke_access["authorized_label_rows_by_job_and_role"] = {
            job_id: access["access_receipt"]["train_validation_signal_key_counts"]
        }
        smoke_access["authorized_sequence_rows_by_job_and_role"] = {
            job_id: access["access_receipt"]["train_validation_signal_key_counts"]
        }
        smoke_access["scaler_fit_rows_by_origin_geometry_fold"] = {
            _cell_id(origin, geometry, fold): scaler.fit_unique_symbol_date_count
        }
        smoke_access["optimizer_steps_by_job"] = {
            job_id: smoke["summary"]["new_optimizer_steps"]
        }
        smoke_access["access_receipts_by_origin_geometry_fold"] = {
            _cell_id(origin, geometry, fold): access["access_receipt"]
        }
        smoke_access_manifest = build_data_access_manifest(smoke_access)
        write_json_atomic(output / "smoke.json", smoke)
        write_json_atomic(output / "smoke_data_access.json", smoke_access_manifest)
        if hasattr(torch, "mps"):
            torch.mps.empty_cache()
        del data, provider
        gc.collect()
    _build_operator_packet(
        metadata,
        "smoke",
        evidence={
            "smoke": output / "smoke.json",
            "data_access": output / "smoke_data_access.json",
        },
    )
    return {
        "decision": smoke["decision"],
        "audit": {"passed": True},
        "summary": smoke["summary"],
        "invocation": {
            "mode": "smoke",
            "new_jobs": 1,
            "new_optimizer_steps": smoke["summary"]["new_optimizer_steps"],
        },
    }


def _write_training_manifests(
    metadata: Mapping[str, Any],
    *,
    checkpoint_rows: list[dict[str, Any]],
    history_rows: list[dict[str, Any]],
    scaler_rows: list[dict[str, Any]],
    access_ledger: Mapping[str, Any],
    resume: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    output: Path = metadata["output_dir"]
    contract = metadata["contract"]
    checkpoint_manifest = build_checkpoint_manifest(
        contract, checkpoint_rows, resume=resume or EMPTY_RESUME
    )
    grid_manifest = build_grid_manifest(contract, checkpoint_rows)
    history_manifest = build_history_manifest(contract, history_rows)
    scaler_manifest = build_scaler_manifest(contract, scaler_rows)
    data_access = build_data_access_manifest(access_ledger)
    write_json_atomic(output / "checkpoint_manifest.json", checkpoint_manifest)
    write_json_atomic(output / "grid_manifest.json", grid_manifest)
    write_json_atomic(output / "history_manifest.json", history_manifest)
    write_json_atomic(output / "scaler_manifest.json", scaler_manifest)
    write_json_atomic(output / "data_access.json", data_access)
    return checkpoint_manifest, grid_manifest, history_manifest, scaler_manifest, data_access


def _run_full(metadata: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    del config
    output: Path = metadata["output_dir"]
    _preflight_receipt(output)
    _smoke_receipt(output)
    _require_operator_packet(output, "operator_packet_smoke.json", "smoke")
    _doctor_or_raise(metadata)
    contract = metadata["contract"]
    checkpoint_root = metadata["root"] / contract["access_contract"]["checkpoint_dir"]
    lock_path = metadata["root"] / contract["runtime_contract"]["process_lock"]
    architecture = metadata["input_values"]["v55_blueprint"]["architecture"]
    optimizer_contract = contract["optimizer_and_early_stopping_contract"]
    sampling_contract = contract["sampling_contract"]
    completed_noop: dict[str, Any] | None = None
    observed_tree = _scan_checkpoint_tree(
        checkpoint_root, metadata["job_ids"], repo_root=metadata["root"]
    )
    if observed_tree["completed_jobs"] == metadata["job_ids"]:
        with _process_lock(lock_path, "full"):
            locked_tree = _scan_checkpoint_tree(
                checkpoint_root, metadata["job_ids"], repo_root=metadata["root"]
            )
            if locked_tree["completed_jobs"] != metadata["job_ids"]:
                raise V58TrainingError("V58 checkpoint tree changed before no-op replay")
            existing_manifest = _load_json(
                output / "checkpoint_manifest.json", "V58 checkpoint manifest"
            )
            before = stable_replay_hashes(output)
            completed_noop = _run_completed_grid_noop(
                metadata, existing_manifest, device="mps"
            )
            after = stable_replay_hashes(output)
            if before != after:
                raise V58TrainingError(
                    "V58 completed full command changed a replay-stable artifact"
                )
    if completed_noop is not None:
        packet_path = output / "operator_packet_full.json"
        if packet_path.is_file():
            _require_operator_packet(output, packet_path.name, "full")
        else:
            _build_operator_packet(
                metadata,
                "full",
                evidence={
                    "data_access": output / "data_access.json",
                    "checkpoint_manifest": output / "checkpoint_manifest.json",
                },
            )
        stable = _training_receipt(output)
        return {
            "decision": stable["decision"],
            "audit": {"passed": True},
            "summary": {
                "checkpoint_count": 36,
                "scaler_count": 12,
                "total_optimizer_steps": stable["total_optimizer_steps"],
            },
            "invocation": {"mode": "full", **completed_noop},
        }
    checkpoint_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    scaler_rows: list[dict[str, Any]] = []
    access_ledger = _base_access_ledger()
    new_jobs = 0
    new_steps = 0
    cell_cache: dict[str, tuple[JobTrainingData | None, TrainOnlyScaler | None, dict[str, Any], dict[str, Any]]] = {}

    with _process_lock(lock_path, "full"):
        tree = _scan_checkpoint_tree(
            checkpoint_root, metadata["job_ids"], repo_root=metadata["root"]
        )
        for job_id in metadata["job_ids"]:
            origin, geometry, fold, seed = _split_job_id(job_id)
            cell_key = _cell_id(origin, geometry, fold)
            if cell_key not in cell_cache:
                cell_job_ids = [
                    item for item in metadata["job_ids"] if item.startswith(cell_key + "|")
                ]
                all_final = all(
                    (_job_directory(checkpoint_root, *_split_job_id(item)) / "final.pt").is_file()
                    for item in cell_job_ids
                )
                if all_final:
                    access, scaler_payload = _load_bound_cell(
                        checkpoint_root, origin, geometry, fold
                    )
                    data = None
                    scaler = None
                else:
                    data, scaler, access, scaler_payload = _read_and_bind_cell(
                        metadata,
                        origin=origin,
                        geometry=geometry,
                        fold=fold,
                        checkpoint_root=checkpoint_root,
                    )
                cell_cache[cell_key] = (data, scaler, access, scaler_payload)
                scaler_rows.append(_scaler_row(scaler_payload))
                receipt = access["access_receipt"]
                access_ledger["access_receipts_by_origin_geometry_fold"][cell_key] = receipt
                access_ledger["scaler_fit_rows_by_origin_geometry_fold"][cell_key] = int(
                    scaler_payload["scaler"]["fit_unique_symbol_date_count"]
                )
            data, scaler, access, scaler_payload = cell_cache[cell_key]
            train_symbols = scaler_payload["scaler"]["fit_symbols"]
            job_metadata = _job_metadata(
                run_kind="full",
                job_id=job_id,
                origin=origin,
                geometry=geometry,
                fold=fold,
                seed=seed,
                train_symbols=train_symbols,
            )
            context = V58CheckpointContext(
                scaler_sha256=scaler_payload["scaler"]["scaler_sha256"],
                data_access_sha256=access["data_access_sha256"],
                phase_contract_sha256=metadata["phase_contract"]["file_sha256"],
                source_bundle_sha256=metadata["source_receipt"]["bundle_sha256"],
                job_metadata=job_metadata,
            )
            job_dir = _job_directory(checkpoint_root, origin, geometry, fold, seed)
            final_path = job_dir / "final.pt"
            was_complete = final_path.is_file()
            if tree.get("pending_resume_job") == job_id:
                _write_training_manifests(
                    metadata,
                    checkpoint_rows=checkpoint_rows,
                    history_rows=history_rows,
                    scaler_rows=scaler_rows,
                    access_ledger=access_ledger,
                    resume={
                        key: tree[key]
                        for key in (
                            "active_resume_artifacts",
                            "pending_resume_artifacts",
                            "pending_resume_job",
                            "orphan_resume_artifacts",
                            "interrupted_resume_matched",
                        )
                    },
                )
            if data is None or scaler is None:
                provider: Any = _empty_provider
            else:
                provider = V58MaterializedBatchProvider(
                    data,
                    scaler,
                    origin=origin,
                    geometry=geometry,
                    fold=fold,
                    job_seed=seed,
                    train_samples=int(sampling_contract["train_samples_per_epoch"]),
                    validation_samples=int(sampling_contract["fixed_validation_samples"]),
                    batch_size=int(sampling_contract["batch_size"]),
                )
            result = run_v58_training_job(
                model_factory=_model_factory(architecture),
                batch_provider=provider,
                job_seed=seed,
                context=context,
                resume_path=job_dir / "resume.pt",
                final_path=final_path,
                device="mps",
                maximum_epochs=int(optimizer_contract["maximum_epochs"]),
                patience=int(optimizer_contract["early_stopping_patience"]),
            )
            if result.get("completed") is not True:
                raise V58TrainingError(f"V58 job did not complete: {job_id}")
            if tree.get("pending_resume_job") == job_id:
                tree = {
                    "completed_jobs": [*tree["completed_jobs"], job_id],
                    **EMPTY_RESUME,
                }
            row = _checkpoint_row(
                metadata["root"],
                result,
                job_id=job_id,
                origin=origin,
                geometry=geometry,
                fold=fold,
                seed=seed,
                job_metadata=job_metadata,
            )
            history = _history_row(job_id, result)
            _write_or_verify_json(job_dir / "complete.json", {"job": row, "history": history})
            checkpoint_rows.append(row)
            history_rows.append(history)
            if not was_complete:
                new_jobs += 1
            new_steps += int(result["new_optimizer_steps"])
            access_ledger["heldout_fold_symbols_loaded_by_job"][job_id] = []
            access_ledger["authorized_panel_rows_by_job"][job_id] = int(
                access["access_receipt"]["authorized_panel_rows"]
            )
            access_ledger["authorized_label_rows_by_job_and_role"][job_id] = access[
                "access_receipt"
            ]["train_validation_signal_key_counts"]
            access_ledger["authorized_sequence_rows_by_job_and_role"][job_id] = access[
                "access_receipt"
            ]["train_validation_signal_key_counts"]
            access_ledger["optimizer_steps_by_job"][job_id] = int(
                result["optimizer_step_count"]
            )
            _write_training_manifests(
                metadata,
                checkpoint_rows=checkpoint_rows,
                history_rows=history_rows,
                scaler_rows=scaler_rows,
                access_ledger=access_ledger,
            )
            print(
                json.dumps(
                    {
                        "v58_job": job_id,
                        "completed": len(checkpoint_rows),
                        "expected": 36,
                        "epochs": result["completed_epoch"],
                        "optimizer_steps": result["optimizer_step_count"],
                        "new_optimizer_steps": result["new_optimizer_steps"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if data is not None and seed == contract["grid_contract"]["seeds"][-1]:
                del data, provider
                cell_cache[cell_key] = (None, None, access, scaler_payload)
                gc.collect()
                if hasattr(torch, "mps"):
                    torch.mps.empty_cache()

        manifests = _write_training_manifests(
            metadata,
            checkpoint_rows=checkpoint_rows,
            history_rows=history_rows,
            scaler_rows=scaler_rows,
            access_ledger=access_ledger,
        )
        checkpoint_manifest, grid_manifest, history_manifest, scaler_manifest, data_access = manifests
        if not (
            checkpoint_manifest["checkpoint_count"] == 36
            and len(checkpoint_manifest["jobs"]) == 36
            and grid_manifest["counts"] == {
                "expected": 36,
                "completed": 36,
                "active": 0,
                "pending": 0,
            }
            and history_manifest["history_count"] == 36
            and scaler_manifest["scaler_count"] == 12
        ):
            raise V58TrainingError("V58 full grid manifest is incomplete")
        total_steps = sum(
            int(row["optimizer_step_count"]) for row in checkpoint_rows
        )
        training_result = {
            "version": "v58_training_result_v1",
            "passed": True,
            "decision": "v58_training_complete_v59_still_unauthorized_until_governor_registration",
            "checkpoint_count": 36,
            "scaler_count": 12,
            "history_count": 36,
            "total_optimizer_steps": total_steps,
            "new_jobs": new_jobs,
            "new_optimizer_steps": new_steps,
            "phase_contract_sha256": metadata["phase_contract"]["file_sha256"],
            "source_bundle_sha256": metadata["source_receipt"]["bundle_sha256"],
            "checkpoint_manifest_sha256": checkpoint_manifest["manifest_sha256"],
            "grid_manifest_sha256": grid_manifest["manifest_sha256"],
            "history_manifest_sha256": history_manifest["manifest_sha256"],
            "scaler_manifest_sha256": scaler_manifest["manifest_sha256"],
            "data_access_sha256": data_access["data_access_sha256"],
            "predictions_written": False,
            "performance_metrics_computed": False,
            "pnl_computed": False,
        }
        write_json_atomic(output / "training_result.json", training_result)
    _build_operator_packet(
        metadata,
        "full",
        evidence={
            "data_access": output / "data_access.json",
            "checkpoint_manifest": output / "checkpoint_manifest.json",
        },
    )
    return {
        "decision": training_result["decision"],
        "audit": {"passed": True},
        "summary": {
            "checkpoint_count": 36,
            "scaler_count": 12,
            "total_optimizer_steps": training_result["total_optimizer_steps"],
        },
        "invocation": {
            "mode": "full",
            "new_jobs": new_jobs,
            "new_optimizer_steps": new_steps,
        },
    }


def _checkpoint_backup(metadata: Mapping[str, Any]) -> dict[str, Any]:
    output: Path = metadata["output_dir"]
    receipt = _load_json(
        output / "checkpoint_backup_receipt.json", "V58 checkpoint backup"
    )
    if receipt.get("verified") is not True or len(receipt.get("checkpoints", [])) != 36:
        raise V58TrainingError("V58 external checkpoint backup is incomplete")
    validator_path = metadata["root"] / VALIDATOR_SCRIPT
    spec = importlib.util.spec_from_file_location(
        "tlm_v58_operator_validator", validator_path
    )
    if spec is None or spec.loader is None:
        raise V58TrainingError("Cannot load the V58 checkpoint-backup validator")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    try:
        validator.validate_checkpoint_backup(
            metadata["root"], receipt, metadata["job_ids"]
        )
    except Exception as exc:
        raise V58TrainingError(
            f"V58 external checkpoint backup validation failed: {exc}"
        ) from exc
    return receipt


def _run_verify(metadata: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    del config
    output: Path = metadata["output_dir"]
    _training_receipt(output)
    _require_operator_packet(output, "operator_packet_full.json", "full")
    doctor = _doctor_or_raise(metadata)
    protection_name, protection_path, _ = _storage_evidence(metadata, doctor)
    contract = metadata["contract"]
    checkpoint_manifest = _load_json(
        output / "checkpoint_manifest.json", "V58 checkpoint manifest"
    )
    manifest_body = dict(checkpoint_manifest)
    manifest_sha256 = manifest_body.pop("manifest_sha256", None)
    if (
        manifest_sha256 != canonical_sha256(manifest_body)
        or checkpoint_manifest.get("expected_jobs") != metadata["job_ids"]
        or [row.get("job_id") for row in checkpoint_manifest.get("jobs", [])]
        != metadata["job_ids"]
        or checkpoint_manifest.get("checkpoint_count") != 36
        or checkpoint_manifest.get("selected_jobs") != []
        or checkpoint_manifest.get("active_jobs") != []
    ):
        raise V58TrainingError("V58 verification grid drift")
    architecture = metadata["input_values"]["v55_blueprint"]["architecture"]
    optimizer_contract = contract["optimizer_and_early_stopping_contract"]
    lock_path = metadata["root"] / contract["runtime_contract"]["process_lock"]
    verified: list[str] = []
    receipts: list[dict[str, Any]] = []
    checkpoint_root = metadata["root"] / contract["access_contract"][
        "checkpoint_dir"
    ]
    with _process_lock(lock_path, "verify"):
        for expected_job_id, row in zip(
            metadata["job_ids"], checkpoint_manifest.get("jobs", []), strict=True
        ):
            job_id = row["job_id"]
            if job_id != expected_job_id:
                raise V58TrainingError("V58 verification job order drift")
            context, checkpoint_path, _, _ = _validate_checkpoint_row_binding(
                metadata, checkpoint_root, row, job_id
            )
            receipt = verify_v58_checkpoint_roundtrip(
                checkpoint_path,
                model_factory=_model_factory(architecture),
                job_seed=int(row["seed"]),
                context=context,
                checkpoint_kind="final",
                device="mps",
                maximum_epochs=int(optimizer_contract["maximum_epochs"]),
                patience=int(optimizer_contract["early_stopping_patience"]),
            )
            if (
                receipt.get("passed") is not True
                or receipt.get("checkpoint_file_sha256")
                != row.get("checkpoint_sha256")
                or receipt.get("semantic_checkpoint_sha256")
                != row.get("semantic_checkpoint_sha256")
                or receipt.get("current_model_state_sha256")
                != row.get("current_model_state_sha256")
                or receipt.get("best_model_state_sha256")
                != row.get("best_model_state_sha256")
                or receipt.get("optimizer_state_sha256")
                != row.get("optimizer_state_sha256")
                or receipt.get("scaler_sha256") != context.scaler_sha256
                or receipt.get("data_access_sha256") != context.data_access_sha256
                or receipt.get("phase_contract_sha256")
                != context.phase_contract_sha256
                or receipt.get("source_bundle_sha256")
                != context.source_bundle_sha256
                or receipt.get("job_metadata") != dict(context.job_metadata)
                or int(receipt.get("completed_epoch", -1))
                != int(row.get("completed_epoch", -2))
                or int(receipt.get("optimizer_step_count", -1))
                != int(row.get("optimizer_step_count", -2))
            ):
                raise V58TrainingError(f"V58 checkpoint roundtrip failed: {job_id}")
            verified.append(job_id)
            receipts.append({"job_id": job_id, **receipt})
        resume_files = sorted(
            _relative(metadata["root"], path)
            for path in checkpoint_root.rglob("resume.pt")
        )
        if resume_files:
            raise V58TrainingError(f"V58 verification found resume artifacts: {resume_files}")
        verification = {
            "version": "v58_verification_v1",
            "verification": {
                "passed": verified == metadata["job_ids"],
                "checkpoint_jobs_verified": verified,
                "all_checkpoints_retained": verified == metadata["job_ids"],
                "checkpoint_roundtrip_passed": all(
                    receipt["passed"] for receipt in receipts
                ),
                "resume_artifacts": resume_files,
                "optimizer_steps": 0,
                "parquet_deserializations": 0,
                "receipts": receipts,
            },
        }
        if verification["verification"]["passed"] is not True:
            raise V58TrainingError("V58 verification did not cover the exact grid")
        write_json_atomic(output / "verification.json", verification)
    verify_evidence = {
        "data_access": output / "data_access.json",
        "checkpoint_manifest": output / "checkpoint_manifest.json",
        "verification": output / "verification.json",
        protection_name: protection_path,
    }
    _build_operator_packet(
        metadata,
        "verify",
        evidence=verify_evidence,
    )
    return {
        "decision": "v58_checkpoints_verified_v59_still_unauthorized",
        "audit": {"passed": True},
        "summary": {"checkpoint_count": 36, "verified_checkpoints": 36},
        "invocation": {"mode": "verify", "new_jobs": 0, "new_optimizer_steps": 0},
    }


def _build_replay_receipt(
    before_hashes: Mapping[str, str],
    after_hashes: Mapping[str, str],
    *,
    checkpoint_hashes_before: Mapping[str, str] | None = None,
    checkpoint_hashes_after: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    checkpoints_before = dict(checkpoint_hashes_before or {})
    checkpoints_after = dict(checkpoint_hashes_after or checkpoints_before)
    hashes_match = dict(before_hashes) == dict(after_hashes)
    checkpoints_match = checkpoints_before == checkpoints_after
    return {
        "version": "v58_replay_v1",
        "replay": {
            "passed": hashes_match and checkpoints_match,
            "new_jobs": 0,
            "new_optimizer_steps": 0,
            "rewritten_checkpoints": 0,
            "artifact_hashes_match": hashes_match,
            "checkpoint_hashes_match": checkpoints_match,
            "stable_artifact_hashes_before": dict(before_hashes),
            "stable_artifact_hashes_after": dict(after_hashes),
            "checkpoint_hashes_before": checkpoints_before,
            "checkpoint_hashes_after": checkpoints_after,
        },
    }


def _final_report(result: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# TLM V58 Frozen Non-Target Training",
            "",
            "## Decision",
            "",
            f"**{result['decision']}**",
            "",
            "- 36/36 registered checkpoints retained and verified.",
            "- 12 train-only scalers retained; no job, seed, fold, origin, or geometry selected.",
            "- Replay created zero jobs and zero optimizer steps.",
            "- Development-evaluation outcomes, predictions, policy actions, performance, PnL, BTC, ETH, and SOL remained sealed.",
            "",
            "V59 remains a separate governed phase; this packet contains no economic result.",
            "",
        ]
    )


def _build_completion_receipt(
    metadata: Mapping[str, Any], output: Path, decision: str
) -> dict[str, Any]:
    protection_mode = _storage_protection_mode(metadata)
    protection_artifact = (
        "backup_policy_receipt.json"
        if protection_mode == "owner_waiver"
        else "checkpoint_backup_receipt.json"
    )
    completion: dict[str, Any] = {
        "version": (
            "v58r1_completion_receipt_v1"
            if protection_mode == "owner_waiver"
            else "v58_completion_receipt_v1"
        ),
        "decision": decision,
        "result_file_sha256": file_sha256(output / "result.json"),
        "audit_file_sha256": file_sha256(output / "audit.json"),
        "training_result_file_sha256": file_sha256(output / "training_result.json"),
        "verification_file_sha256": file_sha256(output / "verification.json"),
        "replay_file_sha256": file_sha256(output / "replay.json"),
        "checkpoint_manifest_file_sha256": file_sha256(
            output / "checkpoint_manifest.json"
        ),
    }
    if protection_mode == "owner_waiver":
        completion.update(
            {
                "storage_protection_artifact": protection_artifact,
                "storage_protection_file_sha256": file_sha256(
                    output / protection_artifact
                ),
            }
        )
    else:
        completion["checkpoint_backup_receipt_file_sha256"] = file_sha256(
            output / protection_artifact
        )
    completion["completion_receipt_sha256"] = canonical_sha256(completion)
    return completion


def _run_replay(metadata: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    del config
    output: Path = metadata["output_dir"]
    _verification_receipt(output)
    _require_operator_packet(output, "operator_packet_verify.json", "verify")
    doctor = _doctor_or_raise(metadata)
    protection_name, protection_path, _ = _storage_evidence(metadata, doctor)
    checkpoint_manifest = _load_json(
        output / "checkpoint_manifest.json", "V58 checkpoint manifest"
    )
    if (
        checkpoint_manifest.get("expected_jobs") != metadata["job_ids"]
        or checkpoint_manifest.get("checkpoint_count") != 36
    ):
        raise V58TrainingError("V58 replay requires the exact completed grid")
    checkpoint_hashes_before = {
        row["job_id"]: file_sha256(metadata["root"] / row["checkpoint_path"])
        for row in checkpoint_manifest["jobs"]
    }
    lock_path = metadata["root"] / metadata["contract"]["runtime_contract"][
        "process_lock"
    ]
    with _process_lock(lock_path, "replay"):
        tree = _scan_checkpoint_tree(
            metadata["root"]
            / metadata["contract"]["access_contract"]["checkpoint_dir"],
            metadata["job_ids"],
            repo_root=metadata["root"],
        )
        if tree["completed_jobs"] != metadata["job_ids"] or tree[
            "pending_resume_artifacts"
        ]:
            raise V58TrainingError("V58 replay found an incomplete checkpoint tree")
        before = stable_replay_hashes(output)
        observed_noop = _run_completed_grid_noop(
            metadata, checkpoint_manifest, device="mps"
        )
        if (
            observed_noop["new_jobs"] != 0
            or observed_noop["new_optimizer_steps"] != 0
            or observed_noop["rewritten_checkpoints"] != 0
        ):
            raise V58TrainingError("V58 replay no-op path created work")
        observed_jobs = observed_noop.get("jobs")
        if (
            not isinstance(observed_jobs, list)
            or [row.get("job_id") for row in observed_jobs] != metadata["job_ids"]
        ):
            raise V58TrainingError(
                "V58 replay no-op did not observe the exact ordered job grid"
            )
        manifest_by_job = {
            row["job_id"]: row for row in checkpoint_manifest["jobs"]
        }
        for observed in observed_jobs:
            job_id = observed["job_id"]
            expected_sha256 = manifest_by_job[job_id].get("checkpoint_sha256")
            if (
                int(observed.get("new_optimizer_steps", -1)) != 0
                or observed.get("checkpoint_sha256_before") != expected_sha256
                or observed.get("checkpoint_sha256_after") != expected_sha256
            ):
                raise V58TrainingError(
                    f"V58 replay no-op checkpoint receipt drift: {job_id}"
                )
        checkpoint_hashes_after = {
            row["job_id"]: file_sha256(metadata["root"] / row["checkpoint_path"])
            for row in checkpoint_manifest["jobs"]
        }
        after = stable_replay_hashes(output)
        replay = _build_replay_receipt(
            before,
            after,
            checkpoint_hashes_before=checkpoint_hashes_before,
            checkpoint_hashes_after=checkpoint_hashes_after,
        )
        replay["replay"]["observed_full_noop"] = observed_noop
        if replay["replay"]["passed"] is not True:
            raise V58TrainingError("V58 zero-step replay changed a stable artifact")
        write_json_atomic(output / "replay.json", replay)
    replay_evidence = {
        "data_access": output / "data_access.json",
        "checkpoint_manifest": output / "checkpoint_manifest.json",
        "verification": output / "verification.json",
        "replay": output / "replay.json",
        protection_name: protection_path,
    }
    _build_operator_packet(
        metadata,
        "replay",
        evidence=replay_evidence,
    )
    training = _training_receipt(output)
    result = {
        "version": "v58",
        "decision": metadata["contract"]["pass_action"],
        "candidate_family_id": metadata["contract"]["family_id"],
        "summary": {
            "checkpoint_count": 36,
            "scaler_count": 12,
            "total_optimizer_steps": training["total_optimizer_steps"],
            "verified_checkpoints": 36,
            "replay_new_jobs": 0,
            "replay_new_optimizer_steps": 0,
        },
        "tested": {
            "development_evaluation_outcomes_read": False,
            "target_assets_loaded": False,
            "predictions_written": False,
            "policy_actions_emitted": False,
            "performance_metrics_computed": False,
            "pnl_computed": False,
            "checkpoint_selection_executed": False,
        },
        "phase_contract_sha256": metadata["phase_contract"]["file_sha256"],
        "source_bundle_sha256": metadata["source_receipt"]["bundle_sha256"],
        "replay": replay["replay"],
    }
    audit_checks = {
        "exact_thirty_six_job_grid_completed": True,
        "all_checkpoints_retained_and_verified": True,
        "train_only_scalers_are_exact": True,
        "interrupted_resume_smoke_matched": _smoke_receipt(output)["resume"][
            "interrupted_resume_matched"
        ],
        "zero_step_replay_passed": replay["replay"]["passed"],
        "no_predictions_performance_or_pnl": True,
        "target_assets_remained_sealed": True,
        "v59_was_not_implemented_or_executed": True,
    }
    audit = {"passed": all(audit_checks.values()), "checks": audit_checks}
    result["audit"] = audit
    result["result_sha256"] = canonical_sha256(result)
    write_json_atomic(output / "result.json", result)
    write_json_atomic(output / "audit.json", audit)
    write_report_atomic(output / "report.md", _final_report(result))
    completion = _build_completion_receipt(metadata, output, result["decision"])
    write_json_atomic(output / "completion_receipt.json", completion)
    packet_files = list(metadata["contract"]["artifact_contract"]["packet_files"])
    required_without_manifest = [
        name for name in packet_files if name != "artifact_manifest.json"
    ]
    missing = [name for name in required_without_manifest if not (output / name).is_file()]
    if missing:
        raise V58TrainingError(f"V58 final artifact packet is incomplete: {missing}")
    manifest = build_artifact_manifest(
        output,
        [*required_without_manifest, "smoke_data_access.json"],
    )
    write_json_atomic(output / "artifact_manifest.json", manifest)
    return {
        "decision": result["decision"],
        "audit": audit,
        "summary": result["summary"],
        "invocation": {"mode": "replay", "new_jobs": 0, "new_optimizer_steps": 0},
    }


def run_state_conditioned_multi_horizon_training(
    config: dict[str, Any], *, mode: str
) -> dict[str, Any]:
    """Execute exactly one authorized V58 operation and stop at its gate."""

    if mode not in MODES:
        raise ValueError(f"V58 mode must be one of {MODES}; received {mode!r}")
    metadata = prepare_training_metadata(config)
    handlers = {
        "preflight": _run_preflight,
        "smoke": _run_smoke,
        "full": _run_full,
        "verify": _run_verify,
        "replay": _run_replay,
    }
    return handlers[mode](metadata, config)
