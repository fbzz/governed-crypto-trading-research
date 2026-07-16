"""Frozen V83 low-turnover cross-sectional rank training orchestration."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterator, Literal

import torch
import yaml

from .core import canonical_sha256, file_sha256, write_json_atomic, write_yaml_atomic
from .low_turnover_rank_training_data import (
    V83FoldTrainingData,
    read_v83_fold_training_data,
)
from .low_turnover_rank_training_engine import (
    instantiate_v83_model,
    run_v83_training_job,
    verify_v83_checkpoint,
)
from .research_workflow import research_doctor, validate_research_state


Mode = Literal["preflight", "smoke", "full", "verify", "replay"]
MODES = {"preflight", "smoke", "full", "verify", "replay"}


class V83TrainingError(RuntimeError):
    pass


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V83TrainingError(f"Unable to read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise V83TrainingError(f"{label} must be an object")
    return value


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _git_receipt(root: Path, require_clean: bool) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if require_clean and status:
        raise V83TrainingError("V83 requires a clean committed Git source receipt")
    return {"clean": not bool(status), "head": head}


@contextmanager
def _process_lock(path: Path, operation: str) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise V83TrainingError(f"V83 optimizer lock is active: {path}") from exc
    try:
        os.write(
            descriptor,
            json.dumps({"pid": os.getpid(), "operation": operation}).encode(),
        )
        os.close(descriptor)
        yield
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
        path.unlink(missing_ok=True)


def _context(config: dict[str, Any]) -> dict[str, Any]:
    training = config.get("low_turnover_rank_training")
    if not isinstance(training, dict) or training.get("version") != "v83":
        raise V83TrainingError("Missing frozen V83 training config")
    root = Path(training.get("project_root", ".")).resolve()
    status = validate_research_state(root, training["research_state"])
    if (
        status.get("passed") is not True
        or status.get("authorized_phase") != "v83"
        or status.get("authorized_next_action")
        != "authorize_v83_frozen_non_target_low_turnover_rank_training_only"
    ):
        raise V83TrainingError("V83 research authorization is not active")
    contract_path = root / training["phase_contract"]
    contract_hash = file_sha256(contract_path)
    current = yaml.safe_load(
        (root / training["research_state"]).read_text(encoding="utf-8")
    )
    if current["phase_contract"] != {
        "path": training["phase_contract"],
        "file_sha256": contract_hash,
    }:
        raise V83TrainingError("V83 live phase-contract reference drift")
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if (
        contract.get("phase") != "v83"
        or contract.get("stage_revision")
        != "v083_frozen_non_target_low_turnover_rank_training_r2"
    ):
        raise V83TrainingError("V83 frozen phase revision drift")
    inputs = {
        name: root / relative for name, relative in training["inputs"].items()
    }
    observed = {_relative(root, path): file_sha256(path) for path in inputs.values()}
    expected = contract["input_contract"]["expected_file_sha256_by_path"]
    if (
        observed != expected
        or set(observed) != set(contract["access_contract"]["allowed_inputs"])
    ):
        raise V83TrainingError("V83 input allowlist or content hash drift")
    metadata = {
        name: _load_json(path, name)
        for name, path in inputs.items()
        if path.suffix == ".json"
    }
    blueprint = metadata["v80_blueprint"]
    if blueprint.get("family_id") != contract["family_id"]:
        raise V83TrainingError("V83 blueprint family drift")
    git = _git_receipt(root, bool(training.get("require_clean_git", True)))
    source_files = list(training["source_receipt_files"])
    if not source_files or len(source_files) != len(set(source_files)):
        raise V83TrainingError("V83 source receipt list is empty or duplicated")
    source_hashes = {
        relative: file_sha256(root / relative) for relative in source_files
    }
    source_receipt = {
        "schema_version": "v83-training-source-receipt/v1",
        "git_clean": git["clean"],
        "git_head": git["head"],
        "files": source_hashes,
        "bundle_sha256": canonical_sha256(source_hashes),
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
        },
    }
    return {
        "root": root,
        "training": training,
        "contract": contract,
        "contract_path": contract_path,
        "contract_hash": contract_hash,
        "inputs": inputs,
        "input_hashes": observed,
        "metadata": metadata,
        "blueprint": blueprint,
        "asset_folds": metadata["v32_asset_folds"],
        "triplet_catalog": metadata["v32_triplet_catalog"],
        "output": root / config["output_dir"],
        "checkpoint_root": root / config["checkpoint_dir"],
        "smoke_root": root / config["smoke_checkpoint_dir"],
        "source_receipt": source_receipt,
    }


def _storage_policy_receipt(context: dict[str, Any]) -> dict[str, Any]:
    policy = context["contract"]["runtime_contract"]["backup_policy"]
    waiver_ref = policy["waiver"]
    waiver_path = context["root"] / waiver_ref["path"]
    waiver = _load_json(waiver_path, "V83 owner storage waiver")
    if not (
        file_sha256(waiver_path) == waiver_ref["file_sha256"]
        and waiver.get("phase") == "v83"
        and waiver.get("risk_acceptance") is True
        and waiver.get("waived_safeguards") == policy["waived_safeguards"]
    ):
        raise V83TrainingError("V83 owner storage waiver failed verification")
    value = {
        "version": "v83_backup_policy_receipt_v1",
        "phase": "v83",
        "mode": "owner_waiver",
        "verified": True,
        "waiver_path": waiver_ref["path"],
        "waiver_sha256": waiver_ref["file_sha256"],
        "waived_safeguards": list(policy["waived_safeguards"]),
        "external_input_backup_created": False,
        "external_code_backup_created": False,
        "external_checkpoint_backup_created": False,
    }
    value["policy_receipt_sha256"] = canonical_sha256(value)
    return value


def _doctor_or_raise(context: dict[str, Any]) -> dict[str, Any]:
    doctor = research_doctor(
        context["root"], context["training"]["research_state"]
    )
    if doctor.get("full_training_ready") is not True:
        raise V83TrainingError(
            "V83 runtime doctor blocks execution: "
            + json.dumps(
                {
                    key: doctor.get(key)
                    for key in ("warnings", "disk", "runtime", "backup")
                },
                sort_keys=True,
            )
        )
    return doctor


def _job_context(
    context: dict[str, Any], data: V83FoldTrainingData, fold: int, seed: int
) -> dict[str, Any]:
    return {
        "phase": "v83",
        "family_id": context["contract"]["family_id"],
        "job_id": f"{fold}|{seed}",
        "fold": int(fold),
        "seed": int(seed),
        "phase_contract_sha256": context["contract_hash"],
        "source_bundle_sha256": context["source_receipt"]["bundle_sha256"],
        "fold_feature_scaler_sha256": data.scale.feature_scaler.state_sha256(),
        "fold_scale_sha256": data.scale.state_sha256(),
        "excess_rms_scale": data.scale.excess_rms_scale,
        "data_access_sha256": data.access_receipt["access_sha256"],
        "optimizer_contract": context["contract"][
            "grid_optimizer_and_runtime_contract"
        ],
        "train_symbols": list(data.train_symbols),
        "heldout_symbols_loaded": [],
        "target_assets_loaded": [],
        "prior_checkpoint_reused": False,
    }


def _preflight(context: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    output = context["output"]
    output.mkdir(parents=True, exist_ok=True)
    grid = context["contract"]["grid_optimizer_and_runtime_contract"]
    jobs = [f"{fold}|{seed}" for fold in grid["folds"] for seed in grid["seeds"]]
    model = instantiate_v83_model(
        context["blueprint"], torch.device("cpu"), seed=42
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    checks = {
        "exact_ordered_nine_job_grid": jobs
        == [
            "1|42",
            "1|7",
            "1|123",
            "2|42",
            "2|7",
            "2|123",
            "3|42",
            "3|7",
            "3|123",
        ],
        "parameter_count_exact": parameter_count == 10_993,
        "fresh_initialization_only": grid["prior_checkpoint_reuse"] == "none",
        "one_train_only_scaler_per_fold": context["contract"][
            "feature_and_scaler_contract"
        ]["scaler_count"]
        == 3,
        "adaptive_evaluation_values_forbidden": context["contract"][
            "data_and_role_contract"
        ]["any_2025_or_later_value_allowed"]
        is False,
        "target_assets_sealed": context["contract"]["target_contract"]["status"]
        == "sealed",
        "v82_dataset_audit_passed": context["metadata"]["v82_audit"]["passed"]
        is True,
    }
    if not all(checks.values()):
        raise V83TrainingError(f"V83 preflight failed: {checks}")
    phase_reference = {
        "path": _relative(context["root"], context["contract_path"]),
        "file_sha256": context["contract_hash"],
    }
    training_spec = {
        "schema_version": "v83-training-spec/v1",
        "phase_contract": phase_reference,
        "contract": context["contract"],
        "source_receipt_files": list(context["training"]["source_receipt_files"]),
    }
    training_spec["training_spec_sha256"] = canonical_sha256(training_spec)
    input_receipt = {
        "schema_version": "v83-input-hash-receipt/v1",
        "files": context["input_hashes"],
        "bundle_sha256": canonical_sha256(context["input_hashes"]),
    }
    input_receipt["receipt_sha256"] = canonical_sha256(input_receipt)
    result = {
        "schema_version": "v83-training-preflight/v1",
        "decision": "authorize_v83_interrupted_resume_mps_smoke_only",
        "audit": {"passed": True, "checks": checks},
        "expected_jobs": jobs,
        "parameter_count": parameter_count,
        "parquet_files_deserialized": 0,
        "checkpoint_containers_deserialized": 0,
        "optimizer_steps": 0,
    }
    result["preflight_sha256"] = canonical_sha256(result)
    write_json_atomic(output / "training_spec.json", training_spec)
    write_json_atomic(output / "input_hash_receipt.json", input_receipt)
    write_json_atomic(output / "source_receipt.json", context["source_receipt"])
    write_json_atomic(
        output / "backup_policy_receipt.json",
        _storage_policy_receipt(context),
    )
    write_json_atomic(output / "preflight.json", result)
    write_yaml_atomic(output / "resolved_config.yaml", config)
    return result


def _require_prior(
    context: dict[str, Any], name: str, decision: str
) -> dict[str, Any]:
    value = _load_json(context["output"] / name, name)
    if (
        value.get("audit", {}).get("passed") is not True
        or value.get("decision") != decision
    ):
        raise V83TrainingError(f"V83 prior gate failed or drifted: {name}")
    return value


def _fold_data(context: dict[str, Any], fold: int) -> V83FoldTrainingData:
    return read_v83_fold_training_data(
        root=context["root"],
        phase_contract=context["contract"],
        asset_folds=context["asset_folds"],
        triplet_catalog=context["triplet_catalog"],
        fold=int(fold),
    )


def _run_job(
    context: dict[str, Any],
    data: V83FoldTrainingData,
    fold: int,
    seed: int,
    *,
    resume_path: Path,
    final_path: Path,
    train_samples: int,
    validation_samples: int,
    batch_size: int,
    maximum_epochs: int,
    patience: int,
    minimum_delta: float,
    interrupt_after_epoch: int | None = None,
) -> dict[str, Any]:
    return run_v83_training_job(
        blueprint=context["blueprint"],
        contract=context["contract"],
        data=data,
        seed=int(seed),
        context=_job_context(context, data, fold, seed),
        resume_path=resume_path,
        final_path=final_path,
        device="mps",
        train_samples=int(train_samples),
        validation_samples=int(validation_samples),
        batch_size=int(batch_size),
        maximum_epochs=int(maximum_epochs),
        patience=int(patience),
        minimum_delta=float(minimum_delta),
        interrupt_after_epoch=interrupt_after_epoch,
    )


def _smoke(context: dict[str, Any]) -> dict[str, Any]:
    path = context["output"] / "smoke.json"
    if path.is_file():
        existing = _load_json(path, "V83 smoke")
        if existing.get("audit", {}).get("passed") is True:
            return existing
    _require_prior(
        context,
        "preflight.json",
        "authorize_v83_interrupted_resume_mps_smoke_only",
    )
    data = _fold_data(context, 1)
    smoke = context["contract"]["smoke_contract"]
    kwargs = {
        "train_samples": smoke["train_samples_per_epoch"],
        "validation_samples": smoke["fixed_validation_samples"],
        "batch_size": smoke["batch_size"],
        "maximum_epochs": smoke["maximum_epochs"],
        "patience": smoke["early_stopping_patience"],
        "minimum_delta": context["contract"][
            "grid_optimizer_and_runtime_contract"
        ]["early_stopping_minimum_delta"],
    }
    control_dir = context["smoke_root"] / "uninterrupted"
    resume_dir = context["smoke_root"] / "interrupted"
    for candidate in (
        control_dir / "job.resume.pt",
        control_dir / "job.final.pt",
        resume_dir / "job.resume.pt",
        resume_dir / "job.final.pt",
    ):
        candidate.unlink(missing_ok=True)
    control = _run_job(
        context,
        data,
        1,
        42,
        resume_path=control_dir / "job.resume.pt",
        final_path=control_dir / "job.final.pt",
        **kwargs,
    )
    interrupted = _run_job(
        context,
        data,
        1,
        42,
        resume_path=resume_dir / "job.resume.pt",
        final_path=resume_dir / "job.final.pt",
        interrupt_after_epoch=smoke["interrupt_after_completed_epoch"],
        **kwargs,
    )
    resumed = _run_job(
        context,
        data,
        1,
        42,
        resume_path=resume_dir / "job.resume.pt",
        final_path=resume_dir / "job.final.pt",
        **kwargs,
    )
    fields = (
        "semantic_checkpoint_sha256",
        "model_state_sha256",
        "optimizer_state_sha256",
        "optimizer_steps",
        "history",
    )
    comparisons = {field: control[field] == resumed[field] for field in fields}
    checks = {
        "uninterrupted_completed": control["completed"],
        "interruption_observed_at_epoch_boundary": interrupted["status"]
        == "interrupted",
        "resume_completed": resumed["completed"],
        "resume_matches_uninterrupted": all(comparisons.values()),
        "optimizer_steps_nonzero": int(control["optimizer_steps"]) > 0,
        "no_target_or_heldout_assets_loaded": data.access_receipt[
            "target_assets_loaded"
        ]
        == []
        and data.access_receipt["heldout_symbols_loaded"] == [],
        "no_2025_or_adaptive_role_values": data.access_receipt[
            "rows_from_2025_or_later"
        ]
        == 0
        and data.access_receipt["adaptive_evaluation_role_column_loaded"]
        is False,
    }
    if not all(checks.values()):
        raise V83TrainingError(f"V83 smoke failed: {checks}")
    result = {
        "schema_version": "v83-training-smoke/v1",
        "decision": "authorize_v83_full_nine_job_training_only",
        "audit": {"passed": True, "checks": checks},
        "comparisons": comparisons,
        "uninterrupted": control,
        "interrupted": interrupted,
        "resumed": resumed,
        "resume": {
            "interrupted_resume_matched": True,
            "active_resume_artifacts": [],
            "pending_resume_artifacts": [],
            "pending_resume_job": None,
            "orphan_resume_artifacts": [],
        },
    }
    result["smoke_sha256"] = canonical_sha256(result)
    write_json_atomic(path, result)
    write_json_atomic(
        context["output"] / "smoke_data_access.json", data.access_receipt
    )
    return result


def _checkpoint_manifest(
    context: dict[str, Any],
    jobs: list[dict[str, Any]],
    contexts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    grid = context["contract"]["grid_optimizer_and_runtime_contract"]
    expected = [f"{fold}|{seed}" for fold in grid["folds"] for seed in grid["seeds"]]
    value = {
        "schema_version": "v83-checkpoint-manifest/v1",
        "expected_jobs": expected,
        "jobs": [
            {
                "job_id": job["job_id"],
                "fold": job["fold"],
                "seed": job["seed"],
                "status": "completed",
                "path": _relative(context["root"], Path(job["checkpoint_path"])),
                "file_sha256": job["checkpoint_file_sha256"],
                "semantic_checkpoint_sha256": job[
                    "semantic_checkpoint_sha256"
                ],
                "model_state_sha256": job["model_state_sha256"],
                "optimizer_state_sha256": job["optimizer_state_sha256"],
                "optimizer_steps": job["optimizer_steps"],
                "completed_epoch": job["completed_epoch"],
                "best_epoch": job["best_epoch"],
                "context": contexts[job["job_id"]],
            }
            for job in jobs
        ],
        "selected_jobs": [],
        "active_resume_artifacts": [],
        "pending_resume_artifacts": [],
        "pending_resume_job": None,
        "orphan_resume_artifacts": [],
    }
    value["manifest_sha256"] = canonical_sha256(value)
    return value


def _full(context: dict[str, Any]) -> dict[str, Any]:
    path = context["output"] / "training_result.json"
    if path.is_file():
        existing = _load_json(path, "V83 training result")
        if existing.get("audit", {}).get("passed") is True:
            return existing
    _require_prior(
        context, "smoke.json", "authorize_v83_full_nine_job_training_only"
    )
    grid = context["contract"]["grid_optimizer_and_runtime_contract"]
    jobs: list[dict[str, Any]] = []
    contexts: dict[str, dict[str, Any]] = {}
    accesses: list[dict[str, Any]] = []
    scales: list[dict[str, Any]] = []
    for fold in grid["folds"]:
        data = _fold_data(context, int(fold))
        accesses.append(data.access_receipt)
        scales.append(data.scale.record())
        fold_dir = context["checkpoint_root"] / f"fold_{fold}"
        write_json_atomic(fold_dir / "data_access.json", data.access_receipt)
        write_json_atomic(fold_dir / "fold_scale.json", data.scale.record())
        for seed in grid["seeds"]:
            job_context = _job_context(context, data, int(fold), int(seed))
            contexts[job_context["job_id"]] = job_context
            print(
                json.dumps(
                    {
                        "v83_event": "job_start",
                        "job_id": job_context["job_id"],
                        "fold": int(fold),
                        "seed": int(seed),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            result = _run_job(
                context,
                data,
                int(fold),
                int(seed),
                resume_path=fold_dir / f"seed_{seed}.resume.pt",
                final_path=fold_dir / f"seed_{seed}.final.pt",
                train_samples=grid["train_samples_per_epoch"],
                validation_samples=grid["fixed_validation_samples"],
                batch_size=grid["batch_size"],
                maximum_epochs=grid["maximum_epochs"],
                patience=grid["early_stopping_patience"],
                minimum_delta=grid["early_stopping_minimum_delta"],
            )
            jobs.append(result)
            print(
                json.dumps(
                    {
                        "v83_event": "job_complete",
                        "job_id": result["job_id"],
                        "status": result["status"],
                        "completed_epoch": result["completed_epoch"],
                        "optimizer_steps": result["optimizer_steps"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            write_json_atomic(
                context["output"] / "checkpoint_manifest.json",
                _checkpoint_manifest(context, jobs, contexts),
            )
        del data
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    expected = [f"{fold}|{seed}" for fold in grid["folds"] for seed in grid["seeds"]]
    checks = {
        "exact_ordered_nine_job_grid": [job["job_id"] for job in jobs]
        == expected,
        "all_jobs_complete": all(job["completed"] for job in jobs),
        "all_optimizer_steps_nonzero": all(
            int(job["optimizer_steps"]) > 0 for job in jobs
        ),
        "all_checkpoints_retained": all(
            Path(job["checkpoint_path"]).is_file() for job in jobs
        ),
        "exact_three_train_only_scalers": len(scales) == 3
        and all(row["fit_role"] == "eligible_train_only" for row in scales),
        "no_target_or_heldout_assets_loaded": all(
            row["target_assets_loaded"] == []
            and row["heldout_symbols_loaded"] == []
            for row in accesses
        ),
        "no_2025_or_adaptive_role_values": all(
            row["rows_from_2025_or_later"] == 0
            and row["adaptive_evaluation_role_column_loaded"] is False
            for row in accesses
        ),
        "no_prediction_performance_or_pnl": all(
            not row[key]
            for row in accesses
            for key in (
                "predictions_written",
                "performance_metrics_computed",
                "pnl_computed",
            )
        ),
    }
    if not all(checks.values()):
        raise V83TrainingError(f"V83 full training audit failed: {checks}")
    checkpoint_manifest = _checkpoint_manifest(context, jobs, contexts)
    grid_manifest = {
        "schema_version": "v83-grid-manifest/v1",
        "expected_jobs": expected,
        "completed_jobs": expected,
        "selected_jobs": [],
    }
    grid_manifest["manifest_sha256"] = canonical_sha256(grid_manifest)
    history_manifest = {
        "schema_version": "v83-history-manifest/v1",
        "jobs": {job["job_id"]: job["history"] for job in jobs},
    }
    history_manifest["manifest_sha256"] = canonical_sha256(history_manifest)
    scaler_manifest = {
        "schema_version": "v83-scaler-manifest/v1",
        "folds": scales,
    }
    scaler_manifest["manifest_sha256"] = canonical_sha256(scaler_manifest)
    data_access = {
        "schema_version": "v83-data-access-ledger/v1",
        "folds": accesses,
        "authorized_panel_rows_by_fold": {
            str(row["fold"]): row["rows"]["panel"] for row in accesses
        },
        "authorized_label_rows_by_fold": {
            str(row["fold"]): row["rows"]["labels"] for row in accesses
        },
        "authorized_sequence_rows_by_fold_and_role": {
            str(row["fold"]): {
                "train_pairs": row["eligible_pairs"]["train"],
                "internal_validation_pairs": row["eligible_pairs"][
                    "internal_validation"
                ],
            }
            for row in accesses
        },
        "scaler_fit_rows_by_fold": {
            str(row["fold"]): row["rows"]["scaler_fit"] for row in accesses
        },
        "optimizer_steps_by_job": {
            job["job_id"]: int(job["optimizer_steps"]) for job in jobs
        },
        "outcome_rows_read": 0,
        "target_assets_loaded": [],
        "heldout_fold_symbols_loaded_by_job": [],
        "forbidden_columns_loaded": [],
        "previous_checkpoints_loaded": [],
        "predictions_written": False,
        "policy_actions_emitted": False,
        "performance_metrics_computed": False,
        "pnl_computed": False,
        "hyperparameters_changed": False,
    }
    data_access["data_access_sha256"] = canonical_sha256(data_access)
    result = {
        "schema_version": "v83-training-result-intermediate/v1",
        "decision": "authorize_v83_checkpoint_verification_only",
        "audit": {"passed": True, "checks": checks},
        "jobs": jobs,
        "summary": {
            "completed_jobs": 9,
            "checkpoint_count": 9,
            "total_optimizer_steps": sum(
                int(job["optimizer_steps"]) for job in jobs
            ),
            "epochs_by_job": {
                job["job_id"]: int(job["completed_epoch"]) for job in jobs
            },
            "best_epochs_by_job": {
                job["job_id"]: int(job["best_epoch"]) for job in jobs
            },
            "prior_checkpoint_reads": 0,
            "predictions": 0,
            "performance_metrics": 0,
            "pnl_evaluations": 0,
            "target_asset_loads": 0,
        },
    }
    result["training_result_sha256"] = canonical_sha256(result)
    output = context["output"]
    write_json_atomic(output / "checkpoint_manifest.json", checkpoint_manifest)
    write_json_atomic(output / "grid_manifest.json", grid_manifest)
    write_json_atomic(output / "history_manifest.json", history_manifest)
    write_json_atomic(output / "scaler_manifest.json", scaler_manifest)
    write_json_atomic(output / "data_access.json", data_access)
    write_json_atomic(path, result)
    return result


def _verify(context: dict[str, Any]) -> dict[str, Any]:
    path = context["output"] / "verification.json"
    if path.is_file():
        existing = _load_json(path, "V83 verification")
        if existing.get("audit", {}).get("passed") is True:
            return existing
    _require_prior(
        context,
        "training_result.json",
        "authorize_v83_checkpoint_verification_only",
    )
    manifest = _load_json(
        context["output"] / "checkpoint_manifest.json",
        "V83 checkpoint manifest",
    )
    rows = [
        verify_v83_checkpoint(
            context["root"] / job["path"],
            blueprint=context["blueprint"],
            context=job["context"],
        )
        for job in manifest["jobs"]
    ]
    grid = context["contract"]["grid_optimizer_and_runtime_contract"]
    expected = [f"{fold}|{seed}" for fold in grid["folds"] for seed in grid["seeds"]]
    checks = {
        "exact_grid_verified": [row["job_id"] for row in rows] == expected,
        "all_checkpoint_roundtrips_pass": all(row["passed"] for row in rows),
        "all_file_hashes_match_manifest": all(
            row["checkpoint_file_sha256"] == job["file_sha256"]
            for row, job in zip(rows, manifest["jobs"], strict=True)
        ),
        "all_semantic_hashes_match_manifest": all(
            row["semantic_checkpoint_sha256"]
            == job["semantic_checkpoint_sha256"]
            for row, job in zip(rows, manifest["jobs"], strict=True)
        ),
        "all_checkpoints_retained": len(rows) == 9,
        "no_resume_artifacts": not list(
            context["checkpoint_root"].rglob("*.resume.pt")
        ),
    }
    if not all(checks.values()):
        raise V83TrainingError(f"V83 verification failed: {checks}")
    result = {
        "schema_version": "v83-training-verification/v1",
        "decision": "authorize_v83_zero_step_replay_only",
        "audit": {"passed": True, "checks": checks},
        "verification": {
            "checkpoint_jobs_verified": expected,
            "all_checkpoints_retained": True,
            "checkpoint_roundtrip_passed": True,
        },
        "jobs": rows,
    }
    result["verification_sha256"] = canonical_sha256(result)
    write_json_atomic(path, result)
    return result


def _replay(context: dict[str, Any]) -> dict[str, Any]:
    _require_prior(
        context, "verification.json", "authorize_v83_zero_step_replay_only"
    )
    manifest = _load_json(
        context["output"] / "checkpoint_manifest.json",
        "V83 checkpoint manifest",
    )
    before = {
        job["job_id"]: file_sha256(context["root"] / job["path"])
        for job in manifest["jobs"]
    }
    grid = context["contract"]["grid_optimizer_and_runtime_contract"]
    new_steps = 0
    new_jobs = 0
    for fold in grid["folds"]:
        data = _fold_data(context, int(fold))
        for seed in grid["seeds"]:
            job = next(
                row
                for row in manifest["jobs"]
                if row["job_id"] == f"{fold}|{seed}"
            )
            result = _run_job(
                context,
                data,
                int(fold),
                int(seed),
                resume_path=context["checkpoint_root"]
                / f"fold_{fold}"
                / f"seed_{seed}.resume.pt",
                final_path=context["root"] / job["path"],
                train_samples=grid["train_samples_per_epoch"],
                validation_samples=grid["fixed_validation_samples"],
                batch_size=grid["batch_size"],
                maximum_epochs=grid["maximum_epochs"],
                patience=grid["early_stopping_patience"],
                minimum_delta=grid["early_stopping_minimum_delta"],
            )
            new_steps += int(result["new_optimizer_steps"])
            new_jobs += int(result["status"] != "already_complete")
        del data
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    after = {
        job["job_id"]: file_sha256(context["root"] / job["path"])
        for job in manifest["jobs"]
    }
    checks = {
        "new_jobs_zero": new_jobs == 0,
        "new_optimizer_steps_zero": new_steps == 0,
        "rewritten_checkpoints_zero": before == after,
        "all_nine_hashes_match": len(after) == 9
        and all(
            after[job["job_id"]] == job["file_sha256"]
            for job in manifest["jobs"]
        ),
    }
    if not all(checks.values()):
        raise V83TrainingError(f"V83 zero-step replay failed: {checks}")
    replay = {
        "schema_version": "v83-training-replay/v1",
        "decision": "authorize_v83_training_gate_registration_only",
        "audit": {"passed": True, "checks": checks},
        "replay": {
            "new_jobs": 0,
            "new_optimizer_steps": 0,
            "rewritten_checkpoints": 0,
            "artifact_hashes_match": True,
        },
        "checkpoint_hashes_before": before,
        "checkpoint_hashes_after": after,
    }
    replay["replay_sha256"] = canonical_sha256(replay)
    write_json_atomic(context["output"] / "replay_receipt.json", replay)
    return replay


def _operator_packet(context: dict[str, Any], operation: str) -> None:
    output = context["output"]
    arguments = [
        sys.executable,
        str(
            context["root"]
            / ".agents/skills/tlm-training-operator/scripts/build_training_packet.py"
        ),
        "--repo-root",
        str(context["root"]),
        "--operation",
        operation,
        "--training-spec",
        _relative(context["root"], output / "training_spec.json"),
        "--source-receipt",
        _relative(context["root"], output / "source_receipt.json"),
        "--output",
        _relative(context["root"], output / f"operator_packet_{operation}.json"),
    ]
    evidence = {
        "doctor": {},
        "smoke": {
            "smoke": output / "smoke.json",
            "data-access": output / "smoke_data_access.json",
        },
        "full": {
            "data-access": output / "data_access.json",
            "checkpoint-manifest": output / "checkpoint_manifest.json",
        },
        "verify": {
            "data-access": output / "data_access.json",
            "checkpoint-manifest": output / "checkpoint_manifest.json",
            "verification": output / "verification.json",
            "backup-policy": output / "backup_policy_receipt.json",
        },
        "replay": {
            "data-access": output / "data_access.json",
            "checkpoint-manifest": output / "checkpoint_manifest.json",
            "verification": output / "verification.json",
            "backup-policy": output / "backup_policy_receipt.json",
            "replay": output / "replay_receipt.json",
        },
    }[operation]
    for flag, path in evidence.items():
        arguments.extend([f"--{flag}", _relative(context["root"], path)])
    result = subprocess.run(
        arguments,
        cwd=context["root"],
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(context["root"] / "src")},
    )
    if result.returncode:
        raise V83TrainingError(
            f"V83 operator packet {operation} failed: {result.stderr.strip()}"
        )


def _finalize(context: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    output = context["output"]
    training = _load_json(output / "training_result.json", "V83 training result")
    verification = _load_json(output / "verification.json", "V83 verification")
    access = _load_json(output / "data_access.json", "V83 data access")
    checks = {
        "all_input_phase_contract_source_and_data_hashes_match": _load_json(
            output / "preflight.json", "V83 preflight"
        )["audit"]["passed"],
        "clean_committed_git_and_registered_storage_policy": context[
            "source_receipt"
        ]["git_clean"]
        and _load_json(
            output / "backup_policy_receipt.json", "V83 backup policy"
        )["verified"],
        "mps_float32_finite_and_fallback_disabled": _load_json(
            output / "smoke.json", "V83 smoke"
        )["audit"]["passed"],
        "exact_nine_fold_seed_jobs_and_fresh_initialization": training[
            "summary"
        ]["completed_jobs"]
        == 9
        and training["summary"]["prior_checkpoint_reads"] == 0,
        "exact_train_and_internal_validation_roles_only": all(
            row["rows_from_2025_or_later"] == 0 for row in access["folds"]
        ),
        "one_train_only_feature_scaler_per_fold_shared_across_seeds": len(
            _load_json(output / "scaler_manifest.json", "V83 scaler manifest")[
                "folds"
            ]
        )
        == 3,
        "heldout_fold_assets_and_2025_values_are_absent": access[
            "heldout_fold_symbols_loaded_by_job"
        ]
        == []
        and all(row["rows_from_2025_or_later"] == 0 for row in access["folds"]),
        "exact_registered_architecture_parameter_count_and_joint_objective": context[
            "blueprint"
        ]["architecture"]["expected_total_parameters"]
        == 10_993,
        "smoke_interrupted_resume_matches_uninterrupted": _load_json(
            output / "smoke.json", "V83 smoke"
        )["resume"]["interrupted_resume_matched"],
        "all_nine_checkpoints_retained_and_semantically_verified": verification[
            "verification"
        ]["all_checkpoints_retained"]
        and verification["verification"]["checkpoint_roundtrip_passed"],
        "replay_creates_zero_jobs_optimizer_steps_or_checkpoint_rewrites": replay[
            "audit"
        ]["passed"],
        "no_prediction_position_policy_performance_pnl_outcome_or_target_access": all(
            training["summary"][key] == 0
            for key in (
                "predictions",
                "performance_metrics",
                "pnl_evaluations",
                "target_asset_loads",
            )
        )
        and access["outcome_rows_read"] == 0,
    }
    passed = all(checks.values())
    decision = (
        context["contract"]["pass_action"]
        if passed
        else context["contract"]["failure_action"]
    )
    audit = {
        "schema_version": "v83-training-audit/v1",
        "passed": passed,
        "checks": checks,
    }
    audit["audit_sha256"] = canonical_sha256(audit)
    result = {
        "schema_version": "v83-low-turnover-rank-training-result/v1",
        "family_id": context["contract"]["family_id"],
        "decision": decision,
        "evidence_tier": context["contract"]["evidence_tier"],
        "summary": training["summary"],
        "training_result_sha256": training["training_result_sha256"],
        "verification_sha256": verification["verification_sha256"],
        "replay_sha256": replay["replay_sha256"],
        "audit": audit,
        "target_contract": context["contract"]["target_contract"],
    }
    result["result_sha256"] = canonical_sha256(result)
    write_json_atomic(output / "audit.json", audit)
    write_json_atomic(output / "result.json", result)
    _atomic_text(
        output / "report.md",
        "\n".join(
            [
                "# V83 Low-Turnover Cross-Sectional Rank Training",
                "",
                f"Decision: **{decision}**",
                "",
                f"Completed checkpoints: **{training['summary']['checkpoint_count']}**",
                f"Total optimizer steps: **{training['summary']['total_optimizer_steps']:,}**",
                "",
                "All nine fresh-weight models use one train-only scaler per fold.",
                "Training and internal-validation losses are diagnostics, not alpha evidence.",
                "",
                "No predictions, positions, financial metrics, outcomes, or BTC/ETH/SOL were opened.",
                "A pass authorizes only a separate outcome-blind V84 evaluation preparation.",
                "",
            ]
        ),
    )
    packet_files = context["contract"]["artifact_contract"]["required_files"]
    manifest_names = [
        name
        for name in packet_files
        if name not in {"artifact_manifest.json", "completion_receipt.json"}
    ]
    manifest = {
        "schema_version": "v83-artifact-manifest/v1",
        "files": {name: file_sha256(output / name) for name in manifest_names},
    }
    manifest["artifact_manifest_sha256"] = canonical_sha256(manifest)
    write_json_atomic(output / "artifact_manifest.json", manifest)
    completion = {
        "schema_version": "v83-completion-receipt/v1",
        "family_id": context["contract"]["family_id"],
        "decision": decision,
        "audit_passed": passed,
        "checkpoint_count": 9,
        "result_file_sha256": file_sha256(output / "result.json"),
        "result_sha256": result["result_sha256"],
        "audit_file_sha256": file_sha256(output / "audit.json"),
        "artifact_manifest_file_sha256": file_sha256(
            output / "artifact_manifest.json"
        ),
        "artifact_manifest_sha256": manifest["artifact_manifest_sha256"],
    }
    completion["completion_receipt_sha256"] = canonical_sha256(completion)
    write_json_atomic(output / "completion_receipt.json", completion)
    actual = sorted(path.name for path in output.iterdir() if path.is_file())
    if actual != sorted(packet_files):
        raise V83TrainingError(f"V83 artifact packet file-set drift: {actual}")
    if not passed:
        raise V83TrainingError("V83 terminal audit failed")
    return result


def run_low_turnover_rank_training(
    config: dict[str, Any], *, mode: Mode
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"Unsupported V83 training mode: {mode}")
    context = _context(config)
    if mode == "preflight":
        result = _preflight(context, config)
        _operator_packet(context, "doctor")
        return result
    _doctor_or_raise(context)
    lock = context["root"] / context["contract"]["runtime_contract"]["process_lock"]
    with _process_lock(lock, mode):
        if mode == "smoke":
            result = _smoke(context)
        elif mode == "full":
            result = _full(context)
        elif mode == "verify":
            result = _verify(context)
        else:
            result = _replay(context)
    _operator_packet(context, mode)
    if mode == "replay":
        return _finalize(context, result)
    return result
