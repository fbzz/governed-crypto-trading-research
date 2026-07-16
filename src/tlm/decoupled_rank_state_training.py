"""Frozen V63 non-target training orchestration.

This module is intentionally training-only.  It cannot construct evaluation
predictions, positions, policy actions, performance metrics, or PnL.
"""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterator, Literal

import numpy as np
import pandas as pd
import pyarrow
import torch
import yaml

from .core.artifacts import canonical_sha256, file_sha256, write_json_atomic, write_yaml_atomic
from .decoupled_rank_state_training_data import read_fold_training_data
from .decoupled_rank_state_training_engine import (
    instantiate_models,
    run_training_job,
    verify_checkpoint,
)
from .research_workflow import research_doctor, validate_research_state


Mode = Literal["preflight", "smoke", "full", "verify", "replay"]
MODES = ("preflight", "smoke", "full", "verify", "replay")


class V63TrainingError(RuntimeError):
    pass


def _load_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise V63TrainingError(f"Missing V63 {name}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise V63TrainingError(f"V63 {name} must be an object")
    return value


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise V63TrainingError(f"V63 path escapes repository: {path}") from exc


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


@contextmanager
def _process_lock(path: Path, operation: Mode) -> Iterator[None]:
    if operation == "preflight":
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise V63TrainingError(f"V63 process lock is active: {path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"operation": operation, "pid": os.getpid()}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _git_receipt(root: Path, require_clean: bool) -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root,
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    if require_clean and status:
        raise V63TrainingError(
            f"V63 source receipt requires a clean Git tree; {len(status)} entries remain"
        )
    return {"head": head, "clean": not status, "changed_entry_count": len(status)}


def _context(config: dict[str, Any], *, require_source_receipt: bool) -> dict[str, Any]:
    training = config["decoupled_rank_state_training"]
    root = Path(training["project_root"]).resolve()
    state_path = root / training["research_state"]
    status = validate_research_state(root, state_path)
    if (
        status.get("authorized_phase") != "v63"
        or status.get("authorized_next_action")
        != "authorize_v63_frozen_non_target_decoupled_rank_state_training_only"
    ):
        raise V63TrainingError("Repository state does not authorize V63 training")
    contract_path = root / training["phase_contract"]
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict) or contract.get("phase") != "v63":
        raise V63TrainingError("V63 phase contract is invalid")
    contract_hash = file_sha256(contract_path)
    inputs = {name: root / value for name, value in training["inputs"].items()}
    configured_paths = {_relative(root, path) for path in inputs.values()}
    allowed_paths = set(contract["access_contract"]["allowed_inputs"])
    if configured_paths != allowed_paths or len(inputs) != len(allowed_paths):
        raise V63TrainingError("V63 config input allowlist differs from phase contract")
    expected = contract["input_contract"]["expected_file_sha256_by_path"]
    observed: dict[str, str] = {}
    for path in inputs.values():
        relative = _relative(root, path)
        if not path.is_file():
            raise V63TrainingError(f"Missing V63 input: {relative}")
        observed[relative] = file_sha256(path)
        if observed[relative] != expected[relative]:
            raise V63TrainingError(f"V63 input hash drift: {relative}")
    blueprint = _load_json(inputs["v60_blueprint"], "V60 blueprint")
    asset_folds = _load_json(inputs["v32_asset_folds"], "V32 asset folds")
    triplet_catalog = _load_json(inputs["v32_triplet_catalog"], "V32 triplet catalog")
    if (
        blueprint.get("candidate_family_id") != contract["family_id"]
        or len(asset_folds.get("folds", [])) != 3
        or len(triplet_catalog.get("folds", [])) != 3
    ):
        raise V63TrainingError("V63 scientific parent metadata drift")
    output = root / config["output_dir"]
    checkpoint_root = root / config["checkpoint_dir"]
    smoke_root = root / config["smoke_checkpoint_dir"]
    source_receipt: dict[str, Any] | None = None
    if require_source_receipt:
        git = _git_receipt(root, bool(training.get("require_clean_git", True)))
        source_files = list(training["source_receipt_files"])
        if not source_files or len(source_files) != len(set(source_files)):
            raise V63TrainingError("V63 source receipt file list is empty or duplicated")
        hashes: dict[str, str] = {}
        for relative in source_files:
            path = root / relative
            if not path.is_file():
                raise V63TrainingError(f"Missing V63 source file: {relative}")
            hashes[relative] = file_sha256(path)
        source_receipt = {
            "schema_version": "v63-training-source-receipt/v1",
            "git_clean": git["clean"],
            "git_head": git["head"],
            "files": hashes,
            "bundle_sha256": canonical_sha256(hashes),
            "runtime": {
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "pyarrow": pyarrow.__version__,
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
        "blueprint": blueprint,
        "asset_folds": asset_folds,
        "triplet_catalog": triplet_catalog,
        "output": output,
        "checkpoint_root": checkpoint_root,
        "smoke_root": smoke_root,
        "source_receipt": source_receipt,
    }


def _storage_policy_receipt(context: dict[str, Any]) -> dict[str, Any]:
    policy = context["contract"]["runtime_contract"]["backup_policy"]
    waiver_ref = policy["waiver"]
    waiver_path = context["root"] / waiver_ref["path"]
    waiver = _load_json(waiver_path, "V63 owner storage waiver")
    verified = (
        file_sha256(waiver_path) == waiver_ref["file_sha256"]
        and waiver.get("phase") == "v63"
        and waiver.get("risk_acceptance") is True
        and waiver.get("waived_safeguards") == policy["waived_safeguards"]
    )
    if not verified:
        raise V63TrainingError("V63 owner storage waiver failed verification")
    value = {
        "version": "v63_backup_policy_receipt_v1",
        "phase": "v63",
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
    doctor = research_doctor(context["root"], context["training"]["research_state"])
    if doctor.get("full_training_ready") is not True:
        raise V63TrainingError(
            "V63 runtime doctor blocks execution: "
            + json.dumps(
                {
                    "warnings": doctor.get("warnings"),
                    "disk": doctor.get("disk"),
                    "runtime": doctor.get("runtime"),
                    "backup": doctor.get("backup"),
                },
                sort_keys=True,
            )
        )
    return doctor


def _write_operator_packet(
    context: dict[str, Any], operation: str, doctor: dict[str, Any], evidence: dict[str, Any]
) -> None:
    output = context["output"]
    packet = {
        "schema_version": "v63-training-operator-packet/v1",
        "phase": "v63",
        "operation": operation,
        "phase_contract_sha256": context["contract_hash"],
        "source_bundle_sha256": context["source_receipt"]["bundle_sha256"],
        "git_head": context["source_receipt"]["git_head"],
        "doctor": {
            "full_training_ready": doctor["full_training_ready"],
            "free_disk_bytes": doctor["disk"]["free_bytes"],
            "required_free_disk_bytes": doctor["disk"]["required_free_bytes"],
            "mps_available": doctor["runtime"]["mps_available"],
            "mps_operational": doctor["runtime"]["mps_operational"],
            "deterministic_algorithms": doctor["runtime"]["deterministic_algorithms"],
            "fallback_enabled": doctor["runtime"]["fallback_enabled"],
            "active_job_count": doctor["process_lock"]["active_job_count"],
            "process_lock": doctor["process_lock"]["path"],
            "backup_mode": doctor["backup"]["mode"],
            "backup_passed": doctor["backup"]["passed"],
        },
        "evidence": evidence,
    }
    packet["packet_sha256"] = canonical_sha256(packet)
    write_json_atomic(output / f"operator_packet_{operation}.json", packet)


def _preflight(context: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    output = context["output"]
    output.mkdir(parents=True, exist_ok=True)
    contract = context["contract"]
    grid = contract["grid_optimizer_and_runtime_contract"]
    expected_jobs = [
        f"{fold}|{seed}" for fold in grid["folds"] for seed in grid["seeds"]
    ]
    if len(expected_jobs) != 9 or len(set(expected_jobs)) != 9:
        raise V63TrainingError("V63 frozen grid is not exactly nine jobs")
    torch.manual_seed(42)
    ranker, gate = instantiate_models(context["blueprint"], torch.device("cpu"))
    parameter_counts = {
        "ranker": sum(p.numel() for p in ranker.parameters()),
        "state_gate": sum(p.numel() for p in gate.parameters()),
    }
    parameter_counts["total"] = sum(parameter_counts.values())
    del ranker, gate
    training_spec = {
        "schema_version": "v63-training-spec/v1",
        "phase_contract_sha256": context["contract_hash"],
        "contract": contract,
    }
    training_spec["training_spec_sha256"] = canonical_sha256(training_spec)
    input_receipt = {
        "schema_version": "v63-input-hash-receipt/v1",
        "files": context["input_hashes"],
        "bundle_sha256": canonical_sha256(context["input_hashes"]),
    }
    input_receipt["receipt_sha256"] = canonical_sha256(input_receipt)
    backup = _storage_policy_receipt(context)
    checks = {
        "exact_input_allowlist_and_hashes": set(context["input_hashes"])
        == set(contract["access_contract"]["allowed_inputs"]),
        "clean_committed_source_receipt": context["source_receipt"]["git_clean"] is True,
        "owner_storage_waiver_verified": backup["verified"] is True,
        "exact_nine_job_grid": expected_jobs
        == ["1|42", "1|7", "1|123", "2|42", "2|7", "2|123", "3|42", "3|7", "3|123"],
        "parameter_counts_exact": parameter_counts
        == {"ranker": 1_231_634, "state_gate": 27_489, "total": 1_259_123},
        "target_assets_sealed": contract["target_contract"]["status"] == "sealed"
        and contract["target_contract"]["target_assets_loaded"] == [],
        "training_only_boundary": "performance_metric_or_pnl"
        in contract["access_contract"]["forbidden_capabilities"],
    }
    if not all(checks.values()):
        raise V63TrainingError(f"V63 preflight audit failed: {checks}")
    result = {
        "schema_version": "v63-training-preflight/v1",
        "decision": "authorize_v63_one_job_mps_smoke_only",
        "audit": {"passed": True, "checks": checks},
        "expected_jobs": expected_jobs,
        "parameter_counts": parameter_counts,
        "parquet_files_deserialized": 0,
        "optimizer_steps": 0,
    }
    result["preflight_sha256"] = canonical_sha256(result)
    write_json_atomic(output / "training_spec.json", training_spec)
    write_json_atomic(output / "input_hash_receipt.json", input_receipt)
    write_json_atomic(output / "source_receipt.json", context["source_receipt"])
    write_json_atomic(output / "backup_policy_receipt.json", backup)
    write_json_atomic(output / "preflight.json", result)
    write_yaml_atomic(output / "resolved_config.yaml", config)
    return result


def _require_prior(context: dict[str, Any], name: str, decision: str | None = None) -> dict[str, Any]:
    value = _load_json(context["output"] / name, name)
    if value.get("audit", {}).get("passed") is not True:
        raise V63TrainingError(f"V63 prior gate did not pass: {name}")
    if decision is not None and value.get("decision") != decision:
        raise V63TrainingError(f"V63 prior gate decision drift: {name}")
    return value


def _job_context(context: dict[str, Any], data: Any, fold: int, seed: int) -> dict[str, Any]:
    return {
        "phase": "v63",
        "family_id": context["contract"]["family_id"],
        "job_id": f"{fold}|{seed}",
        "fold": int(fold),
        "seed": int(seed),
        "phase_contract_sha256": context["contract_hash"],
        "source_bundle_sha256": context["source_receipt"]["bundle_sha256"],
        "fold_scale_sha256": data.scale.record()["fold_scale_sha256"],
        "data_access_sha256": data.access_receipt["access_sha256"],
        "train_symbols": list(data.train_symbols),
        "heldout_symbols_loaded": [],
        "target_assets_loaded": [],
    }


def _smoke(context: dict[str, Any], doctor: dict[str, Any]) -> dict[str, Any]:
    smoke_path = context["output"] / "smoke.json"
    if smoke_path.is_file():
        existing = _load_json(smoke_path, "smoke")
        if existing.get("audit", {}).get("passed") is True:
            return existing
    _require_prior(context, "preflight.json", "authorize_v63_one_job_mps_smoke_only")
    data = read_fold_training_data(
        root=context["root"], phase_contract=context["contract"],
        asset_folds=context["asset_folds"], triplet_catalog=context["triplet_catalog"],
        fold=1,
    )
    job_context = _job_context(context, data, 1, 42)
    contract = context["contract"]
    smoke = contract["smoke_contract"]

    def execute(name: str, interrupt: tuple[str, int] | None = None) -> dict[str, Any]:
        directory = context["smoke_root"] / name
        return run_training_job(
            blueprint=context["blueprint"], contract=contract, data=data, seed=42,
            context=job_context, resume_path=directory / "job.resume.pt",
            final_path=directory / "job.final.pt", device="mps",
            pretraining_samples=int(smoke["train_samples_per_epoch"]),
            supervised_samples=int(smoke["train_samples_per_epoch"]),
            validation_samples=int(smoke["fixed_validation_samples"]),
            batch_size=int(smoke["batch_size"]),
            pretraining_epochs=int(smoke["maximum_pretraining_epochs"]),
            supervised_epochs=int(smoke["maximum_supervised_epochs"]),
            patience=int(smoke["early_stopping_patience"]),
            interrupt_at=interrupt,
        )

    uninterrupted = execute("uninterrupted")
    pre_interrupted = execute(
        "pretraining_interrupted",
        ("pretraining", int(smoke["pretraining_interrupt_after_completed_epoch"])),
    )
    pre_resumed = execute("pretraining_interrupted")
    supervised_interrupted = execute(
        "supervised_interrupted",
        ("supervised", int(smoke["supervised_interrupt_after_completed_epoch"])),
    )
    supervised_resumed = execute("supervised_interrupted")
    compare_fields = (
        "ranker_state_sha256", "gate_state_sha256", "optimizer_steps", "history"
    )
    pre_comparisons = {
        field: uninterrupted[field] == pre_resumed[field] for field in compare_fields
    }
    supervised_comparisons = {
        field: uninterrupted[field] == supervised_resumed[field] for field in compare_fields
    }
    checks = {
        "uninterrupted_completed": uninterrupted["completed"],
        "pretraining_interruption_observed": pre_interrupted["status"] == "interrupted",
        "pretraining_resume_completed": pre_resumed["completed"],
        "pretraining_resume_matches": all(pre_comparisons.values()),
        "supervised_interruption_observed": supervised_interrupted["status"] == "interrupted",
        "supervised_resume_completed": supervised_resumed["completed"],
        "supervised_resume_matches": all(supervised_comparisons.values()),
        "independent_module_steps_nonzero": all(
            uninterrupted["optimizer_steps"][key] > 0
            for key in ("pretraining", "ranker", "gate")
        ),
        "no_target_or_heldout_assets_loaded": data.access_receipt["target_assets_loaded"] == []
        and data.access_receipt["heldout_symbols_loaded"] == [],
    }
    if not all(checks.values()):
        raise V63TrainingError(f"V63 smoke failed: {checks}")
    result = {
        "schema_version": "v63-training-smoke/v1",
        "decision": "authorize_v63_full_nine_job_training_only",
        "audit": {"passed": True, "checks": checks},
        "pretraining_resume_comparisons": pre_comparisons,
        "supervised_resume_comparisons": supervised_comparisons,
        "uninterrupted": uninterrupted,
        "pretraining_interrupted": pre_interrupted,
        "pretraining_resumed": pre_resumed,
        "supervised_interrupted": supervised_interrupted,
        "supervised_resumed": supervised_resumed,
        "data_access": data.access_receipt,
        "fold_scale": data.scale.record(),
        "resume": {
            "interrupted_resume_matched": True,
            "active_resume_artifacts": [],
            "pending_resume_artifacts": [],
            "pending_resume_job": None,
            "orphan_resume_artifacts": [],
        },
    }
    result["smoke_sha256"] = canonical_sha256(result)
    write_json_atomic(smoke_path, result)
    _write_operator_packet(
        context, "smoke", doctor,
        {"smoke": {"path": _relative(context["root"], smoke_path), "sha256": file_sha256(smoke_path)}},
    )
    return result


def _full(context: dict[str, Any], doctor: dict[str, Any]) -> dict[str, Any]:
    existing_path = context["output"] / "training_result.json"
    if existing_path.is_file():
        existing = _load_json(existing_path, "training result")
        if existing.get("audit", {}).get("passed") is True:
            return existing
    _require_prior(context, "smoke.json", "authorize_v63_full_nine_job_training_only")
    contract = context["contract"]
    grid = contract["grid_optimizer_and_runtime_contract"]
    pretraining = contract["model_and_objective_contract"]["ranker"]["pretraining"]
    supervised = grid["supervised_training"]
    jobs: list[dict[str, Any]] = []
    access_rows: list[dict[str, Any]] = []
    scales: list[dict[str, Any]] = []
    contexts: dict[str, dict[str, Any]] = {}
    for fold in grid["folds"]:
        data = read_fold_training_data(
            root=context["root"], phase_contract=contract,
            asset_folds=context["asset_folds"], triplet_catalog=context["triplet_catalog"],
            fold=int(fold),
        )
        access_rows.append(data.access_receipt)
        scales.append(data.scale.record())
        fold_dir = context["checkpoint_root"] / f"fold_{fold}"
        write_json_atomic(fold_dir / "data_access.json", data.access_receipt)
        write_json_atomic(fold_dir / "fold_scale.json", data.scale.record())
        for seed in grid["seeds"]:
            job_context = _job_context(context, data, int(fold), int(seed))
            contexts[job_context["job_id"]] = job_context
            result = run_training_job(
                blueprint=context["blueprint"], contract=contract, data=data,
                seed=int(seed), context=job_context,
                resume_path=fold_dir / f"seed_{seed}.resume.pt",
                final_path=fold_dir / f"seed_{seed}.final.pt",
                device="mps",
                pretraining_samples=int(pretraining["train_samples_per_epoch"]),
                supervised_samples=int(supervised["train_samples_per_epoch"]),
                validation_samples=int(supervised["fixed_validation_samples"]),
                batch_size=int(supervised["batch_size"]),
                pretraining_epochs=int(pretraining["maximum_epochs"]),
                supervised_epochs=int(supervised["maximum_epochs"]),
                patience=int(supervised["early_stopping_patience"]),
            )
            jobs.append(result)
        del data
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    expected = [f"{fold}|{seed}" for fold in grid["folds"] for seed in grid["seeds"]]
    checks = {
        "exact_ordered_nine_job_grid": [job["job_id"] for job in jobs] == expected,
        "all_jobs_complete": all(job["completed"] for job in jobs),
        "all_module_optimizer_steps_nonzero": all(
            all(job["optimizer_steps"][key] > 0 for key in ("pretraining", "ranker", "gate"))
            for job in jobs
        ),
        "all_checkpoints_retained": all(Path(job["checkpoint_path"]).is_file() for job in jobs),
        "three_fold_scalers_only": len(scales) == 3
        and len({row["fold_scale_sha256"] for row in scales}) == 3,
        "no_target_or_heldout_assets_loaded": all(
            row["target_assets_loaded"] == [] and row["heldout_symbols_loaded"] == []
            for row in access_rows
        ),
        "no_prediction_performance_or_pnl": all(
            not row[key]
            for row in access_rows
            for key in ("predictions_written", "performance_metrics_computed", "pnl_computed")
        ),
    }
    if not all(checks.values()):
        raise V63TrainingError(f"V63 full training audit failed: {checks}")
    checkpoint_manifest = {
        "schema_version": "v63-checkpoint-manifest/v1",
        "jobs": [
            {
                "job_id": job["job_id"], "fold": job["fold"], "seed": job["seed"],
                "status": "completed", "path": _relative(context["root"], Path(job["checkpoint_path"])),
                "file_sha256": job["checkpoint_file_sha256"],
                "semantic_checkpoint_sha256": job["semantic_checkpoint_sha256"],
                "ranker_state_sha256": job["ranker_state_sha256"],
                "gate_state_sha256": job["gate_state_sha256"],
                "optimizer_steps": job["optimizer_steps"],
                "context": contexts[job["job_id"]],
            }
            for job in jobs
        ],
        "active_resume_artifacts": [],
        "orphan_resume_artifacts": [],
    }
    checkpoint_manifest["manifest_sha256"] = canonical_sha256(checkpoint_manifest)
    grid_manifest = {
        "schema_version": "v63-grid-manifest/v1", "expected_jobs": expected,
        "completed_jobs": [job["job_id"] for job in jobs], "selected_jobs": [],
    }
    grid_manifest["manifest_sha256"] = canonical_sha256(grid_manifest)
    history_manifest = {
        "schema_version": "v63-history-manifest/v1",
        "jobs": {job["job_id"]: job["history"] for job in jobs},
    }
    history_manifest["manifest_sha256"] = canonical_sha256(history_manifest)
    scaler_manifest = {"schema_version": "v63-scaler-manifest/v1", "folds": scales}
    scaler_manifest["manifest_sha256"] = canonical_sha256(scaler_manifest)
    data_access = {
        "schema_version": "v63-data-access-ledger/v1", "folds": access_rows,
        "development_evaluation_outcome_rows_read": 0, "target_assets_loaded": [],
        "heldout_fold_symbols_loaded_by_job": [], "forbidden_columns_loaded": [],
        "previous_checkpoints_loaded": [], "predictions_written": False,
        "policy_actions_emitted": False, "performance_metrics_computed": False,
        "pnl_computed": False, "hyperparameters_changed": False,
    }
    data_access["data_access_sha256"] = canonical_sha256(data_access)
    result = {
        "schema_version": "v63-training-result-intermediate/v1",
        "decision": "authorize_v63_checkpoint_verification_only",
        "audit": {"passed": True, "checks": checks},
        "jobs": jobs,
        "summary": {
            "completed_jobs": len(jobs),
            "checkpoint_count": len(jobs),
            "total_optimizer_steps": sum(
                sum(job["optimizer_steps"].values()) for job in jobs
            ),
            "predictions": 0, "performance_metrics": 0, "pnl_evaluations": 0,
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
    write_json_atomic(existing_path, result)
    _write_operator_packet(
        context, "full", doctor,
        {"checkpoint_manifest": {"path": _relative(context["root"], output / "checkpoint_manifest.json"), "sha256": file_sha256(output / "checkpoint_manifest.json")}},
    )
    return result


def _verify(context: dict[str, Any], doctor: dict[str, Any]) -> dict[str, Any]:
    existing_path = context["output"] / "verification.json"
    if existing_path.is_file():
        existing = _load_json(existing_path, "verification")
        if existing.get("audit", {}).get("passed") is True:
            return existing
    _require_prior(context, "training_result.json", "authorize_v63_checkpoint_verification_only")
    manifest = _load_json(context["output"] / "checkpoint_manifest.json", "checkpoint manifest")
    rows = []
    for job in manifest["jobs"]:
        path = context["root"] / job["path"]
        row = verify_checkpoint(
            path, blueprint=context["blueprint"], contract=context["contract"],
            context=job["context"], device="cpu",
        )
        rows.append(row)
    expected = [f"{fold}|{seed}" for fold in (1, 2, 3) for seed in (42, 7, 123)]
    checks = {
        "exact_grid_verified": [row["job_id"] for row in rows] == expected,
        "all_checkpoint_roundtrips_pass": all(row["passed"] for row in rows),
        "all_file_hashes_match_manifest": all(
            row["checkpoint_file_sha256"] == job["file_sha256"]
            for row, job in zip(rows, manifest["jobs"], strict=True)
        ),
        "all_semantic_hashes_match_manifest": all(
            row["semantic_checkpoint_sha256"] == job["semantic_checkpoint_sha256"]
            for row, job in zip(rows, manifest["jobs"], strict=True)
        ),
        "all_checkpoints_retained": len(rows) == 9,
        "no_resume_artifacts": not list(context["checkpoint_root"].rglob("*.resume.pt")),
    }
    if not all(checks.values()):
        raise V63TrainingError(f"V63 verification failed: {checks}")
    result = {
        "schema_version": "v63-training-verification/v1",
        "decision": "authorize_v63_zero_step_replay_only",
        "audit": {"passed": True, "checks": checks},
        "verification": {
            "checkpoint_jobs_verified": expected,
            "all_checkpoints_retained": True,
            "checkpoint_roundtrip_passed": True,
        },
        "jobs": rows,
    }
    result["verification_sha256"] = canonical_sha256(result)
    write_json_atomic(existing_path, result)
    _write_operator_packet(
        context, "verify", doctor,
        {"verification": {"path": _relative(context["root"], existing_path), "sha256": file_sha256(existing_path)}},
    )
    return result


def _finalize_packet(context: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    output = context["output"]
    training = _load_json(output / "training_result.json", "training result")
    verification = _load_json(output / "verification.json", "verification")
    checks = {
        "preflight_passed": _load_json(output / "preflight.json", "preflight")["audit"]["passed"],
        "smoke_passed": _load_json(output / "smoke.json", "smoke")["audit"]["passed"],
        "full_training_passed": training["audit"]["passed"],
        "verification_passed": verification["audit"]["passed"],
        "replay_passed": replay["audit"]["passed"],
        "exact_nine_checkpoints": training["summary"]["checkpoint_count"] == 9,
        "zero_predictions_metrics_and_pnl": all(
            training["summary"][key] == 0
            for key in ("predictions", "performance_metrics", "pnl_evaluations", "target_asset_loads")
        ),
        "target_assets_remain_sealed": context["contract"]["target_contract"]["status"] == "sealed",
    }
    decision = (
        "authorize_v64_frozen_adaptive_development_evaluation_only"
        if all(checks.values())
        else "keep_v64_and_later_unauthorized"
    )
    audit = {"schema_version": "v63-training-audit/v1", "passed": all(checks.values()), "checks": checks}
    audit["audit_sha256"] = canonical_sha256(audit)
    result = {
        "schema_version": "v63-decoupled-rank-state-training-result/v1",
        "family_id": context["contract"]["family_id"],
        "decision": decision,
        "evidence_tier": "causal_non_target_training_only",
        "summary": training["summary"],
        "training_result_sha256": training["training_result_sha256"],
        "verification_sha256": verification["verification_sha256"],
        "replay_sha256": replay["replay_sha256"],
        "audit": audit,
        "target_contract": context["contract"]["target_contract"],
    }
    result["result_sha256"] = canonical_sha256(result)
    report = "\n".join([
        "# V63 Decoupled Rank/State Training", "", f"Decision: **{decision}**", "",
        f"Completed checkpoints: **{training['summary']['checkpoint_count']}**",
        f"Total optimizer steps: **{training['summary']['total_optimizer_steps']:,}**", "",
        "The exact three-fold by three-seed grid trained with fresh in-phase masked",
        "pretraining. Ranker and state gate used disjoint parameters, optimizers,",
        "losses, gradients, and early-stopping monitors.", "",
        "BTC/ETH/SOL, evaluation outcomes, saved predictions, positions, performance",
        "metrics, and PnL remained sealed. V64 is authorized only as a separately",
        "governed adaptive development-evaluation phase.", "",
    ])
    write_json_atomic(output / "audit.json", audit)
    write_json_atomic(output / "result.json", result)
    _atomic_text(output / "report.md", report)
    packet_files = context["contract"]["artifact_contract"]["packet_files"]
    manifest_files = [name for name in packet_files if name not in {"artifact_manifest.json", "completion_receipt.json"}]
    artifact_manifest = {
        "schema_version": "v63-artifact-manifest/v1",
        "files": {name: file_sha256(output / name) for name in manifest_files},
    }
    artifact_manifest["artifact_manifest_sha256"] = canonical_sha256(artifact_manifest)
    write_json_atomic(output / "artifact_manifest.json", artifact_manifest)
    completion = {
        "schema_version": "v63-completion-receipt/v1",
        "family_id": context["contract"]["family_id"], "decision": decision,
        "audit_passed": audit["passed"], "checkpoint_count": 9,
        "result_file_sha256": file_sha256(output / "result.json"),
        "result_sha256": result["result_sha256"],
        "audit_file_sha256": file_sha256(output / "audit.json"),
        "artifact_manifest_file_sha256": file_sha256(output / "artifact_manifest.json"),
        "artifact_manifest_sha256": artifact_manifest["artifact_manifest_sha256"],
    }
    completion["completion_receipt_sha256"] = canonical_sha256(completion)
    write_json_atomic(output / "completion_receipt.json", completion)
    actual = sorted(path.name for path in output.iterdir() if path.is_file())
    if actual != sorted(packet_files):
        raise V63TrainingError(f"V63 artifact packet file-set drift: {actual}")
    if not audit["passed"]:
        raise V63TrainingError("V63 terminal audit failed")
    return result


def _replay(context: dict[str, Any], doctor: dict[str, Any]) -> dict[str, Any]:
    _require_prior(context, "verification.json", "authorize_v63_zero_step_replay_only")
    manifest_path = context["output"] / "checkpoint_manifest.json"
    manifest = _load_json(manifest_path, "checkpoint manifest")
    before = {job["job_id"]: file_sha256(context["root"] / job["path"]) for job in manifest["jobs"]}
    new_steps = 0
    new_jobs = 0
    for fold in (1, 2, 3):
        data = read_fold_training_data(
            root=context["root"], phase_contract=context["contract"],
            asset_folds=context["asset_folds"], triplet_catalog=context["triplet_catalog"],
            fold=fold,
        )
        for seed in (42, 7, 123):
            job = next(row for row in manifest["jobs"] if row["job_id"] == f"{fold}|{seed}")
            result = run_training_job(
                blueprint=context["blueprint"], contract=context["contract"], data=data,
                seed=seed, context=job["context"],
                resume_path=context["checkpoint_root"] / f"fold_{fold}" / f"seed_{seed}.resume.pt",
                final_path=context["root"] / job["path"], device="mps",
            )
            new_steps += int(result["new_optimizer_steps"])
            new_jobs += int(result["status"] != "already_complete")
        del data
        torch.mps.empty_cache()
    after = {job["job_id"]: file_sha256(context["root"] / job["path"]) for job in manifest["jobs"]}
    checks = {
        "new_jobs_zero": new_jobs == 0,
        "new_optimizer_steps_zero": new_steps == 0,
        "rewritten_checkpoints_zero": before == after,
        "all_nine_hashes_match": len(after) == 9 and all(
            after[job["job_id"]] == job["file_sha256"] for job in manifest["jobs"]
        ),
    }
    if not all(checks.values()):
        raise V63TrainingError(f"V63 zero-step replay failed: {checks}")
    replay = {
        "schema_version": "v63-training-replay/v1",
        "audit": {"passed": True, "checks": checks},
        "replay": {"new_jobs": 0, "new_optimizer_steps": 0, "rewritten_checkpoints": 0, "artifact_hashes_match": True},
        "checkpoint_hashes_before": before, "checkpoint_hashes_after": after,
    }
    replay["replay_sha256"] = canonical_sha256(replay)
    replay_path = context["output"] / "replay.json"
    write_json_atomic(replay_path, replay)
    _write_operator_packet(
        context, "replay", doctor,
        {"replay": {"path": _relative(context["root"], replay_path), "sha256": file_sha256(replay_path)}},
    )
    return _finalize_packet(context, replay)


def run_decoupled_rank_state_training(config: dict[str, Any], *, mode: Mode) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"Unsupported V63 training mode: {mode}")
    context = _context(config, require_source_receipt=True)
    lock_path = context["root"] / context["contract"]["runtime_contract"]["process_lock"]
    if mode == "preflight":
        return _preflight(context, config)
    doctor = _doctor_or_raise(context)
    with _process_lock(lock_path, mode):
        if mode == "smoke":
            return _smoke(context, doctor)
        if mode == "full":
            return _full(context, doctor)
        if mode == "verify":
            return _verify(context, doctor)
        return _replay(context, doctor)
