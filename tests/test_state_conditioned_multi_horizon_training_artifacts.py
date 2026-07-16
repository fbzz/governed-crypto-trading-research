from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest
import yaml

import tlm.state_conditioned_multi_horizon_training_artifacts as training_artifacts
from tlm.core.artifacts import canonical_sha256, file_sha256
from tlm.state_conditioned_multi_horizon_training_artifacts import (
    TrainingArtifactError,
    build_artifact_manifest,
    build_checkpoint_manifest,
    build_data_access_manifest,
    build_grid_manifest,
    build_history_manifest,
    build_scaler_manifest,
    exact_job_ids,
    prepare_training_metadata,
    resolve_repo_path,
    stable_replay_hashes,
    write_json_atomic,
    write_report_atomic,
    write_yaml_atomic,
    _verify_backup_only_revision,
)


FAMILY = "tlm_state_conditioned_multi_horizon_quantile_small_v1"
V57_ACTION = "authorize_v57_non_target_multi_horizon_dataset_build_only"
V58_ACTION = "authorize_v58_frozen_non_target_training_only"
V59_ACTION = "authorize_v59_frozen_adaptive_development_evaluation_only"


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _yaml(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _self(value: dict, field: str) -> dict:
    result = deepcopy(value)
    result[field] = canonical_sha256(result)
    return result


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _fixture(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "repo"
    root.mkdir()
    preceding = {
        "schema_version": 1,
        "phase": "v57",
        "family_id": FAMILY,
        "authorized_next_action": V57_ACTION,
        "pass_action": V58_ACTION,
    }
    preceding_path = root / "research/phase_contracts/v057.yaml"
    _yaml(preceding_path, preceding)
    preceding_hash = file_sha256(preceding_path)

    blueprint = _self(
        {
            "version": "v55",
            "candidate_family_id": FAMILY,
            "parameter_count_analytic": 465513,
            "registered_job_count": 36,
            "lifecycle": {"v57_pass_action": V58_ACTION, "v58_pass_action": V59_ACTION},
        },
        "blueprint_sha256",
    )
    harness_spec = _self(
        {
            "version": "v56",
            "candidate_family_id": FAMILY,
            "authorized_next_action": V57_ACTION,
            "v55_blueprint_sha256": blueprint["blueprint_sha256"],
        },
        "harness_spec_sha256",
    )
    v56_audit = {"passed": True, "checks": {"synthetic_only": True}}
    v56_result = _self(
        {
            "version": "v56",
            "candidate_family_id": FAMILY,
            "decision": V57_ACTION,
            "audit": v56_audit,
            "harness_spec": harness_spec,
        },
        "result_sha256",
    )
    dataset_spec = _self(
        {
            "version": "v57",
            "candidate_family_id": FAMILY,
            "phase_contract_file_sha256": preceding_hash,
            "pass_action": V58_ACTION,
            "failure_action": "keep_v58_and_later_unauthorized",
        },
        "dataset_spec_sha256",
    )
    v57_audit = {"passed": True, "checks": {"target_assets_are_absent": True}}
    v57_result = _self(
        {
            "version": "v57",
            "candidate_family_id": FAMILY,
            "decision": V58_ACTION,
            "audit": v57_audit,
            "dataset_spec": dataset_spec,
        },
        "result_sha256",
    )
    label_schema = _self({"version": "v57", "columns": ["date", "symbol"]}, "label_schema_sha256")
    source_files = {"src/old.py": "a" * 64}
    source_receipt = {"files": source_files, "bundle_sha256": canonical_sha256(source_files)}

    input_paths = {
        "v55_blueprint": "artifacts/v55/blueprint.json",
        "v56_result": "artifacts/v56/result.json",
        "v56_audit": "artifacts/v56/audit.json",
        "v56_harness_spec": "artifacts/v56/harness_spec.json",
        "v57_result": "artifacts/v57/result.json",
        "v57_audit": "artifacts/v57/audit.json",
        "v57_dataset_spec": "artifacts/v57/dataset_spec.json",
        "v57_dataset_manifest": "artifacts/v57/dataset_manifest.json",
        "v57_label_schema": "artifacts/v57/label_schema.json",
        "v57_source_receipt": "artifacts/v57/source_receipt.json",
        "v57_completion_receipt": "artifacts/v57/completion_receipt.json",
        "v57_artifact_manifest": "artifacts/v57/artifact_manifest.json",
        "v57_data_access": "artifacts/v57/data_access.json",
        "v32_feature_schema": "artifacts/v32/feature_schema.json",
        "v32_asset_folds": "artifacts/v32/asset_folds.json",
        "v32_triplet_catalog": "artifacts/v32/triplet_catalog.json",
        "feature_panel": "data/panel.parquet",
        "labels": "data/labels.parquet",
        "sequence_roles": "data/roles.parquet",
    }
    values: dict[str, object] = {
        "v55_blueprint": blueprint,
        "v56_result": v56_result,
        "v56_audit": v56_audit,
        "v56_harness_spec": harness_spec,
        "v57_result": v57_result,
        "v57_audit": v57_audit,
        "v57_dataset_spec": dataset_spec,
        "v57_label_schema": label_schema,
        "v57_source_receipt": source_receipt,
        "v57_data_access": {"target_assets_loaded": []},
        "v32_feature_schema": {},
        "v32_asset_folds": {},
        "v32_triplet_catalog": {},
    }
    for name in ("feature_panel", "labels", "sequence_roles"):
        path = root / input_paths[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not-a-parquet-table-" + name.encode())
    dataset_manifest = {
        "labels": {"path": input_paths["labels"]},
        "sequence_roles": {"path": input_paths["sequence_roles"]},
    }
    values["v57_dataset_manifest"] = dataset_manifest
    for name, value in values.items():
        _json(root / input_paths[name], value)

    artifact_manifest = _self(
        {
            "version": "v57",
            "files": {},
            "data_files": {
                input_paths["labels"]: file_sha256(root / input_paths["labels"]),
                input_paths["sequence_roles"]: file_sha256(root / input_paths["sequence_roles"]),
            },
        },
        "manifest_sha256",
    )
    _json(root / input_paths["v57_artifact_manifest"], artifact_manifest)
    completion = {
        "decision": V58_ACTION,
        "result_file_sha256": file_sha256(root / input_paths["v57_result"]),
        "audit_file_sha256": file_sha256(root / input_paths["v57_audit"]),
        "dataset_spec_sha256": dataset_spec["dataset_spec_sha256"],
        "artifact_manifest_file_sha256": file_sha256(root / input_paths["v57_artifact_manifest"]),
    }
    _json(root / input_paths["v57_completion_receipt"], completion)

    experiment = {
        "schema_version": 1,
        "experiment_id": "v057_non_target_multi_horizon_dataset",
        "family_id": FAMILY,
        "status": "passed",
        "authorized_next_action": V58_ACTION,
    }
    experiment_path = root / "research/experiments/v057.yaml"
    _yaml(experiment_path, experiment)
    hashes = {name: file_sha256(root / path) for name, path in input_paths.items()}
    contract = {
        "schema_version": 1,
        "phase": "v58",
        "family_id": FAMILY,
        "authorized_next_action": V58_ACTION,
        "pass_action": V59_ACTION,
        "failure_action": "keep_v59_and_later_unauthorized",
        "parent_experiment": {"path": "research/experiments/v057.yaml", "file_sha256": file_sha256(experiment_path)},
        "preceding_phase_contract": {"path": "research/phase_contracts/v057.yaml", "file_sha256": preceding_hash},
        "authorization_receipt": {
            "path": input_paths["v57_result"],
            "file_sha256": hashes["v57_result"],
            "registered_result_sha256": v57_result["result_sha256"],
        },
        "v55_blueprint": {
            "path": input_paths["v55_blueprint"],
            "file_sha256": hashes["v55_blueprint"],
            "canonical_sha256": blueprint["blueprint_sha256"],
        },
        "access_contract": {
            "output_dir": "artifacts/v58",
            "allowed_inputs": list(input_paths.values()),
        },
        "input_contract": {"expected_sha256": hashes},
        "grid_contract": {
            "origins": ["origin_2024", "origin_2025"],
            "geometries": ["expanding", "rolling"],
            "folds": [1, 2, 3],
            "seeds": [42, 7, 123],
            "expected_jobs": 36,
            "job_key_order": "origin_geometry_fold_seed",
        },
        "scaler_contract": {"count": 12},
    }
    phase_path = root / "research/phase_contracts/v058.yaml"
    _yaml(phase_path, contract)
    state = {
        "current_experiment": "research/experiments/v057.yaml",
        "phase_contract": {"path": "research/phase_contracts/v058.yaml", "file_sha256": file_sha256(phase_path)},
        "authorized_phase": "v58",
        "authorized_next_action": V58_ACTION,
        "active_family_id": FAMILY,
        "last_completed_phase": "v57_non_target_multi_horizon_dataset",
    }
    _yaml(root / "research/current.yaml", state)
    source_path = root / "src/training.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text("# frozen\n", encoding="utf-8")
    config = {
        "state_conditioned_multi_horizon_training": {
            "version": "v58",
            "project_root": ".",
            "research_state": "research/current.yaml",
            "experiment_contract": "research/experiments/v057.yaml",
            "phase_contract": "research/phase_contracts/v058.yaml",
            "inputs": input_paths,
            "source_receipt_files": ["configs/v58.yaml", "src/training.py"],
            "require_clean_git": True,
        },
        "output_dir": "artifacts/v58",
    }
    _yaml(root / "configs/v58.yaml", config)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "fixture")
    return root, config


def test_prepare_metadata_is_hash_bound_clean_and_parquet_opaque(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config = _fixture(tmp_path)
    monkeypatch.setattr(
        training_artifacts, "_verify_backup_only_revision", lambda *args: None
    )
    context = prepare_training_metadata(config, repo_root=root)
    assert len(context["input_hashes"]) == 19
    assert len(context["job_ids"]) == 36
    assert context["source_receipt"]["git_clean"] is True
    assert set(context["source_receipt"]) >= {"git_head", "files", "bundle_sha256"}
    assert context["training_spec"]["phase_contract"] == context["phase_contract"]
    assert context["training_spec"]["contract"] == context["contract"]
    assert context["training_spec"]["source_receipt_files"] == ["configs/v58.yaml", "src/training.py"]
    assert set(context["input_values"]).isdisjoint({"feature_panel", "labels", "sequence_roles"})

    (root / "data/labels.parquet").write_bytes(b"drift")
    _git(root, "add", "data/labels.parquet")
    _git(root, "commit", "-qm", "drift")
    with pytest.raises(TrainingArtifactError, match="input hash drift: labels"):
        prepare_training_metadata(config, repo_root=root)


def test_v58r1_is_proven_backup_only_against_immutable_v58_base() -> None:
    root = Path(__file__).resolve().parents[1]
    base = yaml.safe_load(
        (root / "research/phase_contracts/v058.yaml").read_text(encoding="utf-8")
    )
    _verify_backup_only_revision(root, base)
    contract = yaml.safe_load(
        (root / "research/phase_contracts/v058r1.yaml").read_text(encoding="utf-8")
    )
    _verify_backup_only_revision(root, contract)

    drifted = deepcopy(contract)
    drifted["optimizer_and_early_stopping_contract"]["maximum_epochs"] = 31
    with pytest.raises(TrainingArtifactError, match="outside.*backup-only"):
        _verify_backup_only_revision(root, drifted)

    packet_drift = deepcopy(contract)
    packet_drift["artifact_contract"]["packet_files"] = []
    with pytest.raises(TrainingArtifactError, match="outside.*backup-only"):
        _verify_backup_only_revision(root, packet_drift)

    policy_drift = deepcopy(contract)
    policy_drift["runtime_contract"]["backup_policy"]["waived_safeguards"] = [
        "external_input_copy"
    ]
    with pytest.raises(TrainingArtifactError, match="outside.*backup-only"):
        _verify_backup_only_revision(root, policy_drift)

    doctor_field_drift = deepcopy(contract)
    doctor_field_drift["operator_enforcement_contract"][
        "live_doctor_fields_must_match_packet"
    ] = []
    with pytest.raises(TrainingArtifactError, match="outside.*backup-only"):
        _verify_backup_only_revision(root, doctor_field_drift)

    base_binding_drift = deepcopy(contract)
    base_binding_drift["supersedes"]["file_sha256"] = "0" * 64
    with pytest.raises(TrainingArtifactError, match="base binding drift"):
        _verify_backup_only_revision(root, base_binding_drift)

    missing_revision = deepcopy(contract)
    missing_revision.pop("revision")
    missing_revision["optimizer_and_early_stopping_contract"]["maximum_epochs"] = 31
    with pytest.raises(TrainingArtifactError, match="unrevisioned V58"):
        _verify_backup_only_revision(root, missing_revision)


def test_safe_paths_and_dirty_git_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, config = _fixture(tmp_path)
    monkeypatch.setattr(
        training_artifacts, "_verify_backup_only_revision", lambda *args: None
    )
    with pytest.raises(TrainingArtifactError, match="escapes"):
        resolve_repo_path(root, "../outside")
    (root / "dirty.txt").write_text("dirty", encoding="utf-8")
    with pytest.raises(TrainingArtifactError, match="clean"):
        prepare_training_metadata(config, repo_root=root)


def test_canonical_manifests_and_exact_order(tmp_path: Path) -> None:
    root, config = _fixture(tmp_path)
    contract = yaml.safe_load((root / "research/phase_contracts/v058.yaml").read_text())
    jobs = exact_job_ids(contract)
    assert jobs[:4] == [
        "origin_2024|expanding|1|42",
        "origin_2024|expanding|1|7",
        "origin_2024|expanding|1|123",
        "origin_2024|expanding|2|42",
    ]
    checkpoints = build_checkpoint_manifest(contract, [{"job_id": job} for job in jobs])
    assert checkpoints["checkpoint_count"] == 36
    assert checkpoints["manifest_sha256"] == canonical_sha256({k: v for k, v in checkpoints.items() if k != "manifest_sha256"})
    grid = build_grid_manifest(contract, [{"job_id": jobs[0], "status": "completed"}])
    assert grid["completed_jobs"] == jobs[:1] and len(grid["pending_jobs"]) == 35
    histories = build_history_manifest(contract, [{"job_id": jobs[0], "history": [{"epoch": 1}]}])
    assert histories["jobs"][0]["history_sha256"] == canonical_sha256([{"epoch": 1}])
    scalers = [
        {"scaler_id": f"{origin}|{geometry}|{fold}", "scaler_sha256": hashlib.sha256(f"{origin}|{geometry}|{fold}".encode()).hexdigest()}
        for origin in contract["grid_contract"]["origins"]
        for geometry in contract["grid_contract"]["geometries"]
        for fold in contract["grid_contract"]["folds"]
    ]
    scaler_manifest = build_scaler_manifest(contract, scalers)
    assert scaler_manifest["scaler_count"] == 12
    assert scaler_manifest["scalers"][0]["scaler_sha256"] == scalers[0]["scaler_sha256"]
    access = build_data_access_manifest(
        {
            "development_evaluation_outcome_rows_read": 0,
            "target_assets_loaded": [],
            "forbidden_columns_loaded": [],
            "previous_checkpoints_loaded": [],
            "heldout_fold_symbols_loaded_by_job": {},
            "predictions_written": False,
            "policy_actions_emitted": False,
            "performance_metrics_computed": False,
            "pnl_computed": False,
            "hyperparameters_changed": False,
        }
    )
    assert access["data_access_sha256"] == canonical_sha256({k: v for k, v in access.items() if k != "data_access_sha256"})
    with pytest.raises(TrainingArtifactError, match="out of frozen order"):
        build_checkpoint_manifest(contract, [{"job_id": jobs[1]}, {"job_id": jobs[0]}])


def test_atomic_artifact_and_stable_replay_helpers(tmp_path: Path) -> None:
    write_json_atomic(tmp_path / "checkpoint_manifest.json", {"ok": True})
    write_yaml_atomic(tmp_path / "resolved_config.yaml", {"version": "v58"})
    write_report_atomic(tmp_path / "report.md", "invocation report\n")
    write_json_atomic(tmp_path / "result.json", {"invocation": 1})
    write_json_atomic(tmp_path / "replay.json", {"invocation": 1})
    manifest = build_artifact_manifest(tmp_path, ["checkpoint_manifest.json", "resolved_config.yaml"])
    assert list(manifest["files"]) == ["checkpoint_manifest.json", "resolved_config.yaml"]
    with pytest.raises(TrainingArtifactError, match="exclude itself"):
        build_artifact_manifest(tmp_path, ["artifact_manifest.json"])
    hashes = stable_replay_hashes(tmp_path)
    assert set(hashes) == {"checkpoint_manifest.json", "resolved_config.yaml"}
