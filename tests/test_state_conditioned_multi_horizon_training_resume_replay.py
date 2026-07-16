from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import tlm.state_conditioned_multi_horizon_training as training


def _job_ids() -> list[str]:
    return [
        f"{origin}|{geometry}|{fold}|{seed}"
        for origin in ("origin_2024", "origin_2025")
        for geometry in ("expanding", "rolling")
        for fold in (1, 2, 3)
        for seed in (42, 7, 123)
    ]


def _job_dir(checkpoint_root: Path, job_id: str) -> Path:
    origin, geometry, fold, seed = training._split_job_id(job_id)
    return training._job_directory(
        checkpoint_root, origin, geometry, fold, seed
    )


def _write_checkpoint(
    checkpoint_root: Path, job_id: str, name: str, payload: bytes
) -> Path:
    path = _job_dir(checkpoint_root, job_id) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _tree_bytes(root: Path) -> dict[str, bytes]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _checkpoint_manifest(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_root = tmp_path / "data/checkpoints/v58"
    phase_sha = "a" * 64
    source_sha = "b" * 64
    rows: list[dict[str, Any]] = []
    cell_bindings: dict[tuple[str, str, int], tuple[str, str]] = {}
    for index, job_id in enumerate(_job_ids()):
        origin, geometry, fold, seed = training._split_job_id(job_id)
        cell_key = (origin, geometry, fold)
        if cell_key not in cell_bindings:
            cell_id = f"{origin}|{geometry}|{fold}"
            scaler_sha = _sha(f"scaler:{cell_id}".encode("utf-8"))
            access_body = {
                "version": "v58_cell_data_access_v1",
                "cell_id": cell_id,
                "access_receipt": {"synthetic": True},
            }
            access_sha = training.canonical_sha256(access_body)
            access_payload = {
                **access_body,
                "data_access_sha256": access_sha,
            }
            scaler_payload = {
                "version": "v58_train_only_scaler_v1",
                "scaler_id": cell_id,
                "scaler": {
                    "fit_symbols": ["AUSDT", "BUSDT", "CUSDT"],
                    "scaler_sha256": scaler_sha,
                },
            }
            cell_dir = training._cell_directory(
                checkpoint_root, origin, geometry, fold
            )
            cell_dir.mkdir(parents=True, exist_ok=True)
            (cell_dir / "data_access.json").write_text(
                json.dumps(access_payload, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (cell_dir / "scaler.json").write_text(
                json.dumps(scaler_payload, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            cell_bindings[cell_key] = (scaler_sha, access_sha)
        scaler_sha, access_sha = cell_bindings[cell_key]
        payload = f"frozen-final:{job_id}".encode("utf-8")
        final_path = _write_checkpoint(
            checkpoint_root, job_id, "final.pt", payload
        )
        semantic_sha = _sha(f"semantic:{job_id}".encode("utf-8"))
        rows.append(
            {
                "job_id": job_id,
                "origin": origin,
                "geometry": geometry,
                "fold": fold,
                "seed": seed,
                "status": "completed",
                "checkpoint_path": final_path.relative_to(tmp_path).as_posix(),
                "checkpoint_sha256": _sha(payload),
                "checkpoint_size_bytes": len(payload),
                "current_model_state_sha256": semantic_sha,
                "best_model_state_sha256": semantic_sha,
                "optimizer_state_sha256": semantic_sha,
                "semantic_checkpoint_sha256": semantic_sha,
                "completed_epoch": 2,
                "optimizer_step_count": 100 + index,
                "best_epoch": 1,
                "best_validation_total_loss": 0.5,
                "scaler_sha256": scaler_sha,
                "data_access_sha256": access_sha,
                "phase_contract_sha256": phase_sha,
                "source_bundle_sha256": source_sha,
                "job_metadata": {
                    "version": "v58",
                    "run_kind": "full",
                    "job_id": job_id,
                    "origin": origin,
                    "geometry": geometry,
                    "fold": fold,
                    "seed": seed,
                    "train_symbols": ["AUSDT", "BUSDT", "CUSDT"],
                    "prior_checkpoint_or_representation_reuse": False,
                    "selected": False,
                },
            }
        )
    manifest = {
        "version": "v58_checkpoint_manifest_v1",
        "expected_jobs": _job_ids(),
        "jobs": rows,
        "checkpoint_count": 36,
        "selected_jobs": [],
        "active_jobs": [],
    }
    manifest["manifest_sha256"] = training.canonical_sha256(manifest)
    metadata: dict[str, Any] = {
        "root": tmp_path,
        "output_dir": tmp_path / "artifacts/v58",
        "job_ids": _job_ids(),
        "phase_contract": {"file_sha256": phase_sha},
        "source_receipt": {"bundle_sha256": source_sha},
        "input_values": {"v55_blueprint": {"architecture": {}}},
        "contract": {
            "access_contract": {"checkpoint_dir": "data/checkpoints/v58"},
            "runtime_contract": {
                "process_lock": "data/checkpoints/.v58.lock",
                "external_backup_receipt_required": True,
                "external_backup_receipt": "research/backups/v058.yaml",
            },
            "optimizer_and_early_stopping_contract": {
                "maximum_epochs": 30,
                "early_stopping_patience": 5,
            },
            "artifact_contract": {"packet_files": ["artifact_manifest.json"]},
            "family_id": "tlm_state_conditioned_multi_horizon_quantile_small_v1",
            "pass_action": "authorize_v59_frozen_adaptive_development_evaluation_only",
        },
    }
    return metadata, manifest


def test_checkpoint_tree_accepts_completed_prefix_and_only_same_next_resume(
    tmp_path: Path,
) -> None:
    checkpoint_root = tmp_path / "data/checkpoints/v58"
    jobs = _job_ids()
    for job_id in jobs[:2]:
        _write_checkpoint(checkpoint_root, job_id, "final.pt", b"complete")
    resume = _write_checkpoint(
        checkpoint_root, jobs[2], "resume.pt", b"epoch-boundary"
    )

    scan = training._scan_checkpoint_tree(
        checkpoint_root, jobs, repo_root=tmp_path
    )

    assert scan["completed_jobs"] == jobs[:2]
    assert scan["pending_resume_job"] == jobs[2]
    assert scan["pending_resume_artifacts"] == [
        resume.relative_to(tmp_path).as_posix()
    ]
    assert scan["orphan_resume_artifacts"] == []
    assert scan["active_resume_artifacts"] == []


@pytest.mark.parametrize(
    "layout,error_pattern",
    [
        ("multiple_resume", "resume|multiple"),
        ("future_orphan_resume", "resume|orphan|next"),
        ("completed_job_resume", "resume|completed"),
        ("orphan_final", "final|orphan|grid"),
        ("out_of_order_final", "final|prefix|order"),
        ("temporary_checkpoint", "temporary|temp|checkpoint"),
    ],
)
def test_checkpoint_tree_rejects_ambiguous_or_out_of_order_state(
    tmp_path: Path, layout: str, error_pattern: str
) -> None:
    checkpoint_root = tmp_path / "data/checkpoints/v58"
    jobs = _job_ids()
    _write_checkpoint(checkpoint_root, jobs[0], "final.pt", b"complete")
    if layout == "multiple_resume":
        _write_checkpoint(checkpoint_root, jobs[1], "resume.pt", b"next")
        _write_checkpoint(checkpoint_root, jobs[2], "resume.pt", b"also-next")
    elif layout == "future_orphan_resume":
        _write_checkpoint(checkpoint_root, jobs[2], "resume.pt", b"future")
    elif layout == "completed_job_resume":
        _write_checkpoint(checkpoint_root, jobs[0], "resume.pt", b"orphan")
    elif layout == "orphan_final":
        orphan = checkpoint_root / "unregistered" / "final.pt"
        orphan.parent.mkdir(parents=True, exist_ok=True)
        orphan.write_bytes(b"orphan-final")
    elif layout == "out_of_order_final":
        _write_checkpoint(checkpoint_root, jobs[2], "final.pt", b"future")
    elif layout == "temporary_checkpoint":
        _write_checkpoint(
            checkpoint_root, jobs[1], ".resume.pt.12345.tmp", b"partial"
        )
    else:  # pragma: no cover - the parametrization is exhaustive.
        raise AssertionError(layout)

    with pytest.raises(training.V58TrainingError, match=error_pattern):
        training._scan_checkpoint_tree(
            checkpoint_root, jobs, repo_root=tmp_path
        )


def test_completed_grid_noop_observes_every_checkpoint_without_writes_or_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, manifest = _checkpoint_manifest(tmp_path)
    output = metadata["output_dir"]
    output.mkdir(parents=True)
    (output / "training_result.json").write_text(
        '{"checkpoint_count":36,"passed":true}\n', encoding="utf-8"
    )
    checkpoint_root = tmp_path / "data/checkpoints/v58"
    artifact_bytes_before = _tree_bytes(output)
    checkpoint_bytes_before = _tree_bytes(checkpoint_root)
    calls: list[str] = []

    monkeypatch.setattr(training, "_model_factory", lambda architecture: object())

    def observe_existing_checkpoint(**kwargs: Any) -> dict[str, Any]:
        context = kwargs["context"]
        job_id = context.job_metadata["job_id"]
        row = manifest["jobs"][len(calls)]
        assert job_id == row["job_id"]
        assert kwargs["device"] == "cpu"
        assert kwargs["final_path"] == tmp_path / row["checkpoint_path"]
        assert kwargs["resume_path"] == kwargs["final_path"].with_name(
            "resume.pt"
        )
        calls.append(job_id)
        return {
            "status": "already_complete",
            "completed": True,
            "checkpoint_path": str(kwargs["final_path"]),
            "new_optimizer_steps": 0,
            "optimizer_step_count": row["optimizer_step_count"],
            "current_model_state_sha256": row["current_model_state_sha256"],
            "best_model_state_sha256": row["best_model_state_sha256"],
            "optimizer_state_sha256": row["optimizer_state_sha256"],
            "semantic_checkpoint_sha256": row["semantic_checkpoint_sha256"],
            "completed_epoch": row["completed_epoch"],
            "best_epoch": row["best_epoch"],
            "best_validation_total_loss": row[
                "best_validation_total_loss"
            ],
            "scaler_sha256": row["scaler_sha256"],
            "data_access_sha256": row["data_access_sha256"],
            "phase_contract_sha256": row["phase_contract_sha256"],
            "source_bundle_sha256": row["source_bundle_sha256"],
            "history": [],
            "sampler_receipts": {"train": [], "validation": []},
        }

    monkeypatch.setattr(
        training, "run_v58_training_job", observe_existing_checkpoint
    )

    receipt = training._run_completed_grid_noop(
        metadata, manifest, device="cpu"
    )

    assert calls == _job_ids()
    assert receipt["new_jobs"] == 0
    assert receipt["new_optimizer_steps"] == 0
    assert receipt["rewritten_checkpoints"] == 0
    assert [row["job_id"] for row in receipt["jobs"]] == _job_ids()
    assert all(
        row["checkpoint_sha256_before"]
        == row["checkpoint_sha256_after"]
        for row in receipt["jobs"]
    )
    assert _tree_bytes(output) == artifact_bytes_before
    assert _tree_bytes(checkpoint_root) == checkpoint_bytes_before


def _replay_prerequisites(
    tmp_path: Path, metadata: dict[str, Any], manifest: dict[str, Any]
) -> None:
    output = metadata["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoint_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )


@pytest.mark.parametrize(
    "mutation",
    ["new_job", "new_step", "rewritten_checkpoint", "missing_job", "hash_drift"],
)
def test_replay_invokes_observed_noop_and_rejects_any_job_step_or_hash_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    metadata, manifest = _checkpoint_manifest(tmp_path)
    _replay_prerequisites(tmp_path, metadata, manifest)
    calls: list[str] = []
    jobs = [
        {
            "job_id": row["job_id"],
            "checkpoint_sha256_before": row["checkpoint_sha256"],
            "checkpoint_sha256_after": row["checkpoint_sha256"],
            "new_optimizer_steps": 0,
        }
        for row in manifest["jobs"]
    ]
    noop: dict[str, Any] = {
        "new_jobs": 0,
        "new_optimizer_steps": 0,
        "rewritten_checkpoints": 0,
        "jobs": jobs,
    }
    if mutation == "new_job":
        noop["new_jobs"] = 1
    elif mutation == "new_step":
        noop["new_optimizer_steps"] = 1
        jobs[0]["new_optimizer_steps"] = 1
    elif mutation == "rewritten_checkpoint":
        noop["rewritten_checkpoints"] = 1
    elif mutation == "missing_job":
        jobs.pop()
    elif mutation == "hash_drift":
        jobs[0]["checkpoint_sha256_after"] = "f" * 64

    def observed_noop(
        observed_metadata: Any,
        observed_manifest: Any,
        *,
        device: str = "mps",
    ) -> dict[str, Any]:
        assert observed_metadata is metadata
        assert observed_manifest == manifest
        assert device == "mps"
        calls.append("noop")
        return noop

    @contextmanager
    def no_lock(path: Path, operation: str):
        assert operation == "replay"
        yield

    monkeypatch.setattr(training, "_run_completed_grid_noop", observed_noop)
    monkeypatch.setattr(training, "_process_lock", no_lock)
    monkeypatch.setattr(
        training, "_verification_receipt", lambda output: {"passed": True}
    )
    monkeypatch.setattr(
        training,
        "_require_operator_packet",
        lambda output, name, operation: {"operation": operation},
    )
    monkeypatch.setattr(
        training, "_checkpoint_backup", lambda output: {"verified": True}
    )
    monkeypatch.setattr(
        training, "_doctor_or_raise", lambda observed_metadata: {"passed": True}
    )

    with pytest.raises(training.V58TrainingError, match="replay|no-op|job|step|hash|checkpoint"):
        training._run_replay(metadata, {})
    assert calls == ["noop"]


def test_replay_rejects_stable_artifact_hash_drift_after_observed_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata, manifest = _checkpoint_manifest(tmp_path)
    _replay_prerequisites(tmp_path, metadata, manifest)
    calls: list[str] = []
    noop = {
        "new_jobs": 0,
        "new_optimizer_steps": 0,
        "rewritten_checkpoints": 0,
        "jobs": [
            {
                "job_id": row["job_id"],
                "checkpoint_sha256_before": row["checkpoint_sha256"],
                "checkpoint_sha256_after": row["checkpoint_sha256"],
                "new_optimizer_steps": 0,
            }
            for row in manifest["jobs"]
        ],
    }

    def observed_noop(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append("noop")
        return noop

    stable_hashes = iter(
        [
            {"training_result.json": "1" * 64},
            {"training_result.json": "2" * 64},
        ]
    )

    @contextmanager
    def no_lock(path: Path, operation: str):
        assert operation == "replay"
        yield

    monkeypatch.setattr(training, "_run_completed_grid_noop", observed_noop)
    monkeypatch.setattr(training, "_process_lock", no_lock)
    monkeypatch.setattr(
        training, "stable_replay_hashes", lambda output: next(stable_hashes)
    )
    monkeypatch.setattr(
        training, "_verification_receipt", lambda output: {"passed": True}
    )
    monkeypatch.setattr(
        training,
        "_require_operator_packet",
        lambda output, name, operation: {"operation": operation},
    )
    monkeypatch.setattr(
        training, "_checkpoint_backup", lambda output: {"verified": True}
    )
    monkeypatch.setattr(
        training, "_doctor_or_raise", lambda observed_metadata: {"passed": True}
    )

    with pytest.raises(training.V58TrainingError, match="stable|hash|replay"):
        training._run_replay(metadata, {})
    assert calls == ["noop"]
