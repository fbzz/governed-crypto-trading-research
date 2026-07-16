from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import tlm.state_conditioned_multi_horizon_training as training
from tlm.core.artifacts import canonical_sha256
from tlm.state_conditioned_multi_horizon_training_data import (
    MaterializedBatch,
    UniformDateTripletSampler,
)
from tlm.state_conditioned_multi_horizon_training_artifacts import (
    stable_replay_hashes,
)
from tlm.state_conditioned_multi_horizon_training_engine import (
    V58Batch,
    V58BatchStream,
)


def _job_ids() -> list[str]:
    return [
        f"{origin}|{geometry}|{fold}|{seed}"
        for origin in ("origin_2024", "origin_2025")
        for geometry in ("expanding", "rolling")
        for fold in (1, 2, 3)
        for seed in (42, 7, 123)
    ]


def _metadata(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    phase_hash = "a" * 64
    source_hash = "b" * 64
    input_hashes = {"input": "c" * 64}
    contract = {
        "phase": "v58",
        "family_id": "tlm_state_conditioned_multi_horizon_quantile_small_v1",
        "authorized_next_action": "authorize_v58_frozen_non_target_training_only",
        "pass_action": "authorize_v59_frozen_adaptive_development_evaluation_only",
        "grid_contract": {
            "origins": ["origin_2024", "origin_2025"],
            "geometries": ["expanding", "rolling"],
            "folds": [1, 2, 3],
            "seeds": [42, 7, 123],
            "expected_jobs": 36,
            "job_key_order": "origin_geometry_fold_seed",
        },
        "runtime_contract": {
            "process_lock": "data/checkpoints/.v58.lock",
            "external_backup_receipt_required": True,
            "external_backup_receipt": "research/backups/v058.yaml",
            "full_phase_order": [
                "doctor",
                "preflight",
                "smoke",
                "full",
                "verify",
                "replay",
            ],
        },
    }
    training_spec = {
        "version": "v58_training_spec_v1",
        "phase_contract": {
            "path": "research/phase_contracts/v058.yaml",
            "file_sha256": phase_hash,
        },
        "contract": contract,
        "source_receipt_files": ["src/tlm/example.py"],
        "input_hashes": input_hashes,
        "expected_job_ids": _job_ids(),
        "source_bundle_sha256": source_hash,
    }
    source_receipt = {
        "version": "v58_source_receipt_v1",
        "git_clean": True,
        "git_head": "d" * 40,
        "files": {"src/tlm/example.py": "e" * 64},
        "bundle_sha256": source_hash,
    }
    output = tmp_path / "artifacts/v58"
    metadata: dict[str, object] = {
        "root": tmp_path,
        "training": {"version": "v58"},
        "contract": contract,
        "phase_contract": training_spec["phase_contract"],
        "input_hashes": input_hashes,
        "source_receipt": source_receipt,
        "training_spec": training_spec,
        "job_ids": _job_ids(),
        "output_dir": output,
    }
    config: dict[str, object] = {
        "seed": 20260714,
        "state_conditioned_multi_horizon_training": {"version": "v58"},
        "output_dir": "artifacts/v58",
    }
    return metadata, config


def test_invalid_mode_fails_before_metadata_or_data_access(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown operation must not get far enough to inspect research inputs."""

    touched = False

    def forbidden_metadata(*args: object, **kwargs: object) -> object:
        nonlocal touched
        touched = True
        raise AssertionError("invalid mode reached metadata preparation")

    monkeypatch.setattr(
        training, "prepare_training_metadata", forbidden_metadata, raising=False
    )
    with pytest.raises(ValueError, match="mode"):
        training.run_state_conditioned_multi_horizon_training({}, mode="train")
    assert touched is False


def test_global_training_lock_is_live_exclusive_and_stale_file_safe(
    tmp_path: Path,
) -> None:
    """The contract names one flock, not a lock-file existence sentinel."""

    lock_path = tmp_path / "data/checkpoints/.v58.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text("stale metadata\n", encoding="utf-8")

    with training._process_lock(lock_path, "smoke"):
        with lock_path.open("a+", encoding="utf-8") as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(
                    contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB
                )

    # A stale path is reusable once no process owns the advisory lock.
    with training._process_lock(lock_path, "verify"):
        assert lock_path.exists()


def test_public_runner_dispatches_only_the_registered_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = {"phase": "v58"}
    calls: list[tuple[str, object, object]] = []

    monkeypatch.setattr(
        training,
        "prepare_training_metadata",
        lambda config: metadata,
        raising=False,
    )

    def handler(name: str):
        def run(observed_metadata: object, config: object) -> dict[str, str]:
            calls.append((name, observed_metadata, config))
            return {"mode": name}

        return run

    for name in ("preflight", "smoke", "full", "verify", "replay"):
        monkeypatch.setattr(training, f"_run_{name}", handler(name), raising=False)

    config = {"state_conditioned_multi_horizon_training": {"version": "v58"}}
    for name in ("preflight", "smoke", "full", "verify", "replay"):
        assert training.run_state_conditioned_multi_horizon_training(
            config, mode=name
        ) == {"mode": name}
        assert calls[-1] == (name, metadata, config)
    assert [name for name, _, _ in calls] == [
        "preflight",
        "smoke",
        "full",
        "verify",
        "replay",
    ]


def test_preflight_is_metadata_only_and_writes_hash_bound_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, config = _metadata(tmp_path)

    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("metadata-only preflight touched data, model, or optimizer")

    for name in (
        "read_job_training_data",
        "fit_job_train_only_scaler",
        "_model_factory",
        "StateConditionedMultiHorizonTransformer",
        "run_v58_training_job",
    ):
        monkeypatch.setattr(training, name, forbidden)
    doctor = {
        "passed": True,
        "full_training_ready": True,
        "mps_available": True,
        "mps_operational": True,
        "fallback_enabled": False,
    }
    packet_calls: list[str] = []
    monkeypatch.setattr(training, "_doctor_or_raise", lambda metadata: doctor)
    monkeypatch.setattr(
        training,
        "_build_operator_packet",
        lambda metadata, operation, **kwargs: packet_calls.append(operation),
    )

    result = training._run_preflight(metadata, config)
    output = metadata["output_dir"]
    assert isinstance(output, Path)
    required = {
        "grid_manifest.json",
        "input_hash_receipt.json",
        "preflight.json",
        "resolved_config.yaml",
        "source_receipt.json",
        "training_spec.json",
    }
    assert required.issubset({path.name for path in output.iterdir()})
    assert json.loads((output / "training_spec.json").read_text(encoding="utf-8")) \
        == metadata["training_spec"]
    assert json.loads((output / "source_receipt.json").read_text(encoding="utf-8")) \
        == metadata["source_receipt"]

    input_receipt = json.loads(
        (output / "input_hash_receipt.json").read_text(encoding="utf-8")
    )
    assert input_receipt["version"] == "v58_input_hash_receipt_v1"
    assert input_receipt["inputs"] == {"input": "c" * 64}
    input_body = dict(input_receipt)
    registered_input_hash = input_body.pop("input_hash_receipt_sha256")
    assert registered_input_hash == canonical_sha256(input_body)
    grid = json.loads((output / "grid_manifest.json").read_text(encoding="utf-8"))
    assert grid["version"] == "v58_grid_manifest_v1"
    assert grid["expected_jobs"] == _job_ids()
    assert grid["counts"] == {
        "expected": 36,
        "completed": 0,
        "active": 0,
        "pending": 36,
    }
    assert grid["selected_jobs"] == []
    preflight = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
    serialized = json.dumps(preflight, sort_keys=True)
    assert preflight["version"] == "v58_preflight_v1"
    assert preflight["passed"] is True
    assert preflight["decision"] == "authorize_v58_one_job_mps_smoke_only"
    assert preflight["expected_jobs"] == _job_ids()
    assert preflight["parameter_count"] == 465_513
    assert preflight["parquet_deserializations"] == 0
    assert preflight["model_instantiations"] == 0
    assert preflight["optimizer_steps"] == 0
    assert preflight["target_asset_loads"] == 0
    assert preflight["doctor"] == doctor
    assert "a" * 64 in serialized
    assert "b" * 64 in serialized
    assert result["audit"] == {"passed": True}
    assert result["summary"] == {
        "expected_jobs": 36,
        "checkpoint_count": 0,
        "optimizer_steps": 0,
        "parquet_deserializations": 0,
    }
    assert result["invocation"] == {
        "mode": "preflight",
        "new_jobs": 0,
        "new_optimizer_steps": 0,
    }
    assert packet_calls == ["doctor"]


def test_owner_waiver_preflight_writes_exact_policy_receipt_and_tamper_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, config = _metadata(tmp_path)
    waiver_ref = dict(training.V58_OWNER_STORAGE_WAIVER)
    scopes = list(training.V58_WAIVED_STORAGE_SAFEGUARDS)
    metadata["contract"]["revision"] = "v058r1"
    runtime = metadata["contract"]["runtime_contract"]
    runtime["external_backup_receipt_required"] = False
    runtime.pop("external_backup_receipt")
    runtime["backup_policy"] = {
        "mode": "owner_waiver",
        "waiver": waiver_ref,
        "waived_safeguards": scopes,
    }
    doctor = {
        "passed": True,
        "full_training_ready": True,
        "backup": {
            "mode": "owner_waiver",
            "required": False,
            "passed": True,
            "waiver_verified": True,
            "waiver_path": waiver_ref["path"],
            "waiver_sha256": waiver_ref["file_sha256"],
            "objects_verified": 0,
            "code_backup_verified": False,
        },
    }
    monkeypatch.setattr(training, "_doctor_or_raise", lambda metadata: doctor)
    monkeypatch.setattr(training, "_build_operator_packet", lambda *args, **kwargs: {})

    training._run_preflight(metadata, config)
    output = metadata["output_dir"]
    assert isinstance(output, Path)
    receipt_path = output / "backup_policy_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    body = dict(receipt)
    registered = body.pop("policy_receipt_sha256")
    assert registered == canonical_sha256(body)
    assert receipt == training._require_owner_waiver_receipt(metadata, doctor)
    assert receipt["waived_safeguards"] == scopes
    assert receipt["external_checkpoint_backup_created"] is False
    monkeypatch.setattr(
        training,
        "_checkpoint_backup",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("owner waiver fell back to external checkpoint backup")
        ),
    )
    evidence_name, evidence_path, evidence = training._storage_evidence(
        metadata, doctor
    )
    assert evidence_name == "backup_policy"
    assert evidence_path == receipt_path
    assert evidence == receipt

    receipt["external_checkpoint_backup_created"] = True
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    with pytest.raises(training.V58TrainingError, match="receipt drift"):
        training._require_owner_waiver_receipt(metadata, doctor)


def test_completion_receipt_keeps_legacy_external_schema_and_versions_v58r1(
    tmp_path: Path,
) -> None:
    def write_completion_inputs(output: Path, protection_file: str) -> None:
        output.mkdir(parents=True, exist_ok=True)
        for name in (
            "result.json",
            "audit.json",
            "training_result.json",
            "verification.json",
            "replay.json",
            "checkpoint_manifest.json",
            protection_file,
        ):
            (output / name).write_text("{}\n", encoding="utf-8")

    external, _ = _metadata(tmp_path / "external")
    external_output = external["output_dir"]
    assert isinstance(external_output, Path)
    write_completion_inputs(external_output, "checkpoint_backup_receipt.json")
    legacy = training._build_completion_receipt(
        external, external_output, "authorize_v59"
    )
    assert legacy["version"] == "v58_completion_receipt_v1"
    assert "checkpoint_backup_receipt_file_sha256" in legacy
    assert "storage_protection_artifact" not in legacy

    revised, _ = _metadata(tmp_path / "revised")
    revised["contract"]["revision"] = "v058r1"
    runtime = revised["contract"]["runtime_contract"]
    runtime["external_backup_receipt_required"] = False
    runtime.pop("external_backup_receipt")
    runtime["backup_policy"] = {
        "mode": "owner_waiver",
        "waiver": dict(training.V58_OWNER_STORAGE_WAIVER),
        "waived_safeguards": list(training.V58_WAIVED_STORAGE_SAFEGUARDS),
    }
    revised_output = revised["output_dir"]
    assert isinstance(revised_output, Path)
    write_completion_inputs(revised_output, "backup_policy_receipt.json")
    v58r1 = training._build_completion_receipt(
        revised, revised_output, "authorize_v59"
    )
    assert v58r1["version"] == "v58r1_completion_receipt_v1"
    assert v58r1["storage_protection_artifact"] == "backup_policy_receipt.json"
    assert "checkpoint_backup_receipt_file_sha256" not in v58r1


def test_operator_packet_prerequisite_fails_closed_on_missing_or_wrong_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "artifacts/v58"
    output.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="operator_packet_smoke|operator packet"):
        training._require_operator_packet(
            output, "operator_packet_smoke.json", "smoke"
        )

    packet_path = output / "operator_packet_smoke.json"
    packet_path.write_text(
        json.dumps({"operation": "full"}) + "\n", encoding="utf-8"
    )
    reported_operation = "full"

    def validator(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {"valid": True, "operation": reported_operation}
            ),
            stderr="",
        )

    monkeypatch.setattr(training, "_script_command", validator)
    with pytest.raises(RuntimeError, match="operation|smoke"):
        training._require_operator_packet(
            output, "operator_packet_smoke.json", "smoke"
        )

    packet_path.write_text(
        json.dumps({"operation": "smoke"}) + "\n", encoding="utf-8"
    )
    reported_operation = "smoke"
    assert training._require_operator_packet(
        output, "operator_packet_smoke.json", "smoke"
    )["operation"] == reported_operation


def test_materialized_batch_provider_uses_exact_sampler_and_float32_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    symbols = ("AUSDT", "BUSDT", "CUSDT")
    train_date = np.datetime64("2022-12-23")
    validation_date = np.datetime64("2023-01-01")
    data = SimpleNamespace(
        cell=SimpleNamespace(
            origin="origin_2024",
            geometry="expanding",
            fold=1,
            train_triplets=(symbols,),
        ),
        train_availability={train_date: symbols},
        validation_availability={validation_date: symbols},
    )
    scaler = object()
    materializations: list[tuple[str, tuple[int, ...]]] = []

    def fake_materialize(
        observed_data: object,
        draws: object,
        observed_scaler: object,
        *,
        role: str,
    ) -> MaterializedBatch:
        assert observed_data is data
        assert observed_scaler is scaler
        draw_tuple = tuple(draws)
        materializations.append(
            (role, tuple(int(draw.pair_index) for draw in draw_tuple))
        )
        count = len(draw_tuple)
        features = np.zeros((count, 256, 3, 9), dtype=np.float32)
        targets = np.zeros((count, 3, 3), dtype=np.float32)
        return MaterializedBatch(features=features, targets=targets)

    monkeypatch.setattr(training, "materialize_triplet_batch", fake_materialize)
    provider = training.V58MaterializedBatchProvider(
        data,
        scaler,
        origin="origin_2024",
        geometry="expanding",
        fold=1,
        job_seed=42,
        train_samples=5,
        validation_samples=4,
        batch_size=2,
    )
    train_stream = provider("train", 3)
    assert isinstance(train_stream, V58BatchStream)
    train_batches = list(train_stream.batches)
    assert [batch.features.shape[0] for batch in train_batches] == [2, 2, 1]
    assert all(isinstance(batch, V58Batch) for batch in train_batches)
    assert all(batch.features.shape[1:] == (256, 3, 9) for batch in train_batches)
    assert all(batch.targets.shape[1:] == (3, 3) for batch in train_batches)
    assert all(batch.features.dtype == torch.float32 for batch in train_batches)
    assert all(batch.targets.dtype == torch.float32 for batch in train_batches)
    expected = UniformDateTripletSampler(
        data.train_availability, data.cell.train_triplets
    ).sample(
        5,
        origin="origin_2024",
        geometry="expanding",
        fold=1,
        job_seed=42,
        role="train",
        epoch=3,
    )
    assert train_stream.sampler_receipt == expected.ordered_draw_list_sha256
    assert materializations == [
        ("train", tuple(draw.pair_index for draw in expected.draws[:2])),
        ("train", tuple(draw.pair_index for draw in expected.draws[2:4])),
        ("train", tuple(draw.pair_index for draw in expected.draws[4:])),
    ]

    validation_42 = provider("validation", 0)
    provider_seed_7 = training.V58MaterializedBatchProvider(
        data,
        scaler,
        origin="origin_2024",
        geometry="expanding",
        fold=1,
        job_seed=7,
        train_samples=5,
        validation_samples=4,
        batch_size=2,
    )
    validation_7 = provider_seed_7("validation", 0)
    assert validation_42.sampler_receipt == validation_7.sampler_receipt
    assert len(list(validation_42.batches)) == len(list(validation_7.batches)) == 2
    with pytest.raises(ValueError, match="validation.*epoch|epoch zero"):
        provider("validation", 1)


def test_noop_replay_receipt_binds_stable_artifact_and_checkpoint_hashes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "artifacts/v58"
    output.mkdir(parents=True)
    (output / "training_result.json").write_text(
        '{"completed_jobs":36}\n', encoding="utf-8"
    )
    (output / "checkpoint_manifest.json").write_text(
        '{"checkpoint_count":36}\n', encoding="utf-8"
    )
    # Invocation-level files are deliberately mutable and excluded from the
    # frozen replay hash surface.
    (output / "result.json").write_text('{"invocation":1}\n', encoding="utf-8")
    before = stable_replay_hashes(output)
    (output / "result.json").write_text('{"invocation":2}\n', encoding="utf-8")
    after = stable_replay_hashes(output)
    checkpoints = {f"job-{index:02d}": f"{index:064x}" for index in range(36)}

    receipt = training._build_replay_receipt(
        before,
        after,
        checkpoint_hashes_before=checkpoints,
        checkpoint_hashes_after=dict(checkpoints),
    )
    assert receipt["version"] == "v58_replay_v1"
    replay = receipt["replay"]
    assert replay["passed"] is True
    assert replay["new_jobs"] == 0
    assert replay["new_optimizer_steps"] == 0
    assert replay["rewritten_checkpoints"] == 0
    assert replay["stable_artifact_hashes_before"] == before
    assert replay["stable_artifact_hashes_after"] == after
    assert replay["artifact_hashes_match"] is True
    assert replay["checkpoint_hashes_match"] is True
    assert replay["checkpoint_hashes_before"] == checkpoints
    assert replay["checkpoint_hashes_after"] == checkpoints


def test_full_orchestration_persists_exact_unselected_grid_and_access_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise all 36 orchestration jobs without data, MPS, or optimization."""

    metadata, config = _metadata(tmp_path)
    root = metadata["root"]
    output = metadata["output_dir"]
    contract = metadata["contract"]
    assert isinstance(root, Path)
    assert isinstance(output, Path)
    assert isinstance(contract, dict)
    output.mkdir(parents=True)
    contract.update(
        {
            "access_contract": {
                "checkpoint_dir": "data/checkpoints/v58-synthetic",
            },
            "optimizer_and_early_stopping_contract": {
                "maximum_epochs": 30,
                "early_stopping_patience": 5,
            },
            "sampling_contract": {
                "train_samples_per_epoch": 8192,
                "fixed_validation_samples": 2048,
                "batch_size": 128,
            },
            "scaler_contract": {"count": 12},
        }
    )
    metadata["input_values"] = {"v55_blueprint": {"architecture": {}}}

    events: list[str] = []
    lock_active = False

    @contextmanager
    def synthetic_lock(path: Path, operation: str):
        nonlocal lock_active
        assert operation == "full"
        assert path == root / "data/checkpoints/.v58.lock"
        assert lock_active is False
        lock_active = True
        events.append("lock_enter")
        try:
            yield
        finally:
            lock_active = False
            events.append("lock_exit")

    packet_evidence: dict[str, Path] = {}

    def build_packet(
        observed_metadata: object,
        operation: str,
        *,
        evidence: dict[str, Path],
    ) -> dict[str, object]:
        assert observed_metadata is metadata
        assert operation == "full"
        assert lock_active is False
        events.append("operator_full")
        packet_evidence.update(evidence)
        return {"operation": "full"}

    cell_calls: list[str] = []

    def bind_cell(
        observed_metadata: object,
        *,
        origin: str,
        geometry: str,
        fold: int,
        checkpoint_root: Path,
    ) -> tuple[SimpleNamespace, SimpleNamespace, dict[str, object], dict[str, object]]:
        assert observed_metadata is metadata
        assert checkpoint_root == root / "data/checkpoints/v58-synthetic"
        cell_id = f"{origin}|{geometry}|{fold}"
        cell_calls.append(cell_id)
        symbols = ("AUSDT", "BUSDT", "CUSDT")
        signal_date = np.datetime64("2022-12-23")
        data = SimpleNamespace(
            cell=SimpleNamespace(
                origin=origin,
                geometry=geometry,
                fold=fold,
                train_symbols=symbols,
                train_triplets=(symbols,),
            ),
            train_availability={signal_date: symbols},
            validation_availability={signal_date: symbols},
        )
        scaler_sha = hashlib.sha256(cell_id.encode("utf-8")).hexdigest()
        access_body = {
            "version": "v58_cell_data_access_v1",
            "cell_id": cell_id,
            "access_receipt": {
                "authorized_panel_rows": 768,
                "train_validation_signal_key_counts": {
                    "train": 3,
                    "validation": 3,
                },
                "forbidden_column_count_zero": 0,
                "job_relative_development_evaluation_value_count_zero": 0,
                "target_asset_load_count_zero": 0,
            },
        }
        access = {
            **access_body,
            "data_access_sha256": canonical_sha256(access_body),
        }
        scaler_payload = {
            "version": "v58_train_only_scaler_v1",
            "scaler_id": cell_id,
            "scaler": {
                "origin": origin,
                "geometry": geometry,
                "fold": fold,
                "fit_symbols": list(symbols),
                "fit_symbol_count": 3,
                "fit_unique_symbol_date_count": 300,
                "fit_min_date": "2021-03-01",
                "fit_max_date": "2022-12-23",
                "mean": [0.0] * 8,
                "standard_deviation": [1.0] * 8,
                "zero_scale_replacements": 0,
                "scaler_sha256": scaler_sha,
            },
        }
        scaler = SimpleNamespace(scaler_sha256=scaler_sha)
        return data, scaler, access, scaler_payload

    engine_calls: list[str] = []

    def synthetic_training_job(**kwargs: object) -> dict[str, object]:
        assert kwargs["device"] == "mps"
        assert kwargs["maximum_epochs"] == 30
        assert kwargs["patience"] == 5
        assert isinstance(kwargs["batch_provider"], training.V58MaterializedBatchProvider)
        context = kwargs["context"]
        final_path = kwargs["final_path"]
        assert isinstance(final_path, Path)
        job_id = context.job_metadata["job_id"]
        engine_calls.append(job_id)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_bytes(f"synthetic:{job_id}".encode("utf-8"))
        state_sha = hashlib.sha256(job_id.encode("utf-8")).hexdigest()
        return {
            "completed": True,
            "checkpoint_path": str(final_path),
            "current_model_state_sha256": state_sha,
            "best_model_state_sha256": state_sha,
            "optimizer_state_sha256": state_sha,
            "semantic_checkpoint_sha256": state_sha,
            "completed_epoch": 2,
            "optimizer_step_count": 4,
            "new_optimizer_steps": 4,
            "best_epoch": 1,
            "best_validation_total_loss": 0.5,
            "scaler_sha256": context.scaler_sha256,
            "data_access_sha256": context.data_access_sha256,
            "phase_contract_sha256": context.phase_contract_sha256,
            "source_bundle_sha256": context.source_bundle_sha256,
            "history": [
                {
                    "epoch": 1,
                    "train_optimizer_steps": 2,
                    "optimizer_step_count": 2,
                },
                {
                    "epoch": 2,
                    "train_optimizer_steps": 2,
                    "optimizer_step_count": 4,
                },
            ],
            "sampler_receipts": {
                "train": ["1" * 64, "2" * 64],
                "validation": ["3" * 64, "3" * 64],
            },
        }

    monkeypatch.setattr(training, "_preflight_receipt", lambda output: {"passed": True})
    monkeypatch.setattr(training, "_smoke_receipt", lambda output: {"passed": True})
    monkeypatch.setattr(
        training,
        "_require_operator_packet",
        lambda output, name, operation: {"operation": operation},
    )
    monkeypatch.setattr(
        training, "_doctor_or_raise", lambda metadata: {"passed": True}
    )
    monkeypatch.setattr(training, "_process_lock", synthetic_lock)
    monkeypatch.setattr(training, "_build_operator_packet", build_packet)
    monkeypatch.setattr(training, "_read_and_bind_cell", bind_cell)
    monkeypatch.setattr(training, "run_v58_training_job", synthetic_training_job)
    if hasattr(training.torch, "mps"):
        monkeypatch.setattr(training.torch.mps, "empty_cache", lambda: None)

    result = training._run_full(metadata, config)

    assert cell_calls == [
        f"{origin}|{geometry}|{fold}"
        for origin in ("origin_2024", "origin_2025")
        for geometry in ("expanding", "rolling")
        for fold in (1, 2, 3)
    ]
    assert engine_calls == _job_ids()
    assert events == ["lock_enter", "lock_exit", "operator_full"]
    assert packet_evidence == {
        "data_access": output / "data_access.json",
        "checkpoint_manifest": output / "checkpoint_manifest.json",
    }

    checkpoint = json.loads(
        (output / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    grid = json.loads((output / "grid_manifest.json").read_text(encoding="utf-8"))
    history = json.loads(
        (output / "history_manifest.json").read_text(encoding="utf-8")
    )
    scalers = json.loads(
        (output / "scaler_manifest.json").read_text(encoding="utf-8")
    )
    access = json.loads((output / "data_access.json").read_text(encoding="utf-8"))
    training_result = json.loads(
        (output / "training_result.json").read_text(encoding="utf-8")
    )

    assert checkpoint["expected_jobs"] == _job_ids()
    assert checkpoint["checkpoint_count"] == 36
    assert [row["job_id"] for row in checkpoint["jobs"]] == _job_ids()
    assert checkpoint["selected_jobs"] == []
    assert checkpoint["active_jobs"] == []
    assert grid["counts"] == {
        "expected": 36,
        "completed": 36,
        "active": 0,
        "pending": 0,
    }
    assert grid["selected_jobs"] == []
    assert history["history_count"] == 36
    assert history["selected_jobs"] == []
    assert scalers["scaler_count"] == 12
    assert len(scalers["scalers"]) == 12

    ledger = access["data_access"]
    assert ledger["outcome_rows_read"] == 0
    assert ledger["target_assets_loaded"] == []
    assert ledger["forbidden_columns_loaded"] == []
    assert ledger["previous_checkpoints_loaded"] == []
    assert ledger["predictions_written"] is False
    assert ledger["policy_actions_emitted"] is False
    assert ledger["performance_metrics_computed"] is False
    assert ledger["pnl_computed"] is False
    assert ledger["hyperparameters_changed"] is False
    assert len(ledger["optimizer_steps_by_job"]) == 36
    assert len(ledger["scaler_fit_rows_by_origin_geometry_fold"]) == 12
    assert all(
        value == []
        for value in ledger["heldout_fold_symbols_loaded_by_job"].values()
    )

    assert training_result["checkpoint_count"] == 36
    assert training_result["scaler_count"] == 12
    assert training_result["history_count"] == 36
    assert training_result["total_optimizer_steps"] == 144
    assert training_result["new_jobs"] == 36
    assert training_result["new_optimizer_steps"] == 144
    assert training_result["predictions_written"] is False
    assert training_result["performance_metrics_computed"] is False
    assert training_result["pnl_computed"] is False
    assert result["invocation"] == {
        "mode": "full",
        "new_jobs": 36,
        "new_optimizer_steps": 144,
    }
