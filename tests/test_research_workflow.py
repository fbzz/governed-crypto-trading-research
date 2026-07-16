from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
import fcntl
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tlm.research_workflow import (
    ResearchStateError,
    _inspect_training_lock,
    _validate_backup_receipt,
    research_doctor,
    validate_research_state,
)


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, dict]:
    artifacts = tmp_path / "artifacts"
    experiments = tmp_path / "research" / "experiments"
    artifacts.mkdir(parents=True)
    experiments.mkdir(parents=True)
    blueprint = {"candidate_family_id": "family-v1", "parameter_count": 10}
    blueprint["blueprint_sha256"] = _canonical(blueprint)
    paths = {
        "blueprint": artifacts / "blueprint.json",
        "result": artifacts / "result.json",
        "audit": artifacts / "audit.json",
    }
    paths["blueprint"].write_text(json.dumps(blueprint), encoding="utf-8")
    result = {"decision": "next"}
    result["result_sha256"] = _canonical(result)
    paths["result"].write_text(json.dumps(result), encoding="utf-8")
    paths["audit"].write_text(json.dumps({"passed": True}), encoding="utf-8")
    experiment = {
        "family_id": "family-v1",
        "blueprint": {
            "path": "artifacts/blueprint.json",
            "file_sha256": _file_sha(paths["blueprint"]),
            "canonical_sha256": blueprint["blueprint_sha256"],
        },
        "result": {
            "path": "artifacts/result.json",
            "file_sha256": _file_sha(paths["result"]),
            "registered_result_sha256": result["result_sha256"],
        },
        "audit": {
            "path": "artifacts/audit.json",
            "file_sha256": _file_sha(paths["audit"]),
        },
        "authorized_next_action": "next",
    }
    experiment_path = experiments / "v001.yaml"
    experiment_path.write_text(yaml.safe_dump(experiment), encoding="utf-8")
    phase_contracts = tmp_path / "research" / "phase_contracts"
    phase_contracts.mkdir(parents=True)
    phase_contract = {
        "phase": "v2",
        "family_id": "family-v1",
        "authorized_next_action": "next",
        "authorized_command": "run-v2",
        "parent_experiment": {
            "path": "research/experiments/v001.yaml",
            "file_sha256": _file_sha(experiment_path),
        },
        "authorization_receipt": {
            "path": "artifacts/result.json",
            "file_sha256": _file_sha(paths["result"]),
            "registered_result_sha256": result["result_sha256"],
            "decision": "next",
        },
        "access_contract": {
            "output_dir": "artifacts/v2",
            "allowed_inputs": ["artifacts/result.json"],
            "allowed_capabilities": ["run_frozen_v2"],
            "required_checks": ["all_hashes_match"],
            "forbidden_capabilities": ["target_asset_access"],
        },
    }
    phase_contract_path = phase_contracts / "v002.yaml"
    phase_contract_path.write_text(yaml.safe_dump(phase_contract), encoding="utf-8")
    state = {
        "schema_version": 1,
        "current_experiment": "research/experiments/v001.yaml",
        "phase_contract": {
            "path": "research/phase_contracts/v002.yaml",
            "file_sha256": _file_sha(phase_contract_path),
        },
        "active_family_id": "family-v1",
        "active_family_status": "specification_frozen",
        "authorized_next_action": "next",
        "authorized_phase": "v2",
        "authorized_command": "run-v2",
        "target_assets": {"symbols": ["BTCUSDT"], "status": "sealed"},
        "families": [{"family_id": "family-v1", "trained": False}],
        "forbidden_capabilities": ["target_asset_access"],
    }
    current_path = tmp_path / "research" / "current.yaml"
    current_path.write_text(yaml.safe_dump(state), encoding="utf-8")
    return current_path, state


def _rewrite_phase(tmp_path: Path, mutate: Callable[[dict], None]) -> None:
    phase_path = tmp_path / "research" / "phase_contracts" / "v002.yaml"
    phase = yaml.safe_load(phase_path.read_text(encoding="utf-8"))
    mutate(phase)
    phase_path.write_text(yaml.safe_dump(phase), encoding="utf-8")
    current_path = tmp_path / "research" / "current.yaml"
    state = yaml.safe_load(current_path.read_text(encoding="utf-8"))
    state["phase_contract"]["file_sha256"] = _file_sha(phase_path)
    current_path.write_text(yaml.safe_dump(state), encoding="utf-8")


def _rewrite_result_and_rebind(tmp_path: Path, result: dict) -> None:
    result_path = tmp_path / "artifacts" / "result.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    result_file_hash = _file_sha(result_path)

    experiment_path = tmp_path / "research" / "experiments" / "v001.yaml"
    experiment = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    experiment["result"]["file_sha256"] = result_file_hash
    experiment["result"]["registered_result_sha256"] = result["result_sha256"]
    experiment_path.write_text(yaml.safe_dump(experiment), encoding="utf-8")

    phase_path = tmp_path / "research" / "phase_contracts" / "v002.yaml"
    phase = yaml.safe_load(phase_path.read_text(encoding="utf-8"))
    phase["parent_experiment"]["file_sha256"] = _file_sha(experiment_path)
    phase["authorization_receipt"].update(
        {
            "file_sha256": result_file_hash,
            "registered_result_sha256": result["result_sha256"],
            "decision": result["decision"],
        }
    )
    phase_path.write_text(yaml.safe_dump(phase), encoding="utf-8")

    current_path = tmp_path / "research" / "current.yaml"
    state = yaml.safe_load(current_path.read_text(encoding="utf-8"))
    state["phase_contract"]["file_sha256"] = _file_sha(phase_path)
    current_path.write_text(yaml.safe_dump(state), encoding="utf-8")


def test_current_state_validates_lineage_authorization_and_hashes(tmp_path: Path) -> None:
    _fixture(tmp_path)
    result = validate_research_state(tmp_path)
    assert result["passed"]
    assert result["active_family_id"] == "family-v1"
    assert result["authorized_next_action"] == "next"
    assert result["phase_contract_path"] == "research/phase_contracts/v002.yaml"
    assert result["target_asset_status"] == "sealed"


def test_current_state_rejects_authorization_drift(tmp_path: Path) -> None:
    current_path, state = _fixture(tmp_path)
    changed = deepcopy(state)
    changed["authorized_next_action"] = "unauthorized"
    current_path.write_text(yaml.safe_dump(changed), encoding="utf-8")
    with pytest.raises(ResearchStateError, match="authorization differ"):
        validate_research_state(tmp_path)


def test_current_state_rejects_artifact_drift(tmp_path: Path) -> None:
    _fixture(tmp_path)
    (tmp_path / "artifacts" / "result.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ResearchStateError, match="authorization receipt file/hash drift"):
        validate_research_state(tmp_path)


def test_current_state_rejects_phase_contract_drift(tmp_path: Path) -> None:
    _fixture(tmp_path)
    (tmp_path / "research" / "phase_contracts" / "v002.yaml").write_text(
        "phase: v2\n", encoding="utf-8"
    )
    with pytest.raises(ResearchStateError, match="artifact checks failed"):
        validate_research_state(tmp_path)


def test_current_state_requires_phase_contract_pointer(tmp_path: Path) -> None:
    current_path, state = _fixture(tmp_path)
    state.pop("phase_contract")
    current_path.write_text(yaml.safe_dump(state), encoding="utf-8")
    with pytest.raises(ResearchStateError, match="missing keys:.*phase_contract"):
        validate_research_state(tmp_path)


def test_current_state_rejects_phase_contract_parent_hash_drift(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)
    experiment_path = tmp_path / "research" / "experiments" / "v001.yaml"
    experiment = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    experiment["unbound_change"] = True
    experiment_path.write_text(yaml.safe_dump(experiment), encoding="utf-8")
    with pytest.raises(ResearchStateError, match="artifact checks failed"):
        validate_research_state(tmp_path)


def test_current_state_rejects_unsealed_target_assets(tmp_path: Path) -> None:
    current_path, state = _fixture(tmp_path)
    state["target_assets"]["status"] = "open"
    current_path.write_text(yaml.safe_dump(state), encoding="utf-8")
    with pytest.raises(ResearchStateError, match="must remain sealed"):
        validate_research_state(tmp_path)


@pytest.mark.parametrize(
    "key",
    [
        "allowed_inputs",
        "allowed_capabilities",
        "required_checks",
        "forbidden_capabilities",
    ],
)
def test_current_state_rejects_empty_access_contract_lists(
    tmp_path: Path, key: str
) -> None:
    _fixture(tmp_path)
    _rewrite_phase(
        tmp_path,
        lambda phase: phase["access_contract"].update({key: []}),
    )
    with pytest.raises(ResearchStateError, match=rf"access_contract\.{key}.*non-empty"):
        validate_research_state(tmp_path)


def test_current_state_rejects_missing_or_duplicate_access_contract_entries(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)
    _rewrite_phase(
        tmp_path,
        lambda phase: phase["access_contract"].pop("required_checks"),
    )
    with pytest.raises(
        ResearchStateError, match=r"access_contract\.required_checks.*non-empty"
    ):
        validate_research_state(tmp_path)

    duplicate_root = tmp_path / "duplicate"
    _fixture(duplicate_root)
    _rewrite_phase(
        duplicate_root,
        lambda phase: phase["access_contract"].update(
            {"allowed_capabilities": ["run_frozen_v2", "run_frozen_v2"]}
        ),
    )
    with pytest.raises(
        ResearchStateError, match=r"access_contract\.allowed_capabilities.*unique"
    ):
        validate_research_state(duplicate_root)


def test_current_state_rejects_invalid_output_or_capability_overlap(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)
    _rewrite_phase(
        tmp_path,
        lambda phase: phase["access_contract"].update({"output_dir": " "}),
    )
    with pytest.raises(ResearchStateError, match="output_dir.*artifacts subdirectory"):
        validate_research_state(tmp_path)

    dot_root = tmp_path / "dot"
    _fixture(dot_root)
    _rewrite_phase(
        dot_root,
        lambda phase: phase["access_contract"].update({"output_dir": "."}),
    )
    with pytest.raises(ResearchStateError, match="output_dir.*artifacts subdirectory"):
        validate_research_state(dot_root)

    overlap_root = tmp_path / "overlap"
    _fixture(overlap_root)
    _rewrite_phase(
        overlap_root,
        lambda phase: phase["access_contract"].update(
            {"forbidden_capabilities": ["run_frozen_v2"]}
        ),
    )
    with pytest.raises(ResearchStateError, match="allows and forbids"):
        validate_research_state(overlap_root)


def test_current_state_rejects_authorization_receipt_reference_drift(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)
    _rewrite_phase(
        tmp_path,
        lambda phase: phase["authorization_receipt"].update(
            {"registered_result_sha256": "0" * 64}
        ),
    )
    with pytest.raises(ResearchStateError, match="differs from the experiment result"):
        validate_research_state(tmp_path)


def test_current_state_normalizes_malformed_authorization_receipt_types(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)
    _rewrite_phase(
        tmp_path,
        lambda phase: phase["authorization_receipt"].update({"path": 1}),
    )
    with pytest.raises(ResearchStateError, match="path or hashes are invalid"):
        validate_research_state(tmp_path)


def test_current_state_rejects_authorization_receipt_canonical_hash_drift(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)
    _rewrite_result_and_rebind(
        tmp_path,
        {"decision": "next", "result_sha256": "0" * 64},
    )
    with pytest.raises(ResearchStateError, match="canonical result hash drift"):
        validate_research_state(tmp_path)


def test_current_state_rejects_authorization_receipt_decision_drift(
    tmp_path: Path,
) -> None:
    _fixture(tmp_path)
    result = {"decision": "different"}
    result["result_sha256"] = _canonical(result)
    _rewrite_result_and_rebind(tmp_path, result)
    with pytest.raises(ResearchStateError, match="decision differs"):
        validate_research_state(tmp_path)


def test_training_lock_uses_live_flock_not_file_existence(tmp_path: Path) -> None:
    lock = tmp_path / "training.lock"
    lock.write_text('{"job_id":"origin_2024|expanding|1|42"}', encoding="utf-8")
    assert _inspect_training_lock(lock)["active_job_count"] == 0
    with lock.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = _inspect_training_lock(lock)
        assert active["available"] is False
        assert active["active_job_count"] == 1
        assert active["metadata"]["job_id"] == "origin_2024|expanding|1|42"
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def test_backup_receipt_requires_cross_device_hash_verified_inputs_and_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "repo"
    backup_root = tmp_path / "external"
    source = root / "data" / "input.bin"
    copied = backup_root / "objects" / "input.bin"
    code = backup_root / "code" / "source.bundle"
    receipt_path = root / "research" / "backups" / "v058.yaml"
    source.parent.mkdir(parents=True)
    copied.parent.mkdir(parents=True)
    code.parent.mkdir(parents=True)
    receipt_path.parent.mkdir(parents=True)
    source.write_bytes(b"input")
    copied.write_bytes(b"input")
    code.write_bytes(b"bundle")
    source_hash = _file_sha(source)
    code_hash = _file_sha(code)
    receipt = {
        "schema_version": "tlm-external-backup-receipt/v1",
        "phase": "v58",
        "backup_root": str(backup_root),
        "source_device": 10,
        "backup_device": 20,
        "different_device": True,
        "verified": True,
        "objects": [
            {
                "source_path": "data/input.bin",
                "backup_path": "objects/input.bin",
                "sha256": source_hash,
                "size_bytes": 5,
            }
        ],
        "code_backup": {
            "kind": "git_bundle",
            "git_head": "a" * 40,
            "backup_path": "code/source.bundle",
            "sha256": code_hash,
            "size_bytes": 6,
        },
    }
    receipt_path.write_text(yaml.safe_dump(receipt), encoding="utf-8")
    original_stat = Path.stat

    def fake_stat(path: Path, *args: object, **kwargs: object):
        value = original_stat(path, *args, **kwargs)
        resolved = path.resolve()
        device = 10 if resolved == root.resolve() else 20 if resolved == backup_root.resolve() else value.st_dev
        values = list(value)
        values[2] = device
        return type(value)(values)

    monkeypatch.setattr(Path, "stat", fake_stat)
    contract = {
        "allowed_inputs": ["data/input.bin"],
        "safety": {
            "external_backup_receipt_required": True,
            "external_backup_receipt": "research/backups/v058.yaml",
        },
    }
    result = _validate_backup_receipt(root, "v58", contract, "a" * 40)
    assert result["passed"] is True
    assert result["objects_verified"] == 1
    assert result["code_backup_verified"] is True

    receipt["objects"][0]["sha256"] = "0" * 64
    receipt_path.write_text(yaml.safe_dump(receipt), encoding="utf-8")
    failed = _validate_backup_receipt(root, "v58", contract, "a" * 40)
    assert failed["passed"] is False
    assert any("hash drift" in error for error in failed["errors"])

    missing_policy = {
        "runtime_contract": {"external_backup_receipt_required": False},
        "access_contract": {"allowed_inputs": ["data/input.bin"]},
    }
    incoherent = _validate_backup_receipt(
        root, "v58", missing_policy, "a" * 40
    )
    assert incoherent["passed"] is False
    assert any("storage policy" in error for error in incoherent["errors"])

def test_owner_storage_waiver_is_hash_bound_and_fails_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    source_root = Path(__file__).resolve().parents[1]
    base_path = root / "research/phase_contracts/v058.yaml"
    waiver_path = (
        root / "research/waivers/v058r1_external_backup_owner_waiver.json"
    )
    parent_path = root / "research/experiments/v057.yaml"
    for relative, destination in (
        ("research/phase_contracts/v058.yaml", base_path),
        ("research/experiments/v057.yaml", parent_path),
        (
            "research/waivers/v058r1_external_backup_owner_waiver.json",
            waiver_path,
        ),
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source_root / relative).read_bytes())
    waiver = json.loads(waiver_path.read_text(encoding="utf-8"))
    base_ref = waiver["base_phase_contract"]
    parent_ref = waiver["parent_experiment"]
    scopes = waiver["waived_safeguards"]
    waiver_ref = {
        "path": "research/waivers/v058r1_external_backup_owner_waiver.json",
        "file_sha256": _file_sha(waiver_path),
    }
    contract = {
        "revision": "v058r1",
        "family_id": waiver["family_id"],
        "parent_experiment": parent_ref,
        "supersedes": {
            **base_ref,
            "allowed_change_scope": "external_storage_redundancy_only",
        },
        "operational_waiver": waiver_ref,
        "runtime_contract": {
            "external_backup_receipt_required": False,
            "backup_policy": {
                "mode": "owner_waiver",
                "waiver": waiver_ref,
                "waived_safeguards": scopes,
            },
        },
        "access_contract": {"allowed_inputs": ["data/input.bin"]},
    }
    result = _validate_backup_receipt(root, "v58", contract, "a" * 40)
    assert result["passed"] is True
    assert result["mode"] == "owner_waiver"
    assert result["waiver_verified"] is True
    assert result["objects_verified"] == 0
    assert result["code_backup_verified"] is False

    waiver["waived_safeguards"] = scopes[:-1]
    waiver_path.write_text(json.dumps(waiver), encoding="utf-8")
    failed = _validate_backup_receipt(root, "v58", contract, "a" * 40)
    assert failed["passed"] is False
    assert any("hash drift" in error for error in failed["errors"])

    waiver_path.write_bytes(
        (
            source_root
            / "research/waivers/v058r1_external_backup_owner_waiver.json"
        ).read_bytes()
    )
    missing_revision = dict(contract)
    missing_revision.pop("revision")
    incoherent = _validate_backup_receipt(
        root, "v58", missing_revision, "a" * 40
    )
    assert incoherent["passed"] is False
    assert any("V58r1 revision" in error for error in incoherent["errors"])


def test_research_doctor_requires_every_runtime_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fixture(tmp_path)

    def fake_run(command: list[str], **_: object) -> SimpleNamespace:
        if command[1:3] == ["status", "--porcelain"]:
            return SimpleNamespace(stdout="")
        if command[1] == "remote":
            return SimpleNamespace(stdout="origin\n")
        if command[1:3] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout="a" * 40 + "\n")
        raise AssertionError(command)

    class FakeTensor:
        def __add__(self, _: int) -> "FakeTensor":
            return self

        def cpu(self) -> "FakeTensor":
            return self

        def item(self) -> float:
            return 2.0

    deterministic = {"enabled": False}
    monkeypatch.setattr("tlm.research_workflow.subprocess.run", fake_run)
    monkeypatch.setattr(
        "tlm.research_workflow.shutil.disk_usage",
        lambda _: SimpleNamespace(free=60 * 1024**3),
    )
    monkeypatch.setattr("tlm.research_workflow.torch.backends.mps.is_built", lambda: True)
    monkeypatch.setattr("tlm.research_workflow.torch.backends.mps.is_available", lambda: True)
    monkeypatch.setattr("tlm.research_workflow.torch.ones", lambda *args, **kwargs: FakeTensor())
    monkeypatch.setattr("tlm.research_workflow.torch.mps.synchronize", lambda: None)
    monkeypatch.setattr(
        "tlm.research_workflow.torch.are_deterministic_algorithms_enabled",
        lambda: deterministic["enabled"],
    )
    monkeypatch.setattr(
        "tlm.research_workflow.torch.use_deterministic_algorithms",
        lambda enabled: deterministic.update(enabled=enabled),
    )
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    doctor = research_doctor(tmp_path)
    assert doctor["full_training_ready"] is True
    assert doctor["runtime"]["mps_operational"] is True
    assert doctor["runtime"]["deterministic_algorithms"] is True
    assert doctor["runtime"]["fallback_enabled"] is False
    assert doctor["process_lock"]["active_job_count"] == 0
