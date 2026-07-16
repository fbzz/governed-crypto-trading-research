from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import torch
import yaml


V58_BASE_PHASE_CONTRACT = {
    "path": "research/phase_contracts/v058.yaml",
    "file_sha256": "23883bb354d8bb479778435188fb58962bf43b03f431981e4b963a674a09d868",
}
V58_OWNER_STORAGE_WAIVER = {
    "path": "research/waivers/v058r1_external_backup_owner_waiver.json",
    "file_sha256": "067f23f39937e00c5b9ce40a0248c1683bc9329e195fb10bd9ffd24b69b7e6f9",
}
V59_PHASE_CONTRACT_CANONICAL_SHA256 = (
    "a123d30f6fd54947103cca8239c98879218719ef45f0cef796a435fe7cdf6cbb"
)
V59_UNSEAL_PHASE_CONTRACT = {
    "path": "research/phase_contracts/v059_unseal_r1.yaml",
    "file_sha256": "f23bbc7891a5754ecdc8261492fdb85cfe53ba94d3f0d2a9ca6019a49626771f",
    "canonical_sha256": "45b3676513485fb6644d2da89555fa860d3091d4639bf3cdfe7a2a1c7e3fc0dc",
}
V59_TERMINAL_PHASE_CONTRACT = {
    "path": "research/phase_contracts/v059_terminal_r1.yaml",
    "file_sha256": "e7cc29244ba4060caeb24abf52e176dbd9addc8e1738304a963456a0a06f269a",
    "canonical_sha256": "cf64400501b8ebd93ec23a0c1770910fc1d1f799658039fa00dccb8d33355db2",
}


class ResearchStateError(ValueError):
    """Raised when the current research authorization is inconsistent."""


def _load_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ResearchStateError(f"Expected a mapping: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _resolve_project_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise ResearchStateError("Project path must be a non-blank string")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ResearchStateError(f"Path escapes project root: {relative}") from exc
    return candidate


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _nonempty_unique_string_list(value: object, name: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ResearchStateError(f"{name} must be a non-empty list")
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ResearchStateError(f"{name} must contain non-blank strings")
    strings = list(value)
    if len(strings) != len(set(strings)):
        raise ResearchStateError(f"{name} must contain unique strings")
    return strings


def _validate_access_contract(root: Path, phase_contract: dict[str, Any]) -> None:
    access = phase_contract.get("access_contract")
    if not isinstance(access, dict):
        raise ResearchStateError("Phase access_contract must be a mapping")

    output_dir = access.get("output_dir")
    if (
        not isinstance(output_dir, str)
        or not output_dir.strip()
        or Path(output_dir).is_absolute()
        or Path(output_dir).parts[:1] != ("artifacts",)
        or len(Path(output_dir).parts) < 2
    ):
        raise ResearchStateError(
            "access_contract.output_dir must be a non-blank artifacts subdirectory"
        )
    _resolve_project_path(root, output_dir)

    allowed_inputs = _nonempty_unique_string_list(
        access.get("allowed_inputs"), "access_contract.allowed_inputs"
    )
    allowed_capabilities = _nonempty_unique_string_list(
        access.get("allowed_capabilities"), "access_contract.allowed_capabilities"
    )
    _nonempty_unique_string_list(
        access.get("required_checks"), "access_contract.required_checks"
    )
    forbidden_capabilities = _nonempty_unique_string_list(
        access.get("forbidden_capabilities"),
        "access_contract.forbidden_capabilities",
    )
    for relative in allowed_inputs:
        if Path(relative).is_absolute():
            raise ResearchStateError(
                "access_contract.allowed_inputs must be repository-relative paths"
            )
        _resolve_project_path(root, relative)
    overlap = sorted(set(allowed_capabilities) & set(forbidden_capabilities))
    if overlap:
        raise ResearchStateError(
            f"Phase access contract allows and forbids the same capabilities: {overlap}"
        )


def _validate_authorization_receipt(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
) -> None:
    receipt = phase_contract.get("authorization_receipt")
    if not isinstance(receipt, dict):
        raise ResearchStateError("Phase authorization_receipt must be a mapping")
    required = {"path", "file_sha256", "registered_result_sha256"}
    missing = sorted(required - set(receipt))
    if missing:
        raise ResearchStateError(
            f"Phase authorization receipt is missing keys: {missing}"
        )
    if (
        not isinstance(receipt.get("path"), str)
        or not _is_sha256(receipt.get("file_sha256"))
        or not _is_sha256(receipt.get("registered_result_sha256"))
    ):
        raise ResearchStateError("Phase authorization receipt path or hashes are invalid")

    experiment_result = experiment.get("result")
    if not isinstance(experiment_result, dict):
        raise ResearchStateError("Experiment result reference must be a mapping")
    expected = {
        key: experiment_result.get(key)
        for key in ("path", "file_sha256", "registered_result_sha256")
    }
    actual = {key: receipt.get(key) for key in expected}
    if actual != expected or any(not value for value in expected.values()):
        raise ResearchStateError(
            "Phase authorization receipt differs from the experiment result"
        )

    receipt_path = _resolve_project_path(root, receipt["path"])
    if (
        not receipt_path.is_file()
        or _sha256_file(receipt_path) != receipt["file_sha256"]
    ):
        raise ResearchStateError("Phase authorization receipt file/hash drift")
    try:
        result = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchStateError(
            "Phase authorization receipt is not valid JSON"
        ) from exc
    if not isinstance(result, dict):
        raise ResearchStateError("Phase authorization receipt JSON must be a mapping")
    embedded_hash = result.pop("result_sha256", None)
    canonical_hash = _canonical_sha256(result)
    if embedded_hash != receipt["registered_result_sha256"] or (
        canonical_hash != receipt["registered_result_sha256"]
    ):
        raise ResearchStateError(
            "Phase authorization receipt canonical result hash drift"
        )
    action = state["authorized_next_action"]
    if result.get("decision") != action:
        raise ResearchStateError(
            "Phase authorization receipt decision differs from current authorization"
        )
    if "decision" in receipt and receipt.get("decision") != action:
        raise ResearchStateError(
            "Phase authorization receipt metadata decision differs from current authorization"
        )


def _validate_v59_completion_receipt(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
) -> None:
    receipt = phase_contract.get("completion_receipt")
    experiment_receipt = experiment.get("completion_receipt")
    if not isinstance(receipt, dict) or not isinstance(experiment_receipt, dict):
        raise ResearchStateError("V59 requires the completed V58 receipt binding")
    keys = ("path", "file_sha256", "registered_completion_sha256")
    if {key: receipt.get(key) for key in keys} != {
        key: experiment_receipt.get(key) for key in keys
    }:
        raise ResearchStateError("V59 completion receipt differs from V58 experiment")
    if (
        not isinstance(receipt.get("path"), str)
        or not _is_sha256(receipt.get("file_sha256"))
        or not _is_sha256(receipt.get("registered_completion_sha256"))
    ):
        raise ResearchStateError("V59 completion receipt path or hashes are invalid")
    path = _resolve_project_path(root, receipt["path"])
    if not path.is_file() or _sha256_file(path) != receipt["file_sha256"]:
        raise ResearchStateError("V59 completion receipt file/hash drift")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchStateError("V59 completion receipt is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ResearchStateError("V59 completion receipt JSON must be a mapping")
    embedded_hash = value.pop("completion_receipt_sha256", None)
    if embedded_hash != receipt["registered_completion_sha256"] or (
        _canonical_sha256(value) != receipt["registered_completion_sha256"]
    ):
        raise ResearchStateError("V59 completion receipt canonical hash drift")
    if value.get("decision") != state["authorized_next_action"]:
        raise ResearchStateError("V59 completion receipt decision drift")


def _validate_v59_input_bindings(
    root: Path, phase_contract: dict[str, Any]
) -> None:
    access = phase_contract["access_contract"]
    allowed = set(access["allowed_inputs"])
    input_contract = phase_contract.get("input_contract")
    if not isinstance(input_contract, dict):
        raise ResearchStateError("V59 input contract is missing")
    direct = input_contract.get("expected_file_sha256_by_path")
    scalers = input_contract.get("expected_scaler_file_sha256_by_path")
    if not isinstance(direct, dict) or not isinstance(scalers, dict):
        raise ResearchStateError("V59 direct input path/hash bindings are missing")
    for name, bindings in (("input", direct), ("scaler", scalers)):
        if not bindings or not all(
            isinstance(path, str) and path and _is_sha256(digest)
            for path, digest in bindings.items()
        ):
            raise ResearchStateError(f"V59 {name} path/hash bindings are invalid")

    checkpoint_paths = {path for path in allowed if path.endswith("/final.pt")}
    scaler_paths = {path for path in allowed if path.endswith("/scaler.json")}
    if (
        len(checkpoint_paths) != 36
        or len(scaler_paths) != 12
        or set(scalers) != scaler_paths
        or set(direct) | scaler_paths | checkpoint_paths != allowed
        or (set(direct) & (scaler_paths | checkpoint_paths))
        or (scaler_paths & checkpoint_paths)
    ):
        raise ResearchStateError("V59 allowed inputs and path/hash bindings differ")

    for path_text, expected_hash in {**direct, **scalers}.items():
        path = _resolve_project_path(root, path_text)
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise ResearchStateError(f"V59 input file/hash drift: {path_text}")

    manifest_path_text = (
        "artifacts/v58_state_conditioned_multi_horizon_training/"
        "checkpoint_manifest.json"
    )
    manifest_path = _resolve_project_path(root, manifest_path_text)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchStateError("V59 checkpoint manifest is not valid JSON") from exc
    if not isinstance(manifest, dict):
        raise ResearchStateError("V59 checkpoint manifest must be a mapping")
    registered_manifest_hash = manifest.pop("manifest_sha256", None)
    if (
        registered_manifest_hash
        != "336d4b38a119db479c20a96d1d34ea73b0c967c85b38f4c2893c217ace37d384"
        or _canonical_sha256(manifest) != registered_manifest_hash
        or manifest.get("checkpoint_count") != 36
        or manifest.get("selected_jobs") != []
        or manifest.get("active_jobs") != []
    ):
        raise ResearchStateError("V59 checkpoint manifest semantic drift")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 36:
        raise ResearchStateError("V59 checkpoint manifest job grid drift")
    by_path: dict[str, dict[str, Any]] = {}
    for row in jobs:
        if not isinstance(row, dict):
            raise ResearchStateError("V59 checkpoint manifest row is invalid")
        path_text = row.get("checkpoint_path")
        if not isinstance(path_text, str) or path_text in by_path:
            raise ResearchStateError("V59 checkpoint paths must be unique strings")
        by_path[path_text] = row
    if set(by_path) != checkpoint_paths:
        raise ResearchStateError("V59 checkpoint allowlist differs from V58 manifest")
    for path_text, row in by_path.items():
        expected_hash = row.get("checkpoint_sha256")
        expected_size = row.get("checkpoint_size_bytes")
        path = _resolve_project_path(root, path_text)
        if (
            row.get("status") != "completed"
            or not _is_sha256(expected_hash)
            or not isinstance(expected_size, int)
            or expected_size < 1
            or not path.is_file()
            or path.stat().st_size != expected_size
            or _sha256_file(path) != expected_hash
        ):
            raise ResearchStateError(f"V59 checkpoint file/hash drift: {path_text}")


def _validate_v59_storage_waiver(
    root: Path, phase_contract: dict[str, Any]
) -> None:
    expected_ref = {
        "path": "research/waivers/v059_local_artifacts_owner_waiver.json",
        "file_sha256": "c31f88585e248eac17f6c4f1d0c03df9029d4712f9c5c2180bbf677b68c73ae0",
    }
    storage = phase_contract.get("storage_contract")
    if (
        not isinstance(storage, dict)
        or phase_contract.get("operational_waiver") != expected_ref
        or storage.get("waiver") != expected_ref
    ):
        raise ResearchStateError("V59 storage waiver reference drift")
    path = _resolve_project_path(root, expected_ref["path"])
    if not path.is_file() or _sha256_file(path) != expected_ref["file_sha256"]:
        raise ResearchStateError("V59 storage waiver file/hash drift")
    try:
        waiver = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchStateError("V59 storage waiver is not valid JSON") from exc
    expected_scopes = [
        "external_input_copy",
        "external_code_copy",
        "external_prediction_position_and_outcome_copy",
    ]
    if not isinstance(waiver, dict) or (
        waiver.get("schema_version") != "tlm-owner-storage-waiver/v1"
        or waiver.get("waiver_id") != "v059_local_artifacts_owner_waiver"
        or waiver.get("phase") != "v59"
        or waiver.get("family_id") != phase_contract.get("family_id")
        or waiver.get("authorized_action")
        != phase_contract.get("authorized_next_action")
        or waiver.get("accepted_by") != "repository_owner"
        or waiver.get("risk_acceptance") is not True
        or waiver.get("waived_safeguards") != expected_scopes
        or waiver.get("does_not_authorize_outcome_unseal") is not True
        or waiver.get("target_assets_status") != "sealed"
        or waiver.get("parent_experiment") != phase_contract.get("parent_experiment")
    ):
        raise ResearchStateError("V59 storage waiver content drift")


def _validate_v59_outcome_boundary(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
    *,
    validate_files: bool = True,
) -> None:
    if state.get("authorized_phase") != "v59":
        return
    expected_action = "authorize_v59_frozen_adaptive_development_evaluation_only"
    if state.get("authorized_next_action") != expected_action:
        raise ResearchStateError("V59 authorization token drift")
    expected_status = "non_target_training_passed_development_evaluation_not_started"
    if (
        state.get("active_family_status") != expected_status
        or state.get("last_completed_phase")
        != "v58_state_conditioned_multi_horizon_training"
        or state.get("last_completed_result")
        != "artifacts/v58_state_conditioned_multi_horizon_training/result.json"
        or experiment.get("status") != "passed"
        or experiment.get("phase") != "frozen_non_target_training"
    ):
        raise ResearchStateError("V59 active-family completion state drift")
    active_rows = [
        row
        for row in state.get("families", [])
        if isinstance(row, dict)
        and row.get("family_id") == state.get("active_family_id")
    ]
    if len(active_rows) != 1 or (
        active_rows[0].get("trained") is not True
        or active_rows[0].get("status") != expected_status
        or active_rows[0].get("terminal_phase") is not None
    ):
        raise ResearchStateError("V59 active trained-family registry drift")
    expected_families = [
        {
            "family_id": "tlm_multi_asset_target_transfer_v2",
            "trained": True,
            "status": "retired",
            "terminal_phase": "v37",
        },
        {
            "family_id": "tlm_cross_sectional_rank_excess_medium_v1",
            "trained": True,
            "status": "retired",
            "terminal_phase": "v45",
        },
        {
            "family_id": "tlm_joint_absolute_relative_triplet_medium_v1",
            "trained": True,
            "status": "retired",
            "terminal_phase": "v50",
        },
        {
            "family_id": "tlm_state_conditioned_multi_horizon_quantile_small_v1",
            "trained": True,
            "status": expected_status,
            "terminal_phase": None,
        },
    ]
    if state.get("families") != expected_families:
        raise ResearchStateError("V59 family registry drift")
    expected_forbidden = [
        "access_target_assets",
        "access_development_evaluation_outcomes_before_explicit_unseal_authorization",
        "refit_candidate_model_or_scaler",
        "train_or_modify_any_checkpoint",
        "select_discard_or_weight_checkpoints",
        "compute_performance_or_pnl_before_explicit_unseal_authorization",
        "implement_v60_or_later",
        "paper_shadow_live_or_real_money_trading",
    ]
    if state.get("forbidden_capabilities") != expected_forbidden:
        raise ResearchStateError("V59 current-state forbidden capabilities drift")
    expected_safety = {
        "prepare_requires_clean_committed_git": True,
        "target_assets_remain_sealed": True,
        "all_thirty_six_checkpoints_must_be_used_without_selection": True,
        "outcome_blind_prepare_required": True,
        "explicit_new_user_authorization_required_before_unseal": True,
        "generic_continue_is_unseal_authorization": False,
        "maximum_development_outcome_unseals": 1,
        "storage_policy": (
            "repository_owner_accepted_local_only_without_external_redundancy"
        ),
        "storage_waiver": {
            "path": "research/waivers/v059_local_artifacts_owner_waiver.json",
            "file_sha256": (
                "c31f88585e248eac17f6c4f1d0c03df9029d4712f9c5c2180bbf677b68c73ae0"
            ),
        },
    }
    if (
        state.get("deployable_strategy") is not False
        or state.get("evidence_tier") != "causal_non_target_training_only"
        or state.get("safety") != expected_safety
    ):
        raise ResearchStateError("V59 current-state safety or evidence drift")
    _validate_v59_completion_receipt(root, state, experiment, phase_contract)
    if validate_files:
        _validate_v59_input_bindings(root, phase_contract)
        _validate_v59_storage_waiver(root, phase_contract)

    commands = phase_contract.get("commands")
    expected_commands = {
        "prepare": (
            "PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src python3 -m tlm "
            "state-conditioned-multi-horizon-evaluation-prepare "
            "--config configs/v59_state_conditioned_multi_horizon_evaluation.yaml"
        ),
        "unseal": (
            "PYTHONPATH=src python3 -m tlm "
            "state-conditioned-multi-horizon-evaluation-unseal "
            "--config configs/v59_state_conditioned_multi_horizon_evaluation.yaml"
        ),
        "replay": (
            "PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src python3 -m tlm "
            "state-conditioned-multi-horizon-evaluation-replay "
            "--config configs/v59_state_conditioned_multi_horizon_evaluation.yaml"
        ),
    }
    if commands != expected_commands:
        raise ResearchStateError("V59 command contract drift")
    prepare_command = commands.get("prepare")
    unseal_command = commands.get("unseal")
    replay_command = commands.get("replay")
    if state.get("authorized_command") != prepare_command or (
        state.get("authorized_command") == unseal_command
    ):
        raise ResearchStateError("Only the V59 outcome-blind prepare command is authorized")

    one_shot = phase_contract.get("one_shot_contract")
    if not isinstance(one_shot, dict) or one_shot.get("current_stage") != (
        "outcome_blind_prepare"
    ):
        raise ResearchStateError("V59 must begin at the outcome-blind prepare stage")
    prepare = one_shot.get("prepare")
    unseal = one_shot.get("unseal")
    completion = one_shot.get("completion")
    replay = one_shot.get("replay")
    if not all(
        isinstance(section, dict)
        for section in (prepare, unseal, completion, replay)
    ):
        raise ResearchStateError("V59 one-shot stage contracts are incomplete")
    expected_prepare = {
        "development_evaluation_outcome_value_reads": 0,
        "target_asset_loads": 0,
        "freeze_all_thirty_six_checkpoint_predictions": True,
        "freeze_candidate_and_all_control_positions": True,
        "freeze_exact_outcome_key_and_column_request": True,
        "behavior_gates_before_outcomes": True,
    }
    if any(prepare.get(key) != value for key, value in expected_prepare.items()):
        raise ResearchStateError("V59 outcome-blind prepare boundary drift")
    if (
        phase_contract.get("prepare_pass_action")
        != "await_explicit_v59_registered_outcome_unseal_authorization"
        or prepare.get("pass_action") != phase_contract.get("prepare_pass_action")
        or prepare.get("pass_authorizes_unseal") is not False
        or prepare.get("failure_action")
        != phase_contract.get("prepare_failure_action")
    ):
        raise ResearchStateError("V59 prepare transition action drift")
    transition = one_shot.get("stage_transition")
    expected_transition = {
        "automatic_current_state_transition_after_prepare": False,
        "explicit_new_user_authorization_required": True,
        "required_unseal_stage_contract_revision": "v059_unseal_r1",
        "unseal_stage_contract_created_only_after_prepare_and_explicit_authorization": True,
        "current_yaml_must_bind_unseal_stage_contract_before_source_read": True,
        "unseal_stage_authorized_command": "exact_commands_unseal",
        "generic_continue_or_bora_can_create_unseal_stage": False,
    }
    if not isinstance(transition, dict) or any(
        transition.get(key) != value for key, value in expected_transition.items()
    ):
        raise ResearchStateError("V59 prepare-to-unseal stage transition drift")
    required_bindings = transition.get("required_hash_bindings")
    if required_bindings != [
        "base_v059_phase_contract",
        "evaluation_spec",
        "prepare_manifest",
        "prepare_receipt",
        "explicit_user_authorization_receipt",
    ]:
        raise ResearchStateError("V59 unseal stage hash bindings drift")
    expected_unseal = {
        "authorized_now": False,
        "explicit_new_user_authorization_required": True,
        "generic_continue_is_authorization": False,
        "authorization_must_bind_evaluation_spec_and_prepare_receipt_hashes": True,
        "maximum_unseal_count": 1,
        "atomic_authorization_receipt_before_source_read": True,
    }
    if any(unseal.get(key) != value for key, value in expected_unseal.items()):
        raise ResearchStateError("V59 explicit one-shot unseal boundary drift")
    if unseal.get("allowed_columns") != [
        "date",
        "symbol",
        "target_h1_open_to_open_log_return",
        "target_h3_open_to_open_log_return",
        "target_h7_open_to_open_log_return",
    ]:
        raise ResearchStateError("V59 unseal outcome projection drift")
    if completion.get("prediction_or_position_regeneration_after_unseal") is not False:
        raise ResearchStateError("V59 post-unseal prediction freeze drift")
    expected_replay = {
        "source_outcome_reads": 0,
        "new_unseal_authorization_receipts": 0,
        "new_outcome_packets": 0,
        "new_inference": 0,
        "use_existing_prepare_and_outcome_packets_only": True,
    }
    if any(replay.get(key) != value for key, value in expected_replay.items()):
        raise ResearchStateError("V59 replay outcome boundary drift")

    parquet = phase_contract.get("prepare_parquet_access_contract")
    if not isinstance(parquet, dict) or (
        parquet.get("full_table_materialization_then_filtering_allowed") is not False
        or parquet.get("development_label_projection_during_prepare") != []
        or parquet.get("development_outcome_columns_materialized_during_prepare")
        != []
        or parquet.get("development_outcome_value_reads_during_prepare") != 0
        or parquet.get("reader_requirement")
        != "projected_columns_and_predicate_dnf_must_be_passed_to_the_parquet_reader"
    ):
        raise ResearchStateError("V59 prepare Parquet outcome boundary drift")
    if parquet.get("train_label_projection_only") != [
        "date",
        "symbol",
        "target_h7_maturity_date",
        "target_h7_open_to_open_log_return",
        "multi_horizon_label_complete",
    ]:
        raise ResearchStateError("V59 train-only label projection drift")

    scaler_bindings = phase_contract.get("input_contract", {}).get(
        "expected_scaler_file_sha256_by_path"
    )
    allowed_scalers = {
        path
        for path in phase_contract["access_contract"]["allowed_inputs"]
        if path.endswith("/scaler.json")
    }
    if not isinstance(scaler_bindings, dict) or (
        set(scaler_bindings) != allowed_scalers
        or not all(
            isinstance(value, str) and len(value) == 64
            for value in scaler_bindings.values()
        )
    ):
        raise ResearchStateError("V59 scaler path/hash bindings drift")

    inference = phase_contract.get("inference_contract")
    seed_aggregation = (
        inference.get("seed_aggregation") if isinstance(inference, dict) else None
    )
    if not isinstance(seed_aggregation, dict) or (
        inference.get("model_mode") != "eval"
        or inference.get("autograd_context") != "torch_inference_mode"
        or inference.get("batch_size") != 128
        or seed_aggregation.get("ordered_members") != [42, 7, 123]
        or seed_aggregation.get("accumulator_device") != "cpu"
        or seed_aggregation.get("accumulator_dtype") != "float64"
        or seed_aggregation.get("arithmetic")
        != "ordered_sum_seed_42_then_7_then_123_divided_by_exact_integer_3"
    ):
        raise ResearchStateError("V59 deterministic inference contract drift")

    control = phase_contract.get("control_contract")
    momentum = (
        control.get("weekly_dual_momentum_30")
        if isinstance(control, dict)
        else None
    )
    if not isinstance(momentum, dict) or (
        momentum.get("required_consecutive_calendar_rows") != 30
        or momentum.get("every_required_value_must_be_finite") is not True
        or momentum.get("scan_farther_back_to_replace_missing_values") is not False
        or momentum.get("incomplete_asset_score") != "unavailable"
    ):
        raise ResearchStateError("V59 momentum missing-data contract drift")

    accounting = phase_contract.get("accounting_contract")
    if not isinstance(accounting, dict) or (
        accounting.get("episode_initial_position") != "cash"
        or accounting.get("state_carry_across_episodes") is not False
        or accounting.get("final_liquidation_row")
        != "last_registered_return_row_of_each_episode"
        or accounting.get("final_row_order")
        != "apply_last_held_position_gross_return_then_charge_final_liquidation_turnover_on_same_row"
    ):
        raise ResearchStateError("V59 episode accounting boundary drift")

    behavior = phase_contract.get("outcome_blind_gate_contract")
    required_behavior_gates = {
        "input_bindings_exact",
        "parquet_access_exact",
        "checkpoint_grid_exact",
        "scaler_grid_exact",
        "targets_absent",
        "development_outcomes_sealed",
        "prediction_keys_exact",
        "predictions_finite",
        "seed_aggregation_exact",
        "candidate_positions_exact",
        "control_positions_exact",
        "turnover_structure_exact",
        "outcome_request_exact",
        "prepare_packet_complete_atomic",
        "prepare_replay_exact",
    }
    if not isinstance(behavior, dict) or (
        behavior.get("evaluated_before_any_development_outcome_value_read") is not True
        or behavior.get("every_gate_mandatory") is not True
        or set(behavior.get("gates", {})) != required_behavior_gates
        or behavior.get("failure_contract", {}).get(
            "always_keep_development_outcomes_and_targets_sealed"
        )
        is not True
        or behavior.get("failure_contract", {}).get("unseal_authorized_after_failure")
        is not False
    ):
        raise ResearchStateError("V59 outcome-blind behavior gates drift")

    storage = phase_contract.get("storage_contract")
    if not isinstance(storage, dict) or (
        storage.get("phase") != "v59"
        or storage.get("mode")
        != "repository_owner_accepted_local_only_without_external_redundancy"
        or storage.get("accepted_by") != "repository_owner"
        or storage.get("required_local_artifacts_may_be_deleted") is not False
        or storage.get("atomic_writes_and_content_hashes_required") is not True
        or storage.get("applies_only_to_phase") != "v59"
    ):
        raise ResearchStateError("V59 storage policy drift")
    packet_files = phase_contract.get("artifact_contract", {}).get(
        "result_packet_files", []
    )
    if "outcome_packet.parquet" not in packet_files:
        raise ResearchStateError("V59 outcome packet is absent from artifact bindings")

    target = phase_contract.get("target_contract")
    if not isinstance(target, dict) or (
        target.get("symbols") != state["target_assets"].get("symbols")
        or target.get("status") != "sealed"
        or target.get("target_assets_loaded") != []
        or target.get("target_predictions") != 0
        or target.get("target_pnl_evaluations") != 0
    ):
        raise ResearchStateError("V59 target-asset seal drift")
    if phase_contract.get("pass_action") != phase_contract.get("gate_contract", {}).get(
        "pass"
    ) or phase_contract.get("failure_action") != phase_contract.get(
        "gate_contract", {}
    ).get("failure"):
        raise ResearchStateError("V59 gate actions differ from the phase actions")

    evaluation = phase_contract.get("evaluation_cells")
    if not isinstance(evaluation, dict) or (
        evaluation.get("origins")
        != {
            "origin_2024": {
                "signal_start": "2024-01-01",
                "signal_end": "2024-12-23",
                "maturity_end": "2024-12-31",
            },
            "origin_2025": {
                "signal_start": "2025-01-01",
                "signal_end": "2025-12-23",
                "maturity_end": "2025-12-31",
            },
        }
        or evaluation.get("geometries") != ["expanding", "rolling"]
        or evaluation.get("folds") != [1, 2, 3]
        or evaluation.get("seeds") != [42, 7, 123]
        or evaluation.get("checkpoint_count") != 36
        or evaluation.get("selected_jobs") != []
        or evaluation.get("checkpoint_weighting_allowed") is not False
        or evaluation.get("expanding_rolling_blending_allowed") is not False
        or evaluation.get("mandatory_fold_cells") != 12
    ):
        raise ResearchStateError("V59 evaluation grid drift")

    feature = phase_contract.get("feature_tensor_contract")
    expected_feature_order = [
        "log_open_to_open_return",
        "log_close_to_close_return",
        "log_high_low_range",
        "log_close_open_return",
        "log1p_quote_volume_change",
        "log1p_trade_count_change",
        "rolling_realized_volatility_7d",
        "rolling_realized_volatility_30d",
        "within_triplet_relative_strength",
    ]
    relative = feature.get("relative_feature") if isinstance(feature, dict) else None
    if not isinstance(relative, dict) or (
        feature.get("model_feature_order") != expected_feature_order
        or feature.get("exact_output_shape") != [None, 256, 3, 9]
        or feature.get("transform_compute_dtype") != "float64"
        or feature.get("model_input_dtype_after_transform") != "float32"
        or relative.get("source") != "raw_log_close_to_close_return"
        or relative.get("formula")
        != "raw_asset_value_minus_arithmetic_mean_of_the_three_raw_same_date_triplet_values"
        or relative.get("transform")
        != "divide_by_matching_cell_log_close_to_close_return_population_std"
        or relative.get("subtract_scaler_mean") is not False
        or relative.get("source_feature_index") != 1
        or relative.get("output_feature_index") != 8
    ):
        raise ResearchStateError("V59 nine-feature tensor contract drift")

    linear = phase_contract.get("linear_control_contract")
    if not isinstance(linear, dict) or (
        linear.get("fit_scope") != "exact_origin_geometry_fold_train_role_only"
        or linear.get("estimator") != "ridge"
        or linear.get("alpha") != 1.0
        or linear.get("solver") != "svd"
        or linear.get("validation_or_development_evaluation_fit_allowed") is not False
        or linear.get("target") != "target_h7_open_to_open_log_return"
    ):
        raise ResearchStateError("V59 train-only linear control drift")

    accounting_contract = phase_contract.get("accounting_contract")
    if not isinstance(accounting_contract, dict) or (
        accounting_contract.get("reporting_cost_bps") != [10, 20, 30, 50]
        or accounting_contract.get("mandatory_cost_bps") != [10, 20, 30]
        or accounting_contract.get("diagnostic_cost_bps") != [50]
        or accounting_contract.get("positions_are_identical_across_reporting_costs")
        is not True
    ):
        raise ResearchStateError("V59 accounting cost contract drift")

    bootstrap = phase_contract.get("bootstrap_contract")
    if not isinstance(bootstrap, dict) or (
        bootstrap.get("method") != "paired_circular_moving_block"
        or bootstrap.get("paths") != 10000
        or bootstrap.get("block_lengths") != [7, 21, 63]
        or bootstrap.get("quantile_method") != "linear"
        or bootstrap.get("lower_quantile") != 0.05
    ):
        raise ResearchStateError("V59 bootstrap contract drift")

    expected_mandatory_gates = {
        "per_origin_geometry_fold_candidate_return_at_10bps": "strictly_positive",
        "per_origin_geometry_fold_h7_q20_coverage": {
            "lower": 0.15,
            "lower_inclusive": True,
            "upper": 0.25,
            "upper_inclusive": True,
        },
        "per_origin_geometry_fold_h7_q50_pairwise_accuracy": "strictly_above_0.5",
        "per_origin_geometry_fold_h7_q50_spearman": "strictly_positive",
        "per_origin_geometry_aggregate_candidate_return_vs_every_control_at_10_20_30bps": "strictly_greater",
        "per_origin_geometry_aggregate_sharpe_vs_weekly_dual_momentum_at_10_20_30bps": "strictly_greater",
        "per_origin_geometry_fold_candidate_bootstrap_p05_at_every_mandatory_cost_and_block": "strictly_positive",
        "per_origin_geometry_fold_candidate_minus_each_control_bootstrap_p05_at_every_mandatory_cost_and_block": "strictly_positive",
        "per_origin_geometry_fold_and_aggregate_maximum_drawdown_at_every_mandatory_cost": "at_most_0.35",
        "per_origin_geometry_aggregate_turnover_vs_weekly_dual_momentum_at_10bps": "at_most_control",
    }
    gates = phase_contract.get("gate_contract")
    if not isinstance(gates, dict) or (
        gates.get("aggregate_rescue_allowed") is not False
        or gates.get("mandatory") != expected_mandatory_gates
        or gates.get("diagnostic_cost_or_overall_series_can_change_decision") is not False
    ):
        raise ResearchStateError("V59 mandatory scientific gate drift")

    access = phase_contract["access_contract"]
    if access.get("output_dir") != (
        "artifacts/v59_state_conditioned_multi_horizon_evaluation"
    ):
        raise ResearchStateError("V59 output directory drift")
    required_forbidden = {
        "development_evaluation_outcome_value_access_during_prepare",
        "outcome_unseal_without_new_explicit_user_authorization",
        "second_outcome_unseal_or_source_outcome_reread",
        "target_asset_access_inference_prediction_or_pnl",
        "prediction_or_position_regeneration_after_outcome_unseal",
    }
    if not required_forbidden.issubset(set(access["forbidden_capabilities"])):
        raise ResearchStateError("V59 forbidden one-shot capabilities are incomplete")
    required_checks = {
        "only_train_role_label_values_materialized_for_the_linear_control",
        "development_evaluation_outcome_value_reads_equal_zero_during_prepare",
        "no_performance_metric_pnl_gate_or_outcome_unseal_during_prepare",
        "exact_prepare_parquet_projections_predicates_and_materialized_roles_match",
        "prepare_pass_stops_before_a_separate_hash_bound_unseal_stage_contract",
    }
    if not required_checks.issubset(set(access["required_checks"])):
        raise ResearchStateError("V59 outcome-blind required checks are incomplete")
    if _canonical_sha256(phase_contract) != V59_PHASE_CONTRACT_CANONICAL_SHA256:
        raise ResearchStateError("V59 frozen phase-contract semantic hash drift")


def _validate_v59_unseal_boundary(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
) -> None:
    """Validate the post-prepare, hash-authorized V59 one-shot stage."""

    if phase_contract.get("stage_revision") != "v059_unseal_r1":
        raise ResearchStateError("V59 unseal stage revision drift")
    if (
        state.get("authorized_phase") != "v59"
        or state.get("authorized_next_action")
        != "execute_v59_exactly_one_registered_outcome_unseal_and_complete_evaluation"
        or state.get("active_family_status")
        != "development_evaluation_prepared_outcomes_sealed_exactly_one_unseal_authorized"
        or state.get("last_completed_phase") != "v59_outcome_blind_prepare"
        or state.get("last_completed_result")
        != "artifacts/v59_state_conditioned_multi_horizon_evaluation/prepare_receipt.json"
        or state.get("evidence_tier")
        != "adaptive_historical_development_evaluation_prepared_outcomes_sealed"
    ):
        raise ResearchStateError("V59 unseal current-state boundary drift")
    expected_command = (
        "PYTHONPATH=src python3 -m tlm "
        "state-conditioned-multi-horizon-evaluation-unseal "
        "--config configs/v59_state_conditioned_multi_horizon_evaluation.yaml"
    )
    commands = phase_contract.get("commands")
    if (
        not isinstance(commands, dict)
        or commands.get("unseal") != expected_command
        or state.get("authorized_command") != expected_command
        or phase_contract.get("authorized_command") != expected_command
    ):
        raise ResearchStateError("V59 unseal command drift")
    replay_command = (
        "PYTHONPATH=src python3 -m tlm "
        "state-conditioned-multi-horizon-evaluation-replay "
        "--config configs/v59_state_conditioned_multi_horizon_evaluation.yaml"
    )
    if commands.get("replay") != replay_command:
        raise ResearchStateError("V59 replay command drift")

    base = phase_contract.get("base_phase_contract")
    expected_base = {
        "path": "research/phase_contracts/v059.yaml",
        "file_sha256": "321c6a805b94f73d441def62af7478337b5e33f5f23631808dba032a376df6a2",
        "canonical_sha256": V59_PHASE_CONTRACT_CANONICAL_SHA256,
    }
    if base != expected_base:
        raise ResearchStateError("V59 base phase binding drift")
    base_path = _resolve_project_path(root, base["path"])
    if (
        not base_path.is_file()
        or _sha256_file(base_path) != base["file_sha256"]
        or _canonical_sha256(_load_mapping(base_path)) != base["canonical_sha256"]
    ):
        raise ResearchStateError("V59 base phase file/hash drift")

    prepare = phase_contract.get("prepare_packet")
    expected_prepare = {
        "evaluation_spec": (
            "a20aa7aed28c56f13445f2aa07993318723bb2a1e22773664a4db65e82d271fa",
            "31bd555f333e0b3c039f39e09b269b30446c0b372df55946e8bf473283308291",
            "evaluation_spec_sha256",
        ),
        "prepare_manifest": (
            "329094f33bd5bd29e58e1f31edcc07f6cc472ed80824922e0e0bd86b59e4670f",
            "27e171d3d540f0edc0142ad06c3e892aa73d3c8fb39a1020727ca16c17dd1eb5",
            "prepare_manifest_sha256",
        ),
        "prepare_receipt": (
            "7ba2bd928922f50d1b9893426111ae61e113e88c5a1f7ffd99dbe6246c8bef29",
            "be930d7a5166bbba597748c6e8a4a7144ef66d3fffb213d097d77bf64e39e30e",
            "prepare_receipt_sha256",
        ),
        "outcome_request": (
            "2ed2a1d2c271b9011e46e6aa1d35566c4efa082522d5ef7c4be42c0c252b7c34",
            "6414d3d842e55c73ab85d786666301e1e18c6d41a5f1c003cf73250dbe2aea56",
            "outcome_request_sha256",
        ),
    }
    if not isinstance(prepare, dict):
        raise ResearchStateError("V59 prepare packet binding is missing")
    for name, (file_hash, canonical_hash, self_field) in expected_prepare.items():
        reference = prepare.get(name)
        if not isinstance(reference, dict) or (
            reference.get("file_sha256") != file_hash
            or reference.get("canonical_sha256") != canonical_hash
        ):
            raise ResearchStateError(f"V59 {name} binding drift")
        path = _resolve_project_path(root, reference.get("path"))
        if not path.is_file() or _sha256_file(path) != file_hash:
            raise ResearchStateError(f"V59 {name} file/hash drift")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ResearchStateError(f"V59 {name} is not valid JSON") from exc
        registered = value.pop(self_field, None)
        if registered != canonical_hash or _canonical_sha256(value) != canonical_hash:
            raise ResearchStateError(f"V59 {name} canonical hash drift")

    explicit = phase_contract.get("explicit_user_authorization")
    if not isinstance(explicit, dict) or (
        explicit.get("canonical_sha256")
        != "f42d9aa42b72109126a0d420c564804683a266c31bb310b4a58ba28171efd64b"
        or _canonical_sha256(explicit.get("payload"))
        != explicit.get("canonical_sha256")
    ):
        raise ResearchStateError("V59 explicit user authorization hash drift")
    payload = explicit["payload"]
    if (
        payload.get("authorized_action") != state.get("authorized_next_action")
        or payload.get("maximum_unseal_count") != 1
        or payload.get("no_retuning_or_regeneration") is not True
        or payload.get("target_assets") != state.get("target_assets", {}).get("symbols")
        or payload.get("target_assets_status") != "sealed"
    ):
        raise ResearchStateError("V59 explicit user authorization scope drift")

    outcome = phase_contract.get("outcome_access_contract")
    if not isinstance(outcome, dict) or (
        outcome.get("source")
        != "data/processed/state_conditioned_multi_horizon_labels_v57.parquet"
        or outcome.get("source_file_sha256")
        != "6d12e9d49f1be807a1eba5596295fa40f43c3d89745a9a39f3e3f42d76544f50"
        or outcome.get("exact_key_count") != 20410
        or outcome.get("exact_key_sha256")
        != "a135b47500fca47cf5c7c936d7cefbfedfab2251d30acb84a5971b45159f66b6"
        or outcome.get("maximum_source_reads") != 1
        or outcome.get("authorization_receipt_before_source_read") is not True
    ):
        raise ResearchStateError("V59 exact outcome access contract drift")
    source = _resolve_project_path(root, outcome["source"])
    if not source.is_file() or _sha256_file(source) != outcome["source_file_sha256"]:
        raise ResearchStateError("V59 outcome source file/hash drift")

    expected_forbidden = [
        "access_target_assets",
        "second_development_outcome_unseal_or_source_reread",
        "refit_candidate_model_or_scaler",
        "train_or_modify_any_checkpoint",
        "select_discard_or_weight_checkpoints",
        "regenerate_predictions_or_positions",
        "implement_v60_or_later",
        "paper_shadow_live_or_real_money_trading",
    ]
    if state.get("forbidden_capabilities") != expected_forbidden:
        raise ResearchStateError("V59 unseal forbidden capabilities drift")
    if state.get("deployable_strategy") is not False:
        raise ResearchStateError("V59 unseal cannot mark a strategy deployable")
    target = phase_contract.get("target_contract")
    if not isinstance(target, dict) or (
        target.get("symbols") != state.get("target_assets", {}).get("symbols")
        or target.get("status") != "sealed"
        or target.get("target_assets_loaded") != []
        or target.get("target_predictions") != 0
        or target.get("target_pnl_evaluations") != 0
    ):
        raise ResearchStateError("V59 unseal target seal drift")
    if _canonical_sha256(phase_contract) != V59_UNSEAL_PHASE_CONTRACT["canonical_sha256"]:
        raise ResearchStateError("V59 unseal phase-contract semantic hash drift")


def _validate_v59_terminal_boundary(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
) -> None:
    """Validate the immutable V59 retirement boundary after the one-shot unseal."""

    if phase_contract.get("stage_revision") != "v059_terminal_r1":
        raise ResearchStateError("V59 terminal stage revision drift")
    if (
        state.get("authorized_phase") != "v59"
        or state.get("authorized_next_action") != "retire_family_without_tuning"
        or state.get("active_family_status") != "retired"
        or state.get("last_completed_phase")
        != "v59_state_conditioned_multi_horizon_evaluation"
        or state.get("last_completed_result")
        != "artifacts/v59_state_conditioned_multi_horizon_evaluation/result.json"
        or state.get("evidence_tier")
        != "adaptive_historical_non_target_evaluation_negative"
        or state.get("authorized_command") is not None
        or state.get("deployable_strategy") is not False
    ):
        raise ResearchStateError("V59 terminal current-state boundary drift")

    if experiment.get("status") != "retired" or experiment.get(
        "evaluation_summary"
    ) != {
        "decision": "retire_family_without_tuning",
        "mandatory_gates": 700,
        "passed_gates": 97,
        "failed_gates": 603,
        "unseal_count": 1,
        "source_outcome_reads": 1,
        "replay_source_outcome_reads": 0,
        "target_asset_loads": 0,
        "target_predictions": 0,
        "target_pnl_evaluations": 0,
        "retuning_performed": False,
    }:
        raise ResearchStateError("V59 terminal experiment summary drift")

    if any(item.get("status") != "retired" for item in state.get("families", [])):
        raise ResearchStateError("V59 terminal family registry drift")
    safety = state.get("safety")
    if not isinstance(safety, dict) or (
        safety.get("target_assets_remain_sealed") is not True
        or safety.get("terminal_retirement_decision_is_immutable") is not True
        or safety.get("completed_development_outcome_unseals") != 1
        or safety.get("maximum_development_outcome_unseals") != 1
        or safety.get("additional_source_outcome_reads_forbidden") is not True
        or safety.get("replay_source_outcome_reads") != 0
        or safety.get("retuning_or_regeneration_forbidden") is not True
    ):
        raise ResearchStateError("V59 terminal safety boundary drift")

    preceding = phase_contract.get("preceding_phase_contract")
    if preceding != {
        "path": V59_UNSEAL_PHASE_CONTRACT["path"],
        "file_sha256": V59_UNSEAL_PHASE_CONTRACT["file_sha256"],
    }:
        raise ResearchStateError("V59 terminal preceding-stage binding drift")
    preceding_path = _resolve_project_path(root, preceding["path"])
    if (
        not preceding_path.is_file()
        or _sha256_file(preceding_path) != preceding["file_sha256"]
    ):
        raise ResearchStateError("V59 terminal preceding-stage file/hash drift")

    expected_receipts = {
        "completion_receipt": (
            "completion_receipt_sha256",
            "f344d4eb24d599c97c9f8742ff245ad810c5e5a324dd93f9cdd6e1577548298f",
        ),
        "replay_receipt": (
            "replay_sha256",
            "c5a39f5c8a8d94e6f40d0c098133ccc4d46e833b44a8ed5005979e1f05753f18",
        ),
    }
    loaded_receipts: dict[str, dict[str, Any]] = {}
    for name, (self_field, canonical_hash) in expected_receipts.items():
        reference = phase_contract.get(name)
        if not isinstance(reference, dict) or reference.get("canonical_sha256") != canonical_hash:
            raise ResearchStateError(f"V59 terminal {name} binding drift")
        path = _resolve_project_path(root, reference.get("path"))
        if not path.is_file() or _sha256_file(path) != reference.get("file_sha256"):
            raise ResearchStateError(f"V59 terminal {name} file/hash drift")
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ResearchStateError(f"V59 terminal {name} is not valid JSON") from exc
        registered = receipt.pop(self_field, None)
        if registered != canonical_hash or _canonical_sha256(receipt) != canonical_hash:
            raise ResearchStateError(f"V59 terminal {name} canonical hash drift")
        loaded_receipts[name] = receipt

    completion = loaded_receipts["completion_receipt"]
    if (
        completion.get("decision") != "retire_family_without_tuning"
        or completion.get("unseal_count") != 1
        or completion.get("source_outcome_reads") != 1
        or completion.get("replay_source_outcome_reads") != 0
        or completion.get("target_assets_status") != "sealed"
    ):
        raise ResearchStateError("V59 terminal completion receipt drift")
    replay = loaded_receipts["replay_receipt"]
    if (
        replay.get("source_outcome_rows_read") != 0
        or replay.get("result_hashes_match") is not True
        or any(
            replay.get(field) != 0
            for field in (
                "new_checkpoint_loads",
                "new_inference",
                "new_linear_control_fits",
                "new_outcome_packets",
                "new_position_generation",
                "new_unseal_authorization_receipts",
            )
        )
    ):
        raise ResearchStateError("V59 terminal replay receipt drift")

    target = phase_contract.get("target_contract")
    if not isinstance(target, dict) or (
        target.get("symbols") != state.get("target_assets", {}).get("symbols")
        or target.get("status") != "sealed"
        or target.get("target_assets_loaded") != []
        or target.get("target_predictions") != 0
        or target.get("target_pnl_evaluations") != 0
    ):
        raise ResearchStateError("V59 terminal target seal drift")
    if phase_contract.get("terminal_summary") != {
        "decision": "retire_family_without_tuning",
        "mandatory_gates": 700,
        "passed_gates": 97,
        "failed_gates": 603,
        "completed_unseal_count": 1,
        "replay_source_outcome_reads": 0,
    }:
        raise ResearchStateError("V59 terminal gate summary drift")
    required_forbidden = {
        "second_outcome_unseal_or_source_outcome_reread",
        "refit_retrain_or_regenerate_predictions_positions_or_controls",
        "change_costs_metrics_bootstrap_gates_or_failure_decision",
        "access_target_assets",
    }
    if not required_forbidden.issubset(
        set(phase_contract.get("access_contract", {}).get("forbidden_capabilities", []))
    ):
        raise ResearchStateError("V59 terminal forbidden capabilities are incomplete")
    if _canonical_sha256(phase_contract) != V59_TERMINAL_PHASE_CONTRACT["canonical_sha256"]:
        raise ResearchStateError("V59 terminal phase-contract semantic hash drift")


def _validate_v60_specification_boundary(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
) -> None:
    expected_action = "execute_v60_metadata_only_decoupled_rank_state_family_specification"
    expected_command = (
        "PYTHONPATH=src python3 -m tlm decoupled-rank-state-spec "
        "--config configs/v60_decoupled_rank_state_spec.yaml"
    )
    if (
        state.get("authorized_phase") != "v60"
        or state.get("authorized_next_action") != expected_action
        or state.get("authorized_command") != expected_command
        or phase_contract.get("authorized_command") != expected_command
    ):
        raise ResearchStateError("V60 specification authorization drift")
    if (
        state.get("active_family_status") != "specification_authorized_not_started"
        or state.get("last_completed_phase")
        != "v59_state_conditioned_multi_horizon_evaluation"
        or experiment.get("phase") != "explicit_user_authorization"
        or experiment.get("status") != "registered"
    ):
        raise ResearchStateError("V60 specification state drift")
    expected_user_authorization = {
        "path": "artifacts/authorization_v60_decoupled_rank_state/blueprint.json",
        "file_sha256": "9d984e1cf1ae871cce6e32e894c1b71e58ed2e4fb5bfeb67c7c04951435fff10",
        "canonical_sha256": "4aa3efe2f01b378682b7dd230d5f7f25f14fcbb68f4aec28eb6e0b1303b4901b",
        "source_user_message": "Ok faça com a v45",
    }
    if phase_contract.get("explicit_user_authorization") != expected_user_authorization:
        raise ResearchStateError("V60 explicit user authorization drift")
    authorization_path = _resolve_project_path(root, expected_user_authorization["path"])
    if (
        not authorization_path.is_file()
        or _sha256_file(authorization_path)
        != expected_user_authorization["file_sha256"]
    ):
        raise ResearchStateError("V60 explicit user authorization file/hash drift")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    registered = authorization.pop("blueprint_sha256", None)
    if (
        registered != expected_user_authorization["canonical_sha256"]
        or _canonical_sha256(authorization) != registered
        or authorization.get("source_user_message") != "Ok faça com a v45"
        or authorization.get("target_assets_status") != "sealed"
    ):
        raise ResearchStateError("V60 explicit user authorization semantic drift")

    config = _load_mapping(
        _resolve_project_path(root, "configs/v60_decoupled_rank_state_spec.yaml")
    )["decoupled_rank_state_spec"]
    expected_inputs = set(config["inputs"].values())
    access = phase_contract["access_contract"]
    expected_bindings = {
        config["inputs"][name]: digest
        for name, digest in config["expected_input_sha256"].items()
    }
    input_contract = phase_contract.get("input_contract", {})
    if (
        set(access.get("allowed_inputs", [])) != expected_inputs
        or input_contract.get("expected_file_sha256_by_path") != expected_bindings
    ):
        raise ResearchStateError("V60 metadata input allowlist/hash binding drift")
    if any(
        path.endswith((".parquet", ".pt", ".pth", ".ckpt"))
        for path in expected_inputs
    ):
        raise ResearchStateError("V60 metadata allowlist contains data/checkpoint input")
    for path_text, expected_hash in expected_bindings.items():
        path = _resolve_project_path(root, path_text)
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise ResearchStateError(f"V60 metadata input drift: {path_text}")
    required_forbidden = {
        "parquet_or_market_panel_deserialization",
        "checkpoint_access_or_reuse",
        "model_instantiation_training_or_inference",
        "prediction_position_metric_or_pnl_generation",
        "outcome_source_reread",
        "target_asset_access",
        "v61_harness_implementation",
    }
    if not required_forbidden.issubset(
        set(access.get("forbidden_capabilities", []))
    ):
        raise ResearchStateError("V60 forbidden capability boundary drift")
    if (
        state.get("deployable_strategy") is not False
        or state.get("target_assets", {}).get("status") != "sealed"
        or phase_contract.get("target_contract", {}).get("status") != "sealed"
    ):
        raise ResearchStateError("V60 target/deployment boundary drift")


def _validate_v65_v64_r2_specification_boundary(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
) -> None:
    expected_action = (
        "execute_v65_metadata_only_v64_r2_probabilistic_state_gate_specification"
    )
    expected_command = (
        "PYTHONPATH=src python3 -m tlm v64-r2-probabilistic-state-gate-spec "
        "--config configs/v65_v64_r2_probabilistic_state_gate_spec.yaml"
    )
    if (
        state.get("authorized_phase") != "v65"
        or state.get("authorized_next_action") != expected_action
        or state.get("authorized_command") != expected_command
        or phase_contract.get("authorized_command") != expected_command
    ):
        raise ResearchStateError("V65 V64-R2 specification authorization drift")
    if (
        state.get("active_family_status") != "specification_authorized_not_started"
        or state.get("last_completed_phase")
        != "v64_adaptive_development_evaluation"
        or experiment.get("phase") != "explicit_user_authorization"
        or experiment.get("status") != "registered"
        or experiment.get("lineage_label") != "V64-R2"
    ):
        raise ResearchStateError("V65 V64-R2 specification state drift")
    expected_user_authorization = {
        "path": "artifacts/authorization_v65_v64_r2_probabilistic_state_gate/blueprint.json",
        "file_sha256": "8f48d63c538e5540b130b1cd2a3c68f41ccc79f9adb7a540a0afe456a5bc4847",
        "canonical_sha256": "43a49b45f53e2b27352bcd470275967eb67ce029174ffeb0a74767b3df449036",
        "source_user_message": (
            "Autorizo uma especificação V64-R2, sucessora direta da V64, "
            "mantendo o ranker e a arquitetura relativa congelados e alterando "
            "somente o state gate probabilístico e sua regra de abstention, sem "
            "abrir dados, checkpoints ou outcomes nesta fase, mantendo BTC, ETH "
            "e SOL selados."
        ),
    }
    if phase_contract.get("explicit_user_authorization") != expected_user_authorization:
        raise ResearchStateError("V65 explicit user authorization drift")
    authorization_path = _resolve_project_path(root, expected_user_authorization["path"])
    if (
        not authorization_path.is_file()
        or _sha256_file(authorization_path)
        != expected_user_authorization["file_sha256"]
    ):
        raise ResearchStateError("V65 explicit user authorization file/hash drift")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    registered = authorization.pop("blueprint_sha256", None)
    if (
        registered != expected_user_authorization["canonical_sha256"]
        or _canonical_sha256(authorization) != registered
        or authorization.get("source_user_message")
        != expected_user_authorization["source_user_message"]
        or authorization.get("target_assets_status") != "sealed"
    ):
        raise ResearchStateError("V65 explicit user authorization semantic drift")

    config = _load_mapping(
        _resolve_project_path(
            root, "configs/v65_v64_r2_probabilistic_state_gate_spec.yaml"
        )
    )["v64_r2_probabilistic_state_gate_spec"]
    expected_inputs = set(config["inputs"].values())
    expected_bindings = {
        config["inputs"][name]: digest
        for name, digest in config["expected_input_sha256"].items()
    }
    access = phase_contract["access_contract"]
    input_contract = phase_contract.get("input_contract", {})
    if (
        set(access.get("allowed_inputs", [])) != expected_inputs
        or input_contract.get("expected_file_sha256_by_path") != expected_bindings
    ):
        raise ResearchStateError("V65 metadata input allowlist/hash binding drift")
    if any(
        path.endswith((".parquet", ".pt", ".pth", ".ckpt"))
        for path in expected_inputs
    ):
        raise ResearchStateError("V65 metadata allowlist contains data/checkpoint input")
    for path_text, expected_hash in expected_bindings.items():
        path = _resolve_project_path(root, path_text)
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise ResearchStateError(f"V65 metadata input drift: {path_text}")
    required_forbidden = {
        "parquet_market_panel_or_source_data_deserialization",
        "checkpoint_file_access_or_deserialization",
        "prior_gate_state_reuse",
        "model_instantiation_training_or_inference",
        "prediction_position_metric_or_pnl_generation",
        "outcome_source_or_outcome_packet_read",
        "target_asset_access",
        "ranker_architecture_objective_weight_or_score_change",
        "distribution_family_or_abstention_threshold_sweep",
        "v66_harness_implementation",
    }
    if not required_forbidden.issubset(
        set(access.get("forbidden_capabilities", []))
    ):
        raise ResearchStateError("V65 forbidden capability boundary drift")
    ranker = config["ranker_contract"]
    if (
        ranker.get("status") != "frozen_exactly_from_v64"
        or ranker.get("weights", {}).get("checkpoint_deserialization_during_v65")
        is not False
        or ranker.get("weights", {}).get("gate_state_reuse") != "forbidden"
        or config.get("lineage", {}).get("all_other_components_frozen") is not True
    ):
        raise ResearchStateError("V65 frozen-ranker boundary drift")
    if (
        state.get("deployable_strategy") is not False
        or state.get("target_assets", {}).get("status") != "sealed"
        or phase_contract.get("target_contract", {}).get("status") != "sealed"
    ):
        raise ResearchStateError("V65 target/deployment boundary drift")


def _validate_v66_v64_r2_harness_boundary(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
) -> None:
    expected_action = (
        "authorize_v66_synthetic_v64_r2_probabilistic_state_gate_harness_only"
    )
    expected_command = (
        "PYTHONPATH=src python3 -m tlm v64-r2-probabilistic-state-gate-harness "
        "--config configs/v66_v64_r2_probabilistic_state_gate_harness.yaml"
    )
    if (
        state.get("authorized_phase") != "v66"
        or state.get("authorized_next_action") != expected_action
        or state.get("authorized_command") != expected_command
        or phase_contract.get("authorized_command") != expected_command
        or state.get("active_family_status")
        != "specification_frozen_harness_authorized"
        or state.get("last_completed_phase")
        != "v65_v64_r2_probabilistic_state_gate_specification"
        or experiment.get("phase") != "metadata_only_ex_ante_specification"
        or experiment.get("status") != "passed"
    ):
        raise ResearchStateError("V66 V64-R2 synthetic harness state drift")
    access = phase_contract["access_contract"]
    expected_inputs = {
        "artifacts/v65_v64_r2_probabilistic_state_gate_spec/specification.json",
        "artifacts/v65_v64_r2_probabilistic_state_gate_spec/blueprint.json",
        "artifacts/v65_v64_r2_probabilistic_state_gate_spec/audit.json",
        "artifacts/v65_v64_r2_probabilistic_state_gate_spec/result.json",
        "artifacts/v65_v64_r2_probabilistic_state_gate_spec/artifact_manifest.json",
        "artifacts/v65_v64_r2_probabilistic_state_gate_spec/completion_receipt.json",
    }
    if set(access.get("allowed_inputs", [])) != expected_inputs:
        raise ResearchStateError("V66 synthetic-only input allowlist drift")
    expected_bindings = phase_contract.get("input_contract", {}).get(
        "expected_file_sha256_by_path", {}
    )
    if set(expected_bindings) != expected_inputs:
        raise ResearchStateError("V66 synthetic-only hash binding drift")
    for path_text, expected_hash in expected_bindings.items():
        path = _resolve_project_path(root, path_text)
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise ResearchStateError(f"V66 metadata input drift: {path_text}")
    required_forbidden = {
        "real_data_parquet_or_market_panel_access",
        "v63_or_v64_checkpoint_file_access_or_reuse",
        "prior_v64_gate_state_access_or_reuse",
        "real_training_inference_or_prediction",
        "real_performance_metric_or_pnl_generation",
        "outcome_source_or_outcome_packet_read",
        "target_asset_access",
        "ranker_distribution_threshold_size_or_hyperparameter_change",
        "v67_implementation_or_data_build",
    }
    if not required_forbidden.issubset(
        set(access.get("forbidden_capabilities", []))
    ):
        raise ResearchStateError("V66 synthetic-only forbidden boundary drift")
    summary = experiment.get("v65_summary", {})
    zero_fields = {
        "parquet_deserializations",
        "checkpoint_reads",
        "model_instantiations",
        "optimizer_steps",
        "predictions",
        "performance_metrics",
        "pnl_computations",
        "outcome_source_reads",
        "target_asset_rows",
    }
    if (
        summary.get("frozen_ranker_state_receipts") != 9
        or summary.get("abstention_probability_threshold") != 0.60
        or any(summary.get(field) != 0 for field in zero_fields)
    ):
        raise ResearchStateError("V65 completion summary drift at V66 boundary")
    if (
        state.get("deployable_strategy") is not False
        or state.get("target_assets", {}).get("status") != "sealed"
        or phase_contract.get("target_contract", {}).get("status") != "sealed"
    ):
        raise ResearchStateError("V66 target/deployment boundary drift")


def _validate_v67_v64_r2_dataset_boundary(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
) -> None:
    expected_action = (
        "authorize_v67_non_target_v64_r2_probabilistic_state_gate_dataset_only"
    )
    expected_command = (
        "PYTHONPATH=src python3 -m tlm v64-r2-probabilistic-state-gate-dataset "
        "--config configs/v67_v64_r2_probabilistic_state_gate_dataset.yaml"
    )
    if (
        state.get("authorized_phase") != "v67"
        or state.get("authorized_next_action") != expected_action
        or state.get("authorized_command") != expected_command
        or phase_contract.get("authorized_command") != expected_command
        or state.get("active_family_status")
        != "synthetic_harness_passed_dataset_authorized"
        or state.get("last_completed_phase")
        != "v66_synthetic_v64_r2_probabilistic_state_gate_harness"
        or experiment.get("phase") != "synthetic_probabilistic_state_gate_harness"
        or experiment.get("status") != "passed"
    ):
        raise ResearchStateError("V67 V64-R2 dataset state drift")
    access = phase_contract["access_contract"]
    expected_inputs = set(access.get("allowed_inputs", []))
    expected_bindings = phase_contract.get("input_contract", {}).get(
        "expected_file_sha256_by_path", {}
    )
    if set(expected_bindings) != expected_inputs:
        raise ResearchStateError("V67 dataset input allowlist/hash binding drift")
    parquet_inputs = {
        path for path in expected_inputs if path.endswith(".parquet")
    }
    if parquet_inputs != {
        "data/processed/decoupled_rank_state_labels_v62.parquet",
        "data/processed/decoupled_rank_state_sequence_roles_v62.parquet",
    }:
        raise ResearchStateError("V67 dataset Parquet allowlist drift")
    for path_text, expected_hash in expected_bindings.items():
        path = _resolve_project_path(root, path_text)
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise ResearchStateError(f"V67 dataset input drift: {path_text}")
    required_forbidden = {
        "target_asset_access",
        "raw_v32_panel_or_sequence_index_access",
        "v62_consumed_development_validation_or_2025_value_access",
        "universe_fold_triplet_feature_or_label_reselection",
        "missing_row_imputation_or_repair",
        "scaler_fit",
        "model_instantiation",
        "optimizer_step_or_training",
        "v63_or_v64_checkpoint_access",
        "market_prediction",
        "performance_metric_or_pnl",
        "outcome_source_or_packet_read",
        "ranker_distribution_threshold_size_or_hyperparameter_change",
        "v68_implementation_or_training",
    }
    if not required_forbidden.issubset(
        set(access.get("forbidden_capabilities", []))
    ):
        raise ResearchStateError("V67 dataset forbidden boundary drift")
    parquet_contract = phase_contract.get("parquet_access_contract", {})
    roles = phase_contract.get("role_contract", {})
    if (
        parquet_contract.get("maximum_parquet_deserializations") != 2
        or parquet_contract.get("predicate_pushdown_required") is not True
        or parquet_contract.get("full_table_materialization_then_filtering_allowed")
        is not False
        or roles.get("gate_train")
        != {"signal_start": "2021-03-01", "signal_end": "2024-06-30"}
        or roles.get("gate_internal_validation")
        != {"signal_start": "2024-07-01", "signal_end": "2024-12-23"}
        or roles.get("consumed_v64_2025_role_created") is not False
    ):
        raise ResearchStateError("V67 chronology or Parquet boundary drift")
    summary = experiment.get("v66_summary", {})
    if (
        summary.get("audit_checks_passed") != 22
        or summary.get("audit_checks_total") != 22
        or summary.get("ranker_optimizer_present") is not False
        or summary.get("ranker_requires_grad") is not False
        or summary.get("byte_identical_replay") is not True
    ):
        raise ResearchStateError("V66 completion summary drift at V67 boundary")
    if (
        state.get("deployable_strategy") is not False
        or state.get("target_assets", {}).get("status") != "sealed"
        or phase_contract.get("target_contract", {}).get("status") != "sealed"
    ):
        raise ResearchStateError("V67 target/deployment boundary drift")


def _validate_v68_v64_r2_gate_training_boundary(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
) -> None:
    expected_action = "authorize_v68_frozen_non_target_v64_r2_gate_training_only"
    expected_command = (
        "PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src python3 -m tlm "
        "v64-r2-probabilistic-state-gate-training "
        "--config configs/v68_v64_r2_probabilistic_state_gate_training.yaml"
    )
    if (
        state.get("authorized_phase") != "v68"
        or state.get("authorized_next_action") != expected_action
        or state.get("authorized_command") != expected_command
        or phase_contract.get("authorized_command") != expected_command
        or state.get("active_family_status") != "dataset_passed_training_authorized"
        or state.get("last_completed_phase")
        != "v67_non_target_v64_r2_probabilistic_state_gate_dataset"
        or experiment.get("phase") != "non_target_probabilistic_state_gate_dataset"
        or experiment.get("status") != "passed"
    ):
        raise ResearchStateError("V68 V64-R2 gate-training state drift")
    access = phase_contract["access_contract"]
    expected_inputs = set(access.get("allowed_inputs", []))
    expected_bindings = phase_contract.get("input_contract", {}).get(
        "expected_file_sha256_by_path", {}
    )
    if set(expected_bindings) != expected_inputs:
        raise ResearchStateError("V68 training input allowlist/hash binding drift")
    parquet_inputs = {path for path in expected_inputs if path.endswith(".parquet")}
    if parquet_inputs != {
        "data/processed/selected_universe_panel_v32.parquet",
        "data/processed/v67_v64_r2_gate_labels.parquet",
        "data/processed/v67_v64_r2_gate_sequence_roles.parquet",
    }:
        raise ResearchStateError("V68 training Parquet allowlist drift")
    checkpoint_inputs = {
        path
        for path in expected_inputs
        if path.startswith(
            "data/checkpoints/v63_decoupled_rank_state_training/"
        )
        and path.endswith(".final.pt")
    }
    if len(checkpoint_inputs) != 9:
        raise ResearchStateError("V68 exact nine-checkpoint allowlist drift")
    for path_text, expected_hash in expected_bindings.items():
        path = _resolve_project_path(root, path_text)
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise ResearchStateError(f"V68 training input drift: {path_text}")
    required_forbidden = {
        "target_asset_access",
        "v64_consumed_2025_outcome_or_any_outcome_packet_access",
        "post_2024_signal_label_role_or_feature_value_access",
        "heldout_fold_asset_scaler_training_or_validation",
        "old_v63_or_v64_gate_substate_model_load_inspection_selection_or_reuse",
        "ranker_parameter_change_gradient_or_optimizer",
        "ranker_architecture_objective_or_state_identity_change",
        "gate_distribution_threshold_size_seed_fold_or_hyperparameter_change",
        "missing_row_imputation_or_repair",
        "prediction_position_policy_or_portfolio_generation",
        "performance_metric_or_pnl",
        "v69_implementation_or_evaluation",
    }
    if not required_forbidden.issubset(
        set(access.get("forbidden_capabilities", []))
    ):
        raise ResearchStateError("V68 training forbidden boundary drift")
    reuse = phase_contract.get("ranker_and_scaler_reuse_contract", {})
    model = phase_contract.get("gate_model_and_objective_contract", {})
    runtime = phase_contract.get("grid_optimizer_and_runtime_contract", {})
    operator_runtime = phase_contract.get("runtime_contract", {})
    roles = phase_contract.get("data_and_role_contract", {})
    state_hashes = reuse.get("exact_ranker_state_sha256_by_job", {})
    if (
        len(state_hashes) != 9
        or reuse.get("ranker_requires_grad") is not False
        or reuse.get("ranker_optimizer") != "none"
        or reuse.get("source_checkpoint_container_deserialization_includes_legacy_gate_tensors")
        is not True
        or reuse.get("old_gate_substate_loaded_into_model") is not False
        or reuse.get("old_gate_substate_values_inspected_or_hashed") is not False
        or reuse.get("old_gate_substate_selected_or_reused") is not False
        or reuse.get("feature_scaler_source")
        != "exact_registered_v63_fold_scaler_without_refit"
        or reuse.get("feature_scaler_includes_v68_internal_validation_feature_distribution")
        is not True
        or model.get("architecture", {}).get("expected_parameter_count") != 27522
        or model.get("degrees_of_freedom") != 5.0
        or model.get("degrees_of_freedom_trainable") is not False
        or runtime.get("folds") != [1, 2, 3]
        or runtime.get("seeds") != [42, 7, 123]
        or runtime.get("expected_jobs") != 9
        or runtime.get("device") != "mps"
        or runtime.get("mps_fallback_allowed") is not False
        or runtime.get("train_samples_per_epoch") != 8192
        or runtime.get("fixed_validation_samples") != 2048
        or runtime.get("maximum_epochs") != 30
        or runtime.get("batch_size") != 128
        or runtime.get("hyperparameter_search_allowed") is not False
        or operator_runtime.get("minimum_free_gib") != 10
        or operator_runtime.get("backup_policy", {}).get("mode")
        != "owner_waiver"
        or operator_runtime.get("maximum_active_optimizer_jobs") != 1
        or roles.get("train_role") != "gate_train"
        or roles.get("internal_validation_role")
        != "gate_internal_validation"
        or roles.get("any_2025_or_later_value_allowed") is not False
    ):
        raise ResearchStateError("V68 model/grid/role boundary drift")
    summary = experiment.get("v67_summary", {})
    if (
        summary.get("audit_checks_passed") != 15
        or summary.get("audit_checks_total") != 15
        or summary.get("authorized_parquet_deserializations") != 2
        or summary.get("loaded_2025_or_later_values") != 0
        or summary.get("byte_identical_replay") is not True
        or any(
            summary.get(field) != 0
            for field in (
                "scaler_fits",
                "model_instantiations",
                "optimizer_steps",
                "checkpoint_reads",
                "predictions",
                "performance_metrics",
                "pnl_evaluations",
                "target_asset_loads",
            )
        )
    ):
        raise ResearchStateError("V67 completion summary drift at V68 boundary")
    if (
        state.get("deployable_strategy") is not False
        or state.get("target_assets", {}).get("status") != "sealed"
        or phase_contract.get("target_contract", {}).get("status") != "sealed"
    ):
        raise ResearchStateError("V68 target/deployment boundary drift")


def _validate_v69_v64_r2_prospective_prepare_boundary(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
) -> None:
    expected_action = (
        "authorize_v69_outcome_blind_non_target_prospective_confirmation_prepare_only"
    )
    expected_command = (
        "PYTHONPATH=src python3 -m tlm v64-r2-prospective-confirmation-prepare "
        "--config configs/v69_v64_r2_prospective_confirmation_prepare.yaml"
    )
    if (
        state.get("authorized_phase") != "v69"
        or state.get("authorized_next_action") != expected_action
        or state.get("authorized_command") != expected_command
        or phase_contract.get("authorized_command") != expected_command
        or state.get("active_family_status")
        != "training_passed_prospective_prepare_authorized"
        or state.get("last_completed_phase")
        != "v68_frozen_non_target_v64_r2_probabilistic_state_gate_training"
        or experiment.get("phase")
        != "frozen_non_target_probabilistic_state_gate_training"
        or experiment.get("status") != "passed"
        or experiment.get("authorized_next_action") != expected_action
    ):
        raise ResearchStateError("V69 prospective-prepare state drift")
    access = phase_contract.get("access_contract", {})
    allowed = access.get("allowed_inputs", [])
    bindings = phase_contract.get("input_contract", {}).get(
        "expected_file_sha256_by_path", {}
    )
    if (
        not isinstance(allowed, list)
        or len(allowed) != 13
        or set(bindings) != set(allowed)
        or any(not path.endswith(".json") for path in allowed)
        or any("data/" in path or "checkpoints/" in path for path in allowed)
    ):
        raise ResearchStateError("V69 metadata-only input boundary drift")
    for path_text, expected_hash in bindings.items():
        path = _resolve_project_path(root, path_text)
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise ResearchStateError(f"V69 metadata input drift: {path_text}")
    required_forbidden = {
        "parquet_market_panel_label_role_or_raw_market_data_access",
        "checkpoint_deserialization_model_instantiation_inference_or_optimizer",
        "prediction_position_policy_action_or_portfolio_generation",
        "outcome_source_packet_metric_performance_or_pnl_access",
        "consumed_2025_or_2026_h1_reclassification_as_clean_confirmation",
        "target_asset_access",
        "v70_capture_prediction_or_evaluation_implementation",
    }
    prospective = phase_contract.get("prospective_boundary_contract", {})
    summary = experiment.get("v68_summary", {})
    if (
        not required_forbidden.issubset(
            set(access.get("forbidden_capabilities", []))
        )
        or prospective.get("first_admissible_signal_date_rule")
        != "strictly_after_v69_completion_receipt_commit"
        or prospective.get("outcomes_available_during_v69") is not False
        or prospective.get("checkpoints_deserialized_during_v69") is not False
        or prospective.get("predictions_frozen_during_v69") is not False
        or summary.get("audit_passed") is not True
        or summary.get("completed_jobs") != 9
        or summary.get("checkpoint_count") != 9
        or summary.get("ranker_optimizer_steps") != 0
        or summary.get("zero_step_replay_passed") is not True
        or summary.get("target_asset_loads") != 0
    ):
        raise ResearchStateError("V69 prospective evidence boundary drift")
    if (
        state.get("deployable_strategy") is not False
        or state.get("target_assets", {}).get("status") != "sealed"
        or phase_contract.get("target_contract", {}).get("status") != "sealed"
    ):
        raise ResearchStateError("V69 target/deployment boundary drift")


def _validate_v70_v64_r2_prospective_capture_boundary(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
) -> None:
    expected_action = (
        "authorize_v70_prospective_non_target_capture_and_prediction_freeze_only"
    )
    expected_command = (
        "PYTHONPATH=src python3 -m tlm v64-r2-prospective-capture "
        "--config configs/v70_v64_r2_prospective_capture.yaml"
    )
    if (
        state.get("authorized_phase") != "v70"
        or state.get("authorized_next_action") != expected_action
        or state.get("authorized_command") != expected_command
        or phase_contract.get("authorized_command") != expected_command
        or state.get("active_family_status")
        != "prospective_capture_prediction_freeze_authorized_not_started"
        or state.get("last_completed_phase")
        != "v69_outcome_blind_prospective_confirmation_prepare"
        or experiment.get("phase")
        != "metadata_only_ex_ante_prospective_confirmation_prepare"
        or experiment.get("status") != "passed"
        or experiment.get("authorized_next_action") != expected_action
    ):
        raise ResearchStateError("V70 prospective capture state drift")

    access = phase_contract.get("access_contract", {})
    allowed = access.get("allowed_inputs", [])
    bindings = phase_contract.get("input_contract", {}).get(
        "expected_static_file_sha256_by_path", {}
    )
    checkpoint_paths = [path for path in allowed if path.endswith(".pt")]
    amendment_paths = {
        "research/authorizations/v070_r1_ranker_scale_amendment.json",
        "research/incidents/v070_ranker_scale_allowlist_gap.json",
        "research/receipts/v070_ranker_excess_scale_receipt.json",
        "research/amendments/v070_r1_metadata_only.yaml",
    }
    if (
        not isinstance(allowed, list)
        or len(allowed) != 26
        or set(bindings) != set(allowed)
        or not amendment_paths.issubset(set(allowed))
        or len(checkpoint_paths) != 9
        or any("v68_v64_r2_probabilistic_state_gate_training" not in path for path in checkpoint_paths)
    ):
        raise ResearchStateError("V70 static input boundary drift")
    for path_text, expected_hash in bindings.items():
        path = _resolve_project_path(root, path_text)
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise ResearchStateError(f"V70 static input drift: {path_text}")

    protocol_reference = experiment.get("protocol", {})
    protocol_path = _resolve_project_path(root, protocol_reference.get("path", ""))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_hash = protocol.pop("protocol_sha256", None)
    summary = experiment.get("v69_summary", {})
    registered_protocol_hash = protocol_reference.get("canonical_sha256")
    if (
        not isinstance(registered_protocol_hash, str)
        or protocol_hash != registered_protocol_hash
        or protocol_hash != _canonical_sha256(protocol)
        or summary.get("audit_passed") is not True
        or summary.get("audit_check_count") != 24
        or summary.get("mandatory_gate_count") != 36
        or summary.get("minimum_calendar_days") != 120
        or summary.get("minimum_matured_dates_per_fold") != 90
        or summary.get("byte_identical_replay_passed") is not True
        or any(
            summary.get(name) != 0
            for name in (
                "parquet_deserializations",
                "raw_market_data_reads",
                "checkpoint_deserializations",
                "model_instantiations",
                "predictions",
                "positions",
                "performance_metrics",
                "pnl_computations",
                "outcome_source_reads",
                "target_asset_rows",
            )
        )
    ):
        raise ResearchStateError("V70 V69 protocol evidence drift")

    amendment = phase_contract.get("amendment_contract", {})
    authorization = json.loads(
        (root / "research/authorizations/v070_r1_ranker_scale_amendment.json")
        .read_text(encoding="utf-8")
    )
    incident = json.loads(
        (root / "research/incidents/v070_ranker_scale_allowlist_gap.json")
        .read_text(encoding="utf-8")
    )
    ranker_scale = json.loads(
        (root / "research/receipts/v070_ranker_excess_scale_receipt.json")
        .read_text(encoding="utf-8")
    )
    authorization_hash = authorization.pop("authorization_sha256", None)
    incident_hash = incident.pop("incident_sha256", None)
    ranker_scale_hash = ranker_scale.pop("receipt_sha256", None)
    v68_scalers = json.loads(
        (root / "artifacts/v68_v64_r2_probabilistic_state_gate_training/scaler_manifest.json")
        .read_text(encoding="utf-8")
    )
    scale_by_fold = {int(row["fold"]): row for row in ranker_scale.get("folds", [])}
    v68_by_fold = {int(row["fold"]): row for row in v68_scalers.get("folds", [])}
    scale_identities_match = set(scale_by_fold) == set(v68_by_fold) == {1, 2, 3}
    if scale_identities_match:
        scale_identities_match = all(
            scale_by_fold[fold].get("feature_scaler_state_sha256")
            == v68_by_fold[fold].get("feature_scaler_state_sha256")
            and scale_by_fold[fold].get("source_fold_scale_sha256")
            == v68_by_fold[fold].get("source_v63_fold_scale_sha256")
            and isinstance(scale_by_fold[fold].get("ranker_excess_rms"), float)
            and scale_by_fold[fold]["ranker_excess_rms"] > 0.0
            for fold in (1, 2, 3)
        )
    incident_impact = incident.get("impact", {})
    if (
        phase_contract.get("stage_revision")
        != "v070_prospective_non_target_capture_prediction_freeze_r2"
        or authorization_hash != _canonical_sha256(authorization)
        or incident_hash != _canonical_sha256(incident)
        or ranker_scale_hash != _canonical_sha256(ranker_scale)
        or authorization.get("authorization_scope", {}).get("forbidden")
        != [
            "outcome_access",
            "retuning",
            "target_asset_access",
            "architecture_change",
            "objective_change",
            "policy_change",
            "threshold_change",
            "cost_change",
        ]
        or incident.get("scientific_evidence_admitted_under_original_anchor")
        is not False
        or any(
            incident_impact.get(name) != 0
            for name in (
                "market_data_rows_read",
                "outcome_rows_read",
                "performance_metrics_computed",
                "pnl_computations",
                "positions_frozen",
                "predictions_frozen",
            )
        )
        or incident_impact.get("target_assets_loaded") != []
        or ranker_scale.get("derivation")
        != "exact_field_projection_without_fit_refit_recalculation_or_selection"
        or ranker_scale.get("training_or_refit_performed") is not False
        or ranker_scale.get("target_assets_loaded") != []
        or not scale_identities_match
        or amendment.get("scaler_fit_or_refit_performed") is not False
        or amendment.get(
            "policy_architecture_objective_threshold_cost_or_accounting_changed"
        )
        is not False
        or amendment.get("outcome_or_target_accessed") is not False
        or protocol.get("policy")
        != json.loads(
            (root / "artifacts/v65_v64_r2_probabilistic_state_gate_spec/blueprint.json")
            .read_text(encoding="utf-8")
        ).get("policy")
    ):
        raise ResearchStateError("V70-R1 metadata amendment drift")

    anchor = phase_contract.get("registration_anchor_contract", {})
    external = phase_contract.get("external_source_contract", {})
    recurring = phase_contract.get("recurring_capture_contract", {})
    maturity = phase_contract.get("maturity_contract", {})
    if (
        anchor.get("amendment_path")
        != "research/amendments/v070_r1_metadata_only.yaml"
        or anchor.get("amendment_file_sha256")
        != _sha256_file(root / "research/amendments/v070_r1_metadata_only.yaml")
        or anchor.get("resolver")
        != "first_ancestor_commit_containing_exact_amendment_file_sha256"
        or anchor.get("first_admissible_feature_close")
        != "strictly_after_registration_commit_timestamp"
        or anchor.get("pre_registration_signal_position_or_scored_outcome_allowed")
        is not False
        or external.get("authentication_required") is not False
        or external.get("symbols") != "exact_v32_non_target_fold_symbols_only"
        or external.get("interval") != "1d"
        or recurring.get("mode")
        != "append_only_one_packet_per_admissible_feature_date"
        or recurring.get("idempotent_same_date_replay_required") is not True
        or recurring.get("overwrite_or_regeneration_allowed") is not False
        or recurring.get("prediction_freeze_must_precede_h1_maturity") is not True
        or maturity.get("minimum_calendar_days") != 120
        or maturity.get("minimum_eligible_signal_dates_per_fold") != 90
        or maturity.get("minimum_fully_matured_signal_dates_per_fold") != 90
        or maturity.get("minimum_active_position_days_per_fold") != 20
        or maturity.get("maximum_calendar_days") != 365
        or maturity.get("maturity_is_timestamp_and_count_only_without_outcome_values")
        is not True
    ):
        raise ResearchStateError("V70 chronology or maturity boundary drift")

    required_forbidden = {
        "target_asset_access",
        "consumed_2025_or_pre_registration_2026_signal_or_outcome_admission",
        "training_optimizer_scaler_refit_or_checkpoint_mutation",
        "checkpoint_seed_fold_context_or_date_selection",
        "frozen_prediction_or_position_regeneration_or_rewrite",
        "outcome_column_projection_packet_unseal_or_source_role_read",
        "predictive_or_financial_metric_pnl_equity_drawdown_bootstrap_or_gate_computation",
        "interim_performance_or_mark_to_market_access",
        "v71_outcome_prepare_or_any_evaluation_implementation",
        "paper_shadow_live_or_real_money_trading",
    }
    artifact = phase_contract.get("artifact_contract", {})
    if (
        not required_forbidden.issubset(
            set(access.get("forbidden_capabilities", []))
        )
        or phase_contract.get("pass_action")
        != "authorize_v71_outcome_blind_prospective_one_shot_prepare_only"
        or artifact.get("no_outcome_packet_during_v70") is not True
        or artifact.get("no_metric_or_pnl_artifact_during_v70") is not True
        or artifact.get("completed_capture_receipt_required_before_v71") is not True
        or state.get("deployable_strategy") is not False
        or state.get("target_assets", {}).get("status") != "sealed"
        or phase_contract.get("target_contract", {}).get("status") != "sealed"
        or state.get("safety", {}).get("v70_outcome_access_allowed") is not False
        or state.get("safety", {}).get("v70_performance_or_pnl_allowed") is not False
        or state.get("safety", {}).get("v70_shadow_or_order_execution_allowed")
        is not False
        or state.get("safety", {}).get("v70_r1_metadata_amendment_authorized")
        is not True
        or state.get("safety", {}).get(
            "v70_r1_original_anchor_invalidated_before_any_prediction"
        )
        is not True
        or state.get("safety", {}).get(
            "v70_r1_ranker_excess_scale_receipt_registered"
        )
        is not True
        or state.get("safety", {}).get(
            "v70_r1_new_anchor_requires_exact_amendment_commit"
        )
        is not True
        or state.get("safety", {}).get("v70_r1_outcome_or_target_access_count")
        != 0
        or state.get("safety", {}).get(
            "v70_r1_policy_architecture_objective_threshold_cost_or_accounting_changed"
        )
        is not False
    ):
        raise ResearchStateError("V70 outcome, target, or deployment boundary drift")


def _validate_v71_posthoc_retrospective_prepare_boundary(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
) -> None:
    """Validate the owner-authorized, outcome-blind V71 post-hoc prepare boundary."""

    action = "authorize_v71_posthoc_consumed_2025_diagnostic_prepare_only"
    command = (
        "PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src python3 -m tlm "
        "v64-r2-retrospective-diagnostic-prepare "
        "--config configs/v71_v64_r2_retrospective_diagnostic.yaml"
    )
    if (
        state.get("authorized_phase") != "v71"
        or state.get("authorized_next_action") != action
        or state.get("authorized_command") != command
        or state.get("active_family_status")
        != "posthoc_consumed_2025_diagnostic_prepare_authorized_not_started"
        or state.get("evidence_tier")
        != "posthoc_consumed_2025_diagnostic_prepare_only_not_confirmation"
        or experiment.get("authorized_next_action") != action
        or experiment.get("status") != "authorized_not_started"
        or experiment.get("evidence_tier")
        != "posthoc_consumed_2025_diagnostic_only_not_confirmation"
        or phase_contract.get("status")
        != "outcome_blind_posthoc_prepare_authorized_not_started"
        or phase_contract.get("evidence_tier")
        != "posthoc_consumed_2025_diagnostic_only_not_confirmation"
    ):
        raise ResearchStateError("V71 post-hoc prepare state drift")

    authorization_ref = phase_contract.get("authorization_receipt", {})
    expected_authorization = {
        "path": "research/authorizations/v071_posthoc_retrospective_diagnostic.json",
        "file_sha256": "0deea9b57c92967e56116eeb1d5c43d6eb4cefb8afbfcfd5632ecd8061032936",
        "canonical_sha256": "adf1276b64a7a89d0841d04a7a796dceadf57a67cf7fd2520230b0bf38523017",
    }
    if authorization_ref != expected_authorization:
        raise ResearchStateError("V71 owner authorization reference drift")
    authorization_path = _resolve_project_path(root, expected_authorization["path"])
    if _sha256_file(authorization_path) != expected_authorization["file_sha256"]:
        raise ResearchStateError("V71 owner authorization file drift")
    authorization = _load_mapping(authorization_path)
    registered_authorization_hash = authorization.pop("authorization_sha256", None)
    if (
        registered_authorization_hash
        != expected_authorization["canonical_sha256"]
        or _canonical_sha256(authorization) != registered_authorization_hash
        or authorization.get("phase") != "v71"
        or authorization.get("authorized_by") != "repository_owner"
        or authorization.get("explicit_unseal_authorization_present") is not False
        or authorization.get("scientific_label")
        != "posthoc_consumed_2025_diagnostic_only_not_confirmation"
    ):
        raise ResearchStateError("V71 owner authorization semantic drift")

    access = phase_contract.get("access_contract", {})
    allowed = access.get("allowed_inputs", [])
    expected_hashes = phase_contract.get("input_contract", {}).get(
        "expected_static_file_sha256_by_path", {}
    )
    if set(allowed) != set(expected_hashes) or len(allowed) != 25:
        raise ResearchStateError("V71 static input allowlist drift")
    for relative, expected_hash in expected_hashes.items():
        path = _resolve_project_path(root, relative)
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise ResearchStateError(f"V71 static input drift: {relative}")

    incident_ref = phase_contract.get("access_incident", {})
    expected_incident = {
        "path": "research/incidents/v071_prepare_schema_probe_projection_gap.json",
        "file_sha256": "fa297cf700a582bb22f8d2ee3a29c50dff91935980b3cd1a5341df8c8ae3661b",
        "canonical_sha256": "e108271ee8514b10c739ba9afb09a62a268a28adbeb4a5ae00d7670777596b92",
    }
    if incident_ref != expected_incident:
        raise ResearchStateError("V71 access incident reference drift")
    incident = _load_mapping(_resolve_project_path(root, expected_incident["path"]))
    registered_incident_hash = incident.pop("incident_sha256", None)
    if (
        registered_incident_hash != expected_incident["canonical_sha256"]
        or _canonical_sha256(incident) != registered_incident_hash
        or incident.get("assessment", {}).get(
            "2025_evaluation_outcome_contamination"
        )
        is not False
        or incident.get("observed", {}).get("evaluation_window_rows_read") != 0
        or incident.get("observed", {}).get("sealed_v64_outcome_packet_reads")
        != 0
        or incident.get("observed", {}).get("rows_displayed") != 2
        or incident.get("remediation", {}).get("new_prepare_anchor_required")
        is not True
    ):
        raise ResearchStateError("V71 access incident semantic drift")

    anchor = phase_contract.get("prepare_registration_anchor_contract", {})
    projection = phase_contract.get("projection_contract", {})
    expected_feature_columns = [
        "date",
        "symbol",
        "log_open_to_open_return",
        "log_close_to_close_return",
        "log_high_low_range",
        "log_close_open_return",
        "log1p_quote_volume_change",
        "log1p_trade_count_change",
        "rolling_realized_volatility_7d",
        "rolling_realized_volatility_30d",
    ]
    if (
        anchor.get("incident_path") != expected_incident["path"]
        or anchor.get("incident_file_sha256") != expected_incident["file_sha256"]
        or anchor.get("resolver")
        != "first_ancestor_commit_containing_exact_incident_file_sha256"
        or anchor.get("no_prepare_output_before_anchor") is not True
        or projection.get("feature_panel_columns") != expected_feature_columns
        or projection.get("registered_pre_anchor_training_era_target_rows_displayed")
        != 2
        or projection.get(
            "registered_pre_anchor_evaluation_window_outcome_rows_read"
        )
        != 0
        or projection.get("post_anchor_forbidden_column_projection_allowed")
        is not False
    ):
        raise ResearchStateError("V71 projection remediation drift")

    diagnostic = phase_contract.get("diagnostic_contract", {})
    if (
        diagnostic.get("signal_start") != "2025-01-01"
        or diagnostic.get("signal_end") != "2025-12-23"
        or diagnostic.get("expected_signal_dates") != 357
        or diagnostic.get("lookback_days") != 256
        or diagnostic.get("reporting_cost_bps") != [10, 20, 30]
        or diagnostic.get("candidate_policy")
        != "exact_v65_probabilistic_abstention_policy"
        or diagnostic.get("control_policy") != "exact_frozen_v64_positions"
        or diagnostic.get("interpretation", {}).get(
            "pass_or_fail_changes_family_status"
        )
        is not False
        or diagnostic.get("interpretation", {}).get(
            "selection_or_retuning_after_result"
        )
        != "forbidden"
    ):
        raise ResearchStateError("V71 frozen diagnostic contract drift")

    sealed = phase_contract.get("sealed_outcome_contract", {})
    if (
        sealed.get("packet_path")
        != "artifacts/v64_decoupled_rank_state_evaluation/outcome_packet.parquet"
        or sealed.get("packet_sha256")
        != "54dd42d528aa9d77aae277a954c4679adea9dfcda8ba12a29b5d4b5ef1a2d252"
        or sealed.get("may_be_opened_during_v71_prepare") is not False
        or sealed.get("exact_hash_bound_user_authorization_after_prepare_required")
        is not True
        or sealed.get("maximum_new_diagnostic_unseal_count") != 1
        or sealed.get("source_outcome_reread_allowed") is not False
    ):
        raise ResearchStateError("V71 sealed outcome boundary drift")

    artifact = phase_contract.get("artifact_contract", {})
    target = phase_contract.get("target_contract", {})
    safety = state.get("safety", {})
    if (
        phase_contract.get("pass_action")
        != "authorize_v72_exact_hash_bound_posthoc_outcome_unseal_only"
        or artifact.get("outcome_packet_during_prepare_allowed") is not False
        or artifact.get("metric_or_pnl_artifact_during_prepare_allowed") is not False
        or target.get("status") != "sealed"
        or target.get("target_assets_loaded") != []
        or state.get("target_assets", {}).get("status") != "sealed"
        or state.get("deployable_strategy") is not False
        or safety.get("v70_prospective_capture_paused_by_owner") is not True
        or safety.get("v70_immutable_artifacts_preserved") is not True
        or safety.get("v71_posthoc_consumed_2025_diagnostic_prepare_authorized")
        is not True
        or safety.get("v71_outcome_access_allowed_during_prepare") is not False
        or safety.get("v71_performance_or_pnl_allowed_during_prepare") is not False
        or safety.get("v71_exact_hash_bound_unseal_authorization_present")
        is not False
        or safety.get("v71_target_assets_remain_sealed") is not True
        or safety.get("v71_retuning_or_retraining_allowed") is not False
        or safety.get("v71_clean_holdout_or_prospective_claim_allowed") is not False
        or safety.get("v71_schema_probe_projection_incident_registered") is not True
        or safety.get("v71_pre_anchor_training_era_target_rows_displayed") != 2
        or safety.get("v71_pre_anchor_evaluation_window_outcome_rows_read") != 0
        or safety.get("v71_post_anchor_feature_projection_only_required") is not True
    ):
        raise ResearchStateError("V71 outcome, target, or interpretation boundary drift")


def _validate_v72_posthoc_outcome_unseal_boundary(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
) -> None:
    """Validate the exact hash-bound V72 post-hoc packet-unseal boundary."""

    action = (
        "execute_v72_exactly_one_hash_bound_posthoc_outcome_unseal_and_complete_diagnostic"
    )
    command = (
        "PYTHONPATH=src python3 -m tlm v64-r2-retrospective-diagnostic-unseal "
        "--config configs/v72_v64_r2_retrospective_evaluation.yaml"
    )
    if (
        state.get("authorized_phase") != "v72"
        or state.get("authorized_next_action") != action
        or state.get("authorized_command") != command
        or state.get("active_family_status")
        != "posthoc_consumed_2025_diagnostic_exact_unseal_authorized"
        or state.get("evidence_tier")
        != "posthoc_consumed_2025_diagnostic_only_not_confirmation"
        or experiment.get("authorized_next_action") != action
        or experiment.get("status") != "authorized_not_started"
        or experiment.get("evidence_tier")
        != "posthoc_consumed_2025_diagnostic_only_not_confirmation"
        or phase_contract.get("status")
        != "frozen_after_hash_bound_user_authorization_before_exactly_one_packet_unseal"
        or phase_contract.get("evidence_tier")
        != "posthoc_consumed_2025_diagnostic_only_not_confirmation"
    ):
        raise ResearchStateError("V72 post-hoc unseal state drift")

    authorization_ref = phase_contract.get("explicit_user_authorization", {})
    expected_authorization = {
        "path": "research/authorizations/v072_posthoc_outcome_unseal.json",
        "file_sha256": "938d6aeb1b947bb028302887215d15773a51e891ae4bb2d0f1d3067e4ce0866c",
        "canonical_sha256": "817fed1b19ca1ab5b040e964e24676ac6c768ade0038d09c73b67f1f86fc8acb",
    }
    if authorization_ref != expected_authorization:
        raise ResearchStateError("V72 exact authorization reference drift")
    authorization_path = _resolve_project_path(root, expected_authorization["path"])
    if _sha256_file(authorization_path) != expected_authorization["file_sha256"]:
        raise ResearchStateError("V72 exact authorization file drift")
    authorization = _load_mapping(authorization_path)
    registered_hash = authorization.pop("authorization_sha256", None)
    if (
        registered_hash != expected_authorization["canonical_sha256"]
        or _canonical_sha256(authorization) != registered_hash
        or authorization.get("phase") != "v72"
        or authorization.get("authorized_by") != "repository_owner"
        or authorization.get("authorized_action") != action
        or authorization.get("evaluation_spec_sha256")
        != "681258600514d0b66a694c0abb29ccee88cead68a20aa3e67e58152445b01609"
        or authorization.get("prepare_receipt_sha256")
        != "d6d29ff80b6a833f7543673e3cf245d03be926fa024cb1b0e48e83b7357662ce"
        or authorization.get("registered_sha256")
        != "b1825271baffdff55669e84ac2c16e2f751a6edcfe2c49b3c24d38ffe02c67e9"
        or authorization.get("maximum_unseal_count") != 1
        or authorization.get("source_outcome_reread_allowed") is not False
        or authorization.get("target_assets_status") != "sealed"
    ):
        raise ResearchStateError("V72 exact authorization semantic drift")

    prepare = phase_contract.get("prepare_packet", {})
    expected_prepare = {
        "evaluation_spec": "681258600514d0b66a694c0abb29ccee88cead68a20aa3e67e58152445b01609",
        "prepare_receipt": "d6d29ff80b6a833f7543673e3cf245d03be926fa024cb1b0e48e83b7357662ce",
        "one_shot_packet": "2d77ac0e64916c9cb6cf3427db1dd6b1a4d5af829c13f861aa7d05fd5648cbc7",
        "asset_predictions": "488e695f529efaa50ca605ef4465379a0b504602cb5915d1530fec78a50e44af",
        "candidate_positions": "686c27a7054fd2cd4c30cf967966638eab60acc49cb389ee968806285f48103d",
        "equal_weight_positions": "37623d5092cd91210bb8d3054423ed257e81c776f159df86bdc0af544a236955",
        "v64_control_positions": "7722c68e522fba1a3bb708b803d08230677920998255fc3c37d697c1096cd88f",
        "behavior_gates": "bf910eeb590b9f62765b0a290208587607c62156407b640908bb4a16e066e868",
        "data_access_receipt": "1717c8b3bb81ec88875276a09685ec4f299402bb5440df5f45db82ec9151a6e7",
    }
    if (
        prepare.get("output_dir")
        != "artifacts/v71_v64_r2_posthoc_retrospective_diagnostic"
        or prepare.get("registered_sha256")
        != "b1825271baffdff55669e84ac2c16e2f751a6edcfe2c49b3c24d38ffe02c67e9"
    ):
        raise ResearchStateError("V72 prepare binding drift")
    for name, expected_hash in expected_prepare.items():
        reference = prepare.get(name, {})
        path = _resolve_project_path(root, reference.get("path"))
        if (
            reference.get("file_sha256") != expected_hash
            or not path.is_file()
            or _sha256_file(path) != expected_hash
        ):
            raise ResearchStateError(f"V72 prepare artifact drift: {name}")

    outcome = phase_contract.get("outcome_access_contract", {})
    evaluation = phase_contract.get("evaluation_contract", {})
    one_shot = phase_contract.get("one_shot_contract", {})
    if (
        outcome.get("source_packet")
        != "artifacts/v64_decoupled_rank_state_evaluation/outcome_packet.parquet"
        or outcome.get("source_packet_sha256")
        != "54dd42d528aa9d77aae277a954c4679adea9dfcda8ba12a29b5d4b5ef1a2d252"
        or outcome.get("source_receipt_sha256")
        != "44f26a309991f2fa6fbb65990f896fbca23b93957202da02e3881879d6eea163"
        or outcome.get("exact_key_count") != 9794
        or outcome.get("signal_start") != "2025-01-01"
        or outcome.get("signal_end") != "2025-12-23"
        or outcome.get("maximum_sealed_packet_deserializations") != 1
        or outcome.get("maximum_underlying_source_outcome_reads") != 0
        or outcome.get("authorization_receipt_before_packet_access") is not True
        or evaluation.get("portfolios")
        != ["candidate", "v64_control", "equal_weight", "cash"]
        or evaluation.get("costs_bps") != [10, 20, 30]
        or evaluation.get("mandatory_candidate_gate_count") != 24
        or evaluation.get("family_status_change_allowed") is not False
        or evaluation.get("aggregate_rescue_allowed") is not False
        or one_shot.get("explicit_user_authorization_present") is not True
        or one_shot.get("maximum_unseal_count") != 1
        or one_shot.get("underlying_source_outcome_reads") != 0
        or one_shot.get("prediction_or_position_regeneration") is not False
    ):
        raise ResearchStateError("V72 outcome or evaluation contract drift")

    target = phase_contract.get("target_contract", {})
    safety = state.get("safety", {})
    repair = phase_contract.get("contract_repair", {})
    if (
        target.get("status") != "sealed"
        or target.get("target_assets_loaded") != []
        or state.get("target_assets", {}).get("status") != "sealed"
        or state.get("deployable_strategy") is not False
        or safety.get("v71_exact_hash_bound_unseal_authorization_present") is not True
        or safety.get("v72_exact_user_authorization_sha256")
        != expected_authorization["canonical_sha256"]
        or safety.get("v72_maximum_new_diagnostic_unseal_count") != 1
        or safety.get("v72_completed_diagnostic_unseal_count") != 0
        or safety.get("v72_source_packet_deserialization_count") != 0
        or safety.get("v72_underlying_source_outcome_read_count") != 0
        or safety.get("v72_target_assets_remain_sealed") is not True
        or safety.get("v72_retuning_regeneration_or_policy_cost_change_allowed")
        is not False
        or safety.get("v72_family_status_change_from_posthoc_result_allowed")
        is not False
        or safety.get("v72_clean_holdout_prospective_or_deployable_claim_allowed")
        is not False
    ):
        raise ResearchStateError("V72 target or safety boundary drift")


def _validate_v73_v72_diagnostic_record_boundary(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
) -> None:
    """Validate the metadata-only boundary after the completed V72 unseal."""

    action = "record_v73_v72_posthoc_diagnostic_result_metadata_only"
    command = (
        "PYTHONPATH=src python3 -m tlm v72-diagnostic-record "
        "--config configs/v73_v72_diagnostic_record.yaml"
    )
    if (
        state.get("authorized_phase") != "v73"
        or state.get("authorized_next_action") != action
        or state.get("authorized_command") != command
        or state.get("active_family_status")
        != "posthoc_diagnostic_failed_metadata_record_authorized"
        or state.get("last_completed_phase")
        != "v72_posthoc_consumed_2025_diagnostic_evaluation"
        or state.get("evidence_tier")
        != "posthoc_consumed_2025_diagnostic_only_not_confirmation"
        or experiment.get("authorized_next_action") != action
        or experiment.get("status") != "authorized_not_started"
        or experiment.get("phase")
        != "metadata_only_posthoc_diagnostic_result_record"
        or phase_contract.get("status")
        != "metadata_only_result_record_authorized_not_started"
        or phase_contract.get("evidence_tier")
        != "posthoc_consumed_2025_diagnostic_only_not_confirmation"
    ):
        raise ResearchStateError("V73 metadata-only record state drift")

    expected_inputs = {
        "artifacts/v72_v64_r2_posthoc_retrospective_evaluation/result.json": (
            "3a0ff56c77445ac05bf37fc8c020b392792007afd861475123d6ebcf95417c8f"
        ),
        "artifacts/v72_v64_r2_posthoc_retrospective_evaluation/audit.json": (
            "b55010b7f9294d49d95342835d6e1976f10c428f29532fe68ebf7666ab0f2d8b"
        ),
        "artifacts/v72_v64_r2_posthoc_retrospective_evaluation/completion_receipt.json": (
            "025edb0ec18d9f17efce6e0c5b627aef1f3de286ffa2f3aa2fa8cd955c9a3395"
        ),
        "artifacts/v72_v64_r2_posthoc_retrospective_evaluation/replay.json": (
            "b102d3968daf85e92ece8d05294d298e25994be29706141d363e5ba8702c4f20"
        ),
    }
    access = phase_contract.get("access_contract", {})
    input_contract = phase_contract.get("input_contract", {})
    if (
        set(access.get("allowed_inputs", [])) != set(expected_inputs)
        or input_contract.get("expected_static_file_sha256_by_path")
        != expected_inputs
        or input_contract.get("allowed_extensions") != [".json"]
        or input_contract.get("expected_json_metadata_reads") != 4
        or input_contract.get("maximum_parquet_deserializations") != 0
        or input_contract.get("maximum_outcome_packet_reads") != 0
    ):
        raise ResearchStateError("V73 metadata input contract drift")
    for relative, expected_hash in expected_inputs.items():
        path = _resolve_project_path(root, relative)
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise ResearchStateError(f"V73 metadata input drift: {relative}")

    result = _load_mapping(
        _resolve_project_path(
            root,
            "artifacts/v72_v64_r2_posthoc_retrospective_evaluation/result.json",
        )
    )
    audit = _load_mapping(
        _resolve_project_path(
            root,
            "artifacts/v72_v64_r2_posthoc_retrospective_evaluation/audit.json",
        )
    )
    completion = _load_mapping(
        _resolve_project_path(
            root,
            "artifacts/v72_v64_r2_posthoc_retrospective_evaluation/completion_receipt.json",
        )
    )
    replay = _load_mapping(
        _resolve_project_path(
            root,
            "artifacts/v72_v64_r2_posthoc_retrospective_evaluation/replay.json",
        )
    )
    if (
        result.get("decision")
        != "authorize_v73_record_posthoc_diagnostic_result_only"
        or result.get("diagnostic_outcome") != "fail"
        or result.get("one_shot_decision") != "retire"
        or result.get("mandatory_gate_count") != 24
        or result.get("passed_gate_count") != 13
        or result.get("failed_gate_count") != 11
        or result.get("family_status_changed") is not False
        or result.get("sealed_packet_deserializations") != 1
        or result.get("underlying_source_outcome_reads") != 0
        or result.get("retuning_performed") is not False
        or result.get("prediction_or_position_regeneration") is not False
        or result.get("target_assets_loaded") != []
        or audit.get("passed") is not True
        or audit.get("scientific_gates_all_passed") is not False
        or completion.get("unseal_count") != 1
        or completion.get("source_outcome_reads") != 0
        or completion.get("family_status_changed") is not False
        or completion.get("target_assets_status") != "sealed"
        or replay.get("result_hashes_match") is not True
        or replay.get("sealed_source_packet_deserializations") != 0
        or replay.get("source_outcome_rows_read") != 0
        or replay.get("new_inference") != 0
        or replay.get("new_position_generation") != 0
        or replay.get("new_checkpoint_loads") != 0
        or replay.get("target_assets_loaded") != []
    ):
        raise ResearchStateError("V73 frozen V72 result semantic drift")

    record = phase_contract.get("record_contract", {})
    target = phase_contract.get("target_contract", {})
    safety = state.get("safety", {})
    if (
        phase_contract.get("pass_action")
        != "authorize_v74_persistent_duration_family_specification_only"
        or record.get("diagnostic_outcome") != "fail"
        or record.get("mandatory_gate_count") != 24
        or record.get("passed_gate_count") != 13
        or record.get("failed_gate_count") != 11
        or record.get("family_status_changed") is not False
        or record.get("scientific_metrics_recomputed") is not False
        or record.get("policy_or_position_regeneration") is not False
        or record.get("deployable") is not False
        or target.get("status") != "sealed"
        or target.get("target_assets_loaded") != []
        or state.get("target_assets", {}).get("status") != "sealed"
        or state.get("deployable_strategy") is not False
        or safety.get("v72_completed_diagnostic_unseal_count") != 1
        or safety.get("v72_source_packet_deserialization_count") != 1
        or safety.get("v72_underlying_source_outcome_read_count") != 0
        or safety.get("v72_diagnostic_outcome") != "fail"
        or safety.get("v72_passed_gate_count") != 13
        or safety.get("v72_failed_gate_count") != 11
        or safety.get("v72_family_status_changed") is not False
        or safety.get("v72_replay_result_hashes_match") is not True
        or safety.get("v73_metadata_only_recording_phase") is not True
        or safety.get("v73_parquet_or_outcome_packet_access_allowed") is not False
        or safety.get("v73_model_checkpoint_training_or_inference_allowed") is not False
        or safety.get("v73_target_assets_remain_sealed") is not True
    ):
        raise ResearchStateError("V73 target, result, or access boundary drift")


def _validate_v74_persistent_duration_specification_boundary(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
) -> None:
    """Validate the V74 registration without executing its specification."""

    action = "execute_v74_metadata_only_persistent_duration_family_specification"
    command = (
        "PYTHONPATH=src python3 -m tlm persistent-duration-spec "
        "--config configs/v74_persistent_duration_spec.yaml"
    )
    if (
        state.get("authorized_phase") != "v74"
        or state.get("authorized_next_action") != action
        or state.get("authorized_command") != command
        or state.get("active_family_status")
        != "specification_authorized_not_started"
        or state.get("last_completed_phase")
        != "v73_v72_diagnostic_metadata_record"
        or state.get("evidence_tier") != "metadata_only_ex_ante_design"
        or experiment.get("authorized_next_action") != action
        or experiment.get("status") != "authorized_not_started"
        or experiment.get("phase") != "metadata_only_ex_ante_specification"
        or phase_contract.get("status")
        != "metadata_only_specification_authorized_not_started"
    ):
        raise ResearchStateError("V74 specification authorization state drift")

    expected_inputs = {
        "artifacts/v73_v72_posthoc_diagnostic_record/result.json": (
            "a760aa3e29e1158901cd34b7c9774047f61419bc7df8ae8d5cb4b028ee5e659c"
        ),
        "artifacts/v73_v72_posthoc_diagnostic_record/audit.json": (
            "9a9c2e16251c68d0a1ea591be8b4e2f7d252bdcd60b622f5bdf303e4c40b406b"
        ),
        "artifacts/v73_v72_posthoc_diagnostic_record/diagnostic_record.json": (
            "515770ee477e961a256f20091ec6e2e6cde76b1bf7f4d3531ad04c22a25659cb"
        ),
        "artifacts/v73_v72_posthoc_diagnostic_record/artifact_manifest.json": (
            "b5538505bb991bf8d97256f05cad02271a5063c63f1d00dc0b241cebd6671ba2"
        ),
        "research/candidates/v74_persistent_multi_horizon_duration_model.md": (
            "20a3d79099fedaa116daf9053221ee45d89ab64a8b9e60cee743f00305079610"
        ),
        "src/tlm/persistent_multi_horizon_duration_model.py": (
            "3e1e39915e418c5497a7f28325d75642c8d820fc2c1b0593ed4fb0dc09c0b901"
        ),
        "tests/test_persistent_multi_horizon_duration_model.py": (
            "847d4ee3e00f3951c5b4854e568516a4c8b78488bec538c3190b9bd94fbae348"
        ),
    }
    access = phase_contract.get("access_contract", {})
    inputs = phase_contract.get("input_contract", {})
    if (
        set(access.get("allowed_inputs", [])) != set(expected_inputs)
        or inputs.get("expected_static_file_sha256_by_path") != expected_inputs
        or inputs.get("expected_json_metadata_reads") != 4
        or inputs.get("expected_source_hash_checks") != 3
        or inputs.get("maximum_parquet_deserializations") != 0
        or inputs.get("maximum_outcome_packet_reads") != 0
        or inputs.get("maximum_checkpoint_reads") != 0
        or inputs.get("maximum_model_instantiations") != 0
    ):
        raise ResearchStateError("V74 metadata/source input contract drift")
    for relative, expected_hash in expected_inputs.items():
        path = _resolve_project_path(root, relative)
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise ResearchStateError(f"V74 metadata/source input drift: {relative}")

    config = _load_mapping(
        _resolve_project_path(root, "configs/v74_persistent_duration_spec.yaml")
    )["persistent_duration_spec"]
    config_bindings = {
        config["inputs"][name]: digest
        for name, digest in config["expected_input_sha256"].items()
    }
    if config_bindings != expected_inputs:
        raise ResearchStateError("V74 config input hash bindings drift")

    result = _load_mapping(
        _resolve_project_path(
            root, "artifacts/v73_v72_posthoc_diagnostic_record/result.json"
        )
    )
    audit = _load_mapping(
        _resolve_project_path(
            root, "artifacts/v73_v72_posthoc_diagnostic_record/audit.json"
        )
    )
    record = _load_mapping(
        _resolve_project_path(
            root,
            "artifacts/v73_v72_posthoc_diagnostic_record/diagnostic_record.json",
        )
    )
    if (
        result.get("decision")
        != "authorize_v74_persistent_duration_family_specification_only"
        or result.get("family_status_changed") is not False
        or result.get("outcomes_opened") != 0
        or result.get("models_or_checkpoints_loaded") != 0
        or result.get("target_assets_loaded") != []
        or audit.get("passed") is not True
        or record.get("diagnostic_outcome") != "fail"
        or record.get("passed_gate_count") != 13
        or record.get("mandatory_gate_count") != 24
        or record.get("target_assets_status") != "sealed"
    ):
        raise ResearchStateError("V74 V73 authorization receipt semantic drift")

    frozen = phase_contract.get("frozen_family_contract", {})
    architecture = config.get("architecture", {})
    training = config.get("training_contract", {})
    gates = config.get("financial_evaluation_contract", {}).get(
        "mandatory_gates", {}
    )
    if (
        frozen.get("architecture_variant_count") != 1
        or frozen.get("expected_total_parameters") != 1083155
        or frozen.get("parameter_ceiling") != 1100000
        or frozen.get("return_horizons_days") != [1, 3, 7]
        or frozen.get("maximum_label_maturity_days") != 8
        or frozen.get("future_training_jobs") != 9
        or frozen.get("size_or_hyperparameter_sweep_allowed") is not False
        or architecture.get("expected_parameter_count") != 1083155
        or architecture.get("input_shape") != [None, 256, 3, 9]
        or architecture.get("asset_slot_embedding") is not False
        or training.get("expected_job_count") != 9
        or training.get("device") != "mps"
        or training.get("mps_fallback_allowed") is not False
        or gates.get("aggregate_net_total_return_positive_at_cost_bps")
        != [10, 20, 30]
        or gates.get("aggregate_rescue_for_failed_fold") is not False
    ):
        raise ResearchStateError("V74 frozen scientific contract drift")

    required_forbidden = {
        "parquet_market_panel_or_source_data_deserialization",
        "outcome_packet_or_outcome_source_read",
        "checkpoint_scaler_or_prior_weight_access_or_reuse",
        "model_instantiation_training_optimizer_or_inference",
        "prediction_position_metric_pnl_bootstrap_or_gate_computation",
        "target_asset_access",
        "architecture_size_objective_policy_threshold_or_hyperparameter_sweep",
        "v75_harness_implementation_or_execution",
    }
    target = phase_contract.get("target_contract", {})
    safety = state.get("safety", {})
    if (
        not required_forbidden.issubset(
            set(access.get("forbidden_capabilities", []))
        )
        or target.get("status") != "sealed"
        or target.get("target_assets_loaded") != []
        or state.get("target_assets", {}).get("status") != "sealed"
        or state.get("deployable_strategy") is not False
        or safety.get("metadata_only_specification_phase") is not True
        or safety.get("v74_family_is_new") is not True
        or safety.get("v74_expected_total_parameters") != 1083155
        or safety.get("v74_size_sweep_allowed") is not False
        or safety.get("v74_data_checkpoint_model_or_outcome_access_allowed")
        is not False
        or safety.get("v74_target_assets_remain_sealed") is not True
    ):
        raise ResearchStateError("V74 target, access, or safety boundary drift")


def _validate_v75_persistent_duration_harness_boundary(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
) -> None:
    """Validate the synthetic-only V75 handoff from the passed V74 packet."""

    action = "execute_v75_synthetic_persistent_duration_harness"
    command = (
        "PYTHONPATH=src python3 -m tlm persistent-duration-harness "
        "--config configs/v75_persistent_duration_harness.yaml"
    )
    if (
        state.get("authorized_phase") != "v75"
        or state.get("authorized_next_action") != action
        or state.get("authorized_command") != command
        or state.get("active_family_status")
        != "synthetic_harness_authorized_not_started"
        or state.get("last_completed_phase")
        != "v74_persistent_duration_metadata_specification"
        or state.get("evidence_tier")
        != "synthetic_scientific_and_accounting_validation_only"
        or experiment.get("authorized_next_action") != action
        or experiment.get("status") != "authorized_not_started"
        or experiment.get("phase") != "synthetic_harness"
        or phase_contract.get("status")
        != "synthetic_harness_authorized_not_started"
    ):
        raise ResearchStateError("V75 synthetic harness authorization state drift")

    expected_inputs = {
        "artifacts/v74_persistent_duration_spec/specification.json": (
            "58c7127d9d8b3606ce78d277789a6ea85839ce12186cd0277a90ec7f18d9e263"
        ),
        "artifacts/v74_persistent_duration_spec/blueprint.json": (
            "790b14fd3feacbc62f416e63a019f192ea09d0fd3756c9d757286968ece3632d"
        ),
        "artifacts/v74_persistent_duration_spec/audit.json": (
            "2615f0c611dab3b8d0352cb780dce31991eec5663d95bbeeeb28062077e02bda"
        ),
        "artifacts/v74_persistent_duration_spec/result.json": (
            "294da8a230b3fee5dd278816fbc0b1b3a9cfa3ee1c0a37f350a07b7df09d70b2"
        ),
        "artifacts/v74_persistent_duration_spec/artifact_manifest.json": (
            "7059b71727a1028ddaca090a5cac88027eaf5b7843b9439de8c3c636619f233c"
        ),
        "artifacts/v74_persistent_duration_spec/source_receipt.json": (
            "6284d76811469ffc0d2f351a35659d933ac341086f2bcce807a028cd7fa6b93e"
        ),
    }
    access = phase_contract.get("access_contract", {})
    inputs = phase_contract.get("input_contract", {})
    if (
        set(access.get("allowed_inputs", [])) != set(expected_inputs)
        or inputs.get("expected_static_file_sha256_by_path") != expected_inputs
    ):
        raise ResearchStateError("V75 frozen V74 input contract drift")
    for relative, expected_hash in expected_inputs.items():
        path = _resolve_project_path(root, relative)
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise ResearchStateError(f"V75 V74 input drift: {relative}")

    def load_json(relative: str) -> dict[str, Any]:
        return json.loads(
            _resolve_project_path(root, relative).read_text(encoding="utf-8")
        )

    specification = load_json(
        "artifacts/v74_persistent_duration_spec/specification.json"
    )
    blueprint = load_json("artifacts/v74_persistent_duration_spec/blueprint.json")
    audit = load_json("artifacts/v74_persistent_duration_spec/audit.json")
    result = load_json("artifacts/v74_persistent_duration_spec/result.json")
    manifest = load_json(
        "artifacts/v74_persistent_duration_spec/artifact_manifest.json"
    )
    source_receipt = load_json(
        "artifacts/v74_persistent_duration_spec/source_receipt.json"
    )

    def registered_hash(value: dict[str, Any], key: str, expected: str) -> bool:
        payload = dict(value)
        registered = payload.pop(key, None)
        return registered == expected == _canonical_sha256(payload)

    canonical = inputs.get("expected_canonical_sha256", {})
    if (
        not registered_hash(
            specification,
            "specification_sha256",
            canonical.get("specification", ""),
        )
        or not registered_hash(
            blueprint, "blueprint_sha256", canonical.get("blueprint", "")
        )
        or not registered_hash(result, "result_sha256", canonical.get("result", ""))
        or not registered_hash(
            manifest,
            "artifact_manifest_sha256",
            canonical.get("artifact_manifest", ""),
        )
        or not registered_hash(
            source_receipt,
            "source_receipt_sha256",
            canonical.get("source_receipt", ""),
        )
        or audit.get("passed") is not True
        or len(audit.get("checks", {})) != 16
        or not all(audit.get("checks", {}).values())
        or result.get("decision")
        != "authorize_v75_synthetic_persistent_duration_harness_only"
        or result.get("family_id") != "tlm_persistent_multi_horizon_duration_v1"
        or result.get("summary", {}).get("total_parameters") != 1083155
        or result.get("summary", {}).get("registered_training_jobs") != 9
        or result.get("summary", {}).get("json_metadata_reads") != 4
        or any(
            result.get("summary", {}).get(key) != 0
            for key in (
                "parquet_deserializations",
                "checkpoint_reads",
                "model_instantiations",
                "optimizer_steps",
                "predictions",
                "positions",
                "performance_metrics",
                "pnl_computations",
                "outcome_source_reads",
                "target_asset_rows",
            )
        )
    ):
        raise ResearchStateError("V75 V74 receipt semantic or canonical drift")

    harness = phase_contract.get("synthetic_harness_contract", {})
    required_forbidden = {
        "parquet_market_panel_label_role_or_real_data_access",
        "prior_family_or_real_checkpoint_access_or_reuse",
        "real_training_inference_prediction_position_metric_pnl_or_bootstrap",
        "outcome_packet_or_outcome_source_read",
        "target_asset_access",
        "architecture_objective_policy_threshold_size_or_hyperparameter_change",
        "v76_implementation_or_dataset_build",
    }
    target = phase_contract.get("target_contract", {})
    safety = state.get("safety", {})
    if (
        harness.get("expected_total_parameters") != 1083155
        or harness.get("devices") != ["cpu", "mps"]
        or harness.get("dtype") != "float32"
        or harness.get("checkpoint_kind")
        != "synthetic_only_not_trained_candidate"
        or harness.get("byte_identical_replay_required") is not True
        or not required_forbidden.issubset(
            set(access.get("forbidden_capabilities", []))
        )
        or target.get("status") != "sealed"
        or target.get("target_assets_loaded") != []
        or state.get("target_assets", {}).get("status") != "sealed"
        or state.get("deployable_strategy") is not False
        or safety.get("synthetic_only_phase") is not True
        or safety.get("v74_specification_audit_passed") is not True
        or safety.get("v74_byte_identical_replay_passed") is not True
        or safety.get("v75_real_data_checkpoint_outcome_or_target_access_allowed")
        is not False
        or safety.get("v75_target_assets_remain_sealed") is not True
    ):
        raise ResearchStateError("V75 synthetic, target, or access boundary drift")


def _validate_v76_persistent_duration_dataset_boundary(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
) -> None:
    """Validate the dataset-only V76 handoff without opening research tables."""

    action = "authorize_v76_non_target_persistent_duration_dataset_only"
    command = (
        "PYTHONPATH=src python3 -m tlm persistent-duration-dataset "
        "--config configs/v76_persistent_duration_dataset.yaml"
    )
    if (
        state.get("authorized_phase") != "v76"
        or state.get("authorized_next_action") != action
        or state.get("authorized_command") != command
        or state.get("active_family_status") != "dataset_authorized_not_started"
        or state.get("last_completed_phase")
        != "v75_synthetic_persistent_duration_harness"
        or state.get("evidence_tier")
        != "causal_non_target_dataset_construction_only"
        or experiment.get("authorized_next_action") != action
        or experiment.get("status") != "authorized_not_started"
        or experiment.get("phase") != "non_target_persistent_duration_dataset"
        or phase_contract.get("status") != "dataset_authorized_not_started"
    ):
        raise ResearchStateError("V76 dataset authorization state drift")

    metadata_inputs = {
        "artifacts/v75_persistent_duration_harness/result.json": (
            "e99c62c3a339f9c17d1fd3c444e6130be448d214c8f44e80fdc22aa1ec285f96"
        ),
        "artifacts/v75_persistent_duration_harness/audit.json": (
            "a905e90ccd371243d70eac170d6ab672b618dd3bfaea25b45a16489cbbd80730"
        ),
        "artifacts/v75_persistent_duration_harness/harness_spec.json": (
            "22d173850daf9a2c427eea520292ff5adfda8e3d6471a31bab6a909004efa839"
        ),
        "artifacts/v75_persistent_duration_harness/replay_receipt.json": (
            "a7758e4ea3e2d53378359c5504d82f306f890e5831a57b63d292bcc5717024a9"
        ),
        "artifacts/v75_persistent_duration_harness/artifact_manifest.json": (
            "43fbfa52870a7df3fc4ee228a23b8d9728bf1e62c81c9da2d7d6b25db2fda3ab"
        ),
        "artifacts/v75_persistent_duration_harness/source_receipt.json": (
            "897eaadfcaba023db9705e7331c1a5ae87d99bc0c55951201996b4e405df1090"
        ),
        "artifacts/v32_selected_universe_dataset/result.json": (
            "9a41f84a7de9534472c9655a577c84de992604633f091080e3b7906420b1da53"
        ),
        "artifacts/v32_selected_universe_dataset/audit.json": (
            "280c8857cf990cf62411628e1b3fe30bf04ffcc940db927f995aa487ad09f106"
        ),
        "artifacts/v32_selected_universe_dataset/dataset_manifest.json": (
            "921dbb3d85194f4367a2bcdf1475188c31c2bf438a7c7c28bb8770ff688acc53"
        ),
        "artifacts/v32_selected_universe_dataset/feature_schema.json": (
            "95feb4957fee2805d182c7b3820c4a9693da1df558e1a09d89d82ce0a75f920c"
        ),
        "artifacts/v32_selected_universe_dataset/asset_folds.json": (
            "5a242dcb08bbb46afd085ac21d50cc2567d078f1263aa341b1fdaad49b025a45"
        ),
        "artifacts/v32_selected_universe_dataset/triplet_catalog.json": (
            "d243e3afc76c32736ca5f31d55a852b1fb8c01535e5797a55329bfd8a8dd1ce2"
        ),
    }
    parquet_receipts = {
        "data/processed/selected_universe_panel_v32.parquet": (
            "dc8d50af79a9272a25f952cfd266e461ee938d60d8a19654b9eedd93a4ac5f3a"
        ),
        "data/processed/selected_universe_sequence_index_v32.parquet": (
            "3bb609586809bc7fe4d55b55e11df0581f147484005bc5eff9fd45b28b79d861"
        ),
    }
    expected_inputs = {**metadata_inputs, **parquet_receipts}
    access = phase_contract.get("access_contract", {})
    input_contract = phase_contract.get("input_contract", {})
    if (
        set(access.get("allowed_inputs", [])) != set(expected_inputs)
        or input_contract.get("expected_static_file_sha256_by_path")
        != expected_inputs
    ):
        raise ResearchStateError("V76 frozen input contract drift")
    for relative, expected_hash in metadata_inputs.items():
        path = _resolve_project_path(root, relative)
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise ResearchStateError(f"V76 metadata input drift: {relative}")
    if any(
        not _resolve_project_path(root, relative).is_file()
        for relative in parquet_receipts
    ):
        raise ResearchStateError("V76 registered source table is missing")

    def load_json(relative: str) -> dict[str, Any]:
        return json.loads(
            _resolve_project_path(root, relative).read_text(encoding="utf-8")
        )

    result = load_json("artifacts/v75_persistent_duration_harness/result.json")
    audit = load_json("artifacts/v75_persistent_duration_harness/audit.json")
    harness_spec = load_json(
        "artifacts/v75_persistent_duration_harness/harness_spec.json"
    )
    replay = load_json(
        "artifacts/v75_persistent_duration_harness/replay_receipt.json"
    )
    manifest = load_json(
        "artifacts/v75_persistent_duration_harness/artifact_manifest.json"
    )
    source_receipt = load_json(
        "artifacts/v75_persistent_duration_harness/source_receipt.json"
    )
    v32_result = load_json("artifacts/v32_selected_universe_dataset/result.json")
    v32_audit = load_json("artifacts/v32_selected_universe_dataset/audit.json")
    v32_manifest = load_json(
        "artifacts/v32_selected_universe_dataset/dataset_manifest.json"
    )

    def registered_hash(value: dict[str, Any], key: str, expected: str) -> bool:
        payload = dict(value)
        registered = payload.pop(key, None)
        return registered == expected == _canonical_sha256(payload)

    canonical = input_contract.get("expected_canonical_sha256", {})
    ledger = audit.get("operation_ledger", {})
    forbidden_ledger_keys = (
        "parquet_deserializations",
        "real_panel_or_label_reads",
        "previous_checkpoint_reads",
        "real_training_epochs",
        "real_market_predictions",
        "real_performance_metrics",
        "real_pnl_evaluations",
        "target_asset_loads",
    )
    if (
        not registered_hash(
            harness_spec,
            "harness_spec_sha256",
            canonical.get("v75_harness_spec", ""),
        )
        or not registered_hash(
            result, "result_sha256", canonical.get("v75_result", "")
        )
        or not registered_hash(
            replay,
            "replay_receipt_sha256",
            canonical.get("v75_replay_receipt", ""),
        )
        or not registered_hash(
            manifest,
            "artifact_manifest_sha256",
            canonical.get("v75_artifact_manifest", ""),
        )
        or not registered_hash(
            source_receipt,
            "source_receipt_sha256",
            canonical.get("v75_source_receipt", ""),
        )
        or audit.get("passed") is not True
        or len(audit.get("checks", {})) != 17
        or not all(audit.get("checks", {}).values())
        or result.get("decision") != action
        or result.get("family_id")
        != "tlm_persistent_multi_horizon_duration_v1"
        or result.get("smoke", {}).get("parameter_count") != 1083155
        or result.get("smoke", {}).get("cpu_joint_backward_finite") is not True
        or result.get("smoke", {}).get("mps_joint_backward_finite") is not True
        or result.get("smoke", {}).get("optimizer_steps_executed") != 6
        or replay.get("byte_identical") is not True
        or ledger.get("authorized_metadata_reads") != 6
        or ledger.get("synthetic_checkpoint_writes") != 1
        or ledger.get("synthetic_checkpoint_reads") != 1
        or ledger.get("synthetic_optimizer_steps") != 6
        or any(ledger.get(key) != 0 for key in forbidden_ledger_keys)
        or v32_audit.get("passed") is not True
        or v32_result.get("audit", {}).get("checks", {}).get(
            "no_target_symbol_loaded"
        )
        is not True
        or set(v32_manifest.get("symbols", []))
        & {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
        or v32_manifest.get("symbol_count") != 30
        or v32_manifest.get("panel_rows") != 60210
        or v32_manifest.get("sequence_index_rows") != 49919
        or v32_manifest.get("panel_sha256")
        != parquet_receipts["data/processed/selected_universe_panel_v32.parquet"]
        or v32_manifest.get("sequence_index_sha256")
        != parquet_receipts[
            "data/processed/selected_universe_sequence_index_v32.parquet"
        ]
    ):
        raise ResearchStateError("V76 authorization receipt semantic drift")

    correction = phase_contract.get("source_receipt_correction", {})
    hardening = phase_contract.get("pre_deserialization_hardening", {})
    data_access = phase_contract.get("data_access_contract", {})
    output_contract = phase_contract.get("output_contract", {})
    malformed = "3bb609586809bc7fe4d55e11df0581f147484005bc5eff9fd45b28b79d861"
    authoritative = parquet_receipts[
        "data/processed/selected_universe_sequence_index_v32.parquet"
    ]
    required_forbidden = {
        "target_asset_access",
        "universe_fold_or_triplet_reselection",
        "missing_row_imputation_repair_or_future_fill",
        "scaler_fit",
        "model_instantiation",
        "checkpoint_deserialization",
        "optimizer_step_or_training",
        "market_prediction_or_position",
        "performance_metric_pnl_or_bootstrap",
        "outcome_packet_or_outcome_source_read",
        "architecture_objective_policy_threshold_size_or_hyperparameter_change",
        "v77_implementation_or_training",
    }
    target = phase_contract.get("target_contract", {})
    safety = state.get("safety", {})
    if (
        correction.get("malformed_v74_value") != malformed
        or correction.get("malformed_v74_length") != len(malformed) == 61
        or correction.get("authoritative_v32_value") != authoritative
        or correction.get("authoritative_v32_length") != len(authoritative) == 64
        or correction.get("scientific_semantics_changed") is not False
        or correction.get("source_rows_changed") is not False
        or correction.get("source_values_changed") is not False
        or correction.get("panel_or_sequence_deserializations_during_registration")
        != 0
        or correction.get("post_v75_gate_hash_only_reads") != 2
        or hardening.get("status")
        != "registered_before_first_v76_parquet_deserialization"
        or hardening.get("maximum_panel_value_date") != "2024-12-31"
        or hardening.get("adaptive_evaluation_role_source")
        != "sequence_index_dates_only"
        or hardening.get("adaptive_evaluation_label_values_materialized")
        is not False
        or hardening.get("scientific_semantics_changed") is not False
        or hardening.get(
            "first_v76_parquet_deserialization_completed_before_hardening"
        )
        is not False
        or input_contract.get("admitted_panel_rows_after_outcome_blind_filter")
        != 43830
        or input_contract.get("admitted_panel_date_end") != "2024-12-31"
        or data_access.get("panel_filter")
        != {
            "column": "date",
            "operator": "less_than_or_equal",
            "value": "2024-12-31",
        }
        or data_access.get("authorized_parquet_deserializations_per_execution")
        != 2
        or data_access.get("post_2024_panel_value_rows_allowed") != 0
        or data_access.get("adaptive_evaluation_values_allowed") is not False
        or output_contract.get("require_byte_identical_replay") is not True
        or len(output_contract.get("labels_columns", [])) != 17
        or phase_contract.get("role_contract", {}).get("physical_flags")
        != [
            "eligible_train",
            "eligible_internal_validation",
            "eligible_adaptive_development_evaluation",
        ]
        or not required_forbidden.issubset(
            set(access.get("forbidden_capabilities", []))
        )
        or target.get("status") != "sealed"
        or target.get("target_assets_loaded") != []
        or state.get("target_assets", {}).get("status") != "sealed"
        or state.get("deployable_strategy") is not False
        or safety.get("synthetic_only_phase") is not False
        or safety.get("dataset_only_phase") is not True
        or safety.get("v75_harness_audit_passed") is not True
        or safety.get("v75_byte_identical_full_packet_replay_passed") is not True
        or safety.get(
            "v76_source_hash_correction_registered_before_dataset_deserialization"
        )
        is not True
        or safety.get("v76_post_v75_gate_hash_only_reads") != 2
        or safety.get("v76_post_v75_gate_parquet_deserializations") != 0
        or safety.get("v76_pre_deserialization_hardening_registered") is not True
        or safety.get("v76_maximum_panel_value_date") != "2024-12-31"
        or safety.get("v76_adaptive_evaluation_label_values_allowed") is not False
        or safety.get("v76_target_assets_remain_sealed") is not True
    ):
        raise ResearchStateError("V76 correction, target, or access boundary drift")


def _validate_v77_persistent_duration_training_boundary(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
) -> None:
    """Validate V77 training authority without deserializing research tables."""

    action = "authorize_v77_frozen_non_target_persistent_duration_training_only"
    command = (
        "PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src python3 -m tlm "
        "persistent-duration-training "
        "--config configs/v77_persistent_duration_training.yaml"
    )
    if (
        state.get("authorized_phase") != "v77"
        or state.get("authorized_next_action") != action
        or state.get("authorized_command") != command
        or state.get("active_family_status")
        != "dataset_passed_training_authorized"
        or state.get("last_completed_phase")
        != "v76_non_target_persistent_duration_dataset"
        or state.get("last_completed_result")
        != "artifacts/v76_non_target_persistent_duration_dataset/result.json"
        or state.get("evidence_tier") != "causal_non_target_training_only"
        or experiment.get("authorized_next_action") != action
        or experiment.get("status") != "authorized_not_started"
        or experiment.get("phase")
        != "frozen_non_target_persistent_duration_training"
        or phase_contract.get("status")
        != "dataset_passed_training_authorized"
    ):
        raise ResearchStateError("V77 training authorization state drift")

    metadata_inputs = {
        "artifacts/v74_persistent_duration_spec/blueprint.json": (
            "790b14fd3feacbc62f416e63a019f192ea09d0fd3756c9d757286968ece3632d"
        ),
        "artifacts/v75_persistent_duration_harness/result.json": (
            "e99c62c3a339f9c17d1fd3c444e6130be448d214c8f44e80fdc22aa1ec285f96"
        ),
        "artifacts/v75_persistent_duration_harness/audit.json": (
            "a905e90ccd371243d70eac170d6ab672b618dd3bfaea25b45a16489cbbd80730"
        ),
        "artifacts/v75_persistent_duration_harness/harness_spec.json": (
            "22d173850daf9a2c427eea520292ff5adfda8e3d6471a31bab6a909004efa839"
        ),
        "artifacts/v76_non_target_persistent_duration_dataset/result.json": (
            "0c47084321f2da98699bd736742613e92b32a1657ffd0b19ca392a9b86d11cdd"
        ),
        "artifacts/v76_non_target_persistent_duration_dataset/audit.json": (
            "64587b8fde34d33ac2d5a5d1d70bb5423527cd8f10a73fea4dcaedaf57c1b4b0"
        ),
        "artifacts/v76_non_target_persistent_duration_dataset/dataset_spec.json": (
            "fa9b5c8b95b893cea0a360e1573d9b4e4bfc9a28fbb15e70c7ff6943e9f75edc"
        ),
        "artifacts/v76_non_target_persistent_duration_dataset/dataset_manifest.json": (
            "86d021a1e45988aa55c11ecb779b852d238faaa27433df1d26b9c871bafb30b4"
        ),
        "artifacts/v76_non_target_persistent_duration_dataset/label_schema.json": (
            "132e0b146b015431425601f927c525eac0b6fd0a2ac1258349a5b2b7f50743ce"
        ),
        "artifacts/v76_non_target_persistent_duration_dataset/source_receipt.json": (
            "73a4939c0aa3ad20fa02effa0e4a779a5de5afcb2860df4f887a4edc2fc19894"
        ),
        "artifacts/v76_non_target_persistent_duration_dataset/replay_receipt.json": (
            "d427852a3a9d7ae57c28b28d4eb08bca11b557cee5354656ddda9086e1817abb"
        ),
        "artifacts/v76_non_target_persistent_duration_dataset/artifact_manifest.json": (
            "e5487df7162ec2cf5d651d4fe825e09a3f02aa904f7946be448a470af5a9ddb3"
        ),
        "artifacts/v76_non_target_persistent_duration_dataset/data_access.json": (
            "d9012102f12a339c38a470c1f6487537d8e83d914a5d3b631bd58ea3f1d86232"
        ),
        "artifacts/v32_selected_universe_dataset/feature_schema.json": (
            "95feb4957fee2805d182c7b3820c4a9693da1df558e1a09d89d82ce0a75f920c"
        ),
        "artifacts/v32_selected_universe_dataset/asset_folds.json": (
            "5a242dcb08bbb46afd085ac21d50cc2567d078f1263aa341b1fdaad49b025a45"
        ),
        "artifacts/v32_selected_universe_dataset/triplet_catalog.json": (
            "d243e3afc76c32736ca5f31d55a852b1fb8c01535e5797a55329bfd8a8dd1ce2"
        ),
        "research/waivers/v077_external_backup_owner_waiver.json": (
            "c08a7f0711df03159a9bf2f770a71dfcba8d1d11741beeb7c3556732d42e7629"
        ),
    }
    data_receipts = {
        "data/processed/selected_universe_panel_v32.parquet": (
            "dc8d50af79a9272a25f952cfd266e461ee938d60d8a19654b9eedd93a4ac5f3a"
        ),
        "data/processed/persistent_duration_labels_v76.parquet": (
            "cd9df5aee873c2c73f0c46feaf654eff846fb761ce2fb439991c3f5f4882613f"
        ),
        "data/processed/persistent_duration_sequence_roles_v76.parquet": (
            "549c6bb61136d5d38b312b422f6cdc544b997c5162d9282a6414ca8c9451fea3"
        ),
    }
    expected_inputs = {**metadata_inputs, **data_receipts}
    access = phase_contract.get("access_contract", {})
    input_contract = phase_contract.get("input_contract", {})
    if (
        set(access.get("allowed_inputs", [])) != set(expected_inputs)
        or input_contract.get("expected_file_sha256_by_path") != expected_inputs
    ):
        raise ResearchStateError("V77 frozen input contract drift")
    for relative, expected_hash in metadata_inputs.items():
        path = _resolve_project_path(root, relative)
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise ResearchStateError(f"V77 metadata input drift: {relative}")
    if any(
        not _resolve_project_path(root, relative).is_file()
        for relative in data_receipts
    ):
        raise ResearchStateError("V77 registered data input is missing")

    def load_json(relative: str) -> dict[str, Any]:
        return json.loads(
            _resolve_project_path(root, relative).read_text(encoding="utf-8")
        )

    def registered_hash(value: dict[str, Any], key: str, expected: str) -> bool:
        payload = dict(value)
        registered = payload.pop(key, None)
        return registered == expected == _canonical_sha256(payload)

    blueprint = load_json("artifacts/v74_persistent_duration_spec/blueprint.json")
    v75_result = load_json("artifacts/v75_persistent_duration_harness/result.json")
    v75_audit = load_json("artifacts/v75_persistent_duration_harness/audit.json")
    v75_spec = load_json(
        "artifacts/v75_persistent_duration_harness/harness_spec.json"
    )
    result = load_json(
        "artifacts/v76_non_target_persistent_duration_dataset/result.json"
    )
    audit = load_json(
        "artifacts/v76_non_target_persistent_duration_dataset/audit.json"
    )
    dataset_spec = load_json(
        "artifacts/v76_non_target_persistent_duration_dataset/dataset_spec.json"
    )
    dataset_manifest = load_json(
        "artifacts/v76_non_target_persistent_duration_dataset/dataset_manifest.json"
    )
    label_schema = load_json(
        "artifacts/v76_non_target_persistent_duration_dataset/label_schema.json"
    )
    source_receipt = load_json(
        "artifacts/v76_non_target_persistent_duration_dataset/source_receipt.json"
    )
    replay_receipt = load_json(
        "artifacts/v76_non_target_persistent_duration_dataset/replay_receipt.json"
    )
    artifact_manifest = load_json(
        "artifacts/v76_non_target_persistent_duration_dataset/artifact_manifest.json"
    )
    canonical = input_contract.get("expected_canonical_sha256", {})
    ledger = audit.get("operation_ledger", {})
    summary = result.get("summary", {})
    if (
        blueprint.get("blueprint_sha256") != canonical.get("v74_blueprint")
        or blueprint.get("candidate_family_id")
        != "tlm_persistent_multi_horizon_duration_v1"
        or not registered_hash(
            v75_spec,
            "harness_spec_sha256",
            canonical.get("v75_harness_spec", ""),
        )
        or not registered_hash(
            v75_result, "result_sha256", canonical.get("v75_result", "")
        )
        or v75_audit.get("passed") is not True
        or len(v75_audit.get("checks", {})) != 17
        or not all(v75_audit.get("checks", {}).values())
        or not registered_hash(
            result, "result_sha256", canonical.get("v76_result", "")
        )
        or not registered_hash(
            dataset_spec,
            "dataset_spec_sha256",
            canonical.get("v76_dataset_spec", ""),
        )
        or not registered_hash(
            dataset_manifest,
            "dataset_manifest_sha256",
            canonical.get("v76_dataset_manifest", ""),
        )
        or not registered_hash(
            label_schema,
            "label_schema_sha256",
            canonical.get("v76_label_schema", ""),
        )
        or not registered_hash(
            source_receipt,
            "source_receipt_sha256",
            canonical.get("v76_source_receipt", ""),
        )
        or not registered_hash(
            replay_receipt,
            "replay_receipt_sha256",
            canonical.get("v76_replay_receipt", ""),
        )
        or not registered_hash(
            artifact_manifest,
            "artifact_manifest_sha256",
            canonical.get("v76_artifact_manifest", ""),
        )
        or audit.get("passed") is not True
        or len(audit.get("checks", {})) != 16
        or not all(audit.get("checks", {}).values())
        or result.get("decision") != action
        or summary.get("label_rows") != 43830
        or summary.get("sequence_role_rows") != 49919
        or summary.get("complete_persistent_rows") != 43478
        or summary.get("train_eligible_rows") != 24060
        or summary.get("internal_validation_eligible_rows") != 10628
        or summary.get("adaptive_evaluation_role_rows") != 9798
        or summary.get("adaptive_evaluation_label_values_loaded") is not False
        or ledger.get("authorized_parquet_deserializations") != 2
        or any(
            ledger.get(key) != 0
            for key in (
                "scaler_fits",
                "model_instantiations",
                "optimizer_steps",
                "checkpoint_reads",
                "market_predictions",
                "performance_metrics",
                "pnl_evaluations",
                "target_asset_loads",
            )
        )
        or dataset_manifest.get("source_panel_sha256")
        != data_receipts["data/processed/selected_universe_panel_v32.parquet"]
        or dataset_manifest.get("labels", {}).get("sha256")
        != data_receipts["data/processed/persistent_duration_labels_v76.parquet"]
        or dataset_manifest.get("sequence_roles", {}).get("sha256")
        != data_receipts[
            "data/processed/persistent_duration_sequence_roles_v76.parquet"
        ]
        or set(dataset_manifest.get("symbols", []))
        & {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
    ):
        raise ResearchStateError("V77 source authorization receipt semantic drift")

    architecture = phase_contract.get("model_and_objective_contract", {}).get(
        "architecture", {}
    )
    objective = phase_contract.get("model_and_objective_contract", {})
    training = phase_contract.get("grid_optimizer_and_runtime_contract", {})
    roles = phase_contract.get("data_and_role_contract", {})
    scaler = phase_contract.get("feature_and_scaler_contract", {})
    checkpoint = phase_contract.get("checkpoint_and_replay_contract", {})
    runtime = phase_contract.get("runtime_contract", {})
    target = phase_contract.get("target_contract", {})
    safety = state.get("safety", {})
    required_forbidden = {
        "target_asset_access",
        "adaptive_evaluation_role_or_any_2025_value_access",
        "heldout_fold_asset_training_validation_or_scaler_fit",
        "prior_family_or_prior_checkpoint_weight_scaler_optimizer_or_state_reuse",
        "architecture_objective_size_seed_fold_or_hyperparameter_change",
        "prediction_position_policy_or_portfolio_generation",
        "performance_metric_pnl_or_bootstrap",
        "outcome_packet_or_outcome_source_read",
        "seed_fold_checkpoint_or_epoch_selection_for_economic_use",
        "v78_implementation_or_evaluation",
    }
    repair = phase_contract.get("contract_repair", {})
    enforcement = phase_contract.get("operator_enforcement_contract", {})
    ledger_contract = phase_contract.get("data_access_ledger_contract", {})
    backup_policy = runtime.get("backup_policy", {})
    if (
        architecture
        != {
            "input_shape": [None, 256, 3, 9],
            "d_model": 128,
            "patch_length_days": 16,
            "patch_stride_days": 8,
            "patch_count": 31,
            "temporal_encoder_layers": 4,
            "cross_asset_attention_layers": 1,
            "attention_heads": 8,
            "feed_forward_width": 512,
            "dropout": 0.15,
            "activation": "gelu",
            "norm_first": True,
            "shared_causal_asset_encoder": True,
            "asset_slot_embedding": False,
            "maximum_duration_days": 7,
            "expected_parameter_count": 1083155,
        }
        or objective.get("objective_weights")
        != {"return_nll": 1.0, "pairwise_ranking": 0.25, "duration_nll": 0.5}
        or objective.get("student_t_degrees_of_freedom") != 5.0
        or objective.get("early_stopping_monitor")
        != "weighted_validation_joint_objective"
        or training.get("folds") != [1, 2, 3]
        or training.get("seeds") != [42, 7, 123]
        or training.get("expected_jobs") != 9
        or training.get("initialization") != "fresh_registered_seed"
        or training.get("prior_checkpoint_reuse") != "none"
        or training.get("device") != "mps"
        or training.get("dtype") != "float32"
        or training.get("mps_fallback_allowed") is not False
        or training.get("maximum_epochs") != 40
        or training.get("early_stopping_patience") != 6
        or training.get("early_stopping_minimum_delta") != 1.0e-6
        or training.get("hyperparameter_search_allowed") is not False
        or training.get("seed_fold_or_epoch_selection_allowed") is not False
        or roles.get("forbidden_role_columns")
        != ["eligible_adaptive_development_evaluation"]
        or roles.get("any_2025_or_later_value_allowed") is not False
        or roles.get("full_table_materialization_then_filter_allowed") is not False
        or scaler.get("scaler_count") != 3
        or scaler.get("one_scaler_per_fold") is not True
        or scaler.get("shared_across_three_seeds_within_fold") is not True
        or scaler.get("fit_role") != "eligible_train_only"
        or checkpoint.get("expected_final_checkpoint_count") != 9
        or checkpoint.get("zero_step_replay_required") is not True
        or runtime.get("pytorch_enable_mps_fallback") != "0"
        or runtime.get("maximum_active_optimizer_jobs") != 1
        or repair.get("supersedes_revision")
        != "v077_frozen_non_target_persistent_duration_training_r2"
        or repair.get("allowed_change_scope")
        != "add_smoke_data_access_receipt_to_terminal_artifact_packet_only"
        or repair.get("scientific_architecture_objective_grid_roles_and_hyperparameters_changed")
        is not False
        or enforcement.get("operation_order")
        != ["doctor", "smoke", "full", "verify", "replay"]
        or ledger_contract.get("outcome_rows_read") != 0
        or ledger_contract.get("target_assets_loaded") != []
        or ledger_contract.get("performance_metrics_computed") is not False
        or ledger_contract.get("pnl_computed") is not False
        or backup_policy.get("mode") != "owner_waiver"
        or backup_policy.get("waiver")
        != {
            "path": "research/waivers/v077_external_backup_owner_waiver.json",
            "file_sha256": "c08a7f0711df03159a9bf2f770a71dfcba8d1d11741beeb7c3556732d42e7629",
        }
        or not required_forbidden.issubset(
            set(access.get("forbidden_capabilities", []))
        )
        or target.get("status") != "sealed"
        or target.get("target_assets_loaded") != []
        or state.get("target_assets", {}).get("status") != "sealed"
        or state.get("deployable_strategy") is not False
        or safety.get("dataset_only_phase") is not False
        or safety.get("training_only_phase") is not True
        or safety.get("v76_dataset_audit_passed") is not True
        or safety.get("v76_audit_checks_passed") != 16
        or safety.get("v76_byte_identical_replay_passed") is not True
        or safety.get("v76_adaptive_evaluation_label_values_loaded") is not False
        or safety.get("v77_exact_job_count") != 9
        or safety.get("v77_fresh_initialization_only") is not True
        or safety.get("v77_train_only_scaler_per_fold") is not True
        or safety.get("v77_2025_or_later_values_allowed") is not False
        or safety.get("v77_target_assets_remain_sealed") is not True
        or safety.get("v77_prediction_position_metric_pnl_outcome_allowed")
        is not False
        or safety.get(
            "v77_r3_smoke_data_access_receipt_registered_before_data_access"
        )
        is not True
        or safety.get("v77_r3_scientific_contract_changed") is not False
    ):
        raise ResearchStateError("V77 training, target, or access boundary drift")


def _validate_v83_low_turnover_rank_training_boundary(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
) -> None:
    """Validate V83 training authority without opening either Parquet."""

    action = "authorize_v83_frozen_non_target_low_turnover_rank_training_only"
    command = (
        "PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src python3 -m tlm "
        "low-turnover-rank-training "
        "--config configs/v83_low_turnover_rank_training.yaml"
    )
    if (
        state.get("authorized_phase") != "v83"
        or state.get("authorized_next_action") != action
        or state.get("authorized_command") != command
        or state.get("active_family_status") != "dataset_passed_training_authorized"
        or state.get("last_completed_phase") != "v82_non_target_low_turnover_rank_dataset"
        or state.get("last_completed_result")
        != "artifacts/v82_low_turnover_rank_dataset/result.json"
        or state.get("evidence_tier") != "causal_non_target_training_only"
        or experiment.get("authorized_next_action") != action
        or experiment.get("status") != "authorized_not_started"
        or experiment.get("phase") != "frozen_non_target_low_turnover_rank_training"
        or phase_contract.get("status") != "dataset_passed_training_authorized"
    ):
        raise ResearchStateError("V83 training authorization state drift")
    expected = phase_contract.get("input_contract", {}).get(
        "expected_file_sha256_by_path", {}
    )
    allowed = phase_contract.get("access_contract", {}).get("allowed_inputs", [])
    if not isinstance(expected, dict) or set(expected) != set(allowed) or len(expected) != 18:
        raise ResearchStateError("V83 frozen input allowlist drift")
    parquet_paths = {
        "data/processed/low_turnover_rank_development_features_v82.parquet",
        "data/processed/low_turnover_rank_development_labels_v82.parquet",
    }
    forbidden = {
        "data/processed/low_turnover_rank_evaluation_features_v82.parquet",
        "data/processed/low_turnover_rank_evaluation_outcomes_v82.parquet",
    }
    if forbidden.intersection(expected) or not parquet_paths.issubset(expected):
        raise ResearchStateError("V83 dataset boundary drift")
    for relative, digest in expected.items():
        path = _resolve_project_path(root, relative)
        if not path.is_file() or not isinstance(digest, str) or len(digest) != 64:
            raise ResearchStateError(f"V83 registered input missing: {relative}")
        if relative not in parquet_paths and _sha256_file(path) != digest:
            raise ResearchStateError(f"V83 metadata input drift: {relative}")

    result_path = _resolve_project_path(
        root, "artifacts/v82_low_turnover_rank_dataset/result.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    registered = result.get("result_sha256")
    body = {key: value for key, value in result.items() if key != "result_sha256"}
    blueprint = json.loads(
        _resolve_project_path(
            root, "artifacts/v80_low_turnover_rank_spec/blueprint.json"
        ).read_text(encoding="utf-8")
    )
    grid = phase_contract.get("grid_optimizer_and_runtime_contract", {})
    architecture = phase_contract.get("model_and_objective_contract", {}).get(
        "architecture", {}
    )
    runtime = phase_contract.get("runtime_contract", {})
    sampling = phase_contract.get("sampling_contract", {})
    target = phase_contract.get("target_contract", {})
    if (
        registered != "87d0b0dcd5e6b6b07be6572e73eb53eb2dffbddfc2b8c254d2144eb2f67cc0e6"
        or _canonical_sha256(body) != registered
        or result.get("decision") != action
        or result.get("audit", {}).get("passed") is not True
        or blueprint.get("blueprint_sha256")
        != "3b080b6cfcea2be6ef2a3347397e7f669573870abba0f6966bc3eb76eeb1d649"
        or architecture.get("input_shape") != [None, 128, 3, 8]
        or architecture.get("expected_parameter_count") != 10993
        or grid.get("folds") != [1, 2, 3]
        or grid.get("seeds") != [42, 7, 123]
        or grid.get("expected_jobs") != 9
        or grid.get("maximum_epochs") != 100
        or grid.get("early_stopping_patience") != 10
        or grid.get("prior_checkpoint_reuse") != "none"
        or grid.get("mps_fallback_allowed") is not False
        or sampling.get("train_samples_per_epoch") != 12120
        or sampling.get("fixed_validation_samples") != 19380
        or sampling.get("seed_dependent_sampling") is not False
        or runtime.get("backup_policy", {}).get("mode") != "owner_waiver"
        or runtime.get("pytorch_enable_mps_fallback") != "0"
        or target.get("status") != "sealed"
        or state.get("target_assets", {}).get("status") != "sealed"
        or state.get("deployable_strategy") is not False
    ):
        raise ResearchStateError("V83 frozen training contract drift")


def _validate_v84_low_turnover_rank_evaluation_prepare_boundary(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
) -> None:
    """Validate V84 prepare authority without deserializing research tables."""

    action = "authorize_v84_outcome_blind_low_turnover_rank_evaluation_prepare_only"
    command = (
        "PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src python3 -m tlm "
        "low-turnover-rank-evaluation-prepare "
        "--config configs/v84_low_turnover_rank_evaluation.yaml"
    )
    if (
        state.get("authorized_phase") != "v84"
        or state.get("authorized_next_action") != action
        or state.get("authorized_command") != command
        or state.get("active_family_status")
        != "trained_outcome_blind_evaluation_prepare_authorized"
        or state.get("last_completed_phase")
        != "v83_frozen_non_target_low_turnover_rank_training"
        or state.get("last_completed_result")
        != "artifacts/v83_low_turnover_rank_training/result.json"
        or state.get("evidence_tier")
        != "retrospective_non_target_first_use_2026_prepare_outcomes_sealed"
        or experiment.get("authorized_next_action") != action
        or experiment.get("status") != "authorized_not_started"
        or experiment.get("phase")
        != "outcome_blind_retrospective_non_target_low_turnover_rank_evaluation_prepare"
        or phase_contract.get("status")
        != "trained_outcome_blind_evaluation_prepare_authorized"
    ):
        raise ResearchStateError("V84 outcome-blind prepare authorization state drift")

    expected = phase_contract.get("input_contract", {}).get(
        "expected_file_sha256_by_path", {}
    )
    allowed = phase_contract.get("access_contract", {}).get("allowed_inputs", [])
    forbidden_prepare = {
        "data/processed/low_turnover_rank_evaluation_outcomes_v82.parquet",
        "data/processed/low_turnover_rank_development_labels_v82.parquet",
    }
    if (
        not isinstance(expected, dict)
        or len(expected) != 39
        or set(allowed) != set(expected)
        or len(allowed) != len(set(allowed))
        or forbidden_prepare.intersection(expected)
        or "data/processed/low_turnover_rank_evaluation_features_v82.parquet"
        not in expected
    ):
        raise ResearchStateError("V84 frozen input allowlist drift")
    for relative, digest in expected.items():
        path = _resolve_project_path(root, relative)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or not path.is_file()
            or _sha256_file(path) != digest
        ):
            raise ResearchStateError(f"V84 input file/hash drift: {relative}")

    def load_json(relative: str) -> dict[str, Any]:
        return json.loads(
            _resolve_project_path(root, relative).read_text(encoding="utf-8")
        )

    result = load_json("artifacts/v83_low_turnover_rank_training/result.json")
    audit = load_json("artifacts/v83_low_turnover_rank_training/audit.json")
    completion = load_json(
        "artifacts/v83_low_turnover_rank_training/completion_receipt.json"
    )
    checkpoints = load_json(
        "artifacts/v83_low_turnover_rank_training/checkpoint_manifest.json"
    )
    scalers = load_json(
        "artifacts/v83_low_turnover_rank_training/scaler_manifest.json"
    )
    sealed = load_json(
        "artifacts/v82_low_turnover_rank_dataset/sealed_packet_receipt.json"
    )
    jobs = checkpoints.get("jobs", [])
    expected_jobs = {
        f"{fold}|{seed}" for fold in (1, 2, 3) for seed in (42, 7, 123)
    }
    if (
        result.get("decision") != action
        or result.get("result_sha256")
        != "d94bd923444301ead42a82eb5b4f5e8fdb3dd7456ff11db718a47bbdf3278856"
        or result.get("summary", {}).get("completed_jobs") != 9
        or result.get("summary", {}).get("checkpoint_count") != 9
        or result.get("summary", {}).get("total_optimizer_steps") != 5040
        or result.get("summary", {}).get("predictions") != 0
        or result.get("summary", {}).get("performance_metrics") != 0
        or result.get("summary", {}).get("pnl_evaluations") != 0
        or result.get("summary", {}).get("target_asset_loads") != 0
        or audit.get("passed") is not True
        or completion.get("completion_receipt_sha256")
        != "143e7ac9119d2813e4f0a0f3f76b464993e23f1482af37cfededeb0dfa78d182"
        or len(jobs) != 9
        or {row.get("job_id") for row in jobs} != expected_jobs
        or any(row.get("status") != "completed" for row in jobs)
        or len(scalers.get("folds", [])) != 3
        or sealed.get("status") != "sealed"
        or sealed.get("unseal_count") != 0
        or sealed.get("file_sha256")
        != "9cc5be0e9dfdc40b4fe8d6433602769d67bfa6b269b5f02fa2d241e6eca0024a"
    ):
        raise ResearchStateError("V84 terminal training or sealed packet drift")

    evaluation = phase_contract.get("evaluation_contract", {})
    policy = phase_contract.get("policy_contract", {})
    bootstrap = phase_contract.get("registered_bootstrap", {})
    gates = phase_contract.get("outcome_blind_gate_contract", {}).get("gates", [])
    one_shot = phase_contract.get("one_shot_contract", {})
    outcome = phase_contract.get("outcome_request_contract", {})
    target = phase_contract.get("target_contract", {})
    safety = state.get("safety", {})
    repair = phase_contract.get("contract_repair", {})
    required_gates = [
        "all_registered_checkpoints_used_without_selection",
        "exact_fold_triplet_date_scope",
        "missingness_matches_registered_readiness",
        "prediction_distribution_finite_and_nonconstant",
        "centered_scores_and_seed_ensemble_exact",
        "permutation_and_lexical_structure_complete",
        "action_space_and_state_transitions_exact",
        "turnover_and_final_liquidation_exact",
        "control_positions_exact",
        "aggregate_turnover_within_registered_ceiling",
        "exposure_fraction_within_registered_bounds",
        "zero_outcome_and_target_access",
    ]
    config_hash = _sha256_file(
        _resolve_project_path(root, "configs/v84_low_turnover_rank_evaluation.yaml")
    )
    if (
        config_hash
        != "433ee7b42fa6b6554c0b0f5525498b92dad326de1fb4c65bf1e588302a3eff82"
        or evaluation.get("folds") != [1, 2, 3]
        or evaluation.get("seeds") != [42, 7, 123]
        or evaluation.get("window", {}).get("signal_dates") != 159
        or evaluation.get("window", {}).get("signal_end") != "2026-06-08"
        or evaluation.get("triplet_scope")
        != "exact_120_lexical_combinations_of_each_folds_ten_heldout_assets"
        or evaluation.get("inference", {}).get("device") != "mps"
        or evaluation.get("inference", {}).get("mps_fallback_allowed") is not False
        or evaluation.get("inference", {}).get("checkpoint_state")
        != "model_best_state_at_registered_early_stopping_best_epoch"
        or policy.get("decision_interval_eligible_dates") != 21
        or policy.get("switch_margin") != 0.25
        or policy.get("structural_maximum_turnover") != 16.0
        or policy.get("reporting_cost_bps") != [10, 20, 30]
        or bootstrap.get("paths") != 10000
        or bootstrap.get("block_lengths_days") != [7, 21, 42]
        or gates != required_gates
        or one_shot.get("current_stage") != "outcome_blind_prepare"
        or one_shot.get("prepare", {}).get("outcome_rows_read") != 0
        or one_shot.get("unseal", {}).get("maximum_unseal_count") != 1
        or one_shot.get("unseal", {}).get("generic_continue_is_not_authorization")
        is not True
        or outcome.get("sealed_source_sha256")
        != "9cc5be0e9dfdc40b4fe8d6433602769d67bfa6b269b5f02fa2d241e6eca0024a"
        or outcome.get("exact_source_read_count") != 1
        or target.get("status") != "sealed"
        or target.get("target_assets_loaded") != []
        or state.get("target_assets", {}).get("status") != "sealed"
        or state.get("deployable_strategy") is not False
        or safety.get("v83_training_audit_passed") is not True
        or safety.get("v83_completed_jobs") != 9
        or safety.get("v83_checkpoint_count") != 9
        or safety.get("v84_outcome_blind_prepare_authorized") is not True
        or safety.get("v84_outcome_access_allowed_during_prepare") is not False
        or safety.get("v84_performance_or_pnl_allowed_during_prepare") is not False
        or safety.get("v84_all_nine_checkpoints_required_without_selection")
        is not True
        or safety.get("v84_target_assets_remain_sealed") is not True
        or safety.get("v84_explicit_hash_bound_unseal_authorization_present")
        is not False
        or repair.get("supersedes_revision")
        != "v084_outcome_blind_low_turnover_rank_evaluation_prepare_r1"
        or repair.get("timing")
        != "before_any_model_instantiation_inference_prediction_position_or_outcome_access"
        or repair.get(
            "architecture_checkpoint_policy_cost_accounting_gate_and_hyperparameters_changed"
        )
        is not False
        or repair.get("observed_before_repair", {}).get("model_instantiations") != 0
        or repair.get("observed_before_repair", {}).get("inference_batches") != 0
    ):
        raise ResearchStateError("V84 evaluation, one-shot, or target boundary drift")


def _validate_v85_low_turnover_rank_unseal_boundary(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
) -> None:
    """Validate V85 exact unseal authority without deserializing outcome values."""

    action = (
        "execute_v85_exactly_one_registered_non_target_outcome_unseal_and_complete_"
        "evaluation"
    )
    command = (
        "PYTHONPATH=src python3 -m tlm low-turnover-rank-evaluation-unseal "
        "--config configs/v85_low_turnover_rank_evaluation.yaml"
    )
    if (
        state.get("authorized_phase") != "v85"
        or state.get("authorized_next_action") != action
        or state.get("authorized_command") != command
        or state.get("active_family_status")
        != "retrospective_non_target_economic_evaluation_exact_unseal_authorized"
        or state.get("last_completed_phase")
        != "v84_outcome_blind_low_turnover_rank_evaluation_prepare"
        or state.get("last_completed_result")
        != "artifacts/v84_low_turnover_rank_evaluation/result.json"
        or state.get("evidence_tier")
        != "retrospective_non_target_first_use_2026_not_prospective_confirmation"
        or experiment.get("authorized_next_action") != action
        or experiment.get("status") != "authorized_not_started"
        or experiment.get("phase")
        != "retrospective_non_target_first_use_2026_one_shot_economic_evaluation"
        or phase_contract.get("status")
        != "frozen_after_hash_bound_user_authorization_before_exactly_one_outcome_unseal"
    ):
        raise ResearchStateError("V85 one-shot authorization state drift")

    expected = phase_contract.get("input_contract", {}).get(
        "expected_file_sha256_by_path", {}
    )
    allowed = phase_contract.get("access_contract", {}).get("allowed_inputs", [])
    if (
        not isinstance(expected, dict)
        or len(expected) != 12
        or set(allowed) != set(expected)
        or len(allowed) != len(set(allowed))
    ):
        raise ResearchStateError("V85 frozen input allowlist drift")
    for relative, digest in expected.items():
        path = _resolve_project_path(root, relative)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or not path.is_file()
            or _sha256_file(path) != digest
        ):
            raise ResearchStateError(f"V85 input file/hash drift: {relative}")

    def load_json(relative: str) -> dict[str, Any]:
        return json.loads(
            _resolve_project_path(root, relative).read_text(encoding="utf-8")
        )

    result = load_json("artifacts/v84_low_turnover_rank_evaluation/result.json")
    audit = load_json("artifacts/v84_low_turnover_rank_evaluation/audit.json")
    prepare = load_json(
        "artifacts/v84_low_turnover_rank_evaluation/prepare_receipt.json"
    )
    packet = load_json("artifacts/v84_low_turnover_rank_evaluation/one_shot_packet.json")
    behavior = load_json(
        "artifacts/v84_low_turnover_rank_evaluation/behavior_audit.json"
    )
    authorization = load_json(
        "research/authorizations/v085_low_turnover_rank_outcome_unseal.json"
    )
    registered_authorization = authorization.pop("authorization_sha256", None)
    if (
        result.get("decision")
        != "authorize_v85_exactly_one_registered_non_target_outcome_unseal_only_after_explicit_hash_bound_user_authorization"
        or result.get("evaluation_spec_sha256")
        != "70191a2e02e82f405cec9112b0708e20db8e658932ac0448a794f811db12ec80"
        or result.get("prepare_receipt_sha256")
        != "e0d65959afca7b928bb74d466bdbfb8e5ff2c2f46b08fa075c7ab04c649e611e"
        or result.get("registered_sha256")
        != "82ee01cba2815c57a59cde16cf03d9a47b5bcb75421d0e350011793889548560"
        or result.get("one_shot_packet_sha256")
        != "a5a21254eed06b7f95c875aa5b01046e4168e2dde5dcd442dd3cc6d0c4e180ea"
        or audit.get("passed") is not True
        or behavior.get("passed") is not True
        or prepare.get("authorizes_unseal") is not True
        or prepare.get("outcome_rows_read") != 0
        or packet.get("phase") != "prepare"
        or packet.get("authorization", {}).get("explicit_user_authorization")
        is not False
        or packet.get("prepare", {}).get("positions_frozen") is not True
        or packet.get("prepare", {}).get("predictions_frozen") is not True
        or registered_authorization
        != "9400f8987a596237fce6d09ac98019f02d5b1c804ce7706dcdd9e869690009ca"
        or _canonical_sha256(authorization) != registered_authorization
        or authorization.get("authorized_action") != action
        or authorization.get("evaluation_spec_sha256")
        != result.get("evaluation_spec_sha256")
        or authorization.get("prepare_receipt_sha256")
        != result.get("prepare_receipt_sha256")
        or authorization.get("registered_sha256")
        != result.get("registered_sha256")
        or authorization.get("maximum_unseal_count") != 1
        or authorization.get("target_assets_status") != "sealed"
    ):
        raise ResearchStateError("V85 prepare or explicit authorization drift")

    access = phase_contract.get("outcome_access_contract", {})
    evaluation = phase_contract.get("evaluation_contract", {})
    one_shot = phase_contract.get("one_shot_contract", {})
    target = phase_contract.get("target_contract", {})
    safety = state.get("safety", {})
    bootstrap = evaluation.get("bootstrap", {})
    if (
        access.get("source_packet_sha256")
        != "9cc5be0e9dfdc40b4fe8d6433602769d67bfa6b269b5f02fa2d241e6eca0024a"
        or access.get("expected_rows") != 5370
        or access.get("maximum_source_packet_deserializations") != 1
        or access.get("replay_source_packet_deserializations") != 0
        or evaluation.get("costs_bps") != [10, 20, 30]
        or evaluation.get("mandatory_gate_categories") != 9
        or evaluation.get("mandatory_gate_cells") != 19
        or evaluation.get("aggregate_rescue_allowed") is not False
        or evaluation.get("missing_cell_pass_allowed") is not False
        or bootstrap.get("paths") != 10000
        or bootstrap.get("block_lengths_days") != [7, 21, 42]
        or bootstrap.get("base_seed") != 20260716
        or bootstrap.get("seed_derivation") != "base_seed_plus_block_length"
        or one_shot.get("current_stage") != "authorized_unseal_and_completion"
        or one_shot.get("explicit_user_authorization_present") is not True
        or one_shot.get("maximum_unseal_count") != 1
        or one_shot.get("source_outcome_rows_read_first_execution") != 5370
        or one_shot.get("source_outcome_rows_read_replay") != 0
        or target.get("status") != "sealed"
        or target.get("target_assets_loaded") != []
        or state.get("target_assets", {}).get("status") != "sealed"
        or state.get("deployable_strategy") is not False
        or safety.get("v85_exact_hash_bound_user_authorization_present") is not True
        or safety.get("v85_maximum_unseal_count") != 1
        or safety.get("v85_target_assets_remain_sealed") is not True
        or safety.get("v85_retuning_or_regeneration_allowed") is not False
    ):
        raise ResearchStateError("V85 outcome, metric, or target boundary drift")


def _validate_v78_persistent_duration_evaluation_prepare_boundary(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
) -> None:
    """Validate V78 prepare authority without reading research table contents."""

    action = "authorize_v78_outcome_blind_persistent_duration_evaluation_prepare_only"
    command = (
        "PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src python3 -m tlm "
        "persistent-duration-evaluation-prepare "
        "--config configs/v78_persistent_duration_evaluation.yaml"
    )
    if (
        state.get("authorized_phase") != "v78"
        or state.get("authorized_next_action") != action
        or state.get("authorized_command") != command
        or state.get("active_family_status")
        != "trained_outcome_blind_evaluation_prepare_authorized"
        or state.get("last_completed_phase")
        != "v77_frozen_non_target_persistent_duration_training"
        or state.get("last_completed_result")
        != "artifacts/v77_persistent_duration_training/result.json"
        or state.get("evidence_tier")
        != "adaptive_consumed_2025_non_target_development_prepare_outcomes_sealed"
        or experiment.get("authorized_next_action") != action
        or experiment.get("status") != "authorized_not_started"
        or experiment.get("phase")
        != "outcome_blind_adaptive_non_target_persistent_duration_evaluation_prepare"
        or phase_contract.get("status")
        != "trained_outcome_blind_evaluation_prepare_authorized"
    ):
        raise ResearchStateError("V78 outcome-blind prepare authorization state drift")

    expected = phase_contract.get("input_contract", {}).get(
        "expected_file_sha256_by_path", {}
    )
    allowed = phase_contract.get("access_contract", {}).get("allowed_inputs", [])
    if (
        not isinstance(expected, dict)
        or len(expected) != 40
        or set(allowed) != set(expected)
        or len(allowed) != len(set(allowed))
    ):
        raise ResearchStateError("V78 frozen input allowlist drift")
    forbidden_prepare_inputs = {
        "data/processed/persistent_duration_labels_v76.parquet",
    }
    if forbidden_prepare_inputs.intersection(expected):
        raise ResearchStateError("V78 prepare allowlist contains an outcome table")
    for relative, digest in expected.items():
        path = _resolve_project_path(root, relative)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or not path.is_file()
            or _sha256_file(path) != digest
        ):
            raise ResearchStateError(f"V78 input file/hash drift: {relative}")

    def load_json(relative: str) -> dict[str, Any]:
        return json.loads(
            _resolve_project_path(root, relative).read_text(encoding="utf-8")
        )

    result = load_json("artifacts/v77_persistent_duration_training/result.json")
    audit = load_json("artifacts/v77_persistent_duration_training/audit.json")
    completion = load_json(
        "artifacts/v77_persistent_duration_training/completion_receipt.json"
    )
    checkpoint_manifest = load_json(
        "artifacts/v77_persistent_duration_training/checkpoint_manifest.json"
    )
    grid = load_json("artifacts/v77_persistent_duration_training/grid_manifest.json")
    scaler = load_json(
        "artifacts/v77_persistent_duration_training/scaler_manifest.json"
    )
    summary = result.get("summary", {})
    checkpoint_jobs = checkpoint_manifest.get("jobs", [])
    expected_jobs = {
        f"{fold}|{seed}" for fold in (1, 2, 3) for seed in (42, 7, 123)
    }
    if (
        result.get("decision") != action
        or result.get("result_sha256")
        != "4a5fd26a68e68fc44ca0a316bcb44b59de6300bc4a1844e520a56975dedcf097"
        or audit.get("passed") is not True
        or audit.get("audit_sha256")
        != "24e48be46649740c3c456685c8391d933685f6d318dd05a81ccd5ae96a68f11a"
        or completion.get("completion_receipt_sha256")
        != "f3e5bbad74022aef6016a9db2117b0426e592f29ed9435e3234bd8eddc7654d1"
        or summary.get("completed_jobs") != 9
        or summary.get("checkpoint_count") != 9
        or summary.get("total_optimizer_steps") != 6976
        or summary.get("predictions") != 0
        or summary.get("performance_metrics") != 0
        or summary.get("pnl_evaluations") != 0
        or summary.get("target_asset_loads") != 0
        or set(grid.get("completed_jobs", [])) != expected_jobs
        or grid.get("selected_jobs") != []
        or len(checkpoint_jobs) != 9
        or {row.get("job_id") for row in checkpoint_jobs} != expected_jobs
        or any(row.get("selected_for_economic_use") is True for row in checkpoint_jobs)
        or len(scaler.get("folds", [])) != 3
        or scaler.get("manifest_sha256")
        != "3453ffcc490d235b076f9ba6ef4ce67bc60c86c6bfc050cf0c371bada73ae291"
    ):
        raise ResearchStateError("V78 V77-terminal receipt semantic drift")

    evaluation = phase_contract.get("evaluation_contract", {})
    policy = phase_contract.get("policy_contract", {})
    bootstrap = phase_contract.get("registered_bootstrap", {})
    gates = phase_contract.get("outcome_blind_gate_contract", {}).get("gates", [])
    one_shot = phase_contract.get("one_shot_contract", {})
    outcome = phase_contract.get("outcome_request_contract", {})
    target = phase_contract.get("target_contract", {})
    safety = state.get("safety", {})
    repair = phase_contract.get("contract_repair", {})
    required_gates = [
        "all_registered_checkpoints_used_without_selection",
        "exact_fold_triplet_date_scope",
        "missingness_matches_registered_readiness",
        "prediction_distribution_finite_and_nonconstant",
        "positive_scale_and_valid_survival",
        "ensemble_disagreement_finite",
        "permutation_and_lexical_structure_complete",
        "action_space_and_state_transitions_exact",
        "turnover_and_final_liquidation_exact",
        "control_positions_exact",
        "aggregate_turnover_within_registered_ceiling",
        "zero_outcome_and_target_access",
    ]
    config_hash = _sha256_file(
        _resolve_project_path(root, "configs/v78_persistent_duration_evaluation.yaml")
    )
    if (
        config_hash
        != "4f6a4d00a92e799674ca29d10c36bfcf4c80aa9584bdef290ddb057d35c63f22"
        or evaluation.get("folds") != [1, 2, 3]
        or evaluation.get("seeds") != [42, 7, 123]
        or evaluation.get("triplet_scope")
        != "exact_120_lexical_combinations_of_each_folds_ten_heldout_assets"
        or evaluation.get("fold_portfolio_aggregation")
        != "equal_weight_120_exact_triplet_subaccounts_without_selection"
        or evaluation.get("fold_triplet_calendar")
        != "every_triplet_has_all_357_registered_signal_dates"
        or evaluation.get("window", {}).get("signal_dates") != 357
        or evaluation.get("inference", {}).get("device") != "mps"
        or evaluation.get("inference", {}).get("mps_fallback_allowed") is not False
        or evaluation.get("inference", {}).get("checkpoint_state")
        != "model_best_state_at_registered_early_stopping_best_epoch"
        or repair.get("supersedes_revision")
        != "v078_outcome_blind_persistent_duration_evaluation_prepare_r2"
        or repair.get("timing")
        != "before_any_v78_parquet_or_checkpoint_deserialization"
        or repair.get("seed_fold_epoch_architecture_objective_policy_or_gate_selection_changed")
        is not False
        or policy.get("base_cost_bps") != 10
        or policy.get("reporting_cost_bps") != [10, 20, 30]
        or policy.get("horizon_weights") != [0.2, 0.3, 0.5]
        or policy.get("unavailable_triplet_action")
        != "force_cash_with_exact_exit_turnover_and_no_capital_redistribution"
        or policy.get("policy_tuning_allowed") is not False
        or bootstrap.get("paths") != 10000
        or bootstrap.get("block_lengths_days") != [7, 21, 63]
        or bootstrap.get("base_seed") != 20260715
        or gates != required_gates
        or one_shot.get("current_stage") != "outcome_blind_prepare"
        or one_shot.get("prepare", {}).get("outcome_rows_read") != 0
        or one_shot.get("unseal", {}).get("maximum_unseal_count") != 1
        or one_shot.get("unseal", {}).get("generic_continue_is_not_authorization")
        is not True
        or outcome.get("source_projection_after_unseal_only")
        != ["date", "symbol", "raw_open"]
        or outcome.get("exact_source_read_count") != 1
        or target.get("status") != "sealed"
        or target.get("target_assets_loaded") != []
        or state.get("target_assets", {}).get("status") != "sealed"
        or state.get("deployable_strategy") is not False
        or safety.get("v77_training_audit_passed") is not True
        or safety.get("v77_completed_jobs") != 9
        or safety.get("v77_checkpoint_count") != 9
        or safety.get("v78_outcome_blind_prepare_authorized") is not True
        or safety.get("v78_outcome_access_allowed_during_prepare") is not False
        or safety.get("v78_performance_or_pnl_allowed_during_prepare") is not False
        or safety.get("v78_all_nine_checkpoints_required_without_selection")
        is not True
        or safety.get("v78_target_assets_remain_sealed") is not True
        or safety.get("v78_explicit_hash_bound_unseal_authorization_present")
        is not False
        or safety.get("v78_r2_best_epoch_state_binding_registered_before_data_access")
        is not True
        or safety.get("v78_r2_scientific_selection_contract_changed") is not False
        or safety.get(
            "v78_r3_fixed_capital_missing_triplet_cash_registered_before_data_access"
        )
        is not True
    ):
        raise ResearchStateError("V78 evaluation, one-shot, or target boundary drift")


def _validate_v79_v78_terminal_record_boundary(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
) -> None:
    """Validate the metadata-only V78 terminal registration boundary."""

    action = "record_v79_v78_terminal_failure_metadata_only"
    command = (
        "PYTHONPATH=src python3 -m tlm v78-terminal-record "
        "--config configs/v79_v78_terminal_record.yaml"
    )
    evidence = (
        "adaptive_consumed_2025_non_target_development_prepare_outcomes_sealed"
    )
    if (
        state.get("authorized_phase") != "v79"
        or state.get("authorized_next_action") != action
        or state.get("authorized_command") != command
        or state.get("active_family_id")
        != "tlm_persistent_multi_horizon_duration_v1"
        or state.get("active_family_status")
        != "outcome_blind_prepare_failed_metadata_record_authorized"
        or state.get("last_completed_phase")
        != "v78_outcome_blind_persistent_duration_evaluation_prepare_failed"
        or state.get("last_completed_result")
        != "artifacts/v78_persistent_duration_evaluation/result.json"
        or state.get("evidence_tier") != evidence
        or experiment.get("authorized_next_action") != action
        or experiment.get("status") != "authorized_not_started"
        or experiment.get("phase") != "metadata_only_v78_terminal_failure_record"
        or phase_contract.get("status")
        != "metadata_only_terminal_record_authorized_not_started"
        or phase_contract.get("evidence_tier") != evidence
    ):
        raise ResearchStateError("V79 metadata-only terminal record state drift")

    expected_inputs = {
        "artifacts/v78_persistent_duration_evaluation/result.json": (
            "ce50db6966f3ebcbcde63994603a62bd2f20545ecd6e1ffd6f5aef2047db9bba"
        ),
        "artifacts/v78_persistent_duration_evaluation/audit.json": (
            "6003e7e358d52e3e83692bf1499f39551510aff1060d96a3b052dc859a30e818"
        ),
        "artifacts/v78_persistent_duration_evaluation/prepare_failure_receipt.json": (
            "3269eb9ceaa6d92bc96970793a69ac96f6e66bf7608d52740e199c8558e64f3a"
        ),
        "artifacts/v78_persistent_duration_evaluation/replay_receipt.json": (
            "651e8f1814be682c077dc5526c5b4bac90b12de77f4fdade7c541cfb18e134fc"
        ),
    }
    access = phase_contract.get("access_contract", {})
    inputs = phase_contract.get("input_contract", {})
    if (
        set(access.get("allowed_inputs", [])) != set(expected_inputs)
        or inputs.get("expected_static_file_sha256_by_path") != expected_inputs
        or inputs.get("allowed_extensions") != [".json"]
        or inputs.get("expected_json_metadata_reads") != 4
        or inputs.get("maximum_parquet_deserializations") != 0
        or inputs.get("maximum_outcome_packet_reads") != 0
        or inputs.get("maximum_checkpoint_or_model_loads") != 0
    ):
        raise ResearchStateError("V79 metadata input contract drift")
    for relative, expected_hash in expected_inputs.items():
        path = _resolve_project_path(root, relative)
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise ResearchStateError(f"V79 metadata input drift: {relative}")

    result = _load_mapping(
        _resolve_project_path(
            root, "artifacts/v78_persistent_duration_evaluation/result.json"
        )
    )
    audit = _load_mapping(
        _resolve_project_path(
            root, "artifacts/v78_persistent_duration_evaluation/audit.json"
        )
    )
    failure = _load_mapping(
        _resolve_project_path(
            root,
            "artifacts/v78_persistent_duration_evaluation/prepare_failure_receipt.json",
        )
    )
    replay = _load_mapping(
        _resolve_project_path(
            root,
            "artifacts/v78_persistent_duration_evaluation/replay_receipt.json",
        )
    )
    decision = "pivot_away_from_current_family_without_target_evaluation_or_retuning"
    if (
        result.get("decision") != decision
        or result.get("family_id") != "tlm_persistent_multi_horizon_duration_v1"
        or result.get("audit", {}).get("passed") is not False
        or result.get("audit", {}).get("failed_checks")
        != ["aggregate_turnover_within_registered_ceiling"]
        or result.get("summary", {}).get("aggregate_candidate_turnover") != 59.55
        or result.get("summary", {}).get("registered_turnover_ceiling") != 45.0
        or result.get("summary", {}).get("outcome_rows_read") != 0
        or result.get("summary", {}).get("performance_metrics") != 0
        or result.get("summary", {}).get("pnl_evaluations") != 0
        or result.get("summary", {}).get("target_assets_loaded") != 0
        or result.get("one_shot_packet_created") is not False
        or result.get("one_shot_unseal_authorized") is not False
        or audit.get("decision") != decision
        or audit.get("passed") is not False
        or audit.get("failed_checks")
        != ["aggregate_turnover_within_registered_ceiling"]
        or audit.get("failure_is_accounting_bug") is not False
        or audit.get("outcome_rows_read") != 0
        or audit.get("performance_metrics_computed") != 0
        or audit.get("pnl_evaluations") != 0
        or audit.get("target_assets_loaded") != []
        or failure.get("decision") != decision
        or failure.get("authorizes_unseal") is not False
        or failure.get("outcome_rows_read") != 0
        or failure.get("retuning_or_policy_change") is not False
        or failure.get("prediction_or_position_regeneration") is not False
        or replay.get("decision") != decision
        or replay.get("passed") is not True
        or replay.get("frozen_artifact_hashes_match") is not True
        or replay.get("scientific_source_parquet_deserializations") != 0
        or replay.get("checkpoint_container_deserializations") != 0
        or replay.get("model_instantiations") != 0
        or replay.get("outcome_rows_read") != 0
        or replay.get("target_assets_loaded") != []
    ):
        raise ResearchStateError("V79 frozen V78 terminal receipt semantic drift")

    target = phase_contract.get("target_contract", {})
    record = phase_contract.get("record_contract", {})
    safety = state.get("safety", {})
    if (
        target.get("status") != "sealed"
        or target.get("target_assets_loaded") != []
        or state.get("target_assets", {}).get("status") != "sealed"
        or state.get("deployable_strategy") is not False
        or record.get("retired_family_id")
        != "tlm_persistent_multi_horizon_duration_v1"
        or record.get("successor_family_id")
        != "tlm_low_turnover_cross_sectional_rank_v1"
        or record.get("successor_specification_executed") is not False
        or safety.get("v79_metadata_only_terminal_recording_phase") is not True
        or safety.get("v79_parquet_checkpoint_model_outcome_or_target_access_allowed")
        is not False
        or safety.get("v79_scientific_recomputation_allowed") is not False
        or safety.get("v79_v78_family_retirement_authorized") is not True
        or safety.get("v79_v80_specification_execution_allowed") is not False
        or safety.get("v79_target_assets_remain_sealed") is not True
    ):
        raise ResearchStateError("V79 retirement, successor, or target boundary drift")


def _validate_v80_low_turnover_rank_specification_boundary(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
) -> None:
    """Validate authority for V80 metadata-only successor specification."""

    action = "authorize_v80_low_turnover_cross_sectional_rank_specification_only"
    command = (
        "PYTHONPATH=src python3 -m tlm low-turnover-rank-spec "
        "--config configs/v80_low_turnover_rank_spec.yaml"
    )
    if (
        state.get("authorized_phase") != "v80"
        or state.get("authorized_next_action") != action
        or state.get("authorized_command") != command
        or state.get("active_family_id")
        != "tlm_low_turnover_cross_sectional_rank_v1"
        or state.get("active_family_status")
        != "outcome_blind_specification_authorized_not_started"
        or state.get("last_completed_phase")
        != "v79_metadata_only_v78_terminal_record"
        or state.get("last_completed_result")
        != "artifacts/v79_v78_terminal_record/result.json"
        or state.get("evidence_tier") != "metadata_only_final_family_specification"
        or experiment.get("authorized_next_action") != action
        or experiment.get("status") != "authorized_not_started"
        or experiment.get("phase")
        != "outcome_blind_low_turnover_cross_sectional_rank_specification"
        or phase_contract.get("status")
        != "outcome_blind_specification_authorized_not_started"
    ):
        raise ResearchStateError("V80 specification authorization state drift")

    access = phase_contract.get("access_contract", {})
    inputs = phase_contract.get("input_contract", {})
    allowed = access.get("allowed_inputs", [])
    expected = inputs.get("expected_static_file_sha256_by_path", {})
    expected_paths = {
        "artifacts/v79_v78_terminal_record/result.json",
        "artifacts/v79_v78_terminal_record/audit.json",
        "artifacts/v79_v78_terminal_record/terminal_record.json",
        "artifacts/v79_v78_terminal_record/input_hash_receipt.json",
    }
    if (
        set(allowed) != expected_paths
        or set(expected) != expected_paths
        or inputs.get("allowed_extensions") != [".json"]
        or inputs.get("expected_json_metadata_reads") != 4
        or inputs.get("maximum_parquet_deserializations") != 0
        or inputs.get("maximum_checkpoint_or_model_loads") != 0
        or inputs.get("maximum_outcome_or_target_reads") != 0
    ):
        raise ResearchStateError("V80 metadata-only input contract drift")
    for relative, expected_hash in expected.items():
        path = _resolve_project_path(root, relative)
        if (
            not _is_sha256(expected_hash)
            or not path.is_file()
            or _sha256_file(path) != expected_hash
        ):
            raise ResearchStateError(f"V80 V79 authorization input drift: {relative}")

    result = _load_mapping(
        _resolve_project_path(root, "artifacts/v79_v78_terminal_record/result.json")
    )
    audit = _load_mapping(
        _resolve_project_path(root, "artifacts/v79_v78_terminal_record/audit.json")
    )
    record = _load_mapping(
        _resolve_project_path(
            root, "artifacts/v79_v78_terminal_record/terminal_record.json"
        )
    )
    if (
        result.get("decision") != action
        or result.get("retired_family_status") != "retired"
        or result.get("successor_family_id")
        != "tlm_low_turnover_cross_sectional_rank_v1"
        or result.get("successor_specification_executed") is not False
        or result.get("outcome_rows_read") != 0
        or result.get("scientific_metrics_recomputed") != 0
        or result.get("models_or_checkpoints_loaded") != 0
        or result.get("target_assets_loaded") != []
        or audit.get("passed") is not True
        or any(
            value != 0 and key != "json_metadata_reads"
            for key, value in audit.get("access_ledger", {}).items()
            if key != "target_assets_loaded"
        )
        or audit.get("access_ledger", {}).get("json_metadata_reads") != 4
        or audit.get("access_ledger", {}).get("target_assets_loaded") != []
        or record.get("family_status_after") != "retired"
        or record.get("terminal_phase") != "v78"
        or record.get("successor_family_id")
        != "tlm_low_turnover_cross_sectional_rank_v1"
        or record.get("successor_specification_executed") is not False
        or record.get("target_assets_status") != "sealed"
    ):
        raise ResearchStateError("V80 V79 authorization receipt semantic drift")

    target = phase_contract.get("target_contract", {})
    scope = phase_contract.get("specification_scope", {})
    safety = state.get("safety", {})
    if (
        target.get("status") != "sealed"
        or target.get("target_assets_loaded") != []
        or state.get("target_assets", {}).get("status") != "sealed"
        or state.get("deployable_strategy") is not False
        or scope.get("family_is_new") is not True
        or scope.get("prior_checkpoint_weight_scaler_or_optimizer_reuse_allowed")
        is not False
        or scope.get("architecture_or_parameter_count_frozen_before_v80") is not False
        or scope.get("parameter_increase_as_default_direction") is not False
        or scope.get("single_frozen_variant_required") is not True
        or scope.get("turnover_budget_must_be_structural") is not True
        or safety.get("v80_metadata_only_specification_phase") is not True
        or safety.get("v80_data_checkpoint_model_training_inference_or_outcome_allowed")
        is not False
        or safety.get("v80_target_assets_remain_sealed") is not True
        or safety.get("v80_single_final_family") is not True
    ):
        raise ResearchStateError("V80 specification or target boundary drift")

    config_reference = phase_contract.get("specification_config", {})
    config_path = _resolve_project_path(root, config_reference.get("path"))
    if (
        config_reference.get("path") != "configs/v80_low_turnover_rank_spec.yaml"
        or config_reference.get("file_sha256")
        != "abede81bf6820b7c306545cdca5508736a2569a2564e6f18da6e8060c9afbf15"
        or not config_path.is_file()
        or _sha256_file(config_path) != config_reference.get("file_sha256")
    ):
        raise ResearchStateError("V80 specification config drift")
    config = _load_mapping(config_path).get("low_turnover_rank_spec", {})
    design = config.get("frozen_design", {})
    architecture = design.get("architecture", {})
    policy = design.get("policy", {})
    chronology = design.get("chronology", {})
    training = design.get("training", {})
    evaluation = design.get("evaluation", {})
    terminal = design.get("terminal_decision", {})
    frozen = phase_contract.get("frozen_design_contract", {})
    if (
        architecture.get("family")
        != "shared_causal_depthwise_tcn_deepsets_ranker"
        or architecture.get("expected_total_parameters") != 10993
        or architecture.get("architecture_variant_count") != 1
        or architecture.get("receptive_field_days") != 127
        or policy.get("decision_interval_days") != 21
        or policy.get("maximum_evaluation_decisions") != 8
        or policy.get("structural_maximum_turnover") != 16.0
        or policy.get("threshold_or_interval_tuning_allowed") is not False
        or chronology.get("consumed_2025_outcomes_role") != "forbidden"
        or chronology.get("final_evaluation_signal_start") != "2026-01-01"
        or chronology.get("final_evaluation_signal_end") != "2026-06-09"
        or chronology.get("final_evaluation_outcome_maturity_end") != "2026-06-30"
        or training.get("folds") != [1, 2, 3]
        or training.get("seeds") != [42, 7, 123]
        or training.get("future_job_count") != 9
        or training.get("model_or_hyperparameter_selection") is not False
        or evaluation.get("costs_bps") != [10, 20, 30]
        or len(evaluation.get("mandatory_gates", [])) != 9
        or evaluation.get("aggregate_rescue_for_failed_fold") is not False
        or terminal.get("second_variant_or_rescue_allowed") is not False
        or frozen.get("expected_total_parameters") != 10993
        or frozen.get("architecture_variant_count") != 1
        or frozen.get("structural_maximum_turnover") != 16.0
        or frozen.get("future_training_jobs") != 9
        or frozen.get("consumed_2025_outcomes_allowed") is not False
        or frozen.get("mandatory_financial_gate_count") != 9
        or frozen.get("second_variant_or_rescue_allowed") is not False
    ):
        raise ResearchStateError("V80 frozen design drift")


def _validate_v81_low_turnover_rank_harness_boundary(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
) -> None:
    """Validate the exact synthetic-only V81 harness boundary."""

    action = "authorize_v81_synthetic_low_turnover_rank_harness_only"
    command = (
        "PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src python3 -m tlm "
        "low-turnover-rank-harness "
        "--config configs/v81_low_turnover_rank_harness.yaml"
    )
    if (
        state.get("authorized_phase") != "v81"
        or state.get("authorized_next_action") != action
        or state.get("authorized_command") != command
        or state.get("active_family_id")
        != "tlm_low_turnover_cross_sectional_rank_v1"
        or state.get("active_family_status")
        != "specification_frozen_synthetic_harness_authorized"
        or state.get("last_completed_phase")
        != "v80_low_turnover_cross_sectional_rank_specification"
        or state.get("last_completed_result")
        != "artifacts/v80_low_turnover_rank_spec/result.json"
        or state.get("evidence_tier") != "deterministic_synthetic_harness_only"
        or experiment.get("authorized_next_action") != action
        or experiment.get("status") != "authorized_not_started"
        or experiment.get("phase")
        != "deterministic_synthetic_low_turnover_rank_harness"
        or phase_contract.get("status")
        != "specification_frozen_synthetic_harness_authorized"
    ):
        raise ResearchStateError("V81 synthetic harness authorization state drift")

    expected = phase_contract.get("input_contract", {}).get(
        "expected_static_file_sha256_by_path", {}
    )
    allowed = phase_contract.get("access_contract", {}).get("allowed_inputs", [])
    if (
        not isinstance(expected, dict)
        or len(expected) != 6
        or set(allowed) != set(expected)
        or phase_contract.get("input_contract", {}).get("allowed_extensions")
        != [".json"]
        or phase_contract.get("input_contract", {}).get(
            "expected_json_metadata_reads"
        )
        != 6
        or phase_contract.get("input_contract", {}).get(
            "maximum_real_data_or_parquet_reads"
        )
        != 0
        or phase_contract.get("input_contract", {}).get(
            "maximum_prior_checkpoint_reads"
        )
        != 0
        or phase_contract.get("input_contract", {}).get(
            "maximum_outcome_or_target_reads"
        )
        != 0
    ):
        raise ResearchStateError("V81 metadata input contract drift")
    for relative, expected_hash in expected.items():
        path = _resolve_project_path(root, relative)
        if (
            not _is_sha256(expected_hash)
            or not path.is_file()
            or _sha256_file(path) != expected_hash
        ):
            raise ResearchStateError(f"V81 V80 input drift: {relative}")

    result = _load_mapping(
        _resolve_project_path(root, "artifacts/v80_low_turnover_rank_spec/result.json")
    )
    audit = _load_mapping(
        _resolve_project_path(root, "artifacts/v80_low_turnover_rank_spec/audit.json")
    )
    blueprint = _load_mapping(
        _resolve_project_path(
            root, "artifacts/v80_low_turnover_rank_spec/blueprint.json"
        )
    )
    if (
        result.get("decision") != action
        or result.get("family_id") != "tlm_low_turnover_cross_sectional_rank_v1"
        or result.get("parameter_count") != 10993
        or result.get("future_training_jobs") != 9
        or result.get("structural_maximum_evaluation_turnover") != 16.0
        or result.get("scientific_data_reads") != 0
        or result.get("models_or_checkpoints_loaded") != 0
        or result.get("outcome_rows_read") != 0
        or result.get("target_assets_loaded") != []
        or result.get("v81_executed") is not False
        or audit.get("passed") is not True
        or audit.get("checks_passed") != 14
        or audit.get("checks_total") != 14
        or blueprint.get("blueprint_sha256")
        != "3b080b6cfcea2be6ef2a3347397e7f669573870abba0f6966bc3eb76eeb1d649"
        or blueprint.get("architecture", {}).get("expected_total_parameters")
        != 10993
        or blueprint.get("policy", {}).get("structural_maximum_turnover") != 16.0
    ):
        raise ResearchStateError("V81 V80 specification receipt semantic drift")

    synthetic = phase_contract.get("synthetic_contract", {})
    target = phase_contract.get("target_contract", {})
    safety = state.get("safety", {})
    if (
        synthetic.get("input_shape") != [4, 128, 3, 8]
        or synthetic.get("parameter_count") != 10993
        or synthetic.get("temporal_dilations") != [1, 2, 4, 8, 16, 32]
        or synthetic.get("receptive_field_days") != 127
        or synthetic.get("output_shape") != [4, 3]
        or synthetic.get("structural_maximum_turnover") != 16.0
        or synthetic.get("architecture_or_scientific_contract_change_allowed")
        is not False
        or target.get("status") != "sealed"
        or target.get("target_assets_loaded") != []
        or state.get("target_assets", {}).get("status") != "sealed"
        or state.get("deployable_strategy") is not False
        or safety.get("v81_synthetic_only_phase") is not True
        or safety.get("v81_real_data_prior_checkpoint_outcome_or_target_allowed")
        is not False
        or safety.get("v81_architecture_objective_policy_change_allowed") is not False
        or safety.get("v81_target_assets_remain_sealed") is not True
        or safety.get("v81_v82_implementation_allowed") is not False
    ):
        raise ResearchStateError("V81 synthetic or target boundary drift")


def _validate_v82_r0_low_turnover_rank_chronology_erratum_boundary(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
) -> None:
    """Validate the exact metadata-only V82-R0 chronology repair boundary."""

    action = "record_v82_r0_low_turnover_rank_chronology_erratum_metadata_only"
    command = (
        "PYTHONPATH=src python3 -m tlm low-turnover-rank-chronology-erratum "
        "--config configs/v82_r0_low_turnover_rank_chronology_erratum.yaml"
    )
    if (
        state.get("authorized_phase") != "v82-r0"
        or state.get("authorized_next_action") != action
        or state.get("authorized_command") != command
        or state.get("active_family_id")
        != "tlm_low_turnover_cross_sectional_rank_v1"
        or state.get("active_family_status")
        != "metadata_only_chronology_erratum_authorized"
        or state.get("last_completed_phase")
        != "v81_synthetic_low_turnover_rank_harness"
        or state.get("last_completed_result")
        != "artifacts/v81_low_turnover_rank_harness/result.json"
        or state.get("evidence_tier") != "metadata_only_chronology_erratum"
        or experiment.get("authorized_next_action") != action
        or experiment.get("status") != "authorized_not_started"
        or experiment.get("phase")
        != "metadata_only_low_turnover_rank_chronology_erratum"
        or phase_contract.get("status")
        != "metadata_only_chronology_erratum_authorized_not_started"
    ):
        raise ResearchStateError("V82-R0 chronology erratum state drift")

    expected = phase_contract.get("input_contract", {}).get(
        "expected_static_file_sha256_by_path", {}
    )
    allowed = phase_contract.get("access_contract", {}).get("allowed_inputs", [])
    expected_paths = {
        "research/authorizations/v082_r0_chronology_erratum.json",
        "artifacts/v80_low_turnover_rank_spec/specification.json",
        "artifacts/v80_low_turnover_rank_spec/blueprint.json",
        "artifacts/v80_low_turnover_rank_spec/result.json",
        "artifacts/v81_low_turnover_rank_harness/harness_spec.json",
        "artifacts/v81_low_turnover_rank_harness/audit.json",
        "artifacts/v81_low_turnover_rank_harness/result.json",
        "artifacts/v81_low_turnover_rank_harness/artifact_manifest.json",
    }
    inputs = phase_contract.get("input_contract", {})
    if (
        not isinstance(expected, dict)
        or set(expected) != expected_paths
        or set(allowed) != expected_paths
        or inputs.get("allowed_extensions") != [".json"]
        or inputs.get("expected_json_metadata_reads") != 8
        or inputs.get("maximum_parquet_market_panel_raw_data_reads") != 0
        or inputs.get("maximum_outcome_or_target_value_reads") != 0
        or inputs.get("maximum_checkpoint_model_or_scaler_loads") != 0
    ):
        raise ResearchStateError("V82-R0 metadata-only input contract drift")
    for relative, expected_hash in expected.items():
        path = _resolve_project_path(root, relative)
        if (
            not _is_sha256(expected_hash)
            or not path.is_file()
            or _sha256_file(path) != expected_hash
        ):
            raise ResearchStateError(f"V82-R0 input drift: {relative}")

    authorization = _load_mapping(
        _resolve_project_path(
            root, "research/authorizations/v082_r0_chronology_erratum.json"
        )
    )
    authorization_payload = dict(authorization)
    authorization_hash = authorization_payload.pop("authorization_sha256", None)
    v81_result = _load_mapping(
        _resolve_project_path(
            root, "artifacts/v81_low_turnover_rank_harness/result.json"
        )
    )
    v81_audit = _load_mapping(
        _resolve_project_path(
            root, "artifacts/v81_low_turnover_rank_harness/audit.json"
        )
    )
    if (
        authorization_hash
        != "7120faa7e6267e2234bb34b576a65aa4834bc1b527a9012ab100b7d3c409394c"
        or _canonical_sha256(authorization_payload) != authorization_hash
        or authorization.get("authorized_phase") != "v82-r0"
        or authorization.get("target_assets_status") != "sealed"
        or v81_result.get("decision")
        != "authorize_v82_non_target_low_turnover_rank_dataset_only"
        or v81_result.get("v82_executed") is not False
        or v81_result.get("parameter_count") != 10993
        or v81_result.get("structural_maximum_turnover") != 16.0
        or v81_result.get("outcome_rows_read") != 0
        or v81_result.get("target_assets_loaded") != []
        or v81_audit.get("passed") is not True
        or v81_audit.get("checks_passed") != 15
        or v81_audit.get("checks_total") != 15
    ):
        raise ResearchStateError("V82-R0 authorization or V81 receipt drift")

    erratum = phase_contract.get("erratum_contract", {})
    frozen = phase_contract.get("frozen_unchanged_contract", {})
    target = phase_contract.get("target_contract", {})
    safety = state.get("safety", {})
    config_reference = phase_contract.get("specification_config", {})
    config_path = _resolve_project_path(root, config_reference.get("path"))
    if (
        erratum.get("corrected_fields_exactly")
        != {
            "final_evaluation_signal_end": {
                "before": "2026-06-09",
                "after": "2026-06-08",
            },
            "final_evaluation_signal_dates": {"before": 160, "after": 159},
        }
        or erratum.get("maturity_offset_days") != 22
        or erratum.get("corrected_last_maturity") != "2026-06-30"
        or erratum.get("maximum_decisions_after") != 8
        or erratum.get("structural_maximum_turnover_after") != 16.0
        or frozen.get("target_asset_return")
        != "log_open_t_plus_22_div_open_t_plus_1"
        or frozen.get("expected_total_parameters") != 10993
        or frozen.get("architecture_variant_count") != 1
        or frozen.get("costs_bps") != [10, 20, 30]
        or frozen.get("mandatory_financial_gate_count") != 9
        or frozen.get("scientific_change_count") != 0
        or target.get("status") != "sealed"
        or target.get("target_assets_loaded") != []
        or state.get("target_assets", {}).get("status") != "sealed"
        or state.get("deployable_strategy") is not False
        or safety.get("v82_r0_metadata_only_chronology_erratum_phase") is not True
        or safety.get("v82_r0_scientific_or_target_access_allowed") is not False
        or safety.get("v82_r0_v80_or_v81_artifact_rewrite_allowed") is not False
        or safety.get("v82_r0_v82_dataset_execution_allowed") is not False
        or safety.get("v82_r0_target_assets_remain_sealed") is not True
        or config_reference.get("path")
        != "configs/v82_r0_low_turnover_rank_chronology_erratum.yaml"
        or config_reference.get("file_sha256")
        != "a41b8b2cf0d1a709646a290fdeeafbcd19242d8ba1aea2ba0567fb09dc498134"
        or not config_path.is_file()
        or _sha256_file(config_path) != config_reference.get("file_sha256")
    ):
        raise ResearchStateError("V82-R0 correction, target, or safety drift")


def _validate_v82_low_turnover_rank_dataset_boundary(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
) -> None:
    """Validate the exact causal non-target V82 dataset-only boundary."""

    action = "authorize_v82_non_target_low_turnover_rank_dataset_only"
    command = (
        "PYTHONPATH=src python3 -m tlm low-turnover-rank-dataset "
        "--config configs/v82_low_turnover_rank_dataset.yaml"
    )
    if (
        state.get("authorized_phase") != "v82"
        or state.get("authorized_next_action") != action
        or state.get("authorized_command") != command
        or state.get("active_family_id")
        != "tlm_low_turnover_cross_sectional_rank_v1"
        or state.get("active_family_status")
        != "chronology_erratum_passed_dataset_authorized"
        or state.get("last_completed_phase")
        != "v82_r0_metadata_only_chronology_erratum"
        or state.get("last_completed_result")
        != "artifacts/v82_r0_low_turnover_rank_chronology_erratum/result.json"
        or state.get("evidence_tier")
        != "causal_non_target_dataset_and_sealed_evaluation_packet_only"
        or experiment.get("authorized_next_action") != action
        or experiment.get("status") != "authorized_not_started"
        or experiment.get("phase") != "causal_non_target_low_turnover_rank_dataset"
        or phase_contract.get("status") != "dataset_authorized_not_started"
    ):
        raise ResearchStateError("V82 dataset authorization state drift")

    expected = phase_contract.get("input_contract", {}).get(
        "expected_static_file_sha256_by_path", {}
    )
    allowed = phase_contract.get("access_contract", {}).get("allowed_inputs", [])
    expected_paths = {
        "research/authorizations/v082_dataset_only.json",
        "artifacts/v82_r0_low_turnover_rank_chronology_erratum/result.json",
        "artifacts/v82_r0_low_turnover_rank_chronology_erratum/audit.json",
        "artifacts/v82_r0_low_turnover_rank_chronology_erratum/chronology_erratum.json",
        "artifacts/v82_r0_low_turnover_rank_chronology_erratum/artifact_manifest.json",
        "artifacts/v80_low_turnover_rank_spec/blueprint.json",
        "artifacts/v32_selected_universe_dataset/result.json",
        "artifacts/v32_selected_universe_dataset/audit.json",
        "artifacts/v32_selected_universe_dataset/dataset_manifest.json",
        "artifacts/v32_selected_universe_dataset/feature_schema.json",
        "artifacts/v32_selected_universe_dataset/asset_folds.json",
        "artifacts/v32_selected_universe_dataset/triplet_catalog.json",
    }
    if (
        not isinstance(expected, dict)
        or set(expected) != expected_paths
        or set(allowed) != expected_paths
        or phase_contract.get("input_contract", {}).get("expected_metadata_reads")
        != 12
        or phase_contract.get("input_contract", {}).get(
            "maximum_target_asset_reads"
        )
        != 0
        or phase_contract.get("input_contract", {}).get(
            "maximum_prior_checkpoint_or_scaler_reads"
        )
        != 0
    ):
        raise ResearchStateError("V82 static input contract drift")
    for relative, expected_hash in expected.items():
        path = _resolve_project_path(root, relative)
        if (
            not _is_sha256(expected_hash)
            or not path.is_file()
            or _sha256_file(path) != expected_hash
        ):
            raise ResearchStateError(f"V82 static input drift: {relative}")

    authorization = _load_mapping(
        _resolve_project_path(root, "research/authorizations/v082_dataset_only.json")
    )
    authorization_payload = dict(authorization)
    authorization_hash = authorization_payload.pop("authorization_sha256", None)
    result = _load_mapping(
        _resolve_project_path(
            root,
            "artifacts/v82_r0_low_turnover_rank_chronology_erratum/result.json",
        )
    )
    result_payload = dict(result)
    result_hash = result_payload.pop("result_sha256", None)
    v32_manifest = _load_mapping(
        _resolve_project_path(
            root, "artifacts/v32_selected_universe_dataset/dataset_manifest.json"
        )
    )
    folds = _load_mapping(
        _resolve_project_path(
            root, "artifacts/v32_selected_universe_dataset/asset_folds.json"
        )
    )
    catalog = _load_mapping(
        _resolve_project_path(
            root, "artifacts/v32_selected_universe_dataset/triplet_catalog.json"
        )
    )
    canonical = phase_contract.get("input_contract", {}).get(
        "expected_canonical_sha256", {}
    )
    if (
        authorization_hash != canonical.get("v82_user_authorization")
        or _canonical_sha256(authorization_payload) != authorization_hash
        or authorization.get("bound_v82_r0_result_sha256")
        != canonical.get("v82_r0_result")
        or authorization.get("target_assets_status") != "sealed"
        or result_hash != canonical.get("v82_r0_result")
        or _canonical_sha256(result_payload) != result_hash
        or result.get("decision") != action
        or result.get("final_evaluation_signal_end") != "2026-06-08"
        or result.get("final_evaluation_signal_dates") != 159
        or result.get("final_evaluation_outcome_maturity_end") != "2026-06-30"
        or result.get("target_assets_loaded") != []
        or result.get("v82_dataset_executed") is not False
        or v32_manifest.get("symbol_count") != 30
        or v32_manifest.get("symbols")
        != phase_contract.get("source_contract", {}).get("symbols")
        or len(folds.get("folds", [])) != 3
        or len(catalog.get("folds", [])) != 3
        or catalog.get("catalog_sha256") != canonical.get("v32_triplet_catalog")
    ):
        raise ResearchStateError("V82 authorization, chronology, or V32 ancestry drift")

    source = phase_contract.get("source_contract", {})
    feature = phase_contract.get("feature_contract", {})
    label = phase_contract.get("label_contract", {})
    roles = phase_contract.get("role_contract", {})
    evaluation = phase_contract.get("evaluation_contract", {})
    output = phase_contract.get("output_contract", {})
    frozen = phase_contract.get("frozen_unchanged_contract", {})
    target = phase_contract.get("target_contract", {})
    safety = state.get("safety", {})
    config_path = _resolve_project_path(root, "configs/v82_low_turnover_rank_dataset.yaml")
    config = _load_mapping(config_path) if config_path.is_file() else {}
    if (
        source.get("provider") != "binance_public_data_vision"
        or source.get("frequency") != "1d"
        or source.get("month_start") != "2018-01"
        or source.get("month_end") != "2026-06"
        or source.get("checksum_algorithm") != "sha256"
        or len(source.get("symbols", [])) != 30
        or set(source.get("symbols", []))
        & {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
        or feature.get("lookback_days") != 128
        or len(feature.get("columns", [])) != 8
        or feature.get("scaling") != "not_applied_in_v82"
        or label.get("formula") != "log(open[t+22] / open[t+1])"
        or label.get("horizon_intervals") != 21
        or label.get("maturity_days") != 22
        or roles.get("train", {}).get("signal_start") != "2018-05-09"
        or roles.get("train", {}).get("signal_end") != "2023-11-18"
        or roles.get("internal_validation", {}).get("signal_start")
        != "2024-01-01"
        or roles.get("internal_validation", {}).get("signal_end")
        != "2024-11-18"
        or evaluation.get("signal_start") != "2026-01-01"
        or evaluation.get("signal_end") != "2026-06-08"
        or evaluation.get("signal_dates") != 159
        or evaluation.get("final_outcome_maturity") != "2026-06-30"
        or evaluation.get("outcome_packet_status_after_v82") != "sealed"
        or evaluation.get("outcome_packet_unseals_during_v82") != 0
        or output.get("deterministic_fresh_replay_required") is not True
        or len(output.get("packet_files", [])) != 15
        or frozen.get("expected_total_parameters") != 10993
        or frozen.get("architecture_variant_count") != 1
        or frozen.get("structural_maximum_turnover") != 16.0
        or frozen.get("future_training_jobs") != 9
        or target.get("status") != "sealed"
        or target.get("target_assets_loaded") != []
        or state.get("target_assets", {}).get("status") != "sealed"
        or state.get("deployable_strategy") is not False
        or safety.get("v82_dataset_phase") is not True
        or safety.get("v82_exact_user_authorization_present") is not True
        or safety.get("v82_missing_imputation_allowed") is not False
        or safety.get(
            "v82_scaler_model_checkpoint_training_inference_prediction_position_allowed"
        )
        is not False
        or safety.get("v82_performance_metric_pnl_or_bootstrap_allowed") is not False
        or safety.get("v82_sealed_outcome_packet_unseal_allowed") is not False
        or safety.get("v82_target_assets_remain_sealed") is not True
        or safety.get("v82_v83_implementation_or_training_allowed") is not False
        or _sha256_file(config_path)
        != "9ccd35c23f087d813c421db2a50c3c7fd309ea752ae63e56b3b8581f8f30002d"
        or config.get("low_turnover_rank_dataset", {})
        .get("phase_contract", {})
        .get("file_sha256")
        != "939c55c760ab47e0157e2da04b3502aca9a003de3c5e91772c276b762977b378"
    ):
        raise ResearchStateError("V82 source, data, output, target, or safety drift")


def _validate_v61_harness_boundary(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
) -> None:
    del root
    expected_action = "authorize_v61_synthetic_decoupled_rank_state_harness_only"
    if (
        state.get("authorized_phase") != "v61"
        or state.get("authorized_next_action") != expected_action
        or state.get("active_family_status")
        != "specification_frozen_harness_authorized"
        or state.get("last_completed_phase") != "v60_decoupled_rank_state_specification"
        or experiment.get("phase") != "metadata_only_ex_ante_specification"
        or experiment.get("status") != "passed"
    ):
        raise ResearchStateError("V61 synthetic harness state drift")
    access = phase_contract["access_contract"]
    expected_inputs = {
        "artifacts/v60_decoupled_rank_state_spec/specification.json",
        "artifacts/v60_decoupled_rank_state_spec/blueprint.json",
        "artifacts/v60_decoupled_rank_state_spec/audit.json",
        "artifacts/v60_decoupled_rank_state_spec/result.json",
        "artifacts/v60_decoupled_rank_state_spec/artifact_manifest.json",
        "artifacts/v60_decoupled_rank_state_spec/completion_receipt.json",
    }
    if set(access.get("allowed_inputs", [])) != expected_inputs:
        raise ResearchStateError("V61 synthetic-only input allowlist drift")
    required_forbidden = {
        "real_data_or_parquet_access",
        "prior_checkpoint_access_or_reuse",
        "real_training_or_inference",
        "real_prediction_performance_metric_or_pnl",
        "outcome_source_read",
        "target_asset_access",
        "v62_implementation_or_data_build",
    }
    if not required_forbidden.issubset(
        set(access.get("forbidden_capabilities", []))
    ):
        raise ResearchStateError("V61 synthetic-only forbidden boundary drift")
    if (
        state.get("deployable_strategy") is not False
        or state.get("target_assets", {}).get("status") != "sealed"
        or phase_contract.get("target_contract", {}).get("status") != "sealed"
    ):
        raise ResearchStateError("V61 target/deployment boundary drift")


def _validate_v62_dataset_boundary(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
) -> None:
    expected_action = "authorize_v62_non_target_decoupled_rank_state_dataset_only"
    if (
        state.get("authorized_phase") != "v62"
        or state.get("authorized_next_action") != expected_action
        or state.get("active_family_status")
        != "synthetic_harness_passed_dataset_authorized"
        or state.get("last_completed_phase") != "v61_decoupled_rank_state_harness"
        or experiment.get("phase") != "synthetic_harness"
        or experiment.get("status") != "passed"
    ):
        raise ResearchStateError("V62 dataset-only state drift")

    access = phase_contract["access_contract"]
    expected_inputs = {
        "artifacts/v61_decoupled_rank_state_harness/result.json",
        "artifacts/v61_decoupled_rank_state_harness/audit.json",
        "artifacts/v61_decoupled_rank_state_harness/artifact_manifest.json",
        "artifacts/v61_decoupled_rank_state_harness/completion_receipt.json",
        "artifacts/v60_decoupled_rank_state_spec/specification.json",
        "artifacts/v60_decoupled_rank_state_spec/blueprint.json",
        "artifacts/v32_selected_universe_dataset/result.json",
        "artifacts/v32_selected_universe_dataset/audit.json",
        "artifacts/v32_selected_universe_dataset/dataset_manifest.json",
        "artifacts/v32_selected_universe_dataset/feature_schema.json",
        "artifacts/v32_selected_universe_dataset/asset_folds.json",
        "artifacts/v32_selected_universe_dataset/triplet_catalog.json",
        "data/processed/selected_universe_panel_v32.parquet",
        "data/processed/selected_universe_sequence_index_v32.parquet",
    }
    input_contract = phase_contract.get("input_contract")
    if not isinstance(input_contract, dict):
        raise ResearchStateError("V62 dataset input contract is missing")
    expected_bindings = input_contract.get("expected_file_sha256_by_path")
    if (
        set(access.get("allowed_inputs", [])) != expected_inputs
        or not isinstance(expected_bindings, dict)
        or set(expected_bindings) != expected_inputs
        or not all(_is_sha256(digest) for digest in expected_bindings.values())
    ):
        raise ResearchStateError("V62 dataset input allowlist/hash binding drift")
    for path_text, expected_hash in expected_bindings.items():
        path = _resolve_project_path(root, path_text)
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise ResearchStateError(f"V62 dataset input drift: {path_text}")

    required_forbidden = {
        "target_asset_access",
        "universe_fold_or_triplet_reselection",
        "missing_row_imputation_or_repair",
        "scaler_fit",
        "model_instantiation",
        "optimizer_step_or_training",
        "checkpoint_access",
        "market_prediction",
        "performance_metric_or_pnl",
        "outcome_source_read",
        "v63_implementation_or_training",
    }
    if not required_forbidden.issubset(
        set(access.get("forbidden_capabilities", []))
    ):
        raise ResearchStateError("V62 dataset-only forbidden boundary drift")

    data_contract = phase_contract.get("data_contract")
    state_features = (
        data_contract.get("derived_state_features", {})
        if isinstance(data_contract, dict)
        else {}
    )
    action_return = (
        data_contract.get("action_return", {})
        if isinstance(data_contract, dict)
        else {}
    )
    if (
        action_return.get("formula") != "log(open[t+2] / open[t+1])"
        or action_return.get("maturity_offset_days") != 2
        or state_features.get("order")
        != [
            "cross_asset_mean_of_nine_inputs",
            "cross_asset_population_std_of_nine_inputs",
        ]
        or state_features.get("count") != 18
    ):
        raise ResearchStateError("V62 frozen label/state-feature contract drift")

    output_contract = phase_contract.get("output_contract")
    if not isinstance(output_contract, dict) or output_contract != {
        "labels_path": "data/processed/decoupled_rank_state_labels_v62.parquet",
        "sequence_roles_path": (
            "data/processed/decoupled_rank_state_sequence_roles_v62.parquet"
        ),
        "target_assets_written": [],
    }:
        raise ResearchStateError("V62 dataset output contract drift")
    if (
        state.get("deployable_strategy") is not False
        or state.get("target_assets", {}).get("status") != "sealed"
        or phase_contract.get("target_contract", {}).get("status") != "sealed"
        or phase_contract.get("target_contract", {}).get("target_assets_loaded") != []
    ):
        raise ResearchStateError("V62 target/deployment boundary drift")


def _validate_v63_training_boundary(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
) -> None:
    expected_action = (
        "authorize_v63_frozen_non_target_decoupled_rank_state_training_only"
    )
    if (
        state.get("authorized_phase") != "v63"
        or state.get("authorized_next_action") != expected_action
        or state.get("active_family_status") != "dataset_passed_training_authorized"
        or state.get("last_completed_phase")
        != "v62_non_target_decoupled_rank_state_dataset"
        or experiment.get("phase") != "non_target_decoupled_rank_state_dataset"
        or experiment.get("status") != "passed"
    ):
        raise ResearchStateError("V63 training-only state drift")

    access = phase_contract["access_contract"]
    expected_inputs = {
        "artifacts/v60_decoupled_rank_state_spec/specification.json",
        "artifacts/v60_decoupled_rank_state_spec/blueprint.json",
        "artifacts/v61_decoupled_rank_state_harness/result.json",
        "artifacts/v61_decoupled_rank_state_harness/audit.json",
        "artifacts/v61_decoupled_rank_state_harness/harness_spec.json",
        "artifacts/v62_non_target_decoupled_rank_state_dataset/result.json",
        "artifacts/v62_non_target_decoupled_rank_state_dataset/audit.json",
        "artifacts/v62_non_target_decoupled_rank_state_dataset/dataset_spec.json",
        "artifacts/v62_non_target_decoupled_rank_state_dataset/dataset_manifest.json",
        "artifacts/v62_non_target_decoupled_rank_state_dataset/label_schema.json",
        "artifacts/v62_non_target_decoupled_rank_state_dataset/source_receipt.json",
        "artifacts/v62_non_target_decoupled_rank_state_dataset/completion_receipt.json",
        "artifacts/v62_non_target_decoupled_rank_state_dataset/artifact_manifest.json",
        "artifacts/v62_non_target_decoupled_rank_state_dataset/data_access.json",
        "artifacts/v62_non_target_decoupled_rank_state_dataset/triplet_derivation_smoke.json",
        "artifacts/v32_selected_universe_dataset/feature_schema.json",
        "artifacts/v32_selected_universe_dataset/asset_folds.json",
        "artifacts/v32_selected_universe_dataset/triplet_catalog.json",
        "data/processed/selected_universe_panel_v32.parquet",
        "data/processed/decoupled_rank_state_labels_v62.parquet",
        "data/processed/decoupled_rank_state_sequence_roles_v62.parquet",
    }
    input_contract = phase_contract.get("input_contract")
    if not isinstance(input_contract, dict):
        raise ResearchStateError("V63 training input contract is missing")
    expected_bindings = input_contract.get("expected_file_sha256_by_path")
    if (
        set(access.get("allowed_inputs", [])) != expected_inputs
        or not isinstance(expected_bindings, dict)
        or set(expected_bindings) != expected_inputs
        or not all(_is_sha256(digest) for digest in expected_bindings.values())
    ):
        raise ResearchStateError("V63 training input allowlist/hash binding drift")
    for path_text, expected_hash in expected_bindings.items():
        path = _resolve_project_path(root, path_text)
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise ResearchStateError(f"V63 training input drift: {path_text}")

    required_forbidden = {
        "target_asset_access",
        "heldout_fold_asset_scaler_pretraining_training_or_validation",
        "post_2025_signal_or_label_value_access",
        "previous_family_checkpoint_or_representation_reuse",
        "shared_ranker_gate_parameter_optimizer_or_gradient_path",
        "hyperparameter_architecture_seed_fold_or_grid_change",
        "prediction_position_policy_or_portfolio_generation",
        "performance_metric_or_pnl",
        "historical_outcome_source_reread",
        "v64_implementation_or_evaluation",
    }
    if not required_forbidden.issubset(
        set(access.get("forbidden_capabilities", []))
    ):
        raise ResearchStateError("V63 training-only forbidden boundary drift")

    model = phase_contract.get("model_and_objective_contract")
    grid = phase_contract.get("grid_optimizer_and_runtime_contract")
    operator = phase_contract.get("operator_preflight_contract")
    runtime = phase_contract.get("runtime_contract")
    checkpoint = phase_contract.get("checkpoint_contract")
    artifacts = phase_contract.get("artifact_contract")
    access_ledger = phase_contract.get("data_access_ledger_contract")
    enforcement = phase_contract.get("operator_enforcement_contract")
    if (
        not isinstance(model, dict)
        or model.get("ranker", {}).get("parameter_count") != 1_231_634
        or model.get("state_gate", {}).get("parameter_count") != 27_489
        or model.get("total_parameter_count") != 1_259_123
        or model.get("shared_parameters") is not False
        or model.get("combined_scalar_loss") is not False
        or not isinstance(grid, dict)
        or grid.get("folds") != [1, 2, 3]
        or grid.get("seeds") != [42, 7, 123]
        or grid.get("expected_jobs") != 9
        or grid.get("ranker_and_gate_optimizers_independent") is not True
        or grid.get("runtime", {}).get("device") != "mps"
        or grid.get("runtime", {}).get("pytorch_mps_fallback") != "disabled"
        or not isinstance(operator, dict)
        or operator.get("clean_committed_git_required") is not True
        or operator.get("smoke_must_pass_before_full_training") is not True
        or not isinstance(runtime, dict)
        or runtime.get("device") != "mps"
        or runtime.get("pytorch_enable_mps_fallback") != "0"
        or runtime.get("backup_policy", {}).get("mode") != "owner_waiver"
        or not isinstance(checkpoint, dict)
        or checkpoint.get("cross_job_resume_allowed") is not False
        or checkpoint.get("retain_every_final_checkpoint") is not True
        or not isinstance(artifacts, dict)
        or artifacts.get(
            "checkpoint_manifest_requires_all_nine_file_and_semantic_hashes"
        )
        is not True
        or not isinstance(access_ledger, dict)
        or access_ledger.get("target_assets_loaded") != []
        or access_ledger.get("performance_metrics_computed") is not False
        or not isinstance(enforcement, dict)
        or enforcement.get("operation_order")
        != ["doctor", "preflight", "smoke", "full", "verify", "replay"]
    ):
        raise ResearchStateError("V63 frozen model/grid/operator contract drift")
    if (
        state.get("deployable_strategy") is not False
        or state.get("target_assets", {}).get("status") != "sealed"
        or phase_contract.get("target_contract", {}).get("status") != "sealed"
        or phase_contract.get("target_contract", {}).get("target_assets_loaded") != []
    ):
        raise ResearchStateError("V63 target/deployment boundary drift")


def _validate_v64_evaluation_boundary(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
) -> None:
    """Validate the post-V63 adaptive-development evaluation-only gate."""

    expected_action = (
        "authorize_v64_frozen_adaptive_development_evaluation_only"
    )
    if (
        state.get("authorized_phase") != "v64"
        or state.get("authorized_next_action") != expected_action
        or state.get("active_family_status")
        != "trained_adaptive_development_evaluation_authorized"
        or state.get("last_completed_phase")
        != "v63_frozen_non_target_decoupled_rank_state_training"
        or experiment.get("phase")
        != "frozen_non_target_decoupled_rank_state_training"
        or experiment.get("status") != "passed"
        or experiment.get("v63_summary", {}).get("checkpoint_count") != 9
        or experiment.get("v63_summary", {}).get("zero_step_replay_passed")
        is not True
    ):
        raise ResearchStateError("V64 evaluation-only state drift")

    access = phase_contract.get("access_contract")
    inputs = phase_contract.get("input_contract", {}).get(
        "expected_file_sha256_by_path"
    )
    if (
        not isinstance(access, dict)
        or not isinstance(inputs, dict)
        or set(access.get("allowed_inputs", [])) != set(inputs)
        or len(inputs) != 30
        or not all(_is_sha256(value) for value in inputs.values())
    ):
        raise ResearchStateError("V64 evaluation input binding drift")
    for path_text, expected_hash in inputs.items():
        path = _resolve_project_path(root, path_text)
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise ResearchStateError(f"V64 evaluation input drift: {path_text}")

    manifest_path = _resolve_project_path(
        root,
        "artifacts/v63_decoupled_rank_state_training/checkpoint_manifest.json",
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResearchStateError("V64 checkpoint manifest is invalid") from exc
    jobs = manifest.get("jobs", []) if isinstance(manifest, dict) else []
    expected_jobs = [
        f"{fold}|{seed}"
        for fold in (1, 2, 3)
        for seed in (42, 7, 123)
    ]
    if (
        [job.get("job_id") for job in jobs] != expected_jobs
        or any(job.get("status") != "completed" for job in jobs)
        or any(
            inputs.get(job.get("path")) != job.get("file_sha256")
            for job in jobs
        )
        or manifest.get("active_resume_artifacts") != []
        or manifest.get("orphan_resume_artifacts") != []
    ):
        raise ResearchStateError("V64 nine-checkpoint manifest binding drift")

    evaluation = phase_contract.get("evaluation_contract")
    prepare_spec = evaluation.get("registered_prepare_spec", {}) if isinstance(evaluation, dict) else {}
    config_path = _resolve_project_path(
        root,
        prepare_spec.get("config_path", "configs/v64_decoupled_rank_state_evaluation.yaml"),
    )
    if (
        phase_contract.get("evidence_tier")
        != "adaptive_development_only_not_confirmation"
        or not isinstance(evaluation, dict)
        or evaluation.get("role") != "consumed_adaptive_development_only"
        or evaluation.get("clean_holdout_claim") is not False
        or evaluation.get("window")
        != {
            "signal_start": "2025-01-01",
            "signal_end": "2025-12-23",
            "label": "open_t_plus_1_to_open_t_plus_2_log_return",
        }
        or evaluation.get("fold_scope", {}).get("seeds") != [42, 7, 123]
        or evaluation.get("fold_scope", {}).get("inference_assets")
        != "exact_v32_test_symbols_for_same_fold"
        or evaluation.get("one_shot_outcome_open", {}).get(
            "maximum_open_count"
        )
        != 1
        or evaluation.get("decision_semantics", {}).get(
            "v64_can_establish_deployability"
        )
        is not False
        or evaluation.get("decision_semantics", {}).get(
            "v64_can_authorize_target_assets"
        )
        is not False
        or evaluation.get("registered_prepare_spec", {}).get("seeds")
        != [42, 7, 123]
        or not config_path.is_file()
        or _sha256_file(config_path)
        != prepare_spec.get("config_file_sha256")
        or evaluation.get("registered_prepare_spec", {}).get(
            "outcome_projection_forbidden_during_prepare"
        )
        is not True
        or evaluation.get("registered_accounting", {}).get("net_return")
        != "gross_minus_turnover_cost"
        or evaluation.get("registered_bootstrap", {}).get("paths") != 10000
        or evaluation.get("registered_bootstrap", {}).get(
            "block_lengths_days"
        )
        != [7, 21, 63]
        or len(evaluation.get("registered_outcome_blind_gates", [])) != 12
        or evaluation.get("one_shot_lifecycle", {}).get(
            "explicit_user_authorization_after_prepare_required"
        )
        is not True
        or evaluation.get("one_shot_lifecycle", {}).get(
            "generic_continue_is_not_unseal_authorization"
        )
        is not True
    ):
        raise ResearchStateError("V64 adaptive evidence contract drift")

    required_forbidden = {
        "target_asset_access",
        "training_scaler_fit_optimizer_step_or_checkpoint_write",
        "model_seed_fold_architecture_hyperparameter_or_policy_change",
        "regenerate_predictions_after_outcome_access",
        "claim_clean_holdout_prospective_confirmation_or_deployability",
        "authorize_target_evaluation_from_v64_metrics",
        "v65_implementation_or_evaluation",
    }
    if not required_forbidden.issubset(
        set(access.get("forbidden_capabilities", []))
    ):
        raise ResearchStateError("V64 forbidden evaluation boundary drift")
    if (
        state.get("deployable_strategy") is not False
        or state.get("target_assets", {}).get("status") != "sealed"
        or phase_contract.get("target_contract", {}).get("status") != "sealed"
        or phase_contract.get("target_contract", {}).get("target_assets_loaded")
        != []
        or phase_contract.get("stop_condition")
        != "stop_after_hash_valid_outcome_blind_prepare_before_explicit_unseal_authorization"
    ):
        raise ResearchStateError("V64 target/deployment/stop boundary drift")


def _validate_v64_unseal_boundary(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
) -> None:
    """Validate the hash-bound, exactly-once V64 unseal boundary."""

    expected_action = (
        "execute_v64_exactly_one_registered_non_target_outcome_unseal_and_complete_evaluation"
    )
    expected_command = (
        "PYTHONPATH=src python3 -m tlm decoupled-rank-state-evaluation-unseal "
        "--config configs/v64_decoupled_rank_state_evaluation.yaml"
    )
    if (
        state.get("authorized_phase") != "v64"
        or state.get("authorized_next_action") != expected_action
        or state.get("authorized_command") != expected_command
        or state.get("active_family_status")
        != "adaptive_development_evaluation_prepared_exact_unseal_authorized"
        or state.get("last_completed_phase") != "v64_outcome_blind_prepare"
        or experiment.get("phase")
        != "outcome_blind_adaptive_development_evaluation_prepare"
        or experiment.get("status") != "passed_outcomes_sealed"
        or experiment.get("authorized_next_action") != expected_action
    ):
        raise ResearchStateError("V64 authorized-unseal state drift")

    expected_prepare_hashes = {
        "evaluation_spec": "f6fbf371b5e33efdaaf0b0d1622acefce4938efe1e22d63ffa6e086e0a45d134",
        "prepare_receipt": "18429f83790cd16b57dbb4208c4b211c3f2600aed1adaadaef722ca4bded4e4e",
        "one_shot_packet": "a18379581eabf694338837f5f7bd5c00faa73ac5a1be92ca17ee1e23eb94d5f3",
        "context_predictions": "e9668f93eef3bcfe171b94d3f13ad7fe8bed00f25f18ef81ca830ca6f8229782",
        "asset_predictions": "8297d9eb214280eecc69c6805671b28f72db3f821293f802088609a33c432f2b",
        "positions": "7722c68e522fba1a3bb708b803d08230677920998255fc3c37d697c1096cd88f",
        "behavior_gates": "4f63e331f40a9d0f5d267dce771ec809e8aaa7cf0d70a51f90b2a28176c465e4",
        "data_access_receipt": "7ba1f70a36de7043985c4fd3407925c662a85eeee939b45d7a7fef942e8899c6",
    }
    prepare = phase_contract.get("prepare_packet")
    if not isinstance(prepare, dict):
        raise ResearchStateError("V64 prepare packet contract is missing")
    for name, expected_hash in expected_prepare_hashes.items():
        reference = prepare.get(name)
        if (
            not isinstance(reference, dict)
            or reference.get("file_sha256") != expected_hash
        ):
            raise ResearchStateError(f"V64 prepare {name} binding drift")
        path = _resolve_project_path(root, reference.get("path", ""))
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise ResearchStateError(f"V64 prepare {name} file drift")
    if (
        prepare.get("registered_sha256")
        != "cfa1989008c384e31993d2f42b6bcd371a8b177aeebc78d2f75457d6c773fcbe"
        or prepare["context_predictions"].get("rows") != 875_232
        or prepare["asset_predictions"].get("rows") != 9_794
        or prepare["positions"].get("rows") != 32_130
    ):
        raise ResearchStateError("V64 frozen prepare summary drift")

    authorization = phase_contract.get("explicit_user_authorization")
    payload = authorization.get("payload") if isinstance(authorization, dict) else None
    if (
        not isinstance(payload, dict)
        or authorization.get("canonical_sha256")
        != "f3ebab32fa0931a3674dbced5e8d76fbdd15086d16b35b77c2329861c6688b63"
        or _canonical_sha256(payload) != authorization.get("canonical_sha256")
        or payload.get("evaluation_spec_sha256")
        != expected_prepare_hashes["evaluation_spec"]
        or payload.get("prepare_receipt_sha256")
        != expected_prepare_hashes["prepare_receipt"]
        or payload.get("one_shot_packet_sha256")
        != expected_prepare_hashes["one_shot_packet"]
        or payload.get("maximum_unseal_count") != 1
        or payload.get("no_regeneration_retuning_or_policy_cost_change") is not True
        or payload.get("target_assets_status") != "sealed"
    ):
        raise ResearchStateError("V64 explicit user authorization drift")

    outcome = phase_contract.get("outcome_access_contract")
    if (
        not isinstance(outcome, dict)
        or outcome.get("source")
        != "data/processed/decoupled_rank_state_labels_v62.parquet"
        or outcome.get("source_file_sha256")
        != "6ea78e634e2b444a7c83767754c772610a65f4b13a0c6473ae51472918f37ece"
        or outcome.get("allowed_columns")
        != [
            "date",
            "symbol",
            "target_h1_maturity_date",
            "target_h1_open_to_open_log_return",
            "h1_label_complete",
        ]
        or outcome.get("exact_key_count") != 9_794
        or outcome.get("exact_key_sha256")
        != "d1e34726eb63ddeced8ab250181a3fa53671ff5a322773719c39b9379eb3efda"
        or outcome.get("maximum_source_reads") != 1
        or outcome.get("maximum_maturity") != "2025-12-25"
    ):
        raise ResearchStateError("V64 exact outcome access contract drift")
    source_path = _resolve_project_path(root, outcome["source"])
    if not source_path.is_file() or _sha256_file(source_path) != outcome["source_file_sha256"]:
        raise ResearchStateError("V64 registered outcome source drift")

    one_shot = phase_contract.get("one_shot_contract", {})
    evaluation = phase_contract.get("evaluation_contract", {})
    if (
        one_shot.get("authorization_receipt_written_atomically_before_source_read")
        is not True
        or one_shot.get("maximum_unseal_count") != 1
        or one_shot.get("source_outcome_reads_first_execution") != 1
        or one_shot.get("source_outcome_reads_replay") != 0
        or one_shot.get("prediction_or_position_regeneration") is not False
        or evaluation.get("costs_bps") != [10, 20, 30]
        or evaluation.get("bootstrap_paths") != 10_000
        or evaluation.get("bootstrap_block_lengths") != [7, 21, 63]
        or evaluation.get("aggregate_rescue_allowed") is not False
    ):
        raise ResearchStateError("V64 one-shot science or lifecycle drift")

    output_dir = _resolve_project_path(root, phase_contract["access_contract"]["output_dir"])
    envelope = [
        output_dir / "unseal_authorization_receipt.json",
        output_dir / "outcome_packet.parquet",
        output_dir / "outcome_receipt.json",
        output_dir / "completion_receipt.json",
        output_dir / "one_shot_complete_packet.json",
    ]
    if any(path.exists() for path in envelope):
        if not all(path.is_file() for path in envelope):
            raise ResearchStateError("V64 one-shot outcome envelope is partial")
        auth = _load_mapping(envelope[0])
        receipt = _load_mapping(envelope[2])
        completion = _load_mapping(envelope[3])
        if (
            auth.get("unseal_count") != 1
            or auth.get("authorization_payload_sha256")
            != authorization["canonical_sha256"]
            or auth.get("target_assets_loaded") != []
            or receipt.get("unseal_count") != 1
            or receipt.get("source_outcome_reads") != 1
            or receipt.get("outcome_packet_sha256") != _sha256_file(envelope[1])
            or receipt.get("authorization_receipt_sha256") != _sha256_file(envelope[0])
            or receipt.get("target_assets_loaded") != []
            or completion.get("unseal_count") != 1
            or completion.get("source_outcome_reads") != 1
            or completion.get("target_assets_status") != "sealed"
        ):
            raise ResearchStateError("V64 completed one-shot envelope drift")

    required_forbidden = {
        "target_asset_access_inference_prediction_or_pnl",
        "second_outcome_unseal_or_source_outcome_reread",
        "prediction_or_position_regeneration_after_prepare",
        "hyperparameter_policy_control_cost_gate_or_bootstrap_change",
        "aggregate_result_rescue_of_any_failed_mandatory_cell",
        "v65_implementation_or_target_evaluation",
    }
    if not required_forbidden.issubset(
        set(phase_contract["access_contract"].get("forbidden_capabilities", []))
    ):
        raise ResearchStateError("V64 authorized-unseal forbidden boundary drift")
    if (
        state.get("deployable_strategy") is not False
        or state.get("target_assets", {}).get("status") != "sealed"
        or phase_contract.get("target_contract", {}).get("status") != "sealed"
        or phase_contract.get("target_contract", {}).get("target_assets_loaded") != []
    ):
        raise ResearchStateError("V64 authorized-unseal target boundary drift")


def _validate_v64_terminal_boundary(
    root: Path,
    state: dict[str, Any],
    experiment: dict[str, Any],
    phase_contract: dict[str, Any],
) -> None:
    """Validate a terminal V64 pass-or-retire registration and source-free replay."""

    output = _resolve_project_path(
        root, "artifacts/v64_decoupled_rank_state_evaluation"
    )
    required = {
        "evaluation_result": output / "evaluation_result.json",
        "evaluation_audit": output / "evaluation_audit.json",
        "outcome_receipt": output / "outcome_receipt.json",
        "completion_receipt": output / "completion_receipt.json",
        "complete_packet": output / "one_shot_complete_packet.json",
        "replay": output / "replay.json",
        "replay_packet": output / "one_shot_replay_packet.json",
    }
    if not all(path.is_file() for path in required.values()):
        raise ResearchStateError("V64 terminal artifact set is incomplete")
    values = {name: _load_mapping(path) for name, path in required.items()}
    result = values["evaluation_result"]
    decision = result.get("decision")
    allowed = {
        "authorize_future_prospective_non_target_specification_only",
        "retire_family_without_target_evaluation_or_retuning",
    }
    if (
        decision not in allowed
        or experiment.get("decision") != decision
        or experiment.get("authorized_next_action") != decision
        or state.get("authorized_next_action") != decision
        or phase_contract.get("authorized_next_action") != decision
        or result.get("unseal_count") != 1
        or result.get("source_outcome_reads") != 1
        or result.get("retuning_performed") is not False
        or result.get("prediction_or_position_regeneration") is not False
        or result.get("target_assets_loaded") != []
        or result.get("deployable") is not False
    ):
        raise ResearchStateError("V64 terminal decision/result drift")
    receipt = values["outcome_receipt"]
    completion = values["completion_receipt"]
    replay = values["replay"]
    if (
        receipt.get("unseal_count") != 1
        or receipt.get("source_outcome_reads") != 1
        or completion.get("unseal_count") != 1
        or completion.get("source_outcome_reads") != 1
        or replay.get("new_unseal_receipts") != 0
        or replay.get("source_outcome_rows_read") != 0
        or replay.get("result_hashes_match") is not True
        or replay.get("target_assets_loaded") != []
        or values["complete_packet"].get("phase") != "complete"
        or values["replay_packet"].get("phase") != "replay"
        or values["replay_packet"].get("replay", {}).get(
            "source_outcome_rows_read"
        )
        != 0
    ):
        raise ResearchStateError("V64 terminal one-shot/replay drift")
    references = phase_contract.get("terminal_artifacts", {})
    for name, path in required.items():
        reference = references.get(name)
        if (
            not isinstance(reference, dict)
            or reference.get("path") != path.relative_to(root).as_posix()
            or reference.get("file_sha256") != _sha256_file(path)
        ):
            raise ResearchStateError(f"V64 terminal {name} reference drift")
    expected_status = "retired" if decision.startswith("retire_") else "evaluated_not_deployable"
    if (
        state.get("active_family_status") != expected_status
        or state.get("last_completed_phase") != "v64_adaptive_development_evaluation"
        or experiment.get("phase") != "adaptive_development_evaluation"
        or experiment.get("status") != expected_status
        or phase_contract.get("target_contract", {}).get("status") != "sealed"
        or state.get("target_assets", {}).get("status") != "sealed"
        or state.get("deployable_strategy") is not False
    ):
        raise ResearchStateError("V64 terminal state/target boundary drift")


def _parse_false_environment_flag(name: str) -> bool:
    """Return whether an environment flag is explicitly enabled."""

    value = os.environ.get(name, "0").strip().lower()
    return value not in {"", "0", "false", "no", "off"}


def _training_contract(
    root: Path, status: dict[str, Any]
) -> tuple[dict[str, Any], Path]:
    phase_path_raw = status.get("phase_contract_path")
    if isinstance(phase_path_raw, str) and phase_path_raw:
        phase_path = _resolve_project_path(root, phase_path_raw)
        return _load_mapping(phase_path), phase_path
    experiment_path = _resolve_project_path(root, status["experiment_path"])
    experiment = _load_mapping(experiment_path)
    candidate = experiment.get(f"{status['authorized_phase']}_contract", {})
    return (candidate if isinstance(candidate, dict) else {}), experiment_path


def _training_lock_path(
    root: Path, phase: str, contract: dict[str, Any]
) -> Path:
    runtime_contract = contract.get("runtime_contract", {})
    configured = (
        runtime_contract.get("process_lock")
        if isinstance(runtime_contract, dict)
        else None
    ) or contract.get("process_lock_path")
    relative = (
        configured
        if isinstance(configured, str) and configured
        else f"data/checkpoints/.{phase}_state_conditioned_multi_horizon_training.lock"
    )
    return _resolve_project_path(root, relative)


def _inspect_training_lock(path: Path) -> dict[str, Any]:
    """Inspect an advisory lock without treating a stale lock file as active."""

    if not path.exists():
        return {
            "path": str(path),
            "available": True,
            "active_job_count": 0,
            "metadata": None,
        }
    metadata: Any = None
    with path.open("a+", encoding="utf-8") as handle:
        handle.seek(0)
        raw = handle.read().strip()
        if raw:
            try:
                metadata = json.loads(raw)
            except json.JSONDecodeError:
                metadata = {"unparseable": True}
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {
                "path": str(path),
                "available": False,
                "active_job_count": 1,
                "metadata": metadata,
            }
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return {
        "path": str(path),
        "available": True,
        "active_job_count": 0,
        "metadata": metadata,
    }


def _inside_external_root(root: Path, relative: object, name: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ResearchStateError(f"{name} must be a non-empty relative path")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ResearchStateError(f"{name} escapes backup root: {relative}") from exc
    return candidate


def _validate_backup_receipt(
    root: Path,
    phase: str,
    contract: dict[str, Any],
    git_head: str,
) -> dict[str, Any]:
    safety = contract.get("safety", {})
    runtime_contract = contract.get("runtime_contract", {})
    if isinstance(runtime_contract, dict) and runtime_contract:
        safety = runtime_contract
    required = bool(safety.get("external_backup_receipt_required", False))
    configured = safety.get("external_backup_receipt")
    access_contract = contract.get("access_contract", {})
    allowed_inputs = (
        access_contract.get("allowed_inputs", [])
        if isinstance(access_contract, dict) and "allowed_inputs" in access_contract
        else contract.get("allowed_inputs", [])
    )
    result: dict[str, Any] = {
        "mode": "external",
        "required": required,
        "passed": not required,
        "receipt_path": configured,
        "objects_required": len(allowed_inputs),
        "objects_verified": 0,
        "code_backup_verified": False,
        "different_device": False,
        "errors": [],
    }
    backup_policy = safety.get("backup_policy")
    if isinstance(backup_policy, dict):
        mode = backup_policy.get("mode")
        if mode != "owner_waiver":
            result["errors"].append(f"unsupported backup policy mode: {mode}")
            result["passed"] = False
            return result
        result["mode"] = "owner_waiver"
        result["passed"] = False
        try:
            if required:
                raise ResearchStateError(
                    "owner waiver cannot coexist with a required external backup"
                )
            waiver_ref = backup_policy.get("waiver")
            if not isinstance(waiver_ref, dict):
                raise ResearchStateError("owner waiver reference is missing")
            waiver_relative = waiver_ref.get("path")
            waiver_hash = waiver_ref.get("file_sha256")
            if not isinstance(waiver_relative, str) or not waiver_relative:
                raise ResearchStateError("owner waiver path is missing")
            if not isinstance(waiver_hash, str) or len(waiver_hash) != 64:
                raise ResearchStateError("owner waiver hash is invalid")
            waiver_path = _resolve_project_path(root, waiver_relative)
            if not waiver_path.is_file() or _sha256_file(waiver_path) != waiver_hash:
                raise ResearchStateError("owner waiver file/hash drift")
            waiver = _load_mapping(waiver_path)
            expected_scopes = [
                "external_input_copy",
                "external_code_copy",
                "external_checkpoint_copy",
            ]
            if backup_policy.get("waived_safeguards") != expected_scopes:
                raise ResearchStateError("owner waiver safeguard scope drift")
            parent = contract.get("parent_experiment")
            if not isinstance(parent, dict):
                raise ResearchStateError("owner waiver parent experiment is missing")
            if phase == "v58":
                supersedes = contract.get("supersedes")
                operational_waiver = contract.get("operational_waiver")
                if contract.get("revision") != "v058r1":
                    raise ResearchStateError(
                        "owner waiver is valid only for the registered V58r1 revision"
                    )
                if not isinstance(supersedes, dict) or not isinstance(
                    operational_waiver, dict
                ):
                    raise ResearchStateError(
                        "V58 owner waiver contract bindings are missing"
                    )
                if operational_waiver != waiver_ref:
                    raise ResearchStateError("owner waiver contract references differ")
                base_path_raw = supersedes.get("path")
                base_hash = supersedes.get("file_sha256")
                if (
                    supersedes
                    != {
                        **V58_BASE_PHASE_CONTRACT,
                        "allowed_change_scope": "external_storage_redundancy_only",
                    }
                    or operational_waiver != V58_OWNER_STORAGE_WAIVER
                ):
                    raise ResearchStateError(
                        "owner waiver supersession binding is invalid"
                    )
                base_path = _resolve_project_path(root, base_path_raw)
                if not base_path.is_file() or _sha256_file(base_path) != base_hash:
                    raise ResearchStateError("base V58 contract file/hash drift")
                if (
                    waiver.get("waiver_id")
                    != "v058r1_external_backup_owner_waiver"
                    or waiver.get("base_phase_contract")
                    != {"path": base_path_raw, "file_sha256": base_hash}
                ):
                    raise ResearchStateError("V58 owner waiver content drift")
            elif phase == "v63":
                if (
                    contract.get("stage_revision") != "v063_training_r1"
                    or waiver.get("waiver_id")
                    != "v063_external_backup_owner_waiver"
                    or not isinstance(waiver.get("authorization_context"), str)
                    or not waiver.get("authorization_context")
                ):
                    raise ResearchStateError("V63 owner waiver content drift")
            elif phase == "v68":
                if (
                    contract.get("stage_revision")
                    != "v068_frozen_non_target_v64_r2_gate_training_r4"
                    or waiver.get("waiver_id")
                    != "v068_external_backup_owner_waiver"
                    or not isinstance(waiver.get("authorization_context"), str)
                    or not waiver.get("authorization_context")
                ):
                    raise ResearchStateError("V68 owner waiver content drift")
            elif phase == "v77":
                if (
                    contract.get("stage_revision")
                    != "v077_frozen_non_target_persistent_duration_training_r3"
                    or waiver.get("waiver_id")
                    != "v077_external_backup_owner_waiver"
                    or not isinstance(waiver.get("authorization_context"), str)
                    or not waiver.get("authorization_context")
                ):
                    raise ResearchStateError("V77 owner waiver content drift")
            elif phase == "v83":
                if (
                    contract.get("stage_revision")
                    != "v083_frozen_non_target_low_turnover_rank_training_r2"
                    or waiver.get("waiver_id")
                    != "v083_external_backup_owner_waiver"
                    or not isinstance(waiver.get("authorization_context"), str)
                    or not waiver.get("authorization_context")
                ):
                    raise ResearchStateError("V83 owner waiver content drift")
            else:
                raise ResearchStateError(
                    f"owner waiver is not registered for phase {phase}"
                )
            if (
                waiver.get("schema_version") != "tlm-owner-storage-waiver/v1"
                or waiver.get("phase") != phase
                or waiver.get("family_id") != contract.get("family_id")
                or waiver.get("accepted_by") != "repository_owner"
                or waiver.get("risk_acceptance") is not True
                or waiver.get("waived_safeguards") != expected_scopes
                or waiver.get("parent_experiment") != parent
            ):
                raise ResearchStateError("owner storage waiver content drift")
            result.update(
                {
                    "passed": True,
                    "waiver_verified": True,
                    "waiver_path": waiver_relative,
                    "waiver_sha256": waiver_hash,
                    "waived_safeguards": expected_scopes,
                }
            )
        except (OSError, TypeError, ValueError, ResearchStateError) as exc:
            result["waiver_verified"] = False
            result["errors"].append(str(exc))
        return result
    if phase == "v58" and not required:
        result["passed"] = False
        result["errors"].append(
            "V58 storage policy must be required external backup or registered owner waiver"
        )
        return result
    if not required:
        return result
    if not isinstance(configured, str) or not configured:
        result["errors"].append("backup_receipt_path_missing")
        return result
    receipt_path = _resolve_project_path(root, configured)
    if not receipt_path.is_file():
        result["errors"].append("backup_receipt_missing")
        return result
    try:
        receipt = _load_mapping(receipt_path)
        if receipt.get("schema_version") != "tlm-external-backup-receipt/v1":
            raise ResearchStateError("unsupported backup receipt schema")
        if receipt.get("phase") != phase:
            raise ResearchStateError("backup receipt phase drift")
        if receipt.get("verified") is not True:
            raise ResearchStateError("backup receipt is not marked verified")
        backup_root_raw = receipt.get("backup_root")
        if not isinstance(backup_root_raw, str) or not Path(backup_root_raw).is_absolute():
            raise ResearchStateError("backup_root must be absolute")
        backup_root = Path(backup_root_raw).resolve()
        if not backup_root.is_dir():
            raise ResearchStateError("backup_root is not a directory")
        try:
            backup_root.relative_to(root)
        except ValueError:
            pass
        else:
            raise ResearchStateError("backup_root must be outside the repository")

        source_device = int(root.stat().st_dev)
        backup_device = int(backup_root.stat().st_dev)
        different_device = source_device != backup_device
        if receipt.get("different_device") is not True or not different_device:
            raise ResearchStateError("backup is not on a different filesystem device")
        if int(receipt.get("source_device", -1)) != source_device:
            raise ResearchStateError("backup source_device drift")
        if int(receipt.get("backup_device", -1)) != backup_device:
            raise ResearchStateError("backup backup_device drift")
        result["different_device"] = True

        if not isinstance(allowed_inputs, list) or not all(
            isinstance(item, str) and item for item in allowed_inputs
        ):
            raise ResearchStateError("training contract allowed_inputs is invalid")
        objects = receipt.get("objects")
        if not isinstance(objects, list):
            raise ResearchStateError("backup receipt objects must be an array")
        by_source: dict[str, dict[str, Any]] = {}
        for item in objects:
            if not isinstance(item, dict):
                raise ResearchStateError("backup object must be a mapping")
            source = item.get("source_path")
            if not isinstance(source, str) or source in by_source:
                raise ResearchStateError("backup source paths must be unique strings")
            by_source[source] = item
        if set(by_source) != set(allowed_inputs):
            raise ResearchStateError("backup object set differs from allowed inputs")
        for source in allowed_inputs:
            item = by_source[source]
            source_path = _resolve_project_path(root, source)
            backup_path = _inside_external_root(
                backup_root, item.get("backup_path"), "backup object path"
            )
            expected = item.get("sha256")
            size = item.get("size_bytes")
            if not isinstance(expected, str) or len(expected) != 64:
                raise ResearchStateError(f"invalid backup hash: {source}")
            if not source_path.is_file() or not backup_path.is_file():
                raise ResearchStateError(f"missing source or backup object: {source}")
            if not isinstance(size, int) or size < 0:
                raise ResearchStateError(f"invalid backup size: {source}")
            if source_path.stat().st_size != size or backup_path.stat().st_size != size:
                raise ResearchStateError(f"backup size drift: {source}")
            if _sha256_file(source_path) != expected or _sha256_file(backup_path) != expected:
                raise ResearchStateError(f"backup hash drift: {source}")
        result["objects_verified"] = len(allowed_inputs)

        code = receipt.get("code_backup")
        if not isinstance(code, dict):
            raise ResearchStateError("backup receipt lacks code_backup")
        if code.get("kind") not in {"git_bundle", "source_archive"}:
            raise ResearchStateError("unsupported code backup kind")
        if code.get("git_head") != git_head:
            raise ResearchStateError("code backup Git head drift")
        code_path = _inside_external_root(
            backup_root, code.get("backup_path"), "code backup path"
        )
        code_hash = code.get("sha256")
        code_size = code.get("size_bytes")
        if not code_path.is_file():
            raise ResearchStateError("code backup file is missing")
        if not isinstance(code_hash, str) or len(code_hash) != 64:
            raise ResearchStateError("invalid code backup hash")
        if not isinstance(code_size, int) or code_size != code_path.stat().st_size:
            raise ResearchStateError("code backup size drift")
        if _sha256_file(code_path) != code_hash:
            raise ResearchStateError("code backup hash drift")
        result["code_backup_verified"] = True
        result["passed"] = True
        result["receipt_sha256"] = _sha256_file(receipt_path)
        result["backup_root"] = str(backup_root)
    except (OSError, TypeError, ValueError, ResearchStateError) as exc:
        result["errors"].append(str(exc))
    return result


def validate_research_state(
    project_root: str | Path = ".",
    state_path: str | Path = "research/current.yaml",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    current_path = _resolve_project_path(root, str(state_path))
    if not current_path.is_file():
        raise ResearchStateError(f"Missing current research state: {current_path}")
    state = _load_mapping(current_path)
    required = {
        "schema_version",
        "current_experiment",
        "phase_contract",
        "active_family_id",
        "active_family_status",
        "authorized_next_action",
        "authorized_phase",
        "target_assets",
        "families",
        "forbidden_capabilities",
    }
    missing = sorted(required - set(state))
    if missing:
        raise ResearchStateError(f"Current state is missing keys: {missing}")
    if state["schema_version"] != 1:
        raise ResearchStateError("Unsupported current-state schema version")
    if state["target_assets"].get("status") != "sealed":
        raise ResearchStateError("Target assets must remain sealed in the current phase")

    experiment_path = _resolve_project_path(root, state["current_experiment"])
    if not experiment_path.is_file():
        raise ResearchStateError(f"Missing current experiment contract: {experiment_path}")
    experiment = _load_mapping(experiment_path)
    if experiment.get("family_id") != state["active_family_id"]:
        raise ResearchStateError("Current state and experiment family IDs differ")
    if experiment.get("authorized_next_action") != state["authorized_next_action"]:
        raise ResearchStateError("Current state and experiment authorization differ")

    artifact_checks: dict[str, bool] = {}
    phase_contract_path: Path | None = None
    phase_reference = state.get("phase_contract")
    if phase_reference is not None:
        if not isinstance(phase_reference, dict) or not {
            "path",
            "file_sha256",
        }.issubset(phase_reference):
            raise ResearchStateError("Current phase-contract reference is incomplete")
        phase_contract_path = _resolve_project_path(root, phase_reference["path"])
        phase_exists = phase_contract_path.is_file()
        artifact_checks["phase_contract_exists"] = phase_exists
        if phase_exists:
            phase_hash_matches = (
                _sha256_file(phase_contract_path) == phase_reference["file_sha256"]
            )
            artifact_checks["phase_contract_file_sha256_matches"] = phase_hash_matches
        if phase_exists and phase_hash_matches:
            phase_contract = _load_mapping(phase_contract_path)
            if phase_contract.get("phase") != state["authorized_phase"]:
                raise ResearchStateError("Phase contract and authorized phase differ")
            if phase_contract.get("family_id") != state["active_family_id"]:
                raise ResearchStateError("Phase contract and active family differ")
            if (
                phase_contract.get("authorized_next_action")
                != state["authorized_next_action"]
            ):
                raise ResearchStateError("Phase contract and authorization differ")
            if phase_contract.get("authorized_command") != state.get(
                "authorized_command"
            ):
                raise ResearchStateError("Phase contract and command differ")
            parent = phase_contract.get("parent_experiment", {})
            artifact_checks["phase_contract_parent_path_matches"] = (
                parent.get("path") == state["current_experiment"]
            )
            artifact_checks["phase_contract_parent_hash_matches"] = (
                parent.get("file_sha256") == _sha256_file(experiment_path)
            )
            _validate_access_contract(root, phase_contract)
            if phase_contract.get("stage_revision") == "v059_unseal_r1":
                _validate_v59_unseal_boundary(root, state, experiment, phase_contract)
            elif phase_contract.get("stage_revision") == "v059_terminal_r1":
                _validate_authorization_receipt(root, state, experiment, phase_contract)
                _validate_v59_terminal_boundary(root, state, experiment, phase_contract)
            elif phase_contract.get("stage_revision") == "v060_specification_r1":
                _validate_authorization_receipt(root, state, experiment, phase_contract)
                _validate_v60_specification_boundary(
                    root, state, experiment, phase_contract
                )
            elif (
                phase_contract.get("stage_revision")
                == "v065_v64_r2_probabilistic_state_gate_specification_r1"
            ):
                _validate_authorization_receipt(root, state, experiment, phase_contract)
                _validate_v65_v64_r2_specification_boundary(
                    root, state, experiment, phase_contract
                )
            elif (
                phase_contract.get("stage_revision")
                == "v066_synthetic_v64_r2_probabilistic_state_gate_harness_r1"
            ):
                _validate_authorization_receipt(root, state, experiment, phase_contract)
                _validate_v66_v64_r2_harness_boundary(
                    root, state, experiment, phase_contract
                )
            elif (
                phase_contract.get("stage_revision")
                == "v067_non_target_v64_r2_probabilistic_state_gate_dataset_r1"
            ):
                _validate_authorization_receipt(root, state, experiment, phase_contract)
                _validate_v67_v64_r2_dataset_boundary(
                    root, state, experiment, phase_contract
                )
            elif (
                phase_contract.get("stage_revision")
                == "v068_frozen_non_target_v64_r2_gate_training_r4"
            ):
                _validate_authorization_receipt(root, state, experiment, phase_contract)
                _validate_v68_v64_r2_gate_training_boundary(
                    root, state, experiment, phase_contract
                )
            elif (
                phase_contract.get("stage_revision")
                == "v069_outcome_blind_non_target_prospective_confirmation_prepare_r1"
            ):
                _validate_authorization_receipt(root, state, experiment, phase_contract)
                _validate_v69_v64_r2_prospective_prepare_boundary(
                    root, state, experiment, phase_contract
                )
            elif (
                phase_contract.get("stage_revision")
                == "v070_prospective_non_target_capture_prediction_freeze_r1"
            ):
                _validate_authorization_receipt(root, state, experiment, phase_contract)
                _validate_v70_v64_r2_prospective_capture_boundary(
                    root, state, experiment, phase_contract
                )
            elif (
                phase_contract.get("stage_revision")
                == "v071_posthoc_consumed_2025_diagnostic_prepare_r2"
            ):
                _validate_v71_posthoc_retrospective_prepare_boundary(
                    root, state, experiment, phase_contract
                )
            elif (
                phase_contract.get("stage_revision")
                == "v072_hash_bound_posthoc_outcome_unseal_r1"
            ):
                _validate_v72_posthoc_outcome_unseal_boundary(
                    root, state, experiment, phase_contract
                )
            elif (
                phase_contract.get("stage_revision")
                == "v073_metadata_only_v72_diagnostic_record_r1"
            ):
                _validate_v73_v72_diagnostic_record_boundary(
                    root, state, experiment, phase_contract
                )
            elif (
                phase_contract.get("stage_revision")
                == "v074_persistent_duration_family_specification_r1"
            ):
                _validate_v74_persistent_duration_specification_boundary(
                    root, state, experiment, phase_contract
                )
            elif (
                phase_contract.get("stage_revision")
                == "v075_synthetic_persistent_duration_harness_r1"
            ):
                _validate_v75_persistent_duration_harness_boundary(
                    root, state, experiment, phase_contract
                )
            elif (
                phase_contract.get("stage_revision")
                == "v076_non_target_persistent_duration_dataset_r2"
            ):
                _validate_v76_persistent_duration_dataset_boundary(
                    root, state, experiment, phase_contract
                )
            elif (
                phase_contract.get("stage_revision")
                == "v077_frozen_non_target_persistent_duration_training_r3"
            ):
                _validate_authorization_receipt(root, state, experiment, phase_contract)
                _validate_v77_persistent_duration_training_boundary(
                    root, state, experiment, phase_contract
                )
            elif (
                phase_contract.get("stage_revision")
                == "v078_outcome_blind_persistent_duration_evaluation_prepare_r3"
            ):
                _validate_authorization_receipt(root, state, experiment, phase_contract)
                _validate_v78_persistent_duration_evaluation_prepare_boundary(
                    root, state, experiment, phase_contract
                )
            elif (
                phase_contract.get("stage_revision")
                == "v079_metadata_only_v78_terminal_record_r1"
            ):
                _validate_v79_v78_terminal_record_boundary(
                    root, state, experiment, phase_contract
                )
            elif (
                phase_contract.get("stage_revision")
                == "v080_low_turnover_cross_sectional_rank_specification_r1"
            ):
                _validate_v80_low_turnover_rank_specification_boundary(
                    root, state, experiment, phase_contract
                )
            elif (
                phase_contract.get("stage_revision")
                == "v081_synthetic_low_turnover_rank_harness_r1"
            ):
                _validate_v81_low_turnover_rank_harness_boundary(
                    root, state, experiment, phase_contract
                )
            elif (
                phase_contract.get("stage_revision")
                == "v082_r0_metadata_only_chronology_erratum_r1"
            ):
                _validate_v82_r0_low_turnover_rank_chronology_erratum_boundary(
                    root, state, experiment, phase_contract
                )
            elif (
                phase_contract.get("stage_revision")
                == "v082_non_target_low_turnover_rank_dataset_r1"
            ):
                _validate_authorization_receipt(
                    root, state, experiment, phase_contract
                )
                _validate_v82_low_turnover_rank_dataset_boundary(
                    root, state, experiment, phase_contract
                )
            elif (
                phase_contract.get("stage_revision")
                == "v083_frozen_non_target_low_turnover_rank_training_r2"
            ):
                _validate_authorization_receipt(
                    root, state, experiment, phase_contract
                )
                _validate_v83_low_turnover_rank_training_boundary(
                    root, state, experiment, phase_contract
                )
            elif (
                phase_contract.get("stage_revision")
                == "v084_outcome_blind_low_turnover_rank_evaluation_prepare_r2"
            ):
                _validate_authorization_receipt(
                    root, state, experiment, phase_contract
                )
                _validate_v84_low_turnover_rank_evaluation_prepare_boundary(
                    root, state, experiment, phase_contract
                )
            elif (
                phase_contract.get("stage_revision")
                == "v085_hash_bound_low_turnover_rank_unseal_r1"
            ):
                _validate_v85_low_turnover_rank_unseal_boundary(
                    root, state, experiment, phase_contract
                )
            elif phase_contract.get("stage_revision") == "v061_harness_r1":
                _validate_authorization_receipt(root, state, experiment, phase_contract)
                _validate_v61_harness_boundary(root, state, experiment, phase_contract)
            elif phase_contract.get("stage_revision") == "v062_dataset_r1":
                _validate_authorization_receipt(root, state, experiment, phase_contract)
                _validate_v62_dataset_boundary(root, state, experiment, phase_contract)
            elif phase_contract.get("stage_revision") == "v063_training_r1":
                _validate_authorization_receipt(root, state, experiment, phase_contract)
                _validate_v63_training_boundary(root, state, experiment, phase_contract)
            elif (
                phase_contract.get("stage_revision")
                == "v064_adaptive_development_evaluation_r1"
            ):
                _validate_authorization_receipt(root, state, experiment, phase_contract)
                _validate_v64_evaluation_boundary(
                    root, state, experiment, phase_contract
                )
            elif phase_contract.get("stage_revision") == "v064_unseal_r1":
                _validate_v64_unseal_boundary(
                    root, state, experiment, phase_contract
                )
            elif phase_contract.get("stage_revision") == "v064_terminal_r1":
                _validate_v64_terminal_boundary(
                    root, state, experiment, phase_contract
                )
            else:
                _validate_authorization_receipt(root, state, experiment, phase_contract)
                _validate_v59_outcome_boundary(root, state, experiment, phase_contract)
    for name in ("blueprint", "result", "audit"):
        reference = experiment.get(name)
        if not isinstance(reference, dict) or "path" not in reference:
            raise ResearchStateError(f"Experiment is missing the {name} reference")
        path = _resolve_project_path(root, reference["path"])
        exists = path.is_file()
        artifact_checks[f"{name}_exists"] = exists
        if not exists:
            continue
        expected_file_hash = reference.get("file_sha256")
        if expected_file_hash is not None:
            artifact_checks[f"{name}_file_sha256_matches"] = (
                _sha256_file(path) == expected_file_hash
            )
    blueprint_path = _resolve_project_path(root, experiment["blueprint"]["path"])
    if blueprint_path.is_file():
        blueprint = json.loads(blueprint_path.read_text(encoding="utf-8"))
        registered_hash = blueprint.pop("blueprint_sha256", None)
        canonical_hash = _canonical_sha256(blueprint)
        expected_hash = experiment["blueprint"].get("canonical_sha256")
        artifact_checks["blueprint_canonical_sha256_matches"] = (
            registered_hash == expected_hash == canonical_hash
        )

    families = state["families"]
    family_ids = [item.get("family_id") for item in families]
    if len(family_ids) != len(set(family_ids)):
        raise ResearchStateError("Duplicate family IDs in current state")
    if state["active_family_id"] not in family_ids:
        raise ResearchStateError("Active family is absent from family registry")
    if not all(artifact_checks.values()):
        failed = sorted(key for key, passed in artifact_checks.items() if not passed)
        raise ResearchStateError(f"Current artifact checks failed: {failed}")

    trained = sum(bool(item.get("trained")) for item in families)
    return {
        "passed": True,
        "state_path": str(current_path.relative_to(root)),
        "experiment_path": str(experiment_path.relative_to(root)),
        "phase_contract_path": (
            str(phase_contract_path.relative_to(root))
            if phase_contract_path is not None
            else None
        ),
        "active_family_id": state["active_family_id"],
        "active_family_status": state["active_family_status"],
        "authorized_next_action": state["authorized_next_action"],
        "authorized_phase": state["authorized_phase"],
        "authorized_command": state.get("authorized_command"),
        "target_asset_status": state["target_assets"]["status"],
        "deployable_strategy": bool(state.get("deployable_strategy", False)),
        "family_count": len(families),
        "trained_family_count": trained,
        "retired_family_count": sum(
            item.get("status") == "retired" for item in families
        ),
        "forbidden_capabilities": list(state["forbidden_capabilities"]),
        "artifact_checks": artifact_checks,
    }


def research_doctor(
    project_root: str | Path = ".",
    state_path: str | Path = "research/current.yaml",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    status = validate_research_state(root, state_path)
    state = _load_mapping(_resolve_project_path(root, str(state_path)))
    training_contract, _ = _training_contract(root, status)
    safety = state.get("safety", {})
    contract_safety = training_contract.get("safety", {})
    runtime_contract = training_contract.get("runtime_contract", {})
    if isinstance(runtime_contract, dict) and runtime_contract:
        contract_safety = runtime_contract
    minimum_free_gib = float(
        contract_safety.get(
            "minimum_free_gib",
            safety.get("minimum_free_gib_for_full_training", 50),
        )
    )
    disk = shutil.disk_usage(root)
    free_gib = disk.free / (1024**3)
    git_status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    git_remote = subprocess.run(
        ["git", "remote"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    python_ok = sys.version_info >= (3, 11)
    torch_ok = hasattr(torch, "__version__")
    mps_built = bool(torch.backends.mps.is_built())
    mps_available = bool(torch.backends.mps.is_available())
    mps_operational = False
    mps_error: str | None = None
    if mps_available:
        try:
            probe = torch.ones(1, dtype=torch.float32, device="mps")
            mps_operational = float((probe + 1).cpu().item()) == 2.0
            if hasattr(torch, "mps"):
                torch.mps.synchronize()
        except (RuntimeError, TypeError, ValueError) as exc:
            mps_error = str(exc)

    deterministic_algorithms = False
    deterministic_error: str | None = None
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        deterministic_algorithms = torch.are_deterministic_algorithms_enabled()
    except RuntimeError as exc:
        deterministic_error = str(exc)
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
    fallback_enabled = _parse_false_environment_flag("PYTORCH_ENABLE_MPS_FALLBACK")

    lock_path = _training_lock_path(
        root, str(status["authorized_phase"]), training_contract
    )
    process_lock = _inspect_training_lock(lock_path)
    backup = _validate_backup_receipt(
        root,
        str(status["authorized_phase"]),
        training_contract,
        git_head,
    )
    warnings: list[str] = []
    if free_gib < minimum_free_gib:
        warnings.append("free_disk_below_full_training_threshold")
    if git_status:
        warnings.append("working_tree_not_clean")
    if (
        not git_remote
        and not backup.get("code_backup_verified")
        and backup.get("mode") != "owner_waiver"
    ):
        warnings.append("no_git_remote_or_external_code_backup")
    if not mps_available or not mps_operational:
        warnings.append("mps_not_available")
    if fallback_enabled:
        warnings.append("mps_fallback_enabled")
    if not deterministic_algorithms:
        warnings.append("deterministic_algorithms_unavailable")
    if not process_lock["available"]:
        warnings.append("training_process_lock_active")
    if not backup["passed"]:
        warnings.append("storage_protection_policy_invalid")

    full_training_ready = (
        status["passed"]
        and free_gib >= minimum_free_gib
        and (not safety.get("full_training_requires_clean_git", True) or not git_status)
        and python_ok
        and torch_ok
        and deterministic_algorithms
        and not fallback_enabled
        and process_lock["available"]
        and backup["passed"]
        and (
            not safety.get("full_training_requires_mps", True)
            or (mps_available and mps_operational)
        )
    )
    return {
        "passed": True,
        "authorized_phase": status["authorized_phase"],
        "synthetic_phase_ready": status["authorized_phase"] == "v56",
        "full_training_ready": full_training_ready,
        "disk": {
            "free_gib": round(free_gib, 2),
            "free_bytes": int(disk.free),
            "minimum_free_gib_for_full_training": minimum_free_gib,
            "required_free_bytes": int(minimum_free_gib * 1024**3),
        },
        "git": {
            "clean": not git_status,
            "changed_entry_count": len(git_status),
            "remote_configured": bool(git_remote),
            "head": git_head,
        },
        "runtime": {
            "python_ok": python_ok,
            "torch_ok": torch_ok,
            "mps_built": mps_built,
            "mps_available": mps_available,
            "mps_operational": mps_operational,
            "mps_error": mps_error,
            "device": "mps",
            "dtype": "float32",
            "deterministic_algorithms": deterministic_algorithms,
            "deterministic_error": deterministic_error,
            "fallback_enabled": fallback_enabled,
        },
        "process_lock": process_lock,
        "backup": backup,
        "warnings": warnings,
    }
