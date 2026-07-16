"""Metadata-only contracts and artifact helpers for frozen V58 training.

This module deliberately has no pandas, PyArrow, NumPy, or torch imports.  The
three registered Parquet inputs are authenticated only as opaque byte streams.
"""

from __future__ import annotations

from copy import deepcopy
import itertools
import json
from pathlib import Path
import re
import subprocess
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Mapping, Sequence

import yaml

from tlm.core.artifacts import canonical_sha256, file_sha256


FAMILY_ID = "tlm_state_conditioned_multi_horizon_quantile_small_v1"
V58_ACTION = "authorize_v58_frozen_non_target_training_only"
V59_ACTION = "authorize_v59_frozen_adaptive_development_evaluation_only"
V57_ACTION = "authorize_v57_non_target_multi_horizon_dataset_build_only"
INPUT_NAMES = (
    "v55_blueprint",
    "v56_result",
    "v56_audit",
    "v56_harness_spec",
    "v57_result",
    "v57_audit",
    "v57_dataset_spec",
    "v57_dataset_manifest",
    "v57_label_schema",
    "v57_source_receipt",
    "v57_completion_receipt",
    "v57_artifact_manifest",
    "v57_data_access",
    "v32_feature_schema",
    "v32_asset_folds",
    "v32_triplet_catalog",
    "feature_panel",
    "labels",
    "sequence_roles",
)
BINARY_INPUT_NAMES = {"feature_panel", "labels", "sequence_roles"}
REPLAY_MUTABLE_NAMES = {
    "artifact_manifest.json",
    "audit.json",
    "completion_receipt.json",
    "operator_packet_replay.json",
    "replay.json",
    "report.md",
    "result.json",
}
SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_HEAD = re.compile(r"^[0-9a-f]{40,64}$")
V58_BASE_PHASE_CONTRACT = {
    "path": "research/phase_contracts/v058.yaml",
    "file_sha256": "23883bb354d8bb479778435188fb58962bf43b03f431981e4b963a674a09d868",
}
V58_OWNER_STORAGE_WAIVER = {
    "path": "research/waivers/v058r1_external_backup_owner_waiver.json",
    "file_sha256": "067f23f39937e00c5b9ce40a0248c1683bc9329e195fb10bd9ffd24b69b7e6f9",
}
V58_WAIVED_STORAGE_SAFEGUARDS = [
    "external_input_copy",
    "external_code_copy",
    "external_checkpoint_copy",
]


class TrainingArtifactError(RuntimeError):
    """Raised when frozen V58 metadata or an artifact receipt drifts."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainingArtifactError(message)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{name} must be an object")
    return value


def _string_list(value: Any, name: str) -> list[str]:
    _require(isinstance(value, list), f"{name} must be an array")
    _require(
        all(isinstance(item, str) and item for item in value),
        f"{name} must contain non-empty strings",
    )
    return list(value)


def resolve_repo_path(root: str | Path, relative: str | Path, name: str = "path") -> Path:
    """Resolve a non-empty relative path and reject root escapes/symlink escapes."""

    repository = Path(root).resolve()
    raw = Path(relative)
    _require(str(relative) not in {"", "."}, f"{name} must be a file path")
    _require(not raw.is_absolute(), f"{name} must be relative to the repository")
    candidate = (repository / raw).resolve()
    _require(candidate != repository and repository in candidate.parents, f"{name} escapes repository root")
    return candidate


def _load_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingArtifactError(f"{name} is not valid JSON: {exc}") from exc
    return _mapping(value, name)


def _load_yaml(path: Path, name: str) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise TrainingArtifactError(f"{name} is not valid YAML: {exc}") from exc
    return _mapping(value, name)


def _verify_backup_only_revision(
    root: Path, contract: Mapping[str, Any]
) -> None:
    """Accept only the immutable V58 base or its exact storage-only V58r1."""

    immutable_base_path = resolve_repo_path(
        root, V58_BASE_PHASE_CONTRACT["path"], "V58 base contract path"
    )
    _require(
        immutable_base_path.is_file()
        and file_sha256(immutable_base_path)
        == V58_BASE_PHASE_CONTRACT["file_sha256"],
        "immutable V58 base contract file/hash drift",
    )
    base = _load_yaml(immutable_base_path, "V58 base phase contract")
    if contract.get("revision") is None:
        _require(
            dict(contract) == base,
            "unrevisioned V58 contract differs from the immutable base",
        )
        return
    _require(contract.get("revision") == "v058r1", "unsupported V58 revision")
    supersedes = _mapping(contract.get("supersedes"), "supersedes")
    operational_waiver = _mapping(
        contract.get("operational_waiver"), "operational_waiver"
    )
    expected_supersedes = {
        **V58_BASE_PHASE_CONTRACT,
        "allowed_change_scope": "external_storage_redundancy_only",
    }
    _require(supersedes == expected_supersedes, "V58r1 base binding drift")
    _require(
        operational_waiver == V58_OWNER_STORAGE_WAIVER,
        "V58r1 owner waiver binding drift",
    )
    base_path = resolve_repo_path(root, supersedes.get("path", ""), "supersedes.path")
    waiver_path = resolve_repo_path(
        root, operational_waiver.get("path", ""), "operational_waiver.path"
    )
    _require(
        base_path.is_file()
        and file_sha256(base_path) == supersedes.get("file_sha256"),
        "V58r1 base contract file/hash drift",
    )
    _require(
        waiver_path.is_file()
        and file_sha256(waiver_path) == operational_waiver.get("file_sha256"),
        "V58r1 owner waiver file/hash drift",
    )
    def exact_replacement(
        values: Sequence[str], old: str, new: str, name: str
    ) -> list[str]:
        items = list(values)
        _require(
            items.count(old) == 1 and new not in items,
            f"V58 base {name} is not replaceable exactly once",
        )
        return [new if item == old else item for item in items]

    expected = deepcopy(base)
    expected["revision"] = "v058r1"
    expected["supersedes"] = expected_supersedes
    expected["operational_waiver"] = V58_OWNER_STORAGE_WAIVER
    expected["access_contract"]["required_checks"] = exact_replacement(
        base["access_contract"]["required_checks"],
        "clean_committed_git_and_external_backup_receipt",
        "clean_committed_git_and_bound_owner_storage_waiver",
        "access required checks",
    )
    expected_runtime = expected["runtime_contract"]
    expected_runtime["external_backup_receipt_required"] = False
    expected_runtime.pop("external_backup_receipt")
    expected_runtime["backup_policy"] = {
        "mode": "owner_waiver",
        "waiver": V58_OWNER_STORAGE_WAIVER,
        "waived_safeguards": V58_WAIVED_STORAGE_SAFEGUARDS,
    }
    expected["artifact_contract"]["packet_files"] = exact_replacement(
        base["artifact_contract"]["packet_files"],
        "checkpoint_backup_receipt.json",
        "backup_policy_receipt.json",
        "artifact packet files",
    )
    expected["operator_enforcement_contract"][
        "live_doctor_fields_must_match_packet"
    ] = exact_replacement(
        base["operator_enforcement_contract"][
            "live_doctor_fields_must_match_packet"
        ],
        "external_backup_receipt",
        "backup_policy_and_owner_waiver",
        "operator doctor fields",
    )
    expected["required_checks"] = exact_replacement(
        base["required_checks"],
        "clean_committed_git_and_external_backup_receipt",
        "clean_committed_git_and_bound_owner_storage_waiver",
        "top-level required checks",
    )
    _require(
        dict(contract) == expected,
        "V58r1 changes fields outside the registered backup-only allowlist",
    )


def _verify_self_hash(value: Mapping[str, Any], field: str, name: str) -> str:
    registered = value.get(field)
    _require(isinstance(registered, str) and SHA256.fullmatch(registered) is not None, f"{name} lacks a valid {field}")
    body = deepcopy(dict(value))
    body.pop(field, None)
    _require(canonical_sha256(body) == registered, f"{name} canonical hash drift")
    return registered


def _git(root: Path, *args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=not binary, check=False
    )
    stderr = result.stderr.decode(errors="replace") if binary else result.stderr
    _require(result.returncode == 0, f"git {' '.join(args)} failed: {str(stderr).strip()}")
    return result.stdout if binary else result.stdout.strip()


def _source_receipt(root: Path, source_files: Sequence[str]) -> dict[str, Any]:
    files = list(source_files)
    _require(files and len(files) == len(set(files)), "V58 source receipt file set is empty or duplicated")
    _require(all(isinstance(item, str) and item for item in files), "V58 source receipt paths must be non-empty strings")
    top = Path(str(_git(root, "rev-parse", "--show-toplevel"))).resolve()
    _require(top == root, "configured project_root is not the Git repository root")
    head = str(_git(root, "rev-parse", "HEAD"))
    _require(GIT_HEAD.fullmatch(head) is not None, "Git HEAD is not a commit hash")
    _require(
        _git(root, "status", "--porcelain", "--untracked-files=all") == "",
        "V58 requires a clean tracked and untracked Git worktree",
    )
    tracked_raw = _git(root, "ls-files", "-z", "--", *files, binary=True)
    assert isinstance(tracked_raw, bytes)
    tracked = {item.decode("utf-8") for item in tracked_raw.split(b"\0") if item}
    _require(tracked == set(files), "V58 source receipt contains missing, ignored, or untracked files")
    hashes: dict[str, str] = {}
    for relative in files:
        path = resolve_repo_path(root, relative, f"source_receipt_files[{relative}]")
        _require(path.is_file(), f"V58 source receipt file is missing: {relative}")
        hashes[relative] = file_sha256(path)
    return {
        "version": "v58_source_receipt_v1",
        "git_clean": True,
        "git_head": head,
        "files": hashes,
        "bundle_sha256": canonical_sha256(hashes),
    }


def exact_job_ids(contract: Mapping[str, Any]) -> list[str]:
    """Return the registered origin -> geometry -> fold -> seed order."""

    grid = _mapping(contract.get("grid_contract"), "grid_contract")
    axes: list[list[Any]] = []
    for key in ("origins", "geometries", "folds", "seeds"):
        values = grid.get(key)
        _require(isinstance(values, list) and values, f"grid_contract.{key} must be a non-empty array")
        _require(len(values) == len(set(values)), f"grid_contract.{key} contains duplicates")
        axes.append(list(values))
    jobs = [
        f"{origin}|{geometry}|{fold}|{seed}"
        for origin, geometry, fold, seed in itertools.product(*axes)
    ]
    _require(grid.get("expected_jobs") == 36 and len(jobs) == 36, "V58 grid must contain exactly 36 jobs")
    _require(grid.get("job_key_order", "origin_geometry_fold_seed") == "origin_geometry_fold_seed", "V58 job key order drift")
    return jobs


def _verify_lineage(
    contract: dict[str, Any],
    values: dict[str, dict[str, Any]],
    hashes: dict[str, str],
    preceding: dict[str, Any],
    experiment: dict[str, Any],
) -> None:
    blueprint = values["v55_blueprint"]
    blueprint_hash = _verify_self_hash(blueprint, "blueprint_sha256", "V55 blueprint")
    _require(
        blueprint.get("version") == "v55"
        and blueprint.get("candidate_family_id") == FAMILY_ID
        and blueprint.get("parameter_count_analytic") == 465513
        and blueprint.get("registered_job_count") == 36,
        "V55 blueprint identity or architecture drift",
    )
    _require(
        blueprint_hash == contract["v55_blueprint"]["canonical_sha256"],
        "V55 blueprint canonical authorization drift",
    )
    lifecycle = _mapping(blueprint.get("lifecycle"), "V55 lifecycle")
    _require(
        lifecycle.get("v57_pass_action") == V58_ACTION
        and lifecycle.get("v58_pass_action") == V59_ACTION,
        "V55 lifecycle authorization drift",
    )

    v56_result = values["v56_result"]
    v56_audit = values["v56_audit"]
    v56_spec = values["v56_harness_spec"]
    _verify_self_hash(v56_spec, "harness_spec_sha256", "V56 harness spec")
    _verify_self_hash(v56_result, "result_sha256", "V56 result")
    _require(
        v56_result.get("version") == "v56"
        and v56_result.get("candidate_family_id") == FAMILY_ID
        and v56_result.get("decision") == V57_ACTION
        and v56_result.get("audit") == v56_audit
        and v56_result.get("harness_spec") == v56_spec
        and v56_audit.get("passed") is True
        and v56_spec.get("authorized_next_action") == V57_ACTION
        and v56_spec.get("v55_blueprint_sha256") == blueprint_hash,
        "V56 authorization or harness evidence drift",
    )

    v57_result = values["v57_result"]
    v57_audit = values["v57_audit"]
    v57_spec = values["v57_dataset_spec"]
    _verify_self_hash(v57_spec, "dataset_spec_sha256", "V57 dataset spec")
    result_hash = _verify_self_hash(v57_result, "result_sha256", "V57 result")
    _require(
        v57_result.get("version") == "v57"
        and v57_result.get("candidate_family_id") == FAMILY_ID
        and v57_result.get("decision") == V58_ACTION
        and v57_result.get("audit") == v57_audit
        and v57_result.get("dataset_spec") == v57_spec
        and v57_audit.get("passed") is True
        and v57_spec.get("pass_action") == V58_ACTION,
        "V57 authorization or dataset evidence drift",
    )
    preceding_hash = contract["preceding_phase_contract"]["file_sha256"]
    _require(
        v57_spec.get("phase_contract_file_sha256") == preceding_hash
        and preceding.get("phase") == "v57"
        and preceding.get("family_id") == FAMILY_ID
        and preceding.get("authorized_next_action") == V57_ACTION
        and preceding.get("pass_action") == V58_ACTION,
        "V57 preceding phase authorization drift",
    )
    completion = values["v57_completion_receipt"]
    _require(
        completion.get("decision") == V58_ACTION
        and completion.get("result_file_sha256") == hashes["v57_result"]
        and completion.get("audit_file_sha256") == hashes["v57_audit"]
        and completion.get("dataset_spec_sha256") == v57_spec["dataset_spec_sha256"]
        and completion.get("artifact_manifest_file_sha256") == hashes["v57_artifact_manifest"],
        "V57 completion receipt drift",
    )
    manifest = values["v57_artifact_manifest"]
    _verify_self_hash(manifest, "manifest_sha256", "V57 artifact manifest")
    _require(
        manifest.get("data_files", {}).get(str(values["v57_dataset_manifest"]["labels"]["path"])) == hashes["labels"]
        and manifest.get("data_files", {}).get(str(values["v57_dataset_manifest"]["sequence_roles"]["path"])) == hashes["sequence_roles"],
        "V57 data artifact bindings drift",
    )
    source = values["v57_source_receipt"]
    _require(
        isinstance(source.get("files"), dict)
        and canonical_sha256(source["files"]) == source.get("bundle_sha256"),
        "V57 source receipt bundle drift",
    )
    label_schema = values["v57_label_schema"]
    _verify_self_hash(label_schema, "label_schema_sha256", "V57 label schema")
    _require(
        result_hash == contract["authorization_receipt"]["registered_result_sha256"]
        and hashes["v57_result"] == contract["authorization_receipt"]["file_sha256"],
        "V57 authorization receipt drift",
    )
    _require(
        experiment.get("family_id") == FAMILY_ID
        and experiment.get("status") == "passed"
        and experiment.get("authorized_next_action") == V58_ACTION,
        "current experiment does not authorize V58",
    )
    if contract.get("revision") == "v058r1":
        parent_v58 = _mapping(
            experiment.get("v58_contract"), "V57 parent V58 contract"
        )
        parent_safety = _mapping(
            parent_v58.get("safety"), "V57 parent V58 safety"
        )
        _require(
            parent_safety.get("external_backup_receipt_required") is True
            and parent_safety.get("external_backup_receipt")
            == "research/backups/v058.yaml"
            and "clean_committed_git_and_external_backup_receipt"
            in parent_v58.get("required_checks", []),
            "V58r1 parent storage requirement is not the exact waived invariant",
        )


def prepare_training_metadata(
    config: Mapping[str, Any], *, repo_root: str | Path | None = None
) -> dict[str, Any]:
    """Validate V58 metadata, opaque input hashes, Git, and source receipt.

    No Parquet parser is imported or invoked.  Binary inputs are opened only by
    :func:`file_sha256` as byte streams.
    """

    training = _mapping(config.get("state_conditioned_multi_horizon_training"), "state_conditioned_multi_horizon_training")
    _require(training.get("version") == "v58", "training config version must be v58")
    root = Path(repo_root if repo_root is not None else training.get("project_root", ".")).resolve()
    _require(root.is_dir(), "V58 project root is missing")

    state_path = resolve_repo_path(root, training.get("research_state", ""), "research_state")
    state = _load_yaml(state_path, "research state")
    phase_ref = _mapping(state.get("phase_contract"), "research state phase_contract")
    phase_relative = training.get("phase_contract")
    _require(phase_ref.get("path") == phase_relative, "config and research state phase-contract paths differ")
    phase_path = resolve_repo_path(root, phase_relative, "phase_contract")
    _require(phase_path.is_file(), "V58 phase contract is missing")
    phase_hash = file_sha256(phase_path)
    _require(phase_hash == phase_ref.get("file_sha256"), "V58 phase contract hash drift")
    contract = _load_yaml(phase_path, "V58 phase contract")
    _verify_backup_only_revision(root, contract)

    _require(
        state.get("authorized_phase") == "v58"
        and state.get("authorized_next_action") == V58_ACTION
        and state.get("active_family_id") == FAMILY_ID
        and state.get("last_completed_phase") == "v57_non_target_multi_horizon_dataset",
        "research state does not authorize V58",
    )
    _require(
        contract.get("phase") == "v58"
        and contract.get("family_id") == FAMILY_ID
        and contract.get("authorized_next_action") == V58_ACTION
        and contract.get("pass_action") == V59_ACTION
        and contract.get("failure_action") == "keep_v59_and_later_unauthorized",
        "V58 frozen phase contract identity or authorization drift",
    )

    experiment_relative = training.get("experiment_contract")
    _require(state.get("current_experiment") == experiment_relative, "current experiment path drift")
    parent_ref = _mapping(contract.get("parent_experiment"), "parent_experiment")
    _require(parent_ref.get("path") == experiment_relative, "V58 parent experiment path drift")
    experiment_path = resolve_repo_path(root, experiment_relative, "experiment_contract")
    _require(experiment_path.is_file() and file_sha256(experiment_path) == parent_ref.get("file_sha256"), "V58 parent experiment hash drift")
    experiment = _load_yaml(experiment_path, "V57 experiment")

    preceding_ref = _mapping(contract.get("preceding_phase_contract"), "preceding_phase_contract")
    preceding_path = resolve_repo_path(root, preceding_ref.get("path", ""), "preceding_phase_contract")
    _require(preceding_path.is_file() and file_sha256(preceding_path) == preceding_ref.get("file_sha256"), "V57 preceding phase contract hash drift")
    preceding = _load_yaml(preceding_path, "V57 phase contract")

    inputs = _mapping(training.get("inputs"), "training inputs")
    expected = _mapping(contract.get("input_contract"), "input_contract").get("expected_sha256")
    expected = _mapping(expected, "input_contract.expected_sha256")
    _require(tuple(inputs) == INPUT_NAMES and tuple(expected) == INPUT_NAMES, "V58 input-name order or allowlist drift")
    allowed = _string_list(contract.get("access_contract", {}).get("allowed_inputs"), "access_contract.allowed_inputs")
    _require(list(inputs.values()) == allowed, "V58 configured input paths differ from the exact allowlist")

    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    values: dict[str, dict[str, Any]] = {}
    for name in INPUT_NAMES:
        path = resolve_repo_path(root, inputs[name], f"inputs.{name}")
        _require(path.is_file(), f"V58 input is missing: {name}")
        observed = file_sha256(path)
        _require(observed == expected[name], f"V58 input hash drift: {name}")
        paths[name] = path
        hashes[name] = observed
        if name not in BINARY_INPUT_NAMES:
            _require(path.suffix == ".json", f"V58 metadata input is not JSON: {name}")
            values[name] = _load_json(path, f"V58 input {name}")

    _verify_lineage(contract, values, hashes, preceding, experiment)
    _require(training.get("require_clean_git") is True, "V58 clean-Git enforcement cannot be disabled")
    source_files = _string_list(training.get("source_receipt_files"), "source_receipt_files")
    source_receipt = _source_receipt(root, source_files)
    jobs = exact_job_ids(contract)
    _require(config.get("output_dir") == contract["access_contract"]["output_dir"], "V58 output directory drift")
    training_spec: dict[str, Any] = {
        "version": "v58_training_spec_v1",
        "phase_contract": deepcopy(phase_ref),
        "contract": deepcopy(contract),
        "source_receipt_files": list(source_files),
        "input_paths": deepcopy(inputs),
        "input_hashes": hashes,
        "git_head": source_receipt["git_head"],
        "source_bundle_sha256": source_receipt["bundle_sha256"],
        "expected_job_ids": jobs,
        "contract_sha256": canonical_sha256(contract),
    }
    training_spec["training_spec_sha256"] = canonical_sha256(training_spec)
    return {
        "root": root,
        "training": deepcopy(training),
        "state": state,
        "experiment": experiment,
        "preceding_phase_contract": preceding,
        "contract": contract,
        "phase_contract": deepcopy(phase_ref),
        "phase_contract_path": phase_path,
        "input_paths": paths,
        "input_hashes": hashes,
        "input_values": values,
        "source_receipt": source_receipt,
        "training_spec": training_spec,
        "job_ids": jobs,
        "output_dir": resolve_repo_path(root, config["output_dir"], "output_dir"),
    }


def _with_hash(value: dict[str, Any], field: str) -> dict[str, Any]:
    _require(field not in value, f"caller must not pre-populate {field}")
    result = deepcopy(value)
    result[field] = canonical_sha256(result)
    return result


def _ordered_entries(
    entries: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    expected_ids: Sequence[str],
    identity: str,
) -> list[dict[str, Any]]:
    if isinstance(entries, Mapping):
        rows = []
        for key, raw in entries.items():
            row = deepcopy(dict(raw)) if isinstance(raw, Mapping) else {"value": deepcopy(raw)}
            row.setdefault(identity, str(key))
            rows.append(row)
    else:
        rows = [deepcopy(dict(row)) for row in entries]
    positions = {value: index for index, value in enumerate(expected_ids)}
    ids = [row.get(identity) for row in rows]
    _require(all(isinstance(value, str) and value in positions for value in ids), f"manifest contains an unknown {identity}")
    _require(len(ids) == len(set(ids)), f"manifest contains duplicate {identity} values")
    _require([positions[value] for value in ids] == sorted(positions[value] for value in ids), f"manifest {identity} values are out of frozen order")
    return rows


def build_checkpoint_manifest(
    contract: Mapping[str, Any],
    jobs: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    *,
    selected_jobs: Sequence[str] = (),
    resume: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected = exact_job_ids(contract)
    _require(not selected_jobs, "V58 checkpoint selection is forbidden")
    rows = _ordered_entries(jobs, expected, "job_id")
    active: list[str] = []
    for row in rows:
        status = row.setdefault("status", "completed")
        _require(status in {"pending", "active", "completed"}, "unsupported checkpoint job status")
        origin, geometry, fold, seed = str(row["job_id"]).split("|")
        for key, value in (("origin", origin), ("geometry", geometry), ("fold", int(fold)), ("seed", int(seed))):
            _require(key not in row or row[key] == value, f"checkpoint {row['job_id']} {key} drift")
            row[key] = value
        if status == "active":
            active.append(row["job_id"])
    _require(len(active) <= 1, "V58 permits at most one active checkpoint job")
    payload = {
        "version": "v58_checkpoint_manifest_v1",
        "expected_jobs": expected,
        "jobs": rows,
        "selected_jobs": [],
        "active_jobs": active,
        "checkpoint_count": sum(row["status"] == "completed" for row in rows),
        "resume": deepcopy(dict(resume or {})),
    }
    return _with_hash(payload, "manifest_sha256")


def build_grid_manifest(
    contract: Mapping[str, Any], jobs: Sequence[Mapping[str, Any]] | Mapping[str, Any]
) -> dict[str, Any]:
    expected = exact_job_ids(contract)
    rows = _ordered_entries(jobs, expected, "job_id")
    statuses: dict[str, str] = {}
    for row in rows:
        status = row.get("status", "completed")
        _require(status in {"pending", "active", "completed"}, "unsupported grid job status")
        statuses[str(row["job_id"])] = status
    completed = [job for job in expected if statuses.get(job) == "completed"]
    active = [job for job in expected if statuses.get(job) == "active"]
    _require(len(active) <= 1, "V58 permits at most one active grid job")
    pending = [job for job in expected if job not in completed and job not in active]
    payload = {
        "version": "v58_grid_manifest_v1",
        "expected_jobs": expected,
        "jobs": rows,
        "completed_jobs": completed,
        "active_jobs": active,
        "pending_jobs": pending,
        "selected_jobs": [],
        "counts": {"expected": 36, "completed": len(completed), "active": len(active), "pending": len(pending)},
    }
    return _with_hash(payload, "manifest_sha256")


def build_history_manifest(
    contract: Mapping[str, Any], histories: Sequence[Mapping[str, Any]] | Mapping[str, Any]
) -> dict[str, Any]:
    expected = exact_job_ids(contract)
    rows = _ordered_entries(histories, expected, "job_id")
    for row in rows:
        _require("history" in row, f"history is missing for {row['job_id']}")
        observed = canonical_sha256(row["history"])
        _require(row.get("history_sha256", observed) == observed, f"history hash drift: {row['job_id']}")
        row["history_sha256"] = observed
    payload = {
        "version": "v58_history_manifest_v1",
        "expected_jobs": expected,
        "jobs": rows,
        "selected_jobs": [],
        "history_count": len(rows),
    }
    return _with_hash(payload, "manifest_sha256")


def build_scaler_manifest(
    contract: Mapping[str, Any], scalers: Sequence[Mapping[str, Any]] | Mapping[str, Any]
) -> dict[str, Any]:
    grid = _mapping(contract.get("grid_contract"), "grid_contract")
    expected = [
        f"{origin}|{geometry}|{fold}"
        for origin, geometry, fold in itertools.product(grid["origins"], grid["geometries"], grid["folds"])
    ]
    _require(len(expected) == contract.get("scaler_contract", {}).get("count", 12) == 12, "V58 scaler grid must contain exactly 12 cells")
    rows = _ordered_entries(scalers, expected, "scaler_id")
    for row in rows:
        origin, geometry, fold = str(row["scaler_id"]).split("|")
        row.setdefault("origin", origin)
        row.setdefault("geometry", geometry)
        row.setdefault("fold", int(fold))
        registered = row.get("scaler_sha256")
        _require(
            isinstance(registered, str) and SHA256.fullmatch(registered) is not None,
            f"scaler receipt lacks its registered scaler_sha256: {row['scaler_id']}",
        )
    payload = {
        "version": "v58_scaler_manifest_v1",
        "expected_scalers": expected,
        "scalers": rows,
        "scaler_count": len(rows),
    }
    return _with_hash(payload, "manifest_sha256")


def build_data_access_manifest(access: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(access))
    zero_keys = ("development_evaluation_outcome_rows_read", "outcome_rows_read")
    _require(any(key in value for key in zero_keys), "data access lacks the outcome-row counter")
    _require(all(value.get(key, 0) == 0 for key in zero_keys), "development/evaluation outcomes were read")
    for key in ("target_assets_loaded", "forbidden_columns_loaded", "previous_checkpoints_loaded"):
        _require(value.get(key, []) == [], f"data_access.{key} must be empty")
    heldout = value.get("heldout_fold_symbols_loaded_by_job", {})
    _require(isinstance(heldout, dict) and all(item == [] for item in heldout.values()), "held-out fold symbols were loaded")
    for key in ("predictions_written", "policy_actions_emitted", "performance_metrics_computed", "pnl_computed", "hyperparameters_changed"):
        _require(value.get(key) is False, f"data_access.{key} must be false")
    payload = {"version": "v58_data_access_v1", "data_access": value}
    return _with_hash(payload, "data_access_sha256")


def build_artifact_manifest(
    output_dir: str | Path, files: Iterable[str | Path]
) -> dict[str, Any]:
    root = Path(output_dir).resolve()
    names = [Path(item).as_posix() for item in files]
    _require(len(names) == len(set(names)), "artifact manifest paths contain duplicates")
    _require("artifact_manifest.json" not in names, "artifact manifest must exclude itself")
    hashes: dict[str, str] = {}
    for name in sorted(names):
        path = resolve_repo_path(root, name, f"artifact[{name}]")
        _require(path.is_file(), f"artifact is missing: {name}")
        hashes[name] = file_sha256(path)
    payload = {"version": "v58_artifact_manifest_v1", "files": hashes}
    return _with_hash(payload, "manifest_sha256")


def stable_replay_hashes(
    output_dir: str | Path, files: Iterable[str | Path] | None = None
) -> dict[str, str]:
    """Hash replay-stable outputs while excluding invocation/result receipts."""

    root = Path(output_dir).resolve()
    candidates = (
        [Path(item) for item in files]
        if files is not None
        else [path.relative_to(root) for path in root.rglob("*") if path.is_file()]
    )
    hashes: dict[str, str] = {}
    for relative in sorted(candidates, key=lambda item: item.as_posix()):
        name = relative.as_posix()
        if relative.name in REPLAY_MUTABLE_NAMES or relative.name.endswith(".tmp"):
            continue
        path = resolve_repo_path(root, name, f"replay artifact[{name}]")
        _require(path.is_file(), f"replay artifact is missing: {name}")
        hashes[name] = file_sha256(path)
    return hashes


def _write_text_atomic(path: str | Path, text: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            temporary = Path(handle.name)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json_atomic(path: str | Path, value: Any) -> None:
    _write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_yaml_atomic(path: str | Path, value: Any) -> None:
    _write_text_atomic(path, yaml.safe_dump(value, sort_keys=False, allow_unicode=True))


def write_report_atomic(path: str | Path, report: str) -> None:
    _require(isinstance(report, str), "report must be text")
    _write_text_atomic(path, report)
