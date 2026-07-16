#!/usr/bin/env python3
"""Validate a TLM one-shot evaluation lifecycle packet without evaluating outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_HEAD = re.compile(r"^[0-9a-f]{40,64}$")
PHASES = {"prepare", "unseal", "complete", "replay"}


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def mapping(value: Any, name: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{name} must be an object")
    return value


def inside(root: Path, relative: Any, name: str) -> Path:
    require(isinstance(relative, str) and relative, f"{name} must be a relative path")
    candidate = (root / relative).resolve()
    require(candidate == root or root in candidate.parents, f"{name} escapes repo root")
    return candidate


def expected_hash(value: Any, name: str) -> str:
    require(isinstance(value, str) and SHA256.fullmatch(value) is not None, f"invalid {name}")
    return value


def validate_file_ref(root: Path, value: Any, name: str) -> tuple[Path, str]:
    ref = mapping(value, name)
    path = inside(root, ref.get("path"), f"{name}.path")
    digest = expected_hash(ref.get("sha256"), f"{name}.sha256")
    require(path.is_file(), f"missing {name}: {path}")
    require(sha256_file(path) == digest, f"{name} hash drift")
    return path, digest


def load_receipt(root: Path, value: Any, name: str) -> tuple[dict[str, Any], str]:
    path, digest = validate_file_ref(root, value, name)
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{name} is not valid JSON: {exc}") from exc
    return mapping(receipt, name), digest


def workflow_json(root: Path, command: str) -> dict[str, Any]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(root / "src") if not existing else f"{root / 'src'}{os.pathsep}{existing}"
    result = subprocess.run(
        [sys.executable, "-m", "tlm", command],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    require(result.returncode == 0, f"{command} failed: {result.stderr.strip()}")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{command} did not emit JSON: {exc}") from exc
    return mapping(value, command)


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=False
    )
    require(result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def validate_packet(root: Path, packet: dict[str, Any]) -> dict[str, Any]:
    root = root.resolve()
    require(packet.get("schema_version") == "tlm-one-shot-evaluator/v1", "unsupported schema_version")
    phase = packet.get("phase")
    require(phase in PHASES, "phase must be prepare, unseal, complete, or replay")

    state = mapping(packet.get("research_state"), "research_state")
    state_path = inside(root, state.get("path"), "research_state.path")
    state_hash = expected_hash(state.get("sha256"), "research_state.sha256")
    require(state_path.is_file() and sha256_file(state_path) == state_hash, "research state hash drift")
    live_status = workflow_json(root, "research-status")
    require(live_status.get("passed") is True, "live research status failed")
    for key in ("authorized_phase", "authorized_next_action", "authorized_command"):
        require(isinstance(state.get(key), str) and state.get(key), f"research_state.{key} must be non-empty")
        require(live_status.get(key) == state.get(key), f"live {key} differs from evaluation packet")

    spec = mapping(packet.get("evaluation_spec"), "evaluation_spec")
    require(spec.get("frozen") is True, "evaluation spec is not frozen")
    _, spec_hash = validate_file_ref(root, spec, "evaluation_spec")

    source = mapping(packet.get("source_receipt"), "source_receipt")
    require(source.get("git_clean") is True, "source receipt is not clean")
    receipt_head = source.get("git_head")
    require(isinstance(receipt_head, str) and GIT_HEAD.fullmatch(receipt_head) is not None, "invalid source git_head")
    require(git(root, "rev-parse", "HEAD") == receipt_head, "live Git head differs from source receipt")
    require(git(root, "status", "--porcelain", "--untracked-files=all") == "", "tracked or untracked source is dirty")
    source_files = mapping(source.get("files"), "source_receipt.files")
    require(bool(source_files), "source receipt files cannot be empty")
    for relative, digest in source_files.items():
        expected_hash(digest, f"source_receipt.files[{relative}]")
        path = inside(root, relative, f"source_receipt.files[{relative}]")
        require(path.is_file() and sha256_file(path) == digest, f"source hash drift: {relative}")
    source_bundle = expected_hash(source.get("bundle_sha256"), "source_receipt.bundle_sha256")
    require(canonical_sha256(source_files) == source_bundle, "source bundle hash drift")

    registered = mapping(packet.get("registered"), "registered")
    costs = registered.get("cost_bps")
    require(isinstance(costs, list) and costs, "registered.cost_bps must be non-empty")
    require(all(isinstance(cost, int) and cost >= 0 for cost in costs), "registered costs must be non-negative integer bps")
    require(len(costs) == len(set(costs)), "registered costs contain duplicates")
    bound_registered = {
        "cost_bps": costs,
        "accounting": mapping(registered.get("accounting"), "registered.accounting"),
        "controls": mapping(registered.get("controls"), "registered.controls"),
        "gates": mapping(registered.get("gates"), "registered.gates"),
        "outcome_blind_gate_names": registered.get("outcome_blind_gate_names"),
    }
    require(all(bound_registered[key] for key in ("accounting", "controls", "gates")), "registered accounting, controls, and gates must be non-empty")
    blind_gate_names = bound_registered["outcome_blind_gate_names"]
    require(
        isinstance(blind_gate_names, list)
        and blind_gate_names
        and all(isinstance(name, str) and name for name in blind_gate_names),
        "registered.outcome_blind_gate_names must contain names",
    )
    require(len(blind_gate_names) == len(set(blind_gate_names)), "registered outcome-blind gates contain duplicates")
    registered_hash = expected_hash(registered.get("sha256"), "registered.sha256")
    require(canonical_sha256(bound_registered) == registered_hash, "registered costs/accounting/controls/gates hash drift")

    prepare = mapping(packet.get("prepare"), "prepare")
    prepare_receipt, prepare_hash = load_receipt(root, prepare.get("receipt"), "prepare.receipt")
    artifacts = prepare.get("artifacts")
    require(isinstance(artifacts, list) and artifacts, "prepare.artifacts must be non-empty")
    artifact_hashes: dict[str, str] = {}
    artifact_kinds: set[str] = set()
    for index, raw in enumerate(artifacts):
        artifact = mapping(raw, f"prepare.artifacts[{index}]")
        kind = artifact.get("kind")
        require(isinstance(kind, str) and kind, f"prepare.artifacts[{index}].kind is invalid")
        path, digest = validate_file_ref(root, artifact, f"prepare.artifacts[{index}]")
        relative = path.relative_to(root).as_posix()
        require(relative not in artifact_hashes, "prepare artifact path is duplicated")
        artifact_hashes[relative] = digest
        artifact_kinds.add(kind)
    require({"predictions", "positions"} <= artifact_kinds, "prepare must freeze predictions and positions")
    require(prepare.get("outcome_rows_read") == 0, "prepare read registered outcomes")
    require(prepare.get("outcome_artifacts_present") is False, "outcome artifact existed during prepare")
    blind_gates = mapping(prepare.get("outcome_blind_gates"), "prepare.outcome_blind_gates")
    require(set(blind_gates) == set(blind_gate_names), "prepare omitted or added an outcome-blind gate")
    require(all(value is True for value in blind_gates.values()), "an outcome-blind gate failed or is non-boolean")
    for key in ("predictions_frozen", "positions_frozen", "all_checkpoints_used_without_selection", "authorizes_unseal"):
        require(prepare.get(key) is True, f"prepare.{key} must be true")

    require(prepare_receipt.get("schema_version") == "tlm-one-shot-prepare/v1", "invalid prepare receipt schema")
    require(prepare_receipt.get("evaluation_spec_sha256") == spec_hash, "prepare receipt is not bound to evaluation spec")
    require(prepare_receipt.get("registered_sha256") == registered_hash, "prepare receipt is not bound to registered contract")
    require(prepare_receipt.get("artifact_hashes") == artifact_hashes, "prepare receipt artifact hashes drift")
    require(prepare_receipt.get("outcome_rows_read") == 0, "prepare receipt reports outcome access")
    require(prepare_receipt.get("outcome_blind_gates_passed") is True, "prepare receipt did not pass blind gates")
    require(prepare_receipt.get("authorizes_unseal") is True, "prepare receipt does not authorize unseal")

    safety = mapping(packet.get("safety"), "safety")
    target_assets = safety.get("target_assets_loaded")
    require(
        isinstance(target_assets, list)
        and all(isinstance(asset, str) and asset for asset in target_assets),
        "safety.target_assets_loaded must contain strings",
    )
    if live_status.get("target_asset_status") == "sealed":
        require(not target_assets, "live research state seals target assets")
    for key in ("retuning_performed", "thresholds_changed", "costs_or_accounting_changed", "second_unseal_attempted"):
        require(safety.get(key) is False, f"safety.{key} must be false")

    authorization = mapping(packet.get("authorization"), "authorization")
    unseal_data = packet.get("unseal")
    outcome_receipt_hash: str | None = None
    if phase == "prepare":
        require(authorization.get("explicit_user_authorization") is False, "prepare must precede explicit unseal authorization")
        require(authorization.get("exact_registered_unseal") is False, "prepare cannot claim an unseal")
        require(unseal_data is None, "prepare packet must not contain unseal artifacts")
        require(packet.get("completion") is None and packet.get("replay") is None, "prepare packet contains post-unseal state")
    else:
        require(authorization.get("explicit_user_authorization") is True, "exact unseal lacks explicit user authorization")
        require(authorization.get("exact_registered_unseal") is True, "authorization is not for the exact registered unseal")
        unseal = mapping(unseal_data, "unseal")
        auth_receipt, auth_hash = load_receipt(root, unseal.get("authorization_receipt"), "unseal.authorization_receipt")
        require(auth_receipt.get("schema_version") == "tlm-one-shot-unseal-authorization/v1", "invalid unseal authorization schema")
        require(auth_receipt.get("unseal_count") == 1, "unseal authorization count must equal one")
        require(auth_receipt.get("explicit_user_authorization") is True, "unseal receipt lacks explicit user authorization")
        require(auth_receipt.get("authorized_command") == state.get("authorized_command"), "unseal receipt command binding drift")
        require(auth_receipt.get("evaluation_spec_sha256") == spec_hash, "unseal authorization spec binding drift")
        require(auth_receipt.get("prepare_receipt_sha256") == prepare_hash, "unseal authorization prepare binding drift")
        require(auth_receipt.get("registered_sha256") == registered_hash, "unseal authorization registered binding drift")

        _, outcome_hash = validate_file_ref(root, unseal.get("outcome_packet"), "unseal.outcome_packet")
        outcome_receipt, outcome_receipt_hash = load_receipt(root, unseal.get("outcome_receipt"), "unseal.outcome_receipt")
        require(outcome_receipt.get("schema_version") == "tlm-one-shot-outcome/v1", "invalid outcome receipt schema")
        require(outcome_receipt.get("unseal_count") == 1, "outcome receipt unseal count must equal one")
        require(outcome_receipt.get("evaluation_spec_sha256") == spec_hash, "outcome receipt spec binding drift")
        require(outcome_receipt.get("prepare_receipt_sha256") == prepare_hash, "outcome receipt prepare binding drift")
        require(outcome_receipt.get("registered_sha256") == registered_hash, "outcome receipt registered binding drift")
        require(outcome_receipt.get("authorization_receipt_sha256") == auth_hash, "outcome receipt authorization binding drift")
        require(outcome_receipt.get("outcome_packet_sha256") == outcome_hash, "outcome packet binding drift")
        require(outcome_receipt.get("written_atomically") is True, "outcome packet was not written atomically")
        require(outcome_receipt.get("immutable") is True, "outcome packet is not declared immutable")

    completion_hash: str | None = None
    if phase in {"complete", "replay"}:
        completion, completion_hash = load_receipt(root, packet.get("completion"), "completion")
        require(completion.get("schema_version") == "tlm-one-shot-completion/v1", "invalid completion receipt schema")
        require(completion.get("evaluation_spec_sha256") == spec_hash, "completion spec binding drift")
        require(completion.get("prepare_receipt_sha256") == prepare_hash, "completion prepare binding drift")
        require(completion.get("registered_sha256") == registered_hash, "completion registered binding drift")
        require(completion.get("outcome_receipt_sha256") == outcome_receipt_hash, "completion outcome binding drift")
        require(completion.get("decision") in {"pass", "retire"}, "completion decision must be pass or retire")
        result_artifacts = completion.get("result_artifacts")
        require(isinstance(result_artifacts, dict) and result_artifacts, "completion result_artifacts must be non-empty")
        for relative, digest in result_artifacts.items():
            expected_hash(digest, f"completion.result_artifacts[{relative}]")
            path = inside(root, relative, f"completion.result_artifacts[{relative}]")
            require(path.is_file() and sha256_file(path) == digest, f"completion result artifact drift: {relative}")
    else:
        require(packet.get("completion") is None, f"{phase} packet cannot contain completion")

    if phase == "replay":
        replay = mapping(packet.get("replay"), "replay")
        require(replay.get("reused_existing_outcome_packet") is True, "replay did not reuse the immutable outcome packet")
        require(replay.get("new_unseal_receipts") == 0, "replay created another unseal receipt")
        require(replay.get("source_outcome_rows_read") == 0, "replay reread source outcomes")
        require(replay.get("result_hashes_match") is True, "replay result hashes drifted")
    else:
        require(packet.get("replay") is None, f"{phase} packet cannot contain replay state")

    return {
        "valid": True,
        "phase": phase,
        "evaluation_spec_sha256": spec_hash,
        "prepare_receipt_sha256": prepare_hash,
        "outcome_receipt_sha256": outcome_receipt_hash,
        "completion_receipt_sha256": completion_hash,
        "registered_cost_bps": costs,
        "outcomes_sealed": phase == "prepare",
        "unseal_count": 0 if phase == "prepare" else 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    args = parser.parse_args()
    try:
        root = args.repo_root.resolve()
        packet = json.loads(args.packet.read_text(encoding="utf-8"))
        result = validate_packet(root, mapping(packet, "packet"))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
