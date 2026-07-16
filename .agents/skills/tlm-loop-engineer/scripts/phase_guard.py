#!/usr/bin/env python3
"""Preflight one authorized TLM phase and verify its repository-produced gate."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


def find_root(start: Path) -> Path:
    for path in (start.resolve(), *start.resolve().parents):
        if (path / "AGENTS.md").is_file() and (path / ".agents" / "skills").is_dir():
            return path
    raise RuntimeError("not inside the TLM repository")


def normalize_phase(raw: str) -> tuple[str, int]:
    match = re.fullmatch(r"v?(\d+)(?:-?r(\d+))?", raw.strip(), re.IGNORECASE)
    if not match:
        raise ValueError(f"invalid phase: {raw}")
    number = int(match.group(1))
    revision = match.group(2)
    return (f"v{number}-r{revision}" if revision is not None else f"v{number}"), number


def run_governor(root: Path) -> dict[str, Any]:
    script = root / ".agents/skills/tlm-research-governor/scripts/govern.py"
    result = subprocess.run(
        [sys.executable, str(script), "--root", str(root), "--json"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"governor returned invalid JSON: {result.stdout!r}") from exc
    if result.returncode != 0:
        raise RuntimeError(f"governor blocked the loop: {payload}")
    return payload


def task_contract(root: Path, phase: str, phase_number: int) -> str:
    text = (root / "TASKS.md").read_text(encoding="utf-8")
    revision = phase.split("-", maxsplit=1)[1].upper() if "-" in phase else None
    suffix = rf"-{revision}" if revision else ""
    start = re.search(
        rf"^### V{phase_number}{suffix}\b.*$", text, re.MULTILINE
    )
    if not start:
        raise RuntimeError(f"TASKS.md has no V{phase_number} contract section")
    end = re.search(r"^#{1,3} \S", text[start.end() :], re.MULTILINE)
    stop = start.end() + end.start() if end else len(text)
    return text[start.start() : stop].strip()


def preflight(root: Path, phase: str, number: int) -> tuple[dict[str, Any], int]:
    governor = run_governor(root)
    requested_action = str(governor["authorized_next_action"])
    authorized_phase = str(governor["authorized_phase"])
    ready = governor.get("ready") is True and authorized_phase == phase
    contract = task_contract(root, phase, number) if authorized_phase == phase else None
    scope = governor["access_contract"]
    output = {
        "ready": ready,
        "requested_phase": phase,
        "authorized_phase": authorized_phase,
        "authorized_next_action": requested_action,
        "authorization_receipt": governor["authorization_receipt"],
        "active_family": governor["active_family"],
        "scope": scope,
        "task_contract": contract,
        "stop_after": f"{phase.upper()} gate; return to the governor",
        "git": governor["git"],
    }
    if authorized_phase != phase:
        output["error"] = f"{phase} is not authorized; only {authorized_phase} is authorized"
    return output, 0 if ready else 3


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def gate(root: Path, phase: str, number: int) -> tuple[dict[str, Any], int]:
    artifact_prefix = phase.replace("-", "_")
    artifact_dirs = [
        path
        for path in (root / "artifacts").iterdir()
        if path.is_dir()
        and path.name.startswith(f"{artifact_prefix}_")
        and (
            "-r" in phase
            or not re.match(rf"^v{number}_r\d+(?:_|$)", path.name)
        )
    ]
    result_paths = sorted(
        path / "result.json" for path in artifact_dirs if (path / "result.json").is_file()
    )
    audit_paths = sorted(
        path / "audit.json" for path in artifact_dirs if (path / "audit.json").is_file()
    )
    if len(result_paths) != 1 or len(audit_paths) != 1:
        output = {
            "passed": False,
            "phase": phase,
            "error": "expected exactly one V-phase result.json and audit.json",
            "result_paths": [str(path.relative_to(root)) for path in result_paths],
            "audit_paths": [str(path.relative_to(root)) for path in audit_paths],
        }
        return output, 4
    result = load_json(result_paths[0])
    audit = load_json(audit_paths[0])
    audit_passed = audit.get("passed") is True
    decision = result.get("decision")
    next_phase_number = number if "-r" in phase else number + 1
    next_action_is_explicit = isinstance(decision, str) and bool(
        re.match(rf"^(?:authorize|start)_v{next_phase_number}(?:_|$)", decision)
    )
    passed = audit_passed and next_action_is_explicit
    output = {
        "passed": passed,
        "phase": phase,
        "result": str(result_paths[0].relative_to(root)),
        "audit": str(audit_paths[0].relative_to(root)),
        "repository_audit_passed": audit_passed,
        "recorded_next_action": decision,
        "next_action_receipt_is_explicit": next_action_is_explicit,
        "scientific_metrics_recomputed_by_guard": False,
        "stop": f"Stop after {phase.upper()}; return to the governor.",
    }
    return output, 0 if passed else 4


def emit(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            print(f"{key}: {json.dumps(value, sort_keys=True)}")
        else:
            print(f"{key}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "gate"):
        current = subparsers.add_parser(command)
        current.add_argument("--phase", required=True)
        current.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    try:
        root = find_root(args.root)
        phase, number = normalize_phase(args.phase)
        if args.command == "preflight":
            payload, code = preflight(root, phase, number)
        else:
            payload, code = gate(root, phase, number)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"passed": False, "error": str(exc)}, indent=2))
        return 2
    emit(payload, args.as_json)
    return code


if __name__ == "__main__":
    sys.exit(main())
