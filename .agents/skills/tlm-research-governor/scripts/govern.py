#!/usr/bin/env python3
"""Project the validated TLM research state without opening research tables."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def find_root(start: Path) -> Path:
    for path in (start.resolve(), *start.resolve().parents):
        if (path / "AGENTS.md").is_file() and (path / "research/current.yaml").is_file():
            return path
    raise RuntimeError("not inside the TLM repository")


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def repo_status(root: Path) -> dict[str, Any]:
    env = dict(os.environ)
    prior = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{root / 'src'}{os.pathsep}{prior}" if prior else str(root / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tlm",
            "research-status",
            "--root",
            str(root),
            "--state",
            "research/current.yaml",
        ],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"repository research-status failed: {detail}")
    value = json.loads(result.stdout)
    if not isinstance(value, dict) or value.get("passed") is not True:
        raise RuntimeError("repository research-status did not pass")
    return value


def active_tasks_section(text: str) -> str:
    marker = re.search(r"^## Current research decision\s*$", text, re.I | re.M)
    if not marker:
        return "\n".join(text.splitlines()[:100])
    following = re.search(r"^## ", text[marker.end() :], re.M)
    end = marker.end() + following.start() if following else len(text)
    return text[marker.start() : end]


def git_state(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def current_contract(root: Path, status: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    current = load_yaml(root / str(status["state_path"]))
    experiment = load_yaml(root / str(status["experiment_path"]))
    return current, experiment


def phase_scope(
    root: Path,
    current: dict[str, Any],
    experiment: dict[str, Any],
    phase: str,
) -> dict[str, Any]:
    phase_reference = current.get("phase_contract")
    if isinstance(phase_reference, dict) and isinstance(
        phase_reference.get("path"), str
    ):
        phase_contract = load_yaml(root / phase_reference["path"])
        section = phase_contract.get("access_contract")
    else:
        section = experiment.get(f"{phase}_contract")
    if isinstance(section, dict):
        return {
            "mode": "exact_machine_contract",
            "allowed_inputs": list(section.get("allowed_inputs", [])),
            "allowed_capabilities": list(section.get("allowed_capabilities", [])),
            "required_checks": list(section.get("required_checks", [])),
            "forbidden_capabilities": list(section.get("forbidden_capabilities", [])),
            "output_dir": section.get("output_dir"),
        }
    return {
        "mode": "exact_machine_contract_missing",
        "allowed_inputs": [],
        "allowed_capabilities": [],
        "required_checks": [],
        "forbidden_capabilities": ["all work until the phase contract exists"],
        "output_dir": None,
    }


def build_report(root: Path) -> dict[str, Any]:
    status = repo_status(root)
    current, experiment = current_contract(root, status)
    family = str(status["active_family_id"])
    phase = str(status["authorized_phase"]).lower()
    drifts: list[str] = []

    agents = (root / "AGENTS.md").read_text(encoding="utf-8")
    tasks = active_tasks_section((root / "TASKS.md").read_text(encoding="utf-8"))
    status_md = (root / "STATUS.md").read_text(encoding="utf-8") if (root / "STATUS.md").is_file() else ""
    if "research/current.yaml" not in agents:
        drifts.append("AGENTS.md does not route current authorization to research/current.yaml")
    if family not in agents:
        drifts.append("AGENTS.md current-family boundary differs from machine state")
    if phase not in tasks.lower():
        drifts.append("TASKS current-decision section omits the authorized phase")
    if status_md:
        if family not in status_md:
            drifts.append("STATUS.md active family differs from machine state")
        if phase not in status_md.lower():
            drifts.append("STATUS.md omits the authorized phase")

    family_rows = current.get("families", [])
    active_row = next(
        (item for item in family_rows if isinstance(item, dict) and item.get("family_id") == family),
        {},
    )
    not_trained = active_row.get("trained") is False
    scope = phase_scope(root, current, experiment, phase)
    if scope["mode"] != "exact_machine_contract":
        drifts.append(f"experiment contract has no {phase}_contract")

    changes = git_state(root)
    return {
        "ready": not drifts,
        "repository": str(root),
        "state_path": status["state_path"],
        "experiment_contract": status["experiment_path"],
        "phase_contract": status.get("phase_contract_path"),
        "authorization_receipt": current.get("last_completed_result"),
        "completed_stage": current.get("last_completed_phase"),
        "active_family": {
            "kind": "family",
            "id": family,
            "status": status["active_family_status"],
        },
        "family_registry": {
            "families": status["family_count"],
            "trained": status["trained_family_count"],
            "retired": status["retired_family_count"],
        },
        "execution_objects": {
            "runs": {
                "kind": "one execution cell such as origin x geometry x fold x seed",
                "started_for_active_family": 0 if not_trained else "consult family artifacts",
            },
            "checkpoints": {
                "kind": "persisted trained state produced by one run",
                "created_for_active_family": 0 if not_trained else "consult family artifacts",
            },
            "evaluations": {
                "kind": "frozen protocol consuming predictions and authorized outcomes",
                "completed_for_active_family": 0 if not_trained else "consult family artifacts",
            },
            "candidate": {
                "kind": "family registered only after all preceding gates",
                "status": "not_registered",
            },
        },
        "authorized_next_action": status["authorized_next_action"],
        "authorized_phase": phase,
        "authorized_command": status.get("authorized_command"),
        "access_contract": scope,
        "target_asset_status": status["target_asset_status"],
        "forbidden_capabilities": status["forbidden_capabilities"],
        "documentation_drift": drifts,
        "git": {"clean": not changes, "changes": changes},
    }


def classify_path(raw: str, report: dict[str, Any]) -> tuple[bool, str]:
    root = Path(str(report["repository"]))
    path = Path(raw)
    if path.is_absolute():
        try:
            relative = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return False, "path escapes the repository"
    else:
        relative = path.as_posix().lstrip("./")
    lower = relative.lower()

    phase = str(report["authorized_phase"])
    exact_inputs = set(report["access_contract"].get("allowed_inputs", []))
    if relative in exact_inputs:
        return True, f"exact input registered by the {phase.upper()} contract"
    forbidden = set(report["access_contract"].get("forbidden_capabilities", []))
    forbidden_tokens = {
        "btcusdt": "target assets are sealed",
        "ethusdt": "target assets are sealed",
        "solusdt": "target assets are sealed",
    }
    if "parquet_deserialization" in forbidden:
        forbidden_tokens[".parquet"] = "Parquet deserialization is forbidden"
        forbidden_tokens["data/processed"] = "real processed data is forbidden"
        forbidden_tokens["data/raw"] = "real raw data is forbidden"
    if forbidden.intersection({"real_market_prediction", "market_prediction"}):
        forbidden_tokens["predictions"] = "real market predictions are forbidden"
    if forbidden.intersection(
        {"real_performance_metric_or_pnl", "performance_metric_or_pnl"}
    ):
        forbidden_tokens["daily_returns"] = "realized outcomes are forbidden"
    for token, reason in forbidden_tokens.items():
        if token in lower:
            return False, reason
    allowed_prefixes = (
        ".agents/",
        f"configs/{phase}_",
        "src/",
        "tests/",
        f"artifacts/{phase}_",
        "research/",
        "agents.md",
        "tasks.md",
        "status.md",
        "makefile",
    )
    if lower.startswith(allowed_prefixes):
        return True, f"{phase.upper()} implementation, test, contract, or output path"
    return False, f"path is not admitted by the {phase.upper()} contract or implementation allowlist"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--check-path", nargs="+")
    args = parser.parse_args()
    try:
        root = find_root(args.root)
        report = build_report(root)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError, yaml.YAMLError) as exc:
        print(json.dumps({"ready": False, "error": str(exc)}, indent=2))
        return 2

    if args.check_path:
        checks = []
        for path in args.check_path:
            allowed, reason = classify_path(path, report)
            checks.append({"path": path, "allowed": allowed, "reason": reason})
        print(json.dumps({"authorized_phase": report["authorized_phase"], "checks": checks}, indent=2, sort_keys=True))
        return 0 if report["ready"] and all(item["allowed"] for item in checks) else 3

    if args.as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Family: {report['active_family']['id']}")
        print(f"Completed stage: {report['completed_stage']}")
        print(f"Authorized next action: {report['authorized_next_action']}")
        print(f"Ready: {str(report['ready']).lower()}")
        for drift in report["documentation_drift"]:
            print(f"DRIFT: {drift}")
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    sys.exit(main())
