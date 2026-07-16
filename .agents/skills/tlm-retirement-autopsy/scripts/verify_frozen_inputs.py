#!/usr/bin/env python3
"""Verify the complete hash-locked input allowlist without writing files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from lint_autopsy_contract import canonical_sha256, validate_contract


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(contract: dict[str, Any], root: Path) -> dict[str, Any]:
    contract_errors = validate_contract(contract)
    checks: dict[str, bool] = {}
    errors = list(contract_errors)
    if contract_errors:
        return {
            "passed": False,
            "contract_sha256": canonical_sha256(contract),
            "verified_inputs": 0,
            "checks": checks,
            "errors": errors,
        }
    root = root.resolve()
    if not root.is_dir():
        errors.append(f"repository root is not a directory: {root}")
    else:
        for name, receipt in sorted(contract.get("inputs", {}).items()):
            candidate = (root / receipt["path"]).resolve()
            try:
                candidate.relative_to(root)
                inside_root = True
            except ValueError:
                inside_root = False
            checks[f"{name}.inside_root"] = inside_root
            exists = inside_root and candidate.is_file()
            checks[f"{name}.regular_file"] = exists
            matches = exists and sha256_file(candidate) == receipt["sha256"]
            checks[f"{name}.sha256"] = matches
            if not inside_root:
                errors.append(f"{name}: resolved path escapes repository root")
            elif not exists:
                errors.append(f"{name}: input is missing or not a regular file")
            elif not matches:
                errors.append(f"{name}: SHA-256 drift")

        retirement_name = contract["retirement"]["input"]
        retirement_path = (
            root / contract["inputs"][retirement_name]["path"]
        ).resolve()
        try:
            with retirement_path.open(encoding="utf-8") as handle:
                retirement_payload = json.load(handle)
            decision_matches = (
                isinstance(retirement_payload, dict)
                and retirement_payload.get("decision")
                == contract["retirement"]["decision"]
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            decision_matches = False
        checks["retirement.decision_matches_input"] = decision_matches
        if not decision_matches:
            errors.append("retirement decision does not match its frozen input")

    return {
        "passed": not errors and all(checks.values()),
        "contract_sha256": canonical_sha256(contract),
        "verified_inputs": len(contract.get("inputs", {})),
        "checks": checks,
        "errors": errors,
    }


def _self_test(root: Path) -> dict[str, Any]:
    retirement = root.resolve() / "artifacts/v50_joint_absolute_relative_evaluation/result.json"
    relative = retirement.relative_to(root.resolve()).as_posix()
    digest = sha256_file(retirement)
    with retirement.open(encoding="utf-8") as handle:
        decision = json.load(handle)["decision"]
    return {
        "schema_version": "1",
        "family_id": "read_only_self_test",
        "retirement": {
            "input": "frozen_result",
            "decision": decision,
            "immutable": True,
        },
        "inputs": {"frozen_result": {"path": relative, "sha256": digest}},
        "diagnostics": {
            "signal": ["registered_signal_diagnostic"],
            "calibration": ["registered_calibration_diagnostic"],
            "churn": ["registered_churn_diagnostic"],
            "cost": ["registered_cost_diagnostic"],
        },
        "forbidden": {
            "counterfactual_pnl": True,
            "parameter_or_threshold_tuning": True,
            "model_training_or_inference": True,
            "new_bootstrap_or_cost_grid": True,
            "post_hoc_selection": True,
        },
        "outputs": ["artifacts/read_only_self_test/result.json"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", nargs="?", help="Path to contract JSON")
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)

    if args.self_test:
        try:
            contract = _self_test(root)
        except ValueError as exc:
            print(json.dumps({"passed": False, "errors": [str(exc)]}, sort_keys=True))
            return 2
    elif args.contract:
        try:
            with open(args.contract, encoding="utf-8") as handle:
                contract = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            print(json.dumps({"passed": False, "errors": [str(exc)]}, sort_keys=True))
            return 2
    else:
        parser.error("contract is required unless --self-test is used")

    result = verify(contract, root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
