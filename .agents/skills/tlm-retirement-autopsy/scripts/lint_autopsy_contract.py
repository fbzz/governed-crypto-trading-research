#!/usr/bin/env python3
"""Lint a frozen TLM retirement-autopsy contract without writing files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import PurePosixPath
from typing import Any


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TOP_LEVEL_KEYS = {
    "schema_version",
    "family_id",
    "retirement",
    "inputs",
    "diagnostics",
    "forbidden",
    "outputs",
}
DIAGNOSTIC_AXES = {"signal", "calibration", "churn", "cost"}
FORBIDDEN_KEYS = {
    "counterfactual_pnl",
    "parameter_or_threshold_tuning",
    "model_training_or_inference",
    "new_bootstrap_or_cost_grid",
    "post_hoc_selection",
}


def _relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and value != "."


def _nonempty_unique_strings(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and item.strip() for item in value)
        and len(value) == len(set(value))
    )


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_contract(contract: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(contract, dict):
        return ["contract must be a JSON object"]

    keys = set(contract)
    if keys != TOP_LEVEL_KEYS:
        errors.append(
            f"top-level keys must be exactly {sorted(TOP_LEVEL_KEYS)}; "
            f"missing={sorted(TOP_LEVEL_KEYS - keys)} extra={sorted(keys - TOP_LEVEL_KEYS)}"
        )
    if contract.get("schema_version") != "1":
        errors.append("schema_version must equal string '1'")
    family_id = contract.get("family_id")
    if not isinstance(family_id, str) or not family_id.strip():
        errors.append("family_id must be a non-empty string")

    inputs = contract.get("inputs")
    input_paths: list[str] = []
    if not isinstance(inputs, dict) or not inputs:
        errors.append("inputs must be a non-empty logical-name mapping")
        inputs = {}
    for name, receipt in inputs.items():
        if not isinstance(name, str) or not name.strip():
            errors.append("every input logical name must be a non-empty string")
            continue
        if not isinstance(receipt, dict) or set(receipt) != {"path", "sha256"}:
            errors.append(f"inputs.{name} must contain exactly path and sha256")
            continue
        path = receipt.get("path")
        digest = receipt.get("sha256")
        if not _relative_path(path):
            errors.append(f"inputs.{name}.path must be a safe repository-relative path")
        else:
            input_paths.append(path)
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            errors.append(f"inputs.{name}.sha256 must be lowercase SHA-256")
    if len(input_paths) != len(set(input_paths)):
        errors.append("input paths must be unique")

    retirement = contract.get("retirement")
    if not isinstance(retirement, dict) or set(retirement) != {
        "input",
        "decision",
        "immutable",
    }:
        errors.append("retirement must contain exactly input, decision, and immutable")
    else:
        source = retirement.get("input")
        if source not in inputs:
            errors.append("retirement.input must name an allowlisted input")
        decision = retirement.get("decision")
        if not isinstance(decision, str) or "retire" not in decision.lower():
            errors.append("retirement.decision must preserve an explicit retirement decision")
        if retirement.get("immutable") is not True:
            errors.append("retirement.immutable must be true")

    diagnostics = contract.get("diagnostics")
    if not isinstance(diagnostics, dict) or set(diagnostics) != DIAGNOSTIC_AXES:
        errors.append(
            "diagnostics must contain exactly signal, calibration, churn, and cost"
        )
    else:
        for axis in sorted(DIAGNOSTIC_AXES):
            if not _nonempty_unique_strings(diagnostics.get(axis)):
                errors.append(f"diagnostics.{axis} must be a non-empty unique string list")

    forbidden = contract.get("forbidden")
    if not isinstance(forbidden, dict) or set(forbidden) != FORBIDDEN_KEYS:
        errors.append(f"forbidden must contain exactly {sorted(FORBIDDEN_KEYS)}")
    else:
        for key in sorted(FORBIDDEN_KEYS):
            if forbidden.get(key) is not True:
                errors.append(f"forbidden.{key} must be true")

    outputs = contract.get("outputs")
    if not _nonempty_unique_strings(outputs):
        errors.append("outputs must be a non-empty unique string list")
    else:
        for path in outputs:
            if not _relative_path(path):
                errors.append("every output must be a safe repository-relative path")
            if path in input_paths:
                errors.append(f"output collides with frozen input: {path}")
    return errors


def _self_test_contract() -> dict[str, Any]:
    digest = "0" * 64
    return {
        "schema_version": "1",
        "family_id": "example_family",
        "retirement": {
            "input": "evaluation_result",
            "decision": "retire_family_without_tuning",
            "immutable": True,
        },
        "inputs": {
            "evaluation_result": {
                "path": "artifacts/example/result.json",
                "sha256": digest,
            }
        },
        "diagnostics": {
            "signal": ["registered_rank_metrics"],
            "calibration": ["registered_calibration_metrics"],
            "churn": ["registered_transition_metrics"],
            "cost": ["registered_gross_cost_net_decomposition"],
        },
        "forbidden": {key: True for key in sorted(FORBIDDEN_KEYS)},
        "outputs": ["artifacts/example_autopsy/result.json"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", nargs="?", help="Path to contract JSON")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        contract = _self_test_contract()
    elif args.contract:
        try:
            with open(args.contract, encoding="utf-8") as handle:
                contract = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            print(json.dumps({"passed": False, "errors": [str(exc)]}, sort_keys=True))
            return 2
    else:
        parser.error("contract is required unless --self-test is used")

    errors = validate_contract(contract)
    result = {
        "passed": not errors,
        "contract_sha256": canonical_sha256(contract),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
