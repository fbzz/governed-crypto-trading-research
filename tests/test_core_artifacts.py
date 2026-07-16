from __future__ import annotations

import json

import yaml

from tlm.core import (
    SyntheticAccessLedger,
    canonical_sha256,
    file_sha256,
    write_json_atomic,
    write_yaml_atomic,
)


def test_atomic_artifacts_are_deterministic(tmp_path) -> None:
    payload = {"b": [2, 1], "a": True}
    json_path = tmp_path / "packet.json"
    yaml_path = tmp_path / "packet.yaml"
    write_json_atomic(json_path, payload)
    first_hash = file_sha256(json_path)
    write_json_atomic(json_path, payload)
    assert file_sha256(json_path) == first_hash
    assert json.loads(json_path.read_text()) == payload
    write_yaml_atomic(yaml_path, payload)
    assert yaml.safe_load(yaml_path.read_text()) == payload
    assert canonical_sha256(payload) == canonical_sha256({"a": True, "b": [2, 1]})


def test_synthetic_ledger_rejects_any_forbidden_operation() -> None:
    ledger = SyntheticAccessLedger(authorized_metadata_reads=3)
    assert ledger.forbidden_operations_are_zero()
    ledger.target_asset_loads = 1
    assert not ledger.forbidden_operations_are_zero()
