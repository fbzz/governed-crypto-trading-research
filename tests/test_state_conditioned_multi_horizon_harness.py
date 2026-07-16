from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from tlm.core import canonical_sha256, file_sha256
from tlm.state_conditioned_multi_horizon_harness import (
    run_state_conditioned_multi_horizon_harness,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _fixture_config(tmp_path: Path) -> dict:
    v55 = yaml.safe_load(
        Path("configs/v55_state_conditioned_multi_horizon_spec.yaml").read_text(
            encoding="utf-8"
        )
    )["state_conditioned_multi_horizon_spec"]
    blueprint = {
        "version": "v55",
        "candidate_family_id": v55["candidate_family_id"],
        "architecture": v55["architecture"],
        "objective": v55["objective"],
        "policy": v55["policy"],
        "training": v55["training"],
    }
    blueprint["blueprint_sha256"] = canonical_sha256(blueprint)
    payloads = {
        "v55_result": {
            "decision": "authorize_v56_synthetic_state_policy_harness_only",
            "blueprint_sha256": blueprint["blueprint_sha256"],
        },
        "v55_blueprint": blueprint,
        "v55_audit": {"passed": True},
    }
    config = deepcopy(
        yaml.safe_load(
            Path("configs/v56_state_conditioned_multi_horizon_harness.yaml").read_text(
                encoding="utf-8"
            )
        )
    )
    harness = config["state_conditioned_multi_horizon_harness"]
    harness["project_root"] = str(tmp_path)
    source_path = tmp_path / "synthetic_source.py"
    source_path.write_text("VALUE = 56\n", encoding="utf-8")
    harness["source_receipt"]["files"] = [source_path.name]
    harness["expected_blueprint_sha256"] = blueprint["blueprint_sha256"]
    for name, payload in payloads.items():
        path = tmp_path / f"{name}.json"
        _write_json(path, payload)
        harness["inputs"][name] = path.name
        harness["expected_input_sha256"][name] = file_sha256(path)
    config["output_dir"] = "output"
    return config


def test_v56_harness_passes_emits_exact_packet_and_replays_bytes(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    first = run_state_conditioned_multi_horizon_harness(config)
    output = tmp_path / "output"
    first_bytes = {path.name: path.read_bytes() for path in output.iterdir()}
    second = run_state_conditioned_multi_horizon_harness(config)
    second_bytes = {path.name: path.read_bytes() for path in output.iterdir()}
    assert first["decision"] == "authorize_v57_non_target_multi_horizon_dataset_build_only"
    assert first["audit"]["passed"]
    assert first["smoke"]["parameter_count"] == 465_513
    assert first["smoke"]["resume_equivalent"]
    assert first["smoke"]["decision_indexes"] == [0, 7, 14]
    assert first["smoke"]["operation_ledger"]["parquet_deserializations"] == 0
    assert first["smoke"]["operation_ledger"]["target_asset_loads"] == 0
    assert set(first_bytes) == {
        "artifact_manifest.json",
        "audit.json",
        "checkpoint_metadata.json",
        "completion_receipt.json",
        "harness_spec.json",
        "input_hash_receipt.json",
        "report.md",
        "resolved_config.yaml",
        "result.json",
        "smoke.json",
        "source_receipt.json",
        "synthetic_checkpoint.pt",
    }
    assert first_bytes == second_bytes


def test_v56_harness_rejects_input_hash_drift_before_model_work(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    (tmp_path / "v55_result.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="input receipt mismatch"):
        run_state_conditioned_multi_horizon_harness(config)
    assert not (tmp_path / "output" / "synthetic_checkpoint.pt").exists()


def test_v56_harness_rejects_any_extra_runtime_input(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    harness = config["state_conditioned_multi_horizon_harness"]
    harness["inputs"]["real_panel"] = "panel.parquet"
    harness["expected_input_sha256"]["real_panel"] = "forbidden"
    with pytest.raises(ValueError, match="allowlist drift"):
        run_state_conditioned_multi_horizon_harness(config)
