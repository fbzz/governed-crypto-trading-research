from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest
import yaml

from tlm.joint_absolute_relative_spec import (
    _sha256_file,
    analytic_joint_parameter_count,
    build_joint_absolute_relative_spec,
    run_joint_absolute_relative_spec,
)


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture_config(tmp_path: Path) -> dict:
    real_config = yaml.safe_load(
        Path("configs/v47_joint_absolute_relative_triplet_spec.yaml").read_text(
            encoding="utf-8"
        )
    )
    config = deepcopy(real_config)
    spec = config["joint_absolute_relative_spec"]
    spec["project_root"] = str(tmp_path)

    symbols = [f"A{index:02d}USDT" for index in range(30)]
    folds = []
    catalogs = []
    for fold_id in range(1, 4):
        start = (fold_id - 1) * 10
        test_symbols = symbols[start : start + 10]
        train_symbols = [symbol for symbol in symbols if symbol not in test_symbols]
        folds.append(
            {
                "fold": fold_id,
                "train_symbols": train_symbols,
                "test_symbols": test_symbols,
            }
        )
        catalogs.append(
            {
                "fold": fold_id,
                "train_symbols": train_symbols,
                "test_symbols": test_symbols,
                "train_triplets": [None] * math.comb(20, 3),
                "test_triplets": [None] * math.comb(10, 3),
            }
        )

    feature_order = [f"feature_{index}" for index in range(8)] + [
        "within_triplet_relative_strength"
    ]
    payloads = {
        "v32_result": {
            "decision": "authorize_v33_patch_transformer_implementation_only",
            "audit": {"passed": True},
        },
        "v32_audit": {"passed": True},
        "v32_dataset_manifest": {
            "symbol_count": 30,
            "symbols": symbols,
            "panel_sha256": spec["data_contract"]["panel_sha256"],
            "sequence_index_sha256": spec["data_contract"][
                "sequence_index_sha256"
            ],
            "tensor_contract": {
                "dtype": "float32",
                "x_shape": [256, 3, 9],
                "y_shape": [3, 2],
            },
        },
        "v32_feature_schema": {"model_feature_order": feature_order},
        "v32_asset_folds": {"fold_count": 3, "folds": folds},
        "v32_triplet_catalog": {"folds": catalogs},
        "v45_result": {
            "decision": "retire_family_without_target_evaluation_or_parameter_tuning"
        },
        "v45_gate": {"passed": False},
        "v45_audit": {"passed": True},
        "v46_failure_attribution": {
            "family_remains_retired": True,
            "relative_ranking_absolute_return_gap_observed": True,
        },
        "v46_audit": {"passed": True},
        "v46_completion_receipt": {
            "decision": "v45_retirement_confirmed_diagnostic_only"
        },
        "v46_autopsy_spec": {"constraints": {"v45_decision_mutable": False}},
    }
    for name, payload in payloads.items():
        path = tmp_path / f"{name}.json"
        _write(path, payload)
        spec["inputs"][name] = path.name
        spec["expected_input_sha256"][name] = _sha256_file(path)
    config["output_dir"] = str(tmp_path / "output")
    return config


def test_v47_freezes_joint_absolute_relative_family(tmp_path: Path) -> None:
    result = build_joint_absolute_relative_spec(_fixture_config(tmp_path))
    assert result["decision"] == (
        "authorize_v48_joint_absolute_relative_synthetic_harness_only"
    )
    assert result["audit"]["passed"]
    assert all(result["audit"]["checks"].values())
    assert result["blueprint"]["parameter_count_analytic"] == 1_212_930
    assert result["blueprint"]["registered_job_count"] == 36
    assert result["tested"]["parquet_deserializations"] == 0
    assert result["tested"]["model_instantiations"] == 0
    assert result["tested"]["target_asset_loads"] == 0


def test_v47_analytic_parameter_count_matches_frozen_contract() -> None:
    config = yaml.safe_load(
        Path("configs/v47_joint_absolute_relative_triplet_spec.yaml").read_text(
            encoding="utf-8"
        )
    )
    architecture = config["joint_absolute_relative_spec"]["architecture"]
    assert analytic_joint_parameter_count(architecture, 9) == 1_212_930
    assert analytic_joint_parameter_count(architecture, 9) < 1_231_634


def test_v47_rejects_input_hash_drift(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    _write(tmp_path / "v45_gate.json", {"passed": True})
    with pytest.raises(RuntimeError, match="input missing or hash drifted"):
        build_joint_absolute_relative_spec(config)


def test_v47_rejects_changed_grid_before_data_access(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    config["joint_absolute_relative_spec"]["training"]["seeds"] = [42]
    with pytest.raises(RuntimeError, match="specification audit failed"):
        build_joint_absolute_relative_spec(config)


def test_v47_replay_is_byte_identical(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    first = run_joint_absolute_relative_spec(config)
    output = Path(config["output_dir"])
    first_bytes = {path.name: path.read_bytes() for path in output.iterdir()}
    second = run_joint_absolute_relative_spec(config)
    second_bytes = {path.name: path.read_bytes() for path in output.iterdir()}
    assert first["blueprint_sha256"] == second["blueprint_sha256"]
    assert first_bytes == second_bytes
