from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest
import yaml

from tlm.ranking_excess_spec import (
    _sha256_file,
    analytic_parameter_count,
    build_ranking_excess_spec,
)


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture_config(tmp_path: Path) -> dict:
    real_config = yaml.safe_load(
        Path("configs/v41_ranking_excess_spec.yaml").read_text(encoding="utf-8")
    )
    config = deepcopy(real_config)
    spec = config["ranking_excess_spec"]
    spec["project_root"] = str(tmp_path)

    symbols = [f"A{index:02d}USDT" for index in range(30)]
    folds = []
    catalogs = []
    for fold in range(1, 4):
        start = (fold - 1) * 10
        test_symbols = symbols[start : start + 10]
        train_symbols = [symbol for symbol in symbols if symbol not in test_symbols]
        folds.append({
            "fold": fold,
            "train_symbols": train_symbols,
            "test_symbols": test_symbols,
        })
        catalogs.append({
            "fold": fold,
            "train_symbols": train_symbols,
            "test_symbols": test_symbols,
            "train_triplets": [None] * math.comb(20, 3),
            "test_triplets": [None] * math.comb(10, 3),
        })

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
            "tensor_contract": {
                "dtype": "float32",
                "x_shape": [256, 3, 9],
                "y_shape": [3, 2],
            },
        },
        "v32_feature_schema": {"model_feature_order": feature_order},
        "v32_asset_folds": {"fold_count": 3, "folds": folds},
        "v32_triplet_catalog": {"folds": catalogs},
        "v37_result": {
            "decision": "retire_candidate_family_without_target_evaluation"
        },
        "v37_gate": {"passed": False},
        "v37_autopsy_result": {
            "decision": "v37_retirement_confirmed_new_ex_ante_family_required",
            "recommendation": {
                "next_family_primary_change": (
                    "train_cross_sectional_ranking_or_excess_return_objective"
                )
            },
        },
        "v37_autopsy_audit": {"passed": True},
    }
    for name, payload in payloads.items():
        path = tmp_path / f"{name}.json"
        _write(path, payload)
        spec["inputs"][name] = path.name
        spec["expected_input_sha256"][name] = _sha256_file(path)
    config["output_dir"] = str(tmp_path / "output")
    return config


def test_v41_spec_freezes_one_medium_ranking_excess_family(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    result = build_ranking_excess_spec(config)
    assert result["decision"] == "authorize_v42_synthetic_ranking_excess_harness_only"
    assert result["audit"]["passed"]
    assert all(result["audit"]["checks"].values())
    assert result["blueprint"]["parameter_count_analytic"] == 1_231_634
    assert result["tested"]["model_instantiations"] == 0
    assert result["tested"]["target_asset_loads"] == 0


def test_analytic_parameter_count_matches_frozen_medium_contract() -> None:
    config = yaml.safe_load(
        Path("configs/v41_ranking_excess_spec.yaml").read_text(encoding="utf-8")
    )
    architecture = config["ranking_excess_spec"]["architecture"]
    assert analytic_parameter_count(architecture, input_features=9) == 1_231_634


def test_v41_spec_rejects_input_hash_drift(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    input_path = tmp_path / "v37_gate.json"
    _write(input_path, {"passed": True})
    with pytest.raises(RuntimeError, match="input missing or hash drifted"):
        build_ranking_excess_spec(config)
