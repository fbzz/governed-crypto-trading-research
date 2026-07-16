from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from tlm.persistent_duration_harness import run_persistent_duration_harness
from tlm.persistent_multi_horizon_duration_model import (
    PersistentMultiHorizonDurationTransformer,
)


def _registered_config() -> dict:
    return yaml.safe_load(
        Path("configs/v75_persistent_duration_harness.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_registered_v75_capacity_and_input_allowlist_are_exact() -> None:
    config = _registered_config()
    harness = config["persistent_duration_harness"]
    blueprint = yaml.safe_load(
        Path("artifacts/v74_persistent_duration_spec/blueprint.json").read_text(
            encoding="utf-8"
        )
    )
    model = PersistentMultiHorizonDurationTransformer(blueprint["architecture"])
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_083_155
    assert set(harness["inputs"]) == {
        "v74_specification",
        "v74_blueprint",
        "v74_audit",
        "v74_result",
        "v74_artifact_manifest",
        "v74_source_receipt",
    }
    assert all(
        Path(path).suffix == ".json" for path in harness["inputs"].values()
    )


def test_v75_rejects_input_hash_drift_before_synthetic_execution(
    tmp_path: Path,
) -> None:
    config = deepcopy(_registered_config())
    config["output_dir"] = str(tmp_path / "output")
    config["persistent_duration_harness"]["expected_input_sha256"][
        "v74_result"
    ] = "0" * 64
    with pytest.raises(ValueError, match="input receipt mismatch"):
        run_persistent_duration_harness(config)
    assert not (tmp_path / "output" / "synthetic_checkpoint.pt").exists()


def test_v75_rejects_any_extra_runtime_input(tmp_path: Path) -> None:
    config = deepcopy(_registered_config())
    config["output_dir"] = str(tmp_path / "output")
    harness = config["persistent_duration_harness"]
    harness["inputs"]["real_panel"] = "data/real.parquet"
    harness["expected_input_sha256"]["real_panel"] = "forbidden"
    with pytest.raises(ValueError, match="allowlist drift"):
        run_persistent_duration_harness(config)
