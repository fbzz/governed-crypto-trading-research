from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil

import pytest
import yaml

from tlm.low_turnover_rank_spec import run_low_turnover_rank_spec


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/v79_v78_terminal_record"


def _config() -> dict:
    value = yaml.safe_load(
        (ROOT / "configs/v80_low_turnover_rank_spec.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(value, dict)
    return value


def test_v80_spec_freezes_one_compact_structurally_low_turnover_family() -> None:
    design = _config()["low_turnover_rank_spec"]["frozen_design"]
    architecture = design["architecture"]
    policy = design["policy"]
    chronology = design["chronology"]
    training = design["training"]
    evaluation = design["evaluation"]

    assert architecture["family"] == "shared_causal_depthwise_tcn_deepsets_ranker"
    assert architecture["expected_total_parameters"] == 10993
    assert architecture["architecture_variant_count"] == 1
    assert architecture["receptive_field_days"] == 127
    assert policy["decision_interval_days"] == 21
    assert policy["maximum_evaluation_decisions"] == 8
    assert policy["structural_maximum_turnover"] == 16.0
    assert chronology["consumed_2025_outcomes_role"] == "forbidden"
    assert chronology["final_evaluation_signal_start"] == "2026-01-01"
    assert chronology["final_evaluation_signal_end"] == "2026-06-09"
    assert chronology["final_evaluation_outcome_maturity_end"] == "2026-06-30"
    assert training["future_job_count"] == 9
    assert len(evaluation["mandatory_gates"]) == 9
    assert evaluation["aggregate_rescue_for_failed_fold"] is False


def test_v80_runner_is_metadata_only_and_byte_identical(tmp_path: Path) -> None:
    base = _config()
    config = deepcopy(base)
    section = config["low_turnover_rank_spec"]
    section["project_root"] = str(tmp_path)
    inputs = {
        "result": "result.json",
        "audit": "audit.json",
        "terminal_record": "terminal_record.json",
        "input_hash_receipt": "input_hash_receipt.json",
    }
    for name, destination in inputs.items():
        shutil.copyfile(SOURCE / f"{name}.json", tmp_path / destination)
    section["inputs"] = inputs
    section["source_receipt_files"] = ["source.py"]
    (tmp_path / "source.py").write_text("VALUE = 80\n", encoding="utf-8")
    config["output_dir"] = "output"

    first = run_low_turnover_rank_spec(config)
    output = tmp_path / "output"
    first_bytes = {path.name: path.read_bytes() for path in output.iterdir()}
    second = run_low_turnover_rank_spec(config)
    second_bytes = {path.name: path.read_bytes() for path in output.iterdir()}

    assert first["decision"] == (
        "authorize_v81_synthetic_low_turnover_rank_harness_only"
    )
    assert first["result"]["parameter_count"] == 10993
    assert first["result"]["future_training_jobs"] == 9
    assert first["result"]["structural_maximum_evaluation_turnover"] == 16.0
    assert first["audit"]["passed"] is True
    assert first["audit"]["checks_total"] == 14
    assert first["audit"]["access_ledger"]["json_metadata_reads"] == 4
    assert first["audit"]["access_ledger"]["parquet_deserializations"] == 0
    assert first["audit"]["access_ledger"]["model_instantiations"] == 0
    assert first["audit"]["access_ledger"]["outcome_packet_reads"] == 0
    assert first["audit"]["access_ledger"]["target_assets_loaded"] == []
    assert first_bytes == second_bytes


def test_v80_runner_rejects_architecture_or_turnover_drift(tmp_path: Path) -> None:
    config = _config()
    section = config["low_turnover_rank_spec"]
    section["project_root"] = str(ROOT)
    section["source_receipt_files"] = ["pyproject.toml"]
    config["output_dir"] = "artifacts/v80_drift_test_never_written"

    config["low_turnover_rank_spec"]["frozen_design"]["architecture"][
        "expected_total_parameters"
    ] = 10994
    with pytest.raises(ValueError, match="parameter_accounting_is_exact"):
        run_low_turnover_rank_spec(config)

    config = _config()
    section = config["low_turnover_rank_spec"]
    section["project_root"] = str(ROOT)
    section["source_receipt_files"] = ["pyproject.toml"]
    config["output_dir"] = "artifacts/v80_drift_test_never_written"
    config["low_turnover_rank_spec"]["frozen_design"]["policy"][
        "decision_interval_days"
    ] = 7
    with pytest.raises(ValueError, match="turnover_is_structurally_bounded"):
        run_low_turnover_rank_spec(config)
