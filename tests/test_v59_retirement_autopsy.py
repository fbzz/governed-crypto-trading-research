from __future__ import annotations

import json
from pathlib import Path

from tlm.core.artifacts import canonical_sha256, file_sha256
from tlm.v59_retirement_autopsy import (
    CONTRACT_SHA256,
    OUTPUT_ROOT,
    load_contract,
    preflight_v59_retirement_autopsy,
    run_v59_retirement_autopsy,
    verify_v59_retirement_autopsy,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/v59_retirement_autopsy_contract.json"


def test_v59_autopsy_contract_is_hash_locked_and_json_only() -> None:
    contract = load_contract(CONTRACT)

    assert canonical_sha256(contract) == CONTRACT_SHA256
    assert len(contract["inputs"]) == 9
    assert all(
        Path(receipt["path"]).suffix in {".json", ".md"}
        for receipt in contract["inputs"].values()
    )
    assert all(
        receipt["sha256"] == file_sha256(ROOT / receipt["path"])
        for receipt in contract["inputs"].values()
    )
    assert all(contract["forbidden"].values())


def test_v59_autopsy_preflight_has_zero_sensitive_access() -> None:
    result = preflight_v59_retirement_autopsy(CONTRACT, ROOT)

    assert result["audit"]["passed"] is True
    assert result["access_ledger"] == {
        "verified_inputs": 9,
        "source_outcome_reads": 0,
        "parquet_deserializations": 0,
        "checkpoint_loads": 0,
        "model_inference": 0,
        "target_assets_loaded": [],
    }


def test_v59_autopsy_materializes_all_registered_groups_and_replays() -> None:
    first = run_v59_retirement_autopsy(CONTRACT, ROOT)
    first_manifest = first["manifest"]["artifact_manifest_sha256"]
    first_result = first["result"]["result_sha256"]
    second = run_v59_retirement_autopsy(CONTRACT, ROOT)
    verified = verify_v59_retirement_autopsy(CONTRACT, ROOT)

    assert first["audit"]["passed"] is True
    assert second["files_rewritten"] == 0
    assert second["manifest"]["artifact_manifest_sha256"] == first_manifest
    assert second["result"]["result_sha256"] == first_result
    assert verified["passed"] is True
    assert verified["source_outcome_reads"] == 0
    assert verified["target_assets_loaded"] == []

    output = ROOT / OUTPUT_ROOT
    signal = json.loads((output / "signal_diagnostics.json").read_text())
    calibration = json.loads((output / "calibration_diagnostics.json").read_text())
    churn = json.loads((output / "churn_diagnostics.json").read_text())
    cost = json.loads((output / "cost_diagnostics.json").read_text())
    preserved_gates = sum(
        len(cells)
        for cells in (
            signal["registered_signal_gates"],
            calibration["registered_coverage_gates"],
            churn["registered_turnover_gates"],
            cost["registered_economic_gates"],
        )
    )
    assert preserved_gates == 700
    assert len(signal["registered_bootstrap_cells"]) == 108
    assert len(calibration["registered_predictive_cells"]) == 12
    assert len(churn["registered_candidate_position_cells"]) == 12
    assert len(cost["registered_aggregate_economic_cells"]) == 80
    assert len(cost["registered_fold_economic_cells"]) == 240
