from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import yaml

from tlm.core.artifacts import canonical_sha256, file_sha256
from tlm.state_conditioned_multi_horizon_evaluation import (
    PREDICTION_COLUMNS,
    POSITION_COLUMNS,
    CONTROL_POSITION_COLUMNS,
    _accounting_fields,
    _decision_schedule,
    _position_frame,
    _prediction_batch_frame,
    _state_conditioned_positions,
)
from tlm.state_conditioned_multi_horizon_evaluation_artifacts import (
    PREPARE_DECISION,
    PREPARE_SCHEMA,
    REQUIRED_PREPARE_FILES,
    V59_PHASE_CONTRACT_CANONICAL_SHA256,
    V59_PHASE_FILE_SHA256,
    V59_SOURCE_FILES,
    V59PrepareError,
    _prediction_drivers,
    _registered_access_receipts,
    _state_policy,
    _verify_episode_rows,
    build_prepare_manifest,
    registered_projection,
    verify_prepare_packet,
    with_self_hash,
)
from tlm.state_conditioned_multi_horizon_evaluation_data import (
    BASE_FEATURE_COLUMNS,
    EvaluationCell,
    classify_development_samples,
    read_cell_data,
    scaler_from_wrapper,
)


ROOT = Path(__file__).resolve().parents[1]


def _contract() -> dict:
    return yaml.safe_load((ROOT / "research/phase_contracts/v059.yaml").read_text())


def test_v59_config_is_thin_and_binds_unseal_stage() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/v59_state_conditioned_multi_horizon_evaluation.yaml").read_text()
    )["state_conditioned_multi_horizon_evaluation"]
    assert config["version"] == "v59"
    assert config["phase_contract"] == "research/phase_contracts/v059_unseal_r1.yaml"
    assert config["base_phase_contract"] == "research/phase_contracts/v059.yaml"
    assert config["experiment_contract"] == "research/experiments/v059_prepare.yaml"
    assert config["output_dir"] == "artifacts/v59_state_conditioned_multi_horizon_evaluation"
    assert config["require_clean_git"] is True
    assert not {
        "policy",
        "controls",
        "accounting",
        "bootstrap",
        "gates",
        "inference",
    }.intersection(config)


def test_v59_projected_reader_materializes_only_train_h7_labels() -> None:
    train_symbols = ("AUSDT", "BUSDT", "CUSDT")
    test_symbols = ("DUSDT", "EUSDT", "FUSDT")
    train_date = pd.Timestamp("2023-01-01", tz="UTC")
    development_date = pd.Timestamp("2024-01-01", tz="UTC")
    development_start = development_date - pd.Timedelta(days=255)
    cell = EvaluationCell(
        origin="origin_2024",
        geometry="expanding",
        fold=1,
        train_symbols=train_symbols,
        test_symbols=test_symbols,
        train_triplets=(train_symbols,),
        test_triplets=(test_symbols,),
        train_flag="eligible_origin_2024_expanding_train",
        development_flag="eligible_origin_2024_expanding_development_evaluation",
        train_start=train_date,
        train_end=train_date,
        development_start=development_date,
        development_end=development_date,
    )
    sequence_rows = []
    for symbol in train_symbols:
        sequence_rows.append(
            {
                "date": train_date,
                "sequence_start_date": train_date - pd.Timedelta(days=255),
                "symbol": symbol,
                cell.train_flag: True,
                cell.development_flag: False,
            }
        )
    for symbol in test_symbols:
        sequence_rows.append(
            {
                "date": development_date,
                "sequence_start_date": development_start,
                "symbol": symbol,
                cell.train_flag: False,
                cell.development_flag: True,
            }
        )
    sequence = pd.DataFrame(sequence_rows)
    labels = pd.DataFrame(
        {
            "date": [train_date] * 3,
            "symbol": list(train_symbols),
            "target_h7_maturity_date": [train_date + pd.Timedelta(days=8)] * 3,
            "target_h7_open_to_open_log_return": np.asarray([0.01, 0.02, 0.03], dtype=np.float64),
            "multi_horizon_label_complete": [True, True, True],
        }
    )
    panel_rows = []
    for symbol in train_symbols:
        row = {"date": train_date, "symbol": symbol}
        row.update({name: 0.1 for name in BASE_FEATURE_COLUMNS})
        panel_rows.append(row)
    for date in pd.date_range(development_start, development_date, freq="D", tz="UTC"):
        for symbol in test_symbols:
            row = {"date": date, "symbol": symbol}
            row.update({name: 0.2 for name in BASE_FEATURE_COLUMNS})
            panel_rows.append(row)
    panel = pd.DataFrame(panel_rows)
    requests = []

    def reader(path: str, **kwargs: object) -> pd.DataFrame:
        requests.append((path, deepcopy(kwargs)))
        source = {"sequence": sequence, "labels": labels, "panel": panel}[path]
        return source.loc[:, kwargs["columns"]].copy()

    data = read_cell_data(
        cell,
        sequence_path="sequence",
        labels_path="labels",
        panel_path="panel",
        reader=reader,
    )
    assert len(requests) == 3
    assert all(request[1]["engine"] == "pyarrow" for request in requests)
    assert all(request[1]["filters"] for request in requests)
    assert list(data.train_labels.columns) == [
        "date",
        "symbol",
        "target_h7_maturity_date",
        "target_h7_open_to_open_log_return",
        "multi_horizon_label_complete",
    ]
    assert data.access_receipt["projected_columns"]["development_labels"] == []
    assert data.access_receipt["development_outcome_value_reads"] == 0
    assert data.access_receipt["full_table_materializations"] == 0
    assert data.access_receipt["target_asset_loads"] == 0
    assert len(data.development_context_keys) == 256 * 3
    data.development_availability[development_date] = test_symbols[:2]
    available, unavailable = classify_development_samples(data)
    assert available == []
    assert unavailable == [
        {
            "date": "2024-01-01",
            "triplet_key": "DUSDT|EUSDT|FUSDT",
            "reason": "missing_registered_sequence_member",
        }
    ]


def test_v59_scaler_uses_raw_triplet_relative_feature_without_mean_subtraction() -> None:
    scaler = {
        "origin": "origin_2024",
        "geometry": "expanding",
        "fold": 1,
        "feature_names": list(BASE_FEATURE_COLUMNS),
        "fit_symbols": ["AUSDT"],
        "fit_symbol_count": 1,
        "fit_unique_symbol_date_count": 1,
        "fit_min_date": "2022-01-01",
        "fit_max_date": "2022-01-01",
        "mean": [1.0] * 8,
        "standard_deviation": [2.0] * 8,
        "zero_scale_replacements": 0,
    }
    scaler["scaler_sha256"] = canonical_sha256(scaler)
    wrapper = {
        "version": "v58_train_only_scaler_v1",
        "scaler_id": "origin_2024|expanding|1",
        "scaler": scaler,
    }
    resolved = scaler_from_wrapper(wrapper)
    raw = np.ones((1, 1, 3, 8), dtype=np.float64)
    raw[0, 0, :, 1] = [1.0, 3.0, 8.0]
    transformed = resolved.transform(raw)
    expected_relative = (np.asarray([1.0, 3.0, 8.0]) - 4.0) / 2.0
    np.testing.assert_allclose(transformed[0, 0, :, 8], expected_relative)
    assert abs(float(transformed[0, 0, :, 8].sum())) < 1.0e-12


def test_v59_weekly_policy_freezes_clock_ties_forced_cash_and_liquidation() -> None:
    eligible = np.ones(15, dtype=bool)
    np.testing.assert_array_equal(
        np.flatnonzero(_decision_schedule(eligible)), np.asarray([0, 7, 14])
    )
    forecasts = np.zeros((15, 3), dtype=np.float64)
    forecasts[:, 0] = 0.02
    policy = _state_conditioned_positions(forecasts, eligible)
    np.testing.assert_allclose(policy["weights"][:, 0], 1.0 / 3.0)
    accounting = _accounting_fields(policy["weights"])
    assert accounting["base_turnover"][0] == pytest.approx(1.0 / 3.0)
    assert accounting["final_liquidation_turnover"][-1] == pytest.approx(1.0 / 3.0)
    assert accounting["post"][-1].sum() == 0.0

    entry_tie = np.full((1, 3), 0.001, dtype=np.float64)
    tied = _state_conditioned_positions(entry_tie, np.asarray([True]))
    assert tied["weights"].sum() == 0.0

    missing = eligible.copy()
    missing[1] = False
    forced = _state_conditioned_positions(forecasts, missing)
    assert forced["forced"][1]
    assert forced["weights"][1].sum() == 0.0


def test_v59_prediction_schema_freezes_27_seed_and_9_ensemble_values() -> None:
    cell = EvaluationCell(
        "origin_2024",
        "expanding",
        1,
        ("A", "B", "C"),
        ("D", "E", "F"),
        (("A", "B", "C"),),
        (("D", "E", "F"),),
        "train",
        "development",
        pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2022-01-01", tz="UTC"),
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-01-01", tz="UTC"),
    )
    samples = [(pd.Timestamp("2024-01-01", tz="UTC"), ("D", "E", "F"))]
    outputs = [np.full((1, 3, 3, 3), seed, dtype=np.float64) for seed in (42, 7, 123)]
    ensemble = (outputs[0] + outputs[1] + outputs[2]) / 3
    frame = _prediction_batch_frame(
        cell,
        samples,
        outputs,
        ensemble,
        np.ones((1, 3)),
        np.zeros((1, 3)),
    )
    assert list(frame.columns) == list(PREDICTION_COLUMNS)
    assert len(frame) == 3
    assert len([name for name in frame if name.startswith("seed_")]) == 27
    assert len([name for name in frame if name.startswith("ensemble_")]) == 9


def test_v59_independent_prediction_verifier_rejects_seed_mean_mutation(
    tmp_path: Path,
) -> None:
    cell = EvaluationCell(
        "origin_2024",
        "expanding",
        1,
        ("A", "B", "C"),
        ("D", "E", "F"),
        (("A", "B", "C"),),
        (("D", "E", "F"),),
        "train",
        "development",
        pd.Timestamp("2023-01-01", tz="UTC"),
        pd.Timestamp("2023-01-01", tz="UTC"),
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-01-01", tz="UTC"),
    )
    samples = [(pd.Timestamp("2024-01-01", tz="UTC"), ("D", "E", "F"))]
    outputs = [np.full((1, 3, 3, 3), seed, dtype=np.float64) for seed in (42, 7, 123)]
    frame = _prediction_batch_frame(
        cell,
        samples,
        outputs,
        (outputs[0] + outputs[1] + outputs[2]) / 3,
        np.ones((1, 3)),
        np.zeros((1, 3)),
    )
    path = tmp_path / "predictions.parquet"
    frame.to_parquet(path, index=False)
    states = {
        "origin_2024|expanding|1": {
            "dates": pd.date_range("2024-01-01", "2024-01-01", tz="UTC"),
            "availability": {"D|E|F": np.asarray([True])},
            "final_raw_by_key": {
                (pd.Timestamp("2024-01-01", tz="UTC"), symbol): np.zeros(8)
                for symbol in ("D", "E", "F")
            },
        }
    }
    linear_states = {
        "origin_2024|expanding|1": {
            "mean": np.zeros(8),
            "scale": np.ones(8),
            "coefficient": np.zeros(9),
            "intercept": 1.0,
            "residual_q20": -1.0,
        }
    }
    drivers = _prediction_drivers(
        path, states, linear_states, 3, {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
    )
    assert drivers[("origin_2024|expanding|1", "D|E|F")]["present"].all()

    frame.loc[0, "ensemble_h7_q20"] += 1.0e-6
    frame.to_parquet(path, index=False)
    with pytest.raises(V59PrepareError, match="seed aggregation"):
        _prediction_drivers(
            path,
            states,
            linear_states,
            3,
            {"BTCUSDT", "ETHUSDT", "SOLUSDT"},
        )


def test_v59_independent_episode_verifier_rejects_turnover_mutation() -> None:
    dates = pd.date_range("2024-01-01", periods=2, tz="UTC")
    cell = EvaluationCell(
        "origin_2024",
        "expanding",
        1,
        ("A", "B", "C"),
        ("D", "E", "F"),
        (("A", "B", "C"),),
        (("D", "E", "F"),),
        "train",
        "development",
        pd.Timestamp("2023-01-01", tz="UTC"),
        pd.Timestamp("2023-01-01", tz="UTC"),
        dates[0],
        dates[-1],
    )
    eligible = np.asarray([True, True])
    forecasts = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    expected = _state_policy(forecasts, eligible)
    frame = _position_frame(
        cell,
        dates,
        ("D", "E", "F"),
        eligible,
        expected,
    )
    _verify_episode_rows(frame, dates, "D|E|F", eligible, expected)
    frame.loc[0, "turnover"] = 0.0
    with pytest.raises(V59PrepareError, match="turnover"):
        _verify_episode_rows(frame, dates, "D|E|F", eligible, expected)


def _write_minimal_v2_packet(directory: Path, contract: dict) -> None:
    projection = registered_projection(contract)
    source_files = {relative: file_sha256(ROOT / relative) for relative in V59_SOURCE_FILES}
    source = with_self_hash(
        {
            "schema_version": "v59-source-receipt/v1",
            "git_clean": True,
            "git_head": "0" * 40,
            "files": source_files,
            "bundle_sha256": canonical_sha256(source_files),
        },
        "source_receipt_sha256",
    )
    registered_inputs = contract["input_contract"]["expected_file_sha256_by_path"]
    expected_access_receipts, expected_outcome_keys, _ = _registered_access_receipts(
        ROOT, contract, registered_inputs
    )
    input_receipt = with_self_hash(
        {
            "schema_version": "v59-input-hash-receipt/v1",
            "files": registered_inputs,
            "file_count": len(registered_inputs),
            "development_outcome_value_reads": 0,
            "target_asset_loads": 0,
        },
        "input_hash_receipt_sha256",
    )
    spec = with_self_hash(
        {
            "schema_version": "v59-evaluation-spec/v1",
            "phase_contract_file_sha256": V59_PHASE_FILE_SHA256,
            "phase_contract_canonical_sha256": V59_PHASE_CONTRACT_CANONICAL_SHA256,
            "registered_projection": projection,
            "registered_projection_sha256": canonical_sha256(projection),
            "source_receipt_sha256": source["source_receipt_sha256"],
            "input_hash_receipt_sha256": input_receipt[
                "input_hash_receipt_sha256"
            ],
            "prediction_schema": {"columns": list(PREDICTION_COLUMNS)},
            "position_schema": {
                "candidate_columns": list(POSITION_COLUMNS),
                "control_columns": list(CONTROL_POSITION_COLUMNS),
            },
        },
        "evaluation_spec_sha256",
    )
    outcome_keys = [
        {
            "origin": origin,
            "fold": fold,
            "date": date,
            "symbol": symbol,
        }
        for (origin, fold), records in sorted(expected_outcome_keys.items())
        for date, symbol in records
    ]
    outcome_primary = [
        (row["origin"], row["fold"], row["date"], row["symbol"])
        for row in outcome_keys
    ]
    outcome_groups = [
        {
            "origin": origin,
            "fold": fold,
            "key_count": len(records),
            "key_sha256": canonical_sha256(
                [(origin, fold, date, symbol) for date, symbol in records]
            ),
            "development_sequence_key_sha256": canonical_sha256(records),
        }
        for (origin, fold), records in sorted(expected_outcome_keys.items())
    ]
    outcome = with_self_hash(
        {
            "schema_version": "v59-outcome-request/v1",
            "allowed_columns": contract["one_shot_contract"]["unseal"]["allowed_columns"],
            "keys": outcome_keys,
            "groups": outcome_groups,
            "group_count": 6,
            "key_count": len(outcome_keys),
            "key_sha256": canonical_sha256(outcome_primary),
        },
        "outcome_request_sha256",
    )
    checkpoint_manifest = json.loads(
        (ROOT / "artifacts/v58_state_conditioned_multi_horizon_training/checkpoint_manifest.json").read_text()
    )
    checkpoint_entries = [
        {
            "job_id": row["job_id"],
            "origin": row["origin"],
            "geometry": row["geometry"],
            "fold": row["fold"],
            "seed": row["seed"],
            "path": row["checkpoint_path"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "semantic_checkpoint_sha256": row["semantic_checkpoint_sha256"],
            "best_model_state_sha256": row["best_model_state_sha256"],
            "load_count": 1,
            "selected": False,
            "weight": None,
            "checkpoint_state": "best_model_state",
            "optimizer_steps": 0,
        }
        for row in checkpoint_manifest["jobs"]
    ]
    checkpoint_binding = with_self_hash(
        {
            "schema_version": "v59-checkpoint-binding/v1",
            "entries": checkpoint_entries,
            "entry_count": 36,
        },
        "checkpoint_binding_sha256",
    )
    cells = [
        f"{origin}|{geometry}|{fold}"
        for origin in ("origin_2024", "origin_2025")
        for geometry in ("expanding", "rolling")
        for fold in (1, 2, 3)
    ]
    scaler_manifest = json.loads(
        (ROOT / "artifacts/v58_state_conditioned_multi_horizon_training/scaler_manifest.json").read_text()
    )
    scaler_by_cell = {
        f"{row['origin']}|{row['geometry']}|{row['fold']}": row
        for row in scaler_manifest["scalers"]
    }
    scaler_binding = with_self_hash(
        {
            "schema_version": "v59-scaler-binding/v1",
            "entries": [
                {
                    "cell_id": cell,
                    "path": (
                        "data/checkpoints/v58_state_conditioned_multi_horizon_training/"
                        f"{cell.split('|')[0]}/{cell.split('|')[1]}/"
                        f"fold_{cell.split('|')[2]}/scaler.json"
                    ),
                    "file_sha256": contract["input_contract"]
                    ["expected_scaler_file_sha256_by_path"][
                        "data/checkpoints/v58_state_conditioned_multi_horizon_training/"
                        f"{cell.split('|')[0]}/{cell.split('|')[1]}/"
                        f"fold_{cell.split('|')[2]}/scaler.json"
                    ],
                    "scaler_sha256": scaler_by_cell[cell]["scaler_sha256"],
                    "fit_symbols": scaler_by_cell[cell]["fit_symbols"],
                    "load_count": 1,
                    "fit_refit_count": 0,
                }
                for cell in cells
            ],
            "entry_count": 12,
        },
        "scaler_binding_sha256",
    )
    linear_binding = with_self_hash(
        {
            "schema_version": "v59-linear-control-receipt/v1",
            "entries": [
                {
                    "cell_id": cell,
                    "fit_scope": "exact_origin_geometry_fold_train_role_only",
                    "validation_or_development_fit_rows": 0,
                    "development_outcome_value_reads": 0,
                    "state": {
                        "coefficient": [0.0] * 9,
                        "intercept": 0.0,
                        "residual_q20": 0.0,
                    },
                    "state_sha256": canonical_sha256(
                        {
                            "coefficient": [0.0] * 9,
                            "intercept": 0.0,
                            "residual_q20": 0.0,
                        }
                    ),
                }
                for cell in cells
            ],
            "entry_count": 12,
        },
        "linear_control_receipt_sha256",
    )
    behavior_checks = {
        name: True for name in contract["outcome_blind_gate_contract"]["gates"]
    }
    data_access_receipts = [
        with_self_hash(expected_access_receipts[cell], "access_receipt_sha256")
        for cell in sorted(expected_access_receipts)
    ]
    behavior = with_self_hash(
        {
            "schema_version": "v59-behavior-audit/v1",
            "passed": True,
            "checks": behavior_checks,
            "operation_ledger": {
                "checkpoint_loads": 36,
                "scaler_loads": 12,
                "linear_control_fits": 12,
                "optimizer_steps": 0,
                "development_outcome_value_reads": 0,
                "target_asset_loads": 0,
                "performance_metrics_computed": 0,
                "pnl_evaluations": 0,
                "prediction_rows": 1,
                "candidate_position_rows": 1,
                "control_position_rows": 1,
            },
            "data_access_receipts": data_access_receipts,
        },
        "behavior_audit_sha256",
    )
    payloads = {
        "evaluation_spec.json": spec,
        "source_receipt.json": source,
        "input_hash_receipt.json": input_receipt,
        "checkpoint_binding.json": checkpoint_binding,
        "scaler_binding.json": scaler_binding,
        "linear_control_receipt.json": linear_binding,
        "outcome_request.json": outcome,
        "behavior_audit.json": behavior,
    }
    for name, value in payloads.items():
        (directory / name).write_text(json.dumps(value, sort_keys=True) + "\n")
    def frame_for(columns: tuple[str, ...]) -> pd.DataFrame:
        row = {}
        for name in columns:
            if name == "date":
                row[name] = pd.Timestamp("2024-01-01", tz="UTC")
            elif name in {"fold", "asset_slot"}:
                row[name] = 1
            elif name in {
                "available",
                "decision",
                "forced_cash",
                "final_liquidation",
            }:
                row[name] = False
            elif name.startswith("weight_") or name.startswith("post_event_weight_") or name.endswith("turnover") or name.startswith("seed_") or name.startswith("ensemble_") or name.startswith("linear_"):
                row[name] = 0.0
            else:
                row[name] = "x"
        return pd.DataFrame([row], columns=columns)

    parquet_metadata = {}
    for name, columns in (
        ("predictions.parquet", PREDICTION_COLUMNS),
        ("candidate_positions.parquet", POSITION_COLUMNS),
        ("control_positions.parquet", CONTROL_POSITION_COLUMNS),
    ):
        path = directory / name
        frame_for(columns).to_parquet(path, index=False)
        parquet = pq.ParquetFile(path)
        schema_records = [
            {"name": field.name, "type": str(field.type), "nullable": field.nullable}
            for field in parquet.schema_arrow
        ]
        parquet_metadata[name] = {
            "row_count": parquet.metadata.num_rows,
            "columns": list(columns),
            "arrow_schema": schema_records,
            "arrow_schema_sha256": canonical_sha256(schema_records),
        }
    manifest = build_prepare_manifest(directory, parquet_metadata=parquet_metadata)
    (directory / "prepare_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True) + "\n"
    )
    replay_file_hashes = {
        name: file_sha256(directory / name)
        for name in REQUIRED_PREPARE_FILES
        if name != "prepare_receipt.json"
    }
    receipt = with_self_hash(
        {
            "schema_version": PREPARE_SCHEMA,
            "decision": PREPARE_DECISION,
            "pass_authorizes_unseal": False,
            "eligible_to_request_explicit_authorization": True,
            "authorization_state": "awaiting_explicit_user_authorization",
            "next_action": PREPARE_DECISION,
            "required_stage_revision": "v059_unseal_r1",
            "phase_contract_file_sha256": V59_PHASE_FILE_SHA256,
            "phase_contract_canonical_sha256": V59_PHASE_CONTRACT_CANONICAL_SHA256,
            "registered_projection_sha256": canonical_sha256(projection),
            "prepare_git_head": source["git_head"],
            "source_receipt_file_sha256": file_sha256(directory / "source_receipt.json"),
            "evaluation_spec_file_sha256": file_sha256(directory / "evaluation_spec.json"),
            "prepare_manifest_file_sha256": file_sha256(directory / "prepare_manifest.json"),
            "outcome_request_file_sha256": file_sha256(directory / "outcome_request.json"),
            "behavior_audit_file_sha256": file_sha256(directory / "behavior_audit.json"),
            "development_outcome_value_reads": 0,
            "target_asset_loads": 0,
            "performance_metrics_computed": 0,
            "pnl_evaluations": 0,
            "outcome_packet_created": False,
            "authorization_receipt_created": False,
            "cached_replay_binding": {
                "stage": "hidden_staging_before_atomic_publish",
                "files": replay_file_hashes,
                "file_count": len(replay_file_hashes),
                "file_hash_map_sha256": canonical_sha256(replay_file_hashes),
                "new_checkpoint_loads": 0,
                "new_inference": 0,
                "new_linear_control_fits": 0,
                "new_position_generation": 0,
                "new_outcome_reads": 0,
                "files_rewritten": 0,
            },
        },
        "prepare_receipt_sha256",
    )
    (directory / "prepare_receipt.json").write_text(
        json.dumps(receipt, sort_keys=True) + "\n"
    )


def test_v59_v2_packet_validator_requires_prepare_not_to_authorize_unseal(
    tmp_path: Path,
) -> None:
    contract = _contract()
    _write_minimal_v2_packet(tmp_path, contract)
    verified = verify_prepare_packet(
        ROOT,
        tmp_path,
        contract=contract,
        enforce_live_git=False,
        enforce_live_inputs=False,
        verify_prepared_values_gate=False,
        verify_source_commit=False,
    )
    assert verified["passed"]
    assert verified["pass_authorizes_unseal"] is False

    receipt_path = tmp_path / "prepare_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["pass_authorizes_unseal"] = True
    body = dict(receipt)
    body.pop("prepare_receipt_sha256")
    receipt["prepare_receipt_sha256"] = canonical_sha256(body)
    receipt_path.write_text(json.dumps(receipt) + "\n")
    with pytest.raises(V59PrepareError, match="receipt boundary drift"):
        verify_prepare_packet(
            ROOT,
            tmp_path,
            contract=contract,
            enforce_live_git=False,
            enforce_live_inputs=False,
            verify_prepared_values_gate=False,
            verify_source_commit=False,
        )


def test_v59_prepare_packet_rejects_any_outcome_or_authorization_artifact(
    tmp_path: Path,
) -> None:
    contract = _contract()
    _write_minimal_v2_packet(tmp_path, contract)
    (tmp_path / "authorization_receipt.json").write_text("{}\n")
    with pytest.raises(V59PrepareError, match="post-authorization or outcome"):
        verify_prepare_packet(
            ROOT,
            tmp_path,
            contract=contract,
            enforce_live_git=False,
            enforce_live_inputs=False,
            verify_prepared_values_gate=False,
            verify_source_commit=False,
        )
