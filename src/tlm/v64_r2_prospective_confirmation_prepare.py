from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Mapping


TARGET_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
AUTHORIZED_ACTION = (
    "authorize_v70_prospective_non_target_capture_and_prediction_freeze_only"
)
EXPECTED_INPUT_COUNT = 13
FORBIDDEN_INPUT_SUFFIXES = {".parquet", ".pt", ".pth", ".ckpt", ".csv", ".feather"}
CORE_FILES = (
    "specification.json",
    "protocol.json",
    "audit.json",
    "result.json",
    "report.md",
    "source_receipt.json",
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON mapping: {path}")
    return value


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _write_json_atomic(path: Path, value: object) -> None:
    _write_atomic(path, _json_bytes(value))


def _self_hash_matches(value: Mapping[str, Any], key: str) -> bool:
    payload = dict(value)
    observed = payload.pop(key, None)
    return observed == _canonical_sha256(payload)


def _gate_matrix(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    mandatory = protocol["mandatory_cells"]
    thresholds = protocol["outcome_dependent_gates"]
    folds = [int(value) for value in mandatory["predictive_folds"]]
    costs = [int(value) for value in mandatory["economic_cost_bps"]]
    blocks = [int(value) for value in mandatory["bootstrap_block_lengths_days"]]
    base_cost = int(protocol["policy"]["base_cost_bps"])
    rows: list[dict[str, Any]] = []

    def add(
        gate: str,
        scope: str,
        operator: str,
        threshold: float,
    ) -> None:
        rows.append(
            {
                "gate": gate,
                "scope": scope,
                "operator": operator,
                "threshold": float(threshold),
                "mandatory": True,
            }
        )

    for fold in folds:
        add(
            "mean_spearman_strictly_positive_each_fold",
            f"fold_{fold}",
            "strictly_greater_than",
            thresholds["mean_spearman_strictly_positive_each_fold"],
        )
        add(
            "mean_top1_centered_excess_strictly_positive_each_fold",
            f"fold_{fold}",
            "strictly_greater_than",
            thresholds["mean_top1_centered_excess_strictly_positive_each_fold"],
        )
    for name in (
        "aggregate_pairwise_accuracy_strictly_above",
        "aggregate_state_direction_accuracy_strictly_above",
        "aggregate_absolute_direction_accuracy_strictly_above",
    ):
        add(name, "aggregate", "strictly_greater_than", thresholds[name])
    for fold in folds:
        add(
            "positive_net_return_each_fold_at_base_cost",
            f"fold_{fold}_{base_cost}bps",
            "strictly_greater_than",
            thresholds["positive_net_return_each_fold_at_base_cost"],
        )
    for cost in costs:
        add(
            "aggregate_net_return_strictly_positive_all_costs",
            f"aggregate_{cost}bps",
            "strictly_greater_than",
            thresholds["aggregate_net_return_strictly_positive_all_costs"],
        )
        add(
            "aggregate_sharpe_strictly_positive_all_costs",
            f"aggregate_{cost}bps",
            "strictly_greater_than",
            thresholds["aggregate_sharpe_strictly_positive_all_costs"],
        )
    drawdown_threshold = -float(thresholds["maximum_absolute_drawdown"])
    for fold in folds:
        for cost in costs:
            add(
                "maximum_absolute_drawdown",
                f"fold_{fold}_{cost}bps",
                "greater_than_or_equal",
                drawdown_threshold,
            )
    for cost in costs:
        add(
            "maximum_absolute_drawdown",
            f"aggregate_{cost}bps",
            "greater_than_or_equal",
            drawdown_threshold,
        )
    for block in blocks:
        add(
            "top1_bootstrap_p05_strictly_positive_all_blocks",
            f"block_{block}",
            "strictly_greater_than",
            thresholds["top1_bootstrap_p05_strictly_positive_all_blocks"],
        )
        add(
            "economic_bootstrap_p05_strictly_positive_all_blocks",
            f"block_{block}",
            "strictly_greater_than",
            thresholds["economic_bootstrap_p05_strictly_positive_all_blocks"],
        )
    return rows


def _checkpoint_receipts(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "job_id": row["job_id"],
            "fold": int(row["fold"]),
            "seed": int(row["seed"]),
            "checkpoint_path_reference_only": row["path"],
            "checkpoint_file_sha256": row["file_sha256"],
            "semantic_checkpoint_sha256": row["semantic_checkpoint_sha256"],
            "ranker_state_sha256": row["ranker_state_sha256"],
            "gate_state_sha256": row["gate_state_sha256"],
        }
        for row in sorted(
            manifest["jobs"], key=lambda item: (int(item["fold"]), int(item["seed"]))
        )
    ]


def _fold_contract(
    asset_folds: Mapping[str, Any], triplet_catalog: Mapping[str, Any]
) -> list[dict[str, Any]]:
    catalog_by_fold = {
        int(row["fold"]): row for row in triplet_catalog["folds"]
    }
    rows = []
    for fold in sorted(asset_folds["folds"], key=lambda item: int(item["fold"])):
        fold_id = int(fold["fold"])
        catalog = catalog_by_fold[fold_id]
        rows.append(
            {
                "fold": fold_id,
                "test_symbols": list(fold["test_symbols"]),
                "test_symbol_count": len(fold["test_symbols"]),
                "test_triplet_count": len(catalog["test_triplets"]),
                "test_triplet_set_sha256": _canonical_sha256(
                    catalog["test_triplets"]
                ),
            }
        )
    return rows


def _triplet_contract_is_exact(
    asset_folds: Mapping[str, Any], triplet_catalog: Mapping[str, Any]
) -> bool:
    if triplet_catalog.get("triplet_size") != 3:
        return False
    folds = {int(row["fold"]): row for row in asset_folds.get("folds", [])}
    catalog = {
        int(row["fold"]): row for row in triplet_catalog.get("folds", [])
    }
    if set(folds) != {1, 2, 3} or set(catalog) != {1, 2, 3}:
        return False
    for fold_id, fold in folds.items():
        symbols = list(fold["test_symbols"])
        expected = [list(value) for value in itertools.combinations(sorted(symbols), 3)]
        if catalog[fold_id]["test_symbols"] != symbols:
            return False
        if catalog[fold_id]["test_triplets"] != expected:
            return False
    return True


def _report(protocol: Mapping[str, Any], result: Mapping[str, Any]) -> str:
    window = protocol["evidence_window"]
    return "\n".join(
        [
            "# TLM V69 V64-R2 Prospective Confirmation Protocol",
            "",
            f"**Decision:** `{result['decision']}`",
            "",
            "The V64-R2 family is trained, but no prospective signal, position,",
            "outcome, metric, or PnL is created in V69. The protocol starts only",
            "after the Git commit that registers this completion receipt.",
            "",
            "## Frozen confirmation window",
            "",
            f"- Minimum calendar days: {window['minimum_calendar_days']}",
            f"- Minimum matured dates per fold: {window['minimum_fully_matured_signal_dates_per_fold']}",
            f"- Minimum active-position days per fold: {window['minimum_active_position_days_per_fold']}",
            f"- Maximum calendar days: {window['maximum_calendar_days']}",
            "- Three held-out non-target folds; three seeds averaged before policy action",
            "- Costs: 10, 20, and 30 bps; exact 36-gate matrix; all gates required",
            "",
            "## Operational boundary",
            "",
            "This is a counterfactual research evaluation, not shadow trading. It",
            "opens no orders, paper portfolio, live position, or interim mark-to-market.",
            "Predictions and positions must be hash-frozen before each target maturity.",
            "Realized outcomes may be opened exactly once after the full window matures.",
            "",
            "BTC, ETH, and SOL remain sealed. A V69 pass authorizes only a separate",
            "V70 non-target capture and prediction-freeze phase; it does not authorize",
            "outcome access, financial claims, target evaluation, or deployment.",
            "",
        ]
    )


def run_v64_r2_prospective_confirmation_prepare(
    config: dict[str, Any],
) -> dict[str, Any]:
    spec = config["v64_r2_prospective_confirmation_prepare"]
    root = Path(spec["project_root"]).resolve()
    configured_paths = {name: str(value) for name, value in spec["inputs"].items()}
    expected_by_path = dict(spec["expected_file_sha256_by_path"])
    if len(configured_paths) != EXPECTED_INPUT_COUNT:
        raise RuntimeError("V69 requires exactly 13 metadata inputs")
    if set(configured_paths.values()) != set(expected_by_path):
        raise RuntimeError("V69 metadata allowlist differs from its hash bindings")
    if any(Path(value).suffix.lower() != ".json" for value in configured_paths.values()):
        raise RuntimeError("V69 accepts JSON metadata inputs only")
    if any(Path(value).suffix.lower() in FORBIDDEN_INPUT_SUFFIXES for value in configured_paths.values()):
        raise RuntimeError("V69 input allowlist contains data or checkpoint material")

    paths = {name: root / value for name, value in configured_paths.items()}
    if any(not path.is_file() for path in paths.values()):
        missing = sorted(str(path) for path in paths.values() if not path.is_file())
        raise FileNotFoundError(f"V69 metadata input missing: {missing}")
    observed_before = {
        configured_paths[name]: _file_sha256(path) for name, path in paths.items()
    }
    if observed_before != expected_by_path:
        raise RuntimeError("V69 immutable metadata input drift")
    loaded = {name: _load_json(path) for name, path in paths.items()}

    blueprint = loaded["v65_blueprint"]
    asset_folds = loaded["v32_asset_folds"]
    triplets = loaded["v32_triplet_catalog"]
    result68 = loaded["v68_result"]
    audit68 = loaded["v68_audit"]
    training_spec = loaded["v68_training_spec"]
    checkpoints = loaded["v68_checkpoint_manifest"]
    scalers = loaded["v68_scaler_manifest"]
    data_access = loaded["v68_data_access"]
    verification = loaded["v68_verification"]
    replay = loaded["v68_replay"]
    completion = loaded["v68_completion_receipt"]
    manifest = loaded["v68_artifact_manifest"]
    protocol_config = copy.deepcopy(spec["protocol"])
    gate_matrix = _gate_matrix(protocol_config)
    checkpoint_receipts = _checkpoint_receipts(checkpoints)
    fold_contract = _fold_contract(asset_folds, triplets)

    protocol = {
        "schema_version": "v69-v64-r2-prospective-confirmation-protocol/v1",
        "version": spec["version"],
        "lineage_label": spec["lineage_label"],
        "family_id": spec["family_id"],
        **protocol_config,
        "fold_contract": fold_contract,
        "checkpoint_identity_receipts": checkpoint_receipts,
        "gate_matrix": gate_matrix,
        "target_contract": spec["target_contract"],
        "pass_action": spec["pass_action"],
        "failure_action": spec["failure_action"],
    }
    protocol["protocol_sha256"] = _canonical_sha256(protocol)
    specification = {
        "schema_version": "v69-v64-r2-prospective-confirmation-specification/v1",
        "version": spec["version"],
        "family_id": spec["family_id"],
        "state": "frozen_not_started",
        "evidence_tier": protocol["evidence_tier"],
        "protocol_sha256": protocol["protocol_sha256"],
        "input_file_sha256": observed_before,
        "allowed_input_count": len(paths),
        "required_artifacts": [
            "specification.json",
            "protocol.json",
            "audit.json",
            "result.json",
            "report.md",
            "source_receipt.json",
            "artifact_manifest.json",
            "completion_receipt.json",
        ],
        "byte_identical_replay_required": True,
        "target_contract": spec["target_contract"],
        "authorized_next_action": spec["pass_action"],
    }
    specification["specification_sha256"] = _canonical_sha256(specification)

    fold_test_sets = [set(row["test_symbols"]) for row in asset_folds["folds"]]
    all_test_symbols = set().union(*fold_test_sets)
    checkpoint_grid = {
        (int(row["fold"]), int(row["seed"])) for row in checkpoints["jobs"]
    }
    exact_policy = blueprint["policy"]
    data_access_zero = (
        data_access["outcome_rows_read"] == 0
        and data_access["performance_metrics_computed"] is False
        and data_access["pnl_computed"] is False
        and data_access["policy_actions_emitted"] is False
        and data_access["predictions_written"] is False
        and data_access["target_assets_loaded"] == []
    )
    access_ledger = spec["access_ledger"]
    zero_access_fields = [
        name for name in access_ledger if name != "metadata_json_reads"
    ]
    checks = {
        "exact_thirteen_hash_bound_json_inputs": len(paths) == EXPECTED_INPUT_COUNT
        and observed_before == expected_by_path,
        "v65_blueprint_is_self_hash_valid": _self_hash_matches(
            blueprint, "blueprint_sha256"
        )
        and blueprint["candidate_family_id"] == spec["family_id"],
        "v68_result_and_audit_are_hash_valid": _self_hash_matches(
            result68, "result_sha256"
        )
        and _self_hash_matches(audit68, "audit_sha256")
        and audit68["passed"] is True
        and result68["decision"]
        == "authorize_v69_outcome_blind_non_target_prospective_confirmation_prepare_only",
        "v68_training_metadata_receipts_are_hash_valid": _self_hash_matches(
            training_spec, "training_spec_sha256"
        )
        and _self_hash_matches(checkpoints, "manifest_sha256")
        and _self_hash_matches(scalers, "manifest_sha256")
        and _self_hash_matches(data_access, "data_access_sha256")
        and _self_hash_matches(verification, "verification_sha256")
        and _self_hash_matches(replay, "replay_sha256")
        and _self_hash_matches(completion, "completion_receipt_sha256")
        and _self_hash_matches(manifest, "artifact_manifest_sha256"),
        "v68_training_grid_is_complete_and_unselected": len(checkpoints["jobs"])
        == 9
        and checkpoint_grid
        == set(itertools.product([1, 2, 3], [42, 7, 123]))
        and all(row["status"] == "completed" for row in checkpoints["jobs"])
        and all(len(row["checkpoint_file_sha256"]) == 64 for row in checkpoint_receipts)
        and result68["summary"]["completed_jobs"] == 9
        and result68["summary"]["checkpoint_count"] == 9
        and result68["summary"]["ranker_optimizer_steps"] == 0,
        "v68_verification_and_zero_step_replay_passed": verification["audit"][
            "passed"
        ]
        is True
        and verification["verification"]["checkpoint_roundtrip_passed"] is True
        and replay["audit"]["passed"] is True
        and replay["replay"]
        == {
            "artifact_hashes_match": True,
            "new_jobs": 0,
            "new_optimizer_steps": 0,
            "rewritten_checkpoints": 0,
        },
        "v68_prior_access_has_no_outcome_performance_or_target_use": data_access_zero,
        "v32_fold_and_triplet_contract_is_exact": asset_folds["fold_count"] == 3
        and len(all_test_symbols) == 30
        and sum(len(value) for value in fold_test_sets) == 30
        and all_test_symbols.isdisjoint(TARGET_SYMBOLS)
        and _triplet_contract_is_exact(asset_folds, triplets),
        "first_signal_is_strictly_post_registration": protocol["registration"][
            "first_admissible_signal_date_rule"
        ]
        == "strictly_after_v69_completion_receipt_commit"
        and protocol["registration"][
            "registration_commit_must_precede_feature_candle_close"
        ]
        is True,
        "consumed_history_cannot_be_reclassified": set(
            protocol["registration"]["excluded_evidence"]
        )
        == {"consumed_2025_adaptive_development", "pre_registration_2026"},
        "capture_and_timestamp_semantics_are_frozen": protocol["capture"][
            "feature_cutoff"
        ]
        == "fully_closed_daily_candle_only"
        and protocol["capture"]["authentication_required"] is False
        and protocol["timestamp_contract"]["eligible_action_date"]
        == "feature_date_plus_1_calendar_day"
        and protocol["timestamp_contract"]["target_h1_maturity_date"]
        == "feature_date_plus_2_calendar_days",
        "minimum_window_is_substantial_and_bounded": protocol["evidence_window"][
            "minimum_calendar_days"
        ]
        >= 120
        and protocol["evidence_window"]["minimum_eligible_signal_dates_per_fold"]
        >= 90
        and protocol["evidence_window"]["minimum_active_position_days_per_fold"]
        >= 20
        and protocol["evidence_window"]["maximum_calendar_days"] >= 365,
        "exact_fold_seed_context_aggregation_is_frozen": protocol["universe"][
            "folds"
        ]
        == [1, 2, 3]
        and protocol["universe"]["seeds"] == [42, 7, 123]
        and protocol["universe"]["seed_or_context_selection_allowed"] is False
        and protocol["universe"]["seed_aggregation"]
        == "equal_weight_before_policy_action",
        "v65_policy_and_costs_are_unchanged": protocol["policy"] == exact_policy
        and protocol["policy"]["base_cost_bps"] == 10
        and protocol["policy"]["reporting_cost_bps"] == [10, 20, 30]
        and protocol["policy"]["threshold_sweep_allowed"] is False,
        "turnover_and_return_accounting_are_complete": protocol["accounting"][
            "charge_entry_switch_exit_forced_exit_and_final_liquidation"
        ]
        is True
        and protocol["accounting"]["aggregate_across_folds"]
        == "equal_weight_daily_mean"
        and protocol["accounting"]["execution_claim"]
        == "research_counterfactual_only",
        "prediction_freeze_precedes_maturity_without_regeneration": protocol[
            "prediction_freeze"
        ]["freeze_must_precede_target_maturity"]
        is True
        and protocol["prediction_freeze"]["regeneration_after_freeze_allowed"]
        is False
        and protocol["prediction_freeze"]["interim_metric_or_pnl_access"]
        is False,
        "exact_thirty_six_gate_matrix_is_all_or_nothing": len(gate_matrix)
        == protocol["mandatory_cells"]["expected_gate_count"]
        == 36
        and all(row["mandatory"] is True for row in gate_matrix)
        and protocol["outcome_dependent_gates"]["all_gates_required"] is True
        and protocol["outcome_dependent_gates"]["aggregate_rescue_allowed"]
        is False,
        "outcome_packet_and_one_shot_rules_are_frozen": protocol["outcome_packet"][
            "source_reads"
        ]
        == 1
        and protocol["outcome_packet"]["written_atomically"] is True
        and protocol["outcome_packet"]["immutable"] is True
        and protocol["one_shot"]["evaluation_count"] == 1
        and protocol["one_shot"]["outcome_source_read_count"] == 1
        and protocol["one_shot"]["retuning_after_unseal_allowed"] is False
        and protocol["one_shot"]["prediction_or_position_regeneration_allowed"]
        is False,
        "not_shadow_paper_live_or_real_money_trading": all(
            value is False for value in protocol["operational_boundary"].values()
        ),
        "v69_access_ledger_has_only_thirteen_metadata_reads": access_ledger[
            "metadata_json_reads"
        ]
        == 13
        and all(access_ledger[name] == 0 for name in zero_access_fields),
        "targets_remain_sealed": set(spec["target_contract"]["symbols"])
        == TARGET_SYMBOLS
        and spec["target_contract"]["status"] == "sealed"
        and spec["target_contract"]["target_assets_loaded"] == []
        and spec["target_contract"]["target_predictions"] == 0
        and spec["target_contract"]["target_pnl_evaluations"] == 0,
        "v69_authorizes_only_v70_capture_and_prediction_freeze": spec["pass_action"]
        == AUTHORIZED_ACTION
        and protocol["one_shot"]["target_evaluation_authorized_by_pass"] is False
        and protocol["one_shot"]["deployment_authorized_by_pass"] is False,
        "deterministic_core_serialization": _json_bytes(protocol)
        == _json_bytes(copy.deepcopy(protocol))
        and _json_bytes(specification) == _json_bytes(copy.deepcopy(specification)),
    }
    observed_after = {
        configured_paths[name]: _file_sha256(path) for name, path in paths.items()
    }
    checks["all_input_hashes_unchanged_after_metadata_read"] = (
        observed_after == observed_before
    )
    audit = {
        "schema_version": "v69-v64-r2-prospective-confirmation-audit/v1",
        "passed": all(checks.values()),
        "checks": checks,
        "check_count": len(checks),
    }
    audit["audit_sha256"] = _canonical_sha256(audit)
    if not audit["passed"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"V69 prospective protocol audit failed: {failed}")

    source_receipt = {
        "schema_version": "v69-v64-r2-source-receipt/v1",
        "metadata_input_count": len(paths),
        "input_file_sha256_before": observed_before,
        "input_file_sha256_after": observed_after,
        "source_module": "src/tlm/v64_r2_prospective_confirmation_prepare.py",
        "source_module_sha256": _file_sha256(Path(__file__)),
        "config_payload_sha256": _canonical_sha256(config),
        "access_ledger": access_ledger,
    }
    source_receipt["source_receipt_sha256"] = _canonical_sha256(source_receipt)
    result = {
        "schema_version": "v69-v64-r2-prospective-confirmation-result/v1",
        "decision": spec["pass_action"],
        "family_id": spec["family_id"],
        "lineage_label": spec["lineage_label"],
        "evidence_tier": "ex_ante_metadata_only_prospective_protocol",
        "specification_sha256": specification["specification_sha256"],
        "protocol_sha256": protocol["protocol_sha256"],
        "audit": audit,
        "summary": {
            "metadata_json_reads": 13,
            "checkpoint_identity_receipts": len(checkpoint_receipts),
            "heldout_non_target_folds": len(fold_contract),
            "heldout_non_target_symbols": len(all_test_symbols),
            "mandatory_gate_count": len(gate_matrix),
            "minimum_calendar_days": protocol["evidence_window"][
                "minimum_calendar_days"
            ],
            "minimum_matured_dates_per_fold": protocol["evidence_window"][
                "minimum_fully_matured_signal_dates_per_fold"
            ],
            "parquet_deserializations": 0,
            "raw_market_data_reads": 0,
            "checkpoint_deserializations": 0,
            "model_instantiations": 0,
            "predictions": 0,
            "positions": 0,
            "performance_metrics": 0,
            "pnl_computations": 0,
            "outcome_source_reads": 0,
            "target_asset_rows": 0,
            "outcomes_available": False,
        },
        "target_contract": spec["target_contract"],
        "deployable": False,
    }
    result["result_sha256"] = _canonical_sha256(result)
    report = _report(protocol, result)

    configured_output = Path(config["output_dir"])
    output = configured_output if configured_output.is_absolute() else root / configured_output
    output.mkdir(parents=True, exist_ok=True)
    payloads: dict[str, bytes] = {
        "specification.json": _json_bytes(specification),
        "protocol.json": _json_bytes(protocol),
        "audit.json": _json_bytes(audit),
        "result.json": _json_bytes(result),
        "report.md": report.encode("utf-8"),
        "source_receipt.json": _json_bytes(source_receipt),
    }
    for name, payload in payloads.items():
        _write_atomic(output / name, payload)
    artifact_manifest = {
        "schema_version": "v69-v64-r2-artifact-manifest/v1",
        "files": {name: _file_sha256(output / name) for name in CORE_FILES},
    }
    artifact_manifest["artifact_manifest_sha256"] = _canonical_sha256(
        artifact_manifest
    )
    _write_json_atomic(output / "artifact_manifest.json", artifact_manifest)
    completion_receipt = {
        "schema_version": "v69-v64-r2-completion-receipt/v1",
        "decision": result["decision"],
        "family_id": spec["family_id"],
        "artifact_manifest_file_sha256": _file_sha256(
            output / "artifact_manifest.json"
        ),
        "artifact_manifest_sha256": artifact_manifest[
            "artifact_manifest_sha256"
        ],
        "result_file_sha256": _file_sha256(output / "result.json"),
        "result_sha256": result["result_sha256"],
        "audit_file_sha256": _file_sha256(output / "audit.json"),
        "audit_sha256": audit["audit_sha256"],
        "source_receipt_file_sha256": _file_sha256(output / "source_receipt.json"),
        "source_receipt_sha256": source_receipt["source_receipt_sha256"],
        "audit_passed": True,
        "first_admissible_signal_date_rule": protocol["registration"][
            "first_admissible_signal_date_rule"
        ],
    }
    completion_receipt["completion_receipt_sha256"] = _canonical_sha256(
        completion_receipt
    )
    _write_json_atomic(output / "completion_receipt.json", completion_receipt)
    return result
