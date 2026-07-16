from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


TARGET_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
AUTHORIZATION_MESSAGE = (
    "Autorizo uma especificação V64-R2, sucessora direta da V64, mantendo o "
    "ranker e a arquitetura relativa congelados e alterando somente o state gate "
    "probabilístico e sua regra de abstention, sem abrir dados, checkpoints ou "
    "outcomes nesta fase, mantendo BTC, ETH e SOL selados."
)
AUTHORIZED_ACTION = (
    "execute_v65_metadata_only_v64_r2_probabilistic_state_gate_specification"
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON mapping: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _registered_self_hash(value: dict[str, Any], key: str) -> bool:
    payload = dict(value)
    registered = payload.pop(key, None)
    return registered == _canonical_sha256(payload)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def ranker_parameter_count(architecture: dict[str, Any]) -> int:
    d_model = int(architecture["d_model"])
    feed_forward = int(architecture["feed_forward_width"])
    patch_width = int(architecture["patch_length_days"]) * 9
    patch_count = (
        (int(architecture["lookback_days"]) - int(architecture["patch_length_days"]))
        // int(architecture["patch_stride_days"])
        + 1
    )
    layer_count = int(architecture["encoder_layers"]) + int(
        architecture["cross_asset_attention_layers"]
    )
    output_width = 2
    return int(
        patch_width * d_model
        + d_model
        + patch_count * d_model
        + d_model
        + layer_count
        * (
            4 * d_model * d_model
            + 2 * d_model * feed_forward
            + feed_forward
            + 9 * d_model
        )
        + 4 * d_model
        + d_model * output_width
        + output_width
        + d_model * patch_width
        + patch_width
    )


def state_gate_parameter_count(architecture: dict[str, Any]) -> int:
    d_model = int(architecture["d_model"])
    feed_forward = int(architecture["feed_forward_width"])
    patch_width = int(architecture["patch_length_days"]) * int(
        architecture["input_features"]
    )
    patch_count = (
        (int(architecture["lookback_days"]) - int(architecture["patch_length_days"]))
        // int(architecture["patch_stride_days"])
        + 1
    )
    layer_count = int(architecture["encoder_layers"])
    output_width = int(architecture["output_width"])
    return int(
        patch_width * d_model
        + d_model
        + patch_count * d_model
        + layer_count
        * (
            4 * d_model * d_model
            + 2 * d_model * feed_forward
            + feed_forward
            + 9 * d_model
        )
        + 4 * d_model
        + d_model * output_width
        + output_width
    )


def _ranker_identity_receipts(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "job_id": row["job_id"],
            "fold": int(row["fold"]),
            "seed": int(row["seed"]),
            "checkpoint_file_sha256": row["file_sha256"],
            "semantic_checkpoint_sha256": row["semantic_checkpoint_sha256"],
            "ranker_state_sha256": row["ranker_state_sha256"],
        }
        for row in sorted(
            manifest["jobs"], key=lambda item: (int(item["fold"]), int(item["seed"]))
        )
    ]


def run_v64_r2_probabilistic_state_gate_spec(
    config: dict[str, Any],
) -> dict[str, Any]:
    spec = config["v64_r2_probabilistic_state_gate_spec"]
    root = Path(spec["project_root"]).resolve()
    paths = {name: root / value for name, value in spec["inputs"].items()}
    if any(path.suffix.lower() in {".parquet", ".pt", ".pth", ".ckpt"} for path in paths.values()):
        raise RuntimeError("V65 input allowlist contains data or checkpoint material")

    observed_before = {name: _sha256_file(path) for name, path in paths.items()}
    if observed_before != spec["expected_input_sha256"]:
        raise RuntimeError("V65 immutable metadata input drift")
    loaded = {name: _load_json(path) for name, path in paths.items()}

    authorization = loaded["user_authorization_blueprint"]
    authorization_audit = loaded["user_authorization_audit"]
    authorization_result = loaded["user_authorization_result"]
    v60 = loaded["v60_specification"]
    checkpoint_manifest = loaded["v63_checkpoint_manifest"]
    v64_result = loaded["v64_evaluation_result"]
    v64_audit = loaded["v64_evaluation_audit"]
    postmortem_contract = loaded["v64_postmortem_contract"]
    postmortem = loaded["v64_postmortem_result"]

    ranker_architecture = spec["ranker_contract"]["architecture"]
    gate_architecture = spec["state_gate_architecture"]
    ranker_parameters = ranker_parameter_count(ranker_architecture)
    gate_parameters = state_gate_parameter_count(gate_architecture)
    total_parameters = ranker_parameters + gate_parameters
    ranker_identities = _ranker_identity_receipts(checkpoint_manifest)
    constraints = spec["constraints"]

    specification = {
        "schema_version": "v65-v64-r2-probabilistic-state-gate-specification/v1",
        "version": spec["version"],
        "lineage_label": spec["lineage_label"],
        "candidate_family_id": spec["candidate_family_id"],
        "state": "ex_ante_metadata_only_revision_frozen_not_implemented_or_trained",
        "lineage": spec["lineage"],
        "target_contract": spec["target_contract"],
        "ranker_contract": spec["ranker_contract"],
        "ranker_identity_receipts": ranker_identities,
        "state_gate_architecture": gate_architecture,
        "probabilistic_gate": spec["probabilistic_gate"],
        "decomposition": spec["decomposition"],
        "capacity_contract": spec["capacity_contract"],
        "policy": spec["policy"],
        "future_training_contract": spec["future_training_contract"],
        "evidence_contract": spec["evidence_contract"],
        "constraints": constraints,
        "authorized_next_action": spec["authorized_next_action"],
        "source_receipts": spec["expected_input_sha256"],
        "parameter_counts": {
            "ranker": ranker_parameters,
            "state_gate": gate_parameters,
            "total": total_parameters,
        },
    }
    specification["specification_sha256"] = _canonical_sha256(specification)

    permitted_changes = set(spec["lineage"]["permitted_changes"])
    v60_ranker = v60["ranker_architecture"]
    v60_ranker_objective = v60["objective"]["ranker"]
    jobs = checkpoint_manifest["jobs"]
    checks = {
        "all_metadata_hashes_match_before_read": observed_before
        == spec["expected_input_sha256"],
        "explicit_user_authorization_is_hash_valid": _registered_self_hash(
            authorization, "blueprint_sha256"
        )
        and _registered_self_hash(authorization_result, "result_sha256")
        and authorization_audit["passed"] is True,
        "authorization_scope_is_exact": authorization["source_user_message"]
        == AUTHORIZATION_MESSAGE
        and authorization["authorized_action"] == AUTHORIZED_ACTION
        and authorization_result["decision"] == AUTHORIZED_ACTION,
        "v64_retirement_is_immutable": v64_result["decision"]
        == "retire_family_without_target_evaluation_or_retuning"
        and postmortem["retirement"]["immutable"] is True
        and postmortem_contract["retirement"]["immutable"] is True,
        "v64_one_shot_contract_remains_consumed": v64_result["unseal_count"] == 1
        and v64_result["source_outcome_reads"] == 1
        and v64_result["retuning_performed"] is False
        and v64_result["prediction_or_position_regeneration"] is False
        and v64_audit["passed"] is True,
        "v64_postmortem_contract_is_hash_bound": postmortem["contract_sha256"]
        == _canonical_sha256(postmortem_contract),
        "v64_relative_signal_and_state_failure_are_preserved": postmortem["signal"][
            "aggregate"
        ]["spearman"]
        > 0
        and postmortem["signal"]["aggregate"]["pairwise_accuracy"] > 0.5
        and postmortem["signal"]["aggregate"]["top1_centered_excess"] > 0
        and postmortem["verdict"]["primary_failure"]
        == "absolute_market_state_calibration_failed_to_convert_relative_rank_information_into_positive_absolute_returns",
        "technical_identity_is_new_direct_successor": spec["lineage"][
            "direct_successor"
        ]
        is True
        and spec["lineage"]["technical_family_is_new"] is True
        and spec["candidate_family_id"]
        != spec["lineage"]["scientific_parent_family"],
        "only_state_gate_and_abstention_may_change": permitted_changes
        == {
            "state_gate_point_head_to_probabilistic_location_scale_head",
            "deterministic_edge_abstention_to_probability_of_net_positive_return_abstention",
        }
        and spec["lineage"]["all_other_components_frozen"] is True,
        "ranker_architecture_is_byte_semantically_frozen": ranker_architecture
        == v60_ranker,
        "ranker_objective_is_byte_semantically_frozen": spec["ranker_contract"][
            "objective"
        ]
        == v60_ranker_objective,
        "nine_ranker_states_are_identity_bound_without_checkpoint_open": len(jobs)
        == 9
        and len(ranker_identities) == 9
        and len({row["job_id"] for row in jobs}) == 9
        and all(row["status"] == "completed" for row in jobs)
        and all(len(row["ranker_state_sha256"]) == 64 for row in jobs)
        and spec["ranker_contract"]["weights"][
            "checkpoint_deserialization_during_v65"
        ]
        is False
        and spec["ranker_contract"]["weights"]["gate_state_reuse"] == "forbidden",
        "ranker_capacity_is_exact": ranker_parameters
        == ranker_architecture["expected_parameter_count"]
        == 1_231_634,
        "probabilistic_gate_capacity_is_exact": gate_parameters
        == gate_architecture["expected_parameter_count"]
        == 27_522
        and gate_architecture["output_width"] == 2,
        "total_capacity_is_frozen_under_ceiling": total_parameters
        == spec["capacity_contract"]["expected_total_parameter_count"]
        and total_parameters <= spec["capacity_contract"]["parameter_ceiling"]
        and spec["capacity_contract"]["size_sweep_allowed"] is False
        and spec["capacity_contract"]["larger_ranker_allowed"] is False,
        "probabilistic_family_is_single_and_fixed": spec["probabilistic_gate"][
            "distribution"
        ]
        == "student_t_location_scale"
        and spec["probabilistic_gate"]["degrees_of_freedom"] == 5.0
        and spec["probabilistic_gate"]["degrees_of_freedom_trainable"] is False
        and spec["probabilistic_gate"]["loss"]
        == "mean_negative_student_t_log_likelihood"
        and spec["capacity_contract"]["state_gate_variant_count"] == 1,
        "abstention_rule_is_preregistered_without_sweep": spec["policy"][
            "abstention_probability_threshold"
        ]
        == 0.60
        and spec["policy"]["threshold_sweep_allowed"] is False
        and spec["policy"]["reporting_cost_bps"] == [10, 20, 30]
        and spec["policy"]["switch_hurdle"] == 0.002,
        "future_ranker_is_frozen_and_gate_is_fresh": spec["future_training_contract"][
            "ranker_status"
        ]
        == "frozen_no_gradient_no_optimizer"
        and spec["future_training_contract"]["state_gate_initialization"]
        == "fresh_registered_seed"
        and spec["future_training_contract"]["implementation_allowed_during_v65"]
        is False,
        "v64_outcomes_are_not_reused": spec["future_training_contract"][
            "v64_2025_outcomes_allowed_for_training_validation_tuning_or_evaluation"
        ]
        is False
        and spec["evidence_contract"]["v64_outcomes_may_not_be_reread_or_reused"]
        is True,
        "targets_remain_sealed": set(spec["target_contract"]["target_symbols"])
        == TARGET_SYMBOLS
        and spec["target_contract"]["status"] == "sealed"
        and spec["target_contract"]["target_data_allowed"] is False
        and v64_result["target_assets_loaded"] == []
        and v64_result["target_predictions"] == 0
        and v64_result["target_pnl_evaluations"] == 0,
        "v65_is_strictly_metadata_only": constraints["metadata_only"] is True
        and not any(value for key, value in constraints.items() if key != "metadata_only"),
        "v65_authorizes_only_v66_synthetic_harness": spec["authorized_next_action"]
        == "authorize_v66_synthetic_v64_r2_probabilistic_state_gate_harness_only",
    }
    observed_after = {name: _sha256_file(path) for name, path in paths.items()}
    checks["all_metadata_hashes_match_after_read"] = observed_after == observed_before
    audit = {
        "schema_version": "v65-v64-r2-probabilistic-state-gate-audit/v1",
        "passed": all(checks.values()),
        "checks": checks,
    }
    if not audit["passed"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise RuntimeError(f"V65 specification audit failed: {failed}")

    blueprint = {
        "schema_version": "v65-v64-r2-probabilistic-state-gate-blueprint/v1",
        "version": spec["version"],
        "lineage_label": spec["lineage_label"],
        "candidate_family_id": spec["candidate_family_id"],
        "state": specification["state"],
        "lineage": spec["lineage"],
        "ranker_contract": spec["ranker_contract"],
        "ranker_identity_receipts": ranker_identities,
        "state_gate_architecture": gate_architecture,
        "probabilistic_gate": spec["probabilistic_gate"],
        "decomposition": spec["decomposition"],
        "policy": spec["policy"],
        "future_training_contract": spec["future_training_contract"],
        "evidence_contract": spec["evidence_contract"],
        "target_contract": spec["target_contract"],
        "parameter_counts": specification["parameter_counts"],
        "authorized_next_action": spec["authorized_next_action"],
        "specification_sha256": specification["specification_sha256"],
    }
    blueprint["blueprint_sha256"] = _canonical_sha256(blueprint)
    result = {
        "schema_version": "v65-v64-r2-probabilistic-state-gate-result/v1",
        "version": spec["version"],
        "lineage_label": spec["lineage_label"],
        "decision": spec["authorized_next_action"],
        "family_id": spec["candidate_family_id"],
        "specification_sha256": specification["specification_sha256"],
        "blueprint_sha256": blueprint["blueprint_sha256"],
        "audit": audit,
        "input_hash_receipt": observed_after,
        "summary": {
            "ranker_parameters": ranker_parameters,
            "state_gate_parameters": gate_parameters,
            "total_parameters": total_parameters,
            "frozen_ranker_state_receipts": len(ranker_identities),
            "abstention_probability_threshold": spec["policy"][
                "abstention_probability_threshold"
            ],
            "parquet_deserializations": 0,
            "checkpoint_reads": 0,
            "model_instantiations": 0,
            "optimizer_steps": 0,
            "predictions": 0,
            "performance_metrics": 0,
            "pnl_computations": 0,
            "outcome_source_reads": 0,
            "target_asset_rows": 0,
        },
    }
    result["result_sha256"] = _canonical_sha256(result)

    configured_output = Path(config["output_dir"])
    output = configured_output if configured_output.is_absolute() else root / configured_output
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "specification.json", specification)
    _write_json(output / "blueprint.json", blueprint)
    _write_json(output / "audit.json", audit)
    _write_json(output / "input_hash_receipt.json", observed_after)
    _write_json(output / "result.json", result)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    report = "\n".join(
        [
            "# V64-R2 / V65 Probabilistic State-Gate Specification",
            "",
            f"Decision: **{result['decision']}**",
            "",
            "The exact nine V64 ranker states and the full relative architecture are",
            "hash-bound and frozen. The point state gate is replaced by one fixed",
            "Student-t location/scale gate. The long-one/cash policy abstains unless",
            "the selected asset has at least 60% predicted probability of clearing",
            "the exact transition cost. No threshold or distribution sweep is allowed.",
            "",
            f"Ranker parameters: **{ranker_parameters:,}**",
            f"Probabilistic state-gate parameters: **{gate_parameters:,}**",
            f"Total parameters: **{total_parameters:,}**",
            "",
            "No Parquet, checkpoint, model, prediction, metric, PnL, outcome source,",
            "or BTC/ETH/SOL row was opened. V66 may run only the synthetic harness.",
            "",
        ]
    )
    (output / "report.md").write_text(report, encoding="utf-8")
    packet_files = [
        "audit.json",
        "blueprint.json",
        "input_hash_receipt.json",
        "report.md",
        "resolved_config.yaml",
        "result.json",
        "specification.json",
    ]
    artifact_manifest = {
        "schema_version": "v65-v64-r2-artifact-manifest/v1",
        "files": {name: _sha256_file(output / name) for name in packet_files},
    }
    artifact_manifest["artifact_manifest_sha256"] = _canonical_sha256(
        artifact_manifest
    )
    _write_json(output / "artifact_manifest.json", artifact_manifest)
    completion_receipt = {
        "schema_version": "v65-v64-r2-completion-receipt/v1",
        "decision": result["decision"],
        "family_id": spec["candidate_family_id"],
        "lineage_label": spec["lineage_label"],
        "artifact_manifest_file_sha256": _sha256_file(output / "artifact_manifest.json"),
        "artifact_manifest_sha256": artifact_manifest["artifact_manifest_sha256"],
        "result_sha256": result["result_sha256"],
        "audit_passed": audit["passed"],
    }
    completion_receipt["completion_receipt_sha256"] = _canonical_sha256(
        completion_receipt
    )
    _write_json(output / "completion_receipt.json", completion_receipt)
    return result
