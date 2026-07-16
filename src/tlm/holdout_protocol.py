from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import yaml


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"Timestamp must be explicitly UTC: {value}")
    return parsed


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_registration_schema() -> dict[str, object]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "TLM prospective candidate registration",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "candidate_id",
            "registered_at_utc",
            "git_tree_hash",
            "source_hashes",
            "config_sha256",
            "feature_schema_sha256",
            "checkpoint_sha256",
            "seeds",
            "timestamp_contract",
            "policy_contract",
            "cost_bps",
            "deterministic_replay_command",
        ],
        "properties": {
            "candidate_id": {"type": "string", "minLength": 1},
            "registered_at_utc": {"type": "string", "format": "date-time"},
            "git_tree_hash": {"type": "string", "minLength": 7},
            "source_hashes": {
                "type": "object",
                "minProperties": 1,
                "additionalProperties": {"type": "string", "minLength": 64},
            },
            "config_sha256": {"type": "string", "minLength": 64},
            "feature_schema_sha256": {"type": "string", "minLength": 64},
            "checkpoint_sha256": {"type": "string", "minLength": 64},
            "seeds": {"type": "array", "items": {"type": "integer"}},
            "timestamp_contract": {"type": "string", "minLength": 1},
            "policy_contract": {"type": "string", "minLength": 1},
            "cost_bps": {"type": "number", "minimum": 0},
            "deterministic_replay_command": {"type": "string", "minLength": 1},
        },
    }


def build_holdout_protocol(config: dict) -> dict[str, object]:
    protocol = config["holdout_protocol"]
    inputs = {
        "evidence_ledger": Path(protocol["evidence_ledger_path"]),
        "evidence_audit": Path(protocol["evidence_audit_path"]),
        "control_certificate": Path(protocol["control_certificate_path"]),
        "control_audit": Path(protocol["control_audit_path"]),
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Holdout protocol inputs are missing: {missing}")
    ledger = _load_json(inputs["evidence_ledger"])
    ledger_audit = _load_json(inputs["evidence_audit"])
    control = _load_json(inputs["control_certificate"])
    control_audit = _load_json(inputs["control_audit"])

    last_exposed = _parse_utc(protocol["last_exposed_return_at_utc"])
    earliest_registration = _parse_utc(
        protocol["earliest_candidate_registration_at_utc"]
    )
    earliest_signal = _parse_utc(protocol["earliest_holdout_signal_at_utc"])
    state = str(protocol["initial_state"])
    candidate = protocol.get("registered_candidate")
    minimums = {
        "calendar_days": int(protocol["minimum_calendar_days"]),
        "daily_observations": int(protocol["minimum_daily_observations"]),
        "active_position_days": int(protocol["minimum_active_position_days"]),
        "regime_observations": {
            name: int(value)
            for name, value in protocol["minimum_regime_observations"].items()
        },
    }
    evaluation = {
        "mode": "single_deferred_batch_evaluation_after_quarantine_closes",
        "run_candidate_during_accumulation": False,
        "execute_orders": False,
        "simulate_daily_orders": False,
        "interim_performance_access": False,
        "outcome_source": protocol["outcome_source"],
        "outcome_source_verification": protocol["outcome_source_verification"],
        "allowed_integrity_metadata": list(protocol["allowed_integrity_metadata"]),
        "forbidden_until_unseal": list(protocol["forbidden_until_unseal"]),
        "one_final_evaluation": True,
        "failed_candidate_action": "retire_candidate_and_start_a_new_future_holdout",
    }
    promotion_gates = {
        "primary_comparator": control["control"]["name"],
        "secondary_comparator": "equal_weight_buy_hold",
        "cost_bps": list(protocol["evaluation_cost_bps"]),
        "block_lengths_days": list(protocol["bootstrap_block_lengths_days"]),
        "bootstrap_paths": int(protocol["bootstrap_paths"]),
        "candidate_total_return_positive": True,
        "candidate_total_return_above_both_comparators": True,
        "candidate_sharpe_above_primary": True,
        "candidate_max_drawdown_not_worse_than_primary_by_more_than": float(
            protocol["max_drawdown_tolerance"]
        ),
        "paired_return_delta_p05_above_zero_all_blocks": True,
        "nonnegative_primary_return_delta_each_registered_regime": True,
        "all_gates_required": True,
    }
    checks = {
        "all_inputs_exist": not missing,
        "v20_audit_passes": bool(ledger_audit.get("passed")),
        "v20_halts_historical_search": ledger["decision"]
        == "halt_new_historical_model_search",
        "v20_has_no_active_candidate": not ledger["synthesis"][
            "active_candidate_versions"
        ],
        "v21_audit_passes": bool(control_audit.get("passed")),
        "v21_is_research_control_only": control["benchmark_status"]
        == "certified_research_control"
        and control["deployment_status"] == "not_authorized",
        "earliest_registration_after_exposed_history": earliest_registration
        > last_exposed,
        "earliest_signal_after_registration_boundary": earliest_signal
        > earliest_registration,
        "minimum_duration_is_at_least_one_year": minimums["calendar_days"]
        >= 365,
        "minimum_observations_are_substantial": minimums["daily_observations"]
        >= 365,
        "all_three_regimes_required": set(minimums["regime_observations"])
        == {"bull", "bear", "high_volatility"},
        "protocol_is_dormant_without_candidate": state
        == "dormant_no_registered_candidate"
        and candidate is None,
        "no_execution_or_daily_simulation": not evaluation["execute_orders"]
        and not evaluation["simulate_daily_orders"]
        and not evaluation["run_candidate_during_accumulation"],
        "no_interim_performance_access": not evaluation[
            "interim_performance_access"
        ],
        "single_final_evaluation_only": evaluation["one_final_evaluation"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"Prospective holdout protocol audit failed: {checks}")

    return {
        "version": "v22",
        "method": "pre_registered_deferred_batch_prospective_holdout",
        "decision": "freeze_protocol_but_do_not_start_without_candidate",
        "state": state,
        "clean_holdout_status": "not_started",
        "registered_candidate": candidate,
        "timeline": {
            "last_exposed_return_at_utc": protocol[
                "last_exposed_return_at_utc"
            ],
            "earliest_candidate_registration_at_utc": protocol[
                "earliest_candidate_registration_at_utc"
            ],
            "earliest_holdout_signal_at_utc": protocol[
                "earliest_holdout_signal_at_utc"
            ],
            "actual_holdout_start": None,
            "actual_holdout_end": None,
            "clock_starts_only_after_valid_candidate_registration": True,
        },
        "minimums": minimums,
        "regime_definitions": protocol["regime_definitions"],
        "evaluation": evaluation,
        "promotion_gates": promotion_gates,
        "candidate_registration_schema_file": "candidate_registration.schema.json",
        "source_hashes": {
            str(path): _sha256_file(path) for path in inputs.values()
        },
        "audit": {"passed": True, "checks": checks},
    }


def _report(result: dict) -> str:
    minimums = result["minimums"]
    gates = result["promotion_gates"]
    lines = [
        "# TLM v22 Prospective Holdout Protocol",
        "",
        "## Decision",
        "",
        "**PROTOCOL FROZEN; HOLDOUT NOT STARTED.** There is no registered candidate.",
        "",
        "This protocol does not run a strategy, simulate daily orders, or perform shadow trading. A future candidate must be fully hashed and registered first. Future market data is then quarantined, and the frozen candidate is evaluated exactly once in a deferred batch after every maturity condition is satisfied.",
        "",
        "## Timeline",
        "",
        f"- Last exposed return: `{result['timeline']['last_exposed_return_at_utc']}`",
        f"- Earliest candidate registration: `{result['timeline']['earliest_candidate_registration_at_utc']}`",
        f"- Earliest eligible signal: `{result['timeline']['earliest_holdout_signal_at_utc']}`",
        "- Actual start/end: unset until a valid candidate is registered.",
        "",
        "## Maturity gate",
        "",
        f"- At least {minimums['calendar_days']} calendar days",
        f"- At least {minimums['daily_observations']} daily observations",
        f"- At least {minimums['active_position_days']} active-position days",
        f"- Bull/bear/high-volatility observations: {minimums['regime_observations']}",
        "",
        "The holdout continues beyond one year until all duration, activity, and regime requirements are met.",
        "",
        "## One-shot promotion gate",
        "",
        f"Primary comparator: `{gates['primary_comparator']}`. Secondary comparator: `{gates['secondary_comparator']}`.",
        "",
        f"The candidate must be positive and beat both comparators after costs {gates['cost_bps']} bps, exceed the primary Sharpe, stay within the registered drawdown tolerance, and have a positive paired return-delta fifth percentile at every {gates['block_lengths_days']}-day block. It must also avoid a negative primary return delta in every registered regime.",
        "",
        "A failed candidate is retired. Its result cannot be used to tune a replacement against the same holdout; a replacement needs a new future window.",
        "",
        "## During quarantine",
        "",
        "Only integrity metadata may be inspected: packet counts, UTC continuity, source checksums, schema validity, and job health. Candidate actions, realized returns, equity, comparative metrics, and regime scores remain unavailable until the one-shot unseal.",
        "",
    ]
    return "\n".join(lines)


def run_holdout_protocol(config: dict) -> dict[str, object]:
    result = build_holdout_protocol(config)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    protocol_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "candidate_registration.schema.json").write_text(
        json.dumps(candidate_registration_schema(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output / "audit.json").write_text(
        json.dumps(result["audit"], indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    (output / "protocol.sha256").write_text(
        f"{_sha256_file(protocol_path)}  protocol.json\n", encoding="utf-8"
    )
    return result
