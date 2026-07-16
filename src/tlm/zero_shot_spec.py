from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import yaml


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()


def _release_state(root: Path, tag: str, verify: bool) -> dict[str, object]:
    if not verify:
        return {
            "tag": tag,
            "commit": None,
            "object_type": "fixture",
            "verified": True,
        }
    commit = subprocess.run(
        ["git", "rev-parse", f"{tag}^{{}}"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    object_type = subprocess.run(
        ["git", "cat-file", "-t", tag],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "tag": tag,
        "commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "object_type": (
            object_type.stdout.strip() if object_type.returncode == 0 else None
        ),
        "verified": commit.returncode == 0 and object_type.returncode == 0,
    }


def build_zero_shot_spec(config: dict) -> dict[str, object]:
    spec = config["zero_shot_spec"]
    root = Path(spec["project_root"]).resolve()
    inputs = {name: root / path for name, path in spec["inputs"].items()}
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V26 inputs are missing: {missing}")
    completion = _load_json(inputs["v25_completion"])
    completion_audit = _load_json(inputs["v25_audit"])
    holdout = _load_json(inputs["v22_protocol"])
    holdout_audit = _load_json(inputs["v22_audit"])
    release = _release_state(
        root,
        str(spec["release_tag"]),
        bool(spec.get("verify_git_release", True)),
    )

    target_assets = list(spec["target_assets"])
    target_symbols = list(spec["target_symbols"])
    excluded_bases = set(spec["universe"]["excluded_bases"])
    proxy_bases = set(spec["universe"]["target_proxy_bases"])
    architecture = dict(spec["architecture"])
    training = dict(spec["training"])
    policy = dict(spec["policy"])
    source_domain_gates = dict(spec["source_domain_gates"])
    blueprint = {
        "candidate_family_id": spec["candidate_family_id"],
        "state": "design_frozen_not_trained",
        "transfer_mode": "zero_shot_from_non_target_crypto_triplets",
        "target_assets": target_assets,
        "target_symbols": target_symbols,
        "development_universe": {
            **spec["universe"],
            "selection_state": "not_inventoried",
            "selected_symbols": [],
        },
        "data_contract": {
            "source": spec["data_contract"]["source"],
            "frequency": spec["data_contract"]["frequency"],
            "development_start": spec["data_contract"]["development_start"],
            "development_cutoff": spec["data_contract"]["development_cutoff"],
            "raw_fields": list(spec["data_contract"]["raw_fields"]),
            "derived_features": list(
                spec["data_contract"]["derived_features"]
            ),
            "target": spec["data_contract"]["target"],
            "timestamp_rule": spec["data_contract"]["timestamp_rule"],
            "target_asset_data_allowed": False,
            "target_asset_outcomes_allowed": False,
        },
        "chronological_splits": spec["chronological_splits"],
        "architecture": architecture,
        "training": training,
        "policy": policy,
        "source_domain_gates": source_domain_gates,
        "target_domain_contract": {
            "development_evaluation_count": 0,
            "prediction_count": 0,
            "pnl_evaluation_count": 0,
            "allowed_evaluation": "v22_one_shot_after_registration_and_maturity_only",
            "candidate_registration_status": "not_registered",
            "holdout_status": holdout["clean_holdout_status"],
        },
    }
    blueprint_hash = _canonical_hash(blueprint)
    release_commit_matches = (
        True
        if not spec.get("verify_git_release", True)
        else release["commit"] == spec["release_commit"]
    )
    architecture_checks = {
        "lookback_positive": int(architecture["lookback_days"]) > 0,
        "patch_divides_or_fits_lookback": int(architecture["patch_length_days"])
        <= int(architecture["lookback_days"]),
        "heads_divide_model_width": int(architecture["d_model"])
        % int(architecture["attention_heads"])
        == 0,
        "fixed_seed_ensemble": len(training["seeds"]) == 3
        and training["seed_selection_allowed"] is False,
        "no_hyperparameter_sweep": training["hyperparameter_search_allowed"]
        is False,
    }
    checks = {
        "all_inputs_exist": not missing,
        "v25_audit_passes": bool(completion_audit.get("passed")),
        "v25_has_no_deployable_tlm": completion["decision"]
        == "complete_research_framework_no_deployable_tlm"
        and completion["system_status"]["deployable_tlm"] == "not_available",
        "v22_audit_passes": bool(holdout_audit.get("passed")),
        "v22_is_dormant_without_candidate": holdout["state"]
        == "dormant_no_registered_candidate"
        and holdout["registered_candidate"] is None,
        "v25_release_is_annotated_and_verified": release["verified"]
        and release["object_type"] in {"tag", "fixture"},
        "v25_release_commit_matches": release_commit_matches,
        "all_target_bases_are_excluded": set(target_assets).issubset(
            excluded_bases
        ),
        "target_symbols_are_not_selected_for_development": not set(
            target_symbols
        ).intersection(
            blueprint["development_universe"]["selected_symbols"]
        ),
        "target_proxies_are_explicitly_excluded": len(proxy_bases) >= 6
        and not proxy_bases.intersection(
            blueprint["development_universe"]["selected_symbols"]
        ),
        "target_data_and_outcomes_forbidden": not blueprint["data_contract"][
            "target_asset_data_allowed"
        ]
        and not blueprint["data_contract"]["target_asset_outcomes_allowed"],
        "zero_target_predictions_or_pnl": all(
            blueprint["target_domain_contract"][field] == 0
            for field in (
                "development_evaluation_count",
                "prediction_count",
                "pnl_evaluation_count",
            )
        ),
        "candidate_is_not_prematurely_registered": blueprint[
            "target_domain_contract"
        ]["candidate_registration_status"]
        == "not_registered",
        "only_data_inventory_is_authorized_next": spec[
            "authorized_next_action"
        ]
        == "v27_non_target_universe_data_audit_only",
        "architecture_contract_is_valid": all(architecture_checks.values()),
    }
    if not all(checks.values()):
        raise RuntimeError(f"V26 zero-shot specification audit failed: {checks}")

    registration_draft = {
        "candidate_id": spec["candidate_family_id"],
        "status": "incomplete_not_valid_for_v22_registration",
        "blueprint_sha256": blueprint_hash,
        "release_tag": release["tag"],
        "release_commit": release["commit"],
        "config_sha256": None,
        "feature_schema_sha256": None,
        "checkpoint_sha256": None,
        "seeds": training["seeds"],
        "timestamp_contract": blueprint["data_contract"]["timestamp_rule"],
        "policy_contract": policy,
        "cost_bps": policy["base_cost_bps"],
        "deterministic_replay_command": None,
        "missing_before_registration": [
            "exact_non_target_universe_and_source_hashes",
            "trained_checkpoint_hashes",
            "resolved_training_config_hash",
            "feature_schema_hash",
            "deterministic_replay_command",
            "source_domain_gate_result",
        ],
    }
    return {
        "version": "v26",
        "method": "ex_ante_zero_shot_candidate_family_pre_registration",
        "decision": "authorize_v27_non_target_universe_data_audit_only",
        "tested": {
            "model_trained": False,
            "source_domain_performance_measured": False,
            "target_domain_performance_measured": False,
            "improvement_status": "unknown_not_evaluated",
            "drawdown_status": "unknown_not_evaluated",
        },
        "release_anchor": release,
        "blueprint": blueprint,
        "blueprint_sha256": blueprint_hash,
        "registration_draft": registration_draft,
        "source_hashes": {
            str(path.relative_to(root)): _sha256_file(path)
            for path in inputs.values()
        },
        "audit": {
            "passed": True,
            "checks": checks,
            "architecture_checks": architecture_checks,
        },
    }


def _report(result: dict) -> str:
    blueprint = result["blueprint"]
    architecture = blueprint["architecture"]
    gates = blueprint["source_domain_gates"]
    lines = [
        "# TLM v26 Zero-Shot Candidate-Family Specification",
        "",
        "## Decision",
        "",
        "**SPECIFICATION FROZEN; ONLY A NON-TARGET DATA INVENTORY IS AUTHORIZED NEXT.**",
        "",
        "No model was trained. No BTC, ETH, or SOL prediction, return, PnL, Sharpe, drawdown, or policy result was computed. Improvement and risk therefore remain unknown.",
        "",
        f"V25 release anchor: {result['release_anchor']['tag']} at {result['release_anchor']['commit']}.",
        f"Blueprint SHA-256: {result['blueprint_sha256']}.",
        "",
        "## Candidate thesis",
        "",
        "Learn one shared temporal/cross-asset representation from triplets of liquid non-target crypto assets, then apply the frozen model zero-shot to BTC/ETH/SOL only inside the future v22 evaluation.",
        "",
        "## Frozen architecture",
        "",
        f"- Lookback: {architecture['lookback_days']} daily observations",
        f"- Temporal patches: {architecture['patch_length_days']} days, stride {architecture['patch_stride_days']}",
        f"- Width/layers/heads: {architecture['d_model']}/{architecture['encoder_layers']}/{architecture['attention_heads']}",
        f"- Fixed seeds: {blueprint['training']['seeds']} as one ensemble; no seed selection",
        f"- Heads: {architecture['prediction_heads']}",
        "",
        "## Contamination boundary",
        "",
        f"Development excludes target bases {blueprint['target_assets']} and their registered proxies. The development universe is not yet selected, and its selected-symbol list is empty. Target-domain prediction and PnL counts are all zero.",
        "",
        "## Source-domain gate",
        "",
        f"Before registration, the one frozen ensemble must pass asset-disjoint non-target evaluation after costs {gates['cost_bps']} bps, beat {gates['primary_control']} on PnL and Sharpe, stay within the drawdown tolerance, and clear the paired {gates['bootstrap_paths']}-path bootstrap at blocks {gates['block_lengths_days']}.",
        "",
        "Failure retires this candidate family. Passing authorizes checkpoint hashing and v22 registration; it does not authorize target evaluation or trading.",
        "",
        "## Next action",
        "",
        "V27 may only inventory official non-target daily archives, apply the frozen eligibility/exclusion rules, verify checksums and coverage, and publish the exact universe. It may not train a model or evaluate any return.",
        "",
    ]
    return "\n".join(lines)


def run_zero_shot_spec(config: dict) -> dict[str, object]:
    result = build_zero_shot_spec(config)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "specification.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "blueprint.json").write_text(
        json.dumps(result["blueprint"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output / "registration_draft.json").write_text(
        json.dumps(result["registration_draft"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output / "audit.json").write_text(
        json.dumps(result["audit"], indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    return result
