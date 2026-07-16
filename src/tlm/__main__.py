from __future__ import annotations

import argparse
import json

from .audit import audit_artifacts
from .config import load_config, smoke_config
from .consensus import run_consensus_experiment
from .control_certificate import run_control_certificate
from .cftc_feasibility import run_cftc_feasibility
from .data import load_market_data
from .data_family_audit import run_data_family_audit
from .decoupled_rank_state_spec import run_decoupled_rank_state_spec
from .decoupled_rank_state_harness import run_decoupled_rank_state_harness
from .decoupled_rank_state_dataset import run_decoupled_rank_state_dataset
from .decoupled_rank_state_training import run_decoupled_rank_state_training
from .decoupled_rank_state_evaluation import (
    prepare_decoupled_rank_state_evaluation,
)
from .decoupled_rank_state_evaluation_complete import (
    replay_decoupled_rank_state_evaluation,
    unseal_decoupled_rank_state_evaluation,
)
from .v64_r2_probabilistic_state_gate_spec import (
    run_v64_r2_probabilistic_state_gate_spec,
)
from .v64_r2_probabilistic_state_gate_harness import (
    run_v64_r2_probabilistic_state_gate_harness,
)
from .v64_r2_probabilistic_state_gate_dataset import (
    run_v64_r2_probabilistic_state_gate_dataset,
)
from .v64_r2_probabilistic_state_gate_training import (
    run_v64_r2_probabilistic_state_gate_training,
)
from .v64_r2_prospective_confirmation_prepare import (
    run_v64_r2_prospective_confirmation_prepare,
)
from .v64_r2_prospective_capture import run_v64_r2_prospective_capture
from .v64_r2_retrospective_diagnostic import (
    run_v64_r2_retrospective_diagnostic_prepare,
)
from .v64_r2_retrospective_evaluation import (
    replay_v64_r2_retrospective_diagnostic,
    unseal_v64_r2_retrospective_diagnostic,
)
from .v72_diagnostic_record import run_v72_diagnostic_record
from .v78_terminal_record import run_v78_terminal_record
from .low_turnover_rank_spec import run_low_turnover_rank_spec
from .low_turnover_rank_harness import run_low_turnover_rank_harness
from .low_turnover_rank_chronology_erratum import (
    run_low_turnover_rank_chronology_erratum,
)
from .low_turnover_rank_dataset import run_low_turnover_rank_dataset
from .low_turnover_rank_training import run_low_turnover_rank_training
from .low_turnover_rank_evaluation import (
    prepare_low_turnover_rank_evaluation,
    replay_low_turnover_rank_evaluation_prepare,
)
from .low_turnover_rank_evaluation_complete import (
    replay_low_turnover_rank_evaluation,
    unseal_low_turnover_rank_evaluation,
)
from .derivatives_data import run_derivatives_pipeline
from .derivatives_signal_study import run_derivatives_signal_study
from .dvol_data import run_dvol_pipeline
from .dvol_signal_study import run_dvol_signal_study
from .evidence_ledger import run_evidence_ledger
from .features import build_features_and_targets
from .final_audit import run_final_audit
from .holdout_protocol import run_holdout_protocol
from .intraday_path_study import run_intraday_path_signal_study
from .joint_absolute_relative_spec import run_joint_absolute_relative_spec
from .joint_absolute_relative_harness import run_joint_absolute_relative_harness
from .joint_absolute_relative_training import run_joint_absolute_relative_training
from .joint_absolute_relative_evaluation import (
    evaluate_joint_absolute_relative_evaluation,
    preflight_joint_absolute_relative_evaluation,
    prepare_joint_absolute_relative_evaluation,
    verify_joint_absolute_relative_evaluation,
)
from .multi_asset_scope import run_multi_asset_scope_amendment
from .non_target_pretraining import run_non_target_pretraining
from .non_target_inventory import run_non_target_inventory
from .non_target_dataset import run_non_target_dataset
from .override import run_override_suite
from .patch_transformer import run_patch_transformer_implementation
from .persistent_duration_spec import run_persistent_duration_spec
from .persistent_duration_harness import run_persistent_duration_harness
from .persistent_duration_dataset import run_persistent_duration_dataset
from .persistent_duration_training import run_persistent_duration_training
from .persistent_duration_evaluation import (
    finalize_failed_persistent_duration_evaluation_prepare,
    prepare_persistent_duration_evaluation,
    replay_persistent_duration_evaluation_prepare,
)
from .pipeline import run_experiment
from .risk_off import run_risk_off_suite
from .selected_source_manifest import run_selected_source_manifest
from .selected_universe_dataset import run_selected_universe_dataset
from .reproducibility import run_reproducibility_bundle, run_reproducibility_verifier
from .ranking_excess_spec import run_ranking_excess_spec
from .ranking_excess_harness import run_ranking_excess_harness
from .ranking_excess_development_screen import run_ranking_excess_development_screen
from .ranking_excess_failure_autopsy import (
    preflight_ranking_excess_failure_autopsy,
    run_ranking_excess_failure_autopsy,
)
from .ranking_excess_pretraining import run_ranking_excess_pretraining
from .ranking_excess_supervised import run_ranking_excess_supervised
from .research_review import run_research_review
from .research_workflow import research_doctor, validate_research_state
from .scientific_harness import run_scientific_harness
from .source_domain_one_shot import run_source_domain_one_shot
from .state_conditioned_multi_horizon_spec import (
    run_state_conditioned_multi_horizon_spec,
)
from .state_conditioned_multi_horizon_harness import (
    run_state_conditioned_multi_horizon_harness,
)
from .state_conditioned_multi_horizon_dataset import (
    run_state_conditioned_multi_horizon_dataset,
)
from .state_conditioned_multi_horizon_training import (
    run_state_conditioned_multi_horizon_training,
)
from .state_conditioned_multi_horizon_evaluation import (
    prepare_state_conditioned_multi_horizon_evaluation,
)
from .state_conditioned_multi_horizon_evaluation_complete import (
    replay_state_conditioned_multi_horizon_evaluation,
    unseal_state_conditioned_multi_horizon_evaluation,
)
from .supervised_non_target import run_supervised_non_target
from .signal_study import run_signal_study
from .treasury_feasibility import run_treasury_feasibility
from .treasury_data import run_treasury_pipeline
from .treasury_signal_study import run_treasury_signal_study
from .training_universe_inventory import run_training_universe_inventory
from .validation_suite import run_validation_suite
from .v37_failure_autopsy import run_v37_failure_autopsy
from .v50_economic_failure_autopsy import (
    preflight_v50_economic_failure_autopsy,
    run_v50_economic_failure_autopsy,
)
from .v59_retirement_autopsy import (
    preflight_v59_retirement_autopsy,
    run_v59_retirement_autopsy,
    verify_v59_retirement_autopsy,
)
from .zero_shot_spec import run_zero_shot_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tlm", description="TLM daily trading research MVP")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in (
        "data", "features", "baseline", "train", "smoke", "run", "audit",
        "consensus", "validate-suite", "override-suite", "risk-off-suite",
        "signal-study",
        "derivatives-data",
        "derivatives-signal-study",
        "intraday-path-study",
        "data-family-audit",
        "dvol-data",
        "dvol-signal-study",
        "cftc-feasibility",
        "treasury-feasibility",
        "treasury-data",
        "treasury-signal-study",
        "evidence-ledger",
        "control-certificate",
        "holdout-protocol",
        "reproducibility-bundle",
        "verify-bundle",
        "research-review",
        "final-audit",
        "zero-shot-spec",
        "non-target-inventory",
        "non-target-dataset",
        "multi-asset-scope",
        "training-universe-inventory",
        "selected-source-manifest",
        "selected-universe-dataset",
        "patch-transformer",
        "scientific-harness",
        "non-target-pretraining",
        "supervised-non-target",
        "source-domain-one-shot",
        "v37-failure-autopsy",
        "ranking-excess-spec",
        "ranking-excess-harness",
        "ranking-excess-pretraining-preflight",
        "ranking-excess-pretraining-smoke",
        "ranking-excess-pretraining",
        "ranking-excess-supervised-preflight",
        "ranking-excess-supervised-smoke",
        "ranking-excess-supervised",
        "ranking-excess-screen-preflight",
        "ranking-excess-screen-prepare",
        "ranking-excess-screen",
        "ranking-excess-failure-autopsy-preflight",
        "ranking-excess-failure-autopsy",
        "joint-absolute-relative-spec",
        "joint-absolute-relative-harness",
        "joint-absolute-relative-training-preflight",
        "joint-absolute-relative-training-smoke",
        "joint-absolute-relative-training",
        "joint-absolute-relative-training-verify",
        "joint-absolute-relative-evaluation-preflight",
        "joint-absolute-relative-evaluation-prepare",
        "joint-absolute-relative-evaluation",
        "joint-absolute-relative-evaluation-verify",
        "v50-economic-failure-autopsy-preflight",
        "v50-economic-failure-autopsy",
        "state-conditioned-multi-horizon-spec",
        "state-conditioned-multi-horizon-harness",
        "state-conditioned-multi-horizon-dataset",
        "state-conditioned-multi-horizon-training-preflight",
        "state-conditioned-multi-horizon-training-smoke",
        "state-conditioned-multi-horizon-training",
        "state-conditioned-multi-horizon-training-verify",
        "state-conditioned-multi-horizon-training-replay",
        "state-conditioned-multi-horizon-evaluation-prepare",
        "state-conditioned-multi-horizon-evaluation-unseal",
        "state-conditioned-multi-horizon-evaluation-replay",
        "v59-retirement-autopsy-preflight",
        "v59-retirement-autopsy",
        "v59-retirement-autopsy-verify",
        "decoupled-rank-state-spec",
        "decoupled-rank-state-harness",
        "decoupled-rank-state-dataset",
        "decoupled-rank-state-training-preflight",
        "decoupled-rank-state-training-smoke",
        "decoupled-rank-state-training",
        "decoupled-rank-state-training-verify",
        "decoupled-rank-state-training-replay",
        "decoupled-rank-state-evaluation-prepare",
        "decoupled-rank-state-evaluation-unseal",
        "decoupled-rank-state-evaluation-replay",
        "v64-r2-probabilistic-state-gate-spec",
        "v64-r2-probabilistic-state-gate-harness",
        "v64-r2-probabilistic-state-gate-dataset",
        "v64-r2-probabilistic-state-gate-training-preflight",
        "v64-r2-probabilistic-state-gate-training-smoke",
        "v64-r2-probabilistic-state-gate-training",
        "v64-r2-probabilistic-state-gate-training-verify",
        "v64-r2-probabilistic-state-gate-training-replay",
        "v64-r2-prospective-confirmation-prepare",
        "v64-r2-prospective-capture",
        "v64-r2-retrospective-diagnostic-prepare",
        "v64-r2-retrospective-diagnostic-unseal",
        "v64-r2-retrospective-diagnostic-replay",
        "v72-diagnostic-record",
        "v78-terminal-record",
        "low-turnover-rank-spec",
        "low-turnover-rank-harness",
        "low-turnover-rank-chronology-erratum",
        "low-turnover-rank-dataset",
        "low-turnover-rank-training-preflight",
        "low-turnover-rank-training-smoke",
        "low-turnover-rank-training",
        "low-turnover-rank-training-verify",
        "low-turnover-rank-training-replay",
        "low-turnover-rank-evaluation-prepare",
        "low-turnover-rank-evaluation-prepare-replay",
        "low-turnover-rank-evaluation-unseal",
        "low-turnover-rank-evaluation-replay",
        "persistent-duration-spec",
        "persistent-duration-harness",
        "persistent-duration-dataset",
        "persistent-duration-training-preflight",
        "persistent-duration-training-smoke",
        "persistent-duration-training",
        "persistent-duration-training-verify",
        "persistent-duration-training-replay",
        "persistent-duration-evaluation-prepare",
        "persistent-duration-evaluation-prepare-replay",
        "persistent-duration-evaluation-finalize-failure",
        "research-status",
        "research-doctor",
    ):
        command = subparsers.add_parser(name)
        if name in {"research-status", "research-doctor"}:
            command.add_argument("--root", default=".")
            command.add_argument("--state", default="research/current.yaml")
        else:
            command.add_argument("--config", default="configs/mvp.yaml")
        if name == "data":
            command.add_argument("--force", action="store_true")
            command.add_argument("--fixture", action="store_true")
        if name == "derivatives-data":
            command.add_argument("--force", action="store_true")
        if name == "dvol-data":
            command.add_argument("--force", action="store_true")
        if name == "treasury-data":
            command.add_argument("--force", action="store_true")
        if name == "non-target-inventory":
            command.add_argument("--force", action="store_true")
        if name == "training-universe-inventory":
            command.add_argument("--force", action="store_true")
        if name == "selected-source-manifest":
            command.add_argument("--force", action="store_true")
        if name == "non-target-pretraining":
            command.add_argument("--smoke", action="store_true")
        if name == "supervised-non-target":
            command.add_argument("--smoke", action="store_true")
        if name == "source-domain-one-shot":
            command.add_argument("--preflight", action="store_true")
        if name == "train":
            command.add_argument("--model", choices=("transformer",), default="transformer")
        if name == "run":
            command.add_argument("--seed", type=int)
            command.add_argument("--output-dir")
        if name == "consensus":
            command.add_argument("--retrain-members", action="store_true")
        if name == "validate-suite":
            command.add_argument("--retrain-scenarios", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "research-status":
        print(
            json.dumps(
                validate_research_state(args.root, args.state),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "research-doctor":
        print(
            json.dumps(
                research_doctor(args.root, args.state),
                indent=2,
                sort_keys=True,
            )
        )
        return
    config = load_config(args.config)
    if args.command in {
        "state-conditioned-multi-horizon-evaluation-prepare",
        "state-conditioned-multi-horizon-evaluation-unseal",
        "state-conditioned-multi-horizon-evaluation-replay",
        "decoupled-rank-state-evaluation-prepare",
        "decoupled-rank-state-evaluation-unseal",
        "decoupled-rank-state-evaluation-replay",
    }:
        config["_invoked_config_path"] = args.config
    if getattr(args, "seed", None) is not None:
        config["seed"] = args.seed
    if getattr(args, "output_dir", None):
        config["output_dir"] = args.output_dir
    if args.command == "smoke" or getattr(args, "fixture", False):
        config = smoke_config(config)
    if args.command == "data":
        frames = load_market_data(config, force=args.force)
        print(json.dumps({asset: len(frame) for asset, frame in frames.items()}, indent=2))
        return
    if args.command in {
        "persistent-duration-evaluation-prepare",
        "persistent-duration-evaluation-prepare-replay",
        "persistent-duration-evaluation-finalize-failure",
    }:
        replay = args.command.endswith("-replay")
        failure = args.command.endswith("-finalize-failure")
        if failure:
            result = finalize_failed_persistent_duration_evaluation_prepare(config)
        elif replay:
            result = replay_persistent_duration_evaluation_prepare(config)
        else:
            result = prepare_persistent_duration_evaluation(config)
        print(
            json.dumps(
                {
                    "mode": "failure-finalize" if failure else ("replay" if replay else "prepare"),
                    "decision": result.get("decision"),
                    "audit_passed": result.get("audit", {}).get("passed", result.get("passed")),
                    "summary": result.get("summary", {}),
                    "evaluation_spec_sha256": result.get("evaluation_spec_sha256"),
                    "prepare_receipt_sha256": result.get("prepare_receipt_sha256"),
                    "one_shot_packet_sha256": result.get("one_shot_packet_sha256"),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "features":
        frames = load_market_data(config)
        target_mode = config.get("target", {}).get("mode", "next_open_to_close")
        features, targets = build_features_and_targets(frames, target_mode=target_mode)
        print(json.dumps({
            "rows": len(features),
            "feature_count": len(features.columns),
            "assets": list(targets.columns),
        }, indent=2))
        return
    if args.command == "audit":
        print(json.dumps(audit_artifacts(config["output_dir"]), indent=2, sort_keys=True))
        return
    if args.command == "consensus":
        metrics = run_consensus_experiment(config, retrain_members=args.retrain_members)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return
    if args.command == "validate-suite":
        result = run_validation_suite(
            config, retrain_scenarios=args.retrain_scenarios
        )
        print(json.dumps({
            "scenario_count": result["scenario_count"],
            "all_scenarios_accepted": result["all_scenarios_accepted"],
            "tlm_consensus_status": result["tlm_consensus_status"],
            "recommended_shadow_control": result["recommended_shadow_control"],
            "audit_passed": result["audit"]["passed"],
        }, indent=2, sort_keys=True))
        return
    if args.command == "override-suite":
        result = run_override_suite(config)
        print(json.dumps({
            "scenario_count": result["scenario_count"],
            "all_scenarios_accepted": result["all_scenarios_accepted"],
            "candidate_status": result["candidate_status"],
            "control": result["control"],
            "audit_passed": result["audit"]["passed"],
        }, indent=2, sort_keys=True))
        return
    if args.command == "risk-off-suite":
        result = run_risk_off_suite(config)
        print(json.dumps({
            "scenario_count": result["scenario_count"],
            "all_offline_gates_passed": result["all_offline_gates_passed"],
            "candidate_status": result["candidate_status"],
            "control": result["control"],
            "audit_passed": result["audit"]["passed"],
        }, indent=2, sort_keys=True))
        return
    if args.command == "signal-study":
        result = run_signal_study(config)
        print(json.dumps({
            "conclusion": result["conclusion"],
            "robust_signals": result["robust_signals"],
            "signal_count": result["signal_count"],
            "scenario_count": result["scenario_count"],
            "audit_passed": result["audit"]["passed"],
        }, indent=2, sort_keys=True))
        return
    if args.command == "derivatives-data":
        result = run_derivatives_pipeline(config, force=args.force)
        print(json.dumps({
            "archive_count": result["archive_count"],
            "assets": result["assets"],
            "audit_passed": result["audit"]["passed"],
            "output_dir": result["output_dir"],
        }, indent=2, sort_keys=True))
        return
    if args.command == "derivatives-signal-study":
        result = run_derivatives_signal_study(config)
        print(json.dumps({
            "conclusion": result["conclusion"],
            "robust_signals": result["robust_signals"],
            "signal_count": result["signal_count"],
            "scenario_count": result["scenario_count"],
            "audit_passed": result["audit"]["passed"],
        }, indent=2, sort_keys=True))
        return
    if args.command == "intraday-path-study":
        result = run_intraday_path_signal_study(config)
        print(json.dumps({
            "conclusion": result["conclusion"],
            "robust_signals": result["robust_signals"],
            "signal_count": result["signal_count"],
            "scenario_count": result["scenario_count"],
            "path_data_audit_passed": result["path_data"]["audit_passed"],
            "audit_passed": result["audit"]["passed"],
        }, indent=2, sort_keys=True))
        return
    if args.command == "data-family-audit":
        result = run_data_family_audit(config)
        print(json.dumps({
            "decision": result["decision"],
            "selected": result["selected"],
            "dvol_probe": result["dvol_probe"],
            "audit_passed": result["audit"]["passed"],
        }, indent=2, sort_keys=True))
        return
    if args.command == "dvol-data":
        result = run_dvol_pipeline(config, force=args.force)
        print(json.dumps({
            "decision": result["decision"],
            "feature_rows": result["audit"]["feature_rows"],
            "feature_count": result["audit"]["feature_count"],
            "records": result["manifest"]["records"],
            "audit_passed": result["audit"]["passed"],
            "output_dir": result["output_dir"],
        }, indent=2, sort_keys=True))
        return
    if args.command == "dvol-signal-study":
        result = run_dvol_signal_study(config)
        print(json.dumps({
            "conclusion": result["conclusion"],
            "robust_signals": result["robust_signals"],
            "signal_count": result["signal_count"],
            "scenario_count": result["scenario_count"],
            "audit_passed": result["audit"]["passed"],
        }, indent=2, sort_keys=True))
        return
    if args.command == "cftc-feasibility":
        result = run_cftc_feasibility(config)
        print(json.dumps({
            "decision": result["decision"],
            "selected": result["selected"],
            "contracts": result["probe"]["contracts"],
            "hard_gates": result["hard_gates"],
            "audit_passed": result["audit"]["passed"],
        }, indent=2, sort_keys=True))
        return
    if args.command == "treasury-feasibility":
        result = run_treasury_feasibility(config)
        print(json.dumps({
            "decision": result["decision"],
            "selected": result["selected"],
            "probe": result["probe"],
            "hard_gates": result["hard_gates"],
            "audit_passed": result["audit"]["passed"],
        }, indent=2, sort_keys=True))
        return
    if args.command == "treasury-data":
        result = run_treasury_pipeline(config, force=args.force)
        print(json.dumps({
            "decision": result["decision"],
            "raw_rows": result["audit"]["raw_rows"],
            "state_rows": result["audit"]["state_rows"],
            "feature_count": result["audit"]["feature_count"],
            "records": result["manifest"]["records"],
            "audit_passed": result["audit"]["passed"],
        }, indent=2, sort_keys=True))
        return
    if args.command == "treasury-signal-study":
        result = run_treasury_signal_study(config)
        print(json.dumps({
            "conclusion": result["conclusion"],
            "robust_signals": result["robust_signals"],
            "signal_count": result["signal_count"],
            "scenario_count": result["scenario_count"],
            "audit_passed": result["audit"]["passed"],
        }, indent=2, sort_keys=True))
        return
    if args.command == "evidence-ledger":
        result = run_evidence_ledger(config)
        print(json.dumps({
            "decision": result["decision"],
            "synthesis": result["synthesis"],
            "audit_passed": result["audit"]["passed"],
        }, indent=2, sort_keys=True))
        return
    if args.command == "control-certificate":
        result = run_control_certificate(config)
        print(json.dumps({
            "decision": result["decision"],
            "benchmark_status": result["benchmark_status"],
            "deployment_status": result["deployment_status"],
            "risk_summary": result["risk_summary"],
            "audit_passed": result["audit"]["passed"],
        }, indent=2, sort_keys=True))
        return
    if args.command == "holdout-protocol":
        result = run_holdout_protocol(config)
        print(json.dumps({
            "decision": result["decision"],
            "state": result["state"],
            "clean_holdout_status": result["clean_holdout_status"],
            "timeline": result["timeline"],
            "audit_passed": result["audit"]["passed"],
        }, indent=2, sort_keys=True))
        return
    if args.command == "reproducibility-bundle":
        result = run_reproducibility_bundle(config)
        print(json.dumps({
            "decision": result["decision"],
            "file_count": result["manifest"]["file_count"],
            "manifest_sha256": result["manifest"]["manifest_sha256"],
            "passed_test_count": result["tests"]["passed_test_count"],
            "verification_passed": result["verification"]["passed"],
            "audit_passed": result["audit"]["passed"],
        }, indent=2, sort_keys=True))
        return
    if args.command == "verify-bundle":
        result = run_reproducibility_verifier(config)
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["passed"]:
            raise SystemExit(1)
        return
    if args.command == "research-review":
        result = run_research_review(config)
        print(json.dumps({
            "decision": result["decision"],
            "deployment_readiness": result["deployment_readiness"],
            "grades": result["grades"],
            "finding_count": len(result["findings"]),
            "audit_passed": result["audit"]["passed"],
        }, indent=2, sort_keys=True))
        return
    if args.command == "final-audit":
        result = run_final_audit(config)
        print(json.dumps({
            "decision": result["decision"],
            "system_status": result["system_status"],
            "evidence_summary": result["evidence_summary"],
            "audit_passed": result["audit"]["passed"],
        }, indent=2, sort_keys=True))
        return
    if args.command == "zero-shot-spec":
        result = run_zero_shot_spec(config)
        print(json.dumps({
            "decision": result["decision"],
            "blueprint_sha256": result["blueprint_sha256"],
            "tested": result["tested"],
            "registration_status": result["registration_draft"]["status"],
            "audit_passed": result["audit"]["passed"],
        }, indent=2, sort_keys=True))
        return
    if args.command == "non-target-inventory":
        result = run_non_target_inventory(config, force=args.force)
        print(json.dumps({
            "decision": result["decision"],
            "selected_count": result["universe"]["selected_count"],
            "selected_symbols": result["universe"]["selected_symbols"],
            "verified_archive_count": len(result["archive_manifest"]),
            "rejected_archive_count": len(result["archive_rejections"]),
            "tested": result["tested"],
            "audit_passed": result["audit"]["passed"],
        }, indent=2, sort_keys=True))
        return
    if args.command == "non-target-dataset":
        result = run_non_target_dataset(config)
        print(json.dumps({
            "decision": result["decision"],
            "panel_rows": result["dataset_manifest"]["panel_rows"],
            "panel_sha256": result["dataset_manifest"]["panel_sha256"],
            "source_rows": result["source_audit"]["source_rows"],
            "archive_count": result["source_audit"]["archive_count"],
            "tested": result["tested"],
            "audit_passed": result["audit"]["passed"],
        }, indent=2, sort_keys=True))
        return
    if args.command == "multi-asset-scope":
        result = run_multi_asset_scope_amendment(config)
        print(json.dumps({
            "decision": result["decision"],
            "blueprint_sha256": result["blueprint_sha256"],
            "training_asset_count": result["blueprint"]["training_universe"][
                "selected_asset_count"
            ],
            "target_symbols": result["blueprint"]["target_contract"][
                "symbols"
            ],
            "tested": result["tested"],
            "audit_passed": result["audit"]["passed"],
        }, indent=2, sort_keys=True))
        return
    if args.command == "training-universe-inventory":
        result = run_training_universe_inventory(config, force=args.force)
        print(json.dumps({
            "decision": result["decision"],
            "eligible_count": result["universe"]["eligible_count"],
            "selected_count": result["universe"]["selected_count"],
            "selected_symbols": result["universe"]["selected_symbols"],
            "asset_fold_count": result["asset_folds"]["fold_count"],
            "verified_selected_archive_count": len(result["archive_manifest"]),
            "tested": result["tested"],
            "audit_passed": result["audit"]["passed"],
        }, indent=2, sort_keys=True))
        return
    if args.command == "selected-source-manifest":
        result = run_selected_source_manifest(config, force=args.force)
        print(json.dumps({
            "decision": result["decision"],
            "selected_count": result["universe"]["selected_count"],
            "accepted_archive_count": result["manifest_summary"]["accepted_archive_count"],
            "rejected_archive_count": result["manifest_summary"]["rejected_archive_count"],
            "observed_rows": result["manifest_summary"]["observed_rows"],
            "preserved_missing_rows": result["manifest_summary"]["preserved_missing_rows"],
            "tested": result["tested"],
            "audit_passed": result["audit"]["passed"],
        }, indent=2, sort_keys=True))
        return
    if args.command == "selected-universe-dataset":
        result = run_selected_universe_dataset(config)
        print(json.dumps({
            "decision": result["decision"],
            "panel_rows": result["dataset_manifest"]["panel_rows"],
            "observed_raw_rows": result["dataset_manifest"]["observed_raw_rows"],
            "preserved_missing_rows": result["dataset_manifest"]["preserved_missing_rows"],
            "sequence_index_rows": result["dataset_manifest"]["sequence_index_rows"],
            "panel_sha256": result["dataset_manifest"]["panel_sha256"],
            "sequence_index_sha256": result["dataset_manifest"]["sequence_index_sha256"],
            "triplet_catalog_sha256": result["triplet_catalog"]["catalog_sha256"],
            "smoke_contract": result["smoke_contract"],
            "tested": result["tested"],
            "audit_passed": result["audit"]["passed"],
        }, indent=2, sort_keys=True))
        return
    if args.command == "patch-transformer":
        result = run_patch_transformer_implementation(config)
        print(json.dumps({
            "decision": result["decision"],
            "model_spec_sha256": result["model_spec"]["model_spec_sha256"],
            "parameter_count": result["smoke"]["parameter_count"],
            "patch_count": result["smoke"]["patch_count"],
            "checkpoint_sha256": result["smoke"]["checkpoint_sha256"],
            "tested": result["tested"],
            "audit_passed": result["audit"]["passed"],
        }, indent=2, sort_keys=True))
        return
    if args.command == "scientific-harness":
        result = run_scientific_harness(config)
        print(json.dumps({
            "decision": result["decision"],
            "harness_spec_sha256": result["harness_spec"]["harness_spec_sha256"],
            "optimizer_steps": result["smoke"]["optimizer_steps"],
            "masked_patches_per_sample": result["smoke"]["masked_patches_per_sample"],
            "bootstrap_cells": result["smoke"]["bootstrap_cells"],
            "bootstrap_paths_per_cell": result["smoke"]["bootstrap_paths_per_cell"],
            "tested": result["tested"],
            "audit_passed": result["audit"]["passed"],
        }, indent=2, sort_keys=True))
        return
    if args.command == "non-target-pretraining":
        result = run_non_target_pretraining(config, smoke=args.smoke)
        print(json.dumps({
            "decision": result["decision"],
            "pretraining_spec_sha256": result["pretraining_spec"][
                "pretraining_spec_sha256"
            ],
            "checkpoint_count": result["summary"]["checkpoint_count"],
            "fold_seed_jobs": result["summary"]["fold_seed_jobs"],
            "total_optimizer_steps": result["summary"]["total_optimizer_steps"],
            "tested": result["tested"],
            "audit_passed": result["audit"]["passed"],
        }, indent=2, sort_keys=True))
        return
    if args.command == "supervised-non-target":
        result = run_supervised_non_target(config, smoke=args.smoke)
        print(json.dumps({
            "decision": result["decision"],
            "supervised_spec_sha256": result["supervised_spec"][
                "supervised_spec_sha256"
            ],
            "checkpoint_count": result["summary"]["checkpoint_count"],
            "calibration_count": result["summary"]["calibration_count"],
            "total_optimizer_steps": result["summary"]["total_optimizer_steps"],
            "device": result["summary"]["device"],
            "tested": result["tested"],
            "audit_passed": result["audit"]["passed"],
        }, indent=2, sort_keys=True))
        return
    if args.command == "source-domain-one-shot":
        result = run_source_domain_one_shot(
            config, preflight_only=args.preflight
        )
        if args.preflight:
            payload = {
                "decision": result["decision"],
                "evaluation_spec_sha256": result["evaluation_spec"][
                    "evaluation_spec_sha256"
                ],
                "structural_checks": result["structural_checks"],
                "loaded": result["loaded"],
            }
        else:
            payload = {
                "decision": result["decision"],
                "evaluation_spec_sha256": result["evaluation_spec"][
                    "evaluation_spec_sha256"
                ],
                "summary": result["summary"],
                "audit_passed": result["audit"]["passed"],
            }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.command == "v37-failure-autopsy":
        result = run_v37_failure_autopsy(config)
        payload = {
            "decision": result["decision"],
            "autopsy_spec_sha256": result["autopsy_spec"][
                "autopsy_spec_sha256"
            ],
            "failure_attribution": result["failure_attribution"],
            "audit_passed": result["audit"]["passed"],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.command == "ranking-excess-spec":
        result = run_ranking_excess_spec(config)
        payload = {
            "decision": result["decision"],
            "blueprint_sha256": result["blueprint_sha256"],
            "parameter_count_analytic": result["blueprint"][
                "parameter_count_analytic"
            ],
            "audit_passed": result["audit"]["passed"],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.command == "ranking-excess-harness":
        result = run_ranking_excess_harness(config)
        payload = {
            "decision": result["decision"],
            "harness_spec_sha256": result["harness_spec"][
                "harness_spec_sha256"
            ],
            "parameter_count": result["smoke"]["parameter_count"],
            "audit_passed": result["audit"]["passed"],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.command in {
        "ranking-excess-pretraining-preflight",
        "ranking-excess-pretraining-smoke",
        "ranking-excess-pretraining",
    }:
        mode = {
            "ranking-excess-pretraining-preflight": "preflight",
            "ranking-excess-pretraining-smoke": "smoke",
            "ranking-excess-pretraining": "full",
        }[args.command]
        result = run_ranking_excess_pretraining(config, mode)
        payload = {
            "decision": result["decision"],
            "mode": mode,
            "pretraining_spec_sha256": result["pretraining_spec"][
                "pretraining_spec_sha256"
            ],
            "checkpoint_count": result["summary"]["checkpoint_count"],
            "optimizer_steps": result["summary"]["total_optimizer_steps"],
            "audit_passed": result["audit"]["passed"],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.command == "state-conditioned-multi-horizon-evaluation-prepare":
        result = prepare_state_conditioned_multi_horizon_evaluation(config)
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "audit_passed": result["audit"]["passed"],
                    "summary": result["summary"],
                    "invocation": result["invocation"],
                    "verification": result["verification"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "state-conditioned-multi-horizon-evaluation-unseal":
        result = unseal_state_conditioned_multi_horizon_evaluation(config)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "state-conditioned-multi-horizon-evaluation-replay":
        result = replay_state_conditioned_multi_horizon_evaluation(config)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "decoupled-rank-state-evaluation-prepare":
        result = prepare_decoupled_rank_state_evaluation(config)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "decoupled-rank-state-evaluation-unseal":
        result = unseal_decoupled_rank_state_evaluation(config)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "decoupled-rank-state-evaluation-replay":
        result = replay_decoupled_rank_state_evaluation(config)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command in {
        "joint-absolute-relative-evaluation-preflight",
        "joint-absolute-relative-evaluation-prepare",
        "joint-absolute-relative-evaluation",
        "joint-absolute-relative-evaluation-verify",
    }:
        mode = {
            "joint-absolute-relative-evaluation-preflight": "preflight",
            "joint-absolute-relative-evaluation-prepare": "prepare",
            "joint-absolute-relative-evaluation": "evaluate",
            "joint-absolute-relative-evaluation-verify": "verify",
        }[args.command]
        runner = {
            "preflight": preflight_joint_absolute_relative_evaluation,
            "prepare": prepare_joint_absolute_relative_evaluation,
            "evaluate": evaluate_joint_absolute_relative_evaluation,
            "verify": verify_joint_absolute_relative_evaluation,
        }[mode]
        result = runner(config)
        print(json.dumps({
            "mode": mode,
            "decision": result["decision"],
            "evaluation_spec_sha256": result["evaluation_spec"][
                "evaluation_spec_sha256"
            ],
            "summary": result["summary"],
            "audit_passed": result["audit"]["passed"],
        }, indent=2, sort_keys=True))
        return
    if args.command in {
        "ranking-excess-supervised-preflight",
        "ranking-excess-supervised-smoke",
        "ranking-excess-supervised",
    }:
        mode = {
            "ranking-excess-supervised-preflight": "preflight",
            "ranking-excess-supervised-smoke": "smoke",
            "ranking-excess-supervised": "full",
        }[args.command]
        result = run_ranking_excess_supervised(config, mode)
        payload = {
            "decision": result["decision"],
            "mode": mode,
            "supervised_spec_sha256": result["supervised_spec"][
                "supervised_spec_sha256"
            ],
            "checkpoint_count": result["summary"]["checkpoint_count"],
            "optimizer_steps": result["summary"]["total_optimizer_steps"],
            "audit_passed": result["audit"]["passed"],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.command in {
        "ranking-excess-screen-preflight",
        "ranking-excess-screen-prepare",
        "ranking-excess-screen",
    }:
        mode = {
            "ranking-excess-screen-preflight": "preflight",
            "ranking-excess-screen-prepare": "prepare",
            "ranking-excess-screen": "evaluate",
        }[args.command]
        result = run_ranking_excess_development_screen(config, mode)
        payload = {
            "decision": result["decision"],
            "mode": mode,
            "evaluation_spec_sha256": result["evaluation_spec"][
                "evaluation_spec_sha256"
            ],
            "evaluation_execution_count": int(
                result.get("evaluation_execution_count", 0)
            ),
            "audit_passed": result["audit"]["passed"],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.command in {
        "ranking-excess-failure-autopsy-preflight",
        "ranking-excess-failure-autopsy",
    }:
        mode = (
            "preflight"
            if args.command == "ranking-excess-failure-autopsy-preflight"
            else "run"
        )
        result = (
            preflight_ranking_excess_failure_autopsy(config)
            if mode == "preflight"
            else run_ranking_excess_failure_autopsy(config)
        )
        payload = {
            "decision": result["decision"],
            "mode": mode,
            "autopsy_spec_sha256": result["autopsy_spec"][
                "autopsy_spec_sha256"
            ],
            "audit_passed": result["audit"]["passed"],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.command in {
        "v59-retirement-autopsy-preflight",
        "v59-retirement-autopsy",
        "v59-retirement-autopsy-verify",
    }:
        contract_path = args.config
        if args.command == "v59-retirement-autopsy-preflight":
            result = preflight_v59_retirement_autopsy(contract_path)
        elif args.command == "v59-retirement-autopsy":
            result = run_v59_retirement_autopsy(contract_path)
        else:
            result = verify_v59_retirement_autopsy(contract_path)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command in {
        "v50-economic-failure-autopsy-preflight",
        "v50-economic-failure-autopsy",
    }:
        mode = (
            "preflight"
            if args.command == "v50-economic-failure-autopsy-preflight"
            else "run"
        )
        result = (
            preflight_v50_economic_failure_autopsy(config)
            if mode == "preflight"
            else run_v50_economic_failure_autopsy(config)
        )
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "mode": mode,
                    "autopsy_spec_sha256": result["autopsy_spec"][
                        "autopsy_spec_sha256"
                    ],
                    "audit_passed": result["audit"]["passed"],
                    "summary": result["summary"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "state-conditioned-multi-horizon-spec":
        result = run_state_conditioned_multi_horizon_spec(config)
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "blueprint_sha256": result["blueprint_sha256"],
                    "audit_passed": result["audit"]["passed"],
                    "summary": result["summary"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "decoupled-rank-state-spec":
        result = run_decoupled_rank_state_spec(config)
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "blueprint_sha256": result["blueprint_sha256"],
                    "audit_passed": result["audit"]["passed"],
                    "summary": result["summary"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "persistent-duration-spec":
        result = run_persistent_duration_spec(config)
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "blueprint_sha256": result["blueprint_sha256"],
                    "audit_passed": result["audit"]["passed"],
                    "summary": result["summary"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "persistent-duration-harness":
        result = run_persistent_duration_harness(config)
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "harness_spec_sha256": result["harness_spec_sha256"],
                    "parameter_count": result["smoke"]["parameter_count"],
                    "optimizer_steps": result["smoke"][
                        "optimizer_steps_executed"
                    ],
                    "audit_passed": result["audit"]["passed"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "persistent-duration-dataset":
        result = run_persistent_duration_dataset(config)
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "dataset_spec_sha256": result["dataset_spec_sha256"],
                    "dataset_manifest_sha256": result[
                        "dataset_manifest_sha256"
                    ],
                    "summary": result["summary"],
                    "audit_passed": result["audit"]["passed"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command in {
        "persistent-duration-training-preflight",
        "persistent-duration-training-smoke",
        "persistent-duration-training",
        "persistent-duration-training-verify",
        "persistent-duration-training-replay",
    }:
        mode = {
            "persistent-duration-training-preflight": "preflight",
            "persistent-duration-training-smoke": "smoke",
            "persistent-duration-training": "full",
            "persistent-duration-training-verify": "verify",
            "persistent-duration-training-replay": "replay",
        }[args.command]
        result = run_persistent_duration_training(config, mode=mode)
        print(
            json.dumps(
                {
                    "mode": mode,
                    "decision": result.get("decision"),
                    "audit_passed": result.get("audit", {}).get("passed"),
                    "summary": result.get("summary", {}),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "v64-r2-probabilistic-state-gate-spec":
        result = run_v64_r2_probabilistic_state_gate_spec(config)
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "blueprint_sha256": result["blueprint_sha256"],
                    "audit_passed": result["audit"]["passed"],
                    "summary": result["summary"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "v64-r2-probabilistic-state-gate-harness":
        result = run_v64_r2_probabilistic_state_gate_harness(config)
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "harness_spec_sha256": result["harness_spec_sha256"],
                    "ranker_parameters": result["smoke"]["ranker_parameters"],
                    "state_gate_parameters": result["smoke"][
                        "state_gate_parameters"
                    ],
                    "optimizer_steps": result["smoke"][
                        "optimizer_steps_executed"
                    ],
                    "audit_passed": result["audit"]["passed"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "v64-r2-probabilistic-state-gate-dataset":
        result = run_v64_r2_probabilistic_state_gate_dataset(config)
        manifest = result["dataset_manifest"]
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "dataset_spec_sha256": result["dataset_spec"][
                        "dataset_spec_sha256"
                    ],
                    "labels_rows": manifest["labels"]["rows"],
                    "sequence_role_rows": manifest["sequence_roles"]["rows"],
                    "audit_passed": result["audit"]["passed"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command in {
        "v64-r2-probabilistic-state-gate-training-preflight",
        "v64-r2-probabilistic-state-gate-training-smoke",
        "v64-r2-probabilistic-state-gate-training",
        "v64-r2-probabilistic-state-gate-training-verify",
        "v64-r2-probabilistic-state-gate-training-replay",
    }:
        mode = {
            "v64-r2-probabilistic-state-gate-training-preflight": "preflight",
            "v64-r2-probabilistic-state-gate-training-smoke": "smoke",
            "v64-r2-probabilistic-state-gate-training": "full",
            "v64-r2-probabilistic-state-gate-training-verify": "verify",
            "v64-r2-probabilistic-state-gate-training-replay": "replay",
        }[args.command]
        result = run_v64_r2_probabilistic_state_gate_training(config, mode=mode)
        print(
            json.dumps(
                {
                    "mode": mode,
                    "decision": result.get("decision"),
                    "audit_passed": result.get("audit", {}).get("passed"),
                    "summary": result.get("summary", {}),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "v64-r2-prospective-confirmation-prepare":
        result = run_v64_r2_prospective_confirmation_prepare(config)
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "protocol_sha256": result["protocol_sha256"],
                    "audit_passed": result["audit"]["passed"],
                    "summary": result["summary"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "v64-r2-prospective-capture":
        result = run_v64_r2_prospective_capture(config)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "v64-r2-retrospective-diagnostic-prepare":
        result = run_v64_r2_retrospective_diagnostic_prepare(config)
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "evaluation_spec_sha256": result["evaluation_spec_sha256"],
                    "prepare_receipt_sha256": result["prepare_receipt_sha256"],
                    "registered_sha256": result["registered_sha256"],
                    "outcomes_remain_sealed": result["outcomes_remain_sealed"],
                    "outcome_packet_reads": result["outcome_packet_reads"],
                    "performance_or_pnl_computed": result[
                        "performance_or_pnl_computed"
                    ],
                    "target_assets_loaded": result["target_assets_loaded"],
                    "required_exact_user_authorization": result[
                        "required_exact_user_authorization"
                    ],
                    "replay": result.get("replay"),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "v64-r2-retrospective-diagnostic-unseal":
        result = unseal_v64_r2_retrospective_diagnostic(config)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "v64-r2-retrospective-diagnostic-replay":
        result = replay_v64_r2_retrospective_diagnostic(config)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "v72-diagnostic-record":
        result = run_v72_diagnostic_record(config)
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "record_sha256": result["record"]["record_sha256"],
                    "audit_passed": result["audit"]["passed"],
                    "access_ledger": result["audit"]["access_ledger"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "v78-terminal-record":
        result = run_v78_terminal_record(config)
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "record_sha256": result["record"]["record_sha256"],
                    "result_sha256": result["result"]["result_sha256"],
                    "audit_passed": result["audit"]["passed"],
                    "access_ledger": result["audit"]["access_ledger"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "low-turnover-rank-spec":
        result = run_low_turnover_rank_spec(config)
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "specification_sha256": result["specification"][
                        "specification_sha256"
                    ],
                    "blueprint_sha256": result["blueprint"]["blueprint_sha256"],
                    "parameter_count": result["result"]["parameter_count"],
                    "structural_maximum_evaluation_turnover": result["result"][
                        "structural_maximum_evaluation_turnover"
                    ],
                    "audit_passed": result["audit"]["passed"],
                    "audit_checks_passed": result["audit"]["checks_passed"],
                    "access_ledger": result["audit"]["access_ledger"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "low-turnover-rank-harness":
        result = run_low_turnover_rank_harness(config)
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "harness_spec_sha256": result["harness_spec"][
                        "harness_spec_sha256"
                    ],
                    "parameter_count": result["result"]["parameter_count"],
                    "adversarial_turnover": result["result"][
                        "adversarial_turnover"
                    ],
                    "audit_passed": result["audit"]["passed"],
                    "audit_checks_passed": result["audit"]["checks_passed"],
                    "access_ledger": result["audit"]["access_ledger"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "low-turnover-rank-chronology-erratum":
        result = run_low_turnover_rank_chronology_erratum(config)
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "erratum_sha256": result["erratum"]["erratum_sha256"],
                    "final_evaluation_signal_end": result["result"][
                        "final_evaluation_signal_end"
                    ],
                    "final_evaluation_signal_dates": result["result"][
                        "final_evaluation_signal_dates"
                    ],
                    "maximum_evaluation_decisions": result["result"][
                        "maximum_evaluation_decisions"
                    ],
                    "structural_maximum_turnover": result["result"][
                        "structural_maximum_turnover"
                    ],
                    "audit_passed": result["audit"]["passed"],
                    "audit_checks_passed": result["audit"]["checks_passed"],
                    "access_ledger": result["audit"]["access_ledger"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "low-turnover-rank-dataset":
        result = run_low_turnover_rank_dataset(config)
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "dataset_spec_sha256": result["dataset_spec_sha256"],
                    "dataset_manifest_sha256": result["dataset_manifest_sha256"],
                    "source_manifest_sha256": result["source_manifest_sha256"],
                    "sealed_packet_receipt_sha256": result[
                        "sealed_packet_receipt_sha256"
                    ],
                    "summary": result["summary"],
                    "audit_passed": result["audit"]["passed"],
                    "audit_checks_passed": result["audit"]["checks_passed"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command in {
        "low-turnover-rank-evaluation-prepare",
        "low-turnover-rank-evaluation-prepare-replay",
    }:
        replay = args.command.endswith("-replay")
        result = (
            replay_low_turnover_rank_evaluation_prepare(config)
            if replay
            else prepare_low_turnover_rank_evaluation(config)
        )
        print(json.dumps({
            "mode": "replay" if replay else "prepare",
            "decision": result.get("decision"),
            "audit_passed": result.get("audit", {}).get("passed"),
            "summary": result.get("summary", {}),
            "evaluation_spec_sha256": result.get("evaluation_spec_sha256"),
            "prepare_receipt_sha256": result.get("prepare_receipt_sha256"),
            "one_shot_packet_sha256": result.get("one_shot_packet_sha256"),
        }, indent=2, sort_keys=True))
        return
    if args.command in {
        "low-turnover-rank-evaluation-unseal",
        "low-turnover-rank-evaluation-replay",
    }:
        replay = args.command.endswith("-replay")
        result = (
            replay_low_turnover_rank_evaluation(config)
            if replay
            else unseal_low_turnover_rank_evaluation(config)
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command in {
        "low-turnover-rank-training-preflight",
        "low-turnover-rank-training-smoke",
        "low-turnover-rank-training",
        "low-turnover-rank-training-verify",
        "low-turnover-rank-training-replay",
    }:
        mode = {
            "low-turnover-rank-training-preflight": "preflight",
            "low-turnover-rank-training-smoke": "smoke",
            "low-turnover-rank-training": "full",
            "low-turnover-rank-training-verify": "verify",
            "low-turnover-rank-training-replay": "replay",
        }[args.command]
        result = run_low_turnover_rank_training(config, mode=mode)
        print(json.dumps({
            "mode": mode,
            "decision": result.get("decision"),
            "audit_passed": result.get("audit", {}).get("passed"),
            "summary": result.get("summary", {}),
        }, indent=2, sort_keys=True))
        return
    if args.command == "decoupled-rank-state-harness":
        result = run_decoupled_rank_state_harness(config)
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "harness_spec_sha256": result["harness_spec_sha256"],
                    "ranker_parameters": result["smoke"]["ranker_parameters"],
                    "state_gate_parameters": result["smoke"][
                        "state_gate_parameters"
                    ],
                    "optimizer_steps": result["smoke"][
                        "optimizer_steps_executed"
                    ],
                    "audit_passed": result["audit"]["passed"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "decoupled-rank-state-dataset":
        result = run_decoupled_rank_state_dataset(config)
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "dataset_spec_sha256": result["dataset_spec"][
                        "dataset_spec_sha256"
                    ],
                    "labels_sha256": result["dataset_manifest"]["labels"][
                        "sha256"
                    ],
                    "sequence_roles_sha256": result["dataset_manifest"][
                        "sequence_roles"
                    ]["sha256"],
                    "audit_passed": result["audit"]["passed"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command in {
        "decoupled-rank-state-training-preflight",
        "decoupled-rank-state-training-smoke",
        "decoupled-rank-state-training",
        "decoupled-rank-state-training-verify",
        "decoupled-rank-state-training-replay",
    }:
        mode = {
            "decoupled-rank-state-training-preflight": "preflight",
            "decoupled-rank-state-training-smoke": "smoke",
            "decoupled-rank-state-training": "full",
            "decoupled-rank-state-training-verify": "verify",
            "decoupled-rank-state-training-replay": "replay",
        }[args.command]
        result = run_decoupled_rank_state_training(config, mode=mode)
        print(
            json.dumps(
                {
                    "mode": mode,
                    "decision": result.get("decision"),
                    "audit_passed": result.get("audit", {}).get("passed"),
                    "summary": result.get("summary", {}),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "state-conditioned-multi-horizon-harness":
        result = run_state_conditioned_multi_horizon_harness(config)
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "harness_spec_sha256": result["harness_spec"][
                        "harness_spec_sha256"
                    ],
                    "parameter_count": result["smoke"]["parameter_count"],
                    "optimizer_steps": result["smoke"]["optimizer_steps"],
                    "audit_passed": result["audit"]["passed"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "state-conditioned-multi-horizon-dataset":
        result = run_state_conditioned_multi_horizon_dataset(config)
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "dataset_spec_sha256": result["dataset_spec"][
                        "dataset_spec_sha256"
                    ],
                    "labels_sha256": result["dataset_manifest"]["labels"][
                        "sha256"
                    ],
                    "sequence_roles_sha256": result["dataset_manifest"][
                        "sequence_roles"
                    ]["sha256"],
                    "audit_passed": result["audit"]["passed"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command in {
        "state-conditioned-multi-horizon-training-preflight",
        "state-conditioned-multi-horizon-training-smoke",
        "state-conditioned-multi-horizon-training",
        "state-conditioned-multi-horizon-training-verify",
        "state-conditioned-multi-horizon-training-replay",
    }:
        mode = {
            "state-conditioned-multi-horizon-training-preflight": "preflight",
            "state-conditioned-multi-horizon-training-smoke": "smoke",
            "state-conditioned-multi-horizon-training": "full",
            "state-conditioned-multi-horizon-training-verify": "verify",
            "state-conditioned-multi-horizon-training-replay": "replay",
        }[args.command]
        result = run_state_conditioned_multi_horizon_training(config, mode=mode)
        print(
            json.dumps(
                {
                    "mode": mode,
                    "decision": result.get("decision"),
                    "audit_passed": result.get("audit", {}).get("passed"),
                    "summary": result.get("summary", {}),
                    "invocation": result.get("invocation", {}),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "joint-absolute-relative-spec":
        result = run_joint_absolute_relative_spec(config)
        payload = {
            "decision": result["decision"],
            "blueprint_sha256": result["blueprint_sha256"],
            "parameter_count_analytic": result["blueprint"][
                "parameter_count_analytic"
            ],
            "registered_job_count": result["blueprint"]["registered_job_count"],
            "audit_passed": result["audit"]["passed"],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.command == "joint-absolute-relative-harness":
        result = run_joint_absolute_relative_harness(config)
        payload = {
            "decision": result["decision"],
            "harness_spec_sha256": result["harness_spec"][
                "harness_spec_sha256"
            ],
            "parameter_count": result["smoke"]["parameter_count"],
            "optimizer_steps": result["smoke"]["optimizer_steps"],
            "audit_passed": result["audit"]["passed"],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    if args.command in {
        "joint-absolute-relative-training-preflight",
        "joint-absolute-relative-training-smoke",
        "joint-absolute-relative-training",
        "joint-absolute-relative-training-verify",
    }:
        mode = {
            "joint-absolute-relative-training-preflight": "preflight",
            "joint-absolute-relative-training-smoke": "smoke",
            "joint-absolute-relative-training": "full",
            "joint-absolute-relative-training-verify": "verify",
        }[args.command]
        result = run_joint_absolute_relative_training(config, mode)
        payload = {
            "decision": result["decision"],
            "mode": mode,
            "contract_sha256": result["training_spec"]["contract_sha256"],
            "checkpoint_count": result["summary"]["checkpoint_count"],
            "optimizer_steps": result["summary"]["total_optimizer_steps"],
            "audit_passed": result["audit"]["passed"],
        }
        if "invocation" in result:
            payload["invocation"] = result["invocation"]
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    model_map = {
        "baseline": ("ridge",),
        "train": ("transformer",),
        "smoke": ("ridge", "transformer"),
        "run": ("ridge", "transformer"),
    }
    metrics = run_experiment(config, models=model_map[args.command])
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
