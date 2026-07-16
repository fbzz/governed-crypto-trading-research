.PHONY: install test smoke run status doctor public-snapshot public-release preflight-v77 smoke-v77 run-v77 verify-v77 replay-v77 run-v2 run-v3 run-v4 run-v5 validate-v6 run-v7 run-v8 run-v9 run-v10 run-v11 run-v12 run-v13 run-v14 run-v15 run-v16 run-v17 run-v18 run-v19 run-v20 run-v21 run-v22 run-v23 verify-v23 run-v24 run-v25 run-v26 run-v27 run-v28 run-v29 run-v30 run-v31 run-v32 run-v33 run-v34 smoke-v35 run-v35 smoke-v36 run-v36 preflight-v37 run-v37 run-v37-autopsy run-v41 run-v42 preflight-v43 smoke-v43 run-v43 preflight-v44 smoke-v44 run-v44 preflight-v45 prepare-v45 run-v45 preflight-v46 run-v46 run-v47 run-v48 preflight-v49 smoke-v49 run-v49 verify-v49 preflight-v50 prepare-v50 run-v50 verify-v50 preflight-v54 run-v54 run-v55 run-v56 run-v57 preflight-v58 smoke-v58 run-v58 verify-v58 replay-v58 run-v60 run-v61 run-v62 audit audit-v2 audit-v3 audit-v4 audit-v5 clean

install:
	python3 -m pip install -e '.[dev]'

test:
	python3 -m pytest

smoke:
	PYTHONPATH=src python3 -m tlm smoke --config configs/mvp.yaml

run:
	PYTHONPATH=src python3 -m tlm run --config configs/mvp.yaml

status:
	PYTHONPATH=src python3 -m tlm research-status

doctor:
	PYTHONPATH=src python3 -m tlm research-doctor

public-snapshot:
	python3 scripts/build_public_snapshot.py

public-release:
	python3 scripts/build_public_snapshot.py
	python3 scripts/seed_public_repo.py

preflight-v77:
	PYTHONPATH=src python3 -m tlm persistent-duration-training-preflight --config configs/v77_persistent_duration_training.yaml

smoke-v77:
	PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src python3 -m tlm persistent-duration-training-smoke --config configs/v77_persistent_duration_training.yaml

run-v77:
	PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src python3 -m tlm persistent-duration-training --config configs/v77_persistent_duration_training.yaml

verify-v77:
	PYTHONPATH=src python3 -m tlm persistent-duration-training-verify --config configs/v77_persistent_duration_training.yaml

replay-v77:
	PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src python3 -m tlm persistent-duration-training-replay --config configs/v77_persistent_duration_training.yaml

run-v2:
	PYTHONPATH=src python3 -m tlm run --config configs/v2_open_to_open.yaml

run-v3:
	PYTHONPATH=src python3 -m tlm run --config configs/v3_always_long_top1.yaml

run-v4:
	PYTHONPATH=src python3 -m tlm run --config configs/v4_cross_sectional_ranking.yaml

run-v5:
	PYTHONPATH=src python3 -m tlm consensus --config configs/v5_consensus.yaml

validate-v6:
	PYTHONPATH=src python3 -m tlm validate-suite --config configs/v6_validation_suite.yaml

run-v7:
	PYTHONPATH=src python3 -m tlm override-suite --config configs/v7_override_net.yaml

run-v8:
	PYTHONPATH=src python3 -m tlm risk-off-suite --config configs/v8_risk_off.yaml

run-v9:
	PYTHONPATH=src python3 -m tlm signal-study --config configs/v9_signal_existence.yaml

run-v10:
	PYTHONPATH=src python3 -m tlm derivatives-data --config configs/v10_derivatives_data.yaml

run-v11:
	PYTHONPATH=src python3 -m tlm derivatives-signal-study --config configs/v11_derivatives_signal_existence.yaml

run-v12:
	PYTHONPATH=src python3 -m tlm intraday-path-study --config configs/v12_intraday_path_signal_existence.yaml

run-v13:
	PYTHONPATH=src python3 -m tlm data-family-audit --config configs/v13_data_family_feasibility.yaml

run-v14:
	PYTHONPATH=src python3 -m tlm dvol-data --config configs/v14_dvol_data.yaml

run-v15:
	PYTHONPATH=src python3 -m tlm dvol-signal-study --config configs/v15_dvol_signal_existence.yaml

run-v16:
	PYTHONPATH=src python3 -m tlm cftc-feasibility --config configs/v16_cftc_positioning_feasibility.yaml

run-v17:
	PYTHONPATH=src python3 -m tlm treasury-feasibility --config configs/v17_treasury_curve_feasibility.yaml

run-v18:
	PYTHONPATH=src python3 -m tlm treasury-data --config configs/v18_treasury_curve_data.yaml

run-v19:
	PYTHONPATH=src python3 -m tlm treasury-signal-study --config configs/v19_treasury_signal_existence.yaml

run-v20:
	PYTHONPATH=src python3 -m tlm evidence-ledger --config configs/v20_evidence_ledger.yaml

run-v21:
	PYTHONPATH=src python3 -m tlm control-certificate --config configs/v21_control_certificate.yaml

run-v22:
	PYTHONPATH=src python3 -m tlm holdout-protocol --config configs/v22_prospective_holdout.yaml

run-v23:
	PYTHONPATH=src python3 -m tlm reproducibility-bundle --config configs/v23_reproducibility_bundle.yaml

verify-v23:
	PYTHONPATH=src python3 -m tlm verify-bundle --config configs/v23_reproducibility_bundle.yaml

run-v24:
	PYTHONPATH=src python3 -m tlm research-review --config configs/v24_research_review.yaml

run-v25:
	PYTHONPATH=src python3 -m tlm final-audit --config configs/v25_final_audit.yaml

run-v26:
	PYTHONPATH=src python3 -m tlm zero-shot-spec --config configs/v26_zero_shot_candidate_spec.yaml

run-v27:
	PYTHONPATH=src python3 -m tlm non-target-inventory --config configs/v27_non_target_universe_audit.yaml

run-v28:
	PYTHONPATH=src python3 -m tlm non-target-dataset --config configs/v28_non_target_dataset.yaml

run-v29:
	PYTHONPATH=src python3 -m tlm multi-asset-scope --config configs/v29_multi_asset_scope_amendment.yaml

run-v30:
	PYTHONPATH=src python3 -m tlm training-universe-inventory --config configs/v30_training_universe_inventory.yaml

run-v31:
	PYTHONPATH=src python3 -m tlm selected-source-manifest --config configs/v31_selected_source_manifest.yaml

run-v32:
	PYTHONPATH=src python3 -m tlm selected-universe-dataset --config configs/v32_selected_universe_dataset.yaml

run-v33:
	PYTHONPATH=src python3 -m tlm patch-transformer --config configs/v33_patch_transformer.yaml

run-v34:
	PYTHONPATH=src python3 -m tlm scientific-harness --config configs/v34_scientific_harness.yaml

smoke-v35:
	PYTHONPATH=src python3 -m tlm non-target-pretraining --config configs/v35_non_target_pretraining.yaml --smoke

run-v35:
	PYTHONPATH=src python3 -m tlm non-target-pretraining --config configs/v35_non_target_pretraining.yaml

smoke-v36:
	PYTHONPATH=src python3 -m tlm supervised-non-target --config configs/v36_supervised_non_target.yaml --smoke

run-v36:
	PYTHONPATH=src python3 -m tlm supervised-non-target --config configs/v36_supervised_non_target.yaml

preflight-v37:
	PYTHONPATH=src python3 -m tlm source-domain-one-shot --config configs/v37_source_domain_one_shot.yaml --preflight

run-v37:
	PYTHONPATH=src python3 -m tlm source-domain-one-shot --config configs/v37_source_domain_one_shot.yaml

run-v37-autopsy:
	PYTHONPATH=src python3 -m tlm v37-failure-autopsy --config configs/v37_failure_autopsy.yaml

run-v41:
	PYTHONPATH=src python3 -m tlm ranking-excess-spec --config configs/v41_ranking_excess_spec.yaml

run-v42:
	PYTHONPATH=src python3 -m tlm ranking-excess-harness --config configs/v42_ranking_excess_harness.yaml

preflight-v43:
	PYTHONPATH=src python3 -m tlm ranking-excess-pretraining-preflight --config configs/v43_ranking_excess_pretraining.yaml

smoke-v43:
	PYTHONPATH=src python3 -m tlm ranking-excess-pretraining-smoke --config configs/v43_ranking_excess_pretraining.yaml

run-v43:
	PYTHONPATH=src python3 -m tlm ranking-excess-pretraining --config configs/v43_ranking_excess_pretraining.yaml

preflight-v44:
	PYTHONPATH=src python3 -m tlm ranking-excess-supervised-preflight --config configs/v44_ranking_excess_supervised.yaml

smoke-v44:
	PYTHONPATH=src python3 -m tlm ranking-excess-supervised-smoke --config configs/v44_ranking_excess_supervised.yaml

run-v44:
	PYTHONPATH=src python3 -m tlm ranking-excess-supervised --config configs/v44_ranking_excess_supervised.yaml

preflight-v45:
	PYTHONPATH=src python3 -m tlm ranking-excess-screen-preflight --config configs/v45_ranking_excess_screen.yaml

prepare-v45:
	PYTHONPATH=src python3 -m tlm ranking-excess-screen-prepare --config configs/v45_ranking_excess_screen.yaml

run-v45:
	PYTHONPATH=src python3 -m tlm ranking-excess-screen --config configs/v45_ranking_excess_screen.yaml

preflight-v46:
	PYTHONPATH=src python3 -m tlm ranking-excess-failure-autopsy-preflight --config configs/v46_ranking_excess_failure_autopsy.yaml

run-v46:
	PYTHONPATH=src python3 -m tlm ranking-excess-failure-autopsy --config configs/v46_ranking_excess_failure_autopsy.yaml

run-v47:
	PYTHONPATH=src python3 -m tlm joint-absolute-relative-spec --config configs/v47_joint_absolute_relative_triplet_spec.yaml

run-v48:
	PYTHONPATH=src python3 -m tlm joint-absolute-relative-harness --config configs/v48_joint_absolute_relative_harness.yaml

preflight-v49:
	PYTHONPATH=src python3 -m tlm joint-absolute-relative-training-preflight --config configs/v49_joint_absolute_relative_training.yaml

smoke-v49:
	PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src python3 -m tlm joint-absolute-relative-training-smoke --config configs/v49_joint_absolute_relative_training.yaml

run-v49:
	PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src python3 -m tlm joint-absolute-relative-training --config configs/v49_joint_absolute_relative_training.yaml

verify-v49:
	PYTHONPATH=src python3 -m tlm joint-absolute-relative-training-verify --config configs/v49_joint_absolute_relative_training.yaml

preflight-v50:
	PYTHONPATH=src python3 -m tlm joint-absolute-relative-evaluation-preflight --config configs/v50_joint_absolute_relative_evaluation.yaml

prepare-v50:
	PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src python3 -m tlm joint-absolute-relative-evaluation-prepare --config configs/v50_joint_absolute_relative_evaluation.yaml

run-v50:
	PYTHONPATH=src python3 -m tlm joint-absolute-relative-evaluation --config configs/v50_joint_absolute_relative_evaluation.yaml

verify-v50:
	PYTHONPATH=src python3 -m tlm joint-absolute-relative-evaluation-verify --config configs/v50_joint_absolute_relative_evaluation.yaml

preflight-v54:
	PYTHONPATH=src python3 -m tlm v50-economic-failure-autopsy-preflight --config configs/v54_v50_economic_failure_autopsy.yaml

run-v54:
	PYTHONPATH=src python3 -m tlm v50-economic-failure-autopsy --config configs/v54_v50_economic_failure_autopsy.yaml

run-v55:
	PYTHONPATH=src python3 -m tlm state-conditioned-multi-horizon-spec --config configs/v55_state_conditioned_multi_horizon_spec.yaml

run-v56:
	PYTHONPATH=src python3 -m tlm state-conditioned-multi-horizon-harness --config configs/v56_state_conditioned_multi_horizon_harness.yaml

run-v57:
	PYTHONPATH=src python3 -m tlm state-conditioned-multi-horizon-dataset --config configs/v57_state_conditioned_multi_horizon_dataset.yaml

preflight-v58:
	PYTHONPATH=src python3 -m tlm state-conditioned-multi-horizon-training-preflight --config configs/v58_state_conditioned_multi_horizon_training.yaml

smoke-v58:
	PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src python3 -m tlm state-conditioned-multi-horizon-training-smoke --config configs/v58_state_conditioned_multi_horizon_training.yaml

run-v58:
	PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src python3 -m tlm state-conditioned-multi-horizon-training --config configs/v58_state_conditioned_multi_horizon_training.yaml

verify-v58:
	PYTHONPATH=src python3 -m tlm state-conditioned-multi-horizon-training-verify --config configs/v58_state_conditioned_multi_horizon_training.yaml

replay-v58:
	PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src python3 -m tlm state-conditioned-multi-horizon-training-replay --config configs/v58_state_conditioned_multi_horizon_training.yaml

run-v60:
	PYTHONPATH=src python3 -m tlm decoupled-rank-state-spec --config configs/v60_decoupled_rank_state_spec.yaml

run-v61:
	PYTHONPATH=src python3 -m tlm decoupled-rank-state-harness --config configs/v61_decoupled_rank_state_harness.yaml

run-v62:
	PYTHONPATH=src python3 -m tlm decoupled-rank-state-dataset --config configs/v62_decoupled_rank_state_dataset.yaml

preflight-v63:
	PYTHONPATH=src python3 -m tlm decoupled-rank-state-training-preflight --config configs/v63_decoupled_rank_state_training.yaml

smoke-v63:
	PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src python3 -m tlm decoupled-rank-state-training-smoke --config configs/v63_decoupled_rank_state_training.yaml

run-v63:
	PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src python3 -m tlm decoupled-rank-state-training --config configs/v63_decoupled_rank_state_training.yaml

verify-v63:
	PYTHONPATH=src python3 -m tlm decoupled-rank-state-training-verify --config configs/v63_decoupled_rank_state_training.yaml

replay-v63:
	PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src python3 -m tlm decoupled-rank-state-training-replay --config configs/v63_decoupled_rank_state_training.yaml

audit:
	PYTHONPATH=src python3 -m tlm audit --config configs/mvp.yaml

audit-v2:
	PYTHONPATH=src python3 -m tlm audit --config configs/v2_open_to_open.yaml

audit-v3:
	PYTHONPATH=src python3 -m tlm audit --config configs/v3_always_long_top1.yaml

audit-v4:
	PYTHONPATH=src python3 -m tlm audit --config configs/v4_cross_sectional_ranking.yaml

audit-v5:
	PYTHONPATH=src python3 -m tlm audit --config configs/v5_consensus.yaml

clean:
	rm -rf artifacts/smoke artifacts/mvp .pytest_cache
