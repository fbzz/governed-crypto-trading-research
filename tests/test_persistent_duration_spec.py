from __future__ import annotations

from pathlib import Path

from tlm.config import load_config
from tlm.persistent_duration_spec import persistent_duration_parameter_count
from tlm.research_workflow import validate_research_state


ROOT = Path(__file__).resolve().parents[1]


def _spec() -> dict:
    return load_config(ROOT / "configs/v74_persistent_duration_spec.yaml")[
        "persistent_duration_spec"
    ]


def test_v74_exact_capacity_and_single_variant_are_frozen() -> None:
    spec = _spec()
    count = persistent_duration_parameter_count(spec["architecture"])
    assert count == 1_083_155
    assert count == spec["architecture"]["expected_parameter_count"]
    assert count == spec["capacity_contract"]["expected_total_parameter_count"]
    assert count <= spec["capacity_contract"]["parameter_ceiling"]
    assert spec["capacity_contract"]["variant_count"] == 1
    assert spec["capacity_contract"]["size_sweep_allowed"] is False


def test_v74_label_objective_and_policy_contracts_are_frozen() -> None:
    spec = _spec()
    labels = spec["data_and_label_contract"]
    assert labels["maximum_label_maturity_days"] == 8
    assert labels["duration_target"].startswith("earliest_argmax_day")
    assert labels["duration_right_censor_rule"] == (
        "censored_when_earliest_argmax_is_day_7"
    )
    assert spec["objective"]["weights"] == {
        "return_nll": 1.0,
        "pairwise_ranking": 0.25,
        "duration_nll": 0.5,
    }
    policy = spec["policy"]
    assert policy["horizon_weights"] == [0.2, 0.3, 0.5]
    assert policy["transition_turnover"] == {
        "hold": 0.0,
        "enter": 1.0,
        "exit": 1.0,
        "switch": 2.0,
    }
    assert policy["threshold_sweep_allowed"] is False


def test_v74_training_and_financial_decision_are_preregistered() -> None:
    spec = _spec()
    training = spec["training_contract"]
    assert len(training["folds"]) * len(training["seeds"]) == 9
    assert training["expected_job_count"] == 9
    assert training["device"] == "mps"
    assert training["mps_fallback_allowed"] is False
    assert training["fold_selection_allowed"] is False
    assert training["seed_selection_allowed"] is False
    assert training["hyperparameter_search_allowed"] is False

    gates = spec["financial_evaluation_contract"]["mandatory_gates"]
    assert gates["aggregate_net_total_return_positive_at_cost_bps"] == [10, 20, 30]
    assert gates["each_fold_net_total_return_positive_at_cost_bps"] == [10]
    assert gates["bootstrap_block_days"] == [7, 21, 63]
    assert gates["aggregate_rescue_for_failed_fold"] is False


def test_v74_registration_remains_metadata_only_and_targets_sealed() -> None:
    spec = _spec()
    constraints = spec["constraints"]
    assert constraints["metadata_only"] is True
    assert not any(value for key, value in constraints.items() if key != "metadata_only")
    assert spec["target_contract"]["status"] == "sealed"
    assert spec["target_contract"]["target_data_allowed"] is False
    assert spec["authorized_next_action"] == (
        "authorize_v75_synthetic_persistent_duration_harness_only"
    )


def test_current_state_records_v82_r0_and_authorizes_only_v84_prepare() -> None:
    status = validate_research_state(ROOT, "research/current.yaml")
    assert status["passed"] is True
    assert status["active_family_id"] == "tlm_low_turnover_cross_sectional_rank_v1"
    assert status["active_family_status"] == (
        "retrospective_non_target_economic_evaluation_exact_unseal_authorized"
    )
    assert status["authorized_phase"] == "v85"
    assert status["authorized_next_action"] == (
        "execute_v85_exactly_one_registered_non_target_outcome_unseal_and_complete_evaluation"
    )
    assert status["target_asset_status"] == "sealed"
    assert status["deployable_strategy"] is False
