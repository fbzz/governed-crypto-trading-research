import numpy as np

from tlm.monte_carlo import circular_block_indices, paired_block_bootstrap


def test_circular_blocks_preserve_local_order_with_wraparound():
    indexes = circular_block_indices(
        n_observations=10,
        block_length=4,
        n_paths=5,
        rng=np.random.default_rng(3),
    )
    assert indexes.shape == (5, 10)
    for row in indexes:
        for start in range(0, 8, 4):
            block = row[start : start + 4]
            assert np.all(np.diff(block) % 10 == 1)


def test_paired_bootstrap_is_deterministic_and_summarizes_risk():
    returns = {
        "candidate": np.array([0.01, -0.005, 0.008, 0.002] * 30),
        "baseline": np.array([0.004, -0.004, 0.003, 0.001] * 30),
    }
    first = paired_block_bootstrap(
        returns, "candidate", ["baseline"], block_length=4, n_paths=100, seed=9
    )
    second = paired_block_bootstrap(
        returns, "candidate", ["baseline"], block_length=4, n_paths=100, seed=9
    )
    assert first == second
    assert first["candidate_probability_of_loss"] == 0.0
    assert first["comparisons"]["baseline"]["probability_higher_total_return"] == 1.0
    assert first["comparisons"]["baseline"]["paired_total_return_delta"]["p05"] > 0
