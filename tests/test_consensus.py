import numpy as np

from tlm.consensus import _passes, _policy_scores


def test_policy_scores_encode_cash_and_selected_asset():
    scores = _policy_scores(
        np.array([2, 0, 1]), np.array([True, False, True]), n_assets=3
    )
    assert scores.tolist() == [
        [-1.0, -1.0, 1.0],
        [-1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
    ]


def test_acceptance_requires_return_sharpe_and_drawdown_improvement():
    candidate = {"total_return": 2.0, "sharpe": 1.2, "max_drawdown": -0.4}
    baseline = {"total_return": 1.5, "sharpe": 0.8, "max_drawdown": -0.6}
    assert _passes(candidate, [baseline]) is True
    candidate["max_drawdown"] = -0.7
    assert _passes(candidate, [baseline]) is False
