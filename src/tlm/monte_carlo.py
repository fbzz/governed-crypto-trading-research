from __future__ import annotations

import math

import numpy as np


def circular_block_indices(
    n_observations: int,
    block_length: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_observations < 2 or block_length < 1 or n_paths < 1:
        raise ValueError("Invalid block-bootstrap dimensions")
    blocks = math.ceil(n_observations / block_length)
    starts = rng.integers(0, n_observations, size=(n_paths, blocks))
    offsets = np.arange(block_length)
    indexes = (starts[:, :, None] + offsets[None, None, :]) % n_observations
    return indexes.reshape(n_paths, -1)[:, :n_observations]


def _path_metrics(sampled_returns: np.ndarray) -> dict[str, np.ndarray]:
    equity = np.cumprod(1.0 + sampled_returns, axis=1)
    final_equity = equity[:, -1]
    years = sampled_returns.shape[1] / 365.0
    means = sampled_returns.mean(axis=1)
    standard_deviations = sampled_returns.std(axis=1, ddof=1)
    sharpe = np.divide(
        np.sqrt(365.0) * means,
        standard_deviations,
        out=np.zeros_like(means),
        where=standard_deviations > 0,
    )
    running_peak = np.maximum.accumulate(np.maximum(equity, 1.0), axis=1)
    max_drawdown = np.min(equity / running_peak - 1.0, axis=1)
    cagr = np.where(final_equity > 0, final_equity ** (1.0 / years) - 1.0, -1.0)
    return {
        "total_return": final_equity - 1.0,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def _distribution_summary(values: np.ndarray) -> dict[str, float]:
    quantiles = np.quantile(values, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "mean": float(values.mean()),
        "p01": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "p75": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "p99": float(quantiles[6]),
    }


def paired_block_bootstrap(
    returns_by_strategy: dict[str, np.ndarray],
    candidate_name: str,
    baseline_names: list[str],
    block_length: int,
    n_paths: int,
    seed: int,
    batch_size: int = 250,
) -> dict[str, object]:
    if candidate_name not in returns_by_strategy:
        raise ValueError(f"Missing candidate strategy: {candidate_name}")
    lengths = {len(np.asarray(values)) for values in returns_by_strategy.values()}
    if len(lengths) != 1:
        raise ValueError("All strategies must share the same number of observations")
    n_observations = lengths.pop()
    rng = np.random.default_rng(seed)
    stores = {
        strategy: {
            metric: np.empty(n_paths, dtype=np.float64)
            for metric in ("total_return", "cagr", "sharpe", "max_drawdown")
        }
        for strategy in returns_by_strategy
    }

    cursor = 0
    while cursor < n_paths:
        current_batch = min(batch_size, n_paths - cursor)
        indexes = circular_block_indices(
            n_observations, block_length, current_batch, rng
        )
        for strategy, returns in returns_by_strategy.items():
            sampled = np.asarray(returns, dtype=np.float64)[indexes]
            metrics = _path_metrics(sampled)
            for metric, values in metrics.items():
                stores[strategy][metric][cursor : cursor + current_batch] = values
        cursor += current_batch

    distributions = {
        strategy: {
            metric: _distribution_summary(values)
            for metric, values in metrics.items()
        }
        for strategy, metrics in stores.items()
    }
    comparisons: dict[str, dict[str, object]] = {}
    candidate = stores[candidate_name]
    for baseline_name in baseline_names:
        baseline = stores[baseline_name]
        wins_return = candidate["total_return"] > baseline["total_return"]
        wins_sharpe = candidate["sharpe"] > baseline["sharpe"]
        wins_drawdown = candidate["max_drawdown"] > baseline["max_drawdown"]
        comparisons[baseline_name] = {
            "probability_higher_total_return": float(wins_return.mean()),
            "probability_higher_sharpe": float(wins_sharpe.mean()),
            "probability_better_max_drawdown": float(wins_drawdown.mean()),
            "probability_wins_all_three": float(
                (wins_return & wins_sharpe & wins_drawdown).mean()
            ),
            "paired_total_return_delta": _distribution_summary(
                candidate["total_return"] - baseline["total_return"]
            ),
        }
    return {
        "method": "paired_circular_block_bootstrap",
        "block_length": block_length,
        "paths": n_paths,
        "seed": seed,
        "observations_per_path": n_observations,
        "distributions": distributions,
        "comparisons": comparisons,
        "candidate_probability_of_loss": float(
            (stores[candidate_name]["total_return"] < 0).mean()
        ),
    }
