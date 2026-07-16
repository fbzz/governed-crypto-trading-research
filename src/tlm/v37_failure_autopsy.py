from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .non_target_pretraining import (
    TARGET_SYMBOLS,
    _canonical_sha256,
    _sha256_file,
    _write_json,
)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_correlation(
    left: pd.Series,
    right: pd.Series,
    method: str,
) -> float | None:
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(frame) < 3 or frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return None
    if method == "spearman":
        value = frame["left"].rank(method="average").corr(
            frame["right"].rank(method="average")
        )
    elif method == "pearson":
        value = frame["left"].corr(frame["right"])
    else:
        raise ValueError(f"Unsupported correlation: {method}")
    return float(value) if pd.notna(value) else None


def _mean_finite(values: pd.Series | list[float | None]) -> float | None:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(numeric.mean()) if len(numeric) else None


def _compound(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.prod(1.0 + array) - 1.0)


def build_fold_date_diagnostics(
    predictions: pd.DataFrame,
    daily_returns: pd.DataFrame,
    cost_bps: int,
) -> pd.DataFrame:
    base_daily = daily_returns.loc[
        (daily_returns["cost_bps"] == cost_bps)
        & daily_returns["scope"].str.startswith("fold_")
    ].copy()
    lookup = {
        (int(row.scope.split("_")[1]), pd.Timestamp(row.date), row.strategy): row
        for row in base_daily.itertuples(index=False)
    }
    records: list[dict[str, object]] = []
    for (fold, date), frame in predictions.groupby(["fold", "date"], sort=True):
        ordered = frame.sort_values(
            ["calibrated_q50", "symbol"], ascending=[False, True]
        )
        top = ordered.iloc[0]
        actual_order = frame.sort_values(
            ["observed_log_return", "symbol"], ascending=[False, True]
        ).reset_index(drop=True)
        actual_rank_by_symbol = {
            symbol: rank + 1
            for rank, symbol in enumerate(actual_order["symbol"].tolist())
        }
        actual_rank = actual_rank_by_symbol[str(top["symbol"])]
        active_rows = frame.loc[frame["candidate_weight"] > 0.5]
        if len(active_rows) > 1:
            raise RuntimeError("V37 autopsy found multiple candidate assets in a fold-date")
        active = len(active_rows) == 1
        active_symbol = str(active_rows.iloc[0]["symbol"]) if active else None
        if active and active_symbol != str(top["symbol"]):
            raise RuntimeError("Frozen candidate weight does not match q50 top-1")
        selected_simple_return = float(np.expm1(top["observed_log_return"]))
        universe_simple_return = float(np.expm1(frame["observed_log_return"]).mean())
        candidate = lookup[(int(fold), pd.Timestamp(date), "candidate")]
        dual = lookup[(int(fold), pd.Timestamp(date), "dual_momentum_30")]
        equal = lookup[(int(fold), pd.Timestamp(date), "equal_weight_buy_hold")]
        q50_values = ordered["calibrated_q50"].to_numpy(dtype=np.float64)
        records.append({
            "date": pd.Timestamp(date),
            "fold": int(fold),
            "calendar_month": pd.Timestamp(date).strftime("%Y-%m"),
            "available_assets": int(len(frame)),
            "context_count_min": int(frame["context_count"].min()),
            "context_count_max": int(frame["context_count"].max()),
            "predicted_top1_symbol": str(top["symbol"]),
            "actual_top1_symbol": str(actual_order.iloc[0]["symbol"]),
            "predicted_top1_actual_rank": int(actual_rank),
            "top1_hit": bool(actual_rank == 1),
            "top3_hit": bool(actual_rank <= 3),
            "random_top1_expectation": float(1.0 / len(frame)),
            "random_top3_expectation": float(min(3, len(frame)) / len(frame)),
            "spearman_q10": _finite_correlation(
                frame["calibrated_q10"], frame["observed_log_return"], "spearman"
            ),
            "spearman_q50": _finite_correlation(
                frame["calibrated_q50"], frame["observed_log_return"], "spearman"
            ),
            "spearman_q90": _finite_correlation(
                frame["calibrated_q90"], frame["observed_log_return"], "spearman"
            ),
            "top1_q10": float(top["calibrated_q10"]),
            "top1_q50": float(top["calibrated_q50"]),
            "top1_q90": float(top["calibrated_q90"]),
            "top1_q50_margin": float(q50_values[0] - q50_values[1]),
            "top1_observed_simple_return": selected_simple_return,
            "fold_mean_observed_simple_return": universe_simple_return,
            "top1_excess_return_vs_fold_mean": float(
                selected_simple_return - universe_simple_return
            ),
            "candidate_active": bool(active),
            "candidate_symbol": active_symbol,
            "median_momentum_30": float(frame["momentum_30"].median()),
            "positive_momentum_asset_count": int((frame["momentum_30"] > 0).sum()),
            "signal_time_regime": (
                "risk_on" if frame["momentum_30"].median() > 0 else "risk_off"
            ),
            "candidate_gross_return": float(candidate.gross_return),
            "candidate_net_return": float(candidate.net_return),
            "candidate_turnover": float(candidate.turnover),
            "candidate_cost": float(candidate.cost),
            "dual_momentum_net_return": float(dual.net_return),
            "equal_weight_net_return": float(equal.net_return),
        })
    return pd.DataFrame(records).sort_values(["date", "fold"]).reset_index(drop=True)


def audit_frozen_ledger(
    daily_returns: pd.DataFrame,
    fold_dates: pd.DataFrame,
    cost_bps: int,
) -> dict[str, bool]:
    base_daily = daily_returns.loc[daily_returns["cost_bps"] == cost_bps].copy()
    fold_daily = base_daily.loc[base_daily["scope"].str.startswith("fold_")].copy()
    aggregate_daily = base_daily.loc[
        base_daily["scope"] == "aggregate_equal_fold_capital"
    ].copy()
    strategy_names = {
        "candidate",
        "dual_momentum_30",
        "equal_weight_buy_hold",
    }
    expected_candidate_gross = np.where(
        fold_dates["candidate_active"],
        fold_dates["top1_observed_simple_return"],
        0.0,
    )
    fold_components = (
        fold_daily.groupby(["date", "strategy"], as_index=False)[
            ["gross_return", "net_return", "turnover", "cost"]
        ]
        .mean()
        .sort_values(["date", "strategy"])
        .reset_index(drop=True)
    )
    aggregate_components = aggregate_daily[
        ["date", "strategy", "gross_return", "net_return", "turnover", "cost"]
    ].sort_values(["date", "strategy"]).reset_index(drop=True)
    aggregate_reconciliation = fold_components.merge(
        aggregate_components,
        on=["date", "strategy"],
        how="outer",
        suffixes=("_fold_mean", "_aggregate"),
        indicator=True,
        validate="one_to_one",
    )

    def columns_close(left: str, right: str) -> bool:
        return bool(np.allclose(
            aggregate_reconciliation[left].to_numpy(dtype=np.float64),
            aggregate_reconciliation[right].to_numpy(dtype=np.float64),
            rtol=1e-12,
            atol=1e-12,
        ))

    return {
        "daily_row_keys_are_unique": bool(
            ~base_daily.duplicated(["date", "cost_bps", "scope", "strategy"]).any()
        ),
        "every_fold_date_has_all_registered_strategies": bool(
            len(fold_daily) == len(fold_dates) * len(strategy_names)
            and fold_daily.groupby(["date", "scope"])["strategy"]
            .apply(lambda values: set(values) == strategy_names)
            .all()
        ),
        "candidate_gross_matches_frozen_selected_return": bool(np.allclose(
            fold_dates["candidate_gross_return"].to_numpy(dtype=np.float64),
            expected_candidate_gross,
            rtol=1e-12,
            atol=1e-12,
        )),
        "daily_net_equals_gross_minus_cost": bool(np.allclose(
            base_daily["net_return"].to_numpy(dtype=np.float64),
            (
                base_daily["gross_return"] - base_daily["cost"]
            ).to_numpy(dtype=np.float64),
            rtol=1e-12,
            atol=1e-12,
        )),
        "daily_cost_equals_turnover_times_rate": bool(np.allclose(
            base_daily["cost"].to_numpy(dtype=np.float64),
            (
                base_daily["turnover"] * (cost_bps / 10_000.0)
            ).to_numpy(dtype=np.float64),
            rtol=1e-12,
            atol=1e-12,
        )),
        "aggregate_rows_match_fold_date_strategy_grid": bool(
            len(aggregate_reconciliation)
            == fold_dates["date"].nunique() * len(strategy_names)
            and aggregate_reconciliation["_merge"].eq("both").all()
        ),
        "aggregate_gross_is_equal_fold_mean": columns_close(
            "gross_return_fold_mean", "gross_return_aggregate"
        ),
        "aggregate_net_is_equal_fold_mean": columns_close(
            "net_return_fold_mean", "net_return_aggregate"
        ),
        "aggregate_turnover_is_equal_fold_mean": columns_close(
            "turnover_fold_mean", "turnover_aggregate"
        ),
        "aggregate_cost_is_equal_fold_mean": columns_close(
            "cost_fold_mean", "cost_aggregate"
        ),
    }


def summarize_ranking(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "fold_date_count": int(len(frame)),
        "top1_hits": int(frame["top1_hit"].sum()),
        "top1_hit_rate": float(frame["top1_hit"].mean()),
        "random_top1_expectation": float(frame["random_top1_expectation"].mean()),
        "top3_hit_rate": float(frame["top3_hit"].mean()),
        "random_top3_expectation": float(frame["random_top3_expectation"].mean()),
        "mean_actual_rank_of_predicted_top1": float(
            frame["predicted_top1_actual_rank"].mean()
        ),
        "mean_spearman_q10": _mean_finite(frame["spearman_q10"]),
        "mean_spearman_q50": _mean_finite(frame["spearman_q50"]),
        "median_spearman_q50": float(frame["spearman_q50"].dropna().median()),
        "positive_spearman_q50_rate": float(
            (frame["spearman_q50"].dropna() > 0).mean()
        ),
        "mean_spearman_q90": _mean_finite(frame["spearman_q90"]),
        "mean_top1_observed_simple_return": float(
            frame["top1_observed_simple_return"].mean()
        ),
        "mean_fold_universe_simple_return": float(
            frame["fold_mean_observed_simple_return"].mean()
        ),
        "mean_top1_excess_return_vs_fold_mean": float(
            frame["top1_excess_return_vs_fold_mean"].mean()
        ),
    }


def extract_holding_episodes(
    fold_dates: pd.DataFrame,
    round_trip_cost_rate: float,
) -> pd.DataFrame:
    episodes: list[dict[str, object]] = []
    for fold, frame in fold_dates.groupby("fold", sort=True):
        frame = frame.sort_values("date")
        current: list[pd.Series] = []
        prior_date: pd.Timestamp | None = None
        prior_symbol: str | None = None

        def close_episode() -> None:
            nonlocal current
            if not current:
                return
            gross = np.asarray(
                [row["top1_observed_simple_return"] for row in current],
                dtype=np.float64,
            )
            allocated_cost = 2.0 * round_trip_cost_rate
            gross_additive = float(gross.sum())
            net_additive = gross_additive - allocated_cost
            episodes.append({
                "episode": len(episodes) + 1,
                "fold": int(fold),
                "symbol": str(current[0]["candidate_symbol"]),
                "start_date": pd.Timestamp(current[0]["date"]),
                "end_date": pd.Timestamp(current[-1]["date"]),
                "duration_days": len(current),
                "gross_compounded_return": _compound(gross),
                "gross_additive_return": gross_additive,
                "allocated_round_trip_cost": allocated_cost,
                "net_additive_return": net_additive,
                "winner_after_cost": bool(net_additive > 0),
                "signal_time_regime": str(current[0]["signal_time_regime"]),
                "positive_momentum_asset_count": int(
                    current[0]["positive_momentum_asset_count"]
                ),
                "top1_q10": float(current[0]["top1_q10"]),
                "top1_q50": float(current[0]["top1_q50"]),
                "top1_q50_margin": float(current[0]["top1_q50_margin"]),
            })
            current = []

        for _, row in frame.iterrows():
            date = pd.Timestamp(row["date"])
            symbol = row["candidate_symbol"] if row["candidate_active"] else None
            continues = (
                symbol is not None
                and prior_symbol == symbol
                and prior_date is not None
                and date - prior_date == pd.Timedelta(days=1)
            )
            if symbol is None:
                close_episode()
            elif continues:
                current.append(row)
            else:
                close_episode()
                current = [row]
            prior_date = date
            prior_symbol = str(symbol) if symbol is not None else None
        close_episode()
    return pd.DataFrame(episodes)


def loss_concentration(
    values: pd.Series,
    top_n: list[int],
) -> dict[str, object]:
    losses = -np.sort(np.minimum(np.asarray(values, dtype=np.float64), 0.0))
    losses = np.sort(losses[losses > 0])[::-1]
    total = float(losses.sum())
    return {
        "losing_observation_count": int(len(losses)),
        "total_loss_magnitude": total,
        "shares": {
            str(n): float(losses[:n].sum() / total) if total > 0 else 0.0
            for n in top_n
        },
    }


def _bin_summary(
    frame: pd.DataFrame,
    column: str,
    boundaries: list[float],
) -> list[dict[str, object]]:
    edges = [-np.inf, *boundaries, np.inf]
    labels = []
    for index in range(len(edges) - 1):
        left, right = edges[index], edges[index + 1]
        if np.isneginf(left):
            labels.append(f"<{right:g}")
        elif np.isposinf(right):
            labels.append(f">={left:g}")
        else:
            labels.append(f"[{left:g},{right:g})")
    categories = pd.cut(
        frame[column], bins=edges, labels=labels, right=False, include_lowest=True
    )
    rows = []
    for label in labels:
        subset = frame.loc[categories == label]
        rows.append({
            "bin": label,
            "count": int(len(subset)),
            "candidate_active_count": int(subset["candidate_active"].sum()),
            "mean_top1_observed_simple_return": (
                float(subset["top1_observed_simple_return"].mean())
                if len(subset)
                else None
            ),
            "mean_top1_excess_return_vs_fold_mean": (
                float(subset["top1_excess_return_vs_fold_mean"].mean())
                if len(subset)
                else None
            ),
            "top1_hit_rate": float(subset["top1_hit"].mean()) if len(subset) else None,
        })
    return rows


def _monthly_strategy_summary(
    daily_returns: pd.DataFrame,
    cost_bps: int,
) -> list[dict[str, object]]:
    frame = daily_returns.loc[
        (daily_returns["cost_bps"] == cost_bps)
        & (daily_returns["scope"] == "aggregate_equal_fold_capital")
    ].copy()
    frame["calendar_month"] = pd.to_datetime(frame["date"], utc=True).dt.strftime(
        "%Y-%m"
    )
    rows = []
    for (month, strategy), group in frame.groupby(
        ["calendar_month", "strategy"], sort=True
    ):
        rows.append({
            "calendar_month": month,
            "strategy": strategy,
            "observations": int(len(group)),
            "gross_compounded_return": _compound(group["gross_return"]),
            "net_compounded_return": _compound(group["net_return"]),
            "cost_sum": float(group["cost"].sum()),
        })
    return rows


def summarize_signal_time_regimes(
    fold_dates: pd.DataFrame,
    *,
    active_only: bool,
    fold_capital_weight: float,
) -> list[dict[str, object]]:
    source = (
        fold_dates.loc[fold_dates["candidate_active"]]
        if active_only
        else fold_dates
    )
    rows = []
    for regime, frame in source.groupby("signal_time_regime", sort=True):
        rows.append({
            "signal_time_regime": regime,
            "fold_date_count": int(len(frame)),
            "candidate_active_fold_days": int(frame["candidate_active"].sum()),
            "candidate_active_rate": float(frame["candidate_active"].mean()),
            "candidate_net_mean": float(frame["candidate_net_return"].mean()),
            "candidate_net_additive_fold_capital_contribution": float(
                frame["candidate_net_return"].sum() * fold_capital_weight
            ),
            "dual_momentum_net_mean": float(
                frame["dual_momentum_net_return"].mean()
            ),
            "equal_weight_net_mean": float(frame["equal_weight_net_return"].mean()),
            "mean_selected_return": float(
                frame["top1_observed_simple_return"].mean()
            ),
            "mean_universe_return": float(
                frame["fold_mean_observed_simple_return"].mean()
            ),
            "mean_excess_return": float(
                frame["top1_excess_return_vs_fold_mean"].mean()
            ),
        })
    return rows


def _json_episode_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    value = frame.copy()
    for column in ("start_date", "end_date"):
        if column in value:
            value[column] = pd.to_datetime(value[column], utc=True).dt.strftime(
                "%Y-%m-%d"
            )
    return value.to_dict(orient="records")


def _report(result: dict[str, object]) -> str:
    ranking = result["ranking"]["all_fold_dates"]
    active = result["ranking"]["candidate_active_fold_dates"]
    trades = result["trades"]["summary"]
    concentration = result["trades"]["loss_concentration"][
        "aggregate_calendar_day_losses"
    ]
    base = result["baseline_10bps"]
    failure = result["failure_attribution"]
    return "\n".join([
        "# TLM v37 Failure Autopsy",
        "",
        "## Decision",
        "",
        "**V37 RETIREMENT CONFIRMED. A NEW EX-ANTE FAMILY IS REQUIRED.**",
        "",
        "This is retrospective diagnosis over frozen v37 artifacts. It does not alter the failed gate and does not authorize a new policy, target evaluation, or threshold test.",
        "",
        "## Economic failure",
        "",
        f"At 10 bps the candidate returned **{base['candidate']['total_return']:.2%}** net and **{result['trades']['aggregate_gross_compounded_return']:.2%}** gross. Its aggregate turnover charge summed to **{result['trades']['aggregate_cost_sum']:.2%}**. Costs were a secondary drag; the signal was already negative before costs.",
        f"The policy was active on **{trades['active_fold_days']} / {trades['fold_days']} fold-days ({trades['active_rate']:.2%})**. All **{trades['episode_count']}** holding episodes lasted one day; only **{trades['winning_episode_count']}** were positive after round-trip cost.",
        f"The worst three aggregate calendar days produced **{concentration['shares']['3']:.2%}** of total losing-day magnitude; the failure was concentrated, but not explained by a single observation.",
        "",
        "## Ranking failure",
        "",
        f"Q50 top-1 hit rate was **{ranking['top1_hit_rate']:.2%}**, below the **{ranking['random_top1_expectation']:.2%}** random expectation. Mean daily cross-sectional q50 Spearman IC was **{ranking['mean_spearman_q50']:.4f}** and predicted top-1 excess return versus its fold mean was **{ranking['mean_top1_excess_return_vs_fold_mean']:.3%}**.",
        f"On the {active['fold_date_count']} active fold-days, top-1 hit rate was **{active['top1_hit_rate']:.2%}** and the selected asset returned **{active['mean_top1_observed_simple_return']:.3%}** on average.",
        "",
        "## Descriptive residual structure",
        "",
        f"The q10 head had mean cross-sectional IC **{ranking['mean_spearman_q10']:.4f}**, q10/q90 marginal coverage was **{result['ranking']['direction_and_calibration']['coverage_q10']:.2%} / {result['ranking']['direction_and_calibration']['coverage_q90']:.2%}**, and predicted volatility had Spearman **{result['ranking']['continuous_diagnostics']['asset_level_spearman_predicted_log_vol_vs_absolute_return']:.4f}** with absolute realized return. The engineering audit and context-count checks also passed.",
        "These are retrospective associations on a consumed holdout, not proof of learned trading alpha. Testing q10 ranking or a new confidence rule on this window is forbidden.",
        "Per-seed and per-triplet disagreement cannot be recovered from the persisted aggregates without forbidden re-inference.",
        "",
        "## Failure attribution",
        "",
        f"- Ranking alpha failure: **{str(failure['q50_ranking_failure']).lower()}**",
        f"- Active-day alpha failure: **{str(failure['active_day_alpha_failure']).lower()}**",
        f"- Costs were the primary cause: **{str(failure['cost_is_primary_cause']).lower()}**",
        f"- Lower drawdown coincided with cash exposure and negative active-day alpha: **{str(failure['lower_drawdown_coincided_with_cash_not_positive_active_alpha']).lower()}**",
        "",
        "## Next legal research action",
        "",
        "Design a new family before any new clean evaluation. Its primary objective should be cross-sectional ranking/excess return rather than marginal q50 forecasting. The v37 window may be used only as development evidence, and BTC/ETH/SOL remain sealed.",
        "",
    ])


def run_v37_failure_autopsy(config: dict) -> dict[str, object]:
    autopsy = config["failure_autopsy"]
    root = Path(autopsy["project_root"]).resolve()
    paths = {name: root / path for name, path in autopsy["inputs"].items()}
    input_checks = {
        name: path.is_file()
        and _sha256_file(path) == autopsy["expected_input_sha256"][name]
        for name, path in paths.items()
    }
    if not all(input_checks.values()):
        raise RuntimeError(f"V37 autopsy input drift: {input_checks}")
    result_v37 = _load_json(paths["result"])
    metrics_v37 = _load_json(paths["metrics"])
    gate_v37 = _load_json(paths["gate_result"])
    receipt_v37 = _load_json(paths["evaluation_receipt"])
    predictions = pd.read_parquet(paths["predictions"])
    daily_returns = pd.read_parquet(paths["daily_returns"])
    predictions["date"] = pd.to_datetime(predictions["date"], utc=True)
    daily_returns["date"] = pd.to_datetime(daily_returns["date"], utc=True)

    cost_bps = int(autopsy["accounting"]["descriptive_cost_bps"])
    fold_dates = build_fold_date_diagnostics(predictions, daily_returns, cost_bps)
    ledger_checks = audit_frozen_ledger(daily_returns, fold_dates, cost_bps)
    episodes = extract_holding_episodes(fold_dates, cost_bps / 10_000.0)
    ranking_all = summarize_ranking(fold_dates)
    ranking_active = summarize_ranking(fold_dates.loc[fold_dates["candidate_active"]])
    ranking_by_fold = {
        str(fold): summarize_ranking(frame)
        for fold, frame in fold_dates.groupby("fold", sort=True)
    }
    ranking_by_fold_month = [
        {
            "fold": int(fold),
            "calendar_month": month,
            **summarize_ranking(frame),
        }
        for (fold, month), frame in fold_dates.groupby(
            ["fold", "calendar_month"], sort=True
        )
    ]

    observed_positive = predictions["observed_log_return"] > 0
    predicted_positive = predictions["calibrated_q50"] > 0
    true_positive = predicted_positive & observed_positive
    direction = {
        "accuracy": float((predicted_positive == observed_positive).mean()),
        "always_nonpositive_accuracy": float((~observed_positive).mean()),
        "predicted_positive_rate": float(predicted_positive.mean()),
        "observed_positive_rate": float(observed_positive.mean()),
        "positive_precision": float(true_positive.sum() / predicted_positive.sum()),
        "positive_recall": float(true_positive.sum() / observed_positive.sum()),
        "mean_q50_bias_log_return": float(
            (predictions["calibrated_q50"] - predictions["observed_log_return"]).mean()
        ),
        "coverage_q10": float(
            (predictions["observed_log_return"] <= predictions["calibrated_q10"]).mean()
        ),
        "coverage_q50": float(
            (predictions["observed_log_return"] <= predictions["calibrated_q50"]).mean()
        ),
        "coverage_q90": float(
            (predictions["observed_log_return"] <= predictions["calibrated_q90"]).mean()
        ),
    }
    continuous = {
        "asset_level_spearman_q50_vs_return": _finite_correlation(
            predictions["calibrated_q50"], predictions["observed_log_return"], "spearman"
        ),
        "asset_level_spearman_q10_vs_return": _finite_correlation(
            predictions["calibrated_q10"], predictions["observed_log_return"], "spearman"
        ),
        "asset_level_spearman_predicted_log_vol_vs_absolute_return": _finite_correlation(
            predictions["calibrated_log_volatility"],
            predictions["observed_log_return"].abs(),
            "spearman",
        ),
        "top1_spearman_q50_vs_return": _finite_correlation(
            fold_dates["top1_q50"], fold_dates["top1_observed_simple_return"], "spearman"
        ),
        "top1_spearman_q10_vs_return": _finite_correlation(
            fold_dates["top1_q10"], fold_dates["top1_observed_simple_return"], "spearman"
        ),
        "top1_margin_spearman_vs_return": _finite_correlation(
            fold_dates["top1_q50_margin"],
            fold_dates["top1_observed_simple_return"],
            "spearman",
        ),
        "top1_margin_spearman_vs_excess_return": _finite_correlation(
            fold_dates["top1_q50_margin"],
            fold_dates["top1_excess_return_vs_fold_mean"],
            "spearman",
        ),
    }

    episode_summary = {
        "fold_days": int(len(fold_dates)),
        "active_fold_days": int(fold_dates["candidate_active"].sum()),
        "active_rate": float(fold_dates["candidate_active"].mean()),
        "calendar_days_with_any_active_fold": int(
            fold_dates.groupby("date")["candidate_active"].any().sum()
        ),
        "calendar_days_all_cash": int(
            (~fold_dates.groupby("date")["candidate_active"].any()).sum()
        ),
        "episode_count": int(len(episodes)),
        "winning_episode_count": int(episodes["winner_after_cost"].sum()),
        "episode_win_rate": float(episodes["winner_after_cost"].mean()),
        "all_episodes_one_day": bool((episodes["duration_days"] == 1).all()),
        "mean_episode_net_additive_return": float(
            episodes["net_additive_return"].mean()
        ),
        "median_episode_net_additive_return": float(
            episodes["net_additive_return"].median()
        ),
    }
    episode_by_fold = [
        {
            "fold": int(fold),
            "episodes": int(len(frame)),
            "active_days": int(frame["duration_days"].sum()),
            "winners": int(frame["winner_after_cost"].sum()),
            "net_additive_return": float(frame["net_additive_return"].sum()),
        }
        for fold, frame in episodes.groupby("fold", sort=True)
    ]
    episode_by_fold_lookup = {
        int(row["fold"]): row for row in episode_by_fold
    }
    fold_candidate_daily = daily_returns.loc[
        (daily_returns["cost_bps"] == cost_bps)
        & daily_returns["scope"].str.startswith("fold_")
        & (daily_returns["strategy"] == "candidate")
    ].copy()
    fold_pnl = []
    for scope, frame in fold_candidate_daily.groupby("scope", sort=True):
        fold = int(scope.split("_")[1])
        episode_values = episode_by_fold_lookup[fold]
        fold_pnl.append({
            **episode_values,
            "gross_compounded_return": _compound(frame["gross_return"]),
            "net_compounded_return": _compound(frame["net_return"]),
            "turnover_sum": float(frame["turnover"].sum()),
            "cost_sum": float(frame["cost"].sum()),
        })
    episode_by_symbol = [
        {
            "symbol": symbol,
            "episodes": int(len(frame)),
            "winners": int(frame["winner_after_cost"].sum()),
            "net_additive_return": float(frame["net_additive_return"].sum()),
            "equal_fold_capital_contribution": float(
                frame["net_additive_return"].sum()
                * autopsy["accounting"]["fold_capital_weight"]
            ),
        }
        for symbol, frame in episodes.groupby("symbol", sort=True)
    ]
    fold_capital_weight = float(autopsy["accounting"]["fold_capital_weight"])
    regime_all_summary = summarize_signal_time_regimes(
        fold_dates,
        active_only=False,
        fold_capital_weight=fold_capital_weight,
    )
    regime_active_summary = summarize_signal_time_regimes(
        fold_dates,
        active_only=True,
        fold_capital_weight=fold_capital_weight,
    )
    active_cash_conditioning = [
        {
            "candidate_state": "active" if active else "cash",
            "fold_date_count": int(len(frame)),
            "candidate_net_mean": float(frame["candidate_net_return"].mean()),
            "candidate_net_additive_fold_capital_contribution": float(
                frame["candidate_net_return"].sum()
                * autopsy["accounting"]["fold_capital_weight"]
            ),
            "dual_momentum_net_mean": float(frame["dual_momentum_net_return"].mean()),
            "equal_weight_net_mean": float(frame["equal_weight_net_return"].mean()),
            "fold_universe_mean_return": float(
                frame["fold_mean_observed_simple_return"].mean()
            ),
        }
        for active, frame in fold_dates.groupby("candidate_active", sort=True)
    ]

    aggregate_candidate = daily_returns.loc[
        (daily_returns["cost_bps"] == cost_bps)
        & (daily_returns["scope"] == "aggregate_equal_fold_capital")
        & (daily_returns["strategy"] == "candidate")
    ]
    daily_contribution = fold_dates["candidate_net_return"] * fold_capital_weight
    concentration = {
        "aggregate_calendar_day_losses": loss_concentration(
            aggregate_candidate["net_return"],
            autopsy["accounting"]["loss_concentration_top_n_days"],
        ),
        "episode_losses": loss_concentration(
            episodes["net_additive_return"],
            autopsy["accounting"]["loss_concentration_top_n_days"],
        ),
        "fold_day_capital_contribution_losses": loss_concentration(
            daily_contribution,
            autopsy["accounting"]["loss_concentration_top_n_days"],
        ),
    }
    worst_episodes = episodes.nsmallest(10, "net_additive_return").copy()
    best_episodes = episodes.nlargest(10, "net_additive_return").copy()
    fixed_bins = {
        "calibrated_q50": _bin_summary(
            fold_dates,
            "top1_q50",
            list(autopsy["fixed_descriptive_bins"]["calibrated_q50"]),
        ),
        "calibrated_q10": _bin_summary(
            fold_dates,
            "top1_q10",
            list(autopsy["fixed_descriptive_bins"]["calibrated_q10"]),
        ),
        "top1_q50_margin": _bin_summary(
            fold_dates,
            "top1_q50_margin",
            list(autopsy["fixed_descriptive_bins"]["top1_q50_margin"]),
        ),
    }
    aggregate_gross = _compound(aggregate_candidate["gross_return"])
    aggregate_cost = float(aggregate_candidate["cost"].sum())
    aggregate_turnover = float(aggregate_candidate["turnover"].sum())
    base_metrics = metrics_v37["aggregate_metrics"][str(cost_bps)]
    retired_input_decision = (
        result_v37["decision"]
        == "retire_candidate_family_without_target_evaluation"
        and not gate_v37["passed"]
    )
    candidate_drawdown_is_lower_than_controls = bool(
        abs(base_metrics["candidate"]["max_drawdown"])
        < abs(base_metrics["dual_momentum_30"]["max_drawdown"])
        and abs(base_metrics["candidate"]["max_drawdown"])
        < abs(base_metrics["equal_weight_buy_hold"]["max_drawdown"])
    )
    failure_attribution = {
        "q50_ranking_failure": bool(
            ranking_all["top1_hit_rate"] <= ranking_all["random_top1_expectation"]
            and ranking_all["mean_top1_excess_return_vs_fold_mean"] < 0
        ),
        "active_day_alpha_failure": bool(
            ranking_active["mean_top1_observed_simple_return"] < 0
        ),
        "cost_is_primary_cause": bool(
            aggregate_gross >= 0 and base_metrics["candidate"]["total_return"] < 0
        ),
        "lower_drawdown_coincided_with_cash_not_positive_active_alpha": bool(
            candidate_drawdown_is_lower_than_controls
            and episode_summary["active_rate"] < 1.0
            and ranking_active["mean_top1_observed_simple_return"] < 0
        ),
        "tail_structure_is_diagnostic_not_actionable": bool(
            ranking_all["mean_spearman_q10"]
            > ranking_all["mean_spearman_q50"]
            and not autopsy["constraints"][
                "alternative_policy_or_threshold_test_allowed"
            ]
        ),
        "retirement_remains_required": retired_input_decision,
    }
    autopsy_spec = {
        "version": "v37_failure_autopsy",
        "input_sha256": autopsy["expected_input_sha256"],
        "constraints": autopsy["constraints"],
        "accounting": autopsy["accounting"],
        "slices": autopsy["slices"],
        "ranking_diagnostics": autopsy["ranking_diagnostics"],
        "fixed_descriptive_bins": autopsy["fixed_descriptive_bins"],
        "continuous_diagnostics": autopsy["continuous_diagnostics"],
        "limitations": autopsy["limitations"],
    }
    autopsy_spec["autopsy_spec_sha256"] = _canonical_sha256(autopsy_spec)

    checks = {
        "all_frozen_input_hashes_match": all(input_checks.values()),
        "v37_is_retired_and_gate_failed": retired_input_decision,
        "v37_was_executed_exactly_once": receipt_v37[
            "evaluation_execution_count"
        ] == 1,
        "prediction_and_fold_date_counts_match": len(predictions) == 4521
        and len(fold_dates) == 519,
        "exact_three_folds_and_173_dates": predictions["fold"].nunique() == 3
        and predictions["date"].nunique() == 173,
        "target_assets_are_absent": not TARGET_SYMBOLS.intersection(
            predictions["symbol"].unique()
        ),
        "candidate_positions_remain_frozen_long_or_cash": bool(
            predictions["candidate_weight"].isin([0.0, 1.0]).all()
        ) and bool(
            predictions.groupby(["fold", "date"])["candidate_weight"].sum().isin(
                [0.0, 1.0]
            ).all()
        ),
        "only_registered_base_cost_described": cost_bps == 10,
        "context_counts_are_structurally_complete": bool(
            (
                fold_dates["context_count_min"]
                == ((fold_dates["available_assets"] - 1)
                    * (fold_dates["available_assets"] - 2) // 2)
            ).all()
        ) and bool(
            (fold_dates["context_count_min"] == fold_dates["context_count_max"]).all()
        ),
        "signal_time_regime_covers_every_fold_date": sum(
            row["fold_date_count"] for row in regime_all_summary
        ) == len(fold_dates),
        "analysis_inputs_are_exactly_allowlisted": set(paths) == {
            "result",
            "metrics",
            "gate_result",
            "evaluation_spec",
            "evaluation_receipt",
            "predictions",
            "daily_returns",
        },
        "seed_and_context_disagreement_marked_unavailable": autopsy[
            "limitations"
        ]["seed_disagreement"] == "unavailable_not_persisted"
        and autopsy["limitations"]["triplet_context_disagreement"]
        == "unavailable_not_persisted",
        "v37_retirement_decision_unchanged": (
            _load_json(paths["result"])["decision"] == result_v37["decision"]
            and result_v37["decision"]
            == "retire_candidate_family_without_target_evaluation"
        ),
        "input_hashes_still_match_after_analysis": all(
            _sha256_file(paths[name]) == expected
            for name, expected in autopsy["expected_input_sha256"].items()
        ),
        **ledger_checks,
    }
    if not all(checks.values()):
        raise RuntimeError(f"V37 failure-autopsy audit failed: {checks}")
    result = {
        "version": "v37_failure_autopsy",
        "decision": "v37_retirement_confirmed_new_ex_ante_family_required",
        "autopsy_spec": autopsy_spec,
        "baseline_10bps": base_metrics,
        "ranking": {
            "all_fold_dates": ranking_all,
            "candidate_active_fold_dates": ranking_active,
            "by_fold": ranking_by_fold,
            "by_fold_month": ranking_by_fold_month,
            "direction_and_calibration": direction,
            "continuous_diagnostics": continuous,
            "fixed_descriptive_bins": fixed_bins,
        },
        "trades": {
            "summary": episode_summary,
            "by_fold": fold_pnl,
            "by_symbol": episode_by_symbol,
            "aggregate_gross_compounded_return": aggregate_gross,
            "aggregate_turnover_sum": aggregate_turnover,
            "aggregate_cost_sum": aggregate_cost,
            "loss_concentration": concentration,
            "worst_episodes": _json_episode_records(worst_episodes),
            "best_episodes": _json_episode_records(best_episodes),
        },
        "slices": {
            "monthly_strategy_returns": _monthly_strategy_summary(
                daily_returns, cost_bps
            ),
            "signal_time_regime_all_fold_dates": regime_all_summary,
            "signal_time_regime_active_days": regime_active_summary,
            "candidate_active_vs_cash": active_cash_conditioning,
        },
        "failure_attribution": failure_attribution,
        "limitations": {
            **autopsy["limitations"],
            "v37_window_status": "consumed_development_evidence_not_clean_holdout",
            "diagnostic_buckets_are_not_policy_thresholds": True,
        },
        "recommendation": {
            "current_family": "remain_retired",
            "next_family_primary_change": (
                "train_cross_sectional_ranking_or_excess_return_objective"
            ),
            "q10_tail_structure": (
                "hypothesis_for_future_pre_registration_only_no_v37_policy_test"
            ),
            "target_assets": "remain_sealed",
            "next_clean_evaluation": "requires_genuinely_new_future_data",
        },
        "audit": {"passed": True, "checks": checks},
    }
    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    fold_date_path = output / "fold_date_diagnostics.parquet"
    episode_path = output / "holding_episodes.parquet"
    fold_dates.to_parquet(fold_date_path, index=False)
    episodes.to_parquet(episode_path, index=False)
    result["artifacts"] = {
        "fold_date_diagnostics_sha256": _sha256_file(fold_date_path),
        "holding_episodes_sha256": _sha256_file(episode_path),
    }
    _write_json(output / "autopsy_spec.json", autopsy_spec)
    _write_json(output / "ranking_diagnostics.json", result["ranking"])
    _write_json(output / "trade_diagnostics.json", result["trades"])
    _write_json(output / "slice_diagnostics.json", result["slices"])
    _write_json(output / "audit.json", result["audit"])
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    _write_json(output / "result.json", result)
    return result
