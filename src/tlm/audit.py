from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def audit_artifacts(output_dir: str | Path) -> dict[str, object]:
    output = Path(output_dir)
    required = [
        "resolved_config.yaml", "predictions.parquet", "metrics.json",
        "equity_curve.png", "report.md",
    ]
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing experiment artifacts: {missing}")
    predictions = pd.read_parquet(output / "predictions.parquet")
    pred_columns = [column for column in predictions if column.startswith("pred_")]
    actual_columns = [column for column in predictions if column.startswith("actual_")]
    checks: dict[str, bool] = {
        "has_predictions": bool(pred_columns and actual_columns and len(predictions)),
        "finite_predictions": bool(np.isfinite(predictions[pred_columns].to_numpy()).all()),
        "finite_actuals": bool(np.isfinite(predictions[actual_columns].to_numpy()).all()),
        "unique_model_dates": not predictions.duplicated(["model", "date"]).any(),
        "ordered_within_model": all(
            group["date"].is_monotonic_increasing
            for _, group in predictions.groupby("model", sort=False)
        ),
        "contiguous_dates_within_model": all(
            (group["date"].diff().dropna() == pd.Timedelta(days=1)).all()
            for _, group in predictions.groupby("model", sort=False)
        ),
    }
    pivot = predictions.pivot(index="date", columns="model", values=actual_columns)
    checks["actuals_match_across_models"] = all(
        np.allclose(
            pivot[actual].dropna().to_numpy(),
            pivot[actual].dropna().to_numpy()[:, [0]],
        )
        for actual in actual_columns
        if pivot[actual].shape[1] > 1 and len(pivot[actual].dropna())
    )
    with (output / "metrics.json").open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    with (output / "resolved_config.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    numeric_metrics = [
        value for model_metrics in metrics.values() for value in model_metrics.values()
        if isinstance(value, (int, float))
    ]
    checks["finite_metrics"] = bool(numeric_metrics) and bool(np.isfinite(numeric_metrics).all())
    rate = float(config["strategy"]["cost_bps"]) / 10_000.0
    checks["cost_matches_turnover"] = all(
        np.isclose(values["cost_paid"], values["turnover"] * rate)
        for values in metrics.values()
    )
    target_mode = config.get("target", {}).get("mode", "next_open_to_close")
    if target_mode == "next_open_to_close":
        checks["intraday_round_trip_accounting"] = all(
            values["position_changes"] == 2 * values["trade_count"]
            for name, values in metrics.items()
            if name in {"ridge", "transformer", "equal_weight_intraday"}
        )
    else:
        checks["persistent_turnover_accounting"] = all(
            values["position_changes"] == round(values["turnover"])
            for name, values in metrics.items()
            if name in {"ridge", "transformer"}
        )
    checks["equity_remains_positive"] = all(
        values["total_return"] > -1.0 for values in metrics.values()
    )
    daily_path = output / "daily_returns.parquet"
    if daily_path.exists():
        daily = pd.read_parquet(daily_path)
        return_columns = [column for column in daily if column.endswith("__net_return")]
        equity_columns = [column for column in daily if column.endswith("__equity")]
        checks["daily_returns_are_finite"] = bool(return_columns) and bool(
            np.isfinite(daily[return_columns].to_numpy()).all()
        )
        checks["daily_equity_is_positive"] = bool(equity_columns) and bool(
            (daily[equity_columns] > 0).all().all()
        )
        checks["daily_dates_are_unique_and_ordered"] = bool(
            not daily["date"].duplicated().any() and daily["date"].is_monotonic_increasing
        )
    result: dict[str, object] = {
        "passed": all(checks.values()),
        "checks": checks,
        "models": sorted(predictions["model"].unique().tolist()),
        "prediction_rows": int(len(predictions)),
    }
    with (output / "audit.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    if not result["passed"]:
        raise RuntimeError(f"Artifact audit failed: {checks}")
    return result
