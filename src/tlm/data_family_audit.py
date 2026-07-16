from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
import yaml

from .data import _verified_ssl_context


JsonFetcher = Callable[[str, float], dict]


def _http_get_json(url: str, timeout: float) -> dict:
    request = urllib.request.Request(
        url, headers={"User-Agent": "tlm-source-feasibility/0.1"}
    )
    with urllib.request.urlopen(
        request, timeout=timeout, context=_verified_ssl_context()
    ) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object from {url}")
    return payload


def parse_dvol_response(payload: dict, currency: str) -> tuple[pd.DataFrame, int | None]:
    if payload.get("error"):
        raise RuntimeError(f"Deribit returned an error for {currency}: {payload['error']}")
    result = payload.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("data"), list):
        raise ValueError(f"Unexpected Deribit DVOL response for {currency}")
    rows = result["data"]
    if not rows:
        raise ValueError(f"Empty Deribit DVOL response for {currency}")
    if any(not isinstance(row, list) or len(row) != 5 for row in rows):
        raise ValueError(f"Unexpected Deribit DVOL row schema for {currency}")
    frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close"])
    frame["timestamp"] = pd.to_datetime(
        pd.to_numeric(frame["timestamp"], errors="raise"), unit="ms", utc=True
    )
    frame = frame.set_index("timestamp")
    frame = frame.astype(float)
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"Deribit DVOL timestamps are not unique and sorted for {currency}")
    values = frame[["open", "high", "low", "close"]]
    if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"Deribit DVOL contains non-finite values for {currency}")
    if (values <= 0).any().any():
        raise ValueError(f"Deribit DVOL contains non-positive values for {currency}")
    if (frame["high"] < frame[["open", "close"]].max(axis=1)).any():
        raise ValueError(f"Deribit DVOL high is below open/close for {currency}")
    if (frame["low"] > frame[["open", "close"]].min(axis=1)).any():
        raise ValueError(f"Deribit DVOL low is above open/close for {currency}")
    continuation = result.get("continuation")
    return frame, None if continuation is None else int(continuation)


def download_dvol(
    endpoint: str,
    currency: str,
    start: str,
    end: str,
    resolution: str,
    timeout: float,
    fetch_json: JsonFetcher = _http_get_json,
) -> tuple[pd.DataFrame, list[dict]]:
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000)
    cursor_end = end_ms
    frames: list[pd.DataFrame] = []
    payloads: list[dict] = []
    seen_cursors: set[int] = set()
    while True:
        params = {
            "currency": currency,
            "start_timestamp": start_ms,
            "end_timestamp": cursor_end,
            "resolution": resolution,
        }
        url = f"{endpoint}?{urllib.parse.urlencode(params)}"
        payload = fetch_json(url, timeout)
        frame, continuation = parse_dvol_response(payload, currency)
        frames.append(frame)
        payloads.append(payload)
        if continuation is None:
            break
        # Deribit returns the newest 1,000 observations and a timestamp just
        # before that page. Pagination therefore moves end_timestamp backward.
        if continuation >= cursor_end or continuation in seen_cursors:
            raise RuntimeError(f"Non-retreating Deribit continuation for {currency}")
        seen_cursors.add(continuation)
        cursor_end = continuation
    combined = pd.concat(frames).sort_index()
    if combined.index.has_duplicates:
        duplicates = combined.index[combined.index.duplicated()].unique()
        for timestamp in duplicates:
            rows = combined.loc[[timestamp]]
            if not rows.eq(rows.iloc[0]).all(axis=None):
                raise ValueError(f"Conflicting paginated DVOL row for {currency} {timestamp}")
        combined = combined.loc[~combined.index.duplicated(keep="first")]
    return combined.loc[
        pd.Timestamp(start, tz="UTC"):pd.Timestamp(end, tz="UTC")
    ], payloads


def audit_dvol_frame(
    frame: pd.DataFrame,
    start: str,
    end: str,
    minimum_coverage: float,
) -> dict[str, object]:
    expected = pd.date_range(start, end, freq="D", tz="UTC")
    observed = frame.index.intersection(expected)
    coverage = len(observed) / len(expected)
    missing = expected.difference(frame.index)
    spacing_ok = bool(
        len(frame) <= 1
        or (frame.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()
    )
    candle_final_at = frame.index + pd.Timedelta(days=1)
    first_strict_execution_open = frame.index + pd.Timedelta(days=2)
    strict_causality = bool((candle_final_at < first_strict_execution_open).all())
    checks = {
        "coverage_at_least_minimum": bool(coverage >= minimum_coverage),
        "daily_unique_sorted": bool(
            frame.index.is_monotonic_increasing and not frame.index.has_duplicates
        ),
        "daily_spacing_complete": spacing_ok,
        "strict_causal_lag_contract": strict_causality,
        "finite_positive_ohlc": bool(
            np.isfinite(frame.to_numpy()).all() and (frame.to_numpy() > 0).all()
        ),
    }
    return {
        "rows": len(frame),
        "expected_rows": len(expected),
        "coverage": float(coverage),
        "first_timestamp": frame.index.min().isoformat(),
        "last_timestamp": frame.index.max().isoformat(),
        "missing_dates": [value.date().isoformat() for value in missing],
        "timestamp_contract": {
            "candle_interval": "[timestamp, timestamp + 1 day)",
            "candle_final_at": "timestamp + 1 day",
            "first_strict_execution_open": "timestamp + 2 days",
            "required_post_close_buffer_days": 1,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


def score_and_select_candidates(
    candidates: dict[str, dict], weights: dict[str, float]
) -> tuple[dict[str, dict], list[str]]:
    if not np.isclose(sum(weights.values()), 1.0):
        raise ValueError("Data-family audit weights must sum to 1.0")
    evaluated: dict[str, dict] = {}
    for name, candidate in candidates.items():
        criteria = candidate["criteria"]
        missing = set(weights) - set(criteria)
        if missing:
            raise ValueError(f"{name} is missing criteria: {sorted(missing)}")
        score = sum(float(weights[key]) * float(criteria[key]) for key in weights)
        hard_gates = {key: bool(value) for key, value in candidate["hard_gates"].items()}
        row = dict(candidate)
        row["score"] = float(score)
        row["hard_gate_passed"] = all(hard_gates.values())
        row["hard_gates"] = hard_gates
        evaluated[name] = row
    eligible = [name for name, row in evaluated.items() if row["hard_gate_passed"]]
    selected = sorted(eligible, key=lambda name: (-evaluated[name]["score"], name))[:1]
    for name, row in evaluated.items():
        row["selected"] = name in selected
    return evaluated, selected


def _canonical_hash(payloads: list[dict]) -> str:
    canonical = json.dumps(payloads, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _report(result: dict) -> str:
    lines = [
        "# TLM v13 Independent Data Family Feasibility Audit",
        "",
        "## Decision",
        "",
        f"- Selected families: **{', '.join(result['selected']) or 'none'}**",
        "- This is a source-feasibility decision only; no model, signal, or policy was evaluated.",
        "",
        "## Candidate matrix",
        "",
        "| Candidate | Score | Hard gates | Decision | Principal constraint |",
        "|---|---:|---|---|---|",
    ]
    for name, row in result["candidates"].items():
        decision = "SELECT" if row["selected"] else "DEFER/REJECT"
        lines.append(
            f"| {name} | {row['score']:.2f} | "
            f"{'PASS' if row['hard_gate_passed'] else 'FAIL'} | {decision} | "
            f"{row['principal_constraint']} |"
        )
    lines.extend(["", "## Deribit DVOL live probe", ""])
    for currency, probe in result["dvol_probe"].items():
        lines.append(
            f"- {currency}: {probe['rows']}/{probe['expected_rows']} daily rows "
            f"({probe['coverage']:.2%}), {probe['first_timestamp']} through "
            f"{probe['last_timestamp']}; SHA-256 `{probe['payload_sha256']}`."
        )
    lines.extend([
        "",
        "DVOL daily candles become final at the following UTC boundary. To preserve the project's strict `< execution_open` rule, v14 must lag a completed candle until the next day's open: a candle stamped `t` may first affect the portfolio at `t+2`.",
        "",
        "## Constraints recorded",
        "",
        "- Deribit: public unauthenticated market-data method, but subject to API usage limits; cache locally and do not claim redistribution rights.",
        "- FRED/ALFRED macro: revision-safe vintage observations require an API key and explicit release-time handling, so this is deferred under the no-key contract.",
        "- Historical order book/liquidations: the audited Binance public archive catalog does not provide a complete reconstructable history for this window; current snapshots/streams are not a historical dataset.",
        "- Clean future holdout: valid validation evidence, but not an independent feature family and therefore cannot be selected by this audit.",
        "",
        "## Authorized next step",
        "",
        "Build v14 as a cached, hash-audited BTC/ETH DVOL data layer only. Treat DVOL as a market-level options-volatility family available to the BTC/ETH/SOL decision, preserve missing dates, apply the strict one-day post-close buffer, and do not train a model until the data audit passes.",
        "",
    ])
    return "\n".join(lines)


def run_data_family_audit(
    config: dict,
    fetch_json: JsonFetcher = _http_get_json,
) -> dict[str, object]:
    audit_config = config["data_family_audit"]
    output = Path(config["output_dir"])
    probe_dir = output / "probes"
    probe_dir.mkdir(parents=True, exist_ok=True)
    dvol_config = audit_config["deribit_dvol"]
    dvol_probe: dict[str, dict] = {}
    all_probes_passed = True
    for currency in dvol_config["currencies"]:
        frame, payloads = download_dvol(
            endpoint=dvol_config["endpoint"],
            currency=currency,
            start=dvol_config["start"],
            end=dvol_config["end"],
            resolution=dvol_config["resolution"],
            timeout=float(dvol_config["timeout_seconds"]),
            fetch_json=fetch_json,
        )
        probe = audit_dvol_frame(
            frame,
            dvol_config["start"],
            dvol_config["end"],
            float(dvol_config["minimum_daily_coverage"]),
        )
        probe["payload_sha256"] = _canonical_hash(payloads)
        probe["page_count"] = len(payloads)
        dvol_probe[currency] = probe
        all_probes_passed = all_probes_passed and bool(probe["passed"])
        (probe_dir / f"{currency}.json").write_text(
            json.dumps(payloads, indent=2, sort_keys=True), encoding="utf-8"
        )
        frame.to_parquet(probe_dir / f"{currency}.parquet")

    candidates = json.loads(json.dumps(audit_config["candidates"]))
    dvol_candidate = candidates["deribit_dvol"]
    dvol_candidate["hard_gates"]["historical_coverage"] = all_probes_passed
    dvol_candidate["hard_gates"]["strict_causal_contract"] = all(
        probe["checks"]["strict_causal_lag_contract"] for probe in dvol_probe.values()
    )
    dvol_candidate["criteria"]["historical_coverage"] = min(
        float(probe["coverage"]) for probe in dvol_probe.values()
    )
    evaluated, selected = score_and_select_candidates(
        candidates, audit_config["weights"]
    )
    checks = {
        "at_most_one_family_selected": len(selected) <= 1,
        "selected_family_passes_hard_gates": all(
            evaluated[name]["hard_gate_passed"] for name in selected
        ),
        "dvol_live_probe_passed": all_probes_passed,
        "evidence_urls_recorded": all(
            bool(row.get("evidence_urls")) for row in evaluated.values()
        ),
        "no_model_or_policy_configuration": not any(
            key in config for key in ("model", "strategy", "validation_suite")
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Data-family feasibility audit failed: {checks}")
    result: dict[str, object] = {
        "version": "v13",
        "method": "pre_model_independent_data_family_feasibility_audit",
        "selected": selected,
        "candidates": evaluated,
        "dvol_probe": dvol_probe,
        "audit": {"checks": checks, "passed": True},
        "decision": (
            "authorize_v14_deribit_dvol_data_layer_only"
            if selected == ["deribit_dvol"]
            else "stop_independent_data_research"
        ),
    }
    (output / "feasibility.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "audit.json").write_text(
        json.dumps(result["audit"], indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    return result
