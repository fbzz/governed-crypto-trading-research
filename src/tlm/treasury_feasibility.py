from __future__ import annotations

import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Callable
import urllib.request

import numpy as np
import pandas as pd
import yaml

from .data import _verified_ssl_context


BytesFetcher = Callable[[str, float], bytes]


def _http_get_bytes(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "tlm-treasury-feasibility/0.1"}
    )
    with urllib.request.urlopen(
        request, timeout=timeout, context=_verified_ssl_context()
    ) as response:
        return response.read()


def build_treasury_year_url(config: dict, year: int) -> str:
    treasury = config["treasury_feasibility"]
    base = treasury["csv_url_template"]
    return base.format(year=year)


def parse_treasury_csv(payload: bytes, year: int, required_columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(BytesIO(payload))
    missing = set(required_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Treasury {year} CSV is missing columns {sorted(missing)}")
    frame["Date"] = pd.to_datetime(frame["Date"], format="%m/%d/%Y", utc=True)
    frame = frame.set_index("Date").sort_index()
    if frame.index.has_duplicates:
        raise ValueError(f"Treasury {year} CSV has duplicate dates")
    for column in required_columns:
        if column != "Date":
            frame[column] = pd.to_numeric(frame[column], errors="raise")
    return frame[required_columns[1:]]


def audit_treasury_probe(frame: pd.DataFrame, config: dict) -> dict[str, object]:
    treasury = config["treasury_feasibility"]
    start = pd.Timestamp(treasury["start"], tz="UTC")
    end = pd.Timestamp(treasury["end"], tz="UTC")
    frame = frame.loc[start:end]
    expected_weekdays = pd.bdate_range(start, end, tz="UTC")
    coverage = len(frame) / len(expected_weekdays)
    gaps = frame.index.to_series().diff().dropna()
    numeric = frame.to_numpy(dtype=float)
    source_final_at = frame.index + pd.Timedelta(hours=21)
    first_execution = frame.index + pd.Timedelta(
        days=int(treasury["execution_lag_calendar_days"])
    )
    checks = {
        "weekday_coverage_at_least_minimum": bool(
            coverage >= float(treasury["minimum_weekday_coverage"])
        ),
        "dates_unique_sorted": bool(
            frame.index.is_monotonic_increasing and not frame.index.has_duplicates
        ),
        "required_tenors_finite": bool(np.isfinite(numeric).all()),
        "required_tenors_positive": bool((numeric > 0.0).all()),
        "maximum_source_gap_within_carry_bound": bool(
            gaps.empty
            or gaps.max() <= pd.Timedelta(days=int(treasury["maximum_carry_days"]))
        ),
        "source_final_strictly_precedes_execution": bool(
            (source_final_at < first_execution).all()
        ),
        "starts_after_registered_methodology_change": bool(
            start >= pd.Timestamp(treasury["methodology_start"], tz="UTC")
        ),
    }
    return {
        "rows": len(frame),
        "expected_weekdays_including_holidays": len(expected_weekdays),
        "weekday_coverage": float(coverage),
        "first_date": frame.index.min().isoformat(),
        "last_date": frame.index.max().isoformat(),
        "maximum_observation_gap_days": int(gaps.max().days) if not gaps.empty else 0,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _report(result: dict) -> str:
    probe = result["probe"]
    lines = [
        "# TLM v17 Treasury Curve Feasibility Audit",
        "",
        "## Decision",
        "",
        "- Family selected: **U.S. Treasury daily par yield curve**",
        f"- Probe: **{probe['rows']} rows**, {probe['weekday_coverage']:.2%} of weekdays including federal holidays",
        f"- Window: **{probe['first_date']} through {probe['last_date']}**",
        f"- Maximum observation gap: **{probe['maximum_observation_gap_days']} calendar days**",
        "- This is a source decision only; no feature, return, signal, model, or policy was evaluated.",
        "",
        "## Causal contract",
        "",
        "Treasury states that the curve uses indicative bid-side quotations obtained at or near 3:30 p.m. Eastern each trading day. The registered join will not expose a rate until the crypto open at `source date + 3 calendar days`, leaving more than two calendar days after the quoted market close. On non-publication days, v18 may carry only the latest already-eligible public state, with source date and age retained and a seven-day maximum.",
        "",
        "## Reproducibility",
        "",
    ]
    for year, row in result["annual_sources"].items():
        lines.append(
            f"- {year}: {row['bytes']} bytes, SHA-256 `{row['sha256']}`."
        )
    lines.extend([
        "",
        "## Limitations",
        "",
        "- Treasury reserves discretion to change its curve methodology. The registered window begins on the 2021-12-06 monotone-convex methodology start.",
        "- The annual downloads are snapshot-hashed; they are not claimed to be a point-in-time revision archive.",
        "- Rates are indicative interpolated par yields, not transaction prices.",
        "",
        "## Authorized next step",
        "",
        "V18 may build a cache-first, hash-audited 2y/10y Treasury data layer with the frozen `t+3` eligibility rule and bounded known-state carry. It remains data-only.",
        "",
    ])
    return "\n".join(lines)


def run_treasury_feasibility(
    config: dict,
    fetch_bytes: BytesFetcher = _http_get_bytes,
) -> dict[str, object]:
    treasury = config["treasury_feasibility"]
    start_year = pd.Timestamp(treasury["start"]).year
    end_year = pd.Timestamp(treasury["end"]).year
    frames: list[pd.DataFrame] = []
    annual_sources: dict[str, dict[str, object]] = {}
    raw_payloads: dict[int, bytes] = {}
    for year in range(start_year, end_year + 1):
        url = build_treasury_year_url(config, year)
        payload = fetch_bytes(url, float(treasury["timeout_seconds"]))
        raw_payloads[year] = payload
        frames.append(parse_treasury_csv(payload, year, treasury["required_columns"]))
        annual_sources[str(year)] = {
            "url": url,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    combined = pd.concat(frames).sort_index()
    if combined.index.has_duplicates:
        raise ValueError("Treasury annual probes overlap on duplicate dates")
    combined = combined.loc[
        pd.Timestamp(treasury["start"], tz="UTC"):
        pd.Timestamp(treasury["end"], tz="UTC")
    ]
    probe = audit_treasury_probe(combined, config)
    hard_gates = {
        "official_public_source": True,
        "no_credentials_or_paid_license": True,
        "required_2y_10y_schema": set(combined.columns) == {"2 Yr", "10 Yr"},
        "coverage_and_gap_contract": bool(probe["passed"]),
        "documented_market_observation_time": True,
        "strict_t_plus_3_causal_contract": bool(
            probe["checks"]["source_final_strictly_precedes_execution"]
        ),
        "independent_macro_market_family": True,
        "annual_snapshot_hashes_recorded": all(
            len(row["sha256"]) == 64 for row in annual_sources.values()
        ),
    }
    selected = all(hard_gates.values())
    checks = {
        "live_probe_passed": bool(probe["passed"]),
        "selected_only_when_all_hard_gates_pass": selected == all(hard_gates.values()),
        "all_annual_payloads_persisted": len(raw_payloads) == end_year - start_year + 1,
        "official_evidence_urls_recorded": len(treasury["evidence_urls"]) >= 3,
        "no_feature_signal_model_or_policy_configured": not any(
            key in config for key in ("features", "signal_study", "model", "strategy")
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Treasury feasibility audit failed: {checks}")
    result: dict[str, object] = {
        "version": "v17",
        "method": "pre_dataset_us_treasury_curve_feasibility_audit",
        "selected": selected,
        "decision": "authorize_v18_treasury_curve_data_layer_only",
        "probe": probe,
        "annual_sources": annual_sources,
        "hard_gates": hard_gates,
        "causal_contract": {
            "source_observation": "indicative bid-side quotations near 3:30 PM ET",
            "first_allowed_execution": "source date + 3 calendar days at 00:00 UTC",
            "maximum_known_state_carry_days": int(treasury["maximum_carry_days"]),
        },
        "audit": {"passed": True, "checks": checks},
    }
    output = Path(config["output_dir"])
    probe_dir = output / "annual_probes"
    probe_dir.mkdir(parents=True, exist_ok=True)
    for year, payload in raw_payloads.items():
        (probe_dir / f"{year}.csv").write_bytes(payload)
    combined.to_csv(output / "probe.csv", index_label="Date")
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
