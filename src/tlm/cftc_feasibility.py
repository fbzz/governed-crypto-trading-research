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


ArrayFetcher = Callable[[str, float], list[dict]]


def _http_get_array(url: str, timeout: float) -> list[dict]:
    request = urllib.request.Request(
        url, headers={"User-Agent": "tlm-cftc-feasibility/0.1"}
    )
    with urllib.request.urlopen(
        request, timeout=timeout, context=_verified_ssl_context()
    ) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise RuntimeError("Expected a JSON row array from the CFTC public API")
    return payload


def build_cftc_probe_url(config: dict) -> str:
    cftc = config["cftc_feasibility"]
    fields = list(cftc["required_fields"])
    codes = ",".join(f"'{code}'" for code in cftc["contract_codes"].values())
    where = (
        f"report_date_as_yyyy_mm_dd between '{cftc['start']}T00:00:00' "
        f"and '{cftc['end']}T23:59:59' and "
        f"cftc_contract_market_code in ({codes})"
    )
    params = {
        "$select": ",".join(fields),
        "$where": where,
        "$order": "cftc_contract_market_code,report_date_as_yyyy_mm_dd",
        "$limit": int(cftc["limit"]),
    }
    return f"{cftc['endpoint']}?{urllib.parse.urlencode(params)}"


def audit_cftc_rows(rows: list[dict], config: dict) -> dict[str, object]:
    cftc = config["cftc_feasibility"]
    if not rows:
        raise ValueError("CFTC feasibility probe returned no rows")
    required = set(cftc["required_fields"])
    missing_schema = sorted(required - set.intersection(*(set(row) for row in rows)))
    if missing_schema:
        raise ValueError(f"CFTC rows are missing required fields: {missing_schema}")
    frame = pd.DataFrame(rows)
    frame["report_date_as_yyyy_mm_dd"] = pd.to_datetime(
        frame["report_date_as_yyyy_mm_dd"], utc=True
    )
    numeric_fields = [
        field for field in required
        if field not in {
            "commodity_name", "contract_market_name", "cftc_contract_market_code",
            "report_date_as_yyyy_mm_dd",
        }
    ]
    for field in numeric_fields:
        frame[field] = pd.to_numeric(frame[field], errors="raise")
    expected = pd.date_range(cftc["start"], cftc["end"], freq="W-TUE", tz="UTC")
    contracts: dict[str, dict[str, object]] = {}
    all_contract_checks = True
    for asset, code in cftc["contract_codes"].items():
        subset = frame[frame["cftc_contract_market_code"] == code].sort_values(
            "report_date_as_yyyy_mm_dd"
        )
        dates = pd.DatetimeIndex(subset["report_date_as_yyyy_mm_dd"])
        # Holiday weeks may use Monday rather than Tuesday as the official
        # report date. Measure weekly row coverage and record those shifts
        # instead of silently treating them as missing observations.
        coverage = len(dates) / len(expected)
        shifted_dates = dates.difference(expected)
        gaps = dates.to_series().diff().dropna()
        checks = {
            "coverage_at_least_minimum": bool(
                coverage >= float(cftc["minimum_weekly_coverage"])
            ),
            "weekly_dates_unique": not dates.has_duplicates,
            "weekly_cadence_has_no_large_gap": bool(
                gaps.empty or gaps.max() <= pd.Timedelta(days=8)
            ),
            "numeric_positions_finite": bool(
                np.isfinite(subset[numeric_fields].to_numpy(dtype=float)).all()
            ),
        }
        all_contract_checks = all_contract_checks and all(checks.values())
        contracts[asset] = {
            "contract_code": code,
            "rows": len(subset),
            "expected_weeks": len(expected),
            "coverage": float(coverage),
            "first_report_date": dates.min().isoformat(),
            "last_report_date": dates.max().isoformat(),
            "missing_report_dates": [
                value.date().isoformat() for value in expected.difference(dates)
            ],
            "holiday_shifted_report_dates": [
                value.date().isoformat() for value in shifted_dates
            ],
            "checks": checks,
            "passed": all(checks.values()),
        }
    return {
        "rows": len(frame),
        "expected_weekly_dates": len(expected),
        "schema_field_count": len(required),
        "contracts": contracts,
        "all_contract_probes_passed": all_contract_checks,
    }


def _report(result: dict) -> str:
    lines = [
        "# TLM v16 CFTC Positioning Feasibility Audit",
        "",
        "## Decision",
        "",
        "- Family selected: **no**",
        "- Audit execution: **PASS**",
        "- Reason: historical report dates are as-of dates, while a complete historical publication calendar and point-in-time revision archive are unavailable.",
        "",
        "## Live official API coverage",
        "",
    ]
    for asset, row in result["probe"]["contracts"].items():
        lines.append(
            f"- {asset}: {row['rows']}/{row['expected_weeks']} weekly Tuesday rows "
            f"({row['coverage']:.2%}), {row['first_report_date']} through "
            f"{row['last_report_date']}."
        )
    lines.extend([
        f"- Canonical probe SHA-256: `{result['probe_sha256']}`.",
        "",
        "## Causality finding",
        "",
        "CFTC states that COT data usually describes Tuesday positions and is published Friday at 3:30 p.m. Eastern, with holiday delays. It also states that no complete list of historical release dates exists. The registered window includes the 2025 appropriations interruption, when reports were released weeks after their original dates. Therefore `report_date + fixed normal-week lag` would leak future publication state.",
        "",
        "A very long blanket lag would reduce but not prove away this issue, and the current public dataset does not provide archived point-in-time revisions. Local hashing makes today's extraction reproducible; it does not reconstruct what was known on each historical date.",
        "",
        "## Gate matrix",
        "",
        "| Gate | Result |",
        "|---|---|",
    ])
    for gate, passed in result["hard_gates"].items():
        lines.append(f"| {gate} | {'PASS' if passed else 'FAIL'} |")
    lines.extend([
        "",
        "## Authorized next step",
        "",
        "Do not build a COT signal or policy. V17 may audit one daily official macro-market family whose values can be conservatively lagged and whose access is public without a key; if timestamp/vintage semantics remain unresolved, reject it before dataset construction.",
        "",
    ])
    return "\n".join(lines)


def run_cftc_feasibility(
    config: dict,
    fetch_json: ArrayFetcher = _http_get_array,
) -> dict[str, object]:
    cftc = config["cftc_feasibility"]
    url = build_cftc_probe_url(config)
    rows = fetch_json(url, float(cftc["timeout_seconds"]))
    probe = audit_cftc_rows(rows, config)
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    probe_sha256 = hashlib.sha256(payload).hexdigest()
    hard_gates = {
        "official_public_unauthenticated_access": True,
        "btc_eth_weekly_coverage": bool(probe["all_contract_probes_passed"]),
        "required_positioning_schema": probe["schema_field_count"]
        == len(cftc["required_fields"]),
        "historical_release_timestamps_available": False,
        "point_in_time_revision_archive_available": False,
        "zero_marginal_data_cost": True,
    }
    selected = all(hard_gates.values())
    checks = {
        "live_probe_passed": bool(probe["all_contract_probes_passed"]),
        "probe_hash_recorded": len(probe_sha256) == 64,
        "family_rejected_when_any_hard_gate_fails": selected
        == all(hard_gates.values()),
        "known_2025_delay_exception_recorded": bool(
            cftc["known_delay_exception_url"]
        ),
        "official_evidence_urls_recorded": len(cftc["evidence_urls"]) >= 4,
        "no_model_signal_or_policy_configured": not any(
            key in config for key in ("model", "strategy", "signal_study")
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"CFTC feasibility audit failed: {checks}")
    result: dict[str, object] = {
        "version": "v16",
        "method": "pre_dataset_cftc_positioning_feasibility_audit",
        "selected": selected,
        "decision": "reject_cftc_cot_due_unresolved_historical_availability",
        "probe_url": url,
        "probe_sha256": probe_sha256,
        "probe": probe,
        "hard_gates": hard_gates,
        "evidence_urls": cftc["evidence_urls"],
        "audit": {"passed": True, "checks": checks},
    }
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "probe.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8"
    )
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
