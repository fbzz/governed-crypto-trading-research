from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import pandas as pd
import yaml

from .derivatives_data import parse_checksum
from .non_target_dataset import build_asset_folds
from .non_target_inventory import (
    FetchBytes,
    KLINE_COLUMNS,
    S3Object,
    _download_cached,
    _http_get,
    _sha256_bytes,
    _timestamp_to_datetime,
    list_common_prefixes,
    list_s3_objects,
)


SELECTION_COLUMNS = ("symbol", "date", "quote_volume")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _symbol_from_prefix(prefix: str, root_prefix: str) -> str:
    if not prefix.startswith(root_prefix):
        raise ValueError(f"Unexpected symbol prefix: {prefix}")
    return prefix[len(root_prefix):].strip("/")


def scope_exclusion_reason(
    symbol: str,
    quote_asset: str,
    contract: dict,
) -> str | None:
    if not symbol.endswith(quote_asset):
        return "quote_asset"
    base = symbol[: -len(quote_asset)]
    if not base:
        return "empty_base"
    exclusions = contract["exclusions"]
    categories = (
        ("target_bases", "target_base"),
        ("target_proxy_bases", "target_proxy"),
        ("fiat_bases", "fiat_base"),
        ("stablecoin_bases", "stablecoin_base"),
        ("fan_token_bases", "fan_token_base"),
    )
    for key, reason in categories:
        if base in set(exclusions[key]):
            return reason
    if any(base.endswith(suffix) for suffix in exclusions["token_suffixes"]):
        return "excluded_suffix"
    return None


def _monthly_periods(start: str, end: str) -> list[str]:
    return [str(period) for period in pd.period_range(start, end, freq="M")]


def summarize_selection_listing(
    symbol: str,
    objects: list[S3Object],
    months: list[str],
    interval: str,
) -> dict[str, object]:
    expected = {
        month: f"{symbol}-{interval}-{month}.zip" for month in months
    }
    allowed_names = {
        name
        for archive_name in expected.values()
        for name in (archive_name, f"{archive_name}.CHECKSUM")
    }
    selection_objects = sorted(
        (
            {
                "key": item.key,
                "size": item.size,
                "etag": item.etag,
                "last_modified": item.last_modified,
            }
            for item in objects
            if Path(item.key).name in allowed_names
        ),
        key=lambda row: row["key"],
    )
    object_names = {Path(item["key"]).name for item in selection_objects}
    complete_months = [
        month
        for month, name in expected.items()
        if name in object_names and f"{name}.CHECKSUM" in object_names
    ]
    return {
        "symbol": symbol,
        "base": symbol.removesuffix("USDT"),
        "complete_months": complete_months,
        "complete_month_fraction": len(complete_months) / len(months),
        "selection_object_count": len(selection_objects),
        "selection_object_sha256": _canonical_sha256(selection_objects),
        "observation_months_only": complete_months == [
            month for month in complete_months if month in set(months)
        ],
    }


def inspect_quote_volume_zip(payload: bytes) -> dict[str, object]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected one CSV in kline archive, found {csv_names}")
        with archive.open(csv_names[0]) as raw:
            rows = list(csv.reader(io.TextIOWrapper(raw, encoding="utf-8")))
    if not rows:
        raise ValueError("Kline archive is empty")
    if rows[0] and rows[0][0].strip().lower().replace(" ", "_") == "open_time":
        rows = rows[1:]
    if not rows:
        raise ValueError("Kline archive contains only a header")

    dates: list[str] = []
    quote_volumes: list[float] = []
    timestamp_units: set[str] = set()
    for row in rows:
        if len(row) != len(KLINE_COLUMNS):
            raise ValueError(
                f"Kline row has {len(row)} columns, expected {len(KLINE_COLUMNS)}"
            )
        opened, unit = _timestamp_to_datetime(row[0])
        if opened.hour or opened.minute or opened.second or opened.microsecond:
            raise ValueError(f"Daily kline is not aligned to UTC midnight: {opened}")
        quote_volume = float(row[7])
        if not math.isfinite(quote_volume) or quote_volume < 0:
            raise ValueError("Kline archive has invalid quote volume")
        dates.append(opened.date().isoformat())
        quote_volumes.append(quote_volume)
        timestamp_units.add(unit)
    if len(dates) != len(set(dates)):
        raise ValueError("Kline archive contains duplicate UTC dates")
    if dates != sorted(dates):
        raise ValueError("Kline archive dates are not sorted")
    return {
        "row_count": len(dates),
        "dates": dates,
        "quote_volumes": quote_volumes,
        "first_date": dates[0],
        "last_date": dates[-1],
        "timestamp_units": sorted(timestamp_units),
        "selection_columns_read": list(SELECTION_COLUMNS[1:]),
        "schema_valid": True,
    }


def verify_selection_archive(
    symbol: str,
    month: str,
    interval: str,
    download_base_url: str,
    raw_dir: Path,
    timeout: float,
    force: bool,
    fetch_bytes: FetchBytes = _http_get,
) -> dict[str, object]:
    name = f"{symbol}-{interval}-{month}.zip"
    relative = f"data/spot/monthly/klines/{symbol}/{interval}/{name}"
    url = f"{download_base_url.rstrip('/')}/{relative}"
    cache_path = raw_dir / symbol / interval / name
    archive_payload, archive_cached = _download_cached(
        url, cache_path, timeout, force, fetch_bytes
    )
    checksum_path = cache_path.with_suffix(cache_path.suffix + ".CHECKSUM")
    checksum_payload, checksum_cached = _download_cached(
        f"{url}.CHECKSUM", checksum_path, timeout, force, fetch_bytes
    )
    expected = parse_checksum(checksum_payload)
    actual = _sha256_bytes(archive_payload)
    if actual != expected:
        raise ValueError(
            f"Checksum mismatch for {symbol} {month}: {actual} != {expected}"
        )
    inspection = inspect_quote_volume_zip(archive_payload)
    return {
        "symbol": symbol,
        "month": month,
        "url": url,
        "cache_path": str(cache_path),
        "bytes": len(archive_payload),
        "sha256": actual,
        "checksum_sha256": _sha256_bytes(checksum_payload),
        "checksum_verified": True,
        "cached": archive_cached and checksum_cached,
        **inspection,
    }


def audit_liquidity_candidate(
    summary: dict[str, object],
    archive_records: list[dict[str, object]],
    expected_dates: set[str],
    contract: dict,
    archive_rejections: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    archive_rejections = archive_rejections or []
    observations: list[tuple[str, float]] = []
    for record in archive_records:
        observations.extend(zip(record["dates"], record["quote_volumes"], strict=True))
    in_window = [(date, value) for date, value in observations if date in expected_dates]
    observed_dates = [date for date, _ in in_window]
    duplicates = len(observed_dates) - len(set(observed_dates))
    unique_values = {date: value for date, value in in_window}
    coverage = len(unique_values) / len(expected_dates)
    nonzero_fraction = (
        sum(value > 0 for value in unique_values.values()) / len(expected_dates)
    )
    median_quote_volume = (
        float(pd.Series(list(unique_values.values()), dtype="float64").median())
        if unique_values
        else None
    )
    first_date = min(unique_values) if unique_values else None
    last_date = max(unique_values) if unique_values else None
    checks = {
        "first_observation_on_or_before_cutoff": first_date is not None
        and first_date <= str(contract["listed_on_or_before"]),
        "daily_coverage_passes": coverage
        >= float(contract["minimum_daily_coverage"]),
        "nonzero_quote_volume_fraction_passes": nonzero_fraction
        >= float(contract["minimum_nonzero_quote_volume_fraction"]),
        "median_quote_volume_is_finite": median_quote_volume is not None
        and math.isfinite(median_quote_volume),
        "no_duplicate_dates": duplicates == 0,
        "all_used_archive_checksums_verified": bool(archive_records)
        and all(record["checksum_verified"] for record in archive_records),
        "all_used_archive_schemas_valid": bool(archive_records)
        and all(record["schema_valid"] for record in archive_records),
        "only_allowed_columns_read": bool(archive_records)
        and all(
            record["selection_columns_read"] == ["date", "quote_volume"]
            for record in archive_records
        ),
        "only_observation_window_dates_used": all(
            date in expected_dates for date in observed_dates
        ),
    }
    return {
        "symbol": summary["symbol"],
        "base": summary["base"],
        "expected_days": len(expected_dates),
        "observed_days": len(unique_values),
        "coverage": coverage,
        "nonzero_quote_volume_days": sum(
            value > 0 for value in unique_values.values()
        ),
        "nonzero_quote_volume_fraction": nonzero_fraction,
        "median_daily_quote_volume_usdt": median_quote_volume,
        "first_date": first_date,
        "last_date": last_date,
        "duplicate_dates": duplicates,
        "verified_archive_count": len(archive_records),
        "rejected_archive_count": len(archive_rejections),
        "archive_rejections": archive_rejections,
        "timestamp_units": sorted({
            unit for record in archive_records for unit in record["timestamp_units"]
        }),
        "eligible": all(checks.values()),
        "checks": checks,
    }


def _manifest_record(record: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"dates", "quote_volumes", "cache_path"}
    }


def build_training_universe_result(
    amendment: dict,
    discovery: dict[str, object],
    candidate_audits: list[dict[str, object]],
    archive_records: list[dict[str, object]],
    config: dict,
    archive_rejections: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    archive_rejections = archive_rejections or []
    inventory = config["training_universe_inventory"]
    blueprint = amendment["blueprint"]
    contract = blueprint["training_universe"]
    eligible = sorted(
        (row for row in candidate_audits if row["eligible"]),
        key=lambda row: (
            -float(row["median_daily_quote_volume_usdt"]),
            str(row["symbol"]),
        ),
    )
    selected_count = int(contract["selected_asset_count"])
    selected = eligible[:selected_count]
    selected_symbols = [str(row["symbol"]) for row in selected]
    selected_set = set(selected_symbols)
    folds = build_asset_folds(selected_symbols, int(contract["asset_fold_count"]))
    selected_records = [
        _manifest_record(row)
        for row in archive_records
        if row["symbol"] in selected_set
    ]
    selected_records.sort(key=lambda row: (row["symbol"], row["month"]))
    target_symbols = set(blueprint["target_contract"]["symbols"])
    excluded_bases = set().union(
        *(set(values) for values in contract["exclusions"].values())
    )
    selected_bases = {symbol.removesuffix(contract["quote_asset"]) for symbol in selected_symbols}
    test_sets = [set(fold["test_symbols"]) for fold in folds["folds"]]
    all_months = {
        str(record["month"])
        for record in archive_records
    }
    observation_start, observation_end = contract["observation_window"]
    allowed_months = set(_monthly_periods(observation_start, observation_end))
    selection_table = [
        {
            "rank": rank,
            "symbol": row["symbol"],
            "coverage": row["coverage"],
            "nonzero_quote_volume_fraction": row[
                "nonzero_quote_volume_fraction"
            ],
            "median_daily_quote_volume_usdt": row[
                "median_daily_quote_volume_usdt"
            ],
        }
        for rank, row in enumerate(selected, start=1)
    ]
    checks = {
        "v29_blueprint_hash_matches": amendment["blueprint_sha256"]
        == inventory["expected_v29_blueprint_sha256"],
        "v29_authorizes_only_v30": amendment["decision"]
        == "authorize_v30_training_universe_liquidity_inventory_only",
        "exactly_30_assets_selected": len(selected_symbols) == selected_count == 30,
        "selection_is_frozen_liquidity_ranking": selected_symbols
        == [row["symbol"] for row in eligible[:selected_count]],
        "enough_eligible_assets": len(eligible) >= selected_count,
        "all_selected_coverage_passes": bool(selected)
        and all(
            row["coverage"] >= float(contract["minimum_daily_coverage"])
            for row in selected
        ),
        "all_selected_nonzero_volume_passes": bool(selected)
        and all(
            row["nonzero_quote_volume_fraction"]
            >= float(contract["minimum_nonzero_quote_volume_fraction"])
            for row in selected
        ),
        "no_targets_selected": not target_symbols.intersection(selected_set),
        "no_excluded_bases_selected": not excluded_bases.intersection(selected_bases),
        "three_equal_disjoint_folds": len(test_sets) == 3
        and all(len(group) == 10 for group in test_sets)
        and not any(
            test_sets[left].intersection(test_sets[right])
            for left in range(len(test_sets))
            for right in range(left + 1, len(test_sets))
        )
        and set.union(*test_sets) == selected_set,
        "all_selected_archives_checksum_verified": bool(selected_records)
        and all(record["checksum_verified"] for record in selected_records),
        "all_candidate_archives_checksum_verified": bool(archive_records)
        and all(record["checksum_verified"] for record in archive_records),
        "all_archive_jobs_accounted_for": len(archive_records)
        + len(archive_rejections)
        == int(discovery["expected_archive_job_count"]),
        "selection_uses_only_2021_2023": all_months.issubset(allowed_months),
        "selection_columns_are_frozen": list(SELECTION_COLUMNS)
        == contract["ranking"]["input_columns_allowed"],
        "future_availability_not_used": not contract["future_window_usage"][
            "full_future_availability_used_for_selection"
        ]
        and not contract["future_window_usage"][
            "validation_calibration_confirmation_used_for_selection"
        ],
        "source_index_is_hashed": bool(discovery["root_pages"])
        and bool(discovery["symbol_listing_pages"])
        and bool(discovery["selection_listing"])
        and all(
            row["selection_object_sha256"]
            for row in discovery["selection_listing"]
        ),
        "model_training_count_is_zero": True,
        "label_read_count_is_zero": True,
        "return_computation_count_is_zero": True,
        "performance_metric_count_is_zero": True,
        "pnl_evaluation_count_is_zero": True,
        "target_asset_load_count_is_zero": True,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise RuntimeError(f"V30 training-universe audit failed: {checks}")
    return {
        "version": "v30",
        "method": "performance_blind_training_window_liquidity_inventory",
        "decision": "authorize_v31_selected_universe_manifest_refresh_only",
        "blueprint_sha256": amendment["blueprint_sha256"],
        "universe": {
            "selected_symbols": selected_symbols,
            "selected_count": len(selected_symbols),
            "eligible_count": len(eligible),
            "selection_table": selection_table,
            "observation_window": contract["observation_window"],
            "selection_columns": list(SELECTION_COLUMNS),
            "minimum_daily_coverage": contract["minimum_daily_coverage"],
            "minimum_nonzero_quote_volume_fraction": contract[
                "minimum_nonzero_quote_volume_fraction"
            ],
            "nonzero_fraction_denominator": "expected_calendar_days",
            "median_denominator": "observed_days_only_no_imputation",
            "ranking": contract["ranking"],
        },
        "asset_folds": folds,
        "candidate_audits": candidate_audits,
        "archive_manifest": selected_records,
        "archive_rejections": archive_rejections,
        "discovery": discovery,
        "tested": {
            "universe_selected": True,
            "selection_input_columns": list(SELECTION_COLUMNS),
            "selection_observation_window": contract["observation_window"],
            "label_columns_read": False,
            "returns_computed": False,
            "model_trained": False,
            "performance_metrics_computed": False,
            "pnl_computed": False,
            "target_assets_loaded": False,
            "improvement_status": "unknown_not_evaluated",
            "drawdown_status": "unknown_not_evaluated",
        },
        "audit": {"passed": True, "checks": checks},
    }


def _report(result: dict[str, object]) -> str:
    universe = result["universe"]
    folds = result["asset_folds"]["folds"]
    lines = [
        "# TLM v30 Training-Universe Liquidity Inventory",
        "",
        "## Decision",
        "",
        "**30-ASSET MULTI-ASSET TRAINING UNIVERSE SELECTED; TRAINING REMAINS BLOCKED.**",
        "",
        f"Eligible assets: **{universe['eligible_count']}**",
        f"Selected assets: **{universe['selected_count']}**",
        f"Observation window: **{universe['observation_window'][0]} through {universe['observation_window'][1]}**",
        "",
        "Selection used only symbol, UTC date, and daily USDT quote volume. No label, return, model, performance metric, PnL, BTC, ETH, or SOL observation was loaded.",
        "",
        "## Selected universe",
        "",
        "| Rank | Symbol | Coverage | Nonzero volume | Median daily USDT quote volume |",
        "|---:|---|---:|---:|---:|",
    ]
    for row in universe["selection_table"]:
        lines.append(
            f"| {row['rank']} | {row['symbol']} | {row['coverage']:.2%} | "
            f"{row['nonzero_quote_volume_fraction']:.2%} | "
            f"{row['median_daily_quote_volume_usdt']:,.2f} |"
        )
    lines.extend(["", "## Asset-disjoint folds", ""])
    for fold in folds:
        lines.append(
            f"- Fold {fold['fold']}: {', '.join(fold['test_symbols'])}"
        )
    lines.extend([
        "",
        "## Scientific boundary",
        "",
        "The ranking did not inspect 2024 validation, 2025 calibration, 2026 confirmation, future availability, forward labels, or any portfolio result. Missing observations were not imputed.",
        "",
        "## Next action",
        "",
        "V31 may freeze a selected-universe source manifest for these exact 30 symbols. It must reverify checksums, gaps, timestamps, and archive hashes without building labels, training a model, or evaluating performance.",
        "",
    ])
    return "\n".join(lines)


def run_training_universe_inventory(
    config: dict,
    force: bool = False,
    fetch_bytes: FetchBytes = _http_get,
) -> dict[str, object]:
    inventory = config["training_universe_inventory"]
    root = Path(inventory["project_root"]).resolve()
    amendment_path = root / inventory["v29_amendment_path"]
    audit_path = root / inventory["v29_audit_path"]
    amendment = _load_json(amendment_path)
    v29_audit = _load_json(audit_path)
    if not v29_audit.get("passed"):
        raise RuntimeError("V29 audit does not pass")
    if _sha256_file(amendment_path) != inventory["expected_v29_amendment_sha256"]:
        raise RuntimeError("V29 amendment hash drift")
    if _sha256_file(audit_path) != inventory["expected_v29_audit_sha256"]:
        raise RuntimeError("V29 audit hash drift")

    blueprint = amendment["blueprint"]
    contract = blueprint["training_universe"]
    interval = blueprint["data_contract"]["frequency"]
    observation_start, observation_end = contract["observation_window"]
    months = _monthly_periods(observation_start, observation_end)
    expected_dates = {
        date.date().isoformat()
        for date in pd.date_range(observation_start, observation_end, freq="D")
    }
    endpoint = inventory["s3_endpoint"]
    root_prefix = inventory["root_prefix"]
    timeout = float(inventory["timeout_seconds"])
    quote_asset = contract["quote_asset"]

    prefixes, root_pages = list_common_prefixes(
        endpoint, root_prefix, timeout, fetch_bytes
    )
    all_symbols = sorted(
        _symbol_from_prefix(prefix, root_prefix) for prefix in prefixes
    )
    exclusion_counts: dict[str, int] = {}
    candidate_symbols: list[str] = []
    for symbol in all_symbols:
        reason = scope_exclusion_reason(symbol, quote_asset, contract)
        if reason is None:
            candidate_symbols.append(symbol)
        else:
            exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1

    symbol_summaries: list[dict[str, object]] = []
    symbol_listing_pages: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=int(inventory["listing_workers"])) as executor:
        futures = {
            executor.submit(
                list_s3_objects,
                endpoint,
                f"{root_prefix}{symbol}/{interval}/",
                timeout,
                fetch_bytes,
            ): symbol
            for symbol in candidate_symbols
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            symbol = futures[future]
            objects, pages = future.result()
            symbol_summaries.append(
                summarize_selection_listing(symbol, objects, months, interval)
            )
            for page in pages:
                page["symbol"] = symbol
            symbol_listing_pages.extend(pages)
            if completed % 100 == 0 or completed == len(futures):
                print(
                    f"v30 training-universe listings: {completed}/{len(futures)}",
                    flush=True,
                )
    symbol_summaries.sort(key=lambda row: row["symbol"])
    prefiltered = [
        row
        for row in symbol_summaries
        if row["complete_month_fraction"] == 1.0
        and months[0] in row["complete_months"]
    ]

    raw_dir = root / inventory["raw_dir"]
    jobs = [
        (summary["symbol"], month)
        for summary in prefiltered
        for month in summary["complete_months"]
    ]
    archive_records: list[dict[str, object]] = []
    archive_rejections: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=int(inventory["archive_workers"])) as executor:
        futures = {
            executor.submit(
                verify_selection_archive,
                symbol,
                month,
                interval,
                inventory["download_base_url"],
                raw_dir,
                timeout,
                force,
                fetch_bytes,
            ): (symbol, month)
            for symbol, month in jobs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            symbol, month = futures[future]
            try:
                archive_records.append(future.result())
            except Exception as error:
                archive_rejections.append({
                    "symbol": symbol,
                    "month": month,
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
                print(f"v30 archive rejected: {symbol} {month}: {error}", flush=True)
            if completed % 250 == 0 or completed == len(futures):
                print(
                    f"v30 training-universe archives: {completed}/{len(futures)}",
                    flush=True,
                )
    archive_records.sort(key=lambda row: (row["symbol"], row["month"]))
    archive_rejections.sort(key=lambda row: (row["symbol"], row["month"]))
    candidate_audits = []
    for summary in prefiltered:
        symbol = summary["symbol"]
        candidate_audits.append(
            audit_liquidity_candidate(
                summary,
                [row for row in archive_records if row["symbol"] == symbol],
                expected_dates,
                contract,
                [row for row in archive_rejections if row["symbol"] == symbol],
            )
        )
    candidate_audits.sort(key=lambda row: row["symbol"])

    discovery = {
        "root_prefix": root_prefix,
        "all_symbol_prefix_count": len(all_symbols),
        "candidate_symbol_count": len(candidate_symbols),
        "prefiltered_symbol_count": len(prefiltered),
        "audited_candidate_count": len(candidate_audits),
        "expected_archive_job_count": len(jobs),
        "observation_months": months,
        "exclusion_counts": exclusion_counts,
        "root_pages": root_pages,
        "symbol_listing_pages": sorted(
            symbol_listing_pages,
            key=lambda row: (str(row.get("symbol")), str(row["url"])),
        ),
        "selection_listing": [
            {
                "symbol": row["symbol"],
                "complete_month_fraction": row["complete_month_fraction"],
                "selection_object_count": row["selection_object_count"],
                "selection_object_sha256": row["selection_object_sha256"],
            }
            for row in symbol_summaries
        ],
    }
    result = build_training_universe_result(
        amendment,
        discovery,
        candidate_audits,
        archive_records,
        config,
        archive_rejections,
    )
    result["source_hashes"] = {
        str(amendment_path.relative_to(root)): _sha256_file(amendment_path),
        str(audit_path.relative_to(root)): _sha256_file(audit_path),
    }

    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "selected_manifest.jsonl"
    manifest_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in result["archive_manifest"])
        + "\n",
        encoding="utf-8",
    )
    candidate_manifest_path = output / "candidate_manifest.jsonl"
    candidate_manifest = [_manifest_record(row) for row in archive_records]
    candidate_manifest_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in candidate_manifest)
        + "\n",
        encoding="utf-8",
    )
    files = {
        "universe.json": result["universe"],
        "asset_folds.json": result["asset_folds"],
        "candidate_audits.json": result["candidate_audits"],
        "archive_rejections.json": result["archive_rejections"],
        "source_index.json": result["discovery"],
        "audit.json": result["audit"],
    }
    for name, value in files.items():
        (output / name).write_text(
            json.dumps(value, indent=2, sort_keys=True), encoding="utf-8"
        )
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    summary = {
        key: value
        for key, value in result.items()
        if key not in {
            "archive_manifest",
            "archive_rejections",
            "candidate_audits",
            "discovery",
        }
    }
    summary["artifact_references"] = {
        "selected_manifest": {
            "path": "selected_manifest.jsonl",
            "records": len(result["archive_manifest"]),
            "sha256": _sha256_file(manifest_path),
        },
        "candidate_manifest": {
            "path": "candidate_manifest.jsonl",
            "records": len(candidate_manifest),
            "sha256": _sha256_file(candidate_manifest_path),
        },
        **{
            name.removesuffix(".json"): {
                "path": name,
                "sha256": _sha256_file(output / name),
            }
            for name in files
        },
    }
    (output / "inventory.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result
