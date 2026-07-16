from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import time
from typing import Callable
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd
import yaml

from .data import _verified_ssl_context
from .derivatives_data import parse_checksum


FetchBytes = Callable[[str, float], bytes]


@dataclass(frozen=True)
class S3Object:
    key: str
    size: int
    etag: str
    last_modified: str


KLINE_COLUMNS = (
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
)

KNOWN_FIAT_BASES = frozenset({
    "AUD", "BRL", "EUR", "GBP", "NGN", "RUB", "TRY", "UAH", "ZAR"
})


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _http_get(url: str, timeout: float, attempts: int = 3) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "tlm-non-target-inventory/0.1"}
    )
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(
                request, timeout=timeout, context=_verified_ssl_context()
            ) as response:
                return response.read()
        except Exception as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(0.25 * (2**attempt))
    assert last_error is not None
    raise last_error


def _xml_text(node: ET.Element, name: str, default: str = "") -> str:
    child = node.find(f"{{*}}{name}")
    return default if child is None or child.text is None else child.text


def _listing_url(
    endpoint: str,
    prefix: str,
    marker: str | None = None,
    delimiter: str | None = None,
) -> str:
    query = {"prefix": prefix, "max-keys": "1000"}
    if marker:
        query["marker"] = marker
    if delimiter:
        query["delimiter"] = delimiter
    return f"{endpoint}?{urllib.parse.urlencode(query)}"


def list_common_prefixes(
    endpoint: str,
    prefix: str,
    timeout: float,
    fetch_bytes: FetchBytes = _http_get,
) -> tuple[list[str], list[dict[str, object]]]:
    marker: str | None = None
    prefixes: list[str] = []
    pages: list[dict[str, object]] = []
    while True:
        url = _listing_url(endpoint, prefix, marker=marker, delimiter="/")
        payload = fetch_bytes(url, timeout)
        root = ET.fromstring(payload)
        page_prefixes = [
            _xml_text(node, "Prefix")
            for node in root.findall("{*}CommonPrefixes")
        ]
        prefixes.extend(value for value in page_prefixes if value)
        pages.append({
            "url": url,
            "sha256": _sha256_bytes(payload),
            "bytes": len(payload),
            "prefix_count": len(page_prefixes),
        })
        truncated = _xml_text(root, "IsTruncated").lower() == "true"
        if not truncated:
            break
        next_marker = _xml_text(root, "NextMarker")
        marker = next_marker or (page_prefixes[-1] if page_prefixes else None)
        if not marker:
            raise ValueError("Truncated S3 prefix listing has no continuation marker")
    return sorted(set(prefixes)), pages


def list_s3_objects(
    endpoint: str,
    prefix: str,
    timeout: float,
    fetch_bytes: FetchBytes = _http_get,
) -> tuple[list[S3Object], list[dict[str, object]]]:
    marker: str | None = None
    objects: list[S3Object] = []
    pages: list[dict[str, object]] = []
    while True:
        url = _listing_url(endpoint, prefix, marker=marker)
        payload = fetch_bytes(url, timeout)
        root = ET.fromstring(payload)
        page_objects = [
            S3Object(
                key=_xml_text(node, "Key"),
                size=int(_xml_text(node, "Size", "0")),
                etag=_xml_text(node, "ETag").strip('"'),
                last_modified=_xml_text(node, "LastModified"),
            )
            for node in root.findall("{*}Contents")
            if _xml_text(node, "Key")
        ]
        objects.extend(page_objects)
        pages.append({
            "url": url,
            "sha256": _sha256_bytes(payload),
            "bytes": len(payload),
            "object_count": len(page_objects),
        })
        truncated = _xml_text(root, "IsTruncated").lower() == "true"
        if not truncated:
            break
        marker = page_objects[-1].key if page_objects else None
        if not marker:
            raise ValueError("Truncated S3 object listing has no continuation marker")
    unique = {item.key: item for item in objects}
    return sorted(unique.values(), key=lambda item: item.key), pages


def _symbol_from_prefix(prefix: str, root_prefix: str) -> str:
    if not prefix.startswith(root_prefix):
        raise ValueError(f"Unexpected symbol prefix: {prefix}")
    return prefix[len(root_prefix):].strip("/")


def exclusion_reason(
    symbol: str,
    quote_asset: str,
    universe: dict,
) -> str | None:
    if not symbol.endswith(quote_asset):
        return "quote_asset"
    base = symbol[: -len(quote_asset)]
    if not base:
        return "empty_base"
    if base in set(universe["excluded_bases"]):
        return "target_base"
    if base in set(universe["target_proxy_bases"]):
        return "target_proxy"
    if base in set(universe["excluded_stablecoin_bases"]):
        return "stablecoin_base"
    if any(base.endswith(suffix) for suffix in universe["excluded_token_suffixes"]):
        return "excluded_suffix"
    return None


def _monthly_periods(start: str, end: str) -> list[str]:
    return [str(period) for period in pd.period_range(start, end, freq="M")]


def _expected_archive_names(symbol: str, interval: str, months: list[str]) -> dict[str, str]:
    return {
        month: f"{symbol}-{interval}-{month}.zip" for month in months
    }


def summarize_symbol_listing(
    symbol: str,
    objects: list[S3Object],
    months: list[str],
    interval: str,
) -> dict[str, object]:
    names = _expected_archive_names(symbol, interval, months)
    by_name = {Path(item.key).name: item for item in objects}
    archive_months = [
        month for month, name in names.items() if name in by_name
    ]
    checksum_months = [
        month for month, name in names.items() if f"{name}.CHECKSUM" in by_name
    ]
    complete_months = sorted(set(archive_months).intersection(checksum_months))
    return {
        "symbol": symbol,
        "base": symbol[:-4],
        "archive_months": archive_months,
        "checksum_months": checksum_months,
        "complete_months": complete_months,
        "complete_month_fraction": len(complete_months) / len(months),
        "objects": {
            item.key: {
                "size": item.size,
                "etag": item.etag,
                "last_modified": item.last_modified,
            }
            for item in objects
        },
    }


def _timestamp_to_datetime(value: str) -> tuple[datetime, str]:
    integer = int(value)
    unit = "us" if abs(integer) >= 100_000_000_000_000 else "ms"
    divisor = 1_000_000 if unit == "us" else 1_000
    return datetime.fromtimestamp(integer / divisor, tz=timezone.utc), unit


def inspect_kline_zip(payload: bytes) -> dict[str, object]:
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_names = [name for name in archive.namelist() if name.endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected one CSV in kline archive, found {csv_names}")
        with archive.open(csv_names[0]) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8")
            rows = list(csv.reader(text))
    if not rows:
        raise ValueError("Kline archive is empty")
    if rows[0] and rows[0][0].strip().lower().replace(" ", "_") == "open_time":
        rows = rows[1:]
    if not rows:
        raise ValueError("Kline archive contains only a header")
    dates: list[str] = []
    units: set[str] = set()
    for row in rows:
        if len(row) != len(KLINE_COLUMNS):
            raise ValueError(
                f"Kline row has {len(row)} columns, expected {len(KLINE_COLUMNS)}"
            )
        opened, open_unit = _timestamp_to_datetime(row[0])
        closed, close_unit = _timestamp_to_datetime(row[6])
        if opened.hour or opened.minute or opened.second or opened.microsecond:
            raise ValueError(f"Daily kline is not aligned to UTC midnight: {opened}")
        if closed <= opened:
            raise ValueError(f"Kline close is not after open: {opened} -> {closed}")
        numeric = [float(row[index]) for index in range(1, 6)]
        numeric.extend(float(row[index]) for index in range(7, 12))
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("Kline archive contains non-finite numeric values")
        dates.append(opened.date().isoformat())
        units.update((open_unit, close_unit))
    if len(dates) != len(set(dates)):
        raise ValueError("Kline archive contains duplicate UTC dates")
    if dates != sorted(dates):
        raise ValueError("Kline archive dates are not sorted")
    return {
        "row_count": len(dates),
        "dates": dates,
        "first_date": dates[0],
        "last_date": dates[-1],
        "timestamp_units": sorted(units),
        "column_count": len(KLINE_COLUMNS),
        "schema_valid": True,
    }


def _download_cached(
    url: str,
    cache_path: Path,
    timeout: float,
    force: bool,
    fetch_bytes: FetchBytes,
) -> tuple[bytes, bool]:
    if cache_path.is_file() and not force:
        return cache_path.read_bytes(), True
    payload = fetch_bytes(url, timeout)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(cache_path)
    return payload, False


def verify_month_archive(
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
    inspection = inspect_kline_zip(archive_payload)
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


def audit_candidate(
    summary: dict[str, object],
    archive_records: list[dict[str, object]],
    expected_dates: set[str],
    listed_cutoff: str,
    minimum_coverage: float,
    archive_rejections: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    archive_rejections = archive_rejections or []
    observed_dates = [
        date for record in archive_records for date in record["dates"]
    ]
    observed_unique = set(observed_dates).intersection(expected_dates)
    duplicates = len(observed_dates) - len(set(observed_dates))
    coverage = len(observed_unique) / len(expected_dates)
    first_date = min(observed_unique) if observed_unique else None
    last_date = max(observed_unique) if observed_unique else None
    checks = {
        "first_observation_on_or_before_listing_cutoff": first_date is not None
        and first_date <= listed_cutoff,
        "daily_coverage_passes": coverage >= minimum_coverage,
        "no_duplicate_dates": duplicates == 0,
        "all_used_archive_checksums_verified": bool(archive_records)
        and all(record["checksum_verified"] for record in archive_records),
        "all_used_archive_schemas_valid": bool(archive_records)
        and all(record["schema_valid"] for record in archive_records),
    }
    return {
        "symbol": summary["symbol"],
        "base": summary["base"],
        "expected_days": len(expected_dates),
        "observed_days": len(observed_unique),
        "coverage": coverage,
        "first_date": first_date,
        "last_date": last_date,
        "duplicate_dates": duplicates,
        "verified_archive_count": len(archive_records),
        "rejected_archive_count": len(archive_rejections),
        "archive_rejections": archive_rejections,
        "timestamp_units": sorted({
            unit
            for record in archive_records
            for unit in record["timestamp_units"]
        }),
        "eligible": all(checks.values()),
        "checks": checks,
    }


def build_inventory_result(
    blueprint_result: dict,
    discovery: dict[str, object],
    candidate_audits: list[dict[str, object]],
    archive_records: list[dict[str, object]],
    config: dict,
    archive_rejections: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    archive_rejections = archive_rejections or []
    inventory = config["non_target_inventory"]
    universe = blueprint_result["blueprint"]["development_universe"]
    minimum_assets = int(universe["minimum_assets"])
    maximum_assets = int(universe["maximum_assets"])
    eligible = sorted(
        (row for row in candidate_audits if row["eligible"]),
        key=lambda row: row["symbol"],
    )
    selected = eligible[:maximum_assets]
    selected_symbols = [row["symbol"] for row in selected]
    selected_bases = [row["base"] for row in selected]
    selected_fiat_bases = sorted(set(selected_bases).intersection(KNOWN_FIAT_BASES))
    target_symbols = set(blueprint_result["blueprint"]["target_symbols"])
    forbidden_bases = (
        set(universe["excluded_bases"])
        | set(universe["target_proxy_bases"])
        | set(universe["excluded_stablecoin_bases"])
    )
    selected_records_with_dates = [
        row for row in archive_records if row["symbol"] in set(selected_symbols)
    ]
    selected_records = [
        {
            key: value
            for key, value in row.items()
            if key not in {"dates", "cache_path"}
        }
        for row in selected_records_with_dates
    ]
    selected_rejections = [
        row for row in archive_rejections if row["symbol"] in set(selected_symbols)
    ]
    source_page_hashes = [
        page["sha256"]
        for page in discovery["root_pages"]
        + discovery["symbol_listing_pages"]
    ]
    checks = {
        "v26_blueprint_hash_matches": blueprint_result["blueprint_sha256"]
        == inventory["expected_blueprint_sha256"],
        "v26_authorizes_only_v27_inventory": blueprint_result["decision"]
        == "authorize_v27_non_target_universe_data_audit_only",
        "selected_count_within_frozen_bounds": minimum_assets
        <= len(selected)
        <= maximum_assets,
        "selection_is_lexical_first_eligible": selected_symbols
        == [row["symbol"] for row in eligible[:maximum_assets]],
        "no_target_symbols_selected": not target_symbols.intersection(
            selected_symbols
        ),
        "no_forbidden_bases_selected": not forbidden_bases.intersection(
            selected_bases
        ),
        "all_selected_coverage_passes": bool(selected)
        and all(
            row["coverage"] >= float(universe["minimum_daily_coverage"])
            for row in selected
        ),
        "all_selected_archives_checksum_verified": bool(selected_records)
        and all(row["checksum_verified"] for row in selected_records),
        "source_index_is_hashed": bool(source_page_hashes),
        "model_training_count_is_zero": True,
        "target_prediction_count_is_zero": True,
        "pnl_evaluation_count_is_zero": True,
    }
    decision = (
        "authorize_v28_non_target_dataset_build_only"
        if all(checks.values())
        else "reject_non_target_universe"
    )
    if not all(checks.values()):
        raise RuntimeError(f"V27 inventory audit failed: {checks}")
    return {
        "version": "v27",
        "method": "policy_free_official_archive_universe_inventory",
        "decision": decision,
        "tested": {
            "model_trained": False,
            "returns_computed": False,
            "pnl_computed": False,
            "target_assets_loaded": False,
            "improvement_status": "unknown_not_evaluated",
            "drawdown_status": "unknown_not_evaluated",
        },
        "blueprint_sha256": blueprint_result["blueprint_sha256"],
        "universe": {
            "selected_symbols": selected_symbols,
            "selected_bases": selected_bases,
            "selected_count": len(selected),
            "minimum_assets": minimum_assets,
            "maximum_assets": maximum_assets,
            "selection_rule": universe["selection_rule"],
            "minimum_daily_coverage": universe["minimum_daily_coverage"],
            "scope_observations": ([{
                "severity": "warning",
                "code": "fiat_base_selected_by_frozen_lexical_contract",
                "bases": selected_fiat_bases,
                "resolution": (
                    "Preserve v26 selection in v27; require an explicit versioned "
                    "scope decision before model training."
                ),
            }] if selected_fiat_bases else []),
        },
        "selected_asset_audits": selected,
        "candidate_audits": candidate_audits,
        "archive_manifest": selected_records,
        "archive_rejections": archive_rejections,
        "selected_archive_rejections": selected_rejections,
        "discovery": discovery,
        "audit": {"passed": True, "checks": checks},
    }


def _report(result: dict) -> str:
    universe = result["universe"]
    rows = result["selected_asset_audits"]
    coverage = [row["coverage"] for row in rows]
    archive_count = len(result["archive_manifest"])
    rejections = result["selected_archive_rejections"]
    scope_observations = universe["scope_observations"]
    lines = [
        "# TLM v27 Non-Target Universe Audit",
        "",
        "## Decision",
        "",
        "**NON-TARGET UNIVERSE ACCEPTED FOR A DATASET BUILD ONLY.**",
        "",
        f"Selected assets: **{universe['selected_count']}**",
        f"Verified monthly archives: **{archive_count}**",
        f"Coverage range: **{min(coverage):.2%}–{max(coverage):.2%}**",
        "",
        "No model was trained, no returns or PnL were computed, and no BTC/ETH/SOL or registered target proxy data was loaded.",
        "",
        "## Frozen universe",
        "",
        ", ".join(universe["selected_symbols"]),
        "",
        "## Coverage",
        "",
        "| Symbol | Days | Coverage | First | Last | Units |",
        "|---|---:|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['symbol']} | {row['observed_days']} | "
            f"{row['coverage']:.2%} | {row['first_date']} | "
            f"{row['last_date']} | {','.join(row['timestamp_units'])} |"
        )
    if scope_observations:
        lines.extend([
            "",
            "## Scope observation",
            "",
            "`EURUSDT` is retained because the frozen v26 lexical contract did not exclude fiat bases. No post-inventory exclusion was introduced. Before any model training, this crypto-domain thesis mismatch must be explicitly accepted or corrected in a versioned, performance-blind amendment.",
        ])
    lines.extend([
        "",
        "## Source contract",
        "",
        "Every selected monthly ZIP used in coverage was matched to its published checksum and locally hashed. The parser accepts the official millisecond-to-microsecond spot timestamp transition in 2025 and requires UTC-midnight daily candles.",
        "",
    ])
    if rejections:
        lines.extend([
            "### Rejected source archives",
            "",
            "Rejected archives are treated as missing observations; they are never repaired or silently included. An asset can remain eligible only if its accepted observations still clear the frozen 98% coverage gate.",
            "",
            "| Symbol | Month | Error |",
            "|---|---|---|",
        ])
        for row in rejections:
            lines.append(
                f"| {row['symbol']} | {row['month']} | {row['error']} |"
            )
        lines.append("")
    lines.extend([
        "Binance documents that archived files may later be corrected. This manifest is therefore a reproducible snapshot of the current archive, not a point-in-time vintage guarantee.",
        "",
        "## Next action",
        "",
        "V28 may materialize causal non-target features/labels from this exact universe and manifest. It must remain data-only: no model, portfolio, target asset, return metric, or PnL evaluation. The EUR scope observation must be resolved before training.",
        "",
    ])
    return "\n".join(lines)


def run_non_target_inventory(
    config: dict,
    force: bool = False,
    fetch_bytes: FetchBytes = _http_get,
) -> dict[str, object]:
    inventory = config["non_target_inventory"]
    root = Path(inventory["project_root"]).resolve()
    v26_path = root / inventory["v26_specification_path"]
    v26_audit_path = root / inventory["v26_audit_path"]
    blueprint_result = json.loads(v26_path.read_text(encoding="utf-8"))
    v26_audit = json.loads(v26_audit_path.read_text(encoding="utf-8"))
    if not v26_audit.get("passed"):
        raise RuntimeError("V26 audit does not pass")
    universe = blueprint_result["blueprint"]["development_universe"]
    data_contract = blueprint_result["blueprint"]["data_contract"]
    endpoint = inventory["s3_endpoint"]
    root_prefix = inventory["root_prefix"]
    interval = data_contract["frequency"]
    timeout = float(inventory["timeout_seconds"])
    quote_asset = universe["quote_asset"]
    months = _monthly_periods(
        data_contract["development_start"], data_contract["development_cutoff"]
    )
    expected_dates = {
        date.date().isoformat()
        for date in pd.date_range(
            data_contract["development_start"],
            data_contract["development_cutoff"],
            freq="D",
        )
    }

    prefixes, root_pages = list_common_prefixes(
        endpoint, root_prefix, timeout, fetch_bytes
    )
    all_symbols = sorted(
        _symbol_from_prefix(prefix, root_prefix) for prefix in prefixes
    )
    exclusion_counts: dict[str, int] = {}
    candidate_symbols: list[str] = []
    for symbol in all_symbols:
        reason = exclusion_reason(symbol, quote_asset, universe)
        if reason is None:
            candidate_symbols.append(symbol)
        else:
            exclusion_counts[reason] = exclusion_counts.get(reason, 0) + 1

    symbol_summaries: list[dict[str, object]] = []
    symbol_listing_pages: list[dict[str, object]] = []
    listing_workers = int(inventory["listing_workers"])
    with ThreadPoolExecutor(max_workers=listing_workers) as executor:
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
                summarize_symbol_listing(symbol, objects, months, interval)
            )
            for page in pages:
                page["symbol"] = symbol
            symbol_listing_pages.extend(pages)
            if completed % 100 == 0 or completed == len(futures):
                print(
                    f"non-target listings: {completed}/{len(futures)}",
                    flush=True,
                )
    symbol_summaries.sort(key=lambda row: row["symbol"])
    minimum_month_fraction = float(inventory["minimum_month_prefilter"])
    prefiltered = [
        row
        for row in symbol_summaries
        if row["complete_month_fraction"] >= minimum_month_fraction
        and months[0] in row["complete_months"]
    ]

    candidate_audits: list[dict[str, object]] = []
    archive_records: list[dict[str, object]] = []
    archive_rejections: list[dict[str, str]] = []
    raw_dir = root / inventory["raw_dir"]
    maximum_assets = int(universe["maximum_assets"])
    batch_size = int(inventory["candidate_batch_size"])
    archive_workers = int(inventory["archive_workers"])
    offset = 0
    while len([row for row in candidate_audits if row["eligible"]]) < maximum_assets:
        batch = prefiltered[offset: offset + batch_size]
        if not batch:
            break
        offset += len(batch)
        jobs: list[tuple[str, str]] = [
            (summary["symbol"], month)
            for summary in batch
            for month in summary["complete_months"]
        ]
        batch_records: list[dict[str, object]] = []
        with ThreadPoolExecutor(max_workers=archive_workers) as executor:
            futures = {
                executor.submit(
                    verify_month_archive,
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
                    batch_records.append(future.result())
                except Exception as error:
                    archive_rejections.append({
                        "symbol": symbol,
                        "month": month,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    })
                    print(
                        f"archive rejected: {symbol} {month}: {error}",
                        flush=True,
                    )
                if completed % 250 == 0 or completed == len(futures):
                    print(
                        f"non-target archives batch: {completed}/{len(futures)}",
                        flush=True,
                    )
        batch_records.sort(key=lambda row: (row["symbol"], row["month"]))
        archive_records.extend(batch_records)
        for summary in batch:
            records = [
                row
                for row in batch_records
                if row["symbol"] == summary["symbol"]
            ]
            rejections = [
                row
                for row in archive_rejections
                if row["symbol"] == summary["symbol"]
            ]
            candidate_audits.append(
                audit_candidate(
                    summary,
                    records,
                    expected_dates,
                    str(universe["listed_on_or_before"]),
                    float(universe["minimum_daily_coverage"]),
                    rejections,
                )
            )
        candidate_audits.sort(key=lambda row: row["symbol"])
        eligible_count = len(
            [row for row in candidate_audits if row["eligible"]]
        )
        print(
            f"non-target candidates: {len(candidate_audits)} audited, "
            f"{eligible_count} eligible",
            flush=True,
        )

    discovery = {
        "root_prefix": root_prefix,
        "all_symbol_prefix_count": len(all_symbols),
        "candidate_symbol_count": len(candidate_symbols),
        "prefiltered_symbol_count": len(prefiltered),
        "audited_candidate_count": len(candidate_audits),
        "exclusion_counts": exclusion_counts,
        "root_pages": root_pages,
        "symbol_listing_pages": sorted(
            symbol_listing_pages,
            key=lambda row: (str(row.get("symbol")), str(row["url"])),
        ),
    }
    result = build_inventory_result(
        blueprint_result,
        discovery,
        candidate_audits,
        archive_records,
        config,
        archive_rejections,
    )
    result["source_hashes"] = {
        str(v26_path.relative_to(root)): _sha256_file(v26_path),
        str(v26_audit_path.relative_to(root)): _sha256_file(v26_audit_path),
    }
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "universe.json").write_text(
        json.dumps(result["universe"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest_path = output / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            json.dumps(row, sort_keys=True)
            for row in result["archive_manifest"]
        ) + "\n",
        encoding="utf-8",
    )
    legacy_manifest_path = output / "manifest.json"
    if legacy_manifest_path.is_file():
        legacy_manifest_path.unlink()
    (output / "candidate_audits.json").write_text(
        json.dumps(result["candidate_audits"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output / "archive_rejections.json").write_text(
        json.dumps(result["archive_rejections"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output / "source_index.json").write_text(
        json.dumps(result["discovery"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output / "audit.json").write_text(
        json.dumps(result["audit"], indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    (output / "report.md").write_text(_report(result), encoding="utf-8")
    inventory_summary = {
        key: value
        for key, value in result.items()
        if key not in {
            "archive_manifest",
            "archive_rejections",
            "candidate_audits",
            "discovery",
        }
    }
    inventory_summary["artifact_references"] = {
        "archive_manifest": {
            "path": "manifest.jsonl",
            "records": len(result["archive_manifest"]),
            "sha256": _sha256_file(manifest_path),
        },
        "archive_rejections": {
            "path": "archive_rejections.json",
            "records": len(result["archive_rejections"]),
            "sha256": _sha256_file(output / "archive_rejections.json"),
        },
        "candidate_audits": {
            "path": "candidate_audits.json",
            "records": len(result["candidate_audits"]),
            "sha256": _sha256_file(output / "candidate_audits.json"),
        },
        "source_index": {
            "path": "source_index.json",
            "sha256": _sha256_file(output / "source_index.json"),
        },
    }
    (output / "inventory.json").write_text(
        json.dumps(inventory_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result
