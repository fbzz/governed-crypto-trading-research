from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from .non_target_inventory import (
    FetchBytes,
    _http_get,
    audit_candidate,
    list_s3_objects,
    verify_month_archive,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"JSONL manifest is empty: {path}")
    return rows


def _monthly_periods(start: str, end: str) -> list[str]:
    return [str(period) for period in pd.period_range(start, end, freq="M")]


def _manifest_record(record: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"dates", "cache_path"}
    }


def _expected_names(symbol: str, interval: str, months: list[str]) -> dict[str, str]:
    return {
        month: f"{symbol}-{interval}-{month}.zip" for month in months
    }


def summarize_full_listing(
    symbol: str,
    objects: list,
    interval: str,
    months: list[str],
) -> dict[str, object]:
    expected = _expected_names(symbol, interval, months)
    names = {Path(item.key).name for item in objects}
    complete_months = [
        month
        for month, name in expected.items()
        if name in names and f"{name}.CHECKSUM" in names
    ]
    missing_months = [month for month in months if month not in complete_months]
    return {
        "symbol": symbol,
        "base": symbol.removesuffix("USDT"),
        "complete_months": complete_months,
        "missing_months": missing_months,
        "expected_month_count": len(months),
        "complete_month_count": len(complete_months),
    }


def build_selected_manifest_result(
    amendment: dict,
    v30_inventory: dict,
    source_audits: list[dict[str, object]],
    archive_records: list[dict[str, object]],
    archive_rejections: list[dict[str, str]],
    discovery: dict[str, object],
    config: dict,
) -> dict[str, object]:
    manifest_config = config["selected_source_manifest"]
    blueprint = amendment["blueprint"]
    selected_symbols = list(v30_inventory["universe"]["selected_symbols"])
    selected_set = set(selected_symbols)
    manifest = sorted(
        (_manifest_record(record) for record in archive_records),
        key=lambda row: (row["symbol"], row["month"]),
    )
    target_symbols = set(blueprint["target_contract"]["symbols"])
    months = set(discovery["development_months"])
    record_months = {str(record["month"]) for record in manifest}
    observed_rows = sum(int(record["row_count"]) for record in manifest)
    expected_rows = int(discovery["expected_calendar_days"]) * len(selected_symbols)
    checks = {
        "v29_blueprint_hash_matches": amendment["blueprint_sha256"]
        == manifest_config["expected_v29_blueprint_sha256"],
        "v30_inventory_hash_matches": manifest_config["observed_v30_inventory_sha256"]
        == manifest_config["expected_v30_inventory_sha256"],
        "v30_audit_passes": bool(v30_inventory["audit"]["passed"]),
        "v30_authorizes_only_v31": v30_inventory["decision"]
        == "authorize_v31_selected_universe_manifest_refresh_only",
        "universe_is_exactly_v30": len(selected_symbols) == 30
        and {row["symbol"] for row in source_audits} == selected_set
        and {record["symbol"] for record in manifest} == selected_set,
        "universe_order_is_unchanged": selected_symbols
        == v30_inventory["universe"]["selected_symbols"],
        "asset_folds_are_unchanged": v30_inventory["asset_folds"]["fold_count"] == 3
        and all(
            len(fold["test_symbols"]) == 10
            for fold in v30_inventory["asset_folds"]["folds"]
        ),
        "no_target_symbols_loaded": not target_symbols.intersection(selected_set),
        "full_development_window_only": record_months.issubset(months)
        and discovery["development_start"] == "2021-01-01"
        and discovery["development_end"] == "2026-06-30",
        "all_expected_archive_jobs_accounted_for": len(manifest)
        + len(archive_rejections)
        == int(discovery["expected_archive_count"]),
        "all_accepted_checksums_verified": bool(manifest)
        and all(record["checksum_verified"] for record in manifest),
        "all_accepted_schemas_valid": bool(manifest)
        and all(record["schema_valid"] for record in manifest),
        "all_assets_preserved_regardless_of_future_coverage": {
            row["symbol"] for row in source_audits
        } == selected_set
        and all(row["eligible"] for row in source_audits),
        "future_coverage_is_audit_only": manifest_config[
            "full_window_coverage_policy"
        ] == "audit_only_no_reselection_or_replacement",
        "observed_rows_do_not_exceed_panel": observed_rows <= expected_rows,
        "source_index_is_hashed": bool(discovery["symbol_listing_pages"])
        and all(page["sha256"] for page in discovery["symbol_listing_pages"]),
        "selection_was_not_recomputed": True,
        "feature_count_is_zero": True,
        "label_read_count_is_zero": True,
        "return_computation_count_is_zero": True,
        "model_training_count_is_zero": True,
        "portfolio_count_is_zero": True,
        "performance_metric_count_is_zero": True,
        "pnl_evaluation_count_is_zero": True,
        "target_asset_load_count_is_zero": True,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise RuntimeError(f"V31 selected-source manifest audit failed: {checks}")
    coverage = [float(row["coverage"]) for row in source_audits]
    return {
        "version": "v31",
        "method": "fixed_universe_full_window_source_manifest",
        "decision": "authorize_v32_selected_universe_dataset_only",
        "blueprint_sha256": amendment["blueprint_sha256"],
        "universe": {
            "selected_symbols": selected_symbols,
            "selected_count": len(selected_symbols),
            "selection_source": "v30_frozen_no_reselection",
            "asset_folds": v30_inventory["asset_folds"],
        },
        "manifest_summary": {
            "accepted_archive_count": len(manifest),
            "rejected_archive_count": len(archive_rejections),
            "expected_archive_count": discovery["expected_archive_count"],
            "observed_rows": observed_rows,
            "expected_panel_rows": expected_rows,
            "preserved_missing_rows": expected_rows - observed_rows,
            "coverage_min": min(coverage),
            "coverage_max": max(coverage),
            "development_start": discovery["development_start"],
            "development_end": discovery["development_end"],
        },
        "source_audits": source_audits,
        "archive_manifest": manifest,
        "archive_rejections": archive_rejections,
        "discovery": discovery,
        "tested": {
            "universe_reselected": False,
            "features_built": False,
            "labels_read": False,
            "returns_computed": False,
            "model_trained": False,
            "portfolio_constructed": False,
            "performance_metrics_computed": False,
            "pnl_computed": False,
            "target_assets_loaded": False,
            "improvement_status": "unknown_not_evaluated",
            "drawdown_status": "unknown_not_evaluated",
        },
        "audit": {"passed": True, "checks": checks},
    }


def _report(result: dict[str, object]) -> str:
    summary = result["manifest_summary"]
    rows = result["source_audits"]
    lines = [
        "# TLM v31 Selected-Universe Source Manifest",
        "",
        "## Decision",
        "",
        "**FULL SOURCE MANIFEST PASSED FOR THE FIXED 30-ASSET UNIVERSE; TRAINING REMAINS BLOCKED.**",
        "",
        f"Accepted monthly archives: **{summary['accepted_archive_count']:,}**",
        f"Rejected monthly archives: **{summary['rejected_archive_count']:,}**",
        f"Observed daily rows: **{summary['observed_rows']:,}**",
        f"Preserved missing rows: **{summary['preserved_missing_rows']:,}**",
        f"Coverage range: **{summary['coverage_min']:.2%}-{summary['coverage_max']:.2%}**",
        "",
        "The exact v30 universe and folds were preserved. No reselection, target data, feature, label, return, model, portfolio, performance metric, or PnL operation occurred.",
        "",
        "## Coverage",
        "",
        "| Symbol | Days | Coverage | First | Last | Rejections |",
        "|---|---:|---:|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['symbol']} | {row['observed_days']} | {row['coverage']:.2%} | "
            f"{row['first_date']} | {row['last_date']} | {row['rejected_archive_count']} |"
        )
    if result["archive_rejections"]:
        lines.extend([
            "",
            "## Preserved source rejections",
            "",
            "| Symbol | Month | Error |",
            "|---|---|---|",
        ])
        for rejection in result["archive_rejections"]:
            lines.append(
                f"| {rejection['symbol']} | {rejection['month']} | {rejection['error']} |"
            )
    lines.extend([
        "",
        "## Next action",
        "",
        "V32 may build the causal 30-asset feature/label panel from this exact manifest, preserve all gaps, reproduce the panel byte-for-byte, and retain the frozen folds. It may not fit a scaler, train a model, construct a portfolio, or load BTC/ETH/SOL.",
        "",
    ])
    return "\n".join(lines)


def run_selected_source_manifest(
    config: dict,
    force: bool = False,
    fetch_bytes: FetchBytes = _http_get,
) -> dict[str, object]:
    manifest_config = config["selected_source_manifest"]
    root = Path(manifest_config["project_root"]).resolve()
    paths = {
        "v29_amendment": root / manifest_config["v29_amendment_path"],
        "v30_inventory": root / manifest_config["v30_inventory_path"],
        "v30_selected_manifest": root / manifest_config["v30_selected_manifest_path"],
    }
    for name, path in paths.items():
        expected = manifest_config[f"expected_{name}_sha256"]
        if not path.is_file() or _sha256_file(path) != expected:
            raise RuntimeError(f"V31 input missing or hash drifted: {name}")
    amendment = _load_json(paths["v29_amendment"])
    v30_inventory = _load_json(paths["v30_inventory"])
    v30_selection_records = _load_jsonl(paths["v30_selected_manifest"])
    if {row["symbol"] for row in v30_selection_records} != set(
        v30_inventory["universe"]["selected_symbols"]
    ):
        raise RuntimeError("V30 selection manifest symbols drifted")

    blueprint = amendment["blueprint"]
    data_contract = blueprint["data_contract"]
    symbols = list(v30_inventory["universe"]["selected_symbols"])
    interval = str(data_contract["frequency"])
    start = str(data_contract["development_start"][:10])
    end = str(data_contract["development_cutoff"][:10])
    months = _monthly_periods(start, end)
    expected_dates = {
        date.date().isoformat() for date in pd.date_range(start, end, freq="D")
    }
    endpoint = manifest_config["s3_endpoint"]
    root_prefix = manifest_config["root_prefix"]
    timeout = float(manifest_config["timeout_seconds"])

    summaries: list[dict[str, object]] = []
    listing_pages: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=int(manifest_config["listing_workers"])) as executor:
        futures = {
            executor.submit(
                list_s3_objects,
                endpoint,
                f"{root_prefix}{symbol}/{interval}/",
                timeout,
                fetch_bytes,
            ): symbol
            for symbol in symbols
        }
        for future in as_completed(futures):
            symbol = futures[future]
            objects, pages = future.result()
            summaries.append(summarize_full_listing(symbol, objects, interval, months))
            for page in pages:
                page["symbol"] = symbol
            listing_pages.extend(pages)
    summaries.sort(key=lambda row: row["symbol"])

    rejections: list[dict[str, str]] = []
    for summary in summaries:
        for month in summary["missing_months"]:
            rejections.append({
                "symbol": str(summary["symbol"]),
                "month": str(month),
                "error_type": "MissingArchiveOrChecksum",
                "error": "Official archive or published checksum is missing",
            })
    jobs = [
        (str(summary["symbol"]), str(month))
        for summary in summaries
        for month in summary["complete_months"]
    ]
    records: list[dict[str, object]] = []
    raw_dir = root / manifest_config["raw_dir"]
    with ThreadPoolExecutor(max_workers=int(manifest_config["archive_workers"])) as executor:
        futures = {
            executor.submit(
                verify_month_archive,
                symbol,
                month,
                interval,
                manifest_config["download_base_url"],
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
                records.append(future.result())
            except Exception as error:
                rejections.append({
                    "symbol": symbol,
                    "month": month,
                    "error_type": type(error).__name__,
                    "error": str(error),
                })
                print(f"v31 archive rejected: {symbol} {month}: {error}", flush=True)
            if completed % 250 == 0 or completed == len(futures):
                print(f"v31 source archives: {completed}/{len(futures)}", flush=True)
    records.sort(key=lambda row: (row["symbol"], row["month"]))
    rejections.sort(key=lambda row: (row["symbol"], row["month"]))
    source_audits = [
        audit_candidate(
            summary,
            [record for record in records if record["symbol"] == summary["symbol"]],
            expected_dates,
            start,
            0.0,
            [row for row in rejections if row["symbol"] == summary["symbol"]],
        )
        for summary in summaries
    ]
    source_audits.sort(key=lambda row: row["symbol"])
    discovery = {
        "development_start": start,
        "development_end": end,
        "development_months": months,
        "expected_calendar_days": len(expected_dates),
        "expected_archive_count": len(symbols) * len(months),
        "complete_listing_job_count": len(jobs),
        "symbol_listing_pages": sorted(
            listing_pages,
            key=lambda row: (str(row.get("symbol")), str(row["url"])),
        ),
    }
    manifest_config["observed_v30_inventory_sha256"] = _sha256_file(
        paths["v30_inventory"]
    )
    result = build_selected_manifest_result(
        amendment,
        v30_inventory,
        source_audits,
        records,
        rejections,
        discovery,
        config,
    )
    result["source_hashes"] = {
        str(path.relative_to(root)): _sha256_file(path) for path in paths.values()
    }

    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in result["archive_manifest"])
        + "\n",
        encoding="utf-8",
    )
    files = {
        "source_audits.json": result["source_audits"],
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
            "source_audits",
            "discovery",
        }
    }
    summary["artifact_references"] = {
        "manifest": {
            "path": "manifest.jsonl",
            "records": len(result["archive_manifest"]),
            "sha256": _sha256_file(manifest_path),
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
