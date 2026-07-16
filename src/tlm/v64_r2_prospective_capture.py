"""V70-R1 prospective non-target source capture and immutable prediction freeze."""

from __future__ import annotations

import csv
from datetime import datetime, time, timezone
from decimal import Decimal
import fcntl
import gc
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import zipfile

import numpy as np
import pandas as pd
import torch
import yaml

from .core.artifacts import canonical_sha256, file_sha256, write_json_atomic
from .decoupled_rank_state_harness import derive_state_features
from .non_target_dataset import PANEL_FEATURES
from .non_target_pretraining import TripletTensorStore
from .research_workflow import validate_research_state
from .scientific_harness import FeatureScaler
from .state_conditioned_multi_horizon_training_engine import semantic_state_sha256
from .v64_r2_probabilistic_state_gate_harness import probability_of_clearing_cost
from .v64_r2_probabilistic_state_gate_training_engine import (
    FINAL_FORMAT,
    configure_v68_runtime,
    instantiate_v68_models,
)


TARGET_SYMBOLS = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
EXPECTED_FOLDS = (1, 2, 3)
EXPECTED_SEEDS = (42, 7, 123)
CAPTURE_DELAY = time(0, 5, tzinfo=timezone.utc)
KLINE_WIDTH = 12


class V70CaptureError(RuntimeError):
    """Raised when V70 cannot preserve its frozen prospective contract."""


class V70SourceUnavailable(V70CaptureError):
    """Raised when an exact public source row is missing or late."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V70CaptureError(f"Unable to read JSON metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise V70CaptureError(f"Expected JSON object: {path}")
    return value


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_once_json(path: Path, value: object) -> str:
    payload = _json_bytes(value)
    digest = _sha256_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        observed = path.read_bytes()
        if observed != payload:
            raise V70CaptureError(f"Immutable V70 artifact drift: {path}")
        return digest
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return digest


def _as_utc(value: datetime | pd.Timestamp | None) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.now(tz="UTC")
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        result = result.tz_localize("UTC")
    else:
        result = result.tz_convert("UTC")
    return result


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=False
    )
    if result.returncode:
        raise V70CaptureError(
            f"git {' '.join(args)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def resolve_registration_anchor(
    root: Path, anchor_contract: Mapping[str, Any]
) -> dict[str, Any]:
    relative = str(anchor_contract.get("amendment_path", ""))
    expected = str(anchor_contract.get("amendment_file_sha256", ""))
    if (
        relative != "research/amendments/v070_r1_metadata_only.yaml"
        or len(expected) != 64
        or anchor_contract.get("resolver")
        != "first_ancestor_commit_containing_exact_amendment_file_sha256"
    ):
        raise V70CaptureError("V70-R1 registration-anchor contract drift")
    commits = _git(root, "rev-list", "--reverse", "HEAD", "--", relative).splitlines()
    for commit in commits:
        content = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=root,
            capture_output=True,
            check=False,
        )
        if content.returncode == 0 and _sha256_bytes(content.stdout) == expected:
            timestamp = pd.Timestamp(_git(root, "show", "-s", "--format=%cI", commit))
            if timestamp.tzinfo is None:
                raise V70CaptureError("V70-R1 Git timestamp lacks timezone")
            return {
                "commit": commit,
                "commit_timestamp_utc": timestamp.tz_convert("UTC").isoformat(),
                "amendment_path": relative,
                "amendment_file_sha256": expected,
            }
    raise V70CaptureError("Exact V70-R1 amendment commit could not be resolved")


def latest_capture_candidate(
    now_utc: datetime | pd.Timestamp,
    anchor_timestamp_utc: datetime | pd.Timestamp,
) -> pd.Timestamp | None:
    now = _as_utc(now_utc)
    anchor = _as_utc(anchor_timestamp_utc)
    today = now.floor("D")
    after_delay = (now.hour, now.minute, now.second) >= (
        CAPTURE_DELAY.hour,
        CAPTURE_DELAY.minute,
        CAPTURE_DELAY.second,
    )
    feature_date = today - pd.Timedelta(days=1 if after_delay else 2)
    candle_close = feature_date + pd.Timedelta(days=1)
    maturity = feature_date + pd.Timedelta(days=2)
    if candle_close <= anchor or now >= maturity:
        return None
    return feature_date


class PublicHTTPFetcher:
    def __init__(self, timeout_seconds: int) -> None:
        self.timeout_seconds = int(timeout_seconds)

    def __call__(self, url: str) -> bytes:
        request = Request(url, headers={"User-Agent": "TLM-V70-R1/1.0"})
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            raise V70SourceUnavailable(
                f"Public source request unavailable: {url}: {exc}"
            ) from exc


def _all_fold_symbols(asset_folds: Mapping[str, Any]) -> list[str]:
    folds = asset_folds.get("folds", [])
    symbols = sorted(
        set(itertools.chain.from_iterable(row.get("test_symbols", []) for row in folds))
    )
    if len(folds) != 3 or len(symbols) != 30 or TARGET_SYMBOLS.intersection(symbols):
        raise V70CaptureError("V70 exact non-target fold universe drift")
    test_sets = [set(row["test_symbols"]) for row in folds]
    if any(len(values) != 10 for values in test_sets) or any(
        test_sets[left].intersection(test_sets[right])
        for left in range(3)
        for right in range(left + 1, 3)
    ):
        raise V70CaptureError("V70 fold test sets are not three disjoint tens")
    return symbols


def _scaler(record: Mapping[str, Any]) -> FeatureScaler:
    value = record["feature_scaler"]
    scaler = FeatureScaler(
        feature_names=tuple(str(name) for name in value["feature_names"]),
        mean=tuple(float(number) for number in value["mean"]),
        scale=tuple(float(number) for number in value["scale"]),
        source_relative_feature_index=int(value["source_relative_feature_index"]),
        fit_scope=str(value["fit_scope"]),
        fit_start=str(value["fit_start"]),
        fit_end=str(value["fit_end"]),
        fit_rows=int(value["fit_rows"]),
    )
    if scaler.state_sha256() != record["feature_scaler_state_sha256"]:
        raise V70CaptureError("V70 frozen feature-scaler identity drift")
    if scaler.feature_names != tuple(PANEL_FEATURES):
        raise V70CaptureError("V70 feature order drift")
    return scaler


def _source_code_receipt(
    root: Path, spec: Mapping[str, Any], *, require_clean: bool
) -> dict[str, Any]:
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    if require_clean and status:
        raise V70CaptureError("V70 requires a clean committed source tree")
    files = [str(value) for value in spec.get("source_receipt_files", [])]
    if not files or len(files) != len(set(files)):
        raise V70CaptureError("V70 source receipt file list is empty or duplicated")
    hashes = {relative: file_sha256(root / relative) for relative in files}
    return {
        "schema_version": "v70-r1-source-code-receipt/v1",
        "git_clean": not bool(status),
        "git_head": _git(root, "rev-parse", "HEAD"),
        "files": hashes,
        "bundle_sha256": canonical_sha256(hashes),
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__},
    }


def _context(config: Mapping[str, Any], *, require_clean: bool) -> dict[str, Any]:
    spec = config.get("v64_r2_prospective_capture")
    if not isinstance(spec, dict) or spec.get("version") != "v70-r1":
        raise V70CaptureError("Missing frozen V70-R1 capture config")
    root = Path(spec.get("project_root", ".")).resolve()
    state = validate_research_state(root, str(spec["research_state"]))
    active = (
        state.get("passed") is True
        and state.get("authorized_phase") == "v70"
        and state.get("authorized_next_action")
        == "authorize_v70_prospective_non_target_capture_and_prediction_freeze_only"
    )
    historical_test_reconstruction = (
        not require_clean
        and state.get("passed") is True
        and state.get("authorized_phase")
            in {
                "v71",
                "v72",
                "v73",
                "v74",
                "v75",
                "v76",
                "v77",
                "v78",
                "v79",
                "v80",
                "v81",
                        "v82-r0",
                        "v82",
                            "v83",
                            "v84",
                            "v85",
                        }
    )
    if not active and not historical_test_reconstruction:
        raise V70CaptureError("V70 research authorization is not active")
    contract_path = root / str(spec["phase_contract"])
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    if (
        contract.get("phase") != "v70"
        or contract.get("stage_revision")
        != "v070_prospective_non_target_capture_prediction_freeze_r2"
    ):
        raise V70CaptureError("V70-R1 phase revision drift")
    current = yaml.safe_load((root / str(spec["research_state"])).read_text())
    if active and current.get("phase_contract") != {
        "path": str(spec["phase_contract"]),
        "file_sha256": file_sha256(contract_path),
    }:
        raise V70CaptureError("V70 live phase-contract reference drift")
    if historical_test_reconstruction and (
        current.get("safety", {}).get("v70_prospective_capture_paused_by_owner")
        is not True
        or current.get("safety", {}).get("v70_immutable_artifacts_preserved")
        is not True
    ):
        raise V70CaptureError("V70 historical test reconstruction is not registered")
    allowed = list(contract["access_contract"]["allowed_inputs"])
    expected = dict(
        contract["input_contract"]["expected_static_file_sha256_by_path"]
    )
    if len(allowed) != 26 or set(allowed) != set(expected):
        raise V70CaptureError("V70 static input allowlist drift")
    observed = {relative: file_sha256(root / relative) for relative in allowed}
    if observed != expected:
        raise V70CaptureError("V70 immutable static input hash drift")
    loaded = {
        relative: _load_json(root / relative)
        for relative in allowed
        if relative.endswith(".json")
    }
    blueprint = loaded[
        "artifacts/v65_v64_r2_probabilistic_state_gate_spec/blueprint.json"
    ]
    asset_folds = loaded[
        "artifacts/v32_selected_universe_dataset/asset_folds.json"
    ]
    triplet_catalog = loaded[
        "artifacts/v32_selected_universe_dataset/triplet_catalog.json"
    ]
    checkpoint_manifest = loaded[
        "artifacts/v68_v64_r2_probabilistic_state_gate_training/checkpoint_manifest.json"
    ]
    scaler_manifest = loaded[
        "artifacts/v68_v64_r2_probabilistic_state_gate_training/scaler_manifest.json"
    ]
    ranker_scale_receipt = loaded[
        "research/receipts/v070_ranker_excess_scale_receipt.json"
    ]
    protocol = loaded[
        "artifacts/v69_v64_r2_prospective_confirmation_prepare/protocol.json"
    ]
    symbols = _all_fold_symbols(asset_folds)
    if protocol["policy"] != blueprint["policy"]:
        raise V70CaptureError("V70 frozen policy differs from V69/V65")
    if {(int(row["fold"]), int(row["seed"])) for row in checkpoint_manifest["jobs"]} != {
        (fold, seed) for fold in EXPECTED_FOLDS for seed in EXPECTED_SEEDS
    }:
        raise V70CaptureError("V70 checkpoint grid drift")
    scales = {int(row["fold"]): row for row in scaler_manifest["folds"]}
    ranker_scales = {
        int(row["fold"]): row for row in ranker_scale_receipt["folds"]
    }
    if not set(scales) == set(ranker_scales) == set(EXPECTED_FOLDS):
        raise V70CaptureError("V70 fold scaler grid drift")
    for fold in EXPECTED_FOLDS:
        _scaler(scales[fold])
        if (
            scales[fold]["feature_scaler_state_sha256"]
            != ranker_scales[fold]["feature_scaler_state_sha256"]
            or scales[fold]["source_v63_fold_scale_sha256"]
            != ranker_scales[fold]["source_fold_scale_sha256"]
            or float(ranker_scales[fold]["ranker_excess_rms"]) <= 0.0
        ):
            raise V70CaptureError("V70-R1 ranker scale identity drift")
    catalog_by_fold = {
        int(row["fold"]): row for row in triplet_catalog["folds"]
    }
    folds_by_id = {int(row["fold"]): row for row in asset_folds["folds"]}
    for fold in EXPECTED_FOLDS:
        expected_triplets = [
            list(values)
            for values in itertools.combinations(sorted(folds_by_id[fold]["test_symbols"]), 3)
        ]
        if catalog_by_fold[fold]["test_triplets"] != expected_triplets:
            raise V70CaptureError("V70 exact lexical triplet catalog drift")
    output = root / str(config["output_dir"])
    return {
        "root": root,
        "spec": spec,
        "contract": contract,
        "contract_path": contract_path,
        "blueprint": blueprint,
        "asset_folds": asset_folds,
        "triplet_catalog": triplet_catalog,
        "checkpoint_manifest": checkpoint_manifest,
        "scaler_manifest": scaler_manifest,
        "ranker_scale_receipt": ranker_scale_receipt,
        "protocol": protocol,
        "symbols": symbols,
        "output": output,
        "source_code_receipt": _source_code_receipt(
            root, spec, require_clean=require_clean
        ),
        "anchor": resolve_registration_anchor(
            root, contract["registration_anchor_contract"]
        ),
        "static_input_sha256": observed,
    }


def _load_inference_models(
    context: Mapping[str, Any], device: torch.device
) -> tuple[dict[int, list[tuple[int, torch.nn.Module, torch.nn.Module]]], list[dict[str, Any]]]:
    models: dict[int, list[tuple[int, torch.nn.Module, torch.nn.Module]]] = {
        fold: [] for fold in EXPECTED_FOLDS
    }
    receipts: list[dict[str, Any]] = []
    for row in sorted(
        context["checkpoint_manifest"]["jobs"],
        key=lambda item: (
            int(item["fold"]),
            EXPECTED_SEEDS.index(int(item["seed"])),
        ),
    ):
        fold = int(row["fold"])
        seed = int(row["seed"])
        path = context["root"] / row["path"]
        if file_sha256(path) != row["file_sha256"]:
            raise V70CaptureError(f"V70 checkpoint file drift: {row['job_id']}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        registered = payload.get("semantic_checkpoint_sha256")
        body = {key: value for key, value in payload.items() if key != "semantic_checkpoint_sha256"}
        if (
            payload.get("format_version") != FINAL_FORMAT
            or payload.get("kind") != "final"
            or payload.get("stage") != "complete"
            or payload.get("job_context") != row["context"]
            or registered != row["semantic_checkpoint_sha256"]
            or semantic_state_sha256(body) != registered
            or semantic_state_sha256(payload["ranker_state"])
            != row["ranker_state_sha256"]
            or semantic_state_sha256(payload["gate_current_state"])
            != row["gate_state_sha256"]
        ):
            raise V70CaptureError(f"V70 checkpoint semantic drift: {row['job_id']}")
        ranker, gate = instantiate_v68_models(
            context["blueprint"], device, seed=seed
        )
        ranker.load_state_dict(payload["ranker_state"], strict=True)
        gate.load_state_dict(payload["gate_current_state"], strict=True)
        ranker.eval().requires_grad_(False)
        gate.eval().requires_grad_(False)
        models[fold].append((seed, ranker, gate))
        receipts.append(
            {
                "job_id": row["job_id"],
                "fold": fold,
                "seed": seed,
                "path": row["path"],
                "file_sha256": row["file_sha256"],
                "semantic_checkpoint_sha256": registered,
                "ranker_state_sha256": row["ranker_state_sha256"],
                "gate_state_sha256": row["gate_state_sha256"],
                "used_without_selection": True,
                "optimizer_created": False,
            }
        )
        del payload
    if any([seed for seed, _, _ in models[fold]] != list(EXPECTED_SEEDS) for fold in EXPECTED_FOLDS):
        raise V70CaptureError("V70 seed order/grid drift")
    return models, receipts


def _normalize_epoch_ms(value: Any) -> int:
    integer = int(value)
    if abs(integer) >= 100_000_000_000_000:
        integer //= 1000
    return integer


def _validate_kline_rows(
    symbol: str, rows: Any, start: pd.Timestamp, end: pd.Timestamp
) -> list[list[Any]]:
    if not isinstance(rows, list) or not rows:
        raise V70SourceUnavailable(f"Empty public kline response for {symbol}")
    normalized: list[list[Any]] = []
    for raw in rows:
        if not isinstance(raw, list) or len(raw) != KLINE_WIDTH:
            raise V70CaptureError(f"Invalid public kline width for {symbol}")
        row = list(raw)
        row[0] = _normalize_epoch_ms(row[0])
        row[6] = _normalize_epoch_ms(row[6])
        normalized.append(row)
    dates = pd.DatetimeIndex(
        pd.to_datetime([row[0] for row in normalized], unit="ms", utc=True)
    )
    expected = pd.date_range(start, end, freq="D", tz="UTC")
    if not dates.equals(expected):
        raise V70SourceUnavailable(
            f"Public source missing, late, duplicated, or non-contiguous for {symbol}"
        )
    for row, date in zip(normalized, dates):
        if row[6] != int((date + pd.Timedelta(days=1)).timestamp() * 1000) - 1:
            raise V70CaptureError(f"Invalid daily close timestamp for {symbol}")
        prices = [float(row[index]) for index in (1, 2, 3, 4)]
        activity = [float(row[index]) for index in (5, 7, 8)]
        if (
            not all(math.isfinite(value) and value > 0.0 for value in prices)
            or not all(math.isfinite(value) and value >= 0.0 for value in activity)
            or prices[1] < max(prices[0], prices[3])
            or prices[2] > min(prices[0], prices[3])
        ):
            raise V70CaptureError(f"Invalid public daily candle values for {symbol}")
    return normalized


def _fetch_source_packet(
    context: Mapping[str, Any],
    feature_date: pd.Timestamp,
    freeze_time: pd.Timestamp,
    fetch: Callable[[str], bytes],
) -> dict[str, Any]:
    lookback = int(context["spec"]["initial_raw_lookback_days"])
    if lookback != 286:
        raise V70CaptureError("V70 raw lookback must remain 286 days")
    start = feature_date - pd.Timedelta(days=lookback - 1)
    end = feature_date
    rows_by_symbol: dict[str, list[list[Any]]] = {}
    responses = []
    omissions = []
    for symbol in context["symbols"]:
        query = urlencode(
            {
                "symbol": symbol,
                "interval": "1d",
                "startTime": int(start.timestamp() * 1000),
                "endTime": int((end + pd.Timedelta(days=1)).timestamp() * 1000) - 1,
                "limit": lookback,
            }
        )
        url = f"{str(context['spec']['public_rest_base_url']).rstrip('/')}/api/v3/klines?{query}"
        if any(target in url for target in TARGET_SYMBOLS):
            raise V70CaptureError("Target symbol entered V70 public source URL")
        try:
            payload = fetch(url)
        except V70SourceUnavailable as exc:
            omissions.append({"symbol": symbol, "reason": str(exc)})
            continue
        try:
            raw_rows = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise V70CaptureError(f"Invalid public source JSON for {symbol}") from exc
        try:
            rows = _validate_kline_rows(symbol, raw_rows, start, end)
        except V70SourceUnavailable as exc:
            omissions.append(
                {
                    "symbol": symbol,
                    "reason": str(exc),
                    "raw_response_sha256": _sha256_bytes(payload),
                }
            )
            continue
        rows_by_symbol[symbol] = rows
        responses.append(
            {
                "symbol": symbol,
                "url": url,
                "raw_response_sha256": _sha256_bytes(payload),
                "row_count": len(rows),
                "first_open_time_ms": rows[0][0],
                "last_open_time_ms": rows[-1][0],
            }
        )
    maturity = feature_date + pd.Timedelta(days=2)
    if freeze_time >= maturity:
        raise V70CaptureError("V70 source packet missed its h1 maturity deadline")
    fold_readiness = []
    available = set(rows_by_symbol)
    for fold_row in context["asset_folds"]["folds"]:
        fold = int(fold_row["fold"])
        registered = list(fold_row["test_symbols"])
        eligible = [symbol for symbol in registered if symbol in available]
        fold_readiness.append(
            {
                "fold": fold,
                "registered_symbols": registered,
                "eligible_symbols": eligible,
                "missing_symbols": [
                    symbol for symbol in registered if symbol not in available
                ],
                "eligible": len(eligible) >= 3,
                "eligible_triplet_count": math.comb(len(eligible), 3)
                if len(eligible) >= 3
                else 0,
            }
        )
    return {
        "schema_version": "v70-r1-daily-source-packet/v1",
        "feature_date": feature_date.date().isoformat(),
        "feature_candle_close_utc": (feature_date + pd.Timedelta(days=1)).isoformat(),
        "freeze_timestamp_utc": freeze_time.isoformat(),
        "target_h1_maturity_timestamp_utc": maturity.isoformat(),
        "source": "Binance public spot daily UTC klines",
        "authentication_used": False,
        "interval": "1d",
        "raw_lookback_days": lookback,
        "responses": responses,
        "rows_by_symbol": rows_by_symbol,
        "complete": not omissions and len(rows_by_symbol) == 30,
        "omissions": omissions,
        "fold_readiness": fold_readiness,
        "archive_reconciliation": "pending_append_only_sidecar",
        "outcome_rows_read": 0,
        "target_assets_loaded": [],
    }


def _feature_panel(source_packet: Mapping[str, Any]) -> pd.DataFrame:
    feature_date = pd.Timestamp(source_packet["feature_date"], tz="UTC")
    frames = []
    for symbol, rows in sorted(source_packet["rows_by_symbol"].items()):
        dates = pd.DatetimeIndex(
            pd.to_datetime([int(row[0]) for row in rows], unit="ms", utc=True)
        )
        raw = pd.DataFrame(index=dates)
        field_indexes = {
            "open": 1,
            "high": 2,
            "low": 3,
            "close": 4,
            "volume": 5,
            "quote_volume": 7,
            "trade_count": 8,
        }
        for field, index in field_indexes.items():
            raw[field] = [float(row[index]) for row in rows]
        log_open = np.log(raw["open"])
        log_close = np.log(raw["close"])
        close_return = log_close.diff()
        panel = pd.DataFrame(
            {
                "date": dates,
                "symbol": symbol,
                "log_open_to_open_return": log_open.diff().to_numpy(),
                "log_close_to_close_return": close_return.to_numpy(),
                "log_high_low_range": np.log(raw["high"] / raw["low"]).to_numpy(),
                "log_close_open_return": np.log(raw["close"] / raw["open"]).to_numpy(),
                "log1p_quote_volume_change": np.log1p(raw["quote_volume"]).diff().to_numpy(),
                "log1p_trade_count_change": np.log1p(raw["trade_count"]).diff().to_numpy(),
                "rolling_realized_volatility_7d": close_return.pow(2)
                .rolling(7, min_periods=7)
                .sum()
                .pow(0.5)
                .to_numpy(),
                "rolling_realized_volatility_30d": close_return.pow(2)
                .rolling(30, min_periods=30)
                .sum()
                .pow(0.5)
                .to_numpy(),
            }
        )
        frames.append(panel)
    combined = pd.concat(frames, ignore_index=True).sort_values(
        ["symbol", "date"]
    )
    required_start = feature_date - pd.Timedelta(days=255)
    combined = combined.loc[combined["date"] >= required_start].reset_index(drop=True)
    if len(combined) != len(source_packet["rows_by_symbol"]) * 256:
        raise V70CaptureError("V70 feature panel does not contain exact available x 256 cells")
    if not np.isfinite(combined[list(PANEL_FEATURES)].to_numpy(dtype=np.float64)).all():
        raise V70CaptureError("V70 feature-only panel contains non-finite values")
    forbidden = {"target", "label", "outcome", "return_realized"}
    if any(any(token in column.lower() for token in forbidden) for column in combined.columns):
        raise V70CaptureError("V70 feature-only panel contains a forbidden column")
    return combined


def _fold_inference(
    context: Mapping[str, Any],
    panel: pd.DataFrame,
    fold: int,
    models: list[tuple[int, torch.nn.Module, torch.nn.Module]],
    device: torch.device,
) -> dict[str, Any]:
    fold_scope = next(
        row for row in context["asset_folds"]["folds"] if int(row["fold"]) == fold
    )
    catalog = next(
        row for row in context["triplet_catalog"]["folds"] if int(row["fold"]) == fold
    )
    registered_symbols = list(fold_scope["test_symbols"])
    eligible_symbols = [
        symbol
        for symbol in registered_symbols
        if bool((panel["symbol"] == symbol).any())
    ]
    registered_triplets = [tuple(values) for values in catalog["test_triplets"]]
    triplets = [
        values
        for values in registered_triplets
        if set(values).issubset(set(eligible_symbols))
    ]
    if (
        len(registered_symbols) != 10
        or len(registered_triplets) != 120
        or len(eligible_symbols) < 3
        or len(triplets) != math.comb(len(eligible_symbols), 3)
    ):
        raise V70CaptureError(f"V70 fold {fold} scope drift")
    feature_date = pd.Timestamp(panel["date"].max())
    scale_row = next(
        row for row in context["scaler_manifest"]["folds"] if int(row["fold"]) == fold
    )
    ranker_scale_row = next(
        row for row in context["ranker_scale_receipt"]["folds"] if int(row["fold"]) == fold
    )
    scaler = _scaler(scale_row)
    store = TripletTensorStore(
        panel.loc[panel["symbol"].isin(eligible_symbols)],
        list(PANEL_FEATURES),
        256,
        "log_close_to_close_return",
    )
    samples = [{"date": feature_date, "triplet": triplet} for triplet in triplets]
    asset_sum = {symbol: 0.0 for symbol in eligible_symbols}
    asset_count = {symbol: 0 for symbol in eligible_symbols}
    mixture: list[dict[str, Any]] = []
    batch_size = int(context["spec"]["inference_batch_size"])
    excess_rms = float(ranker_scale_row["ranker_excess_rms"])
    market_rms = float(scale_row["market_target_rms"])
    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        values = torch.from_numpy(store.materialize_batch(batch, scaler)).to(
            device=device, dtype=torch.float32
        )
        with torch.inference_mode():
            state = derive_state_features(values)
            for seed, ranker, gate in models:
                ranker_output = ranker(values)
                gate_output = gate(state)
                centered = ranker_output["excess_return_z"] - ranker_output[
                    "excess_return_z"
                ].mean(dim=1, keepdim=True)
                raw_excess = (centered * excess_rms).detach().cpu().numpy()
                location = (gate_output["location"] * market_rms).detach().cpu().numpy()
                scale = (gate_output["scale"] * market_rms).detach().cpu().numpy()
                if (
                    not np.isfinite(raw_excess).all()
                    or not np.isfinite(location).all()
                    or not np.isfinite(scale).all()
                    or (scale <= 0.0).any()
                ):
                    raise V70CaptureError(f"Non-finite V70 inference for fold {fold}")
                for offset, sample in enumerate(batch):
                    triplet = tuple(str(value) for value in sample["triplet"])
                    mixture.append(
                        {
                            "triplet_key": "|".join(triplet),
                            "seed": seed,
                            "market_location": float(location[offset]),
                            "market_scale": float(scale[offset]),
                        }
                    )
                    for slot, symbol in enumerate(triplet):
                        asset_sum[symbol] += float(raw_excess[offset, slot])
                        asset_count[symbol] += 1
        del values, state
    expected_asset_count = math.comb(len(eligible_symbols) - 1, 2) * len(models)
    if (
        len(mixture) != math.comb(len(eligible_symbols), 3) * len(models)
        or set(asset_count.values()) != {expected_asset_count}
    ):
        raise V70CaptureError(f"V70 fold {fold} context/seed aggregation drift")
    momentum = {}
    for symbol in eligible_symbols:
        values = panel.loc[
            panel["symbol"] == symbol, "log_close_to_close_return"
        ].to_numpy(dtype=np.float64)
        momentum[symbol] = float(values[-30:].sum())
    assets = [
        {
            "symbol": symbol,
            "raw_excess": float(asset_sum[symbol] / asset_count[symbol]),
            "momentum_30": momentum[symbol],
            "context_seed_count": asset_count[symbol],
        }
        for symbol in eligible_symbols
    ]
    market_mean = float(np.mean([row["market_location"] for row in mixture]))
    for row in assets:
        row["absolute_location"] = float(market_mean + row["raw_excess"])
    return {
        "fold": fold,
        "registered_symbols": registered_symbols,
        "eligible_symbols": eligible_symbols,
        "missing_symbols": [
            symbol for symbol in registered_symbols if symbol not in eligible_symbols
        ],
        "registered_triplet_context_count": len(registered_triplets),
        "triplet_context_count": len(triplets),
        "seed_count": len(models),
        "market_mixture_component_count": len(mixture),
        "market_location_mean": market_mean,
        "assets": assets,
        "market_mixture": mixture,
    }


def _policy_step(
    fold_prediction: Mapping[str, Any],
    previous_symbol: str | None,
    *,
    cost_bps: int,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    registered_symbols = list(fold_prediction["registered_symbols"])
    symbols = list(fold_prediction["eligible_symbols"])
    assets = {row["symbol"]: row for row in fold_prediction["assets"]}
    if set(assets) != set(symbols):
        raise V70CaptureError("V70 policy asset scope drift")
    previous = previous_symbol if previous_symbol in assets else None
    threshold = float(policy["abstention_probability_threshold"])
    hurdle = float(policy["switch_hurdle"])
    risky_weight = float(policy["risky_gross"])
    base_cost = float(cost_bps) / 10000.0
    locations = np.asarray(
        [row["market_location"] for row in fold_prediction["market_mixture"]],
        dtype=np.float64,
    )
    scales = np.asarray(
        [row["market_scale"] for row in fold_prediction["market_mixture"]],
        dtype=np.float64,
    )
    event_probability: float | None = None
    transition_cost: float | None = None
    selected = previous
    action: str
    if previous_symbol is not None and previous is None:
        action = "forced_exit"
        selected = None
        event_probability = None
        transition_cost = None
    elif all(float(assets[symbol]["momentum_30"]) <= 0.0 for symbol in symbols):
        action = "momentum_exit" if previous is not None else "cash"
        selected = None
    else:
        best = max(float(assets[symbol]["raw_excess"]) for symbol in symbols)
        challenger = next(
            symbol for symbol in symbols if float(assets[symbol]["raw_excess"]) == best
        )
        if previous is None:
            transition_cost = base_cost * risky_weight
            event_probability = probability_of_clearing_cost(
                locations,
                scales,
                asset_excess=float(assets[challenger]["raw_excess"]),
                transition_cost=transition_cost,
                degrees_of_freedom=5.0,
            )
            if event_probability >= threshold:
                selected = challenger
                action = "entry"
            else:
                selected = None
                action = "cash"
        else:
            hold_probability = probability_of_clearing_cost(
                locations,
                scales,
                asset_excess=float(assets[previous]["raw_excess"]),
                transition_cost=0.0,
                degrees_of_freedom=5.0,
            )
            if hold_probability < threshold:
                selected = None
                action = "probability_exit"
                event_probability = hold_probability
                transition_cost = 0.0
            elif challenger != previous and (
                float(assets[challenger]["raw_excess"])
                - float(assets[previous]["raw_excess"])
                > hurdle
            ):
                transition_cost = base_cost * 2.0 * risky_weight
                event_probability = probability_of_clearing_cost(
                    locations,
                    scales,
                    asset_excess=float(assets[challenger]["raw_excess"]),
                    transition_cost=transition_cost,
                    degrees_of_freedom=5.0,
                )
                if event_probability >= threshold:
                    selected = challenger
                    action = "switch"
                else:
                    selected = previous
                    action = "hold"
            else:
                event_probability = hold_probability
                transition_cost = 0.0
                selected = previous
                action = "hold"
    weights = {
        symbol: risky_weight if symbol == selected else 0.0
        for symbol in registered_symbols
    }
    previous_weights = {
        symbol: risky_weight if symbol == previous_symbol else 0.0
        for symbol in registered_symbols
    }
    turnover = float(
        sum(
            abs(weights[symbol] - previous_weights[symbol])
            for symbol in registered_symbols
        )
    )
    return {
        "cost_bps": int(cost_bps),
        "previous_selected_symbol": previous_symbol,
        "selected_symbol": selected,
        "action": action,
        "event_probability": event_probability,
        "transition_cost": transition_cost,
        "transition_turnover": turnover,
        "gross_exposure": float(sum(weights.values())),
        "weights": weights,
    }


def _previous_selected(
    output: Path, feature_date: pd.Timestamp
) -> dict[tuple[int, int], str | None]:
    previous_date = (feature_date - pd.Timedelta(days=1)).date().isoformat()
    path = output / "daily" / previous_date / "position_packet.json"
    if not path.is_file():
        return {(fold, cost): None for fold in EXPECTED_FOLDS for cost in (10, 20, 30)}
    packet = _load_json(path)
    result = {
        (fold, cost): None
        for fold in EXPECTED_FOLDS
        for cost in (10, 20, 30)
    }
    for fold in packet["folds"]:
        for cell in fold["cost_cells"]:
            result[(int(fold["fold"]), int(cell["cost_bps"]))] = cell[
                "selected_symbol"
            ]
    return result


def _prediction_and_position_packets(
    context: Mapping[str, Any],
    source_packet: Mapping[str, Any],
    checkpoint_receipts: list[dict[str, Any]],
    models: Mapping[int, list[tuple[int, torch.nn.Module, torch.nn.Module]]],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    panel = _feature_panel(source_packet)
    feature_date = pd.Timestamp(source_packet["feature_date"], tz="UTC")
    eligible_folds = [
        int(row["fold"])
        for row in source_packet["fold_readiness"]
        if row["eligible"] is True
    ]
    folds = [
        _fold_inference(context, panel, fold, models[fold], device)
        for fold in eligible_folds
    ]
    prediction = {
        "schema_version": "v70-r1-daily-prediction-packet/v1",
        "feature_date": source_packet["feature_date"],
        "freeze_timestamp_utc": source_packet["freeze_timestamp_utc"],
        "target_h1_maturity_timestamp_utc": source_packet[
            "target_h1_maturity_timestamp_utc"
        ],
        "registration_anchor": context["anchor"],
        "source_packet_sha256": _sha256_bytes(_json_bytes(source_packet)),
        "context_aggregation": "equal_weight_exact_lexical_test_triplets",
        "seed_aggregation": "equal_weight_all_three_seeds_before_policy",
        "folds": folds,
        "checkpoint_receipts": checkpoint_receipts,
        "ranker_excess_scale_receipt_sha256": file_sha256(
            context["root"] / "research/receipts/v070_ranker_excess_scale_receipt.json"
        ),
        "outcome_rows_read": 0,
        "performance_or_pnl_computed": False,
        "target_assets_loaded": [],
    }
    previous = _previous_selected(context["output"], feature_date)
    policy = context["protocol"]["policy"]
    costs = list(policy["reporting_cost_bps"])
    position_folds = []
    for fold_prediction in folds:
        fold = int(fold_prediction["fold"])
        cells = [
            _policy_step(
                fold_prediction,
                previous[(fold, int(cost))],
                cost_bps=int(cost),
                policy=policy,
            )
            for cost in costs
        ]
        position_folds.append(
            {
                "fold": fold,
                "registered_symbols": fold_prediction["registered_symbols"],
                "eligible_symbols": fold_prediction["eligible_symbols"],
                "missing_symbols": fold_prediction["missing_symbols"],
                "cost_cells": cells,
            }
        )
    position = {
        "schema_version": "v70-r1-daily-position-packet/v1",
        "feature_date": source_packet["feature_date"],
        "eligible_action_date": (
            feature_date + pd.Timedelta(days=1)
        ).date().isoformat(),
        "freeze_timestamp_utc": source_packet["freeze_timestamp_utc"],
        "target_h1_maturity_timestamp_utc": source_packet[
            "target_h1_maturity_timestamp_utc"
        ],
        "prediction_packet_sha256": _sha256_bytes(_json_bytes(prediction)),
        "policy_sha256": canonical_sha256(policy),
        "folds": position_folds,
        "final_liquidation_applied": False,
        "outcome_rows_read": 0,
        "performance_or_pnl_computed": False,
        "target_assets_loaded": [],
    }
    return prediction, position


def _daily_paths(output: Path, feature_date: pd.Timestamp) -> dict[str, Path]:
    root = output / "daily" / feature_date.date().isoformat()
    return {
        "root": root,
        "source": root / "source_packet.json",
        "prediction": root / "prediction_packet.json",
        "position": root / "position_packet.json",
        "reconciliation": root / "archive_reconciliation.json",
        "capture_ledger": output / "ledger" / "capture" / f"{feature_date.date().isoformat()}.json",
        "prediction_ledger": output / "ledger" / "prediction" / f"{feature_date.date().isoformat()}.json",
    }


def _receipt_path(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _canonical_archive_row(row: list[Any]) -> tuple[Any, ...]:
    if len(row) != KLINE_WIDTH:
        raise V70CaptureError("Binance archive row width drift")
    return (
        _normalize_epoch_ms(row[0]),
        *(str(Decimal(str(row[index])).normalize()) for index in (1, 2, 3, 4, 5)),
        _normalize_epoch_ms(row[6]),
        str(Decimal(str(row[7])).normalize()),
        int(row[8]),
        str(Decimal(str(row[9])).normalize()),
        str(Decimal(str(row[10])).normalize()),
        str(row[11]),
    )


def _try_archive_reconciliation(
    context: Mapping[str, Any],
    source_packet: Mapping[str, Any],
    path: Path,
    now: pd.Timestamp,
    fetch: Callable[[str], bytes],
) -> str:
    def fail_integrity(symbol: str, reason: str) -> None:
        failure = {
            "schema_version": "v70-r1-source-integrity-failure/v1",
            "feature_date": source_packet["feature_date"],
            "symbol": symbol,
            "reason": reason,
            "source_packet_sha256": _sha256_bytes(_json_bytes(source_packet)),
            "original_source_packet_preserved": True,
            "outcome_rows_read": 0,
            "target_assets_loaded": [],
        }
        _write_once_json(path.with_name("archive_integrity_failure.json"), failure)
        raise V70CaptureError(reason)

    if path.is_file():
        return "complete"
    feature_date = pd.Timestamp(source_packet["feature_date"], tz="UTC")
    if now < feature_date + pd.Timedelta(days=3):
        return "pending"
    source_rows = {
        symbol: rows[-1] for symbol, rows in source_packet["rows_by_symbol"].items()
    }
    receipts = []
    base = str(context["spec"]["public_archive_base_url"]).rstrip("/")
    for symbol in sorted(source_rows):
        name = f"{symbol}-1d-{feature_date.date().isoformat()}.zip"
        url = f"{base}/data/spot/daily/klines/{symbol}/1d/{name}"
        try:
            checksum_payload = fetch(f"{url}.CHECKSUM")
            archive_payload = fetch(url)
        except V70CaptureError:
            return "pending"
        published = checksum_payload.decode("utf-8").strip().split()[0]
        archive_sha = _sha256_bytes(archive_payload)
        if published != archive_sha:
            fail_integrity(symbol, f"Binance archive checksum mismatch for {symbol}")
        with zipfile.ZipFile(io.BytesIO(archive_payload)) as archive:
            names = [name for name in archive.namelist() if name.endswith(".csv")]
            if len(names) != 1:
                fail_integrity(symbol, f"Binance archive member drift for {symbol}")
            decoded = archive.read(names[0]).decode("utf-8")
        rows = list(csv.reader(io.StringIO(decoded)))
        if rows and rows[0] and rows[0][0].strip().lower().replace(" ", "_") == "open_time":
            rows = rows[1:]
        if len(rows) != 1:
            fail_integrity(
                symbol, f"Binance daily archive row count drift for {symbol}"
            )
        if _canonical_archive_row(rows[0]) != _canonical_archive_row(source_rows[symbol]):
            fail_integrity(symbol, f"Public REST/archive source revision for {symbol}")
        receipts.append(
            {
                "symbol": symbol,
                "archive_url": url,
                "archive_sha256": archive_sha,
                "checksum_payload_sha256": _sha256_bytes(checksum_payload),
                "source_row_matches": True,
            }
        )
    packet = {
        "schema_version": "v70-r1-archive-reconciliation/v1",
        "feature_date": source_packet["feature_date"],
        "source_packet_sha256": _sha256_bytes(_json_bytes(source_packet)),
        "symbols": receipts,
        "all_public_rest_rows_match_published_archives": True,
        "outcome_rows_read": 0,
        "target_assets_loaded": [],
    }
    _write_once_json(path, packet)
    return "complete"


def _cumulative_status(
    context: Mapping[str, Any], now: pd.Timestamp
) -> dict[str, Any]:
    output = context["output"]
    fold_counts = {
        fold: {"eligible_signal_dates": 0, "fully_matured_dates": 0, "active_position_dates": 0}
        for fold in EXPECTED_FOLDS
    }
    packet_hashes = []
    reconciliation_complete = 0
    dates = []
    for daily in (
        sorted((output / "daily").glob("????-??-??"))
        if (output / "daily").is_dir()
        else []
    ):
        source_path = daily / "source_packet.json"
        prediction_path = daily / "prediction_packet.json"
        position_path = daily / "position_packet.json"
        if not (source_path.is_file() and prediction_path.is_file() and position_path.is_file()):
            continue
        source = _load_json(source_path)
        position = _load_json(position_path)
        date = str(source["feature_date"])
        dates.append(date)
        matured = now >= pd.Timestamp(source["target_h1_maturity_timestamp_utc"])
        for fold_row in position["folds"]:
            fold = int(fold_row["fold"])
            fold_counts[fold]["eligible_signal_dates"] += 1
            fold_counts[fold]["fully_matured_dates"] += int(matured)
            base_cell = next(
                row
                for row in fold_row["cost_cells"]
                if int(row["cost_bps"]) == int(context["protocol"]["policy"]["base_cost_bps"])
            )
            fold_counts[fold]["active_position_dates"] += int(
                base_cell["selected_symbol"] is not None
            )
        reconciliation_complete += int((daily / "archive_reconciliation.json").is_file())
        packet_hashes.append(
            {
                "feature_date": date,
                "source_sha256": file_sha256(source_path),
                "prediction_sha256": file_sha256(prediction_path),
                "position_sha256": file_sha256(position_path),
                "archive_reconciliation_sha256": (
                    file_sha256(daily / "archive_reconciliation.json")
                    if (daily / "archive_reconciliation.json").is_file()
                    else None
                ),
            }
        )
    anchor_timestamp = pd.Timestamp(context["anchor"]["commit_timestamp_utc"])
    calendar_days = max(0, int((now.floor("D") - anchor_timestamp.floor("D")).days))
    maturity = context["contract"]["maturity_contract"]
    minimums_met = (
        calendar_days >= int(maturity["minimum_calendar_days"])
        and all(
            values["eligible_signal_dates"]
            >= int(maturity["minimum_eligible_signal_dates_per_fold"])
            and values["fully_matured_dates"]
            >= int(maturity["minimum_fully_matured_signal_dates_per_fold"])
            and values["active_position_dates"]
            >= int(maturity["minimum_active_position_days_per_fold"])
            for values in fold_counts.values()
        )
        and reconciliation_complete == len(packet_hashes)
        and bool(packet_hashes)
    )
    expired = calendar_days >= int(maturity["maximum_calendar_days"]) and not minimums_met
    return {
        "schema_version": "v70-r1-cumulative-status/v1",
        "as_of_timestamp_utc": now.isoformat(),
        "registration_anchor": context["anchor"],
        "calendar_days": calendar_days,
        "packet_count": len(packet_hashes),
        "first_feature_date": min(dates) if dates else None,
        "last_feature_date": max(dates) if dates else None,
        "fold_counts": {str(key): value for key, value in fold_counts.items()},
        "archive_reconciliation_complete_count": reconciliation_complete,
        "packet_hashes": packet_hashes,
        "minimums_met": minimums_met,
        "maximum_clock_expired": expired,
        "decision": (
            context["contract"]["pass_action"]
            if minimums_met
            else context["contract"]["failure_action"]
            if expired
            else context["contract"]["interim_action"]
        ),
        "outcome_rows_read": 0,
        "performance_or_pnl_computed": False,
        "target_assets_loaded": [],
    }


def _run_v64_r2_prospective_capture_locked(
    config: dict[str, Any],
    *,
    now_utc: datetime | pd.Timestamp | None = None,
    fetcher: Callable[[str], bytes] | None = None,
    device_override: str | None = None,
    require_clean: bool | None = None,
) -> dict[str, Any]:
    now = _as_utc(now_utc)
    configured_clean = bool(
        config.get("v64_r2_prospective_capture", {}).get("require_clean_git", True)
    )
    context = _context(
        config,
        require_clean=configured_clean if require_clean is None else bool(require_clean),
    )
    output = context["output"]
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / "source_code_receipt.json", context["source_code_receipt"])
    device_name = device_override or str(context["spec"]["device"])
    device = configure_v68_runtime(device_name, seed=int(config.get("seed", 20260715)))
    models, checkpoint_receipts = _load_inference_models(context, device)
    preflight = {
        "schema_version": "v70-r1-preflight/v1",
        "phase_contract_sha256": file_sha256(context["contract_path"]),
        "registration_anchor": context["anchor"],
        "static_input_count": len(context["static_input_sha256"]),
        "checkpoint_count": len(checkpoint_receipts),
        "checkpoint_receipts": checkpoint_receipts,
        "fold_count": 3,
        "non_target_symbol_count": 30,
        "device": str(device),
        "optimizer_created": False,
        "training_or_refit_performed": False,
        "outcome_rows_read": 0,
        "performance_or_pnl_computed": False,
        "target_assets_loaded": [],
        "passed": True,
    }
    write_json_atomic(output / "preflight.json", preflight)
    feature_date = latest_capture_candidate(
        now, context["anchor"]["commit_timestamp_utc"]
    )
    fetch = fetcher or PublicHTTPFetcher(int(context["spec"]["request_timeout_seconds"]))
    operation = "waiting_for_first_post_amendment_feature_close"
    if feature_date is not None:
        paths = _daily_paths(output, feature_date)
        if paths["position"].is_file():
            operation = "already_frozen_idempotent_replay"
        else:
            source_packet = (
                _load_json(paths["source"])
                if paths["source"].is_file()
                else _fetch_source_packet(context, feature_date, now, fetch)
            )
            source_sha = _write_once_json(paths["source"], source_packet)
            capture_ledger = {
                "schema_version": "v70-r1-capture-ledger-entry/v1",
                "feature_date": source_packet["feature_date"],
                "source_packet_path": _receipt_path(context["root"], paths["source"]),
                "source_packet_sha256": source_sha,
                "freeze_timestamp_utc": source_packet["freeze_timestamp_utc"],
                "target_h1_maturity_timestamp_utc": source_packet[
                    "target_h1_maturity_timestamp_utc"
                ],
                "outcome_rows_read": 0,
                "target_assets_loaded": [],
            }
            _write_once_json(paths["capture_ledger"], capture_ledger)
            if any(
                row.get("eligible") is True
                for row in source_packet.get("fold_readiness", [])
            ):
                prediction, position = _prediction_and_position_packets(
                    context,
                    source_packet,
                    checkpoint_receipts,
                    models,
                    device,
                )
                prediction_sha = _write_once_json(paths["prediction"], prediction)
                position_sha = _write_once_json(paths["position"], position)
                ledger = {
                    "schema_version": "v70-r1-prediction-ledger-entry/v1",
                    "feature_date": source_packet["feature_date"],
                    "prediction_packet_path": _receipt_path(
                        context["root"], paths["prediction"]
                    ),
                    "prediction_packet_sha256": prediction_sha,
                    "position_packet_path": _receipt_path(
                        context["root"], paths["position"]
                    ),
                    "position_packet_sha256": position_sha,
                    "freeze_precedes_maturity": pd.Timestamp(
                        source_packet["freeze_timestamp_utc"]
                    )
                    < pd.Timestamp(source_packet["target_h1_maturity_timestamp_utc"]),
                    "outcome_rows_read": 0,
                    "target_assets_loaded": [],
                }
                _write_once_json(paths["prediction_ledger"], ledger)
                operation = "captured_and_frozen"
            else:
                operation = "signal_date_omitted_source_missing_or_late"
    if (output / "daily").is_dir():
        for daily in sorted((output / "daily").glob("????-??-??")):
            source_path = daily / "source_packet.json"
            if not source_path.is_file() or not (daily / "position_packet.json").is_file():
                continue
            source_packet = _load_json(source_path)
            _try_archive_reconciliation(
                context,
                source_packet,
                daily / "archive_reconciliation.json",
                now,
                fetch,
            )
    for fold_models in models.values():
        for _, ranker, gate in fold_models:
            ranker.to("cpu")
            gate.to("cpu")
    if device.type == "mps":
        torch.mps.empty_cache()
    gc.collect()
    cumulative = _cumulative_status(context, now)
    cumulative["operation"] = operation
    write_json_atomic(output / "cumulative_manifest.json", cumulative)
    return {
        "decision": cumulative["decision"],
        "operation": operation,
        "registration_anchor": context["anchor"],
        "preflight_passed": True,
        "checkpoint_count": len(checkpoint_receipts),
        "packet_count": cumulative["packet_count"],
        "fold_counts": cumulative["fold_counts"],
        "minimums_met": cumulative["minimums_met"],
        "maximum_clock_expired": cumulative["maximum_clock_expired"],
        "outcome_rows_read": 0,
        "performance_or_pnl_computed": False,
        "target_assets_loaded": [],
    }


def run_v64_r2_prospective_capture(
    config: dict[str, Any],
    *,
    now_utc: datetime | pd.Timestamp | None = None,
    fetcher: Callable[[str], bytes] | None = None,
    device_override: str | None = None,
    require_clean: bool | None = None,
) -> dict[str, Any]:
    spec = config.get("v64_r2_prospective_capture", {})
    root = Path(spec.get("project_root", ".")).resolve()
    output = root / str(config.get("output_dir", ""))
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".capture.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise V70CaptureError("Another V70 capture process is already active") from exc
        return _run_v64_r2_prospective_capture_locked(
            config,
            now_utc=now_utc,
            fetcher=fetcher,
            device_override=device_override,
            require_clean=require_clean,
        )
