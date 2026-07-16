from __future__ import annotations

import copy
import fcntl
import hashlib
import io
import json
import math
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import zipfile

import numpy as np
import pandas as pd
import pytest
import torch

from tlm.config import load_config
from tlm.v64_r2_probabilistic_state_gate_harness import (
    probabilistic_rank_state_positions,
)
from tlm.v64_r2_prospective_capture import (
    V70CaptureError,
    _context,
    _feature_panel,
    _fetch_source_packet,
    _load_inference_models,
    _policy_step,
    _try_archive_reconciliation,
    _write_once_json,
    latest_capture_candidate,
    resolve_registration_anchor,
    run_v64_r2_prospective_capture,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/v70_v64_r2_prospective_capture.yaml"


def _config(output: Path | None = None) -> dict:
    config = copy.deepcopy(load_config(CONFIG))
    config["v64_r2_prospective_capture"]["project_root"] = str(ROOT)
    if output is not None:
        config["output_dir"] = str(output)
    return config


def _daily_rows(symbol: str, start_ms: int, end_ms: int) -> list[list[object]]:
    start = pd.Timestamp(start_ms, unit="ms", tz="UTC").floor("D")
    end = pd.Timestamp(end_ms, unit="ms", tz="UTC").floor("D")
    offset = int(hashlib.sha256(symbol.encode()).hexdigest()[:6], 16) % 1000
    rows = []
    for index, date in enumerate(pd.date_range(start, end, freq="D", tz="UTC")):
        level = 50.0 + offset / 100.0 + index * 0.025
        opening = level * (1.0 + 0.002 * math.sin(index / 11.0))
        close = level * (1.0 + 0.003 * math.cos(index / 7.0))
        high = max(opening, close) * 1.01
        low = min(opening, close) * 0.99
        volume = 1000.0 + offset + index
        quote = volume * close
        open_ms = int(date.timestamp() * 1000)
        close_ms = int((date + pd.Timedelta(days=1)).timestamp() * 1000) - 1
        rows.append(
            [
                open_ms,
                f"{opening:.12f}",
                f"{high:.12f}",
                f"{low:.12f}",
                f"{close:.12f}",
                f"{volume:.8f}",
                close_ms,
                f"{quote:.8f}",
                1000 + index,
                f"{volume / 2:.8f}",
                f"{quote / 2:.8f}",
                "0",
            ]
        )
    return rows


class FakeBinance:
    def __init__(self) -> None:
        self.rest_rows: dict[str, list[list[object]]] = {}
        self.archive_payloads: dict[str, bytes] = {}

    def __call__(self, url: str) -> bytes:
        parsed = urlparse(url)
        if parsed.path.endswith("/api/v3/klines"):
            query = parse_qs(parsed.query)
            symbol = query["symbol"][0]
            rows = _daily_rows(
                symbol, int(query["startTime"][0]), int(query["endTime"][0])
            )
            self.rest_rows[symbol] = rows
            return json.dumps(rows, separators=(",", ":")).encode()
        archive_url = url.removesuffix(".CHECKSUM")
        payload = self.archive_payloads[archive_url]
        if url.endswith(".CHECKSUM"):
            return f"{hashlib.sha256(payload).hexdigest()}  file.zip\n".encode()
        return payload

    def prepare_archives(self, feature_date: str) -> None:
        for symbol, rows in self.rest_rows.items():
            row = next(
                value
                for value in rows
                if pd.Timestamp(int(value[0]), unit="ms", tz="UTC").date().isoformat()
                == feature_date
            )
            name = f"{symbol}-1d-{feature_date}.zip"
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                info = zipfile.ZipInfo(f"{symbol}-1d-{feature_date}.csv")
                info.date_time = (2020, 1, 1, 0, 0, 0)
                archive.writestr(info, ",".join(str(value) for value in row) + "\n")
            url = (
                "https://data.binance.vision/data/spot/daily/klines/"
                f"{symbol}/1d/{name}"
            )
            self.archive_payloads[url] = buffer.getvalue()


def test_v70_r1_anchor_and_candidate_are_strictly_post_commit() -> None:
    contract = load_config(ROOT / "research/phase_contracts/v070.yaml")
    anchor = resolve_registration_anchor(ROOT, contract["registration_anchor_contract"])
    assert anchor["commit"] == "fe5001180a71f63dfb4d7e5b611dd9704256afee"
    timestamp = pd.Timestamp(anchor["commit_timestamp_utc"])
    assert latest_capture_candidate("2026-07-15T23:59:59Z", timestamp) is None
    assert latest_capture_candidate("2026-07-16T00:04:59Z", timestamp) is None
    candidate = latest_capture_candidate("2026-07-16T00:05:00Z", timestamp)
    assert candidate == pd.Timestamp("2026-07-15", tz="UTC")
    assert candidate + pd.Timedelta(days=1) > timestamp


def test_v70_source_and_feature_pipeline_are_feature_only() -> None:
    context = _context(_config(), require_clean=False)
    fetcher = FakeBinance()
    packet = _fetch_source_packet(
        context,
        pd.Timestamp("2026-07-15", tz="UTC"),
        pd.Timestamp("2026-07-16T00:05:30Z"),
        fetcher,
    )
    panel = _feature_panel(packet)

    assert packet["complete"] is True
    assert len(packet["responses"]) == 30
    assert len(packet["rows_by_symbol"]) == 30
    assert packet["outcome_rows_read"] == 0
    assert packet["target_assets_loaded"] == []
    assert len(panel) == 30 * 256
    assert set(panel["symbol"]) == set(context["symbols"])
    assert not any(
        token in column.lower()
        for column in panel.columns
        for token in ("target", "label", "outcome")
    )


def test_v70_registered_missingness_preserves_available_fold_contexts() -> None:
    context = _context(_config(), require_clean=False)
    base = FakeBinance()
    missing = {"MATICUSDT", "EOSUSDT", "FTMUSDT"}

    def fetch(url: str) -> bytes:
        query = parse_qs(urlparse(url).query)
        if query.get("symbol", [None])[0] in missing:
            return b"[]"
        return base(url)

    packet = _fetch_source_packet(
        context,
        pd.Timestamp("2026-07-15", tz="UTC"),
        pd.Timestamp("2026-07-16T00:05:30Z"),
        fetch,
    )
    readiness = {int(row["fold"]): row for row in packet["fold_readiness"]}
    panel = _feature_panel(packet)

    assert packet["complete"] is False
    assert len(packet["rows_by_symbol"]) == 27
    assert len(packet["omissions"]) == 3
    assert readiness[1]["eligible_symbols"] == [
        symbol
        for symbol in readiness[1]["registered_symbols"]
        if symbol != "MATICUSDT"
    ]
    assert len(readiness[2]["eligible_symbols"]) == 8
    assert len(readiness[3]["eligible_symbols"]) == 10
    assert all(row["eligible"] for row in readiness.values())
    assert len(panel) == 27 * 256


def test_v70_archive_reconciliation_matches_the_frozen_rest_row(tmp_path: Path) -> None:
    context = _context(_config(), require_clean=False)
    fetcher = FakeBinance()
    packet = _fetch_source_packet(
        context,
        pd.Timestamp("2026-07-15", tz="UTC"),
        pd.Timestamp("2026-07-16T00:05:30Z"),
        fetcher,
    )
    fetcher.prepare_archives("2026-07-15")
    path = tmp_path / "archive_reconciliation.json"
    status = _try_archive_reconciliation(
        context,
        packet,
        path,
        pd.Timestamp("2026-07-18T00:05:30Z"),
        fetcher,
    )
    receipt = json.loads(path.read_text())

    assert status == "complete"
    assert receipt["all_public_rest_rows_match_published_archives"] is True
    assert len(receipt["symbols"]) == 30
    assert receipt["outcome_rows_read"] == 0


def test_v70_policy_step_matches_the_frozen_vector_policy() -> None:
    symbols = [f"A{index}" for index in range(3)]
    excess = np.asarray(
        [[0.001, 0.004, -0.005], [0.000, 0.005, -0.005], [0.003, -0.002, -0.001]],
        dtype=np.float64,
    )
    momentum = np.ones_like(excess)
    eligible = np.ones_like(excess, dtype=bool)
    locations = np.asarray([[0.008, 0.010, 0.012]] * 3, dtype=np.float64)
    scales = np.asarray([[0.005, 0.006, 0.007]] * 3, dtype=np.float64)
    expected = probabilistic_rank_state_positions(
        excess,
        locations,
        scales,
        momentum,
        eligible,
        base_cost=0.001,
        switch_hurdle=0.002,
        probability_threshold=0.6,
        degrees_of_freedom=5.0,
    )
    policy = {
        "abstention_probability_threshold": 0.6,
        "switch_hurdle": 0.002,
        "risky_gross": 1.0,
    }
    previous = None
    observed_actions = []
    observed_selected = []
    for day in range(3):
        fold = {
            "registered_symbols": symbols,
            "eligible_symbols": symbols,
            "assets": [
                {
                    "symbol": symbol,
                    "raw_excess": float(excess[day, index]),
                    "momentum_30": float(momentum[day, index]),
                }
                for index, symbol in enumerate(symbols)
            ],
            "market_mixture": [
                {"market_location": float(location), "market_scale": float(scale)}
                for location, scale in zip(locations[day], scales[day])
            ],
        }
        step = _policy_step(fold, previous, cost_bps=10, policy=policy)
        previous = step["selected_symbol"]
        observed_actions.append(step["action"])
        observed_selected.append(
            symbols.index(previous) if previous is not None else None
        )
    assert observed_actions == expected["actions"]
    assert observed_selected == expected["selected_assets"]


def test_v70_immutable_writer_rejects_rewrite(tmp_path: Path) -> None:
    path = tmp_path / "packet.json"
    first = _write_once_json(path, {"value": 1})
    second = _write_once_json(path, {"value": 1})
    assert first == second
    with pytest.raises(V70CaptureError, match="Immutable V70 artifact drift"):
        _write_once_json(path, {"value": 2})


def test_v70_rejects_concurrent_capture_process(tmp_path: Path) -> None:
    output = tmp_path / "capture"
    output.mkdir()
    config = _config(output)
    with (output / ".capture.lock").open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(V70CaptureError, match="already active"):
            run_v64_r2_prospective_capture(
                config,
                now_utc=pd.Timestamp("2026-07-15T23:00:00Z"),
                device_override="cpu",
                require_clean=False,
            )


def test_v70_all_nine_checkpoints_load_without_selection() -> None:
    context = _context(_config(), require_clean=False)
    models, receipts = _load_inference_models(context, torch.device("cpu"))
    assert len(receipts) == 9
    assert {(row["fold"], row["seed"]) for row in receipts} == {
        (fold, seed) for fold in (1, 2, 3) for seed in (42, 7, 123)
    }
    assert all(row["used_without_selection"] for row in receipts)
    assert all(row["optimizer_created"] is False for row in receipts)
    assert all(
        not any(parameter.requires_grad for parameter in ranker.parameters())
        and not any(parameter.requires_grad for parameter in gate.parameters())
        for fold_models in models.values()
        for _, ranker, gate in fold_models
    )


def test_v70_full_fake_daily_capture_is_idempotent_and_outcome_blind(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "capture")
    fetcher = FakeBinance()
    result = run_v64_r2_prospective_capture(
        config,
        now_utc=pd.Timestamp("2026-07-16T00:05:30Z"),
        fetcher=fetcher,
        device_override="cpu",
        require_clean=False,
    )
    daily = tmp_path / "capture/daily/2026-07-15"
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in daily.iterdir()
        if path.is_file()
    }
    replay = run_v64_r2_prospective_capture(
        config,
        now_utc=pd.Timestamp("2026-07-16T00:05:30Z"),
        fetcher=fetcher,
        device_override="cpu",
        require_clean=False,
    )
    replay_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in daily.iterdir()
        if path.is_file()
    }
    prediction = json.loads((daily / "prediction_packet.json").read_text())
    position = json.loads((daily / "position_packet.json").read_text())

    assert result["operation"] == "captured_and_frozen"
    assert replay["operation"] == "already_frozen_idempotent_replay"
    assert hashes == replay_hashes
    assert result["packet_count"] == replay["packet_count"] == 1
    assert len(prediction["folds"]) == 3
    assert all(row["triplet_context_count"] == 120 for row in prediction["folds"])
    assert all(row["market_mixture_component_count"] == 360 for row in prediction["folds"])
    assert all(len(row["cost_cells"]) == 3 for row in position["folds"])
    assert result["outcome_rows_read"] == 0
    assert result["performance_or_pnl_computed"] is False
    assert result["target_assets_loaded"] == []
