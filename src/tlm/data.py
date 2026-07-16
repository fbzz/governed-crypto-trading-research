from __future__ import annotations

import json
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _verified_ssl_context() -> ssl.SSLContext:
    """Use a real system CA bundle when the Python.org build has none."""
    paths = ssl.get_default_verify_paths()
    candidates = [
        paths.cafile,
        paths.openssl_cafile,
        "/etc/ssl/cert.pem",
        "/etc/ssl/certs/ca-certificates.crt",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return ssl.create_default_context(cafile=candidate)
    return ssl.create_default_context()


def _to_milliseconds(value: str | pd.Timestamp | None) -> int | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value, tz="UTC")
    return int(timestamp.timestamp() * 1000)


def download_binance_klines(
    symbol: str,
    interval: str = "1d",
    start: str | None = None,
    end: str | None = None,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """Download public spot candles without credentials, using pagination."""
    endpoint = "https://api.binance.com/api/v3/klines"
    cursor = _to_milliseconds(start) or 0
    end_ms = _to_milliseconds(end)
    rows: list[list[object]] = []

    while True:
        params: dict[str, object] = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "limit": 1000,
        }
        if end_ms is not None:
            params["endTime"] = end_ms
        request = urllib.request.Request(
            f"{endpoint}?{urllib.parse.urlencode(params)}",
            headers={"User-Agent": "tlm-research/0.1"},
        )
        with urllib.request.urlopen(
            request, timeout=timeout, context=_verified_ssl_context()
        ) as response:
            page = json.loads(response.read().decode("utf-8"))
        if not isinstance(page, list):
            raise RuntimeError(f"Unexpected Binance response for {symbol}: {page}")
        if not page:
            break
        rows.extend(page)
        next_cursor = int(page[-1][0]) + 1
        if len(page) < 1000 or next_cursor <= cursor:
            break
        cursor = next_cursor
        time.sleep(0.05)

    if not rows:
        raise RuntimeError(f"No candles returned for {symbol}")
    frame = pd.DataFrame(
        rows,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trade_count", "taker_base",
            "taker_quote", "ignore",
        ],
    )
    frame["timestamp"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    frame = frame.set_index("timestamp")[OHLCV_COLUMNS].astype(float)
    return validate_candles(frame, symbol)


def generate_fixture(
    assets: list[str], days: int = 520, seed: int = 42
) -> dict[str, pd.DataFrame]:
    """Generate correlated, non-trivial daily candles for offline tests."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("2020-01-01", periods=days, freq="D", tz="UTC")
    market = rng.normal(0.0005, 0.025, days)
    result: dict[str, pd.DataFrame] = {}
    initial_prices = {"BTC": 9000.0, "ETH": 220.0, "SOL": 2.0}

    for offset, asset in enumerate(assets):
        idiosyncratic = rng.normal(0.0, 0.012 + offset * 0.003, days)
        overnight = rng.normal(0.0, 0.006, days)
        intraday = 0.72 * market + idiosyncratic
        previous_close = initial_prices.get(asset, 100.0)
        records: list[tuple[float, float, float, float, float]] = []
        for day in range(days):
            open_price = previous_close * np.exp(overnight[day])
            close_price = open_price * np.exp(intraday[day])
            spread = abs(rng.normal(0.014, 0.006))
            high = max(open_price, close_price) * (1.0 + spread)
            low = min(open_price, close_price) / (1.0 + spread)
            volume = np.exp(12.0 + 3.0 * abs(intraday[day]) + rng.normal(0, 0.25))
            records.append((open_price, high, low, close_price, volume))
            previous_close = close_price
        result[asset] = pd.DataFrame(records, index=index, columns=OHLCV_COLUMNS)
    return result


def validate_candles(frame: pd.DataFrame, symbol: str = "asset") -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{symbol}: index must be a DatetimeIndex")
    if frame.index.tz is None:
        raise ValueError(f"{symbol}: timestamps must be timezone-aware")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{symbol}: timestamps must be unique and sorted")
    if len(frame) > 1 and not (frame.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError(f"{symbol}: daily candle sequence contains gaps")
    missing = set(OHLCV_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{symbol}: missing columns {sorted(missing)}")
    values = frame[OHLCV_COLUMNS]
    if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"{symbol}: OHLCV contains invalid values")
    if (values <= 0).any().any():
        raise ValueError(f"{symbol}: OHLCV values must be positive")
    if (frame["high"] < frame[["open", "close"]].max(axis=1)).any():
        raise ValueError(f"{symbol}: high is below open/close")
    if (frame["low"] > frame[["open", "close"]].min(axis=1)).any():
        raise ValueError(f"{symbol}: low is above open/close")
    return frame[OHLCV_COLUMNS].copy()


def drop_unclosed_daily_candle(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only candles whose UTC trading day has fully closed."""
    today_utc = pd.Timestamp.now(tz="UTC").floor("D")
    return frame.loc[frame.index < today_utc].copy()


def align_assets(frames: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if not frames:
        raise ValueError("At least one asset is required")
    common = None
    for asset, frame in frames.items():
        validated = validate_candles(frame, asset)
        common = validated.index if common is None else common.intersection(validated.index)
    assert common is not None
    common = common.sort_values()
    if len(common) < 100:
        raise ValueError(f"Only {len(common)} common daily candles across assets")
    return {asset: validate_candles(frame, asset).loc[common] for asset, frame in frames.items()}


def load_market_data(config: dict, force: bool = False) -> dict[str, pd.DataFrame]:
    data_config = config["data"]
    assets = data_config["assets"]
    if data_config.get("interval", "1d") != "1d":
        raise ValueError("The MVP supports daily candles only")
    if data_config.get("source") == "fixture":
        return align_assets(generate_fixture(
            list(assets),
            days=int(data_config.get("fixture_days", 520)),
            seed=int(config.get("seed", 42)),
        ))
    if data_config.get("source") != "binance":
        raise ValueError(f"Unsupported data source: {data_config.get('source')}")

    cache_dir = Path(data_config.get("cache_dir", "data/raw"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    for asset, symbol in assets.items():
        cache_path = cache_dir / f"{symbol}_{data_config.get('interval', '1d')}.parquet"
        if cache_path.exists() and not force:
            frame = pd.read_parquet(cache_path)
            frame.index = pd.DatetimeIndex(frame.index)
            frame = drop_unclosed_daily_candle(frame)
            latest_closed_day = pd.Timestamp.now(tz="UTC").floor("D") - pd.Timedelta(days=1)
            cache_is_fresh = data_config.get("end") is not None or frame.index.max() >= latest_closed_day
            if cache_is_fresh:
                frames[asset] = validate_candles(frame, asset)
                continue
        frame = download_binance_klines(
            symbol=symbol,
            interval=data_config.get("interval", "1d"),
            start=data_config.get("start"),
            end=data_config.get("end"),
        )
        frame = drop_unclosed_daily_candle(frame)
        frame.to_parquet(cache_path)
        frames[asset] = frame
    return align_assets(frames)
