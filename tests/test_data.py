import ssl

import pandas as pd
import pytest

from tlm.data import _verified_ssl_context, align_assets, generate_fixture, validate_candles


def test_fixture_is_deterministic_and_valid():
    first = generate_fixture(["BTC", "ETH", "SOL"], days=160, seed=7)
    second = generate_fixture(["BTC", "ETH", "SOL"], days=160, seed=7)
    pd.testing.assert_frame_equal(first["BTC"], second["BTC"])
    aligned = align_assets(first)
    assert set(aligned) == {"BTC", "ETH", "SOL"}
    assert all(len(frame) == 160 for frame in aligned.values())


def test_validation_rejects_duplicate_timestamps():
    frame = generate_fixture(["BTC"], days=120)["BTC"]
    duplicate = pd.concat([frame, frame.iloc[[-1]]])
    with pytest.raises(ValueError, match="unique and sorted"):
        validate_candles(duplicate, "BTC")


def test_validation_rejects_missing_daily_candle():
    frame = generate_fixture(["BTC"], days=120)["BTC"].drop(
        generate_fixture(["BTC"], days=120)["BTC"].index[60]
    )
    with pytest.raises(ValueError, match="contains gaps"):
        validate_candles(frame, "BTC")


def test_alignment_uses_only_common_timestamps():
    frames = generate_fixture(["BTC", "ETH"], days=160)
    frames["ETH"] = frames["ETH"].iloc[5:]
    aligned = align_assets(frames)
    assert len(aligned["BTC"]) == 155
    assert aligned["BTC"].index.equals(aligned["ETH"].index)


def test_network_context_keeps_certificate_verification_enabled():
    context = _verified_ssl_context()
    assert context.check_hostname is True
    assert context.verify_mode == ssl.CERT_REQUIRED
