"""
Shared fixtures for DeepAlpha test suite.
"""

import sys
import os
import numpy as np
import pytest

# Ensure repo root is on sys.path so we can import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def sample_ohlcv():
    """
    Generate 100 candles of synthetic OHLCV data.

    Returns a dict with keys: open, high, low, close, volume.
    Prices simulate a random walk around 50000 (BTC-like).
    """
    np.random.seed(42)
    n = 100
    base = 50000.0
    returns = np.random.normal(0, 0.005, n)
    close = base * np.cumprod(1 + returns)
    open_ = np.roll(close, 1)
    open_[0] = base
    high = np.maximum(open_, close) * (1 + np.random.uniform(0, 0.003, n))
    low = np.minimum(open_, close) * (1 - np.random.uniform(0, 0.003, n))
    volume = np.random.uniform(100, 5000, n)

    return {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


@pytest.fixture
def btc_close(sample_ohlcv):
    """BTC close prices aligned with sample_ohlcv (just the close array)."""
    return sample_ohlcv["close"].copy()


@pytest.fixture
def mock_config(monkeypatch):
    """
    Patch config module with known test values.
    """
    import config

    monkeypatch.setattr(config, "LEVERAGE", 5)
    monkeypatch.setattr(config, "MAX_POSITIONS", 3)
    monkeypatch.setattr(config, "RISK_PER_TRADE", 0.10)
    monkeypatch.setattr(config, "MAX_DAILY_LOSS_PCT", 0.05)
    monkeypatch.setattr(config, "STOP_LOSS_PCT", 0.02)
    monkeypatch.setattr(config, "TAKE_PROFIT_PCT", 0.03)
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_LOSSES", 3)
    monkeypatch.setattr(config, "CIRCUIT_BREAKER_COOLDOWN", 3600)

    return config
