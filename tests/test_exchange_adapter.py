"""
Tests for DeepAlpha exchange adapter layer (exchange_adapter.py).

These tests cover the factory function, adapter initialization, and symbol
formatting. They do NOT require network access or API keys.
"""

import pytest

from exchange_adapter import (
    get_exchange,
    HyperliquidAdapter,
    BinanceAdapter,
    BybitAdapter,
    ExchangeAdapter,
    _parse_interval,
    _round_price,
)


class TestGetExchangeFactory:
    """Tests for the get_exchange() factory function."""

    def test_returns_hyperliquid_adapter(self):
        adapter = get_exchange("hyperliquid")
        assert isinstance(adapter, HyperliquidAdapter)

    def test_returns_binance_adapter(self):
        adapter = get_exchange("binance")
        assert isinstance(adapter, BinanceAdapter)

    def test_returns_bybit_adapter(self):
        adapter = get_exchange("bybit")
        assert isinstance(adapter, BybitAdapter)

    def test_case_insensitive(self):
        adapter = get_exchange("Hyperliquid")
        assert isinstance(adapter, HyperliquidAdapter)

    def test_strips_whitespace(self):
        adapter = get_exchange("  binance  ")
        assert isinstance(adapter, BinanceAdapter)

    def test_unknown_exchange_raises(self):
        with pytest.raises(ValueError, match="Unknown exchange"):
            get_exchange("kraken")

    def test_all_adapters_are_exchange_adapter(self):
        """Every adapter returned by get_exchange should be an ExchangeAdapter."""
        for name in ("hyperliquid", "binance", "bybit"):
            adapter = get_exchange(name)
            assert isinstance(adapter, ExchangeAdapter)


class TestHyperliquidAdapter:
    """Tests for HyperliquidAdapter initialization."""

    def test_init_defaults(self):
        adapter = HyperliquidAdapter()
        assert adapter.info is None
        assert adapter.exchange is None

    def test_reads_env_vars(self, monkeypatch):
        monkeypatch.setenv("PRIVATE_KEY", "0xtest_key")
        monkeypatch.setenv("WALLET_ADDRESS", "0xtest_wallet")
        adapter = HyperliquidAdapter()
        assert adapter.private_key == "0xtest_key"
        assert adapter.wallet == "0xtest_wallet"


class TestBinanceAdapter:
    """Tests for BinanceAdapter initialization and _symbol."""

    def test_init_defaults(self):
        adapter = BinanceAdapter()
        assert adapter.client is None
        assert adapter.testnet is False

    def test_reads_env_vars(self, monkeypatch):
        monkeypatch.setenv("BINANCE_API_KEY", "test_key")
        monkeypatch.setenv("BINANCE_API_SECRET", "test_secret")
        monkeypatch.setenv("BINANCE_TESTNET", "true")
        adapter = BinanceAdapter()
        assert adapter.api_key == "test_key"
        assert adapter.api_secret == "test_secret"
        assert adapter.testnet is True

    def test_symbol_formatting(self):
        adapter = BinanceAdapter()
        assert adapter._symbol("BTC") == "BTC/USDT:USDT"
        assert adapter._symbol("ETH") == "ETH/USDT:USDT"
        assert adapter._symbol("SOL") == "SOL/USDT:USDT"


class TestBybitAdapter:
    """Tests for BybitAdapter initialization and _symbol."""

    def test_init_defaults(self):
        adapter = BybitAdapter()
        assert adapter.client is None
        assert adapter.testnet is False

    def test_reads_env_vars(self, monkeypatch):
        monkeypatch.setenv("BYBIT_API_KEY", "test_key")
        monkeypatch.setenv("BYBIT_API_SECRET", "test_secret")
        monkeypatch.setenv("BYBIT_TESTNET", "true")
        adapter = BybitAdapter()
        assert adapter.api_key == "test_key"
        assert adapter.api_secret == "test_secret"
        assert adapter.testnet is True

    def test_symbol_formatting(self):
        adapter = BybitAdapter()
        assert adapter._symbol("BTC") == "BTC/USDT:USDT"
        assert adapter._symbol("DOGE") == "DOGE/USDT:USDT"


class TestUtilityHelpers:
    """Tests for module-level utility functions."""

    def test_parse_interval_minutes(self):
        assert _parse_interval("15m") == 900

    def test_parse_interval_hours(self):
        assert _parse_interval("1h") == 3600

    def test_parse_interval_days(self):
        assert _parse_interval("1d") == 86400

    def test_round_price_high(self):
        assert _round_price(65432.789) == 65432.8

    def test_round_price_mid(self):
        assert _round_price(23.456789) == 23.4568

    def test_round_price_low(self):
        assert _round_price(0.123456789) == 0.123457
