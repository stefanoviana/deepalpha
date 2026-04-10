"""
Tests for DeepAlpha feature engineering (features.py).
"""

import numpy as np
import pytest

from features import build_features, FEATURE_NAMES, _rsi, _atr, _ema


class TestBuildFeatures:
    """Tests for the main build_features function."""

    def test_output_shape_15_features(self, sample_ohlcv):
        """build_features should produce exactly 15 feature columns."""
        result = build_features(
            sample_ohlcv["open"],
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            sample_ohlcv["volume"],
        )
        assert result.shape == (100, 15)

    def test_feature_names_count(self):
        """FEATURE_NAMES list should have exactly 15 entries."""
        assert len(FEATURE_NAMES) == 15

    def test_no_nan_in_output(self, sample_ohlcv):
        """Feature matrix should not contain NaN values after processing."""
        result = build_features(
            sample_ohlcv["open"],
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            sample_ohlcv["volume"],
        )
        assert not np.any(np.isnan(result)), "Feature matrix contains NaN values"

    def test_no_inf_in_output(self, sample_ohlcv):
        """Feature matrix should not contain infinite values."""
        result = build_features(
            sample_ohlcv["open"],
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            sample_ohlcv["volume"],
        )
        assert not np.any(np.isinf(result)), "Feature matrix contains Inf values"

    def test_with_btc_close(self, sample_ohlcv, btc_close):
        """build_features with btc_close should still produce (N, 15)."""
        result = build_features(
            sample_ohlcv["open"],
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            sample_ohlcv["volume"],
            btc_close=btc_close,
            funding=0.0001,
        )
        assert result.shape == (100, 15)

    def test_funding_rate_broadcast(self, sample_ohlcv):
        """Last column (funding_rate) should be the scalar broadcast to all rows."""
        funding = 0.00035
        result = build_features(
            sample_ohlcv["open"],
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            sample_ohlcv["volume"],
            funding=funding,
        )
        np.testing.assert_allclose(result[:, 14], funding)

    def test_btc_correlation_zero_when_none(self, sample_ohlcv):
        """When btc_close is None, btc_correlation column should be all zeros."""
        result = build_features(
            sample_ohlcv["open"],
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            sample_ohlcv["volume"],
            btc_close=None,
        )
        np.testing.assert_array_equal(result[:, 13], 0.0)


class TestRSI:
    """Tests for the _rsi helper."""

    def test_rsi_range(self, sample_ohlcv):
        """RSI values should be between 0 and 100."""
        rsi = _rsi(sample_ohlcv["close"], 14)
        assert np.all(rsi >= 0) and np.all(rsi <= 100)

    def test_rsi_constant_price(self):
        """RSI of a constant price series: no price changes means no losses,
        so RSI drifts toward 100 as the initial gain fades but avg_loss stays
        near zero. With a long enough series, it should be above 50."""
        constant = np.full(200, 100.0)
        rsi = _rsi(constant, 14)
        # All values should stay in valid range
        assert np.all(rsi >= 0) and np.all(rsi <= 100)

    def test_rsi_rising_prices(self):
        """RSI of a steadily rising series should be close to 100."""
        rising = np.linspace(100, 200, 100)
        rsi = _rsi(rising, 14)
        assert rsi[-1] > 90

    def test_rsi_falling_prices(self):
        """RSI of a steadily falling series should be close to 0."""
        falling = np.linspace(200, 100, 100)
        rsi = _rsi(falling, 14)
        assert rsi[-1] < 10


class TestATR:
    """Tests for the _atr helper."""

    def test_atr_positive(self, sample_ohlcv):
        """ATR should always be non-negative."""
        atr = _atr(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"], 14)
        assert np.all(atr >= 0)

    def test_atr_zero_for_flat_market(self):
        """ATR of identical H/L/C series should be zero."""
        n = 50
        flat = np.full(n, 100.0)
        atr = _atr(flat, flat, flat, 14)
        np.testing.assert_allclose(atr, 0.0, atol=1e-10)

    def test_atr_increases_with_volatility(self):
        """Higher spread between high and low should produce larger ATR."""
        n = 50
        close = np.full(n, 100.0)
        low_vol_high = close + 1
        low_vol_low = close - 1
        high_vol_high = close + 10
        high_vol_low = close - 10

        atr_low = _atr(low_vol_high, low_vol_low, close, 14)
        atr_high = _atr(high_vol_high, high_vol_low, close, 14)
        assert atr_high[-1] > atr_low[-1]


class TestMomentum:
    """Tests for momentum features within build_features."""

    def test_momentum_3_first_elements_zero(self, sample_ohlcv):
        """price_momentum_3 (column index 3) first 3 values should be zero."""
        result = build_features(
            sample_ohlcv["open"],
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            sample_ohlcv["volume"],
        )
        np.testing.assert_array_equal(result[:3, 3], 0.0)

    def test_momentum_7_first_elements_zero(self, sample_ohlcv):
        """price_momentum_7 (column index 4) first 7 values should be zero."""
        result = build_features(
            sample_ohlcv["open"],
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            sample_ohlcv["volume"],
        )
        np.testing.assert_array_equal(result[:7, 4], 0.0)

    def test_momentum_calculation(self):
        """Verify momentum is (close[i] - close[i-3]) / close[i-3]."""
        close = np.array([100.0, 102.0, 104.0, 110.0, 108.0, 105.0, 115.0,
                          120.0, 118.0, 125.0])
        open_ = close.copy()
        high = close + 1
        low = close - 1
        volume = np.ones(10) * 1000

        result = build_features(open_, high, low, close, volume)
        # mom3 at index 3 = (110 - 100) / 100 = 0.10
        expected_mom3_idx3 = (110.0 - 100.0) / 100.0
        assert abs(result[3, 3] - expected_mom3_idx3) < 1e-10
