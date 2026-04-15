"""
Tests for DeepAlpha feature engineering (features.py).
Updated for v1.4+ (62-feature pipeline matching train_ai_v2.py).
"""

import numpy as np
import pytest

from features import build_features, FEATURE_NAMES


def _make_candles(n=200):
    """Generate n synthetic OHLCV candle dicts."""
    np.random.seed(42)
    base = 50000.0
    returns = np.random.normal(0, 0.005, n)
    close = base * np.cumprod(1 + returns)
    open_ = np.roll(close, 1)
    open_[0] = base
    high = np.maximum(open_, close) * (1 + np.random.uniform(0, 0.003, n))
    low = np.minimum(open_, close) * (1 - np.random.uniform(0, 0.003, n))
    volume = np.random.uniform(100, 5000, n)

    t0 = 1700000000
    candles = []
    for i in range(n):
        candles.append({
            "t": (t0 + i * 3600) * 1000,  # ms timestamps
            "o": float(open_[i]),
            "h": float(high[i]),
            "l": float(low[i]),
            "c": float(close[i]),
            "v": float(volume[i]),
        })
    return candles


class TestBuildFeatures:
    """Tests for the main build_features function."""

    def test_output_shape_62_features(self):
        """build_features should produce exactly 62 feature columns."""
        candles = _make_candles(200)
        result = build_features(candles)
        assert result.shape[1] == 62

    def test_feature_names_count(self):
        """FEATURE_NAMES list should have exactly 62 entries."""
        assert len(FEATURE_NAMES) == 62

    def test_feature_names_unique(self):
        """All feature names should be unique."""
        assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))

    def test_no_nan_in_output(self):
        """Feature matrix should not contain NaN values after processing."""
        candles = _make_candles(200)
        result = build_features(candles)
        assert not np.any(np.isnan(result)), "Feature matrix contains NaN values"

    def test_no_inf_in_output(self):
        """Feature matrix should not contain infinite values."""
        candles = _make_candles(200)
        result = build_features(candles)
        assert not np.any(np.isinf(result)), "Feature matrix contains Inf values"

    def test_row_count_matches_candles(self):
        """Output rows should match input candle count."""
        candles = _make_candles(200)
        result = build_features(candles)
        assert result.shape[0] == 200

    def test_with_btc_closes(self):
        """build_features with btc_closes should still produce (N, 62)."""
        candles = _make_candles(200)
        btc = np.array([c["c"] for c in candles]) * 1.01  # slightly different
        result = build_features(candles, btc_closes=btc)
        assert result.shape == (200, 62)

    def test_with_all_optional_data(self):
        """build_features with all optional maps should produce (N, 62)."""
        candles = _make_candles(200)
        btc = np.array([c["c"] for c in candles])
        funding = {c["t"] // 1000: 0.0001 for c in candles}
        taker_vols = np.random.uniform(50, 2500, 200)
        taker_ratio = {c["t"]: 1.5 for c in candles}
        fg = {c["t"] // 1000: 50 for c in candles}

        result = build_features(
            candles,
            btc_closes=btc,
            funding_map=funding,
            taker_buy_volumes=taker_vols,
            taker_ratio_map=taker_ratio,
            fear_greed_map=fg,
        )
        assert result.shape == (200, 62)
        assert not np.any(np.isnan(result))

    def test_defaults_without_optional_data(self):
        """Without optional data, features should use safe defaults (not crash)."""
        candles = _make_candles(200)
        result = build_features(candles)
        # Funding features (indices 0-3) should be 0.0
        assert np.all(result[:, 0] == 0.0), "funding_rate should default to 0.0"

    def test_minimum_candles(self):
        """Should handle the minimum number of candles without crashing."""
        candles = _make_candles(50)
        result = build_features(candles)
        assert result.shape == (50, 62)

    def test_rsi_in_valid_range(self):
        """RSI feature (index 12) should be between 0 and 100."""
        candles = _make_candles(200)
        result = build_features(candles)
        rsi_col = FEATURE_NAMES.index("rsi_14")
        rsi = result[20:, rsi_col]  # skip warmup period
        assert np.all(rsi >= 0) and np.all(rsi <= 100), f"RSI out of range: min={rsi.min()}, max={rsi.max()}"

    def test_atr_non_negative(self):
        """ATR feature should always be non-negative."""
        candles = _make_candles(200)
        result = build_features(candles)
        atr_col = FEATURE_NAMES.index("atr_14")
        assert np.all(result[:, atr_col] >= 0)

    def test_hour_of_day_range(self):
        """hour_of_day should be 0-23."""
        candles = _make_candles(200)
        result = build_features(candles)
        hour_col = FEATURE_NAMES.index("hour_of_day")
        hours = result[:, hour_col]
        assert np.all(hours >= 0) and np.all(hours <= 23)

    def test_day_of_week_range(self):
        """day_of_week should be 0-6."""
        candles = _make_candles(200)
        result = build_features(candles)
        dow_col = FEATURE_NAMES.index("day_of_week")
        days = result[:, dow_col]
        assert np.all(days >= 0) and np.all(days <= 6)

    def test_order_flow_ratio_default(self):
        """Without taker data, order_flow_ratio should default to 0.5."""
        candles = _make_candles(200)
        result = build_features(candles)
        ofr_col = FEATURE_NAMES.index("order_flow_ratio")
        assert np.allclose(result[:, ofr_col], 0.5)

    def test_fear_greed_default(self):
        """Without fear_greed data, fear_greed_index should default to 0.5 (neutral)."""
        candles = _make_candles(200)
        result = build_features(candles)
        fg_col = FEATURE_NAMES.index("fear_greed_index")
        assert np.allclose(result[:, fg_col], 0.5)
