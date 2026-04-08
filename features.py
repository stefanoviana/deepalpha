"""
DeepAlpha — Feature Engineering (Free Version)
Generates 15 technical features from OHLCV candle data.

Features:
  1.  rsi_14              — Relative Strength Index (14 periods)
  2.  atr_14              — Average True Range (14 periods)
  3.  ema_12_26_diff      — EMA 12/26 crossover spread (normalised)
  4.  price_momentum_3    — 3-period price momentum (% change)
  5.  price_momentum_7    — 7-period price momentum (% change)
  6.  volume_ma_ratio     — Volume vs 20-period moving average
  7.  high_low_range      — (High-Low)/Close — volatility proxy
  8.  close_vs_open       — (Close-Open)/Open — candle direction
  9.  price_vs_vwap       — Price relative to session VWAP
  10. volume_change_pct   — Volume % change from prior candle
  11. candle_body_ratio   — |Close-Open| / (High-Low) — body vs wick
  12. dist_from_24h_high  — Distance from rolling 24h high
  13. dist_from_24h_low   — Distance from rolling 24h low
  14. btc_correlation_20  — 20-period rolling correlation with BTC
  15. funding_rate         — Current funding rate (passed in)
"""

import numpy as np


# ─── Helpers ────────────────────────────────────────────────────────────────

def _ema(series: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average."""
    alpha = 2.0 / (span + 1)
    out = np.empty_like(series, dtype=float)
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = alpha * series[i] + (1 - alpha) * out[i - 1]
    return out


def _sma(series: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average (NaN-padded)."""
    out = np.full_like(series, np.nan, dtype=float)
    cumsum = np.cumsum(series)
    out[window - 1:] = (cumsum[window - 1:] - np.concatenate([[0], cumsum[:-window]])) / window
    return out


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder RSI."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = _ema(gain, period * 2 - 1)
    avg_loss = _ema(loss, period * 2 - 1)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100.0)
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range."""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return _ema(tr, period * 2 - 1)


def _rolling_corr(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """Rolling Pearson correlation."""
    out = np.full(len(x), np.nan, dtype=float)
    for i in range(window - 1, len(x)):
        xs = x[i - window + 1: i + 1]
        ys = y[i - window + 1: i + 1]
        if np.std(xs) == 0 or np.std(ys) == 0:
            out[i] = 0.0
        else:
            out[i] = np.corrcoef(xs, ys)[0, 1]
    return out


# ─── Main function ─────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "rsi_14", "atr_14", "ema_12_26_diff",
    "price_momentum_3", "price_momentum_7",
    "volume_ma_ratio", "high_low_range", "close_vs_open",
    "price_vs_vwap", "volume_change_pct", "candle_body_ratio",
    "dist_from_24h_high", "dist_from_24h_low",
    "btc_correlation_20", "funding_rate",
]


def build_features(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    btc_close: np.ndarray | None = None,
    funding: float = 0.0,
) -> np.ndarray:
    """
    Build a (N, 15) feature matrix from OHLCV arrays.

    Parameters
    ----------
    open_, high, low, close, volume : np.ndarray
        OHLCV arrays, all same length.
    btc_close : np.ndarray | None
        BTC close prices aligned to the same timestamps.
        If None, btc_correlation is set to 0.
    funding : float
        Current funding rate for the asset.

    Returns
    -------
    np.ndarray of shape (N, 15)
    """
    n = len(close)

    # 1. RSI 14
    rsi = _rsi(close, 14)

    # 2. ATR 14 (normalised by close price)
    atr = _atr(high, low, close, 14) / close

    # 3. EMA 12/26 diff (normalised)
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    ema_diff = (ema12 - ema26) / close

    # 4-5. Price momentum
    mom3 = np.zeros(n)
    mom3[3:] = (close[3:] - close[:-3]) / close[:-3]
    mom7 = np.zeros(n)
    mom7[7:] = (close[7:] - close[:-7]) / close[:-7]

    # 6. Volume / 20-period MA
    vol_ma = _sma(volume, 20)
    vol_ratio = np.where(vol_ma > 0, volume / np.where(vol_ma == 0, 1, vol_ma), 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)

    # 7. High-Low range normalised
    hl_range = (high - low) / close

    # 8. Close vs Open
    close_open = (close - open_) / np.where(open_ != 0, open_, 1)

    # 9. Price vs VWAP (rolling 24-period approximation)
    typical = (high + low + close) / 3.0
    cum_tp_vol = np.cumsum(typical * volume)
    cum_vol = np.cumsum(volume)
    vwap = np.where(cum_vol > 0, cum_tp_vol / cum_vol, close)
    price_vwap = (close - vwap) / np.where(vwap != 0, vwap, 1)

    # 10. Volume change %
    prev_vol = np.roll(volume, 1)
    prev_vol[0] = volume[0]
    vol_change = np.where(prev_vol > 0, (volume - prev_vol) / prev_vol, 0.0)

    # 11. Candle body ratio
    body = np.abs(close - open_)
    wick = high - low
    body_ratio = np.where(wick > 0, body / wick, 0.5)

    # 12-13. Distance from rolling 24h high/low
    rolling_high = np.array([np.max(high[max(0, i - 23): i + 1]) for i in range(n)])
    rolling_low = np.array([np.min(low[max(0, i - 23): i + 1]) for i in range(n)])
    dist_high = (close - rolling_high) / np.where(rolling_high != 0, rolling_high, 1)
    dist_low = (close - rolling_low) / np.where(rolling_low != 0, rolling_low, 1)

    # 14. BTC correlation (20 periods)
    if btc_close is not None and len(btc_close) == n:
        btc_corr = _rolling_corr(close, btc_close, 20)
        btc_corr = np.nan_to_num(btc_corr, nan=0.0)
    else:
        btc_corr = np.zeros(n)

    # 15. Funding rate (scalar broadcast)
    funding_arr = np.full(n, funding)

    # Stack into (N, 15) matrix
    features = np.column_stack([
        rsi, atr, ema_diff,
        mom3, mom7,
        vol_ratio, hl_range, close_open,
        price_vwap, vol_change, body_ratio,
        dist_high, dist_low,
        btc_corr, funding_arr,
    ])

    return features
