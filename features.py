"""
DeepAlpha — Feature Engineering (matches train_ai_v2.py exactly)
Generates 62 features from 1h OHLCV candle data.

CRITICAL: Every formula here is copied verbatim from train_ai_v2.py build_features_1h().
Do NOT change any formula without updating training code simultaneously.
"""

import numpy as np
from datetime import datetime


# ─── Feature names (must match FEATURE_NAMES_1H in train_ai_v2.py) ──────────

FEATURE_NAMES = [
    'funding_rate', 'funding_rate_delta_1h', 'funding_rate_delta_4h', 'funding_rate_delta_8h',
    'oi_change_pct', 'price_change_pct', 'volume', 'volume_ma_ratio',
    'volume_spike_3x', 'price_vs_vwap', 'high_low_range', 'close_vs_open',
    'rsi_14', 'price_ma_20_ratio', 'volume_change_pct',
    'oi_value', 'funding_abs', 'candle_body_ratio',
    'atr_14', 'btc_correlation_20', 'hour_of_day', 'day_of_week',
    'dist_from_24h_high', 'dist_from_24h_low',
    'consecutive_green', 'consecutive_red',
    'rsi_divergence', 'price_momentum_3', 'price_momentum_7',
    'volume_momentum_3', 'ema_12_26_diff',
    # Order flow features
    'order_flow_ratio',      # taker_buy_volume / total_volume
    'liquidation_pressure',  # from taker long/short ratio data
    'rsi_4h',                # RSI calculated on 4h aggregated candles
    'obi_proxy',             # candle-based order book imbalance proxy
    'cvd_5',                 # cumulative volume delta 5 candles
    'cvd_20',                # cumulative volume delta 20 candles
    'obi_momentum',          # OBI change over 3 candles
    # V7.1 features
    'fear_greed_index',      # crypto fear & greed (0-100, normalized to 0-1)
    'funding_oi_weighted',   # funding rate * OI magnitude
    # V8.3 advanced features
    'price_skewness_24',     # rolling skewness of returns
    'price_kurtosis_24',     # rolling kurtosis (fat tail detection)
    'linear_trend_slope_24', # OLS slope normalized by price
    'close_stddev_ratio',    # short/long volatility ratio
    'area_ratio_24',         # price position vs recent history [-1,+1]
    'range_position_48',     # where in 48h range (0=bottom, 1=top)
    'atr_ratio_6_48',        # short/long ATR ratio (breakout detector)
    'volume_price_trend',    # cumulative volume-adjusted price changes
    'first_loc_max_24',      # where did max occur in last 24 candles
    'longest_strike_below',  # consecutive candles below rolling mean
    # V9.0 multi-timeframe features
    'sma_4h_ratio',          # price vs 4h SMA ratio
    'momentum_4h',           # 4h momentum (aggregated)
    'daily_return',          # daily aggregated return
    'daily_range',           # daily aggregated high-low range
    'daily_volume_ratio',    # daily volume vs 5-day avg
    'weekly_momentum',       # 7-day momentum
    # V10.0 L2 orderbook proxies
    'book_imbalance_proxy',  # (close - low) / (high - low) — buy pressure proxy
    'depth_ratio_proxy',     # volume / avg_volume_20 — order flow depth proxy
    'large_order_proxy',     # max(high-low) / ATR — large order detection
    'book_pressure_proxy',   # (close - open) / (high - low) — candle body directional pressure
    'spread_proxy',          # (high - low) / close * 100 — spread proxy from range
    'flow_intensity',        # abs(close - open) * volume — price impact * volume
]

NUM_FEATURES = 62
assert len(FEATURE_NAMES) == NUM_FEATURES, f"Expected {NUM_FEATURES} features, got {len(FEATURE_NAMES)}"


# ─── Helpers (identical formulas to train_ai_v2.py) ─────────────────────────

def _compute_rsi(closes, period=14):
    """RSI using simple rolling mean of gains/losses — matches train_ai_v2.py compute_rsi()."""
    rsi = np.full_like(closes, 50.0, dtype=float)
    pc = np.diff(closes, prepend=closes[0])
    for i in range(period, len(closes)):
        g = np.maximum(pc[i - period + 1:i + 1], 0).mean()
        lo = np.maximum(-pc[i - period + 1:i + 1], 0).mean()
        rsi[i] = 100.0 - (100.0 / (1.0 + g / lo)) if lo != 0 else 100.0
    return rsi


def _compute_rsi_4h(closes_1h):
    """RSI on 4h timeframe by aggregating 1h candles — matches train_ai_v2.py compute_rsi_4h()."""
    n = len(closes_1h)
    closes_4h = []
    for i in range(3, n, 4):
        closes_4h.append(closes_1h[i])
    if len(closes_4h) < 20:
        return np.full(n, 50.0)
    closes_4h = np.array(closes_4h)
    rsi_4h = _compute_rsi(closes_4h, period=14)
    result = np.full(n, 50.0)
    for idx_4h in range(len(rsi_4h)):
        start_1h = idx_4h * 4
        end_1h = min(start_1h + 4, n)
        for j in range(start_1h, end_1h):
            result[j] = rsi_4h[idx_4h]
    return result


def _compute_atr(highs, lows, closes, period=14):
    """ATR using simple moving average — matches train_ai_v2.py compute_atr()."""
    atr = np.zeros_like(closes, dtype=float)
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        atr[i] = tr
    result = np.zeros_like(closes, dtype=float)
    for i in range(period, len(closes)):
        result[i] = atr[i - period:i].mean()
    return result


def _compute_ema(data, period):
    """EMA — matches train_ai_v2.py compute_ema()."""
    ema = np.zeros_like(data, dtype=float)
    ema[0] = data[0]
    k = 2.0 / (period + 1)
    for i in range(1, len(data)):
        ema[i] = data[i] * k + ema[i - 1] * (1 - k)
    return ema


def _count_consecutive(closes):
    """Consecutive green/red candles — matches train_ai_v2.py count_consecutive()."""
    green = np.zeros(len(closes), dtype=float)
    red = np.zeros(len(closes), dtype=float)
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            green[i] = green[i - 1] + 1
            red[i] = 0
        elif closes[i] < closes[i - 1]:
            red[i] = red[i - 1] + 1
            green[i] = 0
    return green, red


def _compute_multi_timeframe(opens, highs, lows, closes, volumes):
    """Multi-timeframe features — matches train_ai_v2.py compute_multi_timeframe()."""
    n = len(closes)
    sma_4h_ratio = np.zeros(n, dtype=float)
    momentum_4h = np.zeros(n, dtype=float)
    daily_return = np.zeros(n, dtype=float)
    daily_range = np.zeros(n, dtype=float)
    daily_volume_ratio = np.zeros(n, dtype=float)
    weekly_momentum = np.zeros(n, dtype=float)

    # Build 4h closes
    closes_4h = []
    for i in range(3, n, 4):
        closes_4h.append(closes[i])
    closes_4h = np.array(closes_4h) if closes_4h else np.array([0.0])

    # 4h SMA with period 5 (=20h lookback)
    sma_4h_arr = np.zeros(len(closes_4h), dtype=float)
    for j in range(5, len(closes_4h)):
        sma_4h_arr[j] = closes_4h[j - 5:j].mean()

    # 4h momentum
    mom_4h_arr = np.zeros(len(closes_4h), dtype=float)
    for j in range(1, len(closes_4h)):
        if closes_4h[j - 1] > 0:
            mom_4h_arr[j] = (closes_4h[j] - closes_4h[j - 1]) / closes_4h[j - 1] * 100

    # Expand 4h features back to 1h
    for idx_4h in range(len(closes_4h)):
        start_1h = idx_4h * 4
        end_1h = min(start_1h + 4, n)
        for j in range(start_1h, end_1h):
            if sma_4h_arr[idx_4h] > 0:
                sma_4h_ratio[j] = (closes[j] - sma_4h_arr[idx_4h]) / sma_4h_arr[idx_4h] * 100
            momentum_4h[j] = mom_4h_arr[idx_4h]

    # Daily features (aggregate every 24 candles)
    for i in range(24, n):
        if closes[i - 24] > 0:
            daily_return[i] = (closes[i] - closes[i - 24]) / closes[i - 24] * 100
        dh = highs[i - 24:i].max()
        dl = lows[i - 24:i].min()
        if closes[i] > 0:
            daily_range[i] = (dh - dl) / closes[i] * 100
        vol_24h = volumes[i - 24:i].sum()
        if i >= 144:
            vol_5d_avg = volumes[i - 144:i - 24].sum() / 5.0
            daily_volume_ratio[i] = vol_24h / vol_5d_avg if vol_5d_avg > 0 else 1.0
        else:
            daily_volume_ratio[i] = 1.0

    # Weekly momentum
    for i in range(168, n):
        if closes[i - 168] > 0:
            weekly_momentum[i] = (closes[i] - closes[i - 168]) / closes[i - 168] * 100

    return sma_4h_ratio, momentum_4h, daily_return, daily_range, daily_volume_ratio, weekly_momentum


# ─── Main function ─────────────────────────────────────────────────────────

def build_features(
    candles: list[dict],
    *,
    btc_closes: np.ndarray | None = None,
    funding_map: dict | None = None,
    taker_buy_volumes: np.ndarray | None = None,
    taker_ratio_map: dict | None = None,
    fear_greed_map: dict | None = None,
) -> np.ndarray:
    """
    Build a (N, 62) feature matrix from OHLCV candle data.

    Matches train_ai_v2.py build_features_1h() exactly.

    Parameters
    ----------
    candles : list[dict]
        List of candle dicts with keys: t, o, h, l, c, v
        OR Binance kline arrays [open_time, open, high, low, close, volume, ...]
    btc_closes : np.ndarray | None
        BTC close prices aligned to same timestamps.
        If None, btc_correlation is set to 0.
    funding_map : dict | None
        Dict mapping timestamp_sec -> funding_rate.
        If None, all funding features default to 0.0.
    taker_buy_volumes : np.ndarray | None
        Taker buy base volume per candle.
        If None, order_flow_ratio defaults to 0.5 (neutral).
    taker_ratio_map : dict | None
        Dict mapping timestamp_ms -> taker long/short ratio.
        If None, liquidation_pressure defaults to 0.0 (neutral).
    fear_greed_map : dict | None
        Dict mapping timestamp_sec (hour-aligned) -> fear_greed_index (0-100).
        If None, fear_greed_index defaults to 0.5 (neutral, i.e. 50/100).

    Returns
    -------
    np.ndarray of shape (N, 62)
        Feature matrix. First ~26 rows may have incomplete lookback;
        the caller should use only rows from index 26+ onward.
    """
    # ── Parse candles ──
    times, opens, highs, lows, closes, volumes = [], [], [], [], [], []
    taker_buy_vols_parsed = []

    for c in candles:
        if isinstance(c, list):
            # Binance kline array format
            times.append(c[0])
            opens.append(float(c[1]))
            highs.append(float(c[2]))
            lows.append(float(c[3]))
            closes.append(float(c[4]))
            volumes.append(float(c[5]))
            taker_buy_vols_parsed.append(float(c[9]) if len(c) > 9 else 0.0)
        else:
            t = c.get('t', c.get('T', 0))
            if isinstance(t, str):
                t = int(t)
            times.append(t)
            opens.append(float(c.get('o', 0)))
            highs.append(float(c.get('h', 0)))
            lows.append(float(c.get('l', 0)))
            closes.append(float(c.get('c', 0)))
            volumes.append(float(c.get('v', c.get('vlm', 0))))
            taker_buy_vols_parsed.append(0.0)

    c_times = np.array(times, dtype=float)
    opens = np.array(opens, dtype=float)
    highs = np.array(highs, dtype=float)
    lows = np.array(lows, dtype=float)
    closes = np.array(closes, dtype=float)
    volumes = np.array(volumes, dtype=float)
    n = len(closes)

    # Taker buy volumes (for order_flow_ratio)
    if taker_buy_volumes is not None:
        tbv = taker_buy_volumes
    else:
        tbv = np.array(taker_buy_vols_parsed, dtype=float)

    # ── Order flow ratio: taker_buy_volume / total_volume ──
    order_flow = np.zeros(n, dtype=float)
    for i in range(n):
        if volumes[i] > 0 and tbv[i] > 0:
            order_flow[i] = tbv[i] / volumes[i]
        else:
            order_flow[i] = 0.5  # neutral default

    # ── Funding map (default empty) ──
    # funding_map maps timestamp_sec -> funding_rate
    # If not provided, all funding features will be 0.0
    if funding_map is None:
        funding_map = {}

    # ── Taker ratio map (default empty) ──
    if taker_ratio_map is None:
        taker_ratio_map = {}

    # ── Fear & Greed map (default empty) ──
    # If not provided, fear_greed_index defaults to 50 (neutral) -> 0.5 normalized
    if fear_greed_map is None:
        fear_greed_map = {}

    # ── Compute indicators ──
    rsi = _compute_rsi(closes)
    rsi_4h = _compute_rsi_4h(closes)
    atr = _compute_atr(highs, lows, closes)
    ema12 = _compute_ema(closes, 12)
    ema26 = _compute_ema(closes, 26)
    cons_green, cons_red = _count_consecutive(closes)

    # Multi-timeframe features
    mtf_sma_4h_ratio, mtf_momentum_4h, mtf_daily_return, mtf_daily_range, \
        mtf_daily_volume_ratio, mtf_weekly_momentum = _compute_multi_timeframe(
            opens, highs, lows, closes, volumes)

    # ── Rolling indicators (20-period) ──
    vol_ma = np.ones(n, dtype=float)
    price_ma = np.full(n, np.nan, dtype=float)
    vwap = np.full(n, np.nan, dtype=float)
    typical = (highs + lows + closes) / 3.0

    for i in range(20, n):
        vm = volumes[i - 20:i].mean()
        vol_ma[i] = vm if vm > 0 else 1.0
        price_ma[i] = closes[i - 20:i].mean()
        vs = volumes[i - 20:i].sum()
        vwap[i] = (typical[i - 20:i] * volumes[i - 20:i]).sum() / vs if vs > 0 else closes[i]

    # ── BTC closes for correlation ──
    # btc_closes should be aligned array same length as closes, or None
    # If it's a dict (timestamp -> price), caller should convert before passing

    # ── Build feature rows (matches train_ai_v2.py build_features_1h loop) ──
    feature_matrix = np.zeros((n, NUM_FEATURES), dtype=np.float64)

    for i in range(n):
        t_ms = int(c_times[i])
        t_sec = int(t_ms / 1000) if t_ms > 1e12 else int(t_ms)

        # --- Funding rates and deltas ---
        fr = funding_map.get(t_sec, 0.0)
        fr_1h_ago = funding_map.get(t_sec - 3600, 0.0)
        fr_4h_ago = funding_map.get(t_sec - 14400, 0.0)
        fr_8h_ago = funding_map.get(t_sec - 28800, 0.0)
        fr_delta_1h = fr - fr_1h_ago
        fr_delta_4h = fr - fr_4h_ago
        fr_delta_8h = fr - fr_8h_ago

        # --- OI proxy ---
        oi_cur = volumes[i] * closes[i]
        oi_prev = volumes[i - 1] * closes[i - 1] if (i > 0 and closes[i - 1] > 0) else 1.0
        oi_chg = ((oi_cur - oi_prev) / oi_prev * 100) if oi_prev > 0 else 0.0

        # --- Price change % ---
        price_chg = ((closes[i] - closes[i - 1]) / closes[i - 1] * 100) if (i > 0 and closes[i - 1] > 0) else 0.0

        # --- Volume / MA ratio ---
        vmr = volumes[i] / vol_ma[i] if vol_ma[i] > 0 else 1.0

        # --- Volume spike 3x ---
        vol_spike = 1.0 if vmr > 3.0 else 0.0

        # --- Price vs VWAP (rolling 20-period) ---
        pvw = ((closes[i] - vwap[i]) / vwap[i] * 100) if (not np.isnan(vwap[i]) and vwap[i] > 0) else 0.0

        # --- High-Low range ---
        hlr = ((highs[i] - lows[i]) / closes[i] * 100) if closes[i] > 0 else 0.0

        # --- Close vs Open ---
        cvo = ((closes[i] - opens[i]) / opens[i] * 100) if opens[i] > 0 else 0.0

        # --- Price vs MA(20) ---
        pma = ((closes[i] - price_ma[i]) / price_ma[i] * 100) if (not np.isnan(price_ma[i]) and price_ma[i] > 0) else 0.0

        # --- Volume change % ---
        vol_chg = ((volumes[i] - volumes[i - 1]) / volumes[i - 1] * 100) if (i > 0 and volumes[i - 1] > 0) else 0.0

        # --- Candle body ratio ---
        cbr = abs(closes[i] - opens[i]) / (highs[i] - lows[i]) if (highs[i] - lows[i]) > 0 else 0.0

        # --- ATR normalized ---
        atr_norm = (atr[i] / closes[i] * 100) if closes[i] > 0 else 0.0

        # --- BTC correlation ---
        btc_corr = 0.0
        if btc_closes is not None and len(btc_closes) == n and i >= 20:
            coin_rets = []
            btc_rets = []
            for j in range(i - 19, i + 1):
                if j > 0 and closes[j - 1] > 0:
                    coin_rets.append((closes[j] - closes[j - 1]) / closes[j - 1])
                    bc = btc_closes[j]
                    bc_prev = btc_closes[j - 1]
                    if bc_prev > 0 and bc > 0:
                        btc_rets.append((bc - bc_prev) / bc_prev)
                    else:
                        btc_rets.append(0)
            if len(coin_rets) >= 10:
                cr = np.array(coin_rets)
                br = np.array(btc_rets)
                if cr.std() > 0 and br.std() > 0:
                    btc_corr = np.corrcoef(cr, br)[0, 1]
                    if np.isnan(btc_corr):
                        btc_corr = 0.0

        # --- Time features ---
        hour = 0
        dow = 0
        try:
            dt = datetime.utcfromtimestamp(t_sec)
            hour = dt.hour
            dow = dt.weekday()
        except Exception:
            pass

        # --- Distance from 24h high/low ---
        window_24 = min(24, i)
        high_24 = highs[i - window_24:i + 1].max()
        low_24 = lows[i - window_24:i + 1].min()
        dist_high = ((closes[i] - high_24) / high_24 * 100) if high_24 > 0 else 0.0
        dist_low = ((closes[i] - low_24) / low_24 * 100) if low_24 > 0 else 0.0

        # --- Consecutive green/red ---
        # Already computed vectorized above

        # --- RSI divergence ---
        rsi_div = 0.0
        if i >= 5:
            price_dir = closes[i] - closes[i - 5]
            rsi_dir = rsi[i] - rsi[i - 5]
            if price_dir > 0 and rsi_dir < -3:
                rsi_div = -1.0
            elif price_dir < 0 and rsi_dir > 3:
                rsi_div = 1.0

        # --- Momentum ---
        mom_3 = ((closes[i] - closes[i - 3]) / closes[i - 3] * 100) if (i >= 3 and closes[i - 3] > 0) else 0.0
        mom_7 = ((closes[i] - closes[i - 7]) / closes[i - 7] * 100) if (i >= 7 and closes[i - 7] > 0) else 0.0
        vol_mom_3 = ((volumes[i] - volumes[i - 3]) / volumes[i - 3] * 100) if (i >= 3 and volumes[i - 3] > 0) else 0.0
        ema_diff = ((ema12[i] - ema26[i]) / ema26[i] * 100) if ema26[i] > 0 else 0.0

        # --- Order flow ratio ---
        oflow = order_flow[i]

        # --- Liquidation pressure ---
        # taker_ratio_map: timestamp_ms -> buySellRatio
        # Centered so 0 = neutral (ratio - 1.0)
        liq_pressure = taker_ratio_map.get(t_ms, taker_ratio_map.get(t_sec * 1000, 1.0))
        liq_pressure = liq_pressure - 1.0

        # --- RSI 4h ---
        rsi4h = rsi_4h[i]

        # --- OBI proxy ---
        obi = (closes[i] - lows[i]) / (highs[i] - lows[i]) if (highs[i] - lows[i]) > 0 else 0.5

        # --- CVD proxy ---
        cvd_5_val = sum(
            volumes[max(i - 4, 0):i + 1] * (
                2 * ((closes[max(i - 4, 0):i + 1] - lows[max(i - 4, 0):i + 1]) /
                     np.maximum(highs[max(i - 4, 0):i + 1] - lows[max(i - 4, 0):i + 1], 1e-10)) - 1
            )
        )
        cvd_20_val = sum(
            volumes[max(i - 19, 0):i + 1] * (
                2 * ((closes[max(i - 19, 0):i + 1] - lows[max(i - 19, 0):i + 1]) /
                     np.maximum(highs[max(i - 19, 0):i + 1] - lows[max(i - 19, 0):i + 1], 1e-10)) - 1
            )
        )
        avg_vol = volumes[max(i - 19, 0):i + 1].mean()
        cvd_5_norm = cvd_5_val / avg_vol if avg_vol > 0 else 0
        cvd_20_norm = cvd_20_val / avg_vol if avg_vol > 0 else 0

        # --- OBI momentum ---
        obi_prev = (closes[max(i - 3, 0)] - lows[max(i - 3, 0)]) / (highs[max(i - 3, 0)] - lows[max(i - 3, 0)]) \
            if (highs[max(i - 3, 0)] - lows[max(i - 3, 0)]) > 0 else 0.5
        obi_momentum = obi - obi_prev

        # --- Fear & Greed Index ---
        # Defaults to 50 (neutral) if not available, normalized to 0-1
        fg_val = 50
        if fear_greed_map:
            t_hour = (t_sec // 3600) * 3600
            fg_val = fear_greed_map.get(t_hour, fear_greed_map.get(t_hour - 3600, 50))
        fear_greed_norm = fg_val / 100.0

        # --- Funding * OI weighted ---
        funding_oi = fr * (oi_cur / 1e6) if oi_cur > 0 else 0.0

        # --- V8.3 Advanced Features ---

        # 1. Rolling skewness of returns (24 candles)
        if i >= 24:
            rets_24 = np.diff(closes[i - 24:i + 1]) / closes[i - 24:i]
            _mean = rets_24.mean()
            _std = rets_24.std()
            price_skew = float(((rets_24 - _mean) ** 3).mean() / (_std ** 3)) if _std > 1e-10 else 0.0
        else:
            price_skew = 0.0

        # 2. Rolling kurtosis of returns (24 candles)
        if i >= 24:
            price_kurt = float(((rets_24 - _mean) ** 4).mean() / (_std ** 4) - 3.0) if _std > 1e-10 else 0.0
        else:
            price_kurt = 0.0

        # 3. Linear trend slope (24 candles, normalized)
        if i >= 24:
            _x = np.arange(24)
            _y = closes[i - 23:i + 1]
            _slope = np.polyfit(_x, _y, 1)[0]
            trend_slope = _slope / closes[i] * 100 if closes[i] > 0 else 0.0
        else:
            trend_slope = 0.0

        # 4. Short/long volatility ratio
        if i >= 48:
            rets_s = np.diff(closes[i - 6:i + 1]) / closes[i - 6:i]
            rets_l = np.diff(closes[i - 48:i + 1]) / closes[i - 48:i]
            std_s = rets_s.std()
            std_l = rets_l.std()
            stddev_ratio = std_s / std_l if std_l > 1e-10 else 1.0
        else:
            stddev_ratio = 1.0

        # 5. Area ratio (price position vs recent 24 candles)
        if i >= 24:
            _window = closes[i - 23:i + 1]
            _level = closes[i]
            _diff = _window - _level
            _total = np.sum(np.abs(_diff))
            area_ratio = (2 * np.sum(np.maximum(_diff, 0)) / _total - 1) if _total > 0 else 0.0
        else:
            area_ratio = 0.0

        # 6. Range position (where in 48h range, 0=bottom 1=top)
        if i >= 48:
            h48 = highs[i - 48:i + 1].max()
            l48 = lows[i - 48:i + 1].min()
            range_pos = (closes[i] - l48) / (h48 - l48) if (h48 - l48) > 0 else 0.5
        else:
            range_pos = 0.5

        # 7. ATR ratio short/long (breakout detector)
        if i >= 48:
            atr_short = np.mean([
                max(highs[j] - lows[j], abs(highs[j] - closes[j - 1]), abs(lows[j] - closes[j - 1]))
                for j in range(max(1, i - 5), i + 1)
            ])
            atr_long = np.mean([
                max(highs[j] - lows[j], abs(highs[j] - closes[j - 1]), abs(lows[j] - closes[j - 1]))
                for j in range(max(1, i - 47), i + 1)
            ])
            atr_ratio = atr_short / atr_long if atr_long > 0 else 1.0
        else:
            atr_ratio = 1.0

        # 8. Volume-price trend (normalized)
        if i >= 24:
            vpt = sum(
                volumes[j] * ((closes[j] - closes[j - 1]) / closes[j - 1])
                for j in range(max(1, i - 23), i + 1) if closes[j - 1] > 0
            )
            vpt_norm = vpt / avg_vol if avg_vol > 0 else 0.0
        else:
            vpt_norm = 0.0

        # 9. First location of max in 24 candles (0=start, 1=end)
        if i >= 24:
            first_loc_max = float(np.argmax(closes[i - 23:i + 1])) / 23.0
        else:
            first_loc_max = 0.5

        # 10. Longest strike below mean (24 candles)
        if i >= 24:
            _win = closes[i - 23:i + 1]
            _wmean = _win.mean()
            _below = _win < _wmean
            max_run = 0
            cur_run = 0
            for b in _below:
                if b:
                    cur_run += 1
                    max_run = max(max_run, cur_run)
                else:
                    cur_run = 0
            longest_below = max_run / 24.0
        else:
            longest_below = 0.0

        # --- V10.0 L2 Orderbook Proxies ---
        book_imb = (closes[i] - lows[i]) / (highs[i] - lows[i]) if (highs[i] - lows[i]) > 0 else 0.5
        depth_ratio = volumes[i] / vol_ma[i] if vol_ma[i] > 0 else 1.0
        large_order = (highs[i] - lows[i]) / (atr[i] if atr[i] > 0 else 1e-10)
        book_pressure = (closes[i] - opens[i]) / (highs[i] - lows[i] + 0.001) if (highs[i] - lows[i]) > 0 else 0.0
        spread_proxy = (highs[i] - lows[i]) / closes[i] * 100 if closes[i] > 0 else 0.0
        flow_intensity = abs(closes[i] - opens[i]) * volumes[i]

        # ── Assemble row (order MUST match FEATURE_NAMES) ──
        feature_matrix[i] = [
            fr, fr_delta_1h, fr_delta_4h, fr_delta_8h,
            oi_chg, price_chg, volumes[i], vmr,
            vol_spike, pvw, hlr, cvo,
            rsi[i], pma, vol_chg,
            oi_cur, abs(fr), cbr,
            atr_norm, btc_corr, hour, dow,
            dist_high, dist_low,
            cons_green[i], cons_red[i],
            rsi_div, mom_3, mom_7,
            vol_mom_3, ema_diff,
            # Order flow
            oflow, liq_pressure, rsi4h,
            obi, cvd_5_norm, cvd_20_norm, obi_momentum,
            fear_greed_norm, funding_oi,
            # V8.3 advanced
            price_skew, price_kurt, trend_slope, stddev_ratio,
            area_ratio, range_pos, atr_ratio, vpt_norm,
            first_loc_max, longest_below,
            # V9.0 multi-timeframe
            mtf_sma_4h_ratio[i], mtf_momentum_4h[i],
            mtf_daily_return[i], mtf_daily_range[i],
            mtf_daily_volume_ratio[i], mtf_weekly_momentum[i],
            # V10.0 L2 orderbook proxies
            book_imb, depth_ratio, large_order, book_pressure, spread_proxy, flow_intensity,
        ]

    # Replace any NaN with 0.0 for safety
    np.nan_to_num(feature_matrix, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    return feature_matrix
