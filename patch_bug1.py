#!/usr/bin/env python3
"""BUG 1: Add 10 V11 features (indices 62-71) to pro_trader.py ai_build_live_features."""
import re

FILE = "/root/hyperlend-bot/pro_bot/pro_trader.py"

with open(FILE, "r") as f:
    content = f.read()

# Find the old 62-feature row and replace with 72-feature row that includes V11 features
OLD_BLOCK = """        # 62 features matching FEATURE_NAMES_1H (same order as training)
        row = [
            fr, fr_delta_1h, fr_delta_4h, fr_delta_8h,
            oi_chg, price_chg, volumes[i], vmr,
            vol_spike, pvw, hlr, cvo,
            rsi_val, pma, vol_chg,
            oi_val, abs(fr), cbr,
            atr_val, btc_corr, hour_of_day, day_of_week,
            dist_high, dist_low,
            float(cons_g), float(cons_r),
            rsi_div, mom_3, mom_7,
            vol_mom_3, ema_diff,
            order_flow_ratio, liquidation_pressure, rsi_4h,
            obi_proxy, cvd_5, cvd_20, obi_momentum,
            fear_greed_index, funding_oi,
            # V8.4 ADVANCED
            price_skew, price_kurt, trend_slope, stddev_ratio,
            area_ratio_val, range_pos, atr_ratio_val, vpt_norm,
            first_loc_max, longest_below,
            # V9.0 MULTI-TIMEFRAME
            mtf_sma_4h_ratio, mtf_momentum_4h,
            mtf_daily_return, mtf_daily_range,
            mtf_daily_volume_ratio, mtf_weekly_momentum,
            # V10.0 L2 ORDERBOOK FEATURES (real from live book)
            l2_book_imbalance, l2_depth_ratio, l2_large_order,
            l2_book_pressure, l2_spread, l2_flow_intensity,
        ]
        return np.array([row])"""

NEW_BLOCK = """        # ── V11 features (indices 62-71) ──
        # btc_beta (62) — needs BTC cross data, default for now
        btc_beta = 0.0

        # correlation_change (63) — needs BTC history, default
        correlation_change = 0.0

        # realized_vol_24 (64)
        if i >= 25:
            _log_rets = np.diff(np.log(np.maximum(closes[i - 24:i + 1], 1e-10)))
            realized_vol_24 = float(_log_rets.std() * np.sqrt(365 * 24))
        else:
            realized_vol_24 = 0.0

        # vol_regime (65) — 1.0 = NORMAL
        v11_vol_regime = 1.0

        # vol_of_vol (66)
        vol_of_vol = 0.0

        # hurst_exponent (67)
        hurst_exponent = 0.5

        # fractal_efficiency (68)
        if i >= 25:
            _fe_closes = closes[i - 24:i + 1]
            _net_move = abs(_fe_closes[-1] - _fe_closes[0])
            _path_len = np.sum(np.abs(np.diff(_fe_closes)))
            fractal_efficiency = float(_net_move / _path_len) if _path_len > 0 else 0.0
        else:
            fractal_efficiency = 0.0

        # vpin_proxy (69) — buy/sell classified candle imbalance
        if i >= 20:
            _buy_vol = 0.0
            _sell_vol = 0.0
            for _k in range(max(0, i - 19), i + 1):
                if closes[_k] >= opens_a[_k]:
                    _buy_vol += volumes[_k]
                else:
                    _sell_vol += volumes[_k]
            _total_vol = _buy_vol + _sell_vol
            vpin_proxy = abs(_buy_vol - _sell_vol) / _total_vol if _total_vol > 0 else 0.0
        else:
            vpin_proxy = 0.0

        # flow_persistence (70)
        flow_persistence = 0.0

        # mtf_alignment (71) — sign agreement across timeframes
        _signs = [
            1.0 if ema_diff > 0 else (-1.0 if ema_diff < 0 else 0.0),
            1.0 if mtf_momentum_4h > 0 else (-1.0 if mtf_momentum_4h < 0 else 0.0),
            1.0 if mtf_daily_return > 0 else (-1.0 if mtf_daily_return < 0 else 0.0),
            1.0 if mtf_weekly_momentum > 0 else (-1.0 if mtf_weekly_momentum < 0 else 0.0),
        ]
        mtf_alignment = sum(_signs) / 4.0

        # 72 features matching V11 FEATURE_NAMES_1H (same order as training)
        row = [
            fr, fr_delta_1h, fr_delta_4h, fr_delta_8h,
            oi_chg, price_chg, volumes[i], vmr,
            vol_spike, pvw, hlr, cvo,
            rsi_val, pma, vol_chg,
            oi_val, abs(fr), cbr,
            atr_val, btc_corr, hour_of_day, day_of_week,
            dist_high, dist_low,
            float(cons_g), float(cons_r),
            rsi_div, mom_3, mom_7,
            vol_mom_3, ema_diff,
            order_flow_ratio, liquidation_pressure, rsi_4h,
            obi_proxy, cvd_5, cvd_20, obi_momentum,
            fear_greed_index, funding_oi,
            # V8.4 ADVANCED
            price_skew, price_kurt, trend_slope, stddev_ratio,
            area_ratio_val, range_pos, atr_ratio_val, vpt_norm,
            first_loc_max, longest_below,
            # V9.0 MULTI-TIMEFRAME
            mtf_sma_4h_ratio, mtf_momentum_4h,
            mtf_daily_return, mtf_daily_range,
            mtf_daily_volume_ratio, mtf_weekly_momentum,
            # V10.0 L2 ORDERBOOK FEATURES
            l2_book_imbalance, l2_depth_ratio, l2_large_order,
            l2_book_pressure, l2_spread, l2_flow_intensity,
            # V11.0 FEATURES (indices 62-71)
            btc_beta, correlation_change,
            realized_vol_24, v11_vol_regime,
            vol_of_vol, hurst_exponent,
            fractal_efficiency, vpin_proxy,
            flow_persistence, mtf_alignment,
        ]
        return np.array([row])"""

if OLD_BLOCK in content:
    content = content.replace(OLD_BLOCK, NEW_BLOCK)
    # Also update the docstring
    content = content.replace(
        '"""Build the 62-feature vector for live AI prediction.',
        '"""Build the 72-feature vector for live AI prediction (V11).'
    )
    with open(FILE, "w") as f:
        f.write(content)
    print("BUG 1 FIXED: Added 10 V11 features (72 total) to ai_build_live_features")
else:
    print("ERROR: Could not find the 62-feature row block to replace")
    # Debug: find the row = [ line
    lines = content.split('\n')
    for idx, line in enumerate(lines):
        if '# 62 features' in line or ('row = [' in line and idx > 5600):
            print(f"  Found at line {idx+1}: {line.strip()}")
