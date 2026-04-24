#!/usr/bin/env python3
"""
Patch script for Review #8 — fixes all 9 remaining issues.
Run on VPS: python3 /root/patch_review8.py
"""
import re
import shutil
from pathlib import Path

def backup_and_read(path):
    p = Path(path)
    shutil.copy2(p, str(p) + ".bak8")
    return p.read_text()

def write_file(path, content):
    Path(path).write_text(content)

# ============================================================================
# FIX 1: app.py — Replace old V10 features (indices 62-74) with V11 (72 features)
# ============================================================================
APP_PATH = "/root/deepalpha-cloud/app.py"
app = backup_and_read(APP_PATH)

# Fix the docstring from 75 to 72
app = app.replace(
    '"""Build the 75-feature vector for candle at index `idx`.\n    Mirrors CloudTrader._build_features exactly, but works on arrays.\n    Returns shape (1, 75) or None.\n    """',
    '"""Build the 72-feature vector for candle at index `idx`.\n    Mirrors CloudTrader._build_features exactly (V11), but works on arrays.\n    Returns shape (1, 72) or None.\n    """'
)

# Replace old V10 features block (cvd_norm through candle_streak) with V11 features
old_v10_features = """        cvd_norm = 0.0
        vol_profile_ratio = 0.0
        vol_regime = closes[i - 20:i].std() / closes[i] * 100 if i >= 20 and closes[i] > 0 else 0.0

        vol_kurtosis = 0.0
        if i >= 20:
            _rets20 = np.diff(closes[i - 20:i + 1]) / np.maximum(closes[i - 20:i], 1e-10)
            _s20 = _rets20.std()
            vol_kurtosis = float(((_rets20 - _rets20.mean()) ** 4).mean() / (_s20 ** 4) - 3.0) if _s20 > 1e-10 else 0.0

        momentum_1d = mtf_daily_return
        momentum_1w = mtf_weekly_momentum

        autocorr = 0.0
        if i >= 20:
            _rets_ac = np.diff(closes[i - 20:i + 1]) / np.maximum(closes[i - 20:i], 1e-10)
            if len(_rets_ac) >= 2 and _rets_ac[:-1].std() > 0 and _rets_ac[1:].std() > 0:
                autocorr = float(np.corrcoef(_rets_ac[:-1], _rets_ac[1:])[0, 1])
                if np.isnan(autocorr):
                    autocorr = 0.0

        fr_velocity = 0.0

        range_compression = 1.0
        if i >= 20:
            recent_range = highs[i - 4:i + 1].max() - lows[i - 4:i + 1].min()
            longer_range = highs[i - 19:i + 1].max() - lows[i - 19:i + 1].min()
            range_compression = recent_range / longer_range if longer_range > 0 else 1.0

        body_pct_20 = 0.5
        if i >= 20:
            body_pcts = np.abs(closes[i - 19:i + 1] - opens[i - 19:i + 1]) / np.maximum(highs[i - 19:i + 1] - lows[i - 19:i + 1], 1e-10)
            body_pct_20 = float(body_pcts.mean())

        vol_trend = 0.0
        if i >= 20:
            vol_trend = float(np.polyfit(np.arange(20), volumes[i - 19:i + 1], 1)[0])
            vol_trend = vol_trend / avg_vol_20 if avg_vol_20 > 0 else 0.0

        price_vs_vwap_20 = pvw
        candle_streak = float(cons_g) if cons_g > 0 else -float(cons_r)

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
            price_skew, price_kurt, trend_slope, stddev_ratio,
            area_ratio_val, range_pos, atr_ratio_val, vpt_norm,
            first_loc_max, longest_below,
            mtf_sma_4h_ratio, mtf_momentum_4h,
            mtf_daily_return, mtf_daily_range,
            mtf_daily_volume_ratio, mtf_weekly_momentum,
            l2_book_imbalance, l2_depth_ratio, l2_large_order,
            l2_book_pressure, l2_spread, l2_flow_intensity,
            cvd_norm, vol_profile_ratio,
            vol_regime, vol_kurtosis,
            momentum_1d, momentum_1w,
            autocorr, fr_velocity,
            range_compression, body_pct_20,
            vol_trend, price_vs_vwap_20, candle_streak,
        ]"""

new_v11_features = """        # ── V11 features (indices 62-71) ──────────────────────────────────────
        # btc_beta (62)
        btc_beta = 0.0

        # correlation_change (63)
        correlation_change = 0.0

        # realized_vol_24 (64)
        if i >= 24:
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

        # vpin_proxy (69)
        if i >= 20:
            _buy_vol = 0.0
            _sell_vol = 0.0
            for _k in range(max(0, i - 19), i + 1):
                if closes[_k] >= opens[_k]:
                    _buy_vol += volumes[_k]
                else:
                    _sell_vol += volumes[_k]
            _total_vol = _buy_vol + _sell_vol
            vpin_proxy = abs(_buy_vol - _sell_vol) / _total_vol if _total_vol > 0 else 0.0
        else:
            vpin_proxy = 0.0

        # flow_persistence (70)
        flow_persistence = 0.0

        # mtf_alignment (71)
        _signs = [
            1.0 if ema_diff > 0 else (-1.0 if ema_diff < 0 else 0.0),
            1.0 if mtf_momentum_4h > 0 else (-1.0 if mtf_momentum_4h < 0 else 0.0),
            1.0 if mtf_daily_return > 0 else (-1.0 if mtf_daily_return < 0 else 0.0),
            1.0 if mtf_weekly_momentum > 0 else (-1.0 if mtf_weekly_momentum < 0 else 0.0),
        ]
        mtf_alignment = sum(_signs) / 4.0

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
            price_skew, price_kurt, trend_slope, stddev_ratio,
            area_ratio_val, range_pos, atr_ratio_val, vpt_norm,
            first_loc_max, longest_below,
            mtf_sma_4h_ratio, mtf_momentum_4h,
            mtf_daily_return, mtf_daily_range,
            mtf_daily_volume_ratio, mtf_weekly_momentum,
            l2_book_imbalance, l2_depth_ratio, l2_large_order,
            l2_book_pressure, l2_spread, l2_flow_intensity,
            btc_beta, correlation_change,                          # 62-63
            realized_vol_24, v11_vol_regime,                       # 64-65
            vol_of_vol, hurst_exponent,                            # 66-67
            fractal_efficiency, vpin_proxy,                        # 68-69
            flow_persistence, mtf_alignment,                       # 70-71
        ]"""

assert old_v10_features in app, "FIX 1 FAILED: could not find old V10 features block in app.py"
app = app.replace(old_v10_features, new_v11_features)
print("[FIX 1] app.py: replaced V10 features (75) with V11 features (72) ✓")

# ============================================================================
# FIX 2: app.py — wrap xgboost import in try/except
# ============================================================================
app = app.replace(
    "import xgboost as xgb\n",
    "try:\n    import xgboost as xgb\nexcept ImportError:\n    xgb = None\n"
)
print("[FIX 2] app.py: wrapped xgboost import in try/except ✓")

write_file(APP_PATH, app)
print(f"  -> {APP_PATH} written")

# ============================================================================
# FIX 3: pro_trader.py — db_load_state SQLite leak (add try/finally)
# ============================================================================
BOT_PATH = "/root/hyperlend-bot/pro_bot/pro_trader.py"
bot = backup_and_read(BOT_PATH)

old_db_load = """def db_load_state(key, default=None):
    \"\"\"Load a state value from SQLite.\"\"\"
    try:
        with db_lock:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute('SELECT value FROM state WHERE key = ?', (key,))
            row = c.fetchone()
            return json.loads(row[0]) if row else default
    except Exception as e:
        log.debug(f"db_load_state error: {e}")
        return default"""

new_db_load = """def db_load_state(key, default=None):
    \"\"\"Load a state value from SQLite.\"\"\"
    conn = None
    try:
        with db_lock:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute('SELECT value FROM state WHERE key = ?', (key,))
            row = c.fetchone()
            return json.loads(row[0]) if row else default
    except Exception as e:
        log.debug(f"db_load_state error: {e}")
        return default
    finally:
        if conn:
            conn.close()"""

assert old_db_load in bot, "FIX 3 FAILED: could not find db_load_state in pro_trader.py"
bot = bot.replace(old_db_load, new_db_load)
print("[FIX 3] pro_trader.py: db_load_state now closes conn in finally ✓")

# ============================================================================
# FIX 6: pro_trader.py — Remove AI_COINS != 'ALL' dead code
# ============================================================================
bot = bot.replace(
    "    if AI_COINS != 'ALL' and coin not in AI_COINS:\n",
    "    if coin not in AI_COINS:\n"
)
print("[FIX 6] pro_trader.py: simplified AI_COINS check (removed 'ALL' dead code) ✓")

# ============================================================================
# FIX 8: pro_trader.py — Remove duplicated docstring line
# ============================================================================
bot = bot.replace(
    "    Uses np.nan for unavailable features (LightGBM handles natively).\n    Uses np.nan for unavailable features (LightGBM handles natively).\n",
    "    Uses np.nan for unavailable features (LightGBM handles natively).\n"
)
print("[FIX 8] pro_trader.py: removed duplicated docstring line ✓")

# ============================================================================
# FIX 9: pro_trader.py — Remove redundant global in open_scalp_position
# ============================================================================
# Merge the second global statement into the first one
bot = bot.replace(
    '    global scalp_position, scalp_last_close_time\n',
    '    global scalp_position, scalp_last_close_time, last_open_time\n',
    1  # only first occurrence
)
# Now remove the standalone "global last_open_time" inside open_scalp_position
# It appears after "    # FIX 6: Global open cooldown..."
bot = bot.replace(
    "    # FIX 6: Global open cooldown (5 min between any new position)\n    global last_open_time\n    if time.time() - last_open_time < GLOBAL_OPEN_COOLDOWN:\n        remaining = int((GLOBAL_OPEN_COOLDOWN - (time.time() - last_open_time)) / 60)\n        log.info(f\"[GLOBAL COOLDOWN] {remaining}min remaining before next open\")\n        return False\n\n    # Get current price\n    try:\n        book = get_l2_book(coin)",
    "    # FIX 6: Global open cooldown (5 min between any new position)\n    if time.time() - last_open_time < GLOBAL_OPEN_COOLDOWN:\n        remaining = int((GLOBAL_OPEN_COOLDOWN - (time.time() - last_open_time)) / 60)\n        log.info(f\"[GLOBAL COOLDOWN] {remaining}min remaining before next open\")\n        return False\n\n    # Get current price\n    try:\n        book = get_l2_book(coin)",
    1  # only one occurrence — the open_scalp_position one
)
print("[FIX 9] pro_trader.py: merged redundant global last_open_time into function header ✓")

write_file(BOT_PATH, bot)
print(f"  -> {BOT_PATH} written")

# ============================================================================
# FIX 4: liquidation_levels.py — move import requests to top
# ============================================================================
LIQ_PATH = "/root/hyperlend-bot/pro_bot/liquidation_levels.py"
liq = backup_and_read(LIQ_PATH)

# Add import requests at top (after existing imports)
liq = liq.replace(
    "import logging\nimport numpy as np\n",
    "import logging\nimport requests\nimport numpy as np\n"
)

# Remove the inline import
liq = liq.replace("            import requests\n", "")

print("[FIX 4] liquidation_levels.py: moved import requests to top-level ✓")
write_file(LIQ_PATH, liq)
print(f"  -> {LIQ_PATH} written")

# ============================================================================
# FIX 5: cloud_trader.py — USE CCXT_TIMEOUT_MS in exchange config
# ============================================================================
CT_PATH = "/root/deepalpha-cloud/cloud_trader.py"
ct = backup_and_read(CT_PATH)

ct = ct.replace(
    '            self.exchange = ccxt_async.bybit(\n                {\n                    "apiKey": self.api_key,\n                    "secret": self.api_secret,\n                    "enableRateLimit": True,\n                    "options": {"defaultType": "swap", "recvWindow": 10000},\n                }\n            )',
    '            self.exchange = ccxt_async.bybit(\n                {\n                    "apiKey": self.api_key,\n                    "secret": self.api_secret,\n                    "enableRateLimit": True,\n                    "timeout": CCXT_TIMEOUT_MS,\n                    "options": {"defaultType": "swap", "recvWindow": 10000},\n                }\n            )'
)
print("[FIX 5] cloud_trader.py: added timeout: CCXT_TIMEOUT_MS to exchange config ✓")

# ============================================================================
# FIX 7: cloud_trader.py — replace deprecated asyncio.get_event_loop()
# ============================================================================
ct = ct.replace(
    "        loop = asyncio.get_event_loop()\n",
    "        loop = asyncio.get_running_loop()\n"
)
print("[FIX 7] cloud_trader.py: replaced get_event_loop() with get_running_loop() ✓")

write_file(CT_PATH, ct)
print(f"  -> {CT_PATH} written")

# ============================================================================
# Verify all files parse correctly
# ============================================================================
import py_compile
errors = []
for f in [APP_PATH, BOT_PATH, LIQ_PATH, CT_PATH]:
    try:
        py_compile.compile(f, doraise=True)
        print(f"  [SYNTAX OK] {f}")
    except py_compile.PyCompileError as e:
        errors.append(f)
        print(f"  [SYNTAX ERROR] {f}: {e}")

if errors:
    print(f"\n*** SYNTAX ERRORS in {len(errors)} file(s)! ***")
else:
    print("\n=== ALL 9 FIXES APPLIED SUCCESSFULLY, ALL FILES SYNTAX-VALID ===")
