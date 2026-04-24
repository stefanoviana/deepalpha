#!/usr/bin/env python3
"""
Patch script: Apply all 6 fixes to pro_trader.py and cloud_trader.py
Run on VPS: python3 /tmp/patch_all_fixes.py
"""
import re
import shutil
import os
from datetime import datetime

# Backup first
ts = datetime.now().strftime("%Y%m%d_%H%M%S")

PRO = "/root/hyperlend-bot/pro_bot/pro_trader.py"
CLOUD = "/root/deepalpha-cloud/cloud_trader.py"

for f in [PRO, CLOUD]:
    if os.path.exists(f):
        shutil.copy2(f, f + f".bak_{ts}")
        print(f"Backed up {f}")

# ============================================================
# FIX 1: Compute 6 V11 features live in pro_trader.py
# ============================================================
with open(PRO, "r") as f:
    code = f.read()

# Replace btc_beta = 0.0
old_btc_beta = """        # btc_beta (62) — needs BTC cross data, default for now
        btc_beta = 0.0"""
new_btc_beta = """        # btc_beta (62) — regression beta of coin returns vs BTC returns
        btc_beta = 0.0
        try:
            if i >= 24:
                _btc_arr = np.array(get_btc_closes_cached(), dtype=float)
                if len(_btc_arr) >= 25:
                    _coin_rets_b = np.diff(closes[i-24:i+1]) / np.maximum(closes[i-24:i], 1e-10)
                    _btc_rets_b = np.diff(_btc_arr[-25:]) / np.maximum(_btc_arr[-25:-1], 1e-10)
                    _btc_var = np.var(_btc_rets_b)
                    if _btc_var > 1e-12:
                        btc_beta = float(np.cov(_coin_rets_b, _btc_rets_b)[0, 1] / _btc_var)
                        btc_beta = max(-3.0, min(3.0, btc_beta))
        except Exception:
            btc_beta = 0.0"""

assert old_btc_beta in code, "FIX1: btc_beta block not found"
code = code.replace(old_btc_beta, new_btc_beta, 1)

# Replace correlation_change = 0.0
old_corr = """        # correlation_change (63) — needs BTC history, default
        correlation_change = 0.0"""
new_corr = """        # correlation_change (63) — short-term minus long-term BTC correlation
        correlation_change = 0.0
        try:
            if i >= 49:
                _btc_arr2 = np.array(get_btc_closes_cached(), dtype=float)
                if len(_btc_arr2) >= 50:
                    # short window (10 periods)
                    _cr_s = np.diff(closes[i-9:i+1]) / np.maximum(closes[i-9:i], 1e-10)
                    _br_s = np.diff(_btc_arr2[-10:]) / np.maximum(_btc_arr2[-10:-1], 1e-10)
                    _corr_short = float(np.corrcoef(_cr_s, _br_s)[0,1]) if _cr_s.std() > 0 and _br_s.std() > 0 else 0.0
                    # long window (50 periods)
                    _cr_l = np.diff(closes[i-49:i+1]) / np.maximum(closes[i-49:i], 1e-10)
                    _br_l = np.diff(_btc_arr2[-50:]) / np.maximum(_btc_arr2[-50:-1], 1e-10)
                    _corr_long = float(np.corrcoef(_cr_l, _br_l)[0,1]) if _cr_l.std() > 0 and _br_l.std() > 0 else 0.0
                    if not (np.isnan(_corr_short) or np.isnan(_corr_long)):
                        correlation_change = max(-2.0, min(2.0, _corr_short - _corr_long))
        except Exception:
            correlation_change = 0.0"""

assert old_corr in code, "FIX1: correlation_change block not found"
code = code.replace(old_corr, new_corr, 1)

# Replace v11_vol_regime = 1.0
old_regime = """        # vol_regime (65) — 1.0 = NORMAL
        v11_vol_regime = 1.0"""
new_regime = """        # vol_regime (65) — mapped from current_regime
        if current_regime in ("BEAR", "RANGE"):
            v11_vol_regime = 2.0
        elif current_regime == "BULL":
            v11_vol_regime = 0.0
        else:
            v11_vol_regime = 1.0"""

assert old_regime in code, "FIX1: v11_vol_regime block not found"
code = code.replace(old_regime, new_regime, 1)

# Replace vol_of_vol = 0.0
old_vov = """        # vol_of_vol (66)
        vol_of_vol = 0.0"""
new_vov = """        # vol_of_vol (66) — std of rolling volatilities
        vol_of_vol = 0.0
        try:
            if i >= 14:
                _rolling_vols = []
                for _rv_j in range(max(7, i-12), i+1):
                    _rv_w = closes[max(0, _rv_j-6):_rv_j+1]
                    if len(_rv_w) > 1:
                        _rv_rets = np.diff(_rv_w) / np.maximum(_rv_w[:-1], 1e-10)
                        _rolling_vols.append(float(np.std(_rv_rets)))
                if len(_rolling_vols) > 2:
                    vol_of_vol = float(np.std(_rolling_vols))
        except Exception:
            vol_of_vol = 0.0"""

assert old_vov in code, "FIX1: vol_of_vol block not found"
code = code.replace(old_vov, new_vov, 1)

# Replace hurst_exponent = 0.5
old_hurst = """        # hurst_exponent (67)
        hurst_exponent = 0.5"""
new_hurst = """        # hurst_exponent (67) — simplified R/S estimate
        hurst_exponent = 0.5
        try:
            if i >= 24:
                _h_series = closes[i-23:i+1].copy()
                _h_mean = _h_series.mean()
                _h_devs = np.cumsum(_h_series - _h_mean)
                _h_R = _h_devs.max() - _h_devs.min()
                _h_S = _h_series.std()
                if _h_S > 1e-10 and _h_R > 0:
                    hurst_exponent = float(np.log(max(_h_R / _h_S, 1e-10)) / np.log(len(_h_series)))
                    hurst_exponent = max(0.0, min(1.0, hurst_exponent))
        except Exception:
            hurst_exponent = 0.5"""

assert old_hurst in code, "FIX1: hurst_exponent block not found"
code = code.replace(old_hurst, new_hurst, 1)

# Replace flow_persistence = 0.0
old_flow = """        # flow_persistence (70)
        flow_persistence = 0.0"""
new_flow = """        # flow_persistence (70) — autocorrelation of signed volume
        flow_persistence = 0.0
        try:
            if i >= 20:
                _sv = np.zeros(20)
                for _fp_j in range(1, 21):
                    _fp_idx = i - 20 + _fp_j
                    if closes[_fp_idx] > closes[_fp_idx - 1]:
                        _sv[_fp_j - 1] = volumes[_fp_idx]
                    elif closes[_fp_idx] < closes[_fp_idx - 1]:
                        _sv[_fp_j - 1] = -volumes[_fp_idx]
                _sv_mean = _sv.mean()
                _sv_c = _sv - _sv_mean
                _sv_var = (_sv_c ** 2).mean()
                if _sv_var > 1e-10:
                    _sv_acov = (_sv_c[:-1] * _sv_c[1:]).mean()
                    flow_persistence = float(max(-1.0, min(1.0, _sv_acov / _sv_var)))
        except Exception:
            flow_persistence = 0.0"""

assert old_flow in code, "FIX1: flow_persistence block not found"
code = code.replace(old_flow, new_flow, 1)

# ============================================================
# FIX 3: ATR-adaptive trailing stop
# ============================================================
old_trail = """                    if _atr and _atr > 0:
                        trail_dist = _atr * 1.5"""
new_trail = """                    if _atr and _atr > 0:
                        if current_regime in ("BEAR", "RANGE"):
                            _trail_mult = 2.0
                        elif current_regime == "BULL":
                            _trail_mult = 1.0
                        else:
                            _trail_mult = 1.5
                        trail_dist = _atr * _trail_mult"""

assert old_trail in code, "FIX3: trail_dist block not found"
code = code.replace(old_trail, new_trail, 1)

# ============================================================
# FIX 4: Cost filter with historical win rate
# ============================================================
old_cost = """        expected_edge = (ai_conf - 50) / 100  # e.g. 70% conf = 0.20 edge"""
new_cost = """        # Use historical win rate if available for better edge estimate
        _hist_wr = _get_coin_win_rate(coin)
        if _hist_wr is not None:
            _avg_win = 0.015   # 1.5% average win
            _avg_loss = 0.012  # 1.2% average loss
            expected_edge = _hist_wr * _avg_win - (1 - _hist_wr) * _avg_loss
        else:
            expected_edge = (ai_conf - 50) / 100  # fallback"""

assert old_cost in code, "FIX4: expected_edge block not found"
code = code.replace(old_cost, new_cost, 1)

# ============================================================
# BUG 4: CVD history - try/except around float(cum_cvd)
# ============================================================
old_cvd = """            hist.append((now, float(price), float(cum_cvd)))"""
new_cvd = """            try:
                hist.append((now, float(price), float(cum_cvd)))
            except (TypeError, ValueError):
                pass"""

assert old_cvd in code, "BUG4: CVD hist.append not found"
# Only replace the first occurrence (in the sampling function, not the divergence check)
code = code.replace(old_cvd, new_cvd, 1)

with open(PRO, "w") as f:
    f.write(code)
print(f"Patched {PRO} (FIX 1, 3, 4 + BUG 4)")

# ============================================================
# FIX 2: Funding rate in cloud_trader.py
# ============================================================
with open(CLOUD, "r") as f:
    cloud = f.read()

old_fr = """            # ── Funding rate (0 — Bybit funding not bulk-fetched in cloud) ──
            fr = 0.0
            fr_delta_1h = 0.0
            fr_delta_4h = 0.0
            fr_delta_8h = 0.0"""
new_fr = """            # ── Funding rate (live fetch from Bybit) ──
            fr = 0.0
            fr_delta_1h = 0.0
            fr_delta_4h = 0.0
            fr_delta_8h = 0.0
            try:
                _fr_data = await self.exchange.fetch_funding_rate(symbol)
                fr = float(_fr_data.get('fundingRate', 0) or 0)
            except Exception:
                fr = 0.0"""

assert old_fr in cloud, "FIX2: funding rate block not found in cloud_trader"
cloud = cloud.replace(old_fr, new_fr, 1)

# ============================================================
# BUG 7: Cloud partial close rounding
# ============================================================
old_partial = """        close_size = pos["size"] * fraction
        if close_size <= 0:"""
new_partial = """        close_size = round(pos["size"] * fraction, 4)
        if close_size <= 0:"""

assert old_partial in cloud, "BUG7: partial close block not found in cloud_trader"
cloud = cloud.replace(old_partial, new_partial, 1)

with open(CLOUD, "w") as f:
    f.write(cloud)
print(f"Patched {CLOUD} (FIX 2 + BUG 7)")

print("\n✅ All 6 patches applied successfully!")
print("Restart bot: pm2 restart protrader && pm2 restart cloud-trader")
