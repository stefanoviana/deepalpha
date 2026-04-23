#!/usr/bin/env python3
"""Patch script for /root/hyperlend-bot/pro_bot/pro_trader.py - 10 fixes."""

FILE = "/root/hyperlend-bot/pro_bot/pro_trader.py"

with open(FILE, "r") as f:
    code = f.read()

original = code

changes = 0

# ============================================================================
# P1: 2x fetch_positions per open - combine into one fetch
# ============================================================================
old_count_fix = """    # COUNT-FIX: count REAL positions on exchange, not just internal dict
    # This prevents opening extra positions after a restart when dict is empty
    with lock:
        total_pos = len(positions) + (1 if scalp_position is not None else 0)
    if EXCHANGE in ("bybit", "bitget"):
        try:
            _bg_count = get_ccxt_exchange()
            _pos_count = _bg_count.fetch_positions(None, params={"category": "linear", "settleCoin": "USDT"} if EXCHANGE == "bybit" else {})
            _real_count = sum(1 for p in _pos_count if abs(float(p.get("contracts", 0) or 0)) > 0)
            total_pos = max(total_pos, _real_count)
        except Exception as _ce:
            log.warning(f"[COUNT-FIX] fetch_positions failed: {_ce}")
    with lock:
        if total_pos >= MAX_POSITIONS:
            log.info(f"Max positions ({total_pos}/{MAX_POSITIONS}) reached")
            return False
        if coin in positions:
            log.info(f"Already have position in {coin}")
            return False"""

new_count_fix = """    # COUNT-FIX + DUP-GUARD: single fetch_positions for both checks (P1)
    with lock:
        total_pos = len(positions) + (1 if scalp_position is not None else 0)
    _fetched_positions = None
    if EXCHANGE in ("bybit", "bitget"):
        try:
            _bg_count = get_ccxt_exchange()
            _fetched_positions = _bg_count.fetch_positions(None, params={"category": "linear", "settleCoin": "USDT"} if EXCHANGE == "bybit" else {})
            _real_count = sum(1 for p in _fetched_positions if abs(float(p.get("contracts", 0) or 0)) > 0)
            total_pos = max(total_pos, _real_count)
        except Exception as _ce:
            log.warning(f"[COUNT-FIX] fetch_positions failed: {_ce}")
    with lock:
        if total_pos >= MAX_POSITIONS:
            log.info(f"Max positions ({total_pos}/{MAX_POSITIONS}) reached")
            return False
        if coin in positions:
            log.info(f"Already have position in {coin}")
            return False"""

if old_count_fix in code:
    code = code.replace(old_count_fix, new_count_fix, 1)
    changes += 1
    print("P1a: COUNT-FIX patched")
else:
    print("P1a: SKIPPED - COUNT-FIX block not found")

# Now replace the DUP-GUARD block to use cached _fetched_positions
old_dup = """    # DUP-GUARD: check Bybit directly for existing position BEFORE placing order
    # This prevents the race condition where MAKER_SKIPPED returns None but
    # Bybit has actually filled the order, causing the bot to place duplicates.
    if EXCHANGE in ("bybit", "bitget"):
        try:
            _bg_check = get_ccxt_exchange()
            _sym_check = ccxt_symbol(coin)
            _pos_check = _bg_check.fetch_positions([_sym_check],
                params={"category": "linear", "settleCoin": "USDT"} if EXCHANGE == "bybit" else {})
            _has_live = any(abs(float(p.get("contracts", 0) or 0)) > 0 for p in _pos_check
                           if p.get("symbol") == _sym_check)
            if _has_live:
                log.info(f"[DUP-GUARD] {coin} already has LIVE position on exchange \xe2\x80\x94 skipping open")
                with lock:
                    closing_coins.discard(coin)
                return False
        except Exception as _dge:
            log.warning(f"[DUP-GUARD] check failed for {coin}: {_dge} \xe2\x80\x94 proceeding cautiously")"""

new_dup = """    # DUP-GUARD: reuse _fetched_positions from COUNT-FIX (P1 - no extra API call)
    if EXCHANGE in ("bybit", "bitget") and _fetched_positions is not None:
        try:
            _sym_check = ccxt_symbol(coin)
            _has_live = any(abs(float(p.get("contracts", 0) or 0)) > 0 for p in _fetched_positions
                           if p.get("symbol") == _sym_check)
            if _has_live:
                log.info(f"[DUP-GUARD] {coin} already has LIVE position on exchange \xe2\x80\x94 skipping open")
                with lock:
                    closing_coins.discard(coin)
                return False
        except Exception as _dge:
            log.warning(f"[DUP-GUARD] check failed for {coin}: {_dge} \xe2\x80\x94 proceeding cautiously")"""

if old_dup in code:
    code = code.replace(old_dup, new_dup, 1)
    changes += 1
    print("P1b: DUP-GUARD patched")
else:
    print("P1b: SKIPPED - DUP-GUARD block not found (trying without em dash)")
    # Try with regular dash
    old_dup2 = old_dup.replace("\xe2\x80\x94", "—")
    if old_dup2 in code:
        new_dup2 = new_dup.replace("\xe2\x80\x94", "—")
        code = code.replace(old_dup2, new_dup2, 1)
        changes += 1
        print("P1b: DUP-GUARD patched (em dash variant)")

# ============================================================================
# P2: Candle cache for check_correlation
# ============================================================================
if "ATR_CACHE_TTL = 300" in code and "_candle_cache" not in code:
    code = code.replace(
        "ATR_CACHE_TTL = 300",
        "ATR_CACHE_TTL = 300\n\n# P2: Candle cache to avoid redundant API calls\n_candle_cache = {}  # {(coin, interval, lookback): {\"data\": candles, \"time\": timestamp}}\nCANDLE_CACHE_TTL = 300  # 5 minutes",
        1
    )
    changes += 1
    print("P2a: Candle cache variable added")

# Wrap get_candles to check cache first
old_gc = 'def get_candles(coin, interval="1h", lookback=10):\n    """Get candle data."""'
new_gc = 'def get_candles(coin, interval="1h", lookback=10):\n    """Get candle data (with 5-min cache)."""\n    _cache_key = (coin, interval, lookback)\n    _now = time.time()\n    if _cache_key in _candle_cache and _now - _candle_cache[_cache_key]["time"] < CANDLE_CACHE_TTL:\n        return _candle_cache[_cache_key]["data"]'

if old_gc in code:
    code = code.replace(old_gc, new_gc, 1)
    changes += 1
    print("P2b: get_candles cache check added")

# Store in cache before returning candles (bitget/bybit path)
old_ret1 = '            return candles\n        except Exception as e:\n            log.error(f"get_candles [{_label}] error for {coin}: {e}")\n            return []'
new_ret1 = '            _candle_cache[_cache_key] = {"data": candles, "time": _now}\n            return candles\n        except Exception as e:\n            log.error(f"get_candles [{_label}] error for {coin}: {e}")\n            return []'

if old_ret1 in code:
    code = code.replace(old_ret1, new_ret1, 1)
    changes += 1
    print("P2c: Cache store (ccxt path) added")

# Store in cache for HL path
old_ret2 = '    return hl_post({\n        "type": "candleSnapshot",\n        "req": {"coin": coin, "interval": interval, "startTime": start_ms, "endTime": now_ms}\n    })'
new_ret2 = '    _result = hl_post({\n        "type": "candleSnapshot",\n        "req": {"coin": coin, "interval": interval, "startTime": start_ms, "endTime": now_ms}\n    })\n    if _result:\n        _candle_cache[_cache_key] = {"data": _result, "time": _now}\n    return _result'

if old_ret2 in code:
    code = code.replace(old_ret2, new_ret2, 1)
    changes += 1
    print("P2d: Cache store (HL path) added")

# ============================================================================
# P3: Cache for _get_hourly_win_rates, _get_coin_win_rate, _get_consecutive_losses
# ============================================================================
# Add cache variables before _get_coin_win_rate
old_p3_marker = "# --- PROFITABILITY HELPERS (Patch v2.0) ------------------------------------\ndef _get_coin_win_rate(coin):"
new_p3_marker = "# --- PROFITABILITY HELPERS (Patch v2.0) ------------------------------------\n# P3: Caches for DB-heavy functions (300s TTL)\n_hourly_wr_cache = {\"data\": None, \"time\": 0}\n_coin_wr_cache = {}  # {coin: {\"data\": wr, \"time\": ts}}\n_consec_loss_cache = {\"data\": 0, \"time\": 0}\n_PROF_CACHE_TTL = 300\n\n\ndef _get_coin_win_rate(coin):"

if old_p3_marker in code:
    code = code.replace(old_p3_marker, new_p3_marker, 1)
    changes += 1
    print("P3a: Cache variables added")

# Add cache logic to _get_coin_win_rate
old_cwr = '''def _get_coin_win_rate(coin):
    """Get win rate for a specific coin from recent trades (20+ required)."""
    try:
        recent = db_get_recent_trades(500)
        coin_trades = [t for t in recent if t.get("coin") == coin]
        if len(coin_trades) < 20:
            return None
        wins = sum(1 for t in coin_trades if t.get("pnl", 0) > 0)
        return wins / len(coin_trades)
    except Exception:
        return None'''

new_cwr = '''def _get_coin_win_rate(coin):
    """Get win rate for a specific coin from recent trades (20+ required). Cached 5min."""
    _now = time.time()
    if coin in _coin_wr_cache and _now - _coin_wr_cache[coin]["time"] < _PROF_CACHE_TTL:
        return _coin_wr_cache[coin]["data"]
    try:
        recent = db_get_recent_trades(500)
        coin_trades = [t for t in recent if t.get("coin") == coin]
        if len(coin_trades) < 20:
            _coin_wr_cache[coin] = {"data": None, "time": _now}
            return None
        wins = sum(1 for t in coin_trades if t.get("pnl", 0) > 0)
        result = wins / len(coin_trades)
        _coin_wr_cache[coin] = {"data": result, "time": _now}
        return result
    except Exception:
        return None'''

if old_cwr in code:
    code = code.replace(old_cwr, new_cwr, 1)
    changes += 1
    print("P3b: _get_coin_win_rate cached")

# Add cache logic to _get_consecutive_losses
old_cl = '''def _get_consecutive_losses():
    """Count consecutive losses from the most recent trades."""
    try:
        recent = db_get_recent_trades(20)
        if not recent:
            return 0
        count = 0
        for t in reversed(recent):
            if t.get("pnl", 0) < 0:
                count += 1
            else:
                break
        return count
    except Exception:
        return 0'''

new_cl = '''def _get_consecutive_losses():
    """Count consecutive losses from the most recent trades. Cached 5min."""
    _now = time.time()
    if _now - _consec_loss_cache["time"] < _PROF_CACHE_TTL:
        return _consec_loss_cache["data"]
    try:
        recent = db_get_recent_trades(20)
        if not recent:
            _consec_loss_cache["data"] = 0
            _consec_loss_cache["time"] = _now
            return 0
        count = 0
        for t in reversed(recent):
            if t.get("pnl", 0) < 0:
                count += 1
            else:
                break
        _consec_loss_cache["data"] = count
        _consec_loss_cache["time"] = _now
        return count
    except Exception:
        return 0'''

if old_cl in code:
    code = code.replace(old_cl, new_cl, 1)
    changes += 1
    print("P3c: _get_consecutive_losses cached")

# Add cache to _get_hourly_win_rates
old_hwr_start = '''def _get_hourly_win_rates():
    """Analyze win rates by hour of day (UTC). Returns dict hour -> (wr, count)."""
    try:
        recent = db_get_recent_trades(500)'''

new_hwr_start = '''def _get_hourly_win_rates():
    """Analyze win rates by hour of day (UTC). Returns dict hour -> (wr, count). Cached 5min."""
    _now = time.time()
    if _hourly_wr_cache["data"] is not None and _now - _hourly_wr_cache["time"] < _PROF_CACHE_TTL:
        return _hourly_wr_cache["data"]
    try:
        recent = db_get_recent_trades(500)'''

if old_hwr_start in code:
    code = code.replace(old_hwr_start, new_hwr_start, 1)
    changes += 1
    print("P3d: _get_hourly_win_rates cache check added")

# Cache the result before returning
old_hwr_ret = """        result = {}
        for hour, data in hourly.items():
            if data["total"] >= 5:
                result[hour] = (data["wins"] / data["total"], data["total"])
        return result
    except Exception:
        return {}"""

new_hwr_ret = """        result = {}
        for hour, data in hourly.items():
            if data["total"] >= 5:
                result[hour] = (data["wins"] / data["total"], data["total"])
        _hourly_wr_cache["data"] = result
        _hourly_wr_cache["time"] = _now
        return result
    except Exception:
        return {}"""

if old_hwr_ret in code:
    code = code.replace(old_hwr_ret, new_hwr_ret, 1)
    changes += 1
    print("P3e: _get_hourly_win_rates cache store added")

# ============================================================================
# P6: load_markets every 30s - only load if not already loaded
# ============================================================================
old_lm = "            # Load markets for accurate szDecimals\n            bg_markets = bg.load_markets()"
new_lm = "            # Load markets for accurate szDecimals (reuse if already loaded - P6)\n            bg_markets = bg.markets if bg.markets else bg.load_markets()"

if old_lm in code:
    code = code.replace(old_lm, new_lm, 1)
    changes += 1
    print("P6: load_markets optimized")

# ============================================================================
# PR2: Time filter 00:00-06:00 -> 02:00-05:00
# ============================================================================
old_tf = "        if _utc_hour >= 0 and _utc_hour < 6:"
new_tf = "        if _utc_hour >= 2 and _utc_hour < 5:"

if old_tf in code:
    code = code.replace(old_tf, new_tf, 1)
    changes += 1
    print("PR2: Time filter narrowed to 02:00-05:00")

# ============================================================================
# PR5: Global cooldown 300 -> 90
# ============================================================================
old_gc5 = "GLOBAL_OPEN_COOLDOWN = 300  # 5 minutes between opening any new position (prevents rapid re-entry)"
new_gc5 = "GLOBAL_OPEN_COOLDOWN = 90  # 90 seconds between opening any new position (prevents rapid re-entry)"

if old_gc5 in code:
    code = code.replace(old_gc5, new_gc5, 1)
    changes += 1
    print("PR5: Global cooldown 300 -> 90")

# ============================================================================
# PR6: Market fallback - use IOC instead of another PostOnly limit
# ============================================================================
old_mf = '        log.info(f"[MARKET FALLBACK] {coin}: limit order failed, retrying as market")\n        result = place_order(coin, is_buy, size, price=None)'
new_mf = '        log.info(f"[MARKET FALLBACK] {coin}: limit order failed, retrying as IOC market")\n        result = place_order(coin, is_buy, size, price=None, force_taker=True)'

if old_mf in code:
    code = code.replace(old_mf, new_mf, 1)
    changes += 1
    print("PR6a: Market fallback uses force_taker=True")

# Add force_taker parameter to place_order
old_po_sig = 'def place_order(coin: str, is_buy: bool, size: float, price: float = None,\n                reduce_only: bool = False, order_type: str = "limit") -> dict:'
new_po_sig = 'def place_order(coin: str, is_buy: bool, size: float, price: float = None,\n                reduce_only: bool = False, order_type: str = "limit",\n                force_taker: bool = False) -> dict:'

if old_po_sig in code:
    code = code.replace(old_po_sig, new_po_sig, 1)
    changes += 1
    print("PR6b: place_order signature updated with force_taker")

# Add IOC handling in place_order opening section
old_open = '''                price = round_price(coin, price)
                log.info(f"[{_EX}] OPEN LIMIT {side_str.upper()} {size} {coin} @ {price}")
                order_params = {}
                if EXCHANGE == "bybit":
                    order_params["timeInForce"] = "PostOnly"'''

new_open = '''                price = round_price(coin, price)
                # PR6: force_taker uses IOC for market fallback
                if force_taker:
                    log.info(f"[{_EX}] OPEN IOC {side_str.upper()} {size} {coin} @ {price}")
                    ioc_params = {"timeInForce": "IOC"}
                    ioc_result = bg.create_order(sym, "limit", side_str, size, price, params=ioc_params)
                    _ioc_status = ioc_result.get("status", "")
                    if _ioc_status in ("closed",):
                        avg_price = float(ioc_result.get("average", price) or price)
                        filled = float(ioc_result.get("filled", size) or size)
                        log.info(f"[{_EX}] IOC FILLED: {filled} @ {avg_price}")
                        return {"status": "filled", "price": avg_price, "size": filled, "oid": ioc_result.get("id")}
                    else:
                        log.info(f"[{_EX}] IOC NOT FILLED for {coin}")
                        return None
                log.info(f"[{_EX}] OPEN LIMIT {side_str.upper()} {size} {coin} @ {price}")
                order_params = {}
                if EXCHANGE == "bybit":
                    order_params["timeInForce"] = "PostOnly"'''

# The f-string with {_EX} needs to match literally
old_open_search = '                price = round_price(coin, price)\n                log.info(f"[{_EX}] OPEN LIMIT {side_str.upper()} {size} {coin} @ {price}")\n                order_params = {}\n                if EXCHANGE == "bybit":\n                    order_params["timeInForce"] = "PostOnly"'

if old_open_search in code:
    code = code.replace(old_open_search, new_open, 1)
    changes += 1
    print("PR6c: IOC handling in place_order added")
else:
    print("PR6c: SKIPPED - open limit block not found")

# ============================================================================
# PR8: Dynamic trailing stop after tier2
# ============================================================================
old_tier2 = '''            # Tier 2: up > +3% -> lock +1.5% profit
            if pnl_pct >= TRAIL_TIER2_PCT and current_tier < 2:
                positions[coin]["trail_tier"] = 2
                if side == "LONG":
                    locked_sl = entry_price * (1 + TRAIL_TIER2_LOCK_PCT)
                else:
                    locked_sl = entry_price * (1 - TRAIL_TIER2_LOCK_PCT)
                positions[coin]["sl_price"] = locked_sl
                positions[coin]["trail_active"] = True
                log.info(f"TRAIL TIER 2 {coin}: up {pnl_pct*100:.1f}%, SL locked +1.5% = {locked_sl:.4f}")
                # tg(f"TRAIL TIER 2 {coin}: up {pnl_pct*100:.1f}%, SL locked +1.5%")'''

new_tier2 = '''            # Tier 2: up > +3% -> lock +1.5% profit + dynamic ATR trailing (PR8)
            if pnl_pct >= TRAIL_TIER2_PCT and current_tier < 2:
                positions[coin]["trail_tier"] = 2
                if side == "LONG":
                    locked_sl = entry_price * (1 + TRAIL_TIER2_LOCK_PCT)
                else:
                    locked_sl = entry_price * (1 - TRAIL_TIER2_LOCK_PCT)
                positions[coin]["sl_price"] = locked_sl
                positions[coin]["trail_active"] = True
                positions[coin]["high_water_mark"] = current_price
                log.info(f"TRAIL TIER 2 {coin}: up {pnl_pct*100:.1f}%, SL locked +1.5% = {locked_sl:.4f}")'''

if old_tier2 in code:
    code = code.replace(old_tier2, new_tier2, 1)
    changes += 1
    print("PR8a: Tier2 high_water_mark init added")

# Replace ATR trail with high-water-mark version
old_atr = '''            # ATR-based trailing stop (replaces fixed trailing)
            if positions[coin].get("trail_active") or pnl_pct >= 0.005:  # activate at +0.5%
                try:
                    _atr = get_atr(coin)
                    if _atr and _atr > 0:
                        # Trail distance = 1.5x ATR (dynamic)
                        trail_dist = _atr * 1.5
                    else:
                        trail_dist = current_price * TRAIL_DISTANCE_PCT
                except Exception:
                    trail_dist = current_price * TRAIL_DISTANCE_PCT

                positions[coin]["trail_active"] = True

                if side == "LONG":
                    new_trail = current_price - trail_dist
                    if new_trail > positions[coin]["sl_price"]:
                        positions[coin]["sl_price"] = new_trail
                        log.info(f"[ATR TRAIL] {coin} LONG: SL moved to {new_trail:.4f} (ATR trail)")
                else:
                    new_trail = current_price + trail_dist
                    if new_trail < positions[coin]["sl_price"]:
                        positions[coin]["sl_price"] = new_trail
                        log.info(f"[ATR TRAIL] {coin} SHORT: SL moved to {new_trail:.4f} (ATR trail)")'''

new_atr = '''            # ATR-based trailing stop (PR8: dynamic HWM after tier2)
            if positions[coin].get("trail_active") or pnl_pct >= 0.005:  # activate at +0.5%
                try:
                    _atr = get_atr(coin)
                    if _atr and _atr > 0:
                        trail_dist = _atr * 1.5
                    else:
                        trail_dist = current_price * TRAIL_DISTANCE_PCT
                except Exception:
                    trail_dist = current_price * TRAIL_DISTANCE_PCT

                positions[coin]["trail_active"] = True

                # PR8: Dynamic trailing with high-water mark after tier 2
                if positions[coin].get("trail_tier", 0) >= 2:
                    hwm = positions[coin].get("high_water_mark", current_price)
                    if side == "LONG":
                        hwm = max(hwm, current_price)
                        trail_sl = hwm - trail_dist
                    else:
                        hwm = min(hwm, current_price)
                        trail_sl = hwm + trail_dist
                    positions[coin]["high_water_mark"] = hwm
                    if side == "LONG" and trail_sl > positions[coin]["sl_price"]:
                        positions[coin]["sl_price"] = trail_sl
                        log.info(f"[ATR TRAIL T2] {coin} LONG: HWM={hwm:.4f} SL={trail_sl:.4f}")
                    elif side == "SHORT" and trail_sl < positions[coin]["sl_price"]:
                        positions[coin]["sl_price"] = trail_sl
                        log.info(f"[ATR TRAIL T2] {coin} SHORT: HWM={hwm:.4f} SL={trail_sl:.4f}")
                else:
                    if side == "LONG":
                        new_trail = current_price - trail_dist
                        if new_trail > positions[coin]["sl_price"]:
                            positions[coin]["sl_price"] = new_trail
                            log.info(f"[ATR TRAIL] {coin} LONG: SL moved to {new_trail:.4f} (ATR trail)")
                    else:
                        new_trail = current_price + trail_dist
                        if new_trail < positions[coin]["sl_price"]:
                            positions[coin]["sl_price"] = new_trail
                            log.info(f"[ATR TRAIL] {coin} SHORT: SL moved to {new_trail:.4f} (ATR trail)")'''

if old_atr in code:
    code = code.replace(old_atr, new_atr, 1)
    changes += 1
    print("PR8b: ATR trail with HWM logic added")

# ============================================================================
# Final write
# ============================================================================
if code != original:
    with open(FILE, "w") as f:
        f.write(code)
    print(f"\nSUCCESS: {changes} changes applied to pro_trader.py ({len(original)} -> {len(code)} bytes)")
else:
    print("\nWARNING: No changes were made!")
