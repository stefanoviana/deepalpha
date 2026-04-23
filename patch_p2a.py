#!/usr/bin/env python3
"""Add missing _candle_cache declaration to pro_trader.py"""

FILE = "/root/hyperlend-bot/pro_bot/pro_trader.py"

with open(FILE, "r") as f:
    code = f.read()

old = "_atr_cache = {}  # coin -> (timestamp, value)\nATR_CACHE_TTL = 300  # 5 minutes"
new = "_atr_cache = {}  # coin -> (timestamp, value)\nATR_CACHE_TTL = 300  # 5 minutes\n\n# P2: Candle cache to avoid redundant API calls\n_candle_cache = {}  # {(coin, interval, lookback): {\"data\": candles, \"time\": timestamp}}\nCANDLE_CACHE_TTL = 300  # 5 minutes"

if old in code and "\n_candle_cache = {}" not in code:
    code = code.replace(old, new, 1)
    with open(FILE, "w") as f:
        f.write(code)
    print("P2a: _candle_cache and CANDLE_CACHE_TTL declarations added")
else:
    print("P2a: Already exists or not needed")
