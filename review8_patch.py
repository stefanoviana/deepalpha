#!/usr/bin/env python3
"""
Review #8 — Patch script for remaining issues.

Issue 6a: pro_trader.py line ~4620 — isinstance(AI_COINS, list) dead code
Issue 6b: pro_trader.py line ~7085 — isinstance(AI_COINS, list) dead code
"""

import re
import sys

FIXES = []

# ─── Issue 6a: Simplify isinstance check at sync_positions ───
def fix_6a(path="/root/hyperlend-bot/pro_bot/pro_trader.py"):
    with open(path, "r") as f:
        src = f.read()
    old = "                if isinstance(AI_COINS, list) and coin not in AI_COINS:"
    new = "                if coin not in AI_COINS:"
    if old not in src:
        print(f"[6a] SKIP — pattern not found in {path}")
        return False
    src = src.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(src)
    print(f"[6a] FIXED — simplified isinstance check at sync_positions in {path}")
    FIXES.append("6a")
    return True

# ─── Issue 6b: Remove isinstance branch at main loop ───
def fix_6b(path="/root/hyperlend-bot/pro_bot/pro_trader.py"):
    with open(path, "r") as f:
        src = f.read()
    old = """\
                if isinstance(AI_COINS, list):
                    ai_coin_set = set(AI_COINS)
                else:
                    _all_coins = [c for c in meta.get("universe", []) if c.get("name") not in COIN_BLACKLIST]
                    _all_coins.sort(key=lambda x: _vol_map.get(x.get("name"), 0), reverse=True)
                    ai_coin_set = set(c["name"] for c in _all_coins[:50])"""
    new = "                ai_coin_set = set(AI_COINS)"
    if old not in src:
        print(f"[6b] SKIP — pattern not found in {path}")
        return False
    src = src.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(src)
    print(f"[6b] FIXED — removed isinstance branch, simplified to set(AI_COINS)")
    FIXES.append("6b")
    return True


if __name__ == "__main__":
    fix_6a()
    fix_6b()

    print(f"\n{'='*50}")
    print(f"Applied {len(FIXES)} fixes: {', '.join(FIXES) if FIXES else 'none'}")
    if not FIXES:
        print("All issues were already fixed!")
    print(f"{'='*50}")
