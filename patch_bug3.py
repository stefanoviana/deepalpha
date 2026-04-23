#!/usr/bin/env python3
"""BUG 3: Fix R:R ratio for low confidence trades in pro_trader.py."""

FILE = "/root/hyperlend-bot/pro_bot/pro_trader.py"

with open(FILE, "r") as f:
    content = f.read()

# Fix SL/TP multipliers for low confidence
OLD_SL_TP = """            else:
                _sl_mult = 2.2   # wider SL for low conf
                _tp_mult = 1.5   # very tight TP, grab and go"""

NEW_SL_TP = """            else:
                _sl_mult = 1.5   # TIGHTER SL - cut losses fast
                _tp_mult = 2.0   # wider TP - let it run more (R:R > 1:1)"""

# Fix T1/T2 for low confidence
OLD_T1_T2 = """            else:
                _t1_mult, _t2_mult = 0.6, 1.0   # grab and go"""

NEW_T1_T2 = """            else:
                _t1_mult, _t2_mult = 0.8, 1.3   # moderate targets, still > 1:1 R:R"""

count = 0
if OLD_SL_TP in content:
    content = content.replace(OLD_SL_TP, NEW_SL_TP, 1)
    count += 1
else:
    print("ERROR: Could not find SL/TP block for low confidence")

if OLD_T1_T2 in content:
    content = content.replace(OLD_T1_T2, NEW_T1_T2, 1)
    count += 1
else:
    print("ERROR: Could not find T1/T2 block for low confidence")

if count == 2:
    with open(FILE, "w") as f:
        f.write(content)
    print("BUG 3 FIXED: R:R ratio for low confidence - SL=1.5, TP=2.0, T1=0.8, T2=1.3")
elif count > 0:
    with open(FILE, "w") as f:
        f.write(content)
    print(f"BUG 3 PARTIALLY FIXED: {count}/2 replacements done")
